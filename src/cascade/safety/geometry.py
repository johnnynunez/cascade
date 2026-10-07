"""Segment-to-segment distance: the geometry behind the inter-arm check.

Sampling a robot as a cloud of joint ORIGINS and measuring point-to-point
distance is cheap and wrong in a specific, dangerous way: it sees only the
joints, never the links between them. Two arms can cross with every joint far
from every other joint -- a forearm sweeping through the gap between a
neighbour's elbow and wrist registers as "clear" while the links physically
intersect. The point check compensates with a large margin, which trades one
error (missed collisions) for another (refusing legitimate poses).

Treating each link as a SEGMENT and measuring segment-to-segment distance
removes the blind spot, so the margin can describe real link thickness instead
of hiding a sampling artefact.

The implementation is the standard clamped-parameter solution (Ericson,
*Real-Time Collision Detection*, §5.1.9), vectorised over all pairs. Everything
is closed form -- no iteration, no branching per pair -- because this runs
inside `SafetyHarness.approve()` on every 50 Hz waypoint, where a slow check
is itself a hazard.
"""

from __future__ import annotations

import numpy as np

#: Below this, a segment is treated as a point and the parallel/degenerate
#: branch is taken. Squared-length units (m^2), so 1e-12 is a 1 micron link.
_EPS = 1e-12


def segment_segment_distances(p0, p1, q0, q1) -> np.ndarray:
    """Pairwise shortest distance between two sets of 3D segments.

    `p0`/`p1` are (N, 3) endpoints of N segments, `q0`/`q1` are (M, 3)
    endpoints of M segments. Returns an (N, M) matrix of shortest distances.

    Degenerate input is handled rather than rejected: a zero-length segment
    (two coincident joint origins, which happens at a fully folded joint)
    collapses to the correct point-to-segment distance, and exactly parallel
    segments take the branch that clamps to an endpoint instead of dividing by
    a vanishing determinant.
    """
    p0 = np.asarray(p0, dtype=float).reshape(-1, 3)
    p1 = np.asarray(p1, dtype=float).reshape(-1, 3)
    q0 = np.asarray(q0, dtype=float).reshape(-1, 3)
    q1 = np.asarray(q1, dtype=float).reshape(-1, 3)

    u = p1 - p0                              # (N, 3) direction of each P
    v = q1 - q0                              # (M, 3) direction of each Q
    w = p0[:, None, :] - q0[None, :, :]      # (N, M, 3)

    a = np.einsum("ik,ik->i", u, u)[:, None]        # (N, 1)
    b = u @ v.T                                     # (N, M)
    c = np.einsum("jk,jk->j", v, v)[None, :]        # (1, M)
    d = np.einsum("ik,ijk->ij", u, w)               # (N, M)
    e = np.einsum("jk,ijk->ij", v, w)               # (N, M)

    a, c = np.broadcast_to(a, b.shape).copy(), np.broadcast_to(c, b.shape).copy()
    D = a * c - b * b

    # Non-degenerate solution, then clamp both parameters to [0, 1]. The
    # numerator/denominator pairs are kept unreduced so a clamp can change the
    # denominator without a division having already happened.
    parallel = D < _EPS
    sN = np.where(parallel, 0.0, b * e - c * d)
    sD = np.where(parallel, 1.0, D)
    tN = np.where(parallel, e, a * e - b * d)
    tD = np.where(parallel, c, D)

    # s < 0 -> clamp to the P0 endpoint; s > 1 -> clamp to the P1 endpoint.
    low = sN < 0.0
    sN = np.where(low, 0.0, sN)
    tN = np.where(low, e, tN)
    tD = np.where(low, c, tD)

    high = sN > sD
    sN = np.where(high, sD, sN)
    tN = np.where(high, e + b, tN)
    tD = np.where(high, c, tD)

    # Now the same for t, re-solving s against the clamped t.
    tlow = tN < 0.0
    tN = np.where(tlow, 0.0, tN)
    sN = np.where(tlow & (-d < 0.0), 0.0, np.where(tlow & (-d > a), sD, np.where(tlow, -d, sN)))
    sD = np.where(tlow & (-d >= 0.0) & (-d <= a), a, sD)

    thigh = tN > tD
    tN = np.where(thigh, tD, tN)
    bd = -d + b
    sN = np.where(thigh & (bd < 0.0), 0.0, np.where(thigh & (bd > a), sD, np.where(thigh, bd, sN)))
    sD = np.where(thigh & (bd >= 0.0) & (bd <= a), a, sD)

    # A zero denominator only survives when the corresponding segment has zero
    # length, in which case the parameter is irrelevant and 0 is correct.
    sc = np.where(np.abs(sD) < _EPS, 0.0, sN / np.where(np.abs(sD) < _EPS, 1.0, sD))
    tc = np.where(np.abs(tD) < _EPS, 0.0, tN / np.where(np.abs(tD) < _EPS, 1.0, tD))

    closest = w + sc[..., None] * u[:, None, :] - tc[..., None] * v[None, :, :]
    return np.linalg.norm(closest, axis=2)


def _segment_radii(radii, n_segments: int, which: str) -> np.ndarray | None:
    """Per-segment radii as a (n_segments,) array, or None for zero-radius links.

    Refuses a list that does not describe exactly this chain: silently padding
    or truncating would measure a different arm than the one declared, and
    the harness turns the refusal into a blocked motion (fail closed).
    """
    if radii is None:
        return None
    r = np.asarray(radii, dtype=float).reshape(-1)
    if r.size == 0:
        return None
    if r.size != n_segments:
        raise ValueError(
            f"{which}: {r.size} link radii for a chain of {n_segments} "
            f"segment{'s' if n_segments != 1 else ''}"
        )
    if np.any(r < 0.0) or not np.all(np.isfinite(r)):
        raise ValueError(f"{which}: link radii must be finite and not negative, got {r.tolist()}")
    return r


def chain_distance(chain_a, chain_b, radii_a=None, radii_b=None) -> tuple[float, int, int]:
    """Shortest distance between two polyline chains of link positions.

    Each chain is (K, 3) points in KINEMATIC ORDER (base joint first, tool
    last), so consecutive points bound one physical link. Returns
    `(distance, i, j)` -- the distance and which link of each chain achieved
    it, for an error message that names the offending pair.

    `radii_a` / `radii_b` give each SEGMENT a thickness: the returned value is
    then the clearance between the capsule SURFACES (centreline distance
    minus both radii, negative when they overlap), and `(i, j)` names the
    closest surfaces, which need not be the closest centrelines. A link is
    not a zero-radius line: MEASURED on the Menagerie SO-101 collision
    geometry, the farthest collision surface sits up to 9 cm off the segment
    between two joint origins (tests/test_multi_arm_physics.py). None keeps
    the centreline behaviour.

    A chain of a single point is treated as that point (no segments), which
    keeps a 1-DOF or mocked arm from raising instead of measuring.
    """
    A = np.asarray(chain_a, dtype=float).reshape(-1, 3)
    B = np.asarray(chain_b, dtype=float).reshape(-1, 3)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("inf"), -1, -1
    # Degenerate chains: fall back to treating the lone point as a zero-length
    # segment so the same code path still returns a real distance.
    a0, a1 = (A[:-1], A[1:]) if A.shape[0] > 1 else (A, A)
    b0, b1 = (B[:-1], B[1:]) if B.shape[0] > 1 else (B, B)
    ra = _segment_radii(radii_a, a0.shape[0], "radii_a")
    rb = _segment_radii(radii_b, b0.shape[0], "radii_b")
    d = segment_segment_distances(a0, a1, b0, b1)
    if ra is not None:
        d = d - ra[:, None]
    if rb is not None:
        d = d - rb[None, :]
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return float(d[i, j]), int(i), int(j)
