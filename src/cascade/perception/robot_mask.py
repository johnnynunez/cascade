"""Robot-body masking for the occupancy map.

A depth camera sees the ARM. Integrated as-is, the robot's own links become
"obstacles" sitting 0 mm from the robot's own collision proxies, and the
clearance gate rejects every motion (measured: at the SO-101's home pose,
three of five link points read clearance 0.000-0.007 m against a 0.03 m
minimum -- the first grasp after the map went live was refused). Every
production ESDF stack masks the robot out before integration (nvblox /
Isaac ROS: URDF-driven body masking; cuRobo: robot-sphere carving). This is
cascade's version, built from what the stack already has: the arm's
kinematic link positions and a per-link radius.

`robot_mask(depth, K, T_base_cam, link_points, radii)` returns a boolean
(H,W) mask that is True on pixels whose 3-D point lies within `radius` of
any link SEGMENT (consecutive joint frames, plus the TCP). Those pixels are
zeroed before the frame ships to the bridge, so the ray-casting backends
treat them as "no measurement" -- not free, not occupied. Under-masking
leaves phantom obstacles; over-masking hides real ones next to the arm, so
radii are per-arm profile data (`arm.body_mask_radius_m`, default 0.06 m)
and the held object is NOT masked (it is a real obstacle for other links).
"""

from __future__ import annotations

import numpy as np


def _segment_distances(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point (N,3) to the segment a-b."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - proj, axis=1)


def robot_mask(depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
               link_points: np.ndarray, radius_m: float | np.ndarray = 0.06,
               stride: int = 1) -> np.ndarray:
    """Boolean (H,W) mask of depth pixels that belong to the robot body.

    link_points: (L,3) base-frame points along the arm (joint frames in
    order, TCP last). Segments are consecutive pairs. radius_m: scalar or
    (L-1,) per-segment radius.
    """
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape
    mask = np.zeros((H, W), dtype=bool)
    valid = depth > 0
    if not valid.any() or link_points is None or len(link_points) < 1:
        return mask
    ys, xs = np.nonzero(valid)
    zs = depth[ys, xs]
    K = np.asarray(K, dtype=np.float64)
    pts_cam = np.stack([(xs - K[0, 2]) / K[0, 0] * zs, (ys - K[1, 2]) / K[1, 1] * zs, zs], -1)
    T = np.asarray(T_base_cam, dtype=np.float64)
    pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
    lp = np.asarray(link_points, dtype=np.float64).reshape(-1, 3)
    radii = np.broadcast_to(np.asarray(radius_m, dtype=np.float64), (max(len(lp) - 1, 1),))
    hit = np.zeros(len(pts_base), dtype=bool)
    if len(lp) == 1:
        hit |= np.linalg.norm(pts_base - lp[0], axis=1) <= radii[0]
    for i in range(len(lp) - 1):
        # cheap AABB reject before the exact segment distance
        lo = np.minimum(lp[i], lp[i + 1]) - radii[i]
        hi = np.maximum(lp[i], lp[i + 1]) + radii[i]
        cand = np.all((pts_base >= lo) & (pts_base <= hi), axis=1)
        if not cand.any():
            continue
        d = _segment_distances(pts_base[cand], lp[i], lp[i + 1])
        idx = np.nonzero(cand)[0][d <= radii[i]]
        hit[idx] = True
    mask[ys[hit], xs[hit]] = True
    return mask


def arm_link_points(kin, q: np.ndarray) -> np.ndarray:
    """Joint-frame positions + TCP for an arm at q -- the polyline the mask
    thickens. Base frame."""
    links = np.asarray(kin.link_positions(np.asarray(q, dtype=float)))
    tcp = kin.fk(np.asarray(q, dtype=float))[:3, 3]
    return np.vstack([links, tcp[None, :]])
