"""Segment-to-segment distance, and why the inter-arm check needs it.

Sampling a robot as joint ORIGINS and measuring point-to-point distance has a
specific blind spot: it sees the joints and never the links between them, so
two arms can nearly touch while every origin is far from every other origin.
These tests pin the math against brute force, and pin the blind spot itself
with a pose measured off the shipped dual-arm profiles.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.safety.geometry import chain_distance, segment_segment_distances


def _d(p0, p1, q0, q1) -> float:
    return float(segment_segment_distances([p0], [p1], [q0], [q1])[0, 0])


@pytest.mark.parametrize(
    "name,p0,p1,q0,q1,expected",
    [
        ("perpendicular, offset in z", [-1, 0, 0], [1, 0, 0], [0, -1, 1], [0, 1, 1], 1.0),
        ("parallel", [0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0], 2.0),
        ("collinear with a gap", [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], 1.0),
        ("crossing", [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], 0.0),
        ("endpoint to endpoint", [0, 0, 0], [1, 0, 0], [1, 0, 3], [2, 0, 3], 3.0),
        ("degenerate: point vs segment", [5, 5, 5], [5, 5, 5], [0, 0, 0], [1, 0, 0],
         float(np.linalg.norm([4.0, 5.0, 5.0]))),
        ("degenerate: both points", [0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0], 5.0),
    ],
)
def test_known_configurations(name, p0, p1, q0, q1, expected):
    """Closed-form answers, including the degenerate cases a folded joint
    produces (two coincident origins = a zero-length segment)."""
    assert _d(p0, p1, q0, q1) == pytest.approx(expected, abs=1e-9), name


def test_matches_brute_force_on_random_segments():
    """The real correctness check: dense sampling along both segments. An
    earlier revision passed several hand-written cases while contracting the
    wrong einsum axis, and only brute force caught it."""
    rng = np.random.default_rng(0)
    ts = np.linspace(0.0, 1.0, 600)
    worst = 0.0
    for _ in range(200):
        P0, P1, Q0, Q1 = (rng.normal(0, 1, 3) for _ in range(4))
        A = P0 + ts[:, None] * (P1 - P0)
        B = Q0 + ts[:, None] * (Q1 - Q0)
        brute = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min()
        worst = max(worst, abs(brute - _d(P0, P1, Q0, Q1)))
    assert worst < 5e-3, f"max deviation {worst:.2e} exceeds sampling error"


def test_distances_are_symmetric_and_shaped():
    rng = np.random.default_rng(1)
    p0, p1 = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    q0, q1 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    d = segment_segment_distances(p0, p1, q0, q1)
    assert d.shape == (4, 3)
    assert np.allclose(d, segment_segment_distances(q0, q1, p0, p1).T)
    assert np.all(d >= 0.0)


def test_chain_distance_reports_which_links_are_closest():
    """The index pair goes into the rejection message, so an operator can see
    WHICH links nearly touched rather than just a number."""
    a = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])   # 2 links along x
    b = np.array([[2.0, 0.5, 0], [3.0, 0.5, 0]])            # 1 link, offset y
    dist, i, j = chain_distance(a, b)
    assert dist == pytest.approx(0.5)
    assert (i, j) == (1, 0)          # the SECOND link of a, the only one of b


def test_chain_distance_handles_a_single_point_chain():
    """A 1-point chain has no segments; it must still measure, not raise."""
    dist, _, _ = chain_distance(np.array([[0.0, 0, 0]]),
                                np.array([[3.0, 4, 0], [3.0, 4, 1]]))
    assert dist == pytest.approx(5.0)
    assert chain_distance(np.zeros((0, 3)), np.zeros((2, 3)))[0] == float("inf")


def test_segments_catch_a_near_miss_that_points_do_not():
    """THE reason this module exists, on the shipped dual-arm geometry.

    This joint-space pose pair was found by search over the so101_left /
    so101_right mounting (0.44 m apart): the arms' links pass 3.3 cm apart
    while every joint origin is 6.6 cm from every other origin. A 5 cm gate
    on point distance ACCEPTS it; on segment distance it rejects it.
    """
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.types import pose_to_transform, transform_points

    cfg = load_demo_config(arm="so101_mock")
    kin = Kinematics(
        model_path=cfg.arm.model,
        ee_frame=cfg.arm.get("ee_frame"),
        n_controlled=5,
        joint_signs=cfg.arm.get("joint_signs"),
        ik_task_weights=cfg.arm.get("ik_task_weights"),
    )

    def chain(q, pose):
        pts = np.vstack([kin.link_positions(q), kin.fk(q)[:3, 3][None, :]])
        return transform_points(pose_to_transform(pose), pts)

    # Found by random search over joint space; kept as literals so the test is
    # deterministic. Verified numbers: segment 0.0326 m, point 0.0660 m.
    q_left = np.array([0.89769885, -1.15446207, -0.82460201, -1.65029563, -0.24987628])
    q_right = np.array([-1.69067052, 0.97064568, -1.04908937, 0.21785942, -0.05184079])
    A = chain(q_left, [0.0, 0.22, 0.0, 0.0, 0.0, 0.0])
    B = chain(q_right, [0.0, -0.22, 0.0, 0.0, 0.0, 0.0])

    seg, _, _ = chain_distance(A, B)
    pts = float(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min())

    assert seg < 0.05, f"expected a near miss, links are {seg:.3f} m apart"
    assert pts > 0.05, f"expected points to miss it, they report {pts:.3f} m"
    assert pts - seg > 0.02, "the blind spot should be centimetres, not noise"


def test_the_check_is_cheap_enough_for_50hz():
    """It runs inside approve() on every waypoint; a slow safety check is
    itself a hazard. Budget is 20 ms per waypoint."""
    import time

    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    chain_distance(a, b)                      # warm up
    t0 = time.perf_counter()
    for _ in range(2000):
        chain_distance(a, b)
    per_call = (time.perf_counter() - t0) / 2000
    assert per_call < 2e-3, f"{per_call * 1e6:.0f} us/call is too slow for 50 Hz"
