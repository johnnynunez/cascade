"""Escape rule for TCP/link map clearance, and the route-end rule (spec §3.1).

Offline: CartesianKin / case() from test_home_routes, synthetic occupancy
stand-ins, the real SafetyHarness, trajectory planner and MockArm. Inside the
0.03 m margin a step is admitted only when every point inside it was already
inside at q_prev and strictly retreats (or the step is an exact hold); a
route must end outside the margin. The margin itself never changes.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from test_clearance_lift_route import lift_case
from test_home_routes import CartesianKin, case

from cascade.safety.harness import CLEARANCE_ESCAPE_TOL_M, ESCAPE_NOTE
from cascade.safety.trajectory import vet_route, vet_segment
from cascade.types import SkillError

VOXEL = np.array([.3, 0., .3 - .028])                    # 2.8 cm under the hovering TCP


class PointsMap:
    """Occupancy stand-in: distance to the nearest of a few obstacle points."""

    required = False
    status = None                                         # OccupancyMap.status: the bridge's probe reply

    def __init__(self, *voxels, unknown_blocks=False):
        self.voxels = np.asarray(voxels, dtype=float).reshape(-1, 3)
        self.unknown_blocks = unknown_blocks
        self.queries = []

    def clearance(self, points):
        points = np.asarray(points, dtype=float)
        self.queries.append(len(points))
        return np.linalg.norm(points[:, None, :] - self.voxels[None, :, :], axis=2).min(axis=1)


def points_map(*voxels):
    return PointsMap(*voxels)


def flat_top(z_top):
    """An infinite flat obstacle top: clearance = z - z_top everywhere."""
    return SimpleNamespace(clearance=lambda p: np.asarray(p, dtype=float)[:, 2] - z_top)


class LinkKin(CartesianKin):
    """A second clearance point: the 'link' 5 cm beside the TCP in +x."""

    def link_positions(self, q):
        return np.array([[0., 0., .1], np.asarray(q, dtype=float) + [.05, 0., 0.]])


def test_hover_at_28mm_admits_home_a_3cm_lift_and_up_25mm():
    for goal in ([.3, 0., .33], [.3, 0., .325], [.4, 0., .4]):
        safe, raw, h = case(occupancy=points_map(VOXEL))
        assert safe.move_planned(goal) and raw.commands and h.violations == []
        np.testing.assert_allclose(raw.get_state().q, goal)


def test_lateral_1cm_over_a_flat_top_is_refused():        # clearance = z - 0.272 everywhere
    safe, raw, _ = case(occupancy=flat_top(.272))
    with pytest.raises(SkillError, match=r"0\.028000000 m below 0\.030000000 m.*strictly retreating"):
        safe.move_planned([.31, 0., .3])
    assert raw.commands == []


def test_hold_is_admitted_then_closing_in_is_refused():
    _, _, h = case(occupancy=points_map(VOXEL)); q = np.array([.3, 0., .3])
    assert h.vet_step(q, q, .02) is None
    assert "strictly retreating" in h.vet_step(q, q - [0., 0., .001], .02)


def test_a_micrometre_closing_step_is_not_a_hold():
    # np.allclose's default rtol (1e-5) would call a 2 um step at 0.3 a "hold"
    _, _, h = case(occupancy=points_map(VOXEL)); q = np.array([.3, 0., .3])
    assert "strictly retreating" in h.vet_step(q, q - [0., 0., 2e-6], .02)


def test_a_hold_never_admits_a_point_that_was_outside_the_margin():
    # a 0.5 nm "hold" that crosses the 3 cm boundary: not already inside at q_prev
    _, _, h = case(occupancy=flat_top(.27 - 2.5e-10)); q = np.array([.3, 0., .3])
    reason = h.vet_step(q, q - [0., 0., 5e-10], .02)
    assert reason.startswith("point 0 clearance") and ESCAPE_NOTE not in reason


def test_a_step_improving_by_exactly_the_tolerance_is_refused():
    # strict: d_next must EXCEED d_prev + CLEARANCE_ESCAPE_TOL_M
    d0 = .028
    occupancy = SimpleNamespace(clearance=lambda p: np.where(
        np.asarray(p)[:, 2] > .3, d0 + CLEARANCE_ESCAPE_TOL_M, d0))
    _, _, h = case(occupancy=occupancy); q = np.array([.3, 0., .3])
    assert "strictly retreating" in h.vet_step(q, q + [0., 0., .001], .02)
    occupancy.clearance = lambda p: np.where(np.asarray(p)[:, 2] > .3, d0 + 2 * CLEARANCE_ESCAPE_TOL_M, d0)
    assert h.vet_step(q, q + [0., 0., .001], .02) is None


def test_a_point_entering_the_margin_refuses_although_another_retreats():
    # kin whose link point is tcp + [.05, 0, 0]; voxels: VOXEL and one 3.1 cm beside the link
    # step up-and-toward: TCP 2.8 -> 2.91 cm, link 3.1 -> 2.90 cm
    _, _, h = case(occupancy=points_map(VOXEL, [.35 + .031, 0., .3]))
    h.kin = LinkKin()
    q0 = np.array([.3, 0., .3]); q1 = q0 + [.002, 0., .001]
    reason = h.vet_step(q0, q1, .02)
    assert "point 1 clearance 0.0290" in reason and reason.endswith(ESCAPE_NOTE)
    # the same TCP retreat with nothing beside the link is admitted
    h.occupancy = points_map(VOXEL)
    assert h.vet_step(q0, q1, .02) is None


def test_prev_and_next_points_share_one_map_read():
    occupancy = points_map(VOXEL)
    _, _, h = case(occupancy=occupancy); q = np.array([.3, 0., .3])
    assert h.vet_step(q + [0., 0., .1], q + [0., 0., .101], .02) is None
    assert occupancy.queries == [4]       # outside the margin: TCP + link, at q_next and q_prev
    occupancy.queries.clear()
    assert h.vet_step(q, q, .02) is None
    assert occupancy.queries == [4]       # an exact hold has no interior to sample
    occupancy.queries.clear()
    assert h.vet_step(q, q + [0., 0., .001], .02) is None
    # an escape: the ends, then ONE deciding read of the ends plus 1 interior configuration
    assert occupancy.queries == [4, 4 + 2]


def test_closing_holds_are_refused_at_the_first_one():
    # review probe 1e: each 0.9 nm step is a "hold" (joints within 1e-9), 1000 of them close 0.9 um
    _, _, h = case(occupancy=points_map(VOXEL)); q = np.array([.3, 0., .3])
    step, admitted = np.array([0., 0., -.9e-9]), 0
    for _ in range(1000):
        reason = h.vet_step(q, q + step, .02)
        if reason is not None:
            break
        q, admitted = q + step, admitted + 1
    assert admitted == 0, f"{admitted} closing holds admitted"
    assert reason.startswith("point 0 clearance 0.027999999") and reason.endswith(ESCAPE_NOTE)
    # a hold that does not close in stays admitted
    assert h.vet_step(q, q + [0., 0., .9e-9], .02) is None


@pytest.mark.parametrize("dt", [.02, 0.])
def test_an_escape_step_that_jumps_a_voxel_is_refused(dt):
    # review probe 1g: 5 mm above a point voxel -> 6 mm below it in one 1.1 cm step;
    # both ends "retreat" (5 -> 6 mm), the interior passes through the voxel
    voxel = np.array([.3, 0., .25])
    occupancy = points_map(voxel)
    _, _, h = case(occupancy=occupancy)
    reason = h.vet_step(voxel + [0., 0., .005], voxel - [0., 0., .006], dt)
    assert reason.startswith("point 0 clearance 0.002250000 m below 0.030000000 m")
    assert "mid-step" in reason and reason.endswith(ESCAPE_NOTE)
    assert occupancy.queries == [4, 4 + 2 * 3]             # 1.1 cm at 5 mm spacing: 3 interior samples


def test_an_honest_escape_is_sampled_along_the_step_and_admitted():
    occupancy = points_map([.3, 0., .3 - .015])
    _, _, h = case(occupancy=occupancy); q = np.array([.3, 0., .3])
    assert h.vet_step(q, q + [0., 0., .012], .02) is None  # 15 -> 27 mm; 18, 21, 24 mm between
    assert occupancy.queries == [4, 4 + 2 * 3]


def test_an_escape_is_decided_on_one_map_read():
    # The map refreshes between the two reads. On the new grid the step dips mid-way
    # (29.0 -> 28.5 -> 29.5 mm). Mixing in the first grid's q_prev value (28.0 mm) would admit it.
    grids = [lambda z: z - .272,
             lambda z: np.interp(z, [.3, .3005, .301], [.029, .0285, .0295])]
    reads = []

    def clearance(p):
        reads.append(len(p))
        return grids[min(len(reads), 2) - 1](np.asarray(p, dtype=float)[:, 2])

    _, _, h = case(occupancy=SimpleNamespace(clearance=clearance)); q = np.array([.3, 0., .3])
    reason = h.vet_step(q, q + [0., 0., .001], .02)
    assert reads == [4, 6]
    assert reason.startswith("point 0 clearance 0.028500000") and "mid-step" in reason


def test_interior_spacing_follows_a_finer_reported_voxel_and_is_capped():
    occupancy = points_map(VOXEL); occupancy.status = {"voxel": .001}
    _, _, h = case(occupancy=occupancy); q = np.array([.3, 0., .3])
    assert h.vet_step(q, q + [0., 0., .0012], .02) is None
    assert occupancy.queries == [4, 4 + 2 * 3]             # 1.2 mm at 0.5 mm spacing
    occupancy.queries.clear(); occupancy.status = {"voxel": .05}
    occupancy.voxels = np.array([[.3, 0., .28]])
    assert h.vet_step(q, q + [0., 0., .006], .02) is None  # 20 -> 26 mm
    assert occupancy.queries == [4, 4 + 2 * 2]             # a coarser voxel never thins the 5 mm spacing
    occupancy.queries.clear(); occupancy.status = {"voxel": .001}
    occupancy.voxels = np.array([[.3, 0., .29]])
    assert h.vet_step(q, q + [0., 0., .019], 0.) is None   # 10 -> 29 mm needs 38: capped at 16
    assert occupancy.queries == [4, 4 + 2 * 16]


def test_an_unobserved_interior_sample_refuses_the_escape():
    # `unobserved: blocked`: both ends are observed and retreat, the middle of the step is not
    def clearance(p):
        z = np.asarray(p, dtype=float)[:, 2]
        return np.where((z > .3002) & (z < .3008), np.nan, z - .272)

    occupancy = SimpleNamespace(clearance=clearance, required=True, unknown_blocks=True)
    _, _, h = case(occupancy=occupancy); q = np.array([.3, 0., .3])
    reason = h.vet_step(q, q + [0., 0., .001], .02)
    assert reason.startswith("point 0 has unobserved clearance (occupancy map, mid-step)")


def test_attached_samples_refuse_prev_points_locally():
    occupancy = SimpleNamespace(clearance=lambda p: np.full(len(p), .1),
                                payload_clearance=lambda p: np.full(len(p), .1))
    _, _, h = case(occupancy=occupancy); pts = np.array([[.3, 0., .3]])
    assert h._occupancy_violation(pts, None, attached=True) is None
    reason = h._occupancy_violation(pts, None, attached=True, prev_points=pts, hold=True)
    assert reason is not None and "never escape" in reason


def test_route_ending_inside_the_margin_is_refused():
    safe, raw, h = case(occupancy=points_map(VOXEL)); s = np.array([.3, 0., .3])
    assert h.vet_step(s, s + [0., 0., .001], .02) is None          # the step itself retreats
    assert vet_segment(h, s, s + [0., 0., .001], 3.) is None        # and so does every step
    assert vet_route(h, s, [s + [0., 0., .001]], 3.).startswith(
        "route ends inside the margin: point 0 clearance 0.029")
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned(s + [0., 0., .001])
    assert raw.commands == []


def test_route_end_rule_checks_the_start_of_an_empty_route():
    _, _, h = case(occupancy=points_map(VOXEL))
    assert vet_route(h, np.array([.3, 0., .3]), [], 3.).startswith(
        "route ends inside the margin: point 0 clearance 0.028")
    assert vet_route(h, np.array([.3, 0., .4]), [], 3.) is None


def test_route_end_violation_is_the_static_check_without_escape():
    _, _, h = case(occupancy=points_map(VOXEL))
    reason = h.route_end_violation(np.array([.3, 0., .301]))
    assert reason.startswith("point 0 clearance 0.029") and ESCAPE_NOTE not in reason
    assert h.route_end_violation(np.array([.3, 0., .33])) is None
    h.occupancy = None
    assert h.route_end_violation(np.array([.3, 0., .301])) is None
    h.occupancy, h.kin = points_map(VOXEL), None
    assert h.route_end_violation(np.array([.3, 0., .301])) is None


def test_route_end_violation_uses_the_active_grasp_exemption():
    _, _, h = case(occupancy=points_map(VOXEL))
    h._grasp_exempt = (np.array([.3, 0.]), .05, .2)
    assert h.route_end_violation(np.array([.3, 0., .301])) is None


def test_harness_without_route_end_rule_needs_no_end_check():
    harness = SimpleNamespace(limits=SimpleNamespace(max_joint_vel=1.2),
                              vet_step=lambda *args: None)
    assert vet_route(harness, np.zeros(3), [np.ones(3) * .1], 3.) is None


def test_first_lift_safe_final_leg_blocked():             # test_clearance_lift_route.py:94 pattern
    # lift_case(wall_top=.40) + VOXEL: both +3/+6 cm lifts retreat, the second leg hits the keep-out
    safe, raw, harness, start, goal = lift_case(wall_top=.40)
    harness.occupancy = points_map(VOXEL)
    for height in (.03, .06):
        lift = start + [0., 0., height]
        assert vet_segment(harness, start, lift, 3.) is None
        assert "keep-out" in vet_segment(harness, lift, goal, 3.)
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned(goal)
    assert raw.commands == [] and harness.violations == []
    assert harness.limits.min_clearance_m == .03


def test_payload_samples_never_escape():                  # payload .019 at start, .025 at goal
    occupancy = SimpleNamespace(
        clearance=lambda p: np.full(len(p), .1),
        payload_points=lambda pose: pose[None, :3, 3],
        payload_clearance=lambda p: .019 + (np.asarray(p)[:, 2] - .3) * .06,
    )
    safe, raw, h = case(occupancy=occupancy)
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned([.4, 0., .4])
    assert raw.commands == []
    assert "attached object point 0 clearance 0.019" in h.vet_step(
        np.array([.3, 0., .3]), np.array([.3, 0., .3]), .02)


def test_static_vets_never_escape():
    _, _, h = case(occupancy=points_map(VOXEL))
    reason = h.vet_pose(np.array([.3, 0., .3]))
    assert "0.028000000 m below" in reason and ESCAPE_NOTE not in reason


@pytest.mark.parametrize("unobserved", ["next", "prev"])
@pytest.mark.parametrize("step", ["hold", "retreat"])
def test_unobserved_clearance_never_counts_as_an_escape(unobserved, step):
    """NaN (unobserved, `unobserved: blocked`) at q_prev or q_next is never an escape."""
    q0 = np.array([.3, 0., .3])
    # a hold within 1e-9 that still crosses the NaN boundary, or a 1 mm retreat
    q1 = q0 + [0., 0., 5e-10 if step == "hold" else .001]
    cut = .3 + (2.5e-10 if step == "hold" else .0005)

    def clearance(p):
        z = np.asarray(p, dtype=float)[:, 2]
        d = z - .272
        return np.where(z > cut, np.nan, d) if unobserved == "next" else np.where(z < cut, np.nan, d)

    occupancy = SimpleNamespace(clearance=clearance, required=True, unknown_blocks=True)
    _, _, h = case(occupancy=occupancy)
    reason = h.vet_step(q0, q1, .02)
    if unobserved == "next":
        assert reason.startswith("point 0 has unobserved clearance")
    else:
        assert reason.startswith("point 0 clearance 0.02")
