"""People guard on base_pose arms (#374): everything is judged in the TABLE frame.

#374 made the skill runtime work in the running arm's base frame: FK and IK in
the arm frame, beliefs, agent coordinates and the occupancy map in the shared
table frame (spec 2026-10-17-base-pose-runtime-frame-design.md, §3.1). People
stand around the table, so `safety.people` (visitor_zones and visitor_sides)
is a TABLE-frame setting, like the map: an arm without a base_pose reads it
exactly as before (the two frames coincide), and an arm with one has every
arm-frame point lifted by its base_pose before it is compared:

- the guard's fixed poses (wave, rotate_joint, handover, throw) by FK, then
  `T_world_base`; place_at's point and remembered targets are already table
  frame and are NOT lifted again;
- the harness's TCP (approve, vet_step, vet_pose) by its own base_pose;
- the throw's swing side: the release direction rotated into the table frame.

Each test builds the zone (or person) where the arm PHYSICALLY goes and, as
the control, where the arm-frame numbers would land if read as table-frame
ones: the first must refuse before any command, the second must not.
Real `build_runtime` on the mock stack (tests/base_pose_oracle.py).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from base_pose_oracle import RAISED, YAWED, lift, moved_cfg, recorder, transform
from base_pose_oracle import make_runtime  # noqa: F401  (the fixture, requested through `build`)

from cascade.safety import people_guard

#: a quarter turn: the arm's forward (+x) is the table's left (+y)
QUARTER = [0.03, -0.22, 0.0, 0.0, 0.0, math.pi / 2]
POSES = {"yawed": YAWED, "raised": RAISED, "quarter": QUARTER}
PAIR = ["so101_left", "so101_right"]
#: the shipped pair's mounts (configs/arms/so101_{left,right}.yaml)
PAIR_Y = {"so101_left": 0.22, "so101_right": -0.22}
MARGIN = 0.03       # safety.min_clearance_m of every shipped profile (never relaxed)


def _people(block):
    """A config edit: `safety.people` = block at the top level and in every arm
    entry's resolved view (the arm-scoped `safety` section each harness reads)."""

    def edit(cfg):
        data = cfg._data
        sections = [data.get("safety")]
        for entry in [data.get("arm"), *(data.get("arms") or [])]:
            entry = entry._data if hasattr(entry, "_data") else entry
            resolved = (entry or {}).get("resolved")
            if resolved is not None:
                resolved = resolved._data if hasattr(resolved, "_data") else resolved
                sections.append(resolved.get("safety"))
        for safety in sections:
            if safety is not None:
                safety = safety._data if hasattr(safety, "_data") else safety
                safety["people"] = dict(block)
        return cfg

    return edit


def _zone(name, p):
    """A degenerate visitor zone at `p` (grown by the margin when parsed)."""
    p = [float(v) for v in np.asarray(p, dtype=float)[:3]]
    return {"name": name, "min": p, "max": p}


def _inside(p, centre) -> bool:
    return bool(np.all(np.abs(np.asarray(p, float)[:3] - np.asarray(centre, float)[:3]) <= MARGIN + 1e-9))


def _harness(safe):
    return safe.__dict__["harness"]


def _mark(rt):
    """(arm commands per raw MockArm, jaw commands) so far (base_pose_oracle's recorder)."""
    return recorder(rt).mark()


def _sent(rt, mark):
    """(arm commands, jaw commands) sent since `mark`."""
    cmds, jaws = recorder(rt).since(mark)
    return len(cmds), len(jaws)


@pytest.fixture
def build(request):
    """base_pose_oracle's `make_runtime`: real build_runtime on the mock stack,
    private stores per runtime, every runtime shut down."""
    return request.getfixturevalue("make_runtime")


def _single(build, pose, block=None):
    cfg = moved_cfg("mock", pose)
    if block is not None:
        _people(block)(cfg)
    return build(cfg)


def _refused_before_any_command(rt, out, mark, rule="destination_on_person"):
    assert out["ok"] is False, out
    assert out["refused_by"] == "people_guard", out
    assert out["safety_class"] == "safety:people"
    assert out["people_guard"]["rule"] == rule, out
    assert _sent(rt, mark) == (0, 0), "an arm or the jaws were commanded before the refusal"


# ── 1. the guard's fixed poses: FK in the arm frame, lifted by base_pose ──


@pytest.mark.parametrize("pose", ["yawed", "raised"])
def test_a_wave_is_vetted_where_the_mounted_arm_waves(build, pose):
    """Wave waypoint 2 (a wag) at its TABLE-frame point refuses the wave
    before its first command; the same zone at the arm-frame numbers (where
    an arm at the table origin would wag) does not."""
    T = transform(POSES[pose])
    probe = _single(build, POSES[pose])
    wag_b = probe.kin.fk(probe._wave_waypoints(1)[2])[:3, 3]
    wag_w = lift(T, wag_b)
    lifted = [lift(T, probe.kin.fk(q)[:3, 3]) for q in probe._wave_waypoints(1)]
    assert not _inside(wag_b, wag_w)
    assert not any(_inside(p, wag_b) for p in lifted), "the control zone must stay clear of the real wave"

    rt = _single(build, POSES[pose], {"visitor_zones": [_zone("rope", wag_w)]})
    mark = _mark(rt)
    out = rt.execute("wave", {"cycles": 1})
    _refused_before_any_command(rt, out, mark)
    assert out["people_guard"]["what"] == "the wave"
    assert out["people_guard"]["zone"] == "rope"
    np.testing.assert_allclose(out["people_guard"]["point"], np.round(wag_w, 3), atol=1e-9)

    rt = _single(build, POSES[pose], {"visitor_zones": [_zone("arm-frame numbers", wag_b)]})
    mark = _mark(rt)
    out = rt.execute("wave", {"cycles": 1})
    assert out["ok"] is True, out
    assert _sent(rt, mark)[0] > 0


@pytest.mark.parametrize("pose", ["yawed", "raised"])
def test_table_frame_points_and_remembered_targets_are_not_lifted_again(build, pose):
    """place_at's (x, y, z) and a belief's position are table-frame already:
    a zone on them refuses; the guard never moves them by base_pose."""
    T = transform(POSES[pose])
    point = np.array([0.30, 0.05, 0.10])
    mug = np.array([0.32, 0.12, 0.03])
    assert not _inside(lift(T, point), point) and not _inside(lift(T, mug), mug)
    zones = [_zone("place", point), _zone("mug", mug)]
    rt = _single(build, POSES[pose], {"visitor_zones": zones})
    rt.beliefs.update("blue mug", mug, 0.9, extent=np.array([0.08, 0.08, 0.10]), color="blue")
    mark = _mark(rt)
    rt.held_object = "red cube"
    out = rt.execute("place_at", {"x": point[0], "y": point[1], "z": point[2]})
    _refused_before_any_command(rt, out, mark)
    assert out["people_guard"]["zone"] == "place"
    rt.held_object = None
    out = rt.execute("grasp_object", {"label": "blue mug"})
    _refused_before_any_command(rt, out, mark)
    assert out["people_guard"]["zone"] == "mug"
    # the same targets with the zones where base_pose would move them: clear
    rt2 = _single(build, POSES[pose], {"visitor_zones": [_zone("place", lift(T, point)),
                                                                _zone("mug", lift(T, mug))]})
    rt2.beliefs.update("blue mug", mug, 0.9, extent=np.array([0.08, 0.08, 0.10]), color="blue")
    rt2.held_object = "red cube"
    assert people_guard.check_call(rt2, "place_at", {"x": point[0], "y": point[1], "z": point[2]}, rt2.arm) is None
    rt2.held_object = None
    assert people_guard.check_call(rt2, "grasp_object", {"label": "blue mug"}, rt2.arm) is None


# ── 2. the harness: every TCP waypoint lifted by its own base_pose ────────


@pytest.mark.parametrize("pose", ["yawed", "raised"])
def test_the_harness_reads_visitor_zones_in_the_table_frame(build, pose):
    """move_relative's route is planned inside the skill (the guard cannot
    see it): the harness refuses the route that ENDS in a table-frame zone
    before its first command, and approve()/vet_pose() read the lifted TCP."""
    T = transform(POSES[pose])
    probe = _single(build, POSES[pose])
    q0 = np.asarray(probe.arm.get_state().q, dtype=float)
    up_b = probe.kin.fk(q0)[:3, 3] + np.array([0.0, 0.0, 0.05])
    up_w = lift(T, up_b)
    assert not _inside(up_b, up_w)

    rt = _single(build, POSES[pose], {"visitor_zones": [_zone("above", up_w)]})
    mark = _mark(rt)
    out = rt.execute("move_relative", {"direction": "up", "distance_m": 0.05})
    assert out["ok"] is False, out
    assert "visitor zone 'above'" in out["error"], out
    assert _sent(rt, mark) == (0, 0)
    h = _harness(rt.arm)
    q1 = q0.copy()
    q1[0] += np.deg2rad(5.0)
    tcp1_w = lift(T, rt.kin.fk(q1)[:3, 3])
    h.limits = dataclasses.replace(h.limits, people=people_guard.parse(
        {"visitor_zones": [_zone("turn", tcp1_w)]}, MARGIN))
    assert "visitor zone 'turn'" in (h.vet_pose(q1) or "")
    from cascade.types import SafetyViolation

    with pytest.raises(SafetyViolation, match="visitor zone 'turn'"):
        h.approve(q0, q1, 2.0)

    # the control: the zone at the arm-frame numbers, the arm moves
    rt = _single(build, POSES[pose], {"visitor_zones": [_zone("above", up_b)]})
    mark = _mark(rt)
    out = rt.execute("move_relative", {"direction": "up", "distance_m": 0.05})
    assert out["ok"] is True, out
    assert _sent(rt, mark)[0] > 0


# ── 3. the throw's side: the release direction in the table frame ─────────


@pytest.mark.parametrize("sides, refused", [(["left"], True), (["forward"], False)])
def test_a_throw_side_is_the_table_side_the_mounted_arm_swings_to(build, sides, refused):
    """A quarter-turned mount: the arm's forward swing goes to the TABLE's
    left. visitor_sides name table sides (where people stand), so `left`
    refuses that throw and `forward` does not. Through execute() a base_pose
    arm's throw is still refused by #352's world-target refusal before the
    guard runs, so the side is checked on `check_call` itself (the same call
    execute() makes, on the running arm)."""
    rt = _single(build, QUARTER, {"visitor_sides": sides})
    _, release = rt._throw_keyframes("forward")
    assert rt.kin.fk(release)[0, 3] > 0.3            # forward in the arm frame
    refusal = people_guard.check_call(rt, "throw", {"label": "red cube", "direction": "forward"}, rt.arm)
    if refused:
        assert refusal is not None and refusal.evidence["rule"] == "throw_toward_visitor_side"
        assert refusal.evidence["side"] == "left"
    else:
        assert refusal is None
    mark = _mark(rt)
    out = rt.execute("throw", {"label": "red cube", "direction": "forward"})
    assert out["ok"] is False and _sent(rt, mark) == (0, 0), out


# ── 4. the shipped so101 pair: each arm vetted where IT stands ────────────


def _pair(build, block=None):
    cfg = moved_cfg(PAIR, {}, move_cameras=False)
    if block is not None:
        _people(block)(cfg)
    return build(cfg)


def _rotated_tcp(rt, arm_name, degrees):
    """(arm frame, table frame) TCP of rotate_joint(joint 1, degrees) on `arm_name`."""
    safe = rt.arm_rig.arms[arm_name]
    h = _harness(safe)
    q = np.asarray(safe.get_state().q, dtype=float).copy()
    q[0] += math.radians(degrees)
    p_b = np.asarray(h.kin.fk(q)[:3, 3], dtype=float)
    return p_b, p_b + np.array([0.0, PAIR_Y[arm_name], 0.0])


def test_the_pair_vets_each_arm_against_a_person_where_that_arm_stands(build):
    """A person localized beside the LEFT arm's rotated TCP (table frame, the
    frame beliefs live in): the left arm's rotate_joint is refused before any
    command on either arm; the right arm, 0.44 m away, still turns."""
    rt = _pair(build)
    left_b, left_w = _rotated_tcp(rt, "so101_left", 10.0)
    _, right_w = _rotated_tcp(rt, "so101_right", 10.0)
    centre = left_w + np.array([0.0, 0.26, 0.0])       # column half-width 0.25 + margin 0.03
    half = people_guard.PERSON_HALF_WIDTH_M + MARGIN
    assert abs(left_w[1] - centre[1]) <= half < abs(left_b[1] - centre[1])
    assert abs(right_w[1] - centre[1]) > half
    b = rt.beliefs.update("person", np.array([centre[0], centre[1], 0.4]), 0.9,
                          extent=np.array([0.40, 0.30, 1.0]), color="black")
    assert b.label == "person"
    mark = _mark(rt)
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 10, "arm": "so101_left"})
    _refused_before_any_command(rt, out, mark)
    assert out["people_guard"]["person"] == "person"
    np.testing.assert_allclose(out["people_guard"]["point"], np.round(left_w, 3), atol=1e-9)
    raws = {name: recorder(rt).raws[i] for i, name in enumerate(rt.arm_rig.arms)}
    left0, right0 = len(raws["so101_left"].commands), len(raws["so101_right"].commands)
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 10, "arm": "so101_right"})
    assert out["ok"] is True, out
    assert len(raws["so101_left"].commands) == left0 and len(raws["so101_right"].commands) > right0


def test_the_pair_harnesses_read_one_table_frame_zone(build):
    """One configured zone, in the table frame, around the LEFT arm's rotated
    TCP: the left arm's guard refuses that pose, its harness refuses the same
    pose (vet_pose), and the right arm's harness does not (its own base_pose)."""
    probe = _pair(build)
    left_b, left_w = _rotated_tcp(probe, "so101_left", 10.0)
    right_b, right_w = _rotated_tcp(probe, "so101_right", 10.0)
    assert not _inside(left_b, left_w) and not _inside(right_w, left_w)
    rt = _pair(build, {"visitor_zones": [_zone("left aisle", left_w)]})
    mark = _mark(rt)
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 10, "arm": "so101_left"})
    _refused_before_any_command(rt, out, mark)
    assert out["people_guard"]["zone"] == "left aisle"
    for name, refused in (("so101_left", True), ("so101_right", False)):
        safe = rt.arm_rig.arms[name]
        q = np.asarray(safe.get_state().q, dtype=float).copy()
        q[0] += math.radians(10.0)
        reason = _harness(safe).vet_pose(q) or ""
        assert ("visitor zone 'left aisle'" in reason) is refused, (name, reason)


def test_an_arm_whose_mount_cannot_be_read_is_refused(build):
    """Fail closed: a base_pose that is not a rigid transform gives the guard
    no table frame, so it refuses (execute() already refuses such an arm
    earlier, #374 D4: nothing is sent either way)."""
    rt = _single(build, YAWED)
    h = _harness(rt.arm)
    h.base_pose = np.diag([2.0, 1.0, 1.0, 1.0])
    refusal = people_guard.check_call(rt, "wave", {"cycles": 1}, rt.arm)
    assert refusal is not None and refusal.evidence == {"rule": "unknown_arm_frame"}
    mark = _mark(rt)
    out = rt.execute("wave", {"cycles": 1})
    assert out["ok"] is False and _sent(rt, mark) == (0, 0), out
