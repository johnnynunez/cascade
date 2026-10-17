"""After an air grasp the jaws reopen and the arm withdraws; a grasp and
reset_scene open empty closed jaws first (PR-A jaws fix round).

Newton E2E on PR-A 73bd9eb: the jaws closed fully on air (gripper_pos 0.15,
both fingers 7.5 mm). The air-grasp exit withdrawal was refused ("finger
feedback left the geometrically checked opening interval"), nothing reopened
the jaws, and every later pick failed at route preflight: the observed-finger
gate built from the closed fingers sweeps them closed..open on every descent
and drops every candidate ("descent finger intersects observed surface:
target"). `open_gripper` first made the pick place again.

The observed-finger tests run the REAL `ObservedFingerGate` / `ObservedScene`
(calibrated reBot fingers, a synthetic same-frame cloud) on the fake Isaac
bridge of `test_grasp_evidence`, whose finger feedback follows the commanded
gripper position like Newton's (fingers = gripper_pos x 50 mm). The mapping-only
tests run the real mock stack. No hardware, no GPU, no bridge, no GraspGen-X.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.grasping import observed_scene
from cascade.grasping.observed_scene import (
    OPEN_TRACKING_ENVELOPE_M,
    FingerGeometry,
    ObservedFingerGate,
    ObservedScene,
)
from cascade.skills import carry_attachment
from cascade.skills import runtime as runtime_mod
from cascade.types import Frame, MotionHalted, SafetyViolation, SkillError
from conftest import needs_pin
from test_agent_results_compose import (  # noqa: F401 -- mock_runtime is a fixture
    ARGS,
    COMPOSITES,
    MOVE_HOME_FIRST,
    _failed_once,
    _nudged,
    _red,
    _seen,
    mock_runtime,
)
from test_grasp_evidence import receipt, synthetic_git_metadata  # noqa: F401 -- a fixture
from test_grasp_evidence import runtime as evidence_runtime

AIR = "air grasp: gripper closed fully, object not held"
#: the calibrated finger stroke (m), as the bridge serializes it (float32)
UPPER = float(np.float32(.05))
#: the rig's grasp position and home TCP (translation-only kinematics: q = TCP)
CENTRE = np.array([.2, 0., .05])
HOME = np.array([.2, 0., .2])
HOVER = CENTRE + [0., 0., .1]  # pregrasp_offset_m .1 along the top-down approach
#: the target's two observed faces: between the OPEN fingers at the grasp pose,
#: inside the fingers' closed..open sweep there (Newton's 4 cm cube)
TARGET = ((CENTRE + [0., .02, 0.], True), (CENTRE + [0., -.02, 0.], True))
#: what the rig's commands are (reads aside)
COMMANDS = ("gripper", "joints")
GEOMETRY = FingerGeometry()

# ── a real observed scene from a few base points ────────────────────────────
#
# A camera 1 m above x = 0.2, looking down; one image row (cam y = base x), so
# each point with x = 0.2 gets its own pixel (to ~25 um) and every other pixel
# has no depth.

FX = 2e4
CAMERA = np.array([.2, 0., 1.])
R_CAMERA = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])  # columns: cam x, y, z in base


def scene(points) -> ObservedScene:
    cams = [R_CAMERA.T @ (np.asarray(p, dtype=float) - CAMERA) for p, _ in points]
    assert all(abs(c[1]) < 1e-12 and c[2] > 0 for c in cams)
    cols = [int(round(FX * c[0] / c[2])) for c in cams]
    assert len(set(cols)) == len(cols)
    cx = -min(cols) + 2
    width = max(cols) + cx + 3
    depth = np.zeros((1, width))
    target = np.zeros((1, width), bool)
    for (_, is_target), col, cam in zip(points, cols, cams):
        depth[0, col + cx] = cam[2]
        target[0, col + cx] = is_target
    K = np.array([[FX, 0., cx], [0., FX, 0.], [0., 0., 1.]])
    capture = dict(backend="isaac", source=("test", 0), camera="cam", t=1., proprioception=dict(
        version=1, backend="isaac", robot_id="/robot", joint_convention="asset", producer_epoch="test",
        t=1., time_source="physics_loop_monotonic"))
    frame = Frame(np.zeros((1, width, 3), np.uint8), depth, K, depth_source="sensor",
                  capture=capture, robot_mask=np.zeros((1, width), bool))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_CAMERA, CAMERA
    return ObservedScene(frame, target, T, source=("test", 0), robot_id="/robot", epoch="test")


class Kin:
    """The rig's translation-only kinematics (q = TCP position), with the
    keyword IK the exit's `lift_ik` calls."""

    def ik(self, pose, q_init=None, *_a, **_k):
        return SimpleNamespace(q=np.asarray(pose, dtype=float)[:3, 3].copy(), success=True, error=0.)

    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q, dtype=float)[:3]
        return pose


def rig(monkeypatch, points=TARGET, *, stall_at=None):
    """The fake Isaac bridge (`test_grasp_evidence.runtime`) on Newton with
    individual finger feedback, and the real observed-finger gate on a scene of
    `points`. `stall_at`: the jaws stop there when closing (an object between
    them); None = nothing stops them (an air grasp)."""
    rt, calls, fix, frame = evidence_runtime(monkeypatch)
    bridge = rt.arm.raw._client
    read = bridge.state

    def state(*, timeout_s=None):
        sample = read(timeout_s=timeout_s)
        sample["physics_clock"].update(engine="newton", clock="newton_stage")
        sample["gripper_joints"] = dict(version=1, names=["joint_left", "joint_right"],
                                        position_m=[bridge.gripper_pos * UPPER] * 2,
                                        lower_m=[0., 0.], upper_m=[UPPER] * 2)
        return sample

    def state(*, timeout_s=None, _state=state):
        if bridge.opening:   # slow jaws: each read moves them one step toward open
            bridge.gripper_pos = bridge.opening.pop(0)
        return _state(timeout_s=timeout_s)

    def gripper(pos, effort):
        calls.append(("gripper", pos, effort))
        for hook in bridge.on_gripper:
            hook(pos)
        if bridge.stuck:
            return
        if bridge.slow and pos > bridge.gripper_pos:
            bridge.opening = [p for p in (.5, .97, .99, 1.) if p > bridge.gripper_pos]
            return
        if bridge.open_stall is not None and pos > bridge.gripper_pos:
            bridge.gripper_pos = bridge.open_stall   # opening jaws that stop short
            return
        bridge.gripper_pos = pos if bridge.stall_at is None else max(bridge.stall_at, pos)

    bridge.state, bridge.gripper = state, gripper
    bridge.stall_at, bridge.stuck, bridge.on_gripper = stall_at, False, []
    bridge.slow, bridge.opening, bridge.open_stall = False, [], None
    rt.kin = rt.arm.harness.kin = Kin()
    gates = []

    def for_runtime(runtime, _frame, _fix, measured):
        gates.append(ObservedFingerGate(scene(points), GEOMETRY, runtime.kin, measured,
                                        uncertainty_m=OPEN_TRACKING_ENVELOPE_M))
        return gates[-1]

    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    monkeypatch.setattr(observed_scene, "for_runtime", for_runtime)
    return SimpleNamespace(rt=rt, calls=calls, fix=fix, frame=frame, bridge=bridge, gates=gates,
                           harness=rt.arm.harness)


def grasp(r):
    """The rig's grasp, rigid profile: the stage-2 close commands 0.15 open, so
    an air grasp leaves both fingers at 7.5 mm (Newton)."""
    return r.rt.skill_grasp_object("orange", material="rigid", _fix=r.fix, _frame=r.frame)


def commands(calls):
    return [c for c in calls if c[0] in COMMANDS]


def fingers(r):
    return r.bridge.gripper_pos * UPPER


# ── RED (Newton): the air grasp's exit and the next pick ────────────────────


def test_an_air_grasp_reopens_the_checked_opening_then_withdraws(monkeypatch):
    r = rig(monkeypatch)
    res = grasp(r)
    assert res["error"] == AIR, res                      # nothing failed after the close
    assert res["jaws_reopen"] == "reopened"
    assert res["exit_withdrawal"] == "completed (+3 cm)"
    grips = [c[1] for c in r.calls if c[0] == "gripper"]
    # pre-grasp open, the two close stages, then the reopen to the SAME opening
    assert grips == pytest.approx([1., .5, .15, 1.])
    assert fingers(r) == UPPER
    gate = r.gates[-1]
    assert np.all(gate.lower <= UPPER) and np.all(UPPER <= gate.upper)   # inside the checked interval
    # the withdrawal ran after the reopen and lifted the TCP 3 cm above the hover
    close = [i for i, c in enumerate(r.calls) if c[0] == "gripper" and c[1] == pytest.approx(.15)][0]
    reopen = r.calls.index(("gripper", 1., .8), close)
    assert [c for c in r.calls[reopen:] if c[0] == "joints"], "no withdrawal after the reopen"
    np.testing.assert_allclose(r.bridge.q, HOVER + [0., 0., .03], atol=1e-9)


def test_after_an_air_grasp_the_next_pick_grasps(monkeypatch):
    """The Newton sequence: an air grasp, then the next pick of the same
    object. Today the jaws stay closed and the next pick's selection drops
    every candidate."""
    r = rig(monkeypatch)
    first = grasp(r)
    assert first["ok"] is False and first["error"].startswith(AIR)
    r.bridge.stall_at = .35                               # this time the object stops the jaws
    second = grasp(r)
    assert second["held"] == "orange" and second["grip_verified"] is True, second
    assert np.all(r.gates[-1].lower >= UPPER - 2 * OPEN_TRACKING_ENVELOPE_M)  # selection saw open jaws


def test_a_grasp_that_starts_with_closed_jaws_opens_them_before_selection(monkeypatch):
    r = rig(monkeypatch, stall_at=.35)
    r.bridge.gripper_pos = .15                            # left closed: fingers 7.5 mm
    res = grasp(r)
    assert res["held"] == "orange", res
    probe, chosen = r.gates[0], r.gates[1]
    assert np.all(probe.lower < .008)                     # the probe saw the closed jaws ...
    assert np.all(chosen.lower >= UPPER - 2 * OPEN_TRACKING_ENVELOPE_M)  # ... selection the open ones
    first = commands(r.calls)
    assert first[:2] == [("gripper", 1., .8), ("gripper", 1., .8)], first   # opened, then the pre-grasp open
    assert first[2][0] == "joints"


# ── the reopen after an air grasp ───────────────────────────────────────────


def _gate_at(r, points, q):
    """A selection gate (open jaws) with the arm at `q`, then the jaws closed
    on air there (no command: the close already happened)."""
    r.bridge.q = np.asarray(q, dtype=float).copy()
    gate = ObservedFingerGate(scene(points), GEOMETRY, r.rt.kin, r.rt.arm.get_state(),
                              uncertainty_m=OPEN_TRACKING_ENVELOPE_M)
    r.bridge.gripper_pos = .15
    return gate


def test_an_observed_surface_inside_the_reopening_sweep_refuses_the_reopen(monkeypatch):
    r = rig(monkeypatch)
    beside = HOVER + [0., .02, 0.]   # between the open fingers: inside the closed..open sweep
    gate = _gate_at(r, TARGET + ((beside, False),), HOVER)
    n0 = len(r.calls)
    out = r.rt._air_grasp_reopen(1., gate, halt_generation=0)
    assert out.startswith("refused: reopening fingers intersects observed surface"), out
    assert "other observed surface" in out
    assert commands(r.calls[n0:]) == [] and fingers(r) == pytest.approx(.15 * UPPER)
    # premise: the same reopen runs without that surface
    gate = _gate_at(r, TARGET, HOVER)
    assert r.rt._air_grasp_reopen(1., gate, halt_generation=0) == "reopened"
    assert commands(r.calls[n0:]) == [("gripper", 1., .8)] and fingers(r) == UPPER


def test_the_target_itself_is_not_exempt_from_the_reopening_sweep(monkeypatch):
    r = rig(monkeypatch)
    gate = _gate_at(r, ((HOVER + [0., .02, 0.], True), (HOVER + [0., -.02, 0.], True)), HOVER)
    n0 = len(r.calls)
    out = r.rt._air_grasp_reopen(1., gate, halt_generation=0)
    assert out.startswith("refused: reopening fingers intersects observed surface"), out
    assert "'surface': 'target'" in out
    assert commands(r.calls[n0:]) == []


def test_reopening_fingers_hidden_behind_observed_depth_refuse_the_reopen(monkeypatch):
    r = rig(monkeypatch)
    # a non-target surface 2 cm ABOVE the reopening sweep, on the camera rays
    # that reach the inner finger faces: those faces lie behind it
    above = HOVER + [0., .02, .04]
    gate = _gate_at(r, TARGET + ((above, False),), HOVER)
    assert gate.scene.conflict(r.rt.kin.fk(HOVER), gate.closing_envelopes) is None  # not a contact ...
    n0 = len(r.calls)
    out = r.rt._air_grasp_reopen(1., gate, halt_generation=0)
    assert out.startswith("refused: reopening fingers occluded by non-target depth"), out  # ... occluded
    assert commands(r.calls[n0:]) == []


def test_jaws_that_do_not_reopen_leave_the_arm_where_it_is(monkeypatch):
    r = rig(monkeypatch)
    r.bridge.on_gripper.append(lambda pos: setattr(r.bridge, "stuck", pos == 1. and r.bridge.gripper_pos < 1.))
    res = grasp(r)
    timeout = runtime_mod.JAWS_OPEN_TIMEOUT_S
    assert res["jaws_reopen"] == f"refused: the jaws did not open within {timeout:g} s", res
    assert res["exit_withdrawal"] == "skipped: the jaws did not reopen"
    assert res["error"] == f"{AIR}; jaws not reopened: the jaws did not open within {timeout:g} s"
    reopen = len(r.calls) - r.calls[::-1].index(("gripper", 1., .8)) - 1
    assert commands(r.calls[reopen + 1:]) == [], "the arm moved with jaws that did not reopen"
    np.testing.assert_allclose(r.bridge.q, HOVER, atol=1e-9)


@pytest.mark.parametrize("why", ["provisional hold", "carry attachment"])
def test_the_reopen_never_opens_jaws_that_may_hold_the_object(monkeypatch, why):
    r = rig(monkeypatch)
    gate = _gate_at(r, TARGET, HOVER)
    if why == "provisional hold":
        r.rt._held_provisional = ("orange", "orange", None)
    else:
        monkeypatch.setattr(carry_attachment, "active", lambda runtime: SimpleNamespace())
    n0 = len(r.calls)
    out = r.rt._air_grasp_reopen(1., gate, halt_generation=0)
    assert out.startswith("skipped: ") and "may hold the object" in out, out
    assert commands(r.calls[n0:]) == []


# ── a halt or an e-stop: 0 further commands, the jaws included ──────────────


def _halt(h):
    h._halt_generation += 1
    h.halted = "operator"


def _estop(h):
    h.estopped = True


STOPS = {"halt": (_halt, MotionHalted, "halt"), "e-stop": (_estop, SafetyViolation, "e-stop latched")}


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_after_the_lift_sends_no_reopen(monkeypatch, stop):
    r = rig(monkeypatch)
    latch = STOPS[stop][0]
    lift = r.rt._move_carry
    at = []

    def lift_then_stop(*a, **k):
        settled = lift(*a, **k)
        latch(r.harness)
        at.append(len(r.calls))
        return settled
    monkeypatch.setattr(r.rt, "_move_carry", lift_then_stop)
    res = grasp(r)
    assert res["jaws_reopen"].startswith("skipped: ") and res["exit_withdrawal"].startswith("skipped: "), res
    assert commands(r.calls[at[0]:]) == [] and fingers(r) == pytest.approx(.15 * UPPER)


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_while_the_jaws_reopen_ends_the_grasp(monkeypatch, stop):
    r = rig(monkeypatch)
    latch, error, text = STOPS[stop]
    at = []

    def on_reopen(pos):
        if pos == 1. and r.bridge.gripper_pos < 1. and not at:
            latch(r.harness)
            at.append(len(r.calls))
    r.bridge.on_gripper.append(on_reopen)
    with pytest.raises(error, match=text) as info:
        grasp(r)
    assert isinstance(info.value, MotionHalted) == (stop == "halt")
    assert commands(r.calls[at[0]:]) == [], "a command after the stop"


# ── a grasp that starts with closed empty jaws ──────────────────────────────


def test_closed_jaws_that_may_not_open_here_refuse_the_grasp(monkeypatch):
    r = rig(monkeypatch, TARGET + ((HOME + [0., .02, 0.], False),), stall_at=.35)
    r.bridge.gripper_pos = .15
    with pytest.raises(SkillError, match="the jaws are closed") as info:
        grasp(r)
    assert "reopening fingers intersects observed surface" in str(info.value)
    assert "nothing moved" in str(info.value)
    assert commands(r.calls) == []


def test_closed_jaws_that_do_not_open_refuse_the_grasp(monkeypatch):
    r = rig(monkeypatch, stall_at=.35)
    r.bridge.gripper_pos, r.bridge.stuck = .15, True
    with pytest.raises(SkillError, match="the jaws are closed and did not open"):
        grasp(r)
    assert commands(r.calls) == [("gripper", 1., .8)]


def test_open_jaws_get_no_extra_command(monkeypatch):
    r = rig(monkeypatch, stall_at=.35)
    res = grasp(r)
    assert res["held"] == "orange"
    assert len(r.gates) == 1                              # no probe: nothing to open
    assert [c for c in r.calls if c[0] == "gripper"][0] == ("gripper", 1., .8)
    first = commands(r.calls)
    assert first[0] == ("gripper", 1., .8) and first[1][0] == "joints"


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_while_closed_jaws_open_ends_the_grasp(monkeypatch, stop):
    r = rig(monkeypatch, stall_at=.35)
    r.bridge.gripper_pos = .15
    latch, error, _text = STOPS[stop]
    r.bridge.on_gripper.append(lambda pos: latch(r.harness))
    text = "e-stop latched before the closed jaws were opened" if stop == "e-stop" else "halt received"
    with pytest.raises(error, match=text):
        grasp(r)
    assert commands(r.calls) == [("gripper", 1., .8)] and len(r.gates) == 1   # no selection


def test_slow_jaws_count_as_reopened_only_inside_the_checked_interval(monkeypatch):
    """Jaws that open over several reads: at 97 % and 99 % of the stroke the
    fingers are still outside the opening this grasp was checked with, so
    the exit waits for them (it would be refused there)."""
    r = rig(monkeypatch)
    r.bridge.slow = True
    res = grasp(r)
    assert res["jaws_reopen"] == "reopened" and res["exit_withdrawal"] == "completed (+3 cm)", res
    assert fingers(r) == UPPER and r.bridge.opening == []


def test_jaws_that_stop_short_of_the_checked_opening_are_not_reopened(monkeypatch):
    """Jaws that reopen to 97 % of the stroke and stay there: wide by the
    jaw-width rule (air-grasp tolerance), still outside the opening this
    grasp was checked with, so the reopen is refused and nothing moves."""
    r = rig(monkeypatch)
    monkeypatch.setattr(runtime_mod, "JAWS_OPEN_TIMEOUT_S", .3)
    r.bridge.on_gripper.append(
        lambda pos: setattr(r.bridge, "open_stall", .97) if pos == pytest.approx(.15) else None)
    res = grasp(r)
    assert np.all(.97 * UPPER < r.gates[-1].lower), r.gates[-1].lower       # premise: outside the interval
    assert res["jaws_reopen"] == "refused: the jaws did not open within 0.3 s", res
    assert res["exit_withdrawal"] == "skipped: the jaws did not reopen"
    assert fingers(r) == pytest.approx(.97 * UPPER)
    np.testing.assert_allclose(r.bridge.q, HOVER, atol=1e-9)


def test_finger_feedback_outside_the_closing_interval_refuses_the_reopen(monkeypatch):
    r = rig(monkeypatch)
    gate = _gate_at(r, TARGET, HOVER)
    r.bridge.gripper_pos = 1.2          # beyond the calibrated stroke: feedback nobody checked
    n0 = len(r.calls)
    out = r.rt._air_grasp_reopen(1., gate, halt_generation=0)
    assert out == "refused: finger feedback left the geometrically checked closing interval", out
    assert commands(r.calls[n0:]) == []


def test_reopened_checks_the_measured_fingers_against_the_scene(monkeypatch):
    r = rig(monkeypatch)
    # inside the OPEN right finger with the TCP at `there` (2 cm past the
    # grasp centre), far below the fingers at the hover
    there = CENTRE + [.02, 0., 0.]
    beside = there + [-.02, .065, 0.]
    r.bridge.q = HOVER.copy()
    gate = ObservedFingerGate(scene(TARGET + ((beside, False),)), GEOMETRY, r.rt.kin, r.rt.arm.get_state(),
                              uncertainty_m=OPEN_TRACKING_ENVELOPE_M)
    assert gate.reopened(r.rt.arm.get_state()) is True                             # premise: free at the hover
    r.bridge.q = there.copy()
    with pytest.raises(SafetyViolation, match="measured finger intersects observed surface"):
        gate.reopened(r.rt.arm.get_state())


def test_a_halt_the_jaw_command_is_refused_for_is_raised_not_reported(monkeypatch):
    """SafeArm refuses the jaws for a halt the grasp's own baseline does not
    show (the skill's older baseline, the halt resumed since): it stays the
    operator's MotionHalted, never a refusal string."""
    r = rig(monkeypatch)
    gate = _gate_at(r, TARGET, HOVER)
    real = r.harness._check_halt_generation

    def skill_baseline_halted(expected):
        real(expected)
        raise MotionHalted("halt received during route planning or execution")
    monkeypatch.setattr(r.harness, "_check_halt_generation", skill_baseline_halted)
    n0 = len(r.calls)
    with pytest.raises(MotionHalted):
        r.rt._air_grasp_reopen(1., gate, halt_generation=0)
    assert commands(r.calls[n0:]) == []


def test_the_wait_for_the_jaws_is_bounded_in_time(monkeypatch):
    r = rig(monkeypatch)
    gate = _gate_at(r, TARGET, HOVER)
    r.bridge.stuck = True
    t0 = runtime_mod.time.monotonic()
    out = r.rt._air_grasp_reopen(1., gate, halt_generation=0)
    timeout = runtime_mod.JAWS_OPEN_TIMEOUT_S
    assert out == f"refused: the jaws did not open within {timeout:g} s"
    elapsed = runtime_mod.time.monotonic() - t0
    assert timeout <= elapsed <= timeout + runtime_mod.JAWS_OPEN_POLL_S + .1, elapsed


def test_a_grasp_waits_for_slow_closed_jaws_to_open_before_selection(monkeypatch):
    r = rig(monkeypatch, stall_at=.35)
    r.bridge.gripper_pos, r.bridge.slow = .15, True
    res = grasp(r)
    assert res["held"] == "orange", res
    # 97 % of the stroke is not open yet: selection saw the fingers at >= 99 %
    assert np.all(r.gates[1].lower >= .99 * UPPER - 2 * OPEN_TRACKING_ENVELOPE_M), r.gates[1].lower


#: Newton's measured opening (e2e-newton oc/halt joint_samples): +0.0405
#: gripper_pos per ~0.07 s wall. The rig's fake clock advances 0.02 s per
#: read and 0.05 s per poll sleep, so one step per read opens the jaws from
#: 0.15 in ~1.5 s, as on Newton.
NEWTON_OPENING = [min(1., .15 + .0405 * (i + 1)) for i in range(22)]


@pytest.mark.parametrize("jaws", ["open", "closed"])
def test_the_opening_before_selection_leaves_the_route_budget_whole(monkeypatch, jaws):
    """Round 2, I1 (review probe P1): closed empty jaws open on Newton in
    ~1.5 s, and a kitchen selection with drops needs ~1.6 s of its 3 s route
    budget (`PLAN_BUDGET_S`). The opening is no selection work: the grasp
    is held, as with open jaws (RED: "route preflight exceeded its time
    budget; no route authorized", the Newton symptom this round cures)."""
    r = rig(monkeypatch, stall_at=.35)
    clock = runtime_mod.time.monotonic
    opened_at, selecting_at = [], []
    if jaws == "closed":
        r.bridge.gripper_pos = .15
        newton = r.bridge.gripper

        def gripper(pos, effort):
            if pos == 1. and r.bridge.gripper_pos < .5:   # the closed jaws open like Newton's
                r.calls.append(("gripper", pos, effort))
                opened_at.append(clock())
                r.bridge.opening = list(NEWTON_OPENING)
                return
            return newton(pos, effort)
        r.bridge.gripper = gripper
    real = ObservedFingerGate.closing_pose

    def selection_work(self, q):   # the selection's first candidate check costs 1.6 s
        if not selecting_at:
            selecting_at.append(clock())
            runtime_mod.time.sleep(1.6)
        return real(self, q)
    monkeypatch.setattr(ObservedFingerGate, "closing_pose", selection_work)
    res = grasp(r)
    assert res["held"] == "orange", res
    if jaws == "closed":
        # premise: the opening alone takes ~1.5 s, with the selection over the budget
        from cascade.safety.trajectory import PLAN_BUDGET_S
        assert selecting_at[0] - opened_at[0] >= 1.4, selecting_at[0] - opened_at[0]
        assert selecting_at[0] - opened_at[0] + 1.6 > PLAN_BUDGET_S
        assert np.all(r.gates[1].lower >= UPPER - 2 * OPEN_TRACKING_ENVELOPE_M)   # selection saw them open


def test_unknown_jaws_are_left_as_they_are_at_the_grasp_start(monkeypatch):
    r = rig(monkeypatch, stall_at=.35)
    read = r.bridge.state

    def no_gripper_pos(*, timeout_s=None):
        sample = read(timeout_s=timeout_s)
        sample.pop("gripper_pos")
        return sample
    r.bridge.state = no_gripper_pos
    r.rt.grasp_memory = SimpleNamespace(record=lambda *a, **k: None)
    with pytest.raises(Exception):   # the jaw check cannot verify a grip without feedback
        grasp(r)
    assert len(r.gates) == 1          # no probe gate, no opening before selection


def test_a_grasp_never_opens_closed_jaws_that_may_hold_the_object(monkeypatch):
    r = rig(monkeypatch, stall_at=.35)
    r.bridge.gripper_pos = .15
    monkeypatch.setattr(carry_attachment, "active", lambda runtime: SimpleNamespace())
    monkeypatch.setattr(carry_attachment, "check", lambda runtime: None)
    with pytest.raises(SkillError, match="no executable grasp"):
        grasp(r)
    assert commands(r.calls) == [] and len(r.gates) == 1


@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_while_the_opening_is_vetted_stays_a_stop(monkeypatch, stop):
    r = rig(monkeypatch, stall_at=.35)
    r.bridge.gripper_pos = .15
    latch, error, text = STOPS[stop]
    real = ObservedFingerGate.occluded_pose

    def stop_then_vet(self, q, *, closing=False, check=None):
        latch(r.harness)
        return real(self, q, closing=closing, check=check)
    monkeypatch.setattr(ObservedFingerGate, "occluded_pose", stop_then_vet)
    with pytest.raises(error, match=text) as info:
        grasp(r)
    assert not isinstance(info.value, SkillError)
    assert commands(r.calls) == []


def test_the_evidence_receipt_records_the_jaw_phases(monkeypatch, tmp_path, synthetic_git_metadata):  # noqa: F811
    """Grasp evidence: the opening before selection is its own phase and
    selection is still recorded under "planning"; the air grasp's reopen is
    recorded with its outcome before the exit withdrawal."""
    monkeypatch.setenv("CASCADE_GRASP_EVIDENCE_DIR", str(tmp_path))
    r = rig(monkeypatch)
    r.bridge.gripper_pos = .15
    res = grasp(r)
    assert res["jaws_reopen"] == "reopened", res
    events = receipt(tmp_path)["events"]
    phases = [e["phase"] for e in events]
    assert "jaws_open" in phases and "jaws_reopen" in phases
    selection = next(e for e in events if e["kind"] == "selection_input")
    assert selection["phase"] == "planning"
    kinds = [e["kind"] for e in events]
    reopen = kinds.index("jaws_reopen")
    assert events[reopen]["data"]["outcome"] == "reopened"
    assert reopen < kinds.index("exit_withdrawal")


# ── mapping-only path (mock stack): reopen, withdraw, reset_scene ───────────


@pytest.fixture
def mock_rt(tmp_path, monkeypatch):
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from owned_server import held_dead_port

    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
        runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
        arm.object_stop_frac = None  # every close is an air grasp
        runtime.cfg.grasp._data["open_wait_timeout_s"] = 0.3  # a stuck jaw times out quickly
        grips = []
        real = arm.set_gripper

        def set_gripper(pos, effort=1.0):
            grips.append(float(pos))
            if not getattr(arm, "jaws_stuck", False):
                real(pos, effort)
        arm.set_gripper = set_gripper
        try:
            yield SimpleNamespace(rt=runtime, arm=arm, grips=grips)
        finally:
            shutdown_runtime(runtime, arm)


@needs_pin
def test_mapping_only_air_grasp_reopens_then_withdraws(mock_rt):
    rt = mock_rt.rt
    res = rt.skill_grasp_object("red cube")
    assert res["error"] == AIR and res["jaws_reopen"] == "reopened", res
    assert res["exit_withdrawal"] == "completed (+3 cm)"
    assert mock_rt.grips[-1] == rt._grip_open and rt._gripper_width_frac() == 1.


@needs_pin
def test_mapping_only_jaws_that_do_not_reopen_leave_the_arm_in_place(mock_rt, monkeypatch):
    rt, arm = mock_rt.rt, mock_rt.arm
    close = rt._close_two_stage

    def close_then_stick(*a, **k):
        close(*a, **k)
        arm.jaws_stuck = True
    monkeypatch.setattr(rt, "_close_two_stage", close_then_stick)
    n0 = [None]
    exit_ = rt._air_grasp_exit
    monkeypatch.setattr(rt, "_air_grasp_exit", lambda *a, **k: n0.__setitem__(0, "ran") or exit_(*a, **k))
    res = rt.skill_grasp_object("red cube")
    assert res["jaws_reopen"].startswith("refused: the jaws did not open"), res
    assert res["exit_withdrawal"] == "skipped: the jaws did not reopen" and n0 == [None]


@needs_pin
def test_reset_scene_opens_closed_empty_jaws(mock_rt):
    rt = mock_rt.rt
    rt.execute("close_gripper", {})
    assert rt._gripper_width_frac() < .5
    n0 = len(mock_rt.grips)
    out = rt.execute("reset_scene", {})
    assert out["ok"] is True and out["jaws_open_commanded"] is True, out
    assert mock_rt.grips[n0:] == [rt._grip_open] and rt._gripper_width_frac() == 1.


@needs_pin
def test_reset_scene_with_open_jaws_sends_no_jaw_command(mock_rt):
    rt = mock_rt.rt
    n0 = len(mock_rt.grips)
    out = rt.execute("reset_scene", {})
    assert out["ok"] is True and "jaws_open_commanded" not in out, out
    assert mock_rt.grips[n0:] == []


@needs_pin
def test_reset_scene_never_opens_held_jaws_before_its_release(mock_rt):
    rt, arm = mock_rt.rt, mock_rt.arm
    arm.object_stop_frac = .5                 # the jaws stall on something: a plausible hold
    rt.execute("close_gripper", {})
    rt.held_object = "red cube"
    n0 = len(mock_rt.grips)
    out = rt.execute("reset_scene", {})
    assert "jaws_open_commanded" not in out, out
    assert mock_rt.grips[n0:] == [1.0]          # only reset's own release, as before


@needs_pin
@pytest.mark.parametrize("call", ["execute", "direct"])
@pytest.mark.parametrize("stop", ["halt", "halt resumed", "e-stop"])
def test_a_stop_before_reset_opens_the_jaws_sends_nothing(mock_rt, monkeypatch, stop, call):
    """A halt (pending, or already resumed: the reset's halt generation
    moved) or an e-stop after reset's home move: no jaw command."""
    rt = mock_rt.rt
    rt.execute("close_gripper", {})
    home = rt.skill_move_home
    harness = rt.arm.harness

    def home_then_stop(*a, **k):
        out = home(*a, **k)
        if stop == "e-stop":
            harness.estop("operator")
        else:
            harness.halt("operator")
            if stop == "halt resumed":
                harness.clear_halt()          # the operator's Resume before the jaws
        return out
    monkeypatch.setattr(rt, "skill_move_home", home_then_stop)
    n0 = len(mock_rt.grips)
    out = rt.execute("reset_scene", {}) if call == "execute" else rt.skill_reset_scene()
    assert out["ok"] is False and out.get("stage") == "jaws", out
    assert mock_rt.grips[n0:] == [] and rt._gripper_width_frac() < .5
    if stop == "e-stop":
        harness.reset_estop()
    elif stop == "halt":
        harness.clear_halt()


@needs_pin
@pytest.mark.parametrize("why", ["provisional hold", "carry attachment"])
def test_reset_scene_never_opens_jaws_that_may_hold_an_object(mock_rt, monkeypatch, why):
    rt = mock_rt.rt
    rt.execute("close_gripper", {})
    if why == "provisional hold":
        rt._held_provisional = ("red cube", "cube", "red")
    else:
        attachment = [SimpleNamespace()]
        monkeypatch.setattr(carry_attachment, "active", lambda runtime: attachment[0])
        monkeypatch.setattr(carry_attachment, "check", lambda runtime: None)
    n0 = len(mock_rt.grips)
    out = rt.skill_reset_scene()
    assert "jaws_open_commanded" not in out, out
    assert mock_rt.grips[n0:] == []


@needs_pin
def test_a_jaw_fault_in_reset_scene_is_reported_and_the_reset_goes_on(mock_rt, monkeypatch):
    rt, arm = mock_rt.rt, mock_rt.arm
    rt.execute("close_gripper", {})
    set_gripper = arm.set_gripper

    def faulty(pos, effort=1.0):
        set_gripper(pos, effort)          # recorded, never reaches the jaws
        raise RuntimeError("gripper bus timeout")
    monkeypatch.setattr(arm, "jaws_stuck", True, raising=False)
    monkeypatch.setattr(arm, "set_gripper", faulty)
    out = rt.skill_reset_scene()
    assert out["ok"] is True and out["jaws_open_error"] == "gripper bus timeout", out
    assert "jaws_open_commanded" not in out and out["observation_refreshed"] is True


@needs_pin
def test_reset_scene_with_unreadable_jaws_leaves_them_and_resets(mock_rt, monkeypatch):
    rt = mock_rt.rt
    rt.execute("close_gripper", {})
    width, reads = rt._gripper_width_frac, []

    def unreadable_once():   # the jaw check's read fails; reset's later reads succeed
        reads.append(1)
        if len(reads) == 1:
            raise RuntimeError("gripper feedback timed out")
        return width()
    monkeypatch.setattr(rt, "_gripper_width_frac", unreadable_once)
    n0 = len(mock_rt.grips)
    out = rt.skill_reset_scene()
    assert out["ok"] is True and "jaws_open_commanded" not in out, out
    assert mock_rt.grips[n0:] == []


# ── what the agent reads after an air grasp ────────────────────────────────

#: the step after an air grasp whose jaws could not be reopened
JAWS_CLOSED = "the jaws are still closed"


def test_after_a_completed_exit_the_step_says_the_jaws_were_reopened():
    out = runtime_mod._with_move_home_first(
        {"ok": False, "error": AIR, "jaws_reopen": "reopened", "exit_withdrawal": "completed (+3 cm)"})
    action = out["next_action"]
    assert "the jaws were reopened and the arm withdrew above the object" in action, action
    assert action.count(MOVE_HOME_FIRST) == 1 and out["arm_withdrew_above_object"] is True
    assert "air_grasp_jaws_closed" not in out


def test_after_a_refused_reopen_the_step_says_the_jaws_are_closed_and_the_arm_stayed():
    out = runtime_mod._with_move_home_first(
        {"ok": False, "error": f"{AIR}; jaws not reopened: x", "jaws_reopen": "refused: x",
         "exit_withdrawal": "skipped: the jaws did not reopen"})
    action = out["next_action"]
    assert JAWS_CLOSED in action and "left at the grasp hover" in action, action
    assert action.count(MOVE_HOME_FIRST) == 1 and "opens empty closed jaws first" in action
    assert out["air_grasp_jaws_closed"] is True and "arm_withdrew_above_object" not in out
    # a composite that built its result from that grasp says the same
    composite = runtime_mod._with_move_home_first({"ok": False}, source=out)
    assert composite["next_action"] == action and composite["air_grasp_jaws_closed"] is True


def _jaws_closed_air_grasp(move):
    """A grasp double: the arm moved (`move()` counts it), the jaws closed on
    air and could not be reopened, so the exit withdrawal was skipped."""
    def grasp(*_a, **_k):
        move()
        return {"ok": False, "error": f"{AIR}; jaws not reopened: test refusal",
                "jaws_reopen": "refused: test refusal", "exit_withdrawal": "skipped: the jaws did not reopen"}
    return grasp


def _says_the_jaws_stayed_closed(result):
    action = _failed_once(result)
    assert JAWS_CLOSED in action and action.count(MOVE_HOME_FIRST) == 1, result
    assert result["air_grasp_jaws_closed"] is True and result["arm_left_in_place"] is True, result
    assert "arm_withdrew_above_object" not in result


@pytest.mark.parametrize("mock_runtime", ["gate on"], indirect=True)
@pytest.mark.parametrize("skill", sorted(COMPOSITES) + ["grasp_object"])
def test_every_grasp_entry_says_the_jaws_stayed_closed(mock_runtime, monkeypatch, skill):  # noqa: F811
    runtime, _arm = mock_runtime
    _seen(runtime)
    runtime.cfg._data["grasp"]["max_pick_attempts"] = 1
    runtime.skill_grasp_object = _jaws_closed_air_grasp(
        lambda: runtime._move_joints(_nudged(runtime), duration_s=0.3))
    if skill == "sort_by_color":
        monkeypatch.setattr(runtime.beliefs, "all", lambda: [_red()])
    _says_the_jaws_stayed_closed(runtime.execute(skill, COMPOSITES.get(skill, ARGS.get(skill))))


@pytest.mark.parametrize("composite", ["restore_scene", "search_for_object"])
def test_restore_and_search_say_the_jaws_stayed_closed(monkeypatch, composite):
    import time

    from test_agent_results_compose import _stub_with_real_persistence
    from test_pigey_scene_memory import A, BOUNDS, _cluttered, _stub, _two_cubes

    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    if composite == "restore_scene":
        bs = _two_cubes(time.monotonic())
        rt = _stub(bs, bounds=BOUNDS)
        rt.skill_snapshot_scene(name="t")
        bs.find("red cube").position = np.array(A) + np.array([0.0, 0.08, 0.0])
    else:
        rt = _stub(_cluttered(time.monotonic()), bounds=BOUNDS)
    _stub_with_real_persistence(rt, "moved")
    motions = [0]
    rt._actuation_counts = lambda: {"arm_motions": motions[0], "gripper_closes": 0, "gripper_opens": 0}
    rt.skill_grasp_object = _jaws_closed_air_grasp(lambda: motions.__setitem__(0, motions[0] + 1))
    out = (rt.skill_restore_scene(name="t") if composite == "restore_scene"
           else rt.skill_search_for_object(label="green cube", max_occluders=1))
    assert out["ok"] is False, out
    assert JAWS_CLOSED in out["next_action"] and out["next_action"].count(MOVE_HOME_FIRST) == 1, out
    assert out["air_grasp_jaws_closed"] is True and "arm_withdrew_above_object" not in out
