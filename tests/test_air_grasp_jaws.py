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
#: what a reopen that was refused or not confirmed says (round 2, M2): a jaw
#: command may have moved the jaws, so never "did not open"
NOT_OPEN = "the jaws were not confirmed open"
#: the exit withdrawal after such a reopen
SKIPPED = f"skipped: {NOT_OPEN}"


def unconfirmed(timeout):
    """Why a jaw opening was not confirmed within `timeout` s."""
    return f"no feedback confirmed the opening within {timeout:g} s"


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
    assert res["jaws_reopen"] == f"refused: {unconfirmed(timeout)}", res
    assert res["exit_withdrawal"] == SKIPPED
    assert res["error"] == f"{AIR}; {NOT_OPEN}: {unconfirmed(timeout)}"
    reopen = len(r.calls) - r.calls[::-1].index(("gripper", 1., .8)) - 1
    assert commands(r.calls[reopen + 1:]) == [], "the arm moved with jaws that were not confirmed open"
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


def _jaw_fault_on_reopen(r, where, *, latch=None):
    """The reopen after the close meets a jaw fault (not a SafetyViolation):
    the bridge refuses the command, or the first feedback read after it
    fails. `latch` (a stop) lands with the fault. Returns the call index of
    the fault (a list, filled when it happens)."""
    at = []

    def fault():
        at.append(len(r.calls))
        if latch is not None:
            latch(r.harness)
        raise RuntimeError("gripper bus timeout")

    if where == "command":
        def on_reopen(pos):
            if pos == 1. and r.bridge.gripper_pos < 1. and not at:
                fault()
        r.bridge.on_gripper.append(on_reopen)
    else:
        armed, read = [], r.bridge.state
        r.bridge.on_gripper.append(
            lambda pos: armed.append(1) if pos == 1. and r.bridge.gripper_pos < 1. else None)

        def state(*, timeout_s=None):
            if armed and not at:
                fault()
            return read(timeout_s=timeout_s)
        r.bridge.state = state
    return at


@pytest.mark.parametrize("where", ["command", "feedback"])
def test_a_jaw_fault_while_the_jaws_reopen_is_a_reopen_that_was_not_confirmed(monkeypatch, where):
    """Round 2, M1: a jaw fault that is not a SafetyViolation (a bus timeout)
    keeps the air-grasp result: `jaws_reopen` says the jaw command failed, the
    exit is skipped, the error is the air grasp's, and the arm stays at the
    hover (as reset_scene reports its jaw fault and goes on)."""
    r = rig(monkeypatch)
    at = _jaw_fault_on_reopen(r, where)
    res = grasp(r)
    assert res["jaws_reopen"] == "refused: jaw command failed: gripper bus timeout", res
    assert res["exit_withdrawal"] == SKIPPED
    assert res["error"] == f"{AIR}; {NOT_OPEN}: jaw command failed: gripper bus timeout"
    assert at and commands(r.calls[at[0]:]) == [], "a command after the jaw fault"
    np.testing.assert_allclose(r.bridge.q, HOVER, atol=1e-9)


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


@pytest.mark.parametrize("where", ["command", "feedback"])
@pytest.mark.parametrize("stop", sorted(STOPS))
def test_a_stop_that_lands_with_a_jaw_fault_stays_a_stop(monkeypatch, stop, where):
    r = rig(monkeypatch)
    latch, error, text = STOPS[stop]
    at = _jaw_fault_on_reopen(r, where, latch=latch)
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
    timeout = runtime_mod.JAWS_OPEN_TIMEOUT_S
    with pytest.raises(SkillError) as info:
        grasp(r)
    assert str(info.value) == (f"the jaws were closed and were not confirmed open before this grasp: "
                               f"{unconfirmed(timeout)}")
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
    assert res["jaws_reopen"] == f"refused: {unconfirmed(.3)}", res
    assert res["exit_withdrawal"] == SKIPPED
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
    assert out == f"refused: {unconfirmed(timeout)}"
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


#: the kitchen profile that plans with GraspGen-X on the GPU service under a
#: per-attempt planning budget (`graspgenx.feasibility_timeout_s`)
KITCHEN_GPU = "isaac_kitchen_gpu"
#: a normal GraspGen-X planning batch, well inside that budget
PLANNING_S = 2.


def _kitchen_gpu(r):
    """The rig with the kitchen GPU profile's GraspGen-X planning budget and
    Isaac's jaw wait (both read from the profile, not restated here)."""
    from cascade.config import load_demo_config

    kitchen = load_demo_config(arm=KITCHEN_GPU, camera="mock", llm="mock").grasp
    search = kitchen.get("graspgenx")
    r.rt.cfg._data["grasp"].update(
        backend="graspgenx", open_wait_timeout_s=kitchen.get("open_wait_timeout_s"),
        graspgenx={key: search.get(key) for key in ("feasibility_max_batches", "feasibility_timeout_s")})
    r.rt.grasp_memory = SimpleNamespace(prior=lambda *a: None, record=lambda *a, **k: None)
    return r.rt.cfg._data["grasp"]


@pytest.mark.parametrize("jaws", ["open", "closed"])
def test_the_opening_before_planning_leaves_the_kitchen_planning_budget_whole(monkeypatch, jaws):
    """Round 2b (controller ruling): closed empty jaws open before every
    per-attempt budget clock starts, the GraspGen-X planning budget of the
    kitchen GPU profile (8 s) included. Worst case on Isaac: the jaws confirm
    open just inside the 12 s jaw wait; then a normal planning batch: the
    grasp is held, as with open jaws (RED: "grasp planning budget exhausted
    before actuation", after the jaws had moved)."""
    r = rig(monkeypatch, stall_at=.35)
    grasp_cfg = _kitchen_gpu(r)
    budget, wait = grasp_cfg["graspgenx"]["feasibility_timeout_s"], grasp_cfg["open_wait_timeout_s"]
    assert wait > budget > PLANNING_S, (wait, budget)              # premise: the profile's values
    clock = runtime_mod.time.monotonic
    commanded_at, opened_at, planned_at = [], [], []
    if jaws == "closed":
        r.bridge.gripper_pos = .15
        newton, read = r.bridge.gripper, r.bridge.state
        moving = []

        def gripper(pos, effort):
            if pos == 1. and r.bridge.gripper_pos < .5:   # the slowest opening the jaw wait accepts
                r.calls.append(("gripper", pos, effort))
                commanded_at.append(clock())
                moving.append(True)
                return
            return newton(pos, effort)

        def state(*, timeout_s=None):   # closed -> open over (wait - 0.1) s
            if moving:
                r.bridge.gripper_pos = min(1., .15 + .85 * (clock() - commanded_at[0]) / (wait - .1))
                if r.bridge.gripper_pos == 1.:
                    moving.clear()
            return read(timeout_s=timeout_s)
        r.bridge.gripper, r.bridge.state = gripper, state
        confirms = ObservedFingerGate.fingers_open

        def confirmed(self, measured):   # when the grasp's wait confirms the jaws open
            verdict = confirms(self, measured)
            if verdict and not opened_at:
                opened_at.append(clock())
            return verdict
        monkeypatch.setattr(ObservedFingerGate, "fingers_open", confirmed)
    plan = r.rt._plan_grasps

    def graspgenx(*a, **k):
        planned_at.append(clock())
        runtime_mod.time.sleep(PLANNING_S)
        return plan(*a, **k)
    r.rt._plan_grasps = graspgenx
    res = grasp(r)
    assert res["held"] == "orange", res
    assert len(planned_at) == 1                                    # one batch: no replanning
    if jaws == "closed":
        # premise: the opening, in the jaw wait's last 0.5 s, alone outlasts the planning budget
        assert wait > opened_at[0] - commanded_at[0] >= wait - .5 > budget, opened_at[0] - commanded_at[0]
        assert planned_at[0] >= opened_at[0]                       # planning after the opening
        assert np.all(r.gates[1].lower >= UPPER - 2 * OPEN_TRACKING_ENVELOPE_M)   # selection saw them open


@pytest.mark.parametrize("key,bad", [("feasibility_max_batches", 0), ("feasibility_timeout_s", float("nan"))])
def test_an_invalid_planning_budget_refuses_the_grasp_before_closed_jaws_open(monkeypatch, key, bad):
    """Round 2b: the planning budget starts after the opening, but its
    settings are still checked before anything moves."""
    r = rig(monkeypatch, stall_at=.35)
    r.bridge.gripper_pos = .15
    _kitchen_gpu(r)["graspgenx"][key] = bad
    with pytest.raises(SkillError, match=f"^grasp {key} must be"):
        grasp(r)
    assert commands(r.calls) == []


def test_unknown_jaws_are_left_as_they_are_at_the_grasp_start(monkeypatch):
    r = rig(monkeypatch, stall_at=.35)
    read = r.bridge.state

    def no_gripper_pos(*, timeout_s=None):
        sample = read(timeout_s=timeout_s)
        sample.pop("gripper_pos")
        return sample
    r.bridge.state = no_gripper_pos
    r.rt.grasp_memory = SimpleNamespace(record=lambda *a, **k: None)
    # the close cannot settle without jaw feedback (round 2, M5: this error only)
    with pytest.raises(SkillError, match="^gripper closure did not settle; refusing to lift$"):
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
    assert res["jaws_reopen"] == f"refused: {unconfirmed(.3)}", res
    assert res["exit_withdrawal"] == SKIPPED and n0 == [None]
    assert res["error"] == f"{AIR}; {NOT_OPEN}: {unconfirmed(.3)}"


@needs_pin
def test_a_jaw_fault_while_the_jaws_reopen_keeps_the_air_grasp_result(mock_rt):
    """Round 2, M1 (review probe P6), through execute(): the result keeps the
    air grasp's context and the move_home step says the jaws were not
    confirmed open, instead of "unexpected RuntimeError"."""
    rt, arm = mock_rt.rt, mock_rt.arm
    send, closed = arm.set_gripper, []

    def faulty(pos, effort=1.0):
        if float(pos) == rt._grip_open and closed:   # the reopen after the close
            raise RuntimeError("gripper bus timeout")
        send(pos, effort)
        if float(pos) != rt._grip_open:
            closed.append(float(pos))
    arm.set_gripper = faulty
    res = rt.execute("grasp_object", {"label": "red cube"})
    assert res["jaws_reopen"] == "refused: jaw command failed: gripper bus timeout", res
    assert res["exit_withdrawal"] == SKIPPED
    assert res["error"] == f"{AIR}; {NOT_OPEN}: jaw command failed: gripper bus timeout"
    _says_the_jaws_stayed_closed(res)


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

#: the step after an air grasp whose jaws were not confirmed open (round 2,
#: M2: a jaw command may have moved them, so never "still closed")
JAWS_CLOSED = ("the jaws were not confirmed open afterwards: they may still be closed (or only partly "
               "open), and the arm was left at the grasp hover")


def test_after_a_completed_exit_the_step_says_the_jaws_were_reopened():
    out = runtime_mod._with_move_home_first(
        {"ok": False, "error": AIR, "jaws_reopen": "reopened", "exit_withdrawal": "completed (+3 cm)"})
    action = out["next_action"]
    assert "the jaws were reopened and the arm withdrew above the object" in action, action
    assert action.count(MOVE_HOME_FIRST) == 1 and out["arm_withdrew_above_object"] is True
    assert "air_grasp_jaws_closed" not in out


def test_after_a_refused_reopen_the_step_says_the_jaws_are_closed_and_the_arm_stayed():
    out = runtime_mod._with_move_home_first(
        {"ok": False, "error": f"{AIR}; {NOT_OPEN}: x", "jaws_reopen": "refused: x",
         "exit_withdrawal": SKIPPED})
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
        return {"ok": False, "error": f"{AIR}; {NOT_OPEN}: test refusal",
                "jaws_reopen": "refused: test refusal", "exit_withdrawal": SKIPPED}
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


# ── the jaw wait: Newton-sized on Isaac, validated before anything moves ────


def _isaac_arm_profiles():
    from cascade.config import CONFIG_DIR, load_profile

    return sorted(p.stem for p in (CONFIG_DIR / "arms").glob("*.yaml")
                  if load_profile("arms", p.stem).get("type") == "isaac")


ISAAC_ARMS = _isaac_arm_profiles()
#: Newton runs arm and jaw motion about 4x slower than real time
NEWTON_SLOWDOWN = 4


def test_every_isaac_arm_profile_is_covered():
    assert {"isaac", "isaac_kitchen", "isaac_kitchen_gpu"} <= set(ISAAC_ARMS), ISAAC_ARMS


@pytest.mark.parametrize("arm", ISAAC_ARMS)
def test_isaac_profiles_size_the_jaw_wait_for_newton(arm):
    """Round 2, M3: the empty-jaws wait is wall clock. On Newton the reopen
    took 1.1-1.5 s wall at load ~5, about half the real-time bound, so a
    loaded host timed it out while the jaws were still opening."""
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm=arm, camera="mock", llm="mock")
    bound = runtime_mod.SkillRuntime._jaws_open_timeout_s(SimpleNamespace(cfg=cfg))
    assert bound >= NEWTON_SLOWDOWN * runtime_mod.JAWS_OPEN_TIMEOUT_S, bound


#: values `grasp.open_wait_timeout_s` refuses (None = unset is valid)
BAD_WAITS = [0, -1., float("nan"), float("inf"), True, "6"]
WAIT_REFUSAL = "open_wait_timeout_s must be finite and positive"


@pytest.mark.parametrize("bad", BAD_WAITS, ids=repr)
def test_an_invalid_jaw_wait_refuses_the_build_before_any_arm_exists(tmp_path, monkeypatch, bad):
    """Round 2, M6: checked with the other pure-config checks of the build,
    before any arm is made (an eager build connects it)."""
    from cascade.apps import demo
    from cascade.config import load_demo_config

    made = []

    def make_arm(acfg, kinematics=None):
        made.append(acfg)
        raise RuntimeError("an arm was made")
    monkeypatch.setattr(demo, "make_arm", make_arm)
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["open_wait_timeout_s"] = bad
    with pytest.raises(ValueError, match=WAIT_REFUSAL):
        demo.build_runtime(cfg, tmp_path / "run", lazy_arm=False)
    assert made == []


@pytest.mark.parametrize("bad", BAD_WAITS, ids=repr)
@pytest.mark.parametrize("close", ["held", "air"])
def test_an_invalid_jaw_wait_refuses_the_grasp_before_anything_moves(monkeypatch, bad, close):
    """Round 2, M6: not first after the close, inside the air-grasp branch
    (gate on)."""
    r = rig(monkeypatch, stall_at=.35 if close == "held" else None)
    r.rt.cfg._data["grasp"]["open_wait_timeout_s"] = bad
    with pytest.raises(SkillError, match=WAIT_REFUSAL):
        grasp(r)
    assert commands(r.calls) == []


@needs_pin
def test_an_invalid_jaw_wait_refuses_the_grasp_before_anything_moves_gate_off(mock_rt):
    rt, arm = mock_rt.rt, mock_rt.arm
    rt.cfg.grasp._data["open_wait_timeout_s"] = 0
    n0 = len(arm.commands)
    with pytest.raises(SkillError, match=WAIT_REFUSAL):
        rt.skill_grasp_object("red cube")
    assert mock_rt.grips == [] and len(arm.commands) == n0


@pytest.mark.parametrize("bad", BAD_WAITS, ids=repr)
def test_an_invalid_jaw_wait_refuses_open_gripper_before_the_jaws_move(bad):
    from test_gripper_open_close_commands import opener

    rt, sent, reads = opener([.07], timeout=bad)
    with pytest.raises(SkillError, match=WAIT_REFUSAL):
        rt.skill_open_gripper()
    assert sent == [] and reads == []
