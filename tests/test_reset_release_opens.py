"""`reset_scene` and a held object: a simulated arm releases it with the
profile's OPEN jaw position; a real arm refuses before any motion.

Found during the PR-A fix round (#354): reset_scene's release sent the raw
gripper position 1.0. Jaw positions are motor travel in each profile's own
units (`gripper.open_pos` .. `gripper.closed_pos`), so 1.0 means something
different on every arm. On the reBot RS (open 6.2, closed 0.0) it is about
16 % open: nearly CLOSED, so "release" squeezed the object instead of letting
it go. On the mock (open 0.0, closed 1.0) it is fully closed, on the SO-101
(open 0.60) past the open stop, on the FR3 (open 0.04 m) 25 times the
travel. Only Isaac (open 1.0) happened to be right.

The release happens at HOME: on a real arm a correct opening drops the object
from home height (reBot home TCP at 0.44 m). So a REAL profile (the type of
the arm the reset drives, the `arm=` selection included, is not isaac*,
mujoco* or mock*; an unknown type or arm counts as real) refuses a reset
while holding, before any motion, and asks for place_at first. Simulated and
mock arms keep releasing at home.

The reset takes its halt generation when it starts (before its carry check)
and passes it to the home move and to the opening: a halt anywhere in the
reset sends no further arm or jaw command. On fr3 (`return_home_after_motion`)
the home move is the camera-view home return, which carries it too; a halt
there stays a halt (never a latched e-stop), also one its after-move check
finds, and an e-stop there stays the e-stop (`SafetyViolation: e-stop
latched`, never `MotionHalted`). A refusal by SafeArm's own checks
(halt, e-stop, a gripper gate) is raised before any jaw command: the object
stays held. A driver `GripperFault` comes AFTER commands went out (opening
blocked part-way): the result reports the measured jaw width and keeps the
object unless that width proves the jaws open. A non-safety driver error
keeps the object too, with the error recorded. Every failed-release text
puts the outcome first and names the object once: the agent's failure line
keeps 240 characters.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every jaw command; no driver, no bus.
"""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.control.mock_arm import MockArm
from cascade.types import GripperFault, SafetyViolation

#: Simulated / mock arms and the open position their YAML declares. Literals
#: on purpose: the test pins the semantic open, independently of how the
#: runtime reads it.
SIM_PROFILES = [
    ("so101_mock", 0.60),
    ("so101_mujoco", 0.60),
    ("isaac_kitchen", 1.0),
    ("fr3_mock", 0.04),
    ("mock", 0.0),
]
#: Real grippers, one per hardware arm type: rebot_rs, rebot_rs_mb, so101
#: (Feetech), ros2 (FR3) and unitree_arm.
REAL_PROFILES = ["rebot_rs", "rebot_rs_mb", "so101", "fr3", "g1"]

REFUSAL = "holding red cube: place it first (place_at), then reset"


class _JawArm(MockArm):
    """MockArm that records every jaw command; `fault(arm, pos, effort)`
    replaces the driver's reply when set, `jaw_unknown` fails the jaw read."""

    def __init__(self, cfg, kinematics, log):
        super().__init__(cfg, kinematics)
        self.log = log
        self.fault = None
        self.jaw_unknown = False

    def set_gripper(self, pos, effort=1.0):
        self.log.jaws.append(float(pos))
        self.log.efforts.append(float(effort))
        if self.fault is not None:
            self.fault(self, pos, effort)
            return
        super().set_gripper(pos, effort)

    def get_state(self):
        state = super().get_state()
        return dataclasses.replace(state, gripper_valid=False) if self.jaw_unknown else state


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    def build(profile=None, *, arms=None):
        if arms is None:
            cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        else:
            cfg = load_demo_config(camera="mock", arms=arms, llm="mock")
            profile = "-".join(arms)
        # rebot_rs requires its live camera map (fail closed) and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design. The
        # jaw command is outside the map, so the map is off here.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        r = SimpleNamespace(cfg=cfg, jaws=[], efforts=[], arms=[])

        def make_arm(acfg, kinematics=None):
            arm = _JawArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        # Replaced BEFORE the build: no real driver is ever constructed.
        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _holding(r):
    """The jaws half closed on an object the runtime knows it holds."""
    g = r.cfg.arm.gripper
    half = g.open_pos + 0.5 * (g.closed_pos - g.open_pos)
    r.runtime.arm.set_gripper(half)
    r.runtime.held_object = "red cube"
    r.runtime._held_det_label = "cube"
    r.jaws.clear()
    r.efforts.clear()
    return half


def _arm_commands(arms) -> int:
    return sum(len(a.commands) for a in arms)


def _assert_still_held(r, out):
    assert r.runtime.held_object == "red cube" and r.runtime._held_det_label == "cube"
    assert out["beliefs_forgotten"] == 0 and not out["observation_refreshed"], out


@pytest.mark.parametrize("profile, open_pos", SIM_PROFILES, ids=[p for p, _ in SIM_PROFILES])
def test_sim_reset_releases_with_the_profiles_open_position(rig, profile, open_pos):
    r = rig(profile)
    assert r.cfg.arm.gripper.open_pos == pytest.approx(open_pos)
    _holding(r)

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"], out
    assert r.jaws == [pytest.approx(open_pos)], r.jaws  # one jaw command: the release, fully open
    assert r.efforts == [pytest.approx(0.8)], r.efforts  # the effort of every other opening
    assert r.runtime.arm.get_state().gripper_pos == pytest.approx(open_pos)
    assert r.runtime._gripper_width_frac() == pytest.approx(1.0)
    assert r.runtime.held_object is None and r.runtime._held_det_label is None


@pytest.mark.parametrize("profile", REAL_PROFILES)
def test_real_reset_while_holding_refuses_before_any_motion(rig, profile):
    r = rig(profile)
    half = _holding(r)
    arm_before = _arm_commands(r.arms)

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"] is False and out["stage"] == "release", out
    assert REFUSAL in out["error"], out
    assert r.jaws == [] and _arm_commands(r.arms) == arm_before  # no jaw, no arm command
    _assert_still_held(r, out)
    assert r.runtime.arm.get_state().gripper_pos == pytest.approx(half)


@pytest.mark.parametrize("kind", ["acme_arm", None], ids=["unknown", "missing"])
def test_an_unknown_arm_type_counts_as_a_real_gripper(rig, kind):
    r = rig("mock")
    _holding(r)
    arm = r.runtime.cfg._data["arm"]
    if kind is None:
        del arm["type"]
    else:
        arm["type"] = kind

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"] is False and REFUSAL in out["error"], out
    assert r.jaws == [] and _arm_commands(r.arms) == 0
    _assert_still_held(r, out)


@pytest.mark.parametrize("jaws", ["open", "closed"])
def test_real_reset_with_empty_jaws_still_resets(rig, jaws):
    """Empty-handed, a real arm resets: open jaws get no jaw command; closed
    empty jaws (the MockArm starts at 0.0, closed on the reBot) get exactly
    one opening to the profile's open position (merged with #354's reset
    opening of closed empty jaws)."""
    r = rig("rebot_rs")
    open_pos = r.cfg.arm.gripper.open_pos
    if jaws == "open":
        r.runtime.arm.set_gripper(open_pos)
        r.jaws.clear()

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"], out
    assert r.jaws == ([] if jaws == "open" else [pytest.approx(open_pos)]), r.jaws
    assert out["observation_refreshed"]


def _two_arms(rig, real: str):
    """Two mock SO-101s; the profile named `real` declares a real arm type."""
    r = rig(arms=["so101_left", "so101_right"])
    profiles = r.runtime.cfg._data["arms"]
    assert [p["name"] for p in profiles] == ["so101_left", "so101_right"]
    next(p for p in profiles if p["name"] == real)["type"] = "so101"
    return r


def test_reset_on_a_selected_real_arm_refuses_while_holding(rig):
    """The arm the reset drives decides: a mock primary does not let a real
    `arm=` selection drop the object from home height."""
    r = _two_arms(rig, real="so101_right")
    _holding(r)
    arm_before = _arm_commands(r.arms)

    out = r.runtime.execute("reset_scene", {"arm": "so101_right"})

    assert out["ok"] is False and out["stage"] == "release", out
    assert REFUSAL in out["error"], out
    assert r.jaws == [] and _arm_commands(r.arms) == arm_before  # no jaw, no arm command
    _assert_still_held(r, out)


def test_reset_on_a_selected_sim_arm_releases_beside_a_real_primary(rig):
    r = _two_arms(rig, real="so101_left")
    _holding(r)

    selected = r.runtime.execute("reset_scene", {"arm": "so101_right"})

    assert selected["ok"], selected
    assert r.jaws == [pytest.approx(0.60)], r.jaws  # the mock right arm opened fully
    assert r.runtime.held_object is None

    _holding(r)
    primary = r.runtime.execute("reset_scene", {})  # the real primary refuses

    assert primary["ok"] is False and REFUSAL in primary["error"], primary
    assert r.jaws == []
    _assert_still_held(r, primary)


def test_a_running_arm_the_rig_does_not_hold_counts_as_real(rig):
    r = rig(arms=["so101_left", "so101_right"])  # both mock
    runtime = r.runtime
    runtime._arm_override.arm = runtime.arm_rig.get("so101_right")
    assert not runtime._drives_real_gripper()
    runtime._arm_override.arm = SimpleNamespace(harness=None)  # no profile to read
    try:
        assert runtime._drives_real_gripper()
    finally:
        runtime._arm_override.arm = None


def _stop(runtime, kind):
    harness = runtime.arm.harness
    if kind == "halt":
        harness.halt("operator: stop")
    elif kind == "estop":
        harness.estop("operator e-stop")
    else:  # a gripper gate (a retained episode check) refuses the jaw command
        check = harness.check_contact_episode

        def refuse_gripper(*, gripper=False):
            if gripper:
                raise SafetyViolation("gripper gate refuses the opening")
            return check(gripper=gripper)

        harness.check_contact_episode = refuse_gripper


@pytest.mark.parametrize("kind", ["halt", "estop", "gate"])
@pytest.mark.parametrize("profile", ["so101_mock", "mock"])
def test_a_stop_after_home_sends_nothing_and_keeps_the_object(rig, monkeypatch, profile, kind):
    r = rig(profile)
    half = _holding(r)
    home = r.runtime.skill_move_home
    at_stop = {}

    def home_then_stop(**kwargs):
        result = home(**kwargs)
        _stop(r.runtime, kind)
        at_stop.update(arm=_arm_commands(r.arms), jaws=len(r.jaws))
        return result

    monkeypatch.setattr(r.runtime, "skill_move_home", home_then_stop)

    out = r.runtime.execute("reset_scene", {})

    assert at_stop, "reset_scene never reached its home move"
    assert len(r.jaws) == at_stop["jaws"] == 0, r.jaws      # no jaw command after the stop
    assert _arm_commands(r.arms) == at_stop["arm"]           # nor any arm command
    assert out["ok"] is False and out["stage"] == "release", out
    # SafeArm refused before the driver: nothing moved, and the text says so.
    assert "stopped before the release" in out["error"] and out["release_error"], out
    _assert_still_held(r, out)
    assert r.runtime.arm.get_state().gripper_pos == pytest.approx(half)


@pytest.mark.parametrize("profile, holding", [("rebot_rs", False), ("mock", False), ("mock", True)])
def test_a_halt_before_home_sends_no_arm_command(rig, monkeypatch, profile, holding):
    """The halt lands inside the reset's OWN carry check: the generation the
    reset carries was taken before it, so the home move refuses."""
    from cascade.skills import carry_attachment

    r = rig(profile)
    if holding:
        _holding(r)
    check = carry_attachment.check
    at_halt = {}

    def check_then_halt(runtime):
        check(runtime)
        if not at_halt and sys._getframe(1).f_code.co_name == "skill_reset_scene":
            runtime.arm.harness.halt("operator: stop")
            at_halt.update(arm=_arm_commands(r.arms), jaws=len(r.jaws))

    monkeypatch.setattr(carry_attachment, "check", check_then_halt)

    out = r.runtime.execute("reset_scene", {})

    assert at_halt, "reset_scene never ran its own carry check"
    assert _arm_commands(r.arms) == at_halt["arm"], "an arm command went out after the halt"
    assert len(r.jaws) == at_halt["jaws"] == 0
    assert out["ok"] is False and out["stage"] == "home", out
    assert out["home_error"] == "halt received during route planning or execution", out
    if holding:
        _assert_still_held(r, out)


def _fr3_off_home(r):
    """fr3 (`return_home_after_motion: true`, the only shipped profile with
    it): the reset's home move is the profile's camera-view home return. The
    MockArm starts AT home, where that return sends nothing: move off first."""
    runtime = r.runtime
    assert runtime._home_policy_enabled() and runtime._drives_real_gripper()
    home = np.asarray(runtime.cfg.arm.home_q, dtype=float)
    off = home.copy()
    off[0] += 0.15
    assert runtime.arm.move_joints(off, duration_s=0.5)
    assert _arm_commands(r.arms) > 0


@pytest.mark.parametrize("where", ["status", "carry_check"])
def test_fr3_view_home_return_ends_on_a_halt_raised_inside_it(rig, monkeypatch, where):
    """The halt lands INSIDE the view-home return, after its own "halted"
    flag checks (the status line, or its carry check): the move carries the
    reset's generation, so it refuses. No arm command after the halt, and the
    halt stays a halt: no e-stop latched by the return's cleanup."""
    from cascade.skills import carry_attachment

    r = rig("fr3")
    _fr3_off_home(r)
    runtime = r.runtime
    at_halt = {}

    def halt():
        if not at_halt:
            runtime.arm.harness.halt("operator: stop")
            at_halt.update(arm=_arm_commands(r.arms))

    if where == "status":
        show = runtime._show_status

        def show_then_halt(text):
            show(text)
            if text.startswith("Returning to the saved camera-view home"):
                halt()

        monkeypatch.setattr(runtime, "_show_status", show_then_halt)
    else:
        check = carry_attachment.check

        def check_then_halt(rt):
            check(rt)
            if sys._getframe(1).f_code.co_name == "_return_to_view_home":
                halt()

        monkeypatch.setattr(carry_attachment, "check", check_then_halt)

    out = runtime.execute("reset_scene", {})

    assert at_halt, "the reset never reached the view-home return"
    assert _arm_commands(r.arms) == at_halt["arm"], "an arm command went out after the halt"
    assert r.jaws == []
    assert out["ok"] is False and out["stage"] == "home", out
    assert out["home_error"] == "halt received during route planning or execution", out
    assert not runtime.arm.harness.estopped, "the halt was escalated to an e-stop"


def test_fr3_move_home_halted_mid_route_stays_a_halt(rig):
    """A halt during the view-home route stops it at the next waypoint and is
    reported as the halt it is (MotionHalted), never latched as an e-stop."""
    r = rig("fr3")
    _fr3_off_home(r)
    runtime = r.runtime
    arm = r.arms[0]
    send = arm.send_joint_target
    at_halt = {}

    def send_then_halt(q):
        send(q)
        if not at_halt and len(arm.commands) >= at_start + 5:
            runtime.arm.harness.halt("operator: stop")
            at_halt.update(arm=len(arm.commands))

    at_start = len(arm.commands)
    arm.send_joint_target = send_then_halt

    out = runtime.execute("move_home", {})

    assert at_halt, "the view-home route never streamed"
    assert len(arm.commands) == at_halt["arm"], "an arm command went out after the halt"
    assert out["ok"] is False, out
    assert out["error"] == "MotionHalted: halted: operator: stop", out
    assert not runtime.arm.harness.estopped, "the halt was escalated to an e-stop"


def _stop_at_waypoint(r, runtime, kind, waypoint=5):
    """Stop (`kind`) at the `waypoint`-th arm command from now on."""
    arm = r.arms[0]
    send = arm.send_joint_target
    start = len(arm.commands)
    at_stop = {}

    def send_then_stop(q):
        send(q)
        if not at_stop and len(arm.commands) >= start + waypoint:
            if kind == "estop_harness":
                runtime.arm.harness.estop("operator e-stop")
            else:  # SafeArm.stop(): latches the e-stop and stops the driver
                runtime.arm.stop()
            at_stop.update(arm=len(arm.commands))

    arm.send_joint_target = send_then_stop
    return at_stop


@pytest.mark.parametrize("kind", ["estop_harness", "estop_safearm"])
@pytest.mark.parametrize("skill", ["reset_scene", "move_home"])
def test_fr3_view_home_estop_mid_route_stays_an_estop(rig, skill, kind):
    """An e-stop during the view-home route is the latched e-stop and is
    reported as one (`SafetyViolation: e-stop latched`), never as the
    operator's halt (`MotionHalted`)."""
    r = rig("fr3")
    _fr3_off_home(r)
    runtime = r.runtime
    at_stop = _stop_at_waypoint(r, runtime, kind)

    out = runtime.execute(skill, {})

    assert at_stop, "the view-home route never streamed"
    assert len(r.arms[0].commands) == at_stop["arm"], "an arm command went out after the e-stop"
    assert out["ok"] is False, out
    if skill == "reset_scene":
        assert out["stage"] == "home" and out["home_error"] == "SafetyViolation: e-stop latched", out
    else:
        assert out["error"] == "SkillError: SafetyViolation: e-stop latched", out
    assert "MotionHalted" not in out["error"], out
    assert runtime.arm.harness.estopped


def _halt_as_the_route_ends(r, runtime, monkeypatch):
    """The halt arrives after the view-home route's last approved command,
    before the return's own after-move check."""
    move = runtime._move_carry
    at_halt = {}

    def move_then_halt(*args, **kwargs):
        settled = move(*args, **kwargs)
        if not at_halt and sys._getframe(1).f_code.co_name == "_return_to_view_home":
            runtime.arm.harness.halt("operator: stop")
            at_halt.update(arm=_arm_commands(r.arms))
        return settled

    monkeypatch.setattr(runtime, "_move_carry", move_then_halt)
    return at_halt


@pytest.mark.parametrize("skill", ["reset_scene", "move_home"])
def test_fr3_view_home_halt_at_the_route_end_stays_a_halt(rig, monkeypatch, skill):
    """The return's after-move check finds the halt: it ends the return as
    a halt (MotionHalted), not as a SkillError whose cleanup latches an
    e-stop with arm.stop()."""
    r = rig("fr3")
    _fr3_off_home(r)
    runtime = r.runtime
    at_halt = _halt_as_the_route_ends(r, runtime, monkeypatch)

    out = runtime.execute(skill, {})

    assert at_halt, "the view-home route never ran"
    assert _arm_commands(r.arms) == at_halt["arm"], "an arm command went out after the halt"
    assert out["ok"] is False, out
    if skill == "reset_scene":
        assert out["stage"] == "home" and out["home_error"] == "halted: operator: stop", out
    else:
        assert out["error"] == "MotionHalted: halted: operator: stop", out
    assert not runtime.arm.harness.estopped, "the halt was escalated to an e-stop"


def test_fr3_post_skill_return_halted_mid_route_names_the_halt_once(rig):
    """execute()'s camera-view return after a motion skill, halted mid-route:
    the error reads "return to saved home halted: <reason>", without the
    harness's own "halted: " repeated."""
    r = rig("fr3")
    runtime = r.runtime
    runtime.arm.get_state()  # materialize the LazyArm: the driver exists from here on
    show = runtime._show_status
    returning = {}

    def show_status(text):
        show(text)
        if text.startswith("Returning to the saved camera-view home"):
            returning.update(at=_arm_commands(r.arms))

    runtime._show_status = show_status
    arm = r.arms[0]
    send = arm.send_joint_target
    at_halt = {}

    def send_then_halt(q):
        send(q)
        if returning and not at_halt and _arm_commands(r.arms) >= returning["at"] + 5:
            runtime.arm.harness.halt("operator: stop")
            at_halt.update(arm=_arm_commands(r.arms))

    arm.send_joint_target = send_then_halt

    out = runtime.execute("rotate_joint", {"joint": 1, "degrees": 10.0})

    assert at_halt, "the post-skill return never streamed"
    assert _arm_commands(r.arms) == at_halt["arm"], "an arm command went out after the halt"
    assert out["ok"] is False and out["return_home"]["status"] == "halted", out
    assert out["error"] == "return to saved home halted: operator: stop", out
    assert "halted: halted" not in out["next_action"], out["next_action"]


def _fault(width_frac: float | None, exc: BaseException):
    """A driver that moves the jaws to `width_frac` (None: not at all), then raises."""

    def fault(arm, pos, effort):
        if width_frac is not None:
            span = arm._grip_closed - arm._grip_open
            MockArm.set_gripper(arm, arm._grip_open + (1.0 - width_frac) * span, effort)
        raise exc

    return fault


BLOCKED = GripperFault("gripper blocked while opening at 4.34 rad (target 6.20): "
                       "holding there instead of pushing. Check what is against the jaws.")


@pytest.mark.parametrize("profile", ["mock", "so101_mock"])
def test_a_gripper_fault_part_way_keeps_the_object_and_reports_the_width(rig, profile):
    r = rig(profile)
    _holding(r)
    for arm in r.arms:
        arm.fault = _fault(0.7, BLOCKED)

    out = r.runtime.execute("reset_scene", {})

    assert r.jaws == [pytest.approx(r.runtime._grip_open)]  # the command went out
    assert out["ok"] is False and out["stage"] == "release", out
    assert out["release_error"] == str(BLOCKED), out
    assert out["gripper_open_frac"] == pytest.approx(0.7), out
    assert "stopped before the release" not in out["error"], out
    assert "may have dropped" in out["error"] and "70%" in out["error"], out
    _assert_still_held(r, out)


def test_a_gripper_fault_with_an_unknown_jaw_width_keeps_the_object(rig):
    r = rig("mock")
    _holding(r)

    def fault(arm, pos, effort):
        arm.jaw_unknown = True
        raise BLOCKED

    for arm in r.arms:
        arm.fault = fault

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"] is False and out["stage"] == "release", out
    assert out["gripper_open_frac"] == "unknown", out
    assert "may have dropped" in out["error"], out
    _assert_still_held(r, out)


@pytest.mark.parametrize("width", [0.98, 1.0])
def test_a_gripper_fault_after_the_jaws_measured_open_releases_the_object(rig, width):
    r = rig("mock")
    _holding(r)
    for arm in r.arms:
        arm.fault = _fault(width, BLOCKED)

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"] is False and out["stage"] == "release", out  # the fault is reported
    assert out["release_error"] == str(BLOCKED), out
    assert out["gripper_open_frac"] == pytest.approx(width), out
    assert "stopped before the release" not in out["error"], out
    assert r.runtime.held_object is None and r.runtime._held_det_label is None


@pytest.mark.parametrize("width", [None, 1.0], ids=["jaws-still", "jaws-moved-open"])
def test_a_non_safety_jaw_error_keeps_the_object(rig, width):
    """Whatever the jaws read afterwards: a bus or bridge error says nothing
    reliable about the release, so the object is never forgotten."""
    r = rig("mock")
    half = _holding(r)
    for arm in r.arms:
        arm.fault = _fault(width, RuntimeError("gripper bus timeout"))

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"] is False and out["stage"] == "release", out
    assert out["release_error"] == "gripper bus timeout", out
    assert "gripper bus timeout" in out["error"], out
    _assert_still_held(r, out)
    if width is None:
        assert r.runtime.arm.get_state().gripper_pos == pytest.approx(half)


#: A driver text longer than the agent's failure line keeps (240 chars).
LONG_BUS_ERROR = RuntimeError("gripper bus timeout: " + "no reply from motor 7 on can0; " * 8)


@pytest.mark.parametrize("width, exc, outcome", [
    (0.7, BLOCKED, "is kept as held"),
    (1.0, BLOCKED, "is no longer held"),
    (None, LONG_BUS_ERROR, "is kept as held"),
], ids=["fault-part-way", "fault-measured-open", "bus-error"])
def test_a_failed_release_text_leads_with_the_outcome(rig, width, exc, outcome):
    """The agent's `next_action` keeps the first 240 chars of the error: the
    outcome (what happened to the object) comes before the driver's text, and
    the driver's own full stop is never doubled."""
    r = rig("mock")
    _holding(r)
    for arm in r.arms:
        arm.fault = _fault(width, exc)

    out = r.runtime.execute("reset_scene", {})

    error = out["error"]
    assert error.index(f"red cube {outcome}") < error.index(str(exc)), error
    assert f"red cube {outcome}" in out["next_action"], out["next_action"]
    assert ".." not in error, error
    if exc is BLOCKED:
        assert "may have dropped" in out["next_action"], out["next_action"]


def _unknown_width(exc: BaseException):
    """A driver that raises and leaves the jaw position unreadable."""

    def fault(arm, pos, effort):
        arm.jaw_unknown = True
        raise exc

    return fault


#: A 60-character object name (the agent's labels can be this long).
LONG_NAME = "transparent plastic water bottle with a blue screw-top lid 1"


@pytest.mark.parametrize("fault, outcome", [
    (_fault(0.7, BLOCKED), "is kept as held"),
    (_unknown_width(BLOCKED), "is kept as held"),
    (_fault(1.0, BLOCKED), "is no longer held"),
    (_fault(None, LONG_BUS_ERROR), "is kept as held"),
], ids=["fault-part-way", "fault-unknown-width", "fault-measured-open", "bus-error"])
def test_a_failed_release_text_names_a_long_object_once(rig, fault, outcome):
    """The object is named once, so with a 60-char name the outcome still
    sits inside the 240 chars the agent's failure line keeps."""
    assert len(LONG_NAME) == 60
    r = rig("mock")
    _holding(r)
    r.runtime.held_object = LONG_NAME
    for arm in r.arms:
        arm.fault = fault

    out = r.runtime.execute("reset_scene", {})

    error = out["error"]
    assert error.count(LONG_NAME) == 1, error
    kept = out["next_action"].split(" (truncated; see error)")[0]
    assert f"{LONG_NAME} {outcome}" in kept, out["next_action"]
    if "Gripper fault" in error:
        assert "may have dropped" in kept and "the jaws measured" in kept, out["next_action"]


def test_the_reset_scene_description_says_a_real_arm_refuses_while_holding():
    from cascade.skills.runtime import TOOL_SPECS

    text = next(t for t in TOOL_SPECS if t["name"] == "reset_scene")["description"]
    assert "refuses while the gripper holds an object" in text, text
    assert "place it first (place_at), then reset" in text, text
