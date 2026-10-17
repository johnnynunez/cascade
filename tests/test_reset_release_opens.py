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
from home height (reBot home TCP at 0.44 m). So a REAL profile (the arm type
the runtime builds is not isaac*, mujoco* or mock*; an unknown type counts as
real) refuses a reset while holding, before any motion, and asks for place_at
first. Simulated and mock arms keep releasing at home.

The reset takes its halt generation when it starts (before its carry check)
and passes it to the home move and to the opening: a halt anywhere in the
reset sends no further arm or jaw command. A refusal by SafeArm's own checks
(halt, e-stop, a gripper gate) is raised before any jaw command: the object
stays held. A driver `GripperFault` comes AFTER commands went out (opening
blocked part-way): the result reports the measured jaw width and keeps the
object unless that width proves the jaws open. A non-safety driver error
keeps the object too, with the error recorded.

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

    def build(profile):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
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


def test_real_reset_with_empty_jaws_still_resets(rig):
    r = rig("rebot_rs")

    out = r.runtime.execute("reset_scene", {})

    assert out["ok"], out
    assert r.jaws == [] and out["observation_refreshed"]


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
