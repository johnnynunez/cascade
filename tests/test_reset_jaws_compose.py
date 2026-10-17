"""reset_scene's jaws after merging main's #357 into PR-A (#354).

Two changes meet at reset's one jaw step at home:
- #357: a held object is released to the profile's OPEN position (never a
  raw 1.0) on simulated / mock arms; a real arm refuses the reset while
  holding, before any motion.
- #354 (jaws round, ruling 2): closed EMPTY jaws open at home, so the next
  pick starts with open jaws.

The merged rule: ONE reset-start halt generation (taken before the reset's
carry check), at most ONE jaw command at home (the release, or the opening of
closed empty jaws, never both), effort 0.8, carrying that generation (a
recovery's when one runs). A halt or an e-stop at any point of the reset
sends no further arm or jaw command; a halt stays a halt (no e-stop latch)
and an e-stop stays the e-stop (never reported as MotionHalted).

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every streamed arm command and every
jaw command; no driver, no bus.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.control.mock_arm import MockArm
from cascade.types import GripperFault

#: simulated / mock profiles: a held object is released at home
SIM = ["mock", "isaac_kitchen", "so101_mock"]
#: real grippers: the reset refuses while holding; fr3 homes through its
#: camera-view home return (`return_home_after_motion`)
REAL = ["rebot_rs", "fr3"]
REFUSAL = "holding red cube: place it first (place_at), then reset"


class _RecArm(MockArm):
    """MockArm that records every streamed arm command and every jaw command
    (with its effort) in one log, and runs `r.on_send()` after each arm
    command."""

    def __init__(self, cfg, kinematics, r):
        super().__init__(cfg, kinematics)
        self.r = r
        self.fault = None

    def send_joint_target(self, q):
        self.r.log.append(("arm",))
        super().send_joint_target(q)
        if self.r.on_send is not None:
            self.r.on_send()

    def set_gripper(self, pos, effort=1.0):
        self.r.log.append(("jaw", float(pos), float(effort)))
        if self.fault is not None:
            self.fault(self, pos, effort)
            return
        super().set_gripper(pos, effort)
        if self.r.on_jaw is not None:
            self.r.on_jaw()


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)

    def build(profile):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        # rebot_rs requires its live camera map and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        r = SimpleNamespace(cfg=cfg, log=[], arms=[], on_send=None, on_jaw=None, at_stop=None)

        def make_arm(acfg, kinematics=None):
            arm = _RecArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _jaws(r):
    return [entry[1:] for entry in r.log if entry[0] == "jaw"]


def _start(r, jaws):
    """Arm off home (so the home move streams), then the jaws `open`,
    `closed` (empty) or `held` (half closed on an object the runtime holds).
    Clears the log."""
    runtime, g = r.runtime, r.cfg.arm.gripper
    home = np.asarray(runtime.cfg.arm.home_q, dtype=float)
    off = home.copy()
    off[0] += 0.15
    assert runtime.arm.move_joints(off, duration_s=0.5)
    if jaws == "open":
        runtime.arm.set_gripper(g.open_pos)
    elif jaws == "closed":
        runtime.arm.set_gripper(g.closed_pos)
    else:
        runtime.arm.set_gripper(g.open_pos + 0.5 * (g.closed_pos - g.open_pos))
        runtime.held_object = "red cube"
        runtime._held_det_label = "cube"
    r.log.clear()


def _stop_now(r, kind):
    harness = r.runtime.arm.harness
    if kind == "halt":
        harness.halt("operator: stop")
    elif kind == "halt resumed":       # the operator's Resume: the generation still moved
        harness.halt("operator: stop")
        harness.clear_halt()
    else:
        harness.estop("operator e-stop")
    r.at_stop = len(r.log)


def _stop_at(r, monkeypatch, where, kind):
    """Inject the stop at `where` in the reset."""
    from cascade.skills import carry_attachment

    runtime = r.runtime
    if where == "carry_check":            # the reset's own carry check, before home
        check = carry_attachment.check

        def check_then_stop(rt):
            check(rt)
            if r.at_stop is None and sys._getframe(1).f_code.co_name == "skill_reset_scene":
                _stop_now(r, kind)
        monkeypatch.setattr(carry_attachment, "check", check_then_stop)
    elif where == "home_route":           # the 5th command of the home move
        home = runtime.skill_move_home
        start = []

        def spy_home(**kwargs):
            start.append(len(r.log))
            return home(**kwargs)

        def on_send():
            if r.at_stop is None and start and len(r.log) - start[0] >= 5:
                _stop_now(r, kind)
        monkeypatch.setattr(runtime, "skill_move_home", spy_home)
        r.on_send = on_send
    elif where == "after_home":           # home reached, before the jaw command
        home = runtime.skill_move_home

        def home_then_stop(**kwargs):
            out = home(**kwargs)
            _stop_now(r, kind)
            return out
        monkeypatch.setattr(runtime, "skill_move_home", home_then_stop)
    else:                                  # "jaw_command": inside the driver's jaw command
        def on_jaw():
            if r.at_stop is None:
                _stop_now(r, kind)
        r.on_jaw = on_jaw


# ── one jaw command at home ──────────────────────────────────────────────────


@pytest.mark.parametrize("call", ["execute", "direct"])
@pytest.mark.parametrize("profile", SIM)
def test_sim_held_object_gets_exactly_one_opening_to_the_open_position(rig, profile, call):
    r = rig(profile)
    _start(r, "held")
    out = r.runtime.execute("reset_scene", {}) if call == "execute" else r.runtime.skill_reset_scene()
    assert out["ok"], out
    assert _jaws(r) == [(pytest.approx(r.runtime._grip_open), 0.8)], _jaws(r)
    assert r.runtime._grip_open == pytest.approx(r.cfg.arm.gripper.open_pos)
    assert r.runtime.held_object is None and "jaws_open_commanded" not in out


@pytest.mark.parametrize("call", ["execute", "direct"])
@pytest.mark.parametrize("profile", SIM)
def test_slow_jaws_after_the_release_get_no_second_opening(rig, profile, call):
    """Slow jaws: the release command is recorded but the jaws still read
    closed when the reset looks at them next. The release already was the
    one jaw command at home, so they never get a second (empty-jaws) one."""
    r = rig(profile)
    _start(r, "held")
    r.arms[0].fault = lambda arm, pos, effort: None   # recorded; the jaws have not moved yet
    out = r.runtime.execute("reset_scene", {}) if call == "execute" else r.runtime.skill_reset_scene()
    assert out["ok"], out
    assert r.runtime._gripper_width_frac() < r.runtime._reset_open_frac()   # still read closed
    assert _jaws(r) == [(pytest.approx(r.runtime._grip_open), 0.8)], _jaws(r)
    assert r.runtime.held_object is None and "jaws_open_commanded" not in out


@pytest.mark.parametrize("profile", REAL)
def test_real_held_object_refuses_before_any_command(rig, profile):
    r = rig(profile)
    _start(r, "held")
    out = r.runtime.execute("reset_scene", {})
    assert out["ok"] is False and out["stage"] == "release" and REFUSAL in out["error"], out
    assert r.log == [], r.log                      # no arm, no jaw command
    assert r.runtime.held_object == "red cube"


@pytest.mark.parametrize("call", ["execute", "direct"])
@pytest.mark.parametrize("profile", SIM + REAL)
def test_closed_empty_jaws_get_exactly_one_opening(rig, profile, call):
    r = rig(profile)
    _start(r, "closed")
    out = r.runtime.execute("reset_scene", {}) if call == "execute" else r.runtime.skill_reset_scene()
    assert out["ok"] and out["jaws_open_commanded"] is True, out
    assert _jaws(r) == [(pytest.approx(r.runtime._grip_open), 0.8)], _jaws(r)


@pytest.mark.parametrize("profile", SIM + REAL)
def test_open_empty_jaws_get_no_jaw_command(rig, profile):
    r = rig(profile)
    _start(r, "open")
    out = r.runtime.execute("reset_scene", {})
    assert out["ok"] and "jaws_open_commanded" not in out, out
    assert _jaws(r) == []


# ── a halt or an e-stop at any point: 0 further commands ────────────────────

CASES = ([(p, "held") for p in SIM] + [(p, "closed") for p in SIM + REAL])
#: (where, kind). A halt the operator resumes between two of the reset's
#: steps still ends it (the generation moved); one resumed INSIDE a single
#: command is withdrawn before the stream's next check (the harness's rule).
STOPS = [(where, kind) for where in ("carry_check", "home_route", "after_home", "jaw_command")
         for kind in ("halt", "halt resumed", "estop")
         if not (kind == "halt resumed" and where in ("home_route", "jaw_command"))]


@pytest.mark.parametrize("call", ["execute", "direct"])
@pytest.mark.parametrize("where, kind", STOPS, ids=[f"{w}-{k}" for w, k in STOPS])
@pytest.mark.parametrize("profile, jaws", CASES, ids=[f"{p}-{j}" for p, j in CASES])
def test_a_stop_anywhere_in_the_reset_sends_no_further_command(rig, monkeypatch, profile, jaws, where,
                                                               kind, call):
    r = rig(profile)
    _start(r, jaws)
    _stop_at(r, monkeypatch, where, kind)
    out = r.runtime.execute("reset_scene", {}) if call == "execute" else r.runtime.skill_reset_scene()
    assert r.at_stop is not None, "the stop was never injected"
    assert r.log[r.at_stop:] == [], f"commands after the stop: {r.log[r.at_stop:]}"
    harness = r.runtime.arm.harness
    if where != "jaw_command":         # the jaw command already went out: the reset may finish
        assert out["ok"] is False, out
        if jaws == "held":
            assert r.runtime.held_object == "red cube"  # nothing released it
    if kind == "estop":
        assert harness.estopped and "MotionHalted" not in str(out.get("error")), out
    else:
        assert not harness.estopped, "the halt was escalated to an e-stop"


def test_a_gripper_fault_while_opening_empty_jaws_reports_the_width_and_ends_the_reset(rig):
    """A GripperFault comes AFTER the command went out (#357's rule): the
    text says so, outcome first, with the measured width; nothing is held."""
    r = rig("rebot_rs")
    _start(r, "closed")

    def blocked(arm, pos, effort):
        MockArm.set_gripper(arm, arm._grip_open + 0.6 * (arm._grip_closed - arm._grip_open), effort)
        raise GripperFault("opening blocked at 40%")
    r.arms[0].fault = blocked
    out = r.runtime.execute("reset_scene", {})
    assert out["ok"] is False and out["stage"] == "release", out
    assert out["release_error"] == "opening blocked at 40%" and out["gripper_open_frac"] == pytest.approx(.4)
    assert out["error"] == ("Scene reset stopped: the gripper faulted while opening the empty jaws and the jaws "
                            "measured 40% open; check the gripper. Gripper fault: opening blocked at 40%"), out
    assert not out["observation_refreshed"] and len(_jaws(r)) == 1
