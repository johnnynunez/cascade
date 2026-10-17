"""An operator halt during a retry's re-home ends the whole skill.

The persistence loops (`_grasp_with_persistence` behind handover and
sort_by_color, pick_and_place's grasp and place loops) re-home between
attempts and swallowed every `SkillError` / `SafetyViolation` of that move,
`MotionHalted` included. The next attempt's first motion carries no halt
generation, so it CLEARED the halt and the arm went on after an operator
STOP: on fr3 25-237 arm commands after the halt, on rebot_rs 125-200 (main
too). A halt raised in the re-home now propagates (`except MotionHalted:
raise` before the swallow), and so do the composites that loop over
`_grasp_with_persistence` (sort_by_color, `_move_object_to` behind
restore_scene / search_for_object, including its clear-view homes).

An e-stop is not a halt: on the fr3 view-home return it stays
`SafetyViolation: e-stop latched`, never reported as `MotionHalted`.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that counts every streamed command; the grasp and
place primitives are replaced by fakes that move the arm and miss.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.control.mock_arm import MockArm
from cascade.types import MotionHalted, SkillError

HALT = "halted: operator: stop"
RETRY_PROFILES = ["fr3", "rebot_rs"]


class _SendArm(MockArm):
    """MockArm that counts every streamed command (also refused-by-driver
    ones) and runs `r.on_send()` after each."""

    def __init__(self, cfg, kinematics, r):
        super().__init__(cfg, kinematics)
        self.r = r

    def send_joint_target(self, q):
        self.r.sends += 1
        super().send_joint_target(q)
        if self.r.on_send is not None:
            self.r.on_send()


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    # Observed-finger gate off: with it on, the grasp loops return after the
    # first failed attempt and never re-home (the place loop still does:
    # its test runs both ways).
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)

    def build(profile):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        # rebot_rs requires its live camera map and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        r = SimpleNamespace(cfg=cfg, sends=0, on_send=None, arms=[], grasps=0, places=0,
                            home_start=None, at_stop=None)

        def make_arm(acfg, kinematics=None):
            arm = _SendArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _stop_now(r, kind):
    harness = r.runtime.arm.harness
    if kind == "halt":
        harness.halt("operator: stop")
    elif kind == "estop_harness":
        harness.estop("operator e-stop")
    else:  # "estop_safearm": SafeArm.stop() latches the e-stop and stops the driver
        r.runtime.arm.stop()
    r.at_stop = r.sends


def _stop_in_rehome(r, monkeypatch, kind="halt", waypoint=5, nth_home=1):
    """Stop (halt / e-stop) at the `waypoint`-th command of the `nth_home`
    skill_move_home call (the loop's re-home)."""
    runtime = r.runtime
    home = runtime.skill_move_home
    calls = []

    def spy_home(**kwargs):
        calls.append(kwargs)
        if len(calls) == nth_home:
            r.home_start = r.sends
        return home(**kwargs)

    def on_send():
        if r.at_stop is None and r.home_start is not None and r.sends - r.home_start >= waypoint:
            _stop_now(r, kind)

    monkeypatch.setattr(runtime, "skill_move_home", spy_home)
    r.on_send = on_send


def _move_off(runtime, step):
    q = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    q[0] += step
    assert runtime.arm.move_joints(q, duration_s=1.5)


def _missing_grasp(r, monkeypatch, *, hold_on=None):
    """A grasp that moves the arm and misses; attempt `hold_on` holds."""
    runtime = r.runtime

    def grasp(object, material=None, **kwargs):
        r.grasps += 1
        _move_off(runtime, 0.15 if r.grasps % 2 else -0.15)
        if hold_on is not None and r.grasps >= hold_on:
            runtime.held_object = object
            runtime._held_det_label = "cube"
            return {"held": object, "grip_verified": True, "grip_profile": None}
        return {"ok": False, "error": "grasp missed: closed on air"}

    monkeypatch.setattr(runtime, "skill_grasp_object", grasp)


def _missing_place(r, monkeypatch):
    runtime = r.runtime

    def place(x, y, **kwargs):
        r.places += 1
        _move_off(runtime, 0.15)
        raise SkillError("place missed: target not reachable")

    monkeypatch.setattr(runtime, "skill_place_at", place)


def _assert_halted_result(r, out):
    assert r.at_stop is not None, "the retry never re-homed"
    assert r.sends == r.at_stop, f"{r.sends - r.at_stop} arm commands went out after the halt"
    assert out["ok"] is False, out
    assert out["error"] == f"MotionHalted: {HALT}", out
    assert out["next_action"].count("This attempt FAILED") == 1, out["next_action"]
    assert not r.runtime.arm.harness.estopped, "the halt was escalated to an e-stop"


@pytest.mark.parametrize("profile", RETRY_PROFILES)
@pytest.mark.parametrize("skill, args", [
    ("handover", {"label": "red cube"}),
    ("pick_and_place", {"object": "red cube"}),
], ids=["handover", "pick_and_place"])
def test_a_halt_in_the_grasp_retry_rehome_ends_the_skill(rig, monkeypatch, profile, skill, args):
    r = rig(profile)
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    out = r.runtime.execute(skill, args)

    _assert_halted_result(r, out)
    assert r.grasps == 1, "the loop tried again after the halt"


@pytest.mark.parametrize("gate", ["off", "on"])
@pytest.mark.parametrize("profile", RETRY_PROFILES)
def test_a_halt_in_the_place_retry_rehome_ends_pick_and_place(rig, monkeypatch, profile, gate):
    if gate == "on":
        monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    r = rig(profile)
    _missing_grasp(r, monkeypatch, hold_on=1)
    _missing_place(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    out = r.runtime.execute("pick_and_place", {"object": "red cube"})

    _assert_halted_result(r, out)
    assert r.places == 1, "the place loop tried again after the halt"
    assert r.runtime.held_object == "red cube"  # still in the jaws: nothing released it


@pytest.mark.parametrize("profile", RETRY_PROFILES)
def test_grasp_with_persistence_raises_a_halt_from_its_rehome(rig, monkeypatch, profile):
    r = rig(profile)
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    with pytest.raises(MotionHalted) as raised:
        r.runtime._grasp_with_persistence("red cube")

    assert str(raised.value) == HALT
    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the halt"
    assert r.grasps == 1


def _two_beliefs(runtime, monkeypatch):
    beliefs = [SimpleNamespace(label="cube", color=color, position=np.array([0.32, y, 0.02]))
               for color, y in (("red", 0.12), ("blue", -0.12))]
    monkeypatch.setattr(runtime.beliefs, "all", lambda: list(beliefs))


@pytest.mark.parametrize("profile", RETRY_PROFILES)
def test_a_halt_ends_sort_by_color_not_just_one_object(rig, monkeypatch, profile):
    """sort_by_color went on with the NEXT object after a halt in the first
    one's persistence loop; that object's first grasp cleared the halt."""
    r = rig(profile)
    _two_beliefs(r.runtime, monkeypatch)
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    out = r.runtime.execute("sort_by_color", {})

    _assert_halted_result(r, out)
    assert r.grasps == 1, "the sort went on to the next object after the halt"


@pytest.mark.parametrize("profile", RETRY_PROFILES)
@pytest.mark.parametrize("where", ["grasp_rehome", "home_after_a_miss", "home_after_the_place"])
def test_a_halt_ends_move_object_to(rig, monkeypatch, profile, where):
    """`_move_object_to` is the step of restore_scene's and
    search_for_object's loops: a halt in its grasp loop or in its clear-view
    home must reach them, or their next step moves the arm again."""
    r = rig(profile)
    runtime = r.runtime
    if where == "grasp_rehome":
        _missing_grasp(r, monkeypatch)
        _stop_in_rehome(r, monkeypatch)                 # the persistence loop's re-home
    elif where == "home_after_a_miss":
        runtime.cfg._data.setdefault("grasp", {})["max_pick_attempts"] = 1
        _missing_grasp(r, monkeypatch)
        _stop_in_rehome(r, monkeypatch)                 # its clear-view home after the miss
    else:
        _missing_grasp(r, monkeypatch, hold_on=1)

        def place(x, y, **kwargs):
            r.places += 1
            _move_off(runtime, 0.15)
            runtime.held_object = None
            return {"ok": True, "at": [x, y]}

        monkeypatch.setattr(runtime, "skill_place_at", place)
        _stop_in_rehome(r, monkeypatch)                 # its clear-view home after the place

    with pytest.raises(MotionHalted) as raised:
        runtime._move_object_to("red cube", [0.30, 0.10], budget_s=30.0, kind="restore")

    assert str(raised.value) == HALT
    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the halt"
    assert r.grasps == 1


@pytest.mark.parametrize("kind", ["estop_harness", "estop_safearm"])
def test_an_estop_in_the_fr3_retry_rehome_stays_an_estop(rig, monkeypatch, kind):
    """An e-stop during the view-home re-home is the latched e-stop, never
    reported as an operator halt."""
    r = rig("fr3")
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch, kind=kind)

    out = r.runtime.execute("handover", {"label": "red cube"})

    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the e-stop"
    assert out["ok"] is False, out
    assert "SafetyViolation: e-stop latched" in out["error"], out
    assert "MotionHalted" not in out["error"], out
    assert r.runtime.arm.harness.estopped
