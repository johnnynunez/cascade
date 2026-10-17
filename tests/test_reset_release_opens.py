"""`reset_scene` releases a held object with the profile's OPEN jaw position.

Found during the PR-A fix round (#354): reset_scene's release sent the raw
gripper position 1.0. Jaw positions are motor travel in each profile's own
units (`gripper.open_pos` .. `gripper.closed_pos`), so 1.0 means something
different on every arm. On the reBot RS (open 6.2, closed 0.0) it is about
16 % open: nearly CLOSED, so "release" squeezed the object instead of letting
it go. On the mock (open 0.0, closed 1.0) it is fully closed, on the SO-101
(open 0.60) past the open stop, on the FR3 (open 0.04 m) 25 times the
travel. Only Isaac (open 1.0) happened to be right.

The opening carries the reset's halt generation: a halt, an e-stop or a
gripper gate that refuses after the home move sends no jaw command (and no
arm command), keeps the object in the runtime's hands and ends the reset.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every jaw command; no driver, no bus.
"""

from __future__ import annotations

import pytest

from cascade.control.mock_arm import MockArm
from cascade.types import SafetyViolation

#: (profile, the open position its YAML declares). Literals on purpose: the
#: test pins the semantic open, independently of how the runtime reads it.
PROFILES = [
    ("rebot_rs", 6.2),
    ("so101_mock", 0.60),
    ("isaac_kitchen", 1.0),
    ("fr3", 0.04),
    ("mock", 0.0),
]


class _JawArm(MockArm):
    """MockArm that records every jaw position it is commanded."""

    def __init__(self, cfg, kinematics, jaws):
        super().__init__(cfg, kinematics)
        self.jaws = jaws

    def set_gripper(self, pos, effort=1.0):
        self.jaws.append(float(pos))
        super().set_gripper(pos, effort)


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
        jaws, arms = [], []

        def make_arm(acfg, kinematics=None):
            arm = _JawArm(acfg, kinematics, jaws)
            arms.append(arm)
            return arm

        # Replaced BEFORE the build: no real driver is ever constructed.
        monkeypatch.setattr(demo, "make_arm", make_arm)
        runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((runtime, arm))
        return runtime, cfg, jaws, arms

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _holding(runtime, cfg):
    """The jaws half closed on an object the runtime knows it holds."""
    g = cfg.arm.gripper
    half = g.open_pos + 0.5 * (g.closed_pos - g.open_pos)
    runtime.arm.set_gripper(half)
    runtime.held_object = "red cube"
    runtime._held_det_label = "cube"
    return half


def _arm_commands(arms) -> int:
    return sum(len(a.commands) for a in arms)


@pytest.mark.parametrize("profile, open_pos", PROFILES, ids=[p for p, _ in PROFILES])
def test_reset_releases_with_the_profiles_open_position(rig, profile, open_pos):
    runtime, cfg, jaws, _ = rig(profile)
    assert cfg.arm.gripper.open_pos == pytest.approx(open_pos)
    _holding(runtime, cfg)
    jaws.clear()

    out = runtime.execute("reset_scene", {})

    assert out["ok"], out
    assert jaws == [pytest.approx(open_pos)], jaws  # one jaw command: the release, fully open
    assert runtime.arm.get_state().gripper_pos == pytest.approx(open_pos)
    assert runtime._gripper_width_frac() == pytest.approx(1.0)
    assert runtime.held_object is None and runtime._held_det_label is None


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
@pytest.mark.parametrize("profile", ["rebot_rs", "mock"])
def test_a_stop_after_home_sends_nothing_and_keeps_the_object(rig, monkeypatch, profile, kind):
    runtime, cfg, jaws, arms = rig(profile)
    half = _holding(runtime, cfg)
    jaws.clear()
    home = runtime.skill_move_home
    at_stop = {}

    def home_then_stop(**kwargs):
        result = home(**kwargs)
        _stop(runtime, kind)
        at_stop.update(arm=_arm_commands(arms), jaws=len(jaws))
        return result

    monkeypatch.setattr(runtime, "skill_move_home", home_then_stop)

    out = runtime.execute("reset_scene", {})

    assert at_stop, "reset_scene never reached its home move"
    assert len(jaws) == at_stop["jaws"] == 0, jaws          # no jaw command after the stop
    assert _arm_commands(arms) == at_stop["arm"]           # nor any arm command
    assert out["ok"] is False and out["stage"] == "release", out
    assert out["beliefs_forgotten"] == 0 and not out["observation_refreshed"], out
    assert "release" in out["error"] and out["release_error"], out
    # The object is still in the jaws: the runtime keeps it.
    assert runtime.held_object == "red cube" and runtime._held_det_label == "cube"
    assert runtime.arm.get_state().gripper_pos == pytest.approx(half)
