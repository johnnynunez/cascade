"""Every arm's safety config is validated BEFORE any arm is made or connected.

`SafetyHarness(...)` is where a bad arm safety config refuses: a malformed
`safety.tool_volume` block, or a tool volume on a tilted `base_pose`.
`_build_arm` used to make (and, eager, connect) the backend first and build
the harness after it, so on an eager real arm the refusal came after the
connect had enabled torque, and the failed build left the arm connected. On a
two-arm rig, arm 0 was fully built and connected before arm 1's config was
even read. Every backend below is a recording MockArm (no hardware). Pinned:

1. an eager `_build_arm` whose harness refuses makes and connects nothing;
2. an eager two-arm `build_runtime` whose SECOND arm refuses makes and
   connects neither arm;
3. lazy builds (the MCP server path) still make and connect nothing;
4. a valid build makes the same objects, harnesses first, then each arm in
   order;
5. the tilted-mount refusal names the arm and its roll / pitch in degrees.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import needs_pin, needs_pin_so101

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import ToolVolume
from cascade.types import pose_to_transform

#: roll 0.1 rad (5.73 deg): a tilted mount
TILTED = [0.0, -0.22, 0.0, 0.1, 0.0, 0.0]
BAD = ("tilted mount", "bad tool_volume block")
#: what each case's refusal says
REFUSAL = {"tilted mount": "level base_pose", "bad tool_volume block": "table_band_m must be in"}


@pytest.fixture
def events(monkeypatch):
    """Every make / connect / harness construction, in order. A harness is
    recorded once it is built, by its base_pose y (None: no base_pose)."""
    from cascade.apps import demo
    from cascade.control.mock_arm import MockArm

    log = []

    def make_arm(acfg, kinematics=None):
        name = str(acfg.get("name", acfg.get("type", "arm")))
        log.append(("make", name))

        class Recorded(MockArm):
            def connect(self):
                log.append(("connect", name))
                super().connect()

        return Recorded(acfg, kinematics=kinematics)

    real = demo.SafetyHarness

    def harness(*args, **kwargs):
        h = real(*args, **kwargs)
        log.append(("harness", None if h.base_pose is None else round(float(h.base_pose[1, 3]), 3)))
        return h

    monkeypatch.setattr(demo, "make_arm", make_arm)
    monkeypatch.setattr(demo, "SafetyHarness", harness)
    return log


def _arm_calls(log):
    return [e for e in log if e[0] != "harness"]


def _rebot(bad=None):
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    acfg = cfg.arm
    assert acfg.get("resolved").get("safety").get("tool_volume") is not None   # premise: it has one
    if bad == "tilted mount":
        acfg._data["base_pose"] = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]
    elif bad == "bad tool_volume block":
        acfg._data["resolved"]["safety"]["tool_volume"]["table_band_m"] = 0.2
    return cfg, acfg


def _pair(bad=None):
    """The so101 pair; `bad` breaks the SECOND arm (so101_right) only."""
    from cascade.config import load_demo_config

    cfg = load_demo_config(arms=["so101_left", "so101_right"], camera="mock", llm="mock")
    right = cfg._data["arms"][1]
    assert right["name"] == "so101_right"
    if bad is not None:
        right["resolved"]["safety"]["tool_volume"] = {"wrist_camera": False, "jaws": True}
    if bad == "tilted mount":
        right["base_pose"] = TILTED
    elif bad == "bad tool_volume block":
        right["resolved"]["safety"]["tool_volume"]["table_band_m"] = 0.2
    return cfg


# ── 1. one arm: the harness refuses before the arm exists ────────────────


@needs_pin
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("bad", BAD)
def test_a_refusing_harness_config_makes_and_connects_no_arm(events, bad, lazy):
    """RED on b0df584 for eager: make + connect (torque on a real reBot),
    then the ValueError. Lazy never made the arm: pinned unchanged."""
    from cascade.apps import demo

    cfg, acfg = _rebot(bad)
    with pytest.raises(ValueError, match=REFUSAL[bad]):
        demo._build_arm(acfg, lazy, None, cfg)
    assert _arm_calls(events) == []


# ── 2. two arms: arm 1 refusing leaves arm 0 untouched ──────────────────


@needs_pin_so101
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("bad", BAD)
def test_a_two_arm_build_whose_second_arm_refuses_touches_neither_arm(events, tmp_path, bad, lazy):
    """RED on b0df584 for eager: so101_left was made and connected before
    so101_right's harness refused (and `_abandon_build` leaves arms as they
    are). Every arm's harness is built in a pre-pass now."""
    from cascade.apps import demo

    with pytest.raises(ValueError, match=REFUSAL[bad]):
        demo.build_runtime(_pair(bad), tmp_path / "run", lazy_arm=lazy)
    assert _arm_calls(events) == []


# ── 3. valid configs: same objects, harnesses first ──────────────────────


@needs_pin
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_a_valid_rebot_build_builds_its_harness_before_its_arm(events, lazy):
    from cascade.apps import demo
    from cascade.control.lazy_arm import LazyArm

    cfg, acfg = _rebot()
    raw, safe, kin = demo._build_arm(acfg, lazy, None, cfg)
    if lazy:
        assert events == [("harness", None)]
        assert isinstance(raw, LazyArm) and not raw.connected
    else:
        assert events == [("harness", None), ("make", "rebot_rs"), ("connect", "rebot_rs")]
        assert raw._connected
    h = safe.harness
    assert h.kin is kin and h.base_pose is None and h.tool_volume is not None
    assert h.park_q == list(acfg.get("park_q"))
    assert h.limits.min_clearance_m == pytest.approx(0.03)        # margins untouched


@needs_pin_so101
def test_a_valid_eager_pair_builds_both_harnesses_then_each_arm_in_order(events, tmp_path):
    from cascade.apps import demo

    runtime, owner = demo.build_runtime(_pair(), tmp_path / "run", lazy_arm=False)
    try:
        assert events[:6] == [("harness", 0.22), ("harness", -0.22),
                              ("make", "so101_left"), ("connect", "so101_left"),
                              ("make", "so101_right"), ("connect", "so101_right")]
        assert _arm_calls(events[6:]) == []
        rig = runtime.arm_rig
        for name, y in (("so101_left", 0.22), ("so101_right", -0.22)):
            h = rig.get(name).harness
            assert h.base_pose[1, 3] == pytest.approx(y) and h.kin is not None
        # each arm got its OWN harness, and the primary is still the first arm
        assert rig.get("so101_left").harness is not rig.get("so101_right").harness
        assert runtime.arm.harness is rig.get("so101_left").harness
    finally:
        demo.shutdown_runtime(runtime, owner)


# ── 4. the tilted-mount refusal says which arm and how it is tilted ──────


@needs_pin_so101
def test_the_tilted_mount_refusal_names_the_arm_and_its_roll_and_pitch(events):
    from cascade.apps import demo

    cfg = _pair("tilted mount")
    cfg._data["arms"][1]["base_pose"] = [0.0, -0.22, 0.0, 0.1, -0.05, 0.4]
    acfg = demo._arm_cfgs(cfg)[1]
    with pytest.raises(ValueError) as err:
        demo._build_arm(acfg, False, None, cfg)
    msg = str(err.value)
    assert re.search(r"arm 'so101_right'.*roll 5\.73 deg, pitch -2\.86 deg", msg), msg
    assert "level base_pose" in msg and "safety.tool_volume" in msg, msg


@pytest.mark.parametrize(("rpy", "said"), [
    ((0.1, 0.0, 0.0), "roll 5.73 deg, pitch 0.00 deg"),
    ((0.0, -0.1, 0.0), "roll 0.00 deg, pitch -5.73 deg"),
    ((0.05, 0.05, 0.4), "roll 2.86 deg, pitch 2.86 deg"),          # yaw is not a tilt
    ((np.pi, 0.0, 0.0), "roll 180.00 deg, pitch 0.00 deg"),         # upside down
], ids=["roll", "pitch", "roll+pitch+yaw", "upside down"])
def test_the_harness_refusal_states_the_mounts_roll_and_pitch(rpy, said):
    """The harness alone (no arm name given) still states the tilt."""
    tool = ToolVolume(local=np.zeros((1, 3)), part=np.array([0]), names=("jaws",))
    limits = SafetyLimits(np.array([-1.0, -1.0, -0.2]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="level base_pose") as err:
        SafetyHarness(limits, base_pose=pose_to_transform([0.0, -0.22, 0.0, *rpy]), tool_volume=tool)
    assert said in str(err.value), str(err.value)
    assert "arm '" not in str(err.value)
