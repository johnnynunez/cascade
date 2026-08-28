"""The arm rig: N arms behind one handle, each with its OWN safety envelope.

The rig supported N cameras and exactly one arm. These tests pin the
invariants that make a second arm safe to add, in the order they would bite:

1. single-arm behaviour is unchanged (the rig is not a rewrite),
2. two arms do not share a safety envelope (the merge bug this design exists
   to prevent),
3. teardown reaches EVERY arm (a missed disconnect leaves torque on),
4. the rig never materializes a LazyArm as a side effect (powering motors to
   render a status panel).
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.control.arm_rig import ArmRig


class _FakeArm:
    """Minimal ArmBase-shaped double; records lifecycle calls."""

    def __init__(self, name="a", n_joints=6, fail_connect=False):
        self.name = name
        self.n_joints = n_joints
        self.connected = False
        self.disconnects = 0
        self.stops = 0
        self._fail = fail_connect

    def connect(self):
        if self._fail:
            raise RuntimeError(f"{self.name} refused to connect")
        self.connected = True

    def disconnect(self):
        self.disconnects += 1
        self.connected = False

    def stop(self):
        self.stops += 1

    def get_state(self):
        from cascade.types import RobotState

        return RobotState(q=np.zeros(self.n_joints), t=0.0)


# ── shape: the CameraRig contract, for arms ──────────────────────────────


def test_first_arm_is_primary_and_get_none_returns_it():
    a, b = _FakeArm("left"), _FakeArm("right")
    rig = ArmRig([a, b], ["left", "right"])
    assert rig.primary is a
    assert rig.get(None) is a          # skills that omit `arm` hit the primary
    assert rig.get("right") is b
    assert rig.names == ["left", "right"]
    assert len(rig) == 2
    assert list(rig) == [a, b]


def test_unknown_arm_name_raises_with_the_available_ones():
    rig = ArmRig([_FakeArm("left")], ["left"])
    with pytest.raises(KeyError, match="left"):
        rig.get("nope")


def test_duplicate_names_are_rejected():
    """Names address arms; two 'left' arms make `get` silently ambiguous."""
    with pytest.raises(ValueError, match="duplicate"):
        ArmRig([_FakeArm(), _FakeArm()], ["left", "left"])


def test_empty_rig_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ArmRig([], [])


def test_mismatched_names_are_rejected():
    with pytest.raises(ValueError, match="positional"):
        ArmRig([_FakeArm(), _FakeArm()], ["only-one"])


# ── lifecycle ────────────────────────────────────────────────────────────


def test_failed_connect_unwinds_the_arms_already_connected():
    """A half-connected rig leaves a powered arm the caller does not know to
    tear down -- for a real arm, torque on and unsupervised."""
    good, bad = _FakeArm("left"), _FakeArm("right", fail_connect=True)
    rig = ArmRig([good, bad], ["left", "right"])
    with pytest.raises(RuntimeError, match="refused"):
        rig.connect()
    assert good.disconnects == 1
    assert not good.connected


def test_disconnect_reaches_every_arm_even_when_one_raises():
    class Angry(_FakeArm):
        def disconnect(self):
            super().disconnect()
            raise RuntimeError("bus error")

    a, b = Angry("left"), _FakeArm("right")
    ArmRig([a, b], ["left", "right"]).disconnect()
    assert a.disconnects == 1 and b.disconnects == 1


def test_stop_reaches_every_arm_even_when_one_raises():
    """E-stop is the one path that must never short-circuit: a failure
    stopping arm 1 cannot leave arm 2 running."""

    class Angry(_FakeArm):
        def stop(self):
            super().stop()
            raise RuntimeError("CAN down")

    a, b = Angry("left"), _FakeArm("right")
    ArmRig([a, b], ["left", "right"]).stop()
    assert a.stops == 1 and b.stops == 1


def test_stats_does_not_materialize_a_lazy_arm():
    """Reading state off an unmaterialized LazyArm powers the motors. The
    dashboard calls stats() on a timer, so this must stay a pure read of
    LazyArm's own surface (`connected`)."""

    class Trap(_FakeArm):
        connected = False

        def get_state(self):
            raise AssertionError("stats() must not touch an unconnected arm")

    stats = ArmRig([Trap("left")], ["left"]).stats()
    assert stats["left"]["connected"] is False
    assert "q" not in stats["left"]


def test_rig_lifecycle_works_on_the_real_SafeArm_surface():
    """Regression: the rig holds SafeArms, but SafeArm exposed no
    connect/disconnect -- so teardown of a two-arm rig failed for EVERY arm
    ('SafeArm object has no attribute disconnect') and left the motors
    powered. The fake doubles above hid it by being more capable than the
    real object, so this test asserts against SafeArm itself.
    """
    from cascade.config import load_demo_config as _ldc
    from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits

    backing = [_FakeArm("left"), _FakeArm("right")]
    harness = SafetyHarness(SafetyLimits.from_config(_ldc().safety))
    rig = ArmRig([SafeArm(a, harness) for a in backing], ["left", "right"])

    rig.connect()
    assert all(a.connected for a in backing)
    rig.stop()
    assert all(a.stops == 1 for a in backing)
    rig.disconnect()
    assert all(a.disconnects == 1 for a in backing)
    assert not any(a.connected for a in backing)


# ── the reason this design exists: per-arm safety ────────────────────────


def test_single_arm_config_is_unchanged_by_the_rig():
    """`arms` defaults to [arm]: the primary's view IS the top level."""
    cfg = load_demo_config(arm="so101_mock")
    assert len(cfg.arms) == 1
    assert cfg.arm.n_joints == 5
    # the arm's own resolved view agrees with the global one
    resolved = cfg.arms[0]["resolved"]
    assert resolved["safety"]["workspace"] == cfg.safety.workspace.as_dict()


def test_two_arms_keep_separate_safety_envelopes():
    """The bug this prevents: deep-merging every arm's `overrides:` into one
    global config, so the LAST profile loaded silently defines the safety
    envelope for BOTH arms.

    The left arm owns y >= -0.02, the right owns y <= 0.02. Merged, both
    would read whichever profile was loaded last.
    """
    cfg = load_demo_config(arms=["so101_left", "so101_right"])
    left, right = cfg.arms[0]["resolved"], cfg.arms[1]["resolved"]

    assert left["safety"]["workspace"]["min"][1] == pytest.approx(-0.02)
    assert left["safety"]["workspace"]["max"][1] == pytest.approx(0.36)
    assert right["safety"]["workspace"]["min"][1] == pytest.approx(-0.36)
    assert right["safety"]["workspace"]["max"][1] == pytest.approx(0.02)
    # the primary still defines the top level (single-arm behaviour)
    assert cfg.safety.workspace.min[1] == pytest.approx(-0.02)
    # each arm's keep-out is the OTHER's half, not a shared blob
    assert left["safety"]["keep_out"][0]["max"][1] == pytest.approx(-0.02)
    assert right["safety"]["keep_out"][0]["min"][1] == pytest.approx(0.02)
    # and the drop zones sit on opposite sides of the centre line
    assert left["grasp"]["drop_zone"][1] > 0 > right["grasp"]["drop_zone"][1]


def test_each_arms_resolved_view_describes_its_own_robot():
    """`resolved.arm` must be the arm that owns the view, not its neighbour's
    -- otherwise a per-arm harness would be built from the wrong profile."""
    cfg = load_demo_config(arms=["so101_left", "so101_right"])
    assert cfg.arms[0]["resolved"]["arm"]["name"] == "so101_left"
    assert cfg.arms[1]["resolved"]["arm"]["name"] == "so101_right"
    # and no arm's view nests another arm's view
    assert "resolved" not in cfg.arms[1]["resolved"]["arm"]


def test_repeating_one_profile_yields_distinct_arm_names():
    """Two of the same robot is a legitimate dual-arm setup; the rig needs
    two addressable names for it."""
    cfg = load_demo_config(arms=["so101_mock", "so101_mock"])
    assert [a["name"] for a in cfg.arms] == ["so101_mock0", "so101_mock1"]
