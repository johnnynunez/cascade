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
    global config, so the LAST profile loaded silently defines the envelope
    for BOTH arms.

    The dual profiles share a workspace on purpose (they must reach the same
    table to hand over), so the thing that distinguishes them is `base_pose`
    -- and that is exactly what the inter-arm check depends on. Merged, both
    arms would report the same mounting position and every inter-arm distance
    would be wrong.
    """
    cfg = load_demo_config(arms=["so101_left", "so101_right"])
    left, right = cfg.arms[0], cfg.arms[1]

    assert left["base_pose"][1] == pytest.approx(0.22)
    assert right["base_pose"][1] == pytest.approx(-0.22)
    # each arm's resolved view carries its OWN clearance, not a shared blob
    assert left["resolved"]["safety"]["neighbor_clearance_m"] == pytest.approx(0.05)
    assert right["resolved"]["safety"]["neighbor_clearance_m"] == pytest.approx(0.05)
    # the primary still defines the top level (single-arm behaviour)
    assert cfg.safety.get("neighbor_clearance_m") == pytest.approx(0.05)


def test_an_arms_overrides_do_not_leak_into_its_neighbours_view():
    """Per-arm isolation, on profiles that genuinely differ.

    so101.yaml is the only shipped profile with an `overrides:` block (it
    retunes workspace/grasp for a printed 5-DoF arm). Pairing it with the
    Panda -- which has none, so it should see demo.yaml's defaults -- is the
    case that actually catches a leak: if overrides were merged globally, the
    Panda's view would silently inherit the SO-101's 0.05 m grasp ceiling.
    That exact class of bug cost two LIBERO benchmark runs.
    """
    cfg = load_demo_config(arms=["so101_mock", "libero_panda"])
    so101, panda = cfg.arms[0]["resolved"], cfg.arms[1]["resolved"]

    assert so101["grasp"]["topdown_z_max"] != panda["grasp"]["topdown_z_max"]
    assert so101["safety"]["workspace"] != panda["safety"]["workspace"]
    # the arm WITHOUT overrides sees the untouched demo.yaml defaults
    import yaml

    from cascade.config import CONFIG_DIR

    base = yaml.safe_load((CONFIG_DIR / "demo.yaml").read_text())
    assert panda["safety"]["workspace"] == base["safety"]["workspace"]
    assert panda["grasp"]["topdown_z_max"] == base["grasp"]["topdown_z_max"]
    # and the primary's values are the ones at the top level
    assert cfg.grasp.topdown_z_max == so101["grasp"]["topdown_z_max"]


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


# ── addressing an arm from a skill call ──────────────────────────────────


class _RuntimeStub:
    """SkillRuntime with only the arm-selection machinery wired.

    Built with __new__ on purpose: it is how several existing tests build
    partial runtimes, and the `arm` property has to survive it (the class
    defaults exist for exactly this).
    """

    def __new__(cls, rig=None, primary=None):
        from cascade.skills.runtime import SkillRuntime

        rt = SkillRuntime.__new__(SkillRuntime)
        rt._arm = primary
        rt.arm_rig = rig
        return rt


def test_arm_defaults_to_primary_and_override_is_scoped():
    import threading as _t

    from cascade.skills.runtime import SkillRuntime

    a, b = _FakeArm("left"), _FakeArm("right")
    rt = _RuntimeStub(ArmRig([a, b], ["left", "right"]), a)
    rt._arm_override = _t.local()

    assert rt.arm is a                       # default: the primary
    rt._arm_override.arm = b
    assert rt.arm is b                       # inside a call with arm="right"
    rt._arm_override.arm = None
    assert rt.arm is a                       # and it does not stick

    assert isinstance(SkillRuntime.arm, property)


def test_unknown_arm_name_is_a_skill_error_not_a_silent_default():
    """Falling back to the primary would run the motion on the WRONG robot --
    on a shared table that is a collision, not a wrong answer."""
    from cascade.types import SkillError

    a = _FakeArm("left")
    rt = _RuntimeStub(ArmRig([a], ["left"]), a)
    with pytest.raises(SkillError, match="no arm"):
        rt._select_arm("right")


def test_naming_an_arm_without_a_rig_is_rejected():
    """A single-arm runtime must not accept `arm="whatever"` and quietly
    drive the only arm it has."""
    from cascade.types import SkillError

    rt = _RuntimeStub(None, _FakeArm("only"))
    with pytest.raises(SkillError, match="single arm"):
        rt._select_arm("other")
    assert rt._select_arm(None) is None       # omitted = primary, always fine


def test_every_motion_skill_exposes_an_optional_arm_parameter():
    """The schema is what makes the capability reachable by the LLM: a skill
    the model cannot address is a skill it will never use. `arm` must stay
    OPTIONAL, or every single-arm call becomes invalid."""
    from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS

    by_name = {s["name"]: s for s in TOOL_SPECS}
    for skill in _MOTION_SKILLS:
        params = by_name[skill]["parameters"]
        assert "arm" in params["properties"], f"{skill} cannot be addressed"
        assert "arm" not in params.get("required", []), \
            f"{skill} would reject single-arm calls"


def test_non_motion_skills_do_not_advertise_an_arm():
    """`describe_scene` does not move anything; offering an `arm` there is
    noise in the tool schema the model has to reason about."""
    from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS

    for spec in TOOL_SPECS:
        if spec["name"] not in _MOTION_SKILLS:
            assert "arm" not in spec["parameters"]["properties"], spec["name"]


def test_list_arms_has_a_tool_spec_and_a_method():
    """A skill without a TOOL_SPECS entry fails no existing test and is
    simply invisible to the LLM and MCP -- the documented silent failure."""
    from cascade.skills.runtime import TOOL_SPECS, SkillRuntime

    assert "list_arms" in {s["name"] for s in TOOL_SPECS}
    assert callable(getattr(SkillRuntime, "skill_list_arms"))


def test_arm_argument_moves_that_arm_and_only_that_arm(tmp_path):
    """The end-to-end claim, on the real pipeline: `arm="..."` drives the
    named robot, leaves the other one untouched, and omitting it still
    drives the primary. Everything above this is machinery; this is the
    behaviour a user would notice being wrong.
    """
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(
        arms=["so101_left", "so101_right"], camera="mock", llm="mock"
    )
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        left = runtime.arm_rig.get("so101_left")
        right = runtime.arm_rig.get("so101_right")
        home = np.asarray(cfg.arm.home_q, dtype=float)

        # Nudge both off home. wrist_flex only: a base-yaw offset would push
        # the TCP across the centre line and the partition would (correctly)
        # reject it, which is a different test.
        off = home.copy()
        off[3] += 0.15
        assert left.move_joints(off, duration_s=0.4)
        assert right.move_joints(off, duration_s=0.4)
        parked = left.get_state().q.copy()

        assert runtime.execute("move_home", {"arm": "so101_right"})["ok"]
        assert np.allclose(right.get_state().q, home, atol=2e-2)
        assert np.allclose(left.get_state().q, parked, atol=1e-9)

        # omitted `arm` still means the primary
        assert runtime.execute("move_home", {})["ok"]
        assert np.allclose(left.get_state().q, home, atol=2e-2)

        # and the override never leaks past the call that set it
        assert runtime.arm is left
    finally:
        shutdown_runtime(runtime, arm)


def test_unknown_arm_name_reaches_the_agent_as_a_failed_result(tmp_path):
    """execute() never raises: a bad arm name has to come back as a normal
    {"ok": false} the agent can reason about, WITHOUT having moved anything.
    """
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(
        arms=["so101_left", "so101_right"], camera="mock", llm="mock"
    )
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        before = [a.get_state().q.copy() for a in runtime.arm_rig]
        result = runtime.execute("move_home", {"arm": "third_arm"})
        assert result["ok"] is False
        assert "third_arm" in result["error"]
        for arm_obj, q0 in zip(runtime.arm_rig, before):
            assert np.allclose(arm_obj.get_state().q, q0)
    finally:
        shutdown_runtime(runtime, arm)


# ── inter-arm collision: the real geometric check ────────────────────────


def test_pose_to_transform_matches_the_urdf_rpy_convention():
    """base_pose is copied off a URDF/tape measure, so the rotation order has
    to be the one those use (Z@Y@X extrinsic). A silently different order
    would make every inter-arm distance confidently wrong."""
    from cascade.types import pose_to_transform

    T = pose_to_transform([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    assert np.allclose(T[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)

    T = pose_to_transform([1.0, 2.0, 3.0])          # position-only is allowed
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(T[:3, :3], np.eye(3))

    with pytest.raises(ValueError):
        pose_to_transform([1.0, 2.0])


def test_single_arm_has_no_neighbors_and_no_base_pose():
    """The whole feature must cost a single-arm rig exactly nothing: no
    base_pose means the base frame IS the table frame, and no neighbours
    means the check returns before touching kinematics."""
    from cascade.apps.demo import _base_transform

    cfg = load_demo_config(arm="so101_mock")
    assert _base_transform(cfg.arm) is None


def test_neighbor_check_rejects_a_waypoint_that_closes_on_the_other_arm(tmp_path):
    """The claim this whole change exists to make: with both arms bolted
    0.44 m apart (so they SHARE table), a motion that swings one arm into the
    other is aborted mid-stream by measured link distance -- not by a static
    keep-out wall, and not only when the TCP leaves a box.

    The colliding pose is base-yaw toward the neighbour with the arm EXTENDED
    (shoulder/elbow straightened): measured 0.9 mm between link centrelines,
    against 5.8 cm when merely yawed at the home posture. Yaw alone no longer
    trips the 0.05 m gate now that the check measures true segment distance
    instead of joint-origin distance -- which is the point: the earlier
    version needed a 0.10 m margin to compensate for what it could not see.
    """
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.types import SafetyViolation

    cfg = load_demo_config(
        arms=["so101_left", "so101_right"], camera="mock", llm="mock"
    )
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        left = runtime.arm_rig.get("so101_left")
        right = runtime.arm_rig.get("so101_right")
        assert list(left.harness._neighbors) == ["so101_right"]
        assert list(right.harness._neighbors) == ["so101_left"]

        home = np.asarray(cfg.arm.home_q, dtype=float)
        # Extended and yawed inward: 0.9 mm apart, a real collision.
        inward_left = home.copy()
        inward_left[:3] = [1.57, 0.0, 0.0]
        inward_right = home.copy()
        inward_right[:3] = [-1.57, 0.0, 0.0]

        # neighbour parked at home: the same motion is fine
        assert left.move_joints(inward_left, duration_s=0.5)
        assert left.move_joints(home, duration_s=0.5)

        # neighbour swung into the shared band: now it must abort
        assert right.move_joints(inward_right, duration_s=0.5)
        with pytest.raises(SafetyViolation, match="inter-arm clearance"):
            left.move_joints(inward_left, duration_s=0.5)

        # and the arm can always retreat -- a gate that cannot be escaped
        # would strand both robots mid-demo
        assert left.move_joints(home, duration_s=0.5)
    finally:
        shutdown_runtime(runtime, arm)


def test_unreadable_neighbor_is_skipped_not_blocking():
    """Booth rule, shared with the occupancy map: "cannot read" degrades to
    "skip", never to "blocked". A standby LazyArm must not freeze the rig --
    and reading its joints to find out would power the motors.
    """
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    cfg = load_demo_config(arm="so101_mock")
    h = SafetyHarness(SafetyLimits.from_config(cfg.safety))
    h.add_neighbor("standby", lambda: None)
    assert h._neighbor_violation(np.zeros(5)) is None

    h.add_neighbor("broken", lambda: (_ for _ in ()).throw(RuntimeError("bus")))
    assert h._neighbor_violation(np.zeros(5)) is None


def test_default_and_empty_arm_names_mean_the_primary():
    """Measured on a fresh clone: the chat model sent arm="" and then
    arm="default" (the name list_arms reports for one arm) and both were
    refused, costing two failed tool calls per pick. They mean 'primary';
    a wrong name must still be refused."""
    from cascade.types import SkillError

    a = _FakeArm("left")
    rt = _RuntimeStub(ArmRig([a], ["left"]), a)
    assert rt._select_arm("") is None
    assert rt._select_arm("default") is None
    assert rt._select_arm("Primary") is None
    with pytest.raises(SkillError, match="no arm"):
        rt._select_arm("right")
    rt_single = _RuntimeStub(None, _FakeArm("only"))
    assert rt_single._select_arm("default") is None
    with pytest.raises(SkillError, match="single arm"):
        rt_single._select_arm("other")
