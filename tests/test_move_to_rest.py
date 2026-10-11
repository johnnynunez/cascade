"""`move_to_rest`: "home" / "pose 0" / "rest" fold the arm onto its rest pose,
verify it there by MEASUREMENT, and only then switch the motors off.

The reBot's owner uses two words for two poses:

- "initial pose" (pose inicial) = `home_q`, raised, motors on: `move_home`;
- "home" / "pose 0" / "rest" = `park_q`, the folded rest pose (every joint 0,
  links lying back over the base, wrist camera on top) from which the robot
  can be switched off: `move_to_rest`.

These tests run the real `build_runtime` wiring on the mock stack with a
LazyArm (the MCP server's construction) over a MockArm that behaves like the
physical arm in the two ways that matter here: a backend opened again reads
where the joints physically are (a real arm does not jump back to home when
the bus is reopened), and a joint can be stopped short of its command (an
obstruction, or a joint that sags onto something). The assertion that
matters most is the negative one: an arm that is NOT measured at rest keeps
its torque, because a raised arm without torque falls.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_pin

from cascade.control.mock_arm import MockArm

PARK = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _PhysicalArm(MockArm):
    """MockArm whose pose survives disconnect/reconnect, with an optional
    per-joint floor it cannot be driven below (`world["floor"]`)."""

    def __init__(self, cfg, kinematics, world):
        super().__init__(cfg, kinematics)
        self.world = world
        if world["q"] is not None:
            self._q = np.asarray(world["q"], dtype=float).copy()
        world["instances"].append(self)

    def connect(self):
        super().connect()
        self.world["events"].append("connect")

    def send_joint_target(self, q):
        q = np.asarray(q, dtype=float).copy()
        for joint, floor in self.world["floor"].items():
            q[joint] = max(q[joint], floor)
        super().send_joint_target(q)
        self.world["q"] = self._q.copy()

    def close_gripper_torque(self):
        self.world["events"].append("close_gripper")
        self._gripper = self._grip_closed

    def disconnect(self):
        self.world["q"] = self._q.copy()
        self.world["events"].append("disconnect")
        super().disconnect()


def _build(monkeypatch, tmp_path, *, park_q=PARK, arm_profile="mock"):
    from cascade.apps import demo
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="mock", arm=arm_profile, llm="mock")
    if park_q is not None:
        cfg._data["arm"]["park_q"] = list(park_q)
    # rebot_rs requires its live camera map (fail closed), and the suite pins
    # CASCADE_OCCUPANCY=0: that combination refuses startup by design. These
    # tests are about the rest pose and the motion envelope, so the map is
    # off here; the required map on rebot_rs is exercised end to end by
    # tests/test_rebot_occupancy_live.py.
    occ = cfg._data.get("occupancy")
    if isinstance(occ, dict) and occ.get("required"):
        occ["required"] = False
    world = {"q": None, "floor": {}, "instances": [], "events": []}
    # Every backend is a MockArm: the factory is replaced BEFORE the runtime
    # is built, so no real driver is ever constructed, whatever the profile.
    monkeypatch.setattr(demo, "make_arm",
                        lambda acfg, kinematics=None: _PhysicalArm(acfg, kinematics, world))
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", lazy_arm=True)
    return runtime, arm, world


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps.demo import shutdown_runtime

    runtime, arm, world = _build(monkeypatch, tmp_path)
    try:
        yield runtime, arm, world
    finally:
        shutdown_runtime(runtime, arm)


def _home(runtime):
    return np.asarray(runtime.cfg.arm.home_q, dtype=float)


def _wake(runtime, arm, world):
    """Test setup: open the arm the way any earlier motion would (the arm
    now holds the initial pose with torque on)."""
    runtime.arm.get_state()
    assert arm.connected
    np.testing.assert_allclose(world["q"] if world["q"] is not None else runtime.arm.get_state().q,
                               _home(runtime), atol=1e-9)


def _commands(world) -> int:
    return sum(len(a.commands) for a in world["instances"])


# ── the happy path ──────────────────────────────────────────────────────────


def test_folds_to_rest_verifies_and_switches_the_motors_off(rig):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is True, res
    assert res["at"] == "rest" and res["torque"] == "released" and res["safe_to_power_off"] is True
    assert res["moved"] is True
    np.testing.assert_allclose(world["q"], PARK, atol=1e-9)
    assert res["max_error_rad"] <= res["rest_tolerance_rad"]
    assert arm.connected is False, "the LazyArm must be back in standby (motors off)"
    assert world["events"][-2:] == ["close_gripper", "disconnect"], world["events"]
    assert not runtime.arm.harness.estopped


def test_the_fold_relaxes_the_joint_margin_only_on_its_last_step(rig, monkeypatch):
    """Joints 2/3 have their lower limit AT the rest pose: only the final
    step may drive onto the stop (joint_margin=0), like the shutdown park
    and scripts/home_rebot_rs.py; every earlier step keeps the margin."""
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    safe = runtime.arm
    calls = []
    real = safe.move_joints

    def spy(q, duration_s=2.0, joint_margin=None, **kw):
        calls.append((np.asarray(q, dtype=float).copy(), joint_margin))
        return real(q, duration_s=duration_s, joint_margin=joint_margin, **kw)

    monkeypatch.setattr(safe, "move_joints", spy)
    assert runtime.execute("move_to_rest", {})["ok"] is True
    margins = [m for _, m in calls]
    assert len(calls) >= 2, "the fold is streamed in several measured steps"
    assert margins[-1] == 0.0 and all(m is None for m in margins[:-1]), margins
    np.testing.assert_allclose(calls[-1][0], PARK, atol=1e-12)


def test_the_next_motion_switches_the_motors_back_on_from_rest(rig):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    assert runtime.execute("move_to_rest", {})["ok"] is True
    assert arm.connected is False
    res = runtime.execute("move_home", {})
    assert res["ok"] is True, res
    assert arm.connected is True and len(world["instances"]) == 2
    reopened = world["instances"][1]
    # the reopened backend started from where the arm physically was (rest)
    # and drove it up to the initial pose through the harness
    assert np.max(np.abs(reopened.commands[0] - np.asarray(PARK))) < 0.2
    np.testing.assert_allclose(world["q"], _home(runtime), atol=1e-9)


def test_from_another_pose_it_goes_through_the_initial_pose_first(rig):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    elsewhere = _home(runtime) + np.array([0.25, -0.15, 0.1, 0.0, 0.2, 0.0])
    assert runtime.arm.move_joints(elsewhere, duration_s=2.0)
    n0 = _commands(world)
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is True and res["at"] == "rest", res
    assert res["route"] == ["initial_pose", "fold"]
    cmds = [c for a in world["instances"] for c in a.commands][n0:]
    assert min(np.max(np.abs(c - _home(runtime))) for c in cmds) < 1e-9, "never reached home_q"
    np.testing.assert_allclose(world["q"], PARK, atol=1e-9)


def test_keep_torque_folds_but_leaves_the_motors_on(rig):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    res = runtime.execute("move_to_rest", {"keep_torque": True})
    assert res["ok"] is True and res["at"] == "rest", res
    assert res["torque"] == "on" and res["safe_to_power_off"] is False
    assert arm.connected is True and "disconnect" not in world["events"]
    np.testing.assert_allclose(world["q"], PARK, atol=1e-9)
    # asked again without keep_torque: already at rest, nothing moves, the
    # motors go off
    n0 = _commands(world)
    again = runtime.execute("move_to_rest", {})
    assert again["ok"] is True and again["moved"] is False and again["torque"] == "released"
    assert _commands(world) == n0 and arm.connected is False


@pytest.mark.parametrize("value, released", [(None, True), ("false", True), ("true", False)])
def test_keep_torque_accepts_what_chat_models_send(rig, value, released):
    """Chat hosts fill optional schema fields (null) or send booleans as
    strings; both mean what they say. Anything else is refused before any
    motion."""
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    res = runtime.execute("move_to_rest", {"keep_torque": value})
    assert res["ok"] is True and res["at"] == "rest", res
    assert (res["torque"] == "released") is released and arm.connected is (not released)


def test_a_nonsense_keep_torque_is_refused_before_any_motion(rig):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    n0 = _commands(world)
    res = runtime.execute("move_to_rest", {"keep_torque": "maybe"})
    assert res["ok"] is False and "keep_torque" in res["error"]
    assert _commands(world) == n0 and arm.connected is True


def test_a_standby_arm_is_not_woken_up(rig):
    """Never opened (or already released): the motors hold nothing, so
    nothing can drop -- and reading the pose would power them for nothing."""
    runtime, arm, world = rig
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is True, res
    assert res["torque"] == "released" and res["safe_to_power_off"] is True
    assert res["moved"] is False and res["at"] == "standby"
    assert world["instances"] == [] and arm.connected is False


# ── refusals: nothing moves, torque stays as it was ─────────────────────────


@pytest.mark.parametrize("marker", ["held_object", "_held_provisional"])
def test_refuses_while_holding_or_possibly_holding(rig, marker):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    setattr(runtime, marker, "red cube" if marker == "held_object" else object())
    n0 = _commands(world)
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is False and res["moved"] is False, res
    assert "put it down" in res["error"]
    assert res["torque"] == "on" and res["safe_to_power_off"] is False
    assert _commands(world) == n0 and arm.connected is True
    assert "close_gripper" not in world["events"]


def test_refuses_when_the_e_stop_is_latched(rig):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    runtime.arm.harness.estop("test")
    n0 = _commands(world)
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is False and res["moved"] is False, res
    assert res["refused"] == "estopped" and "e-stop" in res["error"]
    assert res["safe_to_power_off"] is False
    assert _commands(world) == n0 and arm.connected is True
    assert "disconnect" not in world["events"]


def test_an_e_stopped_standby_arm_is_not_woken_even_to_keep_torque(rig):
    """The latch means "do not move": reading the pose of a standby arm
    would open it (on the RS: a stop frame, then enable) before the harness
    ever saw a motion. The refusal comes first."""
    runtime, arm, world = rig
    runtime.arm.harness.estop("test")
    res = runtime.execute("move_to_rest", {"keep_torque": True})
    assert res["ok"] is False and res["refused"] == "estopped", res
    assert world["instances"] == [] and arm.connected is False


def test_an_e_stop_latched_after_the_fold_keeps_the_motors_on(rig, monkeypatch):
    """The latch means "do not move" -- not even the gripper close that
    precedes the release. The arm is at rest, but nothing more is done."""
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    safe = runtime.arm
    real = safe.move_joints

    def spy(q, duration_s=2.0, joint_margin=None, **kw):
        out = real(q, duration_s=duration_s, joint_margin=joint_margin, **kw)
        if joint_margin == 0.0:                     # the last fold step landed
            safe.harness.estop("operator pressed stop")
        return out

    monkeypatch.setattr(safe, "move_joints", spy)
    res = runtime.execute("move_to_rest", {})
    assert res["at"] == "rest" and res["torque"] == "on" and res["safe_to_power_off"] is False, res
    assert "e-stop" in res["note"]
    assert arm.connected is True and "close_gripper" not in world["events"]


def test_a_refused_fold_never_starts(rig):
    """The whole route is vetted with the harness's static API before the
    first command: a fold the harness would refuse half-way does not start
    (the arm is not left somewhere between poses)."""
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    harness = runtime.arm.harness
    real = harness.vet_step

    def vet(q_prev, q_next, dt, joint_margin=None):
        if q_next[1] < 0.5:
            return "test obstacle over the base"
        return real(q_prev, q_next, dt, joint_margin=joint_margin)

    harness.vet_step = vet
    n0 = _commands(world)
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is False and res["moved"] is False, res
    assert "test obstacle over the base" in res["error"]
    assert res["torque"] == "on" and _commands(world) == n0 and arm.connected is True


def test_a_refused_fold_does_not_even_start_the_way_up(rig):
    """From another pose the route is up to the initial pose, then the fold.
    The fold is vetted BEFORE the way up starts: a fold that will be refused
    must not first drive the arm up to home and leave it there."""
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    elsewhere = _home(runtime) + np.array([0.25, -0.15, 0.1, 0.0, 0.2, 0.0])
    assert runtime.arm.move_joints(elsewhere, duration_s=2.0)
    harness = runtime.arm.harness
    real = harness.vet_step

    def vet(q_prev, q_next, dt, joint_margin=None):
        if q_next[1] < 0.5:
            return "test obstacle over the base"
        return real(q_prev, q_next, dt, joint_margin=joint_margin)

    harness.vet_step = vet
    n0 = _commands(world)
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is False and res["moved"] is False, res
    assert "test obstacle over the base" in res["error"]
    assert _commands(world) == n0, "the arm went up before the fold was vetted"
    np.testing.assert_allclose(world["q"], elsewhere, atol=1e-9)


# ── the guard: torque is released ONLY when measured at rest ────────────────


@pytest.mark.parametrize("floor", [
    {2: 0.06},   # joint 3 stops 0.06 rad short: every step tracks, the rest check decides
    {1: 0.30},   # joint 2 blocked well above its stop (an obstruction)
], ids=["sags_short_of_rest", "blocked_on_the_way"])
def test_an_arm_not_measured_at_rest_keeps_its_torque(rig, floor):
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    world["floor"] = floor
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is False, res
    assert res["at"] != "rest" and res["torque"] == "on" and res["safe_to_power_off"] is False
    assert arm.connected is True, "a raised arm without torque falls"
    assert "disconnect" not in world["events"] and "close_gripper" not in world["events"]
    assert res["measured_q"] is not None and res["max_error_rad"] > res["rest_tolerance_rad"]


def test_a_step_that_does_not_track_stops_the_fold_there(rig, monkeypatch):
    """Each fold step is MEASURED against its target before the next one is
    sent (scripts/home_rebot_rs.py's tracking check): joint 2 blocked at
    0.75 rad misses step 2 (target 0.6) by 0.15 rad, so steps 3 and 4 are
    never commanded and the motors stay on."""
    runtime, arm, world = rig
    _wake(runtime, arm, world)
    world["floor"] = {1: 0.75}
    safe = runtime.arm
    targets = []
    real = safe.move_joints

    def spy(q, duration_s=2.0, joint_margin=None, **kw):
        targets.append(np.asarray(q, dtype=float).copy())
        return real(q, duration_s=duration_s, joint_margin=joint_margin, **kw)

    monkeypatch.setattr(safe, "move_joints", spy)
    res = runtime.execute("move_to_rest", {})
    assert res["ok"] is False and res["torque"] == "on", res
    assert "fold step 2/4" in res["error"]
    assert len(targets) == 2, [t.round(3).tolist() for t in targets]
    assert arm.connected is True and "disconnect" not in world["events"]


# ── the shipped reBot RS profile: its own park_q, speed cap, camera-up rule ─


@needs_pin
def test_the_rebot_rs_profile_folds_home_to_rest_through_its_own_harness(monkeypatch, tmp_path):
    """The real profile's safety envelope (0.8 rad/s cap, the wrist camera
    kept on top, joint margins) with its own park_q, on a MockArm backend:
    initial pose -> rest passes every gate, ends exactly at park_q and
    releases; the harness records no violation."""
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    profile = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    assert profile.safety.get("wrist_camera_up") is not None
    runtime, arm, world = _build(monkeypatch, tmp_path, park_q=None, arm_profile="rebot_rs")
    try:
        harness = runtime.arm.harness
        np.testing.assert_allclose(harness.park_q, profile.arm.park_q)
        assert harness.limits.max_joint_vel == pytest.approx(0.8)
        _wake(runtime, arm, world)
        res = runtime.execute("move_to_rest", {})
        assert res["ok"] is True and res["at"] == "rest" and res["torque"] == "released", res
        np.testing.assert_allclose(world["q"], profile.arm.park_q, atol=1e-9)
        assert harness.violations == []
        assert runtime.execute("move_home", {})["ok"] is True        # and back up
        np.testing.assert_allclose(world["q"], profile.arm.home_q, atol=1e-9)
    finally:
        shutdown_runtime(runtime, arm)


def test_an_arm_profile_without_a_rest_pose_goes_to_the_initial_pose_and_keeps_torque(
        monkeypatch, tmp_path):
    """No `park_q` -> no folded pose is known for this arm, and none is
    guessed: the request falls back to the initial pose with the motors on,
    and says so (never safe_to_power_off). A held object is carried along,
    as "go home" (move_home) did on these arms before move_to_rest existed:
    nothing is released, so there is nothing to refuse."""
    from cascade.apps.demo import shutdown_runtime

    runtime, arm, world = _build(monkeypatch, tmp_path, park_q=None)
    try:
        _wake(runtime, arm, world)
        runtime.arm.move_joints(_home(runtime) + np.array([0.2, 0, 0, 0, 0, 0]), duration_s=2.0)
        res = runtime.execute("move_to_rest", {})
        assert res["ok"] is True, res
        assert res["at"] == "initial" and res["torque"] == "on" and res["safe_to_power_off"] is False
        assert res["rest_pose"] is None and "park_q" in res["note"]
        np.testing.assert_allclose(world["q"], _home(runtime), atol=1e-9)
        assert arm.connected is True
        runtime.arm.move_joints(_home(runtime) + np.array([0.2, 0, 0, 0, 0, 0]), duration_s=2.0)
        runtime.held_object = "red cube"
        carried = runtime.execute("move_to_rest", {})
        assert carried["ok"] is True and carried["at"] == "initial", carried
        np.testing.assert_allclose(world["q"], _home(runtime), atol=1e-9)
    finally:
        runtime.held_object = None
        shutdown_runtime(runtime, arm)
