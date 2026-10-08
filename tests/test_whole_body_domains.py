"""Whole-body composition contract (backlog B30): an arm mounted on a mobile base.

Mock-only software evidence. A kinematic mock base or arm can REFUTE a
world-frame outcome but never confirm a physical one. Nothing here is
whole-body control, balance or arm physics; physical mounted compositions stay
refused. See docs/ROBOT_MODULARITY.md -> "Multi-domain embodiments".
"""
import dataclasses
import hashlib
import json
import math
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import needs_pin
from cascade.config import load_robot_config
from cascade.control.mobile_base import BaseState

REPO = Path(__file__).resolve().parents[1]
PROFILE = "mobile_manipulator_mock"
HALF = math.sqrt(0.5)
MOUNT_T = (0.05, 0.0, 0.10)
MOUNT_Q = (HALF, 0.0, 0.0, HALF)  # arm base yawed +90 deg on the base


@pytest.fixture(autouse=True)
def _private_environment(monkeypatch):
    # Private port block of this backlog item; never the shared 5556/5557/8611.
    for key, port in (("CASCADE_GRASPGENX_PORT", "43501"), ("CASCADE_OCCUPANCY_PORT", "43502"),
                      ("CASCADE_BRIDGE_PORT", "43503")):
        monkeypatch.setenv(key, port)
    import os
    for key in list(os.environ):
        if key.startswith("CASCADE_MICRODUCK_") or key in {"CASCADE_ROBOT", "CASCADE_BASE"}:
            monkeypatch.delenv(key, raising=False)


def _quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pose(position, quaternion):
    value = np.eye(4)
    value[:3, :3] = _quat_matrix(quaternion)
    value[:3, 3] = position
    return value


MOUNT = _pose(MOUNT_T, MOUNT_Q)


def _state(t, x=0.0, y=0.0, yaw=0.0, *, epoch="epoch-a", age=0.0, status="ready", vx=0.0, fallen=False):
    return BaseState(robot_id="unit-base", source="unit", epoch=epoch, step=int(t * 100), sim_time_s=t,
                     received_monotonic_s=t + age, producer_age_s=age, position_world=(x, y, 0.2),
                     orientation_wxyz=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                     linear_velocity_world=(vx, 0.0, 0.0), angular_velocity_body=(0.0, 0.0, 0.0),
                     joint_names=(), joint_positions=(), joint_velocities=(), controller_status=status,
                     generation=0, contacts=(), fallen=fallen, latched=False, measurement_kind="kinematic_mock")


def _variant(tmp_path, mutate, *, name=PROFILE, extra=None):
    """Private copy of configs/ with one mutated robot profile (and extra files)."""
    cdir = tmp_path / "configs"
    if not cdir.exists():
        shutil.copytree(REPO / "configs", cdir)
    for relative, data in (extra or {}).items():
        (cdir / relative).write_text(yaml.safe_dump(data, sort_keys=False))
    source = REPO / "configs" / "robots" / f"{PROFILE}.yaml"
    data = yaml.safe_load(source.read_text()) if source.exists() else _fallback_profile()
    mutate(data)
    (cdir / "robots" / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return cdir


def _fallback_profile():
    # Used only while proving RED against code that predates the profile.
    return {"version": 1, "robot_id": PROFILE,
            "whole_body": {"version": 1, "coordination": "exclusive", "max_mount_drift_m": 0.005,
                           "max_mount_drift_rad": 0.01, "world_target_tolerance_m": 0.01},
            "domains": {"locomotion": {"kind": "locomotion", "bases": ["microduck_mock"]},
                        "manipulation": {"kind": "manipulation", "offline": True, "arms": ["mock"],
                                         "cameras": ["mock"],
                                         "mounted_on": {"domain": "locomotion", "base": "microduck_mock",
                                                        "translation_m": list(MOUNT_T),
                                                        "rotation_wxyz": list(MOUNT_Q),
                                                        "calibration_id": f"{PROFILE}/declared-mount-v1"}}}}


# ── declaration, endpoints and tool surface (no runtime is built) ─────────────

def test_mounted_profile_declares_disjoint_domains_and_namespaced_tools():
    from cascade.apps.robot_runtime import describe_robot, robot_tool_descriptors
    cfg = load_robot_config(PROFILE)
    contract = cfg.as_dict()["whole_body"]
    assert contract["coordination"] == "exclusive"
    (mount,) = contract["mounts"]
    assert (mount["arm_domain"], mount["base_domain"], mount["base"]) == ("manipulation", "locomotion", "microduck_mock")
    assert mount["translation_m"] == list(MOUNT_T) and mount["rotation_wxyz"] == pytest.approx(MOUNT_Q)
    # Freshness defaults to the base profile's own validated state-age limit.
    assert mount["max_base_pose_age_s"] == 0.5
    domains = describe_robot(cfg)
    owners = {r.controller_id: name for name, d in domains.items() for r in d.resources if r.controller_id}
    assert sorted(owners.values()) == ["locomotion", "manipulation"] and len(owners) == 2
    tools = robot_tool_descriptors(cfg)
    reach, pose = tools["manipulation.reach_world_point"], tools["manipulation.get_arm_world_pose"]
    assert reach.effect == "motion" and reach.writes == ("manipulation/mock",)
    assert "locomotion/microduck_mock" in reach.requires and "locomotion/microduck_mock" in pose.requires
    assert pose.effect == "read" and not pose.writes
    assert {"locomotion.walk_velocity", "locomotion.turn", "manipulation.pick_and_place",
            "manipulation.move_relative", "emergency_stop", "list_resources"} <= set(tools)
    assert not {"reach_world_point", "get_arm_world_pose", "walk_velocity"} & set(tools)
    reset = tools["reset_stop"].as_spec()["parameters"]
    assert reset["required"] == ["domain"]
    assert reset["properties"]["domain"]["enum"] == ["locomotion", "manipulation"]


def test_mcp_catalog_lists_whole_body_tools_without_building_a_runtime(monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer
    monkeypatch.setenv("CASCADE_ROBOT", PROFILE)
    server = McpSkillServer()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("catalog opened a runtime")
    monkeypatch.setattr(server, "_ensure_runtime", forbidden)
    listed = {t["name"]: t for t in server.list_tools()}
    assert {"manipulation.reach_world_point", "manipulation.get_arm_world_pose",
            "locomotion.walk_velocity", "reset_stop", "emergency_stop"} <= set(listed)
    assert listed["reset_stop"]["inputSchema"]["required"] == ["domain"]
    info = json.loads(server.call_tool("list_resources", {})["content"][-1]["text"])
    assert info["ok"] and info["whole_body"]["coordination"] == "exclusive"
    assert info["whole_body"]["mounts"][0]["base"] == "microduck_mock"
    assert info["whole_body"]["physical_admission"] is False
    assert server._runtime is None


def test_endpoint_overlap_names_both_claimants():
    from cascade.robotics.contracts import ResourceDescriptor
    from cascade.robotics.whole_body import check_disjoint_endpoints

    class Domain:
        def __init__(self, name, resources):
            self.domain_id, self.resources, self.motion_skills = name, resources, frozenset({"move"})
    shared = "ros:0:/lowcmd"
    arm = Domain("manipulation", (ResourceDescriptor("manipulation/left", "arm", "body", controller_id=shared,
                                                     writer_id="manipulation/left"),))
    base = Domain("locomotion", (ResourceDescriptor("locomotion/legs", "base", "body", controller_id=shared,
                                                    writer_id="locomotion/legs"),))
    with pytest.raises(ValueError) as error:
        check_disjoint_endpoints({"manipulation": arm, "locomotion": base})
    message = str(error.value)
    assert "manipulation/left" in message and "locomotion/legs" in message and shared in message
    distinct = Domain("locomotion", (ResourceDescriptor("locomotion/legs", "base", "body", controller_id="ros:0:/legs",
                                                        writer_id="locomotion/legs"),))
    check_disjoint_endpoints({"manipulation": arm, "locomotion": distinct})


def test_mounted_composition_refuses_one_endpoint_for_two_domains(tmp_path, monkeypatch):
    from cascade.apps.robot_runtime import describe_robot
    # CASCADE_BRIDGE_PORT beats an arm profile's `bridge_port` (B34), so the
    # autouse fixture's 43503 would move the arm off the shared endpoint.
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "43510")
    base = yaml.safe_load((REPO / "configs/bases/microduck_isaac.yaml").read_text())
    base.update(bridge_port=43510)
    cdir = _variant(tmp_path, lambda d: (
        d["domains"]["locomotion"].update(bases=["overlap_base"]),
        d["domains"]["manipulation"].update(arms=["overlap_arm"], offline=False),
        d["domains"]["manipulation"]["mounted_on"].update(base="overlap_base")),
        extra={"bases/overlap_base.yaml": base,
               "arms/overlap_arm.yaml": {"extends": "isaac", "bridge_port": 43510}})
    cfg = load_robot_config(PROFILE, config_dir=cdir)
    with pytest.raises(ValueError) as error:
        describe_robot(cfg)
    message = str(error.value)
    assert "isaac:loopback:43510" in message
    assert "manipulation/overlap_arm" in message and "locomotion/overlap_base" in message


@pytest.mark.parametrize("physical", ["arm", "base"])
def test_physical_mounted_composition_stays_refused_with_named_gates(tmp_path, physical):
    from cascade.apps.robot_runtime import describe_robot
    extra = {}
    if physical == "arm":
        extra["arms/physical_arm.yaml"] = {"extends": "isaac", "bridge_port": 43511}
        mutate = lambda d: d["domains"]["manipulation"].update(arms=["physical_arm"], offline=False)
    else:
        base = yaml.safe_load((REPO / "configs/bases/microduck_isaac.yaml").read_text())
        base.update(bridge_port=43512)
        extra["bases/physical_base.yaml"] = base
        mutate = lambda d: (d["domains"]["locomotion"].update(bases=["physical_base"]),
                            d["domains"]["manipulation"]["mounted_on"].update(base="physical_base"))
    cfg = load_robot_config(PROFILE, config_dir=_variant(tmp_path, mutate, extra=extra))
    with pytest.raises(ValueError, match="physical whole-body composition is not admitted") as error:
        describe_robot(cfg)
    for gate in ("mount calibration", "base-pose source", "moving-frame arm limits", "whole-body controller"):
        assert gate in str(error.value)


def _drop_mount(d):
    d["domains"]["manipulation"].pop("mounted_on")


@pytest.mark.parametrize("mutate,expected", [
    (_drop_mount, "at least one mounted manipulation domain"),
    (lambda d: d["domains"]["manipulation"]["mounted_on"].update(domain="nowhere"), "must name a locomotion domain"),
    (lambda d: d["domains"]["manipulation"]["mounted_on"].update(base="other_base"), "is not configured in locomotion domain"),
    (lambda d: d["domains"]["manipulation"]["mounted_on"].update(rotation_wxyz=[1.0, 0.1, 0.0, 0.0]), "unit quaternion"),
    (lambda d: d["domains"]["manipulation"]["mounted_on"].update(translation_m=[0.0, float("nan"), 0.0]), "finite"),
    (lambda d: d["domains"]["manipulation"]["mounted_on"].update(extra=1), "unknown mount fields"),
    (lambda d: d["whole_body"].update(coordination="parallel"), "coordination must be exclusive or concurrent"),
    (lambda d: d["whole_body"].update(max_base_pose_age_s=0.75), "may only tighten"),
    (lambda d: d["whole_body"].pop("world_target_tolerance_m"), "whole_body requires"),
    (lambda d: d["whole_body"].update(max_mount_drift_m=0.0), "positive"),
    (lambda d: d["domains"]["manipulation"].update(arms=["mock", "piper_mock"]), "exactly one arm"),
    (lambda d: d["domains"]["manipulation"].update(arms=["so101_left"], cameras=["mock_small"]), "static base_pose"),
])
def test_malformed_whole_body_contracts_fail_closed_before_any_device(tmp_path, mutate, expected):
    from cascade.apps.robot_runtime import describe_robot
    with pytest.raises(ValueError, match=expected):
        describe_robot(load_robot_config(PROFILE, config_dir=_variant(tmp_path, mutate)))


# ── dynamic frame chain: capture time, zero-order hold, never extrapolation ───

def test_frame_chain_composes_base_pose_at_capture_time_and_never_extrapolates():
    from cascade.robotics.whole_body import FrameChain, Mount, MountFrameError
    mount = Mount(arm_domain="manipulation", base_domain="locomotion", base="unit-base",
                  translation_m=MOUNT_T, rotation_wxyz=MOUNT_Q, calibration_id="unit/mount")
    chain = FrameChain(mount, max_base_pose_age_s=0.5)
    first, moved = _state(10.0), _state(11.0, x=0.3, y=0.1, yaw=0.5, vx=0.15)
    chain.observe(first)
    chain.observe(moved)
    at_first = chain.arm_base_in_world(at_time_s=10.2, epoch="epoch-a")
    assert np.allclose(at_first.matrix, _pose((0.0, 0.0, 0.2), first.orientation_wxyz) @ MOUNT, atol=1e-12)
    assert at_first.base_capture_time_s == 10.0
    # Zero-order hold: the base was moving at 0.15 m/s, yet the frame 0.3 s
    # later is the CAPTURED pose; an extrapolating chain would add 0.045 m.
    held = chain.arm_base_in_world(at_time_s=11.3, epoch="epoch-a")
    assert np.allclose(held.matrix, _pose((0.3, 0.1, 0.2), moved.orientation_wxyz) @ MOUNT, atol=1e-12)
    world = (0.4, -0.2, 0.5)
    assert np.allclose(held.to_world(held.to_arm(world)), world, atol=1e-12)
    for when, reason in ((11.6, "stale"), (9.9, "missing")):
        with pytest.raises(MountFrameError, match=reason):
            chain.arm_base_in_world(at_time_s=when, epoch="epoch-a")
    with pytest.raises(MountFrameError, match="epoch"):
        chain.arm_base_in_world(at_time_s=11.1, epoch="epoch-b")
    # Capture time is receipt minus producer age, not receipt time.
    late = _state(13.0, x=0.5, age=0.4)
    chain.observe(late)
    assert chain.arm_base_in_world(at_time_s=13.45, epoch="epoch-a").base_capture_time_s == 13.0
    with pytest.raises(MountFrameError, match="stale"):
        chain.arm_base_in_world(at_time_s=13.55, epoch="epoch-a")
    # A new base epoch (world reset) starts a new tree; old samples never leak.
    chain.observe(_state(14.0, epoch="epoch-b"))
    with pytest.raises(MountFrameError, match="epoch"):
        chain.arm_base_in_world(at_time_s=14.1, epoch="epoch-a")
    assert chain.arm_base_in_world(at_time_s=14.1, epoch="epoch-b").epoch == "epoch-b"
    with pytest.raises(MountFrameError, match="fallen"):
        chain.observe(_state(15.0, epoch="epoch-b", fallen=True))


# ── built mock composition ─────────────────────────────────────────────────

def _build(tmp_path, config_dir=None):
    from cascade.apps.robot_runtime import build_robot_runtime
    cfg = load_robot_config(PROFILE, config_dir=config_dir)
    runtime, _ = build_robot_runtime(cfg, tmp_path / "run")
    return runtime


def _home_tcp(rt):
    arm_runtime = rt.domains["manipulation"].runtime
    profile = rt.cfg.domains.as_dict()["manipulation"]["resolved"]["arms"][0]
    return arm_runtime.kin.fk(np.asarray(profile["home_q"], dtype=float))[:3, 3]


def _base_raw(rt):
    return rt.domains["locomotion"].runtime.base_rig.get("microduck_mock").raw


@needs_pin
def test_arm_frame_follows_base_motion_and_reach_verifies_world_target_with_start_frame(tmp_path):
    rt = _build(tmp_path)
    try:
        before = rt.execute("manipulation.get_arm_world_pose", {})
        assert before["ok"], before
        start = before["frame"]
        base0 = start["base_state"]
        T0 = np.array(start["world_from_arm_base"])
        assert np.allclose(T0, _pose(base0["position_world"], base0["orientation_wxyz"]) @ MOUNT, atol=1e-9)
        world_target = (T0 @ np.append(_home_tcp(rt), 1.0))[:3]  # where the home TCP is now
        walked = rt.execute("locomotion.walk_velocity", {"vx": 0.04, "vy": 0.0, "wz": 0.0, "duration_s": 1.0})
        assert walked["execution_ok"] is True, walked
        after = rt.execute("manipulation.get_arm_world_pose", {})["frame"]
        base1 = after["base_state"]
        T1 = np.array(after["world_from_arm_base"])
        displacement = np.subtract(base1["position_world"], base0["position_world"])
        assert np.allclose(displacement, (0.04, 0.0, 0.0), atol=1e-6)
        assert np.allclose(T1, _pose(base1["position_world"], base1["orientation_wxyz"]) @ MOUNT, atol=1e-9)
        assert np.allclose(T1[:3, 3] - T0[:3, 3], displacement, atol=1e-9)  # the arm base rides on the base
        reach = rt.execute("manipulation.reach_world_point", dict(zip("xyz", map(float, world_target))))
        assert reach["execution_ok"] is True, reach
        # Resolved once, with the frame valid at command start (the moved base).
        assert np.allclose(reach["frame_at_start"]["world_from_arm_base"], T1, atol=1e-9)
        # Base moved +4 cm along world x; the arm is yawed +90 deg, so the
        # same world point is 4 cm further along the ARM's +y axis.
        assert np.allclose(reach["target_arm_base"], _home_tcp(rt) + (0.0, 0.04, 0.0), atol=1e-9)
        assert np.allclose(reach["tcp_world"], world_target, atol=1e-3)
        verdict = reach["postcondition"]
        assert verdict["status"] == "unverified" and verdict["observed_status"] == "consistent"
        assert "synthetic" in verdict["reason"]
        assert verdict["evidence"]["world_error_m"] < 1e-3 and verdict["evidence"]["mount_drift_m"] == 0.0
        assert reach["ok"] is False  # a kinematic mock never confirms a physical outcome
        assert "manipulation.reach_world_point" in rt.unverified_actions()
    finally:
        assert rt.close()["ok"]


@needs_pin
@pytest.mark.parametrize("fault", ["stale", "missing"])
def test_stale_or_missing_base_pose_refuses_mounted_arm_motion_before_arm_io(tmp_path, monkeypatch, fault):
    rt = _build(tmp_path)
    try:
        assert rt.execute("manipulation.get_arm_world_pose", {})["ok"]  # base connected and fresh
        raw = _base_raw(rt)
        original = raw.get_state

        def faulted():
            if fault == "missing":
                raise RuntimeError("feedback transport lost")
            return dataclasses.replace(original(), producer_age_s=5.0)
        monkeypatch.setattr(raw, "get_state", faulted)
        arm = rt.domains["manipulation"].runtime.arm
        for tool, args in (("manipulation.reach_world_point", {"x": 0.5, "y": 0.0, "z": 0.6}),
                           ("manipulation.open_gripper", {}),
                           ("manipulation.move_relative", {"direction": "up"})):
            result = rt.execute(tool, args)
            assert result["ok"] is False and fault in result["error"], result
        assert arm.raw.connected is False  # refused before the lazy arm materialized
        assert {"manipulation.open_gripper", "manipulation.move_relative"} <= set(rt.unverified_actions())
        pose = rt.execute("manipulation.get_arm_world_pose", {})
        assert pose["ok"] is False and fault in pose["error"]
    finally:
        assert rt.close()["ok"]


def _profile_with(coordination):
    def mutate(d):
        d["whole_body"]["coordination"] = coordination
    return mutate


def _moving_base(rt):
    """Base controller reports an active command and advances 2 cm per read.

    Returns a restore callable; the patch is an instance attribute on the mock.
    """
    raw = _base_raw(rt)
    original = raw.get_state
    reads = []

    def active():
        state = original()
        reads.append(state)
        x, y, z = state.position_world
        # Each read is a new completed sample: fresh capture time, moved pose.
        return dataclasses.replace(state, controller_status="active", position_world=(x + 0.02 * len(reads), y, z),
                                   received_monotonic_s=time.monotonic(), producer_age_s=0.0)
    raw.get_state = active
    return reads, lambda: vars(raw).pop("get_state", None)


@needs_pin
def test_exclusive_policy_refuses_arm_motion_during_base_command_and_vice_versa(tmp_path):
    rt = _build(tmp_path)
    try:
        assert rt.execute("manipulation.get_arm_world_pose", {})["ok"]
        reads, restore = _moving_base(rt)
        try:
            refused = rt.execute("manipulation.open_gripper", {})
        finally:
            restore()
        assert refused["ok"] is False and "active command" in refused["error"], refused
        assert reads and rt.domains["manipulation"].runtime.arm.raw.connected is False
        harness = rt.domains["manipulation"].runtime.arm.harness
        harness._motion_active = True  # an arm stream in flight (fault injection)
        raw = _base_raw(rt)
        generation = raw.get_state().generation
        try:
            walk = rt.execute("locomotion.walk_velocity", {"vx": 0.04, "vy": 0.0, "wz": 0.0, "duration_s": 0.2})
        finally:
            harness._motion_active = False
        assert walk["ok"] is False and "motion in flight" in walk["error"], walk
        assert raw.get_state().generation == generation  # no base command was sent
        assert rt.execute("locomotion.walk_velocity", {"vx": 0.04, "vy": 0.0, "wz": 0.0,
                                                       "duration_s": 0.2})["execution_ok"] is True
    finally:
        assert rt.close()["ok"]


@needs_pin
def test_concurrent_opt_in_admits_overlap_but_refutes_the_drifted_world_frame(tmp_path):
    rt = _build(tmp_path, _variant(tmp_path, _profile_with("concurrent")))
    try:
        assert rt.cfg.as_dict()["whole_body"]["coordination"] == "concurrent"
        assert rt.execute("manipulation.get_arm_world_pose", {})["ok"]
        _, restore = _moving_base(rt)
        try:
            result = rt.execute("manipulation.open_gripper", {})
        finally:
            restore()
        assert result.get("execution_ok") is not False and result.get("self_reported_ok") is True, result
        frame = result["whole_body"]
        assert frame["coordination"] == "concurrent"
        assert frame["world_frame"]["status"] == "refuted" and frame["world_frame"]["mount_drift_m"] > 0.005
        assert result["ok"] is False and result["postcondition"]["status"] == "refuted"
        harness = rt.domains["manipulation"].runtime.arm.harness
        harness._motion_active = True
        try:
            walk = rt.execute("locomotion.walk_velocity", {"vx": 0.04, "vy": 0.0, "wz": 0.0, "duration_s": 0.2})
        finally:
            harness._motion_active = False
        assert walk["execution_ok"] is True, walk
    finally:
        assert rt.close()["ok"]


@needs_pin
def test_global_stop_latches_every_domain_and_reset_is_explicit_per_domain(tmp_path):
    rt = _build(tmp_path)
    walk = {"vx": 0.04, "vy": 0.0, "wz": 0.0, "duration_s": 0.2}
    try:
        still = rt.execute("manipulation.open_gripper", {})
        assert still["whole_body"]["world_frame"]["status"] == "consistent", still
        assert still["whole_body"]["frame_at_start"]["world_from_arm_base"]
        stopped = rt.execute("emergency_stop", {})
        assert stopped["latched"] and set(stopped["domains"]) == {"manipulation", "locomotion"}
        assert rt.stopped
        for tool, args in (("manipulation.open_gripper", {}), ("locomotion.walk_velocity", walk)):
            result = rt.execute(tool, args)
            assert result["ok"] is False and "reset_stop(domain=" in result["error"], result
        blanket = rt.execute("reset_stop", {})
        assert blanket["ok"] is False and "per-domain" in blanket["error"] and rt.stopped
        arm = rt.execute("reset_stop", {"domain": "manipulation"})
        assert arm["ok"] is True and arm["latched_domains"] == ["locomotion"], arm
        assert rt.stopped  # the base is still latched
        assert rt.execute("manipulation.open_gripper", {})["whole_body"]["world_frame"]["status"] == "consistent"
        assert "locomotion" in rt.execute("locomotion.walk_velocity", walk)["error"]
        base = rt.execute("reset_stop", {"domain": "locomotion"})
        assert base["ok"] is True and base["latched_domains"] == [] and not rt.stopped
        assert rt.execute("locomotion.walk_velocity", walk)["execution_ok"] is True
        bad = rt.execute("reset_stop", {"domain": "sensing"})
        assert bad["ok"] is False
    finally:
        assert rt.close()["ok"]


def test_failed_domain_reset_relatches_every_domain():
    from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
    from cascade.robotics.runtime import RobotRuntime

    class Domain:
        def __init__(self, name):
            self.domain_id = name
            self.resources = (ResourceDescriptor(name + "/body", "base", "fixture", controller_id=name,
                                                 writer_id=name + "/body", synthetic=True),)
            self.tool_descriptors = (ToolDescriptor(name + ".move", "fixture motion",
                {"type": "object", "properties": {}, "additionalProperties": False}, name, "move",
                effect="motion", writes=(name + "/body",)),)
            self.reset_ok, self.stops, self.calls = True, 0, []

        def execute(self, name, args):
            self.calls.append(name)
            return {"ok": True, "execution_ok": True, "postcondition": {"status": "unverified", "reason": "fixture"}}

        def stop(self):
            self.stops += 1
            return {"ok": True}

        def reset_stop(self):
            return {"ok": self.reset_ok}

        def begin_task(self):
            pass

        def close(self):
            return {"ok": True}

    class Coordinator:
        reset_domains = ("arm", "base")

        def metadata(self):
            return {"coordination": "exclusive"}

        def admit(self, descriptor, args):
            return None

        def execute(self, context, domain, name, args):
            return domain.execute(name, args)

        def finish(self, context, result):
            return result

    arm, base = Domain("arm"), Domain("base")
    rt = RobotRuntime({"arm": arm, "base": base}, whole_body=Coordinator())
    try:
        assert rt.stop()["latched"]
        assert rt.reset_stop(domain="base")["ok"]
        assert rt.execute("base.move", {})["execution_ok"] and not arm.calls
        arm.reset_ok = False
        failed = rt.reset_stop(domain="arm")
        assert failed["ok"] is False and failed["latched"] is True
        assert sorted(failed["latched_domains"]) == ["arm", "base"]  # re-latched, not just the failed one
        assert not rt.execute("base.move", {})["ok"] and base.calls == ["move"]
        assert not rt.reset_stop()["ok"]  # explicit per-domain reset only
    finally:
        rt.close()


@needs_pin
def test_stdio_mcp_serves_whole_body_tools_and_per_domain_reset(tmp_path):
    import os
    import subprocess
    import sys
    from test_robot_mcp import Client
    client = Client.__new__(Client)  # same reader/teardown, private env (ports kept)
    directory = tmp_path / "stdio"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(CASCADE_ROBOT=PROFILE, CASCADE_PREWARM="0", CASCADE_STREAM="0", CASCADE_VIEW="0",
               CASCADE_RUN_DIR=str(directory), CASCADE_BELIEFS_PATH=str(directory / "beliefs.json"),
               CASCADE_GRASP_MEMORY_PATH=str(directory / "grasp.json"),
               CASCADE_ENVELOPE_PATH=str(directory / "envelope.json"),
               CASCADE_GRASPGENX_PORT="43501", CASCADE_OCCUPANCY_PORT="43502", CASCADE_BRIDGE_PORT="43503",
               PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1")
    client.proc = subprocess.Popen([sys.executable, "-m", "cascade.apps.mcp_server"], cwd=REPO, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, bufsize=1)
    import queue
    client.queue, client.errors, client.next_id = queue.Queue(), [], 0
    client.threads = [threading.Thread(target=client._read, daemon=True),
                      threading.Thread(target=lambda: client.errors.extend(client.proc.stderr), daemon=True)]
    for thread in client.threads:
        thread.start()
    try:
        client.next_id += 1
        client.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": client.next_id, "method": "tools/list"}) + "\n")
        client.proc.stdin.flush()
        listed = {t["name"] for t in client.receive()["result"]["tools"]}
        assert {"manipulation.reach_world_point", "manipulation.get_arm_world_pose", "locomotion.turn"} <= listed
        pose = client.call("manipulation.get_arm_world_pose")
        assert pose["ok"] and pose["frame"]["base_state"]["measurement_kind"] == "kinematic_mock"
        assert client.call("emergency_stop")["latched"]
        assert client.call("reset_stop")["ok"] is False
        assert client.call("reset_stop", {"domain": "manipulation"})["ok"] is True
        assert client.call("reset_stop", {"domain": "locomotion"})["ok"] is True
    finally:
        client.close()
    assert client.proc.returncode == 0, "".join(client.errors)


# ── golden: profiles without the contract behave exactly as before ────────────

#: Resource-catalog digests and the global-tool digest of every robot profile,
#: computed on unchanged origin/main a459429 (sensor epochs dropped, repo path
#: normalised). A whole-body change must not alter any of them.
GOLDEN_GLOBALS = "9529299957f8d4a1a0ab4e3b2bc2b8917f76c42d3a43ed58fbd32022cffdfe65"
GOLDEN_RESOURCES = {
    "conversation_mock": "1ee12a5bd5a0cadb3947bf8cb7996139637fa95e22e5a033ac5ffa361b54fb7d",
    "factory_m20_mounted": "7d6f90d661004566355e2e503f40cde849d53dc59df9cdeab984c67eefdca781",
    "factory_m20_mounted_margin_v2": "7d6f90d661004566355e2e503f40cde849d53dc59df9cdeab984c67eefdca781",
    "factory_m20_precompile_explicit_v2": "ValueError: unreviewed Factory precompile recipe",
    "factory_m20_precompile_writer_v3": "7d6f90d661004566355e2e503f40cde849d53dc59df9cdeab984c67eefdca781",
    "factory_m20_shoulder_seating": "19739966aff3ba7f0227a43739f49cc58fb946564b11b67b916ac2da904655de",
    "factory_m20_shoulder_seating_heap_freeze": "19739966aff3ba7f0227a43739f49cc58fb946564b11b67b916ac2da904655de",
    "fixed_so101_mock": "5a1be7fe08eb8af2537feb844d0f02872ad60b10f04277675555e9d215428dd3",
    "generalized_joint_sensors": "3e64a3aa6ec7183677ac0a52f37a75e07d64c5ad35594d299e3c87b781d4f95e",
    "leap_hand_right_mujoco": "90a469b35ce062c7a28c1b6a0f810c51a9407a2e49431fc40a09c9161ee8ef18",
    "microduck_conversation_mock": "f1cc331f8d8717725bc92f4c70d37756d8018e02f6595b90123cf6a269dd7584",
    "microduck_conversation_native": "aa7fdbba52834b35a56cafd1640ac76eeb07d1b949576eedbdfb93ef7f13321e",
    "mixed_mock": "8c3222015d963e896625db6a63f10fe08085d6bf0ffd9a032819b51349ee31b5",
    "spatial_replay": "dc4b60beb4bd982acb8c18a6a5875495ede59df7f1209ac9640f0811214798f3",
    "wheeled_lift_sensors": "ee70971d39b476e6ccbbefa698a71bf25c46ab3f86fc0f2e67ce909a126b713e",
}


def _stable(value):
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k != "epoch"}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _digest(value):
    blob = json.dumps(_stable(value), sort_keys=True, separators=(",", ":")).replace(str(REPO), "<repo>")
    return hashlib.sha256(blob.encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(GOLDEN_RESOURCES))
def test_existing_profiles_keep_their_resources_tools_and_global_reset(name):
    from cascade.apps.robot_runtime import describe_robot, robot_tool_descriptors
    from cascade.robotics.resources import ResourceCatalog
    from cascade.robotics.runtime import _global_tools
    try:
        cfg = load_robot_config(name)
        domains = describe_robot(cfg)
    except ValueError as exc:
        assert f"{type(exc).__name__}: {exc}" == GOLDEN_RESOURCES[name]
        return
    assert "whole_body" not in cfg.as_dict()
    catalog = ResourceCatalog([r for d in domains.values() for r in d.resources])
    assert _digest(catalog.describe()) == GOLDEN_RESOURCES[name]
    tools = robot_tool_descriptors(cfg)
    assert _digest([tools[t.name].as_dict() for t in _global_tools()]) == GOLDEN_GLOBALS
    assert tools["reset_stop"].as_spec()["parameters"] == {"type": "object", "properties": {}, "required": [],
                                                           "additionalProperties": False}
    assert not [n for n in tools if n.endswith((".reach_world_point", ".get_arm_world_pose"))]


@needs_pin
def test_mixed_mock_without_contract_keeps_global_reset_and_has_no_coordination_veto(tmp_path):
    from cascade.apps.robot_runtime import build_robot_runtime
    rt, _ = build_robot_runtime(load_robot_config("mixed_mock"), tmp_path / "run")
    try:
        assert getattr(rt, "_whole_body", None) is None
        assert rt.execute("locomotion.get_base_state", {})["ok"]
        raw = rt.domains["locomotion"].runtime.base_rig.get("microduck_mock").raw
        original = raw.get_state
        raw.get_state = lambda: dataclasses.replace(original(), controller_status="active")
        try:
            opened = rt.execute("manipulation.open_gripper", {})
        finally:
            vars(raw).pop("get_state", None)
        assert "whole_body" not in opened and "active command" not in str(opened.get("error"))
        assert rt.execute("emergency_stop", {})["latched"]
        assert not rt.execute("reset_stop", {"domain": "manipulation"})["ok"]  # unknown argument here
        reset = rt.execute("reset_stop", {})
        assert reset["ok"] and set(reset["domains"]) == {"manipulation", "locomotion", "sensing"}
        assert not rt.stopped
    finally:
        assert rt.close()["ok"]
