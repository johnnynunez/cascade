"""Mobile composition: no surrogate arm and no physical claims from a mock."""
import base64
import json

import pytest

from cascade.config import load_demo_config
import test_mobile_frames

frame_endpoint = test_mobile_frames.frame_endpoint


@pytest.fixture(autouse=True)
def no_runtime_thread_leaks():
    import threading
    import time
    before = set(threading.enumerate())
    yield
    deadline = time.monotonic() + 1.5
    while True:
        leaked = [t for t in set(threading.enumerate()) - before if t.is_alive()]
        if not leaked or time.monotonic() >= deadline:
            break
        time.sleep(.01)
    assert not leaked, [t.name for t in leaked]


@pytest.mark.parametrize("new_task", ["false", "true", "", 0, 1, 0.0, 1.0, [], [False], {}, None])
def test_task_memory_rejects_nonbool_without_changing_episode(tmp_path, new_task):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    rt, _ = build_mobile_runtime(load_demo_config(base="microduck_mock", llm="mock"), tmp_path)
    try:
        result = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .04})
        assert result["execution_ok"] and not result["ok"]
        before = rt.unverified_actions()
        query = rt.execute("task_memory", {"new_task": new_task, "k": 0})
        assert query["ok"] is False
        assert "boolean" in query["error"]
        assert rt.unverified_actions() == before
        assert not rt.execute("task_done", {"success": True, "summary": "not verified"})["success"]
    finally:
        rt.close()


def test_base_only_mock_episode_never_constructs_arm(tmp_path, monkeypatch):
    from cascade.apps import demo

    def forbidden(*args, **kwargs):
        raise AssertionError("arm/perception pipeline constructed in base-only mode")

    for name in ("_build_arm", "Kinematics", "make_arm", "SkillRuntime", "_make_detector"):
        monkeypatch.setattr(demo, name, forbidden)
    cfg = load_demo_config(base="microduck_mock")
    assert cfg.robot_mode == "mobile"
    assert "arm" not in cfg and "arms" not in cfg and "grasp" not in cfg
    rt, owner = demo.build_runtime(cfg, tmp_path)
    try:
        assert rt.robot_mode == "mobile"
        assert not hasattr(rt, "arm") and not hasattr(rt, "kin")
        assert not hasattr(rt, "grasp_memory")
        info = rt.execute("list_bases", {})
        assert info["ok"] and len(info["bases"]) == 1
        assert not owner.primary.raw.connected  # metadata did not connect/actuate
        state = rt.execute("get_base_state", {})
        assert state["state"]["measurement_kind"] == "kinematic_mock"
        result = rt.execute("walk_velocity", {"vx": 0.05, "vy": 0.0, "wz": 0.0, "duration_s": 0.08})
        assert result["execution_ok"] is True
        assert result["ok"] is False
        assert result["postcondition"]["status"] == "unverified"
        assert result["measured"]["after"]["position_world"][0] > state["state"]["position_world"][0]
        assert "unverified" in rt.memory.digest()
        records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        assert records[-1]["skill"] == "walk_velocity"
        assert records[-1]["result"]["ok"] is False
        assert not rt.execute("grasp_object", {"label": "cube"})["ok"]
        assert not rt.execute("walk_velocity", {"arm": "mock", "vx": 0.05, "vy": 0, "wz": 0, "duration_s": 0.08})["ok"]
    finally:
        demo.shutdown_runtime(rt, owner)
    assert not owner.primary.raw.connected


@pytest.mark.parametrize("selection", [{"base": ""}, {"bases": []}, {"bases": [""]}, {"bases": ["microduck_mock", "microduck_mock"]}])
def test_invalid_base_selection_fails_not_arm_fallback(selection):
    with pytest.raises(ValueError):
        load_demo_config(**selection)


def test_omitted_base_preserves_arm_default(monkeypatch):
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    cfg = load_demo_config()
    assert cfg.arm.type == "mock"
    assert "base" not in cfg
    assert "robot_mode" not in cfg


@pytest.mark.parametrize("status", ["confirmed", "refuted", "unverified", "invalid", None])
def test_mock_never_becomes_physical_success_from_checker(tmp_path, status):
    from cascade.apps.mobile_runtime import build_mobile_runtime

    class Checker:
        closed = False
        calls = []
        def begin(self, skill, args):
            self.calls.append((skill, args.copy()))
            return "token"
        def finish(self, token, result):
            assert token == "token"
            return {"status": status, "reason": "injected unit-test verdict"}
        def close(self):
            self.closed = True

    checker = Checker()
    rt, _ = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path,
                                 checkers={"microduck_mock": checker})
    try:
        result = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .06})
        assert result["execution_ok"] is True and result["ok"] is False
        assert result["postcondition"]["status"] in {"unverified", "refuted"}
        assert checker.calls[-1][0] == "walk_velocity"
        history = rt.execute("verify_last_action", {})
        assert history["recent"][-1]["status"] == result["postcondition"]["status"]
        assert rt.execute("task_memory", {})["steps_recorded"]
    finally:
        rt.close()
    assert checker.closed


def test_checker_crash_preserves_execution_failure_and_trace(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime

    class BrokenChecker:
        def begin(self, *a):
            raise RuntimeError("independent channel crashed")
        def close(self):
            pass

    rt, _ = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path,
                                 checkers={"microduck_mock": BrokenChecker()})
    try:
        result = rt.execute("walk_velocity", {"vx": 999, "vy": 0., "wz": 0., "duration_s": .06})
        assert result["execution_ok"] is False
        assert result["ok"] is False and result["error"]
        assert result["postcondition"]["status"] == "unverified"
        assert "crashed" in result["postcondition"]["reason"]
    finally:
        rt.close()


def test_stop_while_lazy_connect_blocks_and_reset_cannot_revive(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from cascade.apps.mobile_runtime import build_mobile_runtime

    rt, rig = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path)
    entered, release = threading.Event(), threading.Event()
    connect = rig.primary.raw.connect
    def blocked():
        entered.set()
        assert release.wait(3)
        connect()
    monkeypatch.setattr(rig.primary.raw, "connect", blocked)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(rt.execute, "walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": 1.0})
        try:
            assert entered.wait(2)
            result = rt.stop()
            assert result["latched"] is True
            assert not rt.reset_stop()["ok"]
            release.set()
            assert not future.result(3).get("execution_ok", False)
            assert rig.primary.get_state().position_world[0] == 0
            assert rt.reset_stop()["ok"]
            assert rig.primary.get_state().controller_status == "ready"
        finally:
            release.set()
            rt.close()


def test_caps_and_observations_do_not_create_manipulation_memory(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime

    cfg = load_demo_config(base="microduck_mock")
    cfg._data["bases"][0]["capabilities"] = ["walk_velocity", "stop_navigation"]
    rt, rig = build_mobile_runtime(cfg, tmp_path)
    try:
        assert "turn" not in {s["name"] for s in rt.tool_specs}
        assert not rt.execute("turn", {"angle_rad": .1})["ok"]
        observed = rt.execute("get_observation", {})
        assert observed["state"]["measurement_kind"] == "kinematic_mock"
        assert observed["frame"] is None
        for name in ("kin", "arm", "grasp_memory", "held_object"):
            assert not hasattr(rt, name)
        assert rt.frame_jpeg() is None
        assert rt.execute("recall_memory", {})["ok"]
    finally:
        rt.close()


def test_isaac_profile_requires_external_hashes_endpoint_and_remote_device(tmp_path, monkeypatch):
    import socket
    from cascade.apps.mobile_runtime import build_mobile_runtime

    for key in ("ASSET_SHA256", "POLICY_SHA256", "BRIDGE_PORT", "DEVICE"):
        monkeypatch.delenv("CASCADE_MICRODUCK_" + key, raising=False)
    cfg = load_demo_config(base="microduck_isaac")
    assert cfg.base.admission == "pending_physical_admission"
    assert cfg.base.device is None
    assert cfg.base.bridge_port is None
    assert cfg.base.asset_sha256 is None and cfg.base.policy_sha256 is None
    def forbidden(*args, **kwargs):
        raise AssertionError("construction must never dial an unadmitted endpoint")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    with pytest.raises(ValueError, match="hash|sha256"):
        build_mobile_runtime(cfg, tmp_path)
    monkeypatch.setenv("CASCADE_MICRODUCK_ASSET_SHA256", "a" * 64)
    monkeypatch.setenv("CASCADE_MICRODUCK_POLICY_SHA256", "b" * 64)
    monkeypatch.setenv("CASCADE_MICRODUCK_BRIDGE_PORT", "23456")
    cfg = load_demo_config(base="microduck_isaac")
    # Even valid hashes/port must not invent a device or probe the MCP host.
    with pytest.raises(ValueError, match="device"):
        unexpected, _ = build_mobile_runtime(cfg, tmp_path)
        unexpected.close()  # avoid leaks if the negative unexpectedly succeeds
    monkeypatch.setenv("CASCADE_MICRODUCK_DEVICE", "cuda:3")
    cfg = load_demo_config(base="microduck_isaac")
    rt, rig = build_mobile_runtime(cfg, tmp_path)
    try:
        assert cfg.base.device == "cuda:3"  # exact remote identity, no local fallback
        assert cfg.base.bridge_port == 23456
        assert rig.primary.metadata["measurement_kind"] == "physics"
        assert not rig.primary.raw.connected
        assert "verifier" in rt.verifier_errors["microduck_isaac"]
    finally:
        rt.close()


@pytest.mark.parametrize("bad", ["*", "", "a" * 63, "A" * 64, " a" + "a" * 63])
def test_hash_environment_is_validated_not_normalized(monkeypatch, bad):
    monkeypatch.setenv("CASCADE_MICRODUCK_ASSET_SHA256", bad)
    with pytest.raises(ValueError, match="sha256|SHA256"):
        load_demo_config(base="microduck_isaac")


@pytest.mark.parametrize("velocity", [0., .05])
def test_stop_obligations_block_task_done_until_exact_physical_verdict(tmp_path, frame_endpoint, monkeypatch, velocity):
    """Real RPC, independent checker and synthetic ticks; not physics acceptance."""
    import copy
    import threading
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c, velocity=velocity)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    entered, release = threading.Event(), threading.Event()
    observer = rt.stop_observers["microduck_isaac"]
    finish = observer.checker.finish
    def delayed(*args):
        entered.set()
        assert release.wait(1)
        return finish(*args)
    monkeypatch.setattr(observer.checker, "finish", delayed)
    try:
        assert rt.reset_stop()["ok"]
        ack = rt.execute("emergency_stop", {})
        original = copy.deepcopy(ack)
        assert entered.wait(1)
        pending = rt.execute("task_done", {"success": True, "summary": "stopped"})
        assert pending["success"] is False
        assert ack["receipt_id"] in str(pending["unverified"])
        release.set()
        proof = await_stop(rt, ack["receipt_id"])
        assert proof["task_id"] == ack["task_id"]
        done = rt.execute("task_done", {"success": True, "summary": "stopped"})
        assert done["success"] is (velocity == 0.)
        assert ack == original and not ack["physical_stop_verified"]
        if velocity:
            ticks.velocity = 0.
            retry = rt.stop()
            assert await_stop(rt, retry["receipt_id"])["status"] == "confirmed"
            assert not rt.execute("task_done", {"success": True, "summary": "retry"})["success"]
            assert ack["receipt_id"] in str(rt.unverified_actions())
    finally:
        release.set()
        rt.close()
        ticks.close()


def test_failed_admission_cannot_be_erased_by_task_done(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    rt, _ = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path)
    try:
        rt.stop()
        failed = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .06})
        assert not failed["ok"]
        done = rt.execute("task_done", {"success": True, "summary": "done"})
        assert done["success"] is False and done["unverified"]
    finally:
        rt.close()


def test_new_task_boundary_cannot_adopt_an_old_stop_worker(tmp_path, frame_endpoint, monkeypatch):
    import threading
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    entered, release = threading.Event(), threading.Event()
    finish = observer.checker.finish
    def delayed(*args):
        entered.set()
        assert release.wait(1)
        return finish(*args)
    monkeypatch.setattr(observer.checker, "finish", delayed)
    try:
        assert rt.reset_stop()["ok"]
        old = rt.stop()
        assert entered.wait(1)
        assert rt.execute("task_memory", {"new_task": True, "k": 0})["ok"]
        new = rt.stop()
        assert old["task_id"] != new["task_id"]
        assert not rt.execute("task_done", {"success": True, "summary": "pending"})["success"]
        release.set()
        proof = await_stop(rt, new["receipt_id"])
        assert proof["status"] == "confirmed" and proof["task_id"] == new["task_id"]
        rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        prior = next(r["result"] for r in rows if r["skill"] == "stop_verification"
                     and r["result"]["receipt_id"] == old["receipt_id"])
        assert prior["superseded"] and not prior["physical_stop_verified"]
        assert prior["task_id"] == old["task_id"]
    finally:
        release.set()
        rt.close()
        ticks.close()


def test_read_only_memory_and_task_done_do_not_erase_failed_motion(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    rt, rig = build_mobile_runtime(load_demo_config(base="microduck_mock", llm="mock"), tmp_path)
    try:
        failed = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .04})
        assert not failed["ok"]
        before = rt.unverified_actions()
        generation = rig.primary.get_state().generation
        for name, args in (("task_memory", {"new_task": False, "k": 0}),
                           ("recall_memory", {}), ("verify_last_action", {}), ("get_base_state", {})):
            result = rt.execute(name, args)
            assert result["ok"] and result["task_id"] == failed["task_id"]
            assert rt.unverified_actions() == before
        for _ in range(2):
            done = rt.execute("task_done", {"success": True, "summary": "not proved"})
            assert not done["success"] and done["unverified"] == before
        assert rig.primary.get_state().generation == generation
        assert rt.execute("task_memory", {"new_task": True, "k": 0})["ok"]
        assert rt.unverified_actions() == []
        assert rt.execute("task_done", {"success": True, "summary": "new empty task"})["task_id"] != failed["task_id"]
        assert any(r["task_id"] == failed["task_id"] for r in rt._history)
    finally:
        rt.close()


def test_verifier_confirmed_can_upgrade_only_clean_execution(tmp_path):
    # Classification-only executor/checker doubles, not physical evidence.
    from types import SimpleNamespace
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.control.mobile_rig import MobileRig
    from cascade.skills.mobile_runtime import MobileSkillRuntime

    class Checker:
        def begin(self, *args):
            return "unit-token"
        def finish(self, token, result):
            result.pop("error", None)  # cannot rewrite actor's original failure
            return {"status": "confirmed", "reason": "classification fixture"}
        def close(self):
            pass

    fixture_rt, _ = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path)
    result = {"ok": False, "execution_ok": True}
    raw = SimpleNamespace(metadata={"robot_id": "unit", "source": "fixture", "measurement_kind": "physics"},
                          capabilities=frozenset({"walk_velocity"}), connect=lambda: None,
                          walk_velocity=lambda **kw: dict(result), disconnect=lambda: None,
                          stop=lambda **kw: {"ok": True})
    rt = MobileSkillRuntime(MobileRig([raw], ["microduck_mock"]), fixture_rt.memory, fixture_rt.trace,
                            fixture_rt.cfg, checkers={"microduck_mock": Checker()})
    args = {"vx": .05, "vy": 0., "wz": 0., "duration_s": .06}
    try:
        assert rt.execute("walk_velocity", args)["ok"] is True
        result["execution_ok"] = False
        result["error"] = "actuation failed"
        failed = rt.execute("walk_velocity", args)
        assert failed["ok"] is False and failed["error"] == "actuation failed"
        result["execution_ok"] = True
        result.pop("error")
        result["delivery_uncertain"] = True
        assert not rt.execute("walk_velocity", args)["ok"]
    finally:
        rt.close()
        fixture_rt.close()


def test_runtime_rpc_inert_actor_cannot_borrow_preflight_drift(tmp_path, frame_endpoint):
    import threading
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=0.03, verifier=verifier_limits())
    profile.pop('cameras')
    rt, rig = build_mobile_runtime(camera_cfg(profile), tmp_path)
    drift_started, drift_complete, halt = (threading.Event(), threading.Event(), threading.Event())
    tick_errors = []

    def ticks():
        try:
            step, distance, drift_steps = (c.state()['state']['step'], 0.0, 0)
            while not halt.wait(0.005):
                step += 1
                c.control_at(step * 0.005)
                moving = drift_started.is_set() and drift_steps < 10
                if moving:
                    drift_steps += 1
                    distance = drift_steps * 0.001
                c.publish({'step': step, 'sim_time': step * 0.005, 'position': [distance, 0.0, 0.3], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0], 'linear_velocity': [0.2 if moving else 0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0], 'q': [0.0] * 14, 'dq': [0.0] * 14, 'joint_names': [f'fixture-{i}' for i in range(14)], 'contacts': [], 'fallen': False, 'balance_active': True})
                if drift_steps == 10:
                    drift_complete.set()
        except Exception as exc:
            tick_errors.append(str(exc))
    worker = threading.Thread(target=ticks, name='runtime-review-preflight-ticks')
    worker.start()
    raw_get = rig.primary.raw.get_state
    delayed = False

    def scheduled_preflight():
        nonlocal delayed
        if not delayed:
            delayed = True
            drift_started.set()
            assert drift_complete.wait(0.3)
        return raw_get()
    rig.primary.raw.get_state = scheduled_preflight
    try:
        result = rt.execute('walk_velocity', {'vx': 0.1, 'vy': 0.0, 'wz': 0.0, 'duration_s': 0.1})
        assert not tick_errors, tick_errors
        assert result['execution_ok'], result
        admitted = result['ack']['start_sim_time_s']
        samples = result['postcondition']['evidence']['samples']
        post_admission = [e['state'] for e in samples if e['state']['sim_time_s'] >= admitted]
        positions = sorted({s['position_world'][0] for s in post_admission})
        assert positions == [0.01], positions
        assert result['ok'] is False, 'real RPC + SafeBase + checker counted only pre-admission drift as motion success'
    finally:
        rt.close()
        halt.set()
        worker.join(1)
        assert not worker.is_alive()


def verifier_limits():
    # Software-integration limits only; not physical acceptance calibration.
    return dict(sample_interval_s=.005, read_timeout_s=.04, max_wall_duration_s=1.,
                settle_timeout_s=.10, settle_window_s=.04, min_motion_samples=2,
                min_settle_samples=2, max_samples=200, max_history=8,
                max_state_age_s=.2, max_sample_gap_s=.1, max_position_abs_m=10.,
                max_linear_speed_m_s=1., max_angular_speed_rad_s=5., min_height_m=.1,
                max_tilt_rad=.6, min_progress_ratio=.5, max_progress_ratio=1.5,
                translation_tolerance_m=.001, rotation_tolerance_rad=.03,
                max_lateral_drift_m=.003, max_heading_drift_rad=.04,
                stop_linear_speed_m_s=.01, stop_angular_speed_rad_s=.02,
                stop_drift_m=.001, stop_drift_rad=.01)


def test_production_checker_factory_mock_missing_truth_stays_unverified(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.agent.base_effects import BasePostconditionChecker

    cfg = load_demo_config(base="microduck_mock")
    cfg._data["bases"][0]["verifier"] = verifier_limits()
    rt, _ = build_mobile_runtime(cfg, tmp_path)
    try:
        checker = rt.checkers.get("microduck_mock")
        assert isinstance(checker, BasePostconditionChecker)
        result = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .08})
        assert result["execution_ok"] is True and result["ok"] is False
        assert result["postcondition"]["evidence"]["samples"] == []
        assert checker.history()[-1]["status"] == "unverified"
    finally:
        rt.close()


def test_profile_inheritance_safety_views_and_unknown_base(tmp_path):
    import yaml
    from cascade.config import load_profile
    from cascade.apps.mobile_runtime import build_mobile_runtime

    (tmp_path / "bases").mkdir()
    (tmp_path / "llm").mkdir()
    (tmp_path / "demo.yaml").write_text("memory: {}\nsafety: {arm_specific: true}\n")
    (tmp_path / "llm/mock.yaml").write_text("type: mock\n")
    parent = load_profile("bases", "microduck_mock").as_dict()
    (tmp_path / "bases/parent.yaml").write_text(yaml.safe_dump(parent))
    (tmp_path / "bases/left.yaml").write_text("extends: parent\nrobot_id: left\nsafety: {max_vx: 0.12}\n")
    (tmp_path / "bases/right.yaml").write_text("extends: parent\nrobot_id: right\nsafety: {max_vx: 0.02}\n")
    cfg = load_demo_config(bases=["left", "right"], config_dir=tmp_path)
    assert cfg.bases[0]["resolved"]["safety"]["max_vx"] == .12
    assert cfg.bases[1]["resolved"]["safety"]["max_vx"] == .02
    rt, rig = build_mobile_runtime(cfg, tmp_path / "run")
    try:
        failed = rt.execute("walk_velocity", {"base": "right", "vx": .1, "vy": 0., "wz": 0., "duration_s": .04})
        assert failed["execution_ok"] is False
        assert not rig.get("left").raw.connected
        assert not rt.execute("get_base_state", {"base": "unknown"})["ok"]
    finally:
        rt.close()


def test_mock_minimal_import_surface_in_isolated_child(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    code = '''
import importlib.abc, json, sys, threading
from pathlib import Path
blocked = {"torch", "pinocchio", "onnxruntime", "onnx", "isaacsim", "omni", "mujoco", "ultralytics"}
class Ban(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("optional dependency imported: " + fullname)
sys.meta_path.insert(0, Ban())
import cascade
from cascade.config import load_demo_config
from cascade.apps.demo import build_runtime, shutdown_runtime
before = set(threading.enumerate())
rt, owner = build_runtime(load_demo_config(base="microduck_mock"), Path(sys.argv[1]))
try:
    out = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .08})
    assert out["execution_ok"] is True and out["ok"] is False
finally:
    shutdown_runtime(rt, owner)
leaked = [t.name for t in set(threading.enumerate()) - before if t.is_alive()]
assert not leaked, leaked
assert not blocked & set(sys.modules)
print(json.dumps({"cascade": cascade.__file__, "optional_imports": [], "leaked_threads": leaked,
                  "execution_ok": out["execution_ok"], "postcondition": out["postcondition"]}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    proc = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)],
                          env=env, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert Path(data["cascade"]).resolve() == Path(__file__).resolve().parents[1] / "src/cascade/__init__.py"
    assert data["leaked_threads"] == []


@pytest.mark.parametrize("admit_device", [False, True])
def test_actual_checker_refutes_scripted_inert_loopback_actor(tmp_path, admit_device):
    # This deliberately inert software producer is NOT an Isaac physics run.
    import threading
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
    from cascade.apps.mobile_runtime import build_mobile_runtime

    c = MobileBridgeController(robot_id="microduck", source="isaac-microduck", engine="physx",
        device="cuda:0", asset_sha256="a" * 64, policy_sha256="b" * 64,
        max_linear_speed=.15, max_angular_speed=.6, max_duration_s=3., lease_s=.3,
        max_state_age_s=.3, max_action_wall_s=2.)
    server = MobileBridgeServer(c, port=0)
    server.start()
    stop = threading.Event()
    ready = threading.Event()
    def producer():
        step = 0
        while not stop.is_set():
            step += 1
            sim_time = step * .005
            c.control_at(sim_time)
            c.publish({"step": step, "sim_time": sim_time, "position": [0., 0., .3],
                       "orientation_wxyz": [1., 0., 0., 0.], "linear_velocity": [0., 0., 0.],
                       "angular_velocity": [0., 0., 0.], "q": [0.] * 14, "dq": [0.] * 14,
                       "joint_names": [f"fixture-{i}" for i in range(14)], "contacts": [], "fallen": False})
            ready.set()
            stop.wait(.005)
    thread = threading.Thread(target=producer, name="runtime-inert-producer")
    rt = None
    try:
        thread.start()
        assert ready.wait(1)
        cfg = load_demo_config(base="microduck_isaac")
        cfg._data["bases"][0].update(asset_sha256="a" * 64, policy_sha256="b" * 64,
            bridge_port=server.address[1], timeout_s=.03, verifier=verifier_limits())
        if not admit_device:
            with pytest.raises(ValueError, match="device"):
                rt, _ = build_mobile_runtime(cfg, tmp_path)
            return  # finally must stop the fixture even on construction failure
        cfg._data["bases"][0]["device"] = c.hello()["device"]
        rt, _ = build_mobile_runtime(cfg, tmp_path)
        checker = rt.checkers["microduck_isaac"]
        assert checker._reader is not rt.observation_readers["microduck_isaac"]
        result = rt.execute("walk_velocity", {"vx": .1, "vy": 0., "wz": 0., "duration_s": .1})
        assert result["execution_ok"] is True, result
        assert result["ok"] is False
        assert result["postcondition"]["status"] == "refuted", result
        assert result["postcondition"]["evidence"]["samples"]
    finally:
        if rt is not None:
            rt.close()
        stop.set()
        if thread.ident is not None:
            thread.join(timeout=1)
        server.close()
        assert not thread.is_alive()


def test_verifier_exception_closes_its_sampler_before_return(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime

    class Crashed:
        closed = False
        def begin(self, *args):
            return "active-sampler"
        def finish(self, *args):
            raise RuntimeError("sampler failed")
        def close(self):
            self.closed = True

    checker = Crashed()
    rt, _ = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path,
                                 checkers={"microduck_mock": checker})
    try:
        result = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .04})
        assert result["ok"] is False and result["postcondition"]["status"] == "unverified"
        assert checker.closed
    finally:
        rt.close()


def test_teardown_reports_checker_failure_but_still_disconnects_base(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    class CannotClose:
        def close(self):
            raise RuntimeError("checker did not close")
    rt, rig = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path,
                                   checkers={"microduck_mock": CannotClose()})
    assert rt.execute("get_base_state", {})["ok"]
    result = rt.close()
    assert result["ok"] is False
    assert "checker did not close" in str(result["errors"])
    assert not rig.primary.raw.connected


def camera_cfg(profile):
    cfg = load_demo_config(base="microduck_isaac")
    cfg._data["bases"][0].update(profile)
    return cfg


def test_real_camera_observation_cache_memory_and_trace(tmp_path, frame_endpoint):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, packet, jpeg, operations = frame_endpoint
    rt, rig = build_mobile_runtime(camera_cfg(profile), tmp_path)
    try:
        assert not operations
        assert "camera_snapshot" in {s["name"] for s in rt.tool_specs}
        assert rt.frame_jpeg() is None
        observed = rt.execute("get_observation", {})
        assert observed["ok"] and observed["frame"]["step"] == packet["step"], observed
        assert base64.b64decode(observed["image_jpeg_b64"], validate=True) == jpeg
        assert rt.frame_jpeg() == jpeg
        observed["frame"]["source"] = "mutated"
        assert not rig.primary.raw.connected
        snapshot = rt.execute("camera_snapshot", {"camera": "side"})
        assert base64.b64decode(snapshot["image_jpeg_b64"], validate=True) == jpeg
        assert snapshot["frame"]["source"] == packet["source"]
        assert not rt.execute("camera_snapshot", {"camera": "unknown"})["ok"]
        memory = rt.execute("task_memory", {"k": 4})
        assert memory["frames"] >= 1
        assert rt.memory.memory_frames(4)[0]["jpeg"] == jpeg
        assert memory["steps_recorded"][0]["frame"]["source"] == packet["source"]
        records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        obs_trace = next(r for r in records if r["skill"] == "get_observation")
        assert obs_trace["result"]["frame"]["step"] == 1
        assert (tmp_path / obs_trace["keyframe_after"]).read_bytes() == jpeg
        packet["producer_age_s"] = 10.
        assert not rt.execute("camera_snapshot", {})["ok"]
        assert rt.frame_jpeg() is None
        assert not rig.primary.raw.connected
        assert all(r["op"] in {"hello", "state", "frame"} for r in operations)
    finally:
        rt.close()
    assert c.hello()["generation"] == 0


@pytest.mark.parametrize("cameras", [[], {"side": {}}, {"side": {"max_age_s": -1}}])
def test_invalid_runtime_camera_profile_fails_before_connect(tmp_path, frame_endpoint, cameras):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    _, _, profile, _, _, operations = frame_endpoint
    cfg = camera_cfg({**profile, "cameras": cameras})
    with pytest.raises(ValueError):
        build_mobile_runtime(cfg, tmp_path)
    assert not operations


class SyntheticTicks:
    """Completed-step software publisher; balance_active is synthetic, NOT physics."""
    def __init__(self, controller, *, velocity=0., balance_active=True):
        import threading
        self.controller = controller
        self.velocity, self.balance_active = velocity, balance_active
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self.run, name="followup-synthetic-ticks")
        self.thread.start()

    def run(self):
        c = self.controller
        step = c.state()["state"]["step"]
        while not self.halt.is_set():
            step += 1
            c.control_at(step * .005)
            sample = {"step": step, "sim_time": step * .005, "position": [0., 0., .3],
                      "orientation_wxyz": [1., 0., 0., 0.], "linear_velocity": [self.velocity, 0., 0.],
                      "angular_velocity": [0., 0., 0.], "q": [0.] * 14, "dq": [0.] * 14,
                      "joint_names": [f"fixture-{i}" for i in range(14)], "contacts": [], "fallen": False}
            if self.balance_active is not None:
                sample["balance_active"] = self.balance_active
            c.publish(sample)
            self.halt.wait(.005)

    def close(self):
        self.halt.set()
        self.thread.join(1)
        assert not self.thread.is_alive()


def await_stop(rt, receipt_id, timeout=2.):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshots = rt.execute("verify_last_action", {}).get("stop_verifications", [])
        for snapshot in snapshots:
            if snapshot["receipt_id"] == receipt_id and not snapshot["pending"]:
                return snapshot
        time.sleep(.005)
    pytest.fail(f"no completed post-ACK receipt for {receipt_id}: {snapshots}")


@pytest.mark.parametrize("velocity,expected", [(0., "confirmed"), (.05, "refuted")])
def test_post_ack_stop_is_async_independent_and_preserves_original(tmp_path, frame_endpoint, velocity, expected):
    import copy
    import time
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c, velocity=velocity)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    try:
        assert rt.reset_stop()["ok"]  # explicit control session; read-only camera never acquires it
        started = time.monotonic()
        ack = rt.stop()
        assert time.monotonic() - started < .08
        assert ack["ok"] and ack["outcome"] == "unverified"
        assert ack["physical_stop_verified"] is False
        original = copy.deepcopy(ack)
        proof = await_stop(rt, ack["receipt_id"])
        assert proof["status"] == expected, proof
        assert proof["ack"] == ack["bases"]["microduck_isaac"]
        assert proof["serial"] == ack["serial"]
        assert proof["postcondition"]["evidence"]["samples"]
        for sample in proof["postcondition"]["evidence"]["samples"]:
            state = sample["state"]
            assert state["generation"] == proof["ack"]["generation"]
            assert state["received_monotonic_s"] - state["producer_age_s"] >= ack["ack_monotonic_s"]
        assert rt.checkers["microduck_isaac"].history() == []  # active-motion checker untouched
        assert ack == original  # never rewrite cancellation ACK into physical success
    finally:
        result = rt.close()
        ticks.close()
    assert result["ok"], result


@pytest.mark.parametrize("blocked_method", ["reader", "finish"])
def test_blocked_stop_observation_has_deadline_no_spam_workers_and_reset_wins(tmp_path, frame_endpoint, monkeypatch, blocked_method):
    import threading
    import time
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    limits = verifier_limits()
    limits["max_wall_duration_s"] = .3
    profile.update(timeout_s=.03, verifier=limits)
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    entered, release = threading.Event(), threading.Event()
    observer = rt.stop_observers["microduck_isaac"]
    original = observer.reader.reader if blocked_method == "reader" else observer.checker.finish
    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)
    if blocked_method == "reader":
        # A deliberately contract-violating callable: quarantine, not thread multiplication.
        monkeypatch.setattr(observer.reader, "reader", blocked)
    else:
        monkeypatch.setattr(observer.checker, "finish", blocked)
    try:
        assert rt.reset_stop()["ok"]
        first = rt.stop()
        assert entered.wait(1)
        started = time.monotonic()
        for _ in range(20):
            latest = rt.stop()
        assert time.monotonic() - started < .2
        assert len([t for t in threading.enumerate() if t.name == "mobile-stop-microduck_isaac"]) == 1
        assert first["receipt_id"] != latest["receipt_id"]
        proof = await_stop(rt, latest["receipt_id"], timeout=.7)
        assert proof["status"] == "unverified"
        if blocked_method == "reader":
            # The unified sampler's per-read watchdog can quarantine BEFORE
            # the wall deadline; no extra observer is created for the mailbox.
            assert "quarantined" in proof["reason"] and observer._quarantined
            assert observer.checker._worker.is_alive()
        else:
            assert "deadline" in proof["reason"]
        assert not proof["physical_stop_verified"]
        assert rt.reset_stop()["ok"]
        release.set()
        time.sleep(.15)
        after = rt.execute("verify_last_action", {})["stop_verifications"][-1]
        assert after["receipt_id"] == latest["receipt_id"]
        assert after["superseded"] and not after["physical_stop_verified"]
    finally:
        release.set()
        if blocked_method == "reader":
            monkeypatch.setattr(observer.reader, "reader", original)
        rt.close()
        ticks.close()
    assert not observer._thread.is_alive()


@pytest.mark.parametrize("key,bad", [("generation", True), ("generation", 1.0),
                                     ("latched", "true"), ("latched", None)])
def test_stop_rejects_malformed_receipt_before_independent_confirmation(tmp_path, frame_endpoint, monkeypatch, key, bad):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    rt, rig = build_mobile_runtime(camera_cfg(profile), tmp_path)
    stop = rig.stop
    def malformed(**kw):
        result = stop(**kw)
        result["bases"]["microduck_isaac"][key] = bad
        return result
    monkeypatch.setattr(rig, "stop", malformed)
    try:
        rt._connect("microduck_isaac")
        ack = rt.stop()
        proof = await_stop(rt, ack["receipt_id"])
        assert proof["status"] == "unverified"
        assert not rt.execute("task_done", {"success": True, "summary": "not proved"})["success"]
    finally:
        rt.close()
        ticks.close()


def test_reset_revokes_current_stop_confirmation_without_rewriting_ack(tmp_path, frame_endpoint):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    try:
        assert rt.reset_stop()["ok"]
        ack = rt.stop()
        assert await_stop(rt, ack["receipt_id"])["physical_stop_verified"]
        assert rt.reset_stop()["ok"]
        proof = rt.execute("verify_last_action", {})["stop_verifications"][-1]
        assert proof["superseded"] and not proof["physical_stop_verified"]
        assert ack["physical_stop_verified"] is False
    finally:
        rt.close()
        ticks.close()


def test_camera_motion_frames_keep_failed_action_and_source_steps(tmp_path, frame_endpoint):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, jpeg, _ = frame_endpoint
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    ticks = SyntheticTicks(c)
    try:
        # Invalid intent must remain failed, even if the camera delivers valid JPEGs.
        result = rt.execute("walk_velocity", {"vx": 999., "vy": 0., "wz": 0., "duration_s": .04})
        assert not result["execution_ok"] and not result["ok"] and result["error"]
        assert result["camera_evidence"]["before"]["frame"]["step"] == 1
        assert result["camera_evidence"]["after"]["frame"]["step"] == 1  # repeated physical step is not invented progress
        memory = rt.execute("task_memory", {})
        assert any("after walk_velocity" in e["action"] and e["verdict"] == "unverified" for e in memory["steps_recorded"])
        row = next(json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
                   if json.loads(line)["skill"] == "walk_velocity")
        assert (tmp_path / row["keyframe_before"]).read_bytes() == jpeg
        assert (tmp_path / row["keyframe_after"]).read_bytes() == jpeg
        assert not rt.execute("task_done", {"success": True, "summary": "not really done"})["ok"]
    finally:
        rt.close()
        ticks.close()


def test_stop_prebaseline_sampling_has_an_attempt_budget(tmp_path, frame_endpoint, monkeypatch):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    _, server, profile, _, _, _ = frame_endpoint
    limits = verifier_limits()
    limits["max_samples"] = 8  # explicit software quota, not physical tolerance
    profile.update(timeout_s=.03, verifier=limits)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    calls, dispatch = [], server.dispatch
    def count(req):
        if req["op"] == "state":
            calls.append(req)
        return dispatch(req)
    try:
        assert rt.reset_stop()["ok"]
        monkeypatch.setattr(server, "dispatch", count)
        # No publisher: genuinely pre-ACK captures, not None/channel outages.
        ack = rt.stop()
        proof = await_stop(rt, ack["receipt_id"])
        assert proof["status"] == "unverified" and "sample_limit" in proof["reason"]
        evidence = proof["postcondition"]["evidence"]
        assert evidence["samples"] == []
        assert len(evidence["temporal_pending"]) == evidence["attempts"] == len(calls) == 8
        assert not evidence["channel_failed"] and not observer._quarantined
    finally:
        rt.close()


def test_superseded_stop_read_keeps_healthy_observer_reusable(tmp_path, frame_endpoint):
    """Real loopback reader; only scheduling/physical ticks are CPU fixtures."""
    import threading
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    original = observer.reader.reader
    entered, release = threading.Event(), threading.Event()

    class ScheduledRead:
        first = True
        def __call__(self):
            if self.first:
                self.first = False
                entered.set()
                assert release.wait(.03)
            return original()
        @property
        def last_error(self):
            return original.last_error
        def close(self):
            original.close()

    observer.reader.reader = ScheduledRead()
    try:
        assert rt.reset_stop()["ok"]
        first = rt.stop()
        assert entered.wait(1)
        second = rt.stop()
        release.set()
        assert await_stop(rt, second["receipt_id"])["status"] == "confirmed"
        third = rt.stop()
        assert await_stop(rt, third["receipt_id"])["status"] == "confirmed"
        assert not observer._quarantined
        rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        old = next(r["result"] for r in rows if r["skill"] == "stop_verification"
                   and r["result"]["receipt_id"] == first["receipt_id"])
        assert old["superseded"] and not old["physical_stop_verified"]
        assert "superseded" in str(rt.unverified_actions())
    finally:
        release.set()
        rt.close()
        ticks.close()


def test_out_of_order_stop_submission_cannot_replace_latest_mailbox(tmp_path, frame_endpoint, monkeypatch):
    import threading
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    entered, release = threading.Event(), threading.Event()
    original_finish = observer.checker.finish
    def delayed(*args):
        entered.set()
        assert release.wait(2)
        return original_finish(*args)
    monkeypatch.setattr(observer.checker, "finish", delayed)
    jobs, submit = [], observer.submit
    def record(job):
        jobs.append(job)
        submit(job)
    monkeypatch.setattr(observer, "submit", record)
    try:
        assert rt.reset_stop()["ok"]
        rt.stop()
        assert entered.wait(1)
        rt.stop()
        latest = rt.stop()
        submit(jobs[-2])  # delayed prior thread reaches the mailbox after a newer stop
        release.set()
        proof = await_stop(rt, latest["receipt_id"])
        assert proof["status"] == "confirmed", proof
    finally:
        release.set()
        rt.close()
        ticks.close()


def test_camera_state_epoch_conflict_does_not_poison_memory_or_last_frame(tmp_path, frame_endpoint):
    from dataclasses import replace
    from cascade.apps.mobile_runtime import build_mobile_runtime
    _, _, profile, _, _, _ = frame_endpoint
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    reader = rt.observation_readers["microduck_isaac"]
    actual = reader()
    rt.observation_readers["microduck_isaac"] = lambda: replace(actual, epoch="conflicting-state-epoch")
    try:
        result = rt.execute("get_observation", {})
        assert not result["ok"]
        assert rt.frame_jpeg() is None
        assert rt.memory.memory_frames(4) == []
    finally:
        reader.close()
        rt.observation_readers["microduck_isaac"] = None
        rt.close()


@pytest.mark.parametrize("case", ["frozen", "missing", "disabled", "health_absent", "fault", "new_epoch", "bad_ack", "no_limits"])
def test_stop_never_confirms_missing_or_contradictory_evidence(tmp_path, frame_endpoint, monkeypatch, case):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, server, profile, _, _, _ = frame_endpoint
    limits = verifier_limits()
    limits["max_wall_duration_s"] = .35
    profile.update(timeout_s=.03, verifier=None if case == "no_limits" else limits)
    profile.pop("cameras")
    ticks = None if case == "frozen" else SyntheticTicks(c, balance_active=False if case == "disabled" else None if case == "health_absent" else True)
    rt, rig = build_mobile_runtime(camera_cfg(profile), tmp_path)
    try:
        assert rt.reset_stop()["ok"]
        if case == "missing":
            dispatch = server.dispatch
            monkeypatch.setattr(server, "dispatch", lambda req: {"ok": True, "state": None} if req["op"] == "state" else dispatch(req))
        if case == "bad_ack":
            monkeypatch.setattr(rig.primary.raw, "stop", lambda **kw: {"ok": False, "error": "lost stop ACK", "delivery_uncertain": True})
        if case == "fault":
            c.fault("synthetic controller failure")
        if case == "new_epoch":
            # The stop reader's first hello is intentionally bound before a world reset.
            observer = rt.stop_observers["microduck_isaac"]
            assert observer.reader.reader() is not None
            c.begin_epoch()
        ack = rt.stop()
        proof = await_stop(rt, ack["receipt_id"])
        assert proof["status"] != "confirmed", proof
        assert not proof["physical_stop_verified"]
        if case == "bad_ack":
            assert not ack["ok"]
            assert "lost stop ACK" in proof["ack"]["error"]
    finally:
        rt.close()
        if ticks:
            ticks.close()


def test_late_stop_verifier_error_is_preserved_in_receipt(tmp_path, frame_endpoint, monkeypatch):
    import threading
    import time
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    limits = verifier_limits()
    limits["max_wall_duration_s"] = .3
    profile.update(timeout_s=.03, verifier=limits)
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    entered, release = threading.Event(), threading.Event()
    def fails_late(*args):
        entered.set()
        assert release.wait(2)
        raise RuntimeError("original independent channel failure")
    monkeypatch.setattr(observer.checker, "finish", fails_late)
    try:
        assert rt.reset_stop()["ok"]
        ack = rt.stop()
        assert entered.wait(1)
        assert await_stop(rt, ack["receipt_id"])["status"] == "unverified"
        release.set()
        deadline = time.monotonic() + 1
        while True:
            path = tmp_path / "trace.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            proofs = [r for r in rows if r["skill"] == "stop_verification"]
            if proofs or time.monotonic() >= deadline:
                break
            time.sleep(.005)
        assert proofs
        assert "original independent channel failure" in json.dumps(proofs[-1])
        assert proofs[-1]["result"]["status"] == "unverified"
    finally:
        release.set()
        rt.close()
        ticks.close()


def test_shutdown_is_bounded_even_if_stop_checker_breaks_its_contract(tmp_path, frame_endpoint, monkeypatch):
    import threading
    import time
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    limits = verifier_limits()
    limits["max_wall_duration_s"] = .3
    profile.update(timeout_s=.03, verifier=limits)
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    entered, release = threading.Event(), threading.Event()
    finish = observer.checker.finish
    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return finish(*args)
    monkeypatch.setattr(observer.checker, "finish", blocked)
    try:
        assert rt.reset_stop()["ok"]
        rt.stop()
        assert entered.wait(1)
        started = time.monotonic()
        result = rt.close()
        assert time.monotonic() - started < .8
        assert not result["ok"] and "quarantined" in str(result["errors"])
        assert observer._thread.is_alive()  # never pretend Python killed a hostile callable
    finally:
        release.set()
        if observer._thread:
            observer._thread.join(1)
        rt.close()
        ticks.close()
    assert not observer._thread.is_alive()


def test_stop_checker_can_finish_while_motion_checker_is_blocked(tmp_path, frame_endpoint):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    entered, release = threading.Event(), threading.Event()
    class BlockedMotion:
        def begin(self, *args):
            return "motion-token"
        def finish(self, *args):
            entered.set()
            assert release.wait(3)
            return {"status": "confirmed", "reason": "classifier fixture, NOT physics"}
        def close(self):
            pass
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path, checkers={"microduck_isaac": BlockedMotion()})
    try:
        with ThreadPoolExecutor() as pool:
            motion = pool.submit(rt.execute, "walk_velocity", {"vx": 999., "vy": 0., "wz": 0., "duration_s": .05})
            try:
                assert entered.wait(1)
                started = time.monotonic()
                ack = rt.stop()
                assert time.monotonic() - started < .08
                assert await_stop(rt, ack["receipt_id"])["status"] == "confirmed"
                assert not motion.done()
            finally:
                release.set()
            failed = motion.result(1)
            assert not failed["ok"] and not failed["execution_ok"] and failed["error"]
            assert failed["postcondition"]["status"] != "confirmed"
            assert rt.unverified_actions()
    finally:
        release.set()
        rt.close()
        ticks.close()



