"""Isaac reset DELIVERY CONTRACT tests, NOT an Isaac/GPU physics rehearsal.

Execute the actual runtime dispatcher, IsaacArm/BridgeClient, TCP Handler,
main-thread job queue and settle/reset helpers. Only Kit/physics, camera and
safe-arm motion boundaries are doubles. Their coordinates are test inputs,
never evidence that Newton or the shipped asset works on a DGX Spark.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import socketserver
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from conftest import REPO, loopback_host

from cascade.agent.trace import TraceLogger
from cascade.config import Cfg
from cascade.control.isaac_arm import IsaacArm
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.sim.bridge_client import BridgeError
from cascade.skills.runtime import SkillRuntime


class PhysicsBoundary:
    """Deterministic Kit/rigid-body stand-in; it does NOT simulate physics."""

    def __init__(self):
        self.spawns = {"pink_cube": (0.17, 0.15, 0.04), "green_cube": (0.30, 0.16, 0.04)}
        self.base_z = 0.7
        self.positions = {n: np.array([0.48, -0.2, 0.74]) for n in self.spawns}
        self.events = []
        self.pending = {}
        self.steps = 0

    def main_thread(self):
        assert threading.current_thread() is threading.main_thread(), "Kit called on a worker"

    def teleport(self, name, pos):
        self.main_thread()
        self.events.append(("teleport", name))
        self.pending[name] = np.array([pos[0], pos[1], pos[2] + self.base_z])
        return True

    def update(self):
        self.main_thread()
        self.steps += 1
        for name, target in self.pending.items():
            # A fixed response to the requested teleport, not a solver.
            self.positions[name] = target.copy()
            self.positions[name][2] = self.spawns[name][2] + self.base_z
        self.pending.clear()

    def rigid_prim(self, path, **kwargs):
        self.main_thread()
        name = path.rsplit("/", 1)[-1]

        def poses():
            self.main_thread()
            self.events.append(("read", name))
            return SimpleNamespace(numpy=lambda: self.positions[name].reshape(1, 3).copy()), None

        return SimpleNamespace(
            # Newton's reduced-coordinate solver ignores the body-pose write.
            set_world_poses=lambda *a: None,
            set_velocities=lambda *a: None,
            get_world_poses=poses,
        )


def _load_bridge_definitions(monkeypatch, physics):
    """Compile REAL script definitions; omit SimulationApp boot on this Mac.

    Do not copy/reimplement the handler or reset logic in the tests. Loading
    its AST definitions is necessary because importing the script boots Kit.
    """
    path = REPO / "scripts" / "isaac_bridge.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {"Handler", "_run_exec_jobs", "_settle_props", "_zero_prop_velocity",
              "_reset_props_verified"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    prims = ModuleType("isaacsim.core.experimental.prims")
    prims.RigidPrim = physics.rigid_prim
    monkeypatch.setitem(sys.modules, prims.__name__, prims)
    env = {
        "__name__": "isaac_reset_contract_under_test", "np": np,
        "threading": threading, "socketserver": socketserver, "json": json,
        "_exec_lock": threading.Lock(), "_exec_jobs": [],
        "_PROP_SPAWNS": physics.spawns, "BASE_Z": physics.base_z,
        "stage": SimpleNamespace(GetPrimAtPath=lambda p: p.rsplit("/", 1)[-1] in physics.positions),
        "args": SimpleNamespace(engine="newton"), "engine": "newton", "names": [],
        "app": SimpleNamespace(update=physics.update),
        "_tl": SimpleNamespace(is_playing=lambda: True),
        "_newton_teleport": physics.teleport,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


@pytest.fixture
def reset_bridge(monkeypatch):
    physics = PhysicsBoundary()
    env = _load_bridge_definitions(monkeypatch, physics)
    server = socketserver.ThreadingTCPServer((loopback_host(), 0), env["Handler"])
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    arm = IsaacArm(Cfg({"bridge_host": loopback_host(), "bridge_port": server.server_address[1]}))
    arm.connect()
    yield arm, physics, env
    arm.disconnect()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _pump_main_thread(env, call):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call)
        deadline = time.monotonic() + 5
        while not future.done() and time.monotonic() < deadline:
            env["_run_exec_jobs"]()
            time.sleep(0.001)
        assert future.done(), "bridge request did not finish with a running main loop"
        return future.result()


def _runtime(arm, physics, demo_cfg, tmp_path):
    def home(*args, **kwargs):
        physics.events.append(("home",))
        return True

    def no_camera():
        raise RuntimeError("camera boundary absent in CPU protocol contract test")

    safe_arm = SimpleNamespace(raw=arm, move_joints=home)
    return SkillRuntime(
        camera=SimpleNamespace(get_frame=no_camera), depth_provider=None, detector=None,
        extrinsics=None, kin=None, safe_arm=safe_arm, memory=EpisodicMemory(),
        beliefs=BeliefStore(), trace=TraceLogger(tmp_path / "run"), cfg=demo_cfg,
    )


def test_dispatcher_records_measured_isaac_reset_in_its_own_trace(reset_bridge, demo_cfg, tmp_path):
    arm, physics, env = reset_bridge
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))

    assert result["ok"] is True, result
    assert result["world"] == "isaac", result
    assert result["props_reset"] == list(physics.spawns), result
    verification = result["reset_verification"]
    assert verification["channel"] == "physics"
    assert verification["frame"] == "robot_base"
    for name, spawn in physics.spawns.items():
        check = verification["props"][name]
        assert check["position_m"] == pytest.approx(spawn)
        assert check["spawn_position_m"] == list(spawn)
        assert check["error_m"] <= verification["tolerance_m"]
    assert physics.events[0] == ("home",)
    assert {e[1] for e in physics.events if e[0] == "teleport"} == set(physics.spawns)
    rows = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["skill"] == "reset_scene"
    assert rows[0]["result"] == result


@pytest.mark.parametrize("bad_position", [
    [0.48, -0.2, 0.74],  # still calmly resting in the bin, not at spawn
    [float("nan"), 0.15, 0.74],
    [0.17, float("inf"), 0.74],
], ids=["still-in-bin", "nan", "infinity"])
def test_failed_readback_cannot_be_reported_as_reset_success(
    reset_bridge, demo_cfg, tmp_path, monkeypatch, bad_position,
):
    arm, physics, env = reset_bridge
    physics.positions["pink_cube"] = np.asarray(bad_position)

    def failed_teleport(name, pos):
        return False if name == "pink_cube" else physics.teleport(name, pos)

    monkeypatch.setitem(env, "_newton_teleport", failed_teleport)
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))

    assert result["ok"] is False, result
    assert result["world"] == "isaac", result
    assert result["props_reset"] == [], result
    assert "pink_cube" in result["error"], result
    row = json.loads((tmp_path / "run" / "trace.jsonl").read_text())
    assert row["result"]["ok"] is False


@pytest.mark.parametrize("unavailable", ["no-props", "missing-prop", "paused"])
def test_handler_refuses_unverifiable_scene_before_any_physics_write(reset_bridge, monkeypatch, unavailable):
    _, physics, env = reset_bridge
    for name, spawn in physics.spawns.items():
        physics.positions[name] = np.asarray(spawn) + [0, 0, physics.base_z]
    if unavailable == "no-props":
        physics.spawns.clear()
    elif unavailable == "missing-prop":
        # A stale physics handle still reports an old, plausible pose.
        monkeypatch.setattr(env["stage"], "GetPrimAtPath", lambda p: not p.endswith("pink_cube"))
    else:
        monkeypatch.setattr(env["_tl"], "is_playing", lambda: False)
    handler = object.__new__(env["Handler"])
    result = _pump_main_thread(env, lambda: handler._dispatch({"op": "reset_props"}))

    assert result["ok"] is False, result
    assert result.get("props_reset", []) == []
    assert result["error"]
    assert physics.events == [], "an unverifiable scene must be refused BEFORE settling"


def _wire_reset_result(physics):
    """Synthetic OLD/MALFORMED bridge messages, used only to test rejection."""
    return {"ok": True, "props_reset": list(physics.spawns), "reset_verification": {
        "channel": "physics", "frame": "robot_base", "tolerance_m": 0.02,
        "props": {n: {"spawn_position_m": list(p), "position_m": list(p),
                       "error_m": 0.0, "finite": True, "within_tolerance": True}
                  for n, p in physics.spawns.items()},
    }}


@pytest.mark.parametrize("malformation", [
    "legacy-ack", "empty", "string-names", "missing-check", "nan-pose",
    "outlier-ok", "invalid-tolerance",
])
def test_client_rejects_unmeasured_or_contradictory_reset_ack(reset_bridge, monkeypatch, malformation):
    arm, physics, env = reset_bridge
    response = _wire_reset_result(physics)
    verification = response["reset_verification"]
    if malformation == "legacy-ack":
        response = {"ok": True, "stdout": "settled cleanly"}
    elif malformation == "empty":
        response["props_reset"] = []
        verification["props"] = {}
    elif malformation == "string-names":
        response["props_reset"] = "pink_cube"
    elif malformation == "missing-check":
        del verification["props"]["pink_cube"]
    elif malformation == "nan-pose":
        verification["props"]["pink_cube"]["position_m"][0] = float("nan")
    elif malformation == "outlier-ok":
        # Explicit flags and a made-up zero error must NOT override positions.
        verification["props"]["pink_cube"]["position_m"][0] += 0.3
    else:
        verification["tolerance_m"] = float("inf")
    monkeypatch.setitem(env, "_reset_props_verified", lambda: response)

    with pytest.raises(BridgeError, match="reset"):
        _pump_main_thread(env, arm.reset_props)


@pytest.mark.parametrize("returned", ["failure", "empty"])
def test_dispatcher_does_not_trust_an_unchecked_backend_reply(
    reset_bridge, demo_cfg, tmp_path, returned,
):
    arm, physics, env = reset_bridge
    response = _wire_reset_result(physics)
    if returned == "failure":
        response["ok"] = False
        response["error"] = "reset failed"
    else:
        response["props_reset"] = []
        response["reset_verification"]["props"] = {}
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    # Driver boundary: an older adapter returns a dict instead of raising.
    runtime.arm.raw = SimpleNamespace(reset_props=lambda: response)
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))
    assert result["ok"] is False, result
    assert result["world"] == "isaac"
    assert result["props_reset"] == []
    assert "reset" in result["error"].lower()


@pytest.mark.parametrize("refusal", ["home-refused", "soft-stopped"])
def test_no_prop_teleport_after_motion_refusal(reset_bridge, demo_cfg, tmp_path, refusal):
    arm, physics, env = reset_bridge
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    if refusal == "home-refused":
        runtime.arm.move_joints = lambda *a, **k: False
    else:
        arm._stopped = True  # external stop signal, without faking reset_props
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))
    assert result["ok"] is False, result
    assert result["props_reset"] == []
    assert not any(event[0] == "teleport" for event in physics.events)


def test_paused_during_settle_is_not_a_physics_confirmation(reset_bridge, monkeypatch):
    _, physics, env = reset_bridge
    monkeypatch.setattr(env["_tl"], "is_playing", lambda: physics.steps == 0)
    handler = object.__new__(env["Handler"])
    result = _pump_main_thread(env, lambda: handler._dispatch({"op": "reset_props"}))
    assert result["ok"] is False, result
    assert "timeline" in result["error"]


def test_report_retains_measured_offset_instead_of_echoing_the_spawn(reset_bridge, monkeypatch):
    arm, physics, env = reset_bridge

    def update_with_offset():
        physics.update()
        physics.positions["pink_cube"][0] = physics.spawns["pink_cube"][0] + 0.006

    monkeypatch.setattr(env["app"], "update", update_with_offset)
    result = _pump_main_thread(env, arm.reset_props)
    check = result["reset_verification"]["props"]["pink_cube"]
    assert check["position_m"][0] == pytest.approx(0.176)
    assert check["spawn_position_m"][0] == pytest.approx(0.17)
    assert check["error_m"] == pytest.approx(0.006)


def test_handler_itself_rejects_nonfinite_readback_with_json_safe_evidence(reset_bridge, monkeypatch):
    _, physics, env = reset_bridge
    physics.positions["pink_cube"][0] = float("nan")
    monkeypatch.setitem(env, "_newton_teleport", lambda *a: False)
    handler = object.__new__(env["Handler"])
    result = _pump_main_thread(env, lambda: handler._dispatch({"op": "reset_props"}))
    assert result["ok"] is False, result
    assert result["props_reset"] == []
    assert "pink_cube" in result["error"]
    check = result["reset_verification"]["props"]["pink_cube"]
    assert check["finite"] is False and check["within_tolerance"] is False
    assert check["position_m"] is None and check["error_m"] is None
    json.dumps(result, allow_nan=False)  # no NaN/Infinity tokens on the wire


@pytest.mark.parametrize("code,ok", [("print('exec still works')", True),
                                    ("raise RuntimeError('exec failed')", False)])
def test_string_exec_jobs_keep_their_existing_protocol(reset_bridge, code, ok):
    _, _, env = reset_bridge
    handler = object.__new__(env["Handler"])
    result = _pump_main_thread(env, lambda: handler._dispatch({"op": "exec", "code": code}))
    assert result["ok"] is ok, result
    if ok:
        assert result["stdout"] == "exec still works\n"
    else:
        assert "exec failed" in result["error"]
