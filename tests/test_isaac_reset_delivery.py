"""Isaac reset DELIVERY CONTRACT tests, NOT an Isaac/GPU physics rehearsal.

Execute the actual runtime dispatcher, IsaacArm/BridgeClient, TCP Handler,
main-thread job queue and settle/reset helpers. Only Kit/physics, camera and
safe-arm motion boundaries are doubles. Their coordinates are test inputs,
never evidence that Newton or the shipped asset works on a DGX Spark.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib.util
import inspect
import json
import math
from pathlib import Path
import select
import socket
import socketserver
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from conftest import load_isaac_bridge_definitions, loopback_host
from owned_server import held_dead_port

from cascade.agent.trace import TraceLogger
from cascade.config import Cfg
from cascade.control.isaac_arm import IsaacArm
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.sim.bridge_client import BridgeError
from cascade.sim.isaac_reset import seeded_reset, validate_isaac_reset
from cascade.sim.scene_seed import kitchen_layout, seeded_spawns, validate_reset_seed
from cascade.skills.runtime import SkillRuntime
from cascade.types import Frame

REPO = Path(__file__).resolve().parents[1]
KITCHEN = json.loads((REPO / "demo/scene/kitchen_config.json").read_text())
BRIDGE = REPO / "scripts/isaac_bridge.py"
ISAAC_SCENE = REPO / "demo/isaac_scene.py"


def _load_isaac_scene(monkeypatch):
    """demo/isaac_scene.py as the bridge imports it. Its pure boot helpers
    (configure_cubes, PropSpec) never touch USD, so pxr is a bare stub."""
    pxr = ModuleType("pxr")
    for name in ("Gf", "Sdf", "Usd", "UsdGeom", "UsdPhysics", "UsdShade"):
        setattr(pxr, name, SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    spec = importlib.util.spec_from_file_location("isaac_scene_boot_under_test", ISAAC_SCENE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses resolve the module
    spec.loader.exec_module(module)
    return module


def _prepare_scene_prop_specs(isaac_scene, config):
    """The PropSpec records prepare_scene returns, built by its own record
    expression (the USD authoring around it needs Kit)."""
    function = next(node for node in ast.parse(ISAAC_SCENE.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "prepare_scene")
    records = [node.args[0] for node in ast.walk(function)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "append" and ast.unparse(node.func.value) == "specs"]
    assert len(records) == 1, "prepare_scene must build exactly one PropSpec record per prop"
    record = compile(ast.Expression(records[0]), str(ISAAC_SCENE), "eval")
    return [eval(record, {"PropSpec": isaac_scene.PropSpec},
                 {"item": item, "name": item["name"], "path": f"/World_Props/{item['name']}"})
            for item in config["props"]]


def _bridge_boot_prop_spawns(config, monkeypatch):
    """_PROP_SPAWNS exactly as the bridge builds it at boot for this kitchen
    config: its PROPS literal, `configure_cubes`, the scene's PropSpec records
    (`PROPS.extend`) and the `_PROP_SPAWNS` construction, run from the script."""
    isaac_scene = _load_isaac_scene(monkeypatch)

    def boot_step(node):
        text = ast.unparse(node)
        return (text.startswith(("PROPS = [", "PROPS.extend(", "_PROP_SPAWNS = "))
                or (isinstance(node, ast.If) and len(node.body) == 1
                    and ast.unparse(node.body[0]) == "PROPS = configure_cubes(_scene_config, PROPS)"))

    steps = [node for node in ast.parse(BRIDGE.read_text()).body if boot_step(node)]
    assert [ast.unparse(node).split(" ")[0].split("(")[0] for node in steps] == [
        "PROPS", "if", "PROPS.extend", "_PROP_SPAWNS"], "the bridge's prop boot path moved"
    env = {"args": SimpleNamespace(scene_config="demo/scene/kitchen_config.json"),
           "_scene_config": config, "configure_cubes": isaac_scene.configure_cubes,
           "_SCENE_PROP_SPECS": _prepare_scene_prop_specs(isaac_scene, config)}
    exec(compile(ast.Module(body=steps, type_ignores=[]), str(BRIDGE), "exec"), env)
    return env["_PROP_SPAWNS"]


class PhysicsBoundary:
    """Deterministic Kit/rigid-body stand-in; it does NOT simulate physics."""

    def __init__(self, spawns=None):
        self.spawns = dict(spawns) if spawns is not None else {
            "pink_cube": (0.17, 0.15, 0.04), "green_cube": (0.30, 0.16, 0.04)}
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
    wanted = {"Handler", "_run_exec_jobs", "_settle_props", "_zero_prop_velocity",
              "_reset_props_verified", "_reset_spawns", "_set_reset_seed"}
    prims = ModuleType("isaacsim.core.experimental.prims")
    prims.RigidPrim = physics.rigid_prim
    monkeypatch.setitem(sys.modules, prims.__name__, prims)
    env = {
        "__name__": "isaac_reset_contract_under_test", "np": np,
        "threading": threading, "socketserver": socketserver, "json": json,
        "time": time,
        "_exec_lock": threading.Lock(), "_exec_jobs": [],
        "_PROP_SPAWNS": physics.spawns, "BASE_Z": physics.base_z,
        "stage": SimpleNamespace(GetPrimAtPath=lambda p: p.rsplit("/", 1)[-1] in physics.positions),
        "args": SimpleNamespace(engine="newton"), "engine": "newton", "names": [],
        "app": SimpleNamespace(update=physics.update),
        "_tl": SimpleNamespace(is_playing=lambda: True),
        "_newton_teleport": physics.teleport,
        "_update_wrist_cam": lambda: None, "_refresh_frames": lambda: None,
        "_annotators": {},  # This fixture has no render products/publications.
        # Seeded-reset state: no seed and no kitchen config, as a plain bridge boots.
        "_RESET_SEED": None, "_scene_config": None,
        "seeded_spawns": seeded_spawns, "kitchen_layout": kitchen_layout,
        "validate_reset_seed": validate_reset_seed,
    }
    # This suite isolates the reset step count/delivery; frame history has
    # separate real-wrapper tests and never inserts an extra Kit update.
    env["_step_with_frame_history"] = lambda: env["app"].update()
    load_isaac_bridge_definitions(wanted, env)
    return env


def test_reset_keeps_camera_captures_fresh_during_nested_physics_steps(monkeypatch):
    physics = PhysicsBoundary()
    env = _load_bridge_definitions(monkeypatch, physics)
    clock = lambda: physics.steps * .05
    env["time"] = SimpleNamespace(monotonic=clock)
    captures, ages, order = [0.0], [], []

    def update():
        physics.update()
        order.append(("physics", physics.steps))
        ages.append(clock() - captures[-1])

    def wrist():
        physics.main_thread()
        order.append(("wrist", physics.steps))

    def capture():
        physics.main_thread()
        assert order[-2:] == [("wrist", physics.steps - 1), ("physics", physics.steps)]
        captures.append(clock())
        order.append(("camera", physics.steps))

    env.update(app=SimpleNamespace(update=update),
               _update_wrist_cam=wrist, _refresh_frames=capture)

    result = env["_reset_props_verified"]()

    assert result["ok"] and set(result["props_reset"]) == set(physics.spawns)
    assert physics.steps == 180  # The existing successful settle sequence.
    assert len(captures) > 2 and captures[-1] < clock()
    assert max(ages) < .7  # The viewer's two-second stale limit is never reached.


def test_camera_readback_failure_does_not_abort_prop_reset(monkeypatch, capsys):
    physics = PhysicsBoundary()
    env = _load_bridge_definitions(monkeypatch, physics)
    env["time"] = SimpleNamespace(monotonic=lambda: physics.steps * .05)

    def failed_capture():
        physics.main_thread()
        raise RuntimeError("camera unavailable")

    env["_refresh_frames"] = failed_capture

    result = env["_reset_props_verified"]()

    assert result["ok"] and set(result["props_reset"]) == set(physics.spawns)
    assert physics.steps == 180
    assert capsys.readouterr().out.count("reset camera refresh failed") == 1


@contextmanager
def _served_bridge(monkeypatch, physics, scene_config=None):
    env = _load_bridge_definitions(monkeypatch, physics)
    env["_scene_config"] = scene_config
    server = socketserver.ThreadingTCPServer((loopback_host(), 0), env["Handler"])
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    arm = IsaacArm(Cfg({"bridge_host": loopback_host(), "bridge_port": server.server_address[1]}))
    arm.connect()
    try:
        yield arm, env
    finally:
        arm.disconnect()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def reset_bridge(monkeypatch):
    physics = PhysicsBoundary()
    with _served_bridge(monkeypatch, physics) as (arm, env):
        yield arm, physics, env


@pytest.fixture
def kitchen_reset_bridge(monkeypatch):
    """The bridge with the shipped kitchen loaded: its props where the
    bridge's own boot path spawns them, and the scene config."""
    physics = PhysicsBoundary(_bridge_boot_prop_spawns(KITCHEN, monkeypatch))
    with _served_bridge(monkeypatch, physics, scene_config=KITCHEN) as (arm, env):
        yield arm, physics, env


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

    def camera_frame():
        return Frame(rgb=np.zeros((12, 12, 3), np.uint8), depth_m=None, K=np.eye(3))

    jaws = []
    safe_arm = SimpleNamespace(raw=arm, move_joints=home,
        get_state=lambda: SimpleNamespace(q=np.zeros(6), gripper_pos=1.0),
        # 1.0 reads CLOSED on the mock profile (open_pos 0.0): reset_scene
        # opens closed empty jaws (PR-A jaws round), recorded here.
        set_gripper=lambda pos, effort=1.0, **_kw: jaws.append(float(pos)), jaws=jaws,
        harness=SimpleNamespace(heartbeat=lambda: None))
    return SkillRuntime(
        camera=SimpleNamespace(get_frame=camera_frame),
        depth_provider=SimpleNamespace(ensure_depth=lambda frame: frame),
        detector=SimpleNamespace(detect=lambda *a, **kw: []),
        extrinsics=None, kin=SimpleNamespace(fk=lambda q: np.eye(4)),
        safe_arm=safe_arm, memory=EpisodicMemory(),
        beliefs=BeliefStore(), trace=TraceLogger(tmp_path / "run"), cfg=demo_cfg,
    )


def test_dispatcher_records_measured_isaac_reset_in_its_own_trace(reset_bridge, demo_cfg, tmp_path):
    arm, physics, env = reset_bridge
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))

    assert result["ok"] is True, result
    assert result["observation_refreshed"] is True
    assert result["world"] == "isaac", result
    assert result["props_reset"] == list(physics.spawns), result
    assert result["jaws_open_commanded"] is True and runtime.arm.jaws == [runtime._grip_open]  # empty, closed
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


def test_measured_reset_with_failed_camera_is_not_complete_success(reset_bridge, demo_cfg, tmp_path):
    arm, physics, env = reset_bridge
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    def missing_camera():
        raise RuntimeError("camera boundary unavailable")
    runtime.camera.get_frame = missing_camera
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))
    assert result["ok"] is False and result["observation_refreshed"] is False
    assert result["props_reset"] == list(physics.spawns)
    assert result["reset_verification"]["channel"] == "physics"
    assert result["objects_visible"] == [] and result["observe_error"]


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


def test_reset_slow_capture_uses_attempt_start_without_extra_physics(monkeypatch):
    physics = PhysicsBoundary()
    env = _load_bridge_definitions(monkeypatch, physics)
    now, captures, order = [0.], [], []
    env["time"] = SimpleNamespace(monotonic=lambda: now[0])

    def update():
        physics.update()
        order.append("update")
        now[0] += .01

    def capture():
        assert order[-2:] == ["wrist", "update"]
        captures.append(now[0])
        now[0] += .8
        order.append("capture")

    env.update(app=SimpleNamespace(update=update),
               _update_wrist_cam=lambda: order.append("wrist"), _refresh_frames=capture)
    result = env["_reset_props_verified"]()
    assert result["ok"]
    assert physics.steps == len(captures) == 180
    assert np.diff(captures) == pytest.approx(np.full(179, .81))
    assert env["_last_camera_capture_started"] == pytest.approx(captures[-1] - .01)


# -- Seeded reset (ASPIRE verifier scenes): bridge op set_reset_seed ---------


def _dispatch(env, req):
    handler = object.__new__(env["Handler"])
    return _pump_main_thread(env, lambda: handler._dispatch(req))


def _kitchen_seeded(seed):
    spawns, footprints, keepouts = kitchen_layout(KITCHEN)
    return seeded_spawns(spawns, seed, footprints=footprints, keepouts=keepouts)


def _spawns_of(reply):
    return {n: c["spawn_position_m"] for n, c in reply["reset_verification"]["props"].items()}


def test_set_reset_seed_then_reset_places_and_verifies_seeded_spawns(kitchen_reset_bridge):
    _, physics, env = kitchen_reset_bridge
    assert _dispatch(env, {"op": "set_reset_seed", "seed": 5}) == {"ok": True, "seed": 5}
    reply = _dispatch(env, {"op": "reset_props"})
    want, fallbacks = _kitchen_seeded(5)
    assert reply["ok"] is True, reply
    assert reply["seed"] == 5 and reply["seed_fallbacks"] == fallbacks
    assert _spawns_of(reply) == {n: list(p) for n, p in want.items()}
    # One clean settle pass on the seeded spots: no retries against the old spawns.
    assert physics.steps == 180 and "settled cleanly (attempt 1)" in reply["stdout"]
    for name, spawn in want.items():
        check = reply["reset_verification"]["props"][name]
        # The settle teleported the prop to the seeded spot, and the read-back confirms it.
        assert check["position_m"] == pytest.approx(list(spawn))
        assert check["error_m"] <= 0.02 and check["within_tolerance"] is True
    # The seeded layout differs from the default by more than the 2 cm tolerance.
    assert max(math.dist(p[:2], physics.spawns[n][:2]) for n, p in want.items()) > 0.02
    assert validate_isaac_reset(reply) is reply
    again = _dispatch(env, {"op": "reset_props"})
    assert again["seed"] == 5 and _spawns_of(again) == _spawns_of(reply)


def test_no_seed_is_todays_reset(kitchen_reset_bridge):
    _, physics, env = kitchen_reset_bridge
    before = _dispatch(env, {"op": "reset_props"})
    assert before["ok"] is True, before
    assert set(before) == {"ok", "props_reset", "reset_verification", "stdout"}
    assert _spawns_of(before) == {n: list(p) for n, p in physics.spawns.items()}
    assert _dispatch(env, {"op": "set_reset_seed", "seed": 5})["ok"] is True
    assert _dispatch(env, {"op": "reset_props"})["seed"] == 5
    # Clearing the seed restores today's reset, byte for byte.
    assert _dispatch(env, {"op": "set_reset_seed", "seed": None}) == {"ok": True, "seed": None}
    after = _dispatch(env, {"op": "reset_props"})
    assert json.dumps(after) == json.dumps(before)


def test_seed_zero_is_the_default_layout_with_its_seed_stamped(kitchen_reset_bridge):
    _, physics, env = kitchen_reset_bridge
    assert _dispatch(env, {"op": "set_reset_seed", "seed": 0}) == {"ok": True, "seed": 0}
    reply = _dispatch(env, {"op": "reset_props"})
    assert reply["ok"] is True and reply["seed"] == 0 and reply["seed_fallbacks"] == []
    assert _spawns_of(reply) == {n: list(p) for n, p in physics.spawns.items()}


def test_a_prop_the_seed_cannot_place_stays_on_its_spawn_and_is_reported(kitchen_reset_bridge):
    _, physics, env = kitchen_reset_bridge
    # A (hypothetical) green square painted under the lemon: no seeded spot fits it.
    config = json.loads(json.dumps(KITCHEN))
    config["target_pad"].update(center_xy_m=list(physics.spawns["lemon"][:2]), outer_size_m=0.10)
    env["_scene_config"] = config
    spawns, footprints, keepouts = kitchen_layout(config)
    want, fallbacks = seeded_spawns(spawns, 6, footprints=footprints, keepouts=keepouts)
    assert fallbacks == ["lemon"]
    assert _dispatch(env, {"op": "set_reset_seed", "seed": 6})["ok"] is True
    reply = _dispatch(env, {"op": "reset_props"})
    assert reply["ok"] is True and reply["seed_fallbacks"] == ["lemon"], reply
    assert _spawns_of(reply) == {n: list(p) for n, p in want.items()}
    assert _spawns_of(reply)["lemon"] == list(physics.spawns["lemon"])
    assert validate_isaac_reset(reply) is reply


@pytest.mark.parametrize("bad", [{"seed": "5"}, {"seed": 5.0}, {"seed": True}, {"seed": -1},
                                 {"seed": 2**31}, {}],
                         ids=["string", "float", "bool", "negative", "too-large", "missing"])
def test_set_reset_seed_refuses_a_bad_seed_and_keeps_the_previous_one(kitchen_reset_bridge, bad):
    _, physics, env = kitchen_reset_bridge
    assert _dispatch(env, {"op": "set_reset_seed", "seed": 3})["ok"] is True
    reply = _dispatch(env, {"op": "set_reset_seed", **bad})
    assert reply["ok"] is False and "seed" in reply["error"], reply
    assert env["_RESET_SEED"] == 3
    assert physics.events == [], "a seed request never moves a prop"
    assert _dispatch(env, {"op": "reset_props"})["seed"] == 3


@pytest.mark.parametrize("scene,reason", [("no-scene-config", "--scene-config"),
                                          ("props-differ-from-config", "differ from this bridge")])
def test_set_reset_seed_needs_the_kitchen_layout_of_these_props(kitchen_reset_bridge, scene, reason):
    _, physics, env = kitchen_reset_bridge
    if scene == "no-scene-config":
        env["_scene_config"] = None
    else:
        physics.spawns["pink_cube"] = (0.17, 0.15, 0.03)  # not where the config spawns it
    reply = _dispatch(env, {"op": "set_reset_seed", "seed": 4})
    assert reply["ok"] is False and "set_reset_seed refused" in reply["error"], reply
    assert reason in reply["error"]
    assert env["_RESET_SEED"] is None
    assert physics.events == []


def test_the_bridge_boot_path_spawns_the_shipped_kitchen_on_its_layout(monkeypatch):
    booted = _bridge_boot_prop_spawns(KITCHEN, monkeypatch)
    assert set(booted) == {"pink_cube", "green_cube", "tomato_can", "lemon", "orange"}
    assert booted == kitchen_layout(KITCHEN)[0]  # what set_reset_seed compares at run time


@pytest.mark.parametrize("drift", ["prop-moved", "cube-moved", "cube-not-in-config"])
def test_a_config_that_drifted_from_the_booted_props_is_refused(monkeypatch, drift):
    config = json.loads(json.dumps(KITCHEN))
    booted_from = KITCHEN
    if drift == "prop-moved":
        next(p for p in config["props"] if p["name"] == "lemon")["position"][0] += 0.01
    elif drift == "cube-moved":
        config["cube_positions"]["pink_cube"][1] += 0.01
    else:
        # One config boots and seeds: the bridge keeps its built-in green_cube
        # spawn for a cube the config omits; the seeded layout knows no such cube.
        del config["cube_positions"]["green_cube"], config["cube_dimensions"]["green_cube"]
        booted_from = config
    physics = PhysicsBoundary(_bridge_boot_prop_spawns(booted_from, monkeypatch))
    with _served_bridge(monkeypatch, physics, scene_config=config) as (_, env):
        reply = _dispatch(env, {"op": "set_reset_seed", "seed": 4})
        assert reply["ok"] is False and "differ from this bridge" in reply["error"], reply
        assert env["_RESET_SEED"] is None
        env["_RESET_SEED"] = 4  # armed before the drift: the reset itself refuses too
        reply = _dispatch(env, {"op": "reset_props"})
        assert reply["ok"] is False and "differ from this bridge" in reply["error"], reply
    assert physics.events == [], "a drifted layout must be refused BEFORE any prop moves"


def test_the_bridge_boots_with_no_reset_seed():
    # Production state the test loader mirrors (conftest): ping reads it from boot on.
    boots = [node for node in ast.parse(BRIDGE.read_text()).body if isinstance(node, ast.Assign)
             and [ast.unparse(t) for t in node.targets] == ["_RESET_SEED"]]
    assert len(boots) == 1 and ast.literal_eval(boots[0].value) is None


def test_the_bridge_has_no_reset_seed_before_it_serves():
    # ping reads _RESET_SEED on a socket thread: a ping that arrives before the
    # boot statement ran would fail with NameError instead of answering.
    body = ast.parse(BRIDGE.read_text()).body

    def where(match):
        found = [i for i, node in enumerate(body) if match(node)]
        assert len(found) == 1, found
        return found[0]

    seed = where(lambda node: isinstance(node, ast.Assign)
                 and [ast.unparse(t) for t in node.targets] == ["_RESET_SEED"])
    server = where(lambda node: isinstance(node, ast.Assign)
                   and [ast.unparse(t) for t in node.targets] == ["server"]
                   and "ThreadingTCPServer(" in ast.unparse(node.value))
    serving = where(lambda node: isinstance(node, ast.Expr)
                    and "target=server.serve_forever" in ast.unparse(node) and ".start()" in ast.unparse(node))
    assert seed < server < serving


def test_a_seeded_reset_that_cannot_build_its_layout_moves_nothing(kitchen_reset_bridge):
    _, physics, env = kitchen_reset_bridge
    assert _dispatch(env, {"op": "set_reset_seed", "seed": 4})["ok"] is True
    env["_scene_config"] = None
    reply = _dispatch(env, {"op": "reset_props"})
    assert reply["ok"] is False and reply.get("props_reset", []) == [], reply
    assert physics.events == [], "an unbuildable layout must be refused BEFORE settling"


def test_a_seed_set_during_a_reset_applies_to_the_next_one(kitchen_reset_bridge, monkeypatch):
    _, _, env = kitchen_reset_bridge
    assert _dispatch(env, {"op": "set_reset_seed", "seed": 5})["ok"] is True
    settle = env["_settle_props"]

    def settle_then_reseed(*args, **kwargs):
        assert env["_set_reset_seed"]({"seed": 9}) == {"ok": True, "seed": 9}
        return settle(*args, **kwargs)

    monkeypatch.setitem(env, "_settle_props", settle_then_reseed)
    reply = _dispatch(env, {"op": "reset_props"})
    assert reply["ok"] is True and reply["seed"] == 5, reply
    assert _spawns_of(reply) == {n: list(p) for n, p in _kitchen_seeded(5)[0].items()}
    assert env["_RESET_SEED"] == 9


@pytest.mark.parametrize("change,valid", [
    ({}, True),
    ({"seed": 5, "seed_fallbacks": []}, True),
    ({"seed": 5, "seed_fallbacks": ["pink_cube"]}, True),
    ({"seed": 0, "seed_fallbacks": []}, True),
    ({"seed": None}, True),
    ({"seed": "5", "seed_fallbacks": []}, False),
    ({"seed": 5.0, "seed_fallbacks": []}, False),
    ({"seed": True, "seed_fallbacks": []}, False),
    ({"seed": -1, "seed_fallbacks": []}, False),
    ({"seed": 5}, False),
    ({"seed_fallbacks": []}, False),
    ({"seed": None, "seed_fallbacks": ["pink_cube"]}, False),
    ({"seed": 5, "seed_fallbacks": "pink_cube"}, False),
    ({"seed": 5, "seed_fallbacks": ["ghost"]}, False),
    ({"seed": 5, "seed_fallbacks": ["pink_cube", "pink_cube"]}, False),
])
def test_validator_accepts_an_integer_seed_and_rejects_others(change, valid):
    response = {**_wire_reset_result(PhysicsBoundary()), **change}
    if valid:
        assert validate_isaac_reset(response) is response
    else:
        with pytest.raises(BridgeError, match="seed"):
            validate_isaac_reset(response)


def test_reset_scene_result_carries_the_seed(kitchen_reset_bridge, demo_cfg, tmp_path):
    arm, physics, env = kitchen_reset_bridge
    assert arm._client.request({"op": "set_reset_seed", "seed": 5}) == {"ok": True, "seed": 5}
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))
    assert result["ok"] is True, result
    assert result["scene_seed"] == 5
    want, _ = _kitchen_seeded(5)
    assert {n: c["spawn_position_m"] for n, c in result["reset_verification"]["props"].items()} == {
        n: list(p) for n, p in want.items()}
    row = json.loads((tmp_path / "run" / "trace.jsonl").read_text())
    assert row["result"]["scene_seed"] == 5


def test_unseeded_reset_scene_result_has_no_seed(kitchen_reset_bridge, demo_cfg, tmp_path):
    arm, physics, env = kitchen_reset_bridge
    runtime = _runtime(arm, physics, demo_cfg, tmp_path)
    result = _pump_main_thread(env, lambda: runtime.execute("reset_scene", {}))
    assert result["ok"] is True, result
    assert "scene_seed" not in result and "seed" not in result


# -- Seed visibility (ping) and the harness helper seeded_reset --------------

# A kitchen identity as the bridge attaches it at startup (Handler.scene_identity).
_IDENTITY = {"scene_config": "/repo/demo/scene/kitchen_config.json", "scene_config_sha256": "c9a41e06",
             "scene_name": "paai-cocina-asier-v1", "scene_assets_sha256": "6c4c3cb2",
             "scene_content_sha256": "a0031a9e"}


def _port_of(arm):
    return arm._client._addr[1]


def _raw_ping(port):
    """The exact bytes the bridge writes for one ping, on a fresh connection."""
    with socket.create_connection((loopback_host(), port), timeout=5) as conn:
        conn.sendall(b'{"op": "ping"}\n')
        return conn.makefile("rb").readline()


def test_ping_shows_a_set_reset_seed_and_is_unchanged_without_one(kitchen_reset_bridge, monkeypatch):
    arm, _, env = kitchen_reset_bridge
    monkeypatch.setattr(env["Handler"], "scene_identity", _IDENTITY)
    port = _port_of(arm)
    unseeded = _raw_ping(port)
    # Without a seed: main's ping reply ({ok, engine, dofs, **identity}), byte for byte.
    assert unseeded == json.dumps({"ok": True, "engine": "newton", "dofs": [], **_IDENTITY}).encode() + b"\n"
    for seed in (5, 0):  # seed 0 is a set seed too (default layout, stamped)
        assert _dispatch(env, {"op": "set_reset_seed", "seed": seed})["ok"] is True
        assert json.loads(_raw_ping(port)) == {**json.loads(unseeded), "reset_seed": seed}
    assert _dispatch(env, {"op": "set_reset_seed", "seed": None})["ok"] is True
    assert _raw_ping(port) == unseeded


def test_seeded_reset_sets_the_seed_for_the_block_and_clears_it_after(kitchen_reset_bridge):
    arm, _, env = kitchen_reset_bridge
    with seeded_reset(_port_of(arm), 5, host=loopback_host()) as client:
        assert env["_RESET_SEED"] == 5
        assert client.request({"op": "ping"})["reset_seed"] == 5
        assert _dispatch(env, {"op": "reset_props"})["seed"] == 5
    assert env["_RESET_SEED"] is None
    assert "reset_seed" not in _dispatch(env, {"op": "ping"})
    assert "seed" not in _dispatch(env, {"op": "reset_props"})


def test_seeded_reset_clears_the_seed_when_the_block_raises(kitchen_reset_bridge):
    arm, _, env = kitchen_reset_bridge
    with pytest.raises(RuntimeError, match="harness episode failed"), \
            seeded_reset(_port_of(arm), 7, host=loopback_host()):
        assert env["_RESET_SEED"] == 7
        raise RuntimeError("harness episode failed")
    assert env["_RESET_SEED"] is None


def test_seeded_reset_clears_over_a_new_connection_when_the_block_lost_it(kitchen_reset_bridge):
    arm, _, env = kitchen_reset_bridge
    with seeded_reset(_port_of(arm), 7, host=loopback_host()) as client:
        client.close()  # what a timed-out request in the block does to the shared client
    assert env["_RESET_SEED"] is None


@pytest.mark.parametrize("in_flight", [{"op": "ping"}, {"op": "set_reset_seed", "seed": 5}],
                         ids=["ping", "set_reset_seed"])
def test_seeded_reset_clears_on_a_fresh_connection_and_keeps_a_keyboard_interrupt(
        kitchen_reset_bridge, in_flight):
    arm, _, env = kitchen_reset_bridge
    interrupt = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as raised, \
            seeded_reset(_port_of(arm), 5, host=loopback_host()) as client:
        # Ctrl-C after a request was sent: its reply arrives and is never read.
        sock = client._sock
        assert sock is not None
        sock.sendall(json.dumps(in_flight).encode() + b"\n")
        assert select.select([sock], [], [], 5)[0], "the bridge never answered"
        raise interrupt
    # The clear never took that stale reply for its own: the seed is gone and
    # the operator's interrupt is what propagates, not a BridgeError.
    assert raised.value is interrupt
    assert env["_RESET_SEED"] is None


@pytest.mark.parametrize("seed", [5, 0])
def test_seeded_reset_reports_a_set_that_lands_after_its_clear(kitchen_reset_bridge, monkeypatch, seed):
    arm, _, env = kitchen_reset_bridge
    real, cleared, landed = env["_set_reset_seed"], threading.Event(), threading.Event()

    def slowed(req):
        if req["seed"] is None:
            reply = real(req)
            cleared.set()
            landed.wait(5)  # the late set lands before the clear is acknowledged
            return reply
        # Slower than the client's wait: the helper gives up and clears, then this lands.
        if cleared.wait(5):
            reply = real(req)
            landed.set()
            return reply
        return {"ok": False, "error": "the helper never cleared"}

    monkeypatch.setitem(env, "_set_reset_seed", slowed)
    with pytest.raises(BridgeError, match="after its clear"), \
            seeded_reset(_port_of(arm), seed, host=loopback_host(), timeout_s=0.5):
        pytest.fail("the block must not run: the bridge never confirmed the seed")
    assert env["_RESET_SEED"] == seed  # really armed again: the helper must not stay silent


def test_seeded_reset_reports_a_clear_it_could_not_check(kitchen_reset_bridge, monkeypatch):
    arm, _, env = kitchen_reset_bridge
    dispatch = env["Handler"]._dispatch

    def no_ping_once_cleared(self, req):
        if req.get("op") == "ping" and env["_RESET_SEED"] is None:
            return {"ok": False, "error": "ping unavailable"}
        return dispatch(self, req)

    monkeypatch.setattr(env["Handler"], "_dispatch", no_ping_once_cleared)
    with pytest.raises(BridgeError, match="could not check the reset seed clear on port .*: ping unavailable"), \
            seeded_reset(_port_of(arm), 5, host=loopback_host()):
        assert env["_RESET_SEED"] == 5
    assert env["_RESET_SEED"] is None  # the clear itself landed; only its check failed


def test_seeded_reset_refused_by_the_bridge_raises_and_clears_a_leaked_seed(kitchen_reset_bridge):
    arm, _, env = kitchen_reset_bridge
    env["_RESET_SEED"] = 3  # left behind by an earlier client
    env["_scene_config"] = None  # this bridge can build no seeded layout
    entered = []
    with pytest.raises(BridgeError, match="set_reset_seed refused"), \
            seeded_reset(_port_of(arm), 5, host=loopback_host()):
        entered.append(True)
    assert entered == [] and env["_RESET_SEED"] is None


@pytest.mark.parametrize("answer,reason", [
    ({"ok": False, "error": "busy"}, "could not clear the reset seed on port .*: busy"),
    ({"ok": True, "seed": 5}, "did not confirm the reset seed clear")],
    ids=["refused", "acknowledged-without-clearing"])
def test_seeded_reset_reports_a_seed_it_could_not_clear(kitchen_reset_bridge, monkeypatch, answer, reason):
    arm, _, env = kitchen_reset_bridge
    with pytest.raises(BridgeError, match=reason), seeded_reset(_port_of(arm), 5, host=loopback_host()):
        monkeypatch.setitem(env, "_set_reset_seed", lambda req: answer)
    assert env["_RESET_SEED"] == 5  # really still set: the helper must not stay silent


def test_seeded_reset_needs_the_bridge_to_confirm_its_seed(kitchen_reset_bridge, monkeypatch):
    arm, _, env = kitchen_reset_bridge
    real, sent, entered = env["_set_reset_seed"], [], []

    def other_seed(req):  # acknowledges a different seed than the one asked for
        sent.append(req["seed"])
        return real({"seed": 6}) if req["seed"] is not None else real(req)

    monkeypatch.setitem(env, "_set_reset_seed", other_seed)
    with pytest.raises(BridgeError, match="did not confirm reset seed 5"), \
            seeded_reset(_port_of(arm), 5, host=loopback_host()):
        entered.append(True)
    assert entered == [] and sent == [5, None] and env["_RESET_SEED"] is None


def test_seeded_reset_has_no_default_port():
    # The live demo bridge listens on 8611: a harness must name its own bridge.
    assert inspect.signature(seeded_reset).parameters["port"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        seeded_reset(seed=5)


@pytest.mark.parametrize("port", ["str", "float", None, True, 0, 65536, -1])
def test_seeded_reset_refuses_a_bad_port_before_connecting(port):
    with held_dead_port() as dead:
        port = {"str": str(dead), "float": float(dead)}.get(port, port)
        # Without the check, a str/float would reach the dead port: BridgeError, not ValueError.
        with pytest.raises(ValueError, match="port"), seeded_reset(port, 5, timeout_s=0.5):
            pytest.fail("the block must not run")


@pytest.mark.parametrize("seed", [None, "5", 5.0, True, -1, 2**31])
def test_seeded_reset_refuses_a_bad_seed_before_connecting(seed):
    with held_dead_port() as dead, pytest.raises(ValueError, match="seed"), \
            seeded_reset(dead, seed, timeout_s=0.5):
        pytest.fail("the block must not run")
