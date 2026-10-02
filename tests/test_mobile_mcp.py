"""Native stdio mobile MCP: real child and real mock motion, no server stub."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import pytest
from mobile_support_fixture import support_contract
import test_mobile_frames

frame_endpoint = test_mobile_frames.frame_endpoint

REPO = Path(__file__).resolve().parents[1]


class MobileClient:
    def __init__(self, tmp_path, *, config_dir=None, **env_extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
        env.update(CASCADE_BASE="microduck_mock", CASCADE_PREWARM="0",
                   CASCADE_STREAM="0", CASCADE_VIEW="0", CASCADE_RUN_DIR=str(tmp_path),
                   CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(REPO / "src"))
        env.update(env_extra)
        # The child does not inherit pytest's autouse memory fixtures after
        # CASCADE_* is scrubbed above. Keep all persistent stores private.
        env.update(CASCADE_BELIEFS_PATH=str(tmp_path / "beliefs.json"),
                   CASCADE_GRASP_MEMORY_PATH=str(tmp_path / "grasp.json"),
                   CASCADE_ENVELOPE_PATH=str(tmp_path / "envelope.json"))
        command = [sys.executable, "-m", "cascade.apps.mcp_server"]
        if config_dir is not None:
            # Real MCP main, with isolated profiles only; no server substitute.
            command = [sys.executable, "-c", "import sys; from pathlib import Path; "
                       "import cascade.config as c; c.CONFIG_DIR=Path(sys.argv.pop()); "
                       "from cascade.apps.mcp_server import main; raise SystemExit(main())", str(config_dir)]
        self.proc = subprocess.Popen(command,
                                     cwd=REPO, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1)
        self.lines, self.errors = queue.Queue(), []
        self.next_id = 0
        self.out_thread = threading.Thread(target=self._pump, daemon=True)
        self.err_thread = threading.Thread(target=self._errors, daemon=True)
        self.out_thread.start()
        self.err_thread.start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _errors(self):
        self.errors.extend(self.proc.stderr)

    def send(self, method, params=None, notification=False):
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notification:
            msg["id"] = self.next_id
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        return self.next_id

    def recv(self, timeout=8):
        line = self.lines.get(timeout=timeout)
        assert line is not None, "".join(self.errors)
        return json.loads(line)

    def request(self, method, params=None):
        ident = self.send(method, params)
        response = self.recv()
        assert response["id"] == ident
        return response

    def call(self, name, args=None):
        response = self.request("tools/call", {"name": name, "arguments": args or {}})
        return payload(response)

    def close(self):
        try:
            if not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise
        finally:
            self.out_thread.join(2)
            self.err_thread.join(2)
            self.proc.stdout.close()
            self.proc.stderr.close()


def payload(response):
    assert "error" not in response, response
    return json.loads(response["result"]["content"][-1]["text"])


@pytest.fixture
def mobile_client(tmp_path):
    client = MobileClient(tmp_path / "mcp")
    try:
        yield client
    finally:
        client.close()


def test_startup_stop_ack_is_explicitly_unverified_with_local_receipt(mobile_client):
    ack = mobile_client.call("emergency_stop")
    assert ack["physical_stop_verified"] is False
    assert ack["outcome"] == "unverified"
    assert ack["receipt_id"] and ack["receipt_scope"] == "mcp_startup_local"
    assert isinstance(ack["serial"], int)


def test_mobile_catalog_and_complete_mock_episode(mobile_client):
    c = mobile_client
    assert c.request("initialize", {"protocolVersion": "2025-06-18"})["result"]["serverInfo"]["name"] == "cascade"
    c.send("notifications/initialized", notification=True)
    tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
    assert {"list_bases", "get_base_state", "walk_velocity", "turn", "stop_navigation",
            "emergency_stop", "reset_stop", "verify_last_action", "task_memory"} <= tools.keys()
    assert not {"grasp_object", "move_joints", "pick_and_place", "list_arms", "task_done"} & tools.keys()
    assert "base" in tools["walk_velocity"]["inputSchema"]["properties"]
    assert "arm" not in tools["walk_velocity"]["inputSchema"]["properties"]
    assert not c.call("grasp_object", {"label": "cube"})["ok"]
    assert c.call("list_bases")["bases"][0]["measurement_kind"] == "kinematic_mock"
    assert c.call("get_observation")["frame"] is None
    result = c.call("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .08})
    assert result["execution_ok"] is True and result["ok"] is False
    assert result["postcondition"]["status"] == "unverified"
    turn = c.call("turn", {"angle_rad": -.08})
    assert turn["execution_ok"] is True and turn["ok"] is False
    assert c.call("verify_last_action")["recent"][-1]["status"] == "unverified"
    assert c.call("task_memory")["steps_recorded"]


@pytest.mark.parametrize("kind", ["emergency_stop", "stop_navigation", "cancel", "eof"])
def test_reader_interrupts_running_motion_before_duration(mobile_client, kind):
    c = mobile_client
    assert c.call("get_base_state")["ok"]  # child up and connected before the timed trial
    motion = c.send("tools/call", {"name": "walk_velocity", "arguments": {
        "vx": .05, "vy": 0., "wz": 0., "duration_s": 3.0}})
    time.sleep(.18)
    started = time.monotonic()
    if kind == "cancel":
        c.send("notifications/cancelled", {"requestId": motion}, notification=True)
    elif kind == "eof":
        c.proc.stdin.close()
    else:
        stop_id = c.send("tools/call", {"name": kind, "arguments": {}})
        messages = [c.recv(1.5), c.recv(1.5)]
        stopped = payload(next(r for r in messages if r["id"] == stop_id))
        assert stopped["latched"] is True
        response = next(r for r in messages if r["id"] == motion)
    if kind in {"cancel", "eof"}:
        response = c.recv(1.5)
    assert time.monotonic() - started < 1.5
    result = payload(response)
    assert not result["ok"] and not result.get("execution_ok", False)
    assert result["measured"]["before"] is not None  # genuinely started, not merely cancelled in a queue
    if kind == "eof":
        c.proc.wait(timeout=2)
        assert c.proc.returncode == 0
    else:
        before = c.call("get_base_state")["state"]
        assert before["latched"]
        assert c.call("reset_stop")["ok"]
        time.sleep(.05)
        after = c.call("get_base_state")["state"]
        assert after["position_world"] == before["position_world"]
        assert after["controller_status"] == "ready"


def test_mobile_metadata_list_never_builds_runtime(monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer
    monkeypatch.setenv("CASCADE_BASE", "microduck_mock")
    server = McpSkillServer()
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata actuated/initialized the runtime")
    monkeypatch.setattr(server, "_ensure_runtime", forbidden)
    assert "walk_velocity" in {s["name"] for s in server.list_tools()}
    assert json.loads(server.call_tool("list_bases", {})["content"][-1]["text"])["ok"]
    assert server._runtime is None
    assert server.call_tool("grasp_object", {"label": "cube"})["isError"]


def test_stop_invalidates_queued_commands_even_if_reset_is_ahead_of_them(mobile_client):
    c = mobile_client
    assert c.call("get_base_state")["ok"]
    first = c.send("tools/call", {"name": "walk_velocity", "arguments": {
        "vx": .05, "vy": 0., "wz": 0., "duration_s": 3.0}})
    time.sleep(.15)
    reset, old_motion, stop = c.next_id + 1, c.next_id + 2, c.next_id + 3
    c.next_id = stop
    frames = [
        {"jsonrpc": "2.0", "id": reset, "method": "tools/call", "params": {"name": "reset_stop"}},
        {"jsonrpc": "2.0", "id": old_motion, "method": "tools/call", "params": {
            "name": "walk_velocity", "arguments": {"vx": .05, "vy": 0., "wz": 0., "duration_s": .06}}},
        {"jsonrpc": "2.0", "id": stop, "method": "tools/call", "params": {"name": "emergency_stop"}},
    ]
    c.proc.stdin.write(json.dumps(frames) + "\n")
    c.proc.stdin.flush()
    responses = {r["id"]: payload(r) for r in [c.recv(), c.recv(), c.recv(), c.recv()]}
    assert not responses[first]["ok"]
    assert responses[reset]["ok"]
    assert not responses[old_motion].get("execution_ok", False)
    assert "invalidated" in responses[old_motion]["error"]


def test_stop_during_actual_runtime_build_remains_latched(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from cascade.apps import demo
    from cascade.apps.mcp_server import McpSkillServer

    monkeypatch.setenv("CASCADE_BASE", "microduck_mock")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("CASCADE_VIEW", "0")
    monkeypatch.setenv("CASCADE_STREAM", "0")
    entered, release = threading.Event(), threading.Event()
    build = demo.build_runtime
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return build(*args, **kwargs)
    monkeypatch.setattr(demo, "build_runtime", delayed)
    server = McpSkillServer()
    with ThreadPoolExecutor() as pool:
        motion = pool.submit(server.call_tool, "walk_velocity", {
            "vx": .05, "vy": 0., "wz": 0., "duration_s": .06})
        try:
            assert entered.wait(2)
            assert json.loads(server.stop_now()["content"][-1]["text"])["latched"]
            release.set()
            result = json.loads(motion.result(3)["content"][-1]["text"])
            assert not result.get("execution_ok", False)
            assert server._runtime is not None
            assert server._runtime.base_rig.primary.latched
        finally:
            release.set()
            server.shutdown()


def test_real_isaac_mcp_observation_uses_only_read_only_loopback_channel(tmp_path):
    # Real RPC/client/runtime routing, scripted state ONLY. No physics/Kit.
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer

    controller = MobileBridgeController(robot_id="microduck", source="isaac-microduck", engine="physx",
        device="cuda:0", asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64, support_contract=support_contract(),
        max_linear_speed=.15, max_angular_speed=.6, max_duration_s=3., lease_s=.5,
        max_state_age_s=2., max_action_wall_s=8.)
    controller.publish({"step": 1, "sim_time": .005, "position": [0., 0., .3],
                        "orientation_wxyz": [1., 0., 0., 0.], "linear_velocity": [0., 0., 0.],
                        "angular_velocity": [0., 0., 0.], "q": [0.] * 14, "dq": [0.] * 14,
                        "joint_names": [f"fixture-{i}" for i in range(14)], "contacts": [], "fallen": False})
    server = MobileBridgeServer(controller, port=0)
    operations = []
    dispatch = server.dispatch
    def record(request):
        operations.append(dict(request))
        return dispatch(request)
    server.dispatch = record
    server.start()
    config = write_mobile_config(tmp_path / "config", {"support_contract": support_contract()})
    c = MobileClient(tmp_path, config_dir=config, CASCADE_BASE="microduck_isaac",
                     CASCADE_MICRODUCK_ASSET_SHA256="a" * 64,
                     CASCADE_MICRODUCK_POLICY_SHA256="b" * 64,
                     CASCADE_MICRODUCK_MODEL_IDENTITY_SHA256="e" * 64,
                     CASCADE_MICRODUCK_DEVICE=controller.hello()["device"],
                     CASCADE_MICRODUCK_BRIDGE_PORT=str(server.address[1]))
    try:
        c.request("initialize")
        c.request("tools/list")
        c.call("list_bases")
        assert not operations  # metadata didn't even dial
        state = c.call("get_base_state")
        assert state["ok"] and state["state"]["step"] == 1
        c.close()
        assert all(r["op"] in {"hello", "state"} for r in operations)
        assert all(r.get("role", "reader") == "reader" for r in operations)
        assert controller.hello()["generation"] == 0
    finally:
        if c.proc.poll() is None:
            c.close()
        server.close()


def test_priority_stop_and_reset_have_native_trace_receipts(mobile_client, tmp_path):
    c = mobile_client
    c.call("get_base_state")
    stopped = c.call("emergency_stop")
    assert stopped["latched"]
    assert c.call("reset_stop")["ok"]
    records = [json.loads(line) for line in (tmp_path / "mcp/trace.jsonl").read_text().splitlines()]
    by_skill = {r["skill"]: r for r in records}
    assert by_skill["emergency_stop"]["result"]["outcome"] == "unverified"
    assert by_skill["reset_stop"]["result"]["ok"] is True


def test_mobile_hidden_tools_and_reset_arguments_are_enforced(tmp_path):
    c = MobileClient(tmp_path, CASCADE_HIDE_TOOLS="turn,reset_stop")
    try:
        tools = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
        assert "turn" not in tools and "reset_stop" not in tools
        assert not c.call("turn", {"angle_rad": .1})["ok"]
        assert not c.call("reset_stop")["ok"]
        assert c.call("emergency_stop")["latched"]
    finally:
        c.close()
    c = MobileClient(tmp_path / "invalid-reset")
    try:
        c.call("get_base_state")
        c.call("emergency_stop")
        assert not c.call("reset_stop", {"base": "other"})["ok"]
        assert c.call("get_base_state")["state"]["latched"]
    finally:
        c.close()


def write_mobile_config(path, profile):
    import yaml
    from cascade.config import load_profile
    path.mkdir()
    (path / "bases").mkdir()
    (path / "llm").mkdir()
    (path / "demo.yaml").write_text("memory: {}\n")
    (path / "llm/mock.yaml").write_text("type: mock\n")
    full = load_profile("bases", "microduck_isaac").as_dict()
    full.update(profile)
    (path / "bases/microduck_isaac.yaml").write_text(yaml.safe_dump(full))
    return path


def test_mobile_camera_mcp_subprocess_returns_actual_jpeg_and_history(tmp_path, frame_endpoint):
    import base64
    c, _, profile, packet, jpeg, operations = frame_endpoint
    client = MobileClient(tmp_path / "run", config_dir=write_mobile_config(tmp_path / "config", profile),
                          CASCADE_BASE="microduck_isaac")
    try:
        client.request("initialize")
        assert "camera_snapshot" in {s["name"] for s in client.request("tools/list")["result"]["tools"]}
        assert not operations
        for tool in ("get_observation", "camera_snapshot"):
            response = client.request("tools/call", {"name": tool, "arguments": {"camera": "side"}})
            images = [p for p in response["result"]["content"] if p["type"] == "image"]
            assert len(images) == 1, response
            assert images[0]["mimeType"] == "image/jpeg"
            assert base64.b64decode(images[0]["data"], validate=True) == jpeg
            assert payload(response)["frame"]["source"] == packet["source"]
            assert "image_jpeg_b64" not in payload(response)
        history = client.request("tools/call", {"name": "task_memory", "arguments": {"k": 2}})
        assert payload(history)["frames"] == 2
        assert len([p for p in history["result"]["content"] if p["type"] == "image"]) == 3
        packet["producer_age_s"] = 99.
        stale = client.request("tools/call", {"name": "camera_snapshot"})
        assert stale["result"]["isError"]
        assert not any(p["type"] == "image" for p in stale["result"]["content"])
        history = client.request("tools/call", {"name": "task_memory", "arguments": {"k": 2}})
        assert not payload(history)["current_view"]["available"]
        assert len([p for p in history["result"]["content"] if p["type"] == "image"]) == 2
    finally:
        client.close()
    assert c.hello()["generation"] == 0
    assert all(r["op"] in {"hello", "state", "frame"} for r in operations)


@pytest.mark.parametrize("velocity,expected", [(0., "confirmed"), (.05, "refuted")])
def test_real_mcp_stop_post_ack_evidence_never_repairs_failed_motion(tmp_path, frame_endpoint, velocity, expected):
    from test_mobile_runtime import SyntheticTicks, verifier_limits
    c, _, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c, velocity=velocity)
    client = MobileClient(tmp_path / "run", config_dir=write_mobile_config(tmp_path / "config", profile),
                          CASCADE_BASE="microduck_isaac")
    try:
        client.request("initialize")
        assert client.call("get_base_state")["ok"]
        assert client.call("reset_stop")["ok"]
        failed = client.call("walk_velocity", {"vx": 999., "vy": 0., "wz": 0., "duration_s": .04})
        assert not failed["ok"] and failed["error"]
        ack = client.call("emergency_stop")
        assert ack["ok"] and not ack["physical_stop_verified"] and ack["outcome"] == "unverified"
        deadline = time.monotonic() + 2
        while True:
            verification = client.call("verify_last_action")
            proof = verification["stop_verifications"][-1]
            if not proof["pending"] or time.monotonic() > deadline:
                break
            time.sleep(.01)
        assert proof["receipt_id"] == ack["receipt_id"] and proof["status"] == expected, proof
        assert proof["postcondition"]["evidence"]["provenance"]["source"] == "isaac-microduck"
        assert any(r["skill"] == "walk_velocity" and not r["execution_ok"] for r in verification["recent"])
        assert client.call("reset_stop")["ok"]
        assert client.call("verify_last_action")["stop_verifications"][-1]["superseded"]
        rows = [json.loads(s) for s in (tmp_path / "run/trace.jsonl").read_text().splitlines()]
        assert next(r for r in rows if r["skill"] == "emergency_stop")["result"]["outcome"] == "unverified"
        assert next(r for r in rows if r["skill"] == "stop_verification")["result"]["status"] == expected
    finally:
        client.close()
        ticks.close()
    assert client.proc.returncode == 0


@pytest.mark.parametrize("kind", ["cancel", "eof"])
def test_real_mobile_mcp_cancels_admitted_motion_with_both_checkers(tmp_path, frame_endpoint, kind):
    from test_mobile_runtime import SyntheticTicks, verifier_limits
    c, server, profile, _, _, operations = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    client = MobileClient(tmp_path / "run", config_dir=write_mobile_config(tmp_path / "config", profile),
                          CASCADE_BASE="microduck_isaac")
    try:
        assert client.call("get_base_state")["ok"]
        motion = client.send("tools/call", {"name": "walk_velocity", "arguments": {
            "vx": .05, "vy": 0., "wz": 0., "duration_s": 3.}})
        deadline = time.monotonic() + 2
        while not any(op["op"] == "command_velocity" for op in operations):
            assert time.monotonic() < deadline
            time.sleep(.005)
        started = time.monotonic()
        if kind == "cancel":
            client.send("notifications/cancelled", {"requestId": motion}, notification=True)
        else:
            client.proc.stdin.close()
        response = client.recv(1.5)
        assert time.monotonic() - started < 1.5
        result = payload(response)
        assert not result["ok"] and not result["execution_ok"]
        assert result["measured"]["before"] is not None  # genuinely admitted, not queue-only cancellation
        assert c.state()["state"]["latched"]
    finally:
        client.close()
        ticks.close()
    assert client.proc.returncode == 0


def test_real_mcp_ack_and_eof_do_not_wait_for_blocked_stop_reader(tmp_path, frame_endpoint, monkeypatch):
    from test_mobile_runtime import SyntheticTicks, verifier_limits
    c, server, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    entered, release = threading.Event(), threading.Event()
    client = MobileClient(tmp_path / "run", config_dir=write_mobile_config(tmp_path / "config", profile),
                          CASCADE_BASE="microduck_isaac")
    dispatch = server.dispatch
    def blocked(req):
        if req["op"] == "state":
            entered.set()
            release.wait(2)
        return dispatch(req)
    try:
        assert client.call("get_base_state")["ok"]
        assert client.call("reset_stop")["ok"]
        monkeypatch.setattr(server, "dispatch", blocked)
        started = time.monotonic()
        first = client.call("emergency_stop")
        assert time.monotonic() - started < .15
        assert entered.wait(1)
        for _ in range(10):
            started = time.monotonic()
            ack = client.call("stop_navigation")
            assert time.monotonic() - started < .15
            assert ack["receipt_id"] != first["receipt_id"]
            assert not ack["physical_stop_verified"]
        started = time.monotonic()
        client.close()  # EOF/close joins the timed-out observer; it does NOT wait for our callback
        assert time.monotonic() - started < 1.5
        assert client.proc.returncode == 0
    finally:
        release.set()
        if client.proc.poll() is None:
            client.close()
        ticks.close()
