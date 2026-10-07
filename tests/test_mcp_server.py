"""MCP stdio server protocol tests: a real subprocess, real JSON-RPC frames.

Uses the mock camera/arm profiles (env), so this covers everything Hermes /
Claude Code exercise except physical hardware.
"""

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from conftest import REPO, has_pinocchio, loopback_host

pytestmark = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")


class McpClient:
    def __init__(self, tmp_run_dir: str, extra_env: dict | None = None):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src")
        env["CASCADE_CAMERA"] = "mock"
        env["CASCADE_ARM"] = "mock"
        env["CASCADE_RUN_DIR"] = tmp_run_dir
        env["CASCADE_STREAM"] = "0"  # no HTTP port binding inside tests
        env.update(extra_env or {})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cascade.apps.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        self._id = 0
        # server -> client notifications (no id), e.g. tools/list_changed
        # after the capability matrix is probed; recv() skips them so a
        # response wait never trips over one, and tests inspect them here.
        self.notifications: list[dict] = []
        # stdout is pumped through a queue so recv() can time out instead of
        # hanging the whole suite when a response frame is (wrongly) dropped
        self._out_q: queue.Queue = queue.Queue()

        def _pump():
            for line in self.proc.stdout:
                self._out_q.put(line)
            self._out_q.put(None)  # EOF sentinel

        threading.Thread(target=_pump, daemon=True).start()

    def request(self, method: str, params: dict | None = None, timeout=60):
        self.send(method, params)
        resp = self.recv(timeout=timeout)
        assert resp["id"] == self._id
        return resp

    def send(self, method: str, params: dict | None = None) -> int:
        """Write a request frame WITHOUT waiting for the response -- for
        interleaved traffic (out-of-band emergency_stop)."""
        self._id += 1
        frame = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()
        return self._id

    def recv(self, timeout=60) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._out_q.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                pytest.fail(f"no response within {timeout}s from a live server "
                            f"(pid {self.proc.pid}); dropped/suppressed frame?")
            assert line is not None, f"server died: {self.proc.stderr.read()[-2000:]}"
            frame = json.loads(line)
            if "id" not in frame:  # a notification, never an answer
                self.notifications.append(frame)
                continue
            return frame

    def notify(self, method: str, params: dict | None = None):
        frame: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


@pytest.fixture
def client(tmp_path):
    c = McpClient(str(tmp_path / "run"))
    yield c
    c.close()


def _tool_payload(resp):
    result = resp["result"]
    text = next(b["text"] for b in result["content"] if b["type"] == "text")
    return json.loads(text), result.get("isError", False)


def test_initialize_and_tools_list_without_hardware(client):
    resp = client.request("initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    })
    result = resp["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "cascade"
    assert "tools" in result["capabilities"]
    client.notify("notifications/initialized")

    resp = client.request("tools/list")
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    for expected in (
        "get_observation", "grasp_object", "place_at", "push_object",
        "recall_memory", "camera_snapshot", "emergency_stop", "reset_stop",
    ):
        assert expected in tools, f"missing tool {expected}"
    assert "task_done" not in tools  # loop-internal, not exposed
    for t in tools.values():
        assert t["inputSchema"]["type"] == "object"
        assert t["description"]


def test_tools_call_observation_and_memory(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.notify("notifications/initialized")

    response = client.request("tools/call", {"name": "get_observation", "arguments": {}})
    payload, is_err = _tool_payload(response)
    assert not is_err
    assert payload["ok"]
    assert "objects_visible" not in payload
    assert 0 <= payload["observation_frame"]["frame_age_s"] <= 5
    image = next(block for block in response["result"]["content"] if block["type"] == "image")
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(image["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert pixels.shape == (480, 640, 3)
    red_box = pixels[220:240, 300:340].mean(axis=(0, 1))
    assert red_box[2] > red_box[0] + 100 and red_box[2] > red_box[1] + 100

    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "recall_memory",
                                      "arguments": {"query": "red cube"}})
    )
    assert not is_err
    assert "object_memory" in payload
    assert payload["object_memory"]["label"] == "red cube"


def test_annotated_view_returns_its_image_and_numbered_key(client):
    response = client.request("tools/call", {"name": "annotated_view", "arguments": {}})
    data, is_error = _tool_payload(response)

    assert not is_error and data["ok"]
    images = [block for block in response["result"]["content"] if block["type"] == "image"]
    assert len(images) == 1 and images[0]["mimeType"] == "image/jpeg"
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(images[0]["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert pixels.shape == (480, 640, 3)
    assert "Annotated view:" in data["key"]
    assert "image" not in data  # a server-local path is not an image for an MCP client
    assert 0 <= data["observation_frame"]["frame_age_s"] <= 5
    assert "tracked estimates" in data["note"]


def test_camera_snapshot_returns_image_block(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    resp = client.request("tools/call", {"name": "camera_snapshot", "arguments": {}})
    blocks = resp["result"]["content"]
    img = next(b for b in blocks if b["type"] == "image")
    assert img["mimeType"] == "image/jpeg"
    import base64

    raw = base64.b64decode(img["data"])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic


def test_estop_blocks_motion_until_reset(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.request("tools/call", {"name": "get_observation", "arguments": {}})

    payload, _ = _tool_payload(
        client.request("tools/call", {"name": "emergency_stop", "arguments": {}})
    )
    assert payload["stopped"]
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert is_err and "e-stop" in payload["error"]
    payload, _ = _tool_payload(
        client.request("tools/call", {"name": "reset_stop", "arguments": {}})
    )
    assert payload["stopped"] is False
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert not is_err, payload


def test_unknown_tool_and_method(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "warp_drive", "arguments": {}})
    )
    assert is_err and "unknown skill" in payload["error"]
    resp = client.request("frobnicate/all")
    assert resp["error"]["code"] == -32601


def test_ping_and_stdout_purity(client):
    """Every stdout line must be a JSON-RPC frame (Hermes chokes otherwise)."""
    r1 = client.request("initialize", {"protocolVersion": "2024-11-05"})
    assert r1["result"]["protocolVersion"] == "2024-11-05"
    r2 = client.request("ping")
    assert r2["result"] == {}
    r3 = client.request("tools/call", {"name": "get_observation", "arguments": {}})
    assert r3["result"]
    # all three lines parsed as JSON already; nothing else was interleaved


def test_hermes_config_upsert():
    sys.path.insert(0, str(REPO / "scripts"))
    from setup_hermes import upsert, yaml_block

    block = yaml_block("l515", "rebot_rs", "/usr/bin/python3")
    # fresh file
    merged = upsert(None, block)
    assert merged.startswith("mcp_servers:")
    assert 'CASCADE_CAMERAS: "l515"' in merged
    # existing config with another server is preserved
    existing = (
        "model: hermes-4\n"
        "mcp_servers:\n"
        "  agenticros:\n"
        '    command: "node"\n'
        '    args: ["/x/index.js"]\n'
    )
    merged = upsert(existing, block)
    assert "model: hermes-4" in merged
    assert "agenticros:" in merged
    assert "cascade:" in merged
    # idempotent replace: camera changes, no duplicate entries
    block2 = yaml_block("mock", "mock", "/usr/bin/python3")
    merged2 = upsert(merged, block2)
    assert merged2.count("cascade:") == 1
    assert 'CASCADE_CAMERAS: "mock"' in merged2 and 'CASCADE_CAMERAS: "l515"' not in merged2


def test_new_livestream_tools_over_jsonrpc(client):
    """world_state / live_view_url / pick_and_place / camera_snapshot(camera=)
    were untested at the protocol level (2026-07-18 review finding)."""
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.notify("notifications/initialized")

    tools = {t["name"] for t in client.request("tools/list")["result"]["tools"]}
    for expected in ("pick_and_place", "world_state", "live_view_url",
                     "describe_scene", "handover", "wave", "point_at"):
        assert expected in tools, f"missing tool {expected}"

    # world_state: instant text, includes objects + cameras, arm stays lazy
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "world_state", "arguments": {}})
    )
    assert not is_err and payload["ok"]
    assert "objects" in payload and "cameras" in payload
    assert payload["session_state"]["arm_connected"] is False  # LazyArm untouched by a look
    assert payload["session_state"]["source"] == "mcp_runtime"
    assert payload["simulator_status"] == "not_queried"
    assert "holding" not in payload
    assert payload["live_arm_feedback"] is False
    assert payload["gripper_contents"] == "not_measured"
    assert payload["home_pose"] == "not_verified"

    # live_view_url with CASCADE_STREAM=0: honest error, not a bogus URL.
    # The dashboard is lazy now (chat is the UI), so this tool OPENS the view
    # on demand -- but CASCADE_STREAM=0 is a hard kill switch that must still
    # refuse rather than bind a port behind the operator's back.
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "live_view_url", "arguments": {}})
    )
    assert is_err and payload["open"] is False
    assert "disabled" in payload["error"] and "CASCADE_STREAM" in payload["error"]

    # named-camera snapshot
    resp = client.request("tools/call", {"name": "camera_snapshot",
                                         "arguments": {"camera": "mock"}})
    blocks = resp["result"]["content"]
    assert any(b["type"] == "image" for b in blocks)

    # pick_and_place end-to-end over JSON-RPC (mock jaws close on air ->
    # honest structured failure, never a protocol error)
    payload, is_err = _tool_payload(
        client.request("tools/call",
                       {"name": "pick_and_place", "arguments": {"object": "red object"}},
                       timeout=120)
    )
    assert is_err and payload["stage"] == "grasp"
    assert "grasp failed" in payload["error"]


def test_emergency_stop_preempts_running_motion(client):
    """emergency_stop must NOT queue behind a long motion call: the stdin
    reader latches the e-stop the moment the frame arrives, so its response
    lands FIRST and the running motion aborts mid-stream (booth prep
    2026-07-20; was roadmap near-term item 1)."""
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.notify("notifications/initialized")
    # force the runtime up so the stop path has a live harness to latch
    client.request("tools/call", {"name": "get_observation", "arguments": {}})

    pick_id = client.send("tools/call", {"name": "pick_and_place",
                                         "arguments": {"object": "red object"}})
    time.sleep(0.5)  # let the worker enter the motion call
    stop_id = client.send("tools/call", {"name": "emergency_stop", "arguments": {}})

    first = client.recv()
    assert first["id"] == stop_id, "emergency_stop answered after the motion finished"
    payload, _ = _tool_payload(first)
    assert payload["stopped"]

    second = client.recv()
    assert second["id"] == pick_id
    payload, is_err = _tool_payload(second)
    assert is_err
    # the latch must break the persistence loop after the aborted attempt,
    # not burn the remaining retry budget -- "(e-stop latched; not
    # retrying)" is the loop's fail-fast signature (skills/runtime.py)
    assert "not retrying" in payload["error"], payload["error"]

    # the latch holds until reset_stop
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert is_err and "e-stop" in payload["error"]


def test_client_cancellation_of_motion_freezes_arm(client):
    """notifications/cancelled for an in-flight MOTION call (Esc in the MCP
    host mid-pick) must freeze the arm, not orphan the motion server-side."""
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.request("tools/call", {"name": "get_observation", "arguments": {}})

    pick_id = client.send("tools/call", {"name": "pick_and_place",
                                         "arguments": {"object": "red object"}})
    time.sleep(0.5)
    client.notify("notifications/cancelled", {"requestId": pick_id, "reason": "user"})

    resp = client.recv()  # the aborted call still answers; the host discards it
    assert resp["id"] == pick_id
    _, is_err = _tool_payload(resp)
    assert is_err

    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert is_err and "e-stop" in payload["error"]


def test_mcp_mode_dashboard_chat_is_reflex_only(tmp_path, monkeypatch):
    """Booth roadmap item 6: in MCP mode the dashboard chat runs the tier-1
    reflex grammar with zero LLM (the host-outage fallback), and refuses
    free-form text with an honest narration note."""
    import urllib.request

    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_ARM", "mock")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "1")
    monkeypatch.setenv("CASCADE_STREAM_PORT", "0")  # ephemeral port
    monkeypatch.setenv("CASCADE_VIEW", "0")
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    try:
        runtime = server._ensure_runtime()
        assert runtime.stream_server is not None
        base = f"http://{loopback_host()}:{runtime.stream_server.port}"

        def post_task(text: str) -> dict:
            req = urllib.request.Request(
                f"{base}/task", data=json.dumps({"task": text}).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=5).read())

        assert post_task("wave")["accepted"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and runtime.last_path != "reflex":
            time.sleep(0.1)
        assert runtime.last_path == "reflex"

        # free-form text: accepted by HTTP, refused by the handler with a
        # narration note (retry while the wave task drains the busy lock)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if post_task("compose a haiku about the arm")["accepted"]:
                break
            time.sleep(0.2)
        deadline = time.monotonic() + 10
        note_seen = False
        while time.monotonic() < deadline and not note_seen:
            note_seen = any("not a routine command" in ev.text
                            for ev in runtime.memory.events())
            time.sleep(0.1)
        assert note_seen, "free-form chat text must produce the honest note"
    finally:
        server.shutdown()


def test_hidden_tools_are_delisted_and_rejected(tmp_path):
    """CASCADE_HIDE_TOOLS removes tools from the surface AND the call path
    (booth sessions hide reset_stop so a model cannot clear a staff e-stop);
    emergency_stop is never hideable -- even when an operator typo lists it."""
    c = McpClient(str(tmp_path / "run"),
                  extra_env={"CASCADE_HIDE_TOOLS": "reset_stop,throw,emergency_stop"})
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        tools = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
        assert "reset_stop" not in tools and "throw" not in tools
        assert "emergency_stop" in tools  # survives the hide list

        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "reset_stop", "arguments": {}})
        )
        assert is_err and "disabled by the operator" in payload["error"]

        # and it is not just listed -- it answers, despite the hide list
        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "emergency_stop", "arguments": {}})
        )
        assert not is_err and payload["stopped"]
    finally:
        c.close()


# ── capability matrix -> tool surface (ROADMAP #12, Waddle) ──────────────────
#
# The catalog is trimmed by what the BUILT rig can do, evaluated from probed
# state (camera depth source, sidecar probes, ArmRig, verifier, memory) --
# never from a profile's promise -- and every withheld tool carries its
# reason in world_state, the server log and the dashboard /state.
# CASCADE_HIDE_TOOLS stays the explicit operator override; _EXCLUDED_TOOLS
# keeps excluding the loop-internal task_done.

# closed ports from this item's block (ledger: 42100-42199): a probe against
# them times out, so the sidecar test is "server down" on every host, even
# one where a real GraspGen-X / occupancy bridge happens to be listening on
# the default 5556/5557
_DEAD_GRASPGENX_PORT = 42101
_DEAD_OCCUPANCY_PORT = 42102


def _server_env(monkeypatch, tmp_path, camera="mock", arms="mock"):
    monkeypatch.setenv("CASCADE_CAMERA", camera)
    monkeypatch.delenv("CASCADE_CAMERAS", raising=False)
    monkeypatch.setenv("CASCADE_ARMS", arms)
    monkeypatch.delenv("CASCADE_ARM", raising=False)
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "0")
    monkeypatch.setenv("CASCADE_VIEW", "0")
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(_DEAD_GRASPGENX_PORT))


def _dead_sidecars(monkeypatch, mutate=None):
    """Route the in-process server's config through a wrapper that points the
    sidecar probes at closed ports (and applies `mutate(cfg)`); the server
    imports load_demo_config from cascade.config at call time."""
    import cascade.config as config_mod

    real = config_mod.load_demo_config

    def wrapped(**kwargs):
        cfg = real(**kwargs)
        cfg._data["grasp"]["graspgenx"]["port"] = _DEAD_GRASPGENX_PORT
        cfg._data["occupancy"]["port"] = _DEAD_OCCUPANCY_PORT
        if mutate is not None:
            mutate(cfg)
        return cfg

    monkeypatch.setattr(config_mod, "load_demo_config", wrapped)


def test_tool_requirements_name_real_tools_and_never_the_stop_path():
    """TOOL_REQUIREMENTS is hand-maintained like _MOTION_SKILLS, so a typo
    there would silently stop gating the tool it meant: every key must be a
    served tool. The stop path can never depend on a probe, and task_done is
    excluded by _EXCLUDED_TOOLS regardless of any capability."""
    from cascade.apps.capabilities import (
        CAP_DEPTH_3D, CAP_DEPTH_HEIGHTS, CAP_MULTI_ARM, CAP_VERIFIER,
        TOOL_REQUIREMENTS, capability_matrix, withheld_tools,
    )
    from cascade.apps.mcp_server import _EXCLUDED_TOOLS, _EXTRA_TOOLS
    from cascade.skills.runtime import TOOL_SPECS

    served = {t["name"] for t in TOOL_SPECS + _EXTRA_TOOLS} - _EXCLUDED_TOOLS
    assert set(TOOL_REQUIREMENTS) <= served, sorted(set(TOOL_REQUIREMENTS) - served)
    assert not {"emergency_stop", "reset_stop", "task_done"} & set(TOOL_REQUIREMENTS)
    # the matrix's own vocabulary is closed: every requirement is a known capability
    probed = capability_matrix(None)
    assert probed["probed"] is False
    for reqs in TOOL_REQUIREMENTS.values():
        assert reqs, "a tool with an empty requirement tuple gates nothing"
    # the preconditions this change is about are really in the table
    assert CAP_DEPTH_3D in TOOL_REQUIREMENTS["localize_object"]
    assert CAP_DEPTH_3D in TOOL_REQUIREMENTS["grasp_at_pixel"]
    assert set(TOOL_REQUIREMENTS["place_on_object"]) >= {CAP_DEPTH_3D, CAP_DEPTH_HEIGHTS}
    assert TOOL_REQUIREMENTS["list_arms"] == (CAP_MULTI_ARM,)
    assert TOOL_REQUIREMENTS["verify_last_action"] == (CAP_VERIFIER,)
    # UNPROBED is not UNAVAILABLE: before a runtime exists nothing is withheld
    assert withheld_tools(probed) == {}


def test_unknown_capabilities_never_withhold_a_tool():
    """A partial runtime (no camera rig, no backends) is UNKNOWN, not
    RGB-only: a capability is withheld only when probed state positively
    shows the precondition unmet. Tri-state, never a guess."""
    from types import SimpleNamespace

    from cascade.apps.capabilities import (
        CAP_DEPTH_3D, CAP_LEARNED_GRASPS, CAP_OCCUPANCY, CAP_VERIFIER,
        capability_matrix, withheld_tools,
    )

    bare = SimpleNamespace(backends=lambda: {}, arm_rig=None, memory=None)
    matrix = capability_matrix(bare)
    assert matrix["probed"] is True
    assert matrix[CAP_DEPTH_3D]["available"] is None       # no rig -> unknown
    assert matrix[CAP_LEARNED_GRASPS]["available"] is None  # backends() said nothing
    assert matrix[CAP_OCCUPANCY]["available"] is None
    assert matrix[CAP_VERIFIER]["available"] is False       # effects missing = off
    withheld = withheld_tools(matrix)
    assert "localize_object" not in withheld and "grasp_object" not in withheld
    assert "verify_last_action" in withheld  # the one thing positively known
    for reason in withheld.values():
        assert reason  # a hidden tool is never silent


def test_depth_source_for_mirrors_ensure_depth_without_touching_the_frame():
    """The matrix asks the DepthProvider what ensure_depth() WOULD produce
    for a frame (sensor / mono / plane / none) without mutating it, and it
    can answer from a camera's own declaration before the first grab."""
    import numpy as np

    from cascade.config import Cfg
    from cascade.perception.depth_provider import DepthProvider
    from cascade.types import Frame

    K = np.eye(3)
    rgb = np.zeros((4, 4, 3), np.uint8)
    sensor = Frame(rgb=rgb, depth_m=np.ones((4, 4), np.float32), K=K, depth_source="sensor")
    rgb_only = Frame(rgb=rgb, depth_m=None, K=K)

    none = DepthProvider(Cfg({}))
    plane = DepthProvider(Cfg({"table_plane_cam": [0.0, 0.0, -1.0, 0.6]}))
    assert none.depth_source_for(sensor) == "sensor"
    assert none.depth_source_for(rgb_only) == "none"
    assert plane.depth_source_for(rgb_only) == "plane"
    assert plane.depth_source_for(sensor) == "sensor"
    assert rgb_only.depth_m is None and rgb_only.depth_source == "none"  # untouched
    # no frame yet: the driver's declaration stands in, flagged as such by the caller
    assert none.depth_source_for(None, sensor=True) == "sensor"
    assert none.depth_source_for(None, sensor=False) == "none"
    assert plane.depth_source_for(None, sensor=False) == "plane"
    assert none.depth_source_for(None) is None  # nothing to go on


def test_rgb_only_rig_withholds_depth_tools_and_says_why(tmp_path):
    """Waddle's rule, probed: on an RGB-only camera with no table plane,
    nothing can lift a pixel to metres, so the 3D tools are withheld AND
    rejected if called -- with the reason in world_state and the server
    log -- while the RGB and pure-motion tools stay offered."""
    c = McpClient(str(tmp_path / "run"), extra_env={"CASCADE_CAMERA": "mock_rgb"})
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        state, is_err = _tool_payload(
            c.request("tools/call", {"name": "world_state", "arguments": {}})
        )
        assert not is_err and state["ok"]
        caps = state["capabilities"]
        assert caps["probed"] is True
        assert caps["depth_3d"]["available"] is False
        assert caps["cameras"]["mock_rgb"]["depth"] == "none"
        withheld = state["tools_withheld"]
        for name in ("localize_object", "grasp_object", "pick_and_place", "probe_point",
                     "locate_pixel", "grasp_at_pixel", "place_on_object", "point_at",
                     "push_object", "preview_grasp"):
            assert name in withheld, f"{name} offered on an RGB-only rig"
            assert "depth" in withheld[name].lower(), withheld[name]

        tools = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
        assert not set(withheld) & tools, sorted(set(withheld) & tools)
        for name in ("get_observation", "describe_scene", "camera_snapshot", "annotated_view",
                     "move_home", "wave", "open_gripper", "emergency_stop", "reset_stop",
                     "world_state", "recall_memory"):
            assert name in tools, f"{name} missing: the matrix over-trimmed"

        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "localize_object",
                                     "arguments": {"label": "red object"}})
        )
        assert is_err and "not available on this rig" in payload["error"]
        assert "depth" in payload["error"].lower()
        assert payload["capabilities"]["depth_3d"]["available"] is False
        # a snapshot still works: RGB is what this rig has
        resp = c.request("tools/call", {"name": "camera_snapshot", "arguments": {}})
        assert any(b["type"] == "image" for b in resp["result"]["content"])
    finally:
        c.close()
    log = (tmp_path / "run" / "server.log").read_text()
    assert "capabilities:" in log and "depth_3d=no" in log
    assert "withheld" in log and "localize_object" in log


def test_plane_depth_keeps_grasp_tools_but_withholds_stacking(tmp_path, monkeypatch):
    """Table-plane depth is exactly what tabletop grasping needs, so the
    grasp tools stay; it puts every pixel ON the table, so a tool that needs
    an object's real height (place_on_object) is withheld, and the matrix
    distinguishes the two depth capabilities."""
    _server_env(monkeypatch, tmp_path, camera="mock_rgb")

    def plane(cfg):
        for cam in cfg._data["cameras"]:
            cam["table_plane_cam"] = [0.0, 0.0, -1.0, 0.6]
        cfg._data["camera"]["table_plane_cam"] = [0.0, 0.0, -1.0, 0.6]

    _dead_sidecars(monkeypatch, mutate=plane)
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    try:
        runtime = server._ensure_runtime()
        matrix = server.capabilities()
        assert matrix["cameras"]["mock_rgb"]["depth"] == "plane"
        assert matrix["depth_3d"]["available"] is True
        assert matrix["depth_heights"]["available"] is False
        tools = {t["name"] for t in server.list_tools()}
        assert {"grasp_object", "pick_and_place", "localize_object", "probe_point"} <= tools
        assert "place_on_object" not in tools
        withheld = server.withheld_tools()
        assert "height" in withheld["place_on_object"].lower()
        assert runtime.observe_fresh().depth_source == "plane"
    finally:
        server.shutdown()


def test_dead_sidecars_report_the_fallback_without_hiding_grasp_tools(tmp_path, monkeypatch, capsys):
    """GraspGen-X and the occupancy bridge not answering their probe is a
    FALLBACK (analytic OBB; clearance gate off), reported in the matrix,
    never a reason to hide the grasp tools. Only a tool that strictly needs
    the sidecar is withheld -- no shipped tool does today, so the rule is
    pinned with a synthetic requirement."""
    _server_env(monkeypatch, tmp_path)
    _dead_sidecars(monkeypatch)
    from cascade.apps import capabilities
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    try:
        runtime = server._ensure_runtime()
        assert runtime.backends()["grasp_planner"] == "obb (graspgenx down)"
        matrix = server.capabilities()
        assert matrix[capabilities.CAP_LEARNED_GRASPS]["available"] is False
        assert "graspgenx down" in matrix[capabilities.CAP_LEARNED_GRASPS]["detail"]
        assert matrix[capabilities.CAP_OCCUPANCY]["available"] is False
        tools = {t["name"] for t in server.list_tools()}
        assert {"grasp_object", "pick_and_place", "preview_grasp", "grasp_at_pixel"} <= tools
        assert not {"grasp_object", "pick_and_place"} & set(server.withheld_tools())

        # the banner next to `backends:` and the MCP log name the matrix
        err = capsys.readouterr().err
        assert "[cascade] capabilities:" in err
        assert "learned_grasps=no" in err and "occupancy=no" in err

        # a tool that STRICTLY needs the sidecar is withheld with the reason
        monkeypatch.setattr(capabilities, "TOOL_REQUIREMENTS", {
            **capabilities.TOOL_REQUIREMENTS,
            "preview_grasp": (capabilities.CAP_LEARNED_GRASPS,),
            "probe_point": (capabilities.CAP_OCCUPANCY,),
        })
        tools = {t["name"] for t in server.list_tools()}
        assert "preview_grasp" not in tools and "probe_point" not in tools
        assert "grasp_object" in tools  # still falls back, still offered
        withheld = server.withheld_tools()
        assert "GraspGen-X" in withheld["preview_grasp"]
        assert "occupancy" in withheld["probe_point"].lower()
        payload = json.loads(server.call_tool("preview_grasp", {"label": "red object"})["content"][0]["text"])
        assert payload["ok"] is False and "GraspGen-X" in payload["error"]
    finally:
        server.shutdown()


def test_single_arm_rig_drops_list_arms_and_the_arm_parameter(tmp_path):
    """Multi-arm tools appear only with >= 2 arms: one arm means nothing to
    enumerate, and the `arm` parameter the rig injects into every motion
    schema (which the chat model dutifully fills with "" -- ROADMAP
    2026-09-14) is dropped from the served schemas."""
    c = McpClient(str(tmp_path / "run"))
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        state, _ = _tool_payload(
            c.request("tools/call", {"name": "world_state", "arguments": {}})
        )
        assert state["capabilities"]["multi_arm"]["available"] is False
        assert state["capabilities"]["arms"] == ["mock"]
        assert "single-arm" in state["tools_withheld"]["list_arms"]

        tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
        assert "list_arms" not in tools
        assert "arm" not in tools["move_home"]["inputSchema"]["properties"]
        assert "arm" not in tools["pick_and_place"]["inputSchema"]["properties"]
        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "list_arms", "arguments": {}})
        )
        assert is_err and "not available on this rig" in payload["error"]
        # the rig's own arm name still routes (the model may have memorized it)
        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "move_home", "arguments": {"arm": "mock"}})
        )
        assert not is_err, payload
    finally:
        c.close()


def test_two_arm_rig_offers_list_arms_and_the_arm_parameter(tmp_path):
    """CASCADE_ARMS=a,b builds the ArmRig in MCP mode (mirror of
    CASCADE_CAMERAS); with two arms the multi-arm surface is offered and
    list_arms names both."""
    c = McpClient(str(tmp_path / "run"),
                  extra_env={"CASCADE_ARMS": "so101_left,so101_right"})
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        state, _ = _tool_payload(
            c.request("tools/call", {"name": "world_state", "arguments": {}})
        )
        assert state["capabilities"]["multi_arm"]["available"] is True
        assert state["capabilities"]["arms"] == ["so101_left", "so101_right"]
        assert "list_arms" not in state["tools_withheld"]

        tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
        assert "list_arms" in tools
        assert "arm" in tools["move_home"]["inputSchema"]["properties"]
        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "list_arms", "arguments": {}})
        )
        assert not is_err and payload["arms"] == ["so101_left", "so101_right"]
    finally:
        c.close()


def test_operator_hide_list_task_done_and_the_matrix_compose(tmp_path):
    """Three independent filters, each reported where it applies:
    _EXCLUDED_TOOLS (task_done, loop-internal), CASCADE_HIDE_TOOLS (the
    explicit operator override, still honoured) and the capability matrix.
    On the depth-capable single-arm mock rig the matrix withholds exactly
    list_arms."""
    c = McpClient(str(tmp_path / "run"), extra_env={"CASCADE_HIDE_TOOLS": "throw"})
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        state, _ = _tool_payload(
            c.request("tools/call", {"name": "world_state", "arguments": {}})
        )
        assert state["tools_hidden_by_operator"] == ["throw"]
        assert set(state["tools_withheld"]) == {"list_arms"}
        assert state["capabilities"]["depth_3d"]["available"] is True
        assert state["capabilities"]["verifier"]["available"] is True
        assert state["capabilities"]["memory"]["available"] is True

        tools = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
        assert not {"throw", "list_arms", "task_done"} & tools
        assert {"grasp_object", "pick_and_place", "verify_last_action", "task_memory"} <= tools
        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "throw", "arguments": {}})
        )
        assert is_err and "disabled by the operator" in payload["error"]
    finally:
        c.close()


def test_catalog_served_before_the_build_is_refreshed_by_list_changed(tmp_path):
    """Before the runtime exists nothing is probed, so nothing is withheld
    (the full catalog is served, never a guess). When the build then trims
    the surface the server sends notifications/tools/list_changed -- it
    declared listChanged -- so a compliant host re-fetches; a host that does
    not is still protected by the call-time rejection."""
    c = McpClient(str(tmp_path / "run"), extra_env={"CASCADE_PREWARM": "0"})
    try:
        init = c.request("initialize", {"protocolVersion": "2025-06-18"})
        assert init["result"]["capabilities"]["tools"].get("listChanged") is True
        before = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
        assert "list_arms" in before  # unprobed: offered
        assert "arm" in before["move_home"]["inputSchema"]["properties"]
        assert c.notifications == []

        c.request("tools/call", {"name": "world_state", "arguments": {}})  # builds the runtime
        after = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
        assert "list_arms" not in after
        assert "arm" not in after["move_home"]["inputSchema"]["properties"]
        assert any(n["method"] == "notifications/tools/list_changed" for n in c.notifications), \
            c.notifications
    finally:
        c.close()
