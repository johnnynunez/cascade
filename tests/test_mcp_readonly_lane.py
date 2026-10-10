"""B46: the opt-in read-only lane of the MCP server.

Known limitation before B46 (docs/ARCHITECTURE.md): the server executes one
tool call at a time. Stops, cancels and pings are answered out-of-band by the
receive side, but while a pick runs (seconds) a chat host cannot even ask
`world_state`. With `mcp.readonly_lane: true` (or CASCADE_MCP_READONLY_LANE=1)
four provably read-only tools are answered on a separate lane thread while a
motion tool runs; motions stay strictly serialized, every other tool still
waits, the stop path is untouched and a lane result says it was served during
a motion.

Every behaviour test drives the REAL server process (`main()` -> stdio or
`--http` -> receive side -> worker / lane -> McpSkillServer) on the mock stack.
A motion is held mid-stream by a deterministic barrier inside that process
(`hold-<tag>` / `at-<tag>` / `release-<tag>` files, see `_HOOKS`), never by a
sleep, and the barrier records which THREAD reached it -- the serial worker
runs on the main thread, the lane on its own -- so "who served this call" is an
observation, not a timing guess.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import REPO, has_pinocchio, loopback_host

pytestmark = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")

LANE_TOOLS = ("world_state", "robot_knowledge", "verify_last_action", "camera_snapshot")
LANE_THREAD = "cascade-mcp-readonly-lane"
WORKER_THREAD = "MainThread"  # _worker_loop runs on the server's main thread
TOKEN = "b46-readonly-lane-token-" + "k" * 24
# The mock stack dials no sidecar for these tools; the endpoints still point
# into this item's own block so a stray dial can never reach a shared service.
_PORTS = {"CASCADE_GRASPGENX_PORT": "46001", "CASCADE_OCCUPANCY_PORT": "46002",
          "CASCADE_BRIDGE_PORT": "46003", "CASCADE_HUG_PORT": "46004"}

#: Loaded INSIDE the server process before `main()`: patches two seams with
#: file barriers. `MockArm.send_joint_target` is every streamed waypoint of a
#: motion (the arm's control loop); `apps.demo._runtime_state` is the body of
#: `world_state`. Unarmed, both are pass-throughs.
_HOOKS = '''
import os
import threading
import time
from pathlib import Path

DIR = Path(os.environ["B46_BARRIER_DIR"])
_threads_lock = threading.Lock()
_motion_threads = set()


def _publish(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def gate(tag):
    """Hold the caller once per armed `hold-<tag>`: claiming it is an unlink
    (exactly one caller wins), then `at-<tag>` names the caller's thread and
    the caller waits for `release-<tag>`. Bounded so a broken test cannot
    hang the suite."""
    try:
        (DIR / ("hold-" + tag)).unlink()
    except FileNotFoundError:
        return
    _publish(DIR / ("at-" + tag), threading.current_thread().name)
    release = DIR / ("release-" + tag)
    deadline = time.monotonic() + 120.0
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.002)
    try:
        release.unlink()
    except FileNotFoundError:
        pass


from cascade.control import mock_arm as _mock_arm

_send = _mock_arm.MockArm.send_joint_target


def _send_joint_target(self, q):
    name = threading.current_thread().name
    with _threads_lock:
        if name not in _motion_threads:
            _motion_threads.add(name)
            with open(DIR / "motion-threads", "a") as fh:
                fh.write(name + "\\n")
    gate("motion")
    return _send(self, q)


_mock_arm.MockArm.send_joint_target = _send_joint_target

from cascade.apps import demo as _demo

_state = _demo._runtime_state


def _runtime_state(runtime):
    gate("world_state")
    return _state(runtime)


_demo._runtime_state = _runtime_state
'''

#: the server entry point, with the hooks loaded first (sys.argv[1] = hook dir)
_BOOT = ("import sys; sys.path.insert(0, sys.argv.pop(1)); import b46_hooks; "
         "from cascade.apps.mcp_server import main; sys.exit(main())")


def _payload(resp):
    result = resp["result"]
    texts = [b["text"] for b in result["content"] if b["type"] == "text"]
    return json.loads(texts[-1]), result.get("isError", False)


class _Server:
    """One real MCP server process with barrier hooks, stdio or --http."""

    def __init__(self, tmp_path: Path, *, lane: str | None, http: bool = False,
                 extra_env: dict | None = None):
        self.dir = tmp_path / "barriers"
        self.dir.mkdir(parents=True)
        hooks = tmp_path / "hooks"
        hooks.mkdir()
        (hooks / "b46_hooks.py").write_text(_HOOKS)
        env = dict(os.environ)
        for name in ("CASCADE_MCP_READONLY_LANE", "CASCADE_HIDE_TOOLS", "CASCADE_PROGRAMS",
                     "CASCADE_ROBOT", "CASCADE_BASE", "CASCADE_CAMERAS", "CASCADE_ARMS"):
            env.pop(name, None)
        env.update(PYTHONPATH=str(REPO / "src"), CASCADE_CAMERA="mock", CASCADE_ARM="mock",
                   CASCADE_RUN_DIR=str(tmp_path / "run"), CASCADE_STREAM="0", CASCADE_VIEW="0",
                   CASCADE_PREWARM="0", CASCADE_OCCUPANCY="0", B46_BARRIER_DIR=str(self.dir),
                   **_PORTS)
        if lane is not None:
            env["CASCADE_MCP_READONLY_LANE"] = lane
        env.update(extra_env or {})
        argv = [sys.executable, "-c", _BOOT, str(hooks)]
        self.http = http
        self.port_file = tmp_path / "mcp.port"
        if http:
            env["CASCADE_MCP_TOKEN"] = TOKEN
            argv += ["--http", "127.0.0.1:0", "--http-port-file", str(self.port_file)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, text=True, bufsize=1)
        self.stderr: list[str] = []
        self._out: queue.Queue = queue.Queue()
        self._id = 0

        def _pump_out():
            for line in self.proc.stdout:
                self._out.put(line)
            self._out.put(None)

        def _pump_err():  # drained so a chatty server can never block on a full pipe
            for line in self.proc.stderr:
                self.stderr.append(line)

        threading.Thread(target=_pump_out, daemon=True).start()
        threading.Thread(target=_pump_err, daemon=True).start()
        self.port = self._wait_port() if http else None

    # ── transport ──────────────────────────────────────────────────────────

    def _wait_port(self, timeout=60.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.port_file.exists() and self.port_file.read_text().strip():
                return int(self.port_file.read_text().strip())
            self._alive()
            time.sleep(0.01)
        pytest.fail("server never wrote its port file")

    def _alive(self):
        if self.proc.poll() is not None:
            pytest.fail(f"server exited {self.proc.returncode}: {''.join(self.stderr)[-3000:]}")

    def send(self, method: str, params: dict | None = None) -> int:
        self._id += 1
        frame = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()
        return self._id

    def notify(self, method: str, params: dict | None = None) -> None:
        frame: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()

    def call(self, name: str, args: dict | None = None) -> int:
        return self.send("tools/call", {"name": name, "arguments": args or {}})

    def recv(self, timeout=60.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._out.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                pytest.fail(f"no response within {timeout}s; server stderr tail: "
                            f"{''.join(self.stderr)[-3000:]}")
            assert line is not None, f"server died: {''.join(self.stderr)[-3000:]}"
            frame = json.loads(line)  # every stdout line must be a protocol frame
            if "id" not in frame:
                continue  # notifications (tools/list_changed)
            return frame

    def request(self, method: str, params: dict | None = None, timeout=60.0) -> dict:
        req = self.send(method, params)
        resp = self.recv(timeout)
        assert resp["id"] == req
        return resp

    def post(self, method: str, params: dict | None = None, timeout=120.0) -> dict:
        """HTTP: one blocking JSON request (no SSE), the server's own reply."""
        self._id += 1
        frame = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            frame["params"] = params
        conn = http.client.HTTPConnection(loopback_host(), self.port, timeout=timeout)
        try:
            conn.request("POST", "/mcp", body=json.dumps(frame).encode(), headers={
                "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                "Accept": "application/json"})
            resp = conn.getresponse()
            body = resp.read().decode()
            assert resp.status == 200, (resp.status, body)
            return json.loads(body)
        finally:
            conn.close()

    # ── barriers ───────────────────────────────────────────────────────────

    def hold(self, tag: str) -> None:
        (self.dir / f"hold-{tag}").write_text("")

    def wait_at(self, tag: str, timeout=60.0) -> str:
        """The name of the thread that reached barrier `tag` (polled file)."""
        at = self.dir / f"at-{tag}"
        deadline = time.monotonic() + timeout
        while not at.exists():
            self._alive()
            if time.monotonic() > deadline:
                pytest.fail(f"nothing reached barrier {tag!r} within {timeout}s")
            time.sleep(0.005)
        return at.read_text()

    def release(self, tag: str) -> None:
        (self.dir / f"at-{tag}").unlink(missing_ok=True)
        (self.dir / f"release-{tag}").write_text("")

    def motion_threads(self) -> set[str]:
        path = self.dir / "motion-threads"
        return set(path.read_text().split()) if path.exists() else set()

    def close(self):
        for tag in ("motion", "world_state"):  # never leave a held thread behind
            (self.dir / f"release-{tag}").write_text("")
        if self.proc.poll() is None:
            if self.http:
                self.proc.terminate()
            else:
                self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


@pytest.fixture
def server_factory(tmp_path):
    made: list[_Server] = []

    def make(lane: str | None, **kw) -> _Server:
        s = _Server(tmp_path / f"s{len(made)}", lane=lane, **kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def _started(s: _Server) -> _Server:
    """initialize + one serial world_state: the runtime (and, when on, the
    lane) exists before any motion is held."""
    s.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                             "clientInfo": {"name": "pytest-b46", "version": "0"}})
    s.notify("notifications/initialized")
    payload, is_err = _payload(s.request("tools/call", {"name": "world_state", "arguments": {}}))
    assert not is_err and payload["ok"]
    assert "served_during_motion" not in payload  # nothing was moving
    return s


def _hold_motion(s: _Server) -> int:
    """Start a `wave` and hold it at its first streamed waypoint."""
    s.hold("motion")
    motion_id = s.call("wave", {"cycles": 1})
    assert s.wait_at("motion") == WORKER_THREAD  # motions run on the serial worker
    return motion_id


# ── premises (pass on main by design) ──────────────────────────────────────

def test_premise_lane_tools_are_server_side_reads_outside_execute():
    """The four lane tools are MCP server extras answered by the server's own
    read functions -- not TOOL_SPECS skills (which go through
    SkillRuntime.execute(): trace row, memory event, envelope record,
    `last_frame`, watcher pause, per-call scratchpads), not motions, not
    verified effects, not the stop path, and capability-gated only where
    saved verdicts need the verifier."""
    from cascade.agent.effects import POSTCONDITIONS
    from cascade.apps.capabilities import CAP_VERIFIER, TOOL_REQUIREMENTS
    from cascade.apps.mcp_server import _EXTRA_TOOLS, _PROGRAM_TOOLS
    from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS

    extras = {t["name"] for t in _EXTRA_TOOLS}
    specs = {t["name"] for t in TOOL_SPECS}
    for tool in LANE_TOOLS:
        assert tool in extras
        assert tool not in specs | _MOTION_SKILLS | set(POSTCONDITIONS)
        assert tool not in _PROGRAM_TOOLS | {"emergency_stop", "reset_stop"}
    # gated on nothing a probe could flip mid-motion, except that saved
    # verdicts need the verifier (the lane re-checks the matrix like call_tool)
    assert {t: TOOL_REQUIREMENTS[t] for t in LANE_TOOLS if t in TOOL_REQUIREMENTS} == {
        "verify_last_action": (CAP_VERIFIER,)}
    # the observation skills a host would also like mid-motion are execute()
    # skills that fuse beliefs and run the shared detector: NOT lane material
    assert {"describe_scene", "get_observation"} <= specs


@pytest.mark.parametrize("lane", [None, "0"], ids=["flag-absent", "flag-off"])
def test_golden_off_reads_wait_behind_a_held_motion_on_the_serial_worker(server_factory, lane):
    """Golden for the default (and explicit off): the serial server, byte for
    byte -- every read queues behind the held motion, runs on the worker
    thread after it, in arrival order, and carries no lane marker."""
    s = _started(server_factory(lane))
    a = _hold_motion(s)
    s.hold("world_state")
    reads = [s.call(t) for t in LANE_TOOLS]
    ping = s.send("ping")
    assert s.recv()["id"] == ping  # the reader routed every frame above, then answered
    s.release("motion")
    resp = s.recv()
    assert resp["id"] == a
    payload, is_err = _payload(resp)
    assert not is_err and payload["ok"] and "served_during_motion" not in payload
    assert s.wait_at("world_state") == WORKER_THREAD  # after the motion, on the worker
    s.release("world_state")
    answers = [s.recv() for _ in reads]
    assert [r["id"] for r in answers] == reads
    for r in answers:
        payload, _ = _payload(r)
        assert "served_during_motion" not in payload
    assert s.motion_threads() == {WORKER_THREAD}


# ── behaviour (fail on main) ───────────────────────────────────────────────

def test_read_only_calls_answer_while_a_motion_is_held(server_factory):
    s = _started(server_factory("1"))
    a = _hold_motion(s)
    reads = [s.call(t) for t in LANE_TOOLS]
    answers = [s.recv(timeout=30) for _ in reads]  # all four while the motion is HELD
    assert [r["id"] for r in answers] == reads  # one lane, arrival order
    by_tool = {}
    for tool, r in zip(LANE_TOOLS, answers):
        payload, is_err = _payload(r)
        assert not is_err and payload.get("ok") is not False, (tool, payload)
        marker = payload["served_during_motion"]
        assert marker["motion"] == "wave" and marker["lane"] == "read_only"
        assert "in flux" in marker["note"]
        by_tool[tool] = (payload, r)
    world, _ = by_tool["world_state"]
    assert "objects" in world and world["session_state"]["source"] == "mcp_runtime"
    snap, snap_resp = by_tool["camera_snapshot"]
    image = next(b for b in snap_resp["result"]["content"] if b["type"] == "image")
    assert base64.b64decode(image["data"])[:2] == b"\xff\xd8"  # a real JPEG
    assert snap["camera"] == "mock" and "passive" in snap["served_during_motion"]["note"]
    knowledge, _ = by_tool["robot_knowledge"]
    assert "primitive_envelopes" in knowledge
    verified, _ = by_tool["verify_last_action"]
    assert "digest" in verified or verified.get("verification") == "disabled"

    s.release("motion")
    resp = s.recv()
    assert resp["id"] == a
    payload, is_err = _payload(resp)
    assert not is_err and payload["ok"] and "served_during_motion" not in payload
    # nothing moving: the same read is served serially again, unmarked
    payload, is_err = _payload(s.request("tools/call", {"name": "world_state", "arguments": {}}))
    assert not is_err and "served_during_motion" not in payload
    assert s.motion_threads() == {WORKER_THREAD}


def test_a_second_motion_still_waits_for_the_first(server_factory):
    s = _started(server_factory("1"))
    a = _hold_motion(s)
    b = s.call("wave", {"cycles": 1})
    c = s.call("world_state")
    first = s.recv(timeout=30)
    assert first["id"] == c, "the second motion jumped the queue onto the lane"
    assert _payload(first)[0]["served_during_motion"]["motion"] == "wave"
    s.release("motion")
    assert s.recv()["id"] == a
    second = s.recv()
    assert second["id"] == b
    payload, is_err = _payload(second)
    assert not is_err and payload["ok"]
    assert s.motion_threads() == {WORKER_THREAD}  # no motion ever ran off the worker


def test_non_read_only_and_hidden_tools_still_wait(server_factory):
    s = _started(server_factory("1", extra_env={"CASCADE_HIDE_TOOLS": "verify_last_action"}))
    a = _hold_motion(s)
    waiting = [s.call("get_observation"), s.call("describe_scene"), s.call("list_objects"),
               s.call("task_memory"), s.call("live_view_url"), s.call("verify_last_action")]
    c = s.call("world_state")
    first = s.recv(timeout=30)
    assert first["id"] == c  # the lane answered; nothing queued before it did
    assert _payload(first)[0]["served_during_motion"]["motion"] == "wave"
    s.release("motion")
    assert s.recv()["id"] == a
    answers = [s.recv() for _ in waiting]
    assert [r["id"] for r in answers] == waiting  # serial, in order, after the motion
    for r in answers:
        payload, _ = _payload(r)
        assert "served_during_motion" not in payload
    hidden, is_err = _payload(answers[-1])
    assert is_err and "disabled by the operator" in hidden["error"]


def test_stop_is_answered_while_a_lane_call_and_the_motion_are_both_held(server_factory):
    s = _started(server_factory("1"))
    a = _hold_motion(s)
    s.hold("world_state")
    c = s.call("world_state")
    assert s.wait_at("world_state") == LANE_THREAD  # served off the worker
    stop = s.call("emergency_stop")
    first = s.recv(timeout=30)
    assert first["id"] == stop, "a held read-only call delayed the stop"
    assert _payload(first)[0]["stopped"] is True
    s.release("world_state")
    resp = s.recv()
    assert resp["id"] == c and _payload(resp)[0]["served_during_motion"]["motion"] == "wave"
    s.release("motion")
    resp = s.recv()
    assert resp["id"] == a
    payload, is_err = _payload(resp)
    assert is_err and "e-stop" in payload["error"], payload
    payload, is_err = _payload(s.request("tools/call", {"name": "move_home", "arguments": {}}))
    assert is_err and "e-stop" in payload["error"]  # the latch holds until reset_stop


def test_cancel_of_the_held_motion_still_latches_the_estop(server_factory):
    """The lane never publishes itself as the in-flight call: a host cancel
    of the motion is still "stop the robot" while lane calls are served."""
    s = _started(server_factory("1"))
    a = _hold_motion(s)
    c = s.call("world_state")
    assert s.recv(timeout=30)["id"] == c
    s.notify("notifications/cancelled", {"requestId": a, "reason": "user"})
    ping = s.send("ping")
    assert s.recv()["id"] == ping  # the reader handled the cancel before this
    s.release("motion")
    resp = s.recv()
    assert resp["id"] == a and _payload(resp)[1]
    payload, is_err = _payload(s.request("tools/call", {"name": "move_home", "arguments": {}}))
    assert is_err and "e-stop" in payload["error"]


def test_http_transport_shares_the_lane(server_factory):
    s = server_factory("1", http=True)
    s.post("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                          "clientInfo": {"name": "pytest-b46", "version": "0"}})
    payload, _ = _payload(s.post("tools/call", {"name": "world_state", "arguments": {}}))
    assert "served_during_motion" not in payload
    s.hold("motion")
    out: dict = {}
    mover = threading.Thread(target=lambda: out.update(
        motion=s.post("tools/call", {"name": "wave", "arguments": {"cycles": 1}})), daemon=True)
    mover.start()
    assert s.wait_at("motion") == WORKER_THREAD
    payload, is_err = _payload(s.post("tools/call", {"name": "world_state", "arguments": {}},
                                      timeout=30))
    assert not is_err and payload["served_during_motion"]["motion"] == "wave"
    s.release("motion")
    mover.join(timeout=120)
    assert not mover.is_alive()
    payload, is_err = _payload(out["motion"])
    assert not is_err and payload["ok"] and "served_during_motion" not in payload


# ── in-process units (fail on main: the lane does not exist there) ─────────

def test_flag_resolution_env_beats_config_and_demo_yaml_ships_off(monkeypatch):
    from cascade.apps.mcp_server import _readonly_lane_enabled
    from cascade.config import load_demo_config

    monkeypatch.delenv("CASCADE_MCP_READONLY_LANE", raising=False)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert cfg.get("mcp", {}).get("readonly_lane") is False  # shipped default: off
    assert _readonly_lane_enabled(cfg) is False
    assert _readonly_lane_enabled(None) is False
    for raw, want in ((True, True), ("true", True), ("on", True), (False, False),
                      ("false", False), ("0", False), (None, False)):
        cfg._data["mcp"]["readonly_lane"] = raw
        assert _readonly_lane_enabled(cfg) is want, raw
    cfg._data["mcp"]["readonly_lane"] = True
    monkeypatch.setenv("CASCADE_MCP_READONLY_LANE", "0")
    assert _readonly_lane_enabled(cfg) is False  # env beats config
    cfg._data["mcp"]["readonly_lane"] = False
    for raw in ("1", "true", "yes", "on"):
        monkeypatch.setenv("CASCADE_MCP_READONLY_LANE", raw)
        assert _readonly_lane_enabled(cfg) is True, raw
    monkeypatch.setenv("CASCADE_MCP_READONLY_LANE", "")
    assert _readonly_lane_enabled(cfg) is False  # empty = unset -> config


def test_lane_tool_set_is_pinned():
    from cascade.apps.mcp_server import LANE_THREAD_NAME, READONLY_LANE_TOOLS

    assert READONLY_LANE_TOOLS == frozenset(LANE_TOOLS)
    assert LANE_THREAD_NAME == LANE_THREAD


def test_routing_off_is_the_plain_queue_put():
    """Golden at the routing seam: no lane object -> the exact pre-B46
    `inbox.put((m, send))` for every frame, lane tools included."""
    from cascade.apps.mcp_server import McpSkillServer, _enqueue

    server = McpSkillServer()
    assert server._readonly_lane is None
    server._inflight = (7, "wave")  # even mid-motion
    inbox: queue.Queue = queue.Queue()
    sent = []
    for m in ({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
               "params": {"name": "world_state", "arguments": {}}},
              {"jsonrpc": "2.0", "id": 9, "method": "tools/list"}):
        _enqueue(server, inbox, m, sent.append)
        item = inbox.get_nowait()
        assert item[0] is m and item[1] == sent.append and len(item) == 2
    assert inbox.empty() and sent == []


def test_offer_admits_only_lane_tools_with_ids_during_a_motion(monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer, _ReadOnlyLane

    monkeypatch.delenv("CASCADE_HIDE_TOOLS", raising=False)
    server = McpSkillServer()
    lane = _ReadOnlyLane(server, start=False)
    sent = []

    def call(name, req_id=1, method="tools/call"):
        m = {"jsonrpc": "2.0", "method": method, "params": {"name": name, "arguments": {}}}
        if req_id is not None:
            m["id"] = req_id
        return m

    assert not lane.offer(call("world_state"), sent.append)  # nothing in flight
    server._inflight = (1, "get_observation")
    assert not lane.offer(call("world_state"), sent.append)  # in flight, but no motion
    server._inflight = (1, "wave")
    for tool in LANE_TOOLS:
        assert lane.offer(call(tool, req_id=f"r-{tool}"), sent.append), tool
    for tool in ("describe_scene", "get_observation", "move_home", "reset_stop",
                 "emergency_stop", "task_memory", "live_view_url", "list_programs", "run_program"):
        assert not lane.offer(call(tool), sent.append), tool
    assert not lane.offer(call("world_state", req_id=None), sent.append)  # notification
    assert not lane.offer(call("world_state", req_id=[1]), sent.append)  # unhashable id
    assert not lane.offer(call("world_state", method="tools/list"), sent.append)
    assert not lane.offer({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": ["x"]},
                          sent.append)
    monkeypatch.setenv("CASCADE_HIDE_TOOLS", "world_state")
    assert not lane.offer(call("world_state"), sent.append)  # hidden: the worker rejects it
    monkeypatch.delenv("CASCADE_HIDE_TOOLS")
    # run_program is a motion only when the programs tier is on (resolved)
    server._inflight = (1, "run_program")
    server._programs_flag = None
    assert not lane.offer(call("world_state"), sent.append)
    server._programs_flag = False
    assert not lane.offer(call("world_state"), sent.append)
    server._programs_flag = True
    assert lane.offer(call("world_state", req_id="during-program"), sent.append)
    queued = []
    while not lane._queue.empty():
        queued.append(lane._queue.get_nowait())
    assert [(m["id"], motion) for m, _send, motion in queued] == (
        [(f"r-{t}", "wave") for t in LANE_TOOLS] + [("during-program", "run_program")])
    assert sent == []  # offering never answers on the reader


def test_mark_served_during_motion_shapes():
    from cascade.apps.mcp_server import _mark_served_during_motion, _text_result

    image = {"content": [{"type": "image", "data": "x", "mimeType": "image/jpeg"},
                         {"type": "text", "text": json.dumps({"camera": "mock"})}],
             "isError": False}
    out = _mark_served_during_motion(image, "pick_and_place", "camera_snapshot")
    assert out["content"][0] == image["content"][0] and out["isError"] is False
    payload = json.loads(out["content"][-1]["text"])
    assert payload["camera"] == "mock"
    assert payload["served_during_motion"]["motion"] == "pick_and_place"
    assert "passive" in payload["served_during_motion"]["note"]
    assert "served_during_motion" not in json.loads(image["content"][-1]["text"])  # not mutated
    err = _mark_served_during_motion(_text_result({"ok": False, "error": "x"}, is_error=True),
                                     "wave", "world_state")
    payload = json.loads(err["content"][-1]["text"])
    assert err["isError"] is True and payload["error"] == "x"
    assert payload["served_during_motion"]["motion"] == "wave"
    assert "passive" not in payload["served_during_motion"]["note"]
    odd = _mark_served_during_motion({"content": [{"type": "text", "text": "plain"}],
                                      "isError": False}, "wave", "world_state")
    assert odd["content"][0]["text"] == "plain"
    assert json.loads(odd["content"][-1]["text"])["served_during_motion"]["motion"] == "wave"
    # only the LAST text block is the summary: an earlier JSON block is left alone
    two = [{"type": "text", "text": json.dumps({"caption": 1})}, {"type": "text", "text": "plain"}]
    out = _mark_served_during_motion({"content": two, "isError": False}, "wave", "world_state")
    assert out["content"][:2] == two and len(out["content"]) == 3
    bare = _mark_served_during_motion({"isError": True}, "wave", "world_state")
    assert bare["isError"] is True and len(bare["content"]) == 1


def test_lane_thread_drops_cancelled_spends_raced_cancels_and_never_dies(monkeypatch):
    """`_ReadOnlyLane._run` on a pre-filled queue (one loop, deterministic
    order, bounded by its close sentinel): a call cancelled before the lane
    reached it is dropped like the worker drops one; a cancel racing a
    running call is spent afterwards; a raising read, a raising response
    builder and a dead transport each cost one answer, never the lane."""
    import cascade.apps.mcp_server as mcp

    server = mcp.McpSkillServer()
    lane = mcp._ReadOnlyLane(server, start=False)
    ran = []

    def call_readonly(name, args, motion):
        ran.append(args["tag"])
        if args["tag"] == "raced":
            server._cancelled_ids.add("raced")  # the host cancels while it runs
        if args["tag"] == "boom":
            raise RuntimeError("read failed")
        return mcp._mark_served_during_motion(mcp._text_result({"ok": True}), motion, name)

    real_response = mcp._lane_response

    def lane_response(srv, m, motion):
        if m["id"] == "crash":
            raise ValueError("builder bug")
        return real_response(srv, m, motion)

    monkeypatch.setattr(server, "call_readonly", call_readonly)
    monkeypatch.setattr(mcp, "_lane_response", lane_response)
    sent = []

    def send(resp):
        if resp["id"] == "closed":
            raise BrokenPipeError("host went away")
        sent.append(resp)

    server._cancelled_ids.add("gone")
    for tag in ("gone", "raced", "boom", "crash", "closed", "after"):
        lane._queue.put(({"jsonrpc": "2.0", "id": tag, "method": "tools/call",
                          "params": {"name": "world_state", "arguments": {"tag": tag}}},
                         send, "wave"))
    lane.close()
    runner = threading.Thread(target=lane._run, daemon=True)  # the lane loop, nothing else running
    runner.start()
    runner.join(timeout=60)
    assert not runner.is_alive(), "the lane did not stop at its close sentinel"
    assert ran == ["raced", "boom", "closed", "after"]  # "gone" never ran; "crash" never reached it
    assert [r["id"] for r in sent] == ["raced", "boom", "crash", "after"]
    assert server._cancelled_ids == set()  # the dropped and the raced cancel are both spent
    for r in sent:
        payload, _ = _payload(r)
        assert payload["served_during_motion"]["motion"] == "wave"
    for r, error in ((sent[1], "RuntimeError: read failed"), (sent[2], "ValueError: builder bug")):
        payload, is_err = _payload(r)
        assert is_err and payload["error"] == error


def test_call_readonly_refuses_anything_but_a_lane_tool_on_a_built_runtime():
    from types import SimpleNamespace

    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    out = server.call_readonly("world_state", {}, "wave")  # no runtime yet
    payload = json.loads(out["content"][-1]["text"])
    assert out["isError"] and "not served on the read-only lane" in payload["error"]
    assert payload["served_during_motion"]["motion"] == "wave"
    snap = server._lane_camera_snapshot(SimpleNamespace(rig=None), None)
    assert snap["isError"] and "no camera rig" in json.loads(snap["content"][-1]["text"])["error"]


def test_lane_calls_never_enter_execute_lock_or_write_the_runtime(monkeypatch, tmp_path):
    """The real mock runtime, the real lane entry (`call_readonly`): each lane
    tool answers while `execute`, `observe*`, depth filling, the harness
    heartbeat and the execution lock all refuse to be touched, and the
    runtime's frame, trace, memory and envelope are exactly as before."""
    from cascade.apps.mcp_server import McpSkillServer

    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "0")
    monkeypatch.setenv("CASCADE_VIEW", "0")
    monkeypatch.setenv("CASCADE_MCP_READONLY_LANE", "1")
    for name in ("CASCADE_CAMERAS", "CASCADE_ARMS", "CASCADE_HIDE_TOOLS", "CASCADE_PROGRAMS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_ARM", "mock")
    for key, port in _PORTS.items():
        monkeypatch.setenv(key, port)
    server = McpSkillServer()
    try:
        runtime = server._ensure_runtime()
        lane = server._readonly_lane
        assert lane is not None and lane._thread.name == LANE_THREAD and lane._thread.is_alive()
        runtime.observe()  # a known last_frame to compare against
        frame_before = runtime.last_frame
        trace_rows = (runtime.trace.run_dir / "trace.jsonl")
        rows_before = trace_rows.read_text() if trace_rows.exists() else ""
        memory_before = runtime.memory.digest(max_lines=1000)
        envelope_before = runtime.envelope.stats()

        def refuse(*_a, **_k):
            raise AssertionError("a lane call touched the motion's state")

        class _NoLock:
            def __enter__(self):
                refuse()

            def acquire(self, *a, **k):
                refuse()

        with monkeypatch.context() as mp:
            mp.setattr(runtime, "execute", refuse)
            mp.setattr(runtime, "_execute_skill", refuse)
            mp.setattr(runtime, "observe", refuse)
            mp.setattr(runtime, "observe_fresh", refuse)
            mp.setattr(runtime.depth, "ensure_depth", refuse)
            mp.setattr(runtime.arm.harness, "heartbeat", refuse)
            mp.setattr(server, "_exec_lock", _NoLock())
            for tool in LANE_TOOLS:
                out = server.call_readonly(tool, {}, "pick_and_place")
                payload = json.loads(out["content"][-1]["text"])
                assert out["isError"] is False, (tool, payload)
                assert payload["served_during_motion"]["motion"] == "pick_and_place"
            named = server.call_readonly("camera_snapshot", {"camera": "mock"}, "wave")
            assert json.loads(named["content"][-1]["text"])["camera"] == "mock"
            bad = server.call_readonly("camera_snapshot", {"camera": "nope"}, "wave")
            assert bad["isError"] is True
            assert json.loads(bad["content"][-1]["text"])["served_during_motion"]["motion"] == "wave"
            # a tool the capability matrix withholds is rejected on the lane
            # exactly as call_tool rejects it (verdicts need the verifier)
            mp.setattr(runtime, "effects", None)
            withheld = server.call_readonly("verify_last_action", {}, "wave")
            payload = json.loads(withheld["content"][-1]["text"])
            assert withheld["isError"] is True and "not available on this rig" in payload["error"]
            # a non-lane tool can never run on the lane, even called directly
            refused = server.call_readonly("move_home", {}, "wave")
            payload = json.loads(refused["content"][-1]["text"])
            assert refused["isError"] is True and "not served on the read-only lane" in payload["error"]
        assert runtime.last_frame is frame_before
        assert (trace_rows.read_text() if trace_rows.exists() else "") == rows_before
        assert runtime.memory.digest(max_lines=1000) == memory_before
        assert runtime.envelope.stats() == envelope_before
    finally:
        server.shutdown()
    server._readonly_lane._thread.join(timeout=10)
    assert not server._readonly_lane._thread.is_alive()  # shutdown closes the lane


def test_off_builds_no_lane_and_starts_no_thread(monkeypatch, tmp_path):
    from cascade.apps.mcp_server import McpSkillServer

    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "0")
    monkeypatch.setenv("CASCADE_VIEW", "0")
    monkeypatch.delenv("CASCADE_MCP_READONLY_LANE", raising=False)
    for name in ("CASCADE_CAMERAS", "CASCADE_ARMS", "CASCADE_HIDE_TOOLS", "CASCADE_PROGRAMS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_ARM", "mock")
    for key, port in _PORTS.items():
        monkeypatch.setenv(key, port)
    server = McpSkillServer()
    try:
        server._ensure_runtime()
        assert server._readonly_lane is None
        assert LANE_THREAD not in {t.name for t in threading.enumerate()}
    finally:
        server.shutdown()
