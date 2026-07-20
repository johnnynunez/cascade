"""MCP stdio server: expose the arm's skill runtime to any MCP agent.

This turns the demo into a tool surface for agent platforms -- Hermes
(~/.hermes/config.yaml), Claude Code (.mcp.json), Claude Desktop, Codex CLI.
The platform's agent replaces wrc_demo's built-in orchestrator as the brain;
the safety harness, tracing, episodic memory and skills are identical.

Protocol: MCP over stdio, newline-delimited JSON-RPC 2.0. Implemented by
hand (initialize / tools/list / tools/call / ping) so the demo gains no new
dependency. stdout carries ONLY protocol frames; everything else goes to
stderr.

Configuration via environment (set in the MCP server entry):
    WRC_CAMERA          camera profile (default: mock)
    WRC_CAMERAS         comma-separated camera profiles; first one is the
                        manipulation camera (overrides WRC_CAMERA)
    WRC_ARM             arm profile    (default: mock)
    WRC_RUN_DIR         trace directory (default: <repo>/runs/mcp_<pid>)
    WRC_DETECTOR_MODEL  override detector weights (e.g. a yolo11n.pt path
                        for closed-set COCO until the CLIP fork is installed)
    WRC_DETECT_CLASSES  comma-separated default vocabulary for observations
    WRC_VIEW            "0" disables the live camera window (default: open it
                        whenever DISPLAY is set, so the audience always sees
                        what the camera sees)
    WRC_PREWARM         "0" disables perception pre-warm at startup
                        (default: cameras + detector + world model come up
                        immediately so the first command is fast)
    WRC_STREAM          "0" disables the MJPEG livestream dashboard
    WRC_STREAM_PORT     dashboard port (default: from configs/demo.yaml)
    WRC_HIDE_TOOLS      comma-separated tool names to remove from the agent's
                        surface (delisted AND rejected if called anyway).
                        Booth/attendee sessions hide reset_stop so a latched
                        e-stop can only be cleared by staff, not by a model
                        helpfully "fixing" it. emergency_stop cannot be hidden.

Latency contract (why this server is fast): perception pre-warms in the
background the moment the gateway starts -- N camera streams, the detector,
and the WorldWatcher that keeps the belief store hot. The ARM stays
unpowered until the first motion tool call (LazyArm). A routine command like
"pick and place pink object" is ONE pick_and_place tool call that resolves
against the warm world model and runs deterministically: no LLM round-trips
inside the loop, total time ~ arm motion time.

Stop channel (why there is a reader thread): tool calls execute serially on
the worker loop, so a 150 s pick_and_place used to queue emergency_stop
behind it -- useless exactly when it matters. The stdin reader now latches
the e-stop the moment an emergency_stop frame ARRIVES: the harness checks
the latch on every 50 Hz waypoint, so the running motion aborts mid-stream
(the same cross-thread path the dashboard STOP button uses). A stop that
arrives while the runtime is still BUILDING is remembered (_stop_pending)
and applied the instant the build finishes -- an acknowledged stop is never
lost. Likewise, `notifications/cancelled` for an in-flight MOTION tool (Esc
in Claude Code mid-pick) means "stop the robot", not "orphan the motion
server-side and keep moving". `ping` is also answered from the reader so
host keepalives don't starve behind a motion. SIGINT latches the e-stop too
(freeze, no free-fall); a second SIGINT exits -- which disables torque on
the RS arm, so park it first. SIGUSR1 clears the e-stop: that is the STAFF
reset channel when reset_stop is hidden from attendees
(`pkill -USR1 -f wrc_demo.apps.mcp_server` requires shell access to the
rig, which is exactly the staff/attendee boundary).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "wrc-demo", "version": "0.1.0"}

# task_done belongs to the built-in loop; MCP agents manage their own tasks.
_EXCLUDED_TOOLS = {"task_done"}


def _hidden_tools() -> set[str]:
    """Operator-hidden tools (WRC_HIDE_TOOLS), read per call so tests can
    vary it per subprocess. emergency_stop is never hideable: an operator
    typo must not be able to remove the stop path."""
    hidden = {
        t.strip()
        for t in os.environ.get("WRC_HIDE_TOOLS", "").split(",")
        if t.strip()
    }
    hidden.discard("emergency_stop")
    return hidden

_EXTRA_TOOLS = [
    {
        "name": "camera_snapshot",
        "description": (
            "Capture a camera frame and return it as an image, with the "
            "depth source noted. Use this to SEE the workspace. Optional "
            "`camera` selects one of the rig cameras (see world_state)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"camera": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "world_state",
        "description": (
            "INSTANT text snapshot of the live world model: every object "
            "with color + 3D position + freshness, what the gripper holds, "
            "camera FPS, and the livestream URL. Perception runs "
            "continuously, so prefer this over camera_snapshot when you "
            "only need to know WHAT is where -- it costs no image tokens "
            "and returns immediately."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "live_view_url",
        "description": (
            "URL of the live dashboard (N camera MJPEG streams + robot "
            "narration feed). Share it with the human so they can watch "
            "the robot work."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "emergency_stop",
        "description": (
            "Soft-stop the arm immediately: freeze in place and reject all "
            "further motion until reset_stop is called."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "reset_stop",
        "description": "Clear a previous emergency_stop so motion tools work again.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


class McpSkillServer:
    def __init__(self):
        self._runtime = None
        self._arm = None
        self._init_error: str | None = None
        self._init_lock = threading.Lock()
        # (request id, tool name) the worker is executing right now; read by
        # the stdin thread to honor cancellations of in-flight motion calls.
        self._inflight: tuple[object, str | None] | None = None
        # ids cancelled before the worker reached them (client gave up):
        # the worker drops these without executing. _cancel_lock makes the
        # {check set / publish _inflight} and {read _inflight / add to set}
        # sections atomic -- without it a cancel arriving between the
        # worker's check and its publish is lost and a cancelled MOTION
        # call executes (2026-07-20 review).
        self._cancelled_ids: set = set()
        self._cancel_lock = threading.Lock()
        # a stop acknowledged while the runtime is still building must be
        # applied the moment the build finishes, or the queued motion runs
        # after "stopped: true" was already answered (2026-07-20 review,
        # critical). Set BEFORE reading _runtime in stop_now(); checked
        # AFTER assigning _runtime in _ensure_runtime(); cleared only by
        # reset_now().
        self._stop_pending = False
        # one arm, one command: the worker's skill execution and the
        # dashboard reflex chat are the only two threads that can drive the
        # arm in MCP mode -- they must never overlap (concurrent streaming
        # would defeat the per-waypoint velocity gate; 2026-07-20 review,
        # critical). The worker blocks on it; the chat refuses instead.
        self._exec_lock = threading.Lock()

    # ── runtime lifecycle ────────────────────────────────────────────────

    def _ensure_runtime(self, _poison: bool = True):
        with self._init_lock:
            if self._runtime is not None:
                return self._runtime
            if self._init_error is not None:
                raise RuntimeError(f"hardware init failed earlier: {self._init_error}")
            from ..apps.demo import build_runtime
            from ..config import PACKAGE_ROOT, load_demo_config

            cameras = [
                c.strip()
                for c in os.environ.get(
                    "WRC_CAMERAS", os.environ.get("WRC_CAMERA", "mock")
                ).split(",")
                if c.strip()
            ]
            arm = os.environ.get("WRC_ARM", "mock")
            run_dir = Path(
                os.environ.get("WRC_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}")
            )
            try:
                # Anything the stack prints must not corrupt the protocol stream.
                with contextlib.redirect_stdout(sys.stderr):
                    cfg = load_demo_config(cameras=cameras, arm=arm, llm="mock")
                    det_model = os.environ.get("WRC_DETECTOR_MODEL")
                    if det_model:
                        cfg._data["detector"]["model"] = det_model
                    classes = os.environ.get("WRC_DETECT_CLASSES")
                    if classes:
                        cfg._data["detect_classes"] = [
                            c.strip() for c in classes.split(",") if c.strip()
                        ]
                    port = os.environ.get("WRC_STREAM_PORT")
                    if port:
                        cfg._data.setdefault("stream", {})["port"] = int(port)
                    view = os.environ.get("WRC_VIEW", "1") != "0" and bool(
                        os.environ.get("DISPLAY")
                    )
                    serve = os.environ.get("WRC_STREAM", "1") != "0"
                    # lazy_arm: perception comes up now; motors stay untouched
                    # until the first motion command.
                    self._runtime, self._arm = build_runtime(
                        cfg, run_dir, view=view, lazy_arm=True, serve=serve
                    )
                    if self._runtime.stream_server is not None:
                        # Staff stop channel that bypasses stdio entirely:
                        # the dashboard STOP button freezes the arm from any
                        # browser on the LAN, even mid-tool-call. (The demo
                        # CLI wires this in main(); MCP mode must do it here
                        # or the button is dead.)
                        self._runtime.stream_server.set_cancel_fn(
                            self._runtime.arm.stop
                        )
                        # Dashboard chat: tier-1 reflex grammar only, zero
                        # LLM -- the booth's MCP-host-outage fallback.
                        self._runtime.stream_server.set_task_fn(
                            self._reflex_chat_task
                        )
                    if self._stop_pending:
                        # a stop acknowledged during the build: latch now,
                        # before any queued motion call can run
                        self._runtime.arm.stop()
                url = (
                    self._runtime.stream_server.url
                    if self._runtime.stream_server is not None else "disabled"
                )
                print(
                    f"[wrc-mcp] runtime up: cameras={cameras} arm={arm} (lazy) "
                    f"livestream={url}",
                    file=sys.stderr,
                )
            except Exception as e:
                if _poison:
                    self._init_error = f"{type(e).__name__}: {e}"
                raise
            return self._runtime

    def prewarm_async(self) -> None:
        """Bring perception up in the background so the first command is
        instant. A prewarm failure must NOT permanently poison the server:
        transient conditions (camera enumerating, port busy) often clear by
        the time a human sends the first command, so prewarm never sets the
        poison flag at all (a set-then-clear reset used to race the first
        tools/call, failing it spuriously) and the first tools/call rebuilds
        from scratch."""

        def _warm():
            try:
                self._ensure_runtime(_poison=False)
            except Exception as e:
                print(f"[wrc-mcp] prewarm failed (will retry on first call): {e}",
                      file=sys.stderr)

        threading.Thread(target=_warm, daemon=True, name="wrc-prewarm").start()

    def shutdown(self):
        # taking _init_lock waits out an in-flight prewarm build, so a
        # runtime that finishes building after EOF is still torn down
        with self._init_lock, contextlib.redirect_stdout(sys.stderr):
            if self._runtime is not None:
                from ..apps.demo import shutdown_runtime

                shutdown_runtime(self._runtime, self._arm)

    # ── out-of-band stop/reset (called from the stdin thread / signals) ──

    def stop_now(self) -> dict:
        """Latch the e-stop from any thread, without ever building the
        runtime. Ordering matters: _stop_pending is set BEFORE reading
        _runtime, and _ensure_runtime checks it AFTER assigning _runtime,
        so a stop concurrent with the build is caught by one side or the
        other in every interleaving."""
        self._stop_pending = True
        rt = self._runtime
        if rt is None:
            print("[wrc-mcp] EMERGENCY STOP latched (runtime still starting)",
                  file=sys.stderr)
            return _text_result(
                {"ok": True, "stopped": True,
                 "note": "stop latched; motors were never powered and any "
                         "runtime that finishes starting comes up stopped -- "
                         "staff clears it via reset_stop / SIGUSR1"}
            )
        with contextlib.redirect_stdout(sys.stderr):
            rt.arm.stop()
        print("[wrc-mcp] EMERGENCY STOP latched (out-of-band)", file=sys.stderr)
        return _text_result(
            {"ok": True, "stopped": True,
             "note": "arm frozen; call reset_stop to resume"}
        )

    def reset_now(self) -> dict:
        """Clear the e-stop (reset_stop tool, or SIGUSR1 -- the staff
        channel when reset_stop is hidden from attendees)."""
        self._stop_pending = False
        rt = self._runtime
        if rt is None:
            return _text_result({"ok": True, "stopped": False,
                                 "note": "no runtime yet: nothing to clear"})
        with contextlib.redirect_stdout(sys.stderr):
            rt.arm.harness.reset_estop()
            raw = rt.arm.raw
            if hasattr(raw, "resume"):
                raw.resume()
            elif hasattr(raw, "_stopped"):
                raw._stopped = False
        return _text_result({"ok": True, "stopped": False})

    def _reflex_chat_task(self, text: str) -> None:
        """Dashboard chat in MCP mode: tier-1 reflex grammar only, zero LLM
        (the MCP host stays the brain for free-form language). This is the
        booth's host-outage fallback interface -- every cheat-card phrasing
        matches the grammar, so the motion beats survive a dead LLM.
        _exec_lock makes this mutually exclusive with the worker's skill
        execution: two threads must never stream the arm at once."""
        from ..agent.reflex import parse_command

        rt = self._runtime
        if rt is None:
            return
        plan = parse_command(text)
        if plan is None:
            rt.memory.add(
                "note",
                f"chat: {text!r} is not a routine command -- this box is "
                "reflex-only; free-form language goes through the MCP host",
            )
            return
        if not self._exec_lock.acquire(blocking=False):
            rt.memory.add(
                "note",
                "chat: busy -- another command is executing; wait for it "
                "to finish",
            )
            return
        rt.current_task = text
        try:
            # this thread's prints must never reach the protocol stream
            with contextlib.redirect_stdout(sys.stderr):
                for name, args in plan.calls:
                    result = rt.execute(name, args)
                    if not result.get("ok", False):
                        break
            rt.last_path = "reflex"
        finally:
            rt.current_task = None
            self._exec_lock.release()

    def cancel_request(self, request_id) -> None:
        """MCP notifications/cancelled. If the cancelled request is the
        MOTION tool call executing right now, the client (host UI) walked
        away from a moving arm -- freeze it. Non-motion in-flight calls just
        run to completion (their response is discarded client-side)."""
        if request_id is None:
            return
        with self._cancel_lock:
            inflight = self._inflight
            if inflight is None or inflight[0] != request_id:
                # not started yet: mark it so the worker drops it instead
                # of moving an arm nobody is waiting on.
                self._cancelled_ids.add(request_id)
                if len(self._cancelled_ids) > 512:
                    # ids are unique per connection; this many entries means
                    # stale post-completion cancels -- shed them
                    self._cancelled_ids.clear()
                return
        name = inflight[1]
        if name is not None:
            from ..skills.runtime import _MOTION_SKILLS

            if name in _MOTION_SKILLS:
                print(
                    f"[wrc-mcp] client cancelled {name!r} mid-motion -> e-stop",
                    file=sys.stderr,
                )
                self.stop_now()

    # ── tool surface ─────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        from ..skills.runtime import TOOL_SPECS

        dropped = _EXCLUDED_TOOLS | _hidden_tools()
        specs = [t for t in TOOL_SPECS if t["name"] not in dropped]
        specs = specs + [t for t in _EXTRA_TOOLS if t["name"] not in dropped]
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["parameters"],
            }
            for t in specs
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name in _hidden_tools():
            # delisted tools stay callable by a model that memorized them
            # unless the call path rejects too.
            return _text_result(
                {"ok": False,
                 "error": f"tool {name!r} is disabled by the operator "
                          "(WRC_HIDE_TOOLS); ask the booth staff"},
                is_error=True,
            )
        runtime = self._ensure_runtime()
        with contextlib.redirect_stdout(sys.stderr):
            if name == "camera_snapshot":
                return self._camera_snapshot(runtime, (arguments or {}).get("camera"))
            if name == "world_state":
                return _text_result(self._world_state(runtime))
            if name == "live_view_url":
                url = (
                    runtime.stream_server.url
                    if runtime.stream_server is not None else None
                )
                if url:
                    return _text_result({"ok": True, "url": url})
                return _text_result(
                    {"ok": False,
                     "error": "livestream not running: disabled via WRC_STREAM=0 "
                              "or the port was taken at startup (see gateway "
                              "stderr; set WRC_STREAM_PORT to change it)"},
                    is_error=True,
                )
            if name == "emergency_stop":
                # normally intercepted out-of-band by the stdin reader; this
                # branch serves direct/in-process callers with identical
                # semantics (incl. the _stop_pending latch)
                return self.stop_now()
            if name == "reset_stop":
                return self.reset_now()
            with self._exec_lock:  # never overlap with a reflex-chat motion
                if name == "pick_and_place":  # narrate on the dashboard
                    obj = (arguments or {}).get("object", "?")
                    dest = (arguments or {}).get("destination")
                    runtime.current_task = f"pick and place {obj}" + (f" -> {dest}" if dest else "")
                    try:
                        result = runtime.execute(name, arguments or {})
                    finally:
                        runtime.current_task = None
                else:
                    result = runtime.execute(name, arguments or {})
                # after completion, not before: the "via:" chip reports the
                # tier that LAST SERVED a command, never one still running
                runtime.last_path = "mcp-host"
        return _text_result(result, is_error=not result.get("ok", False))

    def _world_state(self, runtime) -> dict:
        from ..apps.demo import _runtime_state

        state = _runtime_state(runtime)
        state["cameras"] = runtime.rig.stats() if getattr(runtime, "rig", None) else {}
        if runtime.stream_server is not None:
            state["live_view_url"] = runtime.stream_server.url
        state["ok"] = True
        return state

    def _camera_snapshot(self, runtime, camera: str | None = None) -> dict:
        rig = getattr(runtime, "rig", None)
        age_s = 0.0
        if camera and rig is not None:
            try:
                stream = rig.get(camera)
            except KeyError as e:
                return _text_result({"ok": False, "error": str(e)}, is_error=True)
            frame = stream.latest()
            if frame is None:
                return _text_result({"ok": False, "error": "no frame yet"}, is_error=True)
            # latest() never blocks -- do not silently serve a pre-glitch
            # frame as if it were live.
            age_s = round(time.monotonic() - frame.t, 1)
            if stream.last_error and age_s > 2.0:
                return _text_result(
                    {"ok": False,
                     "error": f"camera {camera!r} is not delivering frames "
                              f"(last error: {stream.last_error}; newest frame "
                              f"is {age_s}s old)"},
                    is_error=True,
                )
            import cv2

            ok, buf = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg = buf.tobytes() if ok else None
        else:
            frame = runtime.observe()
            jpeg = runtime.frame_jpeg()
        if not jpeg:
            return _text_result({"ok": False, "error": "no frame available"}, is_error=True)
        return {
            "content": [
                {
                    "type": "image",
                    "data": base64.b64encode(jpeg).decode(),
                    "mimeType": "image/jpeg",
                },
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "camera": camera or (runtime.rig.primary.name
                                                 if getattr(runtime, "rig", None) else "primary"),
                            "depth_source": frame.depth_source,
                            "frame_age_s": age_s,
                            "t": time.time(),
                        }
                    ),
                },
            ],
            "isError": False,
        }


def _text_result(payload: dict, is_error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "isError": is_error,
    }


# ── JSON-RPC plumbing ────────────────────────────────────────────────────


def _response(req_id, result=None, error=None) -> dict:
    msg: dict = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def handle_message(server: McpSkillServer, msg: dict) -> dict | None:
    """-> response dict, or None for notifications."""
    method = msg.get("method")
    req_id = msg.get("id")
    is_notification = req_id is None

    try:
        if method == "initialize":
            client_ver = (msg.get("params") or {}).get("protocolVersion", "")
            version = client_ver if client_ver in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return _response(
                req_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return _response(req_id, {})
        if method == "tools/list":
            return _response(req_id, {"tools": server.list_tools()})
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            try:
                return _response(req_id, server.call_tool(name, args))
            except Exception as e:
                return _response(
                    req_id,
                    _text_result(
                        {"ok": False, "error": f"{type(e).__name__}: {e}"}, is_error=True
                    ),
                )
        if is_notification:
            return None
        return _response(req_id, error={"code": -32601, "message": f"method not found: {method}"})
    except Exception as e:  # never kill the server on one bad message
        if is_notification:
            return None
        return _response(req_id, error={"code": -32603, "message": f"internal error: {e}"})


def main() -> int:
    import queue
    import signal

    server = McpSkillServer()
    # The prewarm thread wraps its build in redirect_stdout(sys.stderr),
    # which swaps the PROCESS-GLOBAL sys.stdout. Protocol frames must go
    # through a reference captured before that thread starts, or the
    # initialize/tools/list responses land on stderr and the client hangs.
    protocol_out = sys.stdout
    out_lock = threading.Lock()  # reader + worker both write frames now

    def _send(resp: dict) -> None:
        with out_lock:
            protocol_out.write(json.dumps(resp) + "\n")
            protocol_out.flush()

    print("[wrc-mcp] wrc-demo MCP server on stdio", file=sys.stderr)
    if os.environ.get("WRC_PREWARM", "1") != "0":
        server.prewarm_async()

    # First SIGINT freezes (latch e-stop), second exits. Exit disables
    # torque on the RS arm and a loaded arm falls -- same contract as the
    # demo CLI, and the reason SIGINT must never be the routine stop path.
    # Stop FIRST, then print: stderr can raise on reentrant use inside a
    # signal handler and must never cost the latch.
    def _sigint(_sig, _frm):
        server.stop_now()
        signal.signal(signal.SIGINT, signal.default_int_handler)
        with contextlib.suppress(Exception):
            print("\n[wrc-mcp] SIGINT: e-stop latched (Ctrl+C again to exit; "
                  "exit disables torque -- park the arm first)", file=sys.stderr)

    signal.signal(signal.SIGINT, _sigint)

    # SIGUSR1 clears the e-stop: the staff reset channel when reset_stop is
    # hidden from the attendee session (shell access to the rig == staff).
    def _sigusr1(_sig, _frm):
        server.reset_now()
        with contextlib.suppress(Exception):
            print("[wrc-mcp] SIGUSR1: e-stop cleared by staff", file=sys.stderr)

    signal.signal(signal.SIGUSR1, _sigusr1)

    inbox: queue.Queue = queue.Queue()
    _EOF = object()

    def _read_stdin() -> None:
        """Feed the worker; short-circuit anything that must not queue
        behind a running tool call (see "Stop channel" in the docstring).
        This thread IS the stop channel: no single frame may kill it."""
        try:
            for line in sys.stdin:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"[wrc-mcp] bad JSON frame: {line[:120]}",
                              file=sys.stderr)
                        continue
                    # some clients batch JSON-RPC frames; unwrap them
                    for m in (msg if isinstance(msg, list) else [msg]):
                        if not isinstance(m, dict):
                            print(f"[wrc-mcp] non-object frame skipped: "
                                  f"{str(m)[:120]}", file=sys.stderr)
                            continue
                        method = m.get("method")
                        params = m.get("params") or {}
                        if (method == "tools/call"
                                and params.get("name") == "emergency_stop"):
                            result = server.stop_now()
                            if m.get("id") is not None:
                                _send(_response(m["id"], result))
                            continue
                        if method == "notifications/cancelled":
                            server.cancel_request(params.get("requestId"))
                            continue
                        if method == "ping" and m.get("id") is not None:
                            # host keepalives must not starve behind a motion
                            _send(_response(m["id"], {}))
                            continue
                        inbox.put(m)
                except Exception as e:
                    print(f"[wrc-mcp] reader error (frame skipped): {e}",
                          file=sys.stderr)
        finally:
            inbox.put(_EOF)

    threading.Thread(target=_read_stdin, daemon=True, name="wrc-stdin").start()
    try:
        while True:
            msg = inbox.get()
            if msg is _EOF:
                break
            req_id = msg.get("id")
            is_call = msg.get("method") == "tools/call"
            with server._cancel_lock:
                cancelled = req_id is not None and req_id in server._cancelled_ids
                if cancelled:
                    server._cancelled_ids.discard(req_id)
                else:
                    server._inflight = (
                        req_id,
                        (msg.get("params") or {}).get("name") if is_call else None,
                    )
            if cancelled:
                continue  # client gave up before we started; never move
            try:
                resp = handle_message(server, msg)
            finally:
                with server._cancel_lock:
                    server._inflight = None
                    if req_id is not None:
                        # a cancel that raced with completion is spent now
                        server._cancelled_ids.discard(req_id)
            if resp is not None:
                _send(resp)
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
