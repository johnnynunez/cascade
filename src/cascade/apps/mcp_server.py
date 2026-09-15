"""MCP stdio server: expose the arm's skill runtime to any MCP agent.

This turns the demo into a tool surface for agent platforms -- Hermes
(~/.hermes/config.yaml), Claude Code (.mcp.json), Claude Desktop, Codex CLI.
The platform's agent replaces cascade's built-in orchestrator as the brain;
the safety harness, tracing, episodic memory and skills are identical.

Protocol: MCP over stdio, newline-delimited JSON-RPC 2.0. Implemented by
hand (initialize / tools/list / tools/call / ping) so the demo gains no new
dependency. stdout carries ONLY protocol frames; everything else goes to
stderr.

Configuration via environment (set in the MCP server entry):
    CASCADE_CAMERA          camera profile (default: mock)
    CASCADE_CAMERAS         comma-separated camera profiles; first one is the
                        manipulation camera (overrides CASCADE_CAMERA)
    CASCADE_ARM             arm profile    (default: mock)
    CASCADE_RUN_DIR         trace directory (default: <repo>/runs/mcp_<pid>)
    CASCADE_DETECTOR_MODEL  override detector weights (e.g. a yolo11n.pt path
                        for closed-set COCO until the CLIP fork is installed)
    CASCADE_DETECT_CLASSES  comma-separated default vocabulary for observations
    CASCADE_VIEW            "0" disables the live camera window (default: open it
                        whenever DISPLAY is set, so the audience always sees
                        what the camera sees)
    CASCADE_PREWARM         "0" disables perception pre-warm at startup
                        (default: cameras + detector + world model come up
                        immediately so the first command is fast)
    CASCADE_STREAM          "0" disables the MJPEG livestream dashboard
    CASCADE_STREAM_PORT     dashboard port (default: from configs/demo.yaml)
    CASCADE_HIDE_TOOLS      comma-separated tool names to remove from the agent's
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
(`pkill -USR1 -f cascade.apps.mcp_server` requires shell access to the
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
SERVER_INFO = {"name": "cascade", "version": "0.1.0"}

# task_done belongs to the built-in loop; MCP agents manage their own tasks.
_EXCLUDED_TOOLS = {"task_done"}


def _hidden_tools() -> set[str]:
    """Operator-hidden tools (CASCADE_HIDE_TOOLS), read per call so tests can
    vary it per subprocess. emergency_stop is never hideable: an operator
    typo must not be able to remove the stop path."""
    hidden = {
        t.strip()
        for t in os.environ.get("CASCADE_HIDE_TOOLS", "").split(",")
        if t.strip()
    }
    hidden.discard("emergency_stop")
    return hidden


def _robot_state_provenance(state: dict) -> dict:
    """Session tracking does not measure current gripper contents or home pose."""
    result = dict(state)
    held = result.pop("holding", None)
    if held is not None:
        result["tracked_holding"] = {"label": held, "source": "session_state"}
    result["gripper_contents"] = "not_measured"
    result["home_pose"] = "not_verified"
    return result


_EXTRA_TOOLS = [
    {
        "name": "camera_snapshot",
        "description": (
            "Capture a camera frame and return it as an image, with the "
            "depth source noted. Use this to SEE the workspace. Optional "
            "`camera` selects a rig camera by exact name. When the user "
            "provides a camera name, call this tool directly: it validates "
            "availability and reports any unavailable-camera error. A prior "
            "world_state call is not required. Use world_state to discover "
            "names only when the request does not specify a camera."
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
            "Read tracked objects, MCP session state and cached camera statistics. "
            "session_state.agent_status is last activity text, not a startup or "
            "health check. arm_connected means this session opened its control "
            "connection; false is normal before first control use. Perception "
            "counters describe this session's detection and belief updates. This "
            "does not query simulator health, arm pose or gripper contents. "
            "tracked_holding is session history, not a current contact measurement. "
            "Object labels and positions may be remembered: use get_observation "
            "for the visible scene and localize_object for measured positions."
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
    {
        "name": "robot_knowledge",
        "description": (
            "What this robot has LEARNED from past runs: the proven operating "
            "range of each primitive (where grasps/places actually succeed on "
            "this arm), how each one usually fails, and the grasp strategy "
            "priors per object profile. Read this before planning a tricky "
            "manipulation -- it is the difference between guessing the arm's "
            "envelope and knowing it. Costs no image tokens."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "verify_last_action",
        "description": (
            "Read saved postcondition verdicts for recent actions, including "
            "their evidence and channel. This does not perform a fresh sensor "
            "check or upgrade an unverified result. A belief position written "
            "by the skill does not independently confirm placement. Explain "
            "confirmed, refuted or unverified using the recorded channel; use "
            "get_observation for a new visual inspection."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "task_memory",
        "description": (
            "Your visual memory of the current task: up to K captioned frames "
            "-- the scene BEFORE the first action, then what it looked like "
            "after each action, each with the independent verdict on that "
            "action (confirmed / refuted / unverified) -- and a fresh current "
            "view last when the camera is available. Call it before deciding the next step of any task "
            "with more than one action (counting, sorting, 'put N objects "
            "in', 'find without repeating'): judge what is DONE from these "
            "frames and verdicts, never from what you intended. Pass "
            "`new_task: true` on the first call of a new task to start a "
            "fresh memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "description": "Max past frames (default 4)."},
                "new_task": {
                    "type": "boolean",
                    "description": "Start a new episode: forget the previous task's frames first.",
                },
            },
            "required": [],
        },
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
                    "CASCADE_CAMERAS", os.environ.get("CASCADE_CAMERA", "mock")
                ).split(",")
                if c.strip()
            ]
            arm = os.environ.get("CASCADE_ARM", "mock")
            run_dir = Path(
                os.environ.get("CASCADE_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}")
            )
            try:
                # Anything the stack prints must not corrupt the protocol stream.
                with contextlib.redirect_stdout(sys.stderr):
                    cfg = load_demo_config(cameras=cameras, arm=arm, llm="mock")
                    det_model = os.environ.get("CASCADE_DETECTOR_MODEL")
                    if det_model:
                        cfg._data["detector"]["model"] = det_model
                    classes = os.environ.get("CASCADE_DETECT_CLASSES")
                    if classes:
                        cfg._data["detect_classes"] = [
                            c.strip() for c in classes.split(",") if c.strip()
                        ]
                    port = os.environ.get("CASCADE_STREAM_PORT")
                    if port:
                        cfg._data.setdefault("stream", {})["port"] = int(port)
                    view = os.environ.get("CASCADE_VIEW", "1") != "0" and bool(
                        os.environ.get("DISPLAY")
                    )
                    serve = os.environ.get("CASCADE_STREAM", "1") != "0"
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
                    f"[cascade-mcp] runtime up: cameras={cameras} arm={arm} (lazy) "
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
                print(f"[cascade-mcp] prewarm failed (will retry on first call): {e}",
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
            print("[cascade-mcp] EMERGENCY STOP latched (runtime still starting)",
                  file=sys.stderr)
            return _text_result(
                {"ok": True, "stopped": True,
                 "note": "stop latched; motors were never powered and any "
                         "runtime that finishes starting comes up stopped -- "
                         "staff clears it via reset_stop / SIGUSR1"}
            )
        with contextlib.redirect_stdout(sys.stderr):
            rt.arm.stop()
        print("[cascade-mcp] EMERGENCY STOP latched (out-of-band)", file=sys.stderr)
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
                rt.current_tier = str(getattr(plan, "source", "reflex"))
                try:
                    for name, args in plan.calls:
                        result = rt.execute(name, args)
                        if not result.get("ok", False):
                            break
                finally:
                    rt.current_tier = None
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
                    f"[cascade-mcp] client cancelled {name!r} mid-motion -> e-stop. "
                    "If this arrived at a round number of seconds the HOST's "
                    "per-call budget expired (OpenClaw requestTimeoutMs, default "
                    "60 s) -- a persistent pick legitimately runs longer; raise "
                    "it in the server entry (launch.sh sets 300000).",
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
                          "(CASCADE_HIDE_TOOLS); ask the booth staff"},
                is_error=True,
            )
        runtime = self._ensure_runtime()
        with contextlib.redirect_stdout(sys.stderr):
            if name == "camera_snapshot":
                return self._camera_snapshot(runtime, (arguments or {}).get("camera"))
            if name == "world_state":
                return _text_result(self._world_state(runtime))
            if name == "live_view_url":
                # The dashboard is lazy now: asking for the URL is itself a
                # request to look, so open it rather than reporting "not
                # running". Idempotent when already open.
                lv = getattr(runtime, "live_view", None)
                if lv is None:
                    url = (
                        runtime.stream_server.url
                        if runtime.stream_server is not None else None
                    )
                    if url:
                        return _text_result({"ok": True, "url": url})
                    return _text_result(
                        {"ok": False,
                         "error": "livestream not running: disabled via CASCADE_STREAM=0 "
                                  "or the port was taken at startup (see gateway "
                                  "stderr; set CASCADE_STREAM_PORT to change it)"},
                        is_error=True,
                    )
                out = lv.open(reason="live_view_url requested")
                runtime.stream_server = lv.server
                if not out.get("ok"):
                    return _text_result(out, is_error=True)
                return _text_result(out)
            if name == "emergency_stop":
                # normally intercepted out-of-band by the stdin reader; this
                # branch serves direct/in-process callers with identical
                # semantics (incl. the _stop_pending latch)
                return self.stop_now()
            if name == "reset_stop":
                return self.reset_now()
            if name == "robot_knowledge":
                return _text_result(self._robot_knowledge(runtime))
            if name == "verify_last_action":
                return _text_result(self._verify_last(runtime))
            if name == "task_memory":
                return self._task_memory(runtime, arguments or {})
            with self._exec_lock:  # never overlap with a reflex-chat motion
                runtime.current_tier = "mcp-host"  # the chat host's brain chose this call
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
                runtime.current_tier = None
                # after completion, not before: the "via:" chip reports the
                # tier that LAST SERVED a command, never one still running
                runtime.last_path = "mcp-host"
                if name in {"get_observation", "describe_scene"} and result.get("ok"):
                    return self._image_result(runtime, runtime.last_frame, result)
        return _text_result(result, is_error=not result.get("ok", False))

    def _robot_knowledge(self, runtime) -> dict:
        """Everything the robot has learned from past runs, as text.

        Deliberately cheap and image-free: this is the tool a Hermes/Claude
        host should read BEFORE planning, so it starts from the arm's proven
        envelope instead of rediscovering it one SafetyViolation at a time.
        """
        out: dict = {"ok": True}
        try:
            out["primitive_envelopes"] = runtime.envelope.envelope_digest() or "(nothing learned yet)"
            out["failure_models"] = runtime.envelope.failure_digest() or "(no failures recorded)"
            out["stats"] = runtime.envelope.stats()
        except Exception as e:
            out["primitive_envelopes"] = f"unavailable: {e}"
        try:
            out["grasp_priors"] = runtime.grasp_memory.agent_digest() or "(no grasp history)"
        except Exception:
            pass
        return out

    def _task_memory(self, runtime, arguments: dict) -> dict:
        """Vesta memory harness (arXiv:2606.20905 §2.4) for a chat host.

        The orchestrator's LLM tier gets the same frames injected on every
        turn; a chat host cannot be injected into, so it gets them as a
        tool: image content items interleaved with one caption each, the
        current view last. Same sampler, same captions, same verdicts."""
        mem = runtime.memory
        if arguments.get("new_task"):
            mem.reset_frames()
        try:
            k = int(arguments.get("k") or 4)
        except (TypeError, ValueError):
            k = 4
        frames = mem.memory_frames(max(0, k))
        content: list[dict] = []
        for fr in frames:
            content.append({"type": "text", "text": mem.frame_caption(fr)})
            content.append({
                "type": "image",
                "data": base64.b64encode(fr["jpeg"]).decode(),
                "mimeType": "image/jpeg",
            })
        snapshot = self._camera_snapshot(runtime)
        current = not snapshot["isError"]
        current_view = {"available": current,
                        **json.loads(snapshot["content"][-1]["text"])}
        if current:
            content.append({"type": "text", "text": "current view (now)"})
            content.extend(snapshot["content"])
        summary = {
            "ok": True,
            "current_view": current_view,
            "frames": len(frames),
            "steps_recorded": [
                {"step": fr["step"], "age_s": fr["age_s"], "action": fr["text"],
                 "verdict": fr["verdict"] or "none"}
                for fr in frames
            ],
            "note": (
                "Judge progress from the frames and verdicts above. A step "
                "marked refuted did not happen."
                if frames else
                "No actions recorded yet in this task."
            ) + (" Fresh current view attached." if current else
                 " Current camera view unavailable; saved frames are history only."),
        }
        content.append({"type": "text", "text": json.dumps(summary)})
        return {"content": content, "isError": False}

    def _verify_last(self, runtime) -> dict:
        """Independent verification of recent effects (Pigey postconditions)."""
        checker = getattr(runtime, "effects", None)
        if checker is None:
            return {
                "ok": True,
                "verification": "disabled",
                "note": "effect verification is off (config verify_effects: false)",
            }
        history = checker.history[-5:]
        return {
            "ok": True,
            "digest": checker.digest() or "(no verifiable actions yet)",
            "recent": [pc.as_dict() for pc in history],
            "contradictions": [pc.as_dict() for pc in checker.contradictions()[-3:]],
        }

    def _world_state(self, runtime) -> dict:
        from ..apps.demo import _runtime_state

        state = _robot_state_provenance(_runtime_state(runtime))
        state["session_state"] = {
            "source": "mcp_runtime",
            **{key: state.pop(key) for key in ("agent_status", "arm_connected", "perception")
               if key in state},
        }
        state["live_arm_feedback"] = False
        state["simulator_status"] = "not_queried"
        state["cameras"] = runtime.rig.stats() if getattr(runtime, "rig", None) else {}
        state["camera_statistics_note"] = (
            "Cached stream statistics: frame_id counts client deliveries; FPS is "
            "recent observed capture cadence. A single counter or zero FPS does "
            "not establish simulator health. Use a fresh image for current visibility."
        )
        if runtime.stream_server is not None:
            state["live_view_url"] = runtime.stream_server.url
        state["ok"] = True
        return state

    def _camera_snapshot(self, runtime, camera: str | None = None) -> dict:
        from ..skills.runtime import _fresh_camera_frame

        with self._exec_lock:
            try:
                rig = getattr(runtime, "rig", None)
                frame = (_fresh_camera_frame(rig.get(camera)) if camera and rig is not None
                         else runtime.observe_fresh())
            except Exception as exc:
                return _text_result({"ok": False, "error": str(exc)}, is_error=True)
            return self._image_result(runtime, frame, camera=camera)

    def _image_result(self, runtime, frame, payload=None, camera=None) -> dict:
        from ..skills.runtime import _frame_age_s, _jpeg

        try:
            age_s = _frame_age_s(frame)
            jpeg = _jpeg(frame.rgb)
            if not jpeg:
                raise ValueError("Camera frame could not be encoded")
        except Exception as exc:
            return _text_result({"ok": False, "error": str(exc)}, is_error=True)
        rig = getattr(runtime, "rig", None)
        metadata = {"camera": camera or (rig.primary.name if rig is not None else "primary"),
                    "depth_source": frame.depth_source, "frame_id": int(frame.frame_id),
                    "frame_age_s": age_s, "age_clock": "client_monotonic", "t": time.time()}
        result = metadata
        if payload is not None:
            robot = _robot_state_provenance(payload.get("robot", {}))
            if robot.get("live_arm_feedback") is False:
                robot.pop("status", None)
                robot["pose_status"] = "not_measured"
            result = {
                "ok": payload.get("ok", False),
                "observation_frame": metadata,
                "configured_zones": payload.get("configured_zones", []),
                "robot": robot,
                "observation_note": (
                    "Identify visible objects, colors and relations from the attached image; "
                    "state uncertainty when unclear. Configured zones are fixed destinations, "
                    "not evidence of visibility or occupancy. Use localize_object when a "
                    "measured object position is needed."
                ),
            }
        return {
            "content": [
                {
                    "type": "image",
                    "data": base64.b64encode(jpeg).decode(),
                    "mimeType": "image/jpeg",
                },
                {
                    "type": "text",
                    "text": json.dumps(result),
                },
            ],
            "isError": False,
        }


class _Tee:
    """Write-through to two streams (stderr + server.log); never raises."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for st in self._streams:
            try:
                st.write(data)
            except Exception:  # noqa: BLE001
                pass
        return len(data)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def _short_args(args: dict) -> str:
    try:
        return ", ".join(f"{k}={str(v)[:24]}" for k, v in (args or {}).items())
    except Exception:  # noqa: BLE001
        return "?"


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
            # One stderr line per tool call with the outcome. The chat host
            # (OpenClaw `agent exec`) reports only a failure COUNT and keeps
            # no transcript for isolated runs, so without this line a
            # "failures: 2" next to a physics-confirmed pick is undebuggable.
            t_call = time.monotonic()
            try:
                out = server.call_tool(name, args)
            except Exception as e:
                out = _text_result({"ok": False, "error": f"{type(e).__name__}: {e}"}, is_error=True)
            try:
                # Multi-part results (task_memory: captions + images + a JSON
                # summary LAST) log their summary, not their first caption --
                # a log that showed only "memory frame 1 ..." read as "the
                # memory never grew" while frames 2..k were right there.
                parts = [c for c in (out.get("content") or []) if c.get("type") == "text"]
                n_img = sum(1 for c in (out.get("content") or []) if c.get("type") == "image")
                body = (parts[-1].get("text", "") if parts else "")
                if n_img:
                    body = f"[{n_img} image(s)] " + body
                print(f"[cascade-mcp] tools/call {name}({_short_args(args)}) -> "
                      f"{'ERROR' if out.get('isError') else 'ok'} in {time.monotonic() - t_call:.1f}s: {body[:200]}",
                      file=sys.stderr, flush=True)
            except Exception:  # noqa: BLE001 -- logging must never fail a call
                pass
            return _response(req_id, out)
        if is_notification:
            return None
        return _response(req_id, error={"code": -32601, "message": f"method not found: {method}"})
    except Exception as e:  # never kill the server on one bad message
        if is_notification:
            return None
        return _response(req_id, error={"code": -32603, "message": f"internal error: {e}"})


def main() -> int:
    import argparse
    import queue
    import signal

    parser = argparse.ArgumentParser(description="CASCADE stdio MCP server")
    parser.add_argument("--launch-owner")
    parser.add_argument("--launch-state-dir", type=Path)
    args = parser.parse_args()
    if bool(args.launch_owner) != bool(args.launch_state_dir):
        parser.error("--launch-owner and --launch-state-dir must be supplied together")
    if args.launch_owner:
        from .process_owner import load_owner, register_process
        from ..config import PACKAGE_ROOT
        import uuid

        owner = load_owner(args.launch_state_dir, PACKAGE_ROOT, os.environ.get("CASCADE_OPENCLAW_PROFILE", ""))
        if owner is None or owner["owner"] != args.launch_owner:
            parser.error("launch owner does not match this repo/state/profile")
        # A fresh process always owns a fresh trace, even after PID reuse or
        # when its parent happens to export an old CASCADE_RUN_DIR.
        run_dir = PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}_{uuid.uuid4().hex}"
        os.environ["CASCADE_RUN_DIR"] = str(run_dir)
        register_process(args.launch_state_dir, owner, os.getpid(), "mcp", run_dir=run_dir)

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

    # Mirror stderr to a file. A chat host swallows an MCP child's stderr
    # (OpenClaw shows only a failure count), so on a shared machine the only
    # way to answer "why did that tool call fail?" is this log. Same folder
    # as the run's trace.jsonl / keyframes.
    from ..config import PACKAGE_ROOT

    run_dir = Path(os.environ.get("CASCADE_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}"))
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        sys.stderr = _Tee(sys.stderr, open(run_dir / "server.log", "a", buffering=1))
    except Exception:  # noqa: BLE001 -- a read-only checkout must not kill the server
        pass
    print(f"[cascade-mcp] cascade MCP server on stdio (log: {run_dir / 'server.log'})", file=sys.stderr)
    if os.environ.get("CASCADE_PREWARM", "1") != "0":
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
            print("\n[cascade-mcp] SIGINT: e-stop latched (Ctrl+C again to exit; "
                  "exit disables torque -- park the arm first)", file=sys.stderr)

    signal.signal(signal.SIGINT, _sigint)

    # SIGUSR1 clears the e-stop: the staff reset channel when reset_stop is
    # hidden from the attendee session (shell access to the rig == staff).
    def _sigusr1(_sig, _frm):
        server.reset_now()
        with contextlib.suppress(Exception):
            print("[cascade-mcp] SIGUSR1: e-stop cleared by staff", file=sys.stderr)

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
                        print(f"[cascade-mcp] bad JSON frame: {line[:120]}",
                              file=sys.stderr)
                        continue
                    # some clients batch JSON-RPC frames; unwrap them
                    for m in (msg if isinstance(msg, list) else [msg]):
                        if not isinstance(m, dict):
                            print(f"[cascade-mcp] non-object frame skipped: "
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
                    print(f"[cascade-mcp] reader error (frame skipped): {e}",
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
