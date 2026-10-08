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
    CASCADE_ARMS            comma-separated arm profiles; first one is the
                        manipulation arm (overrides CASCADE_ARM; builds the
                        ArmRig exactly like the CLI's --arms)
    CASCADE_RUN_DIR         trace directory (default: <repo>/runs/mcp_<pid>)
    CASCADE_GRASP_EVIDENCE_DIR  opt-in per-attempt JSON + NPZ evidence directory
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
                        This is the explicit OPERATOR override; the rig's own
                        limits are applied by the capability matrix below.

Tool surface (what the model is offered): `TOOL_SPECS` + `_EXTRA_TOOLS`,
minus `_EXCLUDED_TOOLS` (loop-internal `task_done`), minus the operator's
`CASCADE_HIDE_TOOLS`, minus the tools whose preconditions THIS rig cannot
meet. That last set comes from the capability matrix (`apps/capabilities.py`),
evaluated from the BUILT runtime's probed state -- the camera streams' depth
chain, the sidecar probes behind `runtime.backends()`, the ArmRig, the
verifier, the memory -- never from a profile's promise: on an RGB-only
camera the 3D tools are withheld, on a single-arm rig `list_arms` and the
injected `arm` parameter go away, a GraspGen-X that is down is a REPORTED
fallback (OBB), not a reason to hide anything. Before the runtime exists
nothing is probed and nothing is withheld; once it is, a catalog that was
already served is refreshed via `notifications/tools/list_changed`, and a
withheld tool is rejected if called anyway. Every withheld tool carries its
reason in `world_state.tools_withheld`, the dashboard `/state` and the
`[cascade-mcp] tools withheld` log line -- a hidden tool is never silent.

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
import copy
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
            "for the visible scene and localize_object for measured positions. "
            "`capabilities` is what THIS rig can do as probed at startup (depth "
            "chain per camera, sidecars, arms, verifier, memory); `tools_withheld` "
            "names the tools not offered because a precondition is unmet, with "
            "the reason, and `tools_hidden_by_operator` the ones the operator "
            "removed. Do not ask for a withheld tool; work with what is listed."
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
        # Morphology is selected once for this connection; catalog and dispatch
        # must agree even before a (possibly slow) runtime exists.
        self._robot_profile = os.environ.get("CASCADE_ROBOT")
        self._composed = self._robot_profile is not None
        self._base_profile = os.environ.get("CASCADE_BASE")
        if self._composed and self._base_profile is not None:
            raise ValueError("CASCADE_ROBOT and CASCADE_BASE are mutually exclusive")
        self._mobile = self._base_profile is not None
        self._bounded = self._mobile or self._composed
        self._mobile_config = None
        self._input_closed = False
        self._stop_serial = 0
        self._stop_lock = threading.Lock()
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
        # Capability matrix -> tool surface (ROADMAP #12). The catalog served
        # before the runtime exists is the full one (nothing probed, nothing
        # withheld); once the build has trimmed it, a host that already
        # listed is told via notifications/tools/list_changed (hook set by
        # _serve_stdio; None for in-process callers) and re-fetches.
        self._tools_changed_fn = None
        self._catalog_served = False
        self._published_surface: tuple | None = None
        # list_tools (worker) and _announce_surface (build thread) publish
        # and compare the surface under this lock, so a catalog listed
        # while the build finishes is always either probed or notified.
        self._surface_lock = threading.Lock()

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
            arms = [
                a.strip()
                for a in os.environ.get(
                    "CASCADE_ARMS", os.environ.get("CASCADE_ARM", "mock")
                ).split(",")
                if a.strip()
            ]
            run_dir = Path(
                os.environ.get("CASCADE_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}")
            )
            try:
                # Anything the stack prints must not corrupt the protocol stream.
                with contextlib.redirect_stdout(sys.stderr):
                    cfg = (self._get_mobile_config() if self._bounded else
                           load_demo_config(cameras=cameras, arms=arms, llm="mock"))
                    det_model = os.environ.get("CASCADE_DETECTOR_MODEL")
                    if det_model and not self._bounded:
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
                    signals = (getattr(self, "_signals", None)
                               if threading.current_thread() is threading.main_thread() else None)
                    with signals.defer() if signals is not None else contextlib.nullcontext():
                        self._runtime, self._arm = build_runtime(
                            cfg, run_dir, view=view, lazy_arm=True, serve=serve
                        )
                    if signals is not None and threading.current_thread() is threading.main_thread():
                        signals.checkpoint()
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
                        if self._composed and self._input_closed:
                            self._runtime.request_shutdown()
                        elif self._bounded:
                            self._runtime.stop()
                        else:
                            self._runtime.arm.stop()
                url = (
                    self._runtime.stream_server.url
                    if self._runtime.stream_server is not None else "disabled"
                )
                if self._bounded:
                    names = (self._runtime.cfg.robot_id if self._composed else
                             ",".join(p["name"] for p in self._runtime.cfg.bases))
                    print(f"[cascade-mcp] mobile runtime up: bases={names}", file=sys.stderr)
                else:
                    print(
                        f"[cascade-mcp] runtime up: cameras={cameras} arms={arms} (lazy) "
                        f"livestream={url}",
                        file=sys.stderr,
                    )
                    # the matrix and the tools it withholds, next to the
                    # `backends:` line build_runtime just printed -- and a
                    # host that already listed tools is told to re-fetch
                    self._announce_surface()
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
        from ..lifecycle import retain_teardown_attempt, teardown_receipt, teardown_step

        previous = getattr(self, "_shutdown_receipt", None)
        retry = (isinstance(previous, dict)
                 and getattr(self._runtime, "robot_mode", None) in {"mobile", "composed"}
                 and any(stage.get("stage") == "runtime" and stage.get("complete") is not True
                         for stage in previous.get("stages", ())))
        if isinstance(previous, dict) and not retry:
            return copy.deepcopy(previous)
        stages = ([copy.deepcopy(stage) for stage in previous["stages"]
                   if stage.get("stage") == "stop"] if retry else [])

        def stop():
            if self._composed:
                return self._request_composed_shutdown()
            if self._mobile:
                # stop_now returns a protocol envelope. Its existing stop
                # semantics stay intact; runtime.close owns the final receipt.
                self.stop_now()

        if not retry:
            stages.append(teardown_step("stop", stop))
        # taking _init_lock waits out an in-flight prewarm build, so a
        # runtime that finishes building after EOF is still torn down
        with self._init_lock, contextlib.redirect_stdout(sys.stderr):
            if self._runtime is not None:
                from ..apps.demo import shutdown_runtime

                stages.append(teardown_step("runtime", lambda: shutdown_runtime(self._runtime, self._arm)))
        receipt = teardown_receipt(stages)
        receipt.update(pid=os.getpid(), runtime_built=self._runtime is not None)
        receipt = retain_teardown_attempt(previous, receipt)
        try:
            from .process_owner import _write_json
            from ..config import PACKAGE_ROOT

            run_dir = Path(os.environ.get("CASCADE_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}"))
            _write_json(run_dir / "teardown.json", receipt)
        except Exception as error:
            receipt["stages"].append({"stage": "persist_receipt", "ok": False, "complete": False,
                                      "errors": [{"type": type(error).__name__, "message": str(error)}]})
            receipt.update(ok=False, complete=False)
            print(f"[cascade-mcp] teardown receipt could not be saved: {receipt}", file=sys.stderr)
        self._shutdown_receipt = copy.deepcopy(receipt)
        return receipt

    # ── out-of-band stop/reset (called from the stdin thread / signals) ──

    def stop_now(self, *, navigation=False) -> dict:
        """Latch the e-stop from any thread, without ever building the
        runtime. Ordering matters: _stop_pending is set BEFORE reading
        _runtime, and _ensure_runtime checks it AFTER assigning _runtime,
        so a stop concurrent with the build is caught by one side or the
        other in every interleaving."""
        with self._stop_lock:
            self._stop_pending = True
            self._stop_serial += 1
            stop_serial = self._stop_serial
        rt = self._runtime
        if self._bounded:
            if rt is None:
                import uuid
                return _text_result({"ok": True, "stopped": True, "latched": True,
                                     "receipt_id": uuid.uuid4().hex, "serial": stop_serial,
                                     "receipt_scope": "mcp_startup_local", "physical_stop_verified": False,
                                     "outcome": "unverified", "note": "Local startup stop latched; no physical rest claim."})
            # No runtime build / execution lock; the runtime invalidates a
            # pending lazy connect as well as an already-admitted command.
            result = rt.execute("stop_navigation" if navigation else "emergency_stop", {})
            return _text_result(result, is_error=not result.get("ok", False))
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

    def reset_now(self, arguments=None) -> dict:
        """Clear the e-stop (reset_stop tool, or SIGUSR1 -- the staff
        channel when reset_stop is hidden from attendees).

        Whole-body robots reset one named domain per call; the server-level
        stop latch clears only once no domain remains latched (SIGUSR1 names
        no domain and is therefore refused for them)."""
        if self._bounded:
            rt = self._runtime
            if rt is None or self._input_closed:
                return _text_result({"ok": False, "error": "runtime starting or input closed; reset refused"}, is_error=True)
            with self._stop_lock:
                serial = self._stop_serial
            result = rt.execute("reset_stop", dict(arguments or {}))
            with self._stop_lock:
                raced = serial != self._stop_serial
                if result.get("ok") is True and not raced and not result.get("latched_domains"):
                    self._stop_pending = False
            if raced:
                rt.stop()
                result = {"ok": False, "latched": True, "error": "stop raced reset"}
            return _text_result(result, is_error=not result.get("ok", False))
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
            if self._bounded:
                motion_skills = self._motion_tools()
            else:
                from ..skills.runtime import _MOTION_SKILLS as motion_skills

            if name in motion_skills:
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

    def _get_mobile_config(self):
        if self._mobile_config is None:
            from ..config import load_demo_config

            self._mobile_config = load_demo_config(base=self._base_profile, robot=self._robot_profile, llm="mock")
        return self._mobile_config

    def _motion_tools(self):
        if self._composed:
            from .robot_runtime import robot_tool_descriptors
            return frozenset(n for n, t in robot_tool_descriptors(self._get_mobile_config()).items()
                             if t.effect == "motion")
        from ..skills.mobile_runtime import MOTION_SKILLS
        return MOTION_SKILLS

    def _stop_tools(self):
        if self._composed:
            from .robot_runtime import robot_tool_descriptors
            return frozenset(n for n, t in robot_tool_descriptors(self._get_mobile_config()).items()
                             if t.effect == "stop")
        return frozenset({"emergency_stop", "stop_navigation"} if self._mobile else {"emergency_stop"})

    # ── capability matrix (ROADMAP #12) ──────────────────────────────────

    def capabilities(self) -> dict:
        """What this rig can do, from the BUILT runtime's probed state; the
        unprobed matrix while it is still building. Read-only and cheap
        (never grabs a frame, never touches the lazy arm). Bounded modes
        derive their catalog from the base/robot profile already and are
        not trimmed by this."""
        from .capabilities import capability_matrix

        rt = self._runtime
        return capability_matrix(rt if (rt is not None and not self._bounded) else None)

    def withheld_tools(self) -> dict[str, str]:
        """tool -> reason, for the tools this rig's probed state shows it
        cannot run. Empty until the runtime exists: unprobed is not
        unavailable."""
        from .capabilities import withheld_tools

        return withheld_tools(self.capabilities())

    def _single_arm(self) -> bool:
        """True once the built ArmRig shows exactly one arm, so the `arm`
        parameter injected over _MOTION_SKILLS (a schema entry the chat model
        fills with "" on every call) can be dropped from the served schemas.
        False while unprobed: the full schema is the honest default."""
        from .capabilities import CAP_MULTI_ARM

        matrix = self.capabilities()
        return bool(matrix.get("probed")) and (matrix.get(CAP_MULTI_ARM) or {}).get("available") is False

    def _surface_signature(self) -> tuple:
        return (tuple(sorted(self.withheld_tools())), self._single_arm())

    def _announce_surface(self) -> None:
        """Log the withheld tools (server.log + stderr) next to the
        `[cascade] capabilities:` banner build_runtime just printed, and if a
        catalog was already served with a different surface, tell the host
        to re-fetch it. Reporting must never fail the build."""
        try:
            withheld = self.withheld_tools()
            if withheld:
                by_reason: dict[str, list[str]] = {}
                for name, why in sorted(withheld.items()):
                    by_reason.setdefault(why, []).append(name)
                print("[cascade-mcp] tools withheld by the capability matrix: "
                      + "; ".join(f"{', '.join(names)} ({why})" for why, names in by_reason.items()),
                      file=sys.stderr)
            else:
                print("[cascade-mcp] tools withheld by the capability matrix: none",
                      file=sys.stderr)
            if self._single_arm():
                print("[cascade-mcp] single-arm rig: the `arm` parameter is dropped from the "
                      "motion tools' schemas", file=sys.stderr)
            with self._surface_lock:
                sig = self._surface_signature()
                notify = self._tools_changed_fn if (
                    self._catalog_served and sig != self._published_surface) else None
                if notify is not None:
                    self._published_surface = sig
            if notify is not None:
                print("[cascade-mcp] tool surface changed after it was listed -> "
                      "notifications/tools/list_changed", file=sys.stderr)
                notify()
        except Exception as e:  # noqa: BLE001 -- a report must never poison the runtime
            print(f"[cascade-mcp] capability report failed: {type(e).__name__}: {e}",
                  file=sys.stderr)

    def list_tools(self) -> list[dict]:
        if self._composed:
            from .robot_runtime import robot_tool_descriptors
            candidates = [t.as_spec() for t in robot_tool_descriptors(self._get_mobile_config()).values()]
        elif self._mobile:
            from ..skills.mobile_runtime import tool_specs_for_profiles

            candidates = tool_specs_for_profiles(self._get_mobile_config().bases)
        else:
            from ..skills.runtime import TOOL_SPECS

            candidates = TOOL_SPECS + _EXTRA_TOOLS
        # three independent filters: loop-internal, operator override, and
        # what the probed rig cannot do (empty until the runtime is built).
        # Computed and published under _surface_lock so a build finishing
        # concurrently either shows up here or triggers list_changed.
        with self._surface_lock:
            withheld = self.withheld_tools()
            single_arm = self._single_arm()
            self._catalog_served = True
            self._published_surface = (tuple(sorted(withheld)), single_arm)
        dropped = _EXCLUDED_TOOLS | _hidden_tools() | set(withheld)
        specs = [t for t in candidates if t["name"] not in dropped]
        motion: frozenset | set = frozenset()
        if single_arm:
            from ..skills.runtime import _MOTION_SKILLS as motion
        out = []
        for t in specs:
            schema = t["parameters"]
            if single_arm and t["name"] in motion and "arm" in schema.get("properties", {}):
                schema = copy.deepcopy(schema)  # TOOL_SPECS is shared; never mutate it
                schema["properties"].pop("arm")
            out.append({
                "name": t["name"],
                "description": t["description"],
                "inputSchema": schema,
            })
        return out

    def _reject_withheld(self, name: str) -> dict | None:
        """The call-time twin of the catalog trim: a model that listed tools
        before the build (or memorized one) gets the reason, not a motion."""
        withheld = self.withheld_tools()
        if name not in withheld:
            return None
        return _text_result(
            {"ok": False,
             "error": f"tool {name!r} is not available on this rig: {withheld[name]}",
             "capabilities": self.capabilities(),
             "tools_withheld": withheld},
            is_error=True,
        )

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
        if self._bounded:
            return self._call_mobile_tool(name, arguments)
        runtime = self._ensure_runtime()
        # after the build, never before: unprobed is not unavailable
        rejected = self._reject_withheld(name)
        if rejected is not None:
            return rejected
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
                if name == "annotated_view" and result.get("ok"):
                    return self._image_result(
                        runtime, runtime.last_annotated_frame, annotations=result,
                    )
                if name == "recall_step" and result.get("ok"):
                    return self._recall_result(runtime, result)
        return _text_result(result, is_error=not result.get("ok", False))

    def _call_mobile_tool(self, name, arguments):
        allowed = {s["name"] for s in self.list_tools()}
        if name not in allowed:
            return _text_result({"ok": False, "error": f"unsupported mobile tool: {name!r}"}, is_error=True)
        if name in self._stop_tools():
            return self.stop_now(navigation=name == "stop_navigation")
        if name == "reset_stop":
            whole_body = self._composed and self._get_mobile_config().as_dict().get("whole_body") is not None
            if not isinstance(arguments, dict) or (arguments and not whole_body):
                return _text_result({"ok": False, "error": "reset_stop takes no arguments"}, is_error=True)
            # Whole-body: the runtime validates the single required domain.
            return self.reset_now(arguments if whole_body else None)
        if self._composed and name == "list_resources":
            if not isinstance(arguments, dict) or arguments:
                return _text_result({"ok": False, "error": "list_resources takes no arguments"}, is_error=True)
            from .robot_runtime import describe_robot
            from ..robotics.resources import ResourceCatalog
            from ..robotics.embodiment import embodiment_metadata
            from ..robotics.whole_body import contract_metadata
            cfg = self._get_mobile_config()
            domains = describe_robot(cfg)
            catalog = ResourceCatalog([r for d in domains.values() for r in d.resources])
            return _text_result({"ok": True, "metadata_source": "configured_profile",
                                 **catalog.as_dict(), **embodiment_metadata(cfg.as_dict().get("embodiment"), catalog),
                                 **contract_metadata(cfg.as_dict().get("whole_body"))})
        if name == "list_bases":
            if arguments:
                return _text_result({"ok": False, "error": "list_bases takes no arguments"}, is_error=True)
            return _text_result({"ok": True, "metadata_source": "configured_profile", "bases": [
                {"name": p["name"], "robot_id": p["robot_id"], "source": p["source"],
                 "measurement_kind": "kinematic_mock" if p["type"] == "mock" else "physics",
                 "capabilities": p["capabilities"], "admission": p.get("admission", "pending_physical_admission")}
                for p in self._get_mobile_config().bases]})
        if self._input_closed and name in self._motion_tools():
            return _text_result({"ok": False, "execution_ok": False, "error": "input closed; pending motion invalidated"}, is_error=True)
        runtime = self._ensure_runtime()
        with self._exec_lock, contextlib.redirect_stdout(sys.stderr):
            runtime.current_tier = "mcp-host"
            try:
                result = runtime.execute(name, arguments)
            finally:
                runtime.current_tier = None
            runtime.last_path = "mcp-host"
        # Mobile bytes are already validated JPEG from the passive mobile RPC,
        # never run them through the arm/depth/scene image helpers.
        summary = {k: v for k, v in result.items() if k not in {"image_jpeg_b64", "image_before_jpeg_b64", "frame_images"}}
        content = []
        for frame in result.get("frame_images", []):
            content.append({"type": "text", "text": json.dumps({"caption": frame["caption"], "frame": frame["frame"]})})
            content.append({"type": "image", "data": frame["image_jpeg_b64"], "mimeType": "image/jpeg"})
        if result.get("ok") and result.get("image_jpeg_b64"):
            content.append({"type": "image", "data": result["image_jpeg_b64"], "mimeType": "image/jpeg"})
        content.append({"type": "text", "text": json.dumps(summary)})
        return {"content": content, "isError": not result.get("ok", False)}

    def input_closed(self):
        """EOF is a safety event on the READER, never queued behind motion."""
        self._input_closed = True
        if self._composed:
            self._request_composed_shutdown()
        elif self._mobile:
            self.stop_now()

    def _request_composed_shutdown(self):
        with self._stop_lock:
            self._stop_pending = True
            self._stop_serial += 1
        if self._runtime is not None:
            return self._runtime.request_shutdown()
        return {"ok": True, "shutdown_pending": True}

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

    def _recall_result(self, runtime, result: dict) -> dict:
        """`recall_step` for a chat host: the step's BEFORE/AFTER keyframes
        as image content items, one caption each, the JSON summary LAST --
        the same shape `task_memory` uses, so a host that renders one
        renders the other. The bytes come from `runtime.last_recalled_frames`
        (loaded by this very call; cleared first on a bad index, so an error
        result never reaches here with an old frame attached)."""
        content: list[dict] = []
        for fr in getattr(runtime, "last_recalled_frames", None) or []:
            jpeg = fr.get("jpeg")
            if not jpeg:
                continue
            content.append({"type": "text", "text": str(fr.get("caption") or result.get("caption", ""))})
            content.append({
                "type": "image",
                "data": base64.b64encode(jpeg).decode(),
                "mimeType": "image/jpeg",
            })
        content.append({"type": "text", "text": json.dumps(result)})
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
        # the three catalog filters, each explained (capabilities itself
        # arrives via _runtime_state, shared with the dashboard /state)
        state["tools_withheld"] = self.withheld_tools()
        state["tools_hidden_by_operator"] = sorted(_hidden_tools())
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

    def _image_result(self, runtime, frame, payload=None, camera=None, *, annotations=None) -> dict:
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
        robot_note = (
            "The image does not verify the arm's home pose or gripper contents. "
            "Do not claim the arm is home or the gripper is empty based on this image. "
            "Visible objects may be outside the arm's reach; an image does not verify reachability."
        )
        result = {**metadata, "observation_note": robot_note}
        if annotations is not None:
            result = {
                **{key: value for key, value in annotations.items() if key != "image"},
                "observation_frame": metadata,
                "observation_note": robot_note,
            }
        elif payload is not None:
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
                    "measured object position is needed. " + robot_note
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
                    # listChanged: the catalog served before the runtime is
                    # built is the full one; once the capability matrix has
                    # trimmed it the host is told to re-fetch
                    "capabilities": {"tools": {"listChanged": True}},
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
    from .signal_stop import StopSignals

    parser = argparse.ArgumentParser(description="CASCADE MCP server (stdio by default)")
    parser.add_argument("--launch-owner")
    parser.add_argument("--launch-state-dir", type=Path)
    parser.add_argument("--http", metavar="HOST:PORT",
                        help="serve Streamable HTTP instead of stdio (NemoClaw/OpenShell); needs "
                             "CASCADE_MCP_TOKEN, and TLS for any non-loopback HOST")
    parser.add_argument("--http-path", default="/mcp")
    parser.add_argument("--http-port-file", type=Path,
                        help="write the bound port here once listening (PORT 0 = any free port)")
    parser.add_argument("--token-file", type=Path, help="bearer secret file instead of CASCADE_MCP_TOKEN")
    parser.add_argument("--tls-cert", help="PEM certificate chain for an HTTPS listener")
    parser.add_argument("--tls-key", help="PEM private key for --tls-cert")
    args = parser.parse_args()
    if bool(args.launch_owner) != bool(args.launch_state_dir):
        parser.error("--launch-owner and --launch-state-dir must be supplied together")
    http = None
    if args.http:
        try:
            http = _http_settings(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.tls_cert or args.tls_key or args.http_port_file or args.token_file:
        parser.error("--tls-cert/--tls-key/--http-port-file/--token-file need --http")
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
    previous_stderr = sys.stderr
    with StopSignals(staff_reset=True) as signals:
        server._signals = signals
        try:
            if http is not None:
                exit_code = _serve_http(server, signals, **http)
            else:
                exit_code = _serve_stdio(server, signals)
        except KeyboardInterrupt:
            exit_code = 130
        finally:
            try:
                with signals.defer():
                    teardown = server.shutdown()
            finally:
                if isinstance(sys.stderr, _Tee) and sys.stderr is not previous_stderr:
                    log = sys.stderr._streams[-1]
                    sys.stderr = previous_stderr
                    log.close()
        if not isinstance(teardown, dict) or teardown.get("ok") is not True or teardown.get("complete") is not True:
            return 1
        return exit_code


def _serve_stdio(server, signals):
    import queue
    import select
    import signal
    from .signal_stop import SignalRequest

    with signals.defer():
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

        # The capability matrix may trim the catalog AFTER a host already
        # listed it (prewarm finishing late, or CASCADE_PREWARM=0): the
        # server declared listChanged, so it says so and the host re-fetches.
        server._tools_changed_fn = lambda: _send(
            {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
        )

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
        inbox: queue.Queue = queue.Queue()
        _EOF = object()
        reader_halt = threading.Event()
        input_fd = os.dup(sys.stdin.fileno())

        def _lines():
            # Unlike `for line in sys.stdin`, this can be joined on signal exit
            # even when the host keeps its stdin pipe open indefinitely.
            pending = b""
            while not reader_halt.is_set():
                ready, _, _ = select.select([input_fd], [], [], .05)
                if not ready:
                    continue
                data = os.read(input_fd, 4096)
                if not data:
                    if pending:
                        yield pending.decode("utf-8")
                    return
                pending += data
                while b"\n" in pending and not reader_halt.is_set():
                    line, pending = pending.split(b"\n", 1)
                    yield line.decode("utf-8")

        def _read_stdin() -> None:
            """Feed the worker; short-circuit anything that must not queue
            behind a running tool call (see "Stop channel" in the docstring).
            This thread IS the stop channel: no single frame may kill it."""
            try:
                for line in _lines():
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
                            if not _admit(server, m, _send):
                                inbox.put((m, _send))
                    except Exception as e:
                        print(f"[cascade-mcp] reader error (frame skipped): {e}",
                              file=sys.stderr)
            finally:
                try:
                    server.input_closed()
                finally:
                    inbox.put(_EOF)

        reader = threading.Thread(target=_read_stdin, daemon=True, name="wrc-stdin")
        reader.start()
        if os.environ.get("CASCADE_PREWARM", "1") != "0":
            server.prewarm_async()
    try:
        return _worker_loop(server, signals, inbox, _EOF)
    finally:
        # Event.set is safe HERE, after the signal handler has returned.
        with signals.defer():
            if server._composed and signals.signum is None:
                server._request_composed_shutdown()
            elif server._mobile or signals.signum is not None:
                server.stop_now()
            reader_halt.set()
            reader.join()
            os.close(input_fd)


def _admit(server, m: dict, send) -> bool:
    """Receive-side short circuit shared by every transport: True when the
    frame was handled here and must NOT queue behind a running tool call.

    This is the stop channel (module docstring): emergency_stop and the
    capability's stop tools latch the moment they arrive, a cancel of the
    in-flight motion freezes the arm, and keepalive pings are answered at once.
    Everything else is stamped with the receive-time stop serial and queued.
    """
    method = m.get("method")
    params = m.get("params") or {}
    if (method == "tools/call"
            and (params.get("name") == "emergency_stop"
                 or (params.get("name") in server._stop_tools()
                     and params.get("name") not in _hidden_tools()))):
        result = server.stop_now(navigation=params.get("name") == "stop_navigation")
        if m.get("id") is not None:
            send(_response(m["id"], result))
        return True
    if method == "notifications/cancelled":
        server.cancel_request(params.get("requestId"))
        return True
    if method == "ping" and m.get("id") is not None:
        # host keepalives must not starve behind a motion
        send(_response(m["id"], {}))
        return True
    if server._bounded:
        # Server-owned receive-time fence; never trust a
        # caller-provided value on the wire.
        with server._stop_lock:
            m["_mobile_stop_serial"] = server._stop_serial
    return False


def _worker_loop(server, signals, inbox, eof) -> int:
    """The single serial worker: (frame, send) items in arrival order."""
    import signal
    from .signal_stop import SignalRequest

    while True:
        msg = None
        send = None
        try:
            signals.checkpoint()
            item = inbox.get()
            if item is eof:
                break
            msg, send = item
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
                stale_motion = False
                if server._bounded and is_call:
                    motion_skills = server._motion_tools()

                    with server._stop_lock:
                        stale_motion = ((msg.get("params") or {}).get("name") in motion_skills
                                        and msg.get("_mobile_stop_serial") != server._stop_serial)
                if stale_motion:
                    resp = _response(req_id, _text_result(
                        {"ok": False, "execution_ok": False,
                         "error": "queued motion invalidated by stop; re-issue a new command explicitly"},
                        is_error=True))
                else:
                    resp = handle_message(server, msg)
            finally:
                with server._cancel_lock:
                    server._inflight = None
                    if req_id is not None:
                        # a cancel that raced with completion is spent now
                        server._cancelled_ids.discard(req_id)
            if resp is not None:
                send(resp)
        except SignalRequest as request:
            # All interrupted locks (including runtime._gate, _stop_lock,
            # _cancel_lock and out_lock) have unwound. Never resume that
            # tool call. Stop invalidation precedes reset/logging/replies.
            server.stop_now()
            if request.signum == signal.SIGTERM:
                return 128 + signal.SIGTERM
            if request.signum == getattr(signal, "SIGUSR1", None):
                server.reset_now()
            if isinstance(msg, dict) and msg.get("id") is not None and send is not None:
                send(_response(msg["id"], _text_result(
                    {"ok": False, "execution_ok": False, "error": "interrupted by signal; re-issue explicitly"},
                    is_error=True)))
    return 0


# ── Streamable HTTP transport (NemoClaw / OpenShell) ─────────────────────
#
# NemoClaw registers only authenticated Streamable HTTP MCP endpoints and never
# launches or wraps a stdio server, so an agent sandboxed by OpenShell reaches
# the robot through this listener. The robot side (harness, perception, Isaac
# bridge, GPU) stays on the host; the sandbox gets the tool surface and nothing
# else. The SAME serial worker, `_admit` stop channel and cancellation run
# behind it; only framing differs. One process, one robot runtime, any number
# of MCP sessions (the NemoClaw readiness probe opens its own).

HTTP_MAX_BODY = 131_072      # OpenShell's managed-MCP request-body cap
HTTP_MIN_TOKEN = 24          # characters of bearer secret


def _http_settings(args) -> dict:
    """Validate the --http options BEFORE anything robot-side starts."""
    import ipaddress

    spec = args.http
    host, sep, port = spec.rpartition(":")
    if not sep or not port.isdigit() or not 0 <= int(port) <= 65535 or not host:
        raise ValueError(f"--http must be HOST:PORT, got {spec!r}")
    host = host.strip("[]")
    token = os.environ.get("CASCADE_MCP_TOKEN", "")
    if args.token_file is not None:
        token = Path(args.token_file).read_text().strip()
    if len(token) < HTTP_MIN_TOKEN:
        raise ValueError(
            f"HTTP mode needs a bearer secret of at least {HTTP_MIN_TOKEN} characters in "
            "CASCADE_MCP_TOKEN (or --token-file); refusing to serve the robot unauthenticated")
    if bool(args.tls_cert) != bool(args.tls_key):
        raise ValueError("--tls-cert and --tls-key go together")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not args.tls_cert:
        raise ValueError(
            f"a non-loopback listener ({host}) requires TLS (--tls-cert/--tls-key): NemoClaw "
            "admits only HTTPS private endpoints and the bearer secret must not cross a network in clear")
    path = args.http_path if args.http_path.startswith("/") else "/" + args.http_path
    return {"host": host, "port": int(port), "path": path, "token": token,
            "cert": args.tls_cert, "key": args.tls_key, "port_file": args.http_port_file}


def _serve_http(server, signals, *, host, port, path, token, cert=None, key=None, port_file=None):
    import hmac
    import queue
    import secrets
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    keepalive_s = float(os.environ.get("CASCADE_MCP_SSE_KEEPALIVE_S", "15"))
    expected = f"Bearer {token}".encode()
    sessions: set[str] = set()
    sessions_lock = threading.Lock()
    inbox: queue.Queue = queue.Queue()
    _EOF = object()

    with signals.defer():
        from ..config import PACKAGE_ROOT

        run_dir = Path(os.environ.get("CASCADE_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}"))
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            sys.stderr = _Tee(sys.stderr, open(run_dir / "server.log", "a", buffering=1))
        except Exception:  # noqa: BLE001 -- a read-only checkout must not kill the server
            pass
        # No server->client stream (GET answers 405), so a trimmed catalog
        # cannot be pushed: a withheld tool is still rejected, with its reason.
        server._tools_changed_fn = lambda: print(
            "[cascade-mcp] tool catalog changed (HTTP: clients re-list on their next tools/list)",
            file=sys.stderr)

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "cascade-mcp"
            sys_version = ""

            def log_message(self, fmt, *a):  # one line per refusal, nothing per success
                pass

            def _reply(self, status, body: bytes = b"", ctype="application/json", extra=None):
                self.send_response(status)
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                if body:
                    self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _refuse(self, status, why, extra=None):
                print(f"[cascade-mcp] http {self.command} refused {status}: {why}", file=sys.stderr)
                body = json.dumps({"jsonrpc": "2.0", "id": None,
                                   "error": {"code": -32600, "message": why}}).encode()
                # the body may be unread: never parse its tail as a next request
                self.close_connection = True
                self._reply(status, body, extra={**(extra or {}), "Connection": "close"})

            def _guard(self) -> str | None | bool:
                """-> the request's session id (None when absent), or False
                after a refusal was sent."""
                if self.path.split("?", 1)[0] != path:
                    self._refuse(404, "unknown path")
                    return False
                auth = (self.headers.get("Authorization") or "").encode()
                if not hmac.compare_digest(auth, expected):
                    self._refuse(401, "missing or wrong bearer token",
                                 {"WWW-Authenticate": 'Bearer realm="cascade-mcp"'})
                    return False
                if self.headers.get("Origin"):
                    # browsers send Origin; MCP hosts do not (DNS rebinding)
                    self._refuse(403, "Origin not allowed")
                    return False
                version = self.headers.get("MCP-Protocol-Version")
                if version and version not in PROTOCOL_VERSIONS:
                    self._refuse(400, f"unsupported MCP-Protocol-Version {version}")
                    return False
                return self.headers.get("Mcp-Session-Id")

            def do_GET(self):  # noqa: N802
                if self._guard() is not False:
                    self._reply(405, extra={"Allow": "POST, DELETE"})

            def do_DELETE(self):  # noqa: N802
                session = self._guard()
                if session is False:
                    return
                with sessions_lock:
                    known = session in sessions
                    sessions.discard(session)
                if known:
                    self._reply(200)
                else:
                    self._refuse(404, "unknown session")

            def do_POST(self):  # noqa: N802
                session = self._guard()
                if session is False:
                    return
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype != "application/json":
                    return self._refuse(415, "Content-Type must be application/json")
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    return self._refuse(400, "bad Content-Length")
                if length > HTTP_MAX_BODY:
                    return self._refuse(413, f"body over {HTTP_MAX_BODY} bytes")
                try:
                    m = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return self._refuse(400, "body is not JSON")
                if not isinstance(m, dict):
                    return self._refuse(400, "one JSON-RPC object per request (no batches)")
                method = m.get("method")
                extra = {}
                if method == "initialize":
                    session = secrets.token_urlsafe(24)
                    with sessions_lock:
                        sessions.add(session)
                    extra["Mcp-Session-Id"] = session
                elif session is not None:
                    with sessions_lock:
                        if session not in sessions:
                            return self._refuse(404, "unknown session; re-initialize")
                # JSON-RPC ids are unique per session only: namespace them so a
                # cancel from one session can never stop another's motion.
                ns = f"{session or '-'}|"
                if method == "notifications/cancelled":
                    params = dict(m.get("params") or {})
                    if "requestId" in params:
                        params["requestId"] = ns + json.dumps(params["requestId"])
                    m = {**m, "params": params}
                req_id = m.get("id")
                if method is None or req_id is None:
                    # notification or client response: accept, never answer
                    if method is not None and not _admit(server, m, lambda _r: None):
                        inbox.put((m, lambda _r: None))
                    return self._reply(202, extra=extra)
                done = threading.Event()
                slot: list = []

                def send(resp, _orig=req_id):
                    resp = dict(resp)
                    resp["id"] = _orig
                    slot.append(resp)
                    done.set()

                m = {**m, "id": ns + json.dumps(req_id)}
                if not _admit(server, m, send):
                    inbox.put((m, send))
                accept = self.headers.get("Accept") or ""
                if method == "tools/call" and "text/event-stream" in accept:
                    return self._stream(done, slot, extra)
                done.wait()
                self._reply(200, json.dumps(slot[0]).encode(), extra=extra)

            def _stream(self, done, slot, extra):
                """SSE for tool calls: a 150 s pick keeps the connection
                visibly alive through proxies with idle timeouts."""
                self.send_response(200)
                for k, v in extra.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                alive = True
                while not done.wait(keepalive_s):
                    if alive:
                        try:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                        except OSError:
                            # a disconnect is not a cancel (MCP spec): the call
                            # runs to completion; cancel/stop are explicit
                            alive = False
                if alive:
                    try:
                        self.wfile.write(b"event: message\ndata: " + json.dumps(slot[0]).encode() + b"\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass

        httpd = ThreadingHTTPServer((host, port), Handler)
        httpd.daemon_threads = True
        scheme = "http"
        if cert:
            import ssl

            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(cert, key)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
            scheme = "https"
        bound_port = httpd.server_address[1]
        print(f"[cascade-mcp] cascade MCP server on {scheme}://{host}:{bound_port}{path} "
              f"(log: {run_dir / 'server.log'})", file=sys.stderr, flush=True)
        if port_file is not None:
            tmp = Path(str(port_file) + ".tmp")
            tmp.write_text(f"{bound_port}\n")
            os.replace(tmp, port_file)
        listener = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                                    daemon=True, name="cascade-mcp-http")
        listener.start()
        if os.environ.get("CASCADE_PREWARM", "1") != "0":
            server.prewarm_async()
    try:
        return _worker_loop(server, signals, inbox, _EOF)
    finally:
        with signals.defer():
            server.stop_now()
            httpd.shutdown()
            httpd.server_close()
            listener.join(5)


if __name__ == "__main__":
    sys.exit(main())
