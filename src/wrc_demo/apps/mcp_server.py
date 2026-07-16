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
    WRC_CAMERA   camera profile (default: mock)
    WRC_ARM      arm profile    (default: mock)
    WRC_RUN_DIR  trace directory (default: <repo>/runs/mcp_<pid>)

Hardware is attached lazily on the first tools/call, so initialize and
tools/list always work -- an agent can inspect the toolbox with the robot
powered off (pattern borrowed from AgenticROS's MCP server).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import time
from pathlib import Path

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "wrc-demo", "version": "0.1.0"}

# task_done belongs to the built-in loop; MCP agents manage their own tasks.
_EXCLUDED_TOOLS = {"task_done"}

_EXTRA_TOOLS = [
    {
        "name": "camera_snapshot",
        "description": (
            "Capture a camera frame and return it as an image, with the "
            "depth source noted. Use this to SEE the workspace."
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

    # ── runtime lifecycle ────────────────────────────────────────────────

    def _ensure_runtime(self):
        if self._runtime is not None:
            return self._runtime
        if self._init_error is not None:
            raise RuntimeError(f"hardware init failed earlier: {self._init_error}")
        from ..apps.demo import build_runtime
        from ..config import PACKAGE_ROOT, load_demo_config

        camera = os.environ.get("WRC_CAMERA", "mock")
        arm = os.environ.get("WRC_ARM", "mock")
        run_dir = Path(
            os.environ.get("WRC_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}")
        )
        try:
            # Anything the stack prints must not corrupt the protocol stream.
            with contextlib.redirect_stdout(sys.stderr):
                cfg = load_demo_config(camera=camera, arm=arm, llm="mock")
                self._runtime, self._arm = build_runtime(cfg, run_dir)
            print(f"[wrc-mcp] runtime up: camera={camera} arm={arm}", file=sys.stderr)
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"
            raise
        return self._runtime

    def shutdown(self):
        with contextlib.redirect_stdout(sys.stderr):
            if self._runtime is not None:
                with contextlib.suppress(Exception):
                    self._runtime.camera.close()
            if self._arm is not None:
                with contextlib.suppress(Exception):
                    self._arm.disconnect()

    # ── tool surface ─────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        from ..skills.runtime import TOOL_SPECS

        specs = [t for t in TOOL_SPECS if t["name"] not in _EXCLUDED_TOOLS]
        specs = specs + _EXTRA_TOOLS
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["parameters"],
            }
            for t in specs
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        runtime = self._ensure_runtime()
        with contextlib.redirect_stdout(sys.stderr):
            if name == "camera_snapshot":
                return self._camera_snapshot(runtime)
            if name == "emergency_stop":
                runtime.arm.stop()
                return _text_result({"ok": True, "stopped": True,
                                     "note": "arm frozen; call reset_stop to resume"})
            if name == "reset_stop":
                runtime.arm.harness.reset_estop()
                raw = runtime.arm.raw
                if hasattr(raw, "resume"):
                    raw.resume()
                elif hasattr(raw, "_stopped"):
                    raw._stopped = False
                return _text_result({"ok": True, "stopped": False})
            result = runtime.execute(name, arguments or {})
        return _text_result(result, is_error=not result.get("ok", False))

    def _camera_snapshot(self, runtime) -> dict:
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
                        {"depth_source": frame.depth_source, "t": time.time()}
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
    server = McpSkillServer()
    print("[wrc-mcp] wrc-demo MCP server on stdio", file=sys.stderr)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[wrc-mcp] bad JSON frame: {line[:120]}", file=sys.stderr)
                continue
            resp = handle_message(server, msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
