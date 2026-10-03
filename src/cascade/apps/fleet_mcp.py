"""Bounded stdio MCP routing to explicitly configured independent robots.

The MCP host owns its agent conversations. This frontend preserves one runtime
episode, memory and verification history per robot; it does not run an LLM or
admit shared-world physics. stdout contains newline-delimited JSON-RPC only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import queue
import select
import sys
import threading
import time

from .fleet import build_fleet, load_fleet
from .mcp_server import PROTOCOL_VERSIONS, _response, _text_result
from .signal_stop import SignalRequest, StopSignals


MAX_FRAME_BYTES = 256 * 1024


def _tool(name, description, properties=None, required=()):
    return {"name": name, "description": description, "inputSchema": {
        "type": "object", "properties": properties or {}, "required": list(required),
        "additionalProperties": False}}


TOOLS = [
    _tool("fleet.catalog", "List exact robot IDs, independent episode IDs, resources and namespaced local tools. No physical admission."),
    _tool("fleet.execute", "Call one robot's exact namespaced tool within its current episode. Execution ACKs do not prove physical success.",
        {"robot_id": {"type": "string"}, "episode_id": {"type": "string"},
         "tool": {"type": "string"}, "arguments": {"type": "object"}},
        ("robot_id", "episode_id", "tool", "arguments")),
    _tool("fleet.stop", "Priority stop one exact robot, or all robots when robot_id is omitted. ACK does not prove rest.",
        {"robot_id": {"type": "string"}}),
    _tool("fleet.reset_stop", "Explicit operator reset for one robot; no motion replay or world reset.",
        {"robot_id": {"type": "string"}}, ("robot_id",)),
]


class FleetMcpServer:
    """One fixed daemon pool; queued and in-flight requests share a hard bound.

    Admission tokens/deadlines are captured on receipt, never supplied by the
    client or renewed by a queued worker. Stop/cancel bypass the ordinary queue.
    A blocked domain retains its slot and appears in incomplete shutdown.
    """

    def __init__(self, fleet, emit, *, max_workers=12, max_pending=64, timeout_s=30.):
        if (type(max_workers) is not int or not 1 <= max_workers <= 64
                or type(max_pending) is not int or not max_workers <= max_pending <= 128
                or type(timeout_s) not in (int, float) or not 0 < timeout_s <= 3600):
            raise ValueError("invalid MCP worker, pending-request or deadline bound")
        self.fleet, self.emit = fleet, emit
        self.timeout_s, self.max_pending = timeout_s, max_pending
        self.episodes = {name: rt.episode_id for name, rt in fleet.robots.items()}
        self.hidden = {name.strip() for name in os.environ.get("CASCADE_HIDE_TOOLS", "").split(",") if name.strip()}
        if "reset_stop" in self.hidden:
            self.hidden.add("fleet.reset_stop")
        self.hidden.discard("fleet.stop")
        self._gate = threading.Lock()
        self._pending = {}
        self._jobs = queue.Queue(maxsize=max_pending)
        self._halt = threading.Event()
        self.workers = [threading.Thread(target=self._worker, daemon=True,
            name=f"fleet-mcp-worker-{i}") for i in range(max_workers)]
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name="fleet-mcp-deadlines")
        try:
            for thread in [*self.workers, self.monitor]:
                thread.start()
        except BaseException:
            self._halt.set()
            for thread in [*self.workers, self.monitor]:
                if thread.ident is not None:
                    thread.join(.1)
            raise

    def _member(self, robot_id):
        if not isinstance(robot_id, str) or robot_id not in self.fleet.robots:
            raise ValueError("unknown exact fleet robot ID")
        return self.fleet.robots[robot_id]

    def _reply(self, request_id, value):
        self.emit(_response(request_id, _text_result(value, is_error=value.get("ok") is not True)))

    def _operation(self, name, arguments):
        if name in self.hidden:
            raise ValueError("tool hidden by operator configuration")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        if name == "fleet.catalog":
            if arguments:
                raise ValueError("fleet.catalog takes no arguments")
            return None, None
        if name == "fleet.stop":
            if set(arguments) - {"robot_id"}:
                raise ValueError("fleet.stop accepts only robot_id")
            if "robot_id" in arguments:
                self._member(arguments["robot_id"])
            return arguments.get("robot_id"), None
        if name == "fleet.reset_stop":
            if set(arguments) != {"robot_id"}:
                raise ValueError("fleet.reset_stop requires only robot_id")
            self._member(arguments["robot_id"])
            return arguments["robot_id"], None
        if name != "fleet.execute" or set(arguments) != {"robot_id", "episode_id", "tool", "arguments"}:
            raise ValueError("fleet.execute requires exact robot_id, episode_id, tool and arguments")
        member = self._member(arguments["robot_id"])
        if arguments["episode_id"] != self.episodes[arguments["robot_id"]] or member.episode_id != arguments["episode_id"]:
            raise ValueError("robot episode mismatch; fetch the current catalog")
        descriptor = member.tool_descriptors.get(arguments["tool"]) if isinstance(arguments["tool"], str) else None
        if descriptor is None or descriptor.name == "reset_stop":
            raise ValueError("unknown tool or reset requires explicit fleet.reset_stop")
        if descriptor.effect != "stop" and (descriptor.name in self.hidden or descriptor.local_name in self.hidden):
            raise ValueError("robot tool hidden by operator configuration")
        member._arguments(descriptor, arguments["arguments"])
        if descriptor.local_name == "task_memory" and arguments["arguments"].get("new_task") is True:
            raise ValueError("fleet MCP preserves the robot episode; task history cannot be reset")
        return arguments["robot_id"], descriptor

    def accept(self, message):
        """Accept one JSON-RPC frame from the sole input owner."""
        received = time.monotonic()
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self.emit(_response(None, error={"code": -32600, "message": "JSON-RPC object required"}))
            return
        request_id, method = message.get("id"), message.get("method")
        raw_params = message.get("params")
        raw_args = raw_params.get("arguments") if isinstance(raw_params, dict) else None
        selected = raw_args.get("robot_id") if isinstance(raw_args, dict) else None
        generation = (self.fleet.robots[selected].cancellation_token
                      if isinstance(selected, str) and selected in self.fleet.robots else None)
        if method == "notifications/cancelled":
            params = message.get("params")
            if isinstance(params, dict):
                self.cancel(params.get("requestId"), "client cancelled request")
            return
        if request_id is None:
            return  # Notifications never enqueue robot work.
        if (type(request_id) not in (str, int) or isinstance(request_id, str) and len(request_id) > 128):
            self.emit(_response(None, error={"code": -32600, "message": "invalid request ID"}))
            return
        with self._gate:
            duplicate = request_id in self._pending
        if duplicate:
            self.emit(_response(request_id, error={"code": -32600, "message": "request ID already in flight"}))
            return
        try:
            if self._halt.is_set():
                raise ValueError("fleet MCP server closing")
            if method == "initialize":
                params = message.get("params", {})
                version = params.get("protocolVersion") if isinstance(params, dict) else None
                self.emit(_response(request_id, {"protocolVersion": version if version in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                    "capabilities": {"tools": {}}, "serverInfo": {"name": "cascade-fleet", "version": "0.1.0"}}))
                return
            if method == "ping":
                self.emit(_response(request_id, {}))
                return
            if method == "tools/list":
                self.emit(_response(request_id, {"tools": [copy.deepcopy(t) for t in TOOLS if t["name"] not in self.hidden]}))
                return
            if method != "tools/call":
                self.emit(_response(request_id, error={"code": -32601, "message": "method not found"}))
                return
            params = message.get("params")
            if not isinstance(params, dict):
                raise ValueError("tools/call requires params")
            name, arguments = params.get("name"), params.get("arguments", {})
            robot, descriptor = self._operation(name, arguments)
            if name == "fleet.catalog":
                catalog = self.fleet.catalog()
                for robot_id, row in catalog.items():
                    row["tools"] = [t for t in row["tools"] if t["name"] != "reset_stop"
                        and (t["name"] in self.fleet.robots[robot_id].stop_skills
                             or t["name"] not in self.hidden and t["name"].split(".")[-1] not in self.hidden)]
                    for tool in row["tools"]:
                        properties = tool["parameters"].get("properties", {})
                        if tool["name"].split(".")[-1] == "task_memory" and "new_task" in properties:
                            properties["new_task"] = {"type": "boolean", "const": False}
                self._reply(request_id, {"ok": True, "robots": catalog,
                    "physical_fleet_admission": False, "agent_host": "external MCP client"})
                return
            if name == "fleet.stop" or descriptor is not None and descriptor.effect == "stop":
                self._reply(request_id, self.fleet.stop(robot))
                return
            record = {"id": request_id, "name": name, "arguments": copy.deepcopy(arguments),
                "robot": robot, "generation": generation,
                "deadline": received + self.timeout_s, "cancel": threading.Event(),
                "reason": None, "stops_on_cancel": descriptor is None or descriptor.effect in {"motion", "control"}}
            with self._gate:
                if len(self._pending) >= self.max_pending:
                    raise ValueError("fleet MCP pending request capacity reached; request not admitted")
                self._pending[request_id] = record
                self._jobs.put_nowait(record)
        except Exception as exc:
            self._reply(request_id, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def cancel(self, request_id, reason, *, expected_record=None):
        if type(request_id) not in (str, int):
            return
        with self._gate:
            record = self._pending.get(request_id)
            if (record is None or record["cancel"].is_set()
                    or expected_record is not None and record is not expected_record):
                return
        self._cancel_records([record], reason)

    def _cancel_records(self, records, reason):
        selected = []
        with self._gate:
            for record in records:
                if self._pending.get(record["id"]) is record and not record["cancel"].is_set():
                    record["reason"] = reason
                    record["cancel"].set()
                    if record["stops_on_cancel"]:
                        selected.append(record)
        if selected:
            try:
                receipt = self.fleet.request_stop_robots({record["robot"] for record in selected})
            except Exception as exc:
                receipt = {"ok": False, "error": str(exc), "physical_stop_verified": False}
            for record in selected:
                record["stop"] = receipt

    def _monitor(self):
        while not self._halt.wait(.02):
            with self._gate:
                expired = [value for value in self._pending.values()
                           if time.monotonic() >= value["deadline"] and not value["cancel"].is_set()]
            self._cancel_records(expired, "original request deadline expired")

    def _worker(self):
        while not self._halt.is_set():
            try:
                record = self._jobs.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                if record["cancel"].is_set() or time.monotonic() >= record["deadline"] or self._halt.is_set():
                    raise ValueError(record["reason"] or "request expired before dispatch")
                args, robot = record["arguments"], record["robot"]
                if self.fleet.robots[robot].episode_id != self.episodes[robot]:
                    raise ValueError("robot episode changed before dispatch")
                if record["name"] == "fleet.reset_stop":
                    result = self.fleet.reset_stop(robot, expected_generation=record["generation"],
                                                  deadline_monotonic_s=record["deadline"])
                else:
                    result = self.fleet.execute(robot, args["tool"], args["arguments"],
                        expected_generation=record["generation"], deadline_monotonic_s=record["deadline"])
                if record["cancel"].is_set() or time.monotonic() >= record["deadline"]:
                    self.cancel(record["id"], "original request deadline expired", expected_record=record)
                    result = {"ok": False, "error": record["reason"], "result": result}
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    self.cancel(record["id"], "worker interrupted", expected_record=record)
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                self._reply(record["id"], result)
            except Exception:
                self._halt.set()
                self.fleet.stop()
            finally:
                with self._gate:
                    self._pending.pop(record["id"], None)
                self._jobs.task_done()

    def close(self, timeout_s=2.):
        self._halt.set()
        # Preserve each domain's shutdown/parking policy. Explicit stops have
        # already been latched and cannot be weakened by graceful shutdown.
        errors = {}
        for name, runtime in self.fleet.robots.items():
            try:
                runtime.queue_shutdown()
            except Exception as exc:
                errors[name] = str(exc)
        deadline = time.monotonic() + timeout_s
        for thread in [*self.workers, self.monitor]:
            thread.join(max(0., deadline - time.monotonic()))
        result = self.fleet.close()
        pending = [thread.name for thread in [*self.workers, self.monitor] if thread.is_alive()]
        return {"ok": result["ok"] and not pending and not errors, "complete": result["complete"] and not pending,
                "fleet": result, "pending_workers": pending, "shutdown_errors": errors,
                "physical_rest_verified": False}


def serve_stdio(server, signals):
    pending = b""
    while not server._halt.is_set():
        signals.checkpoint()
        ready, _, _ = select.select([sys.stdin.fileno()], [], [], .05)
        if not ready:
            continue
        data = os.read(sys.stdin.fileno(), 4096)
        if not data:
            return 0
        pending += data
        if len(pending) > MAX_FRAME_BYTES:
            raise ValueError("MCP input frame exceeds byte bound")
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except (ValueError, UnicodeError, RecursionError):
                server.emit(_response(None, error={"code": -32700, "message": "invalid JSON frame"}))
                continue
            with signals.defer():  # finish publishing a request before signal cleanup
                server.accept(message)
            signals.checkpoint()
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-pending", type=int, default=64)
    args = parser.parse_args(argv)
    plan = load_fleet(args.fleet, task="External MCP agent episode")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    protocol_out, output_lock = sys.stdout, threading.Lock()
    sys.stdout = sys.stderr

    def emit(value):
        with output_lock:
            protocol_out.write(json.dumps(value, allow_nan=False) + "\n")
            protocol_out.flush()

    fleet = server = None
    error = None
    code = 1
    closure = {"ok": False, "complete": False}
    with StopSignals() as signals:
        try:
            with signals.defer():
                fleet = build_fleet(plan, args.run_dir)
                server = FleetMcpServer(fleet, emit, max_workers=plan["max_workers"],
                    max_pending=args.max_pending, timeout_s=plan["deadline_s"])
            signals.checkpoint()
            code = serve_stdio(server, signals)
        except (SignalRequest, KeyboardInterrupt):
            if fleet is not None:
                fleet.stop()
            code = 128 + signals.signum if signals.signum else 130
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if fleet is not None:
                fleet.stop()
            closure = getattr(exc, "fleet_teardown", closure)
        finally:
            with signals.defer():
                if server is not None:
                    closure = server.close()
                elif fleet is not None:
                    closure = fleet.close()
                (args.run_dir / "mcp-report.json").write_text(json.dumps({"error": error,
                    "shutdown": closure, "physical_fleet_admission": False}, indent=2) + "\n")
            sys.stdout = protocol_out
    return code if closure.get("ok") and closure.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
