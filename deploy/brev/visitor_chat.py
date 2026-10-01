"""A narrow attendee chat adapter; gateway credentials never leave this process."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import threading
import time
import uuid

_turn_budget = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/native_turn_budget.py"))
NATIVE_TURN_TIMEOUT_S = _turn_budget["NATIVE_TURN_TIMEOUT_S"]
AGENT_EXIT_GRACE_S = _turn_budget["AGENT_EXIT_GRACE_S"]

TOOLS = frozenset({"describe_scene", "localize_object", "pick_and_place",
                   "reset_scene", "camera_snapshot", "world_state"})


class BusyError(RuntimeError):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        temporary.chmod(0o600)
        json.dump(value, stream)
    temporary.replace(path)


class AttendeeChat:
    def __init__(self, repo):
        self.repo = Path(repo).resolve(strict=True)
        self.state = self.repo / "runs/.launch/profile-cascade-demo"
        self.directory = self.state / "visitor"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path = self.directory / "order.lock"
        self.admission_path = self.directory / "admission.lock"
        self.result_path = self.directory / "latest.json"

    def configuration(self):
        proof = json.loads((self.state / "proof.json").read_text())
        config = json.loads((self.state / "openclaw/openclaw.json").read_text())
        expected = {"cascade__" + tool for tool in TOOLS}
        if (proof.get("verified") is not True or proof.get("sim") != "isaac"
                or not str(proof.get("model", "")).endswith("/Qwen/Qwen3.8-27B")
                or not re.fullmatch(r"cascade-proof-[a-f0-9]{32}", proof.get("session_id", ""))
                or set(config.get("tools", {}).get("allow", [])) != expected
                or config.get("agents", {}).get("defaults", {}).get("skills") != []
                or any(entry.get("tools") for entry in config.get("agents", {}).get("entries", {}).values())):
            raise ValueError("The prepared attendee profile is unavailable")
        if not self.live_demo(proof):
            raise ValueError("The prepared attendee world is unavailable")
        return proof, config

    def live_demo(self, proof):
        sys.path.insert(0, str(self.repo / "src"))
        from cascade.apps.process_owner import is_live, live_records, load_owner
        owner = load_owner(self.state, self.repo, "cascade-demo")
        return bool(owner and is_live(proof.get("process"), owner)
                    and {"qwen", "isaac_bridge", "gateway_child"} <=
                    {row["role"] for row in live_records(self.state, owner)})

    @contextmanager
    def admission(self):
        # This lock covers only admission/status bookkeeping, never an agent
        # turn. Health readers cannot impersonate a running order to submit().
        with self.admission_path.open("a+") as mutex:
            self.admission_path.chmod(0o600)
            fcntl.flock(mutex, fcntl.LOCK_EX)
            yield

    def busy(self):
        with self.admission(), self.lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False

    def unfinished(self, session):
        if not self.result_path.exists():
            return None
        record = json.loads(self.result_path.read_text())
        if (record.get("session") == session
                and record.get("order", {}).get("status") in ("running", "uncertain")):
            return record["order"]
        return None

    def order_state(self, session):
        # Serialize the snapshot with admission, but only try the worker lock:
        # a real active order must be reported immediately, never waited out.
        with self.admission(), self.lock_path.open("a+") as lock:
            busy = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy = True
            pending = self.unfinished(session)
        blocked = bool(pending and (pending["status"] == "uncertain" or not busy))
        return busy, blocked

    def status(self, identifier=None):
        try:
            proof, _ = self.configuration()
            ready = True
        except (OSError, ValueError, KeyError, TypeError):
            proof = {}
            ready = False
        busy, blocked = self.order_state(proof.get("session_id"))
        result = {"enabled": True, "ready": ready and not blocked, "busy": busy, "agent": "cascade-demo"}
        path = self.result_path
        if identifier is not None:
            if not re.fullmatch(r"[a-f0-9]{32}", identifier):
                return result
            path = self.directory / ("order-" + identifier + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            previous = record["order"]
            current = record.get("session") == proof.get("session_id")
            if previous.get("status") in ("running", "uncertain") and (not current or blocked):
                previous = {"id": previous["id"], "status": "error",
                            "message": ("The previous order could not be confirmed. The demo is reconnecting."
                                        if current else "The previous order was interrupted. The demo has restarted.")}
                result["order"] = previous
            elif current:
                result["order"] = previous
        return result

    def save_order(self, session, result):
        record = {"session": session, "order": result}
        atomic_json(self.directory / ("order-" + result["id"] + ".json"), record)
        atomic_json(self.result_path, record)

    def submit(self, message):
        if not isinstance(message, str) or not message.strip() or len(message) > 2000:
            raise ValueError("Use a message between 1 and 2000 characters")
        proof, config = self.configuration()
        with self.admission():
            lock = self.lock_path.open("a+")
            self.lock_path.chmod(0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise BusyError("Wait for the current order to finish") from None
            identifier = uuid.uuid4().hex
            try:
                # A lost CLI does not prove its gateway or physical action stopped.
                # Only a new verified proof session clears this durable latch.
                proof, config = self.configuration()
                if self.unfinished(proof["session_id"]):
                    raise BusyError("The demo must recover before another order")
                self.save_order(proof["session_id"], {"id": identifier, "status": "running", "started_at": time.time()})
                for old in sorted(self.directory.glob("order-*.json"), key=lambda path: path.stat().st_mtime)[:-32]:
                    old.unlink()
                worker = threading.Thread(target=self._turn, args=(identifier, message.strip(), proof, config, lock), daemon=True)
                worker.start()
            except BaseException:
                lock.close()
                raise
        return {"id": identifier, "status": "running"}

    def _turn(self, identifier, message, proof, config, lock):
        result = {"id": identifier, "status": "uncertain"}
        try:
            env = {key: value for key, value in os.environ.items() if not key.startswith("OPENCLAW_")}
            env.update(OPENCLAW_PROFILE="cascade-demo", OPENCLAW_STATE_DIR=str(self.state / "openclaw"),
                       OPENCLAW_CONFIG_PATH=str(self.state / "openclaw/openclaw.json"),
                       NODE_COMPILE_CACHE=str(self.state / "openclaw/cache/node-compile"))
            command = [str(self.repo / ".openclaw-cli/bin/openclaw"), "--profile", "cascade-demo", "agent",
                       "--session-id", proof["session_id"], "--model", proof["model"],
                       "--message", message, "--json", "--timeout", str(NATIVE_TURN_TIMEOUT_S)]
            # Keep the order lock alive in the CLI if the visitor is restarted.
            completed = subprocess.run(command, env=env, cwd=self.repo, capture_output=True,
                                       text=True, timeout=NATIVE_TURN_TIMEOUT_S + AGENT_EXIT_GRACE_S,
                                       pass_fds=(lock.fileno(),))
            if completed.returncode:
                raise ValueError("The attendee turn failed")
            envelope = json.loads(completed.stdout)
            answer = envelope.get("result", {})
            meta = answer.get("meta", {})
            agent = meta.get("agentMeta", {})
            summary = meta.get("toolSummary", {})
            names = summary.get("tools", [])
            allowed = {"cascade__" + tool for tool in TOOLS}
            if (envelope.get("status") != "ok" or meta.get("aborted")
                    or agent.get("sessionId") != proof["session_id"]
                    or f"{agent.get('provider')}/{agent.get('model')}" != proof["model"]
                    or not isinstance(names, list) or not set(names) <= allowed):
                raise ValueError("The attendee turn did not match this demo")
            paragraphs = [item["text"] for item in answer.get("payloads", [])
                          if isinstance(item, dict) and isinstance(item.get("text"), str)]
            text = "\n\n".join(paragraphs)[:12000]
            if not text:
                text = "The order finished. Check the cameras and the tool result."
            # Even an unexpected gateway reply cannot publish its credential.
            for secret in config.get("gateway", {}).get("auth", {}).values():
                if isinstance(secret, str) and len(secret) >= 8:
                    text = text.replace(secret, "[private]")
            result = {"id": identifier, "status": "done", "message": text,
                      "tools": [name.removeprefix("cascade__") for name in names],
                      "tool_failures": summary.get("failures", 0), "finished_at": time.time()}
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
            pass  # Never include command output, gateway errors or configuration.
        finally:
            try:
                self.save_order(proof["session_id"], result)
            finally:
                lock.close()
