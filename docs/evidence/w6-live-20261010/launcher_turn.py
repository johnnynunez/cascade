#!/usr/bin/env python3
"""live-w6: a launcher-style MCP registration + proof turn on the live Isaac bridge (B63 + B44).

What is REAL here (from the worktree, unmodified):
  * scripts/launch.sh's MCP registration heredoc -- extracted verbatim at run time (the same regex
    tests/test_mcp_env_forwarding.py uses) and run as launch.sh runs it: `"$PY" - <args>` on stdin;
  * the launch owner: src/cascade/apps/process_owner.py `state-dir` / `init` (launch.sh's ownerctl);
  * the server: entry["command"] entry["args"] in entry["cwd"] with ONLY entry["env"] (+ PATH, HOME and
    PYTHONPATH=<wt>/src, the stand-in for the editable install a checkout's own .venv has);
  * the proof validators: scripts/demo_proof.py `_bound_world`, `validate_pick_trace`,
    `validate_reset_trace`, `_write_receipt` (imported, not re-implemented).
What is NOT the launcher: no OpenClaw gateway and no brain. The tool calls the brain would make in
demo_proof.run_proof (world_state -> pick_and_place pink cube -> drop zone -> reset_scene -> world_state)
are sent directly over the server's stdio, so `model` in the receipt says so.

usage: launcher_turn.py --tag T [--vla-port P] [--proof]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import runpy
import subprocess
import sys
import threading
import time
import uuid

LV = Path("/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6")
WT = LV / "cascade"
PY = "/home/johnny/Projects/demo/cascade/.venv/bin/python"
GPU0 = "GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3"
PROFILE = "hermes-live-w6"
STATE_ROOT = LV / "launch_state"
PORTS = {"CASCADE_BRIDGE_PORT": "47000", "CASCADE_GRASPGENX_PORT": "47001", "CASCADE_OCCUPANCY_PORT": "47002"}
SHARED = {"8611", "8080", "5556", "5557", "18789", "18790"}


def registering_env(vla_port: str | None) -> dict:
    """The launch shell: no inherited CASCADE_*/CUDA_*; what the operator + launch.sh export."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CASCADE_", "CUDA_")) and k != "PYTHONPATH"}
    env.update(PORTS)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU0, CASCADE_OPENCLAW_PROFILE=PROFILE,
               CASCADE_MCP_NAME="cascade", PYTHONPATH=str(WT / "src"))
    env["CASCADE_OCCUPANCY"] = "0"            # launch.sh line: [[ "$OCCUPANCY" != none ]] || export CASCADE_OCCUPANCY=0
    if vla_port:
        env.update(CASCADE_GRASP_EXECUTOR="vla", CASCADE_VLA_PORT=vla_port)
    return env


def ownerctl(env: dict, *args: str) -> str:
    out = subprocess.run([PY, str(WT / "src/cascade/apps/process_owner.py"), "--repo", str(WT), "--state-root",
                          str(STATE_ROOT), "--profile", PROFILE, *args], env=env, capture_output=True, text=True,
                         check=True)
    return out.stdout.strip()


def register(env: dict) -> tuple[dict, str, str, str]:
    state_dir = ownerctl(env, "state-dir")
    owner = ownerctl(env, "init")
    # launch.sh: a profile -> the memory stores live in the state dir (lines after "register the MCP server")
    env.setdefault("CASCADE_GRASP_MEMORY_PATH", f"{state_dir}/memory/grasp_memory.json")
    env.setdefault("CASCADE_ENVELOPE_PATH", f"{state_dir}/memory/envelope.json")
    env.setdefault("CASCADE_BELIEFS_PATH", f"{state_dir}/memory/beliefs.json")
    detector = env.get("CASCADE_DETECTOR_MODEL") or str(WT / "models/yoloe-11s-seg.pt")
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", (WT / "scripts/launch.sh").read_text(), re.DOTALL)
    [heredoc] = [b for b in blocks if '"requestTimeoutMs"' in b]
    # "$PY" - "$MCP_PY" "$REPO" "$CAMERAS" "$ARM" "$DETECTOR" "$CLASSES" "$SIM" "$STATE_DIR" "$LAUNCH_OWNER" "$ISAAC_GUI" "$OCCUPANCY"
    argv = [PY, "-", PY, str(WT), "isaac,isaac_side", "isaac", detector, "", "isaac", state_dir, owner, "0", "none"]
    out = subprocess.run(argv, input=heredoc + "\n", env=env, capture_output=True, text=True, cwd=str(WT))
    if out.returncode:
        raise SystemExit(f"registration heredoc failed: {out.stderr}")
    return json.loads(out.stdout), state_dir, owner, hashlib.sha256(heredoc.encode()).hexdigest()


class Stdio:
    def __init__(self, entry: dict, log_path: Path):
        env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], **entry["env"],
               "PYTHONPATH": str(WT / "src")}
        self.env = env
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen([entry["command"], *entry["args"]], cwd=entry["cwd"], env=env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [self.q.put(ln) for ln in self.proc.stdout], daemon=True).start()
        self.n = 0
        self.answers: dict = {}
        self.calls: list = []

    def call(self, method: str, params=None, timeout: float = 600.0) -> dict:
        self.n += 1
        rid = self.n
        frame = {"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params is not None else {})}
        t0 = time.monotonic()
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()
        deadline = t0 + timeout
        while rid not in self.answers:
            line = self.q.get(timeout=max(0.1, deadline - time.monotonic()))
            msg = json.loads(line)
            if "id" in msg:
                self.answers[msg["id"]] = msg
        msg = self.answers[rid]
        rec = {"id": rid, "method": method, "name": (params or {}).get("name"), "args": (params or {}).get("arguments"),
               "s": round(time.monotonic() - t0, 3)}
        if method == "tools/call":
            res = msg.get("result") or {}
            texts = [b.get("text") for b in res.get("content", []) if b.get("type") == "text"]
            rec.update(isError=res.get("isError"), texts=texts)
        self.calls.append(rec)
        return msg

    def tool(self, name: str, args: dict | None = None, timeout: float = 600.0) -> dict:
        msg = self.call("tools/call", {"name": name, "arguments": args or {}}, timeout)
        texts = [b.get("text") for b in (msg.get("result") or {}).get("content", []) if b.get("type") == "text"]
        try:
            return json.loads(texts[-1])
        except (IndexError, ValueError, TypeError):
            return {"_raw": texts}

    def close(self) -> int | None:
        try:
            self.proc.stdin.close()
            return self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait()
        finally:
            self.log.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vla-port", default=None)
    ap.add_argument("--proof", action="store_true", help="write the launcher receipt (proof.json) with demo_proof's validators")
    ap.add_argument("--fresh-memory", action="store_true",
                    help="operator-set per-run memory stores (launch.sh keeps a set CASCADE_*_PATH; default = the "
                         "profile's shared <state>/memory/ stores)")
    a = ap.parse_args()
    out = LV / "b63" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    env = registering_env(a.vla_port)
    if a.fresh_memory:
        mem = out / "memory"
        env.update(CASCADE_GRASP_MEMORY_PATH=str(mem / "grasp_memory.json"), CASCADE_ENVELOPE_PATH=str(mem / "envelope.json"),
                   CASCADE_BELIEFS_PATH=str(mem / "beliefs.json"))
    entry, state_dir, owner, heredoc_sha = register(env)
    (out / "entry.json").write_text(json.dumps(entry, indent=1) + "\n")
    eenv = entry["env"]
    ports = {k: v for k, v in eenv.items() if k.endswith("_PORT")}
    assert not (set(ports.values()) & SHARED), ports
    summary = {"tag": a.tag, "vla_port_requested": a.vla_port, "fresh_memory": a.fresh_memory, "heredoc_sha256": heredoc_sha,
               "launch_sh_sha256": hashlib.sha256((WT / "scripts/launch.sh").read_bytes()).hexdigest(),
               "entry_has": {k: eenv.get(k) for k in ("CASCADE_GRASP_EXECUTOR", "CASCADE_VLA_PORT", "CASCADE_BRIDGE_PORT",
                                                       "CASCADE_GRASPGENX_PORT", "CASCADE_OCCUPANCY", "CUDA_VISIBLE_DEVICES",
                                                       "CASCADE_OPENCLAW_PROFILE", "CASCADE_BELIEFS_PATH")},
               "entry_env_keys": sorted(eenv), "entry_args": entry["args"][:3], "entry_cwd": entry["cwd"]}

    dp = runpy.run_path(str(WT / "scripts/demo_proof.py"))
    sys.path.insert(0, str(WT / "src"))
    from cascade.apps.process_owner import load_owner, records
    owner_rec = load_owner(Path(state_dir), WT, PROFILE)
    baseline = {r.get("instance_id") for r in records(Path(state_dir))}
    session = "cascade-proof-" + uuid.uuid4().hex
    evidence = Path(state_dir) / session
    report = {"verified": False, "model": "none (direct MCP tool calls, no brain; live-w6 launcher-style turn)",
              "sim": "isaac", "session_id": session, "started_at": time.time(), "evidence_dir": str(evidence),
              "profile": PROFILE}
    if a.proof:
        evidence.mkdir(parents=True, mode=0o700)
        dp["_write_receipt"](report, Path(state_dir))   # invalidate the previous success first, like run_proof

    srv = Stdio(entry, LV / "logs" / f"mcp_{a.tag}.log")
    summary["server_env_keys"] = sorted(srv.env)
    try:
        srv.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "live-w6-launcher-turn", "version": "0"}}, 300)
        tools = srv.call("tools/list", None, 300)
        names = sorted(t["name"] for t in tools["result"]["tools"])
        summary["tools"] = {"n": len(names), "pick_and_place": "pick_and_place" in names,
                            "grasp_object": "grasp_object" in names}
        ws0 = srv.tool("world_state", {}, 600)
        summary["world_state_backends"] = ws0.get("backends")
        record = dp["_bound_world"](Path(state_dir), owner_rec, baseline, report["started_at"])
        trace = Path(record["run_dir"]) / "trace.jsonl"
        report["process"] = record
        started = time.time()
        pick = srv.tool("pick_and_place", {"object": "pink cube", "destination": "drop zone"}, 900)
        summary["pick"] = {k: pick.get(k) for k in ("ok", "verified", "outcome", "error", "stage", "executor",
                                                     "grasp_attempts", "duration_s")}
        summary["pick"]["postcondition"] = {k: (pick.get("postcondition") or {}).get(k)
                                            for k in ("status", "channel", "kind", "evidence")}
        reset_started = time.time()
        reset = srv.tool("reset_scene", {}, 600)
        summary["reset"] = {k: reset.get(k) for k in ("ok", "world", "props_reset", "error")}
        srv.tool("world_state", {}, 300)
        summary["trace"] = str(trace)
        if a.proof:
            try:
                dp["_bound_world"](Path(state_dir), owner_rec, baseline, report["started_at"], record)
                dp["validate_pick_trace"](trace, started, "pink cube")
                props = dp["validate_reset_trace"](trace, reset_started, "isaac", "pink cube")
                report.update(verified=True, trace=str(trace), props_reset=props)
            except dp["ProofError"] as exc:
                report.update(trace=str(trace), error=f"ProofError: {exc}")
            (evidence / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
            dp["_write_receipt"](report, Path(state_dir))
            summary["proof"] = {"path": str(Path(state_dir) / "proof.json"), "verified": report["verified"],
                                "error": report.get("error"), "evidence_dir": str(evidence)}
    finally:
        summary["server_exit"] = srv.close()
        (out / "calls.json").write_text(json.dumps(srv.calls, indent=1) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
