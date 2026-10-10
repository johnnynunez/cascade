#!/usr/bin/env python3
"""B36 wave-8 live probe: one fresh MCP server over stdio, the B36 series task.

Same call sequence as the 8-9 Oct series (live/probe_host.py over HTTPS):
get_observation -> pick_and_place {pink cube, drop zone, rigid} -> get_observation
-> reset_scene. Spawns `python -m cascade.apps.mcp_server` from --rig with only
the env given here (+ the caller's PATH/HOME); prints a JSON report.

usage: probe.py --rig <tree> --run-dir <dir> --server-cwd <models dir> --server-log <f>
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

CALLS = [
    ("get_observation", {}),
    ("pick_and_place", {"object": "pink cube", "destination": "drop zone", "material": "rigid"}),
    ("get_observation", {}),
    ("reset_scene", {}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--server-cwd", required=True)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()
    run = a.run_dir
    os.makedirs(run, exist_ok=True)
    py = "/home/johnny/Projects/demo/cascade/.venv/bin/python"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/home/johnny"),
        "PYTHONPATH": os.path.join(a.rig, "src"),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3",
        "CASCADE_ARM": "isaac", "CASCADE_CAMERAS": "isaac,isaac_side",
        "CASCADE_BRIDGE_PORT": "47701", "CASCADE_GRASPGENX_PORT": "47702", "CASCADE_OCCUPANCY_PORT": "47703",
        "CASCADE_HUG_PORT": "47704", "CASCADE_OCCUPANCY": "0", "CASCADE_VIEW": "0", "CASCADE_STREAM": "0",
        "CASCADE_DETECTOR_MODEL": os.path.join(a.server_cwd, "yoloe-11s-seg.pt"),
        "YOLO_OFFLINE": "True", "ULTRALYTICS_OFFLINE": "True",
        "CASCADE_RUN_DIR": run, "CASCADE_BELIEFS_PATH": f"{run}/beliefs.json",
        "CASCADE_GRASP_MEMORY_PATH": f"{run}/grasp.json", "CASCADE_ENVELOPE_PATH": f"{run}/envelope.json",
        "CASCADE_GRASP_EVIDENCE_DIR": f"{run}/grasp_evidence",
    }
    for k, v in env.items():
        if k.endswith("_PORT") and k.startswith("CASCADE_"):
            assert 47700 <= int(v) <= 47799, (k, v)
    err = open(a.server_log, "w")
    proc = subprocess.Popen([py, "-m", "cascade.apps.mcp_server"], cwd=a.server_cwd, env=env, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=1)
    out: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [out.put(ln) for ln in proc.stdout], daemon=True).start()
    rid = [0]
    deadline = time.monotonic() + a.timeout

    def call(method, params=None):
        rid[0] += 1
        frame = {"jsonrpc": "2.0", "id": rid[0], "method": method}
        if params is not None:
            frame["params"] = params
        t0 = time.monotonic()
        proc.stdin.write(json.dumps(frame) + "\n")
        proc.stdin.flush()
        while True:
            try:
                line = out.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise SystemExit(f"timeout waiting for {method} {params}")
            msg = json.loads(line)
            if msg.get("id") == rid[0]:
                return msg, time.monotonic() - t0

    report = {"rig": a.rig, "run_dir": run, "calls": []}
    call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "b36-w8-probe", "version": "0"}})
    for name, args in CALLS:
        msg, dt = call("tools/call", {"name": name, "arguments": args})
        res = msg.get("result") or {}
        texts = [b.get("text") for b in res.get("content", []) if b.get("type") == "text"]
        try:
            payload = json.loads(texts[-1])
        except (IndexError, TypeError, ValueError):
            payload = {"raw": (texts[-1] if texts else None)}
        entry = {"tool": name, "s": round(dt, 1), "isError": res.get("isError")}
        if name == "pick_and_place":
            entry["result"] = payload
        else:
            entry["result"] = {k: payload.get(k) for k in ("ok", "error", "props_reset", "observation_refreshed")}
        report["calls"].append(entry)
        print(f"{name} isError={res.get('isError')} {dt:.1f}s {json.dumps(payload)[:300]}", file=sys.stderr, flush=True)
    print(json.dumps(report, indent=1))
    proc.stdin.close()
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
