#!/usr/bin/env python3
"""B46 live probe (for the parent's Isaac session; the child has no GPU).

Spawns the REAL MCP server over stdio with the read-only lane on (or off for
the A/B), starts one motion (pick_and_place by default) and immediately sends
the four lane reads, then reports per call: send->answer latency, whether it
answered before the motion did, and its `served_during_motion` marker. The
motion's own result (ok / verified / error) is reported so an A/B lane-on vs
lane-off shows the pick is unchanged. Optionally sends `emergency_stop` N s
into the motion to show the stop still preempts (then reset_stop).

    python live_lane_probe.py --repo <checkout> --python <venv python> \
        --cameras isaac,isaac_side --arm isaac --object "red cube" --lane 1 \
        --run-dir /tmp/b46-live-on > b46_live_on.json

Every CASCADE_* variable of the calling shell is passed through (bridge /
sidecar ports, CASCADE_PROGRAMS, ...); this script sets only the camera/arm
lists, the run dir, the lane switch, CASCADE_STREAM=0 and CASCADE_PREWARM=1.

live-w6 copy (10 Oct 2026, sha256 of the original B46 probe:
7be3a5a7117d954f7f493c155d30bf0237d84e2ca20f1fdfb4c24a59f9be6d0e). The call
sequence and every reported field are unchanged; ADDED only: --server-cwd (the
launcher's models/ dir, where YOLOE finds its text encoder), --server-log (the
server's stderr instead of /dev/null), --raw-out (every answered frame's JSON
text, images dropped) and the motion's postcondition in the report.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cameras", default="mock")
    ap.add_argument("--arm", default="mock")
    ap.add_argument("--motion", default="pick_and_place")
    ap.add_argument("--object", default="red cube")
    ap.add_argument("--destination", default=None)
    ap.add_argument("--lane", choices=("0", "1"), default="1")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stop-after", type=float, default=None,
                    help="send emergency_stop this many seconds into the motion (then reset_stop)")
    ap.add_argument("--read-delay", type=float, default=1.0,
                    help="seconds between starting the motion and sending the reads: the lane "
                         "serves only calls that ARRIVE while the motion is executing")
    ap.add_argument("--timeout", type=float, default=400.0)
    ap.add_argument("--server-cwd", default=None, help="server working directory (default: --repo)")
    ap.add_argument("--server-log", default=None, help="server stderr goes here (default: /dev/null)")
    ap.add_argument("--raw-out", default=None, help="write every answered frame's JSON text here")
    a = ap.parse_args()

    env = dict(os.environ)
    env.update(PYTHONPATH=os.path.join(a.repo, "src"), CASCADE_CAMERAS=a.cameras, CASCADE_ARM=a.arm,
               CASCADE_RUN_DIR=a.run_dir, CASCADE_STREAM="0", CASCADE_PREWARM="1",
               CASCADE_MCP_READONLY_LANE=a.lane)
    err = open(a.server_log, "w") if a.server_log else subprocess.DEVNULL
    proc = subprocess.Popen([a.python, "-m", "cascade.apps.mcp_server"], cwd=a.server_cwd or a.repo, env=env,
                            text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                            bufsize=1)
    out: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [out.put((time.monotonic(), ln)) for ln in proc.stdout], daemon=True).start()
    sent: dict[int, tuple[str, float]] = {}
    answers: dict[int, tuple[float, dict]] = {}
    next_id = [0]

    def send(method, params=None):
        next_id[0] += 1
        frame = {"jsonrpc": "2.0", "id": next_id[0], "method": method}
        if params is not None:
            frame["params"] = params
        sent[next_id[0]] = (params.get("name", method) if params else method, time.monotonic())
        proc.stdin.write(json.dumps(frame) + "\n")
        proc.stdin.flush()
        return next_id[0]

    def wait_for(req_id, deadline):
        while req_id not in answers:
            try:
                t, line = out.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise SystemExit(f"timeout waiting for id {req_id}")
            frame = json.loads(line)
            if "id" in frame:
                answers[frame["id"]] = (t, frame)
        return answers[req_id][1]

    def payload(frame):
        texts = [b["text"] for b in frame["result"]["content"] if b["type"] == "text"]
        try:
            return json.loads(texts[-1])
        except (IndexError, ValueError):
            return {}

    deadline = time.monotonic() + a.timeout
    wait_for(send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "b46-live-probe", "version": "0"}}), deadline)
    wait_for(send("tools/call", {"name": "world_state", "arguments": {}}), deadline)  # runtime built
    args = {"object": a.object} if a.motion == "pick_and_place" else {}
    if a.destination:
        args["destination"] = a.destination
    motion = send("tools/call", {"name": a.motion, "arguments": args})
    time.sleep(a.read_delay)
    reads = [send("tools/call", {"name": t, "arguments": {}})
             for t in ("world_state", "robot_knowledge", "verify_last_action", "camera_snapshot")]
    stop = None
    if a.stop_after is not None:
        time.sleep(max(0.0, a.stop_after - a.read_delay))
        stop = send("tools/call", {"name": "emergency_stop", "arguments": {}})
    for rid in [motion, *reads] + ([stop] if stop else []):
        wait_for(rid, deadline)
    after = wait_for(send("tools/call", {"name": "world_state", "arguments": {}}), deadline)
    if stop:
        wait_for(send("tools/call", {"name": "reset_stop", "arguments": {}}), deadline)
    t_motion = answers[motion][0]
    report = {"lane": a.lane, "motion": a.motion, "cameras": a.cameras, "arm": a.arm,
              "motion_s": round(t_motion - sent[motion][1], 3),
              "motion_result": {k: payload(answers[motion][1]).get(k)
                                for k in ("ok", "verified", "outcome", "error", "stage")},
              "reads": [], "after_motion_world_state_marked": "served_during_motion" in payload(after)}
    pc = payload(answers[motion][1]).get("postcondition") or {}
    report["motion_postcondition"] = {k: pc.get(k) for k in ("status", "channel", "kind", "reason")}
    for rid in reads:
        t, frame = answers[rid]
        p = payload(frame)
        report["reads"].append({"tool": sent[rid][0], "latency_s": round(t - sent[rid][1], 3),
                                "answered_before_motion": t < t_motion,
                                "isError": frame["result"].get("isError"),
                                "served_during_motion": p.get("served_during_motion", {}).get("motion")})
    if stop:
        t, frame = answers[stop]
        report["stop"] = {"after_s": a.stop_after, "latency_s": round(t - sent[stop][1], 3),
                          "answered_before_motion": t < t_motion, "stopped": payload(frame).get("stopped")}
    print(json.dumps(report, indent=1))
    if a.raw_out:
        raw = []
        for rid in sorted(answers):
            t, frame = answers[rid]
            name, t_sent = sent.get(rid, ("?", t))
            texts = [b.get("text") for b in (frame.get("result") or {}).get("content", []) if b.get("type") == "text"]
            raw.append({"id": rid, "tool": name, "sent_mono": round(t_sent, 3), "answered_mono": round(t, 3),
                        "isError": (frame.get("result") or {}).get("isError"), "error": frame.get("error"),
                        "texts": texts})
        with open(a.raw_out, "w") as fh:
            json.dump(raw, fh, indent=1)
    proc.stdin.close()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
