#!/usr/bin/env python3
"""live-w6: build docs/evidence/w6-live-20261010/ from the raw live records (copies + small summaries).
Absolute paths under the item dir are replaced by `<live-w6>`; no images, no keys."""
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

LV = Path("/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6")
WT = LV / "cascade"
EV = WT / "docs/evidence/w6-live-20261010"
EV.mkdir(parents=True, exist_ok=True)
MANIFEST: dict[str, str] = {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def note(p: Path) -> None:
    MANIFEST[str(p.relative_to(LV))] = sha(p)


def red(x):
    if isinstance(x, str):
        return x.replace(str(LV), "<live-w6>")
    if isinstance(x, list):
        return [red(v) for v in x]
    if isinstance(x, dict):
        return {k: red(v) for k, v in x.items()}
    return x


def dump(name: str, data) -> None:
    (EV / name).write_text(json.dumps(red(data), indent=1) + "\n")


# ── B46 ──────────────────────────────────────────────────────────────────
b46 = json.loads(subprocess.run(["python3", str(LV / "scratch/summarize_b46.py")], capture_output=True, text=True,
                                check=True).stdout)
for r in b46["runs"]:
    note(LV / r["sources"]["probe_json"])
    note(LV / r["sources"]["trace"])
smoke = json.loads((LV / "b46/smoke1.json").read_text())
note(LV / "b46/smoke1.json")
b46["smoke"] = {"tag": "smoke1", "lane": smoke["lane"], "motion_s": smoke["motion_s"],
                "motion_result": smoke["motion_result"], "motion_postcondition": smoke["motion_postcondition"],
                "reads": smoke["reads"], "note": "harness smoke before the series; not part of the interleaved A/B"}
b46["protocol"] = {
    "probe": "scratch/live_lane_probe.py = the B46 probe (mcp-readonly-lane/scratch/live_lane_probe.py, sha256 "
             "7be3a5a7117d954f7f493c155d30bf0237d84e2ca20f1fdfb4c24a59f9be6d0e) + --server-cwd/--server-log/--raw-out "
             "and the motion postcondition in its report; call sequence unchanged",
    "motion": "pick_and_place {object: pink cube, destination: drop zone}", "read_delay_s": 1.0,
    "reads": ["world_state", "robot_knowledge", "verify_last_action", "camera_snapshot"],
    "scene": "bare reBot RS (assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda), PhysX, fresh stage per run",
    "order": ["on1", "off1", "on2", "off2", "on3", "off3", "on4", "off4", "stop1"],
    "grasp_backend": "GraspGen-X sidecar :47001 (warm, shared by every run)", "stop_run": "--stop-after 3 (lane on)"}
dump("b46_lane_ab.json", b46)

# ── B63 ──────────────────────────────────────────────────────────────────
B63 = {"runs": [], "protocol": {
    "registration": "scripts/launch.sh's MCP registration heredoc, extracted verbatim and run as launch.sh runs it "
                    "(\"$PY\" - <args>), --sim isaac --arm isaac --cameras isaac,isaac_side --occupancy none "
                    "--headless, profile hermes-live-w6, launch owner from process_owner.py init",
    "server": "entry command + args in entry cwd with ONLY entry env (+ PATH, HOME, PYTHONPATH=<wt>/src as the "
              "editable-install stand-in); stdio JSON-RPC; no OpenClaw gateway, no brain",
    "turn": "world_state -> pick_and_place {pink cube, drop zone} -> reset_scene -> world_state",
    "vla_stub": "vla1: scripts/serve_vla_stub.py --port 47010 --chunks vla_chunks.json; vla2: the same ScriptedPolicy "
                "+ StubPolicyServer classes with a per-request log (vla_stub_logged.py)",
    "vla_chunks": json.loads((LV / "scratch/vla_chunks.json").read_text())}}
for tag in ("vla1", "vla2", "an1", "an2", "an3", "an4", "an5"):
    d = LV / "b63" / tag
    s = json.loads((d / "summary.json").read_text())
    entry = json.loads((d / "entry.json").read_text())
    note(d / "summary.json")
    note(d / "entry.json")
    note(d / "calls.json")
    log = (LV / "logs" / f"mcp_{tag}.log").read_text(errors="replace")
    note(LV / "logs" / f"mcp_{tag}.log")
    lines = [ln for ln in log.splitlines() if ln.startswith(("[cascade] grasp executor", "[cascade] capabilities",
                                                             "[cascade] backends"))]
    trace = Path(s["trace"])
    note(trace) if str(trace).startswith(str(LV)) else None
    rec = {"tag": tag, "arm": "vla" if s["vla_port_requested"] else "analytic",
           "fresh_memory": s.get("fresh_memory", False),
           "entry": {"command": entry["command"], "args": [a if not re.fullmatch(r"[0-9a-f]{32}", a) else "<owner>"
                                                          for a in entry["args"]], "cwd": entry["cwd"],
                     "env": entry["env"], "requestTimeoutMs": entry["requestTimeoutMs"]},
           "server_env_keys": s["server_env_keys"], "server_log": lines,
           "world_state_backends": s["world_state_backends"], "tools": s["tools"], "pick": s["pick"],
           "reset": s["reset"], "proof": s.get("proof"), "server_exit": s["server_exit"],
           "heredoc_sha256": s["heredoc_sha256"], "launch_sh_sha256": s["launch_sh_sha256"],
           "trace_sha256": sha(trace)}
    req = LV / "b63" / f"vla_requests_{tag}.jsonl"
    if req.exists():
        note(req)
        rows = [json.loads(x) for x in req.read_text().splitlines() if x.strip()]
        rec["vla_requests"] = [{"call": r["call"], "prompt": r["prompt"], "state": r["state"],
                                "image_shape": r["image_shape"], "image_dtype": r["image_dtype"],
                                "reply_gripper": r["chunk"][0][-1]} for r in rows]
    B63["runs"].append(rec)
dump("b63_vla_env.json", B63)

# ── B44 ──────────────────────────────────────────────────────────────────
b44 = LV / "b44"
for f in ("run-summary.json", "judge.json", "proof.json", "proof_sha_before.txt", "proof_sha_after.txt",
          "judge_turn.out", "judge_block_extracted.sh"):
    note(b44 / f)
summary = json.loads((b44 / "run-summary.json").read_text())
verdict = json.loads((b44 / "judge.json").read_text())
proof = json.loads((b44 / "proof.json").read_text())
before = [ln.split()[0] for ln in (b44 / "proof_sha_before.txt").read_text().splitlines()]
after = [ln.split()[0] for ln in (b44 / "proof_sha_after.txt").read_text().splitlines()]
banner = [ln for ln in (b44 / "judge_turn.out").read_text().splitlines() if ln.startswith("[launch] judge:")]
steps = [{k: s.get(k) for k in ("step", "skill", "task", "hop", "physics", "channel", "ok", "tier", "duration_s",
                                "error", "wrist_slots")} | {"raw_tail": (s.get("raw") or "")[-400:]}
         for s in verdict["steps"]]
dump("b44_judge_turn.json", {
    "receipt": {k: proof.get(k) for k in ("verified", "model", "sim", "session_id", "profile", "trace", "props_reset",
                                          "evidence_dir")},
    "receipt_sha256_before": before, "receipt_sha256_after": after, "run_summary": summary,
    "judge_verdict": {k: verdict.get(k) for k in ("judge", "mode", "final_progress", "progress", "hops", "confusion",
                                                   "summary_line", "wrist_views")} | {"steps": steps},
    "banner": banner,
    "judge_config_shape": {"backend": "vlm", "base_url": "http://127.0.0.1:8080/v1", "api_key": "<any; not a secret>",
                           "model": "Qwen/Qwen3.8-27B", "mode": "incremental", "temperature": 0.1, "top_p": 0.9,
                           "max_tokens": 1536, "timeout_s": 300,
                           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "source_run": "b63 an5 (the launcher-style proof turn)"})

# ── scripts used (copies) + manifest ─────────────────────────────────────
for f in ("live_lane_probe.py", "launcher_turn.py", "judge_turn.sh", "bridge.sh", "guard.sh", "b46_run.sh",
          "b46_series.sh", "b63_run.sh", "b63_series.sh", "b63_run2.sh", "b63_series2.sh", "vla_stub_logged.py", "vla_chunks.json",
          "summarize_b46.py", "build_evidence.py"):
    shutil.copy2(LV / "scratch" / f, EV / f)
dump("manifest.json", {"note": "sha256 of every raw live record these summaries were built from (paths relative to "
                               "the item dir live-w6/, not committed)", "files": dict(sorted(MANIFEST.items()))})
print("written", sorted(os.listdir(EV)))
