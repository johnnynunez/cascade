"""Persistent wrc_demo runtime against Isaac Sim: dashboard on :8090 +
one scripted pink-cube pick, then stays alive serving the livestream."""
import os
import sys
import time
import urllib.request

# YOLOE re-embeds text prompts on class-list changes; without offline mode
# ultralytics phones GitHub on those paths (rate-limited at the booth ->
# seconds-long stalls that starve the 3 Hz watcher).
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "True")
# YOLOE's text encoder (mobileclip2_b.ts) is resolved relative to the CWD:
# a runner launched from the wrong directory silently loses ALL detections
# ("mobileclip2_b.ts does not exist" per watcher tick). Pin the cwd to the
# repo, where the checkpoint lives.
os.chdir("/home/spark/Projects/demo/wrc_demo")

sys.path.insert(0, "/home/spark/Projects/demo/wrc_demo/src")
from pathlib import Path

from wrc_demo.agent.llm import make_llm
from wrc_demo.agent.orchestrator import AgentOrchestrator
from wrc_demo.agent.reflex import ExperienceMemory, FastPlanner
from wrc_demo.apps.demo import build_runtime, shutdown_runtime
from wrc_demo.config import load_demo_config

# Real deliberation tier when available: Anthropic API key > local Qwen
# server > honest mock. The reflex/experience tiers work the same either way.
llm_profile = "mock"
qwen_up = False
try:
    urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
    qwen_up = True
except Exception:
    pass
if os.environ.get("ANTHROPIC_API_KEY"):
    llm_profile = "anthropic"
elif qwen_up:
    llm_profile = "local_qwen"
print(f"LLM TIER: {llm_profile}", flush=True)

cfg = load_demo_config(cameras=["isaac", "isaac_side", "isaac_wrist"],
                       arm="isaac", llm=llm_profile)
if qwen_up:
    # VLM grounding: 2nd perception filter when YOLOE misses (slow path only)
    cfg._data["grounder"] = {"base_url": "http://127.0.0.1:8080/v1",
                             "model": "qwen3.6-27b"}
cfg._data["detector"]["conf"] = 0.12  # pastel cubes on RTX renders sit ~0.15
# name the booth objects explicitly: beliefs are stored under DETECTOR
# labels, and "banana" cannot resolve a belief labeled "fruit"
cfg._data["detect_classes"] = [
    "banana", "cracker box", "soup can", "cube", "box", "bottle", "toy",
]
cfg._data["grasp"]["backend"] = "graspgenx"  # learned 6-DoF, OBB fallback
runtime, arm = build_runtime(cfg, Path("/tmp/wrc_isaac_live"), serve=True)
print("DASHBOARD:", runtime.stream_server.url if runtime.stream_server else "OFF", flush=True)

agent = AgentOrchestrator(
    make_llm(cfg.llm), runtime, decompose=False,
    fast_planner=FastPlanner(ExperienceMemory(Path("/tmp/wrc_isaac_live/exp.json"))),
)


def run_task(task: str):
    runtime.current_task = task
    try:
        r = agent.run_task(task)
        if r.path == "llm" and llm_profile == "mock":
            # The web runner's deliberation tier is a MOCK: be honest in
            # the narration instead of silently "succeeding".
            runtime.memory.add(
                "note",
                f"I could not act on {task!r} with my reflexes. For "
                "free-form language, talk to me through Claude CLI "
                "(cd wrc_demo && claude), or start the web runner with a "
                "real LLM (ANTHROPIC_API_KEY + llm='anthropic').",
            )
        print(f"TASK {task!r}: success={r.success} path={r.path} {r.duration_s}s",
              flush=True)
        print(f"  SUMMARY: {r.summary[:400]}", flush=True)
        for entry in (r.tool_log or [])[-6:]:
            import json as _json

            print(f"  TOOL {entry.get('tool')}: "
                  f"{_json.dumps(entry.get('result'))[:220]}", flush=True)
        for ev in runtime.memory.events()[-10:]:
            print(f"  NOTE[{ev.kind}]: {ev.text}"[:240], flush=True)
        return r
    finally:
        runtime.current_task = None


if runtime.stream_server is not None:
    runtime.stream_server.set_task_fn(run_task)  # the web chat box

deadline = time.monotonic() + 60
while time.monotonic() < deadline and runtime.beliefs.find("pink object") is None:
    time.sleep(0.3)
print("BELIEFS:", [(o["label"], o["color"], [round(p, 2) for p in o["position"]])
                   for o in runtime.beliefs.summary()], flush=True)

r = run_task("pick and place pink object")
print(f"RESULT: success={r.success} path={r.path} {r.duration_s}s", flush=True)
print("SUMMARY:", r.summary[:300], flush=True)
if r.tool_log:
    import json

    print("LAST:", json.dumps(r.tool_log[-1]["result"])[:400], flush=True)
print("BELIEFS-AFTER:", [(o["label"], o["color"], [round(p, 2) for p in o["position"]])
                         for o in runtime.beliefs.summary()], flush=True)
print("DASHBOARD-LIVE (Ctrl+C to stop)", flush=True)
try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    pass
finally:
    shutdown_runtime(runtime, arm)
