#!/usr/bin/env python3
"""Replay the captured OpenClaw request under the proposed Cosmos agent profile:
OpenClaw's inbound JSON envelopes removed from the user turn, the system prompt
replaced by the profile's short prompt, the tools narrowed to the profile's
allowlist. Scores the FIRST model turn of ~10 operator prompts; generates only.

usage: replay_profile.py <out.jsonl> <reps> <variant>[,<variant>...]
variants: profile_t0 | profile_tunset | captured_t0 (control: envelopes + full prompt + 49 tools)
"""
import copy
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).parent))
from replay_matrix_lib import send  # noqa: E402

D = Path("/home/johnny/.hermes/cache/scratch/cosmos-demo")
OUT = Path(sys.argv[1])
REPS = int(sys.argv[2])
VARIANTS = sys.argv[3].split(",")
SERVERS = {"nvfp4": "http://127.0.0.1:8084/v1", "bf16": "http://127.0.0.1:8085/v1"}
BASE = json.loads((D / "requests/0007.json").read_text())["request"]
PROFILE = json.loads((D / (__import__("os").environ.get("PROFILE_JSON") or "cosmos_profile.json")).read_text())  # {"system_prompt": ..., "tools": [...]}

PROMPTS = [
    ("Point at the lemon.", {"cascade__point_at": {"label": "lemon"}}),
    ("Could you put the green cube in the green square?",
     {"cascade__pick_and_place": {"object": "green cube", "destination": "green square"}}),
    ("Please put the orange in the open box.", {"cascade__pick_and_place": {"object": "orange", "destination": "open box"}}),
    ("Please put the pink cube in the green square, then put the lemon in the open box.",
     {"cascade__pick_and_place": {"object": "pink cube", "destination": "green square"}}),
    ("What can you see on the table?", {"cascade__describe_scene": {}, "cascade__list_objects": {}}),
    ("Let's start over.", {"cascade__reset_scene": {}}),
    ("Open the gripper.", {"cascade__open_gripper": {}}),
    ("Where is the tomato can?", {"cascade__localize_object": {"label": "tomato can"}}),
    ("Go home.", {"cascade__move_home": {}}),
    ("Stop!", {"cascade__emergency_stop": {}}),
    ("Reply with exactly OK and nothing else. Do not call tools.", None),
    ("What is the session status? Read it with the world_state tool.", {"cascade__world_state": {}}),
    ("Let's start over, then show the session status with the world_state tool.", {"cascade__reset_scene": {}}),
    ("Please put the pink cube in the drop zone.", {"cascade__pick_and_place": {"object": "pink cube"}}),
]


PROMPTS_SYS = {
    "p1": ("You control a robot arm through the cascade__* tools. When the user asks for an action or about "
           "the scene, call the one matching tool, wait for its result, then answer briefly in English. "
           "Never invent tool results."),
    "p3": ("You control a robot arm through the cascade__* tools. When the user asks for an action, about the "
           "scene or about the robot's state, call the one matching tool, wait for its result, then answer "
           "briefly in English. Never invent tool results."),
    "p2": ("You control a robot arm through the cascade__* tools. For every request about the robot or the "
           "scene, call exactly one matching tool, wait for its result, then answer briefly in English from "
           "that result. Never invent tool results."),
}


def build(prompt, variant):
    """variant: '+'-joined tokens: captured | p1 | p2 | t0 | t<x> | single | fp<x> | pp<x> | nolist"""
    tok = variant.split("+")
    r = copy.deepcopy(BASE)
    r["max_completion_tokens"] = 512
    if "captured" in tok:
        head = r["messages"][-1]["content"].rsplit("\n\n", 1)[0]
        r["messages"][-1]["content"] = head + "\n\n" + prompt
    else:
        sys_prompt = next((PROMPTS_SYS[t] for t in tok if t in PROMPTS_SYS), PROFILE["system_prompt"])
        r["messages"] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        names = [n for n in PROFILE["tools"] if not ("nolist" in tok and n == "cascade__list_objects")]
        r["tools"] = [t for t in r["tools"] if t["function"]["name"] in set(names)]
    for t in tok:
        if t == "t0":
            r["temperature"] = 0
        elif t.startswith("t") and t[1:].replace(".", "").isdigit():
            r["temperature"] = float(t[1:])
        elif t == "single":
            r["parallel_tool_calls"] = False
        elif t.startswith("fp"):
            r["frequency_penalty"] = float(t[2:])
        elif t.startswith("pp"):
            r["presence_penalty"] = float(t[2:])
    return r


def score(expect, res):
    calls = res.get("calls") or []
    if expect is None:
        return not calls and res.get("text", "").strip().strip(".").upper() == "OK"
    if res.get("finish") != "tool_calls" or len(calls) != 1 or calls[0]["name"] not in expect:
        return False
    try:
        args = json.loads(calls[0]["arguments"] or "{}")
    except ValueError:
        return False
    want = expect[calls[0]["name"]]
    for k, v in want.items():
        got = str(args.get(k, "")).strip().lower()
        if got.strip("\"'") != v or got != v:
            return False
    return True


lock = threading.Lock()


def job(kv, variant, i, rep):
    prompt, expect = PROMPTS[i]
    payload = build(prompt, variant)
    try:
        res = send(SERVERS[kv], payload)
    except Exception as exc:
        res = {"error": repr(exc)[:300]}
    row = {"kv": kv, "variant": variant, "prompt": prompt, "rep": rep, "correct": score(expect, res),
           "native_call": bool(res.get("calls")), "tools_sent": len(payload["tools"]),
           "system_chars": len(payload["messages"][0]["content"]), **res}
    with lock, OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


jobs = [(kv, v, i, rep) for v in VARIANTS for kv in SERVERS for i in range(len(PROMPTS)) for rep in range(REPS)]
random.Random(1).shuffle(jobs)
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lambda a: job(*a), jobs))
print("done", len(jobs))
