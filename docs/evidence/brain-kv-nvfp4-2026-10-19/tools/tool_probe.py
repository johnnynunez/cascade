#!/usr/bin/env python3
"""Tool-call probe: does the brain pick the right CASCADE tool, with parseable arguments?

    tool_probe.py --base URL --model ID --label L --out f.jsonl

30 operator requests, each with CASCADE's full tool list (skills.runtime.TOOL_SPECS, OpenAI function format) and the
built-in agent's system prompt; one turn, temperature 0, thinking off. A case passes when the first tool call names an
accepted tool and its arguments parse as JSON.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, "/home/johnny/Projects/cascade-wt/nvfp4-kv/src")
from cascade.agent.prompts import SYSTEM_PROMPT  # noqa: E402
from cascade.skills.runtime import TOOL_SPECS  # noqa: E402

CASES = [
    ("Pick up the red cube and put it in the drop zone.", {"pick_and_place", "get_observation", "list_objects", "preview_grasp", "localize_object"}),
    ("Where is the blue bowl?", {"localize_object", "get_observation", "list_objects", "describe_scene"}),
    ("Go back to the home position.", {"move_home"}),
    ("What objects can you see on the table?", {"list_objects", "get_observation", "describe_scene"}),
    ("How many cubes are there?", {"count_objects", "list_objects", "get_observation"}),
    ("Point at the green sponge.", {"point_at", "localize_object", "get_observation"}),
    ("Grab the yellow lemon.", {"grasp_object", "pick_and_place", "get_observation", "localize_object", "preview_grasp"}),
    ("Open the gripper.", {"open_gripper"}),
    ("Close the gripper.", {"close_gripper"}),
    ("Wave hello to the visitor.", {"wave"}),
    ("Describe the scene in front of you.", {"describe_scene", "get_observation"}),
    ("Put the cube you are holding down at x=0.2, y=-0.12.", {"place_at"}),
    ("Push the large box 5 cm away from you.", {"push_object", "localize_object", "get_observation"}),
    ("Take a fresh look at the table.", {"get_observation", "describe_scene", "list_objects"}),
    ("Show me how you would grasp the red mug, without moving.", {"preview_grasp", "localize_object"}),
    ("Sort the cubes by colour.", {"sort_by_color", "get_observation", "list_objects"}),
    ("Put the red cube on top of the blue cube.", {"pick_and_place", "grasp_object", "get_observation", "list_objects", "localize_object"}),
    ("Is the drop zone empty?", {"get_observation", "describe_scene", "list_objects", "localize_object"}),
    ("Move the arm a little to the left.", {"move_relative"}),
    ("Stop moving right now!", {"halt_motion"}),
    ("Find the scissors, they might be hidden under something.", {"search_for_object", "get_observation", "list_objects", "localize_object"}),
    ("Count the red objects.", {"count_objects", "list_objects", "get_observation"}),
    ("Hold the cube near the camera so I can see it.", {"move_near_object", "get_observation", "grasp_object", "pick_and_place"}),
    ("Hand me the orange.", {"handover", "pick_and_place", "grasp_object", "get_observation", "localize_object"}),
    ("Which object is closest to the robot?", {"list_objects", "get_observation", "describe_scene", "localize_object"}),
    ("Put everything back where it was.", {"restore_scene", "get_observation"}),
    ("Tell me what you are holding.", {"get_observation", "describe_scene"}),
    ("Pick up the blue cube.", {"grasp_object", "pick_and_place", "get_observation", "localize_object", "preview_grasp"}),
    ("Place the held object on the plate.", {"place_on_object", "localize_object", "get_observation"}),
    ("You are done, the task is complete.", {"task_done"}),
]


def tools() -> list[dict]:
    return [{"type": "function", "function": {"name": s["name"], "description": s.get("description", ""),
                                              "parameters": s.get("input_schema") or s.get("parameters")
                                              or {"type": "object", "properties": {}}}} for s in TOOL_SPECS]


def one(base: str, model: str, prompt: str) -> dict:
    body = {"model": model, "temperature": 0.0, "max_tokens": 512, "tools": tools(), "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t0 = time.monotonic()
    out = json.loads(urllib.request.urlopen(req, timeout=300).read())
    msg = out["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    name = calls[0]["function"]["name"] if calls else None
    try:
        args_ok = bool(calls) and isinstance(json.loads(calls[0]["function"]["arguments"] or "{}"), dict)
    except (ValueError, TypeError):
        args_ok = False
    return {"name": name, "args_ok": args_ok, "s": round(time.monotonic() - t0, 2),
            "content": (msg.get("content") or "")[:120], "prompt_tokens": out.get("usage", {}).get("prompt_tokens")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(lambda c: one(args.base, args.model, c[0]), CASES))
    obs = {"get_observation", "describe_scene", "list_objects", "analyze_scene", "annotated_view", "snapshot_scene",
           "localize_object", "count_objects"}
    accepted = [ok | obs if "get_observation" in ok else ok for _, ok in CASES]
    rows = [{"prompt": p, "accepted": sorted(ok), **r, "ok": r["name"] in ok and r["args_ok"]}
            for (p, _), ok, r in zip(CASES, accepted, res, strict=True)]
    correct = sum(r["ok"] for r in rows)
    with args.out.open("a") as f:
        f.write(json.dumps({"label": args.label, "probe": "tools", "n": len(rows), "correct": correct,
                            "with_call": sum(1 for r in rows if r["name"]), "rows": rows}) + "\n")
    print(args.label, "tools", f"{correct}/{len(rows)}", "with a call:", sum(1 for r in rows if r["name"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
