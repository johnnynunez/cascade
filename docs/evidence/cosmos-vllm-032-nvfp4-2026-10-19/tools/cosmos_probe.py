#!/usr/bin/env python3
"""Probe one running Cosmos3-Edge server: tool calls, image answers, decode, long-prompt TTFT.

    cosmos_probe.py --base http://127.0.0.1:8182/v1 --label C --out results/

Read-only: requests only, never executes a tool. Temperature 0, thinking off. Appends one JSON line per probe
to <out>/<probe>.jsonl:
- tools: 14 operator requests through CASCADE's own Cosmos client (the built-in agent's 42 tools, sent strict,
  one call per turn), and 8 OpenClaw-shaped requests (the attendee catalog as `cascade__*`, non-strict,
  tool_choice=auto). A case passes when the reply carries a native OpenAI tool call to an accepted tool whose
  arguments validate against that tool's schema.
- image: questions about a committed kitchen render and a side-camera frame; objective ones are scored.
- decode: --decode-runs (3) streamed runs of a long deterministic answer (512 tokens); decode tok/s after the
  first token.
- ttft: 3 streamed requests of ~16k prompt tokens (a 420-line robot log with distractor numbers and a fresh
  nonce, so no prefix-cache hit) holding one passkey line; TTFT and whether the passkey comes back.
- ttft-log: the same at ~30k tokens (800 lines), asking for one line's tag (harder; Cosmos3-Edge misses it
  with every KV cache).
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

MODEL = "cosmos3-edge"
OBS = {"get_observation", "describe_scene", "list_objects", "analyze_scene", "annotated_view", "localize_object",
       "locate_pixel", "count_objects"}
BUILTIN = [
    ("Pick up the red cube and put it in the drop zone.", {"pick_and_place", "get_observation", "preview_grasp"}),
    ("Where is the blue bowl?", {"localize_object", "get_observation", "locate_pixel"}),
    ("Go back to the home position.", {"move_home"}),
    ("How many cubes are there?", {"count_objects", "list_objects", "get_observation"}),
    ("Point at the green sponge.", {"point_at", "localize_object", "get_observation"}),
    ("Open the gripper.", {"open_gripper"}),
    ("Wave hello to the visitor.", {"wave"}),
    ("Put the cube you are holding down at x=0.2, y=-0.12.", {"place_at"}),
    ("Push the large box 5 cm away from you.", {"push_object", "localize_object", "get_observation"}),
    ("Move the arm a little to the left.", {"move_relative"}),
    ("Stop moving right now!", {"halt_motion"}),
    ("Find the scissors, they might be hidden under something.", {"search_for_object", "get_observation"}),
    ("Hand me the orange.", {"handover", "pick_and_place", "grasp_object", "get_observation", "localize_object"}),
    ("You are done, the task is complete.", {"task_done"}),
]
OPENCLAW = [
    ("What can you see on the table?", {"describe_scene", "camera_snapshot"}),
    ("Where is the orange?", {"localize_object"}),
    ("Put the green cube in the green square.", {"pick_and_place"}),
    ("Let's start over.", {"reset_scene"}),
    ("Show me the side camera.", {"camera_snapshot"}),
    ("What is the session status?", {"world_state"}),
    ("What did earlier runs teach you?", {"robot_knowledge"}),
    ("Move the tomato can to the open box.", {"pick_and_place"}),
]
OPENCLAW_SYSTEM = ("You are the assistant of a kitchen robot demo. Act through the cascade tools; "
                   "never invent results, call a tool instead.")
KITCHEN = REPO / "docs/evidence/project-status-20261003/kitchen-green-20261004.jpg"
SIDE = REPO / "docs/evidence/b31-isaac-same-colour-20261008/side_cam_bin_named_yellow.jpg"
IMAGE_QUESTIONS = [
    (KITCHEN, "Describe the objects on the kitchen island in one sentence.", None),
    (KITCHEN, ("Is the plate of oranges on the left, in the center or on the right of the image? "
               "Answer with one word."), ("right",)),
    (KITCHEN, ("What colour is the small cube standing between the red can and the yellow lemon? "
               "Answer with one word."), ("pink", "magenta")),
    (KITCHEN, "What colour are the robot arm's links? Answer with one word.", ("green",)),
    (SIDE, "What colour is the open box on the table? Answer with one word.", ("green",)),
]
DECODE_PROMPT = ("List, one per line and numbered, sixty distinct household objects a kitchen robot might be asked "
                 "to pick up, each with its typical colour and material. Do not stop before sixty.")


def post(base, body, timeout=300):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def stream(base, body, timeout=600):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    text, usage = "", None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    last = time.time()
                    first = first or last
                    text += piece
    return t0, first, last, text, usage or {}


def probe_tools(base):
    import jsonschema

    from cascade.agent.llm import make_llm
    from cascade.agent.prompts import SYSTEM_PROMPT
    from cascade.config import Cfg, load_demo_config
    from cascade.skills.runtime import tool_specs_for_config

    cfg = load_demo_config(arm="fr3", camera="d435i", llm="local_cosmos")
    profile = Cfg(dict(cfg.llm.as_dict(), base_url=base, temperature=0.0, request_timeout_s=120.0))
    tools = tool_specs_for_config(cfg)
    rows = []
    for prompt, accepted in BUILTIN:
        client = make_llm(profile)
        raw = {}
        inner = client._call_with_param_fallback

        def capture(kwargs, inner=inner, raw=raw):
            raw["response"] = inner(kwargs)
            return raw["response"]

        client._call_with_param_fallback = capture
        t0 = time.time()
        response = client.chat(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], tools=tools, max_tokens=512)
        message = raw["response"].choices[0].message
        native = [(c.function.name, c.function.arguments) for c in message.tool_calls or []]
        ok_names = accepted | OBS if "get_observation" in accepted else accepted
        validated = [c.name for c in response.tool_calls]  # the client drops calls failing their schema
        rows.append({"set": "builtin", "prompt": prompt, "accepted": sorted(ok_names), "native_calls": native,
                     "content": (message.content or "")[:200], "s": round(time.time() - t0, 2),
                     "ok": bool(native) and native[0][0] in ok_names and validated[:1] == [native[0][0]]})
    catalog = json.loads((REPO / "demo/kitchen/openclaw-plugin/attendee-tools.json").read_text())
    wire = [{"type": "function", "function": {"name": "cascade__" + t["name"], "description": t["description"],
                                              "parameters": t["parameters"]}} for t in catalog]
    schemas = {"cascade__" + t["name"]: t["parameters"] for t in catalog}
    for prompt, accepted in OPENCLAW:
        t0 = time.time()
        out = post(base, {"model": MODEL, "temperature": 0, "max_tokens": 512, "tools": wire, "tool_choice": "auto",
                          "chat_template_kwargs": {"enable_thinking": False},
                          "messages": [{"role": "system", "content": OPENCLAW_SYSTEM},
                                       {"role": "user", "content": prompt}]})
        message = out["choices"][0]["message"]
        native = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
        valid = False
        if native and native[0][0] in schemas:
            try:
                jsonschema.validate(json.loads(native[0][1] or "{}"), schemas[native[0][0]])
                valid = True
            except (ValueError, jsonschema.ValidationError):
                valid = False
        names = {"cascade__" + n for n in accepted}
        rows.append({"set": "openclaw", "prompt": prompt, "accepted": sorted(names), "native_calls": native,
                     "content": (message.get("content") or "")[:200], "s": round(time.time() - t0, 2),
                     "ok": bool(native) and native[0][0] in names and valid})
    return {"n": len(rows), "correct": sum(r["ok"] for r in rows),
            "native": sum(bool(r["native_calls"]) for r in rows),
            "builtin": sum(r["ok"] for r in rows if r["set"] == "builtin"),
            "openclaw": sum(r["ok"] for r in rows if r["set"] == "openclaw"), "rows": rows}


def probe_image(base):
    rows = []
    for path, question, expected in IMAGE_QUESTIONS:
        url = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
        out = post(base, {"model": MODEL, "temperature": 0, "max_tokens": 96,
                          "chat_template_kwargs": {"enable_thinking": False},
                          "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                                    {"type": "text", "text": question}]}]})
        answer = (out["choices"][0]["message"].get("content") or "").strip()
        rows.append({"image": path.name, "question": question, "answer": answer,
                     "prompt_tokens": out["usage"]["prompt_tokens"],
                     "correct": None if expected is None else any(e in answer.lower() for e in expected)})
    scored = [r for r in rows if r["correct"] is not None]
    return {"scored": len(scored), "correct": sum(r["correct"] for r in scored), "rows": rows}


def probe_decode(base, runs=3):
    rows = []
    for _ in range(runs):
        t0, first, last, _, usage = stream(base, {"model": MODEL, "temperature": 0, "max_tokens": 512,
                                                  "chat_template_kwargs": {"enable_thinking": False},
                                                  "messages": [{"role": "user", "content": DECODE_PROMPT}]})
        n = usage.get("completion_tokens") or 0
        rows.append({"completion_tokens": n, "ttft_s": round(first - t0, 3),
                     "decode_tps": round((n - 1) / (last - first), 2) if last and first and last > first else None})
    return {"decode_tps": [r["decode_tps"] for r in rows], "rows": rows}


def long_prompt(seed, lines, passkey):
    """A robot observation log with distractor numbers and a fresh nonce (no prefix-cache hit). passkey: one
    needle line holds a secret passkey to recall; otherwise the question asks for one line's tag."""
    rng = random.Random(seed)
    things = ["cube", "bowl", "lemon", "orange", "mug", "sponge", "plate", "can", "box", "spoon"]
    colours = ["red", "green", "blue", "yellow", "pink", "white", "black", "orange"]
    log = [f"Line {i:04d}: the robot saw a {rng.choice(colours)} {rng.choice(things)} with tag "
           f"{rng.randint(10000, 99999)} at x={rng.uniform(-0.5, 0.5):.3f}, y={rng.uniform(-0.5, 0.5):.3f}."
           for i in range(1, lines + 1)]
    target = rng.randint(lines // 4, lines - lines // 4)
    if passkey:
        answer = str(rng.randint(10000, 99999))
        log[target - 1] = f"Line {target:04d}: the operator said the secret passkey is {answer}. Remember it."
        question = "What is the secret passkey the operator said? Answer with the number only."
    else:
        answer = log[target - 1].split(" with tag ")[1].split(" ")[0]
        question = f"What tag number was recorded at Line {target:04d}? Answer with the number only."
    text = f"Session {seed:08x}. Robot observation log follows.\n" + "\n".join(log) + "\n\n" + question
    return text, answer


def probe_ttft(base, runs=3, lines=420, passkey=True):
    rows = []
    for _ in range(runs):
        seed = random.SystemRandom().getrandbits(32)
        text, answer = long_prompt(seed, lines, passkey)
        t0, first, _last, out, usage = stream(base, {"model": MODEL, "temperature": 0, "max_tokens": 16,
                                                    "chat_template_kwargs": {"enable_thinking": False},
                                                    "messages": [{"role": "user", "content": text}]})
        rows.append({"prompt_tokens": usage.get("prompt_tokens"), "ttft_s": round(first - t0, 3),
                     "answer": out.strip(), "expected": answer, "correct": answer in out})
    return {"ttft_s": [r["ttft_s"] for r in rows], "correct": sum(r["correct"] for r in rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probes", default="tools,image,decode,ttft,ttft-log")
    parser.add_argument("--decode-runs", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    probes = {"tools": probe_tools, "image": probe_image, "decode": lambda base: probe_decode(base, args.decode_runs),
              "ttft": probe_ttft,
              "ttft-log": lambda base: probe_ttft(base, lines=800, passkey=False)}
    for name in args.probes.split(","):
        result = probes[name](args.base)
        with (args.out / f"{name}.jsonl").open("a") as stream_out:
            stream_out.write(json.dumps({"label": args.label, "probe": name, **result}) + "\n")
        print(args.label, name, json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
