#!/usr/bin/env python3
"""Tool-call A/B of two Cosmos3-Edge chat templates on one running vLLM server.

    template_probe.py --base URL --label hub|json --out results.jsonl

Requests are non-strict tools with tool_choice=auto, temperature 0, thinking off, served with
--tool-strict-level auto, so no grammar constrains the output: only the template differs.
A case passes when the reply carries a native tool call to the expected tool whose JSON arguments
validate against that tool's schema.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import jsonschema

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}}]
WEATHER_CASES = [
    ("What's the weather in Paris right now, in celsius?", "get_weather"),
    ("Is it raining in Oslo?", "get_weather"),
    ("Check the weather in Tokyo in fahrenheit.", "get_weather"),
    ("How warm is it in Lima today?", "get_weather"),
]
ROBOT_CASES = [
    ("What can you see on the table?", {"describe_scene", "camera_snapshot"}),
    ("Where is the orange?", {"localize_object"}),
    ("Put the green cube in the green square.", {"pick_and_place"}),
    ("Let's start over.", {"reset_scene"}),
    ("Show me the side camera.", {"camera_snapshot"}),
    ("What is the session status?", {"world_state"}),
    ("What did earlier runs teach you?", {"robot_knowledge"}),
    ("Move the tomato can to the open box.", {"pick_and_place"}),
]


def post(base, body):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def case(base, tools, prompt, accepted, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    out = post(base, {"model": "cosmos3-edge", "temperature": 0, "max_tokens": 256, "tools": tools,
                      "tool_choice": "auto", "chat_template_kwargs": {"enable_thinking": False},
                      "messages": messages})
    message = out["choices"][0]["message"]
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    valid = False
    if calls and calls[0][0] in accepted:
        try:
            jsonschema.validate(json.loads(calls[0][1] or "{}"), schemas[calls[0][0]])
            valid = True
        except (ValueError, jsonschema.ValidationError):
            pass
    return {"prompt": prompt, "calls": calls, "content": (message.get("content") or "")[:400],
            "prompt_tokens": out["usage"]["prompt_tokens"], "ok": valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    robot = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["parameters"]}}
             for t in json.loads(args.catalog.read_text())]
    rows = [dict(case(args.base, WEATHER, p, {name}), set="weather") for p, name in WEATHER_CASES]
    # The raw text behind the weather calls: tool_choice=none keeps the tools in the prompt, parses nothing.
    raw = [post(args.base, {"model": "cosmos3-edge", "temperature": 0, "max_tokens": 128, "tools": WEATHER,
                            "tool_choice": "none", "chat_template_kwargs": {"enable_thinking": False},
                            "messages": [{"role": "user", "content": p}]})["choices"][0]["message"]
           for p, _ in WEATHER_CASES]
    raw = [m.get("content") or m.get("reasoning_content") or m.get("reasoning") for m in raw]
    rows += [dict(case(args.base, robot, p, names, "You are a kitchen robot assistant. Act through the tools."),
                  set="robot") for p, names in ROBOT_CASES]
    with args.out.open("a") as f:
        f.write(json.dumps({"label": args.label, "correct": sum(r["ok"] for r in rows), "n": len(rows),
                            "rows": rows, "raw_weather_text": raw}) + "\n")
    print(args.label, f"{sum(r['ok'] for r in rows)}/{len(rows)}",
          "weather", sum(r["ok"] for r in rows if r["set"] == "weather"),
          "robot", sum(r["ok"] for r in rows if r["set"] == "robot"), flush=True)


if __name__ == "__main__":
    main()
