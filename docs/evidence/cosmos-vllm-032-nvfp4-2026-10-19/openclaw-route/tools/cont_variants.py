#!/usr/bin/env python3
"""Continuation-turn variants on ONE captured live continuation request
(requests-profile/<n>.json). usage: cont_variants.py <n> [reps]"""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_matrix_lib import send  # noqa: E402

BASE = "http://127.0.0.1:8083/v1"
D = Path(__file__).parent
req = json.loads((D / f"requests-profile/{sys.argv[1]}.json").read_text())["request"]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
GDONE = ("You check a robot arm's work. The user's request and the tool results follow. Did the tool results "
         "already do everything the request asks for? Answer only YES or NO.")
GLEFT = ("You check a robot arm's work. The user's request and the tool results follow. Does the request ask for "
         "something that the tool results do not show as done? Answer only YES or NO.")


def edit_tool(r, fn):
    r = copy.deepcopy(r)
    for m in r["messages"]:
        if m["role"] == "tool":
            v = json.loads(m["content"])
            m["content"] = json.dumps(fn(v))
    return r


def text_form(r, header="Tool calls made so far:", tail=""):
    r = copy.deepcopy(r)
    msgs = r["messages"]
    user = next(m for m in msgs if m["role"] == "user")["content"]
    calls, lines = {}, []
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                calls[c["id"]] = c["function"]
        if m["role"] == "tool":
            f = calls.get(m.get("tool_call_id"), {"name": "?", "arguments": "{}"})
            lines.append(f"{len(lines) + 1}. {f['name']}({f['arguments']}) returned {m['content']}")
    r["messages"] = [msgs[0], {"role": "user", "content": f"Request: {user}\n\n{header}\n" + "\n".join(lines) + tail}]
    return r


def gate(r, prompt):
    g = text_form(r, "Tool results:")
    g["messages"][0] = {"role": "system", "content": prompt}
    for k in ("tools", "tool_choice", "parallel_tool_calls"):
        g.pop(k, None)
    g["max_completion_tokens"] = 4
    return g


V = {
    "asis": lambda r: r,
    "nonext": lambda r: edit_tool(r, lambda v: {k: x for k, x in v.items() if k != "next_action"}),
    "nopost": lambda r: edit_tool(r, lambda v: {k: x for k, x in v.items() if k not in ("postcondition",)}),
    "okonly": lambda r: edit_tool(r, lambda v: {k: x for k, x in v.items() if k in ("ok", "picked", "destination", "placed_at", "error")}),
    "textform": lambda r: text_form(r),
}
for name, fn in V.items():
    c = Counter()
    for _ in range(reps):
        out = send(BASE, fn(req))
        c[(out["calls"][0]["name"] + out["calls"][0]["arguments"]) if out["calls"] else "TEXT:" + out["text"][:50]] += 1
    print(f"{name:10} {dict(c)}")
for name, p in (("gate_done", GDONE), ("gate_left", GLEFT)):
    c = Counter(send(BASE, gate(req, p))["text"].strip()[:10] for _ in range(reps))
    print(f"{name:10} {dict(c)}")
