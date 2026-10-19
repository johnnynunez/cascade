#!/usr/bin/env python3
"""Report-turn variants for a multi-call request (payload dumped by
replay_loop2.mjs). usage: report_variants.py <payload.json>..."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_matrix_lib import send  # noqa: E402

BASE = "http://127.0.0.1:8083/v1"
REPLY = ("You control a robot arm. A tool ran for the user's request; its result follows. In one short English "
         "sentence, tell the user what happened: if ok is true, what was done; if ok is false, that it failed and "
         "why. Report only what this result shows.")
REPORT = ("You control a robot arm. Tools ran for the user's request; their results follow, in order. In one or two "
          "short English sentences, tell the user what happened: for each result with ok true, what was done; for a "
          "result with ok false, that it failed and why. Report only what these results show.")


def as_text(p, prompt):
    """user request + every call and result as ONE user text block"""
    msgs = p["messages"]
    user = next(m for m in msgs if m["role"] == "user")
    lines, n = [], 0
    calls = {}
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                calls[c["id"]] = c["function"]
        if m["role"] == "tool":
            n += 1
            f = calls.get(m.get("tool_call_id"), {"name": "?", "arguments": "{}"})
            lines.append(f"{n}. {f['name']}({f['arguments']}) returned {m['content']}")
    q = copy.deepcopy(p)
    q["messages"] = [{"role": "system", "content": prompt},
                     {"role": "user", "content": f"Request: {user['content']}\n\nTool results:\n" + "\n".join(lines)}]
    return q


def last_only(p, prompt):
    q = copy.deepcopy(p)
    msgs = q["messages"]
    user = next(m for m in msgs if m["role"] == "user")
    i = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    q["messages"] = [{"role": "system", "content": prompt}, user] + msgs[i:]
    return q


def sysp(p, prompt):
    q = copy.deepcopy(p)
    q["messages"][0]["content"] = prompt
    return q


def nofp(q):
    q = copy.deepcopy(q)
    q.pop("frequency_penalty", None)
    return q


V = {
    "full_REPORT": lambda p: sysp(p, REPORT),
    "full_REPLY": lambda p: sysp(p, REPLY),
    "full_REPORT_nofp": lambda p: nofp(sysp(p, REPORT)),
    "text_REPORT": lambda p: as_text(p, REPORT),
    "text_REPORT_nofp": lambda p: nofp(as_text(p, REPORT)),
    "last_REPLY": lambda p: last_only(p, REPLY),
}
for f in sys.argv[1:]:
    p = json.loads(Path(f).read_text())
    print("==", Path(f).name, "|", next(m for m in p["messages"] if m["role"] == "user")["content"])
    for name, fn in V.items():
        q = fn(p)
        q["stream"] = False
        q.pop("stream_options", None)
        r = send(BASE, q)
        print(f"  {name:18} {r['finish']:6} {json.dumps(r['text'].strip())[:200]}")
