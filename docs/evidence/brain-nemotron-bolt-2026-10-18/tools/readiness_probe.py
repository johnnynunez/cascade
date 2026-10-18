#!/usr/bin/env python3
"""Ask Bolt the launcher's readiness question inside OpenClaw's real agent prompt (trial proxy body 00005).

Variants: main's question with/without chat_template_kwargs, and the PR's Bolt question. Streaming, as OpenClaw sends.
"""
import gzip, json, sys, time, urllib.request, copy
from pathlib import Path

BASE = sys.argv[1]
KEY = Path(sys.argv[2]).read_text().strip()
OUT = Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
body = json.loads(gzip.open("/home/spark/.hermes/cache/scratch/demo-video/v5-bolt/proxy/bodies/00005.json.gz").read())
MAIN = "Reply with exactly OK and nothing else. Do not call tools."
BOLT = ("Readiness check: a visible reply is required, so NO_REPLY or an empty reply fails it. "
        "Reply with exactly OK and nothing else. Do not call tools.")
variants = {"main-no-kwargs": (MAIN, None), "main-kwargs": (MAIN, {"enable_thinking": False}),
            "bolt-kwargs": (BOLT, {"enable_thinking": False})}
for name, (question, kwargs) in variants.items():
    for i in range(N):
        b = copy.deepcopy(body)
        b["messages"][-1]["content"] = f"[Sun 2026-10-18 07:48 UTC] {question}"
        if kwargs is not None:
            b["chat_template_kwargs"] = kwargs
        req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        t0 = time.time()
        content, reasoning, tools = "", "", 0
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                ev = json.loads(line[5:])
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    content += d.get("content") or ""
                    reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
                    tools += len(d.get("tool_calls") or [])
        row = {"variant": name, "i": i, "content": content, "reasoning_chars": len(reasoning), "tool_deltas": tools,
               "s": round(time.time() - t0, 2)}
        print(json.dumps(row), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
