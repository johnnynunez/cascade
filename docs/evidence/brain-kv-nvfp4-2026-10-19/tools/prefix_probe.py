#!/usr/bin/env python3
"""Prefix-cache reuse on agent-shaped calls: one ~15k-token prompt, then follow-ups that extend it like agent turns.

    prefix_probe.py --base URL --model ID --label L --out f.jsonl [--turns 4]

Per call: TTFT (streamed), prompt tokens, and the prefix-cache hit tokens the server reports for it (vLLM /metrics
`vllm:prefix_cache_hits_total` delta; calls run one at a time). Two sessions with different nonces: the second shares
only the system+tools prefix with the first, as a new visitor's session would.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import secrets
import sys
import time
import urllib.request

sys.path.insert(0, "/home/johnny/Projects/cascade-wt/nvfp4-kv/src")
from cascade.agent.prompts import SYSTEM_PROMPT  # noqa: E402
from cascade.skills.runtime import TOOL_SPECS  # noqa: E402


def metric(base: str, name: str) -> float:
    text = urllib.request.urlopen(base.rstrip("/").removesuffix("/v1") + "/metrics", timeout=20).read().decode()
    return sum(float(m) for m in re.findall(rf"^{name}(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M))


def tools() -> list[dict]:
    return [{"type": "function", "function": {"name": s["name"], "description": s.get("description", ""),
                                              "parameters": s.get("parameters") or {"type": "object", "properties": {}}}}
            for s in TOOL_SPECS]


def turn_text(i: int) -> str:
    rows = "; ".join(f"object {j}: a {c} item at x={0.1 + 0.01 * j:.2f} y={-0.1 + 0.007 * j:.3f}"
                     for j, c in zip(range(40), ["red", "blue", "green", "yellow"] * 10, strict=False))
    return f"Observation {i}: the camera reports {rows}. The last tool result was ok and verified."


def ttft(base: str, model: str, messages: list[dict]) -> tuple[float, int]:
    body = {"model": model, "messages": messages, "tools": tools(), "max_tokens": 16, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t0 = time.monotonic()
    first = None
    prompt = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                prompt = chunk["usage"].get("prompt_tokens", prompt)
            if first is None and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta") or {}
                if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
                    first = time.monotonic() - t0
    return (first if first is not None else time.monotonic() - t0), prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--turns", type=int, default=4)
    args = ap.parse_args()
    rows = []
    for session in range(2):
        nonce = secrets.token_hex(8)
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + "\n".join(turn_text(-k) for k in range(18))},
                    {"role": "user", "content": f"[session {nonce}] Put the red cube in the drop zone."}]
        for t in range(args.turns):
            h0 = metric(args.base, "vllm:prefix_cache_hits_total")
            s, prompt = ttft(args.base, args.model, messages)
            hits = metric(args.base, "vllm:prefix_cache_hits_total") - h0
            rows.append({"session": session, "turn": t, "ttft_s": round(s, 3), "prompt_tokens": prompt,
                         "hit_tokens": int(hits)})
            print(args.label, rows[-1], flush=True)
            messages += [{"role": "assistant", "content": f"Calling get_observation (turn {t})."},
                         {"role": "user", "content": turn_text(t)}]
    with args.out.open("a") as f:
        f.write(json.dumps({"label": args.label, "probe": "prefix", "rows": rows}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
