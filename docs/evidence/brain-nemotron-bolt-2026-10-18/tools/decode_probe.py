#!/usr/bin/env python3
"""Decode speed and short-prompt TTFT of an OpenAI-compatible brain (vLLM or llama-server), streamed, thinking off.

    decode_probe.py --base URL --key-file K --model ID --label L --out f.jsonl [--n 3] [--max-tokens 256]

decode_tps = (completion_tokens - 1) / (last token time - first token time); the prompt asks for a long, deterministic
answer (temperature 0) so every run decodes the full budget. spec acceptance comes from the server (vLLM /metrics
spec-decode counters or llama-server `timings.draft_n` / `draft_n_accepted`) when it reports them.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

PROMPT = ("List, one per line and numbered, sixty distinct household objects a kitchen robot might be asked to pick up, "
          "each with its typical colour and material. Do not stop before sixty.")


def metrics(base: str, key: str) -> dict[str, float]:
    req = urllib.request.Request(base.rstrip("/").removesuffix("/v1") + "/metrics",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        text = urllib.request.urlopen(req, timeout=20).read().decode()
    except Exception:  # noqa: BLE001 - llama-server without --metrics
        return {}
    out: dict[str, float] = {}
    for name in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"):
        out[name] = sum(float(m) for m in re.findall(rf"^{re.escape(name)}{{[^}}]*}} ([0-9.e+]+)$", text, re.M))
    return out


def one(base: str, key: str, model: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    first = last = None
    usage = timings = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    last = time.time()
                    first = first or last
    n = (usage or {}).get("completion_tokens") or 0
    return {"ttft_s": round((first or time.time()) - t0, 3), "completion_tokens": n,
            "decode_tps": round((n - 1) / (last - first), 2) if first and last and last > first and n > 1 else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "llama_draft_n": (timings or {}).get("draft_n"), "llama_draft_accepted": (timings or {}).get("draft_n_accepted"),
            "llama_predicted_per_second": (timings or {}).get("predicted_per_second")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    for i in range(args.n):
        before = metrics(args.base, key)
        row = one(args.base, key, args.model, args.max_tokens)
        after = metrics(args.base, key)
        drafted = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get("vllm:spec_decode_num_draft_tokens_total", 0)
        accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        row.update(label=args.label, run=i, vllm_spec_drafted=drafted or None, vllm_spec_accepted=accepted or None)
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
