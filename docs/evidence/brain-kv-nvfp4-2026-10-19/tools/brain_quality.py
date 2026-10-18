#!/usr/bin/env python3
"""Accuracy probes of an OpenAI-compatible brain (thinking off, temperature 0).

    brain_quality.py --base URL --model ID --label L --out f.jsonl [--gsm8k N] [--needle N]

- gsm8k: the first N problems of the GSM8K test split, 4 at a time; the answer is the last number in the reply.
- needle: N passkey retrievals at depths 10..90 % of a ~CTX_TOKENS filler context (KV quantization error grows with
  context length; this is where an NVFP4 cache would lose first).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time
import urllib.request

GSM8K = Path(__file__).with_name("gsm8k_test.jsonl")


def chat(base: str, model: str, messages: list[dict], max_tokens: int, timeout: float = 600) -> tuple[str, dict, float]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t0 = time.monotonic()
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return out["choices"][0]["message"].get("content") or "", out.get("usage", {}), time.monotonic() - t0


def last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return None
    n = nums[-1].replace(",", "").rstrip(".")
    try:
        f = float(n)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def gsm8k(base: str, model: str, n: int) -> dict:
    rows = [json.loads(line) for line in GSM8K.read_text().splitlines()[:n]]

    def one(row):
        prompt = row["question"] + "\nSolve it step by step, then give the final answer as a number on the last line."
        text, usage, dt = chat(base, model, [{"role": "user", "content": prompt}], 768)
        gold = last_number(row["answer"].split("####")[-1])
        return {"ok": last_number(text) == gold, "gold": gold, "got": last_number(text),
                "tokens": usage.get("completion_tokens"), "s": round(dt, 2)}

    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(one, rows))
    return {"probe": "gsm8k", "n": len(res), "correct": sum(r["ok"] for r in res),
            "accuracy": round(sum(r["ok"] for r in res) / len(res), 4), "rows": res}


FILLER = ("The kitchen robot logs every object it sees: a red mug on the counter, a blue bowl near the sink, "
          "a green sponge beside the tap, a yellow lemon in the fruit basket, and a white plate on the rack. ")


def needle(base: str, model: str, n: int, ctx_tokens: int) -> dict:
    rng = random.Random(7)
    reps = max(1, ctx_tokens // 48)
    res = []
    for i in range(n):
        key = f"{rng.randrange(10**7, 10**8)}"
        depth = 0.1 + 0.8 * i / max(1, n - 1)
        chunks = [FILLER] * reps
        chunks.insert(int(depth * reps), f"IMPORTANT: the access code for the pantry is {key}. Remember it. ")
        prompt = "".join(chunks) + "\nWhat is the access code for the pantry? Reply with the number only."
        text, usage, dt = chat(base, model, [{"role": "user", "content": prompt}], 16)
        res.append({"depth": round(depth, 2), "ok": key in text, "prompt_tokens": usage.get("prompt_tokens"),
                    "s": round(dt, 2), "reply": text.strip()[:40]})
    return {"probe": "needle", "n": n, "correct": sum(r["ok"] for r in res), "rows": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gsm8k", type=int, default=200)
    ap.add_argument("--needle", type=int, default=9)
    ap.add_argument("--needle-tokens", type=int, default=56000)
    args = ap.parse_args()
    results = []
    if args.gsm8k:
        results.append(gsm8k(args.base, args.model, args.gsm8k))
    if args.needle:
        results.append(needle(args.base, args.model, args.needle, args.needle_tokens))
    with args.out.open("a") as f:
        for r in results:
            f.write(json.dumps({"label": args.label, **r}) + "\n")
            print(args.label, r["probe"], f"{r['correct']}/{r['n']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
