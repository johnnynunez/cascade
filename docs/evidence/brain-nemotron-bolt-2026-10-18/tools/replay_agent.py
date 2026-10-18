#!/usr/bin/env python3
"""Replay the Nemotron Bolt trial's real OpenClaw agent calls against a brain and time them.

The trial's measurement proxy saved every agent request body (v5-bolt/proxy/bodies/*.json.gz). One recorded
session = calls 41, 43, 44, 47, 49, 51 (13.6k -> 18.2k prompt tokens, one camera image, 49 tool schemas): each
call's messages extend the previous call's, exactly as OpenClaw sends them. A replay renames the session (the
`session=...` / `sessionId=...` ids in the system prompt's Runtime line) with a fresh nonce, so the fixed
system+tools prefix is shared across replays the way it is across real sessions while the session tail changes.

Per call: TTFT (first streamed content/reasoning/tool-call delta), total time, usage.prompt_tokens, and the
prefix-cache tokens the server hit for this request (vLLM /metrics counter deltas; requests run one at a time).

    replay_agent.py --base http://127.0.0.1:18491/v1 --key-file K --model nemotron-3.5-bolt --replays 2 --out f.jsonl
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import re
import secrets
import time
import urllib.request
import uuid
from pathlib import Path

BODIES = Path("/home/spark/.hermes/cache/scratch/demo-video/v5-bolt/proxy/bodies")
SESSION = (41, 43, 44, 47, 49, 51)
_SESSION_RE = re.compile(r"(session=agent:main:[a-z]+:)([0-9a-f-]{36})|(sessionId=)([0-9a-f-]{36})")


def load_session(seqs=SESSION) -> list[dict]:
    return [json.loads(gzip.open(BODIES / f"{s:05d}.json.gz").read()) for s in seqs]


def rename_session(body: dict, mapping: dict[str, str]) -> dict:
    out = copy.deepcopy(body)
    system = out["messages"][0]
    assert system["role"] == "system" and isinstance(system["content"], str)

    def sub(m):
        prefix = m.group(1) or m.group(3)
        old = m.group(2) or m.group(4)
        new = mapping.setdefault(old, str(uuid.uuid4()))
        return prefix + new

    renamed = _SESSION_RE.sub(sub, system["content"])
    assert renamed != system["content"], "no session id found to rename"
    system["content"] = renamed
    return out


def metrics(base: str, key: str) -> dict[str, float]:
    url = base.rsplit("/v1", 1)[0] + "/metrics"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        text = r.read().decode()
    found = {}
    for name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                 "vllm:prompt_tokens_total", "vllm:mm_cache_queries_total", "vllm:mm_cache_hits_total"):
        total = 0.0
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                total += float(line.rsplit(" ", 1)[1])
        found[name] = total
    return found


def call(base: str, key: str, model: str, body: dict, max_tokens: int, kwargs: dict | None) -> dict:
    payload = {k: v for k, v in body.items() if k not in ("max_completion_tokens", "max_tokens")}
    payload.update(model=model, stream=True, stream_options={"include_usage": True},
                   max_completion_tokens=max_tokens, temperature=0)
    if kwargs is not None:
        payload["chat_template_kwargs"] = kwargs
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    first = None
    usage = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                                      or d.get("tool_calls")):
                    first = time.time()
                n_chunks += 1
    t1 = time.time()
    return {"ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 3),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", default="nemotron-3.5-bolt")
    ap.add_argument("--replays", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--label", default="")
    ap.add_argument("--kwargs", default=None, help='JSON chat_template_kwargs to send, e.g. {"enable_thinking": false}')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    kwargs = json.loads(args.kwargs) if args.kwargs else None
    session = load_session()
    out = Path(args.out)
    for replay in range(args.replays):
        mapping: dict[str, str] = {}
        nonce = secrets.token_hex(4)
        for index, (seq, body) in enumerate(zip(SESSION, session)):
            before = metrics(args.base, key)
            row = call(args.base, key, args.model, rename_session(body, mapping), args.max_tokens, kwargs)
            after = metrics(args.base, key)
            row.update(label=args.label, replay=replay, nonce=nonce, call=index, trial_seq=seq, t=time.time(),
                       prefix_queried=after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"],
                       prefix_hit=after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"])
            print(json.dumps(row), flush=True)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
