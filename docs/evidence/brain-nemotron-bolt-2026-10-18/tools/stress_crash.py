#!/usr/bin/env python3
"""Stress a brain with the load shape that crashed vLLM 0.32.0 (Xid 31): mixed batches of
one or more decodes (cascade perception inventories, images) + a new agent prefill chunk.

Both recorded faults hit a step with 1 decode + a new agent prefill chunk (12:39: 1 + 2047 tokens, no prefix hit;
13:49: 1 + 192 tokens after a 16192-token prefix hit). This keeps that state frequent:
  - P perception threads loop cascade's own BrainDetector requests (2 camera inventories, a REFER), 0.3 s apart;
  - A agent threads replay the recorded OpenClaw session (6 calls, 13.6k -> 18.2k tokens), alternating a FRESH
    session (nonce at the very start of the system prompt: full prefill, no hit) and a RENAMED one (prefix hits);
  - a monitor polls /health every 5 s; two failures in a row (or connection refused) = crash -> stop.

    stress_crash.py --base URL --key-file K --out f.jsonl [--duration 900] [--agents 2] [--perception 2]
Exit 0 = survived the duration, 1 = the server died.
"""
from __future__ import annotations

import argparse
import copy
import json
import secrets
import threading
import time
import urllib.request
from pathlib import Path

import replay_agent as ra
import replay_mixed as rm


def healthy(base: str) -> bool:
    try:
        return urllib.request.urlopen(base.removesuffix("/v1") + "/health", timeout=10).status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nemotron-3.5-bolt")
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--perception", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--agent-max-tokens", type=int, default=32)
    ap.add_argument("--agent-default-sampling", action="store_true",
                    help="send no temperature on agent calls (OpenClaw's own requests carry none)")
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    out = Path(args.out)
    lock = threading.Lock()
    stop = threading.Event()
    state = {"crashed_at": None, "calls": 0, "errors": 0, "started": time.time()}
    session = ra.load_session()
    perception = [rm.load(s) for s in rm.PERCEPTION]

    def write(row: dict) -> None:
        row.update(label=args.label, t=time.time())
        with lock:
            state["calls"] += 1
            state["errors"] += 1 if "error" in row else 0
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def call(body: dict) -> dict:
        try:
            return rm.timed(args.base, key, args.model, body)
        except Exception as exc:  # noqa: BLE001 - record and continue; the monitor decides about a crash
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    def agent(idx: int) -> None:
        time.sleep(1.5 + 2.7 * idx)
        n = 0
        while not stop.is_set():
            fresh = n % 2 == 0
            nonce = secrets.token_hex(6)
            mapping: dict[str, str] = {}
            for index, body in enumerate(session):
                if stop.is_set():
                    return
                b = ra.rename_session(body, mapping)
                if fresh:
                    b = copy.deepcopy(b)
                    b["messages"][0]["content"] = f"[run {nonce}]\n" + b["messages"][0]["content"]
                b = {k: v for k, v in b.items() if k not in ("max_completion_tokens", "max_tokens")}
                b.update(stream=True, max_completion_tokens=args.agent_max_tokens,
                         chat_template_kwargs={"enable_thinking": False})
                if not args.agent_default_sampling:
                    b["temperature"] = 0   # else the server's default sampling, as OpenClaw sends (top-p kernels)
                row = call(b)
                write({"who": f"agent{idx}", "session": n, "fresh": fresh, "call": index, **row})
                time.sleep(0.5)
            n += 1

    def watcher(idx: int) -> None:
        time.sleep(0.5 * idx)
        i = idx
        while not stop.is_set():
            name = rm.PERCEPTION[i % len(rm.PERCEPTION)]
            row = call(perception[i % len(rm.PERCEPTION)])
            write({"who": f"perception{idx}", "request": name, **row})
            i += 1
            stop.wait(0.3)

    def monitor() -> None:
        bad = 0
        while not stop.is_set():
            if healthy(args.base):
                bad = 0
            else:
                bad += 1
                if bad >= 2:
                    state["crashed_at"] = time.time()
                    stop.set()
                    return
            if time.time() - state["started"] > args.duration:
                stop.set()
                return
            stop.wait(5)

    threads = [threading.Thread(target=monitor, daemon=True)]
    threads += [threading.Thread(target=watcher, args=(i,), daemon=True) for i in range(args.perception)]
    threads += [threading.Thread(target=agent, args=(i,), daemon=True) for i in range(args.agents)]
    for t in threads:
        t.start()
    threads[0].join()
    for t in threads[1:]:
        t.join(timeout=60)
    summary = {"summary": True, "label": args.label, "crashed": state["crashed_at"] is not None,
               "crashed_after_s": round(state["crashed_at"] - state["started"], 1) if state["crashed_at"] else None,
               "ran_s": round(time.time() - state["started"], 1), "calls": state["calls"], "errors": state["errors"]}
    print(json.dumps(summary), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    return 1 if summary["crashed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
