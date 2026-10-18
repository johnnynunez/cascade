#!/usr/bin/env python3
"""Find what crashes Bolt on vLLM 0.32.0: run steps one at a time, health-checking after each.

    repro_crash.py --base URL --key-file K --out f.jsonl
Steps: text agent call alone; image inventory alone (x2); REFER alone; two text calls at once; agent + image at once.
"""
from __future__ import annotations

import argparse
import json
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
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    session = ra.load_session()
    agent = ra.rename_session(session[0], {})
    agent = {k: v for k, v in agent.items() if k not in ("max_completion_tokens", "max_tokens")}
    agent.update(stream=True, max_completion_tokens=32, temperature=0, chat_template_kwargs={"enable_thinking": False})
    agent2 = ra.rename_session(session[0], {})
    agent2 = {k: v for k, v in agent2.items() if k not in ("max_completion_tokens", "max_tokens")}
    agent2.update(stream=True, max_completion_tokens=32, temperature=0, chat_template_kwargs={"enable_thinking": False})
    inv, refer = rm.load("inventory-cam0"), rm.load("refer-green-cube")
    short = {"messages": [{"role": "user", "content": "Reply with exactly OK"}], "stream": False, "max_tokens": 8,
             "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}

    def call(body):
        try:
            return rm.timed(args.base, key, args.model, body)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    def together(*bodies):
        out = [None] * len(bodies)

        def go(i, b):
            out[i] = call(b)
        threads = [threading.Thread(target=go, args=(i, b)) for i, b in enumerate(bodies)]
        for t in threads:
            t.start()
            time.sleep(0.3)
        for t in threads:
            t.join()
        return out

    steps = [("short text", lambda: [call(short)]), ("agent text alone", lambda: [call(agent)]),
             ("image inventory alone", lambda: [call(inv)]), ("image inventory alone again", lambda: [call(inv)]),
             ("refer alone (non-stream)", lambda: [call(refer)]), ("two text at once", lambda: together(agent, agent2)),
             ("agent + image at once", lambda: together(agent, inv)), ("image + agent at once", lambda: together(inv, agent))]
    for name, fn in steps:
        t0 = time.time()
        results = fn()
        ok = healthy(args.base)
        row = {"step": name, "results": results, "healthy_after": ok, "s": round(time.time() - t0, 2)}
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
