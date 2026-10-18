#!/usr/bin/env python3
"""Agent replay WITH cascade's perception running beside it, as in the trial's real session.

Thread A replays the recorded OpenClaw session (replay_agent.SESSION), one call after another, 3 s apart. Thread B
replays cascade's own perception requests (BrainDetector prompts on live kitchen frames: streamed background
inventories of two cameras and a non-streamed REFER) in a loop until A finishes, 2 s apart, as the watcher does. Every call is timed (TTFT, total); perception and agent rows are labelled.

    replay_mixed.py --base URL --key-file K --label L --out f.jsonl [--replays 1]
"""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
from pathlib import Path

import replay_agent as ra

# cascade's own BrainDetector requests (INVENTORY_PROMPT / REFER_PROMPT) on live Isaac kitchen frames
# (make_perception_bodies.py): the watcher's streamed background inventories of two cameras and one REFER.
PERCEPTION = ("inventory-cam0", "refer-green-cube", "inventory-side")
PERCEPTION_DIR = Path(__file__).resolve().parent / "perception-bodies"


def load(name: str) -> dict:
    return json.loads(gzip.open(PERCEPTION_DIR / f"{name}.json.gz").read())


def timed(base: str, key: str, model: str, body: dict) -> dict:
    """One request as recorded (stream or not), model renamed; TTFT = first content/tool/reasoning byte."""
    import urllib.request

    payload = dict(body, model=model)
    stream = bool(payload.get("stream"))
    if stream:
        payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    first = None
    usage = None
    timings = None
    with urllib.request.urlopen(req, timeout=900) as r:
        if not stream:
            data = json.loads(r.read())
            first = time.time()
            usage, timings = data.get("usage"), data.get("timings")
        else:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                ev = json.loads(line[5:])
                usage = ev.get("usage") or usage
                timings = ev.get("timings") or timings
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                        first = time.time()
    t1 = time.time()
    return {"t0": t0, "ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 3),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "llama_cache_n": (timings or {}).get("cache_n")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", default="nemotron-3.5-bolt")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replays", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    out = Path(args.out)
    lock = threading.Lock()

    def write(row):
        row["label"] = args.label
        print(json.dumps(row), flush=True)
        with lock, out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    perception = [load(s) for s in PERCEPTION]
    session = ra.load_session()
    for replay in range(args.replays):
        done = threading.Event()

        def watcher():
            i = 0
            while not done.is_set():
                seq = PERCEPTION[i % len(PERCEPTION)]
                try:
                    row = timed(args.base, key, args.model, perception[i % len(PERCEPTION)])
                except Exception as exc:  # noqa: BLE001 - measurement: record and continue
                    row = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                write({"who": "perception", "replay": replay, "request": seq, **row})
                i += 1
                done.wait(2.0)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        time.sleep(1.0)
        mapping: dict[str, str] = {}
        for index, (seq, body) in enumerate(zip(ra.SESSION, session)):
            agent = ra.rename_session(body, mapping)
            agent = {k: v for k, v in agent.items() if k not in ("max_completion_tokens", "max_tokens")}
            agent.update(stream=True, max_completion_tokens=args.max_tokens, temperature=0,
                         chat_template_kwargs={"enable_thinking": False})
            row = timed(args.base, key, args.model, agent)
            write({"who": "agent", "replay": replay, "call": index, "trial_seq": seq, **row})
            time.sleep(3.0)
        done.set()
        thread.join(timeout=900)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
