"""Bolt warm-start measurements, brain alone (throwaway, PR perf/bolt-warm-start).

  BRAIN_URL=http://127.0.0.1:18499/v1 BRAIN_MODEL=nemotron-3.5-bolt BRAIN_KEY_FILE=... python bench_warm.py first OUT.json
  ... python bench_warm.py slots OUT.json
  PYTHONPATH=<snapshot>/src ... python bench_warm.py perception OUT.json N

first       ONE request: the take's OpenClaw request 41 (system text with OpenClaw's STABLE/DYNAMIC markers, 49 tools,
            "What do you see on the table?") with NEW session ids, as the first call of a new OpenClaw chat sends it;
            streamed, temperature 0, thinking off, max_tokens 1 (agent_eval/bench_agent.py call()). Run it as the first
            request after a brain start: that is "the first chat after a start". No nonce: the system text and tools
            are byte-identical to what OpenClaw sent in the take, as in production.
nextday     `first` with the next day's date in OpenClaw's date line (the first line after its cache boundary).
slots       GET /slots: per slot the prompt tokens and the last task id (which slots hold what after the warm-up).
perception  N inventory requests on the bake-off frames, exactly as BrainDetector sends them for Bolt (the snapshot's
            brain_detector: INVENTORY_PROMPT, BOX_LIST_SCHEMA, max_tokens), one at a time, then the `first` request:
            does the warmed prefix survive background perception traffic before the first chat?
"""
import base64
import json
import pathlib
import sys
import time

import requests

sys.path.insert(0, "/home/spark/Projects/perception-bakeoff/agent_eval")
sys.path.insert(0, "/home/spark/Projects/perception-bakeoff/out/brain-comparison-2026-10-18/scripts")
import bench_agent as B
from bench_prefix2 import with_session

FRAMES = pathlib.Path("/home/spark/Projects/perception-bakeoff/frames")


def first(date=None):
    b = with_session(B.body(41), {})
    b["messages"][0]["content"] = b["messages"][0]["content"].replace("cascade/nemotron-3.5-bolt", f"cascade/{B.MODEL}")
    if date:   # another day: OpenClaw's date line right after its boundary changes
        assert "Current date: 2026-10-18" in b["messages"][0]["content"]
        b["messages"][0]["content"] = b["messages"][0]["content"].replace("Current date: 2026-10-18", f"Current date: {date}")
    t = time.time()
    r = B.call(b, 1)
    r["sent_at"] = t
    return r


def slots():
    r = requests.get(B.URL.rsplit("/v1", 1)[0] + "/slots", headers={"Authorization": f"Bearer {B.KEY}"}, timeout=30)
    return [{"id": s.get("id"), "id_task": s.get("id_task"), "n_prompt_tokens": s.get("n_prompt_tokens"),
             "is_processing": s.get("is_processing"),
             "head": (s.get("prompt") or "")[:40], "has_tools": "<tools>" in (s.get("prompt") or "")} for s in r.json()]


def inventory(fid):
    import cascade.perception.brain_detector as bd
    b64 = base64.b64encode((FRAMES / f"{fid}.jpg").read_bytes()).decode()
    payload = {"model": B.MODEL, "temperature": 0, "max_tokens": min(1500, bd.GUARDED_MAX_TOKENS),
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                   {"type": "text", "text": bd.INVENTORY_PROMPT}]}],
               "chat_template_kwargs": {"enable_thinking": False},
               "response_format": {"type": "json_schema", "json_schema": {"name": "box_list", "schema": bd.BOX_LIST_SCHEMA}}}
    t0 = time.perf_counter()
    r = requests.post(B.URL + "/chat/completions", json=payload, headers={"Authorization": f"Bearer {B.KEY}"}, timeout=600)
    return {"frame": fid, "status": r.status_code, "latency_s": round(time.perf_counter() - t0, 3),
            "prompt_n": (r.json().get("timings") or {}).get("prompt_n") if r.status_code == 200 else None}


def main(cmd, out, n=0):
    rep = {"url": B.URL, "model": B.MODEL, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd}
    if cmd == "first":
        rep["first"] = first()
        rep["slots_after"] = slots()
    elif cmd == "nextday":
        rep["first"] = first("2026-10-19")
        rep["slots_after"] = slots()
    elif cmd == "slots":
        rep["slots"] = slots()
    elif cmd == "perception":
        rep["slots_before"] = slots()
        ids = [m["id"] for m in json.loads((FRAMES / "manifest.json").read_text())][:n]
        rep["perception"] = [inventory(fid) for fid in ids]
        rep["slots_between"] = slots()
        rep["first"] = first()
        rep["slots_after"] = slots()
    print(json.dumps({k: v for k, v in rep.items() if k not in ("perception",)}, indent=None)[:2000], flush=True)
    pathlib.Path(out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
