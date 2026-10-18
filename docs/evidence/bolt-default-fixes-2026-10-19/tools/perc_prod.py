"""Bolt perception with the EXACT request BrainDetector sends for a guarded brain (throwaway, PR perf/bolt-default-fixes).

  PYTHONPATH=<snapshot>/src BRAIN_URL=http://127.0.0.1:18499/v1 BRAIN_KEY_FILE=... python perc_prod.py TAG [inventory|refer|both]

The request is built from the snapshot's cascade.perception.brain_detector (the code under test), as BrainDetector._ask
builds it: INVENTORY_PROMPT / REFER_PROMPT, temperature 0, thinking off, max_tokens min(1500, GUARDED_MAX_TOKENS),
response_format json_schema BOX_LIST_SCHEMA. Image: the bake-off frame JPEG as sent to every brain in the 2026-10-18
comparison (Qwen's row used the same bytes and prompts). One request at a time, non-streamed. Every RAW reply is
saved to out/bolt-default-fixes-2026-10-18/raw/perc_<TAG>.jsonl; parsing happens offline (analyze_perc.py, the
snapshot's parse_guarded_boxes), so a parser change never needs new brain calls.
"""
import base64
import json
import os
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path("/home/spark/Projects/perception-bakeoff")
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from common import FRAMES, frame_ids, load_gt  # noqa: E402
import cascade.perception.brain_detector as bd  # noqa: E402

URL = os.environ["BRAIN_URL"].rstrip("/")
MODEL = os.environ.get("BRAIN_MODEL", "nemotron-3.5-bolt")
KEY = pathlib.Path(os.environ["BRAIN_KEY_FILE"]).read_text().strip()
assert MODEL in bd.GUARDED_REPLY_MODELS, MODEL


def ask(prompt, fid):
    b64 = base64.b64encode((FRAMES / f"{fid}.jpg").read_bytes()).decode()
    payload = {"model": MODEL, "temperature": 0, "max_tokens": min(1500, bd.GUARDED_MAX_TOKENS),
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                   {"type": "text", "text": prompt}]}],
               "chat_template_kwargs": {"enable_thinking": False},
               "response_format": {"type": "json_schema", "json_schema": {"name": "box_list", "schema": bd.BOX_LIST_SCHEMA}}}
    t0 = time.perf_counter()
    r = requests.post(URL + "/chat/completions", json=payload, headers={"Authorization": f"Bearer {KEY}"}, timeout=600)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return {"raw": "", "finish": f"http {r.status_code}: {r.text[:300]}", "completion_tokens": None, "latency_s": dt}
    j = r.json()
    ch = j["choices"][0]
    return {"raw": ch["message"].get("content") or "", "finish": ch.get("finish_reason"),
            "completion_tokens": (j.get("usage") or {}).get("completion_tokens"), "latency_s": dt,
            "timings": {k: (j.get("timings") or {}).get(k) for k in ("prompt_n", "predicted_n", "predicted_per_second",
                                                                         "draft_n", "draft_n_accepted")}}


def main(tag, what):
    log = ROOT / f"out/bolt-default-fixes-2026-10-18/raw/perc_{tag}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("module", bd.__file__, "max boxes", bd.GUARDED_MAX_BOXES, "max tokens", bd.GUARDED_MAX_TOKENS, flush=True)
    with log.open("a") as out:
        for fid in frame_ids():
            if what in ("inventory", "both"):
                a = ask(bd.INVENTORY_PROMPT, fid)
                out.write(json.dumps({"task": "inventory", "frame": fid, **a}) + "\n"); out.flush()
                print(fid, a["finish"], a["completion_tokens"], round(a["latency_s"], 2), flush=True)
            if what in ("refer", "both"):
                for obj in (load_gt(fid) or {}).get("objects", []):
                    a = ask(bd.REFER_PROMPT.format(q=obj["name"]), fid)
                    out.write(json.dumps({"task": "refer", "frame": fid, "query": obj["name"], **a}) + "\n"); out.flush()
                print(fid, "refer done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "inventory")
