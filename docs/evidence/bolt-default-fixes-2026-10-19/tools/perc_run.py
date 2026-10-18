"""Bolt perception request variants on the bake-off frames (throwaway, PR perf/bolt-default-fixes).

  BRAIN_URL=http://127.0.0.1:18499/v1 BRAIN_KEY_FILE=... python perc_run.py <variant> [inventory|refer|both]

Same prompts, image bytes and decoding as the 2026-10-18 brain comparison (tools/run_candidates.py: the frame JPEG as
sent, temperature 0, thinking off), one request at a time, NON-streamed. A variant changes only the request:
  v0          the comparison's request (max_tokens 1500, nothing else) -> must reproduce its 6 looping frames
  js<N>       + response_format json_schema: a list of at most N {label, bbox_2d} objects, labels 1-64 chars,
              4 integer coordinates 0..1000 (llama.cpp turns the schema into a grammar)
  js<N>dry    js<N> + DRY (multiplier 0.8, base 1.75, allowed length 2, last 2048 tokens (v0.7.0 refuses -1), default breakers)
  js<N>drynb  js<N> + DRY whose only breaker never occurs (v0.7.0 refuses an empty list; allowed length 3): penalises
              re-emitting a whole earlier item
  t<M>        v0 with max_tokens M (no schema)
Every RAW reply is saved (finish reason, completion tokens, latency): parsing / salvage happens offline, so a parser
change never needs new brain calls. Output: <ROOT>/out/pred/bf_<variant>_boxes/<frame>.json (inventory, bake-off
format, boxes parsed with the bake-off parser; a cut reply parses to nothing, as production rejects it) and
ref_bf_<variant>/<frame>__<md5(name)[:10]>.json (refer, one query per GT name).
"""
import base64
import hashlib
import json
import os
import pathlib
import re
import sys
import time

import requests

ROOT = pathlib.Path("/home/spark/Projects/perception-bakeoff")
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from common import FRAMES, frame_ids, load_gt, save_pred  # noqa: E402
from run_candidates import P_BOXES, P_REFER, brain_boxes, img  # noqa: E402

URL = os.environ["BRAIN_URL"].rstrip("/")
MODEL = os.environ.get("BRAIN_MODEL", "nemotron-3.5-bolt")
KEY = pathlib.Path(os.environ["BRAIN_KEY_FILE"]).read_text().strip()


def schema(n):
    return {"type": "array", "maxItems": n, "items": {
        "type": "object", "additionalProperties": False, "required": ["label", "bbox_2d"],
        "properties": {"label": {"type": "string", "minLength": 1, "maxLength": 64},
                       "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4,
                                   "items": {"type": "integer", "minimum": 0, "maximum": 1000}}}}}


def variant(name):
    body = {"max_tokens": 1500}
    m = re.fullmatch(r"js(\d+)(dry|drynb)?", name)
    if m:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "boxes", "schema": schema(int(m[1]))}}
        if m[2] == "dry":
            body.update(dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2, dry_penalty_last_n=2048)
        elif m[2] == "drynb":
            body.update(dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=3, dry_penalty_last_n=2048,
                        dry_sequence_breakers=["\u00a4"])
    elif re.fullmatch(r"t(\d+)", name):
        body["max_tokens"] = int(name[1:])
    elif name != "v0":
        raise SystemExit(f"unknown variant {name}")
    return body


def ask(prompt, fid, extra):
    b64 = base64.b64encode((FRAMES / f"{fid}.jpg").read_bytes()).decode()
    payload = {"model": MODEL, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                   {"type": "text", "text": prompt}]}], **extra}
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


def main(name, what):
    extra = variant(name)
    log = ROOT / f"out/bolt-default-fixes-2026-10-18/raw/perc_{name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        for fid in frame_ids():
            w, h = img(fid).size
            if what in ("inventory", "both"):
                a = ask(P_BOXES, fid, extra)
                preds = brain_boxes(a["raw"], w, h) if a["finish"] == "stop" else []
                save_pred(f"bf_{name}_boxes", fid, preds, a["latency_s"], extra={k: v for k, v in a.items() if k != "latency_s"})
                out.write(json.dumps({"task": "inventory", "frame": fid, **a}) + "\n"); out.flush()
                print(fid, a["finish"], a["completion_tokens"], round(a["latency_s"], 2), len(preds), flush=True)
            if what in ("refer", "both"):
                gt = load_gt(fid)
                for obj in (gt or {}).get("objects", []):
                    q = obj["name"]
                    a = ask(P_REFER.format(q=q), fid, extra)
                    preds = brain_boxes(a["raw"], w, h) if a["finish"] == "stop" else []
                    save_pred(f"ref_bf_{name}", fid, preds, a["latency_s"], task="refer", query=q,
                              extra={k: v for k, v in a.items() if k != "latency_s"})
                    out.write(json.dumps({"task": "refer", "frame": fid, "query": q, **a}) + "\n"); out.flush()
                print(fid, "refer done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "inventory")
