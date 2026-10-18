"""Offline screening of guarded-parser variants on STORED raw Bolt replies (throwaway, PR perf/bolt-default-fixes).

  PYTHONPATH=<wt>/src .venv/bin/python screen_guards.py <wt> SOURCE NAME:N:EXTRA_REGEX ...

SOURCE = raw/perc_<SOURCE>.jsonl (inventory rows). For each variant: the reply cut after its first N entries (what a
schema maxItems of N gives under greedy decoding; N=20 = as served), parsed with the worktree's parse_guarded_boxes,
then entries whose label matches EXTRA_REGEX (searched at the END of the normalized label, as _EXCLUDED_HEAD) dropped.
Prints loc / lenient / precision with boxes only (score.py matching, certain GT, on-table predictions) and the
completion tokens a schema of N would have spent (sum of the first N entries' share of the reply).
Writes out/pred/scr_<NAME>_boxes/ for score_cmp / fp_kinds.
"""
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("CERTAIN_ONLY", "1")
ROOT = pathlib.Path("/home/spark/Projects/perception-bakeoff")
OUT = ROOT / "out/bolt-default-fixes-2026-10-18"
sys.path.insert(0, str(ROOT / "tools"))
from common import save_pred  # noqa: E402
from score import GT, PRED, centre, inside_poly, iou, label_ok, load_preds, matches  # noqa: E402

WT = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(WT / "src"))
import cascade.perception.brain_detector as bd  # noqa: E402

assert pathlib.Path(bd.__file__).resolve().is_relative_to(WT.resolve()), bd.__file__
W, H = 1280, 720
gts = {p.stem: json.loads(p.read_text()) for p in sorted(GT.glob("*.json"))}


def first_entries(text, n):
    body = bd._reply_body(text)
    items = bd._answer_list(body) or bd._complete_items(body)
    return json.dumps(items[:n]), len(items)


def score(pipe):
    n = ok = loc = idl = n_gt = 0
    for fid, gt in gts.items():
        if not (PRED / pipe / f"{fid}.json").exists():
            continue
        preds = [p for p in load_preds(pipe, fid)["preds"] if inside_poly(centre(p), gt.get("table_polygon", []))]
        objs = gt["objects"]
        unc = {k for k, o in enumerate(objs) if "uncertain" in o.get("notes", "")}
        n_gt += len(objs) - len(unc)
        pairs = sorted(((iou(p["bbox"], o["bbox"]), i, k) for i, p in enumerate(preds) for k, o in enumerate(objs)
                        if matches(p, o)), reverse=True)
        up, uo, drop = set(), set(), set()
        for _, i, k in pairs:
            if i in up or k in uo:
                continue
            up.add(i); uo.add(k)
            if k in unc:
                drop.add(i)
                continue
            loc += 1
            v = label_ok(preds[i]["label"], objs[k])
            idl += v in ("specific", "generic"); ok += v in ("specific", "generic")
        n += len(preds) - len(drop)
    return loc / n_gt, idl / n_gt, ok, n


source = sys.argv[2]
rows = [json.loads(x) for x in (OUT / f"raw/perc_{source}.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["task"] == "inventory"]
for spec in sys.argv[3:]:
    name, n_max, extra = spec.split(":", 2)
    n_max = int(n_max)
    rx = re.compile(f"(?:{extra})$") if extra else None
    toks = 0
    for r in rows:
        text, n_items = first_entries(r["raw"], n_max)
        bound = n_items >= n_max
        old, bd.GUARDED_MAX_BOXES = bd.GUARDED_MAX_BOXES, n_max
        try:
            found = bd.parse_guarded_boxes(text if n_items > n_max else r["raw"], W, H, cut=r["finish"] == "length",
                                           inventory=True)
        finally:
            bd.GUARDED_MAX_BOXES = old
        if rx:
            found = [(lab, b) for lab, b in found if not rx.search(" ".join(lab.lower().split()))]
        preds = [{"label": lab, "score": 1.0, "bbox": [float(v) for v in b], "mask": None} for lab, b in found]
        save_pred(f"scr_{name}_boxes", r["frame"], preds, r["latency_s"], extra={"source": source, "spec": spec})
        ct = r.get("completion_tokens") or 0
        toks += ct * min(1.0, n_max / n_items) if n_items else ct
    loc, idl, ok, n = score(f"scr_{name}_boxes")
    print(f"{name:12s} N={n_max:2d} loc {loc:.3f} lenient {idl:.3f} precision {ok}/{n} = {ok / max(n, 1):.3f} "
          f"~tokens {toks:.0f}  extra={extra!r}")
