"""Kinds of false positives per inventory pipeline (throwaway, PR perf/bolt-default-fixes): same matching as
tools/score.py (certain GT only, on-table predictions), each unmatched / wrong-label prediction classified as
  wrong     matched a GT object but the label is wrong
  dup       unmatched, box IoU >= 0.5 with another prediction of the same frame that did match
  near_gt   unmatched, box IoU >= 0.3 with some GT object (matched by another prediction, or point outside the mask)
  extra     unmatched, overlaps no GT object (something the labelers did not list: booth equipment, a part, ...)

  .venv/bin/python fp_kinds.py PIPE [PIPE ...] [--list]
"""
import collections
import json
import os
import sys

os.environ.setdefault("CERTAIN_ONLY", "1")
sys.path.insert(0, "/home/spark/Projects/perception-bakeoff/tools")
from score import GT, PRED, centre, inside_poly, iou, label_ok, load_preds, matches  # noqa: E402

gts = {p.stem: json.loads(p.read_text()) for p in sorted(GT.glob("*.json"))}
show = "--list" in sys.argv
for pipe in [a for a in sys.argv[1:] if not a.startswith("--")]:
    kinds = collections.Counter(); labels = collections.defaultdict(collections.Counter); n = ok = 0
    for fid, gt in gts.items():
        if not (PRED / pipe / f"{fid}.json").exists():
            continue
        preds = [p for p in load_preds(pipe, fid)["preds"] if inside_poly(centre(p), gt.get("table_polygon", []))]
        objs = gt["objects"]
        unc = {k for k, o in enumerate(objs) if "uncertain" in o.get("notes", "")}
        pairs = sorted(((iou(p["bbox"], o["bbox"]), i, k) for i, p in enumerate(preds) for k, o in enumerate(objs)
                        if matches(p, o)), reverse=True)
        up, uo, st = {}, set(), {}
        for _, i, k in pairs:
            if i in up or k in uo:
                continue
            up[i] = k; uo.add(k)
            st[i] = "uncertain" if k in unc else label_ok(preds[i]["label"], objs[k])
        for i, p in enumerate(preds):
            s = st.get(i, "unmatched")
            if s == "uncertain":
                continue
            n += 1
            if s in ("specific", "generic"):
                ok += 1
                continue
            if s != "unmatched":
                kind = "wrong"
            elif any(j != i and j in up and iou(p["bbox"], q["bbox"]) >= 0.5 for j, q in enumerate(preds)):
                kind = "dup"
            elif any(iou(p["bbox"], o["bbox"]) >= 0.3 for o in objs):
                kind = "near_gt"
            else:
                kind = "extra"
            kinds[kind] += 1
            labels[kind][p["label"]] += 1
    print(f"{pipe}: precision {ok}/{n} = {ok / max(n, 1):.3f}; FP {n - ok}: {dict(kinds)}")
    if show:
        for kind, c in labels.items():
            print(f"  {kind}: {c.most_common()}")
