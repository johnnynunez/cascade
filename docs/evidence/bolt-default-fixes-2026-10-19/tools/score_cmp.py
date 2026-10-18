"""Brain comparison scorer (throwaway): perception metrics for named brain pipelines, with box IoU, coordinate
error and the difficulty / camera breakdown, on top of tools/score.py's matching (same GT, same rules).

  CERTAIN_ONLY=1 .venv/bin/python out/brain-comparison-2026-10-18/scripts/score_cmp.py \
      boltpr:"Bolt vLLM (PR)" qwenpr:"Qwen3.8 vLLM (PR)" qwen:"Qwen3.8 rig vLLM (15 Oct)" ...
  -> <cmp>/raw/perception_scores.json + printed markdown tables

For tag T: inventory pipeline T_boxes_sam2 (brain boxes -> SAM2.1-L masks; falls back to T_boxes when no masks were
made), refer pipeline ref_T (brain boxes).
Inventory, per GT object (uncertain objects skipped): matched = box IoU >= 0.3 and the GT point inside the mask (box
when maskless), greedy by IoU as score.py. Adds:
  iou_mean     mean box IoU of the matched pairs
  ctr_err_px   median distance between matched box centres (pixels, 1280x720 frames)
  ctr_err_rel  median centre distance / GT box diagonal
  point_in_box share of matched pairs whose predicted BOX contains the GT point
  frames ok    frames whose raw reply parsed to a non-empty JSON list (tools/failures.py classes)
Refer, per query (= GT name): hit / top-1 / empty as score.py, plus top-1 box IoU against the GT object
(best GT instance of that name), its centre error, and hit by difficulty / camera.
"""
import collections
import json
import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path("/home/spark/Projects/perception-bakeoff")
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("CERTAIN_ONLY", "1")
from score import CERTAIN_ONLY, GT, PRED, centre, inside_poly, iou, label_ok, load_preds, matches, score_inventory, score_refer  # noqa: E402
import failures  # noqa: E402

CMP = pathlib.Path(__file__).resolve().parent.parent
MAN = {m["id"]: m["camera"] for m in json.loads((ROOT / "frames/manifest.json").read_text())}


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def diag(b):
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def p90(v):
    v = sorted(v)
    return v[int(0.9 * (len(v) - 1))] if v else None


def inventory(pipe, gts, brain_pipe):
    ious, ce, cr, pib = [], [], [], 0
    rows = []
    lat_brain = []
    for fid, gt in gts.items():
        if not (PRED / pipe / f"{fid}.json").exists():
            continue
        j = load_preds(pipe, fid)
        if (PRED / brain_pipe / f"{fid}.json").exists():
            lat_brain.append(json.loads((PRED / brain_pipe / f"{fid}.json").read_text())["latency_s"])
        preds = [p for p in j["preds"] if inside_poly(centre(p), gt.get("table_polygon", []))]
        objs = gt["objects"]
        pairs = sorted(((iou(p["bbox"], o["bbox"]), i, k) for i, p in enumerate(preds) for k, o in enumerate(objs)
                        if matches(p, o)), reverse=True)
        used_p, got = set(), {}
        for v, i, k in pairs:
            if i in used_p or k in got:
                continue
            used_p.add(i); got[k] = (preds[i], v)
        for k, o in enumerate(objs):
            if CERTAIN_ONLY and "uncertain" in o.get("notes", ""):
                continue
            if k in got:
                p, v = got[k]
                ious.append(v)
                d = dist(centre(p), centre(o)); ce.append(d); cr.append(d / max(diag(o["bbox"]), 1e-6))
                b = p["bbox"]; x, y = o["point"]
                pib += b[0] <= x <= b[2] and b[1] <= y <= b[3]
                verdict = label_ok(p["label"], o)
            else:
                verdict = "missed"
            rows.append((MAN.get(fid, "?"), o.get("difficulty", "?"), verdict))
    base = score_inventory(pipe, gts) or {}

    def grp(idx):
        g = collections.defaultdict(list)
        for r in rows:
            g[r[idx]].append(r[2])
        return {k: {"n": len(v), "loc": sum(x != "missed" for x in v) / len(v),
                    "id_lenient": sum(x in ("specific", "generic") for x in v) / len(v)} for k, v in sorted(g.items())}
    base.update(iou_mean=statistics.mean(ious) if ious else None, iou_median=statistics.median(ious) if ious else None,
                ctr_err_px=statistics.median(ce) if ce else None, ctr_err_rel=statistics.median(cr) if cr else None,
                point_in_box=pib / len(ious) if ious else None, by_difficulty=grp(1), by_camera=grp(0),
                brain_latency_median_s=statistics.median(lat_brain) if lat_brain else None,
                brain_latency_p90_s=p90(lat_brain), brain_latency_max_s=max(lat_brain) if lat_brain else None)
    return base


def refer(pipe, gts):
    import hashlib
    out = score_refer(pipe, gts) or {}
    ious, ce, rows, lat = [], [], [], []
    for fid, gt in gts.items():
        for obj in gt["objects"]:
            if CERTAIN_ONLY and "uncertain" in obj.get("notes", ""):
                continue
            stem = f"{fid}__{hashlib.md5(obj['name'].encode()).hexdigest()[:10]}"
            if not (PRED / pipe / f"{stem}.json").exists():
                continue
            j = load_preds(pipe, stem)
            lat.append(j["latency_s"])
            preds = j["preds"]
            hit = any(matches(p, obj) for p in preds)
            rows.append((MAN.get(fid, "?"), obj.get("difficulty", "?"), hit, bool(preds)))
            if preds:
                best = max(preds, key=lambda p: p["score"])
                same = [o for o in gt["objects"] if o["name"] == obj["name"]]
                o = max(same, key=lambda o: iou(best["bbox"], o["bbox"]))
                ious.append(iou(best["bbox"], o["bbox"]))
                ce.append(dist(centre(best), centre(o)))

    def grp(idx):
        g = collections.defaultdict(list)
        for r in rows:
            g[r[idx]].append(r)
        return {k: {"n": len(v), "hit": sum(x[2] for x in v) / len(v), "empty": sum(not x[3] for x in v) / len(v)}
                for k, v in sorted(g.items())}
    out.update(top1_iou_mean=statistics.mean(ious) if ious else None, top1_iou_ge_0_5=(sum(v >= 0.5 for v in ious) / len(ious)) if ious else None,
               top1_ctr_err_px=statistics.median(ce) if ce else None, by_difficulty=grp(1), by_camera=grp(0),
               latency_p90_s=p90(lat), latency_max_s=max(lat) if lat else None)
    return out


def frames_status(tag):
    pipe = f"{tag}_boxes"
    if not (PRED / pipe).exists():
        return None
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        failures.main([pipe])
    return json.loads(buf.getvalue())[pipe]


def main(specs):
    gts = {p.stem: json.loads(p.read_text()) for p in sorted(GT.glob("*.json"))}
    res = {}
    for spec in specs:
        tag, _, name = spec.partition(":")
        inv_pipe = f"{tag}_boxes_sam2" if (PRED / f"{tag}_boxes_sam2").exists() else f"{tag}_boxes"
        res[tag] = {"name": name or tag, "inventory_pipeline": inv_pipe,
                    "inventory": inventory(inv_pipe, gts, f"{tag}_boxes"),
                    "refer": refer(f"ref_{tag}", gts) if (PRED / f"ref_{tag}").exists() else None,
                    "frames": frames_status(tag)}
    (CMP / "raw").mkdir(exist_ok=True)
    (CMP / "raw/perception_scores.json").write_text(json.dumps(res, indent=1))
    f2 = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    print("## Inventory (list everything; brain boxes -> SAM2.1-L masks)\n")
    print("| Brain | frames ok | loc recall | id strict | id lenient | precision | wrong label | box IoU | centre err px (rel) | median s (brain) | p90 s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, r in res.items():
        i = r["inventory"]; fr = r["frames"] or {}
        ok = (fr.get("frames") or {}).get("ok")
        tot = sum((fr.get("frames") or {}).values()) if fr else None
        print(f"| {r['name']} | {ok}/{tot} | {f2(i.get('loc_recall'))} | {f2(i.get('id_strict'))} | {f2(i.get('id_lenient'))} | "
              f"{f2(i.get('precision'))} | {f2(i.get('wrong_label'))} | {f2(i.get('iou_mean'))} | "
              f"{i['ctr_err_px']:.1f} ({i['ctr_err_rel']:.2f}) | {f2(i.get('brain_latency_median_s'))} | {f2(i.get('brain_latency_p90_s'))} |"
              if i.get("ctr_err_px") is not None else f"| {r['name']} | - |")
    keys = sorted({k for r in res.values() for k in r["inventory"].get("by_difficulty", {})})
    cams = sorted({k for r in res.values() for k in r["inventory"].get("by_camera", {})})
    print("\n### Inventory by difficulty and camera: loc / id lenient (n GT objects)\n")
    print("| Subset | " + " | ".join(r["name"] for r in res.values()) + " |")
    print("|---|" + "---|" * len(res))
    for grp, ks in (("by_difficulty", keys), ("by_camera", cams)):
        for k in ks:
            n = next((r["inventory"][grp][k]["n"] for r in res.values() if k in r["inventory"].get(grp, {})), "?")
            cells = [f"{r['inventory'][grp][k]['loc']:.2f} / {r['inventory'][grp][k]['id_lenient']:.2f}"
                     if k in r["inventory"].get(grp, {}) else "-" for r in res.values()]
            print(f"| {k} ({n}) | " + " | ".join(cells) + " |")
    print("\n## Refer (find the named object; brain boxes)\n")
    print("| Brain | queries | hit | top-1 | empty | top-1 box IoU | IoU >= 0.5 | top-1 centre err px | median s | p90 s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag, r in res.items():
        f = r["refer"]
        if not f:
            continue
        print(f"| {r['name']} | {f['queries']} | {f2(f['hit'])} | {f2(f['top1'])} | {f2(f['empty'])} | {f2(f['top1_iou_mean'])} | "
              f"{f2(f['top1_iou_ge_0_5'])} | {f2(f['top1_ctr_err_px'])} | {f2(f['latency_median_s'])} | {f2(f['latency_p90_s'])} |")
    print("\n### Refer hit by difficulty and camera (n queries)\n")
    rr = {t: r for t, r in res.items() if r["refer"]}
    print("| Subset | " + " | ".join(r["name"] for r in rr.values()) + " |")
    print("|---|" + "---|" * len(rr))
    for grp in ("by_difficulty", "by_camera"):
        ks = sorted({k for r in rr.values() for k in r["refer"][grp]})
        for k in ks:
            n = next((r["refer"][grp][k]["n"] for r in rr.values() if k in r["refer"][grp]), "?")
            print(f"| {k} ({n}) | " + " | ".join(f"{r['refer'][grp][k]['hit']:.2f}" if k in r["refer"][grp] else "-"
                                                for r in rr.values()) + " |")
    print("\n### Whole-frame failures (inventory replies) and refer reply classes\n")
    for tag, r in res.items():
        fr = r["frames"]
        if fr:
            print(f"- {r['name']}: frames {fr['frames']}; refer {fr['refer']}; completion tokens median/max "
                  f"{fr['completion_tokens_median']}/{fr['completion_tokens_max']}; reasoning leaks {fr['reasoning_leaks']}")
            for b in fr["failed"]:
                print(f"  - {b}")


if __name__ == "__main__":
    main(sys.argv[1:])
