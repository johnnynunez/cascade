"""Score bake-off predictions against the ground truth (throwaway evaluation code).

  python3 tools/score.py            -> out/scores.md + out/scores.json

Inventory (list everything on the table):
  loc_recall    GT objects matched by a prediction (box IoU >= 0.3 and, with a mask, the GT point inside the mask)
  id_strict     matched AND label names the object specifically (e.g. 'lobster')
  id_lenient    matched AND label is specific or an acceptable category (e.g. 'stuffed toy')
  precision     correct (lenient) predictions / all predictions centred on the table
  wrong_label   matched predictions whose label is wrong (e.g. lobster called 'flower')
Refer (find the object the user named; query = GT name):
  hit           a returned instance matches the named object
  top1          the highest-scoring instance is a correct object of that name
  empty         returned nothing although the object is visible
"""
import json
import pathlib
import re
import statistics

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
GT, PRED = ROOT / "gt", ROOT / "out" / "pred"
import os  # noqa: E402
CERTAIN_ONLY = os.environ.get("CERTAIN_ONLY", "1") == "1"  # skip objects the labelers marked uncertain
EXTRA_LATENCY = {  # pipelines whose runner reused an earlier brain call
    "qwen_boxes_sam2": "qwen_boxes", "llmdet_qn": "qwen_names", "yoloe26_qn": "qwen_names",
    "sam3_qn": "qwen_names", "astra_boxes_sam2": "astra_boxes", "sam3_an": "astra_boxes",
    "fuse_qwen_llmdet": "qwen_names", "fuse_qwen_sam3": "qwen_names", "fuse_qwen_llmdet_hi": "qwen_names",
    "qwen_boxes_sam31": "qwen_boxes", "sam31_qn": "qwen_names",
    "qwen_boxes_sam3n": "qwen_boxes", "sam3n_qn": "qwen_names",
    "opus_boxes_sam2": "opus_boxes", "cosmos_boxes_sam2": "cosmos_boxes"}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return " " + " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.split()) + " "


def has(label, phrases):
    nl = norm(label)
    return any(norm(p) in nl for p in phrases if p.strip())


def label_ok(label, obj):
    if has(label, obj.get("specific", [])):
        return "specific"
    if has(label, obj.get("wrong", [])):
        return "wrong"
    if has(label, obj.get("generic", [])):
        return "generic"
    return "wrong"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def inside_poly(pt, poly):
    if not poly or len(poly) < 3:
        return True
    x, y = pt; c = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1:
            c = not c
    return c


def load_preds(pipe, stem):
    j = json.loads((PRED / pipe / f"{stem}.json").read_text())
    npz = PRED / pipe / f"{stem}.npz"
    masks = np.load(npz) if npz.exists() else {}
    for p in j["preds"]:
        if p.get("mask") and p["mask"] in masks:
            h, w = p["mask_shape"]
            p["_mask"] = np.unpackbits(masks[p["mask"]])[: h * w].reshape(h, w).astype(bool)
    return j


def matches(p, obj):
    if iou(p["bbox"], obj["bbox"]) < 0.3:
        return False
    m = p.get("_mask")
    if m is not None:
        x, y = [int(round(v)) for v in obj["point"]]
        if not (0 <= y < m.shape[0] and 0 <= x < m.shape[1]):
            return False
        y0, y1, x0, x1 = max(0, y - 6), y + 7, max(0, x - 6), x + 7
        return bool(m[y0:y1, x0:x1].any())
    x, y = obj["point"]
    b = p["bbox"]
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def centre(p):
    b = p["bbox"]
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def score_inventory(pipe, gts):
    n_gt = loc = ids = idl = wrong = n_pred = correct = 0
    lat = []
    for fid, gt in gts.items():
        f = PRED / pipe / f"{fid}.json"
        if not f.exists():
            continue
        j = load_preds(pipe, fid)
        t = j["latency_s"]
        if pipe in EXTRA_LATENCY and (PRED / EXTRA_LATENCY[pipe] / f"{fid}.json").exists():
            t += json.loads((PRED / EXTRA_LATENCY[pipe] / f"{fid}.json").read_text())["latency_s"]
        lat.append(t)
        preds = [p for p in j["preds"] if inside_poly(centre(p), gt.get("table_polygon", []))]
        objs = gt["objects"]
        unc = {k for k, o in enumerate(objs) if CERTAIN_ONLY and "uncertain" in o.get("notes", "")}
        n_gt += len(objs) - len(unc)
        pairs = sorted(((iou(p["bbox"], o["bbox"]), i, k) for i, p in enumerate(preds) for k, o in enumerate(objs)
                        if matches(p, o)), reverse=True)
        used_p, used_o, drop = set(), set(), set()
        for _, i, k in pairs:
            if i in used_p or k in used_o:
                continue
            used_p.add(i); used_o.add(k)
            if k in unc:  # a human could not identify it either: neither credit nor penalty
                drop.add(i)
                continue
            loc += 1
            v = label_ok(preds[i]["label"], objs[k])
            ids += v == "specific"; idl += v in ("specific", "generic"); wrong += v == "wrong"
            correct += v in ("specific", "generic")
        n_pred += len(preds) - len(drop)
    if not n_gt:
        return None
    return {"frames": len(lat), "gt": n_gt, "preds": n_pred, "loc_recall": loc / n_gt, "id_strict": ids / n_gt,
            "id_lenient": idl / n_gt, "precision": correct / n_pred if n_pred else 0.0,
            "wrong_label": wrong / loc if loc else 0.0, "latency_median_s": statistics.median(lat) if lat else None}


def score_refer(pipe, gts):
    n = hit = top1 = empty = 0
    lat = []
    for fid, gt in gts.items():
        for obj in gt["objects"]:
            if CERTAIN_ONLY and "uncertain" in obj.get("notes", ""):
                continue
            import hashlib
            stem = f"{fid}__{hashlib.md5(obj['name'].encode()).hexdigest()[:10]}"
            if not (PRED / pipe / f"{stem}.json").exists():
                continue
            j = load_preds(pipe, stem); n += 1; lat.append(j["latency_s"])
            preds = j["preds"]
            same = [o for o in gt["objects"] if o["name"] == obj["name"]]
            if not preds:
                empty += 1
                continue
            hit += any(matches(p, obj) for p in preds)
            best = max(preds, key=lambda p: p["score"])
            top1 += any(matches(best, o) for o in same)
    if not n:
        return None
    return {"queries": n, "hit": hit / n, "top1": top1 / n, "empty": empty / n,
            "latency_median_s": statistics.median(lat)}


def main():
    gts = {p.stem: json.loads(p.read_text()) for p in sorted(GT.glob("*.json"))}
    res = {"inventory": {}, "refer": {}}
    for d in sorted(PRED.iterdir()) if PRED.exists() else []:
        r = score_refer(d.name, gts) if d.name.startswith("ref_") else score_inventory(d.name, gts)
        if r:
            res["refer" if d.name.startswith("ref_") else "inventory"][d.name] = r
    (ROOT / "out/scores.json").write_text(json.dumps(res, indent=1))
    lines = [f"GT: {len(gts)} frames, {sum(len(g['objects']) for g in gts.values())} objects", "",
             "INVENTORY  pipeline | loc_recall | id_strict | id_lenient | precision | wrong_label | median s"]
    for k, r in sorted(res["inventory"].items(), key=lambda kv: -kv[1]["id_lenient"]):
        lines.append(f"{k} | {r['loc_recall']:.2f} | {r['id_strict']:.2f} | {r['id_lenient']:.2f} | "
                     f"{r['precision']:.2f} | {r['wrong_label']:.2f} | {r['latency_median_s']:.2f}")
    lines += ["", "REFER  pipeline | hit | top1 | empty | median s"]
    for k, r in sorted(res["refer"].items(), key=lambda kv: -kv[1]["hit"]):
        lines.append(f"{k} | {r['hit']:.2f} | {r['top1']:.2f} | {r['empty']:.2f} | {r['latency_median_s']:.2f}")
    (ROOT / "out/scores.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
