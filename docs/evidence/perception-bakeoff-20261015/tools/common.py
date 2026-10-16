"""Shared I/O for the perception bake-off (throwaway evaluation code)."""
import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRAMES = ROOT / "frames"
GT = ROOT / "gt"
PRED = ROOT / "out" / "pred"


def frame_ids():
    return [m["id"] for m in json.loads((FRAMES / "manifest.json").read_text())]


def load_gt(fid):
    p = GT / f"{fid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_pred(pipeline, fid, preds, latency_s, extra=None, task="inventory", query=None):
    """preds: list of dict(label, score, bbox[x1,y1,x2,y2], mask=np.bool_ HxW or None)."""
    d = PRED / pipeline
    d.mkdir(parents=True, exist_ok=True)
    import hashlib
    stem = fid if query is None else f"{fid}__{hashlib.md5(query.encode()).hexdigest()[:10]}"
    masks = {}
    rows = []
    for i, p in enumerate(preds):
        m = p.get("mask")
        if m is not None:
            masks[f"m{i}"] = np.packbits(np.asarray(m, dtype=bool), axis=None)
            shape = list(np.asarray(m).shape)
        else:
            shape = None
        rows.append({"label": str(p["label"]), "score": float(p.get("score", 1.0)),
                     "bbox": [float(v) for v in p["bbox"]], "mask": f"m{i}" if m is not None else None,
                     "mask_shape": shape})
    if masks:
        np.savez_compressed(d / f"{stem}.npz", **masks)
    (d / f"{stem}.json").write_text(json.dumps({"id": fid, "pipeline": pipeline, "task": task, "query": query,
                                                "latency_s": latency_s, "preds": rows, "extra": extra or {}}, indent=1))


class Timer:
    def __enter__(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t0
