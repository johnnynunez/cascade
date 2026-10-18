"""SAM2.1-L box -> mask step on STORED brain boxes (throwaway, PR perf/bolt-default-fixes; no brain call).

  .venv/bin/python sam2_stored.py TAG ...   ->  out/pred/<TAG>_boxes_sam2/

Same model, call and storage as tools/run_candidates.py's `<tag>_boxes_sam2` pipeline, but the boxes come from the
stored predictions (<TAG>_boxes, written by analyze_perc.py with cascade's production parsers) instead of re-parsing
the raw reply with the bake-off parser.
"""
import json
import sys
import time

sys.path.insert(0, "/home/spark/Projects/perception-bakeoff/tools")
from common import PRED, frame_ids, save_pred  # noqa: E402
from run_candidates import sam2_masks  # noqa: E402

for tag in sys.argv[1:]:
    for fid in frame_ids():
        src = PRED / f"{tag}_boxes" / f"{fid}.json"
        if not src.exists():
            continue
        d = json.loads(src.read_text())
        preds = [{"label": p["label"], "score": p["score"], "bbox": p["bbox"], "mask": None} for p in d["preds"]]
        t0 = time.perf_counter()
        preds = sam2_masks(fid, preds)
        save_pred(f"{tag}_boxes_sam2", fid, preds, time.perf_counter() - t0, extra={"boxes_from": f"{tag}_boxes"})
    print(tag, "done", flush=True)
