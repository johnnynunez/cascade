"""SAM 3.1 (and native SAM 3) through Meta's official repo (throwaway evaluation code).

  .venv-sam31/bin/python tools/run_sam31.py <ckpt: sam3.1|sam3> <pipeline>
pipelines (same inputs as run_candidates.py):
  qn        Qwen3.8 names -> text prompts            -> out/pred/<tag>_qn
  qboxes    Qwen3.8 boxes -> box prompts (masks)     -> out/pred/qwen_boxes_<tag>
  ref       ground-truth names -> text prompts       -> out/pred/ref_<tag>
The image is encoded ONCE per frame (set_image) and every prompt reuses it.
"""
import json
import pathlib
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import FRAMES, PRED, ROOT, Timer, frame_ids, load_gt, save_pred  # noqa: E402

from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

CKPT = {"sam3.1": ROOT / "weights/sam3.1/sam3.1_multiplex.pt", "sam3": ROOT / "weights/sam3native/sam3.pt"}
TAG = {"sam3.1": "sam31", "sam3": "sam3n"}
import os
FP32 = os.environ.get("SAM_FP32") == "1"


def main(ver, pipe):
    t0 = time.time()
    model = build_sam3_image_model(checkpoint_path=str(CKPT[ver]), load_from_HF=False, device="cuda", eval_mode=True)
    proc = Sam3Processor(model, confidence_threshold=0.5)
    print("load", round(time.time() - t0, 1), "s", flush=True)
    tag = TAG[ver] + ("f32" if FP32 else "")
    out_name = {"qn": f"{tag}_qn", "qboxes": f"qwen_boxes_{tag}", "ref": f"ref_{tag}"}[pipe]
    for fid in frame_ids():
        im = Image.open(FRAMES / f"{fid}.jpg").convert("RGB")
        w, h = im.size
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not FP32):
            with Timer() as tenc:
                state = proc.set_image(im)

            def run_text(q):
                proc.reset_all_prompts(state)
                s = proc.set_text_prompt(prompt=q, state=state)
                return [{"label": q, "score": float(sc), "bbox": b.float().tolist(), "mask": m[0].cpu().numpy()}
                        for m, b, sc in zip(s["masks"], s["boxes"], s["scores"])]

            def run_box(label, bb):
                proc.reset_all_prompts(state)
                cx, cy = (bb[0] + bb[2]) / 2 / w, (bb[1] + bb[3]) / 2 / h
                bw, bh = (bb[2] - bb[0]) / w, (bb[3] - bb[1]) / h
                s = proc.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)
                if len(s["scores"]) == 0:
                    return {"label": label, "score": 0.0, "bbox": bb, "mask": None}
                k = int(torch.argmax(s["scores"]))
                return {"label": label, "score": float(s["scores"][k]), "bbox": bb, "mask": s["masks"][k][0].cpu().numpy()}

            if pipe == "ref":
                gt = load_gt(fid)
                for obj in (gt or {}).get("objects", []):
                    with Timer() as t:
                        preds = run_text(obj["name"])
                    save_pred(out_name, fid, preds, t.dt + tenc.dt, task="refer", query=obj["name"])
                print(fid, flush=True)
                continue
            with Timer() as t:
                if pipe == "qn":
                    names = json.loads((PRED / "qwen_names" / f"{fid}.json").read_text())["extra"]["names"]
                    preds = [p for q in names for p in run_text(q)]
                else:
                    d = json.loads((PRED / "qwen_boxes" / f"{fid}.json").read_text())
                    preds = [run_box(p["label"], p["bbox"]) for p in d["preds"]]
        save_pred(out_name, fid, preds, t.dt + tenc.dt, extra={"encode_s": tenc.dt})
        print(fid, len(preds), round(tenc.dt, 2), round(t.dt, 2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
