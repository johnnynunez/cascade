"""Candidate pipelines for the perception bake-off (throwaway evaluation code).

  .venv/bin/python tools/run_candidates.py <pipeline> [--task inventory|refer]

Pipelines (inventory = list everything on the table, refer = find the object named in the ground truth):
  qwen_boxes        brain lists objects with bbox_2d (0-1000)                        [B]
  qwen_names        brain lists object kinds only (input for the *_qn pipelines)
  qwen_boxes_sam2   brain boxes -> SAM2.1-L box prompts -> masks                    [B + masks]
  llmdet_qn         brain names -> LLMDet-L boxes -> SAM2.1-L masks                  [A, no SAM 3]
  yoloe26_qn        brain names -> YOLOE-26-X text prompts (masks)                   [A, fast]
  sam3_qn           brain names -> SAM 3 text prompts (masks)                        [A]
  llmdet_lvis       LLMDet-L over the LVIS vocabulary, no brain                       [C]
  yoloe26_pf        YOLOE-26-X prompt-free, no brain                                  [C]
  refer: ref_qwen, ref_llmdet, ref_yoloe26, ref_sam3 (query = ground-truth name)
No camera or robot is touched; frames are saved JPEGs.
"""
import argparse
import base64
import json
import pathlib
import re
import sys

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import FRAMES, PRED, ROOT, Timer, frame_ids, load_gt, save_pred  # noqa: E402

W8 = ROOT / "weights"
import os
BRAIN_URL = os.environ.get("BRAIN_URL", "http://127.0.0.1:8081/v1")
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "qwen3.8-27b")
BRAIN_TAG = os.environ.get("BRAIN_TAG", "qwen")  # output prefix: <tag>_boxes, ref_<tag>
BRAIN_EXTRA = json.loads(os.environ.get("BRAIN_EXTRA", "{}"))  # e.g. chat_template_kwargs
BRAIN_MAXTOK = int(os.environ.get("BRAIN_MAXTOK", "1500"))
KEY = (pathlib.Path.home() / ".cascade/nemoclaw/llama.key").read_text().strip()

EXCLUDE = ("Exclude the robot arm and its gripper, people and anything in their hands, screens, monitors, "
           "laptops, keyboards, furniture, walls and the table itself.")
P_BOXES = ("You are the perception module of a tabletop robot arm. List every manipulable object resting on the "
           "table or inside a container on the table, one entry per instance. " + EXCLUDE + " Give each a short "
           "specific noun phrase (e.g. 'red plush lobster', 'green filament spool', 'paper cup', 'gray plastic "
           "basket') and a tight bounding box. Answer ONLY with JSON: [{\"label\": \"...\", \"bbox_2d\": [x1, y1, "
           "x2, y2]}] with coordinates normalized to 0-1000. Answer [] if there is none.")
P_NAMES = ("You are the perception module of a tabletop robot arm. Name every kind of manipulable object resting "
           "on the table or inside a container on the table. " + EXCLUDE + " Answer ONLY with a JSON list of short "
           "noun phrases usable as open-vocabulary detector prompts, one per distinct kind of object (not per "
           "instance), e.g. [\"red plush lobster\", \"paper cup\"]. Answer [] if there is none.")
P_REFER = ("Locate every instance of \"{q}\" in the image. Answer ONLY with JSON: [{{\"label\": \"{q}\", "
           "\"bbox_2d\": [x1, y1, x2, y2]}}] with coordinates normalized to 0-1000, or [] if it is not visible.")


def img(fid):
    return Image.open(FRAMES / f"{fid}.jpg").convert("RGB")


# ---------------------------------------------------------------- brain
_client = None


def brain(prompt, fid):
    global _client
    from openai import OpenAI
    _client = _client or OpenAI(base_url=BRAIN_URL, api_key=KEY)
    b64 = base64.b64encode((FRAMES / f"{fid}.jpg").read_bytes()).decode()
    r = _client.chat.completions.create(
        model=BRAIN_MODEL, temperature=0, max_tokens=BRAIN_MAXTOK, extra_body=BRAIN_EXTRA or None,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt}]}])
    return r.choices[0].message.content or ""


def parse_json(text):
    """First JSON list in the reply (the brain sometimes appends a second '[]' or prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\[", text):
        try:
            v, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(v, list):
            return v
    return []


def brain_boxes(text, w, h):
    out = []
    for o in parse_json(text):
        if not isinstance(o, dict) or "bbox_2d" not in o:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in o["bbox_2d"]]
        except (TypeError, ValueError):
            continue
        out.append({"label": str(o.get("label", "object")), "score": 1.0,
                    "bbox": [x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], "mask": None})
    return out


def qwen_names_for(fid):
    p = PRED / "qwen_names" / f"{fid}.json"
    return json.loads(p.read_text())["extra"]["names"]


def astra_labels_for(fid):
    d = json.loads((PRED / "astra_boxes" / f"{fid}.json").read_text())
    return list(dict.fromkeys(p["label"] for p in d["preds"]))


# ---------------------------------------------------------------- SAM 2.1 box -> mask
_sam2 = None


def sam2_masks(fid, preds):
    global _sam2
    from ultralytics import SAM
    _sam2 = _sam2 or SAM(str(W8 / "sam2.1_l.pt"))
    if not preds:
        return preds
    r = _sam2.predict(str(FRAMES / f"{fid}.jpg"), bboxes=[p["bbox"] for p in preds], verbose=False)[0]
    ms = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else []
    for p, m in zip(preds, ms):
        p["mask"] = m
    return preds


# ---------------------------------------------------------------- eager weight loading
def eager_from_pretrained(cls, path, **kw):
    """GB10: copying memory-mapped tensors runs at ~5 MB/s (even from tmpfs), so transformers' default
    mmap load + .cuda() takes many minutes. Keep transformers' loader (it renames legacy checkpoint
    keys; a hand-built state_dict skips that and silently leaves layers random) but force safetensors'
    pread backend, which transformers itself uses on MPS/Windows."""
    import safetensors
    import transformers
    import transformers.modeling_utils as mu
    if int(transformers.__version__.split(".")[0]) < 5:
        # transformers 4.x: no backend switch, but it accepts path + state_dict and the
        # checkpoint keys already match 4.x naming (no conversion needed).
        import safetensors.torch
        from transformers import AutoConfig
        sd = {}
        for f in sorted(pathlib.Path(path).glob("*.safetensors")):
            sd.update(safetensors.torch.load(f.read_bytes()))
        config = AutoConfig.from_pretrained(path)
        if hasattr(cls, "_model_mapping"):
            cls = cls._model_mapping[type(config)]
        return cls.from_pretrained(None, config=config, state_dict=sd, **kw)
    if not getattr(mu.safe_open, "_pread", False):
        orig = safetensors.safe_open

        def _pread_open(*a, **k):
            k["backend"] = "pread"
            return orig(*a, **k)
        _pread_open._pread = True
        mu.safe_open = _pread_open
    return cls.from_pretrained(path, **kw)


# ---------------------------------------------------------------- LLMDet
_llm = None


def llmdet(fid, labels, thr=0.3):
    global _llm
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    if _llm is None:
        _llm = (AutoProcessor.from_pretrained(W8 / "llmdet_large"),
                eager_from_pretrained(AutoModelForZeroShotObjectDetection, W8 / "llmdet_large").cuda().eval())
    proc, model = _llm
    im = img(fid)
    out = []
    for i in range(0, len(labels), 40):
        chunk = labels[i:i + 40]
        inputs = proc(images=im, text=[chunk], return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**inputs)
        res = proc.post_process_grounded_object_detection(o, threshold=thr, target_sizes=[im.size[::-1]])[0]
        for box, score, lab in zip(res["boxes"], res["scores"], res.get("text_labels", res.get("labels"))):
            out.append({"label": str(lab), "score": float(score), "bbox": box.tolist(), "mask": None})
    return nms(out)


def nms(preds, iou=0.6):
    preds = sorted(preds, key=lambda p: -p["score"])
    keep = []
    for p in preds:
        if all(box_iou(p["bbox"], k["bbox"]) < iou for k in keep):
            keep.append(p)
    return keep


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def lvis_names():
    import ultralytics
    import yaml
    y = yaml.safe_load((pathlib.Path(ultralytics.__file__).parent / "cfg/datasets/lvis.yaml").read_text())
    return [n.replace("_", " ").split("(")[0].strip() for n in y["names"].values()]


# ---------------------------------------------------------------- YOLOE-26
_y26 = {}


def yoloe26(fid, labels=None, conf=0.25):
    from ultralytics import YOLOE
    key = "pf" if labels is None else "txt"
    if key not in _y26:
        _y26[key] = YOLOE("yoloe-26x-seg-pf.pt" if labels is None else str(W8 / "yoloe26-x-seg/model.pt"))
    m = _y26[key]
    if labels is not None:
        m.set_classes(labels, m.get_text_pe(labels))
    r = m.predict(str(FRAMES / f"{fid}.jpg"), conf=conf, retina_masks=True, verbose=False)[0]
    out = []
    ms = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else [None] * len(r.boxes)
    for b, m_ in zip(r.boxes, ms):
        out.append({"label": r.names[int(b.cls)], "score": float(b.conf), "bbox": b.xyxy[0].tolist(), "mask": m_})
    return out


# ---------------------------------------------------------------- SAM 3 (text prompts)
_sam3 = None


def sam3(fid, labels, thr=0.5):
    global _sam3
    import torch
    from transformers import Sam3Model, Sam3Processor
    if _sam3 is None:
        src = str(W8 / "sam3")
        _sam3 = (Sam3Processor.from_pretrained(src), eager_from_pretrained(Sam3Model, src).cuda().eval())
    proc, model = _sam3
    im = img(fid)
    out = []
    for lab in labels:
        inputs = proc(images=im, text=lab, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**inputs)
        res = proc.post_process_instance_segmentation(o, threshold=thr, mask_threshold=0.5,
                                                      target_sizes=inputs.get("original_sizes").tolist())[0]
        for m, b, s in zip(res["masks"], res["boxes"], res["scores"]):
            out.append({"label": lab, "score": float(s), "bbox": b.tolist(), "mask": m.cpu().numpy().astype(bool)})
    return out


# ---------------------------------------------------------------- driver
def run(pipeline):
    out = pipeline
    if BRAIN_TAG != "qwen":
        pipeline = pipeline.replace(BRAIN_TAG, "qwen")
    ids = frame_ids()
    for fid in ids:
        w, h = img(fid).size
        if pipeline.startswith("ref_"):
            gt = load_gt(fid)
            if gt is None:
                continue
            for obj in gt["objects"]:
                q = obj["name"]
                with Timer() as t:
                    if pipeline == "ref_qwen":
                        raw = brain(P_REFER.format(q=q), fid); preds = brain_boxes(raw, w, h)
                    elif pipeline == "ref_llmdet":
                        preds = llmdet(fid, [q])
                    elif pipeline == "ref_yoloe26":
                        preds = yoloe26(fid, [q])
                    elif pipeline == "ref_sam3":
                        preds = sam3(fid, [q])
                save_pred(out, fid, preds, t.dt, task="refer", query=q)
            print(fid, flush=True)
            continue
        extra = {}
        with Timer() as t:
            if pipeline in ("qwen_boxes", "qwen_boxes_sam2"):
                src = PRED / out.replace("_sam2", "") / f"{fid}.json"
                if pipeline == "qwen_boxes_sam2" and src.exists():
                    d = json.loads(src.read_text())
                    preds = brain_boxes(d["extra"]["raw"], w, h) if d["extra"].get("raw") else \
                        [{"label": p["label"], "score": 1.0, "bbox": p["bbox"], "mask": None} for p in d["preds"]]
                else:
                    raw = brain(P_BOXES, fid); preds = brain_boxes(raw, w, h); extra["raw"] = raw
                if pipeline == "qwen_boxes_sam2":
                    preds = sam2_masks(fid, preds)
            elif pipeline == "qwen_names":
                raw = brain(P_NAMES, fid)
                names = [str(n) for n in parse_json(raw) if isinstance(n, (str, int, float))]
                preds, extra = [], {"raw": raw, "names": names}
            elif pipeline == "llmdet_qn":
                preds = sam2_masks(fid, llmdet(fid, qwen_names_for(fid)))
            elif pipeline == "yoloe26_qn":
                names = qwen_names_for(fid)
                preds = yoloe26(fid, names) if names else []
            elif pipeline == "sam3_qn":
                preds = sam3(fid, qwen_names_for(fid))
            elif pipeline == "sam3_an":
                preds = sam3(fid, astra_labels_for(fid))
            elif pipeline == "astra_boxes_sam2":
                d = json.loads((PRED / "astra_boxes" / f"{fid}.json").read_text())
                preds = sam2_masks(fid, [{"label": p["label"], "score": 1.0, "bbox": p["bbox"], "mask": None}
                                         for p in d["preds"]])
            elif pipeline == "llmdet_lvis":
                preds = sam2_masks(fid, llmdet(fid, lvis_names(), thr=0.35))
            elif pipeline == "yoloe26_pf":
                preds = yoloe26(fid, None)
            else:
                raise SystemExit(f"unknown pipeline {pipeline}")
        save_pred(out, fid, preds, t.dt, extra=extra)
        print(fid, len(preds), round(t.dt, 2), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("pipeline")
    run(a.parse_args().pipeline)
