"""Full-VLM open-vocabulary detection: perception on cosmos3-edge alone.

This is the "quitar YOLOE" exploration made selectable: instead of the
YOLOE-first / VLM-second cascade (detector.py + vlm_ground.py), run EVERY
detector pass through the vision LLM that already serves as the brain
(cosmos3-edge on vLLM, or any OpenAI-compatible vision server).

What it buys and what it costs, so the choice stays honest:

* buys: one model for perception AND reasoning (nothing to download beside
  the brain, no ultralytics/torch in the loop venv), genuinely open-world
  naming (a VLM reads text, brands, unusual booth objects YOLOE's text
  embeddings miss), no CWD-relative mobileclip trap.
* costs: one call is ~1-3 s on a local vLLM server vs ~30 ms for YOLOE, so
  the always-on watcher drops from 3 Hz to well under 1 Hz -- belief
  freshness gates (`belief_fallback_age_s`) and motion-watchdog windows are
  sized for the fast detector. And a VLM returns BOXES, not masks: grasping
  degrades to bbox-window depth crops (grounding._bbox_mask), which is
  exactly the mask-quality difference the segmenter exists to paper over --
  `grasp_at_pixel` + SAM keeps working and is the recommended grasp path in
  this mode.

Wire format: the model is asked for a JSON array of every distinct physical
object with a 0-1000-normalized bbox. Normalized coordinates are the same
empirically-validated convention as vlm_ground.py: the mmproj resizes
internally, and absolute pixel answers come back consistently shifted.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from ..types import Detection, Frame
from .detector import Detector

logger = logging.getLogger(__name__)

_DETECT_PROMPT = (
    "You are the perception system of a tabletop robot. List EVERY distinct "
    "physical object visible on or above the table in this image.\n"
    "Rules:\n"
    "- one entry per object instance (two cubes = two entries)\n"
    "- short lowercase names (color + noun when obvious: 'red cube')\n"
    "- skip the robot arm itself, the table surface, walls and background\n"
    "Output ONLY a JSON array, no prose, no code fences. Each element:\n"
    '{"label": "<name>", "bbox_2d": [x0, y0, x1, y1], "confidence": 0.0-1.0}\n'
    "with bbox_2d normalized to 0-1000 of the full image.\n"
    "Empty scene: output []"
)

_RESTRICT_PROMPT = (
    "Only report objects matching these descriptions (skip everything "
    "else): {classes}"
)


def _extract_json_array(text: str) -> list | None:
    """The LAST parseable top-level JSON array in a VLM reply.

    Regexes break on nested brackets (bbox arrays inside objects inside the
    array), so walk balanced '['..']' spans explicitly. Thinking blocks are
    stripped first: reasoning traces routinely quote example JSON before the
    answer, and the answer is what comes last.
    """
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    best = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth = 0
        end = None
        for j in range(i, n):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    try:
                        cand = json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(cand, list):
                        best = cand
                    break
        i = (end + 1) if end is not None else (i + 1)
    return best


def parse_detections(text: str, w: int, h: int, min_conf: float = 0.0) -> list[Detection]:
    """VLM reply -> Detections (0-1000 normalized -> pixels, no masks)."""
    arr = _extract_json_array(text)
    if not arr:
        return []
    dets: list[Detection] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        box = item.get("bbox_2d") or item.get("bbox")
        if not label or not isinstance(box, list) or len(box) < 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
        except (TypeError, ValueError):
            continue
        x0, x1 = x0 * w / 1000.0, x1 * w / 1000.0
        y0, y1 = y0 * h / 1000.0, y1 * h / 1000.0
        # Order FIRST, then clip: a reversed box (x1 < x0, which VLMs do
        # emit) clipped before ordering keeps an out-of-range coordinate.
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, x1 = max(0.0, x0), min(float(w), x1)
        y0, y1 = max(0.0, y0), min(float(h), y1)
        if (x1 - x0) < 4 or (y1 - y0) < 4:
            continue
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = float(np.clip(conf, 0.0, 1.0))
        if conf < min_conf:
            continue
        dets.append(Detection(
            label=label, conf=conf,
            bbox=np.array([x0, y0, x1, y1], dtype=np.float32),
        ))
    return dets


class VLMDetector(Detector):
    """Detector backed by an OpenAI-compatible vision LLM (cosmos3-edge).

    Failure shape matches the booth rule everywhere else in this repo: a
    down/slow/garbled server yields an EMPTY detection list (logged), never
    an exception -- the perception watcher must keep its heartbeat while the
    brain server restarts, and "no detections this pass" already has a
    well-defined meaning upstream (beliefs age, nothing new fuses).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 20.0,
        conf: float = 0.25,
        max_tokens: int = 1024,
        jpeg_quality: int = 85,
    ):
        from openai import OpenAI  # lazy heavy import

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model = model
        self._conf = float(conf)
        self._max_tokens = int(max_tokens)
        self._quality = int(jpeg_quality)
        self._classes: list[str] = []

    def set_classes(self, classes: list[str] | None) -> None:
        # A VLM holds no vocabulary state; the restriction is prompt text.
        self._classes = list(classes) if classes else []

    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        import base64

        import cv2

        if classes:
            self.set_classes(classes)
        h, w = frame.rgb.shape[:2]
        # Frame.rgb is BGR (OpenCV) despite the name; imencode expects BGR,
        # so the JPEG comes out right-side-up color-wise. Same as vlm_ground.
        ok, buf = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return []
        b64 = base64.b64encode(buf.tobytes()).decode()
        prompt = _DETECT_PROMPT
        if self._classes:
            prompt += "\n" + _RESTRICT_PROMPT.format(classes=", ".join(self._classes))
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=self._max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
        except Exception as e:  # server down/timeout: degrade, never crash
            logger.warning("VLM detector call failed (%s); returning no detections", e)
            return []
        text = resp.choices[0].message.content or ""
        return parse_detections(text, w, h, min_conf=self._conf)
