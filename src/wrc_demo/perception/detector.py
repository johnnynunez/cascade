"""Open-vocabulary object detection.

Primary backend: Ultralytics YOLOE / YOLO-World (weights already present in
the baseline repo's models/ dir). Both accept free-text class prompts, which
is what lets the agent say "detect the red mug" for objects never seen in
training. A SAM3-style text-prompted segmenter can be slotted in behind the
same interface later (see docs/ROADMAP.md).
"""

from __future__ import annotations

import abc
import logging

import numpy as np

from ..types import Detection, Frame

logger = logging.getLogger(__name__)


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        """Run detection; `classes` overrides the prompt vocabulary.

        `classes=None` means open-world: detect whatever is there. Passing an
        explicit list narrows the vocabulary for that call only.
        """

    @abc.abstractmethod
    def set_classes(self, classes: list[str] | None) -> None:
        """Restrict the vocabulary; `None`/`[]` restores open-world detection."""


class OpenVocabDetector(Detector):
    """Open-vocabulary detector with a genuinely object-agnostic default.

    Two modes, and the distinction matters for a public booth:

    ``prompt_free=True`` (the default) runs YOLOE's built-in vocabulary of
    ~4500 concepts, so ANY object a visitor puts on the table is detected and
    named without anyone having listed it in advance.  This is what
    "object-agnostic" has to mean when you cannot know the objects beforehand.

    ``prompt_free=False`` restricts the model to an explicit class list.  That
    is strictly a closed set: an object outside the list is invisible, no
    matter how clearly the camera sees it.  Only use it when the task really
    is "find these specific things".

    `set_classes()` still narrows the vocabulary on demand (the agent asking
    for one named object), and `set_classes(None)` restores the prompt-free
    vocabulary.
    """

    def __init__(
        self,
        model_path: str,
        classes: list[str] | None = None,
        device: str = "cuda:0",
        conf: float = 0.25,
        prompt_free: bool = True,
    ):
        from ultralytics import YOLO  # lazy heavy import

        self._model = YOLO(model_path)
        self._device = device
        self._conf = conf
        self._classes: list[str] = []
        self._filter_only = False  # closed-set model: filter by label instead
        self._prompt_free = False
        if classes:
            self.set_classes(classes)
        elif prompt_free:
            self.set_prompt_free()

    def set_prompt_free(self) -> None:
        """Detect anything in YOLOE's built-in vocabulary, with no prompts.

        Falls back to leaving the model as-is on any older ultralytics that
        lacks `get_vocab`/`set_vocab`; the caller then gets whatever default
        vocabulary the checkpoint ships with rather than a hard failure, since
        a booth demo must never fail to start over a detector nicety.
        """
        if self._prompt_free:
            return
        try:
            self._model.set_vocab(self._model.get_vocab([]), names=[])
        except (AttributeError, TypeError, ValueError) as e:  # pragma: no cover
            logger.warning("prompt-free vocabulary unavailable (%s); "
                           "detector keeps its current vocabulary", e)
            return
        self._prompt_free = True
        self._filter_only = False
        self._classes = []

    def set_classes(self, classes: list[str] | None) -> None:
        # None/[] means "stop restricting": go back to open-world detection
        # rather than silently keeping the last narrow vocabulary.
        if not classes:
            self.set_prompt_free()
            return
        if list(classes) == self._classes:
            return
        self._prompt_free = False
        # YOLOE needs text embeddings passed explicitly; YOLO-World does not.
        # Closed-set models (yolo11n, ...) have no set_classes at all: fall
        # back to post-filtering detections by label (self._filter_only).
        self._filter_only = False
        try:
            self._model.set_classes(classes, self._model.get_text_pe(classes))
        except (AttributeError, TypeError):
            try:
                self._model.set_classes(classes)
            except (AttributeError, TypeError):
                self._filter_only = True
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "open-vocabulary text prompts need ultralytics' CLIP fork: "
                "uv pip install git+https://github.com/ultralytics/CLIP.git "
                "(and 'mobileclip' for YOLOE models). Alternatively use a "
                f"closed-set model like yolo11n.pt. ({e})"
            ) from e
        self._classes = list(classes)

    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        if classes:
            self.set_classes(classes)
        results = self._model.predict(
            frame.rgb, conf=self._conf, device=self._device, verbose=False
        )
        dets = self._parse(results[0], frame)
        if self._filter_only and self._classes:
            # Loose word-overlap match so "red cup" still finds COCO's "cup".
            wanted = {w for c in self._classes for w in c.lower().split()}
            dets = [d for d in dets if wanted & set(d.label.lower().split())]
        return dets

    def _parse(self, result, frame: Frame) -> list[Detection]:
        dets: list[Detection] = []
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return dets
        h, w = frame.rgb.shape[:2]
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # (N, mh, mw) float
        for i in range(len(boxes)):
            label = names[int(boxes.cls[i])]
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().numpy().astype(np.float32)
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                if m.shape != (h, w):
                    import cv2

                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = m > 0.5
            obb = None
            if getattr(result, "obb", None) is not None and result.obb is not None:
                try:
                    obb = result.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                except (IndexError, AttributeError):
                    obb = None
            dets.append(Detection(label=label, conf=conf, bbox=xyxy, mask=mask, obb=obb))
        return dets


class MockDetector(Detector):
    """Returns pre-configured detections; understands the synthetic scene.

    If constructed without explicit detections it auto-detects the red box in
    frames produced by `mock_camera.synthetic_tabletop` via color threshold,
    so end-to-end tests exercise real mask geometry.
    """

    def __init__(self, detections: list[Detection] | None = None, label: str = "red cube"):
        self._fixed = detections
        self._label = label
        self._classes: list[str] = []

    def set_classes(self, classes: list[str] | None) -> None:
        self._classes = list(classes) if classes else []

    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        if classes:
            self.set_classes(classes)
        if self._fixed is not None:
            return list(self._fixed)
        # Behave like an open-vocab detector: only "find" the object when the
        # requested vocabulary loosely matches its label.
        if self._classes:
            words = {w for c in self._classes for w in c.lower().split()}
            mine = set(self._label.lower().split())
            if not (words & mine):
                return []
        bgr = frame.rgb
        red = (
            (bgr[:, :, 2].astype(int) - bgr[:, :, 0].astype(int) > 60)
            & (bgr[:, :, 2].astype(int) - bgr[:, :, 1].astype(int) > 60)
        )
        if not red.any():
            return []
        ys, xs = np.nonzero(red)
        bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        return [Detection(label=self._label, conf=0.95, bbox=bbox, mask=red)]
