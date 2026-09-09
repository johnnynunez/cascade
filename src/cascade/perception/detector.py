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
        device: str = "auto",
        conf: float = 0.25,
        prompt_free: bool = True,
    ):
        from ultralytics import YOLO  # lazy heavy import

        from ..device import resolve_device

        self._model_path = str(model_path)
        self._device = resolve_device(device, what="detector")
        self._conf = conf
        self._classes: list[str] = []
        self._filter_only = False  # closed-set model: filter by label instead
        self._prompt_free = False
        # Load the prompt-free checkpoint straight away when no vocabulary is
        # pinned, rather than loading the promptable one and replacing it.
        pf = self._pf_path() if (prompt_free and not classes) else None
        self._model = YOLO(pf or model_path)
        if pf is not None:
            self._prompt_free = True
        elif classes:
            self.set_classes(classes)

    def set_prompt_free(self) -> None:
        """Switch to the prompt-free checkpoint: detect anything, no prompts.

        YOLOE ships a separate `-pf` checkpoint whose ~4.5k-concept vocabulary
        is already fused into the classification head. That is the only real
        prompt-free path: `get_vocab([])` on a normal checkpoint raises inside
        the text encoder, because an empty prompt list has nothing to embed.

        A booth demo must never fail to start over a detector nicety, so a
        missing `-pf` checkpoint logs and leaves the model as it was.
        """
        if self._prompt_free:
            return
        pf_path = self._pf_path()
        if pf_path is None:
            logger.warning(
                "no prompt-free checkpoint alongside %s; perception stays "
                "restricted to the configured vocabulary", self._model_path,
            )
            return
        from ultralytics import YOLO  # lazy heavy import

        self._model = YOLO(pf_path)
        self._prompt_free = True
        self._filter_only = False
        self._classes = []

    def _pf_path(self) -> str | None:
        """`yoloe-11s-seg.pt` -> `yoloe-11s-seg-pf.pt`, if that name is usable.

        Ultralytics resolves a bare checkpoint name against its asset release,
        downloading on first use, so a local file is not required.
        """
        base = str(self._model_path)
        if base.endswith("-pf.pt"):
            return base
        if not base.endswith(".pt"):
            return None
        return base[: -len(".pt")] + "-pf.pt"

    def set_classes(self, classes: list[str] | None) -> None:
        # None/[] means "stop restricting": go back to open-world detection
        # rather than silently keeping the last narrow vocabulary.
        if not classes:
            self.set_prompt_free()
            return
        if list(classes) == self._classes and not self._prompt_free:
            return
        if self._prompt_free:
            # The -pf head is fused to its own vocabulary and cannot take
            # text prompts; reload the promptable checkpoint to narrow it.
            from ultralytics import YOLO  # lazy heavy import

            self._model = YOLO(self._model_path)
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


def _rule_red(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[:, :, i].astype(int) for i in range(3))
    return (r - b > 60) & (r - g > 60)


def _rule_blue(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[:, :, i].astype(int) for i in range(3))
    return (b - r > 60) & (b - g > 40)


def _rule_green(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[:, :, i].astype(int) for i in range(3))
    return (g - r > 50) & (g - b > 50)


#: BGR-threshold rules per colour word, for the mock detector. The SO-101 mesh
#: is yellow (R ~ G, low B) and the table light grey (R ~ G ~ B), so none of
#: these fire on the robot or the background in the rendered scene.
_COLOR_RULES = {"red": _rule_red, "blue": _rule_blue, "green": _rule_green}


class MockDetector(Detector):
    """Returns pre-configured detections; understands the synthetic scene.

    If constructed without explicit detections it auto-detects the red box in
    frames produced by `mock_camera.synthetic_tabletop` via color threshold,
    so end-to-end tests exercise real mask geometry.
    """

    def __init__(self, detections: list[Detection] | None = None, label: str = "red cube",
                 extra_labels: list[str] | None = None):
        self._fixed = detections
        self._label = label
        #: further colour-keyed props ("blue cube", "green cube"): each is
        #: found by its own BGR threshold (`_COLOR_RULES`), tuned to the
        #: rgba the generated MuJoCo scene paints (sim/demo_scene.PROP_RGBA)
        #: and to the mock camera's synthetic colours.
        self._extra_labels = list(extra_labels or [])
        self._classes: list[str] = []

    def set_classes(self, classes: list[str] | None) -> None:
        self._classes = list(classes) if classes else []

    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        if classes:
            self.set_classes(classes)
        if self._fixed is not None:
            return list(self._fixed)
        out: list[Detection] = []
        for label in [self._label, *self._extra_labels]:
            color = next((c for c in _COLOR_RULES if c in label.lower()), "red")
            # Behave like an open-vocab detector: only "find" an object when
            # the requested vocabulary loosely matches its label -- and when
            # the vocabulary names a colour, only THAT colour's prop. With
            # two props "blue cube" shares the token "cube" with "red cube";
            # returning both for a blue query handed localize() the red prop
            # first (equal confidence) and put the blue belief 3.5 cm off --
            # one cube width, on the wrong cube.
            if self._classes:
                words = {w for c in self._classes for w in c.lower().split()}
                if not (words & set(label.lower().split())):
                    continue
                asked = words & set(_COLOR_RULES)
                if asked and color not in asked:
                    continue
            mask = _COLOR_RULES[color](frame.rgb)
            if not mask.any():
                continue
            # one blob per colour: the generated scenes never paint two props
            # of the same colour, so the union IS the object
            ys, xs = np.nonzero(mask)
            bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
            out.append(Detection(label=label, conf=0.95, bbox=bbox, mask=mask))
        return out
