"""Object detection: the `Detector` interface and the offline `MockDetector`.

The shipped backend is `perception.brain_detector.BrainDetector` (`type:
brain`): the brain (a vision LLM) names and boxes the objects, a box-prompted
SAM masks them. Free-text queries are what let the agent say "find the red
mug" for objects never seen in training. Detectors are built from the config
by `apps.demo._make_detector`.
"""

from __future__ import annotations

import abc

import numpy as np

from ..types import Detection, Frame


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


def detect_vocabulary(detector, frame: Frame, classes: list[str] | None) -> list[Detection]:
    """One pass over the configured default vocabulary (`detect_classes`;
    None = open world), for the runtime's own looks: describe_scene,
    _reobserve, the localization vocabulary round.

    A detector whose CLASS defines `detect_vocabulary` (the brain detector,
    and `world.LockedDetector` forwarding it) serves it from its per-camera
    cache, the entry the watcher keeps warm; any other detector gets
    `detect()`. Only the class counts: a Mock answers every attribute. Never
    for an agent's own query: "find the X" stays a fresh `detect(frame, [X])`.
    """
    if callable(getattr(type(detector), "detect_vocabulary", None)):
        return detector.detect_vocabulary(frame, classes=classes)
    return detector.detect(frame, classes=classes)


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
                 extra_labels: list[str] | None = None, instances: bool = False):
        self._fixed = detections
        self._label = label
        #: further colour-keyed props ("blue cube", "green cube"): each is
        #: found by its own BGR threshold (`_COLOR_RULES`), tuned to the
        #: rgba the generated MuJoCo scene paints (sim/demo_scene.PROP_RGBA)
        #: and to the mock camera's synthetic colours.
        self._extra_labels = list(extra_labels or [])
        #: opt-in (`detector: {type: mock, instances: true}`): one detection
        #: per connected blob of a colour instead of one per colour, so two
        #: same-colour props that do not touch IN THE IMAGE come back as two
        #: instances. Off by default: every shipped scene paints one prop per
        #: colour, and touching props stay one blob either way (that is a
        #: detector limit; the belief store cannot split one detection).
        self._instances = bool(instances)
        self._classes: list[str] = []

    def set_classes(self, classes: list[str] | None) -> None:
        self._classes = list(classes) if classes else []

    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        # Same contract as the real detector: `classes=None` means OPEN
        # WORLD, not "keep the last narrow vocabulary". The mock used to
        # remember the `["red cube"]` a grasp had asked for and hide the blue
        # prop from every later open scan -- reset_scene re-observed a table
        # with two visible cubes and reported one.
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
            if self._instances:
                out.extend(Detection(label=label, conf=0.95, bbox=_mask_bbox(blob), mask=blob)
                           for blob in _blobs(mask))
                continue
            # one blob per colour: the generated scenes never paint two props
            # of the same colour, so the union IS the object
            ys, xs = np.nonzero(mask)
            bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
            out.append(Detection(label=label, conf=0.95, bbox=bbox, mask=mask))
        return out


#: Smallest connected blob the instance mode reports (pixels): anti-aliased
#: edge specks of a colour are not objects.
_MIN_BLOB_PX = 20


def _mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _blobs(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components of a colour mask, largest first (cv2 is a base
    dependency; scipy is not)."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= _MIN_BLOB_PX]
    keep.sort(key=lambda k: (-int(stats[k, cv2.CC_STAT_AREA]), k))
    return [labels == k for k in keep]
