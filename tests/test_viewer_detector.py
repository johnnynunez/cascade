"""cascade-view's detector: the demo config's, never a separate weights file.

The viewer used to load its own YOLO weights (`--detect <weights.pt>`). It now
builds the detector the demo config names for the viewed camera
(`apps.demo._make_detector`) and passes `--classes` as per-call queries. A
detector whose class defines `detect_nowait` (the brain detector) is asked
that way, so a multi-second brain call never freezes the window.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from cascade.apps import viewer
from cascade.perception.detector import MockDetector
from cascade.types import Frame


def _frame():
    return Frame(rgb=np.zeros((8, 8, 3), np.uint8), depth_m=None, K=np.eye(3))


def test_the_mock_camera_gets_its_pinned_mock_detector():
    assert type(viewer.make_viewer_detector("mock")) is MockDetector


def test_the_detector_comes_from_the_demo_config_of_the_viewed_camera(monkeypatch):
    import cascade.apps.demo as demo

    seen = []
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: seen.append(cfg) or "built")
    assert viewer.make_viewer_detector("l515") == "built"
    [cfg] = seen
    from cascade.config import load_profile

    profile = load_profile("cameras", "l515")
    assert (cfg.camera.get("type"), cfg.camera.get("width")) == (profile.get("type"), profile.get("width"))
    assert cfg.detector.get("type") == "brain"


def test_a_detector_with_detect_nowait_is_never_asked_to_block():
    class Brainish:
        def __init__(self):
            self.calls = []

        def detect(self, frame, classes=None):
            raise AssertionError("the viewer must not block on detect()")

        def detect_nowait(self, frame, classes=None):
            self.calls.append(classes)
            return ["cached"]

    det = Brainish()
    assert viewer.viewer_detections(det, _frame(), ["red cup"]) == ["cached"]
    assert viewer.viewer_detections(det, _frame(), None) == ["cached"]
    assert det.calls == [["red cup"], None]


def test_a_detector_without_detect_nowait_runs_detect_with_the_classes():
    calls = []

    class Plain:
        def detect(self, frame, classes=None):
            calls.append(classes)
            return []

    assert viewer.viewer_detections(Plain(), _frame(), ["bowl"]) == []
    assert calls == [["bowl"]]


class _CvError(Exception):
    """Stand-in for cv2.error."""


def test_a_headless_window_failure_still_closes_the_camera_and_the_detector(monkeypatch):
    """opencv-python-headless raises from namedWindow (and from
    destroyAllWindows): the camera and the detector (its brain refreshes, the
    SAM warm-up) are released anyway, and the window error is what surfaces."""
    events = []

    class Cam:
        def open(self):
            events.append("cam open")

        def warm_up(self, n):
            pass

        def get_frame(self):
            raise AssertionError("no frame is read without a window")

        def close(self):
            events.append("cam close")

    class Det:
        def detect(self, frame, classes=None):
            return []

        def close(self):
            events.append("detector close")

    def no_gui(*args, **kwargs):
        raise _CvError("namedWindow: the function is not implemented (headless build)")

    def no_gui_destroy():
        raise _CvError("destroyAllWindows: the function is not implemented (headless build)")

    fake_cv2 = types.SimpleNamespace(WINDOW_NORMAL=0, error=_CvError, namedWindow=no_gui,
                                     destroyAllWindows=no_gui_destroy)
    monkeypatch.setattr(viewer, "cv2", fake_cv2)
    monkeypatch.setattr(viewer, "make_camera", lambda profile: Cam())
    monkeypatch.setattr(viewer, "make_viewer_detector", lambda camera: Det())
    with pytest.raises(_CvError, match="namedWindow"):
        viewer.main(["--camera", "mock", "--detect"])
    assert events == ["cam open", "cam close", "detector close"]
