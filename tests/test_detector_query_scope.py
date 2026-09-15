"""A targeted grasp search must not hide unrelated objects from later looks."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.detector import OpenVocabDetector


@pytest.mark.parametrize("open_request", [None, []])
def test_targeted_query_is_followed_by_open_world_detection(monkeypatch, open_request):
    calls = []

    class Model:
        def __init__(self, path):
            self.labels = ["cube", "can", "lemon"] if str(path).endswith("-pf.pt") else []
        def get_text_pe(self, classes): return classes
        def set_classes(self, classes, embeddings=None): self.labels = list(classes)
        def predict(self, *args, **kwargs):
            calls.append(list(self.labels))
            return [SimpleNamespace(labels=list(self.labels))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setattr("cascade.device.resolve_device", lambda *a, **kw: "cpu")
    detector = OpenVocabDetector("scene.pt", device="cpu")
    # Parsing boxes is separately tested. This boundary retains the model's
    # actual requested vocabulary, the state that leaked in the real run.
    monkeypatch.setattr(detector, "_parse", lambda result, frame: result.labels)
    frame = SimpleNamespace(rgb=np.zeros((8, 8, 3), np.uint8))
    assert detector.detect(frame) == ["cube", "can", "lemon"]
    assert detector.detect(frame, ["pink cube"]) == ["pink cube"]
    assert detector.detect(frame, open_request) == ["cube", "can", "lemon"]
    assert detector.detect(frame, ["tomato can"]) == ["tomato can"]
    assert calls == [["cube", "can", "lemon"], ["pink cube"], ["cube", "can", "lemon"], ["tomato can"]]
