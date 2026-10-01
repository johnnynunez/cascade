"""Vocabulary startup must not reload models or reuse image detections."""
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.detector import OpenVocabDetector
from cascade.perception.world import LockedDetector


@pytest.fixture
def models(monkeypatch):
    loaded, encoded, predicted = [], [], []

    class Model:
        def __init__(self, path):
            self.path = str(path)
            self.labels = ["cube", "can", "lemon"] if self.path.endswith("-pf.pt") else []
            self.fail_apply = False
            self.applied = []
            loaded.append(self)

        def get_text_pe(self, classes):
            encoded.append(tuple(classes))
            return SimpleNamespace(labels=list(classes))

        def set_classes(self, classes, embeddings):
            if self.fail_apply:
                raise RuntimeError("cannot apply vocabulary")
            assert list(classes) == embeddings.labels
            self.applied.append(tuple(classes))
            self.labels = list(classes)

        def predict(self, rgb, **kwargs):
            predicted.append((self.path, list(self.labels), rgb))
            return [SimpleNamespace(labels=list(self.labels))]

    monkeypatch.delenv("CASCADE_REQUIRE_CUDA", raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setattr("cascade.device.resolve_device", lambda *a, **kw: "cpu")
    detector = OpenVocabDetector("scene.pt", device="cpu")
    monkeypatch.setattr(detector, "_parse", lambda result, frame: result.labels)
    return SimpleNamespace(detector=detector, loaded=loaded, encoded=encoded, predicted=predicted)


def frame(value=0):
    return SimpleNamespace(rgb=np.full((8, 8, 3), value, np.uint8))


def test_preparation_survives_intervening_watcher_without_caching_images(models):
    d = models.detector
    d.prepare(["orange"])
    assert not models.predicted
    first, second = frame(), frame(1)
    assert d.detect(first) == ["cube", "can", "lemon"]
    assert d.detect(second, ["orange"]) == ["orange"]
    assert d.detect(first) == ["cube", "can", "lemon"]
    assert d.detect(second, ["orange"]) == ["orange"]
    assert [m.path for m in models.loaded] == ["scene-pf.pt", "scene.pt"]
    assert models.encoded == [("orange",)]
    assert models.loaded[1].applied == [("orange",)]
    assert len(models.predicted) == 4
    assert models.predicted[1][2] is second.rgb
    assert models.predicted[3][2] is second.rgb


def test_embedding_cache_is_bounded_and_eviction_reencodes(models):
    d = models.detector
    for i in range(9):
        d.prepare([f"class{i}"])
    assert len(d._text_embeddings) == 8
    assert ("class0",) not in d._text_embeddings
    d.prepare(["class1"])
    assert len(models.encoded) == 9
    d.prepare(["class0"])
    assert models.encoded[-1] == ("class0",)
    assert len(models.encoded) == 10
    assert ("class2",) not in d._text_embeddings


def test_ordered_vocabularies_do_not_share_embeddings(models):
    d = models.detector
    d.prepare(["orange", "can"])
    d.prepare(["can", "orange"])
    d.prepare(["orange", "can"])
    assert models.encoded == [("orange", "can"), ("can", "orange")]
    assert d.detect(frame(), ["orange", "can"]) == ["orange", "can"]


def test_failed_application_does_not_cache_vocabulary(models):
    d = models.detector
    d.prepare(["orange"])
    d._model.fail_apply = True
    with pytest.raises(RuntimeError, match="cannot apply"):
        d.prepare(["can"])
    assert ("can",) not in d._text_embeddings
    assert d._classes == ["orange"]


def test_new_model_attestation_failure_restores_active_model(models, monkeypatch):
    d = models.detector
    original = d._model
    def fail():
        raise RuntimeError("not CUDA")
    monkeypatch.setattr(d, "_prepare_cuda_model", fail)
    with pytest.raises(RuntimeError, match="not CUDA"):
        d.prepare(["orange"])
    assert d._model is original and d._prompt_free
    assert list(d._models) == ["scene-pf.pt"]


def test_cached_embedding_is_rechecked_for_cuda(models, monkeypatch):
    d = models.detector
    d.prepare(["orange"])
    d.prepare(["can"])
    d.set_prompt_free()
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "1")
    with pytest.raises(RuntimeError, match="embeddings did not use CUDA"):
        d.prepare(["orange"])
    assert len(models.encoded) == 2


def test_locked_detector_preparation_excludes_detection():
    entered, release, detected = threading.Event(), threading.Event(), threading.Event()
    def prepare(classes):
        entered.set()
        assert release.wait(2.)
    def detect(image, classes=None):
        detected.set()
        return []
    d = LockedDetector(SimpleNamespace(prepare=prepare, detect=detect))
    worker = threading.Thread(target=d.prepare, args=(["orange"],))
    query = threading.Thread(target=d.detect, args=(frame(),))
    worker.start()
    try:
        assert entered.wait(2.)
        query.start()
        assert not detected.wait(.02)
    finally:
        release.set()
        worker.join(2.)
        if query.ident is not None:
            query.join(2.)
    assert not worker.is_alive() and not query.is_alive() and detected.is_set()


def test_legacy_detector_needs_no_preparation():
    d = LockedDetector(SimpleNamespace(detect=lambda frame, classes=None: ["legacy"]))
    d.prepare(["orange"])
    assert d.detect(frame()) == ["legacy"]


def test_world_style_setter_remains_supported(monkeypatch):
    loads, applied = [], []
    class Model:
        def __init__(self, path):
            loads.append(str(path))
            self.labels = ["open scene"]
        def set_classes(self, classes):
            applied.append(list(classes))
            self.labels = list(classes)
        def predict(self, *args, **kwargs):
            return [SimpleNamespace(labels=list(self.labels))]
    monkeypatch.delenv("CASCADE_REQUIRE_CUDA", raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setattr("cascade.device.resolve_device", lambda *a, **kw: "cpu")
    d = OpenVocabDetector("scene.pt", device="cpu")
    monkeypatch.setattr(d, "_parse", lambda result, frame: result.labels)
    d.prepare(["orange"])
    assert d.detect(frame()) == ["open scene"]
    assert d.detect(frame(), ["orange"]) == ["orange"]
    assert applied == [["orange"]] and loads == ["scene-pf.pt", "scene.pt"]
