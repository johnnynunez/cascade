"""LockedDetector: reservations are re-entrant, preparation excludes detection,
and a detector without `prepare` needs none."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.world import LockedDetector


def frame(value=0):
    return SimpleNamespace(rgb=np.full((8, 8, 3), value, np.uint8))


def test_reservation_is_reentrant_and_releases_after_failure():
    detected = threading.Event()
    d = LockedDetector(SimpleNamespace(detect=lambda *a, **kw: detected.set()))
    worker = threading.Thread(target=d.detect, args=(frame(),))
    with pytest.raises(RuntimeError, match="capture failed"):
        with d.reserve():
            d.prepare(["orange"])
            d.detect(frame())  # Same thread must not deadlock on its own lease.
            assert detected.is_set()
            detected.clear()
            worker.start()
            assert not detected.wait(.02)
            raise RuntimeError("capture failed")
    worker.join(2.)
    assert not worker.is_alive() and detected.is_set()


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
