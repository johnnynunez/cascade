"""Detector model load happens AFTER the camera rig is open (WRC 37285b0).

Seeed's WRC fork moved `_make_detector` behind `rig.open()` after bring-up on
their rig: loading YOLOE (CUDA init + weights) before the USB depth cameras
were streaming reset the cameras' USB link ("demo bring-up issues (USB
stability, ...)", commit 37285b0). The model load costs seconds either way;
doing it with the cameras already streaming is the order the rig was
verified with. A detector that fails to load must not leak the open rig.
"""

from __future__ import annotations

import pytest

from cascade.config import load_demo_config


def test_detector_is_built_after_the_camera_rig_opened(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    import cascade.apps.demo as demo
    from cascade.perception.stream import CameraRig

    order = []
    real_open, real_make = CameraRig.open, demo._make_detector

    def spy_open(rig):
        order.append("rig.open")
        return real_open(rig)

    def spy_make(cfg):
        order.append("detector")
        return real_make(cfg)

    monkeypatch.setattr(CameraRig, "open", spy_open)
    monkeypatch.setattr(demo, "_make_detector", spy_make)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run")
    try:
        assert order == ["rig.open", "detector"], order
        assert runtime.watcher is not None
    finally:
        demo.shutdown_runtime(runtime, arm)


def test_a_detector_that_fails_to_load_closes_the_open_rig(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    import cascade.apps.demo as demo
    from cascade.perception.stream import CameraRig

    closed = []
    real_close = CameraRig.close

    def spy_close(rig):
        closed.append(rig)
        return real_close(rig)

    def broken(cfg):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(CameraRig, "close", spy_close)
    monkeypatch.setattr(demo, "_make_detector", broken)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    with pytest.raises(RuntimeError, match="weights missing"):
        demo.build_runtime(cfg, tmp_path / "run")
    assert len(closed) == 1
