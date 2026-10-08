"""Orbbec Gemini 2 camera backend (`type: orbbec`), against a fake SDK only.

Ported from WRC (Seeed's cascade fork) `tests/test_orbbec_camera.py` and
adapted to cascade's CameraBase contract (AGENTS.md "Adding a camera/arm
backend"): the SDK import stays lazy inside `open()`, `_grab` raises
`CameraError` for retryable faults and never sets `frame_id`, `Frame.rgb` is
BGR and `depth_m` is float32 metres aligned to colour with 0 = invalid.

Never touches a device: `pyorbbecsdk` is replaced in `sys.modules` by
`tests/fake_orbbec_sdk.py` (or by None to simulate it being absent).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from fake_orbbec_sdk import OBFormat, VideoProfile, make_fake_sdk, reference_bgr

from cascade.config import Cfg
from cascade.perception.camera_base import CameraError, make_camera


@pytest.fixture
def sdk(monkeypatch):
    fake = make_fake_sdk()
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", fake)
    return fake


def _use(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", fake)
    return fake


def _cam(**cfg):
    return make_camera(Cfg({"type": "orbbec", **cfg}))


# ── lazy import / missing SDK ──────────────────────────────────────────────


def test_constructing_the_backend_imports_no_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)   # import would raise
    cam = _cam(serial="ABC")
    assert cam.has_depth is True
    assert cam.serial == "ABC"


def test_open_without_the_sdk_is_a_camera_error_naming_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)
    with pytest.raises(CameraError, match="pyorbbecsdk"):
        _cam().open()


# ── open + grab: the Frame contract ────────────────────────────────────────


def test_grab_returns_bgr_metric_depth_aligned_to_colour(sdk):
    cam = _cam(width=1280, height=720, fps=30)
    cam.open()
    try:
        f = cam.get_frame()
    finally:
        cam.close()
    assert f.rgb.shape == (720, 1280, 3) and f.rgb.dtype == np.uint8
    # MJPG at 1280x720 decoded to BGR: the top rows are RED in BGR order
    top = f.rgb[:5].reshape(-1, 3).mean(axis=0)
    assert top[2] > 200 and top[0] < 40, top
    assert f.depth_m.shape == (720, 1280) and f.depth_m.dtype == np.float32
    np.testing.assert_allclose(f.depth_m, 0.6, atol=1e-6)      # 600 mm, 1 mm/unit
    assert f.depth_source == "sensor"
    assert f.K.shape == (3, 3) and f.K[0, 0] == pytest.approx(600.0)
    assert f.K[0, 2] == pytest.approx(640.0) and f.K[1, 2] == pytest.approx(360.0)
    assert f.frame_id == 1                     # set by get_frame, not by _grab
    assert sdk.log.align_calls >= 1


def test_grab_does_not_set_frame_id(sdk):
    cam = _cam()
    cam.open()
    try:
        assert cam._grab().frame_id == 0
    finally:
        cam.close()


# ── serial pinning (WRC HANDOFF issue 12: wrong camera bound silently) ─────


def test_pinned_serial_binds_the_pipeline_to_that_device(monkeypatch):
    sdk = _use(monkeypatch, make_fake_sdk(serials=("AY3794300W4", "AY3Z331006L")))
    cam = _cam(serial="AY3Z331006L")
    cam.open()
    try:
        requested = sdk.log.pipelines[-1].requested
        assert requested is not None
        assert requested.get_device_info().get_serial_number() == "AY3Z331006L"
    finally:
        cam.close()


def test_a_pinned_serial_that_is_not_attached_is_refused(monkeypatch):
    sdk = _use(monkeypatch, make_fake_sdk(serials=("AY3794300W4",)))
    with pytest.raises(CameraError, match=r"AY3Z331006L.*not found"):
        _cam(serial="AY3Z331006L").open()
    assert sdk.log.started == []           # never started any device


def test_a_pipeline_that_opens_another_device_is_refused(monkeypatch):
    """Some SDK paths ignore Pipeline(device) and bind the first device."""
    sdk = _use(monkeypatch, make_fake_sdk(serials=("AY3794300W4", "AY3Z331006L")))
    real = sdk.Pipeline

    class LyingPipeline(real):
        def __init__(self, device=None):
            super().__init__(device)
            self.device = sdk.devices[0]

    sdk.Pipeline = LyingPipeline
    with pytest.raises(CameraError, match="mismatch") as info:
        _cam(serial="AY3Z331006L").open()
    assert "AY3Z331006L" in str(info.value) and "AY3794300W4" in str(info.value)
    assert sdk.log.started == []


def test_unpinned_profile_uses_the_first_device(sdk):
    cam = _cam(serial="")
    assert cam.serial is None
    cam.open()
    cam.close()
    assert sdk.log.pipelines[-1].requested is None


def test_list_orbbec_devices_reads_serials_without_starting_anything(monkeypatch):
    from cascade.perception.orbbec_camera import list_orbbec_devices

    sdk = _use(monkeypatch, make_fake_sdk(serials=("AY3794300W4", "AY3Z331006L")))
    out = list_orbbec_devices()
    assert [d["serial"] for d in out] == ["AY3794300W4", "AY3Z331006L"]
    assert out[0]["name"] == "Orbbec Gemini 2" and out[1]["index"] == 1
    assert sdk.log.pipelines == [] and sdk.log.started == []
    _use(monkeypatch, make_fake_sdk(serials=()))
    assert list_orbbec_devices() == []


# ── depth: metric, ranged, aligned ─────────────────────────────────────────


def _one_frame(monkeypatch, fake, **cfg):
    _use(monkeypatch, fake)
    cam = _cam(**cfg)
    cam.open()
    try:
        return cam.get_frame()
    finally:
        cam.close()


def test_depth_scale_is_read_per_frame_in_millimetres_per_unit(monkeypatch):
    """At 0.1 mm precision a raw 6000 is 0.6 m, not 6 m."""
    f = _one_frame(monkeypatch, make_fake_sdk(depth_mm=6000, depth_scale=0.1))
    np.testing.assert_allclose(f.depth_m, 0.6, atol=1e-6)


def test_depth_without_a_per_frame_scale_is_one_millimetre_per_unit(monkeypatch):
    f = _one_frame(monkeypatch, make_fake_sdk(depth_mm=600, depth_scale=None))
    np.testing.assert_allclose(f.depth_m, 0.6, atol=1e-6)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_implausible_depth_scale_is_a_camera_error(monkeypatch, bad):
    _use(monkeypatch, make_fake_sdk(depth_scale=bad))
    cam = _cam()
    cam.open()
    try:
        with pytest.raises(CameraError, match="depth scale"):
            cam._grab()
    finally:
        cam.close()


def test_depth_outside_the_profile_range_is_invalid(monkeypatch):
    f = _one_frame(monkeypatch, make_fake_sdk(depth_mm=600), max_depth_m=0.5)
    assert not f.depth_m.any()
    f = _one_frame(monkeypatch, make_fake_sdk(depth_mm=600), min_depth_m=0.7)
    assert not f.depth_m.any()


def test_depth_not_aligned_to_colour_is_a_camera_error_not_a_frame(monkeypatch):
    _use(monkeypatch, make_fake_sdk(misaligned_depth=True))
    cam = _cam()
    cam.open()
    try:
        with pytest.raises(CameraError, match="not aligned"):
            cam._grab()
    finally:
        cam.close()


def test_an_align_filter_returning_a_frameset_directly_also_works(monkeypatch):
    f = _one_frame(monkeypatch, make_fake_sdk(align_returns_frame=False))
    assert f.depth_m.shape == f.rgb.shape[:2]


# ── start / intrinsics / teardown ──────────────────────────────────────────


def test_hw_alignment_refused_at_start_retries_with_software_alignment(monkeypatch):
    sdk = _use(monkeypatch, make_fake_sdk(hw_align_breaks_start=True))
    cam = _cam()
    cam.open()
    try:
        f = cam.get_frame()
    finally:
        cam.close()
    assert len(sdk.log.started) == 1
    assert sdk.log.started[0].align_mode != sdk.OBAlignMode.HW_MODE
    assert f.depth_m.shape == f.rgb.shape[:2] and sdk.log.align_calls >= 1


def test_profile_intrinsics_override_the_sdk(monkeypatch):
    K = [[700.0, 0.0, 641.0], [0.0, 701.0, 359.0], [0.0, 0.0, 1.0]]
    f = _one_frame(monkeypatch, make_fake_sdk(), intrinsics=K)
    np.testing.assert_array_equal(f.K, np.array(K))


def test_missing_intrinsics_are_an_error_never_a_synthesized_guess(monkeypatch):
    sdk = _use(monkeypatch, make_fake_sdk(intrinsics=False))
    with pytest.raises(CameraError, match="intrinsics"):
        _cam().open()
    assert sdk.log.stopped == 1            # started, then stopped: no leak
    K = [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]]
    f = _one_frame(monkeypatch, make_fake_sdk(intrinsics=False), intrinsics=K)
    assert f.K[0, 0] == 700.0


def test_distortion_is_exposed_when_the_sdk_reports_it(sdk):
    cam = _cam()
    cam.open()
    try:
        np.testing.assert_allclose(cam.D, [0.1, -0.05, 0.001, -0.002, 0.01])
    finally:
        cam.close()


def test_close_stops_the_pipeline_once_and_is_idempotent(sdk):
    cam = _cam()
    cam.open()
    cam.close()
    cam.close()
    assert sdk.log.stopped == 1
    with pytest.raises(CameraError, match="not started"):
        cam._grab()


def test_a_stalled_stream_is_retryable_then_fatal(monkeypatch, sdk):
    cam = _cam()
    cam.open()
    try:
        monkeypatch.setattr(sdk.log.pipelines[-1], "wait_for_frames", lambda ms: None)
        monkeypatch.setattr(cam, "MAX_CONSECUTIVE_FAILURES", 3)
        monkeypatch.setattr("cascade.perception.camera_base.time.sleep", lambda s: None)
        with pytest.raises(CameraError):
            cam.get_frame()
    finally:
        cam.close()


# ── shipped profiles ───────────────────────────────────────────────────────


def test_overhead_profile_is_an_unpinned_eye_to_hand_orbbec(monkeypatch):
    """Profiles copied from another rig pin THAT rig's units; the shipped one
    must not carry Seeed's serials. Its extrinsic is the Seeed rig's measured
    mount, labelled as a reference that needs recalibration."""
    from cascade.config import load_profile
    from cascade.perception.orbbec_camera import OrbbecCamera

    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)
    prof = load_profile("cameras", "orbbec_overhead")
    assert prof.type == "orbbec" and not prof.get("serial")
    assert (prof.width, prof.height, prof.fps) == (1280, 720, 30)
    ext = prof.extrinsics
    assert ext.mode == "eye_to_hand"
    T = np.asarray(ext.T, float)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)
    assert T[2, 3] > 0.3 and T[2, 2] < -0.5      # above the table, looking down
    assert isinstance(make_camera(prof), OrbbecCamera)


def test_wrist_profile_streams_but_never_fuses():
    from cascade.config import load_profile
    from cascade.perception.camera_base import is_wrist_view

    prof = load_profile("cameras", "orbbec_wrist")
    assert prof.type == "orbbec" and not prof.get("serial")
    assert is_wrist_view(prof)
    assert prof.get("fuse_beliefs") is False and "extrinsics" not in prof
