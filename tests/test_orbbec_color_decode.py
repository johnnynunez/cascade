"""`OrbbecCamera._decode_color_frame` and stream-profile choice.

Ported from WRC (Seeed's cascade fork) `tests/test_orbbec_color_decode.py`:
every colour format the Gemini 2 streams must decode to BGR, because a wrong
one is silent colour corruption in front of the detector. Additions: YUYV
(WRC blocked it), pybind11-strict format enums (WRC compared formats with
plain ints, which only matched its own test double), and the profile picker.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest
from fake_orbbec_sdk import OBFormat, StrictEnum, VideoProfile, bgr_to_yuyv, reference_bgr

from cascade.perception.camera_base import CameraError
from cascade.perception.orbbec_camera import OrbbecCamera

decode = OrbbecCamera._decode_color_frame


def test_bgr_passes_through_byte_identical():
    ref = reference_bgr(60, 80)
    np.testing.assert_array_equal(decode(ref.tobytes(), 80, 60, OBFormat.BGR), ref)


def test_rgb_is_channel_swapped_to_bgr():
    ref = reference_bgr(60, 80)
    raw = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).tobytes()
    out = decode(raw, 80, 60, OBFormat.RGB)
    np.testing.assert_array_equal(out, ref)
    assert not np.array_equal(out, np.frombuffer(raw, np.uint8).reshape(60, 80, 3))


def test_mjpg_is_jpeg_decoded_not_reshaped():
    ref = reference_bgr(120, 160)
    ok, buf = cv2.imencode(".jpg", ref, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok and buf.size < 120 * 160 * 3
    out = decode(buf.tobytes(), 160, 120, OBFormat.MJPG)
    assert out.shape == (120, 160, 3)
    np.testing.assert_allclose(out, ref, atol=8)


def test_mjpg_that_is_not_a_jpeg_is_a_camera_error():
    with pytest.raises(CameraError, match="imdecode returned None"):
        decode(bytes(range(256)) * 50, 80, 60, OBFormat.MJPG)


def test_mjpg_of_the_wrong_size_is_a_camera_error():
    ok, buf = cv2.imencode(".jpg", reference_bgr(60, 80))
    with pytest.raises(CameraError, match="decoded"):
        decode(buf.tobytes(), 160, 120, OBFormat.MJPG)


def test_yuyv_is_decoded_to_bgr():
    ref = reference_bgr(60, 80)
    out = decode(bgr_to_yuyv(ref), 80, 60, OBFormat.YUYV)
    assert out.shape == (60, 80, 3) and out.dtype == np.uint8
    # 4:2:2 chroma subsampling on a vertical gradient is near-lossless
    np.testing.assert_allclose(out.astype(int), ref.astype(int), atol=6)


def test_yuyv_payload_of_the_wrong_size_is_a_camera_error():
    with pytest.raises(CameraError, match="size"):
        decode(b"\x00" * (60 * 80 * 2 - 4), 80, 60, OBFormat.YUYV)


@pytest.mark.parametrize("fmt", [OBFormat.MJPG, 5, "MJPG", StrictEnum("OB_FORMAT_MJPG", 5)])
def test_format_is_matched_by_name_or_value_never_by_int_equality(fmt):
    """pybind11 enums are strict: `OBFormat.MJPG == 5` is False. The decoder
    must still recognise every spelling the SDK can hand back."""
    assert (OBFormat.MJPG == 5) is False          # premise: the fake is strict
    ref = reference_bgr(60, 80)
    ok, buf = cv2.imencode(".jpg", ref, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    np.testing.assert_allclose(decode(buf.tobytes(), 80, 60, fmt), ref, atol=8)


def test_an_undecodable_format_is_a_camera_error_never_raw_bytes():
    payload = reference_bgr(60, 80).tobytes()
    with pytest.raises(CameraError, match="not decodable"):
        decode(payload, 80, 60, OBFormat.H264)
    with pytest.raises(CameraError, match="not decodable"):
        decode(payload, 80, 60, 999)


# ── stream-profile choice ──────────────────────────────────────────────────


class _Pipe:
    def __init__(self, profiles):
        self.profiles = profiles

    def get_stream_profile_list(self, sensor):
        from fake_orbbec_sdk import ProfileList

        return ProfileList(self.profiles)


def _pick(profiles, w=1280, h=720, fps=30):
    return OrbbecCamera._pick_video_profile(_Pipe(profiles), None, w, h, fps)


def test_exact_resolution_beats_a_better_format_at_another_resolution():
    """WRC: refusing MJPG dropped the stream to 640x480 -- resolution first."""
    mjpg = VideoProfile(1280, 720, 30, OBFormat.MJPG)
    p = _pick([VideoProfile(640, 480, 30, OBFormat.RGB), mjpg])
    assert p is mjpg


def test_within_a_resolution_uncompressed_beats_mjpg_beats_yuyv():
    yuyv = VideoProfile(1280, 720, 30, OBFormat.YUYV)
    mjpg = VideoProfile(1280, 720, 30, OBFormat.MJPG)
    rgb = VideoProfile(1280, 720, 30, OBFormat.RGB)
    assert _pick([yuyv, mjpg, rgb]) is rgb
    assert _pick([yuyv, mjpg]) is mjpg
    assert _pick([yuyv]) is yuyv


def test_codecs_this_backend_cannot_decode_are_never_chosen():
    h264 = VideoProfile(1280, 720, 30, OBFormat.H264)
    nv12 = VideoProfile(1280, 720, 30, OBFormat.NV12)
    assert _pick([h264, nv12]) is None
    rgb_small = VideoProfile(640, 480, 30, OBFormat.RGB)
    assert _pick([h264, rgb_small]) is rgb_small


def test_same_resolution_other_fps_beats_other_resolution():
    p15 = VideoProfile(1280, 720, 15, OBFormat.MJPG)
    assert _pick([VideoProfile(640, 480, 30, OBFormat.MJPG), p15]) is p15


def test_unknown_format_logs_nothing_and_is_skipped(caplog):
    with caplog.at_level(logging.WARNING):
        assert _pick([VideoProfile(1280, 720, 30, StrictEnum("WEIRD", 77))]) is None
