"""Actual bridge class, CPU codecs: component reads must preserve one capture."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import threading
from types import SimpleNamespace
import zlib

import cv2
import numpy as np
import pytest

from conftest import load_isaac_bridge_definitions


@pytest.fixture
def frame_codec():
    calls = []

    def jpeg(*args, **kwargs):
        calls.append("jpeg")
        return cv2.imencode(*args, **kwargs)

    def depth(*args, **kwargs):
        calls.append("depth")
        return zlib.compress(*args, **kwargs)

    codecs = SimpleNamespace(
        cv2=SimpleNamespace(cvtColor=cv2.cvtColor, COLOR_RGB2BGR=cv2.COLOR_RGB2BGR,
                            IMWRITE_JPEG_QUALITY=cv2.IMWRITE_JPEG_QUALITY, imencode=jpeg),
        zlib=SimpleNamespace(compress=depth))
    env = dict(cv2=codecs.cv2, zlib=codecs.zlib, np=np, threading=threading, base64=base64)
    load_isaac_bridge_definitions({"_LazyFrame"}, env)
    return SimpleNamespace(cls=env["_LazyFrame"], calls=calls, codecs=codecs)


def capture(depth=True):
    rgb = np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3)
    values = np.linspace(.2, 2., 18 * 24, dtype=np.float32).reshape(18, 24) if depth else None
    if values is not None:
        values.flat[:3] = [np.inf, -np.inf, np.nan]
    entry = {"t": 123.5, "width": 24, "height": 18, "K": np.eye(3).tolist(),
             "proprioception": {"q": [.1, .2], "t": 123.5, "producer_epoch": "old-capture"},
             "render_reference": {"history_physics_step": 120}}
    return entry, rgb, values


def original_wire(entry, rgb, depth):
    """Original wire codec contract, independent of lazy/cache implementation."""
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    depth_b64 = None
    if depth is not None:
        depth = depth.copy()
        depth[~np.isfinite(depth)] = 0.
        depth_b64 = base64.b64encode(zlib.compress(depth.tobytes(), 3)).decode()
    return {**entry, "rgb_jpeg_b64": base64.b64encode(jpeg.tobytes()).decode(),
            "depth_z_b64": depth_b64}


def test_rgb_only_read_does_not_encode_depth(frame_codec):
    entry, rgb, depth = capture()
    frozen = copy.deepcopy(entry)
    frame = frame_codec.cls(entry, rgb, depth)
    first = frame["rgb_jpeg_b64"]
    assert frame_codec.calls == ["jpeg"]
    assert frame.get("rgb_jpeg_b64") == first
    assert frame["rgb_jpeg_b64"] == first
    assert frame_codec.calls == ["jpeg"]
    assert {key: frame[key] for key in entry} == frozen
    assert np.isnan(depth.flat[2]), "RGB access touched depth sanitization"


@pytest.mark.parametrize("has_depth", [True, False])
def test_depth_only_read_does_not_encode_rgb(frame_codec, has_depth):
    entry, rgb, depth = capture(has_depth)
    expected = original_wire(entry, rgb, depth)
    frame = frame_codec.cls(entry, rgb, depth)
    assert frame.get("depth_z_b64") == expected["depth_z_b64"]
    assert frame["depth_z_b64"] == expected["depth_z_b64"]
    assert frame_codec.calls == (["depth"] if has_depth else [])


@pytest.mark.parametrize("first", [None, "rgb_jpeg_b64", "depth_z_b64"])
@pytest.mark.parametrize("has_depth", [True, False])
def test_complete_wire_is_bitwise_original_for_any_access_order(frame_codec, first, has_depth):
    entry, rgb, depth = capture(has_depth)
    expected = original_wire(entry, rgb, depth)
    frozen = copy.deepcopy(entry)
    frame = frame_codec.cls(entry, rgb, depth)
    if first:
        assert frame[first] == expected[first]
    actual = frame.wire()
    assert json.dumps(actual) == json.dumps(expected)
    assert frame.wire() == expected
    assert frame_codec.calls.count("jpeg") == 1
    assert frame_codec.calls.count("depth") == int(has_depth)
    assert {key: actual[key] for key in entry} == frozen
    if has_depth:
        decoded = np.frombuffer(zlib.decompress(base64.b64decode(actual["depth_z_b64"])), dtype=np.float32)
        assert np.isfinite(decoded).all()
        np.testing.assert_array_equal(decoded[:3], 0.)


def test_metadata_and_unknown_keys_do_not_start_encoders(frame_codec):
    entry, rgb, depth = capture()
    frame = frame_codec.cls(entry, rgb, depth)
    assert frame.get("t") == entry["t"]
    assert frame.get("missing", "default") == "default"
    with pytest.raises(KeyError, match="missing"):
        frame["missing"]
    assert frame_codec.calls == []


@pytest.mark.parametrize("rgb_cached", [False, True])
def test_rgb_does_not_wait_for_another_threads_depth_encoder(frame_codec, rgb_cached):
    entry, rgb, depth = capture()
    frame = frame_codec.cls(entry, rgb, depth)
    expected = original_wire(entry, rgb, depth)
    if rgb_cached:
        assert frame["rgb_jpeg_b64"] == expected["rgb_jpeg_b64"]
    entered, release, rgb_done = threading.Event(), threading.Event(), threading.Event()
    real_compress = frame_codec.codecs.zlib.compress

    def blocked_depth(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release the blocked depth encoder"
        return real_compress(*args, **kwargs)

    frame_codec.codecs.zlib.compress = blocked_depth
    def read_rgb():
        value = frame.get("rgb_jpeg_b64")
        rgb_done.set()
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_depth = pool.submit(frame.get, "depth_z_b64")
        try:
            assert entered.wait(2)
            pending_rgb = pool.submit(read_rgb)
            assert rgb_done.wait(2), "RGB reader waited for unrelated depth work"
            assert pending_rgb.result() == expected["rgb_jpeg_b64"]
        finally:
            release.set()
        assert pending_depth.result(timeout=2) == expected["depth_z_b64"]
    assert frame.wire() == expected
    assert frame_codec.calls.count("jpeg") == frame_codec.calls.count("depth") == 1


def test_concurrent_component_and_wire_readers_encode_each_once(frame_codec):
    entry, rgb, depth = capture()
    expected = original_wire(entry, rgb, depth)
    frame = frame_codec.cls(entry, rgb, depth)
    barrier = threading.Barrier(9)
    def read(index):
        barrier.wait(timeout=3)
        if index % 3 == 0:
            return frame.wire()
        key = "rgb_jpeg_b64" if index % 3 == 1 else "depth_z_b64"
        assert frame.get(key) == expected[key]
        return None
    with ThreadPoolExecutor(max_workers=9) as pool:
        replies = list(pool.map(read, range(9)))
    assert all(reply is None or reply == expected for reply in replies)
    assert frame_codec.calls.count("jpeg") == frame_codec.calls.count("depth") == 1
    assert json.dumps(frame.wire()) == json.dumps(expected)


@pytest.mark.parametrize("raises", [False, True])
def test_failed_jpeg_keeps_its_buffer_and_retries_without_touching_depth(frame_codec, raises):
    entry, rgb, depth = capture()
    expected = original_wire(entry, rgb, depth)
    frame = frame_codec.cls(entry, rgb, depth)
    sentinel = RuntimeError("native jpeg error")
    real_encode = frame_codec.codecs.cv2.imencode
    attempts = []
    def flaky(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            if raises:
                raise sentinel
            return False, None
        return real_encode(*args, **kwargs)
    frame_codec.codecs.cv2.imencode = flaky
    with pytest.raises(RuntimeError) as error:
        frame["rgb_jpeg_b64"]
    if raises:
        assert error.value is sentinel
    else:
        assert str(error.value) == "JPEG encode failed for a captured camera frame"
    assert frame_codec.calls == []
    assert frame["rgb_jpeg_b64"] == expected["rgb_jpeg_b64"]
    assert frame_codec.calls == ["jpeg"]
    assert frame.wire() == expected
    assert len(attempts) == 2


def test_failed_depth_retries_without_reencoding_successful_rgb(frame_codec):
    entry, rgb, depth = capture()
    expected = original_wire(entry, rgb, depth)
    frozen = copy.deepcopy(entry)
    frame = frame_codec.cls(entry, rgb, depth)
    sentinel = RuntimeError("native depth error")
    real_compress = frame_codec.codecs.zlib.compress
    attempts = []
    def flaky(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise sentinel
        return real_compress(*args, **kwargs)
    frame_codec.codecs.zlib.compress = flaky
    with pytest.raises(RuntimeError) as error:
        frame.wire()
    assert error.value is sentinel
    assert frame.get("rgb_jpeg_b64") == expected["rgb_jpeg_b64"]
    assert len(attempts) == 1, "RGB access retried a failed depth encoding"
    assert frame.wire() == expected
    assert len(attempts) == 2 and frame_codec.calls.count("jpeg") == 1
    assert {key: frame[key] for key in entry} == frozen
