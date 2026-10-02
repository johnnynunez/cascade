"""Passive mobile JPEG channel. No manipulation Frame, depth, map or Kit calls."""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import json
import socket
import threading
import time

from ..control.mobile_base import finite_real, identifier, nonnegative_int
from .base_truth import BaseTruthReader


FRAME_KEYS = frozenset({"robot_id", "source", "epoch", "engine", "device", "asset_sha256",
                        "policy_sha256", "camera", "step", "sim_time_s", "width", "height",
                        "rgb_jpeg_b64", "producer_age_s"})


def camera_profiles(profile):
    """Validate the opt-in mapping without opening sockets or importing Kit."""
    cameras = profile.get("cameras")
    if cameras is None:
        return {}
    if not isinstance(cameras, dict) or len(cameras) > 8:
        raise ValueError("cameras must be a mapping with at most 8 entries")
    if cameras and profile.get("type") != "isaac":
        raise ValueError("mobile cameras require an explicit Isaac profile, not a mock")
    for name, limits in cameras.items():
        identifier(name, "camera")
        if len(name) > 128 or not isinstance(limits, dict) or set(limits) != {"max_age_s", "max_jpeg_bytes", "max_pixels"}:
            raise ValueError("camera requires exact max_age_s, max_jpeg_bytes, max_pixels limits")
        if not 0 < finite_real(limits["max_age_s"], "max_age_s") <= 60:
            raise ValueError("camera max_age_s must be in (0,60]")
        for key, cap in (("max_jpeg_bytes", 4 * 1024 * 1024), ("max_pixels", 16 * 1024 * 1024)):
            if not 0 < nonnegative_int(limits[key], key) <= cap:
                raise ValueError(f"camera {key} outside resource bound")
    return copy.deepcopy(cameras)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON number")


class _FrameRPC:
    """One outer reader lock; smaller wire cap than the arm RGB-D transport."""
    def __init__(self, profile):
        self.address = (profile["bridge_host"], profile["bridge_port"])
        self.sock = None
        self.max_reply = 16384

    def connect(self, timeout_s):
        if self.sock is None:
            self.sock = socket.create_connection(self.address, timeout=timeout_s)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def request(self, payload, *, timeout_s):
        deadline = time.monotonic() + timeout_s
        def remaining():
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("camera RPC wall deadline")
            self.sock.settimeout(left)
        remaining()
        self.sock.sendall(json.dumps(payload).encode() + b"\n")
        chunks = bytearray()
        while b"\n" not in chunks:
            remaining()
            part = self.sock.recv(min(65536, self.max_reply + 1 - len(chunks)))
            if not part:
                raise OSError("camera RPC EOF")
            chunks.extend(part)
            if len(chunks) > self.max_reply:
                raise ValueError("camera reply exceeds wire size bound")
        line, extra = bytes(chunks).split(b"\n", 1)
        if extra:
            raise ValueError("unsolicited camera reply bytes")
        response = json.loads(line, object_pairs_hook=_json_object, parse_constant=_nonfinite)
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise ValueError("camera RPC requires boolean ok=true")
        remaining()
        return response


def _jpeg_dimensions(jpeg):
    """Read SOF before allocation; only 8-bit three-channel JPEG is supported."""
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("invalid/truncated JPEG")
    pos, dimensions = 2, None
    while pos + 4 <= len(jpeg):
        if jpeg[pos] != 255:
            break
        while pos < len(jpeg) and jpeg[pos] == 255:
            pos += 1
        if pos + 3 > len(jpeg):
            break
        marker = jpeg[pos]
        size = int.from_bytes(jpeg[pos + 1:pos + 3], "big")
        if size < 2 or pos + 1 + size > len(jpeg):
            break
        if marker in (0xC0, 0xC1, 0xC2):
            if dimensions is not None or size != 17 or jpeg[pos + 3] != 8 or jpeg[pos + 8] != 3:
                raise ValueError("invalid or duplicate JPEG SOF")
            dimensions = (int.from_bytes(jpeg[pos + 6:pos + 8], "big"), int.from_bytes(jpeg[pos + 4:pos + 6], "big"))
        elif marker in (0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise ValueError("unsupported JPEG SOF")
        if marker == 0xDA:
            if dimensions is not None:
                return dimensions
            break
        pos += 1 + size
    raise ValueError("JPEG lacks a supported SOF header")


@dataclass(frozen=True)
class MobileFrame:
    """Encoded, source-bound capture. Metadata is returned as a defensive copy."""
    _metadata: dict
    jpeg: bytes
    received_monotonic_s: float
    max_age_s: float

    def age_s(self):
        return self._metadata["producer_age_s"] + time.monotonic() - self.received_monotonic_s

    def as_dict(self):
        return {**copy.deepcopy(self._metadata), "received_monotonic_s": self.received_monotonic_s,
                "frame_age_s": self.age_s(), "age_clock": "producer_age_plus_local_RTT_and_elapsed"}


class MobileFrameReader:
    """Own reader socket; reuse the truth reader's exact identity/hello contract.

    No frame capability is invented: the v1 producer advertises state, with an
    optional frame callback. Epoch never rebinds on reconnect. One capture per
    configured camera is retained; repeated reads cannot rejuvenate its age.
    """

    def __init__(self, profile):
        self.cameras = camera_profiles(profile)
        if not self.cameras:
            raise ValueError("no mobile cameras configured")
        self._identity = BaseTruthReader(profile)  # validation only, no IO
        self._profile = copy.deepcopy(profile)
        self._timeout_s = profile["timeout_s"]
        self._client = _FrameRPC(profile)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._ready = False
        self.last_error = None
        self._frames, self._seen = {}, {}

    def _decode(self, response, camera, received, rtt):
        import cv2
        import numpy as np

        if set(response) != {"ok", "frame"} or not isinstance(response["frame"], dict) or set(response["frame"]) != FRAME_KEYS:
            raise ValueError("invalid mobile frame schema")
        meta = copy.deepcopy(response["frame"])
        for key in ("robot_id", "source", "engine", "device", "asset_sha256", "policy_sha256"):
            if meta[key] != self._profile[key]:
                raise ValueError(f"frame {key} mismatch")
        if meta["epoch"] != self._identity._epoch or meta["camera"] != camera:
            raise ValueError("frame epoch/camera mismatch")
        nonnegative_int(meta["step"], "step")
        for key in ("sim_time_s", "producer_age_s"):
            if finite_real(meta[key], key) < 0:
                raise ValueError(f"negative frame {key}")
        width, height = (nonnegative_int(meta[k], k) for k in ("width", "height"))
        limits = self.cameras[camera]
        if width <= 0 or height <= 0 or width * height > limits["max_pixels"]:
            raise ValueError("frame dimensions exceed pixel bound")
        encoded = meta.pop("rgb_jpeg_b64")
        if not isinstance(encoded, str) or len(encoded) > 4 * ((limits["max_jpeg_bytes"] + 2) // 3):
            raise ValueError("JPEG exceeds size bound")
        jpeg = base64.b64decode(encoded, validate=True)
        if len(jpeg) > limits["max_jpeg_bytes"] or _jpeg_dimensions(jpeg) != (width, height):
            raise ValueError("JPEG size/dimensions mismatch")
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.shape != (height, width, 3):
            raise ValueError("undecodable JPEG content/dimensions")
        meta["producer_age_s"] += rtt
        previous = self._seen.get(camera)
        if previous is not None:
            old = previous._metadata
            ds, dt = meta["step"] - old["step"], meta["sim_time_s"] - old["sim_time_s"]
            if received < previous.received_monotonic_s or ds < 0 or dt < 0 or ((ds == 0) != (dt == 0)):
                raise ValueError("frame clock regressed or inconsistent")
            if ds == 0:
                if jpeg != previous.jpeg or (width, height) != (old["width"], old["height"]):
                    raise ValueError("conflicting duplicate frame")
                meta["producer_age_s"] = max(meta["producer_age_s"],
                    old["producer_age_s"] + received - previous.received_monotonic_s)
        frame = MobileFrame(meta, jpeg, received, limits["max_age_s"])
        if not 0 <= frame.age_s() <= frame.max_age_s:
            raise ValueError("stale camera frame")
        return frame

    def __call__(self, camera):
        deadline = time.monotonic() + self._timeout_s
        if not self._lock.acquire(timeout=self._timeout_s):
            self.last_error = "camera reader lock timeout"
            return None
        try:
            if self._closed.is_set():
                raise ValueError("camera reader is closed")
            if camera not in self.cameras:
                raise ValueError("unknown camera")

            def remaining():
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("camera reader wall deadline")
                return budget

            if not self._ready:
                self._client.max_reply = 16384
                self._client.connect(remaining())
                self._identity._hello(self._client.request({"op": "hello", "role": "reader"}, timeout_s=remaining()))
                self._ready = True
            self._client.max_reply = 16384 + 4 * ((self.cameras[camera]["max_jpeg_bytes"] + 2) // 3)
            started = time.monotonic()
            response = self._client.request({"op": "frame", "camera": camera}, timeout_s=remaining())
            received = time.monotonic()
            frame = self._decode(response, camera, received, received - started)
            remaining()
            if self._closed.is_set():
                raise ValueError("camera reader closed during read")
            self._frames[camera] = self._seen[camera] = frame
            self.last_error = None
            return copy.deepcopy(frame)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:400]
            self._frames.clear()
            self._ready = False
            self._client.close()
            return None
        finally:
            self._lock.release()

    def close(self):
        self._closed.set()
        with self._lock:
            self._client.close()
            self._frames.clear()
            self._seen.clear()
            self._ready = False
        self._identity.close()

    def cached(self, camera):
        # Never wait for IO merely to display the last completed capture.
        if not self._lock.acquire(blocking=False):
            return None
        try:
            frame = self._frames.get(camera)
            if self._closed.is_set() or frame is None or not 0 <= frame.age_s() <= frame.max_age_s:
                return None
            return copy.deepcopy(frame)
        finally:
            self._lock.release()
