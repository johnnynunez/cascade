"""A stand-in for ``pyorbbecsdk`` (v2 API surface) so the Orbbec backend is
tested on every machine without the SDK or a device.

Ported from WRC's ``tests/conftest.py`` fake (Seeed fork of cascade), with
two changes that make it a stricter double of the real module:

* ``OBFormat`` values carry the REAL numbering (YUYV=0, MJPG=5, Y16=8,
  RGB=22, BGR=23, ...) and are pybind11-style STRICT enums: they compare
  equal only to the same enum, never to a bare ``int`` (``OBFormat.MJPG ==
  5`` is False in pybind11 unless the enum is declared arithmetic). The WRC
  fake numbered formats differently from the constants the camera compared
  against, so the format branches were never exercised.
* Colour payloads are ENCODED per the active profile's format (MJPG really
  is JPEG bytes, YUYV really is 4:2:2), so the decode path is tested end to
  end, not just the reshape.

Usage: ``monkeypatch.setitem(sys.modules, "pyorbbecsdk", make_fake_sdk())``.
"""

from __future__ import annotations

import types

import cv2
import numpy as np


class StrictEnum:
    """pybind11-like enum member: ``int()``-able, has ``.name``, and is equal
    only to a member with the same value (never to a plain int)."""

    def __init__(self, name: str, value: int) -> None:
        self.name, self.value = name, int(value)

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __eq__(self, other) -> bool:
        return isinstance(other, StrictEnum) and other.value == self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"<OBFormat.{self.name}: {self.value}>"


def _enum_ns(**members: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(**{k: StrictEnum(k, v) for k, v in members.items()})


#: real pyorbbecsdk numbering (OBFormat in libobsensor ObTypes.h)
OBFormat = _enum_ns(YUYV=0, YUY2=1, UYVY=2, NV12=3, NV21=4, MJPG=5, H264=6, H265=7,
                    Y16=8, Y8=9, RGB=22, BGR=23, Y14=24, BGRA=25, COMPRESSED=26, RVL=27)
OBSensorType = _enum_ns(COLOR_SENSOR=2, DEPTH_SENSOR=3)
OBStreamType = _enum_ns(COLOR_STREAM=2, DEPTH_STREAM=3)
OBAlignMode = _enum_ns(DISABLE=0, HW_MODE=1, SW_MODE=2)


def reference_bgr(h: int, w: int) -> np.ndarray:
    """Deterministic smooth BGR gradient (JPEG/YUYV round-trip friendly):
    top rows are red, the middle green, the bottom blue."""
    yy = np.linspace(0.0, 1.0, h).reshape(h, 1) * np.ones((1, w))
    img = np.zeros((h, w, 3))
    img[:, :, 2] = 255.0 * (1.0 - yy)                      # R
    img[:, :, 1] = 255.0 * np.minimum(yy * 2, 1.0)         # G
    img[:, :, 0] = 255.0 * np.maximum(yy * 2 - 1.0, 0.0)   # B
    return np.clip(img, 0, 255).astype(np.uint8)


def bgr_to_yuyv(bgr: np.ndarray) -> bytes:
    """Pack a BGR image as YUYV 4:2:2 (Y0 U Y1 V per pixel pair), BT.601
    limited ("video") range -- the convention cv2.COLOR_YUV2BGR_YUYV and
    UVC cameras use."""
    b, g, r = (bgr[:, :, i].astype(np.float64) for i in range(3))
    y = 16.0 + 0.257 * r + 0.504 * g + 0.098 * b
    u = 128.0 - 0.148 * r - 0.291 * g + 0.439 * b
    v = 128.0 + 0.439 * r - 0.368 * g - 0.071 * b
    h, w = bgr.shape[:2]
    out = np.empty((h, w, 2), np.uint8)
    out[:, :, 0] = np.clip(y.round(), 0, 255).astype(np.uint8)
    out[:, 0::2, 1] = np.clip(((u[:, 0::2] + u[:, 1::2]) / 2.0).round(), 0, 255).astype(np.uint8)
    out[:, 1::2, 1] = np.clip(((v[:, 0::2] + v[:, 1::2]) / 2.0).round(), 0, 255).astype(np.uint8)
    return out.tobytes()


def encode_color(bgr: np.ndarray, fmt: StrictEnum) -> bytes:
    if fmt == OBFormat.BGR:
        return bgr.tobytes()
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()
    if fmt == OBFormat.MJPG:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        assert ok
        return buf.tobytes()
    if fmt in (OBFormat.YUYV, OBFormat.YUY2):
        return bgr_to_yuyv(bgr)
    raise AssertionError(f"fake SDK cannot encode {fmt!r}")


class Intrinsic:
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height


class Distortion:
    def __init__(self):
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = 0.1, -0.05, 0.01, 0.0, 0.0, 0.0
        self.p1, self.p2 = 0.001, -0.002


class VideoProfile:
    def __init__(self, w, h, fps, fmt, *, intrinsic=True):
        self._w, self._h, self._fps, self._fmt = w, h, fps, fmt
        self._intr = Intrinsic(600.0 * w / 1280, 600.0 * w / 1280, w / 2.0, h / 2.0, w, h) \
            if intrinsic else None

    def is_video_stream_profile(self):
        return True

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_fps(self):
        return self._fps

    def get_format(self):
        return self._fmt

    def get_intrinsic(self):
        if self._intr is None:
            raise RuntimeError("no intrinsic for this profile")
        return self._intr

    def get_distortion(self):
        return Distortion()

    def __repr__(self):
        return f"<VideoProfile {self._w}x{self._h}@{self._fps} {self._fmt.name}>"


class ProfileList:
    def __init__(self, profiles):
        self._p = list(profiles)

    def get_count(self):
        return len(self._p)

    def get_stream_profile_by_index(self, i):
        return self._p[i]


class ColorFrame:
    def __init__(self, bgr, fmt):
        self._h, self._w = bgr.shape[:2]
        self._fmt = fmt
        self._payload = encode_color(bgr, fmt)

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_format(self):
        return self._fmt

    def get_data(self):
        return self._payload


class DepthFrame:
    def __init__(self, raw: np.ndarray, scale: float | None):
        self._raw = raw.astype(np.uint16)
        self._scale = scale

    def get_width(self):
        return self._raw.shape[1]

    def get_height(self):
        return self._raw.shape[0]

    def get_data(self):
        return self._raw.tobytes()

    def get_format(self):
        return OBFormat.Y16


class FrameSet:
    def __init__(self, color, depth):
        self._c, self._d = color, depth

    def get_color_frame(self):
        return self._c

    def get_depth_frame(self):
        return self._d


class _AlignedFrame:
    """What the v2 SDK's ``AlignFilter.process`` returns: a generic Frame that
    must be converted with ``as_frame_set()``."""

    def __init__(self, fs):
        self._fs = fs

    def as_frame_set(self):
        return self._fs


class DeviceInfo:
    def __init__(self, name="Orbbec Gemini 2", serial="FAKE0001", pid=0x0670, vid=0x2BC5):
        self._name, self._serial, self._pid, self._vid = name, serial, pid, vid

    def get_name(self):
        return self._name

    def get_serial_number(self):
        return self._serial

    def get_pid(self):
        return self._pid

    def get_vid(self):
        return self._vid


class Device:
    def __init__(self, serial="FAKE0001"):
        self._info = DeviceInfo(serial=serial)

    def get_device_info(self):
        return self._info


class DeviceList:
    def __init__(self, devices):
        self._d = list(devices)

    def get_count(self):
        return len(self._d)

    def get_device_by_index(self, i):
        return self._d[i]


def make_fake_sdk(*, serials=("FAKE0001",), color_profiles=None, depth_profiles=None,
                  depth_mm=600, depth_scale=1.0, hw_align_breaks_start=False,
                  misaligned_depth=False, intrinsics=True, align_returns_frame=True):
    """Build a fresh fake ``pyorbbecsdk`` module.

    The returned module records what the camera did in ``sdk.log`` (pipelines
    created, configs started, streams enabled, align calls, stop calls).
    """
    sdk = types.SimpleNamespace()
    sdk.OBFormat, sdk.OBSensorType = OBFormat, OBSensorType
    sdk.OBStreamType, sdk.OBAlignMode = OBStreamType, OBAlignMode
    log = sdk.log = types.SimpleNamespace(pipelines=[], started=[], stopped=0, align_calls=0,
                                          frame_sync=0, align_targets=[])
    devices = [Device(s) for s in serials]
    sdk.devices = devices

    if color_profiles is None:
        color_profiles = [VideoProfile(1280, 720, 30, OBFormat.MJPG, intrinsic=intrinsics),
                          VideoProfile(640, 480, 30, OBFormat.RGB, intrinsic=intrinsics)]
    if depth_profiles is None:
        depth_profiles = [VideoProfile(1280, 800, 30, OBFormat.Y16),
                          VideoProfile(640, 400, 30, OBFormat.Y16)]

    class Context:
        def query_devices(self):
            return DeviceList(devices)

    class Config:
        def __init__(self):
            self.streams = []
            self.align_mode = None

        def enable_stream(self, profile):
            self.streams.append(profile)

        def set_align_mode(self, mode):
            self.align_mode = mode

    class Pipeline:
        def __init__(self, device=None):
            self.device = device if device is not None else (devices[0] if devices else None)
            self.requested = device
            self.cfg = None
            self.n = 0
            log.pipelines.append(self)

        def get_device(self):
            return self.device

        def get_stream_profile_list(self, sensor):
            if sensor == OBSensorType.COLOR_SENSOR:
                return ProfileList(color_profiles)
            if sensor == OBSensorType.DEPTH_SENSOR:
                return ProfileList(depth_profiles)
            raise RuntimeError(f"no sensor {sensor!r}")

        def enable_frame_sync(self):
            log.frame_sync += 1

        def start(self, cfg):
            if hw_align_breaks_start and cfg.align_mode == OBAlignMode.HW_MODE:
                raise RuntimeError("HW D2C not supported for this depth profile")
            self.cfg = cfg
            log.started.append(cfg)

        def stop(self):
            log.stopped += 1

        def _color_profile(self):
            return next(p for p in self.cfg.streams
                        if p._fmt not in (OBFormat.Y16,))

        def wait_for_frames(self, timeout_ms):
            self.n += 1
            cp = self._color_profile()
            bgr = reference_bgr(cp._h, cp._w)
            # depth comes at the DEPTH profile's resolution; only alignment
            # brings it to the colour image's size
            dp = next((p for p in self.cfg.streams if p._fmt == OBFormat.Y16), None)
            dh, dw = (dp._h, dp._w) if dp is not None else (cp._h, cp._w)
            raw = np.full((dh, dw), depth_mm, np.uint16)
            depth = DepthFrame(raw, depth_scale)
            if depth_scale is not None:
                depth.get_depth_scale = lambda: depth_scale
            return FrameSet(ColorFrame(bgr, cp._fmt), depth)

    class AlignFilter:
        def __init__(self, align_to_stream=None):
            log.align_targets.append(align_to_stream)

        def process(self, frames):
            log.align_calls += 1
            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if misaligned_depth:
                out = FrameSet(c, d)
            else:
                raw = np.frombuffer(d.get_data(), np.uint16).reshape(d.get_height(), d.get_width())
                aligned = cv2.resize(raw, (c.get_width(), c.get_height()),
                                     interpolation=cv2.INTER_NEAREST)
                nd = DepthFrame(aligned, d._scale)
                if hasattr(d, "get_depth_scale"):
                    nd.get_depth_scale = d.get_depth_scale
                out = FrameSet(c, nd)
            return _AlignedFrame(out) if align_returns_frame else out

    sdk.Context, sdk.Config, sdk.Pipeline, sdk.AlignFilter = Context, Config, Pipeline, AlignFilter
    return sdk
