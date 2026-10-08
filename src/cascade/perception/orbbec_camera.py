"""Orbbec Gemini 2 backend (pyorbbecsdk v2), ported from WRC.

Source: Seeed Studio's cascade fork (WRC) `src/wrc_demo/perception/
orbbec_camera.py`, which drove the Gemini 2 on Seeed's B601 rig. It yields
the same `Frame` every other backend does -- BGR colour, float32 metric depth
aligned to colour (0 = invalid) and the colour intrinsics -- and nothing else
(no FK, no detection, no ArUco).

`pyorbbecsdk` is heavy and lives only on a rig venv, so it is imported inside
`open()` / `list_orbbec_devices()`: constructing the backend, and the whole
mock stack, never imports it (AGENTS.md "Adding a camera/arm backend").

What the WRC rig taught, kept here:

* Serial pinning. pyorbbecsdk v2 has NO `Pipeline.set_device_serial_number`;
  the device is selected by enumerating `Context().query_devices()` and
  constructing `Pipeline(device)`. The earlier code called the missing method
  inside `try/except: pass`, which bound the FIRST enumerated device: with two
  Gemini 2s on the bench "I asked for the top camera but the wrist view is
  showing". A requested serial that is absent, or a pipeline that opens a
  different device anyway, is a `CameraError` -- never a silent fallback.
* MJPG. Over the Gemini 2's USB link 1280x720 colour is offered as MJPG only;
  refusing MJPG silently dropped the stream to 640x480. MJPG payloads are
  JPEG bytes and must go through `cv2.imdecode` -- reshaping them produces
  garbage colours.
* RGB vs BGR. RGB payloads are channel-swapped to BGR (`Frame.rgb` is BGR).

Changes against the WRC code (each pinned by tests/test_orbbec_camera.py):

* Formats are compared by NAME/int value, not `fmt == 5`: pybind11 enums are
  strict and do not compare equal to plain ints, so WRC's int comparisons only
  matched its own test double.
* YUYV/YUY2 is decoded (`COLOR_YUV2BGR_YUYV`) instead of being blocked, as the
  lowest-preference uncompressed fallback.
* Depth scale is read per frame (`DepthFrame.get_depth_scale()`, millimetres
  per unit; 1.0 on a Gemini 2 at default precision) instead of from a device
  depth-sensor accessor the v2 SDK does not provide.
* Depth that is not aligned to colour (shape mismatch after alignment) is a
  `CameraError` rather than a frame whose pixels do not correspond.
* No synthesized intrinsics: a depth camera with made-up K misplaces every
  grasp, so missing intrinsics are an error unless the profile pins
  `intrinsics:` (3x3, row-major) explicitly.
* If the firmware refuses hardware depth-to-colour alignment at `start()`,
  the pipeline is restarted once with software alignment only (`AlignFilter`
  to the colour stream), mirroring the RealSense backend's start fallback.
* A failure after `start()` stops the pipeline before raising, so a partial
  open never leaks a running device.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError

logger = logging.getLogger(__name__)

_SDK_MISSING = ("pyorbbecsdk not installed; install the Orbbec SDK v2 Python "
                "bindings (`pyorbbecsdk2` / github.com/orbbec/pyorbbecsdk) into the rig "
                "venv, or use a different camera profile")

#: real OBFormat numbering (libobsensor ObTypes.h); used when the SDK hands
#: back a bare int, or an enum whose `.name` is missing
_FORMAT_NAMES = {0: "YUYV", 1: "YUY2", 2: "UYVY", 3: "NV12", 4: "NV21", 5: "MJPG",
                 6: "H264", 7: "H265", 8: "Y16", 9: "Y8", 22: "RGB", 23: "BGR",
                 24: "Y14", 25: "BGRA", 26: "COMPRESSED", 27: "RVL"}
#: colour formats this backend decodes, best first: uncompressed needs no
#: decode, MJPG is the only 1280x720 format on the Gemini 2's link, YUYV last
_COLOR_PREFERENCE = ("BGR", "RGB", "MJPG", "YUYV", "YUY2")


def _format_name(fmt) -> str | None:
    """`OBFormat.MJPG` / `5` / `"MJPG"` -> "MJPG" (None if unknown)."""
    if fmt is None:
        return None
    if isinstance(fmt, str):
        return fmt.upper()
    name = getattr(fmt, "name", None)
    if isinstance(name, str) and name:
        return name.upper().removeprefix("OB_FORMAT_")
    try:
        return _FORMAT_NAMES.get(int(fmt))
    except (TypeError, ValueError):
        return None


def _import_sdk():
    try:
        import pyorbbecsdk as ob  # type: ignore
    except ImportError as e:
        raise CameraError(_SDK_MISSING) from e
    if ob is None:  # pragma: no cover - sys.modules sentinel
        raise CameraError(_SDK_MISSING)
    return ob


def _devices(ob) -> list:
    devs = ob.Context().query_devices()
    if devs is None:
        return []
    n = devs.get_count() if hasattr(devs, "get_count") else len(devs)
    out = []
    for i in range(n):
        try:
            out.append(devs.get_device_by_index(i))
        except Exception:  # noqa: BLE001 - SDK raises generic errors
            try:
                out.append(devs[i])
            except Exception:  # noqa: BLE001
                continue
    return out


def list_orbbec_devices() -> list[dict[str, Any]]:
    """Enumerate Orbbec devices WITHOUT opening a pipeline (read-only).

    Returns one dict per device with `index`, and when readable `name`,
    `serial`, `pid`, `vid`. Raises `CameraError` if pyorbbecsdk is missing.
    """
    ob = _import_sdk()
    out: list[dict[str, Any]] = []
    for i, dev in enumerate(_devices(ob)):
        info: dict[str, Any] = {"index": i}
        try:
            di = dev.get_device_info()
            info.update(name=di.get_name(), serial=di.get_serial_number(),
                        pid=di.get_pid(), vid=di.get_vid())
        except Exception:  # noqa: BLE001
            pass
        out.append(info)
    return out


class OrbbecCamera(CameraBase):
    """`type: orbbec` -- an Orbbec RGB-D camera (Gemini 2) via pyorbbecsdk v2."""

    #: wait_for_frames timeout; a miss is a retryable CameraError
    WAIT_MS = 500

    def __init__(self, cfg: Cfg):
        super().__init__()
        self._cfg = cfg
        serial = cfg.get("serial") if cfg is not None else None
        self._serial = str(serial) if serial not in (None, "") else None
        self._pipeline: Any = None
        self._align: Any = None
        self._K: np.ndarray | None = None
        self._D: np.ndarray | None = None

    @property
    def has_depth(self) -> bool:
        return True

    @property
    def serial(self) -> str | None:
        return self._serial

    @property
    def K(self) -> np.ndarray | None:
        return self._K

    @property
    def D(self) -> np.ndarray | None:
        return self._D

    # ── lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> None:
        ob = _import_sdk()
        w = int(self._cfg.get("width", 1280))
        h = int(self._cfg.get("height", 720))
        fps = int(self._cfg.get("fps", 30))

        pipeline = self._bound_pipeline(ob)
        color = self._pick_video_profile(pipeline, ob.OBSensorType.COLOR_SENSOR, w, h, fps,
                                         prefer=_COLOR_PREFERENCE)
        if color is None:
            raise CameraError(f"Orbbec: no decodable colour stream profile near {w}x{h}@{fps}")
        depth = self._pick_video_profile(pipeline, ob.OBSensorType.DEPTH_SENSOR, w, h, fps,
                                         prefer=("Y16",))
        if depth is None:
            raise CameraError("Orbbec: no depth stream profile")

        config, hw_align = self._config(ob, color, depth, hw_align=True)
        try:
            pipeline.start(config)
        except Exception as e:  # noqa: BLE001
            if not hw_align:
                raise CameraError(f"Orbbec pipeline.start failed: {e}") from e
            # Firmware refused hardware D2C for this profile pair: restart
            # with software alignment only (the AlignFilter below).
            logger.warning("Orbbec HW depth-to-colour alignment refused (%s); "
                           "using software alignment", e)
            config, _ = self._config(ob, color, depth, hw_align=False)
            try:
                pipeline.start(config)
            except Exception as e2:  # noqa: BLE001
                raise CameraError(f"Orbbec pipeline.start failed: {e2}") from e2
        self._pipeline = pipeline
        try:
            self._K = self._intrinsics(color)
            self._D = self._distortion(color)
            self._align = ob.AlignFilter(ob.OBStreamType.COLOR_STREAM)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _config(ob, color, depth, *, hw_align: bool):
        """-> (Config, whether HW depth-to-colour alignment was requested).

        The flag is returned, not stored on the Config: pybind11 SDK objects
        reject new attributes."""
        config = ob.Config()
        config.enable_stream(color)
        config.enable_stream(depth)
        if not hw_align:
            return config, False
        try:
            config.set_align_mode(ob.OBAlignMode.HW_MODE)
        except Exception:  # noqa: BLE001 - firmware without HW D2C
            return config, False
        return config, True

    def _bound_pipeline(self, ob):
        """`Pipeline(device)` for the pinned serial; `Pipeline()` unpinned.

        v2 binds a pipeline to a device only through its constructor, so the
        serial is matched by enumeration, then re-read from the pipeline: an
        SDK path that ignores the device argument must not silently stream
        another camera (WRC: the wrist view showed up as the top camera).
        """
        if not self._serial:
            return ob.Pipeline()
        try:
            devices = _devices(ob)
        except Exception as e:  # noqa: BLE001
            raise CameraError(f"Orbbec device enumeration failed while looking for "
                              f"serial={self._serial!r}: {e!r}") from e
        seen, device = [], None
        for d in devices:
            try:
                s = d.get_device_info().get_serial_number()
            except Exception:  # noqa: BLE001
                continue
            seen.append(s)
            if s == self._serial:
                device = d
                break
        if device is None:
            raise CameraError(
                f"Orbbec device with serial={self._serial!r} not found (attached: "
                f"{seen or 'none'}); fix `serial:` in the camera profile -- "
                "`python -c 'from cascade.perception.orbbec_camera import "
                "list_orbbec_devices as l; print(l())'` lists them")
        pipeline = ob.Pipeline(device)
        opened = None
        try:
            got = pipeline.get_device() if hasattr(pipeline, "get_device") else None
            opened = got.get_device_info().get_serial_number() if got is not None else None
        except Exception:  # noqa: BLE001 - unreadable is not a mismatch
            opened = None
        if opened is not None and opened != self._serial:
            raise CameraError(f"Orbbec serial mismatch: requested {self._serial!r} but the "
                              f"pipeline opened {opened!r}; the SDK ignored the device "
                              "selection")
        return pipeline

    def _intrinsics(self, color_profile) -> np.ndarray:
        """Colour intrinsics: the profile's `intrinsics:` (3x3) if pinned,
        else the SDK's for the ACTIVE colour profile. Never synthesized."""
        pinned = self._cfg.get("intrinsics")
        if pinned is not None:
            K = np.asarray(pinned, dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
                raise CameraError(f"Orbbec profile intrinsics are not a valid K: {pinned!r}")
            return K
        try:
            intr = color_profile.get_intrinsic()
            K = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
                         dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            raise CameraError(f"Orbbec colour intrinsics unavailable ({e}); pin "
                              "`intrinsics:` (3x3) in the camera profile") from e
        if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise CameraError(f"Orbbec SDK returned invalid colour intrinsics {K.tolist()}")
        return K

    @staticmethod
    def _distortion(color_profile) -> np.ndarray | None:
        """OpenCV-order [k1, k2, p1, p2, k3] when the SDK reports it."""
        try:
            d = color_profile.get_distortion()
            return np.array([d.k1, d.k2, d.p1, d.p2, d.k3], dtype=np.float64)
        except Exception:  # noqa: BLE001 - informational only
            return None

    @staticmethod
    def _pick_video_profile(pipeline, sensor_type, want_w: int, want_h: int, want_fps: int,
                            prefer=_COLOR_PREFERENCE):
        """Best stream profile for (w, h, fps) among the formats in `prefer`.

        Buckets are tried in order -- exact w/h/fps, same w/h, anything -- and
        inside a bucket the earlier format in `prefer` wins. Formats not in
        `prefer` (H.264/H.265, NV12, ...) are never chosen: this backend
        cannot decode them. Returns None when nothing usable is offered.
        """
        try:
            pl = pipeline.get_stream_profile_list(sensor_type)
        except Exception:  # noqa: BLE001
            return None
        n = pl.get_count() if hasattr(pl, "get_count") else len(pl)
        rank = {name: i for i, name in enumerate(prefer)}
        buckets: list[list] = [[], [], []]
        for i in range(n):
            try:
                p = pl.get_stream_profile_by_index(i)
            except Exception:  # noqa: BLE001
                continue
            if hasattr(p, "is_video_stream_profile") and not p.is_video_stream_profile():
                continue
            name = _format_name(p.get_format())
            if name not in rank:
                continue
            pw, ph, pf = p.get_width(), p.get_height(), p.get_fps()
            b = 0 if (pw, ph, pf) == (want_w, want_h, want_fps) else (
                1 if (pw, ph) == (want_w, want_h) else 2)
            buckets[b].append((rank[name], i, p))
        for bucket in buckets:
            if bucket:
                return min(bucket, key=lambda t: (t[0], t[1]))[2]
        return None

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:  # noqa: BLE001
                pass

    # ── frames ─────────────────────────────────────────────────────────────

    @classmethod
    def _decode_color_frame(cls, payload, w: int, h: int, fmt) -> np.ndarray:
        """Colour payload -> contiguous (h, w, 3) uint8 BGR."""
        name = _format_name(fmt)
        if name == "MJPG":
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise CameraError("cv2.imdecode returned None on MJPG payload "
                                  "(truncated or corrupt JPEG from the SDK)")
            if decoded.shape[:2] != (h, w):
                raise CameraError(f"MJPG decoded {decoded.shape[1]}x{decoded.shape[0]} "
                                  f"!= frame {w}x{h}")
            return decoded
        if name == "RGB":
            return cv2.cvtColor(cls._reshape(payload, h, w, 3, name), cv2.COLOR_RGB2BGR)
        if name == "BGR":
            return cls._reshape(payload, h, w, 3, name).copy()
        if name in ("YUYV", "YUY2"):
            # 4:2:2 packed Y0 U Y1 V; YUY2 is the same byte layout
            return cv2.cvtColor(cls._reshape(payload, h, w, 2, name), cv2.COLOR_YUV2BGR_YUYV)
        raise CameraError(f"Orbbec colour format {fmt!r} is not decodable")

    @staticmethod
    def _reshape(payload, h: int, w: int, channels: int, name: str) -> np.ndarray:
        raw = np.frombuffer(payload, dtype=np.uint8)
        if raw.size != h * w * channels:
            raise CameraError(f"Orbbec {name} payload size mismatch: {raw.size} bytes for "
                              f"{w}x{h}x{channels}")
        return raw.reshape(h, w, channels)

    def _grab(self) -> Frame:
        if self._pipeline is None:
            raise CameraError("Orbbec pipeline not started")
        try:
            frames = self._pipeline.wait_for_frames(self.WAIT_MS)
            if frames is None:
                raise CameraError("no Orbbec frameset within timeout")
            if self._align is not None:
                frames = self._align.process(frames)
                if frames is not None and hasattr(frames, "as_frame_set"):
                    frames = frames.as_frame_set()
            if frames is None:
                raise CameraError("Orbbec depth-to-colour alignment produced no frameset")
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if color is None or depth is None:
                raise CameraError("incomplete frameset from Orbbec")
        except CameraError:
            raise
        except Exception as e:  # noqa: BLE001 - SDK raises generic errors
            raise CameraError(str(e)) from e

        try:
            w, h = int(color.get_width()), int(color.get_height())
            bgr = self._decode_color_frame(color.get_data(), w, h, color.get_format())
        except CameraError:
            raise
        except Exception as e:  # noqa: BLE001
            raise CameraError(f"failed to decode Orbbec colour frame: {e}") from e
        try:
            dw, dh = int(depth.get_width()), int(depth.get_height())
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(dh, dw)
        except Exception as e:  # noqa: BLE001
            raise CameraError(f"failed to decode Orbbec depth frame: {e}") from e
        if raw.shape != bgr.shape[:2]:
            raise CameraError(f"Orbbec depth {dw}x{dh} is not aligned to colour {w}x{h}")
        depth_m = raw.astype(np.float32) * np.float32(self._depth_unit_m(depth))

        min_d = float(self._cfg.get("min_depth_m", 0.15))
        max_d = float(self._cfg.get("max_depth_m", 2.5))
        depth_m[(depth_m < min_d) | (depth_m > max_d)] = 0.0
        return Frame(rgb=bgr, depth_m=depth_m, K=self._K, depth_source="sensor")

    @staticmethod
    def _depth_unit_m(depth_frame) -> float:
        """Metres per raw depth unit. pyorbbecsdk v2 reports the scale per
        frame in MILLIMETRES per unit (1.0 at the Gemini 2's default
        precision, 0.1 at 0.1 mm); absent -> 1 mm/unit."""
        get = getattr(depth_frame, "get_depth_scale", None)
        if get is None:
            return 0.001
        try:
            scale_mm = float(get())
        except Exception as e:  # noqa: BLE001
            raise CameraError(f"Orbbec depth scale unreadable: {e}") from e
        if not np.isfinite(scale_mm) or scale_mm <= 0.0:
            raise CameraError(f"Orbbec depth scale {scale_mm!r} mm/unit is not plausible")
        return scale_mm * 0.001
