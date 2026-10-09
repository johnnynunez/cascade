"""Camera enumeration for --list / --bind (no pipeline is ever started).

Ported from WRC ``perception/orbbec_camera.list_orbbec_devices`` and
``calib_top_orbbec --list``, widened to RealSense: the rig has two identical
D455Fs and the serial is the only thing that tells them apart. Both SDKs
are imported lazily; a missing SDK is a note, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDevice:
    backend: str          # "realsense" | "orbbec"
    serial: str
    name: str = ""
    firmware: str = ""
    usb: str = ""


def list_realsense() -> list[CameraDevice]:
    """Connected RealSense units (raises ImportError without pyrealsense2)."""
    import pyrealsense2 as rs

    out = []
    for dev in rs.context().query_devices():
        def info(key, dev=dev):
            try:
                return str(dev.get_info(getattr(rs.camera_info, key)))
            except Exception:
                return ""
        serial = info("serial_number")
        if serial:
            out.append(CameraDevice("realsense", serial, info("name"),
                                    info("firmware_version"), info("usb_type_descriptor")))
    return out


def list_orbbec() -> list[CameraDevice]:
    """Connected Orbbec units (raises ImportError without pyorbbecsdk)."""
    import pyorbbecsdk as ob

    devices = ob.Context().query_devices()
    if devices is None:
        return []
    n = devices.get_count() if hasattr(devices, "get_count") else len(devices)
    out = []
    for i in range(n):
        try:
            di = devices.get_device_by_index(i).get_device_info()
            serial, name = str(di.get_serial_number()), str(di.get_name())
        except Exception:
            continue
        if serial:
            out.append(CameraDevice("orbbec", serial, name))
    return out


_LISTERS = (("realsense", "pyrealsense2", list_realsense),
            ("orbbec", "pyorbbecsdk", list_orbbec))


def list_devices(backends=None) -> tuple[list[CameraDevice], list[str]]:
    """(devices, notes). A backend whose SDK is missing or fails adds a note."""
    devices, notes = [], []
    for backend, sdk, lister in _LISTERS:
        if backends is not None and backend not in backends:
            continue
        try:
            devices.extend(lister())
        except ImportError:
            notes.append(f"{backend}: {sdk} not installed (cannot enumerate {backend} cameras)")
        except Exception as e:  # a wedged USB stack must not hide the other SDK
            notes.append(f"{backend}: enumeration failed ({type(e).__name__}: {e})")
    return devices, notes
