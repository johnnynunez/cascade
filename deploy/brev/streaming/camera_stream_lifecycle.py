"""Attach optional video to the existing Isaac main thread, with no physics work."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


def camera_video_from_environment(environ: Mapping[str, str] | None = None):
    """Return no owner or native imports when the optional profile is disabled."""
    path = (os.environ if environ is None else environ).get("PAAI_CAMERA_VIDEO_CONFIG")
    if not path:
        return None
    config_path = Path(path)
    if not config_path.is_absolute():
        raise ValueError("PAAI_CAMERA_VIDEO_CONFIG must be an absolute path")
    with config_path.open("rb") as source:
        raw = source.read(8193)
    if len(raw) > 8192:
        raise ValueError("Camera video configuration exceeds 8192 bytes")
    config = json.loads(raw)
    fields = {"schema", "enabled", "tailnet_ipv4", "visitor_origin",
              "rtsp_ports", "http_port", "media_port"}
    if (not isinstance(config, dict) or set(config) - fields
            or type(config.get("schema")) is not int or config["schema"] != 1
            or type(config.get("enabled")) is not bool):
        raise ValueError("Camera video configuration requires schema 1 and an explicit boolean enabled")
    if not config["enabled"]:
        return None
    from camera_stream_profile import StreamProfile

    values = {name: config[name] for name in fields - {"schema", "enabled"} if name in config}
    if "rtsp_ports" in values:
        if not isinstance(values["rtsp_ports"], list):
            raise ValueError("Camera video RTSP ports must be a JSON array")
        values["rtsp_ports"] = tuple(values["rtsp_ports"])
    try:
        profile = StreamProfile(**values)
    except TypeError as exc:
        raise ValueError("Camera video configuration requires its Tailscale address and visitor origin") from exc
    return CameraVideoLifecycle(profile)


class CameraVideoLifecycle:
    """Keep a failed optional stream set stopped until an operator reconciles it."""

    def __init__(
        self, profile: Any, *,
        owner_factory: Callable | None = None,
        verify_isolation: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile = profile
        self._factory, self._guard, self._clock = owner_factory, verify_isolation, clock
        self._owner = None
        self._state = "pending"
        self._error = None
        self._cleanup_error = None
        self._checked_at = None

    @staticmethod
    def _on_main() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Camera video lifecycle requires the Isaac main thread")

    def _verify(self, ports) -> None:
        if self._guard is None:
            from network_guard import verify_isolation

            self._guard = verify_isolation
        self._guard(ports)
        self._checked_at = self._clock()

    def _fail(self, exc: Exception) -> None:
        self._state = "failed"
        self._error = f"{type(exc).__name__}: {exc}"[:500]
        if self._owner is not None:
            try:
                self._owner.stop()
            except Exception as cleanup:
                self._cleanup_error = f"{type(cleanup).__name__}: {cleanup}"[:500]

    def start(self, sensors: Mapping[str, Any]) -> dict:
        self._on_main()
        if self._state != "pending":
            raise RuntimeError("Camera video may only start once per bridge lifecycle")
        self._state = "starting"
        try:
            self._verify(self.profile.rtsp_ports)
            if self._factory is None:
                from camera_streams import RtspCameraStreams

                self._factory = RtspCameraStreams
            self._owner = self._factory(self.profile)
            self._owner.start(sensors, verify_rtsp_isolation=self._verify)
            self._state = "attached"
        except Exception as exc:
            self._fail(exc)
        return self.status()

    def poll(self) -> bool:
        """Recheck host protection every five seconds; failure disables video only."""
        self._on_main()
        if self._state != "attached" or self._clock() - self._checked_at < 5.0:
            return False
        try:
            self._verify(self.profile.rtsp_ports)
        except Exception as exc:
            self._fail(exc)
            return True
        return False

    def restart_camera(self, name: str) -> dict:
        """Explicit recovery for a previously attached writer, via a queued job."""
        self._on_main()
        from camera_stream_profile import CAMERAS

        if name not in {camera for camera, _ in CAMERAS}:
            raise ValueError("Unknown video camera")
        if self._state != "attached":
            raise RuntimeError("Failed or inactive video requires operator reconciliation before restart")
        try:
            self._verify(self.profile.rtsp_ports)
            self._owner.restart_camera(name)
        except Exception as exc:
            self._fail(exc)
        return self.status()

    def status(self) -> dict:
        result = self._owner.status() if self._owner is not None else {}
        result.update(enabled=True, state=self._state, live_verified=False,
                      isolation_checked_monotonic=self._checked_at)
        if self._error is not None:
            result["error"] = self._error
        if self._cleanup_error is not None:
            result["cleanup_error"] = self._cleanup_error
        return result

    def stop(self) -> None:
        self._on_main()
        try:
            if self._owner is not None:
                self._owner.stop()
        except BaseException:
            self._state = "failed"
            raise
        else:
            self._state = "stopped"
