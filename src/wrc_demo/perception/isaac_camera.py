"""Isaac Sim camera backend: RGB-D frames from a live sim over the bridge.

The sim renders metric depth directly (no sensor noise, no min-range), so
frames behave exactly like a perfect RealSense as far as the stack cares.
"""

from __future__ import annotations

from ..config import Cfg
from ..sim.bridge_client import BridgeClient, BridgeError
from ..types import Frame
from .camera_base import CameraBase, CameraError


class IsaacCamera(CameraBase):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self._cfg = cfg
        self._camera = str(cfg.get("sim_camera", "cam0"))
        self._client = BridgeClient(
            host=str(cfg.get("bridge_host", "127.0.0.1")),
            port=int(cfg.get("bridge_port", 8611)),
        )
        self._has_depth = bool(cfg.get("depth", True))

    @property
    def has_depth(self) -> bool:
        return self._has_depth

    def open(self) -> None:
        try:
            self._client.connect()
            self._client.ping()
        except BridgeError as e:
            raise CameraError(str(e)) from e

    def close(self) -> None:
        self._client.close()

    def _grab(self) -> Frame:
        try:
            bgr, depth, K = self._client.frame(self._camera)
        except BridgeError as e:
            raise CameraError(str(e)) from e
        return Frame(
            rgb=bgr,
            depth_m=depth if self._has_depth else None,
            K=K,
            depth_source="sensor" if (depth is not None and self._has_depth) else "none",
        )
