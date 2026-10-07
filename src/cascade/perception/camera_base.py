"""Camera abstraction: every backend yields metric-depth Frames.

A backend only has to produce (bgr, depth_m aligned to color, K). Cameras
without depth return depth_m=None and the DepthProvider chain fills it in
(mono-depth plugin or table-plane geometry), keeping the rest of the stack
depth-source agnostic.
"""

from __future__ import annotations

import abc
import time

from ..config import Cfg
from ..types import Frame


class CameraError(RuntimeError):
    pass


class CameraBase(abc.ABC):
    """Lifecycle + frame source. Frames use float32 meters for depth."""

    #: consecutive failures before get_frame raises
    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self) -> None:
        self._failures = 0
        self._frame_id = 0

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def _grab(self) -> Frame:
        """Fetch one frame or raise CameraError."""

    @property
    @abc.abstractmethod
    def has_depth(self) -> bool: ...

    def get_frame(self) -> Frame:
        try:
            frame = self._grab()
        except CameraError:
            self._failures += 1
            if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
                raise
            time.sleep(0.02)
            return self.get_frame()
        self._failures = 0
        self._frame_id += 1
        frame.frame_id = self._frame_id
        return frame

    def warm_up(self, n: int = 10) -> None:
        for _ in range(n):
            self.get_frame()

    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def is_wrist_view(cfg) -> bool:
    """Does this camera profile look out of the robot's hand?

    Two explicit declarations count, nothing is inferred from a name:
    `role: wrist` (a view that follows the gripper but carries no hand-eye
    calibration, e.g. the rendered `mujoco_wrist`) or `extrinsics.mode:
    eye_in_hand` (a calibrated wrist camera such as `isaac_wrist` /
    `d455f_wrist`, whose extrinsics already say the camera is on the hand).

    The runtime uses this to record the stream's frames as WRIST keyframes
    next to the front ones for motion skills, which is what fills the
    outcome judge's two wrist slots (eval/progress_judge.py) with real wrist
    views instead of the front image repeated. Works on a `Cfg` or a plain
    mapping; anything else is not a wrist view.
    """
    if cfg is None or not hasattr(cfg, "get"):
        return False
    if str(cfg.get("role", "")).lower() == "wrist":
        return True
    extr = cfg.get("extrinsics")
    return bool(extr is not None and hasattr(extr, "get")
                and str(extr.get("mode", "")).lower() == "eye_in_hand")


def make_camera(cfg: Cfg) -> CameraBase:
    """Instantiate a camera backend from a camera profile config."""
    kind = cfg.type
    if kind == "realsense":
        from .realsense_camera import RealSenseCamera

        return RealSenseCamera(cfg)
    if kind == "uvc":
        from .opencv_camera import OpenCVCamera

        return OpenCVCamera(cfg)
    if kind == "mock":
        from .mock_camera import MockCamera

        return MockCamera(cfg)
    if kind == "isaac":
        from .isaac_camera import IsaacCamera

        return IsaacCamera(cfg)
    if kind == "mujoco":
        from .mujoco_camera import MujocoCamera

        return MujocoCamera(cfg)
    if kind == "ovrtx":
        from .ovrtx_camera import OvrtxCamera

        return OvrtxCamera(cfg)
    raise ValueError(f"unknown camera type {kind!r} (realsense|uvc|mock|isaac|mujoco|ovrtx)")
