"""CameraBase integration for an explicitly selected static OVRTX USD scene.

Dynamic physics owners use OvrtxRenderer.render(SceneSnapshot) directly; a
static profile cannot acquire current joints from an unrelated robot.
"""
import time

import numpy as np

from ..sim.ovrtx_renderer import CameraSpec, OvrtxError, OvrtxRenderer, SceneSnapshot
from .camera_base import CameraBase, CameraError


class OvrtxCamera(CameraBase):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.as_dict()
        if c.get("scene_mode") != "static":
            raise CameraError("ovrtx camera profiles require explicit scene_mode: static; use snapshot API for physics")
        if c.get("static_scene") is not True:
            raise CameraError("ovrtx static profiles must declare static_scene: true for motion-verification abstention")
        if c.get("depth", True) is not True:
            raise CameraError("ovrtx camera requires its paired RGB and metric-depth outputs")
        self._name = c.get("name", "cam0")
        w, h = c.get("width", 640), c.get("height", 480)
        if c.get("cx", (w - 1) / 2) != (w - 1) / 2 or c.get("cy", (h - 1) / 2) != (h - 1) / 2:
            raise CameraError("ovrtx currently supports centered pinhole intrinsics only")
        extrinsics = c.get("extrinsics", {})
        if extrinsics.get("mode", "eye_to_hand") != "eye_to_hand" or "T" not in extrinsics:
            raise CameraError("Static ovrtx cameras require explicit eye_to_hand extrinsics.T")
        try:
            camera = CameraSpec(self._name, w, h, c["fx"], c.get("fy", c["fx"]),
                                np.asarray(extrinsics["T"]), c.get("near_m", .01), c.get("far_m", 20.))
            self._renderer = OvrtxRenderer(c["scene_usd"], [camera], device=c.get("device", 0),
                                          render_dt=1 / c.get("fps", 30),
                                          meters_per_unit=c.get("meters_per_unit"))
        except (KeyError, ValueError, TypeError, OSError, OvrtxError, ZeroDivisionError) as e:
            raise CameraError(f"Invalid ovrtx camera profile: {e}") from e
        self._sequence = 0

    @property
    def has_depth(self):
        return True

    def open(self):
        # Native creation occurs in the capture thread on first get_frame.
        pass

    def close(self):
        try:
            self._renderer.close()
        except OvrtxError as e:
            raise CameraError(str(e)) from e

    def _grab(self):
        self._sequence += 1
        stamp = time.monotonic()  # Conservative capture START, before SDK work.
        snapshot = SceneSnapshot(str(self._renderer.scene), self._renderer._epoch,
                                 self._sequence, self._sequence * self._renderer.render_dt, stamp)
        try:
            return self._renderer.render(snapshot)[self._name]
        except OvrtxError as e:
            raise CameraError(str(e)) from e

    def get_frame(self):
        # No CameraBase retries: a partial native transaction invalidates this
        # renderer, and replaying it cannot establish a fresh physical capture.
        return self._grab()
