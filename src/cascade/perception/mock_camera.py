"""Mock camera: replays recorded frames or serves a synthetic tabletop scene.

Two modes:
- `dataset`: replay .npz frame files recorded with `cascade-record` (loops).
- `synthetic` (default): render a flat table at a configurable depth with a
  colored box on it -- enough for the full pipeline (detection via
  MockDetector, depth sampling, grasp planning) to run end-to-end in tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError


def synthetic_tabletop(
    width: int = 640,
    height: int = 480,
    table_depth_m: float = 0.6,
    box_px: tuple[int, int, int, int] = (280, 200, 360, 260),
    box_height_m: float = 0.05,
    box_sides: bool = True,
) -> Frame:
    """A camera looking straight down at a table with one red box.

    `box_sides` renders a one-pixel depth ramp around the box footprint so the
    prop reads as a SOLID with height rather than an infinitely thin lid.
    A perfectly flat lid (the old behaviour) gives the segmenter a cloud whose
    z-extent is exactly 0, so `grounding` places the object's centre ON the top
    face: for a 5 cm box that is 2.5 cm high, the planner then aims the jaws at
    the upper corner and nudges the prop away instead of closing around it
    (measured in MuJoCo: perceived z = 0.045 vs true centre z = 0.025, and a
    2.29 cm shove). The ramp costs one pixel of footprint and makes the
    reported extent match the object.
    """
    rgb = np.full((height, width, 3), 190, dtype=np.uint8)  # light gray table
    x0, y0, x1, y1 = box_px
    rgb[y0:y1, x0:x1] = (40, 40, 200)  # red box (BGR)
    depth = np.full((height, width), table_depth_m, dtype=np.float32)
    depth[y0:y1, x0:x1] = table_depth_m - box_height_m
    if box_sides and box_height_m > 0 and x0 < width and x1 > 0 and y0 < height and y1 > 0:
        # One-pixel skirt at mid-height: a straight-down camera cannot see a
        # vertical face, so without this the depth image carries no evidence
        # that the prop has any thickness at all.
        #
        # Clamped to the frame, and skipped when the box lies wholly outside
        # it: small mocks (the 160x120 stream-test camera) keep the default
        # 640x480 box_px. The box fill above no-ops on that via empty slices,
        # but a skirt row/column is an INDEX, and an out-of-range index is an
        # IndexError on every grab rather than a silent no-op.
        mid = table_depth_m - box_height_m / 2.0
        sx0, sx1 = max(x0, 0), min(x1, width)
        sy0, sy1 = max(y0, 0), min(y1, height)
        top, bot = max(y0 - 1, 0), min(y1, height - 1)
        left, right = max(x0 - 1, 0), min(x1, width - 1)
        depth[top, sx0:sx1] = mid
        depth[bot, sx0:sx1] = mid
        depth[sy0:sy1, left] = mid
        depth[sy0:sy1, right] = mid
        rgb[top, sx0:sx1] = (40, 40, 200)
        rgb[bot, sx0:sx1] = (40, 40, 200)
        rgb[sy0:sy1, left] = (40, 40, 200)
        rgb[sy0:sy1, right] = (40, 40, 200)
    fx = 600.0
    K = np.array(
        [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64
    )
    return Frame(rgb=rgb, depth_m=depth, K=K, depth_source="sensor")


class MockCamera(CameraBase):
    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        self._cfg = cfg
        self._files: list[Path] = []
        self._idx = 0
        self._opened = False

    @property
    def has_depth(self) -> bool:
        return True

    def open(self) -> None:
        if self._cfg is not None:
            dataset = self._cfg.get("dataset")
            if dataset:
                self._files = sorted(Path(dataset).glob("frame_*.npz"))
                if not self._files:
                    raise CameraError(f"no frame_*.npz files in {dataset}")
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def _grab(self) -> Frame:
        if not self._opened:
            raise CameraError("camera not opened")
        if self._files:
            data = np.load(self._files[self._idx % len(self._files)])
            self._idx += 1
            depth = data["depth_m"].astype(np.float32) if "depth_m" in data else None
            return Frame(
                rgb=data["rgb"],
                depth_m=depth,
                K=data["K"].astype(np.float64),
                depth_source="sensor" if depth is not None else "none",
            )
        kw = {}
        if self._cfg is not None:
            kw = {
                "width": int(self._cfg.get("width", 640)),
                "height": int(self._cfg.get("height", 480)),
                "table_depth_m": float(self._cfg.get("table_depth_m", 0.6)),
            }
            # Prop size is part of the scene, and the scene has to suit the
            # robot: the default box is ~73 mm across, which a 55 mm jaw
            # (SO-101) correctly refuses, so a rehearsal on that arm would only
            # ever exercise the width-rejection path. See configs/cameras/
            # mock_small.yaml.
            box = self._cfg.get("box_px")
            if box is not None:
                kw["box_px"] = tuple(int(v) for v in box)
            if self._cfg.get("box_height_m") is not None:
                kw["box_height_m"] = float(self._cfg.get("box_height_m"))
        return synthetic_tabletop(**kw)
