"""Explicit ArUco orientation fixture; the metric fit remains checker-only."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .planar_reference import Board, reference_from_corners

IDS = (0, 1, 2, 3)
CELLS = 6  # DICT_4X4_50 payload plus one black border cell on every side.
MIN_CELL_PX = 2.0
FRAME_MARGIN_PX = 24.0
MIN_CHECKER_CONTRAST = 48.0  # RGB8 luminance levels, not a calibrated confidence.
MAX_DIAGONAL_SPREAD = 32.0


def _api():
    import cv2

    aruco = getattr(cv2, "aruco", None)
    if aruco is None or any(
        not hasattr(aruco, name)
        for name in (
            "ArucoDetector",
            "DetectorParameters",
            "generateImageMarker",
            "getPredefinedDictionary",
            "DICT_4X4_50",
        )
    ):
        raise ValueError("OpenCV ArUco API is required for this explicit fixture")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.errorCorrectionRate = 0.0
    params.maxErroneousBitsInBorderRate = 0.0
    params.markerBorderBits = 1
    params.detectInvertedMarker = False
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return cv2, dictionary, params


@lru_cache(maxsize=2)
def _opencv_package_sha256(root):
    # Bind the imported package's code bytes, including modules not exercised by
    # this frame. External before/after inventories also detect later disk drift.
    root = Path(root)
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and (p.suffix == ".py" or ".so" in p.name or ".pyd" in p.name)
    }
    if not files:
        raise ValueError("OpenCV code inventory unavailable")
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def detector_recipe():
    cv2, dictionary, params = _api()
    values = {
        name: getattr(params, name)
        for name in dir(params)
        if not name.startswith("_")
        and type(getattr(params, name)) in (int, float, bool)
    }

    return {
        "implementation": "OpenCV ArucoDetector",
        "opencv_version": cv2.__version__,
        "opencv_package_sha256": _opencv_package_sha256(str(Path(cv2.__file__).parent)),
        "dictionary": "DICT_4X4_50",
        "dictionary_bytes_sha256": hashlib.sha256(
            dictionary.bytesList.tobytes()
        ).hexdigest(),
        "parameters": values,
        "expected_ids": list(IDS),
        "border_cells": 1,
        "min_projected_cell_edge_px": MIN_CELL_PX,
        "quiet_zone_frame_margin_px": FRAME_MARGIN_PX,
        "checker_quadrant_fraction": 0.2,
        "min_checker_contrast_rgb8": MIN_CHECKER_CONTRAST,
        "max_checker_diagonal_spread_rgb8": MAX_DIAGONAL_SPREAD,
        "use_tags_for_metric_fit": False,
    }


def canonical_detector_json(value):
    """Bound immutable consumer declaration, not a producer detector claim."""
    if not isinstance(value, str) or len(value.encode("utf-8")) > 32768:
        raise ValueError("bounded canonical consumer detector JSON required")
    try:
        parsed = json.loads(value)
        canonical = json.dumps(
            parsed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if canonical != value or not isinstance(parsed, dict):
            raise ValueError("noncanonical consumer detector")
        template = detector_recipe()
        if set(parsed) != set(template):
            raise ValueError("unknown consumer detector fields")
        if not isinstance(parsed["opencv_version"], str) or not re.fullmatch(
            r"\d+\.\d+\.\d+", parsed["opencv_version"]
        ):
            raise ValueError("consumer OpenCV version required")
        for key in ("dictionary_bytes_sha256", "opencv_package_sha256"):
            if not isinstance(parsed[key], str) or not re.fullmatch(
                r"[0-9a-f]{64}", parsed[key]
            ):
                raise ValueError("consumer codebook/implementation digest required")
        # Descriptor policies are fixed for this explicit benchmark variant.
        for key in set(template) - {
            "opencv_version",
            "dictionary_bytes_sha256",
            "opencv_package_sha256",
            "parameters",
        }:
            if json.dumps(parsed[key], sort_keys=True) != json.dumps(
                template[key], sort_keys=True
            ):
                raise ValueError("consumer detector policy differs")
        parameters = parsed["parameters"]
        if not isinstance(parameters, dict) or set(parameters) != set(
            template["parameters"]
        ):
            raise ValueError("unknown consumer detector parameters")
        if any(
            type(v) is not type(template["parameters"][key]) or not np.isfinite(v)
            for key, v in parameters.items()
        ):
            raise ValueError("finite consumer detector parameters required")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid canonical consumer detector declaration") from exc
    return canonical


@dataclass(frozen=True)
class BinaryGroundBoard(Board):
    square_m: float = 0.024
    origin_xyz_m: tuple = (0.09, -0.13, 0.0)
    tag_size_m: float = 0.036
    consumer_detector_json: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if (self.columns, self.rows) != (8, 6) or self.origin_xyz_m[2] != 0.0:
            raise ValueError(
                "binary Ground variant requires the same 35 corners at world Z=0"
            )
        if type(self.tag_size_m) not in (float, int) or not np.isfinite(
            self.tag_size_m
        ):
            raise ValueError("finite tag size required")
        if not 0.02 <= self.tag_size_m <= 0.05 or self.tag_size_m > 1.5 * self.square_m:
            raise ValueError("bounded tag with declared checker clearance required")
        if self.consumer_detector_json is not None:
            canonical_detector_json(self.consumer_detector_json)

    def consumer_detector(self):
        return (
            detector_recipe()
            if self.consumer_detector_json is None
            else json.loads(self.consumer_detector_json)
        )

    def require_consumer_implementation(self):
        if json.dumps(self.consumer_detector(), sort_keys=True) != json.dumps(
            detector_recipe(), sort_keys=True
        ):
            raise ValueError(
                "declared consumer detector differs from the active implementation"
            )

    def marker_centers_xy(self):
        return np.asarray(
            [
                [
                    self.origin_xyz_m[0] + c * self.square_m,
                    self.origin_xyz_m[1] + r * self.square_m,
                ]
                for c, r in (
                    (-1.25, -1.25),
                    (self.columns + 1.25, -1.25),
                    (-1.25, self.rows + 1.25),
                    (self.columns + 1.25, self.rows + 1.25),
                )
            ]
        )

    def description(self):
        from .ground_texture import GroundTextureBoard

        description = GroundTextureBoard(
            self.columns, self.rows, self.square_m, self.origin_xyz_m
        ).description()
        for key in (
            "marker_center_outside_squares",
            "marker_halfwidth_squares",
            "marker_order",
        ):
            description.pop(key)
        description.update(
            schema=3,
            kind="existing_ground_binary_texture",
            tag_size_m=self.tag_size_m,
            tag_center_outside_squares=1.25,
            tag_ids=list(IDS),
            tag_top_row="positive_world_y",
            tag_quiet_zone_m=self.tag_size_m / CELLS,
            detector=self.consumer_detector(),
        )
        return description


def binary_rectangles(board):
    """Every checker/tag bit is an exact texel rectangle, not an image resize."""
    if type(board) is not BinaryGroundBoard:
        raise ValueError("explicit BinaryGroundBoard required")
    cv2, dictionary, _ = _api()
    if (
        hashlib.sha256(dictionary.bytesList.tobytes()).hexdigest()
        != board.consumer_detector()["dictionary_bytes_sha256"]
    ):
        raise ValueError("producer dictionary differs from the consumer codebook")
    x, y = (Fraction(str(v)) for v in board.origin_xyz_m[:2])
    s = Fraction(str(board.square_m))
    tag = Fraction(str(board.tag_size_m))
    cell = tag / CELLS
    rows = []

    def edge(v):
        p = (v + 1) * 2000
        if p.denominator != 1 or not 0 <= p <= 4000:
            raise ValueError(
                "binary reference requires exact in-plane 0.5 mm texel edges"
            )
        return int(p)

    def rect(x0, y0, w, h, value, label):
        u0, u1 = edge(x0), edge(x0 + w)
        v0, v1 = 4000 - edge(y0 + h), 4000 - edge(y0)
        if u0 >= u1 or v0 >= v1:
            raise ValueError("nonempty exact binary cell required")
        rows.append(
            {
                "label": label,
                "bounds_uv": [u0, v0, u1, v1],
                "rgb8": [value] * 3,
                "xy_bounds_m": list(map(float, (x0, y0, x0 + w, y0 + h))),
            }
        )

    outside = Fraction(5, 4) * s
    pad = outside + tag / 2 + cell  # at least one complete white cell beyond every tag.
    rect(
        x - pad,
        y - pad,
        board.columns * s + 2 * pad,
        board.rows * s + 2 * pad,
        255,
        "background",
    )
    for r in range(board.rows):
        for c in range(board.columns):
            rect(
                x + c * s, y + r * s, s, s, 255 if (r + c) % 2 else 0, f"square_{r}_{c}"
            )
    centers = (
        (x - outside, y - outside),
        (x + board.columns * s + outside, y - outside),
        (x - outside, y + board.rows * s + outside),
        (x + board.columns * s + outside, y + board.rows * s + outside),
    )
    for marker, (mx, my) in zip(IDS, centers, strict=True):
        edge(mx)
        edge(my)
        bits = cv2.aruco.generateImageMarker(dictionary, marker, CELLS, borderBits=1)
        for r in range(CELLS):
            for c in range(CELLS):
                rect(
                    mx - tag / 2 + c * cell,
                    my + tag / 2 - (r + 1) * cell,
                    cell,
                    cell,
                    int(bits[r, c]),
                    f"tag_{marker}_{r}_{c}",
                )
    return rows


def detected_tags(rgb):
    """Require the entire declared ID set; never choose a subset of components."""
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or not 0 < rgb.shape[0] * rgb.shape[1] <= 640 * 480
    ):
        raise ValueError("bounded uint8 RGB capture required")
    cv2, dictionary, params = _api()
    corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    )
    values = [] if ids is None else [int(i) for i in ids.ravel()]
    if len(values) != len(IDS) or sorted(values) != list(IDS):
        raise ValueError(
            "exactly one of each declared binary ID required; missing/duplicate/foreign IDs rejected"
        )
    ordered = []
    minimum_cells = []
    for marker in IDS:
        points = np.asarray(corners[values.index(marker)], dtype=float).reshape(4, 2)
        if (
            not np.isfinite(points).all()
            or (points < FRAME_MARGIN_PX).any()
            or (
                points >= np.asarray([rgb.shape[1], rgb.shape[0]]) - FRAME_MARGIN_PX
            ).any()
        ):
            raise ValueError("binary tag violates declared image margin")
        # Support check only: each projected bit/border edge and quiet zone.
        # These tag-derived transforms never enter the metric checker fit.
        canonical = np.array(
            [[0, 0], [CELLS, 0], [CELLS, CELLS], [0, CELLS]], np.float32
        )
        h = cv2.getPerspectiveTransform(canonical, points.astype(np.float32))
        grid = np.array(
            [[[x, y] for x in range(CELLS + 1)] for y in range(CELLS + 1)], np.float32
        )
        grid = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), h).reshape(
            CELLS + 1, CELLS + 1, 2
        )
        minimum = min(
            np.linalg.norm(np.diff(grid, axis=0), axis=2).min(),
            np.linalg.norm(np.diff(grid, axis=1), axis=2).min(),
        )
        if minimum < MIN_CELL_PX:
            raise ValueError("binary tag cells below declared pixel support")
        quiet = np.array(
            [[[-1, -1], [CELLS + 1, -1], [CELLS + 1, CELLS + 1], [-1, CELLS + 1]]],
            np.float32,
        )
        quiet = cv2.perspectiveTransform(quiet, h)[0]
        if (
            not np.isfinite(quiet).all()
            or (quiet < FRAME_MARGIN_PX).any()
            or (
                quiet >= np.asarray([rgb.shape[1], rgb.shape[0]]) - FRAME_MARGIN_PX
            ).any()
        ):
            raise ValueError("binary tag quiet zone violates declared image margin")
        minimum_cells.append(float(minimum))
        ordered.append(points)
    return np.asarray(ordered), {
        "ids": list(IDS),
        "rejected_quadrilaterals": len(rejected),
        "min_projected_cell_edge_px": min(minimum_cells),
    }


def detect_binary_corners(rgb, board):
    if type(board) is not BinaryGroundBoard:
        raise ValueError("explicit binary board required")
    tags, _ = detected_tags(rgb)
    cv2, _, _ = _api()
    found, corners = cv2.findChessboardCornersSB(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        (board.columns - 1, board.rows - 1),
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        raise ValueError("all checkerboard corners required")
    grid = corners.reshape(board.rows - 1, board.columns - 1, 2)
    # IDs orient the grid but never provide points to its metric homography.
    centers = tags.mean(axis=1)
    candidates = []
    for rows in (grid, grid[::-1]):
        for candidate in (rows, rows[:, ::-1]):
            extremes = candidate[[0, 0, -1, -1], [0, -1, 0, -1]]
            nearest = np.argmin(
                np.linalg.norm(centers[:, None, :] - extremes[None, :, :], axis=2),
                axis=1,
            )
            if nearest.tolist() == [0, 1, 2, 3]:
                candidates.append(candidate.reshape(-1, 2))
    if len(candidates) != 1:
        raise ValueError("binary checker orientation is ambiguous")
    result = np.array(candidates[0], dtype=float, copy=True)
    _require_checker_contrast(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        result.reshape(board.rows - 1, board.columns - 1, 2),
    )
    return result


def _require_checker_contrast(gray, grid):
    """Reject interpolated/occluded intersections lacking observed quadrants.

    OpenCV can return an inferred checker corner. Sample all four neighboring
    quadrants using local detected grid directions only, before metric fitting.
    This bounded visibility test is not proof against every possible occluder.
    """
    import cv2

    row_step, col_step = np.gradient(grid, axis=(0, 1))
    samples = []
    for row_sign, col_sign in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        uv = grid + 0.2 * (row_sign * row_step + col_sign * col_step)
        if (
            not np.isfinite(uv).all()
            or (uv < 1).any()
            or (uv >= np.array([gray.shape[1] - 1, gray.shape[0] - 1])).any()
        ):
            raise ValueError("checker quadrant sample outside image")
        samples.append(
            cv2.remap(
                gray,
                uv[:, :, 0].astype(np.float32),
                uv[:, :, 1].astype(np.float32),
                cv2.INTER_LINEAR,
            ).astype(float)
        )
    a, b, c, d = samples
    contrast = np.abs((a + c) / 2 - (b + d) / 2)
    if (
        (contrast < MIN_CHECKER_CONTRAST).any()
        or (np.abs(a - c) > MAX_DIAGONAL_SPREAD).any()
        or (np.abs(b - d) > MAX_DIAGONAL_SPREAD).any()
    ):
        raise ValueError("checker corner lacks observed quadrant contrast")


def binary_reference_from_rgb(rgb, board):
    board.require_consumer_implementation()
    return reference_from_corners(
        detect_binary_corners(rgb, board),
        board,
        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
    )
