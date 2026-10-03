"""Explicit schema5 checker localization; RGB and declared geometry only.

The historical layout-A localizer remains unchanged. This variant enables the
upstream aliasing-accuracy flag and consumes its returned coordinates directly.
"""

from dataclasses import dataclass, field
import hashlib
import json
import re

import numpy as np

from . import binary_layout_a as layout
from . import binary_reference as binary
from .planar_reference import digest, reference_from_corners


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def consumer_recipe():
    aruco = binary.detector_recipe()
    return {
        "schema": 1,
        "aruco": aruco,
        "checker": {
            "implementation": "OpenCV findChessboardCornersSB",
            "opencv_version": aruco["opencv_version"],
            "opencv_package_sha256": aruco["opencv_package_sha256"],
            "pattern_size": [7, 5],
            "grayscale": "cv2.COLOR_RGB2GRAY",
            "flags": ["CALIB_CB_NORMALIZE_IMAGE", "CALIB_CB_ACCURACY"],
            "flags_value": 34,
            "returned_coordinates": "unmodified original-image OpenCV coordinates",
            "post_refinement": None,
        },
    }


def canonical_consumer_json(value):
    """Validate the declaration without pretending a different SDK ran it."""
    if not isinstance(value, str) or len(value.encode()) > 32768:
        raise ValueError("bounded canonical checker consumer JSON required")
    try:
        obj = json.loads(value)
        if _canonical(obj) != value or set(obj) != {"schema", "aruco", "checker"}:
            raise ValueError("exact canonical checker consumer required")
        if type(obj["schema"]) is not int or obj["schema"] != 1:
            raise ValueError("checker consumer schema differs")
        binary.canonical_detector_json(_canonical(obj["aruco"]))
        checker, expected = obj["checker"], consumer_recipe()["checker"]
        if not isinstance(checker, dict) or set(checker) != set(expected):
            raise ValueError("checker descriptor fields differ")
        for key in set(expected) - {"opencv_version", "opencv_package_sha256"}:
            if _canonical(checker[key]) != _canonical(expected[key]):
                raise ValueError("checker policy differs")
        if not isinstance(checker["opencv_version"], str) or not re.fullmatch(
            r"\d+\.\d+\.\d+", checker["opencv_version"]
        ):
            raise ValueError("checker version required")
        if not isinstance(checker["opencv_package_sha256"], str) or not re.fullmatch(
            "[0-9a-f]{64}", checker["opencv_package_sha256"]
        ):
            raise ValueError("checker package digest required")
        if any(
            checker[key] != obj["aruco"][key]
            for key in ("opencv_version", "opencv_package_sha256")
        ):
            raise ValueError("checker/ArUco must declare the same consumer package")
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("malformed checker consumer") from exc
    return value


@dataclass(frozen=True)
class AccuracyBoard:
    """Immutable explicit declaration; no change to the schema4 geometry."""

    consumer_json: str = field(default_factory=lambda: _canonical(consumer_recipe()))

    def __post_init__(self):
        canonical_consumer_json(self.consumer_json)

    @property
    def geometry(self):
        return layout.BinaryLayoutABoard(
            consumer_detector_json=_canonical(json.loads(self.consumer_json)["aruco"])
        )

    columns = 8
    rows = 6
    square_m = 0.024
    origin_xyz_m = layout.ORIGIN

    def corners_xy(self):
        return self.geometry.corners_xy()

    def texture_st(self):
        return self.geometry.texture_st()

    def description(self):
        value = self.geometry.description()
        value.update(
            schema=5,
            kind="existing_ground_layout_a_checker_accuracy",
            checker=json.loads(self.consumer_json)["checker"],
        )
        return value

    @property
    def sha256(self):
        return digest(self.description())

    def require_consumer_implementation(self):
        if json.loads(self.consumer_json) != consumer_recipe():
            raise ValueError(
                "declared checker consumer differs from active implementation"
            )


def orient_corners(gray, corners, tags):
    """ID orientation and observed contrast only, unchanged acceptance gates."""
    points = np.asarray(corners)
    if points.size != 70 or not np.isfinite(points).all():
        raise ValueError("exactly35 finite checker corners required")
    grid = points.reshape(5, 7, 2)
    centers = np.asarray(tags).mean(axis=1)
    candidates = []
    for rows in (grid, grid[::-1]):
        for candidate in (rows, rows[:, ::-1]):
            extremes = candidate[[0, 0, -1, -1], [0, -1, 0, -1]]
            nearest = np.linalg.norm(centers[:, None] - extremes[None], axis=2).argmin(
                axis=1
            )
            if nearest.tolist() == [0, 1, 2, 3]:
                candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError("binary checker orientation is ambiguous")
    binary._require_checker_contrast(gray, candidates[0])
    return np.array(candidates[0].reshape(-1, 2), dtype=float, copy=True)


def detect_accuracy_corners(rgb, board):
    """One SB call; this low-level observation is not a metrology verdict."""
    if type(board) is not AccuracyBoard:
        raise ValueError("explicit schema5 AccuracyBoard required")
    board.require_consumer_implementation()
    tags, stats = layout.detected_layout_tags(rgb, board.geometry)
    cv2, _, _ = binary._api()
    if int(cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY) != 34:
        raise ValueError("active checker flag constants differ")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (7, 5), flags=34)
    if not found:
        raise ValueError("all checkerboard corners required")
    return orient_corners(gray, corners, tags), stats


def accuracy_reference_from_rgb(rgb, board):
    corners, _ = detect_accuracy_corners(rgb, board)
    return reference_from_corners(
        corners, board, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()
    )
