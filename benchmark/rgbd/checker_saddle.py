"""Explicit schema6 local saddle refinement; image samples only, no calibration."""

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import binary_reference as binary
from . import checker_accuracy as seed
from .planar_reference import digest, reference_from_corners


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=1)
def _numpy_sha256():
    root = Path(np.__file__).parent
    files = {}
    for directory in (root, root.parent / "numpy.libs"):
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and (
                    path.suffix == ".py"
                    or ".so" in path.name
                    or ".pyd" in path.name
                    or path.suffix in {".dll", ".dylib"}
                ):
                    files[str(path.relative_to(root.parent))] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    return hashlib.sha256(_canonical(files).encode()).hexdigest()


def consumer_recipe():
    return {
        "schema": 2,
        "seed": seed.consumer_recipe(),
        "refinement": {
            "implementation": "RGB local quadratic stationary saddle v1",
            "numpy_version": np.__version__,
            "numpy_package_sha256": _numpy_sha256(),
            "radius_px": 3,
            "gaussian_sigma_px": 1.5,
            "grayscale_rgb_weights": [0.299, 0.587, 0.114],
            "center": "floor(seed+.5), original integer pixel coordinates",
            "basis": ["1", "x", "y", "x*x", "x*y", "y*y"],
            "solver": "float64 weighted numpy.linalg.lstsq(rcond=None); one Hessian solve",
            "max_seed_offset_px": 1.5,
            "iterations": 1,
            "reject": "incomplete window, constant patch, rank<6, nonfinite, non-saddle or numerically singular Hessian, offset>1.5px",
            "fallback": None,
            "extrapolation": False,
            "output_contrast_check": "unchanged binary checker contrast gate",
        },
    }


def canonical_consumer_json(value):
    if not isinstance(value, str) or len(value.encode()) > 32768:
        raise ValueError("bounded canonical saddle consumer required")
    try:
        obj = json.loads(value)
        if _canonical(obj) != value or set(obj) != {"schema", "seed", "refinement"}:
            raise ValueError("exact canonical saddle consumer required")
        if type(obj["schema"]) is not int or obj["schema"] != 2:
            raise ValueError("saddle consumer schema differs")
        seed.canonical_consumer_json(_canonical(obj["seed"]))
        actual, expected = obj["refinement"], consumer_recipe()["refinement"]
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("saddle descriptor fields differ")
        for key in set(expected) - {"numpy_version", "numpy_package_sha256"}:
            if _canonical(actual[key]) != _canonical(expected[key]):
                raise ValueError("saddle policy differs")
        if not isinstance(actual["numpy_version"], str) or not re.fullmatch(
            r"\d+\.\d+\.\d+", actual["numpy_version"]
        ):
            raise ValueError("saddle NumPy version required")
        if not isinstance(actual["numpy_package_sha256"], str) or not re.fullmatch(
            "[0-9a-f]{64}", actual["numpy_package_sha256"]
        ):
            raise ValueError("saddle NumPy package digest required")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed saddle consumer") from exc
    return value


@dataclass(frozen=True)
class SaddleBoard:
    consumer_json: str = field(default_factory=lambda: _canonical(consumer_recipe()))

    def __post_init__(self):
        canonical_consumer_json(self.consumer_json)

    @property
    def seed_board(self):
        return seed.AccuracyBoard(_canonical(json.loads(self.consumer_json)["seed"]))

    @property
    def geometry(self):
        return self.seed_board.geometry

    columns, rows, square_m = 8, 6, 0.024
    origin_xyz_m = seed.AccuracyBoard.origin_xyz_m

    def corners_xy(self):
        return self.geometry.corners_xy()

    def texture_st(self):
        return self.geometry.texture_st()

    def description(self):
        value = self.seed_board.description()
        value.update(schema=6, kind="existing_ground_layout_a_local_saddle")
        value["checker"]["post_refinement"] = json.loads(self.consumer_json)[
            "refinement"
        ]
        return value

    @property
    def sha256(self):
        return digest(self.description())

    def require_consumer_implementation(self):
        if json.loads(self.consumer_json) != consumer_recipe():
            raise ValueError(
                "declared saddle consumer differs from active implementation"
            )


def refine_corners(rgb, corners):
    """Low-level RGB observation, not an admission verdict; reject every bad patch."""
    image, points = np.asarray(rgb), np.asarray(corners, dtype=float)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or not 0 < image.shape[0] * image.shape[1] <= 640 * 480
    ):
        raise ValueError("bounded RGB8 saddle image required")
    if points.shape != (35, 2) or not np.isfinite(points).all():
        raise ValueError("exactly35 finite saddle seeds required")
    gray = image.astype(float) @ np.asarray([0.299, 0.587, 0.114])
    xx, yy = np.meshgrid(np.arange(-3, 4, dtype=float), np.arange(-3, 4, dtype=float))
    x, y = xx.ravel(), yy.ravel()
    design = np.stack([np.ones(49), x, y, x * x, x * y, y * y], axis=1)
    weight = np.exp(-(x * x + y * y) / (2 * 1.5**2)) ** 0.5
    result = []
    for point in points:
        # Check bounds before converting untrusted coordinates to integers.
        center = np.floor(point + 0.5)
        if (
            (center < 3).any()
            or center[0] + 3 >= image.shape[1]
            or center[1] + 3 >= image.shape[0]
        ):
            raise ValueError("complete saddle window required; no border extrapolation")
        cx, cy = center.astype(int)
        patch = gray[cy - 3 : cy + 4, cx - 3 : cx + 4].ravel()
        if np.ptp(patch) == 0:
            raise ValueError("constant saddle patch")
        try:
            beta, _, rank, _ = np.linalg.lstsq(
                design * weight[:, None], patch * weight, rcond=None
            )
            hessian = np.asarray([[2 * beta[3], beta[4]], [beta[4], 2 * beta[5]]])
            if (
                rank != 6
                or not np.isfinite(beta).all()
                or np.linalg.det(hessian) >= 0
                or np.linalg.cond(hessian) >= 1 / np.finfo(float).eps
            ):
                raise ValueError("degenerate or non-saddle local patch")
            value = center - np.linalg.solve(hessian, beta[1:3])
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular saddle patch") from exc
        if not np.isfinite(value).all() or np.linalg.norm(value - point) > 1.5:
            raise ValueError("saddle center exceeds fixed seed neighborhood")
        result.append(value)
    return np.asarray(result)


def detect_saddle_corners(rgb, board):
    if type(board) is not SaddleBoard:
        raise ValueError("explicit schema6 SaddleBoard required")
    board.require_consumer_implementation()
    corners, stats = seed.detect_accuracy_corners(rgb, board.seed_board)
    refined = refine_corners(rgb, corners)
    cv2, _, _ = binary._api()
    binary._require_checker_contrast(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), refined.reshape(5, 7, 2)
    )
    return refined, stats


def saddle_reference_from_rgb(rgb, board):
    corners, _ = detect_saddle_corners(rgb, board)
    return reference_from_corners(
        corners, board, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()
    )
