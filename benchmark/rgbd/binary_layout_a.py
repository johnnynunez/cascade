"""Explicit rotated Ground layout A; its metric oracle receives RGB+geometry."""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

import numpy as np

from . import binary_reference as binary
from .homography_support import ALGORITHM, require_tag_support
from .planar_reference import Board, reference_from_corners

ORIGIN = (0.03150801262889796, 0.0705392619922785, 0.0)
CENTERS = ((-0.139, 0.0135), (0.2805, 0.011), (-0.116, 0.1545), (0.2595, 0.1525))
YAW_DEG = -44.0
TAG_SIZE_M = 0.084


@dataclass(frozen=True)
class BinaryLayoutABoard(Board):
    square_m: float = 0.024
    origin_xyz_m: tuple = ORIGIN
    frame_id: str = "world"
    yaw_deg: float = YAW_DEG
    tag_centers_local_m: tuple = CENTERS
    tag_size_m: float = TAG_SIZE_M
    consumer_detector_json: str | None = None

    def __post_init__(self):
        super().__post_init__()
        centers = tuple(tuple(row) for row in self.tag_centers_local_m)
        if (
            self.columns != 8
            or self.rows != 6
            or self.square_m != 0.024
            or self.origin_xyz_m != ORIGIN
            or self.frame_id != "world"
            or type(self.yaw_deg) not in (int, float)
            or self.yaw_deg != YAW_DEG
            or type(self.tag_size_m) not in (int, float)
            or self.tag_size_m != TAG_SIZE_M
            or centers != CENTERS
            or any(type(v) not in (int, float) for row in centers for v in row)
        ):
            raise ValueError("exact declared layout A geometry required")
        object.__setattr__(self, "tag_centers_local_m", centers)
        if self.consumer_detector_json is not None:
            binary.canonical_detector_json(self.consumer_detector_json)

    def rotation(self):
        angle = np.deg2rad(self.yaw_deg)
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

    def world_xy(self, local):
        return (
            np.asarray(local, dtype=float) @ self.rotation().T + self.origin_xyz_m[:2]
        )

    def corners_xy(self):
        local = [
            [c * self.square_m, r * self.square_m]
            for r in range(1, 6)
            for c in range(1, 8)
        ]
        return self.world_xy(local)

    def marker_centers_xy(self):
        return self.world_xy(self.tag_centers_local_m)

    def texture_st(self):
        from .ground_texture import ST

        # Explicit Ground vertex mapping, stored in the actual USD float2 type.
        world = np.asarray(ST) * 2 - 1
        local = (world - self.origin_xyz_m[:2]) @ self.rotation()
        return tuple(
            tuple(float(v) for v in row) for row in ((local + 1) / 2).astype(np.float32)
        )

    def consumer_detector(self):
        return (
            binary.detector_recipe()
            if self.consumer_detector_json is None
            else json.loads(self.consumer_detector_json)
        )

    def require_consumer_implementation(self):
        if self.consumer_detector() != binary.detector_recipe():
            raise ValueError(
                "declared consumer detector differs from active implementation"
            )

    def description(self):
        from .ground_texture import GroundTextureBoard

        result = GroundTextureBoard(8, 6, self.square_m, ORIGIN).description()
        for name in (
            "marker_center_outside_squares",
            "marker_halfwidth_squares",
            "marker_order",
        ):
            result.pop(name)
        result.update(
            schema=4,
            kind="existing_ground_binary_layout_a",
            frame_id=self.frame_id,
            axes="declared_board_local_xy_rotated_about_world_z",
            yaw_deg=self.yaw_deg,
            tag_centers_local_m=[list(v) for v in self.tag_centers_local_m],
            tag_size_m=self.tag_size_m,
            tag_ids=list(binary.IDS),
            tag_top_row="positive_board_local_y",
            tag_quiet_zone_m=self.tag_size_m / 6,
            detector=self.consumer_detector(),
            observed_resolution={
                "algorithm": ALGORITHM,
                "min_whole_tag_singular_px_cell": 4.0,
                "input": "detected_tag_corners_only",
                "tiles_per_axis": 8,
            },
        )
        result["texture"]["row_zero"] = "positive_board_local_y"
        result["texture"]["coordinates"] = "board_local_xy_in_minus1_plus1_m"
        result["ground"]["st"] = [list(v) for v in self.texture_st()]
        result["ground"]["st_storage"] = "USD_TexCoord2fArray_quantized_values"
        return result


def layout_rectangles(board):
    """Local metric texture cells; no rotated image or camera-dependent bitmap."""
    if type(board) is not BinaryLayoutABoard:
        raise ValueError("explicit BinaryLayoutABoard required")
    cv2, dictionary, _ = binary._api()
    if (
        hashlib.sha256(dictionary.bytesList.tobytes()).hexdigest()
        != board.consumer_detector()["dictionary_bytes_sha256"]
    ):
        raise ValueError("producer dictionary differs from consumer codebook")
    side = Fraction(str(board.tag_size_m))
    cell = side / 6
    square = Fraction(str(board.square_m))
    centers = [
        tuple(Fraction(str(v)) for v in row) for row in board.tag_centers_local_m
    ]
    rows = []

    def edge(value):
        pixel = (value + 1) * 2000
        if pixel.denominator != 1 or not 0 <= pixel <= 4000:
            raise ValueError("layout A requires exact local 0.5mm texel boundaries")
        return int(pixel)

    def rect(x, y, width, height, value, label):
        coordinates = [x, y, x + width, y + height]
        u0, u1 = edge(x), edge(x + width)
        v0, v1 = 4000 - edge(y + height), 4000 - edge(y)
        if u0 >= u1 or v0 >= v1:
            raise ValueError("nonempty local reference cell required")
        rows.append(
            {
                "label": label,
                "bounds_uv": [u0, v0, u1, v1],
                "rgb8": [value] * 3,
                "local_xy_bounds_m": list(map(float, coordinates)),
            }
        )

    left = min([Fraction(0)] + [x - side / 2 - cell for x, y in centers])
    bottom = min([Fraction(0)] + [y - side / 2 - cell for x, y in centers])
    right = max([8 * square] + [x + side / 2 + cell for x, y in centers])
    top = max([6 * square] + [y + side / 2 + cell for x, y in centers])
    rect(left, bottom, right - left, top - bottom, 255, "background")
    for r in range(6):
        for c in range(8):
            rect(
                c * square,
                r * square,
                square,
                square,
                255 if (r + c) % 2 else 0,
                f"square_{r}_{c}",
            )
    for marker, (x, y) in zip(binary.IDS, centers, strict=True):
        edge(x)
        edge(y)
        bits = cv2.aruco.generateImageMarker(dictionary, marker, 6, borderBits=1)
        for r in range(6):
            for c in range(6):
                rect(
                    x - side / 2 + c * cell,
                    y + side / 2 - (r + 1) * cell,
                    cell,
                    cell,
                    int(bits[r, c]),
                    f"tag_{marker}_{r}_{c}",
                )
    return rows


def detected_layout_tags(rgb, board):
    if type(board) is not BinaryLayoutABoard:
        raise ValueError("explicit layout A required")
    board.require_consumer_implementation()
    tags, stats = binary.detected_tags(rgb)
    stats["whole_tag_support"] = [require_tag_support(points) for points in tags]
    return tags, stats


def layout_reference_from_rgb(rgb, board):
    tags, _ = detected_layout_tags(rgb, board)
    cv2, _, _ = binary._api()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, (7, 5), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise ValueError("all checkerboard corners required")
    grid = corners.reshape(5, 7, 2)
    centers = tags.mean(axis=1)
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
    return reference_from_corners(
        candidates[0].reshape(-1, 2),
        board,
        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
    )
