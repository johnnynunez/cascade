"""Image-only planar metric reference, independent of camera calibration/depth.

No K, camera transform, pixel-center offset or raycast is accepted by the fit.
Scene geometry is declared before capture. This is simulated planar metrology,
not external camera calibration or a source of physical command authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Board:
    columns: int = 8
    rows: int = 6
    square_m: float = 0.03
    origin_xyz_m: tuple = (0.08, 0.01, 0.002)

    def __post_init__(self):
        if (
            type(self.columns) is not int
            or type(self.rows) is not int
            or not 6 <= self.columns <= 12
            or not 6 <= self.rows <= 12
        ):
            raise ValueError(
                "bounded checkerboard with at least 25 internal corners required"
            )
        if (
            type(self.square_m) not in (float, int)
            or not np.isfinite(self.square_m)
            or not 0.005 <= self.square_m <= 0.1
        ):
            raise ValueError("metric square size required")
        origin = tuple(self.origin_xyz_m)
        if len(origin) != 3 or any(
            type(v) not in (float, int) or not np.isfinite(v) for v in origin
        ):
            raise ValueError("finite world origin required")
        object.__setattr__(self, "origin_xyz_m", origin)

    @property
    def sha256(self):
        return digest(self.description())

    def description(self):
        return {
            "schema": 1,
            **asdict(self),
            "axes": "world_x_y",
            "units": "m",
            "surface": "visual_only",
            "background_below_m": 0.0001,
            "marker_center_outside_squares": 0.8,
            "marker_halfwidth_squares": 0.25,
            "marker_order": ["red", "green", "blue", "yellow"],
        }

    def corners_xy(self):
        return np.asarray(
            [
                [
                    self.origin_xyz_m[0] + c * self.square_m,
                    self.origin_xyz_m[1] + r * self.square_m,
                ]
                for r in range(1, self.rows)
                for c in range(1, self.columns)
            ]
        )

    def marker_centers_xy(self):
        return np.asarray(
            [
                [
                    self.origin_xyz_m[0] + c * self.square_m,
                    self.origin_xyz_m[1] + r * self.square_m,
                ]
                for c, r in [
                    (-0.8, -0.8),
                    (self.columns + 0.8, -0.8),
                    (-0.8, self.rows + 0.8),
                    (self.columns + 0.8, self.rows + 0.8),
                ]
            ]
        )


def visual_quads(board=Board()):
    """Return exact world vertices/colors; no camera or physics properties."""
    x0, y0, z = board.origin_xyz_m
    s = board.square_m

    def quad(x, y, w, h, depth, color):
        return {
            "points_m": [
                [x, y, depth],
                [x + w, y, depth],
                [x + w, y + h, depth],
                [x, y + h, depth],
            ],
            "emissive_rgb": list(color),
        }

    result = [
        quad(
            x0 - 1.4 * s,
            y0 - 1.4 * s,
            (board.columns + 2.8) * s,
            (board.rows + 2.8) * s,
            z - 0.0001,
            (1.0, 1.0, 1.0),
        )
    ]
    for r in range(board.rows):
        for c in range(board.columns):
            color = (1.0, 1.0, 1.0) if (r + c) % 2 else (0.0, 0.0, 0.0)
            result.append(quad(x0 + c * s, y0 + r * s, s, s, z, color))
    for (x, y), color in zip(
        board.marker_centers_xy(),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
    ):
        result.append(quad(x - 0.25 * s, y - 0.25 * s, 0.5 * s, 0.5 * s, z, color))
    return result


def author_visual_board(stage, board=Board(), path="/World/RgbdMetricBoard"):
    """CPU USD authoring only; does not step/open Kit or apply collision schemas.

    The caller must bind the composed scene's new digest before any future run.
    Existing prims cannot be replaced and the meter-stage contract is explicit.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    if (
        not path.startswith("/World/")
        or stage.GetPrimAtPath(path)
        or UsdGeom.GetStageMetersPerUnit(stage) != 1.0
    ):
        raise ValueError("new world path in a meter stage required")
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    root.SetCustomDataByKey(
        "rgbdReference", board.description() | {"sha256": board.sha256}
    )
    for i, item in enumerate(visual_quads(board)):
        mesh = UsdGeom.Mesh.Define(stage, f"{path}/quad_{i:03}")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in item["points_m"]])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        material = UsdShade.Material.Define(stage, f"{path}/material_{i:03}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.0)
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*item["emissive_rgb"])
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return {
        "board_sha256": board.sha256,
        "geometry_sha256": digest(visual_quads(board)),
        "path": path,
        "physical_admission": False,
    }


def _points(value, count=None):
    a = np.asarray(value, dtype=float)
    if (
        a.ndim != 2
        or a.shape[1] != 2
        or not np.isfinite(a).all()
        or (count is not None and len(a) != count)
    ):
        raise ValueError("finite point pairs required")
    return a


def apply_homography(matrix, points):
    p = _points(points)
    h = np.asarray(matrix, dtype=float).reshape(3, 3)
    q = np.c_[p, np.ones(len(p))] @ h.T
    if not np.isfinite(q).all() or (np.abs(q[:, 2]) < 1e-12).any():
        raise ValueError("homography at infinity or invalid")
    return q[:, :2] / q[:, 2, None]


def fit_homography(source, target):
    """Normalized DLT using every declared fit point, never outlier selection."""
    a, b = _points(source), _points(target)
    if len(a) != len(b) or not 4 <= len(a) <= 121:
        raise ValueError("bounded paired homography points required")

    def normalize(p):
        mean = p.mean(axis=0)
        distance = np.linalg.norm(p - mean, axis=1).mean()
        if distance < 1e-12 or np.linalg.matrix_rank(p - mean) != 2:
            raise ValueError("degenerate reference geometry")
        scale = np.sqrt(2) / distance
        t = np.array(
            [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]]
        )
        return apply_homography(t, p), t

    an, ta = normalize(a)
    bn, tb = normalize(b)
    rows = []
    for (x, y), (u, v) in zip(an, bn):
        rows += [
            [-x, -y, -1, 0, 0, 0, u * x, u * y, u],
            [0, 0, 0, -x, -y, -1, v * x, v * y, v],
        ]
    _, singular, vh = np.linalg.svd(rows, full_matrices=True)
    if singular[7] < 1e-10:
        raise ValueError("rank deficient reference")
    h = np.linalg.inv(tb) @ vh[-1].reshape(3, 3) @ ta
    if abs(np.linalg.det(h)) < 1e-15 or abs(h[2, 2]) < 1e-12:
        raise ValueError("invalid planar reference")
    return h / h[2, 2]


def _marker_centers(rgb):
    import cv2

    r, g, b = (rgb[:, :, i].astype(float) for i in range(3))
    masks = [
        (r > 80) & (r > 1.8 * g) & (r > 1.8 * b),
        (g > 80) & (g > 1.8 * r) & (g > 1.8 * b),
        (b > 80) & (b > 1.8 * r) & (b > 1.8 * g),
        (r > 100) & (g > 100) & (b < 0.5 * np.minimum(r, g)),
    ]
    centers = []
    for mask in masks:
        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype("uint8"), 8
        )
        matches = [i for i in range(1, len(stats)) if stats[i, cv2.CC_STAT_AREA] >= 5]
        if len(matches) != 1:
            raise ValueError("exactly one unambiguous marker of each color required")
        centers.append(centroids[matches[0]])
    return np.asarray(centers)


def detect_corners(rgb, board=Board()):
    """Detect a bounded RGB array using colors only for orientation, not metric fit."""
    import cv2

    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or not 0 < rgb.shape[0] * rgb.shape[1] <= 640 * 480
    ):
        raise ValueError("bounded uint8 RGB capture required")
    markers = _marker_centers(rgb)
    found, corners = cv2.findChessboardCornersSB(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        (board.columns - 1, board.rows - 1),
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        raise ValueError("checkerboard corners unavailable")
    grid = corners.reshape(board.rows - 1, board.columns - 1, 2)
    oriented = []
    for rows in (grid, grid[::-1]):
        for candidate in (rows, rows[:, ::-1]):
            extremes = candidate[[0, 0, -1, -1], [0, -1, 0, -1]]
            nearest = np.argmin(
                np.linalg.norm(markers[:, None, :] - extremes[None, :, :], axis=2),
                axis=1,
            )
            if list(nearest) == [0, 1, 2, 3]:
                oriented.append(candidate.reshape(-1, 2))
    if len(oriented) != 1:
        raise ValueError("board orientation is ambiguous")
    return np.array(oriented[0], dtype=float, copy=True)


@dataclass(frozen=True)
class Reference:
    board: Board
    image_to_xy: tuple
    corners_uv: tuple
    fit_ids: tuple
    held_ids: tuple
    held_rms_px: float
    held_max_px: float
    rgb_sha256: str

    def pixels(self):
        # Nearest integer array index, ties toward +infinity. The oracle does
        # not read or reuse the producer's declared raster-center offset.
        return tuple(
            tuple(int(v) for v in np.floor(np.asarray(self.corners_uv)[i] + 0.5))
            for i in self.held_ids
        )

    def expected_xyz(self):
        xy = apply_homography(self.image_to_xy, self.pixels())
        return np.c_[xy, np.full(len(xy), self.board.origin_xyz_m[2])]


def reference_from_rgb(rgb, board=Board()):
    from .checker_accuracy import AccuracyBoard, accuracy_reference_from_rgb
    if type(board) is AccuracyBoard:
        return accuracy_reference_from_rgb(rgb, board)
    from .binary_layout_a import BinaryLayoutABoard, layout_reference_from_rgb
    if type(board) is BinaryLayoutABoard:
        return layout_reference_from_rgb(rgb, board)
    from .binary_reference import BinaryGroundBoard, binary_reference_from_rgb
    if type(board) is BinaryGroundBoard:
        return binary_reference_from_rgb(rgb, board)
    corners = detect_corners(rgb, board)
    return reference_from_corners(
        corners, board, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()
    )


def reference_from_corners(corners, board=Board(), *, rgb_sha256):
    """Low-level fit for CPU tests/recorded detections; no implicit image proof."""
    corners = _points(corners, (board.columns - 1) * (board.rows - 1))
    if (
        not isinstance(rgb_sha256, str)
        or len(rgb_sha256) != 64
        or any(c not in "0123456789abcdef" for c in rgb_sha256)
    ):
        raise ValueError("original RGB byte digest required")
    fit = tuple(
        i
        for i in range(len(corners))
        if (i // (board.columns - 1) + i % (board.columns - 1)) % 2 == 0
    )
    held = tuple(i for i in range(len(corners)) if i not in fit)
    if min(len(fit), len(held)) < 12:
        raise ValueError("at least 12 fixed fit and held-out corners required")
    world = board.corners_xy()
    h = fit_homography(corners[list(fit)], world[list(fit)])
    residual = np.linalg.norm(
        apply_homography(np.linalg.inv(h), world[list(held)]) - corners[list(held)],
        axis=1,
    )
    rms, maximum = float(np.sqrt(np.mean(residual**2))), float(residual.max())
    if rms > 0.15 or maximum > 0.35:
        raise ValueError(
            "independent held-out reference error exceeds frozen pixel gates"
        )
    return Reference(
        board,
        tuple(h.flatten()),
        tuple(map(tuple, corners)),
        fit,
        held,
        rms,
        maximum,
        rgb_sha256,
    )


def compare_points(reference, measured_xyz):
    """Diagnostic only. Caller must separately enforce capture/epoch/age bindings."""
    measured = np.asarray(measured_xyz, dtype=float)
    expected = reference.expected_xyz()
    if measured.shape != expected.shape or not np.isfinite(measured).all():
        raise ValueError("all declared finite surface points required")
    errors = measured - expected
    pixels = apply_homography(
        np.linalg.inv(np.asarray(reference.image_to_xy).reshape(3, 3)), measured[:, :2]
    )
    residual = np.linalg.norm(pixels - np.asarray(reference.pixels()), axis=1)
    axes = np.abs(errors).max(axis=0)
    rms = float(np.sqrt(np.mean(residual**2)))
    passed = bool(max(axes) <= 0.001 and rms <= 0.35 and residual.max() <= 0.7)
    return {
        "passed": passed,
        "scope": "planar_render_reference_diagnostic",
        "physical_admission": False,
        "board_sha256": reference.board.sha256,
        "rgb_sha256": reference.rgb_sha256,
        "reference_rms_px": reference.held_rms_px,
        "reference_max_px": reference.held_max_px,
        "max_abs_error_xyz_m": axes.tolist(),
        "signed_errors_xyz_m": errors.tolist(),
        "reference_pixel_rms": rms,
        "reference_pixel_max": float(residual.max()),
        "limits": {"max_abs_axis_error_m": 0.001, "pixel_rms": 0.35, "pixel_max": 0.7},
    }


def compare_annotations(reference, observation, results, *, admitted_max_age_s=2.0):
    """Audit full retained spatial receipts; never read, refresh or admit a frame.

    Recorded consumer ages are checked, not replaced by the audit's later clock.
    These content checks cannot attest a dishonest producer or establish new
    physical authority. The image-only oracle still receives no K/T/depth.
    """
    from cascade.sensing.models import ObservationEnvelope, RgbdPayload

    if (
        type(observation) is not ObservationEnvelope
        or type(observation.payload) is not RgbdPayload
    ):
        raise ValueError("original typed RGB-D capture required")
    if (
        type(admitted_max_age_s) not in (int, float)
        or not np.isfinite(admitted_max_age_s)
        or not 0 < admitted_max_age_s <= 2.0
    ):
        raise ValueError(
            "explicit recorded consumer age bound must not exceed two seconds"
        )
    payload = observation.payload
    if hashlib.sha256(payload.rgb8).hexdigest() != reference.rgb_sha256:
        raise ValueError("reference RGB bytes do not belong to capture")
    if not isinstance(results, (list, tuple)) or len(results) != len(
        reference.held_ids
    ):
        raise ValueError(
            "exactly one result for every declared held-out pixel required"
        )
    binding = {
        "capture_sha256": observation.sha256,
        "source": observation.source,
        "model_identity_sha256": observation.model_identity_sha256,
        "epoch": observation.epoch,
        "sequence": observation.sequence,
        "clock_domain": observation.clock_domain,
        "capture_time_s": observation.capture_time_s,
        "received_monotonic_s": observation.received_monotonic_s,
        "producer_age_s": observation.producer_age_s,
        "calibration_sha256": payload.metadata.calibration_id,
        "world_frame_id": "world",
    }
    points = []
    ids = set()
    for pixel, result in zip(reference.pixels(), results):
        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
            or result.get("physical_admission") is not False
        ):
            raise ValueError("refused or incorrectly scoped annotation")
        age = result.get("capture_age_s")
        if (
            type(age) not in (int, float)
            or not np.isfinite(age)
            or not observation.producer_age_s <= age <= admitted_max_age_s
        ):
            raise ValueError("recorded capture age outside admitted bound")
        entry = result.get("result")
        if not isinstance(entry, dict) or entry.get("sha256") != digest(
            {k: v for k, v in entry.items() if k != "sha256"}
        ):
            raise ValueError("annotation content digest mismatch")
        provenance = entry.get("provenance", {})
        if any(provenance.get(k) != v for k, v in binding.items()) or provenance.get(
            "pixel_uv"
        ) != list(pixel):
            raise ValueError("annotation does not match original capture/pixel binding")
        if (
            entry.get("physical_admission") is not False
            or entry.get("confidence") is not None
            or entry.get("observation_id") in ids
        ):
            raise ValueError("duplicate or incorrectly scoped surface observation")
        ids.add(entry.get("observation_id"))
        points.append(entry["point_map_m"])
    return compare_points(reference, points) | {
        "capture_binding": binding,
        "annotation_sha256": [r["result"]["sha256"] for r in results],
        "recorded_consumer_age_bound_s": admitted_max_age_s,
        "new_capture_admission": False,
    }
