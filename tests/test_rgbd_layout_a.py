"""CPU raster/detector controls; no rendered capture or native admission."""

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmark.rgbd import binary_layout_a as layout
from benchmark.rgbd import binary_reference as legacy
from benchmark.rgbd.ground_texture import ST, png_bytes, texture_rectangles
from benchmark.rgbd.homography_support import require_tag_support, whole_tag_lower_bound
from benchmark.rgbd.planar_reference import compare_points, reference_from_rgb


@pytest.fixture(autouse=True)
def bounded_cv_threads():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(previous)


def raster(*, replacements=None, erase_corner=False):
    """Independent sample-center local raster; no K/T or design projection."""
    board = layout.BinaryLayoutABoard()
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    x, y = (u + 0.5 - 240) / 1000, (360 - v - 0.5) / 1000
    rgb = np.full((480, 640, 3), 96, np.uint8)
    for row in layout.layout_rectangles(board):
        a, b, c, d = row["local_xy_bounds_m"]
        rgb[(x >= a) & (x < c) & (y >= b) & (y < d)] = (replacements or {}).get(
            row["label"], row["rgb8"]
        )
    if erase_corner:
        rgb[302:322, 278:298] = 127
    return board, rgb


def replace_tag(rgb, board, marker, new_id=None):
    x, y = board.tag_centers_local_m[marker]
    u, v = np.rint([240 + x * 1000 - 42, 360 - y * 1000 - 42]).astype(int)
    patch = np.full((84, 84), 255, np.uint8)
    if new_id is not None:
        patch = cv2.aruco.generateImageMarker(legacy._api()[1], new_id, 84)
    rgb[v : v + 84, u : u + 84] = patch[..., None]


def test_new_schema_geometry_and_historical_bytes_remain_distinct():
    board = layout.BinaryLayoutABoard()
    assert board.description()["schema"] == 4
    assert board.origin_xyz_m == layout.ORIGIN and board.yaw_deg == -44.0
    assert board.tag_size_m == 0.084 and board.tag_centers_local_m == layout.CENTERS
    assert board.description()["frame_id"] == "world"
    assert (
        board.description()["observed_resolution"]["min_whole_tag_singular_px_cell"]
        == 4
    )
    assert legacy.BinaryGroundBoard().description()["schema"] == 3
    assert legacy.BinaryGroundBoard().description()["ground"]["st"] == list(
        map(list, ST)
    )
    previous = (
        Path(__file__).parents[1] / "benchmark/rgbd/assets/ground_binary_reference.png"
    )
    assert (
        hashlib.sha256(previous.read_bytes()).hexdigest()
        == "1debc536631b0a9701c82d27db51129bc826bbc23e549f45d8b6ef56f1dbc0eb"
    )
    assert png_bytes(legacy.BinaryGroundBoard()) == previous.read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [
        ("yaw_deg", -43.0),
        ("yaw_deg", True),
        ("tag_size_m", 0.085),
        ("frame_id", "camera"),
        ("square_m", 0.025),
        ("rows", 7),
        ("origin_xyz_m", (0.0315, 0.0705, 0)),
        ("tag_centers_local_m", ((0, 0),) * 4),
    ],
)
def test_other_geometry_cannot_impersonate_variant_a(field, value):
    with pytest.raises(ValueError, match="layout A"):
        replace(layout.BinaryLayoutABoard(), **{field: value})


def test_mutable_centers_are_frozen_and_description_is_defensive():
    centers = list(map(list, layout.CENTERS))
    board = layout.BinaryLayoutABoard(tag_centers_local_m=centers)
    before = board.sha256
    centers[0][0] = 100
    description = board.description()
    description["ground"]["st"][0][0] = 100
    assert board.sha256 == before


def test_local_cells_are_exact_and_uv_quantization_is_explicit():
    board = layout.BinaryLayoutABoard()
    rows = texture_rectangles(board)
    assert len(rows) == 193
    for row in rows:
        u0, v0, u1, v1 = row["bounds_uv"]
        np.testing.assert_array_equal(
            [
                (u0 - 2000) / 2000,
                (2000 - v1) / 2000,
                (u1 - 2000) / 2000,
                (2000 - v0) / 2000,
            ],
            row["local_xy_bounds_m"],
        )
        if row["label"].startswith("tag_"):
            assert u1 - u0 == v1 - v0 == 28
        elif row["label"].startswith("square_"):
            assert u1 - u0 == v1 - v0 == 48
    world = np.asarray(ST) * 2 - 1
    encoded_local = np.asarray(board.texture_st()) * 2 - 1
    reconstructed_world = board.world_xy(encoded_local)
    # Stored float32 UV quantization is bound; it is not claimed to be zero.
    error = np.linalg.norm(reconstructed_world - world, axis=1).max()
    assert 0 < error < 3e-7
    assert (
        np.asarray(board.texture_st()).tolist() == board.description()["ground"]["st"]
    )


@pytest.mark.parametrize("turns", range(4))
def test_rotated_image_preserves_declared_world_geometry_and_18_17_split(turns):
    board, rgb = raster()
    rgb = np.ascontiguousarray(np.rot90(rgb, turns))
    reference = reference_from_rgb(rgb, board)
    assert len(reference.fit_ids) == 18 and len(reference.held_ids) == 17
    local = np.array([[c * 0.024, r * 0.024] for r in range(1, 6) for c in range(1, 8)])
    expected = local * [1000, -1000] + [239.5, 359.5]
    width, height = 640, 480
    for _ in range(turns):
        expected = np.c_[expected[:, 1], width - 1 - expected[:, 0]]
        width, height = height, width
    np.testing.assert_allclose(reference.corners_uv, expected, atol=0.15, rtol=0)
    np.testing.assert_array_equal(board.corners_xy(), board.world_xy(local))
    assert compare_points(reference, reference.expected_xyz())["passed"]
    for axis in (0, 1):
        changed = reference.expected_xyz().copy()
        changed[:, axis] += 0.002
        assert not compare_points(reference, changed)["passed"]


def test_detector_output_order_is_not_orientation_authority(monkeypatch):
    board, rgb = raster()
    expected = reference_from_rgb(rgb, board)
    original = cv2.aruco.ArucoDetector

    class Reverse:
        def __init__(self, *args):
            self.inner = original(*args)

        def detectMarkers(self, image):
            corners, ids, rejected = self.inner.detectMarkers(image)
            return corners[::-1], ids[::-1], rejected

    monkeypatch.setattr(cv2.aruco, "ArucoDetector", Reverse)
    assert reference_from_rgb(rgb, board) == expected


@pytest.mark.parametrize(
    "kind", ["reflected", "missing", "duplicate", "foreign", "bit", "border", "corner"]
)
def test_corruption_and_missing_observation_are_not_metric_acceptance(kind):
    board, rgb = raster()
    if kind == "reflected":
        rgb = np.ascontiguousarray(rgb[:, ::-1])
    elif kind in ("missing", "duplicate", "foreign"):
        replace_tag(
            rgb, board, 0, {"missing": None, "duplicate": 1, "foreign": 49}[kind]
        )
    elif kind == "corner":
        board, rgb = raster(erase_corner=True)
    else:
        label = "tag_0_2_2" if kind == "bit" else "tag_0_0_2"
        row = next(r for r in layout.layout_rectangles(board) if r["label"] == label)
        board, rgb = raster(replacements={label: [255 - row["rgb8"][0]] * 3})
    with pytest.raises(ValueError):
        reference_from_rgb(rgb, board)


def test_ambiguous_tag_location_refuses_orientation(monkeypatch):
    board, rgb = raster()
    tags, stats = layout.detected_layout_tags(rgb, board)
    tags[1] = tags[0]
    monkeypatch.setattr(layout, "detected_layout_tags", lambda *a: (tags, stats))
    with pytest.raises(ValueError, match="orientation"):
        reference_from_rgb(rgb, board)


def test_consumer_pin_checked_before_any_detection(monkeypatch):
    declaration = legacy.detector_recipe()
    declaration["opencv_version"] = "99.0.0"
    board = layout.BinaryLayoutABoard(
        consumer_detector_json=json.dumps(
            declaration, sort_keys=True, separators=(",", ":")
        )
    )
    monkeypatch.setattr(
        legacy, "detected_tags", lambda *a: pytest.fail("mismatch reached detector")
    )
    with pytest.raises(ValueError, match="active implementation"):
        reference_from_rgb(np.zeros((1, 1, 3), np.uint8), board)


def test_shear_can_pass_edge_check_but_fails_whole_quad_resolution():
    # Legacy edge/cell lengths are10 and10.05px, but the small singular value
    # is below1px/cell. No threshold change or successful ID is fabricated.
    h = np.array([[10.0, 10.0, 100.0], [0.0, 1.0, 100.0], [0.0, 0.0, 1.0]])
    corners = np.c_[[[0, 0], [6, 0], [6, 6], [0, 6]], np.ones(4)] @ h.T
    points = corners[:, :2]
    assert np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).min() / 6 >= 2
    with pytest.raises(ValueError, match="singular resolution"):
        require_tag_support(points)


def test_legacy_edge_admission_and_new_whole_tag_gate_have_causal_difference(
    monkeypatch,
):
    board = layout.BinaryLayoutABoard()
    # An explicit detector double isolates geometry admission, not bit decoding.
    first = np.array([[50, 60], [92, 60], [134, 66], [92, 66]], np.float32)
    squares = [
        np.array([[x, y], [x + 60, y], [x + 60, y + 60], [x, y + 60]], np.float32)
        for x, y in ((200, 100), (350, 200), (450, 300))
    ]

    class Detector:
        def __init__(self, *args):
            pass

        def detectMarkers(self, image):
            return [v[None] for v in [first, *squares]], np.arange(4).reshape(4, 1), []

    monkeypatch.setattr(cv2.aruco, "ArucoDetector", Detector)
    rgb = np.zeros((480, 640, 3), np.uint8)
    assert legacy.detected_tags(rgb)[1]["min_projected_cell_edge_px"] >= 2
    with pytest.raises(ValueError, match="singular resolution"):
        layout.detected_layout_tags(rgb, board)


def test_whole_quad_gate_covers_corners_not_just_center():
    # Increasing denominator lowers resolution toward the far edge.
    h = np.array([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.12, 0.12, 1.0]])
    assert whole_tag_lower_bound(h) < 4
    affine = np.array([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 1.0]])
    bound = whole_tag_lower_bound(affine)
    assert Fraction(8) < bound < Fraction(9)
    pts = np.array([[100.0, 100.0], [172.0, 100.0], [172.0, 172.0], [100.0, 172.0]])
    assert require_tag_support(pts)["lower_bound_px_per_cell"] > 8


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((3, 3)),
        np.diag([-1.0, 1.0, 1.0]),
        np.diag([1.0, 1.0, -1.0]),
        np.full((3, 3), np.nan),
        np.eye(2),
    ],
)
def test_invalid_or_reflected_homography_rejected(matrix):
    with pytest.raises(ValueError):
        whole_tag_lower_bound(matrix)


def test_inconclusive_conservative_bound_never_claims_admission():
    value = whole_tag_lower_bound(
        np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert value < Fraction(4)
    with pytest.raises(ValueError, match="four pixels"):
        require_tag_support([[100, 100], [130, 100], [130, 130], [100, 130]])


@pytest.mark.parametrize("scale,sigma", [(1.0, 0.6), (0.75, 0.5)])
def test_supported_cpu_blur_and_rescale_preserve_metric_partition(scale, sigma):
    board, rgb = raster()
    rgb = cv2.GaussianBlur(rgb, (3, 3), sigma)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    reference = reference_from_rgb(rgb, board)
    assert len(reference.fit_ids) == 18 and len(reference.held_ids) == 17
    assert compare_points(reference, reference.expected_xyz())["passed"]


@pytest.mark.parametrize("fault", ["undersampled", "severe_blur"])
def test_raster_outside_declared_support_never_becomes_metric_acceptance(fault):
    board, rgb = raster()
    if fault == "undersampled":
        rgb = cv2.resize(rgb, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    else:
        rgb = cv2.GaussianBlur(rgb, (61, 61), 14)
    with pytest.raises(ValueError):
        reference_from_rgb(rgb, board)


def test_rational_bound_is_below_independent_sampled_jacobian_singular_values():
    h = np.array([[12.0, 1.5, 20.0], [0.7, 11.0, 30.0], [0.007, 0.01, 1.0]])
    bound = whole_tag_lower_bound(h)
    values = []
    for x in np.linspace(0, 6, 19):
        for y in np.linspace(0, 6, 19):
            projected = h @ [x, y, 1]
            jacobian = (
                h[:2, :2] * projected[2] - projected[:2, None] * h[2:3, :2]
            ) / projected[2] ** 2
            values.append(np.linalg.svd(jacobian, compute_uv=False)[-1])
    assert 4 < float(bound) <= min(values)
