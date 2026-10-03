"""Actual OpenCV CPU detection, not RTX rendering or physical acceptance."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmark.rgbd.binary_reference import (
    BinaryGroundBoard,
    binary_rectangles,
    detected_tags,
    detector_recipe,
)
from benchmark.rgbd.ground_texture import (
    GroundTextureBoard,
    png_bytes,
    texture_rectangles,
)
from benchmark.rgbd.planar_reference import (
    compare_points,
    reference_from_rgb,
    reference_from_corners,
)


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(previous)


def raster(*, board=None, scale=1500, replacements=None):
    """Independent sample-center raster of declared rectangles; no K/T/depth."""
    board = board or BinaryGroundBoard()
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    x = board.origin_xyz_m[0] + (u + 0.5 - 180) / scale
    y = board.origin_xyz_m[1] + (360 - v - 0.5) / scale
    rgb = np.full((480, 640, 3), 96, np.uint8)
    for row in binary_rectangles(board):
        a, b, c, d = row["xy_bounds_m"]
        color = (replacements or {}).get(row["label"], row["rgb8"])
        rgb[(x >= a) & (x < c) & (y >= b) & (y < d)] = color
    return board, rgb


def binary_capture(mutation=None):
    """Independent CPU pixels plus a declared projection hypothesis to test."""
    from benchmark.rgbd.planar_reference import digest
    from cascade.sensing.models import (
        MeasurementMetadata,
        ObservationEnvelope,
        RgbdPayload,
    )

    board, rgb = raster()
    k = np.array([[1500.0, 0, 180.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    t = np.diag([1.0, -1.0, -1.0, 1.0])
    t[:3, 3] = [*board.origin_xyz_m[:2], 1.0]
    if mutation == "fx":
        k[0, 0] *= 1.02
    elif mutation == "fy":
        k[1, 1] *= 1.02
    calibration = digest({"k": k.tolist(), "t": t.tolist(), "offset": [0.5, 0.5]})
    payload = RgbdPayload(
        MeasurementMetadata("camera", calibration),
        640,
        480,
        rgb.tobytes(),
        np.ones((480, 640), dtype="<f4").tobytes(),
        k.flatten().tolist(),
        t.flatten().tolist(),
        "world",
        (0.5, 0.5),
    )
    capture = ObservationEnvelope(
        "cpu-binary-texture",
        "overview",
        "synthetic-epoch",
        1,
        "synthetic",
        0.1,
        10.0,
        0.1,
        None,
        "synthetic",
        payload,
    )
    return board, capture


def tag_patch(rgb, board, marker, replacement_id=None, *, erase=False):
    center = board.marker_centers_xy()[marker]
    u, v = (center - board.origin_xyz_m[:2]) * [1500, -1500] + [180, 360]
    lo = np.rint(np.asarray([u, v]) - board.tag_size_m * 1500 / 2).astype(int)
    hi = lo + int(board.tag_size_m * 1500)
    patch = np.full((hi[1] - lo[1], hi[0] - lo[0]), 255, np.uint8)
    if not erase:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        patch = cv2.aruco.generateImageMarker(dictionary, replacement_id, len(patch))
    rgb[lo[1] : hi[1], lo[0] : hi[0]] = patch[..., None]


@pytest.mark.parametrize("turns", range(4))
def test_rotations_preserve_checker_correspondence_and_fixed_split(turns):
    board, rgb = raster()
    image = np.ascontiguousarray(np.rot90(rgb, turns))
    reference = reference_from_rgb(image, board)
    assert len(reference.fit_ids) == 18 and len(reference.held_ids) == 17
    assert set(reference.fit_ids).isdisjoint(reference.held_ids)
    assert reference.held_rms_px <= 0.15 and reference.held_max_px <= 0.35
    expected = (board.corners_xy() - board.origin_xyz_m[:2]) * [1500, -1500] + [
        179.5,
        359.5,
    ]
    width, height = 640, 480
    for _ in range(turns):
        expected = np.c_[expected[:, 1], width - 1 - expected[:, 0]]
        width, height = height, width
    np.testing.assert_allclose(reference.corners_uv, expected, atol=0.15, rtol=0)
    assert reference.rgb_sha256 == hashlib.sha256(image.tobytes()).hexdigest()
    assert compare_points(reference, reference.expected_xyz())["passed"]
    changed = reference.expected_xyz().copy()
    changed[:, 0] += 0.005
    assert not compare_points(reference, changed)["passed"]


def test_detection_output_order_has_no_authority(monkeypatch):
    board, rgb = raster()
    expected = reference_from_rgb(rgb, board)
    original = cv2.aruco.ArucoDetector

    class Reverse:
        def __init__(self, *args):
            self.detector = original(*args)

        def detectMarkers(self, image):
            corners, ids, rejected = self.detector.detectMarkers(image)
            return corners[::-1], ids[::-1], rejected

    monkeypatch.setattr(cv2.aruco, "ArucoDetector", Reverse)
    assert reference_from_rgb(rgb, board) == expected


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "foreign", "bit", "border", "occluded"]
)
def test_corrupt_incomplete_and_extra_identity_rejects(mutation):
    board, rgb = raster()
    if mutation in ("missing", "duplicate", "foreign"):
        tag_patch(
            rgb,
            board,
            0,
            1 if mutation == "duplicate" else 49,
            erase=mutation == "missing",
        )
    elif mutation == "occluded":
        rgb[390:420, 110:170] = 96
    else:
        label = "tag_0_2_2" if mutation == "bit" else "tag_0_0_2"
        row = next(r for r in binary_rectangles(board) if r["label"] == label)
        _, rgb = raster(replacements={label: [255 - row["rgb8"][0]] * 3})
    with pytest.raises(ValueError, match="declared binary ID"):
        reference_from_rgb(rgb, board)


def test_missing_checker_corner_rejects_despite_complete_tags():
    board, rgb = raster()
    rgb[285:315, 225:255] = 127  # covers an interior checker intersection
    assert detected_tags(rgb)[1]["ids"] == [0, 1, 2, 3]
    # A newer SDK may correctly reject before fitting; that is also acceptable.
    with pytest.raises(ValueError):
        reference_from_rgb(rgb, board)


def test_retained_inferred_corner_is_rejected_by_contrast_guard(monkeypatch):
    board, rgb = raster()
    rgb[285:315, 225:255] = 127
    retained = json.loads(
        (
            Path(__file__).parent / "fixtures" / "rgbd_binary_occluded_corners.json"
        ).read_text()
    )
    assert retained["rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
    inferred = np.asarray(retained["corners_uv"], dtype=np.float32).reshape(-1, 1, 2)
    assert len(inferred) == 35
    # Frozen actual 5.0 output models inference without requiring another SDK
    # to repeat that behavior. Detection of tags remains real OpenCV.
    monkeypatch.setattr(
        cv2, "findChessboardCornersSB", lambda *a, **kw: (True, inferred.copy())
    )
    raw_fit = reference_from_corners(
        inferred.reshape(-1, 2),
        board,
        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
    )
    assert raw_fit.held_rms_px <= 0.15 and raw_fit.held_max_px <= 0.35
    with pytest.raises(ValueError, match="quadrant contrast"):
        reference_from_rgb(rgb, board)


@pytest.mark.parametrize("sigma", [0.4, 0.7])
def test_mild_blur_retains_unchanged_metric_gates(sigma):
    board, rgb = raster()
    image = cv2.GaussianBlur(rgb, (5, 5), sigma)
    reference = reference_from_rgb(image, board)
    assert reference.held_rms_px <= 0.15 and reference.held_max_px <= 0.35


@pytest.mark.parametrize("scale", [0.75, 0.6])
def test_downscale_inside_declared_support(scale):
    board, rgb = raster()
    image = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    reference = reference_from_rgb(image, board)
    assert detected_tags(image)[1]["min_projected_cell_edge_px"] >= 2
    assert reference.held_rms_px <= 0.15 and reference.held_max_px <= 0.35


@pytest.mark.parametrize(
    "mutation", ["blur", "small", "margin", "mirror", "rescaled_margin"]
)
def test_outside_support_is_not_metric_acceptance(mutation):
    board, rgb = raster()
    if mutation == "blur":
        image = cv2.GaussianBlur(rgb, (51, 51), 10)
    elif mutation == "small":
        image = cv2.resize(rgb, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    elif mutation == "margin":
        image = rgb[:, 110:]
    elif mutation == "rescaled_margin":
        image = cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    else:
        image = np.ascontiguousarray(rgb[:, ::-1])
    with pytest.raises(ValueError):
        reference_from_rgb(image, board)


def test_new_recipe_exact_texel_edges_and_legacy_bytes():
    board = BinaryGroundBoard()
    rows = texture_rectangles(board)
    assert len(rows) == 1 + 48 + 4 * 36
    assert all(all(type(v) is int for v in r["bounds_uv"]) for r in rows)
    assert board.description()["schema"] == 3
    assert detector_recipe()["use_tags_for_metric_fit"] is False
    assert detector_recipe()["parameters"]["errorCorrectionRate"] == 0
    assert hashlib.sha256(png_bytes(GroundTextureBoard())).hexdigest() == (
        "2638e36175318b42f2b477f3e8539b9afb441738f4737fb4f4177ba9dab05253"
    )
    with pytest.raises(ValueError, match="texel"):
        texture_rectangles(replace(board, tag_size_m=0.029))


def test_missing_optional_api_fails_explicitly(monkeypatch):
    board, rgb = raster()
    monkeypatch.delattr(cv2.aruco, "ArucoDetector")
    with pytest.raises(ValueError, match="API is required"):
        reference_from_rgb(rgb, board)


@pytest.mark.parametrize(
    "bit", [(row, col) for row in range(1, 5) for col in range(1, 5)]
)
def test_every_single_payload_bit_corruption_is_rejected_without_correction(bit):
    board = BinaryGroundBoard()
    label = f"tag_0_{bit[0]}_{bit[1]}"
    row = next(r for r in binary_rectangles(board) if r["label"] == label)
    _, rgb = raster(replacements={label: [255 - row["rgb8"][0]] * 3})
    with pytest.raises(ValueError, match="declared binary ID"):
        reference_from_rgb(rgb, board)


def test_extra_decoded_marker_is_not_selected_away():
    board, rgb = raster()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    rgb[40:120, 500:580] = 255
    rgb[50:110, 510:570] = cv2.aruco.generateImageMarker(dictionary, 49, 60)[..., None]
    with pytest.raises(ValueError, match="declared binary ID"):
        reference_from_rgb(rgb, board)


def test_correct_id_set_in_wrong_geometric_order_rejects():
    board, rgb = raster()
    tag_patch(rgb, board, 0, 1)
    tag_patch(rgb, board, 1, 0)
    assert detected_tags(rgb)[1]["ids"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="orientation is ambiguous"):
        reference_from_rgb(rgb, board)
