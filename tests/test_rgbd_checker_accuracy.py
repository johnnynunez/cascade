"""Image-only schema5 controls, separate from the frozen48-case corpus."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import pytest

from benchmark.rgbd import checker_accuracy as accuracy
from benchmark.rgbd import checker_corpus as corpus
from benchmark.rgbd import binary_layout_a as layout
from benchmark.rgbd.planar_reference import reference_from_rgb
from test_rgbd_layout_a import raster as legacy_raster, replace_tag


@pytest.fixture(autouse=True)
def threads():
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(old)


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def test_explicit_recipe_and_legacy_description_unchanged():
    before = layout.BinaryLayoutABoard().description()
    board = accuracy.AccuracyBoard()
    value = board.description()
    assert value["schema"] == 5 and before["schema"] == 4
    assert value["checker"]["flags_value"] == 34
    assert value["checker"]["post_refinement"] is None
    assert value["ground"] == before["ground"]
    assert value["texture"] == before["texture"]
    assert board.geometry.description() == before
    value["checker"]["flags_value"] = 2
    assert board.description()["checker"]["flags_value"] == 34
    assert layout.BinaryLayoutABoard().description() == before
    assert board.sha256 != board.geometry.sha256


@pytest.mark.parametrize(
    "change", ["flags", "pattern", "extra", "schema", "missing", "bool"]
)
def test_descriptor_policy_cannot_silently_change(change):
    r = accuracy.consumer_recipe()
    if change == "flags":
        r["checker"]["flags_value"] = 2
    elif change == "pattern":
        r["checker"]["pattern_size"] = [5, 7]
    elif change == "extra":
        r["checker"]["bias"] = 0
    elif change == "schema":
        r["schema"] = 2
    elif change == "bool":
        r["schema"] = True
    else:
        del r["checker"]
    with pytest.raises(ValueError):
        accuracy.AccuracyBoard(canonical(r))


@pytest.mark.parametrize(
    "field,value",
    [
        ("opencv_version", "99.0.0"),
        ("opencv_package_sha256", "a" * 64),
        ("dictionary_bytes_sha256", "b" * 64),
    ],
)
def test_consumer_mismatch_before_pixels(field, value, monkeypatch):
    r = accuracy.consumer_recipe()
    r["aruco"][field] = value
    if field in r["checker"]:
        r["checker"][field] = value
    board = accuracy.AccuracyBoard(canonical(r))
    monkeypatch.setattr(
        layout, "detected_layout_tags", lambda *a: pytest.fail("pixels inspected")
    )
    with pytest.raises(ValueError, match="active implementation"):
        reference_from_rgb(np.zeros((1, 1, 3), np.uint8), board)


def test_cross_sdk_authoring_declaration_does_not_claim_consumer_execution():
    r = accuracy.consumer_recipe()
    for section in ("aruco", "checker"):
        r[section]["opencv_version"] = "4.14.0"
        r[section]["opencv_package_sha256"] = "a" * 64
    board = accuracy.AccuracyBoard(canonical(r))
    assert board.description()["checker"]["opencv_version"] == "4.14.0"
    with pytest.raises(ValueError, match="active implementation"):
        board.require_consumer_implementation()


@pytest.mark.parametrize("turns", range(4))
def test_real_checker_accuracy_rotations_preserve_split_and_coordinates(turns):
    _, rgb = legacy_raster()
    rgb = np.ascontiguousarray(np.rot90(rgb, turns))
    ref = reference_from_rgb(rgb, accuracy.AccuracyBoard())
    assert (
        len(ref.corners_uv) == 35 and len(ref.fit_ids) == 18 and len(ref.held_ids) == 17
    )
    assert ref.held_rms_px <= 0.15 and ref.held_max_px <= 0.35
    assert ref.rgb_sha256 == hashlib.sha256(rgb.tobytes()).hexdigest()


@pytest.mark.parametrize(
    "row_reverse,column_reverse",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_detector_order_reversal_is_not_geometry_authority(
    monkeypatch, row_reverse, column_reverse
):
    _, rgb = legacy_raster()
    board = accuracy.AccuracyBoard()
    reference = reference_from_rgb(rgb, board)
    grid = np.array(reference.corners_uv).reshape(5, 7, 2)
    if row_reverse:
        grid = grid[::-1]
    if column_reverse:
        grid = grid[:, ::-1]

    def sb(gray, size, flags):
        assert size == (7, 5) and flags == 34
        return True, grid.reshape(-1, 1, 2)

    monkeypatch.setattr(cv2, "findChessboardCornersSB", sb)
    assert reference_from_rgb(rgb, board) == reference


@pytest.mark.parametrize(
    "kind", ["reflect", "missing", "duplicate", "foreign", "bit", "corner"]
)
def test_invalid_image_never_becomes_reference(kind):
    base, rgb = legacy_raster()
    if kind == "reflect":
        rgb = np.ascontiguousarray(rgb[:, ::-1])
    elif kind in ("missing", "duplicate", "foreign"):
        replace_tag(
            rgb, base, 0, {"missing": None, "duplicate": 1, "foreign": 49}[kind]
        )
    elif kind == "corner":
        _, rgb = legacy_raster(erase_corner=True)
    else:
        row = next(
            r for r in layout.layout_rectangles(base) if r["label"] == "tag_0_2_2"
        )
        _, rgb = legacy_raster(replacements={"tag_0_2_2": [255 - row["rgb8"][0]] * 3})
    with pytest.raises(ValueError):
        reference_from_rgb(rgb, accuracy.AccuracyBoard())


def test_fabricated_corner_without_contrast_is_rejected():
    board, rgb = legacy_raster()
    tags, _ = layout.detected_layout_tags(rgb, board)
    grid = np.array(
        [[239.5 + c * 24, 359.5 - r * 24] for r in range(1, 6) for c in range(1, 8)]
    )
    gray = np.full(rgb.shape[:2], 127, np.uint8)
    with pytest.raises(ValueError, match="contrast"):
        accuracy.orient_corners(gray, grid, tags)


@pytest.mark.parametrize(
    "h",
    [
        np.array([[55.0, 0, 19.2], [0, -60, 27.3], [0, 0, 1]]),
        np.array([[45.0, 13, 18.4], [4, -41, 22.8], [0.2, -0.1, 1]]),
    ],
)
def test_accelerated_raster_matches_exhaustive_all_boundary_samples(h):
    rectangles = [[-0.12, -0.12, 0, 0], [0, 0, 0.12, 0.12]]
    fast = corpus.raster(h, rectangles, width=48, height=48, samples=16)
    slow = corpus.raster(
        h, rectangles, width=48, height=48, samples=16, exhaustive=True
    )
    np.testing.assert_array_equal(fast, slow)


def test_raster_truth_uses_integer_pixel_centers_and_area():
    h = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    rgb = corpus.raster(h, [[0.5, 0.5, 1.5, 1.5]], width=3, height=3)
    assert rgb[1, 1].tolist() == [0, 0, 0]
    assert np.count_nonzero(rgb == 0) == 3
    np.testing.assert_array_equal(corpus.project(h, [[1, 1]]), [[1, 1]])


def test_extracted_legacy_corner_path_preserves_frozen_result(monkeypatch):
    # Detector output retained from the pre-refactor917 source. This isolates
    # extraction/wiring independently of OpenCV detector versions. The
    # subsequent linear algebra may differ in its last bits across platforms.
    fixture = json.loads(
        (
            Path(__file__).with_name("fixtures") / "rgbd_layout_a_reference_917.json"
        ).read_text()
    )
    board, rgb = legacy_raster()
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == fixture["rgb_sha256"]

    def sb(gray, pattern, flags):
        assert pattern == (7, 5) and flags == 2
        return True, np.array(fixture["reference"]["corners_uv"], np.float32).reshape(
            -1, 1, 2
        )

    monkeypatch.setattr(cv2, "findChessboardCornersSB", sb)
    value = json.loads(json.dumps(asdict(reference_from_rgb(rgb, board))))
    expected = fixture["reference"]
    numeric = {"image_to_xy", "held_rms_px", "held_max_px"}
    assert {k: v for k, v in value.items() if k not in numeric} == {
        k: v for k, v in expected.items() if k not in numeric}
    for key in numeric:
        # CI differs by at most 2.51e-13 px in the retained residuals. This
        # comparison concerns floating-point extraction equivalence only;
        # production pixel gates and the frozen reference remain unchanged.
        np.testing.assert_allclose(value[key], expected[key], rtol=0, atol=1e-12)
