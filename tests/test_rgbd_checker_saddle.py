"""Local observation/consumer contracts; held-out images belong to the corpus."""

import json

import numpy as np
import pytest

from benchmark.rgbd import checker_saddle as saddle
from benchmark.rgbd.checker_accuracy import AccuracyBoard
from benchmark.rgbd.planar_reference import reference_from_rgb
from test_rgbd_layout_a import raster


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_local_saddle_recovers_quadratic_center_without_geometry():
    yy, xx = np.mgrid[:24, :24]
    gray = np.rint(128 + 3 * (xx - 10.5) * (yy - 11.25)).clip(0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    centers = saddle.refine_corners(rgb, np.tile([10.8, 11.4], (35, 1)))
    assert np.max(np.abs(centers - [10.5, 11.25])) < 0.03
    with pytest.raises(ValueError, match="neighborhood"):
        saddle.refine_corners(rgb, np.tile([12.4, 11.25], (35, 1)))


@pytest.mark.parametrize("kind", ["flat", "linear", "convex", "border", "huge", "nan"])
def test_invalid_local_patch_never_falls_back_to_seed(kind):
    yy, xx = np.mgrid[:24, :24]
    gray = 128 + 3 * (xx - 10) * (yy - 10)
    point = [10.0, 10.0]
    if kind == "flat":
        gray = np.full_like(gray, 128)
    if kind == "linear":
        gray = 80 + 4 * xx
    if kind == "convex":
        gray = 40 + (xx - 10) ** 2 + (yy - 10) ** 2
    if kind == "border":
        point = [2.0, 2.0]
    if kind == "huge":
        point = [1e100, 0.0]
    if kind == "nan":
        point = [np.nan, 10.0]
    rgb = np.repeat(gray.clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)
    with pytest.raises(ValueError):
        saddle.refine_corners(rgb, np.tile(point, (35, 1)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("radius_px", 4),
        ("gaussian_sigma_px", 1.0),
        ("max_seed_offset_px", 2.0),
        ("fallback", "seed"),
        ("extrapolation", True),
    ],
)
def test_refinement_policy_is_immutable(field, value):
    recipe = saddle.consumer_recipe()
    recipe["refinement"][field] = value
    with pytest.raises(ValueError, match="policy"):
        saddle.SaddleBoard(canonical(recipe))


def test_foreign_consumer_rejected_before_reading_any_pixels(monkeypatch):
    recipe = saddle.consumer_recipe()
    recipe["refinement"]["numpy_package_sha256"] = "a" * 64
    board = saddle.SaddleBoard(canonical(recipe))
    monkeypatch.setattr(
        saddle.seed,
        "detect_accuracy_corners",
        lambda *a: pytest.fail("seed pixels inspected"),
    )
    with pytest.raises(ValueError, match="active implementation"):
        reference_from_rgb(np.zeros((1, 1, 3), np.uint8), board)


def test_missing_seed_is_terminal(monkeypatch):
    def missing(*args):
        raise ValueError("missing checker")

    monkeypatch.setattr(saddle.seed, "detect_accuracy_corners", missing)
    monkeypatch.setattr(
        saddle,
        "refine_corners",
        lambda *a: pytest.fail("refinement after missing seed"),
    )
    with pytest.raises(ValueError, match="missing checker"):
        reference_from_rgb(np.zeros((1, 1, 3), np.uint8), saddle.SaddleBoard())


def test_real_image_refinement_is_explicit_and_preserves_the_original_gates():
    _, rgb = raster()
    board = saddle.SaddleBoard()
    result = reference_from_rgb(rgb, board)
    assert board.description()["schema"] == 6
    assert AccuracyBoard().description()["checker"]["post_refinement"] is None
    assert len(result.fit_ids) == 18 and len(result.held_ids) == 17
    assert result.held_rms_px <= 0.15 and result.held_max_px <= 0.35
    assert board.geometry.description() == AccuracyBoard().geometry.description()
    assert board.sha256 != AccuracyBoard().sha256
