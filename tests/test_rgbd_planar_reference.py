"""CPU raster/reference controls. No native render or physical admission."""

from dataclasses import replace
import copy
import hashlib

import cv2
import numpy as np
import pytest

from benchmark.rgbd.planar_reference import (
    Board,
    apply_homography,
    author_visual_board,
    compare_annotations,
    compare_points,
    detect_corners,
    digest,
    fit_homography,
    reference_from_corners,
    reference_from_rgb,
    visual_quads,
)
from cascade.sensing import BufferedSensorProvider, SensorHub
from cascade.sensing.hub import SensorDescriptor
from cascade.sensing.models import MeasurementMetadata, ObservationEnvelope, RgbdPayload
from cascade.spatial.rgbd import RgbdSpatialDomain


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def raster():
    """Independent orthographic CPU rasterizer, not a simulated camera claim.

    Every RGB sample evaluates authored world quads at array sample centers.
    No product K/T/projection helper participates in generating this image.
    """
    board = Board()
    scale, padding = 50 / board.square_m, 90
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    x = board.origin_xyz_m[0] + (u + 0.5 - padding) / scale
    y = board.origin_xyz_m[1] + (v + 0.5 - padding) / scale
    rgb = np.full((480, 640, 3), 96, dtype=np.uint8)
    for quad in visual_quads(board):
        p = np.asarray(quad["points_m"])
        mask = (
            (x >= p[:, 0].min())
            & (x < p[:, 0].max())
            & (y >= p[:, 1].min())
            & (y < p[:, 1].max())
        )
        rgb[mask] = np.asarray(quad["emissive_rgb"]) * 255
    return board, rgb, scale, padding


def exact_corners(board, image_from_xy):
    return apply_homography(image_from_xy, board.corners_xy())


def test_image_reference_uses_actual_raster_corners_and_array_indices():
    board, rgb, scale, pad = raster()
    reference = reference_from_rgb(rgb, board)
    expected = (board.corners_xy() - board.origin_xyz_m[:2]) * scale + pad - 0.5
    assert (
        np.max(np.linalg.norm(np.asarray(reference.corners_uv) - expected, axis=1))
        < 0.15
    )
    assert len(reference.fit_ids) == 18 and len(reference.held_ids) == 17
    assert set(reference.fit_ids).isdisjoint(reference.held_ids)
    assert reference.rgb_sha256 == hashlib.sha256(rgb.tobytes()).hexdigest()
    xy = (
        np.asarray(board.origin_xyz_m[:2])
        + (np.asarray(reference.pixels()) + 0.5 - pad) / scale
    )
    np.testing.assert_allclose(reference.expected_xyz()[:, :2], xy, atol=0.0001, rtol=0)


@pytest.mark.parametrize("flip", ["horizontal", "vertical", "both"])
def test_rgb_markers_fix_orientation_without_calibration(flip):
    board, rgb, _, _ = raster()
    image = (
        rgb[:, ::-1]
        if flip == "horizontal"
        else rgb[::-1]
        if flip == "vertical"
        else rgb[::-1, ::-1]
    )
    reference = reference_from_rgb(image.copy(), board)
    original = detect_corners(rgb, board)
    if flip in {"horizontal", "both"}:
        original[:, 0] = 639 - original[:, 0]
    if flip in {"vertical", "both"}:
        original[:, 1] = 479 - original[:, 1]
    np.testing.assert_allclose(reference.corners_uv, original, atol=0.2, rtol=0)


def test_projectively_warped_rgb_still_supplies_independent_metric_reference():
    board, rgb, scale, pad = raster()
    warp = np.array([[0.9, 0.05, 20.0], [0.02, 0.9, 10.0], [0.0002, -0.0001, 1.0]])
    transformed = cv2.warpPerspective(rgb, warp, (640, 480), borderValue=(96, 96, 96))
    reference = reference_from_rgb(transformed, board)
    unwarped = (board.corners_xy() - board.origin_xyz_m[:2]) * scale + pad - 0.5
    expected = apply_homography(warp, unwarped)
    assert (
        np.max(np.linalg.norm(np.asarray(reference.corners_uv) - expected, axis=1))
        < 0.2
    )
    inverse_pixels = apply_homography(np.linalg.inv(warp), reference.pixels())
    expected_xy = (
        np.asarray(board.origin_xyz_m[:2]) + (inverse_pixels + 0.5 - pad) / scale
    )
    np.testing.assert_allclose(
        reference.expected_xyz()[:, :2], expected_xy, atol=0.0001, rtol=0
    )


@pytest.mark.parametrize(
    "bad", ["missing_marker", "duplicate_marker", "blank", "oversized", "dtype"]
)
def test_missing_ambiguous_or_invalid_image_is_not_a_reference(bad):
    board, rgb, _, _ = raster()
    if bad == "missing_marker":
        rgb[(rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 30) & (rgb[:, :, 2] < 30)] = 96
    elif bad == "duplicate_marker":
        rgb[5:15, 5:15] = [255, 0, 0]
    elif bad == "blank":
        rgb[:] = 96
    elif bad == "oversized":
        rgb = np.zeros((481, 640, 3), dtype="uint8")
    elif bad == "dtype":
        rgb = rgb.astype(float)
    with pytest.raises(ValueError):
        reference_from_rgb(rgb, board)


@pytest.mark.parametrize(
    "h", [np.eye(3), np.array([[1400, 100, 70], [-60, 1100, 55], [0.2, -0.3, 1.0]])]
)
def test_normalized_projective_reference_has_fixed_holdout(h):
    board = Board()
    corners = exact_corners(board, h)
    reference = reference_from_corners(corners, board, rgb_sha256="a" * 64)
    np.testing.assert_allclose(
        apply_homography(reference.image_to_xy, corners), board.corners_xy(), atol=1e-12
    )
    assert reference.held_max_px < 1e-10


def test_corrupt_heldout_corner_is_not_silently_dropped():
    board = Board()
    corners = exact_corners(board, np.array([[1000, 0, 50], [0, 1000, 50], [0, 0, 1]]))
    corners[1] += [3, 0]
    with pytest.raises(ValueError, match="held-out"):
        reference_from_corners(corners, board, rgb_sha256="a" * 64)


def test_homography_rejects_degenerate_points_and_infinity():
    with pytest.raises(ValueError, match="degenerate"):
        fit_homography(
            [[0, 0], [1, 0], [2, 0], [3, 0]], [[0, 0], [0, 1], [1, 0], [1, 1]]
        )
    with pytest.raises(ValueError, match="infinity"):
        apply_homography([[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0]])


def projection_episode(mutation=None, *, with_capture=False):
    """Real retained SensorHub→spatial dispatch over explicit synthetic RGB-D."""
    board, rgb, scale, pad = raster()
    reference = reference_from_rgb(rgb, board)
    k = np.array([[scale, 0, pad], [0, scale, pad], [0, 0, 1.0]])
    t = np.eye(4)
    t[:3, 3] = [*board.origin_xyz_m[:2], board.origin_xyz_m[2] - 1]
    depth = np.ones((480, 640), dtype="<f4")
    offset = (0.5, 0.5)
    if mutation == "fx":
        k[0, 0] *= 1.02
    elif mutation == "fy":
        k[1, 1] *= 1.02
    elif mutation == "tx":
        t[0, 3] += 0.005
    elif mutation == "ty":
        t[1, 3] += 0.005
    elif mutation == "depth_units":
        depth *= 1000
    elif mutation == "offset":
        offset = (0.0, 0.0)
    elif mutation == "invalid_depth":
        u, v = reference.pixels()[0]
        depth[v, u] = 0
    calibration = digest({"k": k.tolist(), "t": t.tolist(), "offset": offset})
    payload = RgbdPayload(
        MeasurementMetadata("camera", calibration),
        640,
        480,
        rgb.tobytes(),
        depth.tobytes(),
        k.flatten().tolist(),
        t.flatten().tolist(),
        "world",
        offset,
    )
    obs = ObservationEnvelope(
        "cpu-raster",
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
    provider = BufferedSensorProvider(
        SensorDescriptor(
            "overview",
            "fixture",
            obs.source,
            "rgbd",
            "camera",
            "synthetic",
            "synthetic",
            calibration_id=calibration,
            max_age_s=2.0,
            read_timeout_s=2.0,
        )
    )
    now = [10.1]
    hub = SensorHub(clock=lambda: now[0])
    hub.register(provider)
    domain = RgbdSpatialDomain(
        "space",
        "fixture",
        hub,
        sensor_domain="sensing",
        sensor_id="overview",
        map_id="cpu-board",
        world_frame_id="world",
        clock=lambda: now[0],
    )
    provider.publish(obs)
    hub.read("overview")
    watermark = hub._slots["overview"].watermark

    def forbidden():
        pytest.fail("reference projection reacquired a capture")

    provider.read = forbidden
    results = []
    try:
        if mutation == "stale":
            now[0] = 12.1
        for i, pixel in enumerate(reference.pixels()):
            args = dict(
                epoch=obs.epoch,
                sequence=obs.sequence,
                capture_sha256=obs.sha256,
                pixel=list(pixel),
                observation_id=f"point-{i}",
                label="visual reference, not object identity",
            )
            if mutation == "epoch":
                args["epoch"] = "foreign"
            if mutation == "capture":
                args["capture_sha256"] = "a" * 64
            results.append(domain.execute("annotate_pixel", args))
        assert hub.history() == (obs,) and hub._slots["overview"].watermark == watermark
        assert obs.received_monotonic_s == 10.0
        return (reference, results, obs) if with_capture else (reference, results)
    finally:
        assert domain.close()["ok"]
        assert hub.close(2)["ok"]


def test_actual_consumer_matches_image_reference_in_both_axes():
    reference, results = projection_episode()
    assert all(r["ok"] for r in results)
    report = compare_points(reference, [r["result"]["point_map_m"] for r in results])
    assert report["passed"], report
    assert report["physical_admission"] is False
    assert max(report["max_abs_error_xyz_m"]) < 0.0001
    assert all(r["result"]["confidence"] is None for r in results)


@pytest.mark.parametrize("mutation", ["fx", "fy", "tx", "ty", "depth_units", "offset"])
def test_independent_reference_detects_corrupt_product_calibration(mutation):
    reference, results = projection_episode(mutation)
    assert all(r["ok"] for r in results)  # valid schema is not metric correctness
    report = compare_points(reference, [r["result"]["point_map_m"] for r in results])
    assert not report["passed"], report


@pytest.mark.parametrize("mutation", ["invalid_depth", "stale", "epoch", "capture"])
def test_consumer_refusal_never_becomes_geometry_pass(mutation):
    _, results = projection_episode(mutation)
    assert not all(r["ok"] for r in results)


def test_axis_swap_and_missing_points_are_not_accepted():
    reference, _ = projection_episode()
    expected = reference.expected_xyz()
    swapped = expected[:, [1, 0, 2]]
    assert not compare_points(reference, swapped)["passed"]
    with pytest.raises(ValueError):
        compare_points(reference, expected[:-1])


def test_reference_immutable_and_scene_visual_digest_changes():
    board, rgb, _, _ = raster()
    r = reference_from_rgb(rgb, board)
    assert isinstance(r.corners_uv, tuple) and isinstance(r.corners_uv[0], tuple)
    assert replace(board, origin_xyz_m=(0.081, 0.01, 0.002)).sha256 != board.sha256
    assert digest(visual_quads(replace(board, square_m=0.04))) != digest(
        visual_quads(board)
    )


def test_full_receipts_are_bound_to_original_capture_and_remain_unchanged():
    reference, results, obs = projection_episode(with_capture=True)
    original = copy.deepcopy(results)
    report = compare_annotations(reference, obs, results)
    assert report["passed"] and report["new_capture_admission"] is False
    assert report["capture_binding"]["capture_sha256"] == obs.sha256
    assert results == original


@pytest.mark.parametrize(
    "bad",
    ["rgb", "epoch", "calibration", "model", "age", "digest", "duplicate", "count"],
)
def test_mixed_or_mutated_receipt_cannot_earn_reference_credit(bad):
    reference, results, obs = projection_episode(with_capture=True)
    if bad == "rgb":
        obs = replace(
            obs, payload=replace(obs.payload, rgb8=b"\x00" * len(obs.payload.rgb8))
        )
    elif bad == "age":
        results[0]["capture_age_s"] = 2.01
    elif bad == "count":
        results.pop()
    else:
        entry = results[0]["result"]
        if bad == "epoch":
            entry["provenance"]["epoch"] = "foreign"
        elif bad == "calibration":
            entry["provenance"]["calibration_sha256"] = "f" * 64
        elif bad == "model":
            entry["provenance"]["model_identity_sha256"] = "f" * 64
        elif bad == "duplicate":
            results[1] = copy.deepcopy(results[0])
        elif bad == "digest":
            entry["point_map_m"][0] += 0.01
        if bad != "digest":
            entry["sha256"] = digest({k: v for k, v in entry.items() if k != "sha256"})
    with pytest.raises(ValueError):
        compare_annotations(reference, obs, results)


def test_real_usd_visual_fixture_has_no_collision_or_body_schema():
    pxr = pytest.importorskip(
        "pxr",
        reason="OpenUSD unavailable; CPU SDK authoring checked separately when available",
    )
    from pxr import Usd, UsdGeom

    assert pxr is not None
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    authored = author_visual_board(stage)
    assert authored["board_sha256"] == Board().sha256
    assert all(
        not any(
            "Physics" in schema or "Collision" in schema
            for schema in p.GetAppliedSchemas()
        )
        for p in stage.Traverse()
    )
    assert sum(p.IsA(UsdGeom.Mesh) for p in stage.Traverse()) == 53
    with pytest.raises(ValueError):
        author_visual_board(stage)
