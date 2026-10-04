from dataclasses import replace
import json
import math

import numpy as np
import pytest

from cascade.perception.rgb_triangulation import RgbView, TriangulationPolicy, match_rgb_features, triangulate_pairs


def view(name='left', x=0., offset=(.5, .5)):
    transform = np.eye(4); transform[0, 3] = x
    return RgbView(name, 'world', 'a'*64, 'b'*64, 'epoch', 20, .04, 128, 128,
        (100., 0., 64., 0., 100., 64., 0., 0., 1.), tuple(map(float, transform.flat)), offset, bytes(128*128*3))


def pixel(camera, point):
    transform = np.asarray(camera.world_from_camera).reshape(4, 4)
    local = np.linalg.solve(transform, np.r_[point, 1.])[:3]
    uvw = np.asarray(camera.intrinsics).reshape(3, 3) @ local
    return tuple(map(float, uvw[:2]/uvw[2] - camera.pixel_center_offset_uv))


@pytest.mark.parametrize('offset', [(0., 0.), (.5, .5)])
def test_actual_two_view_projection_recovers_metric_points_with_explicit_pixel_convention(offset):
    left, right = view(offset=offset), view('right', .1, offset)
    points = [[-.05, .03, .4], [.12, -.08, .7], [.1, .2, 1.4]]
    pairs = [(*pixel(left, point), *pixel(right, point)) for point in points]
    result = triangulate_pairs(left, right, pairs)
    assert result['estimated_points'] == 3
    for row, expected in zip(result['rows'], points):
        np.testing.assert_allclose(row['position_world_m'], expected, atol=1e-12)
        assert max(row['reprojection_px']) < 1e-11
        assert row['position_error_m'] is None
    assert not result['physical_admission'] and not result['object_identity_verified']
    assert not result['complete_geometry_verified'] and result['position_error_m'] is None
    json.dumps(result, allow_nan=False)


def test_rotated_wrist_camera_uses_its_per_capture_pose():
    left, right = view(), view('wrist', .1)
    c, s = math.cos(.2), math.sin(.2)
    transform = np.asarray(right.world_from_camera).reshape(4, 4).copy()
    transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    right = replace(right, world_from_camera=tuple(map(float, transform.flat)))
    point = [.05, .1, .8]
    pair = (*pixel(left, point), *pixel(right, point))
    result = triangulate_pairs(left, right, [pair])
    np.testing.assert_allclose(result['rows'][0]['position_world_m'], point, atol=1e-12)
    # A wrong but rigid camera pose can still explain both pixels. Small
    # reprojection error does not establish calibration accuracy.
    wrong = triangulate_pairs(left, view('wrist', .1), [pair])['rows'][0]
    assert wrong['status'] == 'estimated' and wrong['position_error_m'] is None
    assert np.linalg.norm(np.asarray(wrong['position_world_m']) - point) > .1


@pytest.mark.parametrize('change', [dict(epoch='new'), dict(solver_step=21), dict(simulation_time_s=.041),
    dict(model_identity_sha256='c'*64), dict(world_frame_id='other'), dict(camera_id='left')])
def test_no_mix_of_models_epochs_steps_times_or_frames(change):
    with pytest.raises(ValueError, match='simultaneous'):
        triangulate_pairs(view(), replace(view('right', .1), **change), [])


@pytest.mark.parametrize('pair,reason', [
    ((64., 64., 64., 64.), 'parallax'),
    ((64., 64., 74., 64.), 'behind'),
    ((64., 64., 54., 100.), 'reprojection'),
    ((-1., 64., 54., 64.), 'outside'),
])
def test_bad_correspondences_are_retained_in_order(pair, reason):
    good = (63.5, 63.5, 43.5, 63.5)
    result = triangulate_pairs(view(), view('right', .1), [pair, good])
    assert result['estimated_points'] == 1
    assert result['rows'][0]['status'] == 'unverified' and reason in result['rows'][0]['reason']
    assert result['rows'][0]['pixels'] == list(pair)
    assert result['rows'][1]['index'] == 1 and result['rows'][1]['status'] == 'estimated'


def test_no_metric_estimate_from_zero_baseline_or_far_points():
    assert triangulate_pairs(view(), view('right'), [(64., 64., 60., 64.)])['estimated_points'] == 0
    left, right = view(), view('right', .1)
    point = [0, 0, 3.]
    result = triangulate_pairs(left, right, [(*pixel(left, point), *pixel(right, point))])
    assert 'depth interval' in result['rows'][0]['reason']


@pytest.mark.parametrize('change', [dict(width=True), dict(solver_step=True), dict(simulation_time_s=float('nan')),
    dict(rgb=bytearray(128*128*3)), dict(intrinsics=(1.,)), dict(world_from_camera=(1.,)),
    dict(pixel_center_offset_uv=(.1, .1)), dict(model_identity_sha256='bad')])
def test_malformed_capture_is_rejected(change):
    with pytest.raises(ValueError):
        replace(view(), **change)


def test_calibration_is_immutable_and_capture_hash_binds_every_rgb_byte():
    original = view()
    matrix = list(original.world_from_camera)
    copy = replace(original, world_from_camera=matrix)
    matrix[3] = .1
    assert copy.sha256 == original.sha256
    assert replace(copy, rgb=b'\x01'+copy.rgb[1:]).sha256 != original.sha256
    mirrored = np.eye(4); mirrored[0, 0] = -1
    with pytest.raises(ValueError, match='rigid'):
        replace(copy, world_from_camera=tuple(map(float, mirrored.flat)))


def test_pair_and_policy_bounds_are_checked_before_math():
    left, right = view(), view('right', .1)
    for pairs in ([(float('nan'), 1., 1., 1.)], [(1.,)], [(True, 1., 1., 1.)], [(1.,)*4]*513):
        with pytest.raises(ValueError):
            triangulate_pairs(left, right, pairs)
    for settings in (dict(max_pairs=True), dict(max_depth_m=101.), dict(min_parallax_rad=0.),
                     dict(max_reprojection_px=float('inf'))):
        with pytest.raises(ValueError):
            TriangulationPolicy(**settings)


def test_degenerate_finite_intrinsics_do_not_emit_nonfinite_estimates():
    left = replace(view(), intrinsics=(1e-320, 0., 64., 0., 100., 64., 0., 0., 1.))
    result = triangulate_pairs(left, view('right', .1), [(32., 64., 22., 64.)])
    assert result['estimated_points'] == 0
    json.dumps(result, allow_nan=False)


def test_real_rgb_feature_matching_uses_image_coordinates_and_never_fills_depth():
    pytest.importorskip('cv2')
    rng = np.random.default_rng(31)
    rgb = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    other = np.zeros_like(rgb); other[:, :-8] = rgb[:, 8:]
    left = replace(view(), rgb=rgb.tobytes())
    right = replace(view('right', .1), rgb=other.tobytes())
    pairs = match_rgb_features(left, right, max_features=256)
    assert 5 <= len(pairs) <= 256
    result = triangulate_pairs(left, right, pairs)
    assert result['estimated_points'] >= 5
    estimated = [row for row in result['rows'] if row['status'] == 'estimated']
    assert np.median([row['position_world_m'][2] for row in estimated]) == pytest.approx(1.25, abs=.01)
    assert match_rgb_features(view(), view('right', .1)) == []
