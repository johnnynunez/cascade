"""Geometry checks only: no simulator or physical admission."""
from dataclasses import replace
import itertools

import numpy as np
import pytest

from cascade.eval.cavity import BoxCavity, BoxWall


def wall(ident, center, size):
    pose = np.eye(4)
    pose[:3, 3] = center
    return BoxWall(ident, tuple(pose.flat), size)


def cavity():
    return BoxCavity('a'*64, 'b'*64, 7, (
        wall(0, (0, 0, -.01), (1.03, 1.03, .01)),
        wall(1, (-1.01, 0, .5), (.01, 1.03, .51)),
        wall(2, (1.01, 0, .5), (.01, 1.03, .51)),
        wall(3, (0, -1.01, .5), (1.03, .01, .51)),
        wall(4, (0, 1.01, .5), (1.03, .01, .51))), (0, 0, .5), 1.)


def cube(center=(0., 0., .5), radius=.1):
    return np.asarray(list(itertools.product((-radius, radius), repeat=3)))+center


def test_all_finite_faces_cover_the_calibrated_convex_volume():
    value = cavity()
    planes, vertices = value.planes_and_vertices()
    assert len(planes) == 6 and len(vertices) == 8
    assert set(vertices) == set(itertools.product((-1., 1.), (-1., 1.), (0., 1.)))
    verdict = value.contains_hulls({10: cube()}, expected_geometry_ids=(10,), position_error_m=.001)
    assert verdict['status'] == 'confirmed' and not verdict['physical_admission']
    assert verdict['minimum_margin_m'] == pytest.approx(.399-1e-9)


@pytest.mark.parametrize('fault', ['short_wall', 'short_floor', 'missing_wall', 'duplicate', 'high_cap', 'bad_point'])
def test_infinite_plane_or_visual_box_is_not_a_complete_physical_cavity(fault):
    value = cavity()
    walls = list(value.walls)
    if fault == 'short_wall': walls[1] = replace(walls[1], half_size_m=(.01, .2, .51))
    if fault == 'short_floor': walls[0] = replace(walls[0], half_size_m=(.2, 1.03, .01))
    if fault == 'missing_wall': walls.pop()
    if fault == 'duplicate': walls[4] = walls[3]
    kwargs = {'walls': tuple(walls)}
    if fault == 'high_cap': kwargs['top_z_m'] = 2.
    if fault == 'bad_point': kwargs['interior_point_m'] = (2, 0, .5)
    with pytest.raises(ValueError): replace(value, **kwargs)


def test_rotated_wall_axes_do_not_change_the_same_solid_or_cavity():
    value = cavity()
    original = value.walls[0]
    pose = np.asarray(original.body_from_geometry).reshape(4, 4).copy()
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    changed = replace(original, body_from_geometry=tuple(pose.flat), half_size_m=(.01, 1.03, 1.03))
    equivalent = replace(value, walls=(changed, *value.walls[1:]))
    assert equivalent.planes_and_vertices() == value.planes_and_vertices()
    assert equivalent.sha256 != value.sha256


@pytest.mark.parametrize('fault', ['protrusion', 'above_cap', 'below_floor', 'uncertainty', 'unknown_error',
                                 'missing_collider', 'robot_as_object', 'nonfinite', 'flat', 'empty'])
def test_complete_geometry_and_error_must_fit_not_just_the_object_center(fault):
    points, ids, error = cube(), (10,), 0.
    hulls = {10: points}
    if fault == 'protrusion': hulls[10] = cube(radius=1.1)
    if fault == 'above_cap': hulls[10] = cube(center=(0, 0, 1))
    if fault == 'below_floor': hulls[10] = cube(center=(0, 0, 0))
    if fault == 'uncertainty': error = .5
    if fault == 'unknown_error': error = None
    if fault == 'missing_collider': ids = (10, 11)
    if fault == 'robot_as_object': ids, hulls = (0,), {0: points}
    if fault == 'nonfinite': points[0, 0] = np.nan
    if fault == 'flat': points[:, 2] = .5
    if fault == 'empty': hulls, ids = {}, ()
    result = cavity().contains_hulls(hulls, expected_geometry_ids=ids, position_error_m=error)
    assert result['status'] == 'unverified' and not result['physical_admission']


def test_thin_protruding_component_cannot_be_hidden_by_an_interior_main_body():
    value = cavity()
    main, protruding = cube(), cube(center=(1.05, 0, .5), radius=.02)
    assert value.contains_hulls({10: main}, expected_geometry_ids=(10,), position_error_m=0)['status'] == 'confirmed'
    assert value.contains_hulls({10: main, 11: protruding}, expected_geometry_ids=(10, 11), position_error_m=0)['status'] == 'unverified'


@pytest.mark.parametrize('short_index,short_size,outside', [
    (1, (.01, .2, .51), (0., .5, .5)),
    (0, (.2, 1.03, .01), (.5, 0., .5)),
])
def test_finite_face_intersection_only_certifies_the_smaller_region(short_index, short_size, outside):
    value = cavity()
    walls = list(value.walls)
    walls[short_index] = replace(walls[short_index], half_size_m=short_size)
    # The original whole-face calibration remains refused for this geometry.
    with pytest.raises(ValueError, match='finite wall'):
        replace(value, walls=tuple(walls))
    clipped = replace(value, walls=tuple(walls), calibration_recipe='finite_face_intersection')
    planes, vertices = clipped.planes_and_vertices()
    assert len(planes) == 26
    for physical_wall in clipped.walls:
        transform = np.asarray(physical_wall.body_from_geometry).reshape(4, 4)
        local = (np.asarray(vertices)-transform[:3, 3]) @ np.linalg.inv(transform[:3, :3]).T
        thin_axis = int(np.argmin(physical_wall.half_size_m))
        for axis in range(3):
            if axis != thin_axis:
                assert np.max(abs(local[:, axis])) < physical_wall.half_size_m[axis]
    assert clipped.contains_hulls({10: cube()}, expected_geometry_ids=(10,), position_error_m=0)['status'] == 'confirmed'
    # This hull fits all six infinite interior planes but not the finite wall.
    assert clipped.contains_hulls({10: cube(outside)}, expected_geometry_ids=(10,), position_error_m=0)['status'] == 'unverified'


def test_high_visual_cap_cannot_extend_the_finite_wall_intersection():
    value = replace(cavity(), top_z_m=2., calibration_recipe='finite_face_intersection')
    assert max(vertex[2] for vertex in value.planes_and_vertices()[1]) < 1.01
    assert value.contains_hulls({10: cube((0, 0, 1.5))}, expected_geometry_ids=(10,), position_error_m=0)['status'] == 'unverified'


def test_recipe_is_explicit_hashed_and_unknown_recipe_is_refused():
    value = cavity()
    assert replace(value, calibration_recipe='finite_face_intersection').sha256 != value.sha256
    with pytest.raises(ValueError, match='recipe'):
        replace(value, calibration_recipe='visual_bounds')


def test_auxiliary_planes_keep_metric_error_under_permitted_pose_encoding_residual():
    value = cavity()
    walls = list(value.walls)
    pose = np.asarray(walls[1].body_from_geometry).reshape(4, 4).copy()
    pose[1, 1] += 1e-8
    walls[1] = replace(walls[1], body_from_geometry=tuple(pose.flat))
    clipped = replace(value, walls=tuple(walls), calibration_recipe='finite_face_intersection')
    planes, _ = clipped.planes_and_vertices()
    assert np.max(abs(np.linalg.norm(np.asarray(planes)[:, :3], axis=1)-1)) < 1e-14
    assert clipped.contains_hulls({10: cube()}, expected_geometry_ids=(10,), position_error_m=.5)['status'] == 'unverified'
