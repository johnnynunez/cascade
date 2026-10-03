"""Pure geometric capacity, not physical placement admission."""
import numpy as np
import pytest

from cascade.skills.placement_packing import pack_rows


def box(identity, width=.035, *, center=(0., 0.), fixed=False):
    center = np.asarray(center)
    return {'id': identity, 'lower': center-width/2, 'upper': center+width/2, 'fixed': fixed}


def test_original_area_cannot_fit_two_with_declared_margin():
    assert pack_rows([[.15, -.16], [.24, -.08]], [box(2), box(1)], margin=.02, separation=.02) == []


def test_interior_layout_reserves_whole_inventory_without_body_labels():
    layouts = pack_rows([[.13, -.18], [.27, -.06]], [box(9), box(10)], margin=.02, separation=.02)
    assert layouts
    for row in layouts:
        assert set(row['footprints']) == {9, 10}
        assert row['minimum_planned_border_m'] >= .02
        a, b = [np.asarray(x) for x in row['footprints'].values()]
        assert max(*(a[0]-b[1]), *(b[0]-a[1])) >= .02-1e-12
    assert layouts[0]['minimum_planned_border_m'] == max(r['minimum_planned_border_m'] for r in layouts)


def test_fixed_observed_object_is_not_silently_repacked():
    fixed = box(9, center=(.172, -.12), fixed=True)
    layouts = pack_rows([[.13, -.18], [.27, -.06]], [fixed, box(10)], margin=.02, separation=.02)
    assert layouts
    for row in layouts:
        np.testing.assert_array_equal(row['footprints'][9], [fixed['lower'], fixed['upper']])


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('edge', ['lower', 'upper'])
@pytest.mark.parametrize('width', [.7, .5])
def test_fixed_border_object_does_not_expand_planning_interior(axis, edge, width):
    # A fixed body need only satisfy exterior containment. Its distance from
    # the boundary cannot replace the larger margin of a future placement.
    lo, hi = np.array([0., .4]), np.array([.05, .5])
    if edge == 'upper':
        lo[0], hi[0] = .95, 1.
    free_hi = np.array([width, .1])
    if axis:
        lo, hi, free_hi = lo[::-1], hi[::-1], free_hi[::-1]
    fixed = {'id': 1, 'lower': lo, 'upper': hi, 'fixed': True}
    free = {'id': 2, 'lower': [0., 0.], 'upper': free_hi, 'fixed': False}
    layouts = pack_rows([[0., 0.], [1., 1.]], [fixed, free], margin=.2, separation=.01)
    if width == .7:
        assert layouts == []  # The available interior is only .6 m wide.
    else:
        assert layouts
        for row in layouts:
            footprint = np.asarray(row['footprints'][2])
            assert (footprint[0] >= .2).all()
            assert (footprint[1] <= .8).all()
            np.testing.assert_array_equal(row['footprints'][1], [lo, hi])


def test_bigger_geometry_or_third_object_can_refuse_capacity():
    assert not pack_rows([[.13, -.18], [.27, -.06]], [box(1, .06), box(2, .06)], margin=.02, separation=.02)
    assert not pack_rows([[.13, -.18], [.27, -.06]], [box(1), box(2), box(3)], margin=.02, separation=.02)


@pytest.mark.parametrize('case', ['nan', 'duplicate', 'too_many', 'outside', 'overlap'])
def test_unknown_or_conflicting_inventory_has_no_capacity(case):
    items = [box(1), box(2)]
    if case == 'nan': items[0]['lower'][0] = np.nan
    if case == 'duplicate': items[1]['id'] = 1
    if case == 'too_many': items = [box(i) for i in range(7)]
    if case == 'outside': items[0] = box(1, center=(.5, -.12), fixed=True)
    if case == 'overlap': items = [box(i, center=(.2, -.12), fixed=True) for i in (1, 2)]
    if case in ('nan', 'duplicate', 'too_many'):
        with pytest.raises(ValueError):
            pack_rows([[.13, -.18], [.27, -.06]], items, margin=.02, separation=.02)
    else:
        assert not pack_rows([[.13, -.18], [.27, -.06]], items, margin=.02, separation=.02)
