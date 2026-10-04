"""Synthetic evidence composition only; no simulator or physical admission."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from cascade.eval.contained_placement import verify_contained_placement_window
from cascade.eval.placement import seal_placement_row
from test_box_geometry import fixture


def window():
    inventory, policy, _, base = fixture()
    policy = replace(policy, solver_dt_s=.01, rest_s=.04)
    inventory = replace(inventory, placement_policy_sha256=policy.sha256)
    cavity = inventory.cavity(interior_point_m=(0., 0., .5), top_z_m=1.)
    base.update(policy_sha256=policy.sha256, coupling_admission=policy.constraint_recipe,
                coverage='all_native_contact_candidates', native_warnings=[], external_forces_zero=True,
                ncon=1, nefc=3,
                contacts=[{'index': 0, 'geom_a': policy.support_geoms[0], 'geom_b': policy.object_geoms[0],
                           'efc_address': 0, 'dimension': 3, 'distance_m': 0., 'position_m': [0., 0., 0.],
                           'frame_rows': [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
                           'wrench_on_b_contact': [.981, 0., 0., 0., 0., 0.],
                           'force_on_b_world_n': [0., 0., .981]}])
    base['box_geometry']['inventory_sha256'] = inventory.sha256
    for body in base['bodies'].values():
        body.update(linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.])
    for geom in base['geometries']:
        geom['bound_radius_m'] = 1.
        if geom['id'] in policy.robot_geoms:
            geom['position_m'] = [10., 10., 10.]
    rows = []
    for step in range(1, 6):
        row = deepcopy(base)
        row.update(solver_step=step, constraint_time_s=(step-1)*.01, advanced_time_s=step*.01)
        rows.append(row)
    return rows, policy, inventory, cavity


def verify(rows, policy, inventory, cavity, error=.001):
    return verify_contained_placement_window([seal_placement_row(r) for r in rows], policy, inventory, cavity,
        first_solver_step=1, last_solver_step=5, support_frame_error_m=error)


def test_same_complete_window_joins_independent_checks_without_task_or_physical_credit():
    args = window()
    before = deepcopy(args[0])
    result = verify(*args)
    assert result['status'] == 'confirmed', result
    assert set(result['checks'].values()) == {'confirmed'}
    assert result['geometry_samples'] == 5 and result['geometry_components_per_sample'] == 1
    assert result['support']['measured']['window_sim_s'] == .04
    assert result['support_frame_error_m'] == .001
    assert not result['physical_admission'] and result['task_success'] == 'unverified'
    assert result['benchmark_success'] is None and args[0] == before


@pytest.mark.parametrize('fault', ['early_force', 'late_force', 'robot_contact', 'motion'])
def test_containment_does_not_override_failed_release_or_rest(fault):
    rows, policy, inventory, cavity = window()
    row = rows[-1] if fault == 'late_force' else rows[0]
    if fault.endswith('force'):
        row.update(ncon=0, contacts=[])
    elif fault == 'robot_contact':
        row['contacts'][0]['geom_a'] = policy.robot_geoms[0]
    else:
        row['bodies']['object']['linear_velocity_m_s'][0] = .1
    result = verify(rows, policy, inventory, cavity)
    assert result['status'] == 'refuted'
    assert result['checks'] == {'released_supported_rest': 'refuted', 'whole_object_containment': 'confirmed'}


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'clock', 'geometry', 'epoch', 'other_cavity'])
def test_join_requires_full_source_bound_coverage(fault):
    rows, policy, inventory, cavity = window()
    if fault == 'missing': rows.pop(2)
    elif fault == 'duplicate': rows[2] = deepcopy(rows[1])
    elif fault == 'clock': rows[2]['constraint_time_s'] += .001
    elif fault == 'geometry': rows[2]['box_geometry']['rotations_world'].pop()
    elif fault == 'epoch': rows[2]['epoch'] = 'other'
    else: cavity = replace(cavity, geometry_source_sha256='b'*64)
    result = verify(rows, policy, inventory, cavity)
    assert result['status'] == 'unverified'
    assert not result['physical_admission']


@pytest.mark.parametrize('error', [None, .8])
def test_unknown_invalid_or_oversized_error_does_not_become_containment(error):
    result = verify(*window(), error=error)
    assert result['status'] == 'unverified'
    assert result['checks']['released_supported_rest'] == 'confirmed'
    assert result['checks']['whole_object_containment'] == 'unverified'
    assert result['support_frame_error_m'] == error


@pytest.mark.parametrize('error', [True, -.1, float('nan'), '0.001'])
def test_invalid_error_contract_is_refused_before_measurement(error):
    result = verify(*window(), error=error)
    assert result['status'] == 'unverified'
    assert set(result['checks'].values()) == {'unverified'}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('step', [True, float('nan'), float('inf'), 0, '1'])
def test_invalid_window_is_refused_with_serializable_diagnostic(step):
    rows, policy, inventory, cavity = window()
    result = verify_contained_placement_window([seal_placement_row(r) for r in rows],
        policy, inventory, cavity, first_solver_step=step, last_solver_step=5, support_frame_error_m=0.)
    assert result['status'] == 'unverified'
    json.dumps(result, allow_nan=False)


def test_single_early_protrusion_cannot_be_hidden_by_a_quiet_contained_suffix():
    rows, policy, inventory, cavity = window()
    # Translate the entire object, including independently recorded geometry.
    # Make the rest drift criterion permissive explicitly in this synthetic
    # fixture to isolate the geometry join, without changing production limits.
    policy = replace(policy, drift_m=2.)
    inventory = replace(inventory, placement_policy_sha256=policy.sha256)
    cavity = inventory.cavity(interior_point_m=(0., 0., .5), top_z_m=1.)
    for row in rows:
        row['policy_sha256'] = policy.sha256
        row['box_geometry']['inventory_sha256'] = inventory.sha256
    rows[0]['bodies']['object']['position_m'][0] += 1.5
    next(g for g in rows[0]['geometries'] if g['id'] == policy.object_geoms[0])['position_m'][0] += 1.5
    result = verify(rows, policy, inventory, cavity)
    assert result['checks']['released_supported_rest'] == 'confirmed'
    assert result['status'] == 'unverified'
    assert result['geometry_samples'] == 5


def test_changed_seal_and_suffix_selection_are_refused():
    rows, policy, inventory, cavity = window()
    sealed = [seal_placement_row(row) for row in rows]
    sealed[0]['contacts'] = []
    result = verify_contained_placement_window(sealed, policy, inventory, cavity,
        first_solver_step=1, last_solver_step=5, support_frame_error_m=0.)
    assert result['status'] == 'unverified'
    result = verify_contained_placement_window(sealed[1:], policy, inventory, cavity,
        first_solver_step=2, last_solver_step=5, support_frame_error_m=0.)
    assert result['status'] == 'unverified'  # shorter than the declared rest


def test_callers_cannot_replace_geometry_between_the_two_predicates(monkeypatch):
    import cascade.eval.contained_placement as subject
    rows, policy, inventory, cavity = window()
    sealed = [seal_placement_row(row) for row in rows]
    expected = subject.verify_contained_placement_window(sealed, policy, inventory, cavity,
        first_solver_step=1, last_solver_step=5, support_frame_error_m=.001)
    original = subject.verify_placement_window

    def mutate_caller_after_support(private_rows, *args, **kwargs):
        assert private_rows is not sealed and private_rows[0] is not sealed[0]
        result = original(private_rows, *args, **kwargs)
        for row in sealed:
            row['box_geometry']['rotations_world'] = []
        return result

    monkeypatch.setattr(subject, 'verify_placement_window', mutate_caller_after_support)
    actual = subject.verify_contained_placement_window(sealed, policy, inventory, cavity,
        first_solver_step=1, last_solver_step=5, support_frame_error_m=.001)
    assert actual == expected and actual['status'] == 'confirmed'


def test_every_component_is_required_even_a_thin_protruding_secondary_box():
    rows, policy, inventory, cavity = window()
    policy = replace(policy, object_geoms=(10, 11))
    local = np.eye(4); local[0, 3] = 1.
    thin = replace(inventory.object_boxes[0], geometry_id=11,
                   body_from_geometry=tuple(local.flat), half_size_m=(.001, .001, .001))
    inventory = replace(inventory, placement_policy_sha256=policy.sha256,
                        object_boxes=(*inventory.object_boxes, thin))
    cavity = inventory.cavity(interior_point_m=(0., 0., .5), top_z_m=1.)
    for row in rows:
        row['policy_sha256'] = policy.sha256
        row['box_geometry']['inventory_sha256'] = inventory.sha256
        rotation = np.asarray(row['box_geometry']['body_rotations_world']['object']).reshape(3, 3)
        center = rotation @ local[:3, 3] + row['bodies']['object']['position_m']
        row['geometries'].append({'id': 11, 'position_m': center.tolist(), 'bound_radius_m': .002})
        row['box_geometry']['rotations_world'].append({'id': 11, 'rotation': rotation.ravel().tolist()})
    result = verify(rows, policy, inventory, cavity)
    assert result['checks']['released_supported_rest'] == 'confirmed'
    assert result['geometry_components_per_sample'] == 2
    assert result['status'] == 'unverified'
