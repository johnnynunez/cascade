"""Host-buffer contract tests; these do not claim native physical validation."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.sim.microduck_contact_support import (
    FOOT_SHAPES, GROUND_SHAPE, read_support, solved_contacts, support_contract,
)


class Buffer:
    def __init__(self, value, dtype=np.float32):
        self.value = np.asarray(value, dtype=dtype)

    def numpy(self):
        return self.value.copy()


def fixture(*, cone=0, dim=3):
    # Ground is A, left sole B. The right-handed frame maps +X contact
    # normal to +Z world; native output is the downward force on ground.
    frame = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    nrows = 2 * (dim - 1) if cone == 0 and dim > 1 else dim
    forces = np.zeros((1, 12), np.float32)
    forces[0, :nrows] = 1. if cone == 0 and dim > 1 else 0.
    if cone == 1 or dim == 1:
        forces[0, 0] = 4.
    normal_force = float(forces[0, :nrows].sum()) if cone == 0 and dim > 1 else 4.
    types = np.zeros((1, 12), np.int32)
    types[0, :nrows] = 5 if dim == 1 else (6 if cone == 0 else 7)
    adr = np.full((4, 10), -1, np.int32)
    adr[0, :nrows] = np.arange(nrows)
    c = NS(worldid=Buffer([0] * 4, np.int32), geom=Buffer([[0, 1]] * 4, np.int32),
           dim=Buffer([dim] * 4, np.int32), pos=Buffer([[.1, .2, 0.]] * 4),
           frame=Buffer([frame] * 4), efc_address=Buffer(adr, np.int32))
    d = NS(njmax=12, naconmax=4, nacon=Buffer([1], np.int32), nefc=Buffer([nrows], np.int32),
           contact=c, efc=NS(type=Buffer(types, np.int32), id=Buffer(np.zeros((1, 12)), np.int32),
                           force=Buffer(forces)))
    contacts = NS(rigid_contact_count=Buffer([1], np.int32),
                  rigid_contact_shape0=Buffer([0] * 4, np.int32),
                  rigid_contact_shape1=Buffer([1] * 4, np.int32),
                  rigid_contact_normal=Buffer([[0., 0., 1.]] * 4),
                  force=Buffer([[0., 0., -normal_force, 0., 0., 0.]] * 4))
    solver = NS(mjw_data=d, mjw_model=NS(opt=NS(cone=cone)),
                mjc_geom_to_newton_shape=Buffer([[0, 1, 2, 3]], np.int32))
    return NS(model=NS(shape_label=[GROUND_SHAPE, *FOOT_SHAPES, '/World/MicroDuck/trunk_collision']),
              solver=solver, contacts=contacts, simulation_step_count=20, sim_time=.1)


def contact_population(count, *, cone=0, dim=3, inactive=()):
    """Independent completed-contact layouts, with unused capacity retained."""
    ns = fixture(cone=cone, dim=dim)
    d, c = ns.solver.mjw_data, ns.contacts
    width = 2 * (dim - 1) if cone == 0 and dim > 1 else dim
    active = [i for i in range(count) if i not in inactive]
    capacity, nefc = count + 1, len(active) * width
    for channel in (d.contact.worldid, d.contact.geom, d.contact.dim, d.contact.pos,
                    d.contact.frame, c.rigid_contact_shape0, c.rigid_contact_shape1,
                    c.rigid_contact_normal, c.force):
        channel.value = np.repeat(channel.value[:1], capacity, axis=0)
    d.nacon.value[0] = c.rigid_contact_count.value[0] = count
    d.naconmax, d.njmax = capacity, nefc + 1
    d.nefc.value[0] = nefc
    d.contact.efc_address.value = np.full((capacity, 10), -1, np.int32)
    row_type = 5 if dim == 1 else (6 if cone == 0 else 7)
    d.efc.type.value = np.full((1, nefc + 1), row_type, np.int32)
    d.efc.id.value = np.zeros((1, nefc + 1), np.int32)
    d.efc.force.value = np.zeros((1, nefc + 1), np.float32)
    for ordinal, i in enumerate(active):
        rows = np.arange(ordinal * width, (ordinal + 1) * width)
        d.contact.efc_address.value[i, :width] = rows
        d.efc.id.value[0, rows] = i
        if cone == 0 and dim > 1:
            d.efc.force.value[0, rows] = 1.
        else:
            d.efc.force.value[0, rows[0]] = 4.
        d.contact.pos.value[i, 0] = i / 100.
    # Neither inactive candidates nor unused storage supplies physical data.
    for i in (*inactive, count):
        d.contact.pos.value[i] = d.contact.frame.value[i] = np.nan
        c.rigid_contact_normal.value[i] = c.force.value[i] = np.nan
    return ns


@pytest.mark.parametrize('cone,dim', [(0, 1), (0, 3), (0, 4), (0, 6), (1, 1), (1, 3), (1, 4), (1, 6)])
def test_many_contact_frames_retain_order_force_and_inactive_slots(cone, dim):
    ns = contact_population(32, cone=cone, dim=dim, inactive=(1, 6, 25))
    rows = solved_contacts(ns)
    active = [i for i in range(32) if i not in (1, 6, 25)]
    assert [r['point_world_m'][0] for r in rows] == [float(np.float32(i / 100.)) for i in active]
    force = 2 * (dim - 1) if cone == 0 and dim > 1 else 4.
    assert all(r['normal_force_n'] == force and r['force_on_b_world_n'] == [0., 0., force]
               and r['normal_a_to_b_world'] == [0., 0., 1.] for r in rows)


@pytest.mark.parametrize('index', [0, 3, 31])
@pytest.mark.parametrize('channel', ['frame', 'point', 'force', 'normal', 'handedness'])
def test_invalid_contact_in_a_batch_cannot_be_hidden_by_valid_neighbors(index, channel):
    ns = contact_population(32)
    d, c = ns.solver.mjw_data, ns.contacts
    if channel == 'frame':
        d.contact.frame.value[index, 1, 0] = np.nan
    elif channel == 'point':
        d.contact.pos.value[index, 0] = np.nan
    elif channel == 'force':
        c.force.value[index, 5] = np.inf
    elif channel == 'normal':
        c.rigid_contact_normal.value[index, 0] = 0.01
    else:
        d.contact.frame.value[index, 1] *= -1
    with pytest.raises(ValueError, match='invalid contact frame/normal'):
        solved_contacts(ns)


@pytest.mark.parametrize('perturbation,accepted', [(8e-6, True), (12e-6, False)])
def test_contact_batch_preserves_original_orthogonality_tolerance(perturbation, accepted):
    ns = contact_population(32)
    ns.solver.mjw_data.contact.frame.value[17, 1, 0] += perturbation
    if accepted:
        assert len(solved_contacts(ns)) == 32
    else:
        with pytest.raises(ValueError, match='invalid contact frame/normal'):
            solved_contacts(ns)


def test_batch_arithmetic_preserves_candidate_failure_order():
    ns = contact_population(3)
    ns.solver.mjw_data.contact.frame.value[0, 0, 0] = 1.
    ns.contacts.rigid_contact_shape1.value[1] = 999
    with pytest.raises(ValueError, match='invalid contact frame/normal'):
        solved_contacts(ns)


@pytest.mark.parametrize('inactive', [(1,), (0, 1, 2)])
def test_inactive_signaling_nan_frame_is_never_converted(inactive):
    ns = contact_population(3, inactive=inactive)
    # Write the IEEE-754 bits directly; assigning a Python float would quiet it.
    ns.solver.mjw_data.contact.frame.value.view(np.uint32)[1, 0, 0] = 0x7f800001
    with np.errstate(invalid='raise'):
        assert len(solved_contacts(ns)) == 3 - len(inactive)


@pytest.mark.parametrize('earlier', ['pair', 'frame'])
def test_later_signaling_nan_does_not_preempt_first_candidate_error(earlier):
    ns = contact_population(8)
    d, c = ns.solver.mjw_data, ns.contacts
    d.contact.frame.value.view(np.uint32)[7, 0, 0] = 0x7f800001
    if earlier == 'pair':
        c.rigid_contact_shape1.value[0] = 999
        message = 'contact geom/shape identity mismatch'
    else:
        d.contact.frame.value[0, 0, 0] = 1.
        message = 'invalid contact frame/normal'
    with np.errstate(invalid='raise'), pytest.raises(ValueError, match=message):
        solved_contacts(ns)


def test_active_signaling_nan_retains_original_strict_fp_failure():
    ns = contact_population(2)
    ns.solver.mjw_data.contact.frame.value.view(np.uint32)[0, 0, 0] = 0x7f800001
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError):
        solved_contacts(ns)


def read(ns):
    return read_support(ns, last_solved_clock=(20, .1), source_admitted=True)


@pytest.mark.parametrize('cone,dim', [(0, 1), (0, 3), (0, 4), (0, 6), (1, 1), (1, 3), (1, 4), (1, 6)])
def test_solved_native_force_normal_and_all_supported_row_layouts(cone, dim):
    ns = fixture(cone=cone, dim=dim)
    value = read(ns)
    assert value['status'] == 'known', value['reason']
    row, = value['contacts']
    assert row['shape_a'] == GROUND_SHAPE and row['shape_b'] == FOOT_SHAPES[0]
    assert row['shape_a_id'] == 0 and row['shape_b_id'] == 1
    assert row['force_on_b_world_n'] == [0., 0., row['normal_force_n']]
    np.testing.assert_allclose(row['point_world_m'], [.1, .2, 0.])
    assert row['normal_a_to_b_world'] == [0., 0., 1.]
    assert value['step'] == 20 and value['sim_time_s'] == .1
    # Epoch/configuration are bound by controller; producer cannot restamp them.
    assert 'epoch' not in value and 'model_identity_sha256' not in value


def test_pair_reversal_preserves_the_same_physical_support_and_tangential_force():
    ns = fixture()
    ns.contacts.force.value[0, 0] = -1.25
    forward = solved_contacts(ns)[0]
    assert forward['force_on_b_world_n'] == [1.25, 0., 4.]
    ns.solver.mjw_data.contact.geom.value[0] = [1, 0]
    ns.contacts.rigid_contact_shape0.value[0] = 1
    ns.contacts.rigid_contact_shape1.value[0] = 0
    ns.contacts.rigid_contact_normal.value[0] *= -1
    ns.solver.mjw_data.contact.frame.value[0, [0, 2]] *= -1
    ns.contacts.force.value[0] *= -1
    reverse = solved_contacts(ns)[0]
    assert reverse['shape_a'] == forward['shape_b'] and reverse['shape_b'] == forward['shape_a']
    np.testing.assert_equal(reverse['force_on_b_world_n'], -np.array(forward['force_on_b_world_n']))
    assert reverse['normal_force_n'] == forward['normal_force_n']


def test_zero_force_active_row_is_recorded_without_inventing_support():
    ns = fixture()
    ns.contacts.force.value[:] = 0
    ns.solver.mjw_data.efc.force.value[:] = 0
    row, = solved_contacts(ns)
    assert row['normal_force_n'] == 0 and row['force_on_b_world_n'] == [0., 0., 0.]


def test_forbidden_body_contact_is_preserved_for_independent_checker():
    ns = fixture()
    ns.solver.mjw_data.contact.geom.value[0, 1] = 3
    ns.contacts.rigid_contact_shape1.value[0] = 3
    value = read(ns)
    assert value['status'] == 'known'
    assert value['contacts'][0]['shape_b'] == '/World/MicroDuck/trunk_collision'
    assert value['contacts'][0]['normal_force_n'] == 4.


@pytest.mark.parametrize('known', [True, False])
def test_raw_evidence_satisfies_wire_schema_after_controller_identity_binding(known):
    from cascade.control.mobile_support import SupportObservation
    value = read_support(fixture(), last_solved_clock=(20, .1), source_admitted=known)
    bound = SupportObservation.from_dict({**value, 'epoch': 'test-epoch', 'model_identity_sha256': 'a' * 64})
    assert bound.status == ('known' if known else 'unavailable')


def test_inactive_candidate_and_unused_storage_never_become_force_evidence():
    ns = fixture()
    d = ns.solver.mjw_data
    d.nacon.value[0] = ns.contacts.rigid_contact_count.value[0] = 2
    # Row1 is a separated candidate with efc_address=-1. Unused native
    # storage can be uninitialized, so its floats cannot reject row0.
    ns.contacts.force.value[1:] = np.nan
    d.contact.frame.value[1:] = np.nan
    assert len(solved_contacts(ns)) == 1
    d.contact.efc_address.value[:] = -1
    d.nefc.value[0] = 0
    assert read(ns)['status'] == 'known' and solved_contacts(ns) == []


@pytest.mark.parametrize('bad', [
    'missing_force', 'float64_force', 'nan_force', 'nan_point', 'bad_normal', 'bad_frame',
    'count', 'capacity', 'world', 'geom_oob', 'mapping', 'unknown_shape', 'duplicate_label',
    'negative_address', 'truncated_rows', 'row_id', 'row_type', 'uncovered_row',
    'normal_disagreement', 'negative_force', 'negative_pyramid_edge', 'invalid_dim', 'invalid_cone',
])
def test_incomplete_or_ambiguous_evidence_is_unknown_not_known_empty(bad):
    ns = fixture()
    d, c = ns.solver.mjw_data, ns.contacts
    if bad == 'missing_force': c.force = None
    elif bad == 'float64_force': c.force.value = c.force.value.astype(np.float64)
    elif bad == 'nan_force': c.force.value[0, 0] = np.nan
    elif bad == 'nan_point': d.contact.pos.value[0, 0] = np.nan
    elif bad == 'bad_normal': c.rigid_contact_normal.value[0] = [1., 0., 0.]
    elif bad == 'bad_frame': d.contact.frame.value[0, 0, 0] = 1.
    elif bad == 'count': c.rigid_contact_count.value[0] = 0
    elif bad == 'capacity': d.nacon.value[0] = 4
    elif bad == 'world': d.contact.worldid.value[0] = 1
    elif bad == 'geom_oob': d.contact.geom.value[0, 0] = 4
    elif bad == 'mapping': ns.solver.mjc_geom_to_newton_shape.value[0, 1] = 2
    elif bad == 'unknown_shape': c.rigid_contact_shape1.value[0] = 10
    elif bad == 'duplicate_label': ns.model.shape_label[2] = ns.model.shape_label[1]
    elif bad == 'negative_address': d.contact.efc_address.value[0, 0] = -2
    elif bad == 'truncated_rows': d.contact.efc_address.value[0, 3] = -1
    elif bad == 'row_id': d.efc.id.value[0, 1] = 1
    elif bad == 'row_type': d.efc.type.value[0, 1] = 1
    elif bad == 'uncovered_row':
        d.nefc.value[0] += 1
        d.efc.type.value[0, 4] = 5
    elif bad == 'normal_disagreement': c.force.value[0, 2] = -3.
    elif bad == 'negative_force':
        c.force.value[0, 2] = 4.
        d.efc.force.value[0, :4] = -1.
    elif bad == 'negative_pyramid_edge':
        d.efc.force.value[0, :4] = [-1., 3., 1., 1.]
    elif bad == 'invalid_dim': d.contact.dim.value[0] = 2
    elif bad == 'invalid_cone': ns.solver.mjw_model.opt.cone = 2
    value = read(ns)
    assert value['status'] == 'unavailable', bad
    assert value['reason'] and value['contacts'] == []


def test_clock_source_admission_and_read_race_are_never_restamped():
    ns = fixture()
    for clock, admitted in [(None, True), ((19, .095), True), ((20, .1), False)]:
        value = read_support(ns, last_solved_clock=clock, source_admitted=admitted)
        assert value['status'] == 'unavailable'
    original = ns.contacts.force.numpy
    def racing_read():
        ns.simulation_step_count += 1
        return original()
    ns.contacts.force.numpy = racing_read
    value = read(ns)
    assert value['status'] == 'unavailable' and value['step'] == 20
    assert 'advanced' in value['reason']


def test_native_arrays_and_clocks_are_not_mutated_by_reading():
    ns = fixture()
    before = deepcopy(ns)
    assert read(ns)['status'] == 'known'
    np.testing.assert_array_equal(ns.contacts.force.value, before.contacts.force.value)
    np.testing.assert_array_equal(ns.solver.mjw_data.contact.efc_address.value,
                                  before.solver.mjw_data.contact.efc_address.value)
    assert (ns.simulation_step_count, ns.sim_time) == (20, .1)


def test_contract_admits_exact_collision_shapes_not_ankle_body_or_visual():
    labels = [GROUND_SHAPE, *FOOT_SHAPES, '/World/MicroDuck/trunk_collision']
    contract = support_contract(labels)
    assert contract['foot_shapes'] == list(FOOT_SHAPES)
    assert set(contract['robot_shapes']) == set(labels[1:])
    for old in FOOT_SHAPES:
        for replacement in [old.rsplit('/', 1)[0], old + '_visual', '/fake/left_foot_collision']:
            with pytest.raises(ValueError):
                support_contract([replacement if x == old else x for x in labels])


def bound_backend():
    """Complete the uncaptured backend contract; retain its actual guards."""
    from cascade.sim.microduck_newton import KitNewtonBackend
    from cascade.sim.microduck_solver_graph import SolverGraphContract
    backend = KitNewtonBackend(None, {'limits': {'max_contacts': 4, 'max_constraints': 12}}, None)
    backend.ns = fixture()
    backend.ns.initialized = True
    backend.ns.cfg = NS(time_step_app=False, num_substeps=1, use_cuda_graph=False)
    backend.ns.graph = None
    backend.ns.model.joint_label = ['fixture_joint']
    backend.ns.model.body_label = ['fixture_body']
    backend._model = backend.ns.model
    backend._layout = tuple(tuple(getattr(backend.ns.model, key))
                            for key in ('joint_label', 'body_label', 'shape_label'))
    backend._dt = float(np.float32(.005))
    backend.SM = NS(get_physics_dt=lambda: .005, get_active_physics_engine=lambda: 'newton')
    backend._solver_graph = SolverGraphContract(backend.ns, enabled=False, wp=None,
                                                dt=backend._dt, source_path=None)
    return backend


@pytest.mark.parametrize('bad', [None, 'exception', 'no_solve', 'two_solves', 'wrong_time'])
def test_backend_only_marks_successful_single_solve_for_support(bad):
    backend = bound_backend()
    assert backend._last_support_solve is None
    def step(*, steps):
        assert steps == 1
        if bad == 'exception':
            raise RuntimeError('injected solver failure')
        if bad == 'no_solve':
            return
        backend.ns.simulation_step_count += 2 if bad == 'two_solves' else 1
        backend.ns.sim_time += .02 if bad == 'wrong_time' else backend._dt
    backend.SM.step = step
    if bad:
        backend._last_support_solve = (20, .1)
        with pytest.raises(RuntimeError):
            backend.step()
        assert backend._last_support_solve is None
    else:
        backend.step()
        assert backend._last_support_solve == (21, .1 + backend._dt)


def test_backend_read_delivers_support_from_matching_solve(monkeypatch):
    from cascade.sim import microduck_newton
    backend = bound_backend()
    backend.q_indices, backend.dof_indices, backend.root_index = [], [], 0
    backend.receipt['support_extraction'] = {'source_admitted': True}
    monkeypatch.setattr(microduck_newton, 'read_native_state', lambda ns, **kwargs:
                        {'step': ns.simulation_step_count, 'sim_time': ns.sim_time})
    assert backend.read()['support']['status'] == 'unavailable'  # HOME bootstrap is not solved evidence
    backend._last_support_solve = (20, .1)
    value = backend.read()
    assert value['support']['status'] == 'known'
    assert value['support']['step'] == value['step']
    assert value['solver_graph']['enabled'] is False


@pytest.mark.parametrize('bad', [None, 'mass_nan', 'mass_negative', 'mass_count', 'gravity_mismatch', 'gravity_direction'])
def test_native_mass_and_gravity_are_measured_not_assumed(bad):
    from cascade.sim.microduck_newton import read_native_body_properties
    ns = fixture()
    ns.model.body_label = ['/World/MicroDuck/trunk', '/World/MicroDuck/foot']
    ns.model.body_mass = Buffer([.6, .14])
    ns.model.gravity = Buffer([[0., 0., -9.81]])
    ns.solver.mjw_model.opt.gravity = Buffer([[0., 0., -9.81]])
    if bad == 'mass_nan': ns.model.body_mass.value[0] = np.nan
    elif bad == 'mass_negative': ns.model.body_mass.value[0] = -.6
    elif bad == 'mass_count': ns.model.body_mass.value = np.array([.6], np.float32)
    elif bad == 'gravity_mismatch': ns.solver.mjw_model.opt.gravity.value[0, 2] = -1.
    elif bad == 'gravity_direction':
        ns.model.gravity.value[0] = [0., -9.81, 0.]
        ns.solver.mjw_model.opt.gravity.value[0] = [0., -9.81, 0.]
    if bad:
        with pytest.raises(ValueError):
            read_native_body_properties(ns)
    else:
        record = read_native_body_properties(ns)
        np.testing.assert_allclose(record['body_mass_kg'], [.6, .14])
        assert record['gravity_world_m_s2'][2] == float(np.float32(-9.81))
