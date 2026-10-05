"""CPU regression of shared native slices; physical admission remains external."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_microduck_shared_scene import layout_fixture
from test_microduck_stepper import Array, native_fixture

from cascade.sim.microduck_newton import prepare_native_model, read_native_state
from cascade.sim.microduck_shared import bind_scene
from cascade.sim.microduck_shared_native import placements, SharedRobotView, SharedKitNewtonBackend


def test_shared_native_read_uses_explicit_noncontiguous_coordinates():
    ns = native_fixture()
    ns.state_0.joint_q = Array(np.arange(42))
    ns.state_0.joint_qd = Array(np.arange(40))
    kwargs = dict(q_indices=np.arange(14)*2+7, dof_indices=np.arange(14)*2+6,
                  root_index=0, max_contacts=3, max_constraints=3)
    with pytest.raises(ValueError):
        read_native_state(ns, **kwargs)
    sample = read_native_state(ns, **kwargs, q_count=42, dof_count=40)
    np.testing.assert_array_equal(sample['q'], np.arange(14)*2+7)
    np.testing.assert_array_equal(sample['dq'], np.arange(14)*2+6)


def test_shared_property_override_does_not_change_peer_or_free_coordinates():
    expected = {'joint_damping': .053, 'joint_armature': .0018, 'joint_effort_limit': 1e6,
                'joint_friction': .0048, 'joint_target_mode': 0, 'joint_target_ke': 0, 'joint_target_kd': 0}
    model = NS(**{key: Array([value]*40, np.int32 if key == 'joint_target_mode' else np.float32)
                  for key, value in expected.items()})
    before = model.joint_damping.numpy()
    indices = np.arange(14)*2+6
    ns = NS(model=model, solver=NS(notify_model_changed=lambda _: None))
    prepare_native_model(ns, indices, source_cap=.96, newton=NS(ModelFlags=NS(JOINT_DOF_PROPERTIES=1)),
                         dof_count=40)
    untouched = np.setdiff1d(np.arange(40), indices)
    np.testing.assert_array_equal(model.joint_damping.numpy()[untouched], before[untouched])
    assert (model.joint_damping.numpy()[indices] != before[indices]).all()


@pytest.mark.parametrize('count', [1, 2, 12])
def test_grid_preserves_independent_names_and_minimum_center_separation(count):
    p = placements(count, 2.)
    assert len(p) == count
    assert len({path for path, _ in p.values()}) == count
    positions = np.array([position for _, position in p.values()])
    for i in range(count):
        assert (np.linalg.norm(positions[i+1:] - positions[i], axis=1) >= 2.).all()
    backend = SharedKitNewtonBackend(NS(robots=count, spacing=2., solver_cuda_graph=True,
                                        reuse_solved_read=True), {}, None)
    assert backend._solver_capacity() == (512*count, 2400*count)


@pytest.mark.parametrize('count', [1, 2, 12])
def test_line_layout_gives_every_robot_its_own_forward_lane(count):
    p = placements(count, 1., 'line', 5.)
    positions = np.array([position for _, position in p.values()])
    assert len({path for path, _ in p.values()}) == count and (positions[:, 0] == 0.).all()
    # Lanes: distinct y, 1 m apart, centred on the origin, so a +x route crosses no other start.
    assert len(set(positions[:, 1].round(9))) == count
    assert np.allclose(np.sort(positions[:, 1]), np.arange(count) - (count - 1) / 2)
    for i in range(count):
        assert (np.abs(positions[i+1:, 1] - positions[i, 1]) >= 1.).all()
    # The retained grid is unchanged by the new parameters.
    assert placements(count, 2.) == placements(count, 2., 'grid', 0.)
    for bad in [dict(layout='row'), dict(layout='line', spacing=.5), dict(layout='grid', spacing=1.),
                dict(layout='line', route_m=-1.), dict(layout='line', route_m=float('nan'))]:
        with pytest.raises(ValueError):
            placements(count, bad.pop('spacing', 1.), bad.pop('layout'), bad.pop('route_m', 0.))


def test_route_frames_the_overview_camera_without_moving_any_robot():
    args = NS(robots=12, spacing=1., layout='line', route_m=5., solver_cuda_graph=True, reuse_solved_read=True)
    backend = SharedKitNewtonBackend(args, {}, None)
    plain = SharedKitNewtonBackend(NS(robots=12, spacing=1., layout='line', route_m=0., solver_cuda_graph=True,
                                      reuse_solved_read=True), {}, None)
    assert backend.placements == plain.placements  # the route writes no pose
    eye, target = (np.array(v) for v in backend._camera_pose())
    assert np.allclose(target[:2], [2.5, 0.])  # centre of start line + 5 m route
    assert eye[0] < 0. and eye[2] > target[2]  # behind the start line, above it
    eye0, target0 = (np.array(v) for v in plain._camera_pose())
    assert np.allclose(target0[:2], [0., 0.]) and eye0[2] > 0.  # retained framing without a route


def test_readonly_robot_view_binds_complete_contacts_without_advancing_owner():
    layout = bind_scene(**layout_fixture())
    clock = (3, .015)
    row = {'step': clock[0], 'sim_time': clock[1], 'support': {
        'version': 1, 'status': 'known', 'reason': '', 'step': clock[0], 'sim_time_s': clock[1], 'contacts': []}}
    owner = NS(layout=layout, physics_clock=clock, dt=.005,
               read_robot=lambda robot: row.copy())
    view = SharedRobotView(owner, layout.robots[0], 'test-epoch')
    sample = view.read()
    assert sample['robot_id'] == 'duck0' and sample['epoch'] == 'test-epoch'
    assert sample['support']['model_identity_sha256'] == layout.robots[0].model_identity_sha256
    assert owner.physics_clock == clock and not hasattr(view, 'step')


def test_selected_robot_read_detaches_its_arrays_and_contacts_without_copying_peers():
    class UnreadPeer:
        def __deepcopy__(self, memo):
            raise AssertionError('selected read copied an unrelated peer')
    cached = {'duck0': {'q': np.zeros(14, np.float32), 'support': {'contacts': [{'force': [1., 2., 3.]}]}},
              'duck1': UnreadPeer()}
    owner = SharedKitNewtonBackend.__new__(SharedKitNewtonBackend)
    owner._read_completed_scene = lambda: cached
    sample = owner.read_robot('duck0')
    sample['q'][0] = 8
    sample['support']['contacts'][0]['force'][0] = 7
    assert cached['duck0']['q'][0] == 0
    assert cached['duck0']['support']['contacts'][0]['force'][0] == 1
    with pytest.raises(KeyError):
        owner.read_robot('unknown')


def test_overview_resolution_is_opt_in_and_validated():
    from cascade.sim.microduck_newton import OVERVIEW_SHAPE, overview_shape, parse_overview_resolution
    assert OVERVIEW_SHAPE == (480, 640)
    assert parse_overview_resolution('1920x1080') == (1080, 1920) and parse_overview_resolution('640x480') == OVERVIEW_SHAPE
    assert overview_shape([1080, 1920]) == (1080, 1920)
    for bad in ('1920', '1920x', 'ax1080', '1920x1080x3', 1920, (480,), (480., 640), (10, 640), (480, 5000)):
        with pytest.raises(ValueError):
            parse_overview_resolution(bad) if isinstance(bad, str) else overview_shape(bad)
    base = NS(robots=1, spacing=2., solver_cuda_graph=True, reuse_solved_read=True)
    assert SharedKitNewtonBackend(base, {}, None).overview_shape == OVERVIEW_SHAPE
    assert 'overview_resolution' not in SharedKitNewtonBackend(base, {}, None).receipt
    wide = SharedKitNewtonBackend(NS(**vars(base), overview_resolution='1920x1080'), {}, None)
    assert wide.overview_shape == (1080, 1920) and wide.receipt['overview_resolution'] == [1080, 1920]
