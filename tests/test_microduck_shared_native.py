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


def test_readonly_robot_view_binds_complete_contacts_without_advancing_owner():
    layout = bind_scene(**layout_fixture())
    clock = (3, .015)
    row = {'step': clock[0], 'sim_time': clock[1], 'support': {
        'version': 1, 'status': 'known', 'reason': '', 'step': clock[0], 'sim_time_s': clock[1], 'contacts': []}}
    owner = NS(layout=layout, physics_clock=clock, dt=.005,
               read_robots=lambda: {'duck0': row.copy()})
    view = SharedRobotView(owner, layout.robots[0], 'test-epoch')
    sample = view.read()
    assert sample['robot_id'] == 'duck0' and sample['epoch'] == 'test-epoch'
    assert sample['support']['model_identity_sha256'] == layout.robots[0].model_identity_sha256
    assert owner.physics_clock == clock and not hasattr(view, 'step')
