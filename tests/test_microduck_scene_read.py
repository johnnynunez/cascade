"""Completed-world reads retain every guard and detach each robot's result."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_microduck_stepper import Array, native_fixture

from cascade.sim.microduck_newton import _read_native_states, read_native_states


def scene(count):
    ns = native_fixture()
    ns.model.body_label = [f'/robot{i}/{part}' for i in range(count) for part in ('trunk', 'foot')]
    ns.model.body_com = Array(np.tile(ns.model.body_com.numpy(), (count, 1)))
    ns.state_0.body_q = Array(np.tile(ns.state_0.body_q.numpy(), (count, 1)))
    ns.state_0.body_qd = Array(np.tile(ns.state_0.body_qd.numpy(), (count, 1)))
    ns.state_0.body_q.value[:, 0] = np.repeat(np.arange(count) * 2., 2)
    ns.state_0.joint_q = Array(np.arange(count * 21))
    ns.state_0.joint_qd = Array(np.arange(count * 20))
    robots = {f'robot{i}': dict(q_indices=np.arange(14) + 21*i + 7,
        dof_indices=np.arange(14) + 20*i + 6, root_index=2*i) for i in range(count)}
    return ns, dict(robots=robots, max_contacts=3, max_constraints=3,
                   q_count=21*count, dof_count=20*count)


@pytest.mark.parametrize('count', [1, 2, 12])
def test_one_global_download_per_channel_keeps_distinct_robot_slices_and_all_contacts(count):
    ns, kwargs = scene(count)
    reads = {}

    class Counted(Array):
        def __init__(self, original, path):
            super().__init__(original.value, original.value.dtype)
            self.path = path

        def numpy(self):
            reads[self.path] = reads.get(self.path, 0) + 1
            return super().numpy()

    def instrument(value, prefix=''):
        for key, member in vars(value).items():
            path = prefix + key
            if isinstance(member, Array):
                setattr(value, key, Counted(member, path))
            elif isinstance(member, NS):
                instrument(member, path + '.')

    instrument(ns)
    samples = read_native_states(ns, **kwargs)
    assert len(reads) == 14 and set(reads.values()) == {1}
    assert list(samples) == list(kwargs['robots'])
    for i, sample in enumerate(samples.values()):
        assert sample['step'] == 2 and sample['sim_time'] == .01
        assert sample['position'] == [2.*i, 0., .125]
        np.testing.assert_allclose(sample['linear_velocity'], [0., .8, 0.])
        np.testing.assert_allclose(sample['angular_velocity'], [0., 0., 2.])
        np.testing.assert_array_equal(sample['q'], np.arange(14) + 21*i + 7)
        np.testing.assert_array_equal(sample['dq'], np.arange(14) + 20*i + 6)
        assert sample['contacts'] == ('/robot0/foot', '/scene/plane')
        assert sample['contact_pairs'] == [['/robot/foot/shape', '/scene/plane']]
        assert sample['contact_constraint_addresses'] == [0]
        assert sample['contact_count'] == sample['contact_candidate_count'] == sample['constraint_count'] == 1
    samples['robot0']['q'][0] = -100
    samples['robot0']['contact_pairs'][0][0] = 'mutated'
    assert ns.state_0.joint_q.value[7] == 7
    if count > 1:
        assert samples['robot1']['q'][0] == 28
        assert samples['robot1']['contact_pairs'][0][0] == '/robot/foot/shape'


@pytest.mark.parametrize('fault', ['clock', 'state', 'model', 'contacts', 'solver', 'solver_data'])
@pytest.mark.parametrize('channel', ['body_q', 'nefc', 'worldid'])
@pytest.mark.parametrize('reader', [read_native_states, _read_native_states])
def test_capture_rejects_a_changed_world_before_any_robot_result(fault, channel, reader):
    ns, kwargs = scene(2)
    owner = (ns.state_0 if channel == 'body_q' else ns.solver.mjw_data
             if channel == 'nefc' else ns.solver.mjw_data.contact)
    original = getattr(owner, channel)

    class Crossed(Array):
        def numpy(self):
            value = super().numpy()
            if fault == 'clock':
                ns.simulation_step_count += 1
            elif fault == 'solver_data':
                ns.solver.mjw_data = NS(**vars(ns.solver.mjw_data))
            else:
                field = 'state_0' if fault == 'state' else fault
                setattr(ns, field, NS(**vars(getattr(ns, field))))
            return value

    setattr(owner, channel, Crossed(original.value, original.value.dtype))
    with pytest.raises(RuntimeError, match='scene changed'):
        reader(ns, **kwargs)


@pytest.mark.parametrize('reader', [read_native_states, _read_native_states])
def test_next_call_reacquires_swapped_state_and_rejects_invalid_peer_data(reader):
    ns, kwargs = scene(2)
    first = reader(ns, **kwargs)
    old = ns.state_0
    ns.state_0 = NS(**{k: Array(v.value, v.value.dtype) for k, v in vars(old).items()})
    ns.state_0.body_q.value[2, 0] = 3.
    ns.simulation_step_count, ns.sim_time = 3, .015
    second = reader(ns, **kwargs)
    assert first['robot1']['position'][0] == 2.
    assert second['robot1']['position'][0] == 3. and second['robot0']['step'] == 3
    ns.state_0.joint_q.value[-1] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        reader(ns, **kwargs)


@pytest.mark.parametrize('fault', ['capacity', 'world', 'address', 'constraint_type', 'shape'])
@pytest.mark.parametrize('reader', [read_native_states, _read_native_states])
def test_global_contact_vetoes_still_reject_every_robot(fault, reader):
    ns, kwargs = scene(2)
    data = ns.solver.mjw_data
    if fault == 'capacity':
        data.nefc.value[0] = 3
    elif fault == 'world':
        data.contact.worldid.value[0] = 1
    elif fault == 'address':
        data.contact.efc_address.value[0, 0] = 2
    elif fault == 'constraint_type':
        data.efc.type.value[0, 0] = 1
    else:
        ns.contacts.rigid_contact_shape0.value[0] = -1
    with pytest.raises((RuntimeError, ValueError)):
        reader(ns, **kwargs)
