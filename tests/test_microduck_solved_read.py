"""Detached read reuse on one completed solve; no synthetic physics claims."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.sim import microduck_contact_support as support
from cascade.sim import microduck_newton as native
from cascade.sim import microduck_solver_graph as graph


class Array:
    pass


class State:
    def __init__(self):
        self.joint_q = Array()

    def assign(self, other):
        pass


@pytest.fixture
def backend(monkeypatch):
    b = native.KitNewtonBackend(NS(solver_cuda_graph=True, reuse_solved_read=True),
                                 {'limits': {'max_contacts': 512, 'max_constraints': 2400}}, None)
    model = NS(joint_label=['j'], body_label=['b'], shape_label=['s'], state=State)
    b.ns = NS(initialized=True, model=model, cfg=NS(time_step_app=False, num_substeps=1, use_cuda_graph=False),
              graph=None, state_0=State(), state_1=State(), control=NS(joint_f=Array()),
              contacts=NS(force=Array()), simulation_step_count=3, sim_time=.015)
    b.SM = NS(get_physics_dt=lambda: .005, get_active_physics_engine=lambda: 'newton')
    b._model, b._layout, b._dt = model, (('j',), ('b',), ('s',)), b.dt
    b._last_support_solve = b.physics_clock
    b.q_indices = b.dof_indices = np.arange(14)
    b.root_index = 0
    b.receipt['support_extraction'] = {'source_admitted': True}
    monkeypatch.setattr(graph, 'sha256', lambda p: graph.STAGE_SHA256)
    b._solver_graph = graph.SolverGraphContract(b.ns, enabled=True, wp=NS(array=Array),
                                               dt=b._dt, source_path='unit-only')
    calls = []
    def read_state(ns, **kwargs):
        calls.append(('state', b.physics_clock))
        return {'step': ns.simulation_step_count, 'sim_time': ns.sim_time,
                'q': np.zeros(14), 'position': [0, 0, .125]}
    def read_support(ns, **kwargs):
        calls.append(('support', b.physics_clock))
        return {'step': ns.simulation_step_count, 'sim_time_s': ns.sim_time,
                'contacts': [{'normal_force_n': 2.0}]}
    monkeypatch.setattr(native, 'read_native_state', read_state)
    monkeypatch.setattr(support, 'read_support', read_support)
    return b, calls


def test_same_solve_payload_detached_from_every_consumer(backend):
    b, calls = backend
    first = b.read()
    first['q'][0] = 5
    first['position'][0] = 1
    first['support']['contacts'][0]['normal_force_n'] = 9
    second = b.read()
    assert second['q'][0] == 0 and second['position'][0] == 0
    assert second['support']['contacts'][0]['normal_force_n'] == 2
    second['support']['contacts'].clear()
    assert b.read()['support']['contacts']
    assert second['step'] == 3 and second['sim_time'] == .015
    assert len(calls) == 2  # native state + same-solve support only once


@pytest.mark.parametrize('change', ['clock_step', 'clock_time', 'support_solve'])
def test_changed_solve_never_reuses_old_payload(backend, change):
    b, calls = backend
    b.read()
    if change == 'clock_step':
        b.ns.simulation_step_count += 1
    elif change == 'clock_time':
        b.ns.sim_time = 0.0  # reset/clock regression is checked by the stepper
    else:
        b._last_support_solve = None
    b.read()
    assert len(calls) == 4


@pytest.mark.parametrize('change', ['model', 'buffer', 'layout', 'dt', 'mode', 'closed'])
def test_guards_apply_before_cache_hit(backend, change):
    b, calls = backend
    b.read()
    if change == 'model':
        b.ns.model = object()
    elif change == 'buffer':
        b.ns.state_0.joint_q = Array()
    elif change == 'layout':
        b.ns.model.shape_label.append('unexpected')
    elif change == 'dt':
        b.SM.get_physics_dt = lambda: .01
    elif change == 'mode':
        b.ns.cfg.use_cuda_graph = False
    else:
        b._closed = True
    with pytest.raises((RuntimeError, ValueError)):
        b.read()
    assert len(calls) == 2


def test_step_invalidates_before_solver_even_if_solver_raises(backend):
    b, calls = backend
    b.read()
    def solve(**kwargs):
        assert b._solved_read is None and b._last_support_solve is None
        raise RuntimeError('solver fixture failure')
    b.SM.step = solve
    with pytest.raises(RuntimeError, match='solver fixture failure'):
        b.step()
    assert b._solved_read is None


def test_containment_clears_cache_without_publishing_or_pose_write(backend):
    b, calls = backend
    b.read()
    clock = b.physics_clock
    b.contain('fault')
    assert b._solved_read is None and b.physics_clock == clock
    b.read()
    assert len(calls) == 4


def test_reuse_requires_explicit_bound_buffer_mode():
    with pytest.raises(ValueError, match='bound solver graph'):
        native.KitNewtonBackend(NS(reuse_solved_read=True, solver_cuda_graph=False), {}, None)
