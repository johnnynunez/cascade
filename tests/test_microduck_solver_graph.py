"""Graph-buffer admission tests; actual GPU execution requires native receipts."""
from types import SimpleNamespace as NS

import pytest

from cascade.sim import microduck_solver_graph as graph


class Array:
    pass


class State:
    def __init__(self):
        self.joint_q = Array()
        self.joint_qd = Array()
        self.assignments = []

    def assign(self, other):
        self.assignments.append(other)


@pytest.fixture
def stage():
    return NS(cfg=NS(use_cuda_graph=False), graph=None, state_0=State(), state_1=State(),
              model=NS(state=State), control=NS(joint_f=Array()), contacts=NS(force=Array()))


def contract(stage, monkeypatch, enabled=True):
    monkeypatch.setattr(graph, 'sha256', lambda path: graph.STAGE_SHA256)
    return graph.SolverGraphContract(stage, enabled=enabled, wp=NS(array=Array),
                                     dt=.005, source_path='reviewed-source')


def test_graph_bootstrap_then_capture_preserves_same_buffers(stage, monkeypatch):
    c = contract(stage, monkeypatch)
    assert stage.state_temp.assignments == [stage.state_0]
    assert stage._kernels_compiled is False and stage.cfg.use_cuda_graph is True
    c.check()  # first uncaptured warm solve is legal
    assert c.telemetry()['captured'] is False
    stage.graph, stage._graph_capture_dt = object(), .005
    c.check()
    assert c.telemetry()['captured'] is True
    stage.control.joint_f.value = 2  # in-place fresh BAM publication is allowed
    c.check()


@pytest.mark.parametrize('field', ['state_0', 'state_1', 'state_temp', 'control', 'contacts'])
def test_graph_rejects_replaced_containers(stage, monkeypatch, field):
    c = contract(stage, monkeypatch)
    setattr(stage, field, object())
    with pytest.raises(RuntimeError, match='buffers replaced'):
        c.check()


@pytest.mark.parametrize('container,field', [('state_0', 'joint_q'), ('state_1', 'joint_qd'),
                                           ('control', 'joint_f'), ('contacts', 'force')])
def test_graph_rejects_replaced_array_channels(stage, monkeypatch, container, field):
    c = contract(stage, monkeypatch)
    setattr(getattr(stage, container), field, Array())
    with pytest.raises(RuntimeError, match='buffers replaced'):
        c.check()


@pytest.mark.parametrize('change', ['discard', 'replace', 'dt', 'disable'])
def test_capture_cannot_silently_rebind_or_change_dt(stage, monkeypatch, change):
    c = contract(stage, monkeypatch)
    stage.graph, stage._graph_capture_dt = object(), .005
    c.check()
    if change == 'discard':
        stage.graph = None
    elif change == 'replace':
        stage.graph = object()
    elif change == 'dt':
        stage._graph_capture_dt = .01
    else:
        stage.cfg.use_cuda_graph = False
    with pytest.raises(RuntimeError):
        c.check()


def test_unknown_sdk_source_rejected_before_allocation(stage, monkeypatch):
    monkeypatch.setattr(graph, 'sha256', lambda path: '0'*64)
    with pytest.raises(RuntimeError, match='reviewed exact'):
        graph.SolverGraphContract(stage, enabled=True, wp=NS(array=Array), dt=.005, source_path='other')
    assert not hasattr(stage, 'state_temp') and stage.cfg.use_cuda_graph is False


def test_default_uncaptured_mode_cannot_be_switched_silently(stage, monkeypatch):
    c = contract(stage, monkeypatch, enabled=False)
    c.check()
    assert not hasattr(stage, 'state_temp')
    stage.graph = object()
    with pytest.raises(RuntimeError, match='unexpected graph'):
        c.check()


def test_internal_graph_pins_source_and_captured_substeps(stage, monkeypatch):
    from cascade.sim.microduck_sdk import INTERNAL_RECIPE, INTERNAL_SOURCE_SHA256
    stage_hash = INTERNAL_SOURCE_SHA256['isaacsim.physics.newton.impl.newton_stage']
    monkeypatch.setattr(graph, 'sha256', lambda path: graph.STAGE_SHA256)
    with pytest.raises(RuntimeError, match='reviewed exact'):
        graph.SolverGraphContract(stage, enabled=True, wp=NS(array=Array), dt=.005,
                                 source_path='legacy-source', sdk_recipe=INTERNAL_RECIPE)
    assert not hasattr(stage, 'state_temp')
    monkeypatch.setattr(graph, 'sha256', lambda path: stage_hash)
    c = graph.SolverGraphContract(stage, enabled=True, wp=NS(array=Array), dt=.005,
                                 source_path='internal-source', sdk_recipe=INTERNAL_RECIPE)
    stage.graph, stage._graph_capture_dt, stage._graph_capture_substeps = object(), .005, 1
    c.check()
    assert c.source_sha256 == stage_hash
    stage._graph_capture_substeps = 2
    with pytest.raises(RuntimeError, match='substep'):
        c.check()
