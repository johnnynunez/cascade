"""Actual selection/identity boundaries with passive SDK doubles, no physics."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import test_mobile_identity as identity_tests
from test_microduck_bridge_cli import cli, forbid_live_imports
from test_microduck_stepper import Array as HostBuffer

from cascade.sim.microduck_integrator import (
    author,
    contract,
    identity,
    observe,
    selected,
)
from cascade.sim.microduck_newton import KitNewtonBackend
from cascade.sim.mobile_identity import build_model_identity

recipe_inputs = identity_tests.recipe_inputs


class Array(HostBuffer):
    @property
    def shape(self):return self.value.shape


class Attribute:
    def __init__(self):
        self.value, self.authored, self.writes = 'euler', False, []

    def GetTypeName(self):return 'token'
    def Get(self):return self.value
    def HasAuthoredValueOpinion(self):return self.authored
    def Set(self, value):
        self.writes.append(value)
        self.value, self.authored = value, True
        return True


def native():
    attr = Attribute()
    prim = NS(GetPath=lambda: '/World/PhysicsScene', GetAttribute=lambda name: attr)
    author(prim, contract('euler-v1'))
    dt = float(np.float32(.005))
    stage = NS(initialized=True, physics_scene_prim=prim, simulation_step_count=2, sim_time=2*dt,
        model=NS(mujoco=NS(integrator=Array([0], np.int32))),
        solver=NS(use_mujoco_cpu=False,
            mj_model=NS(opt=NS(integrator=0, timestep=.002, tolerance=1e-8)),
            mjw_model=NS(opt=NS(integrator=0, timestep=Array([dt], np.float32),
                                tolerance=Array([1e-6], np.float32)))))
    return stage, attr


def test_default_has_no_authoring_or_effective_readback(monkeypatch):
    forbid_live_imports(monkeypatch)
    assert contract('sdk-default') is None
    assert selected(NS(), {}) is None
    author(object(), None)  # Even accessing a USD method would fail.
    owner = KitNewtonBackend(NS(), {}, None)
    owner.verify_integrator_identity()
    assert owner.ns is None and owner.receipt == {}


@pytest.mark.parametrize('profile', ['euler', 'Euler', 'implicitfast', '', None, True, 0])
def test_unknown_profile_never_falls_back(profile):
    with pytest.raises(ValueError):contract(profile)


@pytest.mark.parametrize('change', ['missing', 'default', 'extra', 'bool', 'expected', 'nan'])
def test_selection_drift_refused_before_backend_acquisition(change, monkeypatch):
    forbid_live_imports(monkeypatch)
    row = contract('euler-v1')
    if change == 'missing':row = None
    elif change == 'default':row = contract('sdk-default')
    elif change == 'extra':row['other'] = 1
    elif change == 'bool':row['physical_admission'] = 0
    elif change == 'expected':row['expected_integrator'] = 3
    else:row['expected_integrator'] = float('nan')
    with pytest.raises(ValueError):
        KitNewtonBackend(NS(integrator_profile='euler-v1'), {'integrator_contract':row}, None)


def test_observe_keeps_cpu_warp_timestep_and_tolerance_distinct():
    stage, attr = native()
    row = observe(stage)
    assert row['cpu']['integrator'] == row['warp']['integrator'] == row['model_integrator'] == 0
    assert row['cpu']['timestep_s'] != row['warp']['timestep_s']
    assert row['cpu']['tolerance'] != row['warp']['tolerance']
    assert attr.writes == ['euler'] and stage.simulation_step_count == 2


@pytest.mark.parametrize('change', ['fallback', 'usd', 'cpu', 'warp', 'model', 'worlds', 'dtype',
                                  'bool', 'nan', 'step', 'time', 'cpu_backend'])
def test_incomplete_or_different_effective_selection_refused(change):
    stage, attr = native()
    if change == 'fallback':attr.authored = False
    elif change == 'usd':attr.value = 'implicitfast'
    elif change == 'cpu':stage.solver.mj_model.opt.integrator = 3
    elif change == 'warp':stage.solver.mjw_model.opt.integrator = 3
    elif change == 'model':stage.model.mujoco.integrator = Array([3], np.int32)
    elif change == 'worlds':stage.model.mujoco.integrator = Array([0, 0], np.int32)
    elif change == 'dtype':stage.model.mujoco.integrator = Array([0], np.float32)
    elif change == 'bool':stage.solver.mj_model.opt.integrator = False
    elif change == 'nan':stage.solver.mj_model.opt.tolerance = float('nan')
    elif change == 'step':stage.simulation_step_count = 3
    elif change == 'time':stage.sim_time += .005
    else:stage.solver.use_mujoco_cpu = True
    with pytest.raises(ValueError):observe(stage)


def test_identity_binds_explicit_profile_and_actual_options(recipe_inputs):
    admission, receipt, paths = recipe_inputs
    before = build_model_identity(admission, receipt, **paths)
    assert 'integrator' not in before['recipe']['native']
    stage, _ = native()
    admission['integrator_contract'] = contract('euler-v1')
    receipt['integrator'] = {'contract': contract('euler-v1'), 'after_bootstrap':observe(stage)}
    after = build_model_identity(admission, receipt, **paths)
    assert before['model_identity_sha256'] != after['model_identity_sha256']
    assert after['recipe']['native']['integrator'] == receipt['integrator']
    receipt['integrator']['after_bootstrap']['cpu']['tolerance'] = 2e-8
    assert build_model_identity(admission, receipt, **paths)['model_identity_sha256'] != after['model_identity_sha256']
    assert after['recipe']['native']['integrator']['after_bootstrap']['cpu']['tolerance'] == 1e-8
    del admission['integrator_contract']
    with pytest.raises(ValueError, match='not admitted'):build_model_identity(admission, receipt, **paths)


def test_identity_readback_cannot_rejuvenate_or_change_options():
    stage, _ = native()
    admission = {'integrator_contract':contract('euler-v1')}
    owner = KitNewtonBackend(NS(integrator_profile='euler-v1'), admission, None)
    owner.ns = stage
    owner.receipt = {'integrator': {'contract':contract('euler-v1'), 'after_bootstrap':observe(stage)},
                     'actual_physics_dt':float(np.float32(.005))}
    snapshot = deepcopy(owner.receipt['integrator'])
    owner.verify_integrator_identity()
    assert owner.receipt['integrator_after_identity'] == snapshot['after_bootstrap']
    stage.solver.mj_model.opt.tolerance = 2e-8
    with pytest.raises(ValueError, match='changed'):owner.verify_integrator_identity()
    assert owner.receipt['integrator'] == snapshot
    owner.receipt['actual_physics_dt'] = .006
    with pytest.raises(ValueError, match='timestep'):identity(admission, owner.receipt)


@pytest.mark.parametrize('shared', [False, True])
def test_launch_rejects_missing_optin_admission_before_output_or_sdk(tmp_path, monkeypatch, shared):
    from cascade.sim.microduck_policy_admission import target_contract
    forbid_live_imports(monkeypatch)
    args = NS(out=tmp_path/'absent', integrator_profile='euler-v1', policy_sha256='b'*64,
              target_profile='direct-v1')
    admission = {'target_contract':target_contract('b'*64, 'direct-v1')}
    if shared:
        import isaac_microduck_shared
        call = lambda: isaac_microduck_shared.run(args, admission, None)
    else:
        call = lambda: cli().run(args, admission)
    with pytest.raises(ValueError, match='integrator'):call()
    assert not args.out.exists()
