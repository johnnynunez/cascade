"""Opt-in native integrator selection, without importing or constructing an SDK.

The default leaves the SDK's selection untouched. Euler is a comparison with
the official inference script, not a claim about the checkpoint's training or
about physical locomotion. Selection must survive USD import and both solvers.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real

PROFILES = ('sdk-default', 'euler-v1')
ATTRIBUTE = 'mjc:option:integrator'
SCENE = '/World/PhysicsScene'


def contract(profile):
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError('explicit known integrator profile required')
    if profile == 'sdk-default':
        return None
    return {'schema': 'cascade.microduck.integrator.v1', 'profile': profile,
            'usd_prim': SCENE, 'usd_attribute': ATTRIBUTE, 'usd_value': 'euler',
            'authoring_phase': 'before original native bootstrap',
            'expected_integrator': 0, 'physical_admission': False}


def _equal(a, b):
    try:
        return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def selected(args, admission):
    expected = contract(getattr(args, 'integrator_profile', 'sdk-default'))
    if not _equal(admission.get('integrator_contract'), expected):
        raise ValueError('selected integrator profile differs from admission')
    return expected


def author(prim, selection):
    if selection is None:
        return  # No USD access or authoring on the original path.
    if not _equal(selection, contract('euler-v1')):
        raise ValueError('invalid integrator contract')
    if not prim or str(prim.GetPath()) != SCENE:
        raise ValueError('integrator requires the existing physics scene')
    attr = prim.GetAttribute(ATTRIBUTE)
    if not attr or str(attr.GetTypeName()) != 'token':
        raise ValueError('SDK integrator token attribute missing')
    if not attr.Set('euler') or not attr.HasAuthoredValueOpinion() or attr.Get() != 'euler':
        raise ValueError('Euler integrator authoring failed')


def _integer(value):
    kind = type(value)
    enum = kind.__module__ == 'mujoco._enums' and kind.__name__ == 'mjtIntegrator'
    if isinstance(value, bool) or not (isinstance(value, Integral) or enum):
        raise ValueError('integrator readback requires an integer or MuJoCo enum')
    return int(value)


def _real(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('native option must be a finite real')
    return float(value)


def _array(value, dtype):
    if tuple(value.shape) != (1,):
        raise ValueError('integrator readback requires a single native world')
    copied = value.numpy()
    if tuple(copied.shape) != (1,) or str(copied.dtype) != dtype:
        raise ValueError('native option array shape/dtype differs')
    return _integer(copied[0]) if dtype == 'int32' else _real(copied[0])


def observe(stage):
    """Copy effective options after bootstrap; never set options or solve."""
    if not stage.initialized or stage.solver.use_mujoco_cpu is not False:
        raise ValueError('integrator profile requires the initialized Warp solver')
    prim = stage.physics_scene_prim
    attr = prim.GetAttribute(ATTRIBUTE)
    if (str(prim.GetPath()) != SCENE or not attr or str(attr.GetTypeName()) != 'token'
            or not attr.HasAuthoredValueOpinion() or attr.Get() != 'euler'):
        raise ValueError('effective integrator lacks authored Euler selection')
    cpu, gpu = stage.solver.mj_model.opt, stage.solver.mjw_model.opt
    row = {'native_step': _integer(stage.simulation_step_count),
           'native_time_s': _real(stage.sim_time),
           'usd_authored': True, 'usd_value': 'euler',
           'model_integrator': _array(stage.model.mujoco.integrator, 'int32'),
           'cpu': {'integrator': _integer(cpu.integrator), 'timestep_s': _real(cpu.timestep),
                   'tolerance': _real(cpu.tolerance)},
           'warp': {'integrator': _integer(gpu.integrator),
                    'timestep_s': _array(gpu.timestep, 'float32'),
                    'tolerance': _array(gpu.tolerance, 'float32')}}
    validate_observation(row)
    return row


def validate_observation(row):
    if type(row) is not dict or set(row) != {
            'native_step', 'native_time_s', 'usd_authored', 'usd_value', 'model_integrator', 'cpu', 'warp'}:
        raise ValueError('complete effective integrator observation required')
    if (type(row['native_step']) is not int or row['native_step'] != 2
            or row['usd_authored'] is not True or row['usd_value'] != 'euler'
            or type(row['model_integrator']) is not int or row['model_integrator'] != 0):
        raise ValueError('Euler selection or original bootstrap differs')
    for key in ('cpu', 'warp'):
        value = row[key]
        if (type(value) is not dict or set(value) != {'integrator', 'timestep_s', 'tolerance'}
                or type(value['integrator']) is not int or value['integrator'] != 0
                or _real(value['timestep_s']) <= 0 or _real(value['tolerance']) <= 0):
            raise ValueError('effective CPU/Warp integrator differs from Euler')
    if not math.isclose(_real(row['native_time_s']), 2*row['warp']['timestep_s'], rel_tol=0, abs_tol=1e-10):
        raise ValueError('integrator bootstrap time differs')


def identity(admission, native):
    selection = admission.get('integrator_contract')
    record = native.get('integrator')
    if selection is None:
        if record is not None:
            raise ValueError('native integrator selection was not admitted')
        return None
    if (not _equal(selection, contract('euler-v1')) or type(record) is not dict
            or set(record) != {'contract', 'after_bootstrap'} or not _equal(record['contract'], selection)):
        raise ValueError('effective integrator contract differs from admission')
    validate_observation(record['after_bootstrap'])
    if record['after_bootstrap']['warp']['timestep_s'] != native['actual_physics_dt']:
        raise ValueError('integrator timestep differs from native physics clock')
    return json.loads(json.dumps(record, allow_nan=False))
