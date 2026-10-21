"""Humanoid (Unitree G1) wrist-driver fastening domain for the ordinary composed runtime.

Same route as the mounted SO-101 Factory domain: ``kind: fastening`` in a robot
profile, ``CASCADE_ROBOT=<profile>`` / ``load_robot_config`` discovery without
devices, ``build_robot_runtime`` construction, and the existing
``fastening.turn_screw`` tool executed by the unchanged ``FasteningDomain``
(through ``RecordedFactoryDomain`` and ``FactorySolveOwner``). Only the scene,
its binding/limits and the backend differ; see ``sim/humanoid_fastening.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from ..control.fastening import FasteningFault
from ..robotics.contracts import ResourceDescriptor
from ..skills.fastening_runtime import TURN_SPEC

RECIPE = 'g1_wrist_driver_m20_v1'
ROBOT_ID = 'unitree_g1_29dof_dex3_wrist_driver'
CONTROLLER_ID = 'humanoid_newton:g1_wrist_driver_m20'
OWNER_WALL_S = 900.
RECORD_CAPACITY = 40000


def is_humanoid_profile(profile):
    return isinstance(profile, dict) and profile.get('recipe') == RECIPE


def validate_humanoid_profile(profile):
    required = {'kind', 'recipe', 'assets', 'robot_asset', 'device', 'model_identity_sha256'}
    if not isinstance(profile, dict) or set(profile) - required - {'robot_id', 'gc_policy'} or required - set(profile):
        raise ValueError('humanoid fastening requires the explicit wrist-driver profile fields')
    if profile.get('gc_policy') not in (None, 'freeze-startup-heap'):
        # Same reviewed host-process collector policy as the Factory profiles
        # (sim/heap_freeze.py); not part of the physics model identity.
        raise ValueError('unreviewed humanoid fastening host GC policy')
    if profile['kind'] != 'fastening' or profile['recipe'] != RECIPE or profile.get('robot_id', ROBOT_ID) != ROBOT_ID:
        raise ValueError('only the G1 wrist-driver Factory M20 recipe is implemented for humanoids')
    for name in ('assets', 'robot_asset'):
        if not isinstance(profile[name], str) or not Path(profile[name]).is_absolute():
            raise ValueError('humanoid fastening asset paths must be explicit absolute paths')
    device = profile['device']
    if device is not None and (not isinstance(device, str) or not re.fullmatch(r'cuda:[0-9]+', device)):
        raise ValueError('humanoid fastening device must be an explicit CUDA ordinal or null (unprepared)')
    digest = profile['model_identity_sha256']
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('humanoid model identity must be an exact SHA256 or null (unprepared)')


def humanoid_description(domain_id, profile):
    validate_humanoid_profile(profile)
    resource = domain_id + '/mounted_arm_spindle'
    return (ResourceDescriptor(resource, 'mounted_fastening', ROBOT_ID,
        capabilities=('preengaged_thread_turn',), controller_id=CONTROLLER_ID, writer_id=resource,
        synthetic=False, admission='unvalidated',
        metadata={'model_identity_sha256': profile['model_identity_sha256'], 'mounted_tool': True,
                  'preengaged_fastener': True, 'seating': False, 'pickup': False,
                  'embodiment': 'unitree_g1_29dof_dex3', 'turning_joint': 'right_wrist_roll_joint',
                  'turn': 'whole_arm_rotation_about_tool_axis', 'grasp': 'three_finger_contact',
                  'grasp_fingers': ('thumb', 'index', 'middle'), 'pelvis': 'fixed'}),), [TURN_SPEC]


def prepare_humanoid_model(profile, cache_dir):
    """Construct and bind the model; zero solves, no control upload."""
    validate_humanoid_profile(profile)
    if profile['device'] is None:
        raise FasteningFault('humanoid device is unprepared; select an explicit CUDA ordinal')
    from ..sim.humanoid_fastening import bind_humanoid_model
    from ..sim.newton_humanoid_screw import HumanoidThreadingScene
    scene = HumanoidThreadingScene(profile['assets'], profile['robot_asset'], Path(cache_dir),
                                   device=profile['device'])
    if not scene.model.device.is_cuda:
        raise FasteningFault('requested CUDA humanoid model did not resolve to a CUDA device')
    return bind_humanoid_model(scene)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def build_humanoid_fastening_runtime(profile, directory, *, domain_id):
    from .factory_runtime import RecordedFactoryDomain, _ready
    from ..sim.factory_owner import FactorySolveOwner
    from ..sim.humanoid_fastening import HumanoidFasteningBackend
    validate_humanoid_profile(profile)
    if profile['model_identity_sha256'] is None:
        raise FasteningFault('humanoid model identity is unprepared; no native owner was constructed')
    if profile['device'] is None:
        raise FasteningFault('humanoid device is unprepared; select an explicit CUDA ordinal before construction')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'factory-profile.json').open('x') as stream:
        json.dump(profile, stream, sort_keys=True, indent=2, allow_nan=False)
    domain = None
    try:
        model = prepare_humanoid_model(profile, directory / 'sdf-cache')
        _write(directory / 'model.json', model.document)
        if model.binding.model_sha256 != profile['model_identity_sha256']:
            raise FasteningFault('actual humanoid model differs from the exact profile pin')
        owner = FactorySolveOwner(HumanoidFasteningBackend(model), max_wall_s=OWNER_WALL_S,
                                  record_capacity=RECORD_CAPACITY)
        domain = RecordedFactoryDomain(owner, directory, domain_id=domain_id, controller_id=CONTROLLER_ID)
        applied = None
        if profile.get('gc_policy') == 'freeze-startup-heap':
            # After SDK/model construction and the pin check, before the owner
            # advances a solve (as build_factory_runtime does): measured
            # generation-2 pauses of ~0.27 s otherwise exceed the 0.2 s
            # observation-age limit of this Newton/MJWarp process.
            from ..sim.heap_freeze import StartupHeapFreeze
            domain.heap_freeze = StartupHeapFreeze()
            applied = domain.heap_freeze.apply()
            _write(directory / 'gc-policy.json', {'model_identity_sha256': profile['model_identity_sha256'],
                                                   'domain_id': domain_id, 'profile_gc_policy': profile['gc_policy'],
                                                   'apply': applied})
        owner.start()
        readiness = _ready(owner)
        if applied is not None:
            readiness['gc_policy'] = {'selected': profile['gc_policy'], 'policy': applied['policy'],
                                      'implementation_sha256': applied['implementation_sha256'],
                                      'receipt': 'gc-policy.json'}
        domain.flush_records()
        _write(directory / 'readiness.json', readiness)
        return domain
    except BaseException as exc:
        failure = {'ready': False, 'error': f'{type(exc).__name__}: {exc}'}
        if domain is not None:
            try:
                failure['closure'] = domain.close()
            except BaseException as closing:
                failure['closure'] = {'ok': False, 'error': f'{type(closing).__name__}: {closing}'}
        _write(directory / 'startup-failure.json', failure)
        raise
