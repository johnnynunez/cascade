"""CPU tests of the H2 owner cadence and the bridge lifecycle; a software double never certifies physics."""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.control.h2_policy_contract import H2PolicyContract
from cascade.sim.h2_stepper import H2Stepper, validated_sample
from cascade.sim.microduck_state import body_frame_vectors
from cascade.sim.mobile_bridge import MobileBridgeController

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))


@pytest.fixture(scope='module')
def contract():
    return H2PolicyContract.from_bundle()


def quaternion_about_y(angle):
    return [math.cos(angle / 2), 0., math.sin(angle / 2), 0.]


class SoftwareBackend:
    """Standing H2 double: the deployment chain is a tick counter, the solve a clock."""

    def __init__(self, contract, *, fall_after=None, fall_kind='height'):
        self.contract = contract
        self.dt = contract.physics_dt
        self.step_count, self.sim_time, self.runner_tick = 2, 2 * contract.physics_dt, 0
        self.height = contract.init_root_pos[2]
        self.tilt = 0.
        self.contacts = tuple(contract.foot_bodies)
        self.fall_after, self.fall_kind = fall_after, fall_kind
        self.commands, self.events = [], []
        self.receipt = {'software_fixture': True}
        self.closed = 0
        self.contained = []

    def open(self):
        self.events.append('open')

    @property
    def physics_clock(self):
        return (self.step_count, self.sim_time)

    @property
    def policy_due(self):
        return self.runner_tick % self.contract.decimation == 0

    def read(self):
        c = self.contract
        return {'step': self.step_count, 'sim_time': self.sim_time, 'joint_names': c.policy_joint_names,
                'q': np.asarray(c.action_offset, np.float32), 'dq': np.zeros(14, np.float32),
                'position': np.array([0., 0., self.height]), 'orientation_wxyz': np.array(quaternion_about_y(self.tilt)),
                'linear_velocity': np.zeros(3), 'angular_velocity': np.zeros(3),
                'gravity_body': body_frame_vectors(quaternion_about_y(self.tilt), [0.] * 3, [0.] * 3, [0.] * 3)['gravity_body'],
                'contacts': self.contacts,
                'support_contacts': [{'shape_a_id': 0, 'shape_b_id': 1 + i, 'shape_a': '/World/Ground',
                                      'shape_b': f'/World/H2/{name}', 'force_on_b_world_n': [0., 0., 350.],
                                      'normal_force_n': 350., 'normal_a_to_b_world': [0., 0., 1.],
                                      'point_world_m': [0.1, 0.1 * i, 0.]} for i, name in enumerate(self.contacts)]}

    def control(self, command):
        due = self.policy_due
        self.commands.append((self.runner_tick, tuple(float(v) for v in command), due))
        self.runner_tick += 1
        return {'runner_tick': self.runner_tick - 1, **({'raw_action': [0.0] * 14} if due else {})}

    def step(self):
        self.step_count += 1
        self.sim_time += self.dt
        self.events.append(('step', self.step_count))
        if self.fall_after is not None and self.step_count >= self.fall_after:
            if self.fall_kind == 'height':
                self.height = 0.5
            elif self.fall_kind == 'tilt':
                self.tilt = 0.6
            else:
                self.contacts = tuple(self.contract.foot_bodies) + ('torso_link',)

    def capture(self):
        return {'rgb': np.zeros((24, 32, 3), np.uint8), 'step': self.step_count, 'sim_time_s': self.sim_time,
                'captured_at': 0.}

    def contain(self, reason):
        self.contained.append(reason)

    def close(self):
        self.closed += 1

    def shutdown(self, exit_code):
        self.shutdown_code = exit_code
        return True


def controller_for(contract):
    support = {'version': 1, 'model_identity_sha256': 'd' * 64, 'robot_shapes': ['/World/H2/pelvis/collisions',
               '/World/H2/left_ankle_pitch_link/collisions'], 'foot_shapes': ['/World/H2/left_ankle_pitch_link/collisions'],
               'ground_shapes': ['/World/Ground'], 'gravity_world_m_s2': [0., 0., -9.81]}
    return MobileBridgeController(robot_id='h2', source='software-only', engine='physx', device='cuda:0', kind='h2',
                                  asset_sha256='a' * 64, policy_sha256=contract.model_sha256,
                                  model_identity_sha256='d' * 64, support_contract=support, max_linear_speed=0.5,
                                  max_angular_speed=1.0, max_duration_s=30., lease_s=1., max_state_age_s=0.5,
                                  physics_dt=contract.physics_dt, policy_dt=contract.control_dt)


def test_validated_sample_applies_the_training_fall_criteria(contract):
    backend = SoftwareBackend(contract)
    sample = validated_sample(backend.read(), contract)
    assert sample['fallen'] is False and sample['fall_evidence']['illegal_contacts'] == []
    backend.height = 0.6
    assert validated_sample(backend.read(), contract)['fallen'] is True
    backend.height = contract.init_root_pos[2]
    backend.tilt = 0.6
    assert validated_sample(backend.read(), contract)['fallen'] is True
    backend.tilt = 0.
    backend.contacts = tuple(contract.foot_bodies) + ('pelvis',)
    fallen = validated_sample(backend.read(), contract)
    assert fallen['fallen'] is True and fallen['fall_evidence']['illegal_contacts'] == ['pelvis']
    backend.contacts = tuple(contract.foot_bodies)
    raw = backend.read()
    raw['joint_names'] = tuple(reversed(contract.policy_joint_names))
    with pytest.raises(ValueError, match='joint names'):
        validated_sample(raw, contract)
    raw = backend.read()
    raw['q'] = raw['q'].astype(np.float64)
    with pytest.raises(ValueError, match='float32'):
        validated_sample(raw, contract)


def test_stepper_runs_inference_every_fourth_solve_from_the_admitted_command_only(contract):
    backend, controller = SoftwareBackend(contract), controller_for(contract)
    stepper = H2Stepper(backend, controller, contract, max_steps=40, max_wall_s=30.)
    stepper.start()
    assert controller.state()['feedback_available'] is False  # the reset is not a completed sample
    for _ in range(8):
        stepper.tick()
    assert stepper.steps == 8 and stepper.policy_evaluations == 2
    assert [due for _, _, due in backend.commands] == [True, False, False, False] * 2
    assert all(cmd == (0., 0., 0.) for _, cmd, _ in backend.commands)  # nothing admitted yet
    state = controller.state()
    assert state['feedback_available'] and state['state']['joint_names'] == list(contract.policy_joint_names)
    support = state['state']['support']
    assert support['status'] == 'known' and support['epoch'] == controller.hello()['epoch']
    assert [c['shape_b'] for c in support['contacts']] == [f'/World/H2/{b}' for b in contract.foot_bodies]
    assert support['step'] == state['state']['step']
    assert state['state']['controller_status'] == 'ready' and state['state']['fallen'] is False
    hello = controller.hello()
    assert hello['kind'] == 'h2' and hello['policy_dt'] == contract.control_dt
    accepted = controller.command_velocity(dict(robot_id='h2', source='software-only', epoch=hello['epoch'],
                                                generation=hello['generation'], owner='test', command_id='walk',
                                                vx=0.3, vy=0., wz=0., duration_s=1.0))
    assert accepted['ok']
    for _ in range(4):
        stepper.tick()
    assert backend.commands[-4][1] == pytest.approx((0.3, 0., 0.)) and backend.commands[-4][2] is True
    assert stepper.policy_records[-1]['status'] == 'held'
    with pytest.raises(ValueError, match='training command ranges|admitted limit'):
        controller.command_velocity(dict(robot_id='h2', source='software-only', epoch=hello['epoch'],
                                         generation=controller.hello()['generation'], owner='test', command_id='fast',
                                         vx=0.6, vy=0., wz=0., duration_s=1.0))
    controller.stop(latch=True)
    stepper.tick()
    assert backend.commands[-1][1] == (0., 0., 0.)  # intent changes at once; torque is never cut
    latched = controller.state()['state']
    # Permission latched; the producer still attests live balance (a latch is not torque-off).
    assert latched['latched'] is True and latched['controller_status'] == 'ready'


@pytest.mark.parametrize('kind', ['height', 'tilt', 'contact'])
def test_a_fall_by_any_training_criterion_faults_and_contains_after_being_recorded(contract, kind):
    backend, controller = SoftwareBackend(contract, fall_after=5, fall_kind=kind), controller_for(contract)
    stepper = H2Stepper(backend, controller, contract, max_steps=40, max_wall_s=30.)
    stepper.start()
    for _ in range(2):
        stepper.tick()
    with pytest.raises(ValueError, match='fallen'):
        stepper.tick()
    assert controller.state()['fault'].startswith('physics state is fallen') or 'fallen' in controller.state()['fault']
    assert backend.contained and 'containment' in backend.contained[0]
    with pytest.raises(RuntimeError, match='lifecycle restart'):
        stepper.tick()


def test_clock_and_cadence_drift_are_refused(contract):
    backend, controller = SoftwareBackend(contract), controller_for(contract)
    stepper = H2Stepper(backend, controller, contract, max_steps=40, max_wall_s=30.)
    stepper.start()
    stepper.tick()
    backend.runner_tick += 3  # the deployment chain ran ticks the owner never solved
    with pytest.raises(RuntimeError, match='decimation drifted'):
        stepper.tick()
    backend, controller = SoftwareBackend(contract), controller_for(contract)
    stepper = H2Stepper(backend, controller, contract, max_steps=40, max_wall_s=30.)
    stepper.start()
    original = backend.step

    def two_steps():
        original()
        original()
    backend.step = two_steps
    with pytest.raises(RuntimeError, match='exactly one native step'):
        stepper.tick()
    with pytest.raises(ValueError, match='bundle physics timestep'):
        H2Stepper(NS(dt=0.004), controller_for(contract), contract, max_steps=1, max_wall_s=1.)


def test_isaac_base_pins_the_embodiment_kind():
    from cascade.control.isaac_base import IsaacBase
    support = {'version': 1, 'model_identity_sha256': 'd' * 64, 'robot_shapes': ['/World/H2/a'],
               'foot_shapes': ['/World/H2/a'], 'ground_shapes': ['/World/Ground'], 'gravity_world_m_s2': [0, 0, -9.81]}
    profile = dict(robot_id='h2', source='s', engine='physx', device='cuda:0', asset_sha256='a' * 64,
                   policy_sha256='b' * 64, model_identity_sha256='d' * 64, support_contract=support,
                   bridge_host='127.0.0.1', bridge_port=18400, timeout_s=0.5, kind='h2')
    base = IsaacBase(profile)
    assert base._expected['kind'] == 'h2'
    assert IsaacBase({**profile, 'kind': 'microduck'})._expected['kind'] == 'microduck'
    assert IsaacBase({k: v for k, v in profile.items() if k != 'kind'})._expected['kind'] == 'microduck'
    with pytest.raises(ValueError, match='kind'):
        IsaacBase({**profile, 'kind': 'quadruped'})
    with pytest.raises(ValueError, match='kind'):
        controller_for(H2PolicyContract.from_bundle()).__class__(
            robot_id='x', source='s', engine='physx', device='cuda:0', kind='quadruped', asset_sha256='a' * 64,
            policy_sha256='b' * 64, model_identity_sha256='d' * 64, support_contract=support, max_linear_speed=1.,
            max_angular_speed=1., max_duration_s=1., lease_s=1., max_state_age_s=1.)
    with pytest.raises(Exception, match='hello kind mismatch'):
        base._hello({'ok': True, **{k: base._expected[k] for k in base._expected}, 'kind': 'microduck',
                     'epoch': 'e', 'generation': 0, 'capabilities': ['state', 'velocity', 'stop', 'reset_stop'],
                     'lease_s': 1., 'max_action_wall_s': 1.})
    from cascade.sim.base_truth import BaseTruthReader
    from cascade.sim.mobile_identity import support_contract_digest
    reader = BaseTruthReader(profile)
    hello = {'ok': True, 'protocol': 1, 'kind': 'h2', 'measurement_kind': 'physics',
             'support_contract_sha256': support_contract_digest(support, 'd' * 64),
             **{k: profile[k] for k in ('robot_id', 'source', 'engine', 'device', 'asset_sha256', 'policy_sha256',
                                        'model_identity_sha256')}, 'epoch': 'e', 'generation': 0,
             'physics_dt': 0.005, 'policy_dt': 0.02, 'capabilities': ['state', 'velocity', 'stop', 'reset_stop']}
    try:
        reader._hello(dict(hello))  # the H2 truth channel accepts an H2 owner ...
        with pytest.raises(ValueError, match='pinned embodiment'):
            reader._hello({**hello, 'kind': 'microduck'})  # ... and refuses a MicroDuck one
        with pytest.raises(ValueError, match='pinned embodiment'):
            BaseTruthReader({k: v for k, v in profile.items() if k != 'kind'})._hello(dict(hello))
    finally:
        reader.close()


def cli():
    return importlib.import_module('isaac_h2_bridge')


def _software_identity(admission, native, *, repo):
    return {'model_identity_sha256': 'd' * 64, 'recipe': {'software': True},
            'support_contract': {**native['support_contract'], 'model_identity_sha256': 'd' * 64}}


@pytest.mark.parametrize('mode', ['normal', 'fall', 'boot'])
def test_bridge_run_serves_the_wire_and_persists_receipts_without_kit(tmp_path, monkeypatch, capsys, mode, contract):
    from cascade.sim import h2_identity
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.mobile_bridge import MobileBridgeServer
    monkeypatch.setattr(h2_identity, 'build_h2_model_identity', _software_identity)
    output = tmp_path / 'run'
    args = NS(out=output, device='cuda:0', robot_id='h2', source='software-only', max_wall_s=20., max_steps=9, port=0,
              camera_every=4, max_jpeg_bytes=100000, physics_row_every=1, policy=tmp_path / 'policy.pt',
              overview_resolution='32x24', camera_eye=(3., -3., 2.), camera_target=(0., 0., .8), ground_visual_m=40.,
              camera_follow=False)
    manifest = json.loads((REPO / 'configs/h2/bundle.json').read_text())
    admission = {'bundle': manifest, 'asset_sha256': manifest['files']['H2.usda']['sha256'],
                 'policy_sha256': manifest['files']['policy.pt']['sha256'], 'limits': json.loads(
                     (REPO / 'configs/h2/controller-limits-candidate.json').read_text())}
    created = []

    class Backend(SoftwareBackend):
        def __init__(self, *unused):
            super().__init__(contract, fall_after=6 if mode == 'fall' else None)
            self.receipt['support_contract'] = {'version': 1, 'robot_shapes': ['/World/H2/pelvis/c'],
                                                'foot_shapes': ['/World/H2/pelvis/c'], 'ground_shapes': ['/World/Ground'],
                                                'gravity_world_m_s2': [0., 0., -9.81]}
            created.append(self)

        def open(self):
            if mode == 'boot':
                raise RuntimeError('test boot failure')

        def shutdown(self, exit_code):
            assert (output / 'receipt.json').is_file(), 'Kit may terminate inside close; persist first'
            self.shutdown_code = exit_code
            return True
    seen = {}

    class Server(MobileBridgeServer):
        def start(self):
            super().start()
            client = BridgeClient(self.address[0], self.address[1], timeout_s=1.)
            client.connect()
            try:
                hello = client.request({'op': 'hello', 'role': 'reader'})
                seen['hello'] = hello
                seen['state'] = client.request({'op': 'state'})['state']
                seen['frame'] = client.request({'op': 'frame', 'camera': 'overview'})['frame']
            finally:
                client.close()
    result = cli().run(args, admission, contract, backend_factory=Backend, server_factory=Server)
    success = mode == 'normal'
    assert result['completed'] is success and result['physical_acceptance'] is False
    assert created[0].closed == 1 and created[0].shutdown_code == (0 if success else 1)
    saved = json.loads((output / 'receipt.json').read_text())
    assert saved['teardown_errors'] == [] and saved['exit_code'] == (0 if success else 1)
    assert json.loads((output / 'teardown.json').read_text())['sdk_close_returned'] is True
    if mode == 'boot':
        assert 'test boot failure' in saved['error'] and not (output / 'BRIDGE_LISTENING.json').exists()
        return
    assert seen['hello']['kind'] == 'h2' and seen['hello']['robot_id'] == 'h2'
    assert seen['state']['step'] == 3 and seen['frame']['step'] == 3  # first RPC image follows a NEW owned solve
    assert result['discarded_bootstrap_capture']['published'] is False
    policies = [json.loads(line) for line in (output / 'policy.jsonl').read_text().splitlines()]
    expected_rows = 9 if mode == 'normal' else 4  # the fallen solve is recorded before it faults
    assert len(policies) == expected_rows
    assert [p['inference'] for p in policies] == ([True, False, False, False] * 3)[:expected_rows]
    assert all(p['commands'] == [0., 0., 0.] for p in policies)
    physics = [json.loads(line) for line in (output / 'physics.jsonl').read_text().splitlines()]
    assert physics[0]['step'] == 3 and physics[0]['fallen'] is False
    assert physics[0]['controller']['controller_status'] == 'ready' and physics[0]['fall_evidence']['tilt_rad'] == 0.
    marker = json.loads((output / 'BRIDGE_LISTENING.json').read_text())
    assert marker['physical_acceptance'] is False and marker['hello']['kind'] == 'h2'
    assert 'BRIDGE_LISTENING' in capsys.readouterr().out
    if mode == 'fall':
        assert saved['steps'] == 4 and 'fallen' in saved['error']
        assert saved['last_state']['fault'] and saved['last_fall_evidence']['pelvis_height_m'] == 0.5
    else:
        assert saved['steps'] == 9 and saved['policy_evaluations'] == 3 and saved['frame_count'] == 4  # steps 1, 4, 8, 9


def test_check_only_admission_pins_everything_and_imports_no_kit(tmp_path, monkeypatch, capsys):
    import hashlib
    policy_pin = json.loads((REPO / 'configs/h2/bundle.json').read_text())['files']['policy.pt']['sha256']
    cache = REPO / 'runs/.install-cache/h2/policy.pt'
    if not cache.exists() or hashlib.sha256(cache.read_bytes()).hexdigest() != policy_pin:
        pytest.skip('pinned policy.pt not fetched (scripts/h2_assets.py)')
    release = tmp_path / 'release'
    release.mkdir()
    (release / 'python.sh').write_text('#!/bin/sh\n')
    forbidden = ('isaacsim', 'omni', 'carb', 'torch', 'pxr')
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name.split('.')[0] in forbidden:
            raise AssertionError(f'check-only must not import {name}')
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', guard)
    argv = ['--release', str(release), '--policy', str(cache), '--limits',
            str(REPO / 'configs/h2/controller-limits-candidate.json'), '--robot-id', 'h2', '--source', 'isaac-h2',
            '--device', 'cuda:0', '--port', '0', '--out', str(tmp_path / 'out'), '--max-wall-s', '10',
            '--max-steps', '10', '--check-only']
    assert cli().main(argv) == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report['ok'] and report['policy_sha256'] == policy_pin and report['observation_width'] == 255
    assert not (tmp_path / 'out').exists()
    bad = tmp_path / 'policy.pt'
    bad.write_bytes(cache.read_bytes() + b'x')
    assert cli().main(argv[:3] + [str(bad)] + argv[4:]) == 1
    assert 'pinned SHA-256' in capsys.readouterr().out
    loose = tmp_path / 'limits.json'
    loose.write_text(json.dumps({'max_linear_speed': 0.8, 'max_angular_speed': 1.0, 'max_duration_s': 30.0,
                                 'lease_s': 1.0, 'max_state_age_s': 0.5, 'max_action_wall_s': 600}))
    assert cli().main(argv[:5] + [str(loose)] + argv[6:]) == 1
    assert 'training command ranges' in capsys.readouterr().out
