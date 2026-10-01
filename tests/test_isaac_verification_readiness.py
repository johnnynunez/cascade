"""Startup admission only: no SDK, socket, actuator or GPU in these tests."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.lazy_arm import LazyArm
from cascade.perception.isaac_camera import IsaacCamera
from cascade.sim import startup_readiness as ready
from cascade.sim.bridge_client import BridgeError
from cascade.types import Frame

ENDPOINT = ('127.0.0.1', 8611)
ROBOT = '/robot/base'


def clock(step=1000):
    return dict(version=1, engine='physx', clock='SimulationManager', epoch='epoch-one',
                robot_id=ROBOT, physics_dt_s=.01, physics_step=step, sim_time=step * .01)


def physical():
    return dict(version=1, physics_clock=clock(), q_asset=[0.] * 6,
                joint_names=[f'joint{i}' for i in range(1, 7)], joint_indices=list(range(6)),
                joint_convention='asset', pose_frame='world', meters_per_unit=1.,
                base_position_world=[0., 0., 0.], base_orientation_wxyz=[1., 0., 0., 0.],
                server_monotonic=100., physical_inventory=[dict(name='orange', path='/World_Props/orange')],
                physical_poses={'orange': [.2, .1, .03]}, physical_paths={'orange': '/World_Props/orange'})


def frame(name='cam0', *, stamp=100.2, step=1002):
    ref = dict(version=1, source='rpFabricTime', product='/Render/' + name,
               numerator=step * 10000000, denominator=1000000000, producer_epoch='epoch-one',
               history_physics_step=step, history_simulation_time=step * .01,
               render_simulation_time=step * .01, snapshot_started_monotonic=stamp,
               snapshot_finished_monotonic=stamp + .01)
    state = dict(version=1, backend='isaac', robot_id=ROBOT, joint_convention='asset',
                 q=[0.] * 6, t=stamp, time_source='physics_loop_monotonic', producer_epoch='epoch-one',
                 gripper_joints=dict(version=1, names=['joint_left', 'joint_right'],
                                     position_m=[.05, .05], lower_m=[0., 0.], upper_m=[.05, .05]))
    return Frame(rgb=np.zeros((2, 3, 3), np.uint8), depth_m=np.ones((2, 3), np.float32), K=np.eye(3),
                 capture=dict(backend='isaac', source=ENDPOINT, camera=name, t=stamp,
                              proprioception=state, render_reference=ref))


@pytest.fixture
def scenario(monkeypatch):
    s = SimpleNamespace(sample=physical(), done=dict(clock=clock(), server_monotonic=100.1),
                        now=dict(clock=clock(1005), server_monotonic=100.5),
                        calls=[], frames=[frame(n) for n in ('cam0', 'side', 'proof')],
                        elapsed=0., failure=None, raw_reply=None, native_count=0, clock_count=0)
    cfg = Cfg(dict(type='isaac', bridge_host=ENDPOINT[0], bridge_port=ENDPOINT[1],
                   bridge_robot_id=ROBOT, n_joints=6))
    def forbidden():
        raise AssertionError('startup materialized an actuator')
    s.arm = LazyArm(forbidden, profile_type='isaac')
    streams = []
    for index, name in enumerate(('cam0', 'side', 'proof')):
        camera = IsaacCamera(Cfg(dict(bridge_host=ENDPOINT[0], bridge_port=ENDPOINT[1], sim_camera=name)))
        class Stream:
            _cond = threading.Condition()
            @property
            def _latest(self):
                return s.frames[self.index]
        stream = Stream()
        stream._camera, stream.index = camera, index
        streams.append(stream)
    s.runtime = SimpleNamespace(cfg=SimpleNamespace(arm=cfg), rig=streams, arm=s.arm)
    class Client:
        def __init__(self, *args, timeout_s):
            assert args == ENDPOINT and 0 < timeout_s <= 10.
        def connect(self):
            s.calls.append('connect')
        def close(self):
            s.calls.append('close')
        def request(self, payload, timeout_s):
            assert payload['op'] == 'exec' and 0 < timeout_s <= 10.
            s.elapsed += .01
            s.calls.append(payload['code'])
            if ready._DONE in payload['code']:
                s.native_count += 1
                assert payload['code'].startswith(ready._PROBE % {'roots': repr(ready._PROP_ROOTS)})
                if s.failure:
                    raise s.failure
                if s.raw_reply is not None:
                    return s.raw_reply
                return dict(ok=True, stdout='CASCADE_TRUTH_OBSERVATION ' + json.dumps(s.sample) + '\n'
                            + ready._DONE + json.dumps(s.done))
            s.clock_count += 1
            return dict(ok=True, stdout=ready._CLOCK + json.dumps(s.now))
    monkeypatch.setattr(ready, 'BridgeClient', Client)
    monkeypatch.setattr(ready.time, 'monotonic', lambda: s.elapsed)
    monkeypatch.setattr(ready.time, 'sleep', lambda seconds: setattr(s, 'elapsed', s.elapsed + seconds))
    return s


def test_readiness_warms_once_without_materializing_lazy_arm(scenario):
    s = scenario
    result = ready.prepare_isaac_verification(s.runtime)
    assert result['pass'] and len(result['camera_packets']) == 3
    assert all(p['age_server_s'] == pytest.approx(.3) for p in result['camera_packets'])
    assert result['physical_paths'] == ['/World_Props/orange']
    assert s.native_count == s.clock_count == 1 and s.calls[-1] == 'close'
    assert not s.arm.connected and s.arm.__dict__['_arm'] is None
    assert 'q_asset' not in result and 'physical_poses' not in result  # no action authority


@pytest.mark.parametrize('reason', ['ok', 'atomic_missing', 'done_missing', 'done_duplicate', 'timeout'])
def test_incomplete_or_failed_probe_aborts_without_resubmission(scenario, reason):
    s = scenario
    if reason == 'timeout':
        s.failure = BridgeError('deadline expired, reply may still be pending')
    else:
        stdout = 'CASCADE_TRUTH_OBSERVATION ' + json.dumps(s.sample) + '\n' + ready._DONE + json.dumps(s.done)
        if reason == 'atomic_missing': stdout = stdout.split('\n')[1]
        if reason == 'done_missing': stdout = stdout.split('\n')[0]
        if reason == 'done_duplicate': stdout += '\n' + ready._DONE + json.dumps(s.done)
        s.raw_reply = dict(ok=False if reason == 'ok' else True, stdout=stdout)
    with pytest.raises((RuntimeError, BridgeError)):
        ready.prepare_isaac_verification(s.runtime)
    assert s.native_count == 1 and s.clock_count == 0 and s.calls[-1] == 'close'
    assert not s.arm.connected


@pytest.mark.parametrize('reason', ['empty', 'missing_body', 'duplicate_name', 'foreign_robot', 'foreign_source',
                                   'clock_changed', 'convention', 'joint_map', 'base', 'nonfinite'])
def test_physical_metadata_required_before_camera_admission(scenario, reason):
    s = scenario
    if reason == 'empty': s.sample['physical_inventory'] = []
    if reason == 'missing_body': s.sample['physical_poses'] = {}
    if reason == 'duplicate_name': s.sample['physical_inventory'] *= 2
    if reason == 'foreign_robot': s.sample['physics_clock']['robot_id'] = '/other'
    if reason == 'foreign_source': s.sample['physics_clock']['source'] = ['elsewhere', 9999]
    if reason == 'clock_changed': s.done['clock']['epoch'] = 'new-epoch'
    if reason == 'convention': s.sample['joint_convention'] = 'cascade'
    if reason == 'joint_map': s.sample['joint_indices'] = [0] * 6
    if reason == 'base': s.sample['base_orientation_wxyz'] = [0.] * 4
    if reason == 'nonfinite': s.sample['physical_poses']['orange'][0] = float('nan')
    with pytest.raises(RuntimeError): ready.prepare_isaac_verification(s.runtime)
    assert s.clock_count == 0 and s.calls[-1] == 'close'


@pytest.mark.parametrize('reason', ['foreign_epoch', 'foreign_camera', 'foreign_source', 'missing_reference',
                                   'reference_mismatch', 'duplicate_product', 'bad_step', 'no_jaws', 'no_depth', 'clock_regressed'])
def test_packet_and_current_clock_bindings_cannot_be_relabelled(scenario, reason):
    s = scenario
    cap = s.frames[1].capture
    if reason == 'foreign_epoch': cap['proprioception']['producer_epoch'] = 'other'
    if reason == 'foreign_camera': cap['camera'] = 'cam0'
    if reason == 'foreign_source': cap['source'] = ('other', 9999)
    if reason == 'missing_reference': cap['render_reference'] = None
    if reason == 'reference_mismatch': cap['render_reference']['snapshot_started_monotonic'] -= .001
    if reason == 'duplicate_product': cap['render_reference']['product'] = s.frames[0].capture['render_reference']['product']
    if reason == 'bad_step': cap['render_reference']['history_physics_step'] += 1
    if reason == 'no_jaws': cap['proprioception']['gripper_joints'] = None
    if reason == 'no_depth': s.frames[1].depth_m = None
    if reason == 'clock_regressed': s.now['clock'] = clock(999)
    with pytest.raises(RuntimeError):
        ready.prepare_isaac_verification(s.runtime)
    assert s.calls[-1] == 'close' and not s.arm.connected


@pytest.mark.parametrize('reason', ['duplicate_pre_probe', 'stale', 'missing'])
def test_all_cameras_need_new_post_completion_packets_within_deadline(scenario, reason):
    s = scenario
    if reason == 'duplicate_pre_probe': s.frames[2] = frame('proof', stamp=99.9, step=998)
    if reason == 'stale': s.now['server_monotonic'] = 102.201  # unchanged exact 2 s gate
    if reason == 'missing': s.frames[2] = None
    with pytest.raises(RuntimeError, match='deadline'):
        ready.prepare_isaac_verification(s.runtime)
    assert s.native_count == 1 and s.clock_count > 1 and s.calls[-1] == 'close'


def test_duplicate_old_packet_waits_for_real_update(scenario):
    s = scenario
    s.frames[0] = frame('cam0', stamp=99.9, step=998)
    original_sleep = ready.time.sleep
    def advance(seconds):
        original_sleep(seconds)
        s.frames[0] = frame("cam0")
    ready.time.sleep = advance
    assert ready.prepare_isaac_verification(s.runtime)['pass']
    assert s.native_count == 1 and s.clock_count == 2


def test_other_backends_do_not_inspect_rig_or_arm():
    rt = SimpleNamespace(cfg=SimpleNamespace(arm=Cfg({'type': 'mock'})))
    assert ready.prepare_isaac_verification(rt) is None


def test_real_launch_block_calls_readiness_before_success_and_always_shuts_down(monkeypatch):
    from cascade.apps import demo
    events = []
    cfg = SimpleNamespace(arm=Cfg({'type': 'isaac'}))
    runtime = SimpleNamespace(cfg=cfg, backends=lambda: events.append('success'))
    monkeypatch.setattr('cascade.config.load_demo_config', lambda **kw: cfg)
    monkeypatch.setattr(demo, 'build_runtime', lambda *a, **kw: (runtime, object()))
    monkeypatch.setattr(demo, 'shutdown_runtime', lambda *a: events.append('shutdown'))
    def reject(rt):
        assert rt is runtime
        events.append('readiness')
        raise RuntimeError('physical inventory incomplete')
    monkeypatch.setattr(ready, 'prepare_isaac_verification', reject)
    monkeypatch.setenv('CASCADE_CAMERAS', 'isaac,isaac_side,isaac_proof')
    monkeypatch.setenv('CASCADE_ARM', 'isaac')
    text = (Path(__file__).resolve().parents[1] / 'scripts/launch.sh').read_text()
    block, = [b for b in re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", text, re.S) if 'rt.backends()' in b]
    with pytest.raises(RuntimeError, match='physical inventory incomplete'): exec(compile(block, 'runtime-check', 'exec'), {})
    assert events == ['readiness', 'shutdown']


def test_contended_camera_lock_is_bounded_and_fails_closed(scenario, monkeypatch):
    s = scenario
    monkeypatch.setattr(ready, 'STARTUP_TIMEOUT_S', .03)
    lock = threading.Lock()
    lock.acquire()
    s.runtime.rig[0]._cond = lock
    try:
        with pytest.raises(RuntimeError, match='lock deadline'):
            ready.prepare_isaac_verification(s.runtime)
    finally:
        lock.release()
    assert s.native_count == 1 and s.clock_count == 0 and s.calls[-1] == 'close'


def test_delivery_delay_does_not_claim_current_camera_freshness(scenario):
    s = scenario
    s.now['server_monotonic'] = 102.195  # .005 margin, request costs .01 locally
    with pytest.raises(RuntimeError, match='deadline'):
        ready.prepare_isaac_verification(s.runtime)
    assert s.native_count == 1 and s.clock_count > 1


# Reuse the actual native probe fixture: constructor defaults would author USD,
# and authored transforms deliberately disagree with the physical body pose.
from test_truth_probe_readonly import probe_bridge  # noqa: E402,F401


def test_actual_probe_warms_shared_views_without_writes_and_later_truth_is_fresh(scenario, probe_bridge, monkeypatch):
    import contextlib
    import sys
    from types import ModuleType

    s = scenario
    physical_state, reader = probe_bridge
    prims = sys.modules['isaacsim.core.prims']
    monkeypatch.setattr(prims.RigidPrim, 'is_physics_handle_valid', lambda self: True, raising=False)
    monkeypatch.setattr(sys.modules['pxr'].UsdGeom, 'GetStageMetersPerUnit', lambda stage: 1., raising=False)
    for name in ('isaacsim.core.experimental', 'isaacsim.core.experimental.utils',
                 'isaacsim.core.experimental.utils.backend'):
        mod = ModuleType(name)
        mod.use_backend = lambda *a, **kw: contextlib.nullcontext()
        monkeypatch.setitem(sys.modules, name, mod)
    array = lambda value: SimpleNamespace(numpy=lambda: np.asarray(value))
    env = reader._client.bridge_globals
    env.update(_motion_clock_snapshot=clock, ARM_IDX=list(range(6)),
               names=[f'joint{i}' for i in range(1, 7)],
               art=SimpleNamespace(is_physics_tensor_entity_valid=lambda: True,
                   get_dof_positions=lambda: array([[0.] * 6]),
                   get_world_poses=lambda: (array([[0., 0., 0.]]), array([[1., 0., 0., 0.]]))))
    original = ready.BridgeClient
    class NativeClient(original):
        def request(self, payload, timeout_s):
            fake = super().request(payload, timeout_s)
            return reader._client.request(payload) if ready._DONE in payload['code'] else fake
    monkeypatch.setattr(ready, 'BridgeClient', NativeClient)
    result = ready.prepare_isaac_verification(s.runtime)
    assert result['pass'] and result['physical_paths'] == ['/World_Props/pink_cube']
    assert len(physical_state.views) == 1 and physical_state.writes == []
    assert physical_state.authored_dynamic_reads == 0 and not s.arm.connected
    physical_state.poses = [[.31, -.15, .06]]
    assert reader.pose('pink cube') == [.31, -.15, .06]
    assert len(physical_state.views) == 1 and physical_state.writes == []
