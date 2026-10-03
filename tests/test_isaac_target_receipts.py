"""Queued ACKs cannot masquerade as applied targets or completed physics."""
import ast
from contextlib import nullcontext
import copy
import json
import os
import socketserver
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO, load_isaac_bridge_definitions, loopback_host
from cascade.sim.bridge_client import BridgeClient
from cascade.sim.target_receipts import TargetReceipts, vector
from test_isaac_frame_snapshot import capture_bridge as capture_bridge


@pytest.mark.parametrize('enabled', [False, True])
def test_direct_script_opt_in_bootstraps_checkout_without_pythonpath(enabled, tmp_path):
    # Execute only the script's actual optional initialization, without Kit or
    # any installed/editable CASCADE checkout making its import work by chance.
    program = r'''
import ast, json, os, sys
from pathlib import Path
from types import SimpleNamespace
repo = Path(sys.argv[1])
sys.path[:] = [p for p in sys.path if not (Path(p) / 'cascade').is_dir()]
assert 'cascade' not in sys.modules
before = list(sys.path)
tree = ast.parse((repo / 'scripts/isaac_bridge.py').read_text())
block = next(n for n in tree.body if isinstance(n, ast.If)
             and 'CASCADE_ISAAC_TARGET_RECEIPTS' in ast.unparse(n.test))
env = dict(os=os, sys=sys, _REPO_ROOT=str(repo), _target_receipts=None,
           args=SimpleNamespace(prim='/robot'), names=['joint1'], ARM_IDX=[0],
           _motion_clock_epoch='epoch')
exec(compile(ast.Module(body=[block], type_ignores=[]), 'bridge_init', 'exec'), env)
if os.environ['CASCADE_ISAAC_TARGET_RECEIPTS'] == '1':
    import cascade.sim.target_receipts as receipts
    assert Path(receipts.__file__).resolve() == repo / 'src/cascade/sim/target_receipts.py'
    assert isinstance(env['_target_receipts'], receipts.TargetReceipts)
else:
    assert env['_target_receipts'] is None and sys.path == before
    assert 'cascade' not in sys.modules
print(json.dumps({'enabled': env['_target_receipts'] is not None}))
'''
    env = os.environ.copy()
    env['CASCADE_ISAAC_TARGET_RECEIPTS'] = '1' if enabled else '0'
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-I', '-c', program, str(REPO)],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'enabled': enabled}


def bridge(enabled=True, fail=False, mutate_input=False):
    calls = []
    position = np.zeros(8, dtype=np.float32)
    targets = position.copy()
    count = [0]
    before_put = [None]
    receipts = TargetReceipts('/robot', [f'joint{i}' for i in range(1, 7)], 'epoch') if enabled else None

    def get():
        calls.append('get_targets')
        return SimpleNamespace(numpy=lambda: targets.reshape(1, -1))

    def put(value):
        calls.append(('set', value.copy().tolist()))
        if before_put[0] is not None:
            before_put[0]()
        if fail:
            raise RuntimeError('setter failed')
        targets[:] = value[0]
        if mutate_input:
            value[:] = 99.  # Borrowed buffer reused after accepting the target.

    def update():
        calls.append('update')
        count[0] += 1

    def read(name, value):
        calls.append(name)
        return SimpleNamespace(numpy=lambda: value.reshape(1, -1))

    env = dict(np=np, _target_receipts=receipts, _profile_zone=lambda _: nullcontext(),
        _python_spans=SimpleNamespace(sample_clock_if_due=lambda: None),
        _camera_video=None, _REQUIRE_CUDA=False,
        _tl=SimpleNamespace(is_playing=lambda: True), _was_playing=True,
        _state_lock=threading.Lock(), _targets={'q': None, 'grip_frac': None, 'stopped': False},
        _FINGER_STEP=None, ARM_IDX=list(range(6)), GRIP_IDX=[6, 7],
        names=[f'joint{i}' for i in range(8)],
        lower=np.zeros(8), upper=np.ones(8), step=0,
        app=SimpleNamespace(is_running=lambda: count[0] < env['until']), until=1,
        _camera_capture_due=lambda _: False, _step_with_frame_history=update,
        art=SimpleNamespace(get_dof_position_targets=get, set_dof_position_targets=put,
                            get_dof_positions=lambda: read('positions', position),
                            get_dof_velocities=lambda: read('velocities', np.zeros(8))),
        engine='physx', args=SimpleNamespace(prim='/robot', dt=.01),
        SimulationManager=SimpleNamespace(get_simulation_time=lambda: count[0]*.01,
            get_num_physics_steps=lambda: count[0]),
        _motion_clock_epoch='epoch', _exec_jobs=[], _exec_lock=threading.Lock(),
        socketserver=socketserver, threading=threading, _grip_frac_now=lambda q: 0.)
    load_isaac_bridge_definitions({'Handler', '_run_exec_jobs', '_motion_clock_snapshot'}, env)
    handler = object.__new__(env['Handler'])
    tree = ast.parse((REPO/'scripts/isaac_bridge.py').read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While) and 'app.is_running()' in ast.unparse(n.test))

    def advance():
        env['until'] = count[0] + 1
        if receipts is not None:
            receipts.completed_update(env['_motion_clock_snapshot']())
        exec(compile(ast.Module(body=[loop], type_ignores=[]), 'real_loop', 'exec'), env)

    def state():
        # Same real dispatch/read closure, serviced by the test's main loop.
        handler._on_main = lambda fn, timeout=5: fn()
        return handler._dispatch({'op': 'state'})
    return handler, advance, state, receipts, calls, before_put, count


def test_ack_and_newer_state_do_not_claim_setter_before_loop_consumes_command():
    handler, advance, state, receipts, calls, _, count = bridge()
    ack = handler._dispatch({'op': 'set_joints', 'q': [.123456789] * 6, 'command_id': 'first'})
    assert ack['target_receipt']['status'] == 'queued' and calls == []
    # Existing app.update was already in flight when the TCP worker queued it.
    count[0] += 1
    s = state()
    assert s['physics_clock']['physics_step'] == 1
    assert s['target_receipts']['last_written'] is None
    advance()
    write = state()['target_receipts']['last_written']
    assert write['sequence'] == ack['target_receipt']['sequence']
    assert write['first_write']['prior_completed_update']['physics_step'] == 1
    assert write['first_write']['arm_target'] == vector([.123456789] * 6, '<f4')
    assert write['first_write']['arm_target']['values'] != [.123456789] * 6
    first = copy.deepcopy(write['first_write'])
    advance()
    repeated = state()['target_receipts']['last_written']
    assert repeated['first_write'] == first
    assert repeated['write_count'] == 2
    assert repeated['last_write']['prior_completed_update']['physics_step'] == 2


@pytest.mark.parametrize('race', ['queue', 'stop'])
def test_setter_receipt_binds_snapshot_not_a_newer_queued_command(race):
    handler, advance, state, _, _, before_put, _ = bridge()
    first = handler._dispatch({'op': 'set_joints', 'q': [.2] * 6, 'command_id': 'first'})
    before_put[0] = lambda: handler._dispatch({'op': 'stop'} if race == 'stop' else
        {'op': 'set_joints', 'q': [.5] * 6, 'command_id': 'second'})
    advance()
    result = state()['target_receipts']
    assert result['last_written']['sequence'] == first['target_receipt']['sequence']
    assert result['last_written']['first_write']['arm_target'] == vector([.2] * 6, '<f4')
    if race == 'stop': assert result['stopped_observed'] is True
    else:
        assert result['last_queued']['command_id'] == 'second'
        assert result['last_written']['superseded_before_setter_receipt'] is True
        assert result['coverage']['superseded_before_setter_receipt'] == 1
        assert result['last_written']['write_count'] == 1  # It DID return from the setter.


def test_failed_setter_epoch_and_capacity_never_invent_application():
    handler, advance, state, receipts, _, _, _ = bridge(fail=True)
    handler._dispatch({'op': 'set_joints', 'q': [.2] * 6, 'command_id': 'first'})
    advance()
    assert state()['target_receipts']['last_written'] is None
    assert state()['target_receipts']['last_queued']['setter_failures'] == 1
    receipts.invalidate('new-epoch')
    receipts.setter_returned(1, np.zeros(8), range(6))
    assert receipts.snapshot()['last_written'] is None
    assert receipts.snapshot()['coverage']['diagnostic_errors'] == 1
    tiny = TargetReceipts('/robot', ['joint1'], 'epoch', capacity=1)
    tiny.queued([.1], 'a'); tiny.queued([.2], 'b')
    assert tiny.snapshot()['coverage']['evicted'] == 1
    assert tiny.snapshot()['coverage']['superseded_before_setter_receipt'] == 1


def test_record_copies_the_argument_before_sdk_buffer_reuse():
    handler, advance, state, _, calls, _, _ = bridge(mutate_input=True)
    handler._dispatch({'op': 'set_joints', 'q': [.2] * 6, 'command_id': 'first'})
    advance()
    write = state()['target_receipts']['last_written']['first_write']
    assert write['arm_target'] == vector([.2]*6, '<f4')
    assert next(row for row in calls if isinstance(row, tuple))[1][0][:6] == vector([.2]*6, '<f4')['values']


@pytest.mark.parametrize('stopped', [False, True])
def test_actual_loop_commands_reads_and_update_order_equal_when_disabled(stopped):
    records = []
    for enabled in (False, True):
        handler, advance, state, _, calls, _, _ = bridge(enabled=enabled)
        handler._dispatch({'op': 'set_joints', 'q': [.2] * 6})
        if stopped: handler._dispatch({'op': 'stop'})
        advance(); state(); advance(); state()
        records.append(calls)
    assert records[0] == records[1]


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_tcp_handler_new_receipt_or_legacy_none(enabled):
    handler, _, _, _, _, _, _ = bridge(enabled=enabled)
    handler.__class__.__dict__['handle'].__globals__['json'] = json
    with socketserver.ThreadingTCPServer((loopback_host(), 0), handler.__class__) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        worker.start()
        client = BridgeClient(host=loopback_host(), port=server.server_address[1])
        try:
            client.connect()
            receipt = client.set_joints(np.ones(6), command_id='wire-command')
            if enabled:
                assert receipt['command_id'] == 'wire-command'
                assert receipt['first_write'] is None
            else:
                assert receipt is None
        finally:
            client.close(); server.shutdown(); worker.join(timeout=1)
        assert not worker.is_alive()


def test_completed_update_telemetry_reuses_sdk_reads_exactly(capture_bridge):
    b = capture_bridge
    b.env['engine'] = 'physx'
    b.env['args'].dt = 1/120
    calls = []
    manager = b.env['SimulationManager']
    for name in ('get_simulation_time', 'get_num_physics_steps'):
        original = getattr(manager, name)
        def read(*args, _fn=original, _name=name, **kwargs):
            calls.append(_name)
            return _fn(*args, **kwargs)
        setattr(manager, name, read)
    sequences = []
    for enabled in (False, True):
        ledger = TargetReceipts(b.env['args'].prim, ['a']*6, b.env['_motion_clock_epoch'])
        b.env['_target_receipts'] = ledger if enabled else None
        calls.clear()
        b.env['_step_with_frame_history']()
        sequences.append(list(calls))
    assert sequences[0] == sequences[1] == ['get_simulation_time', 'get_num_physics_steps']
    assert ledger.boundary['physics_step'] == b.physics_index[0]
    assert ledger.boundary['epoch'] == b.env['_motion_clock_epoch']


@pytest.mark.parametrize('failed_stage', ['update', 'capture', 'record'])
def test_failed_update_or_history_cannot_reuse_previous_write_boundary(capture_bridge, failed_stage):
    b = capture_bridge
    b.env['engine'] = 'physx'
    b.env['args'].dt = 1/120
    ledger = TargetReceipts(b.env['args'].prim, ['a']*6, b.env['_motion_clock_epoch'])
    b.env['_target_receipts'] = ledger
    b.env['_step_with_frame_history']()
    old_boundary = copy.deepcopy(ledger.boundary)
    assert old_boundary is not None
    update = b.env['app'].update

    def fail(*args, **kwargs):
        raise RuntimeError('injected ' + failed_stage)

    if failed_stage == 'update':
        def partial_update():
            update()  # SDK may already have advanced before raising.
            fail()
        b.env['app'].update = partial_update
        with pytest.raises(RuntimeError, match='injected update'):
            b.env['_step_with_frame_history']()
    else:
        if failed_stage == 'capture':
            b.env['_capture_frame_state'] = fail
        else:
            b.env['_frame_history'].record = fail
        b.env['_step_with_frame_history']()  # Existing history handler invalidates.
    assert b.physics_index[0] > old_boundary['physics_step']
    assert ledger.boundary is None
    queued = ledger.queued([.2]*6, 'after-failure')
    ledger.setter_returned(queued['sequence'], [.2]*6, list(range(6)))
    assert ledger.snapshot()['last_written']['first_write']['prior_completed_update'] is None
