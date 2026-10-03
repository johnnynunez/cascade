"""Concurrent RPC intent must not mutate a shared tick's prepared controls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
import time

import numpy as np
import pytest
from test_microduck_shared_scene import shared
from test_microduck_stepper import render_times
from test_mobile_bridge import channel

from cascade.sim.bridge_client import BridgeError
from cascade.sim.microduck_admission import BoundaryAdmission, SharedEndpoints


def setup():
    fleet, owner, steppers = shared()
    fleet.start()
    fleet.tick()
    admission = BoundaryAdmission({s.identity['robot_id']: s.controller for s in steppers})
    return fleet, owner, steppers, admission


def request(controller):
    hello = controller.hello()
    return {k: hello[k] for k in ('robot_id', 'source', 'epoch', 'generation')} | {
        'owner': 'client', 'command_id': 'command', 'vx': .2, 'vy': 0., 'wz': 0., 'duration_s': .1}


def pending(admission, robot='duck0'):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with admission._lock:
            if robot in admission._pending:
                return
        time.sleep(.001)
    raise AssertionError('RPC request was not queued')


def test_command_crossing_peer_inference_waits_until_next_tick_boundary():
    fleet, owner, steppers, admission = setup()
    for _ in range(3):
        fleet.tick()
    endpoint = admission.endpoint('duck0')
    original = steppers[1].policy.preview
    with ThreadPoolExecutor(1) as pool:
        future = None
        def crossed(obs):
            nonlocal future
            future = pool.submit(endpoint.command_velocity, request(endpoint))
            pending(admission)
            assert not future.done()
            assert steppers[0].controller.hello()['generation'] == 0
            return original(obs)
        steppers[1].policy.preview = crossed
        before = owner.step_count
        fleet.tick()
        assert owner.step_count == before + 1 and not future.done()
        admission.drain()
        ack = future.result(timeout=1)
    assert ack['accepted'] and not ack['completed']
    assert steppers[0].controller.control_at(owner.physics_clock[1]) == (.2, 0., 0.)
    assert steppers[1].controller.control_at(owner.physics_clock[1]) == (0., 0., 0.)


def test_immediate_stop_withdraws_pending_command_without_waiting_for_boundary():
    fleet, owner, steppers, admission = setup()
    endpoint = admission.endpoint('duck0')
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(endpoint.command_velocity, request(endpoint))
        pending(admission)
        ack = endpoint.stop()
        with pytest.raises(RuntimeError, match='stop withdrew'):
            future.result(timeout=1)
    assert not ack['physical_stop_verified']
    admission.drain()
    assert steppers[0].controller.control_at(owner.physics_clock[1]) == (0., 0., 0.)
    assert steppers[1].controller.hello()['generation'] == 0


def test_stop_crossing_native_preparation_still_withholds_whole_shared_solve():
    fleet, owner, steppers, admission = setup()
    original = steppers[1].actuator.before_step
    def crossed(dt):
        admission.endpoint('duck0').stop()
        original(dt)
    steppers[1].actuator.before_step = crossed
    before = owner.step_count
    with pytest.raises(RuntimeError, match='shared preparation'):
        fleet.tick()
    assert owner.step_count == before
    assert all(s.failure for s in steppers)


def test_post_ack_stop_revokes_motion_and_old_generation_cannot_reenter_queue():
    fleet, owner, steppers, admission = setup()
    endpoint = admission.endpoint('duck0')
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(endpoint.command_velocity, request(endpoint))
        pending(admission)
        admission.drain()
        accepted = future.result(timeout=1)
        stale = request(endpoint)
        stopped = endpoint.stop()
        assert stopped['generation'] > accepted['generation']
        assert steppers[0].controller.control_at(owner.physics_clock[1]) == (0., 0., 0.)
        future = pool.submit(endpoint.command_velocity, stale)
        pending(admission)
        admission.drain()
        with pytest.raises(ValueError, match='generation mismatch'):
            future.result(timeout=1)


@pytest.mark.parametrize('withdraw', ['timeout', 'closed', 'disconnect', 'stale_epoch'])
def test_delayed_withdrawn_request_never_runs_on_a_later_boundary(withdraw):
    fleet, owner, steppers, admission = setup()
    endpoint = admission.endpoint('duck0')
    arguments = request(endpoint)
    if withdraw == 'timeout':
        arguments['_received_wall'] = 1. - endpoint.lease_s + .02
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(endpoint.command_velocity, arguments)
        pending(admission)
        if withdraw == 'closed':
            admission.close()
        elif withdraw == 'disconnect':
            endpoint.owner_disconnected('client')
        elif withdraw == 'stale_epoch':
            steppers[0].controller.begin_epoch()
            admission.drain()
        with pytest.raises((RuntimeError, ValueError)):
            future.result(timeout=1)
    if not admission.closed:
        admission.drain()
    assert steppers[0].controller._active is None
    assert owner.step_count == 3


def test_reset_waits_for_boundary_and_each_robot_has_only_one_waiting_operation():
    fleet, owner, steppers, admission = setup()
    endpoint = admission.endpoint('duck0')
    endpoint.stop()
    reset = request(endpoint)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(endpoint.reset_stop, reset)
        pending(admission)
        duplicate = pool.submit(endpoint.reset_stop, reset)
        with pytest.raises(ValueError, match='already has'):
            duplicate.result(timeout=1)
        assert endpoint.state()['latched']
        admission.drain()
        assert future.result(timeout=1)['resumed_motion'] is False
    assert not endpoint.state()['latched']
    assert steppers[0].controller._active is None


def test_non_owner_cannot_drain_and_owner_cannot_block_waiting_on_itself():
    fleet, owner, steppers, admission = setup()
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match='simulation owner'):
            pool.submit(admission.drain).result(timeout=1)
    with pytest.raises(RuntimeError, match='separate RPC caller'):
        admission.endpoint('duck0').command_velocity(request(steppers[0].controller))
    assert threading.get_ident() == admission.owner_thread


@pytest.mark.parametrize('operation', ['command_velocity', 'reset_stop'])
@pytest.mark.parametrize('expiry', ['inside_callback', 'before_reply', 'wait_timeout'])
def test_late_mutating_admission_or_reply_revokes_permission(operation, expiry, monkeypatch):
    import cascade.sim.microduck_admission as module
    fleet, owner, steppers, admission = setup()
    controller = steppers[0].controller
    clock = [1.]
    controller._clock = lambda: clock[0]
    if operation == 'reset_stop':
        controller.stop()
    if expiry == 'inside_callback':
        original = getattr(controller, operation)
        def late(arguments):
            result = original(arguments)
            clock[0] += controller.lease_s + .01
            return result
        monkeypatch.setattr(controller, operation, late)
    else:
        request_type = module._Request
        def delayed_wait(*args):
            item = request_type(*args)
            original_wait = item.event.wait
            def wait(timeout):
                result = original_wait(timeout)
                if expiry == 'before_reply':
                    clock[0] += controller.lease_s + .01
                # Simulate a wait timing out while drain still owns the lock
                # and has completed the mutating callback before wakeup.
                return False if expiry == 'wait_timeout' else result
            item.event.wait = wait
            return item
        monkeypatch.setattr(module, '_Request', delayed_wait)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(getattr(admission.endpoint('duck0'), operation), request(controller))
        pending(admission)
        admission.drain()
        with pytest.raises((RuntimeError, ValueError), match='expired'):
            future.result(timeout=1)
    assert controller._active is None and controller.state()['latched']
    assert steppers[1].controller.hello()['generation'] == 0
    assert owner.step_count == 3


@pytest.mark.parametrize('operation', ['command_velocity', 'reset_stop'])
@pytest.mark.parametrize('withdraw', ['disconnect', 'close'])
def test_withdrawn_post_admission_reply_revokes_command_and_reset(operation, withdraw):
    fleet, owner, steppers, admission = setup()
    endpoint = admission.endpoint('duck0')
    if operation == 'reset_stop':
        endpoint.stop()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(getattr(endpoint, operation), request(endpoint))
        pending(admission)
        with admission._lock:
            admission.drain()
            assert not endpoint.state()['latched']
            if withdraw == 'disconnect':
                endpoint.owner_disconnected('client')
            else:
                admission.close()
        with pytest.raises(RuntimeError, match='before boundary reply'):
            future.result(timeout=1)
    assert endpoint.state()['latched'] and endpoint._active is None
    assert steppers[1].controller.hello()['generation'] == 0


@pytest.fixture
def endpoints():
    fleet, owner, steppers, _ = setup()
    endpoints = SharedEndpoints(steppers, base_port=0, max_jpeg_bytes=100_000)
    step, sim_time = owner.physics_clock
    endpoints.publish_capture({'rgb': np.zeros((8, 8, 3), np.uint8), 'step': step,
        'sim_time_s': sim_time, 'captured_at': time.monotonic(), 'render_times': render_times(sim_time)})
    try:
        yield endpoints, fleet, owner, steppers
    finally:
        endpoints.close()
        fleet.close()


def test_tcp_endpoints_bind_frame_state_and_motion_to_only_the_selected_robot(endpoints):
    endpoints, fleet, owner, steppers = endpoints
    mapping = endpoints.start()
    assert len({entry['port'] for entry in mapping.values()}) == 2
    with ExitStack() as stack, ThreadPoolExecutor(1) as pool:
        def connect(robot, role='reader'):
            client = channel(endpoints.servers[robot], role, 'client' if role != 'reader' else None)
            stack.callback(client.close)
            return client
        readers = {robot: connect(robot) for robot in mapping}
        control = connect('duck0', 'control')
        stopper = connect('duck0', 'stop')
        for robot, reader in readers.items():
            state = reader.request({'op': 'state'})['state']
            frame = reader.request({'op': 'frame', 'camera': 'overview'})['frame']
            for key in ('robot_id', 'epoch', 'model_identity_sha256', 'step', 'sim_time_s'):
                assert frame[key] == state[key]
            assert state['robot_id'] == robot
        arguments = {'op': 'command_velocity', **request(steppers[0].controller)}
        future = pool.submit(control.request, arguments)
        pending(endpoints.admission)
        # Separate read/stop channels remain responsive while control waits.
        assert readers['duck1'].request({'op': 'state'})['generation'] == 0
        assert stopper.request({'op': 'stop'})['physical_stop_verified'] is False
        with pytest.raises(BridgeError, match='stop withdrew'):
            future.result(timeout=1)
        # The real IsaacMobileBase reset omits owner; transport binds it.
        binding = steppers[0].controller.hello()
        reset = pool.submit(control.request, {'op': 'reset_stop',
            'epoch': binding['epoch'], 'generation': binding['generation']})
        pending(endpoints.admission)
        assert endpoints.admission._pending['duck0'].arguments['owner'] == 'client'
        endpoints.admission.drain()
        assert reset.result(timeout=1)['resumed_motion'] is False
        future = pool.submit(control.request, {'op': 'command_velocity', **request(steppers[0].controller)})
        pending(endpoints.admission)
        endpoints.admission.drain()
        assert future.result(timeout=1)['accepted']
        assert steppers[0].controller.control_at(owner.physics_clock[1]) == (.2, 0., 0.)
        assert steppers[1].controller.control_at(owner.physics_clock[1]) == (0., 0., 0.)
        assert stopper.request({'op': 'stop'})['cancelled']
        assert readers['duck1'].request({'op': 'state'})['generation'] == 0


@pytest.mark.parametrize('failure', ['stale_capture', 'second_listener'])
def test_endpoint_startup_refuses_mismatched_capture_and_rolls_back_partial_start(endpoints, monkeypatch, failure):
    endpoints, fleet, owner, steppers = endpoints
    if failure == 'stale_capture':
        fleet.tick()
        match = 'matching frame'
    else:
        def fail():
            raise OSError('second listener unavailable')
        monkeypatch.setattr(endpoints.servers['duck1'], 'start', fail)
        match = 'second listener'
    with pytest.raises((RuntimeError, OSError), match=match):
        endpoints.start()
    assert endpoints.closed and endpoints.admission.closed
    for server in endpoints.servers.values():
        assert server._halt.is_set()
        if server._accept_thread is not None:
            assert not server._accept_thread.is_alive()
    assert all(cache._closed for cache in endpoints.caches.values())


def test_closing_endpoints_wakes_queued_tcp_control_before_joining_workers(endpoints):
    endpoints, fleet, owner, steppers = endpoints
    endpoints.start()
    control = channel(endpoints.servers['duck0'], 'control', 'client')
    try:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(control.request, {'op': 'command_velocity', **request(steppers[0].controller)})
            pending(endpoints.admission)
            endpoints.close()
            with pytest.raises(BridgeError):
                future.result(timeout=1)
        assert all(not server._workers for server in endpoints.servers.values())
        assert all(s.controller._active is None for s in steppers)
    finally:
        control.close()
