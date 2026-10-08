"""Off-GIL `state()` reader for the shared MicroDuck owner (B12a). Software/TCP fixtures only.

The opt-in reader-server process answers `state` polls from a per-robot shared-memory slot that
the owner rewrites, under the controller lock, after every completed step and every permission
change. These tests pin the CONTRACT, not a speedup: replies byte-identical to the owner's in-GIL
path at the same clock; an age that reads never refresh (an owner stall ages the reply and the
existing client freshness check refuses it); torn or mixed slot reads retried and never served;
slots bound to robot + model identity; the unchanged wire (hello, commands, stop and frames stay
on the owner, a frame on a handed-off reader goes back to the owner); and owner death closing
every reader. Synthetic software states: none of this is physics or a control result.
"""
from __future__ import annotations

import importlib
import json
import marshal
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not hasattr(socket, 'send_fds') or sys.platform == 'win32',
                                reason='the off-GIL state reader passes descriptors over Unix sockets')

ROOT = Path(__file__).resolve().parents[1]
HELLO = {'op': 'hello', 'role': 'reader'}
STATE = {'op': 'state'}
LIMITS = dict(max_vx=.3, max_vy=0., max_wz=1., max_duration_s=60., max_state_age_s=.5, max_no_progress_s=.4,
              max_wall_duration_s=120., poll_interval_s=.02, turn_speed_rad_s=.5, turn_tolerance_rad=.05,
              max_turn_angle_rad=3.)


def wire(value):
    """Exactly the bytes MobileBridgeServer._send writes for an in-GIL reply."""
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode() + b'\n'


def wait_until(predicate, timeout=20.):
    """Generous bound for events that take microseconds; never a speed assertion."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'bounded wait expired'
        time.sleep(.002)


class Clock:
    def __init__(self, now=100.):
        self.now = now

    def __call__(self):
        return self.now


class Raw:
    """Byte-level newline-JSON client: the replies are compared as bytes, not parsed dicts."""
    def __init__(self, address):
        self.sock = socket.create_connection(address, timeout=10.)

    def send(self, payload):
        self.sock.sendall(payload if isinstance(payload, bytes) else json.dumps(payload).encode() + b'\n')

    def receive(self):
        chunks = bytearray()
        while b'\n' not in chunks:
            part = self.sock.recv(65536)
            if not part:
                raise ConnectionError('peer closed the reader channel')
            chunks.extend(part)
        line, extra = bytes(chunks).split(b'\n', 1)
        assert not extra
        return line + b'\n'

    def request(self, payload):
        self.send(payload)
        return self.receive()

    def outcome(self, payload):
        self.send(payload)
        try:
            return 'reply', self.receive()
        except (ConnectionError, OSError):
            return ('closed',)

    def close(self):
        self.sock.close()


@pytest.fixture
def fleet(monkeypatch):
    """The production shared-scene fixture, its controllers built as publishing controllers."""
    from cascade.sim import mobile_bridge
    from cascade.sim.mobile_state_offload import StatePublishingController
    from test_microduck_shared_scene import shared
    monkeypatch.setattr(mobile_bridge, 'MobileBridgeController', StatePublishingController)
    fleet, owner, steppers = shared(2)
    clock = Clock()
    for s in steppers:
        s.controller._clock = clock
    try:
        yield fleet, owner, steppers, clock
    finally:
        fleet.close()


@pytest.fixture
def slots(fleet):
    from cascade.sim.mobile_state_server import SlotSegment
    _, _, steppers, _ = fleet
    hello = [s.controller.hello() for s in steppers]
    segment = SlotSegment.create([(h['robot_id'], h['model_identity_sha256']) for h in hello], payload_bytes=1 << 20)
    try:
        for index, s in enumerate(steppers):
            s.controller.attach_state_slot(segment, index)
        yield segment
    finally:
        segment.close(unlink=True)


def command(controller, **changes):
    hello = controller.hello()
    return controller.command_velocity({k: hello[k] for k in ('robot_id', 'source', 'epoch', 'generation')} | {
        'owner': 'client', 'command_id': 'walk', 'vx': .2, 'vy': 0., 'wz': 0., 'duration_s': .1} | changes)


def drive(scenario, fleet):
    shared_fleet, owner, steppers, _ = fleet
    c = steppers[0].controller
    if scenario == 'latched':
        c.stop(latch=True)
    elif scenario == 'active':
        command(c)
    elif scenario == 'completed':
        command(c, duration_s=.005)
        c.control_at(owner.physics_clock[1] + .01)
    elif scenario == 'fault':
        c.fault('synthetic software fault')
    elif scenario == 'reset':
        c.stop(latch=True)
        hello = c.hello()
        c.reset_stop({'epoch': hello['epoch'], 'generation': hello['generation']})
    elif scenario == 'next_step':
        shared_fleet.tick()
    elif scenario == 'epoch':
        c.begin_epoch()
    return c


def without_ages(reply):
    reply = dict(reply)
    reply.pop('state_age_s')
    if reply['state'] is not None:
        reply['state'] = {k: v for k, v in reply['state'].items() if k != 'producer_age_s'}
    return reply


@pytest.mark.parametrize('scenario', ['published', 'latched', 'active', 'completed', 'fault', 'reset',
                                      'next_step', 'epoch'])
def test_slot_reply_is_byte_identical_to_the_owner_reply_for_the_same_step_and_clock(fleet, slots, scenario):
    from cascade.sim.mobile_state_server import StateRenderer
    shared_fleet, owner, steppers, clock = fleet
    shared_fleet.start()
    shared_fleet.tick()
    clock.now = 100.5
    c = drive(scenario, fleet)
    # Premise the renderer relies on: the reply depends on the clock ONLY through the two age fields.
    held = clock.now
    clock.now, early = 200., c.state()
    clock.now, late = 300., c.state()
    assert without_ages(early) == without_ages(late)
    if early['state'] is not None:
        assert early['state_age_s'] != late['state_age_s']
        assert early['state']['producer_age_s'] == early['state_age_s']
    renderer = StateRenderer()
    wall = c._state_wall
    for now in (held, held + .0123, held + .6, (wall if wall is not None else held) - .25):
        clock.now = now
        expected = wire(c.state())
        record = slots.read(0)
        assert renderer.render(record, now) == expected          # cached template
        assert StateRenderer().render(record, now) == expected   # first render of this publication
    peer = steppers[1].controller
    clock.now = held + .3
    assert StateRenderer().render(slots.read(1), clock.now) == wire(peer.state())


def test_an_owner_stall_ages_the_served_state_and_the_existing_freshness_check_refuses_it(fleet, slots):
    from cascade.control.isaac_base import state_from_wire
    from cascade.safety.base_harness import BaseSafetyHarness
    from cascade.sim.mobile_state_server import StateRenderer
    shared_fleet, owner, steppers, clock = fleet
    shared_fleet.start()
    shared_fleet.tick()
    c = steppers[0].controller
    wall = c._state_wall
    renderer = StateRenderer()

    def served(now):
        record = slots.read(0)
        assert record.state_wall == wall  # a read never refreshes the publish time
        reply = json.loads(renderer.render(record, now))
        return state_from_wire(reply, received_monotonic_s=now, round_trip_s=0.)

    fresh, stale = served(wall + .1), served(wall + .6)
    assert fresh.producer_age_s == pytest.approx(.1) and stale.producer_age_s == pytest.approx(.6)
    assert (fresh.step, fresh.sim_time_s) == (stale.step, stale.sim_time_s) == owner.physics_clock
    harness = BaseSafetyHarness(LIMITS)
    identity = dict(robot_id=fresh.robot_id, source=fresh.source, measurement_kind='physics')
    harness.validate_state(fresh, now=wall + .1, **identity)
    with pytest.raises(ValueError, match='stale'):
        harness.validate_state(stale, now=wall + .6, **identity)


def test_a_write_in_progress_or_a_mixed_slot_is_retried_and_never_served(fleet, slots):
    from cascade.sim.mobile_state_server import SlotSegment, SlotUnavailable
    shared_fleet, owner, steppers, clock = fleet
    shared_fleet.start()
    shared_fleet.tick()
    good = slots.read(0)
    # A writer stopped between the odd and the even sequence number (an owner killed mid-write).
    sequence = slots._begin(0)
    with pytest.raises(SlotUnavailable, match='in progress'):
        slots.read(0)
    slots._commit(0, sequence)
    assert slots.read(0) == good
    # A payload byte that changed under an even sequence number (an unfenced store seen early on
    # a weakly ordered CPU): the CRC refuses it, it is retried, the bounded retries refuse.
    offset = slots.payload_offset(0) + len(good.payload) // 2
    original = slots.buf[offset]
    slots.buf[offset] = original ^ 0xFF
    with pytest.raises(SlotUnavailable, match='torn'):
        slots.read(0)
    slots.buf[offset] = original
    assert slots.read(0) == good
    # A consistent but OLDER publication than one already served is never served again.
    reader = SlotSegment.attach(slots.name, expected=slots.identities)
    try:
        older = bytes(slots.slot_bytes(0))
        clock.now += .02
        shared_fleet.tick()
        newer = reader.read(0)
        assert newer.publication > good.publication and newer.step == good.step + 1
        slots.slot_bytes(0)[:] = older
        with pytest.raises(SlotUnavailable, match='regress'):
            reader.read(0)
    finally:
        reader.close()


def test_concurrent_publications_are_only_ever_read_whole_and_in_order():
    from cascade.sim.mobile_state_server import SlotSegment, SlotUnavailable
    owner = SlotSegment.create([('duck00', 'a' * 64)], payload_bytes=1 << 16)
    reader = SlotSegment.attach(owner.name, expected=owner.identities)
    done = threading.Event()

    def write():
        try:
            for step in range(1, 401):
                body = {'step': step, 'pad': 'x' * (step * 37 % 9000)}
                owner.write(0, marshal.dumps(body), state_wall=float(step), step=step, generation=step)
        finally:
            done.set()

    writer = threading.Thread(target=write)
    writer.start()
    seen, last = 0, 0
    try:
        while not done.is_set() or seen == 0:
            try:
                record = reader.read(0)
            except SlotUnavailable:
                continue
            body = marshal.loads(record.payload)
            assert body['step'] == record.step == record.generation == int(record.state_wall)
            assert record.publication >= last
            last, seen = record.publication, seen + 1
    finally:
        writer.join()
        reader.close()
        owner.close(unlink=True)
    assert seen > 0 and last <= 401  # 400 writes after the creation record (publication 1)


def test_a_slot_is_bound_to_its_robot_and_model_identity():
    from cascade.sim.mobile_state_server import SlotSegment, SlotUnavailable
    ids = [('duck00', 'a' * 64), ('duck01', 'b' * 64)]
    owner = SlotSegment.create(ids, payload_bytes=4096)
    try:
        for wrong in ([('duck00', 'a' * 64)], [('duck00', 'a' * 64), ('duck01', 'c' * 64)],
                      [('duck01', 'b' * 64), ('duck00', 'a' * 64)]):
            with pytest.raises(ValueError, match='identity|slot count'):
                SlotSegment.attach(owner.name, expected=wrong)
        reader = SlotSegment.attach(owner.name, expected=ids)
        try:
            with pytest.raises(SlotUnavailable, match='no state'):
                reader.read(1)
            owner.write(1, marshal.dumps({'ok': True}), state_wall=None, step=None, generation=0)
            assert reader.read(1).publication == 2  # publication 1 is the creation record
            owner.identities[1] = ('duck07', 'b' * 64)  # a record written under another identity
            owner.write(1, marshal.dumps({'ok': True}), state_wall=None, step=None, generation=0)
            with pytest.raises(SlotUnavailable, match='identity'):
                reader.read(1)
            owner.mark_closed('owner closed the endpoints')
            with pytest.raises(SlotUnavailable, match='closed'):
                reader.read(0)
        finally:
            reader.close()
    finally:
        owner.close(unlink=True)


def test_a_publication_the_slot_cannot_hold_is_refused_never_replaced_by_an_older_reply(fleet, slots):
    from cascade.sim.mobile_state_server import SlotUnavailable
    shared_fleet, owner, steppers, clock = fleet
    shared_fleet.start()
    shared_fleet.tick()
    c = steppers[0].controller
    good = slots.read(0)
    # A reply larger than the slot: the slot refuses, the previous (older) reply is not served.
    assert slots.write(0, b'x' * (slots.capacity + 1), state_wall=1., step=1, generation=0) is False
    with pytest.raises(SlotUnavailable, match='could not publish'):
        slots.read(0)
    clock.now += .02
    shared_fleet.tick()
    assert slots.read(0).step == good.step + 1  # the next whole publication is served again
    # A failing mirror never raises into the control path; the slot refuses instead of keeping
    # the previous step.
    def failing(*args, **kwargs):
        raise OSError('synthetic slot failure')
    slots.write = failing
    try:
        clock.now += .02
        shared_fleet.tick()
    finally:
        del slots.write
    assert c.state()['state']['step'] == good.step + 2  # the owner's own path published the step
    assert 'synthetic slot failure' in c.slot_errors[-1]
    with pytest.raises(SlotUnavailable, match='could not publish'):
        slots.read(0)
    clock.now += .02
    shared_fleet.tick()
    assert slots.read(0).step == good.step + 3


def test_the_publishing_controller_wraps_every_mutating_method():
    from cascade.sim.mobile_bridge import MobileBridgeController
    from cascade.sim.mobile_state_offload import PUBLISHING, READ_ONLY, StatePublishingController
    public = {name for name, value in vars(MobileBridgeController).items()
              if callable(value) and not name.startswith('_')}
    assert set(READ_ONLY) == {'hello', 'state'}
    assert public == set(PUBLISHING) | set(READ_ONLY) and not set(PUBLISHING) & set(READ_ONLY)
    for name in PUBLISHING:
        assert vars(StatePublishingController)[name] is not vars(MobileBridgeController)[name]


def test_each_permission_change_is_in_the_slot_before_its_ack_returns(fleet, slots):
    shared_fleet, owner, steppers, clock = fleet
    shared_fleet.start()
    shared_fleet.tick()
    c = steppers[0].controller

    def slot(index=0):
        return marshal.loads(slots.read(index).payload)

    peer = slot(1)
    ack = c.stop(latch=True)
    assert slot()['latched'] and slot()['state']['latched'] and slot()['generation'] == ack['generation']
    hello = c.hello()
    ack = c.reset_stop({'epoch': hello['epoch'], 'generation': hello['generation']})
    assert not slot()['latched'] and slot()['generation'] == ack['generation']
    ack = command(c)
    assert slot()['command_id'] == 'walk' and slot()['controller'] == 'walking'
    assert slot()['generation'] == ack['generation']
    # Polling paths republish only when the reply changes: nothing changed, nothing written.
    publication = slots.read(0).publication
    c.watchdog()
    c.control_at(owner.physics_clock[1])
    assert slots.read(0).publication == publication
    c.control_at(owner.physics_clock[1] + 1.)  # the command completes
    assert slot()['command_id'] is None and slot()['last_completed_command_id'] == 'walk'
    c.fault('synthetic software fault')
    assert slot()['fault'] == 'synthetic software fault' and slot()['controller'] == 'fault'
    assert slot(1) == peer  # a peer's slot never sees this robot's permission changes


@pytest.fixture
def offgil(fleet):
    from cascade.sim.microduck_admission import SharedEndpoints
    from test_microduck_stepper import render_times
    shared_fleet, owner, steppers, _ = fleet
    for s in steppers:
        s.controller._clock = time.monotonic  # the server ages replies on the system monotonic clock
    shared_fleet.start()
    shared_fleet.tick()
    endpoints = SharedEndpoints(steppers, base_port=0, max_jpeg_bytes=100_000, state_reader='process')
    try:
        step, sim_time = owner.physics_clock
        endpoints.publish_capture({'rgb': np.zeros((8, 8, 3), np.uint8), 'step': step, 'sim_time_s': sim_time,
                                   'captured_at': time.monotonic(), 'render_times': render_times(sim_time)})
        endpoints.start()
        yield endpoints, shared_fleet, owner, steppers
    finally:
        endpoints.close()


def handed_off(endpoints, reader):
    before = endpoints.state_reader.counters['handoffs']
    assert json.loads(reader.request(STATE))['ok']  # the first poll is the owner's, then the handoff
    wait_until(lambda: endpoints.state_reader.counters['handoffs'] == before + 1)


def test_reader_polls_move_to_the_state_server_and_match_the_owner_reply(offgil, monkeypatch):
    from test_mobile_bridge import channel
    endpoints, shared_fleet, owner, steppers = offgil
    server, controller = endpoints.servers['duck0'], steppers[0].controller
    reader = Raw(server.address)
    try:
        hello = json.loads(reader.request(HELLO))
        assert hello['ok'] and hello['robot_id'] == 'duck0' and hello['epoch'] == controller.hello()['epoch']
        handed_off(endpoints, reader)
        original = server.dispatch

        def owner_dispatch(request):
            assert request.get('op') != 'state', 'the owner served a handed-off reader'
            return original(request)

        monkeypatch.setattr(server, 'dispatch', owner_dispatch)

        def poll():
            before = time.monotonic()
            raw = reader.request(STATE)
            after = time.monotonic()
            reply = json.loads(raw)
            age, wall = reply['state_age_s'], controller._state_wall
            assert reply['state']['producer_age_s'] == age and reply['state']['received_monotonic_s'] == wall
            assert before - wall <= age <= after - wall  # computed at serve time, never refreshed
            expected = controller.state()
            expected['state_age_s'] = expected['state']['producer_age_s'] = age
            assert raw == wire(expected)
            return reply

        assert poll()['state']['step'] == owner.physics_clock[0]
        # A stop on the owner's stop channel is in the slot before its ACK: read-your-writes.
        control = channel(server, 'control', 'client')
        stopper = channel(server, 'stop', 'client')
        try:
            ack = stopper.request({'op': 'stop'})
            reply = poll()
            assert reply['latched'] and reply['state']['latched'] and reply['generation'] == ack['generation']
        finally:
            stopper.close()
            control.close()
        shared_fleet.tick()
        assert poll()['state']['step'] == owner.physics_clock[0]
        # The peer's endpoint is independent and still on its first (owner) poll.
        assert endpoints.state_reader.counters['handoffs'] == 1
    finally:
        reader.close()


def test_a_frame_on_a_handed_off_reader_goes_back_to_the_owner_and_stays_there(offgil, monkeypatch):
    endpoints, shared_fleet, owner, steppers = offgil
    server = endpoints.servers['duck0']
    reader = Raw(server.address)
    try:
        reader.request(HELLO)
        handed_off(endpoints, reader)
        assert json.loads(reader.request(STATE))['ok']
        frame = json.loads(reader.request({'op': 'frame', 'camera': 'overview'}))
        assert frame['ok'] and frame['frame']['robot_id'] == 'duck0'
        assert (frame['frame']['step'], frame['frame']['sim_time_s']) == owner.physics_clock
        wait_until(lambda: endpoints.state_reader.counters['returned'] == 1)
        calls = []
        original = server.dispatch
        monkeypatch.setattr(server, 'dispatch', lambda request: calls.append(request.get('op')) or original(request))
        assert json.loads(reader.request(STATE))['ok']
        assert calls == ['state']  # pinned to the owner: today's path, no second handoff
        assert endpoints.state_reader.counters['handoffs'] == 1
    finally:
        reader.close()


@pytest.mark.parametrize('payload', [
    b'{"op":"hello","role":"reader"}\n', b'{"op":"command_velocity"}\n', b'{"op":"stop"}\n', b'{"op":"nope"}\n',
    b'[1]\n', b'{"op":"state","x":1,"x":2}\n', b'{"op":"state","v":NaN}\n', b'not json\n',
    b'{"op":"state"}\n{"op":"state"}\n', b'{' + b' ' * 20000,
], ids=['second-hello', 'command', 'stop', 'unknown', 'array', 'duplicate', 'nan', 'garbage', 'pipelined', 'oversize'])
def test_a_handed_off_reader_keeps_the_owner_framing_rules(offgil, payload):
    endpoints, shared_fleet, owner, steppers = offgil
    server = endpoints.servers['duck1']
    outcomes = []
    for move in (False, True):
        reader = Raw(server.address)
        try:
            reader.request(HELLO)
            if move:
                handed_off(endpoints, reader)
            outcomes.append(reader.outcome(payload))
        finally:
            reader.close()
    assert outcomes[0] == outcomes[1]


OWNER = r'''
import json, sys, time
sys.path[:0] = [{tests!r}, {src!r}]
from mobile_support_fixture import support, support_contract
from cascade.sim.mobile_bridge import MobileBridgeServer
from cascade.sim.mobile_state_offload import StatePublishingController, StateServerProcess
c = StatePublishingController(robot_id='duck', source='isolated-bridge', engine='physx', device='cuda:0',
    asset_sha256='a'*64, policy_sha256='b'*64, model_identity_sha256='e'*64, support_contract=support_contract(),
    max_linear_speed=.2, max_angular_speed=.8, max_duration_s=10., lease_s=.5, max_state_age_s=.5)
c.publish(dict(step=1, sim_time=.005, position=[0., 0., .12], orientation_wxyz=[1., 0., 0., 0.],
    linear_velocity=[0.]*3, angular_velocity=[0.]*3, q=[0.]*14, dq=[0.]*14, fallen=False,
    joint_names=['joint_%d' % i for i in range(14)], contacts=['left_foot'], support=support(1, .005)))
reader = StateServerProcess({{'duck': c}})
server = MobileBridgeServer(c, port=0, state_offload=reader.offload('duck'))
reader.bind('duck', server)
server.start()
print(json.dumps({{'port': server.address[1], 'server_pid': reader.pid}}), flush=True)
while reader.counters['handoffs'] < 1:
    time.sleep(.002)
print('handed-off', flush=True)
time.sleep(600)
'''


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists():
        try:
            return stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except OSError:
            return False
    if sys.platform == 'darwin':
        state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(state) and not state.startswith('Z')
    return True


def test_owner_death_closes_every_handed_off_reader_and_ends_the_server():
    script = OWNER.format(tests=str(ROOT / 'tests'), src=str(ROOT / 'src'))
    owner = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True)
    reader = server_pid = None
    try:
        info = json.loads(owner.stdout.readline())
        server_pid = info['server_pid']
        assert server_pid != owner.pid and alive(server_pid)
        reader = Raw(('127.0.0.1', info['port']))
        assert json.loads(reader.request(HELLO))['ok']
        assert json.loads(reader.request(STATE))['ok']  # owner poll, then the handoff
        assert owner.stdout.readline().strip() == 'handed-off'
        assert json.loads(reader.request(STATE))['state']['step'] == 1  # served by the state server
        owner.kill()
        owner.wait(timeout=30)
        wait_until(lambda: not alive(server_pid), timeout=30.)  # control EOF: the server exits
        with pytest.raises((ConnectionError, OSError)):
            reader.request(STATE)
    finally:
        if reader is not None:
            reader.close()
        owner.kill()
        owner.wait(timeout=30)
        if server_pid is not None and alive(server_pid):
            os.kill(server_pid, 9)


def test_the_state_server_runs_isolated_on_the_standard_library_only():
    source = (ROOT / 'src/cascade/sim/mobile_state_server.py').read_text()
    assert 'import cascade' not in source and 'from cascade' not in source and 'from .' not in source
    # -I: no PYTHONPATH, no user site, no script directory on sys.path; a cascade import would fail.
    probe = subprocess.run([sys.executable, '-I', str(ROOT / 'src/cascade/sim/mobile_state_server.py'), '--help'],
                           capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0, probe.stderr
    assert '--control-fd' in probe.stdout


def _argv(tmp_path):
    return ['--robots', '2', '--spacing', '2', '--engine', 'newton', '--release', str(tmp_path),
            '--sdk-recipe', 'isaac62_48b2d951', '--bundle', str(tmp_path), '--bundle-sha256', 'a' * 64,
            '--policy', str(tmp_path / 'p.onnx'), '--policy-sha256', 'b' * 64, '--bam-source-root', str(tmp_path),
            '--bam-profile', 'fixture', '--robot-id', 'microduck', '--source', 'isaac-microduck', '--device', 'cuda:0',
            '--port', '0', '--limits', str(tmp_path / 'limits.json'), '--out', str(tmp_path / 'out'),
            '--max-wall-s', '10', '--max-steps', '10', '--solver-cuda-graph', '--reuse-solved-read']


def test_state_reader_flag_is_validated_before_admission(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    runner = importlib.import_module('isaac_microduck_shared')
    reached = []

    def admit(args):
        reached.append(args.state_reader)
        raise RuntimeError('admission reached')

    monkeypatch.setattr(runner, 'admit', admit)
    monkeypatch.setattr(runner, 'bind_repo', lambda: None)
    with pytest.raises(ValueError, match='--state-reader process requires --serve-base-port'):
        runner.main(_argv(tmp_path) + ['--state-reader', 'process'])
    with pytest.raises(SystemExit):
        runner.main(_argv(tmp_path) + ['--state-reader', 'threads'])
    for extra in (['--serve-base-port', '0', '--state-reader', 'process'], ['--serve-base-port', '0'], []):
        with pytest.raises(RuntimeError, match='admission reached'):
            runner.main(_argv(tmp_path) + extra)
    assert reached == ['process', 'owner-gil', 'owner-gil']
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('serve, mode', [(None, None), (0, None), (0, 'owner-gil'), (0, 'process')])
def test_launcher_names_the_state_reader_mode_and_hashes_only_the_opt_in(tmp_path, monkeypatch, serve, mode):
    from test_microduck_shared_staging import _launch_shared_runner
    run = _launch_shared_runner(tmp_path, monkeypatch, profile_phases=False, gc_policy=None,
                                serve_base_port=serve, state_reader=mode)
    result = run.result
    assert result['completed'] and result['exit_code'] == 0, result
    configuration = run.created[0].receipt['configuration']
    if serve is None:
        assert result['state_reader'] is None
    elif mode != 'process':
        assert result['state_reader'] == {'mode': 'owner-gil'}
        assert (run.out / 'BRIDGE_LISTENING.json').exists()
    else:
        receipt = result['state_reader']
        assert receipt['mode'] == 'process' and receipt['server_exit_code'] == 0
        assert receipt['handoffs'] == 0 and receipt['publish_errors'] == []
        assert receipt['publications']['duck0'] >= result['steps']
        assert (run.out / 'BRIDGE_LISTENING.json').exists()
    assert configuration.get('state_reader') == ('process' if mode == 'process' else None)


def test_profiled_owner_times_each_slot_publication_inside_its_publication_span(tmp_path, monkeypatch):
    from test_microduck_shared_staging import _launch_shared_runner
    run = _launch_shared_runner(tmp_path, monkeypatch, profile_phases=True, gc_policy=None,
                                serve_base_port=0, state_reader='process')
    assert run.result['completed'], run.result
    attempts = [json.loads(line) for line in (run.out / 'timing.jsonl').read_text().splitlines()][:-1]
    solved = [a for a in attempts if a['outcome'] == 'solved']
    assert solved
    for attempt in solved:
        spans = {s['id']: s for s in attempt['spans']}
        published = [s for s in attempt['spans'] if s['phase'] == 'state.publish']
        # Nested, never a top-level phase: a stop crossing the preview republishes inside policy.prepare.
        assert published and all(s['parent'] is not None for s in published)
        assert any(spans[s['parent']]['phase'] == 'publication' for s in published)
        assert {s['robot_id'] for s in published} == {'duck0'}
