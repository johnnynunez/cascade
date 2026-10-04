"""CPU process boundaries only: no SDK imports, model or synthetic task credit."""
import gc
import hashlib
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from cascade.sim.factory_process_protocol import FaultState, FencedDecoder, RawOutbox
from cascade.sim.factory_process_wire import (
    FrameChannel,
    Kind,
    WireFault,
    WireLimits,
    decode,
    encode,
)

LIMITS = WireLimits(65536, 8192, 12, 512)
IDENTITY = ("a" * 32, "b" * 32, "c" * 64)


def fault():
    return FaultState(boot_id=IDENTITY[0], epoch=IDENTITY[1], binding=IDENTITY[2])


def deadline():
    return time.monotonic() + 3.


def test_wire_preserves_exact_primitive_types_order_and_float_bits():
    value = {"second": (0., -0., 2**-1074), "first": [True, False, None, -2**63, 2**63-1],
             "bytes": b"\x00\xff", "unicode": "cámara"}
    copied = decode(encode(value, LIMITS), LIMITS)
    assert list(copied) == list(value)
    assert type(copied["second"]) is tuple and type(copied["first"]) is list
    assert [struct.pack("<d", x) for x in copied["second"]] == [struct.pack("<d", x) for x in value["second"]]
    assert encode(copied, LIMITS) == encode(value, LIMITS)
    copied["first"].append(8)
    assert len(value["first"]) == 5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**63, -(2**63)-1,
                                   {1: "wrong key"}, object(), "\ud800"])
def test_wire_refuses_unknown_nonfinite_or_out_of_range_values(value):
    with pytest.raises(WireFault):
        encode(value, LIMITS)


def test_wire_never_uses_subclass_hooks_or_repr():
    class Hostile(dict):
        def items(self):
            pytest.fail("subclass execution")

        def __repr__(self):
            pytest.fail("repr execution")

    with pytest.raises(WireFault):
        encode(Hostile(), LIMITS)


@pytest.mark.parametrize("data", [b"x", b"nextra", b"l\xff\xff\xff\xff", b"s\x02\0\0\0\xff\xff",
    b"d\x02\0\0\0s\x01\0\0\0ans\x01\0\0\0an",
    b"r" + struct.pack("<d", float("inf"))])
def test_decoder_refuses_malformed_trees_before_unbounded_allocation(data):
    with pytest.raises(WireFault):
        decode(data, LIMITS)


def test_codec_bounds_depth_nodes_and_bytes_on_both_sides():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(WireFault):
        encode(cyclic, LIMITS)
    small = WireLimits(16, 4, 2, 4)
    for value in (b"x" * 20, [None] * 4, "abcde", [[[None]]]):
        with pytest.raises(WireFault):
            encode(value, small)
        with pytest.raises(WireFault):
            decode(encode(value, LIMITS), small)


def test_outbox_archive_ack_cannot_reuse_pending_reader_or_outcome_slot():
    box = RawOutbox(2, 8, 4, fault())
    first = box.append_raw(b"one")
    second = box.append_raw(b"two")
    snapshot = box.prefix(2)
    box.acknowledge(first, box.ARCHIVED)
    box.finish(first, accepted=True)
    box.acknowledge(first, box.OUTCOME_SEEN)
    assert box.prefix(2)[0][0] == first  # still leased by the reader
    box.acknowledge(first, box.READER_DONE)
    third = box.append_raw(b"new")
    assert third == 3 and [x[0] for x in box.prefix(2)] == [second, third]
    assert snapshot == ((1, b"one", 0), (2, b"two", 0))


def test_outbox_prefix_does_not_chase_a_producer_replenishing_between_writes():
    box = RawOutbox(4, 64, 16, fault())
    first = box.append_raw(b"one")
    box.finish(first, accepted=True)
    captured = box.prefix(4)
    written = []
    for seq, payload, outcome in captured:
        written.append(payload)
        for who in (box.ARCHIVED, box.OUTCOME_SEEN, box.READER_DONE):
            box.acknowledge(seq, who)
        new = box.append_raw(b"new")
        box.finish(new, accepted=True)
    assert written == [b"one"] and box.prefix(4) == ((2, b"new", box.ACCEPTED),)


@pytest.mark.parametrize("raised", [False, True])
def test_transaction_archives_before_actual_accept_and_records_its_return_or_raise(raised):
    state = fault()
    box = RawOutbox(2, 64, 32, state)
    calls = []
    original = RuntimeError("original age veto")

    def accept():
        calls.append(True)
        assert box.prefix(1) == ((1, b"raw", box.PENDING),)
        if raised:
            raise original
        return "accepted-result"

    if raised:
        with pytest.raises(RuntimeError) as error:
            box.transact(b"raw", accept)
        assert error.value is original
    else:
        assert box.transact(b"raw", accept) == (1, "accepted-result")
    assert calls == [True]
    assert box.prefix(1) == ((1, b"raw", box.REJECTED if raised else box.ACCEPTED),)


def test_fault_after_accept_return_does_not_relabel_outcome_but_refuses_delivery():
    state = fault()
    box = RawOutbox(2, 64, 32, state)
    seq = box.append_raw(b"raw")
    state.fail("concurrent fault after acceptance returned")
    box.finish(seq, accepted=True)
    assert box.prefix(1) == ((1, b"raw", box.ACCEPTED),)
    reader = FencedDecoder(IDENTITY, lambda n, d: state.status(n))
    with pytest.raises(WireFault):
        reader.read(b"raw", box.ACCEPTED, decode=bytes, validate=lambda _: None, deadline=deadline())


@pytest.mark.parametrize("which", ["count", "bytes", "early_ack", "duplicate", "missing_outcome"])
def test_outbox_incomplete_overflow_and_protocol_errors_are_sticky(which):
    state = fault()
    box = RawOutbox(1 if which == "count" else 2, 4, 4, state)
    seq = box.append_raw(b"1234")
    with pytest.raises(WireFault):
        if which in ("count", "bytes"):
            box.append_raw(b"x")
        elif which == "early_ack":
            box.acknowledge(seq, box.READER_DONE)
        elif which == "duplicate":
            box.finish(seq, accepted=True)
            box.finish(seq, accepted=True)
        else:
            box.end_of_stream()
    with pytest.raises(WireFault):
        state.check()
    assert box.prefix(1)[0][:2] == (1, b"1234")


def test_rejected_terminal_raw_remains_archivable_but_not_deliverable():
    state = fault()
    box = RawOutbox(2, 10, 5, state)
    seq = box.append_raw(b"fault")
    box.finish(seq, accepted=False)
    assert box.prefix(1) == ((seq, b"fault", box.REJECTED),)
    reader = FencedDecoder(IDENTITY, lambda n, d: state.status(n))
    with pytest.raises(WireFault):
        reader.read(b"fault", box.REJECTED, decode=lambda _: pytest.fail("decode rejected raw"),
                    validate=lambda _: None, deadline=deadline())
    for who in (box.ARCHIVED, box.OUTCOME_SEEN, box.READER_DONE):
        box.acknowledge(seq, who)
    box.end_of_stream()
    box.require_drained()
    with pytest.raises(WireFault):
        state.check()


def test_end_requires_drained_consumers_and_prohibits_reopen():
    box = RawOutbox(2, 10, 5, fault())
    seq = box.append_raw(b"row")
    box.finish(seq, accepted=True)
    box.end_of_stream()
    with pytest.raises(WireFault):
        box.require_drained()
    with pytest.raises(WireFault):
        box.append_raw(b"late")


def test_fence_rechecks_remote_fault_after_decode():
    state = fault()
    reader = FencedDecoder(IDENTITY, lambda n, d: state.status(n))

    def decode_then_fault(data):
        state.fail("fault while host decoded")
        return data

    with pytest.raises(WireFault):
        reader.read(b"row", 1, decode=decode_then_fault, validate=lambda _: None, deadline=deadline())


def test_fault_after_second_child_fence_belongs_to_next_read():
    state = fault()
    calls = []

    def status(nonce, deadline):
        result = state.status(nonce)
        calls.append(nonce)
        if len(calls) == 2:
            state.fail("after delivery linearization")
        return result

    reader = FencedDecoder(IDENTITY, status)
    assert reader.read(b"row", 1, decode=bytes, validate=lambda _: None, deadline=deadline()) == b"row"
    assert calls[0] != calls[1]
    with pytest.raises(WireFault):
        reader.read(b"next", 1, decode=bytes, validate=lambda _: None, deadline=deadline())


def test_nonce_replay_and_decode_deadline_cannot_renew_a_read():
    state = fault()
    cached = state.status(b"0" * 16)
    reader = FencedDecoder(IDENTITY, lambda n, d: cached)
    with pytest.raises(WireFault):
        reader.read(b"row", 1, decode=bytes, validate=lambda _: None, deadline=deadline())
    reader = FencedDecoder(IDENTITY, lambda n, d: state.status(n))
    with pytest.raises(TimeoutError):
        reader.read(b"row", 1, decode=lambda b: time.sleep(.02) or b,
                    validate=lambda _: None, deadline=time.monotonic() + .01)


@pytest.mark.parametrize("phase", ["decode", "final_fence"])
def test_waiting_read_timeout_revokes_an_active_read_before_local_delivery(phase):
    state = fault()
    entered = threading.Event()
    release = threading.Event()
    errors, delivered = [], []
    requests = []

    def block():
        entered.set()
        assert release.wait(.15)

    def status(nonce, deadline):
        requests.append(nonce)
        reply = state.status(nonce)
        if len(requests) == 2 and phase == "final_fence":
            block()
        return reply

    def decode_row(raw):
        if phase == "decode":
            block()
        return raw

    reader = FencedDecoder(IDENTITY, status)

    def first():
        try:
            delivered.append(reader.read(b"first", 1, decode=decode_row,
                                         validate=lambda _: None, deadline=time.monotonic() + .2))
        except (WireFault, TimeoutError) as error:
            errors.append(error)

    worker = threading.Thread(target=first)
    worker.start()
    try:
        assert entered.wait(.15)
        with pytest.raises(TimeoutError):
            reader.read(b"second", 1, decode=bytes, validate=lambda _: None,
                        deadline=time.monotonic() + .01)
        release.set()
        worker.join(.15)
        assert not worker.is_alive() and not delivered
        assert len(errors) == 1 and type(errors[0]) is WireFault
    finally:
        release.set()
        worker.join(1)


@contextmanager
def child_process(script, *sockets):
    """Exec a fresh CPU interpreter; all fds are explicitly owned by this test."""
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join((str(repo / "src"), str(repo / "tests"))))
    proc = subprocess.Popen([sys.executable, "-c", script, *[str(s.fileno()) for s in sockets]],
                            pass_fds=tuple(s.fileno() for s in sockets), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for sock in sockets:
        sock.close()
    try:
        yield proc
        out, error = proc.communicate(timeout=4)
        assert proc.returncode == 0, (out, error)
    finally:
        if proc.poll() is None:
            proc.kill()  # only the child created above; failed test, no success credit
            proc.communicate(timeout=4)


def test_actual_exec_child_primitive_exchange_and_gc_settings_unchanged():
    a, b = socket.socketpair()
    channel = FrameChannel(a, LIMITS)
    before = (gc.isenabled(), gc.get_threshold())
    script = '''
import gc, socket, sys, time
from cascade.sim.factory_process_wire import FrameChannel, Kind, WireLimits
c=FrameChannel(socket.socket(fileno=int(sys.argv[1])), WireLimits(65536,8192,12,512))
original=(gc.isenabled(),gc.get_threshold())
k,v=c.receive(deadline=time.monotonic()+3.)
assert k is Kind.HELLO and v=={'zero':-0.,'ordered':('one','two')}
c.send(Kind.ACK, {'value':v,'gc':original,'same_gc':original==(gc.isenabled(),gc.get_threshold())},deadline=time.monotonic()+3.)
c.close()
'''
    try:
        with child_process(script, b):
            channel.send(Kind.HELLO, {"zero": -0., "ordered": ("one", "two")}, deadline=deadline())
            kind, value = channel.receive(deadline=deadline())
            assert kind is Kind.ACK and value["same_gc"] is True
            assert struct.pack("<d", value["value"]["zero"]) == struct.pack("<d", -0.)
            assert value["gc"][0] is True
        assert before == (gc.isenabled(), gc.get_threshold())
    finally:
        channel.close()


def test_priority_stop_separate_socket_responds_while_bulk_sender_is_blocked():
    a, b = socket.socketpair()
    c, d = socket.socketpair()
    b.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    priority = FrameChannel(c, LIMITS)
    script = '''
import socket,sys,threading,time
from cascade.sim.factory_process_wire import FrameChannel, Kind, WireLimits
from test_factory_owner import owner_fixture
_,backend,owner=owner_fixture(active=False)
bulk=FrameChannel(socket.socket(fileno=int(sys.argv[1])),WireLimits(1048576,8192,12,512))
priority=FrameChannel(socket.socket(fileno=int(sys.argv[2])),WireLimits(65536,8192,12,512))
errors=[]
def blocked():
 try: bulk.send(Kind.RAW,b'x'*900000,deadline=time.monotonic()+3.)
 except BaseException as e: errors.append(type(e).__name__)
t=threading.Thread(target=blocked);t.start()
k,v=priority.receive(deadline=time.monotonic()+3.)
assert k is Kind.STOP and t.is_alive()
# Actual controller guard, synthetic backend: no SDK, solve or physical credit.
generation=owner.controller.generation
ack=owner.controller.stop()
assert ack['generation']==generation+1 and owner.controller.guard._latched
priority.send(Kind.ACK,{'latched':True,'generation':ack['generation'],'physical_stop':False},deadline=time.monotonic()+3.)
bulk.close();t.join(2);assert not t.is_alive() and errors
priority.close()
'''
    try:
        with child_process(script, b, d):
            priority.send(Kind.STOP, {"expected_generation": 0}, deadline=deadline())
            kind, result = priority.receive(deadline=deadline())
            assert kind is Kind.ACK and result == {"latched": True, "generation": 1, "physical_stop": False}
    finally:
        priority.close()
        a.close()


def test_real_child_fault_during_host_decode_refuses_second_authoritative_fence():
    a, b = socket.socketpair()
    channel = FrameChannel(a, LIMITS)
    script = '''
import socket,sys,time
from cascade.sim.factory_process_wire import FrameChannel,Kind,WireLimits
from cascade.sim.factory_process_protocol import FaultState
c=FrameChannel(socket.socket(fileno=int(sys.argv[1])),WireLimits(65536,8192,12,512))
state=FaultState(boot_id='a'*32,epoch='b'*32,binding='c'*64)
for _ in range(3):
 k,v=c.receive(deadline=time.monotonic()+3.)
 if k is Kind.FAULT:
  state.fail('child fault during parent decode')
  c.send(Kind.ACK,{'latched':True},deadline=time.monotonic()+3.)
 else:
  assert k is Kind.FENCE
  c.send(Kind.FENCE,state.status(v['nonce']),deadline=time.monotonic()+3.)
c.close()
'''
    try:
        with child_process(script, b):
            def status(nonce, original):
                channel.send(Kind.FENCE, {"nonce": nonce}, deadline=original)
                kind, value = channel.receive(deadline=original)
                assert kind is Kind.FENCE
                return value

            original = deadline()

            def decode_with_remote_fault(raw):
                channel.send(Kind.FAULT, None, deadline=original)
                assert channel.receive(deadline=original) == (Kind.ACK, {"latched": True})
                return raw

            reader = FencedDecoder(IDENTITY, status)
            with pytest.raises(WireFault, match="authoritative"):
                reader.read(b"row", 1, decode=decode_with_remote_fault,
                            validate=lambda _: None, deadline=original)
    finally:
        channel.close()


def test_actual_child_stall_keeps_original_capture_time_and_existing_age_refusal():
    from test_fastening_runtime import binding, limits, row

    from cascade.control.fastening import FasteningObservationAgeFault, check_solve

    a, b = socket.socketpair()
    channel = FrameChannel(a, LIMITS)
    script = '''
import socket,sys,time
from cascade.sim.factory_process_wire import FrameChannel,Kind,WireLimits
from cascade.sim.factory_process_protocol import FaultState
c=FrameChannel(socket.socket(fileno=int(sys.argv[1])),WireLimits(65536,8192,12,512))
state=FaultState(boot_id='a'*32,epoch='b'*32,binding='c'*64)
captured=time.monotonic()
time.sleep(.21)  # Controlled CPU stall; no collector or model manipulation.
c.send(Kind.RAW, {'captured':captured},deadline=time.monotonic()+3.)
k,v=c.receive(deadline=time.monotonic()+3.)
assert k is Kind.FENCE
c.send(Kind.FENCE,state.status(v['nonce']),deadline=time.monotonic()+3.)
c.close()
'''
    try:
        with child_process(script, b):
            kind, value = channel.receive(deadline=deadline())
            assert kind is Kind.RAW
            measured = row(captured=value["captured"])

            def status(nonce, original):
                channel.send(Kind.FENCE, {"nonce": nonce}, deadline=original)
                kind, answer = channel.receive(deadline=original)
                assert kind is Kind.FENCE
                return answer

            reader = FencedDecoder(IDENTITY, status)
            with pytest.raises(FasteningObservationAgeFault) as error:
                reader.read(b"synthetic-row", 1, decode=lambda _: measured,
                            validate=lambda v: check_solve(v, binding(), limits(), time.monotonic()),
                            deadline=deadline())
            assert error.value.observation_age["captured_monotonic_s"] == value["captured"]
            assert error.value.observation_age["age_s"] > .2
    finally:
        channel.close()


@pytest.mark.parametrize("malformation", ["length", "sequence", "digest", "eof"])
def test_frame_refusal_is_sticky_and_does_not_allocate_declared_oversize(malformation):
    a, b = socket.socketpair()
    channel = FrameChannel(a, LIMITS)
    payload = b"n"
    try:
        header = struct.pack("<8sBQQ32s", b"CASCFP01", Kind.RAW,
                             2 if malformation == "sequence" else 1,
                             2**63 if malformation == "length" else len(payload),
                             b"0" * 32 if malformation == "digest" else hashlib.sha256(payload).digest())
        b.sendall(header if malformation == "eof" else header + payload)
        b.close()
        with pytest.raises((WireFault, EOFError)):
            channel.receive(deadline=deadline())
        with pytest.raises(WireFault):
            channel.receive(deadline=deadline())
    finally:
        channel.close()
        b.close()


def test_waiting_reader_deadline_is_sticky_without_peer_activity():
    a, b = socket.socketpair()
    channel = FrameChannel(a, LIMITS)
    try:
        with pytest.raises(TimeoutError):
            channel.receive(deadline=time.monotonic() + .01)
        with pytest.raises(WireFault):
            channel.send(Kind.HELLO, None, deadline=deadline())
    finally:
        channel.close()
        b.close()
