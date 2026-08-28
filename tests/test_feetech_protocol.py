"""Feetech SCS/STS wire protocol.

The arm backend needs hardware; the framing does not, and framing is where the
silent-corruption bugs live. A wrong checksum polynomial or a byte-order slip
does not raise -- it returns a plausible WRONG joint angle, which the safety
harness then trusts. So every frame this repo can emit is pinned byte for byte
against hand-computed expectations.
"""

from __future__ import annotations

import pytest

from cascade.control.feetech import (
    ADDR_GOAL_POSITION,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    BROADCAST_ID,
    INSTR_PING,
    INSTR_READ,
    INSTR_SYNC_WRITE,
    INSTR_WRITE,
    SCS,
    STS,
    FeetechError,
    ServoBus,
    build_packet,
    build_read,
    build_sync_write,
    build_write,
    checksum,
    parse_response,
    sign_magnitude,
)


class FakePort:
    """Minimal stand-in for a pyserial Serial."""

    def __init__(self, responses: list[bytes] | None = None):
        self.written: list[bytes] = []
        self._inbox = list(responses or [])
        self._buf = b""
        self.resets = 0

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        # A real servo answers each request; queue the next scripted reply.
        if self._inbox:
            self._buf += self._inbox.pop(0)
        return len(data)

    def read(self, n: int) -> bytes:
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def reset_input_buffer(self) -> None:
        self.resets += 1
        self._buf = b""


def frame(servo_id: int, params: bytes = b"", err: int = 0) -> bytes:
    """Build a well-formed RESPONSE the way a servo would."""
    body = bytes([servo_id, len(params) + 2, err]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


# ── framing ──────────────────────────────────────────────────────────────


def test_checksum_is_the_inverted_sum():
    assert checksum(bytes([1, 2, 3])) == (~6) & 0xFF == 0xF9


def test_ping_frame_matches_the_documented_bytes():
    # FF FF 01 02 01 FB : id 1, len 2, PING, checksum ~(1+2+1)
    assert build_packet(1, INSTR_PING) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_read_frame_carries_address_and_width():
    pkt = build_read(2, ADDR_PRESENT_POSITION)
    assert pkt[:2] == b"\xff\xff"
    assert pkt[2] == 2  # id
    assert pkt[3] == 4  # len = 2 params + 2
    assert pkt[4] == INSTR_READ
    assert pkt[5:7] == bytes([ADDR_PRESENT_POSITION[0], 2])
    assert pkt[7] == checksum(pkt[2:7])


def test_write_is_little_endian_for_sts_and_big_for_scs():
    """The whole reason `Endian` exists: the wrong order yields 0x0100 for
    0x0001, i.e. a joint 57 degrees from where it actually is."""
    sts = build_write(1, ADDR_GOAL_POSITION, 0x0102, STS)
    scs = build_write(1, ADDR_GOAL_POSITION, 0x0102, SCS)
    assert sts[6:8] == b"\x02\x01"
    assert scs[6:8] == b"\x01\x02"


def test_single_byte_write_has_no_byte_order_ambiguity():
    pkt = build_write(3, ADDR_TORQUE_ENABLE, 1)
    assert pkt[5:7] == bytes([ADDR_TORQUE_ENABLE[0], 1])
    assert pkt[3] == 4


def test_register_width_travels_with_the_address():
    """The builders take the (address, width) control-table entry, so a caller
    cannot pair an address with the wrong width."""
    assert ADDR_GOAL_POSITION[1] == 2 and ADDR_TORQUE_ENABLE[1] == 1
    assert len(build_write(1, ADDR_GOAL_POSITION, 0)) == \
        len(build_write(1, ADDR_TORQUE_ENABLE, 0)) + 1


def test_sync_write_addresses_the_broadcast_id_with_a_length_per_servo():
    pkt = build_sync_write(ADDR_GOAL_POSITION, {1: 0x0102, 2: 0x0304}, STS)
    assert pkt[2] == BROADCAST_ID
    assert pkt[4] == INSTR_SYNC_WRITE
    # params: addr, width, then (id, lo, hi) per servo
    assert pkt[5:7] == bytes([ADDR_GOAL_POSITION[0], 2])
    assert pkt[7:10] == bytes([1, 0x02, 0x01])
    assert pkt[10:13] == bytes([2, 0x04, 0x03])
    # Published SYNC_WRITE length: (bytes_per_servo + 1) * n_servos + 4.
    assert pkt[3] == (2 + 1) * 2 + 4
    assert pkt[3] == len(pkt) - 4  # LEN counts instr + params + checksum
    assert pkt[-1] == checksum(pkt[2:-1])


# ── response parsing ─────────────────────────────────────────────────────


def test_parse_round_trips_a_well_formed_frame():
    sid, err, params = parse_response(frame(5, b"\x34\x12"), expect_id=5)
    assert (sid, err, params) == (5, 0, b"\x34\x12")


def test_parse_rejects_a_corrupted_checksum():
    bad = bytearray(frame(1, b"\x00\x08"))
    bad[-1] ^= 0xFF
    with pytest.raises(FeetechError, match="checksum"):
        parse_response(bytes(bad))


@pytest.mark.parametrize("bad,match", [
    (b"", "short frame"),
    (b"\xff\xff\x01\x02", "short frame"),
    (b"\x00\x01\x02\x03\x04\x05", "bad header"),
])
def test_parse_rejects_malformed_frames(bad, match):
    with pytest.raises(FeetechError, match=match):
        parse_response(bad)


def test_parse_rejects_a_reply_from_the_wrong_servo():
    """An id collision (two servos left on the factory default) otherwise reads
    as one joint mirroring another."""
    with pytest.raises(FeetechError, match="expected 2"):
        parse_response(frame(1, b"\x00\x00"), expect_id=2)


def test_sign_magnitude_is_not_twos_complement():
    assert sign_magnitude(0x0064) == 100
    assert sign_magnitude(0x8064) == -100
    assert sign_magnitude(0x0000) == 0


# ── bus behaviour ────────────────────────────────────────────────────────


def test_read_reg_decodes_the_configured_byte_order():
    bus = ServoBus(FakePort([frame(1, b"\x34\x12")]), endian=STS)
    assert bus.read_reg(1, ADDR_PRESENT_POSITION) == 0x1234

    bus = ServoBus(FakePort([frame(1, b"\x34\x12")]), endian=SCS)
    assert bus.read_reg(1, ADDR_PRESENT_POSITION) == 0x3412


def test_servo_error_flags_are_raised_not_swallowed():
    """An overheat/overload flag is the difference between "tune it" and "this
    servo is about to fail"; it must not be reported as a good read."""
    bus = ServoBus(FakePort([frame(1, b"\x00\x00", err=0x04)]), retries=0)
    with pytest.raises(FeetechError, match="0x04"):
        bus.read_reg(1, ADDR_PRESENT_POSITION)


def test_a_short_read_is_not_silently_zero_extended():
    bus = ServoBus(FakePort([frame(1, b"\x07")]), retries=0)
    with pytest.raises(FeetechError, match="1 bytes for a 2-byte"):
        bus.read_reg(1, ADDR_PRESENT_POSITION)


def test_a_dropped_frame_is_retried_then_succeeds():
    """A single dropped frame on a shared half-duplex TTL bus is normal."""
    port = FakePort([b"", frame(1, b"\x01\x00")])
    bus = ServoBus(port, retries=2)
    assert bus.read_reg(1, ADDR_PRESENT_POSITION) == 1
    assert len(port.written) == 2, "should have re-sent the request"


def test_retries_are_bounded_and_then_raise():
    port = FakePort([b"", b"", b"", b""])
    bus = ServoBus(port, retries=2)
    with pytest.raises(FeetechError):
        bus.read_reg(1, ADDR_PRESENT_POSITION)
    assert len(port.written) == 3, "1 attempt + 2 retries"


def test_input_buffer_is_flushed_before_each_request():
    """Leftover bytes from a timed-out previous reply would otherwise be parsed
    as the head of this one."""
    port = FakePort([frame(1, b"\x00\x00")])
    ServoBus(port).read_reg(1, ADDR_PRESENT_POSITION)
    assert port.resets >= 1


def test_sync_write_expects_no_reply():
    """Broadcast frames are unanswered; waiting for one would stall the 50 Hz
    waypoint stream by a full read timeout per waypoint."""
    port = FakePort()  # nothing queued: a reply would deadlock a naive impl
    ServoBus(port).sync_write_reg(ADDR_GOAL_POSITION, {1: 100, 2: 200})
    assert len(port.written) == 1
    assert port.written[0][2] == BROADCAST_ID


def test_ping_reports_absence_as_false_not_an_exception():
    assert ServoBus(FakePort([]), retries=0).ping(9) is False
    assert ServoBus(FakePort([frame(3)]), retries=0).ping(3) is True
