"""Feetech SCS/STS servo-bus protocol (STS3215 and relatives).

Split out of `feetech_arm.py` on purpose: the wire format is pure byte
manipulation, so it can be tested exhaustively against a fake port
(`tests/test_feetech_protocol.py`) even though nobody here has the servos. The
arm backend on top of it is the part that genuinely needs hardware, and keeping
the two apart means a framing bug cannot hide behind "untestable".

PROTOCOL (identical in shape to Dynamixel 1.0, different register map):

    request   FF FF  ID  LEN  INSTR  PARAM...  CHK
    response  FF FF  ID  LEN  ERR    PARAM...  CHK

    LEN = len(params) + 2
    CHK = (~(ID + LEN + INSTR + sum(params))) & 0xFF

Broadcast ID 0xFE addresses every servo and is answered by none.

REGISTER MAP: the addresses below are the published STS3215 control table. They
are NOT verified against hardware in this repo -- if a read returns nonsense,
suspect these first, and check the model's own datasheet: Feetech ships several
control tables and the SCS series differs from the STS series in both addresses
and byte order (see `Endian`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BROADCAST_ID = 0xFE

# Instructions
INSTR_PING = 0x01
INSTR_READ = 0x02
INSTR_WRITE = 0x03
INSTR_SYNC_WRITE = 0x83

# STS3215 control table (address, byte width)
ADDR_ID = (0x05, 1)
ADDR_MODE = (0x21, 1)
ADDR_TORQUE_ENABLE = (0x28, 1)
ADDR_ACCELERATION = (0x29, 1)
ADDR_GOAL_POSITION = (0x2A, 2)
ADDR_GOAL_SPEED = (0x2E, 2)
ADDR_TORQUE_LIMIT = (0x30, 2)
ADDR_PRESENT_POSITION = (0x38, 2)
ADDR_PRESENT_SPEED = (0x3A, 2)
ADDR_PRESENT_LOAD = (0x3C, 2)
ADDR_PRESENT_VOLTAGE = (0x3E, 1)
ADDR_PRESENT_TEMPERATURE = (0x3F, 1)

MODE_POSITION = 0
#: Full-scale torque limit register value.
TORQUE_LIMIT_MAX = 1000


class FeetechError(RuntimeError):
    """A bus-level failure: bad checksum, timeout, short frame, servo error."""


@dataclass(frozen=True)
class Endian:
    """Multi-byte register order.

    STS servos are little-endian; the older SCS series is big-endian. Reading a
    position with the wrong one does not fail, it returns a plausible WRONG
    number (0x0100 for 0x0001), which is exactly the kind of silent error that
    ends up interpreted as a joint 57 degrees away from where it is.
    """

    little: bool = True

    def pack(self, value: int, width: int) -> bytes:
        return int(value).to_bytes(width, "little" if self.little else "big", signed=False)

    def unpack(self, data: bytes) -> int:
        return int.from_bytes(data, "little" if self.little else "big", signed=False)


STS = Endian(little=True)
SCS = Endian(little=False)


def checksum(payload: bytes) -> int:
    """`payload` is everything after the two 0xFF header bytes, minus the sum byte."""
    return (~sum(payload)) & 0xFF


def build_packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    body = bytes([servo_id & 0xFF, len(params) + 2, instruction]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


#: A control-table entry: (address, byte width). Every builder below takes one
#: of these rather than a loose address, so an address can never be paired with
#: the wrong width -- a 1-byte write to a 2-byte register leaves the high byte
#: holding whatever was there before, which reads back as a wildly wrong value.
Register = tuple[int, int]


def build_read(servo_id: int, reg: Register) -> bytes:
    address, width = reg
    return build_packet(servo_id, INSTR_READ, bytes([address, width]))


def build_write(servo_id: int, reg: Register, value: int,
                endian: Endian = STS) -> bytes:
    address, width = reg
    return build_packet(servo_id, INSTR_WRITE,
                        bytes([address]) + endian.pack(value, width))


def build_sync_write(reg: Register, values: dict[int, int],
                     endian: Endian = STS) -> bytes:
    """One frame that sets the same register on many servos.

    This is what makes 50 Hz streaming possible: N individual writes are N USB
    round trips, and at ~1 ms of host latency each that alone would blow the
    20 ms waypoint budget on a 5-joint arm. SYNC_WRITE is one frame, unanswered.
    """
    address, width = reg
    params = bytearray([address, width])
    for sid, value in values.items():
        params.append(sid & 0xFF)
        params.extend(endian.pack(value, width))
    return build_packet(BROADCAST_ID, INSTR_SYNC_WRITE, bytes(params))


def parse_response(frame: bytes, expect_id: int | None = None) -> tuple[int, int, bytes]:
    """-> (servo_id, error_flags, params). Raises FeetechError on a bad frame."""
    if len(frame) < 6:
        raise FeetechError(f"short frame ({len(frame)} bytes): {frame.hex()}")
    if frame[0] != 0xFF or frame[1] != 0xFF:
        raise FeetechError(f"bad header: {frame[:2].hex()}")
    servo_id, length = frame[2], frame[3]
    end = 4 + length  # length counts err + params + checksum
    if len(frame) < end:
        raise FeetechError(f"truncated frame: need {end} bytes, got {len(frame)}")
    body = frame[2:end - 1]
    if frame[end - 1] != checksum(body):
        raise FeetechError(
            f"checksum mismatch on id {servo_id}: "
            f"got 0x{frame[end - 1]:02x}, computed 0x{checksum(body):02x}"
        )
    if expect_id is not None and servo_id != expect_id:
        raise FeetechError(f"response from id {servo_id}, expected {expect_id}")
    return servo_id, frame[4], bytes(frame[5:end - 1])


def sign_magnitude(raw: int, bits: int = 15) -> int:
    """Feetech speed/load registers are sign-magnitude, not two's complement."""
    sign_bit = 1 << bits
    return -(raw & (sign_bit - 1)) if raw & sign_bit else raw


class ServoBus:
    """Framed request/response over a serial port.

    Takes an already-open port object (anything with read/write/reset_input_buffer)
    so tests can substitute a fake and the backend owns the pyserial import.
    """

    def __init__(self, port, endian: Endian = STS, retries: int = 2):
        self._port = port
        self.endian = endian
        self.retries = max(0, int(retries))

    def _read_frame(self) -> bytes:
        # Header first, then the declared length: reading a fixed block would
        # either truncate a long response or block for the full timeout on a
        # short one.
        head = self._port.read(4)
        if len(head) < 4:
            raise FeetechError(f"no response (got {len(head)} of 4 header bytes)")
        rest = self._port.read(head[3])
        if len(rest) < head[3]:
            raise FeetechError(
                f"truncated body: expected {head[3]} bytes, got {len(rest)}"
            )
        return bytes(head) + bytes(rest)

    def _txrx(self, packet: bytes, expect_id: int) -> bytes:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self._port.reset_input_buffer()
                self._port.write(packet)
                _, err, params = parse_response(self._read_frame(), expect_id)
                if err:
                    # Bit 0 voltage, 1 angle, 2 overheat, 3 range, 4 checksum,
                    # 5 overload, 6 instruction. Reported, not swallowed: an
                    # overheat flag is the difference between "tune the gains"
                    # and "the servo is about to fail".
                    raise FeetechError(f"servo {expect_id} error flags 0x{err:02x}")
                return params
            except FeetechError as e:
                last = e
                if attempt < self.retries:
                    logger.debug("feetech retry %d on id %d: %s", attempt + 1, expect_id, e)
        raise FeetechError(str(last))

    def ping(self, servo_id: int) -> bool:
        try:
            self._txrx(build_packet(servo_id, INSTR_PING), servo_id)
            return True
        except FeetechError:
            return False

    def read_reg(self, servo_id: int, reg: Register) -> int:
        width = reg[1]
        params = self._txrx(build_read(servo_id, reg), servo_id)
        if len(params) != width:
            raise FeetechError(
                f"id {servo_id} returned {len(params)} bytes for a {width}-byte register"
            )
        return self.endian.unpack(params)

    def write_reg(self, servo_id: int, reg: Register, value: int) -> None:
        self._txrx(build_write(servo_id, reg, value, self.endian), servo_id)

    def sync_write_reg(self, reg: Register, values: dict[int, int]) -> None:
        """Unanswered by design: broadcast frames get no response."""
        self._port.write(build_sync_write(reg, values, self.endian))
