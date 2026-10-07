#!/usr/bin/env python3
"""Off-GIL ``state()`` reader server for a mobile bridge owner (standard library only).

Spawned by the owner (``cascade.sim.mobile_state_offload.StateServerProcess``) as
``sys.executable -I <this file> --control-fd N``. Isolated mode keeps the owner's paths out, so
this file cannot import CASCADE, NumPy or Kit: everything it needs arrives over the control
socket. It serves exactly one operation, ``state``, on reader channels the owner already
handshook and answered once, and it never mutates anything:

* Each robot has a slot in ONE shared-memory segment that the owner rewrites under its
  controller lock after every completed step and every permission change
  (``SlotSegment.write``): a seqlock (odd while writing), the publication count, the step, the
  controller's publish time (``_state_wall``) and generation, the robot and model identity, and
  a CRC32 over all of that plus the payload. Python cannot fence stores, so on a weakly ordered
  CPU (aarch64) a reader could see the even sequence number before the payload bytes; the CRC
  turns such a mixed read into a retried torn read. A publication older than one already served
  is never served either.
* The payload is ``marshal`` of the owner's own ``state()`` dict. ``StateRenderer`` re-encodes it
  exactly as the owner's ``json.dumps(..., allow_nan=False, separators=(",", ":"))`` with ONLY
  the two age fields (``state_age_s``, ``state.producer_age_s``) recomputed as
  ``max(0, now - publish time)`` on the system monotonic clock, the owner's own expression.
  Nothing is refreshed: a stalled owner serves a growing age and the clients' unchanged
  freshness checks trip exactly as before.
* Any other reader operation the owner serves (``frame``) hands the connection back to the owner
  with its pending request; the owner keeps it from then on.
* Control-socket EOF (owner exit or death), an owner ``close`` or an owner-closed slot ends
  every connection: fail closed. This is a transport for CPU copies of completed observations,
  never an acceptance or physics component.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import marshal
import math
import os
import socket
import struct
import sys
import threading
import time
import zlib
from multiprocessing import shared_memory

MAGIC = b"CASCST01"
LAYOUT_VERSION = 1
SEGMENT = struct.Struct("<8sIIQQ")  # magic, layout, slots, slot size, payload capacity
SEGMENT_BYTES = 64
SEQUENCE = struct.Struct("<Q")      # slot offset 0: odd while the owner writes
CHECKSUM = struct.Struct("<I")      # slot offset 8: CRC32 of meta + payload
META_OFFSET = 16
# status, marshal version, publication, step, publish monotonic (_state_wall, NaN = none),
# write monotonic, controller generation, payload length, reserved, robot id, model identity, reason
META = struct.Struct("<IIQqddqII64s64s128s")
PAYLOAD_OFFSET = META_OFFSET + META.size
EMPTY, PUBLISHED, UNAVAILABLE, CLOSED = 0, 1, 2, 3
CONTROL_HEADER = struct.Struct(">I")
MAX_CONTROL_BYTES = 1 << 20
IMPLEMENTATION = "shared-memory-seqlock-crc32-marshal-v1"

Record = collections.namedtuple("Record", "publication step state_wall written_at generation payload crc")


class SlotUnavailable(RuntimeError):
    """The slot cannot give a coherent current reply; the client receives an explicit refusal."""


def _identities(values):
    result = []
    for value in values:
        robot, model = value
        if (not isinstance(robot, str) or not robot or len(robot.encode()) > 64
                or not isinstance(model, str) or len(model) != 64
                or any(c not in "0123456789abcdef" for c in model)):
            raise ValueError("state slot identity needs a robot id (<= 64 bytes) and a model sha256")
        result.append((robot, model))
    if not result or len({robot for robot, _ in result}) != len(result):
        raise ValueError("state slots need distinct robot identities")
    return result


def _attach_untracked(name):
    """Attach without registering: the creator (owner) alone owns the segment's lifetime."""
    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)  # Python >= 3.13
    except TypeError:
        pass
    from multiprocessing import resource_tracker
    register = resource_tracker.register
    # Python 3.12 registers attached segments too, and a reader's tracker would unlink the
    # owner's segment when the reader exits.
    resource_tracker.register = lambda *args, **kwargs: None
    try:
        return shared_memory.SharedMemory(name=name, create=False)
    finally:
        resource_tracker.register = register


class SlotSegment:
    """One shared-memory segment holding one seqlocked, checksummed slot per robot."""

    READ_ATTEMPTS = 64

    def __init__(self, shm, identities, *, owner, slot_size, capacity):
        self._shm = shm
        self.buf = shm.buf
        self.name = shm.name
        self.identities = list(identities)
        self.owner = owner
        self.slot_size = slot_size
        self.capacity = capacity
        count = len(self.identities)
        self._sequences = [0] * count
        self._publications = [0] * count
        self._served = [0] * count
        self._locks = [threading.Lock() for _ in range(count)]
        self._closed = False

    @classmethod
    def create(cls, identities, *, payload_bytes):
        identities = _identities(identities)
        if type(payload_bytes) is not int or not 4096 <= payload_bytes <= 64 << 20:
            raise ValueError("state slot payload capacity must be 4 KiB..64 MiB")
        slot_size = -(-(PAYLOAD_OFFSET + payload_bytes) // 64) * 64
        shm = shared_memory.SharedMemory(create=True, size=SEGMENT_BYTES + slot_size * len(identities))
        try:
            SEGMENT.pack_into(shm.buf, 0, MAGIC, LAYOUT_VERSION, len(identities), slot_size, payload_bytes)
            segment = cls(shm, identities, owner=True, slot_size=slot_size, capacity=payload_bytes)
            for index in range(len(identities)):
                segment._publish(index, EMPTY, b"", reason="no state published yet")
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        return segment

    @classmethod
    def attach(cls, name, *, expected):
        expected = _identities(expected)
        shm = _attach_untracked(name)
        try:
            magic, layout, count, slot_size, capacity = SEGMENT.unpack_from(shm.buf, 0)
            if magic != MAGIC or layout != LAYOUT_VERSION:
                raise ValueError("unknown state slot layout")
            if count != len(expected):
                raise ValueError("state slot count differs from the configured robots")
            if slot_size < PAYLOAD_OFFSET + capacity or shm.size < SEGMENT_BYTES + count * slot_size:
                raise ValueError("state slot segment is truncated")
            segment = cls(shm, expected, owner=False, slot_size=slot_size, capacity=capacity)
            for index in range(count):
                meta = META.unpack(segment._coherent(index)[0])
                if segment._identity(meta) != expected[index]:
                    raise ValueError(f"state slot {index} identity differs from the configured robot")
        except BaseException:
            shm.close()
            raise
        return segment

    def _base(self, index):
        if type(index) is not int or not 0 <= index < len(self.identities):
            raise IndexError("unknown state slot")
        return SEGMENT_BYTES + index * self.slot_size

    def payload_offset(self, index):
        return self._base(index) + PAYLOAD_OFFSET

    def slot_bytes(self, index):
        base = self._base(index)
        return self.buf[base:base + self.slot_size]

    @staticmethod
    def _identity(meta):
        return (meta[9].rstrip(b"\0").decode(errors="replace"), meta[10].rstrip(b"\0").decode(errors="replace"))

    # -- owner side ---------------------------------------------------------------------------
    def _begin(self, index):
        sequence = self._sequences[index] + 1  # odd: readers retry
        SEQUENCE.pack_into(self.buf, self._base(index), sequence)
        return sequence

    def _commit(self, index, sequence):
        SEQUENCE.pack_into(self.buf, self._base(index), sequence + 1)
        self._sequences[index] = sequence + 1

    def _publish(self, index, status, payload, *, state_wall=None, step=None, generation=0, reason=""):
        with self._locks[index]:
            if self._closed:
                return False
            robot, model = self.identities[index]
            publication = self._publications[index] + 1
            meta = META.pack(status, marshal.version, publication, -1 if step is None else int(step),
                             math.nan if state_wall is None else float(state_wall), time.monotonic(),
                             int(generation), len(payload), 0, robot.encode(), model.encode(),
                             reason.encode()[:128])
            checksum = zlib.crc32(payload, zlib.crc32(meta))
            base = self._base(index)
            sequence = self._begin(index)
            self.buf[base + META_OFFSET:base + PAYLOAD_OFFSET] = meta
            self.buf[base + PAYLOAD_OFFSET:base + PAYLOAD_OFFSET + len(payload)] = payload
            CHECKSUM.pack_into(self.buf, base + 8, checksum)
            self._commit(index, sequence)
            self._publications[index] = publication
            return True

    def write(self, index, payload, *, state_wall, step, generation):
        """One complete publication; an oversized reply marks the slot unavailable (fail closed)."""
        if type(payload) is not bytes:
            raise TypeError("state slot payload must be bytes")
        if len(payload) > self.capacity:
            self.mark_unavailable(index, f"state reply of {len(payload)} bytes exceeds the "
                                         f"{self.capacity}-byte slot")
            return False
        return self._publish(index, PUBLISHED, payload, state_wall=state_wall, step=step, generation=generation)

    def mark_unavailable(self, index, reason):
        return self._publish(index, UNAVAILABLE, b"", reason=str(reason))

    def mark_closed(self, reason):
        for index in range(len(self.identities)):
            self._publish(index, CLOSED, b"", reason=str(reason))

    # -- reader side --------------------------------------------------------------------------
    def _coherent(self, index):
        """(meta bytes, payload, crc) of one whole publication, or SlotUnavailable."""
        base, buf = self._base(index), self.buf
        error = "state slot write in progress"
        for attempt in range(self.READ_ATTEMPTS):
            if attempt:
                time.sleep(0 if attempt < 8 else .0002)
            (first,) = SEQUENCE.unpack_from(buf, base)
            if first & 1:
                error = "state slot write in progress"
                continue
            meta = bytes(buf[base + META_OFFSET:base + PAYLOAD_OFFSET])
            length = META.unpack(meta)[7]
            if length > self.capacity:
                error = "torn state slot read (length)"
                continue
            payload = bytes(buf[base + PAYLOAD_OFFSET:base + PAYLOAD_OFFSET + length])
            (checksum,) = CHECKSUM.unpack_from(buf, base + 8)
            (second,) = SEQUENCE.unpack_from(buf, base)
            if second != first:
                error = "state slot write in progress"
                continue
            if zlib.crc32(payload, zlib.crc32(meta)) != checksum:
                error = "torn state slot read (CRC mismatch)"
                continue
            return meta, payload, checksum
        raise SlotUnavailable(error)

    def read(self, index):
        """The current publication: whole, of this robot and model, never older than one served."""
        error = "state slot publication regressed below one already served"
        for attempt in range(self.READ_ATTEMPTS):
            if attempt:
                time.sleep(.0002)
            meta, payload, checksum = self._coherent(index)
            (status, version, publication, step, wall, written, generation, _, _, _, _, reason) = META.unpack(meta)
            if self._identity(META.unpack(meta)) != self.identities[index]:
                raise SlotUnavailable("state slot identity differs from the configured robot and model")
            reason = reason.rstrip(b"\0").decode(errors="replace")
            if status == CLOSED:
                raise SlotUnavailable(f"owner closed the state slot: {reason}")
            if status == UNAVAILABLE:
                raise SlotUnavailable(f"owner could not publish this state: {reason}")
            if status != PUBLISHED:
                raise SlotUnavailable("no state published in the slot yet")
            if version != marshal.version:
                raise SlotUnavailable("state slot marshal format differs from this interpreter")
            with self._locks[index]:
                if publication < self._served[index]:
                    continue
                self._served[index] = publication
            return Record(publication, None if step < 0 else step, wall, written, generation, payload, checksum)
        raise SlotUnavailable(error)

    def close(self, *, unlink=False):
        if self._closed:
            return
        for lock in self._locks:
            lock.acquire()
        try:
            self._closed = True
        finally:
            for lock in self._locks:
                lock.release()
        self.buf = None
        try:
            self._shm.close()
        finally:
            if unlink:
                self._shm.unlink()


def encode(value):
    """The owner's exact reply bytes (MobileBridgeServer._send)."""
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"


_SENTINEL = "\0cascade-state-age\0"
_SENTINEL_JSON = json.dumps(_SENTINEL).encode()


def _set_age(reply, age):
    count = 0
    if "state_age_s" in reply:
        reply["state_age_s"] = age
        count += 1
    state = reply.get("state")
    if isinstance(state, dict) and "producer_age_s" in state:
        state["producer_age_s"] = age
        count += 1
    return count


class StateRenderer:
    """In-GIL reply bytes of a slot publication at one monotonic time.

    One JSON encode per publication with a sentinel in the two age fields; each poll then only
    formats the age (``float.__repr__``, json's own float formatting). A payload that already
    contains the sentinel text falls back to a full exact encode per poll.
    """

    def __init__(self):
        self._cache = None  # (publication, crc) -> byte segments around the ages, or None

    def _parts(self, record):
        reply = marshal.loads(record.payload)
        if not isinstance(reply, dict):
            raise ValueError("state slot payload is not a reply object")
        fields = _set_age(reply, _SENTINEL)
        parts = encode(reply).split(_SENTINEL_JSON)
        return parts if len(parts) == fields + 1 else None

    def render(self, record, now):
        key = (record.publication, record.crc)
        cache = self._cache
        if cache is None or cache[0] != key:
            cache = self._cache = (key, self._parts(record))
        age = None if math.isnan(record.state_wall) else max(0.0, now - record.state_wall)
        parts = cache[1]
        if parts is None:
            reply = marshal.loads(record.payload)
            _set_age(reply, age)
            return encode(reply)
        return (b"null" if age is None else float.__repr__(age).encode()).join(parts)


def decode_request(line):
    """MobileBridgeServer._decode: duplicate fields and non-finite constants refuse the stream."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(line, object_pairs_hook=pairs, parse_constant=bad_constant)


def send_message(sock, message, fd=None):
    data = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_CONTROL_BYTES:
        raise ValueError("control message too large")
    frame = CONTROL_HEADER.pack(len(data)) + data
    if fd is None:
        sock.sendall(frame)
        return
    sent = socket.send_fds(sock, [frame], [fd])
    if sent < len(frame):
        sock.sendall(frame[sent:])


def _recv_exact(sock, size, chunk=b""):
    data = bytearray(chunk)
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError("control socket closed mid-message")
        data.extend(part)
    return bytes(data)


def recv_message(sock):
    """(message, fd or None); (None, None) on a clean EOF between messages."""
    head, fds, flags, _ = socket.recv_fds(sock, CONTROL_HEADER.size, 1)
    fd = fds[0] if fds else None
    try:
        for extra in fds[1:]:
            os.close(extra)
        if flags & getattr(socket, "MSG_CTRUNC", 0):
            raise ValueError("control descriptor truncated")
        if not head:
            if fd is not None:
                raise ValueError("descriptor without a control message")
            return None, None
        (length,) = CONTROL_HEADER.unpack(_recv_exact(sock, CONTROL_HEADER.size, head))
        if length > MAX_CONTROL_BYTES:
            raise ValueError("control message too large")
        message = json.loads(_recv_exact(sock, length))
        if not isinstance(message, dict):
            raise ValueError("control message must be an object")
        return message, fd
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise


class StateServer:
    """Serve ``state`` from the slots; hand any other reader op back; die with the owner."""

    def __init__(self, control):
        self.control = control
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._connections = {}
        self.slots = None
        self.stats = []

    def configure(self):
        message, fd = recv_message(self.control)
        if fd is not None:
            os.close(fd)
        if message is None or message.get("op") != "config":
            raise ValueError("first control message must configure the state server")
        if list(message.get("python", ())) != list(sys.version_info[:2]) or message.get("marshal_version") != marshal.version:
            raise ValueError("state server interpreter differs from the owner")
        protocol = message["protocol"]
        self.max_request_bytes = int(protocol["max_request_bytes"])
        self.max_response_bytes = int(protocol["max_response_bytes"])
        self.io_timeout_s = float(protocol["io_timeout_s"])
        reader_ops = frozenset(protocol["reader_ops"])
        if "state" not in reader_ops:
            raise ValueError("state is not a reader operation")
        self.owner_ops = reader_ops - {"state"}
        self.forbidden = encode({"ok": False, "error": str(protocol["forbidden"])})
        robots = [tuple(item) for item in message["robots"]]
        self.slots = SlotSegment.attach(message["segment"], expected=robots)
        self.renderers = [StateRenderer() for _ in robots]
        self.stats = [dict(robot_id=robot, connections=0, served=0, refused=0, returned=0) for robot, _ in robots]

    def _message(self, message, fd=None):
        with self._send_lock:
            send_message(self.control, message, fd)

    def _count(self, index, key):
        with self._lock:
            self.stats[index][key] += 1

    def run(self):
        while not self._halt.is_set():
            message, fd = recv_message(self.control)
            if message is None or message.get("op") == "close":
                if fd is not None:
                    os.close(fd)
                return
            if message.get("op") != "adopt" or fd is None:
                if fd is not None:
                    os.close(fd)
                raise ValueError("unknown state server control message")
            index = message.get("robot")
            conn = socket.socket(fileno=fd)
            if type(index) is not int or not 0 <= index < len(self.stats):
                conn.close()
                raise ValueError("unknown robot slot")
            worker = threading.Thread(target=self._serve, args=(conn, index), name="state-reader", daemon=True)
            with self._lock:
                self._connections[conn] = worker
                self.stats[index]["connections"] += 1
            worker.start()

    def _state(self, index):
        try:
            record = self.slots.read(index)
        except SlotUnavailable as exc:
            self._count(index, "refused")
            return encode({"ok": False, "error": f"off-GIL state unavailable: {exc}"})
        wire = self.renderers[index].render(record, time.monotonic())
        self._count(index, "served")
        return wire

    def _send(self, conn, wire):
        if len(wire) > self.max_response_bytes:
            raise ValueError("response too large")
        conn.settimeout(self.io_timeout_s)
        conn.sendall(wire)

    def _serve(self, conn, index):
        """MobileBridgeServer._serve for a handshaken reader, minus everything but ``state``."""
        released = True
        buffer = bytearray()
        deadline = received = None
        try:
            while not self._halt.is_set():
                left = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
                if left <= 0:
                    raise ValueError("request wall-time deadline expired")
                conn.settimeout(left)
                try:
                    part = conn.recv(4096)
                except socket.timeout:
                    continue
                if not part:
                    return
                if deadline is None:
                    deadline = time.monotonic() + self.io_timeout_s
                    received = time.monotonic()
                buffer.extend(part)
                if len(buffer) > self.max_request_bytes:
                    raise ValueError("request too large")
                if b"\n" not in buffer:
                    continue
                line, extra = bytes(buffer).split(b"\n", 1)
                if extra:
                    raise ValueError("pipelined requests are not supported")
                request = decode_request(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                op = request.get("op")
                if op == "hello":
                    raise ValueError("channel already handshaken")
                if op == "state":
                    wire = self._state(index)
                elif op in self.owner_ops:
                    # The owner serves it, and this channel, from now on (same pending request,
                    # same first-byte deadline). Its connection slot stays held for it.
                    self._message({"op": "return", "robot": index, "deadline": deadline, "received_wall": received,
                                   "line": base64.b64encode(line + b"\n").decode("ascii")}, conn.fileno())
                    released = False
                    self._count(index, "returned")
                    return
                else:
                    wire = self.forbidden
                self._send(conn, wire)
                buffer.clear()
                deadline = None
        except (OSError, ValueError, TypeError, RecursionError):
            pass  # invalid framing poisons the stream, exactly like the owner
        finally:
            conn.close()
            with self._lock:
                self._connections.pop(conn, None)
            if released:
                try:
                    self._message({"op": "released", "robot": index})
                except (OSError, ValueError):
                    pass

    def shutdown(self):
        self._halt.set()
        with self._lock:
            connections = list(self._connections.items())
        for conn, _ in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = time.monotonic() + 2.
        for _, worker in connections:
            worker.join(max(0., deadline - time.monotonic()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="off-GIL state() reader server (spawned by the owner)")
    parser.add_argument("--control-fd", type=int, required=True)
    args = parser.parse_args(argv)
    control = socket.socket(fileno=args.control_fd)
    server = StateServer(control)
    code = 0
    try:
        try:
            server.configure()
        except (ValueError, KeyError, TypeError, OSError) as exc:
            send_message(control, {"op": "refused", "error": f"{type(exc).__name__}: {exc}"})
            return 2
        send_message(control, {"op": "ready", "pid": os.getpid(), "implementation": IMPLEMENTATION})
        try:
            server.run()
        except (ValueError, OSError) as exc:
            code = 3
            print(f"state server: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        server.shutdown()
        try:
            send_message(control, {"op": "stats", "stats": server.stats, "exit_code": code})
        except (OSError, ValueError):
            pass
        if server.slots is not None:
            server.slots.close()
        control.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
