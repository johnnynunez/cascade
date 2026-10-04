"""Bounded primitive transport for an opt-in Factory process boundary.

This module constructs no simulator and admits no physical observations. It is
the wire layer, not the native snapshot schema. Only inherited socketpairs are
used by the CPU integration; no listener or reconnect path is provided.
"""
from __future__ import annotations

import hashlib
import math
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum


class WireFault(RuntimeError):
    """A malformed, incomplete or late exchange is terminal for this channel."""


@dataclass(frozen=True, slots=True)
class WireLimits:
    """Explicit allocation limits; a native adapter must bind its own limits."""

    frame_bytes: int
    nodes: int
    depth: int
    text_bytes: int

    def __post_init__(self):
        for value, upper in ((self.frame_bytes, 16 * 1024 * 1024),
                             (self.nodes, 1_000_000), (self.depth, 32),
                             (self.text_bytes, 65536)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("invalid Factory wire limit")
        if self.text_bytes > self.frame_bytes:
            raise ValueError("text bound exceeds frame bound")


class Kind(IntEnum):
    HELLO = 1
    REQUEST = 2
    ACK = 3
    STOP = 4
    FENCE = 5
    RAW = 6
    OUTCOME = 7
    FAULT = 8
    CLOSE = 9


_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")
_F64 = struct.Struct("<d")
_HEADER = struct.Struct("<8sBQQ32s")
_MAGIC = b"CASCFP01"


def encode(value, limits: WireLimits) -> bytes:
    """Exact builtins only, retaining mapping order, tuple/list and float bits.

    There are no class names, reducers, imports or object construction hooks in
    this format. Reject cycles by the depth bound; never call an object's repr.
    """
    if type(limits) is not WireLimits:
        raise TypeError("explicit wire limits required")
    output = bytearray()
    count = 0

    def put(data):
        if len(output) + len(data) > limits.frame_bytes:
            raise WireFault("encoded frame exceeds byte bound")
        output.extend(data)

    def visit(item, depth):
        nonlocal count
        count += 1
        if count > limits.nodes or depth > limits.depth:
            raise WireFault("encoded structure exceeds bound")
        kind = type(item)
        if item is None:
            put(b"n")
        elif kind is bool:
            put(b"t" if item else b"f")
        elif kind is int:
            if not -(1 << 63) <= item < (1 << 63):
                raise WireFault("integer outside signed 64-bit range")
            put(b"i" + _I64.pack(item))
        elif kind is float:
            if not math.isfinite(item):
                raise WireFault("nonfinite wire float")
            put(b"r" + _F64.pack(item))
        elif kind in (str, bytes):
            # Check characters first so UTF-8 conversion is itself bounded.
            if kind is str and len(item) > limits.text_bytes:
                raise WireFault("text exceeds bound")
            try:
                data = item.encode("utf-8", errors="strict") if kind is str else item
            except UnicodeError as error:
                raise WireFault("invalid wire UTF-8") from error
            if len(data) > (limits.text_bytes if kind is str else limits.frame_bytes):
                raise WireFault("scalar bytes exceed bound")
            put((b"s" if kind is str else b"b") + _U32.pack(len(data)))
            put(data)
        elif kind in (tuple, list, dict):
            if len(item) > limits.nodes - count:
                raise WireFault("container exceeds remaining node bound")
            put({tuple: b"u", list: b"l", dict: b"d"}[kind] + _U32.pack(len(item)))
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise WireFault("wire mapping keys must be exact strings")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        else:
            raise WireFault("unsupported wire type")

    visit(value, 0)
    return bytes(output)


def decode(payload: bytes, limits: WireLimits):
    """Bound allocations before each read; require one complete canonical tree."""
    if type(limits) is not WireLimits or type(payload) is not bytes:
        raise TypeError("exact bytes and limits required")
    if len(payload) > limits.frame_bytes:
        raise WireFault("received frame exceeds byte bound")
    offset = count = 0

    def take(size):
        nonlocal offset
        if size > len(payload) - offset:
            raise WireFault("truncated wire value")
        start = offset
        offset += size
        return payload[start:offset]

    def visit(depth):
        nonlocal count
        count += 1
        if count > limits.nodes or depth > limits.depth:
            raise WireFault("received structure exceeds bound")
        tag = take(1)
        if tag == b"n":
            return None
        if tag in (b"t", b"f"):
            return tag == b"t"
        if tag == b"i":
            return _I64.unpack(take(8))[0]
        if tag == b"r":
            value = _F64.unpack(take(8))[0]
            if not math.isfinite(value):
                raise WireFault("nonfinite received float")
            return value
        if tag in (b"b", b"s", b"u", b"l", b"d"):
            size = _U32.unpack(take(4))[0]
            if tag in (b"b", b"s"):
                if size > (limits.text_bytes if tag == b"s" else limits.frame_bytes):
                    raise WireFault("received scalar exceeds bound")
                data = take(size)
                try:
                    return data.decode("utf-8", errors="strict") if tag == b"s" else data
                except UnicodeError as error:
                    raise WireFault("invalid wire UTF-8") from error
            children = size * (2 if tag == b"d" else 1)
            if children > min(limits.nodes - count, len(payload) - offset):
                raise WireFault("received container exceeds remaining bound")
            if tag == b"d":
                result = {}
                for _ in range(size):
                    key = visit(depth + 1)
                    if type(key) is not str or key in result:
                        raise WireFault("invalid or duplicate wire key")
                    result[key] = visit(depth + 1)
                return result
            result = [visit(depth + 1) for _ in range(size)]
            return tuple(result) if tag == b"u" else result
        raise WireFault("unknown wire tag")

    result = visit(0)
    if offset != len(payload):
        raise WireFault("trailing wire bytes")
    return result


def remaining(deadline: float) -> float:
    if type(deadline) is not float or not math.isfinite(deadline):
        raise ValueError("finite absolute monotonic deadline required")
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("original Factory transport deadline expired")
    return value


class FrameChannel:
    """One ordered stream in each direction, with no background feeder queue.

    Give bulk, admission and priority their own instances/socketpairs. Send and
    receive use independent locks; priority must never share the bulk instance.
    Any partial/late/malformed exchange closes this channel and remains sticky.
    Callers must propagate that fault to the owning controller's real stop path.
    """

    def __init__(self, channel: socket.socket, limits: WireLimits):
        if type(channel) is not socket.socket or type(limits) is not WireLimits:
            raise TypeError("socket and explicit limits required")
        if channel.family != socket.AF_UNIX or channel.type != socket.SOCK_STREAM:
            raise ValueError("inherited Unix stream socket required")
        self._socket = channel
        self._socket.setblocking(False)
        self.limits = limits
        self._state = threading.Lock()
        self._sending = threading.Lock()
        self._receiving = threading.Lock()
        self._fault = None
        self._sent = self._received = 0

    def _check(self):
        with self._state:
            if self._fault is not None:
                raise WireFault(self._fault)

    def close(self, reason="Factory channel closed"):
        with self._state:
            if self._fault is None:
                self._fault = reason
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._socket.close()

    def _wait(self, writing, deadline):
        self._check()
        timeout = remaining(deadline)
        read = [] if writing else [self._socket]
        write = [self._socket] if writing else []
        select.select(read, write, [], timeout)
        self._check()
        remaining(deadline)

    def _read(self, size, deadline):
        result = bytearray()
        while len(result) < size:
            self._wait(False, deadline)
            try:
                part = self._socket.recv(min(size - len(result), 65536))
            except (BlockingIOError, InterruptedError):
                continue
            if not part:
                raise EOFError("Factory peer closed a frame")
            result.extend(part)
        return bytes(result)

    def send(self, kind: Kind, value, *, deadline: float) -> int:
        acquired = False
        try:
            self._check()
            remaining(deadline)
            if type(kind) is not Kind:
                raise WireFault("exact frame kind required")
            payload = encode(value, self.limits)
            acquired = self._sending.acquire(timeout=remaining(deadline))
            if not acquired:
                raise TimeoutError("send lock exceeded original deadline")
            self._check()
            sequence = self._sent + 1
            header = _HEADER.pack(_MAGIC, kind, sequence, len(payload), hashlib.sha256(payload).digest())
            for part in (header, payload):
                view = memoryview(part)
                while view:
                    self._wait(True, deadline)
                    try:
                        sent = self._socket.send(view)
                    except (BlockingIOError, InterruptedError):
                        continue
                    if not sent:
                        raise EOFError("Factory peer stopped receiving")
                    view = view[sent:]
            self._check()
            remaining(deadline)
            self._sent = sequence
            return sequence
        except BaseException:
            self.close("Factory send failed")
            raise
        finally:
            if acquired:
                self._sending.release()

    def receive(self, *, deadline: float):
        acquired = False
        try:
            self._check()
            acquired = self._receiving.acquire(timeout=remaining(deadline))
            if not acquired:
                raise TimeoutError("receive lock exceeded original deadline")
            header = self._read(_HEADER.size, deadline)
            magic, tag, sequence, size, digest = _HEADER.unpack(header)
            if magic != _MAGIC or sequence != self._received + 1 or size > self.limits.frame_bytes:
                raise WireFault("invalid frame identity, sequence or byte count")
            try:
                kind = Kind(tag)
            except ValueError as error:
                raise WireFault("unknown frame kind") from error
            payload = self._read(size, deadline)
            if hashlib.sha256(payload).digest() != digest:
                raise WireFault("frame digest mismatch")
            value = decode(payload, self.limits)
            self._check()
            remaining(deadline)
            self._received = sequence
            return kind, value
        except BaseException:
            self.close("Factory receive failed")
            raise
        finally:
            if acquired:
                self._receiving.release()
