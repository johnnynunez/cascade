"""Bounded PCM media boundary, independent of robot and inference providers."""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AudioFormat:
    encoding: str = "pcm_s16le"
    sample_rate_hz: int = 24000
    channels: int = 1

    def __post_init__(self):
        if (self.encoding, self.sample_rate_hz, self.channels) != ("pcm_s16le", 24000, 1):
            raise ValueError("this protocol requires mono PCM16 little-endian at 24000 Hz")

    def as_dict(self):
        return {"encoding": self.encoding, "sample_rate_hz": self.sample_rate_hz, "channels": self.channels}


@dataclass(frozen=True)
class AudioChunk:
    sequence: int
    pcm: bytes
    receipt_monotonic_s: float

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("invalid audio sequence")
        if not isinstance(self.pcm, bytes) or not 0 < len(self.pcm) <= 4800 or len(self.pcm) % 2:
            raise ValueError("audio chunk must contain 1–2400 PCM16 samples (at most 100 ms)")
        if not isinstance(self.receipt_monotonic_s, (float, int)) or not 0 <= self.receipt_monotonic_s < float("inf"):
            raise ValueError("invalid receipt clock")

    def encoded(self):
        return base64.b64encode(self.pcm).decode("ascii")


def decode_pcm(encoded: str, *, maximum_bytes=96000) -> bytes:
    if not isinstance(encoded, str) or len(encoded) > ((maximum_bytes + 2) // 3) * 4:
        raise ValueError("oversized PCM payload")
    data = base64.b64decode(encoded, validate=True)
    if not data or len(data) % 2 or len(data) > maximum_bytes:
        raise ValueError("invalid PCM16 payload")
    return data


class MediaIO(Protocol):
    """Capture timestamps are local receipt times, never browser clock guesses."""
    async def capture(self) -> AudioChunk: ...
    async def playback(self, pcm: bytes, *, response_id: str) -> None: ...
    async def flush(self, reason: str) -> None: ...
    async def emit(self, event: dict) -> None: ...
    async def close(self) -> None: ...


class QueueMediaIO:
    """One event-loop owner, bounded capture and playback queues.

    Queue pressure is an error: silently losing/reordering microphone samples or
    presenting an arbitrarily late spoken instruction is not permitted.
    """
    format = AudioFormat()

    def __init__(self, *, capacity=32, max_capture_age_s=.5):
        if not 1 <= capacity <= 128 or not 0 < max_capture_age_s <= 2:
            raise ValueError("invalid media queue limits")
        self.incoming = asyncio.Queue(maxsize=capacity)
        self.outgoing = asyncio.Queue(maxsize=capacity)
        self.max_capture_age_s = max_capture_age_s
        self.sequence = -1
        self.playback_generation = 0
        self.closed = False
        self._playback_bytes = 0

    def feed(self, sequence, pcm):
        if self.closed or sequence != self.sequence + 1:
            raise ValueError("closed media or replayed/out-of-order audio")
        chunk = AudioChunk(sequence, pcm, time.monotonic())
        self.incoming.put_nowait(chunk)
        self.sequence = sequence

    async def capture(self):
        chunk = await self.incoming.get()
        if time.monotonic() - chunk.receipt_monotonic_s > self.max_capture_age_s:
            raise ValueError("stale queued microphone audio")
        return chunk

    async def emit(self, event):
        if not self.closed:
            self.outgoing.put_nowait(event)

    async def playback(self, pcm, *, response_id):
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2 or len(pcm) > 96000:
            raise ValueError("invalid playback PCM")
        if self._playback_bytes + len(pcm) > 96000:
            raise ValueError("playback queue exceeds two seconds")
        await self.emit({"type": "audio", "generation": self.playback_generation,
                         "response_id": response_id, "audio": base64.b64encode(pcm).decode("ascii")})
        self._playback_bytes += len(pcm)

    async def receive(self):
        event = await self.outgoing.get()
        if event["type"] == "audio":
            self._playback_bytes -= len(base64.b64decode(event["audio"]))
        return event

    async def flush(self, reason):
        self.playback_generation += 1
        while not self.outgoing.empty():
            self.outgoing.get_nowait()
        self._playback_bytes = 0
        await self.emit({"type": "flush", "generation": self.playback_generation, "reason": reason})

    async def close(self):
        await self.flush("closed")
        self.closed = True
        while not self.incoming.empty():
            self.incoming.get_nowait()
