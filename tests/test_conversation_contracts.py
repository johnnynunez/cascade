"""Media/admission contracts use no provider, audio hardware or optional imports."""
import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cascade.conversation.domain import ConversationDomain, ToolIntent
from cascade.conversation.media import AudioChunk, AudioFormat, QueueMediaIO, decode_pcm
from cascade.conversation.provider import RealtimeConfig
from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime


class ReadDomain:
    domain_id = "sense"
    resources = (ResourceDescriptor("sense/value", "sensor", "fixture", synthetic=True, admission="software_only"),)
    tool_descriptors = (ToolDescriptor("sense.read", "test data", {"type": "object", "properties": {},
                        "additionalProperties": False}, "sense", "read"),)

    def __init__(self):
        self.calls = 0
        self.block = None
        self.entered = threading.Event()

    def execute(self, name, args):
        self.calls += 1
        self.entered.set()
        if self.block:
            assert self.block.wait(3), "fixture release missing"
        return {"ok": True}

    def stop(self):
        return {"ok": True}

    def reset_stop(self):
        return {"ok": True}

    def close(self):
        return {"ok": True}


@pytest.mark.parametrize("values", [{"sample_rate_hz": 16000}, {"channels": 2}, {"encoding": "mp3"}])
def test_format_is_explicit(values):
    with pytest.raises(ValueError):
        AudioFormat(**values)


@pytest.mark.parametrize("seq,pcm", [(-1, b"00"), (True, b"00"), (0, b"0"), (0, b""), (0, b"0"*4802)])
def test_capture_bounds(seq, pcm):
    with pytest.raises(ValueError):
        AudioChunk(seq, pcm, time.monotonic())


def test_media_queue_flush_age_and_sequence_without_clock_rejuvenation():
    async def scenario():
        media = QueueMediaIO(capacity=1)
        media.feed(0, b"\0\0")
        with pytest.raises(asyncio.QueueFull):
            media.feed(1, b"\0\0")
        assert (await media.capture()).sequence == 0
        media.feed(1, b"\0\0")
        with pytest.raises(ValueError, match="replayed"):
            media.feed(1, b"\0\0")
        await media.capture()
        media.incoming.put_nowait(AudioChunk(2, b"\0\0", time.monotonic() - 2))
        with pytest.raises(ValueError, match="stale"):
            await media.capture()
        await media.playback(b"\0\0", response_id="r")
        await media.flush("barge_in")
        assert await media.receive() == {"type": "flush", "generation": 1, "reason": "barge_in"}
        assert media._playback_bytes == 0
        await media.close()
        with pytest.raises(ValueError, match="closed"):
            media.feed(2, b"\0\0")
    asyncio.run(scenario())


def test_playback_byte_bound_not_only_event_count():
    async def scenario():
        media = QueueMediaIO()
        await media.playback(b"\0" * 96000, response_id="r")
        with pytest.raises(ValueError, match="two seconds"):
            await media.playback(b"\0\0", response_id="r")
        await media.receive()
        await media.playback(b"\0\0", response_id="r")
    asyncio.run(scenario())


@pytest.mark.parametrize("url", ["http://localhost/x", "ws://example.com/x", "wss://user:secret@example.com/x"])
def test_provider_url_refuses_insecure_remote_and_embedded_secrets(url):
    with pytest.raises(ValueError):
        RealtimeConfig(url)


def test_domain_binds_robot_session_tool_and_deduplicates_at_action_boundary():
    async def scenario():
        owner = ReadDomain()
        runtime = RobotRuntime({"sense": owner})
        try:
            with pytest.raises(ValueError, match="identity"):
                ConversationDomain(runtime, robot_id="pretend_reachy")
            with pytest.raises(ValueError, match="permission"):
                ConversationDomain(runtime, robot_id="fixture", allow_tools=["reset_stop"])
            domain = ConversationDomain(runtime, robot_id="fixture", allow_tools=["sense.read"])
            domain.claim("session")
            intent = ToolIntent("session", "fixture", "r", "call", 0, time.monotonic()+10, "sense.read", {})
            assert (await domain.dispatch(intent))["ok"]
            assert "duplicate" in (await domain.dispatch(intent))["error"]
            assert owner.calls == 1
            wrong = ToolIntent("old", "fixture", "r", "c2", 0, time.monotonic()+10, "sense.read", {})
            assert not (await domain.dispatch(wrong))["ok"]
            assert owner.calls == 1
            assert (await domain.close())["ok"]
        finally:
            runtime.close()
    asyncio.run(scenario())


def test_timeout_does_not_claim_worker_cancelled_and_new_session_cannot_overlap():
    async def scenario():
        owner = ReadDomain()
        owner.block = threading.Event()
        runtime = RobotRuntime({"sense": owner})
        domain = ConversationDomain(runtime, robot_id="fixture", allow_tools=["sense.read"], execution_timeout_s=.01)
        try:
            domain.claim("s")
            result = await domain.dispatch(ToolIntent("s", "fixture", "r", "c", 0, time.monotonic()+10, "sense.read", {}))
            assert result["delivery_uncertain"] and runtime.stopped
            domain.release("s")
            with pytest.raises(ValueError, match="previous action"):
                domain.claim("s2")
            assert (await domain.close())["action_pending"]
            owner.block.set()
            await asyncio.wait_for(asyncio.wrap_future(domain._future), 3)
            assert (await domain.close())["ok"]
        finally:
            owner.block.set()
            runtime.close()
    asyncio.run(scenario())


def test_passive_imports_do_not_require_network_or_speech_dependencies():
    repo = Path(__file__).resolve().parents[1]
    script = """
import sys
import cascade.conversation.domain, cascade.conversation.provider, cascade.conversation.gateway
import cascade.apps.conversation
assert all(name not in sys.modules for name in ('aiohttp','openai','torch','sounddevice','websockets'))
"""
    subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": str(repo / "src")}, check=True)


@pytest.mark.parametrize("bad", ["!", "AA==", "", "A" * 10000])
def test_invalid_encoded_audio_is_rejected(bad):
    with pytest.raises(ValueError):
        decode_pcm(bad, maximum_bytes=4800)
