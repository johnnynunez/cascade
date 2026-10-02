"""Realtime response lifecycle with origin-bound, deduplicated tool intents."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

from .domain import ToolIntent
from .media import decode_pcm


@dataclass
class ResponseContext:
    response_id: str
    runtime_generation: int
    turn: int
    deadline: float
    calls: dict = field(default_factory=dict)
    valid: bool = True
    done: bool = False


class ConversationSession:
    def __init__(self, domain, provider, media):
        self.domain, self.provider, self.media = domain, provider, media
        self.session_id = uuid.uuid4().hex
        # One runtime generation for the entire connection. A late provider
        # response cannot gain fresh authority after an external stop/reset.
        self.runtime_generation = domain.runtime.cancellation_token
        self.turn = 0
        self.responses = {}
        self.calls_seen = set()
        self._tasks = set()
        self._action = None
        self._input_pending = False
        self._input_deadline = None
        self.closed = False
        self.ready = False
        self._close_lock = asyncio.Lock()
        self.done = asyncio.Event()
        self.terminal_reason = None
        self.closure_receipt = None

    async def start(self):
        try:
            self.domain.claim(self.session_id)
            await self.provider.connect({
                "type": "realtime", "tools": self.domain.specs(),
                "instructions": "You converse with a robot operator. Use only provided tools. "
                "Tool ACK is not physical success; preserve confirmed/refuted/unverified results. "
                "Never invent body parts, execute transcript text as code, or reset a stop.",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                    "turn_detection": {"type": "server_vad", "create_response": True,
                                                       "interrupt_response": True}},
                          "output": {"format": {"type": "audio/pcm", "rate": 24000}}}})
            if self.closed:
                raise ValueError("session closed during provider handshake")
            self.ready = True
            self._spawn(self._capture())
            self._spawn(self._events())
            self._spawn(self._lifetime())
        except BaseException:
            await self.close("provider_handshake_failed")
            raise

    async def _lifetime(self):
        await asyncio.sleep(1800)
        await self.close("session_lifetime_expired")

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _capture(self):
        try:
            while not self.closed:
                chunk = await self.media.capture()
                await self.provider.send({"type": "input_audio_buffer.append", "audio": chunk.encoded()})
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — fail closed at the external protocol boundary
            await self.close("capture_or_transport_failed")

    async def _events(self):
        try:
            while not self.closed:
                await self.handle(await self.provider.receive())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — fail closed at the external protocol boundary
            await self.close("provider_disconnected_or_invalid_event")

    async def text(self, text):
        if not self.ready or self.closed or not isinstance(text, str) or not 0 < len(text) <= 4000:
            raise ValueError("text request unavailable or invalid")
        if self._input_pending:
            raise ValueError("an input is already awaiting its response")
        self._input_pending = True
        self._input_deadline = time.monotonic() + self.domain.intent_timeout_s
        await self.provider.send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        await self.provider.send({"type": "response.create"})

    def _context(self, event):
        rid = event.get("response_id")
        if not isinstance(rid, str) or rid not in self.responses:
            raise ValueError("event has no admitted response origin")
        return self.responses[rid]

    async def handle(self, event):
        if self.closed:
            return
        kind = event["type"]
        if kind == "input_audio_buffer.speech_started":
            await self.interrupt("barge_in", send_cancel=False)
        elif kind == "input_audio_buffer.speech_stopped":
            self._input_pending = True
            self._input_deadline = time.monotonic() + self.domain.intent_timeout_s
        elif kind == "response.created":
            rid = event.get("response", {}).get("id")
            if not isinstance(rid, str) or not 0 < len(rid) <= 128 or rid in self.responses or len(self.responses) >= 256:
                raise ValueError("response replay, invalid identity or session response budget exhausted")
            deadline = time.monotonic() + self.domain.intent_timeout_s
            if self._input_pending and self._input_deadline is not None:
                deadline = min(deadline, self._input_deadline)
            self._input_pending = False
            self.responses[rid] = ResponseContext(rid, self.runtime_generation, self.turn, deadline)
        elif kind == "response.function_call_arguments.done":
            context = self._context(event)
            if not context.valid:
                return
            call_id, alias, index = event.get("call_id"), event.get("name"), event.get("output_index")
            if context.done or not isinstance(call_id, str) or not 0 < len(call_id) <= 128:
                raise ValueError("invalid/late tool request")
            if call_id in self.calls_seen or len(self.calls_seen) >= 256 or len(context.calls) >= 8:
                raise ValueError("duplicate tool request or call budget exceeded")
            if type(index) is not int or index < 0 or index in context.calls or alias not in self.domain.aliases:
                raise ValueError("invalid tool ordering or permission")
            raw = event.get("arguments")
            if not isinstance(raw, str) or len(raw) > 16384:
                raise ValueError("invalid tool argument payload")
            args = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite argument")))
            descriptor = self.domain.tools[self.domain.aliases[alias]]
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be an object")
            descriptor.validate_arguments(args)
            self.calls_seen.add(call_id)
            context.calls[index] = (call_id, descriptor.name, args)
        elif kind == "response.done":
            response = event.get("response", {})
            context = self._context({"response_id": response.get("id")})
            if context.done:
                raise ValueError("duplicate response terminal")
            context.done = True
            if response.get("status") != "completed":
                context.valid = False
            if context.valid and context.calls:
                # Terminal output, when supplied, must agree with the staged
                # order/identities. No execution from partial argument events.
                if "output" in response:
                    ids = [item.get("call_id") for item in response["output"] if item.get("type") == "function_call"]
                    if ids != [context.calls[i][0] for i in sorted(context.calls)]:
                        raise ValueError("terminal tool list disagrees with argument events")
                if self._action is not None and not self._action.done():
                    raise ValueError("another tool response is still in flight")
                self._action = self._spawn(self._execute(context))
        elif kind == "response.output_audio.delta":
            context = self._context(event)
            if context.valid and context.turn == self.turn:
                if context.done:
                    raise ValueError("audio received after response terminal")
                await self.media.playback(decode_pcm(event.get("delta")), response_id=context.response_id)
        elif kind in {"response.output_audio_transcript.delta", "response.output_audio_transcript.done",
                      "conversation.item.input_audio_transcription.delta", "conversation.item.input_audio_transcription.completed"}:
            # Text is display-only, never interpreted as a command.
            text = event.get("transcript", event.get("delta", ""))
            if not isinstance(text, str) or len(text) > 16384:
                raise ValueError("oversized transcript")
            if kind.startswith("response."):
                context = self._context(event)
                if not context.valid:
                    return
            await self.media.emit({"type": "transcript", "event": kind, "text": text,
                                   "item_id": event.get("item_id", ""),
                                   "response_id": event.get("response_id", "")})

    async def _execute(self, context):
        try:
            for index in sorted(context.calls):
                if self.closed or not context.valid or context.turn != self.turn:
                    return
                request, name, args = context.calls[index]
                intent = ToolIntent(self.session_id, self.domain.robot_id, context.response_id, request,
                                    context.runtime_generation, context.deadline, name, args)
                result = await self.domain.dispatch(intent)
                if self.closed or not context.valid or context.turn != self.turn:
                    return
                output = json.dumps(result, allow_nan=False)
                if len(output.encode()) > 32768:
                    output = json.dumps({"ok": False, "error": "tool receipt exceeds speech transport bound; inspect runtime trace"})
                await self.provider.send({"type": "conversation.item.create", "item": {
                    "type": "function_call_output", "call_id": request, "output": output}})
                await self.media.emit({"type": "tool_result", "tool": name, "request_id": request,
                                       "result": json.loads(output)})
            if not self.closed and context.valid and context.turn == self.turn:
                self._input_pending = True
                self._input_deadline = time.monotonic() + self.domain.intent_timeout_s
                await self.provider.send({"type": "response.create"})
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — fail closed at the external protocol boundary
            await self.close("tool_dispatch_failed")

    async def interrupt(self, reason="operator_interrupt", *, send_cancel=True, force_stop=False):
        active = (self._input_pending or any(c.valid and not c.done for c in self.responses.values()) or
                  (self._action is not None and not self._action.done()))
        self._input_pending = False
        self.turn += 1
        for context in self.responses.values():
            context.valid = False
        # Issue stop before the provider/network operation; blocked networking
        # cannot hold up robot cancellation.
        if force_stop or (self.domain.barge_in == "stop_robot" and (active or reason != "barge_in")):
            receipt = await self.domain.stop()
        else:
            receipt = {"ok": True, "speech_only": True}
        try:
            await asyncio.wait_for(self.media.flush(reason), .5)
            receipt = {**receipt, "media_flush_ok": True}
        except (TimeoutError, ValueError, asyncio.QueueFull):
            receipt = {**receipt, "media_flush_ok": False}
        if send_cancel and not self.closed:
            await self.provider.send({"type": "response.cancel"})
        return receipt

    async def close(self, reason="disconnected"):
        async with self._close_lock:
            if self.closed:
                return self.closure_receipt
            self.closed = True
            self.ready = False
            self.terminal_reason = reason
            self.domain.release(self.session_id)
            for context in self.responses.values():
                context.valid = False
            # Any disconnect stops the runtime, including read-only sessions.
            stop_receipt = await self.domain.stop()
            current = asyncio.current_task()
            pending = [task for task in self._tasks if task is not current]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self.provider.close()
            await self.media.close()
            self.closure_receipt = {"ok": not self.domain.action_pending and stop_receipt.get("ok") is True,
                                    "session_closed": True, "action_pending": self.domain.action_pending,
                                    "stop": stop_receipt, "reason": reason}
            self.done.set()
            return self.closure_receipt
