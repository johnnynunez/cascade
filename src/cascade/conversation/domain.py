"""Robot-agnostic conversation authority and a single bounded action worker."""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import math
import threading
import time
from dataclasses import dataclass

from ..robotics.contracts import freeze_json, plain_json

# These curated semantic skills delegate all movement to existing domain owners.
# Raw pose/joint commands and permission/reset controls are intentionally absent.
SEMANTIC_MOTION_SKILLS = frozenset({"walk_velocity", "turn", "navigate_to", "pick_and_place",
                                    "grasp_object", "place_at", "handover", "wave", "go_home"})


@dataclass(frozen=True)
class ToolIntent:
    session_id: str
    robot_id: str
    response_id: str
    request_id: str
    runtime_generation: int
    deadline_monotonic_s: float
    tool: str
    arguments: object

    def __post_init__(self):
        for value in (self.session_id, self.robot_id, self.response_id, self.request_id, self.tool):
            if not isinstance(value, str) or not 0 < len(value) <= 128:
                raise ValueError("invalid intent identity")
        if type(self.runtime_generation) is not int or self.runtime_generation < 0:
            raise ValueError("invalid runtime generation")
        if not math.isfinite(self.deadline_monotonic_s):
            raise ValueError("invalid intent deadline")
        object.__setattr__(self, "arguments", freeze_json(self.arguments))


class ConversationDomain:
    """HRI supervisor above RobotRuntime, never an actuator domain/controller.

    One worker at most; timeout does not pretend to kill a running Python call.
    A new session is refused while the previous worker remains in flight.
    """
    def __init__(self, runtime, *, robot_id, allow_tools=(), allow_motion=False,
                 intent_timeout_s=10, execution_timeout_s=30, barge_in="stop_robot"):
        if not isinstance(robot_id, str) or not 0 < len(robot_id) <= 128:
            raise ValueError("robot_id is required")
        if not 0 < intent_timeout_s <= 60 or not 0 < execution_timeout_s <= 300:
            raise ValueError("invalid conversation deadline")
        if barge_in not in {"stop_robot", "speech_only"}:
            raise ValueError("invalid interruption policy")
        if allow_motion and barge_in != "stop_robot":
            raise ValueError("motion conversations require stop_robot interruption")
        self.runtime, self.robot_id = runtime, robot_id
        if any(resource.robot_id != robot_id for resource in runtime.resources):
            raise ValueError("conversation robot identity differs from the resource catalog")
        self.intent_timeout_s, self.execution_timeout_s = intent_timeout_s, execution_timeout_s
        self.barge_in = barge_in
        self.tools = {}
        for name in allow_tools:
            descriptor = runtime.tool_descriptors.get(name)
            if descriptor is None or descriptor.effect not in {"read", "motion", "stop"} or name == "task_done":
                raise ValueError("conversation tool is absent or is a permission/task control")
            if descriptor.effect == "motion" and (not allow_motion or descriptor.local_name not in SEMANTIC_MOTION_SKILLS):
                raise ValueError("only explicitly enabled curated semantic motion is supported")
            if name in self.tools:
                raise ValueError("duplicate conversation tool")
            self.tools[name] = descriptor
        # Provider function names use a stable legal alphabet without guessing
        # robot names or exposing unselected capabilities.
        self.aliases = {f"robot_tool_{i}": name for i, name in enumerate(self.tools)}
        self._catalog = {name: copy.deepcopy(desc.as_dict()) for name, desc in self.tools.items()}
        self._lock = threading.Lock()
        self._future = None
        self._closed = False
        self._session_id = None
        self._requests = set()
        self._thread = None
        self._stop_task = None

    def specs(self):
        return [{"type": "function", "name": alias, "description": self.tools[name].description,
                 "parameters": plain_json(self.tools[name].parameters)} for alias, name in self.aliases.items()]

    @property
    def action_pending(self):
        with self._lock:
            return self._future is not None and not self._future.done()

    def claim(self, session_id):
        with self._lock:
            if self._closed or self._session_id is not None or (self._future and not self._future.done()):
                raise ValueError("conversation domain closed, occupied or awaiting a previous action")
            self._session_id = session_id
            self._requests.clear()

    def release(self, session_id):
        with self._lock:
            if self._session_id == session_id:
                self._session_id = None

    async def dispatch(self, intent: ToolIntent):
        with self._lock:
            if self._closed or intent.session_id != self._session_id or intent.robot_id != self.robot_id:
                return {"ok": False, "error": "intent session/robot no longer admitted"}
            if self._future and not self._future.done():
                return {"ok": False, "error": "previous conversation action remains in flight"}
            if intent.request_id in self._requests or len(self._requests) >= 256:
                return {"ok": False, "error": "duplicate request or session call budget exceeded"}
            self._requests.add(intent.request_id)
            descriptor = self.tools.get(intent.tool)
            if descriptor is None or descriptor.as_dict() != self._catalog[intent.tool]:
                return {"ok": False, "error": "tool catalog changed or permission absent"}
            current = self.runtime.tool_descriptors.get(intent.tool)
            if current is None or current.as_dict() != self._catalog[intent.tool]:
                return {"ok": False, "error": "runtime tool catalog changed"}
            args = plain_json(intent.arguments)
            descriptor.validate_arguments(args)
            future = concurrent.futures.Future()
            self._future = future

            def invoke():
                try:
                    # Check after any scheduling delay, immediately before the
                    # runtime's atomic generation admission boundary.
                    with self._lock:
                        valid = not self._closed and self._session_id == intent.session_id
                    if not valid or time.monotonic() >= intent.deadline_monotonic_s:
                        result = {"ok": False, "error": "intent expired or session cancelled"}
                    else:
                        result = self.runtime.execute(intent.tool, args,
                                                      expected_generation=intent.runtime_generation,
                                                      deadline_monotonic_s=intent.deadline_monotonic_s)
                    future.set_result(result)
                except BaseException as exc:  # noqa: BLE001 — worker must resolve its receipt on every exit
                    future.set_result({"ok": False, "error": type(exc).__name__})

            self._thread = threading.Thread(target=invoke, name="conversation-action", daemon=True)
            self._thread.start()
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), self.execution_timeout_s)
        except TimeoutError:
            await self.stop()
            return {"ok": False, "execution_ok": False, "delivery_uncertain": True,
                    "error": "action deadline; underlying action may still be in flight"}

    async def stop(self):
        # Coalesce pending stop requests; provider and audio workers are never
        # on this priority path. Runtime itself owns bounded domain stop slots.
        if self._stop_task is None or self._stop_task.done():
            self._stop_task = asyncio.create_task(asyncio.to_thread(self.runtime.stop))
        return await asyncio.shield(self._stop_task)

    async def close(self):
        with self._lock:
            self._closed = True
            self._session_id = None
        receipt = await self.stop()
        with self._lock:
            pending = self._future is not None and not self._future.done()
        return {"ok": not pending and receipt.get("ok") is True, "action_pending": pending,
                "stop": receipt}
