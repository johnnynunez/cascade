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
        # Logging/provider work can saturate asyncio's default executor. Give
        # coalesced stop delivery one separate worker, created lazily by submit.
        self._stop_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="conversation-stop")
        self._stop_records = []
        self._stop_record_requests = set()
        self._closure_receipt = None

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
            pending, errors = self._stop_record_state()
            if pending or errors:
                raise ValueError("previous stop tool records are pending or failed")
            self._session_id = session_id
            self._requests.clear()
            self._stop_records.clear()
            self._stop_record_requests.clear()

    def release(self, session_id):
        with self._lock:
            if self._session_id == session_id:
                self._session_id = None

    async def dispatch(self, intent: ToolIntent):
        with self._lock:
            if self._closed or intent.session_id != self._session_id or intent.robot_id != self.robot_id:
                return {"ok": False, "error": "intent session/robot no longer admitted"}
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
            priority_stop = descriptor.effect == "stop"
            if not priority_stop:
                if self._future and not self._future.done():
                    return {"ok": False, "error": "previous conversation action remains in flight"}
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
        if priority_stop:
            # Never overwrite the action future/thread: that worker may still
            # be running, and close/claim must continue to quarantine it.
            return await self.stop(intent=intent)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), self.execution_timeout_s)
        except TimeoutError:
            await self.stop()
            return {"ok": False, "execution_ok": False, "delivery_uncertain": True,
                    "error": "action deadline; underlying action may still be in flight"}

    async def stop(self, *, intent=None):
        # Coalesce pending stop requests; provider and audio workers are never
        # on this priority path. Runtime itself owns bounded domain stop slots.
        started = time.monotonic()
        if self._stop_executor is None:
            return {"ok": False, "error": "conversation domain closed; no new stop delivered",
                    "stop_worker_closed": True, "physical_stop_verified": False}
        if intent is not None:
            # A model tool still needs current conversation authority. Recheck
            # at the final local admission boundary, immediately before starting
            # or joining the stop task. Operator stop has no intent/deadline.
            # Once admitted, stopping must not be cancelled by a later expiry;
            # this is not a hard real-time delivery guarantee.
            with self._lock:
                descriptor = self.tools.get(intent.tool)
                current = self.runtime.tool_descriptors.get(intent.tool)
                if self._closed or intent.session_id != self._session_id or intent.robot_id != self.robot_id:
                    return {"ok": False, "error": "intent session/robot no longer admitted"}
                if (descriptor is None or descriptor.effect != "stop" or current is None or
                        descriptor.as_dict() != self._catalog[intent.tool] or
                        current.as_dict() != self._catalog[intent.tool]):
                    return {"ok": False, "error": "stop tool catalog changed or permission absent"}
                if time.monotonic() >= intent.deadline_monotonic_s:
                    return {"ok": False, "error": "stop intent expired"}
                if type(intent.runtime_generation) is not int or intent.runtime_generation != self.runtime.cancellation_token:
                    return {"ok": False, "error": "stale stop intent generation"}
                if intent.request_id not in self._requests or intent.request_id in self._stop_record_requests:
                    return {"ok": False, "error": "stop intent not admitted or already recorded"}
                self._stop_record_requests.add(intent.request_id)
        if self._stop_executor is not None and (self._stop_task is None or self._stop_task.done()):
            executor = self._stop_executor
            async def deliver():
                return await asyncio.get_running_loop().run_in_executor(executor, self.runtime.stop)
            self._stop_task = asyncio.create_task(deliver())
        if intent is None:
            return copy.deepcopy(await asyncio.shield(self._stop_task))
        delivery = self._stop_task
        async def complete_record():
            # Keep this obligation after its consumer is cancelled, just like
            # the normal action worker. There are at most 256 admitted requests
            # per session; retained tasks/errors prevent a new overlapping session.
            result = copy.deepcopy(await asyncio.shield(delivery))
            await asyncio.to_thread(self.runtime._record_tool_result, intent.tool,
                plain_json(intent.arguments), result, started_monotonic_s=started)
            return result
        record = asyncio.create_task(complete_record())
        with self._lock:
            self._stop_records.append(record)
        # Retrieve exceptions even if the caller disappeared. Retain the task
        # itself so close can report failure instead of claiming it drained.
        record.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return await asyncio.shield(record)

    def _stop_record_state(self):
        """Called with _lock held; no waiting, cancellation or trace IO."""
        pending = sum(not task.done() for task in self._stop_records)
        errors = ["CancelledError" if task.cancelled() else type(task.exception()).__name__
                  for task in self._stop_records if task.done() and (task.cancelled() or task.exception() is not None)]
        return pending, errors

    async def close(self):
        if self._closure_receipt is not None:
            return copy.deepcopy(self._closure_receipt)
        with self._lock:
            self._closed = True
            self._session_id = None
        receipt = await self.stop()
        with self._lock:
            pending = self._future is not None and not self._future.done()
            records_pending, record_errors = self._stop_record_state()
            delivery_pending = self._stop_task is not None and not self._stop_task.done()
            if self._stop_task is not None and not delivery_pending:
                receipt = copy.deepcopy(self._stop_task.result())
        if not pending and not records_pending and not delivery_pending and self._stop_executor is not None:
            # Delivery completed before this point. Join only our now-idle stop
            # worker; never wait for the default executor or a stuck logger.
            self._stop_executor.shutdown(wait=True)
            self._stop_executor = None
        result = {"ok": not pending and not records_pending and not delivery_pending and not record_errors and receipt.get("ok") is True,
                "action_pending": pending, "stop_tool_records_pending": records_pending,
                "stop_delivery_pending": delivery_pending,
                "stop_tool_record_errors": record_errors, "stop_worker_closed": self._stop_executor is None,
                "stop": receipt}
        if self._stop_executor is None:
            self._closure_receipt = copy.deepcopy(result)
        return result
