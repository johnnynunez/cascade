"""Bounded command admission between shared solves; stop remains immediate.

Only the simulation owner drains commands/resets. RPC threads can observe,
renew a lease or revoke permission while policy/native work is in flight.
This broker never holds its lock across a solve, device call or inference.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import threading


@dataclass
class _Request:
    operation: str
    arguments: dict
    owner_alive: object = None
    event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: Exception | None = None


class BoundaryAdmission:
    """One waiting operation per robot, with the original receipt deadline."""
    def __init__(self, controllers):
        self.controllers = dict(controllers)
        if not self.controllers or len({id(c) for c in self.controllers.values()}) != len(self.controllers):
            raise ValueError('distinct named controllers required')
        self._pending = {}
        self._lock = threading.RLock()
        self.closed = False
        self.owner_thread = threading.get_ident()

    def submit(self, robot_id, operation, request):
        if threading.get_ident() == self.owner_thread:
            raise RuntimeError('boundary requests require a separate RPC caller')
        if operation not in ('command_velocity', 'reset_stop') or not isinstance(request, dict):
            raise ValueError('unsupported boundary operation')
        controller = self.controllers[robot_id]
        arguments = deepcopy(request)
        owner_alive = arguments.pop('_owner_alive', None)
        if owner_alive is not None and not callable(owner_alive):
            raise ValueError('invalid transport owner guard')
        received = arguments.setdefault('_received_wall', controller._clock())
        now = controller._clock()
        if type(received) not in (int, float) or not math.isfinite(received) or received > now:
            raise ValueError('invalid original request receipt')
        remaining = min(controller.lease_s, controller.max_action_wall_s) - (now - received)
        if remaining <= 0:
            raise ValueError('command expired before boundary admission')
        item = _Request(operation, arguments, owner_alive)
        with self._lock:
            if self.closed:
                raise RuntimeError('boundary admission closed')
            if robot_id in self._pending:
                raise ValueError('robot already has a waiting boundary operation')
            self._pending[robot_id] = item
        arrived = item.event.wait(remaining)
        with self._lock:
            if self._pending.get(robot_id) is item:
                del self._pending[robot_id]
            if item.error is None and (not arrived or not self._live(controller, received, owner_alive)):
                if item.result is not None:
                    controller.stop(latch=True)
                item.error = RuntimeError('boundary deadline expired or owner channel closed; operation withdrawn')
            if item.error is not None:
                raise item.error
            if item.result is None:
                raise RuntimeError('boundary operation ended without admission')
            return deepcopy(item.result)

    @staticmethod
    def _live(controller, received, owner_alive=None):
        age = controller._clock() - received
        if not 0 <= age < min(controller.lease_s, controller.max_action_wall_s):
            return False
        try:
            return owner_alive is None or owner_alive() is True
        except Exception:
            return False

    def drain(self):
        """Simulation thread only, immediately before preparing a new tick."""
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('only the simulation owner may drain commands')
        with self._lock:
            if self.closed:
                raise RuntimeError('boundary admission closed')
            for robot_id, item in self._pending.items():
                if item.event.is_set():
                    continue
                controller = self.controllers[robot_id]
                try:
                    received = item.arguments['_received_wall']
                    if not self._live(controller, received, item.owner_alive):
                        raise ValueError('command expired or owner channel closed in boundary admission')
                    item.result = getattr(controller, item.operation)(item.arguments)
                    if not self._live(controller, received, item.owner_alive):
                        controller.stop(latch=True)
                        raise ValueError('command expired or owner channel closed during admission; permission revoked')
                except Exception as exc:
                    item.error = exc
                item.event.set()

    def _withdraw(self, robot_id, reason, *, owner=None, revoke=True):
        item = self._pending.get(robot_id)
        if item is not None and (owner is None or item.arguments.get('owner') == owner):
            item.error = RuntimeError(reason)
            item.event.set()
            del self._pending[robot_id]
            if revoke and item.result is not None:
                self.controllers[robot_id].stop(latch=True)

    def stop(self, robot_id, *, latch=True):
        with self._lock:
            controller = self.controllers[robot_id]
            # Validate before withdrawing; a malformed request is not a stop.
            if type(latch) is not bool:
                raise ValueError('latch must be a boolean')
            self._withdraw(robot_id, 'stop withdrew boundary operation', revoke=False)
            return controller.stop(latch=latch)

    def owner_disconnected(self, robot_id, owner):
        with self._lock:
            self._withdraw(robot_id, 'owner disconnected before boundary reply', owner=owner)
            self.controllers[robot_id].owner_disconnected(owner)

    def close(self):
        with self._lock:
            self.closed = True
            first = None
            for robot_id in tuple(self._pending):
                try:
                    self._withdraw(robot_id, 'shared lifecycle closed before boundary reply')
                except Exception as exc:
                    if first is None:
                        first = exc
            if first is not None:
                raise first

    def endpoint(self, robot_id):
        if robot_id not in self.controllers:
            raise ValueError('unknown robot identity')
        return BoundaryController(self, robot_id)


class BoundaryController:
    """Bridge-compatible facade; only commands/resets wait for the owner."""
    def __init__(self, admission, robot_id):
        self.admission, self.robot_id = admission, robot_id

    def __getattr__(self, name):
        return getattr(self.admission.controllers[self.robot_id], name)

    def command_velocity(self, request):
        return self.admission.submit(self.robot_id, 'command_velocity', request)

    def reset_stop(self, request):
        return self.admission.submit(self.robot_id, 'reset_stop', request)

    def stop(self, *, latch=True):
        return self.admission.stop(self.robot_id, latch=latch)

    def owner_disconnected(self, owner):
        self.admission.owner_disconnected(self.robot_id, owner)


class SharedEndpoints:
    """Independent loopback channels over a single owner; no physics methods."""
    def __init__(self, steppers, *, base_port, max_jpeg_bytes, max_pixels=640*480):
        from .mobile_bridge import MobileBridgeServer
        from .microduck_stepper import FrameCache
        steppers = tuple(steppers)
        if (not 1 <= len(steppers) <= 12 or type(base_port) is not int
                or not 0 <= base_port <= 65536-len(steppers)):
            raise ValueError('explicit available loopback port range required')
        if type(max_pixels) is not int or max_pixels <= 0:
            raise ValueError('explicit positive overview pixel bound required')
        controllers = {s.identity['robot_id']: s.controller for s in steppers}
        if len(controllers) != len(steppers):
            raise ValueError('duplicate shared endpoint robot identity')
        self.admission = BoundaryAdmission(controllers)
        self.caches, self.servers = {}, {}
        self.closed = self.started = False
        try:
            for i, (robot_id, controller) in enumerate(controllers.items()):
                cache = FrameCache(controller.hello(), max_jpeg_bytes=max_jpeg_bytes, max_pixels=max_pixels)
                self.caches[robot_id] = cache
                self.servers[robot_id] = MobileBridgeServer(self.admission.endpoint(robot_id),
                    port=base_port+i if base_port else 0, frame_callback=cache)
        except BaseException as exc:
            self._rollback(exc)
            raise

    def publish_capture(self, capture):
        if self.closed:
            raise RuntimeError('shared endpoints closed')
        for cache in self.caches.values():
            cache.publish(**capture)

    def start(self):
        if self.closed or self.started:
            raise RuntimeError('shared endpoints already started/closed')
        try:
            for robot_id, controller in self.admission.controllers.items():
                state = controller.state()['state']
                frame = self.caches[robot_id]({'camera': 'overview'})['frame']
                if state is None or (state['step'], state['sim_time_s']) != (frame['step'], frame['sim_time_s']):
                    raise RuntimeError('endpoint requires a completed state and matching frame')
            for server in self.servers.values():
                server.start()
            self.started = True
            return {robot_id: {'host': server.address[0], 'port': server.address[1],
                    'hello': server.controller.hello()}
                    for robot_id, server in self.servers.items()}
        except BaseException as exc:
            self._rollback(exc)
            raise

    def _rollback(self, primary):
        try:
            self.close()
        except BaseException as exc:
            add_note = getattr(primary, 'add_note', None)
            if add_note is not None:
                add_note(f'endpoint cleanup failed: {type(exc).__name__}: {exc}')

    def close(self):
        if self.closed:
            return
        self.closed = True
        first = None
        # Wake queued RPCs before server worker joins; attempt every cleanup.
        for resource in (self.admission, *self.servers.values(), *self.caches.values()):
            try:
                resource.close()
            except BaseException as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first
