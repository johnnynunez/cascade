"""KINEMATIC TEST DOUBLE, not a balance policy, dynamics or physical evidence.

The worker advances toy planar motion independently of reads. ``advance`` is a
fixture hook for deterministic tests only. No joints, forces or contacts are
simulated, and every snapshot advertises ``measurement_kind=kinematic_mock``.
"""
from __future__ import annotations

import math
import threading
import time
import uuid

from .mobile_base import BaseState, MobileBase, VelocityCommand, finite_real, identifier, nonnegative_int


class MockMobileBase(MobileBase):
    def __init__(self, *, wall_lease_s: float, robot_id="microduck-mock",
                 source="mock-kinematic", dt_s=.01, auto_step=True):
        self._robot_id = identifier(robot_id, "robot_id")
        self._source = identifier(source, "source")
        self._dt = finite_real(dt_s, "dt_s")
        self._lease = finite_real(wall_lease_s, "wall_lease_s")
        if self._dt <= 0 or self._lease <= 0 or type(auto_step) is not bool:
            raise ValueError("positive dt_s/wall_lease_s and boolean auto_step required")
        self._auto_step = auto_step
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._shutdown = threading.Event()
        self._worker = None
        self.connected = False
        self._epoch = uuid.uuid4().hex
        self._generation = 0
        self._latched = False
        self._fault = False
        self._step = 0
        self._sim_time = 0.
        self._published = time.monotonic()
        self._x = self._y = self._yaw = 0.
        self._command = None
        self._sim_end = self._wall_end = 0.

    @property
    def metadata(self) -> dict:
        return {"robot_id": self._robot_id, "source": self._source,
                "measurement_kind": "kinematic_mock"}

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"walk_velocity", "turn", "stop_navigation"})

    def connect(self) -> None:
        with self._lifecycle:
            with self._lock:
                if self.connected:
                    return
                self.connected = True
                self._shutdown.clear()
                self._published = time.monotonic()
                # A prior stop/disconnect deliberately survives connect.
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name=f"mock-mobile-{self._robot_id}")
            self._worker.start()

    def disconnect(self) -> None:
        self.stop(latch=True)
        with self._lifecycle:
            self._shutdown.set()
            worker = self._worker
            if worker is not None:
                worker.join(timeout=2.)
                if worker.is_alive():
                    raise RuntimeError("mock worker did not terminate")
            with self._lock:
                self.connected = False
                self._worker = None

    def _expire(self, now: float) -> None:
        if self._command is not None and now >= self._wall_end:
            self._command = None
            self._generation += 1
            self._latched = self._fault = True

    def _run(self) -> None:
        while not self._shutdown.wait(min(self._dt, self._lease / 4)):
            if self._auto_step:
                self.advance(min(self._dt, self._lease / 4))
            else:
                with self._lock:
                    self._expire(time.monotonic())

    def advance(self, dt_s=None) -> None:
        """Advance toy kinematics. NEVER use this as physical feedback."""
        dt = self._dt if dt_s is None else finite_real(dt_s, "dt_s")
        if dt <= 0:
            raise ValueError("dt_s must be positive")
        with self._lock:
            if not self.connected:
                raise RuntimeError("mock base is disconnected")
            now = time.monotonic()
            self._expire(now)
            command = self._command
            if command is not None:
                travel_dt = max(0., min(dt, self._sim_end - self._sim_time))
                # This integration is the declared toy, NEVER physical truth.
                mid_yaw = self._yaw + command.wz * travel_dt / 2
                self._x += (command.vx * math.cos(mid_yaw) - command.vy * math.sin(mid_yaw)) * travel_dt
                self._y += (command.vx * math.sin(mid_yaw) + command.vy * math.cos(mid_yaw)) * travel_dt
                self._yaw += command.wz * travel_dt
            self._sim_time += dt
            self._step += 1
            self._published = now
            if self._sim_time >= self._sim_end:
                self._command = None

    def get_state(self) -> BaseState:
        with self._lock:
            if not self.connected:
                raise RuntimeError("mock base is disconnected")
            now = time.monotonic()
            command = self._command
            vx, vy, wz = (command.vx, command.vy, command.wz) if command else (0., 0., 0.)
            return BaseState(
                robot_id=self._robot_id, source=self._source, epoch=self._epoch,
                step=self._step, sim_time_s=self._sim_time,
                received_monotonic_s=now, producer_age_s=max(0., now - self._published),
                position_world=(self._x, self._y, .2),
                orientation_wxyz=(math.cos(self._yaw / 2), 0., 0., math.sin(self._yaw / 2)),
                linear_velocity_world=(vx * math.cos(self._yaw) - vy * math.sin(self._yaw),
                                       vx * math.sin(self._yaw) + vy * math.cos(self._yaw), 0.),
                angular_velocity_body=(0., 0., wz), joint_names=(), joint_positions=(),
                joint_velocities=(), controller_status="fault" if self._fault else "active" if command else "ready",
                generation=self._generation, contacts=(), fallen=False, latched=self._latched,
                measurement_kind="kinematic_mock",
            )

    def _ack(self, ok=True, **extra) -> dict:
        return {"ok": ok, "robot_id": self._robot_id, "source": self._source,
                "epoch": self._epoch, "generation": self._generation,
                "latched": self._latched, **extra}

    def command_velocity(self, command: VelocityCommand, *, generation: int) -> dict:
        if not isinstance(command, VelocityCommand):
            raise ValueError("VelocityCommand required")
        generation = nonnegative_int(generation, "generation")
        with self._lock:
            self._expire(time.monotonic())
            if not self.connected or self._latched or self._fault or generation != self._generation:
                return self._ack(False, accepted=False, error="disconnected, latched, fault or stale generation")
            self._generation += 1
            self._command = command
            self._sim_end = self._sim_time + command.duration_s
            self._wall_end = time.monotonic() + self._lease
            return self._ack(accepted=True, start_sim_time_s=self._sim_time,
                             end_sim_time_s=self._sim_end)

    def stop(self, *, latch=True) -> dict:
        if type(latch) is not bool:
            raise ValueError("latch must be boolean")
        with self._lock:
            self._generation += 1
            self._command = None
            self._latched = self._latched or latch
            return self._ack()

    def reset_stop(self) -> dict:
        with self._lock:
            self._generation += 1
            self._command = None
            if self._fault or not self.connected:
                return self._ack(False, error="reset does not enable a failed/disconnected controller")
            self._latched = False
            return self._ack()
