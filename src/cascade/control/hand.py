"""Single-writer control and passive completed-solve verification for a fixed hand.

The first recipe admits bounded free finger motion. It does not admit grasps,
contact-rich motion, calibrated tactile sensing or a moving wrist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
import time

from ..sensing.models import digest, integer, number, token, vector


class HandFault(RuntimeError):
    pass


@dataclass(frozen=True)
class HandContact:
    candidate: int
    geom_a: str
    geom_b: str
    constraint_address: int
    point_world_m: tuple
    normal_a_to_b_world: tuple
    force_on_b_world_n: tuple
    normal_force_n: float

    def __post_init__(self):
        integer(self.candidate, "contact candidate")
        token(self.geom_a, "geom a"); token(self.geom_b, "geom b")
        if self.geom_a == self.geom_b or type(self.constraint_address) is not int or self.constraint_address < -1:
            raise ValueError("invalid contact identity/constraint")
        for key in ("point_world_m", "normal_a_to_b_world", "force_on_b_world_n"):
            object.__setattr__(self, key, vector(getattr(self, key), 3, key))
        number(self.normal_force_n, "normal force", minimum=0.)
        if (not math.isclose(math.hypot(*self.normal_a_to_b_world), 1., rel_tol=0, abs_tol=1e-8)
                or not math.isclose(sum(a*b for a, b in zip(self.normal_a_to_b_world,
                    self.force_on_b_world_n)), self.normal_force_n, rel_tol=1e-8, abs_tol=1e-10)
                or self.constraint_address == -1 and any(self.force_on_b_world_n)):
            raise ValueError("contact force/frame/constraint disagree")


@dataclass(frozen=True)
class HandSample:
    model_sha256: str
    epoch: str
    generation: int
    step: int
    simulation_time_s: float
    constraint_time_s: float
    captured_monotonic_s: float
    position_rad: tuple
    velocity_rad_s: tuple
    actuator_effort_nm: tuple
    commanded_position_rad: tuple
    contacts: tuple[HandContact, ...]

    def __post_init__(self):
        if self.model_sha256 is None:
            raise ValueError("hand observation needs an exact model identity")
        digest(self.model_sha256); token(self.epoch, "hand epoch")
        integer(self.generation, "generation"); integer(self.step, "step")
        for key in ("simulation_time_s", "constraint_time_s", "captured_monotonic_s"):
            number(getattr(self, key), key, minimum=0.)
        count = len(self.position_rad)
        if not 1 <= count <= 32:
            raise ValueError("hand must have 1..32 independent scalar joints")
        for key in ("position_rad", "velocity_rad_s", "actuator_effort_nm", "commanded_position_rad"):
            object.__setattr__(self, key, vector(getattr(self, key), count, key))
        contacts = tuple(self.contacts)
        if (len(contacts) > 4096 or any(type(c) is not HandContact for c in contacts)
                or tuple(c.candidate for c in contacts) != tuple(range(len(contacts)))):
            raise ValueError("hand observation needs the complete ordered contact ledger")
        object.__setattr__(self, "contacts", contacts)


@dataclass(frozen=True)
class HandLimits:
    joint_lower_rad: tuple
    joint_upper_rad: tuple
    dt_s: float = .002
    joint_margin_rad: float = .02
    maximum_target_speed_rad_s: float = .5
    maximum_observed_speed_rad_s: float = 2.
    maximum_effort_nm: float = .5
    maximum_motion_rad: float = .4
    target_tolerance_rad: float = .02
    rest_speed_rad_s: float = .05
    quiet_window_sim_s: float = .2
    command_sim_s: float = 2.
    command_wall_s: float = 10.
    rest_sim_s: float = 1.
    rest_wall_s: float = 3.
    observation_age_s: float = .2

    def __post_init__(self):
        count = len(self.joint_lower_rad)
        if not 1 <= count <= 32:
            raise ValueError("invalid hand joint count")
        for key in ("joint_lower_rad", "joint_upper_rad"):
            object.__setattr__(self, key, vector(getattr(self, key), count, key))
        for key, value in asdict(self).items():
            if key not in {"joint_lower_rad", "joint_upper_rad"} and number(value, key) <= 0:
                raise ValueError("hand limits must be positive")
        if any(lo+2*self.joint_margin_rad >= hi for lo, hi in zip(self.joint_lower_rad, self.joint_upper_rad)):
            raise ValueError("hand joint interval has no interior")


def check_hand_sample(row, backend, now, *, previous=None):
    limits = backend.limits
    if (type(row) is not HandSample or row.model_sha256 != backend.model_sha256
            or row.epoch != backend.epoch or len(row.position_rad) != len(backend.joint_names)):
        raise HandFault("hand observation identity changed")
    if not 0 <= now-row.captured_monotonic_s <= limits.observation_age_s:
        raise HandFault("hand observation is stale or from the future")
    if (row.step < 1 or not math.isclose(row.simulation_time_s, row.step*limits.dt_s, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(row.constraint_time_s, (row.step-1)*limits.dt_s, rel_tol=0, abs_tol=1e-9)):
        raise HandFault("hand solved-state and constraint clocks differ")
    if previous is not None and (row.step != previous.step+1
            or row.captured_monotonic_s < previous.captured_monotonic_s):
        raise HandFault("hand observation stream has a gap or regressed")
    if any(not lo+limits.joint_margin_rad <= q <= hi-limits.joint_margin_rad
            for q, lo, hi in zip(row.position_rad, limits.joint_lower_rad, limits.joint_upper_rad)):
        raise HandFault("hand joint margin exceeded")
    if any(not lo+limits.joint_margin_rad <= q <= hi-limits.joint_margin_rad
            for q, lo, hi in zip(row.commanded_position_rad, limits.joint_lower_rad, limits.joint_upper_rad)):
        raise HandFault("hand target margin exceeded")
    if previous is not None and any(abs(a-b) > limits.maximum_target_speed_rad_s*limits.dt_s+1e-12
            for a, b in zip(row.commanded_position_rad, previous.commanded_position_rad)):
        raise HandFault("hand target slew rate exceeded")
    if max(abs(v) for v in row.velocity_rad_s) > limits.maximum_observed_speed_rad_s:
        raise HandFault("hand observed speed exceeded")
    if max(abs(v) for v in row.actuator_effort_nm) > limits.maximum_effort_nm+1e-12:
        raise HandFault("hand actuator effort exceeded")
    if any(c.geom_a not in backend.geom_names or c.geom_b not in backend.geom_names for c in row.contacts):
        raise HandFault("contact geometry is outside the bound hand model")
    if any(c.constraint_address >= 0 and c.normal_force_n > 0 for c in row.contacts):
        raise HandFault("free hand motion encountered an enabled contact")


def quiet(row, limits):
    return max(abs(v) for v in row.velocity_rad_s) <= limits.rest_speed_rad_s


class HandController:
    """One private solver thread; readers and stop never advance the simulation.

    Stop revokes target changes and retains the last position-servo target.
    Its ACK is distinct from measured rest. Model construction precedes start.
    """
    def __init__(self, backend, *, clock=time.monotonic, owner_wall_s=30., capacity=16000):
        if type(backend.synthetic) is not bool:
            raise ValueError("explicit hand measurement kind required")
        if number(owner_wall_s, "owner lifetime") <= 0 or type(capacity) is not int or capacity < 2:
            raise ValueError("bounded owner lifetime and archive required")
        self.backend, self.clock, self.limits = backend, clock, backend.limits
        self.owner_wall_s, self.capacity = owner_wall_s, capacity
        self._condition = threading.Condition(threading.RLock())
        self._generation, self._latched = 0, True
        self._history = []
        self._latest = self._error = self._permit = self._thread = None
        self._targets = tuple(backend.initial_targets)
        self._desired = self._targets
        self._shutdown = threading.Event()
        self._closure = None

    @property
    def generation(self):
        with self._condition:
            return self._generation

    def start(self):
        with self._condition:
            if self._thread is not None or self._shutdown.is_set():
                raise HandFault("hand owner cannot be restarted")
            self._started = self.clock()
            self._thread = threading.Thread(target=self._run, name="cascade-hand-owner", daemon=False)
            self._thread.start()

    def _run(self):
        previous = None
        try:
            while not self._shutdown.is_set():
                with self._condition:
                    now = self.clock()
                    if now >= self._started+self.owner_wall_s:
                        raise HandFault("hand owner lifetime expired")
                    if previous is not None:
                        check_hand_sample(previous, self.backend, now)
                    permit = self._permit
                    if permit is not None and (now >= permit["deadline"]
                            or previous.step >= permit["end_step"]):
                        raise HandFault("hand command lease expired")
                    if not self._latched:
                        delta = self.limits.maximum_target_speed_rad_s*self.limits.dt_s
                        self._targets = tuple(q+max(-delta, min(delta, goal-q))
                            for q, goal in zip(self._targets, self._desired))
                    generation = self._generation
                    # The final generation check and native upload share stop's lock.
                    self.backend.upload(self._targets)
                # Stop may ACK while this one solve is in flight; retain its
                # original generation. The next write must use the new hold.
                row = self.backend.advance(generation, self.clock)
                with self._condition:
                    if len(self._history) >= self.capacity:
                        raise HandFault("hand archive capacity exhausted")
                    self._history.append(row)  # Preserve a solved safety veto too.
                if previous is None and row.step != 1:
                    raise HandFault("hand owner did not publish its first solve")
                check_hand_sample(row, self.backend, self.clock(), previous=previous)
                with self._condition:
                    self._latest = previous = row
                    self._condition.notify_all()
                self._shutdown.wait(self.limits.dt_s)
        except BaseException as exc:
            with self._condition:
                try:
                    self._error = type(exc).__name__+": "+str(exc)
                except BaseException:
                    self._error = type(exc).__name__+": unreadable exception"
                self._latched, self._permit = True, None
                self._generation += 1
                self._condition.notify_all()
        finally:
            # This is a fixed in-process simulator: no external motor remains
            # powered. Hold was already uploaded; do not invent another solve.
            self._shutdown.set()

    def read(self, cursor=None, *, timeout_s=.05):
        if cursor is not None:
            integer(cursor, "hand read cursor")
        if not 0 <= number(timeout_s, "read timeout") <= .05:
            raise ValueError("hand reader timeout exceeds its fixed bound")
        with self._condition:
            if self._error:
                raise HandFault(self._error)
            if self._latest is None or cursor is not None and self._latest.step <= cursor:
                self._condition.wait(timeout_s)
            if self._error:
                raise HandFault(self._error)
            if self._latest is None:
                return ()
            if cursor is None:
                rows = (self._latest,)
            else:
                rows = tuple(self._history[cursor:])
            if rows:
                check_hand_sample(rows[-1], self.backend, self.clock())
            return rows

    def stop(self):
        with self._condition:
            self._generation += 1
            self._latched, self._permit = True, None
            self._desired = self._targets
            return {"ok": self._error is None, "generation": self._generation,
                "accepted_monotonic_s": self.clock(), "latched": True,
                "hold": "last position-servo targets", "targets_rad": self._targets,
                "physical_stop_verified": False}

    def reset_stop(self):
        with self._condition:
            if self._shutdown.is_set() or self._error or self._latest is None or self._permit is not None:
                return {"ok": False, "error": "hand owner unavailable or command active"}
            check_hand_sample(self._latest, self.backend, self.clock())
            if not quiet(self._latest, self.limits):
                return {"ok": False, "error": "hand is not observed at rest"}
            self._generation += 1
            generation = self._generation
            self._latched = True
            # Complete the reset generation's hold before admitting a task.
            # One pre-reset solve may still be in flight on the writer thread.
            deadline = self.clock()+self.limits.observation_age_s
            while self._latest.generation != generation:
                if (self._error or self._shutdown.is_set() or self._generation != generation
                        or self.clock() >= deadline):
                    return {"ok": False, "error": "hand reset lacks a fresh completed hold"}
                self._condition.wait(min(.01, max(0., deadline-self.clock())))
            if self._generation != generation or self.clock() >= deadline:
                return {"ok": False, "error": "hand reset was revoked or expired"}
            check_hand_sample(self._latest, self.backend, self.clock())
            if not quiet(self._latest, self.limits):
                return {"ok": False, "error": "hand reset hold is not observed at rest"}
            self._latched = False
            return {"ok": True, "generation": self._generation, "latched": False}

    def admit(self, targets):
        target = vector(targets, len(self.backend.joint_names), "finger targets")
        with self._condition:
            row = self._latest
            if (self._latched or self._shutdown.is_set() or self._permit is not None or row is None
                    or row.generation != self._generation):
                raise HandFault("hand is stopped, unavailable or already owns a command")
            now = self.clock()
            check_hand_sample(row, self.backend, now)
            if not quiet(row, self.limits):
                raise HandFault("hand admission requires observed rest")
            for goal, q, lo, hi in zip(target, row.position_rad,
                    self.limits.joint_lower_rad, self.limits.joint_upper_rad):
                if (not lo+self.limits.joint_margin_rad <= goal <= hi-self.limits.joint_margin_rad
                        or abs(goal-q) > self.limits.maximum_motion_rad):
                    raise HandFault("finger target exceeds the bounded free-motion interval")
            self._generation += 1
            self._desired = target
            self._permit = {"generation": self._generation, "step": row.step,
                "accepted": now, "deadline": now+self.limits.command_wall_s,
                "end_step": row.step+round(self.limits.command_sim_s/self.limits.dt_s)}
            return dict(self._permit)

    def confirm_generation(self, generation, deadline):
        """Final observation credit cannot outlive its controller authority."""
        with self._condition:
            if (self._error or self._shutdown.is_set() or self._generation != generation
                    or self.clock() >= deadline):
                raise HandFault("hand verification was revoked or expired")

    def close(self):
        if self._closure is not None:
            return dict(self._closure)
        ack = self.stop()
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(2.)
        closed = self._thread is None or not self._thread.is_alive()
        self._closure = {"ok": closed and self._error is None, "owner_thread_closed": closed,
            "error": self._error, "stop": ack, "physical_stop_verified": False,
            "completed_solves": len(self._history)}
        return dict(self._closure)

    def records(self):
        if self._thread is not None and self._thread.is_alive():
            raise HandFault("archive may only be exported after owner closure")
        return tuple(self._history)
