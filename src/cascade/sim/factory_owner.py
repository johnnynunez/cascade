"""Single solve owner for the mounted fastening controller.

The owner is explicit and optional. Construction does not start a thread or
load a simulator. The concrete Newton backend below advances one actual solve
per cycle and keeps all generation/clock stamps from BEFORE that solve.
"""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
import math
import queue
import threading
import time

import numpy as np

from ..control.fastening import (
    FasteningController, FasteningFault, FasteningRevoked, SolveJournal,
)
from .factory_observation import _floats, collision_coverage


class NativeSolveClock:
    """Pinned MJWarp float32 recurrence plus Python step; no CUDA graph replay."""

    def __init__(self, solver, dt_s):
        if not math.isfinite(dt_s) or not 0 < dt_s <= .02:
            raise FasteningFault("native timestep must be finite and in (0,.02]")
        if solver.use_mujoco_cpu or solver._step != 0:
            raise FasteningFault("Factory clock requires a fresh manual MJWarp solver")
        self.solver, self.dt_s = solver, float(dt_s)
        self.native_dt = np.float32(dt_s)
        self.step, self.native_time = 0, np.float32(0.)
        if _floats(solver.mjw_data.time, (1,), "native time")[0] != 0:
            raise FasteningFault("Factory native clock did not start at zero")

    def before(self):
        self._require(self.step, self.native_time)

    def after(self):
        expected = np.float32(self.native_time + self.native_dt)
        self._require(self.step + 1, expected)
        dt = _floats(self.solver.mjw_model.opt.timestep, (1,), "native timestep")[0]
        if dt != self.native_dt:
            raise FasteningFault("solver timestep changed")
        self.step += 1
        self.native_time = expected

    def _require(self, step, native_time):
        if (type(self.solver._step) is not int or self.solver._step != step
                or _floats(self.solver.mjw_data.time, (1,), "native time")[0] != native_time):
            raise FasteningFault("native solver count/time changed or skipped a solve")

    def stable(self):
        self._require(self.step, self.native_time)
        if _floats(self.solver.mjw_model.opt.timestep, (1,), "native timestep")[0] != self.native_dt:
            raise FasteningFault("native timestep changed during observation")


@dataclass
class _Request:
    kind: str
    arguments: dict
    generation: int
    deadline: float
    result: Future


class _OwnerController(FasteningController):
    def __init__(self, owner):
        self.owner = owner
        super().__init__(owner.backend.binding, owner.backend.limits, owner.journal,
            synthetic=owner.backend.synthetic, close_owner=owner.close, clock=owner.clock)

    def request_turn(self, **arguments):
        return self.owner.request("turn", arguments)

    def reset_stop(self):
        return self.owner.request("reset", {})


class FactorySolveOwner:
    """Priority stop never queues behind admission, solver, reader or logging.

    Normal requests are admitted BETWEEN completed solves. Stop only latches;
    exact zero upload occurs at the next bounded write opportunity and again
    in finally. The independent domain must observe rest. All raw records are
    retained in a bounded queue; a consumer cannot block actuator stop on IO.
    """

    def __init__(self, backend, *, clock=time.monotonic, max_wall_s=120., request_timeout_s=1.,
                 record_capacity=20000):
        if (type(backend.synthetic) is not bool or not math.isfinite(max_wall_s) or max_wall_s <= 0
                or not math.isfinite(request_timeout_s) or not 0 < request_timeout_s <= 1.
                or type(record_capacity) is not int or record_capacity < 3):
            raise ValueError("explicit backend and bounded owner budgets required")
        self.backend, self.clock = backend, clock
        self.max_wall_s, self.request_timeout_s = max_wall_s, request_timeout_s
        self.journal = SolveJournal(backend.binding)
        self.controller = _OwnerController(self)
        self._requests = queue.Queue(maxsize=2)
        self._records = queue.Queue(maxsize=record_capacity)
        self._exit = threading.Event()
        self._thread = None
        self._row = None
        self._error = None
        self._zero_receipt = None
        self._start = None

    def start(self):
        if self._thread is not None or self._exit.is_set():
            raise FasteningFault("owner cannot be restarted or reused across epochs")
        self._start = self.clock()
        self._thread = threading.Thread(target=self._run, name="factory-solve-owner", daemon=True)
        self._thread.start()

    def request(self, kind, arguments):
        if self._thread is None or not self._thread.is_alive() or self._exit.is_set():
            raise FasteningFault("native solve owner is not running")
        ticket = _Request(kind, dict(arguments), self.controller.generation,
                          self.clock()+self.request_timeout_s, Future())
        try:
            self._requests.put_nowait(ticket)
        except queue.Full as exc:
            raise FasteningFault("bounded admission queue is full") from exc
        try:
            return ticket.result.result(timeout=self.request_timeout_s)
        except FutureTimeout as exc:
            ticket.result.cancel()
            self.controller.stop()  # Delivery uncertain: revoke even if admission just happened.
            raise FasteningFault("admission ACK exceeded its original wall budget") from exc

    def _admit_one(self):
        try:
            request = self._requests.get_nowait()
        except queue.Empty:
            return
        if not request.result.set_running_or_notify_cancel():
            return
        try:
            if self.clock() >= request.deadline or request.generation != self.controller.generation:
                raise FasteningFault("queued admission expired or was revoked")
            if request.kind == "turn":
                result = FasteningController.request_turn(self.controller, **request.arguments)
            elif request.kind == "reset":
                result = FasteningController.reset_stop(self.controller)
            else:
                raise FasteningFault("unsupported owner request")
            if self.clock() >= request.deadline:
                self.controller.stop()
                raise FasteningFault("admission completed after its original deadline")
            request.result.set_result(result)
        except Exception as exc:
            request.result.set_exception(exc)

    def _zero(self):
        stamp = self.controller.guard.zero_hold(self.backend.step, self.backend.upload_zero)
        self._zero_receipt = {"uploaded": True, "generation": stamp.generation,
            "before_step": stamp.before_step, "completed_monotonic_s": stamp.completed_monotonic_s,
            "physical_stop_verified": False}
        return stamp

    def cycle(self):
        """One owner-thread cycle, also usable by deterministic synthetic tests."""
        self._admit_one()
        permit = self.controller.guard.current_permit
        if permit is None:
            upload = self._zero
        else:
            target, bounds, effort = self.backend.plan(self._row)
            def upload():
                try:
                    return self.controller.guard.apply(self._row, target, bounds, effort,
                        self.backend.upload, permit=permit)
                except FasteningRevoked:
                    # Same stop ACK generation; no stale proposal or invented
                    # producer fault when stop arrives during preparation.
                    return self._zero()
        row, raw = self.backend.advance(upload)
        self._records.put_nowait(raw)  # Full journal is a producer fault, never silent sample loss.
        self.controller.accept_solve(row)
        self._row = row

    def _run(self):
        try:
            self.backend.enter_owner()
            while not self._exit.is_set():
                if self.clock()-self._start >= self.max_wall_s:
                    raise FasteningFault("owner lifetime budget expired")
                self.cycle()
        except BaseException as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self.controller.stop()
            self.journal.fail(self._error)
        finally:
            try:
                self._zero()
            except BaseException as exc:
                self._zero_receipt = {"uploaded": False, "error": f"{type(exc).__name__}: {exc}",
                                      "physical_stop_verified": False}
                self._error = self._error or self._zero_receipt["error"]
                self.journal.fail(self._error)
            self._exit.set()
            while True:
                try:
                    ticket = self._requests.get_nowait()
                except queue.Empty:
                    break
                if not ticket.result.done():
                    ticket.result.set_exception(FasteningFault("owner closed before admission"))

    def records(self):
        """Drain detached raw records; never read or advance live SDK arrays."""
        result = []
        while True:
            try:
                result.append(self._records.get_nowait())
            except queue.Empty:
                return result

    def close(self, timeout_s=2.):
        if not 0 < timeout_s <= 2.:
            raise ValueError("owner join budget must be in (0,2] seconds")
        self.controller.guard.stop()
        self._exit.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        closed = self._thread is None or not self._thread.is_alive()
        # A blocked SDK operation requires the external process watchdog. Never
        # claim it was closed or that zero reached the device from this latch.
        return {"ok": closed and self._error is None and bool(self._zero_receipt)
                and self._zero_receipt.get("uploaded") is True,
                "owner_thread_closed": closed, "error": self._error,
                "zero_spindle": self._zero_receipt, "pending_records": self._records.qsize(),
                "physical_stop_verified": False}


class FactoryNewtonBackend:
    """Actual one-subsolve adapter; model admission is supplied by bound model.

    The construction module must build/validate the model document before this
    object is exposed to a runtime. No arbitrary completion flag or ArmBase is
    accepted. The bound model owns its immutable fingerprint checks and readback.
    """

    synthetic = False

    def __init__(self, bound_model):
        self.bound_model = bound_model
        self.scene = bound_model.scene
        self.binding, self.limits = bound_model.binding, bound_model.limits
        self.observer, self.geometry, self.joints = bound_model.observer, bound_model.geometry, bound_model.joints
        self.native_clock = NativeSolveClock(self.scene.solver, self.binding.dt_s)
        self._hold_ctrl = bound_model.initial_control.copy()
        self._target = tuple(float(self._hold_ctrl[j.control_index]-j.reference_rad) for j in self.joints[:-1])

    @property
    def step(self):
        return self.native_clock.step

    def enter_owner(self):
        self.scene.wp.set_device(self.scene.model.device)

    def upload(self, target, effort):
        ctrl = self._hold_ctrl.copy()
        for row, q in zip(self.joints[:-1], target, strict=True):
            ctrl[row.control_index] = q + row.reference_rad
        ctrl[self.joints[-1].control_index] = effort
        self.scene.control.mujoco.ctrl.assign(ctrl)
        self._hold_ctrl, self._target = ctrl, tuple(target)

    def upload_zero(self, effort):
        if effort != 0.:
            raise FasteningFault("emergency writer only accepts exact zero spindle effort")
        # Preserve every approved arm target, even if observation/model reading
        # failed. Only the spindle entry changes on this emergency path.
        ctrl = self._hold_ctrl.copy()
        ctrl[self.joints[-1].control_index] = 0.
        self.scene.control.mujoco.ctrl.assign(ctrl)
        self._hold_ctrl = ctrl

    def plan(self, row):
        self.bound_model.check_immutable()
        if row is None:
            raise FasteningFault("no solved baseline for native target planning")
        proposed = np.r_[self.scene._arm_target(np.asarray(row.fastener_position_m)), .6]
        max_delta = self.limits.max_joint_speed_rad_s * self.binding.dt_s
        target = np.asarray(self._target) + np.clip(proposed-self._target, -max_delta, max_delta)
        qpos = _floats(self.scene.solver.mjw_data.qpos, (1, self.scene.solver.mj_model.nq), "qpos")[0]
        bounds = self.geometry.evaluate(qpos, target=target)
        # Same authored socket velocity servo, bounded requested effort; no nut
        # wrench, pose correction or fixed helical joint is introduced.
        effort = float(np.clip(.02*(self.scene.requested_speed_rad_s-row.spindle_speed_rad_s), -.05, .05))
        return tuple(map(float, target)), bounds, effort

    def advance(self, final_upload):
        self.native_clock.before()
        self.bound_model.check_immutable()
        s = self.scene
        s.state.clear_forces()
        s.pipeline.collide(s.state, s.contacts)
        coverage = collision_coverage(s.pipeline, s.contacts)
        # The final guarded upload is the solve admission fence. Identity/FK/
        # collision preparation can be costly and must precede it. A stop or
        # expired lease during preparation therefore cannot drive this solve.
        # A later stop may coincide with the already-admitted SDK solve; that
        # interval retains this upload generation and earns no new-stop credit.
        stamp = final_upload()
        if stamp.before_step != self.step:
            raise FasteningFault("upload stamp does not precede this native solve")
        s.solver.step(s.state, s.next, s.control, s.contacts, self.binding.dt_s)
        s.state, s.next = s.next, s.state
        self.native_clock.after()
        s.step_id = self.step
        s.time_s = self.step*self.binding.dt_s
        captured = self.bound_model.clock()
        row, raw = self.observer.read(stamp, captured, coverage)
        self.native_clock.stable()
        raw["native_clock"] = {"step": self.step, "time_s": float(self.native_clock.native_time),
            "timestep_s": float(self.native_clock.native_dt),
            "interval_clock_s": s.time_s, "cuda_graph": False}
        raw["collision_interval"] = {"before_step": stamp.before_step,
            "generation": stamp.generation, "after_step": self.step}
        return row, raw
