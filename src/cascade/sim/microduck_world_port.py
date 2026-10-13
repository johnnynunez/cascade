# SPDX-License-Identifier: Apache-2.0
"""Thread bridge between agent threads and a lock-stepped MicroDuck simulation world.

All simulator access happens on the thread that calls :meth:`WorldPort.serve` (the physics owner). Agent threads
submit reads (:meth:`call`) and bounded motions (:meth:`motion`) and block until the owner completes them.

Clock rule (lockstep): the world advances only while at least one motion is active. While every agent is thinking,
simulation time stands still, so the recorded timeline contains physics only; LLM latency is logged as events.
Robots without an active motion run the learned walker at zero twist (standing balance), never a frozen pose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import queue
import threading
import time

from ..memory.robot_map import summarize_frame


class PortClosed(RuntimeError):
    pass


@dataclass
class _Request:
    kind: str
    payload: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None


@dataclass
class _Job:
    robot: int
    mode: str
    command: tuple
    goal: tuple | None
    max_steps: int
    until: object
    guard: dict | None
    request: _Request
    start_step: int
    start_state: dict
    label: str
    upright_steps: int = 0
    min_free_ahead: float = 99.0


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class WorldPort:
    """Serve one world to N agent threads. ``world`` needs: names, time_s, step_index, step(n), set_control(i, mode,
    command, goal_xy), base_state(i), tof_frame(i), policy_rgb(i)."""

    def __init__(self, world, events_path: Path | None = None, *, floor_z: float = 0.0, max_sim_s: float = 300.0,
                 control_dt_s: float = 0.02):
        self.world = world
        self.floor_z = float(floor_z)
        self.max_sim_s = float(max_sim_s)
        self.dt = float(control_dt_s)
        self._q: queue.Queue[_Request] = queue.Queue()
        self._jobs: dict[int, _Job] = {}
        self._closed = False
        self._lock = threading.Lock()
        self.events: list[dict] = []
        self._events_path = Path(events_path) if events_path else None
        self._fh = open(self._events_path, "a", encoding="utf-8") if self._events_path else None
        self.on_step = []  # callables(world) run on the owner after every physics step
        # Optional: callable returning the robot indices whose agents are still running. When set, the world only steps
        # while EVERY such robot has an active motion, i.e. simulation time is paused whenever any agent is thinking
        # (used by the shared-world skating demos, where a robot waiting for its LLM would otherwise lose race time).
        self.sync_running = None

    # ---------------------------------------------------------------- agent side
    def log_event(self, robot: str, kind: str, text: str = "", **data) -> dict:
        ev = {"sim_time_s": round(self.world.time_s, 3), "step": int(self.world.step_index), "wall_time": time.time(),
              "robot": robot, "kind": kind, "text": text, **data}
        with self._lock:
            self.events.append(ev)
            if self._fh:
                self._fh.write(json.dumps(ev, default=str) + "\n")
                self._fh.flush()
        return ev

    def _submit(self, kind, **payload):
        if self._closed:
            raise PortClosed("simulation owner has stopped")
        req = _Request(kind, payload)
        self._q.put(req)
        while not req.done.wait(0.5):
            if self._closed:
                raise PortClosed("simulation owner has stopped")
        if req.error is not None:
            raise req.error
        return req.result

    def call(self, fn):
        """Run ``fn(world)`` on the physics owner; returns its result."""
        return self._submit("call", fn=fn)

    def motion(self, robot: int, mode: str, *, command=(0.0, 0.0, 0.0), goal=None, max_s: float = 5.0, until=None,
               guard: dict | None = None, label: str = "") -> dict:
        """Run one bounded controller episode for ``robot``; blocks until it ends. Returns a receipt dict."""
        return self._submit("motion", robot=robot, mode=mode, command=tuple(command), goal=goal,
                            max_steps=max(1, int(round(max_s / self.dt))), until=until, guard=guard, label=label)

    # ---------------------------------------------------------------- owner side
    def close(self):
        self._closed = True
        while True:
            try:
                req = self._q.get_nowait()
            except queue.Empty:
                break
            req.error = PortClosed("simulation owner has stopped")
            req.done.set()
        for job in list(self._jobs.values()):
            job.request.error = PortClosed("simulation owner has stopped")
            job.request.done.set()
        self._jobs.clear()
        if self._fh:
            self._fh.close()
            self._fh = None

    def _start(self, req: _Request):
        p = req.payload
        i = p["robot"]
        if i in self._jobs:
            raise RuntimeError("robot already has an active motion")
        w = self.world
        w.set_control(i, p["mode"], p["command"], p["goal"])
        self._jobs[i] = _Job(i, p["mode"], p["command"], p["goal"], p["max_steps"], p["until"], p["guard"], req,
                             w.step_index, w.base_state(i), p["label"])

    def _guard(self, job: _Job, state: dict):
        g = job.guard
        if not g or job.command[0] <= 0.02:  # the guard protects forward travel only (turning away stays allowed)
            return None
        if (self.world.step_index - job.start_step) % 5:
            return None
        f = self.world.tof_frame(job.robot)
        r = summarize_frame(f["distance_mm"], f["status"], f["sensor_pos_w"], f["beam_dir_w"], state["yaw_rad"],
                            floor_z=self.floor_z)
        job.min_free_ahead = min(job.min_free_ahead, r.free_ahead_m)
        if g.get("obstacle_m") and r.nearest_obstacle_m is not None and abs(r.nearest_obstacle_bearing_deg) < 22 \
                and r.nearest_obstacle_m < g["obstacle_m"]:
            return f"tof_guard: obstacle {r.nearest_obstacle_m:.2f} m ahead"
        if g.get("drop") and r.drop_ahead and r.drop_distance_m is not None and r.drop_distance_m < g["drop"]:
            return f"tof_guard: drop-off {r.drop_distance_m:.2f} m ahead"
        return None

    def _finish(self, job: _Job, reason: str):
        w = self.world
        w.set_control(job.robot, "stand")
        end = w.base_state(job.robot)
        s0 = job.start_state
        receipt = {
            "ok": reason in ("completed", "duration_elapsed", "goal_reached", "upright", "target_yaw_reached"),
            "controller": job.mode, "stop_reason": reason, "label": job.label,
            "sim_time_start_s": round(job.start_step * self.dt, 3), "sim_time_end_s": round(w.time_s, 3),
            "start_pose": {k: round(s0[k], 3) for k in ("x", "y", "yaw_rad")},
            "end_pose": {k: round(end[k], 3) for k in ("x", "y", "yaw_rad")},
            "traveled_m": round(math.hypot(end["x"] - s0["x"], end["y"] - s0["y"]), 3),
            "yaw_change_deg": round(math.degrees(angle_diff(end["yaw_rad"], s0["yaw_rad"])), 1),
            "fallen": end["fallen"], "tilt_deg": round(end["tilt_deg"], 1),
            "verification": ("end pose = the robot's own pose estimate (onboard visual odometry), read after the motion"
                             if hasattr(w, "camera_rgb") else
                             "end pose read from the completed physics state, not integrated from the command"),
        }
        del self._jobs[job.robot]
        job.request.result = receipt
        job.request.done.set()

    def serve(self, agents_alive) -> str:
        """Owner loop. Returns why it stopped: 'agents_done' or 'sim_time_limit'."""
        w = self.world
        try:
            waiting = False
            while True:
                try:
                    req = self._q.get(timeout=0.02 if (not self._jobs or waiting) else 0.0)
                except queue.Empty:
                    req = None
                while req is not None:
                    try:
                        if req.kind == "call":
                            req.result = req.payload["fn"](w)
                            req.done.set()
                        else:
                            self._start(req)
                    except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                        req.error = exc
                        req.done.set()
                    try:
                        req = self._q.get_nowait()
                    except queue.Empty:
                        req = None
                if not self._jobs:
                    if not agents_alive():
                        return "agents_done"
                    continue
                if self.sync_running is not None:
                    waiting = any(i not in self._jobs for i in self.sync_running())
                    if waiting:
                        if not agents_alive():
                            return "agents_done"
                        continue
                if w.time_s >= self.max_sim_s:
                    for job in list(self._jobs.values()):
                        self._finish(job, "sim_time_limit")
                    return "sim_time_limit"
                w.step(1)
                for fn in self.on_step:
                    fn(w)
                for job in list(self._jobs.values()):
                    state = w.base_state(job.robot)
                    n = w.step_index - job.start_step
                    reason = None
                    if state["fallen"] and job.mode != "getup":
                        reason = "fell"
                    if reason is None and job.until is not None:
                        reason = job.until(w, job.robot, state, job)
                    if reason is None:
                        reason = self._guard(job, state)
                    if reason is None and n >= job.max_steps:
                        reason = "duration_elapsed" if job.until is None else "timeout"
                    if reason is not None:
                        self._finish(job, reason)
        finally:
            self.close()


__all__ = ["WorldPort", "PortClosed", "angle_diff"]
