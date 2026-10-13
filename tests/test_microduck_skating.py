# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the agentic skating MicroDuck demos (no Isaac Lab): track geometry, the obstacle reflex, the skate
runtime's steering/brake tools over the real WorldPort with a kinematic fake world, the pause-while-thinking sync, and
the trajectory verifiers."""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from cascade.apps.microduck_skate_agentic import verify_skate
from cascade.sim.microduck_world_port import WorldPort
from cascade.skills.microduck_agent_runtime import FleetBoard
from cascade.skills.microduck_skate_runtime import Geometry, MicroduckSkateRuntime, Oval


class FakeSkateWorld:
    """Unicycle stand-in for the skate world: speed follows the throttle, heading turns toward the target."""

    dt = 0.02

    def __init__(self, starts):
        self.names = [f"r{i}" for i in range(len(starts))]
        self.n = len(starts)
        self.p = [list(s) for s in starts]  # x, y, yaw
        self.v = [0.0] * self.n
        self.mode = ["coast"] * self.n
        self.throttle = [0.0] * self.n
        self.target = [s[2] for s in starts]
        self.step_index = 0
        self.steps_by_robot_active = 0

    @property
    def time_s(self):
        return self.step_index * self.dt

    def set_control(self, i, mode, command=(0.0, 0.0, 0.0), goal_xy=None):
        mode = "coast" if mode in ("stand", "coast") else mode
        self.mode[i] = mode
        self.throttle[i] = float(command[0]) if mode != "coast" else 0.0
        if mode != "coast" and command is not None and len(command) > 2 and command[2] is not None:
            self.target[i] = float(command[2])

    def step(self, n=1):
        for _ in range(n):
            for i in range(self.n):
                vt = max(0.0, 0.9 * self.throttle[i])
                rate = 4.0 if self.throttle[i] < 0 else 1.5
                self.v[i] += (vt - self.v[i]) * min(1.0, rate * self.dt)
                if self.mode[i] == "coast":
                    self.v[i] *= 0.95
                err = (self.target[i] - self.p[i][2] + math.pi) % (2 * math.pi) - math.pi
                self.p[i][2] += max(-1.0, min(1.0, 3.0 * err)) * self.dt * min(1.0, self.v[i] / 0.2 + 0.2)
                self.p[i][0] += self.v[i] * math.cos(self.p[i][2]) * self.dt
                self.p[i][1] += self.v[i] * math.sin(self.p[i][2]) * self.dt
            self.step_index += 1

    def base_state(self, i):
        x, y, yaw = self.p[i]
        return {"x": x, "y": y, "z": 0.14, "yaw_rad": yaw, "tilt_deg": 0.0, "fallen": False, "speed": self.v[i],
                "mode": self.mode[i], "throttle": self.throttle[i], "target_yaw_rad": self.target[i],
                "sim_time_s": self.time_s}


def _run(world, fn, tmp_path, sync=None):
    port = WorldPort(world, tmp_path / "events.jsonl", max_sim_s=200.0, control_dt_s=world.dt)
    if sync is not None:
        port.sync_running = sync
    out = {}

    def agent():
        try:
            out["r"] = fn(port)
        finally:
            out["done"] = True

    th = threading.Thread(target=agent, daemon=True)
    th.start()
    port.serve(lambda: th.is_alive())
    th.join(5)
    return out.get("r"), port


def _box(cx, cy, hx, hy, h=0.2):
    M = np.eye(4)
    M[:3, 3] = (cx, cy, 0.0)
    return (np.array([-hx, -hy, 0.0]), np.array([hx, hy, h]), M)


def test_oval_fraction_roundtrip_and_lengths():
    o = Oval(1.5, [0.6, 0.9, 1.2])
    for lane in (1, 2, 3):
        for f in np.linspace(0, 3.95, 40):
            x, y = o.point(lane, f)
            assert o.fraction(x, y) == pytest.approx(f, abs=1e-6)
            assert o.lane_of(x, y)[0] == lane
    assert o.length(1) == pytest.approx(9.77, abs=0.01)
    assert o.length(3) == pytest.approx(13.54, abs=0.01)


def test_geometry_ahead_sees_boxes_and_skaters_only_in_the_cone():
    g = Geometry({"Bench": _box(2.0, 0.0, 0.35, 0.125)})
    name, d, b = g.ahead(0.0, 0.0, 0.0)
    assert name == "Bench" and d == pytest.approx(1.65, abs=1e-6) and abs(b) < 1e-6
    assert g.ahead(0.0, 0.0, math.pi)[0] is None  # behind
    name, d, _ = g.ahead(1.0, 0.0, 0.0, others=[("r1", 1.4, 0.05)])
    assert name == "r1" and d == pytest.approx(math.hypot(0.4, 0.05) - 0.12, abs=1e-6)


def test_skate_to_reaches_and_brakes_to_a_halt(tmp_path):
    world = FakeSkateWorld([(0.0, 0.0, 0.0)])

    def agent(port):
        rt = MicroduckSkateRuntime("r0", 0, port, tmp_path / "trace", bounds_xy=[-1, -2, 6, 2], geometry=Geometry({}))
        return rt.execute("skate_to", {"x": 2.0, "y": 0.6, "throttle": 0.5, "radius_m": 0.2, "stop": True})

    r, port = _run(world, agent, tmp_path)
    assert r["stop_reason"] == "goal_reached" and r["closest_approach_m"] <= 0.2
    assert r["brake"]["stop_reason"] == "halted"
    assert world.v[0] < 0.05
    assert "speed_m_s" in r["after"]["pose"]
    kinds = [e["kind"] for e in port.events]
    assert kinds.count("tool_call") == 1 and kinds.count("tool_result") == 1


def test_overshoot_is_reported_as_passed_closest_approach(tmp_path):
    world = FakeSkateWorld([(0.0, 0.0, 0.0)])

    def agent(port):
        rt = MicroduckSkateRuntime("r0", 0, port, tmp_path / "trace", bounds_xy=[-1, -2, 6, 2], geometry=Geometry({}))
        return rt.execute("skate_to", {"x": 1.5, "y": 0.5, "throttle": 0.7, "radius_m": 0.1})

    r, _ = _run(world, agent, tmp_path)
    # a fast unicycle with limited turn rate cannot hit a 0.1 m circle 18 deg off its nose: it reports the overshoot
    assert r["stop_reason"] in ("passed_closest_approach", "goal_reached")
    assert r["closest_approach_m"] < 0.7


def test_obstacle_reflex_brakes_before_the_bench(tmp_path):
    world = FakeSkateWorld([(0.0, 0.0, 0.0)])

    def agent(port):
        rt = MicroduckSkateRuntime("r0", 0, port, tmp_path / "trace", bounds_xy=[-1, -2, 6, 2],
                                   geometry=Geometry({"Bench": _box(2.0, 0.0, 0.35, 0.125)}))
        return rt.execute("skate_to", {"x": 3.0, "y": 0.0, "throttle": 0.6})

    r, _ = _run(world, agent, tmp_path)
    assert r["stop_reason"].startswith("obstacle_brake: Bench")
    assert world.p[0][0] < 2.0 - 0.35


def test_sync_pauses_the_world_while_an_agent_thinks(tmp_path):
    world = FakeSkateWorld([(0.0, 0.0, 0.0), (0.0, -1.0, 0.0)])
    running = {0, 1}
    gate = threading.Event()
    t_seen = {}

    def agent(port):
        rts = [MicroduckSkateRuntime(f"r{i}", i, port, tmp_path / f"t{i}", bounds_xy=[-1, -2, 6, 2],
                                     geometry=Geometry({}), fleet=FleetBoard(), robot_names=["r0", "r1"])
               for i in range(2)]

        def slow():  # robot 1 "thinks" for a while before it moves
            gate.wait(2.0)
            t_seen["r1_start"] = port.world.time_s
            rts[1].execute("set_speed", {"throttle": 0.4, "duration_s": 1.0})
            running.discard(1)

        th = threading.Thread(target=slow, daemon=True)
        th.start()
        threading.Timer(0.3, gate.set).start()
        r = rts[0].execute("set_speed", {"throttle": 0.4, "duration_s": 1.0})
        running.discard(0)
        th.join(5)
        return r

    r, _ = _run(world, agent, tmp_path, sync=lambda: set(running))
    assert t_seen["r1_start"] == pytest.approx(0.0, abs=1e-9)  # no sim time passed while robot 1 was thinking
    assert r["stop_reason"] == "duration_elapsed"


def test_verify_slalom_and_race():
    t = np.arange(0, 10, 0.02)
    xs = np.linspace(0, 6.6, len(t))
    ys = 0.4 * np.sin(xs / 6.6 * 2 * math.pi * 1.5)
    T = len(t)
    pose = np.zeros((T, 1, 19, 7), np.float32)
    pose[:, 0, 0, 0], pose[:, 0, 0, 1], pose[:, 0, 0, 2] = xs, ys, 0.14
    pose[-10:, 0, 0, 0] = 6.6  # standing still at the end
    pose[-10:, 0, 0, 1] = 0.0
    rec = {"body_pose": pose, "time_s": t, "proj_grav": np.tile([0, 0, -1.0], (T, 1, 1))}
    spec = {"type": "slalom", "pass_radius_m": 0.45, "stop": {"x": 6.6, "y": 0.0, "radius": 0.35},
            "cones": [{"name": "A", "x": 1.0, "y": 0.4}, {"name": "B", "x": 3.3, "y": -0.3}]}
    out = verify_skate({"verify": {"r0": spec}}, rec, ["r0"], 0, {})
    v = out["r0"]
    assert v["in_order"] and v["stopped"] and v["passed"], v
    spec["cones"] = spec["cones"][::-1]  # wrong order
    assert not verify_skate({"verify": {"r0": spec}}, rec, ["r0"], 0, {})["r0"]["passed"]
    rec["proj_grav"][T // 2:, 0] = [1.0, 0, 0]  # tipped over: fell
    v = verify_skate({"verify": {"r0": {**spec, "cones": spec["cones"][::-1]}}}, rec, ["r0"], 0, {})["r0"]
    assert v["fell"] and not v["passed"]
