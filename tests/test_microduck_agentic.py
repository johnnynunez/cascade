"""Agentic MicroDuck demo plumbing without Isaac Lab: ToF classification, per-robot map, lockstep world port,
skill runtime + AgentOrchestrator with a scripted LLM, planner, detector stand-in, clearance, scenario configs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading

import numpy as np
import pytest

from cascade.agent.llm import LLMClient, LLMResponse, ToolCall
from cascade.memory.robot_map import RobotMap, classify_frame, summarize_frame
from cascade.sim.microduck_world_port import PortClosed, WorldPort

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------- synthetic sensor
def beams(yaw=0.0):
    """64 world beams of a VL53L8CX-like 8x8 sensor: rows top->bottom (+12..-30 deg), cols left->right."""
    el = np.radians(np.linspace(12, -30, 8))
    az = np.radians(np.linspace(20, -20, 8))
    d = np.zeros((8, 8, 3))
    for r in range(8):
        for c in range(8):
            v = np.array([math.cos(el[r]) * math.cos(az[c]), math.cos(el[r]) * math.sin(az[c]), math.sin(el[r])])
            R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            d[r, c] = R @ v
    return d


def raycast(p0, d, *, floor_z=0.0, wall_x=None, wall_h=0.3, drop_x=None, lower_z=-0.3):
    """Ranges (mm) + status for a floor (optionally ending at drop_x onto lower_z) and an optional wall at x=wall_x."""
    mm = np.zeros(64, np.int32)
    st = np.full(64, 255, np.int32)
    for z, v in enumerate(d.reshape(64, 3)):
        best = np.inf
        if v[2] < 0:
            t = (floor_z - p0[2]) / v[2]
            hit = p0 + v * t
            if drop_x is None or hit[0] < drop_x:
                best = t
            else:
                t2 = (lower_z - p0[2]) / v[2]
                best = t2
        if wall_x is not None and v[0] > 1e-6:
            t = (wall_x - p0[0]) / v[0]
            hz = p0[2] + v[2] * t
            if t > 0 and floor_z <= hz <= floor_z + wall_h and t < best:
                best = t
        if best < 4.0:
            mm[z], st[z] = int(best * 1000), 5
    return mm.reshape(8, 8), st.reshape(8, 8)


def test_classify_floor_obstacle_and_drop():
    p0 = np.array([0.0, 0.0, 0.2])
    d = beams()
    mm, st = raycast(p0, d)
    kind, _, _ = classify_frame(mm, st, p0, d)
    assert set(kind.reshape(8, 8)[0]) == {3}  # upward beams: no target, free
    assert (kind.reshape(8, 8)[4:] == 1).all()  # downward beams: floor
    mm, st = raycast(p0, d, wall_x=0.5)
    r = summarize_frame(mm, st, p0, d, 0.0)
    assert r.nearest_obstacle_m == pytest.approx(0.5, abs=0.02)
    assert abs(r.nearest_obstacle_bearing_deg) <= 21
    assert r.free_ahead_m == pytest.approx(0.5, abs=0.02)
    mm, st = raycast(p0, d, drop_x=0.6)
    r = summarize_frame(mm, st, p0, d, 0.0)
    assert r.drop_ahead and 0.6 <= r.drop_distance_m < 1.0  # upper bound of the edge distance
    assert "DROP-OFF" in r.text()


def test_summary_bearing_follows_robot_yaw():
    p0 = np.array([1.0, 1.0, 0.2])
    yaw = math.pi / 2
    d = beams(yaw)
    mm, st = raycast(p0, d)  # no wall: floor only
    r = summarize_frame(mm, st, p0, d, yaw)
    assert r.nearest_obstacle_m is None and r.free_ahead_m > 0.3


def test_robot_map_integrates_own_frames_only():
    p0 = np.array([0.0, 0.0, 0.2])
    d = beams()
    mm, st = raycast(p0, d, wall_x=0.8)
    a, b = RobotMap("a", (-1, -1, 2, 1)), RobotMap("b", (-1, -1, 2, 1))
    a.integrate(mm, st, p0, d)
    a.add_pose(0, 0, 0)
    a.add_landmark("red box", 0.8, 0.0, 1.0)
    assert a.occupied().any() and a.free().any()
    assert not b.occupied().any() and b.frames == 0  # maps are per robot
    text = a.ascii()
    assert "#" in text and "R" in text and "R" in text.splitlines()[1:][len(text.splitlines()) // 2 - 1] or "R" in text
    s = a.summary()
    assert s["landmarks"][0]["label"] == "red box" and "simulator" in s["pose_source"]
    assert a.image().shape[2] == 3


def test_plan_on_own_map_avoids_known_wall():
    m = RobotMap("a", (-0.5, -1.0, 3.0, 1.0))
    # a known wall at x = 1.0 from y = -1.0 to y = 0.3 (gap above), and known free space around
    m.logodds[:] = -1.0
    for y in np.arange(-1.0, 0.3, 0.02):
        i, j = m.cell(1.0, y)
        m.logodds[j, i] = 2.0
    plan = m.plan((0.0, 0.0), (2.0, 0.0))
    assert plan["ok"] and plan["fraction_through_unknown"] == 0.0
    pts = [(0.0, 0.0)] + [tuple(p) for p in plan["waypoints"]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):  # every segment crossing x=1 does so above the inflated wall
        if (x0 - 1.0) * (x1 - 1.0) < 0:
            y_cross = y0 + (y1 - y0) * (1.0 - x0) / (x1 - x0)
            assert y_cross >= 0.3 + 0.1  # wall end + most of the 0.14 m default inflation
    assert plan["waypoints"][-1] == [2.0, 0.0]
    near_wall = m.plan((0.0, 0.0), (1.05, -0.5))  # goal inside the inflated wall -> nearest free cell
    assert near_wall["ok"] and near_wall["goal_adjusted_to"] is not None
    assert abs(near_wall["goal_adjusted_to"][0] - 1.0) >= 0.14
    escape = m.plan((0.9, -0.5), (0.0, 0.0))  # robot starts inside the inflation band: it may leave it
    assert escape["ok"] and escape["start_inside_inflation"]


# ----------------------------------------------------------------------------------------------- fake world
class FakeWorld:
    """Kinematic stand-in with the MicroduckLabWorld surface (NOT physics)."""

    def __init__(self, n=1, wall_x=None):
        self.names = [f"duck_{i}" for i in range(n)]
        self.n = n
        self.step_index = 0
        self.pose = [[0.0, 0.3 * i, 0.0] for i in range(n)]
        self.ctrl = [("stand", (0, 0, 0), None)] * n
        self.wall_x = wall_x
        self.thread = None

    @property
    def time_s(self):
        return self.step_index * 0.02

    def set_control(self, i, mode, command=(0, 0, 0), goal_xy=None):
        self.ctrl[i] = (mode, tuple(command), goal_xy)

    def step(self, n=1):
        self.thread = threading.current_thread()
        for _ in range(n):
            for i, (mode, cmd, goal) in enumerate(self.ctrl):
                x, y, yaw = self.pose[i]
                if mode == "walk":
                    vx, vy, wz = cmd
                elif mode == "depth_nav":
                    b = math.atan2(goal[1] - y, goal[0] - x) - yaw
                    vx, vy, wz = 0.2, 0.0, max(-1, min(1, 2 * math.atan2(math.sin(b), math.cos(b))))
                elif mode == "onboard_hero":
                    vx, vy, wz = 0.1, 0.0, 0.0
                else:
                    vx = vy = wz = 0.0
                yaw += wz * 0.02
                x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * 0.02
                y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * 0.02
                self.pose[i] = [x, y, yaw]
            self.step_index += 1

    def base_state(self, i):
        x, y, yaw = self.pose[i]
        return {"x": x, "y": y, "z": 0.117, "yaw_rad": yaw, "tilt_deg": 1.0, "fallen": False, "vx_body": 0.0,
                "vy_body": 0.0, "wz": 0.0, "mode": self.ctrl[i][0], "sim_time_s": self.time_s}

    def tof_frame(self, i):
        x, y, yaw = self.pose[i]
        p0 = np.array([x, y, 0.2])
        d = beams(yaw)
        mm, st = raycast(p0, d, wall_x=self.wall_x)
        return {"distance_mm": mm, "status": st, "sensor_pos_w": p0, "beam_dir_w": d, "sim_time_s": self.time_s}

    def policy_rgb(self, i):
        return np.full((36, 48, 3), 100 + 10 * i, np.uint8)


def serve_in_main(port, fn, timeout=30):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    t.start()
    port.serve(lambda: t.is_alive())
    t.join(timeout)
    return out.get("r")


def test_port_lockstep_motion_and_owner_thread(tmp_path):
    w = FakeWorld()
    port = WorldPort(w, tmp_path / "ev.jsonl")
    seen = {}

    def agent():
        t0 = port.call(lambda world: world.time_s)
        r = port.motion(0, "walk", command=(0.2, 0, 0), max_s=1.0)
        seen["t0"], seen["r"] = t0, r
        seen["after"] = port.call(lambda world: world.time_s)
        return r

    r = serve_in_main(port, agent)
    assert seen["t0"] == 0.0  # reads never advance time
    assert r["ok"] and r["stop_reason"] == "duration_elapsed"
    assert r["traveled_m"] == pytest.approx(0.2, abs=0.01)
    assert seen["after"] == pytest.approx(1.0, abs=0.021)
    assert w.thread is threading.main_thread()  # physics only on the owner


def test_port_guard_until_and_close(tmp_path):
    w = FakeWorld(wall_x=0.6)
    port = WorldPort(w)

    def agent():
        a = port.motion(0, "walk", command=(0.2, 0, 0), max_s=5.0, guard={"obstacle_m": 0.22, "drop": 0.3})
        b = port.motion(0, "walk", command=(0, 0, 0.8), max_s=5.0,
                        until=lambda world, i, st, job: "target_yaw_reached" if st["yaw_rad"] > 0.5 else None)
        return a, b

    a, b = serve_in_main(port, agent)
    assert a["stop_reason"].startswith("tof_guard") and a["end_pose"]["x"] < 0.6 - 0.15
    assert b["stop_reason"] == "target_yaw_reached" and b["ok"]
    with pytest.raises(PortClosed):
        port.call(lambda world: 1)


# ----------------------------------------------------------------------------------------------- runtime + agent
class ScriptedLLM(LLMClient):
    supports_vision = False

    def __init__(self, calls):
        self.calls = list(calls)

    def chat(self, system, messages, tools=None, max_tokens=1024):
        if tools is None:
            return LLMResponse(text="1. look\n2. go", tool_calls=[])
        name, args = self.calls.pop(0)
        return LLMResponse(text=f"Reasoning: {name}", tool_calls=[ToolCall(name=name, arguments=args, id="c")])


def test_runtime_with_orchestrator_two_robots_own_memory(tmp_path):
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.skills.microduck_agent_runtime import FleetBoard, MicroduckAgentRuntime

    w = FakeWorld(n=2)
    port = WorldPort(w, tmp_path / "events.jsonl")
    fleet = FleetBoard()
    rts = [MicroduckAgentRuntime(n, i, port, tmp_path / n, bounds_xy=(-1, -1, 3, 1), fleet=fleet)
           for i, n in enumerate(w.names)]
    for rt in rts:
        names = {t["name"] for t in rt.tool_specs}
        assert all(hasattr(rt, f"tool_{n}") for n in names)
        assert {"share_with_fleet", "read_fleet_board", "navigate_to", "get_up"} <= names
    scripts = [
        [("observe", {}), ("walk", {"forward_m_s": 0.2, "duration_s": 1.0}), ("go_to_waypoint", {"x": 0.6, "y": 0.2}),
         ("follow_route", {"waypoints": [[0.9, 0.2], [0.9, -0.2]]}),
         ("mark_landmark", {"label": "crate"}),
         ("share_with_fleet", {"message": "crate ahead"}), ("task_done", {"success": True, "summary": "walked"})],
        [("turn", {"angle_deg": 45}), ("plan_route_on_my_map", {"x": 0.6, "y": 0.4, "inflation_m": 0.3}),
         ("navigate_to", {"x": 0.5, "y": 0.8, "max_s": 6}), ("read_fleet_board", {}),
         ("bearing_to", {"x": 0, "y": 0}), ("my_map", {}), ("task_done", {"success": False, "summary": "partial"})],
    ]
    reports = {}

    def agents():
        ths = []
        for rt, sc in zip(rts, scripts):
            orch = AgentOrchestrator(ScriptedLLM(sc), rt, max_steps=10, decompose=True, attach_images=False,
                                     verify_milestones=False)
            th = threading.Thread(target=lambda o=orch, r=rt: reports.setdefault(r.name, o.run_task("go")))
            th.start()
            ths.append(th)
        for th in ths:
            th.join()
        return reports

    serve_in_main(port, agents)
    assert set(reports) == set(w.names)
    assert rts[0].done == {"success": True, "summary": "walked"}
    wp = [json.loads(ln) for ln in (tmp_path / "events.jsonl").read_text().splitlines()]
    wp = [e for e in wp if e["kind"] == "tool_result" and e["text"] == "go_to_waypoint"][0]["result"]
    assert wp["stop_reason"] == "goal_reached" and wp["distance_to_waypoint_m"] < 0.2
    plans = [e for e in map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines())
             if e["kind"] == "tool_result" and e["text"] == "plan_route_on_my_map"]
    assert plans and plans[0]["result"]["inflations_tried_m"][0] == 0.3
    route = [e for e in map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines())
             if e["kind"] == "tool_result" and e["text"] == "follow_route"][0]["result"]
    assert route["route_complete"] and route["completed_legs"] == 2
    assert rts[1].done["success"] is False
    msgs = fleet.read("duck_1")
    assert msgs and msgs[0]["from"] == "duck_0"
    # memories and maps are per robot
    assert "crate" in rts[0].memory.digest() and "crate" not in rts[1].memory.digest()
    assert rts[0].map.landmarks and not rts[1].map.landmarks
    assert (tmp_path / "duck_0" / "trace.jsonl").exists()
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert any('"tool_result"' in ln for ln in lines)


def test_clearance_flags_possible_contact():
    from cascade.sim.microduck_clearance import apply_to_verification, clearance_report

    boxes = {"Case": (np.array([1.0, -0.2, 0.0]), np.array([1.4, 0.2, 0.3]))}
    pos = np.zeros((3, 1, 2, 3))
    pos[:, 0, 0] = [[0.0, 0, 0.1], [0.5, 0, 0.1], [0.99, 0, 0.1]]  # last frame: 1 cm from the case
    rep = clearance_report(pos, ["a"], boxes)
    assert rep["a"]["possible_contact"] and rep["a"]["closest_obstacle"] == "Case"
    ver = apply_to_verification({"a": {"passed": True}}, {"a": {"no_contact": True}}, rep)
    assert ver["a"]["passed"] is False and "Case" in ver["a"]["failed_because"]
    # a 0.2 m square rotated 45 deg about z (its corners reach 0.141 m along the axes)
    c, sn = math.cos(math.pi / 4), math.sin(math.pi / 4)
    M = np.eye(4)
    M[:2, :2] = [[c, -sn], [sn, c]]
    rot = {"Rot": (np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.2]), M)}
    p_in = np.zeros((1, 1, 1, 3))
    p_in[0, 0, 0] = [0.12, 0.0, 0.1]  # inside the rotated square (its corner reaches 0.141 m along +x)
    p_out = np.zeros((1, 1, 1, 3))
    p_out[0, 0, 0] = [0.12, 0.12, 0.1]  # inside the axis-aligned bound, but 0.17 - 0.1 = 0.07 m from the rotated face
    assert clearance_report(p_in, ["a"], rot)["a"]["min_body_origin_to_obstacle_box_m"] == 0.0
    assert clearance_report(p_out, ["a"], rot)["a"]["min_body_origin_to_obstacle_box_m"] == pytest.approx(0.07, abs=0.002)
    far = clearance_report(pos[:2], ["a"], boxes)
    assert not far["a"]["possible_contact"] and far["a"]["min_body_origin_to_obstacle_box_m"] == 0.5


def test_detector_standin_fov_range_and_occlusion():
    from cascade.sim.microduck_detector_standin import ObjectDetectorStandIn, SceneObject

    red = SceneObject("RedBox", "red box", 2.0, 0.0, 0.1, 0.1)
    blue = SceneObject("BlueBox", "blue box", 0.0, 1.0, 0.1, 0.1)
    wall = SceneObject("Inner_0", "Inner_0", 1.0, 0.0, 0.025, 0.5)
    det = ObjectDetectorStandIn([red, blue], [], max_range_m=2.5)
    seen = det.detect(0.0, 0.0, 0.0)
    assert [d["label"] for d in seen["detections"]] == ["red box"]  # blue is 90 deg off-axis (FOV 82 deg)
    assert "ground truth" in seen["source"]
    assert [d["label"] for d in det.detect(0.0, 0.0, math.pi / 2)["detections"]] == ["blue box"]
    assert det.detect(-1.0, 0.0, 0.0)["detections"] == []  # out of range (3 m)
    occluded = ObjectDetectorStandIn([red], [wall], max_range_m=2.5)
    assert occluded.detect(0.0, 0.0, 0.0)["detections"] == []
    assert occluded.detect(0.0, 1.2, -0.54)["detections"]  # sight line crosses x=1 at y=0.6, beside the wall


# ----------------------------------------------------------------------------------------------- configs + replay
def test_gym_scenario_requires_pinned_external_course(tmp_path, monkeypatch):
    from cascade.apps.microduck_agentic import load_scenario

    monkeypatch.delenv("CASCADE_MICRODUCK_GYM_COURSE", raising=False)
    with pytest.raises(ValueError, match="CASCADE_MICRODUCK_GYM_COURSE"):
        load_scenario(REPO / "configs/microduck_agentic/creative_gym.yaml")
    fake = tmp_path / "course.usda"
    fake.write_text("#usda 1.0\n")
    monkeypatch.setenv("CASCADE_MICRODUCK_GYM_COURSE", str(fake))
    with pytest.raises(ValueError, match="does not match"):
        load_scenario(REPO / "configs/microduck_agentic/creative_gym.yaml")


@pytest.mark.parametrize("name", ["backstage_obstacles", "stage_edge", "multi_room_fetch"])
def test_scenarios_load_and_point_at_existing_scenes(name):
    from cascade.apps.microduck_agentic import load_scenario, verify

    s = load_scenario(REPO / "configs/microduck_agentic" / f"{name}.yaml")
    assert Path(s["scene_path"]).is_file()
    assert {r["name"] for r in s["robots"]} >= set(s["verify"])
    finals = {r["name"]: {"x": 99.0, "y": 99.0, "z": 0.0, "fallen": True} for r in s["robots"]}
    assert not any(v.get("passed") for v in verify(s, finals).values())


def test_verify_reach_and_platform():
    from cascade.apps.microduck_agentic import verify

    scen = {"verify": {"a": {"type": "reach_and_stay_above", "x": 1.0, "y": 0.0, "radius": 0.3, "z": 0.36}}}
    assert verify(scen, {"a": {"x": 1.1, "y": 0.0, "z": 0.42, "fallen": False}})["a"]["passed"]
    assert not verify(scen, {"a": {"x": 1.1, "y": 0.0, "z": 0.12, "fallen": False}})["a"]["passed"]
    near = {"verify": {"c": {"type": "near_object", "footprint": [1.35, 0.75, 1.55, 0.95], "max_m": 0.4}}}
    assert verify(near, {"c": {"x": 0.97, "y": 0.69, "z": 0.12, "fallen": False}})["c"]["passed"]  # 0.385 m
    assert not verify(near, {"c": {"x": 0.90, "y": 0.60, "z": 0.12, "fallen": False}})["c"]["passed"]
    band = {"verify": {"a": {"type": "region", "x_min": 1.7, "x_max": 2.1, "y_min": -0.7, "y_max": 0.7, "z": 0.36}}}
    assert verify(band, {"a": {"x": 1.72, "y": -0.31, "z": 0.42, "fallen": False}})["a"]["passed"]
    assert not verify(band, {"a": {"x": 1.60, "y": -0.31, "z": 0.42, "fallen": False}})["a"]["passed"]
