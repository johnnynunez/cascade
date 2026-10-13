# SPDX-License-Identifier: Apache-2.0
"""Agent-facing skill runtime for ONE simulated MicroDuck in an agentic demo.

Plugs into :class:`cascade.agent.orchestrator.AgentOrchestrator` as a ``robot_mode = "mobile"`` runtime: the
orchestrator owns the LLM loop (one tool call per turn, ``task_done`` ends it); this runtime owns the tools. Each robot
has its OWN :class:`~cascade.memory.episodic.EpisodicMemory` (sim-time clock), its OWN
:class:`~cascade.memory.robot_map.RobotMap` (ToF occupancy + landmarks) and its own trace directory. Robots only learn
about each other through the optional fleet board (explicit messages).

Every motion tool runs one of the unchanged learned policies through :class:`~cascade.sim.microduck_world_port.WorldPort`
and returns a receipt whose end pose is read from completed physics. Controllers:

* ``walk`` / ``turn``  -> learned velocity walker (velocity_flat_existing.onnx), agent-chosen twist;
* ``walk_with_onboard_perception`` -> onboard_hero actor (proprio + ToF 8x8 + head RGB 48x36 -> joints), forward only;
* ``navigate_to`` -> learned depth navigation policy (depth_navigation_v4) commanding the walker toward a point;
* ``get_up`` -> learned get-up policy.
The ToF guard on ``walk`` (stop when an obstacle is within ``0.22 m`` ahead or the floor drops away) is a scripted
safety reflex of this runtime, reported in the receipt as ``stop_reason``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading

import numpy as np

from ..agent.trace import TraceLogger
from ..memory.episodic import EpisodicMemory
from ..memory.robot_map import RobotMap, summarize_frame
from ..sim.microduck_world_port import PortClosed, WorldPort, angle_diff

SYSTEM_PROMPT = """You are the onboard agent of ONE MicroDuck, a 30 cm bipedal duck robot in a physics simulation.
You decide what the robot does by calling exactly one tool per turn; learned policies execute the motion.
Your senses: a VL53L8CX-style 8x8 time-of-flight sensor (45 deg field of view, 4 m) and a tiny 48x36 head RGB camera,
both in the head; plus your own pose and posture. The images you receive are that 48x36 head RGB, upscaled.
World frame: metres, +x forward from the start area, +y left, yaw in degrees counter-clockwise.
Every motion receipt already contains your new pose and a fresh ToF summary ("after"), so you do not need to call
observe after each motion; call observe when you want the camera view or the full 8x8 grid.
Use bearing_to(x, y) to turn a world point into distance + relative bearing before turning toward it.
Budget: you have a limited number of tool calls. Be decisive: when the ToF says the way is clear, move.
The walker is slow: a command of 0.1 m/s actually covers ~0.06 m/s, 0.25 m/s covers ~0.18 m/s.
go_to_waypoint / follow_route are the efficient way to cover distance once you have chosen collision-free waypoints;
they turn toward each waypoint themselves (no need to call bearing_to or turn first). plan_route_on_my_map gives
collision-free waypoints on your own map; after a guard stop, look (scan) and re-plan instead of guessing.
Distances in your task are measured from your base position (the pose in every receipt), not from the head ToF
sensor, which sits a few cm ahead of it: "within 0.4 m of a box" means base-to-box-surface, so a ToF reading of
0.35 m to that surface leaves you ~0.4 m away. Use detected world positions and your pose to check distances.
Efficient loop: scan_surroundings once -> plan_route_on_my_map(goal) -> follow_route(first <= 6 waypoints) -> if it
stopped early, scan_surroundings and re-plan. Avoid bearing_to/turn/my_map calls that this loop makes unnecessary.
The ToF only sees a ~45 deg cone ahead: the guard cannot protect your sides while you turn or pass an obstacle, so
keep your own waypoints well clear of obstacle edges (your body is ~0.15 m wide; a 0.5 m doorway leaves ~0.17 m on each
side, pass it through its centre line). plan_route_on_my_map already applies the clearance; keep its default. Every motion receipt already
carries a fresh ToF summary, so do not call observe after each move.
Rules: look before moving; walking speed 0.15-0.3 m/s;
if you fall, call get_up; keep at least 0.25 m from obstacles and edges; never claim success you did not measure:
your own end pose and the sensors are the evidence. Finish with task_done(success, summary).
"""


def _num(args, key, default, lo, hi):
    try:
        v = float(args.get(key, default))
    except (TypeError, ValueError):
        v = float(default)
    if not math.isfinite(v):
        v = float(default)
    return min(max(v, lo), hi)


def _spec(name, description, props=None, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props or {}, "required": list(required),
                           "additionalProperties": False}}


BASE_TOOLS = [
    _spec("observe", "Read your pose, posture, active policy and the latest ToF 8x8 frame (summarised and as a decimetre "
                     "grid: positive = floor hit, negative = obstacle hit, 99 = no target, 0 = unusable). Also refreshes "
                     "the head camera view you see and adds the frame to YOUR map."),
    _spec("detect_objects", "List the scenario objects inside your head camera's field of view (82 deg, up to 2.5 m, "
                            "walls occlude) with bearing, distance and world position. This is a DETECTOR STAND-IN: "
                            "simulator ground truth restricted to what the camera could see (the 48x36 head RGB only "
                            "draws the first 1.5 m). Detections are added to your own map as landmarks.",
          {}, ()),
    _spec("walk", "Walk with the learned walker for a bounded time. forward 0..0.3 m/s (negative = backwards, max 0.2), "
                  "lateral -0.2..0.2 m/s (+ = left), turn -1..1 rad/s (+ = left). Stops early if the ToF guard sees an "
                  "obstacle closer than 0.22 m ahead or a drop-off (edge) within about 0.45 m.",
          {"forward_m_s": {"type": "number"}, "lateral_m_s": {"type": "number"}, "turn_rad_s": {"type": "number"},
           "duration_s": {"type": "number", "description": "0.5..4"}}, ("forward_m_s", "duration_s")),
    _spec("go_to_waypoint", "Walk to a world point you chose (x, y): a scripted heading/speed loop commands the LEARNED "
                            "walker (turn toward the point, walk up to 0.25 m/s), with the same ToF stop guard as walk. "
                            "Ends within 0.15 m of the point, on an obstacle/drop-off guard stop, a fall, or after max_s "
                            "(<= 20). Choose waypoints that keep 0.25 m from obstacles; it does not plan around them.",
          {"x": {"type": "number"}, "y": {"type": "number"}, "max_s": {"type": "number"}}, ("x", "y")),
    _spec("follow_route", "Walk through a list of 1-6 world waypoints you chose, in order, with go_to_waypoint for each "
                          "leg (learned walker, ToF guard). Stops at the first leg that does not reach its waypoint "
                          "(guard stop, fall or timeout) and reports every leg. Use it once you have a clear route.",
          {"waypoints": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                         "description": "[[x, y], ...] in metres, world frame"},
           "max_s_per_leg": {"type": "number"}}, ("waypoints",)),
    _spec("plan_route_on_my_map", "A* path from your pose to (x, y) on YOUR OWN ToF map: obstacles and drop-offs are "
                                  "inflated by inflation_m (default 0.14 m; your body half-width is ~0.08 m), unknown "
                                  "space is allowed (optimistic). Returns waypoints for follow_route plus the fraction "
                                  "through unknown space. It does not move the robot; re-plan after new observations. "
                                  "Gaps of ~0.35-0.5 m between equipment or in doorways need inflation_m <= 0.15; if no "
                                  "path is found, retry with less inflation (min 0.10) or look closer before giving up.",
          {"x": {"type": "number"}, "y": {"type": "number"}, "inflation_m": {"type": "number"}}, ("x", "y")),
    _spec("turn", "Turn in place by a measured angle in degrees (+ = left/counter-clockwise), learned walker.",
          {"angle_deg": {"type": "number"}}, ("angle_deg",)),
    _spec("walk_with_onboard_perception", "Hand control to the learned onboard-perception actor (inputs: ToF 8x8 + "
          "head RGB 48x36 + proprioception). It walks forward at ~0.1 m/s and steers by itself; trained on ramp/step "
          "lanes. You only choose the duration (1..10 s).",
          {"duration_s": {"type": "number"}}, ("duration_s",)),
    _spec("navigate_to", "Hand control to the learned depth-navigation policy (head depth 64x48 + goal) driving the "
                         "walker toward a point in the world frame. Ends at 0.25 m from the goal, after max_s (<= 30) "
                         "or on a fall. It was trained on backstage obstacle courses and arrives in about 1 of 5 "
                         "held-out courses; check the receipt.",
          {"x": {"type": "number"}, "y": {"type": "number"}, "max_s": {"type": "number"}}, ("x", "y")),
    _spec("get_up", "Run the learned get-up policy until you are upright (max 8 s). Use after a fall."),
    _spec("scan_surroundings", "Turn in place a full circle in four 90 deg steps (ending at the starting heading), "
                               "reading the ToF at each heading; returns free distance per heading and grows YOUR map."),
    _spec("bearing_to", "Distance and relative bearing (deg, + = left of your current heading) from you to a world "
                        "point. Pure geometry from your pose; does not move.",
          {"x": {"type": "number"}, "y": {"type": "number"}}, ("x", "y")),
    _spec("my_map", "Show YOUR own map (built only from your ToF frames): ASCII top view, explored fraction, landmarks."),
    _spec("mark_landmark", "Remember a named landmark in YOUR map at a point ahead of you (distance_ahead_m 0..2).",
          {"label": {"type": "string"}, "distance_ahead_m": {"type": "number"}, "note": {"type": "string"}},
          ("label",)),
    _spec("recall_memory", "Read YOUR episodic memory (actions, outcomes, notes) of this session.",
          {"query": {"type": "string"}}),
    _spec("task_done", "End the task honestly. success must reflect measured evidence.",
          {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ("success", "summary")),
]
FLEET_TOOLS = [
    _spec("share_with_fleet", "Post a short message to the fleet board (other robots read it with read_fleet_board). "
                              "Your current position is attached automatically.", {"message": {"type": "string"}},
          ("message",)),
    _spec("read_fleet_board", "Read the messages other robots posted (with their positions at posting time)."),
]
HONEST_PROMPT = """HONEST ONBOARD MODE: you only know what this robot itself senses.
* Your pose is your OWN estimate in YOUR START FRAME: (0, 0) is where you started, +x your initial forward direction,
  +y your initial left. Heading from your IMU (good); position from leg odometry (joint encoders) corrected by visual
  odometry on your head camera with ToF depth. It drifts: expect 2-4 % of the distance walked (tens of centimetres
  over a course). "LOST" means the camera odometry failed: slow down and re-observe. Task distances are in this frame.
* Your map is built with that drifting pose, so an obstacle you saw a while ago may be 0.1-0.3 m off where your map
  shows it, and your ToF only sees a 45 deg cone AHEAD: it cannot see equipment beside you while you walk past it.
  So pass equipment with margin: plan with inflation_m 0.20-0.25 where the gap allows; when you must squeeze through
  a narrow gap, first face it and observe it from close by, then walk straight through.
* Fast in-place turns hurt your odometry; prefer slow turns and smooth arcs.
* detect_objects runs a real image detector (YOLOE) on your camera image. It was trained on real photos, so its names
  for simple props can be odd (a dark box may come back as "pulpit"); judge by colour, size and position. Its range
  comes from your ToF when the object is in the ToF's 45 deg cone, otherwise from the floor contact point (rougher).
* The image you receive is your own head camera (640x360, downscaled).
* Do not give up while you are upright and have steps left: certainty is never available on a real robot. If you
  cannot verify something, get closer and look again; call task_done only when you believe the task is done or you
  cannot continue."""
HONEST_DETECT = _spec("detect_objects", "Run the onboard object detector (YOLOE, open vocabulary) on your current head "
                                        "camera image: label (with dominant colour), confidence, bearing, distance and "
                                        "position in your own start frame (from your pose estimate). Real detector: it "
                                        "can miss things or give odd names. Detections become landmarks in your map.")
MOTION_TOOLS = frozenset({"walk", "turn", "go_to_waypoint", "follow_route", "walk_with_onboard_perception", "navigate_to", "get_up",
                          "scan_surroundings"})


class FleetBoard:
    def __init__(self):
        self._lock = threading.Lock()
        self.messages: list[dict] = []

    def post(self, robot, text, x, y, t):
        with self._lock:
            self.messages.append({"from": robot, "text": str(text)[:400], "x": round(x, 2), "y": round(y, 2),
                                  "sim_time_s": round(t, 2)})

    def read(self, robot):
        with self._lock:
            return [m for m in self.messages if m["from"] != robot]


class MicroduckAgentRuntime:
    robot_mode = "mobile"

    def __init__(self, name: str, index: int, port: WorldPort, run_dir: Path, *, bounds_xy, floor_z: float = 0.0,
                 fleet: FleetBoard | None = None, has_nav: bool = True, has_getup: bool = True,
                 extra_prompt: str = "", detector=None, honest: bool = False):
        self.name, self.index, self.port = name, int(index), port
        self.detector = detector
        self.run_dir = Path(run_dir)
        self.floor_z = float(floor_z)
        self.fleet = fleet
        self.memory = EpisodicMemory(horizon_s=600.0, frame_horizon_s=600.0, max_frames=24,
                                     clock=lambda: port.world.time_s)
        self.trace = TraceLogger(self.run_dir)
        self.map = RobotMap(name, bounds_xy)
        self.map_snapshots: list[tuple[float, dict]] = []
        self.current_tier = None
        self.pending_plausibility = None
        self.last_path = None
        self.last_frame = None
        tools = [t for t in BASE_TOOLS
                 if (has_nav or t["name"] != "navigate_to") and (has_getup or t["name"] != "get_up")]
        if detector is None:
            tools = [t for t in tools if t["name"] != "detect_objects"]
        if fleet is not None:
            tools = tools[:-1] + FLEET_TOOLS + tools[-1:]
        self.tool_specs = tools
        self.system_prompt = SYSTEM_PROMPT + f"\nYou are robot '{name}'." + (("\n" + extra_prompt) if extra_prompt else "")
        self.done = None
        self.honest = bool(honest)
        if self.honest:
            self.tool_specs = [HONEST_DETECT if t["name"] == "detect_objects" else t for t in self.tool_specs
                               if t["name"] != "navigate_to"]
            self.system_prompt = (SYSTEM_PROMPT.replace("Use detected world positions and your pose", "Use detections "
                                  "and your own pose estimate") + "\n" + HONEST_PROMPT + f"\nYou are robot '{name}'."
                                  + (("\n" + extra_prompt) if extra_prompt else ""))

    # -------------------------------------------------------------- helpers
    def frame_jpeg(self):
        if self.last_frame is None:
            return None
        import cv2

        big = self.last_frame.shape[1] > 100
        img = cv2.resize(self.last_frame, (320, 180) if big else (192, 144),
                         interpolation=cv2.INTER_AREA if big else cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else None

    def _sense(self, integrate=True):
        i = self.index

        def read(w):
            cam = w.camera_rgb(i) if hasattr(w, "camera_rgb") else None
            return w.base_state(i), w.tof_frame(i), w.policy_rgb(i), cam

        state, tof, rgb, cam = self.port.call(read)
        if integrate:
            self.map.integrate(tof["distance_mm"], tof["status"], tof["sensor_pos_w"], tof["beam_dir_w"],
                               floor_z=self.floor_z)
            self.map.add_pose(state["x"], state["y"], state["yaw_rad"])
            self.map_snapshots.append((state["sim_time_s"], self.map.snapshot()))
        reading = summarize_frame(tof["distance_mm"], tof["status"], tof["sensor_pos_w"], tof["beam_dir_w"],
                                  state["yaw_rad"], floor_z=self.floor_z)
        self.last_frame = cam if cam is not None else rgb
        return state, reading, rgb

    @staticmethod
    def _pose(state):
        out = {"x": round(state["x"], 2), "y": round(state["y"], 2), "yaw_deg": round(math.degrees(state["yaw_rad"]), 1),
               "posture": "FALLEN" if state["fallen"] else "upright", "tilt_deg": round(state["tilt_deg"], 1)}
        if "localization" in state:
            out["localization"] = state["localization"]
            out["pose_source"] = "your own visual odometry estimate (start frame)"
        return out

    def _after_motion(self, receipt, name):
        state, reading, rgb = self._sense()
        receipt["after"] = {"pose": self._pose(state), "tof": reading.text()}
        if self.detector is not None:
            receipt["after"]["objects_in_view"] = [
                f"{d['label']} {d['distance_m']} m at {d['bearing_deg']} deg -> "
                f"{'your frame' if self.honest else 'world'} {d['world_xy']}"
                for d in self._detections(state)["detections"]] or (
                "none detected" if self.honest else "none (detector stand-in)")
        verdict = "confirmed" if receipt.get("ok") and not state["fallen"] else "unverified"
        self.memory.add("action", f"{name}: {receipt.get('stop_reason')} -> at ({state['x']:.2f},{state['y']:.2f}), "
                                  f"{'FALLEN' if state['fallen'] else 'upright'}",
                        data={"verdict": verdict}, rgb=rgb[..., ::-1])
        return receipt

    # -------------------------------------------------------------- dispatch
    def begin_task(self):
        self.done = None

    def execute(self, name, args=None):
        args = dict(args or {})
        self.port.log_event(self.name, "tool_call", name, args=args)
        import time as _time

        t0 = _time.monotonic()
        try:
            fn = getattr(self, f"tool_{name}", None)
            if fn is None or name not in {t["name"] for t in self.tool_specs}:
                result = {"ok": False, "error": f"unknown tool {name!r}"}
            else:
                result = fn(args)
        except PortClosed:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure becomes an honest tool result
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.port.log_event(self.name, "tool_result", name, result=result)
        self.trace.record(skill=name, args=args, result=result, duration_ms=(_time.monotonic() - t0) * 1000.0,
                          tier=self.current_tier, context={"robot": self.name})
        return result

    # -------------------------------------------------------------- tools
    def tool_observe(self, args):
        state, reading, rgb = self._sense()
        self.memory.add("observation", f"observe: {reading.text()}", rgb=rgb[..., ::-1])
        return {"ok": True, "pose": self._pose(state), "active_policy": state["mode"],
                "sim_time_s": round(state["sim_time_s"], 2), "tof_summary": reading.text(),
                "tof_grid_dm_rows_top_to_bottom": reading.grid_dm,
                "camera": "head RGB 48x36 attached as the current view"}

    def _detections(self, state):
        if self.detector is None:
            return None
        d = self.detector.detect(state["x"], state["y"], state["yaw_rad"])
        for det in d["detections"]:
            if det.get("world_xy") is None:
                continue
            self.map.add_landmark(det["label"], det["world_xy"][0], det["world_xy"][1], state["sim_time_s"],
                                  note="onboard detector" if self.honest else "detector stand-in")
        return d

    def tool_detect_objects(self, args):
        state, _, _ = self._sense()
        d = self._detections(state)
        self.memory.add("observation", "detect_objects: " + (", ".join(
            f"{x['label']} at {x['distance_m']} m, bearing {x['bearing_deg']} deg" for x in d["detections"]) or "nothing"))
        return {"ok": True, "pose": self._pose(state), **d}

    def tool_walk(self, args):
        vx = _num(args, "forward_m_s", 0.15, -0.2, 0.3)
        vy = _num(args, "lateral_m_s", 0.0, -0.2, 0.2)
        wz = _num(args, "turn_rad_s", 0.0, -1.0, 1.0)
        dur = _num(args, "duration_s", 2.0, 0.5, 4.0)
        guard = {"obstacle_m": 0.22, "drop": 0.45} if vx > 0.02 else None
        r = self.port.motion(self.index, "walk", command=(vx, vy, wz), max_s=dur, guard=guard,
                             label=f"walk vx={vx:.2f} vy={vy:.2f} wz={wz:.2f} {dur:.1f}s")
        r["commanded"] = {"forward_m_s": vx, "lateral_m_s": vy, "turn_rad_s": wz, "duration_s": dur}
        r["policy"] = "learned walker velocity_flat_existing.onnx"
        return self._after_motion(r, "walk")

    def tool_go_to_waypoint(self, args):
        gx, gy = _num(args, "x", 0.0, -50, 50), _num(args, "y", 0.0, -50, 50)
        max_s = _num(args, "max_s", 15.0, 1.0, 20.0)
        i = self.index

        def until(w, idx, st, job):
            dx, dy = gx - st["x"], gy - st["y"]
            dist = math.hypot(dx, dy)
            if dist < 0.15:
                return "goal_reached"
            err = angle_diff(math.atan2(dy, dx), st["yaw_rad"])
            wz = max(-0.9, min(0.9, 1.6 * err))
            vx = 0.0 if abs(err) > 0.7 else min(0.25, 0.6 * dist + 0.08) * (1.0 - abs(err) / 0.7) ** 0.5
            w.set_control(idx, "walk", (vx, 0.0, wz))
            job.command = (vx, 0.0, wz)
            return None

        r = self.port.motion(i, "walk", command=(0.0, 0.0, 0.0), max_s=max_s, until=until,
                             guard={"obstacle_m": 0.22, "drop": 0.45}, label=f"waypoint ({gx:.2f},{gy:.2f})")
        r["waypoint"] = {"x": gx, "y": gy}
        r["policy"] = "learned walker velocity_flat_existing.onnx; scripted heading/speed loop picks its command"
        r = self._after_motion(r, "go_to_waypoint")
        r["distance_to_waypoint_m"] = round(math.hypot(r["end_pose"]["x"] - gx, r["end_pose"]["y"] - gy), 3)
        return r

    def tool_plan_route_on_my_map(self, args):
        gx, gy = _num(args, "x", 0.0, -50, 50), _num(args, "y", 0.0, -50, 50)
        state = self.port.call(lambda w: w.base_state(self.index))
        asked = _num(args, "inflation_m", 0.14, 0.10, 0.35)
        tried = []
        for inflation in [asked] + [v for v in (0.18, 0.14, 0.12) if v < asked]:
            plan = self.map.plan((state["x"], state["y"]), (gx, gy), inflate_m=inflation)
            tried.append(inflation)
            if plan.get("ok"):
                break
        plan["inflations_tried_m"] = tried
        if plan.get("ok") and plan["inflation_m"] < asked:
            plan["note_inflation"] = (f"no path at {asked} m; planned with {plan['inflation_m']} m clearance "
                                      f"(your body half-width is ~0.08 m)")
        plan["pose"] = self._pose(state)
        if plan.get("ok") and len(plan["waypoints"]) > 6:
            plan["note"] = "more than 6 waypoints: pass the first 6 to follow_route, then re-plan"
        return plan

    def tool_follow_route(self, args):
        pts = args.get("waypoints") or []
        if not isinstance(pts, list) or not 1 <= len(pts) <= 6:
            return {"ok": False, "error": "waypoints must be a list of 1-6 [x, y] pairs"}
        try:
            pts = [(float(p[0]), float(p[1])) for p in pts]
        except (TypeError, ValueError, IndexError):
            return {"ok": False, "error": "each waypoint must be [x, y]"}
        legs = []
        for k, (x, y) in enumerate(pts):
            r = self.tool_go_to_waypoint({"x": x, "y": y, "max_s": args.get("max_s_per_leg", 15.0)})
            legs.append({"leg": k + 1, "waypoint": [x, y], "stop_reason": r.get("stop_reason"),
                         "end_pose": r.get("end_pose"), "traveled_m": r.get("traveled_m"),
                         "distance_to_waypoint_m": r.get("distance_to_waypoint_m"), "fallen": r.get("fallen")})
            last = r
            if r.get("stop_reason") != "goal_reached":
                break
        out = dict(last)
        out["legs"] = legs
        out["completed_legs"] = sum(1 for leg in legs if leg["stop_reason"] == "goal_reached")
        out["route_complete"] = out["completed_legs"] == len(pts)
        return out

    def tool_turn(self, args):
        ang = _num(args, "angle_deg", 90.0, -180.0, 180.0)
        state = self.port.call(lambda w: w.base_state(self.index))
        target = state["yaw_rad"] + math.radians(ang)

        def until(w, i, st, job):
            return "target_yaw_reached" if abs(angle_diff(st["yaw_rad"], target)) < math.radians(6) else None

        wz = 0.8 if ang > 0 else -0.8
        r = self.port.motion(self.index, "walk", command=(0.0, 0.0, wz), max_s=abs(math.radians(ang)) / 0.6 + 2.0,
                             until=until, label=f"turn {ang:+.0f} deg")
        r["requested_deg"] = ang
        r["policy"] = "learned walker velocity_flat_existing.onnx (turn command)"
        return self._after_motion(r, "turn")

    def tool_walk_with_onboard_perception(self, args):
        dur = _num(args, "duration_s", 4.0, 1.0, 10.0)
        r = self.port.motion(self.index, "onboard_hero", max_s=dur, label=f"onboard perception actor {dur:.0f}s")
        r["policy"] = "onboard_hero.pt (ToF 8x8 + head RGB 48x36 + proprio -> joints), fixed forward 0.1 m/s"
        return self._after_motion(r, "walk_with_onboard_perception")

    def tool_navigate_to(self, args):
        gx = _num(args, "x", 0.0, -20, 20)
        gy = _num(args, "y", 0.0, -20, 20)
        max_s = _num(args, "max_s", 20.0, 2.0, 30.0)

        def until(w, i, st, job):
            return "goal_reached" if math.hypot(st["x"] - gx, st["y"] - gy) < 0.25 else None

        r = self.port.motion(self.index, "depth_nav", goal=(gx, gy), max_s=max_s, until=until,
                             label=f"depth navigation to ({gx:.2f},{gy:.2f})")
        r["goal"] = {"x": gx, "y": gy}
        r["policy"] = "depth_navigation_v4.onnx (head depth 64x48 + goal -> walker command)"
        r = self._after_motion(r, "navigate_to")
        r["distance_to_goal_m"] = round(math.hypot(r["end_pose"]["x"] - gx, r["end_pose"]["y"] - gy), 3)
        return r

    def tool_get_up(self, args):
        def until(w, i, st, job):
            if st["tilt_deg"] < 15 and st["z"] > 0.1:
                job.upright_steps += 1
            else:
                job.upright_steps = 0
            return "upright" if job.upright_steps >= 25 else None

        r = self.port.motion(self.index, "getup", max_s=8.0, until=until, label="learned get-up")
        r["policy"] = "getup_kitfloor_ft_model_1100.onnx"
        return self._after_motion(r, "get_up")

    def tool_scan_surroundings(self, args):
        headings = []
        for k in range(4):
            state, reading, _ = self._sense()
            headings.append({"yaw_deg": round(math.degrees(state["yaw_rad"]), 0), "tof": reading.text()})
            r = self.tool_turn({"angle_deg": 90})
            if r.get("fallen"):
                return {"ok": False, "error": "fell while scanning", "headings": headings, "last": r}
        self.memory.add("observation", "scan: " + " | ".join(f"{h['yaw_deg']:.0f}deg: {h['tof'][:60]}" for h in headings))
        return {"ok": True, "headings": headings, "map": self.map.summary()}

    def tool_bearing_to(self, args):
        gx, gy = _num(args, "x", 0.0, -50, 50), _num(args, "y", 0.0, -50, 50)
        st = self.port.call(lambda w: w.base_state(self.index))
        d = math.hypot(gx - st["x"], gy - st["y"])
        b = math.degrees(angle_diff(math.atan2(gy - st["y"], gx - st["x"]), st["yaw_rad"]))
        return {"ok": True, "distance_m": round(d, 3), "relative_bearing_deg": round(b, 1),
                "your_pose": self._pose(st)}

    def tool_my_map(self, args):
        state = self.port.call(lambda w: w.base_state(self.index))
        self.map.add_pose(state["x"], state["y"], state["yaw_rad"])
        return {"ok": True, "pose": self._pose(state), "map": self.map.summary(), "ascii": self.map.ascii()}

    def tool_mark_landmark(self, args):
        label = str(args.get("label", "landmark"))[:40]
        dist = _num(args, "distance_ahead_m", 0.0, 0.0, 2.0)
        state = self.port.call(lambda w: w.base_state(self.index))
        x = state["x"] + dist * math.cos(state["yaw_rad"])
        y = state["y"] + dist * math.sin(state["yaw_rad"])
        self.map.add_landmark(label, x, y, state["sim_time_s"], str(args.get("note", ""))[:200])
        self.map_snapshots.append((state["sim_time_s"], self.map.snapshot()))
        self.memory.add("note", f"landmark {label} at ({x:.2f},{y:.2f})")
        return {"ok": True, "landmark": {"label": label, "x": round(x, 2), "y": round(y, 2)}}

    def tool_recall_memory(self, args):
        return {"ok": True, "memory": self.memory.digest(max_lines=25)}

    def tool_share_with_fleet(self, args):
        state = self.port.call(lambda w: w.base_state(self.index))
        self.fleet.post(self.name, args.get("message", ""), state["x"], state["y"], state["sim_time_s"])
        return {"ok": True, "posted": True}

    def tool_read_fleet_board(self, args):
        return {"ok": True, "messages": self.fleet.read(self.name)}

    def tool_task_done(self, args):
        self.done = {"success": bool(args.get("success")), "summary": str(args.get("summary", ""))}
        return {"ok": True, "task_complete": True}


__all__ = ["MicroduckAgentRuntime", "FleetBoard", "SYSTEM_PROMPT", "MOTION_TOOLS", "BASE_TOOLS", "FLEET_TOOLS"]
_ = (json, np)
