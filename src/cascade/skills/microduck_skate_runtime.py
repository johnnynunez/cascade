# SPDX-License-Identifier: Apache-2.0
"""Agent-facing skill runtime for ONE MicroDuck on roller skates (agentic skating demos).

Same orchestrator contract as :class:`cascade.skills.microduck_agent_runtime.MicroduckAgentRuntime` (own
EpisodicMemory, own map of landmarks + trail, own trace, optional fleet board). The motion tools drive the unchanged
learned SKATING actor through its only command input ``twist = [throttle, 0, heading_error]``
(:mod:`cascade.sim.microduck_skate_world`). What is scripted, and therefore disclosed:

* the steering loop: ``skate_to`` / ``follow_route`` / ``skate_lane`` turn a goal into a world heading target
  (bearing to the point, or pure pursuit on a lane centre line like the delivered race pilot); the skate world turns it
  into the heading error each control step; the agent chooses the throttle;
* ``stop``: throttle -0.4 (the policy's own brake command) until the speed is below 0.05 m/s;
* the obstacle reflex: a brake when scene geometry or another skater is close ahead (``guard`` in the receipt).

Senses (STAND-INS, labelled in the videos): the skate robot in this world carries no ToF / RGB model (the skating
actors are blind), so the agent gets its simulator pose and speed (SLAM / odometry stand-in), the objects inside the
head camera's field of view from simulator ground truth (``detect_objects``, the same detector stand-in as the walking
demos; other skaters included), and a range-ahead reading from scene geometry. There is no get-up policy for the skate
robot: a fall ends that robot's run.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..memory.robot_map import summarize_frame
from ..sim.microduck_detector_standin import HEAD_CAMERA_HFOV_DEG, SOURCE as DETECTOR_SOURCE
from ..sim.microduck_world_port import PortClosed, WorldPort, angle_diff
from .microduck_agent_runtime import FLEET_TOOLS, HONEST_DETECT, FleetBoard, MicroduckAgentRuntime, _num, _spec

SKATE_HONEST_PROMPT = """HONEST ONBOARD MODE: you only know what this robot itself senses.
* Your pose is your OWN estimate in YOUR START FRAME ((0, 0) = where you started, +x your initial forward direction,
  +y left): heading from your gyro (good), position from visual odometry on your head camera with depth from your ToF
  (it drifts, expect 10-30 % of the distance skated; "LOST" = the camera odometry failed). There is no wheel odometry.
  Positions in your task are in this start frame and are approximate.
* "ahead" and the brake reflex come from your own ToF (8x8 zones, 45 deg cone, 4 m). It cannot see beside you.
* detect_objects runs a real image detector (YOLOE) on your camera image; names for simple props can be odd; judge by
  colour, size and position. Its range comes from your ToF or from the floor contact point.
* The image you receive is your own head camera.
* Navigate by what you SEE: a fresh detection's world_xy is computed from your current estimate, so it is consistent
  with where you think you are right now, while coordinates you were told (or remembered) drift away from it as you
  skate. Before each short leg, detect again and aim relative to the fresh detection (e.g. pass a cone 0.3 m to its
  side); keep legs short (1-2 m) near objects.
* Do not give up while you are upright and have steps left: certainty is never available on a real robot. If an
  obstacle blocks the way, roll slowly (throttle 0.25-0.3) with a heading 40-60 deg toward the clear side for 1-2 s,
  then detect again. Call task_done only when the course is done or you have fallen."""

SKATE_PROMPT = """You are the onboard agent of ONE MicroDuck, a 30 cm bipedal duck robot on ROLLER SKATES, in a physics
simulation. You decide what the robot does by calling exactly one tool per turn; a learned skating policy moves the
legs. You control it like a skater: a throttle (push harder = faster) and a direction. The policy itself is blind.
Measured skate behaviour of this policy on this floor (use it to plan):
* throttle 0.5 -> ~0.45 m/s after ~2 s of pushing; throttle 0.3 -> slower; above 0.7 it gets less stable.
* from a standstill the plaza policy needs throttle >= 0.25 to start pushing at all (0.15-0.2: it stands still); the
  race policy needs >= 0.4. Once rolling, lower throttles keep it rolling slowly.
* coasting (throttle 0) stops in ~1 s; stop() brakes to a halt in ~0.5-1 s.
* skaters cannot turn on the spot: heading changes need forward speed and take room (turn radius roughly 1 m at
  0.45 m/s, about 0.6 m when slow). Sharp corners make you overshoot or swing wide.
Your senses: your pose and speed (from the simulator, a stand-in for onboard localization), detect_objects (objects in
your head camera's 82 deg field of view, simulator ground truth, a detector stand-in) and a range-ahead reading.
World frame: metres, +x forward from the start, +y left, yaw in degrees counter-clockwise.
Every motion receipt contains your new pose, speed, the closest approach to its target and what is ahead, so you do not
need to call observe after each motion. Budget: you have a limited number of tool calls; be decisive.
Safety: keep 0.3 m from obstacles (your skates and arms need room); a fall ends your run (no get-up on skates).
Never claim success you did not measure: your pose and receipts are the evidence. Finish with task_done(success, summary).
"""

SKATE_TOOLS = [
    _spec("observe", "Your pose, speed, heading, posture, the range-ahead reading and the objects in view."),
    _spec("detect_objects", "Objects inside your head camera's field of view (82 deg, up to 3 m) with bearing, distance "
                            "and world position. DETECTOR STAND-IN: simulator ground truth limited to the field of view. "
                            "Detections are added to your own map as landmarks."),
    _spec("skate_to", "Skate to a world point (x, y): a scripted steering loop points the learned skating policy at it "
                      "while you choose the throttle (0.2-0.7). Ends when you pass within radius_m (default 0.3) of the "
                      "point, when you start moving away from it after the closest approach (overshoot), on an "
                      "obstacle brake, a fall or after max_s (<= 25). stop=true brakes to a halt at the end; otherwise "
                      "you keep rolling into your next command.",
          {"x": {"type": "number"}, "y": {"type": "number"}, "throttle": {"type": "number"},
           "radius_m": {"type": "number"}, "stop": {"type": "boolean"}, "max_s": {"type": "number"}}, ("x", "y")),
    _spec("follow_route", "Skate through 1-8 world waypoints in order without stopping between them (each leg like "
                          "skate_to with radius_m). Ends early at the first leg that brakes, falls or times out. "
                          "stop_at_end=true brakes to a halt after the last waypoint.",
          {"waypoints": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                         "description": "[[x, y], ...] metres, world frame"},
           "throttle": {"type": "number"}, "radius_m": {"type": "number"}, "stop_at_end": {"type": "boolean"}},
          ("waypoints",)),
    _spec("set_speed", "Push (throttle 0.2-0.7), coast (0) or brake (-0.5..-0.1) for duration_s (0.5-6) while holding a "
                       "world heading (heading_deg; default: keep the current one).",
          {"throttle": {"type": "number"}, "duration_s": {"type": "number"}, "heading_deg": {"type": "number"}},
          ("throttle", "duration_s")),
    _spec("stop", "Brake (the policy's brake command) until you are standing still (max 4 s)."),
    _spec("bearing_to", "Distance and relative bearing (deg, + = left) from you to a world point. Does not move.",
          {"x": {"type": "number"}, "y": {"type": "number"}}, ("x", "y")),
    _spec("my_map", "YOUR own memory map: landmarks you or your detector recorded and the trail you skated."),
    _spec("mark_landmark", "Remember a named world point in YOUR map.",
          {"label": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"}, "note": {"type": "string"}},
          ("label", "x", "y")),
    _spec("recall_memory", "Read YOUR episodic memory (actions, outcomes, notes) of this session.",
          {"query": {"type": "string"}}),
    _spec("task_done", "End the task honestly. success must reflect measured evidence.",
          {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ("success", "summary")),
]
LANE_TOOL = _spec(
    "skate_lane", "Race tool: skate along a lane of the oval for distance_m of track (0.5-12) at throttle (0.2-0.8). "
                  "Steering is the delivered race pilot (pure pursuit 0.25 m ahead on the lane centre line, scripted); "
                  "the learned race policy skates. Changing lane = calling it with another lane (it glides over). Ends "
                  "after the distance, on a skater/obstacle brake, a fall or after max_s (<= 30).",
    {"lane": {"type": "integer", "description": "1 (inner, shortest) .. 3 (outer)"}, "distance_m": {"type": "number"},
     "throttle": {"type": "number"}, "max_s": {"type": "number"}}, ("lane", "distance_m"))
RACE_STATUS_TOOL = _spec("race_status", "Your own race progress (track metres, laps, lane, speed) and race time. "
                                        "Other skaters only through the fleet board and detect_objects.")
SKATE_MOTION_TOOLS = frozenset({"skate_to", "follow_route", "set_speed", "stop", "skate_lane"})
SKATER_RADIUS_M = 0.12


class Oval:
    """Stadium track: straights x in [-half, half] at y = +-r, semicircles around (+-half, 0); lanes = centre radii."""

    def __init__(self, half: float, lanes: list[float], finish_x: float = 0.0):
        self.half, self.lanes, self.finish_x = float(half), [float(r) for r in lanes], float(finish_x)

    def fraction(self, x, y):
        """Normalized track coordinate in [0, 4): 0-1 south straight (+x), 1-2 east bend, 2-3 north straight (-x),
        3-4 west bend. Counter-clockwise racing increases it."""
        h = self.half
        if -h <= x <= h:
            return (x + h) / (2 * h) if y < 0 else 2.0 + (h - x) / (2 * h)
        if x > h:
            th = math.atan2(y, x - h)  # -pi/2 .. pi/2
            return 1.0 + (min(max(th, -math.pi / 2), math.pi / 2) + math.pi / 2) / math.pi
        th = math.atan2(y, x + h) % (2 * math.pi)  # pi/2 .. 3pi/2
        return 3.0 + (min(max(th, math.pi / 2), 3 * math.pi / 2) - math.pi / 2) / math.pi

    def length(self, lane: int) -> float:
        return 4 * self.half + 2 * math.pi * self.lanes[lane - 1]

    def point(self, lane: int, f: float):
        """Point on lane centre line at normalized coordinate f."""
        r, h = self.lanes[lane - 1], self.half
        f = f % 4.0
        if f < 1:
            return (-h + 2 * h * f, -r)
        if f < 2:
            a = -math.pi / 2 + math.pi * (f - 1)
            return (h + r * math.cos(a), r * math.sin(a))
        if f < 3:
            return (h - 2 * h * (f - 2), r)
        a = math.pi / 2 + math.pi * (f - 3)
        return (-h + r * math.cos(a), r * math.sin(a))

    def metres_per_unit(self, lane: int, f: float) -> float:
        f = f % 4.0
        return 2 * self.half if (f < 1 or 2 <= f < 3) else math.pi * self.lanes[lane - 1]

    def lane_of(self, x, y) -> tuple[int, float]:
        h = self.half
        cx = min(max(x, -h), h)
        rad = math.hypot(x - cx, y)
        k = min(range(len(self.lanes)), key=lambda i: abs(self.lanes[i] - rad))
        return k + 1, rad - self.lanes[k]


class Geometry:
    """Range-ahead stand-in and obstacle reflex from scene geometry (oriented boxes of the course colliders)."""

    def __init__(self, boxes: dict):
        self.boxes = boxes
        self._inv = {k: np.linalg.inv(b[2]) for k, b in boxes.items()}

    def _dist(self, name, p):
        lo, hi, M = self.boxes[name]
        Minv = self._inv[name]
        q = Minv[:3, :3] @ p + Minv[:3, 3]
        n = np.clip(q, lo, hi)
        w = M[:3, :3] @ n + M[:3, 3]
        return float(np.linalg.norm((p - w)[:2])), w

    def ahead(self, x, y, yaw, others=(), cone_deg=30.0, z=0.08):
        """Nearest obstacle surface or other skater within +-cone_deg of the heading: (name, distance, bearing)."""
        best = (None, 99.0, 0.0)
        p = np.array([x, y, z])
        for name in self.boxes:
            d, w = self._dist(name, p)
            b = math.degrees(angle_diff(math.atan2(w[1] - y, w[0] - x), yaw)) if d > 1e-6 else 0.0
            if abs(b) <= cone_deg and d < best[1]:
                best = (name, d, b)
        for name, ox, oy in others:
            d = math.hypot(ox - x, oy - y) - SKATER_RADIUS_M
            b = math.degrees(angle_diff(math.atan2(oy - y, ox - x), yaw))
            if abs(b) <= cone_deg and d < best[1]:
                best = (name, d, b)
        return best


class MicroduckSkateRuntime(MicroduckAgentRuntime):
    robot_mode = "mobile"

    def __init__(self, name: str, index: int, port: WorldPort, run_dir: Path, *, bounds_xy, geometry: Geometry,
                 fleet: FleetBoard | None = None, detector=None, oval: Oval | None = None, extra_prompt: str = "",
                 robot_names=(), honest: bool = False):
        super().__init__(name, index, port, run_dir, bounds_xy=bounds_xy, floor_z=0.0, fleet=fleet, has_nav=False,
                         has_getup=False, extra_prompt="", detector=detector)
        self.geometry, self.oval = geometry, oval
        self.robot_names = list(robot_names)
        tools = [t for t in SKATE_TOOLS if detector is not None or t["name"] != "detect_objects"]
        if oval is not None:
            tools = tools[:2] + [LANE_TOOL, RACE_STATUS_TOOL] + tools[2:]
        if fleet is not None:
            tools = tools[:-1] + FLEET_TOOLS + tools[-1:]
        self.tool_specs = tools
        self.system_prompt = SKATE_PROMPT + f"\nYou are robot '{name}'." + (("\n" + extra_prompt) if extra_prompt else "")
        self.honest = bool(honest)
        if self.honest:
            self.tool_specs = [HONEST_DETECT if t["name"] == "detect_objects" else t for t in self.tool_specs
                               if t["name"] not in ("skate_lane", "race_status")]
            self.system_prompt = (SKATE_PROMPT.replace(
                "Your senses: your pose and speed (from the simulator, a stand-in for onboard localization), detect_objects (objects in\n"
                "your head camera's 82 deg field of view, simulator ground truth, a detector stand-in) and a range-ahead reading.",
                "Your senses: see HONEST ONBOARD MODE below.") + SKATE_HONEST_PROMPT + f"\nYou are robot '{name}'."
                + (("\n" + extra_prompt) if extra_prompt else ""))
        self.race = {"progress_units": 0.0, "last_f": None, "lane": None, "finish_time_s": None, "start_time_s": None}

    # -------------------------------------------------------------- senses
    def _others(self, w):
        if self.honest:
            return []  # other robots are only seen through the camera / ToF
        out = []
        for j, n in enumerate(self.robot_names):
            if j != self.index:
                s = w.base_state(j)
                out.append((n, s["x"], s["y"]))
        return out

    def _sense(self, integrate=True):
        i = self.index
        if self.honest:
            def read_h(w):
                return w.base_state(i), w.tof_frame(i), w.camera_rgb(i)

            state, tof, cam = self.port.call(read_h)
            if integrate:
                self.map.integrate(tof["distance_mm"], tof["status"], tof["sensor_pos_w"], tof["beam_dir_w"], floor_z=0.0)
                self.map.add_pose(state["x"], state["y"], state["yaw_rad"])
                self.map_snapshots.append((state["sim_time_s"], self.map.snapshot()))
            r = summarize_frame(tof["distance_mm"], tof["status"], tof["sensor_pos_w"], tof["beam_dir_w"],
                                state["yaw_rad"], floor_z=0.0)
            self.last_frame = cam
            return state, r.text() + " [own ToF]", []

        def read(w):
            return w.base_state(i), self._others(w)

        state, others = self.port.call(read)
        if integrate:
            self.map.add_pose(state["x"], state["y"], state["yaw_rad"])
            self.map_snapshots.append((state["sim_time_s"], self.map.snapshot()))
        name, d, b = self.geometry.ahead(state["x"], state["y"], state["yaw_rad"], others)
        reading = ("nothing within 3 m ahead (+-30 deg)" if d > 3.0 else
                   f"{name} {d:.2f} m ahead at {b:+.0f} deg") + " [scene-geometry range stand-in]"
        return state, reading, others

    def _pose(self, state):
        out = {"x": round(state["x"], 2), "y": round(state["y"], 2), "yaw_deg": round(math.degrees(state["yaw_rad"]), 1),
               "speed_m_s": round(state["speed"], 2), "posture": "FALLEN" if state["fallen"] else "upright",
               "tilt_deg": round(state["tilt_deg"], 1)}
        if "localization" in state:
            out["localization"] = state["localization"]
            out["pose_source"] = "your own estimate (start frame)"
        return out

    def _detections(self, state, others=()):
        if self.detector is None:
            return None
        if self.honest:
            d = self.detector.detect()
            for det in d["detections"]:
                if det.get("world_xy") is not None:
                    self.map.add_landmark(det["label"], det["world_xy"][0], det["world_xy"][1], state["sim_time_s"],
                                          note="onboard detector")
            return d
        d = self.detector.detect(state["x"], state["y"], state["yaw_rad"])
        for n, ox, oy in others:  # other skaters: same field-of-view rule, live positions
            dist = math.hypot(ox - state["x"], oy - state["y"])
            b = math.degrees(angle_diff(math.atan2(oy - state["y"], ox - state["x"]), state["yaw_rad"]))
            if dist <= self.detector.max_range_m and abs(b) <= HEAD_CAMERA_HFOV_DEG / 2:
                d["detections"].append({"label": f"skater {n}", "bearing_deg": round(b, 1), "distance_m": round(dist, 2),
                                        "world_xy": [round(ox / 0.05) * 0.05, round(oy / 0.05) * 0.05]})
        d["detections"].sort(key=lambda x: x["distance_m"])
        for det in d["detections"]:
            if not det["label"].startswith("skater "):
                self.map.add_landmark(det["label"], det["world_xy"][0], det["world_xy"][1], state["sim_time_s"],
                                      note="detector stand-in")
        return d

    def _after_motion(self, receipt, name):
        state, reading, others = self._sense()
        receipt["after"] = {"pose": self._pose(state), "ahead": reading}
        if self.detector is not None:
            receipt["after"]["objects_in_view"] = [
                f"{d['label']} {d['distance_m']} m at {d['bearing_deg']} deg -> world {d['world_xy']}"
                for d in self._detections(state, others)["detections"]] or "none (detector stand-in)"
        if self.oval is not None:
            receipt["after"]["race"] = self._race_status(state)
        verdict = "confirmed" if receipt.get("ok") and not state["fallen"] else "unverified"
        self.memory.add("action", f"{name}: {receipt.get('stop_reason')} -> at ({state['x']:.2f},{state['y']:.2f}) "
                                  f"{state['speed']:.2f} m/s, {'FALLEN' if state['fallen'] else 'upright'}",
                        data={"verdict": verdict})
        return receipt

    # -------------------------------------------------------------- race bookkeeping (own progress only)
    def track_update(self, st):
        """Called on the physics owner every step (see :meth:`install_progress`)."""
        if self.oval is None:
            return
        f = self.oval.fraction(st["x"], st["y"])
        last = self.race["last_f"]
        if last is not None:
            df = (f - last + 2.0) % 4.0 - 2.0
            self.race["progress_units"] += df
        self.race["last_f"] = f
        if self.race["finish_time_s"] is None and self.race["progress_units"] >= 4.0:
            self.race["finish_time_s"] = st["sim_time_s"]

    def _race_status(self, st):
        lane, off = self.oval.lane_of(st["x"], st["y"])
        u = self.race["progress_units"]
        t0 = self.race["start_time_s"] or 0.0
        return {"laps_done": round(u / 4.0, 3), "track_m_done_on_lane1": round(u / 4.0 * self.oval.length(1), 2),
                "lap_length_lane1_m": round(self.oval.length(1), 2), "current_lane": lane,
                "offset_from_lane_centre_m": round(off, 2), "speed_m_s": round(st["speed"], 2),
                "race_time_s": round(st["sim_time_s"] - t0, 2),
                "finished_at_race_time_s": (round(self.race["finish_time_s"] - t0, 2)
                                            if self.race["finish_time_s"] is not None else None)}

    # -------------------------------------------------------------- motion helpers
    def _guard(self, w, st, extra_m=0.0):
        limit = 0.18 + 0.45 * max(st["speed"], 0.0) + extra_m
        if self.honest:
            if (w.step_index % 5) != 0:
                return None
            f = w.tof_frame(self.index)
            r = summarize_frame(f["distance_mm"], f["status"], f["sensor_pos_w"], f["beam_dir_w"], st["yaw_rad"],
                                floor_z=0.0)
            if r.nearest_obstacle_m is not None and abs(r.nearest_obstacle_bearing_deg) < 22 and r.nearest_obstacle_m < limit:
                return (f"tof_brake: obstacle {r.nearest_obstacle_m:.2f} m ahead at {r.nearest_obstacle_bearing_deg:+.0f} deg "
                        f"(limit {limit:.2f} m at {st['speed']:.2f} m/s)")
            return None
        others = self._others(w)
        name, d, b = self.geometry.ahead(st["x"], st["y"], st["yaw_rad"], others)
        limit = 0.18 + 0.45 * max(st["speed"], 0.0) + extra_m
        if d < limit:
            return f"obstacle_brake: {name} {d:.2f} m ahead at {b:+.0f} deg (limit {limit:.2f} m at {st['speed']:.2f} m/s)"
        return None

    def _motion(self, mode, throttle, heading, max_s, until, label):
        r = self.port.motion(self.index, mode, command=(throttle, 0.0, heading), max_s=max_s, until=until, label=label)
        return r

    def _brake_to_halt(self, max_s=4.0):
        def until(w, idx, st, job):
            return "halted" if st["speed"] < 0.05 and w.step_index - job.start_step > 10 else None

        st = self.port.call(lambda w: w.base_state(self.index))
        r = self._motion("brake", -0.4, st["target_yaw_rad"], max_s, until, "stop (brake -0.4)")
        r["ok"] = r["stop_reason"] == "halted"
        return r

    def _leg(self, gx, gy, throttle, radius, max_s, label):
        closest = {"d": 1e9, "t": None}

        def until(w, idx, st, job):
            d = math.hypot(gx - st["x"], gy - st["y"])
            if d < closest["d"]:
                closest["d"], closest["t"] = d, st["sim_time_s"]
            if d <= radius:
                return "goal_reached"
            if closest["d"] < radius + 0.6 and d > closest["d"] + 0.12:
                return "passed_closest_approach"
            reason = self._guard(w, st)
            if reason:
                return reason
            w.set_control(idx, "skate", (throttle, 0.0, math.atan2(gy - st["y"], gx - st["x"])))
            return None

        st0 = self.port.call(lambda w: w.base_state(self.index))
        r = self._motion("skate", throttle, math.atan2(gy - st0["y"], gx - st0["x"]), max_s, until, label)
        r["ok"] = r["stop_reason"] == "goal_reached"
        r["closest_approach_m"] = round(closest["d"], 3)
        r["closest_approach_sim_time_s"] = closest["t"]
        return r

    # -------------------------------------------------------------- tools
    def tool_observe(self, args):
        state, reading, others = self._sense()
        res = {"ok": True, "pose": self._pose(state), "skate_mode": state["mode"], "throttle": state["throttle"],
               "sim_time_s": round(state["sim_time_s"], 2), "ahead": reading}
        if self.detector is not None:
            res["objects_in_view"] = self._detections(state, others)["detections"]
        if self.oval is not None:
            res["race"] = self._race_status(state)
        self.memory.add("observation", f"observe: {reading}")
        return res

    def tool_detect_objects(self, args):
        state, _, others = self._sense()
        d = self._detections(state, others)
        if self.honest:
            self.memory.add("observation", "detect_objects: " + (", ".join(
                f"{x['label']} at {x['distance_m']} m, bearing {x['bearing_deg']} deg" for x in d["detections"]) or "nothing"))
            return {"ok": True, "pose": self._pose(state), **d}
        self.memory.add("observation", "detect_objects: " + (", ".join(
            f"{x['label']} at {x['distance_m']} m, bearing {x['bearing_deg']} deg" for x in d["detections"]) or "nothing"))
        return {"ok": True, "pose": self._pose(state), **d, "source": DETECTOR_SOURCE}

    def tool_skate_to(self, args):
        gx, gy = _num(args, "x", 0.0, -50, 50), _num(args, "y", 0.0, -50, 50)
        thr = _num(args, "throttle", 0.5, 0.2, 0.7)
        radius = _num(args, "radius_m", 0.3, 0.1, 1.0)
        max_s = _num(args, "max_s", 15.0, 1.0, 25.0)
        r = self._leg(gx, gy, thr, radius, max_s, f"skate_to ({gx:.2f},{gy:.2f}) throttle {thr:.2f}")
        if bool(args.get("stop")) and not r.get("fallen"):
            b = self._brake_to_halt()
            r["brake"] = {k: b[k] for k in ("stop_reason", "traveled_m", "sim_time_end_s")}
        r["target"] = {"x": gx, "y": gy, "radius_m": radius}
        r["policy"] = "learned skating policy; scripted steering loop sets its heading command; throttle from you"
        r = self._after_motion(r, "skate_to")
        r["distance_to_target_now_m"] = round(math.hypot(r["after"]["pose"]["x"] - gx, r["after"]["pose"]["y"] - gy), 3)
        return r

    def tool_follow_route(self, args):
        pts = []
        for p in list(args.get("waypoints") or [])[:8]:
            try:
                pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        if not pts:
            return {"ok": False, "error": "waypoints must be [[x, y], ...]"}
        thr = _num(args, "throttle", 0.5, 0.2, 0.7)
        radius = _num(args, "radius_m", 0.3, 0.1, 1.0)
        legs = []
        for k, (gx, gy) in enumerate(pts):
            r = self._leg(gx, gy, thr, radius, 20.0, f"route leg {k + 1}/{len(pts)} ({gx:.2f},{gy:.2f})")
            legs.append({"waypoint": [gx, gy], "stop_reason": r["stop_reason"], "closest_approach_m": r["closest_approach_m"],
                         "end_pose": r["end_pose"]})
            if r["stop_reason"] not in ("goal_reached", "passed_closest_approach") or r.get("fallen"):
                break
        res = {"ok": len(legs) == len(pts) and legs[-1]["stop_reason"] in ("goal_reached", "passed_closest_approach"),
               "legs": legs, "throttle": thr}
        if bool(args.get("stop_at_end")) and not r.get("fallen"):
            b = self._brake_to_halt()
            res["brake"] = {k: b[k] for k in ("stop_reason", "traveled_m", "sim_time_end_s")}
        res["policy"] = "learned skating policy; scripted steering loop per leg; throttle from you"
        return self._after_motion(res, "follow_route")

    def tool_set_speed(self, args):
        thr = _num(args, "throttle", 0.0, -0.5, 0.7)
        dur = _num(args, "duration_s", 2.0, 0.5, 6.0)
        st = self.port.call(lambda w: w.base_state(self.index))
        head = math.radians(_num(args, "heading_deg", math.degrees(st["target_yaw_rad"]), -360, 360))
        mode = "skate" if thr > 0.01 else ("brake" if thr < -0.01 else "coast")

        def until(w, idx, s, job):
            return self._guard(w, s) if thr > 0.01 else None

        r = self._motion(mode, thr, head, dur, until, f"set_speed throttle {thr:+.2f} heading {math.degrees(head):.0f} deg")
        if r["stop_reason"] == "timeout":  # the port's name for "max_s elapsed while an until-callback was set"
            r["stop_reason"] = "duration_elapsed"
        r["ok"] = r["stop_reason"] == "duration_elapsed"
        return self._after_motion(r, "set_speed")

    def tool_stop(self, args):
        return self._after_motion(self._brake_to_halt(), "stop")

    def tool_skate_lane(self, args):
        if self.oval is None:
            return {"ok": False, "error": "no oval in this scenario"}
        lane = int(_num(args, "lane", 1, 1, len(self.oval.lanes)))
        dist = _num(args, "distance_m", 3.0, 0.5, 12.0)
        thr = _num(args, "throttle", 0.6, 0.2, 0.8)
        max_s = _num(args, "max_s", 25.0, 2.0, 30.0)
        start_units = self.race["progress_units"]
        if self.race["start_time_s"] is None:
            self.race["start_time_s"] = self.port.call(lambda w: w.time_s)
        target_units = start_units + dist / self.oval.length(1) * 4.0
        lookahead = 0.25

        def until(w, idx, st, job):
            if self.race["progress_units"] >= target_units:
                return "distance_done"
            reason = self._guard(w, st, extra_m=-0.05)
            if reason:
                return reason
            f = self.oval.fraction(st["x"], st["y"])
            gx, gy = self.oval.point(lane, f + lookahead / self.oval.metres_per_unit(lane, f))
            w.set_control(idx, "skate", (thr, 0.0, math.atan2(gy - st["y"], gx - st["x"])))
            return None

        r = self._motion("skate", thr, None, max_s, until, f"skate_lane {lane} {dist:.1f} m throttle {thr:.2f}")
        r["ok"] = r["stop_reason"] == "distance_done"
        r["lane"] = lane
        r["policy"] = "learned race skating policy; delivered race pilot (pure pursuit, scripted) steers; throttle from you"
        return self._after_motion(r, "skate_lane")

    def tool_race_status(self, args):
        st = self.port.call(lambda w: w.base_state(self.index))
        return {"ok": True, "race": self._race_status(st), "pose": self._pose(st)}

    def tool_my_map(self, args):
        state = self.port.call(lambda w: w.base_state(self.index))
        self.map.add_pose(state["x"], state["y"], state["yaw_rad"])
        s = self.map.summary()
        return {"ok": True, "pose": self._pose(state), "landmarks": s.get("landmarks"),
                "trail_points": len(self.map.visited) if hasattr(self.map, "visited") else None}

    def tool_mark_landmark(self, args):
        label = str(args.get("label", "landmark"))[:40]
        x, y = _num(args, "x", 0.0, -50, 50), _num(args, "y", 0.0, -50, 50)
        t = self.port.call(lambda w: w.time_s)
        self.map.add_landmark(label, x, y, t, str(args.get("note", ""))[:200])
        self.map_snapshots.append((t, self.map.snapshot()))
        self.memory.add("note", f"landmark {label} at ({x:.2f},{y:.2f})")
        return {"ok": True, "landmark": {"label": label, "x": round(x, 2), "y": round(y, 2)}}


__all__ = ["MicroduckSkateRuntime", "Oval", "Geometry", "SKATE_PROMPT", "SKATE_TOOLS", "SKATE_MOTION_TOOLS", "PortClosed"]
