"""Observed route execution around the existing mobile controller/verifier.

Providers supply registered base poses AND conservative whole-volume clearance.
A planar grid or an RGB-D annotation alone cannot satisfy this interface.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import math
import threading
import time

from ..agent.base_effects import _validate_limits
from ..robotics.contracts import identifier
from ..sensing.models import digest, number, vector
from ..skills.mobile_runtime import MOTION_SKILLS, _spec
from .frames import TransformSample, sha256
from .grid import GridSnapshot, plan_route
from .robot_volume import RobotVolume, RobotVolumeSample


@dataclass(frozen=True)
class NavigationSample:
    """Independent map-from-base pose, not actuator feedback or optical pose.

    grid is a planar route proposal; volume_sha256 binds the separate collision
    volume queried by swept_clearance. Both must belong to this map revision.
    """
    robot_id: str
    model_identity_sha256: str | None
    geometry_sha256: str
    pose: TransformSample
    grid: GridSnapshot
    volume_sha256: str
    sensor_epoch: str
    sequence: int
    received_monotonic_s: float
    producer_age_s: float
    robot_volume: RobotVolumeSample | None = None

    def __post_init__(self):
        identifier(self.robot_id); identifier(self.sensor_epoch)
        digest(self.model_identity_sha256)
        if digest(self.geometry_sha256) is None or digest(self.volume_sha256) is None:
            raise ValueError("body geometry and collision volume hashes required")
        if not isinstance(self.pose, TransformSample) or not isinstance(self.grid, GridSnapshot):
            raise ValueError("typed independent pose and grid required")
        if self.robot_volume is not None and type(self.robot_volume) is not RobotVolumeSample:
            raise ValueError("typed articulated geometry capture required")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("capture sequence must be nonnegative integer")
        for key in ("received_monotonic_s", "producer_age_s"):
            object.__setattr__(self, key, number(getattr(self, key), key, minimum=0))


@dataclass(frozen=True)
class VolumeClearance:
    """Conservative minimum distance of the ENTIRE requested swept cylinder.

    None means any unknown/unobserved part. A provider must not substitute
    distances of a few points, free grid cells, or an admission boolean.
    """
    query_sha256: str
    volume_sha256: str
    distance_m: float | None

    def __post_init__(self):
        if digest(self.query_sha256) is None or digest(self.volume_sha256) is None:
            raise ValueError("clearance query and volume hashes required")
        if self.distance_m is not None:
            object.__setattr__(self, "distance_m", number(self.distance_m, "clearance"))


def navigation_settings(settings, profiles):
    required = {"base", "base_frame_id", "geometry_sha256", "calibration_sha256",
        "body_radius_m", "body_z_min_m", "body_z_max_m", "clearance_m", "stop_margin_m",
        "position_error_bound_m", "heading_error_bound_rad", "goal_tolerance_m",
        "max_route_m", "max_commands", "wall_timeout_s", "map_max_age_s"}
    if not isinstance(settings, dict) or set(settings)-{"robot_volume"} != required:
        raise ValueError("navigation requires explicit geometry, calibration and route bounds")
    result = dict(settings)
    identifier(result["base"]); identifier(result["base_frame_id"])
    for key in ("geometry_sha256", "calibration_sha256"):
        if digest(result[key]) is None:
            raise ValueError("navigation geometry/calibration SHA256 required")
    for key in required - {"base", "base_frame_id", "geometry_sha256", "calibration_sha256", "max_commands"}:
        result[key] = number(result[key], key)
        if key not in {"body_z_min_m", "body_z_max_m"} and result[key] <= 0:
            raise ValueError("navigation bounds must be positive")
    if (type(result["max_commands"]) is not int or not 1 <= result["max_commands"] <= 256
            or result["wall_timeout_s"] > 300 or result["max_route_m"] > 100
            or not result["body_z_min_m"] < result["body_z_max_m"]
            or result["heading_error_bound_rad"] >= math.pi / 2
            or result["goal_tolerance_m"] <= result["position_error_bound_m"]):
        raise ValueError("invalid navigation budget, volume or goal error bound")
    profile = next((p for p in profiles if p["name"] == result["base"]), None)
    if profile is None or not {"turn", "walk_distance"} <= set(profile["capabilities"]):
        raise ValueError("navigation requires an exact base with turn and measured walk_distance")
    if "robot_volume" in result:
        volume = RobotVolume.from_dict(result["robot_volume"])
        if (volume.sha256 != result["geometry_sha256"] or volume.robot_id != profile["robot_id"]
                or volume.model_identity_sha256 != profile.get("model_identity_sha256")
                or volume.base_frame_id != result["base_frame_id"]):
            raise ValueError("registered whole-robot geometry identity mismatch")
        radius, low, high = volume.cylinder()
        if (result["body_radius_m"] < radius or result["body_z_min_m"] > low
                or result["body_z_max_m"] < high):
            raise ValueError("navigation cylinder omits declared whole-robot reach")
        result["robot_volume"] = volume.as_dict()
    elif profile["type"] != "mock":
        raise ValueError("physical navigation requires registered whole-robot volume")
    limits = _validate_limits(profile["verifier"])
    distance = profile.get("distance_control")
    if not isinstance(distance, dict):
        raise ValueError("navigation requires profile-owned distance control")
    if result["stop_margin_m"] < limits["max_linear_speed_m_s"] * (
            limits["max_state_age_s"] + limits["read_timeout_s"] + limits["sample_interval_s"]) + limits["stop_drift_m"]:
        raise ValueError("stopping margin cannot cover the observation latency and stop drift")
    if (result["goal_tolerance_m"] <= distance["tolerance_m"] + result["position_error_bound_m"]
            or 2*result["position_error_bound_m"] >= limits["stop_drift_m"]
            or 2*result["heading_error_bound_rad"] >= limits["stop_drift_rad"]):
        raise ValueError("localization/control uncertainty cannot resolve arrival and rest")
    return result, profile, limits


def navigation_tools(specs):
    # Free velocity/distance commands are not exposed beside an active route.
    return [copy.deepcopy(s) for s in specs if s["name"] not in MOTION_SKILLS] + [
        _spec("go_to", "Follow a bounded route using independent localization and whole-volume collision queries; verify arrival and rest.",
              {"goal_xy_m": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
               "map_epoch": {"type": "string"},
               "map_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
              ("goal_xy_m", "map_epoch", "map_sha256"))]


def _yaw(pose):
    w, x, y, z = pose.rotation_wxyz
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _rotation_distance(a, b):
    return 2*math.acos(min(1., abs(sum(x*y for x, y in zip(a.rotation_wxyz, b.rotation_wxyz)))))


class NavigationDomain:
    """One route owns this locomotion domain until all segments have ended.

    The borrowed source implements read(deadline_monotonic_s=...) and
    swept_clearance(query, deadline_monotonic_s=...). Calls are serialized in
    one sampler, with an independent watchdog; a stuck source is quarantined.
    The existing MobileSkillRuntime remains the sole actuator/verifier owner.
    """
    motion_skills = frozenset({"go_to"})
    system_prompt = """You control the configured mobile base through bounded go_to routes.
Use exact map epochs and hashes from the registered navigation provider. Free
velocity and distance commands are unavailable. Unknown geometry or localization
fails closed. Only independently confirmed arrival and rest count as success.
Synthetic replay is software evidence. Respect stops; never reset automatically.
No arm, gripper or whole-body manipulation is supplied by this domain.
"""

    def __init__(self, mobile, source, settings):
        self.mobile, self.source = mobile, source
        self.settings, self.profile, self.limits = navigation_settings(settings, mobile.cfg.bases)
        self.robot_volume = (RobotVolume.from_dict(self.settings["robot_volume"])
                             if "robot_volume" in self.settings else None)
        if not callable(getattr(source, "read", None)) or not callable(getattr(source, "swept_clearance", None)):
            raise ValueError("independent localization and whole-volume clearance provider required")
        self.tool_specs = navigation_tools(mobile.tool_specs)
        self._condition = threading.Condition()
        self._active = self._closed = self._quarantined = self._resetting = self._latched = False
        self._serial = 0
        self._operation = None
        self._reader = self._watchdog = None
        self._unverified = None

    def __getattr__(self, name):
        return getattr(self.mobile, name)

    def _fail(self, op, reason):
        with self._condition:
            if op["cancel"].is_set():
                return
            op["error"] = str(reason)[:500]
            op["cancel"].set()
            self._condition.notify_all()
        # No observation lock is held while the existing priority stop fences IO.
        try:
            op["stop"] = self.mobile.stop(latch=True)
        except Exception as exc:
            op["stop"] = {"ok": False, "error": str(exc), "delivery_uncertain": True}
        self._latched = True

    def _check(self, op):
        if (op["cancel"].is_set() or self._serial != op["serial"] or self._closed
                or time.monotonic() >= op["deadline"]):
            raise ValueError(op["error"] or "navigation cancelled or deadline expired")

    def _current(self, op):
        with self._condition:
            now = time.monotonic()
            if (op["cancel"].is_set() or self._serial != op["serial"] or self._closed
                    or now >= op["deadline"]):
                return False
            return not op["samples"] or self._fresh(op["samples"][-1], now)

    def _fresh(self, sample, now):
        captures = (sample,) if self.robot_volume is None else (sample, sample.robot_volume)
        return all(capture is not None and capture.received_monotonic_s <= now
                   and 0 <= now-capture.received_monotonic_s+capture.producer_age_s
                   <= self.limits["max_state_age_s"] for capture in captures)

    def _require_fresh(self, sample):
        if not self._fresh(sample, time.monotonic()):
            raise ValueError("stale or future navigation/articulated geometry observation")

    @staticmethod
    def _age(sample):
        return time.monotonic() - sample.received_monotonic_s + sample.producer_age_s

    def _watch(self, op):
        while not op["cancel"].wait(min(.05, self.limits["sample_interval_s"])):
            now = time.monotonic()
            with self._condition:
                started = op["reading_since"]
            if not self._current(op) or (started is not None and now - started > self.limits["read_timeout_s"]):
                self._fail(op, "navigation/source wall deadline expired")
                return

    def _validate(self, op, sample):
        c, limits = self.settings, self.limits
        if not isinstance(sample, NavigationSample):
            raise ValueError("missing independent registered navigation sample")
        pose, grid = sample.pose, sample.grid
        if (sample.robot_id != self.profile["robot_id"]
                or sample.model_identity_sha256 != self.profile.get("model_identity_sha256")
                or sample.geometry_sha256 != c["geometry_sha256"]
                or pose.child != c["base_frame_id"] or pose.parent != grid.frame_id
                or pose.stamp.context != grid.stamp.context
                or pose.stamp.epoch != op["args"]["map_epoch"]
                or grid.sha256 != op["args"]["map_sha256"]
                or pose.stamp.calibration_id != c["calibration_sha256"]
                or grid.stamp.calibration_id != c["calibration_sha256"]):
            raise ValueError("navigation robot/model/map epoch/frame/calibration identity mismatch")
        if (pose.static or pose.position_error_m is None or pose.angular_error_rad is None
                or pose.position_error_m > c["position_error_bound_m"]
                or pose.angular_error_rad > c["heading_error_bound_rad"]):
            raise ValueError("localization error is unknown or exceeds navigation bound")
        _, x, y, _ = pose.rotation_wxyz
        if math.acos(max(-1., min(1., 1-2*(x*x+y*y)))) + pose.angular_error_rad > self.limits["max_tilt_rad"]:
            raise ValueError("localized base exceeds admitted upright body envelope")
        synthetic = self.profile["type"] == "mock"
        if pose.stamp.measurement_kind not in ({"synthetic"} if synthetic else {"estimated", "measured"}):
            raise ValueError("localization must be independently estimated/measured; replay is mock-only")
        if not synthetic and grid.stamp.measurement_kind == "synthetic":
            raise ValueError("synthetic map cannot authorize physical navigation")
        now = time.monotonic()
        if (sample.received_monotonic_s > now or
                not 0 <= self._age(sample) <= limits["max_state_age_s"]):
            raise ValueError("stale or future navigation observation")
        if self.robot_volume is not None:
            self.robot_volume.validate(sample.robot_volume, sample, now=now,
                                       max_age_s=limits["max_state_age_s"])
        if (pose.stamp.time_s < grid.stamp.time_s or
                not 0 <= pose.stamp.time_s - grid.oldest_capture_time_s <= c["map_max_age_s"]):
            raise ValueError("stale/future collision map")
        previous = op["samples"][-1] if op["samples"] else None
        if previous is not None:
            dt = pose.stamp.time_s - previous.pose.stamp.time_s
            if (pose.stamp.source_id != previous.pose.stamp.source_id
                    or pose.stamp.measurement_kind != previous.pose.stamp.measurement_kind
                    or sample.sensor_epoch != previous.sensor_epoch or sample.volume_sha256 != previous.volume_sha256
                    or sample.sequence <= previous.sequence or not 0 < dt <= limits["max_sample_gap_s"]
                    or sample.received_monotonic_s < previous.received_monotonic_s):
                raise ValueError("localization/map changed, replayed or has a capture gap")
            op["travel_m"] += math.dist(pose.translation_m, previous.pose.translation_m)
            if op["travel_m"] > c["max_route_m"]:
                raise ValueError("observed travel exceeds route budget")
            error = pose.position_error_m + previous.pose.position_error_m
            if math.dist(pose.translation_m, previous.pose.translation_m) > limits["max_linear_speed_m_s"]*dt + error:
                raise ValueError("localization jump or excessive translation")
            if _rotation_distance(pose, previous.pose) > limits["max_angular_speed_rad_s"]*dt + 2*c["heading_error_bound_rad"]:
                raise ValueError("localization jump or excessive rotation")
        self._require_fresh(sample)

    def _clearance(self, sample, target, deadline):
        c = self.settings
        query = {"robot_id": sample.robot_id, "model_identity_sha256": sample.model_identity_sha256,
            "geometry_sha256": c["geometry_sha256"], "calibration_sha256": c["calibration_sha256"],
            "map_sha256": sample.grid.sha256, "volume_sha256": sample.volume_sha256,
            "map_epoch": sample.pose.stamp.epoch, "frame_id": sample.grid.frame_id,
            "start_xy_m": tuple(sample.pose.translation_m[:2]), "end_xy_m": tuple(target),
            "radius_m": self._radius(),
            "z_min_m": sample.pose.translation_m[2] + min(0., c["body_z_min_m"]) - self._vertical_margin(),
            "z_max_m": sample.pose.translation_m[2] + max(0., c["body_z_max_m"]) + self._vertical_margin()}
        if self.robot_volume is not None:
            # Revalidate retained geometry at this final query boundary, not
            # just when the sampler first received it. The sweep always keeps
            # the complete permitted articulation envelope.
            geometry = self.robot_volume.validate(sample.robot_volume, sample, now=time.monotonic(),
                                                   max_age_s=self.limits["max_state_age_s"])
            query["geometry_sample_sha256"] = geometry["geometry_sample_sha256"]
        self._require_fresh(sample)
        if time.monotonic() >= deadline:
            raise ValueError("clearance query deadline expired")
        result = self.source.swept_clearance(query, deadline_monotonic_s=deadline)
        self._require_fresh(sample)
        if (not isinstance(result, VolumeClearance) or result.query_sha256 != sha256(query)
                or result.volume_sha256 != sample.volume_sha256 or result.distance_m is None
                or result.distance_m <= c["clearance_m"]):
            raise ValueError("swept robot volume collision/unknown space or unbound clearance evidence")

    def _vertical_margin(self):
        c = self.settings
        return c["body_radius_m"]*math.sin(self.limits["max_tilt_rad"]) + c["position_error_bound_m"] + c["stop_margin_m"]

    def _radius(self):
        c, d = self.settings, self.profile["distance_control"]
        angle = min(math.pi/2, c["heading_error_bound_rad"] + d["max_heading_drift_rad"]
                    + self.profile["resolved"]["safety"]["turn_tolerance_rad"])
        return (c["body_radius_m"] + max(abs(c["body_z_min_m"]), abs(c["body_z_max_m"]))*math.sin(self.limits["max_tilt_rad"])
                + c["position_error_bound_m"] + c["stop_margin_m"]
                + d["max_lateral_drift_m"] + d["max_distance_m"]*math.sin(angle))

    def _observe(self, op):
        try:
            while not op["cancel"].is_set():
                with self._condition:
                    op["reading_since"] = time.monotonic()
                    request = op["query"]
                deadline = min(op["deadline"], op["reading_since"] + self.limits["read_timeout_s"])
                sample = self.source.read(deadline_monotonic_s=deadline)
                self._validate(op, sample)
                self._clearance(sample, sample.pose.translation_m[:2], deadline)
                if request is not None:
                    self._clearance(sample, request["target"], deadline)
                if time.monotonic() >= deadline:
                    raise ValueError("navigation source returned after deadline")
                with self._condition:
                    self._check(op)
                    self._require_fresh(sample)
                    op["samples"].append(sample)
                    op["reading_since"] = None
                    if request is not None and op["query"] is request:
                        request["sample"] = sample
                    self._condition.notify_all()
                op["cancel"].wait(self.limits["sample_interval_s"])
        except BaseException as exc:
            self._fail(op, f"{type(exc).__name__}: {exc}")

    def _wait(self, op, predicate):
        with self._condition:
            while not predicate():
                self._check(op)
                self._condition.wait(min(.05, max(0., op["deadline"]-time.monotonic())))
            self._check(op)

    def _query(self, op, target):
        request = {"target": tuple(target), "sample": None}
        with self._condition:
            self._check(op)
            op["query"] = request
        self._wait(op, lambda: request["sample"] is not None)
        # Keep checking this swept corridor until the next segment replaces it.
        return request["sample"]

    def _command(self, op, name, args):
        with self._condition:
            self._check(op)
            if len(op["commands"]) >= self.settings["max_commands"]:
                raise ValueError("navigation command budget exhausted")
        result = self.mobile.execute(name, {"base": self.settings["base"], **args},
                                     admission_check=lambda: self._current(op))
        op["commands"].append(copy.deepcopy(result))
        self._check(op)
        if (not result.get("execution_ok") or result.get("delivery_uncertain") or result.get("error")
                or (not result.get("ok") and self.profile["type"] != "mock")):
            raise ValueError("mobile segment failed or remains physically unverified")

    def _route(self, op):
        self._wait(op, lambda: bool(op["samples"]))
        sample = op["samples"][-1]
        c = self.settings
        route = plan_route(sample.grid, sample.pose.translation_m[:2], op["goal"],
            footprint_radius_m=self._radius(), clearance_m=c["clearance_m"], position_error_m=0.,
            time_s=sample.pose.stamp.time_s, epoch=sample.pose.stamp.epoch, clock_id=sample.pose.stamp.clock_id,
            expected_map_sha256=op["args"]["map_sha256"], max_age_s=c["map_max_age_s"])
        if route["length_m"] > c["max_route_m"]:
            raise ValueError("route exceeds declared length budget")
        op["route"] = route
        safety = self.profile["resolved"]["safety"]
        for target in route["waypoints_xy_m"][1:]:
            while True:
                sample = self._query(op, target)
                xy, yaw = sample.pose.translation_m[:2], _yaw(sample.pose)
                distance = math.dist(xy, target)
                if distance + sample.pose.position_error_m <= c["goal_tolerance_m"]:
                    break
                angle = _angle(math.atan2(target[1]-xy[1], target[0]-xy[0])-yaw)
                if abs(angle) > safety["turn_tolerance_rad"]:
                    self._command(op, "turn", {"angle_rad": math.copysign(min(abs(angle), safety["max_turn_angle_rad"]), angle)})
                else:
                    self._command(op, "walk_distance", {"distance_m": min(distance, self.profile["distance_control"]["max_distance_m"])})
        boundary = time.monotonic()
        def settled():
            samples = [s for s in op["samples"] if s.received_monotonic_s - s.producer_age_s >= boundary]
            if len(samples) < self.limits["min_settle_samples"]:
                return False
            last = samples[-1]
            # Keep the first sample spanning the entire required window.
            while len(samples) > 2 and last.pose.stamp.time_s-samples[1].pose.stamp.time_s >= self.limits["settle_window_s"]:
                samples.pop(0)
            if (len(samples) < self.limits["min_settle_samples"]
                    or last.pose.stamp.time_s-samples[0].pose.stamp.time_s < self.limits["settle_window_s"]):
                return False
            for a, b in zip(samples, samples[1:]):
                dt = b.pose.stamp.time_s-a.pose.stamp.time_s
                if (math.dist(a.pose.translation_m, b.pose.translation_m) + a.pose.position_error_m + b.pose.position_error_m
                        > self.limits["stop_linear_speed_m_s"]*dt
                        or _rotation_distance(a.pose, b.pose) + a.pose.angular_error_rad + b.pose.angular_error_rad
                        > self.limits["stop_angular_speed_rad_s"]*dt):
                    return False
            for s in samples:
                if math.dist(s.pose.translation_m[:2], op["goal"]) + s.pose.position_error_m > c["goal_tolerance_m"]:
                    raise ValueError("final pose is outside arrival tolerance")
                if (math.dist(s.pose.translation_m, last.pose.translation_m) + s.pose.position_error_m + last.pose.position_error_m
                        > self.limits["stop_drift_m"]
                        or _rotation_distance(s.pose, last.pose) + s.pose.angular_error_rad + last.pose.angular_error_rad
                        > self.limits["stop_drift_rad"]):
                    return False
            return True
        self._wait(op, settled)

    def execute(self, name, args=None):
        started = time.monotonic()
        if not isinstance(name, str):
            return {"ok": False, "error": "navigation tool name must be a string"}
        if name in {"emergency_stop", "stop_navigation", "reset_stop"} and args not in (None, {}):
            return {"ok": False, "error": "stop/reset tools accept no arguments"}
        if name != "go_to":
            if name in MOTION_SKILLS:
                return {"ok": False, "error": "free motion is not exposed by the navigation domain"}
            if name in {"emergency_stop", "stop_navigation"}:
                return self.stop(latch=True)
            if name == "reset_stop":
                return self.reset_stop()
            if name == "task_memory" and isinstance(args, dict) and args.get("new_task") is True:
                return {"ok": False, "error": "use the explicit navigation begin_task boundary"}
            if name == "task_done" and isinstance(args, dict) and args.get("success") is True:
                missing = self.unverified_actions()
                if missing:
                    return {"ok": False, "success": False, "task_complete": True, "unverified": missing}
            return self.mobile.execute(name, args)
        if not isinstance(args, dict) or set(args) != {"goal_xy_m", "map_epoch", "map_sha256"}:
            return {"ok": False, "error": "exact goal, map epoch and map SHA256 required"}
        try:
            goal = vector(args["goal_xy_m"], 2, "goal")
            identifier(args["map_epoch"])
            if digest(args["map_sha256"]) is None:
                raise ValueError("map SHA256 required")
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        with self._condition:
            if self._active or self._closed or self._quarantined or self._resetting or self._latched:
                return {"ok": False, "error": "navigation active, closed or source quarantined"}
            self._active = True
            op = {"serial": self._serial, "args": dict(args), "goal": goal,
                "deadline": time.monotonic() + self.settings["wall_timeout_s"], "cancel": threading.Event(),
                "samples": deque(maxlen=self.limits["max_samples"]), "query": None,
                "error": None, "reading_since": None, "commands": [], "stop": None, "travel_m": 0.}
            self._operation = op
        try:
            self._reader = threading.Thread(target=self._observe, args=(op,), daemon=True, name="navigation-source")
            self._watchdog = threading.Thread(target=self._watch, args=(op,), daemon=True, name="navigation-watchdog")
            self._watchdog.start()
            self._reader.start()
            self._route(op)
            self._check(op)
            physical = self.profile["type"] != "mock" and bool(op["commands"])
            result = {"ok": physical, "execution_ok": True, "software_complete": True,
                "postcondition": {"status": "confirmed" if physical else "unverified",
                    "reason": "independent arrival/rest observed" if physical else "synthetic replay or no independently verified mobile segment"},
                "physical_admission": False}
        except BaseException as exc:
            self._fail(op, exc)
            result = {"ok": False, "execution_ok": False, "error": str(exc),
                      "postcondition": {"status": "unverified", "reason": str(exc)}, "physical_admission": False}
            if not isinstance(exc, Exception):
                raise
        finally:
            op["cancel"].set()
            for thread in (self._reader, self._watchdog):
                if thread is not None and thread.ident is not None:
                    thread.join(self.limits["read_timeout_s"])
            with self._condition:
                self._quarantined |= any(t is not None and t.is_alive() for t in (self._reader, self._watchdog))
                if self._serial != op["serial"] or op["error"]:
                    result = {"ok": False, "execution_ok": False, "error": op["error"] or "navigation superseded",
                              "postcondition": {"status": "unverified", "reason": op["error"] or "navigation superseded"},
                              "physical_admission": False}
                if not result["ok"]:
                    self._unverified = result["postcondition"]["reason"]
                self._active = False
        result.update(commands=op["commands"], stop=op["stop"], source_quarantined=self._quarantined,
                      route=op.get("route"), original_deadline_monotonic_s=op["deadline"],
                      observed_travel_m=op["travel_m"],
                      final_pose=op["samples"][-1].pose.as_dict() if op["samples"] else None)
        self.mobile.trace.record("go_to", args, result, (time.monotonic()-started)*1000,
                                 context={"robot_mode": "mobile", "navigation": True})
        self.mobile.memory.add("action", "go_to: " + result["postcondition"]["status"], data=copy.deepcopy(result))
        return result

    def stop(self, **kwargs):
        with self._condition:
            self._serial += 1
            self._latched = True
            op = self._operation if self._active else None
            if op is not None:
                op["error"] = "navigation cancelled by stop"
                op["cancel"].set()
                self._condition.notify_all()
        return self.mobile.stop(**kwargs)

    def reset_stop(self):
        with self._condition:
            if self._active or self._quarantined or self._closed or self._resetting:
                return {"ok": False, "error": "navigation owns IO or is closed"}
            self._resetting = True
            self._serial += 1
            serial = self._serial
        try:
            result = self.mobile.reset_stop()
            with self._condition:
                raced = self._serial != serial
                if result.get("ok") is True and not raced:
                    self._latched = False
            if raced:
                self.stop()
                return {"ok": False, "error": "stop superseded navigation reset"}
            return result
        finally:
            with self._condition:
                self._resetting = False

    def begin_task(self):
        with self._condition:
            if self._active or self._resetting or self._closed:
                raise ValueError("navigation owns the current task or is closed")
            self._serial += 1
            self.mobile.begin_task()
            self._unverified = None

    def unverified_actions(self, **kwargs):
        with self._condition:
            own = (["go_to: " + self._unverified] if self._unverified else [])
            if self._active or self._resetting:
                own.append("navigation action still in flight")
        return [*self.mobile.unverified_actions(**kwargs), *own]

    def close(self):
        with self._condition:
            self._closed = True
        self.stop()
        if self._active or any(t is not None and t.is_alive() for t in (self._reader, self._watchdog)):
            return {"ok": False, "complete": False, "error": "navigation/source still owns IO"}
        return self.mobile.close()
