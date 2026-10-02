"""cuMotion 1.1.0 joint-space planning against a caller-owned static world.

No arm, bridge, simulator or live occupancy connection is opened here. Native
success and the checks below produce a candidate, never execution permission.
SDK objects stay under one lock and retain their world/model owners.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module, metadata
import json
import math
from pathlib import Path
from threading import RLock

import numpy as np

from . import PlanningError

SDK_VERSION = "1.1.0"
LINEAR_PATH_TOLERANCE_M = .001
ORIENTATION_PATH_TOLERANCE_RAD = .025
# SDK path constraints are soft penalties. These documented weights enforce
# the small reBot contact corridor; the independent FK check remains decisive.
PATH_POSITION_WEIGHTS = {
    "trajopt/pbo/cost/path_position_error_penalty/weight": 600000.,
    "trajopt/lbfgs/cost/path_position_error_penalty/weight": 50000000.,
}


def _sdk():
    try:
        version = metadata.version("cumotion")
        if version != SDK_VERSION:
            raise PlanningError(f"cuMotion {SDK_VERSION} required; installed {version}")
        return import_module("cumotion")
    except (ImportError, OSError, metadata.PackageNotFoundError) as exc:
        raise PlanningError(
            "cuMotion SDK unavailable: install the NVIDIA cuMotion 1.1.0 wheel "
            "matching this Python, platform and CUDA runtime; see docs/CUMOTION.md"
        ) from exc


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PlanningError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise PlanningError(f"{name} must be positive")
    return float(value)


def _vector(value, size, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PlanningError(f"{name} must contain {size} finite numbers") from exc
    # Accept Eigen's native column vectors as well as the caller's flat vector.
    if array.shape not in ((size,), (size, 1)) or not np.isfinite(array).all():
        raise PlanningError(f"{name} must contain {size} finite numbers")
    return array.reshape(size).copy()


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise PlanningError(f"{name} must be a nonempty string")
    return value


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MotionPlan:
    """Copied samples in caller joint order/sign convention, SI units."""

    joint_names: tuple[str, ...]
    times_s: tuple[float, ...]
    positions: tuple[tuple[float, ...], ...]
    velocities: tuple[tuple[float, ...], ...]
    base_frame: str
    tool_frame: str
    model_sha256: str
    scene_sha256: str
    request_sha256: str
    sdk_version: str = SDK_VERSION
    path_constraint: str = "unconstrained"

    def as_dict(self):
        return {"status": "candidate", "execution_authorized": False,
                "backend": "cumotion", **asdict(self)}


class CumotionPlanner:
    """One immutable robot/world binding. Construct another for a new scene.

    ``joint_signs[i]`` maps local joint i to the URDF convention. Joint order is
    resolved by name, never inferred from URDF declaration or XRDF ordering.
    Only static cuboids are supported; absent geometry cannot become an empty
    world implicitly (``obstacles`` is required, even if explicitly empty).
    """

    def __init__(self, config):
        cfg = config.as_dict() if hasattr(config, "as_dict") else dict(config)
        allowed = {"type", "urdf", "xrdf", "joint_names", "joint_signs",
                   "base_frame", "tool_frame", "obstacles", "sample_dt_s",
                   "max_duration_s", "max_samples", "joint_margin",
                   "endpoint_tolerance"}
        if set(cfg) - allowed:
            raise PlanningError(f"unknown cuMotion options: {sorted(set(cfg) - allowed)}")
        if cfg.get("type") != "cumotion":
            raise PlanningError("cuMotion configuration requires type: cumotion")
        names = cfg.get("joint_names")
        if not isinstance(names, (list, tuple)) or not names:
            raise PlanningError("joint_names must explicitly list the controlled joints")
        self.joint_names = tuple(_name(n, "joint name") for n in names)
        self.n = len(names)
        if len(set(names)) != self.n:
            raise PlanningError("joint_names must be unique")
        signs = cfg.get("joint_signs")
        if (not isinstance(signs, (list, tuple)) or len(signs) != self.n
                or any(type(s) not in (int, float) or s not in (-1, 1) for s in signs)):
            raise PlanningError("joint_signs must explicitly contain one +1/-1 per joint")
        self._signs = np.array(signs, dtype=float)
        self.base_frame = _name(cfg.get("base_frame"), "base_frame")
        self.tool_frame = _name(cfg.get("tool_frame"), "tool_frame")
        self._dt = _number(cfg.get("sample_dt_s", 0.02), "sample_dt_s", positive=True)
        self._max_duration = _number(cfg.get("max_duration_s", 60.0), "max_duration_s", positive=True)
        self._max_samples = cfg.get("max_samples", 10000)
        if type(self._max_samples) is not int or not 2 <= self._max_samples <= 100000:
            raise PlanningError("max_samples must be an integer in [2, 100000]")
        self._margin = _number(cfg.get("joint_margin", 0.025), "joint_margin")
        if self._margin < 0:
            raise PlanningError("joint_margin must be nonnegative")
        self._endpoint_tol = _number(cfg.get("endpoint_tolerance", 1e-4), "endpoint_tolerance", positive=True)
        boxes = self._boxes(cfg.get("obstacles"))
        self.scene_sha256 = _digest({"base_frame": self.base_frame, "obstacles": boxes})
        # Read once, hash and pass these very bytes to the SDK. No source-file
        # reread can silently change the model behind the receipt.
        try:
            urdf = Path(cfg["urdf"]).read_text()
            xrdf = Path(cfg["xrdf"]).read_text()
        except (KeyError, TypeError, OSError, UnicodeError) as exc:
            raise PlanningError("cuMotion requires readable UTF-8 urdf and xrdf files") from exc
        self.model_sha256 = _digest({"urdf": urdf, "xrdf": xrdf})
        self._lock = RLock()
        self._closed = False
        self._poisoned = False
        self._cm = _sdk()
        self._robot = self._world = self._view = self._inspector = self._optimizer = None
        self._contact_optimizer = None
        self._obstacles = []
        try:
            self._robot = self._cm.load_robot_from_memory(xrdf, urdf)
            kin = self._robot.kinematics()
            sdk_names = tuple(self._robot.cspace_coord_name(i)
                              for i in range(self._robot.num_cspace_coords()))
            if len(sdk_names) != self.n or set(sdk_names) != set(self.joint_names):
                raise PlanningError("XRDF c-space joints do not match joint_names exactly")
            if kin.base_frame_name() != self.base_frame:
                raise PlanningError("cuMotion model base_frame mismatch")
            if self.tool_frame not in self._robot.tool_frame_names():
                raise PlanningError("tool_frame is not declared in the XRDF")
            self._sdk_from_local = np.array([self.joint_names.index(n) for n in sdk_names])
            self._local_from_sdk = np.argsort(self._sdk_from_local)
            self._lo = np.array([kin.cspace_coord_limits(i).lower for i in range(self.n)])
            self._hi = np.array([kin.cspace_coord_limits(i).upper for i in range(self.n)])
            self._vmax = np.array([kin.cspace_coord_velocity_limit(i) for i in range(self.n)])
            if (not np.isfinite([self._lo, self._hi, self._vmax]).all()
                    or np.any(self._hi - self._lo <= 2 * self._margin)
                    or np.any(self._vmax <= 0)):
                raise PlanningError("cuMotion requires finite ordered position and positive velocity limits")
            self._world = self._cm.create_world()
            for box in boxes:
                obstacle = self._cm.create_obstacle(self._cm.Obstacle.Type.CUBOID)
                obstacle.set_attribute(self._cm.Obstacle.Attribute.SIDE_LENGTHS,
                                       np.array(box["size_m"]))
                self._world.add_obstacle(obstacle, self._cm.Pose3(np.array(box["T_base_box"])))
                self._obstacles.append(obstacle)
            self._view = self._world.add_world_view()
            self._view.update()
            self._inspector = self._cm.create_robot_world_inspector(self._robot, self._view)
            if (self._inspector.num_world_collision_spheres() <= 0
                    or self._inspector.num_self_collision_spheres() <= 0):
                raise PlanningError("XRDF must provide world and self collision spheres")
            def optimizer(extra):
                sdk_cfg = self._cm.create_default_trajectory_optimizer_config(
                    self._robot, self.tool_frame, self._view)
                for key, value in {"enable_self_collision": True, "enable_world_collision": True,
                                   **extra}.items():
                    if not sdk_cfg.set_param(key, value):
                        raise PlanningError(f"cuMotion rejected required parameter {key}")
                return self._cm.create_trajectory_optimizer(sdk_cfg)
            self._optimizer = optimizer({})
            # A declared contact policy, selected before planning: initialize
            # L-BFGS with the direct joint path instead of particle exploration.
            # Both optimizers still return native curves and retain all gates.
            self._contact_optimizer = optimizer({"trajopt/pbo/enabled": False,
                                                 **PATH_POSITION_WEIGHTS})
        except Exception as exc:
            self.close()
            if isinstance(exc, PlanningError):
                raise
            raise PlanningError(f"cuMotion initialization failed: {exc}") from exc

    @staticmethod
    def _boxes(value):
        if not isinstance(value, (list, tuple)):
            raise PlanningError("obstacles must explicitly be a list of static cuboids")
        boxes, names = [], set()
        for box in value:
            if not isinstance(box, dict) or set(box) != {"name", "size_m", "T_base_box"}:
                raise PlanningError("each cuboid requires name, size_m and T_base_box")
            name = _name(box["name"], "obstacle name")
            if name in names:
                raise PlanningError("obstacle names must be unique")
            names.add(name)
            size = _vector(box["size_m"], 3, "cuboid size_m")
            try:
                T = np.asarray(box["T_base_box"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise PlanningError("cuboid pose must be a rigid 4x4 transform") from exc
            if (np.any(size <= 0) or T.shape != (4, 4) or not np.isfinite(T).all()
                    or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                    or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-8, rtol=0)
                    or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-8, rtol=0)):
                raise PlanningError("cuboid requires positive full dimensions and a rigid 4x4 pose")
            boxes.append({"name": name, "size_m": size.tolist(), "T_base_box": T.tolist()})
        return boxes

    def _to_sdk(self, q):
        return (q * self._signs)[self._sdk_from_local]

    def _to_local(self, q):
        return q[self._local_from_sdk] * self._signs

    def _positions_valid(self, q, margin):
        return np.all(q >= self._lo + margin) and np.all(q <= self._hi - margin)

    def plan(self, start, goal):
        """Return copied samples. Native failures are not retried or substituted."""
        return self._request(start, goal)

    def plan_profile(self, start, goal, *, duration_s, rate_hz, max_velocity,
                     linear_tool_path=False, joint_margin=None):
        """Copy the original native curve at every command and safety time.

        Uniform slowing preserves the geometric curve. It grants no actuator
        authority and never replaces SafeArm's live validation or feedback.
        """
        options = (_number(duration_s, "requested duration", positive=True),
                   _number(rate_hz, "command rate", positive=True),
                   _number(max_velocity, "host velocity limit", positive=True))
        if type(linear_tool_path) is not bool:
            raise PlanningError("linear_tool_path must be a boolean")
        return self._request(start, goal, profile_options=options,
                             linear_tool_path=linear_tool_path, joint_margin=joint_margin)

    def _request(self, start, goal, *, profile_options=None, linear_tool_path=False,
                 joint_margin=None):
        margin = self._margin if joint_margin is None else _number(joint_margin, "joint_margin")
        if margin < 0 or np.any(self._hi - self._lo <= 2 * margin):
            raise PlanningError("joint_margin must be nonnegative and leave a valid joint interval")
        start = _vector(start, self.n, "start")
        goal = _vector(goal, self.n, "goal")
        with self._lock:
            if self._closed or self._poisoned:
                raise PlanningError("cuMotion planner is closed or faulted; construct a new instance")
            qs, qg = self._to_sdk(start), self._to_sdk(goal)
            if not self._positions_valid(qs, margin) or not self._positions_valid(qg, margin):
                raise PlanningError("start/goal violates model joint limits or margin")
            try:
                return self._plan(qs, qg, start, goal, profile_options=profile_options,
                                  linear_tool_path=linear_tool_path, margin=margin)
            except Exception as exc:
                # No assumptions about native scratch state after an exception.
                self._poisoned = True
                if isinstance(exc, PlanningError):
                    raise
                raise PlanningError(f"cuMotion planning failed: {exc}") from exc

    def _plan(self, qs, qg, start, goal, *, profile_options=None, linear_tool_path=False, margin):
        target_type = self._cm.TrajectoryOptimizer.CSpaceTarget
        path_check = lambda q: None
        if linear_tool_path:
            target = target_type(qg,
                # Leave interpolation headroom; independently enforce the
                # 1 mm corridor on every exported command and safety sample.
                target_type.TranslationPathConstraint.linear(LINEAR_PATH_TOLERANCE_M / 10.),
                target_type.OrientationPathConstraint.constant(ORIENTATION_PATH_TOLERANCE_RAD))
            kin = self._robot.kinematics()
            first = np.asarray(kin.pose(qs, self.tool_frame).matrix(), dtype=float)
            last = np.asarray(kin.pose(qg, self.tool_frame).matrix(), dtype=float)
            delta = last[:3, 3] - first[:3, 3]
            length_squared = float(delta @ delta)
            def path_check(q):
                pose = np.asarray(kin.pose(q, self.tool_frame).matrix(), dtype=float)
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise PlanningError("cuMotion returned invalid constrained-path FK")
                fraction = (float((pose[:3, 3] - first[:3, 3]) @ delta) / length_squared
                            if length_squared > 1e-16 else 0.)
                closest = first[:3, 3] + np.clip(fraction, 0., 1.) * delta
                error = np.linalg.norm(pose[:3, 3] - closest)
                angle = np.arccos(np.clip((np.trace(last[:3, :3].T @ pose[:3, :3]) - 1.) / 2., -1., 1.))
                if error > LINEAR_PATH_TOLERANCE_M + 1e-8 or angle > ORIENTATION_PATH_TOLERANCE_RAD + 1e-8:
                    raise PlanningError("cuMotion curve violates the requested linear tool path "
                                        f"(translation={error:.9g} m, orientation={angle:.9g} rad)")
            path_check(qs)
            path_check(qg)
        else:
            target = target_type(qg)
        optimizer = self._contact_optimizer if linear_tool_path else self._optimizer
        result = optimizer.plan_to_cspace_target(qs, target)
        status = result.status()
        if status != self._cm.TrajectoryOptimizer.Results.Status.SUCCESS:
            raise PlanningError(f"cuMotion returned {status}")
        trajectory = result.trajectory()
        if trajectory.num_cspace_coords() != self.n:
            raise PlanningError("cuMotion trajectory joint count mismatch")
        domain = trajectory.domain()
        lower = _number(domain.lower, "trajectory domain lower")
        upper = _number(domain.upper, "trajectory domain upper")
        duration = upper - lower
        if not 0 < duration <= self._max_duration:
            raise PlanningError("cuMotion trajectory duration outside configured budget")
        intervals = duration / self._dt
        if not math.isfinite(intervals) or intervals > self._max_samples - 1:
            raise PlanningError("cuMotion trajectory exceeds sample budget")
        count = max(2, math.ceil(intervals) + 1)
        min_q = _vector(trajectory.min_position(), self.n, "trajectory minima")
        max_q = _vector(trajectory.max_position(), self.n, "trajectory maxima")
        max_v = _vector(trajectory.max_velocity_magnitude(), self.n, "trajectory max velocity")
        if (not self._positions_valid(min_q, margin) or not self._positions_valid(max_q, margin)
                or np.any(min_q > max_q) or np.any(max_v < 0)
                or np.any(max_v > self._vmax + 1e-8)):
            raise PlanningError("cuMotion trajectory extrema violate model limits")
        times = np.linspace(0, duration, count)
        positions, velocities = [], []
        for t in times:
            q = _vector(trajectory.eval(lower + float(t), 0), self.n, "trajectory position")
            v = _vector(trajectory.eval(lower + float(t), 1), self.n, "trajectory velocity")
            if (not self._positions_valid(q, margin) or np.any(np.abs(v) > self._vmax + 1e-8)
                    or self._inspector.in_self_collision(q)
                    or self._inspector.in_collision_with_obstacle(q)):
                raise PlanningError("cuMotion trajectory sample violates limits or collision model")
            positions.append(tuple(self._to_local(q).tolist()))
            velocities.append(tuple(self._to_local(v).tolist()))
        if (not np.allclose(positions[0], start, atol=self._endpoint_tol, rtol=0)
                or not np.allclose(positions[-1], goal, atol=self._endpoint_tol, rtol=0)):
            raise PlanningError("cuMotion trajectory endpoints do not match the request")
        # A corrupt native result must not hide a jump behind a zero derivative.
        local_vmax = self._vmax[self._local_from_sdk]
        if np.any(np.abs(np.diff(positions, axis=0)) > np.diff(times)[:, None] * local_vmax + 1e-8):
            raise PlanningError("cuMotion samples exceed velocity bounds between samples")
        request = {"start": start.tolist(), "goal": goal.tolist(),
                   "joint_names": self.joint_names, "joint_signs": self._signs.tolist(),
                   "base_frame": self.base_frame, "tool_frame": self.tool_frame,
                   "model_sha256": self.model_sha256, "scene_sha256": self.scene_sha256,
                   "sample_dt_s": self._dt, "joint_margin": margin,
                   "max_duration_s": self._max_duration, "max_samples": self._max_samples,
                   "sdk_version": SDK_VERSION,
                   "endpoint_tolerance": self._endpoint_tol,
                   "linear_tool_path": linear_tool_path,
                   "path_position_weights": PATH_POSITION_WEIGHTS if linear_tool_path else {},
                   "particle_seed": not linear_tool_path}
        plan = MotionPlan(self.joint_names, tuple(times.tolist()), tuple(positions),
                          tuple(velocities), self.base_frame, self.tool_frame,
                          self.model_sha256, self.scene_sha256, _digest(request),
                          path_constraint="linear_tool" if linear_tool_path else "unconstrained")
        if profile_options is None:
            return plan
        from .trajectory import TrajectoryProfile
        minimum_duration, rate, velocity_limit = profile_options
        execution_duration = max(duration, minimum_duration,
                                 duration * float(np.max(max_v)) / (.9 * velocity_limit))
        if execution_duration > self._max_duration:
            raise PlanningError("slowed cuMotion trajectory exceeds duration budget")
        cache = {}
        def evaluate(fraction):
            if fraction not in cache:
                q = _vector(trajectory.eval(lower + float(fraction) * duration, 0),
                            self.n, "trajectory safety position")
                if (not self._positions_valid(q, margin) or self._inspector.in_self_collision(q)
                        or self._inspector.in_collision_with_obstacle(q)):
                    raise PlanningError("cuMotion safety sample violates limits or collision model")
                path_check(q)
                cache[fraction] = self._to_local(q)
            return cache[fraction]
        return TrajectoryProfile.from_curve(evaluate, execution_duration, rate, plan,
                                             max_steps=min(10000, self._max_samples - 1))

    def close(self):
        """Release solver before the native owners it references."""
        with self._lock:
            self._closed = True
            self._optimizer = self._contact_optimizer = self._inspector = self._view = None
            self._world = None
            self._obstacles = []
            self._robot = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
