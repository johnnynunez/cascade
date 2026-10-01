"""Observed-surface and endpoint-occlusion vetoes for calibrated reBot fingers.

This is discrete trajectory checking, NOT a continuous swept-volume or free-
space certificate. Occluded surfaces are unknown. Approach retains all target
points. Only deliberate closure excludes the exact same-frame target mask,
while checking each finger's complete mechanical stroke before its command.
Endpoints also reject hull ray intervals behind observed non-target depth;
this does not certify unmeasured space or add a continuous trajectory check.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from ..control.motion_profile import nominal_profile
from ..control.simulation_motion import PhysicsClock
from ..types import SafetyViolation

ROOT = Path(__file__).resolve().parents[3]
GEOMETRY = ROOT / "assets/grasp_geometry/rebot_rs_fingers.json"
PHYSX_GEOMETRY = ROOT / "assets/grasp_geometry/rebot_rs_fingers_physx.json"
# Fixed empirical tracking envelope, including both fingers separately.
# See docs/evidence/observed-finger-gate/open-jaw-tracking.json. Not a universal
# actuator-noise bound; departures abort, never expand it adaptively.
OPEN_TRACKING_ENVELOPE_M = .0001


def _array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise SafetyViolation(f"observed-finger gate requires finite {name} of shape {shape}")
    return a.astype(float)


class FingerGeometry:
    def __init__(self, path=GEOMETRY, *, expected_engine=None):
        path = Path(path)
        try:
            payload = path.read_bytes()
            doc = json.loads(payload)
        except (OSError, ValueError) as exc:
            raise SafetyViolation("calibrated observed-finger geometry unavailable or invalid") from exc
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if doc.get("version") != 1 or doc.get("units") != "m" or doc.get("frame") != "gripper_end":
            raise SafetyViolation("unsupported observed-finger geometry")
        self.physics_engine = doc.get("physics_engine")
        if expected_engine is not None and self.physics_engine != expected_engine:
            raise SafetyViolation("observed-finger collision geometry engine mismatch")
        self.collision_representation = doc.get("collision_representation")
        for relative, expected in doc["sources"].items():
            try:
                source_sha = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            except OSError as exc:
                raise SafetyViolation("observed-finger geometry source unavailable") from exc
            if source_sha != expected:
                raise SafetyViolation("observed-finger geometry source hash mismatch")
        self.sources = doc["sources"]
        self.fingers = doc["fingers"]
        self.names = [f["name"] for f in self.fingers]
        self.lower = np.array([f["lower_m"] for f in self.fingers])
        self.upper = np.array([f["upper_m"] for f in self.fingers])
        if self.names != ["joint_left", "joint_right"] or len(self.fingers) != 2:
            raise SafetyViolation("incomplete calibrated finger geometry")
        for finger in self.fingers:
            finger["axis_tcp"] = _array(finger["axis_tcp"], (3,), "finger axis")
            if not np.isclose(np.linalg.norm(finger["axis_tcp"]), 1., atol=1e-12, rtol=0):
                raise SafetyViolation("finger axis is not unit length")
            if not finger["components"]:
                raise SafetyViolation("missing finger collision components")
            for part in finger["components"]:
                planes = np.asarray(part["planes"], dtype=float)
                if (planes.ndim != 2 or planes.shape[1] != 4 or len(planes) < 4
                        or not np.isfinite(planes).all()
                        or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-12, rtol=0)):
                    raise SafetyViolation("invalid calibrated collision halfspaces")
                part["planes"] = planes
                part["min_m"] = _array(part["min_m"], (3,), "hull bound")
                part["max_m"] = _array(part["max_m"], (3,), "hull bound")

    def strokes(self, snapshot):
        if (not isinstance(snapshot, dict) or type(snapshot.get("version")) is not int
                or snapshot["version"] != 1 or snapshot.get("names") != self.names):
            raise SafetyViolation("individual calibrated finger feedback missing")
        lo = _array(snapshot.get("lower_m"), (2,), "finger lower limits")
        hi = _array(snapshot.get("upper_m"), (2,), "finger upper limits")
        # The bridge serializes float32-authored limits; compare to that exact
        # representation, not a tolerance that could accept another gripper.
        if (not np.array_equal(lo.astype(np.float32), self.lower.astype(np.float32))
                or not np.array_equal(hi.astype(np.float32), self.upper.astype(np.float32))):
            raise SafetyViolation("finger stroke limits differ from calibrated geometry")
        return _array(snapshot.get("position_m"), (2,), "individual finger positions")

    def envelopes(self, lower, upper):
        """Contain every component translated throughout each stroke interval.

        Offsetting its existing unit-normal planes is a conservative outer
        envelope of the extrusion. Concavities may cause extra rejections.
        Bounds/radius contain that tested region: add the original component
        AABB as six more planes so the broadphase cannot discard a positive.
        """
        lower, upper = _array(lower, (2,), "stroke interval"), _array(upper, (2,), "stroke interval")
        if np.any(lower > upper):
            raise SafetyViolation("reversed finger stroke interval")
        result = []
        for i, finger in enumerate(self.fingers):
            axis = finger["axis_tcp"]
            shift_lo = np.minimum(axis * lower[i], axis * upper[i])
            shift_hi = np.maximum(axis * lower[i], axis * upper[i])
            for index, part in enumerate(finger["components"]):
                planes = part["planes"].copy()
                projection = planes[:, :3] @ axis
                planes[:, 3] -= np.maximum(projection * lower[i], projection * upper[i])
                lo, hi = part["min_m"] + shift_lo - 2e-12, part["max_m"] + shift_hi + 2e-12
                bound_planes = np.concatenate((np.c_[np.eye(3), -hi], np.c_[-np.eye(3), lo]))
                planes = np.concatenate((planes, bound_planes))
                center = (lo + hi) / 2
                result.append((finger["name"], index, planes, lo, hi, center,
                               float(np.linalg.norm(hi - lo) / 2)))
        return result


class ObservedScene:
    def __init__(self, frame, target_mask, T_base_cam, *, source, robot_id, epoch):
        from scipy.spatial import cKDTree

        capture = frame.capture
        if (not isinstance(capture, dict) or capture.get("backend") != "isaac"
                or capture.get("source") != source or not isinstance(capture.get("camera"), str)):
            raise SafetyViolation("observed scene is not bound to this Isaac camera source")
        prop = capture.get("proprioception")
        if (not isinstance(prop, dict) or type(prop.get("version")) is not int or prop["version"] != 1
                or prop.get("backend") != "isaac" or prop.get("robot_id") != robot_id
                or prop.get("joint_convention") != "asset" or prop.get("producer_epoch") != epoch
                or not isinstance(epoch, str) or not epoch
                or prop.get("time_source") != "physics_loop_monotonic"
                or type(prop.get("t")) not in (int, float) or not np.isfinite(prop["t"])
                or prop["t"] < 0 or prop["t"] != capture.get("t")):
            raise SafetyViolation("observed scene capture robot, epoch, or timestamp mismatch")
        depth = np.asarray(frame.depth_m)
        if (frame.depth_source != "sensor" or depth.ndim != 2 or depth.dtype.kind not in "fu"
                or frame.rgb.shape[:2] != depth.shape):
            raise SafetyViolation("observed scene requires aligned metric sensor depth")
        masks = []
        for value, name in ((frame.robot_mask, "robot"), (target_mask, "target")):
            # Deliberate host safety boundary after CUDA perception. This does
            # not replace detection/localization with a CPU fallback. Torch is
            # optional: keep its tensor protocol out of this module's imports.
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            mask = np.asarray(value)
            if mask.shape != depth.shape or mask.dtype != np.bool_:
                raise SafetyViolation(f"observed scene requires exact same-frame boolean {name} mask")
            masks.append(mask.copy())
        robot, target = masks
        K = _array(frame.K, (3, 3), "camera intrinsics")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.array_equal(K[2], [0, 0, 1]):
            raise SafetyViolation("invalid observed-scene camera intrinsics")
        T = _array(T_base_cam, (4, 4), "capture camera transform")
        if (not np.array_equal(T[3], [0, 0, 0, 1]) or np.linalg.det(T[:3, :3]) <= 0
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=3e-6, rtol=0)):
            raise SafetyViolation("observed-scene transform must be calibrated and rigid")
        valid = np.isfinite(depth) & (depth > 0) & ~robot
        rows, cols = np.nonzero(valid)
        if not np.any(target & valid):
            raise SafetyViolation("observed scene has no visible target surface")
        rays = np.c_[cols, rows, np.ones(len(rows))] @ np.linalg.inv(K).T
        self.points = (rays * depth[valid, None]) @ T[:3, :3].T + T[:3, 3]
        self.is_target = target[valid]
        self.pixels = np.c_[rows, cols]
        self.tree = cKDTree(self.points)
        # Immutable, same-capture inputs for the additional endpoint veto.
        # Target surfaces still participate in the existing approach check.
        self._depth = depth.copy()
        self._occluder = valid & ~target
        self._K = K.copy()
        self._inverse_K = np.linalg.inv(K)
        self._T_base_cam = T.copy()
        self.identity = (source, robot_id, epoch)
        self.receipt = {"camera": capture["camera"], "capture_t": capture["t"],
                        "source": source, "robot_id": robot_id, "epoch": epoch,
                        "observed_points": len(rows), "target_points": int(self.is_target.sum()),
                        "robot_pixels_excluded": int(robot.sum()),
                        "scope": "observed surfaces only; no occlusion or continuous-path certificate"}

    def conflict(self, T_base_tcp, envelopes, *, allow_target_contact=False):
        T = _array(T_base_tcp, (4, 4), "TCP transform")
        R, t = T[:3, :3], T[:3, 3]
        if (not np.array_equal(T[3], [0, 0, 0, 1]) or np.linalg.det(R) <= 0
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=0)):
            raise SafetyViolation("observed-finger TCP transform must be rigid")
        inverse = np.linalg.inv(R)
        radius_scale = float(np.linalg.norm(R, ord=2))
        for name, part, planes, lo, hi, center, radius in envelopes:
            ids = np.asarray(self.tree.query_ball_point(R @ center + t, radius * radius_scale + 1e-12), dtype=int)
            if allow_target_contact:
                # A local closure-only exemption. Keep the original scene,
                # target mask and spatial index intact for approach checks.
                ids = ids[~self.is_target[ids]]
            if not len(ids):
                continue
            points = (self.points[ids] - t) @ inverse.T
            inside = np.all((points >= lo) & (points <= hi), axis=1)
            ids, points = ids[inside], points[inside]
            # Bounded blocks avoid a dense scene × thousands-of-planes matrix.
            for start in range(0, len(ids), 128):
                subset = points[start:start + 128]
                hits = np.flatnonzero(np.all(subset @ planes[:, :3].T + planes[:, 3] <= 0, axis=1))
                if len(hits):
                    i = int(ids[start + hits[0]])
                    return {"finger": name, "component": part,
                            "surface": "target" if self.is_target[i] else "other observed surface",
                            "pixel_rc": self.pixels[i].tolist(), "point_base_m": self.points[i].tolist()}
        return None

    def occluded(self, T_base_tcp, envelopes, *, check=None):
        """Find a finger ray interval behind observed non-target depth.

        Uses the same discrete pixel rays and depth-Z convention as the point
        cloud. This is an additional endpoint veto, not a free-space or
        swept-volume certificate. Holes, robot pixels, target pixels and
        out-of-view space make no claim here; the surface veto is unchanged.
        """
        def checked():
            if check is not None:
                check()
        checked()
        T = _array(T_base_tcp, (4, 4), "endpoint TCP transform")
        R = T[:3, :3]
        if (not np.array_equal(T[3], [0, 0, 0, 1]) or np.linalg.det(R) <= 0
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=0)):
            raise SafetyViolation("observed-finger endpoint transform must be rigid")
        camera_to_tcp = np.linalg.inv(T) @ self._T_base_cam
        tcp_to_camera = np.linalg.inv(self._T_base_cam) @ T
        origin = camera_to_tcp[:3, 3]
        height, width = self._depth.shape
        for name, part, planes, lo, hi, _center, _radius in envelopes:
            checked()
            corners = np.array(list(itertools.product(*zip(lo, hi))))
            camera = corners @ tcp_to_camera[:3, :3].T + tcp_to_camera[:3, 3]
            if not np.isfinite(camera).all():
                raise SafetyViolation("non-finite finger projection")
            if camera[:, 2].max() <= 0:
                continue  # outside the measured camera half-space
            if camera[:, 2].min() <= 0:
                checked()
                return {"finger": name, "component": part,
                        "surface": "endpoint envelope crosses camera plane"}
            projection = camera @ self._K.T
            pixels = projection[:, :2] / projection[:, 2, None]
            if not np.isfinite(pixels).all():
                raise SafetyViolation("non-finite finger pixel projection")
            # Clamp before integer conversion, including completely off-screen
            # envelopes; no inf/large integer cast can produce an empty ROI.
            left = int(np.clip(np.floor(pixels[:, 0].min()), 0, width))
            right = int(np.clip(np.ceil(pixels[:, 0].max()) + 1, 0, width))
            top = int(np.clip(np.floor(pixels[:, 1].min()), 0, height))
            bottom = int(np.clip(np.ceil(pixels[:, 1].max()) + 1, 0, height))
            depth = self._depth[top:bottom, left:right]
            possible = self._occluder[top:bottom, left:right] & (depth <= camera[:, 2].max())
            rows, cols = np.nonzero(possible)
            rows, cols = rows + top, cols + left
            for start in range(0, len(rows), 1024):
                checked()
                rr, cc = rows[start:start + 1024], cols[start:start + 1024]
                rays = np.c_[cc, rr, np.ones(len(rr))] @ self._inverse_K.T
                directions = rays @ camera_to_tcp[:3, :3].T
                near, far = np.zeros(len(rr)), np.full(len(rr), np.inf)
                live = np.ones(len(rr), dtype=bool)
                for plane in planes:
                    slope = directions @ plane[:3]
                    offset = float(origin @ plane[:3] + plane[3])
                    nonzero = slope != 0
                    bound = np.divide(-offset, slope, out=np.zeros_like(slope), where=nonzero)
                    near = np.maximum(near, np.where(slope < 0, bound, -np.inf))
                    far = np.minimum(far, np.where(slope > 0, bound, np.inf))
                    live &= nonzero | (offset <= 0)
                hits = np.flatnonzero(live & (far >= near) & (far > self._depth[rr, cc]))
                checked()
                if len(hits):
                    i = int(hits[0])
                    return {"finger": name, "component": part,
                            "surface": "occluded behind observed non-target depth",
                            "pixel_rc": [int(rr[i]), int(cc[i])],
                            "observed_depth_z_m": float(self._depth[rr[i], cc[i]]),
                            "finger_ray_near_z_m": float(near[i]),
                            "finger_ray_far_z_m": float(far[i])}
        checked()
        return None


class ObservedFingerGate:
    def __init__(self, scene, geometry, kin, state, *, uncertainty_m=0.):
        if type(uncertainty_m) not in (int, float) or not np.isfinite(uncertainty_m) or uncertainty_m < 0:
            raise SafetyViolation("finger uncertainty must be a nonnegative calibrated metric bound")
        self.scene, self.geometry, self.kin = scene, geometry, kin
        strokes = geometry.strokes(state.gripper_joints)
        self.lower = np.minimum(strokes, geometry.upper) - uncertainty_m
        self.upper = np.maximum(strokes, geometry.upper) + uncertainty_m
        self.envelopes = geometry.envelopes(self.lower, self.upper)
        # Conservative geometric extent, not a tracking guarantee under
        # contact. Both fingers independently sweep the whole stroke, with
        # the same fixed expansion used for opening; never grow it on error.
        self.closing_lower = geometry.lower - uncertainty_m
        self.closing_upper = self.upper.copy()
        self.closing_envelopes = geometry.envelopes(self.closing_lower, self.closing_upper)
        self.clock = PhysicsClock(*scene.identity[:2])
        self.last_state = None
        self._last_q = self._last_strokes = None
        q = np.asarray(state.q)
        if q.ndim != 1 or not q.size:
            raise SafetyViolation("observed-finger gate requires a joint vector")
        self._q_shape = q.shape
        self._cache = {}
        self._closing_cache = {}
        self._occlusion_cache = {}
        self.feedback(state)

    def pose(self, q):
        q = np.asarray(q, dtype=float)
        key = q.tobytes()
        if key not in self._cache:
            self._cache[key] = self.scene.conflict(self.kin.fk(q), self.envelopes)
        return self._cache[key]

    def closing_pose(self, q):
        q = _array(q, self._q_shape, "joint closing pose")
        key = q.tobytes()
        if key not in self._closing_cache:
            self._closing_cache[key] = self.scene.conflict(
                self.kin.fk(q), self.closing_envelopes, allow_target_contact=True)
        return self._closing_cache[key]

    def occluded_pose(self, q, *, closing=False, check=None):
        if check is not None:
            check()
        q = _array(q, self._q_shape, "endpoint joint pose")
        key = (closing, q.tobytes())
        if key not in self._occlusion_cache:
            self._occlusion_cache[key] = self.scene.occluded(
                self.kin.fk(q), self.closing_envelopes if closing else self.envelopes,
                check=check)
        if check is not None:
            check()
        return self._occlusion_cache[key]

    def _observe(self, state):
        """Common atomic snapshot contract for open motion and closing."""
        fresh = self.clock.observe(state.physics_clock)
        if state.physics_clock["epoch"] != self.scene.identity[2]:
            raise SafetyViolation("observed-finger scene belongs to another simulation epoch")
        strokes = self.geometry.strokes(state.gripper_joints)
        q = _array(state.q, self._q_shape, "joint feedback")
        if (not fresh and self._last_q is not None
                and (not np.array_equal(q, self._last_q) or not np.array_equal(strokes, self._last_strokes))):
            raise SafetyViolation("arm or finger feedback changed without a new physics step")
        return q, strokes

    def _remember(self, state, q, strokes):
        self.last_state = state
        self._last_q, self._last_strokes = q.copy(), strokes.copy()

    def feedback(self, state):
        q, strokes = self._observe(state)
        if np.any(strokes < self.lower) or np.any(strokes > self.upper):
            raise SafetyViolation("finger feedback left the geometrically checked opening interval")
        conflict = self.scene.conflict(self.kin.fk(q), self.geometry.envelopes(strokes, strokes))
        if conflict:
            raise SafetyViolation(f"measured finger intersects observed surface: {conflict}")
        self._remember(state, q, strokes)

    def require_closing(self, state, *, check=None):
        """Veto the complete closing envelope at the measured pose pre-command.

        This does not stop an already commanded closure or refresh the scene.
        The second stage may start partly closed but retains the same atomic
        snapshot/epoch contract and immutable geometric stroke bounds.
        """
        q, strokes = self._observe(state)
        if np.any(strokes < self.closing_lower) or np.any(strokes > self.closing_upper):
            raise SafetyViolation("finger feedback left the geometrically checked closing interval")
        conflict = self.closing_pose(q)
        if conflict:
            raise SafetyViolation(f"closing fingers intersects observed non-target surface: {conflict}")
        conflict = self.occluded_pose(q, closing=True, check=check)
        if conflict:
            raise SafetyViolation(f"closing fingers occluded by non-target depth: {conflict}")
        self._remember(state, q, strokes)

    def profile(self, start, target, duration_s, rate_hz, *, check=None):
        """Same discrete samples/complete edges as executor, including lookahead."""
        for sample_index, waypoint in enumerate(nominal_profile(start, target, duration_s, rate_hz)):
            if check is not None:
                check()
            for previous, following, _dt in waypoint.checks:
                for q in (previous, following):
                    conflict = self.pose(q)
                    if conflict:
                        return {"sample_index": sample_index, **conflict}
        return None

    def require_profile(self, start, target, duration_s, rate_hz, *, check=None):
        conflict = self.profile(start, target, duration_s, rate_hz, check=check)
        if conflict:
            raise SafetyViolation(f"finger trajectory intersects observed surface: {conflict}")


def for_runtime(runtime, frame, fix, state):
    """Bind an opt-in gate to existing state/calibration; no arm/camera reads."""
    from ..control.isaac_arm import IsaacArm
    from ..control.lazy_arm import LazyArm

    raw = runtime.arm.raw
    if isinstance(raw, LazyArm):
        raw = raw._arm  # existing runtime get_state already materialized it
    if not isinstance(raw, IsaacArm):
        raise SafetyViolation("observed-finger gate currently requires calibrated Isaac reBot")
    # Select the calibrated collision representation only after the existing
    # clock contract identifies its engine, source, robot and producer epoch.
    # PhysX convex decomposition can protrude beyond the nominal STL mesh.
    # Its envelope retains the nominal mesh and adds requested derived hulls;
    # it does not change solver colliders, contact offsets or tracking limits.
    clock = PhysicsClock(raw._client._addr, raw._cfg.get("bridge_robot_id"))
    clock.observe(state.physics_clock)
    geometry = (FingerGeometry(PHYSX_GEOMETRY, expected_engine="physx")
                if state.physics_clock["engine"] == "physx" else FingerGeometry())
    model = Path(runtime.cfg.arm.model)
    expected = geometry.sources.get("assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf")
    if (runtime.cfg.arm.ee_frame != "gripper_end" or runtime.kin.ee_frame != "gripper_end"
            or hashlib.sha256(model.read_bytes()).hexdigest() != expected):
        raise SafetyViolation("observed-finger geometry does not match runtime kinematics")
    # Capture decoder validates q, robot, endpoint and capture timestamp without
    # transport IO. The producer epoch is then matched to current physics.
    raw.state_from_frame(frame)
    T = frame.T_base_cam
    if T is None:
        # Only the identified primary camera may borrow its static calibration.
        # Secondary localization binds its own static T onto the returned Frame.
        capture = frame.capture or {}
        primary = runtime.cfg.camera.get("sim_camera")
        ext = runtime.extrinsics
        if (capture.get("camera") == primary and getattr(ext, "mode", None) == "eye_to_hand"):
            T = ext.T
    scene = ObservedScene(frame, fix.detection.mask, T, source=raw._client._addr,
                          robot_id=raw._cfg.get("bridge_robot_id"), epoch=state.physics_clock["epoch"])
    gate = ObservedFingerGate(scene, geometry, runtime.kin, state,
                              uncertainty_m=OPEN_TRACKING_ENVELOPE_M)
    from . import evidence
    evidence.event("observed_finger_gate", scene=scene.receipt, geometry_sha256=geometry.sha256,
                   physics_engine=state.physics_clock["engine"],
                   collision_representation=geometry.collision_representation,
                   stroke_lower_m=gate.lower, stroke_upper_m=gate.upper,
                   closing_stroke_lower_m=gate.closing_lower,
                   closing_stroke_upper_m=gate.closing_upper,
                   tracking_envelope_m=OPEN_TRACKING_ENVELOPE_M,
                   scope="CPU safety veto after CUDA perception; discrete approach and pre-command closing stroke; fingers only")
    return gate
