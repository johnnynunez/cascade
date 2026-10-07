"""Explicit whole-body composition: a manipulation arm mounted on a mobile base.

Contract (docs/ROBOT_MODULARITY.md -> "Multi-domain embodiments: mounted arms"):

* Actuating domains claim DISJOINT command endpoints; an overlap names both
  claimants. Disjoint joint subsets are not disjoint endpoints.
* ``world <- base`` is the base state at its capture time (local monotonic
  receipt minus producer age, bound to the base epoch); ``base <- arm_base``
  is the declared mount. Lookups go through the spatial ``FrameTree``: bounded
  zero-order hold, never extrapolation. Missing, stale, re-epoched or fallen
  base poses refuse mounted-arm motion before any arm IO.
* Coordination is ``exclusive`` by default; ``concurrent`` is an explicit
  profile opt-in that lifts only the two active-command vetoes.
* ``RobotRuntime`` owns the per-domain stop latches (global stop, explicit
  per-domain reset).
* World-frame postconditions use the frame valid at command start; a mount
  that moved during the command refutes the outcome, an unavailable end frame
  leaves it unverified, and synthetic channels never confirm.

Only ``reach_world_point`` streams an actuator, and it does so through the
arm's ``SafeArm``: the SafetyHarness remains the sole authority that refuses
motion. Compositions with any physical actuating resource are refused.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .contracts import identifier

if TYPE_CHECKING:  # typing only; the mobile contract stays a lazy import
    from ..control.mobile_base import BaseState


POLICIES = ("exclusive", "concurrent")
_REQUIRED = {"version", "max_mount_drift_m", "max_mount_drift_rad", "world_target_tolerance_m"}
_ALLOWED = _REQUIRED | {"coordination", "max_base_pose_age_s"}
_MOUNT_KEYS = {"domain", "base", "translation_m", "rotation_wxyz", "calibration_id"}
PHYSICAL_GATES = ("a measured mount calibration, an independent base-pose source, moving-frame arm limits "
                  "(static workspace/table limits are arm-base-frame assumptions) and a validated whole-body "
                  "controller with balance-preserving stop are required")
MOUNTED_TOOLS = ("get_arm_world_pose", "reach_world_point")
MOUNTED_TOOL_SPECS = (
    {"name": "get_arm_world_pose",
     "description": ("Read the mounted arm base pose in the base's world frame: the latest base state at its "
                     "capture time composed with the declared mount. Passive and never extrapolated; a kinematic "
                     "mock base pose is synthetic. The TCP is reported only when the arm is already connected."),
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "reach_world_point",
     "description": ("Move the TCP to a WORLD-frame point in metres, keeping the current tool orientation. The "
                     "target is resolved once with the mount frame valid at command start, the SafetyHarness vets "
                     "every waypoint and the outcome is checked in world with that same frame. Refused while the "
                     "base has an active command (exclusive coordination) or its pose is missing or stale."),
     "parameters": {"type": "object", "properties": {k: {"type": "number"} for k in "xyz"},
                    "required": ["x", "y", "z"], "additionalProperties": False}},
)


class MountFrameError(ValueError):
    """The mount frame at a capture time cannot be established; it is never guessed."""


def _finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _vector(values, size, label):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{label} must contain {size} finite values")
    return tuple(_finite(v, label) for v in values)


def pose_matrix(position, rotation_wxyz):
    w, x, y, z = rotation_wxyz
    value = np.eye(4)
    value[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    value[:3, 3] = position
    return value


def _rows(matrix):
    return [[float(v) for v in row] for row in np.asarray(matrix)]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Mount:
    """Declared ``T_base_armbase``: a configuration value, not a calibration."""
    arm_domain: str
    base_domain: str
    base: str
    translation_m: tuple
    rotation_wxyz: tuple
    calibration_id: str

    def __post_init__(self):
        for key in ("arm_domain", "base_domain", "base", "calibration_id"):
            identifier(getattr(self, key), key)
        object.__setattr__(self, "translation_m", _vector(self.translation_m, 3, "mount translation_m"))
        rotation = _vector(self.rotation_wxyz, 4, "mount rotation_wxyz")
        if abs(sum(v * v for v in rotation) - 1.0) > 1e-6:
            raise ValueError("mount rotation_wxyz must be a unit quaternion; no silent normalization")
        object.__setattr__(self, "rotation_wxyz", rotation)

    @property
    def matrix(self):
        return pose_matrix(self.translation_m, self.rotation_wxyz)

    def as_dict(self):
        return {"arm_domain": self.arm_domain, "base_domain": self.base_domain, "base": self.base,
                "translation_m": list(self.translation_m), "rotation_wxyz": list(self.rotation_wxyz),
                "calibration_id": self.calibration_id, "source": "declared_profile"}


# ── configuration / composition admission (no device is constructed) ─────────

def _validate_mount(name, profile, domains, explicit_age):
    if profile.get("kind") != "manipulation":
        raise ValueError("only manipulation domains can be mounted on a locomotion base")
    raw = profile["mounted_on"]
    if not isinstance(raw, Mapping):
        raise ValueError("mounted_on must be a mount object (domain, base, translation_m, rotation_wxyz, calibration_id)")
    if set(raw) - _MOUNT_KEYS:
        raise ValueError(f"unknown mount fields: {sorted(set(raw) - _MOUNT_KEYS)}")
    if _MOUNT_KEYS - set(raw):
        raise ValueError("mount requires domain, base, translation_m, rotation_wxyz and calibration_id")
    base_domain = raw["domain"]
    carrier = domains.get(base_domain) if isinstance(base_domain, str) else None
    if not isinstance(carrier, Mapping) or carrier.get("kind") != "locomotion" or base_domain == name:
        raise ValueError(f"mount domain {base_domain!r} must name a locomotion domain of this robot")
    base = next((b for b in carrier["resolved"]["bases"] if b.get("name") == raw["base"]), None)
    if base is None:
        raise ValueError(f"mount base {raw['base']!r} is not configured in locomotion domain {base_domain!r}")
    mount = Mount(arm_domain=name, base_domain=base_domain, base=raw["base"], translation_m=raw["translation_m"],
                  rotation_wxyz=raw["rotation_wxyz"], calibration_id=raw["calibration_id"])
    arms = profile["resolved"]["arms"]
    if len(arms) != 1:
        raise ValueError("a mounted manipulation domain needs exactly one arm; multi-arm mounts need per-arm mount transforms")
    if arms[0].get("base_pose") is not None:
        raise ValueError("a mounted arm cannot also declare a static base_pose; its pose comes from the base and the mount")
    limit = _finite(base["safety"]["max_state_age_s"], "base safety.max_state_age_s", positive=True)
    age = limit if explicit_age is None else _finite(explicit_age, "max_base_pose_age_s", positive=True)
    if age > limit:
        raise ValueError(f"max_base_pose_age_s {age} may only tighten the base profile's safety.max_state_age_s ({limit})")
    return {**mount.as_dict(), "max_base_pose_age_s": age}


def validate_contract(raw, domains):
    """Normalise a robot profile's ``whole_body`` block against its resolved domains."""
    if not isinstance(raw, Mapping) or set(raw) - _ALLOWED or _REQUIRED - set(raw):
        raise ValueError("whole_body requires version, max_mount_drift_m, max_mount_drift_rad and "
                         "world_target_tolerance_m; optional coordination and max_base_pose_age_s; "
                         "unknown fields are refused")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("whole_body requires version 1")
    coordination = raw.get("coordination", "exclusive")
    if coordination not in POLICIES:
        raise ValueError("coordination must be exclusive or concurrent")
    tolerances = {key: _finite(raw[key], key, positive=True)
                  for key in ("max_mount_drift_m", "max_mount_drift_rad", "world_target_tolerance_m")}
    mounts = [_validate_mount(name, profile, domains, raw.get("max_base_pose_age_s"))
              for name, profile in domains.items() if "mounted_on" in profile]
    if not mounts:
        raise ValueError("whole_body requires at least one mounted manipulation domain")
    return {"version": 1, "coordination": coordination, **tolerances, "mounts": mounts}


def check_disjoint_endpoints(domains):
    """Each command endpoint (controller_id) belongs to exactly one domain."""
    claims = {}
    for name, domain in domains.items():
        for resource in domain.resources:
            if resource.controller_id is None:
                continue
            owner = claims.setdefault(resource.controller_id, (name, resource.resource_id))
            if owner[0] != name:
                raise ValueError(
                    f"command endpoint {resource.controller_id!r} is claimed by {owner[1]!r} (domain {owner[0]}) "
                    f"and {resource.resource_id!r} (domain {name}); whole-body domains require disjoint "
                    "command endpoints")


def admit_composition(domains, contract):
    """Static whole-body admission over described domains; never opens a device."""
    mounted = {m["arm_domain"] for m in contract["mounts"]}
    carriers = {m["base_domain"] for m in contract["mounts"]}
    actuating = {name: d for name, d in domains.items() if d.motion_skills}
    for name, domain in actuating.items():
        kind = domain.profile["kind"]
        if not ((kind == "manipulation" and name in mounted) or (kind == "locomotion" and name in carriers)):
            raise ValueError(f"actuating domain {name!r} ({kind}) is not covered by the whole-body contract; "
                             "only locomotion bases and the manipulation domains mounted on them may join it")
    check_disjoint_endpoints(actuating)
    physical = sorted(r.resource_id for d in actuating.values() for r in d.resources
                      if r.writer_id is not None and not r.synthetic)
    if physical:
        raise ValueError(f"physical whole-body composition is not admitted: {PHYSICAL_GATES}; "
                         f"physical resources: {physical}")
    return actuating


def contract_metadata(contract):
    """Passive discovery metadata; a declared contract is not physical admission."""
    if contract is None:
        return {}
    return {"whole_body": {**copy.deepcopy(dict(contract)), "physical_admission": False,
                           "frames": "world <- base (state at capture time) <- arm_base (declared mount)"}}


# ── dynamic frame chain ─────────────────────────────────────────────────────

_KIND = {"kinematic_mock": "synthetic", "physics": "physics", "hardware": "measured"}


@dataclass(frozen=True)
class ArmFrame:
    """``T_world_armbase`` valid at ``at_time_s``, with the sample that made it."""
    matrix: np.ndarray
    base_state: "BaseState"
    base_capture_time_s: float
    at_time_s: float
    mount: Mount

    @property
    def epoch(self):
        return self.base_state.epoch

    @property
    def synthetic(self):
        return self.base_state.measurement_kind == "kinematic_mock"

    def to_arm(self, point_world):
        return (np.linalg.inv(self.matrix) @ np.append(np.asarray(point_world, dtype=float), 1.0))[:3]

    def to_world(self, point_arm):
        return (self.matrix @ np.append(np.asarray(point_arm, dtype=float), 1.0))[:3]

    def as_dict(self):
        s = self.base_state
        return {"world_from_arm_base": _rows(self.matrix),
                "world_from_base": _rows(pose_matrix(s.position_world, s.orientation_wxyz)),
                "base_from_arm_base": _rows(self.mount.matrix),
                "base_capture_time_s": self.base_capture_time_s, "at_time_s": self.at_time_s,
                "age_s": self.at_time_s - self.base_capture_time_s, "epoch": s.epoch,
                "base_state": {"robot_id": s.robot_id, "source": s.source, "epoch": s.epoch, "step": s.step,
                               "sim_time_s": s.sim_time_s, "position_world": list(s.position_world),
                               "orientation_wxyz": list(s.orientation_wxyz),
                               "controller_status": s.controller_status, "latched": s.latched,
                               "measurement_kind": s.measurement_kind},
                "mount": self.mount.as_dict(), "measurement_kind": _KIND[s.measurement_kind],
                "extrapolated": False, "physical_admission": False}


class FrameChain:
    """``world <- base <- arm_base`` over a spatial FrameTree, one tree per base epoch.

    The clock is the local monotonic clock; a sample's capture time is its
    receipt time minus the producer's age. Lookups hold the newest sample at or
    before the requested time for at most ``max_base_pose_age_s``.
    """
    CLOCK = "local_monotonic"

    def __init__(self, mount, *, max_base_pose_age_s, history=64):
        if not isinstance(mount, Mount):
            raise ValueError("Mount required")
        self.mount = mount
        self.max_age_s = _finite(max_base_pose_age_s, "max_base_pose_age_s", positive=True)
        if type(history) is not int or not 2 <= history <= 4096:
            raise ValueError("history must be in 2..4096")
        self.history = history
        self._lock = threading.Lock()
        self._tree = None
        self._identity = None
        self._samples = OrderedDict()  # capture time -> BaseState, current epoch only

    def _context(self, state):
        map_id = "base-" + _digest([state.robot_id, state.source])[:24]
        return map_id, "epoch-" + _digest(state.epoch)[:24]

    def observe(self, state):
        """Record one completed base state; returns its capture time."""
        from ..control.mobile_base import BaseState
        from ..spatial.frames import FrameTree, SpatialStamp, TransformSample
        if not isinstance(state, BaseState):
            raise MountFrameError("base pose missing: no completed BaseState")
        if state.fallen:
            raise MountFrameError("base reports fallen; a fallen base is not a valid mount frame")
        capture = state.received_monotonic_s - state.producer_age_s
        if capture < 0:
            raise MountFrameError("base pose missing: capture time precedes the local clock origin")
        identity = (state.robot_id, state.source, state.epoch, state.measurement_kind)
        map_id, epoch = self._context(state)
        source = _digest([state.robot_id, state.source, state.epoch, state.model_identity_sha256])
        source_id = "base-source-" + source[:24]
        kind = _KIND[state.measurement_kind]
        with self._lock:
            if self._tree is None or identity != self._identity:
                # A new epoch or provider starts a new tree; old samples never leak across.
                self._tree = FrameTree(map_id, epoch, self.CLOCK, "world", history=self.history)
                self._identity, self._samples = identity, OrderedDict()
                self._tree.add(TransformSample("world", "base", state.position_world, state.orientation_wxyz,
                                               SpatialStamp(map_id, epoch, self.CLOCK, capture, source_id,
                                                            source, "base-state", kind)))
                self._tree.add(TransformSample("base", "arm_base", self.mount.translation_m,
                                               self.mount.rotation_wxyz,
                                               SpatialStamp(map_id, epoch, self.CLOCK, capture,
                                                            "declared-mount", _digest(self.mount.as_dict()),
                                                            self.mount.calibration_id, "synthetic"),
                                               static=True))
            elif capture > next(reversed(self._samples)):
                self._tree.add(TransformSample("world", "base", state.position_world, state.orientation_wxyz,
                                               SpatialStamp(map_id, epoch, self.CLOCK, capture, source_id,
                                                            source, "base-state", kind)))
            elif capture == next(reversed(self._samples)):
                if self._samples[capture].position_world != state.position_world or \
                        self._samples[capture].orientation_wxyz != state.orientation_wxyz:
                    raise MountFrameError("conflicting base poses for one capture time; never averaged")
                return capture  # the same completed state read again
            else:
                raise MountFrameError("stale base pose: feedback older than an already observed sample "
                                      "(out of order); the newer sample is not superseded")
            self._samples[capture] = state
            while len(self._samples) > self.history:
                self._samples.popitem(last=False)
        return capture

    def arm_base_in_world(self, *, at_time_s, epoch):
        """``T_world_armbase`` valid at ``at_time_s``; MountFrameError instead of a guess."""
        when = _finite(at_time_s, "at_time_s")
        with self._lock:
            if self._tree is None:
                raise MountFrameError("base pose missing: no completed base state observed")
            if epoch != self._identity[2]:
                raise MountFrameError("base epoch changed; a frame from another epoch is never reused")
            capture = max((t for t in self._samples if t <= when), default=None)
            if capture is None:
                raise MountFrameError("base pose missing at the requested capture time")
            age = when - capture
            if age > self.max_age_s:
                raise MountFrameError(f"stale base pose: age {age:.3f} s exceeds {self.max_age_s:.3f} s")
            map_id, tree_epoch = self._context(self._samples[capture])
            result = self._tree.lookup("world", "arm_base", time_s=when, epoch=tree_epoch,
                                       clock_id=self.CLOCK, max_age_s=self.max_age_s)
            return ArmFrame(np.array(result.matrix), self._samples[capture], capture, when, self.mount)


# ── runtime coordinator ──────────────────────────────────────────────────────

def _unverified(reason, **extra):
    return {"status": "unverified", "reason": reason, **extra}


class WholeBodyCoordinator:
    """Admission, dynamic frames and world-frame verdicts for one composed robot.

    RobotRuntime calls ``admit`` -> ``execute`` -> ``finish`` for each tool and
    owns the per-domain stop latches; this object never resets a stop.
    """

    def __init__(self, contract, domains):
        self.contract = copy.deepcopy(dict(contract))
        self.coordination = self.contract["coordination"]
        self.domains = domains
        self.mounts, self.chains = {}, {}
        for raw in self.contract["mounts"]:
            mount = Mount(**{k: raw[k] for k in ("arm_domain", "base_domain", "base", "translation_m",
                                                 "rotation_wxyz", "calibration_id")})
            self.mounts[mount.arm_domain] = mount
            self.chains[mount.arm_domain] = FrameChain(mount, max_base_pose_age_s=raw["max_base_pose_age_s"])
        self.carriers = {m.base_domain for m in self.mounts.values()}
        self.reset_domains = tuple(sorted(domains))

    def metadata(self):
        return contract_metadata(self.contract)

    # -- reads -------------------------------------------------------------
    def _read_base(self, mount):
        runtime = self.domains[mount.base_domain].runtime
        try:
            return runtime.passive_state(mount.base)
        except Exception as exc:  # unreadable is missing, never a default pose
            raise MountFrameError(f"base pose missing ({type(exc).__name__}: {exc})") from exc

    def _frame(self, arm_domain, *, epoch=None):
        mount = self.mounts[arm_domain]
        state = self._read_base(mount)
        self.chains[arm_domain].observe(state)
        frame = self.chains[arm_domain].arm_base_in_world(at_time_s=time.monotonic(),
                                                          epoch=state.epoch if epoch is None else epoch)
        return state, frame

    def _arm_motion_in_flight(self, arm_domain):
        runtime = self.domains[arm_domain].runtime
        rig = getattr(runtime, "arm_rig", None)
        arms = dict(rig.arms) if rig is not None else {"default": runtime.arm}
        # Passive harness flag between begin_motion and end_motion; never reads the backend.
        return [name for name, arm in arms.items() if getattr(arm.harness, "_motion_active", False) is True]

    def arm_world_pose(self, arm_domain):
        try:
            _, frame = self._frame(arm_domain)
        except ValueError as exc:
            return {"ok": False, "error": f"mounted arm frame unavailable ({exc}); never extrapolated",
                    "physical_admission": False}
        runtime = self.domains[arm_domain].runtime
        result = {"ok": True, "frame": frame.as_dict(), "tcp_world": None, "tcp_arm_base": None,
                  "arm_connected": False, "physical_admission": False,
                  "tcp_source": "arm not connected; reading its feedback would materialize the backend"}
        # LazyArm.connected is on its passive surface; never power an arm to report a pose.
        if getattr(runtime.arm.raw, "connected", False) is True:
            tcp = runtime.kin.fk(runtime.arm.get_state().q)[:3, 3]
            result.update(tcp_arm_base=[float(v) for v in tcp], tcp_world=[float(v) for v in frame.to_world(tcp)],
                          arm_connected=True, tcp_source="arm joint feedback (the actuator's own channel)")
        return result

    # -- admission -----------------------------------------------------------
    def admit(self, descriptor, args):
        domain = descriptor.domain
        if domain in self.mounts:
            mount = self.mounts[domain]
            try:
                state, frame = self._frame(domain)
            except MountFrameError as exc:
                raise ValueError(f"mounted arm frame unavailable ({exc}); manipulation motion refused, "
                                 "never extrapolated") from exc
            if state.controller_status in {"fault", "disabled"}:
                raise ValueError(f"coordination: base {mount.base!r} controller reports "
                                 f"{state.controller_status!r}; a mounted arm needs a controlled base")
            base_busy = getattr(self.domains[mount.base_domain].runtime, "_active", False) is True
            if self.coordination == "exclusive" and (state.controller_status != "ready" or base_busy):
                raise ValueError(f"coordination (exclusive): base {mount.base!r} reports an active command; "
                                 "mounted arm motion refused until the base is ready")
            return {"kind": "arm", "domain": domain, "mount": mount, "frame": frame, "base_status": state.controller_status}
        if domain in self.carriers:
            if self.coordination == "exclusive":
                for arm_domain, mount in self.mounts.items():
                    busy = self._arm_motion_in_flight(arm_domain) if mount.base_domain == domain else []
                    if busy:
                        raise ValueError(f"coordination (exclusive): mounted arm {busy[0]!r} has a motion in "
                                         "flight; locomotion motion refused")
            return {"kind": "base", "domain": domain}
        return None

    def execute(self, context, domain, name, args):
        if name in MOUNTED_TOOLS and domain.domain_id in self.mounts:
            if name == "get_arm_world_pose":
                return self.arm_world_pose(domain.domain_id)
            if context is None or context.get("kind") != "arm":
                raise ValueError("reach_world_point requires whole-body admission")
            return self._reach(context, domain, args)
        return domain.execute(name, args)

    def _reach(self, context, domain, args):
        frame = context["frame"]
        target_world = np.array([_finite(args[k], k) for k in "xyz"])
        target_arm = frame.to_arm(target_world)
        base = {"target_world": [float(v) for v in target_world], "target_arm_base": [float(v) for v in target_arm],
                "frame_at_start": frame.as_dict()}
        runtime = domain.runtime
        if getattr(runtime, "held_object", None):
            return {**base, "ok": False, "execution_ok": False,
                    "error": "reach_world_point does not carry objects; place or release the held object first",
                    "postcondition": _unverified("no motion")}
        arm, kin = runtime.arm, runtime.kin
        q_now = arm.get_state().q  # motion context: materializes the lazy arm
        goal = kin.fk(q_now).copy()
        goal[:3, 3] = target_arm
        ik = kin.ik(goal, q_now)
        if not ik.success:
            return {**base, "ok": False, "execution_ok": False,
                    "error": "no IK solution keeping the current tool orientation for the world target "
                             "resolved with the frame valid at command start",
                    "postcondition": _unverified("no motion")}
        hold = runtime.watcher.paused() if getattr(runtime, "watcher", None) is not None else contextlib.nullcontext()
        try:
            with hold:
                settled = bool(arm.move_joints(ik.q, duration_s=2.0))  # harness approves every waypoint
        except Exception as exc:
            return {**base, "ok": False, "execution_ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "postcondition": _unverified(f"motion interrupted: {exc}")}
        tcp_arm = kin.fk(arm.get_state().q)[:3, 3]
        tcp_world = frame.to_world(tcp_arm)
        error = float(np.linalg.norm(tcp_world - target_world))
        tolerance = self.contract["world_target_tolerance_m"]
        evidence = {"world_error_m": error, "tolerance_m": tolerance, "frame": "valid at command start",
                    "tcp_channel": "arm joint feedback", "mount_drift_m": None, "mount_drift_rad": None}
        if error > tolerance:
            verdict = {"status": "refuted", "evidence": evidence,
                       "reason": f"TCP {error:.4f} m from the world target (tolerance {tolerance} m), measured "
                                 "with the frame valid at command start"}
        else:
            synthetic = frame.synthetic or any(r.synthetic for r in domain.resources)
            reason = ("synthetic channels (kinematic mock base pose and arm feedback) cannot confirm a physical "
                      "world-frame outcome" if synthetic else
                      "the TCP comes from the arm's own joint feedback; no independent world-frame observation "
                      "confirms it")
            verdict = _unverified(reason, observed_status="consistent", evidence=evidence)
        return {**base, "ok": False, "execution_ok": settled, "tcp_arm_base": [float(v) for v in tcp_arm],
                "tcp_world": [float(v) for v in tcp_world], "postcondition": verdict, "outcome": verdict["status"]}

    # -- world-frame verdict -----------------------------------------------
    def finish(self, context, result):
        if context is None or context.get("kind") != "arm" or not isinstance(result, dict):
            return result
        start = context["frame"]
        annotation = {"coordination": self.coordination, "base_status_at_start": context["base_status"],
                      "frame_at_start": start.as_dict(), "frame_at_end": None}
        try:
            _, end = self._frame(context["domain"], epoch=start.epoch)
        except ValueError as exc:
            world = {"status": "unverified", "mount_drift_m": None, "mount_drift_rad": None,
                     "reason": f"frame at command end unavailable ({exc}); the start frame cannot be shown to hold"}
        else:
            annotation["frame_at_end"] = end.as_dict()
            drift_m = float(np.linalg.norm(end.matrix[:3, 3] - start.matrix[:3, 3]))
            relative = start.matrix[:3, :3].T @ end.matrix[:3, :3]
            drift_rad = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
            limit_m, limit_rad = self.contract["max_mount_drift_m"], self.contract["max_mount_drift_rad"]
            if drift_m > limit_m or drift_rad > limit_rad:
                world = {"status": "refuted", "mount_drift_m": drift_m, "mount_drift_rad": drift_rad,
                         "reason": f"mounted arm base moved {drift_m:.4f} m / {drift_rad:.4f} rad during the "
                                   f"command (tolerance {limit_m} m / {limit_rad} rad); the frame valid at "
                                   "command start no longer holds"}
            else:
                world = {"status": "consistent", "mount_drift_m": drift_m, "mount_drift_rad": drift_rad,
                         "reason": "mount frame held within tolerance from command start to end"
                                   + (" (synthetic base pose: consistency only, not physical evidence)"
                                      if end.synthetic else "")}
        annotation["world_frame"] = world
        result = {**result, "whole_body": annotation}
        postcondition = result.get("postcondition")
        if isinstance(postcondition, dict) and isinstance(postcondition.get("evidence"), dict):
            postcondition = {**postcondition, "evidence": {**postcondition["evidence"],
                                                           "mount_drift_m": world["mount_drift_m"],
                                                           "mount_drift_rad": world["mount_drift_rad"]}}
            result["postcondition"] = postcondition
        if world["status"] == "refuted" or (world["status"] == "unverified" and isinstance(postcondition, dict)
                                            and postcondition.get("status") == "confirmed"):
            # A refuted/unestablished frame downgrades a self-reported success; the claim is kept.
            result["self_reported_ok"] = result.get("ok")
            result["ok"] = False
            result["postcondition"] = {"status": world["status"], "reason": world["reason"],
                                       "arm_postcondition": postcondition}
            result["outcome"] = world["status"]
        return result
