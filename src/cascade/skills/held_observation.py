"""Keep aiming estimates separate from evidence that can justify releasing."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import time

import numpy as np

from ..control.simulation_motion import PhysicsClock
from ..sim.truth import _match_label
from ..sim.render_binding import valid_render_binding
from ..types import SkillError, SafetyViolation


class HeldObservationInvalid(SkillError):
    """The held episode lost its identity or authorization; do not retry it."""


@dataclass
class HeldOffset:
    offset: np.ndarray | None
    channel: str
    release_authority: bool = False
    evidence: dict = field(default_factory=dict)
    invalidated: bool = False


def binding(runtime):
    arm = runtime.arm
    raw = getattr(arm, "raw", arm)
    return (arm, raw, getattr(raw, "_arm", raw), getattr(arm, "harness", None), runtime.cfg)


def _identity(condition, message):
    if not condition:
        raise HeldObservationInvalid(message)


def _clock_identity(clock, endpoint, robot, epoch):
    _identity(isinstance(clock, dict) and clock.get("source") == endpoint
              and clock.get("robot_id") == robot and clock.get("epoch") == epoch,
              "held observation producer identity changed")


def _vector(value, size):
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("invalid held-object measurement")
    return array.astype(float)


def _offset(position, tcp):
    offset = _vector(position, 3) - _vector(tcp, 3)
    if np.linalg.norm(offset[:2]) > .12:
        raise ValueError("object is outside the existing held-offset sanity gate")
    return offset


def _clock_binding(runtime, current):
    cfg = runtime.cfg.arm
    endpoint = (str(cfg.get("bridge_host", "127.0.0.1")), int(cfg.get("bridge_port", 8611)))
    robot = cfg.get("bridge_robot_id")
    clock = current.physics_clock
    floor = getattr(runtime, "_held_observation_floor", None)
    try:
        validator = PhysicsClock(endpoint, robot)
        validator.observe(floor)
        _clock_identity(clock, endpoint, robot, floor["epoch"])
        validator.observe(clock)
    except SafetyViolation as exc:
        if floor is not None:
            raise HeldObservationInvalid("held physics clock changed or regressed: " + str(exc)) from exc
        raise
    return endpoint, robot, floor, clock


def _same_step_q(runtime, q, clock, current, floor):
    if clock["physics_step"] == current.physics_clock["physics_step"]:
        if not np.array_equal(q, current.q):
            raise HeldObservationInvalid("different joints reported for the same physical step")
    if clock["physics_step"] == floor["physics_step"]:
        previous = _vector(runtime._held_observation_floor_q, len(q))
        if not np.array_equal(q, previous):
            raise HeldObservationInvalid("capture contradicts the completed lift step")


def _asset_q(runtime, sample, n):
    """Use exactly the materialized driver's asset-to-local conversion."""
    backend = binding(runtime)[2]
    signs = _vector(runtime.cfg.arm.get("joint_signs", [1] * n), n)
    driver_signs = _vector(getattr(backend, "_signs", None), n)
    retained = getattr(runtime, "_held_observation_joint_signs", driver_signs)
    _identity(np.isin(signs, (-1., 1.)).all() and np.array_equal(signs, driver_signs)
              and np.array_equal(signs, retained), "held joint convention changed")
    q_asset = _vector(sample.get("q_asset"), n)
    return backend._decode_state({"q": q_asset}).q


def _base_position(sample):
    if sample.get("pose_frame") != "world" or type(sample.get("meters_per_unit")) not in (int, float) or sample["meters_per_unit"] != 1.:
        raise ValueError("atomic pose frame or unit conversion unsupported")
    base = _vector(sample.get("base_position_world"), 3)
    w, x, y, z = quaternion = _vector(sample.get("base_orientation_wxyz"), 4)
    if abs(np.linalg.norm(quaternion) - 1.) > 1e-6:
        raise ValueError("atomic base orientation is not a unit quaternion")
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return rotation.T @ (_vector(sample["position_m"], 3) - base)


def measure(runtime, localize) -> HeldOffset:
    """Observe once; a missing/legacy channel remains an aiming estimate.

    Atomic simulator samples include object pose and robot q from the same
    physics step. Images use their own captured q, never a current TCP. Both
    authorities must be after the completed lift in the same producer epoch.
    """
    if not runtime.held_object:
        return HeldOffset(None, "none")
    started = time.monotonic()
    original = binding(runtime)
    retained = getattr(runtime, "_held_observation_binding", original)
    if any(a is not b for a, b in zip(original, retained)):
        return HeldOffset(None, "invalidated", invalidated=True,
                          evidence={"rejected": ["held arm or configuration identity changed"]})
    labels = tuple(dict.fromkeys(filter(None, (runtime.held_object,
        getattr(runtime, "_held_det_label", None)))))
    current = tcp = None
    try:
        cfg = getattr(runtime, "cfg", None)
        isaac = getattr(cfg, "arm", {}).get("type") == "isaac"
        current = runtime.arm.get_state(**({"timeout_s": 1.} if isaac else {}))
        tcp = runtime.kin.fk(current.q)[:3, 3]
        if getattr(runtime, "_held_observation_floor", None) is not None:
            _clock_binding(runtime, current)
    except HeldObservationInvalid as exc:
        return HeldOffset(None, "invalidated", invalidated=True, evidence={"rejected": [str(exc)]})
    except Exception:
        pass
    channel = getattr(runtime, "_object_pose", None)
    observe = getattr(channel, "observation", None)
    rejected = []
    if callable(observe):
        try:
            sample = observe(labels)
            if not isinstance(sample, dict) or type(sample.get("version")) is not int or sample["version"] != 1:
                raise ValueError("atomic object observation unavailable")
            endpoint, robot, floor, now = _clock_binding(runtime, current)
            _identity(tuple(sample.get("source", ())) == endpoint, "atomic object source changed")
            if not sample.get("resolved_path"):
                raise ValueError("atomic object source or identity missing")
            name = sample.get("resolved_name")
            if (not isinstance(name, str)
                    or sample["resolved_path"].rsplit("/", 1)[-1] != name
                    or _match_label(runtime.held_object, {name: name}) != name):
                raise ValueError("atomic object does not match the held identity")
            stamp = sample.get("server_monotonic")
            if type(stamp) not in (int, float) or not np.isfinite(stamp) or stamp < 0:
                raise ValueError("atomic object timestamp missing")
            clock = copy.deepcopy(sample["physics_clock"])
            clock["source"] = endpoint
            _clock_identity(clock, endpoint, robot, now["epoch"])
            try:
                validator = PhysicsClock(endpoint, robot)
                validator.observe(floor)
                validator.observe(now)
                validator.observe(clock)
            except SafetyViolation as exc:
                raise HeldObservationInvalid("atomic physics clock changed or regressed: " + str(exc)) from exc
            n = len(current.q)
            indices = sample.get("joint_indices")
            if (sample.get("joint_convention") != "asset" or n != 6
                    or sample.get("joint_names") != [f"joint{i}" for i in range(1, 7)]
                    or not isinstance(indices, list) or len(indices) != n
                    or any(type(i) is not int or i < 0 for i in indices)
                    or len(set(indices)) != n):
                raise ValueError("atomic joint convention or control-joint map invalid")
            q = _asset_q(runtime, sample, n)
            _same_step_q(runtime, q, clock, current, floor)
            position = _base_position(sample)
            offset = _offset(position, runtime.kin.fk(q)[:3, 3])
            if time.monotonic() - started > 2.:
                raise ValueError("held observation exceeded its freshness interval")
            return HeldOffset(offset, "atomic_physics", True,
                {"resolved_name": sample["resolved_name"], "resolved_path": sample["resolved_path"],
                 "physics_clock": clock, "q": q.tolist(), "position_base_m": position.tolist(),
                 "position_world_m": sample["position_m"]})
        except HeldObservationInvalid as exc:
            return HeldOffset(None, "invalidated", invalidated=True, evidence={"rejected": [str(exc)]})
        except Exception as exc:
            rejected.append(str(exc))
    elif callable(channel) and tcp is not None:
        # Old vector APIs have no timestamp, robot pose, epoch or body identity.
        # Retain compensation compatibility without authorizing an opening.
        try:
            positions = [channel(label) for label in labels]
            positions = [_vector(p, 3) for p in positions if p is not None]
            if positions and all(np.array_equal(p, positions[0]) for p in positions):
                return HeldOffset(_offset(positions[0], tcp), "legacy_pose")
        except Exception as exc:
            rejected.append(str(exc))
    try:
        frame = runtime.observe()
        if getattr(runtime, "_held_observation_floor", None) is not None:
            endpoint, robot, floor, now = _clock_binding(runtime, current)
            capture = getattr(frame, "capture", None)
            if isinstance(capture, dict) and capture.get("backend", "isaac") == "isaac":
                proprio = capture.get("proprioception", {})
                _identity(tuple(capture.get("source", ())) == endpoint
                    and proprio.get("robot_id") == robot
                    and proprio.get("producer_epoch") == now["epoch"],
                    "held image source, robot or epoch changed")
        raw = getattr(runtime.arm, "raw", runtime.arm)
        captured = raw.state_from_frame(frame)
        captured_tcp = runtime.kin.fk(captured.q)[:3, 3]
        fix = localize(frame, getattr(runtime, "_held_det_label", None) or runtime.held_object,
            runtime.detector, runtime.extrinsics, prompts=[labels[-1]], near_xyz=captured_tcp,
            workspace_bounds=runtime._localization_workspace_bounds())
        offset = _offset(fix.position, captured_tcp)
        evidence = {"capture": copy.deepcopy(frame.capture)}
        authority = False
        try:
            endpoint, robot, floor, now = _clock_binding(runtime, current)
            capture = frame.capture
            reference = capture["render_reference"]
            if (tuple(capture["source"]) != endpoint
                    or type(reference.get("version")) is not int or reference["version"] != 1
                    or not valid_render_binding(capture)
                    or capture["t"] != reference["snapshot_started_monotonic"]
                    or capture["proprioception"]["robot_id"] != robot
                    or capture["proprioception"]["producer_epoch"] != now["epoch"]
                    or reference["producer_epoch"] != now["epoch"]):
                raise ValueError("image identity differs")
            history = dict(now, physics_step=reference["history_physics_step"],
                sim_time=reference["history_simulation_time"], source=endpoint)
            validator = PhysicsClock(endpoint, robot)
            validator.observe(floor)
            validator.observe(now)
            validator.observe(history)
            _same_step_q(runtime, captured.q, history, current, floor)
            # A packet already buffered before this check cannot authorize
            # release. Require a newer physical capture acquired within the
            # existing two-second observation bound, using only local elapsed
            # time and same-producer simulation steps (no cross-host clocks).
            if history["physics_step"] <= now["physics_step"] or time.monotonic() - started > 2.:
                raise ValueError("image is not a new post-check physical capture")
            # Robot/camera identity alone does not identify the object. Only
            # an unambiguous measured instance containing every selected pixel
            # supplies that extra authority; ordinary detector results aim only.
            masks = getattr(frame, "prop_masks", None) or {}
            names = {}
            for path in masks:
                name = path.rsplit("/", 1)[-1]
                if name in names:
                    raise ValueError("ambiguous image instance names")
                names[name] = path
            path = _match_label(runtime.held_object, names)
            chosen = np.asarray(fix.detection.mask)
            target = np.asarray(masks[path])
            if (chosen.dtype.kind != "b" or target.dtype.kind != "b"
                    or chosen.shape != target.shape or not chosen.any()
                    or not np.array_equal(chosen, target)):
                raise ValueError("visual object identity is not measured unambiguously")
            evidence["resolved_path"] = path
            authority = True
        except HeldObservationInvalid:
            raise
        except Exception as exc:
            evidence["authority_rejected"] = str(exc)
        return HeldOffset(offset, "captured_image", authority, evidence)
    except HeldObservationInvalid as exc:
        return HeldOffset(None, "invalidated", invalidated=True, evidence={"rejected": [str(exc)]})
    except Exception as exc:
        rejected.append(str(exc))
    cached = getattr(runtime, "_held_offset", None)
    try:
        cached = _vector(cached, 3)
    except Exception:
        cached = None
    return HeldOffset(cached, "cached_aim" if cached is not None else "unknown",
                      evidence={"rejected": rejected})
