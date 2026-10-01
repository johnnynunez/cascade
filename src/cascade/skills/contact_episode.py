"""Bounded post-close geometry and explicitly resumed contact withdrawal.

No recovery opens jaws, resets props, extends contact geometry or changes
collision tolerances. The originating arm retains the failed episode.
"""
from __future__ import annotations

import time
import numpy as np

from ..perception.freshness import capture_marker, read_frame_after
from ..perception.occupancy import OccupancyError
from ..types import SafetyViolation, SkillError


def _identity(runtime):
    cfg = runtime.cfg.arm
    return tuple(cfg.get(key) for key in ("type", "bridge_host", "bridge_port", "bridge_robot_id"))


def _backend(arm):
    from ..control.lazy_arm import LazyArm
    return arm.raw._arm if isinstance(arm.raw, LazyArm) else arm.raw


def begin(runtime, q_pre):
    harness = runtime.arm.harness
    occupancy = getattr(harness, "occupancy", None)
    if not getattr(occupancy, "tracks_payload", False):
        return None
    harness.check_contact_episode()
    cylinder = harness._grasp_exempt
    if cylinder is None:
        raise SkillError("payload close has no original contact exemption")
    episode = {"arm": runtime.arm, "raw": runtime.arm.raw, "backend": _backend(runtime.arm),
               "identity": _identity(runtime),
               "halt_generation": harness._halt_generation,
               "q_pre": np.asarray(q_pre, dtype=float).copy(),
               "cylinder": (cylinder[0].copy(), cylinder[1], cylinder[2]),
               "expected_paths": None, "q_failure": None, "barrier": None,
               "source_identity": None, "camera_keys": {}, "floor_markers": [],
               "camera_streams": None, "stream_capture_keys": {}}
    runtime._contact_episode = episode
    harness._pending_contact_episode = episode
    harness._contact_scope.value = (episode, True)
    return episode


def _guard(runtime, episode):
    arm, harness = runtime.arm, runtime.arm.harness
    if (arm is not episode["arm"] or arm.raw is not episode["raw"]
            or _backend(arm) is not episode["backend"]
            or _identity(runtime) != episode["identity"]
            or harness._pending_contact_episode is not episode):
        raise SafetyViolation("contact recovery arm identity changed")
    # No heartbeat is invented here: live perception must remain available.
    harness.check_stream_start(halt_generation=episode["halt_generation"])


def wait_geometry(runtime, episode, *, recovery=False):
    """Drain post-close deliveries, then await their successful map commits."""
    if episode is None:
        return None
    harness = runtime.arm.harness
    harness._contact_scope.value = (episode, False)
    deadline = time.monotonic() + 5.
    watcher = getattr(runtime, "watcher", None)
    cameras = ([cam.stream for cam in watcher._cams if cam.maps_depth]
               if watcher is not None else [runtime.camera])
    if not cameras:
        raise SkillError("payload tracking has no map-depth cameras")
    if episode["camera_streams"] is None and not recovery:
        episode["camera_streams"] = tuple(cameras)
    if (episode["camera_streams"] is None
            or len(cameras) != len(episode["camera_streams"])
            or {id(c) for c in cameras} != {id(c) for c in episode["camera_streams"]}):
        raise SkillError("retained contact map-depth camera set changed")
    floors = []
    try:
        # Establish every floor before accepting any camera. The first new
        # delivery may have been in flight at close; only a later producer
        # capture on the same identity/clock may satisfy the map barrier.
        for camera in cameras:
            _guard(runtime, episode)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("post-close camera floor deadline expired")
            floor = read_frame_after(camera, timeout_s=remaining)
            marker = capture_marker(floor)
            if marker.get("backend") != "isaac":
                raise ValueError("payload requires an Isaac producer floor")
            source_identity = (tuple(marker["source"]), marker["robot_id"], marker["clock"])
            if episode["source_identity"] is None and not recovery:
                episode["source_identity"] = source_identity
            if source_identity != episode["source_identity"]:
                raise SkillError("retained contact producer/robot/clock identity changed")
            key = harness.occupancy._capture_key(marker)
            previous = episode["camera_keys"].get(marker["camera"])
            stream_previous = episode["stream_capture_keys"].get(id(camera))
            if ((previous is not None and previous != key)
                    or (stream_previous is not None and stream_previous != key)):
                raise SkillError("retained contact camera identity changed")
            episode["camera_keys"][marker["camera"]] = key
            episode["stream_capture_keys"][id(camera)] = key
            if not recovery:
                episode["floor_markers"].append(marker)
            floors.append(floor)
            paths = tuple(sorted((floor.capture or {}).get("contact_paths", [])))
            if paths:
                if episode["expected_paths"] is None and not recovery:
                    episode["expected_paths"] = paths
                elif paths != episode["expected_paths"]:
                    raise SkillError("post-close captured contact identity changed")
        result = harness.occupancy.wait_payload_ready(
            floors, deadline=deadline, guard=lambda: _guard(runtime, episode),
            expected_paths=episode["expected_paths"],
        )
        if episode["expected_paths"] is None:
            episode["expected_paths"] = tuple(result["contact_paths"])
        episode["barrier"] = result
        return result
    except (OccupancyError, TimeoutError, ValueError) as exc:
        raise SkillError(f"post-close payload geometry unavailable: {exc}") from exc


def finish(runtime, episode, *, completed, capture_feedback=True):
    if episode is None:
        return
    harness = runtime.arm.harness
    harness._contact_scope.value = None
    if completed:
        harness._pending_contact_episode = None
        runtime._contact_episode = None
    elif capture_feedback:
        # A partial lift can stop away from q_grasp. Preserve measured final
        # feedback, never infer the failed endpoint from the command target.
        try:
            q = np.asarray(runtime.arm.get_state().q, dtype=float)
            episode["q_failure"] = q.copy() if np.isfinite(q).all() else None
        except Exception:
            episode["q_failure"] = None


def recover(runtime):
    """Only reset_scene may request this original, measured contact retreat."""
    episode = getattr(runtime, "_contact_episode", None)
    if episode is None:
        return None
    harness = runtime.arm.harness
    if not episode["expected_paths"] or episode["q_failure"] is None:
        raise SkillError("contact recovery lacks retained contact identity or measured feedback")
    completed = False
    retreat_requested = False
    try:
        harness._contact_scope.value = (episode, False)
        _guard(runtime, episode)
        before = np.asarray(runtime.arm.get_state().q, dtype=float)
        if (before.shape != episode["q_failure"].shape or not np.isfinite(before).all()
                or np.max(np.abs(before - episode["q_failure"])) > .001):
            raise SafetyViolation("contact recovery feedback changed from the retained failure")
        evidence = wait_geometry(runtime, episode, recovery=True)
        _guard(runtime, episode)
        now = np.asarray(runtime.arm.get_state().q, dtype=float)
        if now.shape != before.shape or not np.isfinite(now).all() or np.max(np.abs(now - before)) > .001:
            raise SafetyViolation("contact recovery feedback moved while waiting for geometry")
        xy, radius, z_min = episode["cylinder"]
        harness.allow_grasp_descent(xy.copy(), radius_m=radius, z_min=z_min)
        retreat_requested = True
        if not runtime.arm.move_planned(episode["q_pre"], duration_s=float(
                runtime.cfg.grasp.get("descend_duration_s", 2.))):
            raise SkillError("contact recovery did not settle at its original pregrasp")
        _guard(runtime, episode)
        completed = True
        return {"retreated_to_original_pregrasp": True, "geometry": evidence}
    finally:
        harness.clear_grasp_exemption()
        finish(runtime, episode, completed=completed, capture_feedback=retreat_requested)
