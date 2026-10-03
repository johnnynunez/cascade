"""Retain one released-object withdrawal; only explicit reset may resume it.

No recovery opens jaws, resets props or enlarges contact geometry. The retained
scope ends at the original retreat endpoint, before the ordinary home planner.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from ..control.arm_base import PREFLIGHT_MAX_DRIFT_RAD
from ..control.simulation_motion import PhysicsClock, positive
from ..perception.freshness import capture_marker, read_frame_after
from ..perception.occupancy import OccupancyError
from ..types import SafetyViolation, SkillError
from .contact_episode import _backend, _identity


def _streams(runtime):
    watcher = getattr(runtime, "watcher", None)
    return tuple(cam.stream for cam in watcher._cams if cam.maps_depth) if watcher else (runtime.camera,)


def _jaws(snapshot, *, expected=None, require_open=False):
    if not isinstance(snapshot, dict) or type(snapshot.get("version")) is not int or snapshot["version"] != 1:
        raise SafetyViolation("release requires individual captured jaw feedback")
    if snapshot.get("names") != ["joint_left", "joint_right"]:
        raise SafetyViolation("release jaw identity changed")
    try:
        values = [np.asarray(snapshot[key]) for key in ("position_m", "lower_m", "upper_m")]
        if any(any(isinstance(value, (bool, np.bool_)) for value in snapshot[key])
               for key in ("position_m", "lower_m", "upper_m")):
            raise ValueError("boolean jaw feedback")
        if any(v.dtype.kind not in "fiu" for v in values):
            raise ValueError("jaw feedback must be numeric")
        q, lower, upper = (v.astype(float) for v in values)
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyViolation("invalid individual jaw feedback") from exc
    if (any(v.shape != (2,) or not np.isfinite(v).all() for v in (q, lower, upper))
            or np.any(upper <= lower) or np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6)):
        raise SafetyViolation("invalid individual jaw feedback")
    identity = (tuple(lower), tuple(upper))
    if expected is not None and identity != expected:
        raise SafetyViolation("release jaw limits changed")
    if require_open and not _both_open(q, identity):
        raise SafetyViolation("release requires both actual jaws fully open")
    return identity


def _both_open(positions, limits):
    lower, upper = (np.asarray(v, float) for v in limits)
    return bool(np.all((np.asarray(positions)-lower)/(upper-lower) >= .98))


def begin(runtime, q_retreat, duration_s):
    harness = runtime.arm.harness
    occupancy = getattr(harness, "occupancy", None)
    if not getattr(occupancy, "tracks_payload", False):
        return None
    harness.check_contact_episode()
    cylinder = harness._grasp_exempt
    if cylinder is None or not runtime.held_object:
        raise SkillError("release has no original object/contact geometry")
    generation = harness._halt_generation
    clock = runtime.arm.raw.validate_simulation_clock()
    validator = PhysicsClock(tuple(clock["source"]), clock["robot_id"])
    validator.observe(clock)
    state = _backend(runtime.arm).get_state(timeout_s=1.)
    # The state already read for release must still carry the original NV
    # attachment. Never authorize opening from a lagging map's old paths.
    from .carry_attachment import observe
    observe(runtime, state)
    validator.observe(state.physics_clock)
    harness._check_halt_generation(generation)
    if harness.estopped:
        raise SafetyViolation("e-stop latched during release preparation")
    jaw_identity = _jaws(state.gripper_joints)
    if type(getattr(_backend(runtime.arm), "_acknowledged_joint_targets", None)) is not int:
        raise SkillError("release requires acknowledged-target provenance")
    q, target = np.asarray(state.q, float), np.asarray(q_retreat, float)
    if q.shape != target.shape or q.shape != (runtime.arm.raw.n_joints,) or not np.isfinite([q,target]).all():
        raise SkillError("release requires finite original joint geometry")
    backend = _backend(runtime.arm)
    n_joints = backend.n_joints
    signs = np.asarray(getattr(backend, "_signs", None))
    if (type(n_joints) is not int or n_joints <= 0 or signs.shape != (n_joints,)
            or signs.dtype.kind not in "fiu" or not np.isin(signs, [-1, 1]).all()):
        raise SafetyViolation("release requires a fixed exact-DOF joint convention")
    streams = _streams(runtime)
    with occupancy._refresh_lock:
        paths = tuple(occupancy._contact_paths or ())
        if (not streams or len(paths) != 1 or occupancy._payload_epoch != clock["epoch"]
                or occupancy._payload_identity != (tuple(clock["source"]), clock["robot_id"], "physics_loop_monotonic")):
            raise SkillError("release requires the same observed attachment and producer epoch")
        floors = dict(occupancy._prop_history_floor)
        attachment_stamp = occupancy._latest_contact_stamp
        floors[paths[0]] = max(floors.get(paths[0], -np.inf), attachment_stamp)
        scene_generation = occupancy.scene_reset_generation
        capture_keys = set(occupancy._integrated_captures)
        if len(capture_keys) != len(streams):
            raise SkillError("release requires every configured map-depth source")
    episode = {"arm": runtime.arm, "raw": runtime.arm.raw, "backend": _backend(runtime.arm),
        "harness": harness, "occupancy": occupancy,
        "identity": _identity(runtime), "clock": clock, "clock_validator": validator,
        "halt_generation": generation, "scene_generation": scene_generation,
        "q_release": q.copy(), "q_retreat": target.copy(), "q_failure": None,
        "n_joints": n_joints, "joint_signs": signs.copy(),
        "config_joint_signs": copy.deepcopy(runtime.cfg.arm.get("joint_signs")),
        "duration_s": float(duration_s), "open_position": runtime._grip_open,
        "jaw_identity": jaw_identity, "feedback_q": q.copy(),
        "feedback_jaws": np.asarray(state.gripper_joints["position_m"], float).copy(),
        "object": runtime.held_object, "paths": paths,
        "cylinder": (cylinder[0].copy(), cylinder[1], cylinder[2]),
        "attachment_stamp": attachment_stamp,
        "streams": streams, "capture_keys": capture_keys, "stream_capture_keys": {}, "prop_floors": floors, "open_acknowledged": False,
        "released": False, "barrier": None, "withdrawal_counter": None}
    episode["opening_attachment_previous"] = _opening_observation(episode, state)
    runtime._release_episode = episode
    harness._pending_release_episode = episode
    return episode


def _same_joint_binding(runtime, episode):
    try:
        backend = episode["backend"]
        signs = np.asarray(getattr(backend, "_signs", None))
        return (type(getattr(backend, "n_joints", None)) is int
                and backend.n_joints == episode["n_joints"]
                and signs.dtype.kind in "fiu"
                and np.array_equal(signs, episode["joint_signs"])
                and np.array_equal(runtime.cfg.arm.get("joint_signs"), episode["config_joint_signs"]))
    except (AttributeError, TypeError, ValueError):
        return False


def _guard(runtime, episode):
    arm, harness = runtime.arm, runtime.arm.harness
    if (arm is not episode["arm"] or arm.raw is not episode["raw"]
            or harness is not episode["harness"] or harness.occupancy is not episode["occupancy"]
            or _backend(arm) is not episode["backend"] or _identity(runtime) != episode["identity"]
            or not _same_joint_binding(runtime, episode)
            or harness._pending_release_episode is not episode
            or harness.occupancy.scene_reset_generation != episode["scene_generation"]
            or _streams(runtime) != episode["streams"]):
        raise SafetyViolation("release recovery arm, scene or camera identity changed")
    harness._check_halt_generation(episode["halt_generation"])
    if harness.estopped or harness._halt is not None:
        raise SafetyViolation("release recovery halted")


def _observe_feedback(episode, state, *, require_open=True):
    _jaws(state.gripper_joints, expected=episode["jaw_identity"], require_open=require_open)
    q = np.asarray(state.q)
    if q.shape != episode["q_release"].shape or q.dtype.kind not in "fiu" or not np.isfinite(q).all():
        raise SafetyViolation("release recovery has no finite measured feedback")
    positions = np.asarray(state.gripper_joints["position_m"], float)
    fresh = episode["clock_validator"].observe(state.physics_clock)
    if not fresh and (not np.array_equal(q, episode["feedback_q"])
                      or not np.array_equal(positions, episode["feedback_jaws"])):
        raise SafetyViolation("release joint feedback changed without a new physics step")
    episode["feedback_q"], episode["feedback_jaws"] = q.copy(), positions.copy()
    return q.astype(float)


def _state_feedback(runtime, episode, *, require_open=True, deadline=None):
    _guard(runtime, episode)
    remaining = 1. if deadline is None else deadline-time.monotonic()
    if remaining <= 0:
        raise SkillError("post-release feedback deadline expired")
    try:
        state = _backend(runtime.arm).get_state(timeout_s=min(1., remaining))
    except Exception as exc:
        raise SafetyViolation(f"release feedback unavailable: {exc}") from exc
    _guard(runtime, episode)
    if deadline is not None and time.monotonic() >= deadline:
        raise SkillError("post-release feedback deadline expired")
    try:
        return _observe_feedback(episode, state, require_open=require_open), state
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SafetyViolation("invalid release feedback: " + str(exc)) from exc


def _state(runtime, episode, *, require_open=True, deadline=None):
    return _state_feedback(runtime, episode, require_open=require_open, deadline=deadline)[0]


def _opening_observation(episode, state):
    """Validate an existing atomic reply; no sensor or mapper request.

    Empty paths mean no observed bilateral attachment, not zero unilateral
    contact. This never substitutes for the subsequent camera/map barrier.
    """
    try:
        return _validated_opening_observation(episode, state)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SafetyViolation("invalid atomic release opening feedback: " + str(exc)) from exc


def _validated_opening_observation(episode, state):
    q, dq = np.asarray(state.q), np.asarray(state.dq)
    if (q.shape != (episode["n_joints"],) or q.dtype.kind not in "fiu" or not np.isfinite(q).all()
            or dq.shape != q.shape or dq.dtype.kind not in "fiu" or not np.isfinite(dq).all()):
        raise SafetyViolation("release opening requires finite, exact-DOF q and dq")
    clock = episode["clock_validator"]
    a = state.attachment
    if (not isinstance(a, dict) or type(a.get("version")) is not int or a["version"] != 1
            or a.get("backend") != "isaac" or a.get("source") != clock.source
            or a.get("robot_id") != clock.robot_id or a.get("producer_epoch") != episode["clock"]["epoch"]
            or type(a.get("physics_step")) is not int or a["physics_step"] != clock.step
            or type(a.get("sim_time")) not in (int, float) or a["sim_time"] != clock.time
            or a.get("joint_convention") != "asset"
            or a.get("channel") != "completed_update_bilateral_contact"
            or a.get("sensor_channel") != {"physx": "physx_gpu_contact_tensor",
                                            "newton": "newton_mjwarp_contact_force"}.get(state.physics_clock["engine"])
            or a.get("tracking") is not True or a.get("error") is not None):
        raise SafetyViolation("release opening attachment is not valid atomic feedback")
    aq = np.asarray(a.get("q"))
    if (aq.shape != q.shape or aq.dtype.kind not in "fiu" or not np.isfinite(aq).all()
            or not np.array_equal(aq * episode["joint_signs"], q)
            or a.get("gripper_joints") != state.gripper_joints):
        raise SafetyViolation("release opening attachment articulation differs from feedback")
    _jaws(state.gripper_joints, expected=episode["jaw_identity"])
    paths = a.get("paths")
    if (not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths)
            or len(set(paths)) != len(paths) or (paths and tuple(paths) != episode["paths"])):
        raise SafetyViolation("release opening attachment paths are unknown or changed")
    previous = episode.get("opening_attachment_previous")
    current = (clock.step, tuple(paths), tuple(dq))
    if previous is not None and previous[0] == clock.step and previous != current:
        raise SafetyViolation("release atomic opening feedback changed without a new physics step")
    return current


def open_hand(runtime, episode, *, halt_generation=None):
    if episode is None:
        runtime.arm.set_gripper(runtime._grip_open, effort=.6,
            **({} if halt_generation is None else {'_halt_generation': halt_generation}))
        return
    _guard(runtime, episode)
    harness = runtime.arm.harness
    harness._release_scope.value = (episode, "open")
    try:
        runtime.arm.set_gripper(episode["open_position"], effort=.6,
                               _halt_generation=episode["halt_generation"])
        _guard(runtime, episode)
        episode["open_acknowledged"] = True
    finally:
        harness._release_scope.value = None


def wait_planned_open(runtime, *, timeout_s, halt_generation):
    """Observe open jaws and contact stability before an Isaac planner snapshot.

    Opening and the existing physical hold share one deadline. This grants no
    release/geometry authority and never changes the planned endpoint or veto.
    """
    from ..planning.runtime import wait_for_contact_stability
    if type(timeout_s) not in (int, float) or not np.isfinite(timeout_s) or timeout_s < 0:
        raise SafetyViolation("release opening timeout must be finite and nonnegative")
    deadline = time.monotonic() + timeout_s
    cfg = runtime.cfg.arm
    source = (str(cfg.get('bridge_host', '127.0.0.1')), int(cfg.get('bridge_port', 8611)))
    robot_id = cfg.bridge_robot_id
    rpc_timeout = positive(cfg.get('motion_rpc_timeout_s', 1.), 'release feedback RPC timeout')
    clock = PhysicsClock(source, robot_id)
    limits, previous = None, None

    def guard():
        runtime.arm.harness.check_stream_start(halt_generation=halt_generation)
        if time.monotonic() >= deadline:
            raise SkillError("post-release opening/stability deadline expired")

    def observe(state, *, require_open=False):
        nonlocal limits, previous
        try:
            limits = _jaws(state.gripper_joints, expected=limits, require_open=require_open)
            q = np.asarray(state.q, float)
            positions = np.asarray(state.gripper_joints['position_m'], float)
            if (q.ndim != 1 or not len(q) or not np.isfinite(q).all()
                    or (previous is not None and q.shape != previous[0].shape)):
                raise SafetyViolation("invalid post-release joint feedback")
            fresh = clock.observe(state.physics_clock)
            if not fresh and previous is not None and (
                    not np.array_equal(q, previous[0]) or not np.array_equal(positions, previous[1])):
                raise SafetyViolation("release joint feedback changed without a new physics step")
            previous = q.copy(), positions.copy()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise SafetyViolation("invalid release feedback: " + str(exc)) from exc

    while True:
        guard()
        try:
            state = runtime.arm.get_state(timeout_s=min(rpc_timeout, deadline-time.monotonic()))
        except Exception as exc:
            raise SafetyViolation("release feedback unavailable: " + str(exc)) from exc
        guard()
        observe(state)
        if _both_open(previous[1], limits):
            # Preserve the opening clock/limits and require both jaws to stay
            # open on every subsequent sample of the shared stability helper.
            try:
                return wait_for_contact_stability(runtime.arm, state, source=source,
                    robot_id=robot_id, timeout_s=deadline-time.monotonic(), rpc_timeout_s=rpc_timeout,
                    check=guard, observe=lambda state: observe(state, require_open=True))
            except (SkillError, SafetyViolation):
                raise
            except Exception as exc:
                raise SafetyViolation("release feedback unavailable: " + str(exc)) from exc
        time.sleep(min(.05, max(0., deadline-time.monotonic())))


def wait_open(runtime, episode, *, timeout_s):
    """Observe open jaws and stable measured q within one opening deadline.

    Candidate stability windows may restart while opening is pending. Once
    open and free of bilateral attachment, regressing either is terminal.
    This establishes no geometry authority and sends no actuator command.
    """
    if not episode["open_acknowledged"] or episode["released"]:
        raise SafetyViolation("release opening is not pending")
    if type(timeout_s) not in (int, float) or not np.isfinite(timeout_s) or timeout_s < 0:
        raise SafetyViolation("release opening timeout must be finite and nonnegative")
    deadline = time.monotonic() + timeout_s
    hold_s = positive(getattr(episode["backend"], "settle_hold_s", None), "release opening physical hold")
    eligible = False
    anchor_q = anchor_time = None
    samples = 0
    while True:
        clock = episode["clock_validator"]
        previous_step = clock.step
        q, state = _state_feedback(runtime, episode, require_open=False, deadline=deadline)
        observation = _opening_observation(episode, state)
        episode["opening_attachment_previous"] = observation
        _guard(runtime, episode)
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise SkillError("release opening stability deadline expired")
        opened = _both_open(episode["feedback_jaws"], episode["jaw_identity"])
        unattached = not observation[1]
        if eligible and (not opened or not unattached):
            raise SafetyViolation("release opening or bilateral attachment regressed")
        if opened and unattached:
            eligible = True  # Never reset this latch with the candidate window.
            if clock.step != previous_step:
                if (anchor_q is None or clock.delta > hold_s+1e-9
                        or np.max(abs(q-anchor_q)) > PREFLIGHT_MAX_DRIFT_RAD):
                    anchor_q, anchor_time, samples = q.copy(), clock.time, 1
                else:
                    samples += 1
                if samples >= 3 and clock.time-anchor_time >= hold_s-1e-9:
                    return
        time.sleep(min(.05, remaining))


def wait_geometry(runtime, episode, *, reference=None):
    if episode is None:
        return None
    if not episode["open_acknowledged"]:
        raise SafetyViolation("release opening was not acknowledged; state remains ambiguous")
    deadline = time.monotonic() + 5.
    def guard():
        _guard(runtime, episode)
        if time.monotonic() >= deadline:
            raise SkillError("post-release geometry deadline expired")
    before = _state(runtime, episode, deadline=deadline)
    guard()
    if reference is not None and np.max(abs(before-reference)) > .001:
        raise SafetyViolation("release feedback changed from retained failure")
    floors = []
    for stream in episode["streams"]:
        after = None
        while True:
            guard()
            try:
                frame = read_frame_after(stream, after=after, timeout_s=deadline-time.monotonic())
                marker = capture_marker(frame)
            except (TimeoutError, ValueError) as exc:
                raise SkillError(f"post-release capture unavailable: {exc}") from exc
            guard()
            snapshot = (frame.capture or {}).get("proprioception", {})
            clock = episode["clock"]
            paths = frame.capture.get("contact_paths")
            if (marker.get("backend") != "isaac" or tuple(marker["source"]) != tuple(clock["source"])
                    or marker["robot_id"] != clock["robot_id"] or snapshot.get("producer_epoch") != clock["epoch"]
                    or marker["t"] <= episode["attachment_stamp"]
                    or not isinstance(paths, list) or (paths != [] and paths != list(episode["paths"]))):
                raise SafetyViolation("post-release capture identity/contact changed")
            key = runtime.arm.harness.occupancy._capture_key(marker)
            previous = episode["stream_capture_keys"].get(id(stream))
            if key not in episode["capture_keys"] or (previous is not None and previous != key):
                raise SafetyViolation("post-release stream source identity changed")
            if previous is None and episode["released"]:
                raise SafetyViolation("post-release stream identity was not retained")
            # Bind the first packet, including an intermediate opening packet.
            # A later camera/epoch replacement cannot become this stream's floor.
            episode["stream_capture_keys"][id(stream)] = key
            if after is not None and marker["t"] <= capture_marker(after)["t"]:
                raise SafetyViolation("post-release capture did not advance")
            captured_jaws = snapshot.get("gripper_joints")
            _jaws(captured_jaws, expected=episode["jaw_identity"])
            if _both_open(captured_jaws["position_m"], episode["jaw_identity"]) and paths == []:
                floors.append(frame)
                break
            if episode["released"]:
                raise SafetyViolation("post-release captured opening/contact regressed")
            # Delivery can lag actual opening. Require a genuinely newer packet
            # under this same deadline; no intermediate packet is a map floor.
            after = frame
    if {runtime.arm.harness.occupancy._capture_key(capture_marker(f)) for f in floors} != episode["capture_keys"]:
        raise SafetyViolation("post-release capture camera set changed")
    # Eligibility is established by actual open fingers AND empty contacts,
    # before waiting for the mapper; held_object=None alone never authorizes it.
    guard()
    episode["released"] = True
    try:
        evidence = runtime.arm.harness.occupancy.wait_released_ready(floors,
            deadline=deadline, guard=guard, producer_epoch=episode["clock"]["epoch"],
            prop_floors=episode["prop_floors"])
    except OccupancyError as exc:
        raise SkillError(f"post-release geometry unavailable: {exc}") from exc
    now = _state(runtime, episode, deadline=deadline)
    if (np.max(abs(now-before)) > .001
            or (reference is not None and np.max(abs(now-reference)) > .001)):
        raise SafetyViolation("release feedback moved while waiting for geometry")
    guard()
    episode["barrier"] = evidence
    episode["withdrawal_start"] = now.copy() if reference is None else reference.copy()
    if episode["q_failure"] is None:
        episode["q_failure"] = now.copy()
    return evidence


def withdraw(runtime, episode):
    """Exactly one original endpoint/duration, with all ordinary collision gates."""
    _guard(runtime, episode)
    if not episode["released"] or episode["barrier"] is None:
        raise SafetyViolation("release withdrawal requires fresh confirmed geometry")
    harness = runtime.arm.harness
    episode["withdrawal_counter"] = episode["backend"]._acknowledged_joint_targets
    harness._release_scope.value = (episode, "retreat")
    def preflight(start, duration):
        from ..safety.trajectory import PLAN_BUDGET_S, geometry_guard, vet_segment
        from ..control.motion_profile import resolve_motion_rate
        _guard(runtime, episode)
        if np.max(abs(start-episode["withdrawal_start"])) > .001:
            raise SafetyViolation("release feedback changed before withdrawal stream")
        deadline = time.monotonic() + PLAN_BUDGET_S
        with geometry_guard(harness, deadline=deadline):
            reason = vet_segment(harness, start, episode["q_retreat"], duration,
                deadline=deadline, stretch=False, rate_hz=resolve_motion_rate(runtime.arm.raw))
        _guard(runtime, episode)
        if reason:
            raise SafetyViolation(f"original release withdrawal became unsafe: {reason}")
    def trajectory_preflight(start, profile, check):
        _guard(runtime, episode)
        if np.max(abs(start-episode["withdrawal_start"])) > .001:
            raise SafetyViolation("release feedback changed before withdrawal stream")
        # SafeArm checks every edge against the current harness/map. The
        # episode's target/duration and fresh geometry authority stay bound;
        # no alternate release, retry or clearing of retained failure occurs.
        check()
        _guard(runtime, episode)
    try:
        return runtime.arm.move_joints(episode["q_retreat"], duration_s=episode["duration_s"],
                    _halt_generation=episode["halt_generation"], _preflight=preflight,
                    _trajectory_preflight=trajectory_preflight)
    finally:
        harness._release_scope.value = None


def finish(runtime, episode, *, completed, capture_feedback=True):
    if episode is None:
        return
    harness = episode["harness"]
    harness._release_scope.value = (episode, "cleanup")
    try:
        harness.clear_grasp_exemption()
    finally:
        harness._release_scope.value = None
    if completed:
        runtime._release_episode = None
        harness._pending_release_episode = None
        return
    episode["barrier"] = None
    if not capture_feedback:
        return
    baseline = episode["withdrawal_counter"]
    if baseline is not None and episode["backend"]._acknowledged_joint_targets <= baseline:
        return  # No acknowledged withdrawal target: never adopt uncommanded drift.
    episode["q_failure"] = None
    try:
        _guard(runtime, episode)
        state = episode["backend"].get_state(timeout_s=1.)
        _guard(runtime, episode)
        q = _observe_feedback(episode, state, require_open=False)
        episode["q_failure"] = q.copy()
    except Exception:
        pass  # Missing measured feedback blocks recovery; never substitute a target.


def recover(runtime):
    episode = runtime._release_episode
    if not episode["released"] or episode["q_failure"] is None:
        raise SkillError("release recovery lacks confirmed release or measured failure feedback")
    completed = False
    retreat_requested = False
    harness = episode["harness"]
    try:
        before = _state(runtime, episode)
        if np.max(abs(before-episode["q_failure"])) > .001:
            raise SafetyViolation("release recovery feedback changed from retained failure")
        evidence = wait_geometry(runtime, episode, reference=episode["q_failure"])
        xy, radius, z_min = episode["cylinder"]
        harness._release_scope.value = (episode, "retreat")
        try:
            harness.allow_grasp_descent(xy.copy(), radius_m=radius, z_min=z_min)
        finally:
            harness._release_scope.value = None
        retreat_requested = True
        if not withdraw(runtime, episode):
            raise SkillError("release recovery did not settle at its original retreat")
        _guard(runtime, episode)
        completed = True
        return {"retreated_to_original_release_target": True, "object": episode["object"],
                "geometry": evidence}
    finally:
        finish(runtime, episode, completed=completed, capture_feedback=retreat_requested)
