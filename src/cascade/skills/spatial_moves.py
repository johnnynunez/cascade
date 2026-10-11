"""Runtime side of the natural-language motion tools: plan, vet, execute.

`relative_motion` interprets words; this module turns an interpretation
into motion on the arm the current call is bound to, under four rules that
the tests pin one by one:

1. Nothing bypasses the SafetyHarness. Every motion is a `SafeArm` call
   (`move_joints` through `carry_attachment.move`, or `move_cartesian`), so
   every streamed tick is `approve()`d exactly as for any other skill.

2. A refused move leaves the arm where it is. Before the first command the
   WHOLE route is vetted with the harness's own static API
   (`safety.trajectory.vet_segment` -> `SafetyHarness.vet_step`, the same
   check `move_planned` uses; for straight lines `vet_step` over the planned
   dense path). Without that preflight a doomed nudge would start and be
   cut off mid-way by `approve()`, leaving the arm somewhere nobody asked
   for. The preflight never authorizes anything: `begin_motion` still checks
   the watchdog and `approve()` still checks every tick.

3. "How much of it would be OK" is answered by VETTING, never by moving.
   When the full move is refused, a bisection over the fraction of the
   displacement (or angle) re-plans and re-vets candidate routes and
   reports the largest one that passed -- `feasible_distance_m` /
   `feasible_angle_deg`. Nothing is commanded while searching; the agent
   offers that number to the person and only a NEW call moves.

4. Two checks the harness cannot know about, applied to every sample of the
   planned route:
   - a HELD object hangs below the TCP, so its estimated bottom must keep
     the same table clearance the harness keeps for the TCP (the drop is
     the grasp's measured support offset, else a conservative default);
   - the wrist camera is mounted on TOP of the reBot gripper (TCP +z): its
     up axis may never go below horizontal (`relative_motion.camera_up_*`).

Straight-line profiles (`cartesian_relative_moves`) never fall back to joint
space: a straight line that cannot be planned or vetted is a refusal.

Every completed move is recorded (measured joints at the start and after
each segment) in a bounded per-arm history; `undo_move` retraces a record
backwards through its own vetted waypoints, and only from where that record
ended -- if anything else moved the arm since, the history is stale and is
cleared instead of being replayed from the wrong place.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..types import SafetyViolation, SkillError
from . import carry_attachment
from . import relative_motion as rm

#: deepest undo one call may ask for
MAX_UNDO_STEPS = 5
#: bisection rounds for the feasible fraction (1/64 resolution: 2.3 mm of a
#: 15 cm move, 0.5 deg of a 30 deg turn)
FRACTION_ITERATIONS = 6
#: joint-space sampling for the runtime's own path checks (held clearance,
#: camera up); the harness vets the executor's real ticks separately
_JOINT_SAMPLE_RAD = 0.02
#: IK residual absorbed by the camera-up check (about 0.06 deg)
_CAMERA_TOL = 1e-3


# ── settings ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    duration_s: float
    held_speed_scale: float
    held_drop_default_m: float
    history_len: int
    viewer_camera: str | None
    tool_up_axis: tuple | None
    tool_up_min_z: float
    max_object_move_m: float


def _positive(value, name: str, *, upper: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isfinite(v) or v <= 0 or (upper is not None and v > upper):
        raise SkillError(f"relative_motion.{name} must be finite and in (0, {upper or 'inf'}], got {value!r}")
    return v


def settings(rt) -> Settings:
    """`relative_motion:` from the config (demo.yaml, arm `overrides:` may
    change it per robot). Invalid values fail closed: a NaN duration or a
    non-finite camera axis would otherwise disable a check silently."""
    c = rt.cfg.get("relative_motion", None) if hasattr(rt.cfg, "get") else None
    c = c if c is not None else {}
    up = c.get("tool_up_axis", [0.0, 0.0, 1.0])
    if up is not None:
        try:
            up_v = np.asarray(up, dtype=float).reshape(3)
        except (TypeError, ValueError) as exc:
            raise SkillError(f"relative_motion.tool_up_axis must be 3 numbers or null, got {up!r}") from exc
        if not np.all(np.isfinite(up_v)) or np.linalg.norm(up_v) < 1e-6:
            raise SkillError(f"relative_motion.tool_up_axis must be a finite non-zero vector, got {up!r}")
        up = tuple(float(x) for x in up_v / np.linalg.norm(up_v))
    min_z = float(c.get("tool_up_min_z", 0.0))
    if not math.isfinite(min_z) or not -1.0 <= min_z <= 1.0:
        raise SkillError(f"relative_motion.tool_up_min_z must be in [-1, 1], got {min_z!r}")
    viewer = c.get("viewer_camera", None)
    return Settings(
        duration_s=_positive(c.get("duration_s", 1.0), "duration_s", upper=30.0),
        held_speed_scale=_positive(c.get("held_speed_scale", 0.5), "held_speed_scale", upper=1.0),
        held_drop_default_m=_positive(c.get("held_drop_default_m", 0.06), "held_drop_default_m", upper=0.3),
        history_len=int(_positive(c.get("history_len", 10), "history_len", upper=100)),
        viewer_camera=None if viewer in (None, "") else str(viewer),
        tool_up_axis=up,
        tool_up_min_z=min_z,
        max_object_move_m=_positive(c.get("max_object_move_m", 0.50), "max_object_move_m", upper=1.0),
    )


# ── per-arm state ───────────────────────────────────────────────────────────


def _state(rt) -> dict:
    """History + last request for the arm THIS call is bound to: on a rig two
    arms must never undo each other's moves."""
    store = rt.__dict__.setdefault("_spatial_state", {})
    key = rt.arm
    if key not in store:
        store[key] = {"history": rm.MoveHistory(settings(rt).history_len), "last": {}}
    return store[key]


def history(rt) -> rm.MoveHistory:
    return _state(rt)["history"]


def _kin(rt):
    kin = getattr(getattr(rt.arm, "harness", None), "kin", None)
    return kin if kin is not None else rt.kin


def _cartesian(rt) -> bool:
    flag = rt.cfg.arm.get("cartesian_relative_moves", False)
    return flag is True or str(flag).strip().lower() in ("1", "true", "yes", "on")


# ── frames from the rig ─────────────────────────────────────────────────────


def viewer_axes(rt) -> rm.ViewerAxes:
    """The on-screen frame: `relative_motion.viewer_camera` by stream name,
    else the first FIXED (eye-to-hand) camera of the rig, primary first.

    A wrist camera is never the viewer: its "right" turns with every move.
    An uncalibrated or drift-invalidated camera refuses (`cam_to_base`
    raises its calibration error) instead of guessing a transform.
    """
    s = settings(rt)
    cams = [(str(getattr(c.stream, "name", f"cam{i}")), c.extrinsics)
            for i, c in enumerate(getattr(getattr(rt, "watcher", None), "_cams", None) or [])]
    if not cams and getattr(rt, "extrinsics", None) is not None:
        cams = [("primary", rt.extrinsics)]
    if s.viewer_camera:
        match = [c for c in cams if c[0] == s.viewer_camera]
        if not match:
            raise SkillError(f"viewer frame unavailable: relative_motion.viewer_camera="
                             f"{s.viewer_camera!r} is not one of {[n for n, _ in cams]}")
        name, ext = match[0]
    else:
        fixed = [c for c in cams if getattr(c[1], "mode", None) == "eye_to_hand"]
        if not fixed:
            raise SkillError("viewer frame unavailable: the rig has no fixed (eye-to-hand) scene camera")
        name, ext = fixed[0]
    if getattr(ext, "mode", None) != "eye_to_hand":
        raise SkillError(f"viewer frame unavailable: camera {name!r} moves with the arm "
                         "(eye-in-hand); its 'right' would turn with every move")
    return rm.viewer_axes(ext.cam_to_base(), camera=name)


def _screen(rt, delta) -> str:
    try:
        axes = viewer_axes(rt)
    except SkillError as exc:
        return f"screen mapping unavailable ({exc})"
    return f"{rm.describe_on_screen(delta, axes)} (camera {axes.camera!r})"


# ── what the gripper carries ────────────────────────────────────────────────


@dataclass(frozen=True)
class Carried:
    label: str | None
    drop_m: float        # lowest carried point below the TCP (0 = empty gripper)
    source: str
    radius_m: float      # horizontal half-size of the jaws or the held object


def carried(rt, s: Settings) -> Carried:
    """Reconcile the held flag with the jaws (read-only), then describe what
    hangs below the TCP. Never commands the gripper."""
    try:
        rt._reconcile_held()
    except (SkillError, SafetyViolation):
        pass  # unknown jaw state: whatever the flag says stands
    half_jaw = float(getattr(rt, "_max_width", 0.09)) / 2.0
    label = rt.held_object
    if not label:
        return Carried(None, 0.0, "empty gripper", half_jaw)
    off = getattr(rt, "_held_support_offset_m", None)
    if off is not None and math.isfinite(float(off)) and 0.0 < float(off) < 0.3:
        drop, source = float(off), "measured at the grasp"
    else:
        drop, source = s.held_drop_default_m, "conservative default: its bottom was not measured"
    width = getattr(rt, "_held_width_m", None)
    radius = half_jaw
    if width is not None and math.isfinite(float(width)) and float(width) > 0:
        radius = max(half_jaw, float(width) / 2.0)
    return Carried(str(label), drop, source, radius)


# ── planning and vetting ────────────────────────────────────────────────────


class Refused(Exception):
    """The route failed a check BEFORE any command: nothing moved."""

    def __init__(self, source: str, reason: str, segment: int = 0):
        super().__init__(reason)
        self.source, self.reason, self.segment = source, reason, segment


@dataclass
class Segment:
    q_start: np.ndarray
    q_goal: np.ndarray
    T_goal: np.ndarray


@dataclass
class Route:
    segments: list
    cartesian: bool


def _joint_samples(q0, q1):
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    n = max(2, math.ceil(float(np.max(np.abs(q1 - q0))) / _JOINT_SAMPLE_RAD))
    return [q0 + (q1 - q0) * (k / n) for k in range(n + 1)]


def _path_checks(rt, kin, samples, load: Carried, s: Settings):
    """(source, reason) for the two checks the harness does not own, or None."""
    poses = [np.asarray(kin.fk(q), dtype=float) for q in samples]
    if s.tool_up_axis is not None:
        reason = rm.camera_up_violation([T[:3, :3] for T in poses], s.tool_up_axis,
                                        s.tool_up_min_z, tol=_CAMERA_TOL)
        if reason:
            return "wrist_camera", reason
    if load.label:
        limits = rt.arm.harness.limits
        floor = float(limits.table_z) + float(limits.table_clearance)
        bottoms = [float(T[2, 3]) - load.drop_m for T in poses]
        start, lowest = bottoms[0], min(bottoms)
        # Escape rule mirroring the harness's below-table rule: from a pose
        # whose load already sits under the clearance plane, only climbing
        # moves (never lower, ending higher) are allowed -- no sliding along
        # the table with the object dragging.
        escaping = start < floor and lowest >= start - 1e-6 and bottoms[-1] > start + 1e-3
        if lowest < floor - 1e-6 and not escaping:
            above = (lowest - float(limits.table_z)) * 100.0
            return "held_object_clearance", (
                f"holding {load.label!r}: its bottom (estimated {load.drop_m * 100:.1f} cm below the "
                f"gripper, {load.source}) would come to {above:.1f} cm above the table; it must stay "
                f"{float(limits.table_clearance) * 100:.1f} cm above it, the clearance the harness "
                "keeps for the gripper itself")
    return None


def plan_route(rt, q_start, goals, *, load: Carried, s: Settings, duration: float) -> Route:
    """Plan and vet every segment of `goals` from `q_start` BEFORE moving.

    `goals` is a list of ("pose", T_4x4) or ("joints", q). Raises `Refused`
    on the first failing segment; never commands anything.
    """
    from ..safety.trajectory import geometry_guard, vet_segment

    kin, harness = _kin(rt), rt.arm.harness
    cart = _cartesian(rt)
    if cart:
        if getattr(rt.arm, "motion_planner", None) is not None:
            raise Refused("configuration", "cartesian_relative_moves is on, but this arm's motion "
                          "planner owns its curves: a straight line cannot be streamed and is never "
                          "swapped for a joint-space move")
        if carry_attachment.active(rt) is not None:
            raise Refused("carry_attachment", "a straight-line move cannot carry the retained-"
                          "attachment guard; no joint-space substitute is taken")
    segments = []
    q = np.asarray(q_start, dtype=float).reshape(-1)
    with geometry_guard(harness):
        for i, (kind, target) in enumerate(goals):
            T_goal = np.asarray(target if kind == "pose" else kin.fk(target), dtype=float)
            if cart:
                from ..planning.cartesian import plan_cartesian_path

                try:
                    path = plan_cartesian_path(kin, q, T_goal)
                except SkillError as exc:
                    raise Refused("kinematics", f"no straight-line path: {exc}", i) from exc
                samples, q_goal = [q] + path, np.asarray(path[-1], dtype=float)
            else:
                if kind == "pose":
                    ik = kin.ik(T_goal, q)
                    if not ik.success:
                        raise Refused("kinematics", "the arm cannot reach that pose from here "
                                      f"(IK found no solution, residual {float(getattr(ik, 'error', float('nan'))):.3f})",
                                      i)
                    q_goal = np.asarray(ik.q, dtype=float).reshape(-1)[: q.size]
                else:
                    q_goal = np.asarray(target, dtype=float).reshape(-1)
                samples = _joint_samples(q, q_goal)
            found = _path_checks(rt, kin, samples, load, s)
            if found:
                raise Refused(found[0], found[1], i)
            if cart:
                for a, b in itertools.pairwise(samples):
                    reason = harness.vet_step(a, b, 0.0)  # geometry; the streamer stretches time
                    if reason:
                        raise Refused("safety_harness", reason, i)
            else:
                reason = vet_segment(harness, q, q_goal, duration)
                if reason:
                    raise Refused("safety_harness", reason, i)
            segments.append(Segment(q.copy(), q_goal, T_goal))
            q = q_goal
    return Route(segments, cart)


def feasible_fraction(try_plan) -> float | None:
    """Largest fraction in (0, 1) whose route VETS, by bisection; None when
    not even 1/64 does. `try_plan(f)` must only plan and vet."""
    lo, hi = 0.0, 1.0
    for _ in range(FRACTION_ITERATIONS):
        mid = (lo + hi) / 2.0
        try:
            try_plan(mid)
            lo = mid
        except (Refused, SkillError):
            hi = mid
    return lo if lo > 0 else None


def _duration(s: Settings, load: Carried) -> float:
    return s.duration_s / (s.held_speed_scale if load.label else 1.0)


def execute_route(rt, route: Route, duration: float, *, tool: str, words: str) -> list:
    """Run a vetted route through SafeArm; record it (or the completed part
    of it, if a later segment is interrupted) in the history."""
    waypoints = [np.asarray(rt.arm.get_state().q, dtype=float).copy()]
    kin = _kin(rt)
    try:
        for i, seg in enumerate(route.segments):
            if route.cartesian:
                ok = rt.arm.move_cartesian(seg.T_goal, duration_s=duration)
            else:
                ok = carry_attachment.move(rt, seg.q_goal, duration_s=duration)
            if not ok:
                raise SkillError(f"did not settle at the end of segment {i + 1}/{len(route.segments)}")
            waypoints.append(np.asarray(rt.arm.get_state().q, dtype=float).copy())
    finally:
        if len(waypoints) > 1:
            history(rt).push(rm.MoveRecord(tool=tool, words=words, waypoints=waypoints,
                                           tcp_end=np.asarray(kin.fk(waypoints[-1]), dtype=float)))
    return waypoints


# ── results ─────────────────────────────────────────────────────────────────

_SOURCES = {
    "safety_harness": "the safety harness",
    "kinematics": "the arm's reach",
    "held_object_clearance": "the held-object clearance check",
    "wrist_camera": "the wrist-camera rule",
    "configuration": "the arm configuration",
    "carry_attachment": "the carry-attachment guard",
}


def _vec(v) -> list:
    return [round(float(x), 4) for x in np.asarray(v, dtype=float)]


def success(rt, *, tool: str, words: str, frame, camera, delta, T0, load: Carried, route: Route,
            extra: dict | None = None) -> dict:
    T1 = _kin(rt).fk(rt.arm.get_state().q)
    out = {
        "moved": True,
        "interpretation": words,
        "frame": frame,
        "camera": camera,
        "displacement_base_m": _vec(delta),
        "as_seen_on_screen": _screen(rt, delta),
        "tcp_before": _vec(T0[:3, 3]),
        "tcp_after": _vec(T1[:3, 3]),
        "path": "straight line (cartesian)" if route.cartesian else "joint space",
        "holding": load.label,
        "speed": "slow (holding an object)" if load.label else "normal",
        "undo_available": len(history(rt)),
    }
    if load.label:
        out["held_object_bottom_below_gripper_m"] = round(load.drop_m, 3)
    out["summary"] = (f"Moved: {words}." + (f" Holding {load.label!r} the whole time." if load.label else ""))
    out.update(extra or {})
    return out


def refusal(rt, err: Refused, *, words: str, frame, delta, load: Carried, feasible: dict | None = None,
            retry_hint: str = "") -> dict:
    who = _SOURCES.get(err.source, err.source)
    out = {
        "ok": False,
        "moved": False,
        "arm_moved": False,
        "refused_by": err.source,
        "error": f"refused by {who}: {err.reason}. The arm did not move.",
        "interpretation": words,
        "frame": frame,
        "displacement_base_m": _vec(delta) if delta is not None else None,
        "holding": load.label,
        "undo_available": len(history(rt)),
    }
    if feasible:
        out.update(feasible)
        offer = ", ".join(f"{k}={v}" for k, v in feasible.items())
        out["next_action"] = (f"Tell the person the move was refused ({who}) and the arm did not move. "
                              f"The largest part of it that passes every safety check is {offer}: "
                              f"offer it ({retry_hint}) or ask for another direction.")
    else:
        out["next_action"] = (f"Tell the person the move was refused ({who}: {err.reason}) and the arm "
                              "did not move; ask for another direction or a smaller move.")
    return out


# ── tools ───────────────────────────────────────────────────────────────────


def _pose(rt):
    q0 = np.asarray(rt.arm.get_state().q, dtype=float).copy()
    return q0, np.asarray(_kin(rt).fk(q0), dtype=float)


def move_relative(rt, direction=None, distance_m=None, amount=None, frame=None, vector=None) -> dict:
    s = settings(rt)
    q0, T0 = _pose(rt)

    def tool_axes():
        return rm.tool_axes(T0[:3, :3], rt._tool_axis_order)

    def viewer():
        return viewer_axes(rt)

    prefix = ""
    if direction is not None and rm.normalize_direction(direction) == rm.AGAIN:
        if frame is not None or vector is not None:
            raise SkillError("'again' repeats the last move as it was: omit frame and vector "
                             "(amount or distance_m may change its size)")
        last = _state(rt)["last"].get("translation")
        if last is None:
            raise SkillError("nothing to repeat: no move_relative has run on this arm yet")
        req, last_d = last
        size = (rm.resolve_distance(distance_m, amount)[0]
                if (distance_m is not None or amount is not None) else last_d)
        if req["vector"] is not None:
            v = np.asarray(req["vector"], dtype=float)
            move = rm.resolve_translation(vector=v / np.linalg.norm(v) * size, frame=req["frame"],
                                          viewer=viewer, tool=tool_axes)
        else:
            move = rm.resolve_translation(req["direction"], frame=req["frame"], distance=size,
                                          viewer=viewer, tool=tool_axes)
        prefix = "again: "
    elif vector is not None:
        given = distance_m if distance_m is not None else amount
        move = rm.resolve_translation(direction, frame=frame, vector=vector, distance=given,
                                      viewer=viewer, tool=tool_axes)
    else:
        size = rm.resolve_distance(distance_m, amount)[0]
        move = rm.resolve_translation(direction, frame=frame, distance=size, viewer=viewer, tool=tool_axes)
    words = prefix + move.words
    load = carried(rt, s)
    duration = _duration(s, load)

    def attempt(f):
        goal = T0.copy()
        goal[:3, 3] = T0[:3, 3] + move.delta * f
        return plan_route(rt, q0, [("pose", goal)], load=load, s=s, duration=duration)

    try:
        route = attempt(1.0)
    except Refused as err:
        frac = feasible_fraction(attempt)
        feasible = None
        if frac is not None and move.distance * frac >= rm.MIN_STEP_M:
            feasible = {"feasible_distance_m": round(move.distance * frac, 3),
                        "feasible_fraction": round(frac, 3)}
        return refusal(rt, err, words=words, frame=move.frame, delta=move.delta, load=load,
                       feasible=feasible, retry_hint="call move_relative again with that distance_m")
    execute_route(rt, route, duration, tool="move_relative", words=words)
    _state(rt)["last"]["translation"] = (move.request(), move.distance)
    return success(rt, tool="move_relative", words=words, frame=move.frame, camera=move.camera,
                   delta=move.delta, T0=T0, load=load, route=route)
