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
import logging
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..safety.harness import table_frame_zone_text
from ..types import SafetyViolation, SkillError
from . import arm_frame, carry_attachment
from . import relative_motion as rm

logger = logging.getLogger(__name__)

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


class CameraDims(NamedTuple):
    """The wrist camera box in the TCP frame (m): it reaches `behind_m` back
    from the fingertip plane and `protrusion_m` up from the finger axis,
    +-`half_width_m` across the jaws. `front_m` / `bottom_m` cut its front and
    bottom faces (a calipered camera, spec 2026-10-16 nvblox §3.4); 0 = today's
    slab, down to the fingertip plane and the finger axis."""

    protrusion_m: float
    behind_m: float
    half_width_m: float
    front_m: float = 0.0
    bottom_m: float = 0.0


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
    #: the wrist camera box (CameraDims, m); None = no camera volume check
    wrist_camera: CameraDims | None
    camera_occupancy: bool


def _positive(value, name: str, *, upper: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isfinite(v) or v <= 0 or (upper is not None and v > upper):
        raise SkillError(f"relative_motion.{name} must be finite and in (0, {upper or 'inf'}], got {value!r}")
    return v


def _tool_up(c):
    """`relative_motion.tool_up_axis` as a unit tuple, or None (no camera)."""
    up = c.get("tool_up_axis", [0.0, 0.0, 1.0])
    if up is not None:
        try:
            up_v = np.asarray(up, dtype=float).reshape(3)
        except (TypeError, ValueError) as exc:
            raise SkillError(f"relative_motion.tool_up_axis must be 3 numbers or null, got {up!r}") from exc
        if not np.all(np.isfinite(up_v)) or np.linalg.norm(up_v) < 1e-6:
            raise SkillError(f"relative_motion.tool_up_axis must be a finite non-zero vector, got {up!r}")
        up = tuple(float(x) for x in up_v / np.linalg.norm(up_v))
    return up


def wrist_camera_dims(c) -> tuple:
    """(CameraDims or None, camera-up axis or None) from a `relative_motion:`
    block, parsed exactly as `settings()` does. The harness samples the SAME
    box for its occupancy check (safety/tool_volume.py), so the two layers
    can never disagree."""
    c = c if c is not None else {}
    up = _tool_up(c)
    dims, _ = _camera_settings(c, up)
    return dims, up


def settings(rt) -> Settings:
    """`relative_motion:` from the config (demo.yaml, arm `overrides:` may
    change it per robot). Invalid values fail closed: a NaN duration or a
    non-finite camera axis would otherwise disable a check silently."""
    c = rt.cfg.get("relative_motion", None) if hasattr(rt.cfg, "get") else None
    c = c if c is not None else {}
    up = _tool_up(c)
    min_z = float(c.get("tool_up_min_z", 0.0))
    if not math.isfinite(min_z) or not -1.0 <= min_z <= 1.0:
        raise SkillError(f"relative_motion.tool_up_min_z must be in [-1, 1], got {min_z!r}")
    viewer = c.get("viewer_camera", None)
    camera, camera_occupancy = _camera_settings(c, up)
    return Settings(
        duration_s=_positive(c.get("duration_s", 1.0), "duration_s", upper=30.0),
        held_speed_scale=_positive(c.get("held_speed_scale", 0.5), "held_speed_scale", upper=1.0),
        held_drop_default_m=_positive(c.get("held_drop_default_m", 0.06), "held_drop_default_m", upper=0.3),
        history_len=int(_positive(c.get("history_len", 10), "history_len", upper=100)),
        viewer_camera=None if viewer in (None, "") else str(viewer),
        tool_up_axis=up,
        tool_up_min_z=min_z,
        max_object_move_m=_positive(c.get("max_object_move_m", 0.50), "max_object_move_m", upper=1.0),
        wrist_camera=camera,
        camera_occupancy=camera_occupancy,
    )


_ABSENT = object()
#: the reBot camera box when a config has no `wrist_camera` key at all: the
#: padded D455F as measured on the rig (configs/demo.yaml has the numbers).
#: `front_m` / `bottom_m` (0 = the slab reaches the fingertip plane and the
#: finger axis) cut the box once the camera is calipered (spec 2026-10-16
#: nvblox §3.4, safety/tool_volume.camera_faces_from_calipers)
DEFAULT_WRIST_CAMERA = {"protrusion_m": 0.08, "behind_tip_m": 0.14, "half_width_m": 0.075,
                        "front_m": 0.0, "bottom_m": 0.0}


def _camera_settings(c, up) -> tuple:
    """(CameraDims or None, check_occupancy). An explicit null disables the
    volume check; a MISSING key falls back to the reBot defaults (a config
    written before the camera was modelled must not silently lose the check).
    Bad sizes raise SkillError, refusing every move until fixed."""
    raw = c.get("wrist_camera", _ABSENT)
    if up is None or raw is None:
        return None, False
    if raw is _ABSENT:
        raw = DEFAULT_WRIST_CAMERA
    keys = set(raw.as_dict() if hasattr(raw, "as_dict") else raw)
    unknown = sorted(keys - set(DEFAULT_WRIST_CAMERA) - {"check_occupancy"})
    if unknown:
        raise SkillError(f"relative_motion.wrist_camera has unknown key(s) {unknown}; expected "
                         f"{sorted(DEFAULT_WRIST_CAMERA)} and check_occupancy")
    get = raw.get
    dims = CameraDims(*(get(key, default) for key, default in (
        ("protrusion_m", DEFAULT_WRIST_CAMERA["protrusion_m"]),
        ("behind_tip_m", DEFAULT_WRIST_CAMERA["behind_tip_m"]),
        ("half_width_m", DEFAULT_WRIST_CAMERA["half_width_m"]),
        ("front_m", DEFAULT_WRIST_CAMERA["front_m"]),
        ("bottom_m", DEFAULT_WRIST_CAMERA["bottom_m"]))))
    rm.camera_box(np.eye(4), [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], protrusion_m=dims.protrusion_m,
                  behind_m=dims.behind_m, half_width_m=dims.half_width_m, front_m=dims.front_m,
                  bottom_m=dims.bottom_m)  # validates, raises on bad sizes
    occ = get("check_occupancy", False)
    occ = occ is True or str(occ).strip().lower() in ("1", "true", "yes", "on")
    return CameraDims(*(float(d) for d in dims)), occ


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
    try:
        T = ext.cam_to_base()
    except SkillError as exc:  # missing/rejected record or drift-invalidated at runtime
        raise SkillError(f"viewer frame unavailable: camera {name!r} is not calibrated ({exc})") from exc
    # The camera's pose is the table's (cam -> W); the moves are the running
    # arm's (cam -> B): on a base_pose arm "right on the screen" is the
    # camera's right seen from that arm's base.
    return rm.viewer_axes(arm_frame.cam_to_arm(_world_from_base(rt), T), camera=name)


def _world_from_base(rt):
    """The running arm's base_pose (table <- base) or None: no base_pose, or
    a stand-in runtime without one (nothing converts)."""
    T = getattr(rt, "T_world_base", None)
    return T if isinstance(T, np.ndarray) else None


def _tcp_in_world(rt, T_tcp) -> list:
    """A TCP pose's position, reported in the table frame (agent coordinates)."""
    return _vec(arm_frame.to_world(_world_from_base(rt), np.asarray(T_tcp, dtype=float)[:3, 3]))


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
    hangs below the TCP. Never commands the gripper.

    On a rig the hold belongs to the arm that grasped it (spec D9): while
    another arm holds the object, this arm's jaws are empty and it moves
    and reports as such (no phantom load, no false "holding")."""
    try:
        rt._reconcile_held()
    except (SkillError, SafetyViolation):
        pass  # unknown jaw state: whatever the flag says stands
    half_jaw = float(getattr(rt, "_max_width", 0.09)) / 2.0
    label = rt.held_object
    held_by_another_arm = getattr(rt, "_held_by_another_arm", None)    # absent on runtime doubles
    if not label or (callable(held_by_another_arm) and held_by_another_arm() is not None):
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


def _camera_volume(rt, poses, s: Settings):
    """Why the wrist camera's BOX would hit the table, a keep-out zone or
    (opt-in) an occupancy-map obstacle along `poses`, or None.

    The orientation rule keeps the camera facing up; this catches what it
    cannot: rolled or tilted, the camera's side (it is ~8 cm tall and sits
    behind the fingertips) can reach the table or a fixture while the TCP --
    the only point the harness checks -- is fine.
    """
    cam = s.wrist_camera
    approach_tcp = np.eye(3)[rm._TOOL_COLUMNS[rt._tool_axis_order][0]]
    boxes = [rm.camera_box(T, approach_tcp, s.tool_up_axis, protrusion_m=cam.protrusion_m,
                           behind_m=cam.behind_m, half_width_m=cam.half_width_m, front_m=cam.front_m,
                           bottom_m=cam.bottom_m) for T in poses]
    harness = rt.arm.harness
    limits = harness.limits
    what = f"the wrist camera (it sticks out {cam.protrusion_m * 100:.1f} cm above the gripper)"
    table_z, clear = float(limits.table_z), float(limits.table_clearance)
    floor = table_z + clear
    lows = [b.min_z() for b in boxes]
    start, lowest = lows[0], min(lows)
    # Escape rule, as for the TCP and a held object: from a pose whose camera
    # is already under the clearance plane, only climbing moves are allowed.
    escaping = start < floor and lowest >= start - 1e-6 and lows[-1] > start + 1e-3
    if lowest < floor - 1e-6 and not escaping:
        return "wrist_camera", (
            f"{what} would come to {(lowest - table_z) * 100:.1f} cm above the table; it must stay "
            f"{clear * 100:.1f} cm above it, the clearance the harness keeps for the gripper tip")
    for kmin, kmax in limits.keep_out:
        hits = [i for i, b in enumerate(boxes) if rm.box_overlaps_aabb(b.center, b.axes, b.half, kmin, kmax)]
        if hits:
            # no escape rule, like the harness's own keep-out check for the TCP
            where = "is already inside" if hits[0] == 0 else "would enter"
            # The zone is this arm's (its base frame); the agent reads
            # table-frame numbers (T3 re-review): the bounds of its lifted
            # corners on a base_pose, with its base-frame corners when the
            # mount rotates it (#407 review follow-up 3).
            T = _world_from_base(rt)
            zone = (f"{np.round(kmin, 3).tolist()}..{np.round(kmax, 3).tolist()}" if T is None
                    else table_frame_zone_text(T, kmin, kmax))
            return "wrist_camera", f"{what} {where} keep-out zone {zone}"
    if s.camera_occupancy and getattr(harness, "occupancy", None) is not None:
        for b in boxes:
            reason = harness._arm_occupancy_violation(b.points(0.02), None)
            if reason:
                return "wrist_camera", f"{what}: {reason}"
    return None


def _path_checks(rt, kin, samples, load: Carried, s: Settings):
    """(source, reason) for the checks the harness does not own, or None."""
    poses = [np.asarray(kin.fk(q), dtype=float) for q in samples]
    if s.tool_up_axis is not None:
        reason = rm.camera_up_violation([T[:3, :3] for T in poses], s.tool_up_axis,
                                        s.tool_up_min_z, tol=_CAMERA_TOL)
        if reason:
            return "wrist_camera", reason
        if s.wrist_camera is not None:
            found = _camera_volume(rt, poses, s)
            if found:
                return found
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
    from ..safety.trajectory import geometry_guard, route_end_reason, vet_segment

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
            # The harness is the authority: when it objects, it is the one
            # reported. The runtime's own checks (wrist camera, held object)
            # run second and so name only what the harness cannot see.
            if cart:
                for a, b in itertools.pairwise(samples):
                    reason = harness.vet_step(a, b, 0.0)  # geometry; the streamer stretches time
                    if reason:
                        raise Refused("safety_harness", reason, i)
            else:
                reason = vet_segment(harness, q, q_goal, duration)
                if reason:
                    raise Refused("safety_harness", reason, i)
            found = _path_checks(rt, kin, samples, load, s)
            if found:
                raise Refused(found[0], found[1], i)
            segments.append(Segment(q.copy(), q_goal, T_goal))
            q = q_goal
        # A step may retreat inside the map margin; the route may not end there.
        reason = route_end_reason(harness, q)
        if reason:
            raise Refused("safety_harness", reason, len(goals) - 1)
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


def execute_route(rt, route: Route, duration: float, *, tool: str, words: str,
                  record: bool = True) -> list:
    """Run a vetted route through SafeArm; record it (or the completed part
    of it, if a later segment is interrupted) in the history. An undo runs
    with `record=False`: it consumes history, it does not add to it."""
    waypoints = [np.asarray(rt.arm.get_state().q, dtype=float).copy()]
    kin = _kin(rt)
    try:
        for i, seg in enumerate(route.segments):
            # The dispatcher may restore a saved camera-view home after
            # motion. Mark only execution of an already vetted segment;
            # planning refusals and an empty route must send no home goal.
            mark_started = getattr(rt, "_mark_motion_started", None)
            if callable(mark_started):
                mark_started()
            if route.cartesian:
                ok = rt.arm.move_cartesian(seg.T_goal, duration_s=duration)
            else:
                ok = carry_attachment.move(rt, seg.q_goal, duration_s=duration)
            if not ok:
                raise SkillError(f"did not settle at the end of segment {i + 1}/{len(route.segments)}")
            waypoints.append(np.asarray(rt.arm.get_state().q, dtype=float).copy())
    finally:
        if record:
            # Interrupted (halt_motion, a superseded call): the arm stopped
            # part-way along a VETTED segment -- on its joint line, or on its
            # straight line for cartesian profiles. Recording that stop as the
            # last waypoint lets "go back" retrace exactly the part that ran,
            # instead of refusing the whole history as stale.
            try:
                q_now = np.asarray(rt.arm.get_state().q, dtype=float).copy()
                if float(np.max(np.abs(q_now - waypoints[-1]))) > 1e-4:
                    waypoints.append(q_now)
            except Exception as exc:  # noqa: BLE001 -- a feedback fault must not mask the original error
                logger.warning("spatial move: no feedback after the route (%s); recording waypoints only", exc)
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
        "tcp_before": _tcp_in_world(rt, T0),
        "tcp_after": _tcp_in_world(rt, T1),
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
    out = success(rt, tool="move_relative", words=words, frame=move.frame, camera=move.camera,
                  delta=move.delta, T0=T0, load=load, route=route)
    # keys of the pre-frames move_relative result, kept for existing readers
    out["distance_m"] = round(move.distance, 4)
    out["tcp_xyz"] = [round(float(v), 3) for v in out["tcp_after"]]
    out["next_action"] = _frame_report(move.frame, words, out["as_seen_on_screen"])
    return out


#: How each frame's directions read to the person (relative_motion frames).
_FRAME_WORDS = {
    "base": "the robot's own directions (frame 'base')",
    "viewer": "directions as seen on the camera (frame 'viewer')",
    "tool": "the gripper's own axis (frame 'tool')",
}


def _frame_report(frame: str, words: str, on_screen: str) -> str:
    """What to tell the person after a relative move: the frame its words
    used and how the move looks on the screen. Measured on the kitchen run:
    the robot's 'left' showed as 'right on the screen' and the answer named
    neither."""
    text = (f"Tell the user this move used {_FRAME_WORDS.get(frame, f'frame {frame!r}')}: {words}. "
            f"How it looks on the screen: {on_screen}.")
    if frame == "base" and not on_screen.startswith("screen mapping unavailable"):
        text += (" If they meant directions as seen on the screen, offer the same move with "
                 "frame='viewer'.")
    return text


def undo_move(rt, steps=1) -> dict:
    """Retrace the last `steps` recorded moves backwards, one record at a time.

    Each record is undone only from where it ENDED (pose_matches): if any
    other command moved the arm since, retracing would start from a place
    the record never vetted a path from, so the whole history is cleared and
    the undo is refused. Each retrace is planned and vetted like a new move
    (the world may have changed: something may now sit where the gripper
    was); a refused retrace leaves the arm and the history untouched.
    """
    if (isinstance(steps, bool) or not isinstance(steps, (int, float, np.integer, np.floating))
            or not math.isfinite(float(steps)) or float(steps) != int(steps)
            or not 1 <= int(steps) <= MAX_UNDO_STEPS):
        raise SkillError(f"steps must be a whole number from 1 to {MAX_UNDO_STEPS}, got {steps!r}")
    s = settings(rt)
    hist = history(rt)
    if not len(hist):
        raise SkillError("nothing to undo: no move_relative / rotate_gripper / move_near_object move "
                         "is recorded for this arm (or the history was cleared because the arm moved "
                         "since)")
    load = carried(rt, s)
    duration = _duration(s, load)
    undone: list[str] = []
    T_start = None
    note = None
    route = None
    for _ in range(int(steps)):
        rec = hist.last()
        if rec is None:
            break
        q0, T0 = _pose(rt)
        T_start = T0 if T_start is None else T_start
        if not rm.pose_matches(T0, rec.tcp_end):
            hist.clear()
            why = ("the arm has moved since that move (another command moved it), so retracing it "
                   "could cross space that was never checked for it; the undo history is cleared")
            if not undone:
                raise SkillError(f"cannot undo: {why}")
            note = why
            break
        goals = [("joints", q) for q in reversed(rec.waypoints[:-1])]
        try:
            route = plan_route(rt, q0, goals, load=load, s=s, duration=duration)
        except Refused as err:
            out = refusal(rt, err, words=f"undo: {rec.words}", frame=None, delta=None, load=load)
            out["undone"] = undone
            out["arm_moved"] = bool(undone)
            return out
        execute_route(rt, route, duration, tool="undo_move", words=rec.words, record=False)
        hist.pop()
        undone.append(rec.words)
    T1 = _kin(rt).fk(rt.arm.get_state().q)
    delta = T1[:3, 3] - T_start[:3, 3]
    out = {
        "moved": True,
        "undone": undone,
        "interpretation": "undid, newest first: " + " | ".join(undone),
        "displacement_base_m": _vec(delta),
        "as_seen_on_screen": _screen(rt, delta),
        "tcp_before": _tcp_in_world(rt, T_start),
        "tcp_after": _tcp_in_world(rt, T1),
        "path": "straight line (cartesian)" if route is not None and route.cartesian else "joint space",
        "holding": load.label,
        "speed": "slow (holding an object)" if load.label else "normal",
        "undo_available": len(hist),
        "summary": f"Went back {len(undone)} step(s): undid " + "; then ".join(undone) + ".",
    }
    if note:
        out["note"] = note
    return out


def _over_cap_message(rt, s: Settings, word, frame, angle_deg, T0, tool_axes, exc) -> str:
    """When an angle is refused by the cap, say whether the requested turn
    would ALSO roll the wrist camera under -- so "rotate it upside down"
    (180 deg) is answered with the reason that matters, and the agent learns
    not to offer it in smaller pieces."""
    try:
        a = float(angle_deg)
    except (TypeError, ValueError):
        return str(exc)
    if not math.isfinite(a) or a <= rm.MAX_TURN_DEG or s.tool_up_axis is None:
        return str(exc)
    a = min(a, 360.0)
    try:
        rot = rm.resolve_rotation(word, frame=frame, angle_deg=a, tool=tool_axes,
                                  viewer=lambda: viewer_axes(rt))
    except SkillError:
        return str(exc)
    turns = [rm.rotation_matrix(rot.axis, math.radians(t)) @ T0[:3, :3]
             for t in np.linspace(0.0, a, 73)]
    cam = rm.camera_up_violation(turns, s.tool_up_axis, s.tool_up_min_z, tol=_CAMERA_TOL)
    if cam:
        return (f"refused: {cam}. Do not offer to turn it upside down, not even in steps; "
                f"at most {rm.MAX_TURN_DEG:g} deg per call and only while the camera stays on top")
    return str(exc)


def rotate_gripper(rt, direction=None, angle_deg=None, amount=None, frame=None) -> dict:
    """Small bounded orientation nudge about the TCP point: roll about the
    gripper's own axis or tilt the jaws toward a direction. Same plan -> vet
    -> execute -> record path as move_relative, so caps, the held-object
    rules, the wrist-camera rule and the vetted feasible offer all apply."""
    if direction is None:
        raise SkillError("give a direction: clockwise/counterclockwise (roll the wrist) or "
                         "up/down/left/right/forward/back/toward_viewer/away_from_viewer (tilt)")
    s = settings(rt)
    q0, T0 = _pose(rt)

    def tool_axes():
        return rm.tool_axes(T0[:3, :3], rt._tool_axis_order)

    word = rm.normalize_rotation_direction(direction)
    prefix = ""
    if word == rm.AGAIN:
        if frame is not None:
            raise SkillError("'again' repeats the last rotation as it was: omit frame "
                             "(angle_deg or amount may change its size)")
        last = _state(rt)["last"].get("rotation")
        if last is None:
            raise SkillError("nothing to repeat: no rotate_gripper has run on this arm yet")
        req, last_angle = last
        word, frame = req["direction"], req["frame"]
        angle = (rm.resolve_angle(angle_deg, amount)[0]
                 if (angle_deg is not None or amount is not None) else last_angle)
        prefix = "again: "
    else:
        try:
            angle = rm.resolve_angle(angle_deg, amount)[0]
        except SkillError as exc:
            raise SkillError(_over_cap_message(rt, s, word, frame, angle_deg, T0, tool_axes, exc)) from exc
    rot = rm.resolve_rotation(word, frame=frame, angle_deg=angle, tool=tool_axes,
                              viewer=lambda: viewer_axes(rt))
    words = prefix + rot.words
    load = carried(rt, s)
    duration = _duration(s, load)

    def attempt(f):
        goal = T0.copy()
        goal[:3, :3] = rm.rotation_matrix(rot.axis, math.radians(angle * f)) @ T0[:3, :3]
        return plan_route(rt, q0, [("pose", goal)], load=load, s=s, duration=duration)

    try:
        route = attempt(1.0)
    except Refused as err:
        frac = feasible_fraction(attempt)
        feasible = None
        if frac is not None and angle * frac >= rm.MIN_TURN_DEG:
            feasible = {"feasible_angle_deg": round(angle * frac, 1), "feasible_fraction": round(frac, 3)}
        return refusal(rt, err, words=words, frame=rot.frame, delta=np.zeros(3), load=load,
                       feasible=feasible, retry_hint="call rotate_gripper again with that angle_deg")
    execute_route(rt, route, duration, tool="rotate_gripper", words=words)
    _state(rt)["last"]["rotation"] = (rot.request(), angle)
    extra = {"kind": rot.kind, "angle_deg": angle, "rotation_axis_base": _vec(rot.axis)}
    if s.tool_up_axis is not None:
        R1 = _kin(rt).fk(rt.arm.get_state().q)[:3, :3]
        extra["camera_up_z"] = round(float((R1 @ np.asarray(s.tool_up_axis))[2]), 3)
    return success(rt, tool="rotate_gripper", words=words, frame=rot.frame, camera=None,
                   delta=np.zeros(3), T0=T0, load=load, route=route, extra=extra)


# ── positions relative to objects ───────────────────────────────────────────

_RELATIONS = {
    "above": "above", "over": "above", "on_top_of": "above", "hover_above": "above",
    "just_above": "just_above", "right_above": "just_above", "barely_above": "just_above",
    "just_over": "just_above", "slightly_above": "just_above",
    "next_to": "next_to", "beside": "next_to", "by": "next_to", "near": "next_to",
    "alongside": "next_to",
    "closer": "closer", "closer_to": "closer", "nearer": "closer", "toward": "closer",
    "towards": "closer", "approach": "closer",
}
#: gap between what the gripper carries and the target (above: vertical;
#: next_to: horizontal, edge to edge)
_DEFAULT_CLEARANCE = {"above": 0.05, "just_above": 0.015, "next_to": 0.03}
#: beside a target the carried bottom still clears its top by this much, so
#: an underestimated footprint can never put the load INTO the target
_BESIDE_OVER_TOP_M = 0.02
#: closer never ends nearer than footprint + carried radius + this
_CLOSER_MARGIN_M = 0.02
_TABLE_WORDS = {"table", "the_table", "tabletop", "table_top", "table_surface", "the_tabletop"}


def _word(text) -> str:
    return "_".join(str(text).strip().lower().replace("-", " ").split())


def _same_object(query: str, held: str) -> bool:
    q, h = _word(query).replace("the_", ""), _word(held).replace("the_", "")
    return bool(q) and (q == h or q in h or h in q)


def _target_geometry(rt, label: str, T0, load: Carried) -> dict:
    """Fresh localization of `label`: centre XY, observed top, footprint
    radius (95th percentile of the cloud around the centre). Uses the same
    `_localize` every grasp uses (fresh frames, belief fallback only within
    its freshness gate). Refuses a match that is the object in the jaws."""
    _, fix = rt._localize(label)
    pts = np.asarray(fix.points, dtype=float).reshape(-1, 3)
    if not len(pts) or not np.all(np.isfinite(pts)):
        raise SkillError(f"{label!r} was detected but has no usable 3D points to measure")
    centre = np.asarray(fix.position, dtype=float)[:2]
    top = float(pts[:, 2].max())
    spread = float(np.percentile(np.linalg.norm(pts[:, :2] - centre, axis=1), 95))
    radius = float(np.clip(spread, 0.01, 0.20))
    if load.label:
        tcp = T0[:3, 3]
        if (float(np.linalg.norm(centre - tcp[:2])) < 0.03
                and top > float(tcp[2]) - load.drop_m - 0.02):
            raise SkillError(f"the {label!r} I see is at the gripper -- it looks like the object I am "
                             f"holding ({load.label!r}); name another target")
    return {"label": str(fix.detection.label or label), "center_xy_m": _vec(centre),
            "top_z_m": round(top, 4), "radius_m": round(radius, 4)}


def _target_in_world(rt, target: dict) -> dict:
    """move_near_object's target as reported: its centre and its top in the
    table frame (the geometry is measured in the running arm's frame, like
    every target)."""
    T = _world_from_base(rt)
    if T is None:
        return target
    c = arm_frame.to_world(T, [*target["center_xy_m"], target["top_z_m"]])
    return {**target, "center_xy_m": _vec(c[:2]), "top_z_m": round(float(c[2]), 4)}


def _top_in_world(rt, centre, top) -> float:
    """The height of a target's top (measured in the running arm's frame over
    `centre`) in the table frame, as the agent's coordinates read it."""
    T = _world_from_base(rt)
    return top if T is None else float(arm_frame.to_world(T, [float(centre[0]), float(centre[1]), top])[2])


def move_near_object(rt, label=None, relation=None, clearance_m=None, side=None, frame=None,
                     distance_m=None, amount=None) -> dict:
    """Position the gripper (and its load) relative to a named object or the
    table: above it, just above it, next to it, or closer to it.

    Heights are measured, not assumed: the target's OBSERVED top (never an
    OBB extent) plus clearance, plus how far the held object hangs below
    the TCP -- so "hold it above the spool" keeps the carried object's
    bottom over the spool's top. The route is carry-shaped and fully vetted
    before the first command: climb vertically first if the goal is higher,
    travel horizontally at the higher of the two heights, descend vertically
    last; the target therefore never sits under the load during transit.
    """
    if not label or not isinstance(label, str):
        raise SkillError("label is required: the object to move relative to (or 'table')")
    if relation is None:
        raise SkillError("relation is required: above, just_above, next_to or closer")
    rel = _RELATIONS.get(_word(relation))
    if rel is None:
        raise SkillError(f"relation must be one of above / just_above / next_to / closer, got {relation!r}")
    if clearance_m is not None:
        if rel == "closer":
            raise SkillError("clearance_m does not apply to 'closer' (use distance_m or amount)")
        c = rm._finite_number(clearance_m, "clearance_m")
        if not rm.MIN_STEP_M <= c <= rm.MAX_STEP_M:
            raise SkillError(f"clearance_m must be within {rm.MIN_STEP_M:g}..{rm.MAX_STEP_M:g} m, got {c:g}")
        clearance = c
    else:
        clearance = _DEFAULT_CLEARANCE.get(rel, 0.0)
    if rel != "next_to" and side is not None:
        raise SkillError("side applies only to relation next_to")
    if rel != "closer" and (distance_m is not None or amount is not None):
        raise SkillError("distance_m/amount apply only to relation closer")
    s = settings(rt)
    q0, T0 = _pose(rt)
    load = carried(rt, s)
    limits = rt.arm.harness.limits
    table_z, table_clear = float(limits.table_z), float(limits.table_clearance)
    is_table = _word(label) in _TABLE_WORDS
    if is_table:
        if rel not in ("above", "just_above"):
            raise SkillError("relative to the table only 'above' and 'just_above' make sense")
        target = {"label": "table", "center_xy_m": _vec(T0[:2, 3]), "top_z_m": round(table_z, 4),
                  "radius_m": None}
        # the harness keeps the TCP table_clearance over the table; a carried
        # bottom keeps the same, so "just above" means just over that plane
        clearance = max(clearance, table_clear + 0.005)
    else:
        if load.label and _same_object(label, load.label):
            raise SkillError(f"I am holding the {load.label!r}: it cannot be the target. Name the "
                             "object to move it relative to")
        target = _target_geometry(rt, label, T0, load)
    centre = np.asarray(target["center_xy_m"], dtype=float)
    top = float(target["top_z_m"])
    gripper_floor = table_z + table_clear + 0.005
    p0 = T0[:3, 3]
    pts = []
    note = None
    if rel in ("above", "just_above"):
        gz = max(top + clearance + load.drop_m, gripper_floor)
        goal = np.array([centre[0], centre[1], gz])
        what = "the held object's bottom" if load.label else "the gripper"
        words = (f"{'just ' if rel == 'just_above' else ''}above {target['label']!r}: {what} "
                 f"{clearance * 100:.1f} cm over its top (top at z={_top_in_world(rt, centre, top):.3f} m)")
    elif rel == "next_to":
        if side is None or _word(side) == "nearest":
            v = p0[:2] - centre
            if float(np.linalg.norm(v)) < 0.01:
                v, note = np.array([-1.0, 0.0]), "the gripper was right above it: chose the side toward the robot base"
            unit, side_words = v / np.linalg.norm(v), "on the side nearest the gripper"
        else:
            mv = rm.resolve_translation(side, frame=frame, distance=1.0, viewer=lambda: viewer_axes(rt),
                                        tool=lambda: rm.tool_axes(T0[:3, :3], rt._tool_axis_order))
            u = np.asarray(mv.delta, dtype=float)[:2]
            if float(np.linalg.norm(u)) < 0.5:
                raise SkillError(f"side must be a horizontal direction, got {side!r}")
            unit = u / np.linalg.norm(u)
            side_words = (f"on its {mv.direction.replace('_', ' ')} side ({mv.frame} frame: "
                          f"{rm.describe_base(np.append(unit, 0.0))})")
        standoff = float(target["radius_m"]) + load.radius_m + clearance
        target["standoff_m"] = round(standoff, 4)
        gz = max(top + _BESIDE_OVER_TOP_M + load.drop_m, gripper_floor)
        goal = np.array([*(centre + unit * standoff), gz])
        words = (f"next to {target['label']!r} {side_words}: {clearance * 100:.1f} cm gap edge to edge, "
                 f"{standoff * 100:.1f} cm from its centre, over the height of its top")
    else:  # closer
        v = centre - p0[:2]
        d_h = float(np.linalg.norm(v))
        standoff = float(target["radius_m"]) + load.radius_m + _CLOSER_MARGIN_M
        target["standoff_m"] = round(standoff, 4)
        room = d_h - standoff
        if room < rm.MIN_STEP_M:
            raise SkillError(f"already as close to {target['label']!r} as it safely gets "
                             f"({d_h * 100:.1f} cm from its centre; the stop is {standoff * 100:.1f} cm)")
        step, _ = rm.resolve_distance(distance_m, amount)
        travel = min(step, room)
        if travel < step:
            note = f"stopped {standoff * 100:.1f} cm from its centre instead of moving the full {step * 100:.1f} cm"
        goal = np.array([*(p0[:2] + v / d_h * travel), p0[2]])
        words = f"{travel * 100:.1f} cm closer to {target['label']!r} (horizontally, same height)"
    delta = goal - p0
    total = float(np.linalg.norm(delta))
    if total > s.max_object_move_m:
        raise SkillError(f"{target['label']!r} is {total:.2f} m away from the gripper; at most "
                         f"{s.max_object_move_m:.2f} m per call -- move closer first or use place_at for a carry")
    if rel == "closer":
        pts = [goal]
    else:
        transit = max(float(p0[2]), float(goal[2]))
        if transit - p0[2] > 1e-3:
            pts.append(np.array([p0[0], p0[1], transit]))
        if float(np.linalg.norm(goal[:2] - p0[:2])) > 1e-3:
            pts.append(np.array([goal[0], goal[1], transit]))
        if transit - goal[2] > 1e-3:
            pts.append(goal)
    if not pts:
        return {"moved": False, "interpretation": words, "target": _target_in_world(rt, target), "relation": rel,
                "holding": load.label, "summary": f"Already there: {words}."}

    def goals_for(f):
        out = []
        for p in pts:
            T = T0.copy()
            T[:3, 3] = p0 + (p - p0) * f if rel == "closer" else p
            out.append(("pose", T))
        return out

    duration = _duration(s, load)
    try:
        route = plan_route(rt, q0, goals_for(1.0), load=load, s=s, duration=duration)
    except Refused as err:
        feasible = None
        if rel == "closer":
            frac = feasible_fraction(lambda f: plan_route(rt, q0, goals_for(f), load=load, s=s,
                                                          duration=duration))
            if frac is not None and total * frac >= rm.MIN_STEP_M:
                feasible = {"feasible_distance_m": round(total * frac, 3), "feasible_fraction": round(frac, 3)}
        out = refusal(rt, err, words=words, frame="base", delta=delta, load=load, feasible=feasible,
                      retry_hint="call move_near_object again with that distance_m")
        out["target"] = _target_in_world(rt, target)
        if err.source == "kinematics":
            out["next_action"] += (" The gripper keeps its current orientation on this move; if it "
                                   "points sideways, pointing the jaws down first (rotate_gripper "
                                   "direction=down) often makes the position reachable.")
        if len(pts) > 1:
            out["error"] += f" (segment {err.segment + 1} of {len(pts)} of the carry route)"
        return out
    execute_route(rt, route, duration, tool="move_near_object", words=words)
    extra = {"relation": rel, "target": _target_in_world(rt, target), "path_segments": len(route.segments)}
    if note:
        extra["note"] = note
    return success(rt, tool="move_near_object", words=words, frame="base", camera=None, delta=delta,
                   T0=T0, load=load, route=route, extra=extra)
