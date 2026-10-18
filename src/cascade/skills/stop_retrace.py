"""Retrace up the kept approach column after an operator STOP below the pregrasp.

An operator STOP (a halt, or a latched e-stop) that lands while a grasp is
below its pregrasp hover (descending, closing, in the post-close waits,
lifting, or withdrawing before the close) ends the grasp with no further
command. That leaves the TCP inside or next to the target, and the grasp's
exemptions are released with the skill: no short step from inside the object
strictly gains clearance (the A1 escape), so every later move was refused and
the arm stayed frozen (PR #354, "Known limitation").

Owner ruling: keep the approach column + the pregrasp after such a stop, and
let the NEXT EXPLICIT `move_home` / `reset_scene` retrace upward inside that
column before its own home move: capped 5 mm above the hover, fully vetted by
the existing gates, every step strictly upward inside the kept column. Never a
move without that explicit call; a new halt / e-stop during the retrace sends
nothing more; any other motion skill, a scene change or a new grasp forfeits
the column.

- `keep` (skill_grasp_object, when its descent / close / lift block ends on a
  stop) records the column the approach entered (`_target_column`: the
  mapping-only path with `grasp.approach_target_exemption`), `q_pre`, the
  hover ceiling, the map's scene-reset generation and when it was kept, per
  arm. Never after a contact (ContactDetected is no operator stop), never
  while a contact episode is retained (payload maps: reset_scene's contact
  recovery owns that case).
- `forget` drops the RUNNING arm's kept column: execute() on any other motion
  skill, skill_grasp_object at its start (another arm's motion leaves its
  column alone: base_pose R5 follow-up); reset_scene, once its home move is
  done, drops every arm's (`every_arm`: the scene changes for all of them).
  A column also expires `MAX_AGE_S` (300 s, monotonic clock) after the stop:
  the explicit call then drops it and today's home move runs (frozen, 0
  commands). The scene changes only through reset_scene on the reBot: its
  map's `scene_reset_generation` moves only for payload-tracking maps.
- `live` is a kept column while it still applies (the retrace's own rule):
  reset_scene's refusal names "move_home first" only then.
  `live_on_another_arm` is another arm's, judged by the same rule for that
  arm: reset_scene on this arm refuses while one exists (its forget would
  strand that arm at its object) and names move_home on that arm.
- `retrace` (skill_move_home, before its own home move) acts only when
  execute()'s ROOT motion skill is move_home or reset_scene. It moves ONE
  direct joint segment from the measured pose to `q_pre` (the reverse of the
  descent / lift segment), never a planner route, inside
  `target_column_exemption(column, z_max=hover + ceiling)`. Before the first
  command every edge of the exact stream profile must raise the TCP strictly
  inside the column (`upward_in_column`), pass `vet_step` (`vet_segment`: every
  ordinary gate and escape rule, payload checks unchanged) and end at a pose
  that passes the static route-end check; the stream re-runs those checks on
  its bound start feedback, and `approve()` gates every tick under the same
  capped column.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..types import ContactDetected, SafetyViolation, SkillError
from . import carry_attachment

#: execute()'s root motion skills that may retrace: an explicit call.
EXPLICIT_SKILLS = frozenset({"move_home", "reset_scene"})
#: A TCP less than this below the hover, inside the column, has nothing left
#: to retrace: the retrace counts as completed.
MIN_DROP_M = 0.001
#: A kept column expires this long after the stop: an explicit call later
#: than that drops it, and today's home move runs (frozen, 0 commands).
MAX_AGE_S = 300.0
#: The clock of `Kept.kept_at` and of the expiry: monotonic, never wall time.
_clock = time.monotonic


@dataclasses.dataclass(frozen=True)
class Kept:
    """The approach column kept for one arm after a stop below its pregrasp."""

    arm: object
    raw: object
    #: (center_xy, radius, z_min) in the arm's base frame, as the approach entered it
    column: tuple
    #: the pregrasp joints: the retrace's goal
    q_pre: np.ndarray
    #: TCP z at q_pre (the harness model)
    hover_z: float
    #: the column's ceiling: hover_z + the air-grasp exit's 5 mm cap
    z_max: float
    #: the retrace's nominal duration (stretched to the velocity cap)
    duration_s: float
    #: the map's scene_reset_generation when the column was kept
    scene: object
    #: `_clock()` when the column was kept (it expires MAX_AGE_S later)
    kept_at: float


def _records(runtime) -> dict | None:
    return getattr(runtime, "_stop_retraces", None)


def kept(runtime) -> Kept | None:
    """The column kept for the runtime's current arm, or None. It may no
    longer apply: decide with `live`."""
    records = _records(runtime)
    return None if not records else records.get(id(runtime.arm))


def live(runtime) -> Kept | None:
    """The kept column while it still applies, or None: the ONE rule both the
    retrace's motion and any text that promises it (reset_scene's
    "move_home first") read, so they never disagree. None when nothing is
    kept, or for another arm or backend, after MAX_AGE_S, a scene reset, a
    contact episode, a carry attachment or a motion planner. Reads only:
    the retrace drops what no longer applies."""
    record = kept(runtime)
    if record is None or _no_longer_applies(runtime, record) is not None:
        return None
    return record


def live_on_another_arm(runtime) -> Kept | None:
    """A column kept for ANOTHER arm than the running one, while it still
    applies to that arm (`live`'s rule, judged for its own arm), or None.
    reset_scene on this arm would forget it (`forget(every_arm=True)`) and
    strand that arm at its object: the reset refuses instead and names
    move_home on that arm (Task 5 review Minor 1). Reads only."""
    for record in list((_records(runtime) or {}).values()):
        if record.arm is not runtime.arm and _no_longer_applies(runtime, record, arm=record.arm) is None:
            return record
    return None


def forget(runtime, *, every_arm: bool = False) -> int:
    """Drop the running arm's kept column (`every_arm`: every arm's); how
    many were dropped. A column is one arm's approach: another arm's motion
    does not forfeit it (it would strand that arm at its object)."""
    records = _records(runtime)
    if not records:
        return 0
    if every_arm:
        count = len(records)
        records.clear()
        return count
    return 0 if records.pop(id(runtime.arm), None) is None else 1


def _scene(harness):
    return getattr(getattr(harness, "occupancy", None), "scene_reset_generation", None)


def keep(runtime, column, q_pre, *, failure, halt_generation, duration_s, ceiling_m) -> Kept | None:
    """Keep `column` and `q_pre` for the next explicit move_home /
    reset_scene when the grasp's block below the pregrasp ended on an
    operator stop: `failure` (the block's SkillError / SafetyViolation, None
    when none was raised) with the e-stop latched or a halt received since
    `halt_generation` (the grasp's start). Returns the record, or None when
    nothing is kept: no stop, a contact, no approach column, a retained
    contact episode, or a harness without a kinematic model."""
    if column is None or failure is None or isinstance(failure, ContactDetected):
        return None
    if getattr(runtime, "_contact_episode", None) is not None:
        return None
    harness = runtime.arm.harness
    stopped = bool(getattr(harness, "estopped", False)) or (
        halt_generation is not None and getattr(harness, "_halt_generation", None) != halt_generation)
    if not stopped:
        return None
    kin = getattr(harness, "kin", None)
    if not callable(getattr(kin, "fk", None)):
        return None
    q_pre = np.asarray(q_pre, dtype=float).copy()
    hover_z = float(kin.fk(q_pre)[2, 3])
    if not np.isfinite(hover_z):
        return None
    center, radius, z_min = column
    record = Kept(arm=runtime.arm, raw=getattr(runtime.arm, "raw", None),
                  column=(np.asarray(center, dtype=float)[:2].copy(), float(radius), float(z_min)),
                  q_pre=q_pre, hover_z=hover_z, z_max=hover_z + float(ceiling_m),
                  duration_s=float(duration_s), scene=_scene(harness), kept_at=_clock())
    records = _records(runtime)
    if records is None:
        records = runtime._stop_retraces = {}
    records[id(runtime.arm)] = record
    return record


def _inside(p, column, z_max) -> bool:
    """`p` (base frame) inside the kept column, below its ceiling."""
    center, radius, z_min = column
    return bool(np.linalg.norm(np.asarray(p[:2], dtype=float) - center) <= radius
                and z_min - 1e-6 <= p[2] <= z_max + 1e-6)


def upward_in_column(kin, start, goal, duration_s, rate_hz, column, z_max) -> str | None:
    """Why the stream start -> goal (the executor's exact min-jerk profile at
    `duration_s` / `rate_hz`, every edge of it) does not move the TCP
    strictly upward inside the kept column, or None."""
    from ..control.motion_profile import nominal_profile

    tcp = {}

    def at(q):
        key = np.asarray(q, dtype=float).tobytes()
        if key not in tcp:
            tcp[key] = kin.fk(np.asarray(q, dtype=float))[:3, 3]
        return tcp[key]

    for target in nominal_profile(start, goal, duration_s, rate_hz, max_steps=10000):
        for previous, waypoint, _dt in target.checks:
            a, b = at(previous), at(waypoint)
            for p in (a, b):
                if not _inside(p, column, z_max):
                    return (f"the TCP would leave the kept approach column at [{p[0]:.4f}, {p[1]:.4f}, "
                            f"{p[2]:.4f}] (radius {column[1]:.3f} m, z {column[2]:.4f}..{z_max:.4f} m)")
            if not b[2] > a[2]:
                return (f"a step would not move the TCP strictly upward ({a[2]:.6f} -> {b[2]:.6f} m)")
    return None


def _no_longer_applies(runtime, record, *, arm=None) -> str | None:
    """Why `record` no longer applies to `arm` (default: the running arm), or None."""
    arm = runtime.arm if arm is None else arm
    if record.arm is not arm or record.raw is not getattr(arm, "raw", None):
        return "the arm changed"
    if _clock() - record.kept_at > MAX_AGE_S:
        return f"the column expired ({MAX_AGE_S:.0f} s after the stop)"
    if _scene(arm.harness) != record.scene:
        return "the scene was reset"
    if getattr(runtime, "_contact_episode", None) is not None:
        return "a contact episode owns the recovery"
    if carry_attachment.active(runtime) is not None:
        return "a carry attachment is active"
    if getattr(arm, "motion_planner", None) is not None:
        return "the arm streams through a motion planner"
    return None


def retrace(runtime, *, halt_generation=None) -> str | None:
    """Retrace the current arm up its kept column to the pregrasp hover.

    Acts only inside an explicit move_home / reset_scene (execute()'s root
    motion skill); returns None and moves nothing otherwise, or when no
    column is kept, or when the kept column no longer applies (another arm or
    backend, older than MAX_AGE_S, a scene reset, a contact episode, a carry
    attachment, a motion planner) or the arm is not inside it -- those drop
    the column, and the caller's home move runs as before.
    "completed: ..." when the arm reached the hover, or was already within
    MIN_DROP_M of it inside the column (review M1): the column is consumed.

    A latched e-stop refuses first, as itself (SafetyViolation "e-stop
    latched"), before any vetting; the column stays. Other refusals before
    the first command raise SkillError("stop retrace refused: ...") and keep
    the column. A halt, an e-stop or any other stop during the
    stream propagates as itself, nothing more is sent, and the column stays
    for the next explicit call (the arm is still below its hover), except
    after a contact, which forfeits it. `halt_generation`: the caller's (the
    reset's); None leaves the running skill's baseline.
    """
    records = _records(runtime)
    if not records:
        return None
    arm = runtime.arm
    record = records.get(id(arm))
    if record is None:
        return None
    context = getattr(getattr(runtime, "_motion_scope", None), "context", None)
    if context is None or context.get("root_skill") not in EXPLICIT_SKILLS:
        return None                      # never automatic, never inside another skill
    if live(runtime) is None:            # the rule reset_scene's refusal text reads too
        records.pop(id(arm), None)
        return None
    harness = arm.harness
    if harness.estopped:
        raise SafetyViolation("e-stop latched")      # named first, never as a vetting refusal
    kin = harness.kin
    q = np.asarray(arm.get_state().q, dtype=float)
    if q.shape != record.q_pre.shape or not np.isfinite(q).all():
        raise SkillError("stop retrace refused: the joint feedback is not a finite pose")
    tcp = kin.fk(q)[:3, 3]
    if not _inside(tcp, record.column, record.z_max):
        records.pop(id(arm), None)       # not inside the column: nothing to retrace
        return None
    if record.hover_z - tcp[2] <= MIN_DROP_M:
        # Already at the hover inside the column (a STOP within MIN_DROP_M
        # of it, late in the lift or the retrace): nothing is left to raise,
        # so the retrace counts as completed (move_home reconciles the jaws).
        records.pop(id(arm), None)
        return "completed: the TCP was already at the pregrasp hover inside the kept approach column"
    from ..control.motion_profile import resolve_motion_rate
    from ..safety.trajectory import (PLAN_BUDGET_S, geometry_guard, motion_duration,
                                     route_end_reason, vet_segment)

    rate = resolve_motion_rate(getattr(arm, "raw", arm))
    goal = record.q_pre

    def vet(start, duration):
        deadline = time.monotonic() + PLAN_BUDGET_S
        with geometry_guard(harness, deadline=deadline):
            reason = upward_in_column(kin, start, goal, duration, rate, record.column, record.z_max)
            if reason is None:
                reason = vet_segment(harness, start, goal, duration, deadline=deadline,
                                     stretch=False, rate_hz=rate)
            if reason is None:
                reason = route_end_reason(harness, goal)
        return reason

    def revalidate(start, duration):
        reason = vet(np.asarray(start, dtype=float), float(duration))
        if reason:
            raise SafetyViolation(f"stop retrace became unsafe: {reason}")

    halt_kw = {} if halt_generation is None else {"_halt_generation": halt_generation}
    with harness.target_column_exemption(record.column, z_max=record.z_max) as held:
        if not held:
            raise SkillError("stop retrace refused: a retained episode owns the grasp exemption")
        reason = vet(q, motion_duration(harness, q, goal, record.duration_s))
        if reason:
            raise SkillError(f"stop retrace refused: {reason}")
        try:
            settled = runtime._move_joints(goal, duration_s=record.duration_s, _preflight=revalidate,
                                           rate_hz=rate, **halt_kw)
        except ContactDetected:
            records.pop(id(arm), None)
            raise
    if not settled:
        raise SkillError("stop retrace did not settle at the pregrasp hover")
    records.pop(id(arm), None)
    return (f"completed: the TCP rose {(record.hover_z - tcp[2]) * 100:.1f} cm up the kept approach "
            "column to the pregrasp hover")
