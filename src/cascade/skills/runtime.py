"""The curated skill API the agent programs against (ASPIRE Appendix E.2
adapted to this rig). Every call is safety-gated, trace-logged with
before/after keyframes, and mirrored into episodic memory.

Skills return plain dicts with an `ok` flag; SkillError and SafetyViolation
become {"ok": false, "error": ...} so the agent can reason about failures
instead of crashing the loop.
"""

from __future__ import annotations

from ..control import motion_evidence
from ..control.jaw_stages import closing_stages, opens_past

import contextlib
import dataclasses
import functools
import logging
from contextlib import nullcontext
import math
import os
import re
import sys
import threading
import time
import unicodedata

import cv2
import numpy as np

from ..agent.trace import TraceLogger
from ..config import Cfg, plain
from ..grasping import plan_grasps_from_fix, select_grasp, select_profile
from ..grasping import evidence as grasp_evidence
from ..grasping.obb_grasp import AngledGrasp
from ..grasping.selector import ALL_TOO_WIDE_MARKER
from . import arm_frame
from . import carry_attachment, stop_retrace
from ..memory import BeliefStore, EpisodicMemory, FrameObservation
from ..memory.beliefs import camera_source
from ..perception.colors import detection_color, parse_color_query
from ..perception.detector import can_reanchor, detect_vocabulary
from ..perception.freshness import observed_at
from ..perception.reference import ReferenceResolutionError, parse_reference
from typing import TYPE_CHECKING, Any, Callable

from ..perception.workspace import WorkspaceFilter
from ..robot_context import declares_base_pose
from ..safety import people_guard
from ..safety.trajectory import ROUTE_LIFT_HEIGHTS_M

from ..perception.grounding import (
    DetectionRejected,
    Extrinsics,
    fix_from_detection,
    localize_object,
    mask_to_points_cam,
    median_point,
    oriented_bbox,
)

if TYPE_CHECKING:  # annotation only; the runtime import stays local (import cycle)
    from ..control.arm_rig import ArmRig
    from ..perception.link_mask import LinkSelfMask
    from ..types import ObjectFix
from ..types import (ContactDetected, Detection, Frame, GraspFailed, GripperFault, MotionHalted, SafetyViolation,
                     SkillError, SkillStuck, make_transform, transform_points)

logger = logging.getLogger(__name__)

#: grasp backends that plan from the RGB-D frame the object was localized in
#: (callers pass `_frame=` to `_plan_grasps` only for these, so every other
#: backend is called exactly as before)
_FRAME_GRASP_BACKENDS = frozenset({"hug", "camera_frame"})
#: The three step outcomes `execute()` stamps on every result (`outcome`).
#: `STUCK` (RPent's third finish status) is always `ok: false` and carries
#: an `ask`: the concrete, human-actionable request that would unblock it.
#: Only persistence exhaustion and terminal reasons a human can cure become
#: stuck; harness refusals and the e-stop stay plain failures.
OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"
OUTCOME_STUCK = "stuck"

#: skills that move the arm: the WorldWatcher is held while they run so the
#: held/handled object is not re-fused at a bogus mid-air position.

def _target_column(gcfg, harness, grasp, fix) -> tuple | None:
    """(center_xy, radius, z_min) of the column directly above the grasp
    target that its own approach may enter, or None.

    Radius: the target's observed footprint around the grasp XY (95th
    percentile + 1 cm), capped by `exempt_radius_m`, so neighbours stay
    outside. Floor: 2 cm below the target's observed top, never below the
    table clearance, so table checks are unchanged. None (the strict legacy
    check) when `grasp.approach_target_exemption` is false, the harness
    cannot hold the exemption, or the target has no usable points.
    """
    if not bool(gcfg.get("approach_target_exemption", False)):
        return None
    if not callable(getattr(harness, "target_column_exemption", None)):
        return None
    return _column_geometry(gcfg, harness, grasp.position[:2], getattr(fix, "points", None))


def _column_geometry(gcfg, harness, center_xy, points) -> tuple | None:
    """(center_xy, radius, z_min) of the target's own column around
    `center_xy`, or None when `points` (the target's observed points) are
    missing, empty or not finite.

    Radius: the 95th-percentile horizontal spread of `points` around
    `center_xy` + 1 cm, clipped to [2 cm, `exempt_radius_m`]. Floor: 2 cm
    below the highest point, never below table_z + table_clearance.
    """
    if points is None:
        return None
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        return None
    center = np.asarray(center_xy, dtype=float)[:2]
    spread = float(np.percentile(np.linalg.norm(points[:, :2] - center, axis=1), 95))
    radius = float(np.clip(spread + 0.01, 0.02, float(gcfg.get("exempt_radius_m", 0.07))))
    limits = harness.limits
    floor = float(limits.table_z) + float(getattr(limits, "table_clearance", 0.0))
    z_min = max(float(points[:, 2].max()) - 0.02, floor)
    return (center, radius, z_min)


#: How far the map's top of the target may raise the pregrasp hover above the
#: observed top. The recorded gaps were 0.2-0.9 cm (NVBLOX.md measured ~1.5 cm
#: on a cup); one residue voxel must not push the hover out of reach.
MAP_TOP_RAISE_CAP_M = 0.02
#: Bound on the half-voxel term added to the map's hover. The rig's voxels
#: are 1 cm (+5 mm, the full half voxel); a coarse bridge (5 cm voxels) must
#: not add +2.5 cm on top of the 2 cm cap.
MAP_TOP_HALF_VOXEL_CAP_M = 0.005


def _pregrasp_min_z(gcfg, harness, fix) -> float | tuple[float, float] | None:
    """Lowest pregrasp TCP z for a top-down approach (select_grasp's
    `pregrasp_min_z`), or None (off: the fixed `pregrasp_offset_m`).

    `grasp.pregrasp_clear_top_m` null -> None. Otherwise the hover clears
    the object's observed top by that much (today's hover, a float). With
    `grasp.pregrasp_map_top` (opt-in per profile) it counts from the
    target's top in the occupancy MAP when that is higher: the top face of
    the highest occupied voxel in the target's own column
    (`_column_geometry` around the fix XY), raised by at most
    MAP_TOP_RAISE_CAP_M, plus half a voxel (at most
    MAP_TOP_HALF_VOXEL_CAP_M). A raised hover comes back as the ladder
    (map hover, today's hover): select_grasp tries the map hover first and
    falls back to today's hover before the fixed offset, so the pose that
    executes is never lower than today's. No map, a stale or failed map, an
    empty column or a non-finite answer keep today's hover. The harness
    still vets the result.
    """
    clear_top = gcfg.get("pregrasp_clear_top_m")
    if clear_top is None:
        return None
    top = float(np.max(fix.points[:, 2])) if len(fix.points) else float("nan")
    if not (np.isfinite(top) and np.isfinite(float(clear_top))):
        return None
    hover = top + float(clear_top)
    if not bool(gcfg.get("pregrasp_map_top", False)):
        return hover
    lift = _map_top_lift(gcfg, harness, fix, top)
    if not lift > 0.0:
        return hover
    return (hover + lift, hover)


def _map_top_lift(gcfg, harness, fix, top: float) -> float:
    """How far the occupancy map's top of the target raises its observed
    `top`: the top face of the highest occupied voxel in the target's own
    column (`_column_geometry` around the fix XY) above `top`, at most
    MAP_TOP_RAISE_CAP_M, plus half a voxel (at most
    MAP_TOP_HALF_VOXEL_CAP_M). 0.0 without a map, for a stale or failed map,
    an empty column or a non-finite answer. The pregrasp hover
    (`_pregrasp_min_z`) and point_at's object top read it the same way."""
    column_top = getattr(getattr(harness, "occupancy", None), "column_top", None)
    if not callable(column_top):
        return 0.0
    column = _column_geometry(gcfg, harness, fix.position[:2], fix.points)
    if column is None:
        return 0.0
    found = _map_column_top(harness, column_top, column)
    if found is None:
        return 0.0
    map_top, voxel = found
    lift = (min(max(float(map_top) - top, 0.0), MAP_TOP_RAISE_CAP_M)
            + min(float(voxel) / 2.0, MAP_TOP_HALF_VOXEL_CAP_M))
    return lift if lift > 0.0 else 0.0   # also NaN: a non-finite map answer is no lift


def _map_column_top(harness, column_top, column) -> tuple[float, float] | None:
    """`column_top` of the target's column, read in the occupancy map's frame.

    The column (`_column_geometry`) is in the arm's base frame; the map is the
    table frame. On an arm with a `base_pose` the centre and the floor are
    lifted as ONE point through the harness's own `_to_map_frame` and the
    map's top is taken back with `_to_base_frame` (level mounts: the grasp is
    refused on a tilted one). Without a base_pose (instance dict, so a harness
    double without one is the same) the query is today's, byte for byte.
    """
    center, radius, z_min = column
    if (getattr(harness, "__dict__", None) or {}).get("base_pose") is None:
        return column_top(center, radius, z_min)
    c = np.asarray(harness._to_map_frame(np.array([[center[0], center[1], z_min]], dtype=float)),
                   dtype=float).reshape(3)
    found = column_top(c[:2], radius, float(c[2]))
    if found is None:
        return None
    top, voxel = found
    back = harness._to_base_frame(np.array([[c[0], c[1], float(top)]], dtype=float))
    return float(np.asarray(back, dtype=float).reshape(3)[2]), voxel


#: point_at aims a hover it had to RAISE to its clearance floor this much
#: higher: IK solves to ~1e-4 and the floor is checked on the solved pose.
POINT_AT_IK_SLACK_M = 0.001


def _pointing_tool_points(gcfg, axis_order: str) -> np.ndarray:
    """(N, 3) TCP-frame points that bound the gripper's lower end in a
    top-down pointing pose: the TCP and the 8 corners of the profile's
    open-jaw sweep box (`safety.tool_volume.jaw_box`, the box the harness's
    tool volume samples). It bounds the closed jaws too: the SO-101 box
    reaches 15 mm past the TCP, the closed beak's MuJoCo collision geometry
    6.6 mm; the reBot's ends at the TCP (its fingertip plane). No sweep: the
    TCP alone. A malformed sweep raises SkillError before any motion."""
    from ..safety import tool_volume

    sweep = (gcfg.get("graspgenx") or {}).get("sweep") or {}
    if sweep.get("extents_open") is None:
        sweep = {"extents_open": [0.0, 0.0, 0.0], "offset_open": sweep.get("offset_open", [0.0, 0.0, 0.0])}
    try:
        centre, axes, half = tool_volume.jaw_box(sweep, axis_order)
    except ValueError as exc:
        raise SkillError(f"{exc}; nothing was moved") from None
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return np.vstack([np.zeros(3), centre + (signs * half) @ axes.T])


#: a remembered object with no size (neither extents nor points: place_at's
#: and push_object's records) is taken as a square column this half-wide ...
UNSIZED_HALF_M = 0.05
#: ... up to this above the table (the tallest prop the demos use is a 20 cm
#: bottle), or this half-width above its remembered point if that is higher
UNSIZED_HEIGHT_M = 0.20


def _unsized(belief) -> bool:
    """A remembered object with neither extents nor points: its position is
    all point_at knows of it."""
    pts = getattr(belief, "points", None)
    return getattr(belief, "extent", None) is None and (pts is None or not len(pts))


def _belief_footprint(belief, table_z: float = 0.0) -> tuple[np.ndarray, float]:
    """A remembered object's prism for point_at: (xy footprint polygon, top).

    The top is its remembered `top_z`, else its centre plus half its
    largest extent. The footprint covers its remembered points' xy and a
    square around its position whose half-side is half the diagonal of its
    two HORIZONTAL extents (any yaw fits in it). Extents are eigenvalue-
    ordered, not axis-aligned: the vertical one is the extent closest to its
    height (twice its top above its centre), so a 20 cm bottle's footprint is
    its 6 cm girth, not its height. An object with neither extents nor points
    (`_unsized`) is taken conservatively: a square UNSIZED_HALF_M half-wide
    around its position, up to its `top_z`, else UNSIZED_HEIGHT_M above the
    table (`table_z`) or UNSIZED_HALF_M above its position if higher. Raises
    ValueError when its position, top or extents (or points) are not finite:
    an object point_at cannot place is never skipped."""
    name = f"the {belief.color + ' ' if getattr(belief, 'color', None) else ''}{belief.label}"
    # strict CUDA mode remembers clouds on the GPU (cuda_math.remember_cloud)
    position = np.asarray(_host(belief.position), dtype=float).reshape(-1)
    ext = _host(getattr(belief, "extent", None))
    ext = None if ext is None else np.asarray(ext, dtype=float).reshape(-1)
    pts = _host(getattr(belief, "points", None))
    pts = None if pts is None or not len(pts) else np.asarray(pts, dtype=float).reshape(-1, 3)
    top = getattr(belief, "top_z", None)
    unsized = ext is None and pts is None
    if top is None and unsized and position.size == 3:
        top = max(float(table_z) + UNSIZED_HEIGHT_M, float(position[2]) + UNSIZED_HALF_M)
    if top is None and ext is not None and ext.size:
        top = float(position[2] + np.max(ext) / 2.0) if position.size == 3 else math.nan
    if top is None and pts is not None:
        top = float(pts[:, 2].max())
    top = math.nan if top is None else float(top)
    if not (position.size == 3 and np.isfinite(position).all() and math.isfinite(top)
            and (ext is None or (ext.size == 3 and np.isfinite(ext).all()))
            and (pts is None or np.isfinite(pts).all())):
        raise ValueError(f"{name}'s remembered box is not finite")
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    if unsized:
        return position[:2] + UNSIZED_HALF_M * square, top
    xy = [] if pts is None else [pts[:, :2]]
    if ext is not None:
        height = 2.0 * max(top - float(position[2]), 0.0)
        horizontal = np.delete(np.abs(ext), int(np.argmin(np.abs(np.abs(ext) - height))))
        half = float(np.hypot(*horizontal)) / 2.0
        xy.append(position[:2] + half * square)
    return np.vstack(xy), top


def _harness_tool_volume(harness):
    """The arm harness's `safety.tool_volume` (`ToolVolume`), or None: a
    profile without one, or a harness double that is not a SafetyHarness."""
    from ..safety.tool_volume import ToolVolume

    tv = getattr(harness, "tool_volume", None)
    return tv if isinstance(tv, ToolVolume) else None


def _raise_if_estopped(harness) -> None:
    """A latched e-stop vetoes every route the harness vets: when point_at
    finds no route, an e-stop is reported as the e-stop, never as the
    geometry it made fail. (`_raise_if_halted_since_start` adds the
    operator's halt.)"""
    if getattr(harness, "estopped", False):
        raise SafetyViolation("e-stop latched")


def _host_fix(fix):
    """`fix` with host numpy geometry, for a host-side planner (point_at).

    Strict CUDA perception (CASCADE_REQUIRE_CUDA=1, the GPU kitchen) keeps
    the fix's cloud a CUDA tensor (`cuda_math.backproject`), and numpy
    refuses one ("can't convert cuda:0 device type tensor to numpy"), so
    point_at's object column, top and footprints failed on that stack before
    any command. This is the deliberate host boundary after CUDA perception
    that the belief box and the GraspGen-X wire take too: a copy of the
    already localized cloud (at most a few thousand points), never a CPU
    re-localization. A fix that is already on the host comes back as is."""
    import dataclasses

    host = {name: _host(getattr(fix, name, None)) for name in ("position", "points", "extent", "axes")}
    host = {name: value for name, value in host.items() if value is not getattr(fix, name, None)}
    return dataclasses.replace(fix, **host) if host else fix


def _host(value):
    """`value` as host data: a torch tensor (strict CUDA mode: localized and
    remembered clouds stay on the GPU) is copied to numpy, anything else comes
    back as is. For host-side geometry only (point_at's planner, a belief's
    footprint); never a CPU re-run of perception."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _raise_if_halted_since_start(harness, generation) -> None:
    """A stop that came while point_at planned is what the agent must hear,
    never the geometry or the clock it made fail (#399 review Minor 2): a
    latched e-stop first (`_raise_if_estopped`), then an operator halt
    received since `generation` (the harness's halt generation when point_at
    started) or since the motion skill running on this thread started
    (`skill_halt_baseline`), named as the halt."""
    _raise_if_estopped(harness)
    check = getattr(harness, "_check_halt_generation", None)
    if not callable(check):
        return
    try:
        check(generation)
    except MotionHalted:
        halted = getattr(harness, "halted", None)
        if halted:
            raise MotionHalted(f"halted: {halted}") from None
        raise


def _hold_on_another_arm(runtime) -> bool:
    """The rig-wide hold belongs to ANOTHER arm of a pair than the one this
    call drives (`SkillRuntime._held_by_another_arm`; getattr-guarded: absent
    on runtime doubles, whose hold stays this arm's)."""
    elsewhere = getattr(runtime, "_held_by_another_arm", None)
    return bool(callable(elsewhere) and elsewhere())


class _PointTargetMoved(Exception):
    """point_at's target moved while it planned (`_point_at_still_there`):
    plan once more from `fix`, where it is now, localized at `localized`
    ((`pointing._wall()`, `time.monotonic()`, the image it came from, that
    image's camera)). Never leaves `skill_point_at`."""

    def __init__(self, fix, localized):
        super().__init__("the target moved while point_at planned")
        self.fix = fix
        self.localized = localized


def _lowest_tool_z(T, tool_points) -> float:
    """Lowest base-frame z of `tool_points` (TCP frame) at the TCP pose T."""
    T = np.asarray(T, dtype=float)
    return float((np.asarray(tool_points) @ T[:3, :3].T + T[:3, 3])[:, 2].min())


def _column_kw(column) -> dict:
    if column is None:
        return {}
    return {"exempt_xy": column[0], "exempt_radius_m": column[1], "exempt_z_min": column[2]}


def _drain_clearance_lost(harness) -> int:
    """`harness.drain_clearance_lost()`, 0 for a double without it."""
    drain = getattr(harness, "drain_clearance_lost", None)
    return int(drain()) if callable(drain) else 0


def _column_exemption(harness, column, z_max=None):
    """The target column exemption, or a no-op for None. `z_max` caps it
    (the air-grasp exit); the approach passes none: today's unbounded call."""
    if column is None:
        return contextlib.nullcontext(False)
    if z_max is None:
        return harness.target_column_exemption(column)
    return harness.target_column_exemption(column, z_max=z_max)


#: Observed-finger gate only (CASCADE_OBSERVED_FINGER_GATE=1): a candidate whose
#: fingers come within this horizontal distance of an observed non-target
#: surface is tried after every other candidate (`grasp.observed_rank_lateral_m`).
#: Ranking only, never an admission margin. Sized to the settle error: Isaac
#: settles within 0.02 rad per joint, ~8 mm at the tool at kitchen reach;
#: rounded up. The early kitchen E2E (attempt C3) chose a grasp whose planned
#: fingers cleared a lemon, then was refused at the MEASURED pose beside it.
OBSERVED_RANK_LATERAL_M = 0.01
#: Upper bound of `grasp.observed_rank_lateral_m`: a wider distance ranks
#: nearly every candidate last, which only reorders them at full cost.
OBSERVED_RANK_LATERAL_MAX_M = 0.03
#: The ranking runs inside the approach deadline (`PLAN_BUDGET_S`, also the
#: route vet's budget), so it is bounded: at most this many candidates of one
#: grasp are ranked (`grasp.observed_rank_max_candidates`, 0 = no ranking; the
#: kitchen's GraspGen-X hands over `topk` 32 per batch) ...
OBSERVED_RANK_MAX_CANDIDATES = 32
#: ... and ranking stops once its own work reaches this time slice. The rest
#: keep their planned order. Measured at 720 x 1280: 1-5 ms per query, up to
#: 8 queries per candidate.
OBSERVED_RANK_TIME_SLICE_S = 0.3
#: Longest wait for EMPTY jaws to reach a commanded opening (an air grasp's
#: reopen, a grasp that starts with closed jaws), unless the profile sets
#: `grasp.open_wait_timeout_s`. A bound only: the wait ends at the first
#: feedback read that confirms the opening. Wall clock: the Isaac profiles
#: set a Newton-sized bound (configs/arms/isaac.yaml).
JAWS_OPEN_TIMEOUT_S = 3.0
#: Feedback read interval of that wait.
JAWS_OPEN_POLL_S = 0.05
#: After a pre-command veto with a completed vetted withdrawal, at most this
#: many further candidates of the same plan are tried.
OBSERVED_NEXT_CANDIDATES = 2
#: The next candidate is never a near-copy of a refused one (learned planners
#: propose clusters of them; the copy would be refused the same way): TCP
#: within this distance ...
OBSERVED_NEAR_COPY_M = 0.01
#: ... with its approach AND its jaws within this angle (either jaw
#: orientation). A grasp at the same centre with the jaws turned further puts
#: the fingers elsewhere and stays a candidate.
OBSERVED_NEAR_COPY_DEG = 15.


#: Stall band of the close-feedback wait (`_wait_gripper_closed`, the wait
#: after each close stage when `grasp.close_feedback_timeout_s` is set): the
#: jaws count as stopped once every read stays within this fraction of jaw
#: travel of an anchor for 0.5 s. Per arm profile: `gripper.close_stall_band`.
#: The Isaac profiles set 0.01: on Newton the jaws slowly turn a cube grasped
#: 10-30 deg off its faces square, 0.003-0.004 per 0.5 s, which this default
#: reads as moving until the 8 s timeout refuses the lift
#: (configs/arms/isaac.yaml).
CLOSE_STALL_BAND = 0.002
#: Exclusive upper limit of a profile's band, with a margin below the
#: smallest free-closing step (0.028 open fraction per Newton bridge
#: iteration): every step leaves the band, so steady free closing (one step
#: per bridge iteration at a steady loop rate above 2 it/s) never reads as
#: stopped. A band of 0.028 or more holds a whole step and reads it as a stop
#: at 2-3.5 it/s. No cap helps a gap of 0.5 s or more without a step (a
#: late first motion after a stage command, a stalled loop): below 0.95 open
#: every band reads it as a stop.
CLOSE_STALL_BAND_MAX = 0.02


def close_stall_band(gripper_cfg) -> float:
    """`gripper.close_stall_band` of an arm profile, CLOSE_STALL_BAND when unset.

    Anything but a number in (0, CLOSE_STALL_BAND_MAX) raises SkillError:
    build_runtime checks every arm profile before any arm exists, and the
    close checks it again before its first jaw command."""
    band = gripper_cfg.get("close_stall_band") if gripper_cfg is not None else None
    if band is None:
        return CLOSE_STALL_BAND
    # NaN, inf, True/False (1/0) and anything that is no number fall outside
    if not isinstance(band, (int, float)) or not 0.0 < band < CLOSE_STALL_BAND_MAX:
        raise SkillError(f"gripper.close_stall_band must be a fraction of jaw travel in "
                         f"(0, {CLOSE_STALL_BAND_MAX}), got {band!r}")
    return float(band)


def open_wait_timeout_s(grasp_cfg) -> float | None:
    """`grasp.open_wait_timeout_s` in seconds, None when unset.

    Anything but a finite positive number raises SkillError: build_runtime
    checks it before any arm exists, and every skill that waits on it checks
    it before it moves anything (never first after a grasp's close)."""
    timeout = grasp_cfg.get("open_wait_timeout_s")
    if timeout is None:
        return None
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise SkillError(f"grasp.open_wait_timeout_s must be finite and positive (seconds), got {timeout!r}")
    return float(timeout)


class ProvisionalHold(tuple):
    """The `_held_provisional` marker a grasp's close leaves: (label, detector
    label, colour), unpacked like the plain tuple, plus what the grasp's own
    verify (`SkillRuntime._air_grasp`) judges the jaws against, as open
    fractions of jaw travel: `commanded_open`, the opening the close last
    commanded (its seat), and `expected_open`, where the object should stop
    the jaws. reset_scene on a real gripper classifies an interrupted grasp's
    jaws with them (`_reconcile_held(seat_aware=True)`); a marker without
    them (None, or a plain tuple) keeps the old rule."""

    commanded_open: float | None
    expected_open: float | None
    #: the SafeArm whose close left this marker (spec D9 / R5); None: unknown
    #: (a plain tuple, or a marker rebuilt without it), judged as before
    arm: object | None

    def __new__(cls, marker, commanded_open: float | None = None, expected_open: float | None = None,
                arm=None):
        hold = super().__new__(cls, marker)
        hold.commanded_open = commanded_open
        hold.expected_open = expected_open
        hold.arm = arm
        return hold


class _HoldNotJudged(Exception):
    """A seat-aware reconcile (reset_scene's refusal, move_home's jaw report
    after a stop retrace) read the jaws but could not judge the interrupted
    grasp's marker against that reading (a malformed marker): not a jaw-read
    failure. Raised only with `seat_aware`; the cause is chained."""


#: How the seat-aware callers name jaws they could not read, and jaws they
#: read but could not judge (`_HoldNotJudged`). Short: the move_home-first
#: refusal needs room for the label within the agent's 240 characters.
_JAWS_NOT_READ = "the jaw state could not be read"
_JAWS_NOT_JUDGED = "the jaw reading could not be judged"
#: Why a persistence loop stops after a failed attempt that left an
#: interrupted grasp's provisional hold (`_grasp_retry_verdict`): the next
#: grasp would refuse it (`_possible_hold_refusal`); the ask names the way out.
#: In #366/#377's words, so the envelope files the stop `state:jaws_unreadable`
#: whatever the attempt's own text ("jaws" alone matched `object_too_wide`).
_POSSIBLE_HOLD_STOP = _JAWS_NOT_READ + " and the jaws may still hold an object; not retrying"
#: The human's way out of a hold the jaws cannot confirm: the ask of a
#: persistence loop that stopped on it (`_stuck_ask`) and of every skill entry
#: that refuses it (`_raise_on_a_possible_hold`). The agent relays it; it never
#: opens the jaws itself.
_POSSIBLE_HOLD_ASK = ("My jaws may still hold an object, and I cannot read whether they do. If one is in them, "
                      "take it out by hand, then say 'open the gripper' (open_gripper), and ask again.")


def _error_text(exc: BaseException) -> str:
    """An exception's text for a refusal's \"Error: \" tail; one without text
    names its type, never a bare tag."""
    return str(exc).strip() or type(exc).__name__


def _cut_text(text: str, limit: int) -> str:
    """`text` within `limit` characters; a cut ends in "..." (the failure
    line keeps that marker, `_with_failure_instruction`). A limit too small
    for the marker keeps that many of its dots (none below 1): never more
    than `limit` characters."""
    limit = max(int(limit), 0)
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..." if limit >= 3 else "." * limit


#: The fewest characters a refusal leaves a label: 18 of its own plus "...".
#: Every shipped template leaves at least this; a longer template keeps the
#: label readable and runs past the 240 characters instead.
_MIN_LABEL_ROOM = 21


def _fit_label(template: str, label, *, keep: int = 0) -> str:
    """`template` with `label` in its `{}`, the label shortened so the text
    plus `keep` more characters fits the error the agent's failure line quotes
    (_FAILED_ERROR_CHARS): a long name is cut, never the sentence. A template
    that leaves less than _MIN_LABEL_ROOM still gets that much of the label
    (the failure line then marks the text truncated), never a bare "..."."""
    room = _FAILED_ERROR_CHARS - keep - len(template.format(""))
    return template.format(_cut_text(str(label), max(room, _MIN_LABEL_ROOM)))


def _unread_hold_refusal(label, *, column_kept: bool, then: str, judged: bool = False,
                         error: BaseException | None = None, prefix: str = "") -> str:
    """The refusal for an interrupted grasp's provisional hold whose jaws
    could not be read (`judged`: read, but the hold could not be judged
    against the reading): the way out first -- move_home first while the
    approach column is kept, then the object out by hand, open_gripper and
    `then` -- and the driver's text after it (" Error: ..."). reset_scene
    (`then` reset_scene) and every skill whose first jaw command opens the
    jaws (`SkillRuntime._possible_hold_refusal`) share it. The text, after
    `prefix` (the exception name execute() puts first), fits the 240
    characters the agent's failure line quotes: a long label is shortened,
    never the sentence, and gives way to the driver's text only as far as it
    needs (at most _REFUSAL_ERROR_ROOM characters)."""
    way = ("call move_home first (it raises the gripper up its approach column), then, if {} is in the "
           "jaws, take it out by hand, open_gripper, %s." if column_kept else
           "if {} is in the jaws, take it out by hand, then open_gripper, then %s.") % then
    head = (_JAWS_NOT_JUDGED if judged else _JAWS_NOT_READ) + ": " + way
    tail = " Error: " + ("no valid jaw feedback" if error is None else _error_text(error))
    way_out = _fit_label(head, label, keep=len(prefix) + min(_REFUSAL_ERROR_ROOM, len(tail)))
    return way_out + _cut_text(tail, _FAILED_ERROR_CHARS - len(prefix) - len(way_out))


def _air_grasp_stop(harness, generation) -> str | None:
    """Why the recovery after an air grasp (the jaws' reopen, the exit
    withdrawal) must not act: a latched e-stop, a pending operator halt, or
    a halt received since `generation` (the grasp's start). None: it may."""
    if getattr(harness, "estopped", False):
        return "e-stop latched"
    if getattr(harness, "halted", None):
        return f"motion halted: {harness.halted}"
    if getattr(harness, "_halt_generation", None) != generation:
        return "halt received"
    return None


def _air_grasp_stop_error(harness, stop: str) -> SafetyViolation:
    """The exception a stop seen during that recovery propagates as: the
    e-stop's SafetyViolation, else the operator halt's MotionHalted."""
    return SafetyViolation(stop) if getattr(harness, "estopped", False) else MotionHalted(stop)


def _same_grasp(a, b) -> bool:
    """`b` is candidate `a` or its jaws-swapped twin (`selector._flip_twin`)."""
    if a is b:
        return True
    from ..grasping.selector import _flip_twin
    try:
        if not (np.array_equal(np.asarray(a.position, float), np.asarray(b.position, float))
                and np.array_equal(np.asarray(a.approach, float), np.asarray(b.approach, float))
                and float(a.width_m) == float(b.width_m)):
            return False
        rb = np.asarray(b.rotation, float)
        return bool(np.allclose(np.asarray(a.rotation, float), rb, rtol=0, atol=1e-9)
                    or np.allclose(_flip_twin(a).rotation, rb, rtol=0, atol=1e-9))
    except (TypeError, ValueError):
        return False


def _near_copy(a, b) -> bool:
    """`b` puts the fingers where refused candidate `a` (or its jaws-swapped
    twin) did: TCP within `OBSERVED_NEAR_COPY_M`, approach and the whole tool
    rotation within `OBSERVED_NEAR_COPY_DEG`."""
    from ..grasping.selector import _flip_twin
    try:
        if not float(np.linalg.norm(np.asarray(a.position, float) - np.asarray(b.position, float))) \
                <= OBSERVED_NEAR_COPY_M:
            return False
        limit = math.cos(math.radians(OBSERVED_NEAR_COPY_DEG))
        da, db = np.asarray(a.approach, float).reshape(3), np.asarray(b.approach, float).reshape(3)
        if not float(da @ db) >= limit * float(np.linalg.norm(da) * np.linalg.norm(db)):
            return False
        rb = np.asarray(b.rotation, float)
        # cos(rotation angle) = (trace(Ra^T Rb) - 1) / 2
        return any((float(np.trace(np.asarray(ra, float).T @ rb)) - 1.) / 2. >= limit
                   for ra in (a.rotation, _flip_twin(a).rotation))
    except (TypeError, ValueError, AttributeError):
        return False


class _ObservedSelection:
    """Which candidates the observed-finger grasp ranked last, dropped and
    attempted (the result's `observed_selection`, failures' exception
    attribute of the same name, and the grasp trace). Indices count every
    planned candidate of this grasp, batches in order. Shared by the
    re-entries for the next candidate, with the bounded neighbour ranking."""

    _MAX_LISTED = 5

    def __init__(self):
        self.candidates, self.attempts = [], []
        self._ids = {}
        self._ranked_last, self._dropped = {}, {}
        self._pending = []          # vet failures of the candidate the selector is on
        self.ranking = None         # observed_scene.NeighbourRanking, once the gate exists
        self.selection_s = 0.
        self.first_refusal = None

    def add_batch(self, grasps):
        for grasp in grasps:
            self._ids.setdefault(id(grasp), len(self.candidates))
            self.candidates.append(grasp)

    def index(self, grasp):
        known = self._ids.get(id(grasp))
        if known is not None and self.candidates[known] is grasp:
            return known
        return next((i for i, c in enumerate(self.candidates) if _same_grasp(c, grasp)), None)

    def ranked(self, grasp, reason):
        index = self.index(grasp)
        if index not in self._ranked_last:
            self._ranked_last[index] = reason
            grasp_evidence.event("observed_neighbour_ranked_last", candidate=index, reason=reason)

    def track(self, grasps):
        """Hand `grasps` to the selector. It asks for the next candidate only
        when neither jaw orientation of the current one passed: then (and
        only then) the current one's vet refusals are drops."""
        for grasp in grasps:
            self._settle()
            yield grasp

    def vet_failed(self, grasp, reason):
        self._pending.append((grasp, str(reason)))

    def _settle(self):
        for grasp, reason in self._pending:
            self._dropped.setdefault(self.index(grasp), reason)
        self._pending = []

    def end_select(self, seconds, *, exhausted):
        """One `select_grasp` call ended: a candidate was chosen (its other
        orientation's refusal is no drop), every one failed (`exhausted`), or
        it was interrupted (undecided: no drop)."""
        if exhausted:
            self._settle()
        self._pending = []
        self.selection_s += max(0., float(seconds))
        grasp_evidence.event("observed_selection_time", selection_s=round(self.selection_s, 3),
                             **({} if self.ranking is None else self.ranking.summary()))

    def attempt(self, grasp):
        self.attempts.append({"candidate": self.index(grasp), "outcome": "started"})

    def finish(self, outcome):
        self.attempts[-1]["outcome"] = outcome

    def attempted(self, grasp):
        return any(_same_grasp(self.candidates[a["candidate"]], grasp)
                   for a in self.attempts if a["candidate"] is not None)

    def untried(self, grasps):
        """Candidates the next attempt may use: never attempted, and not a
        near-copy of an attempted (refused) one."""
        tried = [self.candidates[a["candidate"]] for a in self.attempts if a["candidate"] is not None]
        return [c for c in grasps
                if not any(_same_grasp(t, c) or _near_copy(t, c) for t in tried)]

    def refused(self):
        return sum(1 for a in self.attempts if a["outcome"].startswith("refused before contact"))

    def refused_grasps(self) -> tuple:
        """The candidates this grasp attempted that did not end grasped: what
        the sim retry must not try again (`_without_refused`)."""
        return tuple(self.candidates[a["candidate"]] for a in self.attempts
                     if a["candidate"] is not None and a["outcome"] != "grasped")

    def note(self, why_not=None):
        """Suffix of a failure after candidates of this plan were refused
        before contact and withdrew (empty when none was: the message then
        stays exactly as it was before retries). `why_not`: the final veto
        itself withdrew and no further candidate is tried."""
        refused = self.refused()
        plural = "s" if refused != 1 else ""
        if why_not is not None and refused:
            first = f"; first refusal: {self.first_refusal}" if refused > 1 else ""
            return (f"; {refused} grasp candidate{plural} refused before contact and withdrew; "
                    f"no further candidate ({why_not}){first}")
        if refused:
            return (f"; after {refused} earlier grasp candidate{plural} refused before contact and withdrew "
                    f"(first refusal: {self.first_refusal})")
        return ""

    def annotate(self, exc):
        """Attach the choice record to an exception leaving the grasp."""
        if self.attempts and self.attempts[-1]["outcome"] == "started":
            self.attempts[-1]["outcome"] = f"failed: {str(exc)[:200]}"
        try:
            exc.observed_selection = self.summary()
        except AttributeError:
            pass

    def summary(self):
        def listed(items):
            return [{"candidate": i, "reason": r[:160]} for i, r in list(items.items())[:self._MAX_LISTED]]
        out = {"ranked_last": listed(self._ranked_last), "dropped": listed(self._dropped),
               "dropped_count": len(self._dropped), "attempts": [dict(a) for a in self.attempts],
               "ranking": None if self.ranking is None else self.ranking.summary(),
               "selection_s": round(self.selection_s, 3)}
        if self.first_refusal is not None:
            out["first_refusal"] = self.first_refusal
        if self.attempts and self.attempts[-1]["outcome"] == "grasped":
            out["succeeded_attempt"] = len(self.attempts)
        return out


def _observed_selection_on_failure(function):
    """Every exception leaving `skill_grasp_object` once its observed-finger
    choice record exists carries that record (`exc.observed_selection`), so
    the grasp_object / pick_and_place failure results can report it, and
    leaves the attempted candidates that did not grasp on the call's sim
    retry state (`_SimRetryState.refused_grasps`: the retry excludes them,
    spec §4.5; nothing without a ledger)."""
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        record = kwargs.get("_observed_record")
        if record is None:
            record = kwargs["_observed_record"] = {}
        try:
            return function(self, *args, **kwargs)
        except Exception as exc:
            selection = record.get("selection")
            if selection is not None:
                selection.annotate(exc)
                state = _sim_retry_state(self)
                if state is not None:
                    state.refused_grasps = selection.refused_grasps()
            raise
    return wrapped


class _SimRetryState:
    """THE sim retry's per-call state (spec §4.5) for one outermost motion call
    of a runtime with a ledger. It lives on the calling thread's motion scope
    (`_sim_retry_state`), so another thread (the MCP read-only lane, a stop
    frame) never reads or writes it, whatever lock it holds.
    `exclude`: the grasps THE retry must not repeat (empty unless the call is
    the retry); `refused_grasps`: the attempted candidates of the call's last
    refused observed-finger grasp; `refusal_exc_type`: the class NAME of the
    exception that refused pick_and_place's grasp stage (pick_and_place folds
    that refusal into a result, R6)."""

    __slots__ = ("exclude", "refused_grasps", "refusal_exc_type")

    def __init__(self, exclude=()):
        self.exclude = tuple(exclude)
        self.refused_grasps: tuple = ()
        self.refusal_exc_type: str | None = None


def _sim_retry_state(runtime) -> _SimRetryState | None:
    """This thread's sim retry state, bound by `_execute_skill` on the runtime's
    thread-local motion scope for an outermost motion call of a runtime with a
    ledger; None anywhere else (also for a runtime double without a scope)."""
    return getattr(getattr(runtime, "_motion_scope", None), "sim_retry", None)


#: the top-level result field the kitchen rules' R5 keys on (`agent.learning_surface.LEARNING_LINE_FIELD`)
_LEARNING_LINE_FIELD = "learning_line"


@dataclasses.dataclass(frozen=True)
class _LearningGrant:
    """What `_grant_learning_line` decided for the call `execute()` is finishing: the
    learning event (`kind`, `line` -- None for THE retry without one -- and `outcome`),
    `stopped`, the call's operator-stop reading, taken again on delivery, and
    `trace_step`, the step of the call's trace row (`_bind_trace_step`; None before
    the row is written), where a line withdrawn on delivery is marked withheld."""
    learning: Any
    kind: str
    line: str | None
    outcome: str
    stopped: Callable[[], bool]
    trace_step: int | None = None


def _take_learning_grant(runtime) -> _LearningGrant | None:
    """The decision `_grant_learning_line` left on this thread's motion scope for
    the call `execute()` is finishing, removed (None when there is none)."""
    scope = getattr(runtime, "_motion_scope", None)
    return None if scope is None else scope.__dict__.pop("learning_grant", None)


def _bind_trace_step(runtime, step) -> None:
    """Note on this thread's pending learning grant the step of the trace row that
    carries its line (`TraceLogger.record`'s return value; a double's None binds nothing)."""
    scope = getattr(runtime, "_motion_scope", None)
    grant = getattr(scope, "learning_grant", None)
    if grant is not None and isinstance(step, int):
        scope.learning_grant = dataclasses.replace(grant, trace_step=step)


def _without_refused(grasps, excluded):
    """`grasps` without every candidate that repeats one of `excluded` (the
    refused attempt's grasps): the same candidate or its jaws-swapped twin
    (`_same_grasp`), or a near-copy (`_near_copy`). Nothing excluded: `grasps`
    itself."""
    if not excluded:
        return grasps
    return [grasp for grasp in grasps
            if not any(_same_grasp(refused, grasp) or _near_copy(refused, grasp) for refused in excluded)]


#: The sim retry's plan holds only grasps the refused attempt already tried.
_NO_DIFFERENT_GRASP = "no different grasp candidate: every planned grasp repeats the refused one"


class _OtherThanRefused:
    """THE sim retry's planned batches (spec §4.5), each one `_without_refused`.
    A batch the exclusion empties is an empty batch: the feasibility search
    goes on to its next batch (A7 review Minor 1). `only_repeats`: a batch was
    emptied and none kept a grasp, so the plan is "no different grasp
    candidate", not a planning failure of the object. `last_emptied`: the
    latest batch was one the exclusion emptied (a search that ends on it ends
    with the last vetted batch's refusal; A7 re-review Minor 1). Nothing
    excluded: every batch as planned, `only_repeats` and `last_emptied` never
    true."""

    def __init__(self, excluded):
        self.excluded = tuple(excluded)
        self.emptied = self.kept = self.last_emptied = False

    def __call__(self, grasps):
        kept = _without_refused(grasps, self.excluded)
        self.last_emptied = bool(grasps) and not kept
        self.emptied = self.emptied or self.last_emptied
        self.kept = self.kept or bool(kept)
        return kept

    @property
    def only_repeats(self) -> bool:
        return self.emptied and not self.kept


def _episode_pending(harness) -> bool:
    """A contact or a release episode this harness still holds open (its
    explicit recovery is pending). Registry read only: never a backend."""
    return (getattr(harness, "_pending_contact_episode", None) is not None
            or getattr(harness, "_pending_release_episode", None) is not None)


class _ObservedNextCandidate:
    """Re-entry context of `skill_grasp_object` for the next candidate."""

    def __init__(self, *, gate, halt_generation, remaining, selection):
        self.gate, self.halt_generation = gate, halt_generation
        self.remaining, self.selection = list(remaining), selection


_MOTION_SKILLS = {
    "grasp_object", "place_at", "place_on_object", "push_object",
    "open_gripper", "close_gripper", "move_home", "pick_and_place",
    "point_at", "wave", "handover", "sort_by_color", "move_relative",
    "throw", "grasp_at_pixel", "turn_screw", "reset_scene", "rotate_joint",
    # natural-language spatial nudges (skills/spatial_moves.py): every one
    # streams through SafeArm, so `motion lock` must deny it too
    "undo_move", "rotate_gripper", "move_near_object",
    # "home" / "pose 0" / "rest": folds onto park_q through SafeArm and then
    # switches the motors off (skills/rest_pose.py)
    "move_to_rest",
    # Pigey composites over the belief store: both pick and place through
    # the same primitives every other composite uses (snapshot_scene is
    # perception-only and deliberately absent)
    "restore_scene", "search_for_object",
}

#: Skills that run inside the detector's foreground-perception scope
#: (perception.detector.foreground): every motion skill plus the `_localize`
#: callers that move nothing. Background detector work (the brain's watcher
#: inventories, SAM track steps) never competes with their localization.
_FOREGROUND_PERCEPTION_SKILLS = _MOTION_SKILLS | {"preview_grasp", "localize_object"}

#: The tilt refusal's scope (spec D4 / D10, §5): the brief's 15 world-target
#: skills, FROZEN as the base_pose plan started. Each turns a perceived
#: position (`_localize`, `_belief_pose`, `fix_at_pixel`) or agent table-frame
#: coordinates into a target in the running arm's base frame and reports back
#: in the table frame (spec 2026-10-17 base_pose runtime frame; #352's
#: world-target refusal went with the last conversion, Task 6). The
#: conversion is exact only on a level mount (one `table_z` plane, the
#: top-down yaw, the reach annuli), so an arm whose base_pose has any roll or
#: pitch refuses all 15 before any work. `halt_motion` (never deniable) and the
#: joint-space / relative skills are not in it. Pinned by
#: tests/test_base_pose_tilted_mount.py.
_TILT_REFUSED_SKILLS = frozenset({
    "grasp_object", "pick_and_place", "grasp_at_pixel", "place_at",
    "place_on_object", "push_object", "move_near_object", "sort_by_color",
    "search_for_object", "restore_scene", "turn_screw", "throw", "handover",
    "point_at", "preview_grasp",
})


def _tilt_angle_text(deg: float) -> str:
    """One angle of the tilt refusal: 2 decimals, but an axis that tilts
    beyond the level tolerance and rounds to 0.00 keeps 2 significant digits
    ("1.1e-07"), so the refusal never calls a mount tilted at 0.00 deg. A
    level axis (within the tolerance) and a rounded -0.0 print as 0.00."""
    shown = round(float(deg), 2) + 0.0          # + 0.0: a rounded -0.0 is 0.00
    if shown == 0.0 and abs(np.sin(np.radians(float(deg)))) > arm_frame._RIGID_ATOL:
        return f"{float(deg):.2g}"
    return f"{shown:.2f}"


def _tilted_mount_refusal(name: str, arm_name, roll_deg: float, pitch_deg: float) -> dict:
    """The result of a `_TILT_REFUSED_SKILLS` skill on an arm whose base_pose
    is tilted (spec D4, §5): nothing was perceived, commanded or moved, so
    there is no move_home step."""
    who = "this arm" if arm_name in (None, "default") else f"arm {arm_name!r}"
    roll, pitch = (_tilt_angle_text(a) for a in (roll_deg, pitch_deg))
    return {
        "ok": False,
        "outcome": OUTCOME_FAILED,
        "error": (f"{name} refused on {who}: its base_pose is tilted (roll {roll} deg, pitch "
                  f"{pitch} deg); table-frame targets need a level mount; no motion was sent"),
        "next_action": ("Do not retry this skill on this arm. Level its base_pose in the profile, or use "
                        "move_home, move_to_rest, rotate_joint, open_gripper or close_gripper on it."),
    }


#: Skills execute() runs even when the running arm's frame cannot be
#: resolved: a halt is never deniable (it binds no frame then).
_FRAME_FREE_ALWAYS = frozenset({"halt_motion"})


def _arm_frame_refusal(name: str, arm_name, why) -> dict:
    """A call refused because the running arm's frame cannot be resolved
    (spec D4): no binding, or a base_pose that is not rigid. Nothing was
    perceived, commanded or moved, so there is no move_home step."""
    who = "this arm" if arm_name in (None, "default") else f"arm {arm_name!r}"
    return {"ok": False, "outcome": OUTCOME_FAILED,
            "error": f"{name} refused on {who}: {why}; no motion was sent"}

#: Labels that name hollow things -- what Pigey's occlusion search lifts
#: first ("start with the largest/hollowest container (bowl, basket, box),
#: since those hide the most; then taller objects"). Size breaks ties.
_HOLLOW_WORDS = (
    "bowl", "cup", "mug", "box", "basket", "bin", "container", "pot", "plate",
    "tray", "lid", "jar", "can", "bucket", "glass", "dish", "carton", "crate",
    "cuenco", "taza", "caja", "cesta", "bote", "plato", "bandeja", "tapa", "vaso",
)


def _jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _frame_age_s(frame: Frame, max_age_s: float = 5.0) -> float:
    """Frame.t is client-local receipt time, never the remote capture clock."""
    try:
        age = time.monotonic() - float(frame.t)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SkillError("Camera frame has no valid local timestamp") from exc
    if not np.isfinite(age) or age < 0:
        raise SkillError("Camera frame has an invalid local timestamp")
    if age > max_age_s:
        raise SkillError(f"Camera frame is {age:.1f}s old; wait for reconnection and try again")
    return round(age, 3)


class SlowPerceptionError(SkillError):
    """The frame was fresh when analysis began; the analysis itself outlasted
    the age limit. The camera is live, so "wait for reconnection" would be
    false -- the caller may analyze a newer frame instead."""


class StaleFrameError(SlowPerceptionError):
    """The frame was ALREADY past the age limit when its analysis began: the
    camera stopped delivering (`CameraStream.get_frame` returns a dead
    camera's last frame after its wait). "The camera is live -- retry" would be
    false. A SlowPerceptionError subclass, so every path that ends on slow
    perception (no memory fallback, no motion recovery) ends on this too; only
    the text differs."""


def _analyzed_frame_age_s(frame: Frame, analysis_s: float, max_age_s: float = 5.0, *,
                          age_at_start: float | None = None) -> float:
    """Age of a frame whose freshness was checked BEFORE analysis: past the
    limit now only because the analysis was slow, reported as exactly that.

    `age_at_start`: the frame's age when it was grabbed, from a caller whose
    frame was NOT checked before (`_localize`'s fallbacks). A frame already
    past the limit then was stale before any analysis (a dead camera's last
    frame): StaleFrameError, which says so, instead of "fresh at capture ...
    the camera is live"."""
    age = _frame_age_s(frame, max_age_s=float("inf"))
    if age > max_age_s:
        if age_at_start is not None and age_at_start > max_age_s:
            raise StaleFrameError(f"Camera frame is {age:.1f}s old; wait for the camera to reconnect")
        raise SlowPerceptionError(
            f"Perception took {analysis_s:.1f}s on a frame that was fresh at capture, so the "
            f"result is {age:.1f}s old; the camera is live -- retry the observation"
        )
    return age


class SceneChangedError(SkillError):
    """Two-phase localization: the scene changed between the image the brain
    named the object on and the fresh image its geometry comes from, again
    after one retry. Like SlowPerceptionError it is not detector flicker:
    no memory fallback, no motion recovery."""


class ReanchorError(SkillError):
    """Two-phase localization: SAM could not re-find the object the brain
    named on a fresh image (no mask, an empty one, one off the brain's box,
    too few depth points), again after one retry with a fresh refer. Never
    replaced by the box: a box cloud includes table pixels."""


def _perception_stop(exc: BaseException, attempt: int) -> str | None:
    """The note of a grasp retry loop (pick_and_place, `_grasp_with_persistence`)
    that must end on this refusal with no automatic motion recovery, or None
    when the attempt is a miss to retry. Slow perception and a scene change
    end it at once (another attempt only repeats them); a failed re-anchor is
    a miss once (re-homing clears the arm out of the view), and ends it from
    the second attempt on."""
    if isinstance(exc, SceneChangedError):
        what = "the scene changed while the object was localized"
    elif isinstance(exc, StaleFrameError):
        what = "the camera stopped delivering frames"
    elif (isinstance(exc, SafetyViolation) and not isinstance(exc, MotionHalted)
          and str(exc).startswith("perception watchdog:")):
        # The arm's watchdog refused the attempt before any motion. It no
        # longer latches the e-stop (#414 review Important 1), which ended
        # the loop before; a re-home would be refused alike.
        what = "the perception watchdog refused the motion"
    elif isinstance(exc, SlowPerceptionError):
        what = "localization analysis expired"
    elif isinstance(exc, ReanchorError) and attempt >= 2:
        what = "the object could not be re-anchored on a fresh image after re-homing"
    else:
        return None
    return f"{what}; no automatic motion recovery"


#: Two-phase localization: frame0 -> frame1 limits within which the brain's
#: box from frame0 still describes frame1 (spec 2026-10-16 B1 step 4/6).
_REANCHOR_CAMERA_SHIFT_M = 0.002
_REANCHOR_CAMERA_TURN_DEG = 0.5
_REANCHOR_MAX_MOVE_M = 0.02


def _semantic_age_s(frame: Frame, analysis_s: float, max_age_s: float, what: str = "to name the object") -> float:
    """The brain's naming step (two-phase localization) and frame0's re-mask
    under their own cap `perception_loop.semantic_max_age_s`: nothing
    measured on this frame is acted on, so the 5 s analysed-frame limit
    applies to the fresh frame."""
    age = _frame_age_s(frame, max_age_s=float("inf"))
    if not age <= max_age_s:
        raise SlowPerceptionError(
            f"Perception took {analysis_s:.1f}s {what} on a frame that was fresh at "
            f"capture, so the semantic step is {age:.1f}s old (limit {max_age_s:g}s); the camera is "
            "live -- retry the observation"
        )
    return age


def _view_change(frame0: Frame, frame1: Frame) -> str | None:
    """Why `frame1` does not show the view `frame0` showed (another camera,
    other intrinsics, a camera pose moved past 2 mm / 0.5 deg), or None."""
    if frame0.rgb.shape[:2] != frame1.rgb.shape[:2] or not np.allclose(frame0.K, frame1.K):
        return "the camera intrinsics changed"

    def identity(frame):
        capture = getattr(frame, "capture", None)
        return (capture.get("source"), capture.get("camera")) if isinstance(capture, dict) else None

    if identity(frame0) != identity(frame1):
        return "the image came from another camera"
    T0, T1 = frame0.T_base_cam, frame1.T_base_cam
    if (T0 is None) != (T1 is None):
        return "the camera pose is known for one image only"
    if T0 is not None:
        T0, T1 = np.asarray(T0, dtype=float), np.asarray(T1, dtype=float)
        shift = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        if not shift <= _REANCHOR_CAMERA_SHIFT_M:
            return f"the camera moved {shift * 1000:.1f} mm"
        cos = (float(np.trace(T0[:3, :3].T @ T1[:3, :3])) - 1.0) / 2.0
        turn = math.degrees(math.acos(min(max(cos, -1.0), 1.0))) if math.isfinite(cos) else math.inf
        if not turn <= _REANCHOR_CAMERA_TURN_DEG:
            return f"the camera turned {turn:.2f} deg"
    return None


class _SceneChanged(Exception):
    """One two-phase attempt saw the scene change (retried once)."""


class _ReanchorFailed(Exception):
    """One two-phase attempt could not re-anchor (retried once)."""


def _rounded(value, digits: int):
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _fresh_camera_frame(camera, previous: Frame | None = None) -> Frame:
    from ..perception.freshness import read_frame_after

    if hasattr(camera, "latest"):
        previous = camera.latest() or previous
    try:
        frame = read_frame_after(camera, after=previous, timeout_s=5.0)
    except (TimeoutError, ValueError) as exc:
        raise SkillError("Camera did not provide a fresh frame; wait for reconnection and try again") from exc
    _frame_age_s(frame)
    return frame


class _PostPlaceRetreatPlanError(SkillError):
    """Retraction cannot be planned; stop before placement or recovery motion."""


class _PreCarryLiftError(SkillError):
    """Carry clearance is unavailable; retain the grasp without a home sweep."""


def _squeeze_fraction(raw, name: str) -> float:
    """A post-contact hold squeeze: a fraction of jaw travel in (0, 1), else
    SkillError naming the profile key (raised before any jaw command)."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise SkillError(f"{name} must be a fraction in (0, 1), got {raw!r}")
    return value


def _hold_stall_timeout_s(gripper_cfg) -> float:
    """`gripper.hold_stall_timeout_s` (default 4 s): how long the post-contact
    hold waits for the jaws to stall. Finite and positive, else SkillError
    (a NaN deadline would never expire)."""
    raw = gripper_cfg.get("hold_stall_timeout_s", 4.0) if gripper_cfg is not None else 4.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if not np.isfinite(value) or value <= 0.0:
        raise SkillError(f"gripper.hold_stall_timeout_s must be finite and positive, got {raw!r}")
    return value


def _every_candidate_too_wide(err: str) -> bool:
    """True when a grasp failure says EVERY candidate exceeded the jaw span.

    Over-width is an object property: re-scanning cannot shrink it, so the
    persistence loops stop on it (ROADMAP leftover #6). The selector states
    it explicitly (`ALL_TOO_WIDE_MARKER`, with narrowest width vs jaw span).
    Without the marker, a message is read as all-too-wide only if EVERY
    listed reason is a width reason -- a single IK/vetting reason means a
    fresh plan may still succeed. (The selector lists non-width reasons
    first, so its 4-reason truncation cannot hide them.)"""
    if ALL_TOO_WIDE_MARKER in err:
        return True
    if "> gripper max" not in err or "IK failed" in err:
        return False
    _, sep, tail = err.partition("no executable grasp: ")
    if not sep:
        return True  # a bare width reason from another caller
    reasons = [r for r in tail.split("; ") if r.strip()]
    return bool(reasons) and all("> gripper max" in r for r in reasons)


class SkillRuntime:
    def __init__(
        self,
        camera,
        depth_provider,
        detector,
        extrinsics: Extrinsics,
        kin,
        safe_arm,
        memory: EpisodicMemory,
        beliefs: BeliefStore,
        trace: TraceLogger,
        cfg,
    ):
        self.camera = camera
        self.depth = depth_provider
        self.detector = detector
        self.extrinsics = extrinsics
        self.kin = kin
        self._arm = safe_arm
        #: optional ArmRig (set by the app wiring). None = single arm.
        self.arm_rig: ArmRig | None = None
        #: per-call arm override, set by execute() for the duration of one
        #: skill call. Thread-local because the MCP server answers stop
        #: frames on a reader thread while a skill runs on a worker: a plain
        #: attribute would let one thread's arm selection leak into another's.
        self._arm_override = threading.local()
        self._motion_scope = threading.local()
        # Optional task cancellation hook supplied by the orchestrator.
        self.motion_cancel_requested = None
        self.memory = memory
        # The one choke point of the rules' triggers in memory (#388 re-review, Minor 2): about
        # 30 notes quote the agent's own free text ({object!r}, {label!r}, {query!r}), and MCP
        # world_state / task_memory / the digest read every line back to the agent, so every
        # line and label this runtime's memory records has them withheld, as a result has.
        if callable(getattr(memory, "set_text_filter", None)):
            memory.set_text_filter(_without_rule_triggers)
        self.beliefs = beliefs
        self.trace = trace
        self.cfg = cfg
        self.last_frame: Frame | None = None
        self.last_annotated_frame: Frame | None = None
        self.held_object: str | None = None
        self._held_arm: str | None = None
        # (3,) base-frame vector from TCP to the held object, or None. Set at
        # grasp, refreshed by re-observation before a place.
        self._held_offset = None
        self._held_support_offset_m = None
        #: optional external pose lookup, set by attach_verifier
        self._object_pose = None
        self._held_det_label: str | None = None
        self._held_color: str | None = None
        #: jaw opening MEASURED when the held object was taken (stall after
        #: the lift, or at promotion/adoption), metres; None = unknown. It
        #: decides whether a closed-jaw reading can mean a slip (#2.5).
        self._held_width_m: float | None = None
        #: optional WorldWatcher (set by the app wiring); paused during motion
        self.watcher = None
        #: ExtrinsicDriftMonitor per monitored eye-to-hand camera (app wiring)
        self.drift_monitors = []
        self._reset_observation_pending = False
        #: RGB frame captured immediately before the current motion skill, for
        #: CaP-X visual differencing. Set by execute(); None between motions.
        self._pre_motion_frame = None
        #: LiveViewController (set by the app wiring): the on-demand browser
        #: dashboard. None in bare/unit-test runtimes.
        self.live_view = None
        #: the live StreamServer while the view is open, else None
        self.stream_server = None
        #: B47 WristNarrator (apps/wrist_narration.py), set by the app wiring
        #: only with `stream.wrist_narration: true` AND a wrist stream; None
        #: = execute() dispatches exactly as before.
        self.wrist_narrator: Any = None
        #: natural-language task currently executing (dashboard narration)
        self.current_task: str | None = None
        #: where the persistent world model is written on shutdown (set by the
        #: app wiring). None = persistence disabled, which is the default for
        #: bare/unit-test runtimes.
        self.beliefs_path = None
        # B43: where the visual index persists (memory.persist_episodic); None = off
        self.episodic_path = None
        #: dispatch tier that served the last command ("reflex" |
        #: "experience" | "llm" | "mcp-host"), for the dashboard "via:" chip
        self.last_path: str | None = None
        #: keyframe bytes loaded by the last SUCCESSFUL `recall_step` call:
        #: [{"tag": "before"|"after", "path": <run-dir relative>, "caption",
        #: "jpeg": bytes}]. Kept off the JSON result (the planner receives
        #: json.dumps of it) and served as image content by the MCP server /
        #: attached to the next planner turn by the orchestrator. Cleared at
        #: the start of every recall so a bad index never leaves a stale frame.
        self.last_recalled_frames: list[dict] = []
        #: which dispatch tier is issuing the CURRENT skill calls (reflex /
        #: experience / llm / mcp-host); set by the dispatcher around its
        #: calls and written into every trace row so an offline judge can
        #: score progress per tier (eval/progress_judge.py per_tier()).
        self.current_tier: str | None = None
        #: pre-motion plausibility advisory for the CURRENT call, handed over
        #: by the orchestrator the same way `current_tier` is (set before
        #: execute(), cleared after) so the trace row and the result carry it
        #: as `plausibility`. ADVISORY ONLY: nothing on the dispatch path reads
        #: it -- it never gates, delays or alters a motion (agent/milestones.py).
        self.pending_plausibility: dict | None = None
        #: monotonic time the current top-level MOTION skill started; while
        #: the arm moves the WorldWatcher is paused, so belief ages measured
        #: from "now" are artificially inflated -- staleness checks measure
        #: from this epoch instead.
        self._motion_t0: float | None = None
        #: last time _reobserve completed a fresh scan DURING the current
        #: motion skill: belief-fallback staleness must advance with it, or
        #: a belief that was fresh at task start stays "fresh" through 8
        #: retries even after every re-scan failed to see the object.
        self._last_reobserve_t: float | None = None
        #: Derived envelope features for the CURRENT top-level skill call
        #: (memory/envelope.py DERIVED_FEATURES): a scratchpad the running
        #: skill fills from the runtime's own measurements -- FK of the
        #: joint vector read back when the jaws closed, the localized fix --
        #: via `_note_measurement()`. Owned by the outermost `_execute_skill`
        #: (same ownership pattern as `_motion_t0`), so a grasp nested inside
        #: pick_and_place credits the call the agent actually made. None
        #: between calls; a feature the skill never wrote is recorded by the
        #: envelope as MISSING, never defaulted here.
        self._call_measurements: dict | None = None
        #: Object crops localized during the current outermost call, keyed by
        #: detector label ({label: BGR crop}), for the memory embedder
        #: (ROADMAP #7). Only ever a dict while a call is in flight AND
        #: `self.memory.embedder` is set; None otherwise, so a runtime without
        #: an embedder never copies a pixel for it.
        self._call_crops: dict | None = None
        self._graspgenx = None  # lazy GraspGenXPlanner (grasp.backend)
        #: Optional profiles use a short retry cooldown after a server error.
        #: Required profiles fail visibly and retry on the next command.
        self._graspgenx_down = False
        #: lazy HugPlanner (`grasp.backend: hug`, opt-in) and its outage
        #: latch; same required/optional contract as GraspGen-X above
        self._hug = None
        self._hug_down = False
        #: the opt-in VLA executor (`grasp.executor: vla`, B49;
        #: grasping/vla_executor.py), built and probed by build_runtime. None
        #: with the analytic executor (the default): nothing VLA exists then.
        self.vla_executor = None
        #: which grasp planner actually produced the last candidate list:
        #: "obb" | "graspgenx (learned 6-DoF)" | "graspgenx-stub (...)" |
        #: "obb (graspgenx down)". None until the first grasp. Surfaced by
        #: `backends()` for the demo banner and the run summary.
        self.grasp_planner_used: str | None = None
        self._grounder = None  # lazy VLMGrounder (cfg "grounder", 2nd filter)
        # Fake-RL grasp memory (RPent/Harness-VLA pattern): learns which grasp
        # geometry works per object profile from past attempts (wins AND
        # failures), persisted so accuracy improves across sessions.
        from ..memory.grasp_memory import GraspOutcomeMemory
        from pathlib import Path as _Path
        import os as _os
        # CASCADE_GRASP_MEMORY_PATH overrides, same contract as
        # CASCADE_BELIEFS_PATH: shared-default persistent state leaks between
        # a developer's real runs and the test suite in both directions
        # (observed: diagnostic air-grasps recorded to ~/.cascade poisoned the
        # orchestrator E2E's grasp priors). conftest pins both into tmp_path.
        _gm_path = (_os.environ.get("CASCADE_GRASP_MEMORY_PATH")
                    or cfg.grasp.get("memory_path", "~/.cascade/grasp_memory.json"))
        self.grasp_memory = GraspOutcomeMemory(
            path=_Path(str(_gm_path)).expanduser())
        # Harness-VLA (arXiv:2607.08448) generalised to EVERY primitive: the
        # learned operating envelope + failure model, fed from the same
        # outcomes and injected into the agent's context by the orchestrator.
        from ..memory.envelope import OperatingEnvelope
        _env_path = cfg.get("memory", {}).get(
            "envelope_path", "~/.cascade/envelope.json") if hasattr(cfg, "get") else None
        _env_path = _os.environ.get("CASCADE_ENVELOPE_PATH") or _env_path
        self.envelope = OperatingEnvelope(
            path=_Path(str(_env_path or "~/.cascade/envelope.json")).expanduser())
        # Pigey (arXiv:2607.21725): verify each primitive's physical effect
        # against a channel the actuator does not own. Wired lazily by the
        # app (needs the sim bridge / belief store) via attach_verifier().
        self.effects = None
        from ..agent.task_effects import TaskEffects
        self._task_effects = TaskEffects()
        g = cfg.arm.gripper
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._max_width = float(g.get("max_width_m", 0.09))
        # Optional vocabulary restriction. Empty (the default) means
        # open-world: the detector reports whatever it sees, so an object
        # nobody listed still reaches the agent. A non-empty list is a CLOSED
        # SET and hides everything else -- benchmarks only, never the booth.
        self._default_classes = list(cfg.get("detect_classes") or []) or None
        self._workspace = WorkspaceFilter.from_config(cfg.get("workspace_filter"))
        # Learned point segmenter for pixel addressing, loaded on first use.
        # Off unless `segmenter.enabled`. When absent, `fix_at_pixel` falls
        # back to depth connectivity, which cannot separate touching objects.
        scfg = cfg.get("segmenter") or {}
        if scfg.get("enabled"):
            from ..perception.segmenter import PointSegmenter, configured_segmenter

            # The same strings as the brain detector's BoxSegmenter
            # (apps/demo._make_detector): both share one loaded SAM.
            model_path, device = configured_segmenter(scfg)
            self._segmenter = PointSegmenter(model_path=model_path, device=device)
        else:
            self._segmenter = None

    # ── which arm a skill drives ─────────────────────────────────────────

    #: Class-level defaults so the property below survives a runtime built
    #: with `SkillRuntime.__new__` -- several tests construct partial
    #: runtimes that way and only set the handful of fields they exercise.
    _arm = None
    _arm_override = None
    #: ((camera stream, watchdog feed), ...) per rig camera, set by
    #: `apps.demo.build_runtime` (#412's `WatchedCamera.watchdog_feed`): whose
    #: perception watchdogs a frame of that camera may keep fresh. A feed of
    #: None (one arm), an unlisted camera or a runtime built any other way:
    #: the running arm's own harness (`_mark_perception_fresh`).
    _watchdog_feeds = ()
    _motion_scope = None
    motion_cancel_requested = None
    arm_rig = None
    _call_measurements = None
    #: Gripper close commands this runtime completed (`_close_two_stage`),
    #: monotonic. With each harness's `motions_begun` it is the verifier's
    #: "did this call act at all" signal (`_actuation_counts`).
    _gripper_closes_commanded = 0
    #: RELEASE opens this runtime completed, monotonic: counted after the
    #: open command returns in open_gripper, place_at (its release and the
    #: observed-slip drop) and throw. A handover keeps holding, so it ends
    #: through open_gripper. Grasp-internal opens (pre-grasp, re-open after
    #: a miss, screw strokes) are not releases and are not counted. The
    #: verifier says "released" / "opened" only after a counted open.
    _gripper_opens_completed = 0
    #: LinkSelfMask | None (B39): set by the app wiring when
    #: `workspace_filter.link_self_mask` is on; see perception/link_mask.py.
    _link_self_mask: "LinkSelfMask | None" = None
    #: ASPIRE note library snapshot (`skills.notes.LibrarySnapshot`), taken once
    #: by `apps.demo.build_runtime`; None for runtimes built any other way (bare
    #: `__new__` fixtures, MCP fakes), which serve no note (preflight R29).
    learned = None
    #: ASPIRE sim retry ledger (`agent.sim_retry.SimRetryLedger`): built by
    #: `apps.demo.build_runtime` only for a sim profile with every switch on
    #: (`ledger_for`); None everywhere else (preflight R29). With None nothing
    #: of the sim retry runs and every call runs exactly as before. Its per-call
    #: state is `_SimRetryState`, on the calling thread's motion scope.
    sim_retry = None
    #: ASPIRE learning surface (`agent.learning_surface.LearningState`, spec §4.8):
    #: built by `apps.demo.build_runtime` only for a sim profile with
    #: `agent.learning_surface: true` (`learning_state_for`); None everywhere else
    #: (preflight R29), and then no result ever gets a learning line.
    learning_state = None

    #: The PRIMARY arm's Kinematics and Cfg (the `kin` / `cfg` setters write
    #: these), and the other arms' bindings by rig name (build_runtime's
    #: `register_arm_bindings`). Class defaults for `__new__` runtimes.
    _kin = None
    _cfg = None
    _arm_bindings = None
    #: The rig name of the arm whose jaws hold `held_object` (spec D9 / R5):
    #: set wherever `held_object` is set, None wherever it is cleared. None
    #: with an object held (a runtime double, a hand-set flag): this arm's.
    _held_arm = None

    # ── the running arm's frame (spec 2026-10-17 base_pose runtime frame) ──
    #
    # `execute(arm=X)` binds X's ArmFrame (Kinematics, resolved profile,
    # gripper constants, base_pose) for that one call, thread-locally beside
    # the arm selection (`_arm_override.frame`). The properties below read it;
    # unbound reads (other threads, between calls) and the primary's own calls
    # see the primary. Setters always write the primary.

    def _bound_frame(self):
        override = getattr(self, "_arm_override", None)
        return None if override is None else getattr(override, "frame", None)

    @property
    def kin(self):
        frame = self._bound_frame()
        return self._kin if frame is None else frame.binding.kin

    @kin.setter
    def kin(self, value) -> None:
        self._kin = value

    @property
    def cfg(self):
        frame = self._bound_frame()
        return self._cfg if frame is None else frame.binding.cfg

    @cfg.setter
    def cfg(self, value) -> None:
        self._cfg = value

    @property
    def T_world_base(self):
        """The running arm's base_pose as a 4x4 (table <- base), or None (no
        base_pose or the identity: no transform anywhere). Unbound: the
        primary's."""
        frame = self._bound_frame()
        if frame is not None:
            return frame.T_world_base
        return arm_frame.world_from_base(getattr(self, "_arm", None))

    def _bound_grip(self):
        frame = self._bound_frame()
        return None if frame is None else frame.binding.grip

    @property
    def _grip_open(self) -> float:
        grip = self._bound_grip()
        return self._primary_grip_open if grip is None else grip[0]

    @_grip_open.setter
    def _grip_open(self, value) -> None:
        self._primary_grip_open = value

    @property
    def _grip_closed(self) -> float:
        grip = self._bound_grip()
        return self._primary_grip_closed if grip is None else grip[1]

    @_grip_closed.setter
    def _grip_closed(self, value) -> None:
        self._primary_grip_closed = value

    @property
    def _max_width(self) -> float:
        grip = self._bound_grip()
        return self._primary_max_width if grip is None else grip[2]

    @_max_width.setter
    def _max_width(self, value) -> None:
        self._primary_max_width = value

    def register_arm_bindings(self, bindings) -> None:
        """The Kinematics and resolved profile of each non-primary arm, by
        rig name (spec D3). A selected arm without one is refused (D4)."""
        merged = dict(getattr(self, "_arm_bindings", None) or {})
        merged.update(bindings)
        self._arm_bindings = merged

    def _resolve_arm_frame(self, effective_arm, resolved_arm) -> arm_frame.ArmFrame:
        """The frame execute() binds for one call on `effective_arm`. The
        primary uses the runtime's own `kin` / `cfg` objects; any other arm
        needs a registered binding. Raises ArmFrameError (refuse the call)
        when the binding is missing or the base_pose is not rigid. Reads
        instance dicts only: never materializes a LazyArm."""
        if effective_arm is getattr(self, "_arm", None):
            binding = arm_frame.ArmBinding(self._kin, self._cfg, None)
        else:
            binding = (getattr(self, "_arm_bindings", None) or {}).get(resolved_arm)
            if binding is None:
                raise arm_frame.ArmFrameError("it has no kinematics/profile bound to this runtime")
        return arm_frame.ArmFrame(resolved_arm, arm_frame.world_from_base(effective_arm), binding)

    def _to_arm(self, p):
        """Table-frame point(s) in the running arm's base frame."""
        return arm_frame.to_arm(self.T_world_base, p)

    def _to_world(self, p):
        """Running-arm-frame point(s) in the table frame."""
        return arm_frame.to_world(self.T_world_base, p)

    def _dir_to_world(self, v):
        return arm_frame.dir_to_world(self.T_world_base, v)

    def _cam_to_arm(self, T_table_cam):
        return arm_frame.cam_to_arm(self.T_world_base, T_table_cam)

    def _frame_in_arm(self, frame, extrinsics=None):
        """`frame` with its camera pose in the running arm's frame
        (`extrinsics` None = this runtime's)."""
        T = self.T_world_base
        if T is None:
            return frame
        return arm_frame.frame_to_arm(T, frame, self.extrinsics if extrinsics is None else extrinsics)

    def _fix_to_arm(self, fix):
        return arm_frame.fix_to_arm(self.T_world_base, fix)

    def _fix_to_world(self, fix):
        return arm_frame.fix_to_world(self.T_world_base, fix)

    def _belief_to_arm(self, belief):
        """A belief (table-frame memory) as a target in the running arm's
        frame: position, remembered points and `top_z` move (level mounts:
        `top_z` - base z); label, extent, colour and times are kept. Without
        a base_pose: the belief object itself."""
        T = self.T_world_base
        if T is None:
            return belief
        from dataclasses import replace

        position = np.asarray(_host(belief.position), dtype=float).reshape(3)
        points = _host(getattr(belief, "points", None))
        return replace(
            belief,
            position=arm_frame.to_arm(T, position),
            points=None if points is None else arm_frame.to_arm(T, np.asarray(points, dtype=float).reshape(-1, 3)),
            top_z=None if belief.top_z is None else float(
                arm_frame.to_arm(T, [position[0], position[1], float(belief.top_z)])[2]),
        )

    def _air_grasp_frac(self) -> float:
        """The running arm's `grasp.air_grasp_frac` (the verifier's threshold)."""
        return float(self.cfg.grasp.get("air_grasp_frac", 0.04))

    def _table_z_world(self) -> float:
        """The running arm's `safety.table_z` (its base frame) as a table-frame
        height. Without a base_pose: `table_z` itself."""
        return arm_frame.table_z_in_world(self.T_world_base, float(self.cfg.safety.get("table_z", 0.0)))

    def _height_to_world(self, z: float) -> float:
        """A running-arm-frame height as a table-frame height (level mounts;
        agent text quotes table-frame numbers). Without a base_pose: z."""
        return arm_frame.table_z_in_world(self.T_world_base, z)

    def _xy_to_world(self, xy):
        """A running-arm-frame (x, y) as a table-frame (x, y) of floats (level
        mounts: the height does not move it). Without a base_pose: `xy` itself."""
        T = self.T_world_base
        if T is None:
            return xy
        p = arm_frame.to_world(T, [float(xy[0]), float(xy[1]), 0.0])
        return float(p[0]), float(p[1])

    def _xy_to_arm(self, xy):
        """A table-frame (x, y) in the running arm's base frame, as floats (level
        mounts). Without a base_pose: `xy` itself."""
        T = self.T_world_base
        if T is None:
            return xy
        p = arm_frame.to_arm(T, [float(xy[0]), float(xy[1]), 0.0])
        return float(p[0]), float(p[1])

    def _running_arm_name(self) -> str | None:
        """The rig name of the arm this call drives, as execute() names it
        (identity search, no arm attribute read): "default" without a rig."""
        rig = getattr(self, "arm_rig", None)
        if rig is None:
            return "default"
        running = self.arm
        return next((key for key, arm in rig.arms.items() if arm is running), None)

    def _held_by_another_arm(self) -> str | None:
        """The name of the arm holding `held_object` when it is NOT the arm
        this call drives (spec D9 / R5), else None. A hold without a recorded
        arm counts as this arm's, as before."""
        held_arm = getattr(self, "_held_arm", None)
        if not getattr(self, "held_object", None) or held_arm is None:
            return None
        return None if held_arm == self._running_arm_name() else held_arm

    def _check_holding_arm(self) -> None:
        """Refuse a release on an arm that does not hold the object (spec D9,
        §5): its jaws would open on nothing while the holding arm keeps it.
        Called first, before any work."""
        other = self._held_by_another_arm()
        if other is not None:
            raise SkillError(f"{self.held_object!r} is held by arm {other!r}, not {self._running_arm_name()!r}; "
                             "no motion was sent")

    def _refuse_on_another_arms_provisional(self, then: str) -> None:
        """Refuse a grasp on this arm while ANOTHER arm's grasp, stopped after
        its close, left a provisional hold (R5; Task 5 review Important 1).
        The marker is one slot: this arm's close would take it and its
        promotion or air verdict clear it, and the arm whose jaws may hold the
        object would then open them at its next reset_scene. Called first,
        before any perception or command; names that arm and the way out on it
        (`then`: the call to make after it)."""
        prov = getattr(self, "_held_provisional", None)
        if prov is not None and SkillRuntime._provisional_on_another_arm(self, prov):
            raise SkillError(self._held_elsewhere_refusal(prov[0], self._arm_name_of(prov.arm), provisional=True,
                                                          then=then, keep=len("SkillError: ")))

    def _already_holding_refusal(self, then: str) -> str:
        """The "already holding" refusal of a grasp skill (`then`): on an arm
        that does not hold the object it names the holding arm and the way out
        on it (Task 5 review Minor 2), else today's text."""
        holder = self._held_by_another_arm()
        if holder is not None:
            return self._held_elsewhere_refusal(self.held_object, holder, then=then, keep=len("SkillError: "))
        return f"already holding {self.held_object!r}; place it first"

    def _grasp_memory_label(self, label: str) -> str:
        """The grasp-outcome memory's label for `label` on the running arm
        (Task 5 review Minor 6). On a rig of two or more arms the arm's rig
        name joins it: the memory learns per object profile, and one arm's
        out-of-reach attempts ("pregrasp IK failed" counts as pregrasp_unsafe,
        +8 mm of grasp height per loss) must not raise another arm's grasp of
        that object. One arm keeps the bare label (its rows as before)."""
        rig = getattr(self, "arm_rig", None)
        name = SkillRuntime._running_arm_name(self) if len(getattr(rig, "arms", None) or ()) > 1 else None
        return label if name is None else f"{label}@{name}"

    def _model_release_refusal(self, model_plan=None) -> None:
        """R9: the MuJoCo placement and withdrawal observers admit only an
        articulation root at the kinematic origin, so a base_pose arm with a
        region plan or a declared model release (`mj_delivery_area`,
        `mj_release_*`) is refused before any command. Without a base_pose:
        nothing (today's path)."""
        if self.T_world_base is None:
            return
        data = plain(self.cfg.arm)
        declared = isinstance(data, dict) and any(
            value is not None and (key == "mj_delivery_area" or str(key).startswith("mj_release_"))
            for key, value in data.items())
        if model_plan is not None or declared:
            # The way out (Task 5 review Minor 3): the failure line quotes it.
            raise SkillError("region placement / model release is not supported on a base_pose arm; "
                             "no motion was sent. If the jaws hold something, release it with open_gripper "
                             "(it opens where the gripper is) or reset_scene (it opens at home)")

    @property
    def arm(self):
        """The SafeArm the running skill drives.

        Normally the primary. `execute()` rebinds this for the duration of a
        single call when the caller passes `arm="<name>"`, which is why the
        30-odd skills can keep saying `self.arm` and none of them had to grow
        an `arm` parameter (56 call sites, and every one is inside a motion
        sequence where a half-converted skill would drive two robots).

        The override is thread-local: the MCP server answers stop frames on a
        reader thread while a skill runs on a worker, so a plain attribute
        would let one thread's arm selection leak into another's.
        """
        override = getattr(self, "_arm_override", None)
        if override is not None:
            selected = getattr(override, "arm", None)
            if selected is not None:
                return selected
        return self._arm

    @arm.setter
    def arm(self, value) -> None:
        """Rebind the PRIMARY arm (tests and harnesses do this directly)."""
        self._arm = value

    def _arm_harnesses(self) -> list:
        """The harness of every arm this runtime can command: the rig's arms,
        then the bound primary, each harness once (None for an arm without
        one). Registry only: read from each arm's instance dict, never
        `getattr`, so bookkeeping never reaches a backend's __getattr__ (that
        can probe or materialize a standby LazyArm)."""
        arms = list(self.arm_rig.arms.values()) if getattr(self, "arm_rig", None) is not None else []
        arms.append(getattr(self, "_arm", None))
        out, seen = [], set()
        for arm in arms:
            if arm is None:
                continue
            harness = (getattr(arm, "__dict__", None) or {}).get("harness")
            if id(harness) not in seen:
                seen.add(id(harness))
                out.append(harness)
        return out

    def _contact_harnesses(self) -> list:
        """Harnesses of every arm that files contact-monitor reports."""
        return [h for h in self._arm_harnesses()
                if h is not None and callable(getattr(h, "drain_contact_reports", None))]

    def _clearance_harnesses(self) -> list:
        """Harnesses of every arm that files clearance-refusal reports."""
        return [h for h in self._arm_harnesses()
                if h is not None and callable(getattr(h, "drain_clearance_reports", None))]

    def _skill_halt_baselines(self, arm) -> list:
        """(harness, its halt generation now) for `arm` and every rig arm:
        taken when execute() starts a motion skill (registry only, like
        `_contact_harnesses`: never materializes a standby LazyArm). Harness
        doubles without `skill_halt_baseline` (class-defined) are skipped."""
        arms = [arm, getattr(self, "_arm", None)]
        if getattr(self, "arm_rig", None) is not None:
            arms += list(self.arm_rig.arms.values())
        out, seen = [], set()
        for candidate in arms:
            harness = (getattr(candidate, "__dict__", None) or {}).get("harness")
            if (harness is None or id(harness) in seen
                    or not callable(getattr(type(harness), "skill_halt_baseline", None))):
                continue
            seen.add(id(harness))
            out.append((harness, harness._halt_generation))
        return out

    @staticmethod
    def _skill_halt_scope(baselines):
        """Hold each harness's skill baseline while the skill body runs."""
        import contextlib

        with contextlib.ExitStack() as stack:
            for harness, generation in baselines:
                stack.enter_context(harness.skill_halt_baseline(generation))
            return stack.pop_all()

    @staticmethod
    def _skill_halt_reason(baselines) -> str | None:
        """The operator's halt that arrived at one of these harnesses since
        the skill started (`_skill_halt_baselines`), else None."""
        for harness, generation in baselines:
            if harness._halt_generation != generation:
                halted = harness.halted
                return f"halted: {halted}" if halted else "halt received during route planning or execution"
        return None

    @staticmethod
    def _skill_halt_estopped(baselines) -> bool:
        """A harness that took a halt since the skill started
        (`_skill_halt_reason`) also has its e-stop latched: both stops came."""
        return any(harness._halt_generation != generation and getattr(harness, "estopped", False)
                   for harness, generation in baselines)

    def _select_arm(self, name):
        """Resolve an arm name to a SafeArm, or raise SkillError.

        Unknown names must fail loudly: silently falling back to the primary
        would run a motion on the wrong robot, which on a two-arm table is a
        collision rather than a wrong answer.
        """
        if name is None:
            return None
        # Two spellings of "the arm you have" that chat hosts actually send,
        # measured on a fresh-clone run: `arm=""` (the model fills every
        # optional schema field) and `arm="default"` (the literal name
        # `list_arms` itself reports for a single-arm rig). Refusing them cost
        # two failed tool calls per pick before the model guessed the profile
        # name. Both mean "primary"; a WRONG name still fails loudly below.
        if isinstance(name, str) and name.strip().lower() in ("", "default", "primary"):
            return None
        if self.arm_rig is None:
            raise SkillError(
                f"no arm named {name!r}: this runtime drives a single arm"
            )
        try:
            return self.arm_rig.get(str(name))
        except KeyError as e:
            raise SkillError(str(e)) from e

    # ── ASPIRE sim retry (spec §4.5) ─────────────────────────────────────

    def _sim_retry_ledger(self, name):
        """This call's retry ledger: the runtime's (None on any profile that
        built none), for a motion skill only, and only for the outermost motion
        call of this thread (a nested call runs inside the outer one's)."""
        ledger = self.sim_retry
        if ledger is None or name not in _MOTION_SKILLS:
            return None
        scope = getattr(self, "_motion_scope", None)
        if getattr(scope, "context", None) is not None:
            return None
        return ledger

    def _stop_token(self) -> tuple:
        """The rig's stop token (R7): every arm harness's halt / e-stop count
        (`_observation_cancel_generation`: `halt()` and `estop()` raise it, a
        Resume or `reset_estop` never lowers it). Registry reads only (the
        harnesses `_arm_harnesses` lists), never a backend."""
        return tuple(getattr(harness, "_observation_cancel_generation", None)
                     for harness in self._arm_harnesses())

    def _people_refusal(self, name, args, refusal, trace_context) -> dict:
        """A call the people guard refused (`people_guard.check_call`): the
        failure line, its own trace row (filed `safety:people`), returned
        before any status line, keyframe, retry ledger or command. Nothing
        moved, so there is no move_home step and no envelope outcome."""
        result = _with_failure_instruction(_normalize_outcome(people_guard.refusal_result(name, refusal), name))
        self.trace.record(name, args, result, 0.0, tier=self.current_tier,
                          context={**trace_context, "people_guard": dict(refusal.evidence)})
        return result

    def _sim_retry_refusal(self, name, args, refusal: dict, trace_context: dict) -> dict:
        """The refusal `begin` returned (the dict `precheck` returns): a used retry
        (RETRY_USED) or a stopped one (the operator stopped the arm after the
        refusal: RETRY_STOPPED_NEXT), after the FAILED line, with its own trace
        row (R28: acceptance, admission and grading count it), returned before
        any status line, keyframe or motion."""
        result = _with_retry_text(_with_failure_instruction(_normalize_outcome(dict(refusal), name)), name)
        mark = {"refused": True}
        if (refusal.get("sim_retry") or {}).get("stopped") is True:
            mark["stopped"] = True
        self.trace.record(name, args, result, 0.0, tier=self.current_tier,
                          context={**trace_context, "sim_retry": mark})
        return result

    def _home_before_retry(self, generation) -> None:
        """THE retry's first step when the refused attempt left the arm in
        place: the agent-callable move_home with all its gates, carrying the
        skill's halt baseline `generation`; a halt since then (during or right
        after the home) ends the call as MotionHalted before any pick."""
        self.skill_move_home(_halt_generation=generation)
        self.arm.harness._check_halt_generation(generation)

    def _sim_retry_finish(self, ledger, name, args, result, trace_context, *, retry, retry_home, exc_type, state,
                          arm, harness, baselines, stop_token0) -> dict:
        """Hand the finished call to the ledger, read at the END of the call (the
        hold, a latch, a halt since the skill started, a pending episode, the
        rig's stop token), then compose the agent text and mark the trace row
        (the retry, its home, a ledger that switched itself off, a failed call an
        operator stop ended). `stop_token0`: the stop token `begin` saw."""
        held = bool(self._jaws_may_hold())
        latched = self._stop_latched(arm)
        halted = self._halted_since_skill_start(baselines)
        pending = _episode_pending(harness)
        token = self._stop_token()
        result = ledger.finish(name, args, result, exc_type=exc_type, held=held, latched=latched, halted=halted,
                               episode_pending=pending, refused_grasps=state.refused_grasps, retry=retry,
                               stop_token=token)
        ok = isinstance(result, dict) and result.get("ok") is True
        ledger.note_motion(name, ok)
        if name == "reset_scene" and ok:
            ledger.clear("reset_scene")
        result = _with_retry_text(result, name)
        mark = {"retry": True, "auto_home": retry_home} if retry is not None else {}
        if ledger.off:
            mark["off"] = True
        # a failed call an operator stop ended (latched now, a halt since the skill
        # started, or a halt / e-stop during the call, even one cleared again: the
        # ledger's own rule, R7): ASPIRE never learns from it, also when its error
        # names only the skill's own refusal (`aspire.ended_in_stop`; A8 follow-ups fix)
        if not ok and (latched or halted or token != stop_token0):
            mark["ended_in_stop"] = True
        if mark:
            trace_context["sim_retry"] = mark
        return result

    # ── ASPIRE learning surface (spec §4.8) ──────────────────────────────

    def _learning_for(self, name):
        """This call's learning state: the runtime's (None on every profile that
        built none: a real profile, the switch off), for a motion skill only, and
        only for the outermost motion call of this thread (as the retry ledger)."""
        state = self.learning_state
        if state is None or name not in _MOTION_SKILLS:
            return None
        scope = getattr(self, "_motion_scope", None)
        if getattr(scope, "context", None) is not None:
            return None
        return state

    def _grant_learning_line(self, learning, name, args, result, trace_context, *, retry, arm, baselines,
                             stop_token0) -> None:
        """Decide the one line this call's result grants (spec §4.8:
        `learning_surface.learning_line`) and leave the decision on this thread's
        motion scope (`_LearningGrant`): `execute()` delivers it as its last step
        (`_with_learning_grant`), after the rule-trigger scrub, so only the
        runtime's own field and instruction ever stand in a result. A call an
        operator stop met -- latched, a halt since the skill started, or the rig's
        stop token changed since `stop_token0` (an e-stop raised and reset) -- gets
        none, with or without a retry ledger; that reading is taken here and again
        on delivery (A9/A10 fix, Minor 2). The line is built from runtime texts and
        the served note's title and tag only, never from the agent's arguments; one
        that would read as a rule trigger is no line. Marks the trace row's
        `learning` with the decided line; a line a stop withdraws on delivery is
        marked withheld there (`_mark_line_withheld`). Never fails the call."""
        from ..agent import learning_surface as surface

        def stopped() -> bool:
            return bool(self._stop_latched(arm) or self._halted_since_skill_start(baselines)
                        or self._stop_token() != stop_token0)

        try:
            halted = stopped()
            retried = retry is not None and name == "pick_and_place"
            line = None
            if not halted:
                matched = None if retried else surface.matched_note(learning.snapshot, name, args)
                line = surface.learning_line(
                    retried_placed=retried, result=result, matched=matched, sha=learning.snapshot.sha,
                    grasps_refused=retried and retry.refusal == "safety_refusal" and bool(retry.refused_grasps))
                if line is not None and not surface.grantable(line):
                    line = None  # fail closed: the line itself never reads as a rule's trigger
            ok = isinstance(result, dict) and result.get("ok") is True
            outcome = ("halted" if halted else "failed" if not ok
                       else "unverified" if not surface.succeeded(result)
                       else "placed" if name == "pick_and_place" else "succeeded")
            if line is not None:
                kind = "retry" if retried else "note"
            elif retry is not None:
                kind = "retry"  # THE retry without a line (failed, halted, unverified): the panel says so
            else:
                return
            self._motion_scope.learning_grant = _LearningGrant(learning, kind, line, outcome, stopped)
            if line is not None:
                trace_context["learning"] = {"kind": kind, "line": line}
        except Exception:  # noqa: BLE001 -- the learning surface never fails a call
            logger.exception("learning line for %s not granted", name)

    def attach_verifier(self, object_pose=None) -> None:
        """Enable postcondition checking (Pigey closed loop).

        ``object_pose`` is an optional ground-truth pose lookup -- in sim the
        Isaac bridge can read a RigidPrim directly, which beats perception.
        Without it the checker falls back to the belief store, and to CaP-X
        visual differencing when the belief would only be confirming itself.
        """
        from ..agent.effects import PostconditionChecker

        # Kept on the runtime too: `_held_object_offset` uses it to see where
        # the held object actually sits when the camera cannot (benchmark
        # oracle mode). Same channel, same provenance caveat.
        self._object_pose = object_pose
        self.effects = PostconditionChecker(
            object_pose=object_pose,
            belief_pose=self._belief_pose,
            gripper_frac=self._gripper_width_frac,
            reobserve=lambda: self._reobserve(frames=1),
            visual_diff=self._visual_diff,
            # Read per check, inside execute()'s bind (R4): the RUNNING arm's
            # table height, in the frame the verified poses are in.
            table_z_fn=self._table_z_world,
            # Likewise per check: the RUNNING arm's air-grasp threshold.
            air_grasp_frac_fn=self._air_grasp_frac,
            placement_check=self._verify_configured_placement,
            actuation=self._actuation_counts,
            holding=self._jaws_may_hold,
        )

    def _jaws_may_hold(self) -> bool:
        """The verifier's pre-call holding state: an object is held, or a
        provisional (unconfirmed) grasp may have one in the jaws."""
        return bool(self.held_object) or getattr(self, "_held_provisional", None) is not None

    def _actuation_counts(self) -> dict | None:
        """Monotonic counts of what this runtime commanded, for the verifier.

        ``arm_motions`` sums `motions_begun` over every arm's harness
        (`_arm_harnesses`): motions whose stream started (first approved
        command; SafeArm, `move_planned`, cartesian paths and planner-bound
        streams all approve each command in the harness). ``gripper_closes``
        counts completed close commands (`_close_two_stage`),
        ``gripper_opens`` completed release opens (`_gripper_opens_completed`).
        None when any arm lacks a counting harness (a test double): the
        verifier then claims neither "nothing happened" nor a close/release.
        """
        from ..safety.harness import SafetyHarness

        harnesses = self._arm_harnesses()
        if not harnesses or any(not isinstance(h, SafetyHarness) or type(h.motions_begun) is not int
                                for h in harnesses):
            return None
        return {"arm_motions": sum(h.motions_begun for h in harnesses),
                "gripper_closes": int(self._gripper_closes_commanded),
                "gripper_opens": int(self._gripper_opens_completed)}

    def _verify_configured_placement(self, label, destination):
        """Resolve the requested area in configuration, then read physics only."""
        region = self.cfg.arm.get("mj_delivery_area")
        region_requested = region is not None and (not destination or _names_drop_zone(destination))
        if region_requested:
            from ..sim.mujoco_placement import Region
            canonical = Region.parse(region).name
        elif not destination or _names_drop_zone(destination):
            canonical = self.cfg.grasp.get("drop_zone_name", "drop zone")
        elif str(destination).strip().lower() in _OPEN_BOX_WORDS:
            canonical = "open box"
        else:
            return None
        if not region_requested and canonical not in {"green square", "open box"}:
            return None
        reader = getattr(self._object_pose, "placement", None)
        # A configured region cannot fall back to the weaker point channel.
        if reader is None or (self.arm_rig is not None and len(self.arm_rig.arms) > 1):
            if region_requested:
                return {"status": "unverified", "evidence": "configured region lacks an independent bound reader",
                        "measured": {"destination": canonical}}
            return None
        return reader(label, canonical, evidence_dir=self.trace.run_dir / "placement")

    @property
    def _tool_axis_order(self) -> str:
        """Tool-frame convention of the arm currently wired in.

        MEASURED BUG this centralises: the grasp planner was taught the Panda's
        convention while `place_on_object`, `place_at` and the hover search
        kept calling `_yaw_rotation` with the reBot default. Grasping then
        worked (7/10 lifting ~20 cm and carrying up to 47 cm) while every place
        died with "place pose unreachable", because the two paths were asking
        for frames 92.6 degrees apart on the same robot.
        """
        return str(self.cfg.arm.get("tool_axis_order", "down_open"))

    def _profile_q(self, key: str, what: str) -> np.ndarray:
        """A joint-space keyframe (home_q, handover_q, ...) from the arm profile.

        These poses belong to ONE arm on ONE table. A default baked in here is
        worse than no pose at all: substituting the reBot's 6-element home pose
        on a 5-DoF SO-101 both mis-sizes the vector and aims a different chain
        at a table nobody measured, and the harness cannot catch a pose that is
        geometrically legal yet wrong for this robot. So a missing keyframe is
        an honest failure -- every shipped profile declares its own, pinned by
        tests/test_arm_profiles.py.
        """
        q = self.cfg.arm.get(key)
        if q is None:
            raise SkillError(
                f"arm profile declares no {key!r}, which is needed to {what}; "
                f"add it to configs/arms/<profile>.yaml"
            )
        q = np.asarray(q, dtype=float).reshape(-1)
        n = int(self.cfg.arm.get("n_joints", len(q)))
        if len(q) != n:
            raise SkillError(
                f"arm profile {key!r} has {len(q)} joints but n_joints={n}"
            )
        return q

    @property
    def _gesture_joint(self) -> int:
        """Wrist joint that `wave` wags, as an index into q.

        Declared by the profile because chains differ in length: index 4 is the
        wrist on both a 6-DoF reBot and a 5-DoF SO-101, but it is the elbow on a
        3-DoF arm. Defaults to the wrist-most joint that is not the last one on
        long chains, which reproduces the reBot's tuned choice exactly.
        """
        n = int(self.cfg.arm.get("n_joints", 6))
        idx = self.cfg.arm.get("gesture_joint")
        return int(idx) if idx is not None else min(4, max(0, n - 1))

    def _visual_diff(self, source_xyz=None, target_xyz=None):
        """CaP-X: compare the pre-motion frame with a fresh one.

        `_pre_motion_frame` is captured by execute() before any motion skill,
        so this is a genuine before/after pair. Returns None (abstain) when
        there is no pair, rather than pretending to know.
        """
        before = getattr(self, "_pre_motion_frame", None)
        if before is None:
            return None
        frame = self.last_frame or self.observe()
        if frame is None:
            return None
        # A camera that re-renders the same synthetic image every grab cannot
        # witness motion, so "the pixels did not change" is not evidence about
        # the arm -- it is a property of the sensor. Reporting UNCHANGED from it
        # fails every place whose source and target both land in frame (the
        # reBot mock only escaped this because its drop zone projects out of
        # view). Abstaining is the same answer this method already gives when
        # there is no before/after pair at all.
        if bool(self.cfg.camera.get("static_scene", False)):
            return None

        from ..perception.visual_diff import VisualDiffChannel

        def _project(xyz):
            try:
                import numpy as _np

                T = frame.T_base_cam
                if T is None:
                    T = self.extrinsics.cam_to_base()
                p_cam = (_np.linalg.inv(_np.asarray(T, float))
                         @ _np.append(_np.asarray(xyz, float)[:3], 1.0))[:3]
                if p_cam[2] <= 1e-6:
                    return None
                uv = _np.asarray(frame.K, float) @ (p_cam / p_cam[2])
                return (float(uv[0]), float(uv[1]))
            except Exception:
                return None

        return VisualDiffChannel(_project).compare(
            before, frame.rgb, source_xyz, target_xyz
        )

    def _belief_pose(self, label: str):
        """Best-known 3D position of ``label`` from the belief store."""
        if not label:
            return None
        try:
            best, score = None, 0
            want = set(str(label).lower().split())
            for b in self.beliefs.all():
                have = set(str(getattr(b, "label", "")).lower().split())
                overlap = len(want & have)
                if overlap > score:
                    best, score = b, overlap
            return list(best.position[:3]) if best is not None else None
        except Exception:
            return None

    # ── plumbing ─────────────────────────────────────────────────────────

    def _show_status(self, text: str) -> None:
        """Push agent state to EVERY live view (all rig cameras show the
        same agent, not just the primary). self.camera goes first: it may
        be a FrameHub WRAPPING the primary stream."""
        streams = [self.camera]
        rig = getattr(self, "rig", None)
        if rig is not None:
            streams += [s for s in getattr(rig, "streams", []) if s is not self.camera]
        for s in streams:
            if hasattr(s, "set_overlay"):
                s.set_overlay(status=text)

    def _show_detections(self, dets) -> None:
        if hasattr(self.camera, "set_overlay"):
            self.camera.set_overlay(detections=dets)

    def begin_task(self) -> str:
        """Trusted host boundary; does not reset the world, payload or stop latch."""
        return self._task_effects.begin_task()

    def unverified_actions(self) -> list[str]:
        return self._task_effects.unverified_actions()

    def task_effects(self) -> dict:
        """Detached task/actor-bound evidence for inspection, never an agent reset."""
        return self._task_effects.snapshot()

    def _move_joints(self, *args, **kwargs):
        context = getattr(getattr(self, "_motion_scope", None), "context", None)
        if context is not None:
            context["motion_started"] = True
        return self.arm.move_joints(*args, **kwargs)

    def _set_gripper(self, *args, **kwargs):
        context = getattr(getattr(self, "_motion_scope", None), "context", None)
        if context is not None:
            context["motion_started"] = True
        return self.arm.set_gripper(*args, **kwargs)

    def _mark_motion_started(self):
        context = getattr(getattr(self, "_motion_scope", None), "context", None)
        if context is not None:
            context["motion_started"] = True

    def _move_planned(self, *args, **kwargs):
        self._mark_motion_started()
        return getattr(self.arm, "move_planned", self.arm.move_joints)(*args, **kwargs)

    def _move_carry(self, *args, **kwargs):
        self._mark_motion_started()
        return carry_attachment.move(self, *args, **kwargs)

    def _home_policy_enabled(self) -> bool:
        scope = getattr(self, "_motion_scope", None)
        context = getattr(scope, "context", None)
        if context is not None:
            return bool(context["enabled"])
        return bool(self.cfg.arm.get("return_home_after_motion", False))

    def _arm_motions_counted(self) -> int | None:
        """Arm motions counted so far over every arm (`_actuation_counts`),
        or None when any arm lacks a counting harness."""
        try:
            counts = self._actuation_counts()
        except Exception:  # noqa: BLE001 -- bookkeeping never fails a call
            return None
        return None if counts is None else int(counts["arm_motions"])

    def _arm_moved_since(self, before: int | None) -> bool:
        """An arm motion was counted since `before`; unknown counts say yes
        (the caller's own motion-started rule then decides alone)."""
        now = self._arm_motions_counted()
        return before is None or now is None or now > before

    @staticmethod
    def _stop_latched(arm) -> bool:
        """E-stop or operator halt on this arm: move_home is not the next step."""
        harness = getattr(arm, "harness", None)
        return bool(getattr(harness, "estopped", False) or getattr(harness, "halted", None))

    def _motions_begun(self) -> int | None:
        """The running arm's counted motion starts (`SafetyHarness.motions_begun`:
        a motion counts at its first APPROVED command), or None for a harness
        double that does not count."""
        n = getattr(getattr(self.arm, "harness", None), "motions_begun", None)
        return n if type(n) is int else None

    def _refused_before_motion(self, exc: BaseException, before: int | None) -> bool:
        """`exc` is the move's own refusal before its first approved command:
        the running arm's harness counted no motion start since `before`, and
        `exc` is a refusal -- a SafetyViolation (the e-stop, halt or watchdog
        check at the motion's start, a preflight) or a SkillError (a route
        preflight that found no safe route) -- not an attachment loss and not
        raised from any other exception (a backend fault: `raise
        SafetyViolation(...) from TypeError`). The arm did not move, so the
        caller stops nothing: `arm.stop()` would latch an e-stop only the
        operator can clear (#414 review Important 1). An unknown count (a
        harness double) is never "before"."""
        if before is None or isinstance(exc, carry_attachment.AttachmentInvalid):
            return False
        seen = set()
        link = exc
        while link is not None and id(link) not in seen:
            seen.add(id(link))
            if not isinstance(link, (SafetyViolation, SkillError)):
                return False
            link = link.__cause__ if link.__cause__ is not None else (
                None if link.__suppress_context__ else link.__context__)
        return SkillRuntime._motions_begun(self) == before

    def _latched_note(self, error) -> str:
        """What a failed move's `error` text adds once its `arm.stop()` latched
        the e-stop (`SafeArm.stop()`): a latch is never silent, and never said
        twice (an e-stop that ended the move already says so)."""
        latched = getattr(getattr(self.arm, "harness", None), "estopped", False) is True
        return "; e-stop latched" if latched and "e-stop latched" not in str(error) else ""

    def _halted_since_skill_start(self, baselines=None) -> bool:
        """An operator halt arrived since the running motion skill started,
        even one withdrawn again (a Resume): the skill ended on it (A3).
        `baselines`: execute()'s `_skill_halt_baselines` for the call (None:
        the running skill's, from its motion scope; none outside a skill)."""
        if baselines is None:
            context = getattr(getattr(self, "_motion_scope", None), "context", None)
            baselines = (context or {}).get("halt_baselines", ())
        return self._skill_halt_reason(baselines) is not None

    def _may_ask_move_home(self, before: int | None, arm=None, halt_baselines=None) -> bool:
        """A failed grasp may ask the agent to call move_home first: an arm
        motion was counted since `before` (unknown counts say yes), nothing is
        held, no operator halt or e-stop is latched, and no halt arrived since
        the running skill started (A3: a halt ends the skill, so the next move
        is the operator's call, also after a Resume). One rule for execute()
        (which passes its own `halt_baselines`) and the gate-on returns of the
        persistence loop and pick_and_place."""
        arm = self.arm if arm is None else arm
        return (not self.held_object and not self._stop_latched(arm)
                and not self._halted_since_skill_start(halt_baselines)
                and self._arm_moved_since(before))

    def _home_return_skip_reason(self, failure=None) -> str | None:
        if failure is not None and not isinstance(failure, Exception):
            return "command interrupted"
        if isinstance(failure, SafetyViolation):
            return f"safety abort: {failure}"
        cancel = getattr(self, "motion_cancel_requested", None)
        if callable(cancel):
            try:
                if cancel():
                    return "command cancelled"
            except Exception as exc:
                return f"cannot determine cancellation state: {exc}"
        harness = self.arm.harness
        if getattr(harness, "estopped", False):
            return "e-stop latched"
        context = getattr(getattr(self, "_motion_scope", None), "context", None)
        if getattr(harness, "halted", None):
            # Only a halt raised since the running skill started skips the
            # return. A halt from before it (e.g. left pending by a skill that
            # ended on it) keeps the usual rule: the next motion clears it.
            # Without that baseline (no skill scope), any pending halt skips.
            baseline = next((generation for h, generation in (context or {}).get("halt_baselines", ())
                             if h is harness), None)
            if baseline is None or harness._halt_generation != baseline:
                return f"motion halted: {harness.halted}"
        violations = getattr(harness, "violations", ())
        if context is not None and len(violations) > context["violations_before"]:
            return "safety violation during this motion"
        raw = getattr(self.arm, "raw", None)
        if getattr(raw, "_stopped", False):
            return "arm stop latched"
        if callable(getattr(type(raw), "ensure_motion_ready", None)):
            try:
                raw.ensure_motion_ready()
            except Exception as exc:
                return str(exc)
        return None

    def _return_to_view_home(self, home: np.ndarray, profile: dict, failure=None,
                             halt_generation: int | None = None) -> dict:
        """Return without changing the jaws or clearing any safety latch.

        `halt_generation`: the harness's halt generation when the calling
        skill started. execute() passes the skill's baseline for its return
        after the skill body (outside the skill's halt scope by then), and
        `skill_move_home(_halt_generation=...)` passes its caller's (e.g. the
        reset's); inside a skill's scope the scope's baseline applies too
        (the harness checks both). The move then refuses a halt raised since
        then, also one raised after the skip checks below. With None and no
        skill scope, the move's begin_motion clears a halt raised after those
        checks.
        A halt the skip checks find before the move returns status
        "skipped" (reason "motion halted: ..."). A halt that refuses or
        stops the move, or that the check right after it finds, returns
        status "halted" (reason: the MotionHalted text) and never calls
        `arm.stop()`, which would latch an e-stop; an e-stop stays the
        e-stop."""
        reason = self._home_return_skip_reason(failure)
        if reason:
            return {"ok": False, "status": "skipped", "reason": reason}
        try:
            state = self.arm.get_state()
            q = np.asarray(state.q, dtype=float).reshape(-1)
            if len(q) != len(home) or not np.isfinite(q).all():
                raise RuntimeError("invalid joint feedback")
        except Exception as exc:
            return {"ok": False, "status": "skipped", "reason": f"joint feedback unavailable: {exc}"}
        tol = float(profile.get("settle_tol", 0.03))
        target = home.tolist()
        error = float(np.max(np.abs(q - home)))
        if error <= tol:
            # The measured joints are the evidence (effects._check_at_home):
            # no motion is sent, so no stream is counted.
            return {"ok": True, "status": "already_home", "home_q": target,
                    "joint_error_rad": round(error, 4), "settle_tol_rad": tol}
        # The watcher remains paused during the whole action. Obtain real
        # perception before starting another trajectory, so a long grasp
        # cannot enter the return with an expired perception watchdog.
        try:
            self.observe(require_fresh=True)
        except Exception as exc:
            return {"ok": False, "status": "skipped", "reason": f"home perception unavailable: {exc}"}
        reason = self._home_return_skip_reason(failure)
        if reason:
            return {"ok": False, "status": "skipped", "reason": reason}
        try:
            self._show_status("Returning to the saved camera-view home")
            carry_attachment.check(self)
            halt_kw = {} if halt_generation is None else {"_halt_generation": halt_generation}
            before = SkillRuntime._motions_begun(self)
            try:
                settled = self._move_carry(home, duration_s=3.0, **halt_kw)
            except MotionHalted:
                raise
            except Exception as exc:
                if SkillRuntime._refused_before_motion(self, exc, before):
                    # Refused at its start (the watchdog, the e-stop, a route
                    # preflight): the arm did not move, so nothing is latched;
                    # skipped like the freshness check above (#414 review I1).
                    return {"ok": False, "status": "skipped",
                            "reason": SkillRuntime._refused_return_reason(self, exc)}
                raise
            if not settled:
                raise SkillError("did not settle at the saved home pose")
            reason = self._home_return_skip_reason()
            if reason:
                halted = getattr(self.arm.harness, "halted", None)
                if halted and reason == f"motion halted: {halted}":
                    # The halt came after the route's last approved command:
                    # still the operator's halt, never an arm.stop() latch.
                    raise MotionHalted(f"halted: {halted}")
                raise SkillError(reason)
            final_q = np.asarray(self.arm.get_state().q, dtype=float).reshape(-1)
            if len(final_q) != len(home) or not np.isfinite(final_q).all():
                raise SkillError("invalid joint feedback after returning home")
            error = float(np.max(np.abs(final_q - home)))
            if error > tol:
                raise SkillError(f"home feedback error {error:.4f} rad exceeds {tol:.4f} rad")
            return {"ok": True, "status": "returned", "home_q": target, "joint_error_rad": error}
        except MotionHalted as exc:
            # The operator's halt stopped the return (or refused it before
            # the first command) and stays a halt: no stop latch on the arm
            # (a halt is not the e-stop).
            return {"ok": False, "status": "halted", "reason": str(exc)}
        except Exception as exc:
            # The move started (it may have stopped anywhere) or met a fault
            # outside the safety checks: stop, which latches the e-stop.
            try:
                self.arm.stop()
            except Exception as stop_error:
                logger.error("Stopping after a failed home return failed: %s", stop_error)
            return {"ok": False, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}{SkillRuntime._latched_note(self, exc)}"}

    def _refused_return_reason(self, exc: BaseException) -> str:
        """The skip reason of a view-home return its move refused before its
        first command: the e-stop as itself, the perception watchdog as the
        missing home perception (as the freshness check says it), else the
        refusal."""
        if getattr(getattr(self.arm, "harness", None), "estopped", False) is True:
            return "e-stop latched"
        if isinstance(exc, SafetyViolation) and str(exc).startswith("perception watchdog:"):
            return f"home perception unavailable: {exc}"
        return f"refused before any motion: {type(exc).__name__}: {exc}"

    def execute(self, name: str, args: dict) -> dict:
        """Dispatch with task obligations covering all tiers and preparation faults.

        B47: with a wrist narrator attached (`stream.wrist_narration: true`
        and a wrist stream, apps/wrist_narration.py) a motion skill is
        bracketed by the narrator's begin/end so the dashboard can highlight
        the wrist view while it runs; the `finally` guarantees a raised
        dispatch never leaves the highlight on. Without a narrator this is
        exactly the previous dispatch. The narrator only records -- it never
        alters args, the result, or whether the skill runs."""
        narrator = getattr(self, "wrist_narrator", None)
        if self.learning_state is not None:
            _take_learning_grant(self)  # never a line an earlier call on this thread left behind
        if narrator is None or name not in _MOTION_SKILLS:
            # Every failed result tells the agent so, including the ones the
            # dispatcher refuses before the skill runs (idempotent: results
            # that went through the full fold already carry it). Every result
            # passes `_with_retry_text`, with or without a retry ledger: only a
            # retry the ledger granted may carry the rules' retry trigger.
            return self._with_learning_grant(
                _with_retry_text(_with_failure_instruction(self._execute_with_obligations(name, args)), name))
        owns = narrator.begin(self, name, args)
        result = None
        try:
            result = self._with_learning_grant(
                _with_retry_text(_with_failure_instruction(self._execute_with_obligations(name, args)), name))
            return result
        finally:
            if owns:
                narrator.end(self, result)

    def _with_learning_grant(self, result):
        """`result` as execute() returns it (its last step, AFTER `_with_retry_text`,
        which withholds R5's words and the field's name from every key and string):
        never a top-level `learning_line` field this call was not granted, whatever
        put one there; and when `_grant_learning_line` decided a learning event, that
        event delivered (spec §4.8; A9/A10 fix). The operator-stop reading is taken
        again here, so a stop that came after the decision (during the trace write,
        the status line, the memory event) withdraws the line (Minor 2) and marks it
        withheld in the call's trace row (#398 review, Minor 2); the event is
        written to learning.json here too: a note line withdrawn writes nothing, THE
        retry writes `line: null, outcome: halted`. R5 keys on this field only."""
        if isinstance(result, dict) and _LEARNING_LINE_FIELD in result:
            result.pop(_LEARNING_LINE_FIELD)  # only the delivery below writes the field
        if self.learning_state is None:
            return result
        grant = _take_learning_grant(self)
        if grant is None:
            return result
        from ..agent import learning_surface as surface

        line, outcome, withheld = grant.line, grant.outcome, None
        try:
            if grant.stopped():
                line, outcome, withheld = None, "halted", "stopped"
        except Exception:  # noqa: BLE001 -- fail closed: an unreadable stop is a stop
            logger.exception("learning line withdrawn: the stop reading failed")
            line, outcome, withheld = None, "halted", "stop unreadable"
        if withheld is not None and grant.line is not None:
            self._mark_line_withheld(grant, withheld)
        if line is None and grant.kind != "retry":
            return result
        grant.learning.write(surface.learning_event(grant.kind, line, outcome))
        return result if line is None else surface.with_learning_line(result, line)

    def _mark_line_withheld(self, grant: _LearningGrant, why: str) -> None:
        """The call's trace row, which `_grant_learning_line` marked with the decided line,
        rewritten to say the line was withheld (`why`: "stopped", or "stop unreadable") when a
        stop withdrew it on delivery, so the audit trail never shows a line the agent did not
        get (#398 review, Minor 2). A trace without `amend_context` or an unwritten row: nothing.
        Never fails the call."""
        amend = getattr(self.trace, "amend_context", None)
        if grant.trace_step is None or amend is None:
            return
        try:
            amend(grant.trace_step, "learning", {"kind": grant.kind, "line": None, "withheld": why})
        except Exception:  # noqa: BLE001 -- the learning surface never fails a call
            logger.exception("trace row %s: the withheld learning line was not marked", grant.trace_step)

    def _execute_with_obligations(self, name: str, args: dict) -> dict:
        """Dispatch with task obligations covering all tiers and preparation faults."""
        from ..agent.effects import POSTCONDITIONS

        kind = POSTCONDITIONS.get(name)
        if kind is None:
            return self._execute_skill(name, args)
        # Registry-only identity: do not probe/materialize a LazyArm for bookkeeping.
        try:
            selected = self._select_arm(args.get("arm"))
            effective = self._arm if selected is None else selected
            actor = (next(key for key, arm in self.arm_rig.arms.items() if arm is effective)
                     if self.arm_rig is not None else "default")
        except (SkillError, StopIteration):
            actor = f"unresolved:{args.get('arm')!r}"
        obligation = self._task_effects.begin(actor=actor, skill=name, kind=kind)
        observation = {"postcondition": None}
        result = None
        try:
            result = self._execute_skill(name, args, _effect_observation=observation)
            return result
        finally:
            self._task_effects.finish(obligation, result, observation["postcondition"])

    def _execute_skill(self, name: str, args: dict, *, _effect_observation=None) -> dict:
        """Existing actuator/checker/trace path; completion is owned by execute()."""
        fn = getattr(self, f"skill_{name}", None)
        if fn is None:
            return {"ok": False, "outcome": OUTCOME_FAILED, "error": f"unknown skill {name!r}"}
        # `arm` is handled HERE, not in the skills: it names which robot runs
        # this call and is stripped from the kwargs, so no skill signature had
        # to change. Resolved before anything else -- an unknown name must
        # fail before a keyframe, a pre-state snapshot or a watcher pause
        # implies the motion is under way.
        requested_args = args  # as the caller sent them, `arm` included (the sim retry's goal)
        args = dict(args)
        arm_name = args.pop("arm", None)
        try:
            selected = self._select_arm(arm_name)
        except SkillError as e:
            return {"ok": False, "outcome": OUTCOME_FAILED, "error": f"SkillError: {e}"}
        # Record the resolved identity, not the optional selector stripped
        # above. Consult only the rig registry: reading backend attributes
        # through a LazyArm can power hardware merely to produce a log.
        effective_arm = self._arm if selected is None else selected
        resolved_arm = (
            next((key for key, arm in self.arm_rig.arms.items() if arm is effective_arm), None)
            if self.arm_rig is not None else "default"
        )
        # The running arm's frame (spec D4): its Kinematics, resolved profile
        # and base_pose, resolved before any perception, keyframe, snapshot or
        # command. A selected arm with no binding, or a base_pose that is not a
        # rigid transform, refuses the call here (a halt is never refused).
        try:
            frame = self._resolve_arm_frame(effective_arm, resolved_arm)
        except arm_frame.ArmFrameError as e:
            if name not in _FRAME_FREE_ALWAYS:
                return _arm_frame_refusal(name, resolved_arm, e)
            frame = None
        # A tilted mount (any roll or pitch of the RUNNING arm's bound
        # base_pose) refuses the frozen 15 world-target skills before any
        # perception, status line, keyframe or command (spec D4, §5). Without
        # a base_pose `is_level(None)` is True: nothing changes.
        if name in _TILT_REFUSED_SKILLS and not arm_frame.is_level(frame.T_world_base):
            return _tilted_mount_refusal(name, resolved_arm, *arm_frame.tilt_deg(frame.T_world_base))
        # Bind the selected arm AND its frame (R3): from here until the
        # postcondition fold, so the pre-state snapshot, the skill, its view-home
        # return and the verifier all read the RUNNING arm (its jaws, its
        # Kinematics and profile). Both are restored together in the `finally`,
        # even if anything raises, so a failed call on the second arm can never
        # leave a later skill pointed at it.
        if self._arm_override is None:
            self._arm_override = threading.local()
        override = self._arm_override
        prev_arm = getattr(override, "arm", None)
        prev_frame = getattr(override, "frame", None)
        override.arm = selected
        override.frame = frame
        try:
            trace_context = {"arm": resolved_arm, "held_object": self.held_object}
            # Motor-feedback contact monitor: this call's trace row carries only
            # its own motions' tracking reports (safety.contact_monitor).
            contact_harnesses = self._contact_harnesses()
            for harness in contact_harnesses:
                harness.drain_contact_reports()
            # Map snapshots of clearance refusals (spec 3.3), the same way: a
            # report filed outside this call (between calls) is stale.
            clearance_harnesses = self._clearance_harnesses()
            for harness in clearance_harnesses:
                harness.drain_clearance_reports()
                _drain_clearance_lost(harness)
            from ..robot_context import arm_profiles

            profiles = arm_profiles(self.cfg)
            profile = profiles[0]
            if selected is not None:
                profile = next(p for p in profiles if p.get("name") == arm_name)
            if name in profile.get("unsupported_skills", []):
                return {"ok": False, "error": f"skill {name!r} is unsupported on {profile.get('name', 'this arm')}"}
            # People guard (safety/people_guard.py): a motion call whose target or
            # destination is a person, a body part, the person watching, a visitor
            # side for a throw, a localized person or a visitor zone is refused
            # here, before any perception, status line, keyframe, retry ledger,
            # hardware read or command. Deterministic, whatever the agent's brain
            # decided. (A call this arm cannot run at all was refused just above.)
            if name in _MOTION_SKILLS:
                refusal = people_guard.check_call(self, name, args, effective_arm)
                if refusal is not None:
                    return self._people_refusal(name, requested_args, refusal, trace_context)
            # The skill starts here: an operator halt raised from now on (while it
            # localizes or plans, between two of its moves) ends it. Every motion
            # the skill body issues carries this halt generation (the harness's
            # skill_halt_baseline), so the halt refuses its next motion before the
            # first command instead of being cleared by it (A3 review).
            skill_halt = self._skill_halt_baselines(effective_arm) if name in _MOTION_SKILLS else ()
            if name in _MOTION_SKILLS:
                if name not in stop_retrace.EXPLICIT_SKILLS:
                    # Any other motion skill forfeits an approach column kept
                    # after a STOP below a grasp's pregrasp (stop_retrace.py).
                    stop_retrace.forget(self)
                raw = getattr(selected if selected is not None else self.arm, "raw", None)
                # Class lookup leaves a deferred LazyArm untouched. Real Franka
                # transports refuse the skill before opening the hand or moving
                # home when native feedback reports a hardware fault.
                if callable(getattr(type(raw), "ensure_motion_ready", None)):
                    try:
                        raw.ensure_motion_ready()
                    except RuntimeError as exc:
                        error = f"SkillError: {exc}"
                        self._show_status(error)
                        # STOP, native faults and stale feedback require operator
                        # intervention, not a different model-generated plan.
                        return {"ok": False, "error": error, "motion_blocked": True}
            home_snapshot = None
            return_enabled = name in _MOTION_SKILLS and bool(profile.get("return_home_after_motion", False))
            if return_enabled:
                try:
                    import inspect

                    inspect.signature(fn).bind(**args)
                    home_snapshot = np.asarray(profile.get("home_q"), dtype=float).reshape(-1).copy()
                    if len(home_snapshot) != int(profile["n_joints"]) or not np.isfinite(home_snapshot).all():
                        raise ValueError("home_q must contain one finite value per arm joint")
                except (TypeError, ValueError) as exc:
                    return {"ok": False, "error": f"cannot start {name}: {exc}"}
            # ASPIRE sim retry (spec §4.5; preflight R7, R28; A6-fix rulings): only
            # a runtime that built a ledger (a sim profile with every switch on), only
            # for the outermost motion call, before the status line, any keyframe or
            # motion. The ledger sees the caller's arguments with the RESOLVED arm, at
            # begin and at finish alike. A used retry is refused right here, by
            # `begin` alone: it returns the refusal `precheck` would, and under a
            # pending stop it also spends every other open slot (A7 ruling).
            ledger = self._sim_retry_ledger(name)
            retry = retry_g0 = ledger_args = effective_harness = retry_state = stop_token0 = None
            if ledger is not None:
                ledger_args = (dict(requested_args) if resolved_arm is None
                               else {**requested_args, "arm": resolved_arm})
                effective_harness = (getattr(effective_arm, "__dict__", None) or {}).get("harness")
                raw = getattr(selected if selected is not None else self.arm, "raw", None)
                # R7: the rig's stop token too: a halt or e-stop since the refusal, even
                # one cleared again, spends the slot (no retry, no automatic home); kept
                # for the end of the call (a stop during it, `_sim_retry_finish`)
                step = ledger.begin(name, ledger_args, held=self._jaws_may_hold(),
                                    latched=self._stop_latched(effective_arm),
                                    episode_pending=_episode_pending(effective_harness),
                                    motion_blocked=getattr(raw, "_stopped", False) is True,
                                    stop_token=(stop_token0 := self._stop_token()))
                if isinstance(step, dict):
                    return self._sim_retry_refusal(name, args, step, trace_context)
                retry = step
                # this call's own state, on this thread's motion scope (bound below)
                retry_state = _SimRetryState(() if retry is None else retry.refused_grasps)
                if retry is not None:
                    # R7c: the halt baseline of the harness the retry's home will use,
                    # taken when this skill started (execute()'s skill baseline)
                    retry_g0 = next((generation for harness, generation in skill_halt if harness is effective_harness),
                                    getattr(effective_harness, "_halt_generation", None))
            # ASPIRE learning surface (spec §4.8): only a runtime that built one (the
            # kitchen), only the outermost motion call; the stop token now, so a stop
            # during the call (even one cleared again) grants no line
            learning = self._learning_for(name)
            learning_token0 = self._stop_token() if learning is not None else None
            retry_home = False
            refused_by = None  # the exact class name of the exception that ended the skill (R6)
            # the call's arguments as the agent wrote them (halt_motion's reason, an object
            # name): MCP world_state shows this line, so the rules' triggers are withheld
            self._show_status(f"{name}({_without_rule_triggers(_short(args))})")
            # BEFORE keyframe. `last_frame` is only set by observe(), so the first
            # skill of a run used to record `keyframe_before: null` -- exactly the
            # image an outcome judge (eval/progress_judge.py) needs. Take a fresh
            # frame when none exists yet; never let a camera hiccup block a skill.
            if self.last_frame is None:
                try:
                    self.observe(require_fresh=return_enabled)
                except Exception:  # noqa: BLE001
                    pass
            # Vesta keeps "the first frame ... to preserve the initial state". A
            # memory with no frame yet is a task's first step: pin the scene as
            # it was BEFORE anything moved, so memory_frames()[0] is the state
            # the task is measured against, not the aftermath of step one.
            if (
                name in _MOTION_SKILLS
                and name != "reset_scene"
                and self.last_frame is not None
                and not self.memory.memory_frames(1)
            ):
                self.memory.add("observation", "initial state before the first action",
                                rgb=self.last_frame.rgb)
            before = self.trace.save_keyframe(
                self.last_frame.rgb if self.last_frame is not None else None, f"{name}_before"
            )
            # Wrist views (ROADMAP #15): the judge's GRM prompt has two wrist
            # slots per BEFORE/AFTER set. On a rig with a wrist stream (`role:
            # wrist` / eye-in-hand profile) save its frame next to the front one
            # for motion skills; without one the judge repeats the front image.
            before_wrists = self._save_wrist_keyframes(f"{name}_before", fresh=False) if name in _MOTION_SKILLS else None
            # Pigey: snapshot the target's pose BEFORE the motion so the
            # postcondition can measure a displacement rather than guess one.
            # NOTE the arg name varies across the skill API and getting this wrong
            # silently degrades every check to "unverified": pick_and_place takes
            # `object`, grasp/push take `label`, and place_at takes no object at
            # all (the held one is the subject). Verified against TOOL_SPECS --
            # keep this in sync when adding a motion skill.
            # place_on_object is the exception: its `label` is the DESTINATION,
            # so snapshotting it would make the checker compare the target with
            # itself ("box sits on box"). The subject there is the held object.
            pre_state = {}
            if self.effects is not None and name in _MOTION_SKILLS:
                # CaP-X: keep the pre-motion pixels so the postcondition can ask a
                # channel the actuator does not own. Copied because CameraStream
                # reuses its buffer -- holding the reference would silently give
                # us the AFTER frame twice and confirm everything.
                f = self.last_frame
                self._pre_motion_frame = f.rgb.copy() if f is not None else None
                if name == "place_on_object":
                    target_label = self.held_object
                else:
                    target_label = (
                        args.get("label")
                        or args.get("object")
                        or args.get("query")
                        or self.held_object
                        or None
                    )
                pre_state = self.effects.snapshot(target_label, name)
            region_context = None
            region_context_error = None
            if name == "place_at":
                from .mujoco_region import explicit_context
                try:
                    region_context = explicit_context(effective_arm, self.held_object, args)
                except Exception as exc:
                    # Goal bookkeeping failure never grants a different motion or
                    # clears a physical obligation. The original point path remains.
                    region_context_error = f"{type(exc).__name__}: {exc}"
            t0 = time.monotonic()
            # Derived envelope features: the outermost dispatched call owns a
            # fresh scratchpad; whatever the skill measures lands on THIS row of
            # the envelope and the trace (see `_note_measurement`). Released
            # unconditionally below, after every exception path has produced its
            # result, so one call's measurements can never bleed into the next.
            owns_measurements = self._call_measurements is None
            if owns_measurements:
                self._call_measurements = {}
            # Same ownership for the memory embedder's crop scratchpad; absent an
            # embedder this stays None and `_note_localized` is a pure pass-through.
            owns_crops = (getattr(self, "_call_crops", None) is None
                          and getattr(self.memory, "embedder", None) is not None)
            if owns_crops:
                self._call_crops = {}
            # Same ownership for two-phase localization timings (B1): the
            # outermost call collects every localization it ran.
            owns_perception_timings = getattr(self, "_call_perception_timings", None) is None
            if owns_perception_timings:
                self._call_perception_timings = []
            home_return = None
            # the operator's halt the skill had already met when its view-home
            # return started (None: any halt there came during the return)
            halt_before_return = None
            call_moved = False  # this call commanded the arm or gripper (motion scope owner only)
            # Arm motions counted before a grasp entry: its move_home hint needs
            # one counted in this call (None = no counting harness).
            arm_motions_before = self._arm_motions_counted() if name in _GRASP_ENTRY_SKILLS else None
            result = None
            halted_in_skill = None
            try:
                import contextlib

                hold = (
                    self.watcher.paused()
                    if (self.watcher is not None and name in _MOTION_SKILLS)
                    else contextlib.nullcontext()
                )
                from ..perception.detector import foreground as perception_foreground

                perceiving = (perception_foreground(getattr(self, "detector", None))
                              if name in _FOREGROUND_PERCEPTION_SKILLS else contextlib.nullcontext())
                owns_epoch = name in _MOTION_SKILLS and self._motion_t0 is None
                if owns_epoch:
                    self._motion_t0 = time.monotonic()
                if self._motion_scope is None:
                    self._motion_scope = threading.local()
                scope = self._motion_scope
                # THE sim retry's per-call state rides on this thread's motion scope
                # (never a runtime attribute: another thread never sees it)
                prev_retry_state = getattr(scope, "sim_retry", None)
                if ledger is not None:
                    scope.sim_retry = retry_state
                previous_context = getattr(scope, "context", None)
                owns_motion_scope = name in _MOTION_SKILLS and previous_context is None
                if owns_motion_scope:
                    scope.context = {
                        "enabled": return_enabled, "home": home_snapshot,
                        "profile": profile, "root_skill": name,
                        "violations_before": (len(getattr(self.arm.harness, "violations", ()))
                                              if return_enabled else 0),
                        "motion_started": False,
                        # the halt generations when the skill started: only a halt
                        # since then skips a home return (_home_return_skip_reason)
                        "halt_baselines": skill_halt,
                    }
                try:
                    with motion_evidence.record_skill(name, resolved_arm, trace_context,
                                                      enabled=name in _MOTION_SKILLS), hold, perceiving:
                        failure = None
                        try:
                            from ..planning.backend_evidence import ordinary_call
                            selection = (ordinary_call(self, name, trace_context) if name in _MOTION_SKILLS
                                         else contextlib.nullcontext())
                            with selection, self._skill_halt_scope(skill_halt):
                                if name in _MOTION_SKILLS:
                                    carry_attachment.check(self)
                                if retry is not None and retry.arm_left_in_place:
                                    # THE retry starts where the refused attempt
                                    # left the arm: home first (spec §4.5)
                                    retry_home = True
                                    self._home_before_retry(retry_g0)
                                    if arm_motions_before is not None:
                                        # the grasp's move_home step counts its own motions only
                                        arm_motions_before = self._arm_motions_counted()
                                result = fn(**args)
                                halted_in_skill = self._skill_halt_reason(skill_halt)
                        except BaseException as exc:
                            failure = exc
                            raise
                        finally:
                            call_moved = owns_motion_scope and bool(scope.context["motion_started"])
                            if (owns_motion_scope and return_enabled and scope.context["motion_started"]
                                    and name not in {"move_home", "move_to_rest", "reset_scene"}
                                    and not (isinstance(result, dict) and result.get("home_skipped"))):
                                halt_before_return = self._skill_halt_reason(skill_halt)
                                try:
                                    # Outside the skill's halt scope now: the move
                                    # carries the skill's start generation, so a
                                    # halt raised since then ends it too (pre-PR M1).
                                    return_harness = getattr(self.arm, "harness", None)
                                    home_return = self._return_to_view_home(
                                        home_snapshot, profile, failure,
                                        halt_generation=next((g for h, g in skill_halt if h is return_harness), None))
                                except Exception as exc:
                                    home_return = {"ok": False, "status": "failed",
                                                   "error": f"{type(exc).__name__}: {exc}"}
                            # Sync the held-object ignore BEFORE fusion resumes,
                            # including when a failed grasp provisionally holds.
                            if self.watcher is not None:
                                self.watcher.ignore_label(
                                    self._held_det_label if self.held_object else None
                                )
                finally:
                    if owns_motion_scope:
                        scope.context = previous_context
                    if ledger is not None:
                        scope.sim_retry = prev_retry_state
                    if owns_epoch:
                        self._motion_t0 = None
                        self._last_reobserve_t = None
                if "ok" not in result:
                    result["ok"] = True
                if halted_in_skill is not None and result.get("ok"):
                    # The operator halted this skill after it started, and the
                    # skill swallowed the halt (a best-effort home, a sort keeping
                    # what it moved): never report that as a success (A3 review).
                    # Every other field stays; an object still in the jaws is named.
                    result["ok"] = False
                    # With both stops (that halt AND a latched e-stop) the e-stop
                    # wins, as in begin_motion and `_end_on_halt`: a sort that
                    # kept what it moved after `_end_on_halt` named the e-stop is
                    # filed `safety:estop`, never as the halt.
                    result["error"] = ("SafetyViolation: e-stop latched" if self._skill_halt_estopped(skill_halt)
                                       else f"MotionHalted: {halted_in_skill}")
                    if self.held_object:
                        result.setdefault("holding", self.held_object)
            except (SkillError, SafetyViolation) as e:
                refused_by = type(e).__name__
                if isinstance(e, SkillStuck):
                    # The skill ran out of what it can do on its own and names
                    # what the human should change. Normalised below (ok=false).
                    result = {"ok": False, "outcome": OUTCOME_STUCK, "ask": e.ask,
                              "error": f"SkillStuck: {e}"}
                else:
                    result = (carry_attachment.failure_result(self, e)
                              if isinstance(e, carry_attachment.AttachmentInvalid)
                              else {"ok": False, "error": f"{type(e).__name__}: {e}"})
                    if isinstance(e, ContactDetected):
                        result.update(_contact_result(effective_arm))
                    if isinstance(getattr(e, "observed_selection", None), dict):
                        # Observed-finger grasp: ranked last / dropped / attempted
                        # candidates of the failed choice (grasp_object).
                        result["observed_selection"] = e.observed_selection
                _with_retrace_result(result, e)
                _with_skill_error(result, e)
            except TypeError as e:
                refused_by = type(e).__name__
                result = {"ok": False, "error": f"bad arguments for {name}: {e}"}
            except Exception as e:  # camera dropouts, CAN loss, ... : the agent
                # loop must survive and report, not crash mid-task.
                import traceback

                refused_by = type(e).__name__

                result = {
                    "ok": False,
                    # An exception without text is named by its type alone.
                    "error": f"unexpected {type(e).__name__}" + (f": {e}" if str(e).strip() else ""),
                    "hint": "hardware/runtime fault; consider task_done(success=false)",
                }
                _with_retrace_result(result, e)
                traceback.print_exc()
            if home_return is not None:
                result["return_home"] = home_return
                reason = str(home_return.get("error") or home_return.get("reason"))
                status = home_return["status"]
                # A halt's own text already starts with "halted: ": say it once.
                returned = ("return to saved home " + reason if reason.startswith(f"{status}: ")
                            else f"return to saved home {status}: {reason}")
                if not home_return["ok"] and result.get("ok", False):
                    result["action_ok"] = True
                    result["ok"] = False
                    result["error"] = returned
                elif (not result.get("ok", False) and halt_before_return is None
                      and _home_return_halted(home_return)):
                    # The skill failed, then the operator's halt stopped (or
                    # skipped) its return: the error names that halt first.
                    _stop_first(result, returned)
            if name in _MOTION_SKILLS:
                # Readiness can fail DURING a motion as well as before it. A
                # trajectory abort latches STOP; handing that failure back to
                # the planner would invite retries while the arm cannot move.
                # Read the selected arm's backend explicitly.
                raw = getattr(selected if selected is not None else self.arm, "raw", None)
                blocked = "arm stop latched" if getattr(raw, "_stopped", False) is True else None
                if callable(getattr(type(raw), "ensure_motion_ready", None)):
                    try:
                        raw.ensure_motion_ready()
                    except RuntimeError as exc:
                        blocked = str(exc)
                if blocked:
                    result["motion_blocked"] = True
                    result["motion_block_reason"] = blocked
                    # Keep the original controller failure, which explains WHY
                    # it stopped, instead of replacing it with the STOP symptom.
                    if result.get("ok"):
                        result["action_ok"] = True
                        result["ok"] = False
                        result["error"] = blocked
            if (name in _GRASP_ENTRY_SKILLS and call_moved and not result.get("ok")
                    and not (home_return or {}).get("ok") and not result.get("motion_blocked")
                    and self._may_ask_move_home(arm_motions_before, effective_arm, skill_halt)):
                # The failed grasp moved the arm and nothing brought it back: the
                # next grasp would start (and be refused) where this one stopped.
                # A call whose only command was a gripper open or a motion refused
                # before its first command left the arm where it was (the
                # verifier's "no grasp was attempted"): no hint.
                _with_move_home_first(result)
            dur = (time.monotonic() - t0) * 1000
            # Every result carries one of three outcomes. Stamped BEFORE the
            # postcondition fold so `annotate_result` can see a stuck step and
            # refuse to confirm an effect for it; a stuck result is forced to
            # ok=false (never a success) and always carries a non-empty ask.
            result = _normalize_outcome(result, name)
            # Release the measurement scratchpad for this call. Only skills that
            # declare derived features (DERIVED_FEATURES) get an instrumentation
            # channel on their envelope row: for them an unmeasured feature is a
            # MISSING count; everything else records exactly as before.
            measured = None
            if owns_measurements:
                measured, self._call_measurements = self._call_measurements, None
            call_crops = None
            if owns_crops:
                call_crops, self._call_crops = self._call_crops, None
            perception_timings = None
            if owns_perception_timings:
                perception_timings, self._call_perception_timings = self._call_perception_timings, None
            from ..memory.envelope import DERIVED_FEATURES, failure_signature
            if name not in DERIVED_FEATURES:
                measured = None
            if measured is not None:
                trace_context["measured"] = dict(measured)
            # Pigey closed loop: was the claimed effect real? A refuted
            # postcondition DOWNGRADES a self-reported success (annotate_result).
            if (self.effects is not None or name == "turn_screw") and result.get("ok") is not None:
                from ..agent.effects import UNVERIFIED, Postcondition, PostconditionChecker, annotate_result

                try:
                    if result.get("motion_blocked"):
                        pc = Postcondition(skill=name, kind="motion", status=UNVERIFIED,
                                           evidence="motion interrupted; completion was not verified")
                    elif self.effects is None:
                        # Bare runtimes also lack the admitted fastener observer;
                        # omitting optional verification must not credit fastening.
                        pc = PostconditionChecker().verify(name, args, result, before=pre_state)
                    else:
                        pc = self.effects.verify(name, args, result, before=pre_state)
                    if name == "turn_screw" and pc is None:
                        pc = PostconditionChecker().verify(name, args, result, before=pre_state)
                except Exception as e:  # noqa: BLE001
                    # A verifier fault must remain explicitly UNVERIFIED in the
                    # result and trace; never leave self-reported ok unchecked.
                    pc = Postcondition(
                        skill=name, kind="verifier", status=UNVERIFIED,
                        evidence=f"postcondition check crashed: {type(e).__name__}: {e}",
                    )
                    print(f"[cascade] postcondition verifier for {name} raised: {e!r}", file=sys.stderr)
                result = annotate_result(result, pc)
                if _effect_observation is not None:
                    _effect_observation["postcondition"] = pc
        finally:
            override.arm = prev_arm
            override.frame = prev_frame
        if region_context_error is not None:
            result["region_obligation"] = {"retired": False, "reason": region_context_error}
        if name == "place_at" and region_context is not None:
            from .mujoco_region import finish_explicit
            finish_explicit(region_context, result)
        if name in {"pick_and_place", "place_at"} and (result.get("ok") is False or result.get("verified") is False):
            # A failed attempt may never reach the placement verifier. Retain
            # only its existing owner-produced journal, with no new verdict.
            try:
                from ..sim.mujoco_placement import save_placement_diagnostic
                diagnostic = save_placement_diagnostic(effective_arm, self.trace.run_dir / "placement")
            except Exception as exc:
                diagnostic = {"diagnostic_only": True, "physical_task_verdict": False,
                              "error": f"placement diagnostic unavailable: {type(exc).__name__}: {exc}"}
            if diagnostic is not None:
                result["placement_diagnostic"] = diagnostic
        if name == "pick_and_place" and result.get("ok") is False:
            # Keep a failure path's own recovery step (the observed-finger
            # refusal's move_home); the routine's success wording never
            # speaks for a failed or refuted pick.
            own = str(result.get("next_action") or "").strip()
            if own == _PICK_REPORT_NEXT_ACTION:
                own = ""
            result["next_action"] = (f"{own} " if own else "") + _PICK_FAILED_GUIDANCE
        elif name == "pick_and_place" and result.get("verified") is False:
            result["next_action"] = (
                "Start the reply with 'Placement unverified.' Explain the verification_note. "
                "ok=true means the motion routine finished; it does not confirm the requested "
                "placement. Do not say 'done', 'success', or that the object is inside the "
                "destination. Include any return_home failure, then await the next user "
                "order without another movement."
            )
        elif name == "turn_screw" and result.get("verified") is False:
            result["next_action"] = (
                "Report fastening unverified. execution_ok describes only the wrist routine; "
                "do not claim fastener turns, axial advancement, seating or tightening torque. "
                "Do not repeat motion to obtain confirmation from this unsupported observer."
            )
        # Re-stamp after the verdict fold: a self-reported ok that the world
        # refuted is now ok=false and must read `failed`, not `ok`. A stuck
        # outcome survives the fold and overrides any per-skill next_action:
        # the ask goes to the human, verbatim, and this step is not retried.
        result = _normalize_outcome(result, name)
        result = _with_failure_instruction(result)
        if result.get("outcome") == OUTCOME_STUCK:
            # Prepend, never replace: a skill's own guidance (pick_and_place's
            # "if the user said 'then stop' ... do not start another pick")
            # and the failure line still apply; the relay instruction comes first.
            own = str(result.get("next_action") or "").strip()
            result["next_action"] = (
                "STUCK: relay `ask` to the human verbatim and stop; do not retry this step "
                "or invent coordinates -- wait for the human to change the scene or the "
                "instruction." + (f" {own}" if own else "")
            )
        # Harness-VLA: fold the outcome into the learned operating envelope,
        # raw args plus whatever this call measured (derived features).
        try:
            self.envelope.record(
                name, args, ok=bool(result.get("ok")),
                error=str(result.get("error", "")), duration_ms=dur,
                measured=measured, signature=failure_signature(result),
            )
        except Exception:
            pass
        # AFTER keyframe. For a MOTION skill this must be a FRESH frame taken
        # once the arm has stopped: `last_frame` is whatever the skill's last
        # observe() saw, which is a frame from BEFORE the place motion (or the
        # BEFORE keyframe itself when the skill never re-observed). Measured:
        # two chat-driven picks logged byte-identical before/after JPEGs
        # (md5 8cb509e7...) while physics confirmed a 19.8 cm move -- and the
        # outcome judge, shown no change, scored 0 % on a confirmed pick.
        if name in _MOTION_SKILLS:
            try:
                self.observe(require_fresh=return_enabled)
            except Exception:  # noqa: BLE001 -- camera hiccup: keep the stale frame
                pass
        after = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_after"
        )
        # Same rule for the wrist AFTER frame: a FRESH grab once the arm has
        # stopped, never the pre-motion frame re-saved.
        after_wrists = self._save_wrist_keyframes(f"{name}_after", fresh=True) if name in _MOTION_SKILLS else None
        # Pre-motion plausibility advisory (orchestrator hand-off, see
        # __init__): attached AFTER every verdict/annotation above so it can
        # influence none of them, and consumed here so it cannot leak onto
        # the next call. Absent (None) -> this block is a no-op and the
        # result is byte-identical to the pre-critic path. getattr: bare
        # runtimes built without __init__ (unit fixtures) have no hand-off.
        advisory = getattr(self, "pending_plausibility", None)
        if advisory is not None:
            self.pending_plausibility = None
            result["plausibility"] = advisory
        contact_reports = [r for h in contact_harnesses for r in h.drain_contact_reports()]
        if contact_reports:
            trace_context["contact_monitor"] = contact_reports
        clearance_reports = [r for h in clearance_harnesses for r in h.drain_clearance_reports()]
        if clearance_reports:
            trace_context["clearance"] = clearance_reports
        # reports the harness did not keep (a filing exception, the cap):
        # counted, so a refusal without its report is never silent
        clearance_lost = sum(_drain_clearance_lost(h) for h in clearance_harnesses)
        if clearance_lost:
            trace_context["clearance_lost"] = clearance_lost
        if perception_timings:
            # Two-phase localization (B1): where the time went. The result
            # carries the latest localization of the object the call names
            # (`label` / `object` / `query`, matched by query: the eye-in-hand
            # pick_and_place localizes its destination FIRST); a call naming
            # none of the localized queries, the first one it localized. The
            # trace carries every one of them.
            subject = args.get("label") or args.get("object") or args.get("query")
            mine = [t for t in perception_timings if subject is not None and t.get("query") == str(subject)]
            if not mine:
                first = perception_timings[0].get("query")
                mine = [t for t in perception_timings if t.get("query") == first]
            result["perception_timing"] = dict(mine[-1])
            trace_context["perception_timings"] = [dict(t) for t in perception_timings]
        if ledger is not None:
            # after every verdict and the FAILED line, before the trace row (which
            # carries the ledger's verdict)
            result = self._sim_retry_finish(
                ledger, name, ledger_args, result, trace_context, retry=retry, retry_home=retry_home,
                exc_type=refused_by if refused_by is not None else retry_state.refusal_exc_type, state=retry_state,
                arm=effective_arm, harness=effective_harness, baselines=skill_halt, stop_token0=stop_token0)
        if learning is not None:
            # after the ledger (its verdict on THE retry), before the trace row
            self._grant_learning_line(learning, name, args, result, trace_context, retry=retry, arm=effective_arm,
                                      baselines=skill_halt, stop_token0=learning_token0)
        traced = self.trace.record(name, args, result, dur, before, after,
                                   tier=self.current_tier, context=trace_context,
                                   keyframe_before_wrists=before_wrists, keyframe_after_wrists=after_wrists)
        if learning is not None:
            _bind_trace_step(self, traced)  # the row a line withdrawn on delivery is marked withheld in
        err = str(result.get("error", "failed"))
        stuck = result.get("outcome") == OUTCOME_STUCK
        # one tail for the status line and the memory event: ok / the ask / the error,
        # the retry trigger withheld (A8 fix 2, Minor 2: MCP world_state and task_memory
        # show both to the agent, and only a granted result's next_action may carry it)
        tail = _without_rule_triggers("ok" if result["ok"]
                                      else f"stuck -- {str(result.get('ask', ''))}" if stuck
                                      else err)
        self._show_status(f"{name} -> {tail[:60]}")
        # Vesta memory tuple <step, time, observation, action, verdict>: the
        # action event carries the AFTER frame (what the world looked like
        # once this action was done) and the independent postcondition
        # status, so memory_frames() can show the planner its own history as
        # captioned images instead of text alone. Motion skills only -- an
        # observe/list call changes nothing worth a frame, and the harness
        # would fill with near-duplicates of the current view.
        pc = result.get("postcondition") if isinstance(result, dict) else None
        verdict = str((pc or {}).get("status") or "") if isinstance(pc, dict) else ""
        # ROADMAP #7: with a memory embedder, the objects this call localized
        # ride on its event as crops (what they LOOKED like); the kwarg is
        # only passed when there are some, so the default call is unchanged.
        extra = {"crops": call_crops} if call_crops else {}
        self.memory.add(
            "action" if result["ok"] else "outcome",
            f"{name}({_without_rule_triggers(_short(args))}) -> {tail[:120]}",
            data={"verdict": verdict} if verdict else None,
            rgb=(
                self.last_frame.rgb
                if (name in _MOTION_SKILLS and name != "reset_scene"
                    and self.last_frame is not None)
                else None
            ),
            **extra,
        )
        return result

    def observe(self, *, require_fresh: bool = False) -> Frame:
        frame = self.camera.get_frame()
        frame = self.depth.ensure_depth(frame)
        if require_fresh:
            age = time.monotonic() - float(frame.t)
            watchdog_s = float(self.arm.harness.limits.watchdog_s)
            if not np.isfinite(age) or age < 0 or age > watchdog_s:
                raise RuntimeError(f"camera frame is stale or invalid (age {age:.3f}s)")
        self.last_frame = frame
        self._mark_perception_fresh(frame)
        return frame

    def _watchdog_feed_of(self, camera):
        """Whose perception watchdogs a frame of `camera` keeps fresh: the
        camera's per-arm feed on a multi-arm rig (a fixed camera: every arm; an
        eye-in-hand camera: the arm it rides on; a `role: wrist` view: none),
        else the running arm's harness (one arm: unchanged)."""
        for stream, feed in self._watchdog_feeds:
            if stream is camera and feed is not None:
                return feed
        return self.arm.harness

    def _mark_perception_fresh(self, frame: Frame, camera=None) -> None:
        """Every runtime frame grab that feeds the perception watchdog goes
        through here (`camera`: the frame's camera, the primary by default).

        The beat carries the frame's arrival time (`observed_at`), never the
        time of the grab: `CameraStream.get_frame` returns a dead camera's
        last frame again after its wait, of any age, and on main execute()'s
        BEFORE / AFTER frames and every in-skill observe() then re-armed the
        watchdog, so an arm with a dead camera kept moving. The feed is the
        camera's own (`_watchdog_feed_of`): in-skill frame grabs never keep an
        arm alive from a camera that does not watch it."""
        if camera is None:
            camera = getattr(self, "camera", None)  # bare `__new__` runtimes have none
        self._watchdog_feed_of(camera).heartbeat(at=observed_at(frame))

    def _wrist_streams(self) -> list[tuple[str, Any]]:
        """`(name, stream)` for every rig camera whose profile is a wrist view
        (`perception.camera_base.is_wrist_view`: `role: wrist` or eye-in-hand
        extrinsics), in rig order. Empty without a rig or without one."""
        rig = getattr(self, "rig", None)
        streams = getattr(rig, "streams", None)
        if not streams:
            return []
        from ..perception.camera_base import is_wrist_view

        cams = self.cfg.get("cameras") or [self.cfg.get("camera")]
        out = []
        for i, c in enumerate(cams):
            if c is None or not hasattr(c, "get") or not is_wrist_view(c):
                continue
            stream_name = str(c.get("name", f"cam{i}"))
            if stream_name in streams:
                out.append((stream_name, streams[stream_name]))
        return out

    def _save_wrist_keyframes(self, tag: str, *, fresh: bool) -> dict[str, str] | None:
        """One JPEG per wrist stream for a motion skill's BEFORE/AFTER pair,
        `{stream: path}`, or None when the rig has no wrist view.

        `fresh=True` (AFTER) waits for the stream's next delivery so the frame
        postdates the motion -- the same rule the front AFTER keyframe follows
        (a pre-motion frame re-saved is what scored 0 % on a confirmed pick).
        BEFORE takes the latest frame without waiting. The wait is bounded at
        1 s per stream (evidence must not make a motion skill wait on a
        stalled camera; `CameraStream.get_frame` then degrades to its latest
        frame). A wrist camera that fails or has no frame yet never blocks
        the skill: that stream is simply absent from the dict, and the judge
        falls back to repeating the front view for the step.
        """
        import re

        out: dict[str, str] = {}
        for stream_name, stream in self._wrist_streams():
            try:
                frame = stream.get_frame(timeout_s=1.0) if fresh else stream.latest()
                if frame is None:
                    frame = stream.get_frame(timeout_s=1.0)
            except Exception:  # noqa: BLE001 -- evidence, never a gate
                continue
            if frame is None or getattr(frame, "rgb", None) is None:
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stream_name)
            path = self.trace.save_keyframe(frame.rgb, f"{tag}_wrist_{safe}")
            if path:
                out[stream_name] = path
        return out or None

    def observe_fresh(self) -> Frame:
        frame = _fresh_camera_frame(self.camera, self.last_frame)
        frame = self.depth.ensure_depth(frame)
        self.last_frame = frame
        self._mark_perception_fresh(frame)
        return frame

    def backends(self) -> dict:
        """Which sidecars are REALLY in the loop, as verified at runtime --
        not what configs/demo.yaml wishes. Shown in the demo banner, the
        dashboard and the run summary so a declared-but-absent backend can
        never pass for a working one."""
        occ = getattr(self.arm.harness, "occupancy", None)
        gcfg = self.cfg.grasp
        want = str(gcfg.get("backend", "obb"))
        if self.grasp_planner_used is not None:
            grasp = self.grasp_planner_used
        elif want == "graspgenx":
            grasp = "graspgenx (configured; not yet probed -- first grasp probes it)"
        elif want == "hug":
            grasp = "hug (configured; not yet probed -- first grasp probes it)"
        else:
            grasp = want
        out = {
            "grasp_planner": grasp,
            "occupancy": occ.describe() if occ is not None else "disabled",
            "occupancy_live": bool(occ is not None and occ.status),
            # can the map authorize a step NOW: state, age, voxels (report())
            "occupancy_state": occ.report() if occ is not None and hasattr(occ, "report") else None,
        }
        if getattr(self, "vla_executor", None) is not None:
            # only with `grasp.executor: vla`: which policy server serves grasps
            out["grasp_executor"] = self.vla_executor.describe()
        return out

    def frame_jpeg(self) -> bytes | None:
        if self.last_frame is None:
            return None
        return _jpeg(self.last_frame.rgb)

    def _update_beliefs_from_frame(self, frame: Frame, dets, T=None,
                                   source: str | None = None) -> list[dict]:
        summaries = []
        self_px = self._workspace.self_pixels(frame)  # the robot's own pixels, if rendered
        if not frame.has_depth:
            return [
                {"label": d.label, "color": detection_color(frame.rgb, d),
                 "conf": round(d.conf, 2), "position": None}
                for d in dets
            ]
        if T is None:
            T = (frame.T_base_cam if frame.T_base_cam is not None
                 else self.extrinsics.cam_to_base())
            # No T = a frame of the PRIMARY camera (observe()); its stream
            # name is the colour-identity source (B32b). A caller passing T
            # for another camera names it (`_reobserve`); unnamed stays None.
            if source is None:
                source = camera_source(getattr(self, "camera", None))
        # B39 (opt-in): no render self-mask -> the arm's link geometry at this
        # frame's time feeds the same gate, on a fusion-local copy of the frame.
        link = getattr(self, "_link_self_mask", None)
        if link is not None and self_px is None and dets and self._workspace.self_mask:
            self_px = self._workspace.self_pixels(link.attach(frame, T))
        observations = []
        for d in dets:
            mask = d.mask
            if mask is None:
                h, w = frame.rgb.shape[:2]
                from ..perception.grounding import _bbox_mask
                mask = _bbox_mask(frame, d)
            # Same self-mask gate as the watcher: a detection that is mostly
            # the robot is not an object, and robot pixels never reach 3D.
            rest, _ = self._workspace.exclude_self(mask, self_px)
            if rest is None:
                continue
            pts_cam = mask_to_points_cam(frame, rest)
            if pts_cam.shape[0] < 10:
                continue
            pts_base = transform_points(T, pts_cam)
            center, extents, _ = oriented_bbox(pts_base)
            # Same geometric gate the watcher applies. Without it this path
            # registers the arm and the backdrop as objects, because an
            # open vocabulary has names for them.
            if self._workspace.reject(
                center, extents,
                mask_frac=(float(mask.sum()) / float(mask.numel() if hasattr(mask, "numel") else mask.size)),
            ) is not None:
                continue
            # Colour is what keeps two small props apart in the belief store
            # (beliefs.update refuses to fuse two different measured colours).
            # The WorldWatcher path tagged it; this one did not, so a fresh
            # get_observation on the two-cube scene fused red and blue into
            # one belief 3 cm from either -- both call sites must tag.
            color = detection_color(frame.rgb, d)
            observations.append(FrameObservation(
                d.label, center, d.conf, extent=extents,
                top_z=float(pts_base[:, 2].max()),
                color=color,
                points=pts_base if d.mask is not None else None,
                bbox=getattr(d, "bbox", None), mask=d.mask,
                source=source,
            ))
            summaries.append(
                {
                    "label": d.label,
                    "color": color,
                    "conf": round(d.conf, 2),
                    "position": [round(float(x), 3) for x in center],
                }
            )
        # The whole frame in ONE association, like the watcher: per-detection
        # update() let the second of two identical cubes inside the 8 cm gate
        # land in the belief the first had just written, and count_objects
        # then said 1 (memory/beliefs.py `update_frame`).
        if observations:
            self.beliefs.update_frame(observations, t=frame.t)
        return summaries

    def _tcp(self) -> np.ndarray:
        return self.kin.fk(self.arm.get_state().q)[:3, 3]

    def _tcp_if_awake(self) -> np.ndarray | None:
        """The gripper's TCP for a pure LOOK, or None while the arm is in
        standby.

        `_tcp()` reads `arm.get_state()`, and on a LazyArm that opens the
        arm: on the reBot, connect() (a type-4 stop frame, then enable). A
        call that only reads the cameras (annotated_view, analyze_scene,
        probe_point) must not power the motors as a side effect, the same
        rule world_state and robot_knowledge follow (`connected` is
        LazyArm's own flag and never materializes). Motion skills keep
        `_tcp()`: they are about to open the arm anyway.
        """
        if not getattr(getattr(self.arm, "raw", None), "connected", True):
            return None
        return self._tcp()

    def _tcp_world_if_awake(self) -> np.ndarray | None:
        """`_tcp_if_awake()` lifted into the table frame by the SAME arm's
        base_pose, for the viewers (base_pose spec §3.4, R14): on a skill
        thread the bound arm's TCP and frame, on viewer / MCP threads the
        primary's. Without a base_pose: `_tcp_if_awake()` itself. Raises
        ArmFrameError for an unreadable base_pose (callers draw no TCP)."""
        T = self.T_world_base
        tcp = self._tcp_if_awake()
        return tcp if tcp is None else arm_frame.to_world(T, tcp)

    def _note_measurement(self, **values) -> None:
        """Record derived envelope features for the current top-level call.

        Only finite numbers are kept; anything else is simply not written,
        and the envelope then records that feature as MISSING for this call
        (memory/envelope.py). A no-op outside a dispatched call (direct
        `skill_*` invocation from a test or a composite), so nothing here can
        leak one call's measurements into the next.
        """
        scratch = self._call_measurements
        if scratch is None:
            return
        for key, value in values.items():
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                scratch[key] = f

    @staticmethod
    def _horizontal_footprint_m(points) -> float | None:
        """Narrower extent of the cloud's XY footprint (its own principal
        axes), i.e. the smallest opening a horizontal parallel jaw can take
        it with. The 3D OBB's smallest extent is the cloud's THICKNESS for a
        top-down camera (it only sees the lid), which is not a jaw width."""
        try:
            xy = np.asarray(points, dtype=float)[:, :2]
        except (TypeError, ValueError, IndexError):
            return None
        if xy.shape[0] < 3 or not np.isfinite(xy).all():
            return None
        centred = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        proj = centred @ vt.T
        extents = proj.max(axis=0) - proj.min(axis=0)
        return float(np.min(extents))

    def _provisional_on_another_arm(self, prov) -> bool:
        """The provisional hold `prov` was recorded by a close of an arm other
        than the one running now (spec D9 / R5): the marker carries the arm
        (`ProvisionalHold.arm`; the analytic close and the VLA executor's
        both). A provisional without one (a plain tuple, a marker set
        elsewhere) counts as this arm's, as before. Identity checks only: no
        arm attribute is read."""
        owner = getattr(prov, "arm", None)
        return owner is not None and owner is not self.arm

    def _reconcile_held(self, *, seat_aware: bool = False) -> None:
        """Reconcile the held flag with the jaws, in BOTH directions.

        (a) Stale held-state: we believe we hold something but the jaws report
        (near) fully closed -> it slipped (or a crash left the flag latched);
        clear it, or one dropped object bricks every later pick with
        'already holding'.
        (b) Provisional grasp: a close was commanded but the code never
        reached the `held_object = label` line (exception during lift, an
        e-stop, a feedback timeout). If the jaws are stalled OPEN on
        something, the object IS in the gripper -> promote the marker to the
        real held state so the next skill does not open the jaws on it.
        Jaws closed on air -> drop the marker. Jaws AT the open position
        (the close never landed, or `open_gripper` ran since) -> drop it
        too: open jaws are not an invented hold.
        With `seat_aware` (reset_scene on a real gripper) a marker that
        carries its close's commanded opening (`ProvisionalHold`) is judged
        exactly like the grasp's own verify (`_air_grasp`): jaws nearly shut,
        or within 3 % of travel of that seat while the plan sized the object
        at least 7 % wider, closed on air (the reBot's rigid close on air
        stops at its seat, 85 % of travel, short of closed) -> drop it;
        stalled anywhere else short of open -> promote it. The rule's price
        is the grasp verify's: on the reBot (90 mm of travel, seat 15 % open)
        an object about 13.5-16.2 mm thick that the plan sized at 19.8 mm or
        more stalls the jaws inside that 3 % and reads as air, so the reset
        opens the jaws on it at home. A marker this reconcile cannot judge
        against the reading (malformed) raises `_HoldNotJudged`.
        (c) Thin objects: the held width MEASURED when the object was taken
        (jaw stall after the lift) decides whether a closed-jaw reading can
        mean a slip at all; see below.
        A reading that is no number (NaN or +-inf the driver reports as
        valid) is unknown feedback, never a width that reads empty or open
        (`_finite_width_frac`): a provisional hold and a held flag stay.
        """
        if getattr(self, "_contact_episode", None) is not None:
            return  # only explicit contact recovery may finish this failed close
        air = float(self.cfg.grasp.get("air_grasp_frac", 0.04))
        prov = getattr(self, "_held_provisional", None)
        if prov is not None and SkillRuntime._provisional_on_another_arm(self, prov):
            # R5: only the jaws that closed on it can promote or refute a
            # provisional hold; another arm's jaws say nothing about it.
            prov = None
        if prov is not None and not self.held_object:
            wf = self._finite_width_frac()
            if wf is not None:
                try:
                    seat = getattr(prov, "commanded_open", None) if seat_aware else None
                    # Jaws AT the open position (within the same tolerance that
                    # separates a closed jaw from air) cannot be stalled on
                    # anything: the close never landed (refused before its first
                    # stage) or the jaws were opened since. Promoting that read
                    # every later grasp as "already holding" -- a phantom hold.
                    if wf >= 1.0 - air:
                        self.memory.add(
                            "outcome",
                            f"the jaws are open; the interrupted grasp of {prov[0]!r} "
                            "did not land -- nothing is held",
                        )
                    elif seat is not None and self._air_grasp(wf, seat, prov.expected_open):
                        self.memory.add(
                            "outcome",
                            f"the jaws closed on air at the opening the interrupted grasp of {prov[0]!r} "
                            "commanded -- nothing is held",
                        )
                    elif wf >= air:
                        label, det_label, color = prov
                        self.held_object = label
                        self._held_arm = self._running_arm_name()
                        self._held_det_label = det_label
                        self._held_color = color
                        # the stall width IS the measurement of what is held
                        self._held_width_m = float(wf) * float(self._max_width)
                        self.memory.add(
                            "outcome",
                            f"the jaws are stalled on {label!r} although the grasp did not "
                            "complete -- treating it as held",
                        )
                except Exception as exc:
                    if not seat_aware:
                        raise
                    # The jaws WERE read: judging the marker against them failed.
                    raise _HoldNotJudged(exc) from exc
                self._held_provisional = None  # either promoted or refuted
        if not self.held_object:
            return
        if self._held_by_another_arm() is not None:
            # R5 / D9: only the holding arm's jaws can say the object slipped;
            # this arm's (shut, open or stalled) say nothing about it.
            return
        wf = self._finite_width_frac()
        if wf is None:  # unknown feedback, or no number: keep the cautious assumption
            return
        thin = float((self.cfg.arm.get("gripper") or {}).get("min_object_m", 0.0) or 0.0)
        # #2.5: a legitimately held VERY thin object (a card, a cable) stalls
        # the jaws below `air_grasp_frac` of travel. If the profile declares
        # `gripper.min_object_m`, anything at or above that width is a real
        # hold; without the key the 4 % heuristic stands (booth props are
        # chunky and the SO-101/reBot profiles do not declare it).
        held_w = self._gripper_width_m()
        if thin > 0 and held_w is not None and held_w >= thin:
            return
        # The MEASURED held width (jaw stall after the lift, or at promotion /
        # adoption) decides what a closed-jaw reading means. An object that
        # measured thinner than the air tolerance when it was grasped keeps
        # the jaws below `air_grasp_frac` while HELD, so position feedback
        # cannot tell that hold from a slip: never clear it on width alone.
        # (A chunky known width keeps the rule below exactly as it was.)
        known_w = getattr(self, "_held_width_m", None)
        if known_w is not None and float(known_w) < air * float(self._max_width):
            return
        if wf < air:
            self.memory.add(
                "outcome",
                f"I no longer feel {self.held_object!r} in the gripper "
                "(it must have slipped); clearing the held state",
            )
            self.held_object = None
            self._held_arm = None
            self._held_det_label = None
            self._held_color = None
            self._held_width_m = None

    def _possible_hold_refusal(self) -> str | None:
        """Why a skill whose first jaw command opens the jaws (a grasp's
        pre-grasp opening, turn_screw's at home) refuses before any command,
        or None. An interrupted grasp's provisional hold that survived the
        skill's entry reconcile (`_reconcile_held`: the jaws gave no valid
        reading, or one that is no number) may still be in the jaws: opening
        them would drop it, from about 0.44 m at home on the reBot (#373
        review A). move_home's `holding_unknown` asks for the gripper to be
        checked first; this enforces it, on every arm type, in reset_scene's
        words (`_unread_hold_refusal`), fitted for execute()'s "SkillStuck: "
        in front (`_raise_on_a_possible_hold`). Its move_home-first form needs
        an approach column that still applies: execute() forfeits a kept one
        before every motion skill but move_home and reset_scene, so only a
        direct caller can meet it.
        None with a held flag (its own "already holding" refusal) and during
        an unfinished contact episode: the reconcile never reads those jaws,
        and the harness refuses every command until reset_scene's recovery.
        Jaws that read again are judged by the reconcile as before.
        None, too, for a provisional hold ANOTHER arm's close left (R5): this
        arm's jaws say nothing about it, and the "my jaws" ask would loop on
        this arm's open_gripper (close_gripper with this arm's jaws unread,
        #387 review Minor 2). The skills that would take that hold's slot
        refuse it themselves, naming the holding arm
        (`_refuse_on_another_arms_provisional`)."""
        provisional = getattr(self, "_held_provisional", None)
        if provisional is None or self.held_object or getattr(self, "_contact_episode", None) is not None:
            return None
        if SkillRuntime._provisional_on_another_arm(self, provisional):
            return None
        return _unread_hold_refusal(provisional[0], column_kept=stop_retrace.live(self) is not None,
                                    then="try again", prefix="SkillStuck: ")

    def _raise_on_a_possible_hold(self) -> None:
        """`_possible_hold_refusal` as the SkillStuck a skill's entry raises:
        execute() reports it STUCK with the persistence loops' own way-out
        ask, so the agent relays it to the human instead of opening the jaws
        itself (#379 review minor 3). A persistence loop catches it like any
        refused attempt and stops on the hold (`_grasp_retry_verdict`)."""
        refusal = self._possible_hold_refusal()
        if refusal is not None:
            raise SkillStuck(refusal, _POSSIBLE_HOLD_ASK)

    def _refuse_a_possible_hold_before_grasping(self) -> None:
        """The guard of a composite whose motion starts with a grasp
        (sort_by_color, restore_scene, search_for_object), placed right
        before its FIRST grasp: an interrupted grasp's provisional hold is
        judged as that grasp's entry would judge it (`_reconcile_held`) and
        refused while it survives. Nothing before that point moves the arm
        or the jaws (perception and planning only), so refusing there is as
        safe as at the entry, and a call that grasps nothing (nothing to
        sort, nothing displaced, the object already in view) runs as on main
        (#379 review minor 4). A hold that a later object's failed attempt
        leaves is the persistence loops' (`_grasp_retry_verdict`)."""
        if getattr(self, "_held_provisional", None) is not None and not self.held_object:
            self._reconcile_held()
            self._raise_on_a_possible_hold()

    def _grasp_retry_verdict(self, object: str, attempt: int, last_err: str) -> str | None:
        """Why persistence should STOP retrying a grasp, or None to keep going.

        Shared by pick_and_place, handover and sort_by_color so the three
        agree on what a retry cannot cure: an e-stop, jaws that may still
        hold an object but cannot be read, an object the world model has
        never seen after re-scans (a typo, or not on the table), and an
        object wider than the jaws (re-scanning cannot shrink it)."""
        if getattr(self, "_contact_episode", None) is not None:
            return "unfinished contact episode; explicit reset_scene recovery required"
        if self.arm.harness.estopped:
            return "e-stop latched; not retrying"
        if getattr(self, "_held_provisional", None) is not None and not self.held_object:
            # The attempt ended after its close and the jaws cannot be read
            # (`_held_after_failed_attempt` kept the hold): every next grasp
            # refuses before its first command (`_possible_hold_refusal`), so
            # a re-home and re-scan would only carry the object about.
            return _POSSIBLE_HOLD_STOP
        if _is_contact(last_err):
            # The arm hit something the cameras did not see. Re-homing,
            # re-scanning and planning again would drive it back there.
            return "contact detected: something blocked the arm; not retrying into it"
        if _is_gripper_fault(last_err):
            # Reference failed (something between the jaws) or the jaws were
            # blocked opening: another attempt would only press again.
            return "gripper fault: not retrying until the jaws are cleared"
        if self._home_policy_enabled():
            reason = self._home_return_skip_reason()
            if reason:
                return f"{reason}; not retrying"
        if attempt >= 2 and "no detections" in last_err and self.beliefs.find(object) is None:
            return "never seen after re-scans; giving up early"
        if _every_candidate_too_wide(last_err):
            return "object wider than the jaws; use push_object; giving up early"
        return None

    def _halt_generation_now(self):
        """The harness's halt generation now: the baseline a composite grasp
        skill takes at its start for `_end_on_halt`."""
        return getattr(getattr(self.arm, "harness", None), "_halt_generation", None)

    def _end_on_halt(self, halt_generation, failure: str | None = None) -> None:
        """Raise MotionHalted if the operator's halt arrived since
        `halt_generation` (taken when a composite grasp skill started).

        A halt ends the whole skill, not only the motion in flight: the
        persistence loops (pick_and_place, `_grasp_with_persistence` and the
        composites around it) never re-home, re-scan or try again after one,
        and the halt reaches execute() (A3 review). Their re-home's
        begin_motion used to clear the halt, so the arm went on after an
        operator STOP. The generation also catches a halt cleared again (a
        Resume) and an air grasp whose exit the halt skipped. An e-stop alone
        needs no check here: `_grasp_retry_verdict` stops on the latch, and
        the harness refuses every command under it. When both came (a newer
        halt AND a latched e-stop) the e-stop wins, as in begin_motion and the
        release recovery: it is raised as itself, never as a halt, and
        carries no `failure` (an e-stop is filed `safety:estop`, a loss).
        `failure`: the last attempt's own failure, decided before the halt;
        the halt carries it (`_keep_decided_failure`), so the call is filed
        as that failure."""
        harness = getattr(self.arm, "harness", None)
        if halt_generation is None or getattr(harness, "_halt_generation", None) == halt_generation:
            return
        if getattr(harness, "estopped", False):
            raise SafetyViolation("e-stop latched; not retrying")
        reason = getattr(harness, "halted", None)
        halted = MotionHalted(f"motion halted: {reason}; not retrying" if reason
                              else "halt received during the grasp; not retrying")
        _keep_decided_failure(halted, failure)
        raise halted

    def _held_after_failed_attempt(self, object: str) -> tuple[bool, str] | None:
        """B36: after a grasp attempt FAILED, ask the jaws whether the close
        took the object anyway. Returns None when nothing is held (the retry
        persistence is untouched), else `(is_requested, held_label)`.

        Measured live on Isaac (bare reBot scene): attempt 1 ended "did not
        settle at grasp lift pose" with the cube lifted in the jaws; every
        retry then refused "already holding" until the budget ran out, the
        cube ended at the home pose and the scene reset failed. The held flag
        is reconciled exactly as a skill's entry does (`_reconcile_held`
        promotes the close's provisional marker only when the jaws are stalled
        on something, and drops it for open or empty jaws). Nothing is asked
        during an unfinished contact episode or an e-stop: those stop the loop
        through `_grasp_retry_verdict` as before."""
        harness = getattr(self.arm, "harness", None)
        if (harness is None or getattr(self, "_contact_episode", None) is not None
                or getattr(harness, "estopped", False)):
            return None
        try:
            self._reconcile_held()
        except (SkillError, SafetyViolation):
            pass  # unknown jaw state: whatever the flag says stands
        held = self.held_object
        if not held:
            return None
        hq, oq = str(held).lower(), str(object).lower()
        return (hq in oq or oq in hq), str(held)

    def _grasp_with_persistence(self, object: str, material: str | None = None,
                                budget_s: float | None = None) -> dict:
        """Grasp with the SAME persistence pick_and_place has (re-home,
        re-scan, fresh plan; `grasp.persist_seconds` / `max_pick_attempts`,
        capped by the task budget). handover and sort_by_color used to call
        `skill_grasp_object` ONCE and report the first miss as failure --
        inconsistent with the pick they sit next to on the cheat card.

        The loop holds the skill halt baseline execute() takes for a motion
        skill (`_skill_halt_scope`; inside a skill the outer, older baseline
        stays), also for a caller outside execute(): after an operator halt
        every later motion of the loop is refused before its first command,
        and a halt the re-home's skip checks found, or one raised after the
        re-home or during the re-scan, raises MotionHalted before the next
        attempt."""
        # R5: another arm's provisional hold refuses every attempt, and the
        # retries re-home: refused once, before the loop.
        self._refuse_on_another_arms_provisional("retry")
        with self._skill_halt_scope(self._skill_halt_baselines(self.arm)):
            return self._persistent_grasp_attempts(object, material, budget_s)

    def _persistent_grasp_attempts(self, object: str, material: str | None,
                                   budget_s: float | None) -> dict:
        """`_grasp_with_persistence`'s loop, inside its halt scope."""
        gcfg = self.cfg.grasp
        max_attempts = max(int(gcfg.get("max_pick_attempts", 8)), 1)
        t0 = time.monotonic()
        deadline = t0 + float(budget_s if budget_s is not None else gcfg.get("persist_seconds", 120.0))
        task_deadline = getattr(self, "_task_deadline", None)
        if task_deadline is not None:
            deadline = min(deadline, task_deadline)
        last_err = "grasp failed"
        attempt = 0
        stop = None  # why the loop stopped early (None = budget spent)
        halt_generation = self._halt_generation_now()  # an operator halt since now ends it
        # The last failed attempt's own error: a halt after it (in the re-home,
        # the next attempt) carries it, so the call is filed as that miss.
        decided = None
        while attempt < max_attempts:
            attempt += 1
            if attempt > 1:
                if time.monotonic() > deadline:
                    attempt -= 1
                    last_err += " (persistence budget exhausted)"
                    break
                self.memory.add("note", f"not giving up: attempt {attempt}/{max_attempts} on {object!r}")
                try:
                    self.skill_move_home()
                except MotionHalted as halted:
                    _keep_decided_failure(halted, decided)
                    raise  # STOP during the re-home ends this skill: no re-scan, no retry
                except (SkillError, SafetyViolation):
                    pass
                self._reobserve()
                # A halt the re-home skipped on, or one raised after it or
                # during the re-scan, ends the loop before the next attempt.
                self._end_on_halt(halt_generation, failure=decided)
            # Arm motions counted before this attempt: only an attempt that
            # moved the arm can have left it at a refused grasp pose.
            motions_before = self._arm_motions_counted()
            try:
                res = self.skill_grasp_object(object, material=material,
                    **({"_planning_deadline": deadline}
                       if os.environ.get("CASCADE_OBSERVED_FINGER_GATE") == "1" else {}))
            except MotionHalted as halted:
                _keep_decided_failure(halted, decided)
                raise  # the operator's halt ends this skill: no re-home, no retry
            except (SkillError, SafetyViolation) as e:
                note = _perception_stop(e, attempt)
                if note is not None:
                    # An operator halt since the loop started ends it as the
                    # halt, not as this refusal (A3: `_end_on_halt`).
                    self._end_on_halt(halt_generation, failure=decided)
                    # Same rule as pick_and_place: an expired analysis, a scene
                    # that changed under the two-phase localization (already
                    # retried once) or a re-anchor that failed again after a
                    # re-scan is not a miss to cure with another re-home and
                    # re-scan; another attempt only repeats it.
                    return {"ok": False, "held": False, "error": f"{type(e).__name__}: {e}",
                            "home_skipped": True, "grasp_attempts": attempt, "note": note}
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if res.get("ok", True) and res.get("held"):
                res["grasp_attempts"] = attempt
                return res
            decided = res.get("error") or decided   # this attempt's miss, decided before any later halt
            self._end_on_halt(halt_generation, failure=decided)
            last_err = str(res.get("error", "grasp failed"))
            if os.environ.get("CASCADE_OBSERVED_FINGER_GATE") == "1":
                out = {**res, "ok": False, "home_skipped": True, "grasp_attempts": attempt,
                       "note": "observed-finger attempt failed; automatic recovery was not geometrically checked"}
                if self._may_ask_move_home(motions_before):
                    # The refused attempt moved the arm and nothing brought it
                    # back. A refused route, a gripper-only call, a SkillError
                    # or ReanchorError before motion left it where it was; after
                    # an operator halt or e-stop, or with an object held, the
                    # next move is not the agent's (execute()'s same rule).
                    _with_move_home_first(out)
                return out
            self.memory.add("outcome", f"grasp attempt {attempt} failed: {last_err[:100]}")
            # B36: a failed attempt that still put the object in the jaws is a
            # grasp, not a miss (a retry would only refuse "already holding").
            held = self._held_after_failed_attempt(object)
            if held is not None:
                is_requested, held_label = held
                if is_requested:
                    self.memory.add(
                        "note", f"attempt {attempt} failed after the close but {held_label!r} "
                                "is in the jaws -- treating it as grasped")
                    return {"held": held_label, "grip_verified": None, "grip_profile": None,
                            "grasp_attempts": attempt, "recovered_after": last_err}
                last_err += f" (holding {held_label!r}, not {object!r}; a new grasp cannot start)"
                stop = "holding another object"
                break
            stop = self._grasp_retry_verdict(object, attempt, last_err)
            if stop:
                last_err += f" ({stop})"
                break
        out = {"ok": False, "held": False, "error": last_err, "grasp_attempts": attempt}
        # Same contract as pick_and_place: budget spent or a reason no retry
        # cures -> STUCK with a human-actionable ask; e-stop / contact
        # episode stay plain failures. `stop` is None when the loop ended by
        # the attempt cap or the clock.
        if _is_human_curable(stop):
            out["outcome"] = OUTCOME_STUCK
            out["ask"] = _stuck_ask(object, last_err, attempt)
        return out

    def begin_task_budget(self, seconds: float | None = None) -> None:
        """Open a task-scale persistence budget shared by EVERY tier.

        Called by the orchestrator at task start; `pick_and_place` (and the
        skills composed from it) cap their own `persist_seconds` deadline to
        what is left of it. `grasp.task_persist_seconds` (default 1.5x the
        per-call budget) bounds a whole task, however many tiers retry."""
        gcfg = self.cfg.grasp
        if seconds is None:
            seconds = float(gcfg.get("task_persist_seconds",
                                     1.5 * float(gcfg.get("persist_seconds", 120.0))))
        self._task_deadline = time.monotonic() + float(seconds)

    def end_task_budget(self) -> None:
        self._task_deadline = None

    def _gripper_width_m(self) -> float | None:
        """Current jaw opening in metres (fraction x max width), or None."""
        wf = self._gripper_width_frac()
        return None if wf is None else float(wf) * float(self._max_width)

    def _jaw_state(self):
        """One read of the arm state for its jaw reading; an active carry
        attachment observes it, and a failed read rejects that attachment
        before it raises."""
        try:
            state = self.arm.get_state()
            carry_attachment.observe(self, state)
        except Exception as exc:
            if carry_attachment.active(self) is not None:
                carry_attachment.active(self).reject("jaw feedback unavailable: " + str(exc))
            raise
        return state

    def _gripper_width_frac(self) -> float | None:
        """0 = fully closed, 1 = fully open (from motor angle, linear map).
        None when the gripper position is unknown (feedback failure)."""
        state = self._jaw_state()
        if not getattr(state, "gripper_valid", True):
            return None
        return self._open_frac_of(state.gripper_pos)

    def _finite_width_frac(self) -> float | None:
        """`_gripper_width_frac` for the held-state reconcile, from one read:
        a reading that is no number (NaN or +-inf the driver reports as
        valid) is unknown (None), never a width. It is judged on the raw
        position, before the travel map clips +inf to open and -inf to
        closed (air) and the reconcile drops a hold the jaws may still have."""
        state = self._jaw_state()
        if not getattr(state, "gripper_valid", True):
            return None
        pos = float(state.gripper_pos)
        return self._open_frac_of(pos) if math.isfinite(pos) else None

    # ── language -> world resolution ─────────────────────────────────────

    def _visual_query(self, query: str) -> str:
        """Explicit booth names may use a measured-color description.

        This does not provide a position, change detector labels or rewrite
        referring expressions. Unconfigured visitor queries stay unchanged.
        """
        descriptions = self.cfg.get("object_descriptions") or {}
        name = query.strip().lower()
        if name not in descriptions:
            words = name.split(maxsplit=1)
            # A leading article does not change an exact booth name. Keep
            # every other word so "the second lemon" cannot become "lemon".
            if len(words) == 2 and words[0] in {"the", "a", "an"}:
                name = words[1]
        if name not in descriptions:
            return query
        description = descriptions.get(name)
        if (not isinstance(description, str) or not description.strip()
                or parse_color_query(description)[0] is None):
            raise SkillError("Configured object description needs an explicit measured color")
        return description.strip()

    def _target_resolution(self, query, frame, fix):
        visual_query = self._visual_query(query)
        if visual_query == query:
            return {}
        return {"target_resolution": {
            "requested": query, "visual_query": visual_query,
            "detected_as": fix.detection.label,
            "measured_color": detection_color(frame.rgb, fix.detection),
            "unique_reachable_match": True,
        }}

    def _resolve_query(self, query: str) -> dict:
        """Turn a user phrase ("pink object", "red mug", "bottle") into
        detector inputs, using the live belief store when it already knows
        the answer (the WorldWatcher keeps it warm)."""
        visual_query = self._visual_query(query)
        color, noun = parse_color_query(visual_query)
        if visual_query != query:
            return {
                "prompts": [visual_query], "vocab": None, "color": color,
                "near_xyz": None, "belief": None, "prefer_label": None,
                "configured_description": True,
            }
        belief = self.beliefs.find(visual_query)
        # A ranking hint for candidates measured in the running arm's frame.
        near = self._to_arm(belief.position.copy()) if belief is not None else None
        prompts = vocab = prefer = None
        if noun:
            q = visual_query.strip().lower()
            prompts = [noun] if noun == q else [q, noun]
        elif belief is not None and (color is None or belief.color == color):
            # Warm path: the world model already knows the answer. When the
            # class is in a restricted vocabulary, detect with the WHOLE
            # vocabulary and prefer that label among candidates.
            # Open-world (no restriction) needs neither: just ask by name.
            if self._default_classes and belief.label in self._default_classes:
                vocab, prefer = self._default_classes, belief.label
            else:
                prompts = [belief.label]
        else:
            # No belief, or an UNTAGGED belief matched a color query only by
            # fallback -- don't trust it blindly: scan the vocabulary and let
            # the color filter decide (near still helps ranking).
            vocab = self._default_classes
        return {
            "prompts": prompts, "vocab": vocab, "color": color,
            "near_xyz": near, "belief": belief, "prefer_label": prefer,
            "configured_description": visual_query != query,
        }

    def _plan_grasps(self, fix, label: str | None = None, frame=None, *,
                     _deadline=None, _check=None, _prior_snapshot=None, _frame=None) -> list:
        """Grasp candidates: learned 6-DoF (GraspGen-X server) when
        configured. The Spark profile requires the real model; other profiles
        can use an explicit, visible analytic fallback.

        `grasp.backend: hug` (opt-in) plans from HUG human hands instead and
        needs `_frame`, the RGB-D frame the fix was localized in. The same
        captured frame supplies measured support height for calibrated
        RGB-D grasp profiles.

        Fake-RL layer: past-attempt memory re-ranks the candidates so grasp
        geometry that historically WORKED for this object profile goes first
        (RPent strategy-prior pattern), and applies a learned grasp-z nudge.
        """
        if _check is not None:
            _check()
        gcfg = self.cfg.grasp
        # B45 opt-in: tilted alternates of every top-down footprint candidate
        # (the reach profiles, e.g. arms/rebot_rs_reach.yaml). Unset/empty
        # leaves the planner call exactly as before.
        tilts = gcfg.get("angled_approach_tilts_deg")
        support_z = None
        if str(gcfg.get("support_height_source", "configured")) == "depth":
            from ..perception.support_surface import estimate_support_z

            support_z = estimate_support_z(frame if frame is not None else _frame, fix)
            self.memory.add("note", f"grasp support measured from depth: z={support_z:.4f} m")
        obb = plan_grasps_from_fix(
            fix,
            table_z=float(self.cfg.safety.get("table_z", 0.0)),
            max_width_m=self._max_width,
            depth_fraction=float(gcfg.get("depth_fraction", 0.5)),
            # Jaw clearance around the object. The 15 mm default is sized for
            # the reBot's 90 mm jaw (17% of travel); on a 55 mm SO-101 jaw the
            # same pad is 27%, and on a 35 mm prop it commands 52.6 mm of
            # closure for a 34.8 mm object -- the jaws stop before touching
            # and the run reports an air grasp while the prop is nudged aside
            # (measured in MuJoCo). Per-arm, like every other jaw dimension.
            width_pad_m=float(gcfg.get("width_pad_m", 0.015)),
            axis_order=self._tool_axis_order,
            **({"angled_tilts_deg": tilts} if tilts else {}),
            support_z=support_z,
            min_grasp_z_above_table=float(gcfg.get("min_grasp_z_m", .008)),
        )
        if str(gcfg.get("backend", "obb")) == "hug":
            grasps = SkillRuntime._hug_candidates(self, fix, obb, _frame, _deadline, _check)
        elif str(gcfg.get("backend", "obb")) == "camera_frame":
            grasps = SkillRuntime._camera_frame_candidates(self, fix, obb, _frame)
        elif str(gcfg.get("backend", "obb")) != "graspgenx":
            grasps = obb
            self.grasp_planner_used = "obb"
        elif self._graspgenx_down and time.monotonic() < getattr(self, "_graspgenx_retry_after", 0.0) and not bool(gcfg.graspgenx.get("required", False)):
            # Optional mode: bound repeated connection costs during outages.
            grasps = obb
            self.grasp_planner_used = "obb (graspgenx down)"
        else:
            try:
                if self._graspgenx is None:
                    from ..grasping.graspgenx_backend import GraspGenXPlanner

                    self._graspgenx = GraspGenXPlanner(gcfg)
                if self._graspgenx.status is None or self._graspgenx_down:
                    self._graspgenx.probe(**({} if _deadline is None else {"deadline": _deadline, "check": _check}))
                learned = self._graspgenx.plan(fix, max_width_m=self._max_width,
                    **({} if _deadline is None else {"deadline": _deadline, "check": _check}))
                if _check is not None:
                    _check()
                if not learned:
                    raise RuntimeError("GraspGen-X returned no usable candidates")
                self.memory.add(
                    "note",
                    f"graspgenx: {len(learned)} grasps in {self._graspgenx.last_latency_s}s "
                    f"(top {learned[0].quality:.2f})",
                )
                self._graspgenx_down = False
                grasps = learned if bool(gcfg.graspgenx.get("required", False)) else learned + obb
                self.grasp_planner_used = self._graspgenx.describe()
            except Exception as e:
                if _check is not None:
                    _check()
                    raise  # bounded search never retries model/transport/cancellation errors
                self._graspgenx_down = True
                self._graspgenx_retry_after = time.monotonic() + 5.0
                if self._graspgenx is not None:
                    self._graspgenx.status = None
                if bool(gcfg.graspgenx.get("required", False)):
                    self.grasp_planner_used = "graspgenx (unavailable)"
                    raise SkillError(f"GraspGen-X required but unavailable: {e}") from e
                self.memory.add("note", f"graspgenx unavailable ({str(e)[:90]}); OBB fallback")
                logger.warning("graspgenx unavailable (%s); analytic fallback; retry after 5s", str(e)[:160])
                grasps = obb
                self.grasp_planner_used = "obb (graspgenx down)"

        grasp_evidence.event("backend_candidates", grasps=grasps, backend=self.grasp_planner_used)

        # ---- fake-RL memory prior: re-rank + z-nudge -----------------------
        # Establish the legacy quality order before applying the prior. This
        # also orders mixed learned/analytic batches when there is no memory.
        # The selector must preserve this final order, not undo the prior by
        # sorting on the unchanged model quality a second time.
        grasps = sorted(grasps, key=lambda g: -g.quality)
        lbl = label or getattr(fix, "label", None) or "object"
        # Per arm on a rig of several (Task 5 review Minor 6); one arm: `lbl`.
        memory_lbl = SkillRuntime._grasp_memory_label(self, lbl)
        try:
            prior = self.grasp_memory.prior(memory_lbl, fix) if _prior_snapshot is None else _prior_snapshot[0]
            grasp_evidence.event("memory_prior", label=lbl, prior=prior)
            if prior:
                grasps = (self.grasp_memory.rerank(grasps, memory_lbl, fix) if _prior_snapshot is None
                          else self.grasp_memory.rerank_with_prior(grasps, fix, prior))
                grasp_evidence.event("memory_reranked_candidates", grasps=grasps)
                dz = (float(prior["nudges"].get("grasp_z_delta", 0.0))
                      if bool(gcfg.get("apply_learned_z_nudge", True)) else 0.0)
                if abs(dz) > 1e-4:
                    for g in grasps:
                        p = np.asarray(g.position, dtype=float).copy()
                        p[2] = p[2] + dz
                        g.position = p
                self._last_grasp_z_nudge = dz
                self.memory.add(
                    "note",
                    f"grasp-memory prior for {prior['profile']}: "
                    f"{prior['wins']}W/{prior['losses']}L sr={prior['success_rate']:.0%}"
                    + (f", z_nudge={dz:+.3f}" if abs(dz) > 1e-4 else "")
                    + (f", avoid={prior['top_fail']}" if prior.get("top_fail") else ""),
                )
            else:
                self._last_grasp_z_nudge = 0.0
        except Exception as e:
            if _prior_snapshot is not None:
                raise
            self._last_grasp_z_nudge = 0.0
            self.memory.add("note", f"grasp-memory prior skipped ({str(e)[:60]})")
        if _check is not None:
            _check()
        return grasps

    def _hug_candidates(self, fix, obb, frame, _deadline=None, _check=None) -> list:
        """`grasp.backend: hug`: pinches mapped from HUG human hands.

        Same contract as GraspGen-X: a REQUIRED profile raises (visible,
        never an OBB substitute) and retries on the next command; an
        optional one reports `obb (hug down)` and retries after a 5 s
        cooldown; inside grasp_object's bounded search (`_check` set) the
        deadline reaches the probe and the request and any error propagates
        as-is, unlatched. HUG emits no score: `quality` is CASCADE's
        geometric score (hug_backend.py), and the memory re-rank + selector
        + harness downstream are unchanged."""
        gcfg = self.cfg.grasp
        hcfg = gcfg.get("hug") or {}
        required = bool(hcfg.get("required", True))
        if frame is None:
            # A caller bug, not an outage: never latched, never an OBB substitute.
            raise SkillError("HUG needs the RGB-D frame the object was localized in; "
                             "none was passed")
        if (getattr(self, "_hug_down", False) and not required
                and time.monotonic() < getattr(self, "_hug_retry_after", 0.0)):
            self.grasp_planner_used = "obb (hug down)"
            return obb
        bounded = {} if _deadline is None else {"deadline": _deadline, "check": _check}
        try:
            planner = getattr(self, "_hug", None)
            if planner is None:
                from ..grasping.hug_backend import HugPlanner

                planner = self._hug = HugPlanner(gcfg)
            if planner.status is None or getattr(self, "_hug_down", False):
                planner.probe(**bounded)
            # The same camera->base transform that lifted the fix's points;
            # the planner refuses a frame whose transform does not reproduce it.
            T = (frame.T_base_cam if getattr(frame, "T_base_cam", None) is not None
                 else self.extrinsics.cam_to_base())
            learned = planner.plan(frame, fix, T_base_cam=T, max_width_m=self._max_width,
                                   axis_order=self._tool_axis_order,
                                   width_pad_m=float(gcfg.get("width_pad_m", 0.015)), **bounded)
            if _check is not None:
                _check()
            if not learned:
                raise RuntimeError("HUG returned no usable candidates")
            counts = getattr(planner, "last_counts", {}) or {}
            self.memory.add(
                "note",
                f"hug: {len(learned)} pinch candidates from {counts.get('returned', '?')} hand "
                f"samples in {planner.last_latency_s}s (pinch mapping and ranking are "
                f"cascade's; HUG has no score)")
            self._hug_down = False
            self.grasp_planner_used = planner.describe()
            return learned if required else learned + obb
        except Exception as e:
            if _check is not None:
                _check()
                raise  # bounded search never retries model/transport/cancellation errors
            self._hug_down = True
            self._hug_retry_after = time.monotonic() + 5.0
            if getattr(self, "_hug", None) is not None:
                self._hug.status = None
            if required:
                self.grasp_planner_used = "hug (unavailable)"
                raise SkillError(f"HUG required but unavailable: {e}") from e
            self.memory.add("note", f"hug unavailable ({str(e)[:90]}); OBB fallback")
            logger.warning("hug unavailable (%s); analytic fallback; retry after 5s", str(e)[:160])
            self.grasp_planner_used = "obb (hug down)"
            return obb

    def _camera_frame_candidates(self, fix, obb, frame) -> list:
        """`grasp.backend: camera_frame` (opt-in): the WRC / rebot_grasp mask
        planner (grasping/camera_grasp.py) -- approach along the camera's line
        of sight, TCP pushed into the object.

        Analytic and local, so there is no outage to latch: it either plans
        from this frame or names why not. A `required` profile then raises
        (visible, never an OBB substitute); an optional one returns the OBB
        candidates and says so in `grasp_planner_used` and a memory note. With
        `include_obb` (default) the OBB grasps also follow the mask grasp as
        IK alternates (WRC's layer 3). Either way the memory re-rank, the
        selector's width/IK checks and the harness vetting downstream are
        unchanged."""
        from ..grasping.camera_grasp import camera_frame_grasps

        gcfg = self.cfg.grasp
        ccfg = gcfg.get("camera_frame") or {}
        required = bool(ccfg.get("required", False))
        if frame is None:
            # A caller bug, not a perception gap: never an OBB substitute.
            raise SkillError("the camera-frame grasp planner needs the RGB-D frame the "
                             "object was localized in; none was passed")
        T = (frame.T_base_cam if getattr(frame, "T_base_cam", None) is not None
             else self.extrinsics.cam_to_base())
        grasps, reason = camera_frame_grasps(
            frame, fix, T,
            insertion_depth_m=float(ccfg.get("insertion_depth_m", 0.015)),
            depth_quantile=float(ccfg.get("depth_quantile", 0.5)),
            finger_drop_m=float(ccfg.get("finger_drop_m", 0.030)),
            max_fix_offset_m=float(ccfg.get("max_fix_offset_m", 0.03)),
            max_width_m=self._max_width,
            width_pad_m=float(gcfg.get("width_pad_m", 0.015)),
            axis_order=self._tool_axis_order)
        if grasps:
            g = grasps[0]
            tilt = float(np.degrees(np.arccos(np.clip(-float(g.approach[2]), -1.0, 1.0))))
            self.memory.add("note", f"camera_frame: {len(grasps)} mask grasp(s), approach "
                                    f"{tilt:.0f} deg off vertical, jaw {g.width_m * 1000:.0f} mm")
            self.grasp_planner_used = "camera_frame (mask + depth)"
            return grasps + obb if bool(ccfg.get("include_obb", True)) else grasps
        if required:
            self.grasp_planner_used = "camera_frame (unavailable)"
            raise SkillError(f"camera-frame grasp planner required but produced nothing: {reason}")
        self.memory.add("note", f"camera_frame: no mask grasp ({reason}); OBB fallback")
        logger.warning("camera_frame grasp planner produced nothing (%s); analytic OBB fallback",
                       reason)
        self.grasp_planner_used = f"obb (camera_frame: {reason})"
        return obb

    def _angled_clearance_check(self, fix):
        """B73, opt-in (`grasp.angled_clearance_vet: true`, the `*_reach` arm
        profiles): a validate step for the analytic planner's angled
        candidates (`AngledGrasp`), or None when the flag is false (the
        default: nothing is loaded, nothing changes).

        The returned check places the arm's forearm/wrist/gripper collision
        hulls (grasping/gripper_clearance.py) along the candidate's approach
        q_pre -> q_grasp with the jaws at the commanded pre-grasp opening, and
        refuses the candidate when any hull comes closer than
        `width_pad_m / 2` to the support plane (`safety.table_z`) or to the
        target's observed box -- the next candidate is then tried. Every
        decision and refusal reason is recorded in the grasp evidence
        (`gripper_clearance`). Hulls that cannot be loaded or bound to this
        arm's model refuse every angled candidate, with the reason (fail
        closed); top-down candidates are never vetted here."""
        flag = self.cfg.grasp.get("angled_clearance_vet", False)
        if flag is False:
            return None
        if flag is not True:
            raise SkillError(f"grasp.angled_clearance_vet must be true or false, got {flag!r}")
        from ..grasping import gripper_clearance as gc

        margin = gc.margin_m(self.cfg.grasp)
        open_margin = (self.cfg.arm.get("gripper") or {}).get("pregrasp_open_margin_m")
        support_z = float(self.arm.harness.limits.table_z)
        state: dict = {}

        def check(g, q_pre, q_grasp):
            jaw = gc.jaw_gap_m(g.width_m, self._max_width, open_margin)
            record = {"position": g.position, "approach": g.approach, "margin_m": margin,
                      "support_z_m": support_z, "jaw_gap_m": jaw}
            if "vet" not in state:
                try:
                    state["box"] = gc.target_box(fix, support_z)
                    state["vet"] = gc.ApproachClearance(gc.load_geometry(self.cfg.arm.get("model")), self.kin)
                except Exception as exc:  # noqa: BLE001 -- fail closed; the reason is recorded
                    state["vet"], state["error"] = None, f"{type(exc).__name__}: {exc}"
            result = None
            if state["vet"] is not None:
                try:
                    result = state["vet"].check(q_pre, q_grasp, support_z=support_z, box=state["box"],
                                                margin_m=margin, jaw_gap_m=jaw)
                except Exception as exc:  # noqa: BLE001 -- fail closed for this candidate
                    state["error"] = f"{type(exc).__name__}: {exc}"
            if result is None:
                reason = f"gripper clearance unverifiable: {state['error']}"
                grasp_evidence.event("gripper_clearance", decision="refused", reason=reason, **record)
                return reason
            grasp_evidence.event("gripper_clearance", **result.as_evidence(),
                                 box=state["box"].as_evidence(), **record)
            return None if result.ok else f"gripper clearance: {result.reason}"

        return check

    def _localization_workspace_bounds(self):
        """Read the selected arm's existing limits without connecting it."""
        limits = getattr(getattr(self.arm, "harness", None), "limits", None)
        if limits is None:
            return None  # generic perception-only runtimes have no actuator
        bounds = tuple(np.asarray(getattr(limits, name), dtype=float).reshape(3)
                       for name in ("workspace_min", "workspace_max"))
        if not np.all(np.isfinite(bounds)) or np.any(bounds[0] > bounds[1]):
            raise SkillError("invalid active arm localization workspace")
        return bounds

    def _note_localized(self, query: str, frame, fix):
        """Pass-through for `_localize`'s DETECTION returns (ROADMAP #7).

        With a memory embedder and a call in flight, keep the detected
        object's pixels (its bbox in the frame the fix was computed on) so the
        call's memory event can index what that object LOOKED like. Returns
        exactly ``(frame, fix)``: the localization result is never altered,
        and any fault here costs only the crop."""
        crops = getattr(self, "_call_crops", None)
        if crops is not None:
            try:
                det = fix.detection
                h, w = frame.rgb.shape[:2]
                x0, y0, x1, y1 = (int(round(float(v))) for v in np.asarray(det.bbox).reshape(-1)[:4])
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    label = str(det.label or getattr(fix, "label", "") or query)
                    crops.setdefault(label, frame.rgb[y0:y1, x0:x1].copy())
            except Exception:  # noqa: BLE001 -- bookkeeping never fails a localization
                pass
        return frame, fix

    def _note_perception_timing(self, timing: dict) -> None:
        """Hand one two-phase localization's timing dict (filled in as it
        runs, so a refused one still says how far it got) to the dispatched
        call: execute() puts the latest one of the object the call names in
        the skill result as `perception_timing` (matched by query) and every
        one in the trace context. A no-op outside a dispatched call."""
        timings = getattr(self, "_call_perception_timings", None)
        if timings is not None:
            timings.append(timing)

    def _note_localized_on(self, camera) -> None:
        """Record which camera stream `_localize`'s answer on this thread came
        from (`_localized_camera`); None: none, or not a camera's detection."""
        self._localized_on = (threading.get_ident(), camera)

    def _localized_camera(self):
        """The camera stream the calling thread's last `_localize` answer came
        from: the primary (`self.camera`) or one of `_other_cams()`'s. None
        when that answer was the belief's or the VLM's, when it found nothing,
        or when this thread localized nothing yet. point_at's look again
        compares producer captures only within one camera (#413 review Minor
        1)."""
        found = getattr(self, "_localized_on", None)
        return found[1] if found is not None and found[0] == threading.get_ident() else None

    def _localize(self, query: str, spatial_hint: str | None = None, *, memory: bool = True):
        """Fresh frame + color/proximity-aware 3D fix for a user phrase.

        A single detector pass at low confidence flickers frame to frame, so
        a miss is retried on a couple of FRESH frames before giving up on
        detection. After that it falls back to a fresh belief: the
        WorldWatcher fuses at ~3 Hz with temporal stability, and an object
        the world model has seen recently is still perfectly actionable
        (that is the whole point of keeping the model warm). Staleness is
        measured against the moment the current motion skill STARTED, not
        "now" -- the watcher is paused while the arm moves, so beliefs age
        artificially during exactly the retries that need them.

        `memory` False: never that belief fallback (a detection on a fresh
        image, or the VLM's, else the miss is raised). point_at's look again
        after a long plan needs it: inside the skill the belief is what was
        known before the plan, so it can only read as unmoved (#406 review).
        Which camera its answer came from: `_localized_camera()`."""
        self._note_localized_on(None)
        localization_started = time.monotonic()
        r = self._resolve_query(query)
        workspace_bounds = self._localization_workspace_bounds()
        # Text/checkpoint loading does not consume an image. Finish it before
        # observing, so cold vocabulary initialization cannot expire the
        # image we have already selected. No detection or motion is retried;
        # analyze() below still expires slow inference on its actual frame.
        prepare = getattr(self.detector, "prepare", None)
        if prepare is not None:
            rounds = [r["vocab"]] if r["vocab"] else [[p] for p in (r["prompts"] or [query])]
            try:
                for classes in rounds:
                    prepare(classes)
            except Exception as exc:
                raise SlowPerceptionError(
                    f"Detector preparation failed before localization: {exc}"
                ) from exc
        def in_workspace(position):
            if workspace_bounds is None:
                return True
            point = np.asarray(position, dtype=float).reshape(3)
            return (np.all(np.isfinite(point)) and np.all(point >= workspace_bounds[0])
                    and np.all(point <= workspace_bounds[1]))
        frame = None
        # The primary frame's age when it was grabbed: a dead camera's last
        # frame is already past the limit then, and the fallbacks below say so
        # (StaleFrameError) instead of "fresh at capture ... the camera is live".
        # Measured from its first arrival (`observed_at`): a producer's re-sent
        # capture is as old as its first delivery, not its new receipt.
        frame_age_at_grab = None

        def grabbed(f):
            nonlocal frame_age_at_grab
            age = time.monotonic() - observed_at(f)
            frame_age_at_grab = age if math.isfinite(age) else None
            return f

        last_err: SkillError | None = None
        reserve = getattr(self.detector, "reserve", nullcontext)
        loop_cfg = self.cfg.get("perception_loop", {})
        # B1 (spec 2026-10-16): with a detector that can re-anchor (the brain
        # detector with a segmenter), the brain only NAMES the object on one
        # frame; every geometry acted on comes from a fresh frame. Any other
        # detector keeps the single-frame path below, unchanged.
        two_phase = can_reanchor(self.detector)
        semantic_max_age_s = float(loop_cfg.get("semantic_max_age_s", 20.0))
        retried = {"scene": 0, "reanchor": 0}

        def localize_once(cframe, ext):
            # Today's per-frame localization: the detector's refer, SAM,
            # colour/reference/workspace rules (single-frame path and the
            # two-phase naming step alike). A refusal quotes the point in the
            # table frame (`report_point`), like every report.
            return localize_object(
                cframe, query, self.detector, ext,
                prompts=r["prompts"], spatial_hint=spatial_hint,
                color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                prefer_label=r["prefer_label"],
                workspace_bounds=workspace_bounds,
                require_unique=r["configured_description"],
                report_point=self._to_world,
            )

        def analyze(cframe, ext):
            # Bound the age of the image actually analyzed, including lazy
            # detector loading and a slow miss. Capture clocks may be remote;
            # Frame.t and this analysis clock are both client-local.
            _frame_age_s(cframe)
            started = time.monotonic()
            try:
                return localize_once(cframe, ext)
            finally:
                _analyzed_frame_age_s(cframe, time.monotonic() - started)

        def name_object(cframe, ext):
            # Two-phase step 2: today's localize_object on frame0 (the brain's
            # refer, SAM, colour/reference/workspace rules) picks the object.
            # Nothing measured here is acted on: the semantic cap, not 5 s.
            _frame_age_s(cframe)
            started = time.monotonic()
            try:
                return localize_once(cframe, ext)
            finally:
                _semantic_age_s(cframe, time.monotonic() - started, semantic_max_age_s)

        def check_time_left():
            # Two-phase only (review I2): the naming step may take up to
            # semantic_max_age_s, so no NEW frame or camera starts once the
            # localization is older than that; a slow miss is refused in
            # about one cap plus one naming round, as SlowPerceptionError.
            elapsed = time.monotonic() - localization_started
            if two_phase and not elapsed <= semantic_max_age_s:
                raise SlowPerceptionError(
                    f"Perception took {elapsed:.1f}s without a fresh fix of {query!r} (limit "
                    f"{semantic_max_age_s:g}s to name it and re-measure it on a fresh frame); the camera "
                    "is live -- retry the observation"
                )

        def re_mask(frame0, fix0, timing):
            # Before frame1 is taken (its age is the 5 s rule's, this SAM call
            # is no part of it): frame0 re-masked with the prompt frame1 will
            # get, so every check compares masks made alike. The brain's
            # detection still names the object; a failed re-mask is a failed
            # re-anchor.
            started = time.monotonic()
            try:
                reference = self.detector.reanchor_reference(frame0, fix0.detection)
            except Exception as exc:  # noqa: BLE001 -- a failed SAM call is a failed re-anchor
                raise _ReanchorFailed(f"{type(exc).__name__}: {exc}") from exc
            finally:
                timing["reference_s"] = round(time.monotonic() - started, 3)
            # Evidence only (follow-ups M5): how the naming mask sits against
            # the brain's box, for the rig data that decides a finer rule.
            overhang = getattr(reference, "naming_overhang", None)
            if overhang is not None:
                timing["naming_overhang"] = [_rounded(v, 3) for v in overhang]
                timing["naming_foreign_share"] = _rounded(getattr(reference, "naming_foreign_share", math.nan), 3)
            if reference.detection is None or reference.detection.mask is None:
                raise _ReanchorFailed(reference.reason or "SAM returned no mask on the image the object was named on")
            return reference

        def measure_again(frame0, fix0, reference, ext0, frame1, ext1, timing):
            # Steps 4-7 on frame1: the same view, the brain's box re-anchored
            # by SAM, the geometry from frame1 alone, and the 5 s rule on it.
            started1 = time.monotonic()
            try:
                changed = _view_change(frame0, frame1)
                if changed:
                    raise _SceneChanged(changed)
                timing["frame0_age_at_reanchor_s"] = round(started1 - float(frame0.t), 3)
                try:
                    found = self.detector.reanchor(frame1, fix0.detection, since=frame0, reference=reference)
                except Exception as exc:  # noqa: BLE001 -- a failed SAM call is a failed re-anchor
                    raise _ReanchorFailed(f"{type(exc).__name__}: {exc}") from exc
                finally:
                    timing["reanchor_s"] = round(time.monotonic() - started1, 3)
                timing["scene_diff_global"] = _rounded(found.scene_diff_global, 2)
                timing["scene_diff_local"] = _rounded(found.scene_diff_local, 2)
                gain = getattr(found, "gain", None)
                if gain is not None:
                    timing["scene_gain"] = [_rounded(g, 3) for g in gain]
                prompt_box = getattr(found, "prompt_box", None)
                if prompt_box is not None:
                    timing["reanchor_prompt_box"] = [int(v) for v in prompt_box]
                timing["mask_iou"] = _rounded(getattr(found, "mask_iou", math.nan), 3)
                timing["mask_agreement"] = _rounded(getattr(found, "mask_agreement", math.nan), 3)
                timing["mask_components_dropped"] = int(getattr(found, "dropped_components", 0))
                timing["mask_clipped_px"] = int(getattr(found, "clipped_px", 0))
                if found.scene_changed:
                    raise _SceneChanged(found.reason)
                if found.detection is None or found.detection.mask is None:
                    raise _ReanchorFailed(found.reason or "SAM returned no mask")
                try:
                    fix1 = fix_from_detection(frame1, found.detection, ext1, label=query, color=r["color"],
                                              workspace_bounds=workspace_bounds, report_point=self._to_world)
                except DetectionRejected as rejected:
                    if rejected.kind == "points":
                        raise _ReanchorFailed(rejected.reason) from None
                    raise _SceneChanged(rejected.reason) from None  # colour, or out of the workspace now
                # Both positions measured the same robust way (medians inside
                # the masks eroded by 2 px), on the masks the geometry uses:
                # frame1's clipped to the naming mask, and frame0's same-prompt
                # re-mask clipped alike (a neighbour merged under the grown
                # prompt is in neither). A pixel of SAM edge jitter onto the
                # table tilts the fixes' OBB centres by ~2 cm.
                then = getattr(reference, "geometry", None)
                then = reference.detection if then is None else then
                moved = float(np.linalg.norm(median_point(frame1, found.detection, ext1)
                                             - median_point(frame0, then, ext0)))
                timing["moved_m"] = _rounded(moved, 4)
                if not moved <= _REANCHOR_MAX_MOVE_M:
                    raise _SceneChanged(f"the object moved {moved * 100:.1f} cm between the two images")
                return fix1
            finally:
                timing["frame1_age_s"] = round(time.monotonic() - float(frame1.t), 3)
                _analyzed_frame_age_s(frame1, time.monotonic() - started1)

        def localize_two_phase(take_frame0, take_frame1, extrinsics_of):
            """frame0 -> name the object; a fresh frame1 -> measure it. ONE
            retry (a new frame0 and a fresh refer) after a scene change, and
            one after a failed re-anchor, per localization; the second of a
            kind raises SceneChangedError / ReanchorError. Returns (frame1,
            fix1)."""
            for attempt in range(1, 4):  # at most one retry of each kind
                if attempt > 1:
                    check_time_left()
                frame0 = take_frame0()
                timing = {"query": query, "attempt": attempt}
                self._note_perception_timing(timing)
                started0 = time.monotonic()
                ext0 = extrinsics_of(frame0)
                try:
                    fix0 = name_object(frame0, ext0)
                finally:
                    timing["semantic_s"] = round(time.monotonic() - started0, 3)
                try:
                    reference = re_mask(frame0, fix0, timing)
                    # The re-mask's SAM call counts against the naming step's
                    # cap, not against the fresh frame's 5 s.
                    _semantic_age_s(frame0, time.monotonic() - started0, semantic_max_age_s,
                                    what="to name the object and re-mask it")
                    frame1 = take_frame1(frame0)
                    return frame1, measure_again(frame0, fix0, reference, ext0, frame1, extrinsics_of(frame1),
                                                 timing)
                except _SceneChanged as exc:
                    timing["scene_changed"] = str(exc)
                    retried["scene"] += 1
                    if retried["scene"] > 1:
                        raise SceneChangedError(
                            f"scene changed during perception of {query!r}: {exc}; the image the object "
                            "was named on no longer matches the fresh one -- hold the scene still and retry"
                        ) from None
                except _ReanchorFailed as exc:
                    timing["reanchor_failed"] = str(exc)
                    retried["reanchor"] += 1
                    if retried["reanchor"] > 1:
                        raise ReanchorError(f"could not re-anchor {query!r} on a fresh image: {exc}") from None
            raise ReanchorError(f"could not re-anchor {query!r} on a fresh image")

        # Every frame analysed for the target is in the running arm's base
        # frame (`_frame_in_arm`, spec §3.3): its fix, the workspace bounds and
        # the near hint all are. `frame` itself stays the camera's own
        # (table-frame) capture for the VLM fallback, which converts its views.
        def primary_frame0():
            nonlocal frame
            frame = grabbed(self.observe())
            return self._frame_in_arm(frame)

        def primary_frame1(frame0):
            fresh = self.depth.ensure_depth(_fresh_camera_frame(self.camera, previous=frame0))
            self.last_frame = fresh
            self._mark_perception_fresh(fresh)
            return self._frame_in_arm(fresh)

        tries = int(loop_cfg.get("localize_frames", 3))
        for frame_try in range(max(tries, 1)):
            if frame_try:
                check_time_left()
            try:
                # Waiting behind the watcher consumes request time, not the
                # lifetime of an image selected before inference is available.
                with reserve():
                    if two_phase:
                        analyzed, fix = localize_two_phase(primary_frame0, primary_frame1,
                                                           lambda _frame: self.extrinsics)
                        self._note_localized_on(self.camera)
                        return self._note_localized(query, analyzed, fix)
                    frame = grabbed(self.observe())
                    analyzed = self._frame_in_arm(frame)
                    fix = analyze(analyzed, self.extrinsics)
                self._note_localized_on(self.camera)
                return self._note_localized(query, analyzed, fix)
            except SlowPerceptionError:
                raise  # Not detector flicker; do not fall back to memory.
            except (SceneChangedError, ReanchorError):
                raise  # The object was named; its fresh geometry is refused, not missed.
            except ReferenceResolutionError:
                # An unsatisfied identity constraint is not detector flicker.
                # Retrying/falling back to a remembered noun can select a
                # different object than the visitor explicitly requested.
                raise
            except SkillError as e:
                last_err = e
        # Second filter: THE OTHER CAMERAS. An object 40 px small (or
        # occluded) in the primary view may be plainly visible from the side
        # or wrist camera; each pass costs one detector call.
        from ..perception.grounding import Extrinsics as _Ext

        class _NoDepth(Exception):
            pass

        for cam in self._other_cams():
            check_time_left()

            def ext_of(cframe, cam=cam):
                return (_Ext(T=cframe.T_base_cam)
                        if cframe.T_base_cam is not None else cam.extrinsics)

            def side_frame0(cam=cam):
                cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                if not cframe.has_depth:
                    raise _NoDepth
                return self._frame_in_arm(cframe, cam.extrinsics)

            def side_frame1(frame0, cam=cam):
                return self._frame_in_arm(
                    cam.depth.ensure_depth(_fresh_camera_frame(cam.stream, previous=frame0)), cam.extrinsics)

            try:
                with reserve():
                    if two_phase:
                        cframe, fix = localize_two_phase(side_frame0, side_frame1, ext_of)
                        ext = ext_of(cframe)
                    else:
                        cframe = side_frame0()
                        ext = ext_of(cframe)
                        fix = analyze(cframe, ext)
                if (os.environ.get("CASCADE_OBSERVED_FINGER_GATE") == "1"
                        and cframe.T_base_cam is None and ext.mode == "eye_to_hand"):
                    # Bind the secondary view's actual static calibration to
                    # this result; never substitute the primary camera's T.
                    from dataclasses import replace
                    cframe = replace(cframe, T_base_cam=np.asarray(ext.T).copy())
                self.memory.add(
                    "note",
                    f"localize {query!r}: primary camera missed it; found "
                    f"through {getattr(cam.stream, 'name', 'another camera')}",
                )
                self._note_localized_on(cam.stream)
                return self._note_localized(query, cframe, fix)
            except _NoDepth:
                continue
            except SlowPerceptionError:
                raise
            except (SceneChangedError, ReanchorError):
                raise
            except ReferenceResolutionError:
                raise
            except Exception:
                continue  # a miss or camera hiccup: try the next view

        if r["configured_description"]:
            # A remembered label or unfiltered VLM fix cannot prove the
            # configured color is currently visible and unique.
            raise last_err or SkillError(f"No current visual match for {query!r}")
        if not parse_reference(query).is_plain:
            # A remembered noun (or one guessed bounding box) does not prove
            # ordering, exclusions or relative size among current instances.
            raise ReferenceResolutionError(
                f"Cannot resolve {query!r} from current observations. "
                "Clarify the reference; refusing an unqualified fallback."
            )
        belief = r["belief"] or self.beliefs.find(query)
        # The belief is table-frame memory; as a target it is taken into the
        # running arm's frame BEFORE a fix is synthesized from it, so the
        # remembered box is built upright in the arm frame (equivariant).
        belief_in_arm = None if belief is None else self._belief_to_arm(belief)
        max_age = float(self.cfg.get("perception_loop", {}).get(
            "belief_fallback_age_s", 3.0))
        # Staleness reference: the last fresh re-scan during this motion (a
        # re-scan that FAILED to see the object must age the belief), else
        # the motion start (the watcher is paused while the arm moves), else
        # now.
        if self._last_reobserve_t is not None:
            ref_t = self._last_reobserve_t
        elif self._motion_t0 is not None:
            ref_t = self._motion_t0
        else:
            ref_t = time.monotonic()
        mem_ok = (
            memory
            and belief is not None
            and (ref_t - belief.last_seen_t) <= max_age
            # Grasping from memory demands an EXACT color match: the belief
            # store's neighbor tolerance (pink~red) is for conversation, not
            # for choosing what the jaws close on.
            and (r["color"] is None or belief.color == r["color"])
            and in_workspace(belief_in_arm.position)
        )
        if mem_ok:
            _analyzed_frame_age_s(frame, time.monotonic() - localization_started, age_at_start=frame_age_at_grab)
            self.memory.add(
                "note",
                f"localize {query!r}: instant detection missed; using the "
                f"world-model fix ({belief.label}, "
                f"{time.monotonic() - belief.last_seen_t:.1f}s old)",
            )
            return (frame if frame is None else self._frame_in_arm(frame)), \
                self._fix_from_belief(query, belief_in_arm)
        # Last filter: VLM grounding of this one query, when the detector
        # missed it; one slow call only ever runs on this failure path.
        _analyzed_frame_age_s(frame, time.monotonic() - localization_started, age_at_start=frame_age_at_grab)
        grounded = self._vlm_ground_fix(frame, query)
        if grounded is not None and in_workspace(grounded[1].position):
            return grounded
        if grounded is not None:
            _, fix = grounded
            # The fix is in the arm frame; the agent reads table-frame points.
            last_err = SkillError(
                f"localize {query!r} failed: VLM fix at "
                f"{np.round(self._to_world(fix.position), 3).tolist()} is outside the active arm workspace"
            )
        raise last_err

    def _vlm_ground_fix(self, frame, query: str):
        """Ask the VLM (2nd perception filter) for the object's bbox and
        lift it to a 3D fix -- trying EVERY camera's view (the primary may
        see the object at 40 px while the side camera fills the frame with
        it). Fail-soft: any error returns None so the caller reports the
        original detector failure. A successful result keeps its actual view;
        slow analysis is terminal rather than a miss on another camera."""
        gcfg = self.cfg.get("grounder", None)
        llm_cfg = self.cfg.get("llm", {})
        if not gcfg and llm_cfg.get("grounding_fallback", False):
            gcfg = llm_cfg
        if not gcfg:
            return None
        if frame is not None:
            _frame_age_s(frame)
        started = time.monotonic()
        analyzed_frame = frame
        try:
            if self._grounder is None:
                from ..perception.vlm_ground import VLMGrounder

                # A profile that names its key FILE (local_cosmos_sglang: its
                # brain serves only with a key) sends that key; the others build
                # the grounder exactly as before.
                keyed = {}
                if gcfg.get("api_key_file"):
                    from ..agent.llm import read_api_key_file

                    keyed["api_key"] = read_api_key_file(gcfg.get("api_key_file"), who="grounder")
                self._grounder = VLMGrounder(
                    base_url=str(gcfg.get("base_url")),
                    model=str(gcfg.get("model", "")),
                    timeout_s=float(gcfg.get("timeout_s", 45.0)),
                    enable_thinking=gcfg.get("enable_thinking"),
                    **keyed,
                )
            # Views to try: (frame, cam->base T). Primary first, then the
            # other fusing cameras with their own extrinsics.
            # Each view in the running arm's frame (`_frame_in_arm`: its T is
            # T_world_base^-1 T_table_cam), so the fix, the table shave below
            # and the returned frame share the arm frame.
            views = []
            if frame is not None and frame.has_depth:
                frame = self._frame_in_arm(frame)
                views.append((frame, frame.T_base_cam if frame.T_base_cam
                              is not None else self.extrinsics.cam_to_base()))
            for cam in self._other_cams():
                try:
                    cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                except Exception:
                    continue
                if not cframe.has_depth:
                    continue
                cframe = self._frame_in_arm(cframe, cam.extrinsics)
                views.append((cframe, cframe.T_base_cam if cframe.T_base_cam
                              is not None else cam.extrinsics.cam_to_base()))
            det = gframe = T = None
            for cframe, cT in views:
                analyzed_frame = cframe
                _analyzed_frame_age_s(cframe, time.monotonic() - started)
                try:
                    det = self._grounder.ground(cframe, query)
                finally:
                    _analyzed_frame_age_s(cframe, time.monotonic() - started)
                if det is not None:
                    gframe, T = cframe, cT
                    break
            if det is None:
                return None
            from ..perception.grounding import _bbox_mask
            mask = _bbox_mask(gframe, det)
            pts_cam = mask_to_points_cam(gframe, mask)
            if pts_cam.shape[0] < 30:
                return None
            pts = transform_points(T, pts_cam)
            # A bbox rectangle sweeps in table pixels; shave the table plane
            # or the OBB extent balloons (the GGX bbox-slab lesson).
            table_z = float(self.cfg.safety.get("table_z", 0.0))
            above = pts[pts[:, 2] > table_z + 0.005]
            if above.shape[0] >= 30:
                pts = above
            center, extents, axes = oriented_bbox(pts)
            from ..types import ObjectFix

            self.memory.add(
                "note",
                f"the detector missed {query!r}; the VLM (2nd filter) found it at "
                f"{[round(float(v), 2) for v in self._to_world(center)]}",
            )
            fix = ObjectFix(
                label=query, position=center, points=pts,
                detection=det, extent=extents, axes=axes,
            )
            _analyzed_frame_age_s(gframe, time.monotonic() - started)
            if gframe.T_base_cam is None:
                from dataclasses import replace
                gframe = replace(gframe, T_base_cam=np.asarray(T).copy())
            return gframe, fix
        except SlowPerceptionError:
            raise
        except Exception as e:
            self.memory.add("note", f"VLM grounding unavailable: {str(e)[:80]}")
            return None
        finally:
            # A delayed constructor failure, empty cloud or failed transform
            # is still expired analysis, not a fast miss eligible for recovery.
            if analyzed_frame is not None:
                _analyzed_frame_age_s(analyzed_frame, time.monotonic() - started)

    def _other_cams(self) -> list:
        """Non-primary fusing cameras from the watcher wiring: each entry
        has .stream, .depth and .extrinsics (an object may be visible from
        the side camera while the primary sees it at 5 px)."""
        cams = getattr(self.watcher, "_cams", None) if self.watcher else None
        if not cams:
            return []
        return [c for c in cams[1:] if getattr(c, "fuse", True)]

    def _sample_link_state(self, frame) -> None:
        """B39: record the arm's joint state as a frame is taken, BEFORE
        inference, so the link self-mask is posed at the image's time (a
        sample taken after detection is further from it). A no-op unless
        `workspace_filter.link_self_mask` is on and the frame has no render
        self-mask."""
        link = getattr(self, "_link_self_mask", None)
        if (link is not None and self._workspace.self_mask
                and getattr(frame, "robot_mask", None) is None):
            link.sample()

    def _reobserve(self, frames: int = 2) -> None:
        """Refresh beliefs with fresh detector passes over EVERY fusing
        camera while the WorldWatcher is paused (motion skills hold it):
        fusion by hand, exactly what the 3 Hz loop would do. The held
        object (if any) is excluded -- fusing it dangling mid-air would
        corrupt its belief."""
        held = self._held_det_label if self.held_object else None
        for _ in range(max(int(frames), 1)):
            try:
                frame = self.observe()
                self._sample_link_state(frame)
                dets = detect_vocabulary(self.detector, frame, self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                self._show_detections(dets)
                self._update_beliefs_from_frame(frame, dets)
                self._last_reobserve_t = time.monotonic()
            except Exception:
                return  # best effort: a camera hiccup must not kill the retry
        for cam in self._other_cams():
            try:
                frame = cam.depth.ensure_depth(cam.stream.get_frame())
                if not frame.has_depth:
                    continue
                self._sample_link_state(frame)
                dets = detect_vocabulary(self.detector, frame, self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                T = (frame.T_base_cam if frame.T_base_cam is not None
                     else cam.extrinsics.cam_to_base())
                self._update_beliefs_from_frame(
                    frame, dets, T=T, source=camera_source(cam.stream))
            except Exception:
                continue

    @staticmethod
    def _fix_from_belief(query: str, belief) -> "ObjectFix":
        """Synthesize an ObjectFix from a belief.

        Preferred: the belief's remembered REAL point cloud (true shape ->
        true jaw width; a banana reads 35mm, not the 78mm of its bbox slab),
        re-centered on the fused position. Fallback: an axis-aligned box at
        the remembered extent (dense enough for both the OBB planner and the
        GraspGen-X backend)."""
        from ..types import ObjectFix

        center = np.asarray(belief.position, dtype=float).reshape(3)
        pts_mem = getattr(belief, "points", None)
        if pts_mem is not None and len(pts_mem) >= 50:
            import os
            require_cuda = os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1"
            if require_cuda:
                from ..perception.cuda_math import tensor, shift_cloud
                pts = tensor(pts_mem)
            else:
                pts = np.asarray(pts_mem, dtype=float)
            obb_center, extents, axes = oriented_bbox(pts)
            # Re-center on the fused position in XY ONLY.
            #
            # The fused position tracks a detection centroid and, for a body
            # whose origin is not its geometric centre, differs from the
            # cloud's bbox centre in z for reasons that have nothing to do
            # with where the object is. MEASURED on LIBERO's akita bowl: the
            # body origin sits 52 mm below the mesh top and 26 mm below the
            # cloud's bbox centre, so shifting z dragged the cloud down until
            # its top read 0.9242 against a true rim at 0.9507. The rim grasp
            # was then planned 29.5 mm BELOW the rim, on the outer wall.
            #
            # Heights come from the cloud, which is measured surface; xy comes
            # from the fused estimate, which is what tracking is good at.
            shift = center - obb_center
            shift[2] = 0.0
            pts = shift_cloud(pts, shift) if require_cuda else pts + shift
            det = Detection(
                label=belief.label, conf=float(belief.conf),
                bbox=np.zeros(4, dtype=np.float32),
            )
            return ObjectFix(
                label=query, position=center.copy(), points=pts,
                detection=det, extent=extents, axes=axes,
            )
        ext = (np.sort(np.abs(np.asarray(belief.extent, dtype=float)))[::-1]
               if belief.extent is not None else np.array([0.05, 0.05, 0.05]))
        half = np.clip(ext[:3] / 2.0, 0.01, 0.2)
        top_z = float(belief.top_z) if belief.top_z is not None else float(center[2] + half[2])
        bottom_z = top_z - 2 * half[2]
        import os
        if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
            from ..perception.cuda_math import box_cloud
            pts = box_cloud(center, half, top_z, bottom_z)
        else:
            rng = np.random.default_rng(0)
            pts = []
            for ax in range(3):  # sample the 6 box faces
                for sign in (-1.0, 1.0):
                    p = (rng.random((60, 3)) - 0.5) * 2 * half
                    p[:, ax] = sign * half[ax]
                    pts.append(p)
            pts = np.concatenate(pts)
            pts[:, 2] = np.clip(pts[:, 2] + (top_z + bottom_z) / 2, bottom_z, top_z)
            pts[:, :2] += center[:2]
        det = Detection(
            label=belief.label, conf=float(belief.conf),
            bbox=np.zeros(4, dtype=np.float32),
        )
        return ObjectFix(
            label=query,
            position=np.array([center[0], center[1], (top_z + bottom_z) / 2]),
            points=pts,
            detection=det,
            extent=np.array([2 * half[0], 2 * half[1], 2 * half[2]]),
            axes=np.eye(3),
        )

    # ── skills ───────────────────────────────────────────────────────────

    def skill_halt_motion(self, reason: str = "superseded") -> dict:
        """Stop the motion currently in flight, without latching an e-stop.

        VoLo's `monitor - halt - redirect`: a physical agent has to be able to
        abandon an action that has stopped being the right one, because the
        world keeps moving while it thinks. Takes effect within one 50 Hz
        waypoint since `SafetyHarness.approve()` is what enforces it.

        The halt clears itself when the next motion starts, so the normal
        recovery is simply to issue the corrected command. Use `stop` (e-stop)
        instead when the rig is actually unsafe.
        """
        self.arm.harness.halt(reason)
        # the note reaches the agent (MCP world_state, task_memory): the reason is the
        # agent's own free text, so the rules' triggers are withheld (the harness keeps it whole)
        self.memory.add("note", f"halted motion: {_without_rule_triggers(reason)}")
        return {"ok": True, "halted": reason,
                "note": "clears automatically when the next motion begins"}

    def skill_get_observation(self) -> dict:
        try:
            return self._describe_observation(self.observe_fresh())
        except SlowPerceptionError as slow:
            # The camera was live; the ANALYSIS outlasted the age limit. The
            # first inference in a new process is the slow one (3.5 s cold vs
            # 0.02 s warm, measured on the kitchen scene), and a kitchen
            # launch's first describe_scene found its frame 13.6 s old after
            # analysis -- reported as "wait for reconnection", which failed the
            # proof at step 1. Analyze ONE newer frame (observe_fresh fences on
            # a newer capture); a second slow analysis raises honestly.
            logger.warning("slow perception, analyzing a newer frame: %s", slow)
            observation = self._describe_observation(self.observe_fresh())
            observation["observation_retry"] = str(slow)
            return observation

    def _describe_observation(self, frame: Frame) -> dict:
        """Analyze exactly the supplied frame, including a verified reset frame."""
        _frame_age_s(frame)
        self._sample_link_state(frame)  # B39: the arm's pose for THIS image
        analysis_started = time.monotonic()
        dets = detect_vocabulary(self.detector, frame, self._default_classes)
        self._show_detections(dets)
        objects = self._update_beliefs_from_frame(frame, dets)
        self.memory.add(
            "observation",
            f"saw {[o['label'] for o in objects]} (depth: {frame.depth_source})",
            rgb=frame.rgb,
        )
        # A pure LOOK must not power the motors: a LazyArm that has not
        # materialized yet reports standby instead of being poked awake.
        if getattr(self.arm.raw, "connected", True):
            state = self.arm.get_state()
            tcp = self._to_world(self.kin.fk(state.q)[:3, 3])    # reported in the table frame
            wf = self._gripper_width_frac()
            robot = {
                "q_deg": [round(float(np.degrees(x)), 1) for x in state.q],
                "tcp_xyz": [round(float(x), 3) for x in tcp],
                "gripper_open_frac": round(wf, 2) if wf is not None else "unknown",
                "holding": self.held_object,
            }
        else:
            robot = {"status": ("standby (motors off: arm control not opened in this session, "
                                "or released at rest by move_to_rest)"),
                     "live_arm_feedback": False,
                     "holding": self.held_object}
        if getattr(type(self.arm.raw), "motion_status", None) is not None:
            robot["motion_control"] = self.arm.raw.motion_status
        return {
            "objects_visible": objects,
            # The fused store includes both recent and remembered objects.
            # Calling the whole list 'remembered' made native chat describe
            # freshly detected props as absent from the current camera frame.
            "objects_tracked": self.beliefs.summary(),
            "observation_note": (
                "objects_visible contains detections from this camera frame, with measured colors. "
                "Detector labels are estimates, not confirmed object identities. "
                "objects_tracked is the fused history and may include the same detections; "
                "read each entry's state and age_s. A remembered entry is a last-known position, "
                "not proof of current visibility. Use the current camera image to identify objects; "
                "do not repeat a detector name that the image does not support. "
                "Configured zones identify fixed destinations, not current visibility."
            ),
            "configured_zones": self._configured_zones(),
            "observation_frame": {"frame_id": int(frame.frame_id),
                                  "frame_age_s": _analyzed_frame_age_s(
                                      frame, time.monotonic() - analysis_started)},
            "robot": robot,
            "depth_source": frame.depth_source,
        }

    def _configured_zones(self) -> list[dict]:
        """The running arm's configured destinations, REPORTED in the table
        frame: a profile states them in that arm's base frame (spec §3.6)."""
        grasp = self.cfg.get("grasp", {})
        entries = []
        if grasp.get("drop_zone") is not None:
            entries.append((grasp.get("drop_zone_name", "drop zone"), "delivery_area",
                            grasp.get("drop_zone")))
        box = grasp.get("open_box")
        if box is not None:
            if not hasattr(box, "get"):
                raise SkillError("Configured open box requires a finite XY center")
            entries.append(("open box", "container", box.get("center_xy_m")))
        zones = []
        for name, kind, position in entries:
            try:
                center = np.asarray(position, dtype=float)
            except (TypeError, ValueError) as exc:
                raise SkillError("Configured zone requires a finite XY center") from exc
            if (not isinstance(name, str) or not name.strip()
                    or center.shape != (2,) or not np.isfinite(center).all()):
                raise SkillError("Configured zone requires a name and a finite XY center")
            if self.T_world_base is not None:
                # on the arm's table plane (level mounts: any z gives this xy)
                table_z = float(self.cfg.safety.get("table_z", 0.0))
                center = np.asarray(self._to_world([center[0], center[1], table_z]), dtype=float)[:2]
            zones.append({"name": name, "kind": kind, "source": "configuration",
                          "position_xy_m": center.tolist(), "visibility": "not_established"})
        return zones

    def skill_list_objects(self) -> dict:
        return {
            "objects": self.beliefs.summary(),
            "holding": self.held_object,
            "note": ("Tracked objects only, not an exhaustive scene inventory; labels may be approximate. "
                     "A missing name does not prove absence. get_observation refreshes the camera; "
                     "localize_object or pick_and_place can actively search for a requested object. "
                     "state=remembered means not currently visible; position is last known"),
        }

    def skill_localize_object(self, label: str, spatial_hint: str | None = None) -> dict:
        query = str(label).strip().lower()
        zones = self._configured_zones()
        for zone in zones:
            if (query == zone["name"].strip().lower()
                    or (zone["kind"] == "delivery_area" and _names_drop_zone(query))
                    or (zone["kind"] == "container" and query in _OPEN_BOX_WORDS)):
                return {"label": label, **zone,
                        "note": "Configured destination; use a fresh camera image to establish visibility."}
        if _names_drop_zone(query):
            raise SkillError("No delivery zone is configured")
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        color = detection_color(frame.rgb, fix.detection)
        world = self._fix_to_world(fix)    # the arm-frame fix, reported and remembered in the table frame
        # Beliefs live under DETECTOR labels (what the watcher re-fuses);
        # recording the user's query words would create ghost duplicates.
        self.beliefs.update(
            fix.detection.label or fix.label, world.position, fix.detection.conf,
            extent=fix.extent, top_z=float(world.points[:, 2].max()), t=frame.t,
            color=color,
        )
        return {
            "label": label,
            **self._target_resolution(label, frame, fix),
            "detected_as": fix.detection.label,
            "color": color,
            "position": [round(float(x), 3) for x in world.position],
            "extent_m": [round(float(x), 3) for x in fix.extent],
            "n_points": int(fix.points.shape[0]),
        }

    def skill_preview_grasp(
        self,
        label: str,
        spatial_hint: str | None = None,
    ) -> dict:
        """VIA-style waypoint preview (arXiv 2607.11119): plan the grasp and
        REPORT it without moving. Lets the agent observe the proposed gripper
        waypoint (position, approach, confidence, learned-memory prior) and
        decide before committing -- the observe-then-act loop, not blind
        execution. Follow with grasp_object to actually execute it."""
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        # Frame-backed candidates and measured support use the same RGB-D
        # capture as localization; do not fetch a later wrist-camera pose.
        grasps = self._plan_grasps(fix, label=label, **(
            {"_frame": frame} if str(self.cfg.grasp.get("backend", "obb")) in _FRAME_GRASP_BACKENDS
                or str(self.cfg.grasp.get("support_height_source", "configured")) == "depth"
            else {}))
        if not grasps:
            return {"ok": False, "error": f"no grasp candidates for {label!r}"}
        g = grasps[0]  # already reranked by the fake-RL memory prior
        appr = np.asarray(g.approach, dtype=float)
        vert = float(-appr[2] / (np.linalg.norm(appr) + 1e-9))
        prior = None
        try:
            prior = self.grasp_memory.prior(SkillRuntime._grasp_memory_label(self, label), fix)
        except Exception:
            pass
        out = {
            "ok": True,
            "object": label,
            # planned in the running arm's frame, reported in the table frame
            "object_xyz": [round(float(x), 3) for x in self._to_world(fix.position)],
            "planned_grasp": {
                "tcp_xyz": [round(float(x), 3) for x in self._to_world(g.position)],
                "approach": ("top-down" if vert > 0.7
                             else "angled" if vert > 0.3 else "side"),
                "approach_vert": round(vert, 2),
                "jaw_width_m": round(float(g.width_m), 3),
                "confidence": round(float(getattr(g, "quality", 0.0)), 2),
                "candidates": len(grasps),
            },
            "hint": "call grasp_object to execute, or reposition/re-localize "
                    "if this waypoint looks wrong.",
        }
        if prior:
            out["memory_prior"] = {
                "seen": prior["wins"] + prior["losses"],
                "success_rate": prior["success_rate"],
                "avoid": prior.get("top_fail"),
            }
        self.memory.add("note", f"grasp preview {label!r}: {out['planned_grasp']['approach']} "
                        f"conf {out['planned_grasp']['confidence']}")
        return out

    @grasp_evidence.record_attempt
    @_observed_selection_on_failure
    def skill_grasp_object(
        self,
        label: str,
        material: str | None = None,
        spatial_hint: str | None = None,
        _fix=None,
        _frame=None,
        _planning_deadline=None,
        _next_candidate=None,
        _observed_record=None,
    ) -> dict:
        """Grasp a named object.

        `_fix`/`_frame` are an internal entry point for callers that already
        localized the target by other means (see `skill_grasp_at_pixel`). They
        skip the detector lookup and reuse this method's grasp planning,
        harness checks and postcondition verification verbatim, rather than
        duplicating that logic in a second code path where the two would
        inevitably drift apart.

        `_next_candidate` is internal too: with the observed-finger gate, a
        candidate the live gate vetoed before its step was commanded (nothing
        touched) and that withdrew to its pregrasp over a vetted route
        re-enters here for the next vetted candidate of the SAME plan, with
        the same immutable observed scene, the halt baseline of the first
        call and the first call's bounded neighbour ranking
        (`_ObservedNextCandidate`). `_observed_record` is filled by the
        `_observed_selection_on_failure` wrapper.
        """
        scene_enabled = os.environ.get("CASCADE_OBSERVED_FINGER_GATE") == "1"
        # A latched e-stop refuses the grasp as itself, before any work: its
        # candidates' vets all read "e-stop latched", and the agent was told
        # "NoExecutableGrasp ... approach unsafe: e-stop latched" four times
        # over, a planning failure it may retry (#414 review Important 1).
        if getattr(getattr(self.arm, "harness", None), "estopped", False) is True:
            raise SafetyViolation("e-stop latched")
        # R5: another arm's provisional hold keeps its slot: refused before
        # any work, the VLA route included.
        SkillRuntime._refuse_on_another_arms_provisional(self, "grasp_object")  # runtime doubles too
        # A new grasp forfeits an approach column kept after a STOP below an
        # earlier grasp's pregrasp (stop_retrace.py), whoever calls it.
        stop_retrace.forget(self)
        carry_attachment.check(self)
        # Native endpoint inspection also runs without the observed-finger
        # gate; a stop/clear during that passive work must invalidate it.
        scene_halt_generation = (self.arm.harness._halt_generation if _next_candidate is None
                                 else _next_candidate.halt_generation)
        self._reconcile_held()
        if self.held_object:
            raise SkillError(self._already_holding_refusal("grasp_object"))
        # Before any command: the pre-grasp opening would drop a possible hold.
        self._raise_on_a_possible_hold()
        self._held_support_offset_m = None
        gcfg = self.cfg.grasp
        if _fix is not None and _frame is not None:
            frame, fix = _frame, _fix
        else:
            frame, fix = self._localize(label, spatial_hint=spatial_hint)
        # A runtime double without the arm frame (no base_pose) records as before.
        grasp_evidence.localized(frame, fix, getattr(self, "extrinsics", None),
                                 T_world_base=getattr(self, "T_world_base", None))
        # Derived envelope features from the fix (memory/envelope.py #4):
        # the object's top above its support plane -- a top-down camera
        # never sees the sides, so the cloud's z-span is NOT a height -- and
        # the narrower horizontal footprint extent. Measured here, before
        # planning, so a grasp that dies at IK still records what it saw.
        try:
            support = self.cfg.grasp.get("source_support_top_z_m")
            support = (float(support) if support is not None
                       else float(self.arm.harness.limits.table_z))
            self._note_measurement(
                object_height_m=float(fix.points[:, 2].max()) - support,
                object_width_m=self._horizontal_footprint_m(fix.points),
            )
        except Exception:  # noqa: BLE001 -- a measurement failure is a missing feature, never a grasp failure
            pass
        profile = select_profile(fix.detection.label or label, material)
        grasp_evidence.event("material_profile", profile=profile)
        if not (_fix is not None and _frame is not None):
            from ..grasping.vla_executor import executor_name

            if executor_name(gcfg) == "vla":
                # B49, opt-in: a language-conditioned policy serves this
                # label grasp; pixel-addressed grasps stay analytic.
                return self._vla_grasp(label, frame, fix, profile, scene_halt_generation)
        # B36: the post-contact hold for this backend, validated before
        # anything moves (None = the plain two-stage close).
        hold_squeeze = SkillRuntime._grasp_hold_squeeze(self)
        # The bound on waiting for empty jaws to open (the opening of closed
        # jaws before selection, an air grasp's reopen), validated before
        # anything moves, not first after the close.
        SkillRuntime._jaws_open_timeout_s(self)
        # Ranking distance for candidates beside an observed neighbour, read
        # and validated before anything moves. Ranking only: it never admits
        # or refuses a candidate (0 = rank as main).
        rank_lateral_m = rank_max = None
        if scene_enabled:
            rank_lateral_m = gcfg.get("observed_rank_lateral_m", OBSERVED_RANK_LATERAL_M)
            if (isinstance(rank_lateral_m, bool) or not isinstance(rank_lateral_m, (int, float))
                    or not math.isfinite(rank_lateral_m)
                    or not 0 < rank_lateral_m <= OBSERVED_RANK_LATERAL_MAX_M):
                raise SkillError("grasp.observed_rank_lateral_m must be a distance in "
                                 f"(0, {OBSERVED_RANK_LATERAL_MAX_M}] m")
            rank_lateral_m = float(rank_lateral_m)
            rank_max = gcfg.get("observed_rank_max_candidates", OBSERVED_RANK_MAX_CANDIDATES)
            if isinstance(rank_max, bool) or not isinstance(rank_max, int) or rank_max < 0:
                raise SkillError("grasp.observed_rank_max_candidates must be an integer >= 0 (0 = no ranking)")
        search_cfg = gcfg.get("graspgenx") or {}
        search_limits = None
        if (scene_enabled and str(gcfg.get("backend", "obb")) == "graspgenx"
                and search_cfg.get("feasibility_max_batches") is not None):
            from ..grasping.planning_budget import check_limits
            # The next candidate comes from the plan already made: one batch,
            # a fresh feasibility budget, the same outer deadlines. Checked
            # before anything moves; the budget's clock starts after the
            # opening below.
            search_limits = (1 if _next_candidate is not None else search_cfg.get("feasibility_max_batches"),
                             search_cfg.get("feasibility_timeout_s"))
            check_limits(*search_limits)
        start_state = None
        if scene_enabled and _next_candidate is None:
            # Jaws round 2b (controller ruling): closed empty jaws open before
            # every per-attempt budget clock starts, the GraspGen-X planning
            # budget below and the route budget (`approach_deadline`): their
            # travel (up to `open_wait_timeout_s`, 12 s on Isaac) is never
            # charged to planning. The budgets stay as the profile sets them.
            start_state = self.arm.get_state()
            if SkillRuntime._open_closed_jaws_before_selection(
                    self, frame, fix, scene_halt_generation, start_state):
                start_state = None   # selection reads the opened jaws again
        grasp_evidence.phase("planning")
        search = None
        planning_active = False
        if search_limits is not None:
            from ..grasping.planning_budget import PlanningBudget
            def search_cancel():
                if self.arm.harness.estopped:
                    raise SafetyViolation("e-stop during grasp planning")
                self.arm.harness._check_halt_generation(scene_halt_generation)
            search = PlanningBudget(*search_limits,
                deadlines=(_planning_deadline, getattr(self, "_task_deadline", None)),
                cancel=search_cancel)
            planning_active = True
            # Outcomes may change in another caller; this attempt uses one prior.
            import copy
            frozen_prior = (copy.deepcopy(
                self.grasp_memory.prior(SkillRuntime._grasp_memory_label(self, label), fix)),)
            search.check()

        lim = self.arm.harness.limits
        z_floor = max(float(lim.workspace_min[2]) + 0.002,
                      lim.table_z + float(gcfg.get("min_grasp_z_m", 0.008)))

        def generate_batch(index):
            if search is not None:
                search.check()
                grasp_evidence.event("planning_batch", batch=index + 1,
                    max_batches=search.max_batches, deadline_monotonic_s=search.deadline,
                    seed=None, rng="server RNG; raw response captured, no seed override")
            from ..grasping.graspgenx_backend import NoEligibleGrasps
            try:
                candidates = self._plan_grasps(fix, label=label, **({} if search is None else {
                    "_deadline": search.deadline, "_check": search.check,
                    "_prior_snapshot": frozen_prior}), **(
                    # Observed support and frame backends share the localization capture.
                    {"_frame": frame} if str(gcfg.get("backend", "obb")) in _FRAME_GRASP_BACKENDS
                        or str(gcfg.get("support_height_source", "configured")) == "depth"
                    else {}))
            except NoEligibleGrasps as exc:
                if search is None:
                    raise
                search.check()
                # Valid raw batch, unchanged score/approach constraints removed
                # every pose. Feed explicit exhaustion to the same selector loop.
                grasp_evidence.event("planning_batch_filtered_empty", batch=index + 1, reason=str(exc))
                candidates = []
            grasp_evidence.event("planned_candidates", grasps=candidates,
                backend=getattr(self, "grasp_planner_used", None),
                z_nudge_m=getattr(self, "_last_grasp_z_nudge", None))
            # Preserve the existing small below-table correction, independently
            # on every fresh proposal (never compound a previous batch's nudge).
            for candidate in candidates:
                dz = z_floor - float(candidate.position[2])
                if 0.0 < dz <= 0.03:
                    candidate.position = np.asarray(candidate.position, dtype=float).copy()
                    candidate.position[2] = z_floor
            grasp_evidence.event("floor_adjusted_candidates", grasps=candidates, z_floor_m=z_floor)
            if search is not None:
                search.check()
            return candidates

        def planning_state():
            if search is None:
                return self.arm.get_state()
            # Additional safety reads must not enlarge Isaac's normal RPC cap.
            budget = search.remaining(self.cfg.arm.get("motion_rpc_timeout_s", 1.0))
            result = self.arm.get_state(timeout_s=budget)
            search.check()
            return result

        # Observed-finger choice record (result + trace): ranked last, dropped
        # by the vet, and each attempted candidate. Shared by re-entries.
        selection = (_next_candidate.selection if _next_candidate is not None
                     else _ObservedSelection() if scene_enabled else None)
        if selection is not None and _observed_record is not None:
            _observed_record["selection"] = selection
        # THE sim retry (spec §4.5) plans without the grasps its refused attempt
        # tried; outside it nothing is excluded and every plan stays as it was.
        retry_state = _sim_retry_state(self)
        others = _OtherThanRefused(() if retry_state is None else retry_state.exclude)
        if _next_candidate is not None:
            grasps = list(_next_candidate.remaining)  # no new plan: the next vetted candidate
        else:
            grasps = others(generate_batch(0))
            if selection is not None:
                selection.add_batch(grasps)

        grasp_evidence.event("floor_adjusted_candidates", grasps=grasps, z_floor_m=z_floor)

        # Pre-vet every candidate against the harness geometry (with the
        # exemption cylinder the descent will open) so a doomed candidate
        # loses the ranking up front instead of aborting mid-motion.
        harness = self.arm.harness
        exempt_r = float(gcfg.get("exempt_radius_m", 0.07))
        scene_gate = None

        def _scene_cancel():
            if harness.estopped:
                raise SafetyViolation("e-stop during observed-finger preflight")
            harness._check_halt_generation(scene_halt_generation)
            if planning_active:
                search.check()

        def _scene_duration(start, target, requested):
            # Identical velocity stretch to SafeArm; its actual-start preflight
            # repeats this geometry check with the final stream duration.
            delta = float(np.max(np.abs(np.asarray(target) - np.asarray(start))))
            return max(float(requested), 1.875 * delta / (0.9 * harness.limits.max_joint_vel))

        def _scene_path(start, target, requested):
            from ..control.motion_profile import resolve_motion_rate
            return scene_gate.profile(start, target, _scene_duration(start, target, requested),
                                      resolve_motion_rate(self.arm.raw), check=_scene_cancel)

        def _scene_motion(target):
            if scene_gate is None:
                return {}
            from ..control.motion_profile import resolve_motion_rate
            def preflight(start, duration):
                _scene_cancel()
                rate = resolve_motion_rate(self.arm.raw)
                deadline = time.monotonic() + PLAN_BUDGET_S
                with geometry_guard(harness, deadline=deadline):
                    reason = vet_segment(harness, start, target, duration,
                        deadline=deadline, stretch=False, rate_hz=rate)
                if reason:
                    raise SafetyViolation(f"observed-finger route became unsafe: {reason}")
                _scene_cancel()
                scene_gate.require_profile(start, target, duration,
                                            rate, check=_scene_cancel)
            def before_stream():
                _scene_cancel()
                harness.check_stream_start(halt_generation=scene_halt_generation)
            def trajectory_preflight(start, profile, check):
                _scene_cancel()
                # The native planner's curve replaces the min-jerk profile,
                # so every actual command/50 Hz safety endpoint must pass the
                # same immutable observed-finger geometry before streaming.
                for item in profile:
                    for previous, following, _dt in item.checks:
                        for q in (previous, following):
                            check()
                            _scene_cancel()
                            conflict = scene_gate.pose(q)
                            if conflict:
                                from ..grasping.observed_scene import precommand_refusal
                                raise precommand_refusal(
                                    "planned finger trajectory intersects observed surface", conflict)
                _scene_cancel()
            return {"preflight": preflight, "before_stream": before_stream,
                    "feedback_guard": scene_gate.feedback, "_halt_generation": scene_halt_generation,
                    "_trajectory_preflight": trajectory_preflight}
        # Selection's state: the grasp-start read when the jaws were left as
        # they were (PR-A jaws round: open jaws get no extra read and no extra
        # command), else a read of the opened jaws.
        state = start_state if start_state is not None else planning_state()
        if search is not None:
            search.check()
        # The route budget starts here, after the grasp-start opening above
        # (jaws round 2, I1; round 2b): on Newton the jaws take ~1.5 s to open
        # from closed, and selection needs its whole budget after them.
        from ..safety.trajectory import PLAN_BUDGET_S, geometry_guard, vet_segment
        approach_deadline = time.monotonic() + PLAN_BUDGET_S
        # B73 opt-in: forearm/wrist/gripper clearance of the analytic planner's
        # angled candidates (None unless grasp.angled_clearance_vet is set).
        clearance_check = (self._angled_clearance_check(fix)
                           if gcfg.get("angled_clearance_vet", False) is not False else None)
        preserve_view_home = self._home_policy_enabled()

        def _vet(g, q_pre, q_grasp):
            planner = getattr(self.arm, "motion_planner", None)
            if planner is not None:
                # IK and the harness do not contain the native planner's
                # self-collision spheres. Reject endpoints before choosing a
                # grasp, while another independently vetted candidate can win.
                for phase, q in (("pregrasp", q_pre), ("grasp", q_grasp)):
                    _scene_cancel()
                    reason = planner.configuration_rejection(q)
                    _scene_cancel()
                    if reason:
                        return f"{phase} unsafe: {reason}"
            # The approach leg executes BEFORE allow_grasp_descent opens the
            # cylinder: vet the pregrasp with NO exemption, or a candidate
            # that needed one is guaranteed to abort at the end of the
            # approach. The descent leg holds the exemption, and min-jerk is
            # a straight segment in joint space, so sampling the q_pre ->
            # q_grasp segment vets the actual executed path, not just its
            # endpoints (near-horizontal approaches can dip below the floor
            # OUTSIDE the cylinder mid-descent).
            # ...except the column directly above the TARGET, which the
            # approach enters by construction (see _target_column). The same
            # column is held during the executed approach below, so vetting
            # and execution agree. Neighbours, the table and every other arm
            # point stay fully checked.
            column = _target_column(gcfg, harness, g, fix) if scene_gate is None else None
            reason = harness.vet_pose(q_pre, **_column_kw(column))
            if search is not None:
                search.check()
            if reason:
                return f"pregrasp unsafe: {reason}"
            if scene_gate is not None:
                def endpoint_check():
                    _scene_cancel()
                    if time.monotonic() > approach_deadline:
                        raise SkillError("route preflight exceeded its time budget; no route authorized")
                # The same immutable closing envelope can reject a candidate
                # before its expensive approach/descent profiles. Keep the
                # initial harness/map admission above, and run every original
                # route and endpoint check before accepting any candidate.
                endpoint_check()
                conflict = scene_gate.closing_pose(q_grasp)
                endpoint_check()
                if conflict:
                    grasp_evidence.event("observed_finger_candidate_rejected", phase_name="close",
                                         conflict=conflict, q_pre=q_pre, q_grasp=q_grasp)
                    return f"closing fingers intersects observed non-target surface: {conflict}"
                # Endpoints use the same immutable frame as the profiles.
                # Reject hidden fingers before spending the shared planning
                # budget sampling a route that cannot be accepted anyway.
                for endpoint, q, closing in (("pregrasp", q_pre, False),
                                             ("grasp_open", q_grasp, False),
                                             ("grasp_closing", q_grasp, True)):
                    conflict = scene_gate.occluded_pose(q, closing=closing, check=endpoint_check)
                    if conflict:
                        grasp_evidence.event("observed_finger_candidate_rejected", phase_name=endpoint,
                                             conflict=conflict, q_pre=q_pre, q_grasp=q_grasp)
                        return f"{endpoint} finger endpoint occluded by non-target depth: {conflict}"
            with _column_exemption(harness, column), \
                    geometry_guard(harness, deadline=approach_deadline):
                reason = vet_segment(
                    harness, _seed, q_pre,
                    float(gcfg.get("move_duration_s", 2.5)),
                    deadline=approach_deadline,
                )
            if reason:
                return f"approach unsafe: {reason}"
            # Exemption floor a bit BELOW the table: the reBot RS gripper's
            # finger/wrist links extend ~5 cm below the TCP, so a top-down
            # grasp of a low object legitimately dips a link to z ~ -0.018
            # (below the table plane) while the fingers straddle the object.
            # Flooring the exemption at table_z left that link 18 mm outside
            # the cylinder -> false "link would hit the table" abort. The
            # narrow XY radius still confines this to directly over the
            # target, so allowing a small sub-table dip there is safe.
            z_min = float(harness.limits.table_z) - 0.06
            # Retain the shared study/runtime diagnostic samples, then vet
            # the entire descent more densely for long seven-joint sweeps.
            for fraction in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
                q = q_pre + fraction * (np.asarray(q_grasp) - np.asarray(q_pre))
                reason = harness.vet_pose(
                    q, exempt_xy=g.position[:2],
                    exempt_radius_m=exempt_r, exempt_z_min=z_min,
                )
                if reason:
                    return f"descent unsafe: {reason}"
            reason = self._joint_segment_rejection(
                q_pre, q_grasp, exempt_xy=g.position[:2],
                exempt_radius_m=exempt_r, exempt_z_min=z_min,
            )
            if reason:
                return f"descent unsafe: {reason}"
            if clearance_check is not None and isinstance(g, AngledGrasp):
                # The harness proxies above are joint origins, exempt inside
                # the grasp cylinder; a leaning approach can still put the
                # housing, wrist or a finger into the table or the target.
                reason = clearance_check(g, q_pre, q_grasp)
                if reason:
                    return reason
            if scene_gate is not None:
                for phase, start, end, duration in (
                        ("pregrasp", _seed, q_pre, gcfg.get("move_duration_s", 2.5)),
                        ("descent", q_pre, q_grasp, gcfg.get("descend_duration_s", 2.0))):
                    conflict = _scene_path(start, end, duration)
                    if conflict:
                        grasp_evidence.event("observed_finger_candidate_rejected", phase_name=phase,
                                             conflict=conflict, q_pre=q_pre, q_grasp=q_grasp)
                        return f"{phase} finger intersects observed surface: {conflict}"
            if bool(gcfg.get("pre_carry_lift", False)) and gcfg.get("carry_height_m") is not None:
                # A learned tilted grasp can solve at pickup height yet have
                # no IK at the carry height. Reject it before closing on the
                # object, while another model candidate can still be chosen.
                pose = self.kin.fk(q_pre).copy()
                height = float(gcfg.get("carry_height_m"))
                if pose[2, 3] < height:
                    pose[2, 3] = height
                    lifted = self.kin.ik(pose, q_pre)
                    if (not lifted.success
                            or np.max(np.abs(lifted.q - q_pre)) > np.pi):
                        return "grasp cannot reach the configured carry height"
                    if planner is not None:
                        _scene_cancel()
                        reason = planner.configuration_rejection(lifted.q)
                        _scene_cancel()
                        if reason:
                            return f"grasp carry lift unsafe: {reason}"
                    for fraction in (.15, .3, .45, .6, .75, .9, 1.):
                        reason = harness.vet_pose(q_pre + fraction * (lifted.q - q_pre))
                        if reason:
                            return f"grasp carry lift unsafe: {reason}"
            return None

        # Seed grasp IK from HOME (elbow-up), not the live pose: seeding from
        # an arbitrary current configuration can converge to an elbow-down IK
        # branch whose approach path dips a link under the table.
        _home = self.cfg.arm.get("home_q")
        _seed = (np.asarray(_home, dtype=float)
                 if _home is not None and not preserve_view_home else state.q)
        if scene_enabled:
            from ..grasping.observed_scene import for_runtime
            _scene_cancel()
            if _next_candidate is not None:
                # The same immutable observed scene every candidate of this
                # plan was vetted against; the withdrawal kept it valid. The
                # halt baseline is the first call's: a halt or e-stop since
                # the withdrawal ended the skill at the check above.
                scene_gate = _next_candidate.gate
                scene_gate.feedback(state)  # same captured epoch, never a new gate
                _scene_cancel()
            else:
                scene_gate = for_runtime(self, frame, fix, state)
            if _home is not None and not preserve_view_home:
                conflict = _scene_path(state.q, _seed, 1.5)
                if conflict:
                    raise SafetyViolation(f"home finger trajectory intersects observed surface: {conflict}")
        grasp_evidence.event("selection_input", state=state, seed_q=_seed)
        from ..grasping.selector import NoExecutableGrasp
        from ..grasping.observed_scene import NeighbourRanking, ObservedFingerGate
        pregrasp_offset_m = float(gcfg.get("pregrasp_offset_m", 0.12))
        if (selection is not None and selection.ranking is None and rank_max
                and isinstance(scene_gate, ObservedFingerGate)):
            # Beside an observed neighbour, candidates whose fingers clear it
            # by less than the settle error go last (ranking only; the exact
            # vet below still admits or drops every candidate). Bounded so the
            # choice stays inside `approach_deadline`; a re-entry for the next
            # candidate reuses this ranking (same gate, same budget).
            selection.ranking = NeighbourRanking(
                scene_gate, lateral_m=rank_lateral_m, pregrasp_offset_m=pregrasp_offset_m,
                max_candidates=rank_max, time_slice_s=OBSERVED_RANK_TIME_SLICE_S)

        def _candidates(batch_grasps):
            if selection is None:
                return batch_grasps
            ordered = (batch_grasps if selection.ranking is None
                       else selection.ranking.order(batch_grasps, check=_scene_cancel,
                                                    ranked_last=lambda i, g, why: selection.ranked(g, why)))
            return selection.track(ordered)

        def _validate(g, q_pre, q_grasp):
            reason = _vet(g, q_pre, q_grasp)
            if reason and selection is not None:
                selection.vet_failed(g, reason)
            return reason
        # Opt-in: pregrasp (and so the lift) at least this far above the
        # object's top (observed, or as mapped with grasp.pregrasp_map_top),
        # so the jaws leave it before the exemption closes. A map-raised
        # hover is a ladder (map hover, today's hover): never executed lower
        # than today's. See select_grasp(pregrasp_min_z=...) and _pregrasp_min_z.
        pregrasp_min_z = _pregrasp_min_z(gcfg, harness, fix)
        vetted = None  # (refusal, grasps) of the latest batch the selector vetted
        try:
            for batch in range(1 if search is None else search.max_batches):
                if batch:
                    search.check()
                    state = planning_state()
                    search.check()
                    scene_gate.feedback(state)  # same captured epoch, never a new gate
                    _scene_cancel()
                    if _home is not None and not preserve_view_home:
                        conflict = _scene_path(state.q, _seed, 1.5)
                        if conflict:
                            raise SafetyViolation(f"home finger trajectory intersects observed surface: {conflict}")
                    grasps = others(generate_batch(batch))
                    if selection is not None:
                        selection.add_batch(grasps)
                selecting, exhausted = time.monotonic(), False
                try:
                    grasp, q_pre, q_grasp = select_grasp(
                        _candidates(grasps),
                        self.kin,
                        _seed,
                        max_width_m=self._max_width,
                        pregrasp_offset_m=pregrasp_offset_m,
                        pregrasp_min_z=pregrasp_min_z,
                        validate=_validate,
                        preserve_order=True,
                        parallel_jaw_symmetry=(self.cfg.arm.get("gripper") or {}).get("parallel_jaw_symmetry"),
                        # Single-hinge jaw datum (TOOL frame): the fixed jaw's contact
                        # point and the closing direction toward the moving jaw. Per
                        # arm, like every other jaw dimension: the SO-101 beak closes
                        # against its fixed tip at the frame origin; parallel jaws
                        # omit both. See selector.select_grasp's docstring.
                        jaw_fixed_tip_m=(self.cfg.arm.get("gripper") or {}).get("jaw_fixed_tip_m"),
                        jaw_close_dir=(self.cfg.arm.get("gripper") or {}).get("jaw_close_dir"),
                        **({} if search is None else {"check": search.check}),
                    )
                    if search is not None:
                        search.check()
                        grasp_evidence.event("planning_batch_selected", batch=batch + 1)
                    break
                except NoExecutableGrasp as exc:
                    exhausted = True
                    if search is None:
                        raise
                    search.check()
                    grasp_evidence.event("planning_batch_rejected", batch=batch + 1, reason=str(exc))
                    if grasps:
                        vetted = exc, grasps
                    if batch + 1 == search.max_batches:
                        if others.last_emptied and vetted is not None:
                            # THE sim retry's last batch held only repeats of
                            # its refused attempt: the search ends with the
                            # last vetted batch's refusal, reason and grasps
                            # (grasp memory records the vet's own bucket).
                            exc, grasps = vetted
                            raise exc
                        raise
                finally:
                    if selection is not None:
                        # Choice time (ranking included) for the result/trace;
                        # vet refusals become drops only when every candidate
                        # of the batch failed.
                        selection.end_select(time.monotonic() - selecting, exhausted=exhausted)
        except (SkillError, SafetyViolation) as e:
            if isinstance(e, NoExecutableGrasp) and others.only_repeats:
                # THE sim retry planned nothing but repeats of its refused
                # attempt: no different grasp, and no planning failure of this
                # object for the grasp memory to learn
                raise SkillError(_NO_DIFFERENT_GRASP) from e
            # No candidate survived IK + harness vetting. Log it against the
            # object profile so the fake-RL memory learns to bias future
            # candidates (raise z, prefer top-down) for this object.
            try:
                self.grasp_memory.record(
                    SkillRuntime._grasp_memory_label(self, label), fix, grasps[0] if grasps else None,
                    success=False, reason=str(e),
                    z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
            except Exception:
                pass
            raise

        grasp_evidence.event("selected", grasp=grasp, q_pre=q_pre, q_grasp=q_grasp)
        if selection is not None:
            selection.attempt(grasp)

        # 1. open, go to pregrasp (normal speed). Re-home first so the
        # pregrasp IK seeds from a known elbow-up posture: seeding from an
        # arbitrary current pose can converge to an elbow/wrist-down IK
        # solution whose approach path dips a link below the table near the
        # base (observed: link7 at xy~(0.05,0.04) z=-0.023, far from the
        # target so no grasp-exemption cylinder can cover it).
        grasp_evidence.phase("open")
        if scene_gate is not None:
            _scene_cancel()
            # Planning can be long. This safety read (not logging) binds the
            # opening extrusion to current feedback before the first jaw op.
            open_state = planning_state()
            _scene_cancel()
            scene_gate.feedback(open_state)
            conflict = scene_gate.pose(open_state.q)
            if conflict:
                raise SafetyViolation(f"opening fingers intersects observed surface: {conflict}")
        if search is not None:
            search.check()  # Include the final pre-open read/veto in the same budget.
        planning_active = False  # Motion retains its separate physical-clock budget.
        # Pre-grasp opening: fully open unless the arm profile opts into the
        # adaptive opening (gripper.pregrasp_open_margin_m; Seeed WRC a2d5950
        # opened to grasp width + 10 mm). Never narrower than the grasp width.
        # Unset = exactly the old full-open command, touching nothing else.
        open_pos = self._grip_open
        margin = (self.cfg.arm.get("gripper") or {}).get("pregrasp_open_margin_m")
        if margin is not None:
            from ..grasping.force import pregrasp_open_position
            open_pos = pregrasp_open_position(
                grasp.width_m, margin, self._grip_open, self._grip_closed, self._max_width)
        # The opening goes out only for the arm moves after it: their
        # begin_motion refusals first (the e-stop, a halt since the skill
        # started, the perception watchdog), so a dead camera sends no jaw
        # command either (#414 review Minor 2). A harness double without the
        # check keeps today's order.
        before_motion = getattr(harness, "check_before_motion", None)
        if callable(before_motion):
            before_motion(**({"halt_generation": scene_halt_generation} if scene_enabled else {}))
        self._set_gripper(open_pos, effort=0.8,
                             **({"_halt_generation": scene_halt_generation} if scene_enabled else {}))
        _home = self.cfg.arm.get("home_q")
        if _home is not None and not preserve_view_home:
            grasp_evidence.phase("home")
            grasp_evidence.event("move_target", q=_home, duration_s=1.5)
            if scene_gate is not None:
                if not self._move_joints(np.asarray(_home, dtype=float), duration_s=1.5,
                                            **_scene_motion(np.asarray(_home, dtype=float))):
                    raise SkillError("did not settle at required observed-finger home pose")
            else:
                begun = SkillRuntime._motions_begun(self)
                try:
                    settled_home = self._move_planned(np.asarray(_home, dtype=float), duration_s=1.5)
                except MotionHalted:
                    # The operator pressed STOP, not the e-stop: the halt
                    # ends the grasp as itself, without latching (A3 review).
                    raise
                except Exception as exc:
                    if SkillRuntime._refused_before_motion(self, exc, begun):
                        # Refused at its start (the watchdog, the e-stop, a
                        # route preflight): the arm did not move, so no stop
                        # and no latch; the refusal as itself, the e-stop's
                        # if one is latched (#414 review Important 1).
                        if getattr(harness, "estopped", False) is True:
                            raise SafetyViolation("e-stop latched") from exc
                        raise
                    # Started (it may have stopped anywhere) or a fault
                    # outside the safety checks: stop, which latches.
                    self.arm.stop()
                    raise SkillError(f"home move failed before pregrasp: {exc}"
                                     f"{SkillRuntime._latched_note(self, exc)}") from exc
                if not settled_home:
                    self.arm.stop()
                    raise SkillError("did not settle at home pose before pregrasp"
                                     f"{SkillRuntime._latched_note(self, '')}")
        grasp_evidence.phase("pregrasp")
        grasp_evidence.event("move_target", q=q_pre, duration_s=gcfg.get("move_duration_s", 2.5))
        # The observed gate vets the exact direct profile; the mapping-only
        # path retains its fully vetted deterministic route from measured pose.
        # `approach_column`: the target column this approach entered, which an
        # air grasp's exit withdrawal re-enters (None: strict checks).
        approach_column = None
        if scene_gate is not None:
            approached = self._move_joints(q_pre,
                duration_s=float(gcfg.get("move_duration_s", 2.5)), **_scene_motion(q_pre))
        else:
            column = approach_column = _target_column(gcfg, self.arm.harness, grasp, fix)
            if column is not None:
                grasp_evidence.event("approach_target_column", center_xy=column[0],
                                     radius_m=column[1], z_min=column[2])
            with _column_exemption(self.arm.harness, column):
                approached = self._move_planned(q_pre,
                    duration_s=float(gcfg.get("move_duration_s", 2.5)))
        if not approached:
            raise SkillError("did not settle at pregrasp pose")

        # 2. descend inside the exemption cylinder (slow)
        self.arm.harness.allow_grasp_descent(
            grasp.position[:2],
            radius_m=float(gcfg.get("exempt_radius_m", 0.07)),
            z_min=float(self.arm.harness.limits.table_z) - 0.06,
        )
        from . import contact_episode
        episode = None
        contact_completed = False
        next_after = None  # a pre-command veto withdrawn over a vetted route
        held_offset_at_close = None
        close_hold_open = None  # B36: the post-contact hold's opening, when one was commanded
        stop_failure = None  # the refusal / stop that ended this block below the pregrasp
        # The lift back to the pregrasp (step 4) runs at the profile's lift
        # speed; a column kept after a STOP below the pregrasp retraces at it.
        lift_dur = float(gcfg.get("descend_duration_s", 2.0)) / max(profile.lift_speed_scale, 0.2)
        grasp_evidence.phase("descent")
        grasp_evidence.event("move_target", q=q_grasp,
                             duration_s=gcfg.get("descend_duration_s", 2.0), bias_compensate=True)
        try:
            if not self._move_joints(q_grasp,
                                        duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                                        bias_compensate=True, _linear_tool_path=True,
                                        **_scene_motion(q_grasp)):
                raise SkillError("did not settle at grasp pose")

            # 3. close with the material profile (two-stage, stall-aware).
            # From this instant the jaws may physically hold the object
            # while `held_object` is still None: an exception between here
            # and the assignment below (lift refused by the harness, an
            # e-stop, a feedback timeout) used to leave a PHYSICALLY held
            # object LOGICALLY unheld -- the next grasp_object then opened
            # the jaws on it as "not holding anything". Record a provisional
            # marker first; `_reconcile_held` promotes it (jaws stalled on
            # something) or clears it (jaws closed on air) on the next call.
            # It carries the opening this close commands and the one the
            # object should stop the jaws at: reset_scene judges the jaws of
            # an interrupted grasp with them, as the verify below does.
            provisional = ProvisionalHold(
                (label, fix.detection.label, detection_color(frame.rgb, fix.detection)),
                commanded_open=SkillRuntime._close_commanded_open(self, profile),
                expected_open=min(grasp.width_m / self._max_width, 1.0), arm=self.arm)
            grasp_evidence.phase("close")
            close_guard = None
            if scene_gate is not None:
                _scene_cancel()
                def close_guard():
                    nonlocal episode
                    close_deadline = time.monotonic() + PLAN_BUDGET_S
                    task_deadline = getattr(self, "_task_deadline", None)
                    if task_deadline is not None:
                        close_deadline = min(close_deadline, task_deadline)
                    def close_check():
                        _scene_cancel()
                        if time.monotonic() > close_deadline:
                            raise SafetyViolation("closing preflight exceeded its time budget")
                    close_check()
                    # Separate safety read: the measured pose can differ from
                    # selected IK after descent or between closing stages.
                    _scene_cancel()
                    close_state = self.arm.get_state()
                    _scene_cancel()
                    scene_gate.require_closing(close_state, check=close_check)
                    close_check()
                    grasp_evidence.event("observed_finger_close_preflight",
                                         physics_clock=close_state.physics_clock,
                                         q=close_state.q, gripper_joints=close_state.gripper_joints,
                                         stroke_lower_m=scene_gate.closing_lower,
                                         stroke_upper_m=scene_gate.closing_upper,
                                         scope="pre-command full finger stroke; target mask exempt only for closing")
                    # A failed first veto has not attempted closure. Preserve
                    # a provisional hold only once a jaw command is about to
                    # be attempted; a later stage/transport failure may hold.
                    if episode is None:
                        episode = contact_episode.begin(self, q_pre)
                    self._held_provisional = provisional
            else:
                episode = contact_episode.begin(self, q_pre)
                self._held_provisional = provisional
            try:
                close_hold_open = SkillRuntime._grasp_close(
                    self, profile, hold_squeeze,
                    **({"_halt_generation": scene_halt_generation,
                        "_before_close": close_guard} if scene_enabled else {}))
            except GraspFailed as failed:
                if not self._force_contact_close_enabled() or episode is not None:
                    raise
                # Native Grasp completed without securing an object. Retrace
                # its vetted descent, with the original contact exemption;
                # preserve jaws/provisional state and let STOP/fault checks win.
                # The grasp had already failed: a halt in the retreat carries
                # that failure (`_keep_decided_failure`).
                try:
                    settled = self._move_joints(q_pre, duration_s=float(gcfg.get("descend_duration_s", 2.0)))
                except MotionHalted as halted:
                    _keep_decided_failure(halted, f"{type(failed).__name__}: {failed}")
                    raise
                if not settled:
                    self.arm.stop()
                    raise SkillError("did not settle while retreating after failed grasp")
                raise
            if close_hold_open is not None:
                # The post-contact hold is the opening the close commanded last.
                provisional.commanded_open = SkillRuntime._close_commanded_open(self, profile, close_hold_open)
            contact_episode.wait_geometry(self, episode)
            self._held_support_offset_m = None
            # Approximate aiming compensation uses the closed grasp pose,
            # before lift. It is not an independent measurement of a held body.
            close_state = carry_attachment.arm(self, episode)
            if getattr(self.arm, "motion_planner", None) is not None:
                from ..planning.runtime import wait_for_contact_stability
                generation = scene_halt_generation if scene_enabled else harness._halt_generation
                def stability_guard():
                    harness.check_stream_start(halt_generation=generation)
                    carry_attachment.check(self)
                stability = {}
                try:
                    close_state, stability = wait_for_contact_stability(self.arm, close_state,
                        source=(str(self.cfg.arm.get("bridge_host", "127.0.0.1")),
                                int(self.cfg.arm.get("bridge_port", 8611))),
                        robot_id=self.cfg.arm.bridge_robot_id,
                        timeout_s=gcfg.get("close_feedback_timeout_s") or self.cfg.arm.get("settle_timeout_s", 6.),
                        rpc_timeout_s=self.cfg.arm.get("motion_rpc_timeout_s", 1.), progress=stability,
                        check=stability_guard, observe=lambda state: carry_attachment.observe(self, state))
                finally:
                    grasp_evidence.event("post_close_stability", **stability)

            tcp_close = self.kin.fk(close_state.q)[:3, 3]
            # Derived envelope features from the jaws' own close state: the
            # TCP height when they closed and how far the tool landed from
            # the perceived centre. FK of MEASURED joints, not the planned
            # grasp pose -- the planned value is what the raw args already
            # proxied; this is where the arm actually was.
            try:
                self._note_measurement(
                    tcp_z_at_grasp_m=float(tcp_close[2]),
                    object_tcp_lateral_offset_m=float(np.hypot(
                        float(fix.position[0]) - float(tcp_close[0]),
                        float(fix.position[1]) - float(tcp_close[1]))),
                )
            except Exception:  # noqa: BLE001 -- missing feature, never a grasp failure
                pass
            try:
                held_offset_at_close = np.asarray(fix.position, float) - tcp_close
                if held_offset_at_close.shape != (3,) or not np.isfinite(held_offset_at_close).all():
                    held_offset_at_close = None
            except (AttributeError, TypeError, ValueError):
                held_offset_at_close = None
            if gcfg.get("place_support_clearance_m") is not None and float(-grasp.approach[2]) > .95:
                # A partial cloud's lowest visible point can be several mm
                # above the object's bottom. When the pickup support plane
                # is calibrated, use it; otherwise retain the observed bound.
                bottom = self._grasp_support_height(fix.points)
                offset = float(tcp_close[2]) - bottom
                if np.isfinite(offset) and 0 < offset < .15:
                    self._held_support_offset_m = offset

            # 4. lift back to pregrasp (speed scaled by profile: lift_dur)
            grasp_evidence.phase("lift")
            q_lift = q_pre
            if getattr(self.arm, "motion_planner", None) is not None:
                from ..planning.runtime import contact_lift_target
                q_lift = contact_lift_target(self.kin, close_state.q, q_pre)
            grasp_evidence.event("move_target", q=q_lift, duration_s=lift_dur)
            # A halt since this grasp started (the close included) refuses the
            # lift before its first command, on every path: without the
            # baseline, begin_motion would clear a halt raised during the close
            # and the arm would rise after an operator STOP (A3 review).
            if not self._move_carry(q_lift, duration_s=lift_dur,
                    _linear_tool_path=True, _halt_generation=scene_halt_generation):
                raise SkillError("did not settle at grasp lift pose")
            contact_completed = True
        except (SkillError, SafetyViolation) as exc:
            stop_failure = exc
            # A failed descent can leave the open tool intentionally close to
            # the target. Finish the same contact episode by withdrawing to
            # its vetted pregrasp before removing its original exemption.
            # After close, a possible payload needs its own observed geometry;
            # do not invent an empty-tool recovery or open the jaws here.
            if (getattr(self, "_held_provisional", None) is None and not harness.estopped
                    and getattr(harness, "_halt", None) is None):
                try:
                    if scene_gate is not None:
                        settled = self._move_joints(q_pre,
                            duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                            _linear_tool_path=True,
                            **_scene_motion(q_pre))
                    else:
                        settled = self._move_planned(
                            q_pre, duration_s=float(gcfg.get("descend_duration_s", 2.0)))
                    recovery = "completed" if settled else "did not settle"
                except MotionHalted as halted:
                    # An operator halt during the withdrawal ends the skill as
                    # the halt it is, never folded into the veto's text (the
                    # e-stop and every other refusal are reported below). The
                    # descent had already failed: the halt carries that failure.
                    self.memory.add("outcome", f"grasp descent failed: {exc}; pre-close retreat halted")
                    if selection is not None:
                        selection.finish(f"failed: {str(exc)[:200]}; pre-close retreat halted")
                    _keep_decided_failure(halted, f"{type(exc).__name__}: {exc}")
                    raise
                except (SkillError, SafetyViolation) as retreat_error:
                    recovery = f"refused: {retreat_error}"
                self.memory.add("outcome", f"grasp descent failed: {exc}; pre-close retreat {recovery}")
                # Only a veto issued BEFORE its step was commanded (nothing
                # touched, the observed scene still holds), no jaw command, a
                # withdrawal that COMPLETED over its vetted route, and room in
                # the attempt and time budgets may hand over to the next
                # candidate. Anything else stops exactly as before.
                why_not, withdrawn = SkillRuntime._observed_next_candidate_refusal(
                    self, exc, recovery, episode, selection, grasps, _planning_deadline)
                if selection is not None:
                    if withdrawn and selection.first_refusal is None:
                        selection.first_refusal = str(exc)[:200]
                    selection.finish(f"refused before contact: {str(exc)[:200]}; withdrew to pregrasp"
                                     if withdrawn else
                                     f"failed: {str(exc)[:200]}; pre-close retreat {recovery[:80]}")
                if why_not is None:
                    next_after = exc
                else:
                    failure = type(exc)(f"{exc}; pre-close retreat {recovery}"
                                        + ("" if selection is None
                                           else selection.note(why_not if withdrawn else None)))
                    failure._observed_noted = True
                    raise failure from exc
            else:
                raise
        finally:
            self.arm.harness.clear_grasp_exemption()
            contact_episode.finish(self, episode, completed=contact_completed)
            # An operator STOP below the pregrasp (descent, close, lift, the
            # pre-close retreat) leaves the arm in or at the target: keep this
            # approach's column + pregrasp for the NEXT EXPLICIT move_home /
            # reset_scene to retrace (never an automatic move) at the lift's
            # own speed. keep() decides what is a stop; nothing is kept
            # without a raised refusal.
            try:
                stop_retrace.keep(
                    self, approach_column, q_pre, failure=stop_failure,
                    halt_generation=scene_halt_generation, duration_s=lift_dur,
                    ceiling_m=SkillRuntime.AIR_GRASP_EXIT_CEILING_M)
            except Exception:  # noqa: BLE001 -- bookkeeping never replaces the stop
                logger.exception("could not keep the approach column after a stop")

        if next_after is not None:
            return SkillRuntime._grasp_next_observed_candidate(
                self, next_after, label, material, spatial_hint, fix, frame, _fix, _frame,
                _planning_deadline, scene_gate, scene_halt_generation, selection, grasps)

        # 5. verify: ASPIRE heuristic on jaw travel after close. An object in
        # the jaws stalls them ABOVE the commanded close; if they reached the
        # commanded stage-2 fraction even though the object should be much
        # wider, nothing resisted: air grasp. Unknown feedback -> report the
        # grasp as unverified rather than dropping a possibly-held object.
        grasp_evidence.phase("verify")
        width_after_lift = self._gripper_width_frac()
        grasp_evidence.event("grip_verification", width_after_lift=width_after_lift)
        verified = width_after_lift is not None
        if verified:
            # B36: after a post-contact hold the jaws were last commanded to
            # the hold opening, not the stage-2 one; an object lost in the lift
            # leaves them THERE, so that is the opening "nothing resisted" is
            # judged against (else the hold would blind this check).
            commanded_open = SkillRuntime._close_commanded_open(self, profile, close_hold_open)
            expected_open = min(grasp.width_m / self._max_width, 1.0)
            air_grasp = self._air_grasp(width_after_lift, commanded_open, expected_open)
            if air_grasp:
                if carry_attachment.active(self) is not None:
                    carry_attachment.active(self).reject(
                        "jaw heuristic contradicts retained attachment; no retry authorized")
                self._held_provisional = None
                self.memory.add("outcome", f"grasp {label!r} FAILED: jaws closed on air")
                try:
                    self.grasp_memory.record(
                        SkillRuntime._grasp_memory_label(self, label), fix, grasp, success=False, reason="air grasp",
                        z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
                except Exception:
                    pass
                if selection is not None:
                    selection.finish("air grasp")
                # PR-A jaws round, ruling 1: reopen the empty jaws to the
                # opening this grasp checked before its approach, confirm them
                # there, THEN withdraw. With the observed-finger gate the exit's
                # every step checks the fingers inside that opening, so closed
                # jaws refused it and stayed closed for every later pick.
                # Spec §3.2 rule 1: the jaws are verified empty and the arm
                # sits at the pregrasp hover; withdraw up through the column
                # this approach entered, with the pre-close retreat's motion.
                # A halt since this grasp started (the close included) stops
                # it: the planned move carries the grasp's halt generation.
                descend_s = float(gcfg.get("descend_duration_s", 2.0))

                def exit_move(q):
                    if scene_gate is not None:
                        return self._move_joints(q, duration_s=descend_s,
                                                 _linear_tool_path=True, **_scene_motion(q))
                    return self._move_planned(q, duration_s=descend_s,
                                              _halt_generation=scene_halt_generation)

                error = "air grasp: gripper closed fully, object not held"
                try:
                    reopen = self._air_grasp_reopen(open_pos, scene_gate,
                                                    halt_generation=scene_halt_generation)
                    grasp_evidence.event("jaws_reopen", outcome=reopen)
                    if reopen.startswith("refused"):
                        recovery = "skipped: the jaws were not confirmed open"
                    else:
                        recovery = self._air_grasp_exit(approach_column, exit_move, hover_q=q_pre,
                                                        halt_generation=scene_halt_generation)
                except SafetyViolation as exc:
                    grasp_evidence.event("exit_withdrawal", outcome=f"stopped: {type(exc).__name__}: {exc}")
                    _keep_decided_failure(exc, error)   # a halt in the recovery: still this air grasp
                    raise
                grasp_evidence.event("exit_withdrawal", outcome=recovery)
                if reopen.startswith("refused"):
                    # never "did not open": a jaw command may have moved them
                    error += f"; the jaws were not confirmed open: {reopen.removeprefix('refused: ')}"
                elif recovery.startswith(("refused", "did not settle")):
                    error += f"; exit withdrawal {recovery}"
                return {
                    "ok": False,
                    "error": error,
                    "suggestion": "re-localize the object or try the alternate yaw",
                    "jaws_reopen": reopen,
                    "exit_withdrawal": recovery,
                    **({"observed_selection": selection.summary()} if selection is not None else {}),
                }
        # The held width as MEASURED (jaw stall after the lift); the planned
        # width stands in when feedback was unavailable. `_reconcile_held`
        # reads it before calling a closed jaw a slip.
        self._promote_held(label, frame, fix,
                           held_width_m=(float(width_after_lift) * float(self._max_width)
                                         if verified else float(grasp.width_m)),
                           held_offset=held_offset_at_close)
        try:
            self.grasp_memory.record(
                SkillRuntime._grasp_memory_label(self, label), fix, grasp, success=True,
                z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
        except Exception:
            pass
        self.memory.add(
            "action",
            f"grasped {label!r} (profile {profile.name}, "
            + (f"jaw at {width_after_lift:.2f})" if verified else "grip UNVERIFIED)"),
        )
        if selection is not None:
            selection.finish("grasped")
        return {
            "held": label,
            # Pixel-addressed callers bypass the unique description search.
            **(self._target_resolution(label, frame, fix)
               if _fix is None or _frame is None else {}),
            "grip_profile": profile.name,
            "grip_verified": verified,
            "gripper_open_frac": round(width_after_lift, 2) if verified else None,
            "grasp_width_m": round(grasp.width_m, 3),
            # Observed-finger gate: candidates ranked last / dropped by the
            # vet and every attempted candidate (only with the gate enabled).
            **({"observed_selection": selection.summary()} if selection is not None else {}),
        }

    def _observed_next_candidate_refusal(self, exc, recovery, episode, selection, grasps, planning_deadline):
        """-> (why no next candidate or None, withdrew-before-contact).

        withdrew-before-contact: the observed-finger gate vetoed a step before
        commanding it (`precommand_refusal`: a planned descent profile or the
        jaw close at the measured pose), no jaw command was attempted, and the
        pre-close retreat COMPLETED over its vetted route. Then the next
        candidate needs: attempts left (`OBSERVED_NEXT_CANDIDATES`), a
        candidate of this plan not yet attempted, and time before the
        persistence and task deadlines. Near-copies of an attempted candidate
        do not count (`_ObservedSelection.untried`). A measured-feedback
        refusal, a refused or unsettled withdrawal, a contact, halt or e-stop
        never qualify."""
        from ..grasping.observed_scene import precommand_conflict

        if selection is None:
            return "observed-finger gate disabled", False
        if precommand_conflict(exc) is None:
            return "not a pre-command observed-finger veto", False
        if recovery != "completed":
            return "withdrawal not completed", False
        if episode is not None or getattr(self, "_held_provisional", None) is not None:
            return "jaw command attempted", False
        if len(selection.attempts) > OBSERVED_NEXT_CANDIDATES:
            return f"limit of {OBSERVED_NEXT_CANDIDATES} further candidates reached", True
        deadlines = [d for d in (planning_deadline, getattr(self, "_task_deadline", None)) if d is not None]
        if deadlines and time.monotonic() >= min(deadlines):
            return "grasp time budget spent", True
        if not selection.untried(grasps):
            return "no other candidate in this plan", True
        return None, True

    def _grasp_next_observed_candidate(self, refusal, label, material, spatial_hint, fix, frame,
                                       _fix, _frame, planning_deadline, gate, halt_generation,
                                       selection, grasps) -> dict:
        """The previous candidate withdrew over its vetted route after a
        pre-command veto: select the next vetted candidate of the same plan
        (same observed scene, same halt baseline, same neighbour ranking) and
        execute it through the full `skill_grasp_object` path, home move
        included. Any failure of that attempt keeps its own error and type and
        names the first refusal and how many candidates were refused (a halt
        stays the bare MotionHalted)."""
        from ..grasping.planning_budget import PlanningBudgetExceeded
        from ..grasping.selector import NoExecutableGrasp

        remaining = selection.untried(grasps)
        attempt = len(selection.attempts) + 1
        self.memory.add("note", f"grasp candidate {attempt - 1} was refused before contact and withdrew; "
                                f"trying the next vetted candidate ({attempt}/{1 + OBSERVED_NEXT_CANDIDATES})")
        grasp_evidence.event("observed_next_candidate", attempt=attempt, refusal=str(refusal),
                             remaining=len(remaining),
                             near_copies_skipped=sum(1 for c in grasps if not selection.attempted(c))
                             - len(remaining))
        context = _ObservedNextCandidate(gate=gate, halt_generation=halt_generation,
                                         remaining=remaining, selection=selection)
        try:
            result = SkillRuntime.skill_grasp_object(
                self, label, material=material, spatial_hint=spatial_hint, _fix=fix, _frame=frame,
                _planning_deadline=planning_deadline, _next_candidate=context)
        except (NoExecutableGrasp, PlanningBudgetExceeded) as none_left:
            # Nothing moved after the withdrawal: report the original veto.
            failure = type(refusal)(f"{refusal}; pre-close retreat completed"
                                    + selection.note(f"next candidate: {none_left}"))
            failure._observed_noted = True
            raise failure from none_left
        except MotionHalted as halted:
            # The veto was decided before the halt: the halt carries it,
            # unless the next candidate's own failure came later.
            _keep_decided_failure(halted, f"{type(refusal).__name__}: {refusal}")
            raise
        except (SkillError, SafetyViolation) as failed:
            if not getattr(failed, "_observed_noted", False):
                failed.args = (f"{failed}{selection.note()}",)
                failed._observed_noted = True
            raise
        if _fix is None or _frame is None:
            # The first call localized by description; its resolution stands.
            result = {**result, **self._target_resolution(label, frame, fix)}
        return result

    def _close_commanded_open(self, profile, hold_open: float | None = None) -> float:
        """The open fraction a grasp's close last commanded, the opening its
        verify (`_air_grasp`) judges the jaws against: fully closed for a
        force-contact close, the post-contact hold's opening when one was
        commanded (`hold_open`), else the stage-2 seat."""
        return (0.0 if self._force_contact_close_enabled()
                else hold_open if hold_open is not None
                else max(1.0 - profile.close_frac_stage2, 0.0))

    def _air_grasp(self, width_after: float, commanded_open: float, expected_open: float) -> bool:
        """ASPIRE jaw-travel heuristic, shared by both executors: jaws nearly
        shut, or at the commanded close although the object should have
        stopped them much wider -- nothing resisted, nothing is held."""
        return width_after < float(self.cfg.grasp.get("air_grasp_frac", 0.04)) or (
            width_after <= commanded_open + 0.03
            and expected_open >= commanded_open + 0.07
        )

    #: TCP lifts the air-grasp exit tries, in order (spec §3.2 rule 1): the
    #: route planner's lift fallback heights (`plan_route`), one constant.
    AIR_GRASP_EXIT_LIFTS_M = ROUTE_LIFT_HEIGHTS_M
    #: The exit's column ends this far above the hover TCP (A3 review): the
    #: target's own top lies below the hover; anything above it is a
    #: neighbour, checked at full strength.
    AIR_GRASP_EXIT_CEILING_M = 0.005

    def _air_grasp_exit(self, column, move: Callable[[np.ndarray], bool], *,
                        hover_q=None, halt_generation: int | None = None) -> str:
        """Withdraw the empty tool upward after an air grasp (spec §3.2
        rules 1-2) and say how it went.

        The arm sits at the pregrasp hover `hover_q` (None: the measured
        pose), above the target. Inside `_column_exemption(harness, column)`,
        the column this grasp's approach entered (None: strict checks),
        capped AIR_GRASP_EXIT_CEILING_M above the hover TCP, try a TCP lift
        of +3 cm, then +6 cm, from the measured pose (`lift_ik` with the
        harness's model, as plan_route). `move(q)` executes a lift through
        the same vetted motion as the pre-close retreat; every gate other
        than the column keeps its strength. The column is released on every
        path, an exception included.

        Only an unreachable lift or a preflight refusal (SkillError) moves
        on to the next height. A stop ends the exit at once and propagates
        after the column is released: a contact, a halt or a SafetyViolation
        under a latched e-stop as the original exception; an e-stop or a
        halt seen after a refused lift (or that landed in the same move as
        any other SafetyViolation) as SafetyViolation("e-stop latched") /
        MotionHalted. Any other SafetyViolation (a route that became unsafe,
        the perception watchdog) and a move that does not settle end it with
        a string.

        Nothing moves with a possible payload (`_held_provisional`, an active
        carry attachment): no empty-tool recovery is invented for jaws that
        may hold the object. Nor with a latched e-stop, a halt received since
        `halt_generation` (the grasp's start; None: now), or a harness
        without a kinematic model.

        Returns "completed (+N cm)", "refused: <reasons>", "did not settle
        (+N cm)" or "skipped: <why>".
        """
        from ..safety.trajectory import lift_ik

        if getattr(self, "_held_provisional", None) is not None:
            return "skipped: provisional hold (the jaws may hold the object)"
        if carry_attachment.active(self) is not None:
            return "skipped: carry attachment active (the jaws may hold the object)"
        harness = self.arm.harness
        generation = (getattr(harness, "_halt_generation", None)
                      if halt_generation is None else halt_generation)

        def stopped():
            return _air_grasp_stop(harness, generation)

        def stop_exception(stop):
            return _air_grasp_stop_error(harness, stop)

        stop = stopped()
        if stop:
            return f"skipped: {stop}"
        # The harness's own model proposes the lift, as in plan_route: the
        # same model then vets every route to it.
        kin = getattr(harness, "kin", None)
        if not (callable(getattr(kin, "fk", None)) and callable(getattr(kin, "ik", None))):
            return "skipped: the safety harness has no kinematic model"
        hover = self.arm.get_state().q if hover_q is None else hover_q
        z_max = float(kin.fk(np.asarray(hover, dtype=float))[2, 3]) + self.AIR_GRASP_EXIT_CEILING_M
        grasp_evidence.phase("exit_withdrawal")
        refusals, stop_error = [], None
        with _column_exemption(harness, column, z_max=z_max):
            for height in self.AIR_GRASP_EXIT_LIFTS_M:
                lift = f"+{height * 100:.0f} cm"
                q = lift_ik(kin, self.arm.get_state().q, height)
                if q is None:
                    refusals.append(f"{lift}: no IK solution")
                else:
                    try:
                        settled = move(q)
                    except SkillError as exc:
                        refusals.append(f"{lift}: {exc}")
                    except SafetyViolation as exc:
                        if not (isinstance(exc, (ContactDetected, MotionHalted))
                                or getattr(harness, "estopped", False)):
                            # A halt that landed in the same move is a stop,
                            # not a refusal: it is raised, never a string.
                            stop = stopped()
                            if not stop:
                                return f"refused: {lift}: {exc}"
                            stop_error = stop_exception(stop)
                            stop_error.__cause__ = exc
                            break
                        stop_error = exc
                        break
                    else:
                        return f"completed ({lift})" if settled else f"did not settle ({lift})"
                # An e-stop or a halt that arrived while this lift was refused.
                stop = stopped()
                if stop:
                    stop_error = stop_exception(stop)
                    break
        if stop_error is not None:
            raise stop_error   # after the column is released, as the original
        return "refused: " + "; ".join(refusals)

    def _air_grasp_reopen(self, open_pos: float, gate=None, *, halt_generation: int | None = None) -> str:
        """Reopen jaws that closed on air, before the exit withdrawal, and say
        how it went (PR-A jaws round, ruling 1).

        `open_pos` is the opening this grasp commanded before its approach:
        the open fingers already occupied that volume at the hover, and every
        approach and descent check modelled them open. With the observed-finger
        gate (`gate`: the one this grasp was selected with) the reopening sweep
        at the measured pose is vetted before the command
        (`require_reopening`), and the reopen counts only once the measured
        fingers are back inside the gate's checked opening interval
        (`reopened`); every stream step of the exit then checks them there.
        Without it, the measured jaw opening must reach `open_pos` within the
        air-grasp tolerance.

        Nothing is commanded with a possible payload (`_held_provisional`, an
        active carry attachment), a latched e-stop or a halt received since
        `halt_generation` (the grasp's start; None: now). A stop seen before
        the command skips the reopen; one that refuses the command or lands
        while the jaws open propagates (MotionHalted / SafetyViolation
        "e-stop latched") and nothing else is commanded.

        Returns "reopened", "refused: <why>" (the jaws were not confirmed open:
        refused before the command, or commanded but not confirmed; a jaw
        fault that is not a SafetyViolation is "refused: jaw command failed:
        <error>", or "refused: jaw state could not be read: <error>" when the
        gate's read or check of the jaws before the command failed and
        nothing was sent) or
        "skipped: <why>".
        """
        if getattr(self, "_held_provisional", None) is not None:
            return "skipped: provisional hold (the jaws may hold the object)"
        if carry_attachment.active(self) is not None:
            return "skipped: carry attachment active (the jaws may hold the object)"
        harness = self.arm.harness
        generation = (getattr(harness, "_halt_generation", None)
                      if halt_generation is None else halt_generation)
        stop = _air_grasp_stop(harness, generation)
        if stop:
            return f"skipped: {stop}"

        def check():
            stop = _air_grasp_stop(harness, generation)
            if stop:
                raise _air_grasp_stop_error(harness, stop)

        if gate is not None:
            opened = gate.reopened
        else:
            air = float(self.cfg.grasp.get("air_grasp_frac", 0.04))
            wanted = self._open_frac_of(open_pos)

            def opened(state):
                width = self._jaw_open_frac(state)
                return width is not None and width >= wanted - air

        grasp_evidence.phase("jaws_reopen")
        measured = gate is None   # with the gate, the jaws are read and checked before the command
        try:
            if gate is not None:
                gate.require_reopening(self.arm.get_state(), check=check)
                measured = True
            why = self._open_empty_jaws(open_pos, opened, halt_generation=generation, check=check)
        except SafetyViolation as exc:
            if isinstance(exc, MotionHalted):
                raise   # the operator's halt, as it is (also one SafeArm refused the jaws for)
            stop = _air_grasp_stop(harness, generation)
            if stop:   # an e-stop or a halt that landed with the refusal: a stop, never a string
                raise _air_grasp_stop_error(harness, stop) from exc
            return f"refused: {exc}"
        except Exception as exc:  # noqa: BLE001 -- a jaw fault: the reopen was not confirmed
            # Round 2, M1: a bus timeout or a failed feedback read keeps the
            # air grasp's result (as reset_scene reports its jaw fault); the
            # exit is skipped and nothing else is commanded.
            stop = _air_grasp_stop(harness, generation)
            if stop:
                raise _air_grasp_stop_error(harness, stop) from exc
            if not measured:   # the read or the gate's check before the command failed; nothing was sent
                return f"refused: jaw state could not be read: {exc}"
            return f"refused: jaw command failed: {exc}"
        return "reopened" if why is None else f"refused: {why}"

    def _jaws_open_timeout_s(self) -> float:
        """The bound on waiting for empty jaws to open: the profile's
        `grasp.open_wait_timeout_s`, else JAWS_OPEN_TIMEOUT_S."""
        timeout = open_wait_timeout_s(self.cfg.grasp)
        return JAWS_OPEN_TIMEOUT_S if timeout is None else timeout

    def _open_empty_jaws(self, open_pos: float, opened: Callable, *, halt_generation, check) -> str | None:
        """Command `open_pos` to jaws that hold nothing and wait until
        `opened(state)` confirms it from feedback. The caller has decided that
        nothing is held and that the jaws may open here.

        The command carries `halt_generation` (SafeArm refuses it after a
        halt or an e-stop). `check()` runs after every feedback read: a stop
        it raises ends the wait, and nothing else is commanded here. Returns
        None once confirmed, else why not."""
        timeout = self._jaws_open_timeout_s()
        self._set_gripper(open_pos, effort=0.8, _halt_generation=halt_generation)
        deadline = time.monotonic() + timeout
        for _ in range(int(math.ceil(timeout / JAWS_OPEN_POLL_S)) + 1):
            state = self.arm.get_state()
            check()
            if opened(state):
                return None
            if time.monotonic() >= deadline:
                break
            time.sleep(JAWS_OPEN_POLL_S)
        return f"no feedback confirmed the opening within {timeout:g} s"

    def _open_closed_jaws_before_selection(self, frame, fix, halt_generation, state) -> bool:
        """A grasp that starts empty-handed with closed jaws opens them before
        its planning and selection (PR-A jaws round, ruling 2; observed-finger
        gate), and before any per-attempt budget clock starts (round 2b).

        The gate models the MEASURED fingers: built from closed jaws it sweeps
        them closed..open on every candidate's descent and drops them all
        ("descent finger intersects observed surface: target"). So: a gate on
        this grasp's own observed scene and the measured closed jaws (`state`,
        the grasp's start read) vets the opening at the measured pose
        (`require_reopening`) before the jaw command, and the grasp goes on
        once both fingers read open (`fingers_open`); selection reads the
        jaws again and builds its gate from the open jaws. Returns True when
        it opened them.

        Jaws that read open (the `_reconcile_held` rule) or unknown are left as
        they are. Never with a possible payload (`_held_provisional`, an active
        carry attachment). A refused opening is a SkillError and nothing moved;
        jaws that do not open in time are a SkillError; a halt or an e-stop
        propagates and nothing else is commanded.
        """
        if getattr(self, "_held_provisional", None) is not None or carry_attachment.active(self) is not None:
            return False
        width = self._jaw_open_frac(state)
        if width is None or width >= 1.0 - float(self.cfg.grasp.get("air_grasp_frac", 0.04)):
            return False
        harness = self.arm.harness

        def check():
            if harness.estopped:
                raise SafetyViolation("e-stop latched before the closed jaws were opened")
            harness._check_halt_generation(halt_generation)

        from ..grasping.observed_scene import for_runtime

        check()
        probe = for_runtime(self, frame, fix, state)
        grasp_evidence.phase("jaws_open")
        try:
            probe.require_reopening(state, check=check)
        except SafetyViolation as exc:
            if isinstance(exc, MotionHalted) or harness.estopped:
                raise
            raise SkillError(f"the jaws are closed and opening them before this grasp was refused, so "
                             f"nothing moved: {exc}. Call move_home, then try the grasp again.") from exc
        why = self._open_empty_jaws(self._grip_open, probe.fingers_open,
                                    halt_generation=halt_generation, check=check)
        if why is not None:
            raise SkillError(f"the jaws were closed and were not confirmed open before this grasp: {why}")
        return True

    def _open_frac_of(self, pos: float) -> float:
        """The open fraction (0 closed, 1 open) of gripper position `pos`: a
        linear map of the motor travel (`_gripper_width_frac`, `_jaw_open_frac`)."""
        span = self._grip_closed - self._grip_open
        if abs(span) < 1e-9:
            return 0.0
        return float(np.clip(1.0 - (float(pos) - self._grip_open) / span, 0.0, 1.0))

    def _jaw_open_frac(self, state) -> float | None:
        """The open fraction of `state`'s gripper position, None when unknown."""
        if not getattr(state, "gripper_valid", True):
            return None
        return self._open_frac_of(state.gripper_pos)

    def _promote_held(self, label, frame, fix, *, held_width_m, held_offset) -> None:
        """Record a grasp the jaw check accepted: the held flag, its identity,
        the measured width, the aiming offset and the observation binding.
        Shared by the analytic and the VLA executor so the two cannot drift."""
        self.held_object = label
        self._held_arm = self._running_arm_name()
        self._held_det_label = fix.detection.label
        self._held_color = detection_color(frame.rgb, fix.detection)
        self._held_provisional = None  # promoted: the real flag is set now
        self._held_width_m = held_width_m
        # A cached aiming estimate cannot prove a later slip. Preserve the
        # post-lift clock floor separately for newly acquired hold evidence.
        self._held_offset = held_offset
        self._held_observation_floor = None
        self._held_observation_floor_q = None
        from .held_observation import binding
        self._held_observation_binding = binding(self)
        self._held_observation_joint_signs = np.asarray(
            self.cfg.arm.get("joint_signs", [1] * self.arm.n_joints), float).copy()
        try:
            import copy
            state_after_lift = self.arm.get_state()
            carry_attachment.observe(self, state_after_lift)
            self._held_observation_floor = copy.deepcopy(state_after_lift.physics_clock)
            self._held_observation_floor_q = np.asarray(state_after_lift.q, float).copy()
        except carry_attachment.AttachmentInvalid:
            raise
        except Exception as exc:
            if carry_attachment.active(self) is not None:
                carry_attachment.active(self).reject("post-lift feedback unavailable: " + str(exc))
        # The fix is in the arm frame; the world model is the table frame.
        self.beliefs.mark_removed(self._held_det_label or label, near=self._to_world(fix.position))

    def _vla_grasp(self, label, frame, fix, profile, halt_generation) -> dict:
        """`grasp.executor: vla` (B49, opt-in; grasping/vla_executor.py).

        The target is already localized by `skill_grasp_object` (same
        `_localize`, same measurements). Refusals happen before any motion:
        a rig gate the route does not implement, a policy server that does
        not answer, a missing `vla` extra. Then the jaws open and the arm
        re-homes over the vetted route exactly like the analytic pipeline,
        and the policy's chunks run under the harness (see the executor).
        The verdict is the SAME as the analytic one: the jaw-travel check
        here, then `execute()`'s independent postcondition verifier. Nothing
        the server says about its own success is read. Outcomes do not train
        the analytic grasp memory (there is no candidate geometry to credit).
        """
        from ..grasping.vla_executor import CLOSE_BELOW, VLAExecutor, route_refusal

        reason = route_refusal(self)
        if reason is not None:
            raise SkillError(f"grasp.executor: vla refused: {reason}; no motion sent")
        if getattr(self, "vla_executor", None) is None:
            self.vla_executor = VLAExecutor.from_cfg(self.cfg.grasp)
        executor = self.vla_executor
        client = executor.connect()  # VLAUnavailable: nothing has moved
        try:
            self._set_gripper(self._grip_open, effort=0.8, _halt_generation=halt_generation)
            home = self.cfg.arm.get("home_q")
            if home is not None and not self._move_planned(
                    np.asarray(home, dtype=float), duration_s=1.5, _halt_generation=halt_generation):
                raise SkillError("did not settle at home before the VLA episode")
            # The marker carries the arm whose close may leave it (R5): another
            # arm's call neither judges it nor takes its slot.
            provisional = ProvisionalHold((label, fix.detection.label, detection_color(frame.rgb, fix.detection)),
                                          arm=self.arm)
            try:
                report = executor.run(self, client, label=label, effort=profile.effort,
                                      halt_generation=halt_generation, provisional=provisional)
            except (SkillError, SafetyViolation) as exc:
                self.memory.add("outcome", f"VLA grasp episode for {label!r} ended early: {exc}")
                raise
        finally:
            client.close()
        summary = {k: report[k] for k in ("chunks", "waypoints", "closed", "stop", "server", "infer_ms")}
        if not report["closed"]:
            self.memory.add("outcome", f"VLA grasp {label!r} FAILED: the policy never closed the gripper")
            return {"ok": False, "executor": "vla", "vla": summary,
                    "error": (f"VLA episode ended ({report['stop']}, {report['chunks']} chunks) and the "
                              "policy never closed the gripper: nothing was grasped")}
        if report["last_open_commanded"] > CLOSE_BELOW:
            # closed, then commanded open again: whatever was there was let go
            self._held_provisional = None
            self.memory.add("outcome", f"VLA grasp {label!r} FAILED: the policy re-opened the gripper")
            return {"ok": False, "executor": "vla", "vla": summary,
                    "error": (f"the policy re-opened the gripper (last command {report['last_open_commanded']:.2f} "
                              "open) before the episode ended: nothing is held")}
        width_after = self._gripper_width_frac()
        verified = width_after is not None
        footprint = self._horizontal_footprint_m(fix.points)
        if verified:
            expected_open = min(footprint / self._max_width, 1.0) if footprint else 0.0
            if self._air_grasp(width_after, report["last_open_commanded"], expected_open):
                self._held_provisional = None
                self.memory.add("outcome", f"VLA grasp {label!r} FAILED: jaws closed on air")
                return {"ok": False, "executor": "vla", "vla": summary,
                        "error": "air grasp: gripper closed fully, object not held",
                        "suggestion": "re-localize the object; the policy closed on nothing"}
        held_offset = None
        if report["q_at_close"] is not None:
            offset = np.asarray(fix.position, float) - self.kin.fk(report["q_at_close"])[:3, 3]
            if offset.shape == (3,) and np.isfinite(offset).all():
                held_offset = offset
        self._promote_held(label, frame, fix,
                           held_width_m=(float(width_after) * float(self._max_width) if verified
                                         else footprint),
                           held_offset=held_offset)
        self.memory.add(
            "action",
            f"grasped {label!r} via the VLA policy ({report['chunks']} chunks, "
            + (f"jaw at {width_after:.2f})" if verified else "grip UNVERIFIED)"),
        )
        return {
            "held": label,
            **self._target_resolution(label, frame, fix),
            "grip_profile": profile.name,
            "grip_verified": verified,
            "gripper_open_frac": round(width_after, 2) if verified else None,
            "executor": "vla",
            "vla": summary,
        }
    def _force_contact_close_enabled(self) -> bool:
        return (self.cfg.arm.get("type") == "ros2"
                and bool(self.cfg.arm.get("ros_gripper_grasp_action"))
                and bool((self.cfg.arm.get("gripper") or {}).get("force_contact_close", False)))

    def _grasp_support_height(self, points) -> float:
        support = self.cfg.grasp.get("source_support_top_z_m")
        if support is not None:
            support = float(support)
            if not np.isfinite(support):
                raise SkillError("source_support_top_z_m must be finite")
            return support
        return max(float(self.cfg.safety.get("table_z", 0.0)), float(points[:, 2].min()))

    def _grasp_hold_squeeze(self) -> float | None:
        """B36: the squeeze of the grasp's bounded post-contact hold on this
        backend (`gripper.hold_squeeze_frac`, see `_hold_squeeze_frac`), or
        None = the plain two-stage close. Read once per grasp before anything
        moves, with `gripper.hold_stall_timeout_s`, so a bad profile fails
        closed. A driver with its own two-stage close (the real reBot, whose
        squeeze B38 bounds) never holds here."""
        squeeze = SkillRuntime._hold_squeeze_frac(self)
        if squeeze is None:
            return None
        if hasattr(getattr(self.arm, "raw", None), "close_gripper_two_stage"):
            return None
        _hold_stall_timeout_s(self.cfg.arm.get("gripper"))
        return squeeze

    def _grasp_close(self, profile, squeeze: float | None, **close_kw) -> float | None:
        """The grasp's close: the two-stage close, unchanged, then -- with a
        `squeeze` from `_grasp_hold_squeeze` -- the post-contact hold. Returns
        the hold's open fraction when one was commanded, else None. Only the
        grasp holds (the path it was measured on): `close_gripper` and the
        screw stroke call `_close_two_stage` and stay exactly as before.
        `close_kw` (`_halt_generation`, `_before_close`) reaches both."""
        self._close_two_stage(profile, **close_kw)
        if squeeze is None:
            return None
        return self._hold_after_contact(profile, squeeze, 1.0 - profile.close_frac_stage2, **close_kw)

    def _close_two_stage(self, profile, *, _halt_generation=None, _before_close=None,
                         _never_past=None, _count_skips=False) -> int | None:
        """The two-stage close (scout stage, then seat). A close never opens
        the jaws: a stage more open than `_never_past` (the jaw position the
        caller read before the close; None = not read) or than the stage
        before it is skipped (`closing_stages`), and so is the backend's own
        two-stage close when every stage would be (that driver skips by its
        own reading too, and its own count is the one returned). Returns how
        many stages were skipped when asked (`_count_skips`: close_gripper),
        else None as before. The grasp's and the screw stroke's closes start
        from jaws they just opened and pass no reading: every stage goes,
        exactly as before."""
        skipped = self._close_stages(profile, _halt_generation=_halt_generation,
                                     _before_close=_before_close, _never_past=_never_past)
        return skipped if _count_skips else None

    def _close_stages(self, profile, *, _halt_generation, _before_close, _never_past) -> int:
        # Some hardware drivers expose a raw two-stage escape hatch. It may
        # not bypass a retained model-backed empty-tool withdrawal either.
        check = getattr(self.arm.harness, "check_model_withdrawal", None)
        if check is not None:
            check(command=True, gripper=True)
        if self._force_contact_close_enabled():
            if opens_past(self._grip_closed, _never_past, self._grip_open, self._grip_closed):
                return 1
            if _before_close is not None:
                _before_close()
            self._set_gripper(self._grip_closed, effort=profile.effort,
                              **({"_halt_generation": _halt_generation} if _halt_generation is not None else {}))
            self._gripper_closes_commanded += 1
            return 0
        raw = self.arm.raw
        bulk = hasattr(raw, "close_gripper_two_stage")
        if bulk and _before_close is not None:
            raise SafetyViolation("observed-finger closing requires individually guarded stages")
        span = self._grip_closed - self._grip_open
        stages = tuple((self._grip_open + span * frac, eff, frac) for frac, eff in (
            (profile.close_frac_stage1, profile.effort * 0.7),
            (profile.close_frac_stage2, profile.effort),
        ))
        kept = closing_stages(stages, _never_past, self._grip_open, self._grip_closed)
        if bulk:
            if not kept:
                return len(stages)
            # The backend's own close bypasses SafeArm.set_gripper: a halt
            # since the running skill started keeps the jaws open (A3 review).
            check_halt = getattr(self.arm.harness, "_check_halt_generation", None)
            if callable(check_halt):
                check_halt(_halt_generation)
            self._mark_motion_started()
            raw.close_gripper_two_stage(
                width_frac_stage1=profile.close_frac_stage1,
                width_frac_stage2=profile.close_frac_stage2,
                effort=profile.effort,
            )
            # What the driver itself skipped, by its own start reading (it may
            # differ from ours: a read that failed on one side only, jaws that
            # moved); a backend that keeps no count is judged by ours. A close
            # is credited only when the driver sent a stage.
            sent_skips = getattr(raw, "close_stages_skipped", None)
            skipped = sent_skips if type(sent_skips) is int else len(stages) - len(kept)
            if skipped < len(stages):
                self._gripper_closes_commanded += 1
            return skipped
        timeout = self.cfg.grasp.get("close_feedback_timeout_s")
        band = CLOSE_STALL_BAND
        if timeout is not None:
            timeout = float(timeout)
            if not np.isfinite(timeout) or timeout <= 0:
                raise SkillError("close_feedback_timeout_s must be finite and positive")
            band = close_stall_band(self.cfg.arm.get("gripper"))
        for target, eff, frac in kept:
            if _before_close is not None:
                _before_close()
            grasp_evidence.event("close_stage", closed_frac=frac, effort=eff,
                                 target_pos=target)
            self._set_gripper(target, effort=eff,
                                 **({"_halt_generation": _halt_generation} if _halt_generation is not None else {}))
            # Counted once the command is accepted: the verifier may then say
            # "the jaw closed" (a refused stage leaves the count unchanged).
            self._gripper_closes_commanded += 1
            time.sleep(float(self.cfg.grasp.get("close_settle_s", 0.0)))
            if timeout is not None:
                self._wait_gripper_closed(1.0 - frac, timeout, band)
        return len(stages) - len(kept)

    def _hold_squeeze_frac(self) -> float | None:
        """`gripper.hold_squeeze_frac` of the arm profile, validated (None = off).

        A number applies on any backend. A mapping is keyed by the physics
        engine the arm backend reports (`physics_engine`; the Isaac bridge
        names `physx` or `newton` in its ping reply). Measured on Isaac (bare
        reBot scene, fresh stage per run, learned GraspGen-X grasps; evidence
        docs/evidence/b36-pick-reliability-20261010/): with the hold PhysX
        confirmed 15/16 picks against 9/16 without it; on Newton the hold
        confirmed 1/4 against 14/14 without it (the cube dropped early in the
        carry) -- so the Isaac profile opts in for PhysX only.
        A backend that reports no engine, or an engine the mapping does not
        name, gets no hold. Every value is validated before the jaws move,
        including the entries for other engines (a bad profile fails closed).
        """
        arm_cfg = getattr(self.cfg, "arm", None)
        g = arm_cfg.get("gripper") if arm_cfg is not None else None
        raw = g.get("hold_squeeze_frac") if g is not None else None
        if raw is None:
            return None
        if isinstance(raw, Cfg):
            per_engine = raw.as_dict()
            for name, entry in per_engine.items():
                if entry is not None:
                    _squeeze_fraction(entry, f"gripper.hold_squeeze_frac.{name}")
            engine = getattr(getattr(self.arm, "raw", None), "physics_engine", None)
            if not isinstance(engine, str) or not engine:
                return None
            entry = per_engine.get(engine.lower())
            return None if entry is None else _squeeze_fraction(entry, f"gripper.hold_squeeze_frac.{engine}")
        return _squeeze_fraction(raw, "gripper.hold_squeeze_frac")

    def _hold_after_contact(self, profile, squeeze: float, stage2_open: float, *,
                            _halt_generation=None, _before_close=None) -> float | None:
        """B36: once the jaws stall on the object, hold at the measured contact
        width minus `squeeze` (a fraction of full travel) instead of pushing on
        toward the stage-2 target. Returns the commanded hold open fraction,
        or None when no hold was commanded (the stage-2 command stands).

        The real reBot driver bounds its squeeze after contact (B38,
        `gripper.max_contact_squeeze_rad`). The Isaac bridge has no force
        bound: it drops `effort`, and its finger drives push toward the
        position target at full stiffness. Measured on Isaac PhysX (bare
        reBot scene, first lift of each run), that squeeze left wrist roll a
        median 0.045 rad off its lift target (9 of 35 lifts at or over
        settle_tol 0.045, each one "did not settle at grasp lift pose"); with
        the hold the median was 0.011 rad (max 0.015, 20 lifts). Newton showed
        no such deflection (0.000 rad, 18 lifts). Without a stall the stage-2 command
        stays as it was; the hold only ever presses less. The caller's
        air-grasp check must then compare against the hold target, not the
        stage-2 one.
        """
        g = self.cfg.arm.get("gripper")
        timeout_s = _hold_stall_timeout_s(g)
        width = self._jaw_stall_width(stage2_open, timeout_s)
        if width is None:
            grasp_evidence.event("close_hold", applied=False, reason="no jaw stall observed")
            return None
        hold_open = max(0.0, width - squeeze)
        if width <= stage2_open + 0.02 or hold_open <= stage2_open:
            grasp_evidence.event("close_hold", applied=False, reason="stage-2 target is already as light",
                                 contact_open_frac=width)
            return None
        target = self._grip_open + (self._grip_closed - self._grip_open) * (1.0 - hold_open)
        if _before_close is not None:
            _before_close()
        grasp_evidence.event("close_hold", applied=True, contact_open_frac=width,
                             hold_open_frac=hold_open, target_pos=target)
        self.arm.set_gripper(target, effort=profile.effort,
                             **({"_halt_generation": _halt_generation} if _halt_generation is not None else {}))
        time.sleep(float(self.cfg.grasp.get("close_settle_s", 0.0)))
        return hold_open

    def _jaw_stall_width(self, target_open: float, timeout_s: float) -> float | None:
        """Open fraction where the jaws stopped (reached `target_open` or
        stalled for 0.5 s), or None if they never stopped within `timeout_s`."""
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        anchor = None
        stable_since = time.monotonic()
        while True:
            width = self._gripper_width_frac()
            now = time.monotonic()
            if width is not None and np.isfinite(width):
                if width <= target_open + .01:
                    return float(width)
                if anchor is None or abs(width - anchor) > .002:
                    anchor, stable_since = width, now
                elif width < .95 and now - stable_since >= .5:
                    return float(width)
            else:
                anchor, stable_since = None, now
            if now >= deadline:
                return None
            time.sleep(.05)

    def _wait_gripper_closed(self, target_open: float, timeout_s: float, band: float) -> None:
        """Wait for commanded travel or a measured stall before lifting.

        A fixed dwell expires while rate-limited sim jaws are still closing
        when physics runs slower than wall time. A stall is only a completion
        signal here; the existing post-lift check still adjudicates the grip,
        but it only catches empty jaws (an air grasp). An object slipping
        slowly out of the jaws, still between them at the post-lift read,
        counts as held. The wider Isaac band (0.01) reads such a slip as
        stopped and lifts where 0.002 waited: the measured trade, 24 of 24
        live Newton trials held the cube.
        `band` is the profile's `gripper.close_stall_band` (close_stall_band).
        """
        deadline = time.monotonic() + timeout_s
        anchor = None
        stable_since = time.monotonic()
        while True:
            width = self._gripper_width_frac()
            now = time.monotonic()
            if width is not None and np.isfinite(width):
                if width <= target_open + .01:
                    return
                if anchor is None or abs(width - anchor) > band:
                    anchor, stable_since = width, now
                elif width < .95 and now - stable_since >= .5:
                    return
            else:
                anchor, stable_since = None, now
            if now >= deadline:
                raise SkillError("gripper closure did not settle; refusing to lift")
            time.sleep(.05)

    def _adopt_unknown_held(self) -> None:
        """'save it in the box' must work even when the held object was
        never registered (a crashed task, or a human placed something in
        the jaws): a mid-open gripper stall means SOMETHING is in there --
        adopt it as 'object' so place skills can act on it. (Wrist-camera
        visual confirmation is the ROADMAP upgrade.)"""
        if self.held_object:
            return
        wf = self._gripper_width_frac()
        air = float(self.cfg.grasp.get("air_grasp_frac", 0.04))
        if wf is not None and air < wf < 0.9:
            self.held_object = "object"
            self._held_arm = self._running_arm_name()
            self._held_det_label = None
            self._held_width_m = float(wf) * float(self._max_width)  # measured now
            self.memory.add(
                "note",
                "the jaws are holding something unregistered; "
                "treating it as 'object'",
            )

    def _held_object_observation(self):
        from .held_observation import measure
        return measure(self, localize_object)

    def _held_object_offset(self) -> np.ndarray | None:
        """Compatibility accessor for aiming; it grants no release authority."""
        return self._held_object_observation().offset

    def _supported_release_height(self, support_z: float, fallback: float) -> float:
        clearance = self.cfg.grasp.get("place_support_clearance_m")
        offset = getattr(self, "_held_support_offset_m", None)
        if clearance is None or offset is None:
            return fallback
        clearance = float(clearance)
        if not np.isfinite(clearance) or clearance < 0:
            raise SkillError("placement support clearance must be finite and nonnegative")
        return float(support_z) + float(offset) + clearance

    def _joint_segment_rejection(self, q_start, q_end, **exemption) -> str | None:
        """Vet the joint-interpolated segment without commanding or latching STOP.

        Clear endpoints do not imply a clear path: the wrist can sweep the
        TCP down toward the table while rotating between them. Sample at most
        0.01 rad per joint; transport approval still gates the whole trajectory
        and actual feedback at execution time.
        """
        start, end = np.asarray(q_start, float), np.asarray(q_end, float)
        steps = max(2, int(np.ceil(np.max(np.abs(end - start)) / 0.01)))
        for fraction in np.linspace(0., 1., steps + 1):
            cancel = getattr(self, "motion_cancel_requested", None)
            if callable(cancel) and cancel():
                raise SkillError("motion planning cancelled")
            reason = self.arm.harness.vet_pose(start + fraction * (end - start), **exemption)
            if reason:
                return reason
        return None

    def _place_geometry_rejection(self, q_now, pre, low, retreat, target):
        reason = self._joint_segment_rejection(q_now, pre.q)
        exemption = dict(exempt_xy=target[:2], exempt_radius_m=0.07,
                         exempt_z_min=float(target[2]) - 0.02)
        if reason is None:
            reason = self._joint_segment_rejection(pre.q, low.q, **exemption)
        if reason is None and retreat is not None:
            reason = self._joint_segment_rejection(low.q, retreat.q, **exemption)
        return reason

    def skill_place_at(self, x: float, y: float, z: float | None = None, *, _model_plan=None) -> dict:
        # Before any work: a release on the arm that does not hold the object
        # (spec D9) and a model release on a base_pose arm (R9) are refused.
        self._check_holding_arm()
        self._model_release_refusal(_model_plan)
        # (x, y[, z]) are table-frame agent coordinates (Q1 a): into the
        # running arm's frame. An omitted z stays the arm's own release height.
        if self.T_world_base is not None:
            if z is None:
                x, y = (float(v) for v in self._to_arm(np.array([x, y, 0.0], dtype=float))[:2])
            else:
                x, y, z = (float(v) for v in self._to_arm(np.array([x, y, z], dtype=float)))
        from .mujoco_placement_aim import RegionPlan
        if _model_plan is not None:
            if type(_model_plan) is not RegionPlan:
                raise SkillError('placement requires its private bound region plan')
            _model_plan.aim.guard(self)
        carry_attachment.check(self)
        if self._native_wrist_camera():
            self._reconcile_held()
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        release_z = (float(z) if z is not None else self._supported_release_height(
            table_z, table_z + float(gcfg.get("release_height_m", 0.05))))
        # Strict top-down poses only solve below ~0.15 m on this wrist: a
        # tall destination (the bin walls) must become "release from the
        # ceiling and let it drop", not an unreachable-pose failure.
        z_cap = float(gcfg.get("topdown_carry_z_max", gcfg.get("topdown_z_max", 0.15))) - 0.005
        if release_z > z_cap:
            self.memory.add(
                "note",
                f"place height {self._height_to_world(release_z):.2f} m is above the wrist's "
                f"top-down ceiling; releasing from {self._height_to_world(z_cap):.2f} m instead",
            )
            release_z = z_cap
        if _model_plan is not None:
            # Reject a replayed/changed internal handoff before an observation
            # could take another path (including the legacy slip handler).
            _model_plan.consume(self, x, y, release_z, _model_plan.aim.q)
        target = np.array([x, y, release_z])
        # Aim the OBJECT at the target, not the TCP. The IK below drives the
        # TCP, so the requested point has to be shifted by wherever the object
        # currently sits in the jaws, or the object lands offset by exactly
        # that much. Only the horizontal part is compensated: z is governed by
        # the release height and the wrist ceiling above.
        observation_harness = getattr(self.arm, "harness", None)
        observation_generation = getattr(observation_harness, "_halt_generation", None)
        from .held_observation import HeldObservationInvalid, binding
        observation_binding = binding(self)
        # Reuse the planning state below. On an armed NV hold, validate contact
        # before any visual slip handler could issue an opening command.
        carry_state = None
        if carry_attachment.active(self) is not None:
            try:
                carry_state = self.arm.get_state()
                carry_attachment.observe(self, carry_state)
            except Exception as exc:
                carry_attachment.active(self).reject("pre-carry feedback unavailable: " + str(exc))
        held_observation = self._held_object_observation()
        try:
            if any(a is not b for a, b in zip(observation_binding, binding(self))):
                raise HeldObservationInvalid("held observation arm or configuration changed")
            if observation_generation is not None:
                observation_harness._check_halt_generation(observation_generation)
        except (SafetyViolation, HeldObservationInvalid) as exc:
            if isinstance(exc, MotionHalted) and getattr(observation_harness, "halted", None):
                # The operator's STOP is pending: the skill ends on the halt
                # itself, the held observation and state untouched (A3 review).
                # A halt already withdrawn still invalidates the observation.
                raise
            held_observation.invalidated = True
            held_observation.release_authority = False
            held_observation.evidence.setdefault("rejected", []).append(str(exc))
        held_offset = held_observation.offset
        self.memory.add("note", "held offset channel=" + held_observation.channel
                        + "; release_authority=" + str(held_observation.release_authority),
                        {"held_observation": {
                            "channel": held_observation.channel,
                            "offset_m": None if held_offset is None else held_offset.tolist(),
                            "release_authority": held_observation.release_authority,
                            "invalidated": held_observation.invalidated,
                            "evidence": held_observation.evidence}})
        if held_observation.invalidated:
            raise HeldObservationInvalid("held observation invalidated; preserve the held state: "
                                         + str(held_observation.evidence.get("rejected", [])))
        # The same look also answers "is it still in the jaws?". When an
        # independent channel (sim physics, or the camera seeing the object
        # on the table) puts the object well below the TCP, it slipped
        # during the carry and there is nothing to place: lowering an empty
        # gripper and reporting ok makes pick_and_place's postcondition the
        # only thing that catches it -- measured on a visitor run: red cube
        # dropped 4 cm from where it started, `place_at` returned ok,
        # physics REFUTED it 20 s later. Detecting it here costs nothing and
        # turns a silent bad place into an honest retry.
        if held_offset is not None:
            drop_m = float(self.cfg.grasp.get("slip_drop_m", 0.06))
            if held_observation.release_authority and float(held_offset[2]) < -drop_m:
                if carry_attachment.active(self) is not None:
                    carry_attachment.active(self).reject(
                        "held pose contradicts the retained attachment; no opening authorized")
                self.memory.add(
                    "outcome",
                    f"{self.held_object!r} is {-float(held_offset[2])*100:.0f} cm below the "
                    "gripper -- it slipped during the carry",
                )
                if observation_generation is None:
                    raise SkillError("observed carry slip lacks a gripper authorization context")
                try:
                    self._set_gripper(self._grip_open, effort=0.6,
                                         _halt_generation=observation_generation)
                except SafetyViolation as exc:
                    raise HeldObservationInvalid("observed-slip opening was cancelled") from exc
                self._gripper_opens_completed += 1
                slipped = self.held_object
                self.held_object = None
                self._held_arm = None
                self._held_det_label = None
                self._held_color = None
                self._held_offset = None
                self._held_width_m = None
                self._held_support_offset_m = None
                raise SkillError(f"{slipped!r} slipped out of the gripper during the carry")
            target[0] -= float(held_offset[0])
            target[1] -= float(held_offset[1])
            self.memory.add(
                "note",
                f"current object offset estimate is {np.linalg.norm(held_offset[:2])*100:.1f} cm "
                "from the gripper centre; final TCP aim is resolved with the placement pose",
            )
        q_now = (carry_state if carry_state is not None else self.arm.get_state()).q
        from .place_geometry import plan as plan_geometry
        from .mujoco_placement_aim import capture as capture_aim
        aim = _model_plan.aim if _model_plan is not None else capture_aim(self, q_now)
        if _model_plan is not None:
            geometry = _model_plan.consume(self, x, y, release_z, q_now)
        else:
            geometry = plan_geometry(self, q_now, target, x=x, y=y,
                                     release_z=release_z, z_cap=z_cap,
                                     **({'validate': self._place_geometry_rejection}
                                        if self._force_contact_close_enabled() else {}),
                                     **({} if aim is None else {'attachment_translation_tool': aim.translation}))
        if aim is not None:
            aim.guard(self, q_now)
            target = geometry.target.copy()
        aim_w = self._to_world(target)                     # agent text: table frame
        asked_w = self._to_world(np.array([x, y, release_z], dtype=float))
        self.memory.add('note', f"planned TCP aim [{aim_w[0]:.3f}, {aim_w[1]:.3f}, {aim_w[2]:.3f}] "
                        f"for requested OBJECT XY [{asked_w[0]:.3f}, {asked_w[1]:.3f}]")
        lift, pre, low = geometry.lift, geometry.pre, geometry.low
        retreat, retreat_target = geometry.retreat, geometry.retreat_target
        R, hover = geometry.rotation, geometry.hover

        retreat_error = None
        release_error = None
        release_timeout = gcfg.get("release_open_timeout_s")
        if release_timeout is not None:
            release_timeout = float(release_timeout)
            if not np.isfinite(release_timeout) or release_timeout < 0:
                raise _PostPlaceRetreatPlanError("release opening timeout must be finite and nonnegative")
        # The optional model-backed adapter previews both the held-object
        # transport and an empty-tool escape before touching the destination.
        # It is re-read from actual physics below, before opening.
        from . import mujoco_withdrawal
        try:
            mujoco_withdrawal.prepare(self, make_transform(R, target),
                carry_goals=([lift.q] if lift is not None else []) + [pre.q, low.q],
                **({} if aim is None else {'deadline': aim.deadline}))
            if aim is not None:
                aim.guard(self, q_now)
        except (SkillError, SafetyViolation, ValueError) as exc:
            raise _PostPlaceRetreatPlanError(str(exc)) from exc
        if lift is not None:
            try:
                lifted = self._move_carry(
                    lift.q, duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                    **({} if aim is None else aim.motion_arguments()),
                )
            except carry_attachment.AttachmentInvalid:
                raise
            except (SkillError, SafetyViolation) as exc:
                raise _PreCarryLiftError(f"pre-carry lift was interrupted: {exc}") from exc
            if not lifted:
                raise _PreCarryLiftError("pre-carry lift did not settle; keeping the grasp")
            self.memory.add("action", f"reached planned carry height {self._height_to_world(hover[2]):.3f} m "
                            "before horizontal transport")
        if not self._move_carry(pre.q, duration_s=float(gcfg.get("move_duration_s", 2.5)),
                                     **({} if aim is None else aim.motion_arguments())):
            raise SkillError("did not settle above the place target")
        self.arm.harness.allow_grasp_descent(target[:2], z_min=release_z - 0.02)
        from . import release_episode
        release = None
        model_withdrawal = None
        planned_release_generation = None
        withdrawal_completed = False
        try:
            if not self._move_carry(low.q, duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                                         **({} if aim is None else aim.motion_arguments())):
                raise SkillError("did not settle at place pose")
            # A grasp ceiling must not collapse withdrawal onto release. An
            # explicitly capable simulator plans from its actual colliders;
            # full-pose IK and the existing safety path still have to succeed.
            try:
                if aim is not None:
                    aim.transport_guard()
                model_withdrawal = mujoco_withdrawal.prepare(self, make_transform(R, target))
                if aim is not None:
                    aim.admit_release(model_withdrawal)
            except (SkillError, SafetyViolation, ValueError) as exc:
                raise _PostPlaceRetreatPlanError(str(exc)) from exc
            if model_withdrawal is not None:
                from types import SimpleNamespace
                retreat = SimpleNamespace(q=model_withdrawal.q)
                retreat_target = model_withdrawal.target[:3, 3].copy()
            try:
                if retreat is not None:
                    release = release_episode.begin(self, retreat.q,
                        float(gcfg.get("descend_duration_s", 2.0)))
                if model_withdrawal is not None and release is None:
                    if aim is not None:
                        aim.admit_release(model_withdrawal)
                    model_withdrawal.retain()
                    try:
                        model_withdrawal.open_hand()
                    except (SkillError, SafetyViolation) as exc:
                        raise _PostPlaceRetreatPlanError("release command unavailable: " + str(exc)) from exc
                else:
                    if (release is None and release_timeout is not None
                            and self.cfg.arm.get('type') == 'isaac'
                            and getattr(self.arm, 'motion_planner', None) is not None):
                        planned_release_generation = self.arm.harness._halt_generation
                    release_episode.open_hand(self, release,
                        **({} if planned_release_generation is None else
                           {'halt_generation': planned_release_generation}))
            except carry_attachment.AttachmentInvalid:
                raise
            except Exception as exc:
                if carry_attachment.active(self) is not None:
                    carry_attachment.active(self).reject("release handoff unavailable: " + str(exc))
                raise
            # This is the native intentional release, after the separate
            # release-episode identity/geometry checks. It ends carry scope.
            self._gripper_opens_completed += 1
            self._carry_attachment = None
            time.sleep(float(self.cfg.grasp.get("close_settle_s", 0.0)))
            if release_timeout is not None and release is not None:
                try:
                    release_episode.wait_open(self, release, timeout_s=float(release_timeout))
                except (SkillError, SafetyViolation) as exc:
                    release_error = str(exc)
                    retreat_error = release_error
            elif planned_release_generation is not None:
                try:
                    release_episode.wait_planned_open(self, timeout_s=release_timeout,
                        halt_generation=planned_release_generation)
                except (SkillError, SafetyViolation) as exc:
                    release_error = str(exc)
                    retreat_error = release_error
            elif release_timeout is not None:
                # In simulation, a wall-clock dwell does not guarantee the
                # rate-limited fingers have opened. Retracting while they
                # still touch the payload can tip a successfully placed can.
                deadline = time.monotonic() + float(release_timeout)
                while True:
                    width = self._gripper_width_frac()
                    if width is not None and width >= .98:
                        break
                    if time.monotonic() >= deadline:
                        release_error = "actual jaws did not confirm full opening before retraction"
                        retreat_error = release_error
                        break
                    time.sleep(.05)
            # From here the object IS placed: reconcile the held state BEFORE
            # the ascent, or an ascent abort leaves held_object latched and a
            # retry descends onto the object we just released.
            placed = self.held_object
            self.beliefs.update(
                # Re-register under the DETECTOR label so the watcher's next
                # fusion merges here instead of creating a query-string ghost.
                # Use the OBJECT's aim point, not the TCP's: `target` may have
                # been shifted to compensate for how the object sits in the
                # jaws, and recording that would seed the belief offset by
                # exactly the amount the compensation just removed. The world
                # model is the table frame.
                self._held_det_label or placed,
                self._to_world(np.array([x, y, release_z], dtype=float)),
                0.8, color=self._held_color
            )
            self.held_object = None
            self._held_arm = None
            self._held_det_label = None
            self._held_offset = None
            self._held_support_offset_m = None
            self._held_color = None
            self._held_width_m = None
            self.memory.add("action", f"placed {placed!r} at {np.round(self._to_world(target), 3).tolist()}")
            try:
                if release_error is None and release is not None:
                    release_episode.wait_geometry(self, release)
                    ascended = release_episode.withdraw(self, release)
                    release_episode._guard(self, release)
                    withdrawal_completed = bool(ascended)
                elif model_withdrawal is not None:
                    ascended = release_error is None and model_withdrawal.withdraw()
                else:
                    ascended = release_error is None and self._move_joints(
                        retreat.q if retreat is not None else pre.q,
                        duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                        **({} if planned_release_generation is None else
                           {'_halt_generation': planned_release_generation}),
                    )
                if retreat is not None and not ascended and release_error is None:
                    retreat_error = "did not settle at the post-place retreat pose"
                if model_withdrawal is not None and ascended:
                    model_withdrawal.after_withdrawal()
            except (SkillError, SafetyViolation) as e:
                if retreat is not None:
                    retreat_error = f"post-place retreat aborted: {e}"
                # Keep legacy behavior when no explicit retreat was requested.
                self.memory.add("note", f"placed, but the ascent aborted: {e}")
        finally:
            if release is None:
                if (model_withdrawal is not None
                        and getattr(self.arm.harness, "_pending_model_withdrawal", None) is model_withdrawal):
                    model_withdrawal.clear_grasp_exemption()
                else:
                    self.arm.harness.clear_grasp_exemption()
            else:
                release_episode.finish(self, release, completed=withdrawal_completed)
        # Report where the OBJECT was aimed, not where the TCP was sent. The
        # postcondition channel scores the object's final pose against this,
        # so returning the offset-compensated TCP point would grade the place
        # against the wrong thing and quietly forgive the compensation error.
        result = {}
        if aim is not None:
            result['placement_aim'] = {**aim.receipt(), 'requested_object_xy_m': [x, y],
                                       'target_tcp_m': target.tolist()}
        if model_withdrawal is not None:
            result["release_clearance"] = model_withdrawal.receipt()
            retreat_target = model_withdrawal.target[:3, 3].copy()
        if release_error is not None:
            result.update(ok=False, stage="release", error=release_error, home_skipped=True)
        if retreat_target is not None:
            result["post_place_retreat"] = {
                "ok": retreat_error is None,
                "tcp_at": [round(float(v), 3) for v in self._to_world(retreat_target)],
            }
            if retreat_error is not None:
                # Release already happened. Do not retry this grasp/place or
                # take the unsafe home sweep after a failed clearance motion.
                result.update(ok=False, stage="release" if release_error else "retreat", error=retreat_error,
                              home_skipped=True)
        # Reported in the table frame (the agent's coordinates): `at` is the
        # caller's object aim (x, y), not the compensated `target`.
        x, y, release_z = (float(v) for v in self._to_world(np.array([x, y, release_z], dtype=float)))
        return {"placed": placed,
                "at": [round(float(x), 3), round(float(y), 3),
                       round(float(release_z), 3)],
                "tcp_at": [round(float(v), 3) for v in self._to_world(target)],
                **result}

    def _destination_fix(self, label: str):
        """Where is the destination RIGHT NOW, not when the task started?

        MEASURED FAILURE this guards against (LIBERO-Pro `libero_spatial_swap`,
        task 3): the destination moved after its belief was seeded, the skill
        aimed 3.4 cm off, the object landed 0.8 cm from that stale aim, and the
        postcondition CONFIRMED the place because it asks "did the object reach
        where I aimed", not "did it reach the destination". LIBERO scored it
        false. Verified false claims went 5 -> 13 on that suite while the
        instruction-perturbation suite stayed at 5, so this is specifically a
        targeting failure, not an actuation one.

        A stale belief is worse than no belief here, so a fresh observation
        wins over the stored one whenever it is available. Returns
        `(position, top_z)` in the TABLE frame (the world model's, the pose
        channel's and place_at's) or None when nothing can be observed, in
        which case the caller keeps today's behaviour.
        """
        # The external pose channel, when attached, sees the destination even
        # when the camera cannot (benchmark oracle mode). Same provenance
        # caveat as everywhere else it is used.
        if self._object_pose is not None:
            try:
                p = self._object_pose(label)
                if p is not None:
                    p = np.asarray(p, float)[:3]
                    return p, float(p[2])
            except Exception:
                pass
        try:
            frame, fix = self._localize(label)
            world = self._fix_to_world(fix)      # the arm-frame fix, remembered and returned in the table frame
            self.beliefs.update(
                fix.detection.label or label, world.position, fix.detection.conf,
                extent=fix.extent, top_z=float(world.points[:, 2].max()),
                color=detection_color(frame.rgb, fix.detection),
            )
            return np.asarray(world.position, float), float(world.points[:, 2].max())
        except Exception:
            return None

    def _native_wrist_camera(self) -> bool:
        cfg = getattr(self, "cfg", None)
        camera = getattr(cfg, "camera", {})
        return (camera.get("extrinsics", {}).get("mode") == "eye_in_hand"
                and self._force_contact_close_enabled())

    def _overview_destination(self, label: str):
        """Resolve once from the saved overview, keeping the jaws unchanged."""
        self.skill_move_home()
        self.observe(require_fresh=True)
        reason = self._home_return_skip_reason()
        if reason:
            raise SkillError(f"destination lookup stopped: {reason}")
        fresh = self._destination_fix(label)
        if fresh is None:
            raise SkillError(f"cannot locate destination {label!r} from the overview; object was not released")
        pos, top = fresh
        pos = np.asarray(pos, dtype=float).reshape(-1).copy()
        if pos.shape != (3,) or not np.isfinite(pos).all() or not np.isfinite(top):
            raise SkillError(f"invalid position for destination {label!r}")
        return pos, float(top)

    def _place_at_destination(self, fix) -> dict:
        pos, top = fix
        drop = top + float(self.cfg.grasp.get("release_clearance_m", 0.06))
        return self.skill_place_at(float(pos[0]), float(pos[1]), drop)

    def skill_place_on_object(self, label: str) -> dict:
        self._check_holding_arm()        # spec D9: before any work
        if self._native_wrist_camera():
            self._reconcile_held()
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        # The fixed box grouping prim has no cavity pose. Use its calibrated
        # release pose through the existing place_at safety checks.
        box = self.cfg.grasp.get("open_box")
        if box is not None and str(label).strip().lower() in _OPEN_BOX_WORDS:
            try:
                center = np.asarray(box.get("center_xy_m"), dtype=float)
                release = float(box.get("release_height_m"))
            except (TypeError, ValueError) as exc:
                raise SkillError("configured open box has an invalid release pose") from exc
            if (center.shape != (2,) or not np.isfinite(center).all()
                    or not np.isfinite(release) or release <= 0):
                raise SkillError("configured open box requires finite XY and a positive release height")
            if box.get("support_top_z_m") is not None:
                support_z = float(box.get("support_top_z_m"))
                if not np.isfinite(support_z):
                    raise SkillError("configured open box requires a finite support height")
                release = self._supported_release_height(support_z, release)
            # The box pose is this arm's configuration (its base frame);
            # place_at takes table-frame coordinates.
            at = self._to_world(np.array([center[0], center[1], release], dtype=float))
            result = self.skill_place_at(float(at[0]), float(at[1]), float(at[2]))
            return {**result, "destination": "open box", "destination_kind": "configured_point"}
        if self._native_wrist_camera():
            return self._place_at_destination(self._overview_destination(label))
        # Re-check the destination immediately before committing to a drop
        # point. The belief may be stale: it was seeded when the task started
        # and the world does not hold still, which is the whole premise of the
        # perturbed benchmarks.
        fresh = self._destination_fix(label)
        if fresh is not None:
            pos, top = fresh
        else:
            belief = self.beliefs.find(label)
            if belief is None:
                frame, fix = self._localize(label)
                world = self._fix_to_world(fix)  # table frame: the world model's and place_at's
                self.beliefs.update(
                    fix.detection.label or label, world.position,
                    fix.detection.conf, extent=fix.extent,
                    top_z=float(world.points[:, 2].max()),
                    color=detection_color(frame.rgb, fix.detection),
                )
                # The detector's label may differ from the user's query.
                return self._place_at_destination(
                    (world.position, float(world.points[:, 2].max())))
            # Use the actually observed highest point of the target, never OBB
            # extents (those are eigenvalue-ordered, not axis-aligned).
            pos = belief.position
            top = (belief.top_z if belief.top_z is not None
                   else float(belief.position[2]))
        drop = self._supported_release_height(
            top, top + float(self.cfg.grasp.get("release_clearance_m", 0.06)))
        return self.skill_place_at(float(pos[0]), float(pos[1]), drop)

    def skill_push_object(self, label: str, direction: str, distance_m: float = 0.08) -> dict:
        """ASPIRE push primitive: pre-contact approach then straight-line push."""
        dirs = {
            "forward": np.array([1.0, 0.0]), "back": np.array([-1.0, 0.0]),
            "left": np.array([0.0, 1.0]), "right": np.array([0.0, -1.0]),
        }
        if direction not in dirs:
            raise SkillError(f"direction must be one of {sorted(dirs)}")
        d2 = dirs[direction]
        frame, fix = self._localize(label)
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        # Contact LOW on the object (a third of its height, min 1.5 cm above
        # the table) so the push doesn't topple or skim over it.
        top_z = float(fix.points[:, 2].max())
        obj_h = max(top_z - table_z, 0.02)
        push_z = table_z + max(0.015, min(0.35 * obj_h, 0.06))
        # Pre-contact standoff must clear the object's own footprint.
        half_len = float(fix.extent[0]) / 2 if fix.extent is not None else 0.05
        standoff = 0.06 + half_len
        d3 = np.array([d2[0], d2[1], 0.0])
        start = fix.position - d3 * standoff
        end = fix.position + d3 * distance_m
        start[2] = end[2] = push_z
        from ..grasping.obb_grasp import _yaw_rotation

        R = _yaw_rotation(float(np.arctan2(d2[1], d2[0])),
                          axis_order=self._tool_axis_order)
        q_now = self.arm.get_state().q
        hover = start + np.array([0.0, 0.0, 0.10])
        ik_hover = self.kin.ik(make_transform(R, hover), q_now)
        ik_start = self.kin.ik(make_transform(R, start), ik_hover.q if ik_hover.success else q_now)
        ik_end = self.kin.ik(make_transform(R, end), ik_start.q if ik_start.success else q_now)
        if not (ik_hover.success and ik_start.success and ik_end.success):
            raise SkillError("push poses unreachable")
        self._set_gripper(self._grip_closed, effort=0.8)  # push with closed jaws
        self._move_joints(ik_hover.q, duration_s=2.0)
        center = (start[:2] + end[:2]) / 2
        radius = float(np.linalg.norm(end[:2] - start[:2]) / 2 + 0.10)
        self.arm.harness.allow_grasp_descent(center, radius_m=radius, z_min=push_z - 0.02)
        try:
            self._move_joints(ik_start.q, duration_s=1.5)
            self._move_joints(ik_end.q, duration_s=2.0)
            self._move_joints(ik_hover.q, duration_s=1.5)
        finally:
            self.arm.harness.clear_grasp_exemption()
        end_w = self._to_world(end)            # the world model and the report: table frame
        self.beliefs.update(label, end_w, 0.6)
        self.memory.add("action", f"pushed {label!r} {direction} by {distance_m:.2f} m")
        return {"pushed": label, "direction": direction, "new_position_estimate": np.round(end_w, 3).tolist()}

    def skill_pick_and_place(
        self,
        object: str,
        destination: str | None = None,
        material: str | None = None,
    ) -> dict:
        """One-call pick-and-place that PERSISTS: resolve (color queries
        work) -> grasp -> place on the destination object or the configured
        drop zone -> home. Both stages keep retrying with fresh perception
        (re-home, re-scan, re-plan -- GraspGen-X sampling is stochastic, so
        every retry is a genuinely new grasp) until they succeed or the
        persistence budget (grasp.persist_seconds / grasp.max_pick_attempts)
        runs out. No LLM in the loop; this is the reflex the web chat calls
        for "pick and place pink object"."""
        # An operator halt from here on ends the grasp stage (`_end_on_halt`).
        halt_generation = self._halt_generation_now()
        # R9: its place stage would refuse a model release on a base_pose arm
        # after the grasp; refuse before any command instead.
        self._model_release_refusal()
        # R5: another arm's provisional hold keeps its slot.
        self._refuse_on_another_arms_provisional("pick_and_place")
        try:
            carry_attachment.check(self)
            self._reconcile_held()
        except carry_attachment.AttachmentInvalid as exc:
            return carry_attachment.failure_result(self, exc, grasp_attempts=0, place_attempts=0)
        from .held_observation import HeldObservationInvalid
        already_held = None
        # Another arm's hold: this arm would place what it does not hold (D9).
        self._check_holding_arm()
        if self.held_object:
            hq, oq = self.held_object.lower(), str(object).lower()
            if hq in oq or oq in hq:
                # The jaws already hold what was asked for (a previous task
                # grasped it and died before placing): skip straight to the
                # place stage instead of refusing.
                already_held = self.held_object
                self.memory.add(
                    "note",
                    f"already holding {already_held!r} -- going straight to the place stage",
                )
            else:
                raise SkillError(
                    f"already holding {self.held_object!r}; place it first"
                )
        # Nothing moves while an interrupted grasp's provisional hold survives
        # that reconcile: the grasp stage would open the jaws on it.
        self._raise_on_a_possible_hold()
        t0 = time.monotonic()
        timings: dict[str, float] = {}
        gcfg = self.cfg.grasp
        destination_name = (gcfg.get("drop_zone_name", "drop zone")
                            if not destination or _names_drop_zone(destination) else destination)
        failure_destination = {}
        if not destination or _names_drop_zone(destination):
            dz = gcfg.get("drop_zone", [0.30, -0.20])
            # A failed grasp/carry still aimed at this calibrated floor mark.
            # Without this context, the postcondition checker may resolve its
            # colour name as a movable object or an absent body's zero pose.
            # The mark is this arm's (its base frame); reported table-frame.
            dz_w = self._to_world(np.array([float(dz[0]), float(dz[1]), 0.0]))
            failure_destination = {"destination": destination_name,
                "destination_kind": "configured_point",
                "target": [float(dz_w[0]), float(dz_w[1])]}
            if self.cfg.arm.get("mj_delivery_area") is not None:
                from ..sim.mujoco_placement import Region
                region = Region.parse(self.cfg.arm.get("mj_delivery_area"))
                failure_destination = {"destination": region.name, "destination_kind": "configured_region",
                                       "region": region.as_dict()}
        elif (str(destination).strip().lower() in _OPEN_BOX_WORDS
                and gcfg.get("open_box") is not None):
            box = gcfg.get("open_box")
            try:
                center = np.asarray(box.get("center_xy_m"), dtype=float)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise SkillError("configured open box requires a finite XY center") from exc
            if center.shape != (2,) or not np.isfinite(center).all():
                raise SkillError("configured open box requires a finite XY center")
            # Its grouping prim can sit at the origin while the cavity is
            # elsewhere. A failed grasp needs the same destination provenance
            # as a completed place, before the place stage has been reached.
            failure_destination = {"destination": "open box", "destination_kind": "configured_point",
                "target": self._to_world(np.array([center[0], center[1], 0.0]))[:2].tolist()}
        max_attempts = max(int(gcfg.get("max_pick_attempts", 8)), 1)
        deadline = t0 + float(gcfg.get("persist_seconds", 120.0))
        # #2.3: the budget must not multiply across tiers. The reflex tier
        # burns persist_seconds, escalates, and the LLM tier calls
        # pick_and_place on the same object again with a fresh budget -- a
        # visitor watched 4 minutes of retries on one cube. The orchestrator
        # opens a task epoch (`begin_task_budget`); inside one, the remaining
        # TASK budget caps this call's deadline.
        task_deadline = getattr(self, "_task_deadline", None)
        if task_deadline is not None:
            if time.monotonic() >= task_deadline:
                return {
                    **failure_destination,
                    "ok": False, "stage": "grasp", "outcome": OUTCOME_STUCK,
                    "error": "task persistence budget exhausted (an earlier tier already spent it)",
                    "suggestion": "ask the visitor to reposition the object or pick a different one",
                    "ask": _stuck_ask(object, "persistence budget exhausted", 0),
                }
            deadline = min(deadline, task_deadline)

        # Keep a base-frame destination for this operation while the wrist
        # camera approaches the object. Separate commands never reuse it.
        destination_fix = None
        if destination and not _names_drop_zone(destination) and self._native_wrist_camera():
            destination_fix = self._overview_destination(destination)

        grasp = (
            {"held": already_held, "grip_verified": None, "grip_profile": None}
            if already_held
            else None
        )
        placed = None
        last_err = "unknown"
        place_err = "unknown"
        attempt = 0
        p_attempt = 0
        tp = t0
        stop = None  # why the grasp loop stopped early (None = budget spent)
        # The last failed attempt's own error (a grasp miss, a failed place),
        # until an attempt holds: a halt after it (in the re-home, the next
        # attempt) carries it, so the call is filed as that failure.
        decided = None
        while placed is None:
            # ── grasp stage (skipped when re-entering after a mid-carry slip
            # left something verified in the jaws -- cannot happen today, but
            # the guard keeps the loop honest) ─────────────────────────────
            while grasp is None and attempt < max_attempts:
                attempt += 1
                if attempt == 1:
                    self.memory.add(
                        "note",
                        f"picking up {object!r} (attempt 1/{max_attempts}; "
                        "I will keep trying until it works or the budget runs out)",
                    )
                if attempt > 1:
                    if time.monotonic() > deadline:
                        attempt -= 1  # this attempt never ran
                        break
                    self.memory.add(
                        "note",
                        f"not giving up: attempt {attempt}/{max_attempts} on "
                        f"{object!r} -- re-homing, re-scanning, planning a fresh grasp",
                    )
                    try:  # clear the camera view so the re-scan actually sees
                        self.skill_move_home()
                    except MotionHalted as halted:
                        _keep_decided_failure(halted, decided)
                        raise  # STOP during the re-home ends this skill: no re-scan, no retry
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                tg = time.monotonic()
                grasp_motions_before = self._arm_motions_counted()  # see _grasp_with_persistence
                try:
                    res = self.skill_grasp_object(object, material=material,
                        **({"_planning_deadline": deadline}
                           if os.environ.get("CASCADE_OBSERVED_FINGER_GATE") == "1" else {}))
                except MotionHalted as halted:
                    _keep_decided_failure(halted, decided)
                    raise  # the operator's halt ends this skill: no re-home, no retry
                except (SkillError, SafetyViolation) as e:
                    # the sim retry classifies a refusal by its exact class (R6):
                    # this result keeps only the text (a perception stop has no prefix)
                    retry_state = _sim_retry_state(self)
                    if retry_state is not None:
                        retry_state.refusal_exc_type = type(e).__name__
                    if isinstance(e, carry_attachment.AttachmentInvalid):
                        return carry_attachment.failure_result(self, e,
                            **failure_destination, grasp_attempts=attempt, place_attempts=0)
                    note = _perception_stop(e, attempt)
                    if note is not None:
                        # An operator halt since the skill started ends it as
                        # the halt, not as this refusal (A3: `_end_on_halt`).
                        self._end_on_halt(halt_generation, failure=decided)
                        return {**failure_destination, "ok": False, "stage": "grasp",
                                "error": str(e), "home_skipped": True,
                                "grasp_attempts": attempt, "note": note}
                    res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    if isinstance(getattr(e, "observed_selection", None), dict):
                        res["observed_selection"] = e.observed_selection
                timings[f"grasp_attempt{attempt}_s"] = round(time.monotonic() - tg, 2)
                if res.get("ok", True) and res.get("held"):
                    grasp = res
                    decided = None   # held: the earlier misses no longer decide this call
                    break
                decided = res.get("error") or decided   # this attempt's miss, decided before any later halt
                self._end_on_halt(halt_generation, failure=decided)
                last_err = str(res.get("error", "grasp failed"))
                if os.environ.get("CASCADE_OBSERVED_FINGER_GATE") == "1":
                    out = {**failure_destination, "ok": False, "stage": "grasp",
                           "error": last_err, "home_skipped": True, "grasp_attempts": attempt,
                           "note": "observed-finger attempt failed; automatic recovery was not geometrically checked",
                           **({"observed_selection": res["observed_selection"]}
                              if "observed_selection" in res else {})}
                    if self._may_ask_move_home(grasp_motions_before):
                        _with_move_home_first(out, source=res)  # moved the arm; no halt, nothing held
                    return out
                self.memory.add("outcome", f"pick attempt {attempt} failed: {last_err[:100]}")
                # B36: an attempt can fail AFTER the close took the object
                # (measured on Isaac: "did not settle at grasp lift pose" with
                # the cube in the jaws). Re-running the grasp then refuses
                # "already holding" until the budget is gone and the cube is
                # never placed. If the jaws hold what was asked for, go on to
                # place it -- exactly what the entry of this skill does for an
                # object a previous task left held.
                held = self._held_after_failed_attempt(object)
                if held is not None:
                    is_requested, held_label = held
                    if is_requested:
                        self.memory.add(
                            "note",
                            f"attempt {attempt} failed after the close but {held_label!r} is in "
                            "the jaws -- going on to place it instead of grasping again",
                        )
                        grasp = {"held": held_label, "grip_verified": None, "grip_profile": None,
                                 "recovered_after": last_err}
                        decided = None
                        break
                    last_err += f" (holding {held_label!r}, not {object!r}; a new grasp cannot start)"
                    stop = "holding another object"
                    break
                # Fail fast on what a retry cannot cure (e-stop; an object the
                # world model has NEVER seen after re-scans -- a typo or not
                # on the table; an object wider than the jaws): burning two
                # minutes on those reads as a hang at a live booth.
                stop = self._grasp_retry_verdict(object, attempt, last_err)
                if stop:
                    last_err += f" ({stop})"
                    break
            if grasp is None:
                # Never leave the arm hanging mid-pose over the table after a
                # failed attempt -- park it (best effort, nothing is held).
                # Not under a latched e-stop: the operator stopped the arm.
                park_halt = None
                if not self._home_policy_enabled() and not self._estopped():
                    try:
                        self.skill_move_home()
                    except MotionHalted as exc:
                        park_halt = str(exc)   # the operator's halt stopped the park: named below
                    except (SkillError, SafetyViolation):
                        pass
                out = {
                    **failure_destination,
                    "ok": False, "stage": "grasp",
                    "error": f"grasp failed after {attempt} attempts "
                             f"({round(time.monotonic() - t0, 1)}s): {last_err}",
                    "suggestion": "check world_state / camera_snapshot; the object may be unreachable or mis-detected",
                }
                if park_halt is not None:
                    _stop_first(out, _halted_text("final park", park_halt))
                # The robot did everything it can on its own: the budget is
                # spent, or the stop reason is one no retry cures (never
                # seen, too wide). That is STUCK -- a request to the human,
                # not a failure for the planner to retry. An e-stop or an
                # unfinished contact episode is a harness state and stays a
                # plain failure.
                if _is_human_curable(stop):
                    out["outcome"] = OUTCOME_STUCK
                    out["ask"] = _stuck_ask(object, last_err, attempt)
                return out

            # ── place stage ────────────────────────────────────────────────
            tp = time.monotonic()
            while p_attempt < max_attempts:
                p_attempt += 1
                if p_attempt > 1:
                    if time.monotonic() > deadline:
                        p_attempt -= 1
                        break
                    self.memory.add(
                        "note",
                        f"still holding {object!r}: place attempt "
                        f"{p_attempt}/{max_attempts} -- re-locating the target",
                    )
                    try:  # carry it home first: clears the view of the arm
                        self.skill_move_home()  # + held object for the re-scan
                    except MotionHalted as halted:
                        _keep_decided_failure(halted, decided)
                        raise  # STOP during the re-home ends this skill: no re-scan, no retry
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                try:
                    place_harness = getattr(self.arm, "harness", None)
                    occupancy = getattr(place_harness, "occupancy", None)
                    place_generation = (place_harness._halt_generation
                                        if getattr(occupancy, "tracks_payload", False) else None)
                    if destination and not _names_drop_zone(destination):
                        if destination_fix is not None and p_attempt == 1:
                            res = self._place_at_destination(destination_fix)
                        else:
                            res = self.skill_place_on_object(destination)
                    else:
                        from .mujoco_region import select as select_region
                        region_plan = select_region(self, retain_aim=True)
                        region_selection = None if region_plan is None else region_plan.report
                        dz = (region_selection["target"] if region_selection is not None
                              else gcfg.get("drop_zone", [0.30, -0.20]))
                        # This arm's drop zone (its base frame; a region plan
                        # never runs on a base_pose arm, R9): place_at takes
                        # table-frame coordinates.
                        dz_w = self._to_world(np.array([float(dz[0]), float(dz[1]), 0.0]))
                        res = self.skill_place_at(float(dz_w[0]), float(dz_w[1]),
                            **({} if region_plan is None else {'_model_plan': region_plan}))
                        if region_selection is not None:
                            res.update(destination=region_selection["destination"],
                                       destination_kind="configured_region", region_selection=region_selection)
                    if (res.get("placed") and not self.held_object
                            and (res.get("home_skipped") is True
                                 or res.get("post_place_retreat", {}).get("ok") is False)):
                        # The object was released, so another pick/place retry
                        # could strike it. Preserve the failure and stop here.
                        placed = res
                        break
                    if not res.get("ok", True):
                        raise SkillError(str(res.get("error", "place failed")))
                    placed = res
                    break
                except (SkillError, SafetyViolation) as e:
                    place_err = f"{type(e).__name__}: {e}"
                    if isinstance(e, MotionHalted):
                        _keep_decided_failure(e, decided)
                        raise  # the operator's halt ends this skill: no re-home, no place retry
                    decided = place_err   # this place failed before any later halt
                    if isinstance(e, carry_attachment.AttachmentInvalid):
                        return carry_attachment.failure_result(self, e,
                            **failure_destination, grasp_attempts=attempt, place_attempts=p_attempt)
                    if isinstance(e, HeldObservationInvalid):
                        return {**failure_destination, "ok": False, "stage": "held_observation",
                                "error": place_err, "holding": self.held_object,
                                "home_skipped": True, "grasp_attempts": attempt,
                                "place_attempts": p_attempt}
                    if isinstance(e, (_PostPlaceRetreatPlanError, _PreCarryLiftError)):
                        return {
                            **failure_destination,
                            "ok": False, "stage": ("carry_clearance" if isinstance(e, _PreCarryLiftError)
                                                    else "retreat_plan"), "error": place_err,
                            "holding": self.held_object, "home_skipped": True,
                            "grip_verified": grasp.get("grip_verified"),
                            "grasp_attempts": attempt, "place_attempts": p_attempt,
                        }
                    self.memory.add(
                        "outcome", f"place attempt {p_attempt} failed: {place_err[:100]}"
                    )
                    if self._native_wrist_camera():
                        # Repeating an unobservable target or a refused motion
                        # cannot repair it. Preserve the grasp and report the
                        # reason instead of spending minutes retrying.
                        break
                    if self.arm.harness.estopped:
                        place_err += " (e-stop latched; not retrying)"
                        break
                    if _is_contact(place_err):
                        place_err += " (contact detected: something blocked the arm; not retrying into it)"
                        break
                    if not self.held_object:
                        break  # slipped mid-carry: placing again is pointless
            if placed is None:
                can_regrasp = (
                    not self.held_object
                    and not self.arm.harness.estopped
                    and not _is_contact(place_err)
                    and time.monotonic() < deadline
                    and attempt < max_attempts
                )
                if not can_regrasp:
                    out = {
                        **failure_destination,
                        "ok": False, "stage": "place",
                        "error": f"place failed after {p_attempt} attempts: {place_err}",
                        "note": (
                            f"still holding {object!r}; try place_at with explicit coordinates"
                            if self.held_object
                            else f"{object!r} slipped while carrying; grasp budget exhausted"
                        ),
                        "grip_verified": grasp.get("grip_verified"),
                    }
                    if grasp.get("recovered_after"):
                        # B36: the grasp stage ended on a failed attempt with the object held
                        out["grasp_recovered_after"] = grasp["recovered_after"]
                    # Still holding after every place attempt (destination
                    # blocked / not placeable) or slipped with the grasp
                    # budget spent: the human has to change something. The
                    # e-stop stays a plain failure.
                    if not self.arm.harness.estopped:
                        out["outcome"] = OUTCOME_STUCK
                        out["ask"] = _stuck_ask(object, place_err, p_attempt, stage="place",
                                                destination=destination_name,
                                                holding=bool(self.held_object))
                    return out
                self.memory.add(
                    "note",
                    f"{object!r} slipped while I was carrying it -- starting over",
                )
                grasp = None  # back to the grasp stage within the same budget
        timings["place_s"] = round(time.monotonic() - tp, 2)

        return_home = {"attempted": False, "reason": "disabled_by_profile"}
        if placed.get("home_skipped", False):
            return_home["reason"] = f"{placed.get('stage', 'retreat')}_failed"
        if (not placed.get("home_skipped", False)
                and bool(self.cfg.grasp.get("home_after_place", True))
                and not self._home_policy_enabled()):
            try:  # clear the camera view for the next command; best effort
                if place_generation is None:
                    self.skill_move_home()
                else:
                    self.skill_move_home(_halt_generation=place_generation)
                return_home = {"attempted": True, "ok": True, "at": "home"}
            except (SkillError, SafetyViolation) as exc:
                return_home = {"attempted": True, "ok": False, "error": str(exc)}
        total = round(time.monotonic() - t0, 2)
        destination_name = placed.get("destination", destination_name)
        self.memory.add(
            "action",
            f"pick_and_place {object!r} -> {destination_name} in {total}s",
        )
        result = {
            "picked": object,
            "placed_at": placed.get("at"),
            "destination": destination_name,
            "grip_profile": grasp.get("grip_profile"),
            "grip_verified": grasp.get("grip_verified"),
            "grasp_attempts": attempt,
            "place_attempts": p_attempt,
            "duration_s": total,
            "timings": timings,
            "return_home": return_home,
            "next_action": _PICK_REPORT_NEXT_ACTION,
        }
        if "target_resolution" in grasp:
            result["target_resolution"] = grasp["target_resolution"]
        if "observed_selection" in grasp:
            # Observed-finger gate: ranked-last / dropped / attempted candidates.
            result["observed_selection"] = grasp["observed_selection"]
        if grasp.get("recovered_after"):
            # B36: attempt `grasp_attempts` failed after the close, the jaws
            # held the object, and the place stage went on with it
            result["grasp_recovered_after"] = grasp["recovered_after"]
        if (not destination or _names_drop_zone(destination)
                or placed.get("destination_kind") == "configured_point"):
            result["destination_kind"] = "configured_point"
        if placed.get("destination_kind") == "configured_region":
            result["destination_kind"] = "configured_region"
            result["region_selection"] = placed["region_selection"]
        if "post_place_retreat" in placed:
            result["post_place_retreat"] = placed["post_place_retreat"]
        if "release_clearance" in placed:
            result["release_clearance"] = placed["release_clearance"]
        if "placement_aim" in placed:
            result["placement_aim"] = placed["placement_aim"]
        if placed.get("ok") is False:
            result.update(ok=False, stage=placed.get("stage", "place"),
                          error=placed.get("error", "place failed"),
                          home_skipped=placed.get("home_skipped", False))
        return result

    # ── social / audience skills (deterministic, harness-gated) ─────────

    def skill_describe_scene(self) -> dict:
        """Refresh the camera and distinguish detector estimates from memory."""
        observation = self.skill_get_observation()
        objs = observation["objects_visible"]
        parts = []
        for o in objs:
            color = f"{o['color']} " if o.get("color") else ""
            position = o.get("position")
            location = f" at [{position[0]}, {position[1]}]" if position is not None else ""
            parts.append(f"{color}{o['label']}{location}")
        text = (
            "Unconfirmed detector estimates: " + "; ".join(parts) + "." if parts
            else "No objects were detected in this frame; that does not establish an empty scene."
        )
        text += " Identify visible objects from the current camera image."
        if self.held_object:
            text += f" I am holding the {self.held_object}."
        return {**observation, "description": text, "objects": objs, "holding": self.held_object}

    def skill_count_objects(self, query: str | None = None) -> dict:
        beliefs = self.beliefs.all()
        if query:
            color, noun = parse_color_query(query)
            if color:
                beliefs = [b for b in beliefs if b.color == color]
            if noun:
                beliefs = [
                    b for b in beliefs
                    if noun in b.label.lower() or b.label.lower() in noun
                ]
        return {
            "count": len(beliefs),
            "query": query or "all",
            "labels": [f"{b.color or '?'} {b.label}" for b in beliefs],
        }

    def skill_point_at(self, label: str) -> dict:
        """Deictic gesture: hold the closed gripper over the object for a
        moment ("this one!") and stay there. Answers 'which one is X?'.

        The pointing pose keeps the TCP and the jaws `safety.min_clearance_m`
        from the object, checked on the solved pose: today's top-down hover
        when one clears the object's top; otherwise a lean-in pose that aims
        the tilted gripper at the centre of the top from the arm's side
        (`skills/pointing.py`: tilt 15-45 deg, 4-10 cm from the top, the
        jaws sampled at <= 5 mm, the joints too). Either pose keeps the same
        margin from every other known object, and so does the rest of the
        tool the arm's harness models (`safety.tool_volume`: the reBot's
        wrist camera box), at every pose and leg. The route there is checked
        as well: the straight joint segment from where the arm is, else one
        rise-first waypoint (both keep every known object's margin; on the
        rise the jaws never come closer to the target). Jaws that start
        touching an object (within 5 mm, on their real geometry) first back
        straight out along their approach (never while holding: that
        refuses). The jaws close only where
        everything a close sweeps (the profile's `gripper.jaw_volume`) keeps
        the margin from the target, every known object and the table: where
        the arm starts (or backed out), else at the waypoint or the pointing
        pose. While holding an object it never leans in and never closes the
        jaws. No clear pose, route or close, or planning past its budget (3 s
        of CPU, more on a slower core, 8 s of wall time), refuses before any
        arm or jaw command; an e-stop latched while planning, in the last
        close or in the hold is reported as the e-stop, an operator halt
        received since the call started as the halt (e-stop first), with the
        operator's reason. When more than 3 s of wall time passed between the
        localization and the end of planning, it looks at the object again
        before the first command, on a detection newer than the plan only
        (never the remembered belief, never an image received before that
        look started (an image in flight then aside: 0.1 s) or whose capture
        was first received more than 0.5 s before it, never a camera's
        producer capture no newer than the one the plan was made from on that
        camera): not seen again, it refuses; moved more than
        1 cm (its centre or its top), it is planned once more from where it is
        now, on what is left of the same budget; moved again, refuses (nothing
        was moved). The object top is
        the observed top raised to the occupancy map's column top (capped as
        the pregrasp reads it) and to the top of an object standing over it
        (a cup over the cube)."""
        harness = getattr(self.arm, "harness", None)
        generation = getattr(harness, "_halt_generation", None)
        frame, fix = self._localize(label)
        fix = _host_fix(fix)
        from . import pointing

        # the freshness clock starts here, at the localization, before the
        # budget's core-speed search (#406 review Minor 1): (wall, monotonic,
        # the image the fix came from, that image's camera)
        localized = (pointing._wall(), time.monotonic(), frame, self._localized_camera())
        budget = pointing.Budget()
        try:
            try:
                return self._point_at(label, fix, budget, generation, localized=localized)
            except _PointTargetMoved as moved:
                # once more, from where the target is now, on what is left of
                # the same budget (#399 review Minor 3)
                return self._point_at(label, moved.fix, budget, generation, replanned=True,
                                      localized=moved.localized)
        except pointing.PlanningTimeout as exc:
            # an e-stop latched while planning vetoes every vet, so the search
            # runs long: it is reported as the e-stop (and a halt as the
            # halt), never as the clock
            _raise_if_halted_since_start(harness, generation)
            raise SkillError(f"cannot point at {label!r}: {exc} (the pose and its route are planned before any "
                             f"command); nothing was moved") from None

    def _point_at(self, label: str, fix, budget, generation=None, replanned: bool = False, *,
                  localized) -> dict:
        """skill_point_at after the fix: plan within `budget`, then move.
        `generation`: the harness's halt generation when the call started;
        `replanned`: this is the plan from a fix taken again after a long
        first plan (`_point_at_still_there`); `localized`: (`pointing._wall()`,
        `time.monotonic()`) when `fix` was localized, the image it came
        from and that image's camera (`_localized_camera`)."""
        from . import pointing

        gcfg = self.cfg.grasp
        harness = getattr(self.arm, "harness", None)
        margin = float(self.cfg.safety.get("min_clearance_m", 0.03))
        top, occluders, neighbours = self._pointing_top(label, fix, harness)
        hover = fix.position.copy()
        # Strict top-down poses only solve below ~0.15 m on the B601-RS
        # (wrist limits); clamp the hover or tall objects become unpointable.
        z_max = float(gcfg.get("topdown_z_max", 0.15)) - 0.01
        hover[2] = min(top + float(gcfg.get("pregrasp_offset_m", 0.08)), z_max)
        from ..grasping.obb_grasp import _yaw_rotation

        table_z = float(getattr(getattr(harness, "limits", None), "table_z",
                                self.cfg.safety.get("table_z", 0.0)))
        from .relative_motion import _TOOL_COLUMNS

        approach_col = _TOOL_COLUMNS[self._tool_axis_order][0]
        sweep = (gcfg.get("graspgenx") or {}).get("sweep")
        # a hold on ANOTHER arm of a pair is not this arm's: its jaws are
        # empty and it points as if nothing were held (#400 re-review Minor 1)
        held = None if _hold_on_another_arm(self) else self.held_object
        state = self.arm.get_state()
        q_now = state.q
        try:
            column = pointing.object_column(fix, table_z, top, occluders)
            near = tuple(pointing.neighbour_column(poly, n_top, table_z, name) for poly, n_top, name in neighbours)
            opening = state.gripper_pos if getattr(state, "gripper_valid", True) else None
            jaws = pointing.jaw_samples(sweep, self.cfg.arm.get("gripper"), self._tool_axis_order, opening)
            # the rest of the tool the harness models (`safety.tool_volume`:
            # the reBot's wrist camera box) keeps the same margin from
            # every object at every pose and leg (#385 review I1)
            body = pointing.tool_body(_harness_tool_volume(harness), margin)
        except ValueError as exc:
            raise SkillError(f"cannot point at {label!r}: {exc}; nothing was moved") from None
        # what the jaws are at the pointing pose: closed there, or as they are
        # while holding (no jaw command); the body parts ride along
        posed_jaws = jaws.now if held else jaws.closed
        posed = posed_jaws + body

        # Top-down first: the lowest tool point (TCP or jaws) keeps the
        # margin above the top, and the jaws and the joints keep it from every
        # other known object. `need` is the TCP height that gives the first
        # for a strict top-down pose; a lower hover rises to it within the
        # ceiling, a rung below it is never tried.
        # the box corners and the jaws as they are at the pose (a hinged jaw's
        # measured geometry, `gripper.jaw_volume`, counts too)
        tool = np.vstack([_pointing_tool_points(gcfg, self._tool_axis_order), *posed_jaws,
                          *(part.samples for part in body)])
        clear_z = top + margin
        reach = max(0.0, -_lowest_tool_z(make_transform(_yaw_rotation(0.0, axis_order=self._tool_axis_order),
                                                        [0.0, 0.0, 0.0]), tool))
        need = clear_z + reach
        if not hover[2] >= need:
            hover[2] = min(need + POINT_AT_IK_SLACK_M, z_max)
        rungs = [z for z in (hover[2], z_max - 0.02) if z >= need]

        ik = None
        too_low = None     # the highest lowest-tool-point among rejected solutions
        crowded = None     # (distance, name): the farthest a rejected solution kept from a neighbour
        for z in rungs:
            for yaw in (float(np.arctan2(hover[1], hover[0])), 0.0, np.pi / 4, -np.pi / 4):
                budget.check()
                cand = self.kin.ik(
                    make_transform(_yaw_rotation(yaw, axis_order=self._tool_axis_order),
                                   [hover[0], hover[1], z]), q_now
                )
                if not cand.success:
                    continue
                low = _lowest_tool_z(self.kin.fk(cand.q), tool)
                if not low >= clear_z:     # NaN refuses too
                    too_low = low if too_low is None or low > too_low else too_low
                    continue
                gap = pointing.neighbour_clearance(self.kin, cand.q, posed, near)
                if not gap[0] >= margin:   # NaN refuses too
                    crowded = gap if crowded is None or gap[0] > crowded[0] else crowded
                    continue
                ik = cand
                break
            if ik is not None:
                break

        vet = getattr(harness, "vet_pose", None)
        lean = None
        # Agent text quotes TABLE-frame heights (Q1 a); the checks above and
        # below stay in the arm frame.
        world_z = self._height_to_world
        if ik is None:
            if not rungs:
                from_above = (f"from above, its top-down ceiling is {world_z(z_max):.3f} m, below the object top + "
                              f"margin ({world_z(need):.3f} m: top {world_z(top):.3f} m + clearance {margin:.3f} m + "
                              f"{reach:.3f} m of jaw below the TCP)")
            elif crowded is not None:
                from_above = (f"from above, the best solved pose kept the arm {crowded[0]:.3f} m from "
                              f"{crowded[1]}, less than {margin:.3f} m")
            elif too_low is not None:
                from_above = (f"from above, the best solved pose put the closed jaws at {world_z(too_low):.3f} m, "
                              f"less than {margin:.3f} m above its top ({world_z(top):.3f} m)")
            else:
                from_above = "from above, no pose solved"
            # an e-stop latched while planning is what the agent must hear,
            # never the geometry that failed after it (#385 review M1), and so
            # is a halt received since the call started (#399 review Minor 2)
            _raise_if_halted_since_start(harness, generation)
            if held:
                # the payload is not modelled: no new motion with it in the jaws (main's top-down rule)
                raise SkillError(f"cannot point at {label!r} while holding {held!r}: {from_above}; it leans in "
                                 f"only with empty jaws; nothing was moved")
            lean = pointing.lean_in(self.kin, column, jaws.closed + body, approach_col, q_now, margin=margin,
                                    table_z=table_z, vet=vet, neighbours=near, budget=budget)
            if lean.q is None:
                _raise_if_halted_since_start(harness, generation)
                best = "" if lean.clearance is None else f" (the best solved pose had {lean.clearance.worst(margin)})"
                raise SkillError(
                    f"cannot point at {label!r} on this arm: {from_above}; leaning in "
                    f"{pointing.LEAN_IN_TILTS_DEG[0]:.0f}-{pointing.LEAN_IN_TILTS_DEG[-1]:.0f} deg, no pose "
                    f"keeps the TCP, the jaws and the arm {margin:.3f} m from it, every known object and the "
                    f"table{best}; nothing was moved")
            target = lean.q
        else:
            target = ik.q

        def preflight(q_from, q_to):
            if harness is None or not callable(getattr(harness, "vet_step", None)):
                return None
            from ..safety import trajectory

            try:
                with trajectory.geometry_guard(harness):
                    return trajectory.vet_segment(harness, q_from, q_to, 2.0)
            except SkillError as exc:
                return str(exc)

        # a jaw touching an object first backs straight out: one dense joint
        # path, streamed as one harness-gated motion (never a planner-bound arm's)
        can_withdraw = (callable(getattr(self.arm, "move_joint_path", None))
                        and getattr(self.arm, "motion_planner", None) is None)
        try:
            route = pointing.plan_route(self.kin, q_now, target, column, jaws, approach_col, margin=margin,
                                        table_z=table_z, neighbours=near, close=not held, vet=vet,
                                        preflight=preflight, preflight_direct=lean is not None,
                                        withdraw=can_withdraw, budget=budget, body=body, holding=held)
        except pointing.PlanningTimeout:
            raise
        except pointing.RouteRefused as exc:
            _raise_if_halted_since_start(harness, generation)
            raise SkillError(f"cannot reach the pointing pose over {label!r} keeping {margin:.3f} m from it: "
                             f"{exc}; nothing was moved") from None
        # a long plan acted on an old fix: look at the target again before the
        # first command (#399 review Minor 3)
        self._point_at_still_there(label, fix, localized, harness, generation, replanned)
        # a halt since the call started refuses before the first command, with
        # the operator's reason (outside execute() no skill scope would: the
        # first command's own check clears that halt and moves)
        _raise_if_halted_since_start(harness, generation)
        try:
            if route.withdraw:
                self._mark_motion_started()
                if not self.arm.move_joint_path(route.withdraw, duration_s=2.0):
                    raise SkillError("did not settle after backing the jaws straight out")
            legs = ("did not settle at the rise-first waypoint", "did not settle at the pointing pose")
            for i, q in enumerate(route.targets):
                if route.close_at == i:
                    self._set_gripper(self._grip_closed, effort=0.6)
                if not self._move_joints(q, duration_s=2.0):
                    raise SkillError(legs[i == len(route.targets) - 1])
            if route.close_at == len(route.targets):
                self._set_gripper(self._grip_closed, effort=0.6)
        except MotionHalted:
            # a halt a command noticed (its own check names no reason) is
            # reported with the operator's reason (#406 review Minor 3); an
            # e-stop still wins
            _raise_if_halted_since_start(harness, generation)
            raise
        time.sleep(float(self.cfg.get("gesture", {}).get("point_hold_s", 1.2)))
        # the close can be the last command: an e-stop latched in it (or in
        # the hold) is no success (main closed first, and its next move
        # refused), and neither is a halt (never noted as pointed)
        _raise_if_halted_since_start(harness, generation)
        self.memory.add("action", f"pointed at {label!r}")
        out = {"pointed_at": label, "position": [round(float(x), 3) for x in self._to_world(fix.position)]}
        if lean is not None:
            out["lean_deg"] = int(round(lean.tilt_deg))
        if len(route.targets) > 1:
            out["rose_first"] = True
        if route.withdraw:
            out["backed_out_m"] = round(float(route.withdraw_m), 3)
        return out

    def _point_at_still_there(self, label: str, fix, localized, harness, generation,
                              replanned: bool) -> None:
        """point_at, before its first command: when more than
        `pointing.FRESH_PLAN_S` of wall time passed since `fix` was localized
        (`localized`: (`pointing._wall()`, `time.monotonic()`) then, the image
        `fix` came from and its camera), the plan acted on an old fix, so the
        target is localized again, on a detection newer than the plan only
        (#406 review): `_localize(memory=False)`, never its belief fallback
        (inside the skill that is what was known before the plan, read as
        unmoved), on an image RECEIVED after this look started (`Frame.t`, the
        client's receipt time: a camera that stalled during the plan hands
        back an image received then, #411 review Minor 2), or at most
        `pointing.FRESH_IN_FLIGHT_S` before it (an image the camera's pump had
        received but not yet published, #413 review Minor 3), and never before
        the plan started. When either
        image carries a producer capture (an Isaac / OVRTX camera's own
        clock) and both come from one camera, the new one's must also be newer
        than the one `fix` came from (`freshness.newer_capture`): a bridge
        re-sends an old capture in a new Frame, with a new `Frame.t` (#411
        review Minor 1); one that cannot be compared with it (another
        identity, none any more) refuses too. An image another camera found
        (`_localized_camera`) is never compared with it: two cameras' captures
        are not comparable (#413 review Minor 1). Either way the capture must
        have been FIRST received (`freshness.observed_at`) at most
        `pointing.FRESH_OBSERVED_S` before this look started: a sim that froze
        during the plan while its bridge re-sends the last render looks new
        otherwise (#413 review Minor 2). Not seen again: refuses. Moved more
        than `pointing.FRESH_MOVE_M` (`fix_moved_m`): raises
        `_PointTargetMoved` with the fresh fix (planned once more), or, when
        this already was that plan, refuses. Nothing was commanded either way;
        a stop that came meanwhile is reported first."""
        from ..perception import freshness
        from . import pointing

        planned_from, since, planned_on, planned_camera = localized
        took = pointing._wall() - planned_from
        if took <= pointing.FRESH_PLAN_S:      # NaN looks again
            return

        def not_seen(why: str) -> SkillError:
            _raise_if_halted_since_start(harness, generation)
            return SkillError(f"cannot point at {label!r}: it was not seen again after a long plan ({took:.1f} s): "
                              f"{why}; point_at moves only on a detection newer than the plan, never toward where "
                              f"it was before; nothing was moved")

        looking = time.monotonic()
        try:
            frame, fresh = self._localize(label, memory=False)
        except SkillError as exc:
            raise not_seen(f"{type(exc).__name__}: {exc}") from None
        camera = self._localized_camera()
        seen_again = (pointing._wall(), time.monotonic(), frame, camera)
        received = float(getattr(frame, "t", math.nan))
        # never an image from before the plan started, however close to this
        # look: `took` reads `pointing._wall`, which need not be this clock;
        # NaN, or no image: not newer
        if not (received > since and received > looking - pointing.FRESH_IN_FLIGHT_S):
            raise not_seen("the image it was found on again was taken before the plan started" if not received > since
                           else "the image it was found on again was received during the plan, before this look "
                                "started")
        # producer clocks compare within one camera; an unknown camera (the
        # VLM's answer) compares as before
        one_camera = camera is None or planned_camera is None or camera is planned_camera
        if one_camera and (getattr(frame, "capture", None) is not None
                           or getattr(planned_on, "capture", None) is not None):
            try:
                newer = freshness.newer_capture(frame, planned_on)
            except ValueError as exc:
                raise not_seen(f"its capture cannot be compared with the one the plan was made from ({exc})") from None
            if not newer:
                raise not_seen("its camera re-sent a capture no newer than the one the plan was made from")
        first_received = freshness.observed_at(frame)
        if not first_received >= looking - pointing.FRESH_OBSERVED_S:
            raise not_seen(f"its camera re-sent a capture it first received {looking - first_received:.1f} s before "
                           f"this look started (more than {pointing.FRESH_OBSERVED_S:g} s: its renders stalled)")
        fresh = _host_fix(fresh)
        moved = pointing.fix_moved_m(fix, fresh)
        if moved <= pointing.FRESH_MOVE_M:     # NaN counts as moved
            return
        _raise_if_halted_since_start(harness, generation)
        if replanned:
            raise SkillError(f"cannot point at {label!r}: it moved {100 * moved:.1f} cm again while point_at "
                             f"planned once more ({took:.1f} s); nothing was moved")
        raise _PointTargetMoved(fresh, seen_again)

    def _pointing_top(self, label: str, fix, harness) -> tuple[float, list, list]:
        """point_at's object top, computed once, the objects standing over
        the target ((xy footprint, top) each) and every other known object
        ((xy footprint, top, name) each, `_belief_footprint`).

        The observed top (the fix's highest point; not finite -> refuse),
        raised by the occupancy map's column top as the pregrasp reads it
        (`_map_top_lift`), then to the top of any OTHER remembered object
        whose footprint covers the target's centre and whose top is higher:
        the booth's "point at where it was" points at the cup that now hides
        the cube, and the cup is part of what the jaws must clear. The target's
        own belief and the object in the jaws are neither. A remembered object
        with no size (`_unsized`) is a neighbour taken conservatively (named so
        in a refusal), never an occluder; one with the target's name whose
        place lies inside the target's column is the target's own stale record
        and is skipped. A remembered object whose box is not finite refuses
        (nothing was moved): point_at cannot tell whether it stands over the
        target or in the way."""
        from .pointing import _footprint_distance, _hull_xy, object_column

        pts = np.asarray(fix.points, dtype=float).reshape(-1, 3)
        observed = float(pts[:, 2].max()) if len(pts) else float("nan")
        if not math.isfinite(observed):
            raise SkillError(f"cannot point at {label!r}: its observed top is not a finite height "
                             f"({observed} m over {len(pts)} points); nothing was moved")
        top = observed + _map_top_lift(self.cfg.grasp, harness, fix, observed)
        table_z = float(getattr(getattr(harness, "limits", None), "table_z", self.cfg.safety.get("table_z", 0.0)))
        target = self.beliefs.find(label)
        # only an object in THIS arm's jaws rides along; one in another arm's
        # jaws is a neighbour like any other (#400 re-review Minor 1)
        in_jaws = (self.beliefs.find(self.held_object)
                   if self.held_object and not _hold_on_another_arm(self) else None)
        centre = np.asarray(fix.position, dtype=float)[:2]
        occluders, neighbours = [], []
        for b in self.beliefs.all():
            if b is target or b is in_jaws:
                continue
            name = f"the {b.color + ' ' if b.color else ''}{b.label}"
            try:
                # beliefs are table-frame memory; the fix and the checks are in
                # the running arm's frame (spec D4; level mounts: point_at is
                # refused on a tilted one)
                arm_b = self._belief_to_arm(b)
                poly, b_top = _belief_footprint(arm_b, table_z)
                if _unsized(b):
                    # the target's own stale record (push_object / place_at
                    # write the place only) is no other object
                    if target is not None and (b.label, b.color) == (target.label, target.color) and \
                            not float(object_column(fix, table_z, top).distance(arm_b.position)[0]) > 0.0:
                        continue
                    # never an occluder: a guess must not move the target's top
                    neighbours.append((_hull_xy(poly), b_top, f"{name} (no remembered size: taken as a "
                                       f"{200 * UNSIZED_HALF_M:.0f} x {200 * UNSIZED_HALF_M:.0f} cm column up to "
                                       f"{self._height_to_world(b_top):.2f} m)"))
                    continue
            except ValueError as exc:
                raise SkillError(f"cannot point at {label!r}: {exc}; nothing was moved") from None
            hull = _hull_xy(poly)
            covers = len(hull) >= 3 and float(_footprint_distance(centre[None, :], hull)[0]) <= 0.0
            if covers and b_top > top:
                occluders.append((hull, b_top))
            else:
                neighbours.append((hull, b_top, name))
        if occluders:
            top = max(top, max(t for _, t in occluders))
        return top, occluders, neighbours

    def _wave_waypoints(self, cycles=2) -> list[np.ndarray]:
        """The joint poses `skill_wave` streams, in order: home, the base +
        wrist wags around it, home. Shared with the people guard, which vets
        each one's TCP before the first command."""
        home = self._profile_q("home_q", "wave")
        cycles = int(np.clip(cycles, 1, 4))
        wrist = self._gesture_joint
        out = [home]
        for side in [+1, -1] * cycles:
            q = home.copy()
            q[0] += 0.25 * side
            q[wrist] += 0.30 * side
            out.append(q)
        out.append(home)
        return out

    def skill_wave(self, cycles: int = 2) -> dict:
        """Greeting gesture: wag the base + wrist around home. Every
        waypoint still goes through the safety harness."""
        poses = self._wave_waypoints(cycles)
        if not self._move_joints(poses[0], duration_s=2.0):
            raise SkillError("could not reach home to wave")
        cycles = int(np.clip(cycles, 1, 4))
        for q in poses[1:-1]:
            self._move_joints(q, duration_s=0.7)
        self._move_joints(poses[-1], duration_s=0.8)
        self.memory.add("action", "waved at the audience")
        return {"waved": True, "cycles": cycles}

    def skill_handover(self, label: str | None = None) -> dict:
        """Hand the object to the human: grasp it if needed, present it at
        the handover pose, and KEEP HOLDING -- the human says 'open gripper'
        (or the agent calls open_gripper) once they have grabbed it."""
        self._check_holding_arm()      # D9 / R5: never present another arm's object with these jaws
        if not self.held_object:
            if not label:
                raise SkillError("not holding anything; say which object to hand over")
            if getattr(self, "_held_provisional", None) is not None:
                # An interrupted grasp's hold, judged as the grasp would judge
                # it and refused here while it survives (unread jaws): the
                # refusal itself is the error, whole within the agent's 240
                # characters, never wrapped in this skill's prefix (#379
                # review minor 2). Another arm's hold names that arm (R5).
                self._refuse_on_another_arms_provisional("try again")
                self._reconcile_held()
                self._raise_on_a_possible_hold()
            res = self._grasp_with_persistence(label)
            if not res.get("ok", True) or not res.get("held"):
                why = str(res.get("error"))
                out = {
                    "ok": False,
                    # A long label is cut so the loop's error (its stop last)
                    # stays whole in the agent's failure line.
                    "error": _fit_label("could not grasp {} for handover: ", repr(label), keep=len(why)) + why,
                    "grasp_attempts": res.get("grasp_attempts"),
                }
                if _left_in_place(res):
                    out["home_skipped"] = True
                    _with_move_home_first(out, source=res)
                if res.get("outcome") == OUTCOME_STUCK:
                    # the persistence loop asked the human; pass the ask through
                    out["outcome"] = OUTCOME_STUCK
                    out["ask"] = res.get("ask")
                return out
        hand_q = self._profile_q("handover_q", "present the object to a human")
        if not self._move_joints(hand_q, duration_s=2.0):
            raise SkillError("did not settle at the handover pose")
        self.memory.add("action", f"offering {self.held_object!r} to the human")
        return {
            "offering": self.held_object,
            "note": "holding steady; call open_gripper once the human has it",
        }

    #: base yaw that aims each throw direction (`safety/people_guard.SIDES`)
    _THROW_YAWS = {"forward": 0.0, "left": np.pi / 2, "right": -np.pi / 2, "back": np.pi}

    def _throw_keyframes(self, direction: str = "forward") -> tuple[np.ndarray, np.ndarray]:
        """(wind-up, release) joint poses of a throw toward `direction`.
        Joint-space keyframes keep this reachable on the B601-RS wrist
        envelope (top-down IK is limited above z~0.15): base yaw (j1) aims
        the throw; j2/j3 load the swing. Shared with the people guard."""
        if direction not in self._THROW_YAWS:
            raise SkillError(f"direction must be one of {sorted(self._THROW_YAWS)}")
        yaw = self._THROW_YAWS[direction]
        base = self._profile_q("home_q", "throw")
        windup = base.copy()
        windup[0] = yaw                     # aim
        windup[1] = base[1] - 0.5           # shoulder back/down (loaded)
        windup[2] = base[2] + 0.4           # elbow tucked
        release = base.copy()
        release[0] = yaw                    # same aim
        release[1] = base[1] + 0.6          # shoulder swings up/forward
        release[2] = base[2] - 0.3          # elbow extends
        return windup, release

    def skill_throw(self, label: str | None = None, direction: str = "forward") -> dict:
        """Booth crowd-pleaser: grasp the object (if not already held), wind
        the arm up, swing it toward `direction`, and RELEASE at the top of
        the swing so the object is launched. The safety harness still vets
        every waypoint (velocity cap, workspace AABB, table clearance), so
        this is a *gestural* throw -- arm swing speed + release timing give
        the toss, never an unsafe joint velocity. Great for "grab the banana
        and throw it". The people guard refuses it toward any side listed in
        `safety.people.visitor_sides` (every side unless a profile narrows
        it): nothing vets where the object lands.
        """
        self._check_holding_arm()      # D9 / R5: never swing and open these jaws for another arm's object
        if direction not in self._THROW_YAWS:
            raise SkillError(f"direction must be one of {sorted(self._THROW_YAWS)}")
        # 1) make sure we're holding something.
        if not self.held_object:
            if not label:
                raise SkillError("not holding anything; say which object to throw")
            res = self.skill_grasp_object(label)
            if not res.get("ok", True) or not res.get("held"):
                return {
                    "ok": False,
                    "error": f"could not grasp {label!r} to throw: {res.get('error')}",
                }
        thrown = self.held_object

        # 2) wind-up pose: arm drawn back and low, jaws still closed.
        windup, release = self._throw_keyframes(direction)

        # 3) execute: settle at wind-up, then swing FAST (short duration --
        #    the harness auto-stretches it only if it would breach the cap,
        #    so we get the quickest *safe* swing) and pop the jaws open at
        #    the peak. Releasing while the wrist is still moving imparts the
        #    launch impulse.
        if not self._move_joints(windup, duration_s=1.5):
            raise SkillError("did not settle at the wind-up pose")
        self.memory.add("action", f"winding up to throw {thrown!r} {direction}")
        # kick off the swing in a background stream and release mid-arc.
        self._move_joints(release, duration_s=0.6)
        self._set_gripper(self._grip_open, effort=1.0)  # let it fly
        self._gripper_opens_completed += 1
        released_obj = self.held_object
        self.held_object = None
        self._held_arm = None
        self._held_det_label = None
        self._held_color = None
        self._held_width_m = None
        # follow-through, then home so the camera view clears.
        try:
            self.skill_move_home()
        except (SkillError, SafetyViolation):
            pass
        self.memory.add("action", f"threw {released_obj!r} {direction}")
        return {
            "ok": True,
            "thrown": released_obj,
            "direction": direction,
            "note": "gestural throw: harness-vetted swing + timed release",
        }

    # ── fastening ─────────────────────────────────────────────────────────

    @property
    def _screw_joint(self) -> int:
        """Joint that spins the tool about its approach axis, as an index
        into q. Profile key `screw_joint` wins; the default is the LAST arm
        joint, which is the wrist roll on every shipped chain (SO-101 j4,
        reBot j5, Panda j6) -- and on the SO-101 it is MEASURED collinear
        with the tool approach (rolling it turns the approach by 0.00 deg),
        which is exactly the property a screwing motion needs."""
        n = int(self.cfg.arm.get("n_joints", 6))
        idx = self.cfg.arm.get("screw_joint")
        return int(idx) if idx is not None else max(0, n - 1)

    def skill_turn_screw(
        self,
        label: str,
        direction: str = "tighten",
        turns: float = 1.0,
    ) -> dict:
        """Tighten or loosen a screw/bolt/nut by ratcheting the wrist roll.

        A parallel jaw cannot spin continuously, so this works like a human
        with a stubby screwdriver: engage (close the jaws on the head/nut),
        turn the wrist roll through its free travel, disengage (open),
        counter-rotate back, and repeat until the requested turns are done.
        Every stroke is a normal harness-vetted joint motion; the gripper
        close reuses the standard two-stage grasp so a fragile plastic
        thumbscrew is not crushed.

        Geometric/gestural like `throw`: it commands kinematics, not torque
        -- real cam-out/stall detection needs joint-effort feedback that
        only some backends report. `tighten` turns clockwise seen from
        above (right-hand thread); the profile's `screw.tighten_sign` flips
        it for arms whose roll axis is authored anti-parallel to the
        approach.
        """
        directions = {"tighten": +1.0, "loosen": -1.0}
        if direction not in directions:
            raise SkillError(f"direction must be one of {sorted(directions)}")
        try:
            requested_turns = float(turns)
        except (TypeError, ValueError):
            raise SkillError("turns must be a finite number in [0.05, 6]") from None
        if isinstance(turns, bool) or not np.isfinite(requested_turns) or not 0.05 <= requested_turns <= 6:
            raise SkillError("turns must be a finite number in [0.05, 6]")
        turns = requested_turns
        scfg = self.cfg.get("screw", {})
        sign = directions[direction] * float(
            (scfg.get("tighten_sign") if scfg else None) or 1.0
        )
        joint = self._screw_joint
        if not 0 <= joint < int(self.cfg.arm.get("n_joints", 6)):
            raise SkillError("screw_joint is outside the configured arm joints")
        # The first jaw command opens the jaws at home: never on an object
        # the jaws hold, or may hold after an interrupted grasp.
        self._reconcile_held()
        if self.held_object:
            raise SkillError(self._already_holding_refusal("turn_screw"))   # names another arm's hold (R5)
        self._refuse_on_another_arms_provisional("try again")      # R5: its close would take that slot
        self._raise_on_a_possible_hold()

        _, fix = self._localize(label)
        gcfg = self.cfg.grasp
        top_z = float(fix.points[:, 2].max())
        z_max = float(gcfg.get("topdown_z_max", 0.15)) - 0.01
        engage = fix.position.copy()
        # Engage AT the head: descend to the object's top, minus a small
        # bite so the jaws wrap the head instead of pinching its crown.
        engage[2] = min(max(top_z - float(scfg.get("bite_m", 0.005) if scfg else 0.005),
                            float(gcfg.get("min_grasp_z_m", 0.008))), z_max)
        hover = engage.copy()
        hover[2] = min(engage[2] + float(gcfg.get("pregrasp_offset_m", 0.04)), z_max)

        from ..grasping.obb_grasp import _yaw_rotation

        # The roll joint provides the spin, so the TCP orientation only has
        # to put the approach vertical; seed from home like every grasp.
        home = self._profile_q("home_q", "turn a screw")
        if not self._move_joints(home, duration_s=2.0):
            raise SkillError("could not reach home to start the screw motion")
        self._set_gripper(self._grip_open, effort=0.8)
        yaw0 = float(np.arctan2(engage[1], engage[0]))
        ik_hover = None
        engage_yaw = yaw0
        for yaw in (yaw0, 0.0, np.pi / 4, -np.pi / 4):
            cand = self.kin.ik(
                make_transform(_yaw_rotation(yaw, axis_order=self._tool_axis_order),
                               hover), home)
            if cand.success:
                ik_hover = cand
                engage_yaw = yaw
                break
        if ik_hover is None:
            raise SkillError(f"cannot reach a pose above {label!r} to work the screw")
        if not self._move_joints(ik_hover.q, duration_s=2.0):
            raise SkillError("did not settle above the screw")

        # Stroke span: the roll joint's free travel inside its limits, split
        # around the engage pose, capped per stroke for control.
        lo, hi = self.kin.joint_limits
        margin = 0.08
        q_engage = ik_hover.q.copy()
        stroke_max = float(scfg.get("stroke_rad", 1.2) if scfg else 1.2)
        want = turns * 2.0 * np.pi
        done = 0.0
        strokes = 0
        max_strokes = int(scfg.get("max_strokes", 24) if scfg else 24)
        profile = select_profile(label, None)

        ik_engage = self.kin.ik(
            make_transform(_yaw_rotation(engage_yaw, axis_order=self._tool_axis_order),
                           engage), ik_hover.q)
        if not ik_engage.success:
            raise SkillError(f"cannot reach the engagement pose of {label!r}; no screw stroke sent")
        q_engage = ik_engage.q.copy()

        while done < want and strokes < max_strokes:
            room_fwd = (hi[joint] - margin - q_engage[joint]) if sign > 0 else (
                q_engage[joint] - (lo[joint] + margin))
            start_back = min(stroke_max, want - done)
            room_back = (q_engage[joint] - (lo[joint] + margin)) if sign > 0 else (
                hi[joint] - margin - q_engage[joint])
            wind = min(start_back, room_back + room_fwd)
            if wind <= 0.02:
                break
            # 1) jaws open, wind the roll back to the stroke start
            q_start = q_engage.copy()
            q_start[joint] -= sign * min(start_back, room_back)
            self._set_gripper(self._grip_open, effort=0.8)
            if not self._move_joints(q_start, duration_s=1.0):
                raise SkillError("did not settle at the stroke start")
            # 2) engage the head (two-stage close, force from the material)
            self._close_two_stage(profile)
            # 3) the working stroke
            q_end = q_start.copy()
            stroke = min(wind, stroke_max,
                         (hi[joint] - margin - q_start[joint]) if sign > 0
                         else (q_start[joint] - (lo[joint] + margin)))
            if stroke <= 0.02:
                self._set_gripper(self._grip_open, effort=0.8)
                break
            q_end[joint] += sign * stroke
            if not self._move_joints(q_end, duration_s=max(0.8, stroke / 1.5)):
                unsettled = "stroke did not settle; stopping the screw task"
                try:  # disengage: a halt here carries the stroke's failure
                    self._set_gripper(self._grip_open, effort=0.8)
                except MotionHalted as halted:
                    _keep_decided_failure(halted, f"SkillError: {unsettled}")
                    raise
                raise SkillError(unsettled)
            # 4) disengage
            self._set_gripper(self._grip_open, effort=0.8)
            done += stroke
            strokes += 1

        applied = done / (2.0 * np.pi)
        # The shortfall is decided here, before the retreat: a halt in the
        # retreat carries it (`_keep_decided_failure`), so the call is filed
        # as the shortfall, not as the halt.
        shortfall = None
        if applied <= 0.0:
            shortfall = (f"no roll travel available to {direction} {label!r} "
                         f"(joint {joint} pinned by its limits at this pose)")
        elif done < want - 1e-6:
            shortfall = f"only {applied:.2f} of {turns:.2f} requested wrist turns completed"
        # retreat to hover, then home so the camera view clears
        try:
            self._move_joints(ik_hover.q, duration_s=1.0)
        except MotionHalted as halted:
            _keep_decided_failure(halted, shortfall and f"SkillError: {shortfall}")
            raise
        try:
            self.skill_move_home()
        except (SkillError, SafetyViolation):
            pass
        self.memory.add(
            "action",
            f"commanded {direction} wrist travel for {label!r}: {applied:.2f} turns "
            f"({strokes} strokes); fastener outcome not measured",
        )
        if shortfall is not None:
            raise SkillError(shortfall)
        return {
            "screw": label,
            "direction": direction,
            "turns_requested": round(turns, 2),
            "turns_applied": round(applied, 2),
            "turns_commanded": round(applied, 2),
            "strokes": strokes,
            "physical_verification": {"status": "unverified", "fastener_turns": None,
                                      "axial_advance_m": None, "seating_verified": False},
            "note": "harness-vetted wrist strokes; turns_applied is commanded travel, "
                    "not measured fastener rotation or tightening torque",
        }

    def skill_sort_by_color(self, max_objects: int = 6) -> dict:
        """Crowd-pleaser: group everything on the table into per-color zones
        along the front edge. Pure composition of pick_and_place."""
        self._refuse_on_another_arms_provisional("sort_by_color")       # R5: before any object
        # +-0.24 is outside the top-down IK envelope at x=0.30 on this arm
        # (verified against the RS URDF); stay within +-0.20. The zones run
        # along the RUNNING arm's front edge (its base frame, its drop_zone x);
        # they are placed at and reported in the table frame.
        zones_y = (-0.20, -0.10, 0.0, 0.10, 0.20)
        zone_x = float(self.cfg.grasp.get("drop_zone", [0.30, -0.20])[0])
        colors: dict[str, tuple[float, float]] = {}
        moved, failed = [], []
        left_in_place, left_by = False, None  # the last grasp attempt left the arm where it stopped
        grasping = False  # a grasp was started (the possible-hold guard ran)
        for b in list(self.beliefs.all())[: int(max_objects)]:
            color = b.color or "unknown"
            if color not in colors:
                if len(colors) >= len(zones_y):
                    failed.append({"label": b.label, "error": "no free color zone"})
                    continue
                colors[color] = (zone_x, zones_y[len(colors)])
            zx, zy = colors[color]
            if np.linalg.norm(np.asarray(self._to_arm(b.position))[:2] - np.array([zx, zy])) < 0.07:
                continue  # already sorted
            query = f"{b.color} {b.label}" if b.color else b.label
            if not grasping:
                grasping = True
                self._refuse_a_possible_hold_before_grasping()   # every grasp would open the jaws on it
            try:
                # a per-object slice of the budget: one stubborn object must
                # not eat the whole sort
                res = self._grasp_with_persistence(
                    query, budget_s=float(self.cfg.grasp.get("sort_object_persist_seconds", 40.0)))
            except MotionHalted:
                raise  # the operator's halt ends the sort, not just this object
            except (SkillError, SafetyViolation) as e:
                failed.append({"label": query, "error": str(e)})
                if self._estopped():
                    break  # the operator stopped the arm: no next object
                continue
            left_in_place, left_by = _left_in_place(res), res
            if not res.get("ok", True) or not res.get("held"):
                entry = {"label": query, "error": str(res.get("error", "?"))}
                if res.get("outcome") == OUTCOME_STUCK:
                    entry["ask"] = str(res.get("ask") or "")
                failed.append(entry)
                if self._estopped():
                    break  # the operator stopped the arm: no next object
                continue
            zone = self._xy_to_world((zx, zy))     # place_at takes table-frame coordinates
            try:
                self.skill_place_at(zone[0], zone[1])
            except (SkillError, SafetyViolation) as e:
                # still holding: stop sorting rather than cascade failures
                failed.append({"label": query, "error": f"place: {e}"})
                break
            moved.append({"label": query, "color": color, "zone": [zone[0], zone[1]]})
        ok = not failed or bool(moved)
        out = {
            "ok": ok,
            "moved": moved,
            "failed": failed,
            "zones": {c: list(self._xy_to_world(z)) for c, z in colors.items()},
        }
        if not ok:
            out["error"] = "; ".join(
                f"{f['label']}: {f['error'][:60]}" for f in failed[:3]
            ) or "nothing to sort"
            asks = [f["ask"] for f in failed if f.get("ask")]
            if asks:
                # nothing moved and at least one object needs the human:
                # the sort is stuck, every per-object ask relayed verbatim
                out["outcome"] = OUTCOME_STUCK
                out["ask"] = " ".join(asks)
        if left_in_place:
            out["home_skipped"] = True
            _with_move_home_first(out, source=left_by)
        return out

    # ── Pigey scene memory: composites over the belief store ────────────
    #
    # github.com/lianegalanti/Pigey `real/agent-system.md` keeps "Original
    # positions (memorize)" in the planner's reasoning text and restores via
    # absolute coordinates, blocker-first; its occlusion search lifts the
    # largest hollow occluder, parks it ~0.2 m away in free table, re-perceives
    # and then resumes the ORIGINAL task. Here the same protocol lives in
    # three skills so a chat host gets it as one tool call each. All motion
    # is `_grasp_with_persistence` + `skill_place_at`, exactly what
    # `sort_by_color` composes: every waypoint is vetted by the harness, the
    # per-move verdict is `place_at`'s own postcondition, and the snapshot is
    # ADVISORY -- restored positions are targets, never asserted as achieved
    # without the composite postcondition (`agent/effects.py::_check_restored`).

    @staticmethod
    def _belief_query(b) -> str:
        """The phrase `_localize` resolves back to THIS belief: colour +
        label, without doubling a colour the label already names
        ("blue cube", not "blue blue cube")."""
        label = str(b.label)
        if b.color and str(b.color).lower() not in label.lower().split():
            return f"{b.color} {label}"
        return label

    @staticmethod
    def _footprint_m(b_or_entry) -> float:
        """Largest horizontal-ish dimension of a belief or snapshot entry
        (extents are eigenvalue-sorted, so [0] is the long side); 5 cm when
        unknown -- a chunky prop, the conservative guess for clearance."""
        ext = b_or_entry.get("extent") if isinstance(b_or_entry, dict) else getattr(b_or_entry, "extent", None)
        if ext is None:
            return 0.05
        try:
            return float(np.max(np.abs(np.asarray(ext, dtype=float))))
        except (TypeError, ValueError):
            return 0.05

    def _workspace_contains_xy(self, xy, margin_m: float = 0.0) -> bool:
        bounds = self._localization_workspace_bounds()
        if bounds is None:
            return True
        p = np.asarray(xy, dtype=float).reshape(-1)[:2]
        lo, hi = bounds[0][:2] + margin_m, bounds[1][:2] - margin_m
        return bool(np.all(np.isfinite(p)) and np.all(p >= lo) and np.all(p <= hi))

    def _reachable_place_xy(self, x: float, y: float) -> bool:
        """IK pre-check for a top-down release above (x, y), in the running
        arm's base frame.

        Only a filter: `place_at`'s own planner and the harness remain the
        authority over the actual motion. Its job is to avoid grasping an
        object whose target no top-down pose can reach -- on the SO-101 the
        workspace box admits points the arm cannot solve (the usable region is
        an annulus), and a grasp followed by "place pose unreachable" leaves
        the object in the jaws. Seeded from `home_q`, like the grasp IK.
        """
        kin = getattr(self, "kin", None)
        if kin is None:
            return True  # bare runtimes/tests: nothing to pre-check with
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        z = table_z + float(gcfg.get("release_height_m", 0.05))
        z = min(z, float(gcfg.get("topdown_carry_z_max", gcfg.get("topdown_z_max", 0.15))) - 0.005)
        q_seed = self._profile_q("home_q", "pre-check a place pose")
        from ..grasping.obb_grasp import _yaw_rotation

        for yaw in (float(np.arctan2(y, x)), 0.0, np.pi / 4, -np.pi / 4):
            try:
                ik = kin.ik(make_transform(_yaw_rotation(yaw, axis_order=self._tool_axis_order),
                                           [float(x), float(y), z]), q_seed)
            except Exception:  # noqa: BLE001 -- a solver fault is "not reachable", not a crash
                continue
            if ik.success:
                return True
        return False

    def _free_park_spot(self, around_xy, avoid, offsets_m=(0.20, 0.15, 0.12)):
        """A spot `offset` away from `around_xy` that is inside the workspace,
        clear of every (xy, radius) in `avoid`, and IK-reachable; None when
        the table offers none. Pigey parks occluders "+0.2 m" into free
        table; shorter offsets are fallbacks for a small workspace, never a
        substitute for the clearance test.

        `around_xy`, `avoid` and the spot are table-frame xy (beliefs, the
        snapshot, place_at's coordinates); the search runs in the running
        arm's base frame: its workspace box, its IK and the order of its
        directions are the arm's, so an arm at a base_pose parks where the same
        arm without one would, moved with it."""
        T = self.T_world_base
        if T is not None:
            around_xy = self._xy_to_arm(np.asarray(around_xy, dtype=float).reshape(-1)[:2])
            avoid = [(self._xy_to_arm(np.asarray(a, dtype=float).reshape(-1)[:2]), r) for a, r in avoid]
        around = np.asarray(around_xy, dtype=float).reshape(-1)[:2]
        r2 = 1.0 / np.sqrt(2.0)
        directions = ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0),
                      (r2, r2), (r2, -r2), (-r2, r2), (-r2, -r2))
        for off in offsets_m:
            for d in directions:
                p = around + float(off) * np.asarray(d)
                if not self._workspace_contains_xy(p, margin_m=0.03):
                    continue
                if any(float(np.linalg.norm(p - np.asarray(a, dtype=float)[:2])) < float(r)
                       for a, r in avoid):
                    continue
                if not self._reachable_place_xy(float(p[0]), float(p[1])):
                    continue
                return p if T is None else np.asarray(self._xy_to_world(p), dtype=float)
        return None

    def _task_budget_exhausted(self) -> bool:
        deadline = getattr(self, "_task_deadline", None)
        return deadline is not None and time.monotonic() >= deadline

    #: Camera frames come out of a buffered stream, so a belief refreshed by
    #: a fresh look carries the FRAME's capture time, which can precede the
    #: moment the look began by a frame interval or two. A belief counts as
    #: confirmed by the look when it was seen after this much before it.
    _FRESH_LOOK_SLACK_S = 2.0

    def _fresh_look(self, frames: int = 1) -> float:
        """Re-observe by hand (the watcher is paused during motion skills)
        and return the timestamp floor: beliefs last seen at or after it were
        confirmed by THIS look; older ones are remembered, not seen. The
        composites act only on what the camera confirms now -- a remembered
        position informs the agent, it never aims the jaws."""
        floor = time.monotonic() - self._FRESH_LOOK_SLACK_S
        self._reobserve(frames=frames)
        return floor

    def _estopped(self) -> bool:
        """The latched e-stop, read without materializing a LazyArm (the
        harness is the SafeArm's, not the backend's)."""
        harness = getattr(self.arm, "harness", None)
        return bool(harness is not None and getattr(harness, "estopped", False))

    def _move_object_to(self, query: str, target_xy, *, budget_s: float, kind: str) -> dict:
        """One composite move: grasp `query` with the normal persistence loop,
        place it at `target_xy` through `skill_place_at`, verify with
        `place_at`'s own postcondition, clear the view.

        Returns the move record; a failed grasp is reported in it (the loop
        may go on), a failed place sets `holding` (the caller must stop:
        something is in the jaws) and `stop` marks a release/retreat failure
        after which no further motion is safe.

        An operator halt that ends the grasp's persistence loop (raised in a
        grasp attempt, its re-home or its re-scan) or a clear-view home here
        raises MotionHalted, so the caller's loop ends too; after a failed
        grasp, the halt in its clear-view home carries that failure
        (`skill_error`, filed as it). A halt in the
        place call comes back as the place's failure in the record
        (`holding` names what the jaws still hold; the caller stops on it).
        Under execute() the skill's halt baseline refuses every later motion
        of the call either way, so a caller that goes on gets MotionHalted
        from its next grasp before any command.
        """
        x, y = float(target_xy[0]), float(target_xy[1])
        rec: dict = {"label": query, "kind": kind, "to": [round(x, 3), round(y, 3)], "ok": False}
        before = self.effects.snapshot(query) if self.effects is not None else {}
        try:
            res = self._grasp_with_persistence(query, budget_s=budget_s)
        except MotionHalted:
            raise  # the operator's halt ends the caller's loop too: no clear-view home
        except (SkillError, SafetyViolation) as e:
            res = {"ok": False, "held": False, "error": f"{type(e).__name__}: {e}"}
        rec["grasp_attempts"] = res.get("grasp_attempts")
        if not res.get("ok", True) or not res.get("held"):
            rec["error"] = f"grasp: {res.get('error', 'grasp failed')}"
            if res.get("home_skipped"):
                if _left_in_place(res):
                    rec["home_skipped"] = True  # the arm stays where the attempt stopped
                    if _withdrew_above_object(res):
                        rec["arm_withdrew_above_object"] = True
                    if _jaws_left_closed(res):
                        rec["air_grasp_jaws_closed"] = True
            elif not self._estopped():  # the operator stopped the arm: no re-home (A3 review)
                try:  # never leave the arm over the table after a failed attempt
                    self.skill_move_home()
                except MotionHalted as halted:
                    # this failed grasp's own re-home: the halt carries the failure
                    _keep_decided_failure(halted, res.get("error") or rec["error"])
                    raise  # the caller's next step would clear the halt and move
                except (SkillError, SafetyViolation):
                    pass
            return rec
        if self.effects is not None:
            # The place's own jaw width, holding state and actuation counts
            # (place_at's release rule), taken now: the grasp-time snapshot
            # above predates the close (nothing held, no width).
            before = {**before, **self.effects.snapshot(None, "place_at")}
        try:
            place = self.skill_place_at(x, y)
        except (SkillError, SafetyViolation) as e:
            rec["error"] = f"place: {e}"
            rec["holding"] = self.held_object
            return rec
        rec["placed_at"] = place.get("at")
        if self.effects is not None:
            pc = self.effects.verify("place_at", {"x": x, "y": y}, place, before=before)
            if pc is not None:
                rec["postcondition"] = pc.as_dict()  # inherited from the primitive
        if place.get("ok") is False:
            rec["error"] = str(place.get("error", "place failed"))
            rec["stop"] = True  # released, but the retreat/release failed: no sweep home
            rec["holding"] = self.held_object
            return rec
        rec["ok"] = True
        if not place.get("home_skipped"):
            try:  # clear the camera view for the re-scan that follows
                self.skill_move_home()
            except MotionHalted:
                raise  # the caller's next step would clear the halt and move
            except (SkillError, SafetyViolation):
                pass
        return rec

    def skill_snapshot_scene(self, name: str = "scene") -> dict:
        """Memorize the layout: the CONFIRMED objects' labels, colours and
        centroids under a name, in the belief store. No motion. Advisory
        data: the positions become `restore_scene`'s targets and nothing
        here claims where an object is afterwards."""
        name = str(name or "scene").strip().lower() or "scene"
        # A fresh pass first, so "confirmed" means seen now, not remembered.
        self._fresh_look(frames=1)
        held = self._held_det_label if self.held_object else None
        snap = self.beliefs.snapshot(name, exclude_label=held)
        self.memory.add("note", f"memorized scene {name!r}: {len(snap.entries)} object(s)")
        out = {"snapshot": name, **snap.as_dict()}
        out.pop("name", None)
        out["held_excluded"] = self.held_object
        out["note"] = (
            "Advisory layout memory: these positions are the targets restore_scene will "
            "aim for, not a claim about where objects are now. Nothing moved."
        )
        return out

    def _restore_plan(self, snap, tolerance_m: float, skip: set[str], seen_after: float | None = None) -> dict:
        """Diff the snapshot against the beliefs CONFIRMED by the last look.

        Identity is colour-first (Pigey: "match by visual identity (color,
        shape) ... labels drift"), then label/alias, nearest first; each
        belief answers for at most one snapshot entry. Only beliefs seen at
        or after `seen_after` take part: a remembered position (the drop point
        of a pick made before the world changed) would otherwise match its
        own snapshot entry and hide the displacement, or block a target
        nothing occupies. Returns displaced / unchanged / missing and the
        blocker-first `next` step."""
        beliefs = [b for b in self.beliefs.all()
                   if seen_after is None or float(b.last_seen_t) >= seen_after]
        pairs = []
        for i, e in enumerate(snap.entries):
            e_xy = np.asarray(e["position"], dtype=float)[:2]
            names = {str(e["label"]).lower(), *(str(a).lower() for a in e.get("aliases") or [])}
            for b in beliefs:
                b_names = {str(b.label).lower(), *(str(a).lower() for a in b.aliases)}
                color_known = e.get("color") is not None and b.color is not None
                if color_known and e["color"] != b.color:
                    continue  # two measured colours that differ are two objects
                if not (names & b_names) and not (color_known and e["color"] == b.color):
                    continue
                d = float(np.linalg.norm(np.asarray(b.position, dtype=float)[:2] - e_xy))
                # same label AND same colour beats colour-only, then distance
                pairs.append((0 if names & b_names else 1, d, i, id(b), b))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))
        matched: dict = {}
        taken: set[int] = set()
        for _, _, i, bid, b in pairs:
            if i in matched or bid in taken:
                continue
            matched[i] = b
            taken.add(bid)
        displaced, unchanged, missing = [], [], []
        for i, e in enumerate(snap.entries):
            b = matched.get(i)
            target = [float(v) for v in e["position"][:2]]
            if b is None:
                missing.append(str(e["label"]))
                continue
            cur = np.asarray(b.position, dtype=float)[:2]
            off = float(np.linalg.norm(cur - np.asarray(target)))
            rec = {"label": str(b.label), "query": self._belief_query(b), "color": b.color,
                   "from": [round(float(v), 3) for v in cur], "target": target,
                   "offset_m": round(off, 4), "belief": b, "entry": e}
            (displaced if off > tolerance_m else unchanged).append(rec)
        displaced.sort(key=lambda r: (r["label"], r["color"] or ""))
        plan = {"displaced": displaced, "unchanged": unchanged, "missing": missing, "next": None}
        todo = [r for r in displaced if r["label"] not in skip]
        if not todo:
            return plan

        def blockers(rec):
            tgt = np.asarray(rec["target"], dtype=float)
            out = []
            for b in beliefs:
                if b is rec["belief"]:
                    continue
                radius = max(tolerance_m, 0.5 * (self._footprint_m(b) + self._footprint_m(rec["entry"])) + 0.01)
                if float(np.linalg.norm(np.asarray(b.position, dtype=float)[:2] - tgt)) < radius:
                    out.append(b)
            return out

        free = [r for r in todo if not blockers(r)]
        if free:
            r = free[0]
            plan["next"] = {"query": r["query"], "label": r["label"], "to": r["target"], "kind": "restore"}
            return plan
        # Every outstanding target is occupied: a swap cycle, or a foreign
        # object on a remembered spot. Park the first blocker on free table,
        # clear of everything on it and of every remembered position. A
        # blocker that already failed to move (`skip`) is not retried: the
        # plan reports it and the caller stops instead of burning the budget.
        r = todo[0]
        movable = [b for b in blockers(r) if str(b.label) not in skip]
        if not movable:
            plan["blocked_by"] = str(blockers(r)[0].label)
            return plan
        blk = movable[0]
        avoid = [(np.asarray(b.position, dtype=float)[:2],
                  0.5 * (self._footprint_m(b) + self._footprint_m(blk)) + 0.02)
                 for b in beliefs if b is not blk]
        avoid += [(np.asarray(e["position"], dtype=float)[:2],
                   max(tolerance_m, 0.5 * (self._footprint_m(e) + self._footprint_m(blk)) + 0.02))
                  for e in snap.entries]
        spot = self._free_park_spot(np.asarray(blk.position, dtype=float)[:2], avoid)
        if spot is None:
            plan["blocked_by"] = str(blk.label)
            return plan
        plan["next"] = {"query": self._belief_query(blk), "label": str(blk.label),
                        "to": [float(spot[0]), float(spot[1])], "kind": "park"}
        return plan

    def skill_restore_scene(self, name: str = "scene", tolerance_m: float = 0.05,
                            max_moves: int = 6) -> dict:
        """Put the table back the way `snapshot_scene` memorized it.

        Only objects displaced beyond `tolerance_m` move; an object sitting on
        another's remembered spot moves first (blocker-first), and a swap
        cycle parks one object on free table before both go home. Each move
        is grasp (with the usual persistence) + place_at through the harness;
        the number of moves and the per-task persistence budget bound it.
        The result lists what moved, what was left alone, what could not be
        found and what failed -- and its postcondition says what an
        independent channel measured, which is the only claim of success.
        """
        self._refuse_on_another_arms_provisional("restore_scene")       # R5: before any work
        name = str(name or "scene").strip().lower() or "scene"
        snap = self.beliefs.get_snapshot(name)
        if snap is None:
            known = self.beliefs.snapshots()
            raise SkillError(
                f"no scene snapshot named {name!r}; call snapshot_scene first"
                + (f" (known: {known})" if known else "")
            )
        carry_attachment.check(self)
        self._reconcile_held()
        if self.held_object:
            raise SkillError(self._already_holding_refusal("restore_scene"))
        tolerance_m = float(np.clip(float(tolerance_m), 0.02, 0.15))
        max_moves = int(np.clip(int(max_moves), 1, 8))
        budget_s = float(self.cfg.grasp.get("restore_object_persist_seconds", 40.0))
        self.memory.add("note", f"restoring scene {name!r} ({len(snap.entries)} remembered objects, "
                                f"tolerance {tolerance_m*100:.0f} cm)")
        seen_after = self._fresh_look()
        moves: list[dict] = []
        failed: list[dict] = []
        skip: set[str] = set()
        error = stage = None
        left_in_place, left_by = False, None  # the last grasp attempt left the arm where it stopped
        plan = self._restore_plan(snap, tolerance_m, skip, seen_after)
        initial_displaced = [{k: v for k, v in r.items() if k not in ("belief", "entry", "query")}
                             for r in plan["displaced"]]
        # what was left alone: within tolerance BEFORE any motion
        left_alone = [{"label": r["label"], "target": r["target"]} for r in plan["unchanged"]]
        while plan["displaced"] and any(r["label"] not in skip for r in plan["displaced"]):
            if len(moves) >= max_moves:
                error = (f"move budget ({max_moves}) exhausted; still displaced: "
                         f"{[r['label'] for r in plan['displaced']]}")
                break
            if self._task_budget_exhausted():
                error = "task persistence budget exhausted (an earlier tier already spent it)"
                break
            if self._estopped():
                error = "e-stop latched; not moving"
                break
            step = plan["next"]
            if step is None:
                error = (f"cannot clear {plan.get('blocked_by', 'the blocker')!r} off a remembered "
                         "spot (no free, reachable park spot, or it already failed to move); "
                         "the human may need to clear the table")
                break
            to = self._xy_to_arm(step["to"])     # a table-frame target; the IK pre-check is the arm's
            if not self._reachable_place_xy(to[0], to[1]):
                failed.append({"label": step["label"], "kind": step["kind"], "to": step["to"],
                               "error": "target not reachable for a top-down place; left where it is"})
                skip.add(step["label"])
                plan = self._restore_plan(snap, tolerance_m, skip, seen_after)
                continue
            if not moves:
                self._refuse_a_possible_hold_before_grasping()   # every move starts with a grasp
            try:
                rec = self._move_object_to(step["query"], step["to"], budget_s=budget_s, kind=step["kind"])
            except MotionHalted as halted:
                # An earlier object that failed already fixed this call's
                # ok=False: a halt in a later object's move carries that
                # failure (`_keep_decided_failure`; a failure the move itself
                # had decided rides on it already and wins).
                _keep_decided_failure(halted, _restore_failures(failed))
                raise
            rec["label"] = step["label"]
            moves.append(rec)
            left_in_place, left_by = bool(rec.get("home_skipped")), rec
            if rec.get("holding"):
                error, stage = rec.get("error", "place failed"), "place"
                break
            if rec.get("stop"):
                error, stage = rec.get("error", "release failed"), "release"
                break
            if not rec["ok"]:
                failed.append({k: rec[k] for k in ("label", "kind", "to", "error") if k in rec})
                skip.add(step["label"])
            seen_after = self._fresh_look()
            plan = self._restore_plan(snap, tolerance_m, skip, seen_after)
        final = self._restore_plan(snap, tolerance_m, set(), seen_after)
        still = [r["label"] for r in final["displaced"]]
        result: dict = {
            "snapshot": name,
            "snapshot_age_s": round(snap.age_s(), 1),
            "tolerance_m": round(tolerance_m, 3),
            "objects_in_snapshot": len(snap.entries),
            "displaced": initial_displaced,
            "moves": moves,
            "unchanged": left_alone,
            "still_displaced": still,
            "missing": final["missing"],
            "failed": failed,
            "note": ("Snapshot positions are advisory targets; the postcondition says what an "
                     "independent channel measured. Report that, not the move count."),
        }
        if error is None and final["missing"]:
            error = (f"{len(final['missing'])} remembered object(s) not visible: {final['missing']}; "
                     "use search_for_object before restoring them")
        if error is None and failed:
            error = _restore_failures(failed)
        if error is None and still:
            error = f"still displaced after the restore: {still}"
        if error is not None:
            result.update(ok=False, error=error)
            from ..memory.envelope import HALT_SIGNATURE, normalize_failure
            if failed and normalize_failure(error) == HALT_SIGNATURE:
                # A halted place came back as the place's error (`holding` /
                # `stop`) after an earlier object had failed: that failure,
                # decided first, files the call (`failure_signature`).
                result["skill_error"] = _restore_failures(failed)
            if stage is not None:
                result["stage"] = stage
            if self.held_object:
                result["holding"] = self.held_object
                result["next_action"] = (
                    f"Still holding {self.held_object!r}: the place to its remembered spot was "
                    "refused by the planner/harness. Put it down with place_at at explicit free "
                    "coordinates (or open_gripper over a safe spot), then retry; do not report "
                    "the scene as restored."
                )
        if left_in_place:
            result["home_skipped"] = True
            _with_move_home_first(result, source=left_by)
        self.memory.add("action", f"restore_scene {name!r}: {len(moves)} move(s), "
                                  f"{len(still)} still displaced, {len(final['missing'])} missing")
        return result

    def _belief_is_target(self, b, query: str) -> bool:
        """Does this belief answer to the search query? Mirrors BeliefStore.find
        loosely: label/alias match, or a colour match with a compatible noun.
        Used to keep the target itself OUT of the occluder list."""
        q = str(query).strip().lower()
        names = {str(b.label).lower(), *(str(a).lower() for a in b.aliases)}
        if q in names:
            return True
        color, noun = parse_color_query(q)
        if noun and any(noun in n or n in noun for n in names):
            return color is None or b.color is None or b.color == color
        return bool(color and noun is None and b.color == color)

    def _seen_since(self, query: str, since: float):
        b = self.beliefs.find(query)
        return b if (b is not None and float(b.last_seen_t) >= since) else None

    def skill_search_for_object(self, label: str, max_occluders: int = 3) -> dict:
        """Occlusion search (Pigey): when the named object is not visible,
        lift the largest hollow/large occluder, park it ~0.2 m away on free
        reachable table inside the workspace, re-perceive; repeat for up to
        `max_occluders`. Finding the object is NOT the task: the result says
        to resume the ORIGINAL task on it (`task_complete: false`), and a
        miss is an honest `stuck` result, never a guess."""
        query = str(label or "").strip()
        if not query:
            raise SkillError("search_for_object needs the object's name")
        self._refuse_on_another_arms_provisional("search_for_object")   # R5: before any work
        carry_attachment.check(self)
        self._reconcile_held()
        if self.held_object:
            raise SkillError(self._already_holding_refusal("search_for_object"))
        max_occluders = int(np.clip(int(max_occluders), 1, 4))
        gcfg = self.cfg.grasp
        budget_s = float(gcfg.get("search_occluder_persist_seconds", 40.0))
        park_offset = float(gcfg.get("search_park_offset_m", 0.20))
        offsets = tuple(sorted({round(park_offset, 3), 0.15, 0.12}, reverse=True))

        def resume_note(b):
            pos = [round(float(v), 3) for v in b.position]
            return (f"The search is preparatory, not the task: resume the ORIGINAL task on "
                    f"{query!r} (seen as {b.label!r} at {pos}). Parked occluders stay where "
                    "they are unless the task needs them back (restore_scene).")

        since = self._fresh_look(frames=1)
        seen = self._seen_since(query, since)
        if seen is not None:
            self.memory.add("note", f"{query!r} is already visible; no search needed")
            return {"found": True, "status": "already_visible", "label": query,
                    "detected_as": seen.label, "color": seen.color,
                    "position": [round(float(v), 3) for v in seen.position],
                    "occluders_moved": [], "attempts": 0, "task_complete": False,
                    "next_action": resume_note(seen)}
        self.memory.add("note", f"{query!r} not visible: starting an occlusion search "
                                f"(up to {max_occluders} occluder(s))")
        moved: list[dict] = []
        failed: list[dict] = []
        tried: set[str] = set()
        parked: list[tuple[np.ndarray, float]] = []
        attempts = 0
        found = None
        error = stage = None
        left_in_place, left_by = False, None  # the last grasp attempt left the arm where it stopped
        while attempts < max_occluders and found is None:
            if self._task_budget_exhausted():
                error = "task persistence budget exhausted (an earlier tier already spent it)"
                break
            if self._estopped():
                error = "e-stop latched; not moving"
                break
            beliefs = list(self.beliefs.all())
            # never the target itself, never a remembered-only object (the
            # look just missed it, so a grasp would too), never something
            # already lifted (by name, and by sitting where a park put it)
            candidates = [
                b for b in beliefs
                if not self._belief_is_target(b, query) and str(b.label) not in tried
                and float(b.last_seen_t) >= since
                and not any(float(np.linalg.norm(np.asarray(b.position, dtype=float)[:2] - p)) < r
                            for p, r in parked)
            ]
            if not candidates:
                break
            # hollow first, then the biggest footprint, then a stable name
            candidates.sort(key=lambda b: (
                0 if any(w in str(b.label).lower().split() for w in _HOLLOW_WORDS) else 1,
                -self._footprint_m(b), str(b.label)))
            occ = candidates[0]
            tried.add(str(occ.label))
            avoid = [(np.asarray(b.position, dtype=float)[:2],
                      0.5 * (self._footprint_m(b) + self._footprint_m(occ)) + 0.02)
                     for b in beliefs if b is not occ]
            spot = self._free_park_spot(np.asarray(occ.position, dtype=float)[:2], avoid, offsets_m=offsets)
            if spot is None:
                failed.append({"label": str(occ.label), "error": "no free, reachable spot to park it"})
                continue
            if not moved:
                self._refuse_a_possible_hold_before_grasping()   # lifting an occluder starts with a grasp
            attempts += 1
            self.memory.add("note", f"lifting {occ.label!r} (likely occluder {attempts}/{max_occluders}) "
                                    f"and parking it at {np.round(spot, 3).tolist()}")
            rec = self._move_object_to(self._belief_query(occ), spot, budget_s=budget_s, kind="park")
            rec["label"] = str(occ.label)
            rec["from"] = [round(float(v), 3) for v in np.asarray(occ.position, dtype=float)[:2]]
            moved.append(rec)
            left_in_place, left_by = bool(rec.get("home_skipped")), rec
            if rec.get("holding"):
                error, stage = rec.get("error", "park failed"), "park"
                break
            if rec.get("stop"):
                error, stage = rec.get("error", "release failed"), "release"
                break
            if not rec["ok"]:
                failed.append({"label": rec["label"], "error": rec.get("error", "grasp failed")})
                continue
            parked.append((np.asarray(spot, dtype=float)[:2], max(0.05, self._footprint_m(occ))))
            since = self._fresh_look()
            found = self._seen_since(query, since)
        result: dict = {"label": query, "occluders_moved": moved, "attempts": attempts,
                        "failed": failed, "task_complete": False}
        if found is not None:
            self.memory.add("action", f"found {query!r} after lifting {attempts} occluder(s)")
            result.update(found=True, status="found", detected_as=found.label, color=found.color,
                          position=[round(float(v), 3) for v in found.position],
                          next_action=resume_note(found))
            return result
        lifted = [m["label"] for m in moved if m.get("ok")]
        if error is None:
            if attempts == 0 and not failed:
                error = f"{query!r} not visible and nothing on the table could be hiding it"
            else:
                error = (f"{query!r} not found after lifting {len(lifted)} occluder(s) "
                         f"{lifted}; the search is exhausted")
        result.update(
            ok=False, found=False, status="not_found", stuck=True, error=error,
            ask_human=f"Where is the {query}? Point to it or place it on the table, then try again.",
            next_action=("Do not guess or grasp a look-alike. Report the search as stuck and ask "
                         "the human; the parked occluders stay where they are (restore_scene puts "
                         "them back if a snapshot exists)."),
        )
        if stage is not None:
            result["stage"] = stage
        if self.held_object:
            result["holding"] = self.held_object
            result["next_action"] = (
                f"Still holding {self.held_object!r}: parking it was refused by the planner/harness. "
                "Put it down with place_at at explicit free coordinates (or open_gripper over a safe "
                "spot) before anything else; the search is stuck."
            )
        if left_in_place:
            result["home_skipped"] = True
            _with_move_home_first(result, source=left_by)
        self.memory.add("outcome", f"search for {query!r} failed: {error[:100]}")
        return result

    def skill_rotate_joint(self, joint: int, degrees: float) -> dict:
        """Rotate one numbered joint by a signed relative angle, through SafeArm."""
        if isinstance(joint, bool) or not isinstance(joint, int) or not 1 <= joint <= self.arm.n_joints:
            raise SkillError(f"joint must be an integer from 1 to {self.arm.n_joints}")
        degrees = float(degrees)
        if not np.isfinite(degrees) or not 0 < abs(degrees) <= 90:
            raise SkillError("degrees must be finite, nonzero, and between -90 and 90")
        q = self.arm.get_state().q.copy()
        q[joint - 1] += np.deg2rad(degrees)
        lo, hi = self.kin.joint_limits
        margin = self.arm.harness.limits.joint_margin
        if not lo[joint - 1] + margin <= q[joint - 1] <= hi[joint - 1] - margin:
            raise SkillError("requested rotation exceeds the joint limits; no motion sent")
        # Conservative 10 deg/s peak for the minimum-jerk profile.
        duration = max(2.0, 1.875 * abs(degrees) / 10.0)
        if not self._move_joints(q, duration_s=duration):
            raise SkillError("joint rotation did not settle; do not retry automatically")
        measured = self.arm.get_state().q
        return {"joint": joint, "relative_degrees": degrees,
                "measured_degrees": float(np.rad2deg(measured[joint - 1]))}

    def skill_move_relative(self, direction: str | None = None, distance_m: float | None = None,
                            amount: str | None = None, frame: str | None = None,
                            vector: list | None = None) -> dict:
        """Nudge the TCP (and whatever it holds) by a short straight displacement.

        Fine, chat-driven control: "a bit to the left", "lift it up", "bring
        it toward me", "a bit more". A named direction or a free vector, in
        the robot base frame (the default for bare left/right/forward/back),
        the viewer frame (the fixed scene camera the demo screen shows) or
        the tool frame (in/out along the gripper); `direction="again"`
        repeats the last one. Interpretation, vetting, refusal and history
        live in `skills/relative_motion.py` + `skills/spatial_moves.py`;
        their docstrings explain why the default frame is the robot base and
        why a refused move is vetted, never probed. Every result states the
        frame and the base-frame displacement it used.
        """
        from . import spatial_moves

        return spatial_moves.move_relative(self, direction=direction, distance_m=distance_m,
                                           amount=amount, frame=frame, vector=vector)

    def skill_rotate_gripper(self, direction: str | None = None, angle_deg: float | None = None,
                             amount: str | None = None, frame: str | None = None) -> dict:
        """Small bounded orientation nudge ("rotate it a bit", "tilt it down").

        Roll (clockwise/counterclockwise, about the gripper's own axis, as
        seen looking along the jaws) or tilt (the jaws swing toward a
        direction), at most 30 deg per call, about the TCP point. The rig's
        configured wrist-camera orientation and clearance gates apply;
        "upside down" is refused by name. See
        `spatial_moves.rotate_gripper`.
        """
        from . import spatial_moves

        return spatial_moves.rotate_gripper(self, direction=direction, angle_deg=angle_deg,
                                            amount=amount, frame=frame)

    def skill_move_near_object(self, label: str | None = None, relation: str | None = None,
                               clearance_m: float | None = None, side: str | None = None,
                               frame: str | None = None, distance_m: float | None = None,
                               amount: str | None = None) -> dict:
        """Position the gripper and its load relative to a named object or
        the table: "hold it above the green spool", "next to the cup",
        "closer to the box", "lower it until it's just above the table".

        The target is localized fresh; heights use its OBSERVED top plus
        clearance plus how far a held object hangs below the TCP; the route
        climbs first / descends last and is vetted end to end before the
        first command. Never opens or closes the jaws. See
        `spatial_moves.move_near_object`.
        """
        from . import spatial_moves

        return spatial_moves.move_near_object(self, label=label, relation=relation,
                                              clearance_m=clearance_m, side=side, frame=frame,
                                              distance_m=distance_m, amount=amount)

    def skill_undo_move(self, steps: int = 1) -> dict:
        """"Go back" / "undo that": retrace the last `steps` (1-5) moves made by
        move_relative / rotate_gripper / move_near_object, newest first.

        Only from where the recorded move ended (otherwise the history is
        stale and is cleared), through the recorded waypoints in reverse, and
        vetted by the harness before any command -- see
        `spatial_moves.undo_move`.
        """
        from . import spatial_moves

        return spatial_moves.undo_move(self, steps=steps)

    def skill_open_gripper(self) -> dict:
        # The wait's bound is validated before the jaws move (round 2, M6).
        timeout = open_wait_timeout_s(self.cfg.grasp)
        # R5 / D9: on an arm that does not hold the object (another arm's
        # hold, or another arm's interrupted grasp) this opens ITS OWN jaws
        # only; the other arm's hold, provisional hold and carry attachment
        # stay, or that arm's next grasp would open its jaws on the object.
        held_elsewhere = self._held_by_another_arm() is not None
        prov = getattr(self, "_held_provisional", None)
        prov_elsewhere = prov is not None and self._provisional_on_another_arm(prov)
        self._set_gripper(self._grip_open, effort=0.8)
        self._gripper_opens_completed += 1
        # The open command is the agent's intentional release: it ends the
        # carry scope, as place_at's release does. The wait below reads the
        # jaws, and an attachment still active would reject them once the
        # object left (review N1: a real release reported as a failure). A
        # refused command raised above and keeps the attachment.
        if not (held_elsewhere or prov_elsewhere):
            self._carry_attachment = None
        if self.held_object and not held_elsewhere:
            self.memory.add("action", f"released {self.held_object!r}")
            self.held_object = None
            self._held_arm = None
            self._held_det_label = None
        # An explicit open after an interrupted grasp settles the question
        # the provisional marker left open: whatever the jaws may have held
        # is released now, so nothing remains to promote.
        if not prov_elsewhere:
            self._held_provisional = None
        if not held_elsewhere:
            self._held_width_m = None
        if timeout is not None:
            self._wait_gripper_opened(timeout)
        return {"gripper": "open"}

    def _wait_gripper_opened(self, timeout_s: float) -> None:
        """Return once the jaws are open or have stopped moving.

        Found on the physical reBot: `set_gripper` only sends the target, so
        the "empty" postcondition read the jaws ~0.5 ms after the command and
        reported "gripper still closed" while they were still opening. A stall
        short of open (something in the way) also ends the wait: the
        postcondition then judges the measured width. Unknown feedback ends it
        at once (the postcondition reports it unverified).
        """
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise SkillError("open_wait_timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        anchor, stable_since = None, time.monotonic()
        while True:
            width = self._gripper_width_frac()
            now = time.monotonic()
            if width is None or not np.isfinite(width):
                return
            if width >= .95:
                return
            if anchor is None or abs(width - anchor) > .002:
                anchor, stable_since = width, now
            elif now - stable_since >= .5:
                return
            if now >= deadline:
                raise SkillError(f"gripper still opening after {timeout_s:.1f} s (at {width:.2f} of full travel)")
            time.sleep(.05)

    def skill_close_gripper(self) -> dict:
        """Close the jaws (the rigid two-stage close). A close never opens
        them: a stage more open than the jaws read is skipped, so a close
        from the stage-2 seat (0.93 rad on the reBot) no longer reopens them
        to the 3.1 rad scout stage first, which could drop a thin object
        seated there (#379 review A). Jaws that give no reading (no valid
        feedback, or one that is no number) while an object may be in them
        -- an interrupted grasp's provisional hold, or a held object --
        refuse before any command: no stage can be shown not to open them.
        With nothing held, unread jaws close as before. So do the jaws of an
        arm that does not hold the object (R5 / D9; #400 re-review): only the
        holding arm's jaws can be on it, and this close never commands them.
        Refusing there pointed at a place_at that the holder rule refuses too.
        """
        state = self._jaw_state()
        reading = float(state.gripper_pos) if getattr(state, "gripper_valid", True) else None
        if reading is not None and not math.isfinite(reading):
            reading = None
        if reading is None:
            self._raise_on_a_possible_hold()
            if self.held_object and self._held_by_another_arm() is None:
                raise SkillError(_fit_label(_JAWS_NOT_READ + " while holding {}: a close stage could open the "
                                            "jaws on it; place it first (place_at)", repr(self.held_object),
                                            keep=len("SkillError: ")))
        closes = self._gripper_closes_commanded
        skipped = self._close_two_stage(select_profile("", "rigid"), _never_past=reading, _count_skips=True)
        wf = self._gripper_width_frac()
        out = {"gripper": "closed", "open_frac": round(wf, 2) if wf is not None else "unknown"}
        if skipped:
            out["stages_skipped"] = skipped
            if self._gripper_closes_commanded == closes:
                out["note"] = ("the jaws already read at least as closed as every close stage, so none was "
                               "sent: a close never opens them")
        return out

    def _reset_camera_frames(self, *, require_geometry=False, scene_changed=True):
        """Reopen reset's capture barrier, retaining failed attempts for retry."""
        from ..perception.freshness import frames_after_reset

        pending = getattr(self, "_reset_observation_pending", False)
        scene_changed = bool(scene_changed or (pending and getattr(self, "_reset_scene_changed", True)))
        self._reset_scene_changed = scene_changed
        self._reset_observation_pending = True
        occupancy = getattr(self.arm.harness, "occupancy", None)
        deadline = time.monotonic() + 5.
        producer_clock = None
        guard = lambda: None
        try:
            if not scene_changed:
                if occupancy is None or not occupancy.tracks_payload:
                    raise SkillError("retained-anchor refresh requires payload tracking")
                occupancy.fence_capture_refresh("validating the producer clock")
                if not pending:
                    self._refresh_halt_generation = self.arm.harness._halt_generation
                validate = getattr(self.arm.raw, "validate_simulation_clock", None)
                if not callable(validate):
                    raise SkillError("retained-anchor refresh requires a validated Isaac producer clock")
                producer_clock = validate()
                def guard():
                    self.arm.harness._check_halt_generation(self._refresh_halt_generation)
                    if self.arm.harness.estopped:
                        raise SafetyViolation("e-stop latched during retained-anchor refresh")
                    if time.monotonic() >= deadline:
                        raise SkillError("retained-anchor capture refresh deadline expired")
                guard()
            if self.watcher is not None:
                observed = self.watcher.reset_camera_frames(self.camera,
                    timeout_s=max(.001, deadline-time.monotonic()), scene_changed=scene_changed,
                    producer_clock=producer_clock, guard=guard)
            else:
                resetting_map = (getattr(occupancy, "begin_scene_reset", lambda: False)() if scene_changed
                                 else occupancy.begin_capture_refresh(producer_clock))
                observed = frames_after_reset([self.camera], timeout_s=max(.001, deadline-time.monotonic()))
                if resetting_map:
                    occupancy.finish_scene_reset([floor for _, floor, _ in observed])
                    fresh = self.depth.ensure_depth(observed[0][2])
                    T = fresh.T_base_cam if fresh.T_base_cam is not None else self.extrinsics.cam_to_base()
                    occupancy.refresh(fresh, T)
                    if not scene_changed:
                        occupancy.wait_payload_ready([floor for _, floor, _ in observed], deadline=deadline, guard=guard)
            if not scene_changed:
                current_clock = validate()
                if any(current_clock.get(key) != producer_clock.get(key) for key in ("source", "robot_id", "epoch")):
                    raise SkillError("producer identity changed during retained-anchor refresh")
                guard()
            if require_geometry and occupancy is not None:
                if (occupancy.scene_reset_pending or occupancy.last_error
                        or occupancy.is_stale()):
                    raise SkillError(
                        "reset recovery has no fresh validated occupancy geometry"
                        + (f": {occupancy.last_error}" if occupancy.last_error else "")
                    )
        except Exception as exc:
            if not scene_changed and occupancy is not None:
                occupancy.fence_capture_refresh(exc)
            raise
        self._reset_observation_pending = False
        return observed

    def skill_reset_scene(self) -> dict:
        """Start the demo over: arm home, every sim prop back on its spawn
        pose, the world model and the task's visual memory cleared, one fresh
        observation. This is what runs between two visitors -- the launcher's
        own proof turn already moved the red cube, so without it the first
        visitor of the day starts "put both cubes in the drop zone" with one
        cube already there.

        Sim only for the props: on a real rig the props do not teleport, so
        the result says `props_reset: []` and the human puts them back; the
        rest (home, memory, re-observe) still applies. The arm goes home
        FIRST so a prop respawning under the gripper is not respawned into
        the jaws."""
        out: dict = {"props_reset": [], "world": None, "observation_refreshed": False,
                     "objects_visible": []}
        # The reset's ONE halt generation, taken when it starts (before its
        # carry check): a halt from here on ends the reset. The home move and
        # the one jaw command at home below (the release, or the opening of
        # closed empty jaws) carry it.
        reset_generation = getattr(self.arm.harness, "_halt_generation", None)
        column = stop_retrace.live_on_another_arm(self)
        if column is not None:
            # Another arm was stopped at its object and keeps its approach
            # column (stop_retrace.py): this reset forgets every arm's column
            # once home, and that arm's move_home could no longer raise its
            # gripper up it. Refused before any motion (Task 5 review Minor 1),
            # and before the hold refusal below: its place_at on that arm
            # would forfeit the column too, so move_home comes first.
            return {**out, "ok": False, "stage": "home", "home_skipped": True, "beliefs_forgotten": 0,
                    "error": self._column_elsewhere_refusal(column)}
        holder = self._held_by_another_arm()
        if holder is not None:
            # R5 / D9: another arm holds the object. This reset would home
            # and open the RUNNING arm and forget the object while the other
            # arm still carries it: refuse before any motion, naming it.
            return {**out, "ok": False, "stage": "release", "home_skipped": True, "beliefs_forgotten": 0,
                    "error": self._held_elsewhere_refusal(self.held_object, holder)}
        try:
            carry_attachment.check(self)
        except carry_attachment.AttachmentInvalid as exc:
            return {**out, **carry_attachment.failure_result(self, exc)}
        if ((self.held_object or getattr(self, "_held_provisional", None) is not None)
                and self._drives_real_gripper()):
            refusal = self._reset_real_hold_refusal()
            if refusal is not None:
                # The release happens at home: a real gripper would drop the
                # object from there (~0.44 m on the reBot). Refuse before any
                # motion; the object stays held.
                return {**out, "ok": False, "stage": "release", "home_skipped": True, "beliefs_forgotten": 0,
                        "error": refusal}
        occupancy = getattr(self.arm.harness, "occupancy", None)
        recovery_generation = None
        recovery_cancellation = None
        def recovery_cancelled():
            if recovery_generation is None:
                return False
            try:
                self.arm.harness._check_halt_generation(recovery_generation)
                if recovery_cancellation is not None:
                    self.arm.harness.check_motion_cancellation(recovery_cancellation)
                if self.arm.harness.estopped:
                    raise SafetyViolation("e-stop latched during scene recovery")
            except SafetyViolation as exc:
                out.update(ok=False, stage="reset_recovery", recovery_error=str(exc))
                stop = f"Scene recovery cancelled: {exc}"
                if isinstance(exc, MotionHalted) and out.get("error"):
                    # The reset had already failed (the Isaac prop reset, the
                    # observation): the halt is named first and that failure
                    # files the call (`skill_error`), as a halted park keeps
                    # its skill's failure. An e-stop names itself alone.
                    _stop_first(out, stop)
                else:
                    out["error"] = stop
                out.setdefault("beliefs_forgotten", 0)
                return True
            return False
        if getattr(self, "_release_episode", None) is not None:
            from .release_episode import recover
            recovery_generation = self._release_episode["halt_generation"]
            try:
                out["release_recovery"] = recover(self)
            except (SkillError, SafetyViolation) as e:
                out.update(ok=False, stage="release_recovery", home_skipped=True,
                           recovery_error=str(e), beliefs_forgotten=0,
                           error=f"Reset could not finish the retained release withdrawal: {e}")
                return out
        if (getattr(self, "_reset_observation_pending", False)
                or getattr(occupancy, "scene_reset_pending", False)):
            # A previous reset may have invalidated geometry before a camera
            # timed out. Reacquire it while stationary, before asking home's
            # collision gate to plan any motion or releasing a held object.
            try:
                scene_changed = getattr(self, "_reset_scene_changed", True)
                observed = self._reset_camera_frames(require_geometry=True, scene_changed=scene_changed)
                if not scene_changed:
                    recovery_generation = self._refresh_halt_generation
                    out["retained_anchor_refresh"] = occupancy.last_capture_refresh
                self.last_frame = self.depth.ensure_depth(observed[0][2])
                self._mark_perception_fresh(self.last_frame)  # observed[0]: the primary camera
                out["geometry_recovered"] = True
            except Exception as e:  # noqa: BLE001
                out.update(ok=False, stage="reset_recovery", home_skipped=True,
                           recovery_error=str(e), beliefs_forgotten=0,
                           error=f"Reset recovery could not refresh geometry: {e}")
                return out
        if getattr(self, "_contact_episode", None) is not None:
            from .contact_episode import recover
            try:
                out["contact_recovery"] = recover(self)
                self._reconcile_held()
            except (SkillError, SafetyViolation) as e:
                out.update(ok=False, stage="contact_recovery", home_skipped=True,
                           recovery_error=str(e), beliefs_forgotten=0,
                           error=f"Reset could not withdraw the retained contact episode: {e}")
                return out
        # The human has cleared what blocked the arm: forget the contact
        # zones (safety.contact_monitor) before the home move plans a route.
        out["contact_zones_cleared"] = sum(
            h.clear_contact_zones("reset_scene") for h in self._contact_harnesses()
            if callable(getattr(h, "clear_contact_zones", None)))
        home_ok = True
        try:
            pending_withdrawals = getattr(self, "_mujoco_withdrawals", {})
            pending_withdrawal = pending_withdrawals.get(id(self.arm))
            if pending_withdrawal is not None:
                out["withdrawal_recovery"] = pending_withdrawal.recover_for_reset()
                recovery_generation = out["withdrawal_recovery"]["generation"]
                recovery_cancellation = out["withdrawal_recovery"]["cancellation"]
            else:
                generation = recovery_generation if recovery_generation is not None else reset_generation
                at_home = (self.skill_move_home(_halt_generation=generation) if generation is not None
                           else self.skill_move_home())
                if isinstance(at_home, dict) and "stop_retrace" in at_home:
                    out["stop_retrace"] = at_home["stop_retrace"]
        except (SkillError, SafetyViolation) as e:
            out.update(ok=False, stage="home", home_error=str(e), beliefs_forgotten=0,
                       error=f"Scene reset could not reach home: {e}")
            # A stop retrace that completed before the failed home move: the
            # result still says so, as move_home's own (#359 review M4).
            _with_retrace_result(out, e)
            return out
        # The arm is home and the scene changes next: a column kept after a
        # STOP (stop_retrace.py) no longer describes anything, for any arm.
        stop_retrace.forget(self, every_arm=True)
        if recovery_cancelled():
            return out
        # At most ONE jaw command at home, to the profile's OPEN position
        # (gripper.open_pos: jaw positions are motor travel in each profile's
        # units, so a raw 1.0 was nearly closed on the reBot (open 6.2) and
        # fully closed on the mock (open 0.0)), effort 0.8, carrying one halt
        # generation (a recovery's, else the reset's): a halt since the reset
        # started, or an e-stop, refuses it before the driver.
        # - Holding (simulated and mock arms only, see above): the release. A
        #   reset while carrying is the operator's decision that the episode is
        #   over.
        # - Empty-handed with closed jaws (an air grasp whose reopen was
        #   refused, an explicit close): the opening, so the next pick starts
        #   with open jaws (PR-A jaws round, ruling 2). Never while an object
        #   may be held (a provisional hold, an active carry attachment), never
        #   on jaws that read open or unknown.
        jaw_generation = recovery_generation if recovery_generation is not None else reset_generation
        jaw_kw = {} if jaw_generation is None else {"_halt_generation": jaw_generation}
        if self.held_object:
            try:
                self._set_gripper(self._grip_open, effort=0.8, **jaw_kw)
            except Exception as exc:  # noqa: BLE001
                if recovery_generation is not None:
                    out.update(ok=False, stage="reset_recovery", recovery_error=str(exc),
                               beliefs_forgotten=0, error=f"Scene recovery could not release safely: {exc}")
                    return out
                return self._reset_release_failed(out, exc)
            self._forget_held_object()
        elif self._reset_finds_closed_empty_jaws():
            try:
                self._set_gripper(self._grip_open, effort=0.8, **jaw_kw)
            except Exception as exc:  # noqa: BLE001
                if self._reset_jaws_open_failed(out, exc):
                    return out
            else:
                out["jaws_open_commanded"] = True
        world = None
        raw = getattr(self.arm, "raw", None)
        # MujocoArm exposes its shared world; a LazyArm that never
        # materialized has moved nothing, so there is nothing to put back.
        if raw is not None and getattr(raw, "connected", True):
            world = getattr(raw, "world", None)
        if world is None:
            try:  # MCP mode: the rendered camera may own the only live world
                from ..sim import mujoco_world

                live = mujoco_world.live_worlds()
                world = live[0] if len(live) == 1 else None
            except Exception:  # noqa: BLE001
                world = None
        if recovery_cancelled():
            return out
        if world is not None and hasattr(world, "reset_props"):
            out["props_reset"] = list(world.reset_props())
            out["world"] = "mujoco"
            if pending_withdrawal is not None:
                try:
                    out["withdrawal_reset_verification"] = pending_withdrawal.verify_reset(
                        out["props_reset"], recovery_generation, recovery_cancellation)
                except (SkillError, SafetyViolation) as exc:
                    return {**out, "ok": False, "stage": "reset_verification", "error": str(exc)}
        elif hasattr(raw, "reset_props"):
            out["world"] = "isaac"
            try:  # Isaac bridge returns physics read-back, not camera inference.
                from ..sim.isaac_reset import validate_isaac_reset

                reset = validate_isaac_reset(raw.reset_props())
                out["props_reset"] = reset["props_reset"]
                out["reset_verification"] = reset["reset_verification"]
                if reset.get("seed") is not None:  # seeded layout (bridge set_reset_seed)
                    out["scene_seed"] = reset["seed"]
            except Exception as e:  # noqa: BLE001
                out["world_error"] = str(e)
                out["error"] = f"Isaac prop reset failed: {e}"
                home_ok = False  # do not let the final home result mask a reset failure
        if recovery_cancelled():
            return out
        dropped = self.beliefs.clear()
        out["beliefs_forgotten"] = dropped
        if recovery_cancelled():
            return out
        self.memory.reset_frames()
        if recovery_cancelled():
            return out
        self.memory.add("note", f"scene reset: {len(out['props_reset'])} prop(s) respawned, "
                                f"{dropped} belief(s) forgotten")
        if recovery_cancelled():
            return out
        try:
            # A cached Isaac image can be delivered repeatedly with NEW local
            # frame IDs and receipt times. Fence by the carried producer clock,
            # and analyze that exact frame instead of fetching another copy.
            if out["props_reset"]:
                from ..perception.freshness import capture_marker

                observed = self._reset_camera_frames()
                if recovery_cancelled():
                    return out
                frame = self.depth.ensure_depth(observed[0][2])
                if recovery_cancelled():
                    return out
                self.last_frame = frame
                self._mark_perception_fresh(frame)  # observed[0]: the primary camera
                obs = self._describe_observation(frame)
                if recovery_cancelled():
                    return out
                out["observation_freshness"] = [
                    {"camera": getattr(camera, "name", None),
                     "floor": capture_marker(floor), "observed": capture_marker(fresh)}
                    for camera, floor, fresh in observed
                ]
            else:
                obs = self.skill_get_observation()
                if recovery_cancelled():
                    return out
            out["objects_visible"] = obs.get("objects_visible", [])
            out["observation_refreshed"] = True
        except Exception as e:  # noqa: BLE001
            out["observe_error"] = str(e)
            out.setdefault("error", f"Reset observation could not be refreshed: {e}")
        out["beliefs_forgotten"] = dropped
        if recovery_cancelled():
            return out
        out["ok"] = home_ok and out["observation_refreshed"]
        if out["ok"]:
            self._carry_attachment = None  # The existing explicit scene reset completed.
            if out.get("world") == "mujoco" and out.get("props_reset"):
                # Clear only this arm, after actual prop reset and fresh
                # observation. Failed resets and ordinary retries retain it.
                if pending_withdrawal is not None:
                    try:
                        pending_withdrawal.complete(generation=recovery_generation,
                                                    cancellation=recovery_cancellation)
                    except (SkillError, SafetyViolation) as exc:
                        return {**out, "ok": False, "stage": "reset_completion", "error": str(exc)}
        return out

    #: Arm types (configs/arms `type:`) whose jaws open in a simulator or a
    #: mock. Every other type -- and an unknown or missing one -- drives a
    #: real gripper.
    _SIM_ARM_TYPES = ("isaac", "mujoco", "mock")

    def _running_arm_profile(self):
        """The profile of the arm this call drives: `cfg.arm` for the
        primary, else the `cfg.arms` entry of the arm execute()'s `arm=`
        selected (rig order = `cfg.arms` order). None when the running arm
        is neither (cannot be told). Identity only: a standby LazyArm is
        never probed."""
        running = self.arm
        if running is getattr(self, "_arm", None):
            return self.cfg.get("arm") or {}
        rig = getattr(self, "arm_rig", None)
        index = next((i for i, arm in enumerate(rig or ()) if arm is running), None)
        profiles = self.cfg.get("arms") or []
        return profiles[index] if index is not None and index < len(profiles) else None

    def _drives_real_gripper(self) -> bool:
        """The arm this call drives (the `arm=` selection included) has a
        real gripper: its type is not isaac*, mujoco* or mock*; an unknown
        type or arm counts as real."""
        arm = self._running_arm_profile() or {}
        return not str(arm.get("type") or "").startswith(self._SIM_ARM_TYPES)

    def _reset_real_hold_refusal(self) -> str | None:
        """Why reset_scene on a real gripper refuses before any motion, or
        None. A held object refuses (#357). A grasp ended after its close and
        before its held flag (an operator STOP right after the close or in
        the lift, a lift that did not settle, a contact) leaves only the
        close's provisional hold. It is classified with the jaws first,
        exactly like the grasp's own verify (`_reconcile_held(seat_aware=
        True)`, `_air_grasp` against the opening the close commanded):
        - stalled on something (wider than the verify's air band below): the
          hold is promoted and the reset refuses instead of homing with the
          object (a later open_gripper dropped it);
        - nearly shut, open, or within 3 % of travel of the seat while the
          plan sized the object at least 7 % wider (the reBot's rigid close
          on air stops at its seat, 85 % of travel, 0.93 rad): nothing is
          held, the hold is cleared and the reset runs as for empty jaws.
          That band is the verify's price: on the reBot an object about
          13.5-16.2 mm thick that the plan sized at 19.8 mm or more stalls
          the jaws inside it, and the reset opens the jaws on it at home;
        - unreadable (unknown feedback, a reading that is no number, a
          failed read): refused too (fail closed), naming the way out. On
          the reBot only a jaw command restores a lost turn reference, and it
          closes first: an object still in the jaws must come out by hand
          before open_gripper;
        - read but not judgeable (a malformed marker): refused with its own
          text and the same way out;
        - an unfinished contact episode: the jaws are never read (only its
          recovery may finish that close), refused with its own text.
        While an approach column is kept after a STOP at the object
        (stop_retrace.py), the held, unreadable and not-judged refusals send
        the agent to move_home FIRST: it raises the gripper up that column,
        with the object; place_at or open_gripper first would forfeit the
        column and freeze the arm at the object (#359 review, Important #1).
        "Kept" is `stop_retrace.live`, the rule move_home's retrace acts on: a
        column that expired, or was kept for another arm or backend or before
        a scene reset, names #363's texts, never a move_home that could no
        longer raise the gripper. (A contact episode is never such a column.)
        Each refusal's sentence comes first and fits the 240 characters the
        agent's failure line quotes: a long label is shortened, never the
        sentence, and the driver's text after it is cut to the room left; the
        label gives way to that text only as far as it needs (at most
        _REFUSAL_ERROR_ROOM characters). A held flag is never reconciled
        here: it refuses as it is.
        A provisional hold another arm's close left (R5) is never judged by
        this arm's jaws: refused naming that arm, the one that must settle it."""
        provisional = getattr(self, "_held_provisional", None)
        if not self.held_object and provisional is not None and self._provisional_on_another_arm(provisional):
            return self._held_elsewhere_refusal(provisional[0], self._arm_name_of(provisional.arm), provisional=True)
        contact = getattr(self, "_contact_episode", None) is not None
        read_error = judge_error = None
        if not self.held_object:
            # Judges a provisional hold; without one it reads nothing.
            try:
                self._reconcile_held(seat_aware=True)
            except _HoldNotJudged as exc:
                judge_error = exc.__cause__
            except Exception as exc:  # noqa: BLE001 -- unknown jaws: refused below
                read_error = exc
        column_kept = stop_retrace.live(self) is not None
        if self.held_object:
            if column_kept:
                return _fit_label("holding {}: call move_home first (it raises the gripper up its approach "
                                  "column, with the object), then place_at, then reset_scene", self.held_object)
            return _fit_label("holding {}: place it first (place_at), then reset", self.held_object)
        provisional = getattr(self, "_held_provisional", None)
        if provisional is None:
            return None
        if contact:
            return _fit_label("the grasp of {} ended in an unfinished contact episode, so the jaws were not "
                              "read and it may be in them: a person must check the gripper.", provisional[0])
        return _unread_hold_refusal(provisional[0], column_kept=column_kept, then="reset_scene",
                                    judged=judge_error is not None,
                                    error=judge_error if judge_error is not None else read_error)

    def _arm_name_of(self, arm) -> str | None:
        """The rig name of the SafeArm `arm` (identity search), or None."""
        rig = getattr(self, "arm_rig", None)
        if rig is None:
            return "default" if arm is getattr(self, "_arm", None) else None
        return next((key for key, candidate in rig.arms.items() if candidate is arm), None)

    def _held_elsewhere_refusal(self, label, holder, *, provisional: bool = False, then: str = "reset_scene",
                                keep: int = 0) -> str:
        """The refusal of a call on this arm while ANOTHER arm holds the
        object (R5), or may hold it (its grasp stopped after its close):
        reset_scene's, and a grasp skill's (`then`: the call to make once that
        arm settled it). It names that arm and the way out on it, never this
        arm's jaws. Fits the agent's failure line with `keep` characters to
        spare (a raised refusal's "SkillError: "): a long label is shortened,
        never the sentence."""
        arm = repr(holder).replace("{", "{{").replace("}", "}}")
        here = repr(self._running_arm_name()).replace("{", "{{").replace("}", "}}")
        if provisional:
            template = ("{} may be held by arm " + arm + " (stopped after its close), not " + here
                        + ": settle it with arm=" + arm + " first (move_home, then place_at or open_gripper), "
                        "then " + then + "; no motion was sent")
        else:
            template = ("{} is held by arm " + arm + ", not " + here + ": place it with arm=" + arm
                        + " first (place_at), then " + then + "; no motion was sent")
        return _fit_label(template, repr(str(label)), keep=keep)

    def _column_elsewhere_refusal(self, column) -> str:
        """reset_scene's refusal while ANOTHER arm keeps a live stop column
        (`stop_retrace.live_on_another_arm`): the way out is that arm's
        move_home (it raises the gripper up its column), then this reset.
        When that move_home is refused too (its retrace refused), the
        fallback (Task 5 fix review Minor 1): the column expires (the time
        left, rounded up), or any other motion skill on that arm forfeits it."""
        arm, here = repr(self._arm_name_of(column.arm)), repr(self._running_arm_name())
        left = int(math.ceil(stop_retrace.expires_in_s(column)))
        return _cut_text(f"arm {arm} keeps its approach column after a stop, not {here}: call move_home with "
                         f"arm={arm}, then reset_scene. If refused, the column expires in {left} s, or any other "
                         "motion skill on that arm drops it; no motion was sent", _FAILED_ERROR_CHARS)

    def _forget_held_object(self) -> None:
        """The jaws let the held object go: clear every held-object field."""
        self.held_object = None
        self._held_arm = None
        self._held_det_label = None
        self._held_color = None
        self._held_offset = None
        self._held_width_m = None
        self._held_support_offset_m = None

    def _reset_release_failed(self, out: dict, exc: Exception) -> dict:
        """End reset_scene after its jaw opening raised; the reset goes no
        further. Only a measured open jaw lets the runtime forget the object.

        - Any other SafetyViolation: SafeArm refused before the driver (a
          halt since the reset started, an e-stop, a gripper gate), so nothing
          moved and the object is still held.
        - GripperFault: the driver raised AFTER its commands went out (opening
          blocked part-way, referencing closed onto the object). The jaws may
          have moved: report the measured width; the object stays held unless
          that width shows the jaws open.
        - Any other error (bus or bridge timeout): the object stays held and
          the error is recorded.
        """
        obj = self.held_object
        out.update(ok=False, stage="release", release_error=str(exc), beliefs_forgotten=0)
        if isinstance(exc, SafetyViolation) and not isinstance(exc, GripperFault):
            out["error"] = f"Scene reset stopped before the release: {exc}"
            return out
        # Outcome first, the object named once, the driver's own text last:
        # the agent's failure line keeps only the first 240 characters of the
        # error (a 60-character name still leaves the outcome inside them).
        if not isinstance(exc, GripperFault):
            out["error"] = (f"Scene reset stopped: {obj} is kept as held: the release failed and whether "
                            f"the jaws moved is unknown; check the gripper. Error: {exc}")
            return out
        width, measured = self._reset_measured_jaws(out)
        if width is not None and width >= self._reset_open_frac():
            self._forget_held_object()
            out["error"] = (f"Scene reset stopped: {obj} is no longer held and may have dropped from "
                            f"home height: the gripper faulted while releasing it and the jaws measured "
                            f"{measured}. Gripper fault: {exc}")
        else:
            out["error"] = (f"Scene reset stopped: {obj} is kept as held, but it may have dropped: the "
                            f"gripper faulted while releasing it and the jaws measured {measured}; "
                            f"check the gripper. Gripper fault: {exc}")
        return out

    def _reset_open_frac(self) -> float:
        """The jaw width fraction that counts as AT the open position (the
        tolerance `_reconcile_held` uses)."""
        return 1.0 - float((self.cfg.get("grasp") or {}).get("air_grasp_frac", 0.04))

    def _reset_measured_jaws(self, out: dict) -> tuple[float | None, str]:
        """After a jaw command at reset raised a GripperFault: measure the jaw
        width, record it as `gripper_open_frac` and return (width, its text)."""
        try:
            width = self._gripper_width_frac()
        except Exception:  # noqa: BLE001 -- unknown width
            width = None
        out["gripper_open_frac"] = round(width, 2) if width is not None else "unknown"
        return width, (f"{width:.0%} open" if width is not None else "at an unknown width")

    def _reset_finds_closed_empty_jaws(self) -> bool:
        """reset_scene opens the jaws when nothing is or may be held (no held
        object, no provisional hold, no active carry attachment) and they read
        closed (short of the `_reconcile_held` open tolerance). Jaws that read
        open or cannot be read are left as they are."""
        if (self.held_object or getattr(self, "_held_provisional", None) is not None
                or carry_attachment.active(self) is not None):
            return False
        try:
            width = self._gripper_width_frac()
        except Exception:  # noqa: BLE001 -- unknown jaws stay as they are
            return False
        return width is not None and width < self._reset_open_frac()

    def _reset_jaws_open_failed(self, out: dict, exc: Exception) -> bool:
        """reset_scene's opening of closed EMPTY jaws raised. True ends the
        reset. The same three classes, stage ("release") and `release_error`
        as `_reset_release_failed`, outcome first; nothing is held, so nothing
        can drop.

        - Any other SafetyViolation: SafeArm refused before the driver (a halt
          since the reset started, an e-stop, a gripper gate): nothing moved;
          the reset ends.
        - GripperFault: the driver raised AFTER its command went out: the jaws
          may have moved; the measured width is reported; the reset ends.
        - Any other error (bus or bridge timeout): recorded as
          `jaws_open_error`; the reset goes on (nothing is held, and the next
          grasp opens closed empty jaws first).
        """
        if not isinstance(exc, SafetyViolation):
            out["jaws_open_error"] = (f"the empty jaws were not confirmed open (whether they moved is "
                                      f"unknown): {exc}")
            return False
        out.update(ok=False, stage="release", release_error=str(exc), beliefs_forgotten=0)
        if not isinstance(exc, GripperFault):
            out["error"] = f"Scene reset stopped before opening the empty jaws: {exc}"
            return True
        _, measured = self._reset_measured_jaws(out)
        out["error"] = (f"Scene reset stopped: the gripper faulted while opening the empty jaws and the jaws "
                        f"measured {measured}; check the gripper. Gripper fault: {exc}")
        return True

    @motion_evidence.phase("home")
    def skill_move_home(self, *, _halt_generation=None) -> dict:
        carry_attachment.check(self)
        pending = getattr(self, "_mujoco_withdrawals", {})
        withdrawal = pending.get(id(self.arm))
        if withdrawal is not None:
            result = withdrawal.home()
            withdrawal.complete()
            return result
        context = getattr(getattr(self, "_motion_scope", None), "context", None)
        if context is not None and context["enabled"] and context["root_skill"] != "move_home":
            reason = self._home_return_skip_reason()
            if reason:
                raise SkillError(f"home return suppressed: {reason}")
            home = context["home"]
        else:
            home = self._profile_q("home_q", "move home")
        # After an operator STOP below a grasp's pregrasp, an EXPLICIT
        # move_home / reset_scene (execute()'s root skill) first retraces up
        # the approach column that grasp kept, capped 5 mm above its hover,
        # every step strictly upward inside it (stop_retrace.py). None:
        # nothing kept or applicable, the home move below is today's.
        retraced = stop_retrace.retrace(self, halt_generation=_halt_generation)
        retrace_out = {} if retraced is None else {"stop_retrace": retraced}
        if retraced is not None and context["root_skill"] == "move_home":
            # The retrace was the interrupted lift's own segment: the jaws may
            # hold the object. Say what they hold (`_retrace_jaws_report`);
            # the jaws stay as they are. Not inside reset_scene: its own held
            # rule runs at its start, and a hold promoted mid-reset would be
            # released at home (a drop from ~0.44 m on a real gripper).
            retrace_out.update(self._retrace_jaws_report())
        try:
            if self._home_policy_enabled():
                # The caller's halt generation (the reset's) reaches this move too.
                result = self._return_to_view_home(home, self.cfg.arm, halt_generation=_halt_generation)
                if not result["ok"]:
                    if result["status"] == "halted":
                        raise MotionHalted(result["reason"])     # a halt stays a halt
                    raise SkillError(result.get("error", result.get("reason", "home return failed")))
                return {"at": "home", "return_home": result, **retrace_out}
            # SafeArm plans the complete return before motion and re-vets against
            # actual feedback/map state at every segment. Legacy test doubles may
            # expose only move_joints; all real runtime arms are SafeArms.
            move = getattr(self.arm, "move_planned", self.arm.move_joints)
            kwargs = {} if _halt_generation is None else {"_halt_generation": _halt_generation}
            if not move(home, duration_s=3.0, **kwargs):
                raise SkillError("did not settle at home")
            return {"at": "home", **retrace_out}
        except Exception as exc:
            if retrace_out:
                # A stop or fault in the home move ends the call: its failed
                # result still says what the retrace did and what the jaws
                # hold (execute() merges it; review M2).
                exc.retrace_result = dict(retrace_out)
            raise

    def _retrace_jaws_report(self) -> dict:
        """What the jaws hold after a COMPLETED stop retrace inside an
        explicit move_home, judged like reset_scene and the grasp's own verify
        (`_reconcile_held(seat_aware=True)`, #363): stalled on the object ->
        `holding` + place it first; at the commanded seat, closed or open ->
        nothing (the provisional hold is dropped); unreadable (unknown
        feedback, a reading that is no number, or the read raised) ->
        `holding_unknown`: the object may still be in the jaws, never the
        empty-jaws result (review Important #2, M3). Read but not judgeable
        (a malformed marker, `_HoldNotJudged`, #367): `holding_unknown` too,
        naming that reason; the hold stays and the jaws are not commanded."""
        read_error = judge_error = None
        try:
            self._reconcile_held(seat_aware=True)
        except _HoldNotJudged as exc:  # the jaws were read: the marker could not be judged
            judge_error = exc.__cause__
        except Exception as exc:  # noqa: BLE001 -- the jaw read failed: reported below
            read_error = exc
        if self.held_object:
            return {"holding": self.held_object, "next_action": (
                f"The jaws may still hold {self.held_object!r} (the STOP came after they closed on "
                "it): place it first (place_at), before any other grasp or a reset.")}
        provisional = getattr(self, "_held_provisional", None)
        if provisional is None:
            return {}
        head = _JAWS_NOT_JUDGED if judge_error is not None else _JAWS_NOT_READ
        exc = judge_error if judge_error is not None else read_error
        return {"holding_unknown": True, "next_action": (
            f"{head}: {provisional[0]} may still be in the jaws; check the gripper before any other grasp, open "
            "or reset" + ("" if exc is None else f". Error: {_error_text(exc)}"))}

    def skill_move_to_rest(self, keep_torque: bool = False) -> dict:
        """"Home" / "pose 0" / "rest" / "park": fold the arm onto its rest pose
        (the profile's `park_q`), verify it there by MEASUREMENT, close the
        gripper and switch the motors off (the LazyArm returns to standby), so
        the robot can be powered down. `keep_torque=True` folds it and leaves
        the motors on.

        Not `move_home`: that is the raised INITIAL pose (`home_q`, motors
        on). Refuses while holding (or possibly holding) an object and while
        the e-stop is latched; a pose that is not measured at rest keeps its
        torque. See `skills/rest_pose.py` for the route and the rules.
        """
        from . import rest_pose

        return rest_pose.move_to_rest(self, keep_torque=keep_torque)

    def skill_list_arms(self) -> dict:
        """Which arms this rig has, and which one commands default to.

        Cheap and read-only: reports each arm's DOF and e-stop state without
        touching the motors. Deliberately reads `connected` (LazyArm's own
        surface) rather than joint state, so asking "what arms are there"
        never powers up a standby arm as a side effect.
        """
        if self.arm_rig is None or len(self.arm_rig) <= 1:
            return {
                "arms": ["default"],
                "default": "default",
                "note": "single-arm rig; the `arm` parameter is not needed",
            }
        return {
            "arms": self.arm_rig.names,
            "default": self.arm_rig.names[0],
            "detail": self.arm_rig.stats(),
            "note": (
                "pass arm=<name> to any motion skill; omitting it uses "
                f"{self.arm_rig.names[0]!r}"
            ),
        }

    def skill_recall_memory(self, query: str = "") -> dict:
        out: dict = {"recent_events": self.memory.digest(max_lines=15)}
        if query:
            b = self.beliefs.find(query)
            if b is not None:
                now = time.monotonic()
                out["object_memory"] = {
                    "label": b.label,
                    "last_position": [round(float(x), 3) for x in b.position],
                    "seen_s_ago": round(now - b.last_seen_t, 1),
                    "state": b.state(now),
                }
            # ROADMAP #7, opt-in: "the thing that looked like X". Only with a
            # memory embedder; absent one, this answer is the pre-#7 answer.
            embedder = getattr(self.memory, "embedder", None)
            if embedder is not None:
                if getattr(embedder, "joint_space", False):
                    try:
                        out["looks_like"] = self.memory.recall_visual(str(query), k=3)
                        out["looks_like_note"] = (
                            "remembered appearance matches (cosine >= the embedder's floor), "
                            "not a current observation: localize_object before acting on one")
                        if any(h.get("restored") for h in out["looks_like"]):
                            # B43: a persisted visual index (memory.persist_episodic)
                            out["looks_like_note"] += (
                                ". Hits with restored=true were restored from an earlier session: "
                                "remembered (age_s ago), never seen in this one")
                    except Exception as e:  # noqa: BLE001 -- recall is advisory
                        out["looks_like"] = []
                        out["looks_like_note"] = f"visual recall failed: {type(e).__name__}: {e}"
                else:
                    out["looks_like"] = []
                    out["looks_like_note"] = (
                        "recall by description needs a joint image-text memory embedder "
                        f"(siglip/clip); configured: {embedder.name}")
        return out

    def skill_recall_step(self, n: int = -1) -> dict:
        """Recall ONE executed step of this run (RPent `view_env_state(step=N)`).

        Read-only over the evidence `execute()` already records: the step's
        trace row (skill, args, dispatch tier, outcome, postcondition
        verdict) and its BEFORE/AFTER keyframes. `n` indexes the recorded
        steps, 0 = first, negative = from the end (-1 = the last thing the
        robot did); recall rows themselves are not steps, so -1 keeps
        meaning "the last action" however often the planner looks back.
        An invalid `n` is an explicit error: the recalled frames are cleared
        FIRST, so a bad index can never leave a stale image behind for the
        MCP layer or the next planner turn. Moves nothing, re-verifies
        nothing; the verdict is the one recorded at the time.
        """
        self.last_recalled_frames = []
        try:
            idx = int(n)
            if isinstance(n, bool):
                raise ValueError
        except (TypeError, ValueError):
            raise SkillError(
                f"n must be an integer step index (0 = first, negative = from the end), got {n!r}"
            )
        rows = [r for r in self.trace.rows() if r.get("skill") != "recall_step"]
        total = len(rows)
        if total == 0:
            raise SkillError("no steps recorded yet in this run")
        if not -total <= idx < total:
            raise SkillError(
                f"step {idx} does not exist: {total} step(s) recorded "
                f"(valid n: 0..{total - 1} or -{total}..-1)"
            )
        row = rows[idx]
        step = idx if idx >= 0 else total + idx
        raw_result = row.get("result")
        result: dict = raw_result if isinstance(raw_result, dict) else {}
        raw_pc = result.get("postcondition")
        pc: dict = raw_pc if isinstance(raw_pc, dict) else {}
        verdict = str(pc.get("status") or "none")
        outcome = str(result.get("outcome")
                      or (OUTCOME_OK if result.get("ok") else OUTCOME_FAILED))
        raw_args = row.get("args")
        args: dict = raw_args if isinstance(raw_args, dict) else {}
        tier = str(row.get("tier") or "unknown")
        skill = str(row.get("skill") or "?")
        # the frames' captions reach the agent outside this result (MCP recall_step): the
        # trace row's own text, so the rules' triggers are withheld here too, from the
        # recorded arguments as well (halt_motion's reason; A8 fix 2, A8 follow-ups fix)
        head = f"step {step}: {skill}({_without_rule_triggers(_short(args))})"
        tail = _without_rule_triggers("ok" if outcome == OUTCOME_OK
                                      else f"stuck -- {result.get('ask', '')}" if outcome == OUTCOME_STUCK
                                      else f"failed: {str(result.get('error', ''))[:120]}")
        caption = f"{head} -> {tail} [verdict: {verdict.upper()}; via {tier}]"
        frames: list[dict] = []
        for tag in ("before", "after"):
            rel = row.get(f"keyframe_{tag}")
            if not rel:
                continue
            path = self.trace.run_dir / str(rel)
            try:
                jpeg = path.read_bytes()
            except OSError:
                continue  # a missing file is reported as missing, never substituted
            if not jpeg:
                continue
            frames.append({
                "tag": tag, "path": str(rel), "jpeg": jpeg,
                "caption": (f"{head} -- {tag.upper()} the action"
                            + (f" -> {tail} [verdict: {verdict.upper()}]" if tag == "after" else "")),
            })
        self.last_recalled_frames = frames
        return {
            "step": step,
            "n": idx,
            "trace_step": row.get("step"),
            "steps_recorded": total,
            "skill": skill,
            "args": args,
            "tier": tier,
            "outcome": outcome,
            "ok_reported": result.get("ok"),
            "error": result.get("error"),
            "ask": result.get("ask"),
            "verdict": verdict,
            "postcondition": pc or None,
            "duration_ms": row.get("duration_ms"),
            "t": row.get("t"),
            "keyframes": [{"tag": f["tag"], "path": f["path"]} for f in frames],
            "caption": caption,
            "note": (
                "Read-only recall of the recorded trace: it moves nothing and re-verifies "
                "nothing; the verdict is the one measured at the time. Keyframe images are "
                "attached when the host can show them."
                if frames else
                "Read-only recall of the recorded trace; no keyframe image was recorded for "
                "this step."
            ),
        }

    def skill_annotated_view(self) -> dict:
        """VIA-style annotated interface (arXiv:2607.11119).

        Keep the rendered frame paired with its numbered-object key. MCP
        attaches this image; other callers retain the saved trace path.
        """
        from dataclasses import replace

        from ..perception.visual_interface import VisualInterface, annotate_frame, configured_reach_x

        self.last_annotated_frame = None
        frame = self.observe_fresh()
        img, marks = annotate_frame(self)
        if img is None:
            raise SkillError("no frame available to annotate")
        self.last_annotated_frame = replace(frame, rgb=img)
        path = self.trace.save_keyframe(img, "annotated_view")
        if hasattr(self.camera, "set_overlay"):
            self.camera.set_overlay(status="annotated view")
        table_frame = declares_base_pose(self.cfg)
        if table_frame:
            # base_pose spec §3.5: the grid is the running arm's workspace,
            # drawn where it stands on the table; every number is table-frame.
            note = ("Objects carry numbered badges; grid values and positions are table-frame "
                    "coordinates in metres (X forward, Y left, Z up), the same for every arm on this rig, "
                    "not camera range.")
            frame_words = ("The grid covers this arm's workspace, drawn in the shared table frame "
                           "(+X forward, +Y left, +Z up).")
        else:
            note = (
                "Objects carry numbered badges; grid values are robot base-frame "
                "coordinates in metres (X forward, Y left, Z up), not camera range."
            )
            frame_words = "Grid coordinates use the robot base frame (+X forward, +Y left, +Z up)."
        if configured_reach_x(self.cfg) is not None:
            note += " The green band shows the explicitly configured top-down IK band."
        return {
            "key": VisualInterface.describe(marks, table_frame=table_frame),
            "objects": [m.as_dict() for m in marks],
            "image": path,
            "note": (
                "Badges describe tracked estimates, which may be stale or mislabeled. "
                "No badges does not mean the scene is empty. This refreshes the "
                "camera image, not object detection; use get_observation to refresh "
                f"tracking. {frame_words} Any shaded grasp band is a "
                "configured display hint, not an IK or grasp-reachability check."
            ),
        }

    def skill_open_live_view(self, reason: str = "") -> dict:
        """Open the browser dashboard on demand (it is closed by default).

        The chat client is the primary UI; this binds the HTTP port only when
        a human actually wants to look, and it auto-closes when idle.
        """
        if self.live_view is None:
            raise SkillError(
                "no live view configured in this runtime (headless/unit-test build)"
            )
        out = self.live_view.open(reason=reason)
        self.stream_server = self.live_view.server
        if not out.get("ok"):
            raise SkillError(str(out.get("error", "could not open the live view")))
        out["views"] = {
            "rgb": "detections + HUD (what the detector sees)",
            "depth": "depth colormap + range stats (what the geometry sees)",
            "agent": "numbered marks + metric grid + optional configured grasp band",
        }
        out["hint"] = (
            "Share the URL with the human. Each camera tile has rgb/depth/agent "
            "buttons, an analyze panel, and a chat box that drives this same robot."
        )
        return out

    def skill_close_live_view(self) -> dict:
        """Close the dashboard and release the port."""
        if self.live_view is None:
            return {"ok": True, "open": False, "note": "no live view configured"}
        out = self.live_view.close(reason="closed on request")
        self.stream_server = self.live_view.server
        return out

    def skill_live_view_status(self) -> dict:
        """Is the dashboard open, on what URL, and how idle is it?"""
        if self.live_view is None:
            return {"ok": True, "open": False, "mode": "unavailable"}
        out = dict(self.live_view.status())
        out["ok"] = True
        return out

    def skill_analyze_scene(self, camera: str | None = None) -> dict:
        """Detector and depth diagnostics, including unconfirmed object labels.

        Use an image-bearing observation for questions about visible objects.
        This report retains the raw estimates needed to diagnose perception.
        """
        rig = getattr(self, "rig", None)
        if rig is None:
            raise SkillError("no camera rig in this runtime")
        out: dict = {"ok": True, "cameras": {}}
        names = [camera] if camera else list(rig.names)
        for name in names:
            try:
                stream = rig.get(name)
            except KeyError:
                out["cameras"][name] = {"error": f"unknown camera {name!r}"}
                continue
            frame = stream.latest()
            if frame is None:
                out["cameras"][name] = {"error": "no frame yet"}
                continue
            dets, status = stream.overlay()
            entry: dict = {
                "fps": round(float(stream.fps), 1),
                "resolution": [int(frame.rgb.shape[1]), int(frame.rgb.shape[0])],
                "depth_source": frame.depth_source,
                "agent_status": status,
                "detections": [
                    {"label": getattr(d, "label", "?"),
                     "conf": round(float(getattr(d, "conf", 0.0)), 3)}
                    for d in dets
                ],
            }
            if frame.has_depth and frame.depth_m is not None:
                import os
                if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
                    from ..perception.cuda_math import depth_statistics
                    entry["depth"] = depth_statistics(frame.depth_m)
                else:
                    depth = np.asarray(frame.depth_m, dtype=np.float32)
                    valid = depth[np.isfinite(depth) & (depth > 0)]
                    entry["depth"] = {
                        "valid_fraction": round(float(valid.size) / float(depth.size), 3),
                        "min_m": round(float(valid.min()), 3) if valid.size else None,
                        "median_m": round(float(np.median(valid)), 3) if valid.size else None,
                        "max_m": round(float(valid.max()), 3) if valid.size else None,
                    }
            else:
                entry["depth"] = None
                entry["depth_warning"] = (
                    "no depth: 3D grounding will refuse to localize from this camera"
                )
            out["cameras"][name] = entry
        try:
            out["scene"] = self.skill_describe_scene()
        except Exception as e:
            out["scene"] = {"error": str(e)[:200]}
        try:
            from ..perception.visual_interface import VisualInterface, annotate_frame

            _, marks = annotate_frame(self)
            out["objects"] = [m.as_dict() for m in marks]
            out["key"] = VisualInterface.describe(marks, table_frame=declares_base_pose(self.cfg))
        except Exception:
            pass
        if self.live_view is not None:
            st = self.live_view.status()
            out["live_view"] = {"open": st["open"], "url": st["url"]}
            if not st["open"]:
                out["live_view"]["hint"] = (
                    "call open_live_view to watch this in a browser"
                )
        return out

    def skill_grasp_at_pixel(
        self, u: float, v: float, camera: str | None = None,
        normalized: bool = False, material: str | None = None,
    ) -> dict:
        """Grasp whatever is at this pixel, without needing its class name.

        VIA (arXiv:2607.11119) withholds perception APIs entirely and has the
        agent click in an RGB-D point cloud, on the reasoning that a frontier
        model can already SEE the object; what it lacks is a metric way to
        address it. Every other grasp skill here routes through
        `_localize(label)`, so the agent can only act on things the detector
        names, which is a measured ceiling: on LIBERO frames the detector
        emits 38-44 detections and never says `bowl`, while the bowl is
        plainly visible.

        The pixel is segmented (learned segmenter when configured, depth
        connectivity otherwise) and the OBB centre of that region becomes the
        grasp target. Using the probed surface point directly would grasp
        high: a ray hits the first surface, so the middle of a cube probes its
        top face (+22 mm on a 4.5 cm cube).

        Measured on a LIBERO frame against MuJoCo body poses, the segmenter is
        what makes this usable on a tabletop: depth connectivity alone fills
        the table (34 cm error), SAM2.1 with the same point prompt lands the
        bowl at 3.0 cm and the plate at 0.9 cm.

        Safety is unchanged: the same harness, workspace filter and
        postcondition checks apply, because this produces the same ObjectFix
        the detector path produces.
        """
        from ..perception.segmenter import fix_at_pixel

        self._refuse_on_another_arms_provisional("grasp_at_pixel")      # R5: before the frame is read
        self._reconcile_held()
        if self.held_object:
            raise SkillError(self._already_holding_refusal("grasp_at_pixel"))

        # The frame is analysed in the running arm's frame (its camera pose
        # T^-1 T_table_cam), so the fix and the grasp are arm targets.
        frame = self._frame_in_arm(self.observe())
        h, w = frame.rgb.shape[:2]
        uu, vv = float(u), float(v)
        if normalized or (0.0 <= uu <= 1.0 and 0.0 <= vv <= 1.0 and max(w, h) > 4):
            uu, vv = uu * (w - 1), vv * (h - 1)

        # Eye-in-hand cameras carry per-frame extrinsics; fall back to the
        # camera profile's static ones. Same rule as the detector path.
        T = (frame.T_base_cam if frame.T_base_cam is not None
             else self.extrinsics.cam_to_base())
        fix = fix_at_pixel(frame, T, int(round(uu)), int(round(vv)),
                           segmenter=self._segmenter)

        mask = fix.detection.mask
        # The workspace filter is in the table frame (every arm's base, D6).
        why = self._workspace.reject(
            self._to_world(fix.position), fix.extent,
            mask_frac=(float(mask.sum()) / float(w * h)) if mask is not None else 0.0,
        )
        if why:
            raise SkillError(
                f"the object at pixel ({int(uu)}, {int(vv)}) is not graspable: {why}"
            )
        return self.skill_grasp_object(
            fix.label, material=material, _fix=fix, _frame=frame,
        )

    def skill_probe_point(
        self, u: float, v: float, camera: str | None = None,
        normalized: bool = False,
    ) -> dict:
        """Queryable cursor: what is at this pixel, in metres.

        Anthropic's Claude-Plays-Robotics ablation found that static depth and
        segmentation OVERLAYS were roughly neutral, while a cursor the model
        can move and query lifted manipulation success from 6% to 32% -- the
        signal has to be a number you asked for, not a texture you must read.
        """
        from ..perception.probe import PointProbe

        try:
            return PointProbe(self).probe(u, v, camera=camera, normalized=normalized)
        except KeyError as e:
            raise SkillError(f"unknown camera {camera!r}") from e
        except ValueError as e:
            raise SkillError(str(e)) from e

    def skill_locate_pixel(self, label: str, camera: str | None = None) -> dict:
        """Where is a known object in the image? (inverse of probe_point)

        Lets you move the cursor to something you already track, and reason in
        the same pixel space you are looking at.
        """
        from ..perception.probe import PointProbe

        try:
            return PointProbe(self).locate_pixel(label, camera=camera)
        except KeyError as e:
            raise SkillError(f"unknown camera {camera!r}") from e
        except ValueError as e:
            raise SkillError(str(e)) from e

    def skill_task_done(self, success: bool, summary: str) -> dict:
        if isinstance(success, str):  # schema-lax backends send "false"
            success = success.strip().lower() in ("true", "yes", "1")
        unverified = self.unverified_actions()
        return {"ok": True, "task_complete": True,
                "success": bool(success) and not unverified, "summary": summary,
                "unverified": unverified, "task_effects": self.task_effects()}


def _short(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _normalize_outcome(result: dict, name: str) -> dict:
    """Stamp `outcome` (ok | failed | stuck) on a skill result, fail-closed.

    A result that says `outcome: "stuck"` is forced to `ok: false` (a stuck
    step is never a success) and must carry a non-empty human-readable
    `ask`; when a skill forgot the ask, one is derived from its error so the
    human always gets a request rather than a bare status. Any `verified:
    true` on a stuck result is a claim of effect and is withdrawn. Every
    other result reads `ok` or `failed` from its `ok` flag.
    """
    if not isinstance(result, dict):
        return result
    if result.get("outcome") == OUTCOME_STUCK:
        result["ok"] = False
        ask = result.get("ask")
        if not isinstance(ask, str) or not ask.strip():
            why = str(result.get("error") or f"{name} could not proceed")
            result["ask"] = (f"I am stuck on {name}: {why}. Change the scene or the "
                             "instruction and ask again.")
        if result.get("verified") is True:
            result["verified"] = False
        return result
    result["outcome"] = OUTCOME_OK if result.get("ok") else OUTCOME_FAILED
    return result


#: The first words of the instruction every failed result carries.
FAILED_INSTRUCTION = "This attempt FAILED"
#: Error text quoted in that instruction; the full text stays in `error`.
_FAILED_ERROR_CHARS = 240
#: Most room a hold refusal for unread jaws keeps after its sentence for
#: " Error: <driver text>" (`_unread_hold_refusal`; a shorter text keeps only
#: its own length): a longer label is shortened instead.
_REFUSAL_ERROR_ROOM = 40
#: pick_and_place's next_action for a completed routine. A failure (including a
#: refuted placement) gets the failure instructions instead, never this.
_PICK_REPORT_NEXT_ACTION = ("Report this result, including any return_home failure, and await the next "
                            "user order. Do not add another movement to this request.")
#: pick_and_place's guidance on a failed or refuted pick. On a result that grants
#: the sim retry (`_with_retry_text`) RETRY_ALLOWED stands in its place.
_PICK_FAILED_GUIDANCE = ("If the user said 'then stop', report this failure and end the turn. "
                         "Do not start another pick, rename the requested object or destination, "
                         "or invent a placement using manual coordinates.")
#: A failed grasp that did not withdraw (an observed-finger refusal returns
#: without the automatic home, which is not geometrically checked; a refused
#: grasp_object never goes home on a rig without return_home_after_motion)
#: leaves the arm where the attempt stopped, and the next grasp is refused at
#: the same point. Measured on the kitchen run: 3 identical refusals until
#: move_home.
_REFUSED_GRASP_LEFT_IN_PLACE = (
    "The arm was not moved back after the last failed grasp attempt: it stays where that attempt "
    "stopped, possibly at the refused grasp pose next to other objects: call move_home before any "
    "other grasp.")
#: The same step after an air grasp whose exit withdrawal COMPLETED (spec
#: §3.2 rule 1): the empty jaws were reopened and the tool lifted above the
#: object, so the arm is not at the refused grasp pose; the next grasp still
#: starts from home (pre-PR M5, PR-A jaws round).
_AIR_GRASP_WITHDREW_NOT_HOMED = (
    "The last grasp attempt closed on air; the jaws were reopened and the arm withdrew above the "
    "object, but it was not moved back: call move_home before any other grasp.")
#: ... and after an air grasp whose jaws were NOT confirmed open (refused
#: before the command, a jaw fault, or commanded but not confirmed): no
#: withdrawal ran, so the arm stays at the grasp hover. A jaw command may have
#: moved them, so the step never says they are closed (round 2, M2). A grasp
#: that starts with closed empty jaws opens them first (observed-finger
#: gate), as does reset_scene (PR-A jaws round, ruling 3).
_AIR_GRASP_JAWS_CLOSED = (
    "The last grasp attempt closed on air and the jaws were not confirmed open afterwards: they may "
    "still be closed (or only partly open), and the arm was left at the grasp hover above the object: "
    "call move_home before any other grasp; the next grasp (or reset_scene) opens empty closed jaws first.")
_MOVE_HOME_STEPS = (_REFUSED_GRASP_LEFT_IN_PLACE, _AIR_GRASP_WITHDREW_NOT_HOMED, _AIR_GRASP_JAWS_CLOSED)
#: Skills whose failure is a failed grasp the agent's next grasp would start from.
_GRASP_ENTRY_SKILLS = frozenset({"grasp_object", "grasp_at_pixel", "handover", "throw"})


def _withdrew_above_object(result) -> bool:
    """The failed grasp behind `result` ended with a COMPLETED air-grasp exit
    withdrawal: its own `exit_withdrawal`, or the `arm_withdrew_above_object`
    mark `_with_move_home_first` left on the grasp result a composite read."""
    if not isinstance(result, dict):
        return False
    return bool(result.get("arm_withdrew_above_object")) or str(
        result.get("exit_withdrawal") or "").startswith("completed")


def _jaws_left_closed(result) -> bool:
    """The failed grasp behind `result` closed on air and its jaws were NOT
    confirmed open: its own `jaws_reopen` was refused, or the `air_grasp_jaws_closed`
    mark `_with_move_home_first` left on the grasp result a composite read."""
    if not isinstance(result, dict):
        return False
    return bool(result.get("air_grasp_jaws_closed")) or str(
        result.get("jaws_reopen") or "").startswith("refused")


def _with_move_home_first(result: dict, source: dict | None = None) -> dict:
    """Put the move_home recovery step first in `next_action` of a result
    whose last grasp attempt moved the arm (a counted arm motion) and left it
    where it stopped, and mark it `arm_left_in_place`. `source`: the grasp
    result a composite built `result` from (default: `result` itself); when
    its exit withdrawal completed, the step says the jaws were reopened and
    the arm withdrew above the object and `result` is marked
    `arm_withdrew_above_object`; when its jaws were not confirmed open, the step
    says so and `result` is marked `air_grasp_jaws_closed`. Idempotent."""
    origin = result if source is None else source
    withdrew = _withdrew_above_object(origin)
    closed = not withdrew and _jaws_left_closed(origin)
    step = (_AIR_GRASP_WITHDREW_NOT_HOMED if withdrew
            else _AIR_GRASP_JAWS_CLOSED if closed else _REFUSED_GRASP_LEFT_IN_PLACE)
    own = str(result.get("next_action") or "").strip()
    if not any(known in own for known in _MOVE_HOME_STEPS):
        result["next_action"] = step + (f" {own}" if own else "")
    result["arm_left_in_place"] = True
    if withdrew:
        result["arm_withdrew_above_object"] = True
    if closed:
        result["air_grasp_jaws_closed"] = True
    return result


def _home_return_halted(home_return: dict) -> bool:
    """The operator's halt stopped a view-home return (status "halted") or
    its skip checks found one (status "skipped", "motion halted: ...")."""
    status = home_return.get("status")
    return status == "halted" or (status == "skipped"
                                  and str(home_return.get("reason") or "").startswith("motion halted: "))


def _halted_text(move: str, halt: str) -> str:
    """`move` stopped by the operator's halt, whose MotionHalted text is
    `halt` (a text that starts with "halted: " is said once)."""
    return f"{move} {halt}" if halt.startswith("halted: ") else f"{move} halted: {halt}"


def _stop_first(result: dict, stop: str) -> None:
    """Name the operator's stop that also ended a failed skill first in its
    `error`, then the skill's own error: outcome first, as
    `_reset_release_failed` does, because the agent's FAILED line keeps only
    the first `_FAILED_ERROR_CHARS` characters of the error. The skill's own
    error stays in `skill_error`: it broke before the stop, so that error,
    not the stop, files the call (`memory.envelope.failure_signature`)."""
    error = result.get("error")
    if error:
        result.setdefault("skill_error", error)
    result["error"] = f"{stop}; {error}" if error else stop


def _with_retrace_result(result: dict, exc: BaseException) -> None:
    """A move_home that retraced after a STOP and then failed in its home
    move (a halt, an e-stop, a fault): its failed `result` (execute()'s, or
    reset_scene's when the reset called it) keeps what the retrace did and
    what the jaws hold (`stop_retrace`, `holding` or `holding_unknown`,
    `next_action`), which skill_move_home attached to the exception (review
    M2, M4)."""
    kept = getattr(exc, "retrace_result", None)
    if isinstance(kept, dict):
        result.update(kept)


def _keep_decided_failure(exc: BaseException, failure: str | None) -> None:
    """An operator halt that stopped the recovery or the retry after a
    failure the skill had already decided (an air grasp's reopen or exit, the
    pre-close retreat after a failed descent or a native Grasp that missed,
    the next candidate after a withdrawn veto, a persistence loop's re-home or
    next attempt after a miss, turn_screw's retreat after its strokes fell
    short or its disengage after a stroke that did not settle, restore_scene's
    later objects after one failed) carries that
    failure as `skill_error`, the way skill_move_home attaches
    `retrace_result`. execute() keeps it in the failed result, so the call is
    filed as that failure (`memory.envelope.failure_signature`), not as the
    halt; the error stays the halt's own text. Only a halt: an e-stop is filed
    as the e-stop. The innermost failure wins (an attempt's own air grasp over
    the loop's earlier miss)."""
    if isinstance(exc, MotionHalted) and failure and getattr(exc, "skill_error", None) is None:
        exc.skill_error = str(failure)


def _with_skill_error(result: dict, exc: BaseException) -> None:
    """A skill a halt ended after its own decided failure
    (`_keep_decided_failure`): the failed result keeps that failure in
    `skill_error`, as `_stop_first` does for a halted park or return."""
    failure = getattr(exc, "skill_error", None)
    if failure:
        result.setdefault("skill_error", failure)


def _restore_failures(failed: list[dict]) -> str | None:
    """restore_scene's summary of the objects it could not restore (up to
    three): its error when nothing else stopped it, and the failure a later
    halt carries (`skill_error`). None when no object failed."""
    return "; ".join(f"{f['label']}: {str(f.get('error', '?'))[:60]}" for f in failed[:3]) or None


def _left_in_place(result: dict) -> bool:
    """A grasp result marked by `_with_move_home_first`: the composites pass
    the move_home step on. `home_skipped` alone is not enough: a grasp that
    moved no arm (perception refusal, a refused route, a SkillError before
    motion) also skips the automatic home, but the arm is where it was."""
    return bool(result.get("arm_left_in_place"))


def _with_failure_instruction(result: dict) -> dict:
    """Prepend the unconditional failure instruction to a failed result.

    Every result whose ``ok`` is falsy (a refused or failed call, a self-
    reported success the postcondition REFUTED, and ok=None/0 the way
    `_normalize_outcome` reads them) starts its ``next_action`` with what
    happened and the rule to report it as a failure. Measured on the
    OpenClaw kitchen run: a refuted pick_and_place carried only the "if the
    user said 'then stop'" guidance and the agent answered "Done". A skill's
    own guidance follows unchanged; a success is untouched. Idempotent.
    """
    if not isinstance(result, dict) or result.get("ok"):
        return result
    own = str(result.get("next_action") or "").strip()
    if FAILED_INSTRUCTION in own:
        return result
    error = " ".join(str(result.get("error") or "the call reported ok=false").split())
    if len(error) > _FAILED_ERROR_CHARS:
        error = error[:_FAILED_ERROR_CHARS].rstrip() + " (truncated; see error)"
    # One full stop ends the quoted error; a cut marker ("...", a skill that
    # shortened its own text) already ends it and is kept.
    end = error if error.endswith("...") else error.rstrip(".") + "."
    line = (f"{FAILED_INSTRUCTION}: {end} Tell the user it failed; never report a "
            "failed or refuted result as success.")
    result["next_action"] = line + (f" {own}" if own else "")
    return result


#: The kitchen rules' retry trigger (A8 R1: repeat a motion order once only when ITS
#: result says SIMULATION ONLY and allows one more attempt; the note lint's
#: `rule_trigger` words): only a retry the ledger granted may carry it (A8 fix,
#: Important 3). Read as a reader reads it (`_trigger_reading`; A8 fix 2): in any
#: case, across any separator, also between its letters ("Simulation-only", "ONE  more
#: attempt", "S I M U L A T I O N  O N L Y", a zero-width or other format character
#: inside a word), with the digit 0 for the letter o.
_RETRY_TRIGGER_PHRASES = ("simulation only", "one more attempt", "another attempt", "second attempt")
#: R5's trigger words (A8 follow-ups 5). R5 itself keys on the result's top-level `learning_line`
#: FIELD, never on words (A9/A10 fix: no word list can tell every paraphrase of a reply line from the
#: agent's own text, so that residual is accepted; only `execute()` writes the field). Its words (the
#: note lint's R5 `_RULE_TRIGGERS`, "end*" as end / ends / ending) and the field's own name stay
#: withheld from EVERY result, a granted retry's too, read the same way as R1's, as defence in depth:
#: an operator's halt reason (the agent's own `halt_motion(reason)`), an object name or a key the
#: agent's names make never carries them. Unlike R1's, they need no word start ("Pleaseend your
#: reply", "thisexact line" are read; "send your reply" is withheld too: accepted over-block).
_REPLY_LINE_PHRASES = ("end your reply", "ends your reply", "ending your reply", "end your answer",
                       "ends your answer", "ending your answer", "exact line", "exactly that line",
                       "exactly this line", "learning line")
#: letters a reader takes for another in a trigger phrase: the digit 0 for o, 3 for e, 1 (or a
#: capital I) for l, and i and l for each other (A9/A10 fix: "rep1y", "repIy", "3nd y0ur r3ply")
_TRIGGER_LETTER_CLASSES = {"o": "[o0]", "e": "[e3]", "l": "[l1i]", "i": "[i1l]"}


def _phrases_pattern(phrases) -> str:
    """The phrases as one alternation: any run of separators between letters, the
    `_TRIGGER_LETTER_CLASSES` look-alike digits and letters."""
    return "|".join(r"[\W_]*".join(_TRIGGER_LETTER_CLASSES.get(ch, ch) for ch in phrase.replace(" ", ""))
                    for phrase in phrases)


#: R1's phrases only from a word start (the note lint's "in simulation, only" reading); R5's anywhere
_RULE_TRIGGER_RE = re.compile(r"(?:(?<![a-z0-9])(?P<retry>" + _phrases_pattern(_RETRY_TRIGGER_PHRASES)
                              + r"))|(?P<reply>" + _phrases_pattern(_REPLY_LINE_PHRASES) + ")", re.IGNORECASE)
#: R5's words alone: what a granted retry's result has withheld (it keeps R1's permission)
_REPLY_LINE_RE = re.compile(r"(?P<reply>" + _phrases_pattern(_REPLY_LINE_PHRASES) + ")", re.IGNORECASE)
#: what stands in a trigger's place: R1's in a result the ledger did not grant, R5's in any result
_RETRY_TRIGGER_WITHHELD = "(retry wording withheld)"
_REPLY_LINE_WITHHELD = "(reply-line wording withheld)"
_WITHHELD = {"retry": _RETRY_TRIGGER_WITHHELD, "reply": _REPLY_LINE_WITHHELD}
#: look-alikes of the triggers' letters (a c d e g h i l m n o p r s t u w x y), read as those
#: letters: Cyrillic, Greek, Armenian, Cherokee (capital and small), Latin small capitals
#: (U+1D00-U+1D2B and the IPA ones) and other Latin letters, from Unicode's confusables.txt
#: (security data, single-letter targets), plus Greek nu read as n (A9/A10 fix). NFKD has already
#: folded fullwidth, mathematical and other compatibility forms.
_TRIGGER_LOOKALIKE_LETTERS = {
    "a": "\u0251\u0391\u03b1\u0410\u0430\u13aa\u1d00\uab64\uab7a",
    "c": "\u03f2\u03f9\u0421\u0441\u13df\u1c83\u1d04\uabaf",
    "d": "\u0501\u13a0\u13e7\u1d05\uab70\uabb7\U0001018b",
    "e": "\u0395\u03b5\u0415\u0435\u04bd\u13ac\u1d07\uab32\uab7c",
    "g": "\u018d\u0261\u0262\u050c\u0581\u13c0\u13f3\u13fb\u1d83\uab90",
    "h": "\u029c\u0397\u041d\u04ba\u04bb\u0570\u13bb\u13c2\uab8b\uab92",
    "i": "\u0131\u0269\u026a\u0399\u03b9\u0406\u0456\u0582\u13a5\ua647\uab75",
    "l": "\u0196\u01c0\u029f\u04c0\u04cf\u13de\ua781\ua7ae\ua7fe\uabae\U0001d22a",
    "m": "\u039c\u03fa\u041c\u043c\u13b7\u1d0d\uab87",
    "n": "\u0274\u039d\u03bd\u0578\u057c",
    "o": "\u039f\u03bf\u03c3\u041e\u043e\u0555\u0585\u1c82\u1d0f\u1d11\uab3d",
    "p": "\u00fe\u01bf\u03a1\u03c1\u03f1\u03f8\u0420\u0440\u13e2\u1d18\uabb2",
    "r": "\u01a6\u024c\u0280\u0433\u13a1\u13d2\u1d26\uab47\uab48\uab71\uab81\uaba2\U0001d216",
    "s": "\u01bd\u0405\u0455\u054f\u13d5\u13da\ua731\uaba5\uabaa",
    "t": "\u03a4\u03c4\u0422\u0442\u13a2\u1d1b\uab72",
    "u": "\u028b\u03c5\u054d\u057d\u1d1c\ua79f\uab4e\uab52",
    "w": "\u026f\u0448\u0461\u051c\u051d\u0561\u13b3\u13d4\u1d21\ua7fa\uab83\uaba4\U0001d20c",
    "x": "\u03a7\u0425\u0445\u04b2\u04b3\ua7b3",
    "y": "\u0263\u028f\u03a5\u03b3\u03d2\u0423\u0443\u04ae\u04af\u13a9\u13bd\u1d8c\u1eff\uab5a\uab79\uab8d",
}
_TRIGGER_LOOKALIKES = str.maketrans({ch: letter for letter, chars in _TRIGGER_LOOKALIKE_LETTERS.items()
                                     for ch in chars})
#: escapes a model decodes, read decoded (A9/A10 fix): HTML character references, %xx (a UTF-8
#: run), \xHH, \uHHHH, \UHHHHHHHH and \n \r \t written out
_TRIGGER_ESCAPE_RE = re.compile(r"&#[0-9]{1,7};?|&#[xX][0-9a-fA-F]{1,6};?|&[A-Za-z][A-Za-z0-9]{1,31};"
                                r"|(?:%[0-9A-Fa-f]{2})+|\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}"
                                r"|\\[nrt]")
#: the characters every escape starts with: an ASCII string without any is read as written
_TRIGGER_ESCAPE_STARTS = re.compile(r"[&%\\]")
#: the trigger squeezed (no separators, lower case, the `_TRIGGER_LETTER_CLASSES` digits and
#: letters folded alike): a cheap first look, a superset of `_RULE_TRIGGER_RE` on bytes
_TRIGGER_SQUEEZE = bytes.maketrans(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ013i", b"abcdefghljklmnopqrstuvwxyzolel")
_TRIGGER_SEPARATORS = bytes(byte for byte in range(128) if not chr(byte).isalnum())
_TRIGGER_SQUEEZED = tuple(phrase.replace(" ", "").encode().translate(_TRIGGER_SQUEEZE)
                          for phrase in _RETRY_TRIGGER_PHRASES + _REPLY_LINE_PHRASES)
_REPLY_LINE_SQUEEZED = tuple(phrase.replace(" ", "").encode().translate(_TRIGGER_SQUEEZE)
                             for phrase in _REPLY_LINE_PHRASES)


def _decoded_escape(token: str) -> str | None:
    """What a `_TRIGGER_ESCAPE_RE` token reads as, or None (not a character)."""
    import html

    try:
        if token.startswith("&"):
            decoded = html.unescape(token if token.endswith(";") else token + ";")
            return None if decoded == token else decoded
        if token.startswith("%"):
            return bytes.fromhex(token.replace("%", "")).decode("utf-8", "replace")
        if token[1] in "nrt":
            return " "
        code = int(token[2:], 16)
        return chr(code) if code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF else None
    except (ValueError, OverflowError):
        return None


def _trigger_units(text: str) -> list[tuple[str, int, int]]:
    """`text` as it is read, unit by unit: (reading, start, end) with text[start:end] the characters
    it was read from. Escapes are decoded first (up to three nested levels, `%2520`), then each
    character is read on its own: NFKD, combining marks dropped (an empty reading), look-alike
    letters as Latin letters (`_TRIGGER_LOOKALIKES`). Format characters stay: they are separators to
    `_RULE_TRIGGER_RE` like any other non-letter."""
    chars = [(ch, i, i + 1) for i, ch in enumerate(text)]
    for _ in range(3):
        joined = "".join(ch for ch, _, _ in chars)
        out, last, changed = [], 0, False
        for match in _TRIGGER_ESCAPE_RE.finditer(joined):
            decoded = _decoded_escape(match.group(0))
            if decoded is None:
                continue
            start, end = chars[match.start()][1], chars[match.end() - 1][2]
            out += chars[last:match.start()] + [(ch, start, end) for ch in decoded]
            last, changed = match.end(), True
        if not changed:
            break
        chars = out + chars[last:]
    units = []
    for ch, start, end in chars:
        if ch.isascii():
            units.append((ch, start, end))
            continue
        decomposed = unicodedata.normalize("NFKD", ch)
        kept = "".join(c for c in decomposed if c.isascii() or unicodedata.category(c) not in ("Mn", "Me"))
        units.append((kept.translate(_TRIGGER_LOOKALIKES), start, end))
    return units


def _trigger_reading(text: str) -> str:
    """`text` as the trigger is read (`_trigger_units`, joined)."""
    return "".join(reading for reading, _, _ in _trigger_units(text))


def _may_carry_trigger(reading: str, phrases: tuple = _TRIGGER_SQUEEZED) -> bool:
    """A cheap first look (C-level byte ops): False only when `reading` cannot match
    `_RULE_TRIGGER_RE` (its letters, squeezed and folded, contain no trigger phrase; `phrases`:
    the squeezed phrases of the pattern in use)."""
    squeezed = reading.encode("ascii", "ignore").translate(_TRIGGER_SQUEEZE, _TRIGGER_SEPARATORS)
    return any(phrase in squeezed for phrase in phrases)


def _withheld(match: re.Match) -> str:
    """What stands in the place of a trigger `match` (R1's or R5's)."""
    return _WITHHELD[match.lastgroup]


def _withhold_spans(text: str, pattern: re.Pattern = _RULE_TRIGGER_RE) -> str:
    """`text` with each span that reads as a trigger of `pattern` replaced by its placeholder
    (`_withheld`; the rest kept as written: other scripts, accents, numbers, escapes), read unit by
    unit (`_trigger_units`) so every match maps back to the characters it was read from; the marks
    that only decorate a match's last letter go with it."""
    units = _trigger_units(text)
    origin = [k for k, (reading, _, _) in enumerate(units) for _ in reading]
    out, last = [], 0
    for match in pattern.finditer("".join(reading for reading, _, _ in units)):
        first, final = origin[match.start()], origin[match.end() - 1]
        start, end = units[first][1], units[final][2]
        while final + 1 < len(units) and not units[final + 1][0]:
            final += 1
            end = units[final][2]
        out += [text[last:start], _withheld(match)]  # a span two matches share is withheld once
        last = max(last, end)
    out.append(text[last:])
    return "".join(out)


def _without_rule_triggers(value, pattern: re.Pattern = _RULE_TRIGGER_RE, phrases: tuple = _TRIGGER_SQUEEZED):
    """`value` with every span that reads as a rule trigger (`_RULE_TRIGGER_RE` on
    `_trigger_units`: R1's retry permission, R5's reply line and the `learning_line` field's name)
    withheld from every string in it: dicts (their string keys too: a failure model or a grasp prior
    is keyed by the agent's own object name), lists and tuples are walked, and a container is copied
    only where a string changed (shared objects are never edited). Two keys that come out alike keep
    the FIRST one's value; the collision is logged, never a silent overwrite (A9/A10 fix, Minor 6).
    A plain-ASCII string with no escape character (& % \\) skips the Unicode reading (A8 fix 2, Minor
    6: no mark, compatibility form, look-alike or escape can be in it); a string without a trigger
    comes back as it was, the same object, and so does any other value. `pattern` / `phrases`:
    `_without_reply_line`'s."""
    if isinstance(value, str):
        if value.isascii() and not _TRIGGER_ESCAPE_STARTS.search(value):
            if not _may_carry_trigger(value, phrases) or not pattern.search(value):
                return value
            return pattern.sub(_withheld, value)
        reading = _trigger_reading(value)
        if not _may_carry_trigger(reading, phrases) or not pattern.search(reading):
            return value
        return _withhold_spans(value, pattern)
    if isinstance(value, dict):
        out, changed = {}, False
        for key, item in value.items():
            new_key = _without_rule_triggers(key, pattern, phrases) if isinstance(key, str) else key
            new = _without_rule_triggers(item, pattern, phrases)
            changed = changed or new_key is not key or new is not item
            if new_key in out:
                logger.warning("rule-trigger scrub: key %r would collide with an earlier key %r; kept the first",
                               key, new_key)
                changed = True
                continue
            out[new_key] = new
        return out if changed else value
    if isinstance(value, (list, tuple)):
        items = [_without_rule_triggers(item, pattern, phrases) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return items if isinstance(value, list) else tuple(items)
    return value


def _without_reply_line(value):
    """`value` with R5's reply line withheld (`_without_rule_triggers` with R5's words
    alone): what a granted retry's result gets, so it keeps R1's permission."""
    return _without_rule_triggers(value, _REPLY_LINE_RE, _REPLY_LINE_SQUEEZED)


def _with_retry_text(result: dict, skill: str) -> dict:
    """Spec §4.5's agent text on a result the sim retry ledger marked
    (`sim_retry`): a granted retry puts RETRY_ALLOWED where the failure
    guidance stood (only pick_and_place's results carry it; else appended), a used one appends
    RETRY_USED, a stopped one RETRY_STOPPED_NEXT. The FAILED line and any move_home step stay
    in front. The single choke point of the rules' triggers: every result execute() returns
    passes here, with or without a ledger, and only a granted one keeps R1's retry trigger --
    every other has it withheld from all its text (`_without_rule_triggers`: an operator's halt
    reason, an error, a nested step's reason); R5's reply line and the `learning_line` field's name
    are withheld from every result, a granted one too (`_without_reply_line`; A8 follow-ups 5), in
    its keys as well (A9/A10 fix: a top-level key the agent's words made never reads as the field;
    two keys that come out alike keep the first). Edits `result` in place. Idempotent."""
    from ..agent.sim_retry import RETRY_ALLOWED, RETRY_STOPPED_NEXT, RETRY_USED

    if not isinstance(result, dict):
        return result
    state = result.get("sim_retry")
    state = state if isinstance(state, dict) else {}
    own = str(result.get("next_action") or "").strip()
    if state.get("allowed") is True:
        text = RETRY_ALLOWED.format(skill=skill)
        if text not in own:
            if _PICK_FAILED_GUIDANCE in own:
                own = own.replace(_PICK_FAILED_GUIDANCE, text)
            else:
                own = f"{own} {text}" if own else text
            result["next_action"] = own
        return _scrubbed_in_place(result, _without_reply_line)
    added = (RETRY_STOPPED_NEXT if state.get("stopped") is True
             else RETRY_USED if state.get("used") is True else None)
    if added is not None and added not in own:
        result["next_action"] = f"{own} {added}" if own else added
    return _scrubbed_in_place(result, _without_rule_triggers)


def _scrubbed_in_place(result: dict, scrub) -> dict:
    """`result` with `scrub` applied to its values and its string keys, edited in place (the same
    object: callers and the trace hold it); a dict whose keys all stay is never rebuilt."""
    for key, value in list(result.items()):
        if (new := scrub(value)) is not value:
            result[key] = new
    if any(isinstance(key, str) and scrub(key) is not key for key in result):
        rebuilt = scrub(result)
        result.clear()
        result.update(rebuilt)
    return result


def _stuck_ask(object: str, last_err: str, attempts: int, *, stage: str = "grasp",
               destination: str | None = None, holding: bool = False) -> str:
    """The human-actionable request for a persistence loop that gave up.

    Keyed on the terminal reason the loop recorded, each ask names what to
    change in the SCENE or the INSTRUCTION -- never "try again" to the
    planner, which is what `error`/`suggestion` already are for. Every ask
    ends with "ask again." so the human knows the robot is waiting.
    """
    err = str(last_err or "").lower()
    obj = f"{object!r}" if object else "the object"
    n = max(int(attempts), 1)
    if _POSSIBLE_HOLD_STOP in err:
        return _POSSIBLE_HOLD_ASK
    if _is_gripper_fault(err):
        return (f"My gripper stopped itself while handling {obj}: "
                + ("something is between the jaws, so I cannot find their closed position. "
                   if "between the jaws" in err else
                   "the jaws were blocked while opening, so they hold where they stopped. ")
                + "Clear the jaws (nothing between or against them), then ask again.")
    if _is_contact(err):
        doing = "placing" if stage == "place" else "reaching for"
        held = " I am still holding it." if holding else ""
        return (f"Something blocked my arm while I was {doing} {obj}: it hit an obstacle my "
                f"cameras did not see, stopped and is holding where it was blocked.{held} "
                f"Clear the space around {obj} and around the gripper camera, say 'reset' "
                f"(reset_scene) so I forget that spot, and ask again.")
    if stage == "place":
        dest = f"on/at {destination}" if destination else "at the destination"
        if holding:
            return (f"I am still holding {obj} but could not place it {dest} after {n} "
                    f"attempt(s). Clear or move the destination (or give me explicit "
                    f"coordinates with place_at) and ask again.")
        return (f"{obj} slipped out of my gripper while I was carrying it {dest} and my "
                f"grasp budget is spent. Put it back upright with space around it and ask again.")
    if "never seen" in err or "no detections" in err or "not found" in err or "cannot find" in err:
        return (f"I cannot see {obj} on the table after {n} scan(s). Place it where the "
                f"camera can see it, or tell me its exact colour/name, and ask again.")
    if "> gripper max" in err or "wider than the jaws" in err or "too wide" in err:
        return (f"{obj} is wider than my jaws can open. Replace it with a narrower object, "
                f"or ask me to push it instead, and ask again.")
    if "ik failed" in err or "unreachable" in err or "out of reach" in err or "outside" in err:
        return (f"{obj} is out of my reach. Move it closer to me, towards the centre of "
                f"the table, and ask again.")
    if "closed on air" in err or "air grasp" in err or "slipped" in err or "not held" in err:
        return (f"I could not hold {obj} after {n} grasp attempt(s): the jaws closed on air "
                f"or it slipped. Stand it upright with space around it, or ask for a "
                f"different object, and ask again.")
    return (f"I could not pick up {obj} after {n} attempt(s) and my persistence budget is "
            f"spent. Reposition it (upright, in view, within reach) or change the "
            f"instruction, and ask again.")


def _is_gripper_fault(error) -> bool:
    """The gripper refused or stopped for safety (control/rs_gripper.py)."""
    return "gripperfault" in str(error or "").lower()


def _is_contact(error) -> bool:
    """The failure was the contact monitor stopping a blocked arm."""
    return "contact detected" in str(error or "").lower()


def _contact_result(arm) -> dict:
    """What a ContactDetected adds to a skill result: the trip record and
    an instruction for the planner (never retry into the same spot)."""
    harness = getattr(arm, "harness", None)
    event = dict(getattr(harness, "last_contact", None) or {})
    contact = {k: event[k] for k in ("joint", "error_rad", "raw_error_rad", "tol_rad",
                                     "duration_ms", "phase", "motion", "stall", "hold_q")
               if k in event}
    zones = getattr(harness, "contact_zones", None) or []
    if zones:
        zone = zones[-1]
        contact["zone"] = {"center": [round(float(v), 4) for v in zone["center"]],
                           "radius_m": zone["radius_m"]}
    joint = contact.get("joint", "?")
    return {"contact": contact, "hint": (
        f"The arm hit something at joint {joint} that the cameras did not see; it stopped and "
        "holds where it was blocked (torque on). Do not retry the same motion or re-plan to the "
        "same spot: tell the user, ask them to clear the obstacle, then call reset_scene "
        "(that also lifts the fence around the spot).")}


def _is_human_curable(stop_reason: str | None) -> bool:
    """A persistence stop reason a HUMAN can act on. The e-stop and an
    unfinished contact episode are harness/recovery states, not requests
    to the human: they stay plain failures."""
    if stop_reason is None:
        return True  # exhausted by attempts or by the clock
    reason = stop_reason.lower()
    return not ("e-stop" in reason or "contact episode" in reason)


#: Spellings of "the configured drop zone" a planner sends as `destination`.
#: Measured on a real chat turn: the model read `"destination": "drop zone"`
#: in a previous result and echoed it back; pick_and_place then tried to
#: LOCALIZE an object called "drop zone", failed 8 times, and the planner
#: spent four minutes inventing coordinates. The drop zone is a configured
#: point, not a detected object -- these names mean "use it".
_DROP_ZONE_WORDS = {"drop zone", "dropzone", "drop-zone", "drop_zone", "the drop zone",
                    "bin", "the bin", "default", "zona de descarga",
                    "green square", "the green square", "green target square",
                    "the green target square", "cuadrado verde", "el cuadrado verde",
                    "delivery area", "the delivery area", "delivery zone", "the delivery zone",
                    "green delivery area", "the green delivery area",
                    "green delivery zone", "the green delivery zone",
                    "green zone", "the green zone", "delivery square", "the delivery square"}

_OPEN_BOX_WORDS = {"open box", "the open box", "open_box", "beige box", "the beige box",
                   "box", "the box", "storage box", "the storage box", "wooden box", "the wooden box"}


def _names_drop_zone(destination) -> bool:
    return str(destination or "").strip().lower() in _DROP_ZONE_WORDS


TOOL_SPECS: list[dict] = [
    {
        "name": "grasp_at_pixel",
        "description": (
            "Grasp whatever is at this pixel, WITHOUT needing to name it. Use "
            "this when you can see the object in the image but grasp_object "
            "fails with 'no detections': the detector's vocabulary does not "
            "have to contain the object for this to work. The pixel is "
            "segmented by depth and the object's centre is computed from that "
            "region, so point anywhere on the object body (avoid edges and "
            "shadows). Same safety checks as grasp_object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "u": {"type": "number", "description": "Column in pixels, or 0..1 if normalized."},
                "v": {"type": "number", "description": "Row in pixels, or 0..1 if normalized."},
                "normalized": {
                    "type": "boolean",
                    "description": "True if u,v are fractions of image size.",
                },
                "material": {
                    "type": "string",
                    "description": "Optional material hint for grasp force.",
                },
            },
            "required": ["u", "v"],
        },
    },
    {
        "name": "halt_motion",
        "description": (
            "Stop the motion currently in flight because it is no longer the "
            "right action (wrong object, subgoal already satisfied, the scene "
            "changed). Takes effect within ~20 ms. This is NOT an emergency "
            "stop: it does not latch, it clears when the next motion starts, "
            "and the arm stays powered, so recovery is just issuing the "
            "corrected command. Use 'stop' if the rig is actually unsafe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this motion is being abandoned.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_observation",
        "description": "Capture a fresh camera frame and update object tracking. MCP returns that image with frame freshness, configured destinations and robot status. Identify visible objects from the image. Use localize_object for measured object positions; configured zones do not establish visibility. Inspect before acting when identity or presence is uncertain.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_objects",
        "description": "List tracked objects, including remembered ones with last-known positions and age. This is not exhaustive and names can be approximate. A missing name does not prove absence: refresh with get_observation or actively search with localize_object/pick_and_place.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "annotated_view",
        "description": (
            "Render the annotated camera view: tracked positions projected inside the image "
            "get numbered badges; entries outside the image are flagged in the key. "
            "With a 5 cm base-frame grid (+X forward, +Y left, +Z up) and any explicitly configured display band. "
            "MCP returns the annotated image and a text key mapping each number to its "
            "tracked label and 3D position. Badges are tracked estimates; this refreshes "
            "the camera frame, not object detection. Use get_observation to refresh tracking. "
            "Use it for spatial context. Workspace bounds and display bands do not "
            "establish an IK solution or validate a grasp."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "probe_point",
        "description": (
            "CURSOR: point at a pixel in the camera image and get hard numbers back -- "
            "distance in metres, the 3D position in the robot's base frame, which tracked "
            "object is near that point, configured workspace membership, "
            "and the offset from the gripper position. "
            "Use it whenever you need spatial precision: before a grasp on a cluttered "
            "or ambiguous scene, or to measure a visible surface. It does not solve "
            "inverse kinematics or establish that an object can be grasped. "
            "Coordinates may be pixels or 0..1 normalized."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "u": {"type": "number", "description": "x pixel (or 0..1 if normalized)"},
                "v": {"type": "number", "description": "y pixel (or 0..1 if normalized)"},
                "camera": {"type": "string", "description": "camera name; omit for the manipulation camera"},
                "normalized": {"type": "boolean", "description": "treat u,v as 0..1 fractions"},
            },
            "required": ["u", "v"],
        },
    },
    {
        "name": "locate_pixel",
        "description": (
            "Inverse of probe_point: given an object the robot already tracks, return "
            "WHERE IT IS IN THE IMAGE (pixel + normalized coords) plus its distance and "
            "whether it is currently in view. Use it to aim the cursor at a known object, "
            "or to check an object is visible in a given camera before acting on it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "camera": {"type": "string"},
            },
            "required": ["label"],
        },
    },
    {
        "name": "analyze_scene",
        "description": (
            "Detector and depth diagnostics without opening a window. Returns "
            "per-camera detector estimates, confidence, depth quality and a numbered "
            "object key. Labels are unconfirmed hypotheses, including in the generated "
            "description; confidence does not establish object identity. For 'what "
            "are you seeing?' or any visible-scene question, use describe_scene or "
            "get_observation to inspect a fresh image. This diagnostic report does "
            "not return an image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {"type": "string",
                           "description": "one camera name; omit for all"},
            },
            "required": [],
        },
    },
    {
        "name": "open_live_view",
        "description": (
            "Open the browser dashboard for the human (it is CLOSED by default -- chat "
            "is the interface). Returns a URL to share. The page shows all cameras "
            "together with per-tile rgb / depth / agent view switches, an analyze panel "
            "(detections + depth stats + description), the world model, robot narration, "
            "and a chat box that drives this same robot. Auto-closes when nobody is "
            "watching. Call this when the human asks to see/watch the cameras."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string",
                           "description": "why it is being opened (shown in logs)"},
            },
            "required": [],
        },
    },
    {
        "name": "close_live_view",
        "description": (
            "Close the browser dashboard and release its port. Perception keeps "
            "running; only the HTTP view stops. Call it when the human is done looking."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "live_view_status",
        "description": (
            "Whether the browser dashboard is currently open, its URL, and how long it "
            "has been idle. Check before offering a link so you never hand out a dead URL."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "localize_object",
        "description": "Localize an object from camera depth; returns base-frame position and size in meters. Exact configured-zone names and delivery-area aliases return the configured XY center with configuration provenance, not a detected object or visibility claim. Use spatial_hint to disambiguate objects; clarify if the requested identity remains uncertain.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "spatial_hint": {"type": "string", "enum": ["left", "right", "front", "back", "near", "far"]},
            },
            "required": ["label"],
        },
    },
    {
        "name": "preview_grasp",
        "description": "Plan a grasp on a named object and REPORT the proposed gripper waypoint (position, approach direction, confidence, learned-memory prior) WITHOUT moving. Use it to check a grasp looks right before committing, then call grasp_object. Cheap observe-before-act step.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "spatial_hint": {"type": "string", "enum": ["left", "right", "front", "back", "near", "far"]},
            },
            "required": ["label"],
        },
    },
    {
        "name": "grasp_object",
        "description": "Full grasp pipeline on a named object: localize, plan a top-down grasp, approach, material-aware close, lift, verify. Returns failure details if the object was missed.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "material": {
                    "type": "string",
                    "enum": ["rigid", "fragile", "soft", "deformable", "slippery", "heavy"],
                    "description": "How gently to grip; infer from the object's looks.",
                },
                "spatial_hint": {"type": "string", "enum": ["left", "right", "front", "back", "near", "far"]},
            },
            "required": ["label"],
        },
    },
    {
        "name": "pick_and_place",
        "description": (
            "FAST PATH: complete pick-and-place in ONE call. Resolves the "
            "object against the live world model (color queries like 'pink "
            "object' work), grasps, places on the named destination object "
            "(pass 'green square' for the kitchen's marked green square or "
            "'open box' for its calibrated beige box; "
            "omit `destination` or pass 'drop zone' for the configured placement "
            "point), returns home, and reports stage timings. `destination` puts "
            "the object ON/IN that object. For 'next to / beside / near X' do not "
            "use this tool: localize_object X and the object (position, extent_m), "
            "grasp_object the object, then place_at X's position offset by about "
            "1.5x their average extent_m toward a free side. The result "
            "includes return_home: do not call move_home again after this "
            "tool unless its next_action asks for it. When the user says "
            "'then stop', report the result and "
            "end the turn without any further movement tool, including when "
            "the result is a failure. Do not retry under a guessed name or "
            "invent a manual destination after that failure. "
            "Both stages PERSIST: they keep retrying "
            "with fresh perception and fresh grasp plans until they succeed "
            "or the persistence budget runs out. Prefer this over manual "
            "localize/grasp/place for any 'pick X [put it in Y]' request. "
            "Actively searches the camera for the requested name even when "
            "that exact label is absent from list_objects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object": {"type": "string", "description": "what to pick, e.g. 'pink object', 'red mug'"},
                "destination": {"type": "string", "description": "'green square' for the marked kitchen target; 'open box' for the calibrated beige box; another named object to place ON/IN; omit for the configured placement point"},
                "material": {
                    "type": "string",
                    "enum": ["rigid", "fragile", "soft", "deformable", "slippery", "heavy"],
                },
            },
            "required": ["object"],
        },
    },
    {
        "name": "place_at",
        "description": "Place the held object at base-frame coordinates (meters). Omit z to release just above the table.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "place_on_object",
        "description": "Place the held object on/in another named object (e.g. a bowl or plate).",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    },
    {
        "name": "push_object",
        "description": "Push a named object along the table (for objects too wide to grasp, or to reposition). distance_m defaults to 0.08.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "direction": {"type": "string", "enum": ["forward", "back", "left", "right"]},
                "distance_m": {"type": "number"},
            },
            "required": ["label", "direction"],
        },
    },
    {
        "name": "describe_scene",
        "description": "Inspect the current scene without moving. MCP returns a fresh camera image with frame freshness, configured destinations and robot status. Identify visible objects and distinguish marked delivery areas from movable objects using the image. Use localize_object for measured object positions. Configured destinations do not establish visibility or occupancy.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "count_objects",
        "description": "Count known objects, optionally filtered by a query like 'red' or 'cube' or 'pink object'. Instant, from the live world model.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "point_at",
        "description": "Point at a named object (hover the gripper above it for a moment). Great for answering 'which one is the pink object?'.",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    },
    {
        "name": "wave",
        "description": "Wave at the audience (a friendly greeting gesture around the home pose).",
        "parameters": {
            "type": "object",
            "properties": {"cycles": {"type": "integer", "minimum": 1, "maximum": 4}},
            "required": [],
        },
    },
    {
        "name": "handover",
        "description": "Hand an object to the human: grasp it (if not already held), present it at the handover pose, and keep holding until open_gripper is called. Use for 'hand me / give me the X'.",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "sort_by_color",
        "description": "Sort every known object into per-color zones along the table edge (repeated pick-and-place; takes a while). A crowd favorite.",
        "parameters": {
            "type": "object",
            "properties": {"max_objects": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": [],
        },
    },
    {
        "name": "throw",
        "description": "Grab an object (if not already held) and throw it by winding up and releasing at the top of a harness-vetted swing. direction is forward/left/right/back (default forward). Use for 'grab the banana and throw it' / 'launch the X'. Never at a person: a throw toward a side where visitors may stand is refused (safety.people.visitor_sides, every side unless the rig declares one free of visitors).",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": ["forward", "left", "right", "back"],
                },
            },
            "required": [],
        },
    },
    {
        "name": "turn_screw",
        "description": (
            "Tighten or loosen a screw, bolt, nut or knob by ratcheting the "
            "wrist roll: the jaws engage the head, turn through the wrist's "
            "free travel, release, counter-rotate and re-engage until the "
            "requested wrist travel is commanded. Without an admitted independent "
            "fastener observer the physical result is unverified (ok=false); "
            "execution_ok reports whether the wrist routine completed. It does not "
            "verify threading, seating or torque. tighten = clockwise from "
            "above (right-hand thread). Use for 'tighten the screw' / "
            "'unscrew the bolt' / 'loosen the knob two turns'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "The fastener, e.g. 'screw', 'wing nut', 'bottle cap'."},
                "direction": {
                    "type": "string",
                    "enum": ["tighten", "loosen"],
                },
                "turns": {
                    "type": "number",
                    "description": "Full revolutions to apply (0.05-6, default 1).",
                },
            },
            "required": ["label"],
        },
    },
    {
        "name": "rotate_joint",
        "description": "Rotate exactly one joint by a signed RELATIVE angle in degrees. Joint numbers start at 1. Maximum 90 degrees per command. Never substitute a grasp, reset, or Cartesian motion, and never retry a failed relative rotation.",
        "parameters": {
            "type": "object",
            "properties": {
                "joint": {"type": "integer", "minimum": 1},
                "degrees": {"type": "number", "minimum": -90, "maximum": 90},
            },
            "required": ["joint", "degrees"],
        },
    },
    {
        "name": "move_relative",
        "description": (
            "Move the gripper -- and anything it holds -- a short straight distance, then report "
            "exactly how the request was interpreted. Use for vague nudges. Examples: 'lift it up' "
            "-> direction=up; 'a bit to the left' -> direction=left, amount=small; never toward a "
            "person: toward_viewer ('bring it toward me') is refused (to give someone an object, use "
            "handover); 'push it away from me' -> direction=away_from_viewer; "
            "'move it right on the screen' -> direction=right, frame=viewer; 'a bit more' / 'again' "
            "/ 'keep going' -> direction=again (same direction; add amount to change the size); "
            "'3 cm left and 2 cm up' -> vector=[0, 0.03, 0.02]. SIZE: amount tiny=1 cm ('a hair', "
            "'a tiny bit'), small=2 cm ('a bit', 'slightly', 'a little'), medium=5 cm (default), "
            "large=10 cm ('a lot', 'much'); or distance_m for an exact number; at most 0.15 m per "
            "call. FRAME: left/right/forward/back mean the ROBOT's own directions (frame=base, the "
            "default: forward = away from the robot base, left = the robot's left) unless the "
            "person refers to themselves or the screen ('my left', 'right on the screen', 'as I "
            "see it') -> frame=viewer. toward_viewer/away_from_viewer always use the screen "
            "camera. in/out move along the gripper's pointing direction. up/down are always "
            "vertical. 'back' is a direction, NOT undo: for 'go back to where it was' / 'undo "
            "that' call undo_move. While holding an object it moves slower, never opens or "
            "closes the jaws, and keeps the object's bottom off the table. The rig's configured "
            "wrist-camera orientation and volume checks also apply. Tell the person the "
            "`interpretation` (and `as_seen_on_screen`). If refused (ok=false), the arm did NOT "
            "move: relay the reason; if `feasible_distance_m` is given, offer that smaller move."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "back", "left", "right", "up", "down",
                             "toward_viewer", "away_from_viewer", "in", "out", "again"],
                    "description": "Named direction (omit when giving `vector`). 'again' repeats the last move_relative.",
                },
                "amount": {"type": "string", "enum": ["tiny", "small", "medium", "large"],
                           "description": "Vague size: tiny 1 cm, small 2 cm, medium 5 cm (default), large 10 cm."},
                "distance_m": {"type": "number",
                               "description": "Exact distance in metres (0.005-0.15). Overrides amount."},
                "frame": {"type": "string", "enum": ["base", "viewer", "tool"],
                          "description": ("base = the robot's own directions (default); viewer = as seen on "
                                          "the screen camera; tool = along the gripper (in/out).")},
                "vector": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                           "description": ("Free displacement in metres in `frame`: base [x forward, y left, "
                                           "z up]; viewer [right, away, up]; tool [approach, jaw axis, third]. "
                                           "Length 0.005-0.15 m. Use instead of direction/amount.")},
            },
            "required": [],
        },
    },
    {
        "name": "rotate_gripper",
        "description": (
            "Small orientation nudge of the gripper about its tip (and of anything it holds): "
            "ROLL the wrist (direction=clockwise / counterclockwise, as seen looking along the "
            "jaws; for a gripper pointing down, as seen from above) or TILT it so the jaws point "
            "more toward a direction (direction=up / down / left / right / forward / back in the "
            "robot frame, or toward_viewer / away_from_viewer / frame=viewer for the screen). "
            "Examples: 'rotate it a bit' -> clockwise, amount=small; 'twist the other way' -> "
            "counterclockwise; 'tilt it down a little' -> down, amount=small; never at a person: "
            "toward_viewer ('point it toward me') is refused; 'a bit more' -> direction=again. SIZE: "
            "amount tiny=5, small=10, "
            "medium=15 (default), large=30 deg; or angle_deg 1-30 (at most 30 deg per call). "
            "SAFETY: obey the rig's configured wrist-camera orientation and clearance limits; "
            "never offer to turn the gripper upside down. Camera volume checks apply only when "
            "the profile declares that geometry. While holding an object it turns slower and "
            "never opens the jaws. If "
            "refused (ok=false), the arm did NOT move; relay the reason and, if given, offer "
            "`feasible_angle_deg`. 'undo_move' turns it back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["clockwise", "counterclockwise", "up", "down", "left", "right",
                             "forward", "back", "toward_viewer", "away_from_viewer", "again"],
                    "description": "clockwise/counterclockwise = roll; any other word = tilt toward it.",
                },
                "amount": {"type": "string", "enum": ["tiny", "small", "medium", "large"],
                           "description": "Vague size: tiny 5, small 10, medium 15 (default), large 30 degrees."},
                "angle_deg": {"type": "number", "description": "Exact angle in degrees (1-30). Overrides amount."},
                "frame": {"type": "string", "enum": ["base", "viewer"],
                          "description": "For tilts only: base = robot directions (default); viewer = as seen on the screen."},
            },
            "required": ["direction"],
        },
    },
    {
        "name": "move_near_object",
        "description": (
            "Move the gripper -- and anything it holds -- to a position RELATIVE TO A NAMED "
            "OBJECT or the table, without grasping or releasing anything. relation=above: hover "
            "over it ('hold it above the green spool' -> label='green spool', relation=above); "
            "just_above: barely over it ('lower it until it's just above the table' -> "
            "label='table', relation=just_above); next_to: beside it ('hold it next to the cup' "
            "-> next_to; side=left/right/forward/back or away_from_viewer with frame=viewer picks "
            "the side, default the side nearest the gripper; toward_viewer is refused: never toward "
            "a person); closer: move "
            "toward it ('bring it closer to the box' -> closer, amount/distance_m as in "
            "move_relative; it stops before touching). Heights use the target's measured top; "
            "when holding something, the held object's bottom (not the gripper) is kept "
            "clearance_m above the target's top (default 5 cm for above, 1.5 cm for just_above). "
            "It climbs first and descends last, at most 0.5 m per call. Use place_at / "
            "place_on_object to actually put something down, this only positions. If refused "
            "(ok=false) the arm did NOT move; relay the reason. undo_move goes back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string",
                          "description": "The reference object, e.g. 'green spool', or 'table'."},
                "relation": {"type": "string", "enum": ["above", "just_above", "next_to", "closer"]},
                "clearance_m": {"type": "number",
                                "description": "Gap to keep (0.005-0.15 m): vertical for above/just_above, edge to edge for next_to."},
                "side": {"type": "string",
                         "enum": ["nearest", "left", "right", "forward", "back", "toward_viewer", "away_from_viewer"],
                         "description": "next_to only: which side of the object (default nearest)."},
                "frame": {"type": "string", "enum": ["base", "viewer"],
                          "description": "Frame of `side`: base = robot directions (default), viewer = as seen on the screen."},
                "amount": {"type": "string", "enum": ["tiny", "small", "medium", "large"],
                           "description": "closer only: 1 / 2 / 5 (default) / 10 cm."},
                "distance_m": {"type": "number", "description": "closer only: exact distance (0.005-0.15 m)."},
            },
            "required": ["label", "relation"],
        },
    },
    {
        "name": "undo_move",
        "description": (
            "Go back: return the gripper (and anything it holds) to where it was before the last "
            "move_relative / rotate_gripper / move_near_object, retracing the same path backwards. "
            "Use for 'go back', 'undo that', 'put it back where it was a moment ago', 'no, the "
            "other way' (after undoing, move the other way). steps=2..5 undoes several moves. "
            "It only works right after those moves: if anything else moved the arm since (a "
            "grasp, place, move_home...) it refuses and forgets the history. The path back is "
            "safety-checked first; if refused the arm does not move. Not for 'move back' in the "
            "sense of 'toward the robot base' -- that is move_relative direction=back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {"type": "integer", "minimum": 1, "maximum": 5,
                          "description": "How many recorded moves to undo, newest first (default 1)."},
            },
            "required": [],
        },
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper (drops the held object where it is; also how a handover finishes).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "close_gripper",
        "description": "Close the gripper with the default profile.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_home",
        "description": (
            "Return the arm to its configured INITIAL (ready) pose. Reobserve after moving "
            "an eye-in-hand camera. On a rig with a separate rest pose this means raised, "
            "motors on, gripper forward, "
            "ready for the next task -- what the user means by 'initial pose', 'ready pose', "
            "'start pose', 'starting position' (Spanish: 'pose inicial', 'posición inicial', "
            "'vuelve al inicio'). Also clears the camera view. 'home', 'go home', 'pose 0', "
            "'zero', 'rest', 'park', 'fold' or 'so I can turn it off' are NOT this tool: they "
            "mean the folded rest pose -- call move_to_rest."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_to_rest",
        "description": (
            "Fold the arm onto its REST pose and switch the motors off, so the robot can be "
            "powered down. This is what the user means by 'home', 'go home', 'back home', "
            "'pose 0', 'zero pose', 'back to zero', 'rest', 'park', 'fold it', 'so I can turn "
            "it off' (Spanish: 'vuelve a home', 'a casa', 'pose 0 / pose cero', 'reposo', "
            "'aparca', 'pliega el brazo'). The arm lies folded back over its base (every "
            "joint 0, wrist camera on top); once it is MEASURED there the gripper closes and "
            "the motors are released. NOT the raised initial/ready pose -- that is move_home. "
            "If the gripper holds something it refuses: ask the user to put it down first. It "
            "also refuses while the e-stop is latched. If the arm does not reach the rest "
            "pose the motors stay ON (a raised arm without torque would fall) and the result "
            "says why. Say the robot can be switched off ONLY when the result has "
            "safe_to_power_off=true. keep_torque=true folds it but keeps the motors on. The "
            "next motion command switches the motors on again by itself. An arm with no "
            "folded rest pose goes to its initial pose instead (safe_to_power_off=false)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keep_torque": {"type": "boolean",
                                "description": "Keep the motors on at the rest pose (default "
                                               "false: they are switched off)."},
            },
            "required": [],
        },
    },
    {
        "name": "reset_scene",
        "description": (
            "Start over between demos: arm home, every simulated prop back on "
            "its spawn pose, the world model and the task memory cleared, one "
            "fresh observation. On a real robot the props do not move back "
            "(props_reset is empty): ask the human to replace them. On a real "
            "robot it also refuses while the gripper holds an object, before "
            "any motion (opening at home would drop it): place it first "
            "(place_at), then reset. Not a "
            "recovery tool for a failed grasp -- use it when a task or a "
            "visitor is done."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "snapshot_scene",
        "description": (
            "Memorize the current layout under a name: the confirmed (currently "
            "visible) objects' labels, colours and positions, kept in the world "
            "model as ADVISORY data for restore_scene. No motion. Use before a "
            "task that rearranges the table ('memorize the scene', then later "
            "'put everything back'). The positions are future targets, not a "
            "claim about where anything is afterwards."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "snapshot name (default 'scene')"},
            },
            "required": [],
        },
    },
    {
        "name": "restore_scene",
        "description": (
            "Put the table back the way snapshot_scene memorized it: only objects "
            "displaced beyond tolerance_m move, an object sitting on another's "
            "remembered spot moves first (blocker-first; a swap parks one object "
            "on free table), each move is a harness-vetted grasp + place_at, and "
            "the number of moves and the task's persistence budget bound it. "
            "Read the postcondition: the snapshot is advisory and 'restored' is "
            "only true when an independent channel confirms it. Reports what "
            "moved, what was left alone, what is missing (use search_for_object) "
            "and what failed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "snapshot name (default 'scene')"},
                "tolerance_m": {"type": "number",
                                "description": "displacement that counts as moved (0.02-0.15, default 0.05)"},
                "max_moves": {"type": "integer", "minimum": 1, "maximum": 8,
                              "description": "cap on pick-and-place moves (default 6)"},
            },
            "required": [],
        },
    },
    {
        "name": "search_for_object",
        "description": (
            "Occlusion search for a named object that is not visible: lifts the "
            "largest hollow/large thing that could be hiding it, parks it ~0.2 m "
            "away on free table inside the workspace, re-perceives, and repeats "
            "for up to max_occluders. If the object was visible already, nothing "
            "moves. Finding it is NOT the task: the result has task_complete=false "
            "and tells you to resume the ORIGINAL task on the found object. Not "
            "found -> ok=false with stuck=true: report that and ask the human; do "
            "not guess or grasp a look-alike."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "the object to find, e.g. 'red cube'"},
                "max_occluders": {"type": "integer", "minimum": 1, "maximum": 4,
                                  "description": "how many occluders to lift at most (default 3)"},
            },
            "required": ["label"],
        },
    },
    {
        "name": "recall_memory",
        "description": "Recall recent events (last ~15 s) and, optionally, where a named object was last seen.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "object name to look up"}},
            "required": [],
        },
    },
    {
        "name": "recall_step",
        "description": (
            "Look back at ONE executed step of this run: its skill, arguments, "
            "outcome (ok / failed / stuck with the human-facing ask), the "
            "independent postcondition verdict recorded at the time, which "
            "dispatch tier issued it, and its BEFORE/AFTER keyframe images "
            "(attached as images where the host can show them). Read-only: "
            "moves nothing and re-verifies nothing. `n` indexes the recorded "
            "steps: 0 = the first, negative counts from the end (-1 = the last "
            "action; recalls themselves are not steps). An index that does not "
            "exist is an explicit error, never an old frame. Use it to check "
            "WHAT a past step did and what the scene looked like around it "
            "before deciding whether to redo, undo or move on."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Step index: 0 = first recorded step, -1 = most recent (default).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_arms",
        "description": (
            "List the robot arms available and which one motion commands use "
            "by default. Read-only and safe to call at any time. Call this "
            "before using the `arm` parameter, so you use a real name: on a "
            "single-arm rig it reports one arm and `arm` can be omitted "
            "entirely."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "task_done",
        "description": "Declare the task finished (success or honest failure) with a one-paragraph summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    },
]


#: Every motion skill takes an optional `arm` naming which robot runs it.
#: Injected here rather than written into 15 hand-maintained schemas: the
#: dispatch side already derives the arm from _MOTION_SKILLS (execute()
#: pops `arm` before calling the skill), so a hand-copied list would be a
#: second source of truth that drifts the moment a motion skill is added.
#:
#: Single-arm runs are unaffected: the parameter is optional, omitting it
#: means the primary arm, and `_select_arm` rejects a name when no rig is
#: wired rather than silently driving the only arm there is.
for _spec in TOOL_SPECS:
    if _spec["name"] in _MOTION_SKILLS:
        _spec["parameters"]["properties"]["arm"] = {
            "type": "string",
            "description": (
                "Which arm runs this. Omit for the manipulation arm (the "
                "default, and the only one on a single-arm rig). Use "
                "list_arms to see the names."
            ),
        }
del _spec


#: Rule R2 of the ASPIRE demo port (spec §4.6): appended to `open_gripper`'s
#: description on a sim-class profile that offers place_at (the OpenShell route);
#: a real robot's catalogue stays main's, byte for byte (spec §6.3)
_SET_DOWN_WITH_PLACE_AT = ("To set a held object down, use place_at (it lowers the object before opening). "
                           "open_gripper releases the object where it is, in the air if the arm is raised.")


def tool_specs_for_config(cfg=None):
    """Independent schemas for the selected rig; never mutate the shared catalog."""
    import copy

    from .. import sim_profile
    from ..robot_context import arm_profiles, declares_base_pose

    specs = copy.deepcopy(TOOL_SPECS)
    if cfg is None:
        return specs
    arms = arm_profiles(cfg)
    table_frame = declares_base_pose(cfg)
    cameras = cfg.get("cameras") or [cfg.camera]
    camera_names = [c.get("name") for c in cameras if c.get("name")]
    unsupported = set.intersection(*(set(a.get("unsupported_skills", [])) for a in arms))
    specs = [s for s in specs if s["name"] not in unsupported]
    home_arms = [a.get("name", "primary") for a in arms if a.get("return_home_after_motion")]
    summary = "; ".join(f"{a.get('name', 'primary')}: {a.get('display_name', a.get('name', 'arm'))}, "
                        f"{a['n_joints']}-DOF, TCP {a.get('ee_frame')}" for a in arms)
    set_down = (any(s["name"] == "place_at" for s in specs)
                and sim_profile.profile_class(cfg) == "sim")
    for spec in specs:
        if set_down and spec["name"] == "open_gripper":
            spec["description"] += " " + _SET_DOWN_WITH_PLACE_AT
        if home_arms and len(home_arms) == len(arms) and spec["name"] == "move_home":
            spec["description"] = (
                "Return the arm to the saved operator camera-view home, preserving the gripper "
                "state. This is the overview pose meant by 'home' on this rig. The wrist "
                "camera moves with the arm; reobserve afterward. STOP, cancellation and "
                "hardware/safety faults prevent the return."
            )
        if home_arms and len(home_arms) == len(arms) and spec["name"] == "move_to_rest":
            spec["description"] = (
                "Move to a declared folded rest pose if the arm has one. Without a configured "
                "park_q, return to the saved camera-view home while preserving the gripper "
                "state and keeping the motors on; safe_to_power_off remains false. Never "
                "infer a zero-joint rest pose or claim torque was released."
            )
        parameters = spec["parameters"]
        parameters["additionalProperties"] = False
        for field, prop in parameters.get("properties", {}).items():
            if field == "camera" and camera_names:
                prop["enum"] = camera_names
            if field in {"label", "object", "query"} and prop.get("type") == "string":
                prop.update(minLength=1, pattern=r"^[^<>]+$")
        if spec["name"] in _MOTION_SKILLS or spec["name"] in {"list_arms", "get_observation"}:
            spec["description"] += " Configured arms: " + summary + "."
        if spec["name"] in _MOTION_SKILLS - {"move_home", "move_to_rest", "reset_scene"}:
            if home_arms:
                spec["description"] += (
                    " On " + ", ".join(home_arms)
                    + ", this top-level action returns to the saved operator camera-view home, "
                    "including ordinary grasp failures. Composite pick/place phases stay together. "
                    "STOP, cancellation and hardware/safety faults suppress the return. "
                    "Read return_home for the actual return status; any reported action pose "
                    "was reached before that return."
                )
        if spec["name"] == "rotate_joint":
            spec["parameters"]["properties"]["joint"]["maximum"] = max(int(a['n_joints']) for a in arms)
        if spec["name"] in {"open_gripper", "close_gripper", "grasp_object", "grasp_at_pixel", "preview_grasp"}:
            for arm in arms:
                grip = arm.get("gripper", {})
                spec["description"] += (f" {arm.get('name', 'primary')} gripper: "
                    f"{grip.get('name', 'profile gripper')}, maximum aperture {grip.get('max_width_m')} m. "
                    + str(grip.get("position_description", "")))
        if spec["name"] == "move_relative":
            spec["description"] += (
                " Bare directions default to the robot base frame; use an explicit frame "
                "for viewer- or tool-relative requests."
            )
        if table_frame and spec["name"] in TABLE_FRAME_TOOLS:
            _say_table_frame(spec)
    return specs


#: The coordinate tools: their descriptions name the frame of the positions
#: they take or return (base_pose spec §3.5, D7).
TABLE_FRAME_TOOLS: frozenset[str] = frozenset({"localize_object", "place_at", "annotated_view", "probe_point"})
#: Their base-frame wording and its table-frame replacement, on a rig where
#: any arm declares a base_pose (the agent's coordinates are table-frame, Q1 a).
_BASE_FRAME_WORDS = (("base-frame grid", "workspace grid drawn in the table frame"),
                     ("base-frame", "table-frame"), ("the robot's base frame", "the shared table frame"),
                     ("the robot base frame", "the shared table frame"))
_TABLE_FRAME_SENTENCE = " Coordinates are in the shared table frame: the same numbers for every arm on this rig."


def _say_table_frame(spec: dict) -> None:
    """A coordinate tool's description in table-frame words, plus the one
    sentence that says so (base_pose spec §3.5)."""
    text = spec["description"]
    for old, new in _BASE_FRAME_WORDS:
        text = text.replace(old, new)
    spec["description"] = text + _TABLE_FRAME_SENTENCE
