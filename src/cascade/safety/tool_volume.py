"""The gripper's own volume as occupancy clearance samples.

The harness used to check the occupancy map at the TCP and the joint origins
only. On the reBot the wrist camera (a padded D455F on TOP of the gripper:
8 cm up, 14 cm back from the fingertips, 15 cm across the jaws) and the open
jaws are where the arm is widest, and neither is near any of those points: on
2026-10-11 they swept into clutter beside a grasp target and pressed for ~8 s.

`ToolVolume` holds a fixed grid of sample points in the TCP frame (at most
`spacing_m` apart, faces and interior included, so an obstacle touching or
inside a box lies within ~spacing/sqrt(2) of a sample). `points(T_tcp)` moves
them to a pose. The harness checks them at every vetted pose and every
streamed step (`SafetyHarness._tool_volume_violation`).

The geometry is not duplicated here: `build_tool_volume` reads the camera box
from `relative_motion.wrist_camera` with the spatial tools' own parser and
`relative_motion.camera_box` (one box, two layers), and the jaws from the
open-jaw sweep `grasp.graspgenx.sweep.extents_open/offset_open` (the RS URDF
finger geometry). An arm opts in with `safety.tool_volume`.

Measured parts (spec 2026-10-16 nvblox object-aware §3.4, §8): the optional
`housing` and `fingers` boxes (the URDF `gripper_end` hull and both fingers
over the full stroke) and a camera box cut at its calipered front / bottom
(`relative_motion.wrist_camera.front_m` / `bottom_m`). They ship together
with the owner's calipers; until then the reBot samples today's slab, which
reaches the fingertip plane and the finger axis and so covers the fingers'
and the housing's top. `camera_faces_from_calipers` turns the calipers into
the box (every face outward to the next 5 mm, a missing number keeps
today's face); `uncovered_camera_faces` is the shipped-config check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..types import SkillError

if TYPE_CHECKING:
    from ..skills.spatial_moves import CameraDims

# The camera parser and box live with the spatial tools (one box, two
# layers). They are imported where they are used, never at module level: the
# safety layer must import without the skills package (its runtime).

#: allowed keys of `safety.tool_volume`
_KEYS = {"wrist_camera", "jaws", "housing", "fingers", "spacing_m", "table_band_m"}
#: largest |coordinate| of a part box (m), the camera box's limit: beyond it is a typo
_MAX_PART_M = 0.30
#: today's wrist-camera slab (relative_motion.wrist_camera in configs/demo.yaml):
#: 14 cm back from the fingertip plane, 15 cm across, 8 cm up from the finger
#: axis; `TODAY_SLAB` (module attribute, a spatial_moves.CameraDims) is built
#: from these on first use
_TODAY_SLAB = (0.08, 0.14, 0.075, 0.0, 0.0)
#: the caliper faces (mm; reference frame in spec §8 and the committed record
#: docs/evidence/wrist-camera-calipers/measurement.json)
CALIPER_FACES = ("front_behind_tip", "rear_behind_tip", "minus_y", "plus_y", "bottom_above_axis",
                 "top_above_axis")
#: every measured face moves outward to this grid (mm)
_CALIPER_GRID_MM = 5


@dataclass(frozen=True)
class ToolVolume:
    """Clearance samples rigidly attached to the TCP.

    `local` (N, 3) sample coordinates in the TCP frame; `part` (N,) index into
    `names` (what a refusal names); `table_band_m`: occupied voxels at most
    this far above the table are the table itself, which the harness's table
    rule governs -- a box that reaches the fingertip plane would otherwise be
    refused by the table under every low grasp. `boxes`: each part's box
    (centre, axes, half) in the TCP frame, in `names` order, the boxes
    `local` samples (point_at samples their faces, `pointing.tool_body`);
    empty for a volume built from points alone.
    """

    local: np.ndarray
    part: np.ndarray
    names: tuple
    spacing_m: float = 0.02
    table_band_m: float = 0.02
    boxes: tuple = ()

    def points(self, T_tcp: np.ndarray) -> np.ndarray:
        T = np.asarray(T_tcp, dtype=float)
        return self.local @ T[:3, :3].T + T[:3, 3]


def __getattr__(name: str):
    if name == "TODAY_SLAB":
        from ..skills.spatial_moves import CameraDims

        return CameraDims(*_TODAY_SLAB)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def box_grid(center, axes, half, spacing_m: float) -> np.ndarray:
    """A grid over an oriented box (faces and interior), at most `spacing_m`
    apart along each box axis. `axes` columns are the box axes."""
    grids = []
    for h in np.asarray(half, dtype=float):
        n = max(1, math.ceil(2.0 * float(h) / spacing_m - 1e-9))
        grids.append(np.linspace(-float(h), float(h), n + 1) if h > 0 else np.zeros(1))
    local = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.asarray(center, dtype=float) + local @ np.asarray(axes, dtype=float).T


def jaw_box(sweep, axis_order: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The open-jaw sweep box in the TCP frame: (center, axes, half).

    `sweep` is `grasp.graspgenx.sweep` (`extents_open`, optional
    `offset_open`), given in the sweep frame: origin at the TCP, +X closing,
    +Y third, +Z approach. `axes` columns map those onto the TCP frame of
    `axis_order` (`tool_axis_order`). The one reading of the jaws that the
    harness's tool volume (`build_tool_volume`) and point_at's pointing pose
    both use, so the two cannot drift apart. A missing, non-numeric, negative
    or non-finite box, or an unknown axis order, raises ValueError.
    """
    from ..skills.relative_motion import _TOOL_COLUMNS

    raw_ext = sweep.get("extents_open") if sweep is not None else None
    raw_off = sweep.get("offset_open", [0.0, 0.0, 0.0]) if sweep is not None else None
    try:
        ext = np.asarray(raw_ext, dtype=float).reshape(3)
        off = np.asarray(raw_off, dtype=float).reshape(3)
    except (TypeError, ValueError):
        ext = off = np.full(3, np.nan)
    if not (np.all(np.isfinite(ext)) and np.all(ext >= 0) and np.all(np.isfinite(off))):
        raise ValueError(f"grasp.graspgenx.sweep open jaws must be finite, got {raw_ext!r}, {raw_off!r}")
    if axis_order not in _TOOL_COLUMNS:
        raise ValueError(f"unknown tool_axis_order {axis_order!r}")
    a_col, j_col, t_col = _TOOL_COLUMNS[axis_order]
    # sweep frame: +X closing, +Y third, +Z approach (origin at the TCP)
    axes = np.eye(3)[:, [j_col, t_col, a_col]]
    return axes @ off, axes, ext / 2.0


def _flag(block, key: str) -> bool:
    value = block.get(key, True)
    if not isinstance(value, bool):
        raise ValueError(f"safety.tool_volume.{key} must be true or false, got {value!r}")  # noqa: TRY004 -- a config error
    return value


def _length(block, key: str, default: float, lo: float, hi: float) -> float:
    raw = block.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if isinstance(raw, bool) or not (math.isfinite(value) and lo < value <= hi):
        raise ValueError(f"safety.tool_volume.{key} must be in ({lo}, {hi}] m, got {raw!r}")
    return value


def _part_box(block, key: str):
    """`safety.tool_volume.<key>: {min: [a, j, u], max: [a, j, u]}` (m) as
    (lo, hi) in (approach, jaw, third) TCP coordinates -- the reBot's third
    axis is its camera up -- or None: absent or null = the part is off."""
    raw = block.get(key)
    if raw is None:
        return None
    where = f"safety.tool_volume.{key}"
    if not isinstance(raw, dict) or set(raw) != {"min", "max"}:
        raise ValueError(f"{where} must be {{min: [approach, jaw, up], max: [approach, jaw, up]}} "
                         f"in m, or null; got {raw!r}")
    ends = []
    for end in ("min", "max"):
        v = raw[end]
        if (not isinstance(v, (list, tuple)) or len(v) != 3
                or any(isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating))
                       for x in v)):
            raise ValueError(f"{where}.{end} must be 3 numbers (m), got {v!r}")
        ends.append(np.asarray(v, dtype=float))
    lo, hi = ends
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo < hi)
            and np.all(np.abs(lo) <= _MAX_PART_M) and np.all(np.abs(hi) <= _MAX_PART_M)):
        raise ValueError(f"{where} must be finite, within +-{_MAX_PART_M:g} m and min < max on every "
                         f"axis, got {raw!r}")
    return lo, hi


def build_tool_volume(view, arm_cfg) -> ToolVolume | None:
    """The `ToolVolume` an arm profile asks for, or None (not opted in).

    `view` is the arm's resolved config (safety, relative_motion, grasp);
    `arm_cfg` its profile (`tool_axis_order`). A malformed block raises
    ValueError: a safety volume that silently turns itself off is worse than
    none. So does a camera box cut at its front without the `fingers` part
    or at its bottom without the `housing` part: today's slab is what
    samples them (spec §3.4.4).
    """
    safety = view.get("safety") if hasattr(view, "get") else None
    block = safety.get("tool_volume") if safety is not None and hasattr(safety, "get") else None
    if block is None:
        return None
    if hasattr(block, "as_dict"):
        block = block.as_dict()
    if not isinstance(block, dict):
        raise ValueError(f"safety.tool_volume must be a mapping, got {block!r}")  # noqa: TRY004 -- a config error
    unknown = sorted(set(block) - _KEYS)
    if unknown:
        raise ValueError(f"safety.tool_volume has unknown key(s) {unknown}; allowed {sorted(_KEYS)}")
    spacing = _length(block, "spacing_m", 0.02, 0.0, 0.05)
    band = _length(block, "table_band_m", 0.02, -1e-12, 0.05)
    use_camera, use_jaws = _flag(block, "wrist_camera"), _flag(block, "jaws")
    housing, fingers = _part_box(block, "housing"), _part_box(block, "fingers")
    if not (use_camera or use_jaws or housing is not None or fingers is not None):
        raise ValueError("safety.tool_volume samples nothing (wrist_camera and jaws both false, no "
                         "housing or fingers); remove the block instead")
    from ..skills import relative_motion as rm
    from ..skills.spatial_moves import _tool_up, wrist_camera_dims

    order = str(arm_cfg.get("tool_axis_order", "down_open"))
    a_col, j_col, t_col = rm._TOOL_COLUMNS[order]
    eye = np.eye(3)
    for name, part in (("housing", housing), ("fingers", fingers)):
        # a part box is (approach, jaw, camera-up): only an arm whose camera
        # up IS +third tool column can read it without turning it over
        if part is not None:
            try:
                up = _tool_up(view.get("relative_motion") or {})
            except SkillError as exc:
                raise ValueError(f"safety.tool_volume.{name}: {exc}") from None
            if up is None or not np.allclose(up, eye[t_col], rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"safety.tool_volume.{name} is given in (approach, jaw, camera-up) TCP coordinates and "
                    f"needs relative_motion.tool_up_axis = +third tool column {eye[t_col].tolist()} "
                    f"(tool_axis_order {order!r}), got {None if up is None else list(up)}")
    parts, labels, names, boxes = [], [], [], []
    if use_camera:
        dims, up = wrist_camera_dims(view.get("relative_motion"))
        if dims is None:
            raise ValueError("safety.tool_volume.wrist_camera needs relative_motion.wrist_camera and "
                             "relative_motion.tool_up_axis (both are null here)")
        # the slab reaches the fingertip plane and the finger axis: a cut box
        # needs the parts it covered (spec §3.4.4)
        if dims.front_m > 0.0 and fingers is None:
            raise ValueError("relative_motion.wrist_camera.front_m > 0 needs safety.tool_volume.fingers: "
                             "dropping the slab front leaves the open fingers unsampled")
        if dims.bottom_m > 0.0 and housing is None:
            raise ValueError("relative_motion.wrist_camera.bottom_m > 0 needs safety.tool_volume.housing: "
                             "raising the slab bottom leaves the housing unsampled")
        # the same function the spatial tools plan with, at T = identity:
        # the box in the TCP frame
        box = rm.camera_box(np.eye(4), eye[a_col], up, protrusion_m=dims.protrusion_m,
                            behind_m=dims.behind_m, half_width_m=dims.half_width_m,
                            front_m=dims.front_m, bottom_m=dims.bottom_m)
        parts.append(box_grid(box.center, box.axes, box.half, spacing))
        names.append("wrist camera")
        boxes.append((np.asarray(box.center, dtype=float), np.asarray(box.axes, dtype=float),
                      np.asarray(box.half, dtype=float)))
    tool_axes = np.column_stack([eye[a_col], eye[j_col], eye[t_col]])
    for name, part in (("housing", housing), ("fingers", fingers)):
        if part is not None:
            lo, hi = part
            parts.append(box_grid(tool_axes @ ((lo + hi) / 2.0), tool_axes, (hi - lo) / 2.0, spacing))
            names.append(name)
            boxes.append((tool_axes @ ((lo + hi) / 2.0), tool_axes, (hi - lo) / 2.0))
    if use_jaws:
        sweep = ((view.get("grasp") or {}).get("graspgenx") or {}).get("sweep")
        if sweep is None or sweep.get("extents_open") is None:
            raise ValueError("safety.tool_volume.jaws needs grasp.graspgenx.sweep.extents_open")
        parts.append(box_grid(*jaw_box(sweep, order), spacing))
        names.append("jaws")
        boxes.append(jaw_box(sweep, order))
    for i, p in enumerate(parts):
        labels.append(np.full(len(p), i, dtype=int))
    return ToolVolume(local=np.vstack(parts), part=np.concatenate(labels), names=tuple(names),
                      spacing_m=spacing, table_band_m=band, boxes=tuple(boxes))


# ── calipers -> camera box (spec §3.4, §8) ─────────────────────────────────


def _outward_up(mm: float) -> float:
    """A rear / top / side face (mm) moved OUT to the next grid line, in m."""
    return math.ceil(mm / _CALIPER_GRID_MM) * _CALIPER_GRID_MM / 1000.0


def _outward_down(mm: float) -> float:
    """A front / bottom face (mm): outward is TOWARD the fingertip plane /
    the finger axis, so the face moves down to the grid line below, in m."""
    return math.floor(mm / _CALIPER_GRID_MM) * _CALIPER_GRID_MM / 1000.0


def camera_faces_from_calipers(measured_mm: dict, today: CameraDims | None = None) -> CameraDims:
    """The camera box the owner's calipers need (spec §8: mm, relative to the
    fingertip plane and the finger axis, jaws closed).

    Every measured face moves OUTWARD to the next 5 mm: rear, top and the
    wider of the two sides round up, front and bottom round down (toward the
    tip and the axis). A missing (null) number keeps `today`'s face; the box
    is symmetric across the jaws, so one measured side never narrows today's
    other side. A face in front of the fingertip plane or below the finger
    axis cannot be covered by the box and raises, as do non-numbers, unknown
    face names (a typo would silently keep today's face) and a box the
    camera parser would refuse.
    """
    from ..skills import relative_motion as rm
    from ..skills.spatial_moves import CameraDims

    today = CameraDims(*_TODAY_SLAB) if today is None else today
    unknown = sorted(set(measured_mm) - set(CALIPER_FACES))
    if unknown:
        raise ValueError(f"calipers: unknown face(s) {unknown}; expected {list(CALIPER_FACES)}")
    mm = {}
    for face in CALIPER_FACES:
        v = measured_mm.get(face)
        if v is None:
            continue
        if (isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating))
                or not math.isfinite(float(v))):
            raise ValueError(f"calipers: {face} must be a finite number of mm or null, got {v!r}")
        mm[face] = float(v)
    for face in ("front_behind_tip", "bottom_above_axis"):
        if mm.get(face, 0.0) < 0.0:
            raise ValueError(f"calipers: {face} = {mm[face]:g} mm lies in front of the fingertip plane or "
                             "below the finger axis; the camera box cannot cover it")
    for face in ("rear_behind_tip", "top_above_axis", "minus_y", "plus_y"):
        if mm.get(face, 1.0) <= 0.0:
            raise ValueError(f"calipers: {face} must be > 0 mm, got {mm[face]:g}")
    sides = [_outward_up(mm[f]) if f in mm else today.half_width_m for f in ("minus_y", "plus_y")]
    got = CameraDims(
        protrusion_m=_outward_up(mm["top_above_axis"]) if "top_above_axis" in mm else today.protrusion_m,
        behind_m=_outward_up(mm["rear_behind_tip"]) if "rear_behind_tip" in mm else today.behind_m,
        half_width_m=max(sides),
        front_m=_outward_down(mm["front_behind_tip"]) if "front_behind_tip" in mm else today.front_m,
        bottom_m=_outward_down(mm["bottom_above_axis"]) if "bottom_above_axis" in mm else today.bottom_m)
    try:
        rm.camera_box(np.eye(4), [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], **got._asdict())
    except SkillError as exc:
        raise ValueError(f"calipers: {exc}") from None
    return got


def uncovered_camera_faces(shipped: CameraDims, required: CameraDims) -> list[str]:
    """The faces of `required` (the calipered box) that the `shipped` box
    does not reach, in the order front, rear, sides, bottom, top; [] = the
    shipped box covers the measured camera."""
    missed = (("front", shipped.front_m > required.front_m),
              ("rear", shipped.behind_m < required.behind_m),
              ("sides", shipped.half_width_m < required.half_width_m),
              ("bottom", shipped.bottom_m > required.bottom_m),
              ("top", shipped.protrusion_m < required.protrusion_m))
    return [face for face, miss in missed if miss]
