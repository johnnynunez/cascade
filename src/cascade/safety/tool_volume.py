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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: allowed keys of `safety.tool_volume`
_KEYS = {"wrist_camera", "jaws", "spacing_m", "table_band_m"}


@dataclass(frozen=True)
class ToolVolume:
    """Clearance samples rigidly attached to the TCP.

    `local` (N, 3) sample coordinates in the TCP frame; `part` (N,) index into
    `names` (what a refusal names); `table_band_m`: occupied voxels at most
    this far above the table are the table itself, which the harness's table
    rule governs -- a box that reaches the fingertip plane would otherwise be
    refused by the table under every low grasp.
    """

    local: np.ndarray
    part: np.ndarray
    names: tuple
    spacing_m: float = 0.02
    table_band_m: float = 0.02

    def points(self, T_tcp: np.ndarray) -> np.ndarray:
        T = np.asarray(T_tcp, dtype=float)
        return self.local @ T[:3, :3].T + T[:3, 3]


def box_grid(center, axes, half, spacing_m: float) -> np.ndarray:
    """A grid over an oriented box (faces and interior), at most `spacing_m`
    apart along each box axis. `axes` columns are the box axes."""
    grids = []
    for h in np.asarray(half, dtype=float):
        n = max(1, math.ceil(2.0 * float(h) / spacing_m - 1e-9))
        grids.append(np.linspace(-float(h), float(h), n + 1) if h > 0 else np.zeros(1))
    local = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.asarray(center, dtype=float) + local @ np.asarray(axes, dtype=float).T


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


def build_tool_volume(view, arm_cfg) -> ToolVolume | None:
    """The `ToolVolume` an arm profile asks for, or None (not opted in).

    `view` is the arm's resolved config (safety, relative_motion, grasp);
    `arm_cfg` its profile (`tool_axis_order`). A malformed block raises
    ValueError: a safety volume that silently turns itself off is worse than
    none.
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
    if not (use_camera or use_jaws):
        raise ValueError("safety.tool_volume samples nothing (wrist_camera and jaws both false); "
                         "remove the block instead")

    from ..skills import relative_motion as rm

    order = str(arm_cfg.get("tool_axis_order", "down_open"))
    a_col, j_col, t_col = rm._TOOL_COLUMNS[order]
    eye = np.eye(3)
    parts, labels, names = [], [], []
    if use_camera:
        from ..skills.spatial_moves import wrist_camera_dims

        dims, up = wrist_camera_dims(view.get("relative_motion"))
        if dims is None:
            raise ValueError("safety.tool_volume.wrist_camera needs relative_motion.wrist_camera and "
                             "relative_motion.tool_up_axis (both are null here)")
        prot, behind, half_w = dims
        # the same function the spatial tools plan with, at T = identity:
        # the box in the TCP frame
        box = rm.camera_box(np.eye(4), eye[a_col], up, protrusion_m=prot, behind_m=behind,
                            half_width_m=half_w)
        parts.append(box_grid(box.center, box.axes, box.half, spacing))
        names.append("wrist camera")
    if use_jaws:
        sweep = ((view.get("grasp") or {}).get("graspgenx") or {}).get("sweep")
        if sweep is None or sweep.get("extents_open") is None:
            raise ValueError("safety.tool_volume.jaws needs grasp.graspgenx.sweep.extents_open")
        ext = np.asarray(sweep.get("extents_open"), dtype=float).reshape(3)
        off = np.asarray(sweep.get("offset_open", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        if not (np.all(np.isfinite(ext)) and np.all(ext >= 0) and np.all(np.isfinite(off))):
            raise ValueError(f"grasp.graspgenx.sweep open jaws must be finite, got {ext}, {off}")
        # sweep frame: +X closing, +Y third, +Z approach (origin at the TCP)
        axes = np.column_stack([eye[j_col], eye[t_col], eye[a_col]])
        parts.append(box_grid(axes @ off, axes, ext / 2.0, spacing))
        names.append("jaws")
    for i, p in enumerate(parts):
        labels.append(np.full(len(p), i, dtype=int))
    return ToolVolume(local=np.vstack(parts), part=np.concatenate(labels), names=tuple(names),
                      spacing_m=spacing, table_band_m=band)
