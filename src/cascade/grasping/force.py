"""Material-aware grip effort ("how hard to squeeze" challenge).

The gripper is a single RobStride motor in MIT mode: commanded stiffness
(kp scale) bounds the stall torque, so `effort` is our force proxy until
current-loop telemetry is calibrated onsite. Profiles follow the ASPIRE
two-stage close (scout grip, then seat) with per-material close depth,
effort and speed.

The material class comes from the VLM advisor when available ("ceramic mug,
fragile") or from a keyword heuristic over the object label as fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GripProfile:
    name: str
    close_frac_stage1: float  # fraction of full close travel for the scout grip
    close_frac_stage2: float  # final seat
    effort: float  # 0..1 stiffness scale (bounds stall torque)
    lift_speed_scale: float  # slow down lifts for risky grips


GRIP_PROFILES: dict[str, GripProfile] = {
    "rigid": GripProfile("rigid", 0.50, 0.85, 0.70, 1.0),
    "fragile": GripProfile("fragile", 0.40, 0.60, 0.30, 0.5),
    "soft": GripProfile("soft", 0.60, 0.95, 0.45, 0.8),
    "deformable": GripProfile("deformable", 0.60, 0.95, 0.45, 0.8),
    "slippery": GripProfile("slippery", 0.50, 0.90, 0.85, 0.5),
    "heavy": GripProfile("heavy", 0.50, 0.90, 1.00, 0.5),
}

_KEYWORDS = {
    "fragile": ("glass", "egg", "ceramic", "porcelain", "wine", "light bulb"),
    "soft": ("plush", "sponge", "foam", "stuffed", "toy animal", "bread"),
    "deformable": ("bag", "cloth", "towel", "paper cup"),
    "slippery": ("bottle", "can", "metal", "banana", "apple", "orange"),
    "heavy": ("drill", "hammer", "brick"),
}


def select_profile(label: str, material_hint: str | None = None) -> GripProfile:
    if material_hint:
        hint = material_hint.strip().lower()
        if hint in GRIP_PROFILES:
            return GRIP_PROFILES[hint]
        for name in GRIP_PROFILES:
            if name in hint:
                return GRIP_PROFILES[name]
    low = label.lower()
    for name, words in _KEYWORDS.items():
        if any(w in low for w in words):
            return GRIP_PROFILES[name]
    return GRIP_PROFILES["rigid"]


def pregrasp_open_position(width_m, margin_m, open_pos: float, closed_pos: float,
                           max_width_m: float) -> float:
    """Gripper motor position for the pre-grasp opening.

    Ported from Seeed WRC commit a2d5950 (skill_grasp_object opened the jaws
    to `grasp.width_m + 10 mm` instead of fully, so the fingers sweep less of
    the scene on the descent). Opt-in: `margin_m` is the arm profile's
    `gripper.pregrasp_open_margin_m`, and None keeps the full `open_pos`.

    The opening is `min(width + margin, max_width)` mapped through the
    framework's LINEAR angle->width convention (closed_pos = 0 m, open_pos =
    max_width_m), so it is only as right as that map: verify the profile's
    max_width_m <-> open_pos pair on the hardware before enabling it. It can
    never be narrower than the grasp width -- a negative or non-finite margin
    is a config error -- and a width that is unknown or non-positive gives no
    basis for a partial opening, so the jaws open fully.
    """
    open_pos, closed_pos = float(open_pos), float(closed_pos)
    if margin_m is None:
        return open_pos
    try:
        margin = float(margin_m)
    except (TypeError, ValueError):
        margin = float("nan")
    if not math.isfinite(margin) or margin < 0:
        from ..types import SkillError

        raise SkillError(
            f"gripper.pregrasp_open_margin_m must be a finite, non-negative "
            f"distance in metres (got {margin_m!r}): a negative margin would "
            "open the jaws narrower than the object")
    try:
        width = float(width_m)
    except (TypeError, ValueError):
        return open_pos
    max_width = float(max_width_m)
    if not math.isfinite(width) or width <= 0 or not math.isfinite(max_width) or max_width <= 0:
        return open_pos
    frac = min((width + margin) / max_width, 1.0)
    return closed_pos + (open_pos - closed_pos) * frac
