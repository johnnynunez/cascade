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
