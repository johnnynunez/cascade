"""Grasp selection: filter candidates by physics, then by reachability.

A grasp is only offered to the executor if BOTH the grasp pose and its
pregrasp pose solve under IK inside the workspace -- the baseline's "IK walk"
idea, kept, but applied to every candidate up front so the agent gets an
honest 'nothing reachable' instead of a mid-motion abort.
"""

from __future__ import annotations

import numpy as np

from ..types import Grasp, SkillError, make_transform


def select_grasp(
    grasps: list[Grasp],
    kin,
    q_current: np.ndarray,
    max_width_m: float = 0.09,
    pregrasp_offset_m: float = 0.12,
) -> tuple[Grasp, np.ndarray, np.ndarray]:
    """-> (grasp, q_pregrasp, q_grasp) for the best executable candidate."""
    reasons: list[str] = []
    for g in sorted(grasps, key=lambda g: -g.quality):
        if g.width_m > max_width_m:
            reasons.append(
                f"{g.label}: required width {g.width_m * 1000:.0f}mm > gripper "
                f"max {max_width_m * 1000:.0f}mm (consider push or regrasp)"
            )
            continue
        T_grasp = make_transform(g.rotation, g.position)
        T_pre = make_transform(g.rotation, g.pregrasp_position(pregrasp_offset_m))
        pre = kin.ik(T_pre, q_current)
        if not pre.success:
            reasons.append(f"pregrasp IK failed (err {pre.error:.4f})")
            continue
        grasp = kin.ik(T_grasp, pre.q)
        if not grasp.success:
            reasons.append(f"grasp IK failed (err {grasp.error:.4f})")
            continue
        return g, pre.q, grasp.q
    raise SkillError("no executable grasp: " + "; ".join(reasons[:4]))
