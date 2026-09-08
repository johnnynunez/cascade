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
    validate=None,
    jaw_fixed_tip_m=None,
    jaw_close_dir=None,
) -> tuple[Grasp, np.ndarray, np.ndarray]:
    """-> (grasp, q_pregrasp, q_grasp) for the best executable candidate.

    validate(grasp, q_pre, q_grasp) -> reason-string|None lets the caller
    veto candidates on grounds IK cannot see (safety-harness geometry): a
    candidate that would abort mid-descent must lose the ranking here, not
    kill the attempt later.

    `jaw_fixed_tip_m` + `jaw_close_dir` (both TOOL frame) describe a
    single-hinge jaw whose closing point is not symmetric about the tool
    frame: the FIXED jaw's contact point, and the unit direction from it
    toward the moving jaw. When a grasp of width w closes, the object ends
    pinched between the fixed tip and (fixed tip + close_dir * w), so the
    frame must be posed with the OBJECT at fixed_tip + close_dir * (w/2) --
    i.e. the frame target is g.position - R @ that offset. The SO-101 beak
    is the motivating case: its open-gap centre sits 30 mm along -open from
    the frame origin (which IS the fixed tip, measured rigid to 2.3 um), so
    centring the object in the OPEN gap parks it 30 mm from where the jaws
    actually close -- the finger shoved the prop on descent and every grasp
    read "air grasp" while perception was accurate to 1.4 mm. Parallel-jaw
    arms (reBot, Panda) omit both keys: their gap centre is the frame origin
    at every width, so the offset would be ~0.
    """
    reasons: list[str] = []
    fixed_tip = None
    close_dir = None
    if jaw_fixed_tip_m is not None or jaw_close_dir is not None:
        fixed_tip = np.zeros(3) if jaw_fixed_tip_m is None else \
            np.asarray(jaw_fixed_tip_m, dtype=float).reshape(3)
        close_dir = np.asarray(
            jaw_close_dir if jaw_close_dir is not None else [-1.0, 0.0, 0.0],
            dtype=float).reshape(3)
        n = float(np.linalg.norm(close_dir))
        if n > 1e-9:
            close_dir = close_dir / n
    for g in sorted(grasps, key=lambda g: -g.quality):
        if g.width_m > max_width_m:
            reasons.append(
                f"{g.label}: required width {g.width_m * 1000:.0f}mm > gripper "
                f"max {max_width_m * 1000:.0f}mm (consider push or regrasp)"
            )
            continue
        p_grasp = g.position
        if fixed_tip is not None and close_dir is not None:
            off = fixed_tip + close_dir * (float(g.width_m) / 2.0)
            p_grasp = g.position - g.rotation @ off
        T_grasp = make_transform(g.rotation, p_grasp)
        T_pre = make_transform(
            g.rotation, p_grasp - g.approach * pregrasp_offset_m
        )
        pre = kin.ik(T_pre, q_current)
        if not pre.success:
            reasons.append(f"pregrasp IK failed (err {pre.error:.4f})")
            continue
        grasp = kin.ik(T_grasp, pre.q)
        if not grasp.success:
            reasons.append(f"grasp IK failed (err {grasp.error:.4f})")
            continue
        if validate is not None:
            reason = validate(g, pre.q, grasp.q)
            if reason:
                reasons.append(f"{g.label}: {reason}")
                continue
        return g, pre.q, grasp.q
    raise SkillError("no executable grasp: " + "; ".join(reasons[:4]))
