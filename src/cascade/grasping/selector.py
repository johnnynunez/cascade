"""Grasp selection: filter candidates by physics, then by reachability.

A grasp is only offered to the executor if BOTH the grasp pose and its
pregrasp pose solve under IK inside the workspace -- the baseline's "IK walk"
idea, kept, but applied to every candidate up front so the agent gets an
honest 'nothing reachable' instead of a mid-motion abort.
"""

from __future__ import annotations

import numpy as np

from ..types import Grasp, SkillError, make_transform
from . import evidence


class NoExecutableGrasp(SkillError):
    """All candidates failed geometry/IK checks before any actuation.

    `all_too_wide` is True when EVERY candidate exceeded the jaw span (an
    object property no re-scan can cure); `narrowest_width_m` is then the
    best the planner could offer and `jaw_max_width_m` the span it lost
    to. For a mixed or IK/vetting failure `all_too_wide` is False and
    `narrowest_width_m` None (`jaw_max_width_m` is always the span used)."""

    all_too_wide: bool = False
    narrowest_width_m: float | None = None
    jaw_max_width_m: float | None = None


#: substring of the per-candidate width reason (also matched by the runtime)
_TOO_WIDE_MARK = "> gripper max"
#: the explicit all-candidates refusal, matched by `_grasp_retry_verdict`
ALL_TOO_WIDE_MARKER = "every candidate exceeds the jaw span"


def _flip_twin(g: Grasp) -> Grasp:
    """The same parallel-jaw grasp with the jaws swapped: the TCP rotated
    180 degrees about the approach axis. Position, width and approach are
    unchanged; only which finger lands on which side differs. The twin keeps
    the candidate's kind (`obb_grasp.AngledGrasp` stays one: the opt-in
    clearance vet must see both orientations)."""
    a = np.asarray(g.approach, dtype=float).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    flip = 2.0 * np.outer(a, a) - np.eye(3)  # rotation by pi about `a`
    return type(g)(position=g.position, rotation=flip @ np.asarray(g.rotation, dtype=float),
                   width_m=g.width_m, approach=g.approach, quality=g.quality, label=g.label)


def select_grasp(
    grasps: list[Grasp],
    kin,
    q_current: np.ndarray,
    max_width_m: float = 0.09,
    pregrasp_offset_m: float = 0.12,
    validate=None,
    jaw_fixed_tip_m=None,
    jaw_close_dir=None,
    check=None,
    preserve_order: bool = False,
) -> tuple[Grasp, np.ndarray, np.ndarray]:
    """-> (grasp, q_pregrasp, q_grasp) for the best executable candidate.

    validate(grasp, q_pre, q_grasp) -> reason-string|None lets the caller
    veto candidates on grounds IK cannot see (safety-harness geometry): a
    candidate that would abort mid-descent must lose the ranking here, not
    kill the attempt later.

    By default candidates are ranked by their model quality. A caller that
    already ranked them (for example with an outcome-memory prior) must set
    ``preserve_order=True``. Every candidate still passes the same width,
    IK and validation checks; its stored model quality is never rewritten.

    A symmetric parallel jaw has TWO wrist poses for every grasp (jaws
    swapped, 180 degrees apart about the approach). The planners emit both
    and rank them by quality, which is ~equal by construction -- so the
    ranking picked the flip by a 1 % tie-break, blind to joint travel.
    MEASURED on the kitchen orange: the chosen pose spun the wrist 156.5
    degrees from home while its twin needed ~24; at the kitchen's real-time
    factor the wrist was still creeping (153.2 -> 154.5 deg) when the settle
    window closed, and the proof failed twice with "did not settle at
    pregrasp pose". The best candidate is therefore executed in whichever
    orientation needs the smaller largest-joint excursion from `q_current`.
    Both orientations independently pass IK and validation, even when the
    original fails; rejecting one must not discard its feasible twin.
    Single-hinge jaws (below) are not symmetric and keep their pose.

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
    q_ref = np.asarray(q_current, dtype=float).reshape(-1)
    evidence.event("jaw_datum", fixed_tip_m=fixed_tip, close_dir=close_dir,
                   max_width_m=max_width_m, pregrasp_offset_m=pregrasp_offset_m)

    def _check():
        if check is not None:
            check()

    def _solve(g: Grasp):
        """-> (q_pre, q_grasp) or a failure reason string."""
        _check()
        p_grasp = g.position
        if fixed_tip is not None and close_dir is not None:
            off = fixed_tip + close_dir * (float(g.width_m) / 2.0)
            p_grasp = g.position - g.rotation @ off
        T_grasp = make_transform(g.rotation, p_grasp)
        T_pre = make_transform(
            g.rotation, p_grasp - g.approach * pregrasp_offset_m
        )
        evidence.event("ik_targets", grasp=g, T_grasp=T_grasp, T_pre=T_pre, seed_q=q_current)
        pre = kin.ik(T_pre, q_current)
        _check()
        evidence.ik_result("pregrasp_ik", pre)
        if not pre.success:
            return f"pregrasp IK failed (err {pre.error:.4f})"
        grasp = kin.ik(T_grasp, pre.q)
        _check()
        evidence.ik_result("grasp_ik", grasp)
        if not grasp.success:
            return f"grasp IK failed (err {grasp.error:.4f})"
        if validate is not None:
            _check()
            reason = validate(g, pre.q, grasp.q)
            _check()
            evidence.event("candidate_validation", reason=reason)
            if reason:
                return f"{g.label}: {reason}"
        return pre.q, grasp.q

    def _travel(q_pre) -> float:
        q = np.asarray(q_pre, dtype=float).reshape(-1)
        n = min(q.size, q_ref.size)
        return float(np.max(np.abs(q[:n] - q_ref[:n]))) if n else 0.0

    _check()
    too_wide = 0
    narrowest: float | None = None
    considered = 0
    for g in grasps if preserve_order else sorted(grasps, key=lambda g: -g.quality):
        _check()
        considered += 1
        if g.width_m > max_width_m:
            too_wide += 1
            w = float(g.width_m)
            narrowest = w if narrowest is None else min(narrowest, w)
            reasons.append(
                f"{g.label}: required width {g.width_m * 1000:.0f}mm > gripper "
                f"max {max_width_m * 1000:.0f}mm (consider push or regrasp)"
            )
            continue
        solved = _solve(g)
        best = None
        if isinstance(solved, str):
            reasons.append(solved)
        else:
            best = (g, *solved)
        if fixed_tip is None:
            twin = _flip_twin(g)
            twin_solved = _solve(twin)
            if isinstance(twin_solved, str):
                reasons.append(twin_solved)
            elif best is None or _travel(twin_solved[0]) < _travel(best[1]):
                best = (twin, *twin_solved)
        _check()
        if best is not None:
            return best
    _check()
    # Over-width is an OBJECT property: re-scanning cannot shrink it, so the
    # persistence loops stop on it. Say so explicitly, with the measurement
    # (narrowest candidate vs jaw span), and keep the structured fact on the
    # exception. A MIXED list (some too wide, some vetoed/IK-failed) is NOT
    # that refusal: list the non-width reasons first so the 4-reason
    # truncation below can never make it read as "every candidate too wide".
    all_too_wide = considered > 0 and too_wide == considered
    if not all_too_wide:
        reasons = ([r for r in reasons if _TOO_WIDE_MARK not in r]
                   + [r for r in reasons if _TOO_WIDE_MARK in r])
    message = "no executable grasp: "
    if all_too_wide:
        message += (f"{ALL_TOO_WIDE_MARKER} (narrowest {narrowest * 1000:.0f}mm > gripper "
                    f"max {max_width_m * 1000:.0f}mm; use push_object): ")
    exc = NoExecutableGrasp(message + "; ".join(reasons[:4]))
    exc.all_too_wide = all_too_wide
    exc.narrowest_width_m = narrowest if all_too_wide else None
    exc.jaw_max_width_m = float(max_width_m)
    raise exc
