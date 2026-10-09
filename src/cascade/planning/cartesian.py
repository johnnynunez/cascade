"""Cartesian straight-line paths as dense, continuous joint waypoints.

Port of Seeed WRC e97998c (`src/wrc_demo/planning/cartesian_planner.py`):
WRC wrapped reBotArm_control_py's SE(3)-geodesic sampler and CLIK tracker so
grasp descents, lifts and nudges move the TCP along a line instead of the
curve a joint-space min-jerk traces. This version is self-contained -- it
uses cascade's own `Kinematics` (the model the harness already vets with)
and plain numpy for SE(3), so it imports neither the vendor SDK nor anything
heavy, and works for every arm profile, mock included.

It is deliberately STRICTER than WRC's, which accepted up to 5 % of samples
whose IK did not converge (WRC itself tightened that default on 2026-08-13
because partial failures "silently move the arm short of the target"):

- every sample must solve, each seeded from the previous solution with NO
  random restarts (a restart may land on another IK branch);
- no joint may move more than `max_joint_step_rad` between samples -- an IK
  branch flip is a swing through free space, not a straight line;
- the last sample is the goal itself.

Any violation raises `SkillError` (the agent should reason about it: pick
another pose, or move in joint space). A plan never moves anything: the
caller (`SafeArm.move_cartesian`) preflights the dense path with the harness
and streams it with per-tick `approve()`.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import SkillError

#: default sampling: 5 mm of TCP travel or ~2 deg of tool rotation per sample
MAX_STEP_M = 0.005
MAX_STEP_RAD = 0.035
#: a larger joint change between adjacent 5 mm samples is a branch flip
MAX_JOINT_STEP_RAD = 0.1


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), angle in [0, pi]."""
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Near pi the skew part vanishes; recover the axis from R + I.
        M = (R + np.eye(3)) / 2.0
        axis = M[:, int(np.argmax(np.diag(M)))]
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (angle / (2.0 * math.sin(angle)))


def _so3_exp(w: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def se3_interpolate(T0: np.ndarray, T1: np.ndarray, s: float) -> np.ndarray:
    """Pose at fraction `s` along the straight line T0 -> T1: position on the
    segment, orientation on the rotation geodesic (constant angular rate)."""
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    out = np.eye(4)
    out[:3, 3] = T0[:3, 3] + (T1[:3, 3] - T0[:3, 3]) * s
    w = _so3_log(T0[:3, :3].T @ T1[:3, :3])
    out[:3, :3] = T0[:3, :3] @ _so3_exp(w * s)
    return out


def plan_cartesian_path(kin, q_start, T_goal, *, max_step_m: float = MAX_STEP_M,
                        max_step_rad: float = MAX_STEP_RAD,
                        max_joint_step_rad: float = MAX_JOINT_STEP_RAD) -> list[np.ndarray]:
    """Joint waypoints (excluding the start) whose TCP follows the line from
    FK(q_start) to `T_goal`. Raises SkillError instead of approximating."""
    q_prev = np.asarray(q_start, dtype=float).reshape(-1).copy()
    T_goal = np.asarray(T_goal, dtype=float)
    if T_goal.shape != (4, 4) or not np.isfinite(T_goal).all() or not np.isfinite(q_prev).all():
        raise SkillError("cartesian path needs a finite 4x4 goal and finite joints")
    T_start = np.asarray(kin.fk(q_prev), dtype=float)
    dist = float(np.linalg.norm(T_goal[:3, 3] - T_start[:3, 3]))
    turn = float(np.linalg.norm(_so3_log(T_start[:3, :3].T @ T_goal[:3, :3])))
    n = max(1, int(math.ceil(max(dist / max_step_m, turn / max_step_rad))))
    path = []
    for i in range(1, n + 1):
        T_i = se3_interpolate(T_start, T_goal, i / n)
        res = kin.ik(T_i, q_init=q_prev, retries=0)
        if not res.success:
            raise SkillError(
                f"cartesian path: IK failed at {i}/{n} "
                f"({np.round(T_i[:3, 3], 3).tolist()}, err {float(res.error):.4f}); "
                "the straight line leaves the reachable set -- choose another "
                "pose or move in joint space")
        q_i = np.asarray(res.q, dtype=float).reshape(-1)[: q_prev.size]
        jump = float(np.max(np.abs(q_i - q_prev)))
        if jump > max_joint_step_rad:
            raise SkillError(
                f"cartesian path: joint jump of {jump:.3f} rad at sample {i}/{n} "
                f"(> {max_joint_step_rad:.3f}): the IK changed branch, so the "
                "TCP would swing off the line")
        path.append(q_i)
        q_prev = q_i
    return path
