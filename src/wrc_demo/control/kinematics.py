"""FK/IK on the arm URDF via Pinocchio.

Self-contained on purpose: reBotArm_control_py's kinematics module reads its
own global config/rebotarm.yaml (ignoring the hw_yaml passed to RebotArm), so
using it with the RS arm silently loads the DM URDF. Here the URDF path comes
from the arm profile explicitly, and the RS URDF ships in this repo's assets.

The RS model has nq=8 (6 revolute + 2 passive prismatic finger joints); we
command the first 6 and zero-pad the rest, same convention as the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IKResult:
    q: np.ndarray  # (n_controlled,) solution
    success: bool
    error: float
    iterations: int


class Kinematics:
    def __init__(self, urdf_path: str, ee_frame: str, n_controlled: int = 6):
        import pinocchio as pin  # heavy import, keep local

        self._pin = pin
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.ee_frame = ee_frame
        self.fid = self.model.getFrameId(ee_frame)
        if self.fid >= len(self.model.frames.tolist()):
            raise ValueError(f"frame {ee_frame!r} not in URDF")
        self.n = n_controlled
        self.nq = self.model.nq
        lo = np.asarray(self.model.lowerPositionLimit[: self.n])
        hi = np.asarray(self.model.upperPositionLimit[: self.n])
        self.joint_limits = (lo, hi)

    def _pad(self, q: np.ndarray) -> np.ndarray:
        qf = np.zeros(self.nq)
        qf[: self.n] = np.asarray(q, dtype=float).reshape(-1)[: self.n]
        return qf

    def fk(self, q: np.ndarray) -> np.ndarray:
        """(n,) joints -> 4x4 T_tcp2base."""
        pin = self._pin
        qf = self._pad(q)
        pin.forwardKinematics(self.model, self.data, qf)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.fid].homogeneous)

    def clamp(self, q: np.ndarray, margin: float = 0.0) -> np.ndarray:
        lo, hi = self.joint_limits
        return np.clip(q, lo + margin, hi - margin)

    def ik(
        self,
        T_target: np.ndarray,
        q_init: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-4,
        damping: float = 1e-6,
        step: float = 0.5,
        retries: int = 4,
        seed: int = 0,
        limit_margin: float = 0.025,
    ) -> IKResult:
        """Damped least-squares IK with random restarts inside joint limits.

        `limit_margin` keeps solutions strictly inside the URDF limits so the
        safety harness (which enforces its own margin) never rejects a pose
        the planner called reachable.
        """
        pin = self._pin
        target = pin.SE3(np.asarray(T_target[:3, :3]), np.asarray(T_target[:3, 3]))
        rng = np.random.default_rng(seed)
        lo, hi = self.joint_limits
        best = IKResult(q=np.asarray(q_init, dtype=float)[: self.n].copy(),
                        success=False, error=np.inf, iterations=0)
        starts = [np.asarray(q_init, dtype=float)[: self.n].copy()]
        starts += [rng.uniform(lo + limit_margin, hi - limit_margin) for _ in range(retries)]
        for q0 in starts:
            res = self._ik_once(target, q0, max_iter, tol, damping, step, limit_margin)
            if res.error < best.error:
                best = res
            if res.success:
                return res
        return best

    def _ik_once(self, target, q0, max_iter, tol, damping, step, margin=0.0) -> IKResult:
        pin = self._pin
        q = self.clamp(q0.copy(), margin)
        err_norm = np.inf
        for it in range(max_iter):
            qf = self._pad(q)
            pin.forwardKinematics(self.model, self.data, qf)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.fid]
            err = pin.log6(oMf.actInv(target)).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                return IKResult(q=q.copy(), success=True, error=err_norm, iterations=it)
            J = pin.computeFrameJacobian(
                self.model, self.data, qf, self.fid, pin.ReferenceFrame.LOCAL
            )[:, : self.n]
            lam = damping * max(1.0, err_norm * 10.0)
            JJt = J @ J.T + lam * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = self.clamp(q + step * dq, margin)
        return IKResult(q=q.copy(), success=False, error=err_norm, iterations=max_iter)

    def link_positions(self, q: np.ndarray) -> np.ndarray:
        """Base-frame positions of every joint frame -> (n_joints, 3).

        Used by the safety harness as coarse collision proxy spheres.
        """
        pin = self._pin
        qf = self._pad(q)
        pin.forwardKinematics(self.model, self.data, qf)
        return np.array([np.asarray(self.data.oMi[j].translation)
                         for j in range(1, self.model.njoints)])
