"""FK/IK on the arm model via Pinocchio.

Self-contained on purpose: reBotArm_control_py's kinematics module reads its
own global config/rebotarm.yaml (ignoring the hw_yaml passed to RebotArm), so
using it with the RS arm silently loads the DM model. Here the model path
comes from the arm profile explicitly. Both .urdf and .usd/.usda paths load:
USD goes through usd_model's in-memory USD-physics -> URDF translation
(Pinocchio has no USD reader), so the exact asset Newton/Isaac Sim simulates
can be checked against the host model (see tests/test_usd_model.py).

Both shipped RS assets are authored in the MIRRORED joint convention
(q_asset = -q_local); the SDK, the real arm and every q constant in this
repo use the local one. Arm profiles carry `joint_signs: [-1, ...]` and the
flip is baked into the model at load time, so everything downstream (FK, IK,
limits, safety) speaks local convention.

Models carry more joints than we command (the RS URDF has nq=8: 6 revolute plus
2 passive prismatic finger joints; the SO-101 URDF has nq=6, 5 arm joints plus
the gripper). We command the first `n_controlled` and zero-pad the rest, same
convention as the SDK. Both shipped URDFs happen to order the arm chain before
the gripper, which is what makes that slice correct -- a new model must be
checked, not assumed (`tests/test_kinematics_so101.py` pins it for SO-101).

UNDER-ACTUATED ARMS (`ik_task_weights`)
---------------------------------------
A 6-DoF pose request has no solution on a chain with fewer than 6 useful DOF,
and plain damped least-squares answers that by stalling at a non-zero residual
and reporting `success=False` for every single target -- i.e. "this arm cannot
reach anything", which is not the useful truth. The useful truth is which task
DOF the chain gives up.

An arm profile may therefore declare a 6-vector `ik_task_weights` naming how
much each task DOF matters, ordered [x, y, z, rx, ry, rz] about the WORLD axes
of the base frame (not the tool's -- a mask that rotates with the wrist is not
a mask anyone can reason about). Zero drops that DOF from the problem entirely.

NO SHIPPED PROFILE SETS THIS, including the 5-DoF SO-101, and the SO-101 is
worth spelling out because the intuition is wrong: its wrist-roll axis is
exactly collinear with the tool approach (MEASURED: rolling it turns the
approach by 0.00 deg), so for a VERTICAL approach the roll spends itself
entirely on the jaw yaw and the chain behaves like a full-pose arm -- pan sets
the azimuth, lift/elbow/flex set radius, height and pitch, roll sets the yaw.
Unweighted 6-DoF IK solves those targets at a 0.94 rate over its workspace. The
DOF it truly lacks shows up only for a TILTED approach, whose azimuth is then
pinned to the arm's working plane; nothing in the tabletop pipeline asks for
that, so the honest gate is IK failure rather than a weight that would quietly
accept a pose the arm cannot hold.

When `ik_task_weights` is absent (i.e. always, today), the solver is exactly
the unweighted LOCAL-frame one it always was, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import local

import numpy as np


@dataclass
class IKResult:
    q: np.ndarray  # (n_controlled,) solution
    success: bool
    error: float
    iterations: int


class Kinematics:
    def __init__(
        self,
        model_path: str,
        ee_frame: str,
        n_controlled: int = 6,
        joint_signs: list[int] | None = None,
        ik_task_weights: list[float] | None = None,
    ):
        import pinocchio as pin  # heavy import, keep local

        self._pin = pin
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"robot model not found: {model_path}")
        if path.suffix in (".usd", ".usda"):
            from .usd_model import urdf_xml_from_usd

            xml = urdf_xml_from_usd(path)
        else:
            xml = path.read_text()
        if joint_signs and any(int(s) == -1 for s in joint_signs):
            from .usd_model import apply_joint_signs

            xml = apply_joint_signs(xml, joint_signs)
        self.model = pin.buildModelFromXML(xml)
        # The model is immutable; Pinocchio Data is mutable scratch state.
        # Perception masks and the control/IK loop query this object in
        # parallel, so sharing Data can substitute another thread's pose.
        self._thread_data = local()
        self.ee_frame = ee_frame
        self.fid = self.model.getFrameId(ee_frame)
        if self.fid >= len(self.model.frames.tolist()):
            raise ValueError(f"frame {ee_frame!r} not in model")
        self.n = n_controlled
        self.nq = self.model.nq
        lo = np.asarray(self.model.lowerPositionLimit[: self.n])
        hi = np.asarray(self.model.upperPositionLimit[: self.n])
        self.joint_limits = (lo, hi)
        self.task_weights = (
            None if ik_task_weights is None
            else np.asarray(ik_task_weights, dtype=float).reshape(6)
        )
        if self.task_weights is not None and np.any(self.task_weights < 0):
            raise ValueError(f"ik_task_weights must be >= 0, got {ik_task_weights!r}")

    @property
    def data(self):
        if not hasattr(self._thread_data, "data"):
            self._thread_data.data = self.model.createData()
        return self._thread_data.data

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
        w = self.task_weights
        err_norm = np.inf
        for it in range(max_iter):
            qf = self._pad(q)
            pin.forwardKinematics(self.model, self.data, qf)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.fid]
            err = pin.log6(oMf.actInv(target)).vector
            if w is None:
                frame = pin.ReferenceFrame.LOCAL
            else:
                # Weights name WORLD axes (see `task_weights` docs), so the
                # error and the Jacobian both move to the world-aligned frame.
                # J_lwa = diag(R, R) @ J_local, so rotating the local error the
                # same way keeps the pair consistent.
                R = np.asarray(oMf.rotation)
                err = np.concatenate([R @ err[:3], R @ err[3:]])
                err = w * err
                frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                return IKResult(q=q.copy(), success=True, error=err_norm, iterations=it)
            J = pin.computeFrameJacobian(self.model, self.data, qf, self.fid, frame)[:, : self.n]
            if w is not None:
                J = w[:, None] * J
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
