"""Drive LIBERO with cascade's OWN runtime, skills and verification.

WHY THIS EXISTS
---------------
Every earlier LIBERO number in this directory measured a pick-place primitive
written *inside* LIBERO. That tested the Pigey hypothesis, but not one line of
cascade ran, so it could not answer "how good is the thing I built". This
module closes that hole: it makes LIBERO look like a cascade backend, so
`SkillRuntime` -- the real one, with the real safety harness, the real grasp
pipeline and the real postconditions -- executes the benchmark.

WHAT IS AND IS NOT cascade AFTERWARDS

  runs unchanged   SkillRuntime.execute and every skill_* method, SafeArm +
                   SafetyHarness gating, GraspPlanner/GraspGenX selection,
                   PostconditionChecker, BeliefStore, TraceLogger, the agent
                   orchestrator.
  adapted          ArmBase  -> LiberoArm  (joint targets -> MuJoCo qpos)
                   Kinematics -> MujocoKinematics (fk/ik/link_positions off
                   the Panda model instead of the RS URDF)
                   CameraBase -> LiberoCamera (agentview RGB-D + intrinsics)
  necessarily new  the Panda is 7-DoF where the B601-RS is 6-DoF, so joint
                   vectors are 7 long. That is a property of the ROBOT, not a
                   rewrite of cascade -- ArmBase.n_joints is a class attr
                   precisely so backends can differ.

The honest framing for any table built on this: it measures **cascade's
orchestration and verification stack on a Franka Panda in LIBERO**, not
cascade on its own arm. The skills, the harness and the verification are
identical; the embodiment is not.
"""

from __future__ import annotations

import numpy as np

# ── kinematics backed by the LIBERO MuJoCo model ─────────────────────────


class MujocoKinematics:
    """Duck-typed replacement for cascade.control.kinematics.Kinematics.

    cascade needs exactly five things from a kinematics object: fk, ik,
    clamp, joint_limits and link_positions (the safety harness uses the last
    one as collision proxy spheres). Everything else in the repo goes through
    those, so a MuJoCo-backed implementation drops straight in.

    IK is damped least-squares on the site Jacobian. Pinocchio is not used
    because the Panda lives in a MJCF, and re-deriving a URDF would introduce
    a second source of truth for the geometry the simulator actually steps.
    """

    def __init__(self, model, data, site_name: str = "gripper0_grip_site",
                 joint_names: list[str] | None = None):
        import mujoco

        self._mj = mujoco
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            # fall back to the eef body if the site is named differently
            self.site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_eef")
            self._use_body = True
        else:
            self._use_body = False

        #: robosuite names the arm joints `robot0_joint1..7`, NOT
        #: `panda_joint*` -- getting this wrong silently yields zero
        #: controlled joints and every IK call broadcasts against an empty
        #: limit array.
        names = joint_names or [f"robot0_joint{i}" for i in range(1, 8)]
        self.joint_ids = []
        self.qpos_adr = []
        for n in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid < 0:
                continue
            self.joint_ids.append(jid)
            self.qpos_adr.append(model.jnt_qposadr[jid])
        self.n = len(self.joint_ids)

        lo = np.array([model.jnt_range[j][0] for j in self.joint_ids])
        hi = np.array([model.jnt_range[j][1] for j in self.joint_ids])
        self.joint_limits = (lo, hi)

    # -- helpers ---------------------------------------------------------

    def _set_q(self, q):
        for adr, v in zip(self.qpos_adr, np.asarray(q, float).reshape(-1)):
            self.data.qpos[adr] = v
        self._mj.mj_kinematics(self.model, self.data)

    def _site_pose(self):
        if self._use_body:
            pos = self.data.xpos[self.site_id].copy()
            mat = self.data.xmat[self.site_id].reshape(3, 3).copy()
        else:
            pos = self.data.site_xpos[self.site_id].copy()
            mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        T = np.eye(4)
        T[:3, :3] = mat
        T[:3, 3] = pos
        return T

    # -- the cascade Kinematics interface -------------------------------

    def fk(self, q) -> np.ndarray:
        """(n,) joints -> 4x4 T_tcp2base."""
        saved = self.data.qpos.copy()
        try:
            self._set_q(q)
            return self._site_pose()
        finally:
            self.data.qpos[:] = saved
            self._mj.mj_kinematics(self.model, self.data)

    def clamp(self, q, margin: float = 0.0) -> np.ndarray:
        lo, hi = self.joint_limits
        return np.clip(np.asarray(q, float), lo + margin, hi - margin)

    def link_positions(self, q) -> np.ndarray:
        """Base-frame positions of each controlled joint (collision proxies)."""
        saved = self.data.qpos.copy()
        try:
            self._set_q(q)
            return np.array([self.data.xanchor[j].copy() for j in self.joint_ids])
        finally:
            self.data.qpos[:] = saved
            self._mj.mj_kinematics(self.model, self.data)

    def ik(self, T_target, q_init, max_iter: int = 200, tol: float = 1e-4,
           damping: float = 1e-6, step: float = 0.5, retries: int = 4,
           seed: int = 0, limit_margin: float = 0.025):
        """Damped least-squares IK on the site Jacobian.

        Signature mirrors cascade's Kinematics.ik so callers (grasp planning,
        place_at, the reflex path) need no changes.
        """
        from cascade.control.kinematics import IKResult

        T_target = np.asarray(T_target, float)
        target_p = T_target[:3, 3]
        target_R = T_target[:3, :3]
        rng = np.random.default_rng(seed)
        lo, hi = self.joint_limits
        best = None

        saved = self.data.qpos.copy()
        try:
            for attempt in range(max(1, retries)):
                q = (np.asarray(q_init, float).reshape(-1).copy() if attempt == 0
                     else rng.uniform(lo + limit_margin, hi - limit_margin))
                err = np.inf
                for it in range(max_iter):
                    self._set_q(q)
                    T = self._site_pose()
                    dp = target_p - T[:3, 3]

                    # Orientation error as a rotation vector.
                    #
                    # MEASURED BUG this fixes: this solver used to request only
                    # the positional Jacobian (`mj_jacSite(..., jacp, None,
                    # ...)`) and minimise `dp` alone, so `T_target[:3, :3]` was
                    # silently discarded. Asking for a rim grasp with the jaws
                    # across the wall and asking for it rotated 90 degrees
                    # produced the SAME solution, measured: both yaw=0 and
                    # yaw=pi/2 put the opening axis at [-0.014, 0.999, 0.038].
                    # Both fingers then landed outside the bowl wall (r 59.6
                    # and 63.3 mm against a wall spanning 44.1 to 53.6 mm) and
                    # closed on each other.
                    #
                    # Any grasp whose success depends on jaw orientation was at
                    # the mercy of whichever branch the solver drifted into.
                    R_err = target_R @ T[:3, :3].T
                    angle = float(np.arccos(
                        np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
                    if angle < 1e-9:
                        dr = np.zeros(3)
                    else:
                        dr = np.array([R_err[2, 1] - R_err[1, 2],
                                       R_err[0, 2] - R_err[2, 0],
                                       R_err[1, 0] - R_err[0, 1]])
                        dr = dr * (angle / (2.0 * np.sin(angle)))

                    # Position in metres, orientation in radians: weight the
                    # angular part down so a small pose error does not get
                    # dominated by orientation, and report success on the
                    # positional error the callers already reason about.
                    err = float(np.linalg.norm(dp))
                    if err < tol and angle < 0.05:
                        break

                    jacp = np.zeros((3, self.model.nv))
                    jacr = np.zeros((3, self.model.nv))
                    if self._use_body:
                        self._mj.mj_jacBody(self.model, self.data, jacp, jacr,
                                            self.site_id)
                    else:
                        self._mj.mj_jacSite(self.model, self.data, jacp, jacr,
                                            self.site_id)
                    # columns for our controlled joints only
                    cols = [self.model.jnt_dofadr[j] for j in self.joint_ids]
                    J = np.vstack([jacp[:, cols], 0.35 * jacr[:, cols]])
                    e = np.concatenate([dp, 0.35 * dr])
                    JT = J.T
                    dq = JT @ np.linalg.solve(
                        J @ JT + damping * np.eye(6), e)
                    q = self.clamp(q + step * dq, limit_margin)
                if best is None or err < best.error:
                    best = IKResult(q=q.copy(), success=err < tol,
                                    error=err, iterations=it + 1)
                if best.success:
                    break
            return best
        finally:
            self.data.qpos[:] = saved
            self._mj.mj_kinematics(self.model, self.data)
