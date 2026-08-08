"""LIBERO as a wrc_demo arm + camera backend.

`LiberoArm` implements ArmBase so that SafeArm, SafetyHarness and every
skill_* method drive the Franka in LIBERO exactly as they drive the B601-RS
in Isaac. `LiberoCamera` implements CameraBase so the detector, the belief
store and the grasp pipeline see a normal wrc_demo Frame.

The one structural difference from the real rig: LIBERO's env.step() advances
physics, so motion is DISCRETE. ArmBase.stream_to() paces waypoints with
time.sleep() against a wall clock, which in a stepped simulator would let the
sim stand still while the clock runs. `stream_to` is therefore overridden to
step the environment per waypoint instead of sleeping -- same min-jerk
profile, same per-waypoint harness approval, just driven by sim time.
"""

from __future__ import annotations

import time

import numpy as np

from wrc_demo.control.arm_base import ArmBase, min_jerk
from wrc_demo.perception.camera_base import CameraBase
from wrc_demo.types import Frame, RobotState


class LiberoArm(ArmBase):
    """A 7-DoF Franka in LIBERO behind wrc_demo's ArmBase contract."""

    #: The Panda has 7 arm joints; the B601-RS has 6. ArmBase.n_joints is a
    #: class attribute precisely so backends can differ -- nothing in
    #: wrc_demo hardcodes 6 outside the RS profile.
    n_joints = 7
    settle_tol = 0.05

    def __init__(self, env, kin, grip_open: float = -1.0, grip_closed: float = 1.0):
        self.env = env
        self.kin = kin
        self._grip_open = grip_open
        self._grip_closed = grip_closed
        self._grip_cmd = grip_open
        self._connected = False
        self._stopped = False
        self.last_obs = None
        #: robosuite raises on any step after termination; latch it so a skill
        #: mid-motion degrades to a no-op instead of exploding
        self._terminated = False
        #: set by the runner so every env.step can publish a frame to the
        #: camera from the GL-owning thread
        self.camera = None

    # -- lifecycle -------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def stop(self) -> None:
        """Software e-stop: hold position, refuse further motion."""
        self._stopped = True

    # -- state -----------------------------------------------------------

    def _step(self, action):
        """Step the sim, tolerating both gym APIs.

        robosuite's raw env returns 4 values (obs, reward, done, info); the
        LIBERO wrapper returns the gymnasium 5-tuple. We hold the INNER env
        because the wrc_demo backends need `.sim`, so unpack defensively
        instead of assuming a shape -- guessing here fails at the very first
        motion with an opaque unpack error.

        Once the episode terminates, robosuite raises on any further step.
        A skill mid-motion has no idea the benchmark called time, so swallow
        it and report the terminal state instead of crashing the skill.
        """
        if self._terminated:
            return self.last_obs, True
        try:
            out = self.env.step(action)
        except ValueError as e:
            if "terminated episode" in str(e):
                self._terminated = True
                return self.last_obs, True
            raise
        if len(out) == 5:
            obs, _rew, term, _trunc, _info = out
        else:
            obs, _rew, term, _info = out
        self.last_obs = obs
        self._last_term = bool(term)
        self._terminated = self._terminated or bool(term)
        # Hand the freshly rendered frame to the camera from THIS thread --
        # the one that owns MuJoCo's GL context (see LiberoCamera._grab).
        if self.camera is not None:
            self.camera.publish(obs)
        return obs, bool(term)

    def _joint_pos(self) -> np.ndarray:
        obs = self.last_obs or {}
        q = obs.get("robot0_joint_pos")
        if q is not None:
            return np.asarray(q, float).reshape(-1)[: self.n_joints]
        # fall back to reading the model directly
        return np.array([self.kin.data.qpos[a] for a in self.kin.qpos_adr])

    def get_state(self) -> RobotState:
        obs = self.last_obs or {}
        g = obs.get("robot0_gripper_qpos")
        gripper = float(np.asarray(g).reshape(-1)[0]) if g is not None else 0.0
        return RobotState(q=self._joint_pos(), gripper_pos=gripper,
                          gripper_valid=g is not None, t=time.monotonic())

    # -- actuation -------------------------------------------------------

    def send_joint_target(self, q: np.ndarray) -> None:
        """One joint-space setpoint = one env.step in a stepped simulator."""
        if self._stopped:
            return
        q = np.asarray(q, float).reshape(-1)[: self.n_joints]
        cur = self._joint_pos()
        # JOINT_POSITION control in robosuite takes a DELTA scaled to [-1, 1];
        # the controller's own output_max maps it back to radians.
        delta = np.clip((q - cur) / 0.5, -1.0, 1.0)
        self._step(np.concatenate([delta, [self._grip_cmd]]))

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        """Map wrc_demo's 0..1 gripper position onto LIBERO's -1/+1."""
        self._grip_cmd = self._grip_closed if pos > 0.5 else self._grip_open
        cur = self._joint_pos()
        for _ in range(12):                       # let the jaws actually move
            self._step(np.concatenate([np.zeros(self.n_joints),
                                       [self._grip_cmd]]))
        del cur

    # -- stepped-sim streaming ------------------------------------------

    def stream_to(self, q_target, duration_s: float, rate_hz: float = 50.0,
                  approve=None, settle_tol=None, settle_timeout_s: float = 12.0):
        """Min-jerk interpolation paced by SIM steps, not wall-clock sleep.

        ArmBase's version sleeps between waypoints so a real 50 Hz bus is not
        flooded. In LIBERO time only advances inside env.step(), so sleeping
        would burn wall time while the robot stands still, and the settle
        loop would then time out on a perfectly good motion.

        `settle_timeout_s` was 2.0, which gave 40 settle steps. MEASURED: this
        backend's JOINT_POSITION controller needs far longer than that to
        close the last of a 0.765 rad move, which is the size grasp_object's
        pregrasp asks for:

            extra steps    0     10     20     40     80    160    320
            residual     0.298  0.253  0.215  0.154  0.080  0.023  0.004

        It crossed the 0.05 rad tolerance somewhere between 80 and 160 steps,
        so every pregrasp move returned False at 40 and grasp_object reported
        "did not settle at pregrasp pose" after 8 retries. That is a harness
        bug, not a robot failure: it made libero_spatial score 0/100 in
        conditions that previously scored 18/50, and it predates this session
        (reproduced on 7c525cf).

        12.0 s gives 240 steps, comfortably past the measured 160 with margin
        for larger moves, and the loop still returns as soon as it converges
        so a fast joint costs nothing.
        """
        if settle_tol is None:
            settle_tol = self.settle_tol
        q_start = self.get_state().q.copy()
        q_target = np.asarray(q_target, float).reshape(-1)[: self.n_joints]
        steps = max(2, int(duration_s * rate_hz))
        dt = duration_s / steps
        q_prev = q_start
        for i in range(1, steps + 1):
            if self._stopped:
                return False
            s = min_jerk(i / steps)
            q_i = q_start + (q_target - q_start) * s
            if approve is not None:
                approve(q_prev, q_i, dt)          # SafetyViolation aborts
            self.send_joint_target(q_i)
            q_prev = q_i
        # settle by stepping, not sleeping
        for _ in range(int(settle_timeout_s * 20)):
            if float(np.abs(self.get_state().q - q_target).max()) < settle_tol:
                return True
            self.send_joint_target(q_target)
        return False


class LiberoCamera(CameraBase):
    """LIBERO's agentview as a wrc_demo RGB-D camera."""

    def __init__(self, env, cam_name: str = "agentview", res: int = 256):
        super().__init__()
        self.env = env
        self.cam = cam_name
        self.res = res
        self._K = None
        self._T = None
        #: latest (rgb, depth) published by the env thread
        import threading
        self._lock = threading.Lock()
        self._latest = (None, None)

    def open(self) -> None:
        self._refresh_calib()

    def _refresh_calib(self) -> None:
        """(Re)read intrinsics/extrinsics from the CURRENT sim.

        robosuite rebuilds `sim` on some resets, so caching the matrices once
        at open() is not enough -- and a stale handle surfaces as
        "'NoneType' object has no attribute 'model'" inside the stream
        thread, logged under the STREAM's name (which is the camera profile
        name, e.g. "mock"), not the camera class. That mislabelling is why
        this looked like the wrong backend for several iterations.
        """
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix, get_camera_intrinsic_matrix)
        sim = getattr(self.env, "sim", None)
        if sim is None:
            raise RuntimeError(
                "LiberoCamera: env.sim is None (env not reset yet?)")
        self._K = np.asarray(
            get_camera_intrinsic_matrix(sim, self.cam, self.res, self.res),
            dtype=float)
        self._T = np.asarray(get_camera_extrinsic_matrix(sim, self.cam),
                             dtype=float)

    def close(self) -> None:
        pass

    def has_depth(self) -> bool:
        return True

    def _grab(self) -> Frame:
        """Return the most recent rendered frame.

        CRITICAL: MuJoCo's offscreen render context is NOT thread-safe and is
        owned by the thread that created it. wrc_demo's CameraStream grabs on
        a worker thread, so calling sim.render() here corrupts the context --
        observed as a cascade of 'MjRenderContextOffscreen has no attribute
        con' / '_render_context_offscreen' errors and, eventually, a dead sim.

        So the ENV thread pushes frames in (see LiberoArm._step, which calls
        `publish`), and this method only hands over the latest one. That keeps
        every GL call on the thread that owns the context.
        """
        with self._lock:
            rgb, d = self._latest
        if rgb is None:
            raise RuntimeError("LiberoCamera: no frame published yet")
        return Frame(rgb=rgb.copy(), depth_m=None if d is None else d.copy(),
                     K=self._K.copy(),
                     depth_source="sensor" if d is not None else "none")

    def publish(self, obs: dict) -> None:
        """Called from the ENV thread with a fresh observation dict."""
        if not isinstance(obs, dict):
            return
        rgb = obs.get(f"{self.cam}_image")
        if rgb is None:
            # Surface the mismatch instead of silently never publishing: an
            # env built without camera_obs, or a different camera name, looks
            # exactly like a dead camera from the stream's side.
            if not getattr(self, "_warned_keys", False):
                self._warned_keys = True
                print(f"[LiberoCamera] no '{self.cam}_image' in obs; keys="
                      f"{sorted(k for k in obs if 'image' in k)}", flush=True)
            return
        # LIBERO renders upside down; wrc_demo (and every detector) expects the
        # scene the right way up. Same 180-degree flip the OpenVLA eval applies.
        rgb = np.ascontiguousarray(np.asarray(rgb)[::-1, ::-1, ::-1])  # +RGB->BGR
        d = None
        depth = obs.get(f"{self.cam}_depth")
        if depth is not None:
            try:
                from robosuite.utils.camera_utils import get_real_depth_map
                sim = getattr(self.env, "sim", None)
                dm = np.asarray(get_real_depth_map(sim, depth), dtype=np.float32)
                if dm.ndim == 3:
                    dm = dm[..., 0]
                d = np.ascontiguousarray(dm[::-1, ::-1])
            except Exception:
                d = None
        with self._lock:
            self._latest = (rgb, d)
