"""Generic MuJoCo arm backend: drive any MJCF through the same ArmBase API.

Not SO-101-specific. A profile names the MJCF plus its joint and actuator
names, and this backend maps the framework's q vector onto them. That covers
every robot in MuJoCo Menagerie without new code.

WHY THIS EXISTS next to isaac_arm.py: Isaac Sim is a multi-gigabyte install
that needs an NVIDIA GPU, so on a laptop, a Jetson, or a CPU-only booth machine
"try it in sim first" is not available at all. MuJoCo installs from PyPI and
runs on CPU anywhere, which is what makes the framework's sim path device
agnostic rather than merely portable in principle.

It is real physics, not the kinematic mock: contacts resist, a grasp can slip,
and position actuators have finite gain, so the arm ARRIVES LATE. Profiles set
`settle_tol`/`settle_timeout_s` accordingly (same reason the Isaac profile does)
and grasp success here means something the mock cannot tell you.

Joints and actuators are addressed BY NAME. Index-based addressing breaks the
moment a scene wrapper adds a free-floating prop, because that shifts every
qpos address after it -- and it shifts them silently.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase

logger = logging.getLogger(__name__)


class MujocoArm(ArmBase):
    def __init__(self, cfg: Cfg, kinematics=None):
        self._cfg = cfg
        self.kin = kinematics
        self.n_joints = int(cfg.get("n_joints", 6))
        self._mjcf = str(cfg.mjcf)
        self._joint_names = list(cfg.get("mj_joints") or [])
        self._act_names = list(cfg.get("mj_actuators") or self._joint_names)
        self._grip_joint = cfg.get("mj_gripper_joint")
        self._grip_act = cfg.get("mj_gripper_actuator") or self._grip_joint
        self._substeps = max(1, int(cfg.get("substeps", 10)))
        self._view = bool(cfg.get("view", False))
        self.settle_tol = float(cfg.get("settle_tol", 0.03))
        self.settle_timeout_s = float(cfg.get("settle_timeout_s", 4.0))
        g = cfg.get("gripper") or {}
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._grip_settle_s = float(cfg.get("gripper_settle_s", 0.6))

        if len(self._joint_names) != self.n_joints:
            raise ValueError(
                f"mj_joints has {len(self._joint_names)} names but n_joints="
                f"{self.n_joints}; they address the same q vector"
            )
        if len(self._act_names) != self.n_joints:
            raise ValueError(
                f"mj_actuators has {len(self._act_names)} names but n_joints="
                f"{self.n_joints}"
            )

        self._mj = None
        self._model = None
        self._data = None
        self._viewer = None
        self._qadr: list[int] = []
        self._dadr: list[int] = []
        self._aidx: list[int] = []
        self._grip_qadr: int | None = None
        self._grip_aidx: int | None = None
        self._connected = False
        self._stopped = False
        # One mutex over model+data: the perception loop reads state from its
        # own thread while the motion path steps physics, and mjData is not
        # reentrant. Without this the reader sees half-stepped state.
        self._lock = threading.RLock()

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        try:
            import mujoco  # lazy: the mock stack must not need it
        except ImportError as e:
            raise RuntimeError(
                "the mujoco arm backend needs the `mujoco` package "
                "(pip install mujoco)"
            ) from e
        from pathlib import Path

        path = Path(self._mjcf)
        if not path.exists():
            raise RuntimeError(
                f"MJCF not found: {path}. MuJoCo needs the mesh assets too, "
                f"which are not vendored -- run "
                f"`python scripts/fetch_robot_assets.py so101` (or the matching "
                f"robot) to download them."
            )
        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)

        self._qadr, self._dadr = [], []
        for name in self._joint_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint {name!r} not in {path.name}")
            self._qadr.append(int(self._model.jnt_qposadr[jid]))
            self._dadr.append(int(self._model.jnt_dofadr[jid]))
        self._aidx = [self._actuator_id(name) for name in self._act_names]
        if self._grip_joint:
            jid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, str(self._grip_joint)
            )
            if jid < 0:
                raise RuntimeError(f"gripper joint {self._grip_joint!r} not in {path.name}")
            self._grip_qadr = int(self._model.jnt_qposadr[jid])
            self._grip_aidx = self._actuator_id(str(self._grip_act))

        # Start AT the home pose rather than the MJCF keyframe. Grasp IK is
        # seeded from home_q, and starting somewhere else makes the first
        # command a long unplanned sweep across the table.
        home = self._cfg.get("home_q")
        with self._lock:
            if home is not None:
                for adr, v in zip(self._qadr, np.asarray(home, dtype=float)):
                    self._data.qpos[adr] = float(v)
            if self._grip_qadr is not None:
                self._data.qpos[self._grip_qadr] = self._grip_open
            mujoco.mj_forward(self._model, self._data)
            self._hold_current()
        self._connected = True
        self._stopped = False
        if self._view:
            self._open_viewer()
        logger.info("mujoco arm up: %s (%d joints, %d substeps/waypoint)",
                    path.name, self.n_joints, self._substeps)

    def _actuator_id(self, name: str) -> int:
        aid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"actuator {name!r} not in the MJCF")
        return int(aid)

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        except Exception as e:  # noqa: BLE001 - a headless box has no display
            logger.warning("mujoco viewer unavailable (%s); running headless", e)
            self._viewer = None

    def disconnect(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None
        self._connected = False

    # ── state and commands ───────────────────────────────────────────────

    def get_state(self) -> RobotState:
        if not self._connected:
            raise RuntimeError("mujoco arm not connected")
        with self._lock:
            q = np.array([self._data.qpos[a] for a in self._qadr], dtype=float)
            dq = np.array([self._data.qvel[a] for a in self._dadr], dtype=float)
            grip = (float(self._data.qpos[self._grip_qadr])
                    if self._grip_qadr is not None else 0.0)
        return RobotState(q=q, dq=dq, gripper_pos=grip, t=time.monotonic())

    def _hold_current(self) -> None:
        """Point every actuator at where its joint already is.

        Called on connect and on stop. A position actuator whose ctrl is left
        at 0 drives the arm to 0, so "stop" without this would be a fast move
        to the zero pose -- the opposite of stopping.
        """
        for adr, aid in zip(self._qadr, self._aidx):
            self._data.ctrl[aid] = self._data.qpos[adr]
        if self._grip_aidx is not None and self._grip_qadr is not None:
            self._data.ctrl[self._grip_aidx] = self._data.qpos[self._grip_qadr]

    def _step(self, n: int) -> None:
        for _ in range(n):
            self._mj.mj_step(self._model, self._data)
        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:  # noqa: BLE001
                self._viewer = None

    def send_joint_target(self, q: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("mujoco arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)
        with self._lock:
            for aid, v in zip(self._aidx, q[: self.n_joints]):
                self._data.ctrl[aid] = float(v)
            self._step(self._substeps)

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._stopped or self._grip_aidx is None:
            return
        with self._lock:
            self._data.ctrl[self._grip_aidx] = float(pos)
            # Unlike a joint move, nothing else is stepping physics while the
            # jaws travel, so the command has to carry its own settle time or
            # the caller reads a gripper that never moved. Contact is what
            # stops it early -- that stall IS the grasp signal the skill layer
            # reads back through `gripper_pos`.
            self._step(max(1, int(self._grip_settle_s / self._model.opt.timestep)))

    def stop(self) -> None:
        """Soft stop: freeze in place, torque stays on (nothing falls)."""
        self._stopped = True
        if self._connected:
            with self._lock:
                self._hold_current()
                self._step(1)

    def resume(self) -> None:
        self._stopped = False
