"""Generic MuJoCo arm backend: drive any MJCF through the same ArmBase API,
on either physics runtime MuJoCo ships.

Not SO-101-specific. A profile names the MJCF plus its joint and actuator
names, and this backend maps the framework's q vector onto them. That covers
every robot in MuJoCo Menagerie without new code.

WHY THIS EXISTS next to isaac_arm.py: Isaac Sim is a multi-gigabyte install
that needs an NVIDIA GPU, so on a laptop, a Jetson, or a CPU-only booth machine
"try it in sim first" is not available at all. MuJoCo installs from PyPI and
runs on CPU anywhere, which is what makes the framework's sim path device
agnostic rather than merely portable in principle.

TWO ENGINES behind one class, chosen by `engine:` in the profile:

  - `engine: mjc` (default) -- the MuJoCo C runtime (`mujoco.mj_step`). This is
    what actually "runs on any machine": for a single arm it does ~200k
    steps/s on an Apple laptop (measured 4.9 us/step, ~1000x real time), needs
    only the `mujoco` wheel, and has no accelerator dependency.

  - `engine: warp` -- MuJoCo Warp (`mujoco_warp.step`), the GPU-optimized
    reimplementation Google DeepMind + NVIDIA maintain under the Newton
    project. It exists so the SAME control code exercises the exact runtime the
    reference DGX/Jetson boxes accelerate. Two honest caveats, both measured:
      * MJWarp is a BATCHED engine. Its throughput comes from simulating many
        worlds at once on a GPU; a single arm is its worst case. On a CUDA-less
        Warp build (e.g. macOS arm64, where `warp-lang` reports only `cpu`) one
        arm runs ~3.2 ms/step -- ~650x slower than the C engine, though still
        ~1.5x real time, which is why it is usable for development here but is
        NOT the default. On the DGX it runs on the GPU.
      * First `mjw.step` on a fresh machine JIT-compiles every kernel (~15-30 s,
        then cached under ~/.cache/warp). connect() logs this so it does not
        read as a hang.
    This backend drives ONE world (`nworld=1`); MJWarp's many-world batching is
    for RL rollouts, a separate concern from closed-loop control.

Both engines are real physics, not the kinematic mock: contacts resist, a grasp
can slip, and position actuators have finite gain, so the arm ARRIVES LATE.
Profiles set `settle_tol`/`settle_timeout_s` accordingly (same reason the Isaac
profile does) and grasp success here means something the mock cannot tell you.

Joints and actuators are addressed BY NAME. Index-based addressing breaks the
moment a scene wrapper adds a free-floating prop, because that shifts every
qpos address after it -- and it shifts them silently. The name->address map is
built once from the MjModel and is engine-independent (MJWarp keeps the native
MjModel too), so exactly one mapping serves both runtimes.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase

logger = logging.getLogger(__name__)


# ── physics engines ──────────────────────────────────────────────────────
#
# Each engine owns the runtime-specific data structures and exposes a tiny
# array-oriented surface the arm drives. Everything above this line (name
# mapping, gripper policy, stop/hold, streaming) is shared, so adding the Warp
# runtime cost no duplication of the parts that are easy to get subtly wrong.
#
# The surface is deliberately BATCH-oriented (full qpos/qvel/ctrl vectors, not
# per-joint scalars): a per-scalar read would cross the host<->device boundary
# once per joint per tick under MJWarp, which is exactly the pattern that makes
# GPU physics slow.


class _MjcEngine:
    """The MuJoCo C runtime. `data` is the live simulation state.

    The model/data pair comes from `sim/mujoco_world.py`'s registry rather
    than a private load, so a rendered camera (`perception/mujoco_camera.py`)
    and the physics-truth channel (`sim/truth.py`) opened on the same MJCF
    see THIS arm's state -- a privately loaded copy would leave them watching
    a world in which the arm never moves.
    """

    kind = "mjc"

    def __init__(self, mjcf_path: str):
        try:
            import mujoco
        except ImportError as e:  # pragma: no cover - exercised via connect()
            raise RuntimeError(
                "the mujoco arm backend needs the `mujoco` package "
                "(pip install mujoco, or the [sim] extra)"
            ) from e
        from ..sim import mujoco_world

        self._mj = mujoco
        self._world = mujoco_world.acquire(str(mjcf_path))
        self.model = self._world.model
        self.data = self._world.data
        #: Shared with every reader of this world (camera stream thread,
        #: truth channel). MujocoArm takes it around each step/read.
        self.lock = self._world.lock
        self.device = "cpu"
        self._viewer = None

    def set_qpos(self, adr: int, value: float) -> None:
        self.data.qpos[adr] = value

    def realize(self, view: bool = False) -> None:
        # mj_forward turns the home pose we just wrote into consistent derived
        # state (site/body xpos) before the first command or FK read.
        self._mj.mj_forward(self.model, self.data)
        self._world.realized = True
        self._view_wanted = bool(view)
        self._view_retry_at = 0.0
        if view:
            self._open_viewer()

    def pull(self) -> tuple[np.ndarray, np.ndarray]:
        # Live numpy views; the caller copies out the addresses it wants.
        return self.data.qpos, self.data.qvel

    def push_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl

    def step(self, n: int) -> None:
        for _ in range(n):
            self._mj.mj_step(self.model, self.data)
        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:  # noqa: BLE001
                self._viewer = None
        elif getattr(self, "_view_wanted", False):
            # The window was wanted but skipped (display asleep when the arm
            # connected -- the launcher's proof turn runs with the lid shut
            # often enough). Re-check at most every few seconds of wall time
            # so it appears on the first motion AFTER the screen wakes.
            now = time.monotonic()
            if now >= self._view_retry_at:
                self._view_retry_at = now + 5.0
                if _display_unavailable_reason() is None:
                    self._open_viewer()
                    if self._viewer is None:
                        self._view_wanted = False  # a real failure, not a sleeping display

    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def _open_viewer(self) -> None:
        blocker = _display_unavailable_reason()
        if blocker:
            # A window is a nicety; the process hosting the visitor's session is
            # not. mujoco's simulate reads GLFW's monitor[0] unconditionally,
            # and GLFW builds that list from CGGetActiveDisplayList -- which
            # is EMPTY while the display is asleep/locked or the lid is shut
            # (measured: 1 online display, 0 active, launch_passive -> SIGSEGV
            # in _glfwGetVideoModeCocoa, taking the whole MCP server down mid
            # tool call). Refuse to open rather than crash; say why.
            print(f"[mujoco] viewer skipped: {blocker}; running headless", file=sys.stderr)
            self._viewer = None
            return
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("[mujoco] viewer window open", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - a headless box has no display
            # stderr, not logging: nothing configures the logging tree in the
            # MCP server, so the warning vanished and "no window" had no cause
            # in the run log. (macOS: "launch_passive requires mjpython".)
            print(f"[mujoco] viewer unavailable ({e}); running headless", file=sys.stderr)
            self._viewer = None
            self._view_wanted = False

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None
        if self._world is not None:
            from ..sim import mujoco_world

            mujoco_world.release(self._world)
            self._world = None


def _display_unavailable_reason() -> str | None:
    """Why opening a native window would fail or CRASH here, or None if it is
    safe to try. Cheap (one CoreGraphics call on macOS, env checks elsewhere);
    runs before every viewer launch because the answer changes at runtime
    (display sleep, lid, screen lock)."""
    import os
    import platform

    system = platform.system()
    if system == "Darwin":
        try:
            import ctypes
            import ctypes.util

            cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
            n = ctypes.c_uint32(0)
            arr = (ctypes.c_uint32 * 16)()
            if cg.CGGetActiveDisplayList(16, arr, ctypes.byref(n)) != 0:
                return "CoreGraphics cannot list displays"
            if n.value == 0:
                return ("no ACTIVE display (screen asleep/locked or lid closed) -- GLFW has "
                        "no monitor and mujoco's viewer would segfault the process")
        except Exception as e:  # noqa: BLE001 - no CoreGraphics = not a desktop session
            return f"no window server ({e})"
        return None
    if system == "Linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no DISPLAY/WAYLAND_DISPLAY"
    return None


def _select_warp_device(wp, requested: str) -> str:
    """Resolve `device:` for the Warp engine, degrading like the rest of the
    stack: an explicit accelerator the machine does not have WARNS and falls
    back to CPU rather than killing the run (README "Compute" contract)."""
    req = (requested or "auto").strip().lower()
    try:
        cuda = bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001
        cuda = False
    if req in ("", "auto"):
        return "cuda:0" if cuda else "cpu"
    if req.startswith("cuda"):
        if cuda:
            return requested  # honor an explicit index like cuda:1
        logger.warning(
            "engine=warp asked for device %r but Warp reports no CUDA on this "
            "host; falling back to cpu (dev/debug mode -- one arm on MJWarp CPU "
            "is ~650x slower than engine=mjc, see mujoco_arm.py)", requested
        )
        return "cpu"
    return "cpu"


class _WarpEngine:
    """MuJoCo Warp, single world. Native MjModel + MjData stage the home pose
    on the host; the authoritative state lives in the Warp `mjw.Data` on the
    selected device after realize()."""

    kind = "warp"

    def __init__(self, mjcf_path: str, device: str = "auto"):
        try:
            import mujoco
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "engine=warp needs the `mujoco` package (pip install mujoco)"
            ) from e
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "engine=warp needs `mujoco-warp` and `warp-lang` "
                "(pip install mujoco-warp, or the [sim-warp] extra)"
            ) from e
        self._mj = mujoco
        self._mjw = mjw
        self._wp = wp
        wp.init()
        self.model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).resolve()))
        # MJWarp's iterative linesearch warns (once per step) below 20 on some
        # contact-rich MJCFs; raise the floor so a real run is not drowned in
        # "increase ls_iterations" lines. Never lowers a model that asked for
        # more.
        self.model.opt.ls_iterations = max(int(self.model.opt.ls_iterations), 20)
        self.data = mujoco.MjData(self.model)  # host staging buffer
        self.device = _select_warp_device(wp, device)
        self._m = None
        self._d = None
        self._ctrl_dtype = np.float32
        self._viewer = None
        self._get_data_into = getattr(mjw, "get_data_into", None)

    def set_qpos(self, adr: int, value: float) -> None:
        self.data.qpos[adr] = value  # stage on the host before device upload

    def realize(self, view: bool = False) -> None:
        self._mj.mj_forward(self.model, self.data)
        logger.info(
            "engine=warp: uploading to %s and compiling MJWarp kernels "
            "(first run on a fresh machine ~15-30s, then cached)", self.device
        )
        with self._wp.ScopedDevice(self.device):
            self._m = self._mjw.put_model(self.model)
            self._d = self._mjw.put_data(self.model, self.data)
        self._ctrl_dtype = self._d.ctrl.numpy().dtype
        if view:
            self._open_viewer()

    def pull(self) -> tuple[np.ndarray, np.ndarray]:
        self._wp.synchronize()
        return self._d.qpos.numpy()[0], self._d.qvel.numpy()[0]

    def push_ctrl(self, ctrl: np.ndarray) -> None:
        # `d.ctrl` is (nworld, nu); world 0 is the only one we drive.
        self._d.ctrl.assign(ctrl.reshape(1, -1).astype(self._ctrl_dtype))

    def step(self, n: int) -> None:
        with self._wp.ScopedDevice(self.device):
            for _ in range(n):
                self._mjw.step(self._m, self._d)
        self._wp.synchronize()
        if self._viewer is not None:
            self._sync_viewer()

    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def _open_viewer(self) -> None:
        # The passive viewer renders an MjData. Under Warp the live state is on
        # the device, so a window needs get_data_into() to copy it back each
        # frame. Where that API is absent, run headless rather than show a
        # frozen home pose.
        if self._get_data_into is None:
            logger.warning(
                "engine=warp: this mujoco_warp build has no get_data_into(); "
                "viewer disabled (use engine=mjc for a window, or the dashboard)"
            )
            return
        blocker = _display_unavailable_reason()
        if blocker:
            print(f"[mujoco] viewer skipped: {blocker}; running headless", file=sys.stderr)
            return
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("[mujoco] viewer window open", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[mujoco] viewer unavailable ({e}); running headless", file=sys.stderr)
            self._viewer = None

    def _sync_viewer(self) -> None:
        try:
            self._get_data_into(self.data, self._m, self._d)
            self._viewer.sync()
        except Exception:  # noqa: BLE001
            self._viewer = None

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None


_ENGINES = {
    "mjc": _MjcEngine, "c": _MjcEngine, "mujoco": _MjcEngine, "": _MjcEngine,
    "warp": _WarpEngine, "mjwarp": _WarpEngine, "mjw": _WarpEngine,
}


def _make_engine(kind: str, mjcf_path: str, device: str):
    try:
        cls = _ENGINES[kind]
    except KeyError:
        raise ValueError(
            f"unknown mujoco engine {kind!r} (mjc|warp)"
        ) from None
    if cls is _WarpEngine:
        return cls(mjcf_path, device)
    return cls(mjcf_path)


class MujocoArm(ArmBase):
    def __init__(self, cfg: Cfg, kinematics=None):
        self._cfg = cfg
        self.kin = kinematics
        self.n_joints = int(cfg.get("n_joints", 6))
        self._mjcf = str(cfg.mjcf)
        # Optional: build a scene whose prop matches what the CAMERA renders.
        # `scene.xml` has no prop, so pairing it with a synthetic camera makes
        # the arm reach for an object that does not exist in physics and close
        # on air. See sim/demo_scene.py. Off unless the profile asks, so a rig
        # profile pointing at a real scene is never silently rewritten.
        #
        # The profile authors the OPT-IN as `mj_prop_from_camera: true`;
        # load_demo_config REPLACES that bool with the camera profile dict at
        # load time (config.py) because only the loader knows which camera the
        # run opens. A profile loaded directly (load_profile in tests, or a
        # bare MujocoArm(cfg)) still carries the bool, and a bool names no
        # camera to derive a prop from -- so only a mapping triggers scene
        # generation, and the bare flag means "no camera known, keep the
        # profile's own scene".
        prop_cam = cfg.get("mj_prop_from_camera")
        if prop_cam is not None and not isinstance(prop_cam, (bool,)):
            from ..sim.demo_scene import resolved_scene_path, write_demo_scene

            # `mj_cameras` (planted by the loader) is the run's whole camera
            # list: every rendered profile gets a <camera> in the scene.
            written = write_demo_scene(self._mjcf, prop_cam,
                                       cameras=cfg.get("mj_cameras"))
            expected = resolved_scene_path(self._mjcf, prop_cam)
            # The camera backend and truth channel find this world by the
            # path `resolved_scene_path` predicts; a mismatch here would
            # split the rig across two worlds silently.
            assert Path(written).resolve() == expected.resolve(), (written, expected)
            self._mjcf = str(written)
        self._joint_names = list(cfg.get("mj_joints") or [])
        self._act_names = list(cfg.get("mj_actuators") or self._joint_names)
        self._grip_joint = cfg.get("mj_gripper_joint")
        self._grip_act = cfg.get("mj_gripper_actuator") or self._grip_joint
        self._substeps = max(1, int(cfg.get("substeps", 10)))
        # Profile `view:` is the default; CASCADE_MJ_VIEW=1|0 overrides it
        # per run. The one-click launcher sets it for sim runs: the banner
        # promised "the MuJoCo window opens on the first motion" while the
        # profile said `view: false` and nothing ever flipped it -- measured
        # on a visitor run: no window, no log line, mjpython for nothing.
        env_view = os.environ.get("CASCADE_MJ_VIEW")
        self._view = (env_view != "0") if env_view is not None else bool(cfg.get("view", False))
        self.settle_tol = float(cfg.get("settle_tol", 0.03))
        self.settle_timeout_s = float(cfg.get("settle_timeout_s", 4.0))
        # Which physics runtime backs this arm. `mjc` (C engine) is the default
        # because it is the one that runs fast on any machine; `warp` opts into
        # MuJoCo Warp on the same profile (see the module docstring's tradeoff).
        # Validated HERE, not at connect(): an unknown engine is a profile
        # typo, knowable without touching MuJoCo or the assets -- deferring it
        # to connect() buries it under the missing-MJCF error on any machine
        # that has not fetched meshes (CI, fresh clones).
        self._engine_kind = str(cfg.get("engine", "mjc")).strip().lower()
        if self._engine_kind not in _ENGINES:
            raise ValueError(
                f"unknown mujoco engine {self._engine_kind!r} (mjc|warp)"
            )
        self._device_req = str(cfg.get("device", "auto"))
        g = cfg.get("gripper") or {}
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._grip_settle_s = float(cfg.get("gripper_settle_s", 0.6))
        # Gripper CONTROL units vs JOINT units. A position actuator on the jaw
        # joint takes the joint value itself (SO-101, PiPER: identity). The
        # Franka Hand in Menagerie is different: one tendon actuator whose
        # ctrl is 0..255 while each finger joint reads 0..0.04 m. The profile
        # declares the affine map ctrl = scale * pos + offset; `gripper_pos`
        # keeps reporting the JOINT value so the skill layer's stall/width
        # arithmetic (which is in joint units) is unaffected.
        self._grip_ctrl_scale = float(cfg.get("mj_gripper_ctrl_scale", 1.0))
        self._grip_ctrl_offset = float(cfg.get("mj_gripper_ctrl_offset", 0.0))

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

        self._engine = None
        self._model = None
        self._data = None
        self._ctrl: np.ndarray | None = None
        self._qadr: list[int] = []
        self._dadr: list[int] = []
        self._aidx: list[int] = []
        self._grip_qadr: int | None = None
        self._grip_aidx: int | None = None
        self._connected = False
        self._stopped = False
        # One mutex over the engine: the perception loop reads state from its
        # own thread while the motion path steps physics, and neither runtime's
        # data is reentrant. Without this the reader sees half-stepped state.
        self._lock = threading.RLock()

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        path = Path(self._mjcf)
        if not path.exists():
            raise RuntimeError(
                f"MJCF not found: {path}. MuJoCo needs the mesh assets too, "
                f"which are not vendored -- run "
                f"`python scripts/fetch_robot_assets.py so101` (or the matching "
                f"robot) to download them."
            )
        engine = _make_engine(self._engine_kind, str(path), self._device_req)
        self._engine = engine
        self._model = engine.model
        self._data = engine.data
        # The C engine's world is SHARED (sim/mujoco_world.py): a rendered
        # camera and the truth channel read the same MjData from other
        # threads, so the arm must serialize on the world's lock, not a
        # private one -- two locks over one struct is no lock at all.
        shared = getattr(engine, "lock", None)
        if shared is not None:
            self._lock = shared
        mj = engine._mj  # the mujoco module, for name->id lookups

        self._qadr, self._dadr = [], []
        for name in self._joint_names:
            jid = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint {name!r} not in {path.name}")
            self._qadr.append(int(self._model.jnt_qposadr[jid]))
            self._dadr.append(int(self._model.jnt_dofadr[jid]))
        self._aidx = [self._actuator_id(mj, name) for name in self._act_names]
        if self._grip_joint:
            jid = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_JOINT, str(self._grip_joint))
            if jid < 0:
                raise RuntimeError(f"gripper joint {self._grip_joint!r} not in {path.name}")
            self._grip_qadr = int(self._model.jnt_qposadr[jid])
            self._grip_aidx = self._actuator_id(mj, str(self._grip_act))

        self._ctrl = np.zeros(int(self._model.nu), dtype=float)

        # Start AT the home pose rather than the MJCF keyframe. Grasp IK is
        # seeded from home_q, and starting somewhere else makes the first
        # command a long unplanned sweep across the table. Staged on the host
        # data, then realize() makes it live (and, under Warp, uploads it).
        home = self._cfg.get("home_q")
        with self._lock:
            if home is not None:
                for adr, v in zip(self._qadr, np.asarray(home, dtype=float)):
                    engine.set_qpos(adr, float(v))
            if self._grip_qadr is not None:
                engine.set_qpos(self._grip_qadr, self._grip_open)
            engine.realize(view=self._view)
            self._hold_current()
        self._connected = True
        self._stopped = False
        logger.info(
            "mujoco arm up: %s (engine=%s device=%s, %d joints, %d substeps/waypoint)",
            path.name, engine.kind, engine.device, self.n_joints, self._substeps,
        )

    def _actuator_id(self, mj, name: str) -> int:
        aid = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"actuator {name!r} not in the MJCF")
        return int(aid)

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.close()  # releases the shared world (last holder drops it)
            self._engine = None
        self._connected = False

    @property
    def mjcf_path(self) -> str:
        """The scene this arm simulates (post scene generation), so a camera
        or truth channel can attach to the SAME world by path."""
        return self._mjcf

    @property
    def world(self):
        """The shared `MujocoWorld` while connected on the C engine, else
        None (Warp keeps device-side state; nothing can attach to it)."""
        return getattr(self._engine, "_world", None) if self._engine is not None else None

    # ── state and commands ───────────────────────────────────────────────

    def get_state(self) -> RobotState:
        if not self._connected:
            raise RuntimeError("mujoco arm not connected")
        with self._lock:
            qpos, qvel = self._engine.pull()
            q = np.array([qpos[a] for a in self._qadr], dtype=float)
            dq = np.array([qvel[a] for a in self._dadr], dtype=float)
            grip = (float(qpos[self._grip_qadr])
                    if self._grip_qadr is not None else 0.0)
        return RobotState(q=q, dq=dq, gripper_pos=grip, t=time.monotonic())

    def _hold_current(self) -> None:
        """Point every actuator at where its joint already is.

        Called on connect and on stop. A position actuator whose ctrl is left
        at 0 drives the arm to 0, so "stop" without this would be a fast move
        to the zero pose -- the opposite of stopping.
        """
        qpos, _ = self._engine.pull()
        for adr, aid in zip(self._qadr, self._aidx):
            self._ctrl[aid] = qpos[adr]
        if self._grip_aidx is not None and self._grip_qadr is not None:
            self._ctrl[self._grip_aidx] = self._grip_ctrl(qpos[self._grip_qadr])
        self._engine.push_ctrl(self._ctrl)

    def _grip_ctrl(self, pos: float) -> float:
        """Jaw JOINT value -> actuator ctrl (see mj_gripper_ctrl_scale)."""
        return self._grip_ctrl_scale * float(pos) + self._grip_ctrl_offset

    def send_joint_target(self, q: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("mujoco arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)
        with self._lock:
            for aid, v in zip(self._aidx, q[: self.n_joints]):
                self._ctrl[aid] = float(v)
            self._engine.push_ctrl(self._ctrl)
            self._engine.step(self._substeps)

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._stopped or self._grip_aidx is None:
            return
        with self._lock:
            self._ctrl[self._grip_aidx] = self._grip_ctrl(pos)
            self._engine.push_ctrl(self._ctrl)
            # Unlike a joint move, nothing else is stepping physics while the
            # jaws travel, so the command has to carry its own settle time or
            # the caller reads a gripper that never moved. Contact is what
            # stops it early -- that stall IS the grasp signal the skill layer
            # reads back through `gripper_pos`.
            self._engine.step(max(1, int(self._grip_settle_s / self._engine.timestep())))

    def stop(self) -> None:
        """Soft stop: freeze in place, torque stays on (nothing falls)."""
        self._stopped = True
        if self._connected:
            with self._lock:
                self._hold_current()
                self._engine.step(1)

    def wait_settled(self, q_target: np.ndarray, tol: float,
                     timeout_s: float) -> bool:
        """Step the sim while waiting -- the base-class poll watches a FROZEN
        world here.

        Physics only advances inside this backend's own calls (same trap
        set_gripper documents): after the stream's last waypoint the world
        stops, so polling `get_state()` re-reads the same qpos until the
        wall-clock timeout and any joint that needed more sim time than that
        final waypoint's substeps reports "did not settle" -- observed as
        shoulder_lift 0.143->0.062 rad short of a pregrasp with 4 s of wall
        time and 0 s of sim time. A real arm's servos keep acting during the
        settle window; stepping here is the sim's equivalent. Honors stop():
        a latched e-stop must freeze the world, not keep integrating it.
        """
        deadline = time.monotonic() + timeout_s
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        while time.monotonic() < deadline:
            st = self.get_state()  # raises if not connected -> engine exists
            if np.abs(st.q - q_target[: self.n_joints]).max() < tol:
                return True
            if self._stopped:
                return False
            engine = self._engine
            if engine is None:  # disconnected mid-wait
                return False
            with self._lock:
                engine.step(self._substeps)
        return False

    def resume(self) -> None:
        self._stopped = False
