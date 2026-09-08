"""Rendered RGB-D camera looking into a live MuJoCo world (`type: mujoco`).

WHY THIS EXISTS. The MuJoCo arm gives the demo real physics without a robot,
but until now its only camera partner was the synthetic `mock` camera, which
re-renders the same painted cube every grab. Two consequences, both visible
to whoever is chatting with the robot:

  - the camera cannot witness the arm moving anything, so `pick_and_place`
    always ends `postcondition: unverified` ("only in the belief channel it
    also wrote") -- the demo's closed-loop verification, its headline
    feature, is switched off in the one sim anyone can run on a laptop;
  - a `camera_snapshot` shows a red square on a grey plane, not a robot.

This backend renders RGB + metric depth from a `<camera>` DECLARED in the
arm's MJCF (declared, not reconstructed: `sim/mujoco_rgbd.py` documents the
depth-relative-to-lookat trap that a posed free camera falls into) and reads
the SAME `MjData` the arm steps, through `sim/mujoco_world.py`. The prop that
the arm grasps is the prop the camera sees, at the pose physics says.

Frame conventions match every other camera backend: BGR uint8 colour, float32
metres depth aligned to colour, pinhole `K` from the profile's `fx` and the
image size. The cam->base extrinsic is the profile's `extrinsics.T`, which is
also what posed the MJCF `<camera>` (see `demo_scene.cameras_xml`), so the
renderer and the backprojection agree by construction rather than by
calibration.

THREADING. `mujoco.Renderer` binds an OpenGL context with `make_current()` in
`__init__` and in every `render()`, and GL contexts are thread-affine. The
CameraStream calls `open()` from the thread that builds the rig and `_grab()`
from its own pump thread, so the renderer is created LAZILY on the first grab
-- i.e. on the pump thread that will use it for the rest of its life.
`close()` runs on yet another thread; MuJoCo's `Renderer.close()` tolerates
that (measured), but a render never crosses threads.

WORLD ATTACHMENT. The arm is built before the camera in `build_runtime`, but
under the MCP server the arm is a LazyArm that materializes on the first
motion command -- so the camera may be first to load the world. Both orders
work: whoever is first loads the MJCF, the other attaches. Until the arm
realizes its home pose the render shows the MJCF default pose, which is the
honest state of a world nobody has commanded yet.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError

logger = logging.getLogger(__name__)

#: MuJoCo renders depth as distance along the camera axis, in metres,
#: already linearized by `mujoco.Renderer` -- but on the macOS GL path
#: (no ARB_clip_control) the far plane is quantized; anything at or beyond
#: it is "no return", the same semantics a depth sensor reports as 0.
_DEPTH_INVALID_FRAC = 0.999


class MujocoCamera(CameraBase):
    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        self._cfg = cfg if cfg is not None else Cfg({})
        c = self._cfg
        self.width = int(c.get("width", 640))
        self.height = int(c.get("height", 480))
        self.fx = float(c.get("fx", 600.0))
        self._camera_name = str(c.get("mj_camera", c.get("name", "cam")))
        # The world to render: planted by load_demo_config as `mj_scene`
        # (the primary MuJoCo arm's resolved MJCF path). A profile used raw
        # may name `mjcf` directly instead.
        self._scene = c.get("mj_scene") or c.get("mjcf")
        self._depth = bool(c.get("depth", True))
        self._world = None
        self._renderer = None
        self._render_owner: threading.Thread | None = None
        self._cam_id = -1
        self._opened = False
        self._lock = threading.Lock()
        self.K = np.array(
            [[self.fx, 0.0, self.width / 2.0],
             [0.0, self.fx, self.height / 2.0],
             [0.0, 0.0, 1.0]], dtype=np.float64,
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def has_depth(self) -> bool:
        return self._depth

    def open(self) -> None:
        if self._opened:
            return
        if not self._scene:
            raise CameraError(
                "mujoco camera has no world to render: the profile needs "
                "`mj_scene` (planted by load_demo_config when the primary arm "
                "is type: mujoco) or an explicit `mjcf`"
            )
        from ..sim import mujoco_world

        # MCP mode: the camera may be the first to need the generated scene
        # (the arm is lazy). Regenerate it from the loader-planted inputs --
        # deterministic content, identical to what the arm writes later.
        src = self._cfg.get("mj_scene_source")
        if src is not None and hasattr(src, "get"):
            from ..sim.demo_scene import write_demo_scene

            try:
                write_demo_scene(str(src.get("mjcf")), src.get("prop_cam"),
                                 cameras=src.get("cameras"))
            except FileNotFoundError as e:
                raise CameraError(
                    f"mujoco camera: robot MJCF directory missing ({e}); run "
                    "`python scripts/fetch_robot_assets.py so101` (or the matching robot)"
                ) from e
            except Exception as e:  # noqa: BLE001
                raise CameraError(f"mujoco camera could not generate its scene: {e}") from e
        try:
            self._world = mujoco_world.acquire(str(self._scene))
        except Exception as e:  # noqa: BLE001 - missing assets, bad MJCF
            raise CameraError(f"mujoco camera could not load {self._scene}: {e}") from e
        mj = self._world.mj
        self._cam_id = mj.mj_name2id(
            self._world.model, mj.mjtObj.mjOBJ_CAMERA, self._camera_name
        )
        if self._cam_id < 0:
            names = [mj.mj_id2name(self._world.model, mj.mjtObj.mjOBJ_CAMERA, i)
                     for i in range(self._world.model.ncam)]
            mujoco_world.release(self._world)
            self._world = None
            raise CameraError(
                f"no <camera name={self._camera_name!r}> in {self._scene} "
                f"(cameras present: {names}). A rendered profile is declared "
                "into the generated scene by the arm's mj_prop_from_camera "
                "path; a hand-written MJCF must declare it itself."
            )
        self._opened = True
        logger.info("mujoco camera %r attached to %s", self._camera_name, self._scene)

    def close(self) -> None:
        with self._lock:
            self._opened = False
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:  # noqa: BLE001
                    pass
                self._renderer = None
                self._render_owner = None
            if self._world is not None:
                from ..sim import mujoco_world

                mujoco_world.release(self._world)
                self._world = None

    # ── frames ───────────────────────────────────────────────────────────

    def _ensure_renderer(self):
        """Create the GL renderer on the CALLING thread (see THREADING).

        Ownership is tracked by Thread OBJECT, not `threading.get_ident()`:
        pthread ids are recycled, so a fresh thread can inherit a dead
        thread's id, be handed that thread's renderer, and block forever in
        `CGLLockContext` on a context the dead thread never unlocked
        (measured: the second of two short-lived grab threads hung for good).
        """
        me = threading.current_thread()
        if self._renderer is not None and self._render_owner is me:
            return self._renderer
        if self._renderer is not None:
            # A renderer bound to another thread's GL context: never reuse it
            # from here. Recreate rather than crash inside the driver.
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None
        try:
            self._renderer = self._world.mj.Renderer(
                self._world.model, height=self.height, width=self.width
            )
        except Exception as e:  # noqa: BLE001 - no GL on this box
            raise CameraError(f"mujoco offscreen renderer unavailable: {e}") from e
        self._render_owner = me
        return self._renderer

    def _grab(self) -> Frame:
        if not self._opened or self._world is None:
            raise CameraError("mujoco camera not opened")
        with self._lock:
            r = self._ensure_renderer()
            # Hold the WORLD lock across scene extraction: the arm steps
            # physics from the motion thread and a half-stepped MjData renders
            # as torn geometry. Rendering itself (GPU/CPU raster) happens
            # outside it -- update_scene copies what it needs.
            with self._world.lock:
                r.update_scene(self._world.data, camera=self._cam_id)
            rgb = r.render().copy()
            depth = None
            if self._depth:
                r.enable_depth_rendering()
                try:
                    with self._world.lock:
                        r.update_scene(self._world.data, camera=self._cam_id)
                    depth = r.render().astype(np.float32, copy=True)
                finally:
                    r.disable_depth_rendering()
                # Far-plane returns are "nothing there", not a distance.
                far = float(self._world.model.vis.map.zfar) * float(
                    self._world.model.stat.extent
                )
                if far > 0:
                    depth[depth >= far * _DEPTH_INVALID_FRAC] = 0.0
        return Frame(
            rgb=rgb[:, :, ::-1].copy(),   # renderer gives RGB; cascade Frames carry BGR
            depth_m=depth,
            K=self.K.copy(),
            depth_source="sensor" if depth is not None else "none",
        )

    # ── ground truth (used by sim/truth.py) ──────────────────────────────

    @property
    def world(self):
        return self._world
