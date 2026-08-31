"""Offscreen RGB-D from an MJCF, with an exact extrinsic -- for verifying
perception against physics ground truth on any machine.

WHY THIS EXISTS. `grounding._recentre_by_size` (the fix for the 1.6-1.9 cm
grasp-target bias) was designed and measured entirely on the ISAAC rig: every
script under benchmark/diagnostics/ that produced those numbers hardcodes a
/home/johnny path and a bridge on :8611. A correction derived from ONE
simulator, verified by that same simulator, is exactly the kind of claim this
repo's "engine agreement is the gold metric" rule exists to distrust. This
module makes the same measurement available in MuJoCo, on a laptop, with
`data.xpos` of the prop body as an independent ground-truth channel.

TWO TRAPS, both hit and measured while building this -- do not "simplify" them
back:

1. **Use a camera DECLARED in the MJCF, never a reconstructed free camera.**
   Posing an `MjvCamera` by deriving azimuth/elevation/lookat from a transform
   silently renders depth relative to the LOOKAT PLANE rather than the eye. The
   tell was `reported + true_range = 1.9533` for every range: depth DECREASING
   as the object moved away. A declared `<camera>` gives MuJoCo the exact pose,
   and `data.cam_xpos`/`cam_xmat` then report it back so the extrinsic used for
   backprojection is the renderer's own, not a reconstruction of it.

2. **MuJoCo's camera frame is not OpenCV's.** `cam_xmat` columns are
   (right, up, backward) -- the camera looks down its **-z** with +y up, while
   `mask_to_points_cam` backprojects with +x right, +y DOWN, +z forward. The
   conversion is a 180 deg rotation about x (flip y and z). Skipping it puts
   every point behind the camera, which reads as a calibration error.

On macOS the renderer warns `ARB_clip_control unavailable ... depth accuracy
will be limited`; measured against geometry the residual is ~1e-3 m, well under
the centimetre-scale effects this bench is built to resolve.
"""

from __future__ import annotations

import numpy as np

#: Rendered-depth accuracy floor on the macOS GL path (no ARB_clip_control),
#: measured against known geometry. Method error below this is not meaningful.
DEPTH_NOISE_M = 1e-3

#: MuJoCo camera (right, up, backward) -> OpenCV (right, down, forward).
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


#: Verification scene: the robot's own MJCF plus a coloured prop and a declared
#: camera. Generated NEXT TO the fetched MJCF at runtime rather than shipped as
#: a file, because `so101.xml` declares `<compiler meshdir="assets">` relative
#: to ITS directory and that child `<compiler>` overrides an includer's -- a
#: scene kept anywhere else fails to open every mesh. (`assets/mjcf/` is also
#: gitignored, so a committed scene there would vanish on a fresh clone.)
PROBE_SCENE_TEMPLATE = """<mujoco model="scene_probe">
  <include file="{robot_xml}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="160" elevation="-20"/>
  </visual>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>
    <body name="probe_cube" pos="0.25 0.0 0.025">
      <freejoint/>
      <geom type="box" name="probe_cube" size="0.025 0.025 0.025" condim="3"
            friction="1 .03 .003" rgba="0.05 0.85 0.15 1" contype="2"
            conaffinity="1" solref="0.01 1"/>
    </body>
    {camera}
  </worldbody>
</mujoco>
"""


def write_probe_scene(robot_mjcf, eye=(0.70, -0.30, 0.50),
                      target=(0.25, 0.0, 0.025), fovy_deg: float = 58.0):
    """Write the verification scene beside `robot_mjcf` and return its path.

    Placed next to the robot MJCF so its `meshdir` resolves; the caller is
    responsible for cleaning it up (the test does, in a finally).
    """
    from pathlib import Path

    robot_mjcf = Path(robot_mjcf)
    scene = robot_mjcf.parent / "_scene_probe_generated.xml"
    scene.write_text(PROBE_SCENE_TEMPLATE.format(
        robot_xml=robot_mjcf.name,
        camera=camera_xml("scene", eye, target, fovy_deg),
    ))
    return scene


def look_at_xyaxes(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, str]:
    """(eye, `xyaxes` string) for an MJCF `<camera>` aimed at `target`.

    MJCF wants the camera's own x (right) and y (up) axes in world coordinates;
    it derives -z (the view direction) itself. Returning the string keeps the
    caller from hand-writing an axis order, which is where sign errors live.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    fwd = target - eye
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        raise ValueError("camera eye and target coincide")
    fwd /= n
    right = np.cross(fwd, np.asarray(up, dtype=float))
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        raise ValueError("camera up vector is parallel to the view direction")
    right /= rn
    cam_up = np.cross(right, fwd)
    vals = [*right, *cam_up]
    return eye, " ".join(f"{v:.9f}" for v in vals)


def intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """Pinhole K matching MuJoCo's vertical-FOV convention."""
    f = 0.5 * height / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=float
    )


def camera_xml(name: str, eye, target, fovy_deg: float = 58.0,
               up=(0.0, 0.0, 1.0)) -> str:
    """An MJCF `<camera>` element aimed at `target`."""
    eye, xyaxes = look_at_xyaxes(eye, target, up)
    return (f'<camera name="{name}" pos="{eye[0]:.6f} {eye[1]:.6f} {eye[2]:.6f}" '
            f'xyaxes="{xyaxes}" fovy="{fovy_deg}"/>')


class MujocoRGBD:
    """Render RGB + metric depth from a named camera in an MJCF.

    `camera` must name a `<camera>` in the model (see `camera_xml`); a free
    camera is deliberately not supported -- see trap 1 in the module docstring.
    """

    def __init__(self, mjcf_path, camera: str, width: int = 640, height: int = 480):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.cam_id < 0:
            raise ValueError(f"no <camera name={camera!r}> in {mjcf_path}")
        self.width, self.height = width, height
        fovy = float(self.model.cam_fovy[self.cam_id])
        self.K = intrinsics(width, height, fovy)
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:  # noqa: BLE001
            pass

    # ── ground truth ────────────────────────────────────────────────────

    def body_pos(self, name: str) -> np.ndarray:
        """World position of a body -- the truth channel perception cannot see."""
        mj = self._mj
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"no body {name!r} in the MJCF")
        return np.array(self.data.xpos[bid], dtype=float)

    def place_free_body(self, name: str, pos) -> None:
        """Teleport a free-jointed body, then refresh derived state."""
        mj = self._mj
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"no body {name!r} in the MJCF")
        jadr = int(self.model.body_jntadr[bid])
        if jadr < 0 or self.model.jnt_type[jadr] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError(f"body {name!r} has no free joint to teleport")
        qadr = int(self.model.jnt_qposadr[jadr])
        self.data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(self.model, self.data)

    # ── rendering ───────────────────────────────────────────────────────

    def extrinsic(self) -> np.ndarray:
        """4x4 cam->world in OPENCV axes, read back from the renderer's own
        camera state (never reconstructed -- see trap 2)."""
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[self.cam_id].reshape(3, 3) @ _MJ_TO_CV
        T[:3, 3] = self.data.cam_xpos[self.cam_id]
        return T

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """(rgb uint8 BGR, depth float32 metres from the camera)."""
        self._renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=self.cam_id)
        depth = self._renderer.render().copy().astype(np.float32)
        self._renderer.disable_depth_rendering()
        return rgb[:, :, ::-1].copy(), depth  # cascade Frames carry BGR

    def frame(self):
        """A cascade `Frame`, so the real perception stack can consume this."""
        from ..types import Frame

        rgb, depth = self.render()
        return Frame(rgb=rgb, depth_m=depth, K=self.K, depth_source="sensor")
