"""Generate a MuJoCo scene whose prop matches what the camera profile sees.

WHY THIS EXISTS. `configs/arms/so101_mujoco.yaml` loads `scene.xml`, which is
the robot, a floor and a light -- and NO prop. Paired with `--camera
mock_small`, which *renders* a synthetic red cube, the demo perceives an
object that does not exist in physics: `_localize` returns a fix, IK solves,
the arm descends, and the jaws close on empty air. The run then reports

    air grasp: gripper closed fully, object not held

on every attempt (measured: 8/8, and the grasp memory had accumulated
`red_object|fp3cm: 0W/21L` from earlier runs). That is not a grasping bug --
`red_cube|fp6cm` sits at 69% on the same planner -- it is a scene/camera
mismatch, and it makes the headline `--arm so101_mujoco` demo unable to
complete a pick on a fresh clone.

The shipped `scene_box.xml` is not the fix either: its box is at x = 0.50,
while the SO-101's own profile puts the grasp ceiling at z = 0.05 and the
drop zone at r = 0.233. The prop is simply out of that arm's reach.

So the scene is GENERATED, for the same two reasons `sim/mujoco_rgbd.py`
generates its probe scene:

  - `assets/mjcf/` is gitignored (assets are fetched, not committed), so a
    scene file checked in there would vanish on a fresh clone;
  - `so101.xml` declares `<compiler meshdir="assets">` relative to ITS OWN
    directory, and a child `<compiler>` overrides an includer's -- a scene
    kept anywhere else fails to load every mesh.

The prop's pose is not hardcoded twice: `prop_pose_from_camera()` derives it
from the camera profile's own `box_px` + `extrinsics.T`, so the physics prop
lands where that camera says it sees one. Change the camera scene and the
MuJoCo prop follows; there is no second number to keep in sync.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Scene template: the robot's own MJCF plus a floor, a light and one prop.
#: Deliberately close to `scene.xml` so switching between them changes only
#: the prop, not lighting or ground material.
DEMO_SCENE_TEMPLATE = """<mujoco model="scene_demo">
  <include file="{robot_xml}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-20"/>
  </visual>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="demo_floor" size="0 0 0.05" pos="0 0 0" type="plane"/>
    <body name="{prop_name}" pos="{px:.6f} {py:.6f} {pz:.6f}">
      <freejoint/>
      <geom type="box" name="{prop_name}" size="{hx:.6f} {hy:.6f} {hz:.6f}"
        condim="3" friction="1 .03 .003" rgba="0.75 0.10 0.10 1"
        contype="2" conaffinity="1" solref="0.01 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def prop_pose_from_camera(cam_cfg) -> tuple[np.ndarray, np.ndarray]:
    """Where the prop must sit in BASE frame for `cam_cfg` to be telling the
    truth, and its half-extents.

    Backprojects the camera profile's `box_px` corners at the box's own
    surface depth, then transforms into base frame with the profile's
    `extrinsics.T`. Deriving it (rather than writing x = 0.20 in a second
    place) is the point: the physics prop cannot drift away from the pixels
    the detector is keyed to.
    """
    box_px = cam_cfg.get("box_px") or (280, 200, 360, 260)
    x0, y0, x1, y1 = (int(v) for v in box_px)
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    table_depth = float(cam_cfg.get("table_depth_m", 0.6))
    box_h = float(cam_cfg.get("box_height_m", 0.05))

    # MockCamera.synthetic_tabletop's intrinsics, verbatim.
    fx = 600.0
    cx, cy = width / 2.0, height / 2.0
    z = table_depth - box_h                     # depth of the box's TOP face

    def backproject(u, v):
        return np.array([(u - cx) / fx * z, (v - cy) / fx * z, z])

    corners_cam = np.array([backproject(u, v)
                            for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])

    extr = cam_cfg.get("extrinsics") or {}
    T = np.asarray(extr.get("T"), dtype=float).reshape(4, 4)
    pts_h = np.concatenate([corners_cam, np.ones((4, 1))], axis=1)
    corners_base = (pts_h @ T.T)[:, :3]

    centre_top = corners_base.mean(axis=0)
    span = corners_base.max(axis=0) - corners_base.min(axis=0)
    half = np.array([max(span[0], 1e-3) / 2, max(span[1], 1e-3) / 2, box_h / 2])
    # centre_top is the TOP face (that is the depth we backprojected); the body
    # origin is half a box lower.
    centre = np.array([centre_top[0], centre_top[1], box_h / 2])
    return centre, half


def write_demo_scene(robot_mjcf, cam_cfg, prop_name: str = "red_cube") -> Path:
    """Write a scene beside `robot_mjcf` whose prop matches `cam_cfg`.

    Returns the scene path. Written next to the robot MJCF so `meshdir`
    resolves (see this module's docstring).
    """
    robot_mjcf = Path(robot_mjcf)
    centre, half = prop_pose_from_camera(cam_cfg)
    scene = robot_mjcf.parent / "_scene_demo_generated.xml"
    scene.write_text(DEMO_SCENE_TEMPLATE.format(
        robot_xml=robot_mjcf.name,
        prop_name=prop_name,
        px=float(centre[0]), py=float(centre[1]), pz=float(centre[2]),
        hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
    ))
    return scene
