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

RENDERED CAMERAS (2026-09-09). A `type: mujoco` camera profile
(`perception/mujoco_camera.py`) renders RGB-D out of this same world instead
of drawing a synthetic cube. It needs a `<camera>` DECLARED in the MJCF --
see `sim/mujoco_rgbd.py` trap 1 for why a reconstructed free camera renders
depth relative to the wrong plane. `write_demo_scene(..., cameras=[...])`
therefore emits one `<camera>` per rendered profile, posed from that
profile's `extrinsics.T` (OpenCV axes -> MuJoCo's right/up/back, the trap-2
conversion) and with `fovy` derived from the profile's `fx`/`height`. The
extrinsic the perception stack backprojects with and the pose the renderer
draws from are then the SAME matrix, by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: MuJoCo camera axes are (right, up, backward); OpenCV's are (right, down,
#: forward). Same constant as `sim/mujoco_rgbd.py` -- kept local so this
#: module stays importable without the renderer.
_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def camera_xml_from_extrinsic(name: str, T_cam2base, fx: float, height: int) -> str:
    """An MJCF `<camera>` posed exactly where a cascade camera profile says
    the camera is (its `extrinsics.T`, OpenCV convention, cam -> base).

    `fovy` is MuJoCo's only intrinsic; it is derived from the profile's
    focal length so the rendered image and the profile's K agree pixel for
    pixel (the camera backend re-derives K from the same numbers).
    """
    T = np.asarray(T_cam2base, dtype=float).reshape(4, 4)
    R_mj = T[:3, :3] @ _CV_TO_MJ            # columns: right, up, backward
    right, up = R_mj[:, 0], R_mj[:, 1]
    fovy = float(np.degrees(2.0 * np.arctan(float(height) / (2.0 * float(fx)))))
    pos = T[:3, 3]
    xyaxes = " ".join(f"{v:.9f}" for v in (*right, *up))
    return (f'<camera name="{name}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
            f'xyaxes="{xyaxes}" fovy="{fovy:.6f}"/>')


def cameras_xml(cam_cfgs) -> str:
    """`<camera>` elements for every `type: mujoco` profile in `cam_cfgs`.

    Non-rendered profiles (mock, realsense, ...) are skipped: they do not
    look into this world. A rendered profile without `extrinsics.T` is an
    authoring error, not something to default -- a camera at the origin
    looking nowhere useful would silently produce all-floor frames.
    """
    out = []
    for c in cam_cfgs or []:
        if str(c.get("type", "")) != "mujoco":
            continue
        extr = c.get("extrinsics") or {}
        T = extr.get("T") if hasattr(extr, "get") else None
        if T is None:
            raise ValueError(
                f"camera profile {c.get('name', '?')!r} is type: mujoco but has no "
                "extrinsics.T -- the renderer needs the pose to declare a <camera>"
            )
        out.append(camera_xml_from_extrinsic(
            str(c.get("mj_camera", c.get("name", "cam"))), T,
            fx=float(c.get("fx", 600.0)), height=int(c.get("height", 480)),
        ))
    return "\n    ".join(out)

#: Scene template: the robot's own MJCF plus a floor, a light and one prop.
#: Deliberately close to `scene.xml` so switching between them changes only
#: the prop, not lighting or ground material. `{cameras}` receives the
#: `<camera>` elements of any rendered profiles (see `cameras_xml`).
DEMO_SCENE_TEMPLATE = """<mujoco model="scene_demo">
  <include file="{robot_xml}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-20"/>
  </visual>

  <asset>
    <texture type="2d" name="demo_table" builtin="checker" mark="edge"
      rgb1="0.86 0.86 0.84" rgb2="0.78 0.78 0.76" markrgb="0.6 0.6 0.6"
      width="300" height="300"/>
    <material name="demo_table" texture="demo_table" texuniform="true"
      texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="demo_floor" size="0 0 0.05" pos="0 0 0" type="plane" material="demo_table"/>
    {props}
    {cameras}
  </worldbody>
</mujoco>
"""


#: One free-floating box prop. `rgba` is what the renderer paints; the mock
#: detector (perception/detector.py) keys its colour thresholds to the same
#: `color` word, and the physics truth channel matches labels on the body
#: name -- so a prop declared once in the camera profile is consistent across
#: pixels, physics and verdicts by construction.
PROP_XML = """<body name="{name}" pos="{px:.6f} {py:.6f} {pz:.6f}">
      <freejoint/>
      <geom type="box" name="{name}" size="{hx:.6f} {hy:.6f} {hz:.6f}"
        condim="3" friction="1 .03 .003" rgba="{rgba}"
        contype="2" conaffinity="1" solref="0.01 1"/>
    </body>"""

#: Colour word -> rgba the renderer uses. The mock detector thresholds
#: (detector.py `_COLOR_RULES`) are tuned to exactly these values.
PROP_RGBA = {
    "red": "0.75 0.10 0.10 1",
    "blue": "0.10 0.20 0.80 1",
    "green": "0.10 0.65 0.15 1",
}


def prop_specs(cam_cfg) -> list[dict]:
    """Every prop the camera profile declares, main prop first.

    The main prop is the profile's own `box_px` (+ `detector.label`, colour
    defaults to red, body `red_cube`). `extra_props:` adds more, each
    `{name, box_px, color}` -- a colour word from PROP_RGBA -- sharing the
    profile's `box_height_m`. Used by the scene writer (physics), the mock
    detector (pixels) and the camera tests, so a prop exists in all three or
    in none.
    """
    det = cam_cfg.get("detector") if hasattr(cam_cfg, "get") else None
    main_label = (det.get("label") if det is not None and hasattr(det, "get") else None) or "red cube"
    main_color = next((c for c in PROP_RGBA if c in str(main_label).lower()), "red")
    specs = [{
        "name": str(cam_cfg.get("prop_name") or main_label.replace(" ", "_")),
        "box_px": tuple(int(v) for v in (cam_cfg.get("box_px") or (280, 200, 360, 260))),
        "color": main_color,
        "label": str(main_label),
    }]
    for extra in cam_cfg.get("extra_props") or []:
        color = str(extra.get("color", "blue")).lower()
        if color not in PROP_RGBA:
            raise ValueError(f"extra_props colour {color!r} not in {sorted(PROP_RGBA)}")
        label = str(extra.get("label") or f"{color} cube")
        specs.append({
            "name": str(extra.get("name") or label.replace(" ", "_")),
            "box_px": tuple(int(v) for v in extra["box_px"]),
            "color": color,
            "label": label,
        })
    return specs


def prop_pose_from_camera(cam_cfg, box_px=None) -> tuple[np.ndarray, np.ndarray]:
    """Where the prop must sit in BASE frame for `cam_cfg` to be telling the
    truth, and its half-extents. `box_px` overrides the profile's own box
    (extra props share the camera but not the pixels).

    Backprojects the camera profile's `box_px` corners at the box's own
    surface depth, then transforms into base frame with the profile's
    `extrinsics.T`. Deriving it (rather than writing x = 0.20 in a second
    place) is the point: the physics prop cannot drift away from the pixels
    the detector is keyed to.
    """
    if not hasattr(cam_cfg, "get"):
        # The profile spells the opt-in as `mj_prop_from_camera: true` and
        # load_demo_config swaps the bool for the run's camera profile. A bool
        # (or anything else) landing here means the arm was built from a raw
        # profile that never went through the loader -- there is no camera to
        # derive a prop from, so say that instead of AttributeError'ing.
        raise ValueError(
            "prop_pose_from_camera needs the camera profile mapping "
            f"(got {cam_cfg!r}); `mj_prop_from_camera: true` is only "
            "resolved to a camera by load_demo_config"
        )
    box_px = box_px or cam_cfg.get("box_px") or (280, 200, 360, 260)
    x0, y0, x1, y1 = (int(v) for v in box_px)
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    table_depth = float(cam_cfg.get("table_depth_m", 0.6))
    box_h = float(cam_cfg.get("box_height_m", 0.05))

    # MockCamera.synthetic_tabletop's intrinsics (fx = 600) unless the profile
    # says otherwise; a rendered profile declares `fx` so its K, its MJCF
    # <camera fovy> and this backprojection all come from one number.
    fx = float(cam_cfg.get("fx", 600.0))
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


def resolved_scene_path(robot_mjcf, prop_cam) -> Path:
    """The MJCF a MuJoCo arm will actually load, given its profile.

    `prop_cam` is the profile's `mj_prop_from_camera` value AFTER the config
    loader ran: a camera-profile mapping when scene generation is on, else
    None/False. With generation on, the arm loads the generated scene beside
    `robot_mjcf` (see `write_demo_scene`); otherwise `robot_mjcf` itself.

    Every consumer that must open the SAME world as the arm (rendered
    cameras, the physics-truth channel, the loader that plants `mj_scene`)
    resolves through this function, so there is one derivation of the path.
    """
    robot_mjcf = Path(robot_mjcf)
    if prop_cam is not None and hasattr(prop_cam, "get"):
        return robot_mjcf.parent / "_scene_demo_generated.xml"
    return robot_mjcf


def write_demo_scene(robot_mjcf, cam_cfg, prop_name: str = "red_cube",
                     cameras=None) -> Path:
    """Write a scene beside `robot_mjcf` whose prop matches `cam_cfg`.

    `cameras` is the run's full camera-profile list; every `type: mujoco`
    entry gets a `<camera>` declared at its `extrinsics.T` (see
    `cameras_xml`). When omitted, `cam_cfg` alone is considered, so a run
    whose only camera is the rendered one still gets its `<camera>`.

    Returns the scene path. Written next to the robot MJCF so `meshdir`
    resolves (see this module's docstring).
    """
    robot_mjcf = Path(robot_mjcf)
    cams = list(cameras) if cameras is not None else [cam_cfg]
    scene = resolved_scene_path(robot_mjcf, cam_cfg)
    bodies = []
    for i, spec in enumerate(prop_specs(cam_cfg)):
        centre, half = prop_pose_from_camera(cam_cfg, box_px=spec["box_px"])
        bodies.append(PROP_XML.format(
            # the main prop keeps the caller's name (tests/truth key on it)
            name=prop_name if i == 0 else spec["name"],
            px=float(centre[0]), py=float(centre[1]), pz=float(centre[2]),
            hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
            rgba=PROP_RGBA[spec["color"]],
        ))
    scene.write_text(DEMO_SCENE_TEMPLATE.format(
        robot_xml=robot_mjcf.name,
        props="\n    ".join(bodies),
        cameras=cameras_xml(cams),
    ))
    return scene
