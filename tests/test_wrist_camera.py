"""A wrist camera for the outcome judge (ROADMAP follow-up #15).

Robo-Dopamine's GRM prompt (eval/progress_judge.py) reserves two WRIST image
slots; the SO-101 rig had no wrist view, so both slots repeated the front
image. These tests pin the pieces that give the judge real wrist frames:

  - `configs/cameras/mujoco_wrist.yaml` is a rendered (`type: mujoco`)
    profile flagged `role: wrist` whose `<camera>` is ATTACHED TO THE
    GRIPPER BODY of the robot MJCF (`mj_attach`), so it moves with the arm
    and looks down the fingers at the jaw tips;
  - `sim/demo_scene.write_demo_scene` declares it by copying the robot's
    include chain through ElementTree with one `<camera>` inserted into the
    gripper body -- the fetched asset is never edited and the physics is
    the ORIGINAL model plus one camera, pinned exactly (not approximately);
  - rendered through `perception/mujoco_camera.py` like any other named
    camera, the wrist frame shows the jaws and, during a scripted descent,
    the prop between them (bbox measured on the rendered frame);
  - the runtime records wrist keyframes next to the front ones for motion
    skills (tests/test_keyframes.py) and the judge fills the wrist slots
    from them (tests/test_progress_judge.py).

The wrist view is EVIDENCE for the judge, not a calibrated sensor: the
profile has no `extrinsics` block, fuses no 3-D beliefs, and nothing here
claims a hand-eye calibration (ROADMAP wrist-cam follow-ups stay open).

Physics/render halves need the fetched SO-101 assets (and offscreen GL for
rendering); they skip with the fetch command otherwise.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import REPO, needs_pin_so101
from mujoco_gl_probe import probe_offscreen_gl

from cascade.config import Cfg, load_demo_config

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"


def has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


_offscreen_gl = probe_offscreen_gl(MJCF)

needs_mujoco = pytest.mark.skipif(
    not (has_mujoco() and MJCF.exists()),
    reason="needs mujoco + `python scripts/fetch_robot_assets.py so101`",
)
needs_gl = pytest.mark.skipif(
    not _offscreen_gl.available,
    reason=f"needs mujoco offscreen rendering (GL) + fetched so101 assets: {_offscreen_gl.reason}",
)

WRIST_T = [[1, 0, 0, 0.0], [0, -0.7317, -0.6816, 0.055], [0, 0.6816, -0.7317, -0.045], [0, 0, 0, 1]]
FRONT_T = [[0, -1, 0, 0.28], [-1, 0, 0, 0.0], [0, 0, -1, 0.6], [0, 0, 0, 1]]


@pytest.fixture(autouse=True)
def _fresh_registry():
    from cascade.sim import mujoco_world

    assert not mujoco_world.live_worlds(), "a previous test leaked a world"
    yield
    for w in mujoco_world.live_worlds():
        w.refs = 1
        mujoco_world.release(w)


def _cfg():
    return load_demo_config(cameras=["mujoco_scene", "mujoco_wrist"], arm="so101_mujoco", llm="mock")


# ── profile + wrist-view predicate (always run) ─────────────────────────


def test_wrist_profile_is_a_rendered_wrist_view_without_a_calibration_claim():
    from cascade.perception.camera_base import is_wrist_view

    cfg = _cfg()
    front, wrist = (Cfg(c) for c in cfg.cameras)
    assert wrist.get("name") == "mujoco_wrist" and wrist.type == "mujoco"
    assert wrist.get("role") == "wrist"
    att = wrist.get("mj_attach")
    assert att is not None and att.get("body") == "gripper"
    assert np.asarray(att.get("T"), float).shape == (4, 4)
    # evidence view only: no hand-eye extrinsics, no 3-D fusion from it
    assert wrist.get("extrinsics") is None
    assert wrist.get("fuse_beliefs") is False
    assert wrist.get("static_scene") is False          # the frame changes when the arm moves
    # the loader plants the same world on it as on the front camera
    assert wrist.get("mj_scene") == front.get("mj_scene")
    assert [c.get("name") for c in cfg.arm.get("mj_cameras")] == ["mujoco_scene", "mujoco_wrist"]
    assert is_wrist_view(wrist) and not is_wrist_view(front)


def test_is_wrist_view_is_explicit_role_or_eye_in_hand_extrinsics():
    from cascade.perception.camera_base import is_wrist_view

    assert is_wrist_view(Cfg({"type": "mock", "role": "wrist"}))
    assert is_wrist_view({"type": "isaac", "extrinsics": {"mode": "eye_in_hand", "T": np.eye(4).tolist()}})
    assert not is_wrist_view(Cfg({"type": "mock", "extrinsics": {"mode": "eye_to_hand", "T": FRONT_T}}))
    assert not is_wrist_view(Cfg({"type": "mock"}))
    assert not is_wrist_view(None)
    # the shipped eye-in-hand profiles count as wrist views without any edit
    for name in ("isaac_wrist", "d455f_wrist"):
        from cascade.config import load_profile

        assert is_wrist_view(load_profile("cameras", name)), name


def test_cameras_xml_leaves_attached_cameras_to_the_robot_and_still_refuses_bare_profiles():
    from cascade.sim.demo_scene import attached_cameras, cameras_xml

    cams = [
        Cfg({"type": "mujoco", "name": "front", "extrinsics": {"T": FRONT_T}, "fx": 600, "height": 480}),
        Cfg({"type": "mujoco", "name": "wrist", "mj_attach": {"body": "gripper", "T": WRIST_T},
             "fx": 415.7, "height": 480}),
        Cfg({"type": "mock", "name": "painted", "extrinsics": {"T": FRONT_T}}),
    ]
    xml = cameras_xml(cams)
    assert xml.count("<camera") == 1 and 'name="front"' in xml and "wrist" not in xml
    att = attached_cameras(cams)
    assert [(a["body"], a["xml"].split('"')[1]) for a in att] == [("gripper", "wrist")]
    # same OpenCV -> MuJoCo axis conversion as the fixed cameras: for the
    # wrist T above, camera +x is body +x and camera "up" is body (0, .73, -.68)
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', att[0]["xml"]))
    assert np.allclose([float(v) for v in attrs["pos"].split()], [0.0, 0.055, -0.045])
    assert np.allclose([float(v) for v in attrs["xyaxes"].split()], [1, 0, 0, 0, 0.7317, -0.6816])
    assert float(attrs["fovy"]) == pytest.approx(60.0, abs=0.01)   # from fx=415.7 at 480 px
    with pytest.raises(ValueError, match="extrinsics.T"):
        cameras_xml([Cfg({"type": "mujoco", "name": "bare"})])
    with pytest.raises(ValueError, match="mj_attach"):
        attached_cameras([Cfg({"type": "mujoco", "name": "half", "mj_attach": {"body": "gripper"}})])


# ── generated scene: the camera is a child of the gripper body ──────────


@needs_mujoco
def test_generated_scene_attaches_the_wrist_camera_to_the_gripper_body_without_touching_physics():
    import mujoco

    from cascade.sim.demo_scene import write_demo_scene

    plain = load_demo_config(cameras=["mujoco_scene"], arm="so101_mujoco", llm="mock")
    scene = write_demo_scene(plain.arm.mjcf, plain.arm.get("mj_prop_from_camera"),
                             cameras=plain.arm.get("mj_cameras"))
    # without an attached camera the scene is the plain include wrapper, as before
    assert '<include file="scene.xml"/>' in scene.read_text()
    m0 = mujoco.MjModel.from_xml_path(str(scene))

    cfg = _cfg()
    scene2 = write_demo_scene(cfg.arm.mjcf, cfg.arm.get("mj_prop_from_camera"),
                              cameras=cfg.arm.get("mj_cameras"))
    assert scene2 == scene, "same resolved scene path, whatever cameras look into it"
    text = scene2.read_text()
    assert '<include file="scene.xml"/>' not in text
    inc = [e.get("file") for e in ET.fromstring(text).iter("include")]
    assert len(inc) == 1 and inc[0].startswith(scene.stem) and inc[0].endswith("scene.xml")
    # the include chain was COPIED beside the originals; the fetched asset is untouched
    robot_copy = scene.parent / inc[0]
    assert robot_copy.exists()
    chain = [e.get("file") for e in ET.parse(robot_copy).getroot().iter("include")]
    assert chain and chain[0].startswith(scene.stem) and chain[0].endswith("so101.xml")
    assert "mujoco_wrist" not in (scene.parent / "so101.xml").read_text()
    assert "mujoco_wrist" not in (scene.parent / "scene.xml").read_text()

    m1 = mujoco.MjModel.from_xml_path(str(scene2))
    cid = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_CAMERA, "mujoco_wrist")
    assert cid >= 0
    assert m1.body(m1.cam_bodyid[cid]).name == "gripper"
    assert m1.ncam == m0.ncam + 1
    # physics is the ORIGINAL model plus one camera: exact equality, not approx
    assert (m0.nbody, m0.ngeom, m0.njnt, m0.nu, m0.nq, m0.nmesh) == (m1.nbody, m1.ngeom, m1.njnt, m1.nu, m1.nq, m1.nmesh)
    for arr in ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_pos", "geom_quat", "geom_size",
                "geom_friction", "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp",
                "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss", "actuator_gainprm",
                "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "qpos0", "geom_rgba",
                "mesh_vert", "mesh_face", "site_pos", "site_quat"):
        assert np.array_equal(getattr(m0, arr), getattr(m1, arr)), arr
    assert m0.opt.timestep == m1.opt.timestep
    for i in range(m0.ngeom):
        assert m0.geom(i).name == m1.geom(i).name
    # the camera is rigid to the gripper: its world pose follows the body
    d = mujoco.MjData(m1)
    mujoco.mj_forward(m1, d)
    bid = m1.cam_bodyid[cid]
    T_body_cam0 = np.linalg.inv(_T(d.xpos[bid], d.xmat[bid])) @ _T(d.cam_xpos[cid], d.cam_xmat[cid])
    for joint, q in zip(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
                        (0.6, -0.4, 0.5, 0.3, 0.7)):
        d.qpos[m1.jnt_qposadr[m1.joint(joint).id]] = q
    mujoco.mj_forward(m1, d)
    T_body_cam1 = np.linalg.inv(_T(d.xpos[bid], d.xmat[bid])) @ _T(d.cam_xpos[cid], d.cam_xmat[cid])
    assert np.allclose(T_body_cam0, T_body_cam1, atol=1e-9), "camera is not rigid to the gripper body"
    # ...and that pose is the profile's T (OpenCV axes) converted to MuJoCo's
    T_prof = np.asarray(Cfg(cfg.cameras[1]).get("mj_attach").get("T"), float)
    assert np.allclose(T_body_cam0[:3, 3], T_prof[:3, 3], atol=1e-6)
    assert np.allclose(T_body_cam0[:3, :3] @ np.diag([1, -1, -1]), T_prof[:3, :3], atol=2e-3)


def _T(pos, mat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


@needs_mujoco
def test_attach_refuses_a_body_the_robot_does_not_have():
    from cascade.sim.demo_scene import write_demo_scene

    cfg = _cfg()
    cams = list(cfg.arm.get("mj_cameras"))
    bad = Cfg({**dict(cams[1]), "mj_attach": {"body": "no_such_link", "T": WRIST_T}})
    with pytest.raises(ValueError, match="no_such_link"):
        write_demo_scene(cfg.arm.mjcf, cfg.arm.get("mj_prop_from_camera"), cameras=[cams[0], bad])


# ── rendered: the wrist camera sees the jaws and the prop between them ──


def _gripper_pixels(world, cam_id) -> int:
    """Pixels of the gripper subtree (both jaws + mount) in a segmentation
    render from `cam_id` -- colour-independent evidence that the jaws are in
    view."""
    import mujoco

    m = world.model
    gripper = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in ("gripper", "moving_jaw_so101_v1", "camera_mount")]
    r = mujoco.Renderer(m, height=480, width=640)
    try:
        r.enable_segmentation_rendering()
        with world.lock:
            r.update_scene(world.data, camera=cam_id)
        seg = r.render()
    finally:
        r.close()
    geom = seg[:, :, 0]
    bodies = np.where(geom >= 0, m.geom_bodyid[np.clip(geom, 0, None)], -1)
    return int(np.isin(bodies, gripper).sum())


def _descend(arm, kin, cfg, z: float):
    """Scripted top-down approach of the TCP over the prop at base (0.20, 0.10)."""
    from cascade.grasping.obb_grasp import _yaw_rotation
    from cascade.types import make_transform

    target = make_transform(_yaw_rotation(np.arctan2(0.10, 0.20) + np.pi / 2, axis_order="open_down"),
                            np.array([0.20, 0.10, z]))
    res = kin.ik(target, np.asarray(cfg.arm.get("home_q"), float))
    assert res.success, f"IK failed for z={z}: err={res.error}"
    arm.set_gripper(float(cfg.arm.gripper.get("open_pos")))
    for _ in range(60):
        arm.send_joint_target(res.q)
    assert arm.wait_settled(res.q, tol=float(cfg.arm.get("settle_tol", 0.03)), timeout_s=4.0)


@needs_gl
@needs_pin_so101
def test_wrist_camera_sees_the_jaws_and_the_prop_between_them_during_a_descent():
    from cascade.control.arm_base import make_arm
    from cascade.control.kinematics import Kinematics
    from cascade.perception.camera_base import make_camera
    from cascade.perception.detector import MockDetector

    cfg = _cfg()
    arm = make_arm(cfg.arm)
    arm.connect()
    front = make_camera(Cfg(cfg.cameras[0]))
    wrist = make_camera(Cfg(cfg.cameras[1]))
    front.open()
    wrist.open()
    try:
        assert wrist.world is arm.world is front.world, "one world for the arm and both cameras"
        kin = Kinematics(cfg.arm.model, cfg.arm.ee_frame, n_controlled=int(cfg.arm.n_joints))
        det = MockDetector(label="red cube")

        f_home = wrist.get_frame()
        assert f_home.rgb.shape == (480, 640, 3)
        assert f_home.depth_m is None and f_home.depth_source == "none", "RGB-only evidence view"
        jaws_home = _gripper_pixels(wrist.world, wrist._cam_id)
        assert jaws_home > 5000, f"the wrist camera must look at its own jaws ({jaws_home} px)"
        front_home = front.get_frame()
        assert det.detect(front_home), "the front camera sees the prop at home"

        areas = {}
        for z in (0.08, 0.05):
            _descend(arm, kin, cfg, z)
            frame = wrist.get_frame()
            dets = det.detect(frame)
            assert dets, f"wrist frame at z={z} shows no prop"
            x0, y0, x1, y1 = dets[0].bbox
            areas[z] = (x1 - x0) * (y1 - y0)
            assert _gripper_pixels(wrist.world, wrist._cam_id) > 5000, "jaws left the wrist frame"
            # the prop sits BETWEEN the jaws: the bbox straddles the frame's
            # vertical centre line rather than hugging an edge
            assert x0 < 320 < x1, (x0, x1)
        # closer = bigger: the view is moving with the arm, not a fixed camera
        assert areas[0.05] > areas[0.08] > 2000, areas
        # and the front view did not become the wrist view by accident
        assert not np.array_equal(wrist.get_frame().rgb, front.get_frame().rgb)
    finally:
        wrist.close()
        front.close()
        arm.disconnect()


@needs_gl
def test_runtime_records_a_real_wrist_keyframe_for_a_motion_skill_on_the_rendered_rig(tmp_path):
    """End to end on the MuJoCo rig: config -> generated scene -> rendered
    wrist stream -> runtime trace row with a wrist keyframe that is not the
    front keyframe."""
    import json

    import cv2

    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = _cfg()
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        assert runtime.rig.names == ["mujoco_scene", "mujoco_wrist"]
        runtime.execute("move_home", {})
        row = json.loads((runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()[0])
        kw = row["keyframe_before_wrists"]
        ka = row["keyframe_after_wrists"]
        assert list(kw) == ["mujoco_wrist"] and list(ka) == ["mujoco_wrist"]
        for p in (kw["mujoco_wrist"], ka["mujoco_wrist"]):
            img = cv2.imread(str(runtime.trace.run_dir / p))
            assert img is not None and img.shape == (480, 640, 3)
        front_after = cv2.imread(str(runtime.trace.run_dir / row["keyframe_after"]))
        wrist_after = cv2.imread(str(runtime.trace.run_dir / ka["mujoco_wrist"]))
        assert not np.array_equal(front_after, wrist_after), "wrist keyframe is a copy of the front one"
    finally:
        shutdown_runtime(runtime, arm)
