"""One MuJoCo world for the arm, the rendered camera and the truth channel.

The MuJoCo demo used to run three private copies of the same MJCF: the arm
stepped one, a synthetic camera painted a cube that existed in none, and
there was no physics-truth channel at all -- so every MuJoCo pick reported
`postcondition: unverified`. These tests pin the wiring that fixes it:

  - `sim/mujoco_world.py` hands every holder of one MJCF path the same
    MjModel/MjData, in EITHER construction order (the MCP server materializes
    the arm lazily, so the camera can be first);
  - `perception/mujoco_camera.py` renders the prop where the profile says it
    is (pixels, depth AND the perceived 3-D fix agree with `data.xpos`);
  - `sim/truth.py::MujocoTruthReader` reads free-body poses out of that world
    and `LazyTruthPoseFn` binds to it AFTER the arm connects;
  - the config loader plants `mj_scene` on rendered cameras and refuses a
    rendered camera on a non-MuJoCo primary arm.

Physics/render halves need the fetched SO-101 assets and a working offscreen
GL; they skip with the fetch command otherwise. The loader and label-matching
halves always run.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO

from cascade.config import Cfg, load_demo_config
from cascade.sim.truth import (
    LazyTruthPoseFn, MujocoTruthReader, _match_label, make_truth_pose_fn,
)

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"


def has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


def has_offscreen_gl() -> bool:
    """A renderer can be created (macOS CGL / EGL / OSMesa present)."""
    if not (has_mujoco() and MJCF.exists()):
        return False
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(MJCF))
        r = mujoco.Renderer(m, height=16, width=16)
        r.close()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_mujoco = pytest.mark.skipif(
    not (has_mujoco() and MJCF.exists()),
    reason="needs mujoco + `python scripts/fetch_robot_assets.py so101`",
)
needs_gl = pytest.mark.skipif(
    not has_offscreen_gl(),
    reason="needs mujoco offscreen rendering (GL) + fetched so101 assets",
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    """No world may leak between tests: a leaked holder would make the
    'both orders' tests pass by accident."""
    from cascade.sim import mujoco_world

    assert not mujoco_world.live_worlds(), "a previous test leaked a world"
    yield
    for w in mujoco_world.live_worlds():
        w.refs = 1
        mujoco_world.release(w)


def _rig(order: str):
    """A connected arm + opened camera on the mujoco_scene profile."""
    from cascade.control.arm_base import make_arm
    from cascade.perception.camera_base import make_camera

    cfg = load_demo_config(camera="mujoco_scene", arm="so101_mujoco", llm="mock")
    arm = make_arm(cfg.arm)
    cam = make_camera(cfg.camera)
    if order == "arm_first":
        arm.connect()
        cam.open()
    else:
        # MCP mode: the arm is lazy, so when the camera opens NOTHING has
        # written the generated scene yet (make_arm above did, so undo it).
        # The camera must regenerate it from the loader-planted inputs.
        Path(cfg.camera.get("mj_scene")).unlink(missing_ok=True)
        cam.open()
        arm.connect()
    return cfg, arm, cam


# ── loader ───────────────────────────────────────────────────────────────


def test_loader_plants_the_scene_path_on_rendered_cameras():
    cfg = load_demo_config(camera="mujoco_scene", arm="so101_mujoco", llm="mock")
    assert cfg.camera.type == "mujoco"
    scene = cfg.camera.get("mj_scene")
    assert scene and scene.endswith("_scene_demo_generated.xml")
    assert cfg.camera.get("mj_arm") == "so101_mujoco"
    # ...and everything needed to REGENERATE the scene itself (MCP mode:
    # the lazy arm has not written it when the camera opens).
    src = cfg.camera.get("mj_scene_source")
    assert src is not None and str(src.get("mjcf")).endswith("scene.xml")
    assert [c.get("type") for c in src.get("cameras")] == ["mujoco"]
    # ...and the arm gets the whole camera list, so its generated scene can
    # declare a <camera> for every rendered profile.
    assert [c.get("type") for c in cfg.arm.get("mj_cameras")] == ["mujoco"]
    # both derived from the SAME resolver, so they cannot drift
    from cascade.sim.demo_scene import resolved_scene_path

    assert str(resolved_scene_path(cfg.arm.mjcf, cfg.arm.get("mj_prop_from_camera"))) == scene


def test_loader_refuses_a_rendered_camera_on_a_non_mujoco_arm():
    with pytest.raises(ValueError, match="type: mujoco"):
        load_demo_config(camera="mujoco_scene", arm="so101_mock", llm="mock")


def test_rendered_scene_declares_one_camera_per_rendered_profile(tmp_path):
    from cascade.sim.demo_scene import cameras_xml

    T = [[0, -1, 0, 0.28], [-1, 0, 0, 0.0], [0, 0, -1, 0.6], [0, 0, 0, 1]]
    xml = cameras_xml([
        Cfg({"type": "mujoco", "name": "a", "extrinsics": {"T": T}, "fx": 600, "height": 480}),
        Cfg({"type": "mock", "name": "b", "extrinsics": {"T": T}}),
        Cfg({"type": "mujoco", "name": "c", "mj_camera": "side", "extrinsics": {"T": T}}),
    ])
    assert xml.count("<camera") == 2
    assert 'name="a"' in xml and 'name="side"' in xml and 'name="b"' not in xml
    # OpenCV (right, down, forward) -> MuJoCo (right, up, backward): for a
    # camera looking straight down, MuJoCo's "up" is base +x here.
    assert 'xyaxes="0.000000000 -1.000000000 0.000000000 1.000000000 0.000000000 0.000000000"' in xml
    with pytest.raises(ValueError, match="extrinsics.T"):
        cameras_xml([Cfg({"type": "mujoco", "name": "bare"})])


# ── label matching (shared by Isaac and MuJoCo readers) ─────────────────


def test_truth_label_matching_rules():
    poses = {"red_cube": [1, 0, 0], "green_cube": [2, 0, 0]}
    assert _match_label("red cube", poses) == [1, 0, 0]          # exact
    assert _match_label("the red object", poses) == [1, 0, 0]    # unique colour
    assert _match_label("cubo rojo", poses) == [1, 0, 0]         # ES -> EN
    assert _match_label("the cube", poses) is None               # ambiguous
    assert _match_label("pink cube", poses) is None              # 1 shared token
    two_red = {"red_cube": [1, 0, 0], "red_bowl": [3, 0, 0]}
    assert _match_label("the red object", two_red) is None       # colour not unique
    assert _match_label("red bowl", two_red) == [3, 0, 0]


# ── shared world ─────────────────────────────────────────────────────────


@needs_mujoco
@pytest.mark.parametrize("order", ["arm_first", "cam_first"])
def test_arm_and_camera_share_one_world_in_either_order(order):
    from cascade.sim import mujoco_world

    cfg, arm, cam = _rig(order)
    try:
        worlds = mujoco_world.live_worlds()
        assert len(worlds) == 1 and worlds[0].refs == 2
        assert arm.world is cam.world is worlds[0]
        # one lock, shared: the arm serializes on the world's lock
        assert arm._lock is worlds[0].lock
    finally:
        cam.close()
        arm.disconnect()
    assert not mujoco_world.live_worlds(), "last holder must drop the world"


@needs_mujoco
def test_truth_reader_sees_the_arm_move_the_prop():
    """The camera-independent channel: physics truth follows the prop."""
    cfg, arm, cam = _rig("arm_first")
    try:
        reader = make_truth_pose_fn(arm)
        assert isinstance(reader, MujocoTruthReader) and reader.live
        before = reader("red object")
        assert before is not None and abs(before[0] - 0.20) < 0.01
        # teleport the free body as a stand-in for a grasp+place
        w = arm.world
        bid = w.mj.mj_name2id(w.model, w.mj.mjtObj.mjOBJ_BODY, "red_cube")
        jadr = int(w.model.jnt_qposadr[w.model.body_jntadr[bid]])
        with w.lock:
            w.data.qpos[jadr:jadr + 3] = [0.20, -0.12, 0.025]
            w.mj.mj_forward(w.model, w.data)
        after = reader("red object")
        assert abs(after[1] - (-0.12)) < 1e-6
    finally:
        cam.close()
        arm.disconnect()
    # world gone -> the reader degrades to None, never to a stale pose
    assert reader("red object") is None and not reader.live


@needs_mujoco
def test_lazy_truth_fn_binds_after_the_arm_materializes():
    """MCP mode: verifier attached at startup, arm connects on first motion.
    With NO rendered camera holding the world, nothing can be read until the
    arm itself comes up -- and probing must not be what brings it up."""
    from cascade.control.arm_base import make_arm
    from cascade.control.lazy_arm import LazyArm

    cfg = load_demo_config(camera="mujoco_scene", arm="so101_mujoco", llm="mock")
    lazy = LazyArm(lambda: make_arm(cfg.arm), n_joints=5, profile_type="mujoco")
    fn = LazyTruthPoseFn(lazy)
    assert fn("red object") is None and not lazy.connected, "probe must not materialize"
    lazy.get_state()  # first use: the arm comes up
    assert lazy.connected
    try:
        assert fn.bound
        pose = fn("red object")
        assert pose is not None and abs(pose[0] - 0.20) < 0.01
    finally:
        lazy.disconnect()
    assert fn("red object") is None


@needs_mujoco
def test_lazy_truth_fn_binds_through_a_rendered_camera_before_the_arm_exists():
    """The chat-path bug: under the MCP server the camera opens at prewarm,
    the arm is lazy. The physics channel must be readable from the camera's
    world BEFORE the first motion, so the pre-motion snapshot is physics and
    not a belief restored from a previous episode's drop point."""
    from cascade.control.arm_base import make_arm
    from cascade.control.lazy_arm import LazyArm
    from cascade.perception.camera_base import make_camera

    cfg = load_demo_config(camera="mujoco_scene", arm="so101_mujoco", llm="mock")
    Path(cfg.camera.get("mj_scene")).unlink(missing_ok=True)
    built = []

    def factory():
        built.append(1)
        return make_arm(cfg.arm)

    lazy = LazyArm(factory, n_joints=5, profile_type="mujoco")
    fn = LazyTruthPoseFn(lazy)
    cam = make_camera(cfg.camera)
    cam.open()
    try:
        pose = fn("red object")
        assert not built and not lazy.connected, "reading truth must not power the arm"
        assert pose is not None and abs(pose[0] - 0.20) < 0.01 and abs(pose[1] - 0.10) < 0.01
        # a NON-sim lazy arm next to the same live world must still get nothing
        other = LazyTruthPoseFn(LazyArm(factory, n_joints=6, profile_type="rebot_rs"))
        assert other("red object") is None and not built
    finally:
        cam.close()
    assert fn("red object") is None, "world gone -> channel unbound again"


def test_displacement_never_mixes_channels():
    """physics_after - belief_before is not a displacement. Reproduces the
    chat-path false REFUTED: belief restored at the drop point, arm lazy,
    physics binds only after the motion."""
    from cascade.agent.effects import CONFIRMED, REFUTED, PostconditionChecker

    physics = {}
    belief = {"red cube": [0.170, -0.095, 0.035]}          # stale, from disk
    checker = PostconditionChecker(
        object_pose=lambda l: physics.get(l), belief_pose=lambda l: belief.get(l)
    )
    before = checker.snapshot("red cube")
    assert before["channel"] == "belief"
    physics["red cube"] = [0.1696, -0.0958, 0.0174]        # after: real drop
    pc = checker.verify("pick_and_place", {"object": "red cube"},
                        {"ok": True, "placed_at": [0.17, -0.10, 0.02]}, before=before)
    assert pc.channel == "physics"
    assert pc.status != REFUTED, pc.evidence
    assert "moved_m" not in pc.measured
    assert pc.measured.get("start_channel_mismatch") == "belief->physics"

    # same channel on both sides: the displacement check still bites
    physics["red cube"] = [0.20, 0.10, 0.0175]
    before = checker.snapshot("red cube")
    assert before["channel"] == "physics"
    physics["red cube"] = [0.21, 0.10, 0.0175]             # nudged 1 cm
    pc = checker.verify("pick_and_place", {"object": "red cube"},
                        {"ok": True, "placed_at": [0.21, 0.10, 0.02]}, before=before)
    assert pc.status == REFUTED and "where it started" in pc.evidence
    physics["red cube"] = [0.17, -0.10, 0.0175]
    pc = checker.verify("pick_and_place", {"object": "red cube"},
                        {"ok": True, "placed_at": [0.17, -0.10, 0.02]}, before=before)
    assert pc.status == CONFIRMED and pc.measured["moved_m"] > 0.15


def test_lazy_truth_fn_never_materializes_a_real_arm():
    """Same rule as make_truth_pose_fn: reading the channel on a standby arm
    must not build (power) it -- the factory here would record the call."""
    from cascade.control.lazy_arm import LazyArm

    built = []

    def factory():
        built.append(1)
        raise AssertionError("factory called")

    lazy = LazyArm(factory, n_joints=6, profile_type="rebot_rs")
    fn = LazyTruthPoseFn(lazy)
    for _ in range(3):
        assert fn("pink cube") is None
        assert fn.all_poses() == {}
    assert not built and not lazy.connected


def test_demo_wiring_gives_sim_arms_a_lazy_channel_and_others_none():
    from cascade.apps.demo import _truth_pose_fn
    from cascade.control.lazy_arm import LazyArm
    from cascade.control.mock_arm import MockArm

    class Safe:
        def __init__(self, raw):
            self.raw = raw

    assert _truth_pose_fn(Safe(MockArm(Cfg({"type": "mock"})))) is None
    real = LazyArm(lambda: None, n_joints=6, profile_type="rebot_rs")
    assert _truth_pose_fn(Safe(real)) is None
    sim = LazyArm(lambda: None, n_joints=5, profile_type="mujoco")
    fn = _truth_pose_fn(Safe(sim))
    assert isinstance(fn, LazyTruthPoseFn) and fn("x") is None and not sim.connected


# ── rendered camera ──────────────────────────────────────────────────────


@needs_gl
@pytest.mark.parametrize("order", ["arm_first", "cam_first"])
def test_rendered_camera_sees_the_physics_prop_where_the_profile_says(order):
    from cascade.perception.detector import MockDetector
    from cascade.perception.grounding import Extrinsics, localize_object

    cfg, arm, cam = _rig(order)
    try:
        frame = cam.get_frame()
        assert frame.rgb.shape == (480, 640, 3) and frame.depth_m.shape == (480, 640)
        assert frame.depth_source == "sensor"
        det = MockDetector(label="red cube")
        dets = det.detect(frame)
        assert dets, "the red prop must be visible from the declared camera"
        x0, y0, x1, y1 = dets[0].bbox
        ex = cfg.camera.get("box_px")
        assert max(abs(x0 - ex[0]), abs(y0 - ex[1]), abs(x1 - ex[2]), abs(y1 - ex[3])) <= 3
        # metric depth: prop top face at table_depth - box_height, table at table_depth
        cy, cx = int((y0 + y1) / 2), int((x0 + x1) / 2)
        assert abs(frame.depth_m[cy, cx] - 0.55) < 0.005
        assert abs(frame.depth_m[60, 60] - 0.60) < 0.005
        # the perception stack's 3-D fix lands on the physics prop
        fix = localize_object(frame, "red object", det,
                              Extrinsics.from_config(cfg.camera.get("extrinsics")),
                              prompts=["red cube"])
        truth = arm.world.body_pos("red_cube")
        lateral = float(np.linalg.norm(np.asarray(fix.position)[:2] - np.asarray(truth)[:2]))
        assert lateral < 0.006, f"lateral error {lateral*1000:.1f} mm"
        assert abs(float(fix.points[:, 2].max()) - 0.050) < 0.004
    finally:
        cam.close()
        arm.disconnect()


@needs_gl
def test_rendered_camera_renders_from_a_second_thread_and_follows_the_arm():
    """CameraStream grabs from its own pump thread: the GL renderer must be
    created there, and a frame taken after the arm moves must differ."""
    cfg, arm, cam = _rig("arm_first")
    try:
        out = {}

        def grab():
            out["f0"] = cam.get_frame()

        t = threading.Thread(target=grab)
        t.start()
        t.join()
        assert cam._render_owner is t
        f0 = out["f0"]
        q = arm.get_state().q.copy()
        q[0] += 0.6  # swing the base
        for _ in range(40):
            arm.send_joint_target(q)

        def grab2():
            out["f1"] = cam.get_frame()

        t2 = threading.Thread(target=grab2)
        t2.start()
        t2.join()
        diff = np.abs(out["f1"].rgb.astype(int) - f0.rgb.astype(int)).sum(axis=2) > 30
        assert diff.mean() > 0.002, "camera did not see the arm move"
    finally:
        cam.close()
        arm.disconnect()


@needs_mujoco
def test_rendered_camera_names_the_missing_camera_and_lists_what_exists():
    from cascade.perception.camera_base import CameraError
    from cascade.perception.mujoco_camera import MujocoCamera

    cam = MujocoCamera(Cfg({"type": "mujoco", "mjcf": str(MJCF), "mj_camera": "nope"}))
    with pytest.raises(CameraError, match="no <camera name='nope'>"):
        cam.open()
    from cascade.sim import mujoco_world

    assert not mujoco_world.live_worlds(), "a failed open must not hold the world"
