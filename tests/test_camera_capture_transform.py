"""Camera poses travel with observations, before delayed perception runs."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import needs_pin

from cascade.config import Cfg
from cascade.perception.detector import MockDetector
from cascade.perception.grounding import Extrinsics, localize_object
from cascade.perception.mock_camera import synthetic_tabletop
from cascade.perception.stream import CameraStream
from cascade.types import Frame


class _Frames:
    """An RGB-D source with no sensor or robot dependencies."""

    has_depth = True

    def __init__(self, T=None):
        self.frame = Frame(
            rgb=np.zeros((12, 16, 3), dtype=np.uint8),
            depth_m=np.ones((12, 16), dtype=np.float32),
            K=np.array([[20, 0, 8], [0, 20, 6], [0, 0, 1]], dtype=float),
            T_base_cam=T,
        )
        self.seq = 0

    def open(self):
        pass

    def close(self):
        pass

    def get_frame(self):
        self.seq += 1
        return replace(self.frame, frame_id=self.seq)


def test_stream_freezes_pose_before_later_arm_state_changes():
    live_T = np.eye(4)
    live_T[0, 3] = 0.25
    stream = CameraStream(_Frames(), rate_hz=60, capture_transform=lambda: live_T)
    stream.open()
    try:
        observed = stream.get_frame()
        live_T[0, 3] = 0.75
        later = stream.get_frame()
        assert observed.T_base_cam[0, 3] == 0.25
        assert later.T_base_cam[0, 3] == 0.75
        assert not np.shares_memory(observed.T_base_cam, live_T)
    finally:
        stream.close()


def test_camera_provider_pose_takes_precedence_over_receipt_fk():
    provider_T = np.eye(4)
    provider_T[1, 3] = 0.4
    calls = []

    def receipt_pose():
        calls.append(True)
        return np.eye(4)

    stream = CameraStream(_Frames(provider_T), rate_hz=60, capture_transform=receipt_pose)
    stream.open()
    try:
        observed = stream.get_frame()
        np.testing.assert_array_equal(observed.T_base_cam, provider_T)
        assert not calls
    finally:
        stream.close()


def test_camera_and_body_pose_share_an_immutable_observation_snapshot():
    q = np.array([.25, .1])

    def geometry():
        T = np.eye(4)
        T[0, 3] = q[0]
        return T, {"fr3": q}

    stream = CameraStream(_Frames(), capture_geometry=geometry)
    frame = _Frames().get_frame()
    stream._stamp_transform(frame)
    q[0] = .75
    assert frame.T_base_cam[0, 3] == .25
    assert frame.robot_joint_positions["fr3"][0] == .25
    assert not np.shares_memory(frame.robot_joint_positions["fr3"], q)


@needs_pin
def test_ros_body_mask_uses_frame_joints_and_refuses_missing_snapshot(monkeypatch):
    from types import SimpleNamespace
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import OccupancyMap, OccupancyError
    from cascade.perception.robot_mask import arm_link_points

    cfg = load_demo_config(arm="fr3", camera="d435i", llm="mock")
    current = np.array(cfg.arm.home_q)
    raw = SimpleNamespace(connect=lambda: None, connected=True,
                          get_state=lambda: SimpleNamespace(q=current))
    monkeypatch.setattr(demo, "make_arm", lambda *a, **kw: raw)
    occ = OccupancyMap()
    _, _, kin = demo._build_arm(cfg.arm, False, occ, cfg)
    frame = _Frames().get_frame()
    captured = current.copy()
    frame.robot_joint_positions = {str(cfg.arm.get("name", "arm")): captured}
    current[0] += .5
    points, _ = occ._robot_bodies(frame)[0]
    np.testing.assert_allclose(points, arm_link_points(kin, captured))
    assert not np.allclose(points, arm_link_points(kin, current))
    frame.robot_joint_positions = None
    with pytest.raises(OccupancyError, match="snapshot missing"):
        occ._robot_bodies(frame)


@pytest.mark.parametrize("pose", [
    None,
    np.eye(3),
    np.full((4, 4), np.nan),
    np.diag([1, 1, 1, 0]),
    np.diag([2, 1, 1, 1]),
    np.diag([-1, 1, 1, 1]),
])
def test_invalid_pose_does_not_stop_rgb_stream_or_fabricate_extrinsics(pose):
    stream = CameraStream(_Frames(), rate_hz=60, capture_transform=lambda: pose)
    stream.open()
    try:
        observed = stream.get_frame()
        assert observed.rgb.shape == (12, 16, 3)
        assert observed.T_base_cam is None
        assert stream.last_error is None
        assert stream.get_frame().frame_id > observed.frame_id
    finally:
        stream.close()


def test_unavailable_fk_leaves_rgb_live_then_recovers():
    ready = False

    def receipt_pose():
        if not ready:
            raise RuntimeError("arm is still in standby")
        return np.eye(4)

    stream = CameraStream(_Frames(), rate_hz=60, capture_transform=receipt_pose)
    stream.open()
    try:
        assert stream.get_frame().T_base_cam is None
        assert stream.last_error is None
        ready = True
        np.testing.assert_array_equal(stream.get_frame().T_base_cam, np.eye(4))
    finally:
        stream.close()


def test_grounding_uses_observation_pose_after_live_fk_changes():
    captured_T = np.array([
        [0, -1, 0, 0.28],
        [-1, 0, 0, 0],
        [0, 0, -1, 0.60],
        [0, 0, 0, 1],
    ], dtype=float)
    frame = synthetic_tabletop()
    frame.T_base_cam = captured_T.copy()
    live_T = captured_T.copy()
    live_T[0, 3] += 0.3
    calls = []

    def fk():
        calls.append(True)
        return live_T

    extrinsics = Extrinsics(mode="eye_in_hand", T=np.eye(4), fk_tcp2base=fk)
    detector = MockDetector()
    expected = localize_object(
        synthetic_tabletop(), "red cube", detector, Extrinsics(T=captured_T)
    )
    captured_fix = localize_object(frame, "red cube", detector, extrinsics)
    np.testing.assert_allclose(captured_fix.position, expected.position)
    np.testing.assert_allclose(captured_fix.points, expected.points)
    assert not calls

    # Backends with no frame pose retain the profile's existing FK fallback.
    frame.T_base_cam = None
    fallback_fix = localize_object(frame, "red cube", detector, extrinsics)
    np.testing.assert_allclose(fallback_fix.position, expected.position + [0.3, 0, 0])
    assert len(calls) == 1


@needs_pin
def test_runtime_stamps_each_camera_with_its_own_extrinsics(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    data = demo_cfg.as_dict()
    first = data["camera"]
    first.update(name="over", fps=60, warmup_frames=1)
    second = dict(first, name="side")
    second_T = np.eye(4)
    second_T[:3, 3] = [0.4, -0.2, 0.7]
    second["extrinsics"] = {"mode": "eye_to_hand", "T": second_T.tolist()}
    data["cameras"] = [first, second]
    data["perception_loop"] = {"enabled": False}
    data["verify_effects"] = False
    runtime, arm = build_runtime(Cfg(data), tmp_path, serve=False)
    try:
        np.testing.assert_allclose(
            runtime.rig.get("over").get_frame().T_base_cam,
            np.asarray(first["extrinsics"]["T"]),
        )
        np.testing.assert_allclose(runtime.rig.get("side").get_frame().T_base_cam, second_T)
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_eye_in_hand_capture_does_not_materialize_lazy_arm(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    data = demo_cfg.as_dict()
    camera = data["camera"]
    camera.update(fps=60, warmup_frames=1)
    camera["extrinsics"] = {"mode": "eye_in_hand", "T": np.eye(4).tolist()}
    data["cameras"] = [camera]
    data["perception_loop"] = {"enabled": False}
    data["verify_effects"] = False
    runtime, arm = build_runtime(Cfg(data), tmp_path, lazy_arm=True, serve=False)
    try:
        observed = runtime.camera.get_frame()
        assert observed.T_base_cam is None
        assert not arm.connected
        assert arm.__dict__["_arm"] is None
        assert runtime.camera.last_error is None
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
@pytest.mark.parametrize("backend,mode,attach,fail_state", [
    ("ros2", "eye_in_hand", True, False),
    ("ros2", "eye_to_hand", False, False),
    ("rebot_rs", "eye_in_hand", False, False),
    ("feetech", "eye_in_hand", False, False),
    ("ros2", "eye_in_hand", True, True),
])
def test_only_ros_wrist_feedback_attaches_before_camera_without_motion(
        demo_cfg, tmp_path, monkeypatch, backend, mode, attach, fail_state):
    from cascade.apps import demo
    from cascade.control.lazy_arm import LazyArm
    from cascade.control.mock_arm import MockArm

    data = demo_cfg.as_dict()
    data["arm"]["type"] = backend
    data["arms"] = [data["arm"]]
    camera = data["camera"]
    camera.update(fps=60, warmup_frames=1)
    camera["extrinsics"] = {"mode": mode, "T": np.eye(4).tolist()}
    data["cameras"] = [camera]
    data["perception_loop"] = {"enabled": False}
    data["verify_effects"] = False
    data["occupancy"] = {"enabled": False}
    created = []
    events = []
    class PassiveRos(MockArm):
        shutdown_park_required = False
        def connect(self):
            events.append("feedback_connect")
            super().connect()
        def get_state(self):
            events.append("feedback_read")
            if fail_state:
                raise RuntimeError("feedback unavailable after connection")
            return super().get_state()
        def send_joint_target(self, *args, **kwargs):
            pytest.fail("read-only camera startup commanded an arm target")
        def set_gripper(self, *args, **kwargs):
            pytest.fail("read-only camera startup commanded gripper jaws")
        def resume(self):
            pytest.fail("read-only camera startup cleared STOP")
    def arm_factory(cfg, kinematics):
        assert attach, "non-ROS or static-camera startup materialized a LazyArm"
        raw = PassiveRos(cfg, kinematics)
        created.append(raw)
        return raw
    make_camera = demo.make_camera
    def camera_factory(cfg):
        camera = make_camera(cfg)
        original_open = camera.open
        def open_camera():
            events.append("camera_open")
            assert bool(created) is attach
            if attach:
                assert events[:2] == ["feedback_connect", "feedback_read"]
            original_open()
        camera.open = open_camera
        return camera
    monkeypatch.setattr(demo, "make_arm", arm_factory)
    monkeypatch.setattr(demo, "make_camera", camera_factory)
    if fail_state:
        with pytest.raises(RuntimeError, match="feedback unavailable after connection"):
            demo.build_runtime(Cfg(data), tmp_path, lazy_arm=True, serve=False)
        assert len(created) == 1 and not created[0]._connected
        assert created[0].commands == [] and "camera_open" not in events
        return
    runtime, raw = demo.build_runtime(Cfg(data), tmp_path, lazy_arm=True, serve=False)
    try:
        assert isinstance(raw, LazyArm) and raw.connected is attach
        runtime.arm.stop()  # Startup stop must remain latched during observations.
        frame = runtime.camera.get_frame()
        assert runtime.arm.harness.estopped
        if attach:
            key = str(data["arm"].get("name", "arm"))
            q = frame.robot_joint_positions[key]
            np.testing.assert_allclose(frame.T_base_cam, runtime.kin.fk(q))
            assert created[0].commands == [] and created[0]._stopped
            assert raw.get_state().q.shape == q.shape
            assert runtime.arm.harness.estopped
        else:
            assert not created and raw._arm is None
    finally:
        demo.shutdown_runtime(runtime, raw)


@needs_pin
def test_ros_capture_keeps_base_frame_compensation_and_same_joint_snapshot(demo_cfg, tmp_path, monkeypatch):
    from cascade.apps import demo
    from cascade.control.mock_arm import MockArm

    data = demo_cfg.as_dict()
    data["arm"]["type"] = "ros2"
    data["arms"] = [data["arm"]]
    camera = data["camera"]
    camera.update(fps=60, warmup_frames=1)
    camera["extrinsics"] = {
        "mode": "eye_in_hand", "T": np.eye(4).tolist(),
        "hand_eye_compensation_m": {"x": .02, "y": -.03, "z": .01},
    }
    data["cameras"] = [camera]
    data["perception_loop"] = {"enabled": False}
    data["verify_effects"] = False
    data["occupancy"] = {"enabled": False}
    # Exercise ROS capture binding without importing ROS or opening a robot.
    monkeypatch.setattr(demo, "make_arm", lambda cfg, kinematics: MockArm(cfg, kinematics))
    runtime, raw = demo.build_runtime(Cfg(data), tmp_path, serve=False)
    try:
        frame = runtime.camera.get_frame()
        q = frame.robot_joint_positions[str(data["arm"].get("name", "arm"))]
        expected = runtime.kin.fk(q)
        expected[:3, 3] += [.02, -.03, .01]
        np.testing.assert_allclose(frame.T_base_cam, expected)
        np.testing.assert_allclose(frame.T_base_cam, runtime.extrinsics.cam_to_base())
        assert raw.commands == []
    finally:
        runtime.arm.stop()
        demo.shutdown_runtime(runtime, raw)
