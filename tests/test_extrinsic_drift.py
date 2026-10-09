"""Runtime extrinsic drift monitor (perception/drift_monitor.py).

An eye-to-hand camera that gets knocked keeps streaming perfectly good
images -- and every belief and occupancy voxel fused through its old
extrinsic lands centimetres off. The monitor watches for that with the arm
itself as the target: when the arm is STATIC it compares depth with the
arm's surface at FK through the current extrinsic (a short single-view
ICP). K consecutive large offsets mark that ONE camera uncalibrated: fusion
and depth mapping stop for that stream (the same mechanism as a rejected
calibration record), never for the others, and never by touching the arm.

All synthetic and synchronous: check_once() with an injected clock, the
SyntheticDepthCamera as the stream, the real Extrinsics/WatchedCamera/
WorldWatcher objects.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import JOINT_SIGNS, URDF, needs_pin

from cascade.calibration.frames import so3_exp
from cascade.config import Cfg
from cascade.perception.drift_monitor import DriftMonitorConfig

# ── config ───────────────────────────────────────────────────────────────


def test_defaults_are_conservative_and_off():
    c = DriftMonitorConfig.from_config(None)
    assert c.enabled is False and c.auto_apply is False
    assert (c.period_s, c.max_offset_m, c.max_rot_deg, c.consecutive) == (5.0, 0.010, 2.0, 3)
    assert DriftMonitorConfig.from_config(Cfg({"enabled": True, "consecutive": 4})).consecutive == 4


@pytest.mark.parametrize("bad", [
    {"enabeld": True},                      # typo: unknown key
    {"period_s": float("nan")},
    {"period_s": 0.0},
    {"max_offset_m": -0.01},
    {"max_rot_deg": float("inf")},
    {"consecutive": 0},
    {"consecutive": 2.5},
    {"enabled": "false"},                   # a truthy string is not a bool
    {"auto_apply": 1},
    "yes",
])
def test_bad_config_is_refused(bad):
    with pytest.raises(ValueError):
        DriftMonitorConfig.from_config(bad)


def test_shipped_camera_profiles_validate():
    from cascade.config import load_demo_config

    for cam in ("d455f_scene", "d455f_wrist", "mock"):
        cfg = load_demo_config(camera=cam, arm="mock", llm="mock")
        c = DriftMonitorConfig.from_config((cfg.camera.get("extrinsics") or {}).get("drift_monitor"))
        assert c.enabled is False


# ── the monitor on a synthetic rig ───────────────────────────────────────

DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
T_TRUE = np.eye(4)
T_TRUE[:3, :3] = DOWN @ so3_exp([0.05, -0.06, 0.0])
T_TRUE[:3, 3] = [0.30, -0.03, 0.95]


def _bump(T, deg=2.0, axis=(1.0, 0.3, 0.0)):
    D = np.eye(4)
    a = np.asarray(axis) / np.linalg.norm(axis)
    D[:3, :3] = so3_exp(np.radians(deg) * a)
    return T @ D


@pytest.fixture(scope="module")
def kin():
    pytest.importorskip("pinocchio")
    from cascade.control.kinematics import Kinematics

    return Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)


@pytest.fixture(scope="module")
def surface(kin):
    from cascade.calibration.robot_surface import RobotSurface

    return RobotSurface.from_kinematics(kin, URDF)


@pytest.fixture(scope="module")
def preset_qs(kin):
    from cascade.calibration.session import load_poses
    from cascade.types import pose_to_transform

    home = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    out = []
    for p in load_poses("rebot_rs", "eye_to_hand"):
        sol = kin.ik(pose_to_transform(p), home)
        if sol.success:
            out.append(np.asarray(sol.q)[:6])
    return out


class Stream:
    def __init__(self, camera, name):
        self.camera, self.name = camera, name

    def latest(self):
        return self.camera.get_frame()


class Memory(list):
    def add(self, kind, text, data=None, **_):
        self.append((kind, text, data))


class Rig:
    """Two eye-to-hand streams over one WorldWatcher; only `cam` is monitored."""

    def __init__(self, kin, surface, tmp_path, *, config=None, settle_s=0.5, **cam_kw):
        from cascade.calibration.synthetic_depth import SyntheticDepthCamera
        from cascade.perception.depth_provider import DepthProvider
        from cascade.perception.drift_monitor import ExtrinsicDriftMonitor
        from cascade.perception.grounding import Extrinsics
        from cascade.perception.world import WatchedCamera, WorldWatcher

        self.t = 0.0
        self.q = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
        self.moving = False
        clock = lambda: self.t                                   # noqa: E731
        self.camera = SyntheticDepthCamera(kin, URDF, T_TRUE, lambda: self.q, seed=1,
                                           clock=clock, **cam_kw)
        self.cam = WatchedCamera(stream=Stream(self.camera, "scene"),
                                 depth=DepthProvider(Cfg({})),
                                 extrinsics=Extrinsics("eye_to_hand", T=T_TRUE.copy()))
        self.other = WatchedCamera(stream=Stream(self.camera, "side"),
                                   depth=DepthProvider(Cfg({})),
                                   extrinsics=Extrinsics("eye_to_hand", T=T_TRUE.copy()))
        self.watcher = WorldWatcher([self.cam, self.other], detector=None, beliefs=None)
        self.memory = Memory()
        self.lines = []
        self.monitor = ExtrinsicDriftMonitor(
            "scene", self.cam, watcher=self.watcher, surface_fn=lambda: surface,
            q_fn=lambda: self.q.copy(), motion_fn=lambda: self.moving,
            config=config or DriftMonitorConfig(enabled=True), run_dir=tmp_path,
            memory=self.memory, log=self.lines.append, clock=clock, settle_s=settle_s)

    def dwell(self, q=None):
        """Arm comes to rest at q; the monitor sees it settle, then checks."""
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        first = self.monitor.check_once()
        self.t += 1.0
        return first, self.monitor.check_once()


def _outcomes(results):
    return [r.outcome for r in results]


@needs_pin
def test_noise_alone_never_flags_the_camera(kin, surface, preset_qs, tmp_path):
    rig = Rig(kin, surface, tmp_path)
    results = [rig.dwell(q)[1] for q in preset_qs[::4]]
    assert set(_outcomes(results)) == {"ok"}, [(r.outcome, r.reason) for r in results]
    assert max(r.offset_m for r in results) < 0.006       # well below 10 mm
    assert rig.cam.fuse and rig.cam.extrinsics.calibrated
    assert rig.monitor.status()["state"] == "ok"


@needs_pin
def test_a_bumped_camera_is_flagged_within_k_checks_and_only_that_stream_stops(
        kin, surface, preset_qs, tmp_path):
    from cascade.types import SkillError

    rig = Rig(kin, surface, tmp_path)
    assert rig.dwell(preset_qs[0])[1].outcome == "ok"
    rig.camera.T_cam2base = _bump(T_TRUE, 2.0)          # someone knocks the tripod
    seen = []
    for q in preset_qs[1:6]:
        seen.append(rig.dwell(q)[1])
        if not rig.cam.fuse:
            break
    assert _outcomes(seen) == ["drift", "drift", "drift"], [(r.outcome, r.reason) for r in seen]
    assert all(r.offset_m > 0.015 for r in seen)
    # That stream: no fusion, no depth mapping, extrinsics refuse with why.
    assert rig.cam.fuse is False and rig.cam.maps_depth is False
    assert not rig.cam.extrinsics.calibrated
    with pytest.raises(SkillError, match="drift"):
        rig.cam.extrinsics.cam_to_base()
    # The other stream is untouched.
    assert rig.other.fuse and rig.other.extrinsics.calibrated
    # Surfaced: a log line, an event row, a memory note, the status.
    assert any("drift" in line and "scene" in line for line in rig.lines)
    rows = [json.loads(x) for x in (tmp_path / "extrinsics_drift.jsonl").read_text().splitlines()]
    assert any(r["event"] == "camera_uncalibrated" and r["camera"] == "scene" for r in rows)
    assert any("scene" in text for _, text, _ in rig.memory)
    st = rig.monitor.status()
    assert st["state"] == "drift" and st["calibrated"] is False


@needs_pin
def test_no_checks_while_the_arm_moves_or_settles(kin, surface, preset_qs, tmp_path,
                                                   monkeypatch):
    import cascade.perception.drift_monitor as dm

    calls = []
    real = dm.measure_offset
    monkeypatch.setattr(dm, "measure_offset", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    rig = Rig(kin, surface, tmp_path)
    rig.moving = True
    for q in preset_qs[:3]:
        a, b = rig.dwell(q)
        assert a.outcome == b.outcome == "skipped" and "moving" in b.reason
    rig.moving = False
    # Joints still changing between polls: never settled.
    for q in preset_qs[3:6]:
        rig.q = q
        r = rig.monitor.check_once()
        rig.t += 1.0
        assert r.outcome == "skipped"
    rig.q = None
    rig.monitor._q_fn = lambda: None                   # arm in standby (LazyArm)
    rig.t += 2.0
    assert "unavailable" in rig.monitor.check_once().reason
    assert calls == []


@needs_pin
def test_an_occluded_or_invisible_arm_is_inconclusive_never_drift(kin, surface, preset_qs,
                                                                  tmp_path):
    rig = Rig(kin, surface, tmp_path, config=DriftMonitorConfig(enabled=True, consecutive=2))
    real = rig.camera.get_frame

    def occluded():
        f = real()
        # A person leaning in: a slab in front of the whole arm.
        f.depth_m = np.where(f.depth_m > 0, 0.45, 0.0).astype(np.float32)
        return f
    rig.camera.get_frame = occluded
    results = [rig.dwell(q)[1] for q in preset_qs[:4]]
    assert set(_outcomes(results)) == {"inconclusive"}, [(r.outcome, r.reason) for r in results]
    assert rig.cam.fuse and rig.cam.extrinsics.calibrated


@needs_pin
def test_the_monitor_never_commands_the_arm(kin, surface, preset_qs, tmp_path):
    """Its only arm interface is a joint READER; flagging included."""
    rig = Rig(kin, surface, tmp_path)
    rig.camera.T_cam2base = _bump(T_TRUE, 3.0)
    for q in preset_qs[:4]:
        rig.dwell(q)
    assert rig.cam.fuse is False
    import inspect

    from cascade.perception.drift_monitor import ExtrinsicDriftMonitor
    src = inspect.getsource(ExtrinsicDriftMonitor)
    for verb in ("move_", "send_joint", "set_gripper", ".stop(", "estop", "halt("):
        assert verb not in src.replace("self._stop", ""), verb


# ── the watcher side: switching a stream off is atomic w.r.t. a tick ─────


@pytest.mark.parametrize("flag_mid_tick", [False, True])
def test_a_tick_in_flight_does_not_commit_after_its_camera_is_switched_off(flag_mid_tick):
    """The monitor flags a camera while the watcher is inside detection for
    a frame of that camera (T already read). That frame must not be fused."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from cascade.memory.beliefs import BeliefStore
    from cascade.perception.grounding import Extrinsics
    from cascade.perception.world import WatchedCamera, WorldWatcher
    from cascade.types import Detection, Frame

    H, W = 120, 160
    K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    T[:3, 3] = [0.3, 0.0, 0.6]
    depth = np.full((H, W), 0.6, dtype=np.float32)
    depth[50:70, 30:50] = 0.55
    frame = Frame(rgb=np.zeros((H, W, 3), np.uint8), depth_m=depth, K=K, frame_id=1,
                  depth_source="sensor")
    mask = np.zeros((H, W), dtype=bool)
    mask[50:70, 30:50] = True
    det = Detection("cube", 0.9, np.array([30, 50, 50, 70], dtype=np.float32), mask=mask)
    stream = SimpleNamespace(name="scene", latest=Mock(return_value=frame), set_overlay=Mock())
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f),
                        Extrinsics("eye_to_hand", T=T), fuse=True)
    beliefs = BeliefStore()
    holder = {}

    def detect(f, classes=None):
        if flag_mid_tick:
            holder["w"].set_camera_fusion(cam, fuse=False, map_depth=False)
        return [det]
    watch = WorldWatcher([cam], SimpleNamespace(detect=detect), beliefs,
                         harness=SimpleNamespace(heartbeat=Mock()))
    holder["w"] = watch
    watch._tick(cam)
    assert (len(beliefs.all()) == 0) == flag_mid_tick
    if flag_mid_tick:
        assert cam.generation == 1 and cam.fuse is False and cam.maps_depth is False


def test_extrinsics_invalidate_and_adopt_are_visible_to_every_holder():
    from cascade.perception.grounding import Extrinsics
    from cascade.types import SkillError

    e = Extrinsics("eye_to_hand", T=T_TRUE.copy(), compensation_m={"x": 0.01, "y": 0, "z": 0})
    holder = [e]                                   # e.g. SkillRuntime.extrinsics
    e.invalidate("extrinsic drift: test")
    assert not holder[0].calibrated and np.isnan(holder[0].T).all()
    with pytest.raises(SkillError, match="drift"):
        holder[0].cam_to_base()
    new = _bump(T_TRUE, 1.0)
    e.adopt(new, source="drift_candidate:x.json")
    assert holder[0].calibrated and np.allclose(holder[0].cam_to_base(), new)
    # The candidate is the full measured T_cam2base: compensation is not
    # applied a second time.
    assert np.allclose(e.T_compensation @ e.T_hand_eye, new)
