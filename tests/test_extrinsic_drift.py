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
    assert (c.period_s, c.max_offset_m, c.max_rot_deg, c.consecutive) == (5.0, 0.010, 3.0, 3)
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
        clock = lambda: self.t
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


# ── passive re-calibration ───────────────────────────────────────────────


def _events(tmp_path):
    p = tmp_path / "extrinsics_drift.jsonl"
    return [json.loads(x)["event"] for x in p.read_text().splitlines()] if p.exists() else []


@needs_pin
def test_views_from_before_the_knock_are_dropped_when_the_camera_is_flagged(
        kin, surface, preset_qs, tmp_path):
    rig = Rig(kin, surface, tmp_path)
    for q in preset_qs[0:9:3]:
        assert rig.dwell(q)[1].outcome == "ok"
    assert len(rig.monitor.snapshots) == 3
    rig.camera.T_cam2base = _bump(T_TRUE, 2.0)
    for q in preset_qs[12:24:4]:
        rig.dwell(q)
    assert rig.cam.fuse is False
    # Only the three views that saw the camera where it is NOW remain.
    assert [s.q for s in rig.monitor.snapshots] == [tuple(map(float, q)) for q in preset_qs[12:24:4]]


@needs_pin
@pytest.mark.parametrize("auto_apply", [False, True])
def test_a_passive_candidate_is_written_once_enough_diverse_static_views_exist(
        kin, surface, preset_qs, tmp_path, auto_apply):
    from cascade.calibration.dataset import load_hand_eye
    from cascade.calibration.frames import pose_error, se3_inv

    rig = Rig(kin, surface, tmp_path,
              config=DriftMonitorConfig(enabled=True, auto_apply=auto_apply))
    bumped = _bump(T_TRUE, 2.0)
    rig.camera.T_cam2base = bumped
    for q in preset_qs[::3]:                   # normal work: the arm stops here and there
        rig.dwell(q)
        if rig.monitor.candidate_path is not None:
            break
    path = rig.monitor.candidate_path
    assert path is not None, rig.monitor.status()
    assert path.parent == tmp_path / "extrinsics_candidates"
    rec = load_hand_eye(path)
    assert rec is not None and rec.markerless and "PASSIVE" in rec.note
    e = pose_error(se3_inv(bumped) @ rec.T_cam2base)
    assert 1000 * np.linalg.norm(e[:3]) < 5.0 and np.degrees(np.linalg.norm(e[3:])) < 0.5
    events = _events(tmp_path)
    assert "camera_uncalibrated" in events and "candidate_written" in events
    if auto_apply:
        assert "candidate_applied" in events
        assert rig.cam.fuse and rig.cam.extrinsics.calibrated
        assert np.allclose(rig.cam.extrinsics.cam_to_base(), rec.T_cam2base)
        assert rig.monitor.status()["state"] == "ok"
        assert rig.dwell(preset_qs[1])[1].outcome == "ok"     # checks pass again
    else:
        assert "candidate_applied" not in events
        assert rig.cam.fuse is False and not rig.cam.extrinsics.calibrated
        assert any("hand_eye_json:" in line and str(path) in line for line in rig.lines)


@needs_pin
def test_auto_apply_only_takes_a_candidate_that_explains_current_depth_better(
        kin, surface, preset_qs, tmp_path):
    rig = Rig(kin, surface, tmp_path, config=DriftMonitorConfig(enabled=True, auto_apply=True))
    for q in preset_qs[0:12:4]:
        rig.dwell(q)
    better, new, old = rig.monitor.candidate_is_better(T_TRUE)       # no better than active
    assert not better and new == pytest.approx(old)
    better, new, old = rig.monitor.candidate_is_better(_bump(T_TRUE, 3.0))
    assert not better and new < old


# ── runtime wiring (apps/demo.py) ────────────────────────────────────────


def _profile(mode="eye_to_hand", enabled=True, kind="realsense", T=True, **dm):
    ext = {"mode": mode, "drift_monitor": {"enabled": enabled, **dm}}
    if T:
        ext["T"] = T_TRUE.tolist()
    return Cfg({"type": kind, "name": "scene", "extrinsics": ext})


def _watched(ccfg):
    from types import SimpleNamespace

    from cascade.perception.grounding import Extrinsics

    e = Extrinsics.from_config(ccfg.extrinsics, fk_tcp2base=lambda: np.eye(4))
    return SimpleNamespace(stream=SimpleNamespace(name=ccfg.name), extrinsics=e, fuse=True,
                           map_depth=None, generation=0)


def test_monitors_are_built_only_where_they_can_work(capsys):
    from cascade.apps.demo import _drift_monitors

    kw = {"watcher": None, "surface_fn": lambda: None, "q_fn": lambda: None,
          "motion_fn": lambda: False, "run_dir": None, "memory": None}
    on = _profile()
    assert [m.name for m in _drift_monitors([on], [_watched(on)], **kw)] == ["scene"]
    off = _profile(enabled=False)
    assert _drift_monitors([off], [_watched(off)], **kw) == []
    assert _drift_monitors([Cfg({"type": "mock", "name": "m", "extrinsics": {}})],
                           [_watched(Cfg({"name": "m", "extrinsics": {}}))], **kw) == []
    # Uncalibrated: nothing to monitor -- said, not silent.
    nocal = _profile(T=False)
    nocal._data["extrinsics"]["hand_eye_json"] = "/nonexistent.json"
    assert _drift_monitors([nocal], [_watched(nocal)], **kw) == []
    assert "drift monitor" in capsys.readouterr().err
    # Enabled where it can never work is a configuration error.
    wrist = _profile(mode="eye_in_hand")
    with pytest.raises(ValueError, match="eye_to_hand"):
        _drift_monitors([wrist], [_watched(wrist)], **kw)
    rgb = _profile(kind="uvc")
    with pytest.raises(ValueError, match="depth"):
        _drift_monitors([rgb], [_watched(rgb)], **kw)
    bad = _profile(consecutive=0)
    with pytest.raises(ValueError, match="consecutive"):
        _drift_monitors([bad], [_watched(bad)], **kw)


@needs_pin
def test_build_runtime_starts_reports_and_stops_the_monitor(tmp_path):
    from cascade.apps.demo import (
        _runtime_state,
        build_runtime,
        owned_threads,
        shutdown_runtime,
    )
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["camera"]["extrinsics"]["drift_monitor"] = {"enabled": True, "period_s": 0.5}
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        assert [m.name for m in runtime.drift_monitors] == [runtime.rig.primary.name]
        mon = runtime.drift_monitors[0]
        assert mon._thread is not None and mon._thread.is_alive()
        assert ("drift-monitor", mon._thread) in [(l.split(":")[0], t)
                                                 for l, t in owned_threads(runtime)]
        state = _runtime_state(runtime)
        assert state["extrinsics_drift"][mon.name]["state"] == "ok"
    finally:
        receipt = shutdown_runtime(runtime, arm)
    assert not mon._thread or not mon._thread.is_alive()
    assert "drift_monitors" in [s["stage"] for s in receipt["stages"]]
    assert not receipt["pending_threads"]
