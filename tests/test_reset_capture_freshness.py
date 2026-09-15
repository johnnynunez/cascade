"""Reset capture and watcher regressions; synthetic CPU inputs, never live evidence.

No sockets, camera hardware, simulator, model or robot are started. The real
camera decoder, reset method, belief fusion and watcher run on synthetic inputs.
"""
from __future__ import annotations
import base64
from copy import deepcopy
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import zlib
import cv2
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cascade.config import Cfg
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.perception.isaac_camera import IsaacCamera
from cascade.perception.stream import CameraStream
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.skills.runtime import SkillRuntime
from cascade.types import Detection
OLD = [0.136, -0.279, 0.046]
NEW = [0.191, -0.039, 0.042]

@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    monkeypatch.setenv('CASCADE_REQUIRE_CUDA', '0')
    monkeypatch.setenv('CASCADE_DEVICE', 'cpu')
    import cascade.sim
    fake_world = ModuleType('cascade.sim.mujoco_world')
    fake_world.live_worlds = lambda: []
    monkeypatch.setitem(sys.modules, fake_world.__name__, fake_world)
    monkeypatch.setattr(cascade.sim, 'mujoco_world', fake_world, raising=False)

def packet(position=OLD, capture_t=10.0):
    bgr = np.full((12, 12, 3), [0, 180, 0], np.uint8)
    ok, jpg = cv2.imencode('.jpg', bgr)
    assert ok
    depth = np.full((12, 12), position[2], np.float32)
    transform = np.eye(4)
    transform[:2, 3] = position[:2]
    return {'ok': True, 'width': 12, 'height': 12, 'K': [[10.0, 0.0, 5.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]], 'rgb_jpeg_b64': base64.b64encode(jpg.tobytes()).decode(), 'depth_z_b64': base64.b64encode(zlib.compress(depth.tobytes())).decode(), 'T_base_cam': transform.tolist(), 't': capture_t, 'proprioception': {'version': 1, 'backend': 'isaac', 'robot_id': '/SYNTHETIC', 'joint_convention': 'asset', 'q': [0.0] * 6, 't': capture_t, 'time_source': 'physics_loop_monotonic'}}

def camera(monkeypatch):
    cam = IsaacCamera(Cfg({'bridge_host': 'SYNTHETIC_NO_CONNECTION', 'bridge_port': 1}))
    current = [packet()]
    requests = []

    def request(value):
        assert value == {'op': 'frame', 'camera': 'cam0'}
        requests.append(value)
        return deepcopy(current[0])
    monkeypatch.setattr(cam._client, 'request', request)
    return (cam, current, requests)

def detection():
    mask = np.zeros((12, 12), bool)
    mask[3:9, 3:9] = True
    return Detection('sponge', 0.8, np.array([3, 3, 9, 9]), mask=mask)

def runtime(cam):
    """Only pure skill plumbing; constructor sidecars/memory files are omitted."""
    rt = object.__new__(SkillRuntime)
    spawn = [0.18, -0.03, 0.04]
    reply = {'ok': True, 'props_reset': ['green_cube'], 'reset_verification': {'channel': 'physics', 'frame': 'robot_base', 'tolerance_m': 0.02, 'props': {'green_cube': {'spawn_position_m': spawn, 'position_m': spawn, 'error_m': 0.0, 'finite': True, 'within_tolerance': True}}}}
    rt._arm = SimpleNamespace(raw=SimpleNamespace(connected=False, reset_props=lambda: deepcopy(reply)), harness=SimpleNamespace(heartbeat=lambda: None))
    rt._arm_override = threading.local()
    rt.arm_rig = None
    rt.camera, rt.depth = (cam, SimpleNamespace(ensure_depth=lambda frame: frame))
    rt.beliefs, rt.memory = (BeliefStore(), EpisodicMemory())
    rt.held_object, rt.last_frame = (None, None)
    rt.watcher = None
    rt.detector = SimpleNamespace(detect=lambda *a, **kw: [detection()])
    rt._default_classes = None
    rt.cfg = Cfg({})
    rt._show_detections = lambda *a: None
    rt.skill_move_home = lambda: {'at': 'synthetic_home_only'}
    from cascade.perception.workspace import WorkspaceFilter
    rt._workspace = WorkspaceFilter()
    return rt

def watcher_case(monkeypatch, *, gate_geometry=False, start_paused=False):
    cam, _, _ = camera(monkeypatch)
    frame = cam.get_frame()
    entered, release = (threading.Event(), threading.Event())

    def gate():
        entered.set()
        assert release.wait(2), 'Synthetic test gate timed out'

    def detect(*args, **kwargs):
        if not gate_geometry:
            gate()
        return [detection()]
    stream = SimpleNamespace(latest=lambda: frame, set_overlay=lambda **kw: None, name='synthetic')
    watched = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda value: value), None)
    beliefs = BeliefStore()
    watcher = WorldWatcher([watched], SimpleNamespace(detect=detect), beliefs)
    if gate_geometry:

        def reject(*args, **kwargs):
            gate()
            return None
        watcher._workspace = SimpleNamespace(reject=reject)
    errors = []

    def tick():
        try:
            watcher._tick(watched)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=tick)
    pause = watcher.paused() if start_paused else None
    if pause is not None:
        pause.__enter__()
    worker.start()
    assert entered.wait(2)
    if pause is not None:
        pause.__exit__(None, None, None)
    return (watcher, beliefs, release, worker, errors)


def test_reset_rejects_repeated_old_capture_and_analyzes_exact_new_frame(monkeypatch):
    cam, current, requests = camera(monkeypatch)
    original_request = cam._client.request
    def delayed_refresh(value):
        if len(requests) >= 3:
            current[0] = packet(NEW, capture_t=11.)
        return original_request(value)
    monkeypatch.setattr(cam._client, "request", delayed_refresh)
    rt = runtime(cam)
    rt.beliefs.update("sponge", OLD, .8, color="green")

    result = rt.skill_reset_scene()
    assert result["ok"] and result["observation_refreshed"], result
    assert result["reset_verification"]["props"]["green_cube"]["position_m"] == [.18, -.03, .04]
    assert result["beliefs_forgotten"] == 1
    assert result["objects_visible"][0]["position"] == NEW
    assert len(requests) == 4, "Do not fetch another unchecked frame for detection"
    freshness = result["observation_freshness"][0]
    assert freshness["floor"]["t"] == 10. and freshness["observed"]["t"] == 11.
    assert len(rt.beliefs.all()) == 1


def test_frozen_reset_returns_failure_with_physics_receipt_and_no_stale_objects(monkeypatch):
    from cascade.perception import freshness
    cam, _, _ = camera(monkeypatch)
    rt = runtime(cam)
    rt.beliefs.update("sponge", OLD, .8, color="green")
    original = freshness.frames_after_reset
    monkeypatch.setattr(freshness, "frames_after_reset", lambda cameras: original(cameras, timeout_s=.03))
    result = rt.skill_reset_scene()
    assert not result["ok"] and not result["observation_refreshed"]
    assert result["observe_error"] and result["error"]
    assert result["objects_visible"] == [] and rt.beliefs.all() == []
    assert result["props_reset"] == ["green_cube"]
    assert result["reset_verification"]["props"]["green_cube"]["within_tolerance"]


@pytest.mark.parametrize("wrapper", ["stream", "hub"])
@pytest.mark.parametrize("backend", ["isaac", "local"])
def test_strict_stream_barrier_waits_for_capture_progress_and_preserves_local_backends(monkeypatch, wrapper, backend):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    from cascade.apps.live_view import FrameHub
    cam, current, _ = camera(monkeypatch)
    floor = cam.get_frame()
    duplicate = cam.get_frame()
    current[0] = packet(NEW, capture_t=11.)
    fresh = cam.get_frame()
    if backend == "local":
        for frame in (floor, duplicate, fresh):
            frame.capture = None
        duplicate.t = floor.t
    stream = CameraStream(cam) if wrapper == "stream" else FrameHub(cam, show=False)
    stream._latest, stream._latest_seq = floor, 1
    waiting = threading.Event()
    actual_wait = stream._cond.wait
    def wait(*args, **kwargs):
        waiting.set()
        return actual_wait(*args, **kwargs)
    monkeypatch.setattr(stream._cond, "wait", wait)
    def publish(frame):
        with stream._cond:
            stream._latest = frame
            stream._latest_seq += 1
            stream._cond.notify_all()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(stream.get_fresh_frame, after=floor, timeout_s=1)
        assert waiting.wait(1)
        publish(duplicate)
        with pytest.raises(FutureTimeout):
            future.result(timeout=.02)
        publish(fresh)
        assert future.result(timeout=1) is fresh


@pytest.mark.parametrize("wrapper", ["stream", "hub"])
def test_strict_barrier_raises_on_timeout_while_viewing_keeps_its_existing_fallback(monkeypatch, wrapper):
    from cascade.apps.live_view import FrameHub
    cam, _, _ = camera(monkeypatch)
    stream = CameraStream(cam) if wrapper == "stream" else FrameHub(cam, show=False)
    old = cam.get_frame()
    stream._latest, stream._latest_seq = old, 7
    assert stream.get_frame(timeout_s=0) is old
    with pytest.raises(TimeoutError):
        stream.get_fresh_frame(after=old, timeout_s=.01)


@pytest.mark.parametrize("bad", ["same_time", "wrong_robot", "wrong_source", "nan", "missing_state"])
def test_capture_barrier_cannot_be_bypassed_by_new_local_receipt_or_identity(monkeypatch, bad):
    from cascade.perception.freshness import newer_capture
    cam, current, _ = camera(monkeypatch)
    floor = cam.get_frame()
    current[0] = packet(NEW, capture_t=11.)
    fresh = cam.get_frame()
    if bad == "same_time":
        fresh.capture["t"] = floor.capture["t"]
        fresh.capture["proprioception"]["t"] = floor.capture["t"]
        assert fresh.frame_id > floor.frame_id and fresh.t > floor.t
        assert not newer_capture(fresh, floor)
        return
    if bad == "wrong_robot":
        fresh.capture["proprioception"]["robot_id"] = "/OTHER"
    elif bad == "wrong_source":
        fresh.capture["source"] = ("another-host", 1)
    elif bad == "nan":
        fresh.capture["t"] = float("nan")
    else:
        fresh.capture["proprioception"] = None
    with pytest.raises(ValueError):
        newer_capture(fresh, floor)


@pytest.mark.parametrize("during_pause", [False, True])
def test_watcher_rejects_inference_from_before_or_during_pause_after_resume(monkeypatch, during_pause):
    watcher, beliefs, release, worker, errors = watcher_case(monkeypatch, start_paused=during_pause)
    try:
        if not during_pause:
            with watcher.paused():
                beliefs.clear()
        release.set()
        worker.join(2)
        assert not worker.is_alive() and not errors
        assert beliefs.all() == []
        assert watcher.ticks == 1, "Detection and its heartbeat path still run while fusion is paused"
    finally:
        release.set()
        worker.join(2)


def test_watcher_pause_and_fusion_commit_are_atomic(monkeypatch):
    watcher, beliefs, release, worker, errors = watcher_case(monkeypatch, gate_geometry=True)
    try:
        with watcher.paused():
            beliefs.clear()
            release.set()
            worker.join(2)
            assert not worker.is_alive() and not errors
            assert watcher.is_paused and beliefs.all() == []
    finally:
        release.set()
        worker.join(2)


def scripted_watched_camera(monkeypatch, name, *, fuse=True, fail_floor=False, fail_fresh=False):
    cam, current, _ = camera(monkeypatch)
    floor = cam.get_frame()
    current[0] = packet(NEW, capture_t=11.)
    fresh = cam.get_frame()
    # The decoder camera address is synthetic; each stream is a separate
    # named source for the barrier, with no network I/O in the test.
    for frame in (floor, fresh):
        frame.capture["camera"] = name
    calls = []
    latest = [floor]
    def get_fresh_frame(*, after=None, timeout_s=5):
        calls.append(after)
        if (after is None and fail_floor) or (after is not None and fail_fresh):
            raise TimeoutError("Synthetic camera capture stalled")
        result = floor if after is None else fresh
        latest[0] = result
        return result
    stream = SimpleNamespace(name=name, get_fresh_frame=get_fresh_frame,
        latest=lambda: latest[0], set_overlay=lambda **kw: None)
    watched = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda frame: frame), None, fuse=fuse)
    return watched, calls, floor, fresh, latest


def test_reset_installs_floors_for_every_fusing_camera_and_skips_audience_only_camera(monkeypatch):
    main = scripted_watched_camera(monkeypatch, "main")
    side = scripted_watched_camera(monkeypatch, "side")
    audience = scripted_watched_camera(monkeypatch, "audience", fuse=False)
    beats = []
    beliefs = BeliefStore()
    watcher = WorldWatcher([entry[0] for entry in (main, side, audience)],
        SimpleNamespace(detect=lambda *a, **kw: [detection()]), beliefs,
        harness=SimpleNamespace(heartbeat=lambda: beats.append(True)))
    with watcher.paused():
        observed = watcher.reset_camera_frames(main[0].stream)
    assert [entry[0].name for entry in observed] == ["main", "side"]
    assert len(main[1]) == len(side[1]) == 2 and audience[1] == []
    for entry in (main, side):
        watched, _, floor, fresh, latest = entry
        assert watched.fusion_floor is floor and not watched.reset_pending
        stale_delivery = deepcopy(floor)
        stale_delivery.frame_id = 100
        stale_delivery.t = fresh.t + 10
        latest[0] = stale_delivery
        watcher._tick(watched)
        assert beliefs.all() == []
    assert len(beats) == 2, "Rejecting stale fusion must preserve camera heartbeats"
    main[-1][0] = main[3]
    watcher._tick(main[0])
    assert len(beliefs.all()) == 1


@pytest.mark.parametrize("fail_floor", [True, False])
def test_reset_camera_timeout_leaves_every_camera_fenced_until_real_new_capture(monkeypatch, fail_floor):
    main = scripted_watched_camera(monkeypatch, "main")
    side = scripted_watched_camera(monkeypatch, "side", fail_floor=fail_floor, fail_fresh=not fail_floor)
    beliefs = BeliefStore()
    watcher = WorldWatcher([main[0], side[0]], SimpleNamespace(detect=lambda *a, **kw: [detection()]), beliefs)
    with watcher.paused():
        with pytest.raises(TimeoutError):
            watcher.reset_camera_frames(main[0].stream)
    assert main[0].reset_pending and side[0].reset_pending
    for entry in (main, side):
        stale = deepcopy(entry[2])
        stale.frame_id = 100
        entry[-1][0] = stale
        watcher._tick(entry[0])
        assert beliefs.all() == []
    for entry in (main, side):
        entry[-1][0] = entry[3]
        watcher._tick(entry[0])
        assert not entry[0].reset_pending
    assert len(beliefs.all()) == 1


@pytest.mark.parametrize("side_stalled", [False, True])
def test_runtime_reset_requires_fresh_main_and_side_before_reporting_success(monkeypatch, side_stalled):
    main = scripted_watched_camera(monkeypatch, "main")
    side = scripted_watched_camera(monkeypatch, "side", fail_fresh=side_stalled)
    rt = runtime(main[0].stream)
    rt.watcher = WorldWatcher([main[0], side[0]], rt.detector, rt.beliefs)
    with rt.watcher.paused():
        result = rt.skill_reset_scene()
    assert result["ok"] is (not side_stalled)
    assert result["observation_refreshed"] is (not side_stalled)
    assert result["reset_verification"]["props"]["green_cube"]["within_tolerance"]
    if side_stalled:
        assert result["objects_visible"] == [] and result["observe_error"]
        assert rt.beliefs.all() == []
    else:
        assert [entry["camera"] for entry in result["observation_freshness"]] == ["main", "side"]
        assert result["objects_visible"][0]["position"] == NEW
