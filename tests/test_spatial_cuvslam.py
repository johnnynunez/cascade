"""CPU contracts from cuVSLAM b405f132/python/cuvslam2.cpp; no native SDK proof."""
from dataclasses import replace
import hashlib
import multiprocessing
import os
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.sensing import BufferedSensorProvider, SensorHub
from cascade.sensing.hub import SensorDescriptor
from cascade.sensing.models import MeasurementMetadata, ObservationEnvelope, RgbdPayload
from cascade.spatial import cuvslam
from cascade.spatial.domain import build_spatial_domain
from cascade.spatial.cuvslam_worker import CuVslamProcess
from cascade.spatial import cuvslam_worker


@pytest.mark.parametrize('failure', ['return_false', 'raise'])
def test_native_validation_retains_failure_and_attempts_both_owner_closures(failure):
    from pathlib import Path
    import runpy
    close = runpy.run_path(str(Path(__file__).resolve().parents[1]
                              / 'scripts/validate_cuvslam_rgbd.py'))['close_owners']
    calls = []
    def close_domain():
        calls.append('domain')
        if failure == 'raise':
            raise RuntimeError('worker close failed')
        return {'ok': False}
    def close_hub(timeout):
        calls.append('hub')
        return {'ok': True}
    report = {'ok': True, 'error': 'retained earlier diagnostic'}
    close(report, NS(close=close_domain), NS(close=close_hub))
    assert calls == ['domain', 'hub']
    assert report['ok'] is False and report['domain_close']['ok'] is False
    assert report['hub_close']['ok'] is True
    assert report['error'] == 'retained earlier diagnostic'


class SdkContract:
    """Implements only the signatures and return shapes used from the pinned binding."""
    def __init__(self):
        self.calls = []
        self.effect = None
        self.camera = None
        self.Pose = lambda *, rotation, translation: NS(rotation=rotation, translation=translation)
        self.Camera = NS
        self.Rig = lambda cameras: cameras
        self.Odometry = NS(RGBDSettings=NS, Config=NS, OdometryMode=NS(RGBD="rgbd"))
        self.Slam = NS(Config=NS)
        outer = self

        class Tracker:
            Mode = NS(OdometryWithSlamOffline="offline-slam")

            def __init__(self, rig, mode, odom_config, slam_config):
                assert mode == self.Mode.OdometryWithSlamOffline
                assert not odom_config.async_sba and slam_config.sync_mode
                assert not odom_config.use_motion_model
                assert not slam_config.gt_align_mode and slam_config.map_cache_path == ""
                assert 0 < slam_config.max_map_size <= 4096
                assert odom_config.rgbd_settings.depth_camera_id == 0
                assert odom_config.rgbd_settings.depth_scale_factor == 1000
                outer.camera = rig[0]

            def track(self, timestamp, images, *, depths):
                rgb, depth = images[0], depths[0]
                assert rgb.dtype == np.uint8 and rgb.shape == (*depth.shape, 3)
                assert depth.dtype == np.uint16 and depth.ndim == 2
                assert rgb.flags.c_contiguous and depth.flags.c_contiguous
                outer.calls.append((timestamp, rgb.copy(), depth.copy()))
                pose = NS(rotation=[0., 0., 1., 0.], translation=[1., 2., 3.])
                result = (NS(timestamp_ns=timestamp, world_from_rig=NS(pose=pose)), pose)
                return outer.effect(result) if outer.effect else result

        self.Tracker = Tracker


@pytest.fixture
def rig(monkeypatch, request):
    now = [10.]
    descriptor = SensorDescriptor("rgbd", "duck", "camera", "rgbd", "optical", "simulation",
        "physics", "e"*64, calibration_id="c"*64, max_age_s=.5, read_timeout_s=1.)
    hub = SensorHub(clock=lambda: now[0])
    provider = BufferedSensorProvider(descriptor)
    hub.register(provider)
    sdk = SdkContract()
    monkeypatch.setattr(cuvslam, "_load_sdk", lambda binding: sdk)
    class CpuProcess:
        def __init__(self, settings, *, timeout_s):
            camera, binding, gap, capacity = settings
            self.tracker = cuvslam._tracker(NS(**camera), binding, gap, capacity)

        def warmup(self):
            pass

        def track(self, *args, **kwargs):
            return self.tracker.track(*args, **kwargs)

        def request_stop(self):
            pass

        def close(self):
            return True

    monkeypatch.setattr(cuvslam, "CuVslamProcess", CpuProcess)
    config = dict(sensor_domain="sensors", sensor_id="rgbd", map_id="room", map_epoch="map1",
                  map_frame_id="local-map", binding_sha256="b"*64, max_gap_s=.1)
    config.update(getattr(request, "param", {}))
    domain = cuvslam.CuVslamSpatialDomain("space", "duck", hub, clock=lambda: now[0], **config)
    payload = RgbdPayload(MeasurementMetadata("optical", "c"*64), 4, 3, bytes(range(36)),
        np.full((3, 4), 1.234, dtype="<f4").tobytes(), (2., 0., 1.5, 0., 2., 1.5, 0., 0., 1.),
        pixel_center_offset_uv=(.5, .5))
    observation = ObservationEnvelope("camera", "rgbd", "sensor1", 7, "simulation", .35,
                                      10., 0., "e"*64, "physics", payload)

    def publish(obs=None):
        obs = obs or observation
        provider.publish(obs)
        assert hub.read("rgbd") is obs
        return {"epoch": obs.epoch, "sequence": obs.sequence, "capture_sha256": obs.sha256}

    prepared = domain.execute("warmup_localization", publish(replace(observation, sequence=6, capture_time_s=.3)))
    assert prepared["ok"], prepared
    yield NS(now=now, hub=hub, domain=domain, sdk=sdk, publish=publish, obs=observation, config=config)
    domain.close()
    assert hub.close(2)["ok"]


def test_real_sdk_shapes_local_map_and_no_ground_truth_or_motion(rig):
    assert rig.domain.execute("get_localization", {})["tracking_state"] == "uninitialized"
    truth = np.eye(4)
    truth[:3, 3] = [900, 800, 700]
    args = rig.publish(replace(rig.obs, payload=replace(rig.obs.payload,
        world_from_camera=tuple(truth.flatten()), world_frame_id="sim-world")))
    result = rig.domain.execute("track_capture", args)
    assert result["ok"], result
    assert result["pose"]["translation_m"] == [1, 2, 3]
    assert result["pose"]["rotation_wxyz"] == [0, 0, 0, 1]
    assert result["pose"]["stamp"]["measurement_kind"] == "estimated"
    assert result["pose"]["stamp"]["epoch"] == rig.domain.map_epoch
    assert result["map_epoch_label"] == "map1"
    assert result["sensor_epoch"] == "sensor1" and not result["physical_admission"]
    assert result["pose"]["position_error_m"] is None
    assert rig.sdk.camera.principal == (1., 1.)
    assert rig.sdk.camera.rig_from_camera.translation == [0, 0, 0]
    assert rig.sdk.calls[0][0] == 350_000_000
    assert np.all(rig.sdk.calls[0][2] == 1234)
    assert rig.domain.motion_skills == frozenset()
    assert rig.domain.resources[0].admission == "unvalidated"
    rig.now[0] += .1
    assert rig.domain.execute("get_localization", {})["capture_age_s"] == pytest.approx(.1)
    assert len(rig.sdk.calls) == 1  # Reading an estimate never reruns tracking.


@pytest.mark.parametrize("failure", ["loss", "slam_missing", "timestamp", "nan", "rotation", "exception"])
def test_sdk_bad_output_invalidates_without_identity_fallback(rig, failure):
    def fail(result):
        odom, pose = result
        if failure == "loss":
            odom.world_from_rig = None
        elif failure == "slam_missing":
            return odom, None
        elif failure == "timestamp":
            odom.timestamp_ns += 1
        elif failure == "nan":
            pose.translation[0] = float("nan")
        elif failure == "rotation":
            pose.rotation = [0, 0, 0, 0]
        else:
            raise RuntimeError("tracker failed")
        return result
    rig.sdk.effect = fail
    result = rig.domain.execute("track_capture", rig.publish())
    assert not result["ok"] and result["map_epoch_invalidated"]
    assert rig.domain._latest is None
    assert not rig.domain.execute("get_localization", {})["ok"]
    assert not rig.domain.reset_stop()["ok"]


@pytest.mark.parametrize("change", ["replay", "gap", "calibration", "skew", "stale", "bad_hash"])
def test_capture_continuity_and_binding_refuse_before_next_sdk_call(rig, change):
    args = rig.publish()
    assert rig.domain.execute("track_capture", args)["ok"]
    next_obs = replace(rig.obs, sequence=8, capture_time_s=.4)
    if change == "gap":
        args = rig.publish(replace(next_obs, capture_time_s=.451))
    elif change == "calibration":
        args = rig.publish(replace(next_obs, payload=replace(next_obs.payload,
            intrinsics=(3., 0., 1.5, 0., 2., 1.5, 0., 0., 1.))))
    elif change == "skew":
        args = rig.publish(replace(next_obs, payload=replace(next_obs.payload,
            intrinsics=(2., .1, 1.5, 0., 2., 1.5, 0., 0., 1.))))
    elif change == "stale":
        rig.now[0] += .501
    elif change == "bad_hash":
        args = args | {"capture_sha256": "f"*64}
    result = rig.domain.execute("track_capture", args)
    assert not result["ok"] and result["map_epoch_invalidated"]
    assert len(rig.sdk.calls) == 1 and rig.domain._latest is None


@pytest.mark.parametrize("depth", [100., .0001, 0.])
def test_depth_encoding_refuses_clipping_or_positive_to_invalid(rig, depth):
    obs = replace(rig.obs, payload=replace(rig.obs.payload,
        depth_m_f32le=np.full((3, 4), depth, dtype="<f4").tobytes()))
    result = rig.domain.execute("track_capture", rig.publish(obs))
    assert not result["ok"] and "uint16" in result["error"]
    assert not rig.sdk.calls


@pytest.mark.parametrize("event", ["stale", "stop", "close"])
def test_inflight_result_cannot_restore_invalidated_pose(rig, event):
    entered, release = threading.Event(), threading.Event()
    def wait(result):
        entered.set()
        assert release.wait(2)
        return result
    rig.sdk.effect = wait
    args = rig.publish()
    results = []
    worker = threading.Thread(target=lambda: results.append(rig.domain.execute("track_capture", args)))
    worker.start()
    try:
        assert entered.wait(2)
        if event == "stale":
            rig.now[0] += .501
        elif event == "stop":
            assert rig.domain.stop()["ok"]
        else:
            assert rig.domain.close() == {"ok": False, "actuation": False, "in_flight": True}
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and not results[0]["ok"]
    assert rig.domain._latest is None and rig.domain._tracker is None


def test_builder_is_passive_and_robot_binding_required(rig, monkeypatch):
    monkeypatch.setattr(cuvslam, "_load_sdk", lambda _: pytest.fail("passive construction imported SDK"))
    profile = {"kind": "spatial", "robot_id": "duck", "cuvslam": rig.config}
    built = build_spatial_domain("other", profile, sensor_domains={"sensors": NS(hub=rig.hub)})
    assert built.required_resources == ("sensors/rgbd",)
    assert built.map_epoch != rig.domain.map_epoch
    assert built.map_epoch_label == rig.domain.map_epoch_label == "map1"
    assert built.close()["ok"]
    with pytest.raises(ValueError, match="this robot"):
        build_spatial_domain("other", profile | {"robot_id": "foreign"},
                             sensor_domains={"sensors": NS(hub=rig.hub)})


def test_composed_runtime_routes_localization_with_sensor_requirement(rig):
    from cascade.apps.robot_runtime import DomainAdapter
    from cascade.robotics.runtime import RobotRuntime
    from cascade.sensing.domain import SensorDomain
    sensor = SensorDomain("sensors", rig.hub)
    adapters = {}
    for name, kind, domain in (("sensors", "sensors", sensor), ("space", "spatial", rig.domain)):
        adapters[name] = DomainAdapter(name, {"kind": kind}, domain.resources, domain.tool_specs,
            frozenset(), runtime=domain, required_resources=getattr(domain, "required_resources", ()))
    runtime = RobotRuntime(adapters)
    try:
        spec = runtime.tool_descriptors["space.track_capture"]
        assert "sensors/rgbd" in spec.requires and not spec.writes
        result = runtime.execute("space.track_capture", rig.publish())
        assert result["ok"] and result["pose"]["parent"] == "local-map"
        assert runtime.stop()["latched"]
        assert not runtime.execute("space.get_localization", {})["ok"]
        assert rig.domain._latest is None
    finally:
        assert runtime.close()["ok"]


def test_warmup_does_not_admit_image_captured_before_it_finished(rig):
    # Receipt age is preserved; a newer sequence alone cannot mask old capture.
    rig.domain._prepared_at = 10.1
    rig.now[0] = 10.2
    args = rig.publish(replace(rig.obs, received_monotonic_s=10.2, producer_age_s=.15))
    result = rig.domain.execute("track_capture", args)
    assert not result["ok"] and "follow warmup" in result["error"]
    assert not rig.sdk.calls


def test_cleanup_exception_preserves_owner_but_releases_operation_lock(rig):
    worker = rig.domain._tracker
    close = worker.close
    def fail():
        raise RuntimeError("worker cleanup failed")
    worker.close = fail
    rig.domain.stop()
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            rig.domain.execute("get_localization", {})
        assert rig.domain._tracker is worker
        assert rig.domain._work.acquire(blocking=False)
        rig.domain._work.release()
    finally:
        worker.close = close


def test_optional_import_checks_binary_before_loading_and_version_after(tmp_path, monkeypatch):
    extension = tmp_path / "pycuvslam.test.so"
    extension.write_bytes(b"CPU test fixture, not a binary")
    expected = hashlib.sha256(extension.read_bytes()).hexdigest()
    monkeypatch.setattr(cuvslam.importlib.util, "find_spec", lambda _: NS(origin=str(tmp_path / "__init__.py")))
    imported = []
    sdk = NS(_core=NS(__file__=str(extension)), get_version=lambda: ("17.0.0", 17, 0, 0))
    monkeypatch.setattr(cuvslam.importlib, "import_module", lambda name: imported.append(name) or sdk)
    with pytest.raises(ValueError, match="SHA256"):
        cuvslam._load_sdk("f"*64)
    assert not imported
    assert cuvslam._load_sdk(expected) is sdk
    sdk.get_version = lambda: ("16.0.0", 16, 0, 0)
    with pytest.raises(ValueError, match="version"):
        cuvslam._load_sdk(expected)


def _cpu_worker(connection, behavior):
    """Actual spawned process, never imports the optional native SDK."""
    try:
        entered = None
        if isinstance(behavior, tuple):
            behavior, entered = behavior
        if behavior == "ignore_term":
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if behavior == "warmup_hang":
            time.sleep(20)
        connection.send({"ok": True})
        request = connection.recv()
        if entered is not None:
            entered.set()
        if behavior == "ignore_term":
            import ctypes
            ctypes.PyDLL(None).sleep(20)  # Hold this child's GIL, ignore TERM.
        if behavior == "track_hang":
            time.sleep(20)
        if behavior == "oversized":
            connection.send_bytes(b"x" * 5000)
        else:
            pose = {"translation": (1., 2., 3.), "rotation": (0., 0., 0., 1.)}
            connection.send({"ok": True, "timestamp_ns": request["timestamp"],
                             "odometry": pose, "slam": pose})
            connection.recv()  # Keep the owned process alive until close.
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


@pytest.mark.parametrize("behavior", ["ok", "track_hang", "oversized"])
def test_owned_process_ipc_deadline_and_reaping(behavior):
    worker = CuVslamProcess(behavior, timeout_s=5., target=_cpu_worker)
    try:
        worker.warmup()
        worker.timeout_s = .15
        args = dict(images=[np.zeros((3, 4, 3), dtype=np.uint8)],
                    depths=[np.full((3, 4), 1000, dtype=np.uint16)])
        if behavior == "ok":
            odom, slam = worker.track(100, **args)
            assert odom.timestamp_ns == 100 and slam.translation == (1, 2, 3)
        else:
            with pytest.raises((TimeoutError, RuntimeError)):
                worker.track(100, **args)
            with pytest.raises(RuntimeError, match="closed"):
                worker.track(101, **args)
    finally:
        assert worker.close()
    assert not worker._process.is_alive() and not worker._thread.is_alive()


def test_worker_warmup_timeout_never_leaves_owned_process_running():
    worker = CuVslamProcess("warmup_hang", timeout_s=.15, target=_cpu_worker)
    try:
        with pytest.raises(TimeoutError):
            worker.warmup()
    finally:
        assert worker.close()
    assert worker._process.exitcode is not None


def test_ipc_thread_start_failure_preserves_exception_and_reaps_owned_child(monkeypatch):
    def fail_start(self):
        raise RuntimeError("thread startup failed")
    monkeypatch.setattr(cuvslam_worker.threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread startup failed") as caught:
        CuVslamProcess("warmup_hang", timeout_s=1., target=_cpu_worker)
    worker = caught.value.cuvslam_worker
    assert worker._process.exitcode is not None and not worker._process.is_alive()
    assert worker._connection.closed


def test_done_response_cannot_bypass_elapsed_deadline(monkeypatch):
    worker = CuVslamProcess.__new__(CuVslamProcess)
    worker.timeout_s = 1.
    worker._closed = threading.Event()
    def publish(item):
        _, done, result = item
        result["value"] = {"ok": True}
        done.set()
    worker._requests = NS(put_nowait=publish)
    worker.request_stop = worker._closed.set
    clock = iter([10., 11.])
    monkeypatch.setattr(cuvslam_worker, "time", NS(monotonic=lambda: next(clock)))
    with pytest.raises(TimeoutError, match="after deadline"):
        worker._call(None)
    assert worker._closed.is_set()


@pytest.mark.skipif(os.name != "posix", reason="POSIX TERM/KILL control")
def test_stop_wakes_parent_even_when_native_child_holds_gil_and_ignores_term():
    entered = multiprocessing.get_context("spawn").Event()
    worker = CuVslamProcess(("ignore_term", entered), timeout_s=10., target=_cpu_worker)
    errors = []
    def track():
        try:
            worker.track(100, images=[np.zeros((3, 4, 3), dtype=np.uint8)],
                         depths=[np.ones((3, 4), dtype=np.uint16)])
        except RuntimeError as exc:
            errors.append(str(exc))
    caller = threading.Thread(target=track)
    try:
        worker.warmup()
        caller.start()
        assert entered.wait(5)
        worker.request_stop()
        caller.join(2)
        assert not caller.is_alive() and errors
    finally:
        assert worker.close()
        if caller.ident is not None:
            caller.join(2)
    assert worker._process.exitcode == -9 and not worker._thread.is_alive()
