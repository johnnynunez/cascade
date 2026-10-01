"""Fault-probe contracts with a real camera pump/barrier and synthetic physics."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.perception.freshness import capture_marker, frames_after_reset
from cascade.perception.stream import CameraStream
from test_occupancy_payload import frame as payload_frame

PATH = Path(__file__).resolve().parents[1] / "benchmark/diagnostics/nvblox_camera_recovery.py"
spec = importlib.util.spec_from_file_location("camera_recovery_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_repeated_actual_frame_cannot_pass_the_normal_producer_barrier():
    class Producer:
        has_depth = True
        counter = 0
        def open(self): pass
        def close(self): pass
        def get_frame(self):
            self.counter += 1
            value = payload_frame(False, float(self.counter))
            value.frame_id = self.counter
            return value
    producer = Producer()
    stream = CameraStream(producer, name="side", rate_hz=100.)
    stream.open()
    try:
        freeze = probe.FrozenProducer(stream)
        with freeze:
            assert stream.get_frame() is freeze.frame
            assert producer.get_frame() is freeze.frame
            with pytest.raises(TimeoutError, match="fresh capture"):
                frames_after_reset([stream], timeout_s=.12)
        report = freeze.report()
        assert report["deliveries"] > 2 and report["all_deliveries_same_marker"] and report["unchanged_frame"]
        rows = frames_after_reset([stream], timeout_s=1.)
        assert capture_marker(rows[0][2])["t"] > capture_marker(rows[0][1])["t"]
    finally:
        stream.close()


def test_frozen_getter_is_restored_even_when_diagnostic_raises():
    frame = payload_frame(False, 1.)
    producer = SimpleNamespace(get_frame=lambda: frame)
    stream = SimpleNamespace(_camera=producer, get_frame=producer.get_frame)
    stream.latest = lambda: producer.get_frame()
    original = producer.get_frame
    freeze = probe.FrozenProducer(stream)
    # The preceding test exercises the real pump; this isolates restoration.
    freeze.deliveries = 2
    stream.latest = lambda: frame
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        with freeze:
            raise RuntimeError("injected diagnostic failure")
    assert producer.get_frame is original


def physics_rows():
    rows = []
    for index in range(8):
        physics = {"engine": "physx", "robot_id": "/Robot", "sim_time": index * .1,
            "physics_step": index, "q": [-.3, -.4, -.5, .05, .05], "dq": [0.] * 5,
            "gripper": {"q": [.05, .05], "open_fractions": [1., 1.]},
            "props": {"orange": {"position_m": [.18, .12, .026],
                "linear_velocity_m_s": [0., 0., 0.], "angular_velocity_rad_s": [0., 0., 0.]}},
            "spawn_positions_m": {"orange": [.18, .12, .026]},
            "contacts": {}, "gpu_attestation": {}}
        rows.append({"sequence": index, "client_started_monotonic": 10. + index,
                     "client_finished_monotonic": 10.1 + index, "physics": physics})
    return rows


def reset_report(rows):
    cfg = Cfg({"arm": {"home_q": [.3, .4, .5], "joint_signs": [-1, -1, -1],
                       "bridge_robot_id": "/Robot", "settle_tol": .045}})
    return probe.reset_window(rows, after=0., cfg=cfg, expected_props=["orange"], engine="physx")


def test_reset_only_physics_requires_home_jaws_spawn_and_settling():
    assert reset_report(physics_rows())["pass"]
    for key, value in [("q", [0.] * 5), ("gripper", {"q": [.02, .02], "open_fractions": [.4, .4]})]:
        rows = physics_rows()
        rows[-1]["physics"][key] = value
        assert not reset_report(rows)["pass"]
    for key, value in [("position_m", [.4, .1, .03]), ("linear_velocity_m_s", [.021, 0., 0.]),
                       ("angular_velocity_rad_s", [.21, 0., 0.])]:
        rows = physics_rows()
        rows[-1]["physics"]["props"]["orange"][key] = value
        assert not reset_report(rows)["pass"]
    assert not reset_report(physics_rows()[:3])["pass"]


def test_all_three_camera_barriers_require_matching_identity_and_new_clock():
    rows = []
    for name in ("cam0", "side", "proof"):
        floor = {"channel": "producer_capture", "camera": name, "robot_id": "/Robot", "t": 1.}
        rows.append({"floor": floor, "observed": {**floor, "t": 2.}})
    expected = {"cam0", "side", "proof"}
    assert all(probe.fresh_camera_checks({"observation_freshness": rows}, expected).values())
    stale = copy.deepcopy(rows)
    stale[1]["observed"]["t"] = 1.
    assert not all(probe.fresh_camera_checks({"observation_freshness": stale}, expected).values())
    wrong = copy.deepcopy(rows)
    wrong[1]["observed"]["robot_id"] = "/Different"
    assert not all(probe.fresh_camera_checks({"observation_freshness": wrong}, expected).values())
    assert not all(probe.fresh_camera_checks({"observation_freshness": rows[:2]}, expected).values())


def test_snapshots_must_start_after_requested_boundary(monkeypatch):
    clock = [10.]
    rows = physics_rows()
    old, fresh = rows[0], rows[1]
    old["client_started_monotonic"] = 9.
    fresh["client_started_monotonic"] = 10.1
    observer = SimpleNamespace(records=[old], errors=[], _lock=threading.Lock())
    occupancy = SimpleNamespace(_refresh_lock=threading.RLock(), scene_reset_pending=True,
        is_stale=lambda: True, last_error=None, _body_error="pending", _grid=None,
        _last_refresh=None, _depth_history={})
    runtime = SimpleNamespace(arm=SimpleNamespace(harness=SimpleNamespace(occupancy=occupancy)), held_object="orange")
    def tick(seconds):
        clock[0] += seconds
        observer.records.append(fresh)
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=tick))
    result = probe.fresh_state(observer, runtime)
    assert result["sequence"] == fresh["sequence"]
    assert result["client_started_monotonic"] >= result["requested_after_monotonic"]
    assert result["held"] == "orange" and result["map"]["pending"]


def test_actuator_trace_distinguishes_rejected_requests_from_actual_commands(tmp_path):
    actual = []
    backend = SimpleNamespace(send_joint_target=lambda q: actual.append("joint"),
        set_gripper=lambda p: actual.append("jaw"), reset_props=lambda: actual.append("reset"))
    def refused(*args, **kwargs):
        raise RuntimeError("refused before stream")
    arm = SimpleNamespace(raw=backend, move_joints=refused, set_gripper=lambda p: backend.set_gripper(p))
    runtime = SimpleNamespace(arm=arm, held_object=None)
    recorder = probe.ActuatorTrace(runtime, tmp_path / "commands.jsonl")
    recorder.phase = "blocked"
    with pytest.raises(RuntimeError):
        arm.move_joints([0., 0., 0.])
    calls = recorder.calls("blocked")
    assert len(calls) == 1 and calls[0]["layer"] == "request" and not actual
    recorder.phase = "restored"
    backend.reset_props()
    assert recorder.calls("restored")[0]["layer"] == "actuator"
    assert actual == ["reset"]


def test_cleanup_never_parks_releases_or_resets():
    events = []
    runtime = SimpleNamespace(watcher=SimpleNamespace(stop=lambda: events.append("watcher")),
        rig=SimpleNamespace(close=lambda: events.append("camera")),
        arm=SimpleNamespace(disconnect=lambda: events.append("disconnect"),
            move_joints=lambda *a: pytest.fail("cleanup motion"), set_gripper=lambda *a: pytest.fail("cleanup release")))
    assert probe.close_without_motion(runtime) == []
    assert events == ["watcher", "camera", "disconnect"]
