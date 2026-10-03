"""Synthetic route execution through the real mobile runtime; no physics claim."""
from dataclasses import replace
import json
import math
import threading
import time

import pytest

from cascade.apps.mobile_runtime import build_mobile_runtime
from cascade.config import load_demo_config
from cascade.spatial.frames import SpatialStamp, TransformSample, sha256
from cascade.spatial.grid import GridSnapshot
from cascade.spatial.navigation import NavigationSample, VolumeClearance
from test_mobile_effects import limits


class SyntheticSource:
    """Only this software fixture may derive navigation from mock feedback."""
    def __init__(self):
        stamp = SpatialStamp("test-map", "map-epoch", "mock-time", 0., "fixture",
                             "a"*64, "b"*64, "synthetic")
        self.grid = GridSnapshot("map", stamp, .1, (-3.05, -3.05), 61, 61, (0,)*3721)
        self.raw = None
        self.previous = -1
        self.queries = []
        self.mutate = lambda sample: sample
        self.clearance = 1.
        self.hold = self.entered = None

    def read(self, *, deadline_monotonic_s):
        if self.entered is not None:
            self.entered.set()
            self.hold.wait()
        while True:
            state = self.raw.get_state()
            if state.step > self.previous:
                break
            if time.monotonic() >= deadline_monotonic_s:
                raise TimeoutError("mock capture deadline")
            time.sleep(.001)
        self.previous = state.step
        stamp = replace(self.grid.stamp, time_s=state.sim_time_s,
                        source_sha256=sha256({"step": state.step}))
        pose = TransformSample("map", "base", state.position_world, state.orientation_wxyz,
                               stamp, position_error_m=.00001, angular_error_rad=.00001)
        return self.mutate(NavigationSample(state.robot_id, None, "c"*64, pose,
            self.grid, "d"*64, state.epoch, state.step,
            state.received_monotonic_s, state.producer_age_s))

    def swept_clearance(self, query, *, deadline_monotonic_s):
        self.queries.append(query)
        return VolumeClearance(sha256(query), "d"*64, self.clearance)


@pytest.fixture
def navigation(tmp_path):
    cfg = load_demo_config(base="microduck_mock", llm="mock")
    profile = cfg.bases[0]
    profile["capabilities"] = ["turn", "walk_distance", "stop_navigation"]
    profile["distance_control"] = dict(speed_m_s=.15, max_distance_m=.1, tolerance_m=.005,
        max_lateral_drift_m=.01, max_heading_drift_rad=.04, min_height_m=.1, max_tilt_rad=.5)
    profile["verifier"] = limits(sample_interval_s=.02, read_timeout_s=.15,
        max_state_age_s=.5, max_linear_speed_m_s=.5, max_sample_gap_s=.5)
    settings = dict(base=profile["name"], base_frame_id="base", geometry_sha256="c"*64,
        calibration_sha256="b"*64, body_radius_m=.05, body_z_min_m=-.1, body_z_max_m=.2,
        clearance_m=.01, stop_margin_m=.35, position_error_bound_m=.00001,
        heading_error_bound_rad=.00001, goal_tolerance_m=.015, max_route_m=2.,
        max_commands=12, wall_timeout_s=6., map_max_age_s=10.)
    source = SyntheticSource()
    runtime, rig = build_mobile_runtime(cfg, tmp_path, navigation_source=source,
                                        navigation_settings=settings)
    source.raw = rig.primary.raw
    runtime.mobile.execute("get_base_state", {})
    try:
        yield runtime, source
    finally:
        if source.hold is not None:
            source.hold.set()
        deadline = time.monotonic()+1
        while runtime._reader is not None and runtime._reader.is_alive() and time.monotonic() < deadline:
            runtime._reader.join(.05)
        runtime.close()


def arguments(source, goal=(.2, 0.)):
    return dict(goal_xy_m=list(goal), map_epoch="map-epoch", map_sha256=source.grid.sha256)


def test_route_uses_mobile_control_and_retains_synthetic_verdict(navigation):
    runtime, source = navigation
    result = runtime.execute("go_to", arguments(source))
    assert result.get("software_complete"), result
    assert result["execution_ok"] and not result["ok"] and not result["physical_admission"]
    assert result["commands"] and all(c["execution_ok"] for c in result["commands"])
    assert abs(result["final_pose"]["translation_m"][0]-.2) < .015
    assert any(q["end_xy_m"] != q["start_xy_m"] for q in source.queries)
    assert all(q["radius_m"] > .4 and q["z_min_m"] < .1 and q["z_max_m"] > .4 for q in source.queries)
    assert not runtime.execute("walk_distance", {"distance_m": .1})["ok"]
    assert not runtime.execute("task_done", {"success": True, "summary": "software only"})["success"]
    trace = [json.loads(line) for line in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    assert trace[-1]["skill"] == "go_to" and not trace[-1]["result"]["ok"]


@pytest.mark.parametrize("fault", ["epoch", "frame", "uncertainty", "stale", "future", "unknown", "collision"])
def test_invalid_observation_or_volume_never_dispatches(navigation, monkeypatch, fault):
    runtime, source = navigation
    sent = []
    monkeypatch.setattr(source.raw, "command_velocity", lambda *a, **k: sent.append(a))
    if fault in {"unknown", "collision"}:
        source.clearance = None if fault == "unknown" else 0.
    else:
        def mutate(s):
            if fault == "epoch":
                return replace(s, pose=replace(s.pose, stamp=replace(s.pose.stamp, epoch="new-map")))
            if fault == "frame":
                return replace(s, pose=replace(s.pose, child="optical-camera"))
            if fault == "uncertainty":
                return replace(s, pose=replace(s.pose, position_error_m=None))
            return replace(s, received_monotonic_s=s.received_monotonic_s + (1. if fault == "future" else -1.))
        source.mutate = mutate
    result = runtime.execute("go_to", arguments(source))
    assert not result["execution_ok"] and not sent
    assert runtime._latched and runtime.unverified_actions()
    assert not runtime.execute("task_done", {"success": True, "summary": "invalid"})["success"]


def test_stop_interrupts_blocked_source_and_quarantines_it(navigation):
    runtime, source = navigation
    source.hold, source.entered = threading.Event(), threading.Event()
    result = {}
    worker = threading.Thread(target=lambda: result.update(runtime.execute("go_to", arguments(source))))
    worker.start()
    assert source.entered.wait(1.)
    try:
        assert runtime.execute("stop_navigation", {})["latched"]
        assert not runtime.reset_stop()["ok"]
        assert not runtime.execute("task_done", {"success": True, "summary": "in flight"})["success"]
        worker.join(1.)
        assert not worker.is_alive() and result["source_quarantined"]
        assert not result["execution_ok"] and not result["commands"]
        assert not runtime.close()["complete"]
    finally:
        source.hold.set()
        worker.join(1.)


def test_source_deadline_stops_without_waiting_for_reader(navigation):
    runtime, source = navigation
    source.hold, source.entered = threading.Event(), threading.Event()
    started = time.monotonic()
    result = runtime.execute("go_to", arguments(source))
    assert time.monotonic()-started < 1.
    assert result["source_quarantined"] and not result["execution_ok"]
    assert not runtime.reset_stop()["ok"]
    assert source.raw.get_state().latched


def test_stop_during_mobile_preflight_prevents_late_dispatch(navigation, monkeypatch):
    runtime, source = navigation
    entered, release = threading.Event(), threading.Event()
    class SlowChecker:
        def begin(self, *args):
            entered.set()
            release.wait(1.)
            return "token"
        def finish(self, *args):
            return {"status": "unverified", "reason": "synthetic"}
        def close(self):
            pass
    runtime.mobile.checkers[runtime.settings["base"]] = SlowChecker()
    sent = []
    monkeypatch.setattr(source.raw, "command_velocity", lambda *a, **k: sent.append(a))
    result = {}
    worker = threading.Thread(target=lambda: result.update(runtime.execute("go_to", arguments(source))))
    worker.start()
    try:
        assert entered.wait(1.)
        runtime.stop()
        assert not runtime.reset_stop()["ok"]
    finally:
        release.set()
        worker.join(2.)
    assert not worker.is_alive() and not sent and not result["execution_ok"]


def test_invalid_goal_does_not_leak_route_ownership(navigation):
    runtime, source = navigation
    assert not runtime.execute("go_to", arguments(source, (float("nan"), 0.)))["ok"]
    assert not runtime._active
    assert runtime.execute("go_to", arguments(source, (0., 0.))).get("software_complete")


def test_base_guard_rechecks_after_feedback_before_send(navigation, monkeypatch):
    runtime, source = navigation
    admitted, sent = threading.Event(), []
    admitted.set()
    base = runtime.base_rig.primary
    original = base.get_state
    reads = []
    def state():
        value = original()
        reads.append(value)
        if len(reads) >= 2:
            admitted.clear()
        return value
    monkeypatch.setattr(base, "get_state", state)
    monkeypatch.setattr(source.raw, "command_velocity", lambda *a, **k: sent.append(a))
    result = runtime.mobile.execute("walk_distance", {"distance_m": .1}, admission_check=admitted.is_set)
    assert not result["execution_ok"] and not sent
    assert "authority" in result["error"]


@pytest.mark.parametrize("motion", ["translation", "roll", "tilt"])
def test_arrival_requires_full_orientation_and_speed_rest(navigation, motion):
    runtime, source = navigation
    runtime.settings["wall_timeout_s"] = .3
    def move(sample):
        if motion == "translation":
            pose = replace(sample.pose, translation_m=(sample.pose.stamp.time_s*.02, 0., .2))
        else:
            angle = .8 if motion == "tilt" else sample.pose.stamp.time_s*.1
            pose = replace(sample.pose, rotation_wxyz=(math.cos(angle/2), math.sin(angle/2), 0., 0.))
        return replace(sample, pose=pose)
    source.mutate = move
    result = runtime.execute("go_to", arguments(source, (0., 0.)))
    assert not result.get("software_complete") and not result["execution_ok"]
    assert not result["commands"]


def test_authority_revoked_after_ack_delivers_its_own_stop(navigation, monkeypatch):
    runtime, source = navigation
    allowed = threading.Event()
    allowed.set()
    original = source.raw.command_velocity
    def command(*args, **kwargs):
        ack = original(*args, **kwargs)
        allowed.clear()
        return ack
    monkeypatch.setattr(source.raw, "command_velocity", command)
    result = runtime.mobile.execute("walk_distance", {"distance_m": .1}, admission_check=allowed.is_set)
    assert not result["execution_ok"]
    assert source.raw.get_state().latched and source.raw._command is None


def test_composed_navigation_binding_preserves_resource_and_stop_owner(navigation, tmp_path):
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import Cfg, load_robot_config
    original, _ = navigation
    cfg = load_robot_config("mixed_mock").as_dict()
    cfg["domains"] = {"locomotion": cfg["domains"]["locomotion"]}
    cfg["domains"]["locomotion"]["resolved"] = original.cfg.as_dict()
    source = SyntheticSource()
    runtime, _ = build_robot_runtime(Cfg(cfg), tmp_path / "composed",
        navigation_bindings={"locomotion": {"source": source, "settings": original.settings}})
    try:
        source.raw = runtime.domains["locomotion"].runtime.base_rig.primary.raw
        runtime.execute("locomotion.get_base_state", {})
        assert "locomotion.go_to" in runtime.motion_skills
        assert "locomotion.walk_distance" not in runtime.tool_descriptors
        result = runtime.execute("locomotion.go_to", arguments(source, (0., 0.)))
        assert result["software_complete"] and not result["ok"]
        assert not runtime.execute("task_done", {"success": True, "summary": "synthetic"})["success"]
        assert runtime.stop()["latched"]
    finally:
        runtime.close()


@pytest.mark.parametrize("fault", ["map_epoch", "volume", "stale"])
def test_changed_observation_during_travel_stops_the_base(navigation, monkeypatch, fault):
    runtime, source = navigation
    started = threading.Event()
    original = source.raw.command_velocity
    def command(*args, **kwargs):
        ack = original(*args, **kwargs)
        started.set()
        return ack
    monkeypatch.setattr(source.raw, "command_velocity", command)
    def change(sample):
        if not started.is_set():
            return sample
        if fault == "map_epoch":
            return replace(sample, pose=replace(sample.pose, stamp=replace(sample.pose.stamp, epoch="relocalized")))
        if fault == "volume":
            return replace(sample, volume_sha256="e"*64)
        return replace(sample, producer_age_s=1.)
    source.mutate = change
    result = runtime.execute("go_to", arguments(source))
    assert started.is_set() and not result["execution_ok"]
    assert source.raw.get_state().latched and source.raw._command is None


def test_delayed_reset_cannot_clear_a_newer_route_stop(navigation, monkeypatch):
    runtime, _ = navigation
    entered, release = threading.Event(), threading.Event()
    def reset():
        entered.set()
        release.wait(1.)
        return {"ok": True}
    monkeypatch.setattr(runtime.mobile, "reset_stop", reset)
    result = {}
    worker = threading.Thread(target=lambda: result.update(runtime.reset_stop()))
    worker.start()
    try:
        assert entered.wait(1.)
        runtime.stop()
    finally:
        release.set()
        worker.join(1.)
    assert not worker.is_alive() and not result["ok"] and runtime._latched
