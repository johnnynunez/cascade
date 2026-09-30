"""Reset recovery uses real map/barrier logic with synthetic camera/service I/O."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.perception.occupancy import OccupancyError
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.skills.runtime import SkillError, SkillRuntime
from cascade.types import SafetyViolation
from test_occupancy_payload import frame as payload_frame, mapping
from test_reset_capture_freshness import offline_cpu, runtime


def recovery_case(*, watched):
    occupancy = mapping()
    occupancy.refresh(payload_frame(False, 1.), np.eye(4))
    events = []
    state = {"stamp": 1., "frozen": "side" if watched else "main"}

    def stream(name):
        def fresh(*, after=None, timeout_s=5):
            if after is not None and state["frozen"] == name:
                raise TimeoutError(f"{name} capture stalled")
            state["stamp"] += 1
            frame = payload_frame(False, state["stamp"])
            frame.T_base_cam = np.eye(4)
            frame.capture["camera"] = name
            events.append(("capture", name, state["stamp"]))
            return frame

        return SimpleNamespace(name=name, get_fresh_frame=fresh)

    main = stream("main")
    rt = runtime(main)
    rt.detector = SimpleNamespace(detect=lambda *a, **kw: [])
    rt.cfg = Cfg({"arm": {"home_q": [0.] * 6}})
    rt.arm.harness.occupancy = occupancy
    rt.arm.set_gripper = lambda *a, **kw: events.append(("release",))
    reset_props = rt.arm.raw.reset_props

    def reset():
        events.append(("reset_props",))
        return reset_props()

    def move(*a, **kw):
        # A requested home must already have usable geometry. The actual map
        # gate also rejects pending reset faults, before any synthetic motion.
        clearance = occupancy.clearance(np.zeros((1, 3)))
        assert clearance is not None and not occupancy.scene_reset_pending
        assert not occupancy.is_stale() and occupancy.last_error is None
        events.append(("home", set(occupancy._depth_history)))
        return True

    rt.arm.raw.reset_props = reset
    rt.arm.move_joints = move
    rt.skill_move_home = SkillRuntime.skill_move_home.__get__(rt)
    if watched:
        cameras = [WatchedCamera(main, rt.depth, None),
                   WatchedCamera(stream("side"), rt.depth, None, fuse=False, map_depth=True)]
        rt.watcher = WorldWatcher(cameras, rt.detector, rt.beliefs, occupancy=occupancy)

    def reset_scene():
        with rt.watcher.paused() if rt.watcher else nullcontext():
            return rt.skill_reset_scene()

    return rt, occupancy, events, state, reset_scene


@pytest.mark.parametrize("watched", [False, True], ids=["direct", "map-side-camera"])
def test_retry_after_camera_returns_rebuilds_map_before_requesting_home(watched):
    rt, occupancy, events, state, reset = recovery_case(watched=watched)
    first = reset()
    assert not first["ok"] and first["props_reset"] == ["green_cube"]
    assert occupancy.scene_reset_pending and occupancy.is_stale()
    with pytest.raises(SafetyViolation, match="post-reset"):
        occupancy.clearance(np.zeros((1, 3)))

    events.clear()
    state["frozen"] = None
    result = reset()

    assert result["ok"] and result["geometry_recovered"], result
    assert result["observation_refreshed"] and not occupancy.scene_reset_pending
    home = next(event for event in events if event[0] == "home")
    assert home[1] == ({"main", "side"} if watched else {"main"})
    home_index = events.index(home)
    assert home_index > 0
    assert all(event[0] == "capture" for event in events[:home_index])
    assert [event[0] for event in events if event[0] != "capture"] == ["home", "reset_props"]
    assert not rt._reset_observation_pending


@pytest.mark.parametrize("watched", [False, True], ids=["direct", "map-side-camera"])
@pytest.mark.parametrize("failure", ["camera", "integration"])
def test_failed_recovery_never_requests_motion_or_drops_held_state(monkeypatch, watched, failure):
    rt, occupancy, events, state, reset = recovery_case(watched=watched)
    assert not reset()["ok"]
    events.clear()
    rt.held_object = "carried cube"
    rt._held_det_label = "cube"
    rt._held_offset = np.array([.01, .02, .03])
    rt.beliefs.update("remembered", [0., 0., .1], .8)
    rt.memory.add("note", "retain this episode on failed recovery")
    digest = rt.memory.digest()
    request = occupancy._client.request
    if failure == "integration":
        state["frozen"] = None

        def fail_integration(packet):
            if packet["action"] == "integrate_depth":
                raise OccupancyError("synthetic integration unavailable")
            return request(packet)

        monkeypatch.setattr(occupancy._client, "request", fail_integration)

    result = reset()

    assert not result["ok"] and result["stage"] == "reset_recovery", result
    assert result["home_skipped"] and result["beliefs_forgotten"] == 0
    assert result["props_reset"] == [] and not result["observation_refreshed"]
    assert all(event[0] == "capture" for event in events)
    assert rt.held_object == "carried cube" and rt._held_det_label == "cube"
    np.testing.assert_array_equal(rt._held_offset, [.01, .02, .03])
    assert len(rt.beliefs.all()) == 1 and rt.memory.digest() == digest
    assert occupancy.scene_reset_pending and rt._reset_observation_pending
    with pytest.raises(SafetyViolation, match="occupancy unsafe"):
        occupancy.clearance(np.zeros((1, 3)))

    # A failed recovery itself is retryable once camera and mapping return.
    state["frozen"] = None
    monkeypatch.setattr(occupancy._client, "request", request)
    recovered = reset()
    assert recovered["ok"] and recovered["geometry_recovered"], recovered
    assert rt.held_object is None and not occupancy.scene_reset_pending


@pytest.mark.parametrize("error", [SkillError("home did not settle"),
                                    SafetyViolation("unobserved payload clearance")])
def test_failed_home_keeps_gripper_props_and_episode_unchanged(error):
    rt, _, events, _, reset = recovery_case(watched=False)
    rt.held_object = "carried cube"
    rt._held_det_label = "cube"
    rt.beliefs.update("remembered", [0., 0., .1], .8)
    rt.memory.add("note", "keep this episode")
    digest = rt.memory.digest()

    def rejected_home():
        events.append(("home_rejected",))
        raise error

    rt.skill_move_home = rejected_home
    result = reset()

    assert not result["ok"] and result["stage"] == "home"
    assert result["home_error"] == str(error) and result["beliefs_forgotten"] == 0
    assert result["props_reset"] == [] and not result["observation_refreshed"]
    assert events == [("home_rejected",)]
    assert rt.held_object == "carried cube" and rt._held_det_label == "cube"
    assert len(rt.beliefs.all()) == 1 and rt.memory.digest() == digest
