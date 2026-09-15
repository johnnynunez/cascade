"""Current images, detector estimates and configured zones retain their provenance."""
import base64
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.apps.mcp_server import McpSkillServer
from cascade.config import Cfg
from cascade.skills.runtime import SkillRuntime, _jpeg
from cascade.types import Frame, SkillError
from test_reset_capture_freshness import camera, packet, runtime, offline_cpu


def scene_runtime(monkeypatch):
    cam, current, requests = camera(monkeypatch)
    rt = runtime(cam)
    rt.cfg = Cfg({"grasp": {
        "drop_zone": [.14, -.27], "drop_zone_name": "green square",
        "open_box": {"center_xy_m": [.3, -.14], "release_height_m": .104},
    }})
    rt._object_pose = lambda *_: pytest.fail("Scene observations cannot query physics truth")
    return rt, current, requests


def payload(response):
    return json.loads(next(block["text"] for block in response["content"] if block["type"] == "text"))


@pytest.mark.parametrize("name", ["describe_scene", "get_observation"])
def test_mcp_attaches_the_analyzed_frame_without_a_second_capture(monkeypatch, name):
    rt, _, requests = scene_runtime(monkeypatch)
    seen, analyzed = [], []
    analyze = rt._describe_observation
    def describe(frame):
        seen.append(frame)
        observation = analyze(frame)
        analyzed.append(observation)
        return observation
    rt._describe_observation = describe
    rt.execute = lambda name, args: {**getattr(rt, "skill_" + name)(**args), "ok": True}
    server = McpSkillServer()
    server._runtime = rt

    result = server.call_tool(name, {})

    assert not result["isError"] and len(requests) == 1 and len(seen) == 1
    image = next(block for block in result["content"] if block["type"] == "image")
    assert base64.b64decode(image["data"]) == _jpeg(seen[0].rgb)
    data = payload(result)
    assert data["observation_frame"]["frame_id"] == seen[0].frame_id
    assert data["observation_frame"]["age_clock"] == "client_monotonic"
    assert data["robot"]["live_arm_feedback"] is False
    assert "holding" not in data["robot"]
    assert data["robot"]["gripper_contents"] == "not_measured"
    assert analyzed[0]["objects_visible"][0]["label"] == "sponge"
    assert any(row["label"] == "sponge" for row in rt.beliefs.summary())
    assert "sponge" not in json.dumps(data)
    assert "attached image" in data["observation_note"]
    assert "localize_object" in data["observation_note"]
    assert "does not verify the arm's home pose or gripper contents" in data["observation_note"]
    assert "an image does not verify reachability" in data["observation_note"]
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize("name", ["describe_scene", "get_observation"])
def test_mcp_image_metadata_does_not_promote_detector_estimates_or_mutate_runtime_data(name):
    frame = Frame(np.zeros((8, 8, 3), np.uint8), None, np.eye(3))
    raw = {
        "ok": True,
        "objects_visible": [{"label": "unsupported detector identity", "color": "estimated color"}],
        "objects_tracked": [{"label": "remembered identity"}],
        "objects": [{"label": "duplicate identity"}],
        "description": "Repeated detector prose",
        "holding": None,
        "robot": {"status": "standby", "live_arm_feedback": False, "holding": None},
        "configured_zones": [{"name": "delivery area", "source": "configuration",
                              "position_xy_m": [.14, -.27], "visibility": "not_established"}],
        "observation_note": "Raw estimate diagnostics",
        "observation_frame": {"frame_id": -1},
    }
    before = deepcopy(raw)
    rt = SimpleNamespace(last_frame=frame, execute=lambda *a: raw)
    server = McpSkillServer()
    server._runtime = rt

    result = server.call_tool(name, {})

    data = payload(result)
    assert not result["isError"] and data["ok"]
    assert set(data) == {"ok", "observation_frame", "configured_zones", "robot", "observation_note"}
    assert data["configured_zones"] == raw["configured_zones"]
    assert data["robot"] == {
        "live_arm_feedback": False, "gripper_contents": "not_measured",
        "home_pose": "not_verified", "pose_status": "not_measured",
    }
    assert data["observation_frame"]["frame_id"] == frame.frame_id
    assert "unsupported detector identity" not in json.dumps(data)
    assert "estimated color" not in json.dumps(data)
    assert raw == before
    image = next(block for block in result["content"] if block["type"] == "image")
    assert base64.b64decode(image["data"]) == _jpeg(frame.rgb)


@pytest.mark.parametrize("held", [None, "tracked object"])
@pytest.mark.parametrize("connected", [False, True])
def test_world_state_reports_session_tracking_without_opening_or_reading_arm(monkeypatch, held, connected):
    from cascade.apps.demo import _runtime_state

    rt, _, _ = scene_runtime(monkeypatch)
    rt.held_object, rt.current_task = held, None
    rt.stream_server = None
    rt.grasp_memory = SimpleNamespace(summary=lambda: "")
    rt.backends = lambda: {}
    rt.rig = SimpleNamespace(stats=lambda: {"worktop": {"frame_id": 123}})
    rt.arm.raw.connected = connected
    def forbidden(*_a, **_kw):
        pytest.fail("world_state opened control or read an arm sensor")
    rt.arm.get_state = forbidden
    rt.arm.raw.get_state = forbidden
    rt.arm.raw.connect = forbidden
    rt.arm.raw.open = forbidden
    rt.arm.move_joints = forbidden
    server = McpSkillServer()
    server._runtime = rt

    result = server.call_tool("world_state", {})

    data = payload(result)
    assert not result["isError"] and data["ok"]
    assert "holding" not in data
    assert data["live_arm_feedback"] is False
    assert data["gripper_contents"] == "not_measured"
    assert data["home_pose"] == "not_verified"
    assert data["session_state"]["source"] == "mcp_runtime"
    assert data["session_state"]["arm_connected"] is connected
    assert "arm_connected" not in data
    assert data["simulator_status"] == "not_queried"
    assert data["cameras"] == {"worktop": {"frame_id": 123}}
    if held is None:
        assert "tracked_holding" not in data
    else:
        assert data["tracked_holding"] == {"label": held, "source": "session_state"}
    assert rt.held_object == held
    assert _runtime_state(rt)["holding"] == held
    assert _runtime_state(rt)["arm_connected"] is connected


def test_world_state_preserves_cached_zero_counters_without_claiming_service_health(monkeypatch):
    from cascade.apps.demo import _runtime_state

    rt, _, _ = scene_runtime(monkeypatch)
    counters = {"ticks": 0, "paused": False, "last_update_s_ago": None}
    cameras = {"worktop": {"fps": 2.1, "frame_id": 8, "depth": True, "error": None},
               "kitchen": {"fps": 0.0, "frame_id": 7, "depth": True, "error": None}}
    rt.held_object, rt.current_task, rt.stream_server = None, None, None
    rt.grasp_memory = SimpleNamespace(summary=lambda: "")
    rt.backends = lambda: {}
    rt.watcher = SimpleNamespace(stats=lambda: counters)
    rt.camera = SimpleNamespace(overlay=lambda: ([], "starting..."))
    rt.rig = SimpleNamespace(stats=lambda: cameras)
    def forbidden(*_a, **_kw):
        pytest.fail("world_state performed a new sensor or control operation")
    rt.observe = rt.observe_fresh = forbidden
    rt.camera.get_frame = forbidden
    rt.arm.get_state = rt.arm.raw.get_state = forbidden
    rt.arm.raw.connect = rt.arm.raw.open = rt.arm.move_joints = forbidden
    before = deepcopy(_runtime_state(rt))
    server = McpSkillServer()
    server._runtime = rt

    data = payload(server.call_tool("world_state", {}))

    assert data["session_state"] == {
        "source": "mcp_runtime", "agent_status": "starting...",
        "arm_connected": False, "perception": counters,
    }
    assert not {"agent_status", "arm_connected", "perception"} & data.keys()
    assert data["simulator_status"] == "not_queried"
    assert data["cameras"] == cameras
    assert "client deliveries" in data["camera_statistics_note"]
    assert "capture cadence" in data["camera_statistics_note"]
    assert _runtime_state(rt) == before


def test_saved_verification_keeps_unverified_belief_without_another_sensor_check():
    from cascade.agent.effects import Postcondition, PostconditionChecker, UNVERIFIED

    def forbidden(*_a, **_kw):
        pytest.fail("Reading saved verification performed a fresh observation")
    effects = PostconditionChecker(object_pose=forbidden, belief_pose=forbidden,
                                   gripper_frac=forbidden, reobserve=forbidden)
    recorded = Postcondition(skill="pick_and_place", kind="relocated", status=UNVERIFIED,
                             channel="belief", evidence="Only the skill's own tracked drop point")
    effects.history.append(recorded)
    effects.verify = forbidden
    rt = SimpleNamespace(effects=effects, observe=forbidden, observe_fresh=forbidden)
    server = McpSkillServer()
    server._runtime = rt

    data = payload(server.call_tool("verify_last_action", {}))

    assert data["ok"] and data["recent"] == [recorded.as_dict()]
    assert data["recent"][0]["status"] == "unverified"
    assert data["recent"][0]["channel"] == "belief"
    assert effects.history == [recorded]


def test_mcp_observation_preserves_measured_telemetry_and_labels_holding_as_tracking():
    frame = Frame(np.zeros((8, 8, 3), np.uint8), None, np.eye(3))
    telemetry = {"q_deg": [1, 2, 3, 4, 5, 6], "tcp_xyz": [.2, .1, .3],
                 "gripper_open_frac": .4, "holding": "tracked object"}
    raw = {"ok": True, "robot": deepcopy(telemetry)}
    rt = SimpleNamespace(last_frame=frame, execute=lambda *_a: raw)
    server = McpSkillServer()
    server._runtime = rt

    result = server.call_tool("get_observation", {})

    robot = payload(result)["robot"]
    assert not result["isError"]
    for key in ("q_deg", "tcp_xyz", "gripper_open_frac"):
        assert robot[key] == telemetry[key]
    assert "holding" not in robot
    assert robot["tracked_holding"] == {"label": "tracked object", "source": "session_state"}
    assert robot["gripper_contents"] == "not_measured"
    assert robot["home_pose"] == "not_verified"
    assert raw["robot"] == telemetry


def test_description_refreshes_empty_beliefs_and_keeps_history_out_of_visible_objects(monkeypatch):
    rt, _, requests = scene_runtime(monkeypatch)
    rt.beliefs.update("cup", [.4, .2, .05], .8, color="blue", t=time.monotonic() - 10)

    result = rt.skill_describe_scene()

    assert len(requests) == 1
    assert result["objects"] == result["objects_visible"]
    assert all(row["label"] != "cup" for row in result["objects"])
    assert "cup" not in result["description"]
    assert result["description"].startswith("Unconfirmed detector estimates:")
    remembered = next(row for row in result["objects_tracked"] if row["label"] == "cup")
    assert remembered["state"] == "remembered"
    assert result["configured_zones"][0] == {
        "name": "green square", "kind": "delivery_area", "source": "configuration",
        "position_xy_m": [.14, -.27], "visibility": "not_established",
    }
    json.dumps(result, allow_nan=False)


def test_no_detections_is_not_an_empty_scene_claim(monkeypatch):
    rt, _, _ = scene_runtime(monkeypatch)
    rt.detector.detect = lambda *_, **__: []
    result = rt.skill_describe_scene()
    assert result["objects"] == []
    assert "does not establish an empty scene" in result["description"]
    assert rt.last_frame is not None


@pytest.mark.parametrize("name", ["describe_scene", "get_observation"])
def test_repeated_producer_capture_is_rejected_before_detection(monkeypatch, name):
    from cascade.perception import freshness
    rt, _, requests = scene_runtime(monkeypatch)
    rt.last_frame = rt.camera.get_frame()
    rt.detector.detect = lambda *_, **__: pytest.fail("Repeated capture reached the detector")
    original = freshness.read_frame_after
    monkeypatch.setattr(freshness, "read_frame_after",
                        lambda camera, **kw: original(camera, after=kw["after"], timeout_s=.03))

    with pytest.raises(SkillError, match="fresh frame"):
        getattr(rt, "skill_" + name)()
    assert len(requests) > 1


def test_snapshot_waits_for_a_new_producer_capture_without_cross_host_clock_comparison(monkeypatch):
    rt, current, requests = scene_runtime(monkeypatch)
    current[0] = packet(capture_t=10.)
    rt.last_frame = rt.camera.get_frame()
    current[0] = packet(capture_t=11.)
    server = McpSkillServer()

    result = server._camera_snapshot(rt)

    assert not result["isError"] and len(requests) == 2
    assert 0 <= payload(result)["frame_age_s"] < 1
    assert payload(result)["age_clock"] == "client_monotonic"
    assert "does not verify the arm's home pose or gripper contents" in payload(result)["observation_note"]
    assert "an image does not verify reachability" in payload(result)["observation_note"]


@pytest.mark.parametrize("invalid_time", [float("nan"), float("inf"), -100., None])
@pytest.mark.parametrize("named", [False, True])
def test_snapshot_rejects_stale_or_invalid_frames_without_an_error_flag(invalid_time, named):
    frame = Frame(np.zeros((8, 8, 3), np.uint8), None, np.eye(3))
    frame.t = invalid_time
    stream = SimpleNamespace(latest=lambda: frame, last_error=None,
                             get_fresh_frame=lambda **_: frame)
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.camera, rt.last_frame = stream, None
    rt.depth = SimpleNamespace(ensure_depth=lambda value: value)
    rt.rig = SimpleNamespace(get=lambda name: stream, primary=SimpleNamespace(name="worktop"))
    result = McpSkillServer()._camera_snapshot(rt, "worktop" if named else None)
    assert result["isError"] and payload(result)["ok"] is False
    assert not any(block["type"] == "image" for block in result["content"])


def test_named_snapshot_refuses_a_silently_frozen_producer():
    frame = Frame(np.zeros((8, 8, 3), np.uint8), None, np.eye(3))
    requested = []
    def frozen(**kwargs):
        requested.append(kwargs)
        raise TimeoutError("synthetic producer stayed frozen")
    stream = SimpleNamespace(latest=lambda: frame, last_error=None, get_fresh_frame=frozen)
    rt = SimpleNamespace(rig=SimpleNamespace(get=lambda name: stream))
    result = McpSkillServer()._camera_snapshot(rt, "side")
    assert result["isError"] and "fresh frame" in payload(result)["error"]
    assert requested == [{"after": frame, "timeout_s": 5.0}]


def test_failed_observation_cannot_attach_an_earlier_image():
    frame = Frame(np.zeros((8, 8, 3), np.uint8), None, np.eye(3))
    rt = SimpleNamespace(last_frame=frame,
                         execute=lambda *a: {"ok": False, "error": "camera unavailable"})
    server = McpSkillServer()
    server._runtime = rt
    result = server.call_tool("describe_scene", {})
    assert result["isError"]
    assert not any(block["type"] == "image" for block in result["content"])


def test_image_that_ages_during_analysis_is_not_reported_as_current():
    frame = Frame(np.zeros((8, 8, 3), np.uint8), None, np.eye(3), t=time.monotonic() - 6)
    rt = SimpleNamespace(last_frame=frame, execute=lambda *a: {"ok": True, "objects_visible": []})
    server = McpSkillServer()
    server._runtime = rt
    result = server.call_tool("get_observation", {})
    assert result["isError"] and "old" in payload(result)["error"]
    assert not any(block["type"] == "image" for block in result["content"])


@pytest.mark.parametrize("failure", ["timeout", "stale", "invalid"])
@pytest.mark.parametrize("with_history", [True, False])
def test_task_memory_never_labels_cached_pixels_as_current(failure, with_history):
    historical = _jpeg(np.full((8, 8, 3), 30, np.uint8))
    cached = _jpeg(np.full((8, 8, 3), 90, np.uint8))
    frame = Frame(np.full((8, 8, 3), 170, np.uint8), None, np.eye(3))
    if failure == "stale":
        frame.t = time.monotonic() - 6
    elif failure == "invalid":
        frame.t = float("nan")
    history = [{"jpeg": historical, "step": 1, "age_s": 10.,
                "text": "previous action", "verdict": "refuted"}] if with_history else []
    def fresh():
        if failure == "timeout":
            raise TimeoutError("Camera did not provide a fresh capture")
        return frame
    memory = SimpleNamespace(memory_frames=lambda k: history[:k],
                             frame_caption=lambda row: "memory frame: previous action (refuted)")
    rt = SimpleNamespace(memory=memory, observe=lambda: frame, observe_fresh=fresh,
                         frame_jpeg=lambda: cached)
    server = McpSkillServer()
    server._runtime = rt

    result = server.call_tool("task_memory", {})

    assert not result["isError"]
    images = [base64.b64decode(c["data"]) for c in result["content"] if c["type"] == "image"]
    assert images == ([historical] if with_history else [])
    assert cached not in images
    texts = [c["text"] for c in result["content"] if c["type"] == "text"]
    assert "current view (now)" not in texts
    summary = json.loads(texts[-1])
    assert summary["frames"] == len(history)
    assert summary["current_view"]["available"] is False
    assert summary["current_view"]["error"]
    assert "unavailable" in summary["note"].lower()
    assert "only the current view" not in summary["note"]
    if with_history:
        assert summary["steps_recorded"][0]["verdict"] == "refuted"


def test_task_memory_attaches_the_fresh_frame_and_its_provenance():
    frame = Frame(np.full((8, 8, 3), 170, np.uint8), None, np.eye(3))
    calls = []
    def fresh():
        calls.append("fresh")
        return frame
    def cached():
        pytest.fail("Task memory requested cached camera bytes")
    memory = SimpleNamespace(memory_frames=lambda k: [])
    rt = SimpleNamespace(memory=memory, observe=cached, observe_fresh=fresh, frame_jpeg=cached)
    server = McpSkillServer()
    server._runtime = rt

    result = server.call_tool("task_memory", {})

    assert not result["isError"] and calls == ["fresh"]
    image = next(c for c in result["content"] if c["type"] == "image")
    assert base64.b64decode(image["data"]) == _jpeg(frame.rgb)
    texts = [c["text"] for c in result["content"] if c["type"] == "text"]
    assert texts[0] == "current view (now)"
    summary = json.loads(texts[-1])
    assert summary["current_view"]["available"] is True
    assert summary["current_view"]["age_clock"] == "client_monotonic"
    assert summary["current_view"]["frame_id"] == frame.frame_id
    assert 0 <= summary["current_view"]["frame_age_s"] < 1
