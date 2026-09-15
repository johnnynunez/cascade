"""Fresh native observations must not mislabel the entire tracker as memory."""
import time

from test_reset_capture_freshness import camera, packet, runtime, offline_cpu


def test_fresh_observation_reports_measured_color_and_explicit_tracker_states(monkeypatch):
    cam, current, _ = camera(monkeypatch)
    captured = time.monotonic()
    current[0] = packet(capture_t=captured)
    rt = runtime(cam)
    rt.beliefs.update("cup", [.4, .2, .05], .8, color="blue", t=captured - 10)

    observation = rt.skill_get_observation()

    assert observation["objects_visible"][0]["color"] == "green"
    tracked = {row["color"]: row for row in observation["objects_tracked"]}
    assert tracked["green"]["state"] == "visible"
    assert tracked["blue"]["state"] == "remembered"
    assert tracked["blue"]["age_s"] >= 10
    assert "objects_remembered" not in observation
    assert "estimates" in observation["observation_note"]
    # A lazy controller has no measurement of motor power. Looking does not
    # open it or claim that a running simulator's motors are unpowered.
    assert observation["robot"]["live_arm_feedback"] is False
    assert "unpowered" not in observation["robot"]["status"]


def test_color_remains_available_when_the_fresh_frame_has_no_depth(monkeypatch):
    cam, _, _ = camera(monkeypatch)
    rt = runtime(cam)
    frame = cam.get_frame()
    frame.depth_m = None
    observation = rt._describe_observation(frame)
    assert observation["objects_visible"][0]["color"] == "green"
    assert observation["objects_visible"][0]["position"] is None
    assert observation["objects_tracked"] == []
