"""Network delivery counters must not turn one capture into repeated perception."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.types import Frame


def frame(delivery, stamp, *, camera="cam0", robot="/robot", source="simulator", local=False):
    result = Frame(rgb=np.zeros((8, 8, 3), dtype=np.uint8), depth_m=None, K=np.eye(3),
                   frame_id=delivery, t=1000+delivery)
    if not local:
        result.capture = {"backend": "isaac", "source": (source, 8611), "camera": camera, "t": stamp,
                          "proprioception": {"backend": "isaac", "robot_id": robot, "t": stamp,
                                             "time_source": "physics_loop_monotonic"}}
    return result


def watcher():
    stream = SimpleNamespace(name="worktop", latest=Mock(), set_overlay=Mock())
    depth = SimpleNamespace(ensure_depth=Mock(side_effect=lambda frame: frame))
    detector = SimpleNamespace(detect=Mock(return_value=[]))
    harness = SimpleNamespace(heartbeat=Mock())
    cam = WatchedCamera(stream, depth, None, fuse=False)
    watch = WorldWatcher([cam], detector, SimpleNamespace(), harness=harness)
    return watch, cam, detector, harness


def test_repeated_capture_does_not_repeat_depth_detection_or_safety_heartbeat():
    watch, cam, detector, harness = watcher()
    for delivery in range(1, 31):
        cam.stream.latest.return_value = frame(delivery, 42 if delivery < 21 else 42.5)
        watch._tick(cam)
    assert cam.last_seq == 30
    assert watch.ticks == 2
    assert cam.depth.ensure_depth.call_count == 2
    assert detector.detect.call_count == 2
    assert harness.heartbeat.call_count == 2
    assert cam.stream.set_overlay.call_count == 2


@pytest.mark.parametrize("identity", ["camera", "robot", "source"])
def test_equal_timestamps_from_different_producers_are_not_deduplicated(identity):
    watch, cam, detector, _ = watcher()
    for delivery, changes in [(1, {}), (2, {identity: "another"})]:
        cam.stream.latest.return_value = frame(delivery, 42, **changes)
        watch._tick(cam)
    assert detector.detect.call_count == 2


def test_local_camera_keeps_delivery_based_detection():
    watch, cam, detector, harness = watcher()
    for delivery in [1, 1, 2]:
        cam.stream.latest.return_value = frame(delivery, None, local=True)
        watch._tick(cam)
    assert detector.detect.call_count == harness.heartbeat.call_count == 2


def test_invalid_producer_metadata_cannot_create_a_perception_heartbeat():
    watch, cam, detector, harness = watcher()
    value = frame(1, 42)
    value.capture["proprioception"]["t"] = 43
    cam.stream.latest.return_value = value
    with pytest.raises(ValueError, match="producer capture"):
        watch._tick(cam)
    detector.detect.assert_not_called()
    harness.heartbeat.assert_not_called()
