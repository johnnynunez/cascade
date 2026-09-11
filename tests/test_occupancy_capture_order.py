"""Do not pair old depth with the robot pose after slow inference."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.perception.grounding import Extrinsics


@pytest.mark.parametrize("detector_fails", [False, True])
def test_geometry_refresh_precedes_delayed_semantic_inference(detector_fails):
    pose_epoch = [0]
    fused_epochs = []
    frame = SimpleNamespace(frame_id=1, has_depth=True, T_base_cam=np.eye(4))
    stream = SimpleNamespace(latest=lambda: frame, set_overlay=lambda **kw: None, name="camera")
    cam = WatchedCamera(stream=stream, depth=SimpleNamespace(ensure_depth=lambda f: f), extrinsics=Extrinsics())

    def detect(f, classes=None):
        assert f is frame
        pose_epoch[0] = 1  # the robot moves while a locked/slow detector runs
        if detector_fails:
            raise RuntimeError("detector failed")
        return []

    def refresh(f, transform):
        assert f is frame
        fused_epochs.append(pose_epoch[0])

    watcher = WorldWatcher([cam], SimpleNamespace(detect=detect), beliefs=None,
                           occupancy=SimpleNamespace(refresh=refresh))
    if detector_fails:
        with pytest.raises(RuntimeError, match="detector failed"):
            watcher._tick(cam)
    else:
        watcher._tick(cam)
    assert fused_epochs == [0], "mask depth before inference can age the image relative to the robot"
