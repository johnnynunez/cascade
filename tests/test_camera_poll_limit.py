"""Kitchen polling is bounded without changing other rigs or camera calibration."""
import pytest

from cascade.apps.demo import _camera_cfgs
from cascade.config import Cfg, load_demo_config
from cascade.perception.stream import CameraStream


def test_kitchen_bounds_all_three_pumps_and_preserves_camera_identity():
    names = ["isaac", "isaac_side", "isaac_proof"]
    cfg = load_demo_config(arm="isaac_kitchen_gpu", cameras=names)
    before = cfg.as_dict()
    tuned = _camera_cfgs(cfg)
    assert len(tuned) == 3
    for original, camera in zip(before["cameras"], tuned):
        profile = camera.as_dict()
        assert profile.pop("fps") == 4
        assert profile == {key: value for key, value in original.items() if key != "fps"}
        # The rate reaches the actual pump without opening a camera or arm.
        stream = CameraStream(None, rate_hz=camera.fps)
        assert stream._period == .25
    assert cfg.as_dict() == before


def test_poll_limit_does_not_speed_up_a_slower_camera():
    cfg = Cfg({"cameras": [{"fps": 15}, {"fps": 2}],
               "perception_loop": {"max_camera_poll_hz": 4}})
    assert [camera.fps for camera in _camera_cfgs(cfg)] == [4, 2]


def test_non_kitchen_rigs_keep_their_selected_camera_rates():
    cfg = load_demo_config(arm="isaac", cameras=["isaac", "isaac_side"])
    assert [camera.fps for camera in _camera_cfgs(cfg)] == [15, 15]
    single = Cfg({"camera": {"fps": 7}})
    assert _camera_cfgs(single)[0].fps == 7


@pytest.mark.parametrize("limit", [0, -1, .5, float("inf"), float("nan")])
def test_invalid_poll_limit_cannot_create_an_unbounded_pump(limit):
    cfg = Cfg({"camera": {"fps": 15}, "perception_loop": {"max_camera_poll_hz": limit}})
    with pytest.raises(ValueError, match="max_camera_poll_hz"):
        _camera_cfgs(cfg)
