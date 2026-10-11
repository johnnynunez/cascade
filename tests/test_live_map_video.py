"""The live map recorder (scripts/record_live_map.py) only WATCHES the rig.

It sits next to a running MCP server that owns the arm, the cameras and the
occupancy bridge. Recording must never change any of them: the bridge client
it uses answers `query`/`probe` and refuses everything that writes the map;
cameras come from the live-view server's streams, never from the devices.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from cascade.apps import live_map_video as lmv


def _drawn(img):
    """Pixels the renderer painted (anything but the background)."""
    return (img != np.asarray(lmv.BACKGROUND, np.uint8)).any(axis=2)


class _Wire:
    def __init__(self):
        self.sent = []

    def request(self, packet, timeout_ms=None):
        self.sent.append(packet)
        return {"ok": True, "grid": np.zeros((1, 1, 1), np.float32), "origin": np.zeros(3),
                "voxel": 0.01, "points": np.zeros((0, 3), np.float32)}


@pytest.mark.parametrize("action", ["clear", "integrate_depth", "integrate_masked_depth", "integrate", "reset"])
def test_the_recorder_cannot_write_the_map(action):
    wire = _Wire()
    bridge = lmv.ReadOnlyBridge(wire)
    with pytest.raises(PermissionError):
        bridge.request({"action": action})
    assert wire.sent == []


def test_the_recorder_can_read_the_map():
    wire = _Wire()
    bridge = lmv.ReadOnlyBridge(wire)
    bridge.request({"action": "probe"})
    pts = bridge.occupied_points(np.array([0.0, -0.3, 0.0]), np.array([0.5, 0.3, 0.4]))
    assert [p["action"] for p in wire.sent] == ["probe", "query"]
    assert pts.shape == (0, 3)


def test_a_point_on_the_view_axis_lands_at_the_image_centre():
    view = lmv.View(eye=np.array([1.0, 0.0, 0.5]), target=np.array([0.0, 0.0, 0.5]), size=(640, 360), fov_deg=60)
    img = lmv.render_points(np.array([[0.0, 0.0, 0.5]]), view, point_px=3)
    ys, xs = np.nonzero(_drawn(img))
    assert 0 < ys.size < 50
    assert abs(xs.mean() - 320) <= 1 and abs(ys.mean() - 180) <= 1


def test_up_is_up_and_the_robot_left_is_screen_left_seen_from_the_front():
    """Looking from +x back at the base (the viewer faces the robot), +z
    must render above the centre and +y (the robot's left) on screen RIGHT;
    a mirrored render would teach the audience the wrong side."""
    view = lmv.View(eye=np.array([1.0, 0.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), size=(400, 400), fov_deg=60)
    up = np.nonzero(_drawn(lmv.render_points(np.array([[0.0, 0.0, 0.2]]), view)))
    left = np.nonzero(_drawn(lmv.render_points(np.array([[0.0, 0.2, 0.0]]), view)))
    assert up[0].size and up[0].mean() < 200
    assert left[1].size and left[1].mean() > 200


def test_nearer_points_hide_farther_ones():
    view = lmv.View(eye=np.array([1.0, 0.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), size=(200, 200), fov_deg=60)
    near_low, far_high = np.array([0.5, 0.0, 0.0]), np.array([-0.5, 0.0, 0.0])
    img = lmv.render_points(np.array([far_high, near_low]), view, colors=np.array([[255, 0, 0], [0, 255, 0]], np.uint8),
                            point_px=9)
    assert tuple(img[100, 100]) == (0, 255, 0)


def test_points_behind_the_virtual_camera_are_dropped():
    view = lmv.View(eye=np.array([1.0, 0.0, 0.0]), target=np.array([0.0, 0.0, 0.0]), size=(200, 200), fov_deg=60)
    assert not _drawn(lmv.render_points(np.array([[2.0, 0.0, 0.0]]), view)).any()


def test_mjpeg_frames_are_split_on_content_length():
    jpeg = b"\xff\xd8 not really a jpeg \xff\xd9"
    part = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(jpeg), jpeg)
    frames = list(lmv.iter_mjpeg(io.BytesIO(part * 3)))
    assert frames == [jpeg] * 3


def test_the_composite_keeps_every_panel_and_its_title():
    panels = [np.full((100, 160, 3), v, np.uint8) for v in (10, 60, 110, 160)]
    out = lmv.compose(panels, ["a", "b", "c", "d"], tile=(320, 180))
    assert out.shape == (360, 640, 3)
    for (y, x), v in zip(((120, 100), (120, 420), (300, 100), (300, 420)), (10, 60, 110, 160)):
        assert abs(int(out[y, x, 0]) - v) <= 2


def test_a_missing_camera_is_shown_as_missing_not_as_a_frozen_frame():
    panel = lmv.camera_panel(None, (320, 180), "d455f_wrist")
    assert panel.shape == (180, 320, 3)
    assert panel.mean() < 60   # dark placeholder with a label, never the last good frame
