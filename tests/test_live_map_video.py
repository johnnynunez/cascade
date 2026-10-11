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


# ── shared rendering for the recorder and the live view's map tile ──────────


def _reference_splat(points, view, colors, point_px):
    """The original per-point painter (far to near, nearest wins), kept here
    as the reference the vectorized splatter must reproduce exactly."""
    w, h = view.size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = lmv.BACKGROUND
    u, v, z = view.project(points)
    ok = (z > 0.02) & np.isfinite(u) & np.isfinite(v)
    half = point_px // 2
    ok &= (u >= -half) & (u < w + half) & (v >= -half) & (v < h + half)
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(-z[idx], kind="stable")]
    for i, x, y in zip(idx, np.round(u[idx]).astype(int), np.round(v[idx]).astype(int)):
        img[max(0, y - half):max(0, y + half + 1), max(0, x - half):max(0, x + half + 1)] = colors[i]
    return img


@pytest.mark.parametrize("point_px", [1, 3, 4])
def test_the_vectorized_splatter_paints_exactly_what_the_per_point_painter_did(point_px):
    """The live view renders in the robot's own process: a per-point Python
    loop there holds the GIL the 50 Hz motion stream needs. The numpy
    splatter must still draw the same picture, pixel for pixel."""
    rng = np.random.default_rng(7)
    pts = rng.uniform([-0.2, -0.6, -0.1], [0.9, 0.6, 0.6], (3000, 3))
    cols = rng.integers(0, 255, (3000, 3), dtype=np.uint8)
    view = lmv.orbit_view(3.0, size=(320, 180))
    got = lmv.render_points(pts, view, cols, point_px=point_px)
    assert np.array_equal(got, _reference_splat(pts, view, cols, point_px))


def test_top_down_points_outside_its_extent_are_dropped_not_smeared():
    """A voxel beyond the drawn extent (x > 0.75 m: negative pixel rows) used
    to become a negative slice end and paint most of the panel."""
    far = np.array([[1.5, 0.0, 0.1], [0.3, 2.0, 0.1], [-1.0, 0.0, 0.1]])
    empty = lmv.topdown(np.zeros((0, 3)), [0.1, -0.3, 0.0], [0.5, 0.3, 0.5], size=(320, 180))
    img = lmv.topdown(far, [0.1, -0.3, 0.0], [0.5, 0.3, 0.5], size=(320, 180))
    assert np.array_equal(img, empty)


def test_top_down_shows_the_highest_voxel_of_a_column():
    lo, hi = [0.1, -0.3, 0.0], [0.5, 0.3, 0.5]
    low, high = np.array([0.3, 0.0, 0.0]), np.array([0.3, 0.0, 0.3])
    a = lmv.topdown(np.array([low, high]), lo, hi, size=(320, 180))
    b = lmv.topdown(np.array([high, low]), lo, hi, size=(320, 180))
    only_high = lmv.topdown(np.array([high]), lo, hi, size=(320, 180))
    assert np.array_equal(a, only_high) and np.array_equal(b, only_high)


def test_map_panels_are_shared_by_the_recorder_and_the_live_view():
    """One renderer for both: the 3-D orbit with the workspace and a caption,
    and the top-down height map with the objects."""
    pts = np.array([[0.3, 0.0, 0.05], [0.35, 0.05, 0.1]])
    m3d, td = lmv.map_panels(pts, t_s=1.0, workspace=([0.1, -0.3, -0.01], [0.5, 0.3, 0.55]),
                             objects=[{"label": "cube", "position": [0.3, 0.0, 0.05]}],
                             size=(320, 180), caption="2 occupied voxels", caption_ok=True)
    assert m3d.shape == td.shape == (180, 320, 3)
    assert _drawn(m3d).any() and _drawn(td).any()
    red = lmv.map_panels(pts, t_s=1.0, workspace=([0.1, -0.3, -0.01], [0.5, 0.3, 0.55]),
                         size=(320, 180), caption="map: STALE 9.0 s", caption_ok=False)[0]
    assert ((red[..., 0] > 200) & (red[..., 1] < 120)).any()   # the stale caption is red
