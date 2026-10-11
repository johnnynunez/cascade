"""The nvblox map as one more live-view tile (`nvblox_map`, apps/map_tile.py).

The demo screen shows the cascade camera extension next to OpenClaw; on the
real rig it discovers its tiles from the live view's /state. This tile shows
what the runtime's own occupancy map holds -- or, honestly, why it holds
nothing -- without ever slowing or changing the map:

- it reads `OccupancyMap.view_snapshot()` (no lock, no bridge traffic);
- it renders only when a viewer asks for a frame, at most `fps` times a
  second however many viewers share it, and /state never renders;
- every state that is not a live map has its own placeholder, and a stale
  map shows its age in red.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from cascade.apps import live_map_video as lmv
from cascade.apps import map_tile as mt
from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.types import Frame

CLOUD = np.array([[0.30, 0.00, 0.02], [0.32, 0.05, 0.10], [0.25, -0.1, 0.0]], np.float32)
OK_STATUS = {"ok": True, "backend": "nvblox", "device": "cuda:0", "voxel": 0.01, "esdf": True}


def snap(**over):
    base = {"points": CLOUD, "age_s": 0.3, "last_error": None, "body_error": None,
            "status": dict(OK_STATUS), "probe_error": None, "region_min": [0.1, -0.3, -0.01],
            "region_max": [0.5, 0.3, 0.55], "max_age_s": 10.0, "required": False}
    base.update(over)
    return base


class FakeOcc:
    """Stands in for OccupancyMap: only `view_snapshot()` exists, so any
    other access (refresh, clear_bridge_map, the client) fails the test."""

    def __init__(self, **over):
        self.snapshot = snap(**over)
        self.reads = 0

    def view_snapshot(self):
        self.reads += 1
        return dict(self.snapshot)


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def settings(**over):
    kw = {"enabled": True, "fps": 4.0, "stale_s": 2.0, "quality": 80, "size": (480, 270),
          "off_reason": "CASCADE_OCCUPANCY=0"}
    kw.update(over)
    return mt.MapTileSettings(**kw)


def tile(occ=None, *, arm_up=True, clock=None, wall=None, **over):
    return mt.MapTile(occ, settings(**over), arm_up_fn=lambda: arm_up,
                      objects_fn=lambda: [{"label": "red cube", "position": [0.3, 0.0, 0.02],
                                           "state": "visible"}],
                      clock=clock or Clock(), wall=wall or (lambda: 1_760_000_000.0))


def decode(jpeg):
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None, "not a JPEG"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def reddish(img):
    return (img[..., 0].astype(int) > 190) & (img[..., 1] < 140) & (img[..., 2] < 140)


# ── settings: when is the tile announced ────────────────────────────────────


def test_auto_shows_the_tile_only_when_the_runtime_has_an_occupancy_map():
    """Decision: by default the tile exists when there is a map to show. A
    tile that only ever says "off" on every mock/sim dashboard is clutter,
    and the shipped dashboard stays byte-identical with occupancy off."""
    env = {}
    assert mt.map_tile_settings({}, occupancy=object(), env=env).enabled is True
    assert mt.map_tile_settings({}, occupancy=None, env=env).enabled is False
    assert mt.map_tile_settings({"nvblox_map": "auto"}, occupancy=None, env=env).enabled is False
    assert mt.map_tile_settings(None, occupancy=object(), env=env).enabled is True


def test_true_forces_the_tile_and_says_why_occupancy_is_off():
    s = mt.map_tile_settings({"nvblox_map": True}, occupancy=None, occupancy_cfg={"enabled": True},
                             env={"CASCADE_OCCUPANCY": "0"})
    assert s.enabled and s.off_reason == "CASCADE_OCCUPANCY=0"
    s = mt.map_tile_settings({"nvblox_map": {"enabled": True}}, occupancy=None,
                             occupancy_cfg={"enabled": False}, env={})
    assert s.enabled and s.off_reason == "occupancy.enabled: false"
    s = mt.map_tile_settings({"nvblox_map": True}, occupancy=None, occupancy_cfg={"enabled": True}, env={})
    assert "client" in s.off_reason


def test_false_hides_the_tile_even_with_a_map():
    assert mt.map_tile_settings({"nvblox_map": False}, occupancy=object(), env={}).enabled is False
    assert mt.map_tile_settings({"nvblox_map": {"enabled": "off"}}, occupancy=object(), env={}).enabled is False


@pytest.mark.parametrize(("env", "cfg", "occ", "want"), [
    ("0", True, object(), False),       # env kill switch beats the config
    ("off", "auto", object(), False),
    ("1", "auto", None, True),           # forced on: the placeholder explains
    ("on", False, None, True),
    ("auto", True, None, False),         # env auto: back to "is there a map"
    ("auto", False, object(), True),
])
def test_cascade_nvblox_map_overrides_the_config(env, cfg, occ, want):
    s = mt.map_tile_settings({"nvblox_map": cfg}, occupancy=occ, env={"CASCADE_NVBLOX_MAP": env})
    assert s.enabled is want


def test_rate_quality_and_size_are_configurable_and_bounded():
    s = mt.map_tile_settings({"nvblox_map": {"fps": 2, "stale_s": 5, "quality": 60, "width": 640,
                                             "height": 360}}, occupancy=object(), env={})
    assert (s.fps, s.stale_s, s.quality, s.size) == (2.0, 5.0, 60, (640, 360))
    assert mt.map_tile_settings({"nvblox_map": {"fps": 60}}, occupancy=object(), env={}).fps == 10.0
    assert mt.map_tile_settings({"nvblox_map": {"fps": 0}}, occupancy=object(), env={}).fps == 0.5
    d = mt.map_tile_settings({}, occupancy=object(), env={})
    assert (d.fps, d.stale_s, d.size) == (3.0, 2.0, (960, 540))


def test_a_misspelled_switch_fails_loud():
    with pytest.raises(ValueError, match="nvblox_map"):
        mt.map_tile_settings({"nvblox_map": "maybe"}, occupancy=object(), env={})
    with pytest.raises(ValueError, match="CASCADE_NVBLOX_MAP"):
        mt.map_tile_settings({}, occupancy=object(), env={"CASCADE_NVBLOX_MAP": "sometimes"})


# ── what the tile says, state by state ──────────────────────────────────────


def test_occupancy_off_says_so():
    v = tile(None).view()
    assert v.state == "off" and v.headline == "nvblox off (CASCADE_OCCUPANCY=0)"


def test_a_bridge_that_never_answered_is_named():
    v = tile(FakeOcc(points=None, age_s=None, status=None,
                     probe_error="occupancy bridge at 127.0.0.1:5557 timed out (300 ms)")).view()
    assert v.state == "bridge_down" and v.headline == "nvblox bridge not answering"
    assert "127.0.0.1:5557" in " ".join(v.detail)


def test_a_bridge_that_stopped_answering_is_named_not_shown_as_an_old_map():
    v = tile(FakeOcc(age_s=30.0, last_error="occupancy bridge at 127.0.0.1:5557 timed out (500 ms)")).view()
    assert v.state == "bridge_down" and v.headline == "nvblox bridge not answering"


def test_standby_explains_that_the_map_waits_for_the_arm():
    v = tile(FakeOcc(points=None, age_s=None,
                     body_error="robot body pose 0 unavailable: missing or invalid link points"),
             arm_up=False).view()
    assert v.state == "standby"
    text = " ".join([v.headline, *v.detail])
    assert "map starts when the arm powers on" in text and "(the arm must be masked out)" in text


def test_standby_with_an_old_map_still_says_standby():
    v = tile(FakeOcc(age_s=40.0), arm_up=False).view()
    assert v.state == "standby"


def test_a_fresh_map_is_shown_even_with_the_arm_in_standby():
    """A simulated arm is masked without powering a LazyArm, so its map is
    live in standby: show it rather than a placeholder."""
    v = tile(FakeOcc(age_s=0.2), arm_up=False).view()
    assert v.state == "live" and v.ok


def test_a_live_map_reports_its_voxels_and_age():
    v = tile(FakeOcc(age_s=0.4)).view()
    assert v.state == "live" and v.ok and v.voxels == 3
    assert "3 occupied voxels" in v.caption and "0.4 s" in v.caption


def test_a_stale_map_shows_its_age_in_red():
    t = tile(FakeOcc(age_s=12.0, last_error="robot body pose 0 unavailable"))
    v = t.view()
    assert v.state == "stale" and not v.ok and "12.0 s" in v.caption
    img = t.render_rgb()
    assert reddish(img).sum() > 30            # the age (and the error) are drawn red
    fresh = tile(FakeOcc(age_s=0.4)).render_rgb()
    assert reddish(fresh).sum() < reddish(img).sum() / 3


def test_waiting_for_the_first_masked_frame():
    v = tile(FakeOcc(points=None, age_s=None,
                     body_error="robot body pose not yet validated by a masked depth refresh")).view()
    assert v.state == "waiting" and "first masked depth frame" in v.headline
    assert "not yet validated" in " ".join(v.detail)


def test_an_arm_state_that_cannot_be_read_is_not_standby():
    t = mt.MapTile(FakeOcc(points=None, age_s=None), settings(),
                   arm_up_fn=lambda: (_ for _ in ()).throw(RuntimeError("bus")), clock=Clock())
    assert t.view().state == "waiting"


# ── rendering ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("occ, arm_up", [
    (None, True),
    (FakeOcc(points=None, age_s=None, status=None, probe_error="timed out"), True),
    (FakeOcc(points=None, age_s=None), False),
    (FakeOcc(points=None, age_s=None), True),
    (FakeOcc(age_s=0.3), True),
    (FakeOcc(age_s=9.0), True),
])
def test_every_state_renders_a_jpeg_of_the_configured_size(occ, arm_up):
    img = decode(tile(occ, arm_up=arm_up).jpeg())
    assert img.shape == (270, 480, 3)
    assert img.std() > 5                      # text/voxels drawn, not a blank frame


def test_a_placeholder_is_never_a_frozen_frame():
    """A placeholder carries a clock, so two frames a second apart differ:
    the audience can tell a live "bridge not answering" from a frozen one."""
    now = [1_760_000_000.0]
    clock = Clock()
    t = tile(None, clock=clock, wall=lambda: now[0])
    a = t.render_rgb()
    now[0] += 1.0
    clock.t += 1.0
    assert not np.array_equal(a, t.render_rgb())


def test_the_map_orbits_over_time():
    clock = Clock()
    t = tile(FakeOcc(age_s=0.1), clock=clock)
    a = t.render_rgb()
    clock.t += 5.0
    assert not np.array_equal(a, t.render_rgb())


# ── cost: render only on demand, shared, never blocking ─────────────────────


def test_state_polls_never_render():
    t = tile(FakeOcc(age_s=0.3))
    calls = []
    t._render = lambda *a, **k: calls.append(1)        # any render fails the count
    for _ in range(50):
        entry = t.stats()
    assert calls == [] and t.frame_id == 0
    assert entry["virtual"] is True and entry["kind"] == "map"
    assert entry["map_state"] == "live" and entry["voxels"] == 3 and entry["fps"] == 0.0


def test_viewers_share_one_render_per_period():
    clock = Clock()
    t = tile(FakeOcc(age_s=0.3), clock=clock, fps=4.0)
    first = t.jpeg()
    assert all(t.jpeg() is first for _ in range(5))     # same frame for every viewer
    assert t.frame_id == 1
    clock.t += 0.25                                     # one period later: a new frame
    assert t.jpeg() is not first and t.frame_id == 2


def test_stats_do_not_wait_for_a_render_in_progress():
    t = tile(FakeOcc(age_s=0.3))
    entered, release = threading.Event(), threading.Event()
    real = t._render

    def slow(*a, **k):
        entered.set()
        release.wait(5)
        return real(*a, **k)

    t._render = slow
    th = threading.Thread(target=t.jpeg, daemon=True)
    th.start()
    assert entered.wait(5)
    try:
        t0 = time.monotonic()
        t.stats()
        assert time.monotonic() - t0 < 0.1
    finally:
        release.set()
        th.join(5)


# ── read-only against the runtime's real OccupancyMap ───────────────────────


class StrictBridge:
    """A bridge that served the setup refresh, then turns strict: integrate
    and clear raise, and every request at all is recorded."""

    def __init__(self):
        self.strict = False
        self.seen = []

    def request(self, payload, timeout_ms=None):
        action = payload["action"]
        if self.strict:
            self.seen.append(action)
            if action.startswith("integrate") or action == "clear":
                raise PermissionError(f"the map tile tried to write the map: {action}")
        if action == "probe":
            return dict(OK_STATUS)
        if action == "query":
            return {"points": CLOUD.copy(), "grid": np.full((3, 3, 3), 0.2, np.float32),
                    "origin": np.zeros(3, np.float32), "voxel": 0.1}
        if action in ("integrate_depth", "integrate_masked_depth", "clear"):
            return {}
        raise OccupancyError(f"unknown action {action!r}")

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})


def test_the_tile_never_writes_the_map_nor_waits_for_its_locks():
    bridge = StrictBridge()
    m = OccupancyMap(client=bridge, region_min=np.array([0.1, -0.3, -0.01]),
                     region_max=np.array([0.5, 0.3, 0.55]))
    m.probe()
    depth = np.full((8, 8), 0.5, np.float32)
    m.refresh(Frame(rgb=np.zeros((8, 8, 3), np.uint8), depth_m=depth,
                    K=np.array([[50.0, 0, 4], [0, 50.0, 4], [0, 0, 1]])), np.eye(4))
    assert m.last_error is None
    bridge.strict = True
    cloud, stamp, status = m._occupied, m._last_refresh, dict(m.status)
    before = cloud.copy()

    clock = Clock()
    t = mt.MapTile(m, settings(), arm_up_fn=lambda: True, clock=clock)
    held, release = threading.Event(), threading.Event()

    def owner():                       # the perception thread mid-refresh
        with m._io_lock, m._refresh_lock:
            held.set()
            release.wait(10)

    th = threading.Thread(target=owner, daemon=True)
    th.start()
    assert held.wait(5)
    try:
        t0 = time.monotonic()
        for _ in range(5):
            assert decode(t.jpeg()).shape == (270, 480, 3)
            t.stats()
            clock.t += 1.0
        assert time.monotonic() - t0 < 5.0
    finally:
        release.set()
        th.join(5)
    assert t.view().state == "live" and t.frame_id == 5
    assert bridge.seen == []                       # not one request, not even a probe
    assert m._occupied is cloud and np.array_equal(cloud, before)
    assert m._last_refresh == stamp and m.status == status


def test_points_handed_to_the_renderer_are_read_only():
    occ = FakeOcc(age_s=0.3)
    ro = CLOUD.copy()
    ro.flags.writeable = False
    occ.snapshot["points"] = ro
    tile(occ).render_rgb()                         # would raise if the renderer wrote into it


def test_visible_objects_keeps_only_visible_beliefs():
    objs = [{"label": "a", "position": [0.3, 0, 0], "state": "visible"},
            {"label": "b", "position": [0.3, 0, 0], "state": "remembered"}, "junk"]
    assert [o["label"] for o in lmv.visible_objects(objs)] == ["a"]
