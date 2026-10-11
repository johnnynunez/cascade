"""OccupancyMap.view_snapshot(): what a viewer may show of the map.

The live view's `nvblox_map` tile (apps/map_tile.py) draws the runtime's own
occupancy map from inside the robot process. It must never slow the threads
that own the map -- the perception watcher refreshing it, the 50 Hz harness
reading clearance() -- and never change it: the snapshot takes no lock,
sends nothing to the bridge, and hands out read-only views.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.types import Frame

POINTS = np.array([[0.30, 0.00, 0.02], [0.32, 0.05, 0.10]], np.float32)


class Bridge:
    """Answers probe/integrate/query and records every action it receives."""

    def __init__(self, *, up=True):
        self.up = up
        self.actions = []

    def request(self, payload, timeout_ms=None):
        self.actions.append(payload["action"])
        if not self.up:
            raise OccupancyError("occupancy bridge at 127.0.0.1:5557 timed out (500 ms)")
        if payload["action"] == "probe":
            return {"ok": True, "backend": "nvblox", "device": "cuda:0", "voxel": 0.01, "esdf": True}
        if payload["action"] == "query":
            return {"points": POINTS.copy(), "grid": np.full((3, 3, 3), 0.2, np.float32),
                    "origin": np.zeros(3, np.float32), "voxel": 0.1}
        return {}

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})


def frame():
    depth = np.full((8, 8), 0.5, dtype=np.float32)
    K = np.array([[50.0, 0, 4], [0, 50.0, 4], [0, 0, 1]])
    return Frame(rgb=np.zeros((8, 8, 3), np.uint8), depth_m=depth, K=K)


def refreshed_map(bridge=None):
    bridge = bridge or Bridge()
    m = OccupancyMap(client=bridge, region_min=np.array([0.1, -0.3, -0.01]),
                     region_max=np.array([0.5, 0.3, 0.55]))
    m.probe()
    m.refresh(frame(), np.eye(4))
    assert m.last_error is None
    return m, bridge


def test_the_snapshot_carries_what_the_last_refresh_installed():
    m, _ = refreshed_map()
    snap = m.view_snapshot()
    assert np.array_equal(snap["points"], POINTS)
    assert 0.0 <= snap["age_s"] < 1.0
    assert snap["status"]["backend"] == "nvblox" and snap["last_error"] is None
    assert snap["body_error"] is None and snap["probe_error"] is None
    assert snap["region_min"] == pytest.approx([0.1, -0.3, -0.01])
    assert snap["region_max"] == pytest.approx([0.5, 0.3, 0.55])
    assert snap["max_age_s"] == m.max_age_s


def test_before_any_refresh_there_is_nothing_to_show():
    m = OccupancyMap(client=Bridge(up=False), region_min=np.zeros(3), region_max=np.ones(3))
    m.probe()
    snap = m.view_snapshot()
    assert snap["points"] is None and snap["age_s"] is None
    assert snap["status"] is None and "timed out" in snap["probe_error"]


def test_the_snapshot_is_read_only_and_never_talks_to_the_bridge():
    m, bridge = refreshed_map()
    sent = list(bridge.actions)
    before = m._occupied.copy()
    snap = m.view_snapshot()
    with pytest.raises(ValueError):
        snap["points"][0, 0] = 99.0            # a viewer cannot write the map's cloud
    snap["status"]["backend"] = "tampered"     # nor its probe status
    assert np.array_equal(m._occupied, before)
    assert m.status["backend"] == "nvblox"
    assert bridge.actions == sent              # no probe, no query, nothing


def test_the_snapshot_never_waits_for_the_maps_locks():
    """A refresh holds `_io_lock` (and, on the guarded path, `_refresh_lock`)
    through a bridge round trip; clearance() holds `_refresh_lock` for every
    waypoint. A viewer queued behind either would be harmless for itself but
    a viewer TAKING them would delay the robot. It takes neither."""
    m, _ = refreshed_map()
    held, release = threading.Event(), threading.Event()

    def owner():
        with m._io_lock, m._refresh_lock:
            held.set()
            release.wait(5)

    t = threading.Thread(target=owner, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        t0 = time.monotonic()
        snap = m.view_snapshot()
        assert time.monotonic() - t0 < 0.1
        assert snap["points"] is not None
    finally:
        release.set()
        t.join(5)


def test_a_failed_refresh_is_visible_with_the_old_cloud_still_ageing():
    bridge = Bridge()
    m, _ = refreshed_map(bridge)
    bridge.up = False
    m.refresh(frame(), np.eye(4))
    snap = m.view_snapshot()
    assert "timed out" in snap["last_error"]
    assert np.array_equal(snap["points"], POINTS) and snap["age_s"] >= 0.0
