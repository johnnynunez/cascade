"""clearance() is not blocked by a plain map refresh's bridge round trip.

Found on the physical reBot (warp occupancy on CUDA, 3 Hz refresh): refresh()
held `_refresh_lock` through the body-pose read and the bridge round trip
(~61 ms), and clearance(), which the harness calls for every 50 Hz waypoint,
waited on it. Two to three waypoints then went out late and back to back:
the arm moved in steps, while the same moves were smooth with the map off.
"""

import threading
import time

import numpy as np
import pytest

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.types import Frame

ORIGIN = np.array([-1.0, -1.0, -1.0], dtype=np.float32)


class SlowBridge:
    """Bridge whose `query` blocks until the test opens `gate`."""

    def __init__(self):
        self.gate = threading.Event()
        self.in_query = threading.Event()
        self.value = 0.25
        self.fail = False
        self.actions = []

    def request(self, payload, timeout_ms=None):
        self.actions.append(payload["action"])
        if payload["action"] == "probe":
            return {"ok": True, "backend": "warp", "device": "cuda:0"}
        if payload["action"] in ("integrate_depth", "clear"):
            return {}
        if payload["action"] == "query":
            self.in_query.set()
            assert self.gate.wait(5), "test never released the bridge"
            if self.fail:
                raise OccupancyError("timed out")
            return {"points": np.empty((0, 3), np.float32),
                    "grid": np.full((3, 3, 3), self.value, np.float32),
                    "origin": ORIGIN, "voxel": 1.0}
        raise OccupancyError(f"unknown action {payload['action']!r}")

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})


def frame():
    depth = np.full((8, 8), 0.5, dtype=np.float32)
    K = np.array([[50.0, 0, 4], [0, 50.0, 4], [0, 0, 1]])
    return Frame(rgb=np.zeros((8, 8, 3), np.uint8), depth_m=depth, K=K)


@pytest.fixture
def plain():
    bridge = SlowBridge()
    m = OccupancyMap(client=bridge, region_min=np.full(3, -1.0), region_max=np.full(3, 1.0))
    bridge.gate.set()
    m.refresh(frame(), np.eye(4))                       # first grid: 0.25 everywhere
    assert m.last_error is None
    bridge.gate.clear()
    bridge.in_query.clear()
    return m, bridge


def in_thread(fn, *args):
    out = {}

    def run():
        out["value"] = fn(*args)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


AT = np.array([[0.0, 0.0, 0.0]])


def test_clearance_answers_from_the_last_grid_during_the_round_trip(plain):
    m, bridge = plain
    bridge.value = 0.10
    refresher, _ = in_thread(m.refresh, frame(), np.eye(4))
    assert bridge.in_query.wait(2)
    t0 = time.monotonic()
    reader, out = in_thread(m.clearance, AT)
    reader.join(0.5)
    assert not reader.is_alive(), "clearance() waited for the bridge round trip"
    assert time.monotonic() - t0 < 0.5
    assert out["value"][0] == pytest.approx(0.25)       # previous grid, consistent
    bridge.gate.set()
    refresher.join(2)
    assert m.clearance(AT)[0] == pytest.approx(0.10)    # new grid installed


def test_a_failed_round_trip_keeps_the_last_grid(plain):
    m, bridge = plain
    bridge.fail = True
    bridge.gate.set()
    m.refresh(frame(), np.eye(4))
    assert "timed out" in m.last_error
    assert m.clearance(AT)[0] == pytest.approx(0.25)


def test_a_mode_change_waits_for_the_upload_in_flight(plain):
    m, bridge = plain
    refresher, _ = in_thread(m.refresh, frame(), np.eye(4))
    assert bridge.in_query.wait(2)
    fence, _ = in_thread(m.fence_capture_refresh, "test")
    fence.join(0.3)
    assert fence.is_alive(), "a capture fence began during a plain upload"
    bridge.gate.set()
    refresher.join(2)
    fence.join(2)
    assert not fence.is_alive() and m._reset_pending
    assert m._plain_refresh() is False                  # later refreshes take the locked path


def test_only_plain_optional_maps_refresh_unlocked():
    m = OccupancyMap(client=SlowBridge(), region_min=np.full(3, -1.0), region_max=np.full(3, 1.0))
    assert m._plain_refresh()
    assert not OccupancyMap(client=SlowBridge(), region_min=np.full(3, -1.0),
                            region_max=np.full(3, 1.0), required=True)._plain_refresh()
    m.track_payload(lambda: np.eye(4))
    assert not m._plain_refresh()
    assert not OccupancyMap(client=None, region_min=np.full(3, -1.0),
                            region_max=np.full(3, 1.0))._plain_refresh()
