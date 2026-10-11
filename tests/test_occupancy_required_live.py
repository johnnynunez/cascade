"""A REQUIRED occupancy map on a real arm: always live, never silently skipped.

Rig incident (2026-10-11): the reBot's wrist and wrist camera swept into
clutter beside a grasp target and pressed for ~8 s, because the MCP server ran
with the occupancy map OFF. Turning it on for the reBot means a REQUIRED map
(an expired map blocks instead of being skipped), refreshed while the arm
moves, read by every 50 Hz waypoint. These tests pin the map-side contracts
that makes usable on the real arm.
"""

import threading
import time

import numpy as np
import pytest
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map

from cascade.types import SafetyViolation

AT = np.array([[0.30, 0.0, 0.20]])


def in_thread(fn, *args):
    out = {}

    def run():
        try:
            out["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 -- surfaced by the test
            out["error"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


# ── a required map's refresh must not stall the 50 Hz clearance reads ─────


def test_required_refresh_does_not_block_clearance_during_the_round_trip():
    """Found on the physical reBot for optional maps (warp, 3 Hz): a refresh
    holding `_refresh_lock` through its ~61 ms bridge round trip made every
    50 Hz clearance() read wait, and the arm moved in steps. A required map
    took that locked path unconditionally; on the reBot it is now required."""
    bridge = GridBridge([centred_box([0.30, 0.0, 0.05], [0.04, 0.04, 0.04])])
    occ = make_map(bridge)
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.last_error is None
    before = float(occ.clearance(AT)[0])
    bridge.boxes = [centred_box([0.30, 0.0, 0.12], [0.04, 0.04, 0.04])]
    bridge.gate.clear()
    bridge.in_query.clear()
    refresher, _ = in_thread(occ.refresh, depth_frame(), np.eye(4))
    assert bridge.in_query.wait(2)
    t0 = time.monotonic()
    reader, out = in_thread(occ.clearance, AT)
    reader.join(0.5)
    waited = time.monotonic() - t0
    bridge.gate.set()
    refresher.join(2)
    assert not reader.is_alive(), "clearance() waited for the bridge round trip"
    assert waited < 0.5
    assert float(out["value"][0]) == pytest.approx(before)     # the previous grid, consistent
    assert float(occ.clearance(AT)[0]) < before                # then the new one is installed


def test_required_unlocked_refresh_still_refuses_a_stale_or_empty_frame():
    bridge = GridBridge()
    occ = make_map(bridge, max_age_s=1.0)
    occ.refresh(depth_frame(t=time.monotonic() - 5.0), np.eye(4))
    assert "fresh nonempty depth frame" in occ.last_error
    assert "query" not in bridge.actions
    with pytest.raises(SafetyViolation, match="required occupancy unavailable"):
        occ.clearance(AT)
    empty = depth_frame()
    empty.depth_m[:] = 0.0
    occ.refresh(empty, np.eye(4))
    assert "fresh nonempty depth frame" in occ.last_error


def test_required_map_ages_from_the_capture_not_from_the_commit():
    """A refresh that commits late must not make an old image look new."""
    bridge = GridBridge([TABLE])
    occ = make_map(bridge, max_age_s=1.0)
    occ.refresh(depth_frame(t=time.monotonic() - 0.8), np.eye(4))
    assert occ.last_error is None
    occ.clearance(AT)                                            # 0.8 s old: still usable
    time.sleep(0.25)
    with pytest.raises(SafetyViolation, match="stale"):
        occ.clearance(AT)
