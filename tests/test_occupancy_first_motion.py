"""The first motion after a standby arm wakes waits for one masked map refresh.

With the arm in standby (a LazyArm: the MCP server never powers the motors as
a side effect of looking) there is no measured pose to mask the robot out of
the depth, so the occupancy map integrates NOTHING and latches "robot body
pose unavailable" (by design: an unmasked arm would map itself as an
obstacle). The first motion command materializes the arm; before this change
its very first harness check met that latch and was refused, although the
watcher's next tick (~0.3 s later) would have produced a masked refresh.
Proceeding unchecked is not an option either. The wake now waits, bounded by
`occupancy.ready_timeout_s`, for one masked refresh.
"""

import threading
import time

import numpy as np
import pytest
from conftest import needs_pin, observed
from occupancy_grid_bridge import TABLE, GridBridge, depth_frame, make_map

from cascade.control.lazy_arm import LazyArm
from cascade.control.mock_arm import MockArm
from cascade.types import SafetyViolation

#: a camera 1.2 m in front of the arm, looking back at it
T_CAM = np.array([[0.0, 0.0, -1.0, 1.2], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.3], [0, 0, 0, 1.0]])


class Refresher:
    """The WorldWatcher's occupancy half: refresh every `period_s`."""

    def __init__(self, occ, period_s=0.05):
        self.occ, self.period_s = occ, period_s
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.occ.refresh(depth_frame(), T_CAM)
            self.stop.wait(self.period_s)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(2)


def test_lazy_arm_runs_its_wake_hook_once_after_connecting_outside_its_lock():
    calls = []
    arm = None

    def hook():
        # the watcher reads the mask pose through the arm while we wait: it
        # must not deadlock on the LazyArm lock
        reader = threading.Thread(target=lambda: calls.append(("read", arm.get_state().q.size)))
        reader.start()
        reader.join(1)
        assert not reader.is_alive(), "the wake hook ran under LazyArm's lock"
        calls.append(("hook", arm.connected))

    arm = LazyArm(lambda: MockArm(), n_joints=6, on_materialize=hook)
    assert not arm.connected and calls == []
    arm.get_state()
    arm.get_state()
    assert calls == [("read", 6), ("hook", True)]


def _mock_build(occ, *, lazy=True):
    from cascade.apps import demo
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm="mock", camera="mock", llm="mock")
    raw, safe, kin = demo._build_arm(cfg.arm, lazy, occ, cfg)
    observed(safe.harness)          # the watcher's camera half: a frame came in
    return raw, safe, kin, cfg


@needs_pin
@pytest.mark.parametrize("required", [True, False])
def test_the_first_motion_after_wake_waits_for_a_masked_refresh_then_moves(required):
    occ = make_map(GridBridge([TABLE]), required=required, unobserved="free", ready_timeout_s=3.0)
    raw, safe, _kin, cfg = _mock_build(occ)
    occ.refresh(depth_frame(), T_CAM)               # standby: nothing integrated, latched
    assert "robot body pose" in (occ.last_error or "")
    target = np.asarray(cfg.arm.home_q, float) + np.array([0.15, 0, 0, 0, 0, 0])
    with Refresher(occ):
        t0 = time.monotonic()
        assert safe.move_joints(target, duration_s=0.5)
        waited = time.monotonic() - t0
    assert raw.connected
    assert len(raw._arm.commands) > 0
    np.testing.assert_allclose(raw.get_state().q, target, atol=1e-9)
    assert waited < 1.5


@needs_pin
@pytest.mark.parametrize("required", [True, False])
def test_without_a_masked_refresh_the_wake_is_bounded_and_the_motion_refused(required):
    occ = make_map(GridBridge([TABLE]), required=required, unobserved="free", ready_timeout_s=0.3)
    raw, safe, _kin, cfg = _mock_build(occ)
    occ.refresh(depth_frame(), T_CAM)
    target = np.asarray(cfg.arm.home_q, float) + np.array([0.15, 0, 0, 0, 0, 0])
    t0 = time.monotonic()
    with pytest.raises(SafetyViolation, match="robot body pose"):
        safe.move_joints(target, duration_s=0.5)
    waited = time.monotonic() - t0
    assert 0.25 <= waited < 1.5, waited
    assert raw.connected and raw._arm.commands == []


def test_wait_until_usable_reports_a_ready_required_map_at_once_and_times_out_otherwise():
    occ = make_map(GridBridge([TABLE]), unobserved="free", max_age_s=0.5)
    t0 = time.monotonic()
    assert occ.wait_until_usable(0.2) is False                  # nothing integrated yet
    assert 0.18 <= time.monotonic() - t0 < 0.6
    occ.refresh(depth_frame(), T_CAM)
    assert occ.wait_until_usable(0.0) is True
    time.sleep(0.55)
    assert occ.wait_until_usable(0.0) is False                  # stale again


@pytest.mark.parametrize("bad", [-1.0, float("nan"), "soon", 120.0])
def test_a_bad_ready_timeout_is_refused(bad):
    from cascade.perception.occupancy import OccupancyError, OccupancyMap

    with pytest.raises(OccupancyError, match="ready_timeout_s"):
        OccupancyMap(ready_timeout_s=bad)
