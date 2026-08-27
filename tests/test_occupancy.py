import numpy as np
import pytest

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import Frame, SafetyViolation


class FakeClient:
    """Stands in for OccupancyClient without a real ZMQ socket."""

    def __init__(self, occupied=None, fail=False):
        self.occupied = occupied if occupied is not None else np.empty((0, 3))
        self.fail = fail
        self.requests: list[dict] = []

    def request(self, payload: dict) -> dict:
        self.requests.append(payload)
        if self.fail:
            raise OccupancyError("bridge down")
        if payload["action"] == "integrate":
            return {}
        if payload["action"] == "query":
            return {"points": self.occupied.astype(np.float32)}
        raise AssertionError(f"unexpected action {payload['action']!r}")


def _map(occupied=None, fail=False, **kw):
    return OccupancyMap(
        client=FakeClient(occupied=occupied, fail=fail),
        region_min=np.array([-1, -1, -1]),
        region_max=np.array([1, 1, 1]),
        **kw,
    )


def _frame(depth_val=0.5, size=(8, 8)):
    h, w = size
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), depth_val, dtype=np.float32)
    K = np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]])
    return Frame(rgb=rgb, depth_m=depth, K=K)


def test_from_config_disabled_by_default():
    assert OccupancyMap.from_config(None) is None
    assert OccupancyMap.from_config({"enabled": False}) is None


def test_clearance_none_before_first_refresh():
    m = _map()
    assert m.clearance(np.zeros((1, 3))) is None


def test_clearance_after_refresh():
    occupied = np.array([[0.0, 0.0, 0.0]])
    m = _map(occupied=occupied)
    m.refresh(_frame(), T_base_cam=np.eye(4))
    d = m.clearance(np.array([[0.0, 0.0, 0.05], [1.0, 0.0, 0.0]]))
    assert d is not None
    assert d[0] == pytest.approx(0.05, abs=1e-6)
    assert d[1] == pytest.approx(1.0, abs=1e-6)


def test_stale_cache_is_treated_as_absent():
    m = _map(occupied=np.array([[0.0, 0.0, 0.0]]), max_age_s=0.0)
    m.refresh(_frame(), T_base_cam=np.eye(4))
    assert m.clearance(np.zeros((1, 3))) is None


def test_bridge_error_keeps_last_good_cache_and_records_error():
    m = _map(occupied=np.array([[0.0, 0.0, 0.0]]))
    m.refresh(_frame(), T_base_cam=np.eye(4))
    assert m.last_error is None
    m._client.fail = True
    m.refresh(_frame(), T_base_cam=np.eye(4))
    assert m.last_error is not None
    # stale cache is still readable until max_age_s elapses
    assert m.clearance(np.zeros((1, 3))) is not None


class FakeKin:
    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        return np.array([[0, 0, 0.2], [0, 0, 0.2], [q[0], q[1], max(q[2], 0.1)]])


def limits(**kw):
    base = dict(
        workspace_min=np.array([-1.0, -1.0, -1.0]),
        workspace_max=np.array([1.0, 1.0, 1.0]),
        table_z=-1.0,  # keep the table checks out of the way of this test
        table_clearance=0.0,
        max_joint_vel=10.0,
        watchdog_s=1e9,
        min_clearance_m=0.05,
    )
    base.update(kw)
    return SafetyLimits(**base)


def test_harness_rejects_a_waypoint_too_close_to_an_occupied_point():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    occ.refresh(_frame(), T_base_cam=np.eye(4))
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    far = np.array([0.3, 0.0, 0.5, 0, 0, 0])  # tcp = q[:3]
    h.approve(far, far, dt=1e9)  # 0.3 m away, clear
    near = np.array([0.3, 0.0, 0.21, 0, 0, 0])  # 0.01 m away, too close
    with pytest.raises(SafetyViolation, match="occupancy"):
        h.approve(far, near, dt=1e9)


def test_harness_ignores_occupancy_without_fresh_data():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    # never refreshed -> clearance() returns None -> check is skipped
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    near = np.array([0.3, 0.0, 0.21, 0, 0, 0])
    h.approve(near, near, dt=1e9)


def test_harness_grasp_exemption_covers_occupancy_too():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    occ.refresh(_frame(), T_base_cam=np.eye(4))
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    near = np.array([0.3, 0.0, 0.21, 0, 0, 0])
    with pytest.raises(SafetyViolation, match="occupancy"):
        h.approve(near, near, dt=1e9)
    h.allow_grasp_descent(np.array([0.3, 0.0]), radius_m=0.1, z_min=-1.0)
    h.approve(near, near, dt=1e9)


def test_vet_pose_reports_occupancy_violation():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    occ.refresh(_frame(), T_base_cam=np.eye(4))
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    q = np.array([0.3, 0.0, 0.21, 0, 0, 0])
    reason = h.vet_pose(q)
    assert reason is not None and "occupancy" in reason
