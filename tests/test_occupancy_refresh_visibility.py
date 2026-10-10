"""A refresh that keeps failing must be visible, not only a refused motion.

Found on a Jetson Thor rig: the CPU voxel bridge needed ~0.6 s per 1280x720
depth frame against the 500 ms request timeout. Every refresh timed out,
`refresh()` swallowed the error into `last_error`, the body-mask latch never
cleared, and every motion was refused with "robot body pose not yet validated
by a masked depth refresh" -- with no log line naming the timeout.
"""

import logging

import numpy as np
import pytest

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.types import Frame, SafetyViolation

BODY = np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 1.2]])
TIMEOUT = "occupancy bridge at 127.0.0.1:5557 timed out (500 ms)"


class FlakyClient:
    """Times out until `healthy` is set, then answers like a real bridge."""

    def __init__(self):
        self.healthy = False

    def request(self, payload, timeout_ms=None):
        if not self.healthy:
            raise OccupancyError(TIMEOUT)
        if payload["action"] == "query":
            return {"points": np.empty((0, 3), dtype=np.float32)}
        return {}

    def probe(self, timeout_ms=300):
        return {"ok": True, "backend": "voxel", "device": "cpu"}


def frame():
    return Frame(rgb=np.zeros((2, 2, 3), np.uint8),
                 depth_m=np.ones((2, 2), np.float32), K=np.eye(3))


def masked_map():
    client = FlakyClient()
    occupancy = OccupancyMap(client, region_min=np.zeros(3), region_max=np.ones(3))
    occupancy.add_robot_body(lambda: BODY)
    return occupancy, client


def refresh_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "cascade.perception.occupancy" and "refresh" in r.getMessage()]


def test_persistent_timeout_is_logged_once_and_named_in_the_refusal(caplog):
    occupancy, _ = masked_map()
    with caplog.at_level(logging.WARNING, logger="cascade.perception.occupancy"):
        for _ in range(5):
            occupancy.refresh(frame(), np.eye(4))

    warnings = refresh_warnings(caplog)
    assert len(warnings) == 1, warnings            # once per change, not per frame
    assert TIMEOUT in warnings[0]
    assert "motion stays refused" in warnings[0]
    with pytest.raises(SafetyViolation) as refused:
        occupancy.clearance(np.array([[0.3, 0.0, 0.5]]))
    message = str(refused.value)
    assert message.startswith("occupancy unsafe: robot body pose not yet validated")
    assert f"last refresh failed: {TIMEOUT}" in message


def test_recovery_is_logged_once_and_clears_the_latch(caplog):
    occupancy, client = masked_map()
    with caplog.at_level(logging.WARNING, logger="cascade.perception.occupancy"):
        occupancy.refresh(frame(), np.eye(4))      # fails
        client.healthy = True
        for _ in range(3):
            occupancy.refresh(frame(), np.eye(4))  # recovers, then stays healthy

    assert refresh_warnings(caplog) == [
        refresh_warnings(caplog)[0], "occupancy refresh recovered"]
    assert occupancy.last_error is None
    occupancy.clearance(np.array([[0.3, 0.0, 0.5]]))  # latch cleared: no SafetyViolation


def test_a_healthy_map_logs_nothing(caplog):
    occupancy, client = masked_map()
    client.healthy = True
    with caplog.at_level(logging.WARNING, logger="cascade.perception.occupancy"):
        for _ in range(3):
            occupancy.refresh(frame(), np.eye(4))
    assert refresh_warnings(caplog) == []
