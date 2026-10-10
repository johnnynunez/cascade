"""`occupancy.clear_on_start` empties a bridge map left over from earlier runs.

Found on the physical reBot: the bridge outlived the demo, and cells from an
earlier session sat on the arm's own rest pose (0.2-9.9 cm from its joint line,
inside the body mask) -- unmaskable by the new process -- so the first "go home"
was refused at the gripper tip.
"""

import numpy as np
import pytest

from cascade.perception import occupancy as occ_mod
from cascade.perception.occupancy import OccupancyError, OccupancyMap


class FakeClient:
    def __init__(self, *args, up=True, clear_fails=False, **kwargs):
        self.up, self.clear_fails = up, clear_fails
        self.actions = []

    def probe(self, timeout_ms=300):
        if not self.up:
            raise OccupancyError("bridge down")
        return {"ok": True, "backend": "warp", "device": "cuda:0"}

    def request(self, payload, timeout_ms=None):
        self.actions.append(payload["action"])
        if payload["action"] == "clear" and self.clear_fails:
            raise OccupancyError("timed out")
        return {}

    def close(self):
        pass


@pytest.fixture
def built(monkeypatch):
    clients = []

    def make(**kw):
        def factory(*a, **k):
            clients.append(FakeClient(**kw))
            return clients[-1]
        monkeypatch.setattr(occ_mod, "OccupancyClient", factory)
        monkeypatch.delenv("CASCADE_OCCUPANCY", raising=False)
        return clients
    return make


def cfg(**extra):
    return {"enabled": True, "region_min": [-1, -1, -1], "region_max": [1, 1, 1], **extra}


def test_clear_on_start_empties_the_bridge_once(built):
    clients = built()
    m = OccupancyMap.from_config(cfg(clear_on_start=True))
    assert m is not None and clients[0].actions == ["clear"]


def test_default_leaves_a_shared_bridge_untouched(built):
    clients = built()
    OccupancyMap.from_config(cfg())
    assert clients[0].actions == []


def test_no_clear_when_the_bridge_did_not_answer(built):
    clients = built(up=False)
    OccupancyMap.from_config(cfg(clear_on_start=True))
    assert clients[0].actions == []


def test_a_failed_clear_does_not_break_startup(built):
    clients = built(clear_fails=True)
    m = OccupancyMap.from_config(cfg(clear_on_start=True))
    assert m is not None and clients[0].actions == ["clear"]


def test_clear_bridge_map_without_a_client_is_a_no_op():
    assert OccupancyMap(None, region_min=np.zeros(3), region_max=np.ones(3)).clear_bridge_map() is False
