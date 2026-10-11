"""The `nvblox_map` tile wired into the real runtime (apps/demo.py).

`build_runtime` resolves `stream.nvblox_map` once (a typo fails the build),
and every live view it opens gets a fresh MapTile over the runtime's own
OccupancyMap, the safety workspace, the arm's standby flag (read without
ever materializing a LazyArm) and the visible beliefs.

Everything runs on the mock stack. The occupancy bridge is an in-process
fake: no test here opens a socket to 127.0.0.1:5557 or any live port.
"""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pytest
from conftest import loopback_host

from cascade.apps.capabilities import capability_matrix
from cascade.apps.demo import _runtime_state, build_runtime, shutdown_runtime
from cascade.apps.mcp_server import McpSkillServer
from cascade.config import load_demo_config
from cascade.perception.occupancy import OccupancyMap

CLOUD = np.array([[0.30, 0.00, 0.02], [0.32, 0.05, 0.10]], np.float32)


class FakeBridge:
    """An in-process occupancy bridge: answers the runtime's own probe,
    depth integration and queries (no socket anywhere)."""

    def __init__(self):
        self.actions = []

    def request(self, payload, timeout_ms=None):
        self.actions.append(payload["action"])
        if payload["action"] == "probe":
            return {"ok": True, "backend": "nvblox", "device": "cuda:0", "voxel": 0.01, "esdf": True}
        if payload["action"] == "query":
            return {"points": CLOUD.copy(), "grid": np.full((3, 3, 3), 0.5, np.float32),
                    "origin": np.array([-1.0, -1.0, -1.0], np.float32), "voxel": 1.0}
        return {}

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})


def _cfg(nvblox_map=None):
    cfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock")
    cfg._data["stream"]["port"] = 0                 # ephemeral, never a shared port
    if nvblox_map is not None:
        cfg._data["stream"]["nvblox_map"] = nvblox_map
    return cfg


@pytest.fixture
def make_runtime(tmp_path, monkeypatch):
    built = []

    def make(nvblox_map=None, *, occupancy=False, lazy_arm=False):
        bridge = None
        if occupancy:
            bridge = FakeBridge()

            def from_config(cls, cfg, workspace_min=None, workspace_max=None):
                m = cls(client=bridge, region_min=workspace_min, region_max=workspace_max)
                m.probe()
                return m

            monkeypatch.setattr(OccupancyMap, "from_config", classmethod(from_config))
        runtime, arm = build_runtime(_cfg(nvblox_map), tmp_path / f"run{len(built)}", serve=True,
                                     lazy_arm=lazy_arm)
        built.append((runtime, arm))
        return runtime, bridge

    yield make
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _open(runtime):
    assert runtime.live_view.open(reason="test")["ok"]
    return runtime.live_view.server


def _state(server):
    url = f"http://{loopback_host()}:{server.port}/state"
    return json.loads(urllib.request.urlopen(url, timeout=5).read())


def _one_part(server):
    import http.client

    conn = http.client.HTTPConnection(loopback_host(), server.port, timeout=10)
    conn.request("GET", "/stream/nvblox_map")
    resp = conn.getresponse()
    try:
        assert resp.status == 200
        length = None
        while True:
            line = resp.fp.readline().strip()
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1])
            elif not line and length is not None:
                data = resp.fp.read(length)
                assert data[:2] == b"\xff\xd8"
                return data
    finally:
        resp.close()
        conn.close()


def test_default_without_occupancy_announces_no_map(make_runtime):
    """conftest pins CASCADE_OCCUPANCY=0: `auto` has no map to show, so the
    live view is exactly the camera rig (the golden dashboard holds)."""
    runtime, _ = make_runtime()
    server = _open(runtime)
    assert list(_state(server)["cameras"]) == list(runtime.rig.names)


def test_a_forced_tile_says_why_the_map_is_off(make_runtime):
    runtime, _ = make_runtime(nvblox_map=True)
    server = _open(runtime)
    cams = _state(server)["cameras"]
    assert list(cams)[-1] == "nvblox_map" and cams["nvblox_map"]["map_state"] == "off"
    assert server.map_tile_for("nvblox_map").view().headline == "nvblox off (CASCADE_OCCUPANCY=0)"
    assert _one_part(server)


def test_a_misspelled_switch_fails_the_build(make_runtime):
    with pytest.raises(ValueError, match="nvblox_map"):
        make_runtime(nvblox_map="yes please")


def test_agent_facing_camera_lists_never_contain_the_map(make_runtime):
    """The tile is for people. The agent's camera lists come from the rig:
    MCP world_state, camera_snapshot, analyze, the capability matrix and the
    dashboard state body all stay real-camera-only."""
    runtime, _ = make_runtime(nvblox_map=True)
    server = _open(runtime)
    assert "nvblox_map" in _state(server)["cameras"]
    real = list(runtime.rig.names)
    assert "nvblox_map" not in real
    assert "cameras" not in _runtime_state(runtime)
    mcp = McpSkillServer()
    assert list(mcp._world_state(runtime)["cameras"]) == real
    assert mcp._camera_snapshot(runtime, "nvblox_map").get("isError") is True
    assert set(server.analyze()["cameras"]) == set(real)
    assert set(capability_matrix(runtime)["cameras"]) == set(real)


def test_the_map_follows_the_runtime_from_standby_to_live(make_runtime):
    """End to end: auto announces the tile because the runtime has a map.
    With the arm in standby (LazyArm, never powered by a viewer) nothing is
    integrated and the tile says why; once a motion skill powers the arm the
    watcher masks it, the map fills and the tile shows the voxels."""
    runtime, bridge = make_runtime(occupancy=True, lazy_arm=True)
    server = _open(runtime)
    entry = _state(server)["cameras"]["nvblox_map"]
    assert entry["virtual"] is True and entry["map_state"] == "standby"
    assert _one_part(server)
    assert runtime.arm.raw.connected is False        # watching the map powered nothing

    # The first motion powers the arm (LazyArm connects); with the body not
    # yet masked the map refuses that motion (fail closed, by design), and
    # the watcher's next refresh masks the now-known arm and fills the map.
    runtime.execute("move_home", {})
    assert runtime.arm.raw.connected is True
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        entry = _state(server)["cameras"]["nvblox_map"]
        if entry["map_state"] == "live":
            break
        time.sleep(0.1)
    assert entry["map_state"] == "live" and entry["voxels"] == len(CLOUD)
    assert _one_part(server)
    assert set(bridge.actions) <= {"probe", "integrate_depth", "integrate_masked_depth", "query"}
