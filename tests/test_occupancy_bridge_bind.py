"""The occupancy bridge binds loopback unless told otherwise.

The runtime dials `occupancy.host: 127.0.0.1` (configs/demo.yaml: "launch.sh
starts this bridge on this machine"), yet scripts/serve_occupancy_bridge.py
bound `tcp://0.0.0.0`. The REP socket has no authentication, and its
`integrate*` actions write the obstacle map the motion planner trusts, so
anyone on the LAN (a booth network) could add or erase obstacles in front
of a moving arm. Found on the Spark rig, 2026-10-11: :5557 was listening on
every interface. A remote bridge stays possible, but only explicitly.
"""

from __future__ import annotations

import sys

import pytest
from conftest import REPO
from owned_server import spawn_announced

BRIDGE = REPO / "scripts" / "serve_occupancy_bridge.py"

pytest.importorskip("zmq")
pytest.importorskip("msgpack_numpy")


def _endpoint(tmp_path, *extra) -> str:
    proc, ready = spawn_announced([sys.executable, str(BRIDGE), "--backend", "voxel", *extra],
                                  deadline_s=60.0, stderr_path=tmp_path / "bridge.stderr")
    try:
        return str(ready["endpoint"])
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_the_bridge_binds_loopback_by_default(tmp_path):
    assert _endpoint(tmp_path).startswith("tcp://127.0.0.1:")


def test_a_remote_bridge_is_an_explicit_choice(tmp_path):
    assert _endpoint(tmp_path, "--host", "0.0.0.0").startswith("tcp://0.0.0.0:")
