"""The suite never reaches a live GraspGen-X server (spec 2026-10-16, ruling P2 / B6).

demo.yaml's grasp backend is `graspgenx` on 127.0.0.1:5556 and every mock-
profile build_runtime probes it. On a host that serves the model (the Spark,
the rig) unit tests used to send it requests beside the robot's own: 45 of
them during one recorded sim run. tests/conftest.py points
CASCADE_GRASPGENX_PORT at a port the test process holds bound but not
listening, unless a test sets its own.
"""

from __future__ import annotations

import os
import socket

import pytest

from cascade.config import load_demo_config

# What import-time code of a test module sees (collection runs before any fixture).
_AT_IMPORT = (os.environ.get("CASCADE_GRASPGENX_PORT"), os.environ.get("CASCADE_GRASPGENX_HOST"))


@pytest.fixture(scope="module")
def module_scoped_view():
    """What a module- or session-scoped fixture sees: it is set up before any
    function-scoped fixture (the per-test pin) runs."""
    ggx = load_demo_config(camera="mock", arm="mock", llm="mock").grasp.graspgenx
    return (os.environ.get("CASCADE_GRASPGENX_PORT"), os.environ.get("CASCADE_GRASPGENX_HOST"),
            (ggx.host, ggx.port))


def test_module_fixtures_and_import_time_code_never_see_5556(module_scoped_view):
    port, host, endpoint = module_scoped_view
    assert port is not None, "a module-scoped fixture saw no pinned port: it would dial 5556"
    assert _AT_IMPORT == (port, None), _AT_IMPORT
    assert host is None and int(port) != 5556
    assert endpoint == ("127.0.0.1", int(port))
    with pytest.raises(ConnectionRefusedError):  # held for the whole session, not listening
        socket.create_connection(("127.0.0.1", int(port)), timeout=1).close()
    probe = socket.socket()
    try:
        with pytest.raises(OSError):
            probe.bind(("127.0.0.1", int(port)))
    finally:
        probe.close()


def test_every_test_dials_a_held_dead_port_instead_of_5556():
    port = int(os.environ["CASCADE_GRASPGENX_PORT"])
    assert port != 5556
    assert "CASCADE_GRASPGENX_HOST" not in os.environ
    ggx = load_demo_config(camera="mock", arm="mock", llm="mock").grasp.graspgenx
    assert (ggx.host, ggx.port) == ("127.0.0.1", port)
    with pytest.raises(ConnectionRefusedError):  # held, not listening: nothing answers
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    probe = socket.socket()
    try:
        with pytest.raises(OSError):  # and nothing else can bind it while the test runs
            probe.bind(("127.0.0.1", port))
    finally:
        probe.close()


def test_the_mock_runtime_planner_dials_the_dead_port():
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert cfg.grasp.backend == "graspgenx"
    client = GraspGenXPlanner(cfg.grasp)._client
    assert (client._host, client._port) == ("127.0.0.1", int(os.environ["CASCADE_GRASPGENX_PORT"]))


def test_a_test_sets_its_own_port(monkeypatch):
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "45999")
    assert load_demo_config(camera="mock", arm="mock", llm="mock").grasp.graspgenx.port == 45999
    monkeypatch.delenv("CASCADE_GRASPGENX_PORT")
    assert load_demo_config(camera="mock", arm="mock", llm="mock").grasp.graspgenx.port == 5556
