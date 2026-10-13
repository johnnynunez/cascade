"""MCP-only Franka operation retains the operator's STOP and checked home policy."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cascade.apps.mcp_server import McpSkillServer
from cascade.config import Cfg


def payload(result):
    return json.loads(next(b["text"] for b in result["content"] if b["type"] == "text"))


def server_with_arm(monkeypatch):
    monkeypatch.delenv("CASCADE_ROBOT", raising=False)
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    server = McpSkillServer()
    raw = Mock()
    harness = SimpleNamespace(estopped=True)
    harness.reset_estop = Mock(side_effect=lambda: setattr(harness, "estopped", False))
    arm = SimpleNamespace(raw=raw, harness=harness)
    arm.stop = Mock(side_effect=lambda: setattr(harness, "estopped", True))
    runtime = SimpleNamespace(
        arm=arm, cfg=Cfg({"arm": {"return_home_after_motion": True}}),
        execute=Mock(return_value={"ok": True}), _show_status=Mock(),
    )
    server._runtime = runtime
    server._stop_pending = True
    return server, runtime


def test_startup_stop_cannot_be_cleared_before_build(monkeypatch):
    monkeypatch.setenv("CASCADE_START_STOPPED", "1")
    server = McpSkillServer()
    assert server._stop_pending
    result = server.reset_now()
    assert result["isError"] and server._stop_pending
    assert "starting" in payload(result)["error"]


def test_resume_checks_native_readiness_and_only_moves_home(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    result = payload(server.reset_now())
    assert result["ok"] and result["resumed"] and not result["stopped"]
    rt.arm.raw.check_resume_ready.assert_called_once_with()
    rt.arm.raw.resume.assert_called_once_with()
    rt.execute.assert_called_once_with("move_home", {})
    rt.arm.raw.set_gripper.assert_not_called()
    assert not server._stop_pending and not rt.arm.harness.estopped


def test_busy_resume_cannot_interleave_with_motion(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    with server._exec_lock:
        result = server.reset_now()
    assert result["isError"] and server._stop_pending
    rt.arm.raw.resume.assert_not_called()
    rt.execute.assert_not_called()


@pytest.mark.parametrize("phase", ["readiness", "home"])
def test_concurrent_stop_wins_over_resume(monkeypatch, phase):
    server, rt = server_with_arm(monkeypatch)
    if phase == "readiness":
        rt.arm.raw.check_resume_ready.side_effect = server.stop_now
    else:
        def stop_at_home(*_):
            server.stop_now()
            return {"ok": True}
        rt.execute.side_effect = stop_at_home
    result = server.reset_now()
    assert result["isError"] and server._stop_pending and rt.arm.harness.estopped
    if phase == "readiness":
        rt.arm.raw.resume.assert_not_called()
        rt.execute.assert_not_called()


def test_failed_home_relatches_stop_and_dashboard_reports_failure(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    rt.execute.return_value = {"ok": False, "error": "clearance rejected"}
    result = server._resume_dashboard()
    assert not result["resumed"] and "clearance rejected" in result["error"]
    assert server._stop_pending and rt.arm.harness.estopped


def test_cancelling_resume_stops_home_motion(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    server._inflight = (17, "reset_stop")
    server.cancel_request(17)
    rt.arm.stop.assert_called_once_with()
    assert server._stop_pending


@pytest.mark.parametrize("phase", ["before", "readiness", "home"])
def test_resume_cannot_clear_stop_or_continue_home_during_teardown(monkeypatch, phase):
    server, rt = server_with_arm(monkeypatch)
    if phase == "before":
        server.input_closed()
    elif phase == "readiness":
        rt.arm.raw.check_resume_ready.side_effect = server.input_closed
    else:
        rt.execute.side_effect = lambda *a: (server.input_closed() or {"ok": True})
    result = server._resume_dashboard()
    assert not result["ok"] and not result["resumed"]
    assert server._stop_pending and rt.arm.harness.estopped
    if phase != "home":
        rt.arm.raw.resume.assert_not_called()
        rt.execute.assert_not_called()
    else:
        rt.arm.stop.assert_called_once_with()


@pytest.mark.parametrize("new_halt", [False, True])
def test_resume_supersedes_only_the_halt_before_readiness(monkeypatch, new_halt):
    from cascade.safety.harness import SafetyHarness
    from test_safety import limits

    server, rt = server_with_arm(monkeypatch)
    harness = SafetyHarness(limits())
    token = harness._observation_cancel_generation
    harness.halt("superseded")
    harness.estop("operator stopped")
    rt.arm.harness = harness
    rt.arm.stop.side_effect = lambda: harness.estop("stop requested")
    if new_halt:
        rt.arm.raw.check_resume_ready.side_effect = lambda: harness.halt("new hazard")
    else:
        rt.execute.side_effect = lambda *a: {"ok": harness.halted is None}
    result = payload(server.reset_now())
    assert result["ok"] is not new_halt
    assert harness.estopped is new_halt
    assert harness.halted == ("new hazard" if new_halt else None)
    assert harness._observation_cancel_generation > token
    if new_halt:
        rt.execute.assert_not_called()
        assert "HALT requested during resume" in result["error"]


def test_mcp_wires_stop_and_resume_before_lazy_view_exists(monkeypatch):
    from cascade.apps import demo
    from cascade.apps.live_control import LiveViewController, MODE_LAZY
    from test_live_view import _FakeServer

    server, rt = server_with_arm(monkeypatch)
    rt.stream_server = None
    rt.live_view = LiveViewController(_FakeServer, mode=MODE_LAZY, idle_timeout_s=0)
    server._runtime = None
    monkeypatch.setattr(demo, "build_runtime", lambda *a, **kw: (rt, rt.arm))
    monkeypatch.setattr(server, "_attach_program_tier", Mock())
    monkeypatch.setattr(server, "_announce_surface", Mock())
    assert server._ensure_runtime() is rt
    assert rt.live_view.server is None
    for _ in range(2):
        rt.live_view.open()
        view = rt.live_view.server
        assert view.task_fn == server._reflex_chat_task
        assert view.cancel_fn == server.stop_now
        assert view.resume_fn == server._resume_dashboard
        rt.live_view.close()
