"""Queued arm requests cannot revive after a dashboard STOP / Resume cycle."""
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cascade.apps.mcp_server import _admit, _text_result, _worker_loop
from test_franka_mcp_resume import payload, server_with_arm


def received(server, name, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
               "params": {"name": name, "arguments": {}}}
    assert _admit(server, message, lambda _: None) is False
    return message


def run_worker(server, *messages):
    inbox, eof, replies = queue.Queue(), object(), []
    for message in messages:
        inbox.put((message, replies.append))
    inbox.put(eof)
    assert _worker_loop(server, SimpleNamespace(checkpoint=lambda: None), inbox, eof) == 0
    return replies


@pytest.mark.parametrize("name", ["move_home", "close_gripper", "run_program", "reset_stop"])
def test_stop_resume_invalidates_old_requests_but_allows_fresh_ones(monkeypatch, name):
    server, rt = server_with_arm(monkeypatch)
    server._stop_pending = False
    old = received(server, name)
    server.stop_now()
    assert payload(server.reset_now())["ok"]
    fresh = received(server, name, 2)
    server.call_tool = Mock(return_value=_text_result({"ok": True}))
    replies = run_worker(server, old, fresh)
    assert replies[0]["result"]["isError"]
    assert "invalidated by stop" in payload(replies[0]["result"])["error"]
    assert not replies[1]["result"].get("isError", False)
    server.call_tool.assert_called_once_with(name, {})


@pytest.mark.parametrize("name", ["close_gripper", "reset_stop"])
def test_stop_resume_between_worker_check_and_execution_still_invalidates_motion(monkeypatch, name):
    server, rt = server_with_arm(monkeypatch)
    server._stop_pending = False
    command = received(server, name)

    def runtime_after_stop_resume():
        def operator_resume():
            server.stop_now()
            assert payload(server.reset_now())["ok"]
        # The dashboard owns its own request thread, not the MCP receipt.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(operator_resume).result()
        return rt

    monkeypatch.setattr(server, "_ensure_runtime", runtime_after_stop_resume)
    monkeypatch.setattr(server, "_reject_withheld", lambda _: None)
    reply = run_worker(server, command)[0]["result"]
    assert reply["isError"]
    assert "invalidated by stop" in payload(reply)["error"]
    rt.execute.assert_called_once_with("move_home", {})  # explicit Resume only


def test_program_rechecks_stop_generation_after_proposal_before_execution(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    server._stop_pending = False
    command = received(server, "run_program")
    command["params"]["arguments"] = {"spec": {}, "task": "pick the cube"}
    tier = Mock()

    def proposal_after_stop_resume(*_):
        def operator_resume():
            server.stop_now()
            assert payload(server.reset_now())["ok"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(operator_resume).result()
        return SimpleNamespace(kind="valid", program=SimpleNamespace(name="pick"))

    tier.proposal_for_spec.side_effect = proposal_after_stop_resume
    monkeypatch.setattr(server, "_ensure_runtime", lambda: rt)
    monkeypatch.setattr(server, "_reject_withheld", lambda _: None)
    monkeypatch.setattr(server, "_programs_on", lambda: True)
    monkeypatch.setattr(server, "_program_unavailable", lambda _: None)
    monkeypatch.setattr(server, "_program_call",
                        lambda runtime, name, args: server._run_program(runtime, tier, args))
    result = run_worker(server, command)[0]["result"]
    assert result["isError"] and "invalidated by stop" in payload(result)["error"]
    tier.run.assert_not_called()
    rt.execute.assert_called_once_with("move_home", {})


def test_motion_received_while_stopped_is_not_delayed_until_after_resume(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    command = received(server, "close_gripper")
    assert payload(server.reset_now())["ok"]
    server.call_tool = Mock(return_value=_text_result({"ok": True}))
    result = run_worker(server, command)[0]["result"]
    assert result["isError"] and "e-stop" in payload(result)["error"]
    server.call_tool.assert_not_called()


def test_observation_queued_before_stop_still_runs_without_materializing_an_arm(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    command = received(server, "world_state")
    server.stop_now()
    server._runtime = None
    server.call_tool = Mock(return_value=_text_result({"ok": True}))
    monkeypatch.setattr(server, "_ensure_runtime", Mock(side_effect=AssertionError("unexpected build")))
    assert not run_worker(server, command)[0]["result"].get("isError", False)
    server.call_tool.assert_called_once_with("world_state", {})


def test_stale_motion_never_builds_runtime_or_probes_arm(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    server._stop_pending = False
    command = received(server, "close_gripper")
    server.stop_now()
    server._runtime = None
    ensure_runtime = Mock(side_effect=AssertionError("unexpected build"))
    monkeypatch.setattr(server, "_ensure_runtime", ensure_runtime)
    result = run_worker(server, command)[0]["result"]
    assert result["isError"] and "invalidated by stop" in payload(result)["error"]
    ensure_runtime.assert_not_called()
    rt.arm.raw.get_state.assert_not_called()


def test_receive_stamp_overwrites_forged_wire_fields(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "close_gripper", "arguments": {}},
               "_arm_stop_serial": -1, "_arm_stop_pending": False}
    assert not _admit(server, message, lambda _: None)
    assert message["_arm_stop_serial"] == server._stop_serial
    assert message["_arm_stop_pending"] is True


def test_fresh_explicit_reset_is_allowed_and_receipt_is_cleared(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    monkeypatch.setattr(server, "_ensure_runtime", lambda: rt)
    monkeypatch.setattr(server, "_reject_withheld", lambda _: None)
    result = run_worker(server, received(server, "reset_stop"))[0]["result"]
    assert not result.get("isError", False)
    assert payload(result)["resumed"]
    server.stop_now()
    assert payload(server.reset_now())["ok"]  # no stale worker receipt leaked


def test_cancelled_queued_call_completes_transport_waiter_without_execution(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    server._stop_pending = False
    command = received(server, "close_gripper")
    server.cancel_request(command["id"])
    server.call_tool = Mock(side_effect=AssertionError("cancelled command executed"))
    replies = run_worker(server, command)
    assert len(replies) == 1 and replies[0]["id"] == command["id"]
    assert replies[0]["result"]["isError"]
    assert payload(replies[0]["result"])["cancelled"] is True
    server.call_tool.assert_not_called()


def test_input_closed_invalidates_queued_arm_motion_before_runtime_build(monkeypatch):
    server, rt = server_with_arm(monkeypatch)
    server._stop_pending = False
    command = received(server, "close_gripper")
    server.input_closed()
    server._runtime = None
    build = Mock(side_effect=AssertionError("closed server built runtime"))
    monkeypatch.setattr(server, "_ensure_runtime", build)
    result = run_worker(server, command)[0]["result"]
    assert result["isError"] and "input closed" in payload(result)["error"]
    build.assert_not_called()
