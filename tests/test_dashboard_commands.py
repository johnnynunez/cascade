"""Cancellation must stop planning without allowing two robot commands to overlap."""

import json
import signal
import threading
import time
from types import SimpleNamespace
import urllib.request
from unittest.mock import Mock

import pytest

from conftest import loopback_host
from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator, TaskBusyError
from cascade.agent.reflex import FastPlanner
from cascade.apps.stream_server import StreamServer
from cascade.memory import EpisodicMemory


def runtime():
    calls = []

    def execute(name, args):
        calls.append((name, args))
        return {"ok": True, "task_complete": name == "task_done"}

    return SimpleNamespace(
        memory=EpisodicMemory(), last_frame=None, calls=calls, execute=execute,
        frame_jpeg=lambda: None,
        arm=Mock(), trace=Mock(), begin_task_budget=Mock(), end_task_budget=Mock(),
    )


def agent(llm, rt, **kwargs):
    return AgentOrchestrator(llm, rt, decompose=False, verify_milestones=False, **kwargs)


@pytest.mark.parametrize("error", [
    "Arm is stopped; resume before starting another motion",
    "Franka hardware not ready: current_errors: cartesian_reflex",
    "Franka state feedback is stale",
])
def test_blocked_pick_reports_readiness_error_without_model_fallback(error):
    rt = runtime()
    rt.execute = Mock(return_value={"ok": False, "error": error, "motion_blocked": True})
    llm = MockLLM([])
    report = agent(llm, rt, fast_planner=FastPlanner()).run_task("pick up orange ball")
    assert not report.success and report.summary == error
    assert report.path == "reflex" and report.steps == 1
    rt.execute.assert_called_once_with("grasp_object", {"label": "orange ball"})
    assert not llm.requests
    assert report.tool_log[0]["result"]["motion_blocked"]
    rt.trace.finish.assert_called_once()


def test_blocked_model_motion_ends_task_before_another_tool_or_inference():
    rt = runtime()
    rt.execute = Mock(return_value={"ok": False, "error": "Arm is stopped", "motion_blocked": True})
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("grasp_object", {"label": "ball"})]),
        LLMResponse(tool_calls=[ToolCall("open_gripper", {})]),
    ])
    report = agent(llm, rt).run_task("retrieve the round orange object")
    assert not report.success and report.summary == "Arm is stopped"
    assert report.steps == 1 and len(llm.requests) == 1
    rt.execute.assert_called_once_with("grasp_object", {"label": "ball"})


def test_ready_pick_uses_grasp_skill_without_model_planning():
    rt = runtime()
    llm = MockLLM([])
    report = agent(llm, rt, fast_planner=FastPlanner()).run_task("pick up orange ball")
    assert report.success and report.path == "reflex"
    assert rt.calls == [("grasp_object", {"label": "orange ball"})]
    assert not llm.requests


def test_task_initialization_failure_releases_command_ownership():
    rt = runtime()
    rt.begin_task = Mock(side_effect=RuntimeError("initial observation unavailable"))
    orchestrator = agent(MockLLM([]), rt, fast_planner=FastPlanner())
    with pytest.raises(RuntimeError, match="initial observation unavailable"):
        orchestrator.run_task("pick up orange ball")
    assert rt.current_task is None
    assert orchestrator._cancel_event is None
    rt.end_task_budget.assert_called_once()
    rt.begin_task.side_effect = None
    assert orchestrator.run_task("pick up orange ball").success


def test_home_cleanup_sees_cancellation_before_hardware_stop_completes():
    entered, release = threading.Event(), threading.Event()
    rt = runtime()
    rt.motion_cancel_requested = None

    def delayed_stop():
        entered.set()
        assert release.wait(2)

    rt.arm.stop.side_effect = delayed_stop
    orchestrator = agent(MockLLM([]), rt)
    orchestrator._cancel_event = threading.Event()
    worker = threading.Thread(target=orchestrator.cancel)
    worker.start()
    try:
        assert entered.wait(1)
        assert rt.motion_cancel_requested()
        assert worker.is_alive(), "The callback must gate cleanup before STOP finishes"
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()


def wait_idle(server):
    deadline = time.monotonic() + 2
    while server.command_state()["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not server.command_state()["busy"]


def test_dashboard_cancel_keeps_busy_until_inference_returns_and_drops_motion():
    started, release = threading.Event(), threading.Event()

    class BlockedLLM:
        supports_vision = False

        def chat(self, **kwargs):
            started.set()
            assert release.wait(3)
            return LLMResponse(tool_calls=[ToolCall("move_home", {})])

    rt = runtime()
    orchestrator = agent(BlockedLLM(), rt)
    server = StreamServer(SimpleNamespace(stats=lambda: {}), port=0)
    server.set_task_fn(orchestrator.run_task, cancel_aware=True)
    server.set_cancel_fn(orchestrator.cancel)
    server.start()
    base = f"http://{loopback_host()}:{server.port}"

    def post(path, body=None):
        request = urllib.request.Request(
            base + path, data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        return json.loads(urllib.request.urlopen(request, timeout=2).read())

    try:
        assert post("/task", {"task": "go home"})["accepted"]
        assert started.wait(1)
        state = json.loads(urllib.request.urlopen(base + "/state", timeout=2).read())
        assert state["busy"] and state["active_command"] == "go home"
        assert state["command_started_at"] and state["command_elapsed_s"] >= 0
        assert post("/cancel")["cancelled"]
        rt.arm.stop.assert_called_once()
        assert server.command_state()["busy"] and server.command_state()["cancel_requested"]
        assert not post("/task", {"task": "another command"})["accepted"]
        with pytest.raises(TaskBusyError):
            orchestrator.run_task("a REPL command")
        assert rt.current_task == "go home"
        release.set()
        wait_idle(server)
        assert rt.calls == []
        assert rt.current_task is None
        assert server.command_state()["last_command"]["cancelled"]
        assert "cancelled" in server.command_state()["last_command"]["summary"]
        orchestrator.llm = MockLLM([
            LLMResponse(tool_calls=[ToolCall("task_done", {"success": True})]),
        ])
        assert post("/task", {"task": "report status"})["accepted"]
        wait_idle(server)
        assert server.command_state()["last_command"]["success"]
    finally:
        release.set()
        server.stop()


def test_cancel_before_worker_dispatch_skips_planning_and_actions():
    event = threading.Event()
    event.set()
    rt = runtime()
    llm = MockLLM([])
    report = agent(llm, rt).run_task("go home", cancel_event=event)
    assert report.cancelled and not report.success
    assert not llm.requests and not rt.calls
    rt.end_task_budget.assert_called_once()


def test_cancelled_fast_path_cannot_continue_or_fall_back_to_planner():
    event = threading.Event()
    rt = runtime()
    execute = rt.execute

    def cancel_after_call(name, args):
        result = execute(name, args)
        event.set()
        return result

    rt.execute = cancel_after_call
    planner = Mock()
    planner.plan.return_value = SimpleNamespace(
        source="reflex", calls=[("move_home", {}), ("open_gripper", {})], needs_grounding=False,
    )
    llm = MockLLM([])
    report = agent(llm, rt, fast_planner=planner).run_task("go home and open", event)
    assert report.cancelled and report.steps == 1
    assert rt.calls == [("move_home", {})]
    assert not llm.requests
    planner.note_outcome.assert_not_called()


def test_cancel_during_decomposition_prevents_planner_turn():
    event = threading.Event()
    rt = runtime()
    llm = Mock(supports_vision=False)

    def cancelled_reply(**kwargs):
        event.set()
        return LLMResponse(text="1. Move home")

    llm.chat.side_effect = cancelled_reply
    orchestrator = AgentOrchestrator(llm, rt, verify_milestones=False)
    report = orchestrator.run_task("move home", event)
    assert report.cancelled and not rt.calls
    llm.chat.assert_called_once()


def test_inference_exception_after_stop_is_reported_as_cancellation():
    event = threading.Event()
    rt = runtime()
    llm = Mock(supports_vision=False)

    def cancelled_timeout(**kwargs):
        event.set()
        raise TimeoutError("inference timed out")

    llm.chat.side_effect = cancelled_timeout
    report = agent(llm, rt).run_task("observe", event)
    assert report.cancelled and not rt.calls


def test_planner_stagnation_stops_after_three_text_only_turns():
    llm = MockLLM([LLMResponse(text="thinking") for _ in range(10)])
    rt = runtime()
    report = agent(llm, rt, max_steps=30).run_task("correct an object label")
    assert not report.success and report.steps == 3
    assert "no tool calls" in report.summary
    assert len(llm.requests) == 3 and not rt.calls


def test_tool_call_resets_consecutive_stagnation_counter():
    llm = MockLLM([
        LLMResponse(text="thinking"), LLMResponse(text="thinking"),
        LLMResponse(tool_calls=[ToolCall("get_observation", {})]),
        LLMResponse(text="thinking"), LLMResponse(text="thinking"),
        LLMResponse(tool_calls=[ToolCall("task_done", {"success": True})]),
    ])
    assert agent(llm, runtime()).run_task("observe").success
    assert len(llm.requests) == 6


def test_worker_exception_is_visible_and_does_not_leak_busy_lock():
    server = StreamServer(SimpleNamespace(stats=lambda: {}))
    server.set_task_fn(Mock(side_effect=RuntimeError("model context exceeded")))
    assert server.submit_task("observe")["accepted"]
    wait_idle(server)
    state = server.state()
    assert not state["busy"] and state["active_command"] is None
    assert state["last_command"]["success"] is False
    assert "context exceeded" in state["last_command"]["error"]


def test_cancel_during_verification_prevents_next_action():
    event = threading.Event()
    rt = runtime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("move_home", {})]),
        LLMResponse(tool_calls=[ToolCall("open_gripper", {})]),
    ])
    orchestrator = agent(llm, rt)
    tracker = Mock(active=True)
    tracker.update.side_effect = lambda *args, **kwargs: event.set()
    orchestrator.tracker = tracker
    report = orchestrator.run_task("move and open", event)
    assert report.cancelled
    assert rt.calls == [("move_home", {})]
    assert len(llm.requests) == 1


def test_sigterm_at_repl_holds_arm_and_runs_runtime_cleanup(monkeypatch, tmp_path):
    from cascade.apps import demo

    rt = runtime()
    rt.stream_server = None
    from cascade.config import Cfg
    cfg = Cfg({"llm": {}, "memory": {
        "experience_path": str(tmp_path / "experience.json"),
        "skill_library_path": str(tmp_path / "library"),
    }})
    monkeypatch.setattr(demo, "load_demo_config", lambda **kwargs: cfg)
    monkeypatch.setattr(demo, "build_runtime", lambda *args, **kwargs: (rt, rt.arm))
    monkeypatch.setattr(demo, "make_llm", lambda cfg: MockLLM([]))
    cleanup = Mock()
    monkeypatch.setattr(demo, "shutdown_runtime", cleanup)
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.update({sig: handler}))

    def terminate_at_prompt(prompt):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr("builtins.input", terminate_at_prompt)
    rt.arm_rig = None
    assert demo.main(["--interactive", "--llm", "mock", "--no-view", "--no-serve"]) == 128 + signal.SIGTERM
    rt.arm.harness.halt.assert_called_once()
    cleanup.assert_called_once_with(rt, rt.arm)


@pytest.mark.parametrize("held", [None, "brown wood piece"])
def test_maintenance_restart_preserves_stop_before_accepting_commands(monkeypatch, tmp_path, held):
    from cascade.apps import demo
    from cascade.apps.live_control import LiveViewController, MODE_LAZY
    from test_live_view import _FakeServer

    rt = runtime()
    rt.stream_server = None
    rt.live_view = LiveViewController(_FakeServer, mode=MODE_LAZY, idle_timeout_s=0)
    from cascade.config import Cfg
    cfg = Cfg({"llm": {}, "memory": {
        "experience_path": str(tmp_path / "experience.json"),
        "skill_library_path": str(tmp_path / "library"),
    }})
    monkeypatch.setenv("CASCADE_START_STOPPED", "1")
    if held:
        monkeypatch.setenv("CASCADE_START_HOLDING", held)
    else:
        monkeypatch.delenv("CASCADE_START_HOLDING", raising=False)
    monkeypatch.setattr(demo, "load_demo_config", lambda **kwargs: cfg)
    monkeypatch.setattr(demo, "build_runtime", lambda *args, **kwargs: (rt, rt.arm))
    monkeypatch.setattr(demo, "make_llm", lambda cfg: MockLLM([]))
    monkeypatch.setattr(demo, "shutdown_runtime", Mock())
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    def prompt(text):
        rt.arm.stop.assert_called_once()
        assert getattr(rt, "_held_provisional", None) is None
        for _ in range(2):
            rt.live_view.open()
            view = rt.live_view.server
            assert view.task_cancel_aware is True
            assert view.cancel_fn.__self__.runtime is rt
            assert view.resume_fn.__self__.runtime is rt
            rt.live_view.close()
        return ""

    monkeypatch.setattr("builtins.input", prompt)
    assert demo.main(["--interactive", "--llm", "mock", "--no-view", "--no-serve"]) == 0


def resume_runtime():
    rt = runtime()
    rt._show_status = Mock()
    rt.arm.raw.check_resume_ready = Mock()
    return rt


def test_resume_checks_readiness_and_only_dispatches_home():
    rt = resume_runtime()
    llm = MockLLM([])
    orchestrator = agent(llm, rt)
    result = orchestrator.resume()
    assert result['resumed']
    rt.arm.raw.check_resume_ready.assert_called_once()
    rt.arm.raw.resume.assert_called_once()
    rt.arm.harness.reset_estop.assert_called_once()
    assert rt.calls == [('move_home', {})] and not llm.requests
    rt.arm.set_gripper.assert_not_called()
    assert orchestrator._run_lock.acquire(blocking=False)
    orchestrator._run_lock.release()


@pytest.mark.parametrize("new_halt", [False, True])
def test_dashboard_resume_clears_superseded_halt_but_preserves_new_hazard(new_halt):
    from cascade.safety.harness import SafetyHarness
    from test_safety import limits

    rt = resume_runtime()
    harness = SafetyHarness(limits())
    harness.halt("superseded")
    harness.estop("operator stop")
    rt.arm.harness = harness
    rt.arm.stop.side_effect = lambda: harness.estop("stop requested")
    if new_halt:
        rt.arm.raw.check_resume_ready.side_effect = lambda: harness.halt("new hazard")
    orchestrator = agent(MockLLM([]), rt)
    if new_halt:
        with pytest.raises(Exception, match="HALT requested during resume"):
            orchestrator.resume()
        assert harness.estopped and harness.halted == "new hazard"
        assert rt.calls == []
    else:
        assert orchestrator.resume()["resumed"]
        assert not harness.estopped and harness.halted is None
        assert rt.calls == [("move_home", {})]


def test_resume_refuses_running_repl_task_and_keeps_stop_on_readiness_failure():
    rt = resume_runtime()
    orchestrator = agent(MockLLM([]), rt)
    orchestrator._run_lock.acquire()
    try:
        with pytest.raises(TaskBusyError):
            orchestrator.resume()
    finally:
        orchestrator._run_lock.release()
    rt.arm.raw.check_resume_ready.assert_not_called()
    rt.arm.raw.check_resume_ready.side_effect = RuntimeError('controller inactive')
    with pytest.raises(RuntimeError, match='controller inactive'):
        orchestrator.resume()
    rt.arm.raw.resume.assert_not_called()
    rt.arm.harness.reset_estop.assert_not_called()


def test_stop_during_resume_readiness_check_wins_and_blocks_new_tasks():
    entered, release = threading.Event(), threading.Event()
    rt = resume_runtime()
    def check():
        entered.set()
        assert release.wait(3)
    rt.arm.raw.check_resume_ready.side_effect = check
    orchestrator = agent(MockLLM([]), rt)
    server = StreamServer(SimpleNamespace(stats=lambda: {}))
    server.set_resume_fn(orchestrator.resume)
    server.set_cancel_fn(orchestrator.cancel)
    results = []
    worker = threading.Thread(target=lambda: results.append(server.resume_task()))
    worker.start()
    try:
        assert entered.wait(1)
        assert not server.submit_task('pick ball')['accepted']
        assert not server.resume_task()['resumed']
        assert server.cancel_task()['cancelled']
        rt.arm.stop.assert_called_once()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not results[0]['resumed'] and 'STOP requested' in results[0]['error']
    rt.arm.raw.resume.assert_not_called()
    rt.arm.harness.reset_estop.assert_not_called()
    assert not server.command_state()['busy']


def test_resume_http_endpoint_and_button():
    from urllib.error import HTTPError
    rt = resume_runtime()
    orchestrator = agent(MockLLM([]), rt)
    server = StreamServer(SimpleNamespace(stats=lambda: {}, names=[]), port=0)
    server.set_resume_fn(orchestrator.resume)
    server.start()
    base = f'http://{loopback_host()}:{server.port}'
    def post():
        return urllib.request.urlopen(urllib.request.Request(base+'/resume', data=b'', method='POST'), timeout=2)
    try:
        html = urllib.request.urlopen(base, timeout=2).read().decode()
        assert 'id="resumebtn"' in html and "fetch('/resume'" in html
        assert 'Resume &amp; Home' in html
        assert server.command_state()['resume_supported']
        assert json.load(post())['resumed']
        rt.arm.raw.check_resume_ready.side_effect = RuntimeError('stale feedback')
        with pytest.raises(HTTPError) as exc:
            post()
        assert exc.value.code == 409
        assert json.load(exc.value)['error'] == 'stale feedback'
        assert not server.command_state()['busy']
        rt.arm.raw.resume.assert_called_once()
        assert rt.calls == [('move_home', {})]
    finally:
        server.stop()


def test_resume_cannot_overtake_a_stop_already_in_progress():
    entered, release = threading.Event(), threading.Event()
    rt = resume_runtime()
    def stop():
        entered.set()
        assert release.wait(3)
    rt.arm.stop.side_effect = stop
    orchestrator = agent(MockLLM([]), rt)
    worker = threading.Thread(target=orchestrator.cancel)
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match='STOP is still in progress'):
            orchestrator.resume()
        rt.arm.raw.resume.assert_not_called()
    finally:
        release.set()
        worker.join(3)
    assert orchestrator.resume()['resumed']


def test_resume_home_failure_relatches_stop():
    rt = resume_runtime()
    rt.execute = Mock(return_value={'ok': False, 'error': 'obstacle on home path'})
    orchestrator = agent(MockLLM([]), rt)
    with pytest.raises(RuntimeError, match='obstacle on home path'):
        orchestrator.resume()
    rt.arm.stop.assert_called_once()
    rt.execute.assert_called_once_with('move_home', {})
    assert orchestrator._run_lock.acquire(blocking=False)
    orchestrator._run_lock.release()


def test_stop_during_resume_home_wins_and_blocks_overlapping_commands():
    entered, release = threading.Event(), threading.Event()
    rt = resume_runtime()
    orchestrator = agent(MockLLM([]), rt)
    old_cancel = threading.Event()
    old_cancel.set()
    orchestrator._cancel_event = old_cancel

    def home(name, args):
        assert name == 'move_home'
        assert not orchestrator.cancellation_requested()
        entered.set()
        assert release.wait(3)
        assert orchestrator.cancellation_requested()
        return {'ok': True}  # STOP wins even if motion reports success late

    rt.execute = home
    server = StreamServer(SimpleNamespace(stats=lambda: {}))
    server.set_resume_fn(orchestrator.resume)
    server.set_cancel_fn(orchestrator.cancel)
    results = []
    worker = threading.Thread(target=lambda: results.append(server.resume_task()))
    worker.start()
    try:
        assert entered.wait(1)
        assert not server.submit_task('pick ball')['accepted']
        assert not server.resume_task()['resumed']
        assert server.cancel_task()['cancelled']
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not results[0]['resumed']
    assert 'STOP requested' in results[0]['error']
    assert not server.command_state()['busy']
    rt.arm.set_gripper.assert_not_called()
