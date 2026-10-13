"""Already-connected feedback must not expose dashboard motion before STOP."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from cascade.apps.mcp_server import McpSkillServer


def startup(monkeypatch, *, stopped=True, during_build=None, dashboard_discovery=None):
    from cascade.apps import demo

    for name in ("CASCADE_BASE", "CASCADE_ROBOT", "CASCADE_ARMS", "CASCADE_CAMERAS",
                 "CASCADE_GROUNDING_LLM_PROFILE", "CASCADE_DETECTOR_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CASCADE_ARM", "mock")
    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_START_STOPPED", "1" if stopped else "0")
    server = McpSkillServer()
    events = []
    harness = SimpleNamespace(estopped=False)
    raw = SimpleNamespace(connected=True, resume=Mock(), send_joint_target=Mock(),
                          set_gripper=Mock(), stream_to=Mock())

    def stop():
        events.append("stop")
        harness.estopped = True

    arm = SimpleNamespace(harness=harness, raw=raw, stop=Mock(side_effect=stop))
    observed = []

    class Dashboard:
        def set_cancel_fn(self, fn):
            self.cancel = fn

        def set_task_fn(self, fn):
            # A request thread can invoke a callback as soon as it is
            # registered. Run that interleaving deterministically.
            events.append("task_available")
            observed.append((harness.estopped, runtime.motion_cancel_requested()))
            if server._stop_pending:
                fn("go home")

        def set_resume_fn(self, fn):
            events.append("resume_available")
            self.resume = fn  # registration must never invoke Resume

    dashboard = Dashboard()

    class Runtime(SimpleNamespace):
        @property
        def live_view(self):
            if dashboard_discovery:
                dashboard_discovery(server)
            return dashboard

    runtime = Runtime(arm=arm, stream_server=None, memory=SimpleNamespace(add=Mock()))

    def execute(name, arguments):
        # Model only the admission boundary; no kinematics, ROS or actuator.
        assert harness.estopped and runtime.motion_cancel_requested()
        return {"ok": False, "error": "e-stop latched"}

    runtime.execute = Mock(side_effect=execute)

    def build(cfg, *args, **kwargs):
        runtime.cfg = cfg
        if during_build:
            during_build(server)
        return runtime, raw

    monkeypatch.setattr(demo, "build_runtime", build)
    monkeypatch.setattr(server, "_attach_program_tier", Mock())
    monkeypatch.setattr(server, "_announce_surface", Mock())
    return server, runtime, raw, events, observed


def assert_no_actuation(raw):
    raw.resume.assert_not_called()
    raw.send_joint_target.assert_not_called()
    raw.set_gripper.assert_not_called()
    raw.stream_to.assert_not_called()


def test_startup_stop_is_applied_before_dashboard_can_submit_motion(monkeypatch):
    server, runtime, raw, events, observed = startup(monkeypatch)
    assert server._ensure_runtime() is runtime
    assert observed == [(True, True)]
    assert events.index("stop") < events.index("task_available") < events.index("resume_available")
    runtime.execute.assert_called_once_with("move_to_rest", {})
    runtime.arm.stop.assert_called_once()
    assert server._stop_pending and runtime.arm.harness.estopped
    assert_no_actuation(raw)


def test_stop_received_while_building_survives_connected_feedback_startup(monkeypatch):
    building, finish = threading.Event(), threading.Event()

    def during_build(server):
        assert server._runtime is None
        building.set()
        assert finish.wait(3)

    server, runtime, raw, events, observed = startup(monkeypatch, stopped=False, during_build=during_build)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(server._ensure_runtime)
        try:
            assert building.wait(3)
            server.stop_now()
            assert server._stop_pending
            assert server._runtime is None
            runtime.arm.stop.assert_not_called()
        finally:
            finish.set()
        assert result.result(timeout=3) is runtime
    assert observed == [(True, True)]
    assert events.index("stop") < events.index("task_available")
    runtime.execute.assert_called_once_with("move_to_rest", {})
    assert_no_actuation(raw)


def test_stop_after_runtime_publication_still_precedes_dashboard_motion(monkeypatch):
    def dashboard_discovery(server):
        assert server._runtime is not None
        assert callable(server._runtime.motion_cancel_requested)
        server.stop_now()

    server, runtime, raw, events, observed = startup(
        monkeypatch, stopped=False, dashboard_discovery=dashboard_discovery,
    )
    assert server._ensure_runtime() is runtime
    assert observed == [(True, True)]
    assert events.index("stop") < events.index("task_available")
    assert_no_actuation(raw)


def test_unstopped_startup_installs_guard_without_stop_resume_or_motion(monkeypatch):
    server, runtime, raw, events, observed = startup(monkeypatch, stopped=False)
    assert server._ensure_runtime() is runtime
    assert observed == [(False, False)]
    assert "stop" not in events
    runtime.execute.assert_not_called()
    assert_no_actuation(raw)
    server.stop_now()
    assert runtime.motion_cancel_requested()  # callback stays live after startup
