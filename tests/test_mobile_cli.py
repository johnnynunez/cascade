"""Mobile CLI and orchestrator use capabilities, not manipulation defaults."""
import json
import os
from pathlib import Path
import subprocess
import sys

from cascade.config import load_demo_config
import test_mobile_frames

frame_endpoint = test_mobile_frames.frame_endpoint

REPO = Path(__file__).resolve().parents[1]


def test_cli_base_mock_observation_real_subprocess(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(REPO / "src"))
    result = subprocess.run([sys.executable, "-m", "cascade.apps.demo", "--base", "microduck_mock",
                             "--llm", "mock", "--no-view", "--no-serve", "--run-dir", str(tmp_path),
                             "--task", "observe the mobile base"], cwd=REPO, env=env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [r["skill"] for r in records] == ["get_observation", "task_done"]
    assert records[0]["result"]["state"]["measurement_kind"] == "kinematic_mock"
    assert "arms=" not in result.stdout


def test_orchestrator_cannot_count_unverified_motion_or_call_arm_habits(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.agent.llm import MockLLM, LLMResponse, ToolCall

    class ArmHabit:
        def plan(self, *args):
            raise AssertionError("arm reflex/habit invoked for a base")

    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .06})]),
        LLMResponse(tool_calls=[ToolCall("task_done", {"success": True, "summary": "claimed physical success"})]),
    ])
    rt, _ = build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path)
    try:
        agent = AgentOrchestrator(llm, rt, decompose=False, fast_planner=ArmHabit())
        result = agent.run_task("walk briefly")
        assert result.success is False
        assert result.unverified
        request = llm.requests[0]
        assert "walk_velocity" in {t["name"] for t in request["tools"]}
        assert "grasp_object" not in {t["name"] for t in request["tools"]}
        assert "base-only" in request["system"]
        assert "unverified" in (tmp_path / "summary.txt").read_text()
    finally:
        rt.close()


def test_mobile_cli_interactive_eof_stops_inflight_motion(tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from cascade.apps import mobile_runtime

    rt, rig = mobile_runtime.build_mobile_runtime(load_demo_config(base="microduck_mock"), tmp_path)
    read_fd, write_fd = os.pipe()
    results = []
    def task(_text):
        out = rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": 3.0})
        results.append(out)
        return out
    stream = os.fdopen(read_fd)
    try:
        assert hasattr(mobile_runtime, "run_mobile_interactive")
        with ThreadPoolExecutor() as pool:
            run = pool.submit(mobile_runtime.run_mobile_interactive, rt, task, lambda _: None, input_stream=stream)
            os.write(write_fd, b"walk now\n")
            deadline = time.monotonic() + 2
            while not (rig.primary.raw.connected and rig.primary.get_state().controller_status == "active"):
                assert time.monotonic() < deadline
                time.sleep(.005)
            started = time.monotonic()
            os.close(write_fd)
            write_fd = None
            run.result(1)
            assert time.monotonic() - started < 1
        assert results and not results[-1]["execution_ok"]
        assert rig.primary.latched
        assert not any(t.name == "mobile-cli-input" and t.is_alive() for t in threading.enumerate())
    finally:
        if write_fd is not None:
            os.close(write_fd)
        stream.close()
        rt.close()


def test_mobile_camera_is_json_safe_through_real_orchestrator(tmp_path, frame_endpoint):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.agent.llm import MockLLM, LLMResponse, ToolCall
    from test_mobile_runtime import camera_cfg
    _, _, profile, _, jpeg, _ = frame_endpoint
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("get_observation", {})]),
        LLMResponse(tool_calls=[ToolCall("task_memory", {"k": 1})]),
        LLMResponse(tool_calls=[ToolCall("task_done", {"success": True, "summary": "observed, no motion claim"})]),
    ])
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    try:
        result = AgentOrchestrator(llm, rt, decompose=False).run_task("inspect the base camera")
        assert result.success
        assert rt.memory.memory_frames(1)[0]["jpeg"] == jpeg
        json.dumps(rt.execute("camera_snapshot", {}))
        json.dumps(rt.execute("task_memory", {}))
    finally:
        rt.close()
