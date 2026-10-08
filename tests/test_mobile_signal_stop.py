"""Real entrypoint/signal regressions; CPU fixtures are NOT physical stops."""
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


def run_child(tmp_path, entrypoint, site="gate", signum=signal.SIGINT):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
               CASCADE_BASE="microduck_mock", CASCADE_LLM="mock", CASCADE_PREWARM="0",
               CASCADE_STREAM="0", CASCADE_VIEW="0", CASCADE_BELIEFS="0",
               CASCADE_BELIEFS_PATH=str(tmp_path / "beliefs.json"),
               CASCADE_GRASP_MEMORY_PATH=str(tmp_path / "grasp.json"),
               CASCADE_ENVELOPE_PATH=str(tmp_path / "envelope.json"),
               CASCADE_RUN_DIR=str(tmp_path / "run"))
    if entrypoint == "mcp_arm":
        env.pop("CASCADE_BASE")
    command = [sys.executable, str(Path(__file__).resolve()), "--child", entrypoint,
               site, str(int(signum)), str(tmp_path)]
    notify_read, notify_write = os.pipe()
    env["SIGNAL_TEST_NOTIFY_FD"] = str(notify_write)
    child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             pass_fds=(notify_write,))
    os.close(notify_write)
    # Keep stdin OPEN: MCP must exit on a signal, not on an incidental EOF.
    if entrypoint.startswith("mcp"):
        child.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                      "params": {"name": "walk_velocity", "arguments": {
                                          "vx": .05, "vy": 0., "wz": 0., "duration_s": .04}}}) + "\n")
        child.stdin.flush()
    try:
        if site == "io":
            import select
            import time
            assert select.select([notify_read], [], [], 3)[0], "child never reached blocking IO"
            assert os.read(notify_read, 1) == b"I"
            # Linux kernel confirmation, not an arbitrary sleep: main thread
            # really is blocked in read(2) with the backend lock held.
            deadline = time.monotonic() + 2
            while "pipe_read" not in Path(f"/proc/{child.pid}/wchan").read_text():
                assert time.monotonic() < deadline, "main never entered pipe read"
                os.sched_yield()
            child.send_signal(signum)
        child.wait(timeout=4)
    except subprocess.TimeoutExpired:
        child.kill()
        stdout, stderr = child.communicate(timeout=2)
        pytest.fail(f"signal deadlock (owned pid {child.pid} killed/reaped):\n{stdout}\n{stderr}")
    finally:
        os.close(notify_read)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
    stdout, stderr = child.communicate(timeout=2)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["sent"], (stdout, stderr, receipt)
    assert receipt["stops"] + receipt["server_stops"] > 0, (stdout, stderr, receipt)
    assert not receipt["handler_stop"], receipt
    assert receipt["closed"], receipt
    assert receipt["handlers_restored"], receipt
    assert not receipt["owned_threads"], receipt
    assert not receipt["new_fds"], receipt
    return child.returncode, receipt, stdout, stderr


@pytest.mark.parametrize("entrypoint", ["demo", "mcp"])
def test_signal_inside_real_motion_gate_unwinds_before_stopping(tmp_path, entrypoint):
    code, receipt, _, _ = run_child(tmp_path, entrypoint)
    assert code == 130
    assert receipt["dispatches_after_signal"] == 0


@pytest.mark.parametrize("entrypoint", ["demo", "mcp"])
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("site", ["startup", "safe_gate", "backend_lock", "inflight_read", "record_lock"])
def test_signals_at_startup_and_nested_locks(tmp_path, entrypoint, signum, site):
    code, receipt, _, _ = run_child(tmp_path, entrypoint, site, signum)
    assert code == 128 + signum
    assert receipt["dispatches_after_signal"] == 0
    if site == "inflight_read":
        assert receipt["dispatches"] > 0  # cannot go green before dispatch


@pytest.mark.parametrize("site", ["stop_lock", "cancel_lock", "out_lock"])
def test_mcp_signal_unwinds_its_own_locks(tmp_path, site):
    code, _, _, _ = run_child(tmp_path, "mcp", site)
    assert code == 130


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_default_arm_mcp_signals_unwind_worker_lock(tmp_path, signum):
    code, receipt, _, _ = run_child(tmp_path, "mcp_arm", "cancel_lock", signum)
    assert code == 128 + signum
    assert receipt["runtime_count"] == 0  # no accidental mobile/arm startup


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="kernel wchan proof uses Linux procfs")
@pytest.mark.parametrize("entrypoint", ["demo", "mcp"])
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_interrupts_real_blocked_io_under_backend_lock(tmp_path, entrypoint, signum):
    code, receipt, _, _ = run_child(tmp_path, entrypoint, "io", signum)
    assert code == 128 + signum
    assert receipt["io_entered"]
    assert receipt["dispatches_after_signal"] == 0


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_default_arm_cli_signal_during_brain_setup_cleans_runtime(tmp_path, signum):
    # This boundary used to lie outside demo.main's try/finally. Use the
    # REAL default arm/camera stack, not a fake substitute for the entrypoint.
    code = r'''
import atexit, json, os, signal, sys, threading
from pathlib import Path
from cascade.apps import demo
from cascade.safety.harness import SafetyHarness
state = {"halted": False, "cleaned": False, "in_handler": False}
originals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
old_halt = SafetyHarness.halt
old_shutdown = demo.shutdown_runtime
def halt(self, *args, **kwargs):
    import inspect
    state["in_handler"] |= any(f.function in {"_sigint", "_handle"} for f in inspect.stack())
    state["halted"] = True
    return old_halt(self, *args, **kwargs)
def shutdown(runtime, arm):
    old_shutdown(runtime, arm)
    state["cleaned"] = True
    state["threads"] = [t.name for _, t in demo.owned_threads(runtime) if t.is_alive()]
def save():
    state["restored"] = all(signal.getsignal(s) == old for s, old in originals.items())
    Path(sys.argv[1], "arm-receipt.json").write_text(json.dumps(state))
atexit.register(save)
SafetyHarness.halt = halt
demo.shutdown_runtime = shutdown
demo.make_llm = lambda _: os.kill(os.getpid(), int(sys.argv[2]))
raise SystemExit(demo.main(["--arm", "mock", "--camera", "mock", "--llm", "mock", "--no-view",
                           "--no-serve", "--run-dir", sys.argv[1]]))
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", PYTHONPATH=str(REPO / "src"),
               CASCADE_BELIEFS="0", CASCADE_BELIEFS_PATH=str(tmp_path / "beliefs.json"),
               CASCADE_GRASP_MEMORY_PATH=str(tmp_path / "grasp.json"),
               CASCADE_ENVELOPE_PATH=str(tmp_path / "envelope.json"))
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(int(signum))],
                            env=env, cwd=REPO, capture_output=True, text=True, timeout=8)
    receipt = json.loads((tmp_path / "arm-receipt.json").read_text())
    assert result.returncode == (0 if signum == signal.SIGINT else 143), result.stderr
    assert receipt == {"halted": True, "cleaned": True, "in_handler": False, "threads": [], "restored": True}


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_pending_shutdown_cannot_be_overwritten_by_staff_reset(signum):
    code = '''
import os, signal, sys
from cascade.apps.signal_stop import SignalRequest, StopSignals
with StopSignals(staff_reset=True) as signals:
    with signals.defer():
        os.kill(os.getpid(), int(sys.argv[1]))
        os.kill(os.getpid(), signal.SIGUSR1)
    try:
        signals.checkpoint()
    except SignalRequest as request:
        assert request.signum == int(sys.argv[1]), request.signum
    else:
        raise AssertionError("lost shutdown request")
'''
    result = subprocess.run([sys.executable, "-c", code, str(int(signum))], cwd=REPO,
                            env={**os.environ, "PYTHONPATH": str(REPO / "src"), "CUDA_VISIBLE_DEVICES": "-1",
                                 "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=4)
    assert result.returncode == 0, result.stderr


def _child(entrypoint, site, signum, directory):
    import atexit
    import faulthandler
    import inspect
    import threading
    from cascade.apps import demo, mcp_server
    from cascade.agent.llm import MockLLM, LLMResponse, ToolCall
    from cascade.skills.mobile_runtime import MobileSkillRuntime
    from cascade.control.mock_base import MockMobileBase
    from cascade.safety.base_harness import SafeBase

    directory = Path(directory)
    state = {"sent": False, "stops": 0, "server_stops": 0, "handler_stop": False,
             "dispatches_after_signal": 0, "dispatches": 0}
    runtimes = []

    def fds():
        result = {}
        if sys.platform.startswith("linux"):
            for name in os.listdir("/proc/self/fd"):
                try:
                    result[name] = os.readlink("/proc/self/fd/" + name)
                except FileNotFoundError:
                    pass
        return result

    initial_fds = fds()
    originals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    real_build = demo.build_runtime
    real_stop = MobileSkillRuntime.stop
    real_dispatch = MockMobileBase.command_velocity
    real_server_stop = mcp_server.McpSkillServer.stop_now
    real_get_state = MockMobileBase.get_state

    def blocking_get_state(self):
        if not state["sent"]:
            rfd, wfd = os.pipe()
            try:
                with self._lock:
                    state["sent"] = state["io_entered"] = True
                    os.write(int(os.environ["SIGNAL_TEST_NOTIFY_FD"]), b"I")
                    os.read(rfd, 1)
            finally:
                os.close(rfd)
                os.close(wfd)
        return real_get_state(self)

    if site == "io":
        MockMobileBase.get_state = blocking_get_state

    def send():
        state["sent"] = True
        faulthandler.dump_traceback_later(1)
        os.kill(os.getpid(), signum)

    def build(*args, **kwargs):
        if site == "startup" and not state["sent"]:
            send()
        runtime, arm = real_build(*args, **kwargs)
        runtimes.append(runtime)
        return runtime, arm

    def stop(self, *args, **kwargs):
        state["handler_stop"] |= any(f.function in {"_sigint", "_handle"} for f in inspect.stack())
        result = real_stop(self, *args, **kwargs)
        state["stops"] += 1
        sys.settrace(trace)
        return result

    def server_stop(self, *args, **kwargs):
        state["handler_stop"] |= any(f.function in {"_sigint", "_handle"} for f in inspect.stack())
        result = real_server_stop(self, *args, **kwargs)
        state["server_stops"] += 1
        sys.settrace(trace)
        return result

    def dispatch(self, *args, **kwargs):
        state["dispatches"] += 1
        if state["sent"]:
            state["dispatches_after_signal"] += 1
        return real_dispatch(self, *args, **kwargs)

    demo.build_runtime = build
    MobileSkillRuntime.stop = stop
    mcp_server.McpSkillServer.stop_now = server_stop
    MockMobileBase.command_velocity = dispatch
    demo.make_llm = lambda cfg: MockLLM([
        LLMResponse(tool_calls=[ToolCall("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .04})]),
        LLMResponse(tool_calls=[ToolCall("task_done", {"success": False, "summary": "CPU fixture"})]),
    ])
    # Locate a with BODY by AST/function/context, never stale line numbers.
    cls, function, lock = {
        "gate": (MobileSkillRuntime, "_motion", "self._gate"),
        "io": (MobileSkillRuntime, "_motion", "self._gate"),
        "startup": (MobileSkillRuntime, "_motion", "self._gate"),
        "safe_gate": (SafeBase, "_execute", "self._gate"),
        "backend_lock": (MockMobileBase, "get_state", "self._lock"),
        "inflight_read": (MockMobileBase, "get_state", "self._lock"),
        "record_lock": (MobileSkillRuntime, "execute", "self._record_lock"),
        # The serial worker moved into _worker_loop (shared by the stdio and
        # Streamable HTTP transports); these are the worker's own gates.
        "stop_lock": (mcp_server, "_worker_loop", "server._stop_lock"),
        "cancel_lock": (mcp_server, "_worker_loop", "server._cancel_lock"),
        "out_lock": (mcp_server, "_send", "out_lock"),
    }[site]
    source = Path(inspect.getsourcefile(cls))
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    candidates = [n for n in ast.walk(method) if isinstance(n, ast.With)
                  and any(ast.unparse(i.context_expr) == lock for i in n.items)]
    # Nested reader functions run in another thread. Select the worker's gate.
    if site in {"stop_lock", "cancel_lock"}:
        nested = [n for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n is not method]
        candidates = [n for n in candidates if not any(f.lineno <= n.lineno <= f.end_lineno for f in nested)]
        candidates.sort(key=lambda n: n.lineno)
    target_line = candidates[0].body[0].lineno
    state["target"] = {"file": str(source), "function": function, "line": target_line, "lock": lock}

    def trace(frame, event, arg):
        if event != "line":
            return trace
        if (site != "io" and not state["sent"] and frame.f_code.co_filename == str(source)
                and frame.f_code.co_name == function and frame.f_lineno == target_line
                and (site != "inflight_read" or state["dispatches"] > 0)):
            if "." in lock:
                owner, attr = lock.split(".")
                held = getattr(frame.f_locals[owner], attr)
            else:
                held = frame.f_locals[lock]
            assert held.locked() if hasattr(held, "locked") else held._is_owned()
            send()
        # The first MCP Ctrl+C must freeze, not exit. Once it returned to
        # its idle queue (stdin still open), the second must exit/clean up.
        if (entrypoint.startswith("mcp") and signum == signal.SIGINT and state["sent"]
                and (state["stops"] or state["server_stops"])
                and frame.f_code.co_name == "get" and frame.f_globals.get("__name__") == "queue"):
            state["second_sent"] = True
            sys.settrace(None)
            os.kill(os.getpid(), signal.SIGINT)
        return trace

    def receipt():
        state["closed"] = all(rt._closed for rt in runtimes)
        state["runtime_count"] = len(runtimes)
        state["handlers_restored"] = all(signal.getsignal(s) == old for s, old in originals.items())
        state["owned_threads"] = [t.name for t in threading.enumerate()
                                  if t.name.startswith(("wrc-", "mobile-", "signal-"))]
        state["new_fds"] = {k: v for k, v in fds().items() if initial_fds.get(k) != v}
        (directory / "receipt.json").write_text(json.dumps(state))
    atexit.register(receipt)
    sys.settrace(trace)
    if entrypoint == "demo":
        raise SystemExit(demo.main(["--base", "microduck_mock", "--llm", "mock", "--no-view", "--no-serve",
                                   "--max-steps", "2", "--run-dir", str(directory / "run"),
                                   "--task", "walk CPU fixture"]))
    sys.argv = ["cascade-mcp"]
    raise SystemExit(mcp_server.main())


if __name__ == "__main__" and sys.argv[1] == "--child":
    _child(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
