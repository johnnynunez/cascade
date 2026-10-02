"""Software consumer-fence regressions; these fixtures do not prove physics."""
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.arena import ArenaPolicyAdapter
from cascade.robotics.contracts import ResourceDescriptor
from cascade.robotics.runtime import RobotRuntime


spec = importlib.util.spec_from_file_location(
    "arena_native_owner", Path(__file__).resolve().parents[1] / "benchmark/arena/native_owner.py")
owner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner_module)


@pytest.fixture
def setup():
    class Env:
        steps = 0
        terminal = False

        def step(self, action):
            self.steps += 1
            return {}, 0, np.array([self.terminal]), np.array([False]), {}

    env = Env()
    policy = SimpleNamespace(get_action=lambda *_: np.zeros((1, 7)),
                             reset=lambda _: None, close=lambda: None)
    controller = owner_module.PreflightController(policy)
    resource = ResourceDescriptor("arena/franka", "actuator", "arena_franka",
        controller_id="native/env0", writer_id="native/zero_policy")

    def validate(action):
        if action.shape != (1, 7) or not np.isfinite(action).all() or action.any():
            raise ValueError("expected exact upstream zero action")

    domain = owner_module.NativePreflightDomain(controller, command_resources=(resource,),
        env=env, observation={}, observe=lambda: {"step": env.steps},
        validate_action=validate, deadline_monotonic_s=time.monotonic() + 30)
    runtime = RobotRuntime({"arena_policy": domain})
    adapter = ArenaPolicyAdapter(runtime, controller, actuation_owner="arena_policy")
    domain.bind_adapter(adapter)
    yield env, policy, controller, domain, runtime
    runtime.close()


def run(runtime, steps=20):
    return runtime.execute("arena_policy.run_policy_steps", {"steps": steps})


def test_ordinary_runtime_does_not_promote_foundation_to_task_success(setup):
    env, _, _, domain, runtime = setup
    result = run(runtime)
    assert result["ok"] and result["completed_steps"] == env.steps == 20
    assert [sample["step"] for sample in domain.samples] == list(range(21))
    assert result["postcondition"]["status"] == "unverified"
    assert not runtime.execute("task_done", {"success": True, "summary": "foundation"})["success"]
    assert not run(runtime)["ok"] and env.steps == 20


@pytest.mark.parametrize("steps", [0, 21, True, 1.5])
def test_invalid_count_never_reaches_native_consumer(setup, steps):
    env, _, _, domain, runtime = setup
    assert not run(runtime, steps)["ok"]
    assert env.steps == 0 and not domain.used


def test_stop_after_returned_action_vetoes_final_env_step(setup):
    env, _, _, domain, runtime = setup
    domain.after_action = lambda number: runtime.stop() if number == 4 else None
    result = run(runtime)
    assert not result["ok"]
    assert result["domain_result"]["error"].endswith("cancelled before env.step")
    assert env.steps == domain.completed_steps == 4 and domain.actions_produced == 5
    assert not run(runtime)["ok"] and env.steps == 4
    assert not runtime.reset_stop()["ok"]


def test_deadline_after_returned_action_vetoes_final_env_step(setup):
    env, _, _, domain, runtime = setup
    domain.after_action = lambda _: setattr(domain, "deadline", time.monotonic() - 1)
    result = run(runtime)
    assert not result["ok"] and "deadline expired" in result["error"]
    assert domain.actions_produced == 1 and env.steps == 0


def test_nonzero_or_nonfinite_action_cannot_enter_native_step(setup):
    env, policy, _, _, runtime = setup
    policy.get_action = lambda *_: np.full((1, 7), np.nan)
    assert not run(runtime)["ok"] and env.steps == 0


def test_stop_while_computing_rejects_returned_action(setup):
    env, policy, _, domain, runtime = setup
    policy.get_action = lambda *_: (runtime.stop(), np.zeros((1, 7)))[1]
    assert not run(runtime)["ok"] and env.steps == 0
    assert domain.actions_produced == 0


def test_terminal_step_never_continues_after_possible_upstream_auto_reset(setup):
    env, _, _, domain, runtime = setup
    env.terminal = True
    result = run(runtime)
    assert not result["ok"] and "auto-reset" in result["error"]
    assert domain.completed_steps == env.steps == 1


def test_close_keeps_native_work_owned_until_admitted_step_returns(setup):
    env, _, controller, domain, runtime = setup
    entered, release = threading.Event(), threading.Event()
    original = env.step

    def blocked(action):
        entered.set()
        assert release.wait(5)
        return original(action)

    env.step = blocked
    result = []
    worker = threading.Thread(target=lambda: result.append(run(runtime)))
    worker.start()
    try:
        assert entered.wait(2)
        assert domain.close() == {"ok": False, "pending": True,
                                 "error": "Arena action still owns controller"}
        assert not controller.closed
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not result[0]["ok"] and env.steps == 1
    assert domain.close()["ok"] and controller.closed
