"""VAB protocol and real CASCADE dispatch tests; the environment is synthetic."""
from dataclasses import replace
import json
import subprocess

import numpy as np
import pytest
import yaml

from cascade.eval.trials import verify_episode
from cascade.eval.vab import VabLiberoView, flatten_observation, inspect_trial, run_trial
from cascade.robotics.contracts import ToolDescriptor
from cascade.robotics.runtime import RobotRuntime


def git(root, *args):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def trial(tmp_path):
    checkout = tmp_path / "upstream-contract-fixture"
    checkout.mkdir()
    git(checkout, "init", "-q")
    task = checkout / "tasks" / "unit.yaml"
    task.parent.mkdir()
    task.write_text(yaml.safe_dump({"id": "unit.contract", "language": "put cube in basket",
        "robot": {"name": "panda", "controller": "OSC_POSE"}, "arena": {"name": "floor"},
        "objects": [{"id": "cube", "asset": "cube"}, {"id": "basket", "asset": "basket"}],
        "inits": [{"cube": [0., 0., 0.1, 0., 0., 0., 1.], "basket": [0.1, 0., 0., 0., 0., 0., 1.]}],
        "horizon": 3, "camera_depth": False,
        "success": {"predicate": "contained_in", "args": {"obj": "cube", "container": "basket"}}}))
    git(checkout, "add", "tasks")
    git(checkout, "-c", "user.name=Contract Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "synthetic VAB schema fixture")
    return inspect_trial(checkout, "tasks/unit.yaml", source_revision=git(checkout, "rev-parse", "HEAD"))


def observation():
    return {"images": {"agentview": np.zeros((8, 8, 3), dtype=np.uint8)},
            "proprio": {"joint_pos": np.zeros(7), "joint_vel": np.zeros(7),
                        "eef_pos": np.zeros(3), "eef_quat": np.array([0., 0., 0., 1.]),
                        "gripper_qpos": np.zeros(2)}}


class ContractEnvironment:
    action_dim = 7
    action_spec = (-np.ones(7), np.ones(7))

    def __init__(self):
        self.actions = []
        self.closed = False

    def reset(self, *, init_index):
        assert init_index == 0
        return observation()

    def step(self, action):
        self.actions.append(action.copy())
        return observation(), 0., False, {"success": True}

    def close(self):
        self.closed = True


def test_inspection_is_source_bound_and_never_imports_simulator(trial):
    assert trial.horizon == 3 and not trial.camera_depth
    assert trial.new_trial_id() != trial.new_trial_id()
    with pytest.raises(ValueError, match="revision"):
        inspect_trial(trial.checkout, trial.task_path, source_revision="0" * 40)
    with pytest.raises(ValueError, match="init_index"):
        inspect_trial(trial.checkout, trial.task_path, source_revision=trial.source_revision, init_index=-1)
    trial.task_path.write_text(trial.task_path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="differs"):
        inspect_trial(trial.checkout, trial.task_path, source_revision=trial.source_revision)


@pytest.mark.parametrize("predicate", ["pack_all_into", "lifted_above", "on_top_of"])
def test_mutating_packing_and_partial_multistage_tasks_rejected(trial, predicate):
    data = yaml.safe_load(trial.task_path.read_text())
    data["success"]["predicate"] = predicate
    trial.task_path.write_text(yaml.safe_dump(data))
    git(trial.checkout, "add", "tasks")
    git(trial.checkout, "-c", "user.name=Contract Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "unsupported predicate fixture")
    with pytest.raises(ValueError, match="excludes"):
        inspect_trial(trial.checkout, trial.task_path, source_revision=git(trial.checkout, "rev-parse", "HEAD"))


def test_projection_preserves_only_public_observations_and_copies():
    raw = observation()
    flat = flatten_observation(raw)
    assert set(flat) == {"agentview_image", "robot0_joint_pos", "robot0_joint_vel",
                         "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
    raw["proprio"]["joint_pos"][0] = 1
    assert flat["robot0_joint_pos"][0] == 0
    raw["object_poses"] = {}
    with pytest.raises(ValueError, match="strict"):
        flatten_observation(raw)


def test_raw_done_is_not_success_and_budget_cannot_be_ignored(trial):
    native = ContractEnvironment()
    view = VabLiberoView(native, trial)
    view.reset()
    for i in range(3):
        _, _, done, info = view.step(np.zeros(7))
        assert done is (i == 2)
        assert info["success"] is True  # kept separate from the horizon flag
    with pytest.raises(RuntimeError, match="budget"):
        view.step(np.zeros(7))
    with pytest.raises(RuntimeError, match="fresh"):
        view.reset()
    assert len(native.actions) == 3


@pytest.mark.parametrize("action", [np.ones(8), np.full(7, np.nan), np.full(7, 1.01)])
def test_invalid_actions_never_reach_environment(trial, action):
    native = ContractEnvironment()
    view = VabLiberoView(native, trial)
    view.reset()
    with pytest.raises(ValueError):
        view.step(action)
    assert native.actions == []


class ContractDomain:
    """Synthetic controller only: exercises actual RobotRuntime admission."""
    domain_id = "fixture"
    resources = ()
    tool_descriptors = (ToolDescriptor("fixture.tick", "Advance protocol fixture",
        {"type": "object", "properties": {}}, "fixture", "tick", effect="control"),)

    def __init__(self, view):
        self.view = view

    def execute(self, name, args):
        assert name == "tick" and args == {}
        self.view.step(np.zeros(7))
        return {"ok": True}

    def begin_task(self):
        pass

    def stop(self):
        return {"ok": True}

    def close(self):
        return {"ok": True}


def test_runner_uses_real_robot_runtime_and_does_not_admit_synthetic_success(trial, tmp_path):
    native = ContractEnvironment()
    def factory(view, selected, trial_id):
        assert selected == trial and trial_id.startswith("vab:")
        runtime = RobotRuntime({"fixture": ContractDomain(view)})
        return runtime, runtime.close
    path = tmp_path / "receipt.json"
    episode = run_trial(trial, runtime_factory=factory, skill_plan=[("fixture.tick", {})],
                        output_path=path, env_factory=lambda _: native)
    assert native.closed and len(native.actions) == 1
    assert episode.benchmark_success is True
    assert json.loads(path.read_text())["tools"] == [{"name": "fixture.tick", "execution_ok": True}]
    assert verify_episode(episode, required_checks=frozenset({"contact"})).status == "unverified"
    with pytest.raises(FileExistsError):
        run_trial(trial, runtime_factory=factory, skill_plan=[], output_path=path,
                  env_factory=lambda _: pytest.fail("existing receipt must prevent allocation"))


def test_init_failure_saves_terminal_error_receipt(trial, tmp_path):
    path = tmp_path / "failed.json"
    def fail(_):
        raise RuntimeError("native init failed")
    with pytest.raises(RuntimeError, match="init failed"):
        run_trial(trial, runtime_factory=None, skill_plan=[], output_path=path, env_factory=fail)
    assert json.loads(path.read_text())["status"] == "failed"


def test_bound_configuration_does_not_float_across_episode(trial, tmp_path):
    native = ContractEnvironment()
    def factory(view, *_):
        runtime = RobotRuntime({"fixture": ContractDomain(view)})
        return runtime, runtime.close
    from test_arena_adapters import binding
    with pytest.raises(ValueError, match="observer binding"):
        run_trial(trial, runtime_factory=factory, skill_plan=[("fixture.tick", {})],
                  output_path=tmp_path / "bad-binding.json", env_factory=lambda _: native,
                  binding_reader=lambda trial_id, *_: replace(binding(trial_id), configuration_sha256="f" * 64))
    assert native.closed
