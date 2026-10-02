"""CPU contract tests, not native Arena or physical admission evidence.

The fixture follows Arena's record_core_episode_results and
ArenaExperimentResult.to_dict at c8d04e2199b86abbbb301bb22cec0effd83c4e63.
https://github.com/isaac-sim/IsaacLab-Arena/tree/c8d04e2199b86abbbb301bb22cec0effd83c4e63
No simulator-produced episode is claimed by these synthetic records.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import sys
from types import ModuleType

import pytest

from cascade.eval.arena import ArenaControllerDomain, ArenaPolicyAdapter, import_experiment, make_policy_type
from cascade.eval.trials import Artifact, EpisodeBinding, IndependentEvidence, verify_episode
from cascade.robotics.runtime import RobotRuntime
from cascade.robotics.contracts import ResourceDescriptor


def command_resources():
    return (ResourceDescriptor("arena/commands", "actuator", "test-robot",
        controller_id="fixture:arena/commands", writer_id="arena_policy", synthetic=True),)


def arena_owner(controller):
    return ArenaControllerDomain(controller, command_resources=command_resources())


def arena_record():
    return {"runs": {"pick": {"environment": {"name": "cube_goal_pose", "definition": "cube_goal_pose"},
                             "policy_variant": "zero_action", "status": "completed", "rebuilds": [
        {"index": 0, "episodes": [{"job_name": "pick", "env_id": 0, "episode_in_env": 1,
            "seed": 42, "success": True, "episode_length": 20,
            "language_instruction": "move the cube", "timestamp": "2026-10-02T10:00:00",
            "variations": {"cube_mass": 0.05}, "progress": {"completed": 1}}]}]}}}


def write_record(tmp_path, data=None):
    path = tmp_path / "arena_experiment_result.json"
    path.write_text(json.dumps(arena_record() if data is None else data))
    return path


def binding(trial_id):
    return EpisodeBinding(trial_id, "a" * 64, "b" * 64, "epoch-1", 10, 20,
                          0.05, 0.10, "c" * 64, "d" * 64)


def bound_episode(tmp_path):
    path = write_record(tmp_path)
    raw = import_experiment(path)[0]
    return import_experiment(path, bindings={raw.record_id: binding(raw.record_id)},
                             bound_artifact_sha256=raw.artifact.sha256)[0]


def evidence(episode):
    path = Path(episode.artifact.path).with_name("independent-test-observer.json")
    path.write_text('{"scope":"synthetic verifier unit fixture"}\n')
    return IndependentEvidence(episode.binding, episode.artifact.sha256,
                               episode.record_sha256, "native-observer", Artifact.read(path),
                               {"support": "confirmed", "progress": "confirmed"})


def test_arena_success_is_unverified_without_native_binding(tmp_path):
    ep = import_experiment(write_record(tmp_path))[0]
    called = []
    verdict = verify_episode(ep, required_checks=frozenset({"support"}),
                             independent_verifier=lambda _: called.append(True))
    assert ep.benchmark_success is True
    assert verdict.status == "unverified" and not called


def test_bound_independent_verifier_preserves_upstream_result(tmp_path):
    ep = bound_episode(tmp_path)
    assert verify_episode(ep, required_checks=frozenset({"support", "progress"}),
                           independent_verifier=evidence).status == "confirmed"
    assert verify_episode(replace(ep, benchmark_success=False), required_checks=frozenset({"support"}),
                           independent_verifier=evidence).status == "refuted"
    assert verify_episode(replace(ep, benchmark_success=None), required_checks=frozenset({"support"}),
                           independent_verifier=evidence).status == "unverified"


@pytest.mark.parametrize("field,value", [("epoch", "old"), ("trial_id", "other"),
    ("model_identity_sha256", "f" * 64), ("configuration_sha256", "f" * 64),
    ("first_snapshot_sha256", "f" * 64), ("last_snapshot_sha256", "f" * 64),
    ("last_step", 21), ("last_sim_time_s", .11)])
def test_cannot_reuse_verdict_between_episodes_or_snapshots(tmp_path, field, value):
    ep = bound_episode(tmp_path)
    witness = replace(evidence(ep), binding=replace(ep.binding, **{field: value}))
    assert verify_episode(ep, required_checks=frozenset({"support"}),
                           independent_verifier=lambda _: witness).status == "unverified"


@pytest.mark.parametrize("field", ["external_record_sha256", "external_artifact_sha256"])
def test_record_and_file_both_bound(tmp_path, field):
    ep = bound_episode(tmp_path)
    witness = replace(evidence(ep), **{field: "f" * 64})
    assert verify_episode(ep, required_checks=frozenset({"support"}),
                           independent_verifier=lambda _: witness).status == "unverified"


@pytest.mark.parametrize("checks,expected", [({}, "unverified"), ({"support": "confirmed"}, "unverified"),
    ({"support": "confirmed", "progress": "unverified"}, "unverified"),
    ({"support": "confirmed", "progress": "confirmed", "forbidden_contact": "refuted"}, "refuted")])
def test_partial_or_refuted_contact_evidence_never_passes(tmp_path, checks, expected):
    ep = bound_episode(tmp_path)
    assert verify_episode(ep, required_checks=frozenset({"support", "progress"}),
        independent_verifier=lambda _: replace(evidence(ep), checks=checks)).status == expected


def test_same_backend_result_cannot_confirm_itself_and_exceptions_fail_closed(tmp_path):
    ep = bound_episode(tmp_path)
    for changed in ({"channel": ep.backend}, {"artifact": ep.artifact}):
        assert verify_episode(ep, required_checks=frozenset({"support"}),
            independent_verifier=lambda _: replace(evidence(ep), **changed)).status == "unverified"
    def broken(_):
        raise RuntimeError("observer down")
    assert verify_episode(ep, required_checks=frozenset({"support"}),
                           independent_verifier=broken).status == "unverified"
    with pytest.raises(ValueError, match="explicit"):
        verify_episode(ep, required_checks=frozenset())


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(runs={}),
    lambda x: x["runs"]["pick"].update(status="failed"),
    lambda x: x["runs"]["pick"].update(rebuilds=[]),
    lambda x: x["runs"]["pick"]["rebuilds"][0].update(episodes=[]),
    lambda x: x["runs"]["pick"]["rebuilds"][0]["episodes"][0].update(success="true"),
    lambda x: x["runs"]["pick"]["rebuilds"][0]["episodes"].append(
        copy.deepcopy(x["runs"]["pick"]["rebuilds"][0]["episodes"][0])),
])
def test_empty_failed_malformed_and_duplicate_campaigns_rejected(tmp_path, mutation):
    raw = arena_record()
    mutation(raw)
    with pytest.raises(ValueError):
        import_experiment(write_record(tmp_path, raw))


def test_sidecar_cannot_bind_edited_source(tmp_path):
    path = write_record(tmp_path)
    ep = import_experiment(path)[0]
    raw = arena_record()
    raw["runs"]["pick"]["rebuilds"][0]["episodes"][0]["variations"]["cube_mass"] = 1.0
    write_record(tmp_path, raw)
    with pytest.raises(ValueError, match="sidecar"):
        import_experiment(path, bindings={ep.record_id: binding(ep.record_id)},
                          bound_artifact_sha256=ep.artifact.sha256)


def test_imported_file_modified_later_cannot_reuse_physical_verdict(tmp_path):
    ep = bound_episode(tmp_path)
    Path(ep.artifact.path).write_text('{}\n')
    assert verify_episode(ep, required_checks=frozenset({"support"}),
                           independent_verifier=evidence).status == "unverified"


def test_independent_artifact_must_exist_and_match_digest(tmp_path):
    ep = bound_episode(tmp_path)
    witness = evidence(ep)
    Path(witness.artifact.path).write_text('{"changed":true}')
    assert verify_episode(ep, required_checks=frozenset({"support"}),
                           independent_verifier=lambda _: witness).status == "unverified"


def test_policy_keeps_runtime_skills_and_external_action_owner_separate():
    calls = []
    action = object()
    controller = SimpleNamespace(get_action=lambda env, obs: action,
        reset=lambda ids: calls.append(("reset", ids)), close=lambda: calls.append("close"),
        stop=lambda: calls.append("external_stop"), reset_stop=lambda: True)
    runtime = RobotRuntime({"arena_policy": arena_owner(controller)})
    try:
        adapter = ArenaPolicyAdapter(runtime, controller, actuation_owner="arena_policy")
        assert adapter.get_action(object(), {"policy": "observation"}) is action
        assert calls == []
        assert adapter.execute_skill("list_resources", {})["ok"] is True
        with pytest.raises(ValueError, match="unavailable"):
            adapter.execute_skill("raw.torque", {})
        adapter.reset([1])
        adapter.close()
        adapter.close()
        assert calls == [("reset", [1]), "external_stop", "close"]
        with pytest.raises(RuntimeError, match="closed"):
            adapter.get_action(None, {})
    finally:
        runtime.close()


@pytest.mark.parametrize("when", ["before", "during"])
def test_stop_generation_invalidates_external_action_without_fabricating_zero(when):
    stops = []
    def cancel():
        runtime.stop()
    def act(env, obs):
        if when == "during":
            cancel()
        return object()
    controller = SimpleNamespace(get_action=act, reset=lambda ids: None, close=lambda: None,
        stop=lambda: stops.append(True), reset_stop=lambda: True)
    runtime = RobotRuntime({"arena_policy": arena_owner(controller)})
    try:
        adapter = ArenaPolicyAdapter(runtime, controller, actuation_owner="arena_policy")
        if when == "before":
            cancel()
        with pytest.raises(RuntimeError, match="invalidated"):
            adapter.get_action(None, {})
        assert stops
        adapter.reset([0])
        with pytest.raises(RuntimeError, match="invalidated"):
            adapter.get_action(None, {})
    finally:
        runtime.close()


def test_explicit_stop_also_stops_external_owner_and_reset_needs_both_acknowledgements():
    events = []
    controller = SimpleNamespace(get_action=lambda *_: object(), reset=lambda _: None,
        close=lambda: None, stop=lambda: events.append("external_stop"), reset_stop=lambda: False)
    runtime = RobotRuntime({"arena_policy": arena_owner(controller)})
    try:
        adapter = ArenaPolicyAdapter(runtime, controller, actuation_owner="arena_policy")
        adapter.execute_skill("emergency_stop", {})
        assert events == ["external_stop"]
        assert adapter.execute_skill("reset_stop", {})["ok"] is False
        assert runtime.stopped
        with pytest.raises(RuntimeError):
            adapter.get_action(None, {})
        controller.reset_stop = lambda: True
        assert adapter.execute_skill("reset_stop", {})["ok"] is True
        assert adapter.get_action(None, {}) is not None
    finally:
        runtime.close()


def test_direct_runtime_stop_reaches_owner_while_get_action_is_blocked():
    entered, released, stopped = threading.Event(), threading.Event(), threading.Event()
    outcomes = []
    def get_action(*_):
        entered.set()
        assert released.wait(2), "priority stop did not reach the external owner"
        return object()
    def stop():
        stopped.set()
        released.set()
    controller = SimpleNamespace(get_action=get_action, stop=stop, reset_stop=lambda: True,
                                 reset=lambda _: None, close=lambda: None)
    runtime = RobotRuntime({"arena_policy": arena_owner(controller)})
    adapter = ArenaPolicyAdapter(runtime, controller, actuation_owner="arena_policy")
    def produce():
        try:
            outcomes.append(adapter.get_action(None, {}))
        except RuntimeError:
            outcomes.append("invalidated")
    thread = threading.Thread(target=produce)
    try:
        thread.start()
        assert entered.wait(2)
        runtime.stop()  # same path as MCP, without another adapter call
        assert stopped.is_set()
        thread.join(2)
        assert not thread.is_alive() and outcomes == ["invalidated"]
    finally:
        released.set()
        thread.join(2)
        adapter.close()
        runtime.close()


def test_unregistered_action_owner_is_rejected():
    runtime = RobotRuntime({})
    try:
        with pytest.raises(ValueError, match="registered"):
            ArenaPolicyAdapter(runtime, object(), actuation_owner="invented owner")
    finally:
        runtime.close()


def test_policy_factory_releases_controller_when_runtime_construction_fails(monkeypatch):
    # Only the inspected base-class construction surface is stubbed here;
    # this does not execute installed Arena or assert native compatibility.
    module = ModuleType("isaaclab_arena.policy.policy_base")
    class PolicyBase:
        def __init__(self, config):
            self.config = config
    module.PolicyBase = PolicyBase
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    controller = SimpleNamespace(get_action=lambda *_: None, reset=lambda _: None,
        reset_stop=lambda: True, stop=lambda: calls.append("stop"), close=lambda: calls.append("close"))
    def broken_runtime(config, owner):
        assert owner.controller is controller
        raise RuntimeError("runtime construction failed")
    policy_type = make_policy_type(broken_runtime, lambda _: controller, command_resources=command_resources())
    with pytest.raises(RuntimeError, match="construction failed"):
        policy_type(object())
    assert calls == ["stop", "close"]


def test_external_controller_conflicts_with_other_writer_on_same_command_endpoint():
    controller = SimpleNamespace(get_action=lambda *_: pytest.fail("must not actuate"), reset=lambda _: None,
                                  reset_stop=lambda: True, stop=lambda: None, close=lambda: None)
    owner = arena_owner(controller)
    other = SimpleNamespace(domain_id="arm", tool_descriptors=(), resources=(ResourceDescriptor(
        "arm/commands", "actuator", "test-robot", controller_id="fixture:arena/commands", writer_id="arm"),))
    with pytest.raises(ValueError, match="incompatible writers"):
        RobotRuntime({"arena_policy": owner, "arm": other})


@pytest.mark.parametrize("resources", [(), (ResourceDescriptor("unknown", "actuator", "test-robot"),)])
def test_missing_external_command_ownership_rejected(resources):
    with pytest.raises(ValueError, match="explicit command"):
        ArenaControllerDomain(object(), command_resources=resources)


def test_verifier_copies_callback_owned_check_dictionary(tmp_path):
    ep = bound_episode(tmp_path)
    checks = {"support": "unverified"}
    witness = replace(evidence(ep), checks=checks)
    checks["support"] = "confirmed"
    assert witness.checks["support"] == "unverified"
    with pytest.raises(TypeError):
        witness.checks["support"] = "confirmed"
