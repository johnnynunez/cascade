"""Ordinary composition + synthetic readiness faults, never native admission."""
import builtins
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from cascade.apps.factory_runtime import (
    RecordedFactoryDomain, _ready, build_factory_runtime, prepare_factory_model, validate_factory_profile,
)
from cascade.apps.robot_runtime import build_robot_runtime, describe_robot, robot_tool_descriptors
from cascade.config import load_robot_config
from cascade.control.fastening import FasteningController, FasteningFault, SolveJournal
from test_fastening_runtime import binding, limits, row


def profile():
    # Tests that reach a synthetic constructor select a device explicitly;
    # discovery of the shipped profile keeps this unresolved.
    return load_robot_config("factory_m20_mounted").domains.fastening.as_dict() | {"device": "cuda:0"}


def no_sdk(monkeypatch):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split(".")[0] in {"warp", "newton", "mujoco", "torch", "isaacsim"}:
            pytest.fail("passive configuration tried to import native SDK: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)


def test_unprepared_profile_discovery_uses_normal_namespaced_tools_without_sdk(monkeypatch):
    no_sdk(monkeypatch)
    cfg = load_robot_config("factory_m20_mounted")
    assert cfg.domains.fastening.device is None
    tools = robot_tool_descriptors(cfg)
    assert "fastening.turn_screw" in tools and "turn_screw" not in tools
    tool = tools["fastening.turn_screw"]
    assert tool.effect == "motion" and tool.writes == ("fastening/mounted_arm_spindle",)
    assert tool.parameters["properties"]["turns"]["const"] == 1.
    resource = describe_robot(cfg)["fastening"].resources[0]
    assert not resource.synthetic and resource.admission == "unvalidated"
    assert resource.kind == "mounted_fastening"
    assert resource.metadata["model_identity_sha256"] is None


def test_unpinned_ordinary_build_refuses_before_sdk_or_any_controller(monkeypatch, tmp_path):
    no_sdk(monkeypatch)
    with pytest.raises(FasteningFault, match="unprepared"):
        build_robot_runtime(load_robot_config("factory_m20_mounted"), tmp_path)
    assert not list(tmp_path.rglob("model.json"))
    assert not list(tmp_path.rglob("solves.jsonl"))


def test_shoulder_seating_is_opt_in_and_discovery_cannot_start_native_runtime(monkeypatch, tmp_path):
    no_sdk(monkeypatch)
    cfg = load_robot_config("factory_m20_shoulder_seating")
    tools = robot_tool_descriptors(cfg)
    assert "fastening.seat_fastener" in tools
    assert tools["fastening.seat_fastener"].writes == tools["fastening.turn_screw"].writes
    assert tools["fastening.seat_fastener"].effect == "motion"
    assert "fastening.seat_fastener" not in robot_tool_descriptors(load_robot_config("factory_m20_mounted"))
    resource = describe_robot(cfg)["fastening"].resources[0]
    assert resource.admission == "unvalidated" and resource.metadata["seating"] is True
    with pytest.raises(FasteningFault, match="unprepared"):
        build_robot_runtime(cfg, tmp_path)
    assert not list(tmp_path.rglob("solves.jsonl"))


@pytest.mark.parametrize("change", [{"seating": None}, {"seating": "automatic"},
    {"recipe": "factory_m20_fixed_axis_v1"}])
def test_seating_requires_its_explicit_margin_recipe(change):
    cfg = load_robot_config("factory_m20_shoulder_seating")
    with pytest.raises(ValueError, match="seating recipe"):
        validate_factory_profile(cfg.domains.fastening.as_dict() | change)


@pytest.mark.parametrize("entrypoint", ["prepare", "build"])
def test_unresolved_device_refuses_before_sdk_and_output_creation(monkeypatch, tmp_path, entrypoint):
    no_sdk(monkeypatch)
    value = profile() | {"device": None, "model_identity_sha256": "a"*64}
    validate_factory_profile(value)  # Passive discovery remains possible.
    output = tmp_path/"must-not-exist"
    with pytest.raises(FasteningFault, match="device is unprepared"):
        if entrypoint == "prepare":
            prepare_factory_model(value, output)
        else:
            build_factory_runtime(value, output, domain_id="fastening")
    assert not output.exists()


def test_existing_mcp_catalog_exposes_factory_without_runtime_or_sdk(monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer
    no_sdk(monkeypatch)
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", "factory_m20_mounted")
    server = McpSkillServer()
    monkeypatch.setattr(server, "_ensure_runtime", lambda: pytest.fail("catalog opened runtime"))
    names = {item["name"] for item in server.list_tools()}
    assert {"fastening.turn_screw", "reset_stop", "emergency_stop", "list_resources"} <= names
    info = json.loads(server.call_tool("list_resources", {})["content"][-1]["text"])
    assert info["ok"] and info["resources"][0]["admission"] == "unvalidated"
    assert server._runtime is None


@pytest.mark.parametrize("kind", ["planar", "spherical", "floating"])
def test_dynamic_embodiment_refuses_mounted_fixture_before_construction(kind):
    from test_generalized_joints import coordinate_convention
    cfg = load_robot_config("fixed_so101_mock")
    raw = cfg._data["embodiment"]
    # Keep IDs consistent so this specifically reaches the dynamic-frame veto.
    raw["robot_id"] = cfg._data["robot_id"] = "so101_factory_m20"
    raw["version"] = 2
    raw["links"].append("uncontrolled_link")
    raw["joints"].append({"joint_id": "internal", "parent": raw["root_link"], "child": "uncontrolled_link",
        "type": kind, "actuation": "passive", "coordinates": coordinate_convention(kind)})
    cfg._data["domains"] = {"fastening": profile()}
    # Existing mock-arm transmissions name absent resources; remove them to
    # isolate the physical mounted-domain/dynamic-frame contract.
    raw["transmissions"] = []
    raw["effectors"] = []
    raw["sensors"] = []
    for joint in raw["joints"]:
        joint["actuation"] = "passive"
    with pytest.raises(ValueError, match="dynamic frames"):
        describe_robot(cfg)


@pytest.mark.parametrize("change", [{"recipe": "other"}, {"device": "cpu"}, {"device": "auto"},
    {"model_identity_sha256": True}, {"model_identity_sha256": "wildcard"},
    {"robot_id": "other"}, {"assets": "relative"}, {"arm": "mock"}, {"admission": "confirmed"}])
def test_profile_refuses_recipe_aliases_guessed_paths_and_admission_flags(change):
    value = profile() | change
    with pytest.raises(ValueError): validate_factory_profile(value)


def test_pinned_but_different_constructed_model_cannot_start_owner(monkeypatch, tmp_path):
    value = profile() | {"model_identity_sha256": "b"*64}
    monkeypatch.setattr("cascade.apps.factory_runtime.prepare_factory_model",
        lambda *_: NS(binding=binding(), document={"fixture": "synthetic model document"}))
    monkeypatch.setattr("cascade.apps.factory_runtime._new_owner",
        lambda *_: pytest.fail("mismatched model started a controller"))
    with pytest.raises(FasteningFault, match="differs"):
        build_factory_runtime(value, tmp_path, domain_id="fastening")
    assert json.loads((tmp_path/"startup-failure.json").read_text())["ready"] is False


def test_configured_native_constructor_cannot_substitute_a_mock_owner(monkeypatch, tmp_path):
    value = profile() | {"model_identity_sha256": "a"*64}
    monkeypatch.setattr("cascade.apps.factory_runtime.prepare_factory_model",
        lambda *_: NS(binding=binding(), document={"fixture": "synthetic model document"}))
    monkeypatch.setattr("cascade.apps.factory_runtime._new_owner",
        lambda *_: NS(backend=NS(synthetic=True), start=lambda: pytest.fail("mock started")))
    with pytest.raises(FasteningFault, match="synthetic owner"):
        build_factory_runtime(value, tmp_path, domain_id="fastening")


def test_failed_readiness_closes_owner_and_preserves_failure_receipt(monkeypatch, tmp_path):
    value = profile() | {"model_identity_sha256": "a"*64}
    monkeypatch.setattr("cascade.apps.factory_runtime.prepare_factory_model",
        lambda *_: NS(binding=binding(), document={"fixture": "synthetic model document"}))
    events = []
    journal = SolveJournal(binding())
    actuator = FasteningController(binding(), limits(), journal, synthetic=False,
        close_owner=lambda: events.append("close") or {"ok": False, "zero_upload": False}, clock=lambda: 10.)
    owner = NS(backend=NS(synthetic=False), controller=actuator, journal=journal,
        start=lambda: events.append("start"), records=lambda: [{"synthetic_fault": "missing force"}])
    monkeypatch.setattr("cascade.apps.factory_runtime._new_owner", lambda *_: owner)
    def fail(_):
        raise FasteningFault("synthetic injected missing native force")
    monkeypatch.setattr("cascade.apps.factory_runtime._ready", fail)
    with pytest.raises(FasteningFault, match="missing native force"):
        build_factory_runtime(value, tmp_path, domain_id="fastening")
    assert events == ["start", "close"]
    failure = json.loads((tmp_path/"startup-failure.json").read_text())
    assert failure["ready"] is False and failure["closure"]["ok"] is False
    assert json.loads((tmp_path/"solves.jsonl").read_text())["synthetic_fault"] == "missing force"
    with pytest.raises(FileExistsError):
        build_factory_runtime(value, tmp_path, domain_id="fastening")


@pytest.mark.parametrize("unwritable", [False, True])
def test_startup_fault_survives_raising_close_and_unwritable_failure_receipt(monkeypatch, tmp_path, unwritable):
    import cascade.apps.factory_runtime as module
    value = profile() | {"model_identity_sha256": "a"*64}
    monkeypatch.setattr(module, "prepare_factory_model",
        lambda *_: NS(binding=binding(), document={"fixture": "synthetic model document"}))
    monkeypatch.setattr(module, "_new_owner", lambda *_: NS(backend=NS(synthetic=False), start=lambda: None))
    closed = []
    def close():
        closed.append(True)
        raise OSError("synthetic closure persistence unavailable")
    monkeypatch.setattr(module, "RecordedFactoryDomain", lambda *_args, **_kwargs: NS(close=close))
    def fail(_):
        raise FasteningFault("original synthetic readiness fault")
    monkeypatch.setattr(module, "_ready", fail)
    if unwritable:
        (tmp_path/"startup-failure.json").mkdir()
    with pytest.raises(FasteningFault, match="original synthetic readiness fault") as caught:
        build_factory_runtime(value, tmp_path, domain_id="fastening")
    assert closed == [True]
    if unwritable:
        note = caught.value.__notes__[0]
        assert "closure persistence unavailable" in note and "persistence_error" in note
    else:
        failure = json.loads((tmp_path/"startup-failure.json").read_text())
        assert "original synthetic readiness fault" in failure["error"]
        assert not failure["closure"]["ok"] and "closure persistence unavailable" in failure["closure"]["error"]


class Readiness:
    def __init__(self, transform=lambda x: x, *, late=False):
        self.now = 10.
        self.backend = NS(binding=binding(), limits=limits())
        self.step, self.transform, self.late = 0, transform, late
        self.journal = NS(read=self.read)

    def read(self, cursor, *, timeout_s):
        assert cursor == self.step and 0 <= timeout_s <= .05
        self.step += 1
        self.now += .01
        value = self.transform(row(self.step, captured=self.now))
        if self.late and self.step == 51:
            self.now = 20.01  # Pending return crossed unchanged startup wall deadline.
            value = replace(value, captured_monotonic_s=self.now-.001)
        return (value,)


def test_initial_readiness_retains_latch_and_full_quiet_window_without_reset():
    owner = Readiness()
    result = _ready(owner, clock=lambda: owner.now)
    assert owner.step == 51 and result["generation"] == 0 and result["latched"]
    assert result["quiet_interval_sim_s"] == pytest.approx(.5)
    assert result["captured_monotonic_s"] == owner.now
    assert result["physical_task_admission"] is False


@pytest.mark.parametrize("change,reason", [
    ({"generation": 1}, "admitted early"), ({"captured_monotonic_s": 9.}, "stale"),
    ({"step": 2}, "lost solves"), ({"spindle_effort_nm": .051}, "effort"),
    ({"binding_sha256": "f"*64}, "identity"),
])
def test_startup_missing_stale_or_unsafe_stream_has_no_readiness(change, reason):
    owner = Readiness(lambda value: replace(value, **change))
    with pytest.raises(FasteningFault, match=reason):
        _ready(owner, clock=lambda: owner.now)


def test_late_quiet_completion_cannot_grant_startup_after_deadline():
    owner = Readiness(late=True)
    with pytest.raises(FasteningFault, match="after startup deadline"):
        _ready(owner, clock=lambda: owner.now)


def test_missing_preengagement_never_earns_quiet_readiness():
    owner = Readiness(lambda value: replace(value, contacts=(), collision_count=0,
        solver_count=0, thread_contacts=0, tool_contacts=0))
    with pytest.raises(FasteningFault, match="startup"):
        _ready(owner, clock=lambda: owner.now, timeout_s=.1)


def test_wiring_uses_existing_robot_runtime_and_preserves_unverified_result(monkeypatch, tmp_path):
    calls = []
    domain = NS(execute=lambda name, args: calls.append((name, args)) or {
        "ok": False, "execution_ok": True, "verified": False, "synthetic": True,
        "postcondition": {"status": "unverified", "reason": "synthetic wiring has no physics"}},
        close=lambda: {"ok": True}, stop=lambda: {"ok": True}, reset_stop=lambda: {"ok": True})
    def factory(profile, directory, *, domain_id):
        calls.append((domain_id, str(directory)))
        return domain
    monkeypatch.setattr("cascade.apps.factory_runtime.build_factory_runtime", factory)
    runtime, _ = build_robot_runtime(load_robot_config("factory_m20_mounted"), tmp_path)
    try:
        result = runtime.execute("fastening.turn_screw", {"turns": 1., "direction": "tighten"})
        assert not result["ok"] and result["postcondition"]["status"] == "unverified"
        assert calls[1] == ("turn_screw", {"turns": 1., "direction": "tighten"})
        done = runtime.execute("task_done", {"success": True, "summary": "claimed done"})
        assert not done["success"]
    finally:
        assert runtime.close()["ok"]


def test_evidence_io_failure_latches_and_prevents_reset_or_later_motion(tmp_path):
    journal = SolveJournal(binding())
    actuator = FasteningController(binding(), limits(), journal, synthetic=True,
                                  close_owner=lambda: {"ok": True}, clock=lambda: 10.)
    owner = NS(controller=actuator, journal=journal, records=lambda: [{"step": 1}])
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id="fastening")
    (tmp_path/"solves.jsonl").mkdir()  # Deterministic persistence failure, not real disk outage.
    with pytest.raises(IsADirectoryError): domain.flush_records()
    with pytest.raises(FasteningFault, match="new owned epoch"): domain.reset_stop()
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert not result["ok"] and not result["execution_ok"]
    assert "sticky" in result["postcondition"]["reason"]
    assert not domain.close()["ok"]


def test_action_receipt_failure_cannot_keep_success_or_reset_authority(monkeypatch, tmp_path):
    from cascade.skills.fastening_runtime import FasteningDomain
    journal = SolveJournal(binding())
    actuator = FasteningController(binding(), limits(), journal, synthetic=True,
                                  close_owner=lambda: {"ok": True}, clock=lambda: 10.)
    domain = RecordedFactoryDomain(NS(controller=actuator, journal=journal, records=lambda: []),
                                  tmp_path, domain_id="fastening")
    monkeypatch.setattr(FasteningDomain, "execute", lambda *_: {
        "ok": True, "verified": True, "execution_ok": True, "synthetic": True,
        "postcondition": {"status": "confirmed", "reason": "synthetic injected result"}})
    (tmp_path/"last-action.json").mkdir()
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert result["execution_ok"] and not result["ok"] and not result["verified"]
    assert result["postcondition"]["status"] == "unverified"
    with pytest.raises(FasteningFault, match="new owned epoch"):
        domain.reset_stop()
    later = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert not later["execution_ok"]


def test_gc_policy_values_are_reviewed_and_the_opt_in_profile_stays_passive(monkeypatch, tmp_path):
    no_sdk(monkeypatch)
    validate_factory_profile(profile() | {"gc_policy": "freeze-startup-heap"})
    validate_factory_profile(profile() | {"gc_policy": None})
    for bad in ("disable", True, "freeze-startup-heap-v1", 1):
        with pytest.raises(ValueError, match="GC policy"):
            validate_factory_profile(profile() | {"gc_policy": bad})
    cfg = load_robot_config("factory_m20_shoulder_seating_heap_freeze")
    assert cfg.domains.fastening.gc_policy == "freeze-startup-heap"
    base = load_robot_config("factory_m20_shoulder_seating")
    assert cfg.domains.fastening.as_dict() | {"gc_policy": None} == base.domains.fastening.as_dict() | {"gc_policy": None}
    assert robot_tool_descriptors(cfg).keys() == robot_tool_descriptors(base).keys()
    with pytest.raises(FasteningFault, match="unprepared"):
        build_robot_runtime(cfg, tmp_path)
    assert not list(tmp_path.rglob("gc-policy.json"))


def _synthetic_owner(events, close_receipt, *, start_error=None):
    journal = SolveJournal(binding())
    actuator = FasteningController(binding(), limits(), journal, synthetic=False,
        close_owner=lambda: events.append("close") or close_receipt, clock=lambda: 10.)
    def start():
        events.append("start")
        if start_error is not None:
            raise start_error
    return NS(backend=NS(synthetic=False), controller=actuator, journal=journal, start=start, records=lambda: [])


def _recorded_freeze(monkeypatch, events):
    """Injected collector (never the test interpreter's); record the policy's own calls."""
    from test_heap_freeze import fake_collector
    from cascade.sim import heap_freeze
    collector = fake_collector(foreign_frozen=375)
    class Recorded(heap_freeze.StartupHeapFreeze):
        def __init__(self, **kwargs):
            super().__init__(collector=collector, **kwargs)
        def apply(self):
            events.append("apply")
            return super().apply()
        def release(self, **kwargs):
            events.append("release")
            return super().release(**kwargs)
    monkeypatch.setattr(heap_freeze, "StartupHeapFreeze", Recorded)
    return collector


def _synthetic_model(monkeypatch, module):
    monkeypatch.setattr(module, "prepare_factory_model",
        lambda *_: NS(binding=binding(), document={"fixture": "synthetic model document"}))


@pytest.mark.parametrize("gc_policy", [None, "freeze-startup-heap"])
def test_gc_policy_freezes_after_model_pin_before_owner_start_and_releases_after_owner_close(monkeypatch, tmp_path, gc_policy):
    import cascade.apps.factory_runtime as module
    value = profile() | {"model_identity_sha256": "a"*64} | ({} if gc_policy is None else {"gc_policy": gc_policy})
    _synthetic_model(monkeypatch, module)
    events = []
    collector = _recorded_freeze(monkeypatch, events)
    owner = _synthetic_owner(events, {"ok": True, "owner_thread_closed": True, "zero_spindle": {"uploaded": True}})
    monkeypatch.setattr(module, "_new_owner", lambda *_: owner)
    monkeypatch.setattr(module, "_ready", lambda _: {"ready": True, "fixture": True})
    domain = build_factory_runtime(value, tmp_path, domain_id="fastening")
    try:
        readiness = json.loads((tmp_path/"readiness.json").read_text())
        if gc_policy is None:
            assert events == ["start"] and collector.calls == [] and "gc_policy" not in readiness
            assert not (tmp_path/"gc-policy.json").exists()
        else:
            assert events == ["apply", "start"]  # frozen after the model pin check, before any solve
            receipt = json.loads((tmp_path/"gc-policy.json").read_text())
            assert receipt["model_identity_sha256"] == "a"*64 and receipt["profile_gc_policy"] == gc_policy
            applied = receipt["apply"]
            assert applied["policy"] == "freeze-startup-heap-v1" and applied["freeze"]["frozen"] == 1200
            assert applied["freeze"]["foreign_frozen_before"] == 375
            assert applied["settings"] == {"enabled": True, "thresholds": [700, 10, 10], "callbacks": 1}
            assert readiness["gc_policy"] == {"selected": gc_policy, "policy": "freeze-startup-heap-v1",
                "implementation_sha256": applied["implementation_sha256"], "receipt": "gc-policy.json"}
            assert json.loads((tmp_path/"factory-profile.json").read_text())["gc_policy"] == gc_policy
    finally:
        closure = domain.close()
    assert closure["ok"] is True
    if gc_policy is None:
        assert events == ["start", "close"] and "gc_policy" not in closure
    else:
        assert events == ["apply", "start", "close", "release"]  # released only after the owner closed
        assert collector.calls == [("collect", 2), ("freeze",), ("collect", 2), ("unfreeze",), ("collect", 2)]
        released = closure["gc_policy"]
        assert released["policy_frozen"] == 1200 and released["unfrozen"] == 1575 and released["apply_receipt_complete"]
        assert json.loads((tmp_path/"closure.json").read_text())["gc_policy"]["unfrozen"] == 1575
        # Repeated closure reuses the recorded release and never repeats the collector call.
        again = domain.close()  # the owner is closed again; the collector is not touched again
        assert again["gc_policy"] == released and events == ["apply", "start", "close", "release", "close"]
        assert collector.calls.count(("unfreeze",)) == 1
        assert json.loads((tmp_path/"closure.json").read_text())["gc_policy"]["unfrozen"] == 1575


def test_gc_policy_is_not_released_while_the_owner_thread_may_still_run(monkeypatch, tmp_path):
    import cascade.apps.factory_runtime as module
    value = profile() | {"model_identity_sha256": "a"*64, "gc_policy": "freeze-startup-heap"}
    _synthetic_model(monkeypatch, module)
    events = []
    collector = _recorded_freeze(monkeypatch, events)
    owner = _synthetic_owner(events, {"ok": False, "owner_thread_closed": False, "error": None})
    monkeypatch.setattr(module, "_new_owner", lambda *_: owner)
    monkeypatch.setattr(module, "_ready", lambda _: {"ready": True, "fixture": True})
    domain = build_factory_runtime(value, tmp_path, domain_id="fastening")
    closure = domain.close()
    assert closure["ok"] is False and closure["gc_policy"] == {"released": False, "reason": "owner thread not confirmed closed",
                                                                 "release_failures": []}
    assert events == ["apply", "start", "close"] and domain.heap_freeze.frozen
    assert collector.calls == [("collect", 2), ("freeze",)] and collector.frozen == 1575
    assert json.loads((tmp_path/"closure.json").read_text())["gc_policy"]["released"] is False
    assert domain.close()["gc_policy"]["released"] is False and collector.calls.count(("unfreeze",)) == 0


@pytest.mark.parametrize("failure", ["readiness", "owner_start", "receipt_persistence"])
def test_startup_failure_after_freeze_still_releases_through_domain_closure(monkeypatch, tmp_path, failure):
    import cascade.apps.factory_runtime as module
    value = profile() | {"model_identity_sha256": "a"*64, "gc_policy": "freeze-startup-heap"}
    _synthetic_model(monkeypatch, module)
    events = []
    collector = _recorded_freeze(monkeypatch, events)
    owner = _synthetic_owner(events, {"ok": False, "owner_thread_closed": True, "zero_spindle": {"uploaded": False}},
        start_error=FasteningFault("synthetic owner start fault") if failure == "owner_start" else None)
    monkeypatch.setattr(module, "_new_owner", lambda *_: owner)
    def fail(_):
        assert events == ["apply", "start"]
        raise FasteningFault("synthetic readiness fault under the frozen heap")
    monkeypatch.setattr(module, "_ready", fail)
    if failure == "receipt_persistence":
        (tmp_path/"gc-policy.json").mkdir()  # the apply receipt cannot be written
    expected = {"readiness": "under the frozen heap", "owner_start": "owner start fault",
                "receipt_persistence": "gc-policy.json"}[failure]
    with pytest.raises((FasteningFault, IsADirectoryError), match=expected):
        build_factory_runtime(value, tmp_path, domain_id="fastening")
    assert events == (["apply", "close", "release"] if failure == "receipt_persistence" else ["apply", "start", "close", "release"])
    assert collector.calls == [("collect", 2), ("freeze",), ("collect", 2), ("unfreeze",), ("collect", 2)]
    failure_receipt = json.loads((tmp_path/"startup-failure.json").read_text())
    assert failure_receipt["ready"] is False and failure_receipt["closure"]["gc_policy"]["unfrozen"] == 1575
    if failure != "receipt_persistence":
        assert json.loads((tmp_path/"gc-policy.json").read_text())["apply"]["policy"] == "freeze-startup-heap-v1"


def test_policy_enabled_pin_mismatch_never_freezes(monkeypatch, tmp_path):
    import cascade.apps.factory_runtime as module
    value = profile() | {"model_identity_sha256": "b"*64, "gc_policy": "freeze-startup-heap"}
    _synthetic_model(monkeypatch, module)
    events = []
    collector = _recorded_freeze(monkeypatch, events)
    monkeypatch.setattr(module, "_new_owner", lambda *_: pytest.fail("mismatched model started a controller"))
    with pytest.raises(FasteningFault, match="differs"):
        build_factory_runtime(value, tmp_path, domain_id="fastening")
    assert events == [] and collector.calls == [] and not (tmp_path/"gc-policy.json").exists()


def test_release_failure_is_recorded_and_a_later_closure_retries_with_history(monkeypatch, tmp_path):
    import cascade.apps.factory_runtime as module
    value = profile() | {"model_identity_sha256": "a"*64, "gc_policy": "freeze-startup-heap"}
    _synthetic_model(monkeypatch, module)
    events = []
    collector = _recorded_freeze(monkeypatch, events)
    owner = _synthetic_owner(events, {"ok": True, "owner_thread_closed": True, "zero_spindle": {"uploaded": True}})
    monkeypatch.setattr(module, "_new_owner", lambda *_: owner)
    monkeypatch.setattr(module, "_ready", lambda _: {"ready": True, "fixture": True})
    domain = build_factory_runtime(value, tmp_path, domain_id="fastening")
    original = collector.unfreeze
    def failing_unfreeze():
        raise OSError("synthetic unfreeze failure")
    collector.unfreeze = failing_unfreeze
    first = domain.close()
    assert first["ok"] is False and first["gc_policy"]["released"] is False and first["gc_policy"]["receipt"] is None
    assert first["gc_policy"]["release_failures"] == ["OSError: synthetic unfreeze failure"]
    assert domain.heap_freeze.state == "releasing" and collector.frozen == 1575  # still frozen, retry allowed
    assert json.loads((tmp_path/"closure.json").read_text())["gc_policy"]["release_failures"]
    collector.unfreeze = original
    second = domain.close()
    assert second["gc_policy"]["unfrozen"] == 1575 and second["gc_policy"]["release_failures"] == ["OSError: synthetic unfreeze failure"]
    assert collector.frozen == 0 and domain.heap_freeze.state == "released"
    assert json.loads((tmp_path/"closure.json").read_text())["gc_policy"]["release_failures"]
    assert domain.close()["gc_policy"] == second["gc_policy"]  # retained history on every later closure


def test_owner_closure_exception_keeps_the_heap_frozen_and_persists_the_fault(monkeypatch, tmp_path):
    import cascade.apps.factory_runtime as module
    value = profile() | {"model_identity_sha256": "a"*64, "gc_policy": "freeze-startup-heap"}
    _synthetic_model(monkeypatch, module)
    events = []
    collector = _recorded_freeze(monkeypatch, events)
    journal = SolveJournal(binding())
    def raising_close():
        raise RuntimeError("cannot join thread before it is started")
    actuator = FasteningController(binding(), limits(), journal, synthetic=False, close_owner=raising_close, clock=lambda: 10.)
    owner = NS(backend=NS(synthetic=False), controller=actuator, journal=journal, start=lambda: None, records=lambda: [])
    monkeypatch.setattr(module, "_new_owner", lambda *_: owner)
    monkeypatch.setattr(module, "_ready", lambda _: {"ready": True, "fixture": True})
    domain = build_factory_runtime(value, tmp_path, domain_id="fastening")
    closure = domain.close()
    assert closure["ok"] is False and "cannot join thread" in closure["closure_error"]
    assert closure["gc_policy"]["released"] is False and domain.heap_freeze.frozen and collector.frozen == 1575
    assert json.loads((tmp_path/"closure.json").read_text())["closure_error"].startswith("RuntimeError")
    domain.heap_freeze.release()  # restore the fake collector; the real path is covered by the owner test
