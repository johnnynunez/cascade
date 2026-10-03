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
