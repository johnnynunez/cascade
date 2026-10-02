"""Composition contracts exercise existing domain semantics, not fake physics."""
import copy
import threading

import pytest

from cascade.config import load_robot_config, load_demo_config
from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime


class Domain:
    def __init__(self, name="locomotion", *, synthetic=False, controller=None, result=None):
        self.domain_id = name
        self.resources = (ResourceDescriptor(name + "/body", "base", "fixture",
            controller_id=controller or name, writer_id=name, synthetic=synthetic),)
        self.tool_descriptors = (ToolDescriptor(name + ".move", "bounded fixture motion",
            {"type": "object", "properties": {"distance": {"type": "number"}},
             "required": ["distance"], "additionalProperties": False},
            name, "move", effect="motion", writes=(name + "/body",)),)
        self.result = result or {"ok": True, "execution_ok": True,
                                 "postcondition": {"status": "confirmed", "reason": "independent fixture"}}
        self.calls = []
        self.entered = threading.Event()
        self.release = None
        self.closed = 0
        self.stops = 0

    def execute(self, name, args):
        self.calls.append((name, args))
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(3), "test coordinator did not release domain"
        return self.result

    def stop(self):
        self.stops += 1
        if self.release is not None:
            self.release.set()
        return {"ok": True}

    def reset_stop(self):
        return {"ok": True}

    def begin_task(self):
        pass

    def close(self):
        self.closed += 1
        return {"ok": True}


def test_old_episode_cannot_dispatch_after_operator_stop_and_reset():
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    original_token = runtime.cancellation_token
    assert runtime.stop()["ok"]
    assert runtime.reset_stop()["ok"]
    assert not runtime.stopped
    rejected = runtime.execute("locomotion.move", {"distance": .1}, expected_generation=original_token)
    assert not rejected["ok"] and "stale execution generation" in rejected["error"]
    assert domain.calls == []
    assert runtime.execute("locomotion.move", {"distance": .1},
                           expected_generation=runtime.cancellation_token)["ok"]
    assert len(domain.calls) == 1
    assert runtime.close()["ok"]


@pytest.mark.parametrize("expected", [True, False, 0.0, "0", -1, []])
def test_reset_generation_must_be_an_exact_nonnegative_integer(expected):
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    resets = []
    domain.reset_stop = lambda: resets.append(True) or {"ok": True}
    try:
        assert runtime.stop()["ok"]
        assert not runtime.reset_stop(expected_generation=expected)["ok"]
        assert not resets and runtime.stopped
    finally:
        assert runtime.close()["ok"]


def test_reset_generation_is_checked_inside_admission_gate_after_newer_stop():
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    resets, results = [], []
    domain.reset_stop = lambda: resets.append(True) or {"ok": True}
    waiting, release = threading.Event(), threading.Event()
    original_gate = runtime._gate
    worker = None

    class HeldAdmission:
        def __enter__(self):
            if threading.current_thread() is worker:
                waiting.set()
                assert release.wait(3), "test did not release reset admission"
            original_gate.acquire()

        def __exit__(self, *_args):
            original_gate.release()

    try:
        assert runtime.stop()["ok"]
        observed = runtime.cancellation_token
        # stop's condition still uses the same underlying lock. Only delay the
        # reset thread before acquiring it, then send an actual priority stop.
        runtime._gate = HeldAdmission()
        worker = threading.Thread(target=lambda: results.append(
            runtime.reset_stop(expected_generation=observed)))
        worker.start()
        assert waiting.wait(2)
        newer = runtime.stop()
        assert newer["ok"] and newer["generation"] > observed
        release.set(); worker.join(3)
        assert not worker.is_alive()
        assert results == [{"ok": False, "error": "stale reset generation",
                            "generation": newer["generation"], "latched": True}]
        assert not resets and runtime.stopped
        assert runtime.reset_stop(expected_generation=newer["generation"])["ok"]
        assert resets == [True] and not runtime.stopped
    finally:
        release.set()
        if worker is not None:
            worker.join(3)
        runtime._gate = original_gate
        assert runtime.close()["ok"]


def test_new_stop_during_admitted_generation_reset_still_relatches_domains():
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    entered, release = threading.Event(), threading.Event()
    results = []

    def reset():
        entered.set()
        assert release.wait(3)
        return {"ok": True}

    domain.reset_stop = reset
    worker = None
    try:
        assert runtime.stop()["ok"]
        observed = runtime.cancellation_token
        worker = threading.Thread(target=lambda: results.append(
            runtime.reset_stop(expected_generation=observed)))
        worker.start(); assert entered.wait(2)
        assert runtime.stop()["ok"]
        release.set(); worker.join(3)
        assert not worker.is_alive() and results[0]["ok"] is False
        assert runtime.stopped and domain.stops >= 3
    finally:
        release.set()
        if worker is not None:
            worker.join(3)
        assert runtime.close()["ok"]


@pytest.mark.parametrize("args", [{}, {"distance": "1"}, {"distance": True}, {"distance": 1, "extra": 2}, []])
def test_arguments_refused_before_domain_io(args):
    domain = Domain()
    rt = RobotRuntime({domain.domain_id: domain})
    assert not rt.execute("locomotion.move", args)["ok"]
    assert domain.calls == []


def test_metadata_and_result_preserve_independent_verdict():
    domain = Domain(result={"ok": False, "execution_ok": True,
                           "postcondition": {"status": "refuted", "reason": "no displacement"},
                           "measured": {"dx": 0.0}})
    rt = RobotRuntime({domain.domain_id: domain})
    assert rt.execute("list_resources", {})["metadata_source"] == "configured_profile"
    assert not domain.calls
    before = copy.deepcopy(domain.result)
    assert rt.execute("locomotion.move", {"distance": 1}) == before
    assert domain.result == before
    assert not rt.execute("task_done", {"success": True, "summary": "claim"})["success"]


def test_stop_bypasses_action_and_invalidates_late_success():
    first, second = Domain(), Domain("manipulation")
    first.release = threading.Event()
    rt = RobotRuntime({d.domain_id: d for d in (first, second)})
    result = []
    worker = threading.Thread(target=lambda: result.append(rt.execute("locomotion.move", {"distance": 1})))
    worker.start()
    assert first.entered.wait(2)
    assert not rt.execute("manipulation.move", {"distance": 1})["ok"]
    assert not rt.reset_stop()["ok"]
    assert rt.stop()["latched"]
    worker.join(2)
    assert not worker.is_alive()
    assert first.stops == second.stops == 1
    assert not result[0]["ok"] and result[0]["domain_result"] == first.result
    assert not second.calls
    assert not rt.execute("locomotion.move", {"distance": 1})["ok"]
    assert rt.reset_stop()["ok"]
    assert "locomotion.move" in rt.unverified_actions()
    rt.close()


def test_failure_of_one_stop_still_reaches_other_domains():
    first, second = Domain(), Domain("manipulation")
    def failed():
        raise RuntimeError("transport failed")
    first.stop = failed
    rt = RobotRuntime({d.domain_id: d for d in (first, second)})
    assert not rt.stop()["ok"]
    assert second.stops == 1
    rt.close()


def test_same_controller_cannot_get_two_writers():
    with pytest.raises(ValueError, match="controller"):
        RobotRuntime({n: Domain(n, controller="unitree:rt/arm_sdk") for n in ("left", "right")})


def test_synthetic_and_older_unverified_actions_cannot_be_forgotten():
    domain = Domain(synthetic=True)
    rt = RobotRuntime({domain.domain_id: domain})
    for _ in range(260):
        assert rt.execute("locomotion.move", {"distance": .1}) == domain.result
    assert rt.unverified_actions() == ["locomotion.move"]
    assert not rt.execute("task_done", {"success": True, "summary": "synthetic"})["success"]


def test_close_cancels_then_closes_once_after_operation_returns():
    domain = Domain()
    domain.release = threading.Event()
    rt = RobotRuntime({domain.domain_id: domain})
    worker = threading.Thread(target=lambda: rt.execute("locomotion.move", {"distance": 1}))
    worker.start()
    assert domain.entered.wait(2)
    assert rt.close()["ok"]
    worker.join(2)
    assert domain.closed == 1
    assert rt.close()["already_closed"]
    assert domain.closed == 1
    assert not rt.execute("locomotion.move", {"distance": 1})["ok"]


def test_robot_selection_is_explicit_and_legacy_defaults_unchanged(monkeypatch):
    monkeypatch.delenv("CASCADE_ROBOT", raising=False)
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    assert "robot_mode" not in load_demo_config()
    assert load_demo_config(base="microduck_mock").robot_mode == "mobile"
    monkeypatch.setenv("CASCADE_ROBOT", "mixed_mock")
    cfg = load_demo_config()
    assert cfg.robot_mode == "composed"
    assert set(cfg.domains.as_dict()) == {"manipulation", "locomotion", "sensing"}
    with pytest.raises(ValueError, match="cannot also select"):
        load_demo_config(base="microduck_mock")


def test_physical_mixed_domains_and_mounted_static_arm_are_refused(tmp_path):
    from cascade.apps.robot_runtime import describe_robot
    cfg = load_robot_config("mixed_mock")
    cfg._data["domains"].pop("sensing")
    cfg._data["domains"]["manipulation"]["resolved"]["arms"][0]["type"] = "isaac"
    with pytest.raises(ValueError, match="mixed physical"):
        describe_robot(cfg)
    profile = tmp_path / "robots"
    profile.mkdir()
    (profile / "mounted.yaml").write_text("version: 1\nrobot_id: bad\ndomains:\n  arm:\n    kind: manipulation\n    mounted_on: base\n")
    with pytest.raises(ValueError, match="mobile-mounted"):
        load_robot_config("mounted", config_dir=tmp_path)


def test_two_unitree_arm_profiles_share_whole_lowcmd_controller():
    from cascade.apps.robot_runtime import describe_robot
    cfg = load_robot_config("mixed_mock")
    cfg._data["domains"].pop("sensing")
    cfg._data["domains"].pop("locomotion")
    arms = cfg._data["domains"]["manipulation"]["resolved"]["arms"]
    arms[0]["type"] = "unitree_arm"
    arms.append({**copy.deepcopy(arms[0]), "name": "second", "unitree_joint_index": [7, 8, 9, 10, 11]})
    with pytest.raises(ValueError, match="controller"):
        describe_robot(cfg)


def test_result_exception_after_dispatch_blocks_success_claim():
    domain = Domain()
    def failed(name, args):
        raise RuntimeError("lost execution reply")
    domain.execute = failed
    rt = RobotRuntime({domain.domain_id: domain})
    assert not rt.execute("locomotion.move", {"distance": 1})["ok"]
    assert rt.unverified_actions() == ["locomotion.move"]
    assert not rt.execute("task_done", {"success": True, "summary": "claim"})["success"]


@pytest.mark.parametrize("contradiction", [{"execution_ok": False}, {"delivery_uncertain": True}])
def test_contradictory_confirmed_result_is_preserved_but_cannot_complete_task(contradiction):
    domain = Domain(result={"ok": True, "postcondition": {"status": "confirmed"}, **contradiction})
    rt = RobotRuntime({domain.domain_id: domain})
    assert rt.execute("locomotion.move", {"distance": 1}) == domain.result
    assert rt.unverified_actions() == ["locomotion.move"]
    assert not rt.execute("task_done", {"success": True, "summary": "claim"})["success"]


def test_partial_composition_build_closes_previously_built_domain(tmp_path, monkeypatch):
    from cascade.apps import demo, mobile_runtime
    from cascade.apps.robot_runtime import build_robot_runtime
    cfg = load_robot_config("mixed_mock")
    closed = []
    arm_runtime, owner = object(), object()
    monkeypatch.setattr(demo, "build_runtime", lambda *a, **k: (arm_runtime, owner))
    monkeypatch.setattr(demo, "shutdown_runtime", lambda rt, raw: closed.append((rt, raw)))
    def failed(*args, **kwargs):
        raise RuntimeError("second domain initialization failed")
    monkeypatch.setattr(mobile_runtime, "build_mobile_runtime", failed)
    with pytest.raises(RuntimeError, match="second domain"):
        build_robot_runtime(cfg, tmp_path)
    assert closed == [(arm_runtime, owner)]


def test_blocked_stop_does_not_block_other_domains_or_multiply_workers():
    slow, other = Domain("slow"), Domain("other")
    unblock = threading.Event()
    entered = threading.Event()
    def blocked_stop():
        entered.set()
        assert unblock.wait(4)
        return {"ok": True}
    slow.stop = blocked_stop
    rt = RobotRuntime({d.domain_id: d for d in (slow, other)})
    try:
        first = rt.stop()
        assert entered.is_set() and first["domains"]["slow"]["pending"]
        assert other.stops == 1
        worker = rt._stop_slots["slow"]["thread"]
        assert not rt.reset_stop()["ok"]
        second = rt.stop()
        assert second["domains"]["slow"]["pending"]
        assert rt._stop_slots["slow"]["thread"] is worker
        assert other.stops == 2
    finally:
        unblock.set()
        assert rt.close()["ok"]


def test_pending_domain_close_remains_false_and_completed_domains_are_not_reclosed():
    pending, other = Domain("pending"), Domain("other")
    done = False
    def close_pending():
        pending.closed += 1
        return {"ok": done, "pending_providers": [] if done else ["blocked_sensor"]}
    pending.close = close_pending
    rt = RobotRuntime({d.domain_id: d for d in (pending, other)})
    for _ in range(2):
        result = rt.close()
        assert not result["ok"]
        assert result["domains"]["pending"]["pending_providers"] == ["blocked_sensor"]
    assert other.closed == 1
    done = True
    assert rt.close()["ok"]
    assert pending.closed == 3 and other.closed == 1


def test_graceful_composed_shutdown_preserves_hardware_park_before_torque_off(monkeypatch):
    from cascade.apps.robot_runtime import DomainAdapter
    from test_shutdown_drive_state import hardware, runtime_for, assert_park_before_torque_off
    monkeypatch.setattr("cascade.control.arm_base.time.sleep", lambda _seconds: None)
    events = []
    arm = hardware(events)
    domain = DomainAdapter("manipulation", {"kind": "manipulation"}, (), [], (),
                           runtime=runtime_for(arm, retained="held_object"), owner=arm.raw)
    rt = RobotRuntime({"manipulation": domain})
    assert rt.request_shutdown()["ok"]
    assert not arm.harness.estopped  # graceful cancellation is not a manufactured e-stop
    assert rt.close()["ok"]
    assert_park_before_torque_off(events)


def test_graceful_shutdown_cannot_weaken_a_previous_explicit_estop():
    domain = Domain()
    graceful = []
    domain.request_shutdown = lambda: graceful.append(True) or {"ok": True}
    rt = RobotRuntime({domain.domain_id: domain})
    assert rt.stop()["ok"]
    assert rt.request_shutdown()["ok"]
    assert domain.stops == 2 and not graceful
    rt.close()


@pytest.mark.parametrize("boundary", ["validation", "last_dispatch"])
def test_local_deadline_is_rechecked_at_admission_and_last_dispatch(monkeypatch, boundary):
    from types import SimpleNamespace
    import cascade.robotics.runtime as module
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    clock = [1.0]
    # Only this module's time reference is replaced. No global clock or sleep
    # change; a deterministic scheduling delay crosses the actual admission gate.
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    if boundary == "validation":
        original = runtime._arguments
        def delayed_validation(descriptor, args):
            original(descriptor, args)
            clock[0] = 3.0
        monkeypatch.setattr(runtime, "_arguments", delayed_validation)
    else:
        class DelayedLookup(dict):
            def __getitem__(self, name):
                clock[0] = 3.0
                return super().__getitem__(name)
        runtime.domains = DelayedLookup(runtime.domains)
    result = runtime.execute("locomotion.move", {"distance": .1}, expected_generation=0,
                             deadline_monotonic_s=2.0)
    assert not result["ok"] and "deadline expired" in result["error"]
    assert domain.calls == []
    assert runtime.execute("emergency_stop", {}, expected_generation=0, deadline_monotonic_s=2.0)["ok"]
    runtime.close()


@pytest.mark.parametrize("deadline", [True, "3", float("inf"), float("nan")])
def test_invalid_episode_deadline_is_not_admitted(deadline):
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    assert not runtime.execute("locomotion.move", {"distance": .1}, deadline_monotonic_s=deadline)["ok"]
    assert domain.calls == []
    runtime.close()
