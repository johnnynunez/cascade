"""Synthetic software decisions, never native force or locomotion proof."""
from dataclasses import FrozenInstanceError
import json

import pytest

from mobile_tick_fixture import healthy_episode_gc as healthy_episode_gc  # noqa: PLC0414 — shared pytest fixture

from cascade.agent.base_effects import BasePostconditionChecker
from cascade.control.mobile_base import BaseState
from cascade.control.mobile_support import support_contract
from test_mobile_effects import (
    ScriptedReader, fixture_support, fixture_support_contract, limits, run_window, state,
)
from test_mobile_effects import truth_server as truth_server


def supported(n, *, contacts=None, **changes):
    record = fixture_support(n)
    if contacts is not None:
        record["contacts"] = contacts
    record.update(changes)
    return state(n, support=record)


def contact(**changes):
    value = fixture_support(1)["contacts"][0]
    value.update(changes)
    return value


def test_support_wire_roundtrip_freezes_nested_data_and_defensively_copies():
    source = fixture_support(1)
    observed = state(1, support=source)
    source["contacts"][0]["force_on_b_world_n"][2] = 100.
    assert observed.support.contacts[0].force_on_b_world_n == (0., 0., 1.)
    with pytest.raises(FrozenInstanceError):
        observed.support.contacts[0].normal_force_n = 100.
    assert BaseState.from_dict(json.loads(json.dumps(observed.as_dict()))) == observed


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("version", True), ("step", 2), ("sim_time_s", .04),
    ("epoch", "new-epoch"), ("model_identity_sha256", "f" * 64),
    ("model_identity_sha256", "E" * 64), ("status", "estimated"),
    ("status", "unavailable"),
])
def test_schema_refuses_stale_unbound_partial_or_unknown_evidence(field, value):
    record = fixture_support(1)
    record[field] = value
    with pytest.raises(ValueError):
        state(1, support=record)


@pytest.mark.parametrize("changes", [
    {"normal_force_n": -1.}, {"normal_force_n": float("nan")},
    {"normal_force_n": 2.}, {"force_on_b_world_n": [0., 0., -1.]},
    {"force_on_b_world_n": [float("inf"), 0., 1.]},
    {"normal_a_to_b_world": [0., 0., 0.]}, {"shape_a": "ground"},
    {"shape_b_id": True}, {"shape_b_id": 0}, {"shape_b": "/Fixture/Ground"},
    {"point_world_m": [0., 0., float("nan")]},
])
def test_invalid_force_and_shape_identity_never_enter_independent_history(changes):
    with pytest.raises(ValueError):
        supported(1, contacts=[contact(**changes)])


def test_schema_refuses_conflicting_native_id_to_shape_map():
    with pytest.raises(ValueError, match="conflicting"):
        supported(1, contacts=[contact(), contact(shape_b="/Fixture/Robot/body")])


@pytest.mark.parametrize("changes", [
    {"version": True}, {"model_identity_sha256": "unknown"},
    {"robot_shapes": []}, {"foot_shapes": ["/Unregistered/Foot"]},
    {"ground_shapes": ["/Fixture/Robot/body"]}, {"ground_shapes": ["relative"]},
    {"gravity_world_m_s2": [0., 0., 0.]}, {"unexpected": True},
])
def test_contract_requires_explicit_disjoint_complete_admission(changes):
    contract = fixture_support_contract()
    contract.update(changes)
    with pytest.raises(ValueError):
        support_contract(contract)


@pytest.mark.parametrize("reverse", [False, True])
def test_one_loaded_sole_proves_rest_and_pair_order_does_not_change_meaning(reverse):
    loaded = contact()
    if reverse:
        for a, b in (("shape_a", "shape_b"), ("shape_a_id", "shape_b_id")):
            loaded[a], loaded[b] = loaded[b], loaded[a]
        for key in ("force_on_b_world_n", "normal_a_to_b_world"):
            loaded[key] = [-v for v in loaded[key]]
    verdict = run_window(ScriptedReader(lambda n: supported(n, contacts=[loaded])), "stop_navigation", {})
    assert verdict["status"] == "confirmed", verdict["reason"]
    assert verdict["evidence"]["support_contract"] == support_contract(fixture_support_contract())


@pytest.mark.parametrize("contacts", [
    [],
    [contact(force_on_b_world_n=[0., 0., 0.], normal_force_n=0.)],
    [contact(force_on_b_world_n=[1., 0., 0.], normal_a_to_b_world=[1., 0., 0.])],
    [contact(force_on_b_world_n=[0., 0., -1.], normal_a_to_b_world=[0., 0., -1.])],
    # Upward tangential friction is insufficient when the compressive normal is lateral.
    [contact(force_on_b_world_n=[1., 0., 1.], normal_a_to_b_world=[1., 0., 0.])],
])
def test_stationary_pose_requires_upward_solved_sole_load(contacts):
    verdict = run_window(ScriptedReader(lambda n: supported(n, contacts=contacts)), "stop_navigation", {})
    assert verdict["status"] == "refuted", verdict["reason"]
    assert "positive upward" in verdict["reason"]


@pytest.mark.parametrize("missing", ["record", "unavailable", "contract", "wrong_contract"])
def test_unknown_force_or_admission_is_unverified_despite_stationary_pose(missing):
    def make(n):
        if missing == "record":
            return state(n, support=None)
        if missing == "unavailable":
            return supported(n, contacts=[], status="unavailable", reason="solver force API unavailable")
        return state(n)
    contract = fixture_support_contract()
    if missing == "wrong_contract":
        contract["model_identity_sha256"] = "f" * 64
    checker = BasePostconditionChecker(ScriptedReader(make), limits=limits(),
                                      support_contract=None if missing == "contract" else contract)
    try:
        verdict = checker.finish(checker.begin("stop_navigation", {}), {"ok": True})
        assert verdict["status"] == "unverified", verdict["reason"]
        assert "support" in verdict["reason"]
    finally:
        checker.close()


@pytest.mark.parametrize("forbidden", [
    contact(shape_b="/Fixture/Robot/body", shape_b_id=3),
    contact(shape_a="/Fixture/Wall", shape_a_id=4),
])
def test_forbidden_external_reaction_refutes_even_after_full_recovery(forbidden):
    reader = ScriptedReader(lambda n: state(n,
        support={**fixture_support(n), "contacts": [forbidden]} if n == 4 else fixture_support(n),
        position_world=(min(n - 1, 5) * .002, 0., .3)))
    verdict = run_window(reader)
    assert verdict["status"] == "refuted", verdict["reason"]
    assert "forbidden external" in verdict["reason"]


def test_self_contact_neither_supplies_support_nor_refutes_allowed_support():
    self_contact = contact(shape_a="/Fixture/Robot/body", shape_a_id=3)
    for contacts, expected in (([self_contact], "refuted"), ([contact(), self_contact], "confirmed")):
        verdict = run_window(ScriptedReader(lambda n: supported(n, contacts=contacts)), "stop_navigation", {})
        assert verdict["status"] == expected, verdict["reason"]


@pytest.mark.parametrize("vx,expected", [(.1, "confirmed"), (0., "refuted")])
def test_gait_flight_is_allowed_but_zero_twist_balance_requires_continuous_load(vx, expected, healthy_episode_gc):
    # This semantic support episode excludes unrelated cyclic heap collection;
    # the real sampler and its 40 ms read budget remain unchanged.
    def make(n):
        record = fixture_support(n)
        if n == 4:
            record["contacts"] = []
        return state(n, support=record, position_world=(min(n - 1, 5) * .02 * vx, 0., .3))
    verdict = run_window(ScriptedReader(make), args=dict(vx=vx, vy=0., wz=0., duration_s=.1))
    observed = verdict["evidence"]["observations"]
    diagnostic = {"reason": verdict["reason"], "attempts": verdict["evidence"]["attempts"],
                  "rejected": verdict["evidence"]["rejected"],
                  "max_read_s": max((s["observed_monotonic_s"] - s["read_started_monotonic_s"]
                                     for s in observed), default=None)}
    assert verdict["status"] == expected, diagnostic
    flight = next(s for s in observed if s["state"]["step"] == 4)
    assert flight["valid"] and flight["confirmation_eligible"]
    assert flight["state"]["support"]["contacts"] == ()
    assert any(s["state"]["step"] == 4 and s["phase"] == "during"
               for s in verdict["evidence"]["samples"])
    support_check = next(s for s in verdict["evidence"]["support_checks"] if s["step"] == 4)
    assert support_check["status"] == expected


def test_support_episode_gc_isolation_does_not_hide_a_late_reader(healthy_episode_gc):
    import gc
    import time
    captured = []
    def make(n):
        value = state(n, generation=1 if n == 4 else 0,
                      position_world=(min(n - 1, 5) * .002, 0., .3))
        if n == 4:
            assert not gc.isenabled()
            captured.append(value)
            time.sleep(2 * limits()["read_timeout_s"])
        return value
    verdict = run_window(ScriptedReader(make))
    assert verdict["status"] == "unverified" and verdict["reason"] == "reader_timeout"
    assert verdict["limits"]["read_timeout_s"] == .04
    assert verdict["evidence"]["channel_failed"]
    late = next(s for s in verdict["evidence"]["observations"] if s["state"]["step"] == 4)
    assert late["state"] == captured[0].as_dict()  # No capture timestamp rejuvenation.
    assert late["observed_monotonic_s"] - late["read_started_monotonic_s"] > .04
    assert any(s["state"]["step"] > 4 for s in verdict["evidence"]["observations"])


def test_flight_does_not_waive_unknown_contact_channel_during_walking():
    def make(n):
        record = fixture_support(n)
        if n == 4:
            record.update(status="unavailable", reason="incomplete solve extraction", contacts=[])
        return state(n, support=record, position_world=(min(n - 1, 5) * .002, 0., .3))
    verdict = run_window(ScriptedReader(make))
    assert verdict["status"] == "unverified", verdict["reason"]


def test_stop_fence_cannot_bind_another_model_even_with_same_epoch():
    import time
    checker = BasePostconditionChecker(ScriptedReader(), limits=limits(), support_contract=fixture_support_contract())
    try:
        token = checker.begin("stop_navigation", {}, stop_boundary=dict(
            ack_monotonic_s=time.monotonic(), robot_id="synthetic-microduck", source="scripted-software-fixture",
            epoch="fixture-epoch", generation=0, model_identity_sha256="f" * 64))
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified"
        assert "model_identity_sha256 mismatch" in verdict["reason"]
    finally:
        checker.close()


@pytest.mark.parametrize("mutation", ["omit_body", "wrong_ground", "wrong_sole"])
def test_truth_admission_rejects_registry_changes_with_unchanged_model_digest(truth_server, mutation):
    from copy import deepcopy
    from cascade.sim.base_truth import BaseTruthReader
    profile, payload, requests = truth_server
    # The producer hello keeps its original independently admitted registry.
    altered = deepcopy(profile)
    contract = altered["support_contract"]
    if mutation == "omit_body":
        contract["robot_shapes"].remove("/Fixture/Robot/body")
    elif mutation == "wrong_ground":
        contract["ground_shapes"] = ["/Fixture/Wall"]
    else:
        contract["foot_shapes"] = ["/Fixture/Robot/body"]
    assert altered["model_identity_sha256"] == payload["hello"]["model_identity_sha256"]
    reader = BaseTruthReader(altered)
    try:
        assert reader() is None
        assert "support" in reader.last_error
        assert requests == [{"op": "hello", "role": "reader"}]
    finally:
        reader.close()


@pytest.mark.parametrize("pending_step", [1, 3])
@pytest.mark.parametrize("support_kind", ["forbidden", "unavailable", "missing"])
def test_pending_stop_support_failure_is_retained_before_later_healthy_readings(pending_step, support_kind):
    import time
    def make(n):
        support = fixture_support(n)
        if n == pending_step:
            if support_kind == "forbidden":
                support["contacts"] = [contact(shape_b="/Fixture/Robot/body", shape_b_id=3)]
            elif support_kind == "unavailable":
                support.update(status="unavailable", reason="incomplete solved rows", contacts=[])
            else:
                support = None
        return state(n, support=support, producer_age_s=.15 if n == pending_step else 0.)
    reader = ScriptedReader(make)
    checker = BasePostconditionChecker(reader, limits=limits(), support_contract=fixture_support_contract())
    try:
        token = checker.begin("stop_navigation", {}, stop_boundary=dict(
            ack_monotonic_s=time.monotonic(), robot_id="synthetic-microduck", source="scripted-software-fixture",
            epoch="fixture-epoch", generation=0, model_identity_sha256="e" * 64))
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified", verdict["reason"]
        assert "unsafe pre-ACK support" in verdict["reason"]
        assert verdict["evidence"]["channel_failed"] is True
        pending = verdict["evidence"]["temporal_pending"]
        assert len(pending) == 1
        assert pending[0]["state"]["step"] == pending_step
        assert pending[0]["support_status"] == ("refuted" if support_kind == "forbidden" else "unverified")
        assert pending[0]["capture_margin_s"] < 0
        assert all(e["state"]["step"] != pending_step for e in verdict["evidence"]["samples"])
        assert reader.calls == pending_step  # quarantine before the scripted recovery can mask it
    finally:
        checker.close()


def test_pending_stop_flight_does_not_receive_or_require_rest_credit():
    import time
    reader = ScriptedReader(lambda n: state(n,
        support={**fixture_support(n), "contacts": []} if n == 1 else fixture_support(n),
        producer_age_s=.15 if n == 1 else 0.))
    checker = BasePostconditionChecker(reader, limits=limits(), support_contract=fixture_support_contract())
    try:
        token = checker.begin("stop_navigation", {}, stop_boundary=dict(
            ack_monotonic_s=time.monotonic(), robot_id="synthetic-microduck", source="scripted-software-fixture",
            epoch="fixture-epoch", generation=0, model_identity_sha256="e" * 64))
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "confirmed", verdict["reason"]
        assert verdict["evidence"]["temporal_pending"][0]["state"]["support"]["contacts"] == ()
        assert verdict["evidence"]["samples"][0]["state"]["step"] == 2
        assert verdict["metrics"]["settle_samples"] >= 3
    finally:
        checker.close()


def test_support_gc_isolation_preserves_real_tcp_read_deadline(truth_server, healthy_episode_gc):
    import gc
    import threading
    from cascade.sim.base_truth import BaseTruthReader

    profile, payload, requests = truth_server
    entered, release = threading.Event(), threading.Event()
    captured = []
    def delayed():
        assert not gc.isenabled()
        value = state(1).as_dict()
        captured.append(value)
        entered.set()
        assert release.wait(2), "test did not release its blocked TCP response"
        return {"ok": True, "state": value}
    payload["state"] = delayed
    reader = BaseTruthReader(profile)
    checker = BasePostconditionChecker(reader, limits=limits(), support_contract=fixture_support_contract())
    try:
        token = checker.begin("stop_navigation", {})  # Must time out while reply is held.
        assert entered.wait(2)
        release.set()
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified" and verdict["reason"] == "reader_timeout"
        assert verdict["limits"]["read_timeout_s"] == .04
        assert verdict["evidence"]["channel_failed"]
        # Baseline availability is timed from begin, before the worker may be
        # scheduled; a late-starting worker can have a shorter individual RTT.
        evidence = verdict["evidence"]
        assert evidence["finished_monotonic_s"] - evidence["started_monotonic_s"] >= .04
        observed = evidence["observations"]
        if observed:
            assert len(observed) == 1
            assert observed[0]["state"]["step"] == captured[0]["step"] == 1
            assert observed[0]["state"]["support"] == captured[0]["support"]
        else:
            # A scheduler delay can also exhaust the unchanged socket budget;
            # that stronger transport refusal still cannot become a confirmation.
            assert "timed out" in reader.last_error or "deadline" in reader.last_error
        assert [r["op"] for r in requests] == ["hello", "state"]
    finally:
        release.set()
        checker.close()
