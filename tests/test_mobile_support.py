"""Synthetic software decisions, never native force or locomotion proof."""
from dataclasses import FrozenInstanceError
import json

import pytest

from cascade.agent.base_effects import BasePostconditionChecker
from cascade.control.mobile_base import BaseState
from cascade.control.mobile_support import support_contract
from test_mobile_effects import (
    ScriptedReader, fixture_support, fixture_support_contract, limits, run_window, state, truth_server,
)


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
def test_gait_flight_is_allowed_but_zero_twist_balance_requires_continuous_load(vx, expected):
    def make(n):
        record = fixture_support(n)
        if n == 4:
            record["contacts"] = []
        return state(n, support=record, position_world=(min(n - 1, 5) * .02 * vx, 0., .3))
    verdict = run_window(ScriptedReader(make), args=dict(vx=vx, vy=0., wz=0., duration_s=.1))
    assert verdict["status"] == expected, verdict["reason"]


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
