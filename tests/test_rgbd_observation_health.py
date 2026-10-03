"""Offline synthetic recording alarms; never physical support acceptance."""

import copy
import math
import struct

import pytest

from benchmark.rgbd.observation_health import inspect_states
from test_mobile_effects import state, fixture_support_contract


DT = struct.unpack("<f", struct.pack("<f", 0.005))[0]


def recording():
    rows = []
    for step in range(3, 83):
        value = state(step, sim_time_s=step * DT).as_dict()
        if step == 3:
            value["support"]["contacts"] = []
        rows.append(
            {
                "step": step,
                "sim_time": step * DT,
                "controller": value,
                "fallen": False,
                "permission_generation_at_sample": 0,
            }
        )
    identity = {
        "model_identity_sha256": "e" * 64,
        "support_contract": fixture_support_contract(),
        "recipe": {"native": {"actual_physics_dt": DT}},
    }
    return rows, identity


def check(rows, identity):
    return inspect_states(
        rows,
        identity,
        epoch="fixture-epoch",
        reference_identity={"recipe": {"native": {"actual_physics_dt": DT}}},
        limits={"min_height_m": 0.06, "max_height_m": 0.3, "max_tilt_rad": 0.7},
    )


def test_keeps_first_bootstrap_record_without_claiming_known_empty_support_is_load():
    rows, identity = recording()
    before = copy.deepcopy(rows)
    report = check(rows, identity)
    assert report["passed"] and report["known_empty_contact_steps"] == [3]
    assert (
        report["physical_admission"] is False
        and report["balanced_rest_verified"] is False
    )
    assert rows == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_first",
        "duplicate",
        "epoch",
        "model",
        "clock",
        "fallen",
        "controller_fault",
        "latched",
        "generation",
        "unknown_support",
        "forbidden_contact",
        "height",
        "tilt",
    ],
)
def test_alarm_in_first_record_never_discarded_as_warmup(mutation):
    rows, identity = recording()
    row = rows[0]
    value = row["controller"]
    if mutation == "missing_first":
        rows.pop(0)
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(row)
    elif mutation == "epoch":
        value["epoch"] = "foreign"
        value["support"]["epoch"] = "foreign"
    elif mutation == "model":
        value["model_identity_sha256"] = "f" * 64
        value["support"]["model_identity_sha256"] = "f" * 64
    elif mutation == "clock":
        value["sim_time_s"] = row["sim_time"] = value["support"]["sim_time_s"] = 0.025
    elif mutation == "fallen":
        value["fallen"] = row["fallen"] = True
    elif mutation == "controller_fault":
        value["controller_status"] = "fault"
    elif mutation == "latched":
        value["latched"] = True
    elif mutation == "generation":
        value["generation"] = 1
    elif mutation == "unknown_support":
        value["support"].update(status="unavailable", reason="unvalidated solver API")
    elif mutation == "forbidden_contact":
        value["support"]["contacts"] = copy.deepcopy(
            rows[1]["controller"]["support"]["contacts"]
        )
        value["support"]["contacts"][0]["shape_b"] = "/Fixture/Robot/body"
    elif mutation == "height":
        value["position_world"][2] = 0.01
    else:
        value["orientation_wxyz"] = [0.7071067811865476, 0.7071067811865476, 0, 0]
    report = check(rows, identity)
    assert not report["passed"] and report["alarms"]


@pytest.mark.parametrize(
    "value",
    [0.005, math.nextafter(DT, 0), math.nextafter(DT, 1), 0.01, True, float("nan")],
)
def test_nearby_or_nominal_dt_does_not_replace_exact_native_contract(value):
    rows, identity = recording()
    identity["recipe"]["native"]["actual_physics_dt"] = value
    with pytest.raises(ValueError, match="exact baseline/float32"):
        check(rows, identity)


def test_changed_reference_or_missing_reference_never_implicitly_accepted():
    rows, identity = recording()
    kwargs = dict(
        epoch="fixture-epoch",
        limits={"min_height_m": 0.06, "max_height_m": 0.3, "max_tilt_rad": 0.7},
    )
    with pytest.raises(ValueError, match="exact baseline/float32"):
        inspect_states(
            rows,
            identity,
            reference_identity={"recipe": {"native": {"actual_physics_dt": 0.005}}},
            **kwargs,
        )
    with pytest.raises(TypeError, match="reference_identity"):
        inspect_states(rows, identity, **kwargs)
    result = inspect_states(
        rows, identity, reference_identity=copy.deepcopy(identity), **kwargs
    )
    assert result["passed"] and result["timestep_binding"] == {
        "actual_physics_dt": DT,
        "reference_physics_dt": DT,
        "producer_float32_5ms": DT,
        "comparison": "exact; no numeric tolerance",
    }
