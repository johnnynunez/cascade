"""Offline synthetic recording alarms; never physical support acceptance."""

import copy

import pytest

from benchmark.rgbd.observation_health import inspect_states
from test_mobile_effects import state, fixture_support_contract


def recording():
    rows = []
    for step in range(3, 83):
        value = state(step, sim_time_s=step * 0.005).as_dict()
        if step == 3:
            value["support"]["contacts"] = []
        rows.append(
            {
                "step": step,
                "sim_time": step * 0.005,
                "controller": value,
                "fallen": False,
                "permission_generation_at_sample": 0,
            }
        )
    identity = {
        "model_identity_sha256": "e" * 64,
        "support_contract": fixture_support_contract(),
        "recipe": {"native": {"actual_physics_dt": 0.005}},
    }
    return rows, identity


def check(rows, identity):
    return inspect_states(
        rows,
        identity,
        epoch="fixture-epoch",
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
