"""Ineligible contact observations retain veto power, never positive credit."""

import copy

import pytest

from cascade.agent.base_effects import BasePostconditionChecker, _Window
from test_mobile_effects import fixture_support, fixture_support_contract, limits, state


def window():
    lim = limits()
    checker = BasePostconditionChecker(
        lambda: pytest.fail("no worker expected"),
        limits=lim,
        support_contract=fixture_support_contract(),
    )
    boundary = dict(
        ack_monotonic_s=100.0,
        robot_id="synthetic-microduck",
        source="scripted-software-fixture",
        epoch="fixture-epoch",
        generation=0,
        model_identity_sha256="e" * 64,
    )
    op = _Window(
        "support",
        "stop_navigation",
        {},
        100.0,
        100.0 + lim["max_wall_duration_s"],
        finished=100.0,
        stop_boundary=boundary,
    )
    return checker, op


@pytest.mark.parametrize("kind", ["pending", "late", "cancelled"])
def test_healthy_first_exclusion_never_establishes_baseline(kind):
    checker, op = window()
    try:
        end = (
            100.0 + checker._limits["settle_timeout_s"] + 0.005
            if kind == "late"
            else 100.01
        )
        begin = end - 0.002
        if kind == "cancelled":
            op.cancel.set()
        op.attempts = 1
        value = state(
            1,
            received_monotonic_s=end,
            producer_age_s=0.02 if kind == "pending" else 0.0,
        )
        checker._accept(op, value, begin, end)
        assert not op.samples and not op.first.is_set()
        assert not op.channel_failed
    finally:
        checker.close()


@pytest.mark.parametrize("failure", ["unknown", "forbidden", "no_load", "healthy"])
@pytest.mark.parametrize("kind", ["late", "pending"])
def test_excluded_support_cannot_be_hidden_by_an_older_supported_window(kind, failure):
    checker, op = window()
    try:
        for n, sim in enumerate((0.0, 0.04, 0.08, 0.12), 1):
            end = 100.0 + n * 0.005
            op.attempts += 1
            checker._accept(
                op, state(n, sim_time_s=sim, received_monotonic_s=end), end - 0.002, end
            )
        original = copy.deepcopy(op.samples)
        end = (
            (100.0 + checker._limits["settle_timeout_s"] + 0.005)
            if kind == "late"
            else 100.05
        )
        observed = fixture_support(5, sim_time_s=0.14)
        if failure == "unknown":
            observed.update(
                status="unavailable", reason="software observation missing", contacts=[]
            )
        elif failure == "forbidden":
            observed["contacts"][0]["shape_b"] = "/Fixture/Robot/body"
        elif failure == "no_load":
            observed["contacts"] = []
        value = state(
            5,
            sim_time_s=0.14,
            received_monotonic_s=end,
            producer_age_s=0.1 if kind == "pending" else 0.0,
            support=observed,
        )
        op.attempts += 1
        checker._accept(op, value, end - 0.002, end)
        verdict = checker._receipt(op, {"ok": True})
        checker._evaluate(op, verdict)
        assert op.samples == original, (
            "an exclusion supplied positive count or duration"
        )
        if failure == "healthy":
            assert verdict["status"] == "confirmed", verdict
        else:
            assert verdict["status"] != "confirmed", verdict
    finally:
        checker.close()
