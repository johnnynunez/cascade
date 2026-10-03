"""Software-only geometric-distance evidence; fixtures are not physical proof."""
from copy import deepcopy
import math

import pytest

from cascade.agent.base_effects import _Window
from test_mobile_effects import (ScriptedReader, checker_for, fixture_motion_receipt,
                                limits, state)


def evaluate(*, distance=.01, scale=1., yaw=0., lateral=0., receipt=None,
             transform=None, start=.02, kind="physics", args=None):
    """Drive the real acceptance/evaluation path with explicit completed clocks.

    Wall stamps are fixture data only. No global clock, production threshold or
    reader is patched, and no physical progression is inferred from wall sleeps.
    """
    checker = checker_for(lambda: None)
    op = _Window("fixture", "walk_distance", {"distance_m": distance} if args is None else args,
                 started=100., deadline=103.)
    result = fixture_motion_receipt() if receipt is None else deepcopy(receipt)
    result["ack"].update(start_sim_time_s=start, end_sim_time_s=start + 3.)
    for n in range(1, 14):
        x = min(n-1, 5) * .002 * scale
        y = min(n-1, 5) * lateral / 5
        stamp = 100. + n*.002 if n < 9 else 100.2 + (n-8)*.002
        if n == 9:
            op.finished = 100.2
        value = state(n, generation=0 if n == 1 else 1 if n <= 7 else 2,
            position_world=(math.cos(yaw)*x-math.sin(yaw)*y, math.sin(yaw)*x+math.cos(yaw)*y, .3),
            orientation_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2)),
            received_monotonic_s=stamp+.001, measurement_kind=kind)
        if transform is not None:
            value = transform(n, value)
        op.attempts += 1
        checker._accept(op, value, stamp, stamp+.001)
    verdict = checker._receipt(op, result)
    checker._evaluate(op, verdict)
    checker.close()
    return verdict


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("heading", [0., math.pi/2, -math.pi/2])
def test_signed_distance_uses_independent_body_progress_and_all_existing_gates(sign, heading):
    verdict = evaluate(distance=sign*.01, scale=sign, yaw=heading)
    assert verdict["status"] == "confirmed", verdict["reason"]
    assert verdict["metrics"]["body_displacement_m"] == pytest.approx([sign*.01, 0.])
    assert verdict["metrics"]["settle_sim_duration_s"] >= limits()["settle_window_s"]
    assert verdict["evidence"]["support_checks"]
    interval = verdict["evidence"]["effect_interval"]
    assert interval["last_sim_time_s"] < interval["admitted_end_sim_time_s"]
    assert interval["generation"] == 1 and interval["completion_generation"] == 2


@pytest.mark.parametrize("distance,scale,reason", [
    (.01, 0., "no effect"), (.01, -1., "wrong sign"), (-.01, 1., "wrong sign"),
    (.01, .2, "insufficient progress"), (.01, 3., "excess progress"),
    (.1, 1., "insufficient progress"),
])
def test_distance_matches_caller_target_not_internal_policy_speed_or_cap(distance, scale, reason):
    verdict = evaluate(distance=distance, scale=scale)
    assert verdict["status"] == "refuted", verdict
    assert reason in verdict["reason"]


@pytest.mark.parametrize("target", [None, True, float("nan"), float("inf"), "0.01"])
def test_invalid_distance_intent_never_confirms(target):
    verdict = evaluate(args={"distance_m": target})
    assert verdict["status"] == "unverified"
    assert verdict["reason"] == "invalid caller intent"


@pytest.mark.parametrize("target", [0., 1e-12, -.001])
def test_zero_or_subresolution_distance_is_not_balance_success(target):
    verdict = evaluate(distance=target, scale=0.)
    assert verdict["status"] == "unverified"
    assert "resolution" in verdict["reason"]


def test_pre_admission_travel_does_not_supply_geometric_goal():
    # All .01m travel occurred by .12s; admission starts at .13s. No interpolation
    # from the pre-admission .12 sample is permitted even though it is close.
    verdict = evaluate(start=.13)
    assert verdict["status"] == "refuted"
    assert verdict["metrics"]["body_displacement_m"] == [0., 0.]
    assert verdict["evidence"]["effect_interval"]["baseline_sim_time_s"] == .14


def test_lateral_and_heading_drift_are_independent_vetoes():
    from dataclasses import replace
    lateral = evaluate(lateral=.004)
    assert lateral["status"] == "refuted" and "vy drift" in lateral["reason"]
    def turning(n, value):
        angle = min(n-1, 5)*.01
        return replace(value, orientation_wxyz=(math.cos(angle/2), 0., 0., math.sin(angle/2)))
    heading = evaluate(transform=turning)
    assert heading["status"] == "refuted" and "wz drift" in heading["reason"]


@pytest.mark.parametrize("fault,reason", [
    ("fallen", "fallen"), ("support", "forbidden external"),
    ("unavailable", "support unavailable"), ("residual", "did not settle"),
    ("stale", "stale"), ("generation", "generation"), ("identity", "identity"),
])
def test_distance_cannot_bypass_channel_support_posture_or_rest_vetoes(fault, reason):
    from dataclasses import replace
    def corrupt(n, value):
        if n == 4 and fault == "fallen":
            return replace(value, fallen=True)
        if n == 4 and fault in {"support", "unavailable"}:
            payload = value.as_dict()["support"]
            if fault == "support":
                payload["contacts"][0]["shape_b"] = "/Fixture/Robot/body"
            else:
                payload.update(status="unavailable", reason="fixture unavailable", contacts=[])
            return replace(value, support=payload)
        if n >= 9 and fault == "residual":
            return replace(value, linear_velocity_world=(.02, 0., 0.))
        if n == 4 and fault == "stale":
            return replace(value, producer_age_s=1.)
        if n == 4 and fault == "generation":
            return replace(value, generation=4)
        if n == 4 and fault == "identity":
            # state() regenerates a same-solve support payload for this altered digest.
            raw = value.as_dict()
            raw.pop("support")
            raw["model_identity_sha256"] = "f"*64
            raw.pop("step")
            return state(n, **raw)
        return value
    verdict = evaluate(transform=corrupt)
    assert verdict["status"] != "confirmed", verdict
    assert reason in verdict["reason"]


def test_actor_travel_never_replaces_independent_measurement_or_execution_failure():
    receipt = fixture_motion_receipt()
    receipt.update(measured_distance_m=.01, measured={"dx": .01})
    inert = evaluate(scale=0., receipt=receipt)
    assert inert["status"] == "refuted"
    receipt["execution_ok"] = False
    moved = evaluate(receipt=receipt)
    assert moved["status"] == "unverified" and "execution failed" in moved["reason"]


@pytest.mark.parametrize("which,field,value", [
    ("ack", "accepted", False), ("ack", "model_identity_sha256", "0"*64),
    ("stop_ack", "generation", 5), ("stop_ack", "latched", True),
    ("stop_ack", "delivery_uncertain", True),
])
def test_distance_requires_bound_admission_and_completion_receipts(which, field, value):
    receipt = fixture_motion_receipt()
    receipt[which][field] = value
    verdict = evaluate(receipt=receipt)
    assert verdict["status"] == "unverified" and "unbound command evidence" in verdict["reason"]


def test_synthetic_mock_travel_cannot_confirm_physical_distance():
    verdict = evaluate(kind="kinematic_mock")
    assert verdict["status"] == "unverified" and "not physical evidence" in verdict["reason"]


def test_distance_sampler_lifecycle_reaches_geometric_verdict_and_closes():
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n-1, 5)*.002, 0., .3),
        generation=0 if n == 1 else 1 if n <= 8 else 2))
    checker = checker_for(reader)
    receipt = fixture_motion_receipt()
    receipt["ack"]["end_sim_time_s"] = 3.02
    try:
        token = checker.begin("walk_distance", {"distance_m": .01})
        reader.wait(8)
        result = checker.finish(token, receipt)
        assert result["status"] == "confirmed", result["reason"]
        assert result["metrics"]["body_displacement_m"][0] == pytest.approx(.01)
    finally:
        checker.close()
    assert reader.closed


def test_post_completion_travel_does_not_repair_an_inert_admitted_command():
    from dataclasses import replace
    def after_stop(n, value):
        return replace(value, position_world=(max(0, n-8)*.002, 0., .3))
    verdict = evaluate(transform=after_stop)
    assert verdict["status"] == "refuted"
    assert verdict["metrics"]["body_displacement_m"] == [0., 0.]
    assert "no effect" in verdict["reason"]
    assert verdict["evidence"]["effect_interval"]["last_step"] == 8
