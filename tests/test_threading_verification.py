"""Fastener evidence, including counterexamples that wrist motion cannot pass."""
from dataclasses import replace
import math

import numpy as np
import pytest

from cascade.sim.threading_verification import ThreadContract, ThreadSample, verify_threading


def samples(*, turns=1.0, advance=0.002, direction="tighten"):
    sign = -1 if direction == "tighten" else 1
    rows = []
    for step, fraction in enumerate(np.linspace(0, 1, 121)):
        yaw = sign * 2 * math.pi * turns * fraction
        rows.append(ThreadSample(
            epoch="physical-episode", step=step, time_s=step / 60,
            fastener_id="nut", fixture_id="bolt",
            fastener_position_m=(0, 0, 0.04 + sign * advance * fraction),
            fastener_quaternion_xyzw=(0, 0, math.sin(yaw / 2), math.cos(yaw / 2)),
            fixture_position_m=(0, 0, 0), fixture_quaternion_xyzw=(0, 0, 0, 1),
            thread_contacts=3, tool_contacts=2,
        ))
    return rows


@pytest.mark.parametrize("direction", ["tighten", "loosen"])
def test_signed_actual_fastener_rotation_and_pitch(direction):
    verdict = verify_threading(samples(direction=direction), ThreadContract(0.002, direction=direction))
    assert verdict["status"] == "confirmed", verdict
    assert verdict["measured"]["turns"] == pytest.approx(1)
    assert verdict["measured"]["axial_advance_m"] == pytest.approx(0.002)
    assert verdict["seating_verified"] is False


@pytest.mark.parametrize("rows,failed", [
    (samples(turns=0, advance=0), "requested_fastener_rotation"),
    (samples(turns=1, advance=0), "thread_pitch"),
    (samples(turns=0, advance=0.002), "thread_pitch"),
    (samples(turns=1, advance=0.006), "thread_pitch"),
    (samples(direction="loosen"), "requested_fastener_rotation"),
    ([replace(s, tool_contacts=0) for s in samples()], "thread_and_tool_contact"),
    ([replace(s, thread_contacts=0) for s in samples()], "thread_and_tool_contact"),
    ([replace(s, fastener_position_m=(0.01, 0, s.fastener_position_m[2])) for s in samples()], "axis_alignment"),
])
def test_physical_counterexamples_are_refuted(rows, failed):
    verdict = verify_threading(rows, ThreadContract(0.002))
    assert verdict["status"] == "refuted", verdict
    assert not verdict["checks"][failed]
    assert not verdict["threading_verified"]


def test_fixture_motion_cannot_manufacture_advancement():
    rows = samples()
    rows = [replace(s, fastener_position_m=(0, 0, 0.04),
                    fixture_position_m=(0, 0, 0.04 - s.fastener_position_m[2])) for s in rows]
    verdict = verify_threading(rows, ThreadContract(0.002))
    assert verdict["status"] == "refuted"
    assert not verdict["checks"]["fixed_fixture"]


@pytest.mark.parametrize("change,reason", [
    ({"epoch": "reset"}, "epoch"),
    ({"fastener_id": "wrist"}, "identity"),
    ({"step": 0}, "clock"),
    ({"time_s": 0}, "clock"),
    ({"time_s": float("nan")}, "clock"),
    ({"thread_contacts": -1}, "contact"),
    ({"fastener_quaternion_xyzw": (0, 0, 0, 2)}, "normalized"),
])
def test_invalid_provenance_or_state_cannot_confirm(change, reason):
    rows = samples()
    rows[-1] = replace(rows[-1], **change)
    verdict = verify_threading(rows, ThreadContract(0.002))
    assert verdict["status"] == "unverified"
    assert reason in verdict["reason"]
    assert not verdict["threading_verified"]


def test_large_sampling_gaps_cannot_hide_full_turns():
    verdict = verify_threading(samples()[::30], ThreadContract(0.002))
    assert verdict["status"] == "unverified"
    assert "ambiguous" in verdict["reason"]


def test_declared_angular_speed_is_checked_against_pose_samples():
    verdict = verify_threading(samples(), ThreadContract(0.002, max_angular_speed_rad_s=1))
    assert verdict["status"] == "unverified"
    assert "speed bound" in verdict["reason"]


def test_tolerances_must_not_admit_zero_axial_motion():
    verdict = verify_threading(samples(), ThreadContract(0.002, requested_turns=0.1))
    assert verdict["status"] == "unverified"
    assert "zero axial" in verdict["reason"]


def test_one_contact_at_start_does_not_certify_free_fall():
    rows = [samples()[0]] + [replace(s, thread_contacts=0, tool_contacts=0) for s in samples()[1:]]
    verdict = verify_threading(rows, ThreadContract(0.002))
    assert verdict["status"] == "refuted"
    assert not verdict["checks"]["thread_and_tool_contact"]
