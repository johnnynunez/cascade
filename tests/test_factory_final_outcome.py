"""Synthetic passive-source outcomes, not native rollback or physical evidence."""
from dataclasses import replace
import math

import pytest

from cascade.skills.fastening_runtime import FasteningDomain
from cascade.sim.threading_verification import ThreadContract, verify_threading
from test_fastening_runtime import ScriptedOwner, row


def episode(mode="hold", *, old_generation_rows=0):
    owner = ScriptedOwner()
    original_read = owner.read
    measured = []
    stop_turns = None

    def read(cursor, *, timeout_s):
        nonlocal stop_turns
        sample, = original_read(cursor, timeout_s=timeout_s)
        if owner.stopped:
            if stop_turns is None:
                stop_turns = min(owner.stop_step*.005, .975)
            elapsed_steps = sample.step-owner.stop_step
            rollback = elapsed_steps*.005 if mode in {"unwind", "partial_unwind"} else 0.
            floor = .80 if mode == "partial_unwind" else 0.
            turns = max(floor, stop_turns-rollback) if rollback else stop_turns
            moving = bool(rollback) and turns > floor
            sample = row(sample.step, turns=turns, generation=owner.generation,
                captured=owner.now,
                fastener_angular_speed_rad_s=math.pi if moving else 0.,
                spindle_speed_rad_s=-math.pi if moving else 0.,
                tool_angular_speed_rad_s=math.pi if moving else 0.,
                fastener_linear_speed_m_s=.00125 if moving else 0.)
            if elapsed_steps <= old_generation_rows:
                sample = replace(sample, generation=owner.generation-1)
            if mode == "axial_loss":
                sample = replace(sample, fastener_position_m=(0., 0., .069))
            elif mode == "radial_loss":
                sample = replace(sample, fastener_position_m=(.002, 0., sample.fastener_position_m[2]))
            elif mode == "stale":
                sample = replace(sample, captured_monotonic_s=owner.now-.25)
            elif mode == "epoch":
                sample = replace(sample, epoch="replacement")
            elif mode == "no_rest":
                sample = replace(sample, tool_angular_speed_rad_s=.03)
        measured.append(sample)
        return (sample,)

    domain = FasteningDomain(owner, read, controller_id="synthetic/final-outcome",
                            clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    entire = verify_threading([sample.thread_sample(owner.binding) for sample in measured],
        ThreadContract(pitch_m=.0025, requested_turns=1., direction="tighten"))
    return result, entire, owner, measured


def test_retained_threaded_pose_and_existing_quiet_window_still_confirm():
    result, entire, owner, measured = episode()
    assert result["ok"] and result["verified"]
    assert result["postcondition"]["status"] == entire["status"] == "confirmed"
    assert result["rest"]["window_sim_s"] >= .5-1e-9
    assert result["synthetic"] and not result["physical_stop_verified"]
    assert owner.stopped and measured[-1].generation == owner.generation


@pytest.mark.parametrize("mode", ["unwind", "partial_unwind", "axial_loss", "radial_loss"])
def test_rest_cannot_confirm_a_threaded_outcome_that_was_lost(mode):
    result, entire, owner, measured = episode(mode)
    assert entire["status"] == "refuted", entire
    assert not result["ok"], result
    assert not result["verified"]
    assert result["postcondition"]["status"] == "refuted"
    assert result["postcondition"]["threading"] == entire
    assert result["rest"]["status"] == "confirmed"
    assert result["rest"]["last_step"] == measured[-1].step
    assert owner.stopped and measured[-1].generation == owner.generation
    assert result["synthetic"] and not result["physical_stop_verified"]


def test_inflight_old_generation_rows_do_not_disappear_from_final_pose_interval():
    result, entire, _, _ = episode("unwind", old_generation_rows=3)
    assert entire["status"] == "refuted"
    assert not result["ok"]
    assert result["postcondition"]["threading"] == entire


@pytest.mark.parametrize("mode,reason", [("stale", "stale"), ("epoch", "epoch"),
                                         ("no_rest", "deadline")])
def test_existing_rest_faults_still_prevent_final_credit(mode, reason):
    result, _, owner, _ = episode(mode)
    assert not result["ok"] and not result["verified"]
    assert result["postcondition"]["status"] == "unverified"
    assert reason in result["error"]
    assert owner.stopped and not result["physical_stop_verified"]


def test_passive_post_stop_rotation_cannot_supply_missing_admitted_turn():
    owner = ScriptedOwner()
    original_read = owner.read
    measured = []

    def read(cursor, *, timeout_s):
        sample, = original_read(cursor, timeout_s=timeout_s)
        turns = sample.step*.0005
        moving = True
        if owner.stopped:
            turns = min(.2+(sample.step-owner.stop_step)*.005, 1.1)
            moving = turns < 1.1
        sample = row(sample.step, turns=turns, generation=owner.generation,
            captured=owner.now, fastener_angular_speed_rad_s=math.pi if moving else 0.,
            spindle_speed_rad_s=-math.pi if moving else 0.,
            tool_angular_speed_rad_s=math.pi if moving else 0.,
            fastener_linear_speed_m_s=.00125 if moving else 0.)
        measured.append(sample)
        return (sample,)

    domain = FasteningDomain(owner, read, controller_id="synthetic/late-progress",
                            clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    entire = verify_threading([s.thread_sample(owner.binding) for s in measured],
        ThreadContract(pitch_m=.0025, requested_turns=1., direction="tighten"))
    assert result["pre_stop_threading"]["status"] == "refuted"
    assert result["final_threading"] == entire and entire["status"] == "confirmed"
    assert result["rest"]["status"] == "confirmed"
    assert not result["ok"] and not result["verified"]
    assert result["postcondition"]["status"] == "refuted"
    assert result["execution_ok"] is False
