"""Synthetic seating/causality contracts. These fixtures provide no native proof."""
from dataclasses import fields, replace

import pytest

from cascade.control.fastening import FasteningFault, FasteningPermit, FasteningSolve, FasteningWriteGuard, SolvedPair, check_solve
from cascade.control.fastening_seat import SeatingLimits, SeatingSolve, ShoulderContact, seating_solve
from cascade.robotics.runtime import RobotRuntime
from cascade.skills.fastening_runtime import FasteningDomain
from test_fastening_runtime import binding, limits, row


def configuration():
    lim = replace(limits(), seating=SeatingLimits())
    base = binding(lim)
    return replace(base, collider_names=(*base.collider_names, "seat"),
        allowed_contact_pairs=(*base.allowed_contact_pairs, ("seat", "nut")),
        seat_contact_pair=("seat", "nut")), lim


def observed(step=0, *, bind=None, loaded=False, **kwargs):
    bind = bind or configuration()[0]
    base = row(step, bind=bind, **kwargs)
    witnesses = ()
    if loaded:
        base = replace(base, contacts=(*base.contacts, SolvedPair("seat", "nut", 3.)),
            collision_count=3, solver_count=3)
        witnesses = (ShoulderContact(2, (.012, 0., .023), (0., 0., 1.), 3., (0., 0., 3.)),)
    return SeatingSolve(**{field.name: getattr(base, field.name) for field in fields(FasteningSolve)},
                        shoulder_contacts=witnesses)


class SeatingOwner:
    synthetic = True

    def __init__(self, mode="healthy"):
        self.binding, self.limits = configuration()
        self.generation, self.cursor, self.now = 1, 0, 10.
        self.mode, self.requested, self.stopped, self.closed = mode, False, False, False
        self.stop_step = None

    def request_seating(self, **kwargs):
        assert kwargs == {"expected_generation": 1}
        self.requested, self.generation = True, 2
        return FasteningPermit(self.binding.sha256, "epoch", 2, 0, 0., 10., 1210., 45., "seat")

    def read(self, cursor, *, timeout_s):
        assert cursor == self.cursor
        self.now += .001
        self.cursor += 1
        step = self.cursor
        turns = min(step*.005, 15.2)
        loaded = turns >= 15.2 and self.mode != "no_shoulder"
        effort = 0. if self.stopped else .05
        sample = observed(step, bind=self.binding, loaded=loaded, turns=turns,
            generation=self.generation, captured=self.now,
            spindle_effort_nm=effort, commanded_spindle_effort_nm=effort)
        if loaded and self.mode in {"wrong_height", "downward_support"}:
            witness = sample.shoulder_contacts[0]
            if self.mode == "wrong_height":
                witness = replace(witness, point_world_m=(.012, 0., .020))
            else:
                witness = replace(witness, normal_a_to_b_world=(0., 0., -1.),
                                  force_on_fastener_world_n=(0., 0., -3.))
            sample = replace(sample, shoulder_contacts=(witness,))
        if self.mode == "no_rotation":
            sample = replace(sample, fastener_quaternion_xyzw=(0., 0., 0., 1.))
        if self.mode == "pre_ack" and step <= 51:
            sample = replace(sample, captured_monotonic_s=9.99)
        if self.mode == "epoch" and step == 300:
            sample = replace(sample, epoch="reset")
        if self.mode == "missing" and step == 300:
            sample = replace(sample, step=step+1)
        if self.mode == "generation" and step == 300:
            sample = replace(sample, generation=self.generation+1)
        if self.mode == "late_command" and step == 3090:
            self.now = 1210.
            sample = replace(sample, captured_monotonic_s=self.now)
        if self.stopped:
            rest_step = step-self.stop_step
            if self.mode == "active_brake":
                sample = replace(sample, spindle_effort_nm=.001, commanded_spindle_effort_nm=.001)
            if self.mode == "late_rest" and rest_step == 201:
                self.now += 90.
                sample = replace(sample, captured_monotonic_s=self.now)
            if self.mode == "joint_motion" and rest_step == 120:
                sample = replace(sample, joint_velocity_rad_s=(.021,))
            if self.mode == "lost_support" and rest_step == 120:
                sample = replace(sample, shoulder_contacts=(), contacts=sample.contacts[:2],
                    collision_count=2, solver_count=2)
        return (sample,)

    def stop(self):
        self.stopped, self.stop_step = True, self.cursor
        self.generation += 1
        return {"ok": True, "generation": self.generation,
            "accepted_monotonic_s": self.now+1. if self.mode == "bad_ack" else self.now,
            "physical_stop_verified": False}

    def reset_stop(self):
        return {"ok": False, "error": "fixture has no restart"}

    def close(self):
        self.closed = True
        return {"ok": True}


def domain(owner):
    return FasteningDomain(owner, owner.read, controller_id="synthetic/seating", clock=lambda: owner.now)


def test_ordinary_runtime_checks_approach_load_and_retained_rest_without_physical_task_credit():
    owner = SeatingOwner()
    runtime = RobotRuntime({"fastening": domain(owner)})
    try:
        result = runtime.execute("fastening.seat_fastener", {})
        assert result["ok"] and result["verified"] and result["seating_verified"], result
        assert result["pre_stop_threading"]["measured"]["turns"] >= 15.
        assert result["loaded_window_sim_s"] >= .5-1e-9
        assert result["rest"]["motor_off_window_sim_s"] >= 2.-1e-9
        assert result["rest"]["retained_window_sim_s"] == 1.
        assert result["synthetic"] and not result["physical_stop_verified"] and not result["preload_verified"]
        assert not runtime.execute("task_done", {"success": True, "summary": "synthetic seating"})["success"]
    finally:
        assert runtime.close()["ok"]
    assert owner.closed


@pytest.mark.parametrize("mode", ["no_shoulder", "wrong_height", "downward_support", "no_rotation",
    "pre_ack", "epoch", "missing", "generation", "late_command", "bad_ack", "active_brake",
    "late_rest", "joint_motion", "lost_support"])
def test_ack_or_motion_cannot_replace_same_solve_seating_and_retention(mode):
    owner = SeatingOwner(mode)
    result = domain(owner).execute("seat_fastener", {})
    assert owner.requested and owner.stopped
    assert not result["ok"] and not result["seating_verified"] and not result["physical_stop_verified"], result
    assert result["postcondition"]["status"] != "confirmed"


def test_seating_cannot_change_its_fixed_contract_through_tool_arguments():
    owner = SeatingOwner()
    result = domain(owner).execute("seat_fastener", {"turns": .5})
    assert not result["ok"] and not owner.requested


@pytest.mark.parametrize("operation", ["turn", "seat"])
def test_distinct_leases_preserve_one_turn_budget_and_share_stop_revocation(operation):
    bind, lim = configuration()
    guard = FasteningWriteGuard(bind, lim, clock=lambda: 10.)
    guard.reset_stop(observed(bind=bind))
    sample = observed(bind=bind, generation=1)
    permit = (guard.admit(sample, expected_generation=1) if operation == "turn"
        else guard.admit_seating(sample, expected_generation=1))
    assert permit.operation == operation
    assert permit.deadline_monotonic_s-10. == (40. if operation == "turn" else 1200.)
    assert permit.end_simulation_time_s == (4. if operation == "turn" else 45.)
    guard.stop()
    uploads = []
    with pytest.raises(FasteningFault, match="revoked"):
        guard.apply(replace(sample, generation=permit.generation), (0.,),
            (sample.geometry_min_m, sample.geometry_max_m), .01, lambda *args: uploads.append(args), permit=permit)
    assert not uploads


def test_unconfigured_controller_cannot_admit_seating():
    guard = FasteningWriteGuard(binding(), limits(), clock=lambda: 10.)
    guard.reset_stop(row())
    with pytest.raises(FasteningFault, match="not configured"):
        guard.admit_seating(row(generation=1), expected_generation=1)


def test_fixed_seating_task_refuses_reuse_after_a_prior_turn_without_writing():
    bind, lim = configuration()
    guard = FasteningWriteGuard(bind, lim, clock=lambda: 10.)
    guard.reset_stop(observed(bind=bind, turns=1.))
    with pytest.raises(FasteningFault, match="initial pre-engaged"):
        guard.admit_seating(observed(bind=bind, turns=1., generation=1), expected_generation=1)
    assert guard.generation == 1


@pytest.mark.parametrize("fault", ["missing", "wrong_candidate", "wrong_force", "wrong_side"])
def test_positive_seat_witnesses_must_match_complete_solved_ledger(fault):
    bind, lim = configuration()
    sample = observed(bind=bind, loaded=True)
    witness = sample.shoulder_contacts[0]
    if fault == "missing":
        witnesses = ()
    else:
        changes = {"wrong_candidate": {"candidate": 1}, "wrong_force": {"normal_force_n": 2.},
            "wrong_side": {"force_on_fastener_world_n": (0., 0., -3.)}}[fault]
        witnesses = (replace(witness, **changes),)
    with pytest.raises(FasteningFault, match="shoulder"):
        check_solve(replace(sample, shoulder_contacts=witnesses), bind, lim, 10.)


@pytest.mark.parametrize("fastener_is_b", [True, False])
def test_observer_preserves_force_side_and_original_candidate_indices(fastener_is_b):
    bind, lim = configuration()
    a, b, sign = ("seat", "nut", 1.) if fastener_is_b else ("nut", "seat", -1.)
    base = row(bind=bind, contacts=(SolvedPair("nut", "bolt", 1.), SolvedPair("nut", "socket", 1.),
        SolvedPair(a, b, 3.)), collision_count=3, solver_count=3)
    raw = [{"candidate": 2, "status": "solved", "shape_a": a, "shape_b": b,
        "point_world_m": (.012, 0., .023), "normal_a_to_b_world": (0., 0., sign),
        "normal_force_n": 3., "force_on_b_world_n": (0., 0., sign*3.)}]
    sample = seating_solve(base, raw, bind)
    check_solve(sample, bind, lim, 10.)
    assert sample.shoulder_contacts[0].force_on_fastener_world_n == (0., 0., 3.)
    assert sample.shoulder_contacts[0].candidate == 2
