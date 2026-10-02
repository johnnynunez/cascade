"""Synthetic contract/causality fixtures; none of these rows is native proof."""
from dataclasses import replace
import math
import threading

import pytest

from cascade.control.fastening import (FasteningBinding, FasteningController,
    FasteningFault, FasteningLimits, FasteningSolve, FasteningWriteGuard,
    SolveJournal, SolvedPair, check_solve)
from cascade.robotics.runtime import RobotRuntime
from cascade.skills.fastening_runtime import FasteningDomain


def limits():
    return FasteningLimits((-1.,), (1.,), (1.,), (-1., -1., 0.), (1., 1., 1.), 0.)


def binding(lim=None):
    lim = lim or limits()
    return FasteningBinding("a"*64, lim.sha256, "synthetic_fixture", "bolt", "nut", "socket",
        ("joint",), ("nut", "bolt", "socket", "shell", "ground"),
        (("nut", "bolt"), ("nut", "socket")), ("nut", "bolt"), (("nut", "socket"),), .01,
        fixture_origin_m=(0., 0., 0.))


def row(step=0, *, turns=0., generation=0, captured=10., bind=None, **changes):
    bind = bind or binding()
    angle = -turns*2*math.pi
    result = FasteningSolve(bind.sha256, "epoch", generation, step, step*bind.dt_s, captured,
        (0.,), (0.,), (0.,), (-.1, -.1, .03), (.3, .1, .2),
        (0., 0., .069-turns*.0025), (0., 0., math.sin(angle/2), math.cos(angle/2)),
        (0., 0., 0.), (0., 0., 0., 1.), (0., 0., .069), (0., 0., 0., 1.),
        0., 0., 0., 0., 0., 0., 0., (SolvedPair("nut", "bolt", 1.), SolvedPair("nut", "socket", 1.)),
        1, 1, 128, 2, 128, 2)
    return replace(result, **changes)


def armed():
    now = [10.]
    guard = FasteningWriteGuard(binding(), limits(), clock=lambda: now[0])
    guard.reset_stop(row())
    current = row(generation=1)
    permit = guard.admit(current, expected_generation=1)
    return now, guard, current, permit


def test_solve_and_binding_defensively_copy_sequences():
    joints = [0.]
    sample = row(joint_position_rad=joints)
    joints[0] = 100.
    assert sample.joint_position_rad == (0.,)
    assert row().thread_sample(binding()).fastener_position_m == (0., 0., .069)
    lim = limits()
    with pytest.raises(ValueError, match="limits binding"):
        FasteningWriteGuard(binding(lim), replace(lim, max_joint_speed_rad_s=8.))


@pytest.mark.parametrize("changes,reason", [
    ({"captured_monotonic_s": 9.7}, "stale"),
    ({"captured_monotonic_s": 11.}, "future"),
    ({"binding_sha256": "b"*64}, "identity"),
    ({"joint_position_rad": (.99,)}, "margin"),
    ({"joint_velocity_rad_s": (.81,)}, "joint speed"),
    ({"joint_effort_nm": (1.01,)}, "joint effort"),
    ({"geometry_min_m": (-.1, -.1, -.01)}, "workspace"),
    ({"geometry_max_m": (1.1, .1, .2)}, "workspace"),
    ({"spindle_effort_nm": .05001}, "spindle effort"),
    ({"spindle_speed_rad_s": 10.01}, "speed"),
    ({"thread_contacts": 0}, "witnesses"),
    ({"contacts": (SolvedPair("nut", "bolt", 1.), SolvedPair("shell", "ground", 1.))}, "forbidden"),
    ({"contacts": (SolvedPair("nut", "bolt", 1.), SolvedPair("unknown", "ground", 0.))}, "unknown"),
])
def test_measured_safety_and_identity_veto(changes, reason):
    with pytest.raises(FasteningFault, match=reason):
        check_solve(row(**changes), binding(), limits(), 10.)


@pytest.mark.parametrize("changes", [
    {"collision_count": 128}, {"solver_count": 128},
    {"solver_count": 1}, {"collision_capacity": 0},
    {"fastener_quaternion_xyzw": (0., 0., 0., 2.)},
    {"joint_velocity_rad_s": (float("nan"),)},
])
def test_unknown_overflow_incomplete_or_malformed_solve_has_no_representation(changes):
    with pytest.raises(ValueError):
        row(**changes)


def test_zero_force_does_not_create_contact_engagement():
    current = row(contacts=(SolvedPair("nut", "bolt", 0.), SolvedPair("nut", "socket", 0.)),
                  thread_contacts=0, tool_contacts=0)
    guard = FasteningWriteGuard(binding(), limits(), clock=lambda: 10.)
    guard.reset_stop(current)
    with pytest.raises(FasteningFault, match="pre-engaged"):
        guard.admit(replace(current, generation=1), expected_generation=1)


@pytest.mark.parametrize("failure", ["stop", "wall_expiry", "sim_expiry", "epoch", "duplicate", "speed", "effort"])
def test_no_write_after_revocation_expiry_or_unsafe_proposal(failure):
    now, guard, current, permit = armed()
    writes = []
    target, effort = (0.,), .01
    if failure == "stop":
        guard.stop()
    elif failure == "wall_expiry":
        now[0] = permit.deadline_monotonic_s
        current = replace(current, captured_monotonic_s=now[0])
    elif failure == "sim_expiry":
        current = replace(current, simulation_time_s=permit.end_simulation_time_s)
    elif failure == "epoch":
        current = replace(current, epoch="reset")
    elif failure == "duplicate":
        guard.apply(current, target, (current.geometry_min_m, current.geometry_max_m), effort,
                    lambda *args: None, permit=permit)
    elif failure == "speed":
        target = (.009,)
    else:
        effort = .0500001
    with pytest.raises(FasteningFault):
        guard.apply(current, target, (current.geometry_min_m, current.geometry_max_m), effort,
                    lambda *args: writes.append(args), permit=permit)
    assert not writes and guard.generation > permit.generation


def test_stop_linearizes_after_existing_upload_and_before_next_write():
    _, guard, current, permit = armed()
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    uploads = []
    def upload(*args):
        entered.set()
        assert release.wait(2), "test did not release bounded synthetic upload"
        uploads.append(args)
    worker = threading.Thread(target=lambda: guard.apply(current, (0.,),
        (current.geometry_min_m, current.geometry_max_m), .01, upload, permit=permit))
    stopper = threading.Thread(target=lambda: (guard.stop(), returned.set()))
    try:
        worker.start()
        assert entered.wait(2)
        stopper.start()
        assert not returned.is_set()
        release.set()
        worker.join(2); stopper.join(2)
        assert not worker.is_alive() and not stopper.is_alive() and returned.is_set()
        with pytest.raises(FasteningFault, match="revoked"):
            guard.apply(row(1, generation=permit.generation), (0.,),
                (current.geometry_min_m, current.geometry_max_m), .01, upload, permit=permit)
        assert len(uploads) == 1
    finally:
        release.set()
        worker.join(2)
        if stopper.ident is not None:
            stopper.join(2)


def test_passive_journal_preserves_capture_and_exposes_loss_reset_and_sticky_fault():
    journal = SolveJournal(binding(), capacity=3)
    for step in range(5):
        journal.publish(row(step))
    assert journal.read()[0].captured_monotonic_s == 10.
    assert [s.step for s in journal.read(1)] == [2, 3, 4]
    with pytest.raises(FasteningFault, match="capacity"):
        journal.read(0)
    with pytest.raises(FasteningFault, match="epoch"):
        journal.publish(row(5, epoch="another"))
    with pytest.raises(FasteningFault):
        journal.read(4)


def test_controller_latches_transient_fault_and_reader_cannot_recover_it_silently():
    journal = SolveJournal(binding())
    controller = FasteningController(binding(), limits(), journal, synthetic=True,
        close_owner=lambda: {"ok": True}, clock=lambda: 10.)
    controller.accept_solve(row())
    with pytest.raises(FasteningFault, match="joint speed"):
        controller.accept_solve(row(1, joint_velocity_rad_s=(.9,)))
    with pytest.raises(FasteningFault):
        controller.accept_solve(row(2))
    assert controller.close()["owner"]["ok"]


class ScriptedOwner:
    """Scripted rows exercise a verifier, never simulation or robot admission."""
    synthetic = True

    def __init__(self, mode="healthy"):
        self.binding, self.limits = binding(), limits()
        self.generation = 1
        self.now = 10.
        self.mode, self.turn_requested, self.stopped, self.closed = mode, False, False, False
        self.stop_at = None
        self.stop_step = None
        self.cursor = 0

    def request_turn(self, **args):
        assert args == {"expected_generation": 1, "turns": 1., "direction": "tighten"}
        from cascade.control.fastening import FasteningPermit
        self.turn_requested = True
        self.generation = 2
        return FasteningPermit(self.binding.sha256, "epoch", 2, 0, 0., 10., 50., 4.)

    def read(self, cursor, *, timeout_s):
        assert cursor == self.cursor
        self.now += .001
        self.cursor += 1
        step = self.cursor
        turns = 0. if self.mode == "wrist_only" else min(step*.005, .975)
        current = row(step, turns=turns, generation=self.generation, captured=self.now)
        if self.mode == "missing" and step == 30:
            current = replace(current, step=step+1)
        if self.mode == "epoch" and step == 30:
            current = replace(current, epoch="changed")
        if self.mode == "false_contacts" and step == 30:
            current = replace(current, contacts=(SolvedPair("nut", "bolt", 0.),
                              SolvedPair("nut", "socket", 0.)), thread_contacts=0, tool_contacts=0)
        if self.stopped and self.mode == "no_rest":
            current = replace(current, tool_angular_speed_rad_s=.03)
        if self.stopped and self.mode == "active_brake":
            current = replace(current, commanded_spindle_effort_nm=.001, spindle_effort_nm=.001)
        return (current,)

    def stop(self):
        self.generation += 1
        self.stopped = True
        self.stop_at = self.now
        self.stop_step = self.cursor
        return {"ok": True, "generation": self.generation, "accepted_monotonic_s": self.now,
                "physical_stop_verified": False}

    def reset_stop(self):
        return {"ok": False, "error": "fixture has no restart"}

    def close(self):
        self.closed = True
        return {"ok": True}


def test_normal_robot_runtime_runs_independent_thread_and_rest_checks_but_not_physical_task_credit():
    owner = ScriptedOwner()
    domain = FasteningDomain(owner, owner.read, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    runtime = RobotRuntime({"fastening": domain})
    try:
        result = runtime.execute("fastening.turn_screw", {"turns": 1., "direction": "tighten"})
        assert result["ok"] and result["postcondition"]["status"] == "confirmed", result
        assert result["postcondition"]["threading"]["measured"]["turns"] >= .95
        assert result["rest"]["window_sim_s"] >= .5-1e-9
        assert result["synthetic"] and not result["physical_stop_verified"]
        assert result["seating_verified"] is result["preload_verified"] is False
        task = runtime.execute("task_done", {"success": True, "summary": "synthetic contract"})
        assert not task["success"]
    finally:
        assert runtime.close()["ok"]
    assert owner.closed


@pytest.mark.parametrize("mode", ["wrist_only", "missing", "epoch", "no_rest", "active_brake"])
def test_actor_progress_or_stop_ack_cannot_replace_physical_postcondition(mode):
    owner = ScriptedOwner(mode)
    domain = FasteningDomain(owner, owner.read, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert owner.turn_requested and owner.stopped
    assert not result["ok"] and not result["verified"] and not result["physical_stop_verified"]
    assert result["postcondition"]["status"] != "confirmed"


@pytest.mark.parametrize("args", [{"turns": 2., "direction": "tighten"},
    {"turns": 1., "direction": "loosen"}, {"turns": True, "direction": "tighten"},
    {"turns": 1., "direction": "tighten", "seat": True}])
def test_unimplemented_goals_refuse_before_command(args):
    owner = ScriptedOwner()
    domain = FasteningDomain(owner, owner.read, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", args)
    assert not result["ok"] and not owner.turn_requested


def test_uncertain_admission_delivery_still_invokes_priority_stop():
    owner = ScriptedOwner()
    def partial(**_args):
        owner.turn_requested = True
        raise ConnectionError("ACK lost after possible command admission")
    owner.request_turn = partial
    domain = FasteningDomain(owner, owner.read, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert owner.turn_requested and owner.stopped
    assert result["stop"]["ok"] and result["postcondition"]["status"] == "unverified"


@pytest.mark.parametrize("bad", [{"generation": 2}, {"generation": 4},
    {"accepted_monotonic_s": -1.}, {"accepted_monotonic_s": 10.},
    {"accepted_monotonic_s": float("nan")}])
def test_old_wrong_generation_or_invalid_stop_ack_cannot_confirm_rest(bad):
    owner = ScriptedOwner()
    stop = owner.stop
    owner.stop = lambda: {**stop(), **bad}
    domain = FasteningDomain(owner, owner.read, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert result["execution_ok"]  # Threading itself was observed.
    assert not result["ok"] and result["postcondition"]["status"] == "unverified"
    assert "stop receipt" in result["error"]


def test_pitch_and_fixed_axis_are_explicit_binding_not_generic_thread_support():
    with pytest.raises(ValueError, match="Factory M20"):
        replace(binding(), thread_pitch_m=.001)
    with pytest.raises(FasteningFault, match="fixed world"):
        check_solve(row(fixture_quaternion_xyzw=(0., 0., math.sin(.1), math.cos(.1))),
                    binding(), limits(), 10.)


def test_empty_independent_channel_does_not_borrow_actuator_success():
    owner = ScriptedOwner()
    def empty(_cursor, *, timeout_s):
        owner.now += timeout_s
        return ()
    owner.completion_claim = {"ok": True, "seating_verified": True, "turns": 1.}
    domain = FasteningDomain(owner, empty, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert owner.stopped and not result["ok"]
    assert result["postcondition"]["status"] == "unverified"
    assert not result["seating_verified"]


@pytest.mark.parametrize("phase", ["turn", "rest", "late_fault"])
def test_pending_reader_cannot_earn_positive_credit_after_wall_deadline(phase):
    owner = ScriptedOwner()
    read = owner.read
    def late(cursor, *, timeout_s):
        sample, = read(cursor, timeout_s=timeout_s)
        if (phase in {"turn", "late_fault"} and sample.step == 191 or
                phase == "rest" and owner.stopped and sample.step == owner.stop_step+51):
            owner.now = 50.01 if not owner.stopped else owner.stop_at + 30.01
            sample = replace(sample, captured_monotonic_s=owner.now)
            if phase == "late_fault":
                sample = replace(sample, spindle_effort_nm=.1)
        return (sample,)
    domain = FasteningDomain(owner, late, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert not result["ok"] and not result["physical_stop_verified"]
    assert ("spindle effort" if phase == "late_fault" else "wall deadline") in result["error"]


def test_pre_admission_capture_cannot_supply_turn_baseline_or_progress():
    owner = ScriptedOwner()
    read = owner.read
    def prefence(cursor, *, timeout_s):
        sample, = read(cursor, timeout_s=timeout_s)
        if sample.step <= 190:
            sample = replace(sample, captured_monotonic_s=9.999)
        return (sample,)
    domain = FasteningDomain(owner, prefence, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert not result["ok"] and result["postcondition"]["status"] == "refuted"
    assert result["postcondition"]["measured"]["turns"] < .03


def test_pre_stop_capture_is_safety_observation_but_never_rest_credit():
    owner = ScriptedOwner()
    read = owner.read
    stop = owner.stop
    def acknowledged_after_capture():
        owner.now += .01  # Local ACK latency; original capture remains earlier.
        return stop()
    owner.stop = acknowledged_after_capture
    def prefence(cursor, *, timeout_s):
        sample, = read(cursor, timeout_s=timeout_s)
        if owner.stopped:
            if sample.step <= owner.stop_step+60:
                sample = replace(sample, captured_monotonic_s=owner.stop_at-.001)
            else:
                sample = replace(sample, tool_angular_speed_rad_s=.03)
        return (sample,)
    domain = FasteningDomain(owner, prefence, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert result["execution_ok"] and not result["ok"]
    assert "rest deadline" in result["error"]


def test_ack_replayed_before_request_cannot_admit_a_new_interval():
    owner = ScriptedOwner()
    owner.now = 10.01
    domain = FasteningDomain(owner, owner.read, controller_id="synthetic/endpoint", clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert owner.stopped and not result["ok"]
    assert "admission receipt" in result["error"]


def test_empty_producer_error_is_sticky():
    journal = SolveJournal(binding())
    journal.fail("")
    with pytest.raises(FasteningFault, match="producer failed"):
        journal.publish(row())
