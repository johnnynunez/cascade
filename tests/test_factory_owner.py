"""Synthetic owner/SDK-buffer tests. No native Factory or physical admission."""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from cascade.control.fastening import FasteningFault, FasteningUpload, check_solve
from cascade.sim.factory_model import (
    FactoryBoundModel, MAPPING_ARRAYS, NATIVE_OPTION_ARRAYS, NATIVE_OPTION_SCALARS,
    STATIC_MJW_ARRAYS, STATIC_NEWTON_ARRAYS,
    _array_digest, model_fingerprint,
)
from cascade.sim.factory_owner import FactoryNewtonBackend, FactorySolveOwner, NativeSolveClock, _Request
from test_factory_observation import Array
from test_fastening_runtime import armed, binding, limits, row


def test_upload_stamp_retains_old_generation_when_stop_happens_after_write():
    now, guard, state, permit = armed()
    written = []
    stamp = guard.apply(state, (0.,), (state.geometry_min_m, state.geometry_max_m),
                        .03, lambda *args: written.append(args), permit=permit)
    ack = guard.stop()
    assert stamp.generation == permit.generation and ack["generation"] == stamp.generation + 1
    assert stamp.before_step == state.step and stamp.effort_nm == .03
    assert stamp.started_monotonic_s <= stamp.completed_monotonic_s
    zero = guard.zero_hold(state.step+1, lambda effort: written.append(effort))
    assert written == [((0.,), .03), 0.] and zero.generation == ack["generation"]


def test_revoked_proposal_does_not_issue_second_stop_generation_after_existing_ack():
    _, guard, state, permit = armed()
    ack = guard.stop()
    writes = []
    with pytest.raises(FasteningFault, match="revoked"):
        guard.apply(state, (0.,), (state.geometry_min_m, state.geometry_max_m),
                    .03, lambda *args: writes.append(args), permit=permit)
    assert not writes and guard.generation == ack["generation"]


def test_zero_upload_failure_is_visible_even_when_already_stopped():
    _, guard, state, _ = armed()
    guard.stop()
    def fail(_):
        raise RuntimeError("device rejected zero upload")
    with pytest.raises(RuntimeError, match="zero upload"):
        guard.zero_hold(state.step, fail)


@pytest.mark.parametrize("changes", [{"generation": -1}, {"before_step": -1},
    {"started_monotonic_s": -1.}, {"completed_monotonic_s": 9.}, {"effort_nm": float("nan")}])
def test_upload_stamp_requires_original_valid_nonnegative_clock(changes):
    with pytest.raises(ValueError):
        replace(FasteningUpload(2, 1, .01, 10., 10.), **changes)


def native_solver():
    return NS(use_mujoco_cpu=False, _step=0, mjw_data=NS(time=Array([0.])),
              mjw_model=NS(opt=NS(timestep=Array([.01]))))


def test_native_clock_uses_actual_float32_recurrence_and_each_python_solve():
    solver = native_solver()
    clock = NativeSolveClock(solver, .01)
    for i in range(1, 101):
        clock.before()
        solver._step += 1
        solver.mjw_data.time.value += np.float32(.01)
        clock.after()
        clock.stable()
        assert clock.step == i and clock.native_time == solver.mjw_data.time.value[0]
    assert clock.native_time != 1.  # Accumulated SDK float32 time is retained, not refreshed.


@pytest.mark.parametrize("bad", ["skip", "no_solve", "time_replay", "wrong_dt", "empty_time", "wrong_dtype"])
def test_native_clock_rejects_missing_solve_reset_and_relabelled_time(bad):
    solver = native_solver()
    clock = NativeSolveClock(solver, .01)
    solver._step = 1
    solver.mjw_data.time.value[0] = .01
    if bad == "skip": solver._step = 2
    elif bad == "no_solve": solver._step = 0
    elif bad == "time_replay": solver.mjw_data.time.value[0] = 0
    elif bad == "wrong_dt": solver.mjw_model.opt.timestep.value[0] = .02
    elif bad == "empty_time": solver.mjw_data.time.value = np.array([], np.float32)
    else: solver.mjw_data.time.value = solver.mjw_data.time.value.astype(float)
    with pytest.raises(FasteningFault): clock.after()


def test_native_clock_rejects_physics_advance_during_observation():
    solver = native_solver()
    clock = NativeSolveClock(solver, .01)
    solver._step = 1; solver.mjw_data.time.value[0] = .01
    clock.after()
    solver._step += 1
    with pytest.raises(FasteningFault): clock.stable()


@pytest.mark.parametrize("value", [0., -.01, .1, float("nan"), float("inf")])
def test_native_clock_refuses_invalid_or_out_of_contract_timestep(value):
    with pytest.raises(FasteningFault): NativeSolveClock(native_solver(), value)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_exact_float32_effort_representation_is_allowed_but_next_value_is_not(sign):
    observed = float(np.float32(.05))*sign
    measured = row(spindle_effort_nm=observed, commanded_spindle_effort_nm=.05*sign)
    check_solve(measured, binding(), limits(), 10.)
    assert measured.spindle_effort_nm == observed  # Raw is never clipped.
    exceeded = float(np.nextafter(np.float32(.05), np.float32(np.inf)))*sign
    with pytest.raises(FasteningFault, match="spindle effort"):
        check_solve(replace(measured, spindle_effort_nm=exceeded), binding(), limits(), 10.)
    with pytest.raises(FasteningFault, match="spindle effort"):
        check_solve(replace(measured, commanded_spindle_effort_nm=observed), binding(), limits(), 10.)


class SyntheticBackend:
    synthetic = True

    def __init__(self, now):
        self.binding, self.limits, self.now = binding(), limits(), now
        self.step = 0
        self.effort, self.target = 0., (0.,)
        self.uploads, self.stamps = [], []
        self.plan_hook = self.prepare_hook = self.solve_hook = None
        self.fail_write = self.fail_zero = self.fail_observer = False

    def enter_owner(self):
        pass

    def upload(self, target, effort):
        self.target, self.effort = target, effort
        self.uploads.append((self.step, target, effort))
        if self.fail_write:
            raise RuntimeError("uncertain upload delivered then failed")

    def upload_zero(self, effort):
        assert effort == 0.
        if self.fail_zero:
            raise RuntimeError("zero upload failed")
        self.effort = 0.
        self.uploads.append((self.step, self.target, 0.))

    def plan(self, state):
        if self.plan_hook:
            self.plan_hook()
        return (0.,), (state.geometry_min_m, state.geometry_max_m), .03

    def advance(self, final_upload):
        if self.prepare_hook:
            self.prepare_hook()
        stamp = final_upload()
        self.stamps.append(stamp)
        if self.solve_hook:
            self.solve_hook()
        self.step += 1
        self.now[0] += .001
        if self.fail_observer:
            raise RuntimeError("observer missing native force channel")
        observation = row(self.step, generation=stamp.generation, captured=self.now[0],
                          spindle_effort_nm=self.effort, commanded_spindle_effort_nm=stamp.effort_nm)
        return observation, {"step": self.step, "generation": stamp.generation, "effort": stamp.effort_nm}


def owner_fixture(*, active=True):
    now = [10.]
    backend = SyntheticBackend(now)
    owner = FactorySolveOwner(backend, clock=lambda: now[0], max_wall_s=.02)
    owner.controller.accept_solve(row())
    owner._row = row()
    if active:
        owner.controller.guard.reset_stop(row())
        current = row(generation=1)
        owner.controller._previous = current
        owner._row = current
        owner.controller.guard.admit(current, expected_generation=1)
    return now, backend, owner


@pytest.mark.parametrize("phase", ["plan", "prepare"])
def test_stop_during_preparation_only_uploads_zero_in_the_same_stop_ack_generation(phase):
    _, backend, owner = owner_fixture()
    ack = []
    setattr(backend, phase+"_hook", lambda: ack.append(owner.controller.stop()))
    owner.cycle()
    assert backend.uploads == [(0, (0.,), 0.)]
    assert owner._row.generation == ack[0]["generation"] == owner.controller.generation
    assert owner.records()[0]["effort"] == 0.


def test_lease_expiring_during_collision_preparation_has_no_active_write_or_solve():
    now, backend, owner = owner_fixture()
    backend.prepare_hook = lambda: now.__setitem__(0, owner.controller.guard.current_permit.deadline_monotonic_s)
    owner._start = now[0]
    owner._run()
    assert "lease expired" in owner._error and backend.step == 0
    assert backend.uploads == [(0, (0.,), 0.)]  # Only finally's emergency zero.
    assert not owner.records()


def test_inflight_solve_retains_old_upload_generation_then_next_solve_has_stop_zero():
    _, backend, owner = owner_fixture()
    old = owner.controller.generation
    ack = []
    backend.solve_hook = lambda: ack.append(owner.controller.stop())
    owner.cycle()
    assert owner._row.generation == old
    backend.solve_hook = None
    owner.cycle()
    assert owner._row.generation == ack[0]["generation"] == old+1
    assert [u[2] for u in backend.uploads] == [.03, 0.]
    assert [r["generation"] for r in owner.records()] == [old, old+1]


@pytest.mark.parametrize("fault", ["write", "observer", "zero"])
def test_owner_finally_zero_hold_or_explicit_failure_never_claims_physical_stop(fault):
    now, backend, owner = owner_fixture()
    backend.fail_write = fault == "write"
    backend.fail_observer = fault in {"observer", "zero"}
    backend.fail_zero = fault == "zero"
    owner._start = now[0]
    owner._run()
    receipt = owner.close()
    assert not receipt["ok"] and not receipt["physical_stop_verified"]
    assert receipt["zero_spindle"]["uploaded"] is (fault != "zero")
    if fault != "zero": assert backend.effort == 0. and backend.target == (0.,)
    with pytest.raises(FasteningFault): owner.journal.read()


def test_stop_does_not_wait_for_blocked_native_solve_or_logger_consumer():
    _, backend, owner = owner_fixture()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    def solve():
        entered.set()
        assert release.wait(2), "synthetic solve not released"
        owner._exit.set()
    backend.solve_hook = solve
    owner.start()
    try:
        assert entered.wait(2)
        def stop():
            owner.controller.stop()
            stopped.set()
        stopper = threading.Thread(target=stop)
        stopper.start()
        assert stopped.wait(1), "stop queued behind the solver"
        stopper.join(1)
        assert backend.effort == .03  # ACK is latch, not fabricated physical zero.
        release.set()
        owner._thread.join(2)
        assert backend.effort == 0.
        assert owner.records()  # No consumer/logger was needed for stop delivery.
    finally:
        release.set()
        owner.close()


def test_expired_queued_motion_cannot_run_later_after_clock_recovers():
    now, backend, owner = owner_fixture(active=False)
    result = Future()
    owner._requests.put(_Request("turn", {"expected_generation": 0, "turns": 1., "direction": "tighten"},
                                0, now[0]-.01, result))
    owner.cycle()
    with pytest.raises(FasteningFault, match="expired"): result.result()
    assert backend.effort == 0. and owner.controller.guard.current_permit is None


def test_full_raw_journal_fault_stops_instead_of_dropping_unobserved_solves():
    now, backend, owner = owner_fixture(active=False)
    owner._records = __import__("queue").Queue(maxsize=1)
    owner._start = now[0]
    owner._run()
    assert owner._error.startswith("Full:")
    assert backend.effort == 0.
    assert not owner.close()["ok"]


def test_fingerprint_accepts_declared_unlimited_joint_ranges_but_not_nan_geometry():
    with pytest.raises(FasteningFault): _array_digest(np.array([np.inf]))
    assert _array_digest(np.array([-np.inf, np.inf]), allow_infinite=True)["shape"] == [2]
    with pytest.raises(FasteningFault): _array_digest(np.array([np.nan]), allow_infinite=True)


def model_fixture():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody><body>
       <joint name="arm"/><geom size=".1"/></body></worldbody>
       <actuator><motor joint="arm"/></actuator></mujoco>''')
    arrays = {n: Array([1.]) for n in STATIC_NEWTON_ARRAYS}
    arrays.update(shape_label=["arm"], body_label=["arm"], joint_label=["arm"])
    native = NS(**{n: Array([1.]) for n in STATIC_MJW_ARRAYS})
    native.opt = NS(**{n: Array([1.]) for n in NATIVE_OPTION_ARRAYS},
                    **{n: 1 for n in NATIVE_OPTION_SCALARS})
    native.opt.run_collision_detection = False
    solver = NS(mj_model=model, mjw_model=native, use_mujoco_cpu=False, _use_mujoco_contacts=False,
                **{n: Array([[0]], np.int32) for n in MAPPING_ARRAYS})
    return NS(mujoco=mj, model=NS(**arrays), solver=solver)


def binding_interface_fixture():
    from cascade.sim.newton_screw_seating import SeatingScene
    # Construction is bypassed only in this synthetic contract test. There is
    # deliberately no invented solver.use_mujoco_contacts compatibility alias.
    scene = object.__new__(SeatingScene)
    scene.step_id, scene.time_s = 0, 0.
    scene.solver = NS(use_mujoco_cpu=False, _use_mujoco_contacts=False,
                      mjw_model=NS(opt=NS(run_collision_detection=False)))
    return scene


def test_binding_accepts_only_the_pinned_sdk_collision_interface(monkeypatch):
    scene = binding_interface_fixture()
    class MappingReached(Exception):
        pass
    def mapping(*_):
        raise MappingReached
    monkeypatch.setattr("cascade.sim.factory_model.joint_mapping", mapping)
    with pytest.raises(MappingReached):
        FactoryBoundModel(scene)
    assert not hasattr(scene.solver, "use_mujoco_contacts")


@pytest.mark.parametrize("field", ["use_mujoco_cpu", "_use_mujoco_contacts", "run_collision_detection"])
@pytest.mark.parametrize("value", [True, None, 0, "missing"])
def test_binding_rejects_unavailable_or_wrong_effective_collision_mode(field, value):
    scene = binding_interface_fixture()
    owner = scene.solver.mjw_model.opt if field == "run_collision_detection" else scene.solver
    if value == "missing":
        delattr(owner, field)
    else:
        setattr(owner, field, value)
    with pytest.raises(FasteningFault, match="collision"):
        FactoryBoundModel(scene)


@pytest.mark.parametrize("field", ["use_mujoco_cpu", "_use_mujoco_contacts", "run_collision_detection"])
def test_bound_identity_rechecks_the_actual_collision_route(field):
    scene = model_fixture()
    bound = object.__new__(FactoryBoundModel)
    bound.scene, bound._fingerprint = scene, model_fingerprint(scene)
    owner = scene.solver.mjw_model.opt if field == "run_collision_detection" else scene.solver
    setattr(owner, field, True)
    with pytest.raises(FasteningFault, match="collision"):
        bound.check_immutable()


@pytest.mark.parametrize("mutation", ["compiled_gain", "native_gain", "native_mass", "shape_transform", "dof_map", "labels", "gravity", "integrator"])
def test_bound_identity_detects_compiled_native_geometry_and_mapping_mutations(mutation):
    scene = model_fixture()
    bound = object.__new__(FactoryBoundModel)
    bound.scene, bound._fingerprint = scene, model_fingerprint(scene)
    bound.check_immutable()
    if mutation == "compiled_gain": scene.solver.mj_model.actuator_gainprm[0, 0] += 1
    elif mutation == "native_gain": scene.solver.mjw_model.actuator_gainprm.value[0] += 1
    elif mutation == "native_mass": scene.solver.mjw_model.body_mass.value[0] += 1
    elif mutation == "shape_transform": scene.model.shape_transform.value[0] += 1
    elif mutation == "dof_map": scene.solver.mjc_dof_to_newton_dof.value[0, 0] += 1
    elif mutation == "gravity": scene.solver.mjw_model.opt.gravity.value[0] += 1
    elif mutation == "integrator": scene.solver.mjw_model.opt.integrator += 1
    else: scene.model.shape_label[0] = "other"
    with pytest.raises(FasteningFault, match="changed after identity"):
        bound.check_immutable()


def backend_fixture(monkeypatch):
    # All SDK arrays here are deliberately synthetic; only call ordering and
    # ownership are being tested. No simulator import or physical credit.
    events = []
    solver = native_solver()
    solver.mj_model = NS(nq=1)
    solver.mjw_data.qpos = Array([[0.]])
    solver.mjw_model.opt.timestep.value[0] = np.float32(binding().dt_s)
    ctrl = Array([.1, .0])
    def assign(value):
        events.append(("write", value.copy()))
        ctrl.value = value.copy()
    ctrl.assign = assign
    state = NS(clear_forces=lambda: events.append("clear"))
    next_state = NS(clear_forces=lambda: events.append("clear"))
    def step(old, new, control, contacts, dt):
        events.append("solve")
        solver._step += 1
        solver.mjw_data.time.value += np.float32(dt)
    solver.step = step
    pipeline = NS(collide=lambda *_: events.append("collide"))
    scene = NS(solver=solver, state=state, next=next_state, control=NS(mujoco=NS(ctrl=ctrl)),
        pipeline=pipeline, contacts=object(), step_id=0, time_s=0.,
        requested_speed_rad_s=3., _arm_target=lambda _: np.array([]))
    joints = (NS(control_index=0, reference_rad=.1), NS(control_index=1, reference_rad=0.))
    bound = NS(scene=scene, binding=binding(), limits=limits(),
        initial_control=np.array([.1, 0.], np.float32), joints=joints,
        check_immutable=lambda: events.append("identity"), clock=lambda: 10.,
        geometry=NS(evaluate=lambda *args, **kwargs: ((-.1, -.1, .1), (.1, .1, .3))))
    def read(stamp, capture, coverage):
        events.append("observe")
        assert scene.step_id == solver._step == stamp.before_step + 1
        assert capture == 10. and coverage == {"actual": "counter receipt"}
        observation = row(scene.step_id, generation=stamp.generation, captured=capture)
        return observation, {"generation": stamp.generation}
    bound.observer = NS(read=read)
    def coverage(*args):
        events.append("coverage")
        return {"actual": "counter receipt"}
    monkeypatch.setattr("cascade.sim.factory_owner.collision_coverage", coverage)
    return FactoryNewtonBackend(bound), events


def test_backend_checks_each_real_counter_and_does_not_call_legacy_frame_step(monkeypatch):
    backend, events = backend_fixture(monkeypatch)
    stamp = FasteningUpload(7, 0, .03, 9.9, 9.91)
    def final_upload():
        events.append("fenced-upload")
        return stamp
    observation, raw = backend.advance(final_upload)
    assert events == ["identity", "clear", "collide", "coverage", "fenced-upload", "solve", "observe"]
    assert observation.generation == 7 and raw["collision_interval"] == {
        "before_step": 0, "generation": 7, "after_step": 1}
    assert raw["native_clock"]["step"] == 1 and raw["native_clock"]["cuda_graph"] is False
    assert raw["native_clock"]["time_s"] == float(np.float32(backend.binding.dt_s))
    # Replaying the upload or skipping ahead cannot create another observed row.
    with pytest.raises(FasteningFault, match="upload stamp"):
        backend.advance(final_upload)
    assert events.count("solve") == 1


def test_native_zero_writer_preserves_all_arm_targets_and_exact_zero(monkeypatch):
    backend, events = backend_fixture(monkeypatch)
    backend.upload((.2,), .03)
    previous = backend.scene.control.mujoco.ctrl.value.copy()
    backend.upload_zero(0.)
    assert backend.scene.control.mujoco.ctrl.value[0] == previous[0]
    assert backend.scene.control.mujoco.ctrl.value[1] == 0.
    assert backend._target == (.2,)
    with pytest.raises(FasteningFault, match="exact zero"):
        backend.upload_zero(.001)
    assert len(events) == 2


def test_admissions_happen_between_solves_and_do_not_relabel_initial_row():
    _, backend, owner = owner_fixture(active=False)
    reset = Future()
    owner._requests.put(_Request("reset", {}, 0, 11., reset))
    owner.cycle()
    assert reset.result()["generation"] == 1 and owner._row.generation == 1
    turn = Future()
    owner._requests.put(_Request("turn", {"expected_generation": 1, "turns": 1., "direction": "tighten"},
                                1, 11., turn))
    owner.cycle()
    permit = turn.result()
    assert permit.admission_step == 1 and owner._row.step == 2
    assert permit.generation == owner._row.generation == 2
    assert [s.generation for s in backend.stamps] == [1, 2]
    assert [u[2] for u in backend.uploads] == [0., .03]
