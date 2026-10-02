"""Software-only doubles: these tests make NO physics/Kit acceptance claim."""
from __future__ import annotations

import copy
import importlib

import numpy as np
import pytest
from mobile_support_fixture import support_contract

from cascade.control.microduck_policy import HOME_Q, POLICY_JOINTS
from cascade.sim.mobile_bridge import MobileBridgeController


def test_kit_dt_distinguishes_nominal_manager_from_native_clock():
    # Actual product1 receipt: manager .005; independent Kit probes: two
    # bootstrap solves at native time .009999999776482582.
    from types import SimpleNamespace
    from cascade.sim.microduck_newton import KitNewtonBackend
    backend = KitNewtonBackend(None, None, None)
    backend.SM = SimpleNamespace(get_physics_dt=lambda: .005)
    assert backend.dt == 0.004999999888241291
    assert 2 * backend.dt == 0.009999999776482582


@pytest.mark.parametrize('value', [.01, 0., float('nan'), float('inf'), True, '.005'])
def test_kit_dt_rejects_changed_or_non_numeric_nominal_step(value):
    from types import SimpleNamespace
    from cascade.sim.microduck_newton import KitNewtonBackend
    backend = KitNewtonBackend(None, None, None)
    backend.SM = SimpleNamespace(get_physics_dt=lambda: value)
    with pytest.raises(ValueError, match='nominal'):
        _ = backend.dt


class SoftwareBackend:
    """Synthetic clocks/state, deliberately no dynamics or robot motion."""
    dt = float(np.float32(.005))

    def __init__(self):
        self.events = []
        self.step_count = 2
        self.sim_time = 2 * self.dt
        self.closed = 0
        self.contained = []

    def read(self):
        return dict(step=self.step_count, sim_time=self.sim_time,
                    position=[0., 0., .125], orientation_wxyz=[1., 0., 0., 0.],
                    linear_velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.],
                    gravity_body=[0., 0., -1.], q=HOME_Q.copy(), dq=np.zeros(14, np.float32),
                    joint_names=POLICY_JOINTS, contacts=())

    @property
    def physics_clock(self):
        return self.step_count, self.sim_time

    def step(self):
        self.events.append(('solve', self.step_count))
        self.step_count += 1
        self.sim_time += self.dt

    def contain(self, reason):
        self.contained.append(reason)

    def close(self):
        self.closed += 1


class SoftwarePolicy:
    def __init__(self, backend):
        self.backend = backend
        self.previous_action = np.zeros(14, np.float32)
        self.inputs = []

    def infer(self, obs):
        self.backend.events.append(('infer', self.backend.step_count))
        self.inputs.append(obs.copy())
        return self.previous_action.copy()

    def preview(self, obs):
        # Software-only session boundary. infer hooks below simulate a slow
        # ONNX call; they do not publish/commit policy history themselves.
        return self.infer(obs)

    def commit(self, action):
        self.previous_action = action.copy()

    def targets(self, action):
        return HOME_Q + action


class SoftwareActuator:
    def __init__(self, backend):
        self.backend = backend
        self.targets = []

    def set_targets(self, q):
        self.targets.append(q.copy())

    def before_step(self, dt):
        assert dt == self.backend.dt
        self.backend.events.append(('before', self.backend.step_count))

    def telemetry(self):
        return {'software_fixture': True}


def controller(clock=lambda: 1.):
    return MobileBridgeController(
        robot_id='microduck', source='software-test-not-physics', engine='newton', device='cuda:0',
        asset_sha256='a'*64, policy_sha256='b'*64, model_identity_sha256='e'*64, support_contract=support_contract(), max_linear_speed=.3,
        max_angular_speed=.5, max_duration_s=10., lease_s=.3,
        max_state_age_s=1., max_action_wall_s=10., clock=clock)


def make_stepper(**overrides):
    module = importlib.import_module('cascade.sim.microduck_stepper')
    backend = overrides.pop('backend', SoftwareBackend())
    ctrl = overrides.pop('controller', controller())
    policy = overrides.pop('policy', SoftwarePolicy(backend))
    actuator = overrides.pop('actuator', SoftwareActuator(backend))
    settings = dict(max_steps=1000, max_wall_s=100.,
                    min_height_m=.05, max_height_m=.3, max_tilt_rad=.7)
    settings.update(overrides)
    stepper = module.MicroduckStepper(backend, ctrl, policy, actuator, **settings)
    return stepper, backend, ctrl, policy, actuator


def test_completed_step_drives_policy_once_per_four_and_reads_are_passive():
    stepper, backend, ctrl, policy, actuator = make_stepper()
    stepper.start()
    assert ctrl.state()['state'] is None  # HOME/FK is initialization, not a completed solve
    initial_step = backend.step_count
    for _ in range(8):
        stepper.tick()
        for _ in range(3):
            ctrl.state()
    assert len(policy.inputs) == 2
    assert len(actuator.targets) == 2
    assert [e for e in backend.events if e[0] == 'infer'] == [('infer', 2), ('infer', 6)]
    assert [e for e in backend.events if e[0] != 'infer'] == [
        e for n in range(2, 10) for e in [('before', n), ('solve', n)]]
    assert ctrl.state()['state']['step'] == initial_step + 8
    assert ctrl.state()['state']['controller_status'] == 'ready'
    assert ctrl.state()['state']['contacts'] == []  # never invent a floor
    assert stepper.policy_evaluations == 2
    record = stepper.last_policy
    assert record['observation_step'] == 6
    assert record['observation_sim_time_s'] == 6 * backend.dt
    assert record['source'] == ctrl.hello()['source'] and record['epoch'] == ctrl.hello()['epoch']
    np.testing.assert_array_equal(record['observation'], policy.inputs[-1][0])
    np.testing.assert_array_equal(record['commands'], policy.inputs[-1][0, 48:])
    np.testing.assert_array_equal(record['raw_action'], policy.previous_action)
    np.testing.assert_array_equal(record['targets'], actuator.targets[-1])
    assert len(record['observation']) == 61 and record['status'] == 'evaluated'


@pytest.mark.parametrize('failure', ['inference', 'actuator', 'nonfinite', 'dtype', 'names', 'frozen', 'dt', 'extra_step', 'fall'])
def test_fault_contains_without_another_solve_or_recovery(failure):
    stepper, backend, ctrl, policy, actuator = make_stepper()
    stepper.start()
    if failure == 'inference':
        policy.infer = lambda obs: (_ for _ in ()).throw(RuntimeError('inference failed'))
    elif failure == 'actuator':
        actuator.before_step = lambda dt: (_ for _ in ()).throw(ValueError('actuator failed'))
    elif failure in ('nonfinite', 'dtype', 'names', 'fall'):
        read = backend.read
        def bad_read():
            sample = read()
            if failure == 'nonfinite':
                sample['dq'][0] = np.nan
            elif failure == 'dtype':
                sample['q'] = sample['q'].astype(np.float64)
            elif failure == 'names':
                sample['joint_names'] = tuple(reversed(POLICY_JOINTS))
            else:
                sample['position'][2] = .01
            return sample
        backend.read = bad_read
    elif failure == 'frozen':
        backend.step = lambda: backend.events.append(('solve', backend.step_count))
    elif failure == 'dt':
        backend.dt *= 2
    else:
        backend.step()
    with pytest.raises((ValueError, RuntimeError)):
        stepper.tick()
    solves = [e for e in backend.events if e[0] == 'solve']
    assert len(solves) == (1 if failure in ('frozen', 'extra_step') else 0)
    assert backend.contained
    assert ctrl.state()['controller'] == 'fault'
    with pytest.raises(RuntimeError):
        stepper.tick()
    assert [e for e in backend.events if e[0] == 'solve'] == solves


def command(ctrl):
    h = ctrl.hello()
    return ctrl.command_velocity(dict(robot_id=h['robot_id'], source=h['source'],
        epoch=h['epoch'], generation=h['generation'], owner='test-owner',
        command_id='one', vx=.2, vy=0., wz=0., duration_s=2.))


@pytest.mark.parametrize('cancel', ['stop', 'watchdog', 'disconnect'])
def test_crossed_inference_is_discarded_before_commit_and_balance_recomputed(cancel):
    now = [1.]
    s, b, c, p, a = make_stepper(controller=controller(lambda: now[0]))
    s.start(); s.tick(); command(c)
    for _ in range(3):
        s.tick()
    before_step = b.step_count
    inputs = []
    def crossed(obs):
        inputs.append(obs.copy())
        if len(inputs) == 1:
            if cancel == 'stop':
                c.stop(latch=True)
            elif cancel == 'disconnect':
                c.owner_disconnected('test-owner')
            else:
                now[0] += .31
                c.watchdog()
        return np.full(14, obs[0,48], np.float32)
    p.infer = crossed
    result = s.tick()
    np.testing.assert_array_equal(a.targets[-1], HOME_Q)
    assert len(inputs) == 2 and inputs[0][0,48] == np.float32(.2)
    assert not inputs[1][0,48:].any()
    np.testing.assert_array_equal(inputs[1][0,34:48], np.zeros(14,np.float32))
    np.testing.assert_array_equal(p.previous_action, np.zeros(14,np.float32))
    assert b.step_count == before_step+1 and not b.contained
    assert [r['status'] for r in s.policy_records] == ['discarded', 'evaluated']
    assert s.policy_records[0]['committed'] is False
    assert s.policy_records[1]['committed'] is True
    assert result['policy_target_generation'] == c.hello()['generation']
    assert s.policy_attempts == 3 and s.policy_evaluations == 3 and s.policy_commits == 2
    s.close()


def test_repeated_invalidation_contains_without_targets_or_solver_or_history_write():
    s, b, c, p, a = make_stepper()
    s.start(); s.tick(); command(c)
    for _ in range(3):
        s.tick()
    before = (b.step_count, len(a.targets))
    old = p.previous_action.copy()
    def repeated(obs):
        c.stop(latch=True)
        return np.full(14, .1, np.float32)
    p.infer = repeated
    with pytest.raises(RuntimeError, match='invalidation'):
        s.tick()
    assert before == (b.step_count, len(a.targets)) and b.contained
    assert len(s.policy_records) == 2 and all(r['status']=='discarded' for r in s.policy_records)
    np.testing.assert_array_equal(p.previous_action, old)
    s.close()


def test_stop_does_not_wait_for_onnx_or_resume_discarded_travel():
    import threading
    s, b, c, p, a = make_stepper()
    s.start(); s.tick(); command(c)
    for _ in range(3):
        s.tick()
    entered, release, acknowledged = threading.Event(), threading.Event(), threading.Event()
    errors=[]
    def blocking(obs):
        if obs[0,48] != 0:
            entered.set()
            assert release.wait(2)
        return np.full(14, obs[0,48], np.float32)
    p.infer=blocking
    def tick():
        try: s.tick()
        except BaseException as e: errors.append(e)
    t=threading.Thread(target=tick)
    stopper=threading.Thread(target=lambda: (c.stop(latch=True),acknowledged.set()))
    try:
        t.start(); assert entered.wait(1); stopper.start()
        assert acknowledged.wait(.2), 'permission lock was held across inference'
    finally:
        release.set(); t.join(2)
        if stopper.ident is not None: stopper.join(2)
        s.close()
    assert not t.is_alive() and not stopper.is_alive() and not errors
    np.testing.assert_array_equal(a.targets[-1], HOME_Q)


@pytest.mark.parametrize('phase', ['preview', 'targets', 'permission_snapshot', 'upload', 'bam'])
def test_wall_budget_crossing_blocks_later_commit_or_solve(phase):
    now=[1.]
    s,b,c,p,a=make_stepper(controller=controller(lambda:now[0]), clock=lambda:now[0], max_wall_s=.15)
    s.start(); s.tick(); command(c)
    for _ in range(3): s.tick()
    before=(b.step_count,len(a.targets),s.policy_commits)
    history=p.previous_action.copy()
    p.infer=lambda obs:np.full(14,obs[0,48],np.float32)
    def expire_after(fn):
        def call(*args):
            result=fn(*args)
            now[0]=1.2  # episode .15 elapsed; command lease1.3 remains valid
            return result
        return call
    if phase=='preview': p.infer=expire_after(p.infer)
    elif phase=='targets': p.targets=expire_after(p.targets)
    elif phase=='permission_snapshot':
        original=s._control_snapshot
        calls=0
        def snapshot(t):
            nonlocal calls
            result=original(t); calls+=1
            if calls==2: now[0]=1.2
            return result
        s._control_snapshot=snapshot
    elif phase=='upload': a.set_targets=expire_after(a.set_targets)
    else: a.before_step=expire_after(a.before_step)
    before_bam=sum(e[0]=='before' for e in b.events)
    try:
        with pytest.raises(RuntimeError, match='wall duration'):
            s.tick()
        assert b.step_count==before[0] and b.contained
        assert 'first_step_after_commit' not in s.last_policy
        if phase in ('preview','targets','permission_snapshot'):
            assert len(a.targets)==before[1] and s.policy_commits==before[2]
            np.testing.assert_array_equal(p.previous_action,history)
            assert not s.last_policy['committed']
        if phase!='bam': assert sum(e[0]=='before' for e in b.events)==before_bam
    finally:
        s.close()


def test_wall_budget_crossing_during_held_target_read_blocks_another_prepare():
    now=[1.]
    s,b,c,p,a=make_stepper(controller=controller(lambda:now[0]), clock=lambda:now[0], max_wall_s=.15)
    s.start(); s.tick()
    original=b.read
    def late_read():
        result=original(); now[0]=1.2; return result
    b.read=late_read
    before=(b.step_count,len(a.targets),sum(e[0]=='before' for e in b.events))
    try:
        with pytest.raises(RuntimeError, match='wall duration'):
            s.tick()
        assert (b.step_count,len(a.targets),sum(e[0]=='before' for e in b.events))==before
        assert b.contained
    finally:
        s.close()


def test_physics_tick_crossing_inference_never_commits_stale_targets():
    s, b, c, p, a = make_stepper()
    s.start()
    def foreign_step(obs):
        b.step()  # forbidden second physics owner, not the stepper's own solve
        return np.full(14, .1, np.float32)
    p.infer = foreign_step
    with pytest.raises(RuntimeError):
        s.tick()
    assert not a.targets, 'stale input was committed after its physical state changed'
    assert not any(x[0] == 'before' for x in b.events)
    np.testing.assert_array_equal(p.previous_action, np.zeros(14,np.float32))
    s.close()


def test_stop_keeps_committed_hold_only_until_next_50hz_slot_without_delay_reset():
    s, b, c, p, a = make_stepper()
    def forbidden_reset():
        raise AssertionError('STOP must not erase the physical actuator delay/history')
    a.reset = forbidden_reset
    p.infer = lambda obs: np.full(14, obs[0,48], np.float32)
    s.start(); s.tick(); command(c)
    for _ in range(4):
        s.tick()
    moving_generation = c.hello()['generation']
    np.testing.assert_array_equal(a.targets[-1], HOME_Q + np.float32(.2))
    ack = c.stop(latch=True)
    assert ack['physical_stop_verified'] is False
    before = s.policy_attempts
    for _ in range(3):
        row = s.tick()
        assert row['policy_target_generation'] == moving_generation
        assert c.state()['state']['generation'] > moving_generation
        assert row['policy_target_held'] is True
    assert s.policy_attempts == before
    row = s.tick()
    assert row['policy_target_held'] is False
    assert row['policy_target_generation'] == c.hello()['generation']
    np.testing.assert_array_equal(a.targets[-1], HOME_Q)
    s.close()


@pytest.mark.parametrize('cancel', ['stop', 'watchdog', 'disconnect'])
def test_zero_twist_after_invalidation_never_revives_travel(cancel):
    now = [1.]
    stepper, backend, ctrl, policy, actuator = make_stepper(controller=controller(lambda: now[0]))
    stepper.start()
    stepper.tick()
    command(ctrl)
    for _ in range(4):
        stepper.tick()
    assert policy.inputs[-1][0, 48] == np.float32(.2)
    if cancel == 'stop':
        ctrl.stop(latch=True)
    elif cancel == 'disconnect':
        ctrl.owner_disconnected('test-owner')
    else:
        now[0] += .31
        ctrl.watchdog()  # expires while physics is paused
    for _ in range(8):
        stepper.tick()
    assert all(np.count_nonzero(obs[0, 48:]) == 0 for obs in policy.inputs[2:])
    assert len([e for e in backend.events if e[0] == 'before']) == stepper.steps
    assert ctrl.state()['command_id'] is None
    if cancel != 'watchdog':
        assert ctrl.state()['state']['controller_status'] == 'ready'
        ctrl.reset_stop()
        for _ in range(4):
            stepper.tick()
        assert not np.any(policy.inputs[-1][0, 48:])


def test_float32_dt_clocks_not_nominal_decimal_and_bounded_teardown():
    now = [0.]
    stepper, backend, ctrl, policy, _ = make_stepper(clock=lambda: now[0])
    stepper.start()
    for _ in range(400):
        stepper.tick()
    assert len(policy.inputs) == 100
    assert stepper.last['sim_time'] == 402 * float(np.float32(.005))
    now[0] = 100.
    with pytest.raises(RuntimeError, match='wall'):
        stepper.tick()
    stepper.close()
    stepper.close()
    assert backend.closed == 1
    assert ctrl.state()['latched'] is True


def render_times(t):
    return {'rpFabricTime': {'fabricFrameTimeNumerator': round(t * 1e9),
                             'fabricFrameTimeDenominator': 1_000_000_000},
            'IsaacReadSimulationTime': {'simulationTime': t, 'execOut': 1}}


def test_camera_cache_binds_render_clocks_and_never_encodes_on_rpc(monkeypatch):
    from cascade.sim.microduck_stepper import FrameCache
    import cv2
    now = [2.]
    ctrl = controller()
    cache = FrameCache(ctrl.hello(), max_jpeg_bytes=100000, max_pixels=640*480, clock=lambda: now[0])
    # Deliberately synthetic colored fixture: no rendered/physical evidence.
    rgb = np.zeros((24, 32, 3), np.uint8)
    rgb[:, :, 0] = 255
    cache.publish(rgb, step=2, sim_time_s=.01, captured_at=1.9, render_times=render_times(.01))
    def forbidden(*args, **kwargs):
        raise AssertionError('RPC must never encode')
    monkeypatch.setattr(cv2, 'imencode', forbidden)
    first = cache({'camera': 'overview'})
    now[0] += .1
    second = cache({'camera': 'overview'})
    assert set(second) == {'frame'}
    assert set(second['frame']) == {'robot_id', 'source', 'epoch', 'engine', 'device',
        'asset_sha256', 'policy_sha256', 'model_identity_sha256', 'camera', 'step', 'sim_time_s', 'width', 'height',
        'rgb_jpeg_b64', 'producer_age_s'}
    assert second['frame']['producer_age_s'] > first['frame']['producer_age_s']
    assert second['frame']['step'] == 2
    assert second['frame']['rgb_jpeg_b64'] == first['frame']['rgb_jpeg_b64']
    second['frame']['step'] = 999
    assert cache({'camera': 'overview'})['frame']['step'] == 2
    with pytest.raises(ValueError, match='camera'):
        cache({'camera': 'unknown'})
    cache.close()
    with pytest.raises(RuntimeError):
        cache({'camera': 'overview'})


@pytest.mark.parametrize('bad', ['stale', 'missing', 'nan', 'denominator', 'duplicate', 'shape', 'oversize'])
def test_cache_rejects_bad_capture_without_restamping_old_pixels(bad):
    from cascade.sim.microduck_stepper import FrameCache
    cache = FrameCache(controller().hello(), max_jpeg_bytes=10000, max_pixels=1000)
    rgb = np.zeros((10, 10, 3), np.uint8)
    cache.publish(rgb, step=2, sim_time_s=.01, captured_at=0., render_times=render_times(.01))
    times = render_times(.015)
    if bad == 'stale':
        times = render_times(.01)
    elif bad == 'missing':
        times = {}
    elif bad == 'nan':
        times['IsaacReadSimulationTime']['simulationTime'] = float('nan')
    elif bad == 'denominator':
        times['rpFabricTime']['fabricFrameTimeDenominator'] = 0
    elif bad == 'shape':
        rgb = rgb.astype(float)
    elif bad == 'oversize':
        rgb = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises((ValueError, RuntimeError)):
        cache.publish(rgb, step=2 if bad == 'duplicate' else 3, sim_time_s=.015,
                      captured_at=0., render_times=times)
    assert cache({'camera': 'overview'})['frame']['step'] == 2


class Array:
    """Host-only Warp-shaped test buffer. NOT a simulated physical model."""
    def __init__(self, value, dtype=np.float32):
        self.value = np.array(value, dtype=dtype)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value[:] = value

    def zero_(self):
        self.value[:] = 0


def native_fixture():
    from types import SimpleNamespace as NS
    model = NS(body_label=['/robot/trunk', '/robot/foot'], shape_label=['/robot/foot/shape', '/scene/plane'],
               shape_body=Array([1, -1], np.int32), body_com=Array([[.1, 0, 0], [0, 0, 0]]))
    state = NS(joint_q=Array(np.r_[np.zeros(7), HOME_Q]), joint_qd=Array(np.zeros(20)),
               body_q=Array([[0, 0, .125, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]),
               body_qd=Array([[0, 1, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0]]))
    contacts = NS(rigid_contact_count=Array([1], np.int32), rigid_contact_shape0=Array([0, 0, 0], np.int32),
                  rigid_contact_shape1=Array([1, 1, 1], np.int32))
    solver = NS(mjw_data=NS(nefc=Array([1], np.int32), nacon=Array([1], np.int32),
                           contact=NS(efc_address=Array([[0, -1], [-1, -1], [-1, -1]], np.int32),
                                      worldid=Array([0, 0, 0], np.int32)),
                           efc=NS(type=Array([[5, 0, 0]], np.int32))))
    return NS(model=model, state_0=state, contacts=contacts, solver=solver, simulation_step_count=2, sim_time=.01)


def test_contact_candidates_without_solver_rows_are_not_physical_contacts():
    # Real product2 exposed 33 candidates with only the 14 DOF-friction rows
    # before touchdown, and listed elevated body parts against the floor.
    # This fixture mirrors the source API's -1 address for inactive candidates.
    from cascade.sim.microduck_newton import read_native_state
    ns = native_fixture()
    ns.solver.mjw_data.contact.efc_address.value[0, :] = -1
    sample = read_native_state(ns, q_indices=np.arange(7, 21), dof_indices=np.arange(6, 20),
                              root_index=0, max_contacts=3, max_constraints=3)
    assert sample['contacts'] == ()
    assert sample['contact_pairs'] == [] and sample['contact_count'] == 0
    assert sample['contact_candidate_count'] == 1


@pytest.mark.parametrize('bad', ['count_mismatch', 'wrong_world', 'address_oob', 'friction_row', 'bad_negative'])
def test_contact_constraint_identity_fails_closed(bad):
    from cascade.sim.microduck_newton import read_native_state
    ns = native_fixture()
    d = ns.solver.mjw_data
    if bad == 'count_mismatch':
        d.nacon.value[0] = 2
    elif bad == 'wrong_world':
        d.contact.worldid.value[0] = 1
    elif bad == 'address_oob':
        d.contact.efc_address.value[0, 0] = 1
    elif bad == 'friction_row':
        d.efc.type.value[0, 0] = 1
    else:
        d.contact.efc_address.value[0, 0] = -2
    with pytest.raises((ValueError, RuntimeError)):
        read_native_state(ns, q_indices=np.arange(7, 21), dof_indices=np.arange(6, 20),
                          root_index=0, max_contacts=3, max_constraints=3)


def test_rotated_root_body_gyro_matches_actor_and_public_body_contract():
    # BaseState explicitly declares angular_velocity_BODY, not world. The
    # backend's internal angular_velocity key must preserve that convention.
    from cascade.sim.microduck_newton import read_native_state
    ns = native_fixture()
    root_half = np.sqrt(.5)
    ns.state_0.body_q.value[0, 3:] = [0., 0., root_half, root_half]
    ns.state_0.body_qd.value[0, 3:] = [1., 0., 0.]
    b = SoftwareBackend()
    def read():
        sample = read_native_state(ns, q_indices=np.arange(7,21), dof_indices=np.arange(6,20),
                                  root_index=0, max_contacts=3, max_constraints=3)
        # Clock only is software; body-frame conversion is the production reader.
        sample.update(step=b.step_count, sim_time=b.sim_time)
        return sample
    b.read = read
    s, _, c, p, _ = make_stepper(backend=b)
    try:
        s.start(); s.tick()
        np.testing.assert_allclose(b.read()['angular_velocity'], [0.,-1.,0.], atol=2e-6)
        np.testing.assert_allclose(p.inputs[0][0,:3], [0.,-1.,0.], atol=2e-6)
        np.testing.assert_allclose(c.state()['state']['angular_velocity_body'], [0.,-1.,0.], atol=2e-6)
    finally:
        s.close()


def test_native_read_reacquires_swapped_state_and_actual_contact_names():
    from cascade.sim.microduck_newton import read_native_state
    ns = native_fixture()
    kwargs = dict(q_indices=np.arange(7, 21), dof_indices=np.arange(6, 20), root_index=0,
                  max_contacts=3, max_constraints=3)
    first = read_native_state(ns, **kwargs)
    assert first['contacts'] == ('/robot/foot', '/scene/plane')
    np.testing.assert_allclose(first['linear_velocity'], [0, .8, 0])
    np.testing.assert_allclose(first['angular_velocity'], [0, 0, 2])
    ns.state_0 = copy.deepcopy(ns.state_0)
    ns.state_0.body_q.value[0, 0] = .3
    ns.state_0.joint_q.value[7] += .4
    ns.contacts.rigid_contact_count.value[0] = 0
    ns.solver.mjw_data.nacon.value[0] = 0
    second = read_native_state(ns, **kwargs)
    assert second['position'][0] == pytest.approx(.3)
    assert second['q'][0] != first['q'][0]
    assert second['contacts'] == ()
    assert ns.simulation_step_count == 2


@pytest.mark.parametrize('bad', ['negative_shape', 'shape_oob', 'contacts_full', 'constraints_full', 'nan', 'dtype'])
def test_native_channels_fail_closed(bad):
    from cascade.sim.microduck_newton import read_native_state
    ns = native_fixture()
    if bad == 'negative_shape':
        ns.contacts.rigid_contact_shape0.value[0] = -1
    elif bad == 'shape_oob':
        ns.contacts.rigid_contact_shape0.value[0] = 2
    elif bad == 'contacts_full':
        ns.contacts.rigid_contact_count.value[0] = 3
    elif bad == 'constraints_full':
        ns.solver.mjw_data.nefc.value[0] = 3
    elif bad == 'nan':
        ns.state_0.body_qd.value[0, 0] = np.nan
    else:
        ns.state_0.body_q.value = ns.state_0.body_q.value.astype(np.float64)
    with pytest.raises((ValueError, RuntimeError)):
        read_native_state(ns, q_indices=np.arange(7, 21), dof_indices=np.arange(6, 20),
                          root_index=0, max_contacts=3, max_constraints=3)


def test_render_does_not_tick_physics_and_stale_render_rejected():
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import capture_bound_rgb
    ns = native_fixture()
    events = []
    ns.update_fabric = lambda: events.append('fabric')
    app = NS(update=lambda: events.append('render'))
    readback = NS(get_data=lambda _: (Array(np.zeros((480, 640, 3)), np.uint8), {}),
                  get_render_times=lambda: render_times(.01))
    captured = capture_bound_rgb(ns, app, readback, updates=3)
    assert captured['step'] == 2 and captured['sim_time_s'] == .01
    assert events == ['fabric', 'render', 'render', 'render']
    def uncontrolled():
        ns.simulation_step_count += 1
    app.update = uncontrolled
    with pytest.raises(RuntimeError, match='physics'):
        capture_bound_rgb(ns, app, readback, updates=3)
    app.update = lambda: None
    ns.sim_time = .015
    with pytest.raises(ValueError, match='render'):
        capture_bound_rgb(ns, app, readback, updates=3)


def test_native_property_replacement_is_explicit_and_only_startup():
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import prepare_native_model
    model = NS()
    expected = {'joint_damping': .053, 'joint_armature': .0018, 'joint_effort_limit': 1e6,
                'joint_friction': .0048, 'joint_target_mode': 0, 'joint_target_ke': 0, 'joint_target_kd': 0}
    for key, value in expected.items():
        setattr(model, key, Array([value]*20, np.int32 if key == 'joint_target_mode' else np.float32))
    notifications = []
    ns = NS(model=model, solver=NS(notify_model_changed=notifications.append))
    newton = NS(ModelFlags=NS(JOINT_DOF_PROPERTIES='properties'))
    result = prepare_native_model(ns, np.arange(6, 20), source_cap=.96, newton=newton)
    assert notifications == ['properties']
    assert result['before']['joint_effort_limit'] == [1e6]*14
    np.testing.assert_allclose(result['after']['joint_effort_limit'], [.96]*14)
    np.testing.assert_allclose(result['after']['joint_damping'], [.005359668274599504]*14)
    np.testing.assert_allclose(result['after']['joint_armature'], [.0018077432831600838]*14)
    assert model.joint_damping.numpy()[0] == np.float32(.053)  # never edit free-root properties
    with pytest.raises(ValueError):
        prepare_native_model(ns, np.arange(6, 20), source_cap=.96, newton=newton)


@pytest.mark.parametrize('wrong', ['joint_damping', 'joint_armature', 'joint_effort_limit', 'joint_target_ke'])
def test_unexpected_native_property_rejected_before_mutation(wrong):
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import prepare_native_model
    model = NS()
    for key, value in {'joint_damping': .053, 'joint_armature': .0018, 'joint_effort_limit': 1e6,
                       'joint_friction': .0048, 'joint_target_mode': 0, 'joint_target_ke': 0, 'joint_target_kd': 0}.items():
        setattr(model, key, Array([value]*20, np.int32 if key == 'joint_target_mode' else np.float32))
    getattr(model, wrong).value[6] += .1
    before = {k: v.numpy() for k, v in vars(model).items()}
    ns = NS(model=model, solver=NS(notify_model_changed=lambda _: pytest.fail('must not mutate')))
    with pytest.raises(ValueError):
        prepare_native_model(ns, np.arange(6, 20), source_cap=.96,
                             newton=NS(ModelFlags=NS(JOINT_DOF_PROPERTIES=1)))
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(model, k).numpy(), v)


def test_boot_failure_and_explicit_camera_teardown_close_owned_app(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import KitNewtonBackend
    events = []
    fake_app = NS(close=lambda **kw: events.append(('app-close', kw['exit_code'])))
    monkeypatch.setitem(__import__('sys').modules, 'isaacsim', NS(SimulationApp=lambda *a, **kw: fake_app))
    backend = KitNewtonBackend(NS(device='cuda:0'), {}, tmp_path / 'experience.kit')
    def fail():
        raise ValueError('bootstrap rejected')
    monkeypatch.setattr(backend, '_initialize', fail)
    with pytest.raises(ValueError, match='bootstrap'):
        backend.open()
    assert events == []  # preserve failure receipt before process-terminating SDK close
    backend.close()
    backend.shutdown(exit_code=1)
    backend.shutdown(exit_code=1)
    assert events == [('app-close', 1)]


def test_source_actuators_disabled_only_after_full_validation():
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import disable_source_actuators
    class Prim:
        def __init__(self, name):
            self.name, self.active, self.cap = name, True, .96
        def GetName(self):
            return self.name
        def GetPath(self):
            return '/World/MicroDuck/Physics/' + self.name
        def GetTypeName(self):
            return 'MjcActuator'
        def GetAttribute(self, key):
            return NS(Get=lambda: self.cap if key.endswith('max') else -self.cap)
        def SetActive(self, active):
            self.active = active
    prims = [Prim(n) for n in POLICY_JOINTS]
    layer = NS(anonymous=True)
    stage = NS(Traverse=lambda: iter(prims), GetRootLayer=lambda: layer,
               GetEditTarget=lambda: NS(GetLayer=lambda: layer))
    prims[-1].cap = 1.
    with pytest.raises(ValueError):
        disable_source_actuators(stage)
    assert all(p.active for p in prims)
    prims[-1].cap = .96
    records = disable_source_actuators(stage)
    assert records and len(records) == 14 and all(not p.active for p in prims)


def test_epoch_change_during_episode_is_not_an_automatic_reset():
    stepper, backend, ctrl, _, _ = make_stepper()
    stepper.start()
    stepper.tick()
    ctrl.begin_epoch()  # privileged owner operation must require a NEW stepper
    before = backend.step_count
    with pytest.raises(RuntimeError, match='identity|epoch'):
        stepper.tick()
    assert backend.step_count == before


def test_effective_recipe_change_during_inference_never_commits():
    stepper, backend, ctrl, policy, actuator = make_stepper()
    stepper.start()
    def changed(obs):
        ctrl._identity['model_identity_sha256'] = 'f'*64
        return np.full(14, .1, np.float32)
    policy.infer = changed
    with pytest.raises(RuntimeError, match='identity'):
        stepper.tick()
    assert not actuator.targets
    assert backend.step_count == 2
    np.testing.assert_array_equal(policy.previous_action, np.zeros(14, np.float32))
    stepper.close()


def test_before_step_cannot_mutate_clocks_then_solve_again():
    stepper, backend, ctrl, _, actuator = make_stepper()
    stepper.start()
    actuator.before_step = lambda dt: backend.step()
    with pytest.raises(RuntimeError, match='clock|step'):
        stepper.tick()
    assert backend.step_count == 3  # one illicit tick detected; no second solve
    assert backend.contained


def test_sdk_close_receipt_status_and_camera_cleanup_are_explicit(tmp_path):
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import KitNewtonBackend
    events = []
    backend = KitNewtonBackend(NS(), {}, tmp_path / 'unused.kit')
    backend.timeline = NS(pause=lambda: events.append('pause'))
    backend.readback = NS(detach_render_times=lambda: events.append('times'),
                          detach_annotators=lambda names: events.append(('annotators', names)),
                          _invalidate_sensor=lambda: events.append('sensor'))
    backend.app = NS(close=lambda **kwargs: events.append(('sdk', kwargs['exit_code'])))
    backend.close()
    assert events == ['pause', 'times', ('annotators', ['rgb']), 'sensor']
    assert backend.receipt['owned_resources_closed'] is True
    backend.shutdown(exit_code=1)
    assert events[-1] == ('sdk', 1)
