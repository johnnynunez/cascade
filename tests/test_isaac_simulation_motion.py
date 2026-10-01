"""Physics time is the motion clock, including under pause and slow transport."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.isaac_arm import IsaacArm
from cascade.control.simulation_motion import PhysicsClock, SimulationMotion
from cascade.sim.bridge_client import BridgeError
from cascade.types import RobotState, SafetyViolation


def clock(step=0, **changes):
    value = dict(version=1, engine='physx', clock='SimulationManager',
                 source=('fake', 1), robot_id='/robot', epoch='epoch1',
                 sim_time=step * .01, physics_step=step, physics_dt_s=.01)
    value.update(changes)
    return value


class Sim:
    """Deterministic physics/transport model, separately driven clocks."""
    n_joints = 2
    motion_wall_timeout_s = 10.
    motion_rpc_timeout_s = .2
    settle_hold_s = .1
    _stopped = False
    _client = SimpleNamespace(_addr=('fake', 1))
    _cfg = {'bridge_robot_id': '/robot'}

    def __init__(self, monkeypatch, *, step_advance=1, wall_read=.01):
        self.wall = 0.
        self.step = 0
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.step_advance = step_advance
        self.wall_read = wall_read
        self.sent = []
        self.reads = []
        self.send_steps = 0
        self.on_read = None
        monkeypatch.setattr('time.monotonic', lambda: self.wall)
        monkeypatch.setattr('time.sleep', self.sleep)

    def sleep(self, dt):
        self.wall += dt

    def get_state(self, *, timeout_s):
        assert 0 < timeout_s <= self.motion_rpc_timeout_s
        self.wall += self.wall_read
        self.step += self.step_advance
        if self.on_read:
            self.on_read(self)
        state = RobotState(q=self.q.copy(), dq=self.dq.copy(), physics_clock=clock(self.step))
        self.reads.append((self.wall, self.step, self.q.copy()))
        return state

    def send_joint_target(self, q, *, timeout_s):
        assert 0 < timeout_s <= self.motion_rpc_timeout_s
        self.step += self.send_steps
        self.q = q.copy()
        self.sent.append((self.wall, self.step, q.copy()))


def stream(sim, **kwargs):
    return SimulationMotion(sim, **kwargs).stream(np.array([.1, -.1]), .2, 50., .045, 1., None)


@pytest.mark.parametrize('wall_read', [.001, .1])
def test_duration_and_approval_are_physical_even_with_different_wall_speed(monkeypatch, wall_read):
    sim = Sim(monkeypatch, wall_read=wall_read)
    approvals = []
    assert stream(sim, approve=lambda a, b, dt: approvals.append((a.copy(), b.copy(), dt)))
    assert len(sim.sent) == 10
    assert sim.sent[-1][1] * .01 >= .2
    assert all(b[1] - a[1] >= 2 for a, b in zip(sim.sent, sim.sent[1:]))
    np.testing.assert_allclose(sim.sent[-1][2], [.1, -.1])
    assert all(dt == pytest.approx(.02) for _, _, dt in approvals)
    # Revalidations while waiting retain the same actual nominal edges.
    edges = {tuple(b) for a,b,_ in approvals if not np.array_equal(a,b)}
    assert len(edges) == len(sim.sent)


def test_slow_send_requires_a_new_pacing_interval_after_ack(monkeypatch):
    sim = Sim(monkeypatch)
    sim.send_steps = 100  # Physics keeps running while transport is busy.
    assert stream(sim)
    # The next target must wait beyond those 100 steps and the post-ACK read.
    assert all(b[1] - a[1] >= 103 for a, b in zip(sim.sent, sim.sent[1:]))


def test_large_clock_jumps_do_not_skip_waypoints_or_burst(monkeypatch):
    sim = Sim(monkeypatch, step_advance=200)
    # Large sample gaps never prove settled; shorten the physics timeout.
    motion = SimulationMotion(sim)
    assert not motion.stream(np.array([.1, -.1]), .2, 50., .045, 1., None)
    assert len(sim.sent) == 10
    assert all(b[1] - a[1] >= 400 for a, b in zip(sim.sent, sim.sent[1:]))
    increments = np.diff(np.vstack([np.zeros(2), *[x[2] for x in sim.sent]]), axis=0)
    assert np.max(np.abs(increments)) < .019


def test_frozen_physics_expires_wall_budget_without_targets(monkeypatch):
    sim = Sim(monkeypatch, step_advance=0)
    sim.motion_wall_timeout_s = .12
    with pytest.raises(SafetyViolation, match='wall-time budget'):
        stream(sim)
    assert sim.wall < .14 and sim.sent == []


@pytest.mark.parametrize('kind', ['raw', 'harness'])
def test_cancellation_is_checked_while_physics_is_frozen(monkeypatch, kind):
    sim = Sim(monkeypatch, step_advance=0)
    def cancelled():
        if sim.wall > .05:
            raise SafetyViolation('cancelled by safety harness')
    def approve(*args):
        if kind == 'harness':
            cancelled()
    if kind == 'raw':
        sim.on_read = lambda s: setattr(s, '_stopped', s.wall > .05)
    with pytest.raises(SafetyViolation, match='stopped|cancelled'):
        stream(sim, approve=approve)
    assert sim.wall < .1 and sim.sent == []


@pytest.mark.parametrize('changes', [
    {'version': True}, {'engine': []}, {'clock': 'wall'}, {'epoch': ''},
    {'source': ('other', 1)}, {'robot_id': '/other'}, {'sim_time': float('nan')},
    {'sim_time': True}, {'physics_step': True}, {'physics_step': -1},
    {'physics_dt_s': float('inf')}, {'physics_dt_s': 0},
])
def test_invalid_clock_fails_closed_without_incidental_typeerror(changes):
    with pytest.raises(SafetyViolation):
        PhysicsClock(('fake', 1), '/robot').observe(clock(**changes))


@pytest.mark.parametrize('second', [
    clock(9), clock(11, epoch='restarted'), clock(10, sim_time=.11),
    clock(11, sim_time=.1), clock(11, sim_time=.15), clock(11, physics_dt_s=.02),
])
def test_regression_restart_and_inconsistent_steps_abort(second):
    c = PhysicsClock(('fake', 1), '/robot')
    assert c.observe(clock(10))
    with pytest.raises(SafetyViolation):
        c.observe(second)


def test_missing_clock_and_missing_robot_identity_fail_before_commands(monkeypatch):
    sim = Sim(monkeypatch)
    sim._cfg = {}
    with pytest.raises(SafetyViolation, match='robot identity'):
        stream(sim)
    assert sim.sent == []
    with pytest.raises(SafetyViolation, match='missing'):
        PhysicsClock(('fake', 1), '/robot').observe(None)


def test_duplicate_steps_never_prove_settled(monkeypatch):
    sim = Sim(monkeypatch, step_advance=0)
    sim.motion_wall_timeout_s = .2
    with pytest.raises(SafetyViolation, match='wall-time budget'):
        SimulationMotion(sim).settle(np.zeros(2), .045, 1.)
    assert sim.sent == []


def test_joint_change_without_physics_advance_is_inconsistent(monkeypatch):
    sim = Sim(monkeypatch, step_advance=0)
    motion = SimulationMotion(sim)
    motion.get_state()
    sim.q[0] = .01
    with pytest.raises(SafetyViolation, match='without a new physics step'):
        motion.get_state()


def test_settle_measures_fresh_position_stability_despite_solver_velocity_noise(monkeypatch):
    sim = Sim(monkeypatch)
    sim.dq = np.array([.0993, -.0205])  # Measured held-object solver noise.
    assert SimulationMotion(sim).settle(np.zeros(2), .045, 1.)
    assert sim.step >= 12


def test_position_inside_tolerance_but_still_moving_is_not_settled(monkeypatch):
    sim = Sim(monkeypatch)
    sim.on_read = lambda s: setattr(s, 'q', np.array([.02 * np.sin(s.step), 0.]))
    assert not SimulationMotion(sim).settle(np.zeros(2), .045, .5)


def test_unsampled_physics_gap_does_not_count_as_stability(monkeypatch):
    sim = Sim(monkeypatch, step_advance=20)
    assert not SimulationMotion(sim).settle(np.zeros(2), .045, 1.)


@pytest.mark.parametrize('field,value', [('q', [float('nan'), 0]), ('dq', [0, float('inf')]),
                                        ('dq', None), ('q', [0]), ('dq', ['x', 'y'])])
def test_bad_feedback_fails_before_targets(monkeypatch, field, value):
    sim = Sim(monkeypatch)
    setattr(sim, field, np.asarray(value))
    with pytest.raises(SafetyViolation, match='finite, exact-DOF'):
        stream(sim)
    assert sim.sent == []


def test_preflight_rebind_and_halt_check_are_preserved(monkeypatch):
    sim = Sim(monkeypatch)
    starts = []
    def preflight(q, duration):
        starts.append(q.copy())
        if len(starts) == 1:
            sim.q[0] = .01
    def halted():
        if len(starts) == 2:
            raise SafetyViolation('halt generation changed')
    motion = SimulationMotion(sim, before_stream=halted)
    with pytest.raises(SafetyViolation, match='halt generation'):
        motion.stream(np.ones(2), .2, 50., .045, 1., preflight)
    assert len(starts) == 2
    assert starts[1][0] == .01
    assert sim.sent == []


def test_before_stream_runs_at_start_only_and_wait_checks_preserve_escape(monkeypatch):
    sim = Sim(monkeypatch)
    starts = []
    def before():
        assert not sim.sent
        starts.append(sim.wall)
    def approve(a, b, dt):
        # Represents the harness's escape rule: a parked boundary permits
        # progress toward the valid interval but rejects a hold at that q.
        assert b[0] > a[0]
    assert stream(sim, approve=approve, before_stream=before)
    assert len(starts) == 1


def test_rpc_budget_is_clamped_to_remaining_wall_budget(monkeypatch):
    sim = Sim(monkeypatch)
    sim.motion_wall_timeout_s = .007
    sim.wall_read = .008
    with pytest.raises(SafetyViolation, match='wall-time budget'):
        stream(sim)
    assert sim.sent == []


def test_read_only_capability_binds_endpoint_without_trusting_wire(monkeypatch):
    sim = Sim(monkeypatch)
    arm = IsaacArm(Cfg({'n_joints': 2, 'bridge_robot_id': '/robot'}))
    calls = []
    def state(**kwargs):
        calls.append(kwargs)
        return dict(q=[0.,0.], dq=[0.,0.], physics_clock=clock(source=('spoof', 999)))
    arm._client = SimpleNamespace(_addr=('fake',1), state=state)
    result = arm.validate_simulation_clock()
    assert result['source'] == ('fake', 1)
    assert result['epoch'] == 'epoch1' and len(calls) == 1
    assert calls[0]['timeout_s'] <= 1.


@pytest.mark.parametrize('field', ['q', 'dq'])
@pytest.mark.parametrize('value', [[.2], [.2] * 7, ['.2'] * 6, [True] * 6, [float('nan')] * 6])
def test_driver_rejects_wire_shape_before_sign_broadcast_or_truncation(field, value):
    arm = IsaacArm(Cfg({'n_joints': 6, 'bridge_robot_id': '/robot', 'joint_signs': [-1] * 6}))
    wire = dict(q=[0.] * 6, dq=[0.] * 6, physics_clock=clock())
    wire[field] = value
    arm._client = SimpleNamespace(_addr=('fake', 1), state=lambda **kw: wire)
    with pytest.raises(BridgeError, match='exact-DOF asset'):
        arm.validate_simulation_clock()
