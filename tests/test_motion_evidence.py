"""Optional evidence preserves the existing physical-clock control sequence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control import motion_evidence as evidence
from cascade.control import simulation_motion
from cascade.control.isaac_arm import IsaacArm
from cascade.sim.target_receipts import TargetReceipts
from cascade.types import RobotState, SafetyViolation
from cascade.agent.effects import PostconditionChecker
from cascade.agent.trace import TraceLogger
from cascade.config import load_demo_config
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.skills.runtime import SkillRuntime


@pytest.fixture
def runtime(tmp_path):
    # Real dispatcher and isolated stores, without an actuator or solver.
    rt = SkillRuntime(
        camera=None, depth_provider=None, detector=None, extrinsics=None,
        kin=None, safe_arm=SimpleNamespace(), memory=EpisodicMemory(),
        beliefs=BeliefStore(), trace=TraceLogger(tmp_path / "run"),
        cfg=load_demo_config(camera="mock", arm="mock", llm="mock"),
    )
    rt.observe = lambda: None
    rt.effects = PostconditionChecker()
    return rt


class Sim:
    n_joints = 2
    motion_wall_timeout_s = 10.
    motion_rpc_timeout_s = .2
    settle_hold_s = .1
    _stopped = False
    _client = SimpleNamespace(_addr=('fake', 1))
    _cfg = {'bridge_robot_id': '/robot'}

    def __init__(self, monkeypatch, slow=0, jump=1, cancel=False):
        self.wall, self.step = 0., 0
        self.q = np.zeros(2)
        self.slow, self.jump, self.cancel = slow, jump, cancel
        self.calls = []
        # Replace only this module binding. Never change the global time module.
        monkeypatch.setattr(simulation_motion, 'time', SimpleNamespace(
            monotonic=lambda: self.wall, sleep=self.sleep))

    def sleep(self, dt):
        self.calls.append(('sleep', dt))
        self.wall += dt

    def get_state(self, *, timeout_s):
        self.wall += .01
        self.step += self.jump
        if self.cancel and self.step > 6:
            self._stopped = True
        clock = dict(version=1, engine='physx', clock='SimulationManager',
                     source=self._client._addr, robot_id='/robot', epoch='epoch',
                     physics_step=self.step, sim_time=self.step*.01, physics_dt_s=.01)
        self.calls.append(('read', self.step, timeout_s))
        return RobotState(q=self.q.copy(), dq=np.zeros(2), physics_clock=clock)

    def send_joint_target(self, q, *, timeout_s):
        self.step += self.slow
        self.q = q.copy()
        self.calls.append(('send', self.step, q.tolist(), timeout_s))


@pytest.mark.parametrize('slow,jump,cancel', [(0,1,False), (100,1,False), (0,200,False), (0,1,True), (0,0,False)])
def test_enabled_disabled_commands_reads_approvals_waits_and_outcome_equal(monkeypatch, tmp_path, slow, jump, cancel):
    runs = []
    for enabled in (False, True):
        monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(tmp_path) if enabled else '')
        sim = Sim(monkeypatch, slow, jump, cancel)
        if jump == 0: sim.motion_wall_timeout_s = .12
        approve = lambda a,b,dt: sim.calls.append(('approve', a.tolist(), b.tolist(), dt))
        context = {}
        with evidence.record_skill('test', 'arm', context, enabled=True):
            try:
                result = simulation_motion.SimulationMotion(sim, approve=approve).stream(
                    np.array([.1, -.1]), .2, 50., .045, 1., None)
            except SafetyViolation as exc:
                result = str(exc)
        runs.append((sim.calls, result))
    assert runs[0] == runs[1]
    document = json.loads(Path(context['motion_evidence']['path']).read_text())
    assert document['logging_complete'] is True
    ready = [r for r in document['events'] if r['kind'] == 'waypoint_ready']
    anchors = [r for r in document['events'] if r['kind'] == 'post_ack_anchor']
    if slow:
        steps = [r['data']['clock']['physics_step'] for r in anchors]
        assert all(b-a >= 103 for a,b in zip(steps, steps[1:]))
        assert len(ready) == len(anchors) == 10


def test_runtime_nested_home_labels_all_segments_and_trace_exact_goal(runtime, monkeypatch, tmp_path):
    monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(tmp_path/'receipts'))
    sim = Sim(monkeypatch)
    home = [.123456789012345, -.098765432109876]
    runtime.cfg._data['arm'] = Cfg({'n_joints': 2, 'home_q': home})
    def move(q, duration_s, **_kw):
        assert duration_s == 3.
        motion = simulation_motion.SimulationMotion(sim)
        assert motion.stream(np.asarray(q)/2, .2, 50., .045, 1., None)
        return simulation_motion.SimulationMotion(sim).stream(np.asarray(q), .2, 50., .045, 1., None)
    runtime.arm = SimpleNamespace(move_joints=move, move_planned=move)
    runtime.skill_pick_and_place = lambda **_kw: runtime.skill_move_home()
    result = runtime.execute('pick_and_place', {'object': 'cube', 'target': 'box'})
    assert result['ok'] is True
    row = json.loads(runtime.trace._trace_path.read_text().splitlines()[-1])
    doc = json.loads(Path(row['context']['motion_evidence']['path']).read_text())
    streams = [e for e in doc['events'] if e['kind'] == 'stream_begin']
    assert len(streams) == 2 and len({e['stream'] for e in streams}) == 2
    assert all(e['phase'] == 'home' for e in streams)
    assert streams[-1]['data']['q_goal'] == home
    assert doc['arm'] == 'default' and doc['skill'] == 'pick_and_place'


def test_limits_and_disk_failure_do_not_replace_original_exception(monkeypatch, tmp_path):
    blocked = tmp_path/'file'; blocked.write_text('not a directory')
    monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(blocked))
    context = {}
    with pytest.raises(ValueError, match='original'):
        with evidence.record_skill('test', 'arm', context, enabled=True):
            evidence.event('something', exact=np.array([.1234567890123]))
            raise ValueError('original')
    assert context['motion_evidence']['logging_complete'] is False
    assert context['motion_evidence']['logging_errors'] == 1 and not evidence.active()
    tiny = evidence.Recording('skill', 'arm', max_events=1)
    tiny.record('first', {}); tiny.record('second', {})
    receipt = tiny.finish(tmp_path/'tiny')
    assert not receipt['logging_complete'] and receipt['dropped_events'] == 1
    assert receipt['submission_coverage']['complete'] is False


def test_read_only_runtime_does_not_create_drive_evidence(runtime, monkeypatch, tmp_path):
    out = tmp_path/'absent'
    monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(out))
    runtime.skill_get_observation = lambda: {'observed': True}
    assert runtime.execute('get_observation', {})['ok'] is True
    row = json.loads(runtime.trace._trace_path.read_text().splitlines()[-1])
    assert 'motion_evidence' not in row['context'] and not out.exists()


def test_isaac_driver_none_contract_exact_signs_and_matching_submission(monkeypatch, tmp_path):
    monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(tmp_path))
    ledger = TargetReceipts('/robot', ['a', 'b'], 'epoch')
    clock = dict(version=1, engine='physx', clock='SimulationManager', robot_id='/robot',
                 epoch='epoch', physics_step=1, sim_time=.01, physics_dt_s=.01)
    ledger.completed_update(clock)
    sent = []
    class Client:
        _addr = ('fake', 1)
        def set_joints(self, q, *, command_id=None, timeout_s=None):
            sent.append(q.copy())
            ack = ledger.queued(q, command_id)
            ledger.setter_returned(ack['sequence'], np.asarray(q, np.float32), [0, 1])
            return ack
        def state(self, **_kw):
            return {'q': sent[-1].copy(), 'dq': [0.,0.], 'physics_clock': clock,
                    'target_receipts': ledger.snapshot()}
    arm = IsaacArm(Cfg({'n_joints': 2, 'joint_signs': [-1, 1], 'bridge_robot_id': '/robot'}))
    arm._client = Client()
    context = {}
    with evidence.record_skill('test', 'arm', context, enabled=True):
        assert arm.send_joint_target(np.array([.123456789, .2])) is None
        arm.get_state()
    summary = context['motion_evidence']
    assert summary['submission_coverage']['complete'] is True
    doc = json.loads(Path(summary['path']).read_text())
    request = next(r['data'] for r in doc['events'] if r['kind'] == 'target_request')
    assert request['q_asset'] == [-.123456789, .2]
    assert doc['submission_coverage']['physical_acceptance'] is False
    for mode in ('missing', 'epoch', 'hash', 'changed_first', 'no_clock'):
        events = copy.deepcopy(doc['events'])
        s = next(r for r in events if r['kind'] == 'isaac_state')
        write = s['data']['target_receipts']['last_written']
        if mode == 'missing': s['data']['target_receipts'] = None
        elif mode == 'epoch': write['epoch'] = 'old'
        elif mode == 'hash': write['first_write']['arm_target']['sha256'] = '0'*64
        elif mode == 'no_clock': write['first_write']['prior_completed_update'] = None
        else:
            later = copy.deepcopy(s)
            later['data']['target_receipts']['last_written']['first_write']['producer_monotonic_s'] -= 1
            events.append(later)
        assert evidence.submission_coverage(events)['complete'] is False
