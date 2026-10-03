"""Telemetry must explain attempts without adding robot reads or commands."""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.isaac_arm import IsaacArm
from cascade.grasping import evidence
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.safety.harness import SafeArm, SafetyHarness
from cascade.skills.runtime import SkillRuntime
from cascade.types import Detection, Frame, Grasp, ObjectFix, SkillError


def runtime(monkeypatch, fail=False):
    clock = [0.0]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    monkeypatch.setattr('time.sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    calls = []

    class Bridge:
        _addr = ('test', 0)
        step = 0
        q = np.array([.2, 0, .2])
        gripper_pos = 1.
        def state(self, *, timeout_s=None):
            self.step += 1
            clock[0] += .02
            calls.append(('read', self.q.tolist(), self.gripper_pos))
            return {'q': self.q.copy(), 'dq': np.zeros(3), 'gripper_pos': self.gripper_pos,
                    't': clock[0], 'physics_clock': {
                        'version': 1, 'engine': 'physx', 'clock': 'SimulationManager',
                        'robot_id': '/robot', 'epoch': 'test', 'sim_time': self.step * .02,
                        'physics_step': self.step, 'physics_dt_s': .02}}
        def set_joints(self, q, *, timeout_s=None):
            calls.append(('joints', q.tolist()))
            if fail and q[2] < .12:
                raise SkillError('original descent transport failure')
            self.q = q.copy()
        def gripper(self, pos, effort):
            calls.append(('gripper', pos, effort))
            self.gripper_pos = max(.35, pos)  # deterministic object contact double

    class Kin:
        def ik(self, pose, seed):
            return SimpleNamespace(q=pose[:3, 3].copy(), success=True, error=0.)
        def fk(self, q):
            pose = np.eye(4); pose[:3, 3] = q
            return pose

    raw = IsaacArm(Cfg({'n_joints': 3, 'bridge_robot_id': '/robot'}))
    raw._client = Bridge()
    harness = SimpleNamespace(
        limits=SimpleNamespace(workspace_min=[0, -.3, 0], table_z=0., max_joint_vel=10.),
        estopped=False, vet_pose=lambda *args, **kwargs: None,
        vet_step=lambda *args, **kwargs: None, _halt_generation=0,
        check_stream_start=lambda **kwargs: None,
        check_contact_episode=lambda **kwargs: None,
        begin_motion=lambda **kwargs: None, end_motion=lambda: None,
        approve=lambda *args, **kwargs: None,
        allow_grasp_descent=lambda *args, **kwargs: None,
        clear_grasp_exemption=lambda: None,
    )
    # This telemetry double retains the real generation guard; its command
    # trace still measures only transport reads and writes, not gate calls.
    harness._check_halt_generation = SafetyHarness._check_halt_generation.__get__(harness)
    harness._pending_release_episode = None
    harness.check_release_episode = SafetyHarness.check_release_episode.__get__(harness)
    harness._pending_model_withdrawal = None
    harness.check_model_withdrawal = SafetyHarness.check_model_withdrawal.__get__(harness)
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.arm, rt.kin = SafeArm(raw, harness), Kin()
    rt.cfg = Cfg({'arm': {'home_q': [.2, 0, .2], 'gripper': {}},
                  'grasp': {'pregrasp_offset_m': .1, 'move_duration_s': .1,
                            'descend_duration_s': .1, 'close_feedback_timeout_s': 2.}})
    rt.held_object = None
    rt._max_width, rt._grip_open, rt._grip_closed = .09, 1., 0.
    rt.memory = SimpleNamespace(add=lambda *args, **kwargs: None)
    rt.beliefs = SimpleNamespace(mark_removed=lambda *args, **kwargs: None)
    rt.grasp_memory = SimpleNamespace(record=lambda *args, **kwargs: None)
    g = Grasp(np.array([.2, 0, .05]), _yaw_rotation(0), .05, np.array([0, 0, -1.]), .9, 'orange')
    rt._plan_grasps = lambda *args, **kwargs: [g]
    points = np.arange(180, dtype=np.float32).reshape(60, 3) / 1000
    fix = ObjectFix('orange', np.array([.2, 0, .05]), points,
                    Detection('orange', .95, np.array([0, 0, 3, 3])),
                    np.array([.04, .04, .04]), np.eye(3))
    frame = Frame(np.zeros((3, 3, 3), np.uint8), None, np.eye(3),
                  frame_id=42, capture={'t': 15., 'source': 'test bridge'})
    return rt, calls, fix, frame


def receipt(directory):
    files = list(directory.glob('*.json'))
    assert len(files) == 1
    return json.loads(files[0].read_text())


@pytest.fixture
def synthetic_git_metadata(monkeypatch):
    """Only Git metadata is synthetic; file hashes and receipt writes stay real."""
    state = SimpleNamespace(calls=[], status_timeout=False, timeouts=[])
    real_run = subprocess.run
    repo = str(Path(evidence.__file__).resolve().parents[3])

    def run(argv, *, capture_output, text, check, timeout):
        assert argv[:3] == ['git', '-C', repo]
        assert capture_output is True and text is True and check is True
        assert timeout == 2.0  # the production provenance budget is unchanged
        command = argv[3:]
        assert command in (['rev-parse', 'HEAD'],
                           ['status', '--porcelain', '--untracked-files=no'])
        state.calls.append(command)
        if command[0] == 'status' and state.status_timeout:
            # Exercise the real subprocess timeout/cleanup path, without
            # depending on repository size or the host's Git scheduling.
            try:
                real_run([sys.executable, '-c', 'import time; time.sleep(30)'],
                         capture_output=capture_output, text=text, check=check,
                         timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                state.timeouts.append(exc.timeout)
                raise subprocess.TimeoutExpired(argv, exc.timeout) from exc
            raise AssertionError('controlled metadata process did not time out')
        value = ('synthetic-grasp-evidence-test-head\n' if command[0] == 'rev-parse'
                 else ' M synthetic-test-metadata\n')
        return subprocess.CompletedProcess(argv, 0, stdout=value, stderr='')

    # Replace this module's namespace, not the shared subprocess.run function.
    monkeypatch.setattr(evidence, 'subprocess', SimpleNamespace(run=run))
    return state


@pytest.mark.parametrize('source_status', ['healthy', 'git_timeout'])
def test_enabled_and_disabled_have_identical_actuator_commands_reads_and_result(monkeypatch, tmp_path, source_status):
    # The semantic A/B needs a controlled diagnostic subprocess, not a promise
    # that a loaded CI host completes real git status within the 2 s budget.
    # Source-file hashes and the actual telemetry/actuator paths remain real.
    import subprocess
    git_calls = []
    def git(args, **kwargs):
        git_calls.append(args[3:])
        assert args[:2] == ['git', '-C'] and kwargs['timeout'] == 2.
        if args[3:] == ['status', '--porcelain', '--untracked-files=no']:
            if source_status == 'git_timeout':
                raise subprocess.TimeoutExpired(args, kwargs['timeout'])
            return subprocess.CompletedProcess(args, 0, stdout='')
        assert args[3:] == ['rev-parse', 'HEAD']
        return subprocess.CompletedProcess(args, 0, stdout='software-fixture-head\n')
    monkeypatch.setattr(evidence, 'subprocess', SimpleNamespace(run=git))
    monkeypatch.delenv('CASCADE_GRASP_EVIDENCE_DIR', raising=False)
    rt, calls, fix, frame = runtime(monkeypatch)
    before = rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    expected = calls.copy()
    assert not list(tmp_path.glob('*.json')) and not git_calls
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    rt, calls, fix, frame = runtime(monkeypatch)
    result = rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert result == before
    assert calls == expected  # includes EVERY bridge read, joint command, gripper command
    doc = receipt(tmp_path)
    assert git_calls == [['rev-parse', 'HEAD'], ['status', '--porcelain', '--untracked-files=no']]
    assert doc['logging_ok'] is (source_status == 'healthy')
    if source_status == 'git_timeout':
        assert [(e['operation'], e['type']) for e in doc['logging_errors']] == [('source_at_flush', 'TimeoutExpired')]
    else:
        assert doc['logging_errors'] == []
    assert doc['started_unix_ns'] <= doc['finished_unix_ns']
    assert doc['started_monotonic_s'] == 0.
    assert {'selected', 'jaw_datum', 'localized', 'isaac_feedback', 'ik_targets',
            'isaac_joint_target_sent', 'isaac_gripper_target_sent'} <= {e['kind'] for e in doc['events']}
    assert {'entry', 'planning', 'open', 'home', 'pregrasp', 'descent', 'close', 'lift', 'verify'} <= {e['phase'] for e in doc['events']}
    path = tmp_path / doc['arrays']['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == doc['arrays']['sha256']
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['0000_localized_points_base_m'], fix.points)
    loc = next(e['data'] for e in doc['events'] if e['kind'] == 'localized')
    assert loc['frame_id'] == 42 and loc['capture']['t'] == 15.
    assert loc['position_base_m'] == fix.position.tolist()


def test_cached_aim_uses_close_pose_not_postlift_tcp(monkeypatch):
    """A real runtime grasp must not turn its own lift into a negative offset."""
    rt, calls, fix, frame = runtime(monkeypatch)
    result = rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert result['held'] == 'orange'
    last_close = max(i for i, c in enumerate(calls) if c[0] == 'gripper')
    first_lift = next(i for i in range(last_close + 1, len(calls)) if calls[i][0] == 'joints')
    close_q = [c[1] for c in calls[last_close:first_lift] if c[0] == 'read'][-1]
    np.testing.assert_allclose(rt._held_offset, fix.position - rt.kin.fk(close_q)[:3, 3])
    old_offset = fix.position - rt.kin.fk(rt.arm.raw._client.q)[:3, 3]
    assert old_offset[2] < -.06 and abs(rt._held_offset[2]) < .01
    assert rt._held_observation_floor['physics_step'] > 0
    np.testing.assert_array_equal(rt._held_observation_floor_q, rt.arm.raw._client.q)


def test_failed_attempt_preserves_original_exception_and_exact_command_read_trace(monkeypatch, tmp_path):
    runs = []
    failures = []
    for enabled in (False, True):
        monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path) if enabled else '')
        rt, calls, fix, frame = runtime(monkeypatch, fail=True)
        with pytest.raises(SkillError, match='original descent transport failure') as failure:
            rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
        runs.append(calls)
        failures.append(str(failure.value))
    assert runs[0] == runs[1]
    assert failures[0] == failures[1]
    doc = receipt(tmp_path)
    failure = doc['events'][-1]
    assert failure['kind'] == 'attempt_exception' and failure['phase'] == 'descent'
    assert failure['data']['message'] == failures[0]
    assert 'set_joints' in failure['data']['traceback']
    assert not any(e['phase'] == 'close' for e in doc['events'])


def test_unwritable_directory_is_visible_without_masking_robot_exception(monkeypatch, tmp_path, capsys, synthetic_git_metadata):
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('occupied')
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(blocked))
    rt, calls, fix, frame = runtime(monkeypatch, fail=True)
    with pytest.raises(SkillError, match='original descent transport failure'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    output = capsys.readouterr()
    assert output.out == ''  # MCP stdout remains a clean protocol transport
    report = json.loads(output.err.split('[grasp-evidence] ', 1)[1])
    assert report['logging_ok'] is False and report['receipt'] is None
    assert {e['operation'] for e in report['logging_errors']} == {'write_arrays', 'write_receipt'}
    assert evidence._ACTIVE.get() is None


def test_array_write_failure_still_publishes_a_receipt_with_explicit_error(monkeypatch, tmp_path):
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    def no_space(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(evidence.np, 'savez', no_space)
    rt, _, fix, frame = runtime(monkeypatch)
    result = rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert result['held'] == 'orange'
    doc = receipt(tmp_path)
    assert doc['logging_ok'] is False
    assert doc['logging_errors'][0]['message'] == 'disk full'
    assert 'arrays' not in doc


def test_attempt_buffer_copies_arrays_and_excludes_other_threads(monkeypatch, tmp_path):
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    values = np.array([1., 2.])
    @evidence.record_attempt
    def action(rt, label):
        evidence.event('candidate', position=values)
        evidence.array('points', values)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(evidence.event, 'watcher_unrelated', q=values).result()
        values[:] = 9.
        return {'ok': True}
    action(SimpleNamespace(cfg=SimpleNamespace(arm={}, grasp={})), 'orange')
    doc = receipt(tmp_path)
    assert not any(e['kind'] == 'watcher_unrelated' for e in doc['events'])
    assert next(e['data']['position'] for e in doc['events'] if e['kind'] == 'candidate') == [1., 2.]
    with np.load(tmp_path / doc['arrays']['path']) as arrays:
        np.testing.assert_array_equal(arrays['0000_points'], [1., 2.])


def test_rgbd_masks_and_effective_calibration_are_copied_from_existing_frame(monkeypatch, tmp_path):
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    rt, _, fix, frame = runtime(monkeypatch)
    frame.rgb[:] = 23
    frame.depth_m = np.full((3, 3), .7, np.float32)
    frame.depth_source = 'sensor'
    frame.robot_mask = np.eye(3, dtype=bool)
    frame.prop_masks = {"/World/pink": np.eye(3, dtype=bool),
                        "/World/green": np.fliplr(np.eye(3, dtype=bool)).copy()}
    frame.payload_mask = np.eye(3, dtype=bool)
    fix.detection.mask = np.fliplr(np.eye(3, dtype=bool)).copy()
    T = np.eye(4); T[0, 3] = .3
    extrinsics = SimpleNamespace(mode='eye_to_hand', T=T)
    @evidence.record_attempt
    def action(rt, label):
        evidence.localized(frame, fix, extrinsics)
        frame.rgb[:] = 99
        frame.depth_m[:] = 4
        frame.robot_mask[:] = False
        for mask in frame.prop_masks.values():
            mask[:] = False
        frame.payload_mask[:] = False
        fix.detection.mask[:] = False
        frame.K[:] = 0
        T[:] = 0
        return {'ok': True}
    action(rt, 'orange')
    doc = receipt(tmp_path)
    assert doc['logging_ok']
    with np.load(tmp_path / doc['arrays']['path'], allow_pickle=False) as arrays:
        values = {key[5:]: arrays[key] for key in arrays.files}
        np.testing.assert_array_equal(values['frame_rgb_bgr'], np.full((3, 3, 3), 23))
        np.testing.assert_allclose(values['frame_depth_m'], .7)
        np.testing.assert_array_equal(values['frame_robot_mask'], np.eye(3, dtype=bool))
        np.testing.assert_array_equal(values['frame_target_mask'], np.fliplr(np.eye(3, dtype=bool)))
        prop_event = next(e['data'] for e in doc['events'] if e['kind'] == 'frame_prop_masks')
        assert prop_event['frame_id'] == frame.frame_id
        assert prop_event['capture'] == frame.capture
        assert prop_event['scope'] == 'audit only; not a target-contact exemption'
        np.testing.assert_array_equal(arrays[prop_event['arrays']['/World/pink']], np.eye(3, dtype=bool))
        np.testing.assert_array_equal(arrays[prop_event['arrays']['/World/green']], np.fliplr(np.eye(3, dtype=bool)))
        np.testing.assert_array_equal(values['frame_payload_mask'], np.eye(3, dtype=bool))
        np.testing.assert_array_equal(values['frame_K'], np.eye(3))
        assert values['frame_T_base_cam'][0, 3] == .3
    assert next(e['data'] for e in doc['events'] if e['kind'] == 'localized')['effective_T_base_cam'][0][3] == .3


@pytest.mark.parametrize('has_capture_transform', [False, True])
def test_evidence_never_queries_live_eye_in_hand_fk(monkeypatch, tmp_path, has_capture_transform):
    from cascade.perception.grounding import Extrinsics
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    rt, _, fix, frame = runtime(monkeypatch)
    called = []
    def live_fk():
        called.append('hardware read')
        raise AssertionError('telemetry must not read the arm')
    ext = Extrinsics(mode='eye_in_hand', fk_tcp2base=live_fk)
    frame.T_base_cam = np.eye(4) if has_capture_transform else None
    @evidence.record_attempt
    def action(rt, label):
        evidence.localized(frame, fix, ext)
        return {'ok': True}
    assert action(rt, 'orange') == {'ok': True}
    assert called == []
    doc = receipt(tmp_path)
    assert doc['logging_ok'] is has_capture_transform
    loc = next(e['data'] for e in doc['events'] if e['kind'] == 'localized')
    assert (loc['effective_T_base_cam'] is not None) is has_capture_transform
    if not has_capture_transform:
        assert doc['logging_errors'][0]['operation'] == 'frame_T_base_cam'


def test_capture_error_and_truncation_are_never_silent_success(monkeypatch, tmp_path):
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    monkeypatch.setattr(evidence, '_MAX_EVENTS', 3)
    @evidence.record_attempt
    def action(rt, label):
        evidence.event('bad_value', value=object())
        for _ in range(10):
            evidence.event('bounded')
        return {'ok': True}
    assert action(SimpleNamespace(cfg=SimpleNamespace(arm={}, grasp={})), 'orange') == {'ok': True}
    doc = receipt(tmp_path)
    assert doc['logging_ok'] is False
    assert doc['dropped_events'] > 0
    assert doc['logging_errors'][0]['operation'] == 'event:bad_value'


def test_real_planner_wire_capture_records_empty_retry_and_exact_converted_candidate(monkeypatch, tmp_path):
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner
    from test_graspgenx_backend import _cube_fix
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    planner = GraspGenXPlanner(Cfg({'graspgenx': {'tip_offset_m': .1}}))
    pose = np.eye(4); pose[:3, :3] = np.diag([1., -1., -1.]); pose[:3, 3] = [.2, 0., .15]
    responses = iter([{'grasps': [], 'confidences': []},
                      {'grasps': [pose], 'confidences': [.9], 'branch_tags': ['diff']}])
    payloads = []
    def request(payload):
        payloads.append(payload)
        return next(responses)
    planner._client.request = request
    fix = _cube_fix()
    @evidence.record_attempt
    def action(rt, label):
        grasps = planner.plan(fix)
        evidence.event('returned_candidates', grasps=grasps)
        return {'ok': True}
    action(SimpleNamespace(cfg=SimpleNamespace(arm={}, grasp={})), 'orange')
    assert len(payloads) == 2  # the planner's existing empty-response retry only
    doc = receipt(tmp_path)
    assert doc['logging_ok'] is True
    with np.load(tmp_path / doc['arrays']['path']) as arrays:
        np.testing.assert_array_equal(arrays['0000_ggx_request_points_base_m'], fix.points.astype(np.float32))
        np.testing.assert_array_equal(arrays['0003_ggx_raw_poses'], [pose])
    gs = next(e['data']['grasps'] for e in doc['events'] if e['kind'] == 'returned_candidates')
    assert gs[0]['position'] == pytest.approx([.2, 0., .05])


@pytest.mark.parametrize('response', [None, [], {}, {'grasps': []}])
def test_malformed_ggx_response_keeps_the_same_original_exception(monkeypatch, tmp_path, response):
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner
    from test_graspgenx_backend import _cube_fix
    failures = []
    for enabled in (False, True):
        monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path) if enabled else '')
        planner = GraspGenXPlanner(Cfg({'graspgenx': {}}))
        planner._client.request = lambda _payload: response
        @evidence.record_attempt
        def action(rt, label):
            return planner.plan(_cube_fix())
        try:
            action(SimpleNamespace(cfg=SimpleNamespace(arm={}, grasp={})), 'orange')
        except Exception as exc:
            failures.append((type(exc), str(exc)))
    assert len(failures) == 2 and failures[0] == failures[1]
    assert receipt(tmp_path)['logging_ok'] is False


def test_existing_memory_prior_and_nudge_are_captured_before_candidate_mutation(monkeypatch, tmp_path):
    from cascade.memory.grasp_memory import GraspOutcomeMemory
    from test_graspgenx_backend import _cube_fix
    monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path))
    fix = _cube_fix()
    memory = GraspOutcomeMemory()
    memory.record('orange', fix, None, success=False, reason='air grasp')
    cfg = Cfg({'arm': {}, 'safety': {'table_z': 0.}, 'grasp': {'backend': 'obb'}})
    rt = SimpleNamespace(cfg=cfg, grasp_memory=memory, _max_width=.09,
                         _tool_axis_order='down_open', memory=SimpleNamespace(add=lambda *a: None))
    @evidence.record_attempt
    def action(rt, label):
        candidates = SkillRuntime._plan_grasps(rt, fix, label=label)
        evidence.event('after_memory', candidates=candidates, z_nudge_m=rt._last_grasp_z_nudge)
        return {'ok': True}
    action(rt, 'orange')
    doc = receipt(tmp_path)
    assert doc['logging_ok'] is True
    events = {e['kind']: e['data'] for e in doc['events']}
    assert events['memory_prior']['prior']['losses'] == 1
    assert events['memory_prior']['prior']['nudges']['grasp_z_delta'] == -.005
    original = events['backend_candidates']['grasps'][0]['position'][2]
    reranked = events['memory_reranked_candidates']['grasps'][0]['position'][2]
    final = events['after_memory']['candidates'][0]['position'][2]
    assert original == reranked and final == pytest.approx(original - .005)
    assert events['after_memory']['z_nudge_m'] == -.005
    assert doc['source']['git_head']
    assert doc['source']['sha256']['src/cascade/skills/runtime.py'] == hashlib.sha256(
        (Path(__file__).parents[1] / 'src/cascade/skills/runtime.py').read_bytes()).hexdigest()
    for name in ('rebot_rs_fingers.json', 'rebot_rs_fingers_physx.json',
                 'physx_finger_cooking_source.json'):
        relative = 'assets/grasp_geometry/' + name
        assert doc['source']['sha256'][relative] == hashlib.sha256(
            (Path(__file__).parents[1] / relative).read_bytes()).hexdigest()


@pytest.mark.parametrize('actor_fails', [False, True])
def test_git_metadata_timeout_preserves_commands_result_and_actor_exception(
        monkeypatch, tmp_path, synthetic_git_metadata, actor_fails):
    synthetic_git_metadata.status_timeout = True
    traces, outcomes = [], []
    for enabled in (False, True):
        monkeypatch.setenv('CASCADE_GRASP_EVIDENCE_DIR', str(tmp_path) if enabled else '')
        rt, calls, fix, frame = runtime(monkeypatch, fail=actor_fails)
        if actor_fails:
            with pytest.raises(SkillError, match='original descent transport failure') as error:
                rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
            outcomes.append((type(error.value), str(error.value)))
        else:
            outcomes.append(rt.skill_grasp_object('orange', _fix=fix, _frame=frame))
        traces.append(calls.copy())
    assert outcomes[0] == outcomes[1]
    assert traces[0] == traces[1]
    assert synthetic_git_metadata.timeouts == [2.0]
    doc = receipt(tmp_path)
    assert doc['logging_ok'] is False
    assert doc['dropped_events'] == 0
    assert len(doc['logging_errors']) == 1
    error = doc['logging_errors'][0]
    assert error['operation'] == 'source_at_flush' and error['type'] == 'TimeoutExpired'
    assert "'status', '--porcelain', '--untracked-files=no'" in error['message']
    assert 'timed out after 2.0 seconds' in error['message']
    assert 'git_tracked_status_porcelain' not in doc['source']
    assert doc['source']['sha256']['src/cascade/skills/runtime.py'] == hashlib.sha256(
        (Path(__file__).parents[1] / 'src/cascade/skills/runtime.py').read_bytes()).hexdigest()
    arrays = tmp_path / doc['arrays']['path']
    assert hashlib.sha256(arrays.read_bytes()).hexdigest() == doc['arrays']['sha256']
    assert doc['events'][-1]['kind'] == ('attempt_exception' if actor_fails else 'result')
    assert evidence._ACTIVE.get() is None
