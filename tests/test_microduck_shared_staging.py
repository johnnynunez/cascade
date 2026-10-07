"""Speculation is reversible; an admitted shared solve is not rolled back."""
from concurrent.futures import ThreadPoolExecutor

from cascade.sim.microduck_policy_admission import target_contract

import numpy as np
import pytest
from test_microduck_shared_scene import shared
from test_microduck_stepper import command


def ready(count=2):
    fleet, owner, steppers = shared(count)
    fleet.start()
    fleet.tick()
    for s in steppers:
        command(s.controller)
        s.policy.preview = lambda obs: np.full(14, obs[0, 48], np.float32)
    for _ in range(3):
        fleet.tick()
    return fleet, owner, steppers


@pytest.mark.parametrize('count, interrupted', [(1,0), (2,0), (2,1), (12,0), (12,5), (12,11)])
def test_stop_crossing_each_preview_discards_whole_cohort_without_advancing_histories(count, interrupted):
    fleet, owner, steppers = ready(count)
    native_step = owner.step_count
    before = len(owner.events)
    original = steppers[interrupted].policy.preview
    crossed = False
    with ThreadPoolExecutor(1) as pool:
        def preview(obs):
            nonlocal crossed
            # Every speculative attempt sees only last committed history.
            assert all(s.policy_commits == 1 and s.steps == 4 for s in steppers)
            assert all(len(s.actuator.targets) == 1 for s in steppers)
            assert len(owner.events) == before
            assert not obs[0, 34:48].any()
            if not crossed:
                crossed = True
                # Stop must not wait for peer ONNX under permission locks.
                assert pool.submit(steppers[0].controller.stop).result(timeout=1)['cancelled']
            return original(obs)
        steppers[interrupted].policy.preview = preview
        result = fleet.tick()
    assert owner.step_count == native_step + 1 and not fleet.failure
    assert len(owner.events) == before + count + 1  # one BAM per robot, one solve
    for i, s in enumerate(steppers):
        assert s.policy_commits == 2 and s.policy_attempts == 3 and s.policy_evaluations == 3
        assert [record['status'] for record in s.policy_records] == ['discarded', 'evaluated']
        assert [record['retry'] for record in s.policy_records] == [0, 1]
        assert {record['policy_slot'] for record in s.policy_records} == {1}
        assert not s.policy_records[0]['committed'] and s.policy_records[1]['committed']
        assert s.policy_records[1]['first_step_after_commit'] == native_step + 1
        np.testing.assert_array_equal(s.policy.previous_action, np.full(14, 0. if i == 0 else .2, np.float32))
        assert result[s.identity['robot_id']]['permission_generation_at_sample'] == (2 if i == 0 else 1)


@pytest.mark.parametrize('interrupted', [0, 1])
def test_stop_during_bam_is_after_fence_and_never_rewinds_native_delay(interrupted):
    fleet, owner, steppers = ready()
    original = steppers[interrupted].actuator.before_step
    crossed = False
    before = owner.step_count
    with ThreadPoolExecutor(1) as pool:
        def bam(dt):
            nonlocal crossed
            assert all(s.policy_commits == 2 for s in steppers)
            if not crossed:
                crossed = True
                assert pool.submit(steppers[0].controller.stop).result(timeout=1)['cancelled']
            original(dt)
        steppers[interrupted].actuator.before_step = bam
        result = fleet.tick()
    steppers[interrupted].actuator.before_step = original
    assert owner.step_count == before + 1 and not fleet.failure
    assert result['duck0']['permission_generation_at_sample'] == 1
    assert result['duck0']['policy_target_generation'] == 1
    assert steppers[0].controller.hello()['generation'] == 2
    for _ in range(3):
        held = fleet.tick()['duck0']
        assert held['policy_target_held'] and held['policy_target_generation'] == 1
        np.testing.assert_array_equal(steppers[0].policy.previous_action, np.full(14, .2, np.float32))
    refreshed = fleet.tick()['duck0']
    assert not refreshed['policy_target_held'] and refreshed['policy_target_generation'] == 2
    assert not steppers[0].policy.previous_action.any()
    np.testing.assert_array_equal(steppers[1].policy.previous_action, np.full(14, .2, np.float32))
    assert all(s.policy_commits == 3 for s in steppers)
    assert len([e for e in owner.events if e == ('before', before)]) == 2


def test_withheld_non_policy_tick_changes_no_target_delay_state_or_publication():
    fleet, owner, steppers = shared()
    fleet.start(); fleet.tick()
    original = steppers[1]._stage_tick
    def stage(**kwargs):
        result = original(**kwargs)
        steppers[0].controller.stop()
        return result
    steppers[1]._stage_tick = stage
    before = owner.step_count, list(owner.events)
    assert fleet.tick() is None and fleet.withheld_ticks == 1
    assert (owner.step_count, owner.events) == before
    assert all(s.steps == 1 and s.policy_commits == 1 and len(s.actuator.targets) == 1 for s in steppers)
    assert all(s.controller.state()['state']['step'] == before[0] for s in steppers)
    assert all(s.policy_records == [] for s in steppers)
    steppers[1]._stage_tick = original
    assert fleet.tick()['duck0']['step'] == before[0] + 1


def test_irreversible_partial_history_commit_faults_every_robot_without_native_retry():
    fleet, owner, steppers = ready()
    before = owner.step_count
    original = steppers[1].policy.commit
    def failed_copy(action):
        original(action)
        raise RuntimeError('interrupted history copy')
    steppers[1].policy.commit = failed_copy
    with pytest.raises(RuntimeError, match='history copy'):
        fleet.tick()
    assert owner.step_count == before and all(s.failure for s in steppers)
    assert steppers[0].policy_records[-1]['committed'] is True
    assert steppers[1].policy_records[-1]['committed'] is None
    assert all(len(s.actuator.targets) == 1 for s in steppers)
    with pytest.raises(RuntimeError, match='not running'):
        fleet.tick()


def test_shared_clock_is_rechecked_after_last_preview_before_any_history_copy():
    fleet, owner, steppers = ready()
    original = steppers[-1]._stage_tick
    def changed(**kwargs):
        result = original(**kwargs)
        owner.step_count += 1  # an unowned native clock advance, after valid preview
        return result
    steppers[-1]._stage_tick = changed
    with pytest.raises(RuntimeError, match='clock changed before cohort'):
        fleet.tick()
    assert all(s.policy_commits == 1 and len(s.actuator.targets) == 1 for s in steppers)
    assert all(s.policy_records[-1]['status'] == 'discarded' for s in steppers)


def test_retry_still_checks_episode_wall_budget_and_never_commits_late_policy():
    fleet, owner, steppers = ready()
    original = steppers[1].policy.preview
    def delayed(obs):
        action = original(obs)
        steppers[0].controller.stop()
        steppers[1].clock = lambda: 200.
        return action
    steppers[1].policy.preview = delayed
    before = owner.step_count
    with pytest.raises(RuntimeError, match='wall duration'):
        fleet.tick()
    assert owner.step_count == before and all(s.policy_commits == 1 for s in steppers)
    assert all(len(s.actuator.targets) == 1 for s in steppers)


def _launch_shared_runner(tmp_path, monkeypatch, *, profile_phases, gc_policy, fleet_close_error=None,
                          extra_limits=None, physics_row_every=1, max_steps=2, profile_cprofile=False):
    """Drive the real launcher with software fixtures; returns everything the asserts need."""
    import gc
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    from test_heap_freeze import fake_collector
    from test_microduck_bridge_cli import software_limits
    from test_microduck_shared_scene import CheckedActuator, View, layout_fixture
    from test_microduck_stepper import SoftwareBackend, SoftwarePolicy, render_times
    from cascade.sim import heap_freeze, microduck_shared as shared_module, microduck_shared_native as native
    from cascade.control import microduck_policy

    runner = importlib.import_module('isaac_microduck_shared')
    callbacks_before = tuple(gc.callbacks)
    gc_settings = gc.isenabled(), gc.get_threshold()
    created, policy_calls, initial_steps = [], [], []
    collector = fake_collector(foreign_frozen=375)
    class RecordedFreeze(heap_freeze.StartupHeapFreeze):
        # Injected collector: the test interpreter's heap is never frozen. Absolute
        # freeze counts are not an invariant anyway (CPython parks immortals on
        # every full collection); record the policy's own calls against the owner's
        # step count instead.
        def __init__(self, **kwargs):
            super().__init__(collector=collector, **kwargs)
        def apply(self):
            policy_calls.append(('apply', created[0].step_count))
            return super().apply()
        def release(self, **kwargs):
            policy_calls.append(('release', created[0].step_count))
            return super().release(**kwargs)
    monkeypatch.setattr(heap_freeze, 'StartupHeapFreeze', RecordedFreeze)
    class Backend(SoftwareBackend):
        def __init__(self, *unused):
            super().__init__()
            self.layout = shared_module.bind_scene(**layout_fixture(1))
            # Offers the host snapshot and the cohort output check like the real adapter, so the
            # launcher's profiled attempt has to show the single verify between BAM and solve.
            self.actuators = [CheckedActuator(self)]
            binding = self.layout.robots[0]
            self.actuators[0].coordinate_indices = binding.q_indices, binding.dof_indices
            self.receipt = {'software_fixture': True, 'configuration': {}}
            created.append(self)
        def open(self): pass
        def _read_completed_scene(self): raise AssertionError('fixture does not use native capture')
        def bind_identity(self, **unused):
            assert ('phase_profile' in self.receipt['configuration']) == profile_phases
            assert self.receipt['configuration'].get('gc_policy') == gc_policy
            assert policy_calls == []  # identity binds before any freeze
            initial_steps.append(self.step_count)
            return {'scene_model_sha256': 'a'*64}
        def capture(self):
            assert len(gc.callbacks) == len(callbacks_before) + int(profile_phases)
            assert [c[0] for c in policy_calls] == (['apply'] if gc_policy else [])
            return dict(rgb=np.zeros((8, 8, 3), np.uint8), step=self.step_count,
                        sim_time_s=self.sim_time, captured_at=0., render_times=render_times(self.sim_time))
        def support_probe(self):
            return dict(passed=True, step=self.step_count, sim_time_s=self.sim_time)
        def shutdown(self, code):
            self.shutdown_code = code
            # The cProfile diagnostic must be on disk before the SDK shutdown, which
            # does not return in the real launcher.
            assert (out / 'owner.prof').exists() == profile_cprofile
            return True
    class Interrupted(shared_module.SharedMicroduckStepper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            s = self.steppers[0]
            def preview(obs):
                if s.policy_evaluations < 2:
                    s.controller.stop()
                return np.zeros(14, np.float32)
            s.policy.preview = preview
        def close(self):
            try:
                return super().close()
            finally:
                if fleet_close_error is not None:
                    raise fleet_close_error
    monkeypatch.setattr(native, 'SharedKitNewtonBackend', Backend)
    monkeypatch.setattr(native, 'SharedRobotView', View)
    monkeypatch.setattr(microduck_policy, 'MicroduckPolicy', lambda *args, **kwargs: SoftwarePolicy(created[0]))
    monkeypatch.setattr(shared_module, 'SharedMicroduckStepper', Interrupted)
    out = tmp_path / 'run'
    args = NS(out=out, device='cuda:0', source='software-test-not-physics', max_wall_s=3.,
              max_steps=max_steps, camera_every=1, max_jpeg_bytes=100000, policy=tmp_path / 'fixture.onnx',
              policy_sha256='b'*64, target_profile='direct-v1', python_extra_path=[], robots=1, serve_base_port=None,
              profile_phases=profile_phases, gc_policy=gc_policy, physics_row_every=physics_row_every,
              profile_cprofile=profile_cprofile)
    signals = NS(signum=None, registration_attempts=0, checkpoint=lambda **kwargs: None, defer=nullcontext)
    admission = dict(target_contract=target_contract('b'*64, 'direct-v1'), asset_sha256='a'*64,
                     limits=software_limits() | (extra_limits or {}), experience_text='software fixture\n')
    result = runner.run(args, admission, signals)
    assert tuple(gc.callbacks) == callbacks_before
    assert (gc.isenabled(), gc.get_threshold()) == gc_settings
    return NS(result=result, out=out, created=created, policy_calls=policy_calls, initial_steps=initial_steps,
              collector=collector)


@pytest.mark.parametrize('profile_phases, gc_policy', [(False, None), (True, None), (True, 'freeze-startup-heap'),
                                                       (False, 'freeze-startup-heap')])
def test_launcher_withheld_attempt_preserves_step_budget_and_only_emits_completed_frames(tmp_path, monkeypatch, profile_phases, gc_policy):
    import json
    run = _launch_shared_runner(tmp_path, monkeypatch, profile_phases=profile_phases, gc_policy=gc_policy)
    result, out, created, policy_calls = run.result, run.out, run.created, run.policy_calls
    assert result['completed'] and result['steps'] == 2 and result['withheld_ticks'] == 1, result
    # One software robot: the owner keeps the per-adapter actuation and says so in the receipt.
    assert result['bam_cohort'] == {'path': 'per-adapter', 'adapters': 1, 'reason': 'single robot'}
    assert created[0].closed == 1 and created[0].shutdown_code == 0
    def rows(name): return [json.loads(line) for line in (out / name).read_text().splitlines()]
    for name in ('physics.jsonl', 'frames.jsonl', 'support-probe.jsonl'):
        assert [row['step'] for row in rows(name)] == [3, 4]
    assert [row['status'] for row in rows('policy.jsonl')] == ['discarded', 'discarded', 'evaluated']
    assert result['robots']['duck0']['policy_commits'] == 1
    if gc_policy is None:
        assert policy_calls == [] and run.collector.calls == [] and 'gc_policy' not in result
        assert 'gc_policy' not in created[0].receipt and 'gc_policy' not in created[0].receipt['configuration']
    else:
        # Applied before this run's first solve, released after its last one; nothing in between.
        assert created[0].step_count == run.initial_steps[0] + result['steps'] > run.initial_steps[0]
        assert policy_calls == [('apply', run.initial_steps[0]), ('release', created[0].step_count)]
        assert run.collector.calls == [('collect', 2), ('freeze',), ('collect', 2), ('unfreeze',), ('collect', 2)]  # incl. the measured frozen collection
        applied, released = result['gc_policy']['applied'], result['gc_policy']['released']
        assert applied is created[0].receipt['gc_policy'] and applied['policy'] == 'freeze-startup-heap-v1'
        assert applied['freeze'] == dict(duration_ns=applied['freeze']['duration_ns'], frozen=1200, foreign_frozen_before=375)
        assert released['unfrozen'] == 1575 and released['policy_frozen'] == 1200 and released['apply_receipt_complete']
        assert created[0].receipt['configuration'] == ({'phase_profile': 'owner-thread-inclusive-gc-trigger-v1'} if profile_phases else {}) | {'gc_policy': 'freeze-startup-heap'}
        runtime = json.loads((out / 'runtime.json').read_text())
        assert runtime['backend']['gc_policy']['implementation_sha256'] == applied['implementation_sha256']
    if profile_phases:
        attempts, footer = rows('timing.jsonl')[:-1], rows('timing.jsonl')[-1]
        assert [r['outcome'] for r in attempts] == ['withheld', 'solved', 'solved']
        assert [r['clock_after'][0] for r in attempts] == [2, 3, 4]
        assert footer['attempts'] == 3 and not footer['errors']
        assert footer['gc_summary']['accounting_complete']
        assert result['phase_profile'] == {'file': 'timing.jsonl', 'attempts': 3, 'errors': []}
        phases = {s['phase'] for r in attempts for s in r['spans']}
        assert {'policy.prepare', 'bam.before_step', 'bam.verify', 'solve', 'publication', 'record.physics',
                'write.physics.jsonl', 'camera.overview', 'camera.capture', 'support.probe'} <= phases
        for attempt in attempts[1:]:
            # One cohort verify per solved attempt, after the last bam.before_step and before the solve.
            order = [s['phase'] for s in attempt['spans'] if s['phase'] in ('bam.before_step', 'bam.verify', 'solve')]
            assert order == ['bam.before_step', 'bam.verify', 'solve']
        assert not any(s['phase'] in ('solve', 'publication') for s in attempts[0]['spans'])
        assert created[0].receipt['configuration']['phase_profile'] == 'owner-thread-inclusive-gc-trigger-v1'
    else:
        assert not (out/'timing.jsonl').exists() and 'phase_profile' not in result
        assert 'step' not in vars(created[0]) and 'capture' not in vars(created[0])
        assert 'phase_profile' not in created[0].receipt['configuration']


def test_launcher_releases_the_frozen_heap_even_when_fleet_closure_fails(tmp_path, monkeypatch):
    run = _launch_shared_runner(tmp_path, monkeypatch, profile_phases=False, gc_policy='freeze-startup-heap',
                                fleet_close_error=RuntimeError('synthetic shared teardown fault'))
    result = run.result
    assert not result['completed'] and result['exit_code'] == 1 and 'RuntimeError' in result['teardown_errors']
    assert run.collector.calls == [('collect', 2), ('freeze',), ('collect', 2), ('unfreeze',), ('collect', 2)]
    assert [c[0] for c in run.policy_calls] == ['apply', 'release']
    assert result['gc_policy']['released']['unfrozen'] == 1575 and result['gc_policy']['applied'] is not None


@pytest.mark.parametrize('gains', [None, {'heading_hold_kp': 4.0, 'heading_hold_ki': 2.0}])
def test_launcher_gives_every_shared_controller_the_admitted_heading_hold(tmp_path, monkeypatch, gains):
    import json
    run = _launch_shared_runner(tmp_path, monkeypatch, profile_phases=False, gc_policy=None, extra_limits=gains)
    assert run.result['completed']
    robots = json.loads((run.out / 'runtime.json').read_text())['robots']
    assert robots and all(hello['heading_hold'] == (None if gains is None else {'kp': 4.0, 'ki': 2.0})
                          for hello in robots.values())


def test_launcher_thins_physics_rows_only_when_asked_and_keeps_first_and_last(tmp_path, monkeypatch):
    import json
    run = _launch_shared_runner(tmp_path, monkeypatch, profile_phases=False, gc_policy=None,
                                physics_row_every=3, max_steps=7)
    assert run.result['completed'] and run.result['steps'] == 7
    steps = [json.loads(line)['step'] for line in (run.out / 'physics.jsonl').read_text().splitlines()]
    # solved steps are 3..9 (the first attempt is withheld): first, every third offset by one
    # (i+1+1) % 3 == 0 -> i = 1, 4 -> steps 4, 7), and the last
    assert steps == [3, 4, 7, 9]
    assert run.created[0].receipt['configuration'] == {'physics_row_every': 3}
    frames = [json.loads(line)['step'] for line in (run.out / 'frames.jsonl').read_text().splitlines()]
    assert frames == list(range(3, 10))  # camera cadence is unchanged


@pytest.mark.parametrize('profile_cprofile', [False, True])
def test_launcher_cprofile_diagnostic_is_dumped_before_shutdown_and_off_by_default(tmp_path, monkeypatch, profile_cprofile):
    import pstats
    run = _launch_shared_runner(tmp_path, monkeypatch, profile_phases=True, gc_policy=None,
                                profile_cprofile=profile_cprofile)
    assert run.result['completed'] and run.created[0].shutdown_code == 0
    if not profile_cprofile:
        assert 'cprofile' not in run.result and not (run.out / 'owner.prof').exists()
        return
    assert run.result['cprofile'] == {'file': 'owner.prof'}
    stats = pstats.Stats(str(run.out / 'owner.prof'))
    assert stats.total_tt > 0
    assert any(path.endswith('microduck_shared.py') and name == 'tick' for (path, _, name) in stats.stats)
