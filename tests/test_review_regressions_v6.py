"""A legacy task_done claim must not erase earlier independently unknown effects.

CPU dispatcher/orchestrator controls; only actuator bodies are synthetic. The
pre-fix three-case reproduction remains in the external probe evidence.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from cascade.agent.effects import CONFIRMED, Postcondition, UNVERIFIED
from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.agent.reflex import ExperienceMemory, FastPlanner
from cascade.control.arm_rig import ArmRig
from cascade.types import SkillError
from test_review_regressions_v5 import completed_strokes, runtime as runtime


def agent(runtime, calls, *, reflex=False):
    return AgentOrchestrator(
        MockLLM([LLMResponse(tool_calls=[ToolCall(name, args)]) for name, args in calls]),
        runtime, fast_planner=FastPlanner() if reflex else None,
        decompose=False, attach_images=False, verify_milestones=False,
    )


def done(runtime):
    return runtime.execute('task_done', {'success': True, 'summary': 'Claimed complete'})


@pytest.mark.parametrize('via_reflex', [False, True])
def test_unverified_turn_cannot_be_declared_success(runtime, via_reflex):
    runtime.skill_turn_screw = completed_strokes
    calls = [] if via_reflex else [('turn_screw', {'label': 'nut', 'turns': 1.0})]
    calls.append(('task_done', {'success': True, 'summary': 'The nut is tightened.'}))
    report = agent(runtime, calls, reflex=via_reflex).run_task('tighten the nut one turn')
    rows = [json.loads(line) for line in runtime.trace._trace_path.read_text().splitlines()]
    screw = next(row['result'] for row in rows if row['skill'] == 'turn_screw')
    assert screw['execution_ok'] is True and screw['ok'] is False
    assert screw['postcondition']['status'] == 'unverified'
    assert report.success is False and len(report.unverified) == 1
    assert 'default:turn_screw' in report.unverified[0]
    assert 'success: False' in (runtime.trace.run_dir / 'summary.txt').read_text()
    assert rows[-1]['result']['success'] is False


def test_read_only_task_can_succeed_without_last_tool_physical_rule(runtime):
    report = agent(runtime, [
        ('count_objects', {}), ('task_done', {'success': True, 'summary': 'Count is zero.'}),
    ]).run_task('Count the visible objects')
    assert report.success is True and report.unverified == []


@pytest.mark.parametrize('mode', ['absent', 'no_verdict', 'crashed', 'wrong_skill', 'wrong_kind'])
def test_actor_confirmation_never_substitutes_for_matching_checker(runtime, mode):
    runtime.skill_move_home = lambda: {'ok': True, 'verified': True,
        'postcondition': {'skill': 'move_home', 'kind': 'at_home', 'status': CONFIRMED}}
    if mode == 'absent':
        runtime.effects = None
    elif mode == 'no_verdict':
        runtime.effects.verify = lambda *_a, **_k: None
    elif mode == 'crashed':
        def fail(*_a, **_k):
            raise RuntimeError('checker unavailable')
        runtime.effects.verify = fail
    else:
        runtime.effects.verify = lambda *_a, **_k: Postcondition(
            'close_gripper' if mode == 'wrong_skill' else 'move_home',
            'closed' if mode == 'wrong_kind' else 'at_home', CONFIRMED, 'unrelated confirmation')
    # The existing per-skill execution convention is retained.
    result = runtime.execute('move_home', {})
    assert result['ok'] is True
    assert done(runtime)['success'] is False
    assert runtime.task_effects()['effects'][0]['status'] == UNVERIFIED


@pytest.mark.parametrize('status', ['refuted', 'unverified', 'failed', 'uncertain'])
def test_later_read_or_other_confirmed_effect_never_repairs_debt(runtime, status):
    runtime.skill_push_object = lambda **_args: {'ok': status != 'failed', 'delivery_uncertain': status == 'uncertain'}
    runtime.skill_move_home = lambda: {}
    verify = runtime.effects.verify
    runtime.effects.verify = lambda name, *a, **k: (
        Postcondition(name, 'moved', status if status in ('refuted', 'unverified') else CONFIRMED, 'measured first effect')
        if name == 'push_object' else verify(name, *a, **k))
    runtime.execute('push_object', {'label': 'block'})
    runtime.execute('count_objects', {})
    runtime.execute('move_home', {})
    runtime.execute('count_objects', {})
    final = done(runtime)
    assert final['success'] is False and len(final['unverified']) == 1
    effects = final['task_effects']['effects']
    assert effects[0]['skill'] == 'push_object' and effects[0]['status'] != CONFIRMED
    assert effects[1]['skill'] == 'move_home' and effects[1]['status'] == CONFIRMED


def test_confirmed_multistep_can_finish_after_read_only_call(runtime):
    runtime.skill_move_home = lambda: {}
    runtime.skill_close_gripper = lambda: {}
    runtime.effects._gripper_frac = lambda: .5
    report = agent(runtime, [('move_home', {}), ('close_gripper', {}), ('count_objects', {}),
        ('task_done', {'success': True, 'summary': 'Finished observed effects'})]).run_task('Two effects, then count')
    assert report.success is True and report.unverified == []
    assert [r['status'] for r in runtime.task_effects()['effects']] == [CONFIRMED, CONFIRMED]


def test_task_boundary_is_host_only_and_does_not_reset_payload(runtime):
    runtime.skill_turn_screw = completed_strokes
    first = agent(runtime, [('turn_screw', {'label': 'nut'}),
        ('task_done', {'success': True, 'summary': 'claim'})]).run_task('first')
    assert first.success is False
    previous = runtime.task_effects()
    runtime.held_object = 'unresolved payload'
    runtime._carry_guard_token = sentinel = object()
    runtime.skill_reset_scene = lambda: {'ok': True}  # no world exists in this CPU fixture
    assert runtime.execute('begin_task', {})['ok'] is False
    runtime.execute('reset_scene', {})
    runtime.execute('task_done', {'success': False, 'summary': 'failed'})
    assert done(runtime)['success'] is False and runtime.task_effects()['task_id'] == previous['task_id']
    second = agent(runtime, [('count_objects', {}),
        ('task_done', {'success': True, 'summary': 'read only'})]).run_task('next user task')
    assert second.success is True and runtime.task_effects()['task_id'] != previous['task_id']
    assert runtime.held_object == 'unresolved payload' and runtime._carry_guard_token is sentinel
    assert previous['effects'][0]['status'] != CONFIRMED


def test_arm_identity_and_detached_records_cannot_erase_another_actor_failure(runtime):
    left, right = runtime._arm, SimpleNamespace()
    runtime.arm_rig = ArmRig([left, right], ['left', 'right'])
    runtime.skill_turn_screw = completed_strokes
    runtime.skill_move_home = lambda: {}
    result = runtime.execute('turn_screw', {'label': 'nut', 'arm': 'right'})
    result['ok'] = True; result['postcondition']['status'] = CONFIRMED
    snapshot = runtime.task_effects(); snapshot['effects'].clear()
    runtime.execute('move_home', {'arm': 'left'})
    assert done(runtime)['success'] is False
    records = runtime.task_effects()['effects']
    assert [(r['actor'], r['skill']) for r in records] == [('right', 'turn_screw'), ('left', 'move_home')]


def test_bad_actor_and_preparation_failure_are_not_lost(runtime):
    runtime.execute('move_home', {'arm': 'missing'})
    assert done(runtime)['success'] is False
    runtime.begin_task()
    runtime.effects.snapshot = lambda *_: (_ for _ in ()).throw(RuntimeError('pre-state unavailable'))
    with pytest.raises(RuntimeError, match='pre-state unavailable'):
        runtime.execute('move_home', {})
    assert done(runtime)['success'] is False
    assert runtime.task_effects()['effects'][0]['status'] == 'failed'
    runtime.begin_task()  # exception released active ownership; the next host task can begin


def test_pending_effect_blocks_completion_and_task_rebinding(runtime):
    entered, release = threading.Event(), threading.Event()
    def act():
        entered.set()
        assert release.wait(2), 'test owner did not release actor'
        return {}
    runtime.skill_move_home = act
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(runtime.execute('move_home', {})))
    worker.start()
    try:
        assert entered.wait(1)
        assert done(runtime)['success'] is False
        with pytest.raises(ValueError, match='active'):
            runtime.begin_task()
    finally:
        release.set(); worker.join(2)
    assert not worker.is_alive() and outcome[0]['ok'] is True
    assert done(runtime)['success'] is True


def test_unverified_fast_success_cannot_train_a_reflex(runtime):
    runtime.skill_move_home = lambda: {}
    runtime.effects.verify = lambda *_a, **_k: Postcondition('move_home', 'at_home', UNVERIFIED, 'reader unavailable')
    experience = ExperienceMemory()
    planner = FastPlanner(experience=experience)
    loop = agent(runtime, [('task_done', {'success': True, 'summary': 'claim'})])
    loop.fast_planner = planner
    report = loop.run_task('go home')
    assert report.success is False and report.unverified
    assert len(experience) == 0


def test_effect_execution_error_is_not_hidden_by_confirming_checker(runtime):
    def fail():
        raise SkillError('transport failed')
    runtime.skill_move_home = fail
    result = runtime.execute('move_home', {})
    assert result['postcondition']['status'] == CONFIRMED  # existing checker contract, preserved
    assert result['ok'] is False and done(runtime)['success'] is False


@pytest.mark.parametrize('cancel_phase', ['actor', 'trace'])
def test_interrupted_call_releases_pending_without_confirming_effect(runtime, cancel_phase):
    def cancel(*_args, **_kwargs):
        raise KeyboardInterrupt('cancelled by owner')
    runtime.skill_move_home = cancel if cancel_phase == 'actor' else lambda: {}
    if cancel_phase == 'trace':
        runtime.trace.record = cancel  # checker may confirm; call still never completed
    with pytest.raises(KeyboardInterrupt, match='cancelled by owner'):
        runtime.execute('move_home', {})
    snapshot = runtime.task_effects()
    assert snapshot['effects'][0]['status'] == 'failed'
    assert runtime.unverified_actions()
    runtime.begin_task()  # no active token leaked, and no implicit world reset occurred
    assert runtime.task_effects()['task_id'] != snapshot['task_id']


def test_tool_payload_cannot_rebind_task_identity(runtime):
    from cascade.skills.runtime import TOOL_SPECS
    assert not {'begin_task', 'task_effects', 'unverified_actions'} & {item['name'] for item in TOOL_SPECS}
    runtime.skill_turn_screw = completed_strokes
    runtime.execute('turn_screw', {'label': 'nut'})
    old = runtime.task_effects()
    result = runtime.execute('task_done', {'success': True, 'summary': 'claim', 'task_id': 'fresh'})
    assert result['ok'] is False
    assert runtime.task_effects() == old and done(runtime)['success'] is False


@pytest.mark.parametrize('malformed', [None, [], ['claimed success'], 'success', 1])
def test_malformed_result_releases_obligation_as_failed(malformed):
    from cascade.agent.task_effects import TaskEffects
    ledger = TaskEffects()
    row = ledger.begin(actor='arm', skill='move_home', kind='at_home')
    ledger.finish(row, malformed, Postcondition('move_home', 'at_home', CONFIRMED, 'checker claim'))
    assert ledger.snapshot()['effects'][0]['status'] == 'failed'
    assert ledger.unverified_actions()
    ledger.begin_task()
