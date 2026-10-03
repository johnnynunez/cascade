"""Task reports and composed cancellation retain unresolved action obligations.

Actuator methods are synthetic. Dispatcher, ledger, adapters, coordinator,
planner, trace and completion paths are the production implementations.
"""
import asyncio
import json
from dataclasses import asdict

import pytest

from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.agent.reflex import FastPlanner
from cascade.apps.robot_runtime import DomainAdapter
from cascade.robotics.contracts import ResourceDescriptor
from cascade.robotics.runtime import RobotRuntime
from cascade.skills.runtime import TOOL_SPECS
from test_review_regressions_v5 import completed_strokes, runtime as runtime


def retain(directory, name, data):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')


@pytest.mark.parametrize('reflex', [False, True])
def test_exhausted_report_preserves_outstanding_effect(runtime, reflex, tmp_path):
    runtime.skill_turn_screw = completed_strokes
    responses = [] if reflex else [LLMResponse(tool_calls=[ToolCall('turn_screw', {'label': 'nut'})])]
    agent = AgentOrchestrator(
        MockLLM(responses), runtime, fast_planner=FastPlanner() if reflex else None,
        max_steps=1, decompose=False, attach_images=False, verify_milestones=False,
    )
    report = agent.run_task('tighten the nut one turn')
    outstanding = runtime.unverified_actions()
    retain(tmp_path, 'budget-reflex-' + str(reflex), {
        'report': asdict(report), 'outstanding': outstanding,
        'ledger': runtime.task_effects(),
        'summary': (runtime.trace.run_dir / 'summary.txt').read_text(),
        'trace': [json.loads(line) for line in runtime.trace._trace_path.read_text().splitlines()],
    })
    assert not report.success and outstanding
    assert report.unverified == outstanding
    assert len(report.tool_log) == 1 and report.tool_log[0]['tool'] == 'turn_screw'
    assert report.tool_log[0]['tier'] == ('reflex' if reflex else 'llm')
    summary = (runtime.trace.run_dir / 'summary.txt').read_text()
    assert all(item in report.summary and item in summary for item in outstanding)
    saved = asdict(report)
    # A trusted new task cannot mutate the already returned report/history.
    agent.fast_planner = None
    agent.llm = MockLLM([LLMResponse(tool_calls=[
        ToolCall('task_done', {'success': True, 'summary': 'no action requested'})])])
    next_report = agent.run_task('answer a read-only question')
    assert next_report.success and not next_report.unverified
    assert [row['tool'] for row in next_report.tool_log] == ['task_done']
    assert asdict(report) == saved


@pytest.mark.parametrize('failure', ['cancel', 'exception', 'normal'])
def test_composed_completion_respects_child_cancellation(runtime, failure, tmp_path):
    """The resource is explicitly synthetic, so even normal completion cannot
    prove a physical task. No SDK/backend is constructed by this probe.
    """
    resource = ResourceDescriptor('manip/arm', 'arm', 'cpu_fixture',
                                  capabilities=('joint_motion',),
                                  controller_id='mock:cpu', writer_id='manip/arm',
                                  synthetic=True, admission='software_only')
    spec = next(spec for spec in TOOL_SPECS if spec['name'] == 'move_home')
    domain = DomainAdapter('manip', {'kind': 'manipulation'}, (resource,), [spec],
                           {'move_home'}, runtime=runtime)
    composed = RobotRuntime({'manip': domain}, memory=runtime.memory, trace=runtime.trace)
    composed.begin_task()

    def act():
        if failure == 'cancel':
            raise KeyboardInterrupt('owner cancelled')
        if failure == 'exception':
            raise RuntimeError('actuator failed')
        return {}

    runtime.skill_move_home = act
    try:
        action = composed.execute('manip.move_home', {})
    except KeyboardInterrupt as exc:
        action = {'exception': type(exc).__name__, 'message': str(exc)}
    child_done = runtime.execute('task_done', {'success': True, 'summary': 'claim'})
    parent_done = composed.execute('task_done', {'success': True, 'summary': 'claim'})
    retain(tmp_path, 'composed-' + failure, {
        'action': action, 'child_done': child_done, 'parent_done': parent_done,
        'parent_unverified': composed.unverified_actions(),
    })
    assert parent_done['success'] is False


@pytest.mark.parametrize('terminal', ['done', 'budget'])
def test_failed_fast_path_then_read_retains_both_tiers(runtime, terminal):
    runtime.skill_turn_screw = completed_strokes
    runtime.skill_read_current = lambda: {'ok': True, 'value': 'observed'}
    responses = [LLMResponse(tool_calls=[ToolCall('read_current', {})])]
    if terminal == 'done':
        responses.append(LLMResponse(tool_calls=[ToolCall('task_done', {'success': True, 'summary': 'done'})]))
    agent = AgentOrchestrator(MockLLM(responses), runtime, fast_planner=FastPlanner(),
                              max_steps=len(responses), decompose=False, attach_images=False,
                              verify_milestones=False)
    report = agent.run_task('tighten the nut one turn')
    assert not report.success
    assert report.unverified == runtime.unverified_actions()
    assert [row['tier'] for row in report.tool_log] == ['reflex'] + ['llm'] * len(responses)
    assert [row['tool'] for row in report.tool_log][:2] == ['turn_screw', 'read_current']


@pytest.mark.parametrize('failure_type', [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_composed_cancellation_propagates_and_cannot_be_repaired_by_later_motion(failure_type):
    from test_robot_runtime import Domain
    domain = Domain(synthetic=False)
    runtime = RobotRuntime({domain.domain_id: domain})
    original = domain.execute
    def cancel(*_args):
        raise failure_type('owner interrupted')
    domain.execute = cancel
    try:
        runtime.begin_task()
        task_id = runtime._task_id
        with pytest.raises(failure_type, match='owner interrupted'):
            runtime.execute('locomotion.move', {'distance': .1})
        assert not runtime._active and runtime._drained.is_set()
        assert runtime.unverified_actions() == ['locomotion.move']
        assert runtime._task_id == task_id
        domain.execute = original
        assert runtime.execute('locomotion.move', {'distance': .1})['ok']
        done = runtime.execute('task_done', {'success': True, 'summary': 'later action completed'})
        assert not done['success'] and done['unverified'] == ['locomotion.move']
        # Only the trusted host can delimit a new task, without replay/reset.
        runtime.begin_task()
        assert runtime._task_id != task_id
        assert runtime.execute('task_done', {'success': True, 'summary': 'read-only'})['success']
        assert len(domain.calls) == 1
    finally:
        assert runtime.close()['ok']


def test_cancelled_read_does_not_invent_a_physical_obligation():
    from test_robot_runtime import Domain
    from cascade.robotics.contracts import ToolDescriptor
    domain = Domain()
    domain.tool_descriptors = (ToolDescriptor('locomotion.read', 'read existing state',
        {'type': 'object', 'properties': {}, 'additionalProperties': False},
        'locomotion', 'read'),)
    def cancel(*_args):
        raise asyncio.CancelledError('reader interrupted')
    domain.execute = cancel
    runtime = RobotRuntime({domain.domain_id: domain})
    try:
        with pytest.raises(asyncio.CancelledError, match='reader interrupted'):
            runtime.execute('locomotion.read', {})
        assert runtime.unverified_actions() == []
        assert runtime.execute('task_done', {'success': True, 'summary': 'no motion'})['success']
    finally:
        assert runtime.close()['ok']


@pytest.mark.parametrize('failure_type', [KeyboardInterrupt, asyncio.CancelledError])
def test_orchestrator_propagates_action_cancellation_without_erasing_debt(runtime, failure_type):
    def cancel(**_args):
        raise failure_type('actuator interrupted')
    runtime.skill_turn_screw = cancel
    agent = AgentOrchestrator(MockLLM([LLMResponse(tool_calls=[ToolCall('turn_screw', {})])]),
        runtime, max_steps=1, decompose=False, attach_images=False, verify_milestones=False)
    with pytest.raises(failure_type, match='actuator interrupted'):
        agent.run_task('turn the screw')
    assert runtime.current_tier is None
    outstanding = runtime.unverified_actions()
    assert outstanding and 'turn_screw' in outstanding[0]
    done = runtime.execute('task_done', {'success': True, 'summary': 'claim after cancellation'})
    assert not done['success'] and done['unverified'] == outstanding
