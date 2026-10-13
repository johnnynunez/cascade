from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.agent.reflex import parse_command
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError


@pytest.mark.parametrize('text,degrees', [
    ('rotate joint 7 90 degrees', 90),
    ('rotate joint 7 -9.5 degrees', -9.5),
    ('Please turn joint 7 by 10 degrees.', 10),
])
def test_exact_joint_command(text, degrees):
    assert parse_command(text).calls == [('rotate_joint', {'joint': 7, 'degrees': degrees})]


def runtime():
    r = SkillRuntime.__new__(SkillRuntime)
    q = np.zeros(7)
    r.arm = Mock(n_joints=7)
    r.arm.get_state.return_value = SimpleNamespace(q=q)
    r.arm.harness.limits.joint_margin = 0.02
    r.kin = SimpleNamespace(joint_limits=(-np.ones(7)*2, np.ones(7)*2))
    return r


def test_rotation_changes_only_selected_joint_through_safe_arm():
    r = runtime()
    r.skill_rotate_joint(7, 90)
    q = r.arm.move_joints.call_args.args[0]
    np.testing.assert_allclose(q, [0, 0, 0, 0, 0, 0, np.pi / 2])
    assert r.arm.move_joints.call_args.kwargs['duration_s'] >= 16


@pytest.mark.parametrize('joint,degrees', [(0, 10), (8, 10), (True, 10), (1.5, 10),
                                          (7, 91), (7, float('nan')), (7, 0)])
def test_invalid_rotation_never_commands_arm(joint, degrees):
    r = runtime()
    with pytest.raises(SkillError):
        r.skill_rotate_joint(joint, degrees)
    r.arm.move_joints.assert_not_called()


def test_limit_violation_never_commands_arm():
    r = runtime()
    r.arm.get_state.return_value.q[6] = 1.8
    with pytest.raises(SkillError, match='limits'):
        r.skill_rotate_joint(7, 90)
    r.arm.move_joints.assert_not_called()


def test_failed_relative_rotation_does_not_fall_back_to_llm():
    from cascade.agent.orchestrator import AgentOrchestrator
    import time
    a = AgentOrchestrator.__new__(AgentOrchestrator)
    a.fast_planner = Mock()
    from cascade.agent.reflex import FastPlan
    a.fast_planner.plan.return_value = FastPlan(
        calls=[('rotate_joint', {'joint': 7, 'degrees': 90})], source='reflex')
    a.runtime = Mock()
    a.runtime.execute.return_value = {'ok': False, 'error': 'did not settle'}
    report, fallback = a._try_fast_path('rotate joint 7 90 degrees', time.monotonic())
    assert report.success is False and fallback is None
    a.runtime.execute.assert_called_once()
