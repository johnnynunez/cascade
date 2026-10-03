"""Do not lift while rate-limited jaws are still closing under render load."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.grasping.force import select_profile
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError


def runtime(monkeypatch, feedback, timeout=8.):
    clock = [0.]
    monkeypatch.setattr('cascade.skills.runtime.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('cascade.skills.runtime.time.sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = SimpleNamespace(grasp={'close_settle_s': .4, 'close_feedback_timeout_s': timeout})
    rt._grip_open, rt._grip_closed = 1., 0.
    commands = []
    # A SafeArm always supplies this guard; an idle harness has no retained
    # model-withdrawal obligation. The jaw feedback below remains scripted.
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.), np.full(3, 1.)))
    rt.arm = SimpleNamespace(raw=SimpleNamespace(), harness=harness,
                             set_gripper=lambda p, **kw: commands.append((clock[0], p)))
    rt._gripper_width_frac = lambda: feedback(clock[0], commands)
    return rt, clock, commands


def test_each_stage_waits_for_travel_then_object_stall(monkeypatch):
    # First stage takes two seconds, final stage meets a 35%-width object.
    def width(t, commands):
        start, goal = commands[-1]
        initial = 1. if len(commands) == 1 else .5
        return max(.35, goal, initial - .25 * (t - start))

    rt, clock, commands = runtime(monkeypatch, width)
    rt._close_two_stage(select_profile('cube'))
    assert [p for _, p in commands] == pytest.approx([.5, .15])
    assert commands[1][0] >= 1.9
    assert clock[0] >= commands[1][0] + 1.1


@pytest.mark.parametrize('feedback', [lambda t, c: None, lambda t, c: 1., lambda t, c: 1. - .02 * t])
def test_missing_stuck_open_or_still_moving_feedback_refuses_lift(monkeypatch, feedback):
    rt, _, commands = runtime(monkeypatch, feedback, timeout=1.)
    with pytest.raises(SkillError, match='refusing to lift'):
        rt._close_two_stage(select_profile('cube'))
    assert len(commands) == 1


@pytest.mark.parametrize('timeout', [0., -1., float('inf'), float('nan')])
def test_invalid_timeout_refuses_before_closing(monkeypatch, timeout):
    rt, _, commands = runtime(monkeypatch, lambda t, c: .4, timeout=timeout)
    with pytest.raises(SkillError, match='finite and positive'):
        rt._close_two_stage(select_profile('cube'))
    assert not commands
