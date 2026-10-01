"""No-transport tests for the combined control callback contract."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.types import SafetyViolation


class Raw:
    n_joints=6
    def __init__(self): self.reads=0; self.streams=[]
    def get_state(self, **kwargs):
        self.reads+=1
        return SimpleNamespace(q=np.zeros(6), kwargs=kwargs)
    def stream_to(self, q, duration, **kwargs):
        self.streams.append(kwargs)
        if kwargs.get('preflight'): kwargs['preflight'](np.zeros(6), duration)
        if kwargs.get('before_stream'): kwargs['before_stream']()
        if kwargs.get('feedback_guard'): kwargs['feedback_guard'](self.get_state())
        return True


def setup():
    raw=Raw()
    harness=SafetyHarness(SafetyLimits(np.full(3,-1.), np.ones(3), watchdog_s=100.), object())
    return SafeArm(raw,harness),raw,harness


@pytest.mark.parametrize('name',['preflight','before_stream'])
def test_ambiguous_callbacks_rejected_before_read_or_motion(name):
    arm,raw,harness=setup()
    with pytest.raises(SafetyViolation, match='ambiguous'):
        arm.move_joints(np.zeros(6), _preflight=lambda *a:None, **{name:lambda *a:None})
    assert raw.reads == 0 and raw.streams == [] and not harness._motion_active


def test_nv_preflight_and_spark_feedback_guard_both_preserved():
    arm,raw,harness=setup();events=[]
    assert arm.move_joints(np.zeros(6), _preflight=lambda *a: events.append('route'),
        feedback_guard=lambda state: events.append('feedback'), _halt_generation=0)
    # The pre-existing stretch read is now checked before any stream starts.
    assert events == ['feedback','route','feedback']
    assert raw.reads == 2
    assert len(raw.streams)==1 and not harness._motion_active


def test_spark_callbacks_forwarded_without_replacement():
    arm,raw,harness=setup();events=[]
    assert arm.move_joints(np.zeros(6), preflight=lambda *a:events.append('scene'),
        before_stream=lambda:events.append('cancel'), feedback_guard=lambda state:events.append('feedback'),
        _halt_generation=0)
    assert events == ['feedback','scene','cancel','feedback']
    assert raw.reads == 2


@pytest.mark.parametrize('name', ['preflight','before_stream','feedback_guard'])
def test_callback_typeerror_does_not_trigger_unprotected_retry(name):
    arm,raw,harness=setup()
    def failed(*args): raise TypeError('callback failure')
    with pytest.raises(SafetyViolation, match='no unguarded retry'):
        arm.move_joints(np.zeros(6), **{name:failed})
    assert len(raw.streams) == (0 if name == 'feedback_guard' else 1)
    assert not harness._motion_active


def test_state_timeout_forwarded():
    arm,raw,harness=setup()
    assert arm.get_state(timeout_s=.25).kwargs=={'timeout_s':.25}
