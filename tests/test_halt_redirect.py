"""Halt / redirect: abandoning an action that stopped being right.

VoLo names `monitor - halt - redirect` the core requirement of a physical
agent, because the world does not pause while the agent thinks. HumanCLAW puts
a verifier in front of the body for the same reason. cascade could only run a
skill to completion or latch an e-stop; there was no way to say "this motion is
wrong, stop it, I will reissue".

The distinction these tests protect: a halt must NOT behave like an e-stop.
An e-stop that clears itself is dangerous; a halt that latches bricks the arm
on its first use.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import MotionHalted, SafetyViolation


def _harness() -> SafetyHarness:
    h = SafetyHarness(SafetyLimits(
        workspace_min=np.array([-1.0, -1.0, -0.1]),
        workspace_max=np.array([1.0, 1.0, 1.0]),
    ))
    h.heartbeat()
    return h


def _wp(h: SafetyHarness) -> None:
    q = np.zeros(6)
    h.approve(q, q, dt=0.02)


def test_halt_stops_the_next_waypoint():
    h = _harness()
    _wp(h)                      # fine before
    h.halt("wrong object")
    with pytest.raises(MotionHalted):
        _wp(h)


def test_halt_is_a_safety_violation_subclass():
    """Existing abort paths catch SafetyViolation; they must keep working."""
    h = _harness()
    h.halt("superseded")
    with pytest.raises(SafetyViolation):
        _wp(h)


def test_halt_clears_when_the_next_motion_begins():
    """The recovery path. Forget this and the first halt bricks the arm."""
    h = _harness()
    h.halt("wrong object")
    with pytest.raises(MotionHalted):
        _wp(h)

    h.begin_motion()            # agent reissues a corrected command
    _wp(h)                      # no raise
    assert h.halted is None


def test_estop_does_not_clear_on_a_new_motion():
    """The asymmetry that makes halt safe to use.

    A halt is recoverable by design; an e-stop must survive until a human
    clears it explicitly. If begin_motion() cleared both, the e-stop would be
    worthless.
    """
    h = _harness()
    h.estop("pinch hazard")
    with pytest.raises(SafetyViolation):
        h.begin_motion()
    assert h.estopped is True


def test_halt_is_reported_and_clearable():
    h = _harness()
    assert h.halted is None
    h.halt("subgoal already satisfied")
    assert h.halted == "subgoal already satisfied"
    assert any("HALT" in v for v in h.violations)
    h.clear_halt()
    assert h.halted is None


def test_halt_does_not_power_down_the_arm():
    """A halted arm stays controllable: that is the whole point of redirect."""
    h = _harness()
    h.halt("changed my mind")
    h.begin_motion()
    _wp(h)
    assert h.estopped is False


def test_halt_skill_is_exposed_to_the_llm():
    """A skill missing from TOOL_SPECS is invisible and no test catches it.

    CLAUDE.md documents this trap explicitly, so it is pinned here.
    """
    from cascade.skills.runtime import TOOL_SPECS

    names = {t["name"] for t in TOOL_SPECS}
    assert "halt_motion" in names
