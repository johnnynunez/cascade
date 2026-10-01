"""Fault injection changes one return value, never measurements or commands."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

DIAGNOSTICS = Path(__file__).resolve().parents[1] / "benchmark/diagnostics"
sys.path.insert(0, str(DIAGNOSTICS))
try:
    spec = importlib.util.spec_from_file_location("descent_probe", DIAGNOSTICS / "nvblox_descent_recovery.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
finally:
    sys.path.pop(0)


def runtime(result=True):
    calls = []
    def move(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    return SimpleNamespace(held_object=None, _held_provisional=None,
        arm=SimpleNamespace(move_joints=move, harness=SimpleNamespace(_grasp_exempt=(np.array([.2, .1]), .07, -.06)))), calls


def test_injection_preserves_commands_and_records_original_pregrasp_and_exemption():
    rt, calls = runtime()
    original = rt.arm.move_joints
    pre, target = np.array([1., 2., 3.]), np.array([4., 5., 6.])
    with probe.DescentSettleFault(rt) as fault:
        assert rt.arm.move_joints(pre) is True
        assert rt.arm.move_joints(target, bias_compensate=True) is False
        assert rt.arm.move_joints(pre) is True
    assert rt.arm.move_joints is original
    assert len(calls) == 3 and fault.evidence["actual_move_result"] is True
    np.testing.assert_array_equal(fault.evidence["pregrasp_q"], pre)
    np.testing.assert_array_equal(fault.retreat_calls[0]["target_q"], pre)
    assert len(fault.retreat_calls) == 1 and rt.held_object is None


@pytest.mark.parametrize("condition", ["real_failure", "held", "provisional", "no_descent"])
def test_unmet_injection_preconditions_never_create_synthetic_evidence(condition):
    rt, calls = runtime(result=condition != "real_failure")
    if condition == "held":
        rt.held_object = "orange"
    if condition == "provisional":
        rt._held_provisional = ("orange", "orange", "orange")
    with probe.DescentSettleFault(rt) as fault:
        result = rt.arm.move_joints(np.zeros(3), bias_compensate=condition != "no_descent")
    assert fault.evidence is None and len(calls) == 1
    assert result is (condition != "real_failure")


def test_actual_descent_exception_is_preserved_and_wrapper_restored():
    rt, _ = runtime()
    def refused(*args, **kwargs):
        raise RuntimeError("real transport failure")
    rt.arm.move_joints = refused
    with pytest.raises(RuntimeError, match="real transport failure"):
        with probe.DescentSettleFault(rt) as fault:
            rt.arm.move_joints(np.zeros(3), bias_compensate=True)
    assert rt.arm.move_joints is refused and fault.evidence is None
