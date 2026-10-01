"""A failed descent must retain its original contact scope during withdrawal."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.arm_base import prepare_stream
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.skills import runtime as module
from cascade.types import Grasp, SafetyViolation, SkillError


HOME = np.array([.30, 0., .20])
PRE = np.array([.20, .10, .12])
GRASP = np.array([.20, .10, .02])


def fixture(monkeypatch, failure):
    class Kin:
        joint_limits = (np.full(3, -np.pi), np.full(3, np.pi))
        def fk(self, q):
            T = np.eye(4); T[:3, 3] = q
            return T
        def link_positions(self, q):
            return np.array([[0., 0., .3]])
    kin = Kin()
    limits = SafetyLimits(workspace_min=np.array([0., -.3, -.01]),
                    workspace_max=np.array([.5, .3, .55]), max_joint_vel=3.)
    harness = SafetyHarness(limits, kin)
    def clearance(points):
        return np.linalg.norm(points - GRASP, axis=1) - .025
    harness.occupancy = SimpleNamespace(clearance=clearance)
    if failure == "approach_blocked":
        midpoint = (HOME + PRE) / 2
        harness.limits.keep_out.append((midpoint - .01, midpoint + .01))
    moves, jaws, messages = [], [], []
    class Raw:
        n_joints = 3
        q = HOME.copy()
        def get_state(self):
            return SimpleNamespace(q=self.q.copy())
        def set_gripper(self, *args):
            jaws.append(args)
        def stream_to(self, target, duration, approve, **kwargs):
            prepare_stream(self, target, duration, kwargs.get("preflight"), kwargs.get("before_stream"))
            approve(self.q, target, duration)
            moves.append((target.copy(), harness._grasp_exempt))
            self.q = target.copy()
            if np.allclose(target, GRASP):
                if failure == "post_close":
                    return True
                if failure == "retreat_blocked":
                    harness.limits.keep_out.append((PRE - .01, PRE + .01))
                if failure == "estop":
                    harness.estop("injected stop")
                return False
            if failure == "post_close" and len(moves) > 3:
                raise SafetyViolation("injected lift refusal")
            return True
    arm = SafeArm(Raw(), harness)
    grasp = Grasp(GRASP.copy(), np.eye(3), .05, np.array([0., 0., -1.]))
    def select(*args, **kwargs):
        reason = kwargs["validate"](grasp, PRE, GRASP)
        if reason:
            raise SkillError(reason)
        return grasp, PRE, GRASP
    monkeypatch.setattr(module, "select_grasp", select)
    monkeypatch.setattr(module, "detection_color", lambda *args: "orange")
    rt = SimpleNamespace(
        cfg=Cfg({"arm": {"home_q": HOME.tolist()}, "grasp": {"exempt_radius_m": .07}}),
        arm=arm, kin=kin, held_object=None, _held_provisional=None, _max_width=.09,
        _grip_open=1., _reconcile_held=lambda: None,
        _plan_grasps=lambda *a, **kw: [grasp],
        memory=SimpleNamespace(add=lambda kind, text: messages.append(text)),
        grasp_memory=SimpleNamespace(record=lambda *a, **kw: None),
        _close_two_stage=lambda profile: jaws.append(("close",)),
    )
    def run():
        return module.SkillRuntime.skill_grasp_object(
            rt, "orange", _fix=SimpleNamespace(detection=SimpleNamespace(label="orange")),
            _frame=SimpleNamespace(rgb=np.zeros((1, 1, 3), dtype=np.uint8)))
    return rt, run, moves, jaws, messages


def test_failed_descent_withdraws_before_clearing_original_contact_zone(monkeypatch):
    rt, run, moves, jaws, messages = fixture(monkeypatch, "settle")
    with pytest.raises(SkillError, match="did not settle at grasp pose; pre-close retreat completed"):
        run()
    np.testing.assert_allclose(rt.arm.get_state().q, PRE)
    assert moves[-1][1] is moves[-2][1]
    assert rt.arm.harness._grasp_exempt is None
    assert len(jaws) == 1 and jaws[0][0] == 1.
    assert rt._held_provisional is None and messages


def test_estop_never_attempts_withdrawal(monkeypatch):
    rt, run, moves, jaws, _ = fixture(monkeypatch, "estop")
    with pytest.raises(SkillError, match="did not settle at grasp pose"):
        run()
    np.testing.assert_allclose(moves[-1][0], GRASP)
    assert len(moves) == 3 and len(jaws) == 1
    assert rt.arm.harness.estopped and rt.arm.harness._grasp_exempt is None


def test_blocked_withdrawal_preserves_original_failure_without_actuation(monkeypatch):
    rt, run, moves, jaws, _ = fixture(monkeypatch, "retreat_blocked")
    with pytest.raises(SkillError, match="did not settle at grasp pose; pre-close retreat refused"):
        run()
    np.testing.assert_allclose(rt.arm.get_state().q, GRASP)
    assert len(moves) == 3 and len(jaws) == 1
    assert rt.arm.harness._grasp_exempt is None


def test_safe_endpoints_do_not_authorize_blocked_approach_chord(monkeypatch):
    rt, run, moves, jaws, _ = fixture(monkeypatch, "approach_blocked")
    with pytest.raises(SkillError, match="approach unsafe"):
        run()
    assert not moves and not jaws


def test_possible_payload_is_preserved_without_empty_tool_retreat(monkeypatch):
    rt, run, moves, jaws, _ = fixture(monkeypatch, "post_close")
    with pytest.raises(SafetyViolation, match="injected lift refusal"):
        run()
    assert len(moves) == 4 and len(jaws) == 2
    assert rt._held_provisional[0] == "orange"
    assert rt.arm.harness._grasp_exempt is None
