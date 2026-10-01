"""Observed fingers retain current mapping and contact-episode guards."""
import pytest

from cascade.types import SafetyViolation
from cascade.grasping import observed_scene
from cascade.skills import contact_episode
from test_observed_finger_runtime import runtime, fake_gate, real_harness


def test_map_change_after_selection_rejects_first_joint_target(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    harness = real_harness(rt)
    opened = False
    original = rt.arm.raw.set_gripper

    def opening(*args, **kwargs):
        nonlocal opened
        original(*args, **kwargs)
        opened = True

    monkeypatch.setattr(rt.arm.raw, "set_gripper", opening)
    harness.vet_step = lambda *a, **k: "new mapped obstacle" if opened else None
    with pytest.raises(SafetyViolation, match="route became unsafe: new mapped obstacle"):
        rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    assert [c for c in calls if c[0] == "gripper"] == [("gripper", 1., .8)]
    assert not [c for c in calls if c[0] == "joints"]


def test_stream_start_guard_is_rechecked_after_observed_preflight(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    harness = real_harness(rt)
    checked = False
    factory = observed_scene.for_runtime
    original = harness.check_stream_start

    def make(*args):
        gate = factory(*args)
        def preflight(*a, **k):
            nonlocal checked
            checked = True
        gate.require_profile = preflight
        return gate

    def live_guard(*args, **kwargs):
        if checked:
            raise SafetyViolation("perception expired during preflight")
        return original(*args, **kwargs)

    monkeypatch.setattr(observed_scene, "for_runtime", make)
    harness.check_stream_start = live_guard
    with pytest.raises(SafetyViolation, match="perception expired during preflight"):
        rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    assert checked and not [c for c in calls if c[0] == "joints"]


def test_first_closing_veto_creates_no_payload_episode(monkeypatch):
    checks = fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    real_harness(rt)
    factory = observed_scene.for_runtime
    began = []

    def make(*args):
        gate = factory(*args)
        def veto(state, **kwargs):
            raise SafetyViolation("neighbor blocks closure")
        gate.require_closing = veto
        return gate

    monkeypatch.setattr(observed_scene, "for_runtime", make)
    monkeypatch.setattr(contact_episode, "begin", lambda *a: began.append(a))
    with pytest.raises(SafetyViolation, match="neighbor blocks closure"):
        rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    assert began == [] and getattr(rt, "_held_provisional", None) is None
    assert [c for c in calls if c[0] == "gripper"] == [("gripper", 1., .8)]
    # The explicit pre-close retreat also keeps the observed profile gate.
    assert len([c for c in checks if c[0] == "preflight"]) == 4
