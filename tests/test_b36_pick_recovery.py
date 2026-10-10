"""B36: pick reliability failures measured live on Isaac (8 Oct 2026).

Grasp-evidence receipts from 14 fresh-stage Isaac picks (bare reBot scene,
PhysX) show one mechanism behind most failures:

* With the cube in the jaws, wrist roll (joint 6) sits 0.042-0.049 rad from
  whatever target it is given. This held in 13 of 13 rigid-profile lifts, against
  ``settle_tol`` 0.045. The arm is static (finite-difference velocity 0), so every
  held move is a coin flip: "did not settle at grasp lift pose" and "did not
  settle above the place target". The deflection scales with the close
  squeeze: in E1 the same pick with the fragile close (stage 2 = 0.60) left
  0.017 rad. Two in-carry drops showed 0.23-0.32 rad transient twists.
* The Isaac bridge has no force bound: it drops ``effort``, and the drives push
  the jaws toward the stage-2 position target at full stiffness. The real reBot
  driver re-commands a light hold after contact (``_hold_light``). Here the sim
  analogue is ``gripper.hold_squeeze_frac``: once the jaws stall on the object,
  hold at the measured contact width minus that fraction. Only the Isaac
  profiles opt in, the settle tolerance is unchanged, and other arms stay
  byte-identical.
* An attempt that fails after the close took the object re-ran the grasp,
  which refused "already holding" until the budget ran out (live main3: 5
  attempts). The loop now goes on to place what it holds, as the skill's entry
  already does.

The first live idea, keeping the measured contact rotation in the lift, was
falsified (0.049 rad again against its own target) and is not shipped.
"""
from __future__ import annotations

import numpy as np
import pytest

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.grasping.force import select_profile
from cascade.skills import carry_attachment
from cascade.types import SkillError


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    arm.object_stop_frac = 0.5  # the mock jaws jam halfway: an object is in the gripper
    try:
        yield runtime
    finally:
        shutdown_runtime(runtime, arm)


def _fail_first_lift(rt, monkeypatch):
    """The first motion issued while the close's provisional marker is live is
    the grasp lift: report it as not settled, as the Isaac settle check did."""
    real = carry_attachment.move
    state = {"failed": 0, "after_close": []}

    def move(runtime, q, **kwargs):
        if getattr(runtime, "_held_provisional", None) is not None:
            state["after_close"].append(np.asarray(q, float).copy())
            if not state["failed"]:
                state["failed"] += 1
                real(runtime, q, **kwargs)  # the arm does move; it just does not settle
                return False
        return real(runtime, q, **kwargs)

    monkeypatch.setattr(carry_attachment, "move", move)
    return state


# ── 1. the loop places what it is already holding ───────────────────────────


def test_a_failed_lift_with_the_requested_object_in_the_jaws_goes_on_to_place(rt, monkeypatch):
    state = _fail_first_lift(rt, monkeypatch)
    calls = []
    real_grasp = rt.skill_grasp_object

    def counting(*a, **k):
        calls.append(a)
        return real_grasp(*a, **k)

    monkeypatch.setattr(rt, "skill_grasp_object", counting)
    res = rt.skill_pick_and_place("red cube")
    assert state["failed"] == 1
    assert len(calls) == 1, "the loop re-ran the grasp while the jaws held the cube"
    # the skill reports success as a placement (the ok/outcome fields are added by execute())
    assert res.get("placed_at") is not None and "error" not in res, res
    assert res["picked"] == "red cube" and res["grasp_attempts"] == 1
    assert res["place_attempts"] >= 1
    assert "already holding" not in str(res)
    assert rt.held_object is None  # placed and released


def test_without_the_fix_the_retries_refuse_already_holding(rt, monkeypatch):
    """Pins the measured failure shape: had the loop retried the grasp, every
    retry would refuse with "already holding" (what the live runs logged)."""
    _fail_first_lift(rt, monkeypatch)
    rt.cfg.grasp._data["max_pick_attempts"] = 1  # one attempt: the lift fails, nothing else runs
    rt.skill_pick_and_place("red cube")
    # whatever the outcome, a fresh grasp of the same object while held refuses
    rt.held_object = "red cube"
    with pytest.raises(SkillError, match="already holding"):
        rt.skill_grasp_object("red cube")


def test_holding_a_different_object_after_a_failed_attempt_stops_retrying(rt, monkeypatch):
    attempts = []

    def grasp_then_find_blue(*a, **k):
        attempts.append(a)
        rt.held_object = "blue cube"  # what the jaws turned out to hold
        raise SkillError("did not settle at grasp lift pose")

    monkeypatch.setattr(rt, "skill_grasp_object", grasp_then_find_blue)
    monkeypatch.setattr(rt, "_reconcile_held", lambda: None)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    res = rt.skill_pick_and_place("red cube")
    assert len(attempts) == 1, "retried a grasp while holding another object"
    assert res["ok"] is False and "blue cube" in res["error"]


def test_a_plain_miss_still_retries(rt, monkeypatch):
    """Nothing held after the failure: the old persistence is untouched."""
    attempts = []

    def miss(*a, **k):
        attempts.append(a)
        raise SkillError("no grasp candidates")

    monkeypatch.setattr(rt, "skill_grasp_object", miss)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["max_pick_attempts"] = 3
    rt.cfg.grasp._data["persist_seconds"] = 60.0
    res = rt.skill_pick_and_place("red cube")
    assert len(attempts) == 3 and res["ok"] is False


# ── 2. bounded post-contact hold (Isaac profile) ─────────────────────────


def _spy_gripper(rt, arm, monkeypatch):
    sent = []
    real = arm.set_gripper

    def spy(pos, effort=1.0, **kw):
        sent.append(float(pos))
        return real(pos, effort)

    monkeypatch.setattr(arm, "set_gripper", spy)
    return sent


@pytest.fixture
def rt_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


def _span(rt):
    return rt._grip_closed - rt._grip_open


def test_without_the_key_the_close_ends_at_the_stage2_target(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    assert rt.cfg.arm.gripper.get("hold_squeeze_frac") is None
    sent = _spy_gripper(rt, arm, monkeypatch)
    rt._close_two_stage(select_profile("", "rigid"))
    stage2 = rt._grip_open + _span(rt) * select_profile("", "rigid").close_frac_stage2
    assert sent and sent[-1] == pytest.approx(stage2)


def test_after_contact_the_jaws_hold_at_contact_minus_the_squeeze(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5  # jaws jam at half travel: open fraction 0.5 at contact
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    sent = _spy_gripper(rt, arm, monkeypatch)
    rt._close_two_stage(select_profile("", "rigid"))
    hold = rt._grip_open + _span(rt) * (1.0 - 0.45)
    assert sent[-1] == pytest.approx(hold), sent
    contact = rt._grip_open + _span(rt) * 0.5
    stage2 = rt._grip_open + _span(rt) * 0.85
    # the hold still presses (beyond contact) but much less than stage 2 did
    assert abs(sent[-1] - rt._grip_open) > abs(contact - rt._grip_open)
    assert abs(sent[-1] - contact) < abs(stage2 - contact)


def test_the_hold_keeps_a_grasp_that_still_verifies(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    res = rt.skill_grasp_object("red cube")
    assert res.get("held") == "red cube"


def test_no_object_in_the_jaws_means_no_hold(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = None  # free close: the jaws reach the stage-2 target
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    sent = _spy_gripper(rt, arm, monkeypatch)
    rt._close_two_stage(select_profile("", "rigid"))
    stage2 = rt._grip_open + _span(rt) * 0.85
    assert sent[-1] == pytest.approx(stage2)


def test_no_stall_observed_means_no_hold(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    rt.cfg.arm.gripper._data["hold_stall_timeout_s"] = 0.3
    widths = iter(np.linspace(0.9, 0.5, 10_000))  # jaws never stop moving
    monkeypatch.setattr(rt, "_gripper_width_frac", lambda: float(next(widths)))
    sent = _spy_gripper(rt, arm, monkeypatch)
    rt._close_two_stage(select_profile("", "rigid"))
    stage2 = rt._grip_open + _span(rt) * 0.85
    assert sent[-1] == pytest.approx(stage2)


@pytest.mark.parametrize("bad", [0, 1.0, -0.1, 1.5, "lots", float("nan")])
def test_an_invalid_squeeze_is_refused_before_the_jaws_move(rt_arm, monkeypatch, bad):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = bad
    sent = _spy_gripper(rt, arm, monkeypatch)
    with pytest.raises(SkillError, match="hold_squeeze_frac"):
        rt._close_two_stage(select_profile("", "rigid"))
    assert sent == []


def test_only_the_bare_isaac_scene_profiles_opt_in_and_the_tolerance_is_unchanged():
    """Resolved configs, so `extends:` inheritance is covered: the kitchen
    profiles inherit isaac.yaml and must opt out until measured there. The
    Isaac opt-in is PhysX-only: Newton measured worse with the hold."""
    from pathlib import Path

    arms = Path(__file__).resolve().parents[1] / "configs" / "arms"
    opted = {}
    for p in sorted(arms.glob("*.yaml")):
        g = load_demo_config(camera="mock", arm=p.stem, llm="mock").arm.get("gripper")
        value = g.get("hold_squeeze_frac") if g is not None else None
        if value is not None:
            opted[p.stem] = value.as_dict() if hasattr(value, "as_dict") else value
    assert sorted(opted) == ["isaac", "isaac_cumotion"], opted
    for value in opted.values():
        assert value == {"physx": 0.05}, value  # no newton entry: the plain close there
    for name in ("isaac", "isaac_kitchen"):
        assert float(load_demo_config(camera="mock", arm=name, llm="mock").arm.get("settle_tol")) == 0.045


# ── 3. the hold is keyed by the physics engine the backend reports ──────────


class _EngineArm:
    """Stands in for `SafeArm.raw` reporting a physics engine (Isaac bridge)."""

    def __init__(self, engine):
        self.physics_engine = engine


def _with_engine(rt, monkeypatch, engine):
    safe = rt.arm
    monkeypatch.setattr(type(safe), "raw", property(lambda self: _EngineArm(engine)), raising=False)


@pytest.mark.parametrize("engine,expected", [
    ("physx", 0.05), ("PhysX", 0.05), ("newton", None), ("mujoco", None), (None, None), ("", None),
])
def test_a_per_engine_squeeze_applies_only_on_the_engine_it_names(rt_arm, monkeypatch, engine, expected):
    rt, arm = rt_arm
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05}
    _with_engine(rt, monkeypatch, engine)
    assert rt._hold_squeeze_frac() == expected


def test_a_backend_without_an_engine_gets_no_per_engine_hold(rt_arm, monkeypatch):
    """The mock arm reports no engine: the close ends at the stage-2 target."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05}
    assert getattr(rt.arm.raw, "physics_engine", None) is None
    sent = _spy_gripper(rt, arm, monkeypatch)
    rt._close_two_stage(select_profile("", "rigid"))
    stage2 = rt._grip_open + _span(rt) * 0.85
    assert sent[-1] == pytest.approx(stage2)


def test_the_physx_hold_runs_through_the_close_when_the_backend_reports_physx(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05, "newton": None}
    sent = _spy_gripper(rt, arm, monkeypatch)
    arm.physics_engine = "physx"  # the mock arm is the raw backend here
    rt._close_two_stage(select_profile("", "rigid"))
    assert sent[-1] == pytest.approx(rt._grip_open + _span(rt) * (1.0 - 0.45)), sent
    sent.clear()
    arm.physics_engine = "newton"
    rt._close_two_stage(select_profile("", "rigid"))
    assert sent[-1] == pytest.approx(rt._grip_open + _span(rt) * 0.85), sent


@pytest.mark.parametrize("bad", [{"physx": 0}, {"physx": 1.0}, {"newton": "lots"}, {"physx": float("nan")}])
def test_a_bad_entry_for_any_engine_is_refused_before_the_jaws_move(rt_arm, monkeypatch, bad):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    arm.physics_engine = "physx"
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = bad
    sent = _spy_gripper(rt, arm, monkeypatch)
    with pytest.raises(SkillError, match="hold_squeeze_frac"):
        rt._close_two_stage(select_profile("", "rigid"))
    assert sent == []


def test_isaac_arm_records_the_engine_named_in_the_bridge_ping():
    from cascade.control.isaac_arm import IsaacArm

    cfg = load_demo_config(camera="mock", arm="isaac", llm="mock").arm
    arm = IsaacArm(cfg)
    assert arm.physics_engine is None

    class _Client:
        _addr = ("fake", 1)

        def __init__(self, reply):
            self.reply = reply
            self.ops = []

        def connect(self):
            pass

        def request(self, payload, timeout_s=None):
            self.ops.append(payload["op"])
            return self.reply

    for reply, expected in (({"ok": True, "engine": "physx"}, "physx"),
                            ({"ok": True, "engine": "Newton"}, "newton"),
                            ({"ok": True}, None), ({"ok": True, "engine": 3}, None)):
        arm._client = _Client(reply)
        arm.connect()
        assert arm.physics_engine == expected
        assert arm._client.ops == ["ping"]
