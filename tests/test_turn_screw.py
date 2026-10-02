"""turn_screw: the ratchet-regrip fastening skill (2026-09-03).

Covers the reflex grammar (bilingual verbs, turn counts), the skill against
the mock rig E2E (strokes actually rotate the roll joint, the watcher pause
contract via _MOTION_SKILLS membership, spec exposure), and the failure
shapes (unknown direction, unlocalizable fastener).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import has_pinocchio

from cascade.agent.reflex import parse_command
from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS

needs_pin = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")


# ── reflex grammar ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cmd,direction,turns",
    [
        ("tighten the screw", "tighten", None),
        ("screw in the bolt", "tighten", None),
        ("aprieta el tornillo", "tighten", None),
        ("atornilla la tuerca", "tighten", None),
        ("loosen the screw", "loosen", None),
        ("unscrew the wing nut", "loosen", None),
        ("afloja el tornillo", "loosen", None),
        ("desatornilla el perno", "loosen", None),
        ("tighten the screw 3 turns", "tighten", 3.0),
        ("loosen the knob two turns", "loosen", 2.0),
        ("aprieta el tornillo dos vueltas", "tighten", 2.0),
        ("afloja la tapa una vuelta", "loosen", 1.0),
    ],
)
def test_reflex_turn_screw(cmd, direction, turns):
    plan = parse_command(cmd)
    assert plan is not None, f"{cmd!r} should be a reflex"
    assert plan.intent == "turn_screw"
    (name, args), = plan.calls
    assert name == "turn_screw"
    assert args["direction"] == direction
    if turns is None:
        assert "turns" not in args
    else:
        assert args["turns"] == pytest.approx(turns)


def test_reflex_turn_screw_does_not_shadow_pick():
    plan = parse_command("pick up the screwdriver")
    assert plan is not None
    assert plan.intent == "pick"


def test_turn_screw_is_a_motion_skill_and_has_a_spec():
    # Forgetting _MOTION_SKILLS re-registers the fastener mid-air during the
    # motion; forgetting the spec hides the skill from every LLM/MCP host.
    assert "turn_screw" in _MOTION_SKILLS
    spec = next(s for s in TOOL_SPECS if s["name"] == "turn_screw")
    assert spec["parameters"]["required"] == ["label"]
    # the arm-rig injection at the bottom of runtime.py must have added the
    # optional `arm` param (turn_screw moves a robot on a shared table).
    assert "arm" in spec["parameters"]["properties"]


# ── E2E on the mock rig ───────────────────────────────────────────────────

@pytest.fixture
def rig(tmp_path):
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if runtime.beliefs.find("red object") is not None:
                break
            time.sleep(0.05)
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_turn_screw_tighten_e2e(rig):
    runtime, arm = rig
    res = runtime.execute("turn_screw", {"label": "red object", "turns": 1.0})
    assert res["ok"], res
    assert res["turns_applied"] > 0.0
    assert res["strokes"] >= 1
    assert res["direction"] == "tighten"


@needs_pin
def test_turn_screw_loosen_e2e(rig):
    runtime, arm = rig
    res = runtime.execute(
        "turn_screw", {"label": "red object", "direction": "loosen", "turns": 0.5}
    )
    assert res["ok"], res
    assert res["turns_applied"] > 0.0


@needs_pin
def test_turn_screw_strokes_actually_rotate_the_roll_joint(rig):
    """A green run must mean the wrist really ratcheted: sample the roll
    joint during the skill and require genuine excursion in BOTH directions
    (work stroke + counter-rotation). Guards the 'motion test that cannot
    fail' trap -- a no-op implementation returning ok would pass everything
    else."""
    runtime, arm = rig
    joint = runtime._screw_joint
    samples: list[float] = []
    orig = arm.send_joint_target

    def spy(q):
        samples.append(float(np.asarray(q).reshape(-1)[joint]))
        return orig(q)

    arm.send_joint_target = spy
    try:
        res = runtime.execute("turn_screw", {"label": "red object", "turns": 0.5})
    finally:
        arm.send_joint_target = orig
    assert res["ok"], res
    assert samples, "skill never streamed a waypoint"
    arr = np.asarray(samples)
    span = arr.max() - arr.min()
    assert span > 0.3, f"roll joint barely moved ({span:.3f} rad) - INCONCLUSIVE"
    # ratcheting = direction reverses at least twice (back-wind, stroke, retreat)
    d = np.diff(arr)
    d = d[np.abs(d) > 1e-6]
    flips = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
    assert flips >= 2, f"no ratchet pattern in the roll trace (flips={flips})"


@needs_pin
def test_turn_screw_error_shapes(rig):
    runtime, _ = rig
    res = runtime.execute("turn_screw", {"label": "red object", "direction": "sideways"})
    assert res["ok"] is False
    assert "direction" in res["error"]
    res = runtime.execute("turn_screw", {"label": "unobtainium fastener"})
    assert res["ok"] is False


@needs_pin
@pytest.mark.parametrize("turns", [float("nan"), float("inf"), -1, 0, True, 7, "invalid"])
def test_invalid_turn_count_fails_before_any_motion(rig, turns, monkeypatch):
    runtime, _ = rig
    moves = []
    monkeypatch.setattr(runtime.arm, "move_joints", lambda *a, **kw: moves.append(a) or True)
    result = runtime.execute("turn_screw", {"label": "red object", "turns": turns})
    assert result["ok"] is False
    assert "turns" in result["error"]
    assert not moves


@needs_pin
def test_unreachable_engagement_never_closes_or_spins_at_hover(rig, monkeypatch):
    from types import SimpleNamespace

    runtime, _ = rig
    calls, closes = [], []

    def ik(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(success=len(calls) == 1, q=runtime._profile_q("home_q", "test"))

    monkeypatch.setattr(runtime.kin, "ik", ik)
    monkeypatch.setattr(runtime, "_close_two_stage", lambda *a, **kw: closes.append(a))
    result = runtime.execute("turn_screw", {"label": "red object", "turns": 0.5})
    assert result["ok"] is False
    assert "engagement pose" in result["error"]
    assert not closes


@needs_pin
def test_wrist_rotation_does_not_claim_measured_threading(rig):
    runtime, _ = rig
    result = runtime.execute("turn_screw", {"label": "red object", "turns": 0.5})
    assert result["ok"], result
    assert result["turns_commanded"] >= 0.5
    assert result["physical_verification"] == {
        "status": "unverified", "fastener_turns": None,
        "axial_advance_m": None, "seating_verified": False,
    }
