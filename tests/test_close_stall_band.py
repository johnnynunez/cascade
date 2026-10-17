"""The close-feedback stall band is a per-profile setting (`gripper.close_stall_band`).

`_wait_gripper_closed` lets the arm lift once the jaws reached the stage
target or stalled: every read stays within the band of an anchor for 0.5 s.
On Isaac Newton a cube grasped 10-30 deg off its faces is pinched on its
edges and slowly turned square by the jaws: the width keeps shrinking by
0.003-0.004 per 0.5 s, just above the 0.002 band, so no stall is seen until
the cube is aligned (up to 16.5 s) and the 8 s `close_feedback_timeout_s`
refuses the lift ("gripper closure did not settle; refusing to lift", 4 of 20
recorded closes; newton-diag, 18 Oct 2026). Band 0.01 of jaw travel settled
every recorded close in about 1 s, and 24 of 24 live trials lifted and held
the cube.

The Isaac profiles set 0.01. Every other profile resolves the 0.002 default,
the band main hard-coded, so they behave exactly as before. A bad value fails
the build before any arm exists.

The recorded closes (tests/fixtures/newton_close_traces_20261018.json) are
replayed through the real `_close_two_stage` on a fake clock.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.grasping.force import select_profile
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "newton_close_traces_20261018.json"
_TRACES = {t["trial"]: t for t in json.loads(_FIXTURE.read_text())["trials"]}
#: A jaw read waits for the next bridge iteration to end (measured ~14 it/s
#: under load): each feedback read costs this much wall clock. The runtime
#: then sleeps its own 0.05 s poll interval.
_READ_S = 0.07
#: Recorded closes the live run timed out on at the default band: stage 2
#: took 10.5, 8.7, 16.5 and 15.7 s (jaw 20-30 deg off the cube's face).
_REFUSED_AT_DEFAULT = [9, 14, 16, 17]
#: Jaw on a face (0 deg) or on the diagonal (45 deg): no squaring, settled in
#: about 1 s live at the default band.
_ALIGNED = [0, 1, 2, 3, 10, 11, 18, 19]


def _profile(name):
    return load_demo_config(camera="mock", arm=name, llm="mock")


def _replay(monkeypatch, trace, arm_cfg, grasp_cfg):
    """`_close_two_stage` against one recorded close: each read returns the
    newest bridge sample at the fake clock, measured from the stage command
    the runtime just sent. Returns (outcome, seconds from the stage-2 command
    to the end of the wait, width at the last read, width at the stage-2
    command)."""
    clock = [0.0]
    monkeypatch.setattr("cascade.skills.runtime.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("cascade.skills.runtime.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = SimpleNamespace(arm=arm_cfg, grasp=grasp_cfg)
    rt._grip_open, rt._grip_closed = 1.0, 0.0
    commands, reads = [], []
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.0), np.full(3, 1.0)))
    rt.arm = SimpleNamespace(raw=SimpleNamespace(), harness=harness,
                             set_gripper=lambda p, **kw: commands.append((clock[0], float(p))))

    def width():
        clock[0] += _READ_S
        stage = len(commands)
        t = clock[0] - commands[-1][0]
        series = trace["stage1"] if stage == 1 else trace["stage2"]
        if stage == 2 and t > series[-1][0]:
            raise AssertionError(f"trial {trace['trial']}: replay read past the recording ({t:.2f} s)")
        w = series[0][1]
        for ts, ws in series:
            if ts > t:
                break
            w = ws
        reads.append((stage, w))
        return w

    rt._gripper_width_frac = width
    try:
        rt._close_two_stage(select_profile("", "rigid"))
        outcome = "settled"
    except SkillError as exc:
        assert "refusing to lift" in str(exc), exc
        outcome = "refused"
    assert [p for _, p in commands] == pytest.approx([0.5, 0.15])
    at_stage2 = [w for s, w in reads if s == 1][-1]
    return outcome, clock[0] - commands[1][0], reads[-1][1], at_stage2


def _kitchen():
    cfg = _profile("isaac_kitchen")
    assert float(cfg.grasp.get("close_feedback_timeout_s")) == 8.0
    return cfg


@pytest.mark.parametrize("trial", sorted(_TRACES))
def test_isaac_kitchen_settles_every_recorded_newton_close_onto_the_cube(monkeypatch, trial):
    cfg = _kitchen()
    outcome, settle_s, width, at_stage2 = _replay(monkeypatch, _TRACES[trial], cfg.arm, cfg.grasp)
    assert outcome == "settled"
    # measured: 0.88-1.00 s at the 0.01 band (0.96-16.5 s at 0.002)
    assert settle_s <= 1.4, settle_s
    # the jaws stopped ON the cube (contact at 0.39-0.55 open; free closing
    # would run on to the 0.15 target), never mid-way through free travel
    assert 0.38 <= width <= 0.56, width
    # stage 1 ran its free travel to the 0.5 target or to the cube: the
    # wider band never mistook the closing ramp for a stall
    assert at_stage2 <= 0.56, at_stage2


@pytest.mark.parametrize("name", ["mock", "rebot_rs"])
def test_the_default_band_refuses_the_slow_squaring_closes_as_before(monkeypatch, name):
    """Golden (passes on main): a profile without the key keeps the 0.002
    band. With the kitchen's timing it still refuses the four closes the live
    run timed out on and settles the aligned ones."""
    kitchen = _kitchen()
    arm = _profile(name).arm
    for trial in _REFUSED_AT_DEFAULT:
        assert _replay(monkeypatch, _TRACES[trial], arm, kitchen.grasp)[0] == "refused", trial
    for trial in _ALIGNED:
        outcome, settle_s, _, _ = _replay(monkeypatch, _TRACES[trial], arm, kitchen.grasp)
        assert outcome == "settled" and settle_s <= 1.5, (trial, settle_s)


def test_the_slowest_free_closing_ramp_is_no_stall_at_the_isaac_band(monkeypatch):
    """Free closing moves >= 0.028 open fraction per bridge iteration (2.0 mm
    finger ramp): every read differs from the last by more than 0.01, so the
    jaws run to the target and are never taken for stopped mid-travel."""
    cfg = _kitchen()
    step, t_iter = 0.028, 1.0 / 14.0

    def series(start, goal, n=400):
        return [[k * t_iter, max(goal, start - step * k)] for k in range(n)]

    trace = {"trial": "ramp", "stage1": series(1.0, 0.5), "stage2": [[-0.01, 0.5]] + series(0.5, 0.15)}
    outcome, _, width, at_stage2 = _replay(monkeypatch, trace, cfg.arm, cfg.grasp)
    assert outcome == "settled"
    assert at_stage2 == pytest.approx(0.5) and width == pytest.approx(0.15)


def test_the_measured_creep_settles_at_the_isaac_band_and_not_at_the_default(monkeypatch):
    """The squaring creep: 0.004 per 0.5 s at 0.45 open (synthetic, the
    measured rate)."""
    def trace():
        creep = [[k * 0.0714, 0.45 - 0.004 * k * 0.0714 / 0.5] for k in range(0, 200)]
        return {"trial": "creep", "stage1": [[0.0, 0.5]], "stage2": [[-0.01, 0.5]] + creep}

    kitchen = _kitchen()
    outcome, settle_s, width, _ = _replay(monkeypatch, trace(), kitchen.arm, kitchen.grasp)
    assert outcome == "settled" and settle_s <= 1.4 and width == pytest.approx(0.445, abs=0.005)
    assert _replay(monkeypatch, trace(), _profile("mock").arm, kitchen.grasp)[0] == "refused"


# ── which profiles carry the band ──────────────────────────────────────────


def _band(name):
    from cascade.skills.runtime import close_stall_band

    return close_stall_band(_profile(name).arm.get("gripper"))


def test_every_isaac_profile_resolves_the_measured_band_and_no_other_profile_does():
    """Resolved configs, so `extends:` inheritance is covered."""
    arms = Path(__file__).resolve().parents[1] / "configs" / "arms"
    bands = {p.stem: _band(p.stem) for p in sorted(arms.glob("*.yaml"))}
    isaac = sorted(n for n in bands if n.startswith("isaac"))
    assert isaac == ["isaac", "isaac_cumotion", "isaac_kitchen", "isaac_kitchen_cumotion",
                     "isaac_kitchen_gpu", "isaac_kitchen_hug", "isaac_reach"]
    assert {n: bands[n] for n in isaac} == {n: 0.01 for n in isaac}
    assert {n: b for n, b in bands.items() if n not in isaac} == {
        n: 0.002 for n in bands if n not in isaac}


@pytest.mark.parametrize("name", ["rebot_rs", "rebot_rs_mb", "so101", "so101_mock", "fr3", "fr3_mock", "mock"])
def test_real_and_mock_arms_keep_mains_band(name):
    assert _band(name) == 0.002
    gripper = _profile(name).arm.get("gripper")
    assert gripper is None or gripper.get("close_stall_band") is None  # the default, not a copy of it


# ── a bad value fails closed ───────────────────────────────────────────────

_BAD = [0, 0.0, -0.01, 0.05, 0.5, 1, float("nan"), float("inf"), "wide", True]


@pytest.mark.parametrize("bad", _BAD)
def test_a_bad_band_fails_the_build_before_any_arm_exists(monkeypatch, tmp_path, bad):
    from cascade.apps import demo

    cfg = _profile("mock")
    cfg.arm.gripper._data["close_stall_band"] = bad
    made = []
    monkeypatch.setattr(demo, "make_arm", lambda *a, **k: made.append(a) or pytest.fail("arm made"))
    with pytest.raises(ValueError, match="close_stall_band"):
        demo.build_runtime(cfg, tmp_path / "run", lazy_arm=False)
    assert made == []


def test_a_bad_band_on_a_second_arm_fails_the_build_too(monkeypatch, tmp_path):
    from cascade.apps import demo

    cfg = load_demo_config(camera="mock", arms=["mock", "so101_mock"], llm="mock")
    second = cfg._data["arms"][1]
    second.setdefault("gripper", {})["close_stall_band"] = 0.2
    monkeypatch.setattr(demo, "make_arm", lambda *a, **k: pytest.fail("arm made"))
    with pytest.raises(ValueError, match="close_stall_band"):
        demo.build_runtime(cfg, tmp_path / "run", lazy_arm=False)


def test_a_config_without_an_arms_list_still_checks_its_arm(monkeypatch, tmp_path):
    """The close reads `cfg.arm`; a config with no `arms` list (the rig's
    fallback, `_arm_cfgs`) is checked there."""
    from cascade.apps import demo

    cfg = _profile("mock")
    cfg._data.pop("arms", None)
    cfg.arm.gripper._data["close_stall_band"] = 0.2
    monkeypatch.setattr(demo, "make_arm", lambda *a, **k: pytest.fail("arm made"))
    with pytest.raises(ValueError, match="close_stall_band"):
        demo.build_runtime(cfg, tmp_path / "run", lazy_arm=False)


@pytest.mark.parametrize("good", [0.001, 0.002, 0.01, 0.049])
def test_a_band_inside_the_limits_is_taken_as_set(good):
    from cascade.skills.runtime import close_stall_band

    assert close_stall_band({"close_stall_band": good}) == good
    assert close_stall_band({}) == 0.002 and close_stall_band(None) == 0.002
    assert close_stall_band({"close_stall_band": None}) == 0.002


@pytest.mark.parametrize("bad", _BAD)
def test_a_bad_band_refuses_the_close_before_the_jaws_move(monkeypatch, bad):
    """Defence in depth for a runtime built around the build check."""
    kitchen = _kitchen()
    arm = {"type": "isaac", "gripper": {"close_stall_band": bad}}
    clock = [0.0]
    monkeypatch.setattr("cascade.skills.runtime.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("cascade.skills.runtime.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = SimpleNamespace(arm=arm, grasp=kitchen.grasp)
    rt._grip_open, rt._grip_closed = 1.0, 0.0
    commands = []
    rt.arm = SimpleNamespace(raw=SimpleNamespace(), harness=SafetyHarness(SafetyLimits(np.full(3, -1.0), np.full(3, 1.0))),
                             set_gripper=lambda p, **kw: commands.append(p))
    rt._gripper_width_frac = lambda: 0.4
    with pytest.raises(SkillError, match="close_stall_band"):
        rt._close_two_stage(select_profile("", "rigid"))
    assert commands == []


def test_the_post_contact_hold_keeps_its_own_band(monkeypatch):
    """The PhysX hold's stall search (`_jaw_stall_width`) is not this setting:
    a wider band there is unmeasured (b36 evidence, ROADMAP follow-up). On
    the Isaac profile it still sees the measured creep as moving jaws."""
    cfg = _profile("isaac")
    assert math.isclose(_band("isaac"), 0.01)
    clock = [0.0]
    monkeypatch.setattr("cascade.skills.runtime.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("cascade.skills.runtime.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = SimpleNamespace(arm=cfg.arm, grasp=cfg.grasp)

    def creep():  # 0.004 per 0.5 s at 0.45 open, read every 0.12 s
        clock[0] += _READ_S
        return 0.45 - 0.004 * clock[0] / 0.5

    rt._gripper_width_frac = creep
    assert rt._jaw_stall_width(0.15, 2.0) is None
