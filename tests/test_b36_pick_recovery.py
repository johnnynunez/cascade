"""B36: pick reliability on the bare Isaac reBot scene, measured live (8-9 Oct 2026).

Fresh-stage Isaac picks of the pink cube (bare reBot scene, learned GraspGen-X grasps;
evidence: docs/evidence/b36-pick-reliability-20261010/) failed with three signatures:

* S1 -- attempt 1 ended "did not settle at grasp lift pose" with the cube LIFTED in
  the jaws; every retry then refused "already holding" until the budget ran out, the
  cube ended at the home pose and the scene reset failed. A held object was counted
  as a failed attempt: a logic bug. The persistence loops (pick_and_place and
  `_grasp_with_persistence`, used by handover / sort_by_color / rearrange) now ask
  the jaws after a failed attempt and go on with what they hold.
* S2 -- "did not settle above the place target", and S3 -- a drop in carry after a
  verified grip. Their common driver on PhysX: the bridge drops `effort` and the
  finger drives push to the stage-2 position target at full stiffness; with the cube
  squeezed, wrist roll sat a median 0.045 rad from its lift target (settle_tol
  0.045). `gripper.hold_squeeze_frac` holds at the measured contact width minus a
  fraction once the jaws stall (median 0.011 rad). PhysX: 15/16 confirmed picks with
  it vs 9/16 without; Newton: 1/4 with it vs 14/14 without, so the Isaac profile
  keys it by the engine the bridge reports and opts in for PhysX only.
* The hold must not blind the post-lift air-grasp check: the jaws were last
  commanded to the hold opening, so that is what "nothing resisted" is judged
  against.

Every test here runs the real runtime with the mock arm (no Isaac, no GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.grasping import evidence as grasp_evidence
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


def _count_grasps(rt, monkeypatch):
    calls = []
    real_grasp = rt.skill_grasp_object

    def counting(*a, **k):
        calls.append(a)
        return real_grasp(*a, **k)

    monkeypatch.setattr(rt, "skill_grasp_object", counting)
    return calls


# ── 1. a held object is never counted as a failed attempt ───────────────────


def test_premise_a_fresh_grasp_while_holding_refuses_already_holding(rt):
    """Premise (passes on main): what every live S1 retry logged."""
    rt.held_object = "red cube"
    with pytest.raises(SkillError, match="already holding"):
        rt.skill_grasp_object("red cube")


def test_premise_reconcile_promotes_the_provisional_marker_when_the_jaws_stall(rt):
    """Premise (passes on main): after a close the jaws stalled on something and
    the grasp never finished -> `_reconcile_held` makes it the held object."""
    rt._held_provisional = ("red cube", "cube", "red")
    rt.arm.set_gripper(1.0, effort=0.7)  # jaws jam at 0.5 on the object
    rt._reconcile_held()
    assert rt.held_object == "red cube"


def test_a_failed_lift_with_the_requested_object_in_the_jaws_goes_on_to_place(rt, monkeypatch):
    state = _fail_first_lift(rt, monkeypatch)
    calls = _count_grasps(rt, monkeypatch)
    res = rt.skill_pick_and_place("red cube")
    assert state["failed"] == 1
    assert len(calls) == 1, "the loop re-ran the grasp while the jaws held the cube"
    # the skill reports success as a placement (the ok/outcome fields are added by execute())
    assert res.get("placed_at") is not None and "error" not in res, res
    assert res["picked"] == "red cube" and res["grasp_attempts"] == 1
    assert res["place_attempts"] >= 1
    assert res["grip_verified"] is None  # the post-lift jaw check never ran
    assert "did not settle at grasp lift pose" in res["grasp_recovered_after"]
    assert "already holding" not in str(res)
    assert rt.held_object is None  # placed and released


def test_a_place_failure_after_a_recovered_grasp_names_the_recovery(rt, monkeypatch):
    """Live b361 / ab6pf4: recovered, then the place did not settle. The
    failure keeps saying it still holds the cube and how the grasp ended."""
    _fail_first_lift(rt, monkeypatch)

    def no_place(*a, **k):
        raise SkillError("did not settle above the place target")

    monkeypatch.setattr(rt, "skill_place_at", no_place)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["max_pick_attempts"] = 2
    res = rt.skill_pick_and_place("red cube")
    assert res["ok"] is False and res["stage"] == "place", res
    assert "still holding 'red cube'" in res["note"]
    assert "did not settle at grasp lift pose" in res["grasp_recovered_after"]
    assert "already holding" not in str(res)


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
    assert res["outcome"] == "stuck" and res.get("ask")


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
    assert "grasp_recovered_after" not in res


def test_an_estop_after_a_failed_attempt_is_not_read_as_a_grasp(rt, monkeypatch):
    """The e-stop stays a plain failure: the jaws are not asked."""
    asked, state = [], {"grasped": False}
    monkeypatch.setattr(rt, "_reconcile_held", lambda: asked.append(state["grasped"]))

    def stop_then_fail(*a, **k):
        state["grasped"] = True
        rt.arm.harness.estop("test")
        raise SkillError("did not settle at grasp lift pose")

    monkeypatch.setattr(rt, "skill_grasp_object", stop_then_fail)
    res = rt.skill_pick_and_place("red cube")
    assert True not in asked, "the jaws were asked after the e-stop"
    assert res["ok"] is False and "e-stop" in res["error"]
    assert res.get("outcome") != "stuck"


def test_an_unfinished_contact_episode_is_not_read_as_a_grasp(rt, monkeypatch):
    """Only explicit contact recovery may finish a failed close: even with a
    held flag set, the loop stops on the episode and places nothing."""
    placed = []

    def fail_in_episode(*a, **k):
        rt._contact_episode = object()
        rt.held_object = "red cube"
        raise SkillError("did not settle at grasp lift pose")

    monkeypatch.setattr(rt, "skill_grasp_object", fail_in_episode)
    monkeypatch.setattr(rt, "skill_place_at", lambda *a, **k: placed.append(a) or {"placed": True, "at": a})
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    try:
        res = rt.skill_pick_and_place("red cube")
    finally:
        rt._contact_episode = None
        rt.held_object = None
    assert placed == [], "placed during an unfinished contact episode"
    assert res["ok"] is False and "contact episode" in res["error"]


def test_handover_grasp_persistence_goes_on_with_the_object_it_holds(rt, monkeypatch):
    """`_grasp_with_persistence` (handover, sort_by_color, rearrange) had the
    same loop: a failed lift with the cube in the jaws is a grasp."""
    _fail_first_lift(rt, monkeypatch)
    calls = _count_grasps(rt, monkeypatch)
    res = rt.skill_handover("red cube")
    assert len(calls) == 1, "handover re-ran the grasp while the jaws held the cube"
    assert res.get("offering") == "red cube", res
    assert rt.held_object == "red cube"


def test_grasp_persistence_reports_the_recovery(rt, monkeypatch):
    _fail_first_lift(rt, monkeypatch)
    res = rt._grasp_with_persistence("red cube")
    assert res["held"] == "red cube" and res["grasp_attempts"] == 1
    assert res["grip_verified"] is None
    assert "did not settle at grasp lift pose" in res["recovered_after"]


def test_grasp_persistence_holding_another_object_stops(rt, monkeypatch):
    attempts = []

    def grasp_then_find_blue(*a, **k):
        attempts.append(a)
        rt.held_object = "blue cube"
        raise SkillError("did not settle at grasp lift pose")

    monkeypatch.setattr(rt, "skill_grasp_object", grasp_then_find_blue)
    monkeypatch.setattr(rt, "_reconcile_held", lambda: None)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    res = rt._grasp_with_persistence("red cube")
    assert len(attempts) == 1
    assert res["ok"] is False and res["held"] is False and "blue cube" in res["error"]


# ── 2. bounded post-contact hold (grasp close only) ────────────────────────


def _spy_gripper(rt, arm, monkeypatch):
    sent = []
    real = arm.set_gripper

    def spy(pos, effort=1.0, **kw):
        sent.append((float(pos), float(effort)))
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


def _pos(rt, closed_frac):
    return rt._grip_open + _span(rt) * closed_frac


def _events(monkeypatch):
    seen = []
    real = grasp_evidence.event

    def spy(kind, **data):
        seen.append((kind, data))
        return real(kind, **data)

    monkeypatch.setattr(grasp_evidence, "event", spy)
    return seen


def _grasp_close(rt):
    """The grasp's close as `skill_grasp_object` runs it: the hold squeeze is
    read (and validated) first, then the two-stage close and the hold."""
    return rt._grasp_close(select_profile("", "rigid"), rt._grasp_hold_squeeze())


def _golden_close(rt):
    """The two-stage close exactly as main commands it (rigid profile)."""
    p = select_profile("", "rigid")
    return [(pytest.approx(_pos(rt, p.close_frac_stage1)), pytest.approx(p.effort * 0.7)),
            (pytest.approx(_pos(rt, p.close_frac_stage2)), pytest.approx(p.effort))]


def test_golden_a_grasp_without_the_key_commands_the_jaws_exactly_as_main(rt_arm, monkeypatch):
    """Golden through the real grasp (passes on main by design): no
    `hold_squeeze_frac` -> open, stage 1, stage 2, nothing else."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    sent = _spy_gripper(rt, arm, monkeypatch)
    res = rt.skill_grasp_object("red cube")
    assert res.get("held") == "red cube" and res["grip_verified"] is True
    assert sent == [(pytest.approx(rt._grip_open), pytest.approx(0.8))] + _golden_close(rt)


def test_golden_a_clean_pick_reports_no_recovery(rt):
    """Golden (passes on main by design): a pick that never failed carries no
    `grasp_recovered_after`; one attempt, one place."""
    res = rt.skill_pick_and_place("red cube")
    assert res.get("placed_at") is not None and "error" not in res, res
    assert res["grasp_attempts"] == 1 and res["place_attempts"] == 1
    assert res["grip_verified"] is True
    assert "grasp_recovered_after" not in res


def test_without_the_key_the_close_is_exactly_the_two_stage_close(rt_arm, monkeypatch):
    """Golden: no `hold_squeeze_frac` -> the same two commands, no hold event."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    assert rt.cfg.arm.gripper.get("hold_squeeze_frac") is None
    sent = _spy_gripper(rt, arm, monkeypatch)
    events = _events(monkeypatch)
    assert _grasp_close(rt) is None
    assert sent == _golden_close(rt)
    assert not [k for k, _ in events if k == "close_hold"]


def test_after_contact_the_jaws_hold_at_contact_minus_the_squeeze(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5  # jaws jam at half travel: open fraction 0.5 at contact
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    sent = _spy_gripper(rt, arm, monkeypatch)
    events = _events(monkeypatch)
    hold_open = _grasp_close(rt)
    assert hold_open == pytest.approx(0.45)
    assert sent[:2] == _golden_close(rt)  # the two stages are unchanged
    assert len(sent) == 3 and sent[-1][0] == pytest.approx(_pos(rt, 1.0 - 0.45)), sent
    assert sent[-1][1] == pytest.approx(select_profile("", "rigid").effort)
    contact, stage2 = _pos(rt, 0.5), _pos(rt, 0.85)
    # the hold still presses (beyond contact) but much less than stage 2 did
    assert abs(sent[-1][0] - rt._grip_open) > abs(contact - rt._grip_open)
    assert abs(sent[-1][0] - contact) < abs(stage2 - contact)
    hold = [d for k, d in events if k == "close_hold"]
    assert hold and hold[-1]["applied"] is True
    assert hold[-1]["contact_open_frac"] == pytest.approx(0.5)
    assert hold[-1]["hold_open_frac"] == pytest.approx(0.45)


def test_the_hold_is_only_asked_for_by_the_grasp(rt_arm, monkeypatch):
    """`close_gripper` and the screw stroke (`_close_two_stage`) keep the
    plain close even with the key set and a PhysX backend: only the grasp
    close was measured."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    arm.physics_engine = "physx"
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05}
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert rt._close_two_stage(select_profile("", "rigid")) is None
    assert sent == _golden_close(rt)
    sent.clear()
    rt.skill_close_gripper()
    assert sent == _golden_close(rt)


def test_a_driver_with_its_own_two_stage_close_never_holds_here(rt_arm, monkeypatch):
    """The real reBot driver closes in one call and bounds its own squeeze
    (B38): the grasp close hands it the two stages and adds nothing."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    arm.physics_engine = "physx"
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05}
    calls = []

    def driver_close(**kw):  # the driver closes; the object jams the jaws halfway
        calls.append(kw)
        arm._gripper = _pos(rt, 0.5)

    monkeypatch.setattr(arm, "close_gripper_two_stage", driver_close, raising=False)
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert rt._grasp_hold_squeeze() is None
    assert _grasp_close(rt) is None
    assert len(calls) == 1 and sent == []


def test_the_hold_keeps_a_grasp_that_still_verifies(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    sent = _spy_gripper(rt, arm, monkeypatch)
    res = rt.skill_grasp_object("red cube")
    assert res.get("held") == "red cube" and res["grip_verified"] is True
    assert any(p == pytest.approx(_pos(rt, 0.55)) for p, _ in sent), sent  # the hold ran


def _lose_the_object_in_the_lift(rt, arm, monkeypatch, sent):
    """At the lift the object slips out: the jaws travel on to whatever they
    were last commanded to (the mock only moves its jaws on a command)."""
    real = carry_attachment.move

    def move(runtime, q, **kwargs):
        if getattr(runtime, "_held_provisional", None) is not None and arm.object_stop_frac is not None:
            arm.object_stop_frac = None
            arm._gripper = sent[-1][0]
        return real(runtime, q, **kwargs)

    monkeypatch.setattr(carry_attachment, "move", move)


def test_premise_an_object_lost_in_the_lift_is_an_air_grasp(rt_arm, monkeypatch):
    """Premise (passes on main): without a hold the empty jaws sit at the
    stage-2 opening after the lift and the jaw check refuses the grasp."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    sent = _spy_gripper(rt, arm, monkeypatch)
    _lose_the_object_in_the_lift(rt, arm, monkeypatch, sent)
    res = rt.skill_grasp_object("red cube")
    assert res["ok"] is False and "air grasp" in res["error"], res
    assert rt.held_object is None


def test_an_object_lost_in_the_lift_after_the_hold_is_still_an_air_grasp(rt_arm, monkeypatch):
    """The hold leaves the empty jaws at the HOLD opening (0.45 open), far above
    the stage-2 one (0.15): judged against stage 2 this read as a held cube."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    sent = _spy_gripper(rt, arm, monkeypatch)
    _lose_the_object_in_the_lift(rt, arm, monkeypatch, sent)
    res = rt.skill_grasp_object("red cube")
    assert sent[-1][0] == pytest.approx(_pos(rt, 0.55)), sent  # the hold was the last command
    assert res["ok"] is False and "air grasp" in res["error"], res
    assert rt.held_object is None


def test_no_object_in_the_jaws_means_no_hold(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = None  # free close: the jaws reach the stage-2 target
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert _grasp_close(rt) is None
    assert sent == _golden_close(rt)


def test_a_stall_too_close_to_the_stage2_target_is_not_held(rt_arm, monkeypatch):
    """A thin object stalls the jaws just short of stage 2: the stage-2 command
    already presses no harder than a hold would, so it stands."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.84  # open fraction 0.16, within 0.02 of stage 2 (0.15)
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert _grasp_close(rt) is None
    assert sent == _golden_close(rt)


def test_jaws_that_keep_moving_get_no_hold(rt_arm, monkeypatch):
    """Every read moves the jaws by more than the stall band: no stall within
    a timeout longer than the 0.5 s stall window -> the stage-2 command stands."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    rt.cfg.arm.gripper._data["hold_stall_timeout_s"] = 1.0
    widths = iter(np.arange(0.9, 0.2, -0.004))  # 0.004 per read > the 0.002 band
    monkeypatch.setattr(rt, "_gripper_width_frac", lambda: float(next(widths, 0.2)))
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert _grasp_close(rt) is None
    assert sent == _golden_close(rt)


def test_jaws_that_stop_get_the_hold_at_the_stall_width(rt_arm, monkeypatch):
    """The same reads, but the jaws stop at 0.6: the stall is found there."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    rt.cfg.arm.gripper._data["hold_stall_timeout_s"] = 30.0
    widths = iter(np.arange(0.9, 0.6, -0.004))
    monkeypatch.setattr(rt, "_gripper_width_frac", lambda: float(next(widths, 0.6)))
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert _grasp_close(rt) == pytest.approx(0.55)
    assert sent[-1][0] == pytest.approx(_pos(rt, 0.45))


#: ab6pf4 (live, 9 Oct, PhysX, branch): the jaw widths read after the stage-2
#: command (every ~0.07 s for the 4 s stall timeout). No stall was found, the
#: hold was not applied and the run failed S2 ("did not settle above the place
#: target") after recovering the held cube.
_AB6PF4_WIDTHS = [
    0.5459, 0.5468, 0.5443, 0.545, 0.5431, 0.5461, 0.5451, 0.5424, 0.5466, 0.5441, 0.5453, 0.5434, 0.5451, 0.5432,
    0.5467, 0.545, 0.5435, 0.5452, 0.543, 0.5449, 0.5416, 0.5441, 0.5411, 0.5437, 0.5406, 0.5409, 0.5441, 0.5453,
    0.5463, 0.545, 0.5446, 0.545, 0.5452, 0.5432, 0.5449, 0.5434, 0.546, 0.5443, 0.543, 0.5457, 0.5447, 0.5432,
    0.5439, 0.5451, 0.5465, 0.5459, 0.5437, 0.5465, 0.5422, 0.5448, 0.5451, 0.544, 0.5443, 0.5462, 0.544, 0.5413,
]


def test_recorded_physx_contact_chatter_is_not_a_stall(rt_arm, monkeypatch):
    """Known limitation, pinned as measured: ab6pf4's +-0.003 contact chatter is
    wider than the 0.002 stall band, so no hold applies (1 of 16 PhysX runs).
    A wider band is an unmeasured change (ROADMAP follow-up), not a tweak."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    rt.cfg.arm.gripper._data["hold_stall_timeout_s"] = 2.0  # fewer reads than recorded
    reads = iter(_AB6PF4_WIDTHS)
    monkeypatch.setattr(rt, "_gripper_width_frac", lambda: next(reads, _AB6PF4_WIDTHS[-1]))
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert _grasp_close(rt) is None
    assert sent == _golden_close(rt)


@pytest.mark.parametrize("bad", [0, 1.0, -0.1, 1.5, "lots", float("nan")])
def test_an_invalid_squeeze_is_refused_before_the_grasp_moves(rt_arm, monkeypatch, bad):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = bad
    sent = _spy_gripper(rt, arm, monkeypatch)
    n_moves = len(arm.commands)
    with pytest.raises(SkillError, match="hold_squeeze_frac"):
        rt.skill_grasp_object("red cube")
    assert sent == [] and len(arm.commands) == n_moves, "the grasp moved before refusing"


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), "soon"])
def test_an_invalid_stall_timeout_is_refused_before_the_grasp_moves(rt_arm, monkeypatch, bad):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    rt.cfg.arm.gripper._data["hold_stall_timeout_s"] = bad
    sent = _spy_gripper(rt, arm, monkeypatch)
    n_moves = len(arm.commands)
    with pytest.raises(SkillError, match="hold_stall_timeout_s"):
        rt.skill_grasp_object("red cube")
    assert sent == [] and len(arm.commands) == n_moves, "the grasp moved before refusing"


def test_only_the_bare_isaac_scene_profiles_opt_in_and_the_tolerance_is_unchanged():
    """Resolved configs, so `extends:` inheritance is covered. The bare-scene
    Isaac profiles inherit isaac.yaml's PhysX-only hold (isaac_reach is the
    same scene with extra analytic candidates: its B45 comparison against
    `isaac` must not differ in the close); the kitchen profiles opt out until
    measured there."""
    from pathlib import Path

    arms = Path(__file__).resolve().parents[1] / "configs" / "arms"
    opted = {}
    for p in sorted(arms.glob("*.yaml")):
        g = load_demo_config(camera="mock", arm=p.stem, llm="mock").arm.get("gripper")
        value = g.get("hold_squeeze_frac") if g is not None else None
        if value is not None:
            opted[p.stem] = value.as_dict() if hasattr(value, "as_dict") else value
    assert sorted(opted) == ["isaac", "isaac_cumotion", "isaac_reach"], opted
    for value in opted.values():
        assert value == {"physx": 0.05}, value  # no newton entry: the plain close there
    for name in ("isaac", "isaac_kitchen", "isaac_reach"):
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
    ("physx", 0.05), ("PhysX", 0.05), ("newton", None), ("mujoco", None), (None, None), ("", None), (3, None),
])
def test_a_per_engine_squeeze_applies_only_on_the_engine_it_names(rt_arm, monkeypatch, engine, expected):
    rt, arm = rt_arm
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05}
    _with_engine(rt, monkeypatch, engine)
    assert rt._hold_squeeze_frac() == expected


def test_a_plain_number_applies_on_any_backend(rt_arm, monkeypatch):
    rt, arm = rt_arm
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = 0.05
    _with_engine(rt, monkeypatch, None)
    assert rt._hold_squeeze_frac() == 0.05


def test_a_backend_without_an_engine_gets_no_per_engine_hold(rt_arm, monkeypatch):
    """The mock arm reports no engine: the close is exactly the two-stage close."""
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05}
    assert getattr(rt.arm.raw, "physics_engine", None) is None
    sent = _spy_gripper(rt, arm, monkeypatch)
    assert _grasp_close(rt) is None
    assert sent == _golden_close(rt)


def test_the_physx_hold_runs_through_the_close_when_the_backend_reports_physx(rt_arm, monkeypatch):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = {"physx": 0.05, "newton": None}
    sent = _spy_gripper(rt, arm, monkeypatch)
    arm.physics_engine = "physx"  # the mock arm is the raw backend here
    assert _grasp_close(rt) == pytest.approx(0.45)
    assert sent[-1][0] == pytest.approx(_pos(rt, 1.0 - 0.45)), sent
    sent.clear()
    arm.physics_engine = "newton"
    assert _grasp_close(rt) is None
    assert sent == _golden_close(rt)


@pytest.mark.parametrize("bad", [{"physx": 0}, {"physx": 1.0}, {"newton": "lots"}, {"physx": float("nan")}])
def test_a_bad_entry_for_any_engine_is_refused_before_the_grasp_moves(rt_arm, monkeypatch, bad):
    rt, arm = rt_arm
    arm.object_stop_frac = 0.5
    arm.physics_engine = "physx"
    rt.cfg.arm.gripper._data["hold_squeeze_frac"] = bad
    sent = _spy_gripper(rt, arm, monkeypatch)
    n_moves = len(arm.commands)
    with pytest.raises(SkillError, match="hold_squeeze_frac"):
        rt.skill_grasp_object("red cube")
    assert sent == [] and len(arm.commands) == n_moves, "the grasp moved before refusing"


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
                            ({"ok": True}, None), ({"ok": True, "engine": 3}, None),
                            ({"ok": True, "engine": ""}, None), (None, None)):
        arm._client = _Client(reply)
        arm.connect()
        assert arm.physics_engine == expected
        assert arm._client.ops == ["ping"]
