"""No skill opens the jaws while an interrupted grasp's provisional hold
survives its entry reconcile (#373 review, pre-existing A).

A STOP after the close leaves only the close's provisional hold
(`_held_provisional`). When the jaws then cannot be read (unknown feedback, or
a reading that is NaN / +-inf), `_reconcile_held()` can neither promote nor
drop it, and move_home reports `holding_unknown` ("check the gripper before
any other grasp"). On main nothing enforced that text: grasp_object opened the
jaws to 6.2 at home (z 0.4377 m) or, with no column kept, at the object, and
the object could drop. Every skill whose first jaw command opens the jaws
now refuses before any command while that hold survives, in reset_scene's
words (take it out by hand, open_gripper; move_home first while the approach
column still applies, which only a direct caller meets: execute() forfeits a
kept column before every motion skill but move_home and reset_scene). Jaws
that read empty drop the hold and the grasp runs as on main.

The REAL rebot_rs type over a MockArm (`test_stop_retrace.rebot_real`).
"""
from __future__ import annotations

import dataclasses

import pytest

import test_stop_retrace as T
from cascade.memory.envelope import failure_signature, normalize_failure
from cascade.skills import carry_attachment, stop_retrace
from cascade.skills.runtime import ProvisionalHold, SkillRuntime
from cascade.types import SkillStuck
from conftest import needs_pin

pytestmark = needs_pin

rebot_real = T.rebot_real  # the REAL rebot_rs type (a real gripper), its backend a MockArm

#: jaw readings a provisional hold survives: unknown feedback, and valid
#: readings that are no number
READINGS = ("unknown", "nan", "pinf", "ninf")
#: where the STOP left the jaws: stalled on the object, or at the stage-2 seat
STALLS = pytest.mark.parametrize("stall", [0.5, None], ids=["on-the-object", "at-the-seat"])

#: the refusal of a skill whose first jaw command opens the jaws, as execute()
#: reports it (reset_scene's #363 text, its last step the retry): STUCK, so
#: the agent relays the ask instead of opening the jaws itself
REFUSAL = ("SkillStuck: the jaw state could not be read: if red cube is in the jaws, take it out by hand, then "
           "open_gripper, then try again. Error: no valid jaw feedback")
#: ... while the approach column kept after the STOP still applies
REFUSAL_KEPT = ("SkillStuck: the jaw state could not be read: call move_home first (it raises the gripper up its "
                "approach column), then, if red cube is in the jaws, take it out by hand, open_gripper, try again. "
                "Error: no valid jaw feedback")
#: a persistence loop's stop when a failed attempt leaves a possible hold, in
#: #366/#377's words so the envelope files it `state:jaws_unreadable`
LOOP_STOP = "the jaw state could not be read and the jaws may still hold an object; not retrying"
#: ... and the human-actionable ask it ends STUCK with
LOOP_ASK = ("My jaws may still hold an object, and I cannot read whether they do. If one is in them, take it out "
            "by hand, then say 'open the gripper' (open_gripper), and ask again.")


def unreadable(rt, monkeypatch, reading):
    """From now on the jaws read `reading`: unknown feedback, or a valid
    reading that is no number."""
    if reading == "unknown":
        raw = T.mock_of(rt)
        real = raw.get_state
        monkeypatch.setattr(raw, "get_state", lambda: dataclasses.replace(real(), gripper_valid=False))
    else:
        T.bad_jaws(rt, monkeypatch, reading)


def holding_unknown_at_home(rt, monkeypatch, stall, reading="unknown"):
    """A STOP right after the close, the jaws unreadable once the retrace has
    run: move_home carries the jaws home and reports `holding_unknown` (the
    reviewer's `after_retrace` rows). The column is consumed."""
    T.mock_of(rt).object_stop_frac = stall
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    T.arm_after_the_retrace(rt, monkeypatch, lambda: unreadable(rt, monkeypatch, reading))
    res = rt.execute("move_home", {})
    assert res["ok"] is True and res["holding_unknown"] is True and T.at_home(rt), res
    assert rt._held_provisional is not None and rt.held_object is None
    assert stop_retrace.kept(rt) is None


def assert_refused_before_any_command(rt, res, n0, jaws, prov, error=REFUSAL):
    assert res["ok"] is False and res["error"] == error, res
    assert res["outcome"] == "stuck" and res["ask"] == LOOP_ASK, res     # the human's way out, relayed
    assert res["next_action"].startswith("STUCK: relay `ask` to the human verbatim"), res
    assert len(res["error"]) <= 240 and "(truncated; see error)" not in res["next_action"], res
    assert T.commands(rt) == n0 and jaws == []                  # no arm and no jaw command
    assert rt._held_provisional == prov and rt.held_object is None


@pytest.mark.parametrize("gate", [None, "1"], ids=["gate-off", "gate-on"])
@pytest.mark.parametrize("reading", READINGS)
@STALLS
def test_after_move_home_reports_holding_unknown_a_grasp_refuses_before_any_command(
        rebot_real, monkeypatch, stall, reading, gate):
    """The reviewer's row `after_retrace:grasp`: main opened the jaws to 6.2
    at home (z 0.4377 m) with the object possibly in them."""
    rt = rebot_real
    holding_unknown_at_home(rt, monkeypatch, stall, reading)
    if gate is not None:
        monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", gate)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("grasp_object", {"label": "red cube"})
    assert_refused_before_any_command(rt, res, n0, jaws, prov)


@pytest.mark.parametrize("reading", READINGS)
@STALLS
def test_without_a_column_a_grasp_at_the_object_refuses_before_any_command(rebot_real, monkeypatch, stall, reading):
    """The no-column variant: the arm stays where the STOP left it, at the
    object; main opened the jaws there."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = stall
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    assert stop_retrace.forget(rt) == 1
    unreadable(rt, monkeypatch, reading)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("grasp_object", {"label": "red cube"})
    assert_refused_before_any_command(rt, res, n0, jaws, prov)


@pytest.mark.parametrize("reading", ["unknown", "ninf"])
def test_a_column_kept_after_the_stop_is_forfeited_as_by_every_grasp_and_the_refusal_says_so(
        rebot_real, monkeypatch, reading):
    """execute() forfeits a kept column before every motion skill but
    move_home and reset_scene: the grasp's refusal names the plain way out
    (no move_home first that would no longer raise the gripper)."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    assert stop_retrace.live(rt) is not None
    unreadable(rt, monkeypatch, reading)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("grasp_object", {"label": "red cube"})
    assert_refused_before_any_command(rt, res, n0, jaws, prov)
    assert stop_retrace.kept(rt) is None


def direct_pick(rt):
    """pick_and_place called directly (no execute(): a composite's own call),
    its SkillStuck as execute() would name it."""
    with pytest.raises(SkillStuck) as refused:
        SkillRuntime.skill_pick_and_place(rt, "red cube")
    assert refused.value.ask == LOOP_ASK
    return f"SkillStuck: {refused.value}"


@pytest.mark.parametrize("reading", ["unknown", "ninf"])
def test_a_direct_caller_with_the_column_still_kept_is_sent_to_move_home_first(rebot_real, monkeypatch, reading):
    """The move_home-first form, as reset_scene's: with the column still
    kept (no execute() forfeited it), the refusal sends the agent up it
    first and keeps it; the move_home that follows retraces it and reports
    `holding_unknown`, the jaws untouched."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    kept = stop_retrace.kept(rt)
    unreadable(rt, monkeypatch, reading)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    assert direct_pick(rt) == REFUSAL_KEPT
    assert T.commands(rt) == n0 and jaws == [] and rt._held_provisional == prov
    assert stop_retrace.kept(rt) is kept and stop_retrace.live(rt) is kept
    res = rt.execute("move_home", {})
    assert res["ok"] is True and res["stop_retrace"].startswith("completed") and T.at_home(rt), res
    assert res["holding_unknown"] is True and jaws == [] and rt._held_provisional == prov, res


@pytest.mark.parametrize("column", [False, True], ids=["no-column", "column-kept"])
def test_a_long_label_is_cut_never_the_refusal(rebot_real, monkeypatch, column):
    """The refusal and execute()'s "SkillStuck: " fit the agent's 240
    characters, in both forms (the move_home-first one leaves the label the
    least room): a label too long is cut ("..."), the way out and the error
    stay whole."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    if not column:
        stop_retrace.forget(rt)
    rt._held_provisional = ProvisionalHold((T.LONG_LABEL, *tuple(rt._held_provisional)[1:]),
                                           rt._held_provisional.commanded_open, rt._held_provisional.expected_open)
    unreadable(rt, monkeypatch, "unknown")
    n0 = T.commands(rt)
    if column:
        error = direct_pick(rt)
    else:
        res = rt.execute("grasp_object", {"label": "red cube"})
        error = res["error"]
        assert res["ok"] is False and error in res["next_action"], res        # quoted whole
    assert len(error) <= 240 and T.commands(rt) == n0, (len(error), error)
    assert error.startswith("SkillStuck: the jaw state could not be read: ") and "the very very" in error, error
    assert T.LONG_LABEL not in error and "... is in the jaws, take it out by hand" in error, error
    assert error.endswith("try again. Error: no valid jaw feedback"), error
    assert ("call move_home first" in error) is column, error


@pytest.mark.parametrize("prefix", ["", "SkillStuck: "], ids=["reset", "skill-stuck"])
@pytest.mark.parametrize("column", [False, True], ids=["no-column", "column-kept"])
def test_the_shared_refusal_fits_240_with_its_prefix_and_a_long_driver_error(prefix, column):
    """The text reset_scene and the skill entries share, with the longest
    label and driver error: the prefix and the text together fill the 240
    characters (a cut drops the space before it), the sentence whole, the
    label and then the error cut ("...")."""
    from cascade.skills.runtime import _unread_hold_refusal

    error = RuntimeError("CAN bus: gripper motor 7 did not answer the position request " * 5)
    text = _unread_hold_refusal(T.LONG_LABEL, column_kept=column, then="try again", error=error, prefix=prefix)
    assert 238 <= len(prefix + text) <= 240, (len(prefix + text), text)
    assert text.startswith("the jaw state could not be read: ") and "the very very" in text, text
    sentence, tail = text.split(" Error: ")
    assert sentence.endswith("... is in the jaws, take it out by hand, open_gripper, try again." if column
                             else "... is in the jaws, take it out by hand, then open_gripper, then try again."), text
    assert tail.startswith("CAN bus: gripper motor") and tail.endswith("..."), tail


def test_an_unfinished_contact_episode_is_left_to_the_harness(rebot_real, monkeypatch):
    """A contact episode keeps the jaws unread (the reconcile does not judge
    them) and the harness refuses every command until reset_scene's
    recovery: the grasp still refuses with the harness's own text, before
    any command, as on main."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    stop_retrace.forget(rt)
    episode = {"stand-in": True, "halt_generation": rt.arm.harness._halt_generation}
    monkeypatch.setattr(rt, "_contact_episode", episode, raising=False)
    monkeypatch.setattr(rt.arm.harness, "_pending_contact_episode", episode)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("grasp_object", {"label": "red cube"})
    assert res["ok"] is False and "unfinished contact episode; explicit reset_scene recovery required" in res[
        "error"], res
    assert "could not be read" not in res["error"] and T.commands(rt) == n0 and jaws == [], res
    assert rt._held_provisional == prov


@pytest.mark.parametrize("jaws_read", ["closed", "open"])
def test_jaws_that_read_empty_at_the_grasp_drop_the_hold_and_the_grasp_runs_as_on_main(
        rebot_real, monkeypatch, jaws_read):
    """The jaws read again, empty (taken out by hand: closed, or open): the
    grasp entry's reconcile drops the hold and the grasp runs as on main,
    its first jaw command the pre-grasp opening."""
    rt = rebot_real
    holding_unknown_at_home(rt, monkeypatch, 0.5)
    readable_again(rt, monkeypatch, rt._grip_closed if jaws_read == "closed" else rt._grip_open)
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("grasp_object", {"label": "red cube"})
    assert res["ok"] is True and rt.held_object == "red cube" and rt._held_provisional is None, res
    assert jaws[0] == pytest.approx(rt._grip_open) and T.commands(rt) > n0, jaws


def readable_again(rt, monkeypatch, jaw_pos):
    """The jaws read again, at `jaw_pos` (the object taken out by hand); the
    next close meets the cube."""
    raw = T.mock_of(rt)
    monkeypatch.setattr(raw, "get_state", type(raw).get_state.__get__(raw))
    raw.object_stop_frac = None
    raw._gripper = jaw_pos
    raw.object_stop_frac = 0.5


@pytest.mark.parametrize("skill, args, moves", [("sort_by_color", {}, None),     # whatever is left to sort
                                                ("turn_screw", {"label": "red cube"}, True),
                                                ("pick_and_place", {"object": "red cube"}, True),
                                                ("handover", {"label": "red cube"}, True)],
                         ids=["sort_by_color", "turn_screw", "pick_and_place", "handover"])
def test_jaws_that_read_open_let_every_other_entry_run_as_on_main(rebot_real, monkeypatch, skill, args, moves):
    """The entries without a grasp's own reconcile first (sort_by_color,
    turn_screw, handover) judge the hold as that grasp would: open jaws drop
    it, and the skill runs as on main."""
    rt = rebot_real
    holding_unknown_at_home(rt, monkeypatch, 0.5)
    readable_again(rt, monkeypatch, rt._grip_open)
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute(skill, args)
    assert "could not be read" not in str(res.get("error")) and rt._held_provisional is None, res
    if moves:
        assert T.commands(rt) > n0 and len(jaws) > 0, res


#: every other skill whose first jaw command opens the jaws (directly, or
#: through a grasp), with the error execute() reports for it
ENTRIES = {
    "pick_and_place": ({"object": "red cube"}, REFUSAL),
    "grasp_at_pixel": ({"u": 320, "v": 230}, REFUSAL),
    "search_for_object": ({"label": "green ball"}, REFUSAL),
    "restore_scene": ({"name": "scene"}, REFUSAL),
    "sort_by_color": ({}, REFUSAL),
    "turn_screw": ({"label": "red cube"}, REFUSAL),
    "throw": ({"label": "red cube"}, REFUSAL),
    "handover": ({"label": "red cube"}, REFUSAL),
}


@pytest.mark.parametrize("skill", sorted(ENTRIES))
def test_every_skill_that_opens_the_jaws_first_refuses_before_any_command(rebot_real, monkeypatch, skill):
    """From the same state (at home, holding_unknown): pick_and_place (the
    ASPIRE seam: a provisional hold between a grant and its repeat ran main's
    pick), grasp_at_pixel, search_for_object, restore_scene, sort_by_color,
    turn_screw, throw and handover refuse at their entry (or their grasp's),
    STUCK with the way out. handover refuses before its persistence loop:
    the refusal is its error, whole (its loop wrapped it past 240
    characters, #379 review minor 2)."""
    rt = rebot_real
    assert rt.execute("snapshot_scene", {"name": "scene"})["ok"] is not False
    holding_unknown_at_home(rt, monkeypatch, 0.5)
    if skill == "restore_scene":
        displaced(rt)                       # something to move: the restore would grasp
    args, error = ENTRIES[skill]
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute(skill, args)
    assert_refused_before_any_command(rt, res, n0, jaws, prov, error=error)


def displaced(rt, dy=0.06):
    """The memorized layout puts every object `dy` to the side of where it
    is now: restore_scene has something to move."""
    import numpy as np

    for entry in rt.beliefs.get_snapshot("scene").entries:
        entry["position"] = list(np.asarray(entry["position"], dtype=float) + [0.0, dy, 0.0])


#: composites that would grasp nothing from the same state: main ran them
#: (ok, no command); #379 refused them at their entry
NOTHING_TO_GRASP = {
    "restore_scene": {"name": "scene"},                 # nothing displaced
    "search_for_object": {"label": "red cube"},         # already in view
    "sort_by_color": {"max_objects": 0},                # nothing to sort
}


@pytest.mark.parametrize("skill", sorted(NOTHING_TO_GRASP))
def test_a_composite_that_would_grasp_nothing_runs_as_on_main(rebot_real, monkeypatch, skill):
    """#379 review minor 4: restore_scene, search_for_object and sort_by_color
    refuse a possible hold right before their first grasp, not at their
    entry: nothing before it moves the arm or the jaws (perception and
    planning only), so a call that grasps nothing runs as on main, with no
    command, and the hold is left as it was."""
    rt = rebot_real
    assert rt.execute("snapshot_scene", {"name": "scene"})["ok"] is not False
    holding_unknown_at_home(rt, monkeypatch, 0.5)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute(skill, NOTHING_TO_GRASP[skill])
    assert "could not be read" not in str(res.get("error")) and res.get("outcome") != "stuck", res
    assert res["ok"] is True, res
    assert T.commands(rt) == n0 and jaws == [] and rt._held_provisional == prov and rt.held_object is None


def test_handovers_errors_fit_the_agents_240_characters_with_a_long_label(rebot_real, monkeypatch):
    """#379 review minor 2: handover wrapped the 240-character refusal in
    "could not grasp {label!r} for handover: " plus its loop's stop (248
    characters; a long label cut into the sentence). The refusal now comes
    first, whole; a loop that stops after an attempt keeps its error whole
    and gives way only with the label."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    stop_retrace.forget(rt)
    rt._held_provisional = ProvisionalHold((T.LONG_LABEL, *tuple(rt._held_provisional)[1:]),
                                           rt._held_provisional.commanded_open, rt._held_provisional.expected_open)
    unreadable(rt, monkeypatch, "unknown")
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("handover", {"label": T.LONG_LABEL})
    error = res["error"]
    assert res["ok"] is False and res["outcome"] == "stuck" and res["ask"] == LOOP_ASK, res
    assert len(error) <= 240 and error in res["next_action"] and T.commands(rt) == n0 and jaws == [], (len(error), res)
    assert error.startswith("SkillStuck: the jaw state could not be read: if the very very"), error
    assert error.endswith("... is in the jaws, take it out by hand, then open_gripper, then try again. "
                          "Error: no valid jaw feedback"), error
    # A loop that stopped after an attempt: its error whole, the label cut.
    rt._held_provisional = None
    why = f"SkillError: did not settle at grasp lift pose ({LOOP_STOP})"
    monkeypatch.setattr(rt, "_grasp_with_persistence", lambda label, **kw: {
        "ok": False, "held": False, "error": why, "grasp_attempts": 1, "outcome": "stuck", "ask": LOOP_ASK})
    res = rt.execute("handover", {"label": T.LONG_LABEL})
    error = res["error"]
    assert 238 <= len(error) <= 240 and error in res["next_action"], (len(error), error)
    assert error.startswith("could not grasp 'the very very") and error.endswith("... for handover: " + why), error
    assert normalize_failure(error) == "state:jaws_unreadable" and res["ask"] == LOOP_ASK, res
    monkeypatch.setattr(rt, "_grasp_with_persistence", lambda label, **kw: {
        "ok": False, "held": False, "error": why, "grasp_attempts": 1})
    res = rt.execute("handover", {"label": "red cube"})          # a short label: as before
    assert res["error"] == f"could not grasp 'red cube' for handover: {why}", res


def test_turn_screw_never_opens_the_jaws_on_a_held_object(rebot_real, monkeypatch):
    """turn_screw's first jaw command opens the jaws at home; with an object
    held it refuses like every grasp ("already holding"), before any command
    (main dropped the object at home)."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    assert rt.execute("grasp_object", {"label": "red cube"})["ok"] is True and rt.held_object == "red cube"
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("turn_screw", {"label": "red cube"})
    assert res["ok"] is False and res["error"] == "SkillError: already holding 'red cube'; place it first", res
    assert T.commands(rt) == n0 and jaws == [] and rt.held_object == "red cube"


def lose_the_jaws_in_the_lift(rt, monkeypatch, reading):
    """The first attempt's lift (the first carry move while the close's hold
    is live) does not settle, and from then on the jaws cannot be read: the
    attempt fails with the hold surviving. Returns the jaw commands sent and
    [(arm commands, jaw commands)] at the failed lift."""
    real = carry_attachment.move
    jaws, at = T.jaw_commands(rt, monkeypatch), []

    def move(runtime, q, **kwargs):
        if getattr(runtime, "_held_provisional", None) is not None and not at:
            at.append((T.commands(rt), len(jaws)))
            unreadable(rt, monkeypatch, reading)
            return False                                 # "did not settle at grasp lift pose"
        return real(runtime, q, **kwargs)

    monkeypatch.setattr(carry_attachment, "move", move)
    return jaws, at


@pytest.mark.parametrize("reading", ["unknown", "ninf"])
@pytest.mark.parametrize("skill, args", [("pick_and_place", {"object": "red cube"}),
                                         ("handover", {"label": "red cube"})], ids=["pick_and_place", "handover"])
def test_a_persistence_loop_stops_when_a_failed_attempt_leaves_a_possible_hold(
        rebot_real, monkeypatch, skill, args, reading):
    """Mid-loop: the attempt closed, then failed with the jaws unreadable.
    Main's next attempt re-homed and opened the jaws at home. The loop now
    stops (no second grasp, no jaw command after the close), STUCK with the
    way out; the hold stays."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    grasps = []
    real_grasp = rt.skill_grasp_object
    monkeypatch.setattr(rt, "skill_grasp_object", lambda *a, **k: (grasps.append(a), real_grasp(*a, **k))[1])
    jaws, at = lose_the_jaws_in_the_lift(rt, monkeypatch, reading)
    res = rt.execute(skill, args)
    assert len(at) == 1 and len(grasps) == 1, (at, grasps)
    assert len(jaws) == at[0][1], jaws                     # nothing after the close: no opening
    assert res["ok"] is False and res["outcome"] == "stuck" and res["ask"] == LOOP_ASK, res
    assert f"did not settle at grasp lift pose ({LOOP_STOP})" in res["error"], res
    assert rt._held_provisional is not None and rt.held_object is None
    # Filed by what stopped the loop: jaws it cannot read (#377's class),
    # never as an object too wide for them (the old suffix's `jaw` match).
    assert normalize_failure(res["error"]) == "state:jaws_unreadable", res["error"]
    assert failure_signature(res) == "state:jaws_unreadable", res


@pytest.mark.parametrize("error", [
    "grasp failed after 1 attempts: SkillError: gripper closure did not settle; refusing to lift",
    "could not grasp 'red cube' for handover: SkillError: did not settle at grasp lift pose",
    "SkillError: did not settle at grasp lift pose"])
def test_the_loop_stop_files_as_unreadable_jaws_and_keeps_its_ask(error):
    """The review's row: main filed the same miss `other:grasp_failed_after`
    and #379's suffix turned it into `geometry:object_too_wide`. Whatever
    the attempt's own text, the stop files it as jaws that cannot be read,
    and the persistence ask still names the way out."""
    from cascade.skills.runtime import _stuck_ask

    stopped = f"{error} ({LOOP_STOP})"
    assert normalize_failure(stopped) == "state:jaws_unreadable", stopped
    assert _stuck_ask("red cube", stopped, 1) == LOOP_ASK
