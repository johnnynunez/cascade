"""An operator halt during a retry's re-home ends the whole skill.

The persistence loops (`_grasp_with_persistence` behind handover and
sort_by_color, pick_and_place's grasp and place loops) re-home between
attempts and swallowed every `SkillError` / `SafetyViolation` of that move,
`MotionHalted` included. The next attempt's first motion carries no halt
generation, so it CLEARED the halt and the arm went on after an operator
STOP: on fr3 25-237 arm commands after the halt, on rebot_rs 125-200 (main
too). A halt raised in the re-home now propagates (`except MotionHalted:
raise` before the swallow), and so do the composites that loop over
`_grasp_with_persistence` (sort_by_color, `_move_object_to` behind
restore_scene / search_for_object, including its clear-view homes).

An e-stop is not a halt: on the fr3 view-home return it stays
`SafetyViolation: e-stop latched`, never reported as `MotionHalted`.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that counts every streamed command; the grasp and
place primitives are replaced by fakes that move the arm and miss.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.agent.aspire import diagnose
from cascade.control.mock_arm import MockArm
from cascade.types import MotionHalted, SkillError

HALT = "halted: operator: stop"
RETRY_PROFILES = ["fr3", "rebot_rs"]


class _SendArm(MockArm):
    """MockArm that counts every streamed command (also refused-by-driver
    ones) and runs `r.on_send()` after each."""

    def __init__(self, cfg, kinematics, r):
        super().__init__(cfg, kinematics)
        self.r = r

    def send_joint_target(self, q):
        self.r.sends += 1
        super().send_joint_target(q)
        if self.r.on_send is not None:
            self.r.on_send()


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    # Observed-finger gate off: with it on, the grasp loops return after the
    # first failed attempt and never re-home (the place loop still does:
    # its test runs both ways).
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)

    def build(profile):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        # rebot_rs requires its live camera map and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        r = SimpleNamespace(cfg=cfg, sends=0, on_send=None, arms=[], grasps=0, places=0,
                            home_start=None, at_stop=None)

        def make_arm(acfg, kinematics=None):
            arm = _SendArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _stop_now(r, kind):
    harness = r.runtime.arm.harness
    if kind == "halt":
        harness.halt("operator: stop")
    elif kind == "estop_harness":
        harness.estop("operator e-stop")
    else:  # "estop_safearm": SafeArm.stop() latches the e-stop and stops the driver
        r.runtime.arm.stop()
    r.at_stop = r.sends


def _stop_in_rehome(r, monkeypatch, kind="halt", waypoint=5, nth_home=1):
    """Stop (halt / e-stop) at the `waypoint`-th command of the `nth_home`
    skill_move_home call (the loop's re-home)."""
    runtime = r.runtime
    home = runtime.skill_move_home
    calls = []

    def spy_home(**kwargs):
        calls.append(kwargs)
        if len(calls) == nth_home:
            r.home_start = r.sends
        return home(**kwargs)

    def on_send():
        if r.at_stop is None and r.home_start is not None and r.sends - r.home_start >= waypoint:
            _stop_now(r, kind)

    monkeypatch.setattr(runtime, "skill_move_home", spy_home)
    r.on_send = on_send


def _move_off(runtime, step):
    q = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    q[0] += step
    assert runtime.arm.move_joints(q, duration_s=1.5)


def _missing_grasp(r, monkeypatch, *, hold_on=None):
    """A grasp that moves the arm and misses; attempt `hold_on` holds."""
    runtime = r.runtime

    def grasp(object, material=None, **kwargs):
        r.grasps += 1
        _move_off(runtime, 0.15 if r.grasps % 2 else -0.15)
        if hold_on is not None and r.grasps >= hold_on:
            runtime.held_object = object
            runtime._held_det_label = "cube"
            return {"held": object, "grip_verified": True, "grip_profile": None}
        return {"ok": False, "error": "grasp missed: closed on air"}

    monkeypatch.setattr(runtime, "skill_grasp_object", grasp)


def _missing_place(r, monkeypatch):
    runtime = r.runtime

    def place(x, y, **kwargs):
        r.places += 1
        _move_off(runtime, 0.15)
        raise SkillError("place missed: target not reachable")

    monkeypatch.setattr(runtime, "skill_place_at", place)


def _assert_halted_result(r, out):
    assert r.at_stop is not None, "the retry never re-homed"
    assert r.sends == r.at_stop, f"{r.sends - r.at_stop} arm commands went out after the halt"
    assert out["ok"] is False, out
    assert out["error"] == f"MotionHalted: {HALT}", out
    assert out["next_action"].count("This attempt FAILED") == 1, out["next_action"]
    assert not r.runtime.arm.harness.estopped, "the halt was escalated to an e-stop"


@pytest.mark.parametrize("profile", RETRY_PROFILES)
@pytest.mark.parametrize("skill, args", [
    ("handover", {"label": "red cube"}),
    ("pick_and_place", {"object": "red cube"}),
], ids=["handover", "pick_and_place"])
def test_a_halt_in_the_grasp_retry_rehome_ends_the_skill(rig, monkeypatch, profile, skill, args):
    r = rig(profile)
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    out = r.runtime.execute(skill, args)

    _assert_halted_result(r, out)
    assert r.grasps == 1, "the loop tried again after the halt"


@pytest.mark.parametrize("gate", ["off", "on"])
@pytest.mark.parametrize("profile", RETRY_PROFILES)
def test_a_halt_in_the_place_retry_rehome_ends_pick_and_place(rig, monkeypatch, profile, gate):
    if gate == "on":
        monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    r = rig(profile)
    _missing_grasp(r, monkeypatch, hold_on=1)
    _missing_place(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    out = r.runtime.execute("pick_and_place", {"object": "red cube"})

    _assert_halted_result(r, out)
    assert r.places == 1, "the place loop tried again after the halt"
    assert r.runtime.held_object == "red cube"  # still in the jaws: nothing released it


@pytest.mark.parametrize("profile", RETRY_PROFILES)
def test_grasp_with_persistence_raises_a_halt_from_its_rehome(rig, monkeypatch, profile):
    r = rig(profile)
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    with pytest.raises(MotionHalted) as raised:
        r.runtime._grasp_with_persistence("red cube")

    assert str(raised.value) == HALT
    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the halt"
    assert r.grasps == 1


def _halt_around_rehome(r, monkeypatch, where):
    """Halt around the persistence loop's FIRST re-home: on entry to it, at
    the view-home return's status line (fr3), right after it, or in the
    re-scan that follows it."""
    runtime = r.runtime
    home, reobserve, status = runtime.skill_move_home, runtime._reobserve, runtime._show_status
    homes = []

    def spy_home(**kwargs):
        homes.append(1)
        if where == "entry" and r.at_stop is None:
            _stop_now(r, "halt")
        out = home(**kwargs)
        if where == "after" and r.at_stop is None:
            _stop_now(r, "halt")
        return out

    def spy_reobserve(*a, **k):
        if where == "rescan" and homes and r.at_stop is None:
            _stop_now(r, "halt")
        return reobserve(*a, **k)

    def spy_status(text):
        if where == "status" and homes and r.at_stop is None and text.startswith("Returning to the saved"):
            _stop_now(r, "halt")
        return status(text)

    monkeypatch.setattr(runtime, "skill_move_home", spy_home)
    monkeypatch.setattr(runtime, "_reobserve", spy_reobserve)
    monkeypatch.setattr(runtime, "_show_status", spy_status)


AROUND_REHOME = [(p, w) for p in RETRY_PROFILES + ["mock"] for w in ("entry", "status", "after", "rescan")
                 if w != "status" or p == "fr3"]
#: the re-home's own refusal: SafeArm refuses its first command
REFUSED = "halt received during route planning or execution"
#: the loop's check before its next attempt (`_end_on_halt`)
NOT_RETRYING = "motion halted: operator: stop; not retrying"


def _around_rehome_halt_text(profile, where):
    """What ends the loop for a halt at `where` around its first re-home.

    A halt on entry to a direct home move (rebot_rs, mock) or at fr3's
    view-home status line (after its skip checks) is refused by that move.
    fr3's skip checks find a halt raised on entry, so its view-home return is
    skipped (a SkillError the loop swallows); that halt, and one raised after
    the re-home or during the re-scan, ends the loop before its next attempt."""
    if where == "status" or (where == "entry" and profile != "fr3"):
        return REFUSED
    return NOT_RETRYING


@pytest.mark.parametrize("profile, where", AROUND_REHOME, ids=[f"{p}-{w}" for p, w in AROUND_REHOME])
def test_grasp_with_persistence_called_directly_ends_on_a_halt_around_its_rehome(rig, monkeypatch, profile,
                                                                                 where):
    """Called outside execute(), the loop holds the same skill halt baseline
    execute() takes (a no-op inside a skill) and checks its generation before
    every further attempt: a halt on entry to its re-home, one the re-home's
    skip checks find, one raised after it or during the re-scan ends the loop
    with no further arm command."""
    r = rig(profile)
    _missing_grasp(r, monkeypatch)
    _halt_around_rehome(r, monkeypatch, where)

    with pytest.raises(MotionHalted) as raised:
        r.runtime._grasp_with_persistence("red cube")

    assert str(raised.value) == _around_rehome_halt_text(profile, where)
    assert r.at_stop is not None, "the halt was never raised"
    assert r.sends == r.at_stop, f"{r.sends - r.at_stop} arm commands went out after the halt"
    assert r.grasps == 1, "the loop tried again after the halt"
    assert not r.runtime.arm.harness.estopped


def _two_beliefs(runtime, monkeypatch):
    beliefs = [SimpleNamespace(label="cube", color=color, position=np.array([0.32, y, 0.02]))
               for color, y in (("red", 0.12), ("blue", -0.12))]
    monkeypatch.setattr(runtime.beliefs, "all", lambda: list(beliefs))


@pytest.mark.parametrize("profile", RETRY_PROFILES)
def test_a_halt_ends_sort_by_color_not_just_one_object(rig, monkeypatch, profile):
    """sort_by_color went on with the NEXT object after a halt in the first
    one's persistence loop; that object's first grasp cleared the halt."""
    r = rig(profile)
    _two_beliefs(r.runtime, monkeypatch)
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch)

    out = r.runtime.execute("sort_by_color", {})

    _assert_halted_result(r, out)
    assert r.grasps == 1, "the sort went on to the next object after the halt"


@pytest.mark.parametrize("profile", RETRY_PROFILES)
@pytest.mark.parametrize("where", ["grasp_rehome", "home_after_a_miss", "home_after_the_place"])
def test_a_halt_ends_move_object_to(rig, monkeypatch, profile, where):
    """`_move_object_to` is the step of restore_scene's and
    search_for_object's loops: a halt in its grasp loop or in its clear-view
    home must reach them, or their next step moves the arm again."""
    r = rig(profile)
    runtime = r.runtime
    if where == "grasp_rehome":
        _missing_grasp(r, monkeypatch)
        _stop_in_rehome(r, monkeypatch)                 # the persistence loop's re-home
    elif where == "home_after_a_miss":
        runtime.cfg._data.setdefault("grasp", {})["max_pick_attempts"] = 1
        _missing_grasp(r, monkeypatch)
        _stop_in_rehome(r, monkeypatch)                 # its clear-view home after the miss
    else:
        _missing_grasp(r, monkeypatch, hold_on=1)

        def place(x, y, **kwargs):
            r.places += 1
            _move_off(runtime, 0.15)
            runtime.held_object = None
            return {"ok": True, "at": [x, y]}

        monkeypatch.setattr(runtime, "skill_place_at", place)
        _stop_in_rehome(r, monkeypatch)                 # its clear-view home after the place

    with pytest.raises(MotionHalted) as raised:
        runtime._move_object_to("red cube", [0.30, 0.10], budget_s=30.0, kind="restore")

    assert str(raised.value) == HALT
    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the halt"
    assert r.grasps == 1


@pytest.mark.parametrize("kind", ["estop_harness", "estop_safearm"])
def test_an_estop_in_the_fr3_retry_rehome_stays_an_estop(rig, monkeypatch, kind):
    """An e-stop during the view-home re-home is the latched e-stop, never
    reported as an operator halt."""
    r = rig("fr3")
    _missing_grasp(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch, kind=kind)

    out = r.runtime.execute("handover", {"label": "red cube"})

    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the e-stop"
    assert out["ok"] is False, out
    assert "SafetyViolation: e-stop latched" in out["error"], out
    assert "MotionHalted" not in out["error"], out
    assert r.runtime.arm.harness.estopped


# ── a halt in the home move after a failed skill is named in its error ──────

#: pick_and_place's error after one missed attempt (its duration scrubbed)
PICK_MISSED = "grasp failed after 1 attempts (<t>s): grasp missed: closed on air"
#: handover's error after one missed attempt
HANDOVER_MISSED = "could not grasp 'red cube' for handover: grasp missed: closed on air"


def _scrub_time(error):
    return re.sub(r"\(\d+(\.\d+)?s\)", "(<t>s)", error)


def _filed_as(r, skill):
    """How this call's failure was filed: its envelope signatures (the
    agent's `failure_digest`), the envelope's operator halts (not losses)
    and aspire's diagnosis of the run's trace."""
    stats = r.runtime.envelope.stats()[skill]
    return stats["failures"], stats["halts"], diagnose(r.runtime.trace.run_dir).signature


def _one_marked_miss(r, monkeypatch):
    """One grasp attempt that moves the arm through the runtime (so execute()
    counts it as this skill's motion) and misses."""
    runtime = r.runtime
    runtime.cfg._data.setdefault("grasp", {})["max_pick_attempts"] = 1
    _missing_grasp(r, monkeypatch)
    miss = runtime.skill_grasp_object

    def marked_miss(*a, **k):
        runtime._mark_motion_started()
        return miss(*a, **k)
    monkeypatch.setattr(runtime, "skill_grasp_object", marked_miss)


@pytest.mark.parametrize("kind", ["halt", "estop_harness", "estop_safearm"])
@pytest.mark.parametrize("profile", ["rebot_rs", "mock"])
def test_a_failed_pick_and_place_park_error_names_a_halt_but_not_an_estop(rig, monkeypatch, profile, kind):
    """After its last missed attempt pick_and_place parks the arm (rigs
    without the view-home return). A halt that stops that park is named
    first in the error, before the grasp failure it keeps; the call is
    filed as that grasp failure (`skill_error`), which came first, as on an
    e-stop. An e-stop there (the harness's, or `SafeArm.stop()`) is not the
    operator's halt: the error stays the grasp failure and the e-stop stays
    latched."""
    r = rig(profile)
    _one_marked_miss(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch, kind=kind)      # the 1st home move: the final park

    out = r.runtime.execute("pick_and_place", {"object": "red cube"})

    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the stop"
    assert out["ok"] is False and out["stage"] == "grasp", out
    assert _filed_as(r, "pick_and_place") == ({"other:grasp_failed_after": 1}, 0, "other:grasp_failed_after")
    if kind == "halt":
        assert _scrub_time(out["error"]) == f"final park {HALT}; {PICK_MISSED}", out["error"]
        assert _scrub_time(out["skill_error"]) == PICK_MISSED, out
        assert not r.runtime.arm.harness.estopped
    else:
        assert _scrub_time(out["error"]) == PICK_MISSED, out["error"]
        assert "skill_error" not in out, out
        assert r.runtime.arm.harness.estopped


def test_a_failed_pick_and_place_park_without_a_halt_keeps_its_error(rig, monkeypatch):
    r = rig("rebot_rs")
    _one_marked_miss(r, monkeypatch)

    out = r.runtime.execute("pick_and_place", {"object": "red cube"})

    assert _scrub_time(out["error"]) == PICK_MISSED, out["error"]


def _stop_in_view_home_return(r, monkeypatch, where, kind="halt"):
    """Stop (halt / e-stop) in execute()'s post-skill view-home return (fr3):
    on `entry` (before its skip checks) or at its 2nd `route` command."""
    runtime = r.runtime
    back = runtime._return_to_view_home

    def spy_return(*a, **k):
        r.home_start = r.sends
        if where == "entry":
            _stop_now(r, kind)
        return back(*a, **k)

    def on_send():
        if where == "route" and r.at_stop is None and r.home_start is not None \
                and r.sends - r.home_start >= 2:
            _stop_now(r, kind)
    monkeypatch.setattr(runtime, "_return_to_view_home", spy_return)
    r.on_send = on_send


#: the return's own text, as execute() says it for a skill that succeeded
VIEW_HOME_HALTED = {"route": f"return to saved home {HALT}",
                    "entry": "return to saved home skipped: motion halted: operator: stop"}


@pytest.mark.parametrize("where", sorted(VIEW_HOME_HALTED))
@pytest.mark.parametrize("skill, args, missed, signature", [
    ("pick_and_place", {"object": "red cube"}, PICK_MISSED, "other:grasp_failed_after"),
    ("handover", {"label": "red cube"}, HANDOVER_MISSED, "other:could_not_grasp"),
], ids=["pick_and_place", "handover"])
def test_a_halt_in_fr3_view_home_return_after_a_failed_skill_is_named_in_its_error(
        rig, monkeypatch, skill, args, missed, signature, where):
    """fr3 returns to its camera-view home after every motion skill. When the
    skill failed and the operator's halt stops (or skips) that return, the
    error names the halt first, before the skill's own failure, not only in
    `return_home.status`. The call is filed as the skill's own failure
    (`skill_error`): it came before the halt."""
    r = rig("fr3")
    _one_marked_miss(r, monkeypatch)
    _stop_in_view_home_return(r, monkeypatch, where)

    out = r.runtime.execute(skill, args)

    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the halt"
    assert out["ok"] is False, out
    assert out["return_home"]["status"] == ("halted" if where == "route" else "skipped"), out["return_home"]
    assert _scrub_time(out["error"]) == f"{VIEW_HOME_HALTED[where]}; {missed}", out["error"]
    assert _filed_as(r, skill) == ({signature: 1}, 0, signature)
    assert _scrub_time(out["skill_error"]) == missed, out
    assert not r.runtime.arm.harness.estopped


@pytest.mark.parametrize("stop", ["none", "e-stop"])
def test_fr3_view_home_return_after_a_failed_skill_without_a_halt_keeps_its_error(rig, monkeypatch, stop):
    """A return that completed, or one an e-stop skipped (an e-stop is not
    the operator's halt), leaves the failed skill's error as it is."""
    r = rig("fr3")
    _one_marked_miss(r, monkeypatch)
    if stop == "e-stop":
        _stop_in_view_home_return(r, monkeypatch, "entry", kind="estop_harness")

    out = r.runtime.execute("pick_and_place", {"object": "red cube"})

    assert out["return_home"]["status"] == ("returned" if stop == "none" else "skipped"), out["return_home"]
    assert _scrub_time(out["error"]) == PICK_MISSED, out["error"]
    assert _filed_as(r, "pick_and_place") == ({"other:grasp_failed_after": 1}, 0, "other:grasp_failed_after")


def test_a_halt_inside_an_fr3_skill_is_not_named_again_by_its_return(rig, monkeypatch):
    """Only a halt that came during the view-home return is added to the
    error. One that a grasp swallowed skips the return too; the grasp held,
    so execute() already reports the halt (`MotionHalted: ...`), once."""
    r = rig("fr3")
    runtime = r.runtime

    def grasp(label, **kwargs):
        runtime._mark_motion_started()
        _move_off(runtime, 0.15)
        _stop_now(r, "halt")
        runtime.held_object, runtime._held_det_label = label, "cube"
        return {"held": label, "grip_verified": True}
    monkeypatch.setattr(runtime, "skill_grasp_object", grasp)

    out = runtime.execute("grasp_object", {"label": "red cube"})

    assert r.sends == r.at_stop, "an arm command went out after the halt"
    assert out["return_home"]["reason"] == "motion halted: operator: stop", out["return_home"]
    assert out["error"] == f"MotionHalted: {HALT}", out["error"]
    assert "skill_error" not in out, out
    assert _filed_as(r, "grasp_object") == ({}, 1, "safety:halt")


# ── the stop leads a long error: the agent's FAILED line keeps 240 characters ──

#: the real air-grasp text of a grasp whose reopen could not read the jaws
AIR_UNREAD = ("air grasp: gripper closed fully, object not held; the jaws were not confirmed open: "
              "jaw state could not be read: gripper feedback timed out")
#: (profile, skill, args, the skill's failure, the stop that ends it): a halt
#: on entry to the final park (rebot_rs) or to fr3's view-home return
LONG = [
    ("rebot_rs", "pick_and_place", {"object": "red cube"},
     f"grasp failed after 1 attempts (<t>s): {AIR_UNREAD}", f"final park halted: {REFUSED}"),
    ("fr3", "handover", {"label": "red cube"},
     f"could not grasp 'red cube' for handover: {AIR_UNREAD}",
     "return to saved home skipped: motion halted: operator: stop"),
]


def _one_air_grasp_unread_reopen(r, monkeypatch):
    """One attempt that moves the arm and closes on air; its reopen could not
    read the jaws (the grasp result's real keys and texts)."""
    runtime = r.runtime
    runtime.cfg._data.setdefault("grasp", {})["max_pick_attempts"] = 1

    def grasp(*a, **k):
        r.grasps += 1
        runtime._mark_motion_started()
        _move_off(runtime, 0.15)
        return {"ok": False, "error": AIR_UNREAD,
                "jaws_reopen": "refused: jaw state could not be read: gripper feedback timed out",
                "exit_withdrawal": "skipped: the jaws were not confirmed open"}
    monkeypatch.setattr(runtime, "skill_grasp_object", grasp)


def _stop_on_entry(r, monkeypatch, name):
    """Halt on entry to the runtime's `name` (the final park's
    skill_move_home, or execute()'s view-home return)."""
    move = getattr(r.runtime, name)

    def spy(*a, **k):
        if r.at_stop is None:
            _stop_now(r, "halt")
        return move(*a, **k)
    monkeypatch.setattr(r.runtime, name, spy)


@pytest.mark.parametrize("profile, skill, args, failure, stop", LONG, ids=[p for p, *_ in LONG])
def test_the_stop_leads_a_long_failed_skill_error(rig, monkeypatch, profile, skill, args, failure, stop):
    """The stop comes first, so the FAILED line keeps it when the error is
    longer than the 240 characters that line keeps (appended at the end it
    was cut off). The call is filed as the skill's own air grasp."""
    r = rig(profile)
    _one_air_grasp_unread_reopen(r, monkeypatch)
    _stop_on_entry(r, monkeypatch, "skill_move_home" if profile == "rebot_rs" else "_return_to_view_home")

    out = r.runtime.execute(skill, args)

    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the halt"
    assert out["ok"] is False, out
    assert _scrub_time(out["error"]) == f"{stop}; {failure}", out["error"]
    assert len(out["error"]) > 240, out["error"]
    kept = out["next_action"].split(" (truncated; see error)")
    assert len(kept) == 2 and f"This attempt FAILED: {stop}; " in kept[0], out["next_action"]
    assert _filed_as(r, skill) == ({"contact:air_grasp": 1}, 0, "contact:air_grasp")
    assert _scrub_time(out["skill_error"]) == failure, out
    assert not r.runtime.arm.harness.estopped


def test_a_failed_skill_with_no_error_of_its_own_gets_the_stop_alone(rig, monkeypatch):
    """A skill that failed without an error text: the error is the stop."""
    r = rig("fr3")
    runtime = r.runtime

    def grasp(*a, **k):
        runtime._mark_motion_started()
        _move_off(runtime, 0.15)
        return {"ok": False}
    monkeypatch.setattr(runtime, "skill_grasp_object", grasp)
    _stop_on_entry(r, monkeypatch, "_return_to_view_home")

    out = runtime.execute("grasp_object", {"label": "red cube"})

    assert r.at_stop is not None and r.sends == r.at_stop, "an arm command went out after the halt"
    assert out["ok"] is False and out["error"] == VIEW_HOME_HALTED["entry"], out
    assert "skill_error" not in out, out
    assert _filed_as(r, "grasp_object") == ({}, 1, "safety:halt")
