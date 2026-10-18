"""reset_scene on a real gripper reconciles a provisional hold before any motion.

From the instant a grasp commands its close, the jaws may hold the object
while `held_object` is still None: only the close's provisional marker
(`_held_provisional`) says so. Anything that ends the grasp after the close
and before its `held_object = label` line leaves that marker alone: an
operator STOP right after the close or anywhere in the lift, a lift that did
not settle, a contact in the lift. reset_scene on a real arm checked
`held_object` alone, so it homed with the object in the jaws, sent no jaw
command, forgot every belief and returned ok; a later open_gripper dropped
the object from home height, the case #357 refuses. Now the real-gripper path
classifies the jaws first, exactly like the grasp's own verify (`_air_grasp`
against the opening the close commanded, which the marker carries):

- jaws stalled on something: the hold is promoted and the reset refuses
  with #357's text ("holding X: place it first"), before any command;
- jaws nearly shut, open, or within 3 % of travel of the commanded seat
  while the plan sized the object at least 7 % wider (the reBot's rigid
  close on air stops at its seat, 85 % of travel, 0.93 rad): nothing is
  held, the marker is dropped and the reset runs as with empty jaws (closed
  ones get their one opening at home). The price, as in the grasp's verify:
  an object about 13.5-16.2 mm thick that the plan sized at 19.8 mm or more
  stalls the jaws in that band and is opened at home;
- jaws that cannot be read (a failed read, unknown feedback, a reading
  that is no number): the reset refuses too, before any command, and says
  how to get out (take the object out by hand, open_gripper, reset_scene);
  a marker it cannot judge against the reading, or an unfinished contact
  episode (the jaws are never read), refuses with its own text. A marker
  without its close's opening (a VLA close) is judged by the old rule:
  anything short of closed is a hold.

A held flag is not reconciled: it refuses as before (#357). A grasp's entry
keeps its own reconcile (fail closed at the seat).

Simulated and mock arms are unchanged.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every streamed arm command and every jaw
command, and the real grasp on the mock camera; no driver, no bus.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest

from cascade.control.mock_arm import MockArm
from cascade.skills import stop_retrace
from cascade.skills.runtime import ProvisionalHold
from cascade.types import MotionHalted

REFUSAL = "holding red cube: place it first (place_at), then reset"
UNREAD = ("the jaw state could not be read: if red cube is in the jaws, take it out by hand, then "
          "open_gripper, then reset_scene. Error: ")
#: the jaws were read, but the marker could not be judged against them
UNJUDGED = ("the jaw reading could not be judged: if red cube is in the jaws, take it out by hand, then "
            "open_gripper, then reset_scene. Error: ")
#: an unfinished contact episode: the reconcile never reads the jaws
CONTACT = ("the grasp of red cube ended in an unfinished contact episode, so the jaws were not read and it "
           "may be in them: a person must check the gripper.")
#: the way-out sentence a refusal for an unread or unjudged hold must keep whole
WAY_OUT = " is in the jaws, take it out by hand, then open_gripper, then reset_scene."
#: the halt lands on this streamed command of the grasp lift (~100 commands
#: on rebot_rs): late in the lift, the jaws long stalled on the object
LATE_IN_LIFT = 90
#: where the operator's STOP lands: before the lift's first command (right
#: after the close) or late in the lift
WHERE = {"after_close": 0, "late_in_lift": LATE_IN_LIFT}
#: the reBot's rigid seat: stage 2 closes to 85 % of travel, 15 % open
REBOT_SEAT_RAD = 6.2 * 0.15


class _RecArm(MockArm):
    """MockArm that logs every streamed arm command and every jaw command,
    runs `r.on_send()` after each arm command, and reads the jaws as unknown
    (`jaws = "unknown"`) or fails the whole read (`jaws = "error"`, with
    `error` as the driver's text)."""

    def __init__(self, cfg, kinematics, r):
        super().__init__(cfg, kinematics)
        self.r = r
        self.jaws = "ok"
        self.error = "gripper feedback timed out"

    def send_joint_target(self, q):
        self.r.log.append(("arm",))
        super().send_joint_target(q)
        if self.r.on_send is not None:
            self.r.on_send()

    def set_gripper(self, pos, effort=1.0):
        self.r.log.append(("jaw", float(pos), float(effort)))
        super().set_gripper(pos, effort)

    def get_state(self):
        if self.jaws == "error":
            raise TimeoutError(self.error)
        state = super().get_state()
        if self.jaws == "nan":                     # a "valid" reading that is no number
            return dataclasses.replace(state, gripper_valid=True, gripper_pos=float("nan"))
        return dataclasses.replace(state, gripper_valid=False) if self.jaws == "unknown" else state


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)

    def build(profile, **gripper):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        # rebot_rs requires its live camera map and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        cfg._data["arm"]["gripper"].update(gripper)
        r = SimpleNamespace(cfg=cfg, log=[], arms=[], on_send=None)

        def make_arm(acfg, kinematics=None):
            arm = _RecArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _stop_in_the_grasp(r, *, stall, where="late_in_lift"):
    """The real grasp of the red cube, halted by the operator `where` says:
    right after the close (the lift refused before its first command) or on
    the LATE_IN_LIFT-th command of the lift. `stall`: the jaws stall at that
    fraction of their travel (on the object), None: they close as far as
    commanded (on air). Clears the log."""
    runtime = r.runtime
    runtime.arm.get_state()                      # materialize the LazyArm
    r.arms[0].object_stop_frac = stall
    carry = runtime._move_carry
    after = WHERE[where]
    lift, stop = [], []

    def halt():
        runtime.arm.harness.halt("operator: stop")
        stop.append(len(r.log))

    def lift_spy(*args, **kwargs):
        lift.append(len(r.log))
        if after == 0:
            halt()
        return carry(*args, **kwargs)

    def on_send():
        if lift and not stop and len(r.log) - lift[0] >= after:
            halt()

    runtime._move_carry = lift_spy
    r.on_send = on_send
    out = runtime.execute("grasp_object", {"label": "red cube"})
    r.on_send = None
    del runtime._move_carry
    assert stop and len(lift) == 1, "the halt never reached the lift"
    assert out["ok"] is False and MotionHalted.__name__ in out["error"], out
    assert r.log[stop[0]:] == [], "commands after the halt"
    assert runtime.held_object is None and runtime._held_provisional is not None
    # #359 keeps the approach column after such a STOP (rebot_rs); these tests
    # pin the reset's own rules once it is gone (expired, or given up by
    # another motion skill): tests/test_stop_retrace.py pins the refusal
    # texts while it is kept (move_home first).
    stop_retrace.forget(runtime)
    r.log.clear()


def _reset(r, call):
    return r.runtime.execute("reset_scene", {}) if call == "execute" else r.runtime.skill_reset_scene()


def _refused(out, error):
    assert out["ok"] is False and out["stage"] == "release" and out["home_skipped"] is True, out
    assert out["beliefs_forgotten"] == 0 and not out["observation_refreshed"], out
    assert out["error"].startswith(error), out


def _ran_as_for_empty_jaws(r, out, *, opened):
    """The reset ran with nothing held: arm commands home and, for jaws that
    read closed, the reset's one opening at home as its last command."""
    assert out["ok"], out
    jaws = [e[1:] for e in r.log if e[0] == "jaw"]
    if opened:
        assert out["jaws_open_commanded"] is True, out
        assert jaws == [(pytest.approx(r.runtime._grip_open), 0.8)], jaws
        assert r.log[-1][0] == "jaw" and len(r.log) > 1  # home first, then the opening
    else:
        assert "jaws_open_commanded" not in out and jaws == [], (out, jaws)
        assert r.log, "the reset never homed"
    assert r.runtime.held_object is None and r.runtime._held_provisional is None


@pytest.mark.parametrize("stall", [0.5, 0.8])
@pytest.mark.parametrize("where", sorted(WHERE))
@pytest.mark.parametrize("call", ["execute", "direct"])
def test_jaws_stalled_short_of_the_seat_after_a_stop_refuse_the_reset(rig, call, where, stall):
    """0.8 of travel closed (20 % open) is the widest reading short of the
    rigid seat's tolerance (15 % + 3 %): the object stopped the jaws."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=stall, where=where)
    out = _reset(r, call)
    _refused(out, REFUSAL)
    assert out["error"] == REFUSAL
    assert r.log == [], r.log                      # no arm, no jaw command
    rt = r.runtime
    assert rt.held_object == "red cube" and rt._held_provisional is None    # promoted
    assert rt._held_width_m == pytest.approx((1.0 - stall) * rt._max_width)


@pytest.mark.parametrize("stall", [None, 0.83])
@pytest.mark.parametrize("where", sorted(WHERE))
@pytest.mark.parametrize("call", ["execute", "direct"])
def test_jaws_at_the_commanded_seat_after_a_stop_hold_nothing_and_the_reset_runs(rig, call, where, stall):
    """The reBot's rigid close on AIR stops at its stage-2 seat (85 % of
    travel: 0.93 rad, 15 % open), not at `closed_pos`. The grasp's own verify
    calls that air (`_air_grasp`: at the commanded opening + 3 %, the cube
    should have stopped them much wider); the reset judges it the same way:
    nothing is held, the marker is dropped, the reset runs and the closed
    empty jaws get their one opening at home. 0.83 = 17 % open, inside the
    seat's tolerance."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=stall, where=where)
    seat = REBOT_SEAT_RAD if stall is None else 6.2 * (1.0 - stall)
    assert r.arms[0].get_state().gripper_pos == pytest.approx(seat)
    out = _reset(r, call)
    _ran_as_for_empty_jaws(r, out, opened=True)


def test_a_marker_without_its_close_opening_keeps_the_old_rule(rig):
    """A provisional hold that does not carry its close's commanded opening
    (the VLA executor's marker) cannot be judged against a seat: anything
    short of closed counts as a hold (fail closed), as before."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=None)
    r.runtime._held_provisional = tuple(r.runtime._held_provisional)     # a plain marker
    out = r.runtime.execute("reset_scene", {})
    _refused(out, REFUSAL)
    assert r.log == [] and r.runtime.held_object == "red cube", r.log


def test_a_grasp_entry_keeps_its_rule_at_the_seat(rig):
    """Only reset_scene judges an interrupted grasp's jaws against the seat:
    a grasp's entry keeps its fail-closed reconcile, which promotes jaws at
    the seat and refuses before any command (open_gripper clears it)."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=None)
    out = r.runtime.execute("grasp_object", {"label": "red cube"})
    assert out["ok"] is False and "already holding 'red cube'" in out["error"], out
    assert r.log == [] and r.runtime.held_object == "red cube", r.log


def test_after_a_post_contact_hold_the_seat_is_the_hold_opening(rig):
    """A close that held at the contact width minus a squeeze (B36,
    `gripper.hold_squeeze_frac`) last commanded the HOLD opening: the grasp's
    verify judges against it, and so does the reset. The object slipped out
    after the STOP: the jaws sit at the hold opening (25 % open, wider than
    the stage-2 seat + 3 %), which is air for that close."""
    r = rig("rebot_rs", hold_squeeze_frac=0.05)
    _stop_in_the_grasp(r, stall=0.7)               # stalls 30 % open, holds at 25 %
    arm = r.arms[0]
    assert arm.get_state().gripper_pos == pytest.approx(6.2 * 0.3)
    arm.object_stop_frac = None                    # the object slipped out ...
    arm.set_gripper(6.2 * 0.25)                    # ... the jaws closed to the hold target
    r.log.clear()
    out = r.runtime.execute("reset_scene", {})
    _ran_as_for_empty_jaws(r, out, opened=True)


@pytest.mark.parametrize("jaws, why", [("unknown", "no valid jaw feedback"),
                                       ("error", "gripper feedback timed out")])
@pytest.mark.parametrize("call", ["execute", "direct"])
def test_unreadable_jaws_after_a_stop_refuse_the_reset_and_name_the_way_out(rig, call, jaws, why):
    """Fail closed: the jaws may hold the object; nothing moves. On the reBot
    persistently unknown jaws lost their turn reference, and only a jaw
    command restores it, by closing first: the text says to take the object
    out by hand, then open_gripper, then reset_scene."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.arms[0].jaws = jaws
    out = _reset(r, call)
    _refused(out, UNREAD)
    assert out["error"] == UNREAD + why
    assert len(out["error"]) <= 240, out["error"]  # the agent's failure line keeps 240 characters
    assert r.log == [], r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is not None


@pytest.mark.parametrize("jaws, error, signature", [("ok", REFUSAL, "state:holding"),
                                                     ("unknown", UNREAD, "state:jaws_unreadable")])
def test_a_reset_refused_for_the_jaws_is_filed_by_their_state_not_the_object(rig, jaws, error, signature):
    """The envelope (the agent's failure digest) and aspire key a failure by
    its signature: these refusals file under one stable state signature,
    never the object's name (`other:holding_red_cube`) nor, for unreadable
    jaws, an object too wide for them (`jaw`)."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.arms[0].jaws = jaws
    out = r.runtime.execute("reset_scene", {})
    _refused(out, error)
    assert r.runtime.envelope.stats()["reset_scene"]["failures"] == {signature: 1}


@pytest.mark.parametrize("label", ["short", "long"])
@pytest.mark.parametrize("case", ["unjudged", "contact"])
def test_a_reset_refused_for_an_unjudged_or_unread_hold_is_filed_by_the_jaw_state(rig, case, label):
    """The jaw reading that could not be judged, and the unfinished contact
    episode whose jaws were never read (#367), file under the jaw state too:
    never an object too wide for the jaws (`jaw`), never the label's first
    words (`other:the_grasp_of`), whatever the label's length."""
    r = rig("rebot_rs")
    if label == "long":
        sentence = _long_label_case(r, case)
    else:
        _stop_in_the_grasp(r, stall=0.5)
        if case == "unjudged":
            _malform(r.runtime, "label_only")
            sentence = UNJUDGED
        else:
            r.runtime._contact_episode = {"unfinished": True}
            sentence = CONTACT
    out = r.runtime.execute("reset_scene", {})
    assert out["ok"] is False and sentence in out["error"] and r.log == [], (out, r.log)
    assert r.runtime.envelope.stats()["reset_scene"]["failures"] == {"state:jaws_unreadable": 1}


def test_open_gripper_after_unreadable_jaws_clears_the_hold_and_the_reset_runs(rig):
    """The refusal's own way out, with the jaws still unreadable: the object
    taken out by hand, open_gripper drops the provisional hold, and the reset
    then runs (no jaw command: unknown jaws stay as they are)."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.arms[0].jaws = "unknown"
    _refused(r.runtime.execute("reset_scene", {}), UNREAD)
    assert r.log == []
    r.runtime.execute("open_gripper", {})
    assert [e[0] for e in r.log] == ["jaw"], r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is None
    r.log.clear()
    out = r.runtime.execute("reset_scene", {})
    _ran_as_for_empty_jaws(r, out, opened=False)


def test_a_long_label_and_a_long_driver_error_keep_the_way_out_inside_240_characters(rig):
    """The hold sentence comes first and the driver's text is shortened: the
    agent's failure line keeps only the first 240 characters of the error."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    label = "the tall blue ceramic coffee mug with the chipped handle, left"   # 60 characters
    r.runtime._held_provisional = (label, *tuple(r.runtime._held_provisional)[1:])
    r.arms[0].jaws = "error"
    r.arms[0].error = "CAN bus: gripper motor 7 did not answer the position request " * 5
    out = r.runtime.execute("reset_scene", {})
    head = (f"the jaw state could not be read: if {label} is in the jaws, take it out by hand, then "
            "open_gripper, then reset_scene. Error: CAN bus: gripper motor 7")
    _refused(out, head)
    assert len(out["error"]) <= 240, (len(out["error"]), out["error"])
    assert r.log == [], r.log


@pytest.mark.parametrize("fault, why", [("unknown", "no valid jaw feedback"),
                                        ("error", "gripper feedback timed out")])
@pytest.mark.parametrize("stall", [0.5, None])
def test_one_failed_jaw_read_then_good_reads_decide_on_the_real_reading(rig, monkeypatch, fault, why, stall):
    """A single failed read refuses (fail closed, 0 commands); the next reset
    reads the jaws again and decides on that reading: stalled short of the
    seat refuses \"holding\", at the seat runs as for empty jaws."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=stall)
    rt = r.runtime
    read = rt._gripper_width_frac
    reads = []

    def flaky():
        reads.append(1)
        if len(reads) > 1:
            return read()
        if fault == "error":
            raise TimeoutError("gripper feedback timed out")
        return None

    monkeypatch.setattr(rt, "_gripper_width_frac", flaky)
    first = rt.skill_reset_scene()
    _refused(first, UNREAD)
    assert first["error"] == UNREAD + why and r.log == [], (first, r.log)
    assert rt._held_provisional is not None
    second = rt.skill_reset_scene()
    if stall is None:
        _ran_as_for_empty_jaws(r, second, opened=True)
    else:
        _refused(second, REFUSAL)
        assert r.log == [] and rt.held_object == "red cube", r.log


def test_an_estop_latched_after_the_stop_still_names_the_hold(rig):
    """Both are true; the reset names the hold (the refusal the agent can act
    on once the e-stop is reset), and nothing moves."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.runtime.arm.harness.estop("operator")
    out = r.runtime.execute("reset_scene", {})
    _refused(out, REFUSAL)
    assert out["error"] == REFUSAL and r.log == [], (out, r.log)


@pytest.mark.parametrize("stall", [0.5, None])
@pytest.mark.parametrize("call", ["execute", "direct"])
def test_a_non_finite_jaw_reading_after_a_stop_is_unreadable_never_empty(rig, call, stall):
    """A reading the driver reports valid that is no number (NaN) is no jaw
    width. It fell through every branch of the reconcile and CLEARED the
    hold, so the reset homed with whatever the jaws held (stalled on the cube
    or at the seat alike). It is unreadable: refused before any command, the
    hold kept, and the next real reading decides."""
    r = rig("rebot_rs")
    assert r.cfg.arm.type == "rebot_rs" and r.runtime._drives_real_gripper()
    _stop_in_the_grasp(r, stall=stall)
    r.arms[0].jaws = "nan"
    state = r.runtime.arm.get_state()
    assert state.gripper_valid is True and math.isnan(state.gripper_pos)
    out = _reset(r, call)
    _refused(out, UNREAD)
    assert out["error"] == UNREAD + "no valid jaw feedback"
    assert r.log == [], r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is not None
    r.arms[0].jaws = "ok"                          # the next reading is a number again
    out = _reset(r, call)
    if stall is None:
        _ran_as_for_empty_jaws(r, out, opened=True)
    else:
        _refused(out, REFUSAL)
        assert r.log == [] and r.runtime.held_object == "red cube", r.log


def test_a_grasp_entry_keeps_the_hold_on_a_non_finite_jaw_reading(rig):
    """The entry's own reconcile (fail closed at the seat) reads NaN as
    unknown too: the hold stays, nothing is promoted or cleared."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.arms[0].jaws = "nan"
    r.runtime._reconcile_held()
    assert r.runtime.held_object is None and r.runtime._held_provisional is not None


#: malformed markers, the jaw reading each is judged against, and why judging fails
MALFORMED = {
    # its close's opening and no expected one, the jaws at that seat: `_air_grasp` raises
    "no_expected_open": (None, "'>=' not supported between instances of 'NoneType' and 'float'"),
    # a bare label, the jaws stalled on the cube: promoting it cannot unpack it
    "label_only": (0.5, "not enough values to unpack (expected 3, got 1)"),
}


def _malform(rt, kind):
    prov = rt._held_provisional
    rt._held_provisional = (ProvisionalHold(tuple(prov), prov.commanded_open) if kind == "no_expected_open"
                            else ProvisionalHold(prov[:1]))


@pytest.mark.parametrize("kind", sorted(MALFORMED))
@pytest.mark.parametrize("call", ["execute", "direct"])
def test_a_marker_that_cannot_be_judged_is_refused_as_such_not_as_unread_jaws(rig, call, kind):
    """The jaws read fine, but the marker cannot be judged against them. No
    jaw read failed, so the refusal says the marker could not be judged;
    nothing moves, and the same way out works."""
    stall, why = MALFORMED[kind]
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=stall)
    _malform(r.runtime, kind)
    out = _reset(r, call)
    _refused(out, UNJUDGED)
    assert out["error"] == UNJUDGED + why, out
    assert r.log == [], r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is not None
    r.runtime.execute("open_gripper", {})
    assert [e[0] for e in r.log] == ["jaw"] and r.runtime._held_provisional is None, r.log
    r.log.clear()
    _ran_as_for_empty_jaws(r, r.runtime.execute("reset_scene", {}), opened=False)


def test_only_the_reset_reconcile_names_a_marker_it_cannot_judge(rig):
    """Every other caller of the reconcile (a grasp's entry, the retry's
    check) gets the original error, as before."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    _malform(r.runtime, "label_only")
    with pytest.raises(ValueError, match="not enough values to unpack"):
        r.runtime._reconcile_held()
    assert r.log == [] and r.runtime._held_provisional is not None


@pytest.mark.parametrize("call", ["execute", "direct"])
def test_an_unfinished_contact_episode_is_refused_as_such_not_as_unread_jaws(rig, monkeypatch, call):
    """The reconcile never reads the jaws during an unfinished contact episode
    (only its explicit recovery may finish that close), so the refusal cannot
    say the jaws were unreadable: it names the episode. Nothing moves."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    rt, reads = r.runtime, []
    read = rt._gripper_width_frac
    monkeypatch.setattr(rt, "_gripper_width_frac", lambda: reads.append(1) or read())
    rt._contact_episode = {"unfinished": True}
    out = _reset(r, call)
    _refused(out, CONTACT)
    assert out["error"] == CONTACT and reads == [], (out, reads)
    assert r.log == [], r.log
    assert rt.held_object is None and rt._held_provisional is not None


#: a label longer than the 240 characters the agent's failure line quotes leave
#: for it (the grasp tool accepts names this long)
LONG_LABEL = "the " + "very " * 40 + "long blue mug"


def _long_label_case(r, case):
    """Put LONG_LABEL in a hold of kind `case`; return the sentence the
    refusal must keep whole after the (shortened) label."""
    rt = r.runtime
    if case == "held":
        rt.arm.get_state()
        rt.held_object, rt._held_det_label = LONG_LABEL, "mug"
        return ": place it first (place_at), then reset"
    _stop_in_the_grasp(r, stall=None if case == "unjudged" else 0.5)
    prov = rt._held_provisional
    rt._held_provisional = ProvisionalHold((LONG_LABEL, *prov[1:]), prov.commanded_open,
                                           None if case == "unjudged" else prov.expected_open)
    if case == "contact":
        rt._contact_episode = {"unfinished": True}
        return (" ended in an unfinished contact episode, so the jaws were not read and it may be in them: "
                "a person must check the gripper.")
    if case in ("unknown", "error"):
        r.arms[0].jaws = case
        r.arms[0].error = "CAN bus: gripper motor 7 did not answer the position request " * 5
    return WAY_OUT


@pytest.mark.parametrize("case", ["held", "unknown", "error", "unjudged", "contact"])
def test_a_long_label_is_shortened_never_the_refusal_sentence(rig, case):
    """The agent's failure line quotes the first 240 characters of the error:
    a label too long for them is cut ("..."), so the refusal's sentence stays
    whole in that line and nothing is marked truncated."""
    r = rig("rebot_rs")
    sentence = _long_label_case(r, case)
    out = r.runtime.execute("reset_scene", {})
    error = out["error"]
    assert out["ok"] is False and r.log == [], (out, r.log)
    assert len(error) <= 240 and "... " not in LONG_LABEL, (len(error), error)
    assert sentence in error and LONG_LABEL not in error, error
    cut = error[:error.index(sentence)]
    assert cut.endswith("...") and LONG_LABEL[:40] in cut, error
    assert sentence in out["next_action"] and "(truncated; see error)" not in out["next_action"], out


def test_an_empty_driver_error_names_its_type_not_a_bare_error_tag(rig):
    """A driver exception without text leaves no dangling \"Error: \"."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.arms[0].jaws, r.arms[0].error = "error", ""
    out = r.runtime.execute("reset_scene", {})
    _refused(out, UNREAD)
    assert out["error"] == UNREAD + "TimeoutError", out
    assert r.log == [], r.log


def test_the_failure_line_keeps_the_cut_marker_of_a_shortened_driver_error(rig):
    """The driver's text is shortened with \"...\": the failure line quotes
    the error with that marker, it does not turn it into a full stop."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.arms[0].jaws = "error"
    r.arms[0].error = "CAN bus: gripper motor 7 did not answer the position request " * 5
    out = r.runtime.execute("reset_scene", {})
    error = out["error"]
    assert error.startswith(UNREAD + "CAN bus") and error.endswith("...") and len(error) <= 240, error
    assert out["next_action"].startswith(f"This attempt FAILED: {error} Tell the user it failed"), out


@pytest.mark.parametrize("call", ["execute", "direct"])
def test_jaws_closed_on_nothing_after_a_stop_in_the_lift_get_their_opening_at_home(rig, call):
    """Jaws that read closed (short of the air tolerance) hold nothing: the
    marker is dropped, the reset runs, and the closed empty jaws get the
    reset's one opening at home."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=None)
    r.arms[0].set_gripper(r.cfg.arm.gripper.closed_pos)        # nothing between the jaws
    r.log.clear()
    out = _reset(r, call)
    _ran_as_for_empty_jaws(r, out, opened=True)


def test_jaws_opened_after_a_stop_in_the_lift_let_the_reset_run(rig):
    """Jaws at the open position hold nothing: the reset runs, no jaw command."""
    r = rig("rebot_rs")
    _stop_in_the_grasp(r, stall=0.5)
    r.arms[0].object_stop_frac = None
    r.arms[0].set_gripper(r.cfg.arm.gripper.open_pos)          # opened by hand, not open_gripper
    r.log.clear()
    out = r.runtime.execute("reset_scene", {})
    _ran_as_for_empty_jaws(r, out, opened=False)
    assert all(e[0] == "arm" for e in r.log), r.log


def test_a_held_flag_refuses_as_it_is_whatever_the_jaws_read(rig):
    """Only a provisional hold is reconciled: a held flag keeps #357's
    refusal even on jaws that read closed (no slip is inferred here; on a
    handless mount every reading is "closed")."""
    r = rig("rebot_rs")
    r.runtime.arm.get_state()
    r.runtime.arm.set_gripper(r.cfg.arm.gripper.closed_pos)
    r.runtime.held_object, r.runtime._held_det_label = "red cube", "cube"
    r.log.clear()
    out = r.runtime.execute("reset_scene", {})
    _refused(out, REFUSAL)
    assert r.log == [] and r.runtime.held_object == "red cube", r.log


@pytest.mark.parametrize("stall", [0.5, None])
def test_the_sim_reset_after_the_same_stop_is_unchanged(rig, stall):
    """A mock arm keeps today's reset: it homes and sends no jaw command
    (the provisional hold is neither promoted nor opened), stalled jaws and
    jaws at the seat alike."""
    r = rig("mock")
    _stop_in_the_grasp(r, stall=stall)
    out = r.runtime.execute("reset_scene", {})
    assert out["ok"] and "jaws_open_commanded" not in out, out
    assert r.log and all(e[0] == "arm" for e in r.log), r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is not None
