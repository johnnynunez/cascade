"""close_gripper never opens the jaws (#379 review, pre-existing A).

The rigid two-stage close sends a scout stage (half closed: 3.1 rad on the
reBot) before its seat (85 % closed: 0.93 rad). From open jaws that is a
close. From jaws already past the scout stage -- a second close from the
stage-2 seat, a thin object seated there, jaws stalled on something narrower
than half open -- the scout stage OPENS them first: on main a close from the
seat went 0.93 -> 3.1 -> 0.93 rad, at home (z 0.4377 m) or at the object,
and a thin seated object could drop. The runtime's own stages AND the real
reBot drivers' `close_gripper_two_stage` (fixed and squeeze-capped) did it.

Now a close never commands a stage more open than the jaws read (or than the
stage before it): it is skipped. Jaws that give no reading while an object
may be in them (an interrupted grasp's provisional hold that survives, or a
held object) refuse before any command, in the shared words of #363/#379.
Open jaws, and unread jaws with nothing held, close exactly as on main.

Runtime rows run the REAL rebot_rs type over a MockArm
(`test_stop_retrace.rebot_real`) and every profile over a recording MockArm;
driver rows run both RS transports over the simulated RobStride jaw of
`test_rebot_grip_squeeze_cap`.

One failed jaw read inside the driver's own close (a bus error, or a NaN /
+-inf value the driver refuses) left it no start reading, and with none it
kept every stage: the 3.1 rad scout stage went to the gripper guard, whose
next good read ramped the jaws open on the object (#387 review, Important 1).
An unread start now falls back to the guard's last jaw target, and with none
either (unreferenced jaws) to the most closed stage; the result's
`stages_skipped` is what the driver itself skipped.
"""
from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import pytest

import test_grasp_entry_possible_hold as P
import test_rebot_grip_squeeze_cap as S
import test_stop_retrace as T
from cascade.control import rebot_rs_arm as rs_mod
from cascade.control import rebot_rs_mb_arm as mb_mod
from cascade.control import robstride as rbs
from cascade.control import rs_gripper as rg
from cascade.control.mock_arm import MockArm
from cascade.control.rebot_rs_arm import RebotRSArm
from cascade.grasping.force import GRIP_PROFILES
from cascade.skills import stop_retrace
from cascade.skills.runtime import ProvisionalHold, SkillRuntime
from cascade.types import SkillStuck
from conftest import needs_pin

rebot_real = T.rebot_real   # the REAL rebot_rs type (a real gripper), its backend a MockArm
clock = S.clock             # the drivers' fake clock

RIGID = GRIP_PROFILES["rigid"]
#: the reBot's rigid stages (open 6.2 rad, closed 0.0): scout 3.1, seat 0.93
SCOUT = S._stage_target(RIGID.close_frac_stage1)
SEAT = S._stage_target(RIGID.close_frac_stage2)
#: close_gripper's refusal for unread jaws with an object held
HELD_UNREAD = ("SkillError: the jaw state could not be read while holding 'red cube': a close stage could open "
               "the jaws on it; place it first (place_at)")


def _jaws_at(rt, pos, stop_frac=None):
    """The jaws rest at `pos` (motor units); `stop_frac` = where an object
    between them stops a close (MockArm), None = nothing between them."""
    raw = T.mock_of(rt)
    raw.object_stop_frac = None
    raw._gripper = float(pos)
    raw.object_stop_frac = stop_frac


def _stop_frac_at(pos):
    """The MockArm `object_stop_frac` whose object stops the reBot jaws at `pos`."""
    return (S.OPEN - pos) / (S.OPEN - S.CLOSED)


# ── the runtime's close on the REAL rebot_rs type ──────────────────────────


@needs_pin
@pytest.mark.parametrize("hold", ["nothing", "stop-after-an-air-close", "thin-object-held"])
def test_a_close_from_the_stage2_seat_sends_only_the_seat(rebot_real, monkeypatch, hold):
    """Main reopened the jaws to the 3.1 rad scout stage first: with a STOP
    after an air close that is at the object, and with a thin object seated
    at 0.93 rad it drops (the MockArm object lets the jaws open past it)."""
    rt = rebot_real
    if hold == "stop-after-an-air-close":
        T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
        assert T.mock_of(rt)._gripper == pytest.approx(SEAT) and rt._held_provisional is not None
    elif hold == "thin-object-held":
        _jaws_at(rt, SEAT, _stop_frac_at(SEAT))
        rt.held_object = "red cube"
    else:
        _jaws_at(rt, SEAT)
    prov = getattr(rt, "_held_provisional", None)
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("close_gripper", {})
    assert jaws == [pytest.approx(SEAT)] and T.commands(rt) == n0, jaws          # never 3.1
    assert res["ok"] is True and res["stages_skipped"] == 1 and "note" not in res, res
    assert T.mock_of(rt)._gripper == pytest.approx(SEAT) and res["open_frac"] == pytest.approx(0.15)
    assert getattr(rt, "_held_provisional", None) == prov
    assert rt.held_object == ("red cube" if hold == "thin-object-held" else None)


@needs_pin
@pytest.mark.parametrize("where", ["home", "object"])
@pytest.mark.parametrize("reading", P.READINGS)
@P.STALLS
def test_unread_jaws_with_a_possible_hold_refuse_before_any_command(rebot_real, monkeypatch, stall, reading, where):
    """The review's six rows and their kin: the hold survives (unknown, NaN,
    +-inf readings), at home after the retrace or at the object with no
    column. Main sent 3.1 then 0.93 (`where` home: at z 0.4377 m)."""
    rt = rebot_real
    if where == "home":
        P.holding_unknown_at_home(rt, monkeypatch, stall, reading)
    else:
        T.mock_of(rt).object_stop_frac = stall
        T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
        assert stop_retrace.forget(rt) == 1
        P.unreadable(rt, monkeypatch, reading)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("close_gripper", {})
    P.assert_refused_before_any_command(rt, res, n0, jaws, prov)


@needs_pin
@pytest.mark.parametrize("reading", ["unknown", "ninf"])
def test_a_direct_caller_with_the_column_still_kept_is_sent_to_move_home_first(rebot_real, monkeypatch, reading):
    """execute() forfeits the column before close_gripper; a direct caller
    keeps it and gets the move_home-first form, nothing sent."""
    rt = rebot_real
    T.mock_of(rt).object_stop_frac = 0.5
    T.stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    P.unreadable(rt, monkeypatch, reading)
    prov, jaws, n0 = rt._held_provisional, T.jaw_commands(rt, monkeypatch), T.commands(rt)
    with pytest.raises(SkillStuck) as refused:
        SkillRuntime.skill_close_gripper(rt)
    assert f"SkillStuck: {refused.value}" == P.REFUSAL_KEPT and refused.value.ask == P.LOOP_ASK
    assert T.commands(rt) == n0 and jaws == [] and rt._held_provisional == prov


@needs_pin
@pytest.mark.parametrize("reading", P.READINGS)
def test_unread_jaws_with_an_object_held_refuse_before_any_command(rebot_real, monkeypatch, reading):
    """A held object stalled at 1.86 rad (70 % closed), jaws unreadable: no
    stage can be shown not to open them (main sent 3.1: the object drops)."""
    rt = rebot_real
    _jaws_at(rt, S._stage_target(0.7), 0.7)
    rt.held_object = "red cube"
    P.unreadable(rt, monkeypatch, reading)
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("close_gripper", {})
    assert res["ok"] is False and res["error"] == HELD_UNREAD, res
    assert len(res["error"]) <= 240 and T.commands(rt) == n0 and jaws == [] and rt.held_object == "red cube"


@needs_pin
def test_a_long_held_label_is_cut_never_the_refusal(rebot_real, monkeypatch):
    rt = rebot_real
    _jaws_at(rt, SEAT, _stop_frac_at(SEAT))
    rt.held_object = T.LONG_LABEL
    P.unreadable(rt, monkeypatch, "unknown")
    res = rt.execute("close_gripper", {})
    error = res["error"]
    assert res["ok"] is False and len(error) <= 240 and error in res["next_action"], (len(error), res)
    assert error.startswith("SkillError: the jaw state could not be read while holding 'the very very")
    assert error.endswith("...: a close stage could open the jaws on it; place it first (place_at)"), error


@needs_pin
@pytest.mark.parametrize("jaws_read", ["open", "unread-nothing-held"])
def test_open_jaws_and_unread_jaws_with_nothing_held_close_as_on_main(rebot_real, monkeypatch, jaws_read):
    rt = rebot_real
    _jaws_at(rt, S.OPEN if jaws_read == "open" else SEAT)
    if jaws_read != "open":
        P.unreadable(rt, monkeypatch, "unknown")
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("close_gripper", {})
    assert res["ok"] is True and "stages_skipped" not in res, res
    assert jaws == [pytest.approx(SCOUT), pytest.approx(SEAT)] and T.commands(rt) == n0, jaws


@needs_pin
def test_fully_closed_jaws_get_no_command_and_the_result_says_so(rebot_real, monkeypatch):
    """Both stages are more open than closed jaws: nothing is sent, and the
    verdict credits no close (main opened them to 3.1, then to 0.93)."""
    rt = rebot_real
    _jaws_at(rt, S.CLOSED)
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("close_gripper", {})
    assert jaws == [] and T.commands(rt) == n0 and T.mock_of(rt)._gripper == S.CLOSED, jaws
    assert res["ok"] is True and res["stages_skipped"] == 2 and "none was sent" in res["note"], res
    pc = res["postcondition"]
    assert pc["status"] == "unverified" and pc["evidence"].startswith("no close command completed"), pc


@needs_pin
@pytest.mark.parametrize("stop", ["halt", "estop"])
@pytest.mark.parametrize("start", ["seat", "open"])
def test_a_stop_before_the_close_sends_no_jaw_command(rebot_real, monkeypatch, start, stop):
    """An operator halt or the e-stop once the skill runs, before its first
    stage: no jaw command on either start."""
    rt = rebot_real
    _jaws_at(rt, SEAT if start == "seat" else S.OPEN)
    real = SkillRuntime._close_two_stage

    def close(self, *a, **kw):
        self.arm.harness.halt("operator") if stop == "halt" else self.arm.harness.estop("operator")
        return real(self, *a, **kw)

    monkeypatch.setattr(SkillRuntime, "_close_two_stage", close)
    jaws, n0 = T.jaw_commands(rt, monkeypatch), T.commands(rt)
    res = rt.execute("close_gripper", {})
    assert res["ok"] is False and jaws == [] and T.commands(rt) == n0, res
    assert ("e-stop latched" in res["error"]) is (stop == "estop"), res


# ── the runtime over a driver with its own two-stage close (real reBot) ────


def _driver_close(rt, monkeypatch):
    calls = []
    monkeypatch.setattr(T.mock_of(rt), "close_gripper_two_stage", lambda **kw: calls.append(kw), raising=False)
    return calls


@needs_pin
def test_the_driver_close_is_not_called_when_every_stage_would_open_the_jaws(rebot_real, monkeypatch):
    rt = rebot_real
    _jaws_at(rt, S.CLOSED)
    calls, closes = _driver_close(rt, monkeypatch), rt._gripper_closes_commanded
    res = rt.execute("close_gripper", {})
    assert calls == [] and rt._gripper_closes_commanded == closes, calls
    assert res["ok"] is True and res["stages_skipped"] == 2, res


@needs_pin
@pytest.mark.parametrize("start, skipped", [("seat", 1), ("open", None)])
def test_the_driver_close_runs_when_a_stage_closes_and_skips_by_its_own_reading(
        rebot_real, monkeypatch, start, skipped):
    rt = rebot_real
    _jaws_at(rt, SEAT if start == "seat" else S.OPEN)
    calls = _driver_close(rt, monkeypatch)
    res = rt.execute("close_gripper", {})
    assert res["ok"] is True and res.get("stages_skipped") == skipped, res
    assert calls == [{"width_frac_stage1": RIGID.close_frac_stage1, "width_frac_stage2": RIGID.close_frac_stage2,
                      "effort": RIGID.effort}]


@needs_pin
def test_a_possible_hold_with_unread_jaws_never_reaches_the_driver_close(rebot_real, monkeypatch):
    rt = rebot_real
    P.holding_unknown_at_home(rt, monkeypatch, None)
    calls, n0 = _driver_close(rt, monkeypatch), T.commands(rt)
    res = rt.execute("close_gripper", {})
    assert res["ok"] is False and res["error"] == P.REFUSAL and calls == [] and T.commands(rt) == n0, res


# ── every profile: its own stages, from its own seat ────────────────────────


class _JawArm(MockArm):
    """A MockArm that records every jaw command."""

    def __init__(self, cfg, kinematics, log):
        super().__init__(cfg, kinematics)
        self.log = log

    def set_gripper(self, pos, effort=1.0):
        self.log.append(float(pos))
        super().set_gripper(pos, effort)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    def build(profile):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False      # the jaw command is outside the map
        r = SimpleNamespace(cfg=cfg, jaws=[], arms=[])

        def make_arm(acfg, kinematics=None):
            arm = _JawArm(acfg, kinematics, r.jaws)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        r.runtime.arm.get_state()        # materialize the LazyArm: the jaws live on the MockArm
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


#: profile -> its close's jaw targets from open jaws (main's), as literals
#: of each YAML's open_pos .. closed_pos: rigid scout 50 %, seat 85 %
PROFILES = {
    "rebot_rs": (6.2, 0.0), "rebot_rs_mb": (6.2, 0.0), "so101": (0.60, -0.15), "so101_mock": (0.60, -0.15),
    "isaac_kitchen": (1.0, 0.0), "mock": (0.0, 1.0), "fr3_mock": (0.04, 0.0),
}


@pytest.mark.parametrize("start", ["open", "seat", "closed"])
@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_closes_from_open_as_on_main_and_never_reopens_a_closer_jaw(rig, profile, start):
    open_pos, closed_pos = PROFILES[profile]
    r = rig(profile)
    g = r.cfg.arm.gripper
    assert (g.open_pos, g.closed_pos) == pytest.approx((open_pos, closed_pos))
    scout, seat = (open_pos + (closed_pos - open_pos) * f for f in (0.5, 0.85))
    arm = r.arms[0]
    arm._gripper = {"open": open_pos, "seat": seat, "closed": closed_pos}[start]
    r.jaws.clear()
    res = r.runtime.execute("close_gripper", {})
    assert res["ok"] is True, res
    expected = {"open": [scout, seat], "seat": [seat], "closed": []}[start]
    assert r.jaws == [pytest.approx(v) for v in expected], (profile, start, r.jaws)


def test_fr3s_force_contact_close_and_a_handless_profile_close_as_on_main(rig):
    """fr3 (ros2 force-contact close) sends one fully closed command, never
    more open than any reading in its travel; g1's hand has no travel."""
    r = rig("fr3")
    r.arms[0]._gripper = 0.04 * 0.15
    r.jaws.clear()
    assert r.runtime.execute("close_gripper", {})["ok"] is True and r.jaws == [pytest.approx(0.0)]
    r.arms[0]._gripper = -0.001            # a reading past the closed end: closed is more open
    r.jaws.clear()
    res = r.runtime.execute("close_gripper", {})
    assert res["ok"] is True and res["stages_skipped"] == 1 and r.jaws == [], (res, r.jaws)
    g1 = rig("g1")
    g1.jaws.clear()
    assert g1.runtime.execute("close_gripper", {})["ok"] is True and g1.jaws == [0.0, 0.0]


# ── the real reBot drivers' own two-stage close ─────────────────────────────


TRANSPORTS = pytest.mark.parametrize("make", [S._rs, S._mb], ids=["rebot_rs", "rebot_rs_mb"])
CAPS = pytest.mark.parametrize("cap", [S._UNSET, S.CAP], ids=["fixed", "squeeze-capped"])


@TRANSPORTS
@CAPS
@pytest.mark.parametrize("start", [SEAT, S._contact(0.03), S.CLOSED], ids=["seat", "stalled-on-3cm", "closed"])
def test_the_driver_close_never_commands_the_jaws_past_where_they_are(clock, make, cap, start):
    """An object stops the jaws where they rest (a thin one seated at 0.93
    rad, a 3 cm one, or nothing at the closed stop): main's scout stage
    ramped them open to 3.1 rad first, off the object."""
    jaw = S._SimJaw(pos=start, contact=start)
    S._close(make(jaw, cap), RIGID)
    expected = [] if start == S.CLOSED else [SEAT]
    assert [t for t, _ in jaw.commands()] == [pytest.approx(v) for v in expected], jaw.commands()
    assert jaw.pos == pytest.approx(start)                  # never left the object


@TRANSPORTS
@CAPS
def test_the_driver_close_from_open_jaws_runs_both_stages_as_on_main(clock, make, cap):
    jaw = S._SimJaw(contact=S._contact(0.05))
    S._close(make(jaw, cap), RIGID)
    assert [t for t, _ in jaw.commands()] == [pytest.approx(SCOUT), pytest.approx(SEAT)], jaw.commands()


@TRANSPORTS
def test_unreferenced_jaws_are_referenced_by_closing_first_then_get_only_the_seat(clock, make):
    """No reading and no last jaw target (the guard lost its turn reference):
    only the most closed stage is sent. The gripper guard references the
    jaws by closing to the stop first (rs_gripper.py; an object between them
    faults there), and the scout stage never goes: main and #387 sent 3.1 rad
    after the reference."""
    jaw = S._SimJaw(pos=SEAT, contact=S.CLOSED)
    arm = make(jaw)
    arm._grip_guard.offset = None
    arm._grip_guard._last_target = None
    S._close(arm, RIGID)
    targets = [t for t, _ in jaw.commands()]
    assert targets[0] < SEAT and targets[-1] == pytest.approx(SEAT), targets
    assert max(targets) <= SEAT + 1e-6 and arm.close_stages_skipped == 1, targets


def test_a_seat_reading_lets_the_close_skip_only_the_scout_stage():
    from cascade.control.jaw_stages import closing_stages

    lines = []
    stages = ((SCOUT, 0.49), (SEAT, 0.7))
    assert closing_stages(stages, S.OPEN, S.OPEN, S.CLOSED, log=lines.append) == stages and lines == []
    assert closing_stages(stages, SEAT, S.OPEN, S.CLOSED, log=lines.append) == ((SEAT, 0.7),)
    assert lines == ["close stage to 3.100 skipped: the jaws are already at 0.930, and a close never opens them"]
    assert closing_stages(stages, None, S.OPEN, S.CLOSED) == stages
    assert closing_stages(((SEAT, 1), (SCOUT, 1)), None, S.OPEN, S.CLOSED) == ((SEAT, 1),)
    # the mock's polarity (open 0.0, closed 1.0), and a hand with no travel
    assert closing_stages(((0.5, 1), (0.85, 1)), 0.85, 0.0, 1.0) == ((0.85, 1),)
    assert closing_stages(((0.0, 1), (0.0, 1)), 0.0, 0.0, 0.0) == ((0.0, 1), (0.0, 1))


# ── one failed jaw read inside the driver's own close (#387 review I1) ──────


#: a mechPos read the driver cannot use: a bus error, or a value it refuses
#: as implausible (`robstride.plausible_position`)
BAD_READS = {"err": None, "nan": float("nan"), "pinf": float("inf"), "ninf": float("-inf")}


class _OneBadRead(S._Motor):
    """The jaw's RobStride motor: its `bad_at`-th mechPos read (1-based, None
    = none) is bad; `reads` counts them."""

    def __init__(self, jaw, bad_at=None, how="err"):
        super().__init__(jaw)
        self.bad_at, self.how, self.reads = bad_at, how, 0

    def robstride_get_param_f32(self, param, timeout_ms=None):
        if param == S.MECH_POS:
            self.reads += 1
            if self.reads == self.bad_at:
                if self.how == "err":
                    raise RuntimeError("CAN read timeout (test)")
                return BAD_READS[self.how]
        return self.jaw.read(param)


def _jaw_driver(make, cap, start, contact, bad_at=None, how="err"):
    """A driver over the simulated jaw resting at `start` (an object stops it
    at `contact`, None = nothing between the jaws), its `bad_at`-th mechPos
    read bad. `sent` = every jaw command with where the jaw physically was."""
    jaw = S._SimJaw(pos=start, contact=contact)
    sent, real = [], jaw.command

    def command(pos, kp, kd):
        jaw._advance()
        sent.append((float(pos), jaw.pos))
        real(pos, kp, kd)

    jaw.command = command
    drv = make(jaw, cap)
    motor = _OneBadRead(jaw, bad_at, how)
    if isinstance(drv, RebotRSArm):
        drv._arm.gripper._mm["gripper"] = motor
    else:
        drv._motors[7] = motor
    return SimpleNamespace(jaw=jaw, drv=drv, motor=motor, sent=sent)


def _openings(sent):
    """Commands more open than the jaw was when they went out (RS: open > closed)."""
    return [(round(t, 3), round(p, 3)) for t, p in sent if t > p + 0.02]


#: jaws at rest on an object, closer than the scout stage
ON_AN_OBJECT = pytest.mark.parametrize("start", [SEAT, S._contact(0.03), 2.5],
                                       ids=["seated", "stalled-on-3cm", "half-closed"])


@TRANSPORTS
@CAPS
@ON_AN_OBJECT
@pytest.mark.parametrize("how", sorted(BAD_READS))
def test_one_failed_jaw_read_anywhere_in_the_driver_close_never_opens_the_jaws(clock, make, cap, start, how):
    """#387 sent the scout stage (3.1 rad) when the driver's start read failed
    and the guard's next good read ramped the jaws open off the object. One
    bad read at every read index: never a command more open than the jaw,
    the jaw stays on the object, the scout is the one stage skipped."""
    clean = _jaw_driver(make, cap, start, start)
    S._close(clean.drv, RIGID)
    assert _openings(clean.sent) == []
    for bad_at in range(1, clean.motor.reads + 1):
        row = _jaw_driver(make, cap, start, start, bad_at, how)
        S._close(row.drv, RIGID)
        assert row.motor.reads >= bad_at, (bad_at, row.motor.reads)
        assert _openings(row.sent) == [], (bad_at, _openings(row.sent))
        assert all(t <= SEAT + 1e-6 for t, _ in row.sent), (bad_at, row.sent)
        assert row.jaw.pos == pytest.approx(start) and row.drv.close_stages_skipped == 1, bad_at


@TRANSPORTS
@CAPS
@pytest.mark.parametrize("how", sorted(BAD_READS))
def test_one_failed_jaw_read_from_open_jaws_still_sends_both_stages(clock, make, cap, how):
    """The unread start falls back to the guard's last jaw target: the open
    jaws' own, so from open jaws both stages go, as on main."""
    contact = S._contact(0.05)
    clean = _jaw_driver(make, cap, S.OPEN, contact)
    S._close(clean.drv, RIGID)
    for bad_at in range(1, clean.motor.reads + 1):
        row = _jaw_driver(make, cap, S.OPEN, contact, bad_at, how)
        S._close(row.drv, RIGID)
        assert [t for t, _ in row.jaw.commands()] == [pytest.approx(SCOUT), pytest.approx(SEAT)], bad_at
        assert _openings(row.sent) == [] and row.drv.close_stages_skipped == 0, bad_at


@TRANSPORTS
@CAPS
def test_an_unread_start_is_judged_by_the_last_jaw_target_then_by_the_most_closed_stage(clock, make, cap):
    """The fallback's two steps, each deciding alone: a last target between
    the stages (2.0 rad, jaws stalled there) skips only the scout; none at all
    (and the start unread) sends only the seat."""
    for last, kept in ((2.0, [SEAT]), (S.OPEN, [SCOUT, SEAT]), (None, [SEAT])):
        row = _jaw_driver(make, cap, 2.0, 2.0)
        row.drv._grip_guard._last_target = last
        stages = ((SCOUT, 0.49), (SEAT, 0.7))
        sent = rbs.driver_close_stages(row.drv, stages, None)
        assert [t for t, _ in sent] == [pytest.approx(v) for v in kept], (last, sent)
        assert row.drv.close_stages_skipped == 2 - len(kept)
    # the most closed stage in either polarity (the mock's: open 0, closed 1) and order
    drv = SimpleNamespace(_grip_open=0.0, _grip_closed=1.0, _grip_guard=SimpleNamespace(_last_target=None))
    assert rbs.driver_close_stages(drv, ((0.5, 1), (0.85, 1)), None) == ((0.85, 1),)
    drv = SimpleNamespace(_grip_open=S.OPEN, _grip_closed=S.CLOSED, _grip_guard=SimpleNamespace(_last_target=None))
    assert rbs.driver_close_stages(drv, ((SEAT, 1), (SCOUT, 1)), None) == ((SEAT, 1),)
    assert drv.close_stages_skipped == 1


# ── the runtime over the real driver close: one failed read anywhere ────────


def _driver_jaws(rt, monkeypatch, make, cap, start, bad_at=None, how="err"):
    """The rebot_rs runtime's jaws served by the REAL reBot driver (its RS
    gripper guard and its own close_gripper_two_stage) over the simulated jaw
    resting on an object at `start`; the drivers run on a fake clock, the
    runtime on real time. Only main-thread reads reach the bus (a watcher
    thread gets the last value), so the read indices are deterministic."""
    clock = S._Clock()
    clock.now = 1000.0
    fake = SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep, time=clock.monotonic)
    for mod in (rg, rbs, rs_mod, mb_mod, S):
        monkeypatch.setattr(mod, "time", fake)
    j = _jaw_driver(make, cap, start, start, bad_at, how)
    raw, main, last = T.mock_of(rt), threading.main_thread(), {"pos": start}
    real_state = type(raw).get_state.__get__(raw)

    def get_state():
        if threading.current_thread() is main:
            last["pos"] = j.drv._gripper_pos()
        pos = last["pos"]
        return dataclasses.replace(real_state(), gripper_pos=0.0 if pos is None else float(pos),
                                   gripper_valid=pos is not None)

    monkeypatch.setattr(raw, "get_state", get_state)
    monkeypatch.setattr(raw, "set_gripper", lambda pos, effort=1.0: j.drv.set_gripper(pos, effort=effort))
    monkeypatch.setattr(raw, "close_gripper_two_stage", j.drv.close_gripper_two_stage, raising=False)
    # the MockArm stands in for the driver: its skip count is the driver's
    monkeypatch.setattr(type(raw), "close_stages_skipped", property(lambda _: j.drv.close_stages_skipped),
                        raising=False)
    return j


def _holding(rt, hold):
    rt.held_object = rt._held_det_label = rt._held_color = None
    rt._held_provisional = None
    if hold == "provisional":
        rt._held_provisional = ProvisionalHold(("red cube", "cube", "red"), 0.15, 0.5)
    elif hold == "held":
        rt.held_object, rt._held_det_label, rt._held_color = "red cube", "cube", "red"


@needs_pin
@TRANSPORTS
@CAPS
@pytest.mark.parametrize("start", [SEAT, S._contact(0.03)], ids=["seated", "stalled-on-3cm"])
@pytest.mark.parametrize("hold", ["nothing", "held", "provisional"])
def test_close_gripper_over_the_real_driver_never_opens_on_one_failed_read(rebot_real, monkeypatch, make, cap,
                                                                          start, hold):
    """The review's runtime rows: #387 opened the jaws on the object in 72 of
    them (the driver's own start read failed after the runtime's succeeded)
    and reported `stages_skipped: 1` for a stage it sent. One bad read at
    every read index: no jaw command more open than the jaw, no arm command,
    the jaw stays on the object, and an ok close reports the stage the driver
    skipped (also when it was the runtime's read that failed). Only a failed
    runtime read with an object possibly held refuses, before any command."""
    rt = rebot_real
    with monkeypatch.context() as m:
        _holding(rt, hold)
        clean = _driver_jaws(rt, m, make, cap, start)
        assert rt.execute("close_gripper", {}).get("stages_skipped") == 1
    refused = []
    for how in ("err", "nan"):
        for bad_at in range(1, clean.motor.reads + 1):
            with monkeypatch.context() as m:
                _holding(rt, hold)
                row, n0 = _driver_jaws(rt, m, make, cap, start, bad_at, how), T.commands(rt)
                res = rt.execute("close_gripper", {})
            where = (how, bad_at, res)
            assert _openings(row.sent) == [] and T.commands(rt) == n0, where
            assert row.jaw.pos == pytest.approx(start), where
            if res["ok"]:
                assert res.get("stages_skipped") == 1 and row.drv.close_stages_skipped == 1, where
                assert [t for t, _ in row.jaw.commands()][:1] == [pytest.approx(SEAT)], where
            else:
                refused.append(bad_at)
                assert res["error"] in (P.REFUSAL, HELD_UNREAD) and row.sent == [], where
    assert bool(refused) is (hold != "nothing"), refused


@needs_pin
@TRANSPORTS
@CAPS
def test_stages_skipped_is_what_the_driver_skipped_not_the_runtimes_count(rebot_real, monkeypatch, make, cap):
    """The runtime reads the jaws at the seat (it would skip one stage), the
    driver's own start read finds them shut (it skips both): the result says
    2, credits no close, and notes that none was sent (#387 said 1 and
    counted a close the driver never sent)."""
    rt = rebot_real
    j = _driver_jaws(rt, monkeypatch, make, cap, SEAT)
    real_close = j.drv.close_gripper_two_stage

    def close(**kw):
        j.jaw.pos = j.jaw.target = j.jaw.contact = S.CLOSED     # shut between the two reads
        return real_close(**kw)

    monkeypatch.setattr(T.mock_of(rt), "close_gripper_two_stage", close, raising=False)
    closes = rt._gripper_closes_commanded
    res = rt.execute("close_gripper", {})
    assert res["ok"] is True and res.get("stages_skipped") == 2 and "none was sent" in res.get("note", ""), res
    assert j.sent == [] and rt._gripper_closes_commanded == closes
    assert res["postcondition"]["status"] == "unverified", res["postcondition"]
