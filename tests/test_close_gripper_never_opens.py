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
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import test_grasp_entry_possible_hold as P
import test_rebot_grip_squeeze_cap as S
import test_stop_retrace as T
from cascade.control.mock_arm import MockArm
from cascade.grasping.force import GRIP_PROFILES
from cascade.skills import stop_retrace
from cascade.skills.runtime import SkillRuntime
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
def test_unreferenced_jaws_are_referenced_by_closing_first_as_on_main(clock, make):
    """No reading: the first stage goes to the gripper guard, which closes to
    the stop before anything else (rs_gripper.py), then runs both stages."""
    jaw = S._SimJaw(pos=SEAT, contact=S.CLOSED)
    arm = make(jaw)
    arm._grip_guard.offset = None
    arm._grip_guard._last_target = None
    S._close(arm, RIGID)
    targets = [t for t, _ in jaw.commands()]
    assert targets[0] < SEAT and max(targets) == pytest.approx(SCOUT) and targets[-1] == pytest.approx(SEAT)


def test_a_seat_reading_lets_the_close_skip_only_the_scout_stage():
    from cascade.control.robstride import closing_stages

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
