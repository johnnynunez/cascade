"""After an air grasp the arm withdraws up through its approach column (spec §3.2 rules 1-2).

An air grasp (jaws verified empty) ends with the arm at the pregrasp hover,
above the target. The target's top in the map can hold that hover inside the
3 cm margin, where every later move except a strict retreat is refused. Before
returning, the runtime re-enters the SAME target column the approach entered
(`_column_exemption`, None on the observed-finger path) and lifts the TCP
+3 cm, else +6 cm, through a fully vetted route. Every other gate keeps its
strength, and the column is released on every path.

The exit's column has a ceiling (A3 review I2): 5 mm above the hover TCP.
Arm points and voxels above it are checked at full strength; the approach
keeps today's unbounded column.

Nothing moves with a possible payload (a provisional hold, an active carry
attachment: "do not invent an empty-tool recovery"), with a latched e-stop or
a halt received since the grasp started (a halt during the close included).
A contact, a halt or an e-stop during the exit ends it at once and propagates
as the structured exception, after the column is released; any other refusal
stays in the outcome string (unless a halt landed in the same move). Only a
preflight refusal (SkillError) or an unreachable lift moves on to the next
height.

An operator stop during the close ends the grasp skill (A3 review round 2;
pre-existing on main): the grasp's own lift carries the grasp-start halt
generation, and the persistence loops around grasp_object never re-home,
re-scan or try again after a halt or an e-stop; a halt reaches execute().

An operator halt at ANY point of a motion skill ends it (A3 review round 3;
pre-existing on main): execute() takes the halt generation when the skill
starts and every motion of the skill carries it, so a halt while the grasp
plans, homes, approaches or re-homes refuses the next motion before its first
command, stays a halt (no e-stop latch) and reaches execute().

Every test runs the real runtime with the mock arm and analytic (OBB) grasps
(no hardware, no GPU, never the host's GraspGen-X server).
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.grasping import evidence as grasp_evidence
from cascade.safety import trajectory
from cascade.safety.contact import ContactMonitorConfig
from cascade.safety.harness import SafetyHarness
from cascade.skills import carry_attachment
from cascade.skills.runtime import SkillRuntime
from cascade.types import ContactDetected, MotionHalted, SafetyViolation, SkillError
from conftest import needs_pin
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map
from owned_server import held_dead_port
from test_b36_pick_recovery import _fail_first_lift

pytestmark = needs_pin

AIR = "air grasp: gripper closed fully, object not held"
#: the exit column's ceiling above the hover TCP (A3 review I2)
CEILING_M = 0.005


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    # Analytic grasps only: never the host's GraspGen-X server (a held dead port).
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
        runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
        arm.object_stop_frac = None  # nothing stops the jaws: every close is an air grasp
        try:
            yield runtime
        finally:
            shutdown_runtime(runtime, arm)


def tcp(rt, q=None):
    return rt.kin.fk(rt.arm.get_state().q if q is None else q)[:3, 3]


def commands(rt):
    return len(rt.arm.raw.commands)


def planned(rt, calls, first=None):
    """The exit's `move` on the mapping-only path (a fully vetted
    `_move_planned`). Records every requested lift in `calls`; `first`, when
    given, handles the first call instead (it may raise)."""

    def move(q):
        calls.append(np.asarray(q, dtype=float).copy())
        if first is not None and len(calls) == 1:
            return first(q)
        return rt._move_planned(q, duration_s=2.0)

    return move


def column_at_tcp(rt):
    return (tcp(rt)[:2].copy(), 0.05, 0.0)


def refuse(q):
    raise SkillError("no safe joint route found; direct route: test refusal")


def air_grasp_without_exit(rt, monkeypatch):
    """An air grasp whose exit only records its arguments: the arm stays at
    the pregrasp hover. Returns the (column, move, keywords) the grasp
    handed over."""
    seen = {}

    def capture(column, move, **kw):
        seen.update(column=column, move=move, kw=kw)
        return "skipped: captured by the test"

    monkeypatch.setattr(rt, "_air_grasp_exit", capture)
    res = rt.skill_grasp_object("red cube")
    assert res["exit_withdrawal"] == "skipped: captured by the test"
    return seen["column"], seen["move"], seen["kw"]


def map_of(*boxes):
    occ = make_map(GridBridge([TABLE, *boxes], voxel=.01), required=False,
                   unobserved="free", max_age_s=600.0)
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.last_error is None and occ.usable()
    return occ


# ── the exit after a real air grasp ─────────────────────────────────────────


def test_air_grasp_withdraws_from_the_pregrasp(rt, monkeypatch):
    starts = []
    real = rt._air_grasp_exit

    def spy(column, move, **kw):
        starts.append(rt.arm.get_state().q.copy())
        return real(column, move, **kw)

    monkeypatch.setattr(rt, "_air_grasp_exit", spy)
    res = rt.skill_grasp_object("red cube")
    assert res["ok"] is False and res["error"] == AIR      # a completed exit adds nothing
    assert res["exit_withdrawal"] == "completed (+3 cm)"
    (q_pre,) = starts
    pre, now = tcp(rt, q_pre), tcp(rt)
    assert now[2] >= pre[2] + .03 - 2e-3
    assert np.linalg.norm(now[:2] - pre[:2]) < 2e-3        # straight up


def test_exit_runs_under_the_approach_column_only(rt, monkeypatch):
    rt.cfg.grasp._data["approach_target_exemption"] = True
    harness = rt.arm.harness
    entered, in_exit, during_move = [], [False], []
    real_column = harness.target_column_exemption

    def column_spy(cylinder, **kw):
        entered.append((in_exit[0], cylinder, kw))
        return real_column(cylinder, **kw)

    monkeypatch.setattr(harness, "target_column_exemption", column_spy)
    real_exit = rt._air_grasp_exit
    hover = []

    def exit_spy(column, move, **kw):
        def watched(q):
            during_move.append(harness._grasp_exempt)
            return move(q)

        hover.append(rt.arm.get_state().q.copy())
        in_exit[0] = True
        try:
            return real_exit(column, watched, **kw)
        finally:
            in_exit[0] = False

    monkeypatch.setattr(rt, "_air_grasp_exit", exit_spy)
    res = rt.skill_grasp_object("red cube")
    assert res["exit_withdrawal"] == "completed (+3 cm)"
    approach = [(c, kw) for inside, c, kw in entered if not inside]
    in_the_exit = [(c, kw) for inside, c, kw in entered if inside]
    assert approach, "premise: the approach entered its target column"
    # the approach: today's call, no ceiling
    assert all(kw == {} for _, kw in approach)
    # entered once, with the very column of this grasp's approach, capped
    # 5 mm above the hover TCP
    ((column, kw),) = in_the_exit
    assert column is approach[-1][0]
    z_max = tcp(rt, hover[0])[2] + CEILING_M
    assert kw == {"z_max": pytest.approx(z_max, abs=1e-6)}
    (held,) = during_move
    center, radius, z_min = column
    assert held is not None and np.array_equal(held[0], center)
    assert held[1:] == (radius, z_min, pytest.approx(z_max, abs=1e-6))
    assert harness._grasp_exempt is None                     # released afterwards


def test_exit_refused_by_a_neighbour_above_keeps_the_arm_put(rt, monkeypatch):
    from cascade.safety.trajectory import lift_ik

    rt.cfg.grasp._data["approach_target_exemption"] = True
    column, move, kw = air_grasp_without_exit(rt, monkeypatch)
    assert column is not None
    # The mock cube's column (~5 cm) holds every arm point a lift raises. Use
    # the narrowest column `_column_geometry` builds (a small target, 2 cm),
    # so that the wrist links rise outside it.
    column = (column[0], .02, column[2])
    harness = rt.arm.harness
    q0 = rt.arm.get_state().q.copy()
    q3 = lift_ik(rt.kin, q0, .03)
    # the rising arm point farthest outside the column
    before, after = rt.kin.link_positions(q0), rt.kin.link_positions(q3)
    rising = [i for i, (a, b) in enumerate(zip(before, after)) if b[2] - a[2] > .025]
    i = max(rising, key=lambda i: np.linalg.norm(after[i][:2] - column[0]))
    p = after[i]
    assert np.linalg.norm(p[:2] - column[0]) > column[1] + .02     # the box below stays outside
    # a neighbour 2 cm above that point at the +3 cm lift, outside the column
    neighbour = centred_box([p[0], p[1], p[2] + .03], [.02, .02, .02])
    harness.occupancy = map_of(neighbour)
    with harness.target_column_exemption(column):
        assert harness.route_end_violation(q0) is None                # premise: the hover is clear
        assert "occupancy map" in harness.route_end_violation(q3)     # +3 cm would end inside the margin

    n0 = commands(rt)
    out = SkillRuntime._air_grasp_exit(rt, column, move, **kw)
    assert out.startswith("refused: +3 cm: ") and "; +6 cm: " in out and "occupancy map" in out
    assert commands(rt) == n0
    np.testing.assert_array_equal(rt.arm.get_state().q, q0)
    assert harness._grasp_exempt is None


# ── the exit's column ends 5 mm above the hover (A3 review I2) ──────────────


def test_a_box_above_the_hover_inside_the_real_column_refuses_the_exit(rt, monkeypatch):
    """Review probe: the REAL approach column (r ~5 cm, z_min 3 cm) holds arm
    point 3, ~37 cm up. A 2 cm box 4 cm above that point, inside the column,
    was exempt (no ceiling) and the +3 cm exit drove the point to 1 cm of it.
    Above the hover TCP + 5 mm the column exempts nothing: refused before any
    command."""
    from cascade.safety.trajectory import lift_ik

    rt.cfg.grasp._data["approach_target_exemption"] = True
    column, move, kw = air_grasp_without_exit(rt, monkeypatch)
    harness = rt.arm.harness
    center, radius, z_min = column
    assert radius > .045                          # premise: the real column, not a shrunk one
    q0 = rt.arm.get_state().q.copy()
    q3 = lift_ik(harness.kin, q0, .03)
    points = harness._clearance_points(q0)
    inside = [i for i, p in enumerate(points)
              if np.linalg.norm(p[:2] - center) <= radius and p[2] >= z_min]
    p = points[max(inside, key=lambda i: points[i][2])]     # the highest arm point in the column
    assert p[2] > tcp(rt)[2] + .2
    box = centred_box([p[0], p[1], p[2] + .05], [.02, .02, .02])   # bottom face 4 cm above it
    assert np.linalg.norm(p[:2] - center) + .01 * np.sqrt(2) < radius   # the box is in the column
    harness.occupancy = map_of(box)
    with harness.target_column_exemption(column):
        # premise: the unbounded column hides the box from the +3 cm end
        assert harness.route_end_violation(q3) is None

    n0 = commands(rt)
    out = SkillRuntime._air_grasp_exit(rt, column, move, **kw)
    assert out.startswith("refused: +3 cm: ") and "; +6 cm: " in out and "occupancy map" in out, out
    assert commands(rt) == n0
    np.testing.assert_array_equal(rt.arm.get_state().q, q0)
    assert harness._grasp_exempt is None


def test_a_hover_inside_the_margin_of_the_target_top_still_withdraws(rt, monkeypatch):
    """The case the exit exists for: the target's top in the map holds the
    hover TCP 1.5 cm away (stuck: every route ending there is refused) and
    nothing is above. The capped column still lets the arm lift +3 cm."""
    rt.cfg.grasp._data["approach_target_exemption"] = True
    column, move, kw = air_grasp_without_exit(rt, monkeypatch)
    harness = rt.arm.harness
    q0, z0 = rt.arm.get_state().q.copy(), tcp(rt)[2]
    top = z0 - .015
    harness.occupancy = map_of(centred_box([column[0][0], column[0][1], top / 2], [.04, .04, top]))
    assert "occupancy map" in harness.route_end_violation(q0)     # premise: stuck at the hover

    out = SkillRuntime._air_grasp_exit(rt, column, move, **kw)
    assert out == "completed (+3 cm)"
    assert tcp(rt)[2] >= z0 + .03 - 2e-3
    assert harness._grasp_exempt is None


def test_without_z_max_the_column_is_unbounded_as_before(rt):
    """z_max None (the approach) changes nothing: the exemption is today's
    (center, radius, z_min), and the point test and the voxel filter exempt
    everything above z_min, however high."""
    harness = rt.arm.harness
    center = np.array([.30, .0])
    column = (center, .05, .03)
    for kw in ({}, {"z_max": None}):
        with harness.target_column_exemption(column, **kw) as active:
            assert active and len(harness._grasp_exempt) == 3
            held = harness._grasp_exempt
            assert np.array_equal(held[0], center) and held[1:] == (.05, .03)
    with harness.target_column_exemption(column, z_max=.10):
        held = harness._grasp_exempt
        assert np.array_equal(held[0], center) and held[1:] == (.05, .03, .10)
    assert harness._grasp_exempt is None

    rng = np.random.default_rng(3)
    pts = np.c_[rng.uniform(.2, .4, 400), rng.uniform(-.1, .1, 400), rng.uniform(-.1, 5.0, 400)]
    bounded = (center, .05, .03, np.inf)
    assert ([SafetyHarness._in_cylinder(q, column) for q in pts]
            == [SafetyHarness._in_cylinder(q, bounded) for q in pts])
    assert SafetyHarness._in_cylinder(np.array([.30, .0, 5.0]), column)
    assert not SafetyHarness._in_cylinder(np.array([.30, .0, .11]), (center, .05, .03, .10))
    assert SafetyHarness._in_cylinder(np.array([.30, .0, .10]), (center, .05, .03, .10))

    above = centred_box([.30, .0, .40], [.02, .02, .02])          # in the column, high above
    occ = map_of(above)
    near = np.array([[.30, .0, .375], [.30, .0, .42], [.325, .0, .40]])
    unbounded = occ.clearance_excluding(near, below_z=.02, cylinder=column, within=.03)
    np.testing.assert_array_equal(
        unbounded, occ.clearance_excluding(near, below_z=.02, cylinder=bounded, within=.03))
    assert np.all(unbounded >= .03)                                # today: the box is exempt
    capped = occ.clearance_excluding(near, below_z=.02, cylinder=(center, .05, .03, .10), within=.03)
    assert np.all(capped < .03)                                    # above z_max: an obstacle


@pytest.fixture(scope="module")
def rebot():
    from test_harness_tool_volume import side_grasp_geometry

    return side_grasp_geometry()


def test_the_tool_volume_sees_a_voxel_above_the_ceiling(rebot):
    """The wrist camera rides ~8 cm above the TCP: a voxel 1.5 cm above it,
    inside the column, is the target's own only below the ceiling."""
    from test_harness_tool_volume import harness as tool_harness

    T = rebot.kin.fk(rebot.q_pre)
    vol = rebot.volume
    cam = vol.points(T)[vol.part == vol.names.index("wrist camera")]
    top = cam[np.argmax(cam[:, 2])]
    h = tool_harness(rebot, [centred_box([top[0], top[1], top[2] + .025], [.02, .02, .02])])
    column = (top[:2].copy(), .07, T[2, 3] - .05)
    z_max = T[2, 3] + CEILING_M
    assert top[2] + .015 > z_max + .01                       # premise: the box is above the ceiling
    with h.target_column_exemption(column):
        assert h._tool_volume_violation(T, h._grasp_exempt) is None       # unbounded: exempt
    with h.target_column_exemption(column, z_max=z_max):
        assert "wrist camera" in (h._tool_volume_violation(T, h._grasp_exempt) or "")


@pytest.mark.parametrize("case", ["refused", "did not settle", "skipped"])
def test_the_air_grasp_error_names_a_failed_exit_only(rt, monkeypatch, case):
    real = rt._air_grasp_exit

    def spy(column, move, **kw):
        if case == "skipped":
            rt.arm.harness.halt("operator")   # a halt while the jaws were checked
            return real(column, move, **kw)
        return real(column, refuse if case == "refused" else lambda q: False, **kw)

    monkeypatch.setattr(rt, "_air_grasp_exit", spy)
    res = rt.skill_grasp_object("red cube")
    reason = "no safe joint route found; direct route: test refusal"
    expected = {"refused": f"refused: +3 cm: {reason}; +6 cm: {reason}",
                "did not settle": "did not settle (+3 cm)",
                "skipped": "skipped: motion halted: operator"}[case]
    assert res["exit_withdrawal"] == expected
    assert res["error"] == (AIR if case == "skipped" else f"{AIR}; exit withdrawal {expected}")


# ── no exit with a possible payload, an e-stop or a halt ────────────────────


def test_a_provisional_hold_never_runs_the_exit(rt, monkeypatch):
    state = _fail_first_lift(rt, monkeypatch)    # the lift fails after the close
    calls = []
    monkeypatch.setattr(rt, "_air_grasp_exit", lambda *a, **kw: calls.append(a) or "completed (+3 cm)")
    with pytest.raises(SkillError, match="did not settle at grasp lift pose"):
        rt.skill_grasp_object("red cube")
    assert state["failed"] == 1 and calls == []
    assert rt._held_provisional is not None      # still a possible payload


@pytest.mark.parametrize("stop", ["provisional hold", "carry attachment", "e-stop", "halt",
                                  "no kinematic model"])
def test_nothing_moves_when_the_exit_must_not_run(rt, monkeypatch, stop):
    harness, attachment = rt.arm.harness, [None]
    model = harness.kin
    monkeypatch.setattr(carry_attachment, "active", lambda runtime: attachment[0])
    latch, clear = {
        "provisional hold": (lambda: setattr(rt, "_held_provisional", ("red cube", "cube", "red")),
                             lambda: setattr(rt, "_held_provisional", None)),
        "carry attachment": (lambda: attachment.__setitem__(0, SimpleNamespace()),
                             lambda: attachment.__setitem__(0, None)),
        "e-stop": (lambda: harness.estop("operator"), harness.reset_estop),
        "halt": (lambda: harness.halt("operator"), harness.clear_halt),
        "no kinematic model": (lambda: setattr(harness, "kin", None),
                               lambda: setattr(harness, "kin", model)),
    }[stop]
    calls = []
    move = planned(rt, calls)
    column = column_at_tcp(rt)
    latch()
    n0 = commands(rt)
    out = rt._air_grasp_exit(column, move)
    assert out.startswith("skipped: "), out
    assert calls == [] and commands(rt) == n0 and harness._grasp_exempt is None
    clear()   # premise: the same exit lifts the arm once the condition is gone
    assert rt._air_grasp_exit(column, move) == "completed (+3 cm)" and commands(rt) > n0


# ── a stop during the exit ends it at once and propagates ───────────────────


@pytest.mark.parametrize("stop", ["halt", "e-stop"])
def test_halt_during_the_first_lift_never_tries_6cm(rt, monkeypatch, stop):
    """A halt or an e-stop mid-stream: the motion's own exception propagates,
    after the column is released."""
    raw, harness = rt.arm.raw, rt.arm.harness
    z0, n0 = tcp(rt)[2], commands(rt)
    real_send = raw.send_joint_target

    def send(q):
        real_send(q)
        if len(raw.commands) == n0 + 5:           # mid-stream, after 5 ticks
            harness.halt("operator") if stop == "halt" else harness.estop("operator")

    monkeypatch.setattr(raw, "send_joint_target", send)
    calls = []
    with pytest.raises(SafetyViolation, match="halted: operator" if stop == "halt" else "e-stop latched") as info:
        rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls))
    assert isinstance(info.value, MotionHalted) == (stop == "halt")
    assert len(calls) == 1 and commands(rt) == n0 + 5
    assert tcp(rt)[2] < z0 + .03 - 1e-3
    assert harness._grasp_exempt is None
    harness.reset_estop()


def test_a_safety_stop_during_the_first_lift_never_tries_6cm(rt):
    raised = ContactDetected("contact detected at joint 3: tracking error 0.300 rad for 80 ms",
                             event={"joint": 3})

    def contact(q):
        raise raised

    calls, n0 = [], commands(rt)
    with pytest.raises(ContactDetected) as info:
        rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, contact))
    assert info.value is raised and info.value.event == {"joint": 3}     # the original, as is
    assert len(calls) == 1 and commands(rt) == n0
    assert rt.arm.harness._grasp_exempt is None


@pytest.mark.parametrize("reason", [
    "planned route became unsafe: point 3 clearance 0.010 m below 0.030 m (occupancy map)",
    "perception watchdog: last observation 6.2s old",
])
def test_a_refusal_that_is_not_a_stop_stays_in_the_outcome(rt, reason):
    def refused(q):
        raise SafetyViolation(reason)

    calls, n0 = [], commands(rt)
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, refused))
    assert out == f"refused: +3 cm: {reason}"
    assert len(calls) == 1 and commands(rt) == n0 and rt.arm.harness._grasp_exempt is None


@pytest.mark.parametrize("cleared", [False, True], ids=["halted", "halt cleared"])
def test_a_halt_that_lands_with_a_refusal_is_raised(rt, cleared):
    """A lift refused for another reason (the route became unsafe) while the
    operator's halt arrived in the same move: the halt propagates, it is not
    folded into a "refused" string."""
    harness = rt.arm.harness

    def refused_under_a_halt(q):
        harness.halt("operator")
        if cleared:
            harness.clear_halt()      # withdrawn again: only its generation remains
        raise SafetyViolation("planned route became unsafe: point 3 clearance 0.010 m below 0.030 m")

    calls, n0 = [], commands(rt)
    with pytest.raises(MotionHalted, match="^halt received$" if cleared else "^motion halted: operator$"):
        rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, refused_under_a_halt))
    assert len(calls) == 1 and commands(rt) == n0 and harness._grasp_exempt is None


@pytest.mark.parametrize("stop", ["e-stop", "halt"])
def test_a_stop_during_a_refused_first_lift_never_tries_6cm(rt, stop):
    harness = rt.arm.harness

    def first(q):
        if stop == "e-stop":
            harness.estop("operator")        # the route preflight then refuses: SkillError
            return rt._move_planned(q, duration_s=2.0)
        harness.halt("operator")             # a halt, withdrawn again, while +3 cm was refused
        harness.clear_halt()
        refuse(q)

    calls, n0 = [], commands(rt)
    expected = (SafetyViolation, "^e-stop latched$") if stop == "e-stop" else (MotionHalted, "halt")
    with pytest.raises(expected[0], match=expected[1]) as info:
        rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, first))
    assert isinstance(info.value, MotionHalted) == (stop == "halt")
    assert len(calls) == 1 and commands(rt) == n0 and harness._grasp_exempt is None
    harness.reset_estop()


def test_a_stop_after_the_last_refused_lift_is_raised(rt, monkeypatch):
    """+3 cm unreachable, +6 cm refused by a preflight that met an e-stop:
    the e-stop propagates, not a "refused" string."""
    harness, real = rt.arm.harness, trajectory.lift_ik
    monkeypatch.setattr(trajectory, "lift_ik",
                        lambda kin, start, height, **kw: None if height == .03 else real(kin, start, height, **kw))

    def first(q):
        harness.estop("operator")
        return rt._move_planned(q, duration_s=2.0)

    calls, n0 = [], commands(rt)
    with pytest.raises(SafetyViolation, match="^e-stop latched$"):
        rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, first))
    assert len(calls) == 1 and commands(rt) == n0 and harness._grasp_exempt is None
    harness.reset_estop()


CONTACT = {"enabled": True, "mode": "enforce", "track_tol_rad": 0.10, "lag_s": 0.25,
           "debounce_samples": 2, "contact_zone_m": 0.05}


def test_a_contact_during_the_exit_reaches_the_result_with_its_record(rt, monkeypatch):
    """execute() attaches the trip record and the planner hint
    (`_contact_result`) to a contact during the exit, as to any other."""
    harness, raw, real = rt.arm.harness, rt.arm.raw, rt._air_grasp_exit
    harness.limits = dataclasses.replace(harness.limits, contact=ContactMonitorConfig.from_config(CONTACT))
    walled = []

    def wall_above(column, move, **kw):
        # something above the wrist: the joint the +3 cm lift moves most stops
        q0 = raw.get_state().q.copy()
        dq = trajectory.lift_ik(harness.kin, q0, .03) - q0
        j = int(np.argmax(np.abs(dq)))
        raw.block_joint(j, at=q0[j], direction=np.sign(dq[j]))
        walled.append(j)
        return real(column, move, **kw)

    monkeypatch.setattr(rt, "_air_grasp_exit", wall_above)
    res = rt.execute("grasp_object", {"label": "red cube"})
    (j,) = walled
    assert res["ok"] is False
    assert res["error"].startswith(f"ContactDetected: contact detected at joint {j + 1}"), res["error"]
    assert res["contact"]["joint"] == j + 1 and res["contact"]["hold_q"]
    assert "reset_scene" in res["hint"]
    assert harness._grasp_exempt is None


def test_a_halt_while_the_exit_solves_its_lift_moves_nothing(rt, monkeypatch):
    """A halt after the exit's own stop check: its move carries the grasp's
    halt generation and refuses before the first command."""
    harness, real = rt.arm.harness, trajectory.lift_ik

    def halting_lift_ik(kin, start, height, **kw):
        harness.halt("operator")
        return real(kin, start, height, **kw)

    moved, real_exit = [], rt._air_grasp_exit

    def spy(column, move, **kw):
        monkeypatch.setattr(trajectory, "lift_ik", halting_lift_ik)
        n0 = commands(rt)
        try:
            return real_exit(column, move, **kw)
        finally:
            moved.append(commands(rt) - n0)

    monkeypatch.setattr(rt, "_air_grasp_exit", spy)
    with pytest.raises(MotionHalted):
        rt.skill_grasp_object("red cube")
    assert moved == [0] and harness._grasp_exempt is None


def test_a_lift_that_does_not_settle_ends_the_exit(rt):
    calls = []
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, lambda q: False))
    assert out == "did not settle (+3 cm)" and len(calls) == 1


@pytest.mark.parametrize("ran", [True, False], ids=["ran", "skipped"])
def test_the_exit_phase_is_recorded_only_when_the_exit_runs(rt, monkeypatch, ran):
    phases, real_exit = [], rt._air_grasp_exit
    monkeypatch.setattr(grasp_evidence, "phase", phases.append)

    def spy(column, move, **kw):
        if not ran:
            rt.arm.harness.halt("operator")
        return real_exit(column, move, **kw)

    monkeypatch.setattr(rt, "_air_grasp_exit", spy)
    res = rt.skill_grasp_object("red cube")
    assert res["exit_withdrawal"].startswith("completed" if ran else "skipped")
    assert {"planning", "pregrasp", "verify"} <= set(phases)       # premise: phases are recorded
    assert ("exit_withdrawal" in phases) == ran


def test_the_exit_lifts_are_the_route_planners_lifts():
    assert SkillRuntime.AIR_GRASP_EXIT_LIFTS_M is trajectory.ROUTE_LIFT_HEIGHTS_M
    assert trajectory.ROUTE_LIFT_HEIGHTS_M == (0.03, 0.06)


# ── an operator stop during the close ends the grasp (A3 review, round 2) ───
#
# Pre-existing on main: on the mapping-only path the grasp's own lift after the
# close cleared a halt raised during the close (`begin_motion`), and the
# persistence loops then re-homed and retried after an operator STOP.


def stop_during_the_close(rt, monkeypatch, stop):
    """Press STOP (`halt`) or the e-stop while the jaws close. Returns the arm
    command count at the end of every close."""
    harness, real_close = rt.arm.harness, SkillRuntime._grasp_close
    after_close = []

    def close(self, *a, **kw):
        out = real_close(self, *a, **kw)
        harness.halt("operator") if stop == "halt" else harness.estop("operator")
        after_close.append(commands(rt))
        return out

    monkeypatch.setattr(SkillRuntime, "_grasp_close", close)
    return after_close


def recovery_spies(rt, monkeypatch):
    """Count grasp attempts, re-homes and re-scans; each spy calls the real one."""
    seen = {"grasps": 0, "homes": 0, "rescans": 0}
    for name, key in (("skill_grasp_object", "grasps"), ("skill_move_home", "homes"),
                      ("_reobserve", "rescans")):
        def spy(*a, _real=getattr(rt, name), _key=key, **k):
            seen[_key] += 1
            return _real(*a, **k)

        monkeypatch.setattr(rt, name, spy)
    return seen


#: the halt the grasp's own lift meets (begin_motion with the grasp-start generation)
LIFT_HALT = "halt received during route planning or execution"


def stop_error(res, stop, halt=f"MotionHalted: {LIFT_HALT}"):
    """The execute() result carries the stop: the propagated halt (the
    original exception, as is), or the latched e-stop."""
    assert res["ok"] is False, res
    if stop == "halt":
        assert res["error"] == halt, res["error"]
    else:
        assert "e-stop latched" in res["error"], res["error"]


GRASP = ("grasp_object", {"label": "red cube"})
HANDOVER = ("handover", {"label": "red cube"})            # _grasp_with_persistence
PICK = ("pick_and_place", {"object": "red cube"})         # its own persistence loop


@pytest.mark.parametrize("stop", ["halt", "e-stop"])
def test_a_stop_during_the_close_ends_the_grasp_before_its_lift(rt, monkeypatch, stop):
    """STOP (or the e-stop) while the jaws close: the grasp's own lift back to
    the hover carries the grasp-start halt generation, is refused before its
    first command, and the stop propagates. The exit never runs."""
    after_close = stop_during_the_close(rt, monkeypatch, stop)
    exits = []
    monkeypatch.setattr(rt, "_air_grasp_exit", lambda *a, **kw: exits.append(a) or "completed (+3 cm)")
    expected = (MotionHalted, f"^{LIFT_HALT}$") if stop == "halt" else (SafetyViolation, "^e-stop latched$")
    with pytest.raises(expected[0], match=expected[1]) as info:
        rt.skill_grasp_object("red cube")
    assert isinstance(info.value, MotionHalted) == (stop == "halt")
    assert len(after_close) == 1 and commands(rt) == after_close[0] and exits == []
    harness = rt.arm.harness
    assert harness.halted == ("operator" if stop == "halt" else None)   # no motion cleared it
    assert rt._held_provisional is not None     # the jaws' state is reconciled by the next skill
    harness.reset_estop()


@pytest.mark.parametrize("stop", ["halt", "e-stop"])
@pytest.mark.parametrize("skill, args", [GRASP, HANDOVER, PICK], ids=["grasp_object", "handover", "pick_and_place"])
def test_a_stop_during_the_close_ends_the_skill(rt, monkeypatch, skill, args, stop):
    """Through execute(): one attempt, no re-home, no re-scan, no command
    after the close, and the result names the stop."""
    after_close = stop_during_the_close(rt, monkeypatch, stop)
    seen = recovery_spies(rt, monkeypatch)
    res = rt.execute(skill, args)
    stop_error(res, stop)
    assert seen == {"grasps": 1, "homes": 0, "rescans": 0}
    assert len(after_close) == 1 and commands(rt) == after_close[0]
    rt.arm.harness.reset_estop()


@pytest.mark.parametrize("stop", ["halt", "e-stop"])
def test_a_stop_during_the_close_never_rehomes_a_scene_move(rt, monkeypatch, stop):
    """`_move_object_to` (restore_scene, search_for_object) re-homes after a
    failed grasp: not after an operator stop."""
    after_close = stop_during_the_close(rt, monkeypatch, stop)
    seen = recovery_spies(rt, monkeypatch)
    if stop == "halt":
        with pytest.raises(MotionHalted, match=f"^{LIFT_HALT}$"):
            rt._move_object_to("red cube", (0.30, 0.10), budget_s=30.0, kind="park")
    else:
        rec = rt._move_object_to("red cube", (0.30, 0.10), budget_s=30.0, kind="park")
        assert rec["ok"] is False and "e-stop latched" in rec["error"]
    assert seen == {"grasps": 1, "homes": 0, "rescans": 0}
    assert len(after_close) == 1 and commands(rt) == after_close[0]
    rt.arm.harness.reset_estop()


@pytest.mark.parametrize("stop", ["halt", "e-stop", "e-stop raised"])
def test_a_stop_ends_sort_by_color_before_the_next_object(rt, monkeypatch, stop):
    """sort_by_color grasps object after object: a stop during one ends the sort."""
    harness, calls = rt.arm.harness, []
    cubes = [SimpleNamespace(label="red cube", color="red", position=np.array([.29, .0, .037])),
             SimpleNamespace(label="blue cube", color="blue", position=np.array([.29, .08, .037]))]
    monkeypatch.setattr(rt.beliefs, "all", lambda: list(cubes))

    def grasp(query, **kw):
        calls.append(query)
        if stop == "halt":
            harness.halt("operator")
            raise MotionHalted(LIFT_HALT)
        harness.estop("operator")
        if stop == "e-stop raised":
            raise SafetyViolation("e-stop latched")
        return {"ok": False, "held": False, "error": "SafetyViolation: e-stop latched (e-stop latched; not retrying)"}

    monkeypatch.setattr(rt, "_grasp_with_persistence", grasp)
    seen = recovery_spies(rt, monkeypatch)
    res = rt.execute("sort_by_color", {})
    stop_error(res, "halt" if stop == "halt" else "e-stop")
    assert calls == ["red red cube"] and seen["homes"] == 0
    harness.reset_estop()


@pytest.mark.parametrize("cleared", [False, True], ids=["halted", "halt cleared"])
@pytest.mark.parametrize("skill, args", [HANDOVER, PICK], ids=["handover", "pick_and_place"])
def test_an_exit_skipped_by_a_halt_ends_the_persistence(rt, monkeypatch, skill, args, cleared):
    """A halt after the grasp's lift (while the jaws are checked): the exit is
    skipped, and the persistence loop neither re-homes nor tries again."""
    harness, real_exit, outcomes, at_exit = rt.arm.harness, rt._air_grasp_exit, [], []

    def halting_exit(column, move, **kw):
        harness.halt("operator")
        if cleared:
            harness.clear_halt()     # withdrawn again (a Resume): only its generation remains
        at_exit.append(commands(rt))
        outcomes.append(real_exit(column, move, **kw))
        return outcomes[-1]

    monkeypatch.setattr(rt, "_air_grasp_exit", halting_exit)
    seen = recovery_spies(rt, monkeypatch)
    res = rt.execute(skill, args)
    assert outcomes == ["skipped: halt received" if cleared else "skipped: motion halted: operator"]
    stop_error(res, "halt", halt="MotionHalted: " + ("halt received during the grasp; not retrying" if cleared
                                                    else "motion halted: operator; not retrying"))
    assert seen == {"grasps": 1, "homes": 0, "rescans": 0}
    assert commands(rt) == at_exit[0]


@pytest.mark.parametrize("skill, args, homes", [(*HANDOVER, 1), (*PICK, 2)], ids=["handover", "pick_and_place"])
def test_a_failed_grasp_without_a_stop_still_retries(rt, monkeypatch, skill, args, homes):
    """Pin: an air grasp without any stop keeps today's persistence (re-home,
    re-scan, a fresh attempt; pick_and_place also parks after its last)."""
    rt.cfg.grasp._data["max_pick_attempts"] = 2
    seen = recovery_spies(rt, monkeypatch)
    res = rt.execute(skill, args)
    assert res["ok"] is False and AIR in res["error"], res
    assert seen == {"grasps": 2, "homes": homes, "rescans": 1}


# ── an operator halt at any point of a motion skill ends it (A3 review, round 3) ──
#
# Pre-existing on main: a halt raised while grasp_object PLANNED was cleared by
# the next motion's begin_motion, so the opening, home, approach, descent and
# close still ran (310 commands) and only the lift refused. execute() now takes
# the halt generation when a motion skill starts, and every motion the skill
# issues carries it (the harness's halt-generation check, SafeArm's jaws and the
# backend's own two-stage close included). A halt is never escalated to an
# e-stop latch, and a halt raised before the skill started keeps today's rule
# (the skill's first motion clears it).

#: a normal mock grasp_object through execute(): opening, home (75), pregrasp,
#: descent, close, lift and the +3 cm exit (measured on 6275636)
NORMAL_GRASP_COMMANDS = 510
#: the halt a motion meets at its start (the skill's baseline)
SKILL_HALT = f"MotionHalted: {LIFT_HALT}"
#: the halt a streaming motion meets mid-stream (approve)
STREAM_HALT = "MotionHalted: halted: operator"


def jaw_commands(rt, monkeypatch):
    """Every command the jaws receive (the raw backend's set_gripper)."""
    raw, seen = rt.arm.raw, []
    real = raw.set_gripper

    def set_gripper(pos, *a, **k):
        seen.append(pos)
        return real(pos, *a, **k)

    monkeypatch.setattr(raw, "set_gripper", set_gripper)
    return seen


def halt_during_planning(rt, monkeypatch, jaws):
    """Press STOP once while grasp_object plans: after the skill started,
    before its first motion. Returns the (arm, jaw) command counts at the halt."""
    real_plan, at_halt = rt._plan_grasps, []

    def plan(*a, **k):
        out = real_plan(*a, **k)
        if not at_halt:
            rt.arm.harness.halt("operator")
            at_halt.append((commands(rt), len(jaws)))
        return out

    monkeypatch.setattr(rt, "_plan_grasps", plan)
    return at_halt


def halt_mid_stream(rt, monkeypatch, jaws, streaming, tick=5):
    """Press STOP at the `tick`-th approved step of the first move for which
    `streaming()` is true. Returns the (arm, jaw) command counts at the halt."""
    harness, at_halt, ticks = rt.arm.harness, [], [0]
    real_approve = harness.approve

    def approve(q_prev, q_next, dt, **kw):
        if streaming() and not at_halt:
            ticks[0] += 1
            if ticks[0] == tick:
                harness.halt("operator")
                at_halt.append((commands(rt), len(jaws)))
        return real_approve(q_prev, q_next, dt, **kw)

    monkeypatch.setattr(harness, "approve", approve)
    return at_halt


def grasp_phase(monkeypatch):
    """The grasp's current evidence phase ("home", "pregrasp", ...)."""
    current, real = [None], grasp_evidence.phase

    def phase(name, *a, **k):
        current[0] = name
        return real(name, *a, **k)

    monkeypatch.setattr(grasp_evidence, "phase", phase)
    return current


def nothing_after_the_halt(rt, at_halt, jaws):
    assert len(at_halt) == 1, at_halt
    assert (commands(rt), len(jaws)) == at_halt[0]        # no arm or jaw command after the halt
    assert rt.arm.harness.estopped is False               # STOP is not the e-stop


@pytest.mark.parametrize("skill, args", [GRASP, HANDOVER, PICK], ids=["grasp_object", "handover", "pick_and_place"])
def test_a_halt_during_grasp_planning_ends_the_skill(rt, monkeypatch, skill, args):
    """STOP while the grasp plans: the opening is refused before the jaws
    move, nothing moves after it, the halt reaches execute() as itself and
    nothing re-homes, re-scans or tries again."""
    jaws = jaw_commands(rt, monkeypatch)
    at_halt = halt_during_planning(rt, monkeypatch, jaws)
    seen = recovery_spies(rt, monkeypatch)
    res = rt.execute(skill, args)
    stop_error(res, "halt", halt=SKILL_HALT)
    nothing_after_the_halt(rt, at_halt, jaws)
    assert at_halt[0] == (0, 0)
    assert rt.arm.harness.halted == "operator"            # no motion cleared it
    assert seen == {"grasps": 1, "homes": 0, "rescans": 0}


@pytest.mark.parametrize("skill, args", [GRASP, HANDOVER, PICK], ids=["grasp_object", "handover", "pick_and_place"])
def test_a_halt_during_the_grasps_home_move_stays_a_halt(rt, monkeypatch, skill, args):
    """STOP while the grasp's home move (before the pregrasp) streams: the
    move stops, its error handler does not latch the e-stop, and the halt
    propagates as MotionHalted with nothing moving after it."""
    jaws = jaw_commands(rt, monkeypatch)
    phase = grasp_phase(monkeypatch)
    at_halt = halt_mid_stream(rt, monkeypatch, jaws, lambda: phase[0] == "home")
    seen = recovery_spies(rt, monkeypatch)
    res = rt.execute(skill, args)
    stop_error(res, "halt", halt=STREAM_HALT)
    nothing_after_the_halt(rt, at_halt, jaws)
    assert at_halt[0][1] == 1                             # only the opening before the home move
    assert seen == {"grasps": 1, "homes": 0, "rescans": 0}


@pytest.mark.parametrize("skill, args", [HANDOVER, PICK], ids=["handover", "pick_and_place"])
def test_a_halt_during_the_persistence_rehome_ends_the_skill(rt, monkeypatch, skill, args):
    """STOP while the retry loop re-homes after a failed (air) grasp: the
    re-home stops, the halt propagates, no re-scan and no second attempt."""
    rt.cfg.grasp._data["max_pick_attempts"] = 2
    jaws = jaw_commands(rt, monkeypatch)
    seen = recovery_spies(rt, monkeypatch)
    spy_home, rehoming = rt.skill_move_home, [False]

    def home(*a, **k):
        rehoming[0] = True
        try:
            return spy_home(*a, **k)
        finally:
            rehoming[0] = False

    monkeypatch.setattr(rt, "skill_move_home", home)
    at_halt = halt_mid_stream(rt, monkeypatch, jaws, lambda: rehoming[0])
    res = rt.execute(skill, args)
    stop_error(res, "halt", halt=STREAM_HALT)
    nothing_after_the_halt(rt, at_halt, jaws)
    assert seen == {"grasps": 1, "homes": 1, "rescans": 0}


def test_a_halt_before_the_raw_two_stage_close_keeps_the_jaws_open(rt, monkeypatch):
    """The real reBot closes through its backend's own two-stage close
    (outside SafeArm.set_gripper): a halt after the descent never closes it."""
    raw, closes = rt.arm.raw, []
    monkeypatch.setattr(raw, "close_gripper_two_stage", lambda **kw: closes.append(kw), raising=False)
    jaws = jaw_commands(rt, monkeypatch)
    real_move, at_halt = rt._move_joints, []

    def move(q, *a, **kw):
        out = real_move(q, *a, **kw)
        if kw.get("bias_compensate"):                     # the descent
            rt.arm.harness.halt("operator")
            at_halt.append((commands(rt), len(jaws)))
        return out

    monkeypatch.setattr(rt, "_move_joints", move)
    res = rt.execute(*GRASP)
    stop_error(res, "halt", halt=SKILL_HALT)
    nothing_after_the_halt(rt, at_halt, jaws)
    assert closes == []


def test_a_halt_while_execute_prepares_the_skill_ends_it(rt, monkeypatch):
    """The skill starts when execute() takes it: a halt while execute() saves
    the BEFORE keyframe (the skill body has not run yet) ends it too."""
    jaws = jaw_commands(rt, monkeypatch)
    real_save, at_halt = rt.trace.save_keyframe, []

    def save(rgb, name, *a, **k):
        out = real_save(rgb, name, *a, **k)
        if name == "grasp_object_before":
            rt.arm.harness.halt("operator")
            at_halt.append((commands(rt), len(jaws)))
        return out

    monkeypatch.setattr(rt.trace, "save_keyframe", save)
    res = rt.execute(*GRASP)
    stop_error(res, "halt", halt=SKILL_HALT)
    nothing_after_the_halt(rt, at_halt, jaws)


def test_a_halt_while_pick_and_place_carries_ends_the_skill(rt, monkeypatch):
    """pick_and_place holds the cube and carries it to the place: a STOP
    during the carry ends the skill with the cube held, instead of a re-home
    and a place retry."""
    rt.arm.raw.object_stop_frac = 0.5                     # the jaws stall on the cube: held
    jaws = jaw_commands(rt, monkeypatch)
    seen = recovery_spies(rt, monkeypatch)
    real_place, placing = rt.skill_place_at, [False]

    def place(*a, **k):
        placing[0] = True
        try:
            return real_place(*a, **k)
        finally:
            placing[0] = False

    monkeypatch.setattr(rt, "skill_place_at", place)
    at_halt = halt_mid_stream(rt, monkeypatch, jaws, lambda: placing[0])
    res = rt.execute(*PICK)
    stop_error(res, "halt", halt=STREAM_HALT)
    nothing_after_the_halt(rt, at_halt, jaws)
    assert seen == {"grasps": 1, "homes": 0, "rescans": 0} and rt.held_object == "red cube"


def test_a_nested_motion_skill_keeps_the_outer_skills_baseline(rt, monkeypatch):
    """A motion skill dispatched inside another keeps the OUTER skill's start:
    a halt between the two starts refuses the inner skill's motion and, after
    it returns, the outer skill's next motion."""
    harness, seen = rt.arm.harness, {}
    home = np.asarray(rt.cfg.arm.get("home_q"), dtype=float)

    def wave(*a, **k):
        harness.halt("operator")
        seen["inner"] = rt.execute("move_home", {})
        seen["after inner"] = commands(rt)
        rt._move_planned(home, duration_s=1.0)
        return {"ok": True}

    monkeypatch.setattr(rt, "skill_wave", wave)
    n0 = commands(rt)
    res = rt.execute("wave", {})
    assert seen["inner"]["ok"] is False and seen["inner"]["error"] == SKILL_HALT, seen["inner"]
    stop_error(res, "halt", halt=SKILL_HALT)
    assert commands(rt) == seen["after inner"] == n0 and harness.estopped is False


@pytest.mark.parametrize("before", ["pending", "cleared by a motion", "after a halted skill"])
def test_a_halt_before_the_skill_started_does_not_block_it(rt, monkeypatch, before):
    """Pin today's rule: a halt raised before the skill started (pending, or
    cleared by an earlier motion) does not block it, and a skill that ended
    on a halt leaves no baseline behind. The grasp runs exactly as a normal
    one (same command count)."""
    harness = rt.arm.harness
    if before == "after a halted skill":
        jaws = jaw_commands(rt, monkeypatch)
        halt_during_planning(rt, monkeypatch, jaws)       # halts the first planning only
        stop_error(rt.execute(*GRASP), "halt", halt=SKILL_HALT)
    else:
        harness.halt("earlier")
        if before == "cleared by a motion":
            assert rt.execute("move_home", {})["ok"] is True and harness.halted is None
    n0 = commands(rt)
    res = rt.execute(*GRASP)
    assert res["ok"] is False and AIR in res["error"] and res["exit_withdrawal"] == "completed (+3 cm)", res
    assert commands(rt) - n0 == NORMAL_GRASP_COMMANDS and harness.estopped is False


def test_a_normal_grasp_is_unchanged(rt):
    """Pin: a grasp without any stop streams the same commands as before."""
    res = rt.execute(*GRASP)
    assert res["ok"] is False and AIR in res["error"] and res["exit_withdrawal"] == "completed (+3 cm)", res
    assert commands(rt) == NORMAL_GRASP_COMMANDS


@pytest.mark.parametrize("pending", [True, False], ids=["halted", "halt cleared"])
def test_a_halt_before_the_attachment_is_reported_as_a_halt(pending):
    """Isaac carry attachment: a caller whose baseline predates the attachment
    (a halt arrived before the close) is refused as a halt, not as a
    mismatched token, and the attachment stays failed (fail closed)."""
    from test_carry_attachment import add_attachment, episode, runtime

    att_rt = runtime()
    harness = att_rt.arm.harness
    grasp_start = harness._halt_generation
    harness.halt("operator")
    if not pending:
        harness.clear_halt()                              # a motion cleared it before the close
    state = add_attachment(att_rt.arm.get_state(), att_rt.arm.raw._signs)
    carry_attachment.arm(att_rt, episode(att_rt, state), state)
    att_rt.arm.move_joints = lambda *a, **k: pytest.fail("no motion after a halt")
    reason = "operator" if pending else LIFT_HALT
    with pytest.raises(carry_attachment.AttachmentInvalid, match=f"^halted: {reason}$"):
        carry_attachment.move(att_rt, state.q, _halt_generation=grasp_start)
    assert carry_attachment.active(att_rt).failure == f"halted: {reason}"
    with pytest.raises(carry_attachment.AttachmentInvalid, match=f"^halted: {reason}$"):
        carry_attachment.move(att_rt, state.q)            # the failure is latched


# ── a refused or unreachable +3 cm lift tries +6 cm ─────────────────────────


def test_a_refused_3cm_lift_tries_6cm(rt):
    z0, calls = tcp(rt)[2], []
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, refuse))
    assert out == "completed (+6 cm)" and len(calls) == 2
    assert abs(tcp(rt)[2] - (z0 + .06)) < 2e-3


def test_an_unreachable_3cm_lift_tries_6cm(rt, monkeypatch):
    from cascade.safety import trajectory

    real = trajectory.lift_ik
    monkeypatch.setattr(trajectory, "lift_ik",
                        lambda kin, start, height, **kw: None if height == .03 else real(kin, start, height, **kw))
    z0, calls = tcp(rt)[2], []
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls))
    assert out == "completed (+6 cm)" and len(calls) == 1
    assert abs(tcp(rt)[2] - (z0 + .06)) < 2e-3


# ── the column is released on every path ───────────────────────────────────


@pytest.mark.parametrize("path", ["completed", "refused", "exception"])
def test_the_column_is_released_on_every_path(rt, path):
    harness, column, held = rt.arm.harness, column_at_tcp(rt), []

    def move(q):
        held.append(harness._grasp_exempt)
        if path == "refused":
            refuse(q)
        if path == "exception":
            raise RuntimeError("feedback lost")
        return rt._move_planned(q, duration_s=2.0)

    if path == "exception":
        with pytest.raises(RuntimeError, match="feedback lost"):
            rt._air_grasp_exit(column, move)
    else:
        assert rt._air_grasp_exit(column, move).startswith(path)
    assert held and all(h is not None and np.array_equal(h[0], column[0]) for h in held)
    assert harness._grasp_exempt is None


# ── lift_ik ─────────────────────────────────────────────────────────────────


class _Kin:
    def __init__(self, result):
        self.result, self.calls = result, []

    def fk(self, q):
        T = np.eye(4)
        T[:3, :3] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        T[:3, 3] = q[:3]
        return T

    def ik(self, target, q_init, max_iter, retries):
        self.calls.append((target.copy(), q_init.copy(), max_iter, retries))
        return self.result


def test_lift_ik_raises_the_tcp_in_base_z_at_the_start_orientation():
    from cascade.safety.trajectory import lift_ik

    start = np.array([.3, .1, .2, 0., 0., 0.])
    kin = _Kin(SimpleNamespace(success=True, q=np.array([.3, .1, .23, 0., 0., 0.])))
    np.testing.assert_array_equal(lift_ik(kin, start, .03), [.3, .1, .23, 0., 0., 0.])
    ((target, q_init, max_iter, retries),) = kin.calls
    expected = kin.fk(start)
    expected[2, 3] += .03
    np.testing.assert_array_equal(target, expected)
    np.testing.assert_array_equal(q_init, start)
    assert (max_iter, retries) == (80, 0)


@pytest.mark.parametrize("result", [SimpleNamespace(success=False, q=np.zeros(6)),
                                    SimpleNamespace(success=True, q=np.array([.3, np.nan, .23, 0., 0., 0.]))],
                         ids=["failed", "non-finite"])
def test_lift_ik_returns_none_on_failure_or_a_non_finite_result(result):
    from cascade.safety.trajectory import lift_ik

    assert lift_ik(_Kin(result), np.array([.3, .1, .2, 0., 0., 0.]), .03) is None
