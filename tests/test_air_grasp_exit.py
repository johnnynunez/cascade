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
stays in the outcome string. Only a preflight refusal (SkillError) or an
unreachable lift moves on to the next height.

Every test runs the real runtime with the mock arm (no hardware, no GPU).
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
from test_b36_pick_recovery import _fail_first_lift

pytestmark = needs_pin

AIR = "air grasp: gripper closed fully, object not held"
#: the exit column's ceiling above the hover TCP (A3 review I2)
CEILING_M = 0.005


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
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


def test_a_halt_during_the_close_stops_the_exit(rt, monkeypatch):
    """STOP pressed while the jaws close: the lift back to the hover is the
    grasp's own, but no NEW motion (the exit) starts after it."""
    harness, real_close = rt.arm.harness, SkillRuntime._grasp_close

    def close(self, *a, **kw):
        out = real_close(self, *a, **kw)
        harness.halt("operator")
        return out

    monkeypatch.setattr(SkillRuntime, "_grasp_close", close)
    moved, real_exit = [], rt._air_grasp_exit

    def spy(column, move, **kw):
        n0 = commands(rt)
        try:
            return real_exit(column, move, **kw)
        finally:
            moved.append(commands(rt) - n0)

    monkeypatch.setattr(rt, "_air_grasp_exit", spy)
    res = rt.skill_grasp_object("red cube")
    assert res["exit_withdrawal"] == "skipped: halt received" and moved == [0]
    assert res["error"] == AIR


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
