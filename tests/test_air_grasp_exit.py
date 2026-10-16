"""After an air grasp the arm withdraws up through its approach column (spec §3.2 rules 1-2).

An air grasp (jaws verified empty) ends with the arm at the pregrasp hover,
above the target. The target's top in the map can hold that hover inside the
3 cm margin, where every later move except a strict retreat is refused. Before
returning, the runtime re-enters the SAME target column the approach entered
(`_column_exemption`, None on the observed-finger path) and lifts the TCP
+3 cm, else +6 cm, through a fully vetted route. Every other gate keeps its
strength, and the column is released on every path.

Nothing moves with a possible payload (a provisional hold, an active carry
attachment: "do not invent an empty-tool recovery"), with a latched e-stop or
a pending halt. A halt, an e-stop or any safety stop during the exit ends it
at once; only a preflight refusal (SkillError) or an unreachable lift moves on
to the next height.

Every test runs the real runtime with the mock arm (no hardware, no GPU).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.skills import carry_attachment
from cascade.skills.runtime import SkillRuntime
from cascade.types import ContactDetected, SkillError
from conftest import needs_pin
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map
from test_b36_pick_recovery import _fail_first_lift

pytestmark = needs_pin

AIR = "air grasp: gripper closed fully, object not held"


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
    the pregrasp hover. Returns the (column, move) the grasp handed over."""
    seen = {}

    def capture(column, move):
        seen.update(column=column, move=move)
        return "skipped: captured by the test"

    monkeypatch.setattr(rt, "_air_grasp_exit", capture)
    res = rt.skill_grasp_object("red cube")
    assert res["exit_withdrawal"] == "skipped: captured by the test"
    return seen["column"], seen["move"]


# ── the exit after a real air grasp ─────────────────────────────────────────


def test_air_grasp_withdraws_from_the_pregrasp(rt, monkeypatch):
    starts = []
    real = rt._air_grasp_exit

    def spy(column, move):
        starts.append(rt.arm.get_state().q.copy())
        return real(column, move)

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

    def column_spy(cylinder):
        entered.append((in_exit[0], cylinder))
        return real_column(cylinder)

    monkeypatch.setattr(harness, "target_column_exemption", column_spy)
    real_exit = rt._air_grasp_exit

    def exit_spy(column, move):
        def watched(q):
            during_move.append(harness._grasp_exempt)
            return move(q)

        in_exit[0] = True
        try:
            return real_exit(column, watched)
        finally:
            in_exit[0] = False

    monkeypatch.setattr(rt, "_air_grasp_exit", exit_spy)
    res = rt.skill_grasp_object("red cube")
    assert res["exit_withdrawal"] == "completed (+3 cm)"
    approach = [c for inside, c in entered if not inside]
    in_the_exit = [c for inside, c in entered if inside]
    assert approach, "premise: the approach entered its target column"
    # entered once, with the very column of this grasp's approach
    assert len(in_the_exit) == 1 and in_the_exit[0] is approach[-1]
    (held,) = during_move
    center, radius, z_min = approach[-1]
    assert held is not None and np.array_equal(held[0], center) and held[1:] == (radius, z_min)
    assert harness._grasp_exempt is None                     # released afterwards


def test_exit_refused_by_a_neighbour_above_keeps_the_arm_put(rt, monkeypatch):
    from cascade.safety.trajectory import lift_ik

    rt.cfg.grasp._data["approach_target_exemption"] = True
    column, move = air_grasp_without_exit(rt, monkeypatch)
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
    occ = make_map(GridBridge([TABLE, neighbour], voxel=.01), required=False,
                   unobserved="free", max_age_s=600.0)
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.last_error is None and occ.usable()
    harness.occupancy = occ
    with harness.target_column_exemption(column):
        assert harness.route_end_violation(q0) is None                # premise: the hover is clear
        assert "occupancy map" in harness.route_end_violation(q3)     # +3 cm would end inside the margin

    n0 = commands(rt)
    out = SkillRuntime._air_grasp_exit(rt, column, move)
    assert out.startswith("refused: +3 cm: ") and "; +6 cm: " in out and "occupancy map" in out
    assert commands(rt) == n0
    np.testing.assert_array_equal(rt.arm.get_state().q, q0)
    assert harness._grasp_exempt is None


@pytest.mark.parametrize("case", ["refused", "did not settle", "skipped"])
def test_the_air_grasp_error_names_a_failed_exit_only(rt, monkeypatch, case):
    real = rt._air_grasp_exit

    def spy(column, move):
        if case == "skipped":
            rt.arm.harness.halt("operator")   # a halt while the jaws were checked
            return real(column, move)
        return real(column, refuse if case == "refused" else lambda q: False)

    monkeypatch.setattr(rt, "_air_grasp_exit", spy)
    res = rt.skill_grasp_object("red cube")
    reason = "no safe joint route found; direct route: test refusal"
    expected = {"refused": f"refused: +3 cm: {reason}; +6 cm: {reason}",
                "did not settle": "did not settle",
                "skipped": "skipped: motion halted: operator"}[case]
    assert res["exit_withdrawal"] == expected
    assert res["error"] == (AIR if case == "skipped" else f"{AIR}; exit withdrawal {expected}")


# ── no exit with a possible payload, an e-stop or a halt ────────────────────


def test_a_provisional_hold_never_runs_the_exit(rt, monkeypatch):
    state = _fail_first_lift(rt, monkeypatch)    # the lift fails after the close
    calls = []
    monkeypatch.setattr(rt, "_air_grasp_exit", lambda *a: calls.append(a) or "completed (+3 cm)")
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


# ── a stop during the exit ends it at once ──────────────────────────────────


def test_halt_during_the_first_lift_never_tries_6cm(rt, monkeypatch):
    raw, harness = rt.arm.raw, rt.arm.harness
    z0, n0 = tcp(rt)[2], commands(rt)
    real_send = raw.send_joint_target

    def send(q):
        real_send(q)
        if len(raw.commands) == n0 + 5:
            harness.halt("operator")         # mid-stream, after 5 ticks

    monkeypatch.setattr(raw, "send_joint_target", send)
    calls = []
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls))
    assert out.startswith("refused: +3 cm: halted: operator") and "+6 cm" not in out, out
    assert len(calls) == 1 and commands(rt) == n0 + 5
    assert tcp(rt)[2] < z0 + .03 - 1e-3
    assert harness._grasp_exempt is None


def test_a_safety_stop_during_the_first_lift_never_tries_6cm(rt):
    def contact(q):
        raise ContactDetected("contact detected at joint 3: tracking error 0.300 rad for 80 ms")

    calls, n0 = [], commands(rt)
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, contact))
    assert out.startswith("refused: +3 cm: contact detected") and "+6 cm" not in out, out
    assert len(calls) == 1 and commands(rt) == n0


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
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, first))
    assert out.startswith("refused: +3 cm: ") and "+6 cm not tried" in out, out
    assert len(calls) == 1 and commands(rt) == n0
    harness.reset_estop()


def test_a_lift_that_does_not_settle_ends_the_exit(rt):
    calls = []
    out = rt._air_grasp_exit(column_at_tcp(rt), planned(rt, calls, lambda q: False))
    assert out == "did not settle" and len(calls) == 1


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
                        lambda kin, start, height: None if height == .03 else real(kin, start, height))
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
