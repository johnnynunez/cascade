"""An operator STOP below the pregrasp keeps the approach column for the next
explicit move_home / reset_scene, which retraces up inside it (follow-up to
PR #354, "Known limitation"; final-review probe_pick.py halt_close).

A STOP right after the jaws close (or anywhere below the pregrasp hover:
descent, close, lift) ends the grasp with 0 further commands, the TCP inside
the target. With the grasp's exemptions released, no short step from inside
the object strictly gains clearance, so every later move was refused: a
frozen arm. Ruling: keep the approach column + the pregrasp after such a
stop; the NEXT EXPLICIT move_home / reset_scene retraces upward inside that
column (capped 5 mm above the hover, fully vetted, every step strictly upward
inside the kept column), then goes home. Never a move without that explicit
call; a halt / e-stop during the retrace sends nothing more; any other motion
skill, a scene change or a new grasp forfeits the column.

The reBot profile (rebot_rs: approach_target_exemption + pregrasp_map_top) on
the mock stack, a 1 cm GridBridge map holding the table and the red box with
its MAP top a few mm above the observed 5 cm top (as on the rig), or the same
box as a hollow 4 mm shell read through a real EDT (an nvblox-like ESDF).
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.perception import occupancy as occ_mod
from cascade.skills.runtime import SkillRuntime
from cascade.types import ContactDetected, MotionHalted, SafetyViolation, SkillError
from conftest import needs_pin
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map
from owned_server import held_dead_port

pytestmark = needs_pin

MAP_TOP = 0.058
BOX = ((0.262, -0.037, 0.0), (0.317, 0.037, MAP_TOP))
SHELL_T = 0.004
#: the kept column's ceiling above the hover TCP (the A3 exit's cap)
CEILING_M = 0.005
#: the halt the grasp's own lift meets after a STOP during the close
LIFT_HALT = "halt received during route planning or execution"


class EdtBridge(GridBridge):
    """Each grid node reads its distance to the nearest OCCUPIED voxel centre
    (the real backends' ESDF): inside a hollow shell the clearance is finite."""

    def _grid(self, lo, hi):
        out = super()._grid(lo, hi)
        points = np.asarray(out["points"], dtype=float)
        if len(points):
            occupied = np.zeros(out["grid"].shape, dtype=bool)
            idx = np.round((points - np.asarray(out["origin"], dtype=float)) / self.voxel).astype(int)
            occupied[tuple(idx.T)] = True
            out["grid"] = ndimage.distance_transform_edt(~occupied, sampling=self.voxel).astype(np.float32)
        return out


def shell_boxes():
    (x0, y0, z0), (x1, y1, z1) = BOX
    t = SHELL_T
    return [((x0, y0, z1 - t), (x1, y1, z1)), ((x0, y0, z0), (x0 + t, y1, z1)), ((x1 - t, y0, z0), (x1, y1, z1)),
            ((x0, y0, z0), (x1, y0 + t, z1)), ((x0, y1 - t, z0), (x1, y1, z1))]


def box_map(kind="solid", *extra):
    if kind == "shell":
        bridge = EdtBridge([TABLE, *shell_boxes(), *extra], voxel=0.01)
    else:
        bridge = GridBridge([TABLE, BOX, *extra], voxel=0.01)
    occ = make_map(bridge, required=True, unobserved="free", max_age_s=1e6)
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.last_error is None and occ.usable()
    return occ


@pytest.fixture
def rebot(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
    monkeypatch.setattr(occ_mod.OccupancyMap, "from_config", classmethod(
        lambda cls, *a, **k: make_map(GridBridge([TABLE], voxel=0.01), required=False,
                                      unobserved="free", max_age_s=1e6)))
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
        cfg.arm._data["type"] = "mock"
        runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
        try:
            runtime.arm.harness.occupancy = box_map()
            mock_of(runtime).object_stop_frac = None
            yield runtime
        finally:
            shutdown_runtime(runtime, arm)


def mock_of(rt):
    raw = rt.arm.raw
    return raw.__dict__.get("_arm", raw)


def commands(rt):
    return len(mock_of(rt).commands)


def tcp(rt, q=None):
    kin = rt.arm.harness.kin
    return kin.fk(np.asarray(rt.arm.get_state().q if q is None else q, dtype=float))[:3, 3]


def min_arm_clearance(rt):
    h = rt.arm.harness
    q = np.asarray(rt.arm.get_state().q, float)
    points = np.vstack([h.kin.fk(q)[:3, 3][None, :], h.kin.link_positions(q)[1:]])
    return float(np.min(h.occupancy.clearance(points)))


def stop_after_the_close(rt, monkeypatch, stop="halt"):
    """Press STOP (`halt`) or the e-stop once, right after the jaws close.
    Returns [arm command count at the stop]."""
    harness, real_close = rt.arm.harness, SkillRuntime._grasp_close
    at = []

    def close(self, *a, **kw):
        out = real_close(self, *a, **kw)
        if not at:
            harness.halt("operator") if stop == "halt" else harness.estop("operator")
            at.append(commands(rt))
        return out

    monkeypatch.setattr(SkillRuntime, "_grasp_close", close)
    return at


def stopped_at_the_object(rt, monkeypatch, stop="halt", skill=("pick_and_place", {"object": "red cube"})):
    """The final review's halt_close: a pick stopped right after the close.
    The arm stays inside the target, nothing moved after the stop."""
    at = stop_after_the_close(rt, monkeypatch, stop)
    res = rt.execute(*skill)
    assert res["ok"] is False, res
    if stop == "halt":
        assert res["error"] == f"MotionHalted: {LIFT_HALT}", res["error"]
    else:
        assert "e-stop latched" in res["error"], res["error"]
    assert len(at) == 1 and commands(rt) == at[0]               # 0 commands after the STOP
    assert tcp(rt)[2] < MAP_TOP                                  # the TCP is inside the target
    assert min_arm_clearance(rt) < 0.03
    return at[0]


def at_home(rt):
    home = np.asarray(rt.cfg.arm.get("home_q"), dtype=float)
    return float(np.max(np.abs(np.asarray(rt.arm.get_state().q, float) - home))) < 0.02


def assert_strictly_up_inside(rt, qs, column, z_max):
    """Every commanded step moves the TCP strictly upward, inside the column."""
    center, radius, z_min = column
    zs = []
    for q in qs:
        p = tcp(rt, q)
        assert np.linalg.norm(p[:2] - np.asarray(center)[:2]) <= radius + 1e-6, p
        assert z_min - 1e-6 <= p[2] <= z_max + 1e-6, (p, z_min, z_max)
        zs.append(p[2])
    assert len(zs) > 5 and all(b > a for a, b in zip(zs, zs[1:])), zs[:5]


# ── RED (final review, Important #1): STOP at the object, then home ─────────


def approvals(rt, monkeypatch):
    """[(q_next, the active grasp exemption)] of every approved step."""
    harness = rt.arm.harness
    real, seen = harness.approve, []

    def approve(q_prev, q_next, dt, **kw):
        out = real(q_prev, q_next, dt, **kw)
        seen.append((np.asarray(q_next, dtype=float).copy(), harness._grasp_exempt))
        return out

    monkeypatch.setattr(harness, "approve", approve)
    return seen


@pytest.mark.parametrize("kind", ["solid", "shell"])
def test_move_home_after_a_stop_at_the_object_retraces_the_column_then_goes_home(rebot, monkeypatch, kind):
    rt = rebot
    rt.arm.harness.occupancy = box_map(kind)
    stopped_at_the_object(rt, monkeypatch)
    z0 = tcp(rt)[2]
    seen = approvals(rt, monkeypatch)
    res = rt.execute("move_home", {})
    assert res["ok"] is True, res
    assert at_home(rt)
    assert res["stop_retrace"].startswith("completed"), res
    # The first motion is the retrace: every step under the kept column
    # (capped: a 4-tuple), strictly upward inside it, ending at the hover.
    retrace = [q for q, exempt in seen if exempt is not None]
    assert len(retrace) > 5 and [e is not None for _, e in seen] == [True] * len(retrace) + [False] * (
        len(seen) - len(retrace))
    center, radius, z_min, z_max = seen[0][1]
    assert z_min < z0 < z_max - 0.03
    assert_strictly_up_inside(rt, retrace, (center, radius, z_min), z_max)
    assert tcp(rt, retrace[-1])[2] == pytest.approx(z_max - CEILING_M, abs=1e-9)   # the pregrasp hover
    from cascade.skills import stop_retrace

    assert stop_retrace.kept(rt) is None                          # consumed


def test_reset_scene_after_a_stop_at_the_object_retraces_then_resets(rebot, monkeypatch):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    res = rt.execute("reset_scene", {})
    assert res.get("stage") is None and "could not reach home" not in str(res.get("error")), res
    assert at_home(rt)
    from cascade.skills import stop_retrace

    assert stop_retrace.kept(rt) is None


def test_after_an_estop_at_the_object_the_explicit_home_retraces_once_cleared(rebot, monkeypatch):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch, stop="e-stop")
    n0 = commands(rt)
    res = rt.execute("move_home", {})                            # still latched: refused
    assert res["ok"] is False and "e-stop latched" in res["error"], res
    assert commands(rt) == n0
    rt.arm.harness.reset_estop()
    res = rt.execute("move_home", {})
    assert res["ok"] is True and res["stop_retrace"].startswith("completed"), res
    assert at_home(rt)


def test_the_retrace_runs_with_jaws_that_may_hold_the_object(rebot, monkeypatch):
    """The jaws stalled on the object: the STOP left a provisional hold. The
    explicit home retraces the interrupted lift (no payload gate changes)."""
    rt = rebot
    mock_of(rt).object_stop_frac = 0.5
    stopped_at_the_object(rt, monkeypatch, skill=("grasp_object", {"label": "red cube"}))
    assert rt._held_provisional is not None
    res = rt.execute("move_home", {})
    assert res["ok"] is True and res["stop_retrace"].startswith("completed"), res
    assert at_home(rt)


# ── never automatic ─────────────────────────────────────────────────────────


def test_nothing_moves_until_an_explicit_call_and_an_internal_home_never_retraces(rebot, monkeypatch):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    n0 = commands(rt)
    rt.execute("get_observation", {})
    assert commands(rt) == n0
    # A home move that is not execute()'s root move_home / reset_scene (a
    # composite's re-home, a direct call): today's route, refused, no retrace.
    with pytest.raises(SkillError, match="no safe joint route found"):
        rt.skill_move_home()
    assert commands(rt) == n0
    assert stop_retrace.kept(rt) is not None                      # still kept for the explicit call
    res = rt.execute("move_home", {})
    assert res["ok"] is True and res["stop_retrace"].startswith("completed"), res


# ── a new stop during the retrace ───────────────────────────────────────────


def stop_during_the_retrace(rt, monkeypatch, stop, tick=5):
    """Press STOP / the e-stop / a contact at the `tick`-th approved step of
    the next streamed motion. Returns [command count at the stop]."""
    harness = rt.arm.harness
    real = harness.approve
    seen, at = [0], []

    def approve(q_prev, q_next, dt, **kw):
        seen[0] += 1
        if seen[0] == tick and not at:
            at.append(commands(rt))
            if stop == "halt":
                harness.halt("operator")
            elif stop == "e-stop":
                harness.estop("operator")
            else:
                raise ContactDetected("contact detected (test): joint 2 blocked")
        return real(q_prev, q_next, dt, **kw)

    monkeypatch.setattr(harness, "approve", approve)
    return at


@pytest.mark.parametrize("stop", ["halt", "e-stop"])
def test_a_stop_during_the_retrace_sends_nothing_more_and_the_next_call_resumes(rebot, monkeypatch, stop):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    at = stop_during_the_retrace(rt, monkeypatch, stop)
    z_before = tcp(rt)[2]
    res = rt.execute("move_home", {})
    assert res["ok"] is False, res
    assert len(at) == 1 and commands(rt) == at[0]                 # 0 commands after the stop
    assert ("MotionHalted" in res["error"]) if stop == "halt" else ("e-stop latched" in res["error"])
    assert z_before < tcp(rt)[2] < MAP_TOP + 0.03                 # partly retraced, still below the hover
    assert stop_retrace.kept(rt) is not None                       # kept for the next explicit call
    monkeypatch.setattr(rt.arm.harness, "approve", type(rt.arm.harness).approve.__get__(rt.arm.harness))
    rt.arm.harness.reset_estop()
    res = rt.execute("move_home", {})
    assert res["ok"] is True and res["stop_retrace"].startswith("completed"), res
    assert at_home(rt)


def test_a_contact_during_the_retrace_propagates_and_forfeits_the_column(rebot, monkeypatch):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    at = stop_during_the_retrace(rt, monkeypatch, "contact")
    res = rt.execute("move_home", {})
    assert res["ok"] is False and res["error"].startswith("ContactDetected"), res
    # nothing but the contact monitor's hold at the measured pose (the backend's contact rule)
    held = mock_of(rt).commands[at[0]:]
    assert len(held) <= 1 and all(np.allclose(q, rt.arm.get_state().q) for q in held)
    assert stop_retrace.kept(rt) is None


# ── expiry ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("skill, args", [("move_relative", {"direction": "up", "distance_m": 0.03}),
                                         ("open_gripper", {})], ids=["move_relative", "open_gripper"])
def test_any_other_motion_skill_forfeits_the_column(rebot, monkeypatch, skill, args):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    rt.execute(skill, args)
    assert stop_retrace.kept(rt) is None
    n0 = commands(rt)
    res = rt.execute("move_home", {})                              # main's behaviour: frozen
    assert res["ok"] is False and "no safe joint route found" in res["error"], res
    assert commands(rt) == n0


def test_a_new_grasp_forfeits_the_column(rebot, monkeypatch):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    monkeypatch.setattr(rt, "_reconcile_held", lambda: (_ for _ in ()).throw(SkillError("test: stop here")))
    with pytest.raises(SkillError, match="test: stop here"):
        SkillRuntime.skill_grasp_object(rt, "red cube")             # a direct call (composites)
    assert stop_retrace.kept(rt) is None


def test_a_scene_change_forfeits_the_column(rebot, monkeypatch):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    rt.arm.harness.occupancy.scene_reset_generation += 1
    n0 = commands(rt)
    res = rt.execute("move_home", {})
    assert res["ok"] is False and "no safe joint route found" in res["error"], res
    assert commands(rt) == n0 and stop_retrace.kept(rt) is None


def test_a_reset_that_changed_the_scene_forgets_the_column(rebot):
    """reset_scene called outside execute() (not an explicit retrace) from
    home: its home move needs no retrace; once it went home the scene
    changes, and the column is gone."""
    rt = rebot
    from cascade.skills import stop_retrace

    q = np.asarray(rt.arm.get_state().q, float)
    kept = stop_retrace.Kept(arm=rt.arm, raw=rt.arm.raw, column=(tcp(rt)[:2].copy(), 0.05, 0.0),
                             q_pre=q.copy(), hover_z=float(tcp(rt)[2]) + 0.05,
                             z_max=float(tcp(rt)[2]) + 0.055, duration_s=2.0, scene=0)
    rt._stop_retraces = {id(rt.arm): kept}
    rt.skill_reset_scene()
    assert stop_retrace.kept(rt) is None


# ── the retrace's own guards ────────────────────────────────────────────────


def test_an_obstacle_above_the_hover_inside_the_column_refuses_the_retrace(rebot, monkeypatch):
    """The kept column ends 5 mm above the hover: a box above that, inside the
    column, is checked at full strength, and the retrace is refused before
    its first command (the unbounded approach column would exempt it)."""
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    kept = stop_retrace.kept(rt)
    hover = tcp(rt, kept.q_pre)
    lid = centred_box([hover[0], hover[1], hover[2] + 0.06], [0.03, 0.03, 0.02])
    occ = box_map("solid", lid)
    rt.arm.harness.occupancy = occ
    n0 = commands(rt)
    res = rt.execute("move_home", {})
    assert res["ok"] is False and "stop retrace refused" in res["error"], res
    assert commands(rt) == n0
    assert stop_retrace.kept(rt) is kept                           # nothing moved: still kept


def test_a_retrace_that_would_leave_the_column_is_refused_before_any_command(rebot, monkeypatch):
    """A goal whose segment drifts sideways out of the kept radius (the
    hover moved 2 radii in x): refused by the column guard, nothing sent."""
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    kept = stop_retrace.kept(rt)
    kin = rt.arm.harness.kin
    T = kin.fk(kept.q_pre).copy()
    T[0, 3] += 2 * kept.column[1]
    aside = kin.ik(T, q_init=kept.q_pre.copy(), max_iter=200, retries=0)
    assert aside.success
    rt._stop_retraces[id(rt.arm)] = dataclasses.replace(kept, q_pre=np.asarray(aside.q, dtype=float))
    n0 = commands(rt)
    res = rt.execute("move_home", {})
    assert res["ok"] is False and "stop retrace refused" in res["error"], res
    assert "leave the kept approach column" in res["error"], res
    assert commands(rt) == n0


def test_a_retrace_that_would_not_rise_is_refused_before_any_command(rebot, monkeypatch):
    """A goal BELOW the measured pose (a 1 cm lower 'hover'): refused by the
    strictly-upward guard, nothing sent."""
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    kept = stop_retrace.kept(rt)
    kin = rt.arm.harness.kin
    q = np.asarray(rt.arm.get_state().q, dtype=float)
    T = kin.fk(q).copy()
    T[2, 3] -= 0.01
    lower = kin.ik(T, q_init=q.copy(), max_iter=200, retries=0)
    assert lower.success
    rt._stop_retraces[id(rt.arm)] = dataclasses.replace(kept, q_pre=np.asarray(lower.q, dtype=float))
    n0 = commands(rt)
    res = rt.execute("move_home", {})
    assert res["ok"] is False and "strictly upward" in res["error"], res
    assert commands(rt) == n0


def test_the_arm_not_below_the_hover_has_nothing_to_retrace(rebot):
    """A kept column with the arm at (or above) its hover: dropped, and the
    home move is today's, command for command."""
    rt = rebot
    from cascade.skills import stop_retrace

    q = np.asarray(rt.arm.get_state().q, float)
    p = tcp(rt)
    rt._stop_retraces = {id(rt.arm): stop_retrace.Kept(
        arm=rt.arm, raw=rt.arm.raw, column=(p[:2].copy(), 0.05, 0.0), q_pre=q.copy(),
        hover_z=float(p[2]), z_max=float(p[2]) + CEILING_M, duration_s=2.0, scene=0)}
    res = rt.execute("move_home", {})
    assert res["ok"] is True and "stop_retrace" not in res, res
    assert stop_retrace.kept(rt) is None


# ── what is kept, and when (the helper, stubbed state) ──────────────────────


class LineKin:
    """fk(q): the TCP at q[:3], identity rotation."""

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q, dtype=float)[:3]
        return T


COLUMN = (np.array([0.30, 0.0]), 0.04, 0.02)
Q_PRE = np.array([0.30, 0.0, 0.10])


def stub(*, estopped=False, generation=0, kin=LineKin(), episode=None):
    harness = SimpleNamespace(estopped=estopped, _halt_generation=generation, kin=kin,
                              occupancy=SimpleNamespace(scene_reset_generation=3))
    return SimpleNamespace(arm=SimpleNamespace(harness=harness, raw=object()), _contact_episode=episode)


@pytest.mark.parametrize("state, failure, column, kept", [
    ({"generation": 1}, MotionHalted("halt"), COLUMN, True),                 # a halt since the grasp began
    ({"estopped": True}, SafetyViolation("e-stop latched"), COLUMN, True),   # a latched e-stop
    ({"generation": 1}, SkillError("refused"), COLUMN, True),                # a halt during the retreat
    ({}, SkillError("lift refused"), COLUMN, False),                         # no stop: nothing kept
    ({"estopped": True}, ContactDetected("contact"), COLUMN, False),         # a contact is no operator stop
    ({"generation": 1}, None, COLUMN, False),                                # nothing raised
    ({"generation": 1}, MotionHalted("halt"), None, False),                  # no approach column
    ({"generation": 1, "episode": {"cylinder": 1}}, MotionHalted("h"), COLUMN, False),  # contact episode owns it
    ({"generation": 1, "kin": None}, MotionHalted("halt"), COLUMN, False),   # no kinematic model
], ids=["halt", "estop", "halt-in-retreat", "no-stop", "contact", "no-failure", "no-column",
        "contact-episode", "no-kin"])
def test_the_column_is_kept_only_after_an_operator_stop(state, failure, column, kept):
    from cascade.skills import stop_retrace

    rt = stub(**state)
    out = stop_retrace.keep(rt, column, Q_PRE, failure=failure, halt_generation=0, duration_s=2.5,
                             ceiling_m=CEILING_M)
    assert (out is not None) == kept
    assert (stop_retrace.kept(rt) is not None) == kept
    if kept:
        assert out.arm is rt.arm and out.raw is rt.arm.raw and out.scene == 3 and out.duration_s == 2.5
        assert out.hover_z == pytest.approx(0.10) and out.z_max == pytest.approx(0.10 + CEILING_M)
        np.testing.assert_array_equal(out.q_pre, Q_PRE)
        assert out.q_pre is not Q_PRE                                  # a copy


def test_forget_drops_every_kept_column():
    from cascade.skills import stop_retrace

    rt = stub(generation=1)
    stop_retrace.keep(rt, COLUMN, Q_PRE, failure=MotionHalted("h"), halt_generation=0, duration_s=2.0,
                      ceiling_m=CEILING_M)
    assert stop_retrace.forget(rt) == 1 and stop_retrace.kept(rt) is None
    assert stop_retrace.forget(SimpleNamespace()) == 0                     # a runtime that never kept one


# ── every step strictly upward inside the kept column (the guard) ───────────


@pytest.mark.parametrize("start, goal, ok", [
    ([0.30, 0.0, 0.05], [0.30, 0.0, 0.10], True),       # straight up to the hover
    ([0.30, 0.0, 0.05], [0.32, 0.01, 0.10], True),      # rising, inside the radius
    ([0.30, 0.0, 0.05], [0.30, 0.0, 0.04], False),      # down
    ([0.30, 0.0, 0.05], [0.33, 0.0, 0.05], False),      # sideways at the same height: not strictly up
    ([0.30, 0.0, 0.05], [0.35, 0.0, 0.10], False),      # rising, but out of the column's radius
    ([0.30, 0.0, 0.05], [0.30, 0.0, 0.11], False),      # above the ceiling (hover + 5 mm)
    ([0.30, 0.0, 0.01], [0.30, 0.0, 0.10], False),      # starts below the column floor
], ids=["up", "up-inside", "down", "sideways", "out-of-radius", "above-ceiling", "below-floor"])
def test_every_step_must_rise_strictly_inside_the_column(start, goal, ok):
    from cascade.skills import stop_retrace

    reason = stop_retrace.upward_in_column(LineKin(), np.array(start), np.array(goal), 1.0, 50.0,
                                           COLUMN, 0.10 + CEILING_M)
    assert (reason is None) == ok, reason


def test_an_edge_that_only_holds_its_height_is_refused():
    """The strict rule is per edge: a path that rises, then holds its height
    for one edge (a step of the stream), is refused."""
    from cascade.skills import stop_retrace

    class Plateau(LineKin):
        def fk(self, q):
            T = super().fk(q)
            T[2, 3] = min(T[2, 3], 0.07) if T[2, 3] < 0.075 else T[2, 3]
            return T

    reason = stop_retrace.upward_in_column(Plateau(), np.array([0.30, 0.0, 0.05]), np.array([0.30, 0.0, 0.10]),
                                           1.0, 50.0, COLUMN, 0.10 + CEILING_M)
    assert reason is not None and "strictly upward" in reason, reason


# ── preconditions at the explicit call ─────────────────────────────────────


def explicit(rt, root="move_home"):
    """Run as execute()'s root motion skill `root` would."""
    import threading

    if rt._motion_scope is None:
        rt._motion_scope = threading.local()
    rt._motion_scope.context = {"root_skill": root}


@pytest.mark.parametrize("change", ["arm", "raw", "planner", "carry", "contact_episode", "tcp_outside"])
def test_a_kept_column_that_no_longer_applies_is_dropped_without_a_command(rebot, monkeypatch, change):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import carry_attachment, stop_retrace

    kept = stop_retrace.kept(rt)
    if change == "arm":
        rt._stop_retraces[id(rt.arm)] = dataclasses.replace(kept, arm=object())
    elif change == "raw":
        rt._stop_retraces[id(rt.arm)] = dataclasses.replace(kept, raw=object())
    elif change == "planner":
        monkeypatch.setattr(rt.arm, "motion_planner", object())
    elif change == "carry":
        monkeypatch.setattr(carry_attachment, "active", lambda runtime: object())
    elif change == "contact_episode":
        monkeypatch.setattr(rt, "_contact_episode", {"cylinder": None}, raising=False)
    else:
        center, radius, z_min = kept.column
        rt._stop_retraces[id(rt.arm)] = dataclasses.replace(kept, column=(np.asarray(center) + [3 * radius, 0.0], radius, z_min))
    n0 = commands(rt)
    explicit(rt)
    try:
        assert stop_retrace.retrace(rt) is None
    finally:
        rt._motion_scope.context = None
    assert commands(rt) == n0 and stop_retrace.kept(rt) is None


@pytest.mark.parametrize("root", ["pick_and_place", "grasp_object", None])
def test_only_an_explicit_move_home_or_reset_scene_retraces(rebot, monkeypatch, root):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    kept = stop_retrace.kept(rt)
    n0 = commands(rt)
    if root is not None:
        explicit(rt, root)
    try:
        assert stop_retrace.retrace(rt) is None
    finally:
        if rt._motion_scope is not None:
            rt._motion_scope.context = None
    assert commands(rt) == n0 and stop_retrace.kept(rt) is kept


@pytest.mark.parametrize("root", ["move_home", "reset_scene"])
def test_the_explicit_retrace_streams_to_the_pregrasp_inside_the_capped_column(rebot, monkeypatch, root):
    rt = rebot
    stopped_at_the_object(rt, monkeypatch)
    from cascade.skills import stop_retrace

    kept = stop_retrace.kept(rt)
    harness = rt.arm.harness
    seen = []
    real = harness.approve

    def approve(q_prev, q_next, dt, **kw):
        seen.append(harness._grasp_exempt)
        return real(q_prev, q_next, dt, **kw)

    monkeypatch.setattr(harness, "approve", approve)
    explicit(rt, root)
    try:
        out = stop_retrace.retrace(rt)
    finally:
        rt._motion_scope.context = None
    assert out.startswith("completed"), out
    assert seen and all(e is not None and len(e) == 4 for e in seen)
    center, radius, z_min, z_max = seen[0]
    np.testing.assert_allclose(center, kept.column[0])
    assert (radius, z_min, z_max) == (kept.column[1], kept.column[2], kept.z_max)
    assert harness._grasp_exempt is None                              # released
    np.testing.assert_allclose(rt.arm.get_state().q, kept.q_pre, atol=1e-6)
    assert stop_retrace.kept(rt) is None
