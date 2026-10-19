"""point_at planning follow-ups (#399 post-merge review, Minors 1-4).

1. The core-speed reference (`pointing._fastest_probe_s`, `core_speed`) is
   searched over the PROCESS's CPU set (not the calling thread's affinity),
   one core per `cpu_capacity` class, fastest class first, each helper given
   time to get its core; a search that ran out of time is never kept. main
   searched the caller's own affinity in CPU-number order within 50 ms per
   core: a first call from a thread pinned to a slow core, or helpers that
   wait 60 ms for their core, made a slow core the reference for the life of
   the process.
2. A halt received while planning, followed by a geometric or budget
   refusal, is reported as the halt (an e-stop still wins over it). main
   named the geometry or the clock (the review's stop rows `delay9_timer5
   halt` and `eord * halt`).
3. Planning that took longer than `pointing.FRESH_PLAN_S` of wall time looks
   at the target again before the first command: moved more than
   `pointing.FRESH_MOVE_M` -> planned once more from where it is now, in what
   is left of the same budget; moved again, or not found, -> refused with no
   command. Never a move toward a stale pose.
4. `pointing.tool_body` refuses a tool part known only by its points (no box
   in `ToolVolume.boxes`): its sampled distance could overstate the part's by
   up to spacing / sqrt(2) (14 mm on the harness's 2 cm grid).
5. On a pair (#400 fix re-review, Minor 1), a hold on the OTHER arm is not
   this arm's: the empty arm points exactly as if nothing were held (main
   refused it "while holding"), the object in the other arm's jaws stays a
   neighbour, and the holding arm keeps its hold rules.
"""

from __future__ import annotations

import math
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from base_pose_oracle import make_runtime  # noqa: F401  (the fixture, by name)
from conftest import needs_pin

from cascade.control.mock_arm import MockArm
from cascade.skills.pointing import core_factor as _CORE_FACTOR

needs_affinity = pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or len(os.sched_getaffinity(0)) < 2,
    reason="Linux: per-thread CPU affinity, two cores")


class _RecArm(MockArm):
    def __init__(self, cfg, kinematics, log):
        super().__init__(cfg, kinematics)
        self.log = log

    def set_gripper(self, pos, effort=1.0):
        self.log.jaws.append(float(pos))
        super().set_gripper(pos, effort)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    def build(profile="mock", camera="mock", hold_s=0.0):
        cfg = load_demo_config(camera=camera, arm=profile, llm="mock")
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        cfg._data.setdefault("gesture", {})["point_hold_s"] = float(hold_s)
        r = SimpleNamespace(cfg=cfg, jaws=[], arms=[])

        def make_arm(acfg, kinematics=None):
            arm = _RecArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / f"{profile}-{camera}-{len(built)}", lazy_arm=True)
        built.append((r.runtime, arm))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and r.runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        r.runtime.arm.get_state()
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _commands(r) -> list:
    return [np.asarray(q, dtype=float) for arm in r.arms for q in arm.commands]


# ── 1. the core-speed reference ────────────────────────────────────────────


def _one_fast_core(monkeypatch, fast):
    """`fast` alone in the fastest `cpu_capacity` class; every other core is
    slower (whatever this host's sysfs says)."""
    from cascade import core_speed

    monkeypatch.setattr(core_speed, "_capacities", lambda cpus: {c: (1024 if c == fast else 512) for c in cpus},
                        raising=False)


def _join_probe_helpers(timeout_s=5.0):
    end = time.monotonic() + timeout_s
    for t in [t for t in threading.enumerate() if t.name == "core-speed-probe"]:
        t.join(max(0.0, end - time.monotonic()))


@needs_affinity
def test_a_helper_that_waits_60_ms_for_its_core_still_counts(monkeypatch):
    """The review's probe: a helper waits 60 ms before its core runs it (a
    loaded host). main gave each helper 50 ms and dropped every late answer:
    the caller's own (slow) core became the reference."""
    from cascade import core_speed

    cpus = sorted(os.sched_getaffinity(0))
    fast = cpus[-1]
    _one_fast_core(monkeypatch, fast)
    main = threading.get_ident()
    waited = set()

    def work():
        # the first time a helper runs on a core (keyed by the core: thread
        # idents are reused) it waits 60 ms for it, as the review's probe did
        cores = frozenset(os.sched_getaffinity(0))
        if threading.get_ident() != main and cores not in waited:
            waited.add(cores)
            time.sleep(0.06)
        return 1e-3 if os.sched_getaffinity(0) == {fast} else 3e-3

    try:
        assert core_speed.fastest_seconds(work) == 1e-3
    finally:
        _join_probe_helpers()


@needs_affinity
def test_a_first_call_from_a_thread_pinned_to_a_slow_core_still_finds_the_fast_core(monkeypatch):
    """main searched the CALLER's affinity: from a thread pinned to one slow
    core it timed that core alone and kept it as the reference for the life
    of the process (the planning budget never stretched again). Now the
    process's CPU set is searched; the caller stays where it was pinned."""
    from cascade.skills import pointing

    cpus = sorted(os.sched_getaffinity(0))
    slow, fast = cpus[0], cpus[-1]
    _one_fast_core(monkeypatch, fast)
    monkeypatch.setattr(pointing, "core_factor", _CORE_FACTOR)
    monkeypatch.setattr(pointing, "_speed_probe_s", lambda: 1e-3 if os.sched_getaffinity(0) == {fast} else 3e-3)
    monkeypatch.setattr(pointing, "_FASTEST", {})
    out = {}

    def run():
        os.sched_setaffinity(0, {slow})
        out["ref"] = pointing._fastest_probe_s()
        out["factor"] = pointing.core_factor()
        out["aff"] = os.sched_getaffinity(0)

    th = threading.Thread(target=run)
    th.start()
    th.join(30.0)
    _join_probe_helpers()

    assert out["ref"] == 1e-3, out
    assert out["factor"] == pytest.approx(3.0), out          # the budget stretches on the slow core
    assert out["aff"] == {slow}, "the caller's affinity must stay as it was"
    assert pointing._FASTEST["s"] == 1e-3                    # a complete search is kept


@needs_affinity
def test_one_core_per_capacity_class_fastest_class_first(monkeypatch):
    from cascade import core_speed

    cpus = sorted(os.sched_getaffinity(0))
    fast = cpus[-1]
    _one_fast_core(monkeypatch, fast)
    main = threading.get_ident()
    probed = []

    def work():
        cores = sorted(os.sched_getaffinity(0))
        if threading.get_ident() != main and cores not in probed:
            probed.append(cores)
        return 1e-3

    core_speed.fastest_seconds(work)
    _join_probe_helpers()

    assert probed[0] == [fast], probed
    assert len(probed) == 2 and len(probed[1]) == 1 and probed[1][0] != fast, probed


@needs_affinity
@pytest.mark.parametrize("stall", ["on_its_core", "before_its_core"])
def test_a_search_that_ran_out_of_time_is_not_kept(monkeypatch, stall):
    """No helper got its core in time (stalled before it could even pin
    itself, or on its core): this call uses what it measured (its own core,
    which can only understate the stretch), and the next call searches again
    instead of keeping a slow reference for good."""
    from cascade import core_speed
    from cascade.skills import pointing

    cpus = sorted(os.sched_getaffinity(0))
    fast = cpus[-1]
    _one_fast_core(monkeypatch, fast)
    gate = threading.Event()
    main = threading.get_ident()
    real_pin = os.sched_setaffinity

    def pin(pid, cores):
        if stall == "before_its_core" and threading.current_thread().name == "core-speed-probe":
            gate.wait(10.0)
        return real_pin(pid, cores)

    def probe():
        if stall == "on_its_core" and threading.get_ident() != main:
            gate.wait(10.0)
        return 1e-3 if os.sched_getaffinity(0) == {fast} else 3e-3

    monkeypatch.setattr(os, "sched_setaffinity", pin)
    monkeypatch.setattr(pointing, "_speed_probe_s", probe)
    monkeypatch.setattr(pointing, "_FASTEST", {})
    monkeypatch.setattr(core_speed, "PER_CORE_S", 0.05)
    try:
        assert pointing._fastest_probe_s() == 3e-3
        assert "s" not in pointing._FASTEST, "a search that ran out of time was kept"
    finally:
        gate.set()
        _join_probe_helpers()
    assert pointing._fastest_probe_s() == 1e-3
    assert pointing._FASTEST["s"] == 1e-3


@needs_affinity
def test_a_core_that_cannot_be_pinned_gives_way_to_the_next_of_its_class(monkeypatch):
    """A class's next core is tried only when this one refuses the pin; a
    class none of whose cores the process may use is no part of its set
    (the search still completes)."""
    from cascade import core_speed

    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 3:
        pytest.skip("three cores")
    barred, fast = cpus[-2], cpus[-1]              # barred comes first in their class
    caps = {c: 512 for c in cpus} | {barred: 1024, fast: 1024}
    monkeypatch.setattr(core_speed, "_capacities", lambda cores: {c: caps[c] for c in cores})
    real_pin = os.sched_setaffinity

    def pin(pid, cores):
        if threading.current_thread().name == "core-speed-probe" and set(cores) == {barred}:
            raise OSError(22, "Invalid argument")
        return real_pin(pid, cores)

    monkeypatch.setattr(os, "sched_setaffinity", pin)
    def work():
        return 1e-3 if os.sched_getaffinity(0) == {fast} else 3e-3

    ref = core_speed.fastest(work)
    assert ref == core_speed.Reference(1e-3, True), ref

    caps.update({barred: 2048})                    # barred alone in the fastest class
    ref = core_speed.fastest(work)
    _join_probe_helpers()
    assert ref == core_speed.Reference(1e-3, True), ref


@needs_affinity
def test_a_class_the_search_had_no_time_for_leaves_it_incomplete(monkeypatch):
    """The fastest class answered, then the search window closed before the
    next class was tried: not complete (kept for this call only)."""
    from cascade import core_speed

    cpus = sorted(os.sched_getaffinity(0))
    fast = cpus[-1]
    _one_fast_core(monkeypatch, fast)
    ticks = iter([0.0, 0.5] + [5.0] * 10)          # SEARCH_S = 1: the second class finds the window closed
    monkeypatch.setattr(core_speed, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    ref = core_speed.fastest(lambda: 1e-3 if os.sched_getaffinity(0) == {fast} else 3e-3)
    _join_probe_helpers()

    assert ref == core_speed.Reference(1e-3, False), ref


def test_cpu_capacities_come_from_sysfs_or_not_at_all(tmp_path):
    from cascade import core_speed

    for cpu, text in ((0, "718\n"), (1, "1024\n"), (2, "bad\n")):
        (tmp_path / f"cpu{cpu}").mkdir()
        (tmp_path / f"cpu{cpu}" / "cpu_capacity").write_text(text)
    assert core_speed._capacities({0, 1}, root=tmp_path) == {0: 718, 1: 1024}
    assert core_speed._capacities({0, 2}, root=tmp_path) is None      # unreadable: no classes at all
    assert core_speed._capacities({0, 3}, root=tmp_path) is None      # missing
    assert core_speed._capacities(set(), root=tmp_path) is None


def test_the_search_order_is_one_class_per_capacity_fastest_first(monkeypatch):
    from cascade import core_speed

    caps = {0: 718, 1: 718, 2: 1024, 3: 997, 4: 997, 5: 731}
    monkeypatch.setattr(core_speed, "_capacities", lambda cpus: {c: caps[c] for c in cpus})
    assert core_speed._search_order({0, 1, 2, 3, 4, 5}) == [[2], [3, 4], [5], [0, 1]]
    monkeypatch.setattr(core_speed, "_capacities", lambda cpus: None)
    assert core_speed._search_order({3, 0, 2}) == [[0], [2], [3]]      # no classes: every core


# ── 2. a halt while planning is reported as the halt ───────────────────────


def _fire(harness, kind):
    if kind in ("halt", "halt_then_estop"):
        harness.halt("operator: stop")
    if kind in ("estop", "halt_then_estop"):
        harness.estop("operator e-stop")


@needs_pin
@pytest.mark.parametrize("kind", ["halt", "estop", "halt_then_estop"])
@pytest.mark.parametrize("where", ["topdown_held", "lean", "route", "wall_cap"])
def test_a_stop_while_planning_then_a_refusal_reports_the_stop(rig, monkeypatch, where, kind):
    """The review's rows `eord * halt` (a halt in the top-down search, the
    lean-in or the route, then their geometric refusal) and `delay9_timer5
    halt` (a halt, then the 8 s wall cap): main named the geometry or the
    clock. The halt is what the agent must hear; an e-stop still wins over
    it; nothing is commanded."""
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    harness = rt.arm.harness
    fired = []

    def once():
        if not fired:
            fired.append(True)
            _fire(harness, kind)

    if where == "topdown_held":
        rt.beliefs.update("cup", np.array([0.5, 0.4, 0.04]), 0.95, extent=np.array([0.08, 0.06, 0.06]),
                          top_z=0.08, color="yellow")
        rt.held_object = "yellow cup"

        def ik(T, q_init=None, *a, **k):
            once()
            return SimpleNamespace(q=np.asarray(q_init, float), success=False, error=1.0)

        monkeypatch.setattr(rt.kin, "ik", ik)
    elif where == "lean":
        monkeypatch.setattr(pointing, "neighbour_clearance", lambda kin, q, tool, neighbours=(): (0.0, "a stub"))
        real_lean = pointing.lean_in

        def lean(*a, **k):
            found = real_lean(*a, **k)
            once()
            return pointing.LeanIn(q=None, clearance=found.clearance, solved=found.solved)

        monkeypatch.setattr(pointing, "lean_in", lean)
    elif where == "route":
        def route(*a, **k):
            once()
            raise pointing.RouteRefused("a stubbed geometric refusal")

        monkeypatch.setattr(pointing, "plan_route", route)
    else:
        now = [1000.0]
        monkeypatch.setattr(pointing, "_wall", lambda: now[0])
        monkeypatch.setattr(pointing, "PLAN_WALL_S", 8.0)
        real_check = pointing.Budget.check

        def check(self):
            if not fired:
                once()
                now[0] += 9.0              # the host stalls planning past the wall cap
            return real_check(self)

        monkeypatch.setattr(pointing.Budget, "check", check)

    out = rt.execute("point_at", {"label": "red cube"})

    assert fired, f"{where} never ran"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    if kind == "halt":
        assert out["error"].startswith("MotionHalted: "), out
        assert "operator: stop" in out["error"], out
    else:
        assert out["error"] == "SafetyViolation: e-stop latched", out
    for geometry in ("leaning in", "while holding", "stubbed geometric", "wall-clock cap", "no pose"):
        assert geometry not in out["error"], out


@needs_pin
def test_a_halt_while_planning_is_reported_outside_execute_too(rig, monkeypatch):
    """skill_point_at called directly (no execute(), so no skill halt scope):
    its own start generation still names the halt."""
    from cascade.skills import pointing
    from cascade.types import MotionHalted

    r = rig("mock")
    rt = r.runtime

    def route(*a, **k):
        rt.arm.harness.halt("operator: stop")
        raise pointing.RouteRefused("a stubbed geometric refusal")

    monkeypatch.setattr(pointing, "plan_route", route)
    with pytest.raises(MotionHalted, match="halted: operator: stop"):
        rt.skill_point_at("red cube")
    assert _commands(r) == [] and r.jaws == []


@needs_pin
def test_a_halt_in_the_hold_is_not_remembered_as_a_pointing(rig, monkeypatch):
    """The fifth place: the halt arrives after the last leg, in the hold. The
    result already said halted (execute); the gesture is no longer noted as
    done either, and nothing follows the halt."""
    r = rig("mock")
    rt = r.runtime
    real = rt._move_joints
    calls = []

    def move(q, *a, **k):
        ok = real(q, *a, **k)
        calls.append(len(_commands(r)))
        rt.arm.harness.halt("operator: stop")
        return ok

    monkeypatch.setattr(rt, "_move_joints", move)
    out = rt.execute("point_at", {"label": "red cube"})

    assert len(calls) == 1, "test setup: a direct route (one leg, then the hold)"
    assert out["ok"] is False and "MotionHalted: halted: operator: stop" in out["error"], out
    assert len(_commands(r)) == calls[0]
    assert not [e for e in rt.memory.events(("action",)) if "pointed at" in e.text]


# ── 3. a long plan looks at the target again before the first command ──────


def _shifted(fix, d):
    """The fix moved by d[:3]; a fourth entry raises its points (its observed
    top) that much more, its centre staying where it is (a taller object)."""
    from dataclasses import replace

    d = np.asarray(d, float)
    points = np.asarray(fix.points, float) + d[:3]
    if len(d) > 3:
        points = points + [0.0, 0.0, d[3]]
    return replace(fix, position=np.asarray(fix.position, float) + d[:3], points=points)


def _long_plans(monkeypatch, seconds=4.0):
    """Each route planning takes `seconds` of (fake) wall time."""
    from cascade.skills import pointing

    now = [1000.0]
    monkeypatch.setattr(pointing, "_wall", lambda: now[0])
    monkeypatch.setattr(pointing, "FRESH_PLAN_S", 3.0, raising=False)
    real_route = pointing.plan_route

    def route(*a, **k):
        now[0] += seconds
        return real_route(*a, **k)

    monkeypatch.setattr(pointing, "plan_route", route)
    return now


def _localizations(monkeypatch, rt, shifts):
    """`_localize` answers the real fix moved by shifts[i] on its i-th call
    (the last shift repeats); a shift that is an exception is raised."""
    real = rt._localize
    calls = []

    def localize(label, *a, **k):
        frame, fix = real(label, *a, **k)
        d = shifts[min(len(calls), len(shifts) - 1)]
        calls.append(fix)
        if isinstance(d, Exception):
            raise d
        return frame, _shifted(fix, d)

    monkeypatch.setattr(rt, "_localize", localize)
    return calls


def _tcp_xy(r, q):
    return np.asarray(r.runtime.kin.fk(q), float)[:2, 3]


@needs_pin
def test_a_long_plan_replans_toward_where_the_target_is_now(rig, monkeypatch):
    """Planning took 4 s; the target is 3 cm away from where it was planned
    for. main moved to the stale pose. Now it is planned once more, from the
    fresh fix, and the arm points there."""
    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.03, 0.0)])

    out = rt.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    stale = np.asarray(calls[0].position, float)[:2]
    fresh = stale + [0.0, 0.03]
    final = _tcp_xy(r, _commands(r)[-1])
    assert np.linalg.norm(final - fresh) < 2e-3, (final, fresh)
    assert np.linalg.norm(final - stale) > 0.02
    # the first plan, a look, the plan from the fresh fix (4 s too), a look that confirms it
    assert len(calls) == 3
    assert out["position"] == pytest.approx([round(float(x), 3) for x in rt._to_world(calls[0].position + [0, .03, 0])])


@needs_pin
def test_a_long_plan_with_the_target_still_there_moves_as_planned(rig, monkeypatch):
    """5 mm (perception noise, within FRESH_MOVE_M): the plan stands."""
    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.005, 0.0)])

    out = rt.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert len(calls) == 2, "planning took 4 s: the target must be looked at again"
    stale = np.asarray(calls[0].position, float)[:2]
    assert np.linalg.norm(_tcp_xy(r, _commands(r)[-1]) - stale) < 2e-3


@needs_pin
@pytest.mark.parametrize("again", [(0.0, 0.06, 0.0), (0.0, 0.03, 0.0, 0.015)], ids=["sideways", "its_top_rose"])
def test_a_target_that_moved_again_while_replanning_is_refused(rig, monkeypatch, again):
    """Planned once more and it moved again (sideways, or its top rose 1.5
    cm, its centre staying): refused, nothing commanded (one re-plan only;
    never toward a stale pose)."""
    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.03, 0.0), again])

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(calls) == 3
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "moved" in out["error"] and "nothing was moved" in out["error"], out


@needs_pin
def test_a_fresh_fix_that_is_not_finite_counts_as_moved(rig, monkeypatch):
    """NaN in the fresh fix is never read as 'not moved' (the plan from it
    then refuses the non-finite top); nothing moves."""
    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, math.nan)])

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(calls) == 2
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "not a finite height" in out["error"], out


@needs_pin
def test_a_target_not_found_again_after_a_long_plan_is_refused(rig, monkeypatch):
    from cascade.types import SkillError

    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), SkillError("No current visual match for 'red cube'")])

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(calls) == 2
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "planning took 4.0 s, and looking at it again before the first command failed (SkillError: No " \
           "current visual match for 'red cube'); nothing was moved" in out["error"], out


@needs_pin
@pytest.mark.parametrize("then", ["not_found", "moved_again"])
def test_a_halt_while_looking_again_is_reported_as_the_halt(rig, monkeypatch, then):
    """The operator halts while the target is localized again; that look then
    fails or finds it moved again. The halt is what is reported."""
    from cascade.types import SkillError

    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    last = SkillError("No current visual match for 'red cube'") if then == "not_found" else (0.0, 0.06, 0.0)
    shifts = [(0.0, 0.0, 0.0), last] if then == "not_found" else [(0.0, 0.0, 0.0), (0.0, 0.03, 0.0), last]
    calls = _localizations(monkeypatch, rt, shifts)
    real = rt._localize

    def localize(label, *a, **k):
        if len(calls) == len(shifts) - 1:
            rt.arm.harness.halt("operator: stop")
        return real(label, *a, **k)

    monkeypatch.setattr(rt, "_localize", localize)
    out = rt.execute("point_at", {"label": "red cube"})

    assert len(calls) == len(shifts)
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert out["error"].startswith("MotionHalted: halted: operator: stop"), out


@needs_pin
def test_the_replan_runs_on_what_is_left_of_the_same_budget(rig, monkeypatch):
    """5 s per plan against the 8 s wall cap: the second plan runs past it
    and refuses (a fresh budget would have let it finish)."""
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    monkeypatch.setattr(pointing, "PLAN_WALL_S", 8.0)
    _long_plans(monkeypatch, seconds=5.0)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.03, 0.0)])

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(calls) == 2
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "8.0 s wall-clock cap" in out["error"], out


@needs_pin
def test_a_short_plan_does_not_look_again(rig, monkeypatch):
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    monkeypatch.setattr(pointing, "_wall", lambda: 1000.0)       # planning takes no wall time
    monkeypatch.setattr(pointing, "FRESH_PLAN_S", 3.0, raising=False)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.03, 0.0)])

    out = rt.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and len(calls) == 1, out


def test_the_shipped_freshness_rule_is_pinned():
    import ast
    import inspect

    from cascade.skills import pointing

    shipped = {node.targets[0].id: ast.unparse(node.value) for node in ast.parse(inspect.getsource(pointing)).body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert shipped["FRESH_PLAN_S"] == "3.0"
    assert shipped["FRESH_MOVE_M"] == "AIM_TOLERANCE_M"
    assert pointing.AIM_TOLERANCE_M == 0.01


# ── 4. a tool part needs its box ───────────────────────────────────────────


def test_a_tool_part_known_only_by_its_points_is_refused():
    from cascade.safety.tool_volume import ToolVolume
    from cascade.skills import pointing

    pts = np.array([[0.0, 0.0, 0.05], [0.0, 0.01, 0.05], [0.1, 0.0, 0.0]])
    with pytest.raises(ValueError, match=r"tool part 'probe' has no box"):
        pointing.tool_body(ToolVolume(local=pts, part=np.array([0, 0, 1]), names=("probe", "jaws")), 0.03)
    box = (np.zeros(3), np.eye(3), np.full(3, 0.01))
    with pytest.raises(ValueError, match=r"tool part 'probe' has no box"):
        pointing.tool_body(ToolVolume(local=pts, part=np.array([0, 0, 1]), names=("probe", "jaws"),
                                      boxes=(box,)), 0.03)
    # the jaws need none: every jaw sample group already holds the sweep box
    assert pointing.tool_body(ToolVolume(local=pts[2:], part=np.array([0]), names=("jaws",)), 0.03) == ()


@needs_pin
def test_point_at_refuses_a_harness_tool_part_without_its_box(rig):
    from cascade.safety.tool_volume import ToolVolume

    r = rig("mock")
    pts = np.array([[0.0, 0.0, 0.05], [0.0, 0.01, 0.05]])
    r.runtime.arm.harness.tool_volume = ToolVolume(local=pts, part=np.array([0, 0]), names=("probe",))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "tool part 'probe' has no box" in out["error"] and "nothing was moved" in out["error"], out


def test_the_missing_box_slack_would_have_been_real():
    """Why a point-only part is refused rather than kept with slack 0: a 2 cm
    grid's nearest sample can be spacing / sqrt(2) from the box point nearest
    an object (a face centre between four samples)."""
    s = 0.02
    face = np.array([[x, y, 0.0] for x in np.arange(-0.04, 0.0401, s) for y in np.arange(-0.04, 0.0401, s)])
    point_between = np.array([s / 2, s / 2, 0.0])
    assert np.min(np.linalg.norm(face - point_between, axis=1)) == pytest.approx(s / math.sqrt(2))


# ── 5. a pair: the empty arm points as if nothing were held ────────────────

_PAIR = ("so101_left", "so101_right")
_OTHER = {"so101_left": "so101_right", "so101_right": "so101_left"}
#: puts the cube in reach of that arm (the pair's table-frame camera shift)
_SHIFT = {"so101_left": (0.0, 0.10), "so101_right": (0.0, -0.10)}


def _pair_point_at(make_runtime, monkeypatch, *, side, caller, holder, box, cup=False,  # noqa: F811
                   camera_part=False):
    """point_at by `caller` on the shipped so101 pair (the cube on `side`'s
    side); `holder` (None: nothing held) holds the red cube (what a completed
    grasp leaves, injected), or the yellow cup when `cup`. `camera_part`:
    the caller's harness models a small wrist-camera box (a reBot-like tool
    volume).
    Returns (result fields, every arm's commands, every arm's jaws, the hold)."""
    from contextlib import ExitStack

    from base_pose_oracle import _arm_entries, _data, moved_cfg, raw_arm

    from cascade.safety.tool_volume import ToolVolume, box_grid

    cfg = moved_cfg(list(_PAIR), {}, move_cameras=False, shift_xy=_SHIFT[side])
    # The planning thread can starve the watcher's heartbeat past the 5 s
    # perception watchdog on a loaded CI runner (#406's arm job: both calls
    # refused "last observation 5.6s old" / "5.8s old"); this test is about who
    # holds what, so the watchdog is lifted like the planning budget is: the
    # top level and every arm entry's resolved `safety` (each arm's harness)
    sections = [cfg._data.get("safety")] + [_data(e["resolved"]).get("safety") for e in _arm_entries(cfg._data)
                                            if e.get("resolved") is not None]
    for safety in filter(None, sections):
        _data(safety)["watchdog_s"] = 1e9
    if box == "nobox":
        cfg._data["relative_motion"]["wrist_camera"] = None
    rt = make_runtime(cfg)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and rt.beliefs.find("red cube") is None:
        time.sleep(0.05)
    cube = rt.beliefs.find("red cube")
    assert cube is not None, "test setup: the cube must be seen"
    with ExitStack() as stack:
        if rt.watcher is not None:
            stack.enter_context(rt.watcher.paused())
        if cup:
            # a tall cup 6 cm beside the cube: a neighbour that crowds a top-down pose
            c = np.asarray(cube.position, float) + [0.0, 0.06, 0.0]
            rt.beliefs.update("cup", np.array([c[0], c[1], 0.075]), 0.95, extent=np.array([0.15, 0.04, 0.04]),
                              top_z=0.15, color="yellow")
        raws = {n: raw_arm(rt.arm_rig.get(n)) for n in _PAIR}
        jaws = {n: [] for n in _PAIR}
        for n, raw in raws.items():
            real = raw.set_gripper
            monkeypatch.setattr(raw, "set_gripper",
                                lambda *a, _r=real, _l=jaws[n], **k: (_l.append(float(a[0])), _r(*a, **k))[1])
        if holder is not None:
            raw = raws[holder]
            raw._gripper = raw._grip_closed + 0.5 * (raw._grip_open - raw._grip_closed)
            rt.held_object, rt._held_arm = ("yellow cup" if cup else "red cube"), holder
            rt._held_det_label, rt._held_color = rt.held_object, ("yellow" if cup else "red")
            rt._held_width_m, rt._held_support_offset_m = 0.025, 0.04
        if camera_part:
            c, axes, half = np.array([0.0, 0.0, 0.0]), np.eye(3), np.full(3, 0.006)
            rt.arm_rig.get(caller).harness.tool_volume = ToolVolume(
                local=box_grid(c, axes, half, 0.004), part=np.zeros(len(box_grid(c, axes, half, 0.004)), dtype=int),
                names=("wrist camera",), boxes=((c, axes, half),))
        n0 = {n: len(raw.commands) for n, raw in raws.items()}
        out = rt.execute("point_at", {"label": "red cube", "arm": caller})
        # a test of who holds what, never of a slow runner's perception watchdog
        assert "perception watchdog" not in str(out.get("error")), out
        cmds = {n: [np.round(np.asarray(q, float), 9).tolist() for q in raws[n].commands[n0[n]:]] for n in _PAIR}
        keep = ("ok", "error", "pointed_at", "position", "lean_deg", "rose_first", "backed_out_m")
        return {k: out.get(k) for k in keep}, cmds, jaws, (rt.held_object, getattr(rt, "_held_arm", None))


@needs_pin
@pytest.mark.parametrize("box", ["shipped", "nobox"])
@pytest.mark.parametrize("mover", _PAIR)
@pytest.mark.parametrize("cup", [False, True], ids=["holds_the_cube", "holds_a_cup_beside_it"])
def test_on_a_pair_the_empty_arm_points_as_if_nothing_were_held(make_runtime, monkeypatch, mover, box, cup):  # noqa: F811
    """The #400 re-review's probe rows (8/8 refused on main: "cannot point at
    'red cube' while holding 'red cube'"): the empty arm's point_at equals the
    same call with nothing held (result, its commands, both arms' jaws), the
    holding arm gets no command, and the hold stays where it was. With a cup
    in the other arm's jaws, the cup's remembered place still counts as a
    neighbour (main skipped it as if it were in this arm's jaws)."""
    none = _pair_point_at(make_runtime, monkeypatch, side=mover, caller=mover, holder=None, box=box, cup=cup)
    held = _pair_point_at(make_runtime, monkeypatch, side=mover, caller=mover, holder=_OTHER[mover], box=box, cup=cup)

    assert held[0] == none[0]
    assert held[1] == none[1] and held[1][_OTHER[mover]] == []
    assert held[2] == none[2]
    assert held[3] == (("yellow cup" if cup else "red cube"), _OTHER[mover])
    if not cup:
        assert none[0]["ok"] is True and none[1][mover], "test setup: the empty arm must point"


@needs_pin
def test_on_a_pair_the_empty_arm_keeps_its_own_tool_parts(make_runtime, monkeypatch):  # noqa: F811
    """The camera-box clearance stays on the empty arm: its harness's tool
    parts plan every route whether or not the other arm holds something."""
    from cascade.skills import pointing

    mover = "so101_left"
    seen = {"none": [], "held": []}
    now = ["none"]
    real_route = pointing.plan_route

    def route(*a, **k):
        seen[now[0]].append(sorted(part.name for part in k.get("body", ())))
        return real_route(*a, **k)

    monkeypatch.setattr(pointing, "plan_route", route)
    none = _pair_point_at(make_runtime, monkeypatch, side=mover, caller=mover, holder=None, box="shipped",
                          camera_part=True)
    now[0] = "held"
    held = _pair_point_at(make_runtime, monkeypatch, side=mover, caller=mover, holder=_OTHER[mover], box="shipped",
                          camera_part=True)

    assert seen["none"] and seen["held"] == seen["none"] and all(b == ["wrist camera"] for b in seen["held"]), seen
    assert held[:3] == none[:3] and none[0]["ok"] is True, (none[0], held[0])


@needs_pin
@pytest.mark.parametrize("holder", _PAIR)
def test_on_a_pair_the_holding_arm_keeps_its_hold_rules(make_runtime, monkeypatch, holder):  # noqa: F811
    """Holder control: the arm that holds the cube still points as a holding
    arm does (never leans in, never commands its jaws), and keeps its hold."""
    out, cmds, jaws, hold = _pair_point_at(make_runtime, monkeypatch, side=holder, caller=holder, holder=holder,
                                           box="shipped")
    assert hold == ("red cube", holder)
    assert jaws == {n: [] for n in _PAIR}, jaws
    assert cmds[_OTHER[holder]] == []
    assert out["ok"] is False and "while holding 'red cube'" in out["error"], out
