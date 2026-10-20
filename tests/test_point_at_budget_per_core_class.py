"""point_at's planning budget, counted per core class (#425 post-merge review, Important 1 and follow-up 2).

#425 timed the planning thread's core ONCE, right before planning, and stretched the whole 3 s CPU budget by that
one factor. That timing came right after point_at's localization (a camera and a detector server wait): a thread
just back from a sleep read 1.3-1.7x on an X925 (its clock had not ramped up yet) in 2-7 % of plans, or timed an
A725 it left within 60 ms and planned on X925s with an A725 budget. And a planner moved to a slower core mid-plan
kept the faster core's budget: 3 of 52 heavy plans were refused under a toggling load (none before #425 removed the
re-measure). Now:

1. every budget check charges the planning thread's CPU since the last check by the class of the core it ran on
   (`pointing._cpu`, sampled at both checks; the faster of the two counts): the budget is 3 s of the fastest class's
   CPU, wherever the plan ran, and still never past the 8 s wall cap;
2. a class's factor is measured once per process by a helper thread pinned to one of its cores
   (`core_speed.class_times`: an untimed warm-up, then the median of CLASS_RUNS runs); the planning thread is never
   timed, so a cold core after a sleep cannot stretch its budget. A class not measured in time counts as the
   fastest (no stretch) and is measured again after SEARCH_AGAIN_S;
3. a factor that is not a finite number >= 1 counts as 1; none stretches past the wall cap.
"""

from __future__ import annotations

import math
import os
import threading
import time

import pytest
from conftest import needs_pin
from test_point_at_planning_followups import _commands, _join_probe_helpers, rig  # noqa: F401  (rig: the fixture)

from cascade.skills.pointing import class_factors as _CLASS_FACTORS

needs_three_cores = pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or len(os.sched_getaffinity(0)) < 3,
    reason="Linux: per-thread CPU affinity, three cores")

#: the kernel's CPU time (s) on each fake class: an A725 (2.7x), an X925 of lower capacity, the fastest X925
_SECONDS = {"slow": 1.08e-3, "mid": 0.41e-3, "fast": 0.40e-3}
_CAPACITY = {"slow": 718, "mid": 997, "fast": 1024}


class _Cores:
    """This host's cores as three capacity classes (`core_speed._capacities`): the first core `slow` (an A725), the
    last `fast` (the fastest X925), the rest `mid` (an X925 class of lower capacity), and the planner's own kernel
    (`pointing._speed_probe_s`) timed on them: `_SECONDS[class]`, except that
    - `cold[class] = (n, x)`: the first n readings of every thread on a core of that class read x times slower (the
      core's clock had not ramped up yet);
    - `outlier[class] = (k, s)`: every k-th reading of a thread on that class reads s seconds.
    A core-speed helper (pinned to one core) runs on its core; any other thread, the planning thread, on
    `self.planner` (`pointing._cpu`). `planner_readings` counts the kernel timed outside the helpers."""

    def __init__(self, monkeypatch, *, cold=None, outlier=None):
        from cascade import core_speed
        from cascade.skills import pointing

        cpus = sorted(os.sched_getaffinity(0))
        self.of = {cpu: "mid" for cpu in cpus} | {cpus[0]: "slow", cpus[-1]: "fast"}
        self.core = {"slow": cpus[0], "mid": cpus[1], "fast": cpus[-1]}
        self.planner = self.core["mid"]
        self.cold, self.outlier = dict(cold or {}), dict(outlier or {})
        self.readings: dict = {}
        self.planner_readings = 0
        self._lock = threading.Lock()
        monkeypatch.setattr(core_speed, "_capacities", lambda cores: {c: _CAPACITY[self.of[c]] for c in cores})
        monkeypatch.setattr(core_speed, "PER_CORE_S", 5.0)        # a loaded runner still answers every class
        monkeypatch.setattr(core_speed, "SEARCH_S", 30.0)
        monkeypatch.setattr(pointing, "_speed_probe_s", self.probe)
        monkeypatch.setattr(pointing, "_cpu", lambda: self.planner, raising=False)
        monkeypatch.setattr(pointing, "_CLASSES", {}, raising=False)
        monkeypatch.setattr(pointing, "class_factors", _CLASS_FACTORS)

    def probe(self) -> float:
        helper = threading.current_thread().name == "core-speed-probe"
        cpu = next(iter(os.sched_getaffinity(0))) if helper else self.planner
        cls = self.of[cpu]
        key = (threading.get_ident(), cpu)
        with self._lock:
            n = self.readings[key] = self.readings.get(key, 0) + 1
            if not helper:
                self.planner_readings += 1
        every, value = self.outlier.get(cls, (0, None))
        if every and n % every == 0:
            return value
        first, slower = self.cold.get(cls, (0, 1.0))
        return _SECONDS[cls] * (slower if n <= first else 1.0)


def _spend(monkeypatch, cores, steps):
    """The planning thread's CPU (`pointing._clock`): steps[i] = (seconds of CPU it spends before its i-th budget
    check, the class it ran on from then). Returns the budgets' charged CPU at each check."""
    from cascade.skills import pointing

    cpu = [0.0]
    monkeypatch.setattr(pointing, "_clock", lambda: cpu[0])
    real_check, seen = pointing.Budget.check, []

    def check(self):
        if threading.get_ident() == self.thread and len(seen) < len(steps):
            seconds, cls = steps[len(seen)]
            cores.planner = cores.core[cls]
            cpu[0] += seconds
        try:
            return real_check(self)
        finally:
            seen.append(getattr(self, "charged", None))

    monkeypatch.setattr(pointing.Budget, "check", check)
    return seen


# ── 1. the spin3.5 rows: the planner's CPU, counted by the class it ran on ──

#: row: (where the planner starts, (CPU before each check, the class it ran on), fakes, points?)
_SPIN_ROWS = {
    # a cold X925 class read slow first (the wake-up after the localization's wait): no stretch
    "x925_whose_class_reads_cold_at_first": ("mid", ((1.0, "mid"), (3.0, "mid")), {"cold": {"mid": (3, 1.6)}}, False),
    # one fast reading on the fastest class (every 5th): the median ignores it
    "x925_with_a_fast_outlier_on_the_fastest_class": ("mid", ((1.0, "mid"), (3.0, "mid")),
                                                      {"outlier": {"fast": (5, 0.1e-3)}}, False),
    # one slow reading on the planner's class (every 5th): the median ignores it
    "x925_with_a_slow_outlier_on_its_own_class": ("mid", ((1.0, "mid"), (3.0, "mid")),
                                                  {"outlier": {"mid": (5, 1.5e-3)}}, False),
    # the review's 1/40: started on an A725, then planned on X925s: 1 s there counts 0.375 s, 3 s on the X925 3 s
    "moved_from_an_a725_to_an_x925": ("slow", ((1.0, "slow"), (3.0, "mid")), {}, False),
    # control: on an A725 the whole plan, the documented stretch (3 s x 2.67, the 8 s wall cap): 4 s of CPU points
    "an_a725_the_whole_plan": ("slow", ((1.0, "slow"), (3.0, "slow")), {}, True),
    # ... and its cold first readings change nothing
    "an_a725_whose_class_reads_cold_at_first": ("slow", ((1.0, "slow"), (3.0, "slow")),
                                                {"cold": {"slow": (3, 1.6)}}, True),
}


@needs_pin
@needs_three_cores
@pytest.mark.parametrize("row", list(_SPIN_ROWS))
def test_a_cpu_spin_past_the_budget_is_refused_unless_it_ran_on_a_slower_core(rig, monkeypatch, row):  # noqa: F811
    """The r399f stop lane's `spin3.5_cpu` row, deterministic, with the shipped 3 s budget and 8 s cap: the planning
    thread spends 4 s of CPU over its first two checks. main stretched the whole budget by one timing of the
    planning thread's core before planning, so a core that read cold (1.6x), a fastest class with one fast reading
    or a plan that started on an A725 for its first second pointed (136 commands and a jaw close). The planning
    thread is never timed now; its CPU is charged by the class it ran on, with the classes' factors measured after
    a warm-up (median): those rows refuse with nothing sent, and a plan that ran on an A725 keeps its stretch."""
    from cascade.skills import pointing

    start, steps, fakes, points = _SPIN_ROWS[row]
    r = rig("mock")
    monkeypatch.setattr(pointing, "PLAN_BUDGET_S", 3.0)
    monkeypatch.setattr(pointing, "PLAN_WALL_S", 8.0)
    cores = _Cores(monkeypatch, **fakes)
    cores.planner = cores.core[start]
    charged = _spend(monkeypatch, cores, steps)

    try:
        out = r.runtime.execute("point_at", {"label": "red cube"})
    finally:
        _join_probe_helpers()

    assert len(charged) >= 2, "test setup: planning checked its budget twice"
    if points:
        assert out["ok"] is True and _commands(r) and r.jaws, out
    else:
        assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
        assert "planning took longer than its 3.0 s budget" in out["error"], out
    assert cores.planner_readings == 0, "the planning thread's own core is never timed"


# ── 2. exactly the fastest class's CPU, wherever the plan ran ──────────────


def _factors(monkeypatch, factors):
    """A planner's class factors (`pointing.class_factors`) and its core and CPU clocks (`_cpu`, `_clock`)."""
    from cascade.skills import pointing

    cpu, now = [0], [0.0]
    monkeypatch.setattr(pointing, "class_factors", lambda: dict(factors))
    monkeypatch.setattr(pointing, "_cpu", lambda: cpu[0], raising=False)
    monkeypatch.setattr(pointing, "_clock", lambda: now[0])
    return cpu, now


@pytest.mark.parametrize("order", ["a725_then_x925", "x925_then_a725"])
def test_a_planner_moved_between_core_classes_gets_exactly_the_fastest_classs_budget(monkeypatch, order):
    """3 s of the fastest class's CPU, wherever it was spent: 1.325 s on an A725 (2.65x) counts 0.5 s, so 2.5 s are
    left on the X925 (`a725_then_x925`, the review's planner that started on an A725 and planned on X925s: #425
    allowed it 8 s); 1 s on the X925 leaves 2 s of X925 CPU = 5.3 s on the A725 (`x925_then_a725`, a planner moved
    to a slower core mid-plan: #425 refused it at 3 s). Refused 1 us past it, not before."""
    from cascade.skills import pointing

    cpu, now = _factors(monkeypatch, {0: 2.65, 5: 1.0})
    first, then = (0, 5) if order == "a725_then_x925" else (5, 0)
    cpu[0] = first
    budget = pointing.Budget(3.0, wall=30.0)
    now[0] = 1.325 if first == 0 else 1.0
    budget.check()
    cpu[0] = then                                 # moved (no CPU between these two checks)
    budget.check()
    left = 2.5 if then == 5 else 2.0 * 2.65       # 2.5 s of the X925's CPU, in that core's own seconds
    now[0] += left - 1e-6
    budget.check()
    assert budget.charged == pytest.approx(3.0)
    now[0] += 2e-6
    with pytest.raises(pointing.PlanningTimeout, match=r"planning took longer than its 3\.0 s budget"):
        budget.check()


def test_the_cpu_between_two_checks_on_two_classes_counts_at_the_faster():
    """The core is sampled at each check (milliseconds apart in a plan): CPU spent between a check on an A725 and
    one on an X925 may have run on either, so it counts as the X925's (never a stretch the samples cannot
    justify)."""
    from cascade.skills import pointing

    mp = pytest.MonkeyPatch()
    try:
        cpu, now = _factors(mp, {0: 2.5, 5: 1.0})
        budget = pointing.Budget(3.0, wall=30.0)        # on the A725
        cpu[0] = 5
        now[0] = 1.0
        budget.check()                                  # 1 s between the A725 and the X925: 1 s
        assert budget.charged == pytest.approx(1.0)
        cpu[0] = 0
        now[0] = 2.0
        budget.check()                                  # 1 s between the X925 and the A725: 1 s
        assert budget.charged == pytest.approx(2.0)
        now[0] = 3.0
        budget.check()                                  # 1 s on the A725 at both ends: 0.4 s
        assert budget.charged == pytest.approx(2.4)
        assert budget.spent == pytest.approx(3.0)
    finally:
        mp.undo()


def test_a_core_of_no_measured_class_counts_as_the_fastest(monkeypatch):
    """A core the class search did not measure (not given its turn in time; `sched_getcpu` unavailable: -1) never
    stretches the budget."""
    from cascade.skills import pointing

    cpu, now = _factors(monkeypatch, {0: 2.5})
    for core in (7, -1):
        cpu[0] = core
        now[0] = 0.0
        budget = pointing.Budget(3.0, wall=30.0)
        now[0] = 3.0
        budget.check()
        now[0] = 3.001
        with pytest.raises(pointing.PlanningTimeout, match=r"budget$"):
            budget.check()


_BAD_FACTORS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf, "zero": 0.0, "negative": -2.0,
                "tiny": 1e-300, "below_one": 0.5, "a_string": "2.5", "none": None}


@pytest.mark.parametrize("bad", list(_BAD_FACTORS))
def test_a_factor_that_is_not_a_finite_number_of_at_least_one_counts_as_one(monkeypatch, bad):
    """Fall back to no stretch, never to a longer budget and never to a refusal of the first check."""
    from cascade.skills import pointing

    cpu, now = _factors(monkeypatch, {0: _BAD_FACTORS[bad]})
    budget = pointing.Budget(3.0, wall=30.0)
    now[0] = 3.0
    budget.check()
    now[0] = 3.001
    with pytest.raises(pointing.PlanningTimeout, match=r"budget$"):
        budget.check()


@pytest.mark.parametrize("factor", [9.0, 1e9])
def test_no_factor_stretches_the_budget_past_the_wall_cap(monkeypatch, factor):
    """3 s x a factor above PLAN_WALL_S / PLAN_BUDGET_S is the 8 s cap's CPU, no more; the wall cap itself still
    ends planning at 8 s."""
    from cascade.skills import pointing

    cpu, now = _factors(monkeypatch, {0: factor})
    budget = pointing.Budget(3.0, wall=8.0)
    now[0] = 7.99
    budget.check()
    now[0] = 8.01
    with pytest.raises(pointing.PlanningTimeout,
                       match=r"budget \(8\.0 s of CPU, slower cores counted at their speed\)$"):
        budget.check()


def test_a_cpu_clock_that_goes_back_is_never_credited(monkeypatch):
    """The thread's CPU clock never runs backwards; a reading that does (or NaN) refuses rather than giving the
    budget back."""
    from cascade.skills import pointing

    cpu, now = _factors(monkeypatch, {})
    budget = pointing.Budget(3.0, wall=30.0)
    now[0] = 2.9
    budget.check()
    now[0] = 1.0
    with pytest.raises(pointing.PlanningTimeout):
        budget.check()
    now[0] = 1.1
    with pytest.raises(pointing.PlanningTimeout):
        budget.check()


def test_the_class_factors_are_read_once_when_the_budget_starts(monkeypatch):
    """#423 review follow-up 5's re-measure never comes back: the factors are read once, before planning; each check
    only samples the core and the CPU clock."""
    from cascade.skills import pointing

    calls = []
    cpu, now = _factors(monkeypatch, {})
    monkeypatch.setattr(pointing, "class_factors", lambda: calls.append(1) or ({} if len(calls) == 1 else {0: 2.5}))
    budget = pointing.Budget(3.0, wall=30.0)
    now[0] = 3.0
    budget.check()
    now[0] = 3.5
    for _ in range(3):
        with pytest.raises(pointing.PlanningTimeout, match=r"budget$"):
            budget.check()
    assert calls == [1]


# ── 3. the class factors: measured once per class, on its own core ─────────


@needs_three_cores
def test_each_class_is_timed_by_a_helper_on_one_of_its_cores_after_a_warm_up(monkeypatch):
    """`class_factors`: the median of the kernel's timed runs on one core of each class, after an untimed warm-up of
    10-20 ms of that helper's CPU, over the fastest class's; a class within SLOWER_CORE_MIN of it counts as it. The
    planning thread is neither timed nor moved."""
    from cascade import core_speed
    from cascade.skills import pointing

    assert 0.010 <= core_speed.WARMUP_S <= 0.020 and core_speed.CLASS_RUNS >= 3
    cores = _Cores(monkeypatch, cold={"slow": (3, 3.0), "mid": (3, 3.0), "fast": (3, 3.0)})
    calls = []
    real = cores.probe

    def probe():
        calls.append((threading.get_ident(), frozenset(os.sched_getaffinity(0)), time.thread_time()))
        return real()

    monkeypatch.setattr(pointing, "_speed_probe_s", probe)
    before = os.sched_getaffinity(0)
    try:
        factors = pointing.class_factors()
    finally:
        _join_probe_helpers()

    assert os.sched_getaffinity(0) == before and cores.planner_readings == 0
    assert factors[cores.core["slow"]] == pytest.approx(1.08 / 0.40)
    assert factors[cores.core["mid"]] == 1.0 and factors[cores.core["fast"]] == 1.0
    assert set(factors) == set(before)
    for cls in ("slow", "mid", "fast"):
        mine = [c for c in calls if c[1] == {cores.core[cls]}]
        assert len({c[0] for c in mine}) == 1, "one helper per class"
        timed = mine[-core_speed.CLASS_RUNS:]
        assert timed[0][2] - mine[0][2] >= 0.010, "an untimed warm-up of at least 10 ms of the helper's CPU first"


@needs_three_cores
def test_a_class_time_is_the_median_of_its_class_runs_timed_runs(monkeypatch):
    """Without the warm-up, a class's time is the median of exactly CLASS_RUNS timed runs: one slow first reading on
    the fastest class (10x) changes no factor."""
    from cascade import core_speed
    from cascade.skills import pointing

    monkeypatch.setattr(core_speed, "WARMUP_S", 0.0)
    cores = _Cores(monkeypatch, cold={"fast": (1, 10.0)})
    try:
        factors = pointing.class_factors()
    finally:
        _join_probe_helpers()

    assert factors[cores.core["slow"]] == pytest.approx(1.08 / 0.40) and factors[cores.core["fast"]] == 1.0
    for cls in ("slow", "mid", "fast"):
        assert sum(n for (_, cpu), n in cores.readings.items() if cpu == cores.core[cls]) == core_speed.CLASS_RUNS


@needs_three_cores
def test_a_class_is_measured_once_per_process(monkeypatch):
    from cascade.skills import pointing

    cores = _Cores(monkeypatch)
    try:
        first = pointing.class_factors()
        n = dict(cores.readings)
        again = pointing.class_factors()
        pointing.Budget(3.0, wall=30.0)
    finally:
        _join_probe_helpers()
    assert again == first and cores.readings == n, "measured once, kept for the process"


@needs_three_cores
def test_a_class_not_measured_in_time_is_not_kept_and_is_measured_again_later(monkeypatch):
    """A helper not given its core in time (a loaded core) leaves its class unmeasured: its cores count as the
    fastest's (no stretch), the other classes are kept, every call for SEARCH_AGAIN_S uses that answer (never a
    search on every call), then one call measures the missing class alone."""
    from cascade import core_speed
    from cascade.skills import pointing

    now = [100.0]
    monkeypatch.setattr(pointing, "_wall", lambda: now[0])
    cores = _Cores(monkeypatch)
    monkeypatch.setattr(core_speed, "PER_CORE_S", 0.2)
    gate = threading.Event()
    real = cores.probe

    def probe():
        if os.sched_getaffinity(0) == {cores.core["slow"]} and not gate.is_set():
            gate.wait(10.0)                                     # its helper stalls
        return real()

    monkeypatch.setattr(pointing, "_speed_probe_s", probe)
    try:
        first = pointing.class_factors()
        assert cores.core["slow"] not in first and first[cores.core["mid"]] == 1.0, first
        assert pointing.class_factors() == first                # within SEARCH_AGAIN_S: no search
    finally:
        gate.set()
        _join_probe_helpers()
    measured = {cpu for (_, cpu) in cores.readings}
    now[0] += pointing.SEARCH_AGAIN_S
    cores.readings.clear()
    try:
        later = pointing.class_factors()
    finally:
        _join_probe_helpers()
    assert later[cores.core["slow"]] == pytest.approx(1.08 / 0.40)
    assert {cpu for (_, cpu) in cores.readings} == {cores.core["slow"]}, "only the missing class is measured again"
    assert cores.core["fast"] in measured
    cores.readings.clear()
    now[0] += 10 * pointing.SEARCH_AGAIN_S
    assert pointing.class_factors() == later and cores.readings == {}, "complete: kept for good"


@needs_three_cores
@pytest.mark.parametrize("reading", [math.nan, math.inf, 0.0, -1e-3])
def test_a_class_whose_timing_is_not_a_positive_finite_time_is_not_kept(monkeypatch, reading):
    from cascade.skills import pointing

    cores = _Cores(monkeypatch)
    real = cores.probe
    monkeypatch.setattr(pointing, "_speed_probe_s",
                        lambda: reading if os.sched_getaffinity(0) == {cores.core["slow"]} else real())
    try:
        factors = pointing.class_factors()
    finally:
        _join_probe_helpers()
    assert cores.core["slow"] not in factors and factors[cores.core["fast"]] == 1.0, factors
    # the fastest class's own timing broken: no factor at all
    monkeypatch.setattr(pointing, "_CLASSES", {})
    monkeypatch.setattr(pointing, "_speed_probe_s",
                        lambda: reading if os.sched_getaffinity(0) == {cores.core["fast"]} else real())
    try:
        factors = pointing.class_factors()
    finally:
        _join_probe_helpers()
    assert cores.core["fast"] not in factors and all(f >= 1.0 for f in factors.values()), factors


def test_the_core_the_planning_thread_runs_on_is_read_from_the_kernel():
    from cascade import core_speed
    from cascade.skills import pointing

    cpu = core_speed.current_cpu()
    assert isinstance(cpu, int)
    if hasattr(os, "sched_getaffinity") and cpu != -1:
        assert cpu in os.sched_getaffinity(0)
    assert pointing._cpu is core_speed.current_cpu


@needs_three_cores
def test_measuring_the_classes_leaves_no_thread_behind(monkeypatch):
    from cascade.skills import pointing

    _Cores(monkeypatch)
    for _ in range(3):
        monkeypatch.setattr(pointing, "_CLASSES", {})
        pointing.Budget(3.0, wall=30.0)
    _join_probe_helpers()
    assert [t.name for t in threading.enumerate() if t.name == "core-speed-probe"] == []

# ── 5. a Python without ctypes (#429 review Minor 5) ────────────────────────


def test_a_python_without_ctypes_still_imports_and_knows_no_core(monkeypatch):
    """`_libc_sched_getcpu` caught OSError / AttributeError / TypeError but not ImportError: on a Python built
    without `_ctypes`, `cascade.skills.pointing` failed to import. There no core is known (-1: no stretch)."""
    import sys

    from cascade import core_speed

    monkeypatch.setitem(sys.modules, "ctypes", None)        # `import ctypes` raises ImportError
    assert core_speed._libc_sched_getcpu() is None
    monkeypatch.setattr(core_speed, "_SCHED_GETCPU", None)
    assert core_speed.current_cpu() == -1
