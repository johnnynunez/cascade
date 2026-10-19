"""point_at's core-speed search (#406 post-merge review, Minors 1 and 2).

1. Minor 1: a search that ran out of time (a loaded host: up to ~1.2 s) was
   repeated on every point_at call and on every re-measure of `Budget.check`.
   Its answer (the least of every such search) now serves every call for
   `pointing.SEARCH_AGAIN_S`; then a call searches again; a complete search is
   kept for good.
2. Minor 2: a hung helper left one helper per capacity class alive (5 on the
   GB10, 2.5 s after the search, each then running all its repeats). One
   helper runs at a time, process-wide: a helper that has not answered in time
   ends its search, no search starts another while it runs, and once its
   search is over it runs no more of its work.
"""

from __future__ import annotations

import os
import threading

import pytest
from test_point_at_planning_followups import (
    _CORE_FACTOR,
    _join_probe_helpers,
    _one_fast_core,
    needs_affinity,
)

# ── 1. an incomplete search is not repeated on every call ──────────────────


def test_an_incomplete_core_speed_search_is_not_repeated_on_every_call(monkeypatch):
    """A search that ran out of time (a loaded host, ~1.2 s) serves every
    call for SEARCH_AGAIN_S (the least of every such search), then one call
    searches again; a complete one is kept for good. main searched again on
    every call (and on every re-measure of `Budget.check`)."""
    from cascade import core_speed
    from cascade.skills import pointing

    now = [100.0]
    monkeypatch.setattr(pointing, "_wall", lambda: now[0])
    monkeypatch.setattr(pointing, "_FASTEST", {})
    monkeypatch.setattr(pointing, "core_factor", _CORE_FACTOR)
    answers = iter([core_speed.Reference(3e-3, False), core_speed.Reference(4e-3, False),
                    core_speed.Reference(2e-3, False), core_speed.Reference(1e-3, True)])
    searches = []

    def fastest(workload, repeats=3):
        searches.append(now[0])
        return next(answers)

    monkeypatch.setattr(core_speed, "fastest", fastest)

    for _ in range(3):
        pointing.Budget()                          # one search, not three
    assert searches == [100.0]
    assert pointing._fastest_probe_s() == 3e-3
    now[0] += pointing.SEARCH_AGAIN_S - 1.0
    assert pointing._fastest_probe_s() == 3e-3 and len(searches) == 1
    now[0] += 1.0
    assert pointing._fastest_probe_s() == 3e-3 and len(searches) == 2    # searched again; the least is kept
    now[0] += pointing.SEARCH_AGAIN_S
    assert pointing._fastest_probe_s() == 2e-3 and len(searches) == 3
    now[0] += pointing.SEARCH_AGAIN_S
    assert pointing._fastest_probe_s() == 1e-3 and len(searches) == 4    # complete: kept for good
    now[0] += 1e6
    assert pointing._fastest_probe_s() == 1e-3 and len(searches) == 4


def test_the_shipped_search_again_interval_is_pinned():
    from cascade.skills import pointing

    assert pointing.SEARCH_AGAIN_S == 60.0


# ── 2. a hung helper ───────────────────────────────────────────────────────


def _probe_threads():
    return [t for t in threading.enumerate() if t.name == "core-speed-probe"]


@needs_affinity
def test_a_hung_core_speed_helper_is_the_only_one_and_ends_with_its_hang(monkeypatch):
    """The review's "hang" row: every helper's work hangs. main started one
    helper per capacity class (5 on the GB10, all still alive 2.5 s later,
    each then running all its repeats). Now the search ends at the first
    helper that has not answered, no search starts another while it runs,
    and once its work returns it runs no more of it."""
    from cascade import core_speed

    cpus = sorted(os.sched_getaffinity(0))
    fast = cpus[-1]
    _one_fast_core(monkeypatch, fast)
    # long enough for its helper to reach its work on a loaded runner: the
    # hang must happen inside the search
    monkeypatch.setattr(core_speed, "PER_CORE_S", 0.5)
    _join_probe_helpers()
    assert not _probe_threads(), "test setup: no helper left by an earlier test"
    gate = threading.Event()
    main = threading.get_ident()
    runs = []

    def work():
        if threading.get_ident() != main:
            runs.append(sorted(os.sched_getaffinity(0)))
            gate.wait(10.0)
        return 1e-3 if os.sched_getaffinity(0) == {fast} else 3e-3

    try:
        first = core_speed.fastest(work)
        alive = _probe_threads()
        again = core_speed.fastest(work)
        still = _probe_threads()
    finally:
        gate.set()
    for t in still:
        t.join(5.0)

    assert first == core_speed.Reference(3e-3, False), first
    assert again == core_speed.Reference(3e-3, False), again
    assert len(alive) == 1 and still == alive, (alive, still)
    assert runs == [[fast]], runs                  # one helper, one run of its work
    assert not _probe_threads()


def test_a_helper_that_cannot_start_leaves_no_helper_behind(monkeypatch):
    """A thread that cannot start (the process is out of threads) raises, and
    leaves nothing that would make every later search find a helper running."""
    from cascade import core_speed

    if not hasattr(os, "sched_setaffinity"):
        pytest.skip("Linux: per-thread CPU affinity")
    _join_probe_helpers()
    real_start = threading.Thread.start

    def start(self):
        if self.name == "core-speed-probe":
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        core_speed.fastest(lambda: 1e-3)
    assert core_speed._running == set()
    started = []

    def counted(self):
        started.append(self.name)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", counted)
    core_speed.fastest(lambda: 1e-3)
    _join_probe_helpers()
    assert "core-speed-probe" in started, "a later search must still start its helpers"
