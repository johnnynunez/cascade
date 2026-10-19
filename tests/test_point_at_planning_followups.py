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
"""

from __future__ import annotations

import math
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
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
    work = lambda: 1e-3 if os.sched_getaffinity(0) == {fast} else 3e-3  # noqa: E731
    ref = core_speed.fastest(work)
    assert ref == core_speed.Reference(1e-3, True), ref

    caps.update({barred: 2048})                    # barred alone in the fastest class
    ref = core_speed.fastest(work)
    _join_probe_helpers()
    assert ref == core_speed.Reference(1e-3, True), ref


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
