"""How fast each class of this host's CPU cores runs a workload, and which core a thread runs on.

Hosts with two kinds of core run the same Python work at different speeds: the
GB10 (DGX Spark) has ten Cortex-X925 and ten Cortex-A725 cores, and a planner
on an A725 takes 2.5-3x the CPU time it takes on an X925. A CPU-time budget is
blind to contention but not to that, so it charges the CPU a thread spent by
the class of the core it ran on (`current_cpu`). `class_times` gives each
class's time: a workload timed on one core of every capacity class (sysfs
`cpu_capacity`) of this PROCESS's CPU set, each by a helper thread pinned
there (the caller is neither timed nor moved), fastest class first. The
process's set, not the caller's: a caller pinned to a slow core still finds the
fast one. A helper first runs the workload untimed for WARMUP_S of its CPU (a
core that was idle runs its first milliseconds of work at a low clock), then
CLASS_RUNS timed runs: their median is the class's time. Each helper gets
PER_CORE_S to be given its core (a loaded core may make it wait tens of ms); a
class that did not answer in time is left out (`complete=False`), and its
caller must not take its cores for measured. At most one helper runs at a
time, process-wide: a late helper ends its search and stops after the run in
hand. Without capacities (an x86 host) every core is a class of its own, in
CPU-number order. Linux only (per-thread affinity, `sched_getcpu`); elsewhere
no class is timed and no core is known, i.e. no core counts as slower.

This lives outside `cascade.skills`: a helper thread here must run only the
workload it is given, which must read no runtime state (a skill's arm frame is
thread-local; tests/test_arm_frame_context.py keeps skill modules free of
thread starts).
"""

from __future__ import annotations

import math
import os
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: stop starting helpers after this long (s, wall): the GB10's five capacity
#: classes at PER_CORE_S each
SEARCH_S = 1.0
#: how long to wait for one core's helper (s, wall): a busy core may not give
#: the helper its turn soon (the #399 review's probe: 60 ms)
PER_CORE_S = 0.2
#: a helper's untimed warm-up on its core (s of its own CPU): an idle core's
#: clock has not ramped up for its first milliseconds of work (the #425 review
#: read an X925 1.3-1.7x slow right after a sleep)
WARMUP_S = 0.015
#: timed runs per class after the warm-up; their median is the class's time
CLASS_RUNS = 5
#: where Linux publishes each core's capacity (1024 = the host's fastest)
SYSFS_CPU = "/sys/devices/system/cpu"

#: the helpers still running (one at most, #406 review Minor 2): a helper
#: that has not answered in time ends its search, no search starts a helper
#: while one runs, and a helper whose search is over runs no more of its work
_running: set = set()
_running_lock = threading.Lock()


def _libc_sched_getcpu():
    try:
        import ctypes

        getcpu = ctypes.CDLL(None, use_errno=True).sched_getcpu
    except (OSError, AttributeError, TypeError):
        return None
    getcpu.restype, getcpu.argtypes = ctypes.c_int, []
    return getcpu


_SCHED_GETCPU = _libc_sched_getcpu()


def current_cpu() -> int:
    """The core the calling thread runs on now (`sched_getcpu`), -1 when this
    host cannot tell."""
    if _SCHED_GETCPU is None:
        return -1
    cpu = int(_SCHED_GETCPU())
    return cpu if cpu >= 0 else -1


@dataclass(frozen=True)
class ClassTimes:
    """`class_times`' answer: {a class's cores (a tuple, CPU-number order): the
    workload's median time on one of them} for every class that answered in
    time (`seconds`), and whether every class searched did (`complete`)."""

    seconds: dict = field(default_factory=dict)
    complete: bool = True


def _capacities(cpus, root=SYSFS_CPU) -> dict | None:
    """{cpu: its sysfs `cpu_capacity`} for every core in `cpus`, or None when
    any of them has none readable (then no core is told apart)."""
    if not cpus:
        return None
    out = {}
    for cpu in cpus:
        try:
            out[cpu] = int(Path(root, f"cpu{cpu}", "cpu_capacity").read_text().strip())
        except (OSError, ValueError):
            return None
    return out


def _process_cpus() -> set:
    """Every core this process may run on: the main thread's affinity (the
    process's CPU set, as `taskset -p` reports it) and the caller's own."""
    return set(os.sched_getaffinity(os.getpid())) | set(os.sched_getaffinity(0))


def _search_order(cpus) -> list:
    """`cpus` in capacity classes, the fastest class first, each class's
    cores in CPU-number order; without capacities each core is a class of
    its own, in CPU-number order."""
    caps = _capacities(set(cpus))
    if caps is None:
        return [[cpu] for cpu in sorted(cpus)]
    classes: dict = {}
    for cpu in sorted(cpus):
        classes.setdefault(caps[cpu], []).append(cpu)
    return [classes[cap] for cap in sorted(classes, reverse=True)]


def _a_time(t) -> bool:
    return isinstance(t, (int, float)) and math.isfinite(t) and t > 0.0


def class_times(workload: Callable[[], float], *, skip=frozenset(), runs: int | None = None,
                warmup_s: float | None = None) -> ClassTimes:
    """`workload()` (CPU seconds of a fixed slice of work, timed by the
    workload itself on its own thread) on one core of each capacity class of
    the process's CPU set but those in `skip` (classes already timed, as
    `ClassTimes` keys), fastest class first: run untimed for `warmup_s`
    (WARMUP_S) of the helper's CPU, then the median of `runs` (CLASS_RUNS)
    timed runs (a class's next core only when this one cannot be pinned; a
    class none of whose cores can be is no part of the set). Runs that are
    not all positive finite times are no answer.

    One helper at a time, process-wide (#406 review Minor 2: a hung workload
    left one helper per class alive): a helper that has not answered within
    its PER_CORE_S ends the search (`complete=False`), a search finds one
    still running and starts none (`complete=False`), and a helper whose
    search is over runs no more of its work."""
    if not (hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")):
        return ClassTimes()
    runs = CLASS_RUNS if runs is None else runs
    warmup_s = WARMUP_S if warmup_s is None else warmup_s
    stop = time.monotonic() + SEARCH_S
    over = threading.Event()             # this search is over
    seconds: dict = {}
    complete = True
    try:
        for cores in _search_order(_process_cpus()):
            if tuple(cores) in skip:
                continue
            answered = pinnable = False
            for cpu in cores:
                left = stop - time.monotonic()
                token = object()
                # no helper running, and this search's own registered, in ONE
                # section: a search checking in between finds it running
                # (#411 review Minor 3)
                with _running_lock:
                    busy = bool(_running)
                    if not busy and left > 0.0:
                        _running.add(token)
                if busy or not left > 0.0:
                    pinnable = True        # out of time, or a helper still runs, before this class was tried
                    break
                out: list = []
                pinned: list = []

                def timed(cpu=cpu, out=out, pinned=pinned, token=token):
                    found = None
                    try:
                        try:
                            os.sched_setaffinity(0, {cpu})          # this helper thread only
                        except OSError:
                            return                                 # not this core: the class's next one
                        pinned.append(cpu)
                        warm = time.thread_time() + warmup_s
                        while time.thread_time() < warm:           # untimed: the core's clock ramps up
                            if over.is_set():
                                return                             # its search ended without it
                            workload()
                        times = []
                        for _ in range(runs):
                            if over.is_set():
                                return
                            times.append(workload())
                        if times and all(_a_time(t) for t in times):
                            found = float(statistics.median(times))
                    finally:
                        with _running_lock:                        # its answer and its end, at once
                            if found is not None:
                                out.append(found)
                            _running.discard(token)

                try:
                    helper = threading.Thread(target=timed, daemon=True, name="core-speed-probe")
                    helper.start()
                except BaseException:
                    with _running_lock:
                        _running.discard(token)
                    raise
                helper.join(min(PER_CORE_S, left))
                with _running_lock:
                    running = token in _running
                if pinned or running:
                    # this class's core: its answer, if it came in time (a late,
                    # failed or meaningless one leaves the class out); never
                    # another of its cores
                    pinnable = True
                    if out:
                        seconds[tuple(cores)] = out[0]
                        answered = True
                    break
            if pinnable and not answered:
                complete = False
    finally:
        over.set()
    return ClassTimes(seconds, complete)
