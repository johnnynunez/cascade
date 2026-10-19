"""How fast the calling thread's CPU core is next to the host's fastest.

Hosts with two kinds of core run the same Python work at different speeds: the
GB10 (DGX Spark) has ten Cortex-X925 and ten Cortex-A725 cores, and a planner
moved to an A725 takes 2.5-3x the CPU time it takes on an X925. A CPU-time
budget is blind to contention but not to that. `fastest` gives the reference:
a workload timed on the caller's core and on one core of every capacity class
(sysfs `cpu_capacity`) of this PROCESS's CPU set, each by a helper thread
pinned there (the caller's own affinity is never touched), fastest class
first, the least of them. The process's set, not the caller's: a caller
pinned to a slow core still finds the fast one. Each helper gets PER_CORE_S to
be given its core (a loaded core may make it wait tens of ms); a search that
ran out of time before every class answered is `complete=False`, and its
caller must not keep it for good (it can only overstate the reference). At most one
helper runs at a time, process-wide: a late helper ends its search and stops
after the work in hand. Without
capacities (an x86 host) every core is a class of its own, in CPU-number
order. Linux only (per-thread affinity); elsewhere the caller's own core is
the reference, i.e. no core counts as slower.

This lives outside `cascade.skills`: a helper thread here must run only the
workload it is given, which must read no runtime state (a skill's arm frame is
thread-local; tests/test_arm_frame_context.py keeps skill modules free of
thread starts).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: stop starting helpers after this long (s, wall): the GB10's five capacity
#: classes at PER_CORE_S each
SEARCH_S = 1.0
#: how long to wait for one core's helper (s, wall): a busy core may not give
#: the helper its turn soon (the #399 review's probe: 60 ms)
PER_CORE_S = 0.2
#: where Linux publishes each core's capacity (1024 = the host's fastest)
SYSFS_CPU = "/sys/devices/system/cpu"

#: the helpers still running (one at most, #406 review Minor 2): a helper
#: that has not answered in time ends its search, no search starts a helper
#: while one runs, and a helper whose search is over runs no more of its work
_running: set = set()
_running_lock = threading.Lock()


@dataclass(frozen=True)
class Reference:
    """`fastest`'s answer: the least `seconds` measured, and whether every
    capacity class answered in time (`complete`)."""

    seconds: float
    complete: bool


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


def fastest(workload: Callable[[], float], repeats: int = 3) -> Reference:
    """The least of `workload()` (CPU seconds of a fixed slice of work, timed
    by the workload itself on its own thread), `repeats` times each, over the
    caller's core and one core of each capacity class of the process's CPU
    set (a class's next core only when this one cannot be pinned).

    One helper at a time, process-wide (#406 review Minor 2: a hung workload
    left one helper per class alive): a helper that has not answered within
    its PER_CORE_S ends the search (`complete=False`), a search finds one
    still running and starts none (`complete=False`), and a helper whose
    search is over runs no more of its repeats."""
    best = min(workload() for _ in range(repeats))
    if not (hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")):
        return Reference(best, True)
    stop = time.monotonic() + SEARCH_S
    over = threading.Event()             # this search is over
    complete = True
    try:
        for cores in _search_order(_process_cpus()):
            answered = pinnable = False
            for cpu in cores:
                left = stop - time.monotonic()
                with _running_lock:
                    busy = bool(_running)
                if busy or not left > 0.0:
                    pinnable = True        # out of time, or a helper still runs, before this class was tried
                    break
                out: list = []
                pinned: list = []
                token = object()

                def timed(cpu=cpu, out=out, pinned=pinned, token=token):
                    found = None
                    try:
                        try:
                            os.sched_setaffinity(0, {cpu})          # this helper thread only
                        except OSError:
                            return                                 # not this core: the class's next one
                        pinned.append(cpu)
                        times = []
                        for _ in range(repeats):
                            if over.is_set():
                                return                             # its search ended without it
                            times.append(workload())
                        found = min(times)
                    finally:
                        with _running_lock:                        # its answer and its end, at once
                            if found is not None:
                                out.append(found)
                            _running.discard(token)

                helper = threading.Thread(target=timed, daemon=True, name="core-speed-probe")
                with _running_lock:
                    _running.add(token)
                try:
                    helper.start()
                except BaseException:
                    with _running_lock:
                        _running.discard(token)
                    raise
                helper.join(min(PER_CORE_S, left))
                with _running_lock:
                    running = token in _running
                if pinned or running:
                    # this class's core: its answer, if it came in time (a late or
                    # failed one leaves the class open); never another of its cores
                    pinnable = True
                    if out:
                        best = min(best, out[0])
                        answered = True
                    break
            if pinnable and not answered:
                complete = False
    finally:
        over.set()
    return Reference(best, complete)


def fastest_seconds(workload: Callable[[], float], repeats: int = 3) -> float:
    """`fastest(workload, repeats).seconds`."""
    return fastest(workload, repeats).seconds
