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
caller must not keep it (it can only overstate the reference). Without
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

#: stop starting helpers after this long (s, wall): the GB10's five capacity
#: classes at PER_CORE_S each
SEARCH_S = 1.0
#: how long to wait for one core's helper (s, wall): a busy core may not give
#: the helper its turn soon (the #399 review's probe: 60 ms)
PER_CORE_S = 0.2
#: where Linux publishes each core's capacity (1024 = the host's fastest)
SYSFS_CPU = "/sys/devices/system/cpu"


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
    set (a class's next core only when this one cannot be pinned)."""
    best = min(workload() for _ in range(repeats))
    if not (hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")):
        return Reference(best, True)
    stop = time.monotonic() + SEARCH_S
    complete = True
    for cores in _search_order(_process_cpus()):
        answered = pinnable = False
        for cpu in cores:
            left = stop - time.monotonic()
            if not left > 0.0:
                pinnable = True        # out of time before this class was tried
                break
            out: list = []
            pinned: list = []

            def timed(cpu=cpu, out=out, pinned=pinned):
                try:
                    os.sched_setaffinity(0, {cpu})          # this helper thread only
                except OSError:
                    return                                 # not this core: the class's next one
                pinned.append(cpu)
                out.append(min(workload() for _ in range(repeats)))

            helper = threading.Thread(target=timed, daemon=True, name="core-speed-probe")
            helper.start()
            helper.join(min(PER_CORE_S, left))
            if out:
                best = min(best, out[0])
                answered = pinnable = True
                break
            if pinned or helper.is_alive():
                pinnable = True        # no answer in time (or its workload failed): this class stays open
                break
        if pinnable and not answered:
            complete = False
    return Reference(best, complete)


def fastest_seconds(workload: Callable[[], float], repeats: int = 3) -> float:
    """`fastest(workload, repeats).seconds`."""
    return fastest(workload, repeats).seconds
