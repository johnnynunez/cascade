"""How fast the calling thread's CPU core is next to the host's fastest.

Hosts with two kinds of core run the same Python work at different speeds: the
GB10 (DGX Spark) has ten Cortex-X925 and ten Cortex-A725 cores, and a planner
moved to an A725 takes 2.5-3x the CPU time it takes on an X925. A CPU-time
budget is blind to contention but not to that. `fastest_seconds` gives the
reference: a workload timed on every core this process may use, each time by a
helper thread pinned to that core (the caller's own affinity is never
touched), the least of them. Linux only (per-thread affinity); elsewhere the
caller's own core is the reference, i.e. no core counts as slower.

This lives outside `cascade.skills`: a helper thread here must run only the
workload it is given, which must read no runtime state (a skill's arm frame is
thread-local; tests/test_arm_frame_context.py keeps skill modules free of
thread starts).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

#: stop looking at more cores after this long (s, wall): a busy core may not
#: give the helper its turn soon
SEARCH_S = 0.25
#: how long to wait for one core's helper (s, wall)
PER_CORE_S = 0.05


def fastest_seconds(workload: Callable[[], float], repeats: int = 3) -> float:
    """The least of `workload()` (CPU seconds of a fixed slice of work, timed
    by the workload itself on its own thread) over the caller's core and
    every core this process may run on, `repeats` times each."""
    best = min(workload() for _ in range(repeats))
    if not (hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")):
        return best
    stop = time.monotonic() + SEARCH_S
    for cpu in sorted(os.sched_getaffinity(0)):
        if time.monotonic() > stop:
            break
        out: list = []

        def timed(cpu=cpu, out=out):
            try:
                os.sched_setaffinity(0, {cpu})          # this helper thread only
                out.append(min(workload() for _ in range(repeats)))
            except OSError:
                pass

        helper = threading.Thread(target=timed, daemon=True, name="core-speed-probe")
        helper.start()
        helper.join(PER_CORE_S)
        if out:
            best = min(best, out[0])
    return best
