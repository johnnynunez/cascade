"""Wall scheduling for software-only publishers; never a physics clock."""
from contextlib import contextmanager
import gc
import math
import time

import pytest

NORMAL_WALL_INTERVAL_S = .02


@contextmanager
def quiescent_cyclic_gc():
    """Keep unrelated suite heap collection outside a bounded healthy episode.

    Cyclic GC holds the GIL and can stall both this process's software
    publisher and its RPC server. Readers must still reject a real stale or
    delayed response; this helper changes neither clocks nor those limits.
    Reference counting remains active, and the previous GC mode is restored.
    """
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()


@pytest.fixture
def healthy_episode_gc():
    with quiescent_cyclic_gc():
        yield


def scheduled_tick_steps(halt, *, first_step, wall_interval_s, wait=None,
                         clock=time.monotonic, max_backlog_steps=100):
    """Yield every due step, including catch-up after a delayed wakeup.

    A relative wait after each publication accumulates publication cost and OS
    wakeup delays. Absolute deadlines preserve the fixture's intended rate.
    Each yielded step still needs its own control_at and publish call. Excess
    debt is a fixture error, rather than an unbounded burst or skipped solves.
    ``wait`` and ``clock`` are injectable only here, for scheduler regressions.
    """
    assert math.isfinite(wall_interval_s) and wall_interval_s > 0
    assert isinstance(max_backlog_steps, int) and max_backlog_steps > 0
    wait = halt.wait if wait is None else wait
    origin, emitted = clock(), 0
    while not halt.is_set():
        deadline = origin + emitted * wall_interval_s
        now = clock()
        if now < deadline:
            wait(deadline - now)
            continue
        backlog = math.floor((now - deadline) / wall_interval_s) + 1
        assert backlog <= max_backlog_steps, f"synthetic tick backlog: {backlog} steps"
        yield first_step + emitted
        emitted += 1
