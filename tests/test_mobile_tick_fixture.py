"""The software publisher preserves steps when host wakeups are delayed."""
import threading

import pytest

from mobile_tick_fixture import scheduled_tick_steps


class Clock:
    def __init__(self):
        self.now = 0.
        self.waits = []

    def __call__(self):
        return self.now

    def coalesced_wait(self, requested):
        self.waits.append(requested)
        self.now += max(requested, .14)


def test_coalesced_wakeup_preserves_each_step_and_cancel_interrupts_backlog():
    halt, clock = threading.Event(), Clock()
    ticks = scheduled_tick_steps(halt, first_step=7, wall_interval_s=.03,
                                 wait=clock.coalesced_wait, clock=clock)
    assert [next(ticks) for _ in range(5)] == [7, 8, 9, 10, 11]
    assert clock.now == pytest.approx(.14)
    assert clock.waits == pytest.approx([.03])
    # A new coalesced wakeup has four further due steps. Cancellation must
    # interrupt that debt, not replay the rest of it after a stop.
    assert next(ticks) == 12
    halt.set()
    assert list(ticks) == []


def test_publication_cost_does_not_get_added_to_each_interval():
    halt, clock = threading.Event(), Clock()

    def wait(requested):
        clock.waits.append(requested)
        clock.now += requested

    ticks = scheduled_tick_steps(halt, first_step=1, wall_interval_s=.03,
                                 wait=wait, clock=clock)
    for expected in range(1, 5):
        assert next(ticks) == expected
        clock.now += .02  # time spent publishing, outside the scheduler
    assert clock.waits == pytest.approx([.01, .01, .01])
    halt.set()


def test_excessive_backlog_fails_without_skipping_or_spinning():
    halt, clock = threading.Event(), Clock()
    ticks = scheduled_tick_steps(halt, first_step=1, wall_interval_s=.03,
                                 wait=clock.coalesced_wait, clock=clock,
                                 max_backlog_steps=3)
    assert next(ticks) == 1
    with pytest.raises(AssertionError, match="synthetic tick backlog"):
        next(ticks)
