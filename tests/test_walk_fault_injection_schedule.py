"""B67: the geometric-fault walk test must not depend on worker/reader scheduling.

macOS CI (main e96a3a2) failed
``test_walk_distance_control.py::test_distance_control_stops_on_geometric_fault[lateral]``
with ``execution_ok: True``. Its ``Broken`` fixture injected the fault from the
third ``get_state`` read with ``controller_status == "active"`` while the mock's
free-running worker produced the motion. Reads that see no new step still count,
so the read count is not the robot's timeline.

The scheduling adversary below replaces the free-running producer with scripted
worker ticks taken just before chosen reads: every schedule is an interleaving
the real worker can produce. It only moves a mock whose ``auto_step`` is True,
which is exactly the motion source the flaky fixture depended on.

* ``lagging_worker`` (measured mechanism, B): two post-ACK reads see no new step,
  so the third active read is also the FIRST post-ACK sample, i.e. SafeBase's
  distance baseline. The fault becomes part of the baseline and never changes
  again; lateral/heading drift is integrated from that baseline, so the walk
  completes. A real-scheduler probe (timer slack on both threads, or on the
  worker only) produced only this class of miss (REPORT.md of B67).
* ``stalled_reader`` (A, never observed, still possible): the reader stalls for
  95 worker ticks after the baseline and the walk completes before a third
  active read exists.
* ``late_first_sample``: the first post-ACK read is 40 ticks late. The original
  fixture survives it; a fault keyed to simulated time WITH a free-running
  producer would not (the late baseline already carries the fault).

Kinematic fixtures prove no gait; every result stays ``unverified``.
"""
import math
import time
from dataclasses import replace

import pytest

from cascade.control.mock_base import MockMobileBase
import test_walk_distance_control as walk_tests
from test_walk_distance_control import _SteppedDistanceMock, configured

FAULTS = ["lateral", "heading", "low", "tilt"]

# Worker ticks taken just before post-ACK read i (1-based); every other read gets one tick.
SCHEDULES = {
    "nominal": {},
    "lagging_worker": {1: 0, 2: 0},
    "stalled_reader": {2: 95},
    "late_first_sample": {1: 40},
}


def _fault(state, fault):
    """The original test's geometric faults, applied to one snapshot."""
    if fault == "lateral":
        return replace(state, position_world=(state.position_world[0], .02, .2))
    if fault == "low":
        return replace(state, position_world=(state.position_world[0], 0., .04))
    angle = .2 if fault == "heading" else .7
    q = ((math.cos(angle/2), 0., 0., math.sin(angle/2)) if fault == "heading"
         else (math.cos(angle/2), math.sin(angle/2), 0., 0.))
    return replace(state, orientation_wxyz=q)


@pytest.fixture
def scheduled_worker(monkeypatch):
    """Install a scripted interleaving of worker ticks and reads on MockMobileBase."""
    original_get = MockMobileBase.get_state
    original_command = MockMobileBase.command_velocity

    def install(name):
        plan = SCHEDULES[name]
        trace = []

        def run(self):
            # Same wake-up and wall-lease expiry as the real worker, without motion:
            # motion ticks are taken by the schedule at chosen points between reads.
            while not self._shutdown.wait(min(self._dt, self._lease / 4)):
                with self._lock:
                    self._expire(time.monotonic())

        def command_velocity(self, command, *, generation):
            ack = original_command(self, command, generation=generation)
            if ack.get("accepted") is True:
                self.__dict__["_b67_post_ack_reads"] = 0
            return ack

        def get_state(self):
            index = self.__dict__.get("_b67_post_ack_reads")
            if index is not None:
                index += 1
                self.__dict__["_b67_post_ack_reads"] = index
            ticks = plan.get(index, 1) if self._auto_step else 0
            for _ in range(ticks):
                self.advance(min(self._dt, self._lease / 4))
            trace.append((index, ticks))
            return original_get(self)

        monkeypatch.setattr(MockMobileBase, "_run", run)
        monkeypatch.setattr(MockMobileBase, "command_velocity", command_velocity)
        monkeypatch.setattr(MockMobileBase, "get_state", get_state)
        return trace

    return install


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_geometric_fault_walk_test_holds_under_every_worker_schedule(scheduled_worker, schedule, fault):
    """Run the repository test body itself under a forced, legal interleaving.

    On main: ``lagging_worker`` fails lateral/heading with the CI signature
    (``execution_ok: True``) and ``stalled_reader`` fails all four faults.
    """
    trace = scheduled_worker(schedule)
    walk_tests.test_distance_control_stops_on_geometric_fault(fault)
    assert trace, "scheduling adversary was not installed"


class _OnsetFault(_SteppedDistanceMock):
    """Ten explicit .002 steps per read: post-ACK read k is post-ACK sample k.

    The fault is present from post-ACK sample ``onset`` on. Sample 1 is the
    distance baseline.
    """

    def __init__(self, fault, onset, **kwargs):
        super().__init__(**kwargs)
        self.fault, self.onset, self.post_ack_reads = fault, onset, None

    def command_velocity(self, command, *, generation):
        ack = super().command_velocity(command, generation=generation)
        if ack.get("accepted") is True:
            self.post_ack_reads = 0
        return ack

    def get_state(self):
        state = super().get_state()
        if self.post_ack_reads is None:
            return state
        self.post_ack_reads += 1
        return _fault(state, self.fault) if self.post_ack_reads >= self.onset else state


# B74: opt-in bound on the change from the last pre-ACK sample to that baseline.
_ADMISSION_BOUNDS = dict(max_admission_lateral_m=.005, max_admission_heading_rad=.04)


@pytest.mark.parametrize("admission", [False, True], ids=["flag_off", "flag_on"])
@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("onset", [1, 2])
def test_premise_only_a_fault_first_seen_at_the_post_ack_baseline_escapes_the_drift_veto(onset, fault, admission):
    """Pins the cause, not a desired property (flag off passes on main by design).

    SafeBase integrates lateral/heading drift from the first completed post-ACK
    sample (docs/MICRODUCK_DISTANCE_CANDIDATE.md, "Implemented contract"). A
    constant lateral/heading offset first seen AT that baseline is part of it and
    the walk completes; the same offset one sample later is vetoed. Posture
    (low/tilt) is checked on every sample, baseline included, so it is vetoed at
    either onset: this is why only lateral (and, equally exposed, heading) could
    flake. B74's opt-in ``distance_control.max_admission_lateral_m`` /
    ``max_admission_heading_rad`` bound the pre-baseline segment: flag on, the
    onset-1 offset is vetoed on the would-be baseline; flag off (the default) it
    is still absorbed.
    """
    raw = _OnsetFault(fault, onset, wall_lease_s=2., dt_s=.002, auto_step=False)
    safe = configured(raw, **(_ADMISSION_BOUNDS if admission else {}))
    safe.connect()
    try:
        result = safe.walk_distance(.02)
        samples = result["measured"]["samples"]
        assert all(b["step"] - a["step"] == 10 for a, b in zip(samples, samples[1:]))
        baseline = result["distance_baseline"]
        if onset == 1 and fault in ("lateral", "heading") and not admission:
            assert result["execution_ok"], result
            assert result["ok"] is False and result["outcome"] == "unverified"
            assert baseline["step"] == samples[2]["step"] and baseline == samples[2]  # faulted baseline
            assert (baseline["position_world"][1] == .02 if fault == "lateral"
                    else baseline["orientation_wxyz"][3] == math.sin(.1))
            assert result["measured_lateral_m" if fault == "lateral" else "measured_heading_rad"] == 0.
            assert abs(result["measured_distance_m"] - .02) <= .002
            assert not raw.get_state().latched
        else:
            expected = ("posture bound" if fault in ("low", "tilt") else
                        "admission change exceeded" if onset == 1 else "lateral/heading drift exceeded")
            assert not result["execution_ok"] and expected in result["error"], result
            assert len(samples) == 2 + onset  # vetoed at the first faulted sample
            # Onset 1 posture: the veto fires on the would-be baseline itself.
            assert (baseline is None) if onset == 1 else (baseline == samples[2])
            assert raw.get_state().latched
    finally:
        safe.disconnect()
