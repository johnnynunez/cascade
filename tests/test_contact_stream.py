"""The contact monitor inside the streaming executors (ArmBase, MockArm).

What a trip must do on the arm, tested on two backends:

* MockArm (unpaced, logical time) with `block_joint`: the measured joint
  stops following the command, the way the wrist did against the clutter.
* SimArm: ArmBase's real paced loop and settle phase on a fake clock, with
  per-joint PD dynamics using the reBot RS MIT gains (rebotarm_rs.yaml:
  kp 50/150/150/50/50/50, kd 3/10/10/5/4/4, no velocity feedforward) and
  8 ms single-joint feedback reads like the RS driver's mechPos param reads.

On a trip: no further streamed target, the arm is commanded to hold its
CURRENT MEASURED pose (never the command it could not reach), and
ContactDetected propagates. Lost feedback stops the stream and holds the
last command (there is no measurement to hold).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control import arm_base
from cascade.control.arm_base import ArmBase
from cascade.control.mock_arm import MockArm
from cascade.safety.contact import ContactMonitor, ContactMonitorConfig
from cascade.types import ContactDetected, RobotState, SafetyViolation

KP = np.array([50.0, 150.0, 150.0, 50.0, 50.0, 50.0])
KD = np.array([3.0, 10.0, 10.0, 5.0, 4.0, 4.0])
INERTIA = np.array([0.15, 0.20, 0.08, 0.01, 0.01, 0.005])   # kg m^2, guessed
HOME = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])


def monitor(**kw) -> ContactMonitor:
    base = {"enabled": True, "mode": "enforce", "track_tol_rad": 0.10, "lag_s": 0.25,
            "debounce_samples": 2, "max_feedback_gap_s": 0.35}
    base.update(kw)
    return ContactMonitor(ContactMonitorConfig.from_config(base), label="test")


# ── MockArm ──────────────────────────────────────────────────────────────


def mock_arm():
    arm = MockArm(Cfg({"n_joints": 6, "home_q": HOME.tolist()}))
    arm.connect()
    return arm


def test_mock_without_a_monitor_streams_exactly_as_before():
    arm = mock_arm()
    assert arm.stream_to(HOME + 0.3, 1.0) is True
    assert len(arm.commands) == 50
    np.testing.assert_allclose(arm.get_state().q, HOME + 0.3)


def test_mock_free_motion_never_trips_and_reports():
    arm = mock_arm()
    m = monitor()
    assert arm.stream_to(HOME + 0.3, 1.0, contact_monitor=m) is True
    s = m.summary()
    assert m.tripped is None and min(s["samples"]) >= 50 and s["commands"] == 50
    assert max(s["max_track_err_rad"]) < 1e-9


def test_mock_blocked_joint_trips_holds_the_measured_pose_and_stops_streaming():
    arm = mock_arm()
    target = HOME + np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0])
    arm.block_joint(2, at=HOME[2] + 0.2)             # the elbow meets something
    m = monitor()
    with pytest.raises(ContactDetected, match="contact detected at joint 3") as exc:
        arm.stream_to(target, 2.0, contact_monitor=m)
    held = arm.commands[-1]
    measured = arm.get_state().q
    # the hold IS the measured pose; joint 3 was never snapped to its command
    np.testing.assert_allclose(held, measured)
    assert held[2] == pytest.approx(HOME[2] + 0.2)
    streamed = arm.commands[:-1]
    assert max(c[2] for c in streamed) > held[2] + 0.1     # the command had run ahead
    assert len(streamed) < 100                             # stopped long before the end
    assert m.hold.startswith("measured")
    assert exc.value.event["joint"] == 3 and exc.value.event["hold_q"] == pytest.approx(list(held))


@pytest.mark.parametrize("bad_read", ["raises", "disagrees", "nan"])
def test_no_trustworthy_measured_pose_means_no_hold_command(bad_read):
    """The hold is only sent for a fresh reading that agrees with what the
    monitor just saw; otherwise the motors keep the last streamed target
    (commanding a wrong 'measured' pose could swing the arm)."""
    class Flaky(MockArm):
        def read_positions_for_monitor(self):
            if bad_read == "raises":
                raise RuntimeError("ack timeout")
            q = self.get_state().q.copy()
            q[0] = np.nan if bad_read == "nan" else q[0] + 1.0   # plausible, wrong
            return q
    arm = Flaky(Cfg({"n_joints": 6, "home_q": HOME.tolist()}))
    arm.connect()
    arm.block_joint(2)
    m = monitor()
    with pytest.raises(ContactDetected):
        arm.stream_to(HOME + np.array([0, 0, 0.6, 0, 0, 0]), 2.0, contact_monitor=m)
    assert m.hold.startswith("last command (measured pose unavailable")
    last = arm.commands[-1]
    assert last[0] == HOME[0] and last[2] > HOME[2]          # a streamed tick, no hold
    assert "hold_q" not in (m.tripped or {})


def test_mock_blocked_joint_trips_on_a_joint_path_too():
    arm = mock_arm()
    arm.block_joint(1)
    path = [HOME + np.array([0, 0.2, 0, 0, 0, 0]), HOME + np.array([0, 0.5, 0.1, 0, 0, 0])]
    with pytest.raises(ContactDetected, match="joint 2"):
        arm.stream_path(path, 2.0, contact_monitor=monitor())
    np.testing.assert_allclose(arm.commands[-1], arm.get_state().q)


def test_mock_trip_latency_is_bounded_in_stream_time():
    """50 Hz, every joint sampled every tick: the stall rule fires once one
    stall window (0.2 s) has passed since the block, plus one debounce tick."""
    arm = mock_arm()
    target = HOME + np.array([0.8, 0, 0, 0, 0, 0])
    m = monitor()
    blocked_at = {}
    send = arm.send_joint_target

    def send_and_block(q):
        send(q)
        if len(arm.commands) == 40 and not blocked_at:
            arm.block_joint(0)
            blocked_at["tick"] = 40
    arm.send_joint_target = send_and_block
    with pytest.raises(ContactDetected):
        arm.stream_to(target, 2.0, contact_monitor=m)
    ticks_after = (len(arm.commands) - 1) - blocked_at["tick"]
    assert ticks_after * 0.02 <= 0.30, ticks_after


# ── SimArm: the real paced loop on a fake clock ──────────────────────────


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.arm = None

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.advance(max(0.0, float(s)))

    def advance(self, s):
        if self.arm is not None:
            self.arm.integrate(s)
        self.t += s


class SimArm(ArmBase):
    """Six PD joints integrated on the fake clock. Reads cost bus time."""

    def __init__(self, clock, q0=HOME, inertia=INERTIA, read_s=0.008, offset=None, noise=0.0,
                 per_joint=True):
        self.n_joints = 6
        self.clock = clock
        clock.arm = self
        self.q = np.asarray(q0, float).copy()
        self.dq = np.zeros(6)
        self.target = self.q.copy()
        self.inertia = np.asarray(inertia, float)
        self.read_s = read_s
        self.offset = np.zeros(6) if offset is None else np.asarray(offset, float)
        self.noise = noise
        self.rng = np.random.default_rng(1)
        self.walls: dict[int, float] = {}          # joint -> position it cannot pass
        self.wall_hit: dict[int, float] = {}       # joint -> clock time it hit the wall
        self.fail_reads = lambda j: False
        self.commands: list[np.ndarray] = []
        self.per_joint = per_joint
        self.settle_tol = 0.05
        self.settle_timeout_s = 2.0

    def integrate(self, s):
        h = 0.0005
        steps = max(1, round(s / h))
        h = s / steps
        for _ in range(steps):
            tau = KP * (self.target - self.q) - KD * self.dq
            self.dq += h * tau / self.inertia
            self.q += h * self.dq
            for j, wall in self.walls.items():
                start = self.wall_from[j]
                if (wall - start) * (self.q[j] - wall) > 0:   # went past it
                    self.q[j], self.dq[j] = wall, 0.0
                    self.wall_hit.setdefault(j, self.clock.t)

    def block(self, j, at):
        self.walls[j] = float(at)
        self.wall_from = getattr(self, "wall_from", {})
        self.wall_from[j] = float(self.q[j])

    def _measure(self, j):
        n = self.rng.normal(0, self.noise) if self.noise else 0.0
        return float(self.q[j] + self.offset[j] + n)

    # ArmBase surface
    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_state(self):
        self.clock.advance(6 * self.read_s)
        return RobotState(q=np.array([self._measure(j) for j in range(6)]))

    def send_joint_target(self, q):
        self.clock.advance(0.001)
        self.target = np.asarray(q, float).copy()
        self.commands.append(self.target.copy())

    def set_gripper(self, pos, effort=1.0):
        pass

    def stop(self):
        pass

    def contact_joint_reader(self):
        if not self.per_joint:
            return None

        def read(j, timeout_s):
            self.clock.advance(self.read_s)
            if self.fail_reads(j):
                raise RuntimeError("ack timeout")
            return self._measure(j)
        return read

    def read_positions_for_monitor(self):
        return self.get_state().q


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(arm_base, "time", types.SimpleNamespace(monotonic=c.monotonic, sleep=c.sleep))
    return c


def test_paced_free_motion_with_real_pd_lag_never_trips(clock):
    arm = SimArm(clock, offset=[0.0, -0.013, 0.013, 0.0, 0.0, 0.0], noise=0.001)
    for target, dur in ((HOME + [0.8, -0.6, 0.5, 0.9, -0.9, 1.0], 2.0),   # stretched by the cap
                        (HOME, 1.6), (HOME + [0.05, 0, 0, 0, 0, 0.05], 0.5)):
        dq = np.max(np.abs(np.asarray(target) - arm.q))
        dur = max(dur, 1.875 * dq / (0.9 * 0.8))   # SafeArm's stretch at the 0.8 rad/s cap
        m = monitor()
        assert arm.stream_to(np.asarray(target), dur, contact_monitor=m) is True
        s = m.summary()
        assert m.tripped is None and s["would_trip"] is None, s
        assert max(s["max_track_err_rad"]) < 0.05, s
        assert s["reads"]["per_cycle"] >= 1


@pytest.mark.parametrize("joint", range(6))
def test_paced_blocked_joint_trips_within_a_quarter_second_with_shipped_defaults(clock, joint):
    """Default stall/debounce settings, RS gains, two 8 ms reads per 20 ms
    cycle (each joint every 60 ms): from the instant the joint hits the
    obstacle to the stream stopping."""
    arm = SimArm(clock)
    target = HOME.copy()
    target[joint] += 0.8 if joint != 1 else -0.8
    arm.block(joint, HOME[joint] + (target[joint] - HOME[joint]) * 0.5)
    m = ContactMonitor(ContactMonitorConfig.from_config({"enabled": True}), label="test")
    t0 = clock.t
    with pytest.raises(ContactDetected, match=f"contact detected at joint {joint + 1}"):
        arm.stream_to(target, 2.1, contact_monitor=m)
    hit = arm.wall_hit[joint]
    stopped = t0 + m.tripped["t_s"]
    assert 0.0 < stopped - hit <= 0.25, stopped - hit
    np.testing.assert_allclose(arm.commands[-1], arm.q, atol=0.01)   # held where it is


def test_paced_blocked_joint_trips_fast_and_holds_measured(clock):
    arm = SimArm(clock)
    target = HOME + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target[0] = 0.9
    arm.block(0, 0.35)                    # the base joint meets clutter at 0.35 rad
    m = monitor()
    t0 = clock.t
    with pytest.raises(ContactDetected, match="contact detected at joint 1") as exc:
        arm.stream_to(target, 2.5, contact_monitor=m)
    t_trip = t0 + exc.value.event["t_s"]
    assert t_trip - arm.wall_hit[0] < 0.45, t_trip - arm.wall_hit[0]
    held = arm.commands[-1]
    assert held[0] == pytest.approx(0.35, abs=0.01)        # where it is, not 0.9
    np.testing.assert_allclose(held[1:], arm.q[1:], atol=0.02)
    assert m.hold.startswith("measured")


def test_paced_block_during_settle_trips_instead_of_pressing_until_timeout(clock):
    """The incident's last 2 s: the stream had ended, the arm pressed against
    the obstacle until settle_timeout_s. Now the monitor watches the settle."""
    arm = SimArm(clock)
    target = HOME.copy()
    target[2] = 1.45
    arm.block(2, 1.30)
    m = monitor(stall_min_travel_rad=None)     # band only
    t_start = clock.t
    with pytest.raises(ContactDetected, match="joint 3"):
        arm.stream_to(target, 0.6, contact_monitor=m)
    assert clock.t - t_start < 0.6 + 1.0     # well before the 2 s settle timeout
    assert arm.commands[-1][2] == pytest.approx(1.30, abs=0.01)


def test_paced_lost_feedback_stops_and_holds_the_last_command(clock):
    arm = SimArm(clock)
    target = HOME + np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0])
    t0 = clock.t
    arm.fail_reads = lambda j: j == 4 and clock.t - t0 > 0.5
    m = monitor()
    with pytest.raises(SafetyViolation, match="lost joint 5 feedback") as exc:
        arm.stream_to(target, 2.0, contact_monitor=m)
    assert not isinstance(exc.value, ContactDetected)
    n = len(arm.commands)
    assert n < 100 and clock.t - t0 < 0.5 + 0.35 + 0.1
    # no hold-to-measured: the last command is a streamed tick, not a read
    assert arm.commands[-1][3] > HOME[3] and m.hold.startswith("last command")


def test_paced_reads_stay_inside_the_cycle_budget(clock):
    arm = SimArm(clock, read_s=0.008)
    m = monitor()
    arm.stream_to(HOME + 0.2, 1.0, contact_monitor=m)
    reads = m.summary()["reads"]
    assert 1.0 <= reads["per_cycle"] <= 2.0        # 8 ms reads in a 20 ms cycle
    assert reads["overruns"] <= reads["cycles"] // 10


def test_paced_full_state_backend_samples_every_cycle(clock):
    arm = SimArm(clock, read_s=0.001, per_joint=False)
    m = monitor()
    arm.stream_to(HOME + 0.2, 1.0, contact_monitor=m)
    assert min(m.summary()["samples"]) >= 50


def test_without_a_monitor_the_paced_loop_reads_nothing_mid_stream(clock):
    arm = SimArm(clock)
    reads = []
    arm.contact_joint_reader = lambda: (lambda j, t: reads.append(j))
    arm.stream_to(HOME + 0.2, 1.0)
    assert reads == []
