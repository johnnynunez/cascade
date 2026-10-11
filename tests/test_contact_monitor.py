"""Motor-feedback contact monitor: the pure detection logic.

Rig incident 2026-10-11: a side-approach grasp drove the wrist camera into
clutter beside the target and the arm PRESSED against it for ~8 s, until the
end-of-move settle check reported "did not settle at grasp pose". Nothing
compared the measured joints with the command while the motion streamed.

`cascade.safety.contact.ContactMonitor` is that comparison, independent of
any camera. These tests drive it with synthetic streams on a simulated clock:
50 Hz commands along a min-jerk profile, measured joints that follow the
command with a pure delay (the PD lag of the RS MIT loop is kd/kp = 60-100 ms
with zero velocity feedforward), one or more single-joint feedback reads per
cycle (the RS driver's mechPos param reads, ~8 ms each), and the settle
phase after the last tick.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.control.motion_profile import min_jerk
from cascade.safety.contact import ContactMonitor, ContactMonitorConfig
from cascade.types import ContactDetected, SafetyViolation

N = 6
DT = 0.02


def cfg(**kw) -> ContactMonitorConfig:
    base = {"enabled": True, "mode": "enforce", "track_tol_rad": 0.10, "lag_s": 0.25,
            "debounce_samples": 2, "stall_window_s": 0.2, "stall_min_travel_rad": 0.04,
            "stall_ratio": 0.2, "max_feedback_gap_s": 0.35}
    base.update(kw)
    return ContactMonitorConfig.from_config(base)


def profile(q0, q1, T):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)

    def at(t):
        return q0 + (q1 - q0) * min_jerk(min(max(t / T, 0.0), 1.0))
    return at


def run(monitor, command, measured, T, *, reads_per_cycle=1, read_s=0.008,
        settle_s=0.6, settle_period=0.1, fail=None):
    """Stream `command(t)` for T s at 50 Hz, then settle.

    `measured(t, j)` is the arm; `fail(t, j)` -> True makes that read raise.
    Returns the simulated time of the last processed sample (the trip time
    when the monitor raised -- the exception propagates to the caller after
    `monitor.t_last` is set by the monitor itself)."""
    steps = max(2, int(T / DT))
    monitor.begin(command(0.0))
    for i in range(1, steps + 1):
        t_cmd = (i - 1) * DT
        monitor.commanded(t_cmd, command(i * T / steps))   # ArmBase.stream_to's tick i
        t = t_cmd + 0.003
        for _ in range(reads_per_cycle):
            j = monitor.next_joint()
            t += read_s
            if fail is not None and fail(t, j):
                monitor.read_failed(t, j, RuntimeError("ack timeout"))
            else:
                monitor.observe(t, j, measured(t, j))
        monitor.check_feedback(t)
    t = steps * DT
    end = t + settle_s
    while t < end:
        t += settle_period
        q = np.array([measured(t, j) for j in range(N)])
        monitor.observe_all(t, q)
        monitor.check_feedback(t)
    return t


def delayed(cmd, lag, offset=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lags = np.broadcast_to(np.asarray(lag, float), (N,))
    offs = np.broadcast_to(np.asarray(offset, float), (N,))

    def m(t, j):
        return float(cmd(t - lags[j])[j] + offs[j] + (rng.normal(0, noise) if noise else 0.0))
    return m


def blocked(base, joint, t_block):
    """`base` until t_block; joint `joint` then stays where it was."""
    frozen = {}

    def m(t, j):
        if j == joint and t >= t_block:
            if j not in frozen:
                frozen[j] = base(t_block, j)
            return frozen[j]
        return base(t, j)
    return m


# ── config ───────────────────────────────────────────────────────────────


def test_absent_block_means_no_monitor():
    assert ContactMonitorConfig.from_config(None) is None


def test_block_without_enabled_stays_off():
    c = ContactMonitorConfig.from_config({"track_tol_rad": 0.1})
    assert c is not None and not c.enabled


def test_scalar_and_per_joint_tolerances():
    assert cfg(track_tol_rad=0.1).tolerances(6).tolist() == [0.1] * 6
    per = [0.1, 0.2, 0.2, 0.15, 0.12, 0.12]
    assert cfg(track_tol_rad=per).tolerances(6).tolist() == per
    with pytest.raises(ValueError, match="6 joints"):
        cfg(track_tol_rad=per[:5]).tolerances(6)


@pytest.mark.parametrize("bad", [
    {"mode": "audit"},
    {"track_tol_rad": 0.0},
    {"track_tol_rad": -0.1},
    {"track_tol_rad": float("nan")},
    {"lag_s": -0.1},
    {"debounce_samples": 0},
    {"debounce_samples": 1.5},
    {"stall_ratio": 1.5},
    {"max_feedback_gap_s": 0.0},
    {"contact_zone_m": -0.01},
    {"enabled": "yes"},
    {"unknown_key": 1},
])
def test_malformed_block_is_a_config_error(bad):
    with pytest.raises(ValueError):
        cfg(**bad)


# ── trips ────────────────────────────────────────────────────────────────


def test_blocked_joint_trips_within_the_debounce_window():
    cmd = profile(np.zeros(N), np.full(N, 1.0), 2.0)
    m = ContactMonitor(cfg(), label="move_joints")
    t_block = 0.8
    with pytest.raises(ContactDetected) as exc:
        run(m, cmd, blocked(delayed(cmd, 0.08), 2, t_block), 2.0)
    t_trip = m.t_last
    assert t_trip > t_block
    # one joint read per 20 ms cycle = each joint every 120 ms; the stall rule
    # needs one stall window (0.2 s) plus the debounce (2 samples)
    assert t_trip - t_block < 0.5, t_trip - t_block
    msg = str(exc.value)
    assert msg.startswith("contact detected at joint 3: tracking error ")
    assert "rad for " in msg and " ms -- something is blocking the arm" in msg
    assert isinstance(exc.value, SafetyViolation)
    assert m.tripped is not None and m.tripped["joint"] == 3


def test_block_during_settle_trips_on_the_band_alone():
    """The command has stopped; the arm is held short of it (the incident:
    pressed for the whole settle window). Stall needs a moving command, so
    this is the tracking band, after lag_s, plus the debounce."""
    cmd = profile(np.zeros(N), np.full(N, 1.0), 2.0)
    m = ContactMonitor(cfg(), label="move_joints")

    def short_of_target(t, j):
        v = float(cmd(t - 0.08)[j])
        return min(v, 0.85) if j == 4 else v
    with pytest.raises(ContactDetected, match="joint 5"):
        run(m, cmd, short_of_target, 2.0, settle_s=2.0)
    assert m.tripped["phase"] in ("stream", "settle")


def test_stall_trips_before_the_band_would():
    """A huge band tolerance leaves only the stall rule: the joint stopped
    advancing while the command kept moving at speed."""
    cmd = profile(np.zeros(N), np.full(N, 1.2), 2.0)
    m = ContactMonitor(cfg(track_tol_rad=1.0), label="move_joints")
    with pytest.raises(ContactDetected, match="stalled") as exc:
        run(m, cmd, blocked(delayed(cmd, 0.08), 1, 0.7), 2.0, reads_per_cycle=2)
    assert m.t_last - 0.7 < 0.4
    assert "contact detected at joint 2" in str(exc.value)


def test_a_single_bad_sample_is_debounced():
    cmd = profile(np.zeros(N), np.full(N, 0.6), 2.0)
    base = delayed(cmd, 0.08)
    spiked = set()

    def spike(t, j):
        if j == 0 and 1.0 <= t and not spiked:
            spiked.add(t)
            return base(t, j) + 0.25   # a bad sample, still a plausible reading
        return base(t, j)
    m = ContactMonitor(cfg(), label="move_joints")
    run(m, cmd, spike, 2.0)
    assert m.tripped is None
    assert m.summary()["max_track_err_rad"][0] > 0.10


def test_debounce_one_trips_on_the_first_bad_sample():
    cmd = profile(np.zeros(N), np.full(N, 1.0), 2.0)
    one = ContactMonitor(cfg(debounce_samples=1), label="m")
    with pytest.raises(ContactDetected):
        run(one, cmd, blocked(delayed(cmd, 0.08), 2, 0.8), 2.0)
    two = ContactMonitor(cfg(debounce_samples=2), label="m")
    with pytest.raises(ContactDetected):
        run(two, cmd, blocked(delayed(cmd, 0.08), 2, 0.8), 2.0)
    assert one.t_last < two.t_last


# ── never trips ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("T,dq,lag", [
    (2.0, 1.0, 0.0), (2.0, 1.0, 0.1), (1.2, 0.6, 0.25),  # peak 0.94 rad/s
    (4.0, 1.5, 0.2), (0.5, 0.05, 0.25), (3.0, -1.4, 0.15),
])
def test_lagged_tracking_never_trips(T, dq, lag):
    cmd = profile(np.zeros(N), np.full(N, dq), T)
    lags = np.linspace(0.0, lag, N)                # every joint its own lag
    m = ContactMonitor(cfg(), label="move_joints")
    run(m, cmd, delayed(cmd, lags, offset=0.013, noise=0.002), T, reads_per_cycle=2)
    s = m.summary()
    assert m.tripped is None and s["would_trip"] is None
    assert max(s["max_track_err_rad"]) < 0.05, s


def test_joint_reversal_mid_path_never_stalls():
    """A Cartesian line (stream_path) can reverse a joint mid-motion; a lag
    shorter than lag_s must not read as 'stopped while commanded'."""
    def cmd(t):
        s = min_jerk(min(max(t / 2.0, 0.0), 1.0))
        return np.full(N, 0.8 * np.sin(np.pi * s))  # out and back
    m = ContactMonitor(cfg(), label="move_joint_path")
    run(m, cmd, delayed(cmd, np.linspace(0.0, 0.1, N)), 2.0, reads_per_cycle=2)
    assert m.tripped is None and m.summary()["would_trip"] is None


@pytest.mark.parametrize("lags", [np.linspace(0.0, 0.25, N), np.linspace(0.02, 0.23, N),
                                  np.linspace(0.04, 0.24, N)])
@pytest.mark.parametrize("reads,read_s", [(6, 0.003), (2, 0.008), (1, 0.008)])
def test_sharp_reversal_never_stalls(lags, reads, read_s):
    """Out at the 0.8 rad/s cap and straight back: across the turn the
    command advanced one way under a short lag and the other way under a
    long one. Only a command that moved the SAME way under every lag in
    [0, lag_s] can call a still joint stalled."""
    def cmd(t):
        t = min(max(t, 0.0), 1.2)
        return np.full(N, 0.8 * t if t <= 0.6 else 0.48 - 0.8 * (t - 0.6))
    m = ContactMonitor(cfg(), label="move_joint_path")
    run(m, cmd, delayed(cmd, lags), 1.2, reads_per_cycle=reads, read_s=read_s)
    assert m.tripped is None


def test_slow_drift_under_the_threshold_never_trips():
    cmd = profile(np.zeros(N), np.full(N, 0.8), 2.0)
    base = delayed(cmd, 0.08)

    def drifting(t, j):        # e.g. a payload sagging j2/j3: 0.08 rad at 2.6 s
        return base(t, j) - (0.08 * min(t / 2.6, 1.0) if j in (1, 2) else 0.0)
    m = ContactMonitor(cfg(), label="move_joints")
    run(m, cmd, drifting, 2.0)
    assert m.tripped is None


# ── feedback loss ────────────────────────────────────────────────────────


def test_lost_joint_feedback_mid_stream_fails_safe():
    cmd = profile(np.zeros(N), np.full(N, 1.0), 2.0)
    m = ContactMonitor(cfg(), label="move_joints")
    with pytest.raises(SafetyViolation) as exc:
        run(m, cmd, delayed(cmd, 0.08), 2.0, fail=lambda t, j: j == 1 and t > 0.5)
    assert not isinstance(exc.value, ContactDetected)
    assert "lost joint 2 feedback" in str(exc.value)
    assert m.t_last - 0.5 <= 0.35 + 0.15


def test_one_failed_read_is_tolerated():
    cmd = profile(np.zeros(N), np.full(N, 1.0), 2.0)
    m = ContactMonitor(cfg(), label="move_joints")
    once = []

    def fail(t, j):
        if j == 3 and t > 0.9 and not once:
            once.append(t)
            return True
        return False
    run(m, cmd, delayed(cmd, 0.08), 2.0, fail=fail)
    assert once and m.summary()["read_failures"] == 1 and m.tripped is None


def test_an_impossible_jump_is_a_read_failure_not_a_contact():
    cmd = profile(np.zeros(N), np.full(N, 0.5), 2.0)
    base = delayed(cmd, 0.08)
    hits = []

    def garbage(t, j):
        if j == 0 and 1.0 <= t and len(hits) < 1:
            hits.append(t)
            return base(t, j) + 2.0               # cannot be reached in 120 ms
        return base(t, j)
    m = ContactMonitor(cfg(), label="m")
    run(m, cmd, garbage, 2.0)
    assert hits and m.tripped is None and m.summary()["read_failures"] == 1


# ── measure mode ─────────────────────────────────────────────────────────


def test_measure_mode_never_trips_and_reports_what_would_have():
    cmd = profile(np.zeros(N), np.full(N, 1.0), 2.0)
    m = ContactMonitor(cfg(mode="measure"), label="move_joints")
    run(m, cmd, blocked(delayed(cmd, 0.08), 2, 0.8), 2.0,
        fail=lambda t, j: j == 5 and t > 1.0)    # nor stops on lost feedback
    s = m.summary()
    assert m.tripped is None
    assert s["mode"] == "measure"
    assert s["would_trip"] and "contact detected at joint 3" in s["would_trip"]
    assert s["max_track_err_rad"][2] > 0.10
    assert len(s["max_track_err_rad"]) == N and len(s["samples"]) == N
    assert s["max_feedback_gap_s"] > 0.35


def test_summary_reports_effective_lag_for_calibration():
    cmd = profile(np.zeros(N), np.full(N, 1.0), 2.0)
    m = ContactMonitor(cfg(mode="measure"), label="move_joints")
    run(m, cmd, delayed(cmd, 0.1), 2.0, reads_per_cycle=6, read_s=0.003)
    lag = m.summary()["max_effective_lag_s"]
    assert all(0.06 <= v <= 0.16 for v in lag), lag
