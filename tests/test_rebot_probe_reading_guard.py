"""The RS bring-up probes never command a position they have not vetted.

Rig incident, 2026-10-10 (`scripts/sign_check_rebot_mb.py --joints 2,3`):
joint 2's +0.06 rad probe tracked fine (97 %); on the way back a mechPos
parameter read returned +2.3e18. The stall guard took that as "where the
joint actually is" and commanded the motor THERE -- MIT clamps it to the
motor's +4*pi end -- so the shoulder drove at full stiffness until the
operator cut power. And all seven motors had been registered as `rs-00`
while joints 1-3 are `rs-06` (the SDK's rebotarm_rs.yaml), so the MIT gains
were encoded on the wrong scale (about 10x stiffer than configured).

The contract pinned here, for both bring-up scripts:

* every reading is checked (finite, inside the motor's +-4*pi, inside the
  URDF limits +- a small tolerance) before it is used for anything;
* an implausible reading mid-motion stops the run and holds the LAST
  COMMANDED (already vetted) pose -- it is never commanded, and the way
  back never starts from a reading;
* every position sent to the probed joint stays inside the probe envelope
  (start +- delta), whatever the bus returns;
* an implausible reading at startup refuses before anything is energized;
* motors are registered with their per-joint model from the SDK config, and
  without that config the scripts refuse rather than guess one model for all.

Offline: motorbridge, the SDK config and the URDF limits are faked.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest
from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))

GARBAGE = 2.3354605539310961e18          # the value the rig returned
SDK = {1: ("rs-06", 50.0, 3.0), 2: ("rs-06", 150.0, 10.0), 3: ("rs-06", 150.0, 10.0),
       4: ("rs-00", 50.0, 5.0), 5: ("rs-00", 50.0, 4.0), 6: ("rs-00", 50.0, 4.0),
       7: ("rs-00", 50.0, 4.0)}


def _limits():
    return (np.array([-2.8, 0.0, 0.0, -1.69, -1.57, -3.14]),
            np.array([2.8, 3.14, 3.14, 1.79, 1.57, 3.14]))


# ── the shared guard ─────────────────────────────────────────────────────


def test_plausible_position():
    from cascade.control.robstride import MIT_P_MAX, plausible_position

    assert MIT_P_MAX == pytest.approx(4 * math.pi)
    assert plausible_position(0.5, 0.0, 3.14)
    assert plausible_position(-0.0009, 0.0, 3.14)        # resting on its stop
    assert plausible_position(3.149, -3.14, 3.14)        # the wrist before re-zeroing
    assert plausible_position(1.0)                        # no limits: the motor range only
    for bad in (None, float("nan"), float("inf"), -float("inf"), GARBAGE, 13.0, -13.0, True, "1"):
        assert not plausible_position(bad), bad
    assert not plausible_position(-0.2, 0.0, 3.14)
    assert not plausible_position(3.4, 0.0, 3.14)


def test_probe_envelope_clamps_and_refuses_non_finite():
    from cascade.control.robstride import ProbeEnvelope

    env = ProbeEnvelope.around(0.0, 0.06)
    assert (env.lo, env.hi) == (-0.06, 0.06)
    assert env.clamp(0.03) == 0.03
    assert env.clamp(GARBAGE) == 0.06 and env.clamp(-GARBAGE) == -0.06
    with pytest.raises(ValueError):
        env.clamp(float("nan"))
    span = ProbeEnvelope.spanning(0.5, 0.2)
    assert (span.lo, span.hi) == (0.2, 0.5)
    with pytest.raises(ValueError):
        ProbeEnvelope.around(0.0, float("inf"))


# ── fakes ────────────────────────────────────────────────────────────────


class Motor:
    def __init__(self, mid, log):
        self.mid, self.log = mid, log

    def ensure_mode(self, *a):
        self.log.append(("ensure_mode", self.mid))

    def clear_error(self):
        self.log.append(("clear_error", self.mid))

    def send_mit(self, pos, vel, kp, kd, tau):
        self.log.append(("send_mit", self.mid, float(pos), kp, kd))


def _fake_motorbridge(monkeypatch, log):
    class Mode:
        MIT = "mit"

    class Controller:
        def __init__(self, channel=None):
            pass

        def add_robstride_motor(self, motor_id, feedback_id, model):
            log.append(("add", motor_id, model))
            return Motor(motor_id, log)

        def enable_all(self):
            log.append(("enable_all", None))

        def disable_all(self):
            log.append(("disable_all", None))

        def close_bus(self):
            pass

        def close(self):
            pass

    mod = types.ModuleType("motorbridge")
    mod.Controller, mod.Mode = Controller, Mode
    monkeypatch.setitem(sys.modules, "motorbridge", mod)


class Bus:
    """mechPos as the motors would report it: each motor sits at its last
    MIT target (or its rest value before any), except where a test injects
    a garbage reply on the Nth read of one motor."""

    def __init__(self, log, rest, garbage_on=None):
        self.log, self.rest = log, dict(rest)
        self.reads = {}
        self.garbage_on = garbage_on or {}        # mid -> set of read indices

    def read(self, motor, param, timeout_ms=300):
        n = self.reads.get(motor.mid, 0)
        self.reads[motor.mid] = n + 1
        if n in self.garbage_on.get(motor.mid, ()):
            self.log.append(("read", motor.mid, GARBAGE))
            return GARBAGE
        sent = [e[2] for e in self.log if e[0] == "send_mit" and e[1] == motor.mid]
        return sent[-1] if sent else self.rest[motor.mid]


def _sent(log, mid):
    return [e[2] for e in log if e[0] == "send_mit" and e[1] == mid]


REST = {1: 0.03, 2: -0.0001, 3: -0.0001, 4: 0.0, 5: -0.008, 6: 0.0, 7: 0.0}


def _sign_check(monkeypatch, log, bus, argv):
    import sign_check_rebot_mb as sc

    _fake_motorbridge(monkeypatch, log)
    monkeypatch.setattr(sc, "_sdk_motors", lambda: dict(SDK))
    monkeypatch.setattr(sc, "_profile_gains", lambda mid: SDK[mid][1:])
    monkeypatch.setattr(sc, "_read", bus.read)
    monkeypatch.setattr(sc, "_urdf_local_limits", _limits)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["sign_check", *argv])
    return sc


# ── sign_check ───────────────────────────────────────────────────────────


def test_sign_check_registers_each_motor_with_its_own_model(monkeypatch):
    log = []
    sc = _sign_check(monkeypatch, log, Bus(log, REST), ["--joints", "2", "--dry-run"])
    assert sc.main() == 0
    assert {mid: model for ev, mid, model in (e for e in log if e[0] == "add")} == \
        {mid: SDK[mid][0] for mid in SDK}


def test_sign_check_refuses_without_the_sdk_models(monkeypatch):
    log = []
    sc = _sign_check(monkeypatch, log, Bus(log, REST), ["--joints", "2", "--yes"])
    monkeypatch.setattr(sc, "_sdk_motors", lambda: None)
    assert sc.main() != 0
    assert not any(e[0] in ("enable_all", "send_mit", "ensure_mode") for e in log)


def test_sign_check_refuses_an_implausible_reading_before_energizing(monkeypatch):
    log = []
    bus = Bus(log, REST, garbage_on={2: set(range(10))})     # every read of j2 is garbage
    sc = _sign_check(monkeypatch, log, bus, ["--joints", "2", "--yes"])
    assert sc.main() != 0
    assert not any(e[0] in ("enable_all", "send_mit") for e in log)


def test_the_incident_replayed_never_commands_the_garbage(monkeypatch):
    """j2 tracks its +probe, then a read on the way back is +2.3e18."""
    import sign_check_rebot_mb as sc0

    log = []
    # startup read (1) + energized drift read (1) + the +probe's checks; the
    # first check of the return ramp is garbage.
    steps = max(4, int(sc0.RAMP_S * sc0.RATE_HZ))
    checks = steps // sc0.CHECK_EVERY
    first_back_check = 2 + checks + 1 + 0        # +1: the +probe's final read
    bus = Bus(log, REST, garbage_on={2: {first_back_check}})
    sc = _sign_check(monkeypatch, log, bus, ["--joints", "2,3", "--yes"])
    rc = sc.main()
    assert rc != 0                                   # the run reports it, not a verdict
    j2 = _sent(log, 2)
    start = REST[2]
    assert j2, "j2 was probed"
    assert all(start - 0.06 - 1e-9 <= p <= start + 0.06 + 1e-9 for p in j2), max(j2)
    assert max(j2) > start + 0.05                    # the +probe did run
    # After the bad read, j2 only re-receives its last commanded pose.
    i_bad = next(i for i, e in enumerate(log) if e[0] == "read")
    before, after = _sent(log[:i_bad], 2), _sent(log[i_bad:], 2)
    assert after and all(abs(p - before[-1]) < 1e-12 for p in after), (before[-1], after)
    # j3 is never probed after a bad read: it only ever gets its hold pose.
    assert set(np.round(_sent(log, 3), 9)) == {round(REST[3], 9)}
    # every other motor only ever held its own startup pose
    for mid in (1, 4, 5, 6, 7):
        assert set(np.round(_sent(log, mid), 9)) == {round(REST[mid], 9)}, mid


def test_the_way_back_never_starts_from_a_reading(monkeypatch):
    """The +probe's FINAL read is garbage: the return ramp still starts from
    the last commanded position, not from the reading."""
    import sign_check_rebot_mb as sc0

    log = []
    steps = max(4, int(sc0.RAMP_S * sc0.RATE_HZ))
    checks = steps // sc0.CHECK_EVERY
    final_read = 2 + checks                       # after startup + drift + checks
    bus = Bus(log, REST, garbage_on={2: {final_read}})
    sc = _sign_check(monkeypatch, log, bus, ["--joints", "2", "--yes"])
    sc.main()
    j2 = _sent(log, 2)
    assert all(REST[2] - 0.06 - 1e-9 <= p <= REST[2] + 0.06 + 1e-9 for p in j2), max(j2)


def test_a_genuine_stall_holds_where_the_joint_is_inside_the_envelope(monkeypatch):
    """j2 below its stop: the reading stays at the start while the command
    ramps down. That is a real stall; holding at the (vetted) reading is right."""
    log = []

    class StuckBus(Bus):
        def read(self, motor, param, timeout_ms=300):
            if motor.mid == 2:
                sent = _sent(self.log, 2)
                # follows upward commands, blocked below the start
                return max(sent[-1], REST[2]) if sent else REST[2]
            return super().read(motor, param, timeout_ms)

    sc = _sign_check(monkeypatch, log, StuckBus(log, REST), ["--joints", "2", "--yes"])
    assert sc.main() == 0
    j2 = _sent(log, 2)
    assert all(REST[2] - 0.06 - 1e-9 <= p <= REST[2] + 0.06 + 1e-9 for p in j2)


# ── jog ──────────────────────────────────────────────────────────────────


def _jog(monkeypatch, log, bus, argv):
    import jog_rebot_mb as jog

    _fake_motorbridge(monkeypatch, log)
    monkeypatch.setattr(jog, "_sdk_motors", lambda: dict(SDK))
    monkeypatch.setattr(jog, "_profile_gains", lambda mid: SDK[mid][1:])
    monkeypatch.setattr(jog, "_read", bus.read)
    monkeypatch.setattr(jog, "_urdf_local_limits", _limits)
    monkeypatch.setattr(jog.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["jog", *argv])
    return jog


def test_jog_registers_each_motor_with_its_own_model_and_refuses_without(monkeypatch):
    log = []
    jog = _jog(monkeypatch, log, Bus(log, REST), ["--joint", "1", "--delta", "0.05", "--dry-run"])
    assert jog.main() == 0
    assert {e[1]: e[2] for e in log if e[0] == "add"} == {mid: SDK[mid][0] for mid in SDK}
    log.clear()
    monkeypatch.setattr(jog, "_sdk_motors", lambda: None)
    monkeypatch.setattr(sys, "argv", ["jog", "--joint", "1", "--kp", "5", "--kd", "0.5", "--yes"])
    assert jog.main() != 0
    assert not any(e[0] in ("enable_all", "send_mit", "ensure_mode") for e in log)


def test_jog_refuses_an_implausible_reading_before_energizing(monkeypatch):
    # (joint 1: mid-range, so nothing else in the jog refuses first)
    log = []
    jog = _jog(monkeypatch, log, Bus(log, REST, garbage_on={5: set(range(10))}),
               ["--joint", "1", "--delta", "0.05", "--yes"])
    assert jog.main() != 0
    assert not any(e[0] in ("enable_all", "send_mit") for e in log)


def test_jog_stops_on_an_implausible_reading_mid_ramp(monkeypatch):
    log = []
    # startup (1) + drift (1) + first stall check of the ramp -> garbage
    jog = _jog(monkeypatch, log, Bus(log, REST, garbage_on={1: {2}}),
               ["--joint", "1", "--delta", "0.05", "--yes"])
    assert jog.main() != 0
    j1 = _sent(log, 1)
    assert all(REST[1] - 1e-9 <= p <= REST[1] + 0.05 + 1e-9 for p in j1), max(j1)
    assert max(j1) < REST[1] + 0.05                       # it did not finish the ramp


def test_jog_ramp_stall_hold_is_clamped_to_the_envelope():
    """Unit level: whatever the stall reading, the hold sent stays in
    [start, goal] (here a plausible-but-wrong reading far past the goal)."""
    import jog_rebot_mb as jog

    from cascade.control.robstride import ProbeEnvelope

    log = []
    motors = {mid: Motor(mid, log) for mid in (1, 2)}
    q_hold = {1: 0.0, 2: 0.0}
    gains = {1: (5.0, 0.5), 2: (5.0, 0.5)}
    args = types.SimpleNamespace(duration=0.2, rate_hz=50.0)
    orig = jog._read
    jog._read = lambda m, p, timeout_ms=300: 2.5          # plausible for j2, far off
    try:
        ok = jog._ramp(motors, q_hold, 2, 0.0, 0.05, gains, args,
                       env=ProbeEnvelope.spanning(0.0, 0.05), limits=(0.0, 3.14))
    finally:
        jog._read = orig
    assert ok is False
    assert all(0.0 - 1e-9 <= p <= 0.05 + 1e-9 for p in _sent(log, 2))
