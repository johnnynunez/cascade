"""The RS bring-up scripts clear latched faults before they energize.

`scripts/jog_rebot_mb.py` and `scripts/sign_check_rebot_mb.py` are the first
things run on the rig, and they enable the motors themselves (not through
RebotRSMotorBridgeArm). WRC rig finding #1 applies to them too: a motor with
a latched fault (fault_raw 0x4) silently ignores MIT, so a jog "moves
nothing" -- and on 2026-08-27 a "barely moved" jog was misread as low gains,
the gains were raised, and the next run drove a blocked joint (see
jog_rebot_mb.STALL_LAG_RAD). The clear goes after the mode write and before
enable_all, exactly as in the backends.

Offline: motorbridge and the config/URDF lookups are faked; the fake
`enable_all` raises a sentinel so nothing after it can run.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))


class _Energized(Exception):
    pass


def _fake_motorbridge(monkeypatch, log):
    class Mode:
        MIT = "mit"

    class Motor:
        def __init__(self, mid):
            self.mid = mid

        def ensure_mode(self, mode, *a):
            log.append(("ensure_mode", self.mid))

        def clear_error(self):
            log.append(("clear_error", self.mid))

        def send_mit(self, *a):
            log.append(("send_mit", self.mid))

    class Controller:
        def __init__(self, channel=None):
            pass

        def add_robstride_motor(self, motor_id, feedback_id, model):
            return Motor(motor_id)

        def enable_all(self):
            log.append(("enable_all", None))
            raise _Energized

        def disable_all(self):
            log.append(("disable_all", None))

        def close_bus(self):
            pass

        def close(self):
            pass

    mod = types.ModuleType("motorbridge")
    mod.Controller, mod.Mode = Controller, Mode
    monkeypatch.setitem(sys.modules, "motorbridge", mod)


def _assert_cleared_before_enable(log):
    clears = [i for i, ev in enumerate(log) if ev[0] == "clear_error"]
    modes = [i for i, ev in enumerate(log) if ev[0] == "ensure_mode"]
    enable = next(i for i, ev in enumerate(log) if ev[0] == "enable_all")
    assert sorted(mid for ev, mid in log if ev == "clear_error") == [1, 2, 3, 4, 5, 6, 7]
    assert clears and max(modes) < min(clears) and max(clears) < enable, log


def _limits():
    return (np.array([-2.8, 0.0, 0.0, -1.69, -1.57, -3.14]),
            np.array([2.8, 3.14, 3.14, 1.79, 1.57, 3.14]))


def test_jog_clears_faults_before_enable(monkeypatch):
    import jog_rebot_mb as jog

    log = []
    _fake_motorbridge(monkeypatch, log)
    monkeypatch.setattr(jog, "_sdk_motors", lambda: None)
    monkeypatch.setattr(jog, "_profile_gains", lambda mid: (5.0, 0.5))
    monkeypatch.setattr(jog, "_read", lambda m, p, timeout_ms=300: 0.5)
    monkeypatch.setattr(jog, "_urdf_local_limits", _limits)
    monkeypatch.setattr(sys, "argv", ["jog", "--joint", "1", "--hold-only", "--yes"])
    with pytest.raises(_Energized):
        jog.main()
    _assert_cleared_before_enable(log)


def test_sign_check_clears_faults_before_enable(monkeypatch):
    import sign_check_rebot_mb as sc

    log = []
    _fake_motorbridge(monkeypatch, log)
    monkeypatch.setattr(sc, "_profile_gains", lambda mid: (5.0, 0.5))
    monkeypatch.setattr(sc, "_read", lambda m, p, timeout_ms=300: 0.5)
    monkeypatch.setattr(sc, "_urdf_local_limits", _limits)
    monkeypatch.setattr(sys, "argv", ["sign_check", "--joints", "1", "--yes"])
    with pytest.raises(_Energized):
        sc.main()
    _assert_cleared_before_enable(log)


def test_jog_dry_run_touches_no_motor(monkeypatch):
    """--dry-run must stay read-only: no mode write, no clear, no enable."""
    import jog_rebot_mb as jog

    log = []
    _fake_motorbridge(monkeypatch, log)
    monkeypatch.setattr(jog, "_sdk_motors", lambda: None)
    monkeypatch.setattr(jog, "_profile_gains", lambda mid: (5.0, 0.5))
    monkeypatch.setattr(jog, "_read", lambda m, p, timeout_ms=300: 0.5)
    monkeypatch.setattr(jog, "_urdf_local_limits", _limits)
    monkeypatch.setattr(sys, "argv", ["jog", "--joint", "1", "--dry-run"])
    assert jog.main() == 0
    assert log == []
