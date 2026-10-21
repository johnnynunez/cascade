"""The reBot RS gripper is referenced before it moves, and opening never pushes.

Rig incident, 2026-10-12 00:21: the arm was powered on with the jaws fully
open. A RobStride motor knows its position only within one turn after
power-on, and the jaw travels 0 -> 6.39 rad, more than one turn (2*pi =
6.283). So the motor read ~0.07, "closed". The first grasp then opened "to
6.2 rad", 6.2 rad past where the jaw really was, and broke the plastic stops.

Offline, on a fake clock: a physical jaw with hard stops at 0 and 6.39 rad,
driven by MIT PD (tau = kp * (target - pos) - kd * vel) with the profile's
gains. The motor frame is offset from the jaw by whole turns, the way a
power-on leaves it. No CAN bus, SDK or motorbridge.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
from conftest import needs_pin

from cascade.config import Cfg
from cascade.control import rebot_rs_arm as rs
from cascade.control import rebot_rs_mb_arm as mb
from cascade.control import robstride
from cascade.control import rs_gripper as rg
from cascade.control.rs_gripper import RSGripperGuard
from cascade.types import GripperFault, SafetyViolation

TWO_PI = 2 * math.pi
OPEN_END = 6.39          # measured open hard end
KD = 0.4
GRIP_KP = 2.0


class Jaw:
    """One jaw on a fake clock: MIT PD, hard stops at 0 and OPEN_END, an
    optional obstacle, and the motor frame offset by `raw_offset`."""

    def __init__(self, x0, raw_offset=0.0, inertia=0.002, friction=0.05):
        self.t = 100.0
        self.x = float(x0)
        self.v = 0.0
        self.raw_offset = float(raw_offset)
        self.inertia, self.friction = inertia, friction
        self.cmd = None                      # (raw target, kp)
        self.sent = []
        self.reads = 0
        self.obstacle = None                 # (position, "opening"|"closing")
        self.press = {"open_end": 0.0, "closed_end": 0.0, "obstacle": 0.0}

    # clock
    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.advance(max(0.0, float(s)))

    def advance(self, s):
        steps = max(1, round(s / 0.0005))
        h = s / steps
        for _ in range(steps):
            tau = 0.0
            if self.cmd is not None:
                target, kp = self.cmd
                tau = kp * (target - (self.x + self.raw_offset)) - KD * self.v
            if abs(self.v) > 1e-6:
                tau -= self.friction * math.copysign(1.0, self.v)
            elif abs(tau) <= self.friction:
                tau = 0.0
            self.v += h * tau / self.inertia
            self.x += h * self.v
            self._stops(tau)
        self.t += s

    def _stops(self, tau):
        if self.x >= OPEN_END:
            self.x, self.v = OPEN_END, 0.0
            self.press["open_end"] = max(self.press["open_end"], tau)
        if self.x <= 0.0:
            self.x, self.v = 0.0, 0.0
            self.press["closed_end"] = max(self.press["closed_end"], -tau)
        if self.obstacle is not None:
            at, side = self.obstacle
            if side == "opening" and self.x >= at:
                self.x, self.v = at, 0.0
                self.press["obstacle"] = max(self.press["obstacle"], tau)
            if side == "closing" and self.x <= at:
                self.x, self.v = at, 0.0
                self.press["obstacle"] = max(self.press["obstacle"], -tau)

    # motor
    def read_raw(self):
        self.reads += 1
        self.advance(0.008)                  # one mechPos param read
        return self.x + self.raw_offset

    def send_raw(self, pos, kp):
        self.advance(0.001)
        self.cmd = (float(pos), float(kp))
        self.sent.append(float(pos))


@pytest.fixture
def clock(monkeypatch):
    holder = {}

    def bind(jaw):
        fake = types.SimpleNamespace(monotonic=jaw.monotonic, sleep=jaw.sleep)
        for module in (rg, rs, mb, robstride):
            monkeypatch.setattr(module, "time", fake)
        holder["jaw"] = jaw
        return jaw
    return bind


def guard_for(jaw, stopped=lambda: False):
    return RSGripperGuard(open_pos=6.2, closed_pos=0.0, ref_kp=GRIP_KP, read_raw=jaw.read_raw,
                          send_raw=jaw.send_raw, stopped=stopped, log=lambda m: None)


# ── what happened on the rig, and what the guard does instead ───────────


def test_the_incident_a_jaw_powered_on_open_reads_closed_and_an_open_breaks_the_stop(clock):
    """The old path: a target in the motor's frame. Powered on at 6.35 rad
    the motor reads 0.07, and `open_pos` 6.2 drives the jaw into the open
    end pressing hard (the sim's check that it reproduces the incident)."""
    jaw = clock(Jaw(6.35, raw_offset=-TWO_PI))
    assert jaw.read_raw() == pytest.approx(6.35 - TWO_PI)        # "closed"
    jaw.send_raw(6.2, 0.8 * GRIP_KP)                               # open_gripper, before
    jaw.sleep(3.0)
    assert jaw.x == pytest.approx(OPEN_END)
    assert jaw.press["open_end"] > 8.0                              # Nm into the plastic stop


POWER_ON = [
    pytest.param(6.35, -TWO_PI, id="fully-open-reads-closed"),     # the incident
    pytest.param(5.0, -TWO_PI, id="open-in-a-[-pi,pi)-frame"),
    pytest.param(3.0, 0.0, id="half-open"),
    pytest.param(0.0, 0.0, id="closed"),
    pytest.param(0.0, TWO_PI, id="closed-reads-6.28"),
    pytest.param(6.1, 0.0, id="open-reads-open"),
]


@pytest.mark.parametrize(("x0", "raw_offset"), POWER_ON)
def test_any_power_on_pose_opens_to_open_pos_without_touching_the_open_end(clock, x0, raw_offset):
    jaw = clock(Jaw(x0, raw_offset))
    g = guard_for(jaw)
    assert g.position() is None                                     # unknown, never guessed
    g.command(6.2, 0.8 * GRIP_KP)                                   # open_gripper
    assert g.offset == pytest.approx(raw_offset)
    assert jaw.x == pytest.approx(6.2, abs=0.25)
    assert jaw.press["open_end"] == 0.0                             # never reached the stop
    assert g.position() == pytest.approx(jaw.x, abs=1e-6)
    assert max(abs(p) for p in jaw.sent) < robstride.MIT_P_MAX
    g.command(0.0, GRIP_KP)                                         # and closes again
    jaw.sleep(2.0)
    assert jaw.x == pytest.approx(0.0, abs=0.1)


def test_the_reference_closes_gently(clock):
    """Closing to the stop presses about kp * (lag + window travel): ~2 Nm,
    well under what a normal torque close already applies."""
    jaw = clock(Jaw(6.35, -TWO_PI))
    guard_for(jaw).reference()
    assert 0.0 < jaw.press["closed_end"] < 3.0
    assert jaw.press["open_end"] == 0.0


def test_something_between_the_jaws_fails_the_reference_and_moves_nothing_more(clock):
    jaw = clock(Jaw(4.0))
    jaw.obstacle = (1.5, "closing")                                 # an object in the jaws
    g = guard_for(jaw)
    with pytest.raises(GripperFault, match="something is between the jaws") as exc:
        g.command(6.2, 1.6)
    assert "raw +1.5" in str(exc.value) and "re-zero" in str(exc.value)
    assert not g.referenced and g.position() is None
    assert jaw.sent[-1] == pytest.approx(1.5, abs=0.05)             # holds where it stopped
    assert jaw.press["obstacle"] < 3.0
    n = len(jaw.sent)
    with pytest.raises(GripperFault):                               # still refuses, no surprise
        g.command(6.2, 1.6)                                         # (it tries the reference again)
    assert all(abs(p - 1.5) < 0.9 for p in jaw.sent[n:])            # only the gentle close


def test_opening_into_something_holds_there_instead_of_pushing(clock):
    jaw = clock(Jaw(0.0))
    g = guard_for(jaw)
    g.reference()
    jaw.obstacle = (3.0, "opening")
    with pytest.raises(GripperFault, match=r"blocked while opening at 3\.0"):
        g.command(6.2, 0.8 * GRIP_KP)
    assert jaw.sent[-1] == pytest.approx(3.0, abs=0.05)             # hold, not 6.2
    assert jaw.press["obstacle"] < 3.0                              # vs ~5 Nm for a direct 6.2
    jaw.sleep(1.0)
    assert jaw.press["obstacle"] < 3.0


@pytest.mark.parametrize("at", [5.7, 6.0])
def test_a_jaw_stopped_near_the_end_of_its_opening_is_judged_after_the_ramp(clock, at):
    """Blocked late: the ramp finishes before the stall window, and the jaw
    rests short of the target. More than OPEN_SHORT_RAD short is blocked
    (hold, fault); closer is accepted as open (friction, a soft finger)."""
    jaw = clock(Jaw(0.0))
    g = guard_for(jaw)
    g.reference()
    jaw.obstacle = (at, "opening")
    if 6.2 - at > rg.OPEN_SHORT_RAD:
        with pytest.raises(GripperFault, match="blocked while opening"):
            g.command(6.2, 0.8 * GRIP_KP)
        assert jaw.sent[-1] == pytest.approx(at, abs=0.05)
    else:
        g.command(6.2, 0.8 * GRIP_KP)
    assert jaw.press["obstacle"] < 3.0


@pytest.mark.parametrize("how", ["reading-outside-travel", "feedback-lost"])
def test_a_power_cycle_drops_the_reference_and_the_next_command_re_references(clock, how):
    """Referenced with the jaw powered on open (offset -2*pi). Then the
    arm is power-cycled: either the new reading lands outside the jaw's
    travel in the old frame, or the reads fail for a while. Both drop the
    reference; the next command references again and opens safely."""
    # reading-outside-travel: referenced powered-on-open (offset -2*pi), back
    # on with a 0 frame. feedback-lost: referenced closed (offset 0), back on
    # powered open: the new reading (0.07) is INSIDE the travel, so only the
    # failed reads tell.
    start_offset, new_offset = (-TWO_PI, 0.0) if how == "reading-outside-travel" else (0.0, -TWO_PI)
    jaw = clock(Jaw(3.0 if start_offset == 0.0 else 6.35, start_offset))
    g = guard_for(jaw)
    g.command(3.0, GRIP_KP)
    jaw.sleep(1.0)
    real_read = jaw.read_raw
    if how == "feedback-lost":
        def dead():
            raise RuntimeError("parameter 0x7019 not received within 1s")
        g._read_raw = dead
        for _ in range(rg.MAX_READ_FAILURES):
            assert g.position() is None
        g._read_raw = real_read
    jaw.raw_offset = new_offset                                      # power-on frame changed
    jaw.x, jaw.cmd = 6.35, None
    assert g.position() is None and not g.referenced
    g.command(6.2, 1.6)
    assert jaw.press["open_end"] == 0.0 and jaw.x == pytest.approx(6.2, abs=0.25)


def test_no_target_leaves_the_mit_range(clock):
    jaw = clock(Jaw(0.0, TWO_PI))
    g = guard_for(jaw)
    g.reference()
    with pytest.raises(GripperFault, match="MIT range"):
        g._send(robstride.MIT_P_MAX, 1.0)


def test_a_stop_ends_referencing_and_opening(clock):
    jaw = clock(Jaw(6.35, -TWO_PI))
    flag = {"stop": False}
    g = guard_for(jaw, stopped=lambda: flag["stop"])
    orig = jaw.read_raw

    def read():
        if len(jaw.sent) > 10:
            flag["stop"] = True
        return orig()
    jaw.read_raw = read
    g._read_raw = read
    with pytest.raises(GripperFault, match="interrupted"):
        g.command(6.2, 1.6)
    n = len(jaw.sent)
    g.command(6.2, 1.6)                                             # stopped: nothing sent
    assert len(jaw.sent) == n


# ── both RS drivers route every gripper command through the guard ───────


class _SdkMotor:
    def __init__(self, jaw):
        self.jaw = jaw

    def robstride_get_param_f32(self, param, *timeout):
        assert param == rs.MECH_POS
        return self.jaw.read_raw()


class _SdkGripper:
    def __init__(self, jaw):
        self.jaw = jaw
        self.joint_names = ["gripper"]
        self._mm = {"gripper": _SdkMotor(jaw)}

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None):
        self.jaw.send_raw(float(np.asarray(pos).reshape(-1)[0]),
                          float(np.asarray(kp).reshape(-1)[0]))


def sdk_arm(jaw):
    arm = rs.RebotRSArm(Cfg({"n_joints": 6, "gripper": {"open_pos": 6.2, "closed_pos": 0.0,
                                                         "kp": GRIP_KP, "kd": KD}}))
    arm._arm = types.SimpleNamespace(has_gripper=True, gripper=_SdkGripper(jaw))
    return arm


class _MbMotor:
    def __init__(self, jaw):
        self.jaw = jaw

    def robstride_get_param_f32(self, param, *timeout):
        assert param == mb.MECH_POS
        return self.jaw.read_raw()

    def send_mit(self, pos, vel, kp, kd, tau):
        self.jaw.send_raw(float(pos), float(kp))


def mb_arm(jaw):
    arm = mb.RebotRSMotorBridgeArm(Cfg({"n_joints": 6, "joint_ids": [1, 2, 3, 4, 5, 6],
                                        "gripper_id": 7, "wire_signs": [1] * 6,
                                        "mit_kp": [1.0] * 6, "mit_kd": [0.1] * 6,
                                        "gripper": {"open_pos": 6.2, "closed_pos": 0.0,
                                                    "kp": GRIP_KP, "kd": KD}}))
    arm._ctrl = object()
    arm._motors = {7: _MbMotor(jaw)}
    return arm


@pytest.mark.parametrize("make", [sdk_arm, mb_arm], ids=["rebot_rs", "rebot_rs_mb"])
def test_the_driver_opens_a_jaw_powered_on_open_without_breaking_it(clock, make):
    jaw = clock(Jaw(6.35, -TWO_PI))
    arm = make(jaw)
    assert arm._gripper_pos() is None                               # unknown until referenced
    arm.open_gripper()
    assert jaw.press["open_end"] == 0.0
    assert jaw.x == pytest.approx(6.2, abs=0.25)
    assert arm._gripper_pos() == pytest.approx(jaw.x, abs=1e-6)
    arm.close_gripper_torque() if hasattr(arm, "close_gripper_torque") else arm.set_gripper(0.0)
    jaw.sleep(1.0)
    assert jaw.x == pytest.approx(0.0, abs=0.15)
    assert jaw.press["open_end"] == 0.0


@pytest.mark.parametrize("make", [sdk_arm, mb_arm], ids=["rebot_rs", "rebot_rs_mb"])
def test_the_driver_reports_a_blocked_open_as_a_safety_stop(clock, make):
    jaw = clock(Jaw(0.0))
    arm = make(jaw)
    arm.set_gripper(1.0)
    jaw.obstacle = (2.5, "opening")
    with pytest.raises(SafetyViolation, match="blocked while opening"):
        arm.open_gripper()
    assert jaw.press["obstacle"] < 3.0


def test_rs_feedback_lost_drops_the_gripper_reference(clock):
    """The arm went quiet on the bus: it may have been powered off, and the
    gripper's turn count with it."""
    jaw = clock(Jaw(0.0))
    arm = sdk_arm(jaw)
    arm.set_gripper(2.0)
    assert arm._grip_guard.referenced

    def dead(*a, **k):
        raise RuntimeError("parameter 0x7019 not received within 1s")
    arm._fresh_positions = dead
    arm._read_failures = 0
    arm._last_q = np.zeros(6)
    for _ in range(arm.MAX_READ_FAILURES - 1):
        arm.get_state()
    with pytest.raises(RuntimeError, match="CAN feedback lost"):
        arm.get_state()
    assert not arm._grip_guard.referenced


# ── the runtime: a gripper fault is reported, not retried ───────────────


@pytest.fixture
def mock_runtime(tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


@pytest.mark.parametrize("fault", [
    ("gripper reference failed: the jaws stopped 1.50 rad from closed, not at the closed "
     "stop -- something is between the jaws."),
    "gripper blocked while opening at 3.00 rad (target 6.20): holding there instead of pushing.",
])
@needs_pin
def test_pick_and_place_stops_on_a_gripper_fault_and_asks_to_clear_the_jaws(mock_runtime, fault):
    runtime, arm = mock_runtime
    calls = []

    def set_gripper(pos, effort=1.0):
        calls.append(pos)
        raise GripperFault(fault)
    arm.set_gripper = set_gripper
    res = runtime.execute("pick_and_place", {"object": "red cube"})
    assert res["ok"] is False and "GripperFault" in res["error"]
    assert "gripper fault: not retrying" in res["error"]
    assert res.get("outcome") == "stuck"
    assert "jaws" in res["ask"] and "Clear the jaws" in res["ask"]
    assert len(calls) == 1


# ── scripts/check_rebot_gripper.py: the on-rig check ────────────────────


def _check_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "check_rebot_gripper.py"
    spec = importlib.util.spec_from_file_location("check_rebot_gripper", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(("q0", "code"), [(np.zeros(6), 0), (np.array([0, 1.2, 1.2, 0, 0, 0]), 2)])
def test_the_rig_check_moves_only_the_gripper_and_only_from_rest(clock, q0, code):
    jaw = clock(Jaw(6.35, -TWO_PI))                                  # the incident's power-on
    arm = sdk_arm(jaw)
    held = []
    arm.get_state = lambda: types.SimpleNamespace(q=q0.copy())
    arm.send_joint_target = lambda q: held.append(np.asarray(q, float).copy())
    out = []
    assert _check_script().run(arm, np.zeros(6), log=out.append) == code
    assert len(held) == 1 and np.allclose(held[0], q0)              # held where it was, once
    if code == 2:
        assert jaw.sent == [] and "not folded at rest" in out[-1]
        return
    assert any("raw reading at connect: +0.0668" in line for line in out)
    assert any("offset -1 turn(s)" in line for line in out)
    assert jaw.press["open_end"] == 0.0 and out[-1].startswith("[+] gripper OK")


def test_the_rig_check_needs_yes(capsys):
    assert _check_script().main([]) == 2
    assert "Nothing done" in capsys.readouterr().out


# ── --reference-only: the startup check run by the operator's `up` ───────
#
# Rig 2026-10-12: the gripper's zero no longer matched its closed stop (a rack
# had jumped teeth on the pinion after the 00:21 over-open), and nothing said
# so until the agent's first grasp failed mid-demo. The startup check finds it
# before the robot server starts: the arm must already be folded at rest (read
# with NO motor enabled), then only the gripper closes onto its stop.


def _rig_arm(jaw, q0=(0.0,) * 6):
    arm = sdk_arm(jaw)
    arm.get_state = lambda: types.SimpleNamespace(q=np.asarray(q0, float).copy())
    arm.send_joint_target = lambda q: None
    return arm


def test_the_startup_check_references_and_leaves_the_jaws_closed(clock):
    jaw = clock(Jaw(0.0, -TWO_PI))                                   # closed, one turn off
    out = []
    assert _check_script().run(_rig_arm(jaw), np.zeros(6), log=out.append, reference_only=True) == 0
    assert max(jaw.sent) <= -TWO_PI + 1e-9, "the startup check only ever closes"
    assert jaw.x == pytest.approx(0.0, abs=0.02) and jaw.press["open_end"] == 0.0
    assert any("offset -1 turn(s)" in line for line in out)
    assert out[-1].startswith("[+] gripper referenced")


def test_the_startup_check_fails_on_a_zero_that_no_longer_matches_the_closed_stop(clock):
    jaw = clock(Jaw(0.0, -2.78))                                     # tonight's shifted zero
    out = []
    assert _check_script().run(_rig_arm(jaw), np.zeros(6), log=out.append, reference_only=True) == 1
    assert "GripperFault" in out[-1] and "closed stop" in out[-1]
    assert max(jaw.sent) <= -2.78 + 1e-9, "a failed reference must not open the jaws"


class _ArmFactory:
    """Stands in for RebotRSArm in main(): records whether torque was ever
    asked for (connect) and released (disconnect)."""

    def __init__(self, arm):
        self.arm, self.calls = arm, []
        arm.connect = lambda: self.calls.append("connect")
        arm.disconnect = lambda: self.calls.append("disconnect")
        arm.stop = lambda: self.calls.append("stop")

    def __call__(self, cfg):
        self.calls.append("construct")
        return self.arm


@pytest.mark.parametrize(("q", "why"), [([0, 1.2, 1.2, 0, 0, 0], "not folded at rest"),
                                        ([0, 0, None, 0, 0, 0], "could not read every arm joint")],
                         ids=["raised", "unreadable"])
@pytest.mark.parametrize("mode", [["--reference-only"], []], ids=["startup", "full"])
def test_the_check_refuses_before_any_torque_unless_the_arm_rests(clock, monkeypatch, capsys, q, why, mode):
    jaw = clock(Jaw(0.0))
    factory = _ArmFactory(_rig_arm(jaw))
    mod = _check_script()
    monkeypatch.setattr(mod, "read_rest_pose", lambda: (list(q), list(q)))
    monkeypatch.setattr(rs, "RebotRSArm", factory)
    assert mod.main(["--yes", *mode]) == 2
    assert factory.calls == [] and jaw.sent == []
    out = capsys.readouterr().out
    assert why in out and "Nothing energized" in out


def test_the_startup_check_connects_references_and_releases_torque(clock, monkeypatch, capsys):
    jaw = clock(Jaw(0.0, -TWO_PI))
    factory = _ArmFactory(_rig_arm(jaw))
    mod = _check_script()
    rest = [0.001, 0.0, -0.002, 0.0, 0.0, 0.003]
    monkeypatch.setattr(mod, "read_rest_pose", lambda: (list(rest), list(rest)))
    monkeypatch.setattr(rs, "RebotRSArm", factory)
    assert mod.main(["--yes", "--reference-only"]) == 0
    assert factory.calls == ["construct", "connect", "disconnect"]
    assert max(jaw.sent) <= -TWO_PI + 1e-9
    assert "torque released" in capsys.readouterr().out
