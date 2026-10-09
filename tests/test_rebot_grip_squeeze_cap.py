"""B38: opt-in contact-relative squeeze cap in the real reBot RS gripper close.

Seeed feedback (2026-10-08): "no gripper constraint on object size; the paper
cup should not have been crushed". The cause is in both RS drivers'
`close_gripper_two_stage`: the jaws are sent to a FIXED fraction of travel
(`grasping/force.py` GRIP_PROFILES, rigid stage 2 = 0.85) at kp*effort and left
there, so under MIT (tau = kp*(target - pos)) the steady holding torque is
kp_eff*(contact - target) and GROWS with object width -- a 7.5 cm cup gets more
than the 5 cm cube. `_hold_light` (kp cut after contact) only runs in the park
close (`close_gripper_torque`).

The fix (`gripper.max_contact_squeeze_rad`, null = the old close unchanged):
after the close sees the jaw stopped short of a stage target (mechPos-only
stall), every later jaw target is at most `cap` past that contact, so the
steady torque is min(today's, kp_eff*cap): identical for objects whose squeeze
already fits the cap (the 5 cm booth cube under the rigid profile with the
recommended start value), bounded for wide or deformable ones. A jaw that then
reaches the capped target met nothing (a stick-slip or a misread stall), so
the stage target is restored -- the close is never left weaker than today in
free air.

Offline only. `_SimJaw` is a RobStride gripper motor under MIT position
control (overdamped, tau = kp*(target - pos) - kd*vel), integrated lazily to a
fake clock, with an object that stops the jaws at its width (rigid stop or a
compliant spring) and mechPos/mechVel served through
`robstride_get_param_f32` exactly where the drivers read them. The width map is
the profile's linear one (open_pos 6.2 rad <-> max_width_m 0.09 m), which is
unverified on hardware (docs/WRC_CONTROL_PORT.md); the cap bound kp_eff*cap
does not depend on it, the per-object numbers do.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from cascade.config import Cfg
from cascade.control import rebot_rs_arm as rs_mod
from cascade.control.rebot_rs_arm import MECH_POS, MECH_VEL, RebotRSArm
from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm
from cascade.grasping.force import GRIP_PROFILES

REPO = Path(__file__).resolve().parents[1]
OPEN, CLOSED, MAX_W, KD = 6.2, 0.0, 0.09, 0.4
RS_KP = 2.0   # configs/arms/rebot_rs.yaml gripper.kp
MB_KP = 6.0   # configs/arms/rebot_rs_mb.yaml gripper.kp
OBJECTS_M = (0.03, 0.05, 0.075, 0.085)
# Recommended starting cap: the squeeze the 5 cm booth cube gets today under
# the rigid profile, rounded UP: contact 0.05/0.09*6.2 = 3.4444 rad, stage-2
# target 6.2*(1 - 0.85) = 0.93 rad -> 2.5144 rad.
CAP = 2.52
_UNSET = object()


def _contact(width_m: float) -> float:
    return CLOSED + (OPEN - CLOSED) * width_m / MAX_W


def _stage_target(frac: float) -> float:
    return OPEN + (CLOSED - OPEN) * frac


def _kp_eff(kp: float, effort: float) -> float:
    return kp * float(np.clip(effort, 0.05, 1.0))


# ── simulated RobStride gripper ─────────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += float(s)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    # rs_mod.time IS the stdlib module, so robstride.py sees the same clock.
    monkeypatch.setattr(rs_mod.time, "sleep", c.sleep)
    monkeypatch.setattr(rs_mod.time, "monotonic", c.monotonic)
    return c


class _SimJaw:
    """MIT-controlled jaw: kd*vel = kp*(target - pos) + tau_object (no libm,
    so the trajectory is bit-identical on every CI platform)."""

    DT = 0.002

    def __init__(self, *, pos=OPEN, contact=None, k_obj=None, stick=None,
                 vel_reading=None):
        self.pos, self.vel = float(pos), 0.0
        self.target, self.kp, self.kd = float(pos), 0.0, KD
        self.contact = contact          # jaw angle where the fingers touch the object
        self.k_obj = k_obj              # None = rigid stop, else N*m/rad spring
        self.stick = stick              # (angle, until t - t0): free-air stick-slip
        self.vel_reading = vel_reading  # None = true rad/s; a number = broken mechVel
        self.t = self.t0 = time.monotonic()
        self.events: list[tuple] = []  # times relative to t0
        self.on_read = None

    def _advance(self):
        now = time.monotonic()
        while self.t < now - 1e-12:
            dt = min(self.DT, now - self.t)
            tau_obj = 0.0
            if self.contact is not None and self.k_obj is not None and self.pos < self.contact:
                tau_obj = self.k_obj * (self.contact - self.pos)
            v = (self.kp * (self.target - self.pos) + tau_obj) / self.kd
            new = self.pos + v * dt
            if self.contact is not None and self.k_obj is None and new < self.contact:
                new = self.contact
            if (self.stick is not None and self.t - self.t0 < self.stick[1]
                    and new < self.stick[0] <= self.pos):
                new = self.stick[0]
            self.vel = (new - self.pos) / dt
            self.pos = new
            self.t += dt

    def command(self, pos, kp, kd):
        self._advance()
        self.target, self.kp, self.kd = float(pos), float(kp), float(kd)
        self.events.append(("cmd", round(time.monotonic() - self.t0, 6), round(self.target, 6),
                            round(self.kp, 6), round(self.kd, 6)))

    def read(self, param):
        self._advance()
        self.events.append(("read", round(time.monotonic() - self.t0, 6), hex(param)))
        if self.on_read is not None:
            self.on_read()
        if param == MECH_POS:
            return self.pos
        if param == MECH_VEL:
            return self.vel if self.vel_reading is None else float(self.vel_reading)
        raise AssertionError(f"unexpected param {param:#x}")

    def settle(self, seconds=2.0):
        time.sleep(seconds)
        self._advance()

    def commands(self):
        return [(e[2], e[3]) for e in self.events if e[0] == "cmd"]

    def squeeze_torque(self):
        """Steady MIT torque toward closed (closed < open on the RS jaw)."""
        self._advance()
        return self.kp * (self.pos - self.target)

    def trace_digest(self):
        return hashlib.sha256(repr(self.events).encode()).hexdigest()[:16]


class _Motor:
    """A RobStride motor handle: motorbridge's Motor and the SDK's _mm entry."""

    def __init__(self, jaw):
        self.jaw = jaw

    def robstride_get_param_f32(self, param, timeout_ms=None):
        return self.jaw.read(param)

    def send_mit(self, pos, vel, kp, kd, tau):          # motorbridge signature
        self.jaw.command(pos, kp, kd)


class _SdkGripper:
    """reBotArm_control_py JointGroup double for the gripper group."""

    joint_names = ["gripper"]

    def __init__(self, jaw):
        self.jaw = jaw
        self._mm = {"gripper": _Motor(jaw)}

    def send_mit(self, pos, kp=None, kd=None, **kw):
        self.jaw.command(float(np.asarray(pos).reshape(-1)[0]),
                         float(np.asarray(kp).reshape(-1)[0]),
                         float(np.asarray(kd).reshape(-1)[0]))


class _SdkArm:
    has_gripper = True

    def __init__(self, jaw):
        self.gripper = _SdkGripper(jaw)


def _gripper_cfg(cap, kp):
    g = {"open_pos": OPEN, "closed_pos": CLOSED, "max_width_m": MAX_W, "kp": kp, "kd": KD}
    if cap is not _UNSET:
        g["max_contact_squeeze_rad"] = cap
    return g


def _rs(jaw, cap=_UNSET, kp=RS_KP):
    arm = RebotRSArm(Cfg({"gripper": _gripper_cfg(cap, kp)}))
    arm._arm = _SdkArm(jaw)
    arm._last_cmd_q = None
    return arm


def _mb(jaw, cap=_UNSET, kp=MB_KP):
    arm = RebotRSMotorBridgeArm(Cfg({
        "n_joints": 6, "gripper_id": 7, "mit_kp": [1.0] * 6, "mit_kd": [0.1] * 6,
        "gripper": _gripper_cfg(cap, kp)}))
    arm._ctrl = object()
    arm._motors = {7: _Motor(jaw)}
    return arm


def _close(arm, profile):
    """Exactly what runtime._close_two_stage passes to the raw driver."""
    return arm.close_gripper_two_stage(
        width_frac_stage1=profile.close_frac_stage1,
        width_frac_stage2=profile.close_frac_stage2,
        effort=profile.effort)


def _old_torque(kp, profile, width_m):
    squeeze = _contact(width_m) - _stage_target(profile.close_frac_stage2)
    return _kp_eff(kp, profile.effort) * max(squeeze, 0.0)


# ── premises: what the driver does today (pass on main by design) ────────────


def test_premise_holding_torque_grows_with_object_width(clock):
    """Today the rigid close leaves kp_eff*(contact - 0.93 rad) on the object:
    3 < 5 < 7.5 < 8.5 cm, the wider the harder (Seeed's crushed paper cup)."""
    rigid = GRIP_PROFILES["rigid"]
    torques = []
    for w in OBJECTS_M:
        jaw = _SimJaw(contact=_contact(w))
        _close(_rs(jaw), rigid)
        jaw.settle()
        torques.append(jaw.squeeze_torque())
        assert jaw.squeeze_torque() == pytest.approx(_old_torque(RS_KP, rigid, w), abs=1e-6)
    assert torques == sorted(torques) and torques[-1] > 2 * torques[0]


def test_premise_pick_close_never_relaxes_after_contact(clock):
    """The pick close is two jaw commands and nothing after contact: the
    light hold (`_hold_light`) belongs to the park close only."""
    rigid = GRIP_PROFILES["rigid"]
    jaw = _SimJaw(contact=_contact(0.075))
    _close(_rs(jaw), rigid)
    assert jaw.commands() == [
        (pytest.approx(_stage_target(0.50)), pytest.approx(RS_KP * 0.7 * 0.7)),
        (pytest.approx(_stage_target(0.85)), pytest.approx(RS_KP * 0.7)),
    ]


# Golden traces of the close with the cap off, captured on origin/main
# (4d0947b) with this simulator: every command AND every mechPos/mechVel read,
# with its fake-clock time. `null` must stay byte-identical to them.
_GOLDEN = {
    ("rigid", None): (28, "b2abc688f1e80dc7"),        # cmds (3.1, 0.98) @0, (0.93, 1.4) @1.5
    ("rigid", 0.05): (18, "96d5c16532b44132"),        # cmds (3.1, 0.98) @0, (0.93, 1.4) @1.05
    ("deformable", 0.075): (12, "95ed1f489bd375a2"),  # cmds (2.48, 0.63) @0, (0.31, 0.9) @0.6
}


@pytest.mark.parametrize("cap", [_UNSET, None], ids=["key-absent", "key-null"])
@pytest.mark.parametrize("profile, width", list(_GOLDEN), ids=lambda v: str(v))
def test_null_cap_is_byte_identical_to_main(clock, cap, profile, width):
    jaw = _SimJaw(contact=None if width is None else _contact(width))
    _close(_rs(jaw, cap=cap), GRIP_PROFILES[profile])
    assert (len(jaw.events), jaw.trace_digest()) == _GOLDEN[(profile, width)]


def test_stop_mid_close_sends_no_further_jaw_command(clock):
    """stop() during the close: set_gripper refuses every later command,
    the cap's re-command included (it goes through set_gripper)."""
    jaw = _SimJaw(contact=_contact(0.085))
    arm = _rs(jaw, cap=CAP)
    jaw.on_read = lambda: arm.stop() if time.monotonic() >= 0.4 and not arm._stopped else None
    _close(arm, GRIP_PROFILES["rigid"])
    assert jaw.commands() == [(pytest.approx(_stage_target(0.50)), pytest.approx(RS_KP * 0.49))]


# ── the cap ──────────────────────────────────────────────────────────────────


def _row(kp, profile, cap, build=_rs):
    out = []
    for w in OBJECTS_M:
        jaw = _SimJaw(contact=_contact(w))
        _close(build(jaw, cap=cap, kp=kp), profile)
        jaw.settle(8.0)  # the slowest free close (fragile, kp_eff 0.6) converges too
        out.append(jaw.squeeze_torque())
    return out


@pytest.mark.parametrize("name", sorted(GRIP_PROFILES))
def test_steady_torque_is_min_of_today_and_cap_for_every_profile(clock, name):
    """Rigid objects 3/5/7.5/8.5 cm: new = min(old, kp_eff*cap) -- never weaker
    than today where the squeeze fits the cap, bounded where it does not."""
    profile = GRIP_PROFILES[name]
    bound = _kp_eff(RS_KP, profile.effort) * CAP
    new = _row(RS_KP, profile, CAP)
    for w, torque in zip(OBJECTS_M, new):
        old = _old_torque(RS_KP, profile, w)
        assert torque == pytest.approx(min(old, bound), abs=1e-6), (name, w)
        assert torque <= old + 1e-6
    # every profile has at least one object (8.5 cm) the cap actually bounds
    assert new[-1] == pytest.approx(bound)
    assert bound < _old_torque(RS_KP, profile, OBJECTS_M[-1])


def test_reference_cube_and_narrower_objects_keep_todays_exact_commands(clock):
    """B36 lesson (a lighter post-contact hold dropped cubes in PhysX): with the
    recommended cap the 5 cm booth cube and the 3 cm prop get the SAME jaw
    targets and kp as today under the rigid profile."""
    rigid = GRIP_PROFILES["rigid"]
    for w in (0.03, 0.05):
        old, new = _SimJaw(contact=_contact(w)), _SimJaw(contact=_contact(w))
        _close(_rs(old), rigid)
        _close(_rs(new, cap=CAP), rigid)
        assert new.commands() == old.commands(), w


def test_wide_object_gets_capped_target_and_contact_is_recorded(clock):
    rigid = GRIP_PROFILES["rigid"]
    c = _contact(0.075)
    jaw = _SimJaw(contact=c)
    arm = _rs(jaw, cap=CAP)
    final = _close(arm, rigid)
    assert final == pytest.approx(c)
    assert arm._grip_contact_pos == pytest.approx(c)
    assert jaw.commands()[-1] == (pytest.approx(c - CAP), pytest.approx(RS_KP * 0.7))
    # the next close on the same arm starts from a clean slate: an empty jaw
    # reports no contact (no stale angle from the previous object)
    jaw.contact = None
    arm.open_gripper()
    jaw.settle(3.0)
    _close(arm, rigid)
    assert arm._grip_contact_pos is None


def test_tight_cap_bounds_contacts_found_in_either_stage(clock):
    """cap 0.5 rad, rigid profile: the 3 cm prop is first touched in stage 2
    (re-commanded in that stage), the wider objects in stage 1 -- all end at
    kp_eff*0.5 on the object."""
    rigid = GRIP_PROFILES["rigid"]
    for w in OBJECTS_M:
        jaw = _SimJaw(contact=_contact(w))
        arm = _rs(jaw, cap=0.5)
        _close(arm, rigid)
        jaw.settle()
        assert arm._grip_contact_pos == pytest.approx(_contact(w)), w
        assert jaw.target == pytest.approx(_contact(w) - 0.5), w
        assert jaw.squeeze_torque() == pytest.approx(RS_KP * 0.7 * 0.5), w


@pytest.mark.parametrize("stick", [None, (OPEN, 0.9), (4.0, 0.85)],
                         ids=["free", "late-spin-up", "one-poll-stick"])
def test_free_air_motion_that_is_not_a_contact_keeps_todays_commands(clock, stick):
    """Even with a tight cap, a free close, a jaw that has not started moving
    yet (no travel) and a stick-slip lasting a single poll are not contacts:
    the jaw gets today's targets and kp."""
    rigid = GRIP_PROFILES["rigid"]
    old = _SimJaw(stick=stick)
    _close(_rs(old), rigid)
    new = _SimJaw(stick=stick)
    arm = _rs(new, cap=0.5)
    _close(arm, rigid)
    assert new.commands() == old.commands()
    assert arm._grip_contact_pos is None


@pytest.mark.parametrize("name", sorted(GRIP_PROFILES))
def test_free_close_is_unchanged_by_the_cap(clock, name):
    """No object: the same jaw commands as today, ending at the stage target."""
    profile = GRIP_PROFILES[name]
    old, new = _SimJaw(), _SimJaw()
    _close(_rs(old), profile)
    arm = _rs(new, cap=CAP)
    _close(arm, profile)
    assert new.commands() == old.commands()
    new.settle(6.0)
    assert new.pos == pytest.approx(_stage_target(profile.close_frac_stage2), abs=1e-3)


def test_compliant_cup_is_squeezed_less_and_never_past_the_cap(clock):
    """7.5 cm paper cup (deformable profile, spring object): the jaw squeezes
    less than today and never more than `cap` past the first detected stall."""
    deformable = GRIP_PROFILES["deformable"]
    c = _contact(0.075)
    old, new = _SimJaw(contact=c, k_obj=1.0), _SimJaw(contact=c, k_obj=1.0)
    _close(_rs(old), deformable)
    arm = _rs(new, cap=CAP)
    _close(arm, deformable)
    old.settle()
    new.settle()
    first = arm._grip_contact_pos
    # the first stall is the stage-1 (scout) equilibrium on the cup spring:
    # 0.63*(p - 2.48) = 1.0*(5.167 - p) -> p = 4.128 rad; it is never moved
    kp1 = _kp_eff(RS_KP, deformable.effort * 0.7)
    stage1_eq = (1.0 * c + kp1 * _stage_target(deformable.close_frac_stage1)) / (1.0 + kp1)
    assert first == pytest.approx(stage1_eq, abs=0.07)
    assert new.pos < first
    assert first - new.target == pytest.approx(CAP)
    assert new.pos > old.pos + 0.3                      # the cup is crushed less
    assert new.squeeze_torque() < old.squeeze_torque()
    assert new.squeeze_torque() <= _kp_eff(RS_KP, deformable.effort) * CAP


@pytest.mark.parametrize("stick, cap", [((4.0, 1.5), CAP), ((2.0, 2.45), 0.5)],
                         ids=["stage1-stick", "stage2-stick-restalls-once"])
def test_false_contact_in_free_air_restores_the_stage_target(clock, stick, cap):
    """A stick-slip in free air looks like a stall; once released the jaw runs
    to the capped target with nothing in the way -> restore today's target, so
    the post-lift air-grasp check still sees the jaws fully closed. The second
    case stays stuck for one more poll after the capped re-command: the capped
    hold is still watched (two fresh still polls), not taken at once."""
    rigid = GRIP_PROFILES["rigid"]
    jaw = _SimJaw(stick=stick)
    arm = _rs(jaw, cap=cap)
    _close(arm, rigid)
    stage2 = _stage_target(0.85)
    assert (pytest.approx(stick[0] - cap), pytest.approx(RS_KP * 0.7)) in jaw.commands()
    assert jaw.commands()[-1] == (pytest.approx(stage2), pytest.approx(RS_KP * 0.7))
    assert arm._grip_contact_pos is None
    jaw.settle()
    assert jaw.pos == pytest.approx(stage2, abs=1e-3)


@pytest.mark.parametrize("vel_reading", [0.0, 50.0], ids=["mechVel-zero", "mechVel-huge"])
def test_cap_uses_mechpos_only(clock, vel_reading):
    """mechVel (0x701A) is not rad/s on this firmware: the cap's contact signal
    is mechPos alone, so a stuck-at-zero mechVel cannot cap a free close and a
    huge one cannot hide the contact."""
    rigid = GRIP_PROFILES["rigid"]
    free = _SimJaw(vel_reading=vel_reading)
    _close(_rs(free, cap=CAP), rigid)
    free.settle()
    assert free.pos == pytest.approx(_stage_target(0.85), abs=1e-3)
    c = _contact(0.085)
    held = _SimJaw(contact=c, vel_reading=vel_reading)
    _close(_rs(held, cap=CAP), rigid)
    held.settle()
    assert held.squeeze_torque() == pytest.approx(RS_KP * 0.7 * CAP)


def test_motorbridge_transport_caps_the_same_way(clock):
    for name in ("rigid", "deformable"):
        profile = GRIP_PROFILES[name]
        new = _row(MB_KP, profile, CAP, build=_mb)
        bound = _kp_eff(MB_KP, profile.effort) * CAP
        for w, torque in zip(OBJECTS_M, new):
            assert torque == pytest.approx(min(_old_torque(MB_KP, profile, w), bound), abs=1e-6)


def test_motorbridge_null_cap_keeps_todays_commands(clock):
    rigid = GRIP_PROFILES["rigid"]
    old = _SimJaw(contact=_contact(0.075))
    _close(_mb(old), rigid)
    new = _SimJaw(contact=_contact(0.075))  # its own t0: traces compare in relative time
    _close(_mb(new, cap=None), rigid)
    assert new.events == old.events
    assert new.commands()[-1] == (pytest.approx(_stage_target(0.85)), pytest.approx(MB_KP * 0.7))


@pytest.mark.parametrize("bad", [-1.0, 0.0, 0.05, 0.1, float("nan"), float("inf"), "2.5", True, [2.5]])
@pytest.mark.parametrize("build", [_rs, _mb], ids=["rebot_rs", "rebot_rs_mb"])
def test_invalid_cap_fails_closed_before_any_jaw_command(clock, build, bad):
    """A cap must be a finite number of radians above the 0.1 rad stage-reach
    tolerance (below it a real contact would read as 'target reached' and the
    cap would silently undo itself). Anything else refuses the profile: the
    driver is never built, so no jaw command can follow."""
    jaw = _SimJaw()
    with pytest.raises(ValueError, match="max_contact_squeeze_rad"):
        build(jaw, cap=bad)
    assert jaw.events == []


# ── the runtime's grip verification keeps working ─────────────────────────────


def _grasp_runtime(monkeypatch, jaw, cap):
    """The real skill_grasp_object (test_grasp_evidence's runtime double) with
    the jaw served by the reBot RS driver's close and its mechPos."""
    from test_grasp_evidence import runtime as basic_runtime

    rt, calls, fix, frame = basic_runtime(monkeypatch)
    jaw.t = jaw.t0 = time.monotonic()  # the double installs its own fake clock
    rebot = _rs(jaw, cap=cap)
    bridge = rt.arm.raw._client
    monkeypatch.setattr(type(bridge), "gripper_pos", property(
        lambda self: jaw.read(MECH_POS),
        lambda self, pos: setattr(jaw, "pos", float(pos))), raising=False)
    monkeypatch.setattr(rt.arm.raw, "close_gripper_two_stage",
                        rebot.close_gripper_two_stage, raising=False)
    rt._grip_open, rt._grip_closed = OPEN, CLOSED
    return rt, fix, frame


def test_runtime_verifies_a_capped_grip_and_keeps_it_held(monkeypatch):
    """'orange' -> slippery profile (0.90, effort 0.85); a 5 cm rigid object.
    The capped close holds at the contact, so the post-lift check measures the
    object width, reports grip_verified and `_reconcile_held` keeps it."""
    jaw = _SimJaw(contact=_contact(0.05))
    rt, fix, frame = _grasp_runtime(monkeypatch, jaw, CAP)
    result = rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    assert result["held"] == "orange" and result["grip_profile"] == "slippery"
    assert result["grip_verified"] is True
    assert result["gripper_open_frac"] == pytest.approx(0.56, abs=0.01)
    slippery = GRIP_PROFILES["slippery"]
    assert jaw.squeeze_torque() == pytest.approx(_kp_eff(RS_KP, slippery.effort) * CAP)
    assert jaw.squeeze_torque() < _old_torque(RS_KP, slippery, 0.05)
    rt._reconcile_held()
    assert rt.held_object == "orange"


def test_runtime_still_reports_an_air_grasp_with_the_cap_on(monkeypatch):
    jaw = _SimJaw()
    rt, fix, frame = _grasp_runtime(monkeypatch, jaw, CAP)
    result = rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    assert result["ok"] is False and "air grasp" in result["error"]
    assert rt.held_object is None


# ── shipped profiles and docs ─────────────────────────────────────────────────


@pytest.mark.parametrize("profile", ["rebot_rs", "rebot_rs_mb"])
def test_shipped_profiles_declare_the_cap_and_leave_it_off(profile):
    data = yaml.safe_load((REPO / "configs" / "arms" / f"{profile}.yaml").read_text())
    assert "max_contact_squeeze_rad" in data["gripper"]
    assert data["gripper"]["max_contact_squeeze_rad"] is None


def test_architecture_known_limitation_names_the_cap():
    text = (REPO / "docs" / "ARCHITECTURE.md").read_text()
    limitation = text[text.index("- Grip force is a stiffness proxy"):]
    limitation = limitation[: limitation.index("\n- ", 3)]
    assert "max_contact_squeeze_rad" in limitation
