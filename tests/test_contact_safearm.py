"""The contact monitor through SafeArm and the SafetyHarness.

`safety.contact_monitor` lives in the arm's safety envelope (SafetyLimits);
SafeArm builds one ContactMonitor per streamed motion and hands it to the
backend, records every motion's report on the harness (stderr in measure
mode, the run trace via the skill runtime), and turns a contact into a
contact zone: the TCP may not re-enter the spot where the arm was blocked
until reset_scene, so nothing re-plans straight back into the obstacle.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_safety import FakeKin, limits

from cascade.config import Cfg
from cascade.control.mock_arm import MockArm
from cascade.safety.contact import ContactMonitorConfig
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.types import ContactDetected, SafetyViolation, SkillError

START = np.array([0.30, 0.0, 0.20, 0.0, 0.0, 0.0])   # FakeKin: TCP = q[:3]


def contact_cfg(**kw):
    base = {"enabled": True, "mode": "enforce", "track_tol_rad": 0.10, "lag_s": 0.25,
            "debounce_samples": 2, "contact_zone_m": 0.05}
    base.update(kw)
    return ContactMonitorConfig.from_config(base)


def rig(contact=None, kin=True):
    h = SafetyHarness(limits(contact=contact), kinematics=FakeKin() if kin else None)
    arm = MockArm(Cfg({"n_joints": 6, "home_q": START.tolist()}))
    arm.connect()
    return arm, SafeArm(arm, h), h


# ── config ───────────────────────────────────────────────────────────────


def test_safety_limits_parse_the_contact_monitor_block():
    cfg = Cfg({"workspace": {"min": [0, -1, 0], "max": [1, 1, 1]},
               "contact_monitor": {"enabled": True, "track_tol_rad": [0.1] * 6, "lag_s": 0.2}})
    lim = SafetyLimits.from_config(cfg)
    assert lim.contact.enabled and lim.contact.lag_s == 0.2
    assert SafetyLimits.from_config(Cfg({"workspace": cfg.workspace})).contact is None
    with pytest.raises(ValueError, match="unknown key"):
        SafetyLimits.from_config(Cfg({"workspace": cfg.workspace,
                                      "contact_monitor": {"enabled": True, "track_tol": 0.1}}))


def test_disabled_monitor_passes_nothing_to_the_backend():
    seen = []

    class Recording(MockArm):
        def stream_to(self, q, duration_s, **kw):
            seen.append(sorted(kw))
            return super().stream_to(q, duration_s, **kw)
    for contact in (None, contact_cfg(enabled=False)):
        h = SafetyHarness(limits(contact=contact), kinematics=FakeKin())
        arm = Recording(Cfg({"n_joints": 6, "home_q": START.tolist()}))
        arm.connect()
        assert SafeArm(arm, h).move_joints(START + [0.05, 0, 0, 0, 0, 0], 1.0)
    assert seen == [["approve"], ["approve"]]


# ── trip through SafeArm ─────────────────────────────────────────────────


def test_safearm_contact_trips_holds_and_is_recorded():
    arm, safe, h = rig(contact_cfg())
    arm.block_joint(0, at=START[0] + 0.04)
    with pytest.raises(ContactDetected, match="contact detected at joint 1") as exc:
        safe.move_joints(START + [0.15, 0, 0, 0, 0, 0], 2.0)
    assert isinstance(exc.value, SafetyViolation)
    np.testing.assert_allclose(arm.commands[-1], arm.get_state().q)   # held where it is
    assert not h.motion_active
    assert h.last_contact["joint"] == 1 and h.last_contact["motion"] == "move_joints"
    assert any("contact detected" in v for v in h.violations)
    reports = h.drain_contact_reports()
    assert len(reports) == 1 and reports[0]["tripped"] and reports[0]["hold"].startswith("measured")
    assert h.drain_contact_reports() == []


def test_safearm_joint_path_is_monitored_too():
    arm, safe, h = rig(contact_cfg())
    arm.block_joint(1)
    with pytest.raises(ContactDetected, match="joint 2"):
        safe.move_joint_path([START + [0, 0.05, 0, 0, 0, 0], START + [0, 0.12, 0, 0, 0, 0]], 2.0)
    assert h.drain_contact_reports()[0]["motion"] == "move_joint_path"


def test_free_motion_is_reported_without_a_trip():
    _arm, safe, h = rig(contact_cfg())
    assert safe.move_joints(START + [0.1, 0, 0, 0, 0, 0], 1.0)
    (report,) = h.drain_contact_reports()
    assert report["tripped"] is None and min(report["samples"]) > 0
    assert h.last_contact is None and h.contact_zones == []


def test_measure_mode_logs_and_never_stops(capsys):
    arm, safe, h = rig(contact_cfg(mode="measure"))
    arm.block_joint(0)
    safe.move_joints(START + [0.15, 0, 0, 0, 0, 0], 2.0)
    assert len(arm.commands) == 100                    # streamed to the end
    err = capsys.readouterr().err
    assert "[contact] measure move_joints" in err and "WOULD_TRIP" in err
    (report,) = h.drain_contact_reports()
    assert report["would_trip"] and h.last_contact is None and h.contact_zones == []


def test_enforce_mode_logs_the_trip(capsys):
    arm, safe, _ = rig(contact_cfg())
    arm.block_joint(0)
    with pytest.raises(ContactDetected):
        safe.move_joints(START + [0.15, 0, 0, 0, 0, 0], 2.0)
    assert "TRIPPED" in capsys.readouterr().err


def test_backend_hints_are_still_dropped_but_the_monitor_is_kept():
    """The grasp descent passes `bias_compensate=True`, which ArmBase/MockArm
    do not accept: SafeArm retries without the hint. The monitor must ride
    along on that retry, never be dropped with the hint."""
    arm, safe, _h = rig(contact_cfg())
    arm.block_joint(2)
    with pytest.raises(ContactDetected):
        safe.move_joints(START + [0, 0, 0.12, 0, 0, 0], 2.0, bias_compensate=True)


def test_a_backend_that_cannot_run_the_monitor_refuses_before_any_command():
    class Legacy(MockArm):
        def stream_to(self, q, duration_s, rate_hz=50.0, approve=None):  # no contact_monitor
            return super().stream_to(q, duration_s, rate_hz=rate_hz, approve=approve)
    h = SafetyHarness(limits(contact=contact_cfg()), kinematics=FakeKin())
    arm = Legacy(Cfg({"n_joints": 6, "home_q": START.tolist()}))
    arm.connect()
    with pytest.raises(SafetyViolation, match="cannot stream under the contact monitor"):
        SafeArm(arm, h).move_joints(START + [0.05, 0, 0, 0, 0, 0], 1.0)
    assert arm.commands == []


def test_a_planner_bound_arm_refuses_rather_than_stream_unmonitored():
    class Planner:
        def close(self):
            pass
    h = SafetyHarness(limits(contact=contact_cfg()), kinematics=FakeKin())
    arm = MockArm(Cfg({"n_joints": 6, "home_q": START.tolist()}))
    arm.connect()
    with pytest.raises(SafetyViolation, match="contact monitor"):
        SafeArm(arm, h, motion_planner=Planner()).move_joints(START + [0.05, 0, 0, 0, 0, 0], 1.0)
    assert arm.commands == []


# ── contact zone: no re-planning into the same obstacle ──────────────────


def tripped_rig():
    arm, safe, h = rig(contact_cfg())
    arm.block_joint(0, at=START[0] + 0.04)
    with pytest.raises(ContactDetected):
        safe.move_joints(START + [0.15, 0, 0, 0, 0, 0], 2.0)
    arm.unblock_joints()
    return arm, safe, h


def test_a_contact_leaves_a_zone_around_the_tcp_where_it_stopped():
    arm, _safe, h = tripped_rig()
    (zone,) = h.contact_zones
    np.testing.assert_allclose(zone["center"], arm.get_state().q[:3], atol=1e-9)
    assert zone["radius_m"] == 0.05 and "joint 1" in zone["reason"]


def test_static_vetting_refuses_poses_inside_the_contact_zone():
    arm, _safe, h = tripped_rig()
    here = arm.get_state().q
    reason = h.vet_pose(here + [0.02, 0, 0, 0, 0, 0])
    assert reason and "contact" in reason and "reset_scene" in reason
    assert h.vet_pose(here + [0.0, 0.2, 0, 0, 0, 0]) is None


def test_the_arm_may_leave_the_zone_but_not_come_back():
    arm, safe, h = tripped_rig()
    here = arm.get_state().q
    zone = h.contact_zones[0]
    away = here - [0.12, 0, 0, 0, 0, 0]
    assert safe.move_joints(away, 2.0)                      # escape is always allowed
    n = len(arm.commands)
    # a planned route is vetted whole: refused before any command
    with pytest.raises(SkillError, match="contact zone"):
        safe.move_planned(here, 2.0)
    assert len(arm.commands) == n
    assert h.vet_step(away, away + [0.08, 0, 0, 0, 0, 0], 0.5) is not None   # 4 cm from it
    # a direct stream is stopped by approve() at the zone's edge
    with pytest.raises(SafetyViolation, match="contact zone"):
        safe.move_joints(here, 2.0)
    streamed = arm.commands[n:]
    assert streamed and all(np.linalg.norm(c[:3] - zone["center"]) > zone["radius_m"]
                            for c in streamed)


def test_clearing_the_zones_allows_the_spot_again():
    arm, safe, h = tripped_rig()
    here = arm.get_state().q
    assert safe.move_joints(here - [0.12, 0, 0, 0, 0, 0], 2.0)
    assert h.clear_contact_zones("operator cleared the clutter") == 1
    assert h.contact_zones == [] and h.vet_pose(here) is None
    assert safe.move_joints(here, 2.0)


def test_zone_radius_zero_disables_zones_but_not_the_stop():
    arm, safe, h = rig(contact_cfg(contact_zone_m=0.0))
    arm.block_joint(0)
    with pytest.raises(ContactDetected):
        safe.move_joints(START + [0.15, 0, 0, 0, 0, 0], 2.0)
    assert h.contact_zones == [] and h.last_contact is not None
