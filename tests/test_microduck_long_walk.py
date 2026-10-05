"""Long-walk candidate: bridge heading hold, its limits contract and the opt-in profile.

Software fixtures only; none of this is physical locomotion admission.
"""
from __future__ import annotations

import json
import math

import pytest
from test_mobile_bridge import Clock, _module, command, publish
from test_microduck_bridge_cli import cli, software_limits
from test_walk_distance_control import _SteppedDistanceMock
from mobile_support_fixture import support_contract


def _controller(kp=0.0, ki=0.0):
    clock = Clock()
    c = _module().MobileBridgeController(
        robot_id="duck", source="isolated-bridge", engine="newton", device="cuda:0",
        asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64,
        support_contract=support_contract(), max_linear_speed=0.2, max_angular_speed=0.8,
        max_duration_s=10.0, lease_s=0.5, max_state_age_s=0.5, clock=clock,
        heading_hold_kp=kp, heading_hold_ki=ki)
    return c, clock


def _yawed(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def test_default_is_open_loop_and_hello_says_so():
    c, _ = _controller()
    assert c.hello()["heading_hold"] is None
    publish(c, orientation_wxyz=_yawed(0.3))
    command(c)
    publish(c, step=2, sim_time=0.010, orientation_wxyz=_yawed(0.1))
    assert c.control_at(0.010) == (0.1, 0.0, 0.0)


def test_straight_command_holds_admission_heading_with_bounded_rate():
    c, _ = _controller(kp=2.0)
    assert c.hello()["heading_hold"] == {"kp": 2.0, "ki": 0.0}
    publish(c, orientation_wxyz=_yawed(0.3))
    command(c)
    publish(c, step=2, sim_time=0.010, orientation_wxyz=_yawed(0.2))  # drifted clockwise 0.1
    vx, vy, wz = c.control_at(0.010)
    assert (vx, vy) == (0.1, 0.0) and wz == pytest.approx(0.2)
    publish(c, step=3, sim_time=0.015, orientation_wxyz=_yawed(-0.5))  # large error is clipped
    assert c.control_at(0.015)[2] == pytest.approx(0.8)
    publish(c, step=4, sim_time=0.020, orientation_wxyz=_yawed(0.3 + 2 * math.pi - 0.05))  # wraps
    assert c.control_at(0.020)[2] == pytest.approx(0.1)


def test_integral_accumulates_once_per_simulation_time_with_anti_windup():
    c, _ = _controller(kp=1.0, ki=2.0)
    publish(c, orientation_wxyz=_yawed(0.0))
    command(c, duration_s=10.0)
    publish(c, step=2, sim_time=0.105, orientation_wxyz=_yawed(-0.1))  # error +0.1 for 0.1 s
    first = c.control_at(0.105)[2]
    assert first == pytest.approx(1.0 * 0.1 + 2.0 * 0.1 * 0.1)
    assert c.control_at(0.105)[2] == first  # a repeated read never integrates twice
    publish(c, step=3, sim_time=9.0, orientation_wxyz=_yawed(-0.1))
    assert c.control_at(9.0)[2] == pytest.approx(0.8)  # output clipped, integral bounded at limit/ki
    publish(c, step=4, sim_time=9.005, orientation_wxyz=_yawed(0.0))
    assert c.control_at(9.005)[2] == pytest.approx(2.0 * 0.4)


def test_turns_and_zero_commands_are_never_rewritten():
    c, _ = _controller(kp=2.0, ki=1.0)
    publish(c)
    command(c, vx=0.0, wz=0.5)
    publish(c, step=2, sim_time=0.010, orientation_wxyz=_yawed(0.4))
    assert c.control_at(0.010) == (0.0, 0.0, 0.5)
    c2, _ = _controller(kp=2.0)
    publish(c2)
    command(c2, vx=0.0, wz=0.0)
    publish(c2, step=2, sim_time=0.010, orientation_wxyz=_yawed(0.4))
    assert c2.control_at(0.010) == (0.0, 0.0, 0.0)


def test_nothing_is_commanded_after_completion_or_stop():
    c, _ = _controller(kp=2.0, ki=1.0)
    publish(c, orientation_wxyz=_yawed(0.0))
    command(c, duration_s=0.5)
    publish(c, step=2, sim_time=0.6, orientation_wxyz=_yawed(-0.3))
    assert c.control_at(0.6) == (0.0, 0.0, 0.0)  # past the admitted physical deadline
    publish(c, step=3, sim_time=0.605, orientation_wxyz=_yawed(-0.3))
    command(c, command_id="move-2", duration_s=5.0)
    c.stop(latch=True)
    assert c.control_at(0.61) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("kp,ki", [(-1.0, 0.0), (float("nan"), 0.0), (float("inf"), 0.0), (True, 0.0),
                                   (0.0, 1.0), (1.0, -0.5), (1.0, float("nan"))])
def test_invalid_gains_refused(kp, ki):
    with pytest.raises(ValueError):
        _controller(kp=kp, ki=ki)


def test_limits_accept_both_heading_gains_or_neither(tmp_path):
    p = tmp_path / "limits.json"
    for gains in ({}, {"heading_hold_kp": 4.0, "heading_hold_ki": 2.0}, {"heading_hold_kp": 4, "heading_hold_ki": 0}):
        p.write_text(json.dumps({**software_limits(), **gains}))
        assert cli().load_limits(p) == {**software_limits(), **gains}
    for gains in ({"heading_hold_kp": 4.0}, {"heading_hold_ki": 2.0}, {"heading_hold_kp": 0.0, "heading_hold_ki": 0.0},
                  {"heading_hold_kp": -4.0, "heading_hold_ki": 2.0}, {"heading_hold_kp": 4.0, "heading_hold_ki": -1.0},
                  {"heading_hold_kp": True, "heading_hold_ki": 2.0}, {"heading_hold_kp": 4.0, "heading_hold_ki": 2.0, "x": 1}):
        p.write_text(json.dumps({**software_limits(), **gains}))
        with pytest.raises(ValueError):
            cli().load_limits(p)


def _profile(name):
    from cascade.config import load_profile
    return load_profile("bases", name).as_dict()


def test_long_profile_is_opt_in_and_changes_only_declared_bounds():
    from cascade.agent.base_effects import _validate_limits
    slow, long = _profile("microduck_distance_native_slow"), _profile("microduck_distance_long")
    assert long["admission"] == "pending_long_walk_geometric_validation"
    assert long["model_identity_sha256"] is None and long["support_contract"] is None
    assert long["safety"] == {**slow["safety"], "max_duration_s": 25.0, "max_wall_duration_s": 190.0}
    assert long["distance_control"] == {**slow["distance_control"], "max_distance_m": 2.0,
                                        "max_lateral_drift_m": 0.10, "max_heading_drift_rad": 0.25}
    assert long["verifier"] == {**slow["verifier"], "max_wall_duration_s": 194.0,
                                "max_samples": math.ceil(194 / .02) + 1, "max_position_abs_m": 3.0,
                                "max_lateral_drift_m": 0.10, "max_heading_drift_rad": 0.25}
    assert long["turn_control"] == slow["turn_control"]
    _validate_limits(long["verifier"])
    for name in ("microduck_distance_candidate", "microduck_distance_native_slow"):
        assert _profile(name)["distance_control"]["max_distance_m"] == .1
        assert _profile(name)["safety"]["max_duration_s"] == 3.


def test_long_profile_walks_a_measured_metre_on_the_kinematic_fixture_only():
    from cascade.safety.base_harness import SafeBase
    long = _profile("microduck_distance_long")
    raw = _SteppedDistanceMock(wall_lease_s=2., dt_s=.002, auto_step=False)
    safe = SafeBase(raw, long["safety"], distance_control=long["distance_control"],
                    turn_control=long["turn_control"])
    assert not safe.walk_distance(2.5)["execution_ok"] and not raw.connected  # beyond the profile
    safe.connect()
    try:
        result = safe.walk_distance(1.0)
        assert result["execution_ok"], result
        assert result["measured_distance_m"] == pytest.approx(1.0, abs=long["distance_control"]["tolerance_m"])
        assert result["outcome"] != "confirmed"  # a kinematic mock never confirms physics
    finally:
        safe.disconnect()


def test_route_profile_extends_the_long_walk_to_six_metres_with_derived_wall_budgets():
    from cascade.agent.base_effects import _validate_limits
    long, route = _profile("microduck_distance_long"), _profile("microduck_distance_route")
    assert route["admission"] == "pending_route_geometric_validation"
    assert route["model_identity_sha256"] is None and route["support_contract"] is None
    # Physical command cap 60 s; wall from the retained twelve-robot owner attempt (17.2 wall s / physical s).
    assert route["safety"] == {**long["safety"], "max_duration_s": 60.0,
                               "max_wall_duration_s": float(math.ceil(60 * 17.2 + 2 * .5 + 1))}
    assert route["distance_control"] == {**long["distance_control"], "max_distance_m": 6.0,
                                         "max_lateral_drift_m": 0.30, "max_heading_drift_rad": 0.25}
    wall = route["safety"]["max_wall_duration_s"] + 4
    assert route["verifier"] == {**long["verifier"], "max_wall_duration_s": wall,
                                 "max_samples": math.ceil(wall / .02) + 1, "max_position_abs_m": 10.0,
                                 "max_lateral_drift_m": 0.30, "max_heading_drift_rad": 0.25}
    # The shared-world variant changes client sampling density only; deadlines stay the originals.
    shared = _profile("microduck_distance_route_shared")
    assert shared["admission"] == "pending_shared_route_geometric_validation"
    assert shared["safety"] == {**route["safety"], "poll_interval_s": 0.1}
    assert shared["verifier"] == {**route["verifier"], "sample_interval_s": 0.1, "max_samples": math.ceil(wall / .1) + 1}
    assert shared["distance_control"] == route["distance_control"] and shared["turn_control"] == route["turn_control"]
    for prof in (route, shared):
        assert prof["verifier"]["read_timeout_s"] == 0.5 and prof["verifier"]["max_sample_gap_s"] == 0.15
        assert prof["verifier"]["max_state_age_s"] == 0.5 and prof["safety"]["max_no_progress_s"] == 0.4
    _validate_limits(shared["verifier"])
    # Freshness and progress limits are the original ones, not relaxed for twelve robots.
    assert route["safety"]["max_state_age_s"] == 0.5 and route["safety"]["max_no_progress_s"] == 0.4
    assert route["turn_control"] == long["turn_control"]
    _validate_limits(route["verifier"])
    assert long["distance_control"]["max_distance_m"] == 2.0 and long["safety"]["max_duration_s"] == 25.0


def test_route_limits_file_matches_the_route_profile_and_holds_heading():
    import json
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    from isaac_microduck_bridge import load_limits
    limits = load_limits(repo / "configs/microduck/controller-limits-route.json")
    route = _profile("microduck_distance_route")
    assert limits["max_duration_s"] == route["safety"]["max_duration_s"]
    assert limits["max_action_wall_s"] == route["safety"]["max_wall_duration_s"]
    assert limits["max_state_age_s"] == route["safety"]["max_state_age_s"]
    assert limits["heading_hold_kp"] == 4.0 and limits["heading_hold_ki"] == 2.0
    assert limits["max_linear_speed"] == route["safety"]["max_vx"] >= route["distance_control"]["speed_m_s"]
    assert json.loads((repo / "configs/microduck/controller-limits-route.json").read_text()) == limits
