"""A contact reaches the skill result, the MCP tool result and the run trace,
and the runtime does not drive straight back into the obstacle.

Real mock runtime (build_runtime on the mock profiles, offline). The mock
arm's harness gets `safety.contact_monitor` enabled; `MockArm.block_joint`
puts a wall in a joint's way, the way clutter stopped the reBot's wrist on
2026-10-11 during a side-approach grasp descent.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from conftest import needs_pin

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.safety.contact import ContactMonitorConfig

pytestmark = needs_pin

CONTACT = {"enabled": True, "mode": "enforce", "track_tol_rad": 0.10, "lag_s": 0.25,
           "debounce_samples": 2, "contact_zone_m": 0.05}


@pytest.fixture
def rt(tmp_path):
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    arm.object_stop_frac = 0.5          # the mock jaws close on an object
    h = runtime.arm.harness
    h.limits = dataclasses.replace(h.limits, contact=ContactMonitorConfig.from_config(CONTACT))
    runtime._raw_test_arm = arm
    try:
        yield runtime
    finally:
        shutdown_runtime(runtime, arm)


def rows(rt):
    return [json.loads(x) for x in (rt.trace.run_dir / "trace.jsonl").read_text().splitlines()]


def away_from_home(rt):
    """Park somewhere move_home has to travel from (through the harness)."""
    home = np.asarray(rt.cfg.arm.home_q, dtype=float)
    q = home.copy()
    q[0] += 0.4
    assert rt.arm.move_joints(q, duration_s=2.0)
    rt.arm.harness.drain_contact_reports()
    return home, q


def wall_on_the_descent(rt, monkeypatch, fraction=0.3, short_rad=None):
    """Put a wall in the way of the grasp descent's biggest joint motion:
    at `fraction` of its travel, or `short_rad` before its target."""
    raw = rt._raw_test_arm
    move = rt.arm.move_joints
    walls = []

    def move_joints(q_target, *a, **kw):
        if kw.get("bias_compensate"):                   # the grasp descent
            q0 = raw.get_state().q
            dq = np.asarray(q_target, dtype=float) - q0
            j = int(np.argmax(np.abs(dq)))
            at = (q0[j] + fraction * dq[j] if short_rad is None
                  else float(q_target[j]) - short_rad * np.sign(dq[j]))
            raw.block_joint(j, at=at, direction=np.sign(dq[j]))
            walls.append(j)
        return move(q_target, *a, **kw)
    monkeypatch.setattr(rt.arm, "move_joints", move_joints)
    return walls


# ── the result says what happened ────────────────────────────────────────


def test_move_home_into_an_obstacle_fails_with_the_contact_reason(rt):
    _home, q = away_from_home(rt)
    rt._raw_test_arm.block_joint(0, at=q[0] - 0.1, direction=-1)
    res = rt.execute("move_home", {})
    assert res["ok"] is False
    assert res["error"].startswith("ContactDetected: contact detected at joint 1: tracking error")
    assert "something is blocking the arm" in res["error"]
    assert res["contact"]["joint"] == 1 and res["contact"]["hold_q"]
    assert "reset_scene" in res["hint"] and "do not retry" in res["hint"].lower()
    measured = rt._raw_test_arm.get_state().q
    assert measured[0] == pytest.approx(q[0] - 0.1)
    np.testing.assert_allclose(rt._raw_test_arm.commands[-1], measured)   # held, not pressing
    row = rows(rt)[-1]
    assert row["skill"] == "move_home"
    (report,) = row["context"]["contact_monitor"]
    assert report["tripped"] and report["motion"] == "move_joints"


def test_the_contact_reaches_the_mcp_tool_result(rt):
    from cascade.apps.mcp_server import McpSkillServer

    _, q = away_from_home(rt)
    rt._raw_test_arm.block_joint(0, at=q[0] - 0.1, direction=-1)
    server = McpSkillServer()
    server._runtime = rt
    out = server.call_tool("move_home", {})
    assert out["isError"] is True
    text = out["content"][-1]["text"]
    assert "contact detected at joint 1" in text and "something is blocking the arm" in text
    assert json.loads(text)["contact"]["joint"] == 1


def test_motions_without_a_contact_still_log_their_tracking_to_the_trace(rt):
    away_from_home(rt)
    res = rt.execute("move_home", {})
    assert res["ok"] is True
    (report,) = rows(rt)[-1]["context"]["contact_monitor"]
    assert report["tripped"] is None and len(report["max_track_err_rad"]) == 6


@pytest.mark.parametrize(("skill", "args", "holding"), [
    ("move_relative", {"direction": "left", "distance_m": 0.05}, None),
    ("place_at", {"x": 0.3, "y": 0.1}, "red cube"),
])
def test_other_motion_skills_fail_with_the_contact_reason(rt, skill, args, holding):
    """The first joint that moves meets a wall 2 mrad later."""
    raw = rt._raw_test_arm
    q0 = raw.get_state().q.copy()
    send = raw.send_joint_target

    def send_joint_target(q):
        dq = np.asarray(q, dtype=float) - q0
        if not raw._blocked and np.max(np.abs(dq)) > 1e-4:
            j = int(np.argmax(np.abs(dq)))
            raw.block_joint(j, at=q0[j] + 0.002 * np.sign(dq[j]), direction=np.sign(dq[j]))
        return send(q)
    raw.send_joint_target = send_joint_target
    rt.held_object = holding
    res = rt.execute(skill, args)
    assert res["ok"] is False
    assert res["error"].startswith("ContactDetected: contact detected at joint")
    assert "something is blocking the arm" in res["error"]
    assert res["contact"]["hold_q"] and "reset_scene" in res["hint"]


# ── grasp descent: the incident ──────────────────────────────────────────


def test_a_light_press_at_the_end_of_the_descent_still_grasps(rt, monkeypatch):
    """Fingertips touching the table/object just short of the grasp pose
    (within the settle tolerance) is today's grasp contact: no trip, the
    jaws close as before."""
    walls = wall_on_the_descent(rt, monkeypatch, short_rad=0.02)
    res = rt.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert walls
    assert res["ok"] is True and res.get("held") == "red cube", res
    assert not rt.arm.harness.contact_zones
    reports = rows(rt)[-1]["context"]["contact_monitor"]
    assert all(r["tripped"] is None for r in reports)


def test_grasp_descent_into_clutter_stops_retreats_and_reports(rt, monkeypatch):
    walls = wall_on_the_descent(rt, monkeypatch)
    res = rt.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert walls, "the grasp never reached its descent"
    assert res["ok"] is False
    assert "contact detected at joint" in res["error"]
    assert "pre-close retreat completed" in res["error"]       # backed out of it
    assert res["contact"]["joint"] == walls[0] + 1
    assert rt.arm.harness.contact_zones                         # the spot is fenced


def test_pick_and_place_does_not_retry_into_the_obstacle(rt, monkeypatch):
    walls = wall_on_the_descent(rt, monkeypatch)
    res = rt.execute("pick_and_place", {"object": "red cube"})
    assert res["ok"] is False
    assert len(walls) == 1, f"retried the grasp {len(walls)} times into the same obstacle"
    assert "grasp failed after 1 attempts" in res["error"]       # no re-home/re-scan/re-plan
    assert "contact detected" in res["error"]
    assert res["outcome"] == "stuck"
    assert "blocked" in res["ask"].lower() and "reset" in res["ask"].lower()


def test_a_second_grasp_refuses_the_fenced_spot_without_moving_into_it(rt, monkeypatch):
    wall_on_the_descent(rt, monkeypatch)
    first = rt.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert "contact detected" in first["error"]
    rt._raw_test_arm.unblock_joints()
    zone = rt.arm.harness.contact_zones[0]
    n = len(rt._raw_test_arm.commands)
    second = rt.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert second["ok"] is False
    for q in rt._raw_test_arm.commands[n:]:
        tcp = rt.kin.fk(q)[:3, 3]
        assert np.linalg.norm(tcp - zone["center"]) > zone["radius_m"] - 1e-6


@pytest.mark.parametrize("slipped", [False, True])
def test_a_contact_while_placing_is_not_retried_either(rt, monkeypatch, slipped):
    """Blocked on the way to the drop zone: no second place attempt, and --
    if the jaws lost the object in the collision -- no fresh grasp either."""
    from cascade.types import ContactDetected

    raw = rt._raw_test_arm
    move = rt.arm.move_joints
    walls, descents = [], []

    def move_joints(q_target, *a, **kw):
        if kw.get("bias_compensate") and not rt.held_object:
            descents.append(1)
        if rt.held_object and not walls:             # first motion of the carry
            q0 = raw.get_state().q
            dq = np.asarray(q_target, dtype=float) - q0
            j = int(np.argmax(np.abs(dq)))
            raw.block_joint(j, at=q0[j] + 0.3 * dq[j], direction=np.sign(dq[j]))
            walls.append(j)
        try:
            return move(q_target, *a, **kw)
        except ContactDetected:
            if slipped:
                rt.held_object = None                # knocked out of the jaws
            raise
    monkeypatch.setattr(rt.arm, "move_joints", move_joints)
    res = rt.execute("pick_and_place", {"object": "red cube"})
    assert walls and res["ok"] is False and res["stage"] == "place"
    assert "place failed after 1 attempts" in res["error"] and "contact detected" in res["error"]
    assert len(descents) == 1, "grasped again after a contact while placing"
    assert res["outcome"] == "stuck"
    assert "blocked" in res["ask"].lower()
    assert ("still holding" in res["ask"].lower()) is (not slipped)


def test_reset_scene_clears_the_contact_zones(rt, monkeypatch):
    wall_on_the_descent(rt, monkeypatch)
    rt.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    rt._raw_test_arm.unblock_joints()
    assert rt.arm.harness.contact_zones
    res = rt.execute("reset_scene", {})
    assert res["contact_zones_cleared"] == 1
    assert rt.arm.harness.contact_zones == []
