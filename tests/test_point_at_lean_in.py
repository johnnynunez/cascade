"""`point_at` leans in when no top-down pose clears the object, and checks its route.

#369 made point_at keep the TCP and the closed jaws `safety.min_clearance_m`
(0.03) above the object's top on the IK-solved pose. The SO-101's top-down
ceiling (0.04 m) then refused every object, and the reBot every top above
0.11 m.

Now, when no top-down rung clears, point_at aims the tilted gripper at the
centre of the object's top from the arm's side: approach a = cos(phi)(-z) +
sin(phi) u, with u the horizontal direction from the arm's first (pan) joint
to the aim point, TCP = aim - L a, phi in {15, 25, 35, 45} deg and L 4..10 cm,
smallest tilt first. A pose is accepted only on its solved FK: every tool
sample (the TCP and the jaw box at <= 5 mm, faces included) keeps the margin
from the object's box extruded to the table and from the table, every joint
origin the harness checks keeps it from the box, and the approach ray passes
within 1 cm of the aim point. The ROUTE to every pointing pose (top-down
too) is checked by FK samples along the joint segment; a route that fails
gets a rise-first waypoint; failing that, point_at refuses before any arm or
jaw command.

Oracles are independent of the implementation: the PHYSICS prop
(`sim.demo_scene.prop_pose_from_camera`, the box MuJoCo builds from the same
pixels) and the profile's own sweep box, sampled here at 4 mm. The mock
camera renders the prop one pixel short of the physics box on one side
(0.9 mm at 0.55 m depth), so clearances from the physics box allow
`PIXEL_M`; the code itself keeps the full margin from what it observes.
"""

from __future__ import annotations

import dataclasses
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import REPO, SO101_URDF, needs_pin, needs_pin_so101

from cascade.control.mock_arm import MockArm

#: safety.min_clearance_m in configs/demo.yaml (no arm profile overrides it)
MARGIN = 0.03
#: one mock-camera pixel at the prop's top-face depth (0.55 m / fx 600)
PIXEL_M = 0.001
LEAN_TILTS = (15, 25, 35, 45)


class _RecArm(MockArm):
    """MockArm that records every jaw command in the rig log."""

    def __init__(self, cfg, kinematics, log):
        super().__init__(cfg, kinematics)
        self.log = log

    def set_gripper(self, pos, effort=1.0):
        self.log.jaws.append(float(pos))
        self.log.jaws_at.append(sum(len(a.commands) for a in self.log.arms))
        super().set_gripper(pos, effort)


def cube_box_px(cam, x, y, side, height):
    """The mock camera's `box_px` that puts a `side` x `side` x `height` prop
    at base (x, y): the inverse of `prop_pose_from_camera`."""
    T = np.asarray(cam.get("extrinsics").get("T"), dtype=float)
    inv = np.linalg.inv(T)
    us, vs = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = inv @ np.array([x + sx * side / 2, y + sy * side / 2, height, 1.0])
            us.append(320 + 600 * p[0] / p[2])
            vs.append(240 + 600 * p[1] / p[2])
    return [int(round(min(us))), int(round(min(vs))), int(round(max(us))), int(round(max(vs)))]


def place_prop(cfg, x, y, side, height):
    cfg.camera._data["box_px"] = cube_box_px(cfg.camera, x, y, side, height)
    cfg.camera._data["box_height_m"] = float(height)
    prop_cam = cfg.arm.get("mj_prop_from_camera")
    if hasattr(prop_cam, "_data") and prop_cam._data is not cfg.camera._data:
        prop_cam._data["box_px"] = cfg.camera._data["box_px"]
        prop_cam._data["box_height_m"] = float(height)


def at(r, az):
    return r * math.cos(az), r * math.sin(az)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.skills import pointing

    # the suite runs on slower, loaded CI runners: the planning budget's own
    # tests set it back to the shipped 3 s and drive it with a fake clock
    monkeypatch.setattr(pointing, "PLAN_BUDGET_S", 30.0)
    built = []

    def build(profile, camera="mock_small", prop=None):
        cfg = load_demo_config(camera=camera, arm=profile, llm="mock")
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        if prop is not None:
            place_prop(cfg, *prop)
        r = SimpleNamespace(cfg=cfg, jaws=[], jaws_at=[], arms=[])

        def make_arm(acfg, kinematics=None):
            arm = _RecArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / f"{profile}-{camera}-{len(built)}", lazy_arm=True)
        built.append((r.runtime, arm))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and r.runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _commands(r) -> list:
    return [np.asarray(q, dtype=float) for arm in r.arms for q in arm.commands]


def _truth_box(cfg):
    """The physics prop extruded to the table: (lo, hi) corners."""
    from cascade.sim.demo_scene import prop_pose_from_camera

    centre, half = prop_pose_from_camera(cfg.camera)
    lo, hi = centre - half, centre + half
    lo[2] = float(cfg.safety.get("table_z", 0.0))
    return lo, hi


def _box_distance(points, lo, hi) -> np.ndarray:
    return np.linalg.norm(np.maximum(np.maximum(lo - points, 0.0), points - hi), axis=1)


def _tool_grid(cfg) -> np.ndarray:
    """Oracle: the TCP and the profile's open-jaw sweep box (origin at the
    TCP, +X closing, +Y third, +Z approach) on a 4 mm grid, TCP frame."""
    sweep = cfg.grasp.graspgenx.sweep
    ext = np.asarray(sweep.get("extents_open"), float)
    off = np.asarray(sweep.get("offset_open", [0.0, 0.0, 0.0]), float)
    a, j, t = {"down_open": (0, 1, 2), "open_down": (2, 0, 1),
               "third_open_down": (2, 1, 0)}[str(cfg.arm.get("tool_axis_order", "down_open"))]
    axes = np.eye(3)[:, [j, t, a]]
    lines = [np.linspace(-h, h, max(2, int(np.ceil(2 * h / 0.004)) + 1)) for h in ext / 2.0]
    grid = np.stack(np.meshgrid(*lines, indexing="ij"), axis=-1).reshape(-1, 3) + off
    return np.vstack([np.zeros(3), grid @ axes.T])


def _approach_col(cfg) -> int:
    return {"down_open": 0, "open_down": 2, "third_open_down": 2}[str(cfg.arm.get("tool_axis_order", "down_open"))]


def _clearances(r, q):
    """(tool-to-prop, lowest tool z above the table, joint-origins-to-prop)."""
    lo, hi = _truth_box(r.cfg)
    T = r.runtime.kin.fk(q)
    tool = _tool_grid(r.cfg) @ T[:3, :3].T + T[:3, 3]
    links = r.runtime.kin.link_positions(q)[1:]
    table = float(r.cfg.safety.get("table_z", 0.0))
    return (float(_box_distance(tool, lo, hi).min()), float(tool[:, 2].min()) - table,
            float(_box_distance(links, lo, hi).min()))


def _assert_every_command_keeps_the_margin(r):
    """Every streamed arm target (the min-jerk route, not only its end). An
    arm that STARTS inside the margin (the SO-101's home hovers 1 cm over the
    mock cube) may only move away until it is out, then never back in."""
    cmds = _commands(r)
    assert cmds, "nothing was commanded"
    rows = [_clearances(r, q) for q in cmds]
    worst = [min(c[0] + PIXEL_M, c[1], c[2] + PIXEL_M) for c in rows]
    out = next((i for i, w in enumerate(worst) if w >= MARGIN), None)
    assert out is not None, f"the route never left the margin (best {max(worst):.4f} m)"
    assert all(b >= a - 1e-4 for a, b in zip(worst[:out], worst[1:out + 1])), \
        f"closed in while inside the margin: {np.round(worst[:out + 1], 4).tolist()}"
    after = [min(c) for c in zip(*rows[out:])]
    assert after[0] >= MARGIN - PIXEL_M, f"the jaws came {after[0]:.4f} m from the prop"
    assert after[1] >= MARGIN, f"the jaws came {after[1]:.4f} m above the table"
    assert after[2] >= MARGIN - PIXEL_M, f"a joint origin came {after[2]:.4f} m from the prop"


def _aim_and_tilt(r, q):
    """Distance from the prop's top-face centre to the approach ray, and the
    approach's tilt from straight down (deg)."""
    from cascade.sim.demo_scene import prop_pose_from_camera

    centre, half = prop_pose_from_camera(r.cfg.camera)
    aim = np.array([centre[0], centre[1], centre[2] + half[2]])
    T = r.runtime.kin.fk(q)
    a = T[:3, _approach_col(r.cfg)]
    v = aim - T[:3, 3]
    s = max(float(v @ a), 0.0)
    return float(np.linalg.norm(v - s * a)), float(np.degrees(np.arccos(np.clip(-a[2], -1.0, 1.0))))


def _refused_untouched(r, out):
    assert out["ok"] is False, out
    assert _commands(r) == [] and r.jaws == [], "point_at sent a command before refusing"
    assert "nothing was moved" in out["error"], out


# ── SO-101: lean in over its annulus, refuse inside it ─────────────────────

SO101_ADMISSIBLE = [(side, rr, az) for side in (0.035, 0.05) for rr in (0.18, 0.22, 0.26)
                    for az in (0.0, 0.5, -0.5)]


@needs_pin_so101
@pytest.mark.parametrize("camera", ["mock_small", "mock"])
def test_so101_points_at_the_default_props_by_leaning_in(rig, camera):
    """#369 left the SO-101 refusing these (top-down ceiling 0.04 m): the
    35 x 35 x 50 mm mock_small prop at r 0.20, the mock 55 x 73 x 50 mm box at
    r 0.29. The property is asserted on real commands: one jaw close, a
    streamed route, every command clear (after the escape from home, which
    holds the jaw box 1 cm over the mock_small prop)."""
    r = rig("so101_mock", camera)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and out["lean_deg"] in LEAN_TILTS, out
    assert len(r.jaws) == 1 and len(_commands(r)) > 10, (r.jaws, len(_commands(r)))
    _assert_every_command_keeps_the_margin(r)


@needs_pin_so101
@pytest.mark.parametrize("side, radius, az", SO101_ADMISSIBLE)
def test_so101_leans_in_to_point_at_a_cube_in_its_annulus(rig, side, radius, az):
    r = rig("so101_mock", prop=(*at(radius, az), side, side))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert out["lean_deg"] in LEAN_TILTS, out
    assert len(r.jaws) == 1 and r.jaws[0] == pytest.approx(float(r.cfg.arm.gripper.get("closed_pos")))
    _assert_every_command_keeps_the_margin(r)
    miss, tilt = _aim_and_tilt(r, _commands(r)[-1])
    assert miss <= 0.01 + PIXEL_M, f"the approach ray misses the top's centre by {miss:.4f} m"
    assert tilt == pytest.approx(out["lean_deg"], abs=0.5)


@needs_pin_so101
@pytest.mark.parametrize("side", [0.035, 0.05])
@pytest.mark.parametrize("az", [0.0, 0.5])
def test_so101_refuses_a_cube_inside_its_annulus_before_any_command(rig, side, az):
    """r = 0.14: no lean-in pose keeps the margin (the reviewer's 150k-sample
    search found none either)."""
    r = rig("so101_mock", prop=(*at(0.14, az), side, side))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "cannot point at 'red cube' on this arm: from above," in out["error"], out
    assert "leaning in" in out["error"], out


# ── reBot (mock URDF): tall bottles ─────────────────────────────────────────


@needs_pin
@pytest.mark.parametrize("height", [0.15, 0.20])
@pytest.mark.parametrize("radius", [0.30, 0.40])
def test_rebot_leans_in_to_point_at_a_bottle(rig, height, radius):
    """Top-down poses clear only tops below 0.11 m (ceiling 0.14 m)."""
    r = rig("mock", camera="mock", prop=(radius, 0.0, 0.06, height))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert 15 <= out["lean_deg"] <= 30, out
    _assert_every_command_keeps_the_margin(r)
    miss, _ = _aim_and_tilt(r, _commands(r)[-1])
    assert miss <= 0.01 + PIXEL_M


@needs_pin
def test_a_top_down_pose_that_clears_never_leans_and_goes_straight(rig, monkeypatch):
    """The default mock scene: today's top-down hover, one leg, no lean-in
    IK at all (the commands are main's: tests/test_point_at_clearance.py)."""
    r = rig("mock", camera="mock")
    calls = []
    real = r.runtime.kin.ik_approach
    monkeypatch.setattr(r.runtime.kin, "ik_approach", lambda *a, **k: calls.append(a) or real(*a, **k))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert calls == [] and "lean_deg" not in out and "rose_first" not in out, out
    assert len(r.jaws) == 1 and _commands(r)


@needs_pin
def test_the_lean_in_family_runs_smallest_tilt_first(rig, monkeypatch):
    """The first admissible candidate wins; the search order is tilt, then
    distance (a 15 cm bottle at r 0.30 takes the first: 15 deg, 4 cm)."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.15))
    asked = []
    real = r.runtime.kin.ik_approach

    def spy(position, approach, *a, **k):
        asked.append((round(float(np.degrees(np.arccos(-approach[2]))), 3), position.copy()))
        return real(position, approach, *a, **k)

    monkeypatch.setattr(r.runtime.kin, "ik_approach", spy)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and out["lean_deg"] == 15, out
    assert asked[0][0] == pytest.approx(15.0)


@needs_pin
def test_a_lean_in_pose_the_harness_would_refuse_is_skipped(rig, monkeypatch):
    """`SafetyHarness.vet_pose` vetoes the first admissible candidate (the
    15 deg one); point_at takes the next, never the vetoed pose."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.15))
    harness = r.runtime.arm.harness
    real = harness.vet_pose
    vetoed = []

    def vet(q, *a, **k):
        T = r.runtime.kin.fk(q)
        tilt = float(np.degrees(np.arccos(-T[2, _approach_col(r.cfg)])))
        if abs(tilt - 15.0) < 0.5:
            vetoed.append(np.asarray(q, float).copy())
            return "test veto: keep-out"
        return real(q, *a, **k)

    monkeypatch.setattr(harness, "vet_pose", vet)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and out["lean_deg"] == 25, out
    assert vetoed and not any(np.allclose(c, v) for c in _commands(r) for v in vetoed)


@needs_pin
def test_a_rise_waypoint_the_harness_would_refuse_is_skipped(rig, monkeypatch):
    """Every waypoint vetoed: no rise, so the route through the object is
    refused before any command."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.10))
    _park(r, [0.18, 0.0, 0.06], yaw=math.pi / 2)
    monkeypatch.setattr(r.runtime.arm.harness, "vet_pose", lambda q, *a, **k: "test veto")

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "no rise-first waypoint clears it" in out["error"], out


# ── the acceptance rule, one condition at a time ───────────────────────────


@needs_pin
@pytest.mark.parametrize("bad", ["tool", "table", "links", "aim"])
def test_a_lean_in_pose_failing_one_condition_is_never_commanded(rig, monkeypatch, bad):
    """Each condition alone vetoes: the solved pose is moved so that exactly
    one fails, and point_at refuses before any command."""
    from cascade.skills import pointing

    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.15))
    real = pointing.pose_clearance

    def worse(*a, **k):
        c = real(*a, **k)
        return dataclasses.replace(c, **{
            "tool": {"tool_m": MARGIN - 1e-4}, "table": {"table_m": MARGIN - 1e-4},
            "links": {"links_m": MARGIN - 1e-4}, "aim": {"aim_m": pointing.AIM_TOLERANCE_M + 1e-4}}[bad])

    monkeypatch.setattr(pointing, "pose_clearance", worse)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)


def test_pose_clearance_measures_the_jaw_grid_not_its_corners():
    """A tilted jaw box's lowest point inside the column is not a corner:
    the object's top edge pokes into the middle of the box's face."""
    from cascade.skills import pointing

    column = pointing.ObjectColumn(footprints=(np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]),),
                                   tops=(0.10,), z_lo=0.0, aim=np.array([0.0, 0.0, 0.10]))
    # a 6 cm-long sample row 1 cm over the top, its ends 3 cm past the edges
    row = np.c_[np.linspace(-0.13, 0.13, 53), np.zeros(53), np.full(53, 0.11)]
    assert column.distance(row).min() == pytest.approx(0.01)
    assert column.distance(row[[0, -1]]).min() == pytest.approx(math.hypot(0.03, 0.01))


def test_the_column_extrudes_the_box_to_the_table_and_covers_the_points():
    from cascade.skills import pointing

    fix = SimpleNamespace(position=np.array([0.3, 0.0, 0.10]), extent=np.array([0.04, 0.04, 0.02]),
                          axes=np.eye(3), points=np.array([[0.3, 0.05, 0.11], [0.3, 0.0, 0.09]]))
    column = pointing.object_column(fix, table_z=0.0, top=0.11)
    # under the box, down to the table, is inside
    assert column.distance(np.array([[0.3, 0.0, 0.001]]))[0] <= 0.0
    # the box's own corners count where no point was observed
    assert column.distance(np.array([[0.318, -0.018, 0.05]]))[0] <= 0.0
    # a stray point outside the box widens the footprint
    assert column.distance(np.array([[0.3, 0.05, 0.05]]))[0] <= 0.0
    # above the top by 2 cm
    assert column.distance(np.array([[0.3, 0.0, 0.13]]))[0] == pytest.approx(0.02)
    np.testing.assert_allclose(column.aim, [0.3, 0.0, 0.11])


def test_an_object_over_the_target_is_part_of_the_column():
    """A cup (6 cm square, top 0.09 m) standing over a 4 cm cube (top 0.05):
    its own prism counts, wider and taller than the cube, and the aim is
    the centre of its top."""
    from cascade.skills import pointing

    fix = SimpleNamespace(position=np.array([0.3, 0.0, 0.025]), extent=np.array([0.04, 0.04, 0.05]),
                          axes=np.eye(3), points=np.array([[0.3, 0.0, 0.05]]))
    cup = np.array([[0.27, -0.03], [0.33, -0.03], [0.33, 0.03], [0.27, 0.03]])
    column = pointing.object_column(fix, table_z=0.0, top=0.09, occluders=[(cup, 0.09)])
    # beside the cube but inside the cup's wall: inside
    assert column.distance(np.array([[0.325, 0.0, 0.07]]))[0] <= 0.0
    # 1 cm outside the cup's wall, at the cube's height
    assert column.distance(np.array([[0.34, 0.0, 0.04]]))[0] == pytest.approx(0.01)
    np.testing.assert_allclose(column.aim, [0.3, 0.0, 0.09])


# ── the route: rise first, or refuse before any command ────────────────────


def _park(r, tcp, yaw=0.0):
    """Put the (mock) arm at a top-down TCP pose before the call."""
    from cascade.grasping.obb_grasp import _yaw_rotation
    from cascade.types import make_transform

    runtime = r.runtime
    runtime.arm.get_state()
    arm = r.arms[0]
    T = make_transform(_yaw_rotation(yaw, axis_order=runtime._tool_axis_order), np.asarray(tcp, float))
    ik = runtime.kin.ik(T, arm.get_state().q)
    assert ik.success, "test setup: the parking pose must solve"
    arm._q = np.asarray(ik.q, float).copy()
    arm.commands.clear()
    return ik.q


def _straight_min_clearance(r, q0, q1, n=200):
    return min(_clearances(r, q0 + s * (q1 - q0))[0] for s in np.linspace(0.0, 1.0, n))


@needs_pin
def test_a_direct_route_through_the_object_rises_first(rig, monkeypatch):
    """The reviewer's probe: the arm low beside a 10 cm object, 12 cm toward
    the base; the straight joint segment to the hover sweeps the jaws through
    the object's top edge. point_at rises first, and every streamed command
    keeps the margin."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.10))
    q0 = _park(r, [0.18, 0.0, 0.06], yaw=math.pi / 2)    # jaws across: 5 cm clear
    goal = {}
    real_move = r.runtime._move_joints

    def move(q, *a, **k):
        goal.setdefault("all", []).append(np.asarray(q, float).copy())
        return real_move(q, *a, **k)

    monkeypatch.setattr(r.runtime, "_move_joints", move)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and out.get("rose_first") is True, out
    assert len(goal["all"]) == 2
    # the precondition that makes this a route test: going straight would not clear
    assert _straight_min_clearance(r, q0, goal["all"][-1]) < MARGIN - PIXEL_M
    _assert_every_command_keeps_the_margin(r)


@needs_pin
def test_no_rise_waypoint_that_clears_refuses_before_any_command(rig, monkeypatch):
    from cascade.skills import pointing

    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.10))
    _park(r, [0.18, 0.0, 0.06], yaw=math.pi / 2)
    monkeypatch.setattr(pointing, "RISE_WAYPOINTS", ())

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "no rise-first waypoint clears it" in out["error"], out


@needs_pin
def test_a_route_that_only_dips_mid_segment_is_caught(rig, monkeypatch):
    """Both ends clear; only interior samples do not. Without the rise the
    route check alone stands between the jaws and the object."""
    from cascade.skills import pointing

    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.10))
    q0 = _park(r, [0.18, 0.0, 0.06], yaw=math.pi / 2)
    monkeypatch.setattr(pointing, "RISE_WAYPOINTS", ())
    seen = []
    real = pointing.segment_violation

    def spy(kin, a, b, *rest, **k):
        reason = real(kin, a, b, *rest, **k)
        seen.append((np.asarray(a).copy(), np.asarray(b).copy(), reason))
        return reason

    monkeypatch.setattr(pointing, "segment_violation", spy)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    a, b, reason = seen[0]
    np.testing.assert_allclose(a, q0)
    assert reason is not None and "% of the way" in reason
    assert min(_clearances(r, a)[0], _clearances(r, b)[0]) >= MARGIN - PIXEL_M, "both ends clear"


@needs_pin
def test_a_start_inside_the_margin_only_moves_away(rig):
    """TCP parked 2 cm over a 5 cm top: the straight rise to today's hover
    (0.09 m) moves away the whole time, so it is the route."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.05))
    _park(r, [0.30, 0.0, 0.07])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and "rose_first" not in out, out
    _assert_every_command_keeps_the_margin(r)


class _SlideKin:
    """FK = a pure translation by q (3 joints); no joint origins to check.
    Every point moves exactly |dq| per unit of each joint (`lever_arms`)."""

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q, dtype=float)[:3]
        return T

    def link_positions(self, q):
        return np.zeros((1, 3))

    def lever_arms(self, q, tool):
        return np.ones(3), np.zeros((0, 3))


def _box_column():
    from cascade.skills import pointing

    square = np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]])
    return pointing.ObjectColumn(footprints=(square,), tops=(0.05,), z_lo=0.0, aim=np.array([0.0, 0.0, 0.05]))


@pytest.mark.parametrize("start, goal, from_inside, ok", [
    ([0.0, 0.0, 0.06], [0.0, 0.0, 0.10], True, True),      # inside, straight up and out
    ([0.0, 0.0, 0.06], [0.0, 0.0, 0.10], False, False),    # the same from a waypoint: not allowed
    ([0.0, 0.0, 0.07], [0.0, 0.0, 0.06], True, False),     # inside and closing in
    ([0.0, 0.0, 0.06], [0.2, 0.0, 0.06], True, True),      # inside, sliding off the top: never closer
    ([0.04, 0.0, 0.06], [0.0, 0.0, 0.06], True, False),    # inside, sliding in over the top: closer
    ([-0.1, 0.0, 0.07], [0.1, 0.0, 0.07], False, False),   # both ends clear, 2 cm over the top mid-way
    ([-0.1, 0.0, 0.07], [0.1, 0.0, 0.07], True, False),    # the same from where the arm is
    ([-0.1, 0.0, 0.09], [0.1, 0.0, 0.09], False, True),    # 4 cm over the top all the way
    ([0.0, 0.0, 0.06], [0.0, 0.0, 0.20], True, True),
    # starting INSIDE the object: deeper is closing in, out is escaping
    ([0.0, 0.0, 0.04], [0.0, 0.0, 0.03], True, False),     # 1 cm deep, going 2 cm deep
    ([0.015, 0.0, 0.03], [0.0, 0.0, 0.03], True, False),   # inside, sliding to the middle
    ([0.0, 0.0, 0.04], [0.0, 0.0, 0.10], True, True),      # inside, straight up and out
    ([0.015, 0.0, 0.03], [0.1, 0.0, 0.03], True, True),    # inside, sideways out
])
def test_the_route_rule_on_a_straight_segment(start, goal, from_inside, ok):
    from cascade.skills import pointing

    reason = pointing.segment_violation(_SlideKin(), np.array(start), np.array(goal), _box_column(),
                                        np.zeros((1, 3)), MARGIN, 0.0, from_inside=from_inside)
    assert (reason is None) is ok, reason


def test_the_least_distance_is_found_exactly_without_measuring_every_point():
    """`min_distance` (bounding-box first) equals `distance(...).min()` to the
    bit, inside and outside, with an object stacked over the target too."""
    from cascade.skills import pointing

    rng = np.random.default_rng(378)
    for _ in range(200):
        square = rng.uniform(0.02, 0.05) * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) + rng.uniform(-0.1, 0.1, 2)
        cup = square * rng.uniform(0.5, 1.5) + rng.uniform(-0.03, 0.03, 2)
        column = pointing.ObjectColumn(footprints=(square, cup), tops=tuple(rng.uniform(0.03, 0.2, 2)), z_lo=0.0,
                                       aim=np.zeros(3))
        points = rng.normal(0.0, rng.uniform(0.02, 0.3), size=(int(rng.integers(1, 3000)), 3)) + [0.0, 0.0, 0.08]
        assert column.min_distance(points) == float(column.distance(points).min())
    points[5] = np.nan
    assert math.isnan(column.min_distance(points))
    # where only the bound's order decides: 64 points inside a diamond's box
    # 7 mm off the diamond (their bound -2 cm), one 1 mm off its box ...
    diamond = pointing.ObjectColumn(footprints=(np.array([[0.05, 0.0], [0.0, 0.05], [-0.05, 0.0], [0.0, -0.05]]),),
                                    tops=(0.1,), z_lo=0.0, aim=np.zeros(3))
    corner = np.array([0.03, 0.03, 0.05]) + np.linspace(0.0, 1e-4, 64)[:, None] * [1.0, -1.0, 0.0]
    points = np.vstack([corner, [0.051, 0.0, 0.05]])
    assert diamond.min_distance(points) == pytest.approx(0.001) == float(diamond.distance(points).min())
    # ... and 64 points 3 mm off a box, one 5 mm inside it
    outside = np.column_stack([np.full(64, 0.023), np.linspace(-0.015, 0.015, 64), np.full(64, 0.03)])
    points = np.vstack([outside, [0.015, 0.0, 0.03]])
    assert _box_column().min_distance(points) == pytest.approx(-0.005) == float(_box_column().distance(points).min())


def test_the_column_distance_is_signed_inside():
    """Inside the object the distance is minus the depth to its nearest side
    or top (the table is no way out): going deeper is visible."""
    col = _box_column()
    d = col.distance(np.array([[0.0, 0.0, 0.04], [0.0, 0.0, 0.01], [0.015, 0.0, 0.03], [0.0, 0.0, 0.08]]))
    assert d == pytest.approx([-0.01, -0.02, -0.005, 0.03])


def test_escaping_is_judged_pair_by_pair():
    """Two tool samples: A at the TCP holds 5 mm under the table (the
    smallest clearance, flat); B, 6 cm behind and 4 cm above it, starts 2 cm
    from the object and slides to touch it. The smallest clearance never
    shrinks, yet the jaws close in on the object: refused, that pair named."""
    from cascade.skills import pointing

    tool = np.array([[0.0, 0.0, 0.0], [-0.06, 0.0, 0.04]])
    start, goal = np.array([0.10, 0.0, -0.005]), np.array([0.08, 0.0, -0.005])
    reason = pointing.segment_violation(_SlideKin(), start, goal, _box_column(), tool, MARGIN, 0.0,
                                        from_inside=True)
    assert reason is not None and reason.startswith("the jaws 0.0") and "from it" in reason, reason
    assert "closing in from inside the margin" in reason, reason


class _SlideIK(_SlideKin):
    """_SlideKin that solves any TCP position (orientation ignored)."""

    def ik(self, T, q0, *a, **k):
        return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=True)

    def ik_approach(self, position, approach, q0, axis, *a, **k):
        return SimpleNamespace(q=np.asarray(position, float).copy(), success=True)


def test_every_rise_leg_is_preflighted_by_the_harness():
    """Going straight dips 2.6 cm over the top's edge; the lowest rise (TCP
    0.07 m) clears the object but the harness's route vet refuses legs ending
    under 0.095 m (a stand-in for a workspace box it would stop mid-way): the
    waypoint is the first one both accept, and both legs were asked."""
    from cascade.skills import pointing

    asked = []

    def preflight(a, b):
        asked.append((np.round(a, 3).tolist(), np.round(b, 3).tolist()))
        return "TCP outside workspace" if b[2] < 0.095 and not np.allclose(b, goal) else None

    start, goal = np.array([-0.1, 0.0, 0.04]), np.array([0.0, 0.0, 0.085])
    route = pointing.plan_route(_SlideIK(), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                                table_z=0.0, preflight=preflight).targets

    assert len(route) == 2 and route[0][2] == pytest.approx(0.10), route
    assert [start.tolist(), np.round(route[0], 3).tolist()] in [list(x) for x in asked]
    assert [np.round(route[0], 3).tolist(), goal.tolist()] in [list(x) for x in asked]
    # without the harness the lowest clear rise would have been taken
    assert pointing.plan_route(_SlideIK(), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                               table_z=0.0).targets[0][2] == pytest.approx(0.07)


@needs_pin_so101
def test_a_lean_in_route_the_harness_would_stop_is_refused_before_any_command(rig, monkeypatch):
    """The lean-in is a new motion: its direct leg (and any rise) goes
    through the harness's static route vet first."""
    r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    monkeypatch.setattr(r.runtime.arm.harness, "vet_step", lambda *a, **k: "test: outside the workspace")

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "the harness would stop it (test: outside the workspace)" in out["error"], out


@needs_pin
def test_a_top_down_route_is_streamed_as_before_without_the_static_vet(rig, monkeypatch):
    """Byte-identity guard: today's top-down pose keeps today's single leg,
    vetted step by step while it streams, never by the static route vet."""
    r = rig("mock", camera="mock")
    monkeypatch.setattr(r.runtime.arm.harness, "vet_step", lambda *a, **k: "test: never asked")

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and "rose_first" not in out, out


# ── halts and e-stops: zero further commands, at every point ──────────────


def _stop(runtime, kind):
    if kind == "halt":
        runtime.arm.harness.halt("operator: stop")
    else:
        runtime.arm.harness.estop("operator e-stop")


@needs_pin_so101
@pytest.mark.parametrize("kind", ["halt", "estop"])
@pytest.mark.parametrize("profile, point", [("mock", "topdown_ik"), ("mock", "route_check"),
                                            ("so101_mock", "lean_ik"), ("so101_mock", "pose_check"),
                                            ("so101_mock", "route_check")])
def test_a_stop_while_planning_sends_nothing(rig, monkeypatch, kind, profile, point):
    """mock: today's top-down pose; so101: the lean-in. A stop at any step of
    the planning leaves the arm and the jaws without a single command."""
    from cascade.skills import pointing

    if profile == "mock":
        r = rig("mock", camera="mock")
    else:
        r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    runtime = r.runtime
    fired = []

    def once():
        if not fired:
            fired.append(True)
            _stop(runtime, kind)

    if point == "topdown_ik":
        real_ik = runtime.kin.ik
        monkeypatch.setattr(runtime.kin, "ik", lambda *a, **k: (once(), real_ik(*a, **k))[1])
    elif point == "lean_ik":
        real_lean = runtime.kin.ik_approach
        monkeypatch.setattr(runtime.kin, "ik_approach", lambda *a, **k: (once(), real_lean(*a, **k))[1])
    elif point == "pose_check":
        real_pose = pointing.pose_clearance
        monkeypatch.setattr(pointing, "pose_clearance", lambda *a, **k: (once(), real_pose(*a, **k))[1])
    else:
        real_seg = pointing.segment_violation
        monkeypatch.setattr(pointing, "segment_violation", lambda *a, **k: (once(), real_seg(*a, **k))[1])

    out = runtime.execute("point_at", {"label": "red cube"})

    assert fired, f"{point} never ran"
    assert out["ok"] is False, out
    assert _commands(r) == [] and r.jaws == []


@needs_pin
@pytest.mark.parametrize("kind", ["halt", "estop"])
def test_a_stop_after_the_rise_leg_sends_no_further_command(rig, monkeypatch, kind):
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.10))
    _park(r, [0.18, 0.0, 0.06], yaw=math.pi / 2)
    arm = r.arms[0]
    real_move = r.runtime._move_joints
    at_stop = {}

    def move(q, *a, **k):
        ok = real_move(q, *a, **k)
        if not at_stop:
            _stop(r.runtime, kind)
            at_stop.update(arm=len(arm.commands), jaws=len(r.jaws))
        return ok

    monkeypatch.setattr(r.runtime, "_move_joints", move)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert at_stop and at_stop["arm"] > 0, "the rise leg never ran"
    assert len(arm.commands) == at_stop["arm"] and len(r.jaws) == at_stop["jaws"]
    assert out["ok"] is False, out


@needs_pin_so101
@pytest.mark.parametrize("kind", ["halt", "estop"])
def test_a_stop_mid_lean_in_sends_no_further_command(rig, kind):
    r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    r.runtime.arm.get_state()
    arm = r.arms[0]
    send = arm.send_joint_target
    at_stop = {}

    def send_then_stop(q):
        send(q)
        if not at_stop and len(arm.commands) >= 5:
            _stop(r.runtime, kind)
            at_stop.update(arm=len(arm.commands), jaws=len(r.jaws))

    arm.send_joint_target = send_then_stop
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert at_stop, "the gesture never streamed"
    assert len(arm.commands) == at_stop["arm"] and len(r.jaws) == at_stop["jaws"]
    assert out["ok"] is False, out


# ── the object top: NaN, the map's column top, an occluder ────────────────


@needs_pin
def test_a_nan_top_refuses_with_an_honest_text(rig, monkeypatch):
    r = rig("mock", camera="mock")
    orig = r.runtime._localize

    def localize(label, *a, **k):
        frame, fix = orig(label, *a, **k)
        pts = fix.points.copy()
        pts[0, 2] = np.nan
        return frame, dataclasses.replace(fix, points=pts)

    monkeypatch.setattr(r.runtime, "_localize", localize)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "cannot point at 'red cube': its observed top is not a finite height" in out["error"], out


def test_the_map_lift_is_the_pregrasps():
    """One rule for both readers (`_map_top_lift`): the map's column top above
    the observed top raises it by at most 2 cm, plus half a voxel (<= 5 mm);
    no map, an empty column or a non-finite answer: no lift."""
    from cascade.skills.runtime import _map_top_lift

    fix = SimpleNamespace(position=np.array([0.3, 0.0, 0.025]),
                          points=np.array([[0.3, 0.0, 0.05], [0.31, 0.01, 0.0], [0.29, -0.01, 0.02]]))

    def harness(answer):
        return SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=0.02),
                               occupancy=SimpleNamespace(column_top=lambda c, rad, z: answer))

    gcfg = {"exempt_radius_m": 0.07}
    assert _map_top_lift(gcfg, harness((0.059, 0.01)), fix, 0.05) == pytest.approx(0.009 + 0.005)
    assert _map_top_lift(gcfg, harness((0.15, 0.05)), fix, 0.05) == pytest.approx(0.02 + 0.005)
    assert _map_top_lift(gcfg, harness(None), fix, 0.05) == 0.0
    assert _map_top_lift(gcfg, harness((float("nan"), 0.01)), fix, 0.05) == 0.0
    assert _map_top_lift(gcfg, SimpleNamespace(occupancy=None), fix, 0.05) == 0.0


@needs_pin
def test_point_at_counts_the_maps_column_top(rig, monkeypatch):
    """The map's top 1 cm over the observed top (1 cm voxels): the hover
    counts from 0.05 + 0.01 + 0.005, like the pregrasp's."""
    from cascade.skills import runtime as runtime_mod

    r = rig("mock", camera="mock")
    real = runtime_mod._map_top_lift
    stand_in = SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=0.02),
                               occupancy=SimpleNamespace(column_top=lambda c, rad, z: (0.06, 0.01)))
    seen = []

    def lift(gcfg, harness, fix, top):
        seen.append(top)
        return real(gcfg, stand_in, fix, top)

    monkeypatch.setattr(runtime_mod, "_map_top_lift", lift)
    asked = []
    real_ik = r.runtime.kin.ik
    monkeypatch.setattr(r.runtime.kin, "ik", lambda T, *a, **k: (asked.append(float(T[2, 3])), real_ik(T, *a, **k))[1])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert seen == [pytest.approx(0.05, abs=1e-6)]
    assert asked[0] == pytest.approx(0.05 + 0.015 + 0.04, abs=1e-6)


@needs_pin
@pytest.mark.parametrize("cup_top, expect", [(0.09, "top-down"), (0.13, "lean")])
def test_an_object_under_another_is_pointed_at_over_the_top_one(rig, monkeypatch, cup_top, expect):
    """The booth's 'point at where it was': the cube sits under a cup. The
    top is the cup's, and the cup is part of the column."""
    r = rig("mock", camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    r.runtime.beliefs.update("cup", np.array([cube.position[0] + 0.01, cube.position[1], cup_top / 2]), 0.9,
                             extent=np.array([cup_top, 0.09, 0.09]), top_z=cup_top, color="blue")
    assert len(r.runtime.beliefs.all()) == 2, "test setup: the cup is its own belief"
    asked = []
    real_ik = r.runtime.kin.ik
    monkeypatch.setattr(r.runtime.kin, "ik", lambda T, *a, **k: (asked.append(float(T[2, 3])), real_ik(T, *a, **k))[1])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    if expect == "top-down":
        assert asked[0] == pytest.approx(cup_top + 0.04, abs=1e-6)
        assert "lean_deg" not in out
    else:
        assert out["lean_deg"] in LEAN_TILTS
    # nothing the arm commanded came within the margin of the cup's column
    cup = np.array([cube.position[0] + 0.01, cube.position[1]])
    for q in _commands(r):
        T = r.runtime.kin.fk(q)
        tool = _tool_grid(r.cfg) @ T[:3, :3].T + T[:3, 3]
        lo = np.array([*(cup - 0.045), 0.0])
        hi = np.array([*(cup + 0.045), cup_top])
        assert _box_distance(tool, lo, hi).min() >= MARGIN - 1e-6


@needs_pin
def test_the_target_is_not_its_own_occluder(rig, monkeypatch):
    """The cube's own belief remembers a 5 cm top; the fix (moved 2 cm down
    here) says 3 cm. The cube does not stand over itself: 0.03 + 0.04."""
    r = rig("mock", camera="mock")
    orig = r.runtime._localize

    def localize(label, *a, **k):
        frame, fix = orig(label, *a, **k)
        return frame, dataclasses.replace(fix, points=fix.points - [0.0, 0.0, 0.02],
                                          position=fix.position - [0.0, 0.0, 0.02])

    monkeypatch.setattr(r.runtime, "_localize", localize)
    assert r.runtime.beliefs.find("red cube").top_z == pytest.approx(0.05, abs=1e-6)
    asked = []
    real_ik = r.runtime.kin.ik
    monkeypatch.setattr(r.runtime.kin, "ik", lambda T, *a, **k: (asked.append(float(T[2, 3])), real_ik(T, *a, **k))[1])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert asked[0] == pytest.approx(0.03 + 0.04, abs=1e-6)


@needs_pin
def test_a_neighbour_that_does_not_cover_the_aim_is_not_an_occluder(rig, monkeypatch):
    """Byte-identity guard: a taller object beside (not over) the target does
    not change today's hover."""
    r = rig("mock", camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    r.runtime.beliefs.update("bottle", np.array([cube.position[0], cube.position[1] + 0.15, 0.1]), 0.9,
                             extent=np.array([0.2, 0.06, 0.06]), top_z=0.2, color="blue")
    asked = []
    real_ik = r.runtime.kin.ik
    monkeypatch.setattr(r.runtime.kin, "ik", lambda T, *a, **k: (asked.append(float(T[2, 3])), real_ik(T, *a, **k))[1])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert asked[0] == pytest.approx(0.05 + 0.04, abs=1e-6)


# ── one jaw box for the harness and point_at ──────────────────────────────


@pytest.mark.parametrize("order", ["down_open", "open_down", "third_open_down"])
def test_the_harness_and_point_at_read_one_jaw_box(order):
    from cascade.safety.tool_volume import box_grid, build_tool_volume, jaw_box
    from cascade.skills.runtime import _pointing_tool_points

    sweep = {"extents_open": [0.09, 0.02, 0.045], "offset_open": [0.0, 0.0, -0.0225]}
    view = {"safety": {"tool_volume": {"wrist_camera": False, "jaws": True, "spacing_m": 0.01}},
            "grasp": {"graspgenx": {"sweep": sweep}}}
    vol = build_tool_volume(view, {"tool_axis_order": order})
    np.testing.assert_array_equal(vol.local, box_grid(*jaw_box(sweep, order), 0.01))
    centre, axes, half = jaw_box(sweep, order)
    corners = np.array([centre + axes @ (np.array(s) * half)
                        for s in [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]])
    got = _pointing_tool_points({"graspgenx": {"sweep": sweep}}, order)
    np.testing.assert_allclose(got[0], 0.0)
    np.testing.assert_allclose(np.sort(got[1:], axis=0), np.sort(corners, axis=0), atol=1e-12)


def test_point_at_follows_the_shared_jaw_box(monkeypatch):
    """A change to the one jaw box reaches point_at (no private copy)."""
    from cascade.safety import tool_volume
    from cascade.skills.runtime import _pointing_tool_points

    real = tool_volume.jaw_box
    monkeypatch.setattr(tool_volume, "jaw_box", lambda s, o: (lambda c, a, h: (c + [0, 0, 0.5], a, h))(*real(s, o)))
    got = _pointing_tool_points({"graspgenx": {"sweep": {"extents_open": [0.01, 0.01, 0.01]}}}, "down_open")
    assert got[1:, 2].min() == pytest.approx(0.495)


@pytest.mark.parametrize("sweep", [{"extents_open": [0.1, float("nan"), 0.0]}, {"extents_open": "wide"},
                                   {"extents_open": [0.1, -0.01, 0.0]}])
def test_a_malformed_jaw_box_raises(sweep):
    from cascade.safety.tool_volume import jaw_box

    with pytest.raises(ValueError, match="open jaws must be finite"):
        jaw_box(sweep, "down_open")


# ── the jaws close only where everything a close sweeps keeps the margin ───


def _sweep_jaws(points):
    """Jaws whose closed and current samples are the TCP alone and whose
    close sweeps `points` (TCP frame)."""
    from cascade.skills import pointing

    tcp = np.zeros((1, 3))
    return pointing.Jaws(closed=(tcp,), now=(tcp,), sweep=(tcp, np.asarray(points, float)))


def test_the_jaws_close_where_the_arm_starts_when_the_close_clears():
    """main's order: close, then move (byte-identity for every clear start)."""
    from cascade.skills import pointing

    route = pointing.plan_route(_SlideIK(), np.array([-0.2, 0.0, 0.2]), np.array([0.0, 0.0, 0.1]), _box_column(),
                                _sweep_jaws(np.array([[0.04, 0.0, 0.0]])), 2, margin=MARGIN, table_z=0.0)
    assert route.close_at == 0 and len(route.targets) == 1


def test_a_close_that_would_sweep_into_the_object_waits_for_the_pointing_pose():
    """The reviewer's Critical: the start's close sweeps 4 cm sideways, 2 cm
    from the object's side; the TCP itself is clear. The arm goes first and
    the jaws close at the goal, where the sweep is clear."""
    from cascade.skills import pointing

    start = np.array([-0.08, 0.0, 0.04])          # the sweep reaches x -0.04: 2 cm from the side
    route = pointing.plan_route(_SlideIK(), start, np.array([-0.1, 0.0, 0.2]), _box_column(),
                                _sweep_jaws(np.array([[0.04, 0.0, 0.0]])), 2, margin=MARGIN, table_z=0.0)
    assert route.close_at == 1 and len(route.targets) == 1
    assert pointing.close_violation(_SlideIK(), start, _box_column(), _sweep_jaws(
        np.array([[0.04, 0.0, 0.0]])).sweep, MARGIN, 0.0) == "the jaws, closing, 0.020 m from it"


def test_a_close_clear_only_at_the_waypoint_closes_there():
    """Neither the start nor the goal clears the close (it sweeps 4 cm along
    +x, toward the object); the rise-first waypoint does: two legs, the
    close between them, and the second leg is checked with closed jaws."""
    from cascade.skills import pointing

    start, goal = np.array([-0.08, 0.0, 0.04]), np.array([-0.075, 0.0, 0.07])
    jaws = _sweep_jaws(np.array([[0.04, 0.0, 0.0]]))
    route = pointing.plan_route(_SlideIK(), start, goal, _box_column(), jaws, 2, margin=MARGIN, table_z=0.0)
    assert len(route.targets) == 2 and route.close_at == 1, route
    assert pointing.close_violation(_SlideIK(), route.targets[0], _box_column(), jaws.sweep, MARGIN, 0.0) is None
    assert pointing.close_violation(_SlideIK(), goal, _box_column(), jaws.sweep, MARGIN, 0.0) is not None


def test_a_close_clear_only_at_the_pointing_pose_waits_past_the_waypoint():
    """The close sweeps 5 cm under the TCP: clear only 8 cm above the table.
    The start (4 cm) and the rise-first waypoint (7 cm) are too low, the
    pointing pose (10 cm) is not: the arm goes over the object with the jaws
    as they are and closes them only at the end."""
    from cascade.skills import pointing

    start, goal = np.array([-0.1, 0.0, 0.04]), np.array([0.1, 0.0, 0.10])
    jaws = _sweep_jaws(np.array([[0.0, 0.0, -0.05]]))
    route = pointing.plan_route(_SlideIK(), start, goal, _box_column(), jaws, 2, margin=MARGIN, table_z=0.0)
    assert route.close_at == 2 and len(route.targets) == 2, route
    np.testing.assert_allclose(route.targets[0], [-0.1, 0.0, 0.07])


def test_the_close_keeps_the_margin_from_the_table():
    from cascade.skills import pointing

    sweep = (np.zeros((1, 3)), np.array([[0.0, 0.0, -0.04]]))
    why = pointing.close_violation(_SlideIK(), np.array([0.2, 0.0, 0.05]), _box_column(), sweep, MARGIN, 0.0)
    assert why == "the jaws, closing, 0.010 m above the table", why
    assert pointing.close_violation(_SlideIK(), np.array([0.2, 0.0, 0.08]), _box_column(), sweep, MARGIN, 0.0) is None


def test_the_close_keeps_the_margin_from_a_neighbour():
    """The start's close sweeps 1 cm from a neighbour (8 cm from the target):
    the jaws close at the pointing pose instead."""
    from cascade.skills import pointing

    near = pointing.neighbour_column(np.array([[-0.17, -0.02], [-0.13, -0.02], [-0.13, 0.02], [-0.17, 0.02]]), 0.05,
                                      0.0, "the blue box")
    jaws = _sweep_jaws(np.array([[-0.04, 0.0, -0.06]]))
    start, goal = np.array([-0.08, 0.0, 0.10]), np.array([0.12, 0.0, 0.12])
    assert pointing.plan_route(_SlideIK(), start, goal, _box_column(), jaws, 2, margin=MARGIN,
                               table_z=0.0).close_at == 0
    route = pointing.plan_route(_SlideIK(), start, goal, _box_column(), jaws, 2, margin=MARGIN, table_z=0.0,
                                neighbours=(near,))
    assert route == pointing.Route([goal], 1), route


def test_the_direct_leg_keeps_the_margin_from_a_neighbour():
    """Straight to the pointing pose (top-down or lean-in: the same leg)
    passes 2 cm over a neighbour's top: a rise-first waypoint; without the
    neighbour the same leg goes straight (#378 review Critical 1: the
    top-down leg ignored neighbours)."""
    from cascade.skills import pointing

    near = pointing.neighbour_column(np.array([[-0.12, -0.02], [-0.08, -0.02], [-0.08, 0.02], [-0.12, 0.02]]), 0.06,
                                      0.0, "the blue box")
    start, goal = np.array([-0.2, 0.0, 0.06]), np.array([0.0, 0.0, 0.10])
    route = pointing.plan_route(_SlideIK(), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                                table_z=0.0, neighbours=(near,))
    assert len(route.targets) == 2 and route.targets[0][2] > start[2], route
    alone = pointing.plan_route(_SlideIK(), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                                table_z=0.0)
    assert alone == pointing.Route([goal], 0), alone


def test_a_top_down_pose_counts_neighbours():
    """The top-down acceptance: a pose whose jaws (or a joint) come within
    the margin of a neighbour is not taken."""
    from cascade.skills import pointing

    near = pointing.neighbour_column(np.array([[0.025, -0.02], [0.065, -0.02], [0.065, 0.02], [0.025, 0.02]]), 0.2,
                                     0.0, "the blue bottle")
    assert pointing.neighbour_clearance(_SlideKin(), np.array([0.0, 0.0, 0.1]), np.zeros((1, 3)), (near,)) == \
        pytest.approx((0.025, "the blue bottle"))
    # the jaws, not only the TCP: a jaw sample 2 cm toward it
    jaws = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    assert pointing.neighbour_clearance(_SlideKin(), np.array([0.0, 0.0, 0.1]), jaws, (near,)) == \
        pytest.approx((0.005, "the blue bottle"))
    assert pointing.neighbour_clearance(_SlideKin(), np.array([0.0, 0.0, 0.1]), np.zeros((1, 3)), ()) == \
        (math.inf, "")


def test_each_jaw_group_is_its_own_pair():
    """Group A (the TCP) 1 cm over the top rises away; group B, 2.5 cm beside
    the object, slides 4 mm closer. Their minimum grows, but B closes in from
    inside the margin: refused."""
    from cascade.skills import pointing

    a, b = np.zeros((1, 3)), np.array([[0.045, 0.0, -0.03]])
    start, goal = np.array([0.0, 0.0, 0.06]), np.array([-0.004, 0.0, 0.07])
    why = pointing.segment_violation(_SlideKin(), start, goal, _box_column(), (a, b), MARGIN, 0.0, from_inside=True)
    assert why is not None and "closing in from inside the margin" in why, why
    assert pointing.segment_violation(_SlideKin(), start, goal, _box_column(), np.vstack([a, b]), MARGIN, 0.0,
                                      from_inside=True) is None


def test_a_rise_leg_outside_the_margin_never_dips_between_two_samples():
    """4 cm over the top, 6 cm across it in one sample step: both samples
    4.1 cm from it, 4.0 cm over the middle. Outside the margin, but on a rise
    the jaws never come closer to the target: refused."""
    from cascade.skills import pointing

    args = (_box_column(), np.zeros((1, 3)), MARGIN, 0.0)
    start, goal = np.array([-0.03, 0.0, 0.09]), np.array([0.03, 0.0, 0.09])
    why = pointing.segment_violation(_SlideKin(), start, goal, *args, step_rad=0.06, from_inside=True, rise=True)
    assert why is not None and "between two samples" in why, why
    assert pointing.segment_violation(_SlideKin(), start, goal, *args, step_rad=0.06, from_inside=True) is None


def test_no_pose_where_the_jaws_may_close_refuses():
    from cascade.skills import pointing

    start, goal = np.array([-0.08, 0.0, 0.04]), np.array([-0.075, 0.0, 0.07])
    # a close that sweeps 20 cm all around and below the TCP: it reaches the object from every pose
    xs, zs = np.meshgrid(np.linspace(-0.2, 0.2, 41), np.linspace(-0.25, 0.0, 26))
    everywhere = np.c_[xs.ravel(), np.zeros(xs.size), zs.ravel()]
    with pytest.raises(pointing.RouteRefused, match="may close neither where it starts"):
        pointing.plan_route(_SlideIK(), start, goal, _box_column(), _sweep_jaws(everywhere), 2,
                            margin=MARGIN, table_z=-1.0)


def test_while_holding_the_jaws_never_close_and_keep_their_reading():
    """close=False: no jaw command; the legs use the jaws as they are (`now`)."""
    from cascade.skills import pointing

    tcp = np.zeros((1, 3))
    wide = pointing.Jaws(closed=(tcp,), now=(tcp, np.array([[0.0, 0.0, -0.04]])), sweep=(tcp,))
    start, goal = np.array([-0.1, 0.0, 0.09]), np.array([0.1, 0.0, 0.09])
    assert pointing.plan_route(_SlideIK(), start, goal, _box_column(), tcp, 2, margin=MARGIN, table_z=0.0,
                               close=False) == pointing.Route([goal], None)
    with pytest.raises(pointing.RouteRefused):        # the held jaws reach 4 cm lower: over the top they touch it
        pointing.plan_route(_SlideIK(), start, goal, _box_column(), wide, 2, margin=MARGIN, table_z=0.0,
                            close=False)


def _hinged_gripper(reading_box=None):
    return {"closed_pos": -0.15, "open_pos": 0.60, "jaw_volume": {
        "stop_rad": -0.1745, "hinge": {"point": [0.0, 0.0, -0.08], "axis": [0.0, -1.0, 0.0]},
        "fixed": [{"extents": [0.01, 0.01, 0.01], "offset": [0.005, 0.0, 0.0]}],
        "moving": [reading_box or {"extents": [0.01, 0.01, 0.01], "offset": [-0.005, 0.0, 0.0]}]}}


def test_the_moving_jaw_is_turned_about_its_hinge_to_the_reading():
    """`now` holds the moving jaw at the reading (+- the reading's
    tolerance), `sweep` from the stop to there, `closed` from the stop to
    closed_pos; the fixed jaw is in all three; the sweep box too."""
    from cascade.skills import pointing

    sweep = {"extents_open": [0.0, 0.0, 0.0]}
    def stacked(jaws):
        return SimpleNamespace(**{k: np.vstack(getattr(jaws, k)) for k in ("closed", "now", "sweep")})

    for reading in (0.0, 0.6):
        jaws = stacked(pointing.jaw_samples(sweep, _hinged_gripper(), "open_down", opening=reading))
        # the moving jaw's tip, 8 cm from the hinge, turned by (angle - stop) about -y opens toward -x
        tips = []
        for angle in (reading - pointing.JAW_READING_TOL_RAD, reading + pointing.JAW_READING_TOL_RAD):
            a = angle + 0.1745
            tips.append([-0.08 * math.sin(a), 0.0, -0.08 + 0.08 * math.cos(a)])
        for tip in tips:
            assert np.linalg.norm(jaws.now - tip, axis=1).min() < 0.011
            assert np.linalg.norm(jaws.sweep - tip, axis=1).min() < 0.011
        assert jaws.now[:, 0].min() > jaws.sweep[:, 0].min() - 1e-9
        assert np.linalg.norm(jaws.closed - [0.0, 0.0, 0.0], axis=1).min() < 1e-9      # the TCP
        assert np.linalg.norm(jaws.closed - [0.01, 0.0, 0.005], axis=1).min() < 1e-9   # the fixed jaw's corner
    wide = stacked(pointing.jaw_samples(sweep, _hinged_gripper(), "open_down", opening=0.6))
    assert wide.sweep[:, 0].min() < -0.05 and wide.closed[:, 0].min() > -0.02
    # no reading: the widest the profile opens them
    unknown = stacked(pointing.jaw_samples(sweep, _hinged_gripper(), "open_down", opening=None))
    assert unknown.now[:, 0].min() == pytest.approx(wide.now[:, 0].min())


def test_parallel_jaws_are_the_sweep_box_in_every_state():
    from cascade.skills import pointing

    sweep = {"extents_open": [0.09, 0.02, 0.045], "offset_open": [0.0, 0.0, -0.0225]}
    jaws = pointing.jaw_samples(sweep, {"closed_pos": 1.0, "open_pos": 0.0}, "down_open", opening=0.3)
    base = pointing.tool_samples(sweep, "down_open")
    for got in (jaws.closed, jaws.now, jaws.sweep):
        assert len(got) == 1
        np.testing.assert_array_equal(got[0], base)


@pytest.mark.parametrize("bad", [{"moving": []}, {"hinge": {"point": [0.0, 0.0, float("nan")], "axis": [0, 1, 0]}},
                                 {"hinge": {"point": [0.0, 0.0, 0.0], "axis": [0, 0, 0]}}, {"stop_rad": "x"},
                                 {"fixed": [{"extents": [0.01, -0.01, 0.01]}]}])
def test_a_malformed_jaw_volume_raises(bad):
    from cascade.skills import pointing

    gripper = _hinged_gripper()
    gripper["jaw_volume"].update(bad)
    with pytest.raises(ValueError, match="jaw_volume"):
        pointing.jaw_samples({"extents_open": [0.01, 0.01, 0.01]}, gripper, "open_down", opening=0.0)


@needs_pin_so101
def test_the_so101_jaws_close_once_the_arm_left_the_cube(rig):
    """mock_small: home holds the jaws over the cube (the sweep box 5 mm into
    its top). #372 closed them there, before moving; now the arm leaves
    first and the jaws close only where their whole close clears it."""
    from cascade.skills import pointing

    r = rig("so101_mock")
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert len(r.jaws) == 1 and r.jaws_at[0] > 0, (r.jaws, r.jaws_at)
    q_close = _commands(r)[r.jaws_at[0] - 1]
    _, fix = r.runtime._localize("red cube")
    column = pointing.object_column(fix, 0.0, float(fix.points[:, 2].max()))
    jaws = pointing.jaw_samples(r.cfg.grasp.graspgenx.sweep, r.cfg.arm.get("gripper"), "open_down", opening=0.0)
    assert pointing.close_violation(r.runtime.kin, q_close, column, jaws.sweep, MARGIN, 0.0) is None
    assert pointing.close_violation(r.runtime.kin, _commands(r)[0], column, jaws.sweep, MARGIN, 0.0) is not None


@needs_pin_so101
def test_the_so101_open_beak_beside_a_cube_backs_off_before_it_closes(rig, monkeypatch):
    """so101_mujoco's home reads the gripper open (0.60): the beak reaches
    6 cm sideways, 1.4 cm from a 5 cm cube at r 0.26, and closing it there
    sweeps to 1 mm. Every rise from there brings the open beak closer first;
    the arm backs off sideways with the jaws open, closes them at the
    waypoint, then leans in."""
    from cascade.skills import pointing

    r = rig("so101_mock", prop=(*at(0.26, 0.0), 0.05, 0.05))
    real_state = r.runtime.arm.get_state
    monkeypatch.setattr(r.runtime.arm, "get_state", lambda: dataclasses.replace(real_state(), gripper_pos=0.60))
    q0 = np.asarray(real_state().q, float)
    _, fix = r.runtime._localize("red cube")
    column = pointing.object_column(fix, 0.0, float(fix.points[:, 2].max()))
    jaws = pointing.jaw_samples(r.cfg.grasp.graspgenx.sweep, r.cfg.arm.get("gripper"), "open_down", opening=0.60)
    assert pointing.close_violation(r.runtime.kin, q0, column, jaws.sweep, MARGIN, 0.0) is not None

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and out["rose_first"] is True and out["lean_deg"] in LEAN_TILTS, out
    first_leg = r.jaws_at[0]
    assert len(r.jaws) == 1 and 0 < first_leg < len(_commands(r)), (r.jaws_at, len(_commands(r)))
    q_close = _commands(r)[first_leg - 1]
    assert pointing.close_violation(r.runtime.kin, q_close, column, jaws.sweep, MARGIN, 0.0) is None
    # the open beak only moved away on the first leg (sampled at every streamed command)
    d = [float(column.distance(np.vstack(jaws.now) @ np.asarray(r.runtime.kin.fk(q))[:3, :3].T
                               + r.runtime.kin.fk(q)[:3, 3]).min()) for q in [q0, *_commands(r)[:first_leg]]]
    assert all(b >= a - 1e-6 for a, b in zip(d, d[1:])), d


@needs_pin_so101
@pytest.mark.parametrize("valid", [True, False])
def test_the_jaws_as_they_are_come_from_the_gripper_reading(rig, monkeypatch, valid):
    """`now` is built from the arm's gripper reading; an invalid reading is
    the widest the profile opens them (None)."""
    from cascade.skills import pointing

    r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    real_state = r.runtime.arm.get_state
    monkeypatch.setattr(r.runtime.arm, "get_state",
                        lambda: dataclasses.replace(real_state(), gripper_pos=0.31, gripper_valid=valid))
    seen = []
    real = pointing.jaw_samples
    monkeypatch.setattr(pointing, "jaw_samples", lambda *a, **k: (seen.append(a[3] if len(a) > 3 else
                                                                               k.get("opening")), real(*a, **k))[1])

    r.runtime.execute("point_at", {"label": "red cube"})

    assert seen == [0.31 if valid else None], seen


@needs_pin
def test_a_top_down_pose_from_a_clear_start_closes_the_jaws_before_moving(rig):
    """Byte-identity guard: main's order on the mock rig (close, then move)."""
    r = rig("mock", camera="mock")
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert r.jaws_at == [0] and len(_commands(r)) > 10


# ── known neighbours keep the same margin ─────────────────────────────────


def _neighbour(r, label, xy, side, height):
    r.runtime.beliefs.update(label, np.array([xy[0], xy[1], height / 2]), 0.9,
                             extent=np.array([height, side, side]), top_z=height, color="blue")
    lo = np.array([xy[0] - side / 2, xy[1] - side / 2, 0.0])
    hi = np.array([xy[0] + side / 2, xy[1] + side / 2, height])
    return lo, hi


def _arm_to_box(r, q, lo, hi) -> float:
    T = r.runtime.kin.fk(q)
    tool = _tool_grid(r.cfg) @ T[:3, :3].T + T[:3, 3]
    links = r.runtime.kin.link_positions(q)[1:]
    return min(float(_box_distance(tool, lo, hi).min()), float(_box_distance(links, lo, hi).min()))


def _assert_commands_keep_the_margin_from(r, lo, hi):
    """Every streamed command keeps the margin from the box, after an escape
    from a start already inside it (never closer while inside)."""
    d = [_arm_to_box(r, q, lo, hi) for q in _commands(r)]
    out = next((i for i, v in enumerate(d) if v >= MARGIN), None)
    assert out is not None, f"never left the neighbour's margin (best {max(d):.4f} m)"
    assert all(b >= a - 1e-4 for a, b in zip(d[:out], d[1:out + 1])), d[:out + 1]
    assert min(d[out:]) >= MARGIN - 1e-6, f"the arm came {min(d[out:]):.4f} m from the neighbour"


@needs_pin_so101
def test_the_so101_lean_in_keeps_the_margin_from_a_neighbour_in_front(rig):
    """A 4 cm box, 6 cm tall, 7 cm in front of the target (toward the base),
    where #372's lean-in drove the jaws (the reviewer's probe: 4 of 28 rows
    ended inside a neighbour). Every command keeps 3 cm from it, or nothing
    moves."""
    x, y = at(0.24, 0.0)
    r = rig("so101_mock", prop=(x, y, 0.035, 0.035))
    lo, hi = _neighbour(r, "box", (x - 0.07, y), 0.04, 0.06)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    if out["ok"]:
        _assert_commands_keep_the_margin_from(r, lo, hi)
    else:
        _refused_untouched(r, out)


@needs_pin_so101
@pytest.mark.parametrize("seed", range(12))
def test_so101_lean_ins_keep_the_margin_from_random_neighbours(rig, seed):
    """The reviewer's neighbour probe, seeded: target cube 3.5 cm at r
    0.18-0.26; a neighbour box 3-5 cm wide, 3-9 cm tall, 5-9 cm from it at any
    angle. ok -> every command keeps 3 cm from the neighbour (escape aside);
    refused -> no command."""
    rng = np.random.default_rng([372, seed])
    radius, az = rng.uniform(0.18, 0.26), rng.uniform(-0.4, 0.4)
    x, y = at(radius, az)
    r = rig("so101_mock", prop=(x, y, 0.035, 0.035))
    side, height, dist, ang = rng.uniform(0.03, 0.05), rng.uniform(0.03, 0.09), rng.uniform(0.05, 0.09), \
        rng.uniform(-math.pi, math.pi)
    lo, hi = _neighbour(r, "box", (x + dist * math.cos(ang), y + dist * math.sin(ang)), side, height)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    if out["ok"]:
        _assert_commands_keep_the_margin_from(r, lo, hi)
        _assert_every_command_keeps_the_margin(r)
    else:
        _refused_untouched(r, out)


@needs_pin_so101
@pytest.mark.parametrize("bottle", [False, True])
def test_a_neighbour_that_blocks_every_lean_in_refuses(rig, bottle):
    """A 15 cm bottle 5 cm in front of the target: no lean-in pose keeps 3 cm
    from it, so nothing moves; without it the same cube is pointed at."""
    x, y = at(0.22, 0.0)
    r = rig("so101_mock", prop=(x, y, 0.035, 0.035))
    if bottle:
        _neighbour(r, "bottle", (x - 0.05, y), 0.04, 0.15)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    if bottle:
        _refused_untouched(r, out)
        assert "every known object" in out["error"], out
    else:
        assert out["ok"] is True and out["lean_deg"] in LEAN_TILTS, out


def test_the_lean_in_pose_counts_neighbours():
    from cascade.skills import pointing

    near = pointing.neighbour_column(np.array([[0.05, -0.02], [0.09, -0.02], [0.09, 0.02], [0.05, 0.02]]), 0.08, 0.0,
                                     "the blue box")
    clr = pointing.pose_clearance(_SlideKin(), np.array([0.0, 0.0, 0.1]), _box_column(), np.zeros((1, 3)), 2, 0.0,
                                  neighbours=(near,))
    assert clr.near == "the blue box" and clr.near_m == pytest.approx(math.hypot(0.05, 0.02))
    assert not dataclasses.replace(clr, near_m=MARGIN - 1e-4).admissible(MARGIN)
    assert "the arm 0.030 m from the blue box" in dataclasses.replace(clr, near_m=0.0304).worst(0.031)


def test_a_rise_leg_keeps_the_margin_from_a_neighbour():
    """A rise that drifts 1 cm toward a tall neighbour: the jaws' back end,
    3.5 cm from it where the arm starts, ends 2.5 cm from it. Refused, the
    neighbour named; the target alone would let it pass."""
    from cascade.skills import pointing

    near = pointing.neighbour_column(np.array([[-0.155, -0.02], [-0.115, -0.02], [-0.115, 0.02], [-0.155, 0.02]]),
                                     0.3, 0.0, "the blue bottle")
    tool = np.array([[0.0, 0.0, 0.0], [-0.03, 0.0, 0.0]])
    start, goal = np.array([-0.05, 0.0, 0.1]), np.array([-0.06, 0.0, 0.2])
    why = pointing.segment_violation(_SlideKin(), start, goal, (_box_column(), near), tool, MARGIN, 0.0,
                                     from_inside=True, rise=True)
    assert why is not None and "from the blue bottle" in why, why
    assert pointing.segment_violation(_SlideKin(), start, goal, _box_column(), tool, MARGIN, 0.0, from_inside=True,
                                      rise=True) is None


# ── top-down profiles: known neighbours keep the margin too ──────────────

#: the #378 review's top-down rows (a tall bottle beside the cube went 11.8-12.9 mm
#: inside the jaw box with ok: true, where main kept >= 36 mm)
TOP_DOWN_PROFILES = ("mock", "mock_reach", "fr3", "isaac_kitchen", "rebot_rs_reach")


def _table_neighbour(r, label, xy, side, height):
    """A `side` x `side` x `height` box standing on the profile's table at
    xy, remembered as a belief; its (lo, hi) corners."""
    table = float(r.cfg.safety.get("table_z", 0.0))
    r.runtime.beliefs.update(label, np.array([xy[0], xy[1], table + height / 2]), 0.9,
                             extent=np.array([height, side, side]), top_z=table + height, color="blue")
    return (np.array([xy[0] - side / 2, xy[1] - side / 2, table]),
            np.array([xy[0] + side / 2, xy[1] + side / 2, table + height]))


@pytest.mark.parametrize("profile", TOP_DOWN_PROFILES)
@pytest.mark.parametrize("dist", [0.056, 0.09])
@pytest.mark.parametrize("deg", range(0, 360, 45))
def test_a_top_down_point_keeps_the_margin_from_a_tall_neighbour(rig, profile, dist, deg):
    """#378 review Critical 1: a 16 cm bottle (5 cm girth) 5.6 cm from the
    cube. The top-down pose and its direct leg ignored it, so the jaw box
    ended 12.9 mm inside it with ok: true. Now every command keeps 3 cm from
    it (and from the cube), or nothing moves. (At 9 cm some rows point: fr3
    every row, the others 2 of 8.)"""
    r = rig(profile, camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    assert cube is not None, "test setup: the mock camera saw no cube"
    a = math.radians(deg)
    xy = np.asarray(cube.position, float)[:2] + dist * np.array([math.cos(a), math.sin(a)])
    lo, hi = _table_neighbour(r, "bottle", xy, 0.05, 0.16)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    if out["ok"]:
        _assert_commands_keep_the_margin_from(r, lo, hi)
        _assert_every_command_keeps_the_margin(r)
    else:
        _refused_untouched(r, out)


# ── a jaw touching an object backs straight out first (#378 review I3) ─────


class _DownKin(_SlideKin):
    """_SlideKin whose tool points down (approach column 2 = -z) and whose IK
    solves any TCP position (the line `planning.cartesian` follows)."""

    def fk(self, q):
        T = super().fk(q)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return T

    def ik(self, T, q_init=None, *a, **k):
        return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=True, error=0.0)


def test_a_jaw_touching_the_object_backs_straight_out_first():
    """The TCP 3 mm over the top: back straight out along the approach (up
    here), 5 mm at a time on the line, until it is more than TOUCH_M clear;
    the route starts where that ends."""
    from cascade.skills import pointing

    start, goal = np.array([0.0, 0.0, 0.053]), np.array([0.0, 0.0, 0.10])
    route = pointing.plan_route(_DownKin(), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                                table_z=0.0)
    assert route.withdraw_m == pytest.approx(0.005)
    np.testing.assert_allclose(np.array(route.withdraw), [[0.0, 0.0, 0.058]], atol=1e-12)
    assert route.targets == [goal] and route.close_at == 1, route
    # touching the top: 1 cm back (5 mm is exactly TOUCH_M, still touching)
    route = pointing.plan_route(_DownKin(), np.array([0.0, 0.0, 0.05]), goal, _box_column(), np.zeros((1, 3)), 2,
                                margin=MARGIN, table_z=0.0)
    np.testing.assert_allclose(np.array(route.withdraw)[:, 2], [0.055, 0.06], atol=1e-12)
    assert route.withdraw_m == pytest.approx(0.01)


def test_a_jaw_clear_of_every_object_goes_as_before():
    """6 mm over the top (beyond TOUCH_M): no withdrawal."""
    from cascade.skills import pointing

    start, goal = np.array([0.0, 0.0, 0.056]), np.array([0.0, 0.0, 0.10])
    route = pointing.plan_route(_DownKin(), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                                table_z=0.0)
    assert route == pointing.Route([goal], 1), route


def test_a_touching_jaw_on_an_arm_that_cannot_back_out_refuses():
    from cascade.skills import pointing

    with pytest.raises(pointing.RouteRefused, match=r"the jaws start 0\.003 m from it and this arm cannot back them"):
        pointing.plan_route(_DownKin(), np.array([0.0, 0.0, 0.053]), np.array([0.0, 0.0, 0.10]), _box_column(),
                            np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0, withdraw=False)


def test_a_touching_jaw_with_no_straight_line_back_refuses():
    """No pose on the line back solves (IK fails above the start): refused,
    naming the last distance tried."""
    from cascade.skills import pointing

    class _Stuck(_DownKin):
        def ik(self, T, q_init=None, *a, **k):
            ok = float(np.asarray(T)[2, 3]) <= 0.055
            return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=ok, error=0.1)

    with pytest.raises(pointing.RouteRefused, match="backing straight out along the approach, no pose solves 0.5 cm"):
        pointing.plan_route(_Stuck(), np.array([0.0, 0.0, 0.053]), np.array([0.0, 0.0, 0.10]), _box_column(),
                            np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0)


def test_a_withdrawal_that_closes_in_is_not_taken():
    """A tool whose approach points UP backs out downward, into the object:
    every distance closes in from inside the margin, so it is refused."""
    from cascade.skills import pointing

    class _UpKin(_DownKin):
        def fk(self, q):
            return _SlideKin.fk(self, q)

    with pytest.raises(pointing.RouteRefused, match="closing in from inside the margin"):
        pointing.plan_route(_UpKin(), np.array([0.0, 0.0, 0.053]), np.array([0.0, 0.0, 0.10]), _box_column(),
                            np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0)


def test_a_withdrawal_may_wobble_dip_tol_but_never_closes_in_further():
    """A jaw 2 mm beside the object, backing straight up along its side: a
    face sliding along the line keeps its distance, but the arm's turn drifts
    it toward the object by `drift` per metre. Up to DIP_TOL_M closer than
    where the withdrawal started is taken (a 0.03 mm wobble over the 1.5 cm
    back); 0.1 mm is not."""
    from cascade.skills import pointing

    class _Drift(_DownKin):
        def __init__(self, drift):
            self.drift = drift

        def fk(self, q):
            T = super().fk(q)
            T[0, 3] -= self.drift * (float(q[2]) - 0.04)
            return T

    start, goal = np.array([0.022, 0.0, 0.04]), np.array([0.022, 0.0, 0.12])
    route = pointing.plan_route(_Drift(0.002), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                                table_z=0.0)
    assert route.withdraw_m == pytest.approx(0.015), route
    with pytest.raises(pointing.RouteRefused, match="cm back: the jaws 0.002 m from it .*closing in from inside"):
        pointing.plan_route(_Drift(0.01), start, goal, _box_column(), np.zeros((1, 3)), 2, margin=MARGIN,
                            table_z=0.0)


def test_a_withdrawal_step_the_harness_would_stop_is_refused():
    """Each step back is vetted by the harness's static route check too."""
    from cascade.skills import pointing

    with pytest.raises(pointing.RouteRefused, match=r"the jaws start 0\.003 m from it; backing straight out along "
                                                    r"the approach, 0\.5 cm back: the harness would stop it \(too fast\)"):
        pointing.plan_route(_DownKin(), np.array([0.0, 0.0, 0.053]), np.array([0.0, 0.0, 0.10]), _box_column(),
                            np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0, preflight=lambda a, b: "too fast")


def test_a_withdrawal_step_that_jumps_a_joint_is_refused():
    """Each step back is solved from the previous pose with no restart: a
    solution more than a branch-flip bound away (here 0.5 on one joint) is
    no step on the line, so nothing moves."""
    from cascade.skills import pointing

    class _Flip(_DownKin):
        def ik(self, T, q_init=None, *a, **k):
            q = np.asarray(T, float)[:3, 3].copy()
            q[0] += 0.5
            return SimpleNamespace(q=q, success=True, error=0.0)

    with pytest.raises(pointing.RouteRefused, match="backing straight out along the approach, no pose solves 0.5 cm"):
        pointing.plan_route(_Flip(), np.array([0.0, 0.0, 0.053]), np.array([0.0, 0.0, 0.10]), _box_column(),
                            np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0)


def test_the_jaws_touch_on_their_real_geometry_not_the_sweep_box():
    """A hinged jaw's touch is judged on its measured geometry (`now[1:]`):
    the inflated sweep box 3 mm over the top is no touch, the measured jaw
    3 mm over it is."""
    from cascade.skills import pointing

    far, near = np.zeros((1, 3)), np.array([[0.0, 0.0, 0.047]])      # TCP-frame; +z is the approach (down)
    start, goal = np.array([0.0, 0.0, 0.10]), np.array([0.0, 0.0, 0.15])
    box_only = pointing.Jaws(closed=(far, far), now=(near, far), sweep=(far, far))
    route = pointing.plan_route(_DownKin(), start, goal, _box_column(), box_only, 2, margin=MARGIN, table_z=0.0)
    assert route.withdraw == [] and route.targets == [goal], route
    jaw = pointing.Jaws(closed=(far, far), now=(far, near), sweep=(far, far))
    route = pointing.plan_route(_DownKin(), start, goal, _box_column(), jaw, 2, margin=MARGIN, table_z=0.0)
    assert route.withdraw_m == pytest.approx(0.005), route


@needs_pin
@pytest.mark.parametrize("gap, backs", [(0.002, True), (0.008, False)])
def test_point_at_backs_the_jaws_straight_out_of_a_touch_first(rig, gap, backs):
    """The mock arm parked top-down with its jaw box `gap` over the cube's
    top: within TOUCH_M it first backs straight up (the TCP on a vertical
    line, never closer to the cube), then points; beyond it, it goes as
    before. Every command keeps the escape rule; the jaws close after the
    withdrawal."""
    from cascade.skills import pointing

    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.05, 0.05))
    cube = r.runtime.beliefs.find("red cube")
    q = _park(r, [0.30, 0.0, 0.12])
    T = r.runtime.kin.fk(q)
    low = float((_tool_grid(r.cfg) @ T[:3, :3].T + T[:3, 3])[:, 2].min())
    q0 = _park(r, [0.30, 0.0, 0.12 - (low - float(cube.top_z)) + gap])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    _assert_every_command_keeps_the_margin(r)
    cmds = _commands(r)
    if not backs:
        assert "backed_out_m" not in out
        return
    assert 0 < out["backed_out_m"] <= pointing.WITHDRAW_MAX_M, out
    t0 = r.runtime.kin.fk(q0)[:3, 3]
    tcp = np.array([r.runtime.kin.fk(c)[:3, 3] for c in cmds])
    up = tcp[:, 2] - t0[2]
    n = int(np.argmax(up >= out["backed_out_m"] - 5e-4)) + 1     # the withdrawal's commands
    assert np.abs(tcp[:n, :2] - t0[:2]).max() < 1e-3, "the withdrawal left the line"
    assert np.all(np.diff(up[:n]) >= -1e-9), "the withdrawal went back down"
    assert r.jaws and r.jaws_at[0] >= n, (r.jaws_at, n)


@needs_pin
@pytest.mark.parametrize("why", ["no_path", "planner"])
def test_a_touching_start_on_an_arm_that_cannot_stream_a_path_refuses(rig, monkeypatch, why):
    """No `move_joint_path`, or a motion planner that owns every move (the
    harness refuses a raw joint path then): refused while planning."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.05, 0.05))
    cube = r.runtime.beliefs.find("red cube")
    q = _park(r, [0.30, 0.0, 0.12])
    T = r.runtime.kin.fk(q)
    low = float((_tool_grid(r.cfg) @ T[:3, :3].T + T[:3, 3])[:, 2].min())
    _park(r, [0.30, 0.0, 0.12 - (low - float(cube.top_z)) + 0.002])
    if why == "no_path":
        monkeypatch.setattr(r.runtime.arm, "move_joint_path", None)
    else:
        monkeypatch.setattr(r.runtime.arm, "motion_planner", SimpleNamespace(close=lambda: None))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "this arm cannot back them straight out" in out["error"], out


@needs_pin
@pytest.mark.parametrize("kind", ["halt", "estop"])
@pytest.mark.parametrize("when", ["during", "after"])
def test_a_stop_in_or_after_the_withdrawal_sends_no_further_command(rig, monkeypatch, kind, when):
    """A stop while the jaws back out (at its second streamed command) or
    right after it: not one more arm or jaw command, and no success."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.05, 0.05))
    cube = r.runtime.beliefs.find("red cube")
    q = _park(r, [0.30, 0.0, 0.12])
    T = r.runtime.kin.fk(q)
    low = float((_tool_grid(r.cfg) @ T[:3, :3].T + T[:3, 3])[:, 2].min())
    _park(r, [0.30, 0.0, 0.12 - (low - float(cube.top_z)) + 0.002])
    arm = r.arms[0]
    at_stop = {}

    def stop():
        if not at_stop:
            _stop(r.runtime, kind)
            at_stop.update(arm=len(arm.commands), jaws=len(r.jaws))

    if when == "during":
        send = arm.send_joint_target

        def send_then_stop(q):
            send(q)
            if len(arm.commands) >= 2:
                stop()

        arm.send_joint_target = send_then_stop
    else:
        real_path = r.runtime.arm.move_joint_path

        def path_then_stop(*a, **k):
            ok = real_path(*a, **k)
            stop()
            return ok

        monkeypatch.setattr(r.runtime.arm, "move_joint_path", path_then_stop)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert at_stop and at_stop["arm"] > 0, "the withdrawal never streamed"
    assert len(arm.commands) == at_stop["arm"] and len(r.jaws) == at_stop["jaws"] == 0
    assert out["ok"] is False, out


# ── planning has a time budget (#378 review M7) ────────────────────────────


def test_the_planning_budget_runs_out_on_its_clock(monkeypatch):
    from cascade.skills import pointing

    now = [100.0]
    monkeypatch.setattr(pointing, "_clock", lambda: now[0])
    budget = pointing.Budget(3.0)
    now[0] = 103.0
    budget.check()
    now[0] = 103.001
    with pytest.raises(pointing.PlanningTimeout, match=r"planning took longer than its 3\.0 s budget"):
        budget.check()
    # a long leg checks it at every pose it evaluates
    now[0] = 0.0
    ticks = iter(np.arange(0.0, 100.0, 0.25))
    monkeypatch.setattr(pointing, "_clock", lambda: float(next(ticks)))
    with pytest.raises(pointing.PlanningTimeout):
        pointing.plan_route(_SlideIK(), np.array([0.3, 0.0, 0.2]), np.array([-0.3, 0.0, 0.2]), _box_column(),
                            np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0, budget=pointing.Budget(3.0))


def test_the_lean_in_search_checks_the_budget(monkeypatch):
    """A spent budget stops the lean-in search before its next candidate."""
    from cascade.skills import pointing

    now = [0.0]
    monkeypatch.setattr(pointing, "_clock", lambda: now[0])
    spent = pointing.Budget(3.0)
    now[0] = 3.5
    square = np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]]) + [0.2, 0.0]
    column = pointing.ObjectColumn(footprints=(square,), tops=(0.05,), z_lo=0.0, aim=np.array([0.2, 0.0, 0.05]))
    assert pointing.lean_in(_SlideIK(), column, np.zeros((1, 3)), 2, np.zeros(3), margin=MARGIN,
                            table_z=0.0).solved > 0, "test setup: the search must try poses"
    with pytest.raises(pointing.PlanningTimeout):
        pointing.lean_in(_SlideIK(), column, np.zeros((1, 3)), 2, np.zeros(3), margin=MARGIN, table_z=0.0,
                         budget=spent)


@needs_pin_so101
@pytest.mark.parametrize("profile", ["mock", "so101_mock"])
def test_planning_past_its_budget_refuses_before_any_command(rig, monkeypatch, profile):
    """Every planning step reads the clock; one that has run 3 s refuses
    with that reason and sends nothing (g1_mock took up to 8.9 s)."""
    from cascade.skills import pointing

    r = rig(profile, camera="mock") if profile == "mock" else rig(profile, prop=(*at(0.22, 0.0), 0.035, 0.035))
    monkeypatch.setattr(pointing, "PLAN_BUDGET_S", 3.0)
    ticks = iter(np.arange(0.0, 1e6, 0.4))
    monkeypatch.setattr(pointing, "_clock", lambda: float(next(ticks)))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "planning took longer than its 3.0 s budget" in out["error"], out


# ── remembered objects without a size (#378 review M6) ─────────────────────


def test_an_object_remembered_without_a_size_is_a_conservative_column():
    """push_object and place_at remember only a place: a 10 x 10 cm square
    up to 0.20 m above the table (or 5 cm over the place), not 'not finite'."""
    from cascade.skills.runtime import UNSIZED_HALF_M, UNSIZED_HEIGHT_M, _belief_footprint

    pushed = SimpleNamespace(label="box", color="blue", position=np.array([0.3, 0.1, 0.02]), extent=None,
                             top_z=None, points=None)
    poly, top = _belief_footprint(pushed, table_z=0.0)
    assert (UNSIZED_HALF_M, UNSIZED_HEIGHT_M) == (0.05, 0.20)
    np.testing.assert_allclose(np.abs(poly - [0.3, 0.1]), 0.05)
    assert top == pytest.approx(0.20)
    high = SimpleNamespace(**{**vars(pushed), "position": np.array([0.3, 0.1, 0.18])})
    assert _belief_footprint(high, table_z=0.0)[1] == pytest.approx(0.23)
    with pytest.raises(ValueError, match="not finite"):
        _belief_footprint(SimpleNamespace(**{**vars(pushed), "position": np.array([0.3, np.nan, 0.1])}))


@needs_pin
def test_point_at_keeps_the_margin_from_an_object_remembered_without_a_size(rig):
    """A pushed box remembered at a place only, 9 cm from the cube: point_at
    no longer refuses 'not finite'; it keeps 3 cm from a 10 x 10 cm column
    0.20 m tall there, or refuses naming it."""
    r = rig("mock", camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    where = np.asarray(cube.position, float) + [0.0, 0.09, 0.0]
    r.runtime.beliefs.update("box", np.array([where[0], where[1], 0.015]), 0.6, color="blue")

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert "not finite" not in str(out.get("error")), out
    if out["ok"]:
        lo = np.array([where[0] - 0.05, where[1] - 0.05, 0.0])
        _assert_commands_keep_the_margin_from(r, lo, lo + [0.1, 0.1, 0.2])
    else:
        _refused_untouched(r, out)
        assert "the blue box (no remembered size: taken as a 10 x 10 cm column up to 0.20 m)" in out["error"], out


@needs_pin
def test_the_targets_own_record_without_a_size_is_not_a_neighbour(rig, monkeypatch):
    """push_object re-remembers the pushed object at its new place only: a
    second 'red cube' record inside the cube's own column is the cube."""
    r = rig("mock", camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    stale = SimpleNamespace(label=cube.label, color=cube.color, extent=None, top_z=None, points=None,
                            position=np.asarray(cube.position, float) + [0.005, 0.0, -0.01])
    real_all = r.runtime.beliefs.all
    monkeypatch.setattr(r.runtime.beliefs, "all", lambda: [*real_all(), stale])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    # the same record under another name is another object, taken as a column
    other = SimpleNamespace(**{**vars(stale), "label": "box", "color": "blue"})
    monkeypatch.setattr(r.runtime.beliefs, "all", lambda: [*real_all(), other])
    for arm in r.arms:
        arm.commands.clear()
    r.jaws.clear()
    out = r.runtime.execute("point_at", {"label": "red cube"})
    _refused_untouched(r, out)
    assert "the blue box (no remembered size" in out["error"], out


# ── an e-stop in the last close or the hold is no success (#378 review M5) ─


@needs_pin_so101
@pytest.mark.parametrize("where", ["close", "hold"])
def test_an_estop_in_the_final_close_or_the_hold_is_reported(rig, monkeypatch, where):
    """The close can be the last command (the so101 lean-in closes at the
    pointing pose): an e-stop latched in it, or during the hold, returns ok
    false naming the e-stop, with no command after it (main closed first and
    its next move refused)."""
    import threading

    import cascade.skills.runtime as R

    r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    runtime = r.runtime
    seen = {}
    if where == "close":
        real = runtime._set_gripper

        def close(*a, **k):
            res = real(*a, **k)
            runtime.arm.harness.estop("operator e-stop")
            seen["arm"] = len(_commands(r))
            return res

        monkeypatch.setattr(runtime, "_set_gripper", close)
    else:
        real_sleep = R.time.sleep

        class _Time:
            def __getattr__(self, name):
                return getattr(time, name)

            def sleep(self, s):
                if s >= 1.0 and not seen and threading.current_thread() is threading.main_thread():
                    runtime.arm.harness.estop("operator e-stop")
                    seen["arm"] = len(_commands(r))
                return real_sleep(min(s, 0.05))

        monkeypatch.setattr(R, "time", _Time())

    out = runtime.execute("point_at", {"label": "red cube"})

    assert seen, "the stop was never injected"
    if where == "close":
        assert r.jaws_at[-1] == len(_commands(r)), "test setup: the close must be the last command"
    assert out["ok"] is False and "e-stop" in out["error"], out
    assert len(_commands(r)) == seen["arm"] and len(r.jaws) == 1


# ── the rise never comes closer; the escape never dips between samples ────


def test_a_rise_leg_never_brings_the_jaws_closer_to_the_target():
    """Both ends 8-9 cm from the object (far outside the margin), the jaws
    closing in by 1.3 cm on the way: fine for a leg to the pointing pose,
    refused for a rise-first leg."""
    from cascade.skills import pointing

    start, goal = np.array([0.1, 0.0, 0.1]), np.array([0.06, 0.0, 0.12])
    args = (_box_column(), np.zeros((1, 3)), MARGIN, 0.0)
    assert pointing.segment_violation(_SlideKin(), start, goal, *args, from_inside=True) is None
    why = pointing.segment_violation(_SlideKin(), start, goal, *args, from_inside=True, rise=True)
    assert why is not None and why.endswith("closing in on the rise"), why
    # rising straight up only moves away
    assert pointing.segment_violation(_SlideKin(), start, start + [0.0, 0.0, 0.05], *args, from_inside=True,
                                      rise=True) is None


class _OutAndBack(_SlideKin):
    """q[0] from start[0] to start[0] - 0.04 moves the TCP 4.5 cm out along
    -x and 3.5 cm back (a joint-space arc): it ends 1 cm further out."""

    def __init__(self, start):
        self.start = np.asarray(start, float)

    def fk(self, q):
        s = float(np.clip((np.asarray(q, float)[0] - self.start[0]) / -0.04, 0.0, 1.0))
        T = np.eye(4)
        T[:3, 3] = self.start + [-(0.04 * math.sin(math.pi * s) + 0.01 * s), 0.0, 0.0]
        return T

    def lever_arms(self, q, tool):
        return np.full(3, math.pi + 0.25), np.zeros((0, 3))


def test_a_rise_leg_may_wiggle_but_never_below_where_it_started():
    """The rule is the leg's start clearance: a rise that goes away and comes
    partly back (8.0 -> 12 -> 9 cm) is admitted; one that ends at 7.9 cm is
    not."""
    from cascade.skills import pointing

    args = (_box_column(), np.zeros((1, 3)), MARGIN, 0.0)
    start = np.array([-0.1, 0.0, 0.05])                      # 8 cm beside the object
    up = start + [-0.04, 0.0, 0.0]
    assert pointing.segment_violation(_SlideKin(), start, up, *args, from_inside=True, rise=True) is None

    assert pointing.segment_violation(_OutAndBack(start), start, up, *args, from_inside=True, rise=True) is None
    assert pointing.segment_violation(_SlideKin(), start, start + [0.001, 0.0, 0.0], *args, from_inside=True,
                                      rise=True).endswith("closing in on the rise")


def test_a_rise_leg_that_left_the_margin_never_comes_back_in():
    """From 1 cm beside the object: out of the margin (5 cm) and back to 2 cm.
    Never below where it started, but back inside the margin: refused."""
    from cascade.skills import pointing

    start = np.array([-0.03, 0.0, 0.05])
    why = pointing.segment_violation(_OutAndBack(start), start, start + [-0.04, 0.0, 0.0], _box_column(),
                                     np.zeros((1, 3)), MARGIN, 0.0, from_inside=True, rise=True)
    assert why is not None and why.endswith("closing in on the rise"), why


def test_an_escape_never_dips_between_two_samples():
    """1 cm over the top, 6 cm across it in ONE sample step: both samples
    2.24 cm from the edge, 2.0 cm over the middle. The samples alone pass it;
    the dip bound refuses it."""
    from cascade.skills import pointing

    args = (_box_column(), np.zeros((1, 3)), MARGIN, 0.0)
    start, goal = np.array([-0.03, 0.0, 0.07]), np.array([0.03, 0.0, 0.07])
    why = pointing.segment_violation(_SlideKin(), start, goal, *args, step_rad=0.06, from_inside=True)
    assert why is not None and "between two samples" in why, why
    # straight up and out: the bound clears it without a single extra sample
    assert pointing.segment_violation(_SlideKin(), start, start + [0.0, 0.0, 0.1], *args, step_rad=0.06,
                                      from_inside=True) is None


@needs_pin_so101
def test_the_so101_escape_from_home_never_dips_below_where_it_started(rig, monkeypatch):
    """The commanded route (a straight joint segment per leg) resampled
    20001 times per leg: every pair that starts inside the margin keeps
    >= its start value - 0.05 mm until it is out. mock_small: home holds the
    sweep box 5 mm into the cube's top."""
    from cascade.skills import pointing

    r = rig("so101_mock")
    legs = []
    real_move = r.runtime._move_joints
    monkeypatch.setattr(r.runtime, "_move_joints", lambda q, *a, **k: (legs.append(np.asarray(q, float).copy()),
                                                                         real_move(q, *a, **k))[1])
    q0 = np.asarray(r.runtime.arm.get_state().q, float).copy()
    _, fix = r.runtime._localize("red cube")
    out = r.runtime.execute("point_at", {"label": "red cube"})
    assert out["ok"] is True, out

    column = pointing.object_column(fix, 0.0, float(fix.points[:, 2].max()))
    jaws = pointing.jaw_samples(r.cfg.grasp.graspgenx.sweep, r.cfg.arm.get("gripper"), "open_down", opening=0.0)
    q_from = q0
    for q_to in legs[:1]:
        start = pointing._pair_clearances(r.runtime.kin, q_from, (column,), jaws.now, 0.0)[0]
        inside = ~(start >= MARGIN)
        assert inside.any(), "test setup: home must start inside the margin"
        for s in np.linspace(0.0, 1.0, 20001):
            v = pointing._pair_clearances(r.runtime.kin, q_from + s * (q_to - q_from), (column,), jaws.now, 0.0)[0]
            assert (v[inside] >= start[inside] - pointing.DIP_TOL_M).all(), (s, (v - start)[inside].min())
            inside &= ~(v >= MARGIN)
            if not inside.any():
                break
        q_from = q_to


# ── held objects, occluder footprints, stops ──────────────────────────────


@needs_pin_so101
def test_holding_an_object_never_leans_in(rig, monkeypatch):
    """The payload is not modelled: with something in the jaws the SO-101
    refuses as main did (no top-down pose), before any command and without a
    single lean-in solve."""
    r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    r.runtime.held_object = "cup"
    calls = []
    real = r.runtime.kin.ik_approach
    monkeypatch.setattr(r.runtime.kin, "ik_approach", lambda *a, **k: calls.append(a) or real(*a, **k))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "while holding 'cup'" in out["error"] and calls == [], out


@needs_pin
def test_the_object_in_the_jaws_is_neither_an_occluder_nor_a_neighbour(rig, monkeypatch):
    """The held cup's belief hangs over the cube with the jaws: it is not an
    object standing over the target (the hover counts from the cube's top)."""
    r = rig("mock", camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    r.runtime.beliefs.update("cup", np.array([cube.position[0], cube.position[1], 0.15]), 0.9,
                             extent=np.array([0.1, 0.08, 0.08]), top_z=0.2, color="blue")
    r.runtime.held_object = "cup"
    asked = []
    real_ik = r.runtime.kin.ik
    monkeypatch.setattr(r.runtime.kin, "ik", lambda T, *a, **k: (asked.append(float(T[2, 3])), real_ik(T, *a, **k))[1])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and r.jaws == [], out
    assert asked[0] == pytest.approx(0.05 + 0.04, abs=1e-6)


@needs_pin
def test_holding_an_object_points_top_down_without_a_jaw_command(rig):
    r = rig("mock", camera="mock")
    r.runtime.held_object = "cup"

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and "lean_deg" not in out, out
    assert r.jaws == [] and _commands(r)


def test_a_tall_object_beside_the_target_has_its_girth_as_footprint():
    """A 20 cm bottle (extents 0.2 x 0.06 x 0.06): its footprint is a square
    of half-side hypot(0.06, 0.06) / 2, not 10 cm."""
    from cascade.skills.runtime import _belief_footprint

    bottle = SimpleNamespace(label="bottle", color="blue", position=np.array([0.3, 0.1, 0.1]),
                             extent=np.array([0.2, 0.06, 0.06]), top_z=0.2, points=None)
    poly, top = _belief_footprint(bottle)
    assert top == pytest.approx(0.2)
    np.testing.assert_allclose(np.abs(poly - [0.3, 0.1]).max(axis=0), [math.hypot(0.06, 0.06) / 2] * 2)
    # a flat tray (0.3 x 0.2 x 0.02): the two large extents are horizontal
    tray = SimpleNamespace(label="tray", color=None, position=np.array([0.3, 0.0, 0.01]),
                           extent=np.array([0.3, 0.2, 0.02]), top_z=0.02, points=None)
    poly, _ = _belief_footprint(tray)
    np.testing.assert_allclose(np.abs(poly - [0.3, 0.0]).max(axis=0), [math.hypot(0.3, 0.2) / 2] * 2)


@pytest.mark.parametrize("field, value", [("position", np.array([0.3, np.nan, 0.1])), ("top_z", float("nan")),
                                          ("extent", np.array([0.2, np.inf, 0.06]))])
def test_a_non_finite_remembered_object_raises(field, value):
    from cascade.skills.runtime import _belief_footprint

    bottle = dict(label="bottle", color="blue", position=np.array([0.3, 0.1, 0.1]),
                  extent=np.array([0.2, 0.06, 0.06]), top_z=0.2, points=None)
    bottle[field] = value
    with pytest.raises(ValueError, match="the blue bottle's remembered box is not finite"):
        _belief_footprint(SimpleNamespace(**bottle))


@needs_pin
def test_a_20cm_bottle_8cm_away_is_not_an_occluder(rig, monkeypatch):
    """#372 took half its largest extent (10 cm) as its footprint radius, so
    it 'stood over' the cube and the hover counted from 0.2 m. Its girth
    does not reach the cube: today's hover, from the cube's own top. It is
    a neighbour, though: that hover's jaws would pass 2 cm from it, so
    nothing moves (#378 review Critical 1: they did, with ok: true)."""
    r = rig("mock", camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    r.runtime.beliefs.update("bottle", np.array([cube.position[0], cube.position[1] + 0.08, 0.1]), 0.9,
                             extent=np.array([0.2, 0.06, 0.06]), top_z=0.2, color="blue")
    asked = []
    real_ik = r.runtime.kin.ik
    monkeypatch.setattr(r.runtime.kin, "ik", lambda T, *a, **k: (asked.append(float(T[2, 3])), real_ik(T, *a, **k))[1])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert asked[0] == pytest.approx(0.05 + 0.04, abs=1e-6)
    _refused_untouched(r, out)
    assert "from above, the best solved pose kept the arm 0.0" in out["error"], out
    assert "from the blue bottle, less than 0.030 m" in out["error"], out


@needs_pin
def test_a_non_finite_remembered_object_refuses_before_any_command(rig):
    r = rig("mock", camera="mock")
    cube = r.runtime.beliefs.find("red cube")
    r.runtime.beliefs.update("cup", np.array([cube.position[0], cube.position[1], 0.05]), 0.9,
                             extent=np.array([0.1, 0.09, 0.09]), top_z=0.1, color="blue")
    r.runtime.beliefs.find("cup").top_z = float("nan")

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "the blue cup's remembered box is not finite" in out["error"], out


@needs_pin
@pytest.mark.parametrize("kind", ["halt", "estop"])
def test_a_stop_while_vetting_the_rise_is_reported_as_the_stop(rig, monkeypatch, kind):
    """The harness's route vet answers a latched e-stop with a reason, which
    used to read as geometry ("no rise-first waypoint clears it"). The stop
    is reported as the stop, and nothing moves."""
    r = rig("mock", camera="mock", prop=(0.30, 0.0, 0.06, 0.10))
    _park(r, [0.18, 0.0, 0.06], yaw=math.pi / 2)
    harness = r.runtime.arm.harness
    real = harness.vet_step

    def vet_step(*a, **k):
        _stop(r.runtime, kind)
        return real(*a, **k)

    monkeypatch.setattr(harness, "vet_step", vet_step)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    _assert_stopped(r, out, kind)


def _assert_stopped(r, out, kind):
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "no rise-first waypoint" not in out["error"] and "leaning in" not in out["error"], out
    assert ("e-stop" in out["error"]) if kind == "estop" else ("halt" in out["error"].lower()), out


@needs_pin_so101
def test_an_estop_while_planning_past_the_budget_is_reported_as_the_estop(rig, monkeypatch):
    """An e-stop latched while planning vetoes every vet, so the search runs
    long; when the budget then runs out the e-stop is reported, not the clock
    (a slow CI runner hit exactly this)."""
    from cascade.skills import pointing

    r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    monkeypatch.setattr(pointing, "PLAN_BUDGET_S", 3.0)
    ticks = iter(np.arange(0.0, 1e6, 0.4))
    monkeypatch.setattr(pointing, "_clock", lambda: float(next(ticks)))
    real = r.runtime.kin.ik_approach

    def ik_approach(*a, **k):
        if not r.runtime.arm.harness.estopped:
            _stop(r.runtime, "estop")
        return real(*a, **k)

    monkeypatch.setattr(r.runtime.kin, "ik_approach", ik_approach)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert r.runtime.arm.harness.estopped, "the lean-in search never ran"
    _assert_stopped(r, out, "estop")


@needs_pin_so101
@pytest.mark.parametrize("kind", ["halt", "estop"])
def test_a_stop_while_vetting_the_lean_in_is_reported_as_the_stop(rig, monkeypatch, kind):
    r = rig("so101_mock", prop=(*at(0.22, 0.0), 0.035, 0.035))
    harness = r.runtime.arm.harness
    real = harness.vet_pose

    def vet_pose(*a, **k):
        _stop(r.runtime, kind)
        return real(*a, **k)

    monkeypatch.setattr(harness, "vet_pose", vet_pose)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    _assert_stopped(r, out, kind)


# ── MuJoCo physics: the prop stays where it was ────────────────────────────

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True


needs_mujoco = pytest.mark.skipif(not (_has_mujoco() and MJCF.exists()),
                                  reason="needs mujoco + `python scripts/fetch_robot_assets.py so101`")


def _mujoco_point_at(monkeypatch, tmp_path, prop, touching=None):
    """point_at on so101_mujoco with the prop at (x, y, side, height): (result,
    the largest prop displacement, arm targets sent, jaw commands sent).
    `touching` collects the arm bodies the prop touches before the call."""
    run = _mujoco_run(monkeypatch, tmp_path, prop)
    if touching is not None:
        touching.extend(run.touching)
    return run.out, run.moved, run.sent, run.jaws


#: the control's straight lift of the TCP along its approach axis (m)
LIFT_M = 0.06


def _mujoco_run(monkeypatch, tmp_path, prop, lift=False):
    """so101_mujoco from home with the prop at (x, y, side, height):
    point_at, or (`lift`) the control: the TCP straight back LIFT_M along its
    approach (a dense joint path, as one motion). A namespace: out, moved
    (the largest prop displacement), sent / jaws (arm targets and jaw
    commands, with the number of targets sent before each jaw command),
    touching (the arm bodies the prop touched before), depth (the prop's
    distance to the arm's collision geometry where it starts, and the least
    over every commanded pose, the jaws at their commanded angle and every
    close swept: `_prop_depth`)."""
    monkeypatch.setenv("MUJOCO_GL", "disable")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.sim.truth import make_truth_pose_fn

    cfg = load_demo_config(camera="mock_small", arm="so101_mujoco", llm="mock")
    if prop is not None:
        place_prop(cfg, *prop)
    runtime, raw = demo.build_runtime(cfg, tmp_path / ("lift" if lift else "run"))
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        truth = make_truth_pose_fn(raw)
        before = {k: np.asarray(v, float)[:3].copy() for k, v in truth.all_poses().items()}
        assert before, "the scene has no prop"
        qpos0 = np.array(raw._engine.data.qpos, dtype=float).copy()
        q0 = np.asarray(raw.get_state().q, float).copy()
        grip0 = float(raw.get_state().gripper_pos)
        sent, jaws = [], []
        send, grip = raw.send_joint_target, raw.set_gripper
        monkeypatch.setattr(raw, "send_joint_target", lambda q: (sent.append(np.asarray(q, float).copy()), send(q))[1])
        monkeypatch.setattr(raw, "set_gripper", lambda *a, **k: (jaws.append((len(sent), float(a[0]))),
                                                                 grip(*a, **k))[1])

        touching = _prop_contacts(raw)
        if lift:
            # the backend directly (a physics control, not a gated motion):
            # the TCP straight back along its approach, 5 mm IK steps from the
            # previous pose (the approach held, the roll free: a 5-joint arm),
            # as far as the line solves (the SO-101 at home: 2.5 cm), up to LIFT_M
            T = np.asarray(runtime.kin.fk(q0), float)
            col = _approach_col(cfg)
            path, q = [], q0
            for k in range(1, int(round(LIFT_M / 0.005)) + 1):
                res = runtime.kin.ik_approach(T[:3, 3] - k * 0.005 * T[:3, col], T[:3, col], q, np.eye(3)[:, col],
                                              retries=0)
                q_next = np.asarray(res.q, float)[: q0.size]
                if not (res.success and np.abs(q_next - q).max() < 0.1):
                    break
                path.append(q_next)
                q = q_next
            assert len(path) >= 4, f"control: the straight lift ends {len(path) * 5} mm up"
            raw.stream_path(path, duration_s=2.0)
            out = {"ok": None}
        else:
            out = runtime.execute("point_at", {"label": "red cube"})
        time.sleep(0.3)

        after = {k: np.asarray(v, float)[:3] for k, v in truth.all_poses().items()}
        moved = max(float(np.linalg.norm(after[k] - before[k])) for k in before)
        closed = float(cfg.arm.get("gripper").get("closed_pos"))
        # copies under new names: the shutdown below parks the arm through
        # the same hooks, which append to `sent` / `jaws` (closure cells)
        sent_now, jaws_now = list(sent), list(jaws)
        depth = _prop_depth(raw._engine.model, qpos0, q0, grip0, sent_now, jaws_now, closed)
        return SimpleNamespace(out=out, moved=moved, sent=sent_now, jaws=jaws_now, touching=touching, depth=depth)
    finally:
        demo.shutdown_runtime(runtime, raw)


_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _prop_depth(m, qpos0, q0, grip0, sent, jaws, closed) -> tuple[float, float]:
    """(where it starts, the least): the red cube's distance (MuJoCo
    `mj_geomDistance`, negative = inside) to the arm's collision geoms, the
    props where they started, the arm at its start and at every commanded
    target, the jaws at their commanded angle, every jaw command swept from
    the angle before it."""
    import mujoco

    d = mujoco.MjData(m)
    adr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in (*_ARM_JOINTS, "gripper")}
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "" for g in range(m.ngeom)]
    cube = [g for g in range(m.ngeom) if names[g] == "red_cube"]
    arm = [g for g in range(m.ngeom) if names[g] not in ("world", "") and not names[g].endswith("cube")
           and not names[g].startswith("extra") and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert cube and arm, "test setup: no red cube or no arm geoms"
    fromto = np.zeros(6)

    def dist(q, angle):
        d.qpos[:] = qpos0
        for n, v in zip(_ARM_JOINTS, q):
            d.qpos[adr[n]] = v
        d.qpos[adr["gripper"]] = angle
        mujoco.mj_forward(m, d)
        return min(mujoco.mj_geomDistance(m, d, c, g, 0.1, fromto) for c in cube for g in arm)

    angle, poses = grip0, [np.asarray(q0, float)] + list(sent)
    start = dist(poses[0], angle)
    least = start
    commands = iter(sorted(jaws))
    nxt = next(commands, None)
    for i, q in enumerate(poses):
        while nxt is not None and nxt[0] == i:          # a jaw command before target i (or after the last)
            for a in np.linspace(angle, nxt[1], 12):
                least = min(least, dist(poses[i], a))
            angle = nxt[1]
            nxt = next(commands, None)
        least = min(least, dist(q, angle))
    return start, least


def _prop_contacts(raw) -> list:
    """The arm's bodies the prop touches (MuJoCo contacts), floor aside."""
    import mujoco

    m, d = raw._engine.model, raw._engine.data
    found = []
    for c in d.contact[:d.ncon]:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "" for g in (c.geom1,
                                                                                                        c.geom2)]
        if "red_cube" in names and "world" not in names:
            found.append(sorted(names))
    return found


@needs_pin_so101
@needs_mujoco
@pytest.mark.parametrize("side, radius, az", [(None, None, None), (0.035, 0.18, 0.0), (0.05, 0.22, 0.5),
                                              (0.035, 0.26, -0.5), (0.05, 0.26, 0.0)])
def test_so101_mujoco_leans_in_and_leaves_the_prop_where_it_was(monkeypatch, tmp_path, side, radius, az):
    """(None: mock_small's own 35 x 35 x 50 mm prop at r 0.20. Before #369 the
    closed jaws came down to 0.04 m on it and pushed it 7.7 mm; #369 refused.)"""
    prop = None if side is None else (*at(radius, az), side, side)
    out, moved, sent, jaws = _mujoco_point_at(monkeypatch, tmp_path, prop)

    assert moved < 0.001, f"the prop moved {moved * 1000:.1f} mm: {out}"
    assert out["ok"] is True and out["lean_deg"] in LEAN_TILTS, out
    assert sent and len(jaws) == 1


@needs_pin_so101
@needs_mujoco
def test_so101_mujoco_refuses_inside_its_annulus_and_sends_nothing(monkeypatch, tmp_path):
    out, moved, sent, jaws = _mujoco_point_at(monkeypatch, tmp_path, (0.14, 0.0, 0.05, 0.05))

    assert moved < 0.001, f"the prop moved {moved * 1000:.1f} mm: {out}"
    assert out["ok"] is False and "nothing was moved" in out["error"], out
    assert sent == [] and jaws == []


# The reviewer's repro: a 49.6 mm x 54.5 mm prop at r 0.2331, az 0.0369, its
# top 0.3 mm under the jaws at home. #372 closed the jaws there before moving:
# the prop moved 12.9 mm and point_at said ok.
CLOSE_REPRO = (0.23312004552150348, 0.0369095717248229, 0.04956236216333736, 0.05454514680454995)


@needs_pin_so101
@needs_mujoco
def test_so101_mujoco_never_closes_the_jaws_into_a_prop_beside_them(monkeypatch, tmp_path):
    radius, az, side, height = CLOSE_REPRO
    touching = []
    out, moved, sent, jaws = _mujoco_point_at(monkeypatch, tmp_path, (*at(radius, az), side, height), touching)

    assert touching == [], f"test setup: the prop must not spawn touching the arm ({touching})"
    assert moved < 0.001, f"the prop moved {moved * 1000:.1f} mm: {out}"
    if out["ok"] is not True:
        assert sent == [] and jaws == [] and "nothing was moved" in out["error"], out


def _sweep_props() -> list:
    """27 random props (seed 37202: r 0.17-0.27 m, az +-0.6 rad, side 3-5 cm,
    height 3-7 cm), every draw: three (0, 16, 25) spawn 1-7.5 mm deep through
    the open jaws at home."""
    rng = np.random.default_rng(37202)
    return [tuple(float(rng.uniform(lo, hi)) for lo, hi in ((0.17, 0.27), (-0.6, 0.6), (0.03, 0.05), (0.03, 0.07)))
            for _ in range(27)]


#: the #378 review's props that spawn touching the open jaws at home
#: (-0.5..0.1 mm: its rows 15-20; row 17 was pushed 4.9 mm with ok: true) and
#: one 3 mm inside them (its row 48, pushed 9.2 mm), as (r, az, side, height)
TOUCHING_PROPS = [
    (0.25734242237027244, 0.08925956387866209, 0.038745371620075936, 0.10482883081992844),
    (0.23937353027307368, -0.2912019679078725, 0.03259293691162826, 0.07101454004912552),
    (0.251368827244064, -0.09388168869707424, 0.042772055144915826, 0.0632440173211075),
    (0.16761774466685683, 0.14442380210101538, 0.031111323835186486, 0.09409644281004863),
    (0.2288725523360636, 0.08515222590635628, 0.04382901120232272, 0.09768749606812635),
    (0.17093100559115967, 0.13095612812444785, 0.047534798071054936, 0.05856056091506852),
    (0.1966719978476447, -0.06532871111329619, 0.0392272200583173, 0.06706985553620733),
]


@needs_pin_so101
@needs_mujoco
@pytest.mark.parametrize("prop", _sweep_props() + TOUCHING_PROPS,
                         ids=lambda p: f"r{p[0]:.3f}-az{p[1]:+.2f}-s{p[2]:.3f}-h{p[3]:.3f}")
def test_so101_mujoco_never_pushes_a_prop_more_than_a_straight_lift(monkeypatch, tmp_path, prop):
    """Physics, not the code's own geometry, with props touching and inside
    the open jaws too (#378 review Important 3). Every ok: true never takes
    the arm deeper into the prop than where it started (MuJoCo collision
    geometry, every commanded pose, every close swept) and moves it no more
    than 1 mm beyond what a straight lift (LIFT_M back along the approach,
    from the same start) moves it; a refusal sends nothing."""
    radius, az, side, height = prop
    placed = (*at(radius, az), side, height)
    run = _mujoco_run(monkeypatch, tmp_path, placed)
    start, least = run.depth
    print(f"\n[sweep] r{radius:.3f} az{az:+.2f} ok={run.out['ok']} start={start * 1000:.2f}mm least={least * 1000:.2f}mm "
          f"moved={run.moved * 1000:.3f}mm backed={run.out.get('backed_out_m')} lean={run.out.get('lean_deg')} "
          f"error={str(run.out.get('error'))[:160]}")

    if run.out["ok"] is not True:
        assert run.sent == [] and run.jaws == [] and "nothing was moved" in run.out["error"], run.out
        return
    assert least >= min(start, 0.0) - 2e-4, \
        f"the arm went {(min(start, 0.0) - least) * 1000:.2f} mm deeper into the prop than where it started: {run.out}"
    allowed = 0.001
    if run.moved >= allowed and start < 0.001:     # it starts touching: a straight lift moves it too
        allowed += _mujoco_run(monkeypatch, tmp_path, placed, lift=True).moved
        print(f"[sweep]   straight lift moved it {(allowed - 0.001) * 1000:.3f}mm")
    assert run.moved < allowed, f"the prop moved {run.moved * 1000:.2f} mm (allowed {allowed * 1000:.2f}): {run.out}"
    assert len(run.jaws) == 1, run.jaws


def _surface(m, g, step=0.002):
    """Surface samples of MuJoCo collision geom g in its own frame."""
    import mujoco

    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        faces = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        pts = [verts]
        for f in faces:
            a, b, c = verts[f]
            n = max(1, int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a)) / step)))
            u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
            k = (u + v) <= 1.0
            pts.append(a + np.outer(u[k], b - a) + np.outer(v[k], c - a))
        return np.vstack(pts)
    half = {int(mujoco.mjtGeom.mjGEOM_BOX): s, int(mujoco.mjtGeom.mjGEOM_SPHERE): np.full(3, s[0]),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): np.array([s[0], s[0], s[1] + s[0]])}[int(t)]
    lines = [np.linspace(-h, h, max(2, int(np.ceil(2 * h / step)) + 1)) for h in half]
    grid = np.stack(np.meshgrid(*lines, indexing="ij"), -1).reshape(-1, 3)
    return grid[np.isclose(np.abs(grid), half).any(axis=1)]


def _jaw_volume_outside(vol, points, which):
    """`points` (TCP frame; the moving jaw's turned back to the stop) that no
    `jaw_volume` box of `which` holds (0.1 mm tolerance)."""
    from cascade.safety.tool_volume import jaw_box

    inside = np.zeros(len(points), dtype=bool)
    for b in vol.get(which):
        centre, axes, half = jaw_box({"extents_open": b.get("extents"), "offset_open": b.get("offset")}, "open_down")
        inside |= (np.abs((points - centre) @ axes) <= half + 1e-4).all(axis=1)
    return points[~inside]


def _so101_jaw_volume():
    from cascade.config import load_demo_config

    acfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock").arm
    return acfg, acfg.gripper.get("jaw_volume")


@needs_pin_so101
@needs_mujoco
def test_the_so101_jaw_volume_holds_the_mujoco_jaws_at_every_angle():
    """so101.yaml `gripper.jaw_volume` against the Menagerie model: every
    surface, collision AND visual (the visual geoms are the STL meshes), of
    the gripper body and its wrist camera mount (fixed boxes) and of the
    moving jaw (its boxes, turned about the hinge from `stop_rad` to the
    angle) lies inside the configured boxes, at gripper angles from the stop
    to 1.0 rad (past open_pos 0.60). (#378 review: the STL meshes poked out
    7.8 mm and 4.1 mm, the camera mount 60 mm.)"""
    import mujoco

    from cascade.control.kinematics import Kinematics
    from cascade.skills import pointing

    acfg, vol = _so101_jaw_volume()
    kin = Kinematics(acfg.get("model"), acfg.get("ee_frame"), int(acfg.get("n_joints")), acfg.get("joint_signs"))
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    q = np.asarray(acfg.get("home_q"), float)
    for name, v in zip(acfg.get("mj_joints") or ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                 "wrist_roll"], q):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = v
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    moving = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("gripper", "camera_mount")} | {moving}
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]
    kinds = {"collision": 0, "visual": 0}
    for g in geoms:
        kinds["collision" if (m.geom_contype[g] or m.geom_conaffinity[g]) else "visual"] += 1
    assert kinds["collision"] >= 10 and kinds["visual"] >= 4, f"test setup: {kinds}"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "camera_mount") >= 0
    local = {g: _surface(m, g) for g in geoms}
    hinge = np.asarray(vol.get("hinge").get("point"), float)
    axis = np.asarray(vol.get("hinge").get("axis"), float)
    inv = np.linalg.inv(kin.fk(q))

    stop = float(vol.get("stop_rad"))
    assert stop == pytest.approx(float(m.jnt_range[grip][0]), abs=1e-4), "stop_rad is the MJCF's closed stop"
    for angle in np.linspace(stop, 1.0, 25):
        d.qpos[m.jnt_qposadr[grip]] = angle
        mujoco.mj_forward(m, d)
        for g in geoms:
            world = d.geom_xpos[g] + local[g] @ d.geom_xmat[g].reshape(3, 3).T
            tcp = world @ inv[:3, :3].T + inv[:3, 3]
            if m.geom_bodyid[g] == moving:     # turned back about the hinge to the stop
                tcp = (tcp - hinge) @ pointing._rotation(axis, stop - angle).T + hinge
                bad = _jaw_volume_outside(vol, tcp, "moving")
            else:
                bad = _jaw_volume_outside(vol, tcp, "fixed")
            assert not len(bad), (f"geom {g} at {angle:.3f} rad pokes out of the jaw volume by "
                                  f"{np.round(bad[0] * 1000, 1).tolist()} mm (TCP frame)")


STL_DIR = REPO / "assets" / "mjcf" / "so101" / "assets"


def _read_stl(path) -> np.ndarray:
    """(n, 3, 3) triangles of a binary or ASCII STL file."""
    import struct

    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        verts = [[float(v) for v in ln.split()[1:4]] for ln in data.decode().splitlines()
                 if ln.strip().startswith("vertex")]
        return np.asarray(verts, float).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)


def _triangle_samples(tris, step=0.0015) -> np.ndarray:
    """The vertices and a grid at most `step` apart on every triangle."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n_all = np.maximum(1, np.ceil(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)) / step))
    out = [tris.reshape(-1, 3)]
    for n in np.unique(n_all.astype(int)):
        k = np.flatnonzero(n_all == n)
        uu, vv = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        sel = (uu + vv) <= 1 + 1e-12
        out.append((a[k, None, :] + uu[sel][None, :, None] * (b[k] - a[k])[:, None, :]
                    + vv[sel][None, :, None] * (c[k] - a[k])[:, None, :]).reshape(-1, 3))
    return np.vstack(out)


@needs_pin_so101
@pytest.mark.skipif(not (STL_DIR / "moving_jaw_so101_v1.stl").exists(),
                    reason="needs `python scripts/fetch_robot_assets.py so101` (the STL meshes)")
def test_the_so101_jaw_volume_holds_the_urdf_stl_meshes_at_every_angle():
    """The same check on the real parts, without MuJoCo: the URDF's own
    visual meshes (the STL files, read here) placed by the URDF through
    pinocchio, gripper_link fixed and moving_jaw_so101_v1_link turned back
    about the hinge, at gripper angles from the stop to 1.0 rad."""
    import xml.etree.ElementTree as ET

    import pinocchio as pin

    from cascade.skills import pointing

    acfg, vol = _so101_jaw_volume()
    urdf = SO101_URDF
    pm = pin.buildModelFromUrdf(str(urdf))
    pdata = pm.createData()
    parts = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        if link.get("name") not in ("gripper_link", "moving_jaw_so101_v1_link"):
            continue
        for vis in link.findall("visual"):
            o = vis.find("origin")
            T = np.eye(4)
            T[:3, :3] = pin.rpy.rpyToMatrix(*[float(v) for v in o.get("rpy").split()])
            T[:3, 3] = [float(v) for v in o.get("xyz").split()]
            mesh = STL_DIR / vis.find("geometry/mesh").get("filename").split("/")[-1]
            parts.setdefault(link.get("name"), []).append((mesh.name, T, _triangle_samples(_read_stl(mesh))))
    assert sorted(len(v) for v in parts.values()) == [1, 2], f"test setup: {parts.keys()}"
    hinge = np.asarray(vol.get("hinge").get("point"), float)
    axis = np.asarray(vol.get("hinge").get("axis"), float)
    stop = float(vol.get("stop_rad"))
    q5 = np.asarray(acfg.get("home_q"), float)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for angle in np.linspace(stop, 1.0, 9):
        qf = np.zeros(pm.nq)
        for name, v in zip(names, [*q5, angle]):
            qf[pm.idx_qs[pm.getJointId(name)]] = v
        pin.forwardKinematics(pm, pdata, qf)
        pin.updateFramePlacements(pm, pdata)
        inv = np.linalg.inv(np.array(pdata.oMf[pm.getFrameId(acfg.get("ee_frame"))].homogeneous))
        for link, meshes in parts.items():
            T_link = inv @ np.array(pdata.oMf[pm.getFrameId(link)].homogeneous)
            for mesh, T_mesh, pts in meshes:
                T = T_link @ T_mesh
                tcp = pts @ T[:3, :3].T + T[:3, 3]
                if link == "moving_jaw_so101_v1_link":
                    tcp = (tcp - hinge) @ pointing._rotation(axis, stop - angle).T + hinge
                    bad = _jaw_volume_outside(vol, tcp, "moving")
                else:
                    bad = _jaw_volume_outside(vol, tcp, "fixed")
                assert not len(bad), (f"{mesh} at {angle:.3f} rad pokes out of the jaw volume at "
                                      f"{np.round(bad[0] * 1000, 1).tolist()} mm (TCP frame)")
