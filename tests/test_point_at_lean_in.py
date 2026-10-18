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
from conftest import REPO, needs_pin, needs_pin_so101

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

    built = []

    def build(profile, camera="mock_small", prop=None):
        cfg = load_demo_config(camera=camera, arm=profile, llm="mock")
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        if prop is not None:
            place_prop(cfg, *prop)
        r = SimpleNamespace(cfg=cfg, jaws=[], arms=[])

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
    assert column.distance(np.array([[0.3, 0.0, 0.001]]))[0] == 0.0
    # a stray point outside the box widens the footprint
    assert column.distance(np.array([[0.3, 0.05, 0.05]]))[0] == 0.0
    # above the top by 2 cm
    assert column.distance(np.array([[0.3, 0.0, 0.13]]))[0] == pytest.approx(0.02)
    np.testing.assert_allclose(column.aim, [0.3, 0.0, 0.11])


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
    """FK = a pure translation by q (3 joints); no joint origins to check."""

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q, dtype=float)[:3]
        return T

    def link_positions(self, q):
        return np.zeros((1, 3))


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
])
def test_the_route_rule_on_a_straight_segment(start, goal, from_inside, ok):
    from cascade.skills import pointing

    reason = pointing.segment_violation(_SlideKin(), np.array(start), np.array(goal), _box_column(),
                                        np.zeros((1, 3)), MARGIN, 0.0, from_inside=from_inside)
    assert (reason is None) is ok, reason


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


def _mujoco_point_at(monkeypatch, tmp_path, prop):
    monkeypatch.setenv("MUJOCO_GL", "disable")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.sim.truth import make_truth_pose_fn

    cfg = load_demo_config(camera="mock_small", arm="so101_mujoco", llm="mock")
    if prop is not None:
        place_prop(cfg, *prop)
    runtime, raw = demo.build_runtime(cfg, tmp_path / "run")
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        truth = make_truth_pose_fn(raw)
        before = {k: np.asarray(v, float)[:3].copy() for k, v in truth.all_poses().items()}
        assert before, "the scene has no prop"
        sent, jaws = [], []
        send, grip = raw.send_joint_target, raw.set_gripper
        monkeypatch.setattr(raw, "send_joint_target", lambda q: (sent.append(q), send(q))[1])
        monkeypatch.setattr(raw, "set_gripper", lambda *a, **k: (jaws.append(a), grip(*a, **k))[1])

        out = runtime.execute("point_at", {"label": "red cube"})
        time.sleep(0.3)

        after = {k: np.asarray(v, float)[:3] for k, v in truth.all_poses().items()}
        moved = max(float(np.linalg.norm(after[k] - before[k])) for k in before)
        # copies: the shutdown below parks the arm through the same hooks
        return out, moved, list(sent), list(jaws)
    finally:
        demo.shutdown_runtime(runtime, raw)


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
