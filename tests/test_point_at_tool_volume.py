"""point_at keeps the margin with the harness's whole tool (#385 review I1, M1, M5).

The reBot's harness checks its wrist camera (`safety.tool_volume`: a 15 x 14 x 8
cm box on top of the gripper) against the occupancy map at every step, but
point_at's own check of known objects modelled only the jaws and the joint
origins: a top-down pose could park the camera box 1-2 cm from a remembered
bottle with ok: true (the #385 review's random rows: 8 of 29 ok rows on
rebot_rs / rebot_rs_reach, up to 18.5 mm inside a neighbour). Only the live map
caught that on the rig.

Now every part of the harness's tool volume beyond the jaws (`pointing.tool_body`:
the camera box, sampled on its faces) is a tool part of its own: the top-down
pose, the lean-in pose, every route sample, rise and withdrawal keep it the
margin from the target, every known object and the table, or nothing moves.
A part's sampled clearance minus its `slack_m` bounds the BOX's clearance from
below, so an accepted pose keeps the margin from the whole box, not only from
its samples. Profiles without a tool volume plan exactly as before.

The oracle here is independent of `pointing`: the camera box from the profile's
numbers (`relative_motion.wrist_camera`, `tool_up_axis`) and the exact distance
between two boxes (separating axes, else vertex-to-box and edge-to-edge).
"""

from __future__ import annotations

import math
import re
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin

from cascade.control.mock_arm import MockArm

#: safety.min_clearance_m in configs/demo.yaml (no arm profile overrides it)
MARGIN = 0.03
REBOT_PROFILES = ("rebot_rs", "rebot_rs_reach")


class _RecArm(MockArm):
    def __init__(self, cfg, kinematics, log):
        super().__init__(cfg, kinematics)
        self.log = log

    def set_gripper(self, pos, effort=1.0):
        self.log.jaws.append(float(pos))
        super().set_gripper(pos, effort)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    def build(profile, camera="mock"):
        cfg = load_demo_config(camera=camera, arm=profile, llm="mock")
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
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


# ── the oracle: the camera box from the profile's numbers, exact box distances ──

_ORDER = {"down_open": (0, 1, 2), "open_down": (2, 0, 1), "third_open_down": (2, 1, 0)}


def _camera_box_tcp(cfg):
    """(centre, axes, half) of the wrist-camera box in the TCP frame."""
    wc = cfg.get("relative_motion").get("wrist_camera")
    up = np.asarray(cfg.get("relative_motion").get("tool_up_axis"), float)
    a = np.eye(3)[:, _ORDER[str(cfg.arm.get("tool_axis_order", "down_open"))][0]]
    u = up - float(up @ a) * a
    u /= np.linalg.norm(u)
    behind, top, hw = float(wc.get("behind_tip_m")), float(wc.get("protrusion_m")), float(wc.get("half_width_m"))
    return -a * behind / 2.0 + u * top / 2.0, np.column_stack([a, np.cross(u, a), u]), np.array([behind / 2, hw, top / 2])


def _corners(c, R, h):
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return np.asarray(c, float) + (s * h) @ np.asarray(R, float).T


_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def _point_box(P, c, R, h):
    return np.linalg.norm(np.maximum(np.abs((P - c) @ R) - h, 0.0), axis=1)


def _segments(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, b = (d1 * d1).sum(1), (d2 * d2).sum(1), (d1 * d2).sum(1)
    c, f = (d1 * r).sum(1), (d2 * r).sum(1)
    den = a * e - b * b
    s = np.where(den > 1e-15, np.clip((b * f - c * e) / np.where(den > 1e-15, den, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p1 + d1 * s[:, None] - p2 - d2 * t[:, None], axis=1)


def box_distance(a, b) -> float:
    """Exact distance between two oriented boxes (centre, axes, half); <= 0 when they meet."""
    (ca, Ra, ha), (cb, Rb, hb) = a, b
    A, B = _corners(ca, Ra, ha), _corners(cb, Rb, hb)
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, k]) for i in range(3) for k in range(3)]
    if not any(float(np.linalg.norm(v)) > 1e-9
               and max((B @ v).min() - (A @ v).max(), (A @ v).min() - (B @ v).max()) > 0 for v in axes):
        return 0.0
    i, j = np.array(_EDGES).T
    return min(float(_point_box(A, cb, Rb, hb).min()), float(_point_box(B, ca, Ra, ha).min()),
               float(_segments(np.repeat(A[i], 12, 0), np.repeat(A[j], 12, 0), np.tile(B[i], (12, 1)),
                               np.tile(B[j], (12, 1))).min()))


def _upright(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (lo + hi) / 2.0, np.eye(3), (hi - lo) / 2.0


def test_the_oracle_measures_two_boxes_exactly():
    """Against dense surface samples: never more, and within their spacing."""
    rng = np.random.default_rng(385)
    for _ in range(60):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        a = (rng.uniform(-0.05, 0.05, 3), q, rng.uniform(0.01, 0.05, 3))
        b = _upright([0.08, -0.03, 0.0], [0.08 + rng.uniform(0.01, 0.05), 0.03, rng.uniform(0.02, 0.1)])
        g = np.stack(np.meshgrid(*[np.linspace(-1, 1, 21)] * 3, indexing="ij"), -1).reshape(-1, 3)
        g = g[(np.abs(g) > 0.999).any(1)]
        pa, pb = a[0] + (g * a[2]) @ a[1].T, b[0] + (g * b[2]) @ b[1].T
        sampled = min(float(np.sqrt(((pa[k:k + 300, None] - pb[None]) ** 2).sum(-1)).min())
                      for k in range(0, len(pa), 300))
        exact = box_distance(a, b)
        assert exact <= sampled + 1e-12 and (exact > 0 or sampled < 0.01), (exact, sampled)
        assert sampled - exact < 0.006, (exact, sampled)


def _camera_world(r, q):
    T = np.asarray(r.runtime.kin.fk(q), float)
    c, R, h = _camera_box_tcp(r.cfg)
    return T[:3, :3] @ c + T[:3, 3], T[:3, :3] @ R, h


def _bottle(r, label, xy, side, top):
    """A `side` x `side` column from the table to `top` at xy, remembered as a belief; its box."""
    table = float(r.cfg.safety.get("table_z", 0.0))
    r.runtime.beliefs.update(label, np.array([xy[0], xy[1], (table + top) / 2]), 0.9,
                             extent=np.array([top - table, side, side]), top_z=top, color="blue")
    return _upright([xy[0] - side / 2, xy[1] - side / 2, table], [xy[0] + side / 2, xy[1] + side / 2, top])


def _assert_the_camera_keeps_the_margin_from(r, box):
    """Every streamed command: the camera box keeps the margin from the box
    (an arm that starts inside it may only move away first)."""
    d = [box_distance(_camera_world(r, q), box) for q in _commands(r)]
    out = next((i for i, v in enumerate(d) if v >= MARGIN), None)
    assert out is not None, f"the camera box never left the margin (best {max(d):.4f} m)"
    assert all(b >= a - 1e-4 for a, b in zip(d[:out], d[1:out + 1])), d[:out + 1]
    assert min(d[out:]) >= MARGIN, f"the camera box came {min(d[out:]):.4f} m from the bottle"


def _refused_untouched(r, out):
    assert out["ok"] is False, out
    assert _commands(r) == [] and r.jaws == [], "point_at sent a command before refusing"
    assert "nothing was moved" in out["error"], out


# ── the reBot: a bottle the jaws clear but the camera box does not ─────────


@needs_pin
@pytest.mark.parametrize("profile", REBOT_PROFILES)
@pytest.mark.parametrize("gap", [0.0125, 0.02])
def test_a_rebot_point_keeps_its_wrist_camera_off_a_known_bottle(rig, profile, gap):
    """The #385 review's Important 1, deterministic: a 4 cm bottle beside the
    cube on the side the top-down pose turns its camera to, `gap` from the
    camera box there and 8 cm or more from the jaws and the joints. main
    pointed with ok: true and the camera box 12.5 / 20 mm from the bottle.
    Now every command keeps the camera box 3 cm from it (another yaw, a lean)
    or nothing moves."""
    clear = rig(profile)
    out = clear.runtime.execute("point_at", {"label": "red cube"})
    assert out["ok"] is True and "lean_deg" not in out, f"test setup: top-down without the bottle: {out}"
    cam = _camera_world(clear, _commands(clear)[-1])
    cube = np.asarray(clear.runtime.beliefs.find("red cube").position, float)
    # straight out from the cube along the camera's horizontal reach (its up axis)
    up = cam[1][:, 2].copy()
    up[2] = 0.0
    up /= np.linalg.norm(up)
    reach = max(float((c - cube)[:2] @ up[:2]) for c in _corners(*cam))
    xy = cube[:2] + (reach + gap + 0.02) * up[:2]
    tcp_z = float(np.asarray(clear.runtime.kin.fk(_commands(clear)[-1]))[2, 3])

    r = rig(profile)
    box = _bottle(r, "bottle", xy, 0.04, tcp_z + 0.06)
    assert box_distance(cam, box) == pytest.approx(gap, abs=1e-3), "test setup: the bottle must sit `gap` off"

    out = r.runtime.execute("point_at", {"label": "red cube"})

    if out["ok"]:
        _assert_the_camera_keeps_the_margin_from(r, box)
    else:
        _refused_untouched(r, out)


@needs_pin
@pytest.mark.parametrize("profile", REBOT_PROFILES)
@pytest.mark.parametrize("deg", range(0, 360, 45))
def test_a_rebot_point_keeps_its_wrist_camera_off_a_bottle_at_any_side(rig, profile, deg):
    """A 16 cm bottle (4 cm girth) 11 cm from the cube, where the camera box
    reaches and the jaws do not: ok -> the camera box keeps 3 cm from it at
    every command; refused -> no command."""
    r = rig(profile)
    cube = np.asarray(r.runtime.beliefs.find("red cube").position, float)
    a = math.radians(deg)
    box = _bottle(r, "bottle", cube[:2] + 0.11 * np.array([math.cos(a), math.sin(a)]), 0.04,
                  float(r.cfg.safety.get("table_z", 0.0)) + 0.16)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    if out["ok"]:
        _assert_the_camera_keeps_the_margin_from(r, box)
    else:
        _refused_untouched(r, out)


@needs_pin
@pytest.mark.parametrize("profile, parts", [("rebot_rs", ["wrist camera"]), ("rebot_rs_reach", ["wrist camera"]),
                                            ("mock", [])])
def test_every_check_gets_the_camera_and_a_profile_without_one_gets_nothing(rig, monkeypatch, profile, parts):
    """Wiring: the top-down rung's neighbour check, the lean-in search and the
    route all get the harness's parts. A check without them is not unsafe
    (the route still keeps the camera off) but it refuses what another yaw or
    lean clears. Every top-down pose is made crowded here, so it leans in."""
    from cascade.skills import pointing

    seen = {}
    real_lean, real_route = pointing.lean_in, pointing.plan_route

    def names(tool):
        return sorted(g.name for g in tool if isinstance(g, pointing.Part)) if isinstance(tool, (tuple, list)) else []

    def crowded(kin, q, tool, neighbours=()):
        seen.setdefault("top-down", names(tool))
        return 0.0, "a crowding stub"

    def lean(kin, column, tool, *a, **k):
        seen["lean-in"] = names(tool)
        return real_lean(kin, column, tool, *a, **k)

    def route(*a, **k):
        seen["route"] = sorted(p.name for p in k.get("body", ()))
        return real_route(*a, **k)

    monkeypatch.setattr(pointing, "neighbour_clearance", crowded)
    monkeypatch.setattr(pointing, "lean_in", lean)
    monkeypatch.setattr(pointing, "plan_route", route)
    r = rig(profile)

    r.runtime.execute("point_at", {"label": "red cube"})

    assert seen.get("top-down") == parts and seen.get("lean-in") == parts, seen
    assert seen.get("route") == parts, seen


@needs_pin
def test_an_estop_latched_in_the_lean_in_search_is_reported_not_the_failed_lean(rig, monkeypatch):
    """#385 review M1, the second place: the e-stop latches while the lean-in
    search runs and the search then finds no pose. The e-stop is reported."""
    from cascade.skills import pointing

    r = rig("mock")
    harness = r.runtime.arm.harness
    real_lean = pointing.lean_in
    fired = []

    def lean(*a, **k):
        found = real_lean(*a, **k)
        fired.append(True)
        harness.estop("operator e-stop")
        return pointing.LeanIn(q=None, clearance=found.clearance, solved=found.solved)

    monkeypatch.setattr(pointing, "lean_in", lean)
    monkeypatch.setattr(pointing, "neighbour_clearance", lambda kin, q, tool, neighbours=(): (0.0, "a crowding stub"))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert fired, "the lean-in search never ran"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "e-stop latched" in out["error"], out
    assert "leaning in" not in out["error"], out


def _rebot_tool_volume():
    from cascade.config import load_demo_config
    from cascade.safety.tool_volume import build_tool_volume

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    return cfg, build_tool_volume(cfg, cfg.arm)


def test_the_body_is_the_camera_box_on_its_faces_and_not_the_jaws_twice():
    """The reBot's tool volume holds the camera slab and the open-jaw sweep
    box. point_at's jaws already ARE that sweep box (the same `jaw_box`), so
    the body adds the camera only: every corner of the box, every sample on
    its faces, at most BODY_SPACING_M apart along each edge."""
    from cascade.skills import pointing

    cfg, tv = _rebot_tool_volume()
    assert tv is not None and tv.names == ("wrist camera", "jaws"), "test setup: the reBot tool volume"
    body = pointing.tool_body(tv, MARGIN)
    assert [p.name for p in body] == ["wrist camera"]
    c, R, h = _camera_box_tcp(cfg)
    local = (body[0].samples - c) @ R
    assert np.abs(local).max(axis=0) == pytest.approx(h, abs=1e-12)
    on_face = np.isclose(np.abs(local), h, rtol=0.0, atol=1e-9).any(axis=1)
    assert on_face.all(), "a sample inside the box (the faces are what an object reaches first)"
    for corner in _corners(np.zeros(3), np.eye(3), h):
        assert np.min(np.linalg.norm(local - corner, axis=1)) < 1e-12
    for k in range(3):
        steps = np.diff(np.unique(np.round(local[:, k], 12)))
        assert steps.max() <= pointing.BODY_SPACING_M + 1e-12
    s = pointing.BODY_SPACING_M
    assert body[0].slack_m == pytest.approx(s * s / (4 * MARGIN))


def test_no_tool_volume_no_body_and_a_hand_built_part_needs_its_box():
    """A part known only by its points refuses (#399 review Minor 4,
    tests/test_point_at_planning_followups.py); the jaws need no box."""
    from cascade.safety.tool_volume import ToolVolume
    from cascade.skills import pointing

    assert pointing.tool_body(None, MARGIN) == ()
    pts = np.array([[0.0, 0.0, 0.05], [0.0, 0.01, 0.05], [0.1, 0.0, 0.0]])
    tv = ToolVolume(local=pts, part=np.array([0, 0, 1]), names=("probe", "jaws"))
    with pytest.raises(ValueError, match="tool part 'probe' has no box"):
        pointing.tool_body(tv, MARGIN)
    box = (np.array([0.0, 0.005, 0.05]), np.eye(3), np.array([0.0, 0.005, 0.0]))
    jaw = (np.array([0.1, 0.0, 0.0]), np.eye(3), np.zeros(3))
    body = pointing.tool_body(ToolVolume(local=pts, part=np.array([0, 0, 1]), names=("probe", "jaws"),
                                         boxes=(box, jaw)), MARGIN)
    assert [p.name for p in body] == ["probe"] and body[0].slack_m > 0.0


def test_a_parts_sampled_clearance_minus_its_slack_bounds_the_box():
    """`Part.slack_m` makes a sampled clearance a bound on the box: whenever
    the faces' samples keep `margin + slack` from a column, the whole box
    keeps `margin` from it (exact oracle), for any pose of the box."""
    from cascade.skills import pointing

    rng = np.random.default_rng(1385)
    cfg, tv = _rebot_tool_volume()
    part = pointing.tool_body(tv, MARGIN)[0]
    c, R, h = _camera_box_tcp(cfg)
    accepted = 0
    for _ in range(400):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        t = rng.uniform(-0.15, 0.15, 3) + [0.0, 0.0, 0.2]
        side = rng.uniform(0.01, 0.05, 2)
        lo, hi = np.array([-side[0], -side[1], 0.0]), np.array([side[0], side[1], rng.uniform(0.05, 0.3)])
        col = pointing.ObjectColumn(footprints=(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]],
                                                         [lo[0], hi[1]]]),), tops=(hi[2],), z_lo=0.0,
                                    aim=np.zeros(3))
        sampled = col.min_distance(part.samples @ q.T + t)
        if sampled - part.slack_m >= MARGIN:
            accepted += 1
            assert box_distance((q @ c + t, q @ R, h), _upright(lo, hi)) >= MARGIN - 1e-12
    assert accepted > 50, "test setup: too few accepted poses to say anything"


def test_the_slack_is_what_the_bound_needs_and_no_less():
    """At the margin, a box face square to a column edge between two samples
    is `spacing / sqrt(2)` sideways off the nearest sample: its sampled
    clearance overstates the box's by up to s^2 / (4 margin)."""
    from cascade.skills import pointing

    s = pointing.BODY_SPACING_M
    # a face in the plane x = 0 sampled on a grid s apart, an edge s/2 off two grid lines
    grid = np.array([[0.0, y, z] for y in np.arange(-0.05, 0.0501, s) for z in np.arange(-0.05, 0.0501, s)])
    edge_y, edge_z = s / 2, s / 2
    d_true = MARGIN
    sampled = float(np.min(np.hypot(np.hypot(grid[:, 1] - edge_y, grid[:, 2] - edge_z), d_true)))
    part = pointing.Part(grid, "probe", pointing.part_slack(s, MARGIN))
    assert sampled - part.slack_m <= d_true + 1e-12
    assert sampled - d_true == pytest.approx(math.hypot(d_true, s / math.sqrt(2)) - d_true)
    assert part.slack_m >= sampled - d_true


class _SlideKin:
    """FK = a pure translation by q (3 joints); no joint origins to check."""

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q, dtype=float)[:3]
        return T

    def link_positions(self, q):
        return np.zeros((1, 3))

    def lever_arms(self, q, tool):
        return np.ones(3), np.zeros((0, 3))


def _square(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _target():
    from cascade.skills import pointing

    return pointing.ObjectColumn(footprints=(_square(-0.02, 0.02, -0.02, 0.02),), tops=(0.05,), z_lo=0.0,
                                 aim=np.array([0.0, 0.0, 0.05]))


def _camera_part(offset_y=0.08):
    """A 4 cm cube of samples 8 cm to the side of the TCP (a camera-like part)."""
    from cascade.skills import pointing

    g = np.stack(np.meshgrid(*[np.linspace(-0.02, 0.02, 9)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return pointing.Part(g + [0.0, offset_y, 0.0], "wrist camera", 0.0)


def test_a_route_keeps_the_margin_with_every_part_and_names_it():
    """A straight leg 10 cm over the table: the jaws (the TCP) stay 8 cm from
    a tall bottle, the camera part passes 1 cm from it. Refused, naming the
    camera and the bottle; the jaws alone would let it pass."""
    from cascade.skills import pointing

    bottle = pointing.neighbour_column(_square(-0.02, 0.02, 0.11, 0.15), 0.3, 0.0, "the blue bottle")
    start, goal = np.array([-0.2, 0.0, 0.1]), np.array([0.2, 0.0, 0.1])
    jaws = (np.zeros((1, 3)),)
    assert pointing.segment_violation(_SlideKin(), start, goal, (_target(), bottle), jaws, MARGIN, 0.0) is None
    why = pointing.segment_violation(_SlideKin(), start, goal, (_target(), bottle), jaws + (_camera_part(),),
                                     MARGIN, 0.0)
    assert why is not None and re.match(r"the wrist camera 0\.0[0-2]\d m from the blue bottle \d+% of the way$",
                                        why), why


def test_a_route_counts_a_parts_slack():
    """The camera's samples pass 3.2 cm from the bottle: with no slack that
    keeps the margin; with 5 mm of slack (what its sampling may overstate)
    it is 2.7 cm and the leg is refused."""
    from cascade.skills import pointing

    bottle = pointing.neighbour_column(_square(-0.02, 0.02, 0.132, 0.172), 0.3, 0.0, "the blue bottle")
    start, goal = np.array([-0.2, 0.0, 0.1]), np.array([0.2, 0.0, 0.1])
    jaws = (np.zeros((1, 3)),)
    exact = _camera_part()
    assert pointing.segment_violation(_SlideKin(), start, goal, (_target(), bottle), jaws + (exact,),
                                      MARGIN, 0.0) is None
    slack = pointing.Part(exact.samples, exact.name, 0.005)
    why = pointing.segment_violation(_SlideKin(), start, goal, (_target(), bottle), jaws + (slack,), MARGIN, 0.0)
    assert why is not None and re.match(r"the wrist camera 0\.02[7-9] m from the blue bottle \d+% of the way$", why), why


def test_each_part_is_its_own_pair_against_the_table():
    """The table pair of a part is its own: a camera part 1 cm over the
    table refuses as 'the wrist camera ... above the table' while the jaws
    are 12 cm up."""
    from cascade.skills import pointing

    low = _camera_part()
    low = pointing.Part(low.samples - [0.0, 0.0, 0.09], low.name, 0.0)    # 1-5 cm over the table
    start, goal = np.array([0.3, 0.0, 0.12]), np.array([0.3, 0.1, 0.12])
    why = pointing.segment_violation(_SlideKin(), start, goal, _target(), (np.zeros((1, 3)), low), MARGIN, 0.0)
    assert why == "the wrist camera 0.010 m above the table where it starts", why
    assert pointing.segment_violation(_SlideKin(), start, goal, _target(), (np.zeros((1, 3)),), MARGIN, 0.0) is None
    values, kinds, owners, which = pointing._pair_clearances(_SlideKin(), start, (_target(),),
                                                             (np.zeros((1, 3)), low), 0.0)
    assert list(kinds) == [0, 0, 1, 1] and list(which) == [-1, -2, -1, -2]


def test_a_pose_counts_every_part_and_names_the_nearest():
    """`pose_clearance` and `neighbour_clearance` measure the parts too (each
    less its slack) and say which part failed."""
    from cascade.skills import pointing

    bottle = pointing.neighbour_column(_square(-0.02, 0.02, 0.11, 0.15), 0.3, 0.0, "the blue bottle")
    q = np.array([0.0, 0.0, 0.1])
    jaws = (np.zeros((1, 3)),)
    assert pointing.neighbour_clearance(_SlideKin(), q, jaws, (bottle,))[0] == pytest.approx(0.11)
    cam = _camera_part()
    slack = pointing.Part(cam.samples, cam.name, 0.002)
    assert pointing.neighbour_clearance(_SlideKin(), q, jaws + (cam,), (bottle,)) == (pytest.approx(0.01),
                                                                                       "the blue bottle")
    assert pointing.neighbour_clearance(_SlideKin(), q, jaws + (slack,), (bottle,))[0] == pytest.approx(0.008)
    # the target 1 cm from the camera part's far face: the part is named
    near_target = pointing.Part(cam.samples - [0.0, 0.04, 0.0], cam.name, 0.0)
    clr = pointing.pose_clearance(_SlideKin(), np.array([0.0, -0.09, 0.07]), _target(), jaws + (near_target,), 2,
                                  0.0)
    assert clr.tool_m == pytest.approx(0.01) and clr.table_m == pytest.approx(0.05), clr
    assert clr.tool_part == "wrist camera" and clr.worst(MARGIN) == "the wrist camera 0.010 m from it", clr
    assert pointing.pose_clearance(_SlideKin(), q, _target(), jaws, 2, 0.0).tool_part == "jaws"


def test_a_withdrawal_and_a_rise_carry_the_body():
    """`plan_route(body=...)`: the parts ride every leg (here a direct leg
    whose camera part sweeps through a bottle the jaws clear)."""
    from cascade.skills import pointing

    bottle = pointing.neighbour_column(_square(-0.02, 0.02, 0.11, 0.15), 0.3, 0.0, "the blue bottle")

    class _Kin(_SlideKin):
        def ik(self, T, q_init=None, *a, **k):
            return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=True, error=0.0)

        def ik_approach(self, p, *a, **k):
            return SimpleNamespace(q=np.asarray(p, float).copy(), success=True, error=0.0)

    start, goal = np.array([-0.2, 0.0, 0.1]), np.array([0.2, 0.0, 0.1])
    route = pointing.plan_route(_Kin(), start, goal, _target(), np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0,
                                neighbours=(bottle,))
    assert route.targets == [goal]
    with pytest.raises(pointing.RouteRefused, match="the wrist camera"):
        pointing.plan_route(_Kin(), start, goal, _target(), np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0,
                            neighbours=(bottle,), body=(_camera_part(),))


def test_a_withdrawal_carries_the_body():
    """The jaws start 2 mm over the cube's top on a 45 deg approach and back
    straight out (up and to -x). The camera part starts 3.2 cm from a bottle
    on the -x side and the first 5 mm step takes it to 2.85 cm: refused,
    naming the camera. Without the part the back-out would pass and the
    route would start with the camera inside the margin."""
    from cascade.skills import pointing

    s = 1.0 / math.sqrt(2.0)
    R = np.array([[-s, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, -s]])      # approach (column 2) = down and +x

    class _TiltKin(_SlideKin):
        def fk(self, q):
            T = super().fk(q)
            T[:3, :3] = R
            return T

        def ik(self, T, q_init=None, *a, **k):
            return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=True, error=0.0)

    bottle = pointing.neighbour_column(_square(-0.20, -0.16, -0.02, 0.02), 0.3, 0.0, "the blue bottle")
    start = np.array([0.0, 0.0, 0.052])
    world = np.stack(np.meshgrid(*[np.linspace(-0.02, 0.02, 9)] * 3, indexing="ij"), -1).reshape(-1, 3)
    camera = pointing.Part((world + [-0.108, 0.0, 0.06]) @ R, "wrist camera", 0.0)     # TCP frame
    assert pointing.neighbour_clearance(_TiltKin(), start, (np.zeros((1, 3)), camera), (bottle,))[0] == \
        pytest.approx(0.032)
    goal = np.array([0.0, 0.0, 0.15])

    def plan(body):
        return pointing.plan_route(_TiltKin(), start, goal, _target(), np.zeros((1, 3)), 2, margin=MARGIN,
                                   table_z=0.0, neighbours=(bottle,), close=False, body=body)

    assert plan(()).withdraw, "test setup: the jaws must start touching and back out"
    with pytest.raises(pointing.RouteRefused, match=r"the wrist camera 0\.02\d m from the blue bottle"):
        plan((camera,))


def test_a_rise_first_waypoint_carries_the_body_after_the_jaws_close():
    """The jaws start 2 cm over the cube (they may not close there), so a
    route closes them at a rise-first waypoint and goes on with them closed.
    The camera part passes 1 cm from a tall bottle on the straight leg and on
    every waypoint's second leg: refused. The jaws alone go straight."""
    from cascade.skills import pointing

    class _Kin(_SlideKin):
        def ik(self, T, q_init=None, *a, **k):
            return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=True, error=0.0)

        def ik_approach(self, p, *a, **k):
            return SimpleNamespace(q=np.asarray(p, float).copy(), success=False, error=1.0)

    bottle = pointing.neighbour_column(_square(0.13, 0.17, 0.11, 0.15), 0.5, 0.0, "the blue bottle")
    start, goal = np.array([0.0, 0.0, 0.07]), np.array([0.3, 0.0, 0.07])

    def plan(body):
        return pointing.plan_route(_Kin(), start, goal, _target(), np.zeros((1, 3)), 2, margin=MARGIN,
                                   table_z=0.0, neighbours=(bottle,), body=body)

    route = plan(())
    assert route.targets == [goal] and route.close_at == 1, "test setup: jaws alone go straight, closing there"
    with pytest.raises(pointing.RouteRefused, match=r"going straight, the wrist camera 0\.0[0-2]\d m from the blue "
                                                    r"bottle \d+% of the way, and no rise-first waypoint clears it"):
        plan((_camera_part(),))


def test_a_waypoint_leg_with_the_jaws_still_open_carries_the_body():
    """The jaws may close neither where they start nor at any waypoint (their
    close sweeps 5 cm sideways into a tall bottle) but at the pointing pose:
    the waypoint's second leg runs with the jaws as they are, and the camera
    part with them. A low wall between: the jaws alone clear it from the
    12 cm waypoint, the camera only from the 15 cm one."""
    from cascade.skills import pointing

    class _Kin(_SlideKin):
        def ik(self, T, q_init=None, *a, **k):
            return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=True, error=0.0)

        def ik_approach(self, p, *a, **k):
            return SimpleNamespace(q=np.asarray(p, float).copy(), success=False, error=1.0)

    bottle = pointing.neighbour_column(_square(0.06, 0.10, -0.02, 0.02), 0.5, 0.0, "the blue bottle")
    wall = pointing.neighbour_column(_square(-0.17, -0.13, -0.10, 0.10), 0.08, 0.0, "the low wall")
    tcp = (np.zeros((1, 3)),)
    jaws = pointing.Jaws(closed=tcp, now=tcp, sweep=(np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]),))
    start, goal = np.array([0.0, 0.0, 0.07]), np.array([-0.3, 0.0, 0.12])

    def plan(body):
        return pointing.plan_route(_Kin(), start, goal, _target(), jaws, 2, margin=MARGIN, table_z=0.0,
                                   neighbours=(bottle, wall), body=body)

    alone = plan(())
    assert alone.close_at == 2 and alone.targets[0][2] == pytest.approx(0.12), "test setup: the 12 cm waypoint"
    route = plan((_camera_part(),))
    assert route.close_at == 2 and route.targets[0][2] == pytest.approx(0.15), route.targets


# ── back-out while holding (#385 review M5) ────────────────────────────────


class _DownKin(_SlideKin):
    def fk(self, q):
        T = super().fk(q)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return T

    def ik(self, T, q_init=None, *a, **k):
        return SimpleNamespace(q=np.asarray(T, float)[:3, 3].copy(), success=True, error=0.0)


def test_a_touching_start_while_holding_never_backs_out():
    """The payload in the jaws is not modelled: a jaw touching an object
    while holding refuses (nothing commanded) instead of backing out."""
    from cascade.skills import pointing

    start, goal = np.array([0.0, 0.0, 0.053]), np.array([0.0, 0.0, 0.10])
    with pytest.raises(pointing.RouteRefused,
                       match=r"the jaws start 0\.003 m from it while holding 'the cup'; they back out only empty"):
        pointing.plan_route(_DownKin(), start, goal, _target(), np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0,
                            close=False, holding="the cup")
    route = pointing.plan_route(_DownKin(), start, goal, _target(), np.zeros((1, 3)), 2, margin=MARGIN, table_z=0.0,
                                close=False)
    assert route.withdraw_m > 0, "test setup: empty jaws back out"


def _park(r, tcp):
    from cascade.grasping.obb_grasp import _yaw_rotation
    from cascade.types import make_transform

    runtime = r.runtime
    runtime.arm.get_state()
    arm = r.arms[0]
    T = make_transform(_yaw_rotation(0.0, axis_order=runtime._tool_axis_order), np.asarray(tcp, float))
    ik = runtime.kin.ik(T, arm.get_state().q)
    assert ik.success, "test setup: the parking pose must solve"
    arm._q = np.asarray(ik.q, float).copy()
    arm.commands.clear()
    return ik.q


@needs_pin
def test_point_at_while_holding_refuses_to_back_out_of_a_touch(rig):
    """The mock arm parked top-down 2 mm over the cube's top, holding a cup:
    main backed out with the cup in the jaws, unmodelled. Now nothing moves
    and the refusal says why."""
    from cascade.safety.tool_volume import jaw_box

    r = rig("mock")
    cube = r.runtime.beliefs.find("red cube")
    r.runtime.beliefs.update("cup", np.array([0.5, 0.4, 0.04]), 0.95, extent=np.array([0.08, 0.06, 0.06]),
                             top_z=0.08, color="yellow")
    r.runtime.held_object = "yellow cup"
    q = _park(r, [float(cube.position[0]), float(cube.position[1]), 0.12])
    T = np.asarray(r.runtime.kin.fk(q), float)
    c, R, h = jaw_box(r.cfg.grasp.graspgenx.sweep, str(r.cfg.arm.get("tool_axis_order", "down_open")))
    low = float(_corners(T[:3, :3] @ c + T[:3, 3], T[:3, :3] @ R, h)[:, 2].min())
    _park(r, [float(cube.position[0]), float(cube.position[1]), 0.12 - (low - float(cube.top_z)) + 0.002])

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert "while holding 'yellow cup'; they back out only empty" in out["error"], out


# ── an e-stop while planning wins over a geometric refusal (#385 review M1) ──


@needs_pin
@pytest.mark.parametrize("held", [False, True])
def test_an_estop_latched_while_planning_is_reported_before_the_geometry(rig, monkeypatch, held):
    """The e-stop latches in the top-down search, and then the search fails
    for its own geometric reason (no top-down pose; holding: no lean-in;
    empty: no lean-in pose either). main named the geometry; the operator's
    e-stop is what the agent must hear. Nothing is commanded either way."""
    r = rig("mock")
    if held:
        r.runtime.beliefs.update("cup", np.array([0.5, 0.4, 0.04]), 0.95, extent=np.array([0.08, 0.06, 0.06]),
                                 top_z=0.08, color="yellow")
        r.runtime.held_object = "yellow cup"
    harness = r.runtime.arm.harness
    fired = []

    def ik(T, q_init=None, *a, **k):
        if not fired:
            fired.append(True)
            harness.estop("operator e-stop")
        return SimpleNamespace(q=np.asarray(q_init, float), success=False, error=1.0)

    monkeypatch.setattr(r.runtime.kin, "ik", ik)
    monkeypatch.setattr(r.runtime.kin, "ik_approach",
                        lambda p, a, q_init, *x, **k: SimpleNamespace(q=np.asarray(q_init, float), success=False,
                                                                      error=1.0))

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert fired, "the top-down search never ran"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "e-stop latched" in out["error"], out
    assert "leaning in" not in out["error"] and "while holding" not in out["error"], out
