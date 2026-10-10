"""B73: gripper-housing / finger clearance vet for the analytic planner's
angled and side candidates (B45 follow-up, opt-in `*_reach` profiles).

What is pinned here, and why:

* PREMISES (pass on main by design): a side approach the runtime's IK and
  harness vet ADMIT under `rebot_rs_reach` drives the wrist motor's
  collision mesh (`link5`) below the table on its approach -- every harness
  link proxy is a joint origin inside the grasp exemption cylinder, so the
  harness cannot see it; and a 30/45 degree lean never puts any of that
  geometry below the fingertips (the housing's front edge sits 73 mm behind
  the tips and 41 mm off the approach axis: atan(73/41) ~ 61 degrees).
* GEOMETRY: the clearance hulls are bound to the URDF + STL hashes, contain
  every mesh vertex of their link, and keep the mesh's exact lowest point.
* THE VET: support plane and observed box, the margin (width_pad_m / 2, the
  clearance the planner already gives each open jaw), the dense approach
  sampling, fail-closed geometry.
* BEHAVIOUR through the real runtime (mock stack): a penetrating side
  candidate is refused and the grasp falls back to the next candidate; the
  decision and reason are in the grasp evidence; `false` and the default
  profiles are byte-identical (no geometry load, no evidence event).
* EVIDENCE: the B45-grid measurement is bound to its inputs, re-evaluates,
  and the numbers in the docs are the committed ones.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pytest
from conftest import REPO, URDF, needs_pin

from cascade.config import Cfg, load_demo_config
from cascade.types import Detection, Grasp, ObjectFix

ASSET = REPO / "assets" / "grasp_geometry" / "rebot_rs_clearance_hulls.json"
MESHES = URDF.parent.parent / "meshes"
EVIDENCE = REPO / "docs" / "evidence" / "angled-clearance-20261010" / "rebot_angled_clearance.json"
B45_EVIDENCE = REPO / "docs" / "evidence" / "reach-envelope-20261009" / "rebot_reachability.json"
REACH_PROFILES = ("rebot_rs_reach", "isaac_reach", "mock_reach")
VETTED_LINKS = ("link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")
#: a side point the B45 envelope pass admits at 5 cm grasp height
SIDE_POINT = (0.40, 0.40, 0.05)


@pytest.fixture(autouse=True)
def _private_ports(monkeypatch):
    """Ports inside this item's block (47500-47599); nothing listens there."""
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "47521")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "47522")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "47523")
    monkeypatch.setenv("CASCADE_HUG_PORT", "47524")
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")


# ── inline helpers (no new-module imports: the premises must run on main) ────


def _stl_vertices(name: str) -> np.ndarray:
    raw = (MESHES / f"{name}.STL").read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    dtype = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    faces = np.frombuffer(raw, dtype, count=count, offset=84)["v"].astype(float)
    return np.unique(faces.reshape(-1, 3), axis=0)


def _rig(profile: str):
    from cascade.control.kinematics import Kinematics
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    cfg = load_demo_config(arm=profile)
    a, view = Cfg(cfg.arms[0]), Cfg(cfg.arms[0]["resolved"])
    kin = Kinematics(a.model, a.get("ee_frame", "gripper_end"), n_controlled=int(a.n_joints),
                     joint_signs=a.get("joint_signs"))
    harness = SafetyHarness(SafetyLimits.from_config(view.safety), kinematics=kin)
    return view, a, kin, harness


def _harness_vet(view, home, harness):
    """The harness half of the runtime's `_vet`, in its order (B45's replica)."""
    from cascade.safety.trajectory import vet_segment

    def vet(g, q_pre, q_grasp):
        reason = harness.vet_pose(q_pre)
        if reason:
            return f"pregrasp unsafe: {reason}"
        reason = vet_segment(harness, home, q_pre, float(view.grasp.get("move_duration_s", 2.5)))
        if reason:
            return f"approach unsafe: {reason}"
        for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
            q = q_pre + s * (np.asarray(q_grasp) - np.asarray(q_pre))
            reason = harness.vet_pose(q, exempt_xy=g.position[:2],
                                      exempt_radius_m=float(view.grasp.get("exempt_radius_m", 0.07)),
                                      exempt_z_min=float(harness.limits.table_z) - 0.06)
            if reason:
                return f"descent unsafe: {reason}"
        return None

    return vet


def _tilted(x, y, z, tilt_deg, cls=Grasp, quality=1.0) -> Grasp:
    """Approach leaning `tilt_deg` away from the base, jaw horizontal and
    tangential: the planner's `_angled_alternates` pose (B45 `out*`/`side`)."""
    from cascade.grasping.obb_grasp import tool_rotation

    r = np.array([x, y, 0.0]) / np.hypot(x, y)
    t = np.cross([0.0, 0.0, 1.0], r)
    th = np.radians(tilt_deg)
    approach = np.cos(th) * np.array([0.0, 0.0, -1.0]) + np.sin(th) * r
    return cls(position=np.array([x, y, z], dtype=float), rotation=tool_rotation(approach, t),
               width_m=0.065, approach=approach, quality=quality, label=f"tilt{tilt_deg}")


def _solve(profile, grasp):
    """(q_pre, q_grasp) through the real selector + the harness replica."""
    from cascade.grasping.selector import select_grasp

    view, a, kin, harness = _rig(profile)
    home = np.asarray(a.home_q, dtype=float)
    g, q_pre, q_grasp = select_grasp([grasp], kin, home, max_width_m=0.09,
                                     pregrasp_offset_m=float(view.grasp.pregrasp_offset_m),
                                     validate=_harness_vet(view, home, harness), preserve_order=True)
    return kin, g, q_pre, q_grasp


def _link_min_z(kin, q, link: str, vertices: np.ndarray, finger_q: float = 0.045) -> float:
    import pinocchio as pin

    model, data = kin.model, kin.data
    qf = np.zeros(model.nq)
    qf[:kin.n] = q
    for joint in ("joint_left", "joint_right"):
        qf[model.idx_qs[model.getJointId(joint)]] = finger_q
    pin.forwardKinematics(model, data, qf)
    pin.updateFramePlacements(model, data)
    M = data.oMf[model.getFrameId(link)]
    return float((vertices @ np.asarray(M.rotation).T + np.asarray(M.translation))[:, 2].min())


def _approach(q_pre, q_grasp, n=41):
    return [q_pre + s * (q_grasp - q_pre) for s in np.linspace(0.0, 1.0, n)]


# ── premises (pass on main by design) ─────────────────────────────────────────


@needs_pin
def test_premise_the_harness_admits_a_side_grasp_whose_wrist_mesh_dips_below_the_table():
    """The B45 study (and the runtime) admit a 5 cm side grasp at
    (0.40, 0.40): IK solves and every harness check passes. Yet the wrist
    motor's collision mesh (`link5`) goes below the table on the approach:
    the harness's link proxies are joint origins inside the grasp exemption
    cylinder (floor table_z - 0.06), so nothing refuses it today."""
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(*SIDE_POINT, 90.0))
    np.testing.assert_allclose(kin.fk(q_grasp)[:3, 3], SIDE_POINT, atol=1e-3)
    link5 = _stl_vertices("link5")
    lowest = min(_link_min_z(kin, q, "link5", link5) for q in _approach(q_pre, q_grasp))
    assert lowest < 0.0, lowest  # below the table plane (z = 0)
    # the harness proxies (joint origins) all stay above the table there
    assert min(kin.link_positions(q)[1:, 2].min() for q in _approach(q_pre, q_grasp)) > 0.0


@needs_pin
@pytest.mark.parametrize("tilt", [30.0, 45.0])
def test_premise_a_lean_up_to_45_degrees_keeps_every_mesh_above_the_fingertips(tilt):
    """Leaning out by 30/45 degrees at 2 cm grasp height, the lowest point of
    the forearm, wrist, housing and fingers over the whole approach is the
    fingertip plane (the TCP): the support plane alone can never refuse
    these tilts at any reachable grasp height."""
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(0.35, 0.10, 0.02, tilt))
    meshes = {name: _stl_vertices(name) for name in VETTED_LINKS}
    lowest = min(_link_min_z(kin, q, name, v) for q in _approach(q_pre, q_grasp)
                 for name, v in meshes.items())
    assert lowest == pytest.approx(0.02, abs=0.0005)


def test_premise_the_housing_drops_below_the_fingertips_only_beyond_61_degrees():
    """Housing (`gripper_end`) in its own frame: +x is the approach and the
    fingertips are at x = 0. Its front face is 73 mm behind them and it
    reaches 41 mm off-axis, so a tilt t lowers its edge below the tips only
    when 0.073 cos t < 0.041 sin t, i.e. t > ~61 degrees."""
    v = _stl_vertices("gripper_end")
    front = -float(v[:, 0].max())
    off_axis = float(np.abs(v[:, 2]).max())
    assert front == pytest.approx(0.0732, abs=0.0005)
    assert off_axis == pytest.approx(0.041, abs=0.0005)
    assert 60.0 < np.degrees(np.arctan2(front, off_axis)) < 61.5


# ── the clearance hulls (asset) ───────────────────────────────────────────────


def _asset() -> dict:
    return json.loads(ASSET.read_text())


def test_clearance_hulls_are_bound_to_the_urdf_and_its_collision_meshes():
    data = _asset()
    assert data["schema"] == "cascade.gripper_clearance_hulls/1"
    assert tuple(data["links"]) == VETTED_LINKS
    src = data["sources"]
    assert src[str(URDF.relative_to(REPO))] == hashlib.sha256(URDF.read_bytes()).hexdigest()
    for name in VETTED_LINKS:
        rel = str((MESHES / f"{name}.STL").relative_to(REPO))
        assert src[rel] == hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert [j["name"] for j in data["finger_joints"]] == ["joint_left", "joint_right"]
    assert all((j["lower_m"], j["upper_m"]) == (0.0, 0.05) for j in data["finger_joints"])


@pytest.mark.parametrize("link", VETTED_LINKS)
def test_hulls_keep_the_exact_lowest_mesh_point_in_every_direction(link):
    """Support function: for any direction the hull vertices reach exactly as
    far as the mesh (0.1 um rounding). Hence the support-plane test on hull
    vertices is exact for the mesh at every pose."""
    mesh = _stl_vertices(link)
    hull = np.concatenate([np.asarray(c["vertices"]) for c in _asset()["links"][link]])
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(64, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for d in [*np.eye(3), *-np.eye(3), *dirs]:
        assert float((hull @ d).min()) == pytest.approx(float((mesh @ d).min()), abs=2e-7)


def test_hulls_rederive_from_the_meshes():
    """The committed asset is what the build script computes (SciPy only
    offline; skipped where it is not installed)."""
    pytest.importorskip("scipy")
    sys.path.insert(0, str(REPO / "scripts"))
    import build_gripper_clearance_hulls as build

    assert json.loads(json.dumps(build.build())) == _asset()


# ── the vet ───────────────────────────────────────────────────────────────────


def _vet(profile="rebot_rs_reach"):
    from cascade.grasping import gripper_clearance as gc

    view, a, kin, harness = _rig(profile)
    return gc, gc.ApproachClearance(gc.load_geometry(a.model), kin)


def _upright_box(x, y, size, jaw):
    from cascade.grasping.gripper_clearance import Box

    u = np.array([jaw[0], jaw[1], 0.0])
    u /= np.linalg.norm(u)
    v = np.cross([0.0, 0.0, 1.0], u)
    return Box(centre=np.array([x, y, size[2] / 2]), axes=np.column_stack([u, v, [0.0, 0.0, 1.0]]),
               half=np.asarray(size, dtype=float) / 2)


@needs_pin
def test_margin_is_half_the_planners_jaw_pad():
    from cascade.grasping import gripper_clearance as gc

    for profile in REACH_PROFILES:
        view = Cfg(load_demo_config(arm=profile).arms[0]["resolved"])
        assert view.grasp.get("width_pad_m", 0.015) == 0.015  # the planner's default pad
        assert gc.margin_m(view.grasp) == pytest.approx(0.0075)
    assert gc.margin_m({"width_pad_m": 0.006}) == pytest.approx(0.003)
    assert gc.margin_m({}) == pytest.approx(0.0075)  # the planner's own default pad


def test_jaw_gap_is_the_commanded_pregrasp_opening():
    from cascade.grasping.gripper_clearance import jaw_gap_m

    assert jaw_gap_m(0.065, 0.09, None) == 0.09          # full open (default)
    assert jaw_gap_m(0.065, 0.09, 0.01) == pytest.approx(0.075)  # adaptive opening
    assert jaw_gap_m(0.085, 0.09, 0.01) == 0.09          # never wider than the jaw


@needs_pin
def test_finger_hulls_open_to_the_commanded_gap():
    """Fingers placed by the model's prismatic joints at q = gap / 2: the
    inner faces of the parts in front of the palm are gap (+0.1 mm) apart,
    clipped at the joints' travel (0.05 m each: 100 mm)."""
    gc, vet = _vet()
    q = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    T = vet.kin.fk(q)
    for gap, expected in ((0.03, 0.03), (0.06, 0.06), (0.09, 0.09), (0.12, 0.10)):
        placed = vet.placed(q, gap)
        inner = []
        for finger in ("gripper_left", "gripper_right"):
            local = (placed[finger] - T[:3, 3]) @ T[:3, :3]   # TCP frame: x approach, y opening
            ahead = local[local[:, 0] > -0.07]
            inner.append(np.abs(ahead[:, 1]).min())
        assert inner[0] + inner[1] == pytest.approx(expected + 0.0001, abs=0.0002)


@needs_pin
def test_frame_poses_place_named_links_and_refuse_what_the_model_lacks():
    view, a, kin, harness = _rig("rebot_rs_reach")
    q = np.asarray(a.home_q, dtype=float)
    (tcp,) = kin.frame_poses(q, ["gripper_end"])
    np.testing.assert_allclose(tcp, kin.fk(q), atol=1e-12)
    closed, opened = (kin.frame_poses(q, ["gripper_left"], {"joint_left": v})[0] for v in (0.0, 0.03))
    assert np.linalg.norm(opened[:3, 3] - closed[:3, 3]) == pytest.approx(0.03, abs=1e-9)
    with pytest.raises(ValueError, match="controlled joint"):
        kin.frame_poses(q, ["gripper_end"], {"joint1": 0.3})
    with pytest.raises(ValueError, match="not in model"):
        kin.frame_poses(q, ["gripper_end"], {"no_such_joint": 0.0})
    with pytest.raises(ValueError, match="not in model"):
        kin.frame_poses(q, ["no_such_link"])


@needs_pin
def test_hulls_bind_only_to_a_model_that_has_their_links_and_fingers():
    from cascade.grasping import gripper_clearance as gc
    from conftest import SO101_URDF

    from cascade.control.kinematics import Kinematics

    geom = gc.load_geometry(URDF)
    so101 = Kinematics(str(SO101_URDF), "gripper_frame_link", n_controlled=5)
    with pytest.raises(gc.ClearanceGeometryError, match="link 'link3' not in the arm model"):
        gc.ApproachClearance(geom, so101)
    view, a, kin, harness = _rig("rebot_rs_reach")
    data = _asset()
    data["finger_joints"] = [{"name": "no_such_finger", "lower_m": 0.0, "upper_m": 0.05}]
    with pytest.raises(gc.ClearanceGeometryError, match="finger joint 'no_such_finger'"):
        gc.ApproachClearance(gc.ClearanceGeometry(data, "x"), kin)


@needs_pin
def test_an_approach_needing_too_many_poses_fails_closed(monkeypatch):
    gc, vet = _vet()
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(0.40, 0.40, 0.07, 90.0))
    monkeypatch.setattr(gc, "_MAX_SAMPLES", 16)
    with pytest.raises(gc.ClearanceGeometryError, match="more than 16 poses"):
        vet.check(q_pre, q_grasp, support_z=0.0, box=None, margin_m=0.0075, jaw_gap_m=0.09)


@needs_pin
def test_side_grasp_at_5cm_is_refused_on_the_support_plane_by_the_wrist():
    gc, vet = _vet()
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(*SIDE_POINT, 90.0))
    res = vet.check(q_pre, q_grasp, support_z=0.0, box=None, margin_m=0.0075, jaw_gap_m=0.09)
    assert res.ok is False
    assert res.plane_part == "link5" and res.plane_clearance_m < 0.0
    assert "link5" in res.reason and "support plane" in res.reason


@needs_pin
def test_side_grasp_at_7cm_clears_the_table():
    gc, vet = _vet()
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(0.40, 0.40, 0.07, 90.0))
    res = vet.check(q_pre, q_grasp, support_z=0.0, box=None, margin_m=0.0075, jaw_gap_m=0.09)
    assert res.ok is True and res.reason is None
    assert 0.010 < res.plane_clearance_m < 0.025


@needs_pin
def test_the_margin_decides_at_the_measured_clearance():
    """Shift the support plane to 7 mm / 8 mm under the measured lowest point:
    7.5 mm refuses the first and admits the second (strict >= margin)."""
    gc, vet = _vet()
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(0.40, 0.40, 0.07, 90.0))
    lowest = vet.check(q_pre, q_grasp, support_z=0.0, box=None, margin_m=0.0,
                       jaw_gap_m=0.09).plane_clearance_m
    near = vet.check(q_pre, q_grasp, support_z=lowest - 0.007, box=None, margin_m=0.0075, jaw_gap_m=0.09)
    far = vet.check(q_pre, q_grasp, support_z=lowest - 0.008, box=None, margin_m=0.0075, jaw_gap_m=0.09)
    assert near.ok is False and near.plane_clearance_m == pytest.approx(0.007, abs=1e-9)
    assert far.ok is True and far.plane_clearance_m == pytest.approx(0.008, abs=1e-9)


@needs_pin
def test_the_approach_is_sampled_densely_from_pregrasp_to_grasp():
    """Every vetted pose pair along q_pre -> q_grasp moves no hull vertex
    more than APPROACH_STEP_M; s = 0 (pregrasp) and 1 (grasp) are included,
    as are the runtime's descent samples."""
    gc, vet = _vet()
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(0.40, 0.40, 0.07, 90.0))
    res = vet.check(q_pre, q_grasp, support_z=0.0, box=None, margin_m=0.0075, jaw_gap_m=0.09)
    s = np.asarray(res.samples)
    assert s[0] == 0.0 and s[-1] == 1.0 and np.all(np.diff(s) > 0)
    for d in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
        assert np.isclose(s, d).any()
    assert len(s) > 40  # a 4 cm approach in <= 1 mm steps
    poses = [np.concatenate(list(vet.placed(q_pre + si * (q_grasp - q_pre), 0.09).values())) for si in s]
    chord = max(float(np.linalg.norm(b - a, axis=1).max()) for a, b in zip(poses, poses[1:]))
    assert chord <= gc.APPROACH_STEP_M + 1e-9


@needs_pin
def test_refusal_names_the_pose_where_the_pregrasp_already_collides():
    """A box sitting where the housing is at the PREGRASP (s = 0) but which the
    grasp pose has left behind is still caught: the approach start is vetted,
    not only the grasp pose."""
    gc, vet = _vet()
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(0.40, 0.40, 0.07, 90.0))
    housing_pre = vet.placed(q_pre, 0.09)["gripper_end"]
    # The housing's rear corner on its wide flank (|jaw coordinate| > 80 mm,
    # where nothing behind the housing reaches: the wrist links are at most
    # ~66 mm off-axis) at the pregrasp. The grasp pose has carried the
    # housing 40 mm forward along the approach, out of that corner.
    jaw = g.rotation[:, 1]
    flank = housing_pre[np.abs((housing_pre - g.position) @ jaw) > 0.08]
    corner = flank[np.argmin(flank @ g.approach)]
    box = gc.Box(centre=corner, axes=np.eye(3), half=np.full(3, 0.002))
    res = vet.check(q_pre, q_grasp, support_z=-1.0, box=box, margin_m=0.0, jaw_gap_m=0.09)
    assert res.ok is False and res.box_part == "gripper_end" and res.box_s == 0.0
    assert "observed box" in res.reason
    # the grasp pose alone is clear of that box
    alone = vet.check(q_grasp, q_grasp, support_z=-1.0, box=box, margin_m=0.0, jaw_gap_m=0.09)
    assert alone.ok is True


@needs_pin
def test_observed_box_inside_the_jaw_gap_passes_and_a_box_wider_than_the_gap_is_refused():
    """The jaw gap is exempt by geometry (no collision hull there): a 5 cm
    cube centred between the open 90 mm jaws clears both fingers by 20 mm. A
    box wider than the opening puts the fingers into it."""
    gc, vet = _vet()
    x, y, z = 0.35, 0.10, 0.07
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(x, y, z, 45.0))
    jaw = g.rotation[:, 1]
    fits = vet.check(q_pre, q_grasp, support_z=0.0, box=_upright_box(x, y, (0.05, 0.05, 0.08), jaw),
                     margin_m=0.0075, jaw_gap_m=0.09)
    assert fits.ok is True
    assert fits.box_part in ("gripper_left", "gripper_right")
    assert fits.box_clearance_m == pytest.approx(0.020, abs=0.0015)
    wide = vet.check(q_pre, q_grasp, support_z=0.0, box=_upright_box(x, y, (0.10, 0.05, 0.08), jaw),
                     margin_m=0.0075, jaw_gap_m=0.09)
    assert wide.ok is False and wide.box_part in ("gripper_left", "gripper_right")
    assert "observed box" in wide.reason
    # the same wide box rotated so the jaws close across its 5 cm side fits
    across = vet.check(q_pre, q_grasp, support_z=0.0,
                       box=_upright_box(x, y, (0.05, 0.10, 0.08), jaw),
                       margin_m=0.0075, jaw_gap_m=0.09)
    assert across.ok is True


@needs_pin
def test_a_narrower_jaw_opening_moves_the_fingers_onto_the_box():
    gc, vet = _vet()
    x, y, z = 0.35, 0.10, 0.07
    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(x, y, z, 45.0))
    box = _upright_box(x, y, (0.05, 0.05, 0.08), g.rotation[:, 1])
    assert vet.check(q_pre, q_grasp, support_z=0.0, box=box, margin_m=0.0075, jaw_gap_m=0.07).ok
    narrow = vet.check(q_pre, q_grasp, support_z=0.0, box=box, margin_m=0.0075, jaw_gap_m=0.06)
    assert narrow.ok is False and narrow.box_clearance_m == pytest.approx(0.005, abs=0.0015)


def _cube_fix(centre=(0.45, 0.35, 0.025), size=(0.05, 0.07, 0.05), yaw=0.3, top_only=False):
    rng = np.random.default_rng(0)
    c, half = np.asarray(centre, dtype=float), np.asarray(size, dtype=float) / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    local = rng.uniform(-half, half, size=(600, 3))
    if top_only:
        local[:, 2] = half[2]
    pts = c + local @ R.T
    order = np.argsort(-np.asarray(size))
    return ObjectFix(label="cube", position=c, points=pts,
                     detection=Detection(label="cube", conf=0.9, bbox=np.zeros(4)),
                     extent=np.asarray(size)[order], axes=R[:, order])


def test_target_box_is_the_observed_box_down_to_the_support():
    """Upright, yawed to the planner's jaw axis (the narrow footprint side),
    enclosing the OBB footprint, from the support plane (when the cloud does
    not resolve the bottom -- the planner's rule) to the observed top."""
    from cascade.grasping.gripper_clearance import target_box

    fix = _cube_fix()
    box = target_box(fix, table_z=0.0)
    u = box.axes[:, 0]
    assert abs(abs(u @ [np.cos(0.3), np.sin(0.3), 0.0]) - 1.0) < 1e-9  # jaw axis = 5 cm side
    np.testing.assert_allclose(box.axes[:, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(box.half[:2], [0.025, 0.035], atol=1e-9)
    assert box.centre[2] - box.half[2] == pytest.approx(0.0, abs=1e-3)
    assert box.centre[2] + box.half[2] == pytest.approx(fix.points[:, 2].max())
    np.testing.assert_allclose(box.centre[:2], fix.position[:2])
    top = target_box(_cube_fix(top_only=True), table_z=0.0)  # top-only cloud: down to the table
    assert top.centre[2] - top.half[2] == pytest.approx(0.0)
    raised = target_box(_cube_fix(centre=(0.45, 0.35, 0.125)), table_z=0.0)  # resolved bottom
    assert raised.centre[2] - raised.half[2] == pytest.approx(0.1, abs=1e-3)


@needs_pin
def test_geometry_for_another_model_or_stale_meshes_fails_closed(tmp_path):
    from cascade.grasping import gripper_clearance as gc
    from conftest import SO101_URDF

    with pytest.raises(gc.ClearanceGeometryError, match="model"):
        gc.load_geometry(SO101_URDF)
    stale = _asset()
    stale["sources"][next(k for k in stale["sources"] if k.endswith("link5.STL"))] = "0" * 64
    path = tmp_path / "hulls.json"
    path.write_text(json.dumps(stale))
    with pytest.raises(gc.ClearanceGeometryError, match="link5.STL"):
        gc.load_geometry(URDF, path=path)
    wrong = _asset()
    wrong["schema"] = "x"
    path.write_text(json.dumps(wrong))
    with pytest.raises(gc.ClearanceGeometryError, match="schema"):
        gc.load_geometry(URDF, path=path)


# ── golden: the default path is unchanged ─────────────────────────────────────


def test_golden_only_the_reach_profiles_enable_the_vet():
    from cascade.config import _load_profile_raw

    arms = REPO / "configs" / "arms"
    for path in sorted(arms.glob("*.yaml")):
        if _load_profile_raw("arms", path.stem, arms.parent).get("template") is True:
            continue
        cfg = load_demo_config(arm=path.stem)
        expected = path.stem in REACH_PROFILES
        assert cfg.grasp.get("angled_clearance_vet", False) is expected, path.stem
        assert cfg.arms[0]["resolved"]["grasp"].get("angled_clearance_vet", False) is expected
    text = (REPO / "configs" / "demo.yaml").read_text()
    assert "angled_clearance_vet: false" in text and "B73" in text
    assert load_demo_config().grasp.angled_clearance_vet is False


def test_golden_planner_marks_only_its_tilted_candidates():
    from cascade.grasping.obb_grasp import AngledGrasp, plan_grasps_from_fix

    fix = _cube_fix(centre=(0.30, 0.10, 0.04), size=(0.05, 0.07, 0.08))
    plain = plan_grasps_from_fix(fix, table_z=0.0, depth_fraction=0.15)
    assert all(type(g) is Grasp for g in plain)
    tilted = plan_grasps_from_fix(fix, table_z=0.0, depth_fraction=0.15, angled_tilts_deg=[30, 45, 90])
    assert all(type(g) is Grasp for g in tilted[:4])
    assert len(tilted) == 16 and all(type(g) is AngledGrasp for g in tilted[4:])


def test_golden_flip_twin_keeps_the_candidate_kind_and_evidence_shape():
    from dataclasses import fields

    from cascade.grasping.obb_grasp import AngledGrasp
    from cascade.grasping.selector import _flip_twin

    plain = _tilted(0.35, 0.1, 0.05, 45.0)
    angled = _tilted(0.35, 0.1, 0.05, 45.0, cls=AngledGrasp)
    assert type(_flip_twin(plain)) is Grasp and type(_flip_twin(angled)) is AngledGrasp
    for a, b in ((plain, angled), (_flip_twin(plain), _flip_twin(angled))):
        assert a.rotation.tobytes() == b.rotation.tobytes()
        assert [f.name for f in fields(a)] == [f.name for f in fields(b)]


# ── behaviour: the real runtime on the mock stack ─────────────────────────────


#: where the mock cube sits: side, 30 and 45 degree candidates all solve there
CUBE_XY = (0.45, 0.35)


def _mock_runtime(tmp_path, profile, *, vet=None):
    from cascade.apps.demo import build_runtime

    cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
    cfg._data["grasp"]["backend"] = "obb"
    if vet is not None:
        cfg._data["grasp"]["angled_clearance_vet"] = vet
    for cam in [cfg._data["camera"], *(cfg._data.get("cameras") or [])]:
        cam["extrinsics"]["T"][0][3] = 0.28 + (CUBE_XY[0] - 0.29)
        cam["extrinsics"]["T"][1][3] = CUBE_XY[1]
    return build_runtime(cfg, tmp_path / f"run-{profile}-{vet}")


def _grasp_with_side_first(tmp_path, monkeypatch, profile, *, vet=None):
    """Real grasp_object; the analytic planner's output with ONE extra side
    candidate ranked first (the planner itself ranks side last, behind 30
    and 45 degrees). On main (no vet) the side grasp is executed."""
    import cascade.grasping.obb_grasp as obb
    import cascade.skills.runtime as rtmod
    from cascade.apps.demo import shutdown_runtime

    evidence = tmp_path / "evidence"
    monkeypatch.setenv("CASCADE_GRASP_EVIDENCE_DIR", str(evidence))
    cls = getattr(obb, "AngledGrasp", Grasp)
    chosen, loads = [], []
    real_plan, real_select = rtmod.plan_grasps_from_fix, rtmod.select_grasp

    def spy_plan(fix, **kw):
        out = real_plan(fix, **kw)
        x, y, z = out[0].position
        return [_tilted(x, y, z, 90.0, cls=cls, quality=1.0), *out]

    def spy_select(grasps, *a, **kw):
        out = real_select(grasps, *a, **kw)
        chosen.append(out[0])
        return out

    monkeypatch.setattr(rtmod, "plan_grasps_from_fix", spy_plan)
    monkeypatch.setattr(rtmod, "select_grasp", spy_select)
    try:
        import cascade.grasping.gripper_clearance as gc

        real_load = gc.load_geometry
        monkeypatch.setattr(gc, "load_geometry", lambda *a, **k: loads.append(a) or real_load(*a, **k))
    except ImportError:
        pass
    runtime, arm = _mock_runtime(tmp_path, profile, vet=vet)
    try:
        arm.object_stop_frac = 0.5
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        with runtime.watcher.paused():
            result = runtime.execute("grasp_object", {"label": "red cube"})
    finally:
        shutdown_runtime(runtime, arm)
    events = []
    for receipt in sorted(evidence.glob("*.json")):
        events += json.loads(receipt.read_text())["events"]
    return result, chosen, events, loads


def _tilt_deg(g) -> float:
    return float(np.degrees(np.arccos(np.clip(-g.approach[2], -1.0, 1.0))))


@needs_pin
def test_reach_profile_refuses_the_penetrating_side_candidate_and_falls_back(tmp_path, monkeypatch):
    result, chosen, events, loads = _grasp_with_side_first(tmp_path, monkeypatch, "mock_reach")
    assert result["ok"] is True, result
    g = chosen[-1]
    assert 25.0 < _tilt_deg(g) < 50.0, _tilt_deg(g)  # the next candidate: a 30/45 lean
    clear = [e["data"] for e in events if e["kind"] == "gripper_clearance"]
    refused = [c for c in clear if c["decision"] == "refused"]
    admitted = [c for c in clear if c["decision"] == "admitted"]
    assert refused and all(c["plane_part"] == "link5" and c["plane_clearance_m"] < 0 for c in refused)
    assert all("support plane" in c["reason"] for c in refused)
    assert all(abs(c["approach"][2]) < 1e-9 for c in refused)  # the side candidate (both jaws)
    assert admitted and admitted[-1]["reason"] is None
    assert admitted[-1]["margin_m"] == pytest.approx(0.0075)
    assert admitted[-1]["jaw_gap_m"] == pytest.approx(0.09)
    reasons = [e["data"]["reason"] for e in events if e["kind"] == "candidate_validation"]
    assert any(r and "gripper clearance" in r and "link5" in r for r in reasons)
    assert len(loads) == 1


@needs_pin
def test_reach_profile_with_the_vet_off_executes_the_side_candidate_unvetted(tmp_path, monkeypatch):
    """The A/B switch: `angled_clearance_vet: false` is the B45 behaviour."""
    result, chosen, events, loads = _grasp_with_side_first(tmp_path, monkeypatch, "mock_reach", vet=False)
    assert result["ok"] is True, result
    assert _tilt_deg(chosen[-1]) == pytest.approx(90.0)
    assert not [e for e in events if e["kind"] == "gripper_clearance"]
    assert loads == []


@needs_pin
def test_reach_profile_rejects_a_malformed_vet_flag(tmp_path, monkeypatch):
    result, chosen, events, loads = _grasp_with_side_first(tmp_path, monkeypatch, "mock_reach", vet="yes")
    assert result["ok"] is False and "angled_clearance_vet" in str(result), result
    assert chosen == []


@needs_pin
def test_unverifiable_geometry_refuses_every_angled_candidate_but_not_topdown(tmp_path, monkeypatch):
    """Fail closed: with the hulls unloadable, no angled candidate is admitted
    (the object at (0.45, 0.35) is beyond top-down reach, so nothing is), and
    the reason says why."""
    import cascade.grasping.gripper_clearance as gc

    def broken(*a, **k):
        raise gc.ClearanceGeometryError("geometry unavailable for this test")

    monkeypatch.setattr(gc, "load_geometry", broken)
    result, chosen, events, _ = _grasp_with_side_first(tmp_path, monkeypatch, "mock_reach")
    assert result["ok"] is False and "no executable grasp" in result["error"], result
    assert "gripper clearance unverifiable" in result["error"]
    assert chosen == []


@needs_pin
def test_a_crashing_clearance_check_refuses_the_candidate_with_the_reason(tmp_path, monkeypatch):
    """Fail closed per candidate: an exception inside the check is a refusal
    whose reason names it (and is in the evidence), never an admission."""
    import cascade.grasping.gripper_clearance as gc

    def boom(self, *a, **k):
        raise RuntimeError("hull check crashed")

    monkeypatch.setattr(gc.ApproachClearance, "check", boom)
    result, chosen, events, _ = _grasp_with_side_first(tmp_path, monkeypatch, "mock_reach")
    assert result["ok"] is False and chosen == [], result
    assert "gripper clearance unverifiable: RuntimeError: hull check crashed" in result["error"]
    clear = [e["data"] for e in events if e["kind"] == "gripper_clearance"]
    assert clear and all(c["decision"] == "refused" and "hull check crashed" in c["reason"] for c in clear)


@needs_pin
def test_a_learned_style_tilted_candidate_is_not_vetted(tmp_path, monkeypatch):
    """Scope: only the planner's own tilted candidates (`AngledGrasp`). A
    plain `Grasp` leaning the same way -- what a learned backend returns --
    is executed as before even with the vet on (B45: learned candidates gain
    the box, nothing else)."""
    import cascade.grasping.obb_grasp as obb

    monkeypatch.setattr(obb, "AngledGrasp", Grasp)  # the injected side candidate becomes plain
    result, chosen, events, _ = _grasp_with_side_first(tmp_path, monkeypatch, "mock_reach")
    assert result["ok"] is True, result
    assert _tilt_deg(chosen[-1]) == pytest.approx(90.0)
    assert not [e for e in events if e["kind"] == "gripper_clearance"
                and abs(e["data"]["approach"][2]) < 1e-9]


@needs_pin
def test_the_adaptive_pregrasp_opening_sets_the_vetted_jaw_gap(tmp_path, monkeypatch):
    """With `gripper.pregrasp_open_margin_m` the jaws open to width + margin,
    and the fingers are vetted there (not at full open)."""
    import cascade.apps.demo as demo

    real_build = demo.build_runtime

    def build(cfg, run_dir, *a, **k):
        for arm in [cfg._data["arm"], *(cfg._data.get("arms") or [])]:
            arm.setdefault("gripper", {})["pregrasp_open_margin_m"] = 0.01
        return real_build(cfg, run_dir, *a, **k)

    monkeypatch.setattr(demo, "build_runtime", build)
    result, chosen, events, _ = _grasp_with_side_first(tmp_path, monkeypatch, "mock_reach")
    clear = [e["data"] for e in events if e["kind"] == "gripper_clearance"]
    assert clear
    for c in clear:
        width = c["jaw_gap_m"] - 0.01
        assert 0.03 < width < 0.08  # the planner's width (object + pad), not the full 0.09
    assert result["ok"] is True, result


@needs_pin
def test_the_check_uses_the_harness_table_height_for_plane_and_box(tmp_path):
    """The support plane is the arm harness's `table_z` (here 12 mm up): the
    7 cm side grasp, ~17 mm over z = 0, is ~5 mm over this table and
    refused; the observed box reaches down to it; both are in the evidence."""
    from types import SimpleNamespace

    from cascade.grasping import evidence as ev
    from cascade.skills.runtime import SkillRuntime

    kin, g, q_pre, q_grasp = _solve("rebot_rs_reach", _tilted(0.40, 0.40, 0.07, 90.0, cls=Grasp))
    stub = SimpleNamespace(
        cfg=SimpleNamespace(grasp={"angled_clearance_vet": True, "width_pad_m": 0.015},
                            arm={"model": str(URDF), "gripper": {}}),
        arm=SimpleNamespace(harness=SimpleNamespace(limits=SimpleNamespace(table_z=0.012))),
        _max_width=0.09, kin=kin)
    fix = _cube_fix(centre=(0.40, 0.40, 0.04), size=(0.05, 0.05, 0.08), yaw=0.0, top_only=True)
    check = SkillRuntime._angled_clearance_check(stub, fix)
    attempt = ev.Attempt(str(tmp_path))
    token = ev._ACTIVE.set(attempt)
    try:
        reason = check(g, q_pre, q_grasp)
    finally:
        ev._ACTIVE.reset(token)
    assert reason and reason.startswith("gripper clearance: link5 clears the support plane by +")
    (event,) = [e["data"] for e in attempt.events if e["kind"] == "gripper_clearance"]
    assert event["decision"] == "refused" and event["support_z_m"] == 0.012
    centre, half = np.asarray(event["box"]["centre"]), np.asarray(event["box"]["half"])
    assert centre[2] - half[2] == pytest.approx(0.012)
    gc, vet = _vet()
    on_floor = vet.check(q_pre, q_grasp, support_z=0.0, box=None, margin_m=0.0075, jaw_gap_m=0.09)
    assert on_floor.ok and on_floor.plane_clearance_m > 0.0075
    assert event["plane_clearance_m"] == pytest.approx(on_floor.plane_clearance_m - 0.012, abs=1e-9)
    assert event["plane_clearance_m"] < 0.0075


@needs_pin
def test_default_profile_never_loads_the_geometry(tmp_path, monkeypatch):
    """Golden: the shipped `mock` profile plans top-down at (0.29, 0) exactly
    as before -- no geometry load, no clearance evidence, and the grasp skill
    does not even build the check (stub runtimes in older tests lack it)."""
    import cascade.grasping.gripper_clearance as gc
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.skills.runtime import SkillRuntime

    loads, built = [], []
    monkeypatch.setattr(gc, "load_geometry", lambda *a, **k: loads.append(a))
    monkeypatch.setattr(SkillRuntime, "_angled_clearance_check", lambda self, fix: built.append(fix))
    monkeypatch.setenv("CASCADE_GRASP_EVIDENCE_DIR", str(tmp_path / "evidence"))
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["backend"] = "obb"
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        arm.object_stop_frac = 0.5
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        with runtime.watcher.paused():
            result = runtime.execute("grasp_object", {"label": "red cube"})
    finally:
        shutdown_runtime(runtime, arm)
    assert result["ok"] is True, result
    assert loads == [] and built == []
    events = [e for p in (tmp_path / "evidence").glob("*.json") for e in json.loads(p.read_text())["events"]]
    assert events and not [e for e in events if e["kind"] == "gripper_clearance"]


# ── the measurement on the B45 grid ───────────────────────────────────────────


def _study():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import angled_clearance_study

    return angled_clearance_study


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text())


def test_study_evidence_is_bound_to_its_inputs():
    data = _evidence()
    prov = data["provenance"]
    assert data["schema"] == "cascade.angled_clearance_study/1"
    assert prov["model_sha256"] == hashlib.sha256(URDF.read_bytes()).hexdigest()
    assert prov["hulls_sha256"] == hashlib.sha256(ASSET.read_bytes()).hexdigest()
    assert prov["b45_evidence_sha256"] == hashlib.sha256(B45_EVIDENCE.read_bytes()).hexdigest()
    assert prov["arm"] == "rebot_rs_reach"
    assert prov["margin_m"] == 0.0075 and prov["jaw_gap_m"] == 0.09
    from cascade.grasping.gripper_clearance import APPROACH_STEP_M

    assert prov["approach_step_m"] == APPROACH_STEP_M
    assert prov["families"] == {"out30": 30.0, "out45": 45.0, "side": 90.0}
    assert set(prov["objects"]) == {"0.02", "0.05", "0.07", "0.1"}


def test_study_covers_every_b45_reachable_point_of_the_planners_tilts():
    """Every (point, tilt) the B45 envelope pass reached with the planner's
    roll is in the study, and every refusal is the clearance vet's (IK and
    the harness still admit all of them)."""
    study = _study()
    b45 = json.loads(B45_EVIDENCE.read_text())
    import reachability_study as rs

    expected = {(x, y, z, f) for x, y, z, fam in rs.rows_from_json(b45["passes"]["envelope_box"]["rows"])
                for f in ("out30", "out45", "side") if fam[f][0] & 1}
    rows = study.rows_from_json(_evidence()["rows"])
    assert {(r["x"], r["y"], r["z"], r["family"]) for r in rows} == expected
    assert all(r["harness_ok"] for r in rows)
    assert _evidence()["summary"]["non_clearance_refusals"] == 0


def test_study_summary_is_rederived_from_its_rows():
    study = _study()
    data = _evidence()
    rows = study.rows_from_json(data["rows"])
    assert study.summarize(rows, data["provenance"]["margin_m"]) == data["summary"]
    assert study.envelope_check(rows, json.loads(B45_EVIDENCE.read_text())) == data["envelope"]
    assert data["envelope"]["cells_losing_admission"] == []
    assert data["envelope"]["admitted_cells_before"] == data["envelope"]["admitted_cells_after"] > 0


def test_envelope_check_counts_only_the_surviving_tilts():
    """The envelope rule with the vet on: a cell keeps admission only through
    top-down or a SURVIVING tilt. Refusing every tilt leaves the top-down
    cells (the same cells B45 measured for top-down alone)."""
    study = _study()
    b45 = json.loads(B45_EVIDENCE.read_text())
    rows = study.rows_from_json(_evidence()["rows"])
    real = study.envelope_check(rows, b45)
    none = study.envelope_check([dict(r, survived=False) for r in rows], b45)
    assert none["admitted_cells_before"] == real["admitted_cells_before"]
    assert 0 < none["admitted_cells_after"] < real["admitted_cells_after"]
    assert len(none["cells_losing_admission"]) == real["admitted_cells_after"] - none["admitted_cells_after"]


def test_docs_quote_the_measured_numbers():
    study = _study()
    summary = _evidence()["summary"]
    doc = (REPO / "docs" / "REACH_ENVELOPE.md").read_text()
    for line in study.doc_table(summary):
        assert line in doc, line
    roadmap = (REPO / "docs" / "ROADMAP.md").read_text()
    assert study.roadmap_sentence(summary) in roadmap
    assert "contact/gripper-housing collision of angled and side grasps" in roadmap
    assert "~~contact/gripper-housing collision of angled and side grasps~~" in roadmap


@needs_pin
def test_re_evaluating_study_points_reproduces_the_committed_rows():
    study = _study()
    committed = {(r["x"], r["y"], r["z"], r["family"]): r for r in study.rows_from_json(_evidence()["rows"])}
    ctx = study.context()
    for key in [(0.4, 0.4, 0.05, "side"), (0.4, 0.4, 0.07, "side"), (0.35, 0.1, 0.02, "out45"),
                (0.125, -0.4, 0.02, "out30"), (0.45, 0.5, 0.1, "side"), (0.3, 0.2, 0.1, "out45")]:
        assert key in committed, key
        fresh, row = study.evaluate(ctx, *key), committed[key]
        assert {k: v for k, v in fresh.items() if not k.endswith("_mm")} == \
            {k: v for k, v in row.items() if not k.endswith("_mm")}, key
        for k in (k for k in row if k.endswith("_mm")):  # IK last-bit noise across platforms
            assert fresh[k] == pytest.approx(row[k], abs=0.05), (key, k)


@needs_pin
def test_the_studys_vet_is_the_runtimes_vet(tmp_path, monkeypatch):
    """The study's combined vet (harness, then clearance) gives the reasons
    the runtime's own `_vet` gives for angled candidates under `mock_reach`:
    the validate callback the real grasp_object hands select_grasp is
    captured and called on the same candidates and joint solutions, with the
    study's object box replaced by the runtime's (observed) one."""
    import cascade.skills.runtime as rtmod
    from cascade.grasping.obb_grasp import AngledGrasp
    from cascade.types import make_transform

    study = _study()
    ctx = study.context("mock_reach")
    cases = []
    for x, y, z, tilt in [(0.40, 0.40, 0.05, 90.0), (0.40, 0.40, 0.07, 90.0), (0.35, 0.10, 0.02, 45.0),
                          (0.30, -0.30, 0.02, 45.0), (0.575, 0.0, 0.07, 45.0)]:
        g = _tilted(x, y, z, tilt, cls=AngledGrasp)
        pre = ctx["kin"].ik(make_transform(g.rotation, g.position - g.approach * ctx["pregrasp_offset_m"]),
                            ctx["home_q"])
        grasp = ctx["kin"].ik(make_transform(g.rotation, g.position), pre.q)
        assert pre.success and grasp.success, (x, y, z, tilt)
        cases.append((g, pre.q, grasp.q))
    theirs, boxes = [], []
    real_select = rtmod.select_grasp

    def spy_select(grasps, *a, **kw):
        if not theirs:
            theirs.extend(kw["validate"](*c) for c in cases)
        return real_select(grasps, *a, **kw)

    import cascade.grasping.gripper_clearance as gc

    real_box = gc.target_box
    monkeypatch.setattr(gc, "target_box", lambda *a, **k: boxes.append(real_box(*a, **k)) or boxes[-1])
    monkeypatch.setattr(rtmod, "select_grasp", spy_select)
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock_reach", llm="mock")
    cfg._data["grasp"]["backend"] = "obb"
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        arm.object_stop_frac = 0.5
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        with runtime.watcher.paused():
            result = runtime.execute("grasp_object", {"label": "red cube"})
    finally:
        shutdown_runtime(runtime, arm)
    assert result["ok"] is True, result
    assert len(boxes) == 1
    ours = [study.combined_vet(ctx, boxes[0])(*c) for c in cases]
    assert theirs == ours
    kinds = [None if r is None else r.split(":")[0] for r in ours]
    assert kinds == ["gripper clearance", None, "gripper clearance", None, "descent unsafe"], ours
    # the side grasp at 5 cm hits the table; the 45 degree grasp next to the
    # mock cube at (0.29, 0) brings a finger within the margin of its box
    assert "support plane" in ours[0] and "link5" in ours[0]
    assert "observed box" in ours[2]
