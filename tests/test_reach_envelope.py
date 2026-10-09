"""B45: measured reach envelope of the reBot RS (Seeed feedback, 8 Oct 2026:
"very limited AREA constraints for skills, the object has to be really close
to the arm").

What is pinned here, and why:

* PREMISES (pass on main by design): the shipped box was drawn from a
  top-down-only probe, but the runtime's own IK + harness vet cannot reach
  the far part of that box top-down, while an angled approach can -- inside
  the box today and beyond it if the box allowed it. So the limit was the
  approach family as much as the box.
* EVIDENCE: `scripts/reachability_study.py` measured every approach family
  over a table grid through the real selector and the runtime's harness vet;
  its JSON is bound to the shipped URDF and settings, and the opt-in
  envelope is re-derivable from its own rows (no hand-tuned numbers).
* OPT-IN: the `*_reach` arm profiles carry exactly the measured box and the
  measured tilts and change nothing else; every other profile is golden.
* BEHAVIOUR through the real paths: study points pass IK + `vet_pose` under
  the reach profile and are refused by the default one; points beyond the
  envelope still refuse; the analytic planner's tilted candidates; and a full
  mock-stack `grasp_object` beyond the default box (refused by `mock`,
  executed by `mock_reach` with a tilted grasp).
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

EVIDENCE = REPO / "docs" / "evidence" / "reach-envelope-20261009" / "rebot_reachability.json"
REACH_PROFILES = {"rebot_rs_reach": "rebot_rs", "isaac_reach": "isaac", "mock_reach": "mock"}
DEFAULT_BOX = [[0.10, -0.30, -0.01], [0.50, 0.30, 0.55]]
#: derived by the study (`derive_envelope`), pinned in the reach profiles
ENVELOPE_BOX = [[0.10, -0.50, -0.01], [0.55, 0.50, 0.55]]
ENVELOPE_TILTS = [30, 45, 90]
#: every shipped profile of the reBot family that must keep the shipped box
REBOT_FAMILY = ("rebot_rs", "rebot_rs_mb", "mock", "isaac", "isaac_cumotion", "isaac_kitchen",
                "isaac_kitchen_cumotion", "isaac_kitchen_gpu", "isaac_kitchen_hug")


@pytest.fixture(autouse=True)
def _private_ports(monkeypatch):
    """Ports inside this item's block (45800-45899); nothing listens there."""
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "45811")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "45812")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "45813")
    monkeypatch.setenv("CASCADE_HUG_PORT", "45814")
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")


def _study():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import reachability_study

    return reachability_study


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text())


# ── inline rig (no script import: the premises must run on main) ─────────────


def _rig(profile: str):
    """Kinematics + harness exactly as apps.demo._build_arm builds them."""
    from cascade.control.kinematics import Kinematics
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    cfg = load_demo_config(arm=profile)
    a, view = Cfg(cfg.arms[0]), Cfg(cfg.arms[0]["resolved"])
    kin = Kinematics(a.model, a.get("ee_frame", "gripper_end"), n_controlled=int(a.n_joints),
                     joint_signs=a.get("joint_signs"))
    harness = SafetyHarness(SafetyLimits.from_config(view.safety), kinematics=kin)
    return view, a, kin, harness


def _tilted(x, y, z, tilt_deg) -> Grasp:
    """Approach leaning `tilt_deg` from straight down AWAY from the base, jaw
    horizontal and tangential (the study's `out*` family, roll 0)."""
    from cascade.grasping.obb_grasp import tool_rotation

    r = np.array([x, y, 0.0]) / np.hypot(x, y)
    t = np.cross([0.0, 0.0, 1.0], r)
    th = np.radians(tilt_deg)
    approach = np.cos(th) * np.array([0.0, 0.0, -1.0]) + np.sin(th) * r
    return Grasp(position=np.array([x, y, z], dtype=float), rotation=tool_rotation(approach, t),
                 width_m=0.05, approach=approach, quality=1.0, label=f"tilt{tilt_deg}")


def _vet(view, home, harness):
    """The harness half of the runtime's `_vet`, in its order."""
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


def _select(profile, grasp):
    from cascade.grasping.selector import select_grasp

    view, a, kin, harness = _rig(profile)
    home = np.asarray(a.home_q, dtype=float)
    return select_grasp([grasp], kin, home, max_width_m=0.09,
                        pregrasp_offset_m=float(view.grasp.get("pregrasp_offset_m", 0.12)),
                        validate=_vet(view, home, harness), preserve_order=True)


# ── premises (pass on main by design) ─────────────────────────────────────────


@needs_pin
def test_premise_topdown_cannot_reach_the_far_part_of_the_shipped_box_but_an_angled_approach_can():
    """(0.475, 0, 0.07) is INSIDE the shipped box (x <= 0.50). The runtime's
    top-down candidate fails IK there; leaning 45 degrees out solves and
    passes the shipped harness. The top-down probe the box came from was
    optimistic, and the approach family is part of the limit."""
    from cascade.grasping.selector import NoExecutableGrasp

    with pytest.raises(NoExecutableGrasp, match="IK failed"):
        _select("rebot_rs", _tilted(0.475, 0.0, 0.07, 0.0))
    grasp, q_pre, q_grasp = _select("rebot_rs", _tilted(0.475, 0.0, 0.07, 45.0))
    assert grasp.approach[2] > -0.75


@needs_pin
def test_premise_the_shipped_box_refuses_a_solvable_angled_grasp_beyond_it():
    """At (0.525, 0, 0.07) the out45 grasp has IK; only the shipped box
    (x <= 0.50) refuses it -- the region the opt-in envelope admits."""
    from cascade.grasping.selector import NoExecutableGrasp
    from cascade.types import make_transform

    _, a, kin, _ = _rig("rebot_rs")
    g = _tilted(0.525, 0.0, 0.07, 45.0)
    assert kin.ik(make_transform(g.rotation, g.position), np.asarray(a.home_q, float)).success
    with pytest.raises(NoExecutableGrasp, match="outside workspace"):
        _select("rebot_rs", g)


# ── the evidence ──────────────────────────────────────────────────────────────


def test_study_evidence_is_bound_to_the_shipped_urdf_and_runtime_settings():
    """A URDF, home pose, pregrasp offset, exemption or margin change makes
    the measured envelope stale: re-run scripts/reachability_study.py."""
    data = _evidence()
    prov = data["provenance"]
    assert data["schema"] == "cascade.reachability_study/1"
    assert prov["model_sha256"] == hashlib.sha256(URDF.read_bytes()).hexdigest()
    assert prov["model"] == str(URDF.relative_to(REPO))
    rebot = load_demo_config(arm="rebot_rs")
    view = Cfg(rebot.arms[0]["resolved"])
    assert prov["arm"] == "rebot_rs"
    assert prov["home_q"] == list(rebot.arm.home_q)
    assert prov["pregrasp_offset_m"] == view.grasp.pregrasp_offset_m
    assert prov["exempt_radius_m"] == view.grasp.exempt_radius_m
    assert prov["move_duration_s"] == view.grasp.move_duration_s
    assert prov["table_z"] == view.safety.table_z
    assert prov["joint_margin"] == view.safety.joint_margin < prov["ik_limit_margin"] == 0.025
    assert prov["default_workspace"] == DEFAULT_BOX
    assert prov["keep_out"] == []
    assert data["grid"]["z"] == [0.02, 0.05, 0.07, 0.10]
    assert set(data["grid"]["families"]) == {
        "topdown", "out15", "out30", "out45", "about_radial15", "about_radial30", "about_radial45",
        "diag30", "diag45", "side"}


def test_study_measures_the_topdown_limit_and_what_each_family_adds():
    """The numbers the docs quote, read back from the committed result."""
    summary = _evidence()["summary"]["open"]
    for z in ("0.02", "0.05", "0.07", "0.1"):
        s = summary[z]
        top = s["topdown"]["bbox"]["r"][1]
        assert top <= 0.45, (z, top)                     # top-down: r <= 0.45 m
        assert s["out30"]["bbox"]["r"][1] >= top + 0.15  # leaning out reaches further
        assert s["out45"]["bbox"]["r"][1] >= top + 0.24
        assert s["side"]["bbox"]["r"][1] >= 0.74
        for tilt in (30, 45):  # leaning SIDEWAYS (about the radial axis) does not
            assert s[f"about_radial{tilt}"]["bbox"] is None or \
                s[f"about_radial{tilt}"]["bbox"]["r"][1] <= top + 1e-9
        assert s["any_angled"]["beyond_topdown"] > 0
    assert summary["0.1"]["topdown"]["bbox"]["r"][1] < 0.39  # 10 cm grasps: r < 0.39 m


def test_envelope_is_rederived_from_the_rows_and_verified_under_its_own_box():
    study = _study()
    data = _evidence()
    rows = study.rows_from_json(data["passes"]["open"]["rows"])
    env = study.derive_envelope(study.rows_to_table(rows), data["provenance"]["default_workspace"],
                                study._axis_values(tuple(data["grid"]["x"])),
                                study._axis_values(tuple(data["grid"]["y"])))
    assert [env["workspace_min"], env["workspace_max"]] == ENVELOPE_BOX
    assert data["envelope"]["workspace_min"] == env["workspace_min"]
    assert data["envelope"]["workspace_max"] == env["workspace_max"]
    assert data["passes"]["envelope_box"]["workspace"] == ENVELOPE_BOX
    v = data["envelope"]["verified"]
    assert v["envelope_pass_points_outside_box"] == 0
    assert v["lost"] == 0 and v["gained"] > 0
    assert v["envelope_families_roll0_envelope_box"] == v["topdown_roll0_default_box"] + v["gained"]
    # every grid sample the box adds outside the default box is admitted
    assert env["admitted_cells_outside_default"] > 0


def test_derive_envelope_never_admits_an_unmeasured_cell():
    """Inscribed, not bounding: one unreachable sample in a strip keeps the
    box from growing over it."""
    study = _study()
    xs, ys = [0.10, 0.20, 0.30], [-0.2, -0.1, 0.0, 0.1, 0.2]
    default = [[0.10, -0.1, -0.01], [0.20, 0.1, 0.55]]
    full = {f: 1 for f in study.FAMILY_NAMES}
    table = {(x, y, z): dict(full) for x in xs for y in ys for z in study.Z_LEVELS}
    env = study.derive_envelope(table, default, xs, ys)
    assert env["workspace_max"][:2] == [0.30, 0.2] and env["workspace_min"][:2] == [0.10, -0.2]
    table[(0.30, 0.2, 0.07)] = {f: 0 for f in study.FAMILY_NAMES}  # one height, one corner
    env = study.derive_envelope(table, default, xs, ys)
    hi = env["workspace_max"]
    assert not (hi[0] >= 0.30 and hi[1] >= 0.2), env
    # a reachable family outside the admitted set does not count
    table[(0.30, 0.2, 0.07)] = {f: (1 if f == "about_radial45" else 0) for f in study.FAMILY_NAMES}
    assert study.derive_envelope(table, default, xs, ys)["workspace_max"] == hi
    # roll 45 only (not the planner's horizontal jaw) does not count either
    table[(0.30, 0.2, 0.07)] = {f: 2 for f in study.FAMILY_NAMES}
    assert study.derive_envelope(table, default, xs, ys)["workspace_max"] == hi
    # the default box itself is never re-litigated: an unreachable sample
    # inside it neither shrinks it nor blocks growth elsewhere
    table[(0.30, 0.2, 0.07)] = dict(full)
    table[(0.10, 0.0, 0.05)] = {f: 0 for f in study.FAMILY_NAMES}
    env = study.derive_envelope(table, default, xs, ys)
    assert env["workspace_max"][:2] == [0.30, 0.2] and env["workspace_min"][:2] == [0.10, -0.2]


# ── the opt-in profiles; every other profile golden ───────────────────────────


@pytest.mark.parametrize("profile, parent", sorted(REACH_PROFILES.items()))
def test_reach_profiles_carry_the_measured_envelope_and_change_nothing_else(profile, parent):
    """The extended box is the new LIMIT of the opt-in profile; every other
    safety value (margins, velocity cap, clearance, keep-outs) is the
    parent's, so no gate is relaxed or changed in kind."""
    reach, base = load_demo_config(arm=profile), load_demo_config(arm=parent)
    rs, bs = reach.arms[0]["resolved"], base.arms[0]["resolved"]
    assert [rs["safety"]["workspace"]["min"], rs["safety"]["workspace"]["max"]] == ENVELOPE_BOX
    assert {k: v for k, v in rs["safety"].items() if k != "workspace"} == \
        {k: v for k, v in bs["safety"].items() if k != "workspace"}
    assert rs["grasp"]["angled_approach_tilts_deg"] == ENVELOPE_TILTS
    assert {k: v for k, v in rs["grasp"].items() if k != "angled_approach_tilts_deg"} == \
        {k: v for k, v in bs["grasp"].items() if k != "angled_approach_tilts_deg"}
    strip = ("resolved", "overrides", "name")
    assert {k: v for k, v in reach.arms[0].items() if k not in strip} == \
        {k: v for k, v in base.arms[0].items() if k not in strip}
    # the top-level view (single-arm behaviour) is the same envelope
    assert [list(reach.safety.workspace.min), list(reach.safety.workspace.max)] == ENVELOPE_BOX


def test_reach_profile_tilts_are_the_study_families():
    study = _study()
    tilts = sorted(int(study.FAMILIES[f]["tilt_deg"]) for f in study.ENVELOPE_FAMILIES
                   if f != "topdown")
    assert tilts == ENVELOPE_TILTS
    assert all(study.FAMILIES[f]["lean"] == "out" for f in study.ENVELOPE_FAMILIES if f != "topdown")
    assert list(study.ENVELOPE_ROLLS_DEG) == [0.0]
    assert _evidence()["envelope"]["rule"]["families"] == list(study.ENVELOPE_FAMILIES)


@pytest.mark.parametrize("profile", REBOT_FAMILY)
def test_golden_shipped_rebot_profiles_keep_the_shipped_box_and_topdown_planning(profile):
    """Golden (true on main): the reBot family keeps x 0.10..0.50, y +-0.30,
    no keep-outs, and no tilted analytic candidates."""
    cfg = load_demo_config(arm=profile)
    view = cfg.arms[0]["resolved"]
    assert [view["safety"]["workspace"]["min"], view["safety"]["workspace"]["max"]] == DEFAULT_BOX
    assert view["safety"].get("keep_out", []) == []
    assert not view["grasp"].get("angled_approach_tilts_deg")


def test_golden_no_other_profile_enables_tilted_analytic_candidates():
    from cascade.config import _load_profile_raw

    arms = REPO / "configs" / "arms"
    for path in sorted(arms.glob("*.yaml")):
        if path.stem in REACH_PROFILES:
            continue
        if _load_profile_raw("arms", path.stem, arms.parent).get("template") is True:
            continue
        assert not load_demo_config(arm=path.stem).grasp.get("angled_approach_tilts_deg"), path.stem


def test_demo_yaml_documents_the_flag_with_the_old_behaviour_as_default():
    assert load_demo_config().grasp.angled_approach_tilts_deg == []
    text = (REPO / "configs" / "demo.yaml").read_text()
    assert "angled_approach_tilts_deg: []" in text and "REACH_ENVELOPE.md" in text


# ── behaviour: study points through IK + the harness ─────────────────────────


def _extension_points(per_family=4):
    """Deterministic sample of envelope-pass points OUTSIDE the default box,
    `per_family` for each envelope family, each reached by that family with
    the planner's roll (0)."""
    study = _study()
    rows = study.rows_from_json(_evidence()["passes"]["envelope_box"]["rows"])
    out = []
    for f in study.ENVELOPE_FAMILIES:
        pts = sorted((x, y, z, f) for x, y, z, fam in rows
                     if not (0.10 <= x <= 0.50 + 1e-9 and -0.30 - 1e-9 <= y <= 0.30 + 1e-9)
                     and fam[f][0] & 1)
        step = max(1, len(pts) // per_family)
        out += pts[::step][:per_family]
    return out


@needs_pin
def test_study_reachable_points_pass_ik_and_vet_pose_under_the_reach_profile_and_not_the_default():
    from cascade.grasping.selector import select_grasp

    study = _study()
    reach = study.build_context("rebot_rs_reach")
    default = study.build_context("rebot_rs")
    assert reach["workspace"] == ENVELOPE_BOX and default["workspace"] == DEFAULT_BOX
    points = _extension_points()
    assert len(points) == 16 and {p[3] for p in points} == set(study.ENVELOPE_FAMILIES)
    for x, y, z, family in points:
        g = study.candidate(reach, x, y, z, family, 0.0)
        g, q_pre, q_grasp = select_grasp([g], reach["kin"], reach["home_q"], max_width_m=0.09,
                                         pregrasp_offset_m=reach["pregrasp_offset_m"],
                                         validate=study.runtime_vet(reach), preserve_order=True)
        np.testing.assert_allclose(reach["kin"].fk(q_grasp)[:3, 3], [x, y, z], atol=1e-3)
        exempt = dict(exempt_xy=g.position[:2], exempt_radius_m=reach["exempt_radius_m"],
                      exempt_z_min=reach["table_z"] - 0.06)
        assert reach["harness"].vet_pose(q_pre) is None
        assert reach["harness"].vet_pose(q_grasp, **exempt) is None
        # the shipped box refuses the very same joint solution
        refusals = [default["harness"].vet_pose(q_pre), default["harness"].vet_pose(q_grasp, **exempt)]
        assert any(r and "outside workspace" in r for r in refusals), (x, y, z, family, refusals)


@needs_pin
@pytest.mark.parametrize("x, y, z", [(0.575, 0.0, 0.07), (0.30, 0.525, 0.07), (0.30, -0.525, 0.05)])
def test_points_beyond_the_envelope_still_refuse_although_ik_reaches_them(x, y, z):
    """The study found these reachable with the box open; the reach profile's
    box (the new limit) refuses every family and roll there."""
    from cascade.grasping.selector import NoExecutableGrasp, select_grasp

    study = _study()
    table = study.rows_to_table(study.rows_from_json(_evidence()["passes"]["open"]["rows"]))
    assert any(table[study._key(x, y, z)].values()), "premise: reachable with the box open"
    reach = study.build_context("rebot_rs_reach")
    for family in study.FAMILY_NAMES:
        for roll in study.ROLLS_DEG:
            for sign in (1, -1):
                g = study.candidate(reach, x, y, z, family, roll, sign)
                with pytest.raises(NoExecutableGrasp) as exc:
                    select_grasp([g], reach["kin"], reach["home_q"], max_width_m=0.09,
                                 pregrasp_offset_m=reach["pregrasp_offset_m"],
                                 validate=study.runtime_vet(reach), preserve_order=True)
                assert "IK failed" in str(exc.value) or "outside workspace" in str(exc.value)


# ── the analytic planner ──────────────────────────────────────────────────────


def _box_fix(centre=(0.30, 0.10, 0.04), size=(0.05, 0.07, 0.08), yaw=0.3) -> ObjectFix:
    rng = np.random.default_rng(0)
    c, half = np.asarray(centre, dtype=float), np.asarray(size, dtype=float) / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    pts = c + rng.uniform(-half, half, size=(600, 3)) @ R.T
    order = np.argsort(-np.asarray(size))
    return ObjectFix(label="cube", position=c, points=pts,
                     detection=Detection(label="cube", conf=0.9, bbox=np.zeros(4)),
                     extent=np.asarray(size)[order], axes=R[:, order])


def test_golden_obb_planner_without_tilts_is_the_topdown_planner():
    """Golden (true on main): 2 yaws x (yaw, yaw + pi), all straight down,
    qualities 0.9 x (1, 0.99, 0.9, 0.89)."""
    from cascade.grasping.obb_grasp import plan_grasps_from_fix

    grasps = plan_grasps_from_fix(_box_fix(), table_z=0.0, depth_fraction=0.15)
    assert len(grasps) == 4
    for g in grasps:
        np.testing.assert_allclose(g.approach, [0.0, 0.0, -1.0])
        np.testing.assert_allclose(g.rotation[:, 0], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose([g.quality for g in grasps], [0.9, 0.891, 0.81, 0.801])
    np.testing.assert_allclose([g.width_m for g in grasps], [0.065, 0.065, 0.085, 0.085])


def test_empty_tilts_are_byte_identical_to_no_tilts():
    from cascade.grasping.obb_grasp import plan_grasps_from_fix

    base = plan_grasps_from_fix(_box_fix(), table_z=0.0, depth_fraction=0.15)
    for tilts in ((), [], None):
        other = plan_grasps_from_fix(_box_fix(), table_z=0.0, depth_fraction=0.15,
                                     angled_tilts_deg=tilts)
        assert len(other) == len(base)
        for a, b in zip(base, other):
            assert a.position.tobytes() == b.position.tobytes()
            assert a.rotation.tobytes() == b.rotation.tobytes()
            assert (a.quality, a.width_m, a.label) == (b.quality, b.width_m, b.label)


@pytest.mark.parametrize("axis_order", ["down_open", "open_down", "third_open_down"])
def test_tilted_candidates_lean_away_from_the_base_with_the_topdown_jaw_axis(axis_order):
    from cascade.grasping.obb_grasp import plan_grasps_from_fix

    fix = _box_fix()
    top = plan_grasps_from_fix(fix, table_z=0.0, depth_fraction=0.15, axis_order=axis_order)
    grasps = plan_grasps_from_fix(fix, table_z=0.0, depth_fraction=0.15, axis_order=axis_order,
                                  angled_tilts_deg=ENVELOPE_TILTS)
    assert len(grasps) == 4 + 2 * 3 * 2
    for a, b in zip(top, grasps[:4]):  # top-down candidates first, unchanged
        assert a.rotation.tobytes() == b.rotation.tobytes() and a.quality == b.quality
    col = {"down_open": (0, 1), "open_down": (2, 0), "third_open_down": (2, 1)}[axis_order]
    tilted = grasps[4:]
    assert max(g.quality for g in tilted) < min(g.quality for g in grasps[:4])
    radial = fix.position[:2] / np.linalg.norm(fix.position[:2])
    jaws = [g.rotation[:, col[1]] for g in top]
    for i, g in enumerate(tilted):
        rank, rest = divmod(i, 6)
        tilt = ENVELOPE_TILTS[rest // 2]
        R = g.rotation
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(R[:, col[0]], g.approach, atol=1e-12)
        angle = np.degrees(np.arccos(np.clip(-g.approach[2], -1, 1)))
        assert angle == pytest.approx(tilt)
        lean = g.approach[:2]
        assert lean @ radial > -1e-9 and (tilt == 0 or np.linalg.norm(lean) > 0.4)
        jaw = R[:, col[1]]
        assert abs(jaw[2]) < 1e-12 and abs(jaw @ g.approach) < 1e-12
        assert abs(abs(jaw @ jaws[2 * rank]) - 1.0) < 1e-12  # same closing axis as top-down
        np.testing.assert_allclose(g.position, top[2 * rank].position)
        assert g.width_m == top[2 * rank].width_m
    assert all(g.quality > 0 for g in tilted)
    # tilt order is preserved in the ranking, smaller tilts first, then the twin
    q = [g.quality for g in tilted[:6]]
    assert all(a > b for a, b in zip(q, q[1:])), q
    for i in range(0, len(tilted), 2):  # each pair: the same grasp, jaws swapped
        a, b = tilted[i].rotation[:, col[1]], tilted[i + 1].rotation[:, col[1]]
        np.testing.assert_allclose(a, -b, atol=1e-12)
        np.testing.assert_allclose(tilted[i].approach, tilted[i + 1].approach, atol=1e-12)


def test_tilted_candidate_quality_is_half_its_footprint_candidates_including_infeasible_ones():
    """quality = (1 if the jaw fits else 0.2) x (1 - 0.1 rank) x conf x 0.5
    x (1 - 0.01 tilt index) x (1 - 0.001 twin): a too-wide footprint stays
    as unattractive tilted as top-down (the selector refuses it on width)."""
    from cascade.grasping.obb_grasp import plan_grasps_from_fix

    fix = _box_fix(size=(0.05, 0.085, 0.08))  # 0.085 + 0.015 pad > 0.09 jaw
    grasps = plan_grasps_from_fix(fix, table_z=0.0, depth_fraction=0.15,
                                  angled_tilts_deg=[30, 45])
    assert [g.width_m for g in grasps[:4]] == pytest.approx([0.065, 0.065, 0.1, 0.1])
    expected = [f * (1 - 0.1 * rank) * 0.9 * 0.5 * (1 - 0.01 * k) * (1 - 0.001 * sub)
                for rank, f in enumerate((1.0, 0.2)) for k in range(2) for sub in range(2)]
    assert [g.quality for g in grasps[4:]] == pytest.approx(expected, rel=1e-12)
    assert [g.width_m for g in grasps[4:]] == pytest.approx([0.065] * 4 + [0.1] * 4)


def test_a_horizontal_tilt_is_a_side_approach_toward_plus_radial():
    from cascade.grasping.obb_grasp import plan_grasps_from_fix

    fix = _box_fix(centre=(0.45, 0.0, 0.04), yaw=0.0)
    grasps = plan_grasps_from_fix(fix, table_z=0.0, depth_fraction=0.15, angled_tilts_deg=[90])
    side = grasps[4:]
    # the jaw closes across y for one footprint axis (lean +x) and across x
    # for the other (lean +-y: perpendicular to the radial line, either side)
    assert any(np.allclose(g.approach, [1.0, 0.0, 0.0], atol=1e-12) for g in side)
    assert all(abs(g.approach[2]) < 1e-12 for g in side)


@pytest.mark.parametrize("bad", [[0], [-15], [91], [float("nan")], ["x"], [True], 30, "30"],
                         ids=["zero", "negative", "above-90", "nan", "text", "bool", "scalar", "string"])
def test_malformed_tilts_fail_closed(bad):
    from cascade.grasping.obb_grasp import plan_grasps_from_fix

    with pytest.raises(ValueError, match="angled_approach_tilts_deg"):
        plan_grasps_from_fix(_box_fix(), table_z=0.0, angled_tilts_deg=bad)


# ── behaviour: the real runtime on the mock stack ─────────────────────────────


#: mock box 8 cm along base x, 6 cm along y: the only jaw that fits closes
#: across y (tangential), so the planner's tilt leans radially OUT
TANGENTIAL_JAW_BOX_PX = [290, 190, 350, 270]


def _mock_runtime(tmp_path, profile, box_x, box_px=None):
    """Mock stack with the red box moved to base x ~= box_x (the mock camera
    looks straight down; shifting its extrinsic moves the scene)."""
    from cascade.apps.demo import build_runtime

    cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
    cfg._data["grasp"]["backend"] = "obb"
    for cam in [cfg._data["camera"], *(cfg._data.get("cameras") or [])]:
        cam["extrinsics"]["T"][0][3] = 0.28 + (box_x - 0.29)
        if box_px is not None:
            cam["box_px"] = list(box_px)
    return build_runtime(cfg, tmp_path / f"run-{profile}")


def _grasp(tmp_path, monkeypatch, profile, box_x, box_px=TANGENTIAL_JAW_BOX_PX):
    import cascade.skills.runtime as rtmod
    from cascade.apps.demo import shutdown_runtime

    planned, chosen = [], []
    real_plan, real_select = rtmod.plan_grasps_from_fix, rtmod.select_grasp

    def spy_plan(fix, **kw):
        planned.append(kw)
        return real_plan(fix, **kw)

    def spy_select(grasps, *a, **kw):
        out = real_select(grasps, *a, **kw)
        chosen.append(out[0])
        return out

    monkeypatch.setattr(rtmod, "plan_grasps_from_fix", spy_plan)
    monkeypatch.setattr(rtmod, "select_grasp", spy_select)
    runtime, arm = _mock_runtime(tmp_path, profile, box_x, box_px)
    try:
        arm.object_stop_frac = 0.5
        q0 = np.asarray(arm.get_state().q, dtype=float).copy()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        with runtime.watcher.paused():
            result = runtime.execute("grasp_object", {"label": "red cube"})
        moved = not np.allclose(np.asarray(arm.get_state().q, dtype=float), q0)
        return result, planned, chosen, moved
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_mock_stack_default_profile_refuses_a_box_beyond_the_shipped_box_without_moving(
        tmp_path, monkeypatch):
    result, planned, chosen, moved = _grasp(tmp_path, monkeypatch, "mock", 0.535)
    assert result["ok"] is False
    assert "outside the active arm workspace" in str(result), result
    assert chosen == [] and moved is False


@needs_pin
def test_mock_stack_reach_profile_grasps_the_same_box_with_a_tilted_approach(tmp_path, monkeypatch):
    result, planned, chosen, moved = _grasp(tmp_path, monkeypatch, "mock_reach", 0.535)
    assert result["ok"] is True, result
    assert planned and all(kw.get("angled_tilts_deg") == ENVELOPE_TILTS for kw in planned)
    g = chosen[-1]
    assert g.position[0] > 0.50  # beyond the shipped box
    assert g.approach[2] > -0.99 and g.approach[:2] @ g.position[:2] > 0  # tilted, leaning out
    assert moved


@needs_pin
def test_mock_stack_reach_profile_refuses_a_box_it_can_only_close_on_radially(tmp_path, monkeypatch):
    """The documented limit of the analytic tilts: the planner leans
    perpendicular to the jaw, so a box whose only fitting jaw axis is RADIAL
    (narrow side facing the robot) gets a sideways lean, which the study
    measured to add no reach. Beyond top-down reach it is refused before any
    motion -- never forced."""
    result, planned, chosen, moved = _grasp(tmp_path, monkeypatch, "mock_reach", 0.535,
                                            box_px=(280, 200, 360, 260))
    assert result["ok"] is False and "no executable grasp" in result["error"], result
    assert chosen == [] and moved is False


@needs_pin
def test_mock_stack_reach_profile_keeps_topdown_first_where_topdown_reaches(tmp_path, monkeypatch):
    result, planned, chosen, _ = _grasp(tmp_path, monkeypatch, "mock_reach", 0.29)
    assert result["ok"] is True, result
    np.testing.assert_allclose(chosen[-1].approach, [0.0, 0.0, -1.0], atol=1e-12)


@needs_pin
def test_mock_stack_default_profile_plans_without_the_tilt_argument(tmp_path, monkeypatch):
    """Golden call shape: the default profile's planner call is main's."""
    result, planned, chosen, _ = _grasp(tmp_path, monkeypatch, "mock", 0.29)
    assert result["ok"] is True, result
    assert planned and all("angled_tilts_deg" not in kw for kw in planned)
    np.testing.assert_allclose(chosen[-1].approach, [0.0, 0.0, -1.0], atol=1e-12)


@needs_pin
def test_the_studys_vet_is_the_runtimes_vet(tmp_path, monkeypatch):
    """The study's `runtime_vet` must give the reasons the runtime's own
    `_vet` gives, on the same candidates and joint solutions: the validate
    callback the real `grasp_object` hands `select_grasp` is captured and
    called (inside its planning budget) next to the study's replica, under
    the same profile (mock_reach)."""
    import cascade.skills.runtime as rtmod

    study = _study()
    ctx = study.build_context("mock_reach")
    from cascade.types import make_transform

    cases = []
    for x, y, z, family in [(0.30, 0.0, 0.07, "topdown"), (0.535, 0.0, 0.05, "out30"),
                            (0.575, 0.0, 0.07, "out45"), (0.60, 0.0, 0.07, "side"),
                            (0.30, 0.525, 0.07, "out30")]:
        g = study.candidate(ctx, x, y, z, family, 0.0)
        pre = ctx["kin"].ik(make_transform(g.rotation, g.position - g.approach * ctx["pregrasp_offset_m"]),
                            ctx["home_q"])
        grasp = ctx["kin"].ik(make_transform(g.rotation, g.position), pre.q)
        assert pre.success and grasp.success, (x, y, z, family)
        cases.append((g, pre.q, grasp.q))
    ours = [study.runtime_vet(ctx)(*c) for c in cases]
    theirs = []
    real_select = rtmod.select_grasp

    def spy_select(grasps, *a, **kw):
        if not theirs:
            theirs.extend(kw["validate"](*c) for c in cases)
        return real_select(grasps, *a, **kw)

    monkeypatch.setattr(rtmod, "select_grasp", spy_select)
    result, _, _, _ = _grasp(tmp_path, monkeypatch, "mock_reach", 0.29)
    assert result["ok"] is True, result
    assert theirs == ours
    kinds = [None if r is None else r.split(":")[0] for r in ours]
    assert kinds == [None, None, "descent unsafe", "pregrasp unsafe", "pregrasp unsafe"], ours


@needs_pin
def test_re_evaluating_grid_points_reproduces_the_committed_masks():
    """The committed rows are what the script computes: four points with
    distinctive masks (angled-only, out-but-not-diag, partial rolls, the
    side ring) re-evaluated through the study's pipeline, box opened."""
    study = _study()
    data = _evidence()
    table = study.rows_to_table(study.rows_from_json(data["passes"]["open"]["rows"]))
    study._worker_init("rebot_rs", "open")
    for point in [(0.55, 0.0, 0.07), (0.0, -0.6, 0.02), (0.275, 0.3, 0.05), (0.0, -0.7, 0.02)]:
        committed = table.get(study._key(*point), {f: 0 for f in study.FAMILY_NAMES})
        x, y, z, fam = study._point_task((*point, list(study.FAMILY_NAMES)))
        assert {f: v[0] for f, v in fam.items()} == committed, point


#: Where the parent's live Isaac A/B places the bare scene's 5x5x8 cm pink
#: cube (axis-aligned, `place_prop`): one inside the shipped box but beyond
#: top-down reach, five in the newly admitted region (docs/REACH_ENVELOPE.md).
LIVE_POSITIONS = [(0.47, 0.0), (0.53, 0.0), (0.53, 0.25), (0.35, 0.40), (0.20, -0.42), (0.45, -0.40)]


@needs_pin
@pytest.mark.parametrize("x, y", LIVE_POSITIONS)
def test_live_validation_positions_are_reachable_offline_only_with_the_reach_profile(x, y):
    """Offline pre-check of the live A/B: the analytic planner's candidates
    for the cube pass width, IK and the harness vet under isaac_reach with a
    tilted approach, and are refused under isaac (top-down, shipped box)."""
    from cascade.grasping.obb_grasp import plan_grasps_from_fix
    from cascade.grasping.selector import NoExecutableGrasp, select_grasp

    fix = _box_fix(centre=(x, y, 0.04), size=(0.05, 0.05, 0.08), yaw=0.0)
    for profile, expect in (("isaac_reach", True), ("isaac", False)):
        view, a, kin, harness = _rig(profile)
        home = np.asarray(a.home_q, dtype=float)
        tilts = view.grasp.get("angled_approach_tilts_deg")
        grasps = plan_grasps_from_fix(fix, table_z=0.0, depth_fraction=view.grasp.depth_fraction,
                                      axis_order=a.get("tool_axis_order", "down_open"),
                                      **({"angled_tilts_deg": tilts} if tilts else {}))
        args = dict(max_width_m=0.09, pregrasp_offset_m=view.grasp.pregrasp_offset_m,
                    validate=_vet(view, home, harness), preserve_order=True)
        ordered = sorted(grasps, key=lambda g: -g.quality)
        if expect:
            g, _, _ = select_grasp(ordered, kin, home, **args)
            assert g.approach[2] > -0.99, (profile, x, y)
        else:
            with pytest.raises(NoExecutableGrasp):
                select_grasp(ordered, kin, home, **args)
