"""Two SO-101s in ONE MuJoCo world: the inter-arm gate measured against physics.

ROADMAP item 8 asked for the gate to be *measured, not simulated*. These tests
attach two prefixed copies of the Menagerie SO-101 at the shipped `base_pose`
of `so101_left` / `so101_right`, drive the real `SafetyHarness` chain through
Pinocchio on the same poses, and compare the gate's verdict with MuJoCo's
collision geometry (`mj_geomDistance` over every collision-geom pair).

What they pin:

* the harness chain IS the physics chain (joint origins coincide to 1e-6 m),
  so a disagreement can only come from link SHAPE, never from kinematics;
* the per-segment `link_radii_m` the profile declares envelope every
  collision geom of that segment, at every pose and every gripper opening;
* therefore the gate's surface clearance is a LOWER BOUND on the physical
  clearance: every approved pose pair is at least `neighbor_clearance_m`
  apart in MuJoCo, over thousands of random pairs;
* and the measured reason the radii exist: a centreline gate approves pairs
  whose meshes overlap.

Needs `mujoco`, `pinocchio`, the fetched MJCF (`scripts/fetch_robot_assets.py
so101`) and the URDF; skips otherwise, like the other MuJoCo tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO, SO101_URDF, has_pinocchio

MJCF = REPO / "assets" / "mjcf" / "so101" / "so101.xml"


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


needs_rig = pytest.mark.skipif(
    not (_has_mujoco() and MJCF.exists() and has_pinocchio() and SO101_URDF.exists()),
    reason="needs mujoco + pinocchio + `python scripts/fetch_robot_assets.py so101` (MJCF and URDF)",
)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
#: MJCF body -> harness chain segment (consecutive joint origins; the last
#: segment runs from the gripper joint to the TCP). The base body carries no
#: collision geometry and is not part of the chain.
SEGMENT_OF_BODY = {
    "shoulder": 0, "upper_arm": 1, "lower_arm": 2, "wrist": 3,
    "gripper": 4, "camera_mount": 4, "moving_jaw_so101_v1": 5,
}
_BOX_CORNERS = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    import mujoco

    from cascade.config import Cfg, load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.safety.harness import SafetyHarness, SafetyLimits
    from cascade.sim.demo_scene import multi_arm_scene_xml
    from cascade.types import pose_to_transform

    cfg = load_demo_config(arms=["so101_left", "so101_right"], camera="mock", llm="mock")
    names = [a["name"] for a in cfg.arms]
    prefixes = [n + "/" for n in names]
    xml = multi_arm_scene_xml(MJCF, [(p, a["base_pose"]) for p, a in zip(prefixes, cfg.arms)])
    scene = tmp_path_factory.mktemp("two_arm") / "two_so101.xml"
    scene.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    arms = []
    for name, prefix, acfg in zip(names, prefixes, cfg.arms):
        kin = Kinematics(
            model_path=acfg["model"], ee_frame=acfg.get("ee_frame", "gripper_end"),
            n_controlled=int(acfg.get("n_joints", 5)), joint_signs=acfg.get("joint_signs"),
        )
        limits = SafetyLimits.from_config(Cfg(acfg["resolved"]).safety)
        harness = SafetyHarness(limits, kinematics=kin, base_pose=pose_to_transform(acfg["base_pose"]))
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + j) for j in ARM_JOINTS]
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "gripper")
        assert min(jid) >= 0 and gid >= 0, f"{prefix}: joints not found"
        geoms = []
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if not body.startswith(prefix):
                continue
            local = body[len(prefix):]
            if local == "base":
                continue
            geoms.append((g, SEGMENT_OF_BODY[local]))
        arms.append(SimpleNamespace(
            name=name, prefix=prefix, cfg=acfg, kin=kin, harness=harness, jid=jid, gid=gid,
            geoms=geoms, home=np.asarray(acfg["home_q"], dtype=float)[:5],
            lo=m.jnt_range[jid, 0].copy(), hi=m.jnt_range[jid, 1].copy(),
            grip_range=m.jnt_range[gid].copy(),
        ))
    return SimpleNamespace(m=m, d=d, cfg=cfg, arms=arms, scene=scene, xml=xml)


def _set_q(rig, arm, q, grip=0.6):
    for j, v in zip(arm.jid, q):
        rig.d.qpos[rig.m.jnt_qposadr[j]] = v
    rig.d.qpos[rig.m.jnt_qposadr[arm.gid]] = grip


def _physics_clearance(rig, a, b, limit=0.5) -> float:
    """Smallest MuJoCo distance between any collision geom of `a` and of `b`
    (negative = penetration), bounding-sphere broadphase then mj_geomDistance."""
    import mujoco

    m, d = rig.m, rig.d
    fromto = np.zeros(6)
    best = limit
    for gi, _ in a.geoms:
        for gj, _ in b.geoms:
            lower = np.linalg.norm(d.geom_xpos[gi] - d.geom_xpos[gj]) - m.geom_rbound[gi] - m.geom_rbound[gj]
            if lower < best:
                best = min(best, mujoco.mj_geomDistance(m, d, gi, gj, best, fromto))
    return float(best)


def _extreme_points(rig, g):
    """Points whose farthest distance from a segment bounds the whole geom:
    box corners, mesh vertices, capsule end centres (+radius), sphere centre
    (+radius). Distance-to-segment is convex, so its maximum over a convex
    body is at a vertex."""
    m, d = rig.m, rig.d
    t = int(m.geom_type[g])
    c, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    if t == 6:  # box
        return c + (_BOX_CORNERS * s) @ R.T, 0.0
    if t == 7:  # mesh
        mid = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return c + V @ R.T, 0.0
    if t == 3:  # capsule
        return c + np.array([[0, 0, -s[1]], [0, 0, s[1]]]) @ R.T, float(s[0])
    if t == 2:  # sphere
        return c[None, :], float(s[0])
    raise AssertionError(f"collision geom {g} has unexpected type {t}")


def _point_segment_distances(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


@needs_rig
def test_two_arm_scene_attaches_prefixed_copies_at_the_profile_base_poses(rig):
    import mujoco

    m, d = rig.m, rig.d
    assert m.nq == 12 and m.nu == 12
    for arm in rig.arms:
        joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        assert [arm.prefix + j for j in ARM_JOINTS + ["gripper"]] == [j for j in joints if j.startswith(arm.prefix)]
        mujoco.mj_forward(m, d)
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, arm.prefix + "base")
        assert d.xpos[base] == pytest.approx(np.asarray(arm.cfg["base_pose"][:3], dtype=float), abs=1e-9)
    # physics options come from the ROBOT file, not MuJoCo's defaults: attach
    # keeps the parent's <option>, so the scene must restate the child's
    robot = mujoco.MjModel.from_xml_path(str(MJCF))
    assert (m.opt.timestep, m.opt.integrator, m.opt.cone, m.opt.impratio) == (
        robot.opt.timestep, robot.opt.integrator, robot.opt.cone, robot.opt.impratio)
    # thirty collision geoms per arm, six chain segments, none on the base
    for arm in rig.arms:
        assert len(arm.geoms) == 30
        assert sorted({seg for _, seg in arm.geoms}) == [0, 1, 2, 3, 4, 5]


@needs_rig
def test_two_arm_scene_refuses_ambiguous_prefixes():
    from cascade.sim.demo_scene import multi_arm_scene_xml

    with pytest.raises(ValueError, match="prefix"):
        multi_arm_scene_xml(MJCF, [("a/", [0, 0.2, 0, 0, 0, 0]), ("a/", [0, -0.2, 0, 0, 0, 0])])
    with pytest.raises(ValueError, match="prefix"):
        multi_arm_scene_xml(MJCF, [("", [0, 0.2, 0, 0, 0, 0]), ("b/", [0, -0.2, 0, 0, 0, 0])])
    with pytest.raises(ValueError, match="base_pose"):
        multi_arm_scene_xml(MJCF, [("a/", [0, 0.2, 0]), ("b/", [0, -0.2, 0, 0, 0, 0])])


@needs_rig
def test_harness_chain_matches_mujoco_joint_anchors_in_the_table_frame(rig):
    """The gate's segments run between the SAME points MuJoCo articulates
    about, in the SAME (table) frame -- otherwise any agreement below would be
    a coincidence of this base_pose."""
    import mujoco

    rng = np.random.default_rng(11)
    worst = 0.0
    for arm in rig.arms:
        for _ in range(50):
            q = rng.uniform(arm.lo, arm.hi)
            _set_q(rig, arm, q)
            mujoco.mj_forward(rig.m, rig.d)
            pts = arm.harness.link_points_table_frame(q)
            assert pts.shape == (7, 3)  # 5 arm joints + gripper joint + TCP
            anchors = rig.d.xanchor[arm.jid + [arm.gid]]
            worst = max(worst, float(np.max(np.linalg.norm(pts[:6] - anchors, axis=1))))
    # URDF and MJCF are two exports of one CAD model; they agree to microns
    assert worst < 1e-5, f"harness chain and MuJoCo joint anchors differ by {worst:.2e} m"


@needs_rig
def test_declared_link_radii_envelope_the_collision_geometry(rig):
    """Every collision geom of segment k lies within `link_radii_m[k]` of that
    segment, at any pose and any gripper opening -- and within 1 cm of it, so
    the radii track the geometry instead of padding the gate shut."""
    import mujoco

    rng = np.random.default_rng(5)
    for arm in rig.arms:
        radii = np.asarray(arm.harness.limits.link_radii_m, dtype=float)
        assert radii.shape == (6,)
        extent = np.zeros(6)
        openings = np.concatenate([np.linspace(*arm.grip_range, 33), rng.uniform(*arm.grip_range, 67)])
        for k in range(100):
            q = rng.uniform(arm.lo, arm.hi)
            _set_q(rig, arm, q, grip=openings[k])
            mujoco.mj_forward(rig.m, rig.d)
            pts = arm.harness.link_points_table_frame(q)
            for g, seg in arm.geoms:
                P, r = _extreme_points(rig, g)
                extent[seg] = max(extent[seg], float(_point_segment_distances(P, pts[seg], pts[seg + 1]).max()) + r)
        slack = radii - extent
        assert np.all(slack >= 0.0), f"{arm.name}: declared radii {radii} do not cover measured extents {extent}"
        assert np.all(slack <= 0.01), f"{arm.name}: declared radii {radii} overshoot measured extents {extent}"


def _wire(rig, a, b):
    """Point a's harness at b exactly the way demo._wire_neighbors does."""
    state = {"q": b.home.copy()}
    a.harness._neighbors.clear()
    a.harness._neighbor_radii.clear()
    a.harness.add_neighbor(b.name, lambda: b.harness.link_points_table_frame(state["q"]),
                           link_radii_m=b.harness.limits.link_radii_m)
    return state


@needs_rig
def test_gate_is_a_lower_bound_on_mujoco_clearance_over_random_pose_pairs(rig):
    """The claim: every pose pair the gate approves is at least
    `neighbor_clearance_m` apart in MuJoCo, and the gate is not vacuous."""
    import mujoco

    L, R = rig.arms
    tol = float(L.harness.limits.neighbor_clearance_m)
    assert tol == pytest.approx(0.03)
    other = _wire(rig, L, R)
    rng = np.random.default_rng(2026)
    rows = []
    for _ in range(2000):
        ql, qr = rng.uniform(L.lo, L.hi), rng.uniform(R.lo, R.hi)
        gl, gr = rng.uniform(*L.grip_range), rng.uniform(*R.grip_range)
        _set_q(rig, L, ql, gl)
        _set_q(rig, R, qr, gr)
        mujoco.mj_forward(rig.m, rig.d)
        other["q"] = qr
        approved = L.harness._neighbor_violation(ql) is None
        rows.append((approved, _physics_clearance(rig, L, R)))
    approved = np.array([r[0] for r in rows])
    physics = np.array([r[1] for r in rows])
    assert physics[approved].min() >= tol, (
        f"approved a pair only {physics[approved].min():.4f} m apart in MuJoCo (gate {tol} m)")
    assert (physics[approved] <= 0.0).sum() == 0
    # not vacuous: most of joint space is still usable, and the rejections
    # include real near misses
    assert approved.mean() > 0.7, f"gate approves only {approved.mean():.1%} of random pose pairs"
    assert (physics[~approved] < tol).sum() > 50
    print(f"\nINTER_ARM_GATE_VS_PHYSICS approves={approved.mean():.1%} "
          f"min_physics_among_approved={physics[approved].min():.4f} m "
          f"rejected_with_physics_below_tol={(physics[~approved] < tol).sum()} "
          f"rejected_total={(~approved).sum()} of {len(rows)}")


@needs_rig
def test_centreline_gate_approves_physical_overlap_which_is_why_radii_exist(rig):
    """The measured hazard this change removes: the same gate with zero-radius
    links approves pose pairs whose collision meshes already overlap. Kept as
    a measurement, not a wish -- it must stay true for the radii to be the
    right fix rather than a bigger centreline margin."""
    import mujoco

    L, R = rig.arms
    other = _wire(rig, L, R)
    saved = (L.harness.limits.link_radii_m, dict(L.harness._neighbor_radii))
    try:
        L.harness.limits.link_radii_m = None
        L.harness._neighbor_radii[R.name] = None
        rng = np.random.default_rng(2026)
        overlapping = 0
        for _ in range(2000):
            ql, qr = rng.uniform(L.lo, L.hi), rng.uniform(R.lo, R.hi)
            _set_q(rig, L, ql)
            _set_q(rig, R, qr)
            mujoco.mj_forward(rig.m, rig.d)
            other["q"] = qr
            if L.harness._neighbor_violation(ql) is None and _physics_clearance(rig, L, R) <= 0.0:
                overlapping += 1
    finally:
        L.harness.limits.link_radii_m, L.harness._neighbor_radii = saved
    assert overlapping >= 1, "the centreline gate no longer approves overlaps: re-measure the radii"
    print(f"\nCENTRELINE_GATE_OVERLAPS approved_but_overlapping={overlapping} of 2000")


@needs_rig
def test_shipped_dual_arm_scenario_matches_physics(rig):
    """The two poses test_arm_rig's rig test drives: extended-and-yawed-inward
    against a neighbour at home is approved AND physically clear; against the
    neighbour also swung inward it is refused AND physically overlapping."""
    import mujoco

    L, R = rig.arms
    other = _wire(rig, L, R)
    inward_left, inward_right = L.home.copy(), R.home.copy()
    inward_left[:3] = [1.57, 0.0, 0.0]
    inward_right[:3] = [-1.57, 0.0, 0.0]

    _set_q(rig, L, inward_left)
    _set_q(rig, R, R.home)
    mujoco.mj_forward(rig.m, rig.d)
    other["q"] = R.home
    assert L.harness._neighbor_violation(inward_left) is None
    assert _physics_clearance(rig, L, R) > 0.10
    # the mirror image is the tighter one: 0.032 m of surface clearance for
    # 0.13 m in physics -- the envelope is loose where the slim wrist faces
    # the neighbour, which is why the margin is 0.03 and not 0.05
    _set_q(rig, L, L.home)
    _set_q(rig, R, inward_right)
    mujoco.mj_forward(rig.m, rig.d)
    mirror = _wire(rig, R, L)
    mirror["q"] = L.home
    assert R.harness._neighbor_violation(inward_right) is None
    assert _physics_clearance(rig, L, R) > 0.10
    other = _wire(rig, L, R)
    _set_q(rig, L, inward_left)

    _set_q(rig, R, inward_right)
    mujoco.mj_forward(rig.m, rig.d)
    other["q"] = inward_right
    reason = L.harness._neighbor_violation(inward_left)
    assert reason is not None and "inter-arm clearance" in reason
    assert _physics_clearance(rig, L, R) < 0.0, "the shipped 'collision' pose should really collide"
