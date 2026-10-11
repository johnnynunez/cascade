"""The wrist camera and the jaws are clearance samples, not just TCP + joints.

Rig incident (2026-10-11): grasp_object(green object) with a SIDE approach
(object at [0.28, 0.129, 0.043]). The wrist and the padded wrist D455F (on
top of the gripper: 8 cm up, 14 cm back from the fingertips, 15 cm across the
jaws) swept into clutter beside the target -- a blue object, a camera mount,
cables, a power-supply box -- and pressed ~8 s until "did not settle at grasp
pose". The harness sampled only the TCP and the joint origins; the camera box
and the jaws were invisible to the occupancy check even with a map.

These tests reproduce that geometry offline on the reBot's URDF and a
synthetic distance grid (tests/occupancy_grid_bridge.py).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import ToolVolume, build_tool_volume

OBJ = np.array([0.28, 0.129, 0.043])
#: the target itself as the map holds it: ~5 x 5 x 8.6 cm standing on the table
TARGET = centred_box([OBJ[0], OBJ[1], 0.043], [0.05, 0.05, 0.086])
DESCENT_EXEMPT = {"exempt_xy": OBJ[:2], "exempt_radius_m": 0.07, "exempt_z_min": -0.06}
T_CAM = np.array([[0.0, 0.0, -1.0, 1.2], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.3], [0, 0, 0, 1.0]])


def _rz(t):
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])


def _ry(t):
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1.0, 0], [-np.sin(t), 0, np.cos(t)]])


@pytest.fixture(scope="module")
def rebot():
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    arm = cfg.arm
    kin = Kinematics(arm.model, arm.get("ee_frame", "gripper_end"), 6, joint_signs=arm.get("joint_signs"))
    home = np.asarray(arm.home_q, dtype=float)

    def solve(R, p):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        ik = kin.ik(T, home)
        assert ik.success, p
        return np.asarray(ik.q)

    # the side approach: yawed off the radial line, leaning 30 deg down, camera up
    R = _rz(np.arctan2(OBJ[1], OBJ[0]) + 1.0) @ _ry(np.radians(30))
    approach, up = R[:, 0], R[:, 2]
    jaw = np.cross(up, approach)
    side = jaw[:2] / np.linalg.norm(jaw[:2])
    return SimpleNamespace(cfg=cfg, kin=kin, home=home, R=R, approach=approach, side=side,
                           q_grasp=solve(R, OBJ), q_pre=solve(R, OBJ - approach * 0.12),
                           volume=build_tool_volume(cfg, cfg.arm))


def harness(rebot, boxes, *, volume=True):
    occ = make_map(GridBridge([TABLE, *boxes], voxel=0.01), unobserved="free")
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()
    return SafetyHarness(SafetyLimits.from_config(rebot.cfg.safety), kinematics=rebot.kin,
                         occupancy=occ, tool_volume=rebot.volume if volume else None)


def neighbour(rebot, gap_m=0.10, size=(0.05, 0.05, 0.09)):
    """The blue object: beside the target, across the jaws, `gap_m` centre to centre."""
    c = OBJ[:2] + rebot.side * gap_m
    return centred_box([c[0], c[1], size[2] / 2], size)


def descent(q_pre, q_grasp):
    return [q_pre + s * (q_grasp - q_pre) for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)]


# ── the sampled volume is the configured camera box plus the open jaws ──────


@needs_pin
def test_the_rebot_samples_the_configured_wrist_camera_box_and_the_open_jaws(rebot):
    vol = rebot.volume
    assert isinstance(vol, ToolVolume) and set(vol.names) == {"wrist camera", "jaws"}
    wc = rebot.cfg.relative_motion.wrist_camera
    cam = vol.local[vol.part == vol.names.index("wrist camera")]
    # TCP frame of the reBot (tool_axis_order down_open): x approach, y jaws, z camera up
    np.testing.assert_allclose(cam.min(axis=0), [-wc.behind_tip_m, -wc.half_width_m, 0.0], atol=1e-9)
    np.testing.assert_allclose(cam.max(axis=0), [0.0, wc.half_width_m, wc.protrusion_m], atol=1e-9)
    sweep = rebot.cfg.grasp.graspgenx.sweep
    jaws = vol.local[vol.part == vol.names.index("jaws")]
    ext, off = np.asarray(sweep.extents_open), np.asarray(sweep.offset_open)
    # sweep frame: +X closing (= TCP y), +Y third (= TCP z), +Z approach (= TCP x)
    np.testing.assert_allclose(jaws.min(axis=0), [off[2] - ext[2] / 2, off[0] - ext[0] / 2, off[1] - ext[1] / 2],
                               atol=1e-9)
    np.testing.assert_allclose(jaws.max(axis=0), [off[2] + ext[2] / 2, off[0] + ext[0] / 2, off[1] + ext[1] / 2],
                               atol=1e-9)
    spacing = float(rebot.cfg.safety.tool_volume.spacing_m)
    for part in (cam, jaws):
        for axis in range(3):
            steps = np.diff(np.unique(np.round(part[:, axis], 9)))
            assert steps.size == 0 or steps.max() <= spacing + 1e-9


@needs_pin
def test_the_harness_box_is_the_spatial_tools_box(rebot):
    """Two layers guard the camera (spatial tools plan with it; the harness
    vets every step with it): same config, same geometry code, same box."""
    from cascade.skills import relative_motion as rm

    wc = rebot.cfg.relative_motion.wrist_camera
    up = rebot.cfg.relative_motion.tool_up_axis
    approach_tcp = np.eye(3)[rm._TOOL_COLUMNS[str(rebot.cfg.arm.get("tool_axis_order", "down_open"))][0]]
    for q in (rebot.home, rebot.q_grasp, rebot.q_pre):
        T = rebot.kin.fk(q)
        box = rm.camera_box(T, approach_tcp, up, protrusion_m=wc.protrusion_m, behind_m=wc.behind_tip_m,
                            half_width_m=wc.half_width_m)
        samples = rebot.volume.points(T)[rebot.volume.part == rebot.volume.names.index("wrist camera")]
        local = (samples - box.center) @ box.axes
        assert np.all(np.abs(local) <= box.half + 1e-9)
        for corner in box.corners():
            assert np.min(np.linalg.norm(samples - corner, axis=1)) < 1e-9


def test_profiles_without_the_opt_in_sample_no_volume():
    from cascade.config import load_demo_config

    for arm in ("mock", "isaac", "so101_mock"):
        cfg = load_demo_config(arm=arm, camera="mock", llm="mock")
        assert build_tool_volume(cfg, cfg.arm) is None, arm


@pytest.mark.parametrize("block, why", [
    ({"wrist_camera": "yes"}, "wrist_camera"), ({"spacing_m": 0.0}, "spacing_m"),
    ({"spacing_m": float("nan")}, "spacing_m"), ({"table_band_m": -0.01}, "table_band_m"),
    ({"wrist_camera": False, "jaws": False}, "samples nothing"), ({"camera": True}, "unknown"),
])
def test_a_bad_tool_volume_block_is_refused_loudly(block, why):
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    cfg.safety._data["tool_volume"] = {"wrist_camera": True, "jaws": True, **block}
    with pytest.raises(ValueError, match=why):
        build_tool_volume(cfg, cfg.arm)


# ── the incident, offline ──────────────────────────────────────────────────


@needs_pin
def test_premise_the_old_proxies_never_saw_the_neighbour(rebot):
    """TCP + joint origins alone: the side grasp beside the blue object passes
    every vet the grasp pipeline runs -- the blind spot of the incident."""
    h = harness(rebot, [TARGET, neighbour(rebot)], volume=False)
    assert h.vet_pose(rebot.q_pre) is None
    assert all(h.vet_pose(q, **DESCENT_EXEMPT) is None for q in descent(rebot.q_pre, rebot.q_grasp))


@needs_pin
def test_incident_side_grasp_beside_a_neighbour_is_refused_before_any_motion(rebot):
    h = harness(rebot, [TARGET, neighbour(rebot)])
    reasons = [h.vet_pose(q, **DESCENT_EXEMPT) for q in descent(rebot.q_pre, rebot.q_grasp)]
    assert reasons[-1] is not None and "wrist camera" in reasons[-1] and "occupancy map" in reasons[-1]


@needs_pin
def test_a_tall_box_behind_the_wrist_blocks_the_camera_on_the_approach(rebot):
    """The power-supply box: behind and above the grasp, where the camera
    passes on the way in, far from the TCP and the joint origins."""
    vol = rebot.volume
    cam = vol.points(rebot.kin.fk(rebot.q_pre))[vol.part == vol.names.index("wrist camera")]
    top = cam[np.argmax(cam[:, 2])]
    psu = centred_box(top + np.array([0.0, 0.0, 0.035]), [0.04, 0.04, 0.03])
    assert harness(rebot, [TARGET, psu], volume=False).vet_pose(rebot.q_pre) is None
    reason = harness(rebot, [TARGET, psu]).vet_pose(rebot.q_pre)
    assert reason and "wrist camera" in reason


@needs_pin
def test_the_target_itself_may_still_be_approached(rebot):
    """The same side grasp without the neighbour is admitted, although the
    camera box and jaws come well within 3 cm of the target and the table:
    the target's own voxels (inside the descent cylinder) and the table plane
    are not obstacles to the gripper's volume."""
    h = harness(rebot, [TARGET])
    for q in descent(rebot.q_pre, rebot.q_grasp):
        assert h.vet_pose(q, **DESCENT_EXEMPT) is None
    samples = rebot.volume.points(rebot.kin.fk(rebot.q_grasp))
    assert np.nanmin(h.occupancy.clearance(samples)) < 0.01      # premise: they are that close


@needs_pin
def test_a_wide_cup_grasped_from_above_is_admitted(rebot):
    """Top-down onto a 9 cm cup: camera-box corners stick out of the 7 cm
    descent cylinder right next to the cup's wall."""
    from cascade.grasping.obb_grasp import _yaw_rotation

    xy = np.array([0.33, 0.0])
    cup = centred_box([xy[0], xy[1], 0.05], [0.09, 0.09, 0.10])
    R = _yaw_rotation(0.0, axis_order="down_open")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [xy[0], xy[1], 0.07]
    ik = rebot.kin.ik(T, rebot.home)
    assert ik.success
    h = harness(rebot, [cup])
    assert h.vet_pose(ik.q, exempt_xy=xy, exempt_radius_m=0.07, exempt_z_min=-0.06) is None


@needs_pin
def test_rest_to_home_on_a_clean_table_is_never_refused(rebot):
    from cascade.safety.trajectory import vet_segment

    h = harness(rebot, [])
    park = np.asarray(rebot.cfg.arm.park_q, dtype=float)
    assert vet_segment(h, park, rebot.home, 2.0) is None
    # back down to just above the joint-2/3 stops (the park itself relaxes
    # the joint margin, which vet_segment does not)
    assert vet_segment(h, rebot.home, park + np.array([0, 0.03, 0.03, 0, 0, 0]), 2.0) is None


# ── the obstacle model of the volume: attributed by OBSTACLE, not by sample ──


def test_clearance_excluding_drops_target_and_table_voxels_but_keeps_neighbours():
    target = centred_box([0.30, 0.0, 0.05], [0.06, 0.06, 0.10])
    beside = centred_box([0.40, 0.0, 0.05], [0.04, 0.04, 0.10])      # face at x = 0.38
    occ = make_map(GridBridge([TABLE, target, beside], voxel=0.01), unobserved="free")
    occ.refresh(depth_frame(), np.eye(4))
    cylinder = (np.array([0.30, 0.0]), 0.07, -0.06)                    # edge at x = 0.37
    inside_near_neighbour = [0.365, 0.0, 0.05]          # in the cylinder, 1.5 cm from the neighbour
    outside_near_target = [0.30, 0.075, 0.05]           # out of it, 4.5 cm from the target's wall
    above_table = [0.30, -0.25, 0.025]                  # 2.5 cm over the bare table
    pts = np.array([inside_near_neighbour, outside_near_target, above_table])
    raw = occ.clearance(pts)
    assert raw[0] < 0.03 and raw[2] < 0.03
    d = occ.clearance_excluding(pts, below_z=0.02, cylinder=cylinder, within=0.03)
    assert d[0] < 0.03                                  # a neighbour is a neighbour, wherever the sample is
    assert d[1] >= 0.03
    assert d[2] >= 0.03                                 # the table plane is the table rule's business


# ── a stopped arm may back away, never closer ──────────────────────────────


@needs_pin
def test_a_camera_already_too_close_may_back_away_but_never_close_in(rebot):
    """Stopped at the grasp pose with the camera box against the neighbour
    (or an object put down next to a resting arm): a step that moves it away
    is open, a hold too; one that closes in, or that enters the margin from
    outside, is refused."""
    path = descent(rebot.q_pre, rebot.q_grasp)
    h = harness(rebot, [TARGET, neighbour(rebot)])
    lateral = np.r_[rebot.side * 0.01, 0.0]

    def shifted(offset):
        T = rebot.kin.fk(rebot.q_grasp)
        T[:3, 3] += offset
        ik = rebot.kin.ik(T, rebot.q_grasp)
        assert ik.success
        return np.asarray(ik.q)

    assert "wrist camera" in (h.vet_pose(rebot.q_grasp, **DESCENT_EXEMPT) or "")
    with h.target_column_exemption((OBJ[:2], 0.07, -0.06)):
        assert h.vet_step(path[0], path[1], 1.0) is None
        assert "wrist camera" in (h.vet_step(path[1], path[2], 1.0) or "")       # entering: refused
        assert h.vet_step(rebot.q_grasp, shifted(-lateral), 1.0) is None          # backing away: open
        assert h.vet_step(rebot.q_grasp, rebot.q_grasp, 1.0) is None              # holding: open
        assert "wrist camera" in (h.vet_step(rebot.q_grasp, shifted(lateral), 1.0) or "")  # closer: refused
    # a static vet never applies the retreat rule
    assert "wrist camera" in (h.vet_pose(shifted(-lateral), **DESCENT_EXEMPT) or "")


@needs_pin
@pytest.mark.parametrize("policy", ["blocked", "free"])
def test_an_unobserved_camera_volume_follows_the_map_policy(rebot, policy):
    T = rebot.kin.fk(rebot.home)
    vol = rebot.volume
    cam = vol.points(T)[vol.part == vol.names.index("wrist camera")]
    bridge = GridBridge([TABLE], voxel=0.01)
    bridge.unknown = [(tuple(cam.min(axis=0) - 0.01), tuple(cam.max(axis=0) + 0.01))]
    occ = make_map(bridge, unobserved=policy)
    occ.refresh(depth_frame(), T_CAM)
    h = SafetyHarness(SafetyLimits.from_config(rebot.cfg.safety), kinematics=rebot.kin,
                      occupancy=occ, tool_volume=vol)
    # the TCP/joint check has its own policy test; isolate the volume here
    reason = h._tool_volume_violation(T, None)
    if policy == "blocked":
        assert reason and "wrist camera has unobserved clearance" in reason
    else:
        assert reason is None


@needs_pin
def test_build_arm_gives_the_rebot_harness_its_volume_and_the_mock_none(monkeypatch):
    from cascade.apps import demo
    from cascade.config import load_demo_config

    monkeypatch.setattr(demo, "make_arm", lambda *a, **k: pytest.fail("must not build an arm"))
    for arm, expected in (("rebot_rs", True), ("mock", False)):
        cfg = load_demo_config(arm=arm, camera="mock", llm="mock")
        raw, safe, _ = demo._build_arm(cfg.arm, True, None, cfg)
        assert not raw.connected
        assert (safe.harness.tool_volume is not None) is expected, arm
