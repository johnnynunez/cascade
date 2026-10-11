"""The reBot's live map end to end: rendered depth -> mesh body mask -> map -> harness.

Offline and without hardware: the depth a front camera would see (a table
plane, the reBot's own URDF collision meshes posed by Pinocchio at the
MockArm's joint state, the wrist D455F box, objects) is rendered here, goes
through the profile's mesh body mask inside OccupancyMap.refresh, into a fake
bridge that maps every unmasked pixel (tests/occupancy_grid_bridge.py
DepthBridge, 1 cm voxels like the rig's nvblox), and the SafetyHarness built
by `_build_arm` (tool volume included) vets against that map.

Why: the MCP server ran with CASCADE_OCCUPANCY=0 because the capsule mask
left the reBot's own jaws in the map and refused motions from rest. These
tests pin that the mesh mask keeps the robot usable while the map is on
(rest, home, the motion between them, a held object), and that the
2026-10-11 side grasp beside the blue object is refused before any motion.
"""

import itertools

import numpy as np
import pytest
from occupancy_grid_bridge import (
    DepthBridge,
    box_surface,
    make_map,
    render_depth,
    triangle_surface,
)
from test_harness_tool_volume import (
    DESCENT_EXEMPT,
    TARGET,
    descent,
    neighbour,
    side_grasp_geometry,
)
from test_occupancy_mesh_body_mask import H, K, W, _rebot_profile, look_at, needs_rs

from cascade.types import Frame

#: the front camera: in front of the arm, looking back at the work area
CAM = look_at((0.85, 0.10, 0.55), (0.25, 0.0, 0.10))
REST_OFFSET = np.array([0.0, 0.03, 0.03, 0.0, 0.0, 0.0])   # park relaxes the joint margin; a vet does not


@pytest.fixture(scope="module")
def meshes():
    from test_link_self_mask import _rs_truth_triangles

    return _rs_truth_triangles()


class Rig:
    def __init__(self, meshes, *, body_mask=None, voxel=0.01):
        from cascade.apps import demo

        override = {"type": "mock"} if body_mask is None else {"type": "mock", "body_mask": body_mask}
        self.cfg, self.acfg = _rebot_profile(**override)
        occ_cfg = self.cfg.occupancy
        self.bridge = DepthBridge(voxel=voxel)
        self.occ = make_map(self.bridge, required=bool(occ_cfg.required), unobserved=occ_cfg.unobserved)
        self.arm, self.safe, self.kin = demo._build_arm(self.acfg, False, self.occ, self.cfg)
        self.arm.connect()
        self.harness = self.safe.harness
        self.meshes = meshes
        lo, hi = self.kin.model.lowerPositionLimit[6:], self.kin.model.upperPositionLimit[6:]
        self.finger_range = (float(lo[0]), float(hi[0]))
        # the real camera is in the image whatever the mask: its box from the profile
        mesh = _rebot_profile()[1].body_mask
        att = next(a for a in mesh.as_dict()["attachments"] if a["name"] == "wrist_d455f")
        c, s = np.asarray(att["xyz"], float), np.asarray(att["box"], float)
        self.camera_box = box_surface(c - s / 2, c + s / 2)          # in the gripper_end frame
        self.objects = []                                               # base-frame surface point sets

    def set(self, q, opening=1.0):
        """Put the MockArm at q with the jaws `opening` (0 closed .. 1 open)."""
        g = self.acfg.gripper
        self.arm._q = np.asarray(q, float).copy()
        self.arm._gripper = float(g.closed_pos) + opening * (float(g.open_pos) - float(g.closed_pos))
        self.opening = opening

    def robot_surface(self):
        from test_link_self_mask import _pin_link_poses, _posed

        lo, hi = self.finger_range
        poses = _pin_link_poses(self.kin, self.arm.get_state().q, lo + self.opening * (hi - lo))
        T = poses["gripper_end"]
        return [triangle_surface(_posed(self.meshes, poses)), self.camera_box @ T[:3, :3].T + T[:3, 3]]

    def look(self):
        """One camera frame of the scene as it is now, through the map."""
        depth = render_depth(K, CAM, (H, W), [*self.robot_surface(), *self.objects])
        self.occ.refresh(Frame(rgb=np.zeros((H, W, 3), np.uint8), depth_m=depth, K=K.copy()), CAM)
        assert self.occ.last_error is None, self.occ.last_error
        return self.bridge.cloud

    def above_table(self, z=0.03):
        return int((self.bridge.cloud[:, 2] > z).sum())


def _home(rig):
    return np.asarray(rig.acfg.home_q, float)


def _rest(rig):
    return np.asarray(rig.acfg.park_q, float) + REST_OFFSET


def test_the_rebot_profile_requires_a_live_map_that_reads_its_own_masked_volume_as_free():
    from cascade.config import load_demo_config

    occ = load_demo_config(arm="rebot_rs", camera="mock", llm="mock").occupancy
    assert occ.enabled is True and occ.required is True and occ.unobserved == "free"
    # a map older than a few perception ticks no longer authorizes a step
    assert 0 < float(occ.max_age_s) <= 2.0
    assert 0 < float(occ.ready_timeout_s) <= 5.0


@pytest.mark.parametrize("arm", ["mock", "isaac", "so101_mock"])
def test_other_profiles_keep_the_optional_map(arm):
    from cascade.config import load_demo_config

    occ = load_demo_config(arm=arm, camera="mock", llm="mock").occupancy
    assert occ.required is False and occ.get("unobserved") is None
    assert float(occ.max_age_s) == 10.0


def test_the_rebot_refuses_to_start_with_its_required_map_switched_off(monkeypatch):
    """CASCADE_OCCUPANCY=0 (the MCP unit on 2026-10-11) is now a startup
    error for the reBot, not a silent run without obstacles."""
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import OccupancyError, OccupancyMap

    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    occ = load_demo_config(arm="rebot_rs", camera="mock", llm="mock").occupancy
    with pytest.raises(OccupancyError, match="required occupancy conflicts with CASCADE_OCCUPANCY=off"):
        OccupancyMap.from_config(occ)


@needs_rs
def test_premise_the_capsule_mask_maps_the_rebots_own_jaws_and_refuses_home(meshes):
    rig = Rig(meshes, body_mask="capsule")
    rig.set(_home(rig))
    rig.look()
    assert rig.above_table() > 20
    assert "clearance" in (rig.harness.vet_pose(_home(rig)) or "")


@needs_rs
def test_the_mesh_masked_map_never_refuses_rest_home_or_the_motion_between(meshes):
    from cascade.safety.trajectory import vet_segment

    rig = Rig(meshes)
    rest, home = _rest(rig), _home(rig)
    path = [rest + s * (home - rest) for s in np.linspace(0.0, 1.0, 7)]
    for opening in (1.0, 0.0):
        rig.set(rest, opening)
        rig.look()
        assert rig.above_table() == 0
        assert vet_segment(rig.harness, rest, home, 2.0) is None
        for a, b in itertools.pairwise(path):
            rig.set(a, opening)
            rig.look()                                 # the map refreshed with the arm where it is
            assert rig.above_table() == 0, a
            assert rig.harness.vet_step(a, b, 0.25) is None, (a, b)
        rig.set(home, opening)
        rig.look()
        assert rig.harness.vet_pose(home) is None
        assert vet_segment(rig.harness, home, rest, 2.0) is None


@needs_rs
def test_a_held_object_does_not_refuse_the_motion_that_carries_it(meshes):
    rig = Rig(meshes)
    home = _home(rig)
    rig.set(home, opening=0.08 / 0.09)                 # the jaws closed on an 8 cm object
    T = rig.kin.fk(home)
    local = box_surface((-0.03, -0.04, -0.04), (0.05, 0.04, 0.04))   # straddling the fingertips
    rig.objects = [local @ T[:3, :3].T + T[:3, 3]]
    lift = home + np.array([0.0, 0.05, 0.05, 0.0, 0.0, 0.0])

    rig.safe.body_mask.holding_fn = lambda: False      # premise: not holding -> the object is an obstacle
    rig.look()
    assert rig.above_table() > 0
    assert "clearance" in (rig.harness.vet_pose(home) or "")

    rig.safe.body_mask.holding_fn = lambda: True       # the runtime holds it
    rig.look()
    assert rig.above_table() == 0
    assert rig.harness.vet_pose(home) is None
    assert rig.harness.vet_pose(lift) is None and rig.harness.vet_step(home, lift, 0.25) is None


@needs_rs
def test_the_incident_side_grasp_beside_the_blue_object_is_refused_before_any_motion(meshes):
    """2026-10-11: grasp_object(green object) at [0.28, 0.129, 0.043] from the
    side, the blue object beside it. Seen by the camera and mapped through the
    mask, the descent is refused at the preflight, before a command is sent."""
    rebot = side_grasp_geometry()
    rig = Rig(meshes)
    rig.set(rebot.q_pre, 1.0)
    target = box_surface(*TARGET)
    rig.objects = [target]
    rig.look()
    steps = descent(rebot.q_pre, rebot.q_grasp)
    assert all(rig.harness.vet_pose(q, **DESCENT_EXEMPT) is None for q in steps)    # alone: admitted

    rig.objects = [target, box_surface(*neighbour(rebot))]
    rig.look()
    reasons = [rig.harness.vet_pose(q, **DESCENT_EXEMPT) for q in steps]
    refused = [r for r in reasons if r]
    assert refused and all("occupancy map" in r for r in refused)
    assert any("wrist camera" in r or "jaws" in r for r in refused)
