"""The grasp target's own top must not refuse its own approach.

Found on the physical reBot (front D435i, warp occupancy, 1 cm voxels): the
pregrasp is vetted and approached before the descent cylinder opens, and a
~10 cm cup's map surface sat ~1.5 cm above its rim. Every candidate was refused
("pregrasp unsafe: point 0 clearance 0.010-0.022 m below 0.030 m (occupancy
map)"), and grasp memory turned each refusal into "raise the grasp" until the
jaws closed above the cup.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import needs_pin
from cascade.memory.grasp_memory import GraspOutcomeMemory, _reason_key
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.skills.runtime import _target_column

CUP_XY = np.array([0.33, 0.107])
TOP_Z = 0.13            # map surface of the cup top: real rim ~0.105, inflated by
                        # 1 cm voxels + depth noise; 1.2 cm under the pregrasp TCP
PREGRASP_Z = 0.142


class PointObstacles:
    """Minimal occupancy stand-in: clearance = distance to listed points."""

    required = False

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def clearance(self, query):
        query = np.atleast_2d(np.asarray(query, dtype=float))
        return np.linalg.norm(query[:, None, :] - self.points[None, :, :], axis=2).min(axis=1)


def rim(center_xy=CUP_XY, z=TOP_Z, r=0.045, step=0.01):
    """The cup's top as the map stores it: a filled 1 cm-voxel disc."""
    g = np.arange(-r, r + 1e-9, step)
    xx, yy = np.meshgrid(g, g)
    keep = xx ** 2 + yy ** 2 <= r ** 2
    return np.c_[center_xy[0] + xx[keep], center_xy[1] + yy[keep], np.full(keep.sum(), z)]


@pytest.fixture
def rebot():
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.grasping.obb_grasp import _yaw_rotation
    from cascade.types import make_transform

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    arm = cfg.arm
    kin = Kinematics(arm.model, arm.get("ee_frame", "gripper_end"), 6,
                     joint_signs=arm.get("joint_signs"))

    def above(xy, z):
        yaw = float(np.arctan2(xy[1], xy[0]))
        ik = kin.ik(make_transform(_yaw_rotation(yaw, axis_order="down_open"), [xy[0], xy[1], z]),
                    np.asarray(arm.home_q, dtype=float))
        assert ik.success
        return np.asarray(ik.q)

    def harness(obstacles):
        return SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin,
                             occupancy=PointObstacles(obstacles))

    return SimpleNamespace(cfg=cfg, kin=kin, above=above, harness=harness)


COLUMN = (CUP_XY, 0.06, 0.11)


@needs_pin
def test_premise_the_cup_rim_refuses_its_own_pregrasp(rebot):
    h = rebot.harness(rim())
    reason = h.vet_pose(rebot.above(CUP_XY, PREGRASP_Z))
    assert reason and "occupancy map" in reason


@needs_pin
def test_column_admits_the_pregrasp_check_and_the_approach_step(rebot):
    h = rebot.harness(rim())
    q_pre = rebot.above(CUP_XY, PREGRASP_Z)
    q_before = rebot.above(CUP_XY, PREGRASP_Z + 0.01)
    assert h.vet_pose(q_pre, exempt_xy=COLUMN[0], exempt_radius_m=COLUMN[1],
                      exempt_z_min=COLUMN[2]) is None
    with h.target_column_exemption(COLUMN) as active:
        assert active
        assert h.vet_step(q_before, q_pre, 1.0) is None
    # outside the block the live approve path is back to the strict check
    assert h._grasp_exempt is None
    assert "occupancy map" in h.vet_step(q_before, q_pre, 1.0)


@needs_pin
def test_arm_points_outside_the_column_stay_checked(rebot):
    q_pre = rebot.above(CUP_XY, PREGRASP_Z)
    links = rebot.kin.link_positions(q_pre)
    outside = [p for p in links[1:] if np.linalg.norm(p[:2] - CUP_XY) > COLUMN[1] + 0.03
               and p[2] > 0.05]
    assert outside, "expected an elbow/wrist joint away from the target column"
    obstacle = outside[-1] + np.array([0.0, 0.0, 0.015])  # 1.5 cm from that joint
    h = rebot.harness(np.vstack([rim(), obstacle]))
    with h.target_column_exemption(COLUMN):
        reason = h.vet_step(rebot.above(CUP_XY, PREGRASP_Z + 0.01), q_pre, 1.0)
    assert reason and "occupancy map" in reason
    assert h.vet_pose(q_pre, exempt_xy=COLUMN[0], exempt_radius_m=COLUMN[1],
                      exempt_z_min=COLUMN[2]) is not None


@needs_pin
def test_retained_episodes_keep_ownership_of_the_exemption(rebot):
    h = rebot.harness(rim())
    h._pending_release_episode = {"cylinder": (CUP_XY, 0.07, 0.0)}
    with h.target_column_exemption(COLUMN) as active:
        assert not active
        assert h._grasp_exempt is None


@needs_pin
def test_column_is_restored_even_if_the_block_raises(rebot):
    h = rebot.harness(rim())
    with pytest.raises(RuntimeError):
        with h.target_column_exemption(COLUMN):
            raise RuntimeError("approach failed")
    assert h._grasp_exempt is None


# ── the column the runtime derives from the observed target ─────────────


def column(fix, *, grasp_z=0.0, harness_supports=True, **grasp_cfg):
    gcfg = {"exempt_radius_m": 0.15, "approach_target_exemption": True, **grasp_cfg}
    harness = SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=0.01))
    if harness_supports:
        harness.target_column_exemption = lambda cylinder: None
    return _target_column(gcfg, harness, SimpleNamespace(position=[*CUP_XY, grasp_z]), fix)


def cup_fix(top=0.105, r=0.045):
    a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    side = np.c_[CUP_XY[0] + r * np.cos(a), CUP_XY[1] + r * np.sin(a), np.linspace(0.0, top, 40)]
    return SimpleNamespace(points=side)


def test_column_follows_the_target_footprint_and_top():
    center, radius, z_min = column(cup_fix(), grasp_z=0.06)
    np.testing.assert_allclose(center, CUP_XY)
    assert radius == pytest.approx(0.055, abs=1e-3)        # 4.5 cm footprint + 1 cm
    assert z_min == pytest.approx(0.085)                    # 2 cm under the observed top


def test_flat_target_keeps_the_table_floor():
    _, _, z_min = column(cup_fix(top=0.012))
    assert z_min == pytest.approx(0.01)                     # table_z + table_clearance


def test_radius_is_capped_and_feature_can_be_disabled():
    _, radius, _ = column(cup_fix(r=0.30), exempt_radius_m=0.07)
    assert radius == pytest.approx(0.07)
    assert column(cup_fix(), approach_target_exemption=False) is None


def test_off_unless_the_arm_profile_turns_it_on():
    harness = SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=0.01),
                              target_column_exemption=lambda cylinder: None)
    assert _target_column({"exempt_radius_m": 0.15}, harness,
                          SimpleNamespace(position=[*CUP_XY, 0.0]), cup_fix()) is None
    from cascade.config import load_demo_config
    assert load_demo_config(arm="rebot_rs", camera="mock", llm="mock").grasp.get("approach_target_exemption") is True
    assert load_demo_config(arm="mock", camera="mock", llm="mock").grasp.get("approach_target_exemption") is False


def test_unusable_points_or_a_harness_without_support_keep_the_strict_check():
    assert column(SimpleNamespace(points=np.array([[np.nan, 0.0, 0.0]]))) is None
    assert column(SimpleNamespace()) is None
    assert column(cup_fix(), harness_supports=False) is None


# ── grasp memory no longer learns "raise the grasp" from map refusals ───

MAP_REFUSAL = ("no executable grasp: left paper cup: pregrasp unsafe: point 0 clearance "
               "0.010 m below 0.030 m (occupancy map) at [0.3291, 0.1070, 0.1177]")


def test_map_refusals_have_their_own_bucket():
    assert _reason_key(MAP_REFUSAL) == "occupancy_refused"
    assert _reason_key("pregrasp unsafe: link/joint 4 at z=-0.010 would hit the table") == "link_hits_table"
    assert _reason_key("pregrasp unsafe: TCP outside workspace") == "pregrasp_unsafe"


def test_map_refusals_do_not_raise_the_grasp(tmp_path):
    memory = GraspOutcomeMemory(path=tmp_path / "grasp_memory.json")
    fix = SimpleNamespace(extent=np.array([0.09, 0.09, 0.06]), position=np.array([*CUP_XY, 0.04]),
                          points=cup_fix().points)
    for _ in range(6):
        memory.record("paper cup", fix, None, success=False, reason=MAP_REFUSAL)
    assert memory.grasp_z_nudge("paper cup", fix) == pytest.approx(0.0)
    memory.record("paper cup", fix, None, success=False, reason="pregrasp unsafe: link/joint 3 would hit the table")
    assert memory.grasp_z_nudge("paper cup", fix) == pytest.approx(0.008)  # real height evidence still counts
