import numpy as np

from cascade.grasping.force import select_profile
from cascade.grasping.obb_grasp import _yaw_rotation, plan_grasps_from_fix
from cascade.perception.grounding import oriented_bbox
from cascade.types import Detection, ObjectFix


def make_fix(center=(0.3, 0.0), size=(0.10, 0.04, 0.06), yaw=0.0, n=400):
    """Synthesize a box point cloud sitting on the table (z in [0, h])."""
    rng = np.random.default_rng(0)
    l, w, h = size
    pts = rng.uniform([-l / 2, -w / 2, 0], [l / 2, w / 2, h], size=(n, 3))
    R = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    pts = pts @ R.T + np.array([center[0], center[1], 0.0])
    c, extents, axes = oriented_bbox(pts)
    det = Detection(label="box", conf=0.9, bbox=np.array([0, 0, 10, 10], dtype=np.float32))
    return ObjectFix(label="box", position=c, points=pts, detection=det, extent=extents, axes=axes)


def test_grasp_across_short_axis():
    fix = make_fix(size=(0.10, 0.04, 0.06), yaw=0.0)
    grasps = plan_grasps_from_fix(fix, table_z=0.0, max_width_m=0.09)
    assert grasps
    g = grasps[0]
    # Opening axis should be along the box's short side (y here).
    open_axis = g.rotation[:, 1]
    assert abs(open_axis[1]) > 0.95
    assert g.width_m < 0.07  # 4 cm + pad
    assert 0.0 < g.position[2] < 0.06
    assert np.allclose(g.approach, [0, 0, -1])


def test_grasp_yaw_follows_object():
    yaw = 0.6
    fix = make_fix(size=(0.12, 0.03, 0.05), yaw=yaw)
    grasps = plan_grasps_from_fix(fix, table_z=0.0)
    open_axis = grasps[0].rotation[:, 1]
    got = np.arctan2(open_axis[1], open_axis[0])
    # Opening across the short side = perpendicular to the box's long axis.
    want = yaw + np.pi / 2
    diff = (got - want + np.pi / 2) % np.pi - np.pi / 2
    assert abs(diff) < 0.15


def test_too_wide_object_flagged():
    fix = make_fix(size=(0.15, 0.13, 0.05))
    grasps = plan_grasps_from_fix(fix, table_z=0.0, max_width_m=0.09)
    assert all(g.quality < 0.5 for g in grasps)  # feasible=False downweights


def test_grasp_z_clamped_above_table():
    fix = make_fix(size=(0.06, 0.03, 0.008))  # very flat object
    grasps = plan_grasps_from_fix(fix, table_z=0.0, min_grasp_z_above_table=0.005)
    assert grasps[0].position[2] >= 0.005 - 1e-9


def test_yaw_rotation_orthonormal():
    for yaw in [0.0, 0.5, -1.2, 3.0]:
        R = _yaw_rotation(yaw)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R[:, 0], [0, 0, -1])  # approach down


def test_force_profiles():
    assert select_profile("wine glass").name == "fragile"
    assert select_profile("plush toy").name == "soft"
    assert select_profile("soda can").name == "slippery"
    assert select_profile("box").name == "rigid"
    assert select_profile("box", material_hint="fragile").name == "fragile"
    assert select_profile("box", material_hint="very fragile item").name == "fragile"


def test_select_grasp_offsets_a_single_hinge_jaw():
    """A hinge jaw closes against its FIXED tip, not about the frame origin.

    With jaw_fixed_tip_m/jaw_close_dir set, the IK target must be displaced
    by -R @ (fixed_tip + close_dir*w/2): posing the frame origin on the
    object parks the SO-101's closing point 30 mm away, the finger shoves
    the prop on descent and the grasp reads "air grasp" -- the measured
    failure of every so101_mujoco grasp while perception was 1.4 mm accurate.
    Parallel-jaw callers omit the keys and the target must be untouched.
    """
    from cascade.grasping import select_grasp

    fix = make_fix(center=(0.20, 0.0), size=(0.035, 0.035, 0.05))
    grasps = plan_grasps_from_fix(fix, table_z=0.0, max_width_m=0.055,
                                  width_pad_m=0.006, axis_order="open_down")

    seen = {}

    class _Kin:
        def ik(self, T, seed):
            class R:
                success, error, q = True, 0.0, np.zeros(5)
            seen.setdefault("targets", []).append(T[:3, 3].copy())
            return R()

    kw = dict(max_width_m=0.055, pregrasp_offset_m=0.03)
    g_plain, _, _ = select_grasp(grasps, _Kin(), np.zeros(5), **kw)
    plain_target = seen["targets"][1]  # [0]=pregrasp, [1]=grasp

    seen.clear()
    tip = np.array([0.0, 0.0, 0.0])
    close = np.array([-1.0, 0.0, 0.0])
    g_off, _, _ = select_grasp(grasps, _Kin(), np.zeros(5), **kw,
                               jaw_fixed_tip_m=tip, jaw_close_dir=close)
    off_target = seen["targets"][1]

    assert g_off.width_m == g_plain.width_m
    expect = plain_target - g_off.rotation @ (tip + close * g_off.width_m / 2)
    assert np.allclose(off_target, expect, atol=1e-12), (
        f"frame target {off_target} != object - R@datum {expect}"
    )
    shift = float(np.linalg.norm(off_target - plain_target))
    assert abs(shift - g_off.width_m / 2) < 1e-9, (
        "the offset must be half the closing width along the close direction"
    )
