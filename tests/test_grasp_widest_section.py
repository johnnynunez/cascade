"""Parallel jaws must reach the widest cross-section of a rounded object.

A top-down grasp at `depth_fraction` of the height from the top is right for
a box (vertical side faces). On a sphere it puts the fingertips above the
equator, where the surface normals tilt upward: the squeeze pushes the
object down and out of the jaws. Measured in the kitchen: the 52 mm orange
pinched 16 mm above its centre is lost on lift in plain MuJoCo (vendor
reBot, any friction up to 2.0 and condim 3/4/6) and under Newton; with the
equator inside the pads it holds.
"""
import numpy as np
import pytest

from cascade.grasping.obb_grasp import (
    ROUND_GRASP_INSET_M,
    _widest_section_z,
    plan_grasps_from_fix,
)
from cascade.perception.grounding import oriented_bbox
from cascade.types import Detection, ObjectFix


def _fix(points, label):
    c, extents, axes = oriented_bbox(points)
    det = Detection(label=label, conf=0.9, bbox=np.array([0, 0, 10, 10], dtype=np.float32))
    return ObjectFix(label=label, position=c, points=points, detection=det, extent=extents, axes=axes)


def _sphere(center, r, n=2500, min_z=None, seed=0):
    """Surface samples as a depth camera sees them: everything above `min_z`."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n * 3, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = np.asarray(center) + r * v
    if min_z is not None:
        p = p[p[:, 2] >= min_z]
    return p[:n]


def test_sphere_grasp_puts_the_equator_inside_the_pads():
    r = 0.026
    pts = _sphere((0.19, 0.12, r), r, min_z=0.004)
    grasps = plan_grasps_from_fix(_fix(pts, "orange"), table_z=0.0, depth_fraction=0.15)
    assert grasps
    for g in grasps:
        # TCP = fingertip plane, ROUND_GRASP_INSET_M below the equator
        assert g.position[2] == pytest.approx(r - ROUND_GRASP_INSET_M, abs=0.006)


@pytest.mark.parametrize('shape', ['sphere', 'cylinder', 'sparse'])
def test_cuda_cloud_uses_device_reductions_and_matches_cpu_height(monkeypatch, shape):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip('NVIDIA CUDA geometry test')
    monkeypatch.delenv('CASCADE_REQUIRE_CUDA', raising=False)
    monkeypatch.delenv('CASCADE_GPU_PERCEPTION_EVIDENCE_DIR', raising=False)
    pts = _sphere((.19, .12, .026), .026, min_z=.004)
    if shape == 'sparse':
        pts = pts[:30]
    elif shape == 'cylinder':
        rng = np.random.default_rng(23)
        theta = rng.uniform(0, 2*np.pi, 2500)
        pts = np.c_[.025*np.cos(theta), .025*np.sin(theta), rng.uniform(0, .08, len(theta))]
    top = float(pts[:, 2].max())
    args = (np.array([.6, .8, 0.]), .85*top, 0., top, .005)
    expected = _widest_section_z(pts, *args)
    monkeypatch.setenv('CASCADE_REQUIRE_CUDA', '1')
    from cascade.perception import cuda_math
    actual = _widest_section_z(cuda_math.tensor(pts), *args)
    assert actual == pytest.approx(expected, abs=1e-9)
    if shape != 'sparse':
        assert cuda_math._stages['horizontal_width_profile']['devices'] == ['cuda:0']


def test_small_sphere_grasp_never_goes_below_the_table_clamp():
    r = 0.012
    pts = _sphere((0.2, 0.0, r), r, min_z=0.002)
    grasps = plan_grasps_from_fix(_fix(pts, "ball"), table_z=0.0, depth_fraction=0.15,
                                  min_grasp_z_above_table=0.005)
    assert all(g.position[2] >= 0.005 - 1e-9 for g in grasps)


def test_lying_ellipsoid_grasp_reaches_its_widest_section():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(6000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    semi = np.array([0.034, 0.022, 0.022])  # lemon lying on its side
    pts = np.array([0.32, 0.0, semi[2]]) + v * semi
    pts = pts[pts[:, 2] >= 0.004]
    grasps = plan_grasps_from_fix(_fix(pts, "lemon"), table_z=0.0, depth_fraction=0.15)
    assert all(g.position[2] == pytest.approx(semi[2] - ROUND_GRASP_INSET_M, abs=0.006) for g in grasps)


@pytest.mark.parametrize("size", [(0.06, 0.0375, 0.0375), (0.04, 0.04, 0.05), (0.10, 0.04, 0.06)])
def test_box_keeps_the_top_biased_grasp_height(size):
    rng = np.random.default_rng(2)
    l, w, h = size
    pts = rng.uniform([-l / 2, -w / 2, 0], [l / 2, w / 2, h], size=(3000, 3)) + np.array([0.24, 0.01, 0.0])
    grasps = plan_grasps_from_fix(_fix(pts, "box"), table_z=0.0, depth_fraction=0.15)
    top = pts[:, 2].max()
    assert all(g.position[2] == pytest.approx(top - 0.15 * top, abs=1e-6) for g in grasps)


def test_upright_cylinder_keeps_the_top_biased_grasp_height():
    rng = np.random.default_rng(3)
    th = rng.uniform(0, 2 * np.pi, 4000)
    z = rng.uniform(0, 0.082, 4000)
    pts = np.c_[0.225 + 0.034 * np.cos(th), 0.215 + 0.034 * np.sin(th), z]
    grasps = plan_grasps_from_fix(_fix(pts, "can"), table_z=0.0, depth_fraction=0.15)
    top = pts[:, 2].max()
    assert all(g.position[2] == pytest.approx(top - 0.15 * top, abs=1e-6) for g in grasps)


@pytest.mark.parametrize("axis", [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
def test_a_noisy_top_point_does_not_move_the_widest_section(axis):
    """A 1 mm change at the pole must not re-bin the whole equator."""
    pts = _sphere((0.19, 0.12, 0.026), 0.026, min_z=0.004)
    heights = []
    for dz in np.linspace(0.0, 0.001, 21):
        cloud = pts.copy()
        cloud[np.argmax(cloud[:, 2]), 2] += dz
        top = float(cloud[:, 2].max())
        heights.append(_widest_section_z(cloud, np.asarray(axis), 0.85 * top,
                                         0.0, top, 0.005))
    assert np.ptp(heights) < 0.00025
    assert np.mean(heights) == pytest.approx(0.026 - ROUND_GRASP_INSET_M, abs=0.002)


@pytest.mark.parametrize("axis", [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
def test_widest_section_is_stable_under_depth_noise_and_resampling(axis):
    """Repeat observations must not jump between disjoint 4 mm bands."""
    pts = _sphere((0.19, 0.12, 0.026), 0.026, min_z=0.004)
    for perturbation in ("noise", "resample"):
        heights = []
        for seed in range(21):
            rng = np.random.default_rng(seed)
            cloud = (pts + rng.normal(scale=0.0005, size=pts.shape)
                     if perturbation == "noise"
                     else pts[rng.choice(len(pts), 1000, replace=False)])
            top = float(cloud[:, 2].max())
            heights.append(_widest_section_z(cloud, np.asarray(axis), 0.85 * top,
                                             0.0, top, 0.005))
        assert np.ptp(heights) < 0.002
        assert np.max(np.abs(np.asarray(heights) - 0.016)) < 0.002


def test_widest_section_moves_with_the_cloud_and_table():
    pts = _sphere((0.19, 0.12, 0.026), 0.026, min_z=0.004)
    axis = np.array([0.6, 0.8, 0.0])
    top = float(pts[:, 2].max())
    before = _widest_section_z(pts, axis, 0.85 * top, 0.0, top, 0.005)
    translation = np.array([0.13, -0.27, 0.743])
    dz = translation[2]
    after = _widest_section_z(pts + translation, axis, 0.85 * top + dz,
                              dz, top + dz, 0.005 + dz)
    assert after == pytest.approx(before + dz, abs=1e-9)


def test_overlapping_slices_do_not_make_a_sparse_cloud_sufficient():
    pts = _sphere((0.19, 0.12, 0.026), 0.026, min_z=0.004)
    # There are many points, but too little vertical coverage to identify a
    # rounded profile. Overlap must not inflate the independent slice count.
    pts = pts[(pts[:, 2] >= 0.040) & (pts[:, 2] <= 0.051)]
    planned = 0.045
    assert _widest_section_z(pts, np.array([1., 0., 0.]), planned,
                             0.0, 0.052, 0.005) == planned


def test_partial_cylinder_view_includes_the_actual_top_surface():
    """The lid can sit between the last regular slice and the measured top."""
    rng = np.random.default_rng(8)
    radius, height = 0.034, 0.082674
    z = rng.uniform(0.0, height, 3000)
    # Only a narrow strip of the side is visible, tapering with occlusion;
    # the lid still establishes that the top has the object's full width.
    visible_half_width = 0.011 * (0.5 + 0.5 * np.sin(np.pi * z / height))
    x = rng.uniform(-1.0, 1.0, len(z)) * visible_half_width
    sides = np.c_[x, np.sqrt(radius ** 2 - x ** 2), z]
    theta = rng.uniform(0.0, 2 * np.pi, 200)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, len(theta)))
    lid = np.c_[r * np.cos(theta), r * np.sin(theta), np.full(len(theta), height)]
    pts = np.vstack([sides, lid])
    planned = 0.85 * height
    assert _widest_section_z(pts, np.array([1., 0., 0.]), planned,
                             0.0, height, 0.005) == planned


def test_isolated_wide_band_does_not_pull_the_grasp_to_the_support_plane():
    pts = _sphere((0.0, 0.0, 0.026), 0.026, min_z=0.004)
    # A few table-edge points survive the mask near the support plane.
    # The wide body is supported across many consecutive height bands.
    edge = np.c_[np.linspace(-0.03, 0.03, 40), np.zeros(40), np.full(40, 0.0005)]
    pts = np.vstack([pts, edge])
    top = float(pts[:, 2].max())
    result = _widest_section_z(pts, np.array([1., 0., 0.]), 0.85 * top,
                               0.0, top, 0.005)
    assert result == pytest.approx(0.026 - ROUND_GRASP_INSET_M, abs=0.002)
