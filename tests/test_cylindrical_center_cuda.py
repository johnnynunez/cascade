"""Measured-shell centering without known object radius or privileged poses."""
import os
import numpy as np
import pytest


def cuda(monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip('NVIDIA CUDA geometry test')
    monkeypatch.setenv('CASCADE_REQUIRE_CUDA', '1')
    monkeypatch.setenv('CASCADE_DEVICE', 'cuda:0')
    monkeypatch.delenv('CASCADE_GPU_PERCEPTION_EVIDENCE_DIR', raising=False)
    from cascade.perception import cuda_math
    return cuda_math


def shell(*, center=(.24, .21), radius=.033, arc=172, noise=.0004, seed=37):
    rng = np.random.default_rng(seed)
    a = np.deg2rad(np.linspace(-arc/2, arc/2, 75))
    z = np.linspace(.005, 3 * radius - .005, 45)
    a, z = np.meshgrid(a, z)
    pts = np.column_stack((center[0]+radius*np.cos(a.ravel()),
                           center[1]+radius*np.sin(a.ravel()), z.ravel()))
    return pts+rng.normal(0, noise, pts.shape)


@pytest.mark.parametrize('center,radius', [((.24, .21), .033), ((.19, -.12), .022), ((.35, .04), .042)])
def test_partial_visible_cylinder_has_measured_center_not_visible_shell_center(monkeypatch, center, radius):
    gpu = cuda(monkeypatch)
    pts = gpu.tensor(shell(center=center, radius=radius))
    old = np.array([center[0]+.027, center[1]-.012, .059])
    refined, fit = gpu.refine_upright_cylinder(old, pts)
    assert fit['accepted']
    assert np.linalg.norm(refined[:2]-center) < .002
    assert abs(fit['radius_m']-radius) < .002
    assert refined[2] == old[2], 'Existing grasp-height semantics must remain unchanged'
    assert gpu._stages['cylinder_surface_fit']['devices'] == ['cuda:0']


@pytest.mark.parametrize('kind', ['short_arc', 'noisy', 'cube', 'sideways', 'sparse', 'flat'])
def test_insufficient_or_wrong_shape_keeps_prior_center(monkeypatch, kind):
    gpu = cuda(monkeypatch)
    pts = shell()
    if kind == 'short_arc':
        pts = shell(arc=45, noise=.00002)
    elif kind == 'noisy':
        pts = shell(noise=.006)
    elif kind == 'cube':
        t, z = np.meshgrid(np.linspace(-.025, .025, 70), np.linspace(.003, .083, 40))
        pts = np.concatenate([np.column_stack((t.ravel(), np.full(t.size, y), z.ravel())) for y in (-.025, .025)]
                             + [np.column_stack((np.full(t.size, x), t.ravel(), z.ravel())) for x in (-.025, .025)])
        pts[:, :2] += [.24, .21]
    elif kind == 'sideways':
        pts = pts[:, [2, 1, 0]]
    elif kind == 'sparse':
        pts = pts[:20]
    elif kind == 'flat':
        pts[:, 2] = .05
    old = np.array([.267, .198, .059])
    refined, fit = gpu.refine_upright_cylinder(old, gpu.tensor(pts))
    assert not fit['accepted'], kind
    np.testing.assert_array_equal(refined, old)


def test_localization_routes_can_geometry_but_preserves_cube_path(monkeypatch):
    gpu = cuda(monkeypatch)
    from cascade.perception import grounding
    from cascade.types import Frame, Detection
    points = gpu.tensor(shell(center=(.24, .21)))
    monkeypatch.setattr(grounding, 'mask_to_points_cam', lambda *_: points)
    frame = Frame(rgb=np.zeros((4, 4, 3), np.uint8), depth_m=np.ones((4, 4)), K=np.eye(3))
    class Detector:
        label = 'tomato can'
        def detect(self, *_args, **_kwargs):
            return [Detection(self.label, 1., np.array([0, 0, 4, 4]), mask=np.ones((4, 4), bool))]
    detector = Detector()
    ext = grounding.Extrinsics(T=np.eye(4))
    can = grounding.localize_object(frame, 'tomato can', detector, ext)
    assert np.linalg.norm(can.position[:2]-[.24, .21]) < .002
    detector.label = 'green cube'
    cube = grounding.localize_object(frame, 'green cube', detector, ext)
    center, extents, _ = grounding.oriented_bbox(points)
    expected = grounding._recentre_by_size(center, points, extents, np.zeros(3))
    np.testing.assert_array_equal(cube.position, expected)
