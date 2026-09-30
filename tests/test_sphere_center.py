"""A visibly spherical object is centred from its own surface, not the view-ray step.

`_recentre_by_size` steps half the smallest box extent along the view ray;
for a ball the visible cap is thin along that ray, so the centre stays on the
near side. Measured on the Spark kitchen orange: 13.4 mm off under PhysX and
13.5 mm under Newton, and the jaws pinched its near shoulder. A sphere fit
through the observed surface is used only when the cloud is unmistakably a
sphere; boxes, cans and noisy clouds keep the prior centre.
"""
import numpy as np
import pytest

from cascade.perception.grounding import refine_sphere_center


def cap(center=(0.18, 0.12), r=0.026, cam=(0.60, -0.45, 0.55), noise=0.0001, n=3000, seed=0):
    """What a depth camera sees of a ball resting on the table: the camera-facing half above z=0."""
    rng = np.random.default_rng(seed)
    c = np.array([center[0], center[1], r])
    v = rng.normal(size=(n * 4, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = c + r * v
    toward = np.asarray(cam) - p
    p = p[(np.einsum("ij,ij->i", v, toward) > 0) & (p[:, 2] > 0.0003)][:n]
    return p + rng.normal(0, noise, p.shape)


@pytest.mark.parametrize("center,r", [((0.18, 0.12), 0.026), ((0.30, -0.10), 0.02), ((0.25, 0.20), 0.035)])
def test_visible_cap_of_a_ball_gives_its_true_centre(center, r):
    pts = cap(center, r)
    prior = np.array([center[0] + 0.010, center[1] - 0.008, 0.026])   # the measured near-side bias
    out, rep = refine_sphere_center(prior, pts)
    assert rep["accepted"], rep
    assert np.linalg.norm(out[:2] - np.asarray(center)) < 0.0015
    assert out[2] == prior[2], "grasp height semantics are the planner's, not this fit's"


def box_shell(size=(0.06, 0.0375, 0.0375), center=(0.24, 0.01), n=4000, seed=1):
    rng = np.random.default_rng(seed)
    lx, ly, lz = size
    pts = rng.uniform([-lx / 2, -ly / 2, 0], [lx / 2, ly / 2, lz], size=(n, 3))
    face = rng.integers(0, 3, n)
    pts[face == 0, 2] = lz
    pts[face == 1, 0] = lx / 2
    pts[face == 2, 1] = -ly / 2
    return pts + np.array([center[0], center[1], 0.0])


def can_shell(r=0.034, h=0.082, center=(0.225, 0.215), n=4000, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi / 2, np.pi / 2, n)
    z = rng.uniform(0, h, n)
    return np.c_[center[0] + r * np.cos(a), center[1] + r * np.sin(a), z]


@pytest.mark.parametrize("kind", ["box", "can", "noisy_ball", "sparse", "tiny", "sliver"])
def test_non_spherical_or_weak_evidence_keeps_the_prior(kind):
    if kind == "box":
        pts = box_shell()
    elif kind == "can":
        pts = can_shell()
    elif kind == "noisy_ball":
        pts = cap(noise=0.004)
    elif kind == "sparse":
        pts = cap(n=120)
    elif kind == "tiny":
        pts = cap(r=0.008)
    else:  # a thin sliver of the cap seen edge-on: < 90 degrees of arc
        pts = cap()
        ang = np.arctan2(pts[:, 1] - 0.12, pts[:, 0] - 0.18)
        pts = pts[np.abs(ang - np.median(ang)) < 0.3]
    prior = np.array([0.2, 0.1, 0.03])
    out, rep = refine_sphere_center(prior, pts)
    assert not rep["accepted"], rep
    assert np.array_equal(out, prior)


def test_cuda_twin_matches_the_cpu_fit(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA geometry test")
    monkeypatch.delenv("CASCADE_GPU_PERCEPTION_EVIDENCE_DIR", raising=False)
    pts = cap()
    prior = np.array([0.19, 0.112, 0.026])
    cpu, cpu_rep = refine_sphere_center(prior, pts)
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "1")
    monkeypatch.setenv("CASCADE_DEVICE", "cuda:0")
    from cascade.perception import cuda_math
    gpu, gpu_rep = cuda_math.refine_sphere(prior, cuda_math.tensor(pts))
    assert cpu_rep["accepted"] and gpu_rep["accepted"]
    assert np.allclose(cpu, gpu, atol=2e-4)
    assert cuda_math._stages["sphere_surface_fit"]["devices"] == ["cuda:0"]
