"""Occupancy body mask from the arm's URDF MESHES (real, non-Isaac arms).

Measured on the real reBot rig (2026-10-11, calibrated fixed D455F, arm at
its rest pose q = 0): the production occupancy body mask -- capsules of
`body_mask_radius_m` (6 cm) around the joint polyline -- left 108 occupied
voxels on the gripper and wrist (the open jaws are ~14 cm wide, the wrist
D455F sits on top of them), the nearest 2.8 cm from the arm chain, under
`min_clearance_m` 0.03. The robot's own body polluted the map and motion from
rest could be refused.

`body_mask: mesh` in an arm profile rasterizes the arm's URDF geometry (every
link, the fingers at the measured opening, plus profile-declared rigid
attachments such as the wrist camera) into each depth frame's mask before
integration, with a metric inflation and a pixel dilation. These tests pin:

- the geometry operations (metric inflation, attachments) against independent
  geometric truth;
- the mask against depth RENDERED from the test's own triangles (a z-buffer
  written here, not the code under test), including the jaw case the capsule
  misses and an eye-in-hand camera whose jaws straddle the near plane;
- the fail-closed contract: no joint state -> nothing is integrated;
- the profile parsing and the refusal to fall back silently to capsules.
"""
from __future__ import annotations

import numpy as np
import pytest

from cascade.perception import link_mask as lm


def _hull_contains(points: np.ndarray, query: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Independent containment truth: scipy's hull half-spaces."""
    from scipy.spatial import ConvexHull

    eq = ConvexHull(points).equations
    return (query @ eq[:, :3].T + eq[:, 3] <= tol).all(axis=1)


def _unit_dirs(n: int, seed: int = 0) -> np.ndarray:
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _cube_geometry(size=0.04) -> lm.LinkGeometry:
    corners = lm._box_points((size, size, size))
    sliver = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.002, 0], [0, 0, 0.002]], dtype=float)
    return lm.LinkGeometry(pieces={"cube": (corners,), "sliver": (sliver,)},
                           sources={"cube": "box", "sliver": "mesh"})


# ── slice 1: metric inflation ─────────────────────────────────────────────


@pytest.mark.parametrize("margin", [0.005, 0.02])
def test_inflation_contains_every_point_within_the_margin_of_each_piece(margin):
    geo = _cube_geometry()
    fat = geo.inflated(margin)
    assert fat.links == geo.links and fat.sources == geo.sources
    for link in geo.links:
        for piece, fat_piece in zip(geo.pieces[link], fat.pieces[link], strict=True):
            # every point at exactly `margin` from any vertex (the Minkowski
            # sum's extreme points) and the piece itself must lie inside
            probe = (piece[:, None, :] + margin * _unit_dirs(400)[None]).reshape(-1, 3)
            assert _hull_contains(fat_piece, np.concatenate([piece, probe])).all()


def test_inflation_is_bounded_by_the_circumscribed_cover():
    """A metric inflation, not unbounded bloat: no inflated vertex lies more
    than the cover's circumradius (1.26 x margin for an icosahedron whose
    inradius is the margin) from the original piece."""
    geo, margin = _cube_geometry(), 0.01
    fat = geo.inflated(margin)
    for link in geo.links:
        for piece, fat_piece in zip(geo.pieces[link], fat.pieces[link], strict=True):
            gap = np.linalg.norm(fat_piece[:, None, :] - piece[None], axis=2).min(axis=1)
            assert gap.max() <= 1.26 * margin + 1e-12


def test_zero_inflation_is_the_identity_and_bad_margins_are_refused():
    geo = _cube_geometry()
    assert geo.inflated(0.0) is geo
    for bad in (-0.01, float("nan"), float("inf"), "1cm", True):
        with pytest.raises(ValueError, match="margin"):
            geo.inflated(bad)


# ── slice 2: rigid attachments the URDF does not model ──────────────────


def _rpy_T(xyz, rpy) -> np.ndarray:
    from cascade.types import pose_to_transform

    return pose_to_transform(list(xyz) + list(rpy))


def test_a_box_attachment_becomes_a_piece_of_its_link_in_the_link_frame():
    """The reBot URDF has no wrist D455F: its body must be declarable on the
    link it is bolted to, posed in that link's frame."""
    geo = _cube_geometry()
    T = _rpy_T((-0.09, 0.0, 0.06), (0.0, 0.0, np.pi / 2))
    out = geo.with_boxes([("wrist_cam", "cube", (0.06, 0.14, 0.05), T)])
    assert len(out.pieces["cube"]) == len(geo.pieces["cube"]) + 1
    added = out.pieces["cube"][-1]
    # independent truth: the box's 8 corners, rotated and moved by hand
    half = np.array([0.03, 0.07, 0.025])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    want = corners @ T[:3, :3].T + T[:3, 3]
    assert sorted(map(tuple, np.round(added, 9))) == sorted(map(tuple, np.round(want, 9)))
    assert "wrist_cam" in out.sources["cube"]
    assert geo.pieces["cube"] is not out.pieces["cube"], "the original geometry must not change"


def test_a_box_on_a_link_without_geometry_creates_that_link():
    geo = _cube_geometry()
    out = geo.with_boxes([("mount", "flange", (0.02, 0.02, 0.02), np.eye(4))])
    assert "flange" in out.links and len(out.pieces["flange"]) == 1
    assert "mount" in out.sources["flange"]


@pytest.mark.parametrize("size", [(0.0, 0.1, 0.1), (-0.01, 0.1, 0.1), (np.nan, 0.1, 0.1), (0.1, 0.1)])
def test_a_degenerate_box_is_refused(size):
    with pytest.raises(ValueError, match="box"):
        _cube_geometry().with_boxes([("bad", "cube", size, np.eye(4))])
