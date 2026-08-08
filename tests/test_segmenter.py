"""Learned segmentation behind pixel addressing, and its fallback contract.

Depth connectivity cannot separate an object from the surface it rests on.
Measured on a LIBERO agentview frame against MuJoCo body poses:

    object                    depth fill      SAM2.1 point prompt
    akita_black_bowl_2_main   FAIL (34 cm)    3.0 cm
    plate_1_main              FAIL (22 cm)    0.9 cm

These tests do not re-run that measurement (it needs the renderer and a GPU);
they pin the CONTRACT around it, which is where silent breakage would hide: a
segmenter that dies must degrade to depth, never take the grasp down with it,
and a degraded run must still refuse to return a wrong pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrc_demo.perception.pixel_target import fix_from_mask
from wrc_demo.perception.segmenter import fix_at_pixel
from wrc_demo.types import Frame, SkillError

K = np.array([[200.0, 0.0, 64.0],
              [0.0, 200.0, 64.0],
              [0.0, 0.0, 1.0]])


def _scene() -> np.ndarray:
    d = np.full((128, 128), 1.0, dtype=np.float32)
    d[50:70, 50:70] = 0.8
    return d


def _frame(depth: np.ndarray | None = None) -> Frame:
    depth = _scene() if depth is None else depth
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    return Frame(rgb=rgb, depth_m=depth, K=K, t=0.0, depth_source="sensor")


class _FakeSegmenter:
    """Returns a fixed mask, or raises, to exercise the contract."""

    def __init__(self, mask=None, exc=None):
        self._mask, self._exc = mask, exc
        self.calls = 0

    def mask_at(self, frame, u, v):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._mask


# ── mask lifting is shared by both paths ─────────────────────────────────

def test_fix_from_mask_matches_the_depth_path_geometry():
    """Both paths must compute 3D identically, or comparing them is meaningless."""
    from wrc_demo.perception.pixel_target import fix_from_pixel

    frame = _frame()
    mask = np.zeros((128, 128), dtype=bool)
    mask[50:70, 50:70] = True

    a = fix_from_pixel(frame, np.eye(4), u=60, v=60)
    b = fix_from_mask(frame, np.eye(4), mask)
    assert np.allclose(a.position, b.position, atol=1e-9)
    assert np.allclose(a.extent, b.extent, atol=1e-9)


def test_fix_from_mask_ignores_invalid_depth_inside_the_mask():
    """A segmenter mask may cover depth holes; those pixels must not become 3D."""
    d = _scene()
    d[55:60, 55:60] = 0.0            # hole inside the object
    mask = np.zeros((128, 128), dtype=bool)
    mask[50:70, 50:70] = True
    fix = fix_from_mask(_frame(d), np.eye(4), mask)
    assert fix.points.shape[0] == 400 - 25


def test_fix_from_mask_rejects_a_mask_with_too_little_depth():
    mask = np.zeros((128, 128), dtype=bool)
    mask[0:3, 0:3] = True
    with pytest.raises(SkillError, match="too few"):
        fix_from_mask(_frame(), np.eye(4), mask)


# ── the fallback contract ────────────────────────────────────────────────

def test_segmenter_mask_is_preferred_when_it_works():
    mask = np.zeros((128, 128), dtype=bool)
    mask[50:70, 50:70] = True
    seg = _FakeSegmenter(mask=mask)
    fix = fix_at_pixel(_frame(), np.eye(4), u=60, v=60, segmenter=seg)
    assert seg.calls == 1
    assert 0.05 < float(max(fix.extent)) < 0.12


def test_broken_segmenter_falls_back_to_depth_instead_of_failing():
    """RPent's `segment` degrades to image inspection when its service is
    absent. A dead model must not take the grasp down with it."""
    seg = _FakeSegmenter(exc=RuntimeError("CUDA out of memory"))
    fix = fix_at_pixel(_frame(), np.eye(4), u=60, v=60, segmenter=seg)
    assert seg.calls == 1
    assert 0.05 < float(max(fix.extent)) < 0.12      # depth path still works


def test_empty_segmenter_mask_falls_back_to_depth():
    seg = _FakeSegmenter(mask=np.zeros((128, 128), dtype=bool))
    fix = fix_at_pixel(_frame(), np.eye(4), u=60, v=60, segmenter=seg)
    assert 0.05 < float(max(fix.extent)) < 0.12


def test_no_segmenter_is_the_depth_path():
    fix = fix_at_pixel(_frame(), np.eye(4), u=60, v=60, segmenter=None)
    assert 0.05 < float(max(fix.extent)) < 0.12


def test_degraded_run_still_refuses_to_return_a_wrong_pose():
    """The safety property that must survive the fallback.

    With a dead segmenter AND an object flush with the table, the depth path
    cannot separate them. It must raise rather than hand back the 34 cm-wrong
    centre that would drive the arm into the table.
    """
    flat = np.full((128, 128), 1.0, dtype=np.float32)
    seg = _FakeSegmenter(exc=RuntimeError("model gone"))
    with pytest.raises(SkillError, match="not separable by depth"):
        fix_at_pixel(_frame(flat), np.eye(4), u=64, v=64, segmenter=seg)


def test_skill_errors_from_the_segmenter_are_not_swallowed():
    """A SkillError is a deliberate, agent-readable message; hiding it behind
    a silent fallback would lose the explanation."""
    seg = _FakeSegmenter(exc=SkillError("segmenter returned no mask at (60, 60)"))
    with pytest.raises(SkillError, match="no mask"):
        fix_at_pixel(_frame(), np.eye(4), u=60, v=60, segmenter=seg)
