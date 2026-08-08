"""Pixel-addressed grasping: acting on objects the detector cannot name.

VIA (arXiv:2607.11119, line 169) withholds perception APIs entirely and has the
agent click in an RGB-D point cloud, because a frontier model can already see
the object; what it lacks is a metric way to address it.

The measured motivation is in docs/SOTA_PERCEPTION_AND_EVALUATION.md: on LIBERO
frames the open-vocabulary detector emits 38-44 detections and never once says
`bowl`, so a label-addressed interface cannot express the task while the object
is plainly visible.

The property these tests protect is the one that is easy to get wrong: the
grasp target must be the object's CENTRE, not the probed surface point.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrc_demo.perception.pixel_target import (
    fix_from_pixel,
    segment_at_pixel,
)
from wrc_demo.types import Frame, SkillError

K = np.array([[200.0, 0.0, 64.0],
              [0.0, 200.0, 64.0],
              [0.0, 0.0, 1.0]])


def _scene() -> np.ndarray:
    """128x128 depth: table at 1.0 m, one 'box' face at 0.8 m."""
    d = np.full((128, 128), 1.0, dtype=np.float32)
    d[50:70, 50:70] = 0.8          # a raised object
    return d


def _frame(depth: np.ndarray) -> Frame:
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    return Frame(rgb=rgb, depth_m=depth, K=K, t=0.0, depth_source="sensor")


# ── segmentation ─────────────────────────────────────────────────────────

def test_flood_fill_stops_at_the_depth_step():
    """The region must be the object, not the whole table."""
    mask = segment_at_pixel(_scene(), u=60, v=60)
    assert mask[60, 60]
    assert mask.sum() == 400          # exactly the 20x20 patch
    assert not mask[40, 40]           # table not included


def test_fill_on_the_table_does_not_swallow_the_object():
    mask = segment_at_pixel(_scene(), u=10, v=10)
    assert mask[10, 10]
    assert not mask[60, 60]           # the raised patch is excluded


def test_region_is_capped():
    """A fill that escapes must not consume unbounded memory/time."""
    flat = np.full((128, 128), 1.0, dtype=np.float32)
    mask = segment_at_pixel(flat, u=64, v=64, max_px=500)
    assert mask.sum() <= 500


def test_missing_depth_is_a_clear_error():
    d = _scene()
    d[60, 60] = 0.0                   # depth hole
    with pytest.raises(SkillError, match="no depth"):
        segment_at_pixel(d, u=60, v=60)


def test_out_of_bounds_pixel_is_rejected():
    with pytest.raises(SkillError, match="outside"):
        segment_at_pixel(_scene(), u=500, v=500)


# ── lifting to a graspable fix ───────────────────────────────────────────

def test_fix_centre_is_the_region_centre_not_the_probed_pixel():
    """The property that makes this usable as a grasp target.

    A ray hits the FIRST surface, so probing an off-centre pixel returns a
    point near the object's edge. Feeding that into a grasp would grasp the
    rim. The OBB centre of the segmented region must be independent of which
    pixel on the object was clicked.
    """
    frame = _frame(_scene())
    T = np.eye(4)
    centre = fix_from_pixel(frame, T, u=60, v=60).position
    corner = fix_from_pixel(frame, T, u=52, v=52).position
    assert np.linalg.norm(centre - corner) < 1e-6


def test_fix_extent_matches_the_object_not_the_scene():
    frame = _frame(_scene())
    fix = fix_from_pixel(frame, np.eye(4), u=60, v=60)
    # 20 px at f=200 and z=0.8 m -> 0.08 m across.
    assert 0.05 < float(max(fix.extent)) < 0.12


def test_fix_carries_a_mask_and_bbox_like_a_detection():
    """Downstream code (grasp planning, workspace filter) treats this exactly
    like a detector output, so the shape of the payload must match."""
    fix = fix_from_pixel(_frame(_scene()), np.eye(4), u=60, v=60)
    assert fix.detection.mask is not None
    assert fix.detection.mask.shape == (128, 128)
    assert fix.detection.bbox.shape == (4,)
    assert fix.points.shape[1] == 3


def test_speckle_is_rejected_rather_than_grasped():
    """One stray pixel of depth must not become a grasp target."""
    d = np.full((128, 128), 1.0, dtype=np.float32)
    d[60, 60] = 0.5                   # a single isolated pixel
    with pytest.raises(SkillError, match="speckle"):
        fix_from_pixel(_frame(d), np.eye(4), u=60, v=60)


def test_extrinsics_are_applied():
    """The fix must be in the base frame, not the camera frame."""
    frame = _frame(_scene())
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    a = fix_from_pixel(frame, np.eye(4), u=60, v=60).position
    b = fix_from_pixel(frame, T, u=60, v=60).position
    assert np.allclose(b - a, [1.0, 2.0, 3.0], atol=1e-6)


def test_no_depth_camera_is_a_clear_error():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb, depth_m=None, K=K, t=0.0)
    with pytest.raises(SkillError, match="no depth"):
        fix_from_pixel(frame, np.eye(4), u=60, v=60)


def test_escaped_fill_fails_instead_of_returning_a_wrong_pose():
    """MEASURED on LIBERO: probing a bowl on a table fills the whole table.

    Depth connectivity cannot separate an object from the surface it touches:
    the depth step across the contact line is below DEPTH_STEP_M. The fill
    reached its 10000 px cap and the OBB centre landed 34 cm from truth with a
    102 cm extent.

    Returning that as a grasp target would drive the arm into the table, so a
    capped region must raise. Do NOT "fix" this by raising the cap: that makes
    the wrong answer bigger, not righter. The real fix is an appearance-based
    segmenter (SAM3), which is what ASPIRE, VoLo and RPent use.
    """
    flat = np.full((128, 128), 1.0, dtype=np.float32)   # object flush with table
    with pytest.raises(SkillError, match="not separable by depth"):
        fix_from_pixel(_frame(flat), np.eye(4), u=64, v=64)


def test_grasp_at_pixel_is_exposed_to_the_llm():
    """A skill missing from TOOL_SPECS is invisible; CLAUDE.md flags this trap."""
    from wrc_demo.skills.runtime import TOOL_SPECS

    assert "grasp_at_pixel" in {t["name"] for t in TOOL_SPECS}
