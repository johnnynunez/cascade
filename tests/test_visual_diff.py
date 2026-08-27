"""Tests for CaP-X visual differencing as an independent verification channel.

The contract: on real hardware `belief` is written by the same perception pass
the skill just ran, so agreeing with it proves nothing. Pixels come from a
camera the actuator does not control, so they are genuinely independent --
but only when the frame is readable. An honest UNKNOWN beats a confident
guess, because a verifier that always answers is exactly the bug being fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.agent.effects import (
    CONFIRMED,
    REFUTED,
    UNVERIFIED,
    PostconditionChecker,
)
from cascade.perception.visual_diff import (
    CHANGED,
    UNCHANGED,
    UNKNOWN,
    VisualDiffChannel,
    frames_differ_globally,
)

H, W = 240, 320


def _scene(blob_at=None, noise=3, seed=0):
    """Flat grey frame plus optional bright blob, with mild sensor noise."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 110, dtype=np.uint8)
    img = np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape), 0, 255)
    img = img.astype(np.uint8)
    if blob_at is not None:
        u, v = blob_at
        img[max(0, v - 25):v + 25, max(0, u - 25):u + 25] = 240
    return img


#: identity projector: 3D (x, y, z) -> pixel (x, y), so tests can reason in px
def _proj(xyz):
    return (float(xyz[0]), float(xyz[1]))


# ── the core signal ──────────────────────────────────────────────────────


def test_object_left_source_and_arrived_at_target():
    before = _scene(blob_at=(80, 80))
    after = _scene(blob_at=(240, 160), seed=1)
    v = VisualDiffChannel(_proj).compare(before, after, (80, 80, 0), (240, 160, 0))
    assert v.status == CHANGED
    assert v.source_roi == CHANGED and v.target_roi == CHANGED


def test_untouched_scene_reads_as_unchanged():
    before = _scene(blob_at=(80, 80))
    after = _scene(blob_at=(80, 80), seed=7)   # same layout, different noise
    v = VisualDiffChannel(_proj).compare(before, after, (80, 80, 0), (240, 160, 0))
    assert v.status == UNCHANGED
    assert v.source_roi == UNCHANGED and v.target_roi == UNCHANGED


def test_sensor_noise_alone_does_not_register_as_change():
    """The whole point of a threshold: noise must not look like motion."""
    assert frames_differ_globally(_scene(seed=1), _scene(seed=2)) < 0.01


# ── the pitfall: the arm is in frame ─────────────────────────────────────


def test_arm_sweeping_through_the_view_abstains_instead_of_confirming():
    """A naive diff confirms everything when the gripper is parked in frame."""
    before = _scene(blob_at=(80, 80))
    after = _scene(blob_at=(80, 80), seed=3)
    after[:, 40:260] = 20          # a big dark arm across most of the frame
    v = VisualDiffChannel(_proj).compare(before, after, (80, 80, 0), (240, 160, 0))
    assert v.status == UNKNOWN
    assert "still in view" in v.detail
    assert v.global_change > 0.45


def test_out_of_view_roi_is_unknown_not_unchanged():
    before, after = _scene(), _scene(seed=4)
    v = VisualDiffChannel(_proj).compare(before, after, (9999, 9999, 0), None)
    assert v.source_roi == UNKNOWN


def test_missing_frames_abstain():
    v = VisualDiffChannel(_proj).compare(None, _scene(), (10, 10, 0))
    assert v.status == UNKNOWN and "no before/after" in v.detail


def test_frame_size_change_abstains():
    v = VisualDiffChannel(_proj).compare(_scene(), np.zeros((10, 10, 3), np.uint8))
    assert v.status == UNKNOWN and "frame size changed" in v.detail


def test_no_projector_degrades_to_unknown_regions():
    before = _scene(blob_at=(80, 80))
    after = _scene(blob_at=(240, 160), seed=1)
    v = VisualDiffChannel(project=None).compare(before, after, (80, 80, 0), (240, 160, 0))
    assert v.source_roi == UNKNOWN and v.target_roi == UNKNOWN


# ── integration: the tautology this exists to break ──────────────────────


def _placement_args():
    """The exact shape that produced the live-rig false confirmation."""
    return (
        "pick_and_place",
        {"object": "cubo rosa", "destination": "caja"},
        {"picked": "cubo rosa", "placed_at": [240, 160, 0], "ok": True},
        {"label": "cubo rosa", "pose": [80, 80, 0]},
    )


def test_belief_only_placement_is_unverified_without_pixels():
    """Baseline: no visual channel -> honest abstain (the 2026-07-31 fix)."""
    ck = PostconditionChecker(belief_pose=lambda l: [240, 160, 0])
    skill, args, result, before = _placement_args()
    pc = ck.verify(skill, args, result, before=before)
    assert pc.status == UNVERIFIED
    assert "no independent confirmation" in pc.evidence


def test_pixels_promote_a_belief_only_placement_to_confirmed():
    b = _scene(blob_at=(80, 80))
    a = _scene(blob_at=(240, 160), seed=1)
    diff = VisualDiffChannel(_proj)
    ck = PostconditionChecker(
        belief_pose=lambda l: [240, 160, 0],
        visual_diff=lambda src, tgt: diff.compare(b, a, src, tgt),
    )
    skill, args, result, before = _placement_args()
    pc = ck.verify(skill, args, result, before=before)
    assert pc.status == CONFIRMED
    assert pc.channel == "visual_diff"
    assert "corroborated by pixels" in pc.evidence


def test_pixels_refute_a_placement_the_world_model_believed():
    """The valuable case: the belief says moved, the camera says nothing did."""
    b = _scene(blob_at=(80, 80))
    a = _scene(blob_at=(80, 80), seed=9)      # nothing actually moved
    diff = VisualDiffChannel(_proj)
    ck = PostconditionChecker(
        belief_pose=lambda l: [240, 160, 0],   # world model is confidently wrong
        visual_diff=lambda src, tgt: diff.compare(b, a, src, tgt),
    )
    skill, args, result, before = _placement_args()
    pc = ck.verify(skill, args, result, before=before)
    assert pc.status == REFUTED
    assert pc.channel == "visual_diff"
    assert "camera sees no change" in pc.evidence


def test_unreadable_pixels_fall_back_to_unverified_not_confirmed():
    """When the arm blocks the view we must NOT upgrade the tautology."""
    b = _scene(blob_at=(80, 80))
    a = _scene(blob_at=(80, 80), seed=5)
    a[:, 40:260] = 20                          # arm in frame
    diff = VisualDiffChannel(_proj)
    ck = PostconditionChecker(
        belief_pose=lambda l: [240, 160, 0],
        visual_diff=lambda src, tgt: diff.compare(b, a, src, tgt),
    )
    skill, args, result, before = _placement_args()
    pc = ck.verify(skill, args, result, before=before)
    assert pc.status == UNVERIFIED


def test_physics_channel_still_wins_and_ignores_pixels():
    """Sim keeps its exact channel; visual diff is the real-hardware fallback."""
    ck = PostconditionChecker(
        object_pose=lambda l: [240, 160, 0],
        visual_diff=lambda src, tgt: pytest.fail("physics should not need pixels"),
    )
    skill, args, result, before = _placement_args()
    pc = ck.verify(skill, args, result, before=before)
    assert pc.status == CONFIRMED and pc.channel == "physics"


def test_a_broken_visual_channel_never_crashes_verification():
    def _boom(src, tgt):
        raise RuntimeError("camera died")

    ck = PostconditionChecker(belief_pose=lambda l: [240, 160, 0], visual_diff=_boom)
    skill, args, result, before = _placement_args()
    pc = ck.verify(skill, args, result, before=before)
    assert pc.status == UNVERIFIED       # degrades, does not explode
