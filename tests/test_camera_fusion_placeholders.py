"""Real cameras never fuse 3D beliefs through a placeholder hand-eye.

A camera profile's `extrinsics:` block turns 3D fusion on by default
(apps/demo.py `_camera_fusion`). A hand-eye *record* is gated (a missing,
rejected or other-serial record turns fusion off), but an inline placeholder
T is trusted as given. For an eye-in-hand unit the shipped placeholder is
identity, i.e. "the camera frame IS the gripper frame", which is never true
on a real mount: once the arm moves, every object the wrist camera sees
would land centimetres away from where it is, and the planner would grasp
there. The Orbbec wrist profile already carries `fuse_beliefs: false` for
this reason; d455f_wrist (the reBot rig's wrist unit) did not.

These tests pin both the rule (for every real-hardware profile) and its
effect through the runtime's own `_camera_fusion`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CAMERAS = REPO / "configs" / "cameras"
REAL_TYPES = {"realsense", "orbbec"}


def _profiles():
    for f in sorted(CAMERAS.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text()) or {}
        if cfg.get("type") in REAL_TYPES and isinstance(cfg.get("extrinsics"), dict):
            yield f.stem, cfg


def _identity_placeholder(ext: dict) -> bool:
    T = ext.get("T")
    return not ext.get("hand_eye_json") and T is not None and np.allclose(np.asarray(T, float), np.eye(4))


@pytest.mark.parametrize("name,cfg", list(_profiles()), ids=lambda v: v if isinstance(v, str) else "")
def test_no_real_camera_fuses_through_an_identity_placeholder(name, cfg):
    ext = cfg["extrinsics"]
    if _identity_placeholder(ext):
        assert cfg.get("fuse_beliefs") is False, (
            f"{name}: identity placeholder extrinsics with 3D fusion on would place every "
            "detection where the gripper is, not where the object is")
        assert cfg.get("map_depth") is not True, f"{name}: occupancy through a placeholder"


def test_the_rig_wrist_camera_streams_but_does_not_fuse():
    """Through the runtime's own decision: overlay, heartbeat and images
    stay; beliefs and occupancy do not use it until it is calibrated."""
    from cascade.apps.demo import _camera_fusion
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="d455f_wrist", arm="rebot_rs", llm="mock")
    fuse, map_depth = _camera_fusion(cfg.camera, extrinsics=None)
    assert fuse is False
    assert not map_depth


def test_the_rig_scene_camera_fuses_through_its_accepted_record():
    """The fixed unit is calibrated on this rig (markerless, gated record
    bound to its serial): it is the camera that places objects."""
    from cascade.apps.demo import _camera_fusion
    from cascade.config import load_demo_config
    from cascade.perception.grounding import Extrinsics

    cfg = load_demo_config(camera="d455f_scene", arm="rebot_rs", llm="mock")
    ext = Extrinsics.from_config(cfg.camera.extrinsics, camera_serial=str(cfg.camera.serial))
    assert ext.calibrated, ext.calibration_error
    assert ext.source != "inline"
    fuse, _ = _camera_fusion(cfg.camera, ext)
    assert fuse is True
