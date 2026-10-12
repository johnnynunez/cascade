"""OVRTX replay of recorded MicroDuck physics: transform math, the first-write sentinel, and the SDK pair gate.
CPU only (no OVRTX runtime needed: the SDK gate is checked before any native call)."""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_replay_local_transforms_and_look_at():
    from cascade.sim.microduck_ovrtx_replay import ReplayPoses, look_at, pose_matrix

    layout = {"bodies": {"r": {"trunk_base": "/World/Ducks/r/Geometry/trunk_base",
                               "neck": "/World/Ducks/r/Geometry/trunk_base/neck"}},
              "parents": {"/World/Ducks/r/Geometry/trunk_base": ("static", np.eye(4).tolist()),
                          "/World/Ducks/r/Geometry/trunk_base/neck": ("body", "/World/Ducks/r/Geometry/trunk_base")}}
    rp = ReplayPoses(layout, ["trunk_base", "neck"], ["r"])
    assert rp.paths[0] == "/World/Ducks/r/Geometry/trunk_base"  # no sentinel declared -> none written
    with_sentinel = ReplayPoses({**layout, "sentinel": "/World/S"}, ["trunk_base", "neck"], ["r"])
    assert with_sentinel.paths[0] == "/World/S"  # written FIRST (ovrtx 0.5.1 first-query workaround)
    q = [0, 0, math.sin(0.3), math.cos(0.3)]
    frame = np.array([[[1, 2, 0.1, *q], [1.0, 2.1, 0.2, *q]]], np.float32)
    loc = rp.transforms(frame)
    world_neck = loc["/World/Ducks/r/Geometry/trunk_base"] @ loc["/World/Ducks/r/Geometry/trunk_base/neck"]
    assert np.allclose(world_neck, pose_matrix(frame[0, 1]), atol=1e-5)
    assert np.allclose(with_sentinel.transforms(frame)["/World/S"], np.eye(4))
    T = look_at([0, -2, 1], [0, 0, 0])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9)
    assert T[:3, 2] @ (np.array([0, 2, -1]) / math.sqrt(5)) == pytest.approx(1.0)


def test_ovrtx_sdk_pair_must_be_known(tmp_path):
    from cascade.sim.ovrtx_renderer import CameraSpec, OvrtxError, OvrtxRenderer

    scene = tmp_path / "s.usda"
    scene.write_text("#usda 1.0\n")
    cam = [CameraSpec("c", 8, 8, 10.0, 10.0, np.eye(4))]
    OvrtxRenderer(scene, cam, sdk_versions=(("ovrtx", "0.5.1.385782"), ("ovstage", "0.2.1.385922")))
    OvrtxRenderer(scene, cam)  # pinned default
    with pytest.raises(OvrtxError):
        OvrtxRenderer(scene, cam, sdk_versions=(("ovrtx", "9.9"), ("ovstage", "9.9")))
