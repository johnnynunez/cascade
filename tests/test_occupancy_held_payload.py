"""A held object is the robot's payload, not an obstacle, in the mesh-masked map.

The capsule body mask (6 cm around the joint chain, TCP included) used to
hide a held object as a side effect. The URDF mesh mask (`body_mask: mesh`,
on for the reBot) is tight around the links: with the jaws closed on an
object and the arm idle (the measured opening poses the fingers), the object
between and beyond the fingers is integrated into the map at the TCP, and
the next motion -- the place -- meets an obstacle at 0 m from its TCP.
Payload tracking (`occupancy.track_payload`) is the Isaac-only capture
contract and is refused together with the mesh mask.

`body_mask.held_payload` is an envelope box in the end-effector frame that
the mask also draws while the runtime holds something.
"""

import numpy as np
import pytest
from test_occupancy_mesh_body_mask import (
    H,
    K,
    W,
    _Bridge,
    _frame,
    _occupancy,
    _rebot_profile,
    look_at,
    needs_rs,
)

CAMERA = look_at((0.25, -0.75, 0.45), (0.25, 0.0, 0.20))


def _held_cube_pixels(kin, q, size=0.08, ahead=0.05):
    """Pixels of a cube held in the jaws (a cup grasped near its rim): centred
    on the jaw axis, reaching `ahead` m past the fingertips, rendered as its
    surface points through the camera."""
    T = kin.fk(q)
    g = np.linspace(-size / 2, size / 2, 15)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    local = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    local = local[np.any(np.abs(local) >= size / 2 - 1e-9, axis=1)]        # the surface
    local[:, 0] += ahead - size / 2                    # TCP x is the approach: straddling the tips
    world = local @ T[:3, :3].T + T[:3, 3]
    cam = (world - CAMERA[:3, 3]) @ CAMERA[:3, :3]
    uv = np.round(cam[:, :2] / cam[:, 2:] * K[0, 0] + K[:2, 2]).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (cam[:, 2] > 0)
    px = np.zeros((H, W), dtype=bool)
    px[uv[ok, 1], uv[ok, 0]] = True
    return px


def _holding_rebot(**block):
    from cascade.apps import demo

    mesh = {"mode": "mesh", **block}
    cfg, acfg = _rebot_profile(body_mask=mesh, type="mock")
    bridge = _Bridge()
    occ = _occupancy(bridge)
    arm, safe, kin = demo._build_arm(acfg, False, occ, cfg)
    # jaws closed on an 8 cm object (profile: closed 0 rad, open 6.2 rad = 9 cm)
    arm._gripper = 6.2 * 0.08 / 0.09
    return arm, safe, kin, occ, bridge


ENVELOPE = {"box": [0.15, 0.13, 0.13], "xyz": [0.025, 0.0, 0.0]}


@needs_rs
def test_premise_the_mesh_mask_leaves_a_held_object_in_the_map():
    arm, _safe, kin, occ, bridge = _holding_rebot()
    cube = _held_cube_pixels(kin, arm.get_state().q)
    occ.refresh(_frame(np.full((H, W), 1.0, np.float32)), CAMERA)
    assert occ.last_error is None
    shipped = bridge.sent("integrate_depth")[-1]["depth"]
    assert cube.sum() > 200 and (shipped[cube] > 0).sum() > 50   # integrated: an obstacle at the TCP


@needs_rs
def test_while_holding_the_payload_envelope_masks_the_held_object():
    arm, safe, kin, occ, bridge = _holding_rebot(held_payload=ENVELOPE)
    holding = {"now": True}
    safe.body_mask.holding_fn = lambda: holding["now"]
    cube = _held_cube_pixels(kin, arm.get_state().q)
    occ.refresh(_frame(np.full((H, W), 1.0, np.float32)), CAMERA)
    assert occ.last_error is None
    shipped = bridge.sent("integrate_depth")[-1]["depth"]
    assert (shipped[cube] == 0).all()
    # empty-handed, the envelope is not drawn: the space in front of the tips stays visible
    holding["now"] = False
    occ.refresh(_frame(np.full((H, W), 1.0, np.float32)), CAMERA)
    shipped = bridge.sent("integrate_depth")[-1]["depth"]
    assert (shipped[cube] > 0).sum() > 50


@needs_rs
def test_a_failing_holding_read_refuses_the_frame():
    _arm, safe, _kin, occ, bridge = _holding_rebot(held_payload=ENVELOPE)

    def broken():
        raise RuntimeError("runtime state unreadable")

    safe.body_mask.holding_fn = broken
    occ.refresh(_frame(np.full((H, W), 1.0, np.float32)), CAMERA)
    assert "runtime state unreadable" in (occ.last_error or "")
    assert not bridge.sent("integrate_depth")


@pytest.mark.parametrize("bad, why", [
    ({"box": [0.1, 0.1]}, "held_payload box"), ({"box": [0.1, -0.1, 0.1]}, "positive"),
    ({"box": [0.1, 0.1, 0.1], "link": "gripper_end"}, "unknown keys"), ("big", "mapping"),
])
def test_a_bad_held_payload_block_is_refused(bad, why):
    from cascade.config import Cfg
    from cascade.perception.robot_mask import body_mask_settings

    with pytest.raises(ValueError, match=why):
        body_mask_settings(Cfg({"body_mask": {"mode": "mesh", "held_payload": bad}}))


def test_the_runtime_reports_holding_for_the_mask():
    from types import SimpleNamespace

    from cascade.apps.demo import _wire_held_payload

    mesh = SimpleNamespace(holding_fn=None, has_held_payload=True)
    runtime = SimpleNamespace(held_object=None, _held_provisional=None,
                              arm_rig=None, arm=SimpleNamespace(body_mask=mesh))
    _wire_held_payload(runtime)
    assert mesh.holding_fn() is False
    runtime.held_object = "green object"
    assert mesh.holding_fn() is True
    runtime.held_object, runtime._held_provisional = None, ("green object", "cube", "green")
    assert mesh.holding_fn() is True


def test_the_rebot_profile_declares_a_held_payload_envelope():
    from cascade.config import load_demo_config
    from cascade.perception.robot_mask import body_mask_settings

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    settings = body_mask_settings(cfg.arm)
    size, T = settings["held_payload"]
    # it must reach past the fingertips and cover the widest jaw opening
    max_w = float(cfg.arm.gripper.max_width_m)
    assert size[1] >= max_w and T[0, 3] + size[0] / 2 >= 0.05


@needs_rs
def test_build_runtime_wires_the_runtimes_holding_state_into_the_mask(tmp_path, monkeypatch):
    from occupancy_grid_bridge import GridBridge, make_map

    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception import occupancy as occ_mod

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    cfg.arm._data["type"] = "mock"                     # never a real arm
    monkeypatch.setattr(occ_mod.OccupancyMap, "from_config",
                        classmethod(lambda cls, *a, **k: make_map(GridBridge(), required=False)))
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run")
    try:
        mesh = runtime.arm.body_mask
        assert mesh is not None and mesh.holding_fn is not None
        assert mesh.holding_fn() is False
        runtime.held_object = "green object"
        assert mesh.holding_fn() is True
    finally:
        runtime.held_object = None
        demo.shutdown_runtime(runtime, arm)
