"""Live-rig smoke tests (deselected by default; run with `pytest -m hardware`).

Read-only: streams the camera and reads motor positions over CAN. Nothing
here moves the arm.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.hardware


def test_l515_streams_metric_depth():
    from wrc_demo.config import load_profile
    from wrc_demo.perception.camera_base import make_camera

    cam = make_camera(load_profile("cameras", "l515"))
    cam.open()
    try:
        cam.warm_up(10)
        f = cam.get_frame()
        assert f.rgb.ndim == 3 and f.has_depth
        valid = f.depth_m[f.depth_m > 0]
        assert valid.size > 1000
        # L515 range: metric meters, not raw units (0.25 mm scale bug guard).
        assert 0.25 <= np.median(valid) <= 2.5
        assert f.K[0, 0] > 100
    finally:
        cam.close()


def test_robstride_mech_pos_param_reads():
    """Read mechPos (0x7019) from every arm motor over can0. No motion."""
    from motorbridge import Controller

    ctrl = Controller("can0")
    models = ["rs-06", "rs-06", "rs-06", "rs-00", "rs-00", "rs-00", "rs-00"]
    positions = {}
    for motor_id, model in zip(range(1, 8), models):
        m = ctrl.add_robstride_motor(motor_id, 0xFD, model)
        positions[motor_id] = m.robstride_get_param_f32(0x7019)
    assert len(positions) == 7
    for mid, pos in positions.items():
        assert -13.0 < pos < 13.0, f"motor {mid} mechPos {pos} out of +-4pi"
    print("mechPos:", {k: round(v, 3) for k, v in positions.items()})
