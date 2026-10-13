# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the onboard-only estimator pieces (no simulator, no cuVSLAM)."""
import math

import numpy as np

from cascade.memory.robot_map import classify_frame
from cascade.sim import microduck_onboard as mo


class _NoVO:
    def __init__(self, name):
        self.n_obs = 0

    def start(self, T):
        pass


def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_ahrs_heading_under_skating_sway(monkeypatch):
    """Turning at 0.4 rad/s while the trunk rolls +-14 deg at 2 Hz (skating pushes): the AHRS heading must follow the
    true yaw (a scalar integration of the vertical gyro component drifts here)."""
    monkeypatch.setattr(mo, "OnboardLocalizer", _NoVO)
    dt = 0.02
    ob = mo.Onboard("t", dt, seed=1)
    ob.imu.gyro_bias[:] = 0.0  # this test isolates the integration scheme
    for _ in range(50):
        ob.imu_step(np.eye(3), np.zeros(3), np.zeros(3))
    ob.started = True
    yaw = 0.0
    for k in range(500):
        t = k * dt
        yaw += 0.4 * dt
        R = mo._Rz(yaw) @ _Rx(math.radians(14) * math.sin(2 * math.pi * 2 * t))
        ob.imu_step(R, np.zeros(3), np.zeros(3))
    err = math.degrees((ob.yaw_f - yaw + math.pi) % (2 * math.pi) - math.pi)
    assert abs(err) < 2.0, err
    assert ob.tilt_deg() < 20.0


def _beams_level(elev_deg):
    """64 beams of an 8x8 ToF looking along +x, rows top->bottom over +-22.5 deg elevation, tilted by elev_deg."""
    out = []
    for r in range(8):
        for c in range(8):
            el = math.radians(elev_deg + 22.5 - (r + 0.5) * 45 / 8)
            az = math.radians(22.5 - (c + 0.5) * 45 / 8)
            out.append([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.array(out)


def test_grazing_floor_zone_is_floor_and_wall_is_obstacle():
    p0 = np.array([0.0, 0.0, 0.17])
    d = _beams_level(-25.0)
    # floor hit along each zone's LOWEST ray (as a real ToF zone would report the nearest floor in its cone)
    lo = mo_tilt(d, -math.radians(45 / 16))
    rng = np.where(lo[:, 2] < -1e-3, -p0[2] / np.minimum(lo[:, 2], -1e-3), 5.0)
    status = np.where(rng < 4.0, 5, 255)
    mm = np.round(np.minimum(rng, 4.0) * 1000)
    kind, _, _ = classify_frame(mm, status, p0, d, floor_z=0.0)
    assert not np.any(kind == 2), "floor returns must not become obstacles"
    # a wall 0.5 m ahead
    rng_w = 0.5 / np.maximum(d[:, 0], 1e-3)
    kind_w, _, _ = classify_frame(np.round(rng_w * 1000), np.full(64, 5), p0, d, floor_z=0.0)
    # zones whose whole cone stays above 3 cm at the wall are obstacles; lower ones cannot be told from the floor
    assert np.all(kind_w.reshape(8, 8)[:2] == 2)


def mo_tilt(d, delta):
    from cascade.memory.robot_map import _tilt_beams

    return _tilt_beams(d, delta)


def test_floor_depth_image_matches_plane():
    T = np.eye(4)
    # camera at 0.2 m, optical z = world +x, optical x = world -y, optical y = world -z
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # columns: optical x, y, z in the world
    T[2, 3] = 0.2
    kind = np.ones(64, np.int8)
    depth = mo.tof_floor_depth_image(np.zeros(64), np.full(64, 5), T, kind)
    v = mo.CAM_H - 20
    u = mo.CAM_W // 2
    z = depth[v, u] / 1000.0
    if z > 0:  # inside the ToF field of view
        ray_down = (v - (mo.CAM_H - 1) / 2) / mo.CAM_FX
        assert abs(z * ray_down - 0.2) < 0.01
    assert depth[: mo.CAM_H // 2 - 5].max() == 0  # above the horizon: no floor depth
