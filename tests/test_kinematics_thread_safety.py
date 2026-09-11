"""Real Pinocchio data must not cross perception/control threads."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from cascade.apps.demo import _build_arm
from cascade.config import load_demo_config


@pytest.mark.parametrize("method", ["fk", "link_positions"])
def test_parallel_kinematics_keeps_each_call_joint_pose(monkeypatch, method):
    cfg = load_demo_config(arm="isaac", cameras=["mock"], llm="mock")
    arm, _, kin = _build_arm(cfg.arm, True, None, cfg)
    qa = np.array([0., 1.2, 1.2, 0., .75, 0.])
    qb = np.array([-.779, .573, .542, -.824, .142, -.083])
    expected = getattr(kin, method)(qa)
    a_written, b_done = Event(), Event()
    forward = kin._pin.forwardKinematics

    def interleaved(model, data, q):
        forward(model, data, q)
        if np.array_equal(q[:kin.n], qa):
            a_written.set()
            assert b_done.wait(5), "parallel FK did not complete"

    monkeypatch.setattr(kin._pin, "forwardKinematics", interleaved)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(getattr(kin, method), qa)
            try:
                assert a_written.wait(5)
                pool.submit(kin.fk, qb).result(timeout=5)
            finally:
                b_done.set()
            np.testing.assert_allclose(a.result(timeout=5), expected, atol=1e-12)
    finally:
        arm.disconnect()
