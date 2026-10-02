"""CPU contracts for the optional native runner; no benchmark success claims."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.vab.native_preflight import PandaArm, PandaKinematics, PublicCamera, verify_move
from cascade.types import MotionHalted


def state(step=0, z=0.3):
    return {"epoch": "same-episode", "model_identity_sha256": "a" * 64,
            "step": step, "sim_time_s": step * .05,
            "tcp_xyz_m": [0.1, 0.0, z], "tcp_rotation": np.eye(3).tolist()}


def test_move_verifier_uses_actual_signed_displacement():
    assert verify_move(state(), state(10, .34), "up", .04).status == "confirmed"
    assert verify_move(state(), state(10, .26), "up", .04).status == "refuted"
    assert verify_move(state(), state(10, .30), "up", .04).status == "refuted"


@pytest.mark.parametrize("field,value", [("epoch", "reset"), ("model_identity_sha256", "b" * 64),
                                         ("step", 0), ("sim_time_s", 0.)])
def test_move_verifier_refuses_cross_episode_or_stalled_witness(field, value):
    after = state(10, .34)
    after[field] = value
    assert verify_move(state(), after, "up", .04).status == "unverified"


def test_move_verifier_rejects_lateral_or_orientation_error():
    after = state(10, .34)
    after["tcp_xyz_m"][0] += .01
    assert verify_move(state(), after, "up", .04).status == "refuted"
    after = state(10, .34)
    angle = .06
    after["tcp_rotation"] = [[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    assert verify_move(state(), after, "up", .04).status == "refuted"


class CartesianFixtureKin:
    def fk(self, q):
        result = np.eye(4)
        result[0, 3] = q[0]
        return result


def stepped_arm():
    arm = PandaArm.__new__(PandaArm)
    arm.kin = CartesianFixtureKin()
    arm._stopped = False
    arm.env = SimpleNamespace(trial=SimpleNamespace(control_freq=20),
                              sim=SimpleNamespace(data=SimpleNamespace(time=0.)))
    arm.q = np.zeros(7)
    arm.get_state = lambda: SimpleNamespace(q=arm.q.copy())
    arm.sent = []
    def send(target):
        arm.sent.append(target.copy())
        arm.q += (target - arm.q) * .5  # need settling, never perfect feedback
        arm.env.sim.data.time += .05
    arm.send_joint_target = send
    return arm


def test_stream_uses_physical_cadence_and_vets_feedback_and_settling():
    arm = stepped_arm()
    calls = []
    target = np.zeros(7)
    target[0] = .04
    assert arm.stream_to(target, .1, approve=lambda a, b, dt: calls.append((a.copy(), b.copy(), dt)))
    assert len(arm.sent) > 2
    assert len(calls) == 2 * len(arm.sent)
    assert all(np.isclose(call[2], .05) for call in calls)
    np.testing.assert_allclose(calls[1][1], arm.sent[0] * .5)
    np.testing.assert_allclose(calls[2][0], calls[1][1])


def test_stream_stop_refuses_the_next_step():
    arm = stepped_arm()
    original = arm.send_joint_target
    def send(target):
        original(target)
        arm._stopped = True
    arm.send_joint_target = send
    with pytest.raises(MotionHalted):
        arm.stream_to(np.ones(7) * .01, .1)
    assert len(arm.sent) == 1


def test_stream_refuses_a_changed_physics_clock():
    arm = stepped_arm()
    original = arm.send_joint_target
    def send(target):
        original(target)
        arm.env.sim.data.time += .01
    arm.send_joint_target = send
    with pytest.raises(RuntimeError, match="clock changed"):
        arm.stream_to(np.ones(7) * .01, .1)


def test_fk_and_ik_scratch_never_write_physics_data():
    mujoco = pytest.importorskip("mujoco")
    nested = ""
    for index in range(1, 8):
        nested += f'<body pos="0 0 .1"><joint name="robot0_joint{index}" axis="0 1 0" limited="true" range="-2 2"/><geom type="sphere" size=".01" mass="1"/>'
    nested += '<site name="gripper0_grip_site" pos="0 0 .1"/>' + '</body>' * 7
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody>' + nested + '</worldbody></mujoco>')
    physical = mujoco.MjData(model)
    physical.qpos[:] = .2
    mujoco.mj_forward(model, physical)
    saved = {key: np.asarray(getattr(physical, key)).copy() for key in ("qpos", "qvel", "xpos", "site_xpos", "ctrl")}
    scratch = mujoco.MjData(model)
    kin = PandaKinematics(model, scratch)
    target = kin.fk(np.ones(7) * .1)
    kin.link_positions(np.ones(7) * .3)
    kin.ik(target, np.ones(7) * .15, retries=1)
    for key, value in saved.items():
        np.testing.assert_array_equal(getattr(physical, key), value)
    assert physical.time == 0


def test_camera_orientation_and_depth_match_public_rgb_contract():
    camera = PublicCamera(None, res=128)
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[-1, 0] = [10, 20, 30]
    camera.publish({"agentview_image": rgb})
    assert camera.has_depth is False
    np.testing.assert_array_equal(camera._latest[0][0, 0], [30, 20, 10])
    assert camera._latest[1] is None
