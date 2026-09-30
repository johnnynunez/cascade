"""A reset selects a Newton free body by identity, even after its pose is lost."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from conftest import load_isaac_bridge_definitions


class Array:
    def __init__(self, data):
        self.data = np.asarray(data).copy()

    def numpy(self):
        return self.data.copy()

    def assign(self, data):
        self.data[:] = data


@pytest.fixture
def teleport(monkeypatch):
    # A fixed joint repeats offset zero. Joint order differs from body order,
    # and a revolute joint follows the free joints: neither positional search
    # nor deriving qd from the tail of q can identify the target reliably.
    identity = [0., 0., 0., 0., 0., 0., 1.]
    model = SimpleNamespace(
        body_label=["/World_Props/green_cube", "/Robot/base", "/World_Props/orange", "/Robot/link"],
        joint_child=Array([1, 2, 0, 3]),
        joint_type=Array([3, 4, 4, 1]),
        joint_q_start=Array([0, 0, 7, 14, 15]),
        joint_qd_start=Array([0, 0, 6, 12, 13]),
        joint_parent=Array([-1, -1, -1, 1]),
        joint_X_p=Array([identity] * 4),
        joint_X_c=Array([identity] * 4),
    )

    def state():
        return SimpleNamespace(
            joint_q=Array(np.array([.1, .2, .3, 0., 0., 0., 1.] * 2 + [.7], np.float32)),
            joint_qd=Array(np.arange(13, dtype=np.float32) + 1),
            body_q=Array(np.array([[.1, .2, .3, 0., 0., 0., 1.]] * 4, np.float32)),
            body_qd=Array(np.arange(24, dtype=np.float32).reshape(4, 6)),
        )

    stage = SimpleNamespace(model=model, state_0=state(), state_1=state())
    parent = None
    for name in ("isaacsim", "isaacsim.physics", "isaacsim.physics.newton",
                 "isaacsim.physics.newton.impl", "isaacsim.physics.newton.impl.extension"):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[-1], module)
        parent = module
    parent._newton_stage = stage
    monkeypatch.setitem(sys.modules, "newton", SimpleNamespace(JointType=SimpleNamespace(FREE=4)))
    env = {"np": np, "BASE_Z": .8}
    load_isaac_bridge_definitions(["_newton_teleport"], env)
    return env["_newton_teleport"], stage


def snapshot(stage):
    return [[getattr(state, name).numpy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")]
            for state in (stage.state_0, stage.state_1)]


@pytest.mark.parametrize("old_pose", ["coincident", "displaced", "nan"])
def test_reset_uses_identity_and_only_changes_target_in_both_buffers(teleport, old_pose):
    reset, stage = teleport
    for state in (stage.state_0, stage.state_1):
        if old_pose == "displaced":
            # Fabric/body pose need not match joint coordinates.
            state.body_q.data[0, :3] = [9., 8., -88.]
        elif old_pose == "nan":
            state.body_q.data[0] = np.nan
            state.body_qd.data[0] = np.nan
            state.joint_q.data[7:14] = np.nan
            state.joint_qd.data[6:12] = np.nan
    before = snapshot(stage)
    assert reset("green_cube", [.4, -.2, .03])
    expected_pose = [.4, -.2, .83, 0., 0., 0., 1.]
    for state, original in zip((stage.state_0, stage.state_1), before):
        np.testing.assert_allclose(state.joint_q.data[7:14], expected_pose)
        np.testing.assert_allclose(state.body_q.data[0], expected_pose)
        np.testing.assert_array_equal(state.joint_qd.data[6:12], 0.)
        np.testing.assert_array_equal(state.body_qd.data[0], 0.)
        np.testing.assert_array_equal(state.joint_q.data[np.r_[:7, 14:15]], original[0][np.r_[:7, 14:15]])
        np.testing.assert_array_equal(state.joint_qd.data[np.r_[:6, 12:13]], original[1][np.r_[:6, 12:13]])
        np.testing.assert_array_equal(state.body_q.data[1:], original[2][1:])
        np.testing.assert_array_equal(state.body_qd.data[1:], original[3][1:])


@pytest.mark.parametrize("target", [[np.nan, 0., 0.], [0., np.inf, 0.], [0., 0.], [[0., 0., 0.]]])
def test_invalid_target_does_not_write_either_buffer(teleport, target):
    reset, stage = teleport
    before = snapshot(stage)
    assert not reset("green_cube", target)
    for original, actual in zip(before, snapshot(stage)):
        for a, b in zip(original, actual):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bad_layout", ["unknown", "ambiguous", "wrong_span", "parent", "frame", "short_state_1"])
def test_incompatible_identity_or_layout_fails_before_any_write(teleport, bad_layout):
    reset, stage = teleport
    name = "green_cube"
    if bad_layout == "unknown":
        name = "missing"
    elif bad_layout == "ambiguous":
        stage.model.joint_child.data[1] = 0
    elif bad_layout == "wrong_span":
        stage.model.joint_qd_start.data[3] = 11
    elif bad_layout == "parent":
        stage.model.joint_parent.data[2] = 1
    elif bad_layout == "frame":
        stage.model.joint_X_p.data[2, 0] = .1
    elif bad_layout == "short_state_1":
        stage.state_1.joint_qd = Array(np.zeros(10, np.float32))
    before = snapshot(stage)
    assert not reset(name, [.4, -.2, .03])
    for original, actual in zip(before, snapshot(stage)):
        for a, b in zip(original, actual):
            np.testing.assert_array_equal(a, b)
