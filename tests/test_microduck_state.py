"""Pins backend-to-policy frames, not policy self-reported motion."""
from __future__ import annotations
import importlib
import importlib.util
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec("cascade.sim.microduck_state") is not None, "frame adapter is missing"
    return importlib.import_module("cascade.sim.microduck_state")


def test_identity_body_frame_gravity_and_gyro():
    out = module().body_frame_vectors([1.,0.,0.,0.], [0.1,0.2,0.3], [1.,2.,3.], [0.,0.,0.])
    np.testing.assert_allclose(out["angular_velocity_body"], [0.1,0.2,0.3])
    np.testing.assert_allclose(out["gravity_body"], [0.,0.,-1.])
    np.testing.assert_allclose(out["linear_velocity_origin_world"], [1.,2.,3.])


def test_yaw_half_turn_rotates_gyro_but_not_world_velocity():
    out = module().body_frame_vectors([0.,0.,0.,1.], [1.,2.,3.], [2.,1.,0.], [0.,0.,0.])
    np.testing.assert_allclose(out["angular_velocity_body"], [-1.,-2.,3.])
    np.testing.assert_allclose(out["gravity_body"], [0.,0.,-1.])
    np.testing.assert_allclose(out["linear_velocity_origin_world"], [2.,1.,0.])


def test_com_velocity_is_shifted_to_trunk_origin_before_publication():
    out = module().body_frame_vectors([0.,0.,0.,1.], [0.,0.,2.], [1.,2.,3.], [0.05,0.,0.])
    np.testing.assert_allclose(out["linear_velocity_origin_world"], [1.,2.1,3.])


def test_tilted_body_gravity_is_not_the_authored_world_direction():
    out = module().body_frame_vectors([0.,1.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.])
    np.testing.assert_allclose(out["gravity_body"], [0.,0.,1.])


@pytest.mark.parametrize("quaternion", [[0.,0.,0.,0.], [2.,0.,0.,0.], [float("nan"),0.,0.,0.], [1.,0.,0.], [True,0.,0.,0.]])
def test_bad_quaternion_is_not_silently_normalized(quaternion):
    with pytest.raises(ValueError):
        module().body_frame_vectors(quaternion, [0.,0.,0.], [0.,0.,0.], [0.,0.,0.])


def test_newton_joint_maps_are_by_name_not_articulation_traversal_order():
    from cascade.control.microduck_policy import POLICY_JOINTS
    labels = ["free"] + ["/World/Robot/" + n for n in reversed(POLICY_JOINTS)]
    q_starts = [0] + list(range(7, 22))
    d_starts = [0] + list(range(6, 21))
    q, d = module().newton_joint_indices(labels, q_starts, d_starts)
    np.testing.assert_array_equal(q, list(range(20, 6, -1)))
    np.testing.assert_array_equal(d, list(range(19, 5, -1)))


def test_duplicate_or_missing_joint_cannot_be_assigned_a_policy_slot():
    from cascade.control.microduck_policy import POLICY_JOINTS
    labels = ["free"] + list(POLICY_JOINTS)
    labels[-1] = labels[-2]
    with pytest.raises(ValueError):
        module().newton_joint_indices(labels, [0] + list(range(7, 22)), [0] + list(range(6, 21)))


def test_multiaxis_joint_is_not_treated_as_a_single_hinge():
    from cascade.control.microduck_policy import POLICY_JOINTS
    labels = ["free"] + list(POLICY_JOINTS)
    q_starts = [0] + list(range(7, 22))
    q_starts[-1] = 22
    with pytest.raises(ValueError):
        module().newton_joint_indices(labels, q_starts, [0] + list(range(6, 21)))
