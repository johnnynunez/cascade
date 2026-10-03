"""Explicit simulator-to-MicroDuck frames; no physics or policy side effects.

Newton body_qd is world-frame COM linear velocity followed by world angular
velocity. Public base position is the trunk origin, so its velocity is shifted
from COM before publication. All public quaternions in this module are wxyz.
"""
from __future__ import annotations

import numpy as np

from cascade.control.microduck_policy import POLICY_JOINTS


def _vector(value, size, name):
    try:
        raw = np.asarray(value)
        if raw.shape != (size,) or raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a numeric vector of length {size}")
        if any(isinstance(x, (bool, np.bool_)) for x in value):
            raise ValueError(f"{name} cannot contain booleans")
        result = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite {name}")
    return result


def body_frame_vectors(quaternion_wxyz, angular_velocity_world,
                       linear_velocity_com_world, com_local) -> dict:
    """Rotate gyro/gravity and shift a COM velocity to the observed root origin.

    com_local is the COM offset from that root in its local body frame. A
    backend already providing origin velocity must pass a zero offset, rather
    than applying the COM correction twice. No quaternion normalization masks
    a bad or wrongly ordered physics observation.
    """
    q = _vector(quaternion_wxyz, 4, "quaternion_wxyz")
    if not np.isclose(np.dot(q, q), 1.0, rtol=0, atol=2e-5):
        raise ValueError("orientation must be a unit wxyz quaternion")
    omega = _vector(angular_velocity_world, 3, "angular_velocity_world")
    com_velocity = _vector(linear_velocity_com_world, 3, "linear_velocity_com_world")
    offset = _vector(com_local, 3, "com_local")
    w, x, y, z = q
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return {
        "angular_velocity_body": rotation.T @ omega,
        "gravity_body": rotation.T @ np.array([0., 0., -1.]),
        "linear_velocity_origin_world": com_velocity - np.cross(omega, rotation @ offset),
    }


def newton_joint_indices(joint_labels, joint_q_start, joint_qd_start, *, root_path=None) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the admitted fourteen hinges without assuming backend ordering.

    Newton's start arrays include the terminal coordinate/DOF count. Verify
    each selected joint has exactly one coordinate and one velocity; passive,
    free and multi-axis joints never become policy actions by index slicing.
    """
    labels = tuple(joint_labels)
    if any(not isinstance(s, str) or not s or not s.rsplit("/", 1)[-1] for s in labels):
        raise ValueError("invalid Newton joint labels")
    if root_path is not None and (type(root_path) is not str or not root_path.startswith('/')
                                  or root_path.endswith('/')):
        raise ValueError("invalid robot root path")
    selected = [i for i, label in enumerate(labels)
                if root_path is None or label.startswith(root_path + '/')]
    names = [labels[i].rsplit("/", 1)[-1] for i in selected]
    if len(set(names)) != len(names) or not set(POLICY_JOINTS).issubset(names):
        raise ValueError("duplicate or missing Newton policy joints")
    starts = []
    for raw in (joint_q_start, joint_qd_start):
        values = np.asarray(raw)
        if (values.shape != (len(labels) + 1,) or values.dtype.kind not in "iu"
                or (values < 0).any() or (np.diff(values) < 0).any()):
            raise ValueError("invalid Newton joint start indices")
        starts.append(values.astype(np.int64))
    chosen = np.array([selected[names.index(name)] for name in POLICY_JOINTS], dtype=np.int64)
    for values in starts:
        if not np.all(values[chosen + 1] - values[chosen] == 1):
            raise ValueError("policy joints must be single-axis hinges")
    return starts[0][chosen].copy(), starts[1][chosen].copy()
