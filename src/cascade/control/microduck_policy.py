"""Pinned MicroDuck feed-forward policy contract, CPU only.

Observation assembly/action mapping adapted from Pollen Robotics microduck_rl
(8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec) and microduck
(1fa84386f07884e27866411bc1ba166977bced95), licensed Apache-2.0.
See assets/microduck/NOTICE.md for attribution and separate asset terms.

All joint arrays are in POLICY_JOINTS order, radians and radians/second.
Gyro and projected unit gravity are already in the trunk frame: this module
neither rotates nor renormalizes them. Command is the 13-wide upstream block:
[vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll, body_x, body_y,
 body_z, body_roll, body_pitch, body_yaw]. Callers own command semantics;
some admitted weights encode phases instead of velocities. Admission of an
ONNX is not admission of its behavior for locomotion.

The graph owns observation normalization. No simulator, ONNX or network
package is imported until explicitly constructing a policy (ONNX Runtime
only). No weights are downloaded. Unknown/recurrent layouts fail closed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

POLICY_JOINTS: tuple[str, ...] = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw",
    "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
# Exact source HOME, not the rounded ONNX metadata. Immutable buffer.
HOME_Q = np.frombuffer(np.array([
    0., -.0873, -.4579, -.0049, .4530, .3491, .3491,
    0., 0., 0., .0873, .4579, .0049, -.4530,
], dtype=np.float32).tobytes(), dtype=np.float32)


def _vector(value, size: int, name: str) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector ({size},)") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite numeric vector ({size},)")
    return array


def policy_order(joint_names, values) -> np.ndarray:
    """Select named actuated joints from backend order, excluding mouth/passives.

    Missing or duplicate names are errors, including duplicates among extras.
    This selects values only; there is no sign, encoder or backlash correction.
    """
    names = tuple(joint_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint_names must contain nonempty names")
    if len(set(names)) != len(names) or not set(POLICY_JOINTS).issubset(names):
        raise ValueError("joint_names contain duplicates or miss policy joints")
    array = _vector(values, len(names), "joint values")
    return array[[names.index(name) for name in POLICY_JOINTS]].copy()


def observation(q, dq, angular_velocity_body, gravity_body, previous_action, command) -> np.ndarray:
    """Assemble unnormalized float32 (1,61); previous_action MUST be raw.

    q/dq are encoder-side measurements; no mouth or passive slots. For the
    initial non-backlash profile they are simply measured joint q/dq.
    """
    result = np.concatenate((
        _vector(angular_velocity_body, 3, "angular_velocity_body"),
        _vector(gravity_body, 3, "gravity_body"),
        _vector(q, 14, "q") - HOME_Q,
        _vector(dq, 14, "dq"),
        _vector(previous_action, 14, "previous_action"),
        _vector(command, 13, "command"),
    ))
    if not np.isfinite(result).all():
        raise ValueError("observation overflow")
    return result.reshape(1, 61)


class MicroduckPolicy:
    """Hash-bound CPU ONNX runner; no clipping, filtering or implicit history.

    infer() keeps its immediate-commit API. A fenced host instead uses
    preview() (no history mutation), then commit() only after the command is
    still valid. Discarded speculative outputs never become previous_action.
    reset() clears raw history, not a simulator or a command lease.
    """

    def __init__(self, path, expected_sha256, action_scale=1.0):
        if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("expected_sha256 must be an exact lowercase SHA-256")
        if isinstance(action_scale, (bool, np.bool_)) or not np.isscalar(action_scale):
            raise ValueError("action_scale must be positive and finite")
        self.action_scale = float(action_scale)
        if not np.isfinite(self.action_scale) or self.action_scale <= 0:
            raise ValueError("action_scale must be positive and finite")
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("policy SHA-256 mismatch")
        # Import optional runtime ONLY after admission. Execute the same verified
        # bytes, avoiding a second pathname read and external tensor sidecars.
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            data, sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if not self._matches(inputs, "obs", [1, 61]) or not self._matches(outputs, "actions", [1, 14]):
            raise ValueError("unsupported policy: require only obs float32[1,61] -> actions float32[1,14]")
        self.sha256 = expected_sha256
        self.reset()

    @staticmethod
    def _matches(nodes, name, shape):
        return (len(nodes) == 1 and nodes[0].name == name
                and nodes[0].type == "tensor(float)" and nodes[0].shape == shape)

    @property
    def previous_action(self) -> np.ndarray:
        return self._previous_action.copy()

    def infer(self, obs) -> np.ndarray:
        action = self.preview(obs)
        self.commit(action)
        return action

    def preview(self, obs) -> np.ndarray:
        """Evaluate the actual graph without committing its speculative output."""
        if not isinstance(obs, np.ndarray) or obs.shape != (1, 61) or obs.dtype != np.float32:
            raise ValueError("obs must be a float32 ndarray (1,61)")
        if not np.isfinite(obs).all():
            raise ValueError("nonfinite observation")
        outputs = self._session.run(["actions"], {"obs": np.ascontiguousarray(obs)})
        if (len(outputs) != 1 or not isinstance(outputs[0], np.ndarray)
                or outputs[0].shape != (1, 14) or outputs[0].dtype != np.float32
                or not np.isfinite(outputs[0]).all()):
            raise ValueError("invalid policy output; require finite float32[1,14]")
        return outputs[0][0].copy()

    def commit(self, action):
        """Pure bounded memory update, called at the host's actuation fence."""
        if (not isinstance(action, np.ndarray) or action.dtype != np.float32
                or action.shape != (14,) or not np.isfinite(action).all()):
            raise ValueError("committed action must be finite raw float32[14]")
        self._previous_action = action.copy()

    def targets(self, action) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            result = HOME_Q + self.action_scale * _vector(action, 14, "action")
        return _vector(result, 14, "targets").copy()

    def reset(self):
        self._previous_action = np.zeros(14, dtype=np.float32)
