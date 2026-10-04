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
# Exact source HOME, not the rounded ONNX metadata. Immutable buffers.
_HOME = (
    0., -.0873, -.4579, -.0049, .4530, .3491, .3491,
    0., 0., 0., .0873, .4579, .0049, -.4530,
)
HOME_Q = np.frombuffer(np.array(_HOME, dtype=np.float32).tobytes(), dtype=np.float32)
_ROBOTD_HOME = np.frombuffer(np.array(_HOME, dtype=np.float64).tobytes(), dtype=np.float64)
ROBOTD_SOURCE = '9136aa4ee88e81edf2bcaf3527e90b65da25f1eb'
TARGET_PROFILES = ('direct-v1', 'robotd-targets-v1')


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


class MicroduckTargets:
    """Output transform only; no model, command smoothing or actuator delay.

    The opt-in robotd profile follows control.rs at ROBOTD_SOURCE: f64 home
    plus 0.9 * raw f32 action, then head/leg target EMA. Its anchor advances
    only in commit(), never in preview(). Reset removes the anchor; the first
    target is unfiltered. Arrays use POLICY_JOINTS order (mouth excluded).
    """

    def __init__(self, profile='direct-v1', action_scale=None):
        if profile not in TARGET_PROFILES:
            raise ValueError('unknown MicroDuck target profile')
        expected = .9 if profile == 'robotd-targets-v1' else 1.
        scale = expected if action_scale is None else action_scale
        if isinstance(scale, (bool, np.bool_)) or not np.isscalar(scale):
            raise ValueError('action_scale must be positive and finite')
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('action_scale must be positive and finite')
        if profile == 'robotd-targets-v1' and scale != expected:
            raise ValueError('robotd-targets-v1 requires action_scale=0.9')
        self._profile, self._scale = profile, scale
        self.reset()

    @property
    def profile(self):
        return self._profile

    @property
    def action_scale(self):
        return self._scale

    @property
    def contract(self):
        robotd = self._profile == 'robotd-targets-v1'
        return {'profile': self._profile, 'action_scale': self._scale,
                'head_lowpass': .5 if robotd else None, 'legs_lowpass': .7 if robotd else None,
                'target_dtype': 'float64' if robotd else 'float32',
                'previous_action': 'raw_float32', 'first_target': 'unfiltered',
                'upstream_commit': ROBOTD_SOURCE if robotd else None,
                'scope': 'target_transform_only', 'physical_admission': False}

    @property
    def committed(self):
        return None if self._previous is None else self._previous.copy()

    def preview(self, action):
        raw = _vector(action, 14, 'action')
        with np.errstate(over='ignore', invalid='ignore'):
            if self._profile == 'direct-v1':
                result = HOME_Q + self._scale * raw
                return _vector(result, 14, 'targets').copy()
            result = _ROBOTD_HOME + self._scale * raw.astype(np.float64)
            if self._previous is not None:
                for i in range(14):
                    alpha = .5 if 5 <= i < 9 else .7
                    result[i] = alpha * result[i] + (1. - alpha) * self._previous[i]
        if not np.isfinite(result).all():
            raise ValueError('nonfinite targets')
        return result

    def commit(self, action):
        target = self.preview(action)
        self._previous = target
        return target.copy()

    def reset(self):
        self._previous = None


class MicroduckPolicy:
    """Hash-bound CPU ONNX runner with an explicit output-transform profile.

    infer() keeps its immediate-commit API. A fenced host instead uses
    preview() (no history mutation), then commit() only after the command is
    still valid. Discarded speculative outputs never become previous_action.
    targets() previews the NEXT target without mutation; after infer()/commit()
    use committed_targets for the accepted target. reset() clears raw and
    filter history, not a simulator, physical actuator delay or command lease.
    """

    def __init__(self, path, expected_sha256, action_scale=None, *, target_profile='direct-v1'):
        if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("expected_sha256 must be an exact lowercase SHA-256")
        self._targets = MicroduckTargets(target_profile, action_scale)
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

    @property
    def action_scale(self):
        return self._targets.action_scale

    @property
    def target_contract(self):
        return dict(self._targets.contract, policy_sha256=self.sha256)

    @property
    def committed_targets(self):
        if self._targets.profile == 'direct-v1':
            return self.targets(self._previous_action) if self._has_committed else None
        return self._targets.committed

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
        raw = action.copy()
        if self._targets.profile != 'direct-v1':
            self._targets.commit(raw)
        self._previous_action = raw
        self._has_committed = True

    def targets(self, action) -> np.ndarray:
        return self._targets.preview(action)

    def reset(self):
        self._previous_action = np.zeros(14, dtype=np.float32)
        self._targets.reset()
        self._has_committed = False
