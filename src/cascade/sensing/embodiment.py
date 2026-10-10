"""Passive structural binding: never converts units, refreshes clocks or actuates."""
from ..robotics.embodiment import EmbodimentDescriptor, JOINT_UNITS
from ..robotics.joint_coordinates import coordinate_convention
from ..robotics.contracts import plain_json
from .hub import SensorError
from .models import GeneralizedJointStatePayload, JointStatePayload, ProprioceptionPayload


class EmbodimentBoundProvider:
    def __init__(self, provider, domain_id, embodiment):
        self._provider = provider
        self.descriptor = provider.descriptor
        self.embodiment = EmbodimentDescriptor.from_dict(embodiment)
        self.attachment = self.embodiment.sensor_attachment(f"{domain_id}/{self.descriptor.sensor_id}")
        if self.descriptor.robot_id != self.embodiment.robot_id:
            raise ValueError("sensor and embodiment robot identities disagree")
        if self.descriptor.frame_id != self.attachment["frame_id"]:
            raise ValueError("sensor frame and embodiment attachment disagree")
        self._joints = {j["joint_id"]: j for j in self.embodiment.joints}
        joint_sensor = self.descriptor.modality in {"joint_state", "proprioception", "generalized_joint_state"}
        if joint_sensor != bool(self.attachment["joint_ids"]):
            raise ValueError("joint sensor requires exact attachment joint_ids")
        if self.descriptor.modality == "proprioception" and any(
                self._joints[name]["type"] not in {"revolute", "continuous"} for name in self.attachment["joint_ids"]):
            raise ValueError("angular proprioception cannot describe prismatic or multi-DoF joints")
        if self.descriptor.modality == "joint_state" and any(
                self._joints[name]["type"] not in JOINT_UNITS for name in self.attachment["joint_ids"]):
            raise ValueError("scalar joint_state cannot describe multi-DoF joints")
        if self.descriptor.modality == "generalized_joint_state" and self.embodiment.version != 2:
            raise ValueError("generalized joint observations require embodiment version 2")

    def validate_payload(self, payload):
        names = self.attachment["joint_ids"]
        if payload.modality != self.descriptor.modality:
            raise SensorError("payload modality disagrees with embodiment sensor binding")
        if isinstance(payload, GeneralizedJointStatePayload):
            if payload.embodiment_sha256 != self.embodiment.sha256:
                raise SensorError("generalized joint state embodiment digest mismatch")
            if tuple(j.joint_id for j in payload.joints) != names:
                raise SensorError("generalized joint identities/order disagree with attachment")
            for measurement in payload.joints:
                joint = self._joints[measurement.joint_id]
                expected = joint.get("coordinates", coordinate_convention(joint["type"]))
                if measurement.joint_type != joint["type"] or plain_json(measurement.coordinates) != plain_json(expected):
                    raise SensorError("generalized joint type/coordinates disagree with embodiment")
        elif isinstance(payload, JointStatePayload):
            if any(self._joints[name]["type"] not in JOINT_UNITS for name in names):
                raise SensorError("scalar joint_state cannot describe multi-DoF joints")
            if payload.embodiment_sha256 != self.embodiment.sha256:
                raise SensorError("joint state embodiment digest mismatch")
            if tuple(j.joint_id for j in payload.joints) != names:
                raise SensorError("joint state identities/order disagree with attachment")
            for measurement in payload.joints:
                joint = self._joints[measurement.joint_id]
                if measurement.joint_type != joint["type"] or (
                        measurement.position_unit, measurement.velocity_unit, measurement.effort_unit) != JOINT_UNITS[joint["type"]]:
                    raise SensorError("joint state type/units disagree with embodiment")
        elif isinstance(payload, ProprioceptionPayload):
            if any(self._joints[name]["type"] not in {"revolute", "continuous"} for name in names):
                raise SensorError("angular proprioception cannot describe prismatic or multi-DoF joints")
            if payload.joint_names != names:
                raise SensorError("angular joint identities/order disagree with attachment")
        elif names:
            raise SensorError("joint attachment requires a joint observation")
        # Limit violations are observations, not malformed packets: retain them
        # for health/safety consumers instead of clipping or hiding evidence.

    def read(self):
        observation = self._provider.read()
        self.validate_payload(observation.payload)
        return observation

    def read_produced(self, after=None):
        """Producer history (B72) under the same structural binding as `read`."""
        method = getattr(self._provider, "read_produced", None)
        if not callable(method):
            raise SensorError("provider records no produced samples")
        observations = method(after)
        for observation in observations:
            self.validate_payload(observation.payload)
        return observations

    def close(self):
        return self._provider.close()
