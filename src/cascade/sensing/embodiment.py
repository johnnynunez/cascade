"""Passive structural binding: never converts units, refreshes clocks or actuates."""
from ..robotics.embodiment import EmbodimentDescriptor, JOINT_UNITS
from .hub import SensorError
from .models import JointStatePayload, ProprioceptionPayload


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
        joint_sensor = self.descriptor.modality in {"joint_state", "proprioception"}
        if joint_sensor != bool(self.attachment["joint_ids"]):
            raise ValueError("joint sensor requires exact attachment joint_ids")
        if self.descriptor.modality == "proprioception" and any(
                self._joints[name]["type"] == "prismatic" for name in self.attachment["joint_ids"]):
            raise ValueError("angular proprioception cannot describe a prismatic joint")

    def validate_payload(self, payload):
        names = self.attachment["joint_ids"]
        if isinstance(payload, JointStatePayload):
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

    def close(self):
        return self._provider.close()
