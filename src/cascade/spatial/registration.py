"""Explicit rigid camera-to-base registration; never invent localization bounds."""
from dataclasses import dataclass, replace
import json
import math

import numpy as np

from ..robotics.contracts import freeze_json, identifier, plain_json
from ..sensing.models import digest, number, vector
from .frames import TransformSample, sha256


@dataclass(frozen=True, init=False)
class CameraBaseRegistration:
    """T_camera_base maps base coordinates into the calibrated optical frame.

    The content hash binds an explicit declaration, not proof of calibration.
    Only a rigid camera mount is supported: moving head/joint extrinsics require
    an independently time-aligned kinematic provider instead.
    """
    record: object
    sha256: str

    def __init__(self, config, *, robot_id, descriptor, map_frame_id):
        fields = {"schema", "robot_id", "model_identity_sha256", "camera_calibration_sha256",
                  "camera_frame_id", "base_frame_id", "translation_m", "rotation_wxyz",
                  "position_error_m", "angular_error_rad"}
        if not isinstance(config, dict) or set(config) != fields | {"sha256"}:
            raise ValueError("explicit hashed rigid camera/base registration required")
        # JSON round-trip detaches containers and scalar subclasses before the
        # immutable contract is shared. The hash covers the exact JSON values.
        record = json.loads(json.dumps({k: v for k, v in config.items() if k != "sha256"}, allow_nan=False))
        if digest(config["sha256"]) is None or sha256(record) != config["sha256"]:
            raise ValueError("camera/base registration SHA256 mismatch")
        if record["schema"] != "cascade.rigid-camera-base.v1":
            raise ValueError("unsupported camera/base registration schema")
        for name in ("robot_id", "camera_frame_id", "base_frame_id"):
            identifier(record[name], name)
        if (digest(record["model_identity_sha256"]) is None
                or digest(record["camera_calibration_sha256"]) is None
                or record["robot_id"] != robot_id
                or record["model_identity_sha256"] != descriptor.model_identity_sha256
                or record["camera_calibration_sha256"] != descriptor.calibration_id
                or record["camera_frame_id"] != descriptor.frame_id
                or record["base_frame_id"] in {descriptor.frame_id, map_frame_id}):
            raise ValueError("camera/base registration robot/model/calibration/frame mismatch")
        vector(record["translation_m"], 3, "camera_from_base translation")
        q = vector(record["rotation_wxyz"], 4, "camera_from_base rotation")
        if abs(sum(v*v for v in q)-1.) > 1e-6:
            raise ValueError("camera/base registration rotation must be unit quaternion")
        for name in ("position_error_m", "angular_error_rad"):
            if record[name] is not None:
                number(record[name], name, minimum=0)
        if record["angular_error_rad"] is not None and record["angular_error_rad"] > math.pi:
            raise ValueError("registration angular error cannot exceed pi")
        object.__setattr__(self, "record", freeze_json(record))
        object.__setattr__(self, "sha256", str(config["sha256"]))

    @property
    def base_frame_id(self):
        return self.record["base_frame_id"]

    def pose(self, camera_pose):
        r = self.record
        if (not isinstance(camera_pose, TransformSample) or camera_pose.static
                or camera_pose.child != r["camera_frame_id"]
                or camera_pose.stamp.calibration_id != r["camera_calibration_sha256"]
                or camera_pose.parent == self.base_frame_id):
            raise ValueError("registration requires a matching dynamic optical pose")
        mount = TransformSample(camera_pose.child, self.base_frame_id,
            r["translation_m"], r["rotation_wxyz"], camera_pose.stamp, static=True,
            position_error_m=r["position_error_m"], angular_error_rad=r["angular_error_rad"])
        translation = camera_pose.matrix @ (*mount.translation_m, 1.)
        a, b, c, d = camera_pose.rotation_wxyz
        e, f, g, h = mount.rotation_wxyz
        q = np.asarray((a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g,
                        a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e))
        q /= np.linalg.norm(q)
        # Rotation uncertainty moves the base origin through the camera/base
        # lever arm. Unknown terms stay unknown; registration cannot cure SLAM.
        position = None
        if all(v is not None for v in (camera_pose.position_error_m,
                                      camera_pose.angular_error_rad, mount.position_error_m)):
            position = (camera_pose.position_error_m + mount.position_error_m
                + 2*math.dist((0., 0., 0.), mount.translation_m)
                * math.sin(min(math.pi, camera_pose.angular_error_rad)/2))
        angle = None if camera_pose.angular_error_rad is None or mount.angular_error_rad is None else min(
            math.pi, camera_pose.angular_error_rad+mount.angular_error_rad)
        return TransformSample(camera_pose.parent, self.base_frame_id,
            tuple(float(v) for v in translation[:3]), tuple(float(v) for v in q),
            replace(camera_pose.stamp, calibration_id=self.sha256),
            position_error_m=position, angular_error_rad=angle)

    def as_dict(self):
        return {**plain_json(self.record), "sha256": self.sha256}
