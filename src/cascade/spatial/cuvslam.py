"""Optional cuVSLAM RGB-D localization of retained captures, without actuation.

API contract: nvidia-isaac/cuVSLAM b405f132b8fb1d861a570f3aea64c2c5d4b59525,
python/cuvslam2.cpp (SHA256 515dfdde55a5007270c94fb1b670a9ea80ecd040d7405d14eff83dcb7e66c559).
This reviewed source reference is not an attestation of an installed binary.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from ..robotics.contracts import ResourceDescriptor, identifier
from ..sensing.hub import SensorHub
from ..sensing.models import MAX_PIXELS, RgbdPayload, digest, integer, number, vector, wire
from .frames import SpatialStamp, TransformSample
from .cuvslam_worker import CuVslamProcess
from .cuvslam_uncertainty import OdometryDiagnostic
from .registration import CameraBaseRegistration


API_REVISION = "b405f132b8fb1d861a570f3aea64c2c5d4b59525"
DEPTH_SCALE = 1000.0  # The Python binding accepts uint16, not metric float32.


def _declared_intrinsics(value, descriptor):
    """Initialization data only; actual capture calibration must match before tracking."""
    keys = {"schema", "width", "height", "intrinsics", "pixel_center_offset_uv",
            "frame_id", "calibration_id", "model_identity_sha256"}
    if (not isinstance(value, dict) or set(value) != keys
            or value["schema"] != "cascade.rgbd-localization-intrinsics.v1"):
        raise ValueError("explicit localization intrinsics schema required")
    if (value["frame_id"] != descriptor.frame_id
            or value["calibration_id"] != descriptor.calibration_id
            or value["model_identity_sha256"] != descriptor.model_identity_sha256):
        raise ValueError("preparation intrinsics must bind the declared sensor frame/calibration/model")
    width, height = (integer(value[k], k, maximum=MAX_PIXELS) for k in ("width", "height"))
    if not 0 < width * height <= MAX_PIXELS:
        raise ValueError("preparation intrinsics exceed the RGB-D pixel bound")
    k = vector(value["intrinsics"], 9, "intrinsics")
    if k[0] <= 0 or k[4] <= 0 or k[1] != 0 or k[3] != 0 or k[6:] != (0., 0., 1.):
        raise ValueError("invalid zero-skew pinhole intrinsics")
    offset = value["pixel_center_offset_uv"]
    if offset is not None:
        offset = vector(offset, 2, "pixel_center_offset_uv")
        if offset not in ((0., 0.), (.5, .5)):
            raise ValueError("unsupported RGB-D pixel-center convention")
    return width, height, k, offset, descriptor.frame_id, descriptor.calibration_id


def _load_sdk(binding_sha256):
    """Check the selected extension before importing this optional CUDA SDK."""
    spec = importlib.util.find_spec("cuvslam")
    if spec is None or spec.origin is None:
        raise RuntimeError("optional cuvslam 17.0.0 is not installed")
    files = list(Path(spec.origin).parent.glob("pycuvslam*.so"))
    if len(files) != 1:
        raise RuntimeError("exactly one installed pycuvslam extension required")
    binding = files[0].resolve()
    with binding.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != binding_sha256:
        raise ValueError("cuVSLAM binding SHA256 mismatch")
    sdk = importlib.import_module("cuvslam")
    if (Path(sdk._core.__file__).resolve() != binding
            or tuple(sdk.get_version()[1:]) != (17, 0, 0)):
        raise ValueError("cuVSLAM loaded binding/version differs from reviewed API")
    return sdk


def _arrays(payload):
    depth = np.frombuffer(payload.depth_m_f32le, dtype="<f4").reshape(payload.height, payload.width)
    scaled = np.rint(depth.astype(np.float64) * DEPTH_SCALE)
    if (np.any(scaled > 65535) or np.any((depth > 0) & (scaled == 0))
            or not np.any(scaled > 0)):
        raise ValueError("depth not representable as uint16 millimeters; no clipping or invented returns")
    rgb = np.frombuffer(payload.rgb8, dtype=np.uint8).reshape(payload.height, payload.width, 3)
    return rgb, scaled.astype(np.uint16)


def _tracker(payload, binding_sha256, max_gap_s, max_poses):
    sdk = _load_sdk(binding_sha256)
    k = payload.intrinsics
    offset = payload.pixel_center_offset_uv or (0., 0.)
    camera = sdk.Camera()
    camera.size = (payload.width, payload.height)
    camera.focal = (k[0], k[4])
    camera.rig_from_camera = sdk.Pose(rotation=[0., 0., 0., 1.], translation=[0., 0., 0.])
    # SDK coordinates use integer pixel centers; preserve the supplied rays.
    camera.principal = (k[2] - offset[0], k[5] - offset[1])
    rgbd = sdk.Odometry.RGBDSettings(depth_scale_factor=DEPTH_SCALE,
        depth_camera_id=0, enable_depth_stereo_tracking=False)
    odometry = sdk.Odometry.Config(odometry_mode=sdk.Odometry.OdometryMode.RGBD,
        rgbd_settings=rgbd, async_sba=False, use_motion_model=False,
        enable_observations_export=False, enable_landmarks_export=False,
        enable_final_landmarks_export=False, max_frame_delta_s=max_gap_s)
    slam = sdk.Slam.Config(sync_mode=True, map_cache_path="", max_map_size=max_poses,
                           enable_reading_internals=False, gt_align_mode=False)
    return sdk.Tracker(sdk.Rig([camera]), sdk.Tracker.Mode.OdometryWithSlamOffline, odometry, slam)


class CuVslamSpatialDomain:
    """One robot/camera/map session; loss requires a new domain and map epoch.

    Native calls are isolated in one owned process with bounded IPC deadlines.
    stop invalidates publication and terminates that process; a busy close
    honestly reports incomplete cleanup and can be retried.
    """
    motion_skills = frozenset()

    def __init__(self, domain_id, robot_id, hub, *, sensor_domain, sensor_id,
                 map_id, map_epoch, map_frame_id, binding_sha256, max_gap_s,
                 max_poses=256, timeout_s=5., base_registration=None,
                 preparation_intrinsics=None, clock=time.monotonic):
        self.domain_id, self.robot_id = identifier(domain_id), identifier(robot_id)
        self.sensor_domain, self.sensor_id = identifier(sensor_domain), identifier(sensor_id)
        self.map_id, self.map_epoch_label = identifier(map_id), identifier(map_epoch)
        # Reusing a profile must never alias two independently initialized maps.
        self.map_epoch = uuid.uuid4().hex
        self.map_frame_id = identifier(map_frame_id)
        self.odometry_frame_id = 'cuvslam-odometry-' + self.map_epoch
        if not isinstance(hub, SensorHub):
            raise ValueError("explicit SensorHub required")
        descriptor = next((d for d in hub.descriptors if d.sensor_id == sensor_id), None)
        if descriptor is None or descriptor.robot_id != robot_id or descriptor.modality != "rgbd":
            raise ValueError("SLAM source must be this robot's declared RGB-D sensor")
        if digest(descriptor.calibration_id) is None or digest(binding_sha256) is None:
            raise ValueError("calibration and installed binding SHA256 required")
        if map_frame_id == descriptor.frame_id:
            raise ValueError("local map frame must differ from camera frame")
        self._base_registration = None if base_registration is None else CameraBaseRegistration(
            base_registration, robot_id=robot_id, descriptor=descriptor, map_frame_id=map_frame_id)
        self.max_gap_s = number(max_gap_s, "max_gap_s", minimum=0)
        if not 0 < self.max_gap_s <= 5:
            raise ValueError("max_gap_s must be in (0, 5]")
        if type(max_poses) is not int or not 1 <= max_poses <= 4096:
            raise ValueError("max_poses must be 1..4096")
        self.timeout_s = number(timeout_s, "timeout_s", minimum=0)
        if not 0 < self.timeout_s <= 60:
            raise ValueError("timeout_s must be in (0, 60]")
        self.hub, self.descriptor = hub, descriptor
        self._declared = (None if preparation_intrinsics is None else
                          _declared_intrinsics(preparation_intrinsics, descriptor))
        self.binding_sha256, self.max_poses, self._clock = binding_sha256, max_poses, clock
        self._lock, self._work = threading.RLock(), threading.Lock()
        self._tracker = self._calibration = self._watermark = self._latest = None
        self._prepared_capture = self._prepared_at = None
        self._failure = None
        self.closed = False
        hub.seal()
        self.required_resources = (f"{sensor_domain}/{sensor_id}",)
        self.resources = (ResourceDescriptor(f"{domain_id}/localization", "frames", robot_id,
            capabilities=("rgbd_slam_localization",),
            synthetic=descriptor.measurement_kind == "synthetic", admission="unvalidated",
            metadata={"sensor_resource": self.required_resources[0], "map_id": map_id,
                "map_epoch": self.map_epoch, "map_epoch_label": self.map_epoch_label,
                "map_frame_id": map_frame_id,
                "odometry_frame_id": self.odometry_frame_id,
                "calibration_sha256": descriptor.calibration_id,
                "model_identity_sha256": descriptor.model_identity_sha256,
                "reviewed_api_revision": API_REVISION, "binding_sha256": binding_sha256,
                "measurement_kind": "estimated", "physical_admission": False}),)
        if self._base_registration is not None:
            from dataclasses import replace
            resource = self.resources[0]
            self.resources = (replace(resource,
                capabilities=(*resource.capabilities, "registered_base_localization"),
                metadata={**resource.metadata, "base_registration": self._base_registration.as_dict()}),)
        capture = {"epoch": {"type": "string"}, "sequence": {"type": "integer", "minimum": 0},
                   "capture_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}
        self.tool_specs = [
            {"name": "warmup_localization", "description": "Prepare the isolated SDK from retained calibration only. Capture fresh tracking images after this returns.",
             "parameters": {"type": "object", "properties": capture, "required": list(capture),
                            "additionalProperties": False}},
            {"name": "track_capture", "description": "Estimate local SLAM pose from an exact retained RGB-D capture; no acquisition or motion.",
             "parameters": {"type": "object", "properties": capture, "required": list(capture),
                            "additionalProperties": False}},
            {"name": "get_localization", "description": "Read the last still-fresh estimate. No occupancy or motion authorization.",
             "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        ]
        if self._declared is not None:
            self.tool_specs.append({"name": "prepare_localization",
                "description": "Initialize the isolated SDK from declared intrinsics before acquisition. No capture or pose admission; fresh tracking inputs must match exactly.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}})

    def _invalidate(self, reason):
        with self._lock:
            self._failure = self._failure or str(reason)[:400]
            self._latest = None

    def _admitted(self):
        if self.closed or self._failure is not None:
            raise ValueError(self._failure or "localization closed")

    def _retained(self, args):
        return self.hub.retained(self.sensor_id, **args)

    def _pose(self, pose, stamp, *, parent=None):
        q = tuple(float(v) for v in pose.rotation)
        if len(q) != 4:
            raise ValueError("invalid SDK quaternion")
        return TransformSample(self.map_frame_id if parent is None else parent, self.descriptor.frame_id,
            tuple(float(v) for v in pose.translation), (q[3], *q[:3]), stamp,
            position_error_m=None, angular_error_rad=None)

    def _capture(self, args):
        if set(args) != {"epoch", "sequence", "capture_sha256"}:
            raise ValueError("exact capture identity required")
        observation = self._retained(args)
        p = observation.payload
        if type(p) is not RgbdPayload or p.metadata.saturated is True:
            raise ValueError("unsaturated registered RGB-D required")
        if p.intrinsics[1] != 0 or p.intrinsics[3] != 0:
            raise ValueError("cuVSLAM pinhole camera does not support skew")
        calibration = (p.width, p.height, p.intrinsics, p.pixel_center_offset_uv,
                       p.metadata.frame_id, p.metadata.calibration_id)
        timestamp = round(observation.capture_time_s * 1e9)
        if not 0 <= timestamp < 2**63:
            raise ValueError("capture timestamp outside SDK int64 range")
        if self._calibration is not None and calibration != self._calibration:
            raise ValueError("calibration changed; new map epoch required")
        return observation, calibration, timestamp

    def _warmup(self, args):
        observation, calibration, timestamp = self._capture(args)
        if self._declared is not None and calibration != self._declared:
            raise ValueError("capture differs from declared preparation intrinsics")
        return self._prepare(calibration, (observation.epoch, observation.sequence, timestamp))

    def _prepare(self, calibration, prepared_capture=None):
        if self._tracker is not None:
            raise ValueError("localization already prepared")
        width, height, intrinsics, offset, _, _ = calibration
        settings = ({"width": width, "height": height, "intrinsics": intrinsics,
                     "pixel_center_offset_uv": offset},
                    self.binding_sha256, self.max_gap_s, self.max_poses)
        self._calibration = calibration
        with self._lock:
            self._admitted()
            try:
                self._tracker = CuVslamProcess(settings, timeout_s=self.timeout_s)
            except BaseException as exc:
                self._tracker = getattr(exc, "cuvslam_worker", None)
                raise
        self._tracker.warmup()
        with self._lock:
            self._admitted()
            self._prepared_capture = prepared_capture
            self._prepared_at = self._clock()
        return {"ok": True, "tracking_state": "ready", "map_epoch": self.map_epoch,
                "map_epoch_label": self.map_epoch_label, "physical_admission": False}

    def _track(self, args):
        if self._prepared_at is None:
            raise ValueError("localization preparation must complete before tracking captures")
        observation, _, timestamp = self._capture(args)
        p = observation.payload
        if observation.received_monotonic_s - observation.producer_age_s < self._prepared_at:
            raise ValueError("tracking capture must follow warmup in the same sensor epoch")
        if self._prepared_capture is not None:
            epoch, sequence, prepared_timestamp = self._prepared_capture
            if observation.epoch != epoch or observation.sequence <= sequence or timestamp <= prepared_timestamp:
                raise ValueError("tracking capture must follow warmup in the same sensor epoch")
        if self._watermark is not None:
            epoch, seq, previous = self._watermark
            if (observation.epoch != epoch or observation.sequence <= seq or timestamp <= previous
                    or timestamp - previous > self.max_gap_s * 1e9):
                raise ValueError("capture epoch/replay/clock gap; new map epoch required")
        rgb, depth = _arrays(p)
        self._watermark = (observation.epoch, observation.sequence, timestamp)
        with self._lock:
            self._admitted()
        self._retained(args)
        odometry, slam = self._tracker.track(timestamp, images=[rgb], depths=[depth])
        if odometry.timestamp_ns != timestamp or odometry.world_from_rig is None or slam is None:
            raise ValueError("tracking lost or SDK timestamp mismatch; new map epoch required")
        stamp = SpatialStamp(self.map_id, self.map_epoch, observation.clock_domain,
            observation.capture_time_s, self.required_resources[0], args["capture_sha256"],
            p.metadata.calibration_id, "estimated")
        odometry_diagnostic = OdometryDiagnostic(
            self._pose(odometry.world_from_rig.pose, stamp, parent=self.odometry_frame_id),
            getattr(odometry.world_from_rig, 'covariance_xyz_rpy', None))
        pose = self._pose(slam, stamp)
        if self._retained(args) is not observation:
            raise ValueError("capture changed during estimation")
        with self._lock:
            self._admitted()
            self._latest = (pose, observation, dict(args), odometry_diagnostic)
            return self._read()

    def _read(self):
        self._admitted()
        if self._latest is None:
            return {"ok": False, "tracking_state": "uninitialized", "physical_admission": False}
        pose, observation, args, odometry_diagnostic = self._latest
        self._retained(args)  # A delayed call/result must never rejuvenate its input.
        result = {"ok": True, "pose": wire(pose), "robot_id": self.robot_id,
            "odometry_diagnostic": odometry_diagnostic.as_dict(),
            "sensor_epoch": observation.epoch, "sequence": observation.sequence,
            "map_epoch_label": self.map_epoch_label,
            "source": observation.source, "model_identity_sha256": observation.model_identity_sha256,
            "capture_age_s": observation.age_s(self._clock()), "physical_admission": False,
            "reviewed_api_revision": API_REVISION, "binding_sha256": self.binding_sha256,
            "depth_quantization_m": 1 / DEPTH_SCALE, "loop_closure_jumps_possible": True}
        if self._base_registration is not None:
            result.update(base_pose=wire(self._base_registration.pose(pose)),
                          base_registration_sha256=self._base_registration.sha256)
            if self._retained(args) is not observation:
                raise ValueError("capture changed during base registration")
            result["capture_age_s"] = observation.age_s(self._clock())
        return result

    def execute(self, name, args):
        names = {"warmup_localization", "track_capture", "get_localization"}
        if self._declared is not None:
            names.add("prepare_localization")
        if name not in names or not isinstance(args, dict):
            return {"ok": False, "error": "unknown localization tool or arguments", "physical_admission": False}
        if not self._work.acquire(blocking=False):
            return {"ok": False, "error": "localization call in flight", "physical_admission": False}
        try:
            with self._lock:
                self._admitted()
            if name == "warmup_localization":
                return self._warmup(args)
            if name == "prepare_localization":
                if args:
                    raise ValueError("prepare_localization takes no arguments")
                return self._prepare(self._declared)
            if name == "track_capture":
                return self._track(args)
            if args:
                raise ValueError("get_localization takes no arguments")
            with self._lock:
                return self._read()
        except BaseException as exc:
            self._invalidate(exc)
            if not isinstance(exc, Exception):
                raise
            return {"ok": False, "error": str(exc)[:500], "map_epoch_invalidated": True,
                    "physical_admission": False}
        finally:
            # A failed/cancelled worker is never reused under this map epoch.
            try:
                if self.closed or self._failure is not None:
                    if self._tracker is not None and self._tracker.close():
                        self._tracker = None
            finally:
                self._work.release()

    def stop(self):
        self._invalidate("localization stopped; new map epoch required")
        if self._tracker is not None:
            self._tracker.request_stop()
        return {"ok": True, "actuation": False, "map_epoch_invalidated": True}

    def reset_stop(self):
        return {"ok": False, "actuation": False, "error": "rebuild localization with a new map epoch"}

    def close(self):
        with self._lock:
            self.closed = True
            self._latest = None
        acquired = self._work.acquire(blocking=False)
        if acquired:
            try:
                if self._tracker is not None:
                    acquired = self._tracker.close()
                    if acquired:
                        self._tracker = None
            finally:
                self._work.release()
        elif self._tracker is not None:
            self._tracker.request_stop()
        return {"ok": acquired, "actuation": False, "in_flight": not acquired}


def build_cuvslam_domain(domain_id, profile, sensor_domains):
    if set(profile) != {"kind", "robot_id", "cuvslam"}:
        raise ValueError("cuVSLAM spatial profile requires only kind, robot_id, cuvslam")
    settings = profile["cuvslam"]
    required = {"sensor_domain", "sensor_id", "map_id", "map_epoch", "map_frame_id",
                "binding_sha256", "max_gap_s"}
    if (not isinstance(settings, dict) or not required <= set(settings)
            or set(settings) - (required | {"max_poses", "timeout_s", "base_registration", "preparation_intrinsics"})):
        raise ValueError("explicit cuVSLAM capture/map/binary configuration required")
    source = sensor_domains.get(settings["sensor_domain"])
    if source is None:
        raise ValueError("cuVSLAM requires an explicit sensors domain")
    return CuVslamSpatialDomain(domain_id, profile["robot_id"], source.hub, **settings)
