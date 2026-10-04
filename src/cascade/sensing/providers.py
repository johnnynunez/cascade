"""Passive adapters. Independent observation clients never construct actuators."""
from __future__ import annotations

import math
import threading
import time
import uuid

from .hub import SensorDescriptor, SensorError
from .models import (ImuPayload, MeasurementMetadata, ObservationEnvelope,
                     ProprioceptionPayload, JointStatePayload, GeneralizedJointStatePayload,
                     RgbPayload, RgbdPayload, RgbdCapturePose, SolvedContactPayload, number)


class BufferedSensorProvider:
    """External producers publish completed immutable captures, including RGB-D.

    Reading the same capture twice does not generate another sequence/timestamp.
    The producer, not this adapter, is responsible for capture and calibration.
    """
    def __init__(self, descriptor: SensorDescriptor):
        if not isinstance(descriptor, SensorDescriptor):
            raise ValueError("typed descriptor required")
        self.descriptor = descriptor
        self._lock = threading.Lock()
        self._observation = None
        self._closed = False

    def publish(self, observation):
        if type(observation) is not ObservationEnvelope:
            raise ValueError("immutable observation required")
        with self._lock:
            if self._closed:
                raise SensorError("buffered provider closed")
            self._observation = observation

    def read(self):
        with self._lock:
            if self._closed or self._observation is None:
                raise SensorError("no completed capture available")
            return self._observation

    def close(self):
        with self._lock:
            self._closed = True
            self._observation = None


class SyntheticSensorProvider:
    """Explicit software fixture, sampled on a wall-clock lattice, without physics.

    A read within the same period returns the same capture identity. These values
    do not certify physical IMU, proprioception or tactile hardware.
    """
    def __init__(self, sensor_id, robot_id, payload, *, period_s=0.02,
                 max_age_s=0.5, read_timeout_s=0.25, clock=time.monotonic):
        if type(payload) not in (ImuPayload, ProprioceptionPayload, JointStatePayload, GeneralizedJointStatePayload):
            raise ValueError("synthetic profile supports explicit IMU/joint fixtures")
        self._period = number(period_s, "synthetic period")
        if not 0.001 <= self._period <= 60:
            raise ValueError("synthetic period outside .001..60 seconds")
        self._clock, self._origin = clock, number(clock(), "clock", minimum=0)
        self._payload, self._closed = payload, False
        self.descriptor = SensorDescriptor(
            sensor_id=sensor_id, robot_id=robot_id, source=f"synthetic:{sensor_id}",
            modality=payload.modality, frame_id=payload.metadata.frame_id,
            clock_domain="monotonic", measurement_kind="synthetic", epoch=uuid.uuid4().hex,
            calibration_id=payload.metadata.calibration_id,
            max_age_s=max_age_s, read_timeout_s=read_timeout_s)

    def read(self):
        if self._closed:
            raise SensorError("synthetic provider closed")
        now = number(self._clock(), "clock", minimum=0)
        if now < self._origin:
            raise SensorError("synthetic clock regressed")
        sequence = math.floor((now - self._origin) / self._period)
        capture = self._origin + sequence * self._period
        return ObservationEnvelope(
            source=self.descriptor.source, sensor_id=self.descriptor.sensor_id,
            epoch=self.descriptor.epoch, sequence=sequence, clock_domain="monotonic",
            capture_time_s=capture, received_monotonic_s=now, producer_age_s=now - capture,
            model_identity_sha256=None, measurement_kind="synthetic", payload=self._payload)

    def close(self):
        self._closed = True


class MobileStateSensorProvider:
    """State-only socket; preserves completed solve identity and round-trip age.

    The current mobile state exports body angular velocity, not acceleration or
    actuator torque. Missing channels remain absent instead of invented zeros.
    """
    def __init__(self, sensor_id, profile, modality, *, max_age_s=0.5,
                 read_timeout_s=0.25, calibration_id=None):
        from ..sim.base_truth import BaseTruthReader

        frames = {"imu": "body", "proprioception": "joints", "solved_contact": "world"}
        if modality not in frames:
            raise ValueError("unsupported mobile state modality")
        self.descriptor = SensorDescriptor(
            sensor_id=sensor_id, robot_id=profile["robot_id"], source=profile["source"],
            modality=modality, frame_id=frames[modality], clock_domain="simulation",
            measurement_kind="physics", model_identity_sha256=profile["model_identity_sha256"],
            epoch=profile.get("epoch"), calibration_id=calibration_id,
            max_age_s=max_age_s, read_timeout_s=read_timeout_s)
        if profile["timeout_s"] > read_timeout_s:
            raise ValueError("reader timeout exceeds sensor deadline")
        self._reader = BaseTruthReader(profile)  # constructor validates without I/O

    def read(self):
        state = self._reader()
        if state is None:
            raise SensorError(f"mobile state unavailable: {self._reader.last_error}")
        if state.robot_id != self.descriptor.robot_id:
            raise SensorError("mobile state robot identity mismatch")
        meta = MeasurementMetadata(self.descriptor.frame_id, self.descriptor.calibration_id)
        if self.descriptor.modality == "imu":
            payload = ImuPayload(meta, state.angular_velocity_body)
        elif self.descriptor.modality == "proprioception":
            payload = ProprioceptionPayload(meta, state.joint_names, state.joint_positions,
                                             state.joint_velocities)
        else:
            if state.support is None:
                raise SensorError("solved contact channel unavailable")
            payload = SolvedContactPayload(meta, state.support)
        return ObservationEnvelope(
            source=state.source, sensor_id=self.descriptor.sensor_id, epoch=state.epoch,
            sequence=state.step, clock_domain="simulation", capture_time_s=state.sim_time_s,
            received_monotonic_s=state.received_monotonic_s, producer_age_s=state.producer_age_s,
            model_identity_sha256=state.model_identity_sha256,
            measurement_kind=state.measurement_kind, payload=payload)

    def close(self):
        self._reader.close()


class MobileRgbSensorProvider:
    """Independent RGB/JPEG reader. The mobile channel does not export depth/K."""
    def __init__(self, sensor_id, profile, camera, *, max_age_s=0.5,
                 read_timeout_s=0.25, calibration_id=None):
        from ..sim.mobile_frames import MobileFrameReader

        self.descriptor = SensorDescriptor(
            sensor_id=sensor_id, robot_id=profile["robot_id"], source=profile["source"],
            modality="rgb", frame_id=f"camera:{camera}", clock_domain="simulation",
            measurement_kind="physics", model_identity_sha256=profile["model_identity_sha256"],
            epoch=profile.get("epoch"), calibration_id=calibration_id,
            max_age_s=max_age_s, read_timeout_s=read_timeout_s)
        if profile["timeout_s"] > read_timeout_s:
            raise ValueError("reader timeout exceeds sensor deadline")
        self._reader = MobileFrameReader(profile)
        if camera not in self._reader.cameras:
            raise ValueError("camera not explicitly configured")
        self._camera = camera

    def read(self):
        frame = self._reader(self._camera)
        if frame is None:
            raise SensorError(f"mobile camera unavailable: {self._reader.last_error}")
        metadata = frame.as_dict()
        if metadata["robot_id"] != self.descriptor.robot_id or metadata["camera"] != self._camera:
            raise SensorError("mobile camera robot/camera mismatch")
        payload = RgbPayload(MeasurementMetadata(self.descriptor.frame_id, self.descriptor.calibration_id),
                             metadata["width"], metadata["height"], "jpeg", frame.jpeg)
        return ObservationEnvelope(
            source=metadata["source"], sensor_id=self.descriptor.sensor_id, epoch=metadata["epoch"],
            sequence=metadata["step"], clock_domain="simulation", capture_time_s=metadata["sim_time_s"],
            received_monotonic_s=metadata["received_monotonic_s"],
            producer_age_s=metadata["producer_age_s"],
            model_identity_sha256=metadata["model_identity_sha256"],
            measurement_kind="physics", payload=payload)

    def close(self):
        self._reader.close()


class MobileRgbdSensorProvider:
    """Opt-in registered capture from the independent mobile RGB-D channel.

    The explicit calibration pin is compared with the producer's complete
    calibration on every capture. Reading opens only a reader-role socket.
    """
    def __init__(self, sensor_id, profile, camera, *, calibration_sha256,
                 max_pixels=640*480, max_age_s=.5, read_timeout_s=.25):
        from ..sim.mobile_rgbd import MobileRgbdReader
        self._reader = MobileRgbdReader(profile, camera, calibration_sha256=calibration_sha256,
                                        max_pixels=max_pixels, max_age_s=max_age_s)
        self.descriptor = SensorDescriptor(
            sensor_id=sensor_id, robot_id=profile['robot_id'], source=profile['source'],
            modality='rgbd', frame_id=f'camera:{camera}', clock_domain='simulation',
            measurement_kind='physics', model_identity_sha256=profile['model_identity_sha256'],
            epoch=profile.get('epoch'), calibration_id=calibration_sha256,
            max_age_s=max_age_s, read_timeout_s=read_timeout_s)
        if profile['timeout_s'] > read_timeout_s:
            raise ValueError('reader timeout exceeds sensor deadline')

    def read(self):
        frame = self._reader()
        if frame is None:
            raise SensorError(f'RGB-D unavailable: {self._reader.last_error}')
        m = frame.metadata
        c = m['calibration']
        pose = None
        if c['version'] == 3:
            p = m['capture_pose']
            pose = RgbdCapturePose(p['epoch'], p['step'], 'simulation', p['sim_time_s'],
                p['model_identity_sha256'], p['world_frame_id'], c['rig_frame_id'],
                p['world_from_rig'], c['rig_from_camera'], p['position_error_m'], p['angular_error_rad'],
                c['mount_position_error_m'], c['mount_angular_error_rad'])
        payload = RgbdPayload(MeasurementMetadata(c['frame_id'], m['calibration_sha256']),
            m['width'], m['height'], frame.rgb8, frame.depth_m_f32le, c['intrinsics'],
            c['world_from_camera'] if pose is None else pose.world_from_camera,
            c['world_frame_id'] if pose is None else pose.world_frame_id, c['pixel_center_offset_uv'], pose)
        return ObservationEnvelope(source=m['source'], sensor_id=self.descriptor.sensor_id,
            epoch=m['epoch'], sequence=m['step'], clock_domain='simulation',
            capture_time_s=m['sim_time_s'], received_monotonic_s=frame.received_monotonic_s,
            producer_age_s=m['producer_age_s'], model_identity_sha256=m['model_identity_sha256'],
            measurement_kind='physics', payload=payload)

    def close(self):
        self._reader.close()
