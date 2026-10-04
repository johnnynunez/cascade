"""Opt-in, bounded RGB-D capture transport; reader sockets never step physics.

RGB bytes are RGB8 (not OpenCV BGR). Depth is optical-axis distance in meters,
float32 little endian, with zero for invalid pixels. Calibration is content
addressed. Legacy v2 calibration is static; v3 separates a rigid optical mount
from an explicit pose tied to the RGB/depth render capture.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
import threading
import time
import zlib

from ..control.mobile_base import finite_real, nonnegative_int
from .base_truth import BaseTruthReader
from .mobile_frames import _FrameRPC

MAX_PIXELS = 1024 * 1024
IDENTITY_KEYS = ('robot_id', 'source', 'epoch', 'engine', 'device', 'asset_sha256',
                 'policy_sha256', 'model_identity_sha256')
CALIBRATION_KEYS = {'version', 'camera', 'frame_id', 'world_frame_id', 'width', 'height',
                    'intrinsics', 'world_from_camera', 'depth_convention', 'pixel_center_offset_uv'}
MOUNT_CALIBRATION_KEYS = CALIBRATION_KEYS - {'world_from_camera'} | {
    'rig_frame_id', 'rig_from_camera', 'mount_position_error_m', 'mount_angular_error_rad'}
FRAME_KEYS = set(IDENTITY_KEYS) | {'version', 'camera', 'step', 'sim_time_s', 'width', 'height',
    'rgb8_z_b64', 'depth_m_f32le_z_b64', 'calibration', 'calibration_sha256', 'producer_age_s', 'render_reference'}


def read_static_calibration(stage, path, *, width=640, height=480):
    """Read the owned USD perspective camera; optical frame is X right/Y down/Z forward.

    This initial producer supports a static camera only. Animated mounts/lenses
    need a separate Fabric pose contract, and must not silently use USD defaults.
    """
    return _read_calibration(stage, path, width=width, height=height)


def read_mount_calibration(stage, path, mount, *, width=640, height=480):
    """Validate fixed local optics/mount; never use the rig's USD world pose."""
    from .mobile_camera_pose import mount_record
    return _read_calibration(stage, path, width=width, height=height, mount=mount_record(mount))


def _read_calibration(stage, path, *, width, height, mount=None):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(path)
    camera = UsdGeom.Camera(prim)
    if not camera or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('RGB-D requires an actual meter-stage camera')
    ancestor = prim
    while ancestor and not ancestor.IsPseudoRoot():
        # One authored time sample can override a different default even when
        # ValueMightBeTimeVarying() is false. Default-time optics cannot bind it.
        if any(attr.GetNumTimeSamples() != 0 for attr in ancestor.GetAttributes()):
            raise ValueError('RGB-D overview calibration must be static')
        if mount is not None:
            # The camera is a direct rigid child. The body's changing Fabric
            # world pose is supplied separately, never admitted from defaults.
            break
        ancestor = ancestor.GetParent()
    if (camera.GetProjectionAttr().Get() != 'perspective'
            or camera.GetHorizontalApertureOffsetAttr().Get() != 0
            or camera.GetVerticalApertureOffsetAttr().Get() != 0
            or any('LensDistortion' in str(schema) for schema in prim.GetAppliedSchemas())):
        raise ValueError('RGB-D currently requires an undistorted centered perspective camera')
    f, horizontal, vertical = (float(attr.Get()) for attr in (
        camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(), camera.GetVerticalApertureAttr()))
    if not all(math.isfinite(v) and v > 0 for v in (f, horizontal, vertical)):
        raise ValueError('invalid observed camera optics')
    # Gf uses row vectors; change USD -Z/+Y camera axes to optical +Z/-Y.
    if mount is None:
        transform = np.array(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)).T
    else:
        rig = stage.GetPrimAtPath(mount['rig_prim_path'])
        local = UsdGeom.Xformable(prim)
        if (not rig or not rig.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetParent() != rig
                or local.GetResetXformStack()):
            raise ValueError('RGB-D mount requires a direct non-reset child of the named rigid body')
        transform = np.array(local.GetLocalTransformation()).T
    transform = transform @ np.diag([1., -1., -1., 1.])
    # USD K uses the image boundary as raster origin. Array element [v,u]
    # samples its center at (u+.5,v+.5); do not shift the observed USD optics.
    record = dict(version=2, camera='overview', frame_id='camera:overview', world_frame_id='world',
        width=width, height=height,
        intrinsics=[width*f/horizontal, 0., width/2., 0., height*f/vertical, height/2., 0., 0., 1.],
        world_from_camera=transform.flatten().tolist(), depth_convention='optical_z_m_zero_invalid',
        pixel_center_offset_uv=[.5, .5])
    if mount is not None:
        if not np.allclose(transform, np.array(mount['rig_from_camera']).reshape(4,4), rtol=0, atol=1e-7):
            raise ValueError('authored local optical mount differs from explicit mount')
        record.pop('world_from_camera')
        record.update(version=3, rig_frame_id=mount['rig_frame_id'], rig_from_camera=transform.flatten().tolist(),
                      mount_position_error_m=mount['position_error_m'], mount_angular_error_rad=mount['angular_error_rad'])
    return calibration_record(record)[0]


def calibration_record(value):
    """Canonical pinhole calibration, static v2 or explicitly mounted v3."""
    import numpy as np
    if not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] not in (2, 3):
        raise ValueError('unsupported RGB-D calibration schema version')
    mounted = value['version'] == 3
    if set(value) != (MOUNT_CALIBRATION_KEYS if mounted else CALIBRATION_KEYS):
        raise ValueError('invalid RGB-D calibration schema')
    if (value['camera'] != 'overview' or value['frame_id'] != 'camera:overview'
            or (not mounted and value['world_frame_id'] != 'world')
            or value['depth_convention'] != 'optical_z_m_zero_invalid'):
        raise ValueError('unsupported RGB-D calibration frame/convention')
    if mounted:
        from ..robotics.contracts import identifier
        if len({identifier(value['rig_frame_id']), identifier(value['world_frame_id']), value['frame_id']}) != 3:
            raise ValueError('world, rig and optical frame must differ')
        for key in ('mount_position_error_m', 'mount_angular_error_rad'):
            error = value[key]
            if error is not None and (finite_real(error, key) < 0 or (key.endswith('rad') and error > math.pi)):
                raise ValueError('invalid rigid mount error bound')
    offset = value['pixel_center_offset_uv']
    if (not isinstance(offset, (tuple, list)) or len(offset) != 2
            or [finite_real(v, 'pixel center offset') for v in offset] != [.5, .5]):
        raise ValueError('native RGB-D requires raster pixel-center offset [.5, .5]')
    w, h = (nonnegative_int(value[k], k) for k in ('width', 'height'))
    if not 0 < w * h <= MAX_PIXELS:
        raise ValueError('RGB-D calibration exceeds pixel bound')
    transform_key = 'rig_from_camera' if mounted else 'world_from_camera'
    k, transform = value['intrinsics'], value[transform_key]
    if not isinstance(k, (tuple, list)) or len(k) != 9 or not isinstance(transform, (tuple, list)) or len(transform) != 16:
        raise ValueError('RGB-D calibration matrix shape mismatch')
    k = [finite_real(v, 'intrinsics') for v in k]
    transform = [finite_real(v, 'world_from_camera') for v in transform]
    if k[0] <= 0 or k[4] <= 0 or k[1] != 0 or k[3] != 0 or k[6:] != [0, 0, 1]:
        raise ValueError('invalid pinhole intrinsics')
    t = np.array(transform).reshape(4, 4)
    if (not np.array_equal(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), rtol=0, atol=1e-7)
            or not math.isclose(np.linalg.det(t[:3, :3]), 1., abs_tol=1e-7)):
        raise ValueError('world_from_camera must be a rigid optical-frame transform')
    result = {**copy.deepcopy(value), 'intrinsics': k, transform_key: transform,
              'pixel_center_offset_uv': [.5, .5]}
    encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return result, hashlib.sha256(encoded).hexdigest()


def render_reference_record(value, sim_time_s):
    """Normalize exact SDK scalar types, without truncating or dropping fields."""
    from .microduck_stepper import validate_render_times
    validate_render_times(value, sim_time_s)
    if not isinstance(value, dict) or set(value) != {'rpFabricTime', 'IsaacReadSimulationTime'}:
        raise ValueError('unknown RGB-D render reference fields')
    fabric, simulation = value['rpFabricTime'], value['IsaacReadSimulationTime']
    if (not isinstance(fabric, dict) or set(fabric) != {'fabricFrameTimeNumerator', 'fabricFrameTimeDenominator'}
            or not isinstance(simulation, dict) or set(simulation) - {'simulationTime', 'execOut'}):
        raise ValueError('unknown RGB-D render clock fields')
    def exact_integer(v):
        if isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v <= 2**63-1:
            raise ValueError('RGB-D render reference requires exact bounded integers')
        return int(v)
    result = {'rpFabricTime': {k: exact_integer(v) for k, v in fabric.items()},
              'IsaacReadSimulationTime': {'simulationTime': float(simulation['simulationTime'])}}
    if 'execOut' in simulation:
        result['IsaacReadSimulationTime']['execOut'] = exact_integer(simulation['execOut'])
    return result


def capture_pose_record(value, calibration, identity, step, sim_time_s, reference):
    """Bind an explicit pose to this exact model/epoch and render completion."""
    from ..sensing.models import RgbdCapturePose
    keys = {'epoch', 'step', 'sim_time_s', 'model_identity_sha256', 'world_frame_id',
            'world_from_rig', 'position_error_m', 'angular_error_rad', 'render_reference'}
    if not isinstance(value, dict) or set(value) != keys or calibration['version'] != 3:
        raise ValueError('explicit capture pose requires mounted RGB-D calibration')
    pose = RgbdCapturePose(value['epoch'], value['step'], 'simulation', value['sim_time_s'],
        value['model_identity_sha256'], value['world_frame_id'], calibration['rig_frame_id'],
        value['world_from_rig'], calibration['rig_from_camera'], value['position_error_m'],
        value['angular_error_rad'], calibration['mount_position_error_m'], calibration['mount_angular_error_rad'])
    if (pose.epoch != identity['epoch'] or pose.model_identity_sha256 != identity['model_identity_sha256']
            or pose.sequence != step or pose.capture_time_s != sim_time_s
            or pose.world_frame_id != calibration['world_frame_id']
            or render_reference_record(value['render_reference'], sim_time_s) != reference):
        raise ValueError('RGB-D pose model/epoch/capture/render mismatch')
    return {'epoch': pose.epoch, 'step': pose.sequence, 'sim_time_s': pose.capture_time_s,
            'model_identity_sha256': pose.model_identity_sha256, 'world_frame_id': pose.world_frame_id,
            'world_from_rig': list(pose.world_from_rig),
            'position_error_m': pose.position_error_m, 'angular_error_rad': pose.angular_error_rad,
            'render_reference': copy.deepcopy(reference)}


def _encode(raw):
    return base64.b64encode(zlib.compress(raw, 3)).decode('ascii')


def _decode(encoded, expected):
    # A valid zlib stream needs only small fixed overhead over incompressible
    # input. Bound encoded input AND decompressed output before any image array.
    cap = expected + 65536
    if not isinstance(encoded, str) or len(encoded) > 4 * ((cap + 2) // 3):
        raise ValueError('RGB-D compressed input exceeds byte bound')
    compressed = base64.b64decode(encoded, validate=True)
    if len(compressed) > cap:
        raise ValueError('RGB-D compressed input exceeds byte bound')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError('RGB-D decompressed size/stream mismatch')
    return raw


class RgbdFrameCache:
    """One completed registered capture; legacy RGB replies remain unchanged."""
    rgbd_enabled = True
    rgbd_wait_next = True

    def __init__(self, identity, *, calibration, max_jpeg_bytes, max_pixels, clock=time.monotonic):
        from .microduck_stepper import FrameCache
        self._calibration, self.calibration_sha256 = calibration_record(calibration)
        if self._calibration['width'] * self._calibration['height'] > max_pixels:
            raise ValueError('calibration exceeds configured pixel bound')
        self._rgb = FrameCache(identity, max_jpeg_bytes=max_jpeg_bytes, max_pixels=max_pixels, clock=clock)
        self._clock, self._lock, self._frame, self._closed = clock, threading.Lock(), None, False
        self._published = threading.Condition(self._lock)

    def publish(self, rgb, *, depth_m, calibration, step, sim_time_s, captured_at, render_times,
                rgbd_render_times, capture_pose=None):
        import numpy as np
        if not isinstance(rgbd_render_times, dict) or set(rgbd_render_times) != {'rgb', 'depth'}:
            raise ValueError('RGB-D requires both product readback references')
        reference = render_reference_record(render_times, sim_time_s)
        channel_refs = {key: render_reference_record(times, sim_time_s) for key, times in rgbd_render_times.items()}
        for times in channel_refs.values():
            if times != reference:
                raise ValueError('RGB-D render references are not aligned')
        record, digest = calibration_record(calibration)
        if digest != self.calibration_sha256 or record != self._calibration:
            raise ValueError('RGB-D calibration changed after model admission')
        mounted = record['version'] == 3
        if mounted:
            capture_pose = capture_pose_record(capture_pose, record, self._rgb.identity, step, sim_time_s, reference)
        elif capture_pose is not None:
            raise ValueError('static RGB-D calibration cannot accept a dynamic pose')
        h, w = record['height'], record['width']
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (h, w, 3):
            raise ValueError('RGB-D requires registered RGB8')
        if (not isinstance(depth_m, np.ndarray) or depth_m.dtype != np.float32 or depth_m.shape != (h, w)
                or not np.isfinite(depth_m).all() or (depth_m < 0).any()):
            raise ValueError('RGB-D requires aligned finite nonnegative metric depth')
        packet = {**self._rgb.identity, 'version': 2 if mounted else 1, 'camera': record['camera'],
                  'step': step, 'sim_time_s': sim_time_s, 'width': w, 'height': h,
                  'rgb8_z_b64': _encode(rgb.tobytes()),
                  'depth_m_f32le_z_b64': _encode(depth_m.astype('<f4', copy=False).tobytes()),
                  'calibration': record, 'calibration_sha256': digest,
                  'render_reference': channel_refs}
        if mounted:
            packet['capture_pose'] = capture_pose
        with self._lock:
            if self._closed:
                raise RuntimeError('RGB-D cache closed')
            self._rgb.publish(rgb, step=step, sim_time_s=sim_time_s, captured_at=captured_at, render_times=render_times)
            self._frame, self._captured_at = packet, captured_at
            self._published.notify_all()

    def _wait_next(self, request):
        selection = request['wait_next']
        if not isinstance(selection, dict) or set(selection) != {'epoch', 'after_step', 'timeout_s'}:
            raise ValueError('wait_next requires exact epoch, after_step and timeout_s')
        step = nonnegative_int(selection['after_step'], 'after_step')
        timeout = finite_real(selection['timeout_s'], 'wait_next timeout_s')
        if not 0 < timeout <= 1.:
            raise ValueError('wait_next timeout_s must be in (0, 1]')
        if selection['epoch'] != self._rgb.identity['epoch']:
            raise ValueError('wait_next epoch mismatch')
        if self._frame is None or step > self._frame['step']:
            raise ValueError('wait_next cannot request a future or unobserved step')
        # The transport supplies its original request deadline, never a
        # wire-supplied timestamp. Direct cache callers still have a bounded wait.
        deadline = time.monotonic() + timeout
        if '_frame_deadline' in request:
            deadline = min(deadline, finite_real(request['_frame_deadline'], 'frame deadline'))
        cancelled = request.get('_frame_cancelled', lambda: False)
        while True:
            if self._closed or cancelled():
                raise RuntimeError('RGB-D wait closed')
            if self._frame['epoch'] != selection['epoch']:
                raise ValueError('RGB-D epoch changed while waiting')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('RGB-D next capture deadline expired')
            if self._frame['step'] > step:
                return
            # Publication/close notify the condition. The bounded wake interval
            # only observes transport shutdown; it never selects a physics phase.
            self._published.wait(min(remaining, .05))

    def __call__(self, request):
        with self._lock:
            if self._closed:
                raise RuntimeError('RGB-D cache closed')
            if request.get('modality', 'rgb') == 'rgb':
                if 'wait_next' in request:
                    raise ValueError('wait_next requires RGB-D')
                return self._rgb(request)
            if request.get('modality') != 'rgbd' or request.get('camera') != 'overview':
                raise ValueError('unknown RGB-D camera/modality')
            if self._frame is None:
                raise RuntimeError('no completed RGB-D capture')
            if 'wait_next' in request:
                self._wait_next(request)
            age = self._clock() - self._captured_at
            if not math.isfinite(age) or age < 0:
                raise RuntimeError('RGB-D producer clock regressed')
            return {'rgbd': copy.deepcopy(self._frame) | {'producer_age_s': age}}

    def close(self):
        with self._lock:
            self._closed, self._frame = True, None
            self._published.notify_all()
            self._rgb.close()


@dataclass(frozen=True)
class MobileRgbdFrame:
    _metadata: object
    rgb8: bytes
    depth_m_f32le: bytes
    received_monotonic_s: float

    def __post_init__(self):
        from ..robotics.contracts import freeze_json
        object.__setattr__(self, '_metadata', freeze_json(self._metadata))
        if type(self.rgb8) is not bytes or type(self.depth_m_f32le) is not bytes:
            raise ValueError('RGB-D capture buffers must be immutable bytes')

    @property
    def metadata(self):
        from ..robotics.contracts import plain_json
        return plain_json(self._metadata)


class MobileRgbdReader:
    """Independent reader with pinned source/model/epoch/calibration, no owner."""
    def __init__(self, profile, camera, *, calibration_sha256, max_pixels=640*480, max_age_s=.5,
                 wait_next=False):
        from .microduck_newton import digest_token
        if camera != 'overview':
            raise ValueError('RGB-D currently supports the explicit overview camera')
        if type(max_pixels) is not int or not 0 < max_pixels <= MAX_PIXELS:
            raise ValueError('RGB-D pixel limit outside bound')
        self.calibration_sha256 = digest_token(calibration_sha256)
        self.max_age_s = finite_real(max_age_s, 'max_age_s')
        if not 0 < self.max_age_s <= 60:
            raise ValueError('RGB-D max_age_s outside bound')
        self._identity = BaseTruthReader(profile)  # validation only
        self._profile, self.camera, self._pixels = copy.deepcopy(profile), camera, max_pixels
        self._client, self._lock = _FrameRPC(profile), threading.Lock()
        self._closed, self._ready, self._seen = False, False, None
        if type(wait_next) is not bool:
            raise ValueError('wait_next must be an explicit boolean')
        self._wait_for_next = wait_next
        self.last_error = None

    def _decode(self, response, received, rtt):
        import numpy as np
        if set(response) != {'ok', 'rgbd'} or not isinstance(response['rgbd'], dict):
            raise ValueError('invalid RGB-D packet schema')
        m = copy.deepcopy(response['rgbd'])
        if type(m.get('version')) is not int or m['version'] not in (1, 2):
            raise ValueError('RGB-D version/camera mismatch')
        mounted = m['version'] == 2
        if set(m) != FRAME_KEYS | ({'capture_pose'} if mounted else set()):
            raise ValueError('invalid RGB-D packet schema')
        for key in IDENTITY_KEYS:
            expected = self._identity._epoch if key == 'epoch' else self._profile[key]
            if m[key] != expected:
                raise ValueError('RGB-D ' + key + ' mismatch')
        if m['camera'] != self.camera:
            raise ValueError('RGB-D version/camera mismatch')
        nonnegative_int(m['step'], 'step')
        w, h = (nonnegative_int(m[k], k) for k in ('width', 'height'))
        if not 0 < w*h <= self._pixels:
            raise ValueError('RGB-D dimensions exceed pixel bound')
        for key in ('sim_time_s', 'producer_age_s'):
            if finite_real(m[key], key) < 0:
                raise ValueError('negative RGB-D clock')
        refs = m['render_reference']
        if not isinstance(refs, dict) or set(refs) != {'rgb', 'depth'} or refs['rgb'] != refs['depth']:
            raise ValueError('RGB-D render references differ')
        m['render_reference'] = {key: render_reference_record(times, m['sim_time_s']) for key, times in refs.items()}
        cal, digest = calibration_record(m['calibration'])
        if (digest != m['calibration_sha256'] or digest != self.calibration_sha256
                or (cal['width'], cal['height']) != (w, h) or cal['version'] != (3 if mounted else 2)):
            raise ValueError('RGB-D calibration identity/dimensions mismatch')
        if mounted:
            m['capture_pose'] = capture_pose_record(m['capture_pose'], cal, m, m['step'], m['sim_time_s'],
                                                   m['render_reference']['rgb'])
        rgb = _decode(m.pop('rgb8_z_b64'), w*h*3)
        depth = _decode(m.pop('depth_m_f32le_z_b64'), w*h*4)
        d = np.frombuffer(depth, dtype='<f4')
        if not np.isfinite(d).all() or (d < 0).any():
            raise ValueError('invalid RGB-D metric depth')
        m['producer_age_s'] += rtt
        previous = self._seen
        if previous is not None:
            old = previous.metadata
            if m['step'] <= old['step'] or m['sim_time_s'] <= old['sim_time_s']:
                raise ValueError('RGB-D replay or capture clock regression')
            if received < previous.received_monotonic_s:
                raise ValueError('RGB-D receipt clock regression')
        frame = MobileRgbdFrame(m, rgb, depth, received)
        # Retain seen identities even when stale, preventing rejuvenation.
        self._seen = frame
        if m['producer_age_s'] + time.monotonic() - received > self.max_age_s:
            raise ValueError('stale RGB-D capture')
        return frame

    def __call__(self):
        deadline = time.monotonic() + self._profile['timeout_s']
        if not self._lock.acquire(timeout=self._profile['timeout_s']):
            self.last_error = 'RGB-D reader lock timeout'
            return None
        try:
            if self._closed:
                raise ValueError('RGB-D reader closed')
            def remaining():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('RGB-D reader deadline')
                return left
            if not self._ready:
                self._client.max_reply = 16384
                self._client.connect(remaining())
                hello = self._client.request({'op': 'hello', 'role': 'reader'}, timeout_s=remaining())
                self._identity._hello(hello)
                if 'rgbd' not in hello['capabilities']:
                    raise ValueError('producer does not advertise RGB-D')
                if self._wait_for_next and 'rgbd_wait_next' not in hello['capabilities']:
                    raise ValueError('producer does not advertise bounded RGB-D wait_next')
                self._ready = True
            self._client.max_reply = 16384 + 4*((7*self._pixels + 2*65536 + 2)//3)
            started = time.monotonic()
            request = {'op': 'frame', 'camera': self.camera, 'modality': 'rgbd'}
            if self._wait_for_next and self._seen is not None:
                previous = self._seen.metadata
                request['wait_next'] = {'epoch': previous['epoch'], 'after_step': previous['step'],
                                        'timeout_s': min(1., remaining())}
            reply = self._client.request(request, timeout_s=remaining())
            received = time.monotonic()
            frame = self._decode(reply, received, received-started)
            remaining()
            self.last_error = None
            return frame  # deep-frozen metadata and immutable channel bytes
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'[:400]
            self._ready = False
            self._client.close()
            return None
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            self._closed = True
            self._client.close()
            self._identity.close()
            self._seen = None
