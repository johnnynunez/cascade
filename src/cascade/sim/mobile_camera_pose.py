"""Opt-in rigid camera mount and same-render Fabric pose checks; no pose writes."""
from __future__ import annotations

import copy
import hashlib
from numbers import Integral
from pathlib import Path
import re

from ..robotics.contracts import identifier
from ..sensing.models import RgbdCapturePose, rigid_transform
from .microduck_newton import digest_token, strict_json


CAMERA_LEAF = 'CascadeRgbd'
FABRIC_MATRIX = 'omni:fabric:worldMatrix'
FABRIC_INDEX = 'newton:index'


def _checked_transform(matrix, name):
    try:
        return rigid_transform(tuple(matrix.flat), name)
    except ValueError as error:
        # Keep the offending source and all 16 values when startup fails before
        # any capture is published. This changes diagnostics, not admission.
        raise ValueError(f'{name}: {error}; matrix={matrix.tolist()}') from error


def _render_optical_pose(view):
    """Canonicalize only the homogeneous scalar of an affine SDK view.

    Gf's double matrix inverse can return w=1 +/- a few ULP even for an
    exactly affine input. Divide the entire homogeneous matrix by w; never
    repair its rotation, perspective terms, or metric translation.
    """
    import numpy as np
    transform = view.T
    w = transform[3, 3]
    if (not np.array_equal(transform[3, :3], [0., 0., 0.])
            or abs(w - 1.) > 8 * np.spacing(1.)):
        raise ValueError(f'render world-to-view is not an affine homogeneous matrix: {transform.tolist()}')
    transform = transform / w
    _checked_transform(transform, 'render world-to-view')
    # Invert the affine blocks rather than a general 4x4 matrix: the latter
    # can itself introduce roundoff in the exact homogeneous output row.
    optical = np.eye(4)
    optical[:3, :3] = np.linalg.inv(transform[:3, :3])
    optical[:3, 3] = -optical[:3, :3] @ transform[:3, 3]
    optical = optical @ np.diag([1., -1., -1., 1.])
    return _checked_transform(optical, 'render world-from-optical')


def mount_record(value):
    fields = {'schema', 'rig_prim_path', 'rig_frame_id', 'rig_from_camera',
              'position_error_m', 'angular_error_rad'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['schema'] != 'cascade.rigid-render-camera.v1'):
        raise ValueError('explicit rigid render camera schema required')
    path = value['rig_prim_path']
    if not isinstance(path, str) or not re.fullmatch(r'/World/[A-Za-z_][A-Za-z_0-9]*(?:/[A-Za-z_][A-Za-z_0-9]*)*', path):
        raise ValueError('rig requires a canonical owned absolute prim path')
    frame = identifier(value['rig_frame_id'])
    if frame in ('world', 'camera:overview'):
        raise ValueError('rig, world and optical frame must differ')
    # Reuse the public rigid-pose validation; these are mount bounds, not
    # estimates of the dynamic Fabric pose or the depth measurement.
    identity = (1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.)
    pose = RgbdCapturePose('mount', 0, 'simulation', 0., '0'*64, 'world', frame,
        identity, value['rig_from_camera'], None, None, value['position_error_m'], value['angular_error_rad'])
    return {'schema': value['schema'], 'rig_prim_path': path, 'rig_frame_id': frame,
            'rig_from_camera': list(pose.rig_from_camera),
            'position_error_m': pose.mount_position_error_m,
            'angular_error_rad': pose.mount_angular_error_rad}


def admit_mount(path, expected):
    path = Path(path).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest_token(expected):
        raise ValueError('camera mount file differs from explicit SHA256')
    return {'path': str(path), 'sha256': expected, 'definition': mount_record(strict_json(data))}


def camera_path(mount):
    return mount['rig_prim_path'] + '/' + CAMERA_LEAF


def camera_params_record(value, calibration):
    """Validate actual render-product optics and retain its row-vector view.

    SDK48's CameraParams example translates a camera at +100 by -100;
    PointCloudGenerator and camera_projection_utils invert this world-to-view
    matrix. The prose calling it camera-to-world is inconsistent with those
    implementations. No USD/default transform is read here.
    """
    import numpy as np
    if not isinstance(value, dict) or value.get('cameraModel') != 'pinhole':
        raise ValueError('render camera must report a pinhole model')
    record = {'cameraModel': value['cameraModel']}
    for name, count in (('cameraViewTransform', 16), ('cameraProjection', 16),
                        ('cameraAperture', 2), ('cameraApertureOffset', 2), ('renderProductResolution', 2)):
        array = np.asarray(value[name])
        if array.size != count or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError('invalid render camera field: '+name)
        record[name] = array.reshape(-1).astype(float).tolist()
    for name in ('cameraFocalLength', 'metersPerSceneUnit'):
        scalar = value[name]
        if isinstance(scalar, bool) or not isinstance(scalar, (float, int, np.floating)) or not np.isfinite(scalar):
            raise ValueError('invalid render camera scalar: '+name)
        record[name] = float(scalar)
    w, h = calibration['width'], calibration['height']
    if (record['metersPerSceneUnit'] != 1. or record['renderProductResolution'] != [w,h]
            or record['cameraApertureOffset'] != [0.,0.] or record['cameraFocalLength'] <= 0
            or min(record['cameraAperture']) <= 0):
        raise ValueError('render camera units/resolution/optics differ')
    f = record['cameraFocalLength']; a, b = record['cameraAperture']
    observed = [w*f/a,0.,w/2,0.,h*f/b,h/2,0.,0.,1.]
    # Float32 annotator optics versus authored double calibration. This checks
    # numeric agreement, not a calibrated physical uncertainty bound.
    if not np.allclose(observed, calibration['intrinsics'], rtol=1e-6, atol=1e-7):
        raise ValueError('render camera intrinsics differ from admitted calibration')
    projection = np.asarray(record['cameraProjection']).reshape(4,4)
    if (not np.allclose(projection[:, :2], [[2*f/a,0.],[0.,2*f/b],[0.,0.],[0.,0.]], rtol=1e-6, atol=1e-7)
            or not np.allclose(projection[:,3], [0.,0.,-1.,0.], rtol=0, atol=1e-7)):
        raise ValueError('render projection differs from centered pinhole calibration')
    view = np.asarray(record['cameraViewTransform']).reshape(4,4)
    return record, _render_optical_pose(view)


class FabricRigReader:
    """Read the registered body's Fabric matrix beside its rendered camera."""
    def __init__(self, native, mount, readback, annotator):
        self.native, self.mount, self.readback, self.annotator = native, copy.deepcopy(mount), readback, annotator
        labels = list(native.model.body_label)
        self.model, self.labels, self.manager = native.model, tuple(labels), native.fabric_manager
        self.stage = self.manager.stage
        path = mount['rig_prim_path']
        if labels.count(path) != 1:
            raise ValueError('camera rig is not one exact registered native body')
        self.index = labels.index(path)
        self.prim = native.fabric_manager.stage.GetPrimAtPath(path)
        self._snapshot(lambda: None)

    def capture_objects(self):
        """Private references for the caller's complete RGB/depth read fence."""
        native = self.native
        return (native.model,native.fabric_manager,native.fabric_manager.stage,
                native.state_0,native.state_0.body_q,native.fabric_manager._body_scales)

    def _snapshot(self, checkpoint):
        import numpy as np
        from .mobile_camera_encoding import canonical_rig
        checkpoint()
        clock = (self.native.simulation_step_count, float(self.native.sim_time))
        state = self.native.state_0  # Never retain a buffer from a prior solve.
        body_buffer = state.body_q
        scale_buffer = self.native.fabric_manager._body_scales
        if (self.native.model is not self.model or tuple(self.model.body_label) != self.labels
                or self.native.fabric_manager is not self.manager or self.manager.stage is not self.stage):
            raise ValueError('registered camera rig model or Fabric stage changed')
        if not self.prim or not self.prim.IsValid():
            raise ValueError('registered camera rig disappeared from Fabric')
        index = self.prim.GetAttribute(FABRIC_INDEX)
        matrix = self.prim.GetAttribute(FABRIC_MATRIX)
        observed_index = index.Get() if index else None
        if (not matrix or isinstance(observed_index, bool) or not isinstance(observed_index, Integral)
                or observed_index != self.index):
            raise ValueError('camera rig Fabric/native index mismatch')
        value = np.asarray(matrix.Get(), dtype=float)
        if value.shape != (4,4):
            raise ValueError('camera rig Fabric world matrix unavailable')
        checkpoint()
        body = body_buffer.numpy()
        scales = scale_buffer.numpy()
        checkpoint()
        if (body.dtype != np.float32 or body.shape != (len(self.labels),7)
                or scales.dtype != np.float32 or scales.shape != (len(self.labels),3)
                or self.native.scene_scale != 1.):
            raise ValueError('registered native rig pose/scale layout or metric units changed')
        pose, scale = body[self.index].copy(), scales[self.index].copy()
        if (self.native.state_0 is not state
                or state.body_q is not body_buffer or self.native.fabric_manager._body_scales is not scale_buffer
                or self.native.model is not self.model or tuple(self.model.body_label) != self.labels
                or self.native.fabric_manager is not self.manager or self.manager.stage is not self.stage
                or (self.native.simulation_step_count,float(self.native.sim_time)) != clock):
            raise RuntimeError('native rig capture changed solve or state buffer')
        rigid, evidence = canonical_rig(value.T,pose,scale)
        evidence['native_clock'] = {'step':clock[0],'simulation_time_s':clock[1]}
        return rigid,evidence,(state,body_buffer,scale_buffer)

    def __call__(self, calibration, checkpoint):
        import numpy as np
        checkpoint()
        before = copy.deepcopy(self.readback.get_render_times())
        checkpoint()
        rig, encoding, objects = self._snapshot(checkpoint)
        from .microduck_stepper import validate_render_times
        validate_render_times(before,encoding['native_clock']['simulation_time_s'])
        checkpoint()
        params, camera = camera_params_record(copy.deepcopy(self.annotator.get_data()), calibration)
        checkpoint()
        after_rig, after_encoding, after_objects = self._snapshot(checkpoint)
        after = copy.deepcopy(self.readback.get_render_times())
        checkpoint()
        if (before != after or rig != after_rig or encoding != after_encoding
                or any(a is not b for a,b in zip(objects,after_objects))):
            raise RuntimeError('rig or render reference changed during camera pose readback')
        expected = np.asarray(rig).reshape(4,4) @ np.asarray(calibration['rig_from_camera']).reshape(4,4)
        if not np.allclose(expected, np.asarray(camera).reshape(4,4), rtol=0, atol=1e-5):
            raise RuntimeError('rendered camera pose differs from registered Fabric rig and rigid mount')
        return {'world_from_rig': list(rig), 'position_error_m': None, 'angular_error_rad': None,
                'render_reference': after,
                'evidence': {'rig_prim_path': self.mount['rig_prim_path'], 'native_body_index': self.index,
                    'fabric_attribute': FABRIC_MATRIX, 'camera_params': params,
                    'quaternion_encoding': encoding,
                    'consistency_tolerance': 1e-5, 'physical_uncertainty_estimated': False}}
