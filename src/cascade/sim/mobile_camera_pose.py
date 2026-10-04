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
    rigid_transform(tuple(view.T.flat), 'render world-to-view')
    optical = np.linalg.inv(view.T) @ np.diag([1.,-1.,-1.,1.])
    return record, tuple(optical.flat)


class FabricRigReader:
    """Read the registered body's Fabric matrix beside its rendered camera."""
    def __init__(self, native, mount, readback, annotator):
        self.native, self.mount, self.readback, self.annotator = native, copy.deepcopy(mount), readback, annotator
        labels = list(native.model.body_label)
        self.model, self.labels, self.stage = native.model, tuple(labels), native.fabric_manager.stage
        path = mount['rig_prim_path']
        if labels.count(path) != 1:
            raise ValueError('camera rig is not one exact registered native body')
        self.index = labels.index(path)
        self.prim = native.fabric_manager.stage.GetPrimAtPath(path)
        self._matrix()

    def _matrix(self):
        import numpy as np
        if (self.native.model is not self.model or tuple(self.model.body_label) != self.labels
                or self.native.fabric_manager.stage is not self.stage):
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
        return rigid_transform(tuple(value.T.flat), 'Fabric world-from-rig')

    def __call__(self, calibration, checkpoint):
        import numpy as np
        checkpoint()
        before = copy.deepcopy(self.readback.get_render_times())
        checkpoint()
        rig = self._matrix()
        checkpoint()
        params, camera = camera_params_record(copy.deepcopy(self.annotator.get_data()), calibration)
        checkpoint()
        after_rig = self._matrix()
        after = copy.deepcopy(self.readback.get_render_times())
        checkpoint()
        if before != after or rig != after_rig:
            raise RuntimeError('rig or render reference changed during camera pose readback')
        expected = np.asarray(rig).reshape(4,4) @ np.asarray(calibration['rig_from_camera']).reshape(4,4)
        if not np.allclose(expected, np.asarray(camera).reshape(4,4), rtol=0, atol=1e-5):
            raise RuntimeError('rendered camera pose differs from registered Fabric rig and rigid mount')
        return {'world_from_rig': list(rig), 'position_error_m': None, 'angular_error_rad': None,
                'render_reference': after,
                'evidence': {'rig_prim_path': self.mount['rig_prim_path'], 'native_body_index': self.index,
                    'fabric_attribute': FABRIC_MATRIX, 'camera_params': params,
                    'consistency_tolerance': 1e-5, 'physical_uncertainty_estimated': False}}
