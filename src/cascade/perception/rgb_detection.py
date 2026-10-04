"""Image-only detector observations bound to immutable RGB captures.

These are predicted labels and pixel regions, not measured object identities,
3D geometry, free space or authority to move a robot.
"""
from __future__ import annotations

import base64
import hashlib
import math
import time

import numpy as np

from cascade.types import Frame
from .rgb_triangulation import RgbView


def detect_rgb_view(view: RgbView, detector, *, max_detections=300):
    """Run the existing detector once using only a copied BGR image and K.

    Capture clocks remain in the returned provenance. Frame.t is zero because
    this adapter does not assert that an archived image is fresh now.
    """
    if not isinstance(view, RgbView):
        raise ValueError('validated RGB view required')
    if type(max_detections) is not int or not 1 <= max_detections <= 300:
        raise ValueError('detector output bound must be 1..300')
    rgb = np.frombuffer(view.rgb, dtype=np.uint8).reshape(view.height, view.width, 3)
    bgr = rgb[:, :, ::-1].copy()
    bgr.setflags(write=False)
    frame = Frame(rgb=bgr, depth_m=None, K=np.array(view.intrinsics).reshape(3, 3), t=0.)
    started = time.monotonic()
    predictions = detector.detect(frame, classes=None)
    finished = time.monotonic()
    if type(predictions) not in (list, tuple) or len(predictions) > max_detections:
        raise ValueError('detector returned an unbounded prediction collection')

    def coordinates(value, shape):
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError('detector coordinates must have the declared finite shape')
        return array.astype(np.float64)

    records = []
    for detection in predictions:
        if type(detection.label) is not str or not detection.label.strip() or len(detection.label) > 256:
            raise ValueError('detector label must be bounded nonempty text')
        if isinstance(detection.conf, (bool, np.bool_)) or not isinstance(detection.conf, (float, int, np.floating)):
            raise ValueError('detector score must be numeric')
        score = float(detection.conf)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('detector score must be finite and within 0..1')
        box = coordinates(detection.bbox, (4,))
        if not (0 <= box[0] < box[2] <= view.width and 0 <= box[1] < box[3] <= view.height):
            raise ValueError('detector box is outside the image or reversed')
        record = {'predicted_label': detection.label, 'detector_score': score,
                  'bbox_xyxy_px': box.tolist(), 'mask': None, 'obb_xy_px': None}
        if detection.mask is not None:
            mask = np.asarray(detection.mask)
            if mask.shape != (view.height, view.width) or mask.dtype != np.bool_:
                raise ValueError('detector mask must be full-image boolean pixels')
            packed = np.packbits(mask.reshape(-1), bitorder='little').tobytes()
            record['mask'] = {'encoding': 'base64-packbits-little-row-major',
                              'width': view.width, 'height': view.height,
                              'data': base64.b64encode(packed).decode('ascii'),
                              'sha256': hashlib.sha256(packed).hexdigest(),
                              'foreground_pixels': int(mask.sum())}
        if detection.obb is not None:
            obb = coordinates(detection.obb, (4, 2))
            if (np.any(obb < 0) or np.any(obb[:, 0] > view.width) or np.any(obb[:, 1] > view.height)):
                raise ValueError('detector oriented box is outside the image')
            record['obb_xy_px'] = obb.tolist()
        records.append(record)
    # No detector write, even through a temporarily writable view, can change
    # which pixels this observation claims to describe.
    if bgr[:, :, ::-1].tobytes() != view.rgb:
        raise ValueError('detector modified its input pixels')
    return {'schema': 'cascade.rgb-detections.v1', 'capture_sha256': view.sha256,
            'rgb_sha256': hashlib.sha256(view.rgb).hexdigest(), 'camera_id': view.camera_id,
            'model_identity_sha256': view.model_identity_sha256, 'epoch': view.epoch,
            'calibration_sha256': view.calibration_sha256, 'capture_solver_step': view.solver_step,
            'capture_simulation_time_s': view.simulation_time_s,
            'started_monotonic_s': started, 'finished_monotonic_s': finished,
            'detections': records, 'depth_supplied': False, 'object_identity_verified': False,
            'position_error_m': None, 'complete_geometry_verified': False,
            'physical_admission': False}
