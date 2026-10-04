import base64
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.rgb_detection import detect_rgb_view
from cascade.types import Detection
from test_rgb_triangulation import view


def test_detector_gets_only_bgr_pixels_and_preserves_original_capture_identity():
    rgb = np.empty((128, 128, 3), dtype=np.uint8)
    rgb[:] = [17, 83, 241]
    rgb[3, 5] = [9, 21, 103]
    source = replace(view(), rgb=rgb.tobytes())
    mask = np.zeros((128, 128), bool)
    mask[3:7, 5:11] = True
    calls = []
    def detect(frame, classes):
        calls.append(frame)
        assert classes is None and frame.depth_m is None and frame.t == 0.
        assert frame.capture is frame.T_base_cam is frame.robot_mask is frame.prop_masks is None
        np.testing.assert_array_equal(frame.rgb, rgb[:, :, ::-1])
        assert not frame.rgb.flags.writeable
        return [Detection('predicted can', np.float32(.81), np.array([5., 3., 11., 7.]), mask)]
    result = detect_rgb_view(source, SimpleNamespace(detect=detect))
    assert len(calls) == 1 and result['capture_sha256'] == source.sha256
    assert result['capture_simulation_time_s'] == .04 and result['capture_solver_step'] == 20
    assert not result['depth_supplied'] and not result['object_identity_verified']
    assert not result['physical_admission'] and not result['complete_geometry_verified']
    assert result['position_error_m'] is None
    retained = result['detections'][0]['mask']
    packed = base64.b64decode(retained['data'])
    assert hashlib.sha256(packed).hexdigest() == retained['sha256']
    np.testing.assert_array_equal(np.unpackbits(np.frombuffer(packed, np.uint8), bitorder='little').reshape(128, 128), mask)
    assert retained['foreground_pixels'] == 24
    json.dumps(result, allow_nan=False)
    mask[:] = False
    assert result['detections'][0]['mask']['foreground_pixels'] == 24


def test_empty_predictions_remain_empty_and_failure_is_not_relabelled_as_empty():
    result = detect_rgb_view(view(), SimpleNamespace(detect=lambda *a, **k: []))
    assert result['detections'] == []
    def failed(*args, **kwargs):
        raise RuntimeError('model failed')
    with pytest.raises(RuntimeError, match='model failed'):
        detect_rgb_view(view(), SimpleNamespace(detect=failed))


@pytest.mark.parametrize('fault', ['nan_score', 'bool_score', 'reversed', 'outside', 'broadcast', 'nan_box',
                                  'mask_shape', 'mask_type', 'label', 'obb', 'count'])
def test_invalid_predictions_are_refused_without_clipping_or_mask_repair(fault):
    detection = Detection('can', .8, np.array([1., 2., 10., 11.]))
    if fault == 'nan_score': detection.conf = float('nan')
    if fault == 'bool_score': detection.conf = True
    if fault == 'reversed': detection.bbox = [10., 2., 1., 11.]
    if fault == 'outside': detection.bbox = [-1., 2., 10., 11.]
    if fault == 'broadcast': detection.bbox = [[1., 2., 10., 11.]]
    if fault == 'nan_box': detection.bbox[0] = np.nan
    if fault == 'mask_shape': detection.mask = np.zeros((1, 128), bool)
    if fault == 'mask_type': detection.mask = np.zeros((128, 128), np.uint8)
    if fault == 'label': detection.label = ' '
    if fault == 'obb': detection.obb = np.full((4, 2), 129.)
    predictions = [detection] * (301 if fault == 'count' else 1)
    with pytest.raises(ValueError):
        detect_rgb_view(view(), SimpleNamespace(detect=lambda *a, **k: predictions))


def test_detector_cannot_rebind_detections_to_mutated_pixels():
    def mutate(frame, **kwargs):
        frame.rgb.setflags(write=True)
        frame.rgb[0, 0] = 255
        return []
    with pytest.raises(ValueError, match='modified'):
        detect_rgb_view(view(), SimpleNamespace(detect=mutate))
