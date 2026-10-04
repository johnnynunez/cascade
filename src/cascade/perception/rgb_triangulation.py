"""Sparse metric estimates from simultaneous RGB views and declared calibration.

No depth image, object pose, simulator, detector or actuator is consulted here.
Matching features and reprojection agreement do not establish correspondence
truth, calibration uncertainty, object extent or free space. Results cannot
authorize motion or substitute for a measured RGB-D capture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


def _numbers(value, size, name):
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f'{name} requires {size} numbers')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError(f'{name} must contain finite numbers')
    return tuple(float(v) for v in value)


def _token(value, name):
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f'{name} requires a bounded identifier')


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} requires a SHA256')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RgbView:
    """Immutable RGB bytes and camera pose from the same completed capture.

    Coordinates are RGB array indices, x right/y down; optical z is forward.
    The explicit pixel offset translates an array index into the declared K
    convention. A changing wrist-camera pose belongs to each view, not a cache.
    Identities bind supplied data; they cannot attest an untrusted producer.
    """
    camera_id: str
    world_frame_id: str
    model_identity_sha256: str
    calibration_sha256: str
    epoch: str
    solver_step: int
    simulation_time_s: float
    width: int
    height: int
    intrinsics: tuple[float, ...]
    world_from_camera: tuple[float, ...]
    pixel_center_offset_uv: tuple[float, float]
    rgb: bytes

    def __post_init__(self):
        for key in ('camera_id', 'world_frame_id', 'epoch'):
            _token(getattr(self, key), key)
        for key in ('model_identity_sha256', 'calibration_sha256'):
            _sha(getattr(self, key), key)
        if type(self.solver_step) is not int or not 0 <= self.solver_step < 2**53:
            raise ValueError('solver_step requires an exact nonnegative integer')
        if type(self.simulation_time_s) not in (int, float) or not math.isfinite(self.simulation_time_s) or self.simulation_time_s < 0:
            raise ValueError('simulation_time_s must be finite and nonnegative')
        if (type(self.width) is not int or type(self.height) is not int
                or not 1 <= self.width <= 2048 or not 1 <= self.height <= 2048
                or self.width * self.height > 1024*1024):
            raise ValueError('RGB dimensions exceed the pixel bound')
        if type(self.rgb) is not bytes or len(self.rgb) != self.width*self.height*3:
            raise ValueError('complete immutable packed RGB bytes required')
        k = _numbers(self.intrinsics, 9, 'intrinsics')
        if (not 0 < k[0] <= 1e6 or not 0 < k[4] <= 1e6
                or k[1] != 0 or k[3] != 0 or k[6:] != (0., 0., 1.)
                or not 0 <= k[2] <= self.width or not 0 <= k[5] <= self.height):
            raise ValueError('finite zero-skew pinhole intrinsics required')
        transform = _numbers(self.world_from_camera, 16, 'world_from_camera')
        matrix = np.asarray(transform).reshape(4, 4)
        rotation = matrix[:3, :3]
        if (transform[12:] != (0., 0., 0., 1.) or np.max(np.abs(matrix[:3, 3])) > 10000
                or np.max(np.abs(rotation)) > 1.0000001
                or np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-7
                or abs(np.linalg.det(rotation) - 1.) > 1e-7):
            raise ValueError('bounded rigid camera transform required')
        offset = _numbers(self.pixel_center_offset_uv, 2, 'pixel_center_offset_uv')
        if offset not in ((0., 0.), (.5, .5)):
            raise ValueError('unsupported pixel-center convention')
        for key, value in (('intrinsics', k), ('world_from_camera', transform),
                           ('pixel_center_offset_uv', offset)):
            object.__setattr__(self, key, value)

    @property
    def sha256(self):
        data = asdict(self)
        data['rgb'] = hashlib.sha256(self.rgb).hexdigest()
        return _digest(data)


@dataclass(frozen=True)
class TriangulationPolicy:
    max_pairs: int = 512
    min_parallax_rad: float = math.radians(1.)
    max_reprojection_px: float = .75
    max_ray_gap_m: float = .01
    min_depth_m: float = .01
    max_depth_m: float = 2.

    def __post_init__(self):
        if type(self.max_pairs) is not int or not 1 <= self.max_pairs <= 4096:
            raise ValueError('max_pairs must be in 1..4096')
        for key in ('min_parallax_rad', 'max_reprojection_px', 'max_ray_gap_m', 'min_depth_m', 'max_depth_m'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('finite positive triangulation gates required')
        if not 1e-4 <= self.min_parallax_rad < math.pi/2 or not self.min_depth_m < self.max_depth_m <= 100:
            raise ValueError('bounded triangulation angle and depth interval required')


def _ray(view, uv):
    u, v = uv
    if not 0 <= u < view.width or not 0 <= v < view.height:
        raise ValueError('feature lies outside its RGB image')
    k, offset = view.intrinsics, view.pixel_center_offset_uv
    direction = np.array([(u + offset[0] - k[2])/k[0], (v + offset[1] - k[5])/k[4], 1.])
    # hypot scales before squaring, including for unusual but finite intrinsics.
    direction /= math.hypot(*direction)
    transform = np.asarray(view.world_from_camera).reshape(4, 4)
    return transform[:3, 3], transform[:3, :3] @ direction


def _project(view, point):
    transform = np.asarray(view.world_from_camera).reshape(4, 4)
    camera = transform[:3, :3].T @ (point - transform[:3, 3])
    k, offset = view.intrinsics, view.pixel_center_offset_uv
    if camera[2] <= 0:
        raise ValueError('triangulated point is behind a camera')
    return (np.array([k[0]*camera[0]/camera[2]+k[2]-offset[0],
                      k[4]*camera[1]/camera[2]+k[5]-offset[1]]), camera[2])


def triangulate_pairs(left, right, pairs, *, policy=TriangulationPolicy()):
    """Retain a result for every proposed (left_u,v,right_u,v) correspondence.

    A passing row is an estimate with unknown uncertainty, never a confirmed
    object location. Caller-selected matches must name the same surface point;
    bounding-box centers from different views do not meet that condition.
    """
    if type(left) is not RgbView or type(right) is not RgbView or type(policy) is not TriangulationPolicy:
        raise ValueError('typed RGB views and triangulation policy required')
    identity = ('world_frame_id', 'model_identity_sha256', 'epoch', 'solver_step', 'simulation_time_s')
    if left.camera_id == right.camera_id or any(getattr(left, k) != getattr(right, k) for k in identity):
        raise ValueError('two distinct simultaneous cameras in the same model/epoch/frame required')
    if not isinstance(pairs, (tuple, list)) or len(pairs) > policy.max_pairs:
        raise ValueError('correspondence count exceeds the explicit bound')
    pairs = tuple(_numbers(pair, 4, 'pixel correspondence') for pair in pairs)
    rows = []
    for index, pair in enumerate(pairs):
        row = {'index': index, 'pixels': list(pair), 'status': 'unverified', 'position_error_m': None}
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                a, da = _ray(left, pair[:2]); b, db = _ray(right, pair[2:])
                cosine = float(np.clip(da @ db, -1., 1.))
                angle = math.acos(abs(cosine))
                if angle < policy.min_parallax_rad:
                    raise ValueError('insufficient ray parallax')
                delta = b-a
                aa, bb = float(da @ delta), float(db @ delta)
                distance_a = (aa-cosine*bb)/(1.-cosine*cosine)
                distance_b = (cosine*aa-bb)/(1.-cosine*cosine)
                if min(distance_a, distance_b) <= 0:
                    raise ValueError('correspondence intersects behind a camera')
                pa, pb = a + distance_a*da, b + distance_b*db
                point = (pa+pb)/2.
                gap = float(np.linalg.norm(pa-pb))
                reprojections = [_project(view, point) for view in (left, right)]
                errors = [float(np.linalg.norm(uv - pair[2*j:2*j+2]))
                          for j, (uv, _) in enumerate(reprojections)]
                depths = [float(depth) for _, depth in reprojections]
                if not np.isfinite([*point, gap, *errors, *depths]).all():
                    raise ValueError('nonfinite triangulation result')
                if any(not policy.min_depth_m <= depth <= policy.max_depth_m for depth in depths):
                    raise ValueError('triangulation outside declared optical depth interval')
                if gap > policy.max_ray_gap_m or max(errors) > policy.max_reprojection_px:
                    raise ValueError('ray gap or reprojection gate failed')
                row.update(status='estimated', position_world_m=point.tolist(),
                           parallax_rad=angle, ray_gap_m=gap, reprojection_px=errors, optical_depth_m=depths)
        except (ValueError, ArithmeticError) as exc:
            row['reason'] = str(exc)
        rows.append(row)
    return {'schema': 'cascade.sparse-rgb-triangulation.v1', 'views_sha256': [left.sha256, right.sha256],
            'policy': asdict(policy), 'pairs_sha256': _digest(pairs), 'rows': rows,
            'estimated_points': sum(row['status'] == 'estimated' for row in rows),
            'position_error_m': None, 'physical_admission': False,
            'object_identity_verified': False, 'complete_geometry_verified': False}


def match_rgb_features(left, right, *, max_features=512, ratio=.7):
    """Optional CPU SIFT: mutual descriptor matches, not object identification.

    Only the supplied RGB bytes enter OpenCV. A geometric check must follow;
    repeated texture may still yield a plausible but incorrect correspondence.
    """
    if type(left) is not RgbView or type(right) is not RgbView:
        raise ValueError('typed RGB views required')
    if type(max_features) is not int or not 2 <= max_features <= 4096:
        raise ValueError('max_features must be in 2..4096')
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError('descriptor ratio must be in (0,1)')
    import cv2
    sift = cv2.SIFT_create(nfeatures=max_features)
    extracted = []
    for view in (left, right):
        rgb = np.frombuffer(view.rgb, np.uint8).reshape(view.height, view.width, 3)
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        # SIFT may exceed nfeatures when responses tie; bound matcher inputs too.
        extracted.append((keypoints[:max_features], None if descriptors is None else descriptors[:max_features]))
    (ka, da), (kb, db) = extracted
    if da is None or db is None or min(len(da), len(db)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def selected(a, b):
        return {m.queryIdx: m.trainIdx for neighbors in matcher.knnMatch(a, b, k=2)
                if len(neighbors) == 2 for m, n in [neighbors] if m.distance < ratio*n.distance}
    forward, backward = selected(da, db), selected(db, da)
    return [(*ka[i].pt, *kb[j].pt) for i, j in sorted(forward.items())
            if backward.get(j) == i][:max_features]
