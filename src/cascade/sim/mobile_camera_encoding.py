"""Explicit SDK48 quaternion encoding, separate from unknown physical error."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from ..sensing.models import rigid_transform


SCHEMA = 'cascade.fabric-quaternion-encoding.v1'
SOURCE_PATHS = {
    'isaacsim.physics.newton.impl.fabric': 'exts/isaacsim.physics.newton/isaacsim/physics/newton/impl/fabric.py',
    'newton._src.solvers.mujoco.kernels': 'exts/isaacsim.pip.newton/pip_prebundle/newton/_src/solvers/mujoco/kernels.py',
    'mujoco_warp._src.smooth': 'exts/isaacsim.pip.newton/pip_prebundle/mujoco_warp/_src/smooth.py',
    'warp._src.math': 'extscache/omni.warp.core-1.17.0+lx64/warp/_src/math.py',
    'warp/native/quat.h': 'extscache/omni.warp.core-1.17.0+lx64/warp/native/quat.h',
}
SOURCE_SHA256 = dict(zip(SOURCE_PATHS, (
    'c5f2fd8de9e069e96c853432899b13df4489f5312112051ab8dca0cddb1c1766',
    'bd770d39d20c208e980477e0186799f531364b03f16d76968ee69bacdc490a3d',
    'f2993c13ba76c85d9d948eda316de76c987dbecb765b34f31df8e8c9d3d99554',
    '49d27059ca60350cdcb7e775d51b6df4c36b7837b66b8fe4675648b9b1c709b8',
    '1ef29e23117b3da391c12d5e4959f52f8cc83ccb2c8b5ced474a63499a20ccd2',
)))
U32 = 2.**-24
GAMMA7 = 7*U32/(1-7*U32)
# Positive four-term dot (at most seven rounded operations), then sqrt,
# reciprocal and component multiply. FMA contraction only reduces this bound.
NORM2_MIN = (1-U32)**4 / ((1+GAMMA7)*(1+U32)**2)
NORM2_MAX = (1+U32)**4 / ((1-GAMMA7)*(1-U32)**2)


def encoding_descriptor():
    return {'schema':SCHEMA, 'source_sha256':dict(SOURCE_SHA256), 'float32_unit_roundoff':U32,
            'quaternion_norm_squared_range':[NORM2_MIN,NORM2_MAX],
            'fabric_check':'outward float32 arithmetic intervals; unit scale',
            'fast_math':False, 'physical_uncertainty_estimated':False}


def encoding_policy(release, recipe):
    if recipe != 'isaac62_48b2d951':
        raise ValueError('Fabric quaternion encoding requires exact SDK48')
    actual = {name:hashlib.sha256((Path(release)/path).read_bytes()).hexdigest()
              for name,path in SOURCE_PATHS.items()}
    if actual != SOURCE_SHA256:
        raise ValueError('Fabric quaternion encoding source mismatch')
    return encoding_descriptor()


def verify_encoding_sources(release, recipe):
    policy = encoding_policy(release,recipe)
    warp = importlib.import_module('warp')
    for name,relative in SOURCE_PATHS.items():
        path = (Path(warp.__file__).parent/'native/quat.h' if name=='warp/native/quat.h'
                else Path(importlib.import_module(name).__file__))
        if (path.resolve() != (Path(release)/relative).resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256[name]):
            raise ValueError('actual Fabric encoding import differs: '+name)
    for name in ('mujoco_warp._src.smooth','isaacsim.physics.newton.impl.fabric'):
        if warp.get_module(name).options.get('fast_math') is not False:
            raise ValueError('Fabric encoding requires ordinary floating-point arithmetic')
    return policy


def _outward(low, high):
    """Enclose exact intermediates and their float32 roundings, including FMA."""
    import numpy as np
    low,high = np.nextafter(float(low),-np.inf),np.nextafter(float(high),np.inf)
    a,b = np.float32(low),np.float32(high)
    if float(a)>low:a=np.nextafter(a,np.float32(-np.inf))
    if float(b)<high:b=np.nextafter(b,np.float32(np.inf))
    return float(a),float(b)


def _add(a,b):
    return _outward(a[0]+b[0],a[1]+b[1])


def _mul(a,b):
    values = [x*y for x in a for y in b]
    return _outward(min(values),max(values))


def _sub(a,b):
    return _add(a,(-b[1],-b[0]))


def _rotation_intervals(q):
    # quat_to_matrix calls quat_rotate(q,e_j). Use that literal expression,
    # not a mathematically equivalent unit-quaternion formula.
    one,two = (1.,1.),(2.,2.)
    w = (float(q[3]),)*2
    v = [(float(x),)*2 for x in q[:3]]
    c = _sub(_mul(_mul(two,w),w),one)
    columns = []
    for j in range(3):
        axis = [((1.,1.) if i==j else (0.,0.)) for i in range(3)]
        d = _mul(two,_add(_add(_mul(v[0],axis[0]),_mul(v[1],axis[1])),_mul(v[2],axis[2])))
        columns.append([_add(_add(_mul(axis[i],c),_mul(v[i],d)),
            _mul(_mul(_sub(_mul(v[(i+1)%3],axis[(i+2)%3]),
                          _mul(v[(i+2)%3],axis[(i+1)%3])),w),two)) for i in range(3)])
    return [[columns[j][i] for j in range(3)] for i in range(3)]


def canonical_rig(matrix, body_pose, body_scale):
    """Decode one co-captured native pose; never infer a pose from a matrix."""
    import numpy as np
    matrix,body_pose,body_scale = map(np.asarray,(matrix,body_pose,body_scale))
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or np.max(abs(matrix))>np.finfo(np.float32).max
            or not np.array_equal(matrix,matrix.astype(np.float32).astype(float))
            or body_pose.dtype!=np.float32 or body_pose.shape!=(7,) or not np.isfinite(body_pose).all()
            or body_scale.dtype!=np.float32 or body_scale.shape!=(3,) or not np.array_equal(body_scale,[1,1,1])
            or not np.array_equal(matrix[3],[0,0,0,1])
            or not np.array_equal(matrix[:3,3],body_pose[:3])):
        raise ValueError('Fabric encoding requires exact float32 pose, translation and unit scale')
    q = body_pose[3:].astype(float)
    norm2 = float(q@q)
    if not NORM2_MIN<=norm2<=NORM2_MAX:
        raise ValueError('quaternion exceeds derived float32 normalization bound')
    intervals = np.asarray(_rotation_intervals(q))
    if not ((intervals[:,:,0]<=matrix[:3,:3]).all() and (matrix[:3,:3]<=intervals[:,:,1]).all()):
        raise ValueError('Fabric matrix is not the captured quaternion encoding')
    x,y,z,w = q/np.sqrt(norm2)
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    result = np.eye(4);result[:3,:3]=rotation;result[:3,3]=body_pose[:3]
    rigid = rigid_transform(tuple(result.flat),'decoded Fabric quaternion')
    # This bounds coordinate differences per metre about the rig origin,
    # not rotation uncertainty, sensor accuracy or physical localization.
    bound = np.nextafter(np.maximum(abs(rotation-intervals[:,:,0]),abs(rotation-intervals[:,:,1])),np.inf)
    # ||E||2 <= ||E||F <= 3*max_ij |E_ij|. Round outward after both the
    # subtraction above and this multiplication; do not rely on a BLAS norm.
    point_bound = float(np.nextafter(3*float(np.max(bound)),np.inf))
    return rigid, {'schema':SCHEMA,'raw_body_pose_xyzw':body_pose.astype(float).tolist(),
        'raw_fabric_world_from_rig':matrix.astype(float).tolist(),
        'body_scale':body_scale.astype(float).tolist(),'quaternion_norm_squared':norm2,
        'rotation_entry_intervals':intervals.tolist(),
        'rotation_max_encoding_difference':float(np.max(abs(rotation-matrix[:3,:3]))),
        'point_encoding_error_per_m':point_bound,
        'translation_encoding_difference_m':0., 'physical_uncertainty_estimated':False}
