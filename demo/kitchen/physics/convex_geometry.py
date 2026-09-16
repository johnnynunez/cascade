"""Bind authored can/fruit hulls to live USD and physical tensor readback.

This module never changes the stage or issues motion. Authored vertices retain
the full footprint; PhysX's returned collision representation supplies support.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

CONVEX_TARGETS = {'tomato_can': 'cylinder', 'lemon': 'ellipsoid', 'orange': 'ellipsoid'}


def validate_specs(value, dimensions):
    if not isinstance(dimensions, dict):
        raise ValueError('Convex specifications require independently configured prop dimensions')
    configured = set(dimensions) - {'green_cube', 'pink_cube'}
    if (not {'tomato_can', 'lemon'} <= configured <= set(CONVEX_TARGETS)
            or not isinstance(value, dict) or set(value) != configured):
        raise ValueError('Expected configured can and lemon specifications, plus orange only when configured')
    result = {}
    for name, kind in CONVEX_TARGETS.items():
        if name not in configured:
            continue
        spec = value[name]
        if not isinstance(spec, dict) or spec.get('proxy') != kind:
            raise ValueError('Convex target shape differs from its reviewed name')
        mass = spec.get('mass_kg')
        if type(mass) not in (int, float) or not math.isfinite(mass) or not 0 < mass <= 1:
            raise ValueError('Convex target requires a bounded positive mass')
        result[name] = {'proxy': kind, 'mass_kg': float(mass),
                        'dimensions_m': list(dimensions[name])}
        if spec.get('dimensions_m') != result[name]['dimensions_m']:
            raise ValueError('Convex target dimensions must match the scene contract')
    return result


def expected_hull(spec):
    """Independent metric contract: 24 radial segments, 12 ellipsoid rings."""
    a, b, c = np.asarray(spec['dimensions_m'], float) / 2
    angles = [2 * math.pi * j / 24 for j in range(24)]
    if spec['proxy'] == 'cylinder':
        points = [[a * math.cos(t), b * math.sin(t), z]
                  for z in (-c, c) for t in angles]
        faces = [list(range(23, -1, -1)), list(range(24, 48))]
        faces += [[j, (j + 1) % 24, (j + 1) % 24 + 24, j + 24] for j in range(24)]
    elif spec['proxy'] == 'ellipsoid':
        points = [[0, 0, -c]]
        for i in range(1, 12):
            latitude = -math.pi / 2 + math.pi * i / 12
            points.extend([[a * math.cos(latitude) * math.cos(t),
                            b * math.cos(latitude) * math.sin(t), c * math.sin(latitude)]
                           for t in angles])
        points.append([0, 0, c])
        faces = [[0, 1 + (j + 1) % 24, 1 + j] for j in range(24)]
        for ring in range(10):
            lo, hi = 1 + ring * 24, 1 + (ring + 1) * 24
            faces.extend([[lo + j, lo + (j + 1) % 24, hi + (j + 1) % 24, hi + j]
                          for j in range(24)])
        faces.extend([[241 + j, 241 + (j + 1) % 24, 265] for j in range(24)])
    else:
        raise ValueError('Unsupported authored convex hull')
    return (np.asarray(points, dtype=np.float32).astype(float),
            [len(f) for f in faces], [i for f in faces for i in f])


def snapshot_code(object_name):
    if object_name not in CONVEX_TARGETS:
        raise ValueError('Expected a named convex target')
    return '''
from pxr import Usd as _conv_USD, UsdGeom as _conv_UG, Gf as _conv_GF
def _convex_live_snapshot():
    name = OBJECT_LITERAL
    path = "/World_Props/" + name
    body = stage.GetPrimAtPath(path)
    prim = stage.GetPrimAtPath(path + "/Collision")
    if not body or not prim or not body.IsActive() or not prim.IsActive() or not prim.IsA(_conv_UG.Mesh):
        raise RuntimeError("Missing active convex target body/collider")
    mesh = _conv_UG.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if points is None or not 4 <= len(points) <= 10000:
        raise RuntimeError("Expected a bounded convex point array")
    vertices = _obs_np.asarray(points, dtype=float)
    cache = _conv_UG.XformCache()
    linear = _obs_np.asarray(cache.GetLocalToWorldTransform(body), dtype=float)[:3,:3]
    local = _conv_UG.Xformable(prim).GetLocalTransformation()
    colliders = sorted(str(p.GetPath()) for p in _conv_USD.PrimRange(body)
                       if p.IsActive() and p.HasAPI(_obs_UP.CollisionAPI))
    bodies = sorted(str(p.GetPath()) for p in _conv_USD.PrimRange(body)
                    if p.IsActive() and p.HasAPI(_obs_UP.RigidBodyAPI))
    view = globals()["_kitchen_passive_observer_views_v1"][path]
    with _obs_backend("tensor", raise_on_unsupported=True, raise_on_fallback=True):
        if not view.is_physics_tensor_entity_valid():
            raise RuntimeError("Convex target tensor view is invalid")
        mass = view.get_masses()
        com, com_quat = view.get_coms()
        inertia = view.get_inertias()
    support = _convex_support_snapshot(prim, vertices,
        list(mesh.GetFaceVertexCountsAttr().Get()), list(mesh.GetFaceVertexIndicesAttr().Get()))
    return {"body_name": name, "collider_path": path + "/Collision",
        "stage_id": support["stage_id"], "physx_support": support,
        "frame": "body_local", "units": "m", "vertices_m": vertices.tolist(),
        "face_vertex_counts": list(mesh.GetFaceVertexCountsAttr().Get()),
        "face_vertex_indices": list(mesh.GetFaceVertexIndicesAttr().Get()),
        "subdivision_none": mesh.GetSubdivisionSchemeAttr().Get() == "none",
        "collision_approximation": _obs_UP.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
        "collision_enabled": _obs_UP.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True,
        "one_collider_one_body": colliders == [path + "/Collision"] and bodies == [path],
        "dynamic_body": (_obs_UP.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is True
            and _obs_UP.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is False),
        "body_no_scale_or_shear": bool(_obs_np.allclose(linear @ linear.T, _obs_np.eye(3), atol=1e-6, rtol=0)),
        "collider_to_body_identity": bool(not _conv_UG.Xformable(prim).GetResetXformStack()
            and _obs_np.allclose(_obs_np.asarray(local), _obs_np.eye(4), atol=1e-9, rtol=0)
            and _obs_np.allclose(_obs_np.asarray(cache.GetLocalToWorldTransform(prim)),
                _obs_np.asarray(cache.GetLocalToWorldTransform(body)), atol=1e-9, rtol=0)),
        "mass_kg": float(_obs_array(mass).reshape(1)[0]),
        "com_body_m": _obs_array(com).reshape(3).tolist(),
        "com_orientation_wxyz": _obs_array(com_quat).reshape(4).tolist(),
        "inertia_kg_m2": _obs_array(inertia).reshape(3,3).tolist(),
        "mass_properties_channel": "physics_tensor",
        "tensor_devices": [str(v.device) for v in (mass, com, com_quat, inertia)]}
_gpu_observed["scene_geometry"]["convex_collider"] = _convex_live_snapshot()
'''.replace('OBJECT_LITERAL', repr(object_name)).replace(
    'def _convex_live_snapshot():', support_snapshot_code() + '\ndef _convex_live_snapshot():')


def support_snapshot_code():
    """Query the exact live stage/prim; never author physics or read saved proof."""
    return '''
def _convex_support_snapshot(prim, vertices, counts, indices):
    import omni.physx as _support_physx
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult as _support_result
    from pxr import UsdUtils as _support_utils, PhysicsSchemaTools as _support_paths
    if str(engine).lower() != "physx" or not _tl.is_playing():
        raise RuntimeError("Convex support requires live PhysX")
    stage_id = int(_support_utils.StageCache.Get().GetId(stage).ToLongInt())
    path = str(prim.GetPath())
    if stage_id <= 0 or stage.GetPrimAtPath(path) != prim:
        raise RuntimeError("Convex support stage/prim binding failed")
    replies = []
    def received(result, convexes):
        replies.append((result, convexes))
    task = _support_physx.get_physx_cooking_interface().request_convex_collision_representation(
        stage_id, _support_paths.sdfPathToInt(path), False, received)
    if (len(replies) != 1 or replies[0][0] != _support_result.RESULT_VALID
            or len(replies[0][1]) != 1):
        raise RuntimeError("PhysX did not return one valid synchronous convex representation")
    returned = _obs_np.asarray([list(v) for v in replies[0][1][0].vertices], dtype=float)
    if (returned.ndim != 2 or returned.shape[1:] != (3,)
            or not 4 <= len(returned) <= len(vertices)
            or not _obs_np.isfinite(returned).all()
            or _obs_np.linalg.matrix_rank(returned-returned[0]) != 3):
        raise RuntimeError("PhysX returned invalid solid convex support geometry")
    # Retain exact returned coordinates as indices into the independently bound
    # authored array. This is lossless and stays within the bridge's wire limit.
    lookup = {tuple(v): i for i, v in enumerate(vertices)}
    if len(lookup) != len(vertices) or any(tuple(v) not in lookup for v in returned):
        raise RuntimeError("Unsupported PhysX representation outside authored vertices")
    selected = [lookup[tuple(v)] for v in returned]
    if (len(set(selected)) != len(selected)
            or not _obs_np.array_equal(returned.min(axis=0), vertices.min(axis=0))
            or not _obs_np.array_equal(returned.max(axis=0), vertices.max(axis=0))):
        raise RuntimeError("PhysX support representation has repeated vertices or changed bounds")
    def digest(points):
        return _obs_hash.sha256(_obs_np.ascontiguousarray(points, dtype="<f4").tobytes()).hexdigest()
    topology = _obs_json.dumps([counts, indices], separators=(",", ":")).encode()
    return {"method": "physx_collision_representation", "engine": "physx",
        "stage_id": stage_id, "collider_path": path,
        "physics_step": int(_obs_SM.get_num_physics_steps()),
        "result": "RESULT_VALID", "convex_count": 1, "frame": "body_local", "units": "m",
        "source_vertices_f32_sha256": digest(vertices),
        "source_topology_sha256": _obs_hash.sha256(topology).hexdigest(),
        "vertex_count": len(returned), "vertices_f32_sha256": digest(returned),
        "authored_vertex_indices": selected}
'''


def audit_support_binding(samples, *, object_name):
    """Bind every returned support hull to that sample's authored collider."""
    checks = {key: bool(samples) for key in (
        'physx_support_live_identity', 'physx_support_source_matches',
        'physx_support_solid_subset', 'physx_support_geometry_unchanged')}
    reference = None
    first_vertices = None
    for sample in samples:
        item = sample.get('scene_geometry', {}).get('convex_collider', {})
        support = item.get('physx_support', {})
        try:
            stage_id = item.get('stage_id')
            checks['physx_support_live_identity'] &= bool(
                sample.get('engine') == support.get('engine') == 'physx'
                and support.get('method') == 'physx_collision_representation'
                and type(stage_id) is int and stage_id > 0
                and type(support.get('stage_id')) is int and support.get('stage_id') == stage_id
                and item.get('collider_path') == support.get('collider_path')
                    == '/World_Props/' + object_name + '/Collision'
                and type(support.get('physics_step')) is int
                and support['physics_step'] == sample['physics_step']
                and support.get('result') == 'RESULT_VALID'
                and type(support.get('convex_count')) is int and support.get('convex_count') == 1
                and support.get('frame') == 'body_local' and support.get('units') == 'm'
                and item.get('collider_to_body_identity') is True
                and item.get('body_no_scale_or_shear') is True)
            vertices = np.asarray(item['vertices_m'], float)
            source_hash = hashlib.sha256(np.ascontiguousarray(vertices, dtype='<f4').tobytes()).hexdigest()
            topology = json.dumps([item['face_vertex_counts'], item['face_vertex_indices']],
                                  separators=(',', ':')).encode()
            checks['physx_support_source_matches'] &= bool(
                support.get('source_vertices_f32_sha256') == source_hash
                and support.get('source_topology_sha256') == hashlib.sha256(topology).hexdigest())
            selected = support.get('authored_vertex_indices')
            valid = bool(isinstance(selected, list) and 4 <= len(selected) <= len(vertices)
                and all(type(i) is int and 0 <= i < len(vertices) for i in selected)
                and len(set(selected)) == len(selected)
                and type(support.get('vertex_count')) is int and support['vertex_count'] == len(selected))
            if not valid:
                raise ValueError('Missing or invalid support vertex indices')
            returned = vertices[selected]
            returned_hash = hashlib.sha256(np.ascontiguousarray(returned, dtype='<f4').tobytes()).hexdigest()
            checks['physx_support_solid_subset'] &= bool(np.isfinite(returned).all()
                and np.linalg.matrix_rank(returned-returned[0]) == 3
                and np.array_equal(returned.min(axis=0), vertices.min(axis=0))
                and np.array_equal(returned.max(axis=0), vertices.max(axis=0))
                and support.get('vertices_f32_sha256') == returned_hash)
            identity = {key: value for key, value in support.items() if key != 'physics_step'}
            if reference is None:
                reference, first_vertices = identity, returned.copy()
            checks['physx_support_geometry_unchanged'] &= identity == reference
        except (AttributeError, KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
            checks['physx_support_solid_subset'] = False
    return {'pass': bool(all(checks.values())), 'checks': checks, 'samples': len(samples),
            'source': 'live PhysX collision representation for support only; authored footprint retained',
            'binding': reference, 'vertices_m': first_vertices.tolist() if first_vertices is not None else None}


def audit_binding(samples, *, object_name, expected):
    if object_name not in CONVEX_TARGETS or object_name not in expected.get('convex_colliders', {}):
        raise ValueError('Convex proof requires an independently configured collider')
    spec = expected['convex_colliders'][object_name]
    vertices, counts, indices = expected_hull(spec)
    checks = {key: bool(samples) for key in ('convex_body_identity', 'authored_convex_vertices_match',
        'convex_topology_matches', 'convex_body_and_collider_enabled',
        'convex_body_and_child_transforms_match', 'convex_tensor_mass_and_com_match',
        'convex_tensor_inertia_valid')}
    first = None
    for sample in samples:
        item = sample.get('scene_geometry', {}).get('convex_collider', {})
        checks['convex_body_identity'] &= bool(item.get('body_name') == object_name
            and item.get('collider_path') == '/World_Props/' + object_name + '/Collision'
            and item.get('frame') == 'body_local' and item.get('units') == 'm')
        try:
            actual = np.asarray(item.get('vertices_m'), float)
            valid_vertices = bool(actual.shape == vertices.shape and np.isfinite(actual).all()
                and np.array_equal(actual.astype(np.float32).astype(float), actual)
                and np.allclose(actual, vertices, rtol=0, atol=1e-9))
            if first is None:
                first = actual.copy()
            valid_vertices &= np.array_equal(actual, first)
        except (ValueError, TypeError):
            valid_vertices = False
        checks['authored_convex_vertices_match'] &= bool(valid_vertices)
        checks['convex_topology_matches'] &= bool(item.get('face_vertex_counts') == counts
            and item.get('face_vertex_indices') == indices and item.get('subdivision_none') is True
            and item.get('collision_approximation') == 'convexHull')
        checks['convex_body_and_collider_enabled'] &= all(item.get(k) is True for k in
            ('one_collider_one_body', 'dynamic_body', 'collision_enabled'))
        checks['convex_body_and_child_transforms_match'] &= all(item.get(k) is True for k in
            ('body_no_scale_or_shear', 'collider_to_body_identity'))
        try:
            com = np.asarray(item.get('com_body_m'), float)
            quat = np.asarray(item.get('com_orientation_wxyz'), float)
            devices = item.get('tensor_devices', [])
            mass_ok = bool(item.get('mass_properties_channel') == 'physics_tensor'
                and len(devices) == 4 and all(isinstance(d, str) and d.startswith('cuda:') for d in devices)
                and np.isclose(item.get('mass_kg'), spec['mass_kg'], atol=1e-7, rtol=0)
                and com.shape == (3,) and np.isfinite(com).all() and np.max(np.abs(com)) <= 1e-6
                and quat.shape == (4,) and np.isfinite(quat).all() and abs(np.linalg.norm(quat)-1) < 1e-3)
            inertia = np.asarray(item.get('inertia_kg_m2'), float)
            tensor_ok = inertia.shape == (3, 3) and np.isfinite(inertia).all()
            if tensor_ok:
                eigen = np.linalg.eigvalsh(inertia)
                upper = spec['mass_kg'] * sum(d*d for d in spec['dimensions_m']) / 4
                tensor_ok = bool(np.allclose(inertia, inertia.T, rtol=0, atol=1e-10)
                    and (eigen > 0).all() and eigen[-1] <= upper
                    and eigen[-1] <= eigen[0] + eigen[1] + 1e-9)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            mass_ok, tensor_ok = False, False
        checks['convex_tensor_mass_and_com_match'] &= mass_ok
        checks['convex_tensor_inertia_valid'] &= bool(tensor_ok)
    return {'pass': bool(all(checks.values())), 'checks': checks, 'samples': len(samples),
        'object_name': object_name, 'scene_config_sha256': expected['scene_config_sha256'],
        'vertices_f32_sha256': hashlib.sha256(np.ascontiguousarray(first, dtype='<f4').tobytes()).hexdigest() if first is not None else None,
        'expected_proxy': spec['proxy'], 'expected_mass_kg': spec['mass_kg'],
        'vertex_count': len(vertices), 'source': 'live authored body-local collider, bound in every physics sample',
        'cooked_hull_equality_claimed': False}
