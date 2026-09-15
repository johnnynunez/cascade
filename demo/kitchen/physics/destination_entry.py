"""Lifted bilateral support with a full destination footprint.

This is an additional observed-entry subcheck, not an independent cycle verdict.
The caller must bind vertices/bounds to reviewed live scene geometry, supply only
pre-reset pick samples, and retain all existing source, CUDA, release, settle,
reset and camera checks. No endpoint-success verdict is changed by this module.

Two supported physics steps retain the existing >0.01 N / >=15 mm thresholds.
Intervening zero-force samples do not prove a slip: discrete solver contacts can
be intermittent. The gate makes no continuous or unsampled-retention claim and
cannot timestamp a release command absent a command-boundary sensor record.
"""
from __future__ import annotations

import numpy as np


def audit_supported_destination_entry(samples, *, object_name, vertices_body_m,
                                      inner_bounds_xy_m, destination_name="green square"):
    """Attest the whole rotated collider over the pad while still held aloft.

    Before-release here means positive support at the reported witness samples;
    no commanded release timestamp is inferred. A subsequent slip after valid
    entry can still pass this subcheck; ordinary final/support/release checks
    remain mandatory. This specifically rejects drops before destination entry.
    """
    if destination_name not in {'green square', 'open box'}:
        raise ValueError('Expected the marked square or measured open-box cavity')
    if object_name not in {'green_cube', 'pink_cube', 'tomato_can', 'lemon', 'orange'}:
        raise ValueError('Unsupported named physical target')
    vertices = np.asarray(vertices_body_m, dtype=float)
    bounds = np.asarray(inner_bounds_xy_m, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not 4 <= len(vertices) <= 10000
            or not np.isfinite(vertices).all() or np.abs(vertices).max() > 10
            or np.linalg.matrix_rank(vertices-vertices[0]) < 3):
        raise ValueError('Finite solid body-local collider vertices required')
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or not (bounds[1] > bounds[0]).all()):
        raise ValueError('Finite measured destination interior bounds required')
    physics = [sample.get('physics', sample) for sample in samples]
    if not physics:
        raise ValueError('Observed pick samples required')
    clocks = np.asarray([p['sim_time'] for p in physics], float)
    steps = [p['physics_step'] for p in physics]
    if (not np.isfinite(clocks).all() or (np.diff(clocks) < 0).any()
            or any(type(step) is not int or step < 0 for step in steps)
            or any(b < a for a, b in zip(steps, steps[1:]))):
        raise ValueError('Ordered finite physics clocks and integer steps required')
    initial = np.asarray(physics[0]['props'][object_name]['position_m'], float)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError('Finite baseline position required')
    witnesses = []
    supported = []
    inside_without_support = []
    seen_steps = set()
    robot = physics[0]['robot_id']
    for index, p in enumerate(physics):
        prop = p['props'][object_name]
        position = np.asarray(prop['position_m'], float)
        quat = np.asarray(prop['orientation_wxyz'], float)
        if (position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1) >= 1e-3):
            raise ValueError('Finite measured position and unit wxyz orientation required')
        contact = p['contacts']
        forces = np.asarray(contact['jaw_forces_n'], float)
        counts = contact['jaw_contact_counts']
        jaw_root = robot+'/link1/link2/link3/link4/link5/link6/gripper_end'
        if (p['robot_id'] != robot or p.get('channel') != 'physics_tensor'
                or p.get('engine') != 'physx' or prop.get('tensor_device') != 'cuda:0'
                or contact.get('channel') != 'physx_gpu_contact_tensor'
                or contact.get('device') != 'cuda:0'
                or contact.get('sensor_paths') != ['/World_Props/'+object_name]
                or contact.get('filter_paths') != [[jaw_root+'/gripper_left', jaw_root+'/gripper_right']]
                or contact.get('physics_step') != p['physics_step']
                or forces.shape != (2,3) or not np.isfinite(forces).all()
                or not isinstance(counts, list) or len(counts) != 2
                or any(type(count) is not int or count < 0 for count in counts)):
            raise ValueError('Bound current CUDA contact forces from both actual jaws required')
        w, x, y, z = quat/np.linalg.norm(quat)
        rotation = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                               [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                               [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        world = vertices@rotation.T + position
        lo, hi = world[:,:2].min(axis=0), world[:,:2].max(axis=0)
        margin = float(min((lo-bounds[0]).min(), (bounds[1]-hi).min()))
        magnitudes = np.linalg.norm(forces, axis=1)
        lifted = float(position[2]-initial[2]) >= .015
        held = bool((magnitudes>.01).all() and all(count>0 for count in counts))
        receipt = {'index': index, 'sequence': samples[index].get('sequence', index),
                   'physics_step': p['physics_step'], 'sim_time': p['sim_time'],
                   'position_m': position.tolist(), 'lift_m': float(position[2]-initial[2]),
                   'inner_clearance_m': margin, 'jaw_force_norm_n': magnitudes.tolist(),
                   'jaw_contact_counts': counts, 'footprint_bounds_xy_m': [lo.tolist(), hi.tolist()]}
        if held and lifted:
            supported.append(receipt)
            if margin >= 0 and p['physics_step'] not in seen_steps:
                witnesses.append(receipt)
                seen_steps.add(p['physics_step'])
        elif margin >= 0 and lifted:
            inside_without_support.append(receipt)
    return {'pass': len(witnesses) >= 2,
            'check': 'whole_footprint_enters_destination_while_lifted_with_bilateral_support',
            'object_name': object_name, 'destination_name': destination_name, 'samples': len(physics),
            'criteria': {'jaw_force_exclusive_min_n': .01, 'min_lift_m': .015,
                         'min_distinct_supported_inside_steps': 2, 'boundary_tolerance_m': 0.0},
            'inner_bounds_xy_m': bounds.tolist(), 'collider_vertices': len(vertices),
            'supported_lift_samples': len(supported), 'supported_inside_distinct_steps': len(witnesses),
            'best_supported_inner_clearance_m': max((r['inner_clearance_m'] for r in supported), default=None),
            'first_supported_inside': witnesses[0] if witnesses else None,
            'last_supported_inside': witnesses[-1] if witnesses else None,
            'supported_inside': witnesses,
            'inside_lifted_without_bilateral_support_samples': len(inside_without_support),
            'scope': 'Positive sampled grip at full destination entry; no continuous retention or commanded-release timing assertion'}
