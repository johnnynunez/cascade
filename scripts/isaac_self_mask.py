"""Render-product self segmentation, not object detection or truth poses."""
import base64
import zlib

import numpy as np


def bilateral_contact_paths(snapshots, min_force_n=0.05):
    """Only props with measured force and contact on BOTH actual jaws attach."""
    paths = []
    for name, snapshot in snapshots.items():
        forces = np.asarray(snapshot["jaw_forces_n"], dtype=float)
        counts = np.asarray(snapshot["jaw_contact_counts"], dtype=int)
        if (forces.shape != (2, 3) or counts.shape != (2,)
                or not np.isfinite(forces).all()):
            raise ValueError("invalid jaw contact evidence")
        if np.all(counts > 0) and np.all(np.linalg.norm(forces, axis=1) > min_force_n):
            paths.append("/World_Props/" + name)
    return sorted(paths)


def encode_robot_mask(instance_ids, info, robot_id, timestamp, *, contact_paths=(), payload_tracking=False,
                      scene_prop_paths=()):
    ids = np.asarray(instance_ids)
    if ids.ndim == 3 and ids.shape[-1] == 1:
        ids = ids[..., 0]
    if ids.ndim != 2 or ids.dtype.kind not in 'iu':
        raise ValueError('invalid instance ID image')
    labels = info.get('idToLabels') if isinstance(info, dict) else None
    if not isinstance(labels, dict) or not labels:
        raise ValueError('instance labels unavailable')
    robot_ids = [int(i) for i, path in labels.items()
                 if isinstance(path, str) and (path == robot_id or path.startswith(robot_id + '/'))]
    if not robot_ids:
        raise ValueError('robot absent from instance labels')
    contact_paths = list(contact_paths)
    if any(not isinstance(p, str) or not p.startswith('/World_Props/') for p in contact_paths):
        raise ValueError('contact paths must name explicit scene props')
    payload_ids = [int(i) for i, path in labels.items()
                   if isinstance(path, str) and any(path == p or path.startswith(p + '/') for p in contact_paths)]
    robot_ids += payload_ids
    mask = np.isin(ids, robot_ids).astype(np.uint8)
    result = {'version': 1, 'robot_id': robot_id, 't': float(timestamp),
            'contact_paths': contact_paths,
            'shape': list(mask.shape), 'encoding': 'zlib-u8-base64',
            'data': base64.b64encode(zlib.compress(mask.tobytes(), 3)).decode()}
    if payload_tracking:
        payload = np.isin(ids, payload_ids).astype(np.uint8)
        result['payload_data'] = base64.b64encode(zlib.compress(payload.tobytes(), 3)).decode()
        result['prop_data'] = {}
        for path in scene_prop_paths:
            prop_ids = [int(i) for i, label in labels.items() if isinstance(label, str)
                        and (label == path or label.startswith(path + '/'))]
            prop_mask = np.isin(ids, prop_ids).astype(np.uint8)
            result['prop_data'][path] = base64.b64encode(zlib.compress(prop_mask.tobytes(), 3)).decode()
    return result
