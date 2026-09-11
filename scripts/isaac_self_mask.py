"""Render-product self segmentation, not object detection or truth poses."""
import base64
import zlib

import numpy as np


def encode_robot_mask(instance_ids, info, robot_id, timestamp, *, contact_paths=()):
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
    robot_ids += [int(i) for i, path in labels.items() if path in contact_paths]
    mask = np.isin(ids, robot_ids).astype(np.uint8)
    return {'version': 1, 'robot_id': robot_id, 't': float(timestamp),
            'contact_paths': contact_paths,
            'shape': list(mask.shape), 'encoding': 'zlib-u8-base64',
            'data': base64.b64encode(zlib.compress(mask.tobytes(), 3)).decode()}
