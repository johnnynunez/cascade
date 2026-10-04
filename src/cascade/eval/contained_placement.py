"""Join release/support/rest and complete box containment on one solve window.

The geometry witness uses the support-frame AABB of each component over the
entire window. These conservative envelopes contain all observed corners, so
fitting them inside the convex cavity proves inclusion at every retained solve.
An envelope that does not fit is inconclusive, not proof that an object escaped.
No quiet suffix is selected and no physical uncertainty is estimated here.
"""
from __future__ import annotations

import itertools
import io
import json
import math

import numpy as np

from .box_geometry import BoxGeometryInventory
from .cavity import BoxCavity
from .placement import PlacementPolicy, verify_placement_window
from .trials import digest_json


def _private_window(rows):
    """Both predicates consume one private JSON snapshot (<=256 MiB).

    Caller mutation during serialization still has to produce individually
    valid seals and a complete window. After this boundary the predicates
    cannot read different versions of caller-owned dictionaries/array views.
    """
    if not isinstance(rows, (list, tuple)) or not 2 <= len(rows) <= 16000:
        raise ValueError('explicit bounded solve window required (2..16000 rows)')
    with io.StringIO() as stream:
        length = 0
        for chunk in json.JSONEncoder(allow_nan=False, ensure_ascii=True,
                                      separators=(',', ':')).iterencode(rows):
            length += len(chunk)
            if length > 256*1024*1024:
                raise ValueError('placement window exceeds 256 MiB')
            stream.write(chunk)
        copied = json.loads(stream.getvalue())
    if not 2 <= len(copied) <= 16000:
        raise ValueError('serialized solve window exceeds row bound')
    return copied


def verify_contained_placement_window(rows, policy, inventory, cavity, *,
                                      first_solver_step, last_solver_step,
                                      support_frame_error_m):
    """Read an explicit complete rest window; never authorize an actuator/task.

    ``support_frame_error_m`` is a caller-supplied bound for every reconstructed
    corner relative to this support, including pose/shape/calibration errors.
    None means unknown. Its independent calibration remains the caller's
    responsibility; the bound is retained verbatim, not inferred from residuals.
    The native reader is a trusted measurement boundary, not a physics oracle.
    """
    result = {
        'status': 'unverified', 'physical_admission': False,
        'scope': 'released_supported_contained_rest_window',
        'benchmark_success': None, 'task_success': 'unverified',
        'checks': {'released_supported_rest': 'unverified',
                   'whole_object_containment': 'unverified'},
        'geometry_recipe': 'support_frame_component_window_aabbs_v1',
    }
    try:
        if (type(policy) is not PlacementPolicy or type(inventory) is not BoxGeometryInventory
                or type(cavity) is not BoxCavity):
            raise ValueError('exact registered placement, box inventory and cavity required')
        inventory.bind(policy)
        if (cavity.model_identity_sha256 != inventory.model_identity_sha256
                or cavity.geometry_source_sha256 != inventory.sha256
                or cavity.support_body_id != inventory.support_body_id
                or cavity.walls != inventory.support_boxes):
            raise ValueError('cavity does not bind the complete compiled support geometry')
        if (type(first_solver_step) is not int or type(last_solver_step) is not int
                or first_solver_step < 1 or last_solver_step <= first_solver_step):
            raise ValueError('positive advancing integer solver window required')
        if support_frame_error_m is not None and (
                type(support_frame_error_m) not in (int, float)
                or not math.isfinite(support_frame_error_m) or support_frame_error_m < 0):
            raise ValueError('support-frame error must be unknown or a finite nonnegative bound')
        rows = _private_window(rows)
        result.update(model_identity_sha256=policy.model_identity_sha256,
                      epoch=policy.epoch, placement_policy_sha256=policy.sha256,
                      inventory_sha256=inventory.sha256, cavity_sha256=cavity.sha256,
                      first_solver_step=first_solver_step, last_solver_step=last_solver_step,
                      support_frame_error_m=support_frame_error_m,
                      uncertainty_source='unknown' if support_frame_error_m is None
                      else 'caller_declared_bound_not_estimated')
        support = verify_placement_window(rows, policy, first_solver_step=first_solver_step,
                                          last_solver_step=last_solver_step)
        result['support'] = support
        result['checks']['released_supported_rest'] = support['status']
        if support['status'] == 'unverified':
            result['reason'] = 'release/support/rest coverage is unverified'
            return result

        # The support verifier has checked exact contiguous steps, clocks,
        # source/epoch/policy and every sealed row before this geometry pass.
        seals = tuple(row['snapshot_sha256'] for row in rows)
        result['window_sha256'] = digest_json({
            'first_solver_step': first_solver_step, 'last_solver_step': last_solver_step,
            'snapshot_sha256': seals, 'placement_policy_sha256': policy.sha256,
            'inventory_sha256': inventory.sha256, 'cavity_sha256': cavity.sha256})
        lower, upper = {}, {}
        for row in rows:
            for gid, vertices in inventory.object_hulls(row, policy).items():
                if not np.isfinite(vertices).all():
                    raise ValueError('support-frame corner reconstruction is not finite')
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                lower[gid] = np.minimum(lower[gid], lo) if gid in lower else lo
                upper[gid] = np.maximum(upper[gid], hi) if gid in upper else hi
        envelopes = {gid: np.asarray(list(itertools.product(*zip(lower[gid], upper[gid]))))
                     for gid in policy.object_geoms}
        geometry = cavity.contains_hulls(envelopes, expected_geometry_ids=policy.object_geoms,
                                         position_error_m=support_frame_error_m)
        # The two internal read-only predicates must also leave the private
        # snapshot intact. External caller mutation cannot reach these rows.
        if len(rows) != len(seals) or any(
                row['snapshot_sha256'] != seal or digest_json({k: v for k, v in row.items()
                if k != 'snapshot_sha256'}) != seal for row, seal in zip(rows, seals)):
            raise ValueError('placement rows changed during window verification')
        result['geometry'] = geometry
        result['checks']['whole_object_containment'] = geometry['status']
        result['component_envelopes_m'] = {
            str(gid): {'minimum': lower[gid].tolist(), 'maximum': upper[gid].tolist()}
            for gid in policy.object_geoms}
        result['geometry_samples'] = len(rows)
        result['geometry_components_per_sample'] = len(policy.object_geoms)
        if geometry['status'] != 'confirmed':
            result['reason'] = 'complete window envelopes do not certify containment'
        result['status'] = ('refuted' if support['status'] == 'refuted' else geometry['status'])
        return result
    except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
        # A failed join must not retain a partial positive from an earlier pass.
        result['status'] = 'unverified'
        result['reason'] = str(error)
        result['checks'] = {key: 'unverified' for key in result['checks']}
        return result
