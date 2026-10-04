"""Complete rigid box geometry witnesses for offline containment analysis.

These records describe the observed simulation model. They do not establish
physical calibration, support/rest, task success, or ordinary episode closure.
"""
from dataclasses import asdict, dataclass
import itertools
import math

import numpy as np

from .cavity import BoxCavity, BoxWall
from .trials import digest_json, require_digest
from ..sensing.models import rigid_transform, vector


def _pose(rotation, position):
    rotation = np.asarray(vector(rotation, 9, 'complete row-major rotation'))
    transform = np.eye(4)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = vector(position, 3, 'complete box geometry position')
    return np.asarray(rigid_transform(tuple(transform.flat), 'box geometry pose')).reshape(4, 4)


@dataclass(frozen=True)
class BoxGeometryInventory:
    model_identity_sha256: str
    placement_policy_sha256: str
    object_body_id: int
    support_body_id: int
    object_boxes: tuple
    support_boxes: tuple
    version: int = 1

    def __post_init__(self):
        require_digest(self.model_identity_sha256)
        require_digest(self.placement_policy_sha256)
        if type(self.version) is not int or self.version != 1:
            raise ValueError('unsupported box geometry inventory')
        if (any(type(i) is not int or i < 1 for i in (self.object_body_id, self.support_body_id))
                or self.object_body_id == self.support_body_id):
            raise ValueError('distinct native object/support bodies required')
        for name in ('object_boxes', 'support_boxes'):
            boxes = tuple(getattr(self, name))
            if not 1 <= len(boxes) <= 4096 or any(type(box) is not BoxWall for box in boxes):
                raise ValueError('bounded exact box inventory required')
            object.__setattr__(self, name, boxes)
        ids = [box.geometry_id for box in self.object_boxes+self.support_boxes]
        if len(ids) != len(set(ids)):
            raise ValueError('geometry inventory contains duplicate IDs')

    @property
    def sha256(self):
        return digest_json(asdict(self))

    def bind(self, policy):
        if (self.model_identity_sha256 != policy.model_identity_sha256
                or self.placement_policy_sha256 != policy.sha256
                or self.object_body_id != policy.object_body_id
                or self.support_body_id != policy.support_body_id
                or tuple(b.geometry_id for b in self.object_boxes) != policy.object_geoms
                or tuple(b.geometry_id for b in self.support_boxes) != policy.support_geoms):
            raise ValueError('complete collision inventory does not match placement policy')

    def cavity(self, *, interior_point_m, top_z_m, calibration_recipe='complete_faces'):
        return BoxCavity(self.model_identity_sha256, self.sha256, self.support_body_id,
                         self.support_boxes, interior_point_m, top_z_m,
                         calibration_recipe=calibration_recipe)

    def object_hulls(self, row, policy):
        """Reconstruct all object boxes in the same solve's support frame.

        The sealed row binds every raw matrix and center. The compiled local
        geometry is checked against independently retained body/geometry poses.
        No clock, support force or physical uncertainty is inferred here.
        """
        self.bind(policy)
        if (row['model_identity_sha256'] != self.model_identity_sha256
                or row['policy_sha256'] != self.placement_policy_sha256
                or row['epoch'] != policy.epoch
                or type(row['native_ngeom']) is not int or row['native_ngeom'] != policy.native_ngeom
                or type(row['solver_step']) is not int or row['solver_step'] < 1
                or row['phase'] != 'euler_constraint_before_integration'
                or digest_json({k:v for k,v in row.items() if k != 'snapshot_sha256'}) != row['snapshot_sha256']):
            raise ValueError('box geometry snapshot identity, phase or seal changed')
        for key in ('constraint_time_s', 'advanced_time_s'):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('box geometry clock is invalid')
        if not math.isclose(row['advanced_time_s']-row['constraint_time_s'], policy.solver_dt_s, rel_tol=0, abs_tol=1e-9):
            raise ValueError('box geometry phase clock is inconsistent')
        capture = row['box_geometry']
        if capture['inventory_sha256'] != self.sha256:
            raise ValueError('box geometry inventory changed')
        bodies = {}
        if set(capture['body_rotations_world']) != {'object', 'support'}:
            raise ValueError('box body rotation coverage is incomplete')
        for name in ('object', 'support'):
            body = row['bodies'][name]
            if type(body['body_id']) is not int or body['body_id'] != getattr(self, name+'_body_id'):
                raise ValueError('box body identity changed')
            bodies[name] = _pose(capture['body_rotations_world'][name], body['position_m'])
        centers, rotations = {}, {}
        for entry in row['geometries']:
            gid = entry['id']
            if type(gid) is not int or gid in centers:
                raise ValueError('duplicate or invalid geometry center')
            centers[gid] = entry['position_m']
        for entry in capture['rotations_world']:
            gid = entry['id']
            if type(gid) is not int or gid in rotations:
                raise ValueError('duplicate or invalid box rotation')
            rotations[gid] = entry['rotation']
        boxes = self.object_boxes+self.support_boxes
        if (set(rotations) != {box.geometry_id for box in boxes}
                or set(centers) != set(policy.object_geoms+policy.support_geoms+policy.robot_geoms)):
            raise ValueError('complete geometry center and box rotation coverage required')
        support_from_world = np.linalg.inv(bodies['support'])
        hulls = {}
        for name in ('object', 'support'):
            for box in getattr(self, name+'_boxes'):
                observed = _pose(rotations[box.geometry_id], centers[box.geometry_id])
                expected = bodies[name] @ np.asarray(box.body_from_geometry).reshape(4, 4)
                # These are consistency gates, not an uncertainty estimate.
                if (np.max(abs(observed[:3, 3]-expected[:3, 3])) > 1e-9
                        or np.max(abs(observed[:3, :3]-expected[:3, :3])) > 1e-7):
                    raise ValueError('observed geometry does not match compiled local box')
                if name == 'object':
                    corners = np.asarray(list(itertools.product(*[(-s,s) for s in box.half_size_m])))
                    transform = support_from_world @ observed
                    hulls[box.geometry_id] = corners @ transform[:3, :3].T+transform[:3, 3]
        return hulls

    def verify_snapshot(self, row, policy, cavity, *, support_frame_error_m):
        """Geometry only; the caller must separately verify every rest solve."""
        try:
            if (cavity.model_identity_sha256 != self.model_identity_sha256
                    or cavity.geometry_source_sha256 != self.sha256
                    or cavity.support_body_id != self.support_body_id
                    or cavity.walls != self.support_boxes):
                raise ValueError('cavity calibration is not bound to this complete support geometry')
            hulls = self.object_hulls(row, policy)
            return cavity.contains_hulls(hulls, expected_geometry_ids=policy.object_geoms,
                                         position_error_m=support_frame_error_m)
        except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
            return {'status':'unverified', 'reason':str(error), 'physical_admission':False}
