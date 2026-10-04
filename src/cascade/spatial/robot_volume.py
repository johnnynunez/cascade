"""Declared whole-robot reach envelopes and capture-bound articulated geometry.

Bounds must come from the registered model, including payloads/attachments and
the full permitted articulation range. A snapshot cannot establish that range,
and structural declarations do not establish native or hardware admission.
"""
from dataclasses import asdict, dataclass
import itertools
import math

from ..robotics.contracts import identifier
from ..robotics.embodiment import EmbodimentDescriptor
from ..sensing.models import digest, number, vector
from .frames import SpatialStamp, TransformSample, sha256


def _name(value):
    return str(identifier(value))


def _digest(value, *, optional=False):
    value = digest(value)
    if value is None and not optional:
        raise ValueError('explicit geometry/calibration SHA256 required')
    return None if value is None else str(value)


def _bounds(value, low, high):
    for key in (low, high):
        object.__setattr__(value, key, vector(getattr(value, key), 3, key))
    if any(a > b for a, b in zip(getattr(value, low), getattr(value, high))):
        raise ValueError('reversed geometry bounds')


@dataclass(frozen=True)
class CollisionBox:
    collider_id: str
    link_id: str
    minimum_m: tuple
    maximum_m: tuple
    error_m: float

    def __post_init__(self):
        for key in ('collider_id', 'link_id'):
            object.__setattr__(self, key, _name(getattr(self, key)))
        _bounds(self, 'minimum_m', 'maximum_m')
        object.__setattr__(self, 'error_m', number(self.error_m, 'geometry error', minimum=0))
        if any(a == b for a, b in zip(self.minimum_m, self.maximum_m)):
            raise ValueError('collision volume must have positive extent')

    @property
    def radius_m(self):
        # All local corners, not half-extents about an assumed link center.
        return math.sqrt(sum(max(abs(a), abs(b))**2 for a, b in
                             zip(self.minimum_m, self.maximum_m))) + self.error_m


@dataclass(frozen=True)
class LinkReach:
    link_id: str
    origin_min_m: tuple
    origin_max_m: tuple
    max_position_error_m: float
    max_angular_error_rad: float

    def __post_init__(self):
        object.__setattr__(self, 'link_id', _name(self.link_id))
        _bounds(self, 'origin_min_m', 'origin_max_m')
        for key in ('max_position_error_m', 'max_angular_error_rad'):
            object.__setattr__(self, key, number(getattr(self, key), key, minimum=0))
        if self.max_angular_error_rad > math.pi:
            raise ValueError('angular error exceeds pi')


@dataclass(frozen=True)
class RobotVolume:
    robot_id: str
    model_identity_sha256: str | None
    geometry_source_sha256: str
    base_frame_id: str
    pose_source_id: str
    pose_calibration_sha256: str
    embodiment: EmbodimentDescriptor
    collider_inventory: tuple
    colliders: tuple
    reach: tuple
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError('unsupported robot volume schema')
        for key in ('robot_id', 'base_frame_id', 'pose_source_id'):
            object.__setattr__(self, key, _name(getattr(self, key)))
        for key in ('geometry_source_sha256', 'pose_calibration_sha256', 'model_identity_sha256'):
            object.__setattr__(self, key, _digest(getattr(self, key), optional=key=='model_identity_sha256'))
        if type(self.embodiment) is not EmbodimentDescriptor:
            raise ValueError('explicit embodiment required for whole-volume coverage')
        if (self.embodiment.robot_id != self.robot_id or self.embodiment.root_link != self.base_frame_id
                or self.embodiment.root_mode != 'floating'):
            raise ValueError('volume robot/base differs from embodiment')
        if not isinstance(self.collider_inventory, (list, tuple)):
            raise ValueError('collider inventory must be an explicit array')
        inventory = tuple(_name(x) for x in self.collider_inventory)
        if not 1 <= len(inventory) <= 4096 or len(set(inventory)) != len(inventory):
            raise ValueError('invalid complete collider inventory')
        object.__setattr__(self, 'collider_inventory', inventory)
        for key, kind in (('colliders', CollisionBox), ('reach', LinkReach)):
            records = tuple(getattr(self, key))
            if not 1 <= len(records) <= 4096 or any(type(v) is not kind for v in records):
                raise ValueError('bounded typed geometry records required')
            object.__setattr__(self, key, records)
        names = [c.collider_id for c in self.colliders]
        links = [r.link_id for r in self.reach]
        if (len(set(names)) != len(names) or set(names) != set(inventory)
                or len(set(links)) != len(links) or set(links) != set(self.embodiment.links)
                or {c.link_id for c in self.colliders} != set(links)):
            raise ValueError('geometry must cover every declared collider and link exactly')
        root = next(r for r in self.reach if r.link_id == self.base_frame_id)
        if (root.origin_min_m != (0.,)*3 or root.origin_max_m != (0.,)*3
                or root.max_position_error_m != 0 or root.max_angular_error_rad != 0):
            raise ValueError('base origin must be fixed at the base frame origin')
        if self.cylinder()[0] > 10 or max(abs(v) for v in self.cylinder()[1:]) > 10:
            raise ValueError('robot volume exceeds bounded navigation workspace')

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError('explicit complete robot volume configuration required')
        value = dict(raw)
        value['embodiment'] = EmbodimentDescriptor.from_dict(value['embodiment'])
        for key, kind in (('colliders', CollisionBox), ('reach', LinkReach)):
            value[key] = tuple(kind(**row) for row in value[key])
        return cls(**value)

    def as_dict(self):
        return {**{key: getattr(self,key) for key in ('version','robot_id','model_identity_sha256',
            'geometry_source_sha256','base_frame_id','pose_source_id','pose_calibration_sha256')},
            'embodiment':self.embodiment.as_dict(), 'collider_inventory':list(self.collider_inventory),
            'colliders':[asdict(c) for c in self.colliders], 'reach':[asdict(r) for r in self.reach]}

    @property
    def sha256(self):
        return sha256(self.as_dict())

    def cylinder(self):
        """Enclose ALL permitted translations and rotations of every collider."""
        reaches = {r.link_id:r for r in self.reach}
        bounds = []
        for c in self.colliders:
            r = reaches[c.link_id]
            radius = c.radius_m
            bounds.append((math.hypot(*(max(abs(a),abs(b)) for a,b in
                zip(r.origin_min_m[:2],r.origin_max_m[:2]))) + radius,
                r.origin_min_m[2]-radius, r.origin_max_m[2]+radius))
        return max(v[0] for v in bounds), min(v[1] for v in bounds), max(v[2] for v in bounds)

    def validate(self, sample, navigation, *, now, max_age_s):
        """Require current complete link poses without shrinking the reach sweep."""
        if type(sample) is not RobotVolumeSample:
            raise ValueError('missing complete articulated geometry capture')
        if (sample.geometry_sha256 != self.sha256 or sample.model_identity_sha256 != self.model_identity_sha256
                or navigation.robot_id != self.robot_id or navigation.model_identity_sha256 != self.model_identity_sha256
                or sample.sensor_epoch != navigation.sensor_epoch or sample.sequence != navigation.sequence
                or sample.stamp.context != navigation.pose.stamp.context
                or sample.stamp.time_s != navigation.pose.stamp.time_s
                or sample.stamp.source_id != self.pose_source_id
                or sample.stamp.calibration_id != self.pose_calibration_sha256):
            raise ValueError('articulated geometry model/epoch/capture/calibration mismatch')
        if (sample.received_monotonic_s > now or
                not 0 <= now-sample.received_monotonic_s+sample.producer_age_s <= max_age_s):
            raise ValueError('stale or future articulated geometry')
        allowed = {'synthetic'} if navigation.pose.stamp.measurement_kind == 'synthetic' else {'measured','estimated'}
        if sample.stamp.measurement_kind not in allowed:
            raise ValueError('geometry pose provenance cannot authorize navigation')
        poses = {p.child:p for p in sample.link_poses}
        if len(poses) != len(sample.link_poses) or set(poses) != set(self.embodiment.links)-{self.base_frame_id}:
            raise ValueError('geometry snapshot must include every non-base link exactly')
        for reach in self.reach:
            if reach.link_id == self.base_frame_id:
                continue
            pose = poses[reach.link_id]
            if pose.parent != self.base_frame_id or pose.static or pose.stamp != sample.stamp:
                raise ValueError('geometry requires same-capture base-relative link poses')
            if (pose.position_error_m is None or pose.angular_error_rad is None
                    or pose.position_error_m > reach.max_position_error_m
                    or pose.angular_error_rad > reach.max_angular_error_rad):
                raise ValueError('unknown or excessive articulated geometry uncertainty')
            if any(p-pose.position_error_m < a or p+pose.position_error_m > b for p,a,b in
                   zip(pose.translation_m,reach.origin_min_m,reach.origin_max_m)):
                raise ValueError('observed articulated pose leaves declared full reach')
        # A diagnostic instantaneous bound: uncertainty rotates each local
        # corner about its own link origin, including any long lever arm.
        points = []
        matrices = {name: pose.matrix for name, pose in poses.items()}
        for collider in self.colliders:
            pose = poses.get(collider.link_id)
            for corner in itertools.product(*zip(collider.minimum_m,collider.maximum_m)):
                if pose is None:
                    point, margin = corner, collider.error_m
                else:
                    point = (matrices[collider.link_id] @ (*corner,1.))[:3]
                    margin = collider.error_m + pose.position_error_m + 2*math.sqrt(sum(v*v for v in corner))*math.sin(pose.angular_error_rad/2)
                points.append((tuple(float(v)-margin for v in point),tuple(float(v)+margin for v in point)))
        return {'minimum_m':tuple(min(p[0][i] for p in points) for i in range(3)),
                'maximum_m':tuple(max(p[1][i] for p in points) for i in range(3)),
                'geometry_sample_sha256':sample.sha256}


@dataclass(frozen=True)
class RobotVolumeSample:
    geometry_sha256: str
    model_identity_sha256: str | None
    sensor_epoch: str
    sequence: int
    stamp: SpatialStamp
    received_monotonic_s: float
    producer_age_s: float
    link_poses: tuple

    def __post_init__(self):
        object.__setattr__(self,'geometry_sha256',_digest(self.geometry_sha256))
        object.__setattr__(self,'model_identity_sha256',_digest(self.model_identity_sha256,optional=True))
        object.__setattr__(self,'sensor_epoch',_name(self.sensor_epoch))
        if type(self.sequence) is not int or self.sequence < 0 or type(self.stamp) is not SpatialStamp:
            raise ValueError('typed geometry capture stamp/sequence required')
        for key in ('received_monotonic_s','producer_age_s'):
            object.__setattr__(self,key,number(getattr(self,key),key,minimum=0))
        poses = tuple(self.link_poses)
        if len(poses) > 4096 or any(type(p) is not TransformSample for p in poses):
            raise ValueError('bounded typed link transforms required')
        object.__setattr__(self,'link_poses',poses)

    @property
    def sha256(self):
        return sha256(asdict(self))
