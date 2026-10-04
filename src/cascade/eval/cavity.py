"""Conservative collision-cavity calibration from five complete box walls.

This is a geometry primitive, not task admission. Callers must bind the complete
compiled collider inventory, synchronized transforms and object hull vertices
to an independently closed episode. Visual bounds and object centers do not
establish containment.
"""
from dataclasses import asdict, dataclass
import itertools
import math

import numpy as np

from .trials import digest_json, require_digest
from ..sensing.models import rigid_transform, vector


@dataclass(frozen=True)
class BoxWall:
    geometry_id: int
    body_from_geometry: tuple
    half_size_m: tuple

    def __post_init__(self):
        if type(self.geometry_id) is not int or self.geometry_id < 0:
            raise ValueError('explicit native collision geometry ID required')
        object.__setattr__(self, 'body_from_geometry', rigid_transform(self.body_from_geometry, 'box wall pose'))
        object.__setattr__(self, 'half_size_m', vector(self.half_size_m, 3, 'wall half size'))
        if min(self.half_size_m) <= 0 or max(self.half_size_m) > 10:
            raise ValueError('wall dimensions must be positive and bounded')


@dataclass(frozen=True)
class BoxCavity:
    model_identity_sha256: str
    geometry_source_sha256: str
    support_body_id: int
    walls: tuple
    interior_point_m: tuple
    top_z_m: float
    calibration_recipe: str = 'complete_faces'
    version: int = 1

    def __post_init__(self):
        require_digest(self.model_identity_sha256)
        require_digest(self.geometry_source_sha256)
        if type(self.version) is not int or self.version != 1:
            raise ValueError('unsupported cavity version')
        if self.calibration_recipe not in ('complete_faces', 'finite_face_intersection'):
            raise ValueError('unsupported cavity calibration recipe')
        if type(self.support_body_id) is not int or self.support_body_id < 1:
            raise ValueError('explicit support body required')
        walls = tuple(self.walls)
        if len(walls) != 5 or any(type(w) is not BoxWall for w in walls) or len({w.geometry_id for w in walls}) != 5:
            raise ValueError('five unique complete box collision walls required')
        object.__setattr__(self, 'walls', walls)
        object.__setattr__(self, 'interior_point_m', vector(self.interior_point_m, 3, 'interior point'))
        if type(self.top_z_m) not in (int, float) or not math.isfinite(self.top_z_m) or abs(self.top_z_m) > 10:
            raise ValueError('finite bounded cavity top required')
        self.planes_and_vertices()

    @property
    def sha256(self):
        return digest_json(asdict(self))

    def planes_and_vertices(self):
        """Prove a bounded convex interior with finite physical wall coverage.

        Outward unit planes satisfy n·x <= d. Their major axes must cover
        +/-x, +/-y, +/-z with cos(angle)>.95. Any nonzero recession direction
        has a maximal component, whose matching normal has positive dot:
        .95-sqrt(2)*sqrt(1-.95²)>0. Hence the six-plane volume is bounded.
        The complete_faces recipe requires every non-cap face to lie on its
        actual finite box face. The finite_face_intersection recipe instead
        intersects this interior with all five finite faces' tangential
        projections. It establishes a smaller conservative region, never an
        extension of a wall or a claim that every cavity point was recovered.
        """
        point = np.asarray(self.interior_point_m)
        planes, faces, directions = [], [], set()
        for wall in self.walls:
            transform = np.asarray(wall.body_from_geometry).reshape(4, 4)
            rotation, center = transform[:3, :3], transform[:3, 3]
            # Use inverse rather than transpose: the shared rigid-transform
            # contract permits tiny encoding residuals, which cannot silently
            # become a wall enlargement in this calibration.
            inverse = np.linalg.inv(rotation)
            local = inverse @ (point-center)
            outside = abs(local)-wall.half_size_m
            axis = int(np.argmax(outside))
            if outside[axis] <= 0 or wall.half_size_m[axis] != min(wall.half_size_m):
                raise ValueError('interior point must face the thin side of each wall')
            sign = 1. if local[axis] > 0 else -1.
            normal = -sign*inverse[axis]
            normal /= np.linalg.norm(normal)
            face_point = center+sign*wall.half_size_m[axis]*rotation[:, axis]
            distance = float(normal @ face_point)
            cardinal = int(np.argmax(abs(normal)))
            if abs(normal[cardinal]) <= .95:
                raise ValueError('wall orientation is outside the bounded cavity recipe')
            direction = (cardinal, 1 if normal[cardinal] > 0 else -1)
            if direction in directions:
                raise ValueError('duplicate cavity wall direction')
            directions.add(direction)
            planes.append((*map(float, normal), distance))
            faces.append((wall, axis, sign, inverse, center))
        if (2, 1) in directions:
            raise ValueError('cavity must have an open top, not a sixth physical lid')
        directions.add((2, 1))
        if directions != {(axis, sign) for axis in range(3) for sign in (-1, 1)}:
            raise ValueError('incomplete cavity wall directions')
        planes.append((0., 0., 1., float(self.top_z_m)))
        clipped = self.calibration_recipe == 'finite_face_intersection'
        if clipped:
            for wall, axis, _, inverse, center in faces:
                for tangent in (i for i in range(3) if i != axis):
                    for sign in (-1., 1.):
                        normal = sign*inverse[tangent]
                        norm = float(np.linalg.norm(normal))
                        distance = (wall.half_size_m[tangent]+float(normal @ center))/norm
                        # Inset the projection before finding vertices, so
                        # roundoff cannot extend a finite physical wall.
                        planes.append((*map(float, normal/norm), distance-1e-9))
        planes = np.asarray(planes)
        normals, distances = planes[:, :3], planes[:, 3]
        if not np.all(normals @ point < distances-1e-9):
            raise ValueError('declared cavity has no strict interior at its registered point')
        vertices = []
        for indices in itertools.combinations(range(len(planes)), 3):
            selected = normals[list(indices)]
            if abs(float(np.linalg.det(selected))) < 1e-10:
                continue
            vertex = np.linalg.solve(selected, distances[list(indices)])
            if np.isfinite(vertex).all() and np.all(normals @ vertex <= distances+1e-10):
                if not any(np.max(abs(vertex-other)) < 1e-10 for other in vertices):
                    vertices.append(vertex)
        if len(vertices) < 4 or np.linalg.matrix_rank(np.asarray(vertices)[1:]-vertices[0]) != 3:
            raise ValueError('degenerate cavity volume')
        vertices = np.asarray(vertices)
        for index, (wall, axis, sign, inverse, center) in enumerate(faces):
            face = vertices if clipped else vertices[abs(vertices @ normals[index]-distances[index]) < 1e-9]
            if len(face) < 3:
                raise ValueError('a registered wall does not bound a cavity face')
            local = (face-center) @ inverse.T
            # Numerical slack only identifies equality on the wall plane;
            # tangential coverage is strict and never extends a finite wall.
            tangent = [i for i in range(3) if i != axis]
            if np.any(abs(local[:, tangent]) > np.asarray(wall.half_size_m)[tangent]):
                raise ValueError('finite wall does not cover the complete cavity face')
            if not clipped and np.max(abs(local[:, axis]-sign*wall.half_size_m[axis])) > 1e-9:
                raise ValueError('cavity face is not on its registered wall')
        return tuple(map(tuple, planes.tolist())), tuple(map(tuple, vertices.tolist()))

    def contains_hulls(self, hulls, *, expected_geometry_ids, position_error_m):
        """Check complete bounded hulls in the support body's local frame.

        This method does not localize an object. Hull capture, uncertainty and
        full inventory must be independently registered by the caller. Missing
        or oversized bounds are unverified, not proof of physical protrusion.
        """
        result = {'status': 'unverified', 'cavity_sha256': self.sha256, 'physical_admission': False}
        try:
            ids = tuple(expected_geometry_ids)
            if (not 1 <= len(ids) <= 4096 or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids)
                    or set(ids) & {wall.geometry_id for wall in self.walls}
                    or not isinstance(hulls, dict) or set(hulls) != set(ids)):
                raise ValueError('complete registered object collision hull inventory required')
            if type(position_error_m) not in (int, float) or not math.isfinite(position_error_m) or position_error_m < 0:
                raise ValueError('known finite position error required')
            planes, _ = self.planes_and_vertices()
            planes = np.asarray(planes)
            margins, count = [], 0
            for points in hulls.values():
                points = np.asarray(points, dtype=float)
                if (points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= 4096
                        or not np.isfinite(points).all() or np.max(abs(points)) > 10):
                    raise ValueError('bounded complete collider hull vertices required')
                count += len(points)
                if count > 65536 or np.linalg.matrix_rank(points[1:]-points[0]) != 3:
                    raise ValueError('bounded nondegenerate collider hulls required')
                # Error includes pose, geometry and frame uncertainty supplied
                # by the reader. A further 1nm numeric reserve is conservative.
                margins.extend((planes[:, 3]-points @ planes[:, :3].T-position_error_m-1e-9).flat)
            minimum = float(min(margins))
            result['minimum_margin_m'] = minimum
            if minimum < 0:
                raise ValueError('complete collision hull bounds do not fit inside the calibrated cavity')
            result.update(status='confirmed', reason='complete registered collision hull bounds fit in the cavity')
        except (TypeError, ValueError, OverflowError) as exc:
            result['reason'] = str(exc)
        return result
