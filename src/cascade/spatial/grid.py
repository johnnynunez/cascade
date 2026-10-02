"""Immutable planar occupancy snapshots and conservative bounded route planning.

This is a flat-ground geometric planner. It neither proves ground support nor
provides legged footholds, localization, SLAM or an actuator controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from ..robotics.contracts import identifier
from ..sensing.models import number, vector
from .frames import SpatialStamp, TransformSample, sha256


@dataclass(frozen=True)
class GridSnapshot:
    frame_id: str
    stamp: SpatialStamp
    resolution_m: float
    origin_xy_m: tuple
    width: int
    height: int
    cells: tuple
    capture_sources: tuple = ()
    transform_sources: tuple = ()

    def __post_init__(self):
        identifier(self.frame_id, "grid frame")
        if not isinstance(self.stamp, SpatialStamp):
            raise ValueError("typed grid stamp required")
        res = number(self.resolution_m, "resolution_m", minimum=.001)
        if res > 10:
            raise ValueError("resolution too large")
        object.__setattr__(self, "resolution_m", res)
        object.__setattr__(self, "origin_xy_m", vector(self.origin_xy_m, 2, "origin_xy_m"))
        if any(type(v) is not int or not 1 <= v <= 1024 for v in (self.width, self.height)) or self.width*self.height > 262144:
            raise ValueError("grid dimensions exceed bound")
        if not isinstance(self.cells, (tuple, list)) or len(self.cells) != self.width*self.height:
            raise ValueError("grid cell count mismatch")
        if any(type(v) is not int or v not in (-1, 0, 100) for v in self.cells):
            raise ValueError("cells must be unknown(-1), free(0), or occupied(100)")
        object.__setattr__(self, "cells", tuple(self.cells))
        for key, kind, limit in (("capture_sources", SpatialStamp, 128), ("transform_sources", TransformSample, 512)):
            values = getattr(self, key)
            if not isinstance(values, (tuple, list)) or len(values) > limit or any(not isinstance(v, kind) for v in values):
                raise ValueError("grid provenance exceeds bound or has invalid types")
            for value in values:
                s = value if isinstance(value, SpatialStamp) else value.stamp
                if s.context != self.stamp.context or s.measurement_kind != self.stamp.measurement_kind:
                    raise ValueError("grid source context or measurement kind mismatch")
                if s.time_s > self.stamp.time_s:
                    raise ValueError("grid source is newer than snapshot")
            object.__setattr__(self, key, tuple(values))

    @property
    def oldest_capture_time_s(self):
        return min((self.stamp.time_s, *(v.time_s for v in self.capture_sources)))

    def as_dict(self):
        return {"frame_id": self.frame_id, "stamp": self.stamp.as_dict(),
                "resolution_m": self.resolution_m, "origin_xy_m": self.origin_xy_m,
                "width": self.width, "height": self.height, "cells": self.cells,
                "capture_sources": [v.as_dict() for v in self.capture_sources],
                "transform_sources": [v.as_dict() for v in self.transform_sources],
                "oldest_capture_time_s": self.oldest_capture_time_s,
                "geometry_uncertainty": "not_estimated", "geometry": "planar_occupancy_only",
                "physical_admission": False}

    @property
    def sha256(self):
        return sha256(self.as_dict())

    def cell(self, xy):
        xy = vector(xy, 2, "point_xy_m")
        value = tuple(math.floor((v-o)/self.resolution_m) for v, o in zip(xy, self.origin_xy_m))
        if not self.contains(value):
            raise ValueError("point outside map")
        return value

    def contains(self, cell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def center(self, cell):
        return tuple(o+(v+.5)*self.resolution_m for o, v in zip(self.origin_xy_m, cell))

    def value(self, cell):
        return self.cells[cell[1]*self.width+cell[0]]


def ray_cells(start, end):
    """DDA through continuous grid coordinates; corner-only touches are omitted.

    Replacing true endpoints with cell centers would free unobserved cells.
    Only intervals of positive length inside a cell supply free-ray evidence.
    """
    start, end = vector(start, 2, "ray start"), vector(end, 2, "ray end")
    cells = [math.floor(v) for v in start]
    delta = [b-a for a, b in zip(start, end)]
    steps = [1 if v > 0 else -1 for v in delta]
    crosses, increments = [], []
    for pos, cell, change, step in zip(start, cells, delta, steps):
        crosses.append((cell+(1 if step > 0 else 0)-pos)/change if change else math.inf)
        increments.append(abs(1/change) if change else math.inf)
    progress = 0.
    budget = sum(abs(math.floor(b)-math.floor(a)) for a, b in zip(start, end))+3
    for _ in range(budget):
        boundary = min(*crosses, 1.)
        if boundary-progress > 1e-12:
            yield tuple(cells)
        if boundary >= 1.:
            return
        for axis in (0, 1):
            if crosses[axis] <= boundary+1e-12:
                cells[axis] += steps[axis]
                crosses[axis] += increments[axis]
        progress = boundary
    raise ValueError("ray traversal budget exhausted")


def integrate_scan(grid, frames, *, frame_id, stamp, bearings_rad, ranges_m, max_range_m,
                   max_transform_age_s=.2):
    """Integrate explicit planar range returns. None means observed clear range.

    Invalid/missing measurements must not be encoded as None: callers omit those
    rays. Endpoints outside the declared map are refused, never clipped to free.
    All occupied endpoints dominate free rays within this capture.
    """
    if not isinstance(stamp, SpatialStamp) or stamp.context != grid.stamp.context:
        raise ValueError("scan context mismatch")
    if stamp.time_s <= grid.stamp.time_s:
        raise ValueError("scan replay or out-of-order capture")
    if (stamp.calibration_id != grid.stamp.calibration_id or
            stamp.measurement_kind != grid.stamp.measurement_kind):
        raise ValueError("scan calibration or measurement kind mismatch")
    if not isinstance(bearings_rad, (list, tuple)) or not 1 <= len(bearings_rad) <= 8192:
        raise ValueError("scan must have 1..8192 rays")
    if not isinstance(ranges_m, (list, tuple)) or len(ranges_m) != len(bearings_rad):
        raise ValueError("scan dimensions mismatch")
    limit = number(max_range_m, "max_range_m", minimum=.001)
    if limit > 100:
        raise ValueError("scan range exceeds bound")
    pose = frames.lookup(grid.frame_id, frame_id, time_s=stamp.time_s, epoch=stamp.epoch,
                         clock_id=stamp.clock_id, max_age_s=max_transform_age_s)
    own_edge = next((v for v in pose.samples if v.child == frame_id), None)
    if own_edge and own_edge.stamp.calibration_id != stamp.calibration_id:
        raise ValueError("scan calibration mismatch")
    if any(v.stamp.measurement_kind != stamp.measurement_kind for v in pose.samples):
        raise ValueError("scan transform measurement kind mismatch")
    # A tilted ray plane is not a traversability map. Require horizontal axes.
    if abs(pose.matrix[2][0]) > 1e-6 or abs(pose.matrix[2][1]) > 1e-6:
        raise ValueError("planar scan requires a horizontal sensor plane")
    start_xy = pose.point((0., 0., 0.))[:2]
    grid.cell(start_xy)  # bounds check before processing rays
    def coordinate(point):
        return tuple((v-o)/grid.resolution_m for v, o in zip(point, grid.origin_xy_m))
    free, occupied = set(), set()
    for angle, distance in zip(bearings_rad, ranges_m):
        angle = number(angle, "bearing_rad")
        value = limit if distance is None else number(distance, "range_m", minimum=.001)
        if value > limit:
            raise ValueError("range exceeds declared maximum")
        end_xy = pose.point((value*math.cos(angle), value*math.sin(angle), 0.))[:2]
        end = grid.cell(end_xy)
        cells = list(ray_cells(coordinate(start_xy), coordinate(end_xy)))
        free.update(cells if distance is None else (cell for cell in cells if cell != end))
        if distance is not None:
            occupied.add(end)
    values = list(grid.cells)
    for x, y in free:
        values[y*grid.width+x] = 0
    for x, y in occupied:
        values[y*grid.width+x] = 100
    captures = (*grid.capture_sources, grid.stamp) if any(v != -1 for v in grid.cells) else ()
    captures = tuple(dict.fromkeys((*captures, stamp)))
    transforms = tuple(dict.fromkeys((*grid.transform_sources, *pose.samples)))
    return GridSnapshot(grid.frame_id, stamp, grid.resolution_m, grid.origin_xy_m,
                        grid.width, grid.height, values, captures, transforms)


def plan_route(grid, start_xy_m, goal_xy_m, *, footprint_radius_m, clearance_m,
               position_error_m, time_s, epoch, clock_id, expected_map_sha256,
               max_age_s=1., max_expansions=100000):
    """A* over cell centers, unknown blocked and a circumscribed circular body.

    The caller must include payload and all projected body geometry in the
    radius. An inscribed radius is unsafe. This does not validate that contract.
    """
    if not isinstance(grid, GridSnapshot) or expected_map_sha256 != grid.sha256:
        raise ValueError("map identity mismatch")
    if (epoch, clock_id) != grid.stamp.context[1:]:
        raise ValueError("map epoch or clock mismatch")
    now, age = number(time_s, "planning time", minimum=0), number(max_age_s, "max_age_s", minimum=0)
    if not 0 <= now-grid.stamp.time_s <= age or now-grid.oldest_capture_time_s > age:
        raise ValueError("map is stale or from the future")
    radius = sum(number(v, k, minimum=0) for k, v in
                 (("footprint_radius_m", footprint_radius_m), ("clearance_m", clearance_m),
                  ("position_error_m", position_error_m)))
    if radius <= 0 or radius > 10:
        raise ValueError("bounded positive footprint required")
    if type(max_expansions) is not int or not 1 <= max_expansions <= 262144:
        raise ValueError("invalid planner expansion budget")
    start, goal = grid.cell(start_xy_m), grid.cell(goal_xy_m)
    # Exact Euclidean distance to each blocked cell square, evaluated through
    # bounded offsets. A half-cell diagonal also covers anywhere inside the
    # traversed cell; this deliberately over-approximates swept occupancy.
    reach = radius + math.sqrt(.5)*grid.resolution_m
    n = min(max(grid.width, grid.height), math.ceil(reach/grid.resolution_m+.5))
    if (2*n+1)**2*len(grid.cells) > 20000000:
        raise ValueError("inflation work exceeds planner budget")
    offsets = [(dx, dy) for dx in range(-n, n+1) for dy in range(-n, n+1)
               if math.hypot(max(abs(dx)-.5, 0), max(abs(dy)-.5, 0))*grid.resolution_m <= reach]
    blocked = set()
    for y in range(grid.height):
        for x in range(grid.width):
            boundary = min(x+.5, y+.5, grid.width-x-.5, grid.height-y-.5)*grid.resolution_m
            if boundary <= reach or grid.value((x, y)) != 0:
                blocked.add((x, y))
            if grid.value((x, y)) != 0:
                blocked.update((x+dx, y+dy) for dx, dy in offsets if grid.contains((x+dx, y+dy)))
    if start in blocked or goal in blocked:
        raise ValueError("start or goal lacks known free footprint clearance")
    queue, costs, parents = [(0, 0, start)], {start: 0}, {}
    expanded = 0
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == goal:
            cells = [goal]
            while cells[-1] != start:
                cells.append(parents[cells[-1]])
            points = [tuple(start_xy_m), *(grid.center(c) for c in reversed(cells)), tuple(goal_xy_m)]
            points = [p for i, p in enumerate(points) if i == 0 or p != points[i-1]]
            return {"waypoints_xy_m": points, "frame_id": grid.frame_id,
                    "map_sha256": grid.sha256, "stamp": grid.stamp.as_dict(),
                    "length_m": sum(math.dist(a, b) for a, b in zip(points, points[1:])),
                    "expanded": expanded, "footprint_radius_m": footprint_radius_m,
                    "clearance_m": clearance_m, "position_error_m": position_error_m,
                    "unknown_policy": "blocked", "execution": "not_executed",
                    "physical_admission": False}
        expanded += 1
        if expanded > max_expansions:
            raise ValueError("route planning expansion budget exhausted")
        x, y = current
        for nxt in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            nc = cost+1
            if grid.contains(nxt) and nxt not in blocked and nc < costs.get(nxt, math.inf):
                costs[nxt], parents[nxt] = nc, current
                priority = nc+abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1])
                heapq.heappush(queue, (priority, nc, nxt))
    raise ValueError("no route through known free space")
