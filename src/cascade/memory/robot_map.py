# SPDX-License-Identifier: Apache-2.0
"""Per-robot 2D world model built from the robot's own 8x8 ToF frames.

Each MicroDuck in an agentic demo owns one :class:`RobotMap`: a log-odds occupancy grid on the floor plane plus named
landmarks. Nothing is shared between robots unless an agent explicitly posts a message to the fleet board.

Pose source: the caller passes the sensor pose for every frame. In the simulated demos that pose comes from the
simulator's base/lens state (a stand-in for onboard SLAM; no visual odometry runs here), and the map metadata says so.

Classification of one ToF zone (VL53L8CX driver frame: ``distance_mm`` + ``status``):
* valid (status 5/9, >= 100 mm): the hit point is projected with the zone's world beam direction. Hit height above the
  floor > ``obstacle_min_height_m`` -> obstacle cell; otherwise floor (free). Cells along the beam before the hit are
  free;
* no target (status 255) on a beam that should have met the floor within range, or a valid return more than 5 cm
  BELOW the robot's floor -> a *drop* (edge/cliff) observation, placed where the beam would have met the robot's floor
  (the edge is at or before that point, so ``drop_distance_m`` is an upper bound); other no-target beams mark the
  first ``free_no_target_m`` along the beam free;
* anything else is unusable and ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

STATUS_VALID = (5, 9)
STATUS_NO_TARGET = 255
TOO_CLOSE_MM = 100
MAX_RANGE_M = 4.0


@dataclass
class Landmark:
    label: str
    x: float
    y: float
    sim_time_s: float
    note: str = ""


@dataclass
class ToFReading:
    """Summary of one frame in the robot's body frame, for agents and overlays."""

    free_ahead_m: float
    free_left_m: float
    free_right_m: float
    nearest_obstacle_m: float | None
    nearest_obstacle_bearing_deg: float | None
    drop_ahead: bool
    drop_distance_m: float | None
    valid_zones: int
    obstacle_zones: int
    grid_dm: list = field(default_factory=list)

    def text(self) -> str:
        def fmt(v):
            return ">2.0 m" if v >= 2.0 else f"{v:.2f} m"
        parts = [f"free ahead {fmt(self.free_ahead_m)}", f"left {fmt(self.free_left_m)}",
                 f"right {fmt(self.free_right_m)}"]
        if self.nearest_obstacle_m is not None:
            parts.append(f"nearest obstacle {self.nearest_obstacle_m:.2f} m at bearing "
                         f"{self.nearest_obstacle_bearing_deg:+.0f} deg (+ = left)")
        else:
            parts.append("no obstacle in view")
        if self.drop_ahead:
            parts.append(f"DROP-OFF ahead at ~{self.drop_distance_m:.2f} m: the floor ends there (edge/cliff)")
        parts.append(f"{self.valid_zones}/64 valid zones")
        return "; ".join(parts)


def classify_frame(distance_mm, status, sensor_pos_w, beam_dir_w, *, floor_z: float = 0.0,
                   obstacle_min_height_m: float = 0.03):
    """Per-zone arrays: kind (0 unusable, 1 floor, 2 obstacle, 3 no-target free, 4 drop), hit points (64, 3)."""
    mm = np.asarray(distance_mm, dtype=np.float64).reshape(64)
    st = np.asarray(status).reshape(64)
    d = np.asarray(beam_dir_w, dtype=np.float64).reshape(64, 3)
    p0 = np.asarray(sensor_pos_w, dtype=np.float64).reshape(3)
    valid = np.isin(st, STATUS_VALID) & (mm >= TOO_CLOSE_MM)
    rng = mm / 1000.0
    hits = p0[None] + d * rng[:, None]
    kind = np.zeros(64, np.int8)
    height = hits[:, 2] - floor_z
    kind[valid & (height <= obstacle_min_height_m)] = 1
    kind[valid & (height > obstacle_min_height_m)] = 2
    kind[valid & (height < -0.05)] = 4  # the floor is lower than the robot's floor: an edge / drop-off
    down = d[:, 2] < -1e-3
    t_floor = np.where(down, (floor_z - p0[2]) / np.where(down, d[:, 2], -1.0), np.inf)
    expect_floor = down & (t_floor > 0) & (t_floor < MAX_RANGE_M * 0.9)
    no_target = st == STATUS_NO_TARGET
    kind[no_target & ~expect_floor] = 3
    kind[no_target & expect_floor] = 4
    # A drop zone is placed where its beam WOULD have met the robot's floor: the edge lies at or before that point.
    floor_pts = p0[None] + d * np.where(np.isfinite(t_floor), t_floor, 0.0)[:, None]
    hits = np.where(((kind == 4) & expect_floor)[:, None], floor_pts, hits)
    return kind, hits, rng


def summarize_frame(distance_mm, status, sensor_pos_w, beam_dir_w, yaw_rad: float, *, floor_z: float = 0.0):
    """Agent-facing summary in the robot's heading frame (bearings + = left)."""
    kind, hits, rng = classify_frame(distance_mm, status, sensor_pos_w, beam_dir_w, floor_z=floor_z)
    p0 = np.asarray(sensor_pos_w, dtype=np.float64).reshape(3)
    rel = hits[:, :2] - p0[None, :2]
    c, s = math.cos(-yaw_rad), math.sin(-yaw_rad)
    fwd = c * rel[:, 0] - s * rel[:, 1]
    left = s * rel[:, 0] + c * rel[:, 1]
    planar = np.hypot(fwd, left)
    bearing = np.degrees(np.arctan2(left, fwd))
    obstacle = kind == 2

    def free_in(lo, hi):
        sel = (bearing >= lo) & (bearing < hi)
        obs = sel & obstacle
        if obs.any():
            return float(planar[obs].min())
        seen = sel & np.isin(kind, (1, 3))
        return float(min(2.0, planar[seen].max())) if seen.any() else 0.0

    nearest = nearest_b = None
    if obstacle.any():
        j = int(np.argmin(np.where(obstacle, planar, np.inf)))
        nearest, nearest_b = float(planar[j]), float(bearing[j])
    centre = (np.abs(bearing) < 15) & (kind == 4)
    drop = bool(centre.any())
    drop_d = float(planar[centre].min()) if drop else None
    grid = np.where(kind == 2, -np.round(rng * 10), np.where(kind == 1, np.round(rng * 10),
                    np.where(np.isin(kind, (3, 4)), 99, 0))).reshape(8, 8).astype(int).tolist()
    return ToFReading(free_ahead_m=free_in(-12, 12), free_left_m=free_in(12, 30), free_right_m=free_in(-30, -12),
                      nearest_obstacle_m=nearest, nearest_obstacle_bearing_deg=nearest_b, drop_ahead=drop,
                      drop_distance_m=drop_d,
                      valid_zones=int(np.isin(kind, (1, 2)).sum()), obstacle_zones=int(obstacle.sum()), grid_dm=grid)


class RobotMap:
    """Log-odds occupancy grid (floor plane) + landmarks for ONE robot."""

    def __init__(self, robot: str, bounds_xy: tuple[float, float, float, float], resolution_m: float = 0.04,
                 pose_source: str = "simulator base pose (stand-in for SLAM)"):
        self.robot = robot
        self.x0, self.y0, self.x1, self.y1 = (float(v) for v in bounds_xy)
        self.res = float(resolution_m)
        self.w = int(math.ceil((self.x1 - self.x0) / self.res))
        self.h = int(math.ceil((self.y1 - self.y0) / self.res))
        self.logodds = np.zeros((self.h, self.w), np.float32)
        self.drop = np.zeros((self.h, self.w), bool)
        self.visited: list[tuple[float, float, float]] = []
        self.landmarks: list[Landmark] = []
        self.frames = 0
        self.pose_source = pose_source

    def cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def _inside(self, i, j):
        return 0 <= i < self.w and 0 <= j < self.h

    def _ray_free(self, x0, y0, x1, y1, delta=-0.4):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / (self.res * 0.7)) + 1
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            i, j = self.cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if self._inside(i, j):
                self.logodds[j, i] = max(-4.0, self.logodds[j, i] + delta)

    def integrate(self, distance_mm, status, sensor_pos_w, beam_dir_w, *, floor_z=0.0, free_no_target_m=1.0):
        kind, hits, _ = classify_frame(distance_mm, status, sensor_pos_w, beam_dir_w, floor_z=floor_z)
        p0 = np.asarray(sensor_pos_w, dtype=np.float64).reshape(3)
        d = np.asarray(beam_dir_w, dtype=np.float64).reshape(64, 3)
        for z in range(64):
            k = int(kind[z])
            if k == 0:
                continue
            if k == 3:
                end = p0 + d[z] * free_no_target_m
                self._ray_free(p0[0], p0[1], end[0], end[1], delta=-0.2)
                continue
            hx, hy = hits[z, 0], hits[z, 1]
            self._ray_free(p0[0], p0[1], hx, hy)
            i, j = self.cell(hx, hy)
            if not self._inside(i, j):
                continue
            if k == 2:
                self.logodds[j, i] = min(4.0, self.logodds[j, i] + 1.2)
            elif k == 4:
                self.drop[j, i] = True
        self.frames += 1

    def add_pose(self, x, y, yaw):
        self.visited.append((float(x), float(y), float(yaw)))

    def add_landmark(self, label, x, y, t, note=""):
        self.landmarks = [lm for lm in self.landmarks if lm.label != label]
        self.landmarks.append(Landmark(str(label), float(x), float(y), float(t), str(note)))

    def occupied(self):
        return self.logodds > 0.6

    def free(self):
        return self.logodds < -0.6

    def plan(self, start_xy, goal_xy, *, inflate_m: float = 0.14, unknown_cost: float = 1.5) -> dict:
        """A* over THIS robot's map (8-connected). Obstacles and drop cells are inflated by ``inflate_m``; unknown cells
        are traversable at ``unknown_cost`` per step (optimistic, like a classical planner on a partial map). Returns
        line-of-sight-pruned waypoints and how much of the path crosses unknown space."""
        import heapq

        blocked = self.occupied() | self.drop
        r = int(math.ceil(inflate_m / self.res))
        if blocked.any() and r > 0:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            disk = (xx * xx + yy * yy) <= r * r
            grown = np.zeros_like(blocked)
            for dj, di in zip(*np.nonzero(disk)):
                dj, di = dj - r, di - r
                grown[max(0, dj):self.h + min(0, dj), max(0, di):self.w + min(0, di)] |= \
                    blocked[max(0, -dj):self.h - max(0, dj), max(0, -di):self.w - max(0, di)]
            blocked = grown
        # keep a 1-cell border blocked so paths stay inside the scenario bounds
        blocked[0, :] = blocked[-1, :] = True
        blocked[:, 0] = blocked[:, -1] = True
        free = self.free()
        si, sj = self.cell(*start_xy)
        gi, gj = self.cell(*goal_xy)
        if not (self._inside(si, sj) and self._inside(gi, gj)):
            return {"ok": False, "error": "start or goal outside the map bounds"}
        start_blocked = bool(blocked[sj, si])
        if start_blocked:
            # The robot stands where it stands: let it leave the inflated zone around it (raw obstacle cells stay
            # blocked).
            raw = self.occupied() | self.drop
            jj, ii = np.ogrid[:self.h, :self.w]
            near = (ii - si) ** 2 + (jj - sj) ** 2 <= r * r
            blocked[near & ~raw] = False
        blocked[sj, si] = False
        goal_adjusted = None
        if blocked[gj, gi]:
            # nearest unblocked cell within 0.3 m of the requested goal
            k = int(math.ceil(0.3 / self.res))
            best = None
            for dj in range(-k, k + 1):
                for di in range(-k, k + 1):
                    ni, nj = gi + di, gj + dj
                    if self._inside(ni, nj) and not blocked[nj, ni] and di * di + dj * dj <= k * k:
                        dd = di * di + dj * dj
                        if best is None or dd < best[0]:
                            best = (dd, ni, nj)
            if best is None:
                return {"ok": False, "error": f"goal and everything within 0.3 m of it lie within {inflate_m} m of an "
                                              f"obstacle or drop-off in your map"}
            gi, gj = best[1], best[2]
            goal_adjusted = [round(self.x0 + (gi + 0.5) * self.res, 2), round(self.y0 + (gj + 0.5) * self.res, 2)]
        step_cost = np.where(free, 1.0, unknown_cost).astype(np.float64)
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414),
                 (-1, -1, 1.414)]
        g = {(si, sj): 0.0}
        parent = {}
        heap = [(math.hypot(gi - si, gj - sj), 0.0, si, sj)]
        while heap:
            _, gc, i, j = heapq.heappop(heap)
            if (i, j) == (gi, gj):
                break
            if gc > g.get((i, j), math.inf):
                continue
            for di, dj, c in moves:
                ni, nj = i + di, j + dj
                if not self._inside(ni, nj) or blocked[nj, ni]:
                    continue
                ng = gc + c * step_cost[nj, ni]
                if ng < g.get((ni, nj), math.inf):
                    g[(ni, nj)] = ng
                    parent[(ni, nj)] = (i, j)
                    heapq.heappush(heap, (ng + math.hypot(gi - ni, gj - nj), ng, ni, nj))
        if (gi, gj) not in g:
            return {"ok": False, "error": "no path in your map (obstacles/drop-offs block every route)"}
        cells = [(gi, gj)]
        while cells[-1] != (si, sj):
            cells.append(parent[cells[-1]])
        cells.reverse()

        def clear(a, b):
            n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
            for t in np.linspace(0.0, 1.0, n):
                i, j = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
                if blocked[j, i]:
                    return False
            return True

        keep = [cells[0]]
        k = 0
        while k < len(cells) - 1:
            nxt = len(cells) - 1
            while nxt > k + 1 and not clear(cells[k], cells[nxt]):
                nxt -= 1
            keep.append(cells[nxt])
            k = nxt
        pts = [[round(self.x0 + (i + 0.5) * self.res, 2), round(self.y0 + (j + 0.5) * self.res, 2)] for i, j in keep[1:]]
        pts[-1] = goal_adjusted or [round(float(goal_xy[0]), 2), round(float(goal_xy[1]), 2)]
        unknown = sum(1 for i, j in cells if not free[j, i]) / max(1, len(cells))
        length = sum(math.dist(a, b) for a, b in zip([list(start_xy)] + pts[:-1], pts))
        return {"ok": True, "waypoints": pts, "length_m": round(length, 2), "goal_adjusted_to": goal_adjusted,
                "fraction_through_unknown": round(unknown, 2), "inflation_m": inflate_m,
                "start_inside_inflation": start_blocked,
                "source": "A* on this robot's own ToF map (unknown = optimistic)"}

    def explored_fraction(self) -> float:
        return float((self.occupied() | self.free()).mean())

    def ascii(self, cols: int = 32) -> str:
        """Coarse top view: '#' obstacle, '.' free, '!' drop-off, ' ' unknown, R robot, landmarks by first letter."""
        step = max(1, int(math.ceil(self.w / cols)))
        rows = []
        occ, fre = self.occupied(), self.free()
        robot = self.visited[-1] if self.visited else None
        rc = self.cell(robot[0], robot[1]) if robot else None
        marks = {}
        for lm in self.landmarks:
            i, j = self.cell(lm.x, lm.y)
            marks[(i // step, j // step)] = lm.label[:1].upper()
        for jj in range(self.h // step, -1, -1):  # +y up
            line = []
            for ii in range(self.w // step + 1):
                sl = (slice(jj * step, (jj + 1) * step), slice(ii * step, (ii + 1) * step))
                ch = " "
                if self.drop[sl].any():
                    ch = "!"
                elif occ[sl].any():
                    ch = "#"
                elif fre[sl].any():
                    ch = "."
                if (ii, jj) in marks:
                    ch = marks[(ii, jj)]
                if rc and (rc[0] // step, rc[1] // step) == (ii, jj):
                    ch = "R"
                line.append(ch)
            rows.append("".join(line).rstrip())
        legend = (f"top view, +x right, +y up, {step * self.res:.2f} m per char, x {self.x0:+.1f}..{self.x1:+.1f}, "
                  f"y {self.y0:+.1f}..{self.y1:+.1f}")
        return legend + "\n" + "\n".join(rows)

    def summary(self) -> dict:
        lm = [{"label": m.label, "x": round(m.x, 2), "y": round(m.y, 2), "note": m.note} for m in self.landmarks]
        return {"robot": self.robot, "frames_integrated": self.frames, "explored_fraction": round(self.explored_fraction(), 3),
                "landmarks": lm, "pose_source": self.pose_source}

    def image(self, scale: int = 4) -> np.ndarray:
        """RGB uint8 top view for videos: unknown grey, free white, obstacle dark, drop red, path blue."""
        img = np.full((self.h, self.w, 3), 70, np.uint8)
        img[self.free()] = (225, 228, 232)
        img[self.occupied()] = (30, 30, 34)
        img[self.drop] = (220, 60, 50)
        for x, y, _ in self.visited:
            i, j = self.cell(x, y)
            if self._inside(i, j):
                img[j, i] = (40, 110, 230)
        img = np.flipud(img)
        return np.repeat(np.repeat(img, scale, 0), scale, 1)

    def snapshot(self) -> dict:
        return {"logodds": self.logodds.copy(), "drop": self.drop.copy(), "visited": list(self.visited),
                "landmarks": [vars(m).copy() for m in self.landmarks]}
