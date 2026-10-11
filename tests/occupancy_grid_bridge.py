"""A synthetic occupancy bridge for offline tests (no ZMQ, no GPU).

`GridBridge` answers the OccupancyClient protocol (`probe`, `integrate_depth`,
`integrate_masked_depth`, `clear`, `query`) with an exact distance grid over the
requested region, computed from axis-aligned obstacle boxes the test places
(`bridge.boxes`). `query` also returns the occupied voxel centres, like every
real backend. Boxes can be changed between refreshes to simulate an object that
appears in the camera view while the arm moves.

Not a test module (no `test_` prefix); tests import it from the tests dir.
"""

from __future__ import annotations

import threading

import numpy as np

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.types import Frame

#: a table plane as the camera maps it: one voxel layer at z = 0
TABLE = ((-1.0, -1.0, -0.01), (1.0, 1.0, 0.0))


def box_distance(points: np.ndarray, lo, hi) -> np.ndarray:
    """Exact Euclidean distance from each point to an axis-aligned box (0 inside)."""
    p = np.asarray(points, dtype=float)
    d = np.maximum(np.maximum(np.asarray(lo, float) - p, 0.0), p - np.asarray(hi, float))
    return np.linalg.norm(d, axis=-1)


def centred_box(center, size):
    c, h = np.asarray(center, float), np.asarray(size, float) / 2.0
    return (tuple(c - h), tuple(c + h))


class GridBridge:
    """Fake bridge: distance grid from `boxes`, optional `unknown` boxes."""

    def __init__(self, boxes=(), *, voxel=0.02, backend="nvblox", masked=True):
        self.boxes = list(boxes)
        self.unknown = []          # boxes whose voxels read as unobserved (inf)
        self.voxel = float(voxel)
        self.backend = backend
        self.masked = masked
        self.actions = []
        self.fail = None           # an exception message: every query raises it
        self.gate = threading.Event()
        self.gate.set()            # clear() it to block `query` until set again
        self.in_query = threading.Event()
        self.queries = 0
        self.last_depth = None
        self.last_active = None

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})

    def request(self, payload, timeout_ms=None):
        action = payload["action"]
        self.actions.append(action)
        if action == "probe":
            return {"ok": True, "backend": self.backend, "device": "cuda:0", "voxel": self.voxel,
                    "esdf": True, "masked_depth": self.masked}
        if action in ("integrate_depth", "integrate_masked_depth"):
            self.last_depth = np.asarray(payload["depth"])
            self.last_active = payload.get("active_mask")
            return {}
        if action == "clear":
            return {}
        if action == "query":
            self.in_query.set()
            assert self.gate.wait(10), "the test never released the bridge"
            if self.fail:
                raise OccupancyError(self.fail)
            self.queries += 1
            return self._grid(np.asarray(payload["region_min"], float),
                              np.asarray(payload["region_max"], float))
        raise OccupancyError(f"unknown action {action!r}")

    def _grid(self, lo, hi):
        v = self.voxel
        shape = np.maximum(np.round((hi - lo) / v).astype(int), 1)
        axes = [lo[i] + v * (np.arange(shape[i]) + 0.5) for i in range(3)]
        centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        grid = np.full(tuple(shape), np.inf, dtype=np.float32)
        for b_lo, b_hi in self.boxes:
            grid = np.minimum(grid, box_distance(centres, b_lo, b_hi).astype(np.float32))
        occupied = centres[grid <= v / 2.0 + 1e-9]
        for u_lo, u_hi in self.unknown:
            grid[box_distance(centres, u_lo, u_hi) <= 1e-9] = np.inf
        return {"grid": grid, "origin": np.asarray([a[0] for a in axes], np.float32),
                "voxel": v, "points": occupied.astype(np.float32)}


def depth_frame(t=None, shape=(12, 16)):
    """A small depth frame (2 m everywhere); `t` defaults to now (fresh)."""
    kw = {} if t is None else {"t": t}
    return Frame(rgb=np.zeros((*shape, 3), np.uint8), depth_m=np.full(shape, 2.0, np.float32),
                 K=np.array([[20.0, 0, shape[1] / 2], [0, 20.0, shape[0] / 2], [0, 0, 1]]), **kw)


REGION_MIN = np.array([-0.30, -0.45, -0.02])
REGION_MAX = np.array([0.66, 0.45, 0.60])


def make_map(bridge, *, required=True, **kw) -> OccupancyMap:
    occ = OccupancyMap(client=bridge, region_min=REGION_MIN, region_max=REGION_MAX,
                       required=required, **kw)
    occ.probe()
    return occ
