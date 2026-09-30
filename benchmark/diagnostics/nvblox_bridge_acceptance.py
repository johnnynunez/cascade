"""Real bridge acceptance against a known depth scene and obstacle removal."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--port", type=int, default=25557)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.repo / "src"))
from cascade.perception.occupancy import OccupancyClient, OccupancyMap
from cascade.types import Frame

W, H, fx = 320, 240, 300.0
us, vs = np.meshgrid(np.arange(W), np.arange(H))
K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
T = np.array([[0, -1, 0, 0.3], [-1, 0, 0, 0], [0, 0, -1, 0.6], [0, 0, 0, 1.0]])
d = np.full((H, W), 0.6, np.float32)
x = (us - W / 2) / fx * 0.55
y = (vs - H / 2) / fx * 0.55
d[(np.abs(x) < 0.025) & (np.abs(y) < 0.025)] = 0.55
frame = Frame(rgb=np.zeros((H, W, 3), np.uint8), depth_m=d, K=K)
c = OccupancyClient(port=a.port, timeout_ms=10000)
m = OccupancyMap(
    c,
    region_min=np.array([0, -0.3, -0.02]),
    region_max=np.array([0.6, 0.3, 0.3]),
    depth_stride=1,
)
probe = m.probe(timeout_ms=10000)
assert probe and probe["backend"] == "nvblox" and probe["device"].startswith("cuda:"), (
    probe
)
lat = []
for _ in range(6):
    t = time.perf_counter()
    m.refresh(frame, T)
    lat.append((time.perf_counter() - t) * 1000)
    assert not m.last_error, m.last_error
q = np.array(
    [
        [0.3, 0, 0.03],
        [0.3, 0, 0.08],
        [0.3, 0, 0.15],
        [0.3, 0.15, 0.1],
        [0.45, -0.15, 0.05],
    ]
)
expected = np.array([0, 0.03, 0.10, 0.10, 0.05])
dist = m.clearance(q)
assert dist is not None and np.isfinite(dist).all(), dist
assert np.max(np.abs(dist - expected)) <= 0.02, (dist, expected)
before = float(dist[0])
frame.depth_m = np.full((H, W), 0.6, np.float32)
for _ in range(20):
    m.refresh(frame, T)
    assert not m.last_error, m.last_error
after = m.clearance(q)
assert after is not None
assert before <= 0.01 and after[0] >= 0.015, (before, after[0])
result = {
    "scope": "Synthetic metric depth through real Cascade ZMQ bridge, CUDA TSDF/ESDF and OccupancyMap; fixed box, known clearances, removal carving. Not a robot manipulation test.",
    "passed": True,
    "probe": probe,
    "query_points_m": q.tolist(),
    "expected_before_m": expected.tolist(),
    "measured_before_m": dist.tolist(),
    "max_abs_error_m": float(np.max(np.abs(dist - expected))),
    "measured_after_removal_m": after.tolist(),
    "initial_integrations": 6,
    "removal_integrations": 20,
    "refresh_round_trip_ms": lat,
    "cold_first_ms": lat[0],
    "warm_refresh_median_ms": float(np.median(lat[1:])),
}
a.output.parent.mkdir(exist_ok=True, parents=True)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
