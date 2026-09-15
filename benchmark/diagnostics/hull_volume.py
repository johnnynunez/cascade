"""Bounding boxes match. Does the hull FILL a concavity the visual mesh has?

The bbox test came back 0.00 mm on every axis, so the collision mesh is not a
scaled-up copy. But a convex hull of an L-shaped finger has the SAME bounding
box as the L while filling the notch between its arms -- bbox cannot see that.
Volume can.

If the finger is L-shaped and the collision mesh is its convex hull, the hull
volume will be markedly larger than the visual volume, and the extra material
sits exactly in the notch: the inner face where the two fingers meet.

Compute the signed volume of each closed mesh via the divergence theorem
(sum of tetrahedra from the origin), then check whether the collision mesh is
actually convex by testing its own vertices against its hull.

Then answer the question that matters: along the axis the fingers close on,
where does the collision surface sit versus the visual surface?

Set REBOT_MJCF_ROOT to the reBot-Isaacsim mjcf/rebot_devarm directory.
"""
import os
import struct
from pathlib import Path

import numpy as np

REBOT_ROOT = Path(os.environ.get(
    "REBOT_MJCF_ROOT",
    Path(__file__).resolve().parents[3] / "reBot-Isaacsim/mjcf/rebot_devarm",
)).expanduser()
ASSETS = REBOT_ROOT / "assets"
if not ASSETS.is_dir():
    raise SystemExit("Set REBOT_MJCF_ROOT to a reBot-Isaacsim mjcf/rebot_devarm directory with assets.")


def load_obj_faces(path):
    v, f = [], []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            p = line.split()
            v.append([float(p[1]), float(p[2]), float(p[3])])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                f.append([idx[0], idx[k], idx[k + 1]])
    return np.array(v, float), np.array(f, int)


def load_stl_faces(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + n * 50 != len(data):
        raise ValueError("not binary stl")
    tris = np.zeros((n, 3, 3))
    for i in range(n):
        off = 84 + i * 50 + 12
        tris[i] = np.array(struct.unpack("<9f", data[off:off + 36])).reshape(3, 3)
    v = tris.reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    return v, f


def volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0))


print(f"{'finger':<7} {'visual cm3':>11} {'collision cm3':>14} {'ratio':>8} {'extra cm3':>10}")
print("-" * 54)

for side in ("left", "right"):
    vv, vf = load_obj_faces(ASSETS / f"gripper_{side}_merged_gripper.obj")
    cv, cf = load_stl_faces(ASSETS / f"gripper_{side}_convex.stl")
    volv = volume(vv, vf) * 1e6      # m3 -> cm3
    volc = volume(cv, cf) * 1e6
    print(f"{side:<7} {volv:>11.3f} {volc:>14.3f} {volc/max(volv,1e-9):>8.2f}x "
          f"{volc-volv:>+9.3f}")

print("\n=== where does the extra material sit? ===")
print("slice both meshes along each axis and compare the extent of the")
print("collision surface against the visual one, in 10 bands")

for side in ("left",):
    vv, _ = load_obj_faces(ASSETS / f"gripper_{side}_merged_gripper.obj")
    cv, _ = load_stl_faces(ASSETS / f"gripper_{side}_convex.stl")
    lo = np.minimum(vv.min(0), cv.min(0))
    hi = np.maximum(vv.max(0), cv.max(0))
    axis_names = "xyz"
    # the fingers translate along one axis; report all three so we can see
    # which one carries the filled notch
    for ax in range(3):
        edges = np.linspace(lo[ax], hi[ax], 11)
        gaps = []
        for i in range(10):
            m_v = (vv[:, ax] >= edges[i]) & (vv[:, ax] < edges[i + 1])
            m_c = (cv[:, ax] >= edges[i]) & (cv[:, ax] < edges[i + 1])
            if m_v.sum() < 3 or m_c.sum() < 3:
                gaps.append(None)
                continue
            other = [a for a in range(3) if a != ax]
            spread_v = (vv[m_v][:, other].max(0) - vv[m_v][:, other].min(0))
            spread_c = (cv[m_c][:, other].max(0) - cv[m_c][:, other].min(0))
            gaps.append(float(np.max(spread_c - spread_v)) * 1000)
        shown = " ".join("  --  " if g is None else f"{g:+6.2f}" for g in gaps)
        print(f"  {side} axis {axis_names[ax]}: {shown}")

print("\nmm of extra cross-section in each band (collision minus visual).")
print("a large positive run in one region = the filled notch.")
