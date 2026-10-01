#!/usr/bin/env python3
"""Build conservative collision hulls from every connected finger mesh part.

Offline only (SciPy); the artifact keeps original source hashes, metric units,
and coverage certificates. No components, including small ones, are dropped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
OUTPUT = ROOT / "assets/grasp_geometry/rebot_rs_fingers.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin(element):
    e = element.find("origin")
    xyz = np.fromstring(e.get("xyz", "0 0 0"), sep=" ") if e is not None else np.zeros(3)
    rpy = np.fromstring(e.get("rpy", "0 0 0"), sep=" ") if e is not None else np.zeros(3)
    return Rotation.from_euler("xyz", rpy).as_matrix(), xyz


def mesh_parts(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + count * 50:
        raise ValueError("expected exact binary STL length")
    dtype = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    faces = np.frombuffer(raw, dtype, count=count, offset=84)["v"].astype(float)
    if not np.isfinite(faces).all():
        raise ValueError("nonfinite STL vertex")
    vertices, indices = np.unique(faces.reshape(-1, 3), axis=0, return_inverse=True)
    indices = indices.reshape(-1, 3)
    parents = list(range(len(vertices)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for a, b, c in indices:
        parents[find(b)] = parents[find(a)]
        parents[find(c)] = parents[find(a)]
    labels = np.array([find(i) for i in range(len(vertices))])
    parts = []
    for label in sorted(set(labels)):
        selected = labels == label
        face_mask = selected[indices].all(axis=1)
        parts.append((vertices[selected], int(face_mask.sum())))
    if sum(n for _, n in parts) != count:
        raise ValueError("not every source triangle belongs to a component")
    return parts


def build():
    root = ET.parse(URDF).getroot()
    sources = {str(URDF.relative_to(ROOT)): sha(URDF)}
    fingers = []
    for name in ("joint_left", "joint_right"):
        joint = root.find(f"joint[@name='{name}']")
        if joint.get("type") != "prismatic" or joint.find("parent").get("link") != "gripper_end":
            raise ValueError("finger must be prismatic in the configured TCP frame")
        Rj, tj = origin(joint)
        axis = Rj @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if not np.isclose(np.linalg.norm(axis), 1., atol=1e-12):
            raise ValueError("joint axis must be unit length")
        link = root.find(f"link[@name='{joint.find('child').get('link')}']")
        components = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("unsupported collision primitive; never omit it")
            uri = mesh.get("filename")
            if not uri.startswith("package://00-arm-rs_asm-v3/"):
                raise ValueError("unresolved collision mesh URI")
            path = URDF.parent.parent / uri.split("package://00-arm-rs_asm-v3/", 1)[1]
            sources[str(path.relative_to(ROOT))] = sha(path)
            Rc, tc = origin(collision)
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("invalid mesh scale")
            for vertices, face_count in mesh_parts(path):
                points = ((vertices * scale) @ Rc.T + tc) @ Rj.T + tj
                # Qhull raises on degenerate parts: no joggling or omission.
                hull = ConvexHull(points)
                planes = hull.equations
                if not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-12):
                    raise ValueError("nonunit halfspace normals")
                worst = max(float((chunk @ planes[:, :3].T + planes[:, 3]).max())
                            for chunk in np.array_split(points, max(1, len(points) // 128)))
                if worst > 1e-12:
                    raise ValueError("hull does not cover every source vertex/triangle")
                # Inflate by floating-point construction error only. Convexity
                # then also contains each whole triangle spanned by vertices.
                planes[:, 3] -= max(0., worst) + 1e-12
                components.append({"planes": planes.tolist(), "min_m": points.min(0).tolist(),
                                   "max_m": points.max(0).tolist(), "vertices": len(points),
                                   "triangles": face_count, "coverage_error_m": worst})
        if not components:
            raise ValueError("finger has no collision components")
        limit = joint.find("limit")
        fingers.append({"name": name, "axis_tcp": axis.tolist(),
                        "lower_m": float(limit.get("lower")), "upper_m": float(limit.get("upper")),
                        "components": components})
    return {"version": 1, "units": "m", "frame": "gripper_end", "sources": sources,
            "scope": "convex hull of each complete connected collision-mesh component; fingers only",
            "fingers": fingers}


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), separators=(",", ":"), allow_nan=False) + "\n")
    print(f"{OUTPUT.relative_to(ROOT)} {sha(OUTPUT)}")
