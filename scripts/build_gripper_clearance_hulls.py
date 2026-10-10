#!/usr/bin/env python3
"""Build the reBot RS forearm/wrist/gripper clearance hulls (B73, offline, SciPy).

The analytic planner's angled and side candidates (B45, the opt-in `*_reach`
arm profiles) are vetted against the support plane and the target's observed
box by `cascade.grasping.gripper_clearance`. That vet needs the arm's
collision geometry without SciPy at runtime, so this script turns the URDF's
collision meshes into a JSON asset once:

* one convex hull per CONNECTED COMPONENT of each collision mesh (the
  convention of `build_gripper_scene_geometry.py`, the observed-finger
  asset). A whole-link hull is too loose for L-shaped parts: the finger's
  slide carriage behind the palm reaches 8.6 mm past the jaw centreline, so a
  single finger hull slants into the jaw gap (measured: 2 mm "clearance" to a
  5 cm cube the open jaws clear by 20 mm);
* components whose hull lies inside a larger kept hull of the same link
  (screws, bearings inside the housing) are dropped: they can touch nothing
  their container does not touch first, and the union is unchanged;
* hull vertices (link frame). The lowest point of a convex hull is one of its
  vertices and every hull vertex is a mesh vertex, so the support-plane test
  is EXACT for the mesh, not only for the hull;
* separating-axis candidates: each hull's face normals, rounded to 1e-3 and
  de-duplicated. Any unit axis gives a valid lower bound of the distance
  between two convex sets, so the rounding only costs tightness;
* the source URDF and STL sha256, so a changed mesh makes the asset stale.

Coordinates are written with 7 decimals (0.1 um), far below any margin.

Links: the forearm `link3` onward (`link3`..`link6`, the housing
`gripper_end` and both fingers). Measured on the B45 grid: a side approach at
5 cm drops `link5` (the wrist motor) 3.1 mm below the table and `link3` to
8 mm above it while every joint origin passes the harness (they sit inside
the grasp exemption cylinder); `link2` stays >= 97 mm above it.

Run (writes the asset, prints its sha256):

    PYTHONPATH=src python scripts/build_gripper_clearance_hulls.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
OUTPUT = ROOT / "assets/grasp_geometry/rebot_rs_clearance_hulls.json"
SCHEMA = "cascade.gripper_clearance_hulls/1"
#: forearm, wrist, housing and fingers (see the module docstring)
LINKS = ("link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")
FINGER_JOINTS = ("joint_left", "joint_right")
DECIMALS = 7
AXIS_DECIMALS = 3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _origin(element):
    from scipy.spatial.transform import Rotation

    e = element.find("origin")
    xyz = np.fromstring(e.get("xyz", "0 0 0"), sep=" ") if e is not None else np.zeros(3)
    rpy = np.fromstring(e.get("rpy", "0 0 0"), sep=" ") if e is not None else np.zeros(3)
    return Rotation.from_euler("xyz", rpy).as_matrix(), xyz


def mesh_components(path: Path):
    """Binary STL -> [(component vertices (n, 3), triangle count)] (the
    observed-finger asset's reader: exact length, finite, every triangle in
    exactly one component)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_gripper_scene_geometry import mesh_parts

    return mesh_parts(path)


def link_components(points_list):
    """[(n_i, 3) source points] -> kept hull components, dropped count.

    Larger hulls first; a component whose hull vertices all lie inside an
    already kept hull is dropped (its container is kept, so the union and
    the lowest vertex are unchanged)."""
    from scipy.spatial import ConvexHull

    hulls = []
    for points in points_list:
        hull = ConvexHull(points)  # Qhull raises on degenerate parts: never omitted
        if not np.allclose(np.linalg.norm(hull.equations[:, :3], axis=1), 1.0, atol=1e-12):
            raise ValueError("nonunit halfspace normals")
        # coverage: every source point inside its own hull (float error only)
        if float((points @ hull.equations[:, :3].T + hull.equations[:, 3]).max()) > 1e-9:
            raise ValueError("hull does not cover its source vertices")
        hulls.append((float(hull.volume), points[hull.vertices], hull.equations))
    hulls.sort(key=lambda h: -h[0])
    kept, dropped = [], 0
    for _, vertices, planes in hulls:
        if any(float((vertices @ p[:, :3].T + p[:, 3]).max()) <= 1e-9 for _, p in kept):
            dropped += 1
            continue
        kept.append((vertices, planes))
    components = []
    for vertices, planes in kept:
        axes = np.unique(np.round(planes[:, :3], AXIS_DECIMALS), axis=0)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        components.append({"vertices": np.round(vertices, DECIMALS).tolist(),
                           "axes": np.round(axes, DECIMALS).tolist()})
    return components, dropped


def build() -> dict:
    root = ET.parse(URDF).getroot()
    sources = {str(URDF.relative_to(ROOT)): sha(URDF)}
    links, dropped = {}, {}
    for name in LINKS:
        link = root.find(f"link[@name='{name}']")
        if link is None:
            raise ValueError(f"link {name!r} not in the URDF")
        points = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("unsupported collision primitive; never omit it")
            uri = mesh.get("filename") or ""
            if not uri.startswith("package://00-arm-rs_asm-v3/"):
                raise ValueError("unresolved collision mesh URI")
            path = URDF.parent.parent / uri.split("package://00-arm-rs_asm-v3/", 1)[1]
            sources[str(path.relative_to(ROOT))] = sha(path)
            Rc, tc = _origin(collision)
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("invalid mesh scale")
            points += [(vertices * scale) @ Rc.T + tc for vertices, _ in mesh_components(path)]
        if not points:
            raise ValueError(f"link {name!r} has no collision components")
        links[name], dropped[name] = link_components(points)
    fingers = []
    for name in FINGER_JOINTS:
        joint = root.find(f"joint[@name='{name}']")
        if joint is None or joint.get("type") != "prismatic":
            raise ValueError(f"finger joint {name!r} must be prismatic")
        limit = joint.find("limit")
        fingers.append({"name": name, "lower_m": float(limit.get("lower", "nan")),
                        "upper_m": float(limit.get("upper", "nan"))})
    return {"schema": SCHEMA, "units": "m", "frame": "each link's own URDF frame",
            "scope": "convex hull of each connected collision-mesh component, forearm onward; "
                     "components inside a larger kept hull of the same link dropped",
            "sources": sources, "links": links, "dropped_contained_components": dropped,
            "finger_joints": fingers}


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), separators=(",", ":"), allow_nan=False) + "\n")
    print(f"{OUTPUT.relative_to(ROOT)} {sha(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
