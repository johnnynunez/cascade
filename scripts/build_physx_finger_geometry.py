#!/usr/bin/env python3
"""Build nominal + source-bound PhysX finger envelopes; audit a passive export.

No Kit, sockets, physics steps or authoring. The requested PhysX representation
is derived cooking output, not a direct read of active PxShape vertices.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
NOMINAL = Path("assets/grasp_geometry/rebot_rs_fingers.json")
CALIBRATION = Path("assets/grasp_geometry/physx_finger_cooking_source.json")
OUTPUT = Path("assets/grasp_geometry/rebot_rs_fingers_physx.json")
EPS_M = 1e-12  # Same floating-point construction allowance as nominal hulls.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _vertices(value):
    points = np.asarray(value)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
            or points.dtype.kind not in "fiu" or not np.isfinite(points).all()):
        raise ValueError("invalid convex vertices")
    return points.astype(float)


def _transform(value):
    T = np.asarray(value)
    if (T.dtype.kind not in "fiu" or T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.array_equal(T[:, 3], [0, 0, 0, 1])
            or np.linalg.det(T[:3, :3]) <= 0
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9, rtol=0)):
        raise ValueError("invalid row-vector mesh-to-TCP transform")
    return T.astype(float)


def tcp_vertices(vertices, transform):
    points = _vertices(vertices)
    return (np.c_[points, np.ones(len(points))] @ _transform(transform))[:, :3]


def certify_hull(points):
    """Recompute QHull; raw SDK planes are provenance only, never collision tests."""
    points = _vertices(points)
    hull = ConvexHull(points)  # Degenerate components fail; never joggle or omit.
    planes = hull.equations.copy()
    if not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-12, rtol=0):
        raise ValueError("nonunit QHull normals")
    residual = float((points @ planes[:, :3].T + planes[:, 3]).max())
    if residual > EPS_M:
        raise ValueError("QHull does not cover source vertices")
    planes[:, 3] -= max(0., residual) + EPS_M
    return {"planes": planes.tolist(), "min_m": points.min(0).tolist(),
            "max_m": points.max(0).tolist(), "vertices": len(points),
            "triangles": len(hull.simplices), "coverage_error_m": residual}


def verify_sources(root, sources):
    root = Path(root).resolve()
    for relative, expected in sources.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("geometry source hash mismatch: " + relative)


def validate_calibration_exports(calibration):
    """Bind every original convex and coordinate contract to its own export."""
    exports = calibration.get("exports")
    provenance = calibration.get("request_provenance", {}).get("exports")
    if (calibration.get("version") != 2 or not isinstance(exports, dict) or not exports
            or not isinstance(provenance, dict) or set(exports) != set(provenance)):
        raise ValueError("invalid calibration export inventory")
    fingers = calibration["fingers"]
    names = [f["name"] for f in fingers]
    if names != ["joint_left", "joint_right"]:
        raise ValueError("calibration must contain both fingers exactly once")
    for export_id, exported in exports.items():
        if not isinstance(export_id, str) or not export_id:
            raise ValueError("invalid export identity")
        if (exported.get("units") != calibration.get("units") or calibration.get("units") != "m"
                or exported.get("physics_engine") != calibration.get("physics_engine")
                or calibration.get("physics_engine") != "physx"
                or exported.get("robot_id") != calibration["robot_id"]):
            raise ValueError("export coordinate/engine identity mismatch")
        epoch = exported.get("producer_epoch")
        if not isinstance(epoch, str) or not epoch or epoch != provenance[export_id].get("clock_epoch"):
            raise ValueError("export epoch/provenance mismatch")
        for field in ("representation_sha256", "receipt_sha256"):
            digest = provenance[export_id].get(field)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid export source hash")
        rows = exported["finger_hulls"]
        if [row["name"] for row in rows] != names:
            raise ValueError("export finger inventory mismatch")
        for finger, row in zip(fingers, rows, strict=True):
            metadata = {k: v for k, v in finger.items() if k != "hulls"}
            if row["metadata"] != metadata or row["collision_prim"] != finger["collision_prim"]:
                raise ValueError("export mesh/kinematics/parameters mismatch")
            hulls = [h for h in finger["hulls"] if h.get("export_id") == export_id]
            count = row["hull_count"]
            if (type(count) is not int or count <= 0 or len(hulls) != count
                    or [h.get("source_hull_index") for h in hulls] != list(range(count))
                    or any(type(h.get("source_hull_index")) is not int for h in hulls)):
                raise ValueError("export original hull inventory mismatch")
            raw = [{k: v for k, v in h.items() if k not in ("export_id", "source_hull_index")}
                   for h in hulls]
            if canonical_sha(raw) != row["raw_hulls_sha256"]:
                raise ValueError("export raw hull hash mismatch")
    for finger in fingers:
        if any(h.get("export_id") not in exports for h in finger["hulls"]):
            raise ValueError("unbound hull export identity")


def build(root=ROOT):
    root = Path(root)
    nominal = json.loads((root / NOMINAL).read_text())
    calibration = json.loads((root / CALIBRATION).read_text())
    if (calibration.get("version") != 2 or calibration.get("physics_engine") != "physx"
            or calibration.get("units") != "m"
            or nominal.get("frame") != "gripper_end"):
        raise ValueError("unsupported calibration")
    validate_calibration_exports(calibration)
    sources = {**nominal["sources"], **calibration["sources"],
               str(NOMINAL): sha(root / NOMINAL), str(CALIBRATION): sha(root / CALIBRATION)}
    verify_sources(root, sources)
    result = copy.deepcopy(nominal)
    result["sources"] = sources
    result["physics_engine"] = "physx"
    result["scope"] = ("union of complete nominal finger components and source-bound requested "
                       "PhysX convex components; fingers only; passive local coverage admission required")
    result["collision_representation"] = {
        "kind": "nominal_plus_physx_derived_convex_union",
        "calibration_sha256": sha(root / CALIBRATION),
        "construction_epsilon_m": EPS_M,
        "request_provenance": calibration["request_provenance"],
        "scope": "Derived cooking output, not a direct active-shape getter. No contact/rest-offset expansion.",
    }
    if [f["name"] for f in calibration["fingers"]] != ["joint_left", "joint_right"]:
        raise ValueError("calibration must contain both fingers exactly once")
    for finger, derived in zip(result["fingers"], calibration["fingers"], strict=True):
        if finger["name"] != derived["name"] or not derived["hulls"]:
            raise ValueError("finger calibration mismatch")
        # Do not replace, simplify or drop any nominal component.
        for raw in derived["hulls"]:
            part = certify_hull(tcp_vertices(raw["vertices"], derived["mesh_to_tcp_at_zero_row_major"]))
            part["provenance"] = {"source": "requested_physx_representation",
                                  "export_id": raw["export_id"],
                                  "source_hull_index": raw["source_hull_index"]}
            finger["components"].append(part)
    return result


def cover_vertices(points, components):
    """One entire incoming convex must fit inside one artifact convex.

    Testing vertices against different members of a union would miss bridges
    across its gaps. Convexity proves containment of all faces/interior only
    when the same containing component covers every vertex.
    """
    points = _vertices(points)
    best = float("inf")
    for index, part in enumerate(components):
        planes = np.asarray(part["planes"], dtype=float)
        lo, hi = np.asarray(part["min_m"], dtype=float), np.asarray(part["max_m"], dtype=float)
        if (planes.ndim != 2 or planes.shape[1] != 4 or len(planes) < 4
                or not np.isfinite(planes).all()
                or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-12, rtol=0)
                or lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
                or np.any(lo > hi)):
            raise ValueError("invalid containing convex")
        residual = float((points @ planes[:, :3].T + planes[:, 3]).max())
        best = min(best, residual)
        if (residual <= 0 and np.all(points >= np.asarray(part["min_m"]) - EPS_M)
                and np.all(points <= np.asarray(part["max_m"]) + EPS_M)):
            return {"covered": True, "component": index, "max_halfspace_residual_m": residual}
    return {"covered": False, "component": None, "best_component_residual_m": best}


def validate_kinematic_metadata(artifact, nominal):
    for key in ("version", "units", "frame"):
        if artifact.get(key) != nominal.get(key):
            raise ValueError("artifact coordinate metadata mismatch: " + key)
    if len(artifact["fingers"]) != len(nominal["fingers"]):
        raise ValueError("artifact finger count mismatch")
    for saved, original in zip(artifact["fingers"], nominal["fingers"], strict=True):
        a = {k: v for k, v in saved.items() if k != "components"}
        b = {k: v for k, v in original.items() if k != "components"}
        if a != b:
            raise ValueError("artifact finger kinematics mismatch")


def validate_artifact_metadata(artifact, generated):
    validate_kinematic_metadata(artifact, generated)
    a = {k: v for k, v in artifact.items() if k != "fingers"}
    b = {k: v for k, v in generated.items() if k != "fingers"}
    if a != b:
        raise ValueError("artifact representation/provenance mismatch")


def audit_representation(representation, artifact, calibration, *, nominal=None):
    """Audit a separately guarded passive export; this does not acquire it."""
    validate_calibration_exports(calibration)
    if nominal is None:
        nominal = json.loads((ROOT / NOMINAL).read_text())
    validate_kinematic_metadata(artifact, nominal)
    if (representation.get("version") != 1 or representation.get("pass") is not True
            or representation.get("geometry_unchanged") is not True
            or representation.get("geometry_before") != representation.get("geometry_after")
            or artifact.get("physics_engine") != "physx"
            or calibration.get("physics_engine") != "physx"):
        raise ValueError("invalid or changed passive representation")
    before, after = representation["clock_before"], representation["clock_after"]
    if (before.get("engine") != "physx" or after.get("engine") != "physx"
            or not before.get("epoch") or before["epoch"] != after.get("epoch")
            or before.get("robot_id") != calibration["robot_id"]
            or after.get("robot_id") != calibration["robot_id"]):
        raise ValueError("passive representation identity mismatch")
    geometry = representation["geometry_before"]
    if geometry.get("meters_per_unit") != 1.:
        raise ValueError("passive representation is not metric")
    rows = {x["collision_prim"]: x for x in geometry["colliders"]}
    if len(rows) != len(geometry["colliders"]):
        raise ValueError("duplicate collider identity")
    answers = []
    for finger, derived in zip(artifact["fingers"], calibration["fingers"], strict=True):
        if finger["name"] != derived["name"]:
            raise ValueError("artifact/calibration finger mismatch")
        path = derived["collision_prim"]
        raw = rows[path]
        if (raw["body_path"] != derived["body_path"]
                or canonical_sha(raw["mesh"]) != derived["mesh_sha256"]
                or raw["mesh_sha256"] != derived["mesh_sha256"]
                or raw["local_to_body_row_major"] != derived["local_to_body_row_major"]
                or raw["attributes"] != derived["attributes"]):
            raise ValueError("passive source mesh, transform or physics parameters changed")
        result = representation["results"][path]
        if (result["result"] != representation["result_valid_enum_value"]
                or result.get("error") or not result["hulls"]):
            raise ValueError("passive convex request failed")
        hulls = [cover_vertices(tcp_vertices(h["vertices"], derived["mesh_to_tcp_at_zero_row_major"]),
                               finger["components"]) for h in result["hulls"]]
        answers.append({"finger": finger["name"], "collision_prim": path,
                        "local_hull_count": len(hulls), "hulls": hulls,
                        "covered": all(h["covered"] for h in hulls)})
    return {"pass": all(f["covered"] for f in answers), "physics_engine": "physx",
            "epoch": before["epoch"], "construction_epsilon_m": EPS_M, "fingers": answers,
            "scope": "Each requested local convex is entirely covered by one artifact convex. "
                     "No direct active-PxShape identity, contact-offset or unseen-surface guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="validate stored artifact geometry and provenance")
    parser.add_argument("--audit-representation", type=Path, help="passive export JSON; no network or Kit")
    parser.add_argument("--output", type=Path, help="audit receipt path")
    args = parser.parse_args()
    generated = build(args.root)
    if args.audit_representation:
        if not args.output:
            parser.error("--audit-representation requires --output")
        artifact = json.loads((args.root / OUTPUT).read_text())
        verify_sources(args.root, artifact["sources"])
        validate_artifact_metadata(artifact, generated)
        calibration = json.loads((args.root / CALIBRATION).read_text())
        result = audit_representation(json.loads(args.audit_representation.read_text()), artifact, calibration,
                                      nominal=json.loads((args.root / NOMINAL).read_text()))
        result.update({"representation_sha256": sha(args.audit_representation),
                       "artifact_sha256": sha(args.root / OUTPUT), "calibration_sha256": sha(args.root / CALIBRATION),
                       "script_sha256": sha(__file__)})
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("PASS" if result["pass"] else "FAIL", args.output)
        return 0 if result["pass"] else 1
    if args.check:
        stored = json.loads((args.root / OUTPUT).read_text())
        verify_sources(args.root, stored["sources"])
        validate_artifact_metadata(stored, generated)
        if stored["sources"] != generated["sources"] or stored["physics_engine"] != "physx":
            raise ValueError("stored artifact source/engine mismatch")
        # QHull plane ordering can vary by platform; verify all original hull
        # vertices directly, not byte equality with a reconstructed hull.
        calibration = json.loads((args.root / CALIBRATION).read_text())
        nominal = json.loads((args.root / NOMINAL).read_text())
        for saved, orig, derived in zip(stored["fingers"], nominal["fingers"], calibration["fingers"], strict=True):
            n = len(orig["components"])
            if saved["components"][:n] != orig["components"] or len(saved["components"]) != n+len(derived["hulls"]):
                raise ValueError("nominal/derived components were omitted or changed")
            for part, raw in zip(saved["components"][n:], derived["hulls"], strict=True):
                if part["provenance"] != {"source": "requested_physx_representation",
                                          "export_id": raw["export_id"],
                                          "source_hull_index": raw["source_hull_index"]}:
                    raise ValueError("stored component export provenance mismatch")
                points = tcp_vertices(raw["vertices"], derived["mesh_to_tcp_at_zero_row_major"])
                if not cover_vertices(points, [part])["covered"]:
                    raise ValueError("stored derived component does not cover its vertices")
                planes = np.asarray(part["planes"], dtype=float)
                support = (points @ planes[:, :3].T + planes[:, 3]).max(axis=0)
                if np.max(np.abs(support)) > 3*EPS_M:
                    raise ValueError("stored planes are not tight source-vertex supports")
        print("PASS", args.root / OUTPUT)
        return 0
    path = args.root / OUTPUT
    path.write_text(json.dumps(generated, separators=(",", ":"), allow_nan=False) + "\n")
    print(path, sha(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
