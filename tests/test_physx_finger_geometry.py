"""Derived hull coverage tests run offline, without Kit, sockets or a renderer."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("physx_finger_builder", ROOT / "scripts/build_physx_finger_geometry.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def calibration():
    return json.loads((ROOT / builder.CALIBRATION).read_text())


def artifact():
    return json.loads((ROOT / builder.OUTPUT).read_text())


def cube(center):
    return np.array(list(itertools.product((-1., 1.), repeat=3))) + center


def test_complete_nominal_components_preserved_and_all_derived_vertices_covered():
    nominal = json.loads((ROOT / builder.NOMINAL).read_text())
    result, source = artifact(), calibration()
    assert result["physics_engine"] == "physx"
    for f, old, new in zip(result["fingers"], nominal["fingers"], source["fingers"], strict=True):
        assert f["components"][:8] == old["components"]
        assert len(f["components"]) == 40 and len(new["hulls"]) == 32
        for part, raw in zip(f["components"][8:], new["hulls"], strict=True):
            points = builder.tcp_vertices(raw["vertices"], new["mesh_to_tcp_at_zero_row_major"])
            assert builder.cover_vertices(points, [part])["covered"]
            planes = np.asarray(part["planes"])
            np.testing.assert_allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-12, rtol=0)
            # No broad geometric margin disguised as numerical construction.
            support = (points @ planes[:, :3].T + planes[:, 3]).max(axis=0)
            assert np.max(np.abs(support)) < 3e-12


def test_sdk_raw_planes_are_not_trusted_for_conservative_coverage():
    source = calibration()
    raw_residuals = []
    for finger in source["fingers"]:
        for raw in finger["hulls"]:
            points = np.asarray(raw["vertices"])
            raw_planes = np.asarray([p["plane"] for p in raw["polygons"]])
            raw_residuals.append(float((points @ raw_planes[:, :3].T + raw_planes[:, 3]).max()))
            part = builder.certify_hull(points)
            assert builder.cover_vertices(points, [part])["covered"]
    assert max(raw_residuals) > .0002  # The retained SDK export demonstrates this mismatch.


def test_convex_vertices_in_different_union_members_do_not_certify_interior():
    left, right = cube([0., 0., 0.]), cube([3., 0., 0.])
    components = [builder.certify_hull(left), builder.certify_hull(right)]
    bridge = np.array(list(itertools.product((-1., 4.), (-1., 1.), (-1., 1.))))
    for point in bridge:
        assert any(np.all(np.asarray(c["planes"])[:, :3] @ point + np.asarray(c["planes"])[:, 3] <= 0)
                   for c in components)
    assert not builder.cover_vertices(bridge, components)["covered"]


def test_construction_epsilon_is_not_tracking_or_contact_margin():
    points = cube([0., 0., 0.])
    part = builder.certify_hull(points)
    assert builder.cover_vertices(points, [part])["covered"]
    shifted = points.copy(); shifted[0, 0] -= 1e-8
    assert not builder.cover_vertices(shifted, [part])["covered"]


@pytest.mark.parametrize("points", [np.ones((4, 3)), np.zeros((3, 3)), [[False]*3]*4,
                                   [[float("nan"), 0., 0.]]*4])
def test_degenerate_or_invalid_components_are_never_omitted(points):
    with pytest.raises(Exception):
        builder.certify_hull(points)


def passive_fixture(export_id="nv_x86_09"):
    """A minimal valid export around the reviewed convex arrays; no private files."""
    source = calibration()
    rows, results = [], {}
    for finger in source["fingers"]:
        # Raw source mesh bytes are separately guarded in a real export. Here
        # use a small deterministic mesh and bind its hash in both operands.
        mesh = {"points": cube([0., 0., 0.]).tolist(), "face_vertex_counts": [], "face_vertex_indices": []}
        mesh_sha = builder.canonical_sha(mesh)
        finger["mesh_sha256"] = mesh_sha
        rows.append({"body_path": finger["body_path"], "collision_prim": finger["collision_prim"],
                     "mesh": mesh, "mesh_sha256": mesh_sha,
                     "local_to_body_row_major": copy.deepcopy(finger["local_to_body_row_major"]),
                     "attributes": copy.deepcopy(finger["attributes"])})
        results[finger["collision_prim"]] = {"result": 0, "error": None, "hulls": [copy.deepcopy(h) for h in finger["hulls"] if h["export_id"] == export_id]}
    for exported in source["exports"].values():
        for row, finger in zip(exported["finger_hulls"], source["fingers"], strict=True):
            row["metadata"] = {k: v for k, v in finger.items() if k != "hulls"}
    clock = {"engine": "physx", "epoch": "test-epoch", "robot_id": source["robot_id"]}
    geometry = {"meters_per_unit": 1., "colliders": rows}
    exported = {"version": 1, "pass": True, "geometry_unchanged": True,
                "geometry_before": geometry, "geometry_after": copy.deepcopy(geometry),
                "clock_before": clock, "clock_after": clock.copy(),
                "result_valid_enum_value": 0, "results": results}
    return exported, artifact(), source


@pytest.mark.parametrize("export_id", ["nv_x86_09", "arm_spark_03"])
def test_matching_local_hulls_admitted_without_hull_index_correspondence(export_id):
    exported, geometry, source = passive_fixture(export_id)
    for result in exported["results"].values():
        result["hulls"].reverse()
    report = builder.audit_representation(exported, geometry, source)
    assert report["pass"]
    assert [f["local_hull_count"] for f in report["fingers"]] == [16, 16]


def test_local_hull_beyond_envelope_rejects_without_adapting_artifact():
    exported, geometry, source = passive_fixture()
    before = copy.deepcopy(geometry)
    path = source["fingers"][0]["collision_prim"]
    exported["results"][path]["hulls"][0]["vertices"][0][0] += .1
    assert not builder.audit_representation(exported, geometry, source)["pass"]
    assert geometry == before


@pytest.mark.parametrize("fault", ["engine", "epoch", "units", "transform", "mesh", "parameters", "failed"])
def test_passive_metadata_mismatch_fails_closed(fault):
    exported, geometry, source = passive_fixture()
    if fault == "engine": exported["clock_after"]["engine"] = "newton"
    elif fault == "epoch": exported["clock_after"]["epoch"] = "different"
    elif fault == "failed": exported["pass"] = False
    else:
        for field in ("geometry_before", "geometry_after"):
            item = exported[field]
            if fault == "units": item["meters_per_unit"] = .01
            elif fault == "mesh": item["colliders"][0]["mesh"]["points"][0][0] += .1
            elif fault == "transform": item["colliders"][0]["local_to_body_row_major"][3][0] += .1
            elif fault == "parameters": item["colliders"][0]["attributes"] = {}
    with pytest.raises(ValueError):
        builder.audit_representation(exported, geometry, source)


def test_tracked_source_hash_mismatch_rejects(tmp_path):
    p = tmp_path / "mesh.usda"; p.write_text("source")
    source = {p.name: builder.sha(p)}
    builder.verify_sources(tmp_path, source)
    p.write_text("changed")
    with pytest.raises(ValueError, match="source hash mismatch"):
        builder.verify_sources(tmp_path, source)


def test_retained_can_rim_point_is_missed_by_nominal_and_rejected_by_union():
    # Spark02 frozen source477, witness step12642, exact sensor pixel[154,783].
    # NPZ SHA4133a8c09981d4fd076d56becd93740962aa54e6de791019163bcb67885cfccc.
    # Base point [.21735339400289644,.1885716509809543,.08273729044779554]
    # transformed by FK(q_local) below; target=false and robotmask=false.
    # Retain TCP coordinates so this coverage regression needs no Pinocchio.
    # q_local=[-.5301125049591064,1.509192943572998,.9398283958435059,
    #          -1.0809636116027832,.0051458170637488365,-.8502564430236816]
    point_tcp = np.array([-.05771986007167669, .0831421387761342, .004152247458677672])
    right_stroke = .04990166053175926
    nominal = json.loads((ROOT / builder.NOMINAL).read_text())["fingers"][1]
    union = artifact()["fingers"][1]
    point_at_zero = point_tcp - np.asarray(nominal["axis_tcp"]) * right_stroke

    def residuals(finger):
        return [float((np.asarray(c["planes"])[:, :3] @ point_at_zero
                       + np.asarray(c["planes"])[:, 3]).max()) for c in finger["components"]]

    assert min(residuals(nominal)) > 0  # Historical nominal check misses it.
    assert min(residuals(union)) < 0    # Union rejects it, without a new margin.


@pytest.mark.parametrize("field", ["axis_tcp", "lower_m", "upper_m", "name", "units", "frame", "version", "provenance"])
def test_artifact_metadata_cannot_change_geometry_semantics(field):
    original = artifact()
    changed = copy.deepcopy(original)
    if field in ("axis_tcp", "lower_m", "upper_m", "name"):
        values = {"axis_tcp": [1., 0., 0.], "lower_m": -.01, "upper_m": .1, "name": "other"}
        changed["fingers"][0][field] = values[field]
    elif field == "provenance":
        changed["collision_representation"]["kind"] = "different"
    else:
        changed[field] = {"units": "cm", "frame": "different", "version": 2}[field]
    with pytest.raises(ValueError):
        builder.validate_artifact_metadata(changed, original)
    if field != "provenance":
        exported, _, source = passive_fixture()
        with pytest.raises(ValueError):
            builder.audit_representation(exported, changed, source)


@pytest.mark.parametrize("fault", ["mesh", "zero_transform", "local_transform", "units", "engine",
                                   "robot", "epoch", "empty_epoch", "duplicate_export_id", "missing_hull", "index", "raw_vertex"])
def test_each_export_is_bound_independently_without_omission(fault):
    c = calibration()
    arm = c["exports"]["arm_spark_03"]
    raw = c["fingers"][0]["hulls"][16]
    if fault == "mesh": arm["finger_hulls"][0]["metadata"]["mesh_sha256"] = "0"*64
    elif fault == "zero_transform": arm["finger_hulls"][0]["metadata"]["mesh_to_tcp_at_zero_row_major"][3][0] += .001
    elif fault == "local_transform": arm["finger_hulls"][0]["metadata"]["local_to_body_row_major"][3][0] += .001
    elif fault == "units": arm["units"] = "cm"
    elif fault == "engine": arm["physics_engine"] = "newton"
    elif fault == "robot": arm["robot_id"] = "other"
    elif fault == "epoch": arm["producer_epoch"] = "different"
    elif fault == "empty_epoch": arm["producer_epoch"] = ""
    elif fault == "duplicate_export_id": raw["export_id"] = "nv_x86_09"
    elif fault == "missing_hull": c["fingers"][0]["hulls"].pop()
    elif fault == "index": raw["source_hull_index"] = 1
    elif fault == "raw_vertex": raw["vertices"][0][0] += .001
    with pytest.raises(ValueError):
        builder.validate_calibration_exports(c)


def test_component_provenance_identifies_original_export_and_index():
    c, result = calibration(), artifact()
    builder.validate_calibration_exports(c)
    for finger, saved in zip(c["fingers"], result["fingers"], strict=True):
        assert [(h["export_id"], h["source_hull_index"]) for h in finger["hulls"]] == [
            (export_id, index) for export_id in ("nv_x86_09", "arm_spark_03") for index in range(16)]
        for part, raw in zip(saved["components"][8:], finger["hulls"], strict=True):
            assert part["provenance"] == {"source": "requested_physx_representation",
                "export_id": raw["export_id"], "source_hull_index": raw["source_hull_index"]}
