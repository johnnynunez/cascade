"""Synthetic support-source regressions; no live acceptance or engine claims."""
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_spark_placement_proof import audit, gpu, support_fixture, trajectory


def simplified_trajectory():
    rows, expected = trajectory()
    geometry = rows[0]["physics"]["scene_geometry"]["convex_collider"]
    vertices = np.asarray(geometry["vertices_m"])
    # A synthetic six-vertex representation preserves the source's bounds.
    # Its support differs from the full authored ellipsoid when tilted.
    selected = sorted(set(vertices.argmin(axis=0).tolist() + vertices.argmax(axis=0).tolist()))
    angle = np.pi / 6
    quaternion = [np.cos(angle / 2), 0., np.sin(angle / 2), 0.]
    support_z = -np.sin(angle) * vertices[selected, 0] + np.cos(angle) * vertices[selected, 2]
    body_z = expected["open_box"]["support_top_z_m"] - support_z.min()
    for i, row in enumerate(rows):
        shape = row["physics"]["scene_geometry"]["convex_collider"]
        shape["physx_support"] = support_fixture(vertices, shape["face_vertex_counts"],
                                                shape["face_vertex_indices"], selected)
        shape["physx_support"]["physics_step"] = row["physics"]["physics_step"]
        if 12 <= i < 26:
            prop = row["physics"]["props"]["orange"]
            prop["position_m"][2] = float(body_z)
            prop["orientation_wxyz"] = list(quaternion)
    return rows, expected


def test_returned_vertices_change_only_support_not_authored_footprint():
    rows, expected = simplified_trajectory()
    result = audit(rows, expected)
    assert result["pass"], result
    footprint = result["convex_footprint"]
    assert footprint["vertex_count"] == 266 and footprint["support_vertex_count"] == 6
    assert footprint["max_support_gap_m"] == footprint["max_penetration_m"] == .001
    assert footprint["max_abs_support_gap_m"] < 1e-12
    for row in rows[12:26]:
        row["physics"]["props"]["orange"]["position_m"][0] = .3395
    result = audit(rows, expected)
    assert result["pick"]["pass"]  # centre and support are still valid
    assert result["convex_footprint"]["checks"]["actual_lowest_vertex_supported_at_every_recorded_pose"]
    assert not result["convex_footprint"]["checks"]["whole_collider_inside_square_at_every_recorded_pose"]
    assert not result["pass"]


def test_returned_support_keeps_thin_floor_and_quaternion_sign_checks():
    rows, expected = simplified_trajectory()
    for row in rows[12:26]:
        prop = row["physics"]["props"]["orange"]
        prop["orientation_wxyz"] = [-v for v in prop["orientation_wxyz"]]
    assert audit(rows, expected)["pass"]
    for row in rows[12:26]:
        row["physics"]["props"]["orange"]["position_m"][2] -= .004
    result = audit(rows, expected)
    assert result["pick"]["pass"]  # the legacy5mm test alone cannot prove this4mm floor
    assert not result["convex_footprint"]["checks"]["actual_lowest_vertex_supported_at_every_recorded_pose"]
    assert not result["pass"]


@pytest.mark.parametrize("fault", ["missing", "null", "source_hash", "topology_hash", "returned_hash", "prim", "stage",
    "physics_step", "result", "count", "frame", "engine", "duplicate", "negative_index", "fractional_index",
    "wrong_count", "boolean_count", "degenerate", "changed_bounds", "drift", "stage_drift"])
def test_support_data_must_match_its_live_authored_shape_in_every_sample(fault):
    rows, expected = simplified_trajectory()
    for i, row in enumerate(rows):
        geometry = row["physics"]["scene_geometry"]["convex_collider"]
        support = geometry["physx_support"]
        if fault == "missing":
            geometry.pop("physx_support")
        elif fault == "null":
            geometry["physx_support"] = None
        elif fault == "boolean_count":
            support["convex_count"] = True
        elif fault in {"source_hash", "topology_hash", "returned_hash"}:
            key = {"source_hash": "source_vertices_f32_sha256", "topology_hash": "source_topology_sha256",
                   "returned_hash": "vertices_f32_sha256"}[fault]
            support[key] = "0" * 64
        elif fault in {"prim", "stage", "physics_step", "result", "count", "frame", "engine", "wrong_count"}:
            key, value = {"prim": ("collider_path", "/other/Collision"), "stage": ("stage_id", 9),
                "physics_step": ("physics_step", -1), "result": ("result", "RESULT_ERROR_NOT_READY"),
                "count": ("convex_count", 2), "frame": ("frame", "world"), "engine": ("engine", "newton"),
                "wrong_count": ("vertex_count", 100)}[fault]
            support[key] = value
        elif fault == "duplicate":
            support["authored_vertex_indices"][1] = support["authored_vertex_indices"][0]
        elif fault == "negative_index":
            support["authored_vertex_indices"][0] = -1
        elif fault == "fractional_index":
            support["authored_vertex_indices"][0] = 0.5
        elif fault in {"degenerate", "changed_bounds"}:
            selected = [121, 122, 123, 124] if fault == "degenerate" else [0, 1, 2, 25]
            geometry["physx_support"] = support_fixture(geometry["vertices_m"],
                geometry["face_vertex_counts"], geometry["face_vertex_indices"], selected)
            geometry["physx_support"]["physics_step"] = row["physics"]["physics_step"]
        elif fault == "drift" and i == 20:
            selected = list(reversed(support["authored_vertex_indices"]))
            geometry["physx_support"] = support_fixture(geometry["vertices_m"],
                geometry["face_vertex_counts"], geometry["face_vertex_indices"], selected)
            geometry["physx_support"]["physics_step"] = row["physics"]["physics_step"]
        elif fault == "stage_drift" and i == 20:
            geometry["stage_id"] = support["stage_id"] = 8
    assert not audit(rows, expected)["pass"]


@pytest.mark.parametrize("fault", [None, "not_ready", "no_callback", "two_hulls", "outside_authored", "bad_stage", "duplicate"])
def test_snapshot_calls_exact_stage_physx_api_and_rejects_invalid_reply(monkeypatch, fault):
    rows, _ = simplified_trajectory()
    geometry = rows[0]["physics"]["scene_geometry"]["convex_collider"]
    vertices = np.asarray(geometry["vertices_m"])
    selected = geometry["physx_support"]["authored_vertex_indices"]
    returned = vertices[selected].copy()
    if fault == "outside_authored": returned[0, 2] += .001
    if fault == "duplicate": returned[1] = returned[0]
    valid = object()
    requests = []
    def request(stage_id, prim_id, asynchronous, callback):
        requests.append((stage_id, prim_id, asynchronous))
        if fault != "no_callback":
            callback(object() if fault == "not_ready" else valid,
                     [SimpleNamespace(vertices=returned)] * (2 if fault == "two_hulls" else 1))
        return object()
    for name in ("omni", "omni.physx", "omni.physx.bindings", "omni.physx.bindings._physx", "pxr"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["omni"].physx = sys.modules["omni.physx"]
    sys.modules["omni.physx"].get_physx_cooking_interface = lambda: SimpleNamespace(request_convex_collision_representation=request)
    sys.modules["omni.physx.bindings._physx"].PhysxCollisionRepresentationResult = SimpleNamespace(RESULT_VALID=valid)
    prim = SimpleNamespace(GetPath=lambda: "/World_Props/orange/Collision")
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prim)
    class StageCache:
        @staticmethod
        def Get(): return StageCache()
        def GetId(self, actual):
            assert actual is stage
            return SimpleNamespace(ToLongInt=lambda: 0 if fault == "bad_stage" else 7)
    sys.modules["pxr"].UsdUtils = SimpleNamespace(StageCache=StageCache)
    sys.modules["pxr"].PhysicsSchemaTools = SimpleNamespace(sdfPathToInt=lambda path: "id:" + path)
    namespace = {"_obs_np": np, "_obs_hash": hashlib, "_obs_json": json,
        "_obs_SM": SimpleNamespace(get_num_physics_steps=lambda: 0), "stage": stage, "engine": "physx",
        "_tl": SimpleNamespace(is_playing=lambda: True)}
    exec(gpu.convex.support_snapshot_code(), namespace)
    call = lambda: namespace["_convex_support_snapshot"](prim, vertices,
        geometry["face_vertex_counts"], geometry["face_vertex_indices"])
    if fault:
        with pytest.raises(RuntimeError): call()
    else:
        result = call()
        assert result == geometry["physx_support"]
    assert requests == ([] if fault == "bad_stage" else [(7, "id:/World_Props/orange/Collision", False)])
