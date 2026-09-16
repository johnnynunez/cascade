"""Synthetic adversarial fixtures only; these tests are never live acceptance."""
import ast
import base64
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spark_placement", ROOT / "demo/kitchen/physics/spark_proof.py")
proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proof)
gpu = proof.shared
MARKS = {name: {"monotonic": t} for name, t in zip(
    ("pick_begin", "pick_end", "reset_begin", "reset_end"), (99.9, 102.55, 102.59, 104.1))}


def trajectory(object_name="orange", destination_name="open box"):
    expected = gpu.load_expected_scene_geometry(ROOT / "demo/scene/kitchen_config.json")
    config = json.loads((ROOT / "demo/scene/kitchen_config.json").read_text())
    spawns = {**config["cube_positions"], **{p["name"]: p["position"] for p in config["props"]}}
    pad = expected["target_pad"]
    destination = expected["open_box" if destination_name == "open box" else "target_pad"]
    outer, inner = pad["outer_size_m"] / 2, pad["outer_size_m"] / 2 - pad["border_width_m"]
    geometry = {
        "scene_config_sha256": expected["scene_config_sha256"], "prop_dimensions_m": expected["prop_dimensions_m"],
        "cube_colliders": {name: {"dimensions_m": expected["prop_dimensions_m"][name],
            "symmetric_about_origin": True, "no_scale_or_shear": True, "bounding_cube": True,
            "rigid_body": True} for name in ("pink_cube", "green_cube")},
        "target_pad": {"name": "green square", "border_vertices_m": np.concatenate((
            gpu._square_vertices(pad, outer), gpu._square_vertices(pad, inner))).tolist(),
            "fill_vertices_m": gpu._square_vertices(pad, inner).tolist(),
            "visible": True, "visual_only": True, "topology_valid": True, "green_materials": True,
            "support_top_z_m": 0., "support_bounds_xy_m": [config["counter"]["min"][:2], config["counter"]["max"][:2]],
            "support_is_static_collider": True},
        "open_box": {"name": "open box", "parts_bounds_xyz_m": gpu._open_box_part_bounds(expected["open_box"]),
                     "five_static_colliders": True, "axis_aligned": True, "visible": True}}
    if object_name == "orange":
        spec = expected["convex_colliders"][object_name]
        vertices, counts, indices = gpu.convex.expected_hull(spec)
        geometry["convex_collider"] = {"body_name": object_name, "collider_path": "/World_Props/orange/Collision",
            "frame": "body_local", "units": "m", "vertices_m": vertices.tolist(),
            "face_vertex_counts": counts, "face_vertex_indices": indices, "subdivision_none": True,
            "collision_approximation": "convexHull", "collision_enabled": True, "one_collider_one_body": True,
            "dynamic_body": True, "body_no_scale_or_shear": True, "collider_to_body_identity": True,
            "mass_kg": spec["mass_kg"], "com_body_m": [0., 0., 0.], "com_orientation_wxyz": [1., 0., 0., 0.],
            "inertia_kg_m2": (np.eye(3) * 1e-5).tolist(), "mass_properties_channel": "physics_tensor",
            "tensor_devices": ["cuda:0"] * 4}
    records = []
    robot_id = "/SYNTHETIC_TEST_ROBOT"
    jaw_root = robot_id + "/link1/link2/link3/link4/link5/link6/gripper_end"
    for i in range(41):
        t = i * .1
        lifting = 3 <= i < 12
        grip = .025 if lifting else .05
        q = [0.] * 6 + [grip, grip]
        frame = {"available": True, "capture_monotonic": 100 + t - .03, "robot_id": robot_id,
                 "producer_time_source": "physics_loop_monotonic", "jpeg_sha256": f"synthetic-{i}"}
        sample = {"version": 1, "synthetic_test_only": True, "channel": "physics_tensor", "engine": "physx",
            "robot_id": robot_id, "sim_time": t, "physics_step": i * 12, "server_monotonic": 100 + t,
            "q": q, "dq": [0.] * 8, "joint_lower": [-3.] * 6 + [0., 0.], "joint_upper": [3.] * 6 + [.05, .05],
            "gripper": {"indices": [6, 7], "q": q[6:], "open_fractions": [grip / .05] * 2},
            "props": {}, "spawn_positions_m": deepcopy(spawns), "frame": deepcopy(frame),
            "cameras": {camera: deepcopy(frame) for camera in proof.CAMERAS},
            "gpu_attestation": {"required": True, "backend": "physx", "gpu_dynamics": True,
                "broadphase": "GPU", "cuda_context_present": True, "tensor_device_ordinal": 0,
                "device": "cuda:0", "tensor_device": "cuda:0", "cpu_fallback_allowed": False, "fallback_log_count": 0},
            "contacts": {"channel": "physx_gpu_contact_tensor", "device": "cuda:0",
                "sensor_paths": ["/World_Props/" + object_name],
                "filter_paths": [[jaw_root + "/gripper_left", jaw_root + "/gripper_right"]],
                "physics_step": i * 12, "jaw_forces_n": [[.2, 0., 0.], [-.2, 0., 0.]] if lifting else [[0.] * 3] * 2,
                "jaw_contact_counts": [1, 1] if lifting else [0, 0]}, "scene_geometry": deepcopy(geometry)}
        for name, spawn in spawns.items():
            xyz = list(spawn)
            if name == object_name:
                if lifting:
                    xyz = [.20, -.08, .12]
                    if i >= 10:
                        xyz[:2] = destination["center_xy_m"]
                elif 12 <= i < 26:
                    xyz = [*destination["center_xy_m"], destination["support_top_z_m"] + expected["prop_dimensions_m"][name][2] / 2]
            sample["props"][name] = {"position_m": xyz, "orientation_wxyz": [1., 0., 0., 0.],
                "linear_velocity_m_s": [0.] * 3, "angular_velocity_rad_s": [0.] * 3, "tensor_device": "cuda:0"}
        records.append({"sequence": i, "client_started_monotonic": 100 + t,
                        "client_finished_monotonic": 100 + t + .02, "physics": sample})
    return records, expected


def audit(records, expected, object_name="orange", destination_name="open box"):
    return gpu.audit_records(records, marks=MARKS, complete=True, object_name=object_name,
                             destination_name=destination_name, expected_scene_geometry=expected)


@pytest.mark.parametrize("object_name,destination", proof.CASES)
def test_both_cases_keep_contact_destination_release_and_reset_requirements(object_name, destination):
    rows, expected = trajectory(object_name, destination)
    result = audit(rows, expected, object_name, destination)
    assert result["pass"], result
    assert result["checks"]["two_jaw_contacts_during_real_lift"]
    assert result["checks"]["whole_footprint_enters_destination_with_bilateral_support"]
    assert result["checks"]["all_props_physically_reset_and_settled"]
    assert proof.audit_cameras(rows, MARKS)["pass"]


@pytest.mark.parametrize("fault", ["counter_under_box", "outside_wall", "held", "no_contacts", "reset",
                                  "wrong_scene", "cpu", "no_lift", "wrong_hull", "missing_prop"])
def test_orange_box_proof_rejects_incomplete_or_wrong_physical_evidence(fault):
    rows, expected = trajectory()
    for i, row in enumerate(rows):
        sample, prop = row["physics"], row["physics"]["props"]["orange"]
        if fault == "counter_under_box" and 12 <= i < 26:
            prop["position_m"][2] -= .004
        elif fault == "outside_wall" and 12 <= i < 26:
            prop["position_m"][0] += .04  # center passes 40mm; hull crosses cavity wall
        elif fault == "held" and 12 <= i < 26:
            sample["q"][6:] = [.025, .025]
            sample["gripper"].update(q=[.025, .025], open_fractions=[.5, .5])
        elif fault == "no_contacts":
            sample["contacts"]["jaw_contact_counts"][1] = 0
        elif fault == "reset" and i >= 26:
            sample["props"]["green_cube"]["position_m"][0] += .03
        elif fault == "wrong_scene":
            sample["scene_geometry"]["scene_config_sha256"] = "0" * 64
        elif fault == "cpu":
            prop["tensor_device"] = "cpu"
        elif fault == "no_lift":
            prop["position_m"][2] = .02638
        elif fault == "wrong_hull":
            sample["scene_geometry"]["convex_collider"]["vertices_m"][0][0] += .001
        elif fault == "missing_prop":
            sample["props"].pop("lemon")
    assert not audit(rows, expected)["pass"]


@pytest.mark.parametrize("fault", ["stale", "frozen", "foreign_robot", "missing", "reset_frozen"])
def test_each_event_camera_must_advance_in_both_phases(fault):
    rows, _ = trajectory()
    for i, row in enumerate(rows):
        camera = row["physics"]["cameras"]["side"]
        if fault == "stale":
            camera["capture_monotonic"] -= 5
        elif fault == "frozen":
            camera["capture_monotonic"] = 100
        elif fault == "foreign_robot":
            camera["robot_id"] = "/other"
        elif fault == "missing":
            row["physics"]["cameras"].pop("side")
        elif fault == "reset_frozen" and i >= 26:
            camera["capture_monotonic"] = 102.59
    assert not proof.audit_cameras(rows, MARKS)["pass"]


@pytest.mark.parametrize("object_name,destination", proof.CASES)
def test_snapshot_is_read_only_and_keeps_exact_convex_codec(object_name, destination):
    witness = proof.SparkKitchenWitness(ROOT / "runs/unused-synthetic-test",
        scene_config=ROOT / "demo/scene/kitchen_config.json", object_name=object_name, destination_name=destination)
    code = witness._snapshot_code()
    tree = ast.parse(code)
    forbidden = {"update", "play", "stop", "simulate", "step", "Set", "Apply", "reset"}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        assert call.func.attr not in forbidden and not call.func.attr.startswith("set_")
    assert '_gpu_contact_snapshot' in code and '"cameras"' in code and 'KITCHEN_OBSERVER_ZLIB' in code
    rows, _ = trajectory(object_name, destination)
    sample = rows[0]["physics"]
    namespace = {"_gpu_observed": deepcopy(sample), "_obs_payload": json.dumps(sample),
                 "_obs_np": np, "_obs_json": json, "_obs_base64": base64}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec('import zlib as _obs_zlib' + code.split('import zlib as _obs_zlib', 1)[1], namespace)
    assert gpu.original.parse_snapshot_reply({"ok": True, "stdout": output.getvalue()}) == sample
