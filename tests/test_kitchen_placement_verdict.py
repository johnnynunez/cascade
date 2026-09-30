"""Synthetic placement tails; these tests never establish live robot acceptance."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from test_spark_placement_proof import support_fixture, trajectory
from test_spark_placement_proof_newton import as_newton


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kitchen_placement_verdict", ROOT / "demo/kitchen/physics/placement_verdict.py")
placement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(placement)
ROBOT_ID = "/SYNTHETIC_TEST_ROBOT"


def settled(object_name="orange", destination="open box", engine="physx"):
    rows, expected = trajectory(object_name, destination)
    if object_name in ("lemon", "tomato_can"):
        # The shared campaign fixture authors orange's convex identity. Extend
        # that same explicit synthetic geometry to the other configured props.
        source, _ = trajectory("orange", destination)
        template = source[0]["physics"]["scene_geometry"]["convex_collider"]
        spec = expected["convex_colliders"][object_name]
        vertices, counts, indices = placement.gpu.convex.expected_hull(spec)
        for row in rows:
            geometry = deepcopy(template)
            geometry.update(body_name=object_name, collider_path=f"/World_Props/{object_name}/Collision",
                            vertices_m=vertices.tolist(), face_vertex_counts=counts,
                            face_vertex_indices=indices, mass_kg=spec["mass_kg"])
            geometry["physx_support"] = support_fixture(vertices, counts, indices)
            geometry["physx_support"].update(collider_path=geometry["collider_path"],
                                             physics_step=row["physics"]["physics_step"])
            row["physics"]["scene_geometry"]["convex_collider"] = geometry
    rows = rows[12:26]  # Only the final released placement; no pick/lift/reset.
    if engine == "newton":
        rows = as_newton(rows)
    return rows, expected


def audit(rows, expected, object_name="orange", destination="open box", robot_id=ROBOT_ID):
    result = placement.audit_placement(rows, object_name=object_name, destination_name=destination,
                                     expected_scene_geometry=expected, robot_id=robot_id)
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize("engine", ["physx", "newton"])
@pytest.mark.parametrize("object_name,destination", [
    ("orange", "open box"), ("orange", "green square"),
    ("pink_cube", "green square"), ("green_cube", "open box"),
    ("lemon", "open box"), ("tomato_can", "green square"),
])
def test_confirmed_tail_does_not_claim_campaign_lift_reset_or_cameras(engine, object_name, destination):
    rows, expected = settled(object_name, destination, engine)
    for row in rows:
        row["physics"].pop("frame")
        row["physics"].pop("cameras")
    result = audit(rows, expected, object_name, destination)
    assert result["status"] == "confirmed", result
    assert "pick/lift was not observed" in result["evidence"]
    assert "lift, transport, cameras and reset unverified" in result["measured"]["scope"]
    assert result["measured"]["boundary_outward_tolerance_m"] == 0
    assert result["measured"]["support_tolerance_m"] == (.001 if destination == "open box" else .005)
    assert not {"observed_lift", "observed_xy_displacement", "camera_pass", "reset", "pass"}.intersection(result)
    assert not {"observed_lift", "observed_xy_displacement"}.intersection(result["measured"]["checks"])


@pytest.mark.parametrize("object_name,destination", [
    ("orange", "open box"), ("pink_cube", "green square"), ("green_cube", "open box")])
def test_center_inside_cannot_hide_a_collider_crossing_the_boundary(object_name, destination):
    rows, expected = settled(object_name, destination)
    target = expected["open_box" if destination == "open box" else "target_pad"]
    if destination == "open box":
        bounds = placement.gpu._box_inner_bounds(rows[0]["physics"]["scene_geometry"]["open_box"]["parts_bounds_xyz_m"])
        half_width = bounds[1][0] - target["center_xy_m"][0]
    else:
        half_width = target["outer_size_m"] / 2 - target["border_width_m"]
    offset = half_width - expected["prop_dimensions_m"][object_name][0] / 2 + .001
    assert offset < half_width  # Center is inside while a collider edge is not.
    for row in rows:
        row["physics"]["props"][object_name]["position_m"][0] += offset
    result = audit(rows, expected, object_name, destination)
    assert result["status"] == "refuted", result
    if offset <= .04:
        assert result["measured"]["checks"]["destination_xy"]
    assert result["measured"]["min_inside_clearance_m"] < 0


@pytest.mark.parametrize("object_name", ["orange", "green_cube"])
def test_counter_under_thin_box_floor_is_refuted_even_when_legacy_height_passes(object_name):
    rows, expected = settled(object_name)
    for row in rows:
        row["physics"]["props"][object_name]["position_m"][2] -= .004
    result = audit(rows, expected, object_name)
    assert result["status"] == "refuted", result
    assert result["measured"]["checks"]["supported_z"]
    assert result["measured"]["max_abs_support_gap_m"] > .0039


@pytest.mark.parametrize("fault", ["held", "linear_speed", "angular_speed", "spread", "unsupported", "outside"])
def test_valid_measurement_of_wrong_physical_state_is_refuted(fault):
    rows, expected = settled()
    for i, row in enumerate(rows):
        sample = row["physics"]
        prop = sample["props"]["orange"]
        if fault == "held":
            sample["q"][6:] = [.025, .025]
            sample["gripper"].update(q=[.025, .025], open_fractions=[.5, .5])
        elif fault == "linear_speed":
            prop["linear_velocity_m_s"] = [.020001, 0., 0.]
        elif fault == "angular_speed":
            prop["angular_velocity_rad_s"] = [0., 0., .200001]
        elif fault == "spread" and i == len(rows) - 2:
            prop["position_m"][0] += .00501
        elif fault == "unsupported":
            prop["position_m"][2] += .01
        elif fault == "outside":
            prop["position_m"][0] += .2
    result = audit(rows, expected)
    assert result["status"] == "refuted", result
    assert result["measured"]["failed_checks"]


def test_valid_last_pose_does_not_erase_failure_earlier_in_settle_window():
    rows, expected = settled()
    rows[-4]["physics"]["props"]["orange"]["position_m"][0] += .05
    assert audit(rows, expected)["status"] == "refuted"


def test_inverted_can_is_refuted_even_when_containment_and_support_pass():
    rows, expected = settled("tomato_can", "green square")
    for row in rows:
        row["physics"]["props"]["tomato_can"]["orientation_wxyz"] = [0., 1., 0., 0.]
    result = audit(rows, expected, "tomato_can", "green square")
    assert result["status"] == "refuted", result
    assert result["measured"]["checks"]["whole_collider_contained_and_supported"]
    assert not result["measured"]["checks"]["tomato_can_settles_upright"]


@pytest.mark.parametrize("fault", [
    "short", "too_few", "frozen", "gap", "backwards", "fractional_step", "clock_step_mismatch",
    "wrong_robot", "wrong_engine", "scene", "hull", "support", "prop_missing", "nan_pose", "pose_shape",
    "jaw_index", "jaw_mismatch", "jaw_limits", "contact_step", "contact_body", "contact_jaw",
    "contact_engine", "contact_shape", "contact_nan", "contact_count", "cpu", "gpu_backend", "quat",
])
def test_missing_malformed_or_foreign_evidence_is_unverified(fault):
    rows, expected = settled()
    sample = rows[-1]["physics"]
    if fault == "short": rows = rows[-4:]
    elif fault == "too_few": rows = rows[-3:]
    elif fault == "frozen":
        for row in rows:
            row["physics"].update(sim_time=1., physics_step=12)
    elif fault == "gap": rows = rows[-8:-5] + rows[-1:]
    elif fault == "backwards": rows[-2], rows[-1] = rows[-1], rows[-2]
    elif fault == "fractional_step": sample["physics_step"] += .5
    elif fault == "clock_step_mismatch": sample["physics_step"] = rows[-2]["physics"]["physics_step"]
    elif fault == "wrong_robot": sample["robot_id"] = "/another_robot"
    elif fault == "wrong_engine": sample["engine"] = "newton"
    elif fault == "scene": sample["scene_geometry"]["scene_config_sha256"] = "0" * 64
    elif fault == "hull": sample["scene_geometry"]["convex_collider"]["vertices_m"][0][0] += .001
    elif fault == "support": sample["scene_geometry"]["convex_collider"]["physx_support"]["physics_step"] -= 1
    elif fault == "prop_missing": sample["props"].pop("lemon")
    elif fault == "nan_pose": sample["props"]["orange"]["position_m"][0] = float("nan")
    elif fault == "pose_shape": sample["props"]["orange"]["position_m"].append(0.)
    elif fault == "jaw_index": sample["gripper"]["indices"][0] = 6.5
    elif fault == "jaw_mismatch": sample["gripper"]["q"][0] = .025
    elif fault == "jaw_limits": sample["joint_upper"][6] = sample["joint_lower"][6]
    elif fault == "contact_step": sample["contacts"]["physics_step"] -= 1
    elif fault == "contact_body": sample["contacts"]["sensor_paths"] = ["/World_Props/lemon"]
    elif fault == "contact_jaw": sample["contacts"]["filter_paths"][0][0] = "/other_robot/gripper_left"
    elif fault == "contact_engine": sample["contacts"]["channel"] = "newton_mjwarp_contact_force"
    elif fault == "contact_shape": sample["contacts"]["jaw_forces_n"] = [0., 0.]
    elif fault == "contact_nan": sample["contacts"]["jaw_forces_n"][0][0] = float("nan")
    elif fault == "contact_count": sample["contacts"]["jaw_contact_counts"][0] = -.5
    elif fault == "cpu": sample["props"]["orange"]["tensor_device"] = "cpu"
    elif fault == "gpu_backend": sample["gpu_attestation"]["gpu_dynamics"] = False
    elif fault == "quat": sample["props"]["orange"]["orientation_wxyz"] = [0., 0., 0., 0.]
    result = audit(rows, expected)
    assert result["status"] == "unverified", (fault, result)


@pytest.mark.parametrize("robot_id", [None, "", "robot", "/another_robot"])
def test_robot_identity_must_be_supplied_independently(robot_id):
    rows, expected = settled()
    assert audit(rows, expected, robot_id=robot_id)["status"] == "unverified"


def test_exact_release_and_motion_thresholds_remain_unchanged():
    rows, expected = settled()
    for row in rows:
        sample = row["physics"]
        sample["q"][6:] = [.045, .045]
        sample["gripper"].update(q=[.045, .045], open_fractions=[.9, .9])
        sample["props"]["orange"]["linear_velocity_m_s"] = [.02, 0., 0.]
        sample["props"]["orange"]["angular_velocity_rad_s"] = [.2, 0., 0.]
    # Binary division .045 / .05 is slightly below .9: choose the next float
    # above the exact joint threshold, without changing the verifier threshold.
    for row in rows:
        row["physics"]["q"][6:] = [float(np.nextafter(.045, 1))] * 2
        row["physics"]["gripper"]["q"] = row["physics"]["q"][6:]
    assert audit(rows, expected)["status"] == "confirmed"
