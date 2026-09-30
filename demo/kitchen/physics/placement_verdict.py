"""Audit a freshly collected placement tail, without claiming a witnessed pick.

This optional checkout-local adapter shares the campaign's geometry and physical
thresholds. It never issues a bridge request, moves the robot, or treats the
campaign's lift, contact-during-lift, reset, or camera checks as established.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu = _load("cascade_placement_gpu_audit", Path(__file__).with_name("gpu_proof_audit.py"))

# Explicit allowlists: adding a campaign check must never silently change the
# meaning of a placement verdict, and a missing required check cannot pass.
EVIDENCE_CHECKS = (
    "physics_channel", "same_robot_and_engine", "same_observed_prop_set",
    "physics_clock_advances", "finite_all_observed_state", "final_window_long_enough",
    "valid_final_orientations", "gripper_readback_matches_actual_q",
)
PLACEMENT_CHECKS = (
    "actual_joints_within_limits", "released_actual_jaws", "destination_xy",
    "supported_z", "settled_linear_speed", "settled_angular_speed", "stable_final_position",
)
METRICS = (
    "final_samples", "final_window_sim_s", "max_final_xy_error_m",
    "max_final_support_z_error_m", "max_final_linear_speed_m_s",
    "max_final_angular_speed_rad_s", "max_final_position_spread_m",
    "min_final_actual_open_fraction",
)
SCOPE = "post-placement containment, support, release and settling only; lift, transport, cameras and reset unverified"


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _validate_state(sample, *, object_name, robot_id, engine):
    """Reject malformed or foreign observations before judging a physical miss."""
    _require(sample.get("version") == 1 and sample.get("channel") == "physics_tensor",
             "missing physics tensor observation")
    _require(sample.get("robot_id") == robot_id and sample.get("engine") == engine,
             "robot or engine identity mismatch")
    q, dq, lo, hi = [np.asarray(sample[key], float) for key in
                     ("q", "dq", "joint_lower", "joint_upper")]
    _require(q.ndim == 1 and len(q) >= 2 and q.shape == dq.shape == lo.shape == hi.shape
             and all(np.isfinite(a).all() for a in (q, dq, lo, hi)) and (hi > lo).all(),
             "malformed actual articulation state")
    gripper = sample["gripper"]
    indices = gripper["indices"]
    _require(isinstance(indices, list) and len(indices) == 2
             and all(type(i) is int and 0 <= i < len(q) for i in indices)
             and indices[0] != indices[1], "malformed actual jaw indices")
    for key in ("q", "open_fractions"):
        values = np.asarray(gripper[key], float)
        _require(values.shape == (2,) and np.isfinite(values).all(), "malformed jaw readback")
    for prop in sample["props"].values():
        for key, width in (("position_m", 3), ("orientation_wxyz", 4),
                           ("linear_velocity_m_s", 3), ("angular_velocity_rad_s", 3)):
            values = np.asarray(prop[key], float)
            _require(values.shape == (width,) and np.isfinite(values).all(),
                     "malformed observed prop state")

    # Same established GPU provenance contract as the campaign. A CPU fallback
    # or unbound contact read is missing evidence, not a physical placement miss.
    attestation = sample["gpu_attestation"]
    ordinal = attestation.get("tensor_device_ordinal")
    device = f"cuda:{ordinal}"
    _require(type(ordinal) is int and ordinal >= 0
             and attestation.get("required") is True and attestation.get("backend") == engine
             and attestation.get("cuda_context_present") is True
             and attestation.get("device") == attestation.get("tensor_device") == device
             and attestation.get("cpu_fallback_allowed") is False
             and attestation.get("fallback_log_count") == 0
             and all(p.get("tensor_device") == device for p in sample["props"].values()),
             "GPU tensor provenance unavailable")
    if engine == "physx":
        backend_ok = attestation.get("gpu_dynamics") is True and attestation.get("broadphase") == "GPU"
    else:
        newton = attestation.get("newton") or {}
        backend_ok = (newton.get("cuda") is True and newton.get("mujoco_cpu") is False
                      and newton.get("solver") == "SolverMuJoCo" and newton.get("array_devices") == [device])
    _require(backend_ok, "GPU backend attestation unavailable")
    contact = sample["contacts"]
    jaw_root = robot_id + "/link1/link2/link3/link4/link5/link6/gripper_end"
    _require(contact.get("sensor_paths") == ["/World_Props/" + object_name]
             and contact.get("filter_paths") == [[jaw_root + "/gripper_left", jaw_root + "/gripper_right"]]
             and contact.get("channel") == gpu.CONTACT_CHANNELS[engine]
             and contact.get("device") == device
             and type(contact.get("physics_step")) is int
             and contact["physics_step"] == sample["physics_step"],
             "contact observation is stale or belongs to another body, robot or engine")
    forces = np.asarray(contact["jaw_forces_n"], float)
    counts = contact["jaw_contact_counts"]
    _require(forces.shape == (2, 3) and np.isfinite(forces).all()
             and isinstance(counts, list) and len(counts) == 2
             and all(type(n) is int and n >= 0 for n in counts), "malformed contact readback")


def audit_placement(records, *, object_name, destination_name, expected_scene_geometry, robot_id):
    """Return confirmed/refuted/unverified for a freshly acquired placement tail.

    The caller owns freshness relative to the completed command. This function
    requires advancing, ordered physics and contacts bound to those exact steps.
    Missing/invalid evidence is UNVERIFIED; a valid measured physical miss is
    REFUTED. No historical campaign receipt or actuator success flag is accepted.
    """
    result = {"status": "unverified", "evidence": "placement evidence unavailable",
              "measured": {"scope": SCOPE}}
    measured = result["measured"]
    try:
        object_name = gpu.validate_object_name(object_name)
        destination_name = gpu.validate_destination_name(destination_name)
        expected = gpu.validate_expected_scene_geometry(expected_scene_geometry)
        _require(isinstance(robot_id, str) and robot_id.startswith("/") and len(robot_id) > 1,
                 "explicit intended robot identity is required")
        _require(isinstance(records, (list, tuple)) and len(records) >= 4,
                 "at least four fresh physics samples are required")
        samples = [r.get("physics", r) for r in records]
        _require(all(isinstance(s, dict) for s in samples), "malformed physics records")
        engine = samples[0].get("engine")
        _require(engine in gpu.CONTACT_CHANNELS, "unsupported physics engine")
        _require(object_name in expected["prop_dimensions_m"], "object absent from expected scene")
        inventory = gpu.audit_prop_inventory(samples, tuple(expected["prop_dimensions_m"]))
        _require(all(inventory.values()), "observed props or spawns differ from the expected scene")
        clocks = np.asarray([s["sim_time"] for s in samples], float)
        steps = [s["physics_step"] for s in samples]
        _require(np.isfinite(clocks).all() and (clocks >= 0).all()
                 and all(type(step) is int and step >= 0 for step in steps)
                 and (np.diff(clocks) >= 0).all() and (np.diff(steps) >= 0).all()
                 and np.array_equal(np.diff(clocks) > 0, np.diff(steps) > 0),
                 "physics clocks or steps are malformed, unordered or inconsistent")
        start = max(0, int(np.searchsorted(clocks, clocks[-1] - .5, side="right")) - 1)
        tail = samples[start:]
        duration = float(clocks[-1] - clocks[start])
        max_gap = float(np.diff(clocks[start:]).max()) if len(tail) > 1 else 0.
        _require(len(tail) >= 4 and len(set(steps[start:])) >= 4 and duration >= .5 - 1e-9
                 and max_gap <= .25, "incomplete or sparsely sampled 0.5-second settling window")
        measured.update(object=object_name, destination=destination_name, robot_id=robot_id,
                        engine=engine, final_samples=len(tail), final_window_sim_s=duration,
                        max_sample_gap_sim_s=max_gap)
        for sample in samples:
            _validate_state(sample, object_name=object_name, robot_id=robot_id, engine=engine)
        scene = gpu.audit_scene_geometry(samples, expected)
        _require(scene["pass"], "live scene geometry or identity does not match the reviewed configuration")
        measured["scene_config_sha256"] = expected["scene_config_sha256"]

        destination_key = "open_box" if destination_name == "open box" else "target_pad"
        destination = expected[destination_key]
        if destination_name == "open box":
            parts = samples[0]["scene_geometry"]["open_box"]["parts_bounds_xyz_m"]
            bounds = gpu._box_inner_bounds(parts)
            support_z = parts["floor"][1][2]
            support_tolerance = min(.001, (support_z - parts["floor"][0][2]) / 4)
        else:
            center = np.asarray(destination["center_xy_m"], float)
            half = destination["outer_size_m"] / 2 - destination["border_width_m"]
            bounds = [(center - half).tolist(), (center + half).tolist()]
            support_z = samples[0]["scene_geometry"]["target_pad"]["support_top_z_m"]
            support_tolerance = .005

        support_geometry = None
        if object_name in gpu.convex.CONVEX_TARGETS:
            binding = gpu.convex.audit_binding(samples, object_name=object_name, expected=expected)
            _require(binding["pass"], "authored collider is not bound to the expected object")
            support_binding = gpu.convex.audit_support_binding(samples, object_name=object_name)
            _require(support_binding["pass"], "live support collider binding unavailable")
            support_geometry = gpu.strict.VerifiedColliderGeometry(
                vertices_m=support_binding["vertices_m"], body_name=object_name,
                receipt={"method": "live engine collision representation", "half_height_argument_used": False})
        dimensions = expected["prop_dimensions_m"][object_name]
        # Cameras are outside this verdict. Suppress camera processing entirely
        # so absent/malformed image metadata cannot alter valid physical evidence.
        strict = gpu.strict.audit_records([{**s, "frame": {}} for s in samples],
            object_name=object_name, target_xy=destination["center_xy_m"], support_top_z=support_z,
            object_half_height=dimensions[2] / 2, collider_geometry=support_geometry)
        _require("error" not in strict, "malformed physical settling evidence")
        checks = strict["checks"]
        _require(all(checks.get(name) is True for name in EVIDENCE_CHECKS),
                 "physical observation or actual jaw readback is incomplete or inconsistent")
        physical = {name: checks[name] for name in PLACEMENT_CHECKS}
        physical["support_geometry"] = checks["projected_support_geometry" if support_geometry else "upright_support_geometry"]
        measured.update({name: strict["metrics"][name] for name in METRICS})
        rows = [{"physics": s} for s in samples]
        if support_geometry is not None:
            footprint = gpu.convex_settle.audit_convex_settle_geometry(rows, object_name=object_name,
                vertices_body_m=samples[0]["scene_geometry"]["convex_collider"]["vertices_m"],
                support_vertices_body_m=support_binding["vertices_m"], inner_bounds_xy_m=bounds,
                support_top_z_m=support_z, max_support_gap_m=support_tolerance,
                max_penetration_m=support_tolerance)
            physical["whole_collider_contained_and_supported"] = footprint["geometry_pass"]
            measured.update(min_inside_clearance_m=footprint["min_inside_clearance_m"],
                            max_abs_support_gap_m=footprint["max_abs_support_gap_m"])
            if object_name == "tomato_can":
                quats = np.asarray([s["props"][object_name]["orientation_wxyz"] for s in tail], float)
                quats /= np.linalg.norm(quats, axis=1)[:, None]
                physical["tomato_can_settles_upright"] = bool((1 - 2 * (quats[:, 1]**2 + quats[:, 2]**2)
                                                             >= np.cos(np.deg2rad(5))).all())
        else:
            footprint = gpu.audit_cube_footprint(rows, object_name=object_name, dimensions_m=dimensions,
                pad=destination, destination_name=destination_name, inner_bounds_xy_m=bounds)
            low = footprint["min_final_bottom_z_m"] - support_z
            high = footprint["max_final_bottom_z_m"] - support_z
            physical["whole_collider_contained"] = footprint["pass"]
            physical["lowest_collider_point_supported"] = bool(low >= -support_tolerance and high <= support_tolerance)
            measured.update(min_inside_clearance_m=footprint["min_inner_clearance_m"],
                            max_abs_support_gap_m=max(abs(low), abs(high)))
        measured.update(support_tolerance_m=support_tolerance, boundary_outward_tolerance_m=0.,
                        checks=physical, failed_checks=[name for name, passed in physical.items() if not passed])
        if all(physical.values()):
            result.update(status="confirmed", evidence=(f"{object_name} is fully contained in {destination_name}, "
                "supported, released by the actual jaws and settled across the observed window; the pick/lift was not observed"))
        else:
            result.update(status="refuted", evidence="measured placement failed: " + ", ".join(measured["failed_checks"]))
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, OverflowError, np.linalg.LinAlgError) as exc:
        result["evidence"] = "placement unverified: " + str(exc)
    return result
