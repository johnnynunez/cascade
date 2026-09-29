"""Offline geometric subcheck for an independently verified convex collider.

This module does not load evidence seals, verify live geometry, infer a collider
from dimensions, or grant a campaign pass. The caller must do those bindings and
retain CUDA/contact/lift/release/camera/reset/source checks independently.
"""
from __future__ import annotations

import numpy as np


def audit_convex_settle_geometry(samples, *, object_name, vertices_body_m,
                                inner_bounds_xy_m, support_top_z_m,
                                support_vertices_body_m=None,
                                settle_sim_s=.5, min_final_samples=4,
                                max_sample_gap_sim_s=.25,
                                min_inside_margin_m=0.,
                                max_support_gap_m=.005, max_penetration_m=.005):
    """Check all recorded tail poses using actual body-local convex vertices.

    Vertices must come from the reviewed active /Collision geometry after its
    child-to-body transform is baked. Positions are the measured BODY origin,
    not the COM; orientations are unit wxyz quaternions in the same metre frame
    as the measured square/counter. An off-centre local collider is supported.

    A linear coordinate functional reaches each hull extremum at a vertex.
    Thus all transformed vertices inside a convex square imply the complete
    authored hull footprint is inside. The minimum world Z is an actual hull
    support vertex, rather than an artificial corner of its bounding box.
    A separately bound collision representation may supply support vertices;
    the complete authored vertices still determine every XY clearance.
    No unsampled trajectory or contact-force claim is made.
    """
    if object_name not in {"tomato_can", "lemon", "orange"}:
        raise ValueError("Convex settling supports tomato_can, lemon and configured orange only")
    vertices = np.asarray(vertices_body_m, dtype=float)
    bounds = np.asarray(inner_bounds_xy_m, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not 4 <= len(vertices) <= 10000
            or not np.isfinite(vertices).all() or np.abs(vertices).max() > 10
            or np.linalg.matrix_rank(vertices - vertices[0]) < 3):
        raise ValueError("Expected finite solid convex-collider body-local vertices")
    support_vertices = vertices if support_vertices_body_m is None else np.asarray(support_vertices_body_m, float)
    if (support_vertices.ndim != 2 or support_vertices.shape[1:] != (3,)
            or not 4 <= len(support_vertices) <= 10000
            or not np.isfinite(support_vertices).all() or np.abs(support_vertices).max() > 10
            or np.linalg.matrix_rank(support_vertices-support_vertices[0]) < 3):
        raise ValueError("Expected separately verified solid body-local support vertices")
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or not (bounds[1] > bounds[0]).all()):
        raise ValueError("Expected measured nonempty axis-aligned square inner bounds")
    criteria = [support_top_z_m, settle_sim_s, max_sample_gap_sim_s,
                min_inside_margin_m, max_support_gap_m, max_penetration_m]
    if (not np.isfinite(criteria).all() or not 0 < settle_sim_s <= 10
            or not 0 < max_sample_gap_sim_s <= settle_sim_s
            or type(min_final_samples) is not int or min_final_samples < 4
            or min_inside_margin_m < 0 or not 0 <= max_support_gap_m <= .005
            or not 0 <= max_penetration_m <= .005):
        raise ValueError("Invalid settle, margin or support criteria")
    physics = [sample.get("physics", sample) for sample in samples]
    if not physics:
        raise ValueError("No physics poses supplied")
    clocks = np.asarray([s["sim_time"] for s in physics], dtype=float)
    steps_raw = [s["physics_step"] for s in physics]
    if any(type(step) is not int or step < 0 for step in steps_raw):
        raise ValueError("Physics steps must be nonnegative integers")
    steps = np.asarray(steps_raw, dtype=np.int64)
    if (not np.isfinite(clocks).all() or (np.diff(clocks) < 0).any()
            or (np.diff(steps) < 0).any()):
        raise ValueError("Physics clocks and steps must be finite and ordered")
    poses = np.asarray([s["props"][object_name]["position_m"] for s in physics], dtype=float)
    quats = np.asarray([s["props"][object_name]["orientation_wxyz"] for s in physics], dtype=float)
    if (poses.shape != (len(physics), 3) or quats.shape != (len(physics), 4)
            or not np.isfinite(poses).all() or not np.isfinite(quats).all()):
        raise ValueError("Expected finite measured body positions and wxyz quaternions")
    norms = np.linalg.norm(quats, axis=1)
    if not (np.abs(norms - 1) < 1e-3).all():
        raise ValueError("Expected unit wxyz quaternions")
    quats = quats / norms[:, None]
    # Include the sample immediately preceding the requested trailing window.
    # Do not hide an escape by checking only the final pose or a short tail.
    start = max(0, int(np.searchsorted(clocks, clocks[-1] - settle_sim_s, side="right")) - 1)
    indices = range(start, len(physics))
    tail_clocks, tail_steps = clocks[start:], steps[start:]
    duration = float(tail_clocks[-1] - tail_clocks[0])
    complete = bool(len(tail_clocks) >= min_final_samples
                    and len(set(tail_steps)) >= min_final_samples
                    and duration >= settle_sim_s - 1e-9)
    max_gap = float(np.diff(tail_clocks).max()) if len(tail_clocks) > 1 else 0.
    checks = {"settle_window_complete": complete,
              "bounded_sample_gaps": max_gap <= max_sample_gap_sim_s,
              "whole_collider_inside_square_at_every_recorded_pose": True,
              "actual_lowest_vertex_supported_at_every_recorded_pose": True}
    receipts = []
    labels = ("left", "front", "right", "back")
    for index in indices:
        w, x, y, z = quats[index]
        rotation = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                               [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                               [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        world = vertices @ rotation.T + poses[index]
        clearances = np.column_stack((world[:, :2] - bounds[0], bounds[1] - world[:, :2]))
        vertex, boundary = np.unravel_index(int(np.argmin(clearances)), clearances.shape)
        clearance = float(clearances[vertex, boundary])
        support_world = support_vertices @ rotation.T + poses[index]
        support = int(np.argmin(support_world[:, 2]))
        bottom = float(support_world[support, 2])
        gap = bottom - support_top_z_m
        inside = clearance >= min_inside_margin_m  # no outward tolerance
        supported = -max_penetration_m <= gap <= max_support_gap_m
        checks["whole_collider_inside_square_at_every_recorded_pose"] &= inside
        checks["actual_lowest_vertex_supported_at_every_recorded_pose"] &= supported
        receipts.append({"sample_index": index, "physics_step": int(steps[index]),
                         "sim_time": float(clocks[index]), "min_inside_clearance_m": clearance,
                         "worst_boundary": labels[boundary], "worst_boundary_vertex_index": int(vertex),
                         "footprint_bounds_xy_m": [world[:, :2].min(axis=0).tolist(), world[:, :2].max(axis=0).tolist()],
                         "support_vertex_index": support, "support_vertex_body_m": support_vertices[support].tolist(),
                         "support_vertex_world_m": support_world[support].tolist(),
                         "lowest_vertex_world_z_m": bottom, "support_gap_m": float(gap),
                         "inside": bool(inside), "supported": bool(supported)})
    return {"geometry_pass": bool(all(checks.values())), "checks": checks,
            "scope": "geometry only, at recorded settle poses; not a campaign pass",
            "geometry_source_verified_by_this_function": False,
            "method": "all authored vertices for XY; minimum transformed support vertex for Z",
            "object_name": object_name, "vertex_count": len(vertices),
            "support_vertex_count": len(support_vertices),
            "inner_bounds_xy_m": bounds.tolist(), "support_top_z_m": float(support_top_z_m),
            "settle_window_sim_s": duration, "final_samples": len(receipts),
            "max_sample_gap_sim_s": max_gap, "min_inside_margin_m": min_inside_margin_m,
            "boundary_outward_tolerance_m": 0., "max_support_gap_m": max_support_gap_m,
            "max_penetration_m": max_penetration_m,
            "min_inside_clearance_m": min(r["min_inside_clearance_m"] for r in receipts),
            "max_abs_support_gap_m": max(abs(r["support_gap_m"]) for r in receipts),
            "samples": receipts}
