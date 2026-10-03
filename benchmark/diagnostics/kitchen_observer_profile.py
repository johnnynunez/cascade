"""Bounded wall-time zones around the unchanged passive kitchen observer.

This instruments generated snapshot Python, not a controller or validator.
Original read calls, return values, exceptions and stdout are preserved. Timings
include synchronization/waiting; they are not GPU kernel or CPU self-time proof.
The owned bridge retains small aggregates, fetched in a separate exec request.
"""
from __future__ import annotations

import ast
import hashlib


_FUNCTION_ZONES = {
    "_obs_collect": "state_collection",
    "_obs_array": "array_readback",
    "physics_device_identity": "gpu_identity",
    "_gpu_contact_snapshot": "contacts",
    "_gpu_scene_geometry_snapshot": "scene_geometry",
    "_convex_live_snapshot": "convex_geometry",
    "_convex_support_snapshot": "convex_support",
}
_METHOD_ZONES = {
    "get_dof_positions": "articulation_positions",
    "get_dof_velocities": "articulation_velocities",
    "get_dof_limits": "articulation_limits",
    "get_world_poses": "rigid_poses",
    "get_velocities": "rigid_velocities",
    "is_physics_tensor_entity_valid": "view_validity",
    "get_masses": "mass_properties",
    "get_coms": "mass_properties",
    "get_inertias": "mass_properties",
    "request_convex_collision_representation": "physx_convex_representation",
    "GetLocalToWorldTransform": "usd_transform",
    "ComputeBoundMaterial": "usd_material",
}
_QUALIFIED_ZONES = {
    ("_obs_zlib", "compress"): "compression",
    ("_obs_json", "dumps"): "json_encoding",
    ("_obs_base64", "b64decode"): "frame_decode",
    ("_obs_hash", "sha256"): "hashing",
}


class _Calls(ast.NodeTransformer):
    def __init__(self):
        self.counts = {}

    def visit_Call(self, node):
        self.generic_visit(node)
        zone = None
        if isinstance(node.func, ast.Name):
            zone = _FUNCTION_ZONES.get(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            zone = _METHOD_ZONES.get(node.func.attr)
            if isinstance(node.func.value, ast.Name):
                zone = _QUALIFIED_ZONES.get((node.func.value.id, node.func.attr), zone)
        if zone is None:
            return node
        self.counts[zone] = self.counts.get(zone, 0) + 1
        return ast.copy_location(ast.Call(func=ast.Name(id="_observer_profile_call", ctx=ast.Load()),
            args=[ast.Constant(zone), node.func, *node.args], keywords=node.keywords), node)


_PREFIX = '''
import time as _observer_profile_time
import threading as _observer_profile_threading
_observer_profiles = globals().setdefault("_cascade_observer_profiles_v1", {})
if PROFILE_ID not in _observer_profiles:
    if len(_observer_profiles) >= 4:
        raise RuntimeError("observer profile inventory exhausted")
    _observer_profiles[PROFILE_ID] = {"source_sha256": PROFILE_ID, "samples": 0,
        "thread_id": _observer_profile_threading.get_ident(), "stack": [], "zones": {}}
_observer_profile = _observer_profiles[PROFILE_ID]
if _observer_profile["thread_id"] != _observer_profile_threading.get_ident():
    raise RuntimeError("observer profiling moved off its original native thread")
if _observer_profile["stack"] or _observer_profile["samples"] >= MAX_SAMPLES:
    raise RuntimeError("observer profile pending or sample budget exhausted")
_observer_profile["samples"] += 1
class _ObserverProfileZone:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.node = [_observer_profile_time.perf_counter_ns(), 0]
        _observer_profile["stack"].append(self.node)
    def __exit__(self, kind, value, traceback):
        elapsed = _observer_profile_time.perf_counter_ns() - self.node[0]
        assert _observer_profile["stack"].pop() is self.node
        if _observer_profile["stack"]:
            _observer_profile["stack"][-1][1] += elapsed
        row = _observer_profile["zones"].setdefault(self.name,
            {"count": 0, "inclusive_ns": 0, "exclusive_ns": 0,
             "min_ns": elapsed, "max_ns": 0, "exceptions": 0})
        row["count"] += 1
        row["inclusive_ns"] += elapsed
        row["exclusive_ns"] += elapsed - self.node[1]
        row["min_ns"] = min(row["min_ns"], elapsed)
        row["max_ns"] = max(row["max_ns"], elapsed)
        row["exceptions"] += int(kind is not None)
        return False
def _observer_profile_call(name, function, /, *args, **kwargs):
    with _ObserverProfileZone(name):
        return function(*args, **kwargs)
'''


def instrument_snapshot(source: str, *, max_samples: int = 256) -> tuple[str, dict]:
    """Return a diagnostic variant and its exact original-source binding."""
    if type(source) is not str or not source or "_observer_profile" in source:
        raise ValueError("expected one original snapshot source")
    if type(max_samples) is not int or not 1 <= max_samples <= 1024:
        raise ValueError("bounded profile sample count required")
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    tree = ast.parse(source)
    if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(tree)):
        raise ValueError("snapshot with scope declarations requires a new review")
    calls = _Calls()
    tree = calls.visit(tree)
    if not calls.counts:
        raise ValueError("no recognized observer calls")
    tree.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id="_ObserverProfileZone", ctx=ast.Load()),
        args=[ast.Constant("snapshot_total")], keywords=[]))], body=tree.body)]
    ast.fix_missing_locations(tree)
    prefix = _PREFIX.replace("PROFILE_ID", repr(original_hash)).replace("MAX_SAMPLES", str(max_samples))
    result = prefix + "\n" + ast.unparse(tree) + "\n"
    return result, {"source_sha256": original_hash,
                    "instrumented_sha256": hashlib.sha256(result.encode()).hexdigest(),
                    "max_samples": max_samples, "call_sites": calls.counts,
                    "units": "wall_nanoseconds", "gpu_kernel_attribution": False}


def read_profiles_code() -> str:
    """Separate bounded wire message: never enlarges the original snapshot."""
    return ('import json, zlib, base64\n'
            'print("KITCHEN_PROFILE_ZLIB " + base64.b64encode(zlib.compress(json.dumps('
            'globals().get("_cascade_observer_profiles_v1", {}), allow_nan=False).encode(), 6)).decode())\n')
