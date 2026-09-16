"""Passive GPU trajectory/contact evidence around the unchanged native-tool proof.

This module never issues actuator commands. It instruments only proof phase
boundaries and reuses the prior strict physical destination auditor unchanged.
"""
from __future__ import annotations
from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = _load("cascade_gpu_existing_observer", ROOT / "demo/kitchen/observer/observer.py")
strict = _load("cascade_gpu_existing_audit", ROOT / "demo/kitchen/observer/audit.py")
convex = _load("cascade_convex_geometry", Path(__file__).with_name("convex_geometry.py"))
convex_settle = _load("cascade_convex_settle", Path(__file__).with_name("convex_settle.py"))
codec = _load("cascade_physics_codec", Path(__file__).with_name("snapshot_codec.py"))
destination_entry = _load("cascade_destination_entry", Path(__file__).with_name("destination_entry.py"))


def validate_object_name(object_name):
    if not isinstance(object_name, str) or object_name not in original.ALLOWED_PROPS:
        raise ValueError("Expected a named kitchen prop: " + ", ".join(original.ALLOWED_PROPS))
    return object_name


def validate_expected_props(props):
    """Only the original four-prop scene or its explicit orange extension is valid."""
    if (not isinstance(props, (list, tuple))
            or any(not isinstance(name, str) for name in props)
            or len(set(props)) != len(props)
            or set(props) not in (set(original.PROPS), set(original.ALLOWED_PROPS))):
        raise ValueError("Expected the four base kitchen props, optionally plus orange, exactly once")
    return tuple(name for name in original.ALLOWED_PROPS if name in props)


def validate_destination_name(destination_name):
    if not isinstance(destination_name, str) or destination_name not in ("green square", "open box"):
        raise ValueError("Expected destination green square or open box")
    return destination_name


def validate_expected_scene_geometry(value):
    """Copy the independently supplied scene identity and bounded metric geometry."""
    if not isinstance(value, dict):
        raise ValueError("Expected scene geometry must be a dictionary")
    digest = value.get("scene_config_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Expected scene geometry requires a SHA-256 scene identity")

    def vector(values, length, *, positive=False):
        if (not isinstance(values, (list, tuple)) or len(values) != length
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)
                or any(abs(v) > 10 or (positive and v <= 0) for v in values)):
            raise ValueError("Scene geometry requires bounded finite metre values")
        return [float(v) for v in values]

    dimensions = value.get("prop_dimensions_m")
    if not isinstance(dimensions, dict):
        raise ValueError("Scene geometry must include the actual prop dimensions")
    expected_props = validate_expected_props(tuple(dimensions))
    dimensions = {name: vector(dimensions[name], 3, positive=True) for name in expected_props}
    pad = value.get("target_pad")
    if not isinstance(pad, dict) or pad.get("name") != "green square":
        raise ValueError("Expected physical destination must be named green square")
    normalized_pad = {"name": "green square", "center_xy_m": vector(pad.get("center_xy_m"), 2)}
    for key in ("outer_size_m", "border_width_m", "surface_z_m", "support_top_z_m"):
        normalized_pad[key] = vector([pad.get(key)], 1)[0]
    if (not 0 < normalized_pad["border_width_m"] < normalized_pad["outer_size_m"] / 2
            or not 0 <= normalized_pad["surface_z_m"] - normalized_pad["support_top_z_m"] <= .003):
        raise ValueError("Green square requires a nonempty interior on its support surface")
    result = {"scene_config_sha256": digest, "prop_dimensions_m": dimensions,
              "target_pad": normalized_pad}
    if "convex_colliders" in value:
        result["convex_colliders"] = convex.validate_specs(value["convex_colliders"], dimensions)
    if "open_box" in value:
        box = value["open_box"]
        if not isinstance(box, dict) or box.get("name") != "open box":
            raise ValueError("Expected box geometry must be named open box")
        normalized_box = {"name": "open box", "center_xy_m": vector(box.get("center_xy_m"), 2)}
        for key in ("outer_dimensions_m", "interior_dimensions_m"):
            normalized_box[key] = vector(box.get(key), 3, positive=True)
        for key in ("wall_thickness_m", "base_thickness_m", "support_top_z_m", "rim_top_z_m"):
            normalized_box[key] = vector([box.get(key)], 1, positive=True)[0]
        width, depth, height = normalized_box["outer_dimensions_m"]
        wall, floor = normalized_box["wall_thickness_m"], normalized_box["base_thickness_m"]
        if (not wall < min(width, depth) / 2 or not floor < height
                or not np.allclose(normalized_box["interior_dimensions_m"],
                                   [width-2*wall, depth-2*wall, height-floor], atol=1e-9, rtol=0)
                or abs(normalized_box["support_top_z_m"]-floor) > 1e-9
                or abs(normalized_box["rim_top_z_m"]-height) > 1e-9):
            raise ValueError("Open box cavity, floor and rim must match its five solid parts")
        result["open_box"] = normalized_box
    return result


def load_expected_scene_geometry(path):
    """Freeze reviewed configuration bytes before contacting the live bridge."""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path.stat().st_size > 1024 * 1024:
        raise ValueError("Scene configuration must be a bounded local checkout file")
    raw = path.read_bytes()
    config = json.loads(raw, object_pairs_hook=original.unique_json_object)
    if config.get("version") != 1:
        raise ValueError("Unsupported scene configuration")
    dimensions = dict(config.get("cube_dimensions", {}))
    for prop in config.get("props", []):
        if prop["name"] in dimensions:
            raise ValueError("Duplicate configured prop dimensions")
        dimensions[prop["name"]] = prop["dimensions"]
    expected = {
        "scene_config_sha256": hashlib.sha256(raw).hexdigest(),
        "prop_dimensions_m": dimensions, "target_pad": config.get("target_pad")}
    # Legacy cube-only geometry receipts did not supply proxy/mass contracts.
    # They remain usable for cubes; convex proof requires the complete contract.
    if any('proxy' in p or 'mass' in p for p in config.get('props', [])):
        expected['convex_colliders'] = {p['name']: {'proxy': p.get('proxy'), 'mass_kg': p.get('mass'),
            'dimensions_m': p['dimensions']} for p in config.get('props', [])}
    if "open_box" in config:
        expected["open_box"] = config["open_box"]
    return validate_expected_scene_geometry(expected)


def scene_geometry_snapshot_code(*, include_open_box=False, convex_object_name=None, expected_props=original.PROPS):
    """Read existing, fixed-size marker meshes and collider metadata without edits."""
    expected_props = validate_expected_props(expected_props)
    if convex_object_name is not None and convex_object_name not in expected_props:
        raise ValueError("Convex target is absent from the independently expected scene")
    code = '''from pxr import UsdGeom as _gpu_UG, Gf as _gpu_GF, UsdShade as _gpu_US
def _gpu_scene_geometry_snapshot():
    if not isinstance(_PROP_DIMENSIONS, dict) or set(_PROP_DIMENSIONS) != set(EXPECTED_PROPS_LITERAL):
        raise RuntimeError("Live prop dimension names differ from the independently expected scene")
    _cache = _gpu_UG.XformCache()
    def _mesh(path, point_count=None):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(_gpu_UG.Mesh) or not prim.IsActive():
            raise RuntimeError("Missing active geometry: " + path)
        mesh = _gpu_UG.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None or not 4 <= len(points) <= 10000 or (point_count is not None and len(points) != point_count):
            raise RuntimeError("Unexpected bounded geometry: " + path)
        points = _obs_np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not _obs_np.isfinite(points).all():
            raise RuntimeError("Invalid geometry vertices: " + path)
        return prim, mesh, points
    def _world(prim, points):
        transform = _cache.GetLocalToWorldTransform(prim)
        world = _obs_np.asarray([transform.Transform(_gpu_GF.Vec3d(*p)) for p in points], dtype=float)
        world[:,2] -= float(BASE_Z)
        return world
    border, bm, bp = _mesh("/World_Props/green_square/border", 8)
    fill, fm, fp = _mesh("/World_Props/green_square/fill", 4)
    table, tm, tp = _mesh("/World_Props/table", 8)
    bw, fw, tw = _world(border, bp), _world(fill, fp), _world(table, tp)
    def _green_material(prim, translucent):
        material, _ = _gpu_US.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            return False
        shader, _, _ = material.ComputeSurfaceSource()
        if not shader:
            return False
        color = shader.GetInput("diffuseColor").Get()
        opacity = shader.GetInput("opacity").Get()
        return bool(color is not None and len(color) == 3 and color[1] >= .2
            and color[1] > 1.5 * max(color[0], color[2]) and opacity is not None
            and (0 < float(opacity) < 1 if translucent else abs(float(opacity) - 1) < 1e-6))
    cubes = {}
    for name in ("pink_cube", "green_cube"):
        prim, mesh, points = _mesh("/World_Props/" + name)
        lo, hi = points.min(axis=0), points.max(axis=0)
        linear = _obs_np.asarray(_cache.GetLocalToWorldTransform(prim), dtype=float)[:3,:3]
        cubes[name] = {"dimensions_m": (hi-lo).tolist(),
            "symmetric_about_origin": bool(_obs_np.max(_obs_np.abs(lo+hi)) < 1e-7),
            "no_scale_or_shear": bool(_obs_np.allclose(linear @ linear.T, _obs_np.eye(3), atol=1e-6, rtol=0)),
            "bounding_cube": bool(prim.HasAPI(_obs_UP.CollisionAPI)
                and _obs_UP.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "boundingCube"),
            "rigid_body": bool(prim.HasAPI(_obs_UP.RigidBodyAPI))}
    _result = {"scene_config_sha256": _SCENE_IDENTITY.get("scene_config_sha256"),
        "prop_dimensions_m": {name: list(_PROP_DIMENSIONS[name]) for name in EXPECTED_PROPS_LITERAL},
        "cube_colliders": cubes,
        "target_pad": {"name": "green square", "border_vertices_m": bw.tolist(), "fill_vertices_m": fw.tolist(),
            "visible": all(_gpu_UG.Imageable(prim).ComputeVisibility() != _gpu_UG.Tokens.invisible for prim in (border, fill)),
            "visual_only": all(not prim.HasAPI(_obs_UP.CollisionAPI) and not prim.HasAPI(_obs_UP.RigidBodyAPI) for prim in (border, fill)),
            "topology_valid": (list(bm.GetFaceVertexCountsAttr().Get()) == [4]*4
                and list(bm.GetFaceVertexIndicesAttr().Get()) == [0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
                and list(fm.GetFaceVertexCountsAttr().Get()) == [4]
                and list(fm.GetFaceVertexIndicesAttr().Get()) == [0,1,2,3]),
            "green_materials": _green_material(border, False) and _green_material(fill, True),
            "support_top_z_m": float(tw[:,2].max()),
            "support_bounds_xy_m": [tw[:,:2].min(axis=0).tolist(), tw[:,:2].max(axis=0).tolist()],
            "support_is_static_collider": bool(table.HasAPI(_obs_UP.CollisionAPI)
                and not table.HasAPI(_obs_UP.RigidBodyAPI)
                and _obs_UP.MeshCollisionAPI(table).GetApproximationAttr().Get() == "boundingCube")}}
OPEN_BOX_READBACK
    return _result
_gpu_observed["scene_geometry"] = _gpu_scene_geometry_snapshot()'''.replace("EXPECTED_PROPS_LITERAL", repr(expected_props)).replace("OPEN_BOX_READBACK", '''    root = stage.GetPrimAtPath("/World_Props/open_box")
    names = ("floor", "left", "right", "front", "back")
    if not root or not root.IsActive():
        raise RuntimeError("Missing active open box")
    exact = set(str(child.GetName()) for child in root.GetChildren()) == set(names)
    static = bool(exact and not root.HasAPI(_obs_UP.RigidBodyAPI) and not root.HasAPI(_obs_UP.CollisionAPI))
    visible, aligned, parts = True, True, {}
    for name in names:
        prim, mesh, points = _mesh("/World_Props/open_box/" + name)
        world = _world(prim, points)
        # One-nanometre rounding keeps the fixed five-part receipt compact;
        # independent geometry matching retains its one-micrometre tolerance.
        parts[name] = _obs_np.round(_obs_np.stack((world.min(axis=0), world.max(axis=0))), 9).tolist()
        linear = _obs_np.asarray(_cache.GetLocalToWorldTransform(prim), dtype=float)[:3,:3]
        aligned &= bool(_obs_np.allclose(_obs_np.abs(linear), _obs_np.eye(3), atol=1e-6, rtol=0))
        static &= bool(not prim.GetChildren() and prim.HasAPI(_obs_UP.CollisionAPI)
            and not prim.HasAPI(_obs_UP.RigidBodyAPI)
            and _obs_UP.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
            and _obs_UP.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "boundingCube")
        visible &= _gpu_UG.Imageable(prim).ComputeVisibility() != _gpu_UG.Tokens.invisible
    _result["open_box"] = {"name": "open box", "parts_bounds_xyz_m": parts,
        "five_static_colliders": bool(static), "axis_aligned": bool(aligned), "visible": bool(visible)}''' if include_open_box else "")
    return code + (convex.snapshot_code(convex_object_name) if convex_object_name is not None else "")


def snapshot_code(camera="proof", *, object_name="pink_cube", include_scene_geometry=False, include_open_box=False,
                  expected_props=original.PROPS):
    object_name = validate_object_name(object_name)
    expected_props = validate_expected_props(expected_props)
    if object_name not in expected_props:
        raise ValueError("Pick target is absent from the independently expected scene")
    if include_open_box and not include_scene_geometry:
        raise ValueError("Box readback requires complete scene geometry")
    code = original.build_snapshot_code(props=expected_props, camera=camera)
    old = '_obs_payload = _obs_json.dumps(_obs_collect(), separators=(",",":"), allow_nan=False)'
    new = '''_gpu_expected_props = EXPECTED_PROPS_LITERAL
_gpu_prop_root = stage.GetPrimAtPath("/World_Props")
_gpu_live_prop_names = [str(prim.GetName()) for prim in _gpu_prop_root.GetChildren()
    if prim.IsActive() and prim.HasAPI(_obs_UP.RigidBodyAPI)] if _gpu_prop_root else []
if len(_gpu_live_prop_names) != len(_gpu_expected_props) or set(_gpu_live_prop_names) != set(_gpu_expected_props):
    raise RuntimeError("Live rigid prop names differ from the independently expected scene")
if not isinstance(_PROP_SPAWNS, dict) or set(_PROP_SPAWNS) != set(_gpu_expected_props):
    raise RuntimeError("Live spawn names differ from the independently expected scene")
_gpu_observed = _obs_collect()
_gpu_observed["gpu_attestation"] = physics_device_identity(SimulationManager, require_cuda=True)["gpu_attestation"]
_gpu_observed["gpu_attestation"]["fallback_log_count"] = len(_gpu_log_guard.failures)
_gpu_log_guard.check()
_gpu_observed["contacts"] = _gpu_contact_snapshot(OBJECT_NAME_LITERAL)
_gpu_observed["spawn_positions_m"] = {name:list(_PROP_SPAWNS[name]) for name in _gpu_expected_props}
SCENE_GEOMETRY_READBACK
_obs_payload = _obs_json.dumps(_gpu_observed, separators=(",",":"), allow_nan=False)'''.replace(
        "OBJECT_NAME_LITERAL", repr(object_name)).replace("EXPECTED_PROPS_LITERAL", repr(expected_props)).replace("SCENE_GEOMETRY_READBACK",
        scene_geometry_snapshot_code(include_open_box=include_open_box,
            convex_object_name=object_name if object_name in convex.CONVEX_TARGETS else None,
            expected_props=expected_props)
        if include_scene_geometry else "")
    if code.count(old) != 1:
        raise RuntimeError("Existing observer source boundary changed; review its new snapshot contract")
    code = code.replace(old, new)
    if include_scene_geometry and object_name in convex.CONVEX_TARGETS:
        boundary = 'if len((_obs_payload + '
        if code.count(boundary) != 1:
            raise RuntimeError('Observer output boundary changed; review before encoding')
        code = code.split(boundary)[0] + codec.encode_code()
    return code


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class GpuProofObserver:
    def __init__(self, out, *, host="127.0.0.1", port=8611, interval_s=.15, budget_s=1800,
                 object_name="pink_cube", expected_scene_geometry=None, destination_name="green square"):
        self.object_name = validate_object_name(object_name)
        self.destination_name = validate_destination_name(destination_name)
        self.expected_scene_geometry = (validate_expected_scene_geometry(expected_scene_geometry)
                                        if expected_scene_geometry is not None else None)
        self.expected_props = (validate_expected_props(tuple(self.expected_scene_geometry["prop_dimensions_m"]))
                               if self.expected_scene_geometry is not None else original.PROPS)
        if self.object_name not in self.expected_props:
            raise ValueError("Pick target is absent from the independently expected scene")
        if self.destination_name == "open box" and (
                self.expected_scene_geometry is None or "open_box" not in self.expected_scene_geometry):
            raise ValueError("Open-box proof requires independently expected box geometry")
        if self.object_name in convex.CONVEX_TARGETS:
            if (self.expected_scene_geometry is None or self.destination_name != "green square"
                    or self.object_name not in self.expected_scene_geometry.get("convex_colliders", {})):
                raise ValueError("Convex proof requires its configured collider and the green square")
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(ROOT) or not .05 <= interval_s <= 1 or not 1 <= budget_s <= 7200:
            raise ValueError("Invalid bounded observer path or timing")
        self.host, self.port, self.interval_s, self.budget_s = host, port, interval_s, budget_s
        self.records, self.errors, self.marks = [], [], {}
        self._lock = threading.Lock()
        self._stop, self._ready = threading.Event(), threading.Event()
        self._thread = None
        self.complete = False

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=False)
        self.code = snapshot_code(object_name=self.object_name,
            include_scene_geometry=self.expected_scene_geometry is not None,
            include_open_box=self.expected_scene_geometry is not None and "open_box" in self.expected_scene_geometry,
            expected_props=self.expected_props)
        (self.out / "snapshot-code.py").write_text(self.code)
        self.started = time.monotonic()
        self._thread = threading.Thread(target=self._collect, name="gpu-proof-observer", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15):
            self.stop()
            raise RuntimeError("GPU observer could not obtain its first live snapshot in 15 seconds")
        if self.errors:
            self.stop()
            raise RuntimeError("GPU observer failed: " + self.errors[0])
        if self.expected_scene_geometry is not None:
            with self._lock:
                first_scene = audit_scene_geometry([self.records[0]["physics"]], self.expected_scene_geometry)
                first_scene["checks"].update(audit_prop_inventory([self.records[0]["physics"]], self.expected_props))
                first_scene["pass"] = bool(all(first_scene["checks"].values()))
            _write(self.out / "first-scene-geometry.json", first_scene)
            if not first_scene["pass"]:
                failed = ", ".join(name for name, passed in first_scene["checks"].items() if not passed)
                self.errors.append("First live scene differs from reviewed geometry: " + failed)
                self.stop()
                raise RuntimeError(self.errors[-1])
        return self

    def __exit__(self, typ, value, traceback):
        self.stop()

    def _collect(self):
        from cascade.sim.bridge_client import BridgeClient
        client = BridgeClient(host=self.host, port=self.port, timeout_s=8)
        try:
            client.connect()
            self.ping_before = client.request({"op": "ping"})
            _write(self.out / "ping-before.json", self.ping_before)
            with (self.out / "wire.jsonl").open("w") as wires, (self.out / "samples.jsonl").open("w") as samples:
                while not self._stop.is_set() and time.monotonic() - self.started < self.budget_s:
                    began = time.monotonic()
                    reply = client.request({"op": "exec", "code": self.code}, timeout_s=8)
                    finished = time.monotonic()
                    wires.write(json.dumps({"sequence": len(self.records), "reply": reply}) + "\n")
                    wires.flush()
                    physics = original.parse_snapshot_reply(reply)
                    row = {"sequence": len(self.records), "client_started_monotonic": began,
                           "client_finished_monotonic": finished, "physics": physics}
                    with self._lock:
                        self.records.append(row)
                    samples.write(json.dumps(row, allow_nan=False) + "\n"); samples.flush()
                    self._ready.set()
                    self._stop.wait(max(0, self.interval_s - (time.monotonic() - began)))
                self.ping_after = client.request({"op": "ping"})
                _write(self.out / "ping-after.json", self.ping_after)
                self.complete = self._stop.is_set() and bool(self.records)
                if not self.complete:
                    self.errors.append("GPU observer exhausted its finite budget")
        except Exception as exc:
            self.errors.append(repr(exc))
            self._ready.set()
        finally:
            client.close()

    def settle(self, simulation_seconds=.65, wall_timeout=20):
        """Observe a stationary post-action window without moving or freezing physics."""
        with self._lock:
            start_sim = self.records[-1]["physics"]["sim_time"] if self.records else None
        if start_sim is None:
            raise RuntimeError("GPU observer has no physics clock")
        deadline = time.monotonic() + wall_timeout
        while time.monotonic() < deadline:
            if self.errors:
                raise RuntimeError("GPU observer failed: " + self.errors[0])
            with self._lock:
                elapsed = self.records[-1]["physics"]["sim_time"] - start_sim
            if elapsed >= simulation_seconds:
                return
            self._stop.wait(.05)
        raise RuntimeError("GPU proof could not observe a settling window on its advancing physics clock")

    def mark(self, phase):
        if phase in self.marks:
            raise RuntimeError("Duplicate proof phase " + phase)
        self.marks[phase] = {"monotonic": time.monotonic(), "unix": time.time()}
        _write(self.out / "phases.json", self.marks)

    @contextmanager
    def instrument(self, proof_module):
        """Wrap native proof boundaries while preserving every command/threshold."""
        actual = proof_module.agent_turn
        signature = inspect.signature(actual)
        @functools.wraps(actual)
        def turn(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            required = bound.arguments.get("required_tool")
            required = (required,) if isinstance(required, str) else (required or ())
            phase = "pick" if "pick_and_place" in required else "reset" if "reset_scene" in required else None
            if phase:
                self.mark(phase + "_begin")
            result = actual(*args, **kwargs)
            if phase:
                self.settle()
                self.mark(phase + "_end")
            return result
        proof_module.agent_turn = turn
        try:
            yield self
        finally:
            proof_module.agent_turn = actual

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=12)
            if self._thread.is_alive():
                self.errors.append("GPU observer thread did not stop within the socket timeout")
        if self.out.is_dir():
            _write(self.out / "metadata.json", {"kind": "passive GPU physics/contact observation",
                "complete": self.complete, "samples": len(self.records), "errors": self.errors,
                "object_name": self.object_name,
                "expected_props": list(self.expected_props),
                "destination_name": self.destination_name,
                "expected_scene_geometry": self.expected_scene_geometry,
                "marks": self.marks, "snapshot_sha256": hashlib.sha256(self.code.encode()).hexdigest(),
                "existing_auditor_sha256": hashlib.sha256((ROOT/"demo/kitchen/observer/audit.py").read_bytes()).hexdigest()})

    def audit(self, *, target_xy=None, object_half_height=None):
        result = audit_records(self.records, marks=self.marks, complete=self.complete,
                               errors=self.errors, target_xy=target_xy, object_half_height=object_half_height,
                               object_name=self.object_name, expected_scene_geometry=self.expected_scene_geometry,
                               destination_name=self.destination_name)
        _write(self.out / "gpu-physical-audit.json", result)
        return result


def _square_vertices(pad, half):
    x, y = pad["center_xy_m"]
    z = pad["surface_z_m"]
    return np.asarray([[x-half, y-half, z], [x+half, y-half, z],
                       [x+half, y+half, z], [x-half, y+half, z]])


def _open_box_part_bounds(box):
    x, y = box["center_xy_m"]
    width, depth, height = box["outer_dimensions_m"]
    wall, floor = box["wall_thickness_m"], box["base_thickness_m"]
    x0, x1, y0, y1 = x-width/2, x+width/2, y-depth/2, y+depth/2
    return {
        "floor": [[x0, y0, 0], [x1, y1, floor]],
        "left": [[x0, y0, floor], [x0+wall, y1, height]],
        "right": [[x1-wall, y0, floor], [x1, y1, height]],
        "front": [[x0+wall, y0, floor], [x1-wall, y0+wall, height]],
        "back": [[x0+wall, y1-wall, floor], [x1-wall, y1, height]]}


def _box_inner_bounds(parts):
    return [[parts["left"][1][0], parts["front"][1][1]],
            [parts["right"][0][0], parts["back"][0][1]]]


def audit_scene_geometry(samples, expected):
    """Bind every sample to reviewed bytes and measured, visible stage geometry."""
    pad = expected["target_pad"]
    outer = _square_vertices(pad, pad["outer_size_m"] / 2)
    inner = _square_vertices(pad, pad["outer_size_m"] / 2 - pad["border_width_m"])
    checks = {"scene_config_sha256_matches": bool(samples), "prop_dimensions_match": bool(samples),
              "cube_collider_geometry_matches": bool(samples), "green_square_geometry_matches": bool(samples)}
    if "open_box" in expected:
        checks["open_box_geometry_matches"] = bool(samples)
        expected_parts = _open_box_part_bounds(expected["open_box"])
    result = {"pass": False, "checks": checks, "samples": len(samples),
              "expected": expected, "geometry_readback_tolerance_m": 1e-6}

    def matches(actual, wanted, tolerance=1e-6):
        try:
            actual, wanted = np.asarray(actual, float), np.asarray(wanted, float)
            return bool(actual.shape == wanted.shape and np.isfinite(actual).all()
                        and np.allclose(actual, wanted, atol=tolerance, rtol=0))
        except (TypeError, ValueError):
            return False

    for sample in samples:
        geometry = sample.get("scene_geometry")
        if not isinstance(geometry, dict):
            for check in checks:
                checks[check] = False
            continue
        checks["scene_config_sha256_matches"] &= geometry.get("scene_config_sha256") == expected["scene_config_sha256"]
        dimensions = geometry.get("prop_dimensions_m", {})
        checks["prop_dimensions_match"] &= bool(isinstance(dimensions, dict)
            and set(dimensions) == set(expected["prop_dimensions_m"])
            and all(matches(dimensions[name], values, 1e-9) for name, values in expected["prop_dimensions_m"].items()))
        cubes = geometry.get("cube_colliders", {})
        checks["cube_collider_geometry_matches"] &= bool(isinstance(cubes, dict)
            and set(cubes) == {"pink_cube", "green_cube"}
            and all(isinstance(cubes[name], dict)
                and matches(cubes[name].get("dimensions_m"), expected["prop_dimensions_m"][name])
                and all(cubes[name].get(key) is True for key in
                    ("symmetric_about_origin", "no_scale_or_shear", "bounding_cube", "rigid_body"))
                for name in ("pink_cube", "green_cube")))
        observed = geometry.get("target_pad", {})
        try:
            bounds = np.asarray(observed.get("support_bounds_xy_m"), float)
            support_contains = bool(bounds.shape == (2, 2) and np.isfinite(bounds).all()
                and (bounds[0] <= outer[:,:2].min(axis=0)).all()
                and (bounds[1] >= outer[:,:2].max(axis=0)).all())
            marker_matches = bool(observed.get("name") == "green square"
                and matches(observed.get("border_vertices_m"), np.concatenate((outer, inner)))
                and matches(observed.get("fill_vertices_m"), inner)
                and matches(observed.get("support_top_z_m"), pad["support_top_z_m"])
                and support_contains and all(observed.get(key) is True for key in
                    ("visible", "visual_only", "topology_valid", "green_materials", "support_is_static_collider")))
        except (AttributeError, TypeError, ValueError):
            marker_matches = False
        checks["green_square_geometry_matches"] &= marker_matches
        if "open_box" in expected:
            box = geometry.get("open_box", {})
            try:
                parts = box.get("parts_bounds_xyz_m", {})
                box_matches = bool(box.get("name") == "open box"
                    and set(parts) == set(expected_parts)
                    and all(matches(parts[name], bounds) for name, bounds in expected_parts.items())
                    and all(box.get(key) is True for key in ("five_static_colliders", "axis_aligned", "visible"))
                    and matches(parts["floor"][0][2], observed.get("support_top_z_m"))
                    and (np.asarray(observed["support_bounds_xy_m"])[0] <= np.asarray(parts["floor"])[0,:2]).all()
                    and (np.asarray(observed["support_bounds_xy_m"])[1] >= np.asarray(parts["floor"])[1,:2]).all())
            except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                box_matches = False
            checks["open_box_geometry_matches"] &= box_matches
    result["pass"] = bool(all(checks.values()))
    return result


def audit_cube_footprint(pick, *, object_name, dimensions_m, pad, settle_sim_s=.5, min_final_samples=4,
                         destination_name="green square", inner_bounds_xy_m=None):
    """Project every corner using measured orientation across the full settle tail."""
    samples = [row["physics"] for row in pick]
    clocks = np.asarray([sample["sim_time"] for sample in samples], float)
    if not len(clocks) or not np.isfinite(clocks).all() or not (np.diff(clocks) >= 0).all():
        raise ValueError("Footprint requires finite, ordered physics clocks")
    start = max(0, int(np.searchsorted(clocks, clocks[-1] - settle_sim_s, side="right")) - 1)
    tail = samples[start:]
    quats = np.asarray([s["props"][object_name]["orientation_wxyz"] for s in tail], float)
    poses = np.asarray([s["props"][object_name]["position_m"] for s in tail], float)
    norms = np.linalg.norm(quats, axis=1)
    if (quats.shape != (len(tail), 4) or poses.shape != (len(tail), 3)
            or not np.isfinite(quats).all() or not np.isfinite(poses).all()
            or not (np.abs(norms - 1) < 1e-3).all()):
        raise ValueError("Footprint requires valid measured cuboid poses")
    vertices = np.asarray(list(itertools.product((-1, 1), repeat=3)), float) * np.asarray(dimensions_m) / 2
    center = np.asarray(pad["center_xy_m"], float)
    if inner_bounds_xy_m is None:
        inner_half = pad["outer_size_m"] / 2 - pad["border_width_m"]
        inner_lo, inner_hi = center - inner_half, center + inner_half
    else:
        bounds = np.asarray(inner_bounds_xy_m, float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
            raise ValueError("Footprint requires finite nonempty destination interior bounds")
        inner_lo, inner_hi = bounds
    clearances, observed_lows, observed_highs, observed_bottoms = [], [], [], []
    for position, (w, x, y, z) in zip(poses, quats / norms[:, None]):
        rotation = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                               [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                               [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        world = vertices @ rotation.T + position
        lo, hi = world[:,:2].min(axis=0), world[:,:2].max(axis=0)
        observed_lows.append(lo); observed_highs.append(hi)
        observed_bottoms.append(float(world[:,2].min()))
        clearances.append(float(min((lo - inner_lo).min(), (inner_hi - hi).min())))
    window = bool(len(tail) >= min_final_samples
        and len({s["physics_step"] for s in tail}) >= min_final_samples
        and clocks[-1] - clocks[start] >= settle_sim_s - 1e-9)
    center_errors = np.linalg.norm(poses[:,:2] - center, axis=1)
    return {"pass": bool(window and min(clearances) >= 0 and (center_errors <= .04).all()),
            "destination_name": destination_name, "method": "all eight rotated cuboid corners",
            "dimensions_m": list(dimensions_m), "final_samples": len(tail),
            "final_window_sim_s": float(clocks[-1] - clocks[start]), "settle_window_complete": window,
            "inner_bounds_xy_m": [inner_lo.tolist(), inner_hi.tolist()],
            "observed_footprint_bounds_xy_m": [np.min(observed_lows, axis=0).tolist(),
                                                np.max(observed_highs, axis=0).tolist()],
            "min_final_bottom_z_m": min(observed_bottoms), "max_final_bottom_z_m": max(observed_bottoms),
            "min_inner_clearance_m": min(clearances), "max_final_center_error_m": float(center_errors.max()),
            "center_tolerance_m": .04, "boundary_tolerance_m": 0.0}


def audit_prop_inventory(samples, expected_props):
    """Check every sample against independent names, never a supplied-dict subset."""
    expected_props = validate_expected_props(expected_props)
    wanted = set(expected_props)
    checks = {"live_prop_names_match_expected": bool(samples),
              "spawn_prop_names_match_expected": bool(samples),
              "spawn_positions_finite_and_unchanged": bool(samples)}
    reference = None
    for sample in samples:
        props = sample.get("props") if isinstance(sample, dict) else None
        spawns = sample.get("spawn_positions_m") if isinstance(sample, dict) else None
        checks["live_prop_names_match_expected"] &= isinstance(props, dict) and set(props) == wanted
        exact_spawns = isinstance(spawns, dict) and set(spawns) == wanted
        checks["spawn_prop_names_match_expected"] &= exact_spawns
        if not exact_spawns:
            checks["spawn_positions_finite_and_unchanged"] = False
            continue
        try:
            positions = np.asarray([spawns[name] for name in expected_props], dtype=float)
            valid = positions.shape == (len(expected_props), 3) and np.isfinite(positions).all()
            if reference is None:
                reference = positions
            checks["spawn_positions_finite_and_unchanged"] &= bool(valid
                and np.array_equal(positions, reference))
        except (TypeError, ValueError):
            checks["spawn_positions_finite_and_unchanged"] = False
    return checks


def audit_records(records, *, marks, complete, errors=(), target_xy=None, object_half_height=None,
                  object_name="pink_cube", expected_scene_geometry=None, destination_name="green square"):
    """Add CUDA/contact/reset requirements to the unmodified prior strict audit."""
    object_name = validate_object_name(object_name)
    destination_name = validate_destination_name(destination_name)
    result = {"pass": False, "object_name": object_name, "checks": {}, "errors": list(errors),
              "pick": None, "reset": {}}
    if expected_scene_geometry is not None:
        result["destination_name"] = destination_name
    checks = result["checks"]
    checks["complete_observation"] = bool(complete and not errors and records)
    checks["all_phase_boundaries"] = set(marks) == {"pick_begin", "pick_end", "reset_begin", "reset_end"}
    if not checks["complete_observation"] or not checks["all_phase_boundaries"]:
        return result
    try:
        all_samples = [r["physics"] for r in records]
        support_top_z = 0
        expected = (validate_expected_scene_geometry(expected_scene_geometry)
                    if expected_scene_geometry is not None else None)
        expected_props = (validate_expected_props(tuple(expected["prop_dimensions_m"]))
                          if expected is not None else original.PROPS)
        if object_name not in expected_props:
            raise ValueError("Pick target is absent from the independently expected scene")
        result["expected_props"] = list(expected_props)
        inventory = audit_prop_inventory(all_samples, expected_props)
        checks.update(inventory)
        checks["all_props_physically_reset_and_settled"] = False
        if not all(inventory.values()):
            return result
        collider_geometry = None
        if object_name in convex.CONVEX_TARGETS and expected_scene_geometry is None:
            raise ValueError("Convex proof requires its configured collider and the green square")
        if destination_name == "open box" and expected_scene_geometry is None:
            raise ValueError("Open-box proof requires independently expected box geometry")
        if expected_scene_geometry is not None:
            if destination_name == "open box" and "open_box" not in expected:
                raise ValueError("Open-box proof requires independently expected box geometry")
            if object_name in convex.CONVEX_TARGETS and destination_name != "green square":
                raise ValueError("Convex target proof currently supports the green square only")
            result["scene_geometry"] = audit_scene_geometry(all_samples, expected)
            checks.update(result["scene_geometry"]["checks"])
            if not result["scene_geometry"]["pass"]:
                return result
            if object_name in convex.CONVEX_TARGETS:
                binding = convex.audit_binding(all_samples, object_name=object_name, expected=expected)
                result["convex_geometry"] = binding
                checks.update(binding["checks"])
                if not binding["pass"]:
                    return result
                collider_geometry = strict.VerifiedColliderGeometry(
                    vertices_m=all_samples[0]["scene_geometry"]["convex_collider"]["vertices_m"],
                    body_name=object_name, receipt={
                        "method": "projected_baked_convex_vertices", "body_name": object_name,
                        "frame": "body_local", "units": "m", "collision_approximation": "convexHull",
                        "vertices_f32_sha256": binding["vertices_f32_sha256"],
                        "vertex_count": binding["vertex_count"], "scene_config_binding_verified": True,
                        "scene_config_sha256": expected["scene_config_sha256"],
                        "half_height_argument_used": False, "upright_tilt_limit_applied": False})
            verified_dimensions = all_samples[0]["scene_geometry"]["prop_dimensions_m"][object_name]
            verified_half_height = verified_dimensions[2] / 2
            destination_key = "open_box" if destination_name == "open box" else "target_pad"
            verified_destination = expected[destination_key]
            verified_target = verified_destination["center_xy_m"]
            if ((object_half_height is not None and not np.isclose(object_half_height, verified_half_height, atol=1e-9, rtol=0))
                    or (target_xy is not None and (np.asarray(target_xy).shape != (2,)
                        or not np.allclose(target_xy, verified_target, atol=1e-9, rtol=0)))):
                raise ValueError("Requested target or half-height differs from verified scene geometry")
            object_half_height, target_xy = verified_half_height, verified_target
            interior_bounds = None
            if destination_name == "open box":
                box_parts = all_samples[0]["scene_geometry"]["open_box"]["parts_bounds_xyz_m"]
                support_top_z = box_parts["floor"][1][2]
                interior_bounds = _box_inner_bounds(box_parts)
            else:
                support_top_z = all_samples[0]["scene_geometry"]["target_pad"]["support_top_z_m"]
            result["object_geometry"] = {"dimensions_m": list(verified_dimensions),
                "half_height_m": object_half_height,
                "source": ("live authored convex collider and tensor mass properties matched to reviewed scene SHA-256"
                    if collider_geometry is not None else
                    "live _PROP_DIMENSIONS matched to reviewed scene SHA-256 and measured cube collider")}
        else:
            target_xy = (.14, -.27) if target_xy is None else target_xy
            object_half_height = .04 if object_half_height is None else object_half_height
        times = [marks[name]["monotonic"] for name in ("pick_begin", "pick_end", "reset_begin", "reset_end")]
        checks["ordered_phases"] = times == sorted(times) and len(set(times)) == 4
        pick = [r for r in records if r["client_started_monotonic"] >= times[0]
                and r["client_finished_monotonic"] <= times[2]]
        result["pick"] = strict.audit_records(pick, object_name=object_name, target_xy=target_xy,
            support_top_z=support_top_z, object_half_height=object_half_height,
            **({"collider_geometry": collider_geometry} if collider_geometry is not None else {}))
        checks["strict_pick_place_and_cameras"] = result["pick"]["pass"]
        if expected is not None and collider_geometry is not None:
            result["pick"]["destination_name"] = destination_name
            pad = expected["target_pad"]
            half = pad["outer_size_m"] / 2 - pad["border_width_m"]
            center = np.asarray(pad["center_xy_m"])
            footprint = convex_settle.audit_convex_settle_geometry(pick, object_name=object_name,
                vertices_body_m=collider_geometry.vertices_m,
                inner_bounds_xy_m=[(center-half).tolist(), (center+half).tolist()],
                support_top_z_m=support_top_z,
                settle_sim_s=result["pick"]["criteria"]["settle_sim_s"],
                min_final_samples=result["pick"]["criteria"]["min_final_samples"])
            result["convex_footprint"] = footprint
            checks["whole_convex_collider_inside_green_square_and_supported"] = footprint["geometry_pass"]
            if object_name == "tomato_can":
                # A stable can lying on its side is not the requested upright place.
                final_indices = [r["sample_index"] for r in footprint["samples"]]
                quats = np.asarray([pick[i]["physics"]["props"][object_name]["orientation_wxyz"] for i in final_indices])
                quats /= np.linalg.norm(quats, axis=1)[:, None]
                checks["tomato_can_settles_upright"] = bool((1 - 2 * (quats[:,1]**2 + quats[:,2]**2)
                    >= np.cos(np.deg2rad(5))).all())
        elif expected is not None:
            result["pick"]["destination_name"] = destination_name
            result["footprint"] = audit_cube_footprint(pick, object_name=object_name,
                dimensions_m=verified_dimensions, pad=verified_destination,
                destination_name=destination_name, inner_bounds_xy_m=interior_bounds,
                settle_sim_s=result["pick"]["criteria"]["settle_sim_s"],
                min_final_samples=result["pick"]["criteria"]["min_final_samples"])
            footprint_check = ("whole_cube_footprint_inside_open_box_throughout_settle"
                if destination_name == "open box" else "whole_cube_inside_green_square_throughout_settle")
            checks[footprint_check] = result["footprint"]["pass"]
            if destination_name == "open box":
                # The unchanged 5 mm centre-height tolerance cannot distinguish
                # this 4 mm floor from the counter beneath it. Independently
                # require the lowest rotated corner to remain on the measured
                # floor throughout settling, with at most 1 mm gap/penetration.
                footprint = result["footprint"]
                floor_thickness = box_parts["floor"][1][2] - box_parts["floor"][0][2]
                tolerance = min(.001, floor_thickness / 4)
                min_clearance = footprint["min_final_bottom_z_m"] - support_top_z
                max_clearance = footprint["max_final_bottom_z_m"] - support_top_z
                supported = bool(footprint["settle_window_complete"]
                    and min_clearance >= -tolerance and max_clearance <= tolerance)
                result["box_support"] = {"pass": supported,
                    "method": "lowest of all eight rotated cuboid corners in every settle sample",
                    "support_top_z_m": support_top_z, "floor_thickness_m": floor_thickness,
                    "tolerance_m": tolerance, "min_bottom_clearance_m": min_clearance,
                    "max_bottom_clearance_m": max_clearance,
                    "max_absolute_support_error_m": max(abs(min_clearance), abs(max_clearance)),
                    "final_samples": footprint["final_samples"],
                    "final_window_sim_s": footprint["final_window_sim_s"],
                    "settle_window_complete": footprint["settle_window_complete"]}
                checks["cube_bottom_supported_on_open_box_floor_throughout_settle"] = supported
        checks["all_physics_tensors_cuda"] = all(all(str(sample["props"][name]["tensor_device"]).startswith("cuda:")
            for name in expected_props) for sample in all_samples)
        checks["gpu_backend_no_fallback"] = all(
            (a := sample["gpu_attestation"]).get("required") is True
            and a.get("backend") == "physx" and a.get("gpu_dynamics") is True
            and a.get("broadphase") == "GPU" and a.get("cuda_context_present") is True
            and a.get("tensor_device_ordinal", -1) >= 0
            and a.get("device") == a.get("tensor_device") == f"cuda:{a.get('tensor_device_ordinal')}"
            and a.get("cpu_fallback_allowed") is False and a.get("fallback_log_count") == 0
            for sample in all_samples)
        def contact_binding_matches(sample):
            contact = sample["contacts"]
            jaw_root = sample["robot_id"] + "/link1/link2/link3/link4/link5/link6/gripper_end"
            jaws = [[jaw_root + "/gripper_left", jaw_root + "/gripper_right"]]
            return (contact.get("sensor_paths") == ["/World_Props/" + object_name]
                    and contact.get("filter_paths") == jaws)
        checks["contact_sensor_and_actual_jaws_bound"] = all(contact_binding_matches(s) for s in all_samples)
        baseline_z = pick[0]["physics"]["props"][object_name]["position_m"][2]
        bilateral = []
        for row in pick:
            sample = row["physics"]; contact = sample["contacts"]
            forces = np.asarray(contact["jaw_forces_n"], float)
            counts = np.asarray(contact["jaw_contact_counts"], int)
            if (contact_binding_matches(sample) and contact.get("channel") == "physx_gpu_contact_tensor"
                and str(contact.get("device", "")).startswith("cuda:")
                and forces.shape == (2, 3) and counts.shape == (2,) and np.isfinite(forces).all()
                and (np.linalg.norm(forces, axis=1) > .01).all() and (counts > 0).all()
                and sample["props"][object_name]["position_m"][2] - baseline_z >= .015):
                bilateral.append({"physics_step": sample["physics_step"], "forces_n": forces.tolist(), "counts": counts.tolist(),
                                  "position_m": sample["props"][object_name]["position_m"]})
        result["bilateral_lift_contacts"] = bilateral
        checks["two_jaw_contacts_during_real_lift"] = len({r["physics_step"] for r in bilateral}) >= 2
        reset_samples = [r["physics"] for r in records if r["client_started_monotonic"] >= times[2]]
        reset_clocks = np.asarray([s["sim_time"] for s in reset_samples])
        start = max(0, int(np.searchsorted(reset_clocks, reset_clocks[-1] - .5, side="right")) - 1)
        tail = reset_samples[start:]
        checks["reset_settle_window"] = (len(tail) >= 4 and len({s["physics_step"] for s in tail}) >= 4
            and tail[-1]["sim_time"] - tail[0]["sim_time"] >= .5)
        for name in expected_props:
            spawn = tail[-1]["spawn_positions_m"][name]
            positions = np.asarray([s["props"][name]["position_m"] for s in tail])
            speed = np.asarray([s["props"][name]["linear_velocity_m_s"] for s in tail])
            spin = np.asarray([s["props"][name]["angular_velocity_rad_s"] for s in tail])
            error = np.linalg.norm(positions - np.asarray(spawn), axis=1)
            report = {"max_spawn_error_m": float(error.max()), "max_speed_m_s": float(np.linalg.norm(speed, axis=1).max()),
                      "max_spin_rad_s": float(np.linalg.norm(spin, axis=1).max()),
                      "max_spread_m": float(np.linalg.norm(positions - positions[-1], axis=1).max())}
            report["pass"] = (np.isfinite([*report.values()]).all() and report["max_spawn_error_m"] <= .02
                and report["max_speed_m_s"] <= .02 and report["max_spin_rad_s"] <= .2 and report["max_spread_m"] <= .005)
            report["pass"] = bool(report["pass"])
            result["reset"][name] = report
        checks["all_props_physically_reset_and_settled"] = (set(result["reset"]) == set(expected_props)
            and all(result["reset"][name]["pass"] for name in expected_props))
        if expected is not None:
            # Prospective scene-bound proof: show the complete held footprint
            # reaching this destination before counting its final landing.
            # The geometry-free legacy path keeps its historical contract.
            if collider_geometry is not None:
                entry_vertices = collider_geometry.vertices_m
            else:
                entry_dimensions = all_samples[0]["scene_geometry"]["cube_colliders"][object_name]["dimensions_m"]
                entry_vertices = (np.asarray(list(itertools.product((-1, 1), repeat=3)), float)
                                  * np.asarray(entry_dimensions) / 2)
            if destination_name == "open box":
                entry_bounds = interior_bounds
            else:
                entry_fill = np.asarray(all_samples[0]["scene_geometry"]["target_pad"]["fill_vertices_m"], float)
                entry_bounds = [entry_fill[:, :2].min(axis=0), entry_fill[:, :2].max(axis=0)]
            result["supported_destination_entry"] = destination_entry.audit_supported_destination_entry(
                pick, object_name=object_name, vertices_body_m=entry_vertices,
                inner_bounds_xy_m=entry_bounds, destination_name=destination_name)
            checks["whole_footprint_enters_destination_with_bilateral_support"] = result["supported_destination_entry"]["pass"]
        result["pass"] = bool(all(checks.values()))
    except Exception as exc:
        result["errors"].append(repr(exc))
    return result
