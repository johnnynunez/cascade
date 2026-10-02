"""Optional, synchronous OVRTX 0.5 camera renderer, independent of physics.

Only this object owns/writes its ovstage. A snapshot is held unchanged from
publication through copying all outputs. An ordinal is a publication gate,
NOT a historical-state selector. Never attach an externally writable stage.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from ..types import Frame


class OvrtxError(RuntimeError):
    pass


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OvrtxError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise OvrtxError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _transform(value, name):
    a = np.asarray(value, dtype=np.float64)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(float(np.linalg.det(a[:3, :3])), 1., abs_tol=1e-8)):
        raise OvrtxError(f"{name} must be a finite rigid 4x4 transform")
    return a.copy()


def _prim_path(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z_0-9]*)+", value):
        raise OvrtxError("Expected an absolute USD prim path with identifier components")
    return value


@dataclass(frozen=True)
class CameraSpec:
    """Square-pixel pinhole; integer pixel centers, optical +Y down/+Z forward.

    OVRTX 0.5 RT2 did not honor unequal focal lengths in the measured USD
    RenderProduct path, including adjustPixelAspectRatio. Reject that request
    instead of returning images with a calibration the renderer did not use.
    """
    name: str
    width: int
    height: int
    fx: float
    fy: float
    T_base_cam: np.ndarray
    near_m: float = .01
    far_m: float = 20.

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.name):
            raise OvrtxError("Invalid camera name")
        for n in ("width", "height"):
            if type(getattr(self, n)) is not int or not 1 <= getattr(self, n) <= 8192:
                raise OvrtxError(f"{n} must be an integer in [1,8192]")
        for n in ("fx", "fy", "near_m", "far_m"):
            object.__setattr__(self, n, _finite(getattr(self, n), n, positive=True))
        if self.fx != self.fy:
            raise OvrtxError("OVRTX 0.5 adapter requires square pixels (fx == fy)")
        if self.near_m >= self.far_m:
            raise OvrtxError("near_m must be smaller than far_m")
        object.__setattr__(self, "T_base_cam", _transform(self.T_base_cam, "T_base_cam"))

    @property
    def K(self):
        # USD raster samples lie at half-integer raster coordinates. Cascade
        # backprojects integer array indices, so subtract half a pixel here.
        return np.array([[self.fx, 0., (self.width - 1) / 2],
                         [0., self.fy, (self.height - 1) / 2], [0., 0., 1.]])


@dataclass(frozen=True)
class SceneSnapshot:
    """One caller-owned scene sample; transforms are LOCAL parent->prim in m.

    Capture this under the physics owner's own lock, including eye-in-hand
    camera transforms. The adapter never reads current q or steps physics.
    Metadata is provenance only, never Isaac proprioception/attachment proof.
    """
    source: str
    epoch: str
    sequence: int
    sim_time: float
    captured_monotonic: float
    transforms: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _snapshot(snapshot, paths, camera_names):
    if not isinstance(snapshot, SceneSnapshot):
        raise OvrtxError("Expected SceneSnapshot")
    if not all(isinstance(v, str) and v for v in (snapshot.source, snapshot.epoch)):
        raise OvrtxError("Snapshot source and epoch must be nonempty")
    if type(snapshot.sequence) is not int or not 0 <= snapshot.sequence < 2**63:
        raise OvrtxError("Snapshot sequence must be a nonnegative integer")
    if not isinstance(snapshot.transforms, dict) or set(snapshot.transforms) != set(paths):
        raise OvrtxError("Snapshot must supply every configured dynamic prim, with no extras")
    if not isinstance(snapshot.cameras, dict) or set(snapshot.cameras) - set(camera_names):
        raise OvrtxError("Snapshot camera names are not configured")
    transforms = {k: _transform(v, k) for k, v in snapshot.transforms.items()}
    cameras = {k: _transform(v, k) for k, v in snapshot.cameras.items()}
    if not isinstance(snapshot.metadata, dict):
        raise OvrtxError("Snapshot metadata must be a JSON object")
    try:
        metadata = json.loads(json.dumps(snapshot.metadata, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise OvrtxError("Snapshot metadata must be finite JSON") from e
    state = {"source": snapshot.source, "epoch": snapshot.epoch,
             "sequence": snapshot.sequence,
             "sim_time": _finite(snapshot.sim_time, "sim_time"),
             "captured_monotonic": _finite(snapshot.captured_monotonic, "captured_monotonic"),
             "transforms": {k: v.tolist() for k, v in transforms.items()},
             "cameras": {k: v.tolist() for k, v in cameras.items()}, "metadata": metadata}
    digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return state, digest, transforms, cameras


def _matrix_usda(matrix):
    # USD/ovstage use row-vector matrices; Cascade uses column vectors.
    return "(" + ", ".join("(" + ", ".join(format(x, ".17g") for x in row) + ")"
                            for row in matrix.T) + ")"


def _camera_layer(scene, specs, device, semantic_paths=None, world_paths=(), instance_paths=()):
    # No pxr/Kit import and no edits to the source scene. Paths are local files;
    # disallow USD asset delimiters rather than interpolating untrusted syntax.
    if any(c in str(scene) for c in "@\n\r"):
        raise OvrtxError("Scene path contains a USD asset delimiter")
    optical_to_usd = np.diag([1., -1., -1., 1.])
    cameras, products = [], []
    for c in specs:
        cameras.append(f'''    def Camera "{c.name}" {{
        token projection = "perspective"
        float focalLength = 24
        float horizontalAperture = {24 * c.width / c.fx:.17g}
        float verticalAperture = {24 * c.height / c.fy:.17g}
        float horizontalApertureOffset = 0
        float verticalApertureOffset = 0
        float2 clippingRange = ({c.near_m}, {c.far_m})
        float fStop = 0
        double shutter:open = 0
        double shutter:close = 0
        matrix4d xformOp:transform = {_matrix_usda(c.T_base_cam @ optical_to_usd)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}''')
        semantic_vars = (f', </CascadeRender/{c.name}/Semantic>, </CascadeRender/{c.name}/Labels>'
                         if semantic_paths else '')
        semantic_defs = '''
        def RenderVar "Semantic" {
            string sourceName = "SemanticSegmentation"
        }
        def RenderVar "Labels" {
            string sourceName = "SemanticIdMap"
        }
''' if semantic_paths else ''
        products.append(f'''    def RenderProduct "{c.name}" {{
        int2 resolution = ({c.width}, {c.height})
        uniform token aspectRatioConformPolicy = "expandAperture"
        uniform float pixelAspectRatio = 1
        uint[] deviceIds = [{device}]
        rel camera = </CascadeCameras/{c.name}>
        token omni:rtx:rendermode = "RealTimePathTracing"
        rel orderedVars = [</CascadeRender/{c.name}/Color>, </CascadeRender/{c.name}/Depth>{semantic_vars}]
        def RenderVar "Color" {{
            string sourceName = "LdrColor"
        }}
        def RenderVar "Depth" {{
            string sourceName = "DistanceToImagePlaneSD"
        }}
        {semantic_defs}
    }}''')
    from .ovrtx_masks import semantic_layer
    return ('#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n'
            f'    subLayers = [@{scene}@]\n)\n'
            'def "CascadeCameras" {\n' + "\n".join(cameras) + '\n}\n'
            'def "CascadeRender" {\n' + "\n".join(products) + '\n}\n'
            + (semantic_layer(semantic_paths, world_paths=world_paths, instance_paths=instance_paths)
               if semantic_paths else ''))


class OvrtxRenderer:
    """Own one isolated stage and produce copied, coherent RGB-D packets.

    No robot transport, physics stepping, or attachment inference. Optional
    semantic outputs identify explicitly inventoried visible body geometry.
    Synchronous native calls may compile shaders; use a process supervisor for
    a hard startup deadline. Errors poison the owner until close, no retries.
    """
    def __init__(self, scene, cameras, *, dynamic_paths=(), device=0, render_dt=1/30,
                 meters_per_unit=1., semantic_paths=None, world_paths=(), instance_paths=(), _modules=None):
        self.scene = Path(scene).expanduser().resolve(strict=True)
        if not self.scene.is_file() or meters_per_unit != 1.:
            raise OvrtxError("OVRTX scenes must be local files authored in metres (meters_per_unit=1)")
        self.cameras = tuple(copy.deepcopy(cameras))
        if (not self.cameras or not all(isinstance(c, CameraSpec) for c in self.cameras)
                or len({c.name for c in self.cameras}) != len(self.cameras)):
            raise OvrtxError("Expected distinct calibrated cameras")
        self.paths = tuple(_prim_path(p) for p in dynamic_paths)
        if (len(set(self.paths)) != len(self.paths)
                or any(p.startswith(("/CascadeCameras", "/CascadeRender")) for p in self.paths)):
            raise OvrtxError("Dynamic prim paths must be distinct and outside renderer-owned paths")
        if type(device) is not int or not 0 <= device < 64:
            raise OvrtxError("device must be a CUDA-visible integer index in [0,63]")
        self.device = device
        self.world_paths = tuple(_prim_path(p) for p in world_paths)
        self.instance_paths = tuple(_prim_path(p) for p in instance_paths)
        if set(self.world_paths) - set(self.paths) or ((self.world_paths or self.instance_paths) and not semantic_paths):
            raise OvrtxError("World-pose overrides require inventoried dynamic paths and semantic geometry")
        self.semantic_paths = copy.deepcopy(semantic_paths or {})
        if self.semantic_paths:
            from .ovrtx_masks import semantic_layer
            semantic_layer(self.semantic_paths)  # Validate before native creation.
        self.render_dt = _finite(render_dt, "render_dt", positive=True)
        self._modules = _modules  # Dependency injection for CPU contract tests only.
        self._lock = threading.RLock()
        self._renderer = self._stage = self._dictionary = None
        self._queries = []
        self._ordinal = 1
        self._epoch = uuid.uuid4().hex
        self._last_state = self._last_digest = self._last_frames = None
        self._poisoned = False
        self._closed = False
        self.scene_sha256 = hashlib.sha256(self.scene.read_bytes()).hexdigest()

    def open(self):
        with self._lock:
            if self._closed or self._poisoned:
                raise OvrtxError("Renderer is closed or invalid; create a new owner")
            if self._renderer is not None:
                return
            try:
                if hashlib.sha256(self.scene.read_bytes()).hexdigest() != self.scene_sha256:
                    raise OvrtxError("Root USD changed before loading its recorded source")
                if self._modules is None:
                    if not (3, 10) <= sys.version_info[:2] < (3, 14):
                        raise OvrtxError("OVRTX 0.5 requires Python 3.10-3.13")
                    for package, expected in (("ovrtx", "0.5.0.377615"), ("ovstage", "0.2.0.377349")):
                        if importlib.metadata.version(package) != expected:
                            raise OvrtxError(f"Expected {package}=={expected}; use the ovrtx extra")
                    self._ovrtx, self._ovstage = (importlib.import_module(p) for p in ("ovrtx", "ovstage"))
                else:
                    self._ovrtx, self._ovstage = self._modules
                self._renderer = self._ovrtx.Renderer(config=self._ovrtx.RendererConfig(
                    active_cuda_gpus=str(self.device), enable_geometry_streaming=False,
                    keep_system_alive=False))
                self._stage = self._ovstage.Stage(f"cascade.ovrtx.{self._epoch}")
                self._renderer.attach_ovstage(self._stage)
                self._ovstage.population.open_usd_from_string(
                    self._stage, _camera_layer(self.scene, self.cameras, self.device, self.semantic_paths,
                                              self.world_paths, self.instance_paths), ordinal=1)
                self._stage.advance_write_floor(1, self._ovstage.Scope.ALL).wait()
                if hashlib.sha256(self.scene.read_bytes()).hexdigest() != self.scene_sha256:
                    raise OvrtxError("Root USD changed while loading its recorded source")
                self._dictionary = self._ovstage.PathDictionary(self._stage)
                # Single-path queries prevent dependence on unspecified multi-prim
                # query ordering. Require every path to exist, never UPSERT bodies.
                for path in (*self.paths, *(f"/CascadeCameras/{c.name}" for c in self.cameras)):
                    path_list = self._dictionary.create_path_list_from_strings([path])
                    query = self._stage.query_from_path_list(path_list)
                    self._queries.append((path, path_list, query))
                    if query.result().total_prim_count != 1:
                        raise OvrtxError(f"Scene is missing exactly one prim {path}")
            except Exception as e:
                self._poisoned = True
                cleanup_errors = self._cleanup()
                raise OvrtxError(f"OVRTX initialization failed: {e}; cleanup={cleanup_errors}") from e

    def _read(self, render_var):
        mapped = render_var.map(device=self._ovrtx.Device.CPU)
        try:
            # The copy outlives native result/mapping teardown. No consumer view
            # survives unmap, including on errors.
            return np.from_dlpack(mapped).copy()
        finally:
            mapped.unmap()

    def render(self, snapshot):
        with self._lock:
            if self._poisoned or self._closed:
                raise OvrtxError("Renderer is closed or invalid")
            state, digest, transforms, camera_poses = _snapshot(snapshot, self.paths, [c.name for c in self.cameras])
            if state["captured_monotonic"] > time.monotonic():
                raise OvrtxError("Snapshot capture time is in the future of this host clock")
            prior = self._last_state
            if prior is not None:
                if (state["source"], state["epoch"]) != (prior["source"], prior["epoch"]):
                    raise OvrtxError("Snapshot source/epoch changed; create a new renderer owner")
                if state["sequence"] == prior["sequence"]:
                    if digest != self._last_digest:
                        raise OvrtxError("Contradictory data for the same snapshot sequence")
                    return copy.deepcopy(self._last_frames)
                if (state["sequence"] < prior["sequence"] or state["sim_time"] <= prior["sim_time"]
                        or state["captured_monotonic"] <= prior["captured_monotonic"]):
                    raise OvrtxError("Snapshot clock/sequence did not advance")
            self.open()
            try:
                self._ordinal += 1
                optical_to_usd = np.diag([1., -1., -1., 1.])
                for c in self.cameras:
                    transforms[f"/CascadeCameras/{c.name}"] = camera_poses.get(c.name, c.T_base_cam) @ optical_to_usd
                for path, _, query in self._queries:
                    self._stage.write_attribute(query, "omni:xform", ordinal=self._ordinal,
                        tensors=np.ascontiguousarray(transforms[path].T[None]), is_array=False).wait()
                self._stage.advance_write_floor(self._ordinal, self._ovstage.Scope.ALL).wait()
                products = self._renderer.step(
                    render_products={f"/CascadeRender/{c.name}" for c in self.cameras},
                    delta_time=self.render_dt, ordinal=self._ordinal)
                frames = {}
                for c in self.cameras:
                    product_path = f"/CascadeRender/{c.name}"
                    product = products[product_path]
                    if len(product.frames) != 1:
                        raise OvrtxError("Expected one instantaneous camera frame, not missing/interpolated frames")
                    frame = product.frames[0]
                    start = _finite(frame.start_time, "sensor start_time")
                    end = _finite(frame.end_time, "sensor end_time")
                    step_start = _finite(products.simulation_start_time, "render step start")
                    step_end = _finite(products.simulation_end_time, "render step end")
                    if start != end:
                        raise OvrtxError("Camera exposure spans time; this adapter requires instantaneous snapshots")
                    if not step_start <= start <= end <= step_end or step_start >= step_end:
                        raise OvrtxError("Sensor capture is not inside this renderer step")
                    if self._last_frames is not None and end <= self._last_frames[c.name].capture["render_reference"]["sensor_end_s"]:
                        raise OvrtxError("Sensor output repeated/regressed across different scene snapshots")
                    color = self._read(frame.render_vars[f"{product_path}/Color"])
                    depth = self._read(frame.render_vars[f"{product_path}/Depth"])
                    if color.shape != (c.height, c.width, 4) or color.dtype != np.uint8:
                        raise OvrtxError("LdrColor must be RGBA uint8 at the calibrated resolution")
                    if depth.shape != (c.height, c.width, 1) or depth.dtype != np.float32:
                        raise OvrtxError("DistanceToImagePlaneSD must be HxWx1 float32 metres")
                    depth = depth[..., 0].copy()
                    # SDK background infinity/clip values become explicit invalid
                    # depth. Never synthesize depth from RGB or a different frame.
                    valid = np.isfinite(depth) & (depth >= c.near_m) & (depth < c.far_m)
                    depth[~valid] = 0
                    masks = None
                    semantic_labels = None
                    if self.semantic_paths:
                        from .ovrtx_masks import body_masks, decode_id_map
                        semantic_buffer = self._read(frame.render_vars[f"{product_path}/Labels"])
                        semantic_labels = decode_id_map(semantic_buffer)
                        masks = body_masks(self._read(frame.render_vars[f"{product_path}/Semantic"]), semantic_buffer,
                            (c.height, c.width), sorted(set(self.semantic_paths.values())))
                    capture = {"backend": "ovrtx", "source": str(self.scene), "camera": c.name,
                               "producer_epoch": self._epoch, "t": state["captured_monotonic"],
                               "time_source": "snapshot_monotonic", "scene_sha256": self.scene_sha256,
                               "scene_hash_scope": "root USD file only; referenced assets are not attested",
                               "scene_state": copy.deepcopy(state), "scene_state_sha256": digest,
                               "semantic_labels": semantic_labels,
                               "render_reference": {"ordinal": self._ordinal,
                                   "sensor_start_s": start, "sensor_end_s": end,
                                   "step_start_s": step_start, "step_end_s": step_end},
                               "capabilities": {"metric_depth": True, "robot_mask": False,
                                   "payload_mask": False, "physics_step": False,
                                   "body_masks": masks is not None}}
                    frames[c.name] = Frame(color[..., 2::-1].copy(), depth, c.K,
                        t=state["captured_monotonic"], frame_id=state["sequence"],
                        depth_source="sensor", T_base_cam=camera_poses.get(c.name, c.T_base_cam).copy(),
                        capture=capture, prop_masks=masks)
                self._last_state, self._last_digest = state, digest
                self._last_frames = copy.deepcopy(frames)
                return frames
            except Exception as e:
                # A partially published stage or failed readback is not retryable
                # as the previous packet with a newly minted capture timestamp.
                self._poisoned = True
                self._last_frames = None
                raise OvrtxError(f"OVRTX frame transaction failed: {e}") from e

    def _cleanup(self):
        errors = []
        def run(label, fn):
            try:
                fn()
            except Exception as e:
                errors.append(f"{label}: {e}")
        for _, path_list, query in reversed(self._queries):
            run("query", lambda q=query: q.release().wait())
            run("path_list", lambda p=path_list: self._dictionary.destroy_path_list(p))
        self._queries.clear()
        if self._dictionary is not None:
            run("dictionary", self._dictionary.destroy)
            self._dictionary = None
        if self._renderer is not None and self._stage is not None:
            run("detach", self._renderer.detach_ovstage)
        if self._stage is not None:
            run("stage", self._stage.destroy)
            self._stage = None
        if self._renderer is not None:
            run("renderer", self._renderer.destroy)
            self._renderer = None
        return errors

    def close(self):
        with self._lock:
            self._closed = True
            self._last_frames = None
            errors = self._cleanup()
            if errors:
                raise OvrtxError(f"OVRTX cleanup failed: {errors}")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
