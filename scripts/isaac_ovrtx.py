"""Authoritative Isaac physics snapshots for an isolated optional OVRTX owner.

All reads run on the bridge main thread between physics updates. The private
render USD resets each body's xform stack, so nested links receive WORLD
poses without composing their ancestors twice. The live USD is never edited.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np

from cascade.sim.ovrtx_process import OvrtxProcess
from cascade.sim.ovrtx_renderer import CameraSpec, SceneSnapshot, _transform, _snapshot


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def pose_matrix(position, quaternion):
    p, q = np.asarray(position, float), np.asarray(quaternion, float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Malformed physics body pose")
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-5, rtol=0):
        raise ValueError("Physics body quaternion is not normalized")
    w, x, y, z = q / np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    T[:3, 3] = p
    return _transform(T, "physics body")


def encode_masks(masks, robot_id, contact, t):
    if contact["error"] is not None:
        raise ValueError("Contact state unavailable: " + contact["error"])
    robot_paths = [p for p in masks if p == robot_id or p.startswith(robot_id + "/")]
    if not robot_paths:
        raise ValueError("Renderer body inventory omits robot")
    if set(contact["paths"]) - set(masks) or set(contact["scene_prop_paths"]) - set(masks):
        raise ValueError("Renderer body inventory omits captured physical props")
    shape = masks[robot_paths[0]].shape
    payload = np.zeros(shape, bool)
    for path in contact["paths"]:
        payload |= masks[path]
    robot = payload.copy()
    for path in robot_paths:
        robot |= masks[path]

    def encode(mask):
        return base64.b64encode(zlib.compress(mask.astype(np.uint8).tobytes(), 3)).decode()

    result = {"version": 1, "robot_id": robot_id, "t": t, "shape": list(shape),
              "encoding": "zlib-u8-base64", "data": encode(robot),
              "contact_paths": list(contact["paths"])}
    if contact["tracking"]:
        result.update(payload_data=encode(payload),
                      prop_data={p: encode(masks[p]) for p in contact["scene_prop_paths"]})
    return result


def bridge_packet(frame, snapshot, *, robot_id, base_z):
    """Translate the renderer packet without substituting later physics state."""
    state = frame.capture["scene_state"]
    expected, digest, _, _ = _snapshot(snapshot, snapshot.transforms, snapshot.cameras)
    if state != expected or frame.capture["scene_state_sha256"] != digest:
        raise ValueError("OVRTX output does not belong to the supplied physical snapshot")
    meta = state["metadata"]
    proprioception = meta["proprioception"]
    if not isinstance(proprioception, dict):
        raise ValueError("Physical joint capture unavailable")
    T = frame.T_base_cam.copy()
    T[2, 3] -= base_z
    ref = {"version": 1, "source": "ovrtx_snapshot", "product": "/CascadeRender/" + frame.capture["camera"],
           "producer_epoch": snapshot.epoch, "history_physics_step": snapshot.sequence,
           "history_simulation_time": snapshot.sim_time,
           "snapshot_started_monotonic": snapshot.captured_monotonic,
           "snapshot_finished_monotonic": meta["snapshot_finished_monotonic"],
           "snapshot_sha256": frame.capture["scene_state_sha256"],
           "proprioception_sha256": _json_hash(proprioception),
           "scene_sha256": frame.capture["scene_sha256"],
           "renderer_epoch": frame.capture["producer_epoch"],
           "renderer": copy.deepcopy(frame.capture["render_reference"])}
    result = {"ok": True, "width": frame.rgb.shape[1], "height": frame.rgb.shape[0],
              "K": frame.K.tolist(), "t": snapshot.captured_monotonic,
              "T_base_cam": T.tolist(), "proprioception": copy.deepcopy(proprioception),
              "render_reference": ref, "renderer": "ovrtx"}
    if frame.prop_masks is not None:
        try:
            result["robot_pixel_mask"] = encode_masks(frame.prop_masks, robot_id,
                                                     meta["contact_state"], snapshot.captured_monotonic)
        except ValueError as exc:
            result["robot_pixel_mask_error"] = str(exc)
    return result


class PhysicsBodies:
    """Read the existing tensor scene without authoring any live USD schema."""
    def __init__(self, paths):
        self.paths = tuple(paths)
        self._simulation = self._view = None
        if not self.is_physics_tensor_entity_valid():
            raise ValueError("Physical rigid-body tensor view unavailable")

    def is_physics_tensor_entity_valid(self):
        from isaacsim.core.simulation_manager import SimulationManager
        simulation = SimulationManager._physics_sim_view__warp
        if simulation is None or not simulation.is_valid:
            return False
        if simulation is not self._simulation:
            view = simulation.create_rigid_body_view(list(self.paths))
            if view is None or not view.check() or set(view.prim_paths) != set(self.paths):
                return False
            self._simulation, self._view = simulation, view
        return self._view is not None and self._view.check()

    def get_world_poses(self):
        if not self.is_physics_tensor_entity_valid():
            raise ValueError("Physical tensor view lost during snapshot")
        data = self._view.get_transforms().numpy()
        order = {path: i for i, path in enumerate(self._view.prim_paths)}
        data = data[[order[p] for p in self.paths]]
        if data.shape != (len(self.paths), 7):
            raise ValueError("Physical tensor pose inventory changed")
        return data[:, :3].copy(), data[:, [6, 3, 4, 5]].copy()  # tensor xyzw -> wxyz


class IsaacOvrtx:
    def __init__(self, stage, cameras, *, python, output, robot_id, base_z, width, height,
                 wrist_link=None, wrist_mount=None, device=0, frame_timeout_s=15.,
                 process_factory=OvrtxProcess):
        from pxr import Usd, UsdGeom, UsdPhysics

        if UsdGeom.GetStageMetersPerUnit(stage) != 1. or str(UsdGeom.GetStageUpAxis(stage)) != "Z":
            raise ValueError("Live OVRTX producer requires a metre-authored Z-up stage")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.robot_id, self.base_z = robot_id, base_z
        self.wrist_link, self.wrist_mount = wrist_link, wrist_mount
        self._python, self._factory = python, process_factory
        self._frame_timeout_s = frame_timeout_s
        self._owner = None
        self._epoch = None
        self.latest = None
        self.error = None
        # Kit's PhysX notice handlers can react to same-path edits even on a
        # second Usd.Stage. Only read/flatten here; stronger reset/instance
        # opinions are authored by the isolated renderer process.
        prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
        paths = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not paths or robot_id not in paths:
            raise ValueError("OVRTX scene has no complete physical robot inventory")
        self._bodies = PhysicsBodies(paths)
        self.paths = tuple(self._bodies.paths)
        if set(self.paths) != set(paths):
            raise ValueError("Physics tensor inventory differs from render scene")
        cache = UsdGeom.XformCache()
        for path in self.paths:
            prim = stage.GetPrimAtPath(path)
            # Reject inherited nonrigid scale/shear rather than altering shape.
            _transform(np.asarray(cache.GetLocalToWorldTransform(prim)).T, path)
        semantics = {}
        for prim in prims:
            if prim.IsA(UsdGeom.Gprim):
                path = str(prim.GetPath())
                owners = [p for p in self.paths if path == p or path.startswith(p + "/")]
                if owners:
                    semantics[path] = max(owners, key=len)
        scene = self.output / "render-scene.usda"
        if scene.exists():
            raise ValueError("OVRTX output already contains a scene; use a fresh run directory")
        stage.Flatten().Export(str(scene))
        self.specs = []
        optical = np.diag([1., -1., -1., 1.])
        for name, (_, K) in cameras.items():
            T = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World_Cams/" + name))).T @ optical
            self.specs.append(CameraSpec(name, width, height, float(K[0][0]), float(K[0][0]), T))
        self.config = dict(scene=str(scene), cameras=self.specs, dynamic_paths=self.paths,
                           semantic_paths=semantics, world_paths=self.paths,
                           instance_paths=[str(p.GetPath()) for p in prims if p.IsInstance()], device=device)
        manifest = {"version": 1, "scene": str(scene), "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                    "hash_scope": "flattened USD only; external textures/material files are not attested",
                    "source": stage.GetRootLayer().identifier, "dynamic_paths": list(self.paths),
                    "semantic_paths": semantics, "robot_id": robot_id, "base_z_m": base_z}
        (self.output / "scene-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def capture(self, *, epoch, physics_step, simulation_time, started, payload):
        repeated = False
        if self.latest is not None and self.latest.epoch == epoch:
            repeated = physics_step == self.latest.sequence
            if physics_step < self.latest.sequence or (not repeated and simulation_time <= self.latest.sim_time):
                raise ValueError("OVRTX physical clock regressed")
        if not self._bodies.is_physics_tensor_entity_valid():
            raise ValueError("OVRTX physical body tensor view unavailable; USD fallback forbidden")
        positions, quaternions = self._bodies.get_world_poses()
        positions = positions.numpy() if hasattr(positions, "numpy") else np.asarray(positions)
        quaternions = quaternions.numpy() if hasattr(quaternions, "numpy") else np.asarray(quaternions)
        transforms = {path: pose_matrix(p, q) for path, p, q in
                      zip(self.paths, positions, quaternions, strict=True)}
        cameras = {c.name: c.T_base_cam.copy() for c in self.specs}
        if "wrist" in cameras:
            if self.wrist_link not in transforms or self.wrist_mount is None:
                raise ValueError("OVRTX wrist camera has no captured physical mount")
            cameras["wrist"] = (transforms[self.wrist_link] @ self.wrist_mount
                                @ np.diag([1., -1., -1., 1.]))
        metadata = copy.deepcopy(payload)
        if "wrist" in cameras:
            wrist_base = cameras["wrist"].copy()
            wrist_base[2, 3] -= self.base_z
            metadata["wrist_T"] = wrist_base.tolist()
        metadata["snapshot_finished_monotonic"] = time.monotonic()
        metadata["transform_frame"] = "world_metres_reset_stack"
        if not isinstance(metadata["proprioception"], dict):
            raise ValueError("OVRTX physical joint snapshot unavailable")
        if repeated:
            # Compare every physical measurement while retaining the first
            # capture anchor. A pose edit without a physics step is ambiguous.
            started = self.latest.captured_monotonic
            metadata["proprioception"]["t"] = started
            metadata["snapshot_finished_monotonic"] = self.latest.metadata["snapshot_finished_monotonic"]
        snapshot = SceneSnapshot("isaac_physics_tensor", epoch, physics_step, simulation_time,
                                 started, transforms, cameras, metadata)
        normalized = _snapshot(snapshot, self.paths, cameras)[0]
        if repeated:
            if normalized != _snapshot(self.latest, self.paths, cameras)[0]:
                raise ValueError("Contradictory OVRTX state at the same physical step")
            return
        self.latest = snapshot
        self.error = None

    def render(self):
        if self.error or self.latest is None:
            raise ValueError(self.error or "OVRTX physical snapshot unavailable")
        if self._epoch != self.latest.epoch:
            self.close()
            self._owner = self._factory(self._python, self.config, frame_timeout_s=self._frame_timeout_s)
            self._epoch = self.latest.epoch
        try:
            frames = self._owner.render(self.latest)
            return {name: (bridge_packet(frame, self.latest, robot_id=self.robot_id, base_z=self.base_z),
                           frame.rgb[..., ::-1].copy(), frame.depth_m) for name, frame in frames.items()}
        except Exception as exc:
            self.error = str(exc)
            raise

    def close(self):
        if self._owner is not None:
            self._owner.close()
            self._owner = None
        self._epoch = None
