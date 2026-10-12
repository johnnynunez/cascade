# SPDX-License-Identifier: Apache-2.0
"""Replay recorded MicroDuck physics through Cascade's OVRTX renderer (no physics stepping, no pose repair).

``build_render_scene`` composes, then flattens, one metre-authored Z-up USD: the episode's course, a floor, studio
lights, and the canonical ``MicroDuck.usd`` (sha256 pinned) per robot. Each of the 16 rigid bodies becomes a FLAT prim
``/World/DuckBodies/<robot>/<body>`` carrying that body's own visual subtree, so every body receives its recorded world
pose directly and no frame depends on a parent's written transform. Course gprims without a material get a
UsdPreviewSurface from their displayColor (or one neutral blue-grey). An empty ``/World/ReplayWriteSentinel`` prim is
written first on every frame: with ovrtx 0.5.1 / ovstage 0.2.1 the first single-prim write query of a renderer only
takes effect on frame 1 (see docs/MICRODUCK_OVRTX_REPLAY.md). ``ReplayPoses`` turns recorded body poses into the
transforms :class:`cascade.sim.ovrtx_renderer.SceneSnapshot` requires.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

MICRODUCK_USD_SHA256 = "feca1ff67d2e4140df48469cbe51fad69886c9447929941fc056a0e668b31235"


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_matrix(p7) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(p7[3:7])
    T[:3, 3] = p7[:3]
    return T


def rigid_inverse(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Scene-frame pose of an OPTICAL camera (+X right, +Y down, +Z forward) at ``eye`` looking at ``target``."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, d, f, eye
    return T


def build_render_scene(out_path: Path, *, course_usd: Path | None, microduck_usd: Path, robots: list[str],
                       floor_color=(0.62, 0.63, 0.66), floor_size=12.0, extra_course_offset=(0.0, 0.0, 0.0)) -> dict:
    """Write a flattened render scene; returns the body prim layout ({robot: {body: prim_path}}, parents, sentinel)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    microduck_usd = Path(microduck_usd).resolve(strict=True)
    digest = hashlib.sha256(microduck_usd.read_bytes()).hexdigest()
    if digest != MICRODUCK_USD_SHA256:
        raise ValueError(f"MicroDuck.usd sha256 {digest} is not the canonical {MICRODUCK_USD_SHA256}")
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, "Z")
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    world = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(world.GetPrim())
    looks = "/World/Looks"
    UsdGeom.Scope.Define(st, looks)

    def material(name, color, roughness=0.75, metallic=0.0):
        m = UsdShade.Material.Define(st, f"{looks}/{name}")
        sh = UsdShade.Shader.Define(st, f"{looks}/{name}/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return m

    s = floor_size / 2
    floor = UsdGeom.Mesh.Define(st, "/World/Floor")
    floor.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(material("FloorMat", floor_color, 0.9))
    if course_usd is not None:
        course = st.DefinePrim("/World/Course", "Xform")
        course.GetReferences().AddReference(str(Path(course_usd).resolve(strict=True)))
        if any(abs(v) > 0 for v in extra_course_offset):
            UsdGeom.Xformable(course).AddTranslateOp().Set(Gf.Vec3d(*extra_course_offset))
    for r in robots:
        prim = st.DefinePrim(f"/World/Ducks/{r}", "Xform")
        prim.GetReferences().AddReference(str(microduck_usd))
    dome = UsdLux.DomeLight.Define(st, "/World/Lights/Dome")
    dome.CreateIntensityAttr(700.0)
    dome.CreateColorAttr(Gf.Vec3f(0.86, 0.89, 0.95))
    key = UsdLux.DistantLight.Define(st, "/World/Lights/Key")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(2.5)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(40.0, 0.0, -30.0))
    flat = st.Flatten()
    out = Usd.Stage.Open(flat)
    # Physics is irrelevant for rendering; deactivate actuator prims, bind preview materials from displayColor.
    n_mat = 0
    for prim in list(out.Traverse()):
        if prim.GetTypeName() == "NewtonActuator":
            prim.SetActive(False)
            continue
        if not prim.IsA(UsdGeom.Gprim) or not str(prim.GetPath()).startswith("/World/Course"):
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        colors = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
        if purpose == "guide":
            continue
        if bound:
            continue
        # Course meshes without displayColor (e.g. the Creative gym asset) get one neutral blue-grey so they stand
        # out from the floor.
        c = tuple(float(v) for v in colors[0]) if colors else (0.42, 0.52, 0.66)
        m = UsdShade.Material.Define(out, f"/World/Looks/Course_{n_mat}")
        sh = UsdShade.Shader.Define(out, f"/World/Looks/Course_{n_mat}/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*c))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(m)
        n_mat += 1
    bodies, parents = {}, {}
    cache = UsdGeom.XformCache()
    layer = out.GetRootLayer()
    for r in robots:
        root = out.GetPrimAtPath(f"/World/Ducks/{r}")
        found = {}
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                found[prim.GetName()] = prim
        if len(found) != 16:
            raise ValueError(f"{r}: expected 16 rigid bodies in MicroDuck.usd, found {len(found)}")
        # One FLAT Xform per body under /World/DuckBodies/<robot>/<body>, carrying that body's own visual subtree
        # (children that are not themselves bodies or joints). Each flat body then receives its recorded WORLD pose
        # directly, so no frame depends on a parent's written transform (one stale write cannot drag a subtree).
        bodies[r] = {}
        for name, prim in found.items():
            dst = Sdf.Path(f"/World/DuckBodies/{r}/{name}")
            UsdGeom.Xform.Define(out, dst)
            for child in prim.GetChildren():
                if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.IsA(UsdPhysics.Joint):
                    continue
                Sdf.CopySpec(layer, child.GetPath(), layer, dst.AppendChild(child.GetName()))
            bodies[r][name] = str(dst)
            parents[str(dst)] = ("static", np.eye(4).tolist())
        out.GetPrimAtPath(f"/World/Ducks/{r}/Geometry").SetActive(False)
    del cache
    # Sacrificial first dynamic prim (empty Xform). Measured with ovrtx 0.5.1 / ovstage 0.2.1: the FIRST single-prim
    # omni:xform write query of an OvrtxRenderer only takes effect on the first frame; later writes to that prim are
    # not observed (whichever body was listed first stayed at its first-frame pose). Writing this empty prim first
    # keeps every robot body live. See docs/MICRODUCK_AGENTIC_DEMOS.md.
    UsdGeom.Xform.Define(out, SENTINEL_PATH)
    out_path = Path(out_path)
    out.GetRootLayer().Export(str(out_path))
    return {"bodies": bodies, "parents": parents, "course_materials": n_mat, "microduck_usd_sha256": digest,
            "sentinel": SENTINEL_PATH}


SENTINEL_PATH = "/World/ReplayWriteSentinel"


class ReplayPoses:
    """Recorded body world poses -> per-prim transforms (identity parents in the flat layout; nested layouts still
    get parent-relative transforms), plus the identity sentinel written first."""

    def __init__(self, layout: dict, body_names: list[str], robot_names: list[str]):
        self.layout = layout
        self.body_index = {n: i for i, n in enumerate(body_names)}
        self.robot_names = list(robot_names)
        missing = [n for r in robot_names for n in layout["bodies"][r] if n not in self.body_index]
        if missing:
            raise ValueError(f"recording lacks bodies {sorted(set(missing))}")
        bodies = [p for r in robot_names for p in layout["bodies"][r].values()]
        self.sentinel = layout.get("sentinel")
        self.paths = ([self.sentinel] if self.sentinel else []) + bodies
        # parent-before-child order for local transform computation
        self.order = sorted(bodies, key=lambda p: p.count("/"))

    def transforms(self, body_pose_frame: np.ndarray) -> dict:
        """body_pose_frame: (robots, bodies, 7) [x y z qx qy qz qw] -> {prim_path: local 4x4}."""
        world = {}
        for ri, r in enumerate(self.robot_names):
            for name, path in self.layout["bodies"][r].items():
                world[path] = pose_matrix(body_pose_frame[ri, self.body_index[name]])
        local = {}
        for path in self.order:
            kind, ref = self.layout["parents"][path]
            parent_world = world[ref] if kind == "body" else np.asarray(ref)
            L = rigid_inverse(parent_world) @ world[path] if kind == "body" else np.linalg.inv(parent_world) @ world[path]
            # Re-orthonormalize (float32 recordings) so the renderer's rigid-transform check holds.
            u, _, vt = np.linalg.svd(L[:3, :3])
            R = u @ vt
            if np.linalg.det(R) < 0:
                u[:, -1] *= -1
                R = u @ vt
            L[:3, :3] = R
            L[3] = (0, 0, 0, 1)
            local[path] = L
        if self.sentinel:
            local[self.sentinel] = np.eye(4)
        return local


__all__ = ["build_render_scene", "ReplayPoses", "look_at", "pose_matrix", "MICRODUCK_USD_SHA256"]
