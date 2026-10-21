"""Newton (MuJoCo-Warp) compile fix for Isaac Lab-Arena's Galileo G1 loco-manipulation scene.

Runs INSIDE Arena's Python environment: ``scripts/arena_locomanip_bridge.py`` imports it before the
environment is built when the bridge runs with ``--presets newton`` (Arena 0.3.1, Newton 1.5.2,
MuJoCo 3.11). It changes no contact, friction or controller setting: it only lets MuJoCo compile
the static room's placeholder meshes, which Newton's own collision pipeline does not use for
contacts (Arena's Newton preset sets ``use_mujoco_contacts=False``), and applies MuJoCo's own
``balanceinertia`` remedy to a body whose inertia violates A + B >= C by rounding (logged).

MuJoCo-Warp solver build fails on the Galileo background (static room shell) with errors like

    ValueError: Error: mesh volume is too small: .../Structure/doors/door_1/BL_Door___..._x_133_67 .
                Try setting inertia to shell
    ValueError: Error: eigenvalue of mesh inertia must be positive: .../Structure/walls/POLYFACE_4085_27

Newton exports every collision mesh to an mjSpec only to keep stable geom ids: Arena's Newton preset uses
``use_mujoco_contacts=False`` (Newton's own collision pipeline generates all contacts from Newton's shape data),
and every body with mass gets an explicit inertia. For geoms on the static world body the MuJoCo mesh asset's
volume/inertia is never used, but MuJoCo still validates it at compile time and rejects zero-thickness wall/door
panels.

Fix (only for meshes whose MuJoCo copy is degenerate, i.e. thinner than 1e-4 of their size, or that fail the
compile; bodies with mass get explicit inertia from Newton, so a geom mesh's own inertia is unused): append a copy of the
vertices offset by 1 mm along the panel normal (and along the second-thinnest axis for line-like slivers) and
use ``inertia="convex"``. Newton's own collision geometry, every dynamic body (robot, box, bin) and every mesh
that already compiles are untouched. All modified meshes are logged.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

_ORIG_COMPILE = mujoco.MjSpec.compile
_PAT = re.compile(r"Error: (mesh volume is too small|eigenvalue of mesh inertia must be positive|"
                  r"mesh '.*?' has coplanar vertices.*?|.*?mesh.*?): ?(.*?)(?: \. Try.*)?\nElement name '(.+?)'",
                  re.S)
THICKEN_M = 1.0e-3
PATCHED: list[str] = []


def _thicken(mesh) -> bool:
    v = np.asarray(mesh.uservert, dtype=np.float64).reshape(-1, 3)
    if len(v) < 3:
        return False
    c = v - v.mean(axis=0)
    _, s, vh = np.linalg.svd(c, full_matrices=True)
    ext = np.ptp(c @ vh.T, axis=0)  # extents along principal axes (descending variance)
    add = [v + vh[2] * THICKEN_M]
    if ext[1] < THICKEN_M:  # line-like sliver: give it a second dimension as well
        add.append(v + vh[1] * THICKEN_M)
        add.append(v + (vh[1] + vh[2]) * THICKEN_M)
    mesh.uservert = np.concatenate([v] + add).flatten().tolist()
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    return True


def _world_mesh_names(spec) -> set[str]:
    return {g.meshname for g in spec.worldbody.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname}


def _is_degenerate(mesh) -> bool:
    v = np.asarray(mesh.uservert, dtype=np.float64).reshape(-1, 3)
    if len(v) < 4:
        return True
    c = v - v.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    ext = float(np.ptp(c, axis=0).max())
    # thinnest principal spread below 1e-5 of the size (or below 1e-6 m): MuJoCo rejects volume/inertia
    return s[-1] / np.sqrt(len(v)) < max(1e-4 * ext, 1e-5)


ARENA_BYTES = 2 << 30


def _compile_with_static_mesh_fix(self, *args, **kwargs):
    # The Galileo room has thousands of collision geoms (CoACD pieces); Newton's one-off CPU mj_forward after the
    # compile then overflows MuJoCo's default CPU arena ("mj_stackAlloc: out of memory, stack overflow").
    # Enlarge only the CPU-side MjData arena (host RAM); no effect on the GPU solver or on the dynamics.
    if self.memory < ARENA_BYTES:
        self.memory = ARENA_BYTES
    world_meshes = _world_mesh_names(self)
    pre = 0
    for mesh in self.meshes:
        if mesh.name in world_meshes and _is_degenerate(mesh) and _thicken(mesh):
            PATCHED.append(mesh.name)
            pre += 1
    if pre:
        n_world = sum(1 for n in PATCHED if n in world_meshes)
        print(f"[arena_newton_compile_fix] thickened {pre} degenerate placeholder meshes ({n_world} on the static world body) by "
              f"{THICKEN_M * 1e3:.0f} mm for the MuJoCo compile", flush=True)
    for _ in range(300):
        try:
            return _ORIG_COMPILE(self, *args, **kwargs)
        except ValueError as exc:
            msg = str(exc)
            if "use 'balanceinertia' to fix" in msg and not self.compiler.balanceinertia:
                # A static task-asset body (e.g. the Galileo table) whose inertia violates A + B >= C by rounding;
                # MuJoCo's own remedy. Only bodies that violate the inequality are adjusted.
                self.compiler.balanceinertia = True
                print(f"[arena_newton_compile_fix] {msg.splitlines()[0][:80]} ({msg.splitlines()[-1][:120]}) -> "
                      "compiler.balanceinertia=True", flush=True)
                continue
            m = re.search(r"Element name '(.+?)'", msg)
            name = m.group(1) if m else None
            if name is None or name in PATCHED or "mesh" not in msg:
                raise
            mesh = self.mesh(name)
            if mesh is None or not _thicken(mesh):
                raise
            PATCHED.append(name)
            where = "world body" if name in world_meshes else "non-world body"
            print(f"[arena_newton_compile_fix] placeholder mesh '{name}' ({where}) failed MuJoCo compile "
                  f"({msg.splitlines()[0][:90]}); thickened by {THICKEN_M * 1e3:.0f} mm", flush=True)
    raise RuntimeError("arena_newton_compile_fix: too many degenerate meshes")


mujoco.MjSpec.compile = _compile_with_static_mesh_fix
print("[arena_newton_compile_fix] installed MjSpec.compile fix for degenerate static placeholder meshes", flush=True)
