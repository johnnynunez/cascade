#!/usr/bin/env python3
"""Isolated physics-divergence diagnostic for the RS arm asset.

Loads ONLY the arm (no props, no cameras), steps physics one tick at a time,
and reports the first step where any DOF goes non-finite -- plus the joint
state, drive targets and per-link contact census right before divergence.

    export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-$(uname -m)/release
    $ISAACSIM_PATH/python.sh scripts/diag_physics.py --engine physx --device cpu
"""
from __future__ import annotations

import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--engine", default="physx", choices=["newton", "physx"])
p.add_argument("--device", default="cpu")
p.add_argument("--usd", default=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "usd", "RS-rebot-dev-arm", "RS-rebot-dev-arm.usda"))
p.add_argument("--prim", default="/tn__00armrs_asmv3_hJ6D/Geometry/base_link")
p.add_argument("--steps", type=int, default=120)
p.add_argument("--self-collision", action="store_true",
               help="leave self-collision ON (default: force OFF via PhysX schema)")
p.add_argument("--command-home", action="store_true",
               help="command the bridge HOME_Q=[0,1.2,1.2,0,0.75,0] (ILLEGAL j2/j3) each step")
p.add_argument("--add-table", action="store_true",
               help="spawn a static table slab under the arm like the bridge does")
p.add_argument("--filter-arm-table", action="store_true",
               help="filter arm<->table collisions via collision groups (the fix)")
p.add_argument("--add-ycb", action="store_true",
               help="also spawn the 3 YCB SDF props (reproduce dense-contact tunnelling)")
args = p.parse_args()

_release = os.environ.get(
    "ISAACSIM_PATH",
    os.path.expanduser(f"~/Projects/isaac/IsaacSim/_build/linux-{os.uname().machine}/release"),
)
from isaacsim import SimulationApp  # noqa: E402

_kwargs = {}
if args.engine == "newton":
    _kwargs["experience"] = os.path.join(_release, "apps", "isaacsim.exp.full.newton.kit")
app = SimulationApp({"headless": True}, **_kwargs)

import numpy as np  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

SimulationManager.switch_physics_engine(args.engine)
stage_utils.open_stage(args.usd)
while stage_utils.is_stage_loading():
    app.update()

import omni.usd  # noqa: E402
from pxr import PhysxSchema, Sdf, UsdPhysics  # noqa: E402

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath(args.prim)

if not args.self_collision:
    px = PhysxSchema.PhysxArticulationAPI.Apply(root)
    px.CreateEnabledSelfCollisionsAttr(False)
    print(f"[diag] PhysxArticulationAPI applied, self-collision OFF", flush=True)

# clamp newton solver caps like the bridge does
for prim in stage.Traverse():
    a = prim.GetAttribute("newton:solver:nconmax")
    if a and a.IsValid() and a.HasValue() and a.Get() and a.Get() > 4600:
        a.Set(4600)

for _ in range(30):
    app.update()

# Optionally spawn a static table slab exactly like the bridge (boundingCube
# collider, bound at z just under the arm base) to test if the arm-vs-table
# contact is what diverges the solver.
if args.add_table:
    from pxr import Gf, UsdGeom, UsdShade  # noqa: E402

    _FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6),
              (0, 2, 6, 4), (1, 5, 7, 3)]

    def _cube(path, pos, size, dynamic, mass=0.05):
        mesh = UsdGeom.Mesh.Define(stage, path)
        hx, hy, hz = [s / 2 for s in size]
        pts = [Gf.Vec3f(sx * hx, sy * hy, sz * hz)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([i for f in _FACES for i in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        ext = Gf.Vec3f(hx, hy, hz)
        mesh.CreateExtentAttr([-ext, ext])
        xf = UsdGeom.Xformable(mesh.GetPrim())
        xf.ClearXformOpOrder()
        m = Gf.Matrix4d(1.0)
        m.SetTranslateOnly(Gf.Vec3d(*pos))
        xf.AddTransformOp().Set(m)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mc.CreateApproximationAttr("boundingCube")
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
            UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(mass)
        return mesh

    _cube("/World_Props/table", (0.30, 0.0, -0.015), (0.9, 0.9, 0.03), dynamic=False)
    _cube("/World_Props/pink_cube", (0.28, 0.08, 0.026), (0.05, 0.05, 0.05), dynamic=True)
    print("[diag] spawned static table + 1 dynamic cube", flush=True)

    if args.add_ycb:
        # Reproduce the YCB SDF colliders that the bridge spawns -- the
        # "SDF Mesh not watertight" warning + dense SDF-vs-boundingCube
        # contacts are the suspected cause of props tunnelling the table.
        from isaacsim.storage.native import get_assets_root_path  # noqa: E402

        _assets = get_assets_root_path()
        if _assets:
            from pxr import PhysxSchema, Sdf, Usd  # noqa: E402

            for _n, _f, _p in [("banana", "011_banana.usd", (0.24, 0.14, 0.018)),
                               ("cracker_box", "003_cracker_box.usd", (0.36, -0.02, 0.107)),
                               ("soup_can", "005_tomato_soup_can.usd", (0.20, -0.12, 0.052))]:
                pp = f"/World_Props/{_n}"
                pr = stage.DefinePrim(pp, "Xform")
                pr.GetReferences().AddReference(
                    f"{_assets}/Isaac/Props/YCB/Axis_Aligned/{_f}")
                from pxr import Gf as _Gf
                xf = UsdGeom.Xformable(pr)
                xf.ClearXformOpOrder()
                rot = _Gf.Matrix4d().SetRotate(_Gf.Quatd(0.5, 0.5, 0.5, 0.5))
                trs = _Gf.Matrix4d(1.0)
                trs.SetTranslateOnly(_Gf.Vec3d(*_p))
                xf.AddTransformOp().Set(rot * trs)
                if not pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI.Apply(pr)
                UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.1)
                for desc in Usd.PrimRange(pr):
                    if desc.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(desc)
                        mc2 = UsdPhysics.MeshCollisionAPI.Apply(desc)
                        mc2.CreateApproximationAttr("sdf")
                        try:
                            s = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(desc)
                            s.CreateSdfResolutionAttr(256)
                        except Exception:
                            pass
            print("[diag] spawned 3 YCB SDF props", flush=True)

    if args.filter_arm_table:
        # The arm base_link collider sits at z=0 and the table top is also at
        # z=0 -> deep static-vs-static interpenetration detonates the solver.
        # Put the arm and the static furniture (table/bin walls) in collision
        # groups that DON'T collide with each other; props still collide with
        # both. This is the correct fix: two static bodies never need mutual
        # contacts, and the arm should rest ON the table plane by convention,
        # not be pushed out of it.
        from pxr import Usd as _Usd  # noqa: E402

        arm_grp = UsdPhysics.CollisionGroup.Define(stage, "/World_Props/arm_group")
        furn_grp = UsdPhysics.CollisionGroup.Define(stage, "/World_Props/furniture_group")
        # arm_group excludes furniture_group (mutual)
        arm_grp.CreateFilteredGroupsRel().AddTarget(furn_grp.GetPath())
        # populate members
        arm_col = arm_grp.GetCollidersCollectionAPI()
        arm_col.GetIncludesRel().AddTarget(args.prim)
        furn_col = furn_grp.GetCollidersCollectionAPI()
        furn_col.GetIncludesRel().AddTarget("/World_Props/table")
        print("[diag] arm<->table collision filtered (two static bodies)", flush=True)

try:
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device=args.device)
    print(f"[diag] device={args.device}", flush=True)
except Exception as e:
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    print(f"[diag] device fallback cpu ({e})", flush=True)

from isaacsim.core.experimental.prims import Articulation  # noqa: E402

app_utils.play(commit=True)
for _ in range(5):
    app.update()

art = Articulation(args.prim)
print(f"[diag] engine={SimulationManager.get_active_physics_engine()} dofs={art.num_dofs}", flush=True)
names = list(art.dof_names)
lo, hi = [x.numpy()[0] for x in art.get_dof_limits()]
q0 = art.get_dof_positions().numpy()[0]
print(f"[diag] names={names}", flush=True)
print(f"[diag] limits lo={np.round(lo,3).tolist()}", flush=True)
print(f"[diag] limits hi={np.round(hi,3).tolist()}", flush=True)
print(f"[diag] q0(authored)={np.round(q0,4).tolist()} finite={bool(np.all(np.isfinite(q0)))}", flush=True)

# DO NOT command any target -- let it settle under gravity from the authored pose
HOME_Q = [0.0, 1.2, 1.2, 0.0, 0.75, 0.0]  # bridge's LOCAL-convention home
arm_idx = [names.index(f"joint{i}") for i in range(1, 7)]
diverged_at = None
for step in range(args.steps):
    if args.command_home:
        tgt = art.get_dof_position_targets().numpy()[0].astype(np.float32).copy()
        for k, i in enumerate(arm_idx):
            tgt[i] = HOME_Q[k]
        art.set_dof_position_targets(tgt.reshape(1, -1))
    app.update()
    q = art.get_dof_positions().numpy()[0]
    dq = art.get_dof_velocities().numpy()[0]
    if not np.all(np.isfinite(q)):
        diverged_at = step
        print(f"[diag] *** DIVERGED at step {step} ***", flush=True)
        break
    if step < 8 or step % 20 == 0:
        print(f"[diag] step {step:3d} q={np.round(q,4).tolist()} "
              f"|dq|max={float(np.max(np.abs(dq))):.4f}", flush=True)

if diverged_at is None:
    q = art.get_dof_positions().numpy()[0]
    print(f"[diag] SURVIVED {args.steps} steps. final q={np.round(q,4).tolist()}", flush=True)
    print(f"[diag] finite={bool(np.all(np.isfinite(q)))}", flush=True)
    if args.add_table:
        from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402

        rp = RigidPrim("/World_Props/pink_cube")
        cp = rp.get_world_poses()[0].numpy().reshape(-1)[:3]
        rested = abs(cp[2]) < 0.2
        print(f"[diag] pink_cube pos=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}) "
              f"{'RESTS ON TABLE' if rested else '<-- FELL THROUGH TABLE'}", flush=True)

app_utils.stop()
app.close()
