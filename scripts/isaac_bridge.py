#!/usr/bin/env python3
"""Isaac Sim side of the wrc_demo bridge -- Isaac Sim 6.0 (develop, Newton).

Run with Isaac Sim's Python:

    export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-$(uname -m)/release
    $ISAACSIM_PATH/python.sh scripts/isaac_bridge.py \
        --usd /home/spark/Projects/isaac/00-arm-rs_asm-v3-plus/00-arm-rs_asm-v3-plus.usda

Opens the gain-tuned reBot RS asset, adds a tabletop + colored props (incl.
a PINK cube) + two RTX cameras, and serves the newline-JSON protocol from
wrc_demo.sim.bridge_client so the FULL agentic demo (reflex tier, livestream
dashboard, Hermes MCP) runs against the sim:

    wrc-demo --cameras isaac,isaac_side --arm isaac --interactive
    # or MCP:  WRC_CAMERAS=isaac,isaac_side WRC_ARM=isaac wrc-mcp

Written against the isaacsim.core.experimental API (the classic
isaacsim.core.api/prims/sensors modules do NOT exist in the 6.0 develop
build). Patterns follow the validated gain-tuner scripts in
~/Projects/isaac/00-arm-rs_asm-v3-plus/ (Articulation AFTER play, Newton
needs the isaacsim.exp.full.newton.kit experience, solver caps persisted on
the asset).

Gripper protocol unit: FRACTION 0..1 (0 = closed, 1 = open); the bridge maps
it to the two prismatic finger joints' limit ranges. configs/arms/isaac.yaml
declares open_pos: 1.0 / closed_pos: 0.0 accordingly.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import socketserver
import sys
import threading
import time
import zlib

DEFAULT_USD = "/home/spark/Projects/isaac/00-arm-rs_asm-v3-plus/00-arm-rs_asm-v3-plus.usda"
DEFAULT_PRIM = "/tn__00armrs_asmv3_hJ6D/Geometry/base_link"

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--usd", default=DEFAULT_USD, help="reBot RS scene USD (gain-tuned asset)")
p.add_argument("--prim", default=DEFAULT_PRIM, help="articulation root prim path")
p.add_argument("--engine", default="newton", choices=["newton", "physx"])
p.add_argument("--port", type=int, default=8611)
p.add_argument("--gui", action="store_true", help="run with the editor window (default: headless)")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--dt", type=float, default=1.0 / 60.0)
p.add_argument("--cam-every", type=int, default=2, help="refresh camera cache every N sim steps")
args = p.parse_args()

# Demo spawn/ready pose (LOCAL joint convention). NOTE: the straight-up
# pose is blocked on this asset -- drive-travel from q=0 sweeps the
# props and jams, while joint-state authoring or tensor teleports NaN
# the solver (custom fixed-joint stack). The upstream asset gets the
# straight spawn properly via Seeed-Projects/reBot-Isaacsim#9.
HOME_Q = [0.0, 1.2, 1.2, 0.0, 0.75, 0.0]

# SimulationApp must exist before ANY isaacsim/omni import.
#
# Engine <-> experience coupling (validated on this rig 2026-07-18):
# - newton REQUIRES the newton kit experience (enabling the ext post-boot
#   does not create the tensor backend -- gain-tuner gotcha), BUT offscreen
#   render products (CameraSensor/replicator) return garbage under it; only
#   viewport capture works. Fine for physics work, useless for RGB-D.
# - physx runs under the DEFAULT experience where the RTX sensor stack is
#   the tested path. The asset's colliders are engine-agreement validated
#   (gain tuner 8/8), so the perception demo uses physx by default.
from isaacsim import SimulationApp  # noqa: E402

_release = os.environ.get(
    "ISAACSIM_PATH",
    os.path.expanduser("~/Projects/isaac/IsaacSim/_build/linux-aarch64/release"),
)
_kwargs = {}
if args.engine == "newton":
    _kwargs["experience"] = os.path.join(_release, "apps", "isaacsim.exp.full.newton.kit")
app = SimulationApp(
    {"headless": not args.gui, "renderer": "RayTracedLighting",
     "width": args.width, "height": args.height},
    **_kwargs,
)

import numpy as np  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

SimulationManager.switch_physics_engine(args.engine)
stage_utils.open_stage(args.usd)
while stage_utils.is_stage_loading():
    app.update()

import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

stage = omni.usd.get_context().get_stage()

# The gain-tuned asset persists newton:solver:nconmax=8192, sized for the
# patched newton_stage build. Stock develop computes rigid_contact_max from
# geometry (~4.7k here) and errors out EVERY step on the mismatch, killing
# contacts. Clamp the persisted cap IMMEDIATELY after load, before any
# update tick lets the physics parser read it.
# PhysX articulation self-collision must be OFF for this asset (it ships
# newton:selfCollisionEnabled=0 but no PhysX equivalent): the straight-up
# pose puts adjacent elbow colliders in deep penetration -> NaN explosion.
_root = stage.GetPrimAtPath(args.prim)
if _root:
    _root.CreateAttribute("physxArticulation:enabledSelfCollisions",
                          Sdf.ValueTypeNames.Bool).Set(False)
    print("[bridge] articulation self-collision disabled (PhysX)", flush=True)

for prim in stage.Traverse():
    attr = prim.GetAttribute("newton:solver:nconmax")
    if attr and attr.IsValid() and attr.HasValue():
        old = attr.Get()
        if old and old > 4000:
            attr.Set(4000)
            print(f"[bridge] clamped {prim.GetPath()} newton:solver:nconmax {old} -> 4000",
                  flush=True)

for _ in range(30):
    app.update()

# Stage units: author everything in stage units so a cm-authored asset does
# not end up with the camera inside the base link.
MPU = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
U = 1.0 / MPU  # multiply meters by U to get stage units
UP_AXIS = UsdGeom.GetStageUpAxis(stage)
print(f"[bridge] metersPerUnit={MPU} upAxis={UP_AXIS}", flush=True)
print(f"[bridge] root prims: {[p.GetName() for p in stage.GetPseudoRoot().GetChildren()]}",
      flush=True)
scene_prim = stage.GetPrimAtPath("/PhysicsScene")
if scene_prim:
    g_dir = scene_prim.GetAttribute("physxScene:gravityDirection")
    g_dir2 = scene_prim.GetAttribute("physics:gravityDirection")
    print(f"[bridge] PhysicsScene gravityDirection: "
          f"{(g_dir.Get() if g_dir else None) or (g_dir2.Get() if g_dir2 else None)}",
          flush=True)

# This asset ships with the arm hoisted in the air (contact-free gravity
# validation). Find where the base actually sits and author the tabletop
# world at THAT height -- wrc_demo only ever sees base-relative geometry,
# so the world offset is invisible to it.
_bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
_robot_range = _bb.ComputeWorldBound(stage.GetPrimAtPath(args.prim)).ComputeAlignedRange()
BASE_Z = float(_robot_range.GetMin()[2]) * MPU  # meters
print(f"[bridge] robot base plane at world z={BASE_Z:.3f} m; "
      f"authoring the tabletop there", flush=True)

# GPU physics: device="cpu" (inherited from the gain-tuner precision
# scripts) runs MuJoCo-Warp on the CPU and stutters badly with the full
# booth scene while RTX renders on the GPU.
try:
    SimulationManager.setup_simulation(dt=args.dt, device="cuda:0")
except Exception:
    SimulationManager.setup_simulation(dt=args.dt, device="cpu")
    print("[bridge] WARNING: GPU physics unavailable, using CPU", flush=True)
else:
    print("[bridge] physics device: cuda:0 (GPU)", flush=True)

# ── lights ───────────────────────────────────────────────────────────────
# Moderate intensities: overexposure washes saturated albedos to pastel,
# which blinds both the detector and the HSV color tagger.
dome = UsdLux.DomeLight.Define(stage, "/World_Lights/Dome")
dome.CreateIntensityAttr(350.0)
sun = UsdLux.DistantLight.Define(stage, "/World_Lights/Sun")
sun.CreateIntensityAttr(1200.0)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-45, 30, 0))


# One shared physics material: without friction/restitution authored,
# Newton props jitter and creep across the table indefinitely.
from pxr import UsdShade  # noqa: E402

_pmat = UsdShade.Material.Define(stage, "/World_Props/physics_material")
_pmat_api = UsdPhysics.MaterialAPI.Apply(_pmat.GetPrim())
_pmat_api.CreateStaticFrictionAttr(0.9)
_pmat_api.CreateDynamicFrictionAttr(0.8)
_pmat_api.CreateRestitutionAttr(0.0)


def _bind_pmat(prim):
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(
        _pmat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


# ── tabletop + props (base frame: robot base at origin ON the table) ─────
_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6),
          (0, 2, 6, 4), (1, 5, 7, 3)]


def _cube(path, pos, size, color, dynamic, mass=0.05):
    """Axis-aligned box MESH, sized in the vertices themselves. NO xform
    scale ops: hydra on this build renders translate/orient but silently
    ignores xformOp:scale (physics honors it -- 5 cm collision boxes were
    rendering as 1 m slabs)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    hx, hy, hz = [s * U / 2 for s in size]
    pts = [Gf.Vec3f(sx * hx, sy * hy, sz * hz)
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in _FACES for i in f])
    # Explicit: the default catmullClark subdivision renders these boxes as
    # inflated blobs -- the depth cloud then overstates the object and the
    # planned grasp closes on air above the real (collider-sized) cube.
    mesh.CreateSubdivisionSchemeAttr("none")
    ext = Gf.Vec3f(hx, hy, hz)
    mesh.CreateExtentAttr([-ext, ext])
    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.ClearXformOpOrder()
    world = (pos[0], pos[1], pos[2] + BASE_Z)
    m = Gf.Matrix4d(1.0)
    m.SetTranslateOnly(Gf.Vec3d(*[v * U for v in world]))
    # Matrix op ONLY: this build's renderer honors xformOp:transform but
    # silently drops translate/orient/scale vector ops (cameras -- matrix
    # ops -- always rendered right; everything else floated at the origin).
    xformable.AddTransformOp().Set(m)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    _bind_pmat(mesh.GetPrim())
    if dynamic:
        # Dynamic bodies cannot use raw triangle-mesh collision (PhysX
        # error banner + convexHull fallback); boundingCube is EXACT for
        # these axis-aligned boxes.
        mcol = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mcol.CreateApproximationAttr("boundingCube")
        UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        mapi = UsdPhysics.MassAPI.Apply(mesh.GetPrim())
        mapi.CreateMassAttr(mass)
    return mesh


# Open-top bin centered on the demo DROP ZONE (grasp.drop_zone in
# configs/demo.yaml): destination-less pick_and_place drops INTO it, and
# "put it in the box" resolves it as a named destination. Walls only --
# the table is the floor; static colliders so visitors can't topple it.
_BIN = (0.30, -0.20)
for _i, (_wpos, _wsize) in enumerate([
    ((_BIN[0] + 0.07, _BIN[1], 0.03), (0.01, 0.15, 0.06)),
    ((_BIN[0] - 0.07, _BIN[1], 0.03), (0.01, 0.15, 0.06)),
    ((_BIN[0], _BIN[1] + 0.07, 0.03), (0.15, 0.01, 0.06)),
    ((_BIN[0], _BIN[1] - 0.07, 0.03), (0.15, 0.01, 0.06)),
]):
    _cube(f"/World_Props/bin_wall{_i}", _wpos, _wsize, (0.72, 0.52, 0.22), dynamic=False)

# Table top surface at z=0 (the real base sits on the table -> table_z=0
# in configs/demo.yaml). Slab top edge exactly at z=0.
_cube("/World_Props/table", (0.30, 0.0, -0.015), (0.9, 0.9, 0.03), (0.55, 0.45, 0.35), dynamic=False)
PROPS = [
    ("pink_cube", (0.28, 0.08, 0.026), (0.97, 0.38, 0.56), 0.05),
    ("green_cube", (0.32, -0.10, 0.026), (0.20, 0.65, 0.30), 0.05),
    ("blue_cube", (0.22, -0.02, 0.021), (0.10, 0.35, 0.90), 0.04),
]
for name, pos, rgb, size in PROPS:
    _cube(f"/World_Props/{name}", pos, (size,) * 3, rgb, dynamic=True)

# YCB props from the Isaac asset library: textured REAL objects the
# open-vocab detector actually recognizes (flat-shaded cubes register as
# nothing). Skipped gracefully when the asset CDN is unreachable.
try:
    from isaacsim.storage.native import get_assets_root_path

    _assets = get_assets_root_path()
    if _assets:
        # Spawn AT rest height: dropping groceries onto the table reads
        # as "flying cereal" -- born settled = realistic from frame one.
        YCB = [
            ("banana", "011_banana.usd", (0.24, 0.14, 0.018)),
            ("cracker_box", "003_cracker_box.usd", (0.36, -0.02, 0.107)),
            ("soup_can", "005_tomato_soup_can.usd", (0.20, -0.12, 0.052)),
        ]
        for name, usd_file, pos in YCB:
            prim_path = f"/World_Props/{name}"
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(
                f"{_assets}/Isaac/Props/YCB/Axis_Aligned/{usd_file}"
            )
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            world = (pos[0], pos[1], pos[2] + BASE_Z)
            # These YCB USDs are authored in METERS (verified: a 0.01 scale
            # made them millimetric and they tunneled through the table).
            rot = Gf.Matrix4d().SetRotate(
                Gf.Quatd(0.5, 0.5, 0.5, 0.5)  # YCB axis fix
            )
            trs = Gf.Matrix4d(1.0)
            trs.SetTranslateOnly(Gf.Vec3d(*[v * U for v in world]))
            xf.AddTransformOp().Set(rot * trs)  # row-vector order
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(0.1)
            # The Axis_Aligned YCB variants are VISUAL-ONLY: without
            # colliders the groceries fall straight through the table.
            # SDF collision (not convexHull): a hull fills the banana's
            # curve and the can's rim, so fingers touch phantom volume --
            # SDF keeps the true surface for fine grasping.
            n_col = 0
            for desc in Usd.PrimRange(prim):
                if desc.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(desc)
                    _bind_pmat(desc)
                    mcol = UsdPhysics.MeshCollisionAPI.Apply(desc)
                    if name == "cracker_box":
                        mcol.CreateApproximationAttr("boundingCube")
                    else:
                        mcol.CreateApproximationAttr(
                            "sdf" if args.engine == "physx" else "convexHull")
                    try:
                        from pxr import PhysxSchema

                        sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(desc)
                        sdf.CreateSdfResolutionAttr(256)
                    except Exception:
                        pass  # engine falls back to its default SDF params
                    n_col += 1
            print(f"[bridge] YCB {name}: {n_col} SDF colliders", flush=True)
            print(f"[bridge] YCB prop {name} at {pos}", flush=True)
    else:
        print("[bridge] YCB props skipped: asset root unreachable", flush=True)
except Exception as e:
    print(f"[bridge] YCB props skipped: {e}", flush=True)


# ── cameras: overhead (matches configs/cameras/isaac.yaml extrinsics) ────
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = camera axes in world) -> wxyz quaternion."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) * 2
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _camera(path, eye, target, up, focal_mm=18.0, haperture_mm=20.955):
    """eye/target in METERS, base-frame z (shifted by BASE_Z like _cube).

    Uses the Isaac Sim 6.0 sensor stack (RtxCamera + CameraSensor): raw
    replicator render products silently ignored our USD cameras on this
    build (both cameras rendered content unrelated to their poses)."""
    eye_w = np.array([eye[0], eye[1], eye[2] + BASE_Z], dtype=float) * U
    tgt_w = np.array([target[0], target[1], target[2] + BASE_Z], dtype=float) * U
    fwd = tgt_w - eye_w
    fwd /= np.linalg.norm(fwd)
    upv = np.array(up, dtype=float)
    right = np.cross(fwd, upv)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    # USD camera: +X right, +Y up, looks along -Z. Columns = axes in world.
    R = np.column_stack([right, true_up, -fwd])
    quat = _mat_to_quat_wxyz(R)

    cam = RtxCamera(
        path,
        tick_rate=1.0 / args.dt / max(args.cam_every, 1),
        translations=eye_w,
        orientations=quat,
    )
    cam.camera.set_clipping_ranges(0.01 * U, 20.0 * U)
    # Author the optics DIRECTLY on the USD prim: the 6.0 Camera wrapper's
    # set_focal_lengths(18) authored focalLength=180 on this build (x10
    # unit conversion) -> hFOV 6.7 deg, an accidental telephoto that showed
    # 5 cm cubes as room-filling slabs for four debugging rounds.
    usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
    usd_cam.GetFocalLengthAttr().Set(focal_mm)
    usd_cam.GetHorizontalApertureAttr().Set(haperture_mm)
    usd_cam.GetVerticalApertureAttr().Set(haperture_mm * args.height / args.width)
    sensor = CameraSensor(
        cam,
        resolution=(args.height, args.width),  # (H, W) convention
        annotators=["rgb", "distance_to_image_plane"],
    )
    fx = focal_mm / haperture_mm * args.width
    fy = focal_mm / (haperture_mm * args.height / args.width) * args.height
    K = [[fx, 0.0, args.width / 2], [0.0, fy, args.height / 2], [0.0, 0.0, 1.0]]

    # Print the wrc_demo extrinsics (OpenCV camera -> BASE frame) for the
    # camera profile YAML: x_cv = right, y_cv = -up (image y is down),
    # z_cv = forward; translation relative to the robot base plane.
    R_cv = np.column_stack([right, -true_up, fwd])
    t_cv = np.array([eye[0], eye[1], eye[2]])
    rows = [
        [round(float(R_cv[r, 0]), 6), round(float(R_cv[r, 1]), 6),
         round(float(R_cv[r, 2]), 6), round(float(t_cv[r]), 4)]
        for r in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    print(f"[bridge] {path} extrinsics T (cam->base, paste into camera yaml):",
          flush=True)
    for row in rows:
        print(f"[bridge]     - {row}", flush=True)
    return sensor, K


# cam0 = the manipulation camera, mounted like the REAL rig's tripod: off to
# the front-right, tilted down at the workspace -- an overhead camera at the
# TCP's vertical is blocked by the arm's own home posture (validated the
# hard way). "side" = audience view. The bridge prints each camera's
# cam->base extrinsics at boot; keep configs/cameras/isaac.yaml in sync.
CAM_DEFS = {
    "cam0": _camera("/World_Cams/cam0", (0.78, -0.35, 0.60), (0.28, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "side": _camera("/World_Cams/side", (0.95, 0.75, 0.45), (0.28, 0.0, 0.10), (0.0, 0.0, 1.0)),
}
_annotators: dict[str, tuple] = {
    name: (sensor, K) for name, (sensor, K) in CAM_DEFS.items()
}

if args.gui:
    try:
        _persp = stage.GetPrimAtPath("/OmniverseKit_Persp")
        _eye = np.array([1.5, 1.1, BASE_Z + 0.85]) * U
        _tgt = np.array([0.30, 0.0, BASE_Z + 0.10]) * U
        _fwd = _tgt - _eye; _fwd /= np.linalg.norm(_fwd)
        _rgt = np.cross(_fwd, [0.0, 0.0, 1.0]); _rgt /= np.linalg.norm(_rgt)
        _up = np.cross(_rgt, _fwd)
        _m = Gf.Matrix4d(
            _rgt[0], _rgt[1], _rgt[2], 0.0,
            _up[0], _up[1], _up[2], 0.0,
            -_fwd[0], -_fwd[1], -_fwd[2], 0.0,
            _eye[0], _eye[1], _eye[2], 1.0,
        )
        _xf = UsdGeom.Xformable(_persp)
        _xf.ClearXformOpOrder()
        _xf.AddTransformOp().Set(_m)
        print("[bridge] viewport camera framed on the booth", flush=True)
    except Exception as _e:
        print(f"[bridge] viewport framing skipped: {_e}", flush=True)

# Companion-pack python server: standard live-inspection endpoint (Johnny's
# tooling), alongside the bridge's own exec op.
try:
    import omni.kit.app as _kit_app

    _mgr = _kit_app.get_app().get_extension_manager()
    _mgr.add_path("/home/spark/Downloads/isaac-companion-v1-franka/isaacsim_local_exts")
    _mgr.set_extension_enabled_immediate("isaacsim.code_editor.python_server", True)
    print("[bridge] isaacsim.code_editor.python_server enabled", flush=True)
except Exception as _e:
    print(f"[bridge] python_server not enabled: {_e}", flush=True)

# ── articulation (create AFTER play, gain-tuner gotcha) ──────────────────
from isaacsim.core.experimental.prims import Articulation  # noqa: E402

app_utils.play(commit=True)
for _ in range(10):
    app.update()
engine = str(SimulationManager.get_active_physics_engine()).lower()
print(f"[bridge] physics engine: {engine}", flush=True)

art = Articulation(args.prim)
assert art.num_dofs > 0, "0 DOFs: articulation created before play?"
names = list(art.dof_names)
print(f"[bridge] {art.num_dofs} DOFs: {names}", flush=True)

ARM_IDX = [names.index(f"joint{i}") for i in range(1, 7)]
GRIP_IDX = [names.index(n) for n in ("joint_left", "joint_right") if n in names]
lower, upper = [x.numpy()[0].astype(float) for x in art.get_dof_limits()]
print(f"[bridge] arm idx {ARM_IDX} grip idx {GRIP_IDX} "
      f"grip range {[(round(lower[i], 4), round(upper[i], 4)) for i in GRIP_IDX]}",
      flush=True)


# World-spawn poses for every dynamic prop: re-applied when the user
# presses Stop/Play in the editor (physics re-parse scatters them).
_PROP_SPAWNS = {
    "pink_cube": (0.28, 0.08, 0.026), "green_cube": (0.32, -0.10, 0.026),
    "blue_cube": (0.22, -0.02, 0.021), "banana": (0.24, 0.14, 0.018),
    "cracker_box": (0.36, -0.02, 0.107), "soup_can": (0.20, -0.12, 0.052),
}

_state_lock = threading.Lock()
# Spawn STRAIGHT UP (presentation pose): q=0 lies flat OVER the table and
# sits exactly ON the j2/j3 lower limits. Straight vertical = j2 at +90 deg
# (local convention), j3 kept 1 deg inside its 0 lower limit. The TCP is
# outside the demo workspace AABB here (x~0) -- the harness's workspace
# escape rule lets the first commanded motion come home.
# WRC_BRIDGE_NO_TARGETS=1: asset-inspection mode -- apply NO runtime targets
# so the asset's own authored joint state/drive targets are what you see
# (used to validate the initial-pose PR; also note HOME_Q is in the LOCAL
# joint convention and would fight a mirror-convention asset).
_NO_TARGETS = os.environ.get("WRC_BRIDGE_NO_TARGETS", "0") == "1"
_targets: dict = {
    "q": None if _NO_TARGETS else list(HOME_Q),
    "grip_frac": None if _NO_TARGETS else 1.0,
    "stopped": False,
}
_frames: dict[str, dict] = {}  # camera cache refreshed by the main loop
_exec_lock = threading.Lock()
_exec_jobs: list = []  # (code, result_holder, done_event) -> main loop


def _run_exec_jobs() -> None:
    with _exec_lock:
        jobs, _exec_jobs[:] = list(_exec_jobs), []
    for code, holder, done in jobs:
        import contextlib
        import io
        import traceback

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, globals())
            holder["resp"] = {"ok": True, "stdout": buf.getvalue()[-8000:]}
        except Exception:
            holder["resp"] = {"ok": False, "error": traceback.format_exc()[-2000:],
                              "stdout": buf.getvalue()[-2000:]}
        done.set()


def _grip_frac_now(q_full: np.ndarray) -> float:
    if not GRIP_IDX:
        return 0.0
    fr = [
        (q_full[i] - lower[i]) / max(upper[i] - lower[i], 1e-9) for i in GRIP_IDX
    ]
    return float(np.clip(np.mean(fr), 0.0, 1.0))


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            try:
                resp = self._dispatch(json.loads(raw))
            except Exception as e:  # keep serving; report the error
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            self.wfile.write(json.dumps(resp).encode() + b"\n")
            self.wfile.flush()

    def _dispatch(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "engine": engine, "dofs": names}
        if op == "frame":
            cached = _frames.get(req.get("camera", "cam0"))
            if cached is None:
                return {"ok": False,
                        "error": f"unknown/not-ready camera; have {list(_frames)}"}
            return cached
        if op == "state":
            q = art.get_dof_positions().numpy()[0].astype(float)
            dq = art.get_dof_velocities().numpy()[0].astype(float)
            return {
                "ok": True,
                "q": [float(q[i]) for i in ARM_IDX],
                "dq": [float(dq[i]) for i in ARM_IDX],
                "gripper_pos": _grip_frac_now(q),
            }
        if op == "set_joints":
            with _state_lock:
                _targets["q"] = [float(x) for x in req["q"]][:6]
                _targets["stopped"] = False
            return {"ok": True}
        if op == "gripper":
            with _state_lock:
                _targets["grip_frac"] = float(np.clip(req["pos"], 0.0, 1.0))
            return {"ok": True}
        if op == "stop":
            with _state_lock:
                _targets["stopped"] = True
                _targets["q"] = None
            return {"ok": True}
        if op == "exec":
            # Live-introspection escape hatch (same idea as the companion
            # pack's isaacsim.code_editor.python_server). The bridge binds
            # 127.0.0.1 by default precisely because of this op. Code runs
            # on the MAIN thread between sim steps: authoring USD from a
            # handler thread while hydra renders segfaults the app.
            holder: dict = {}
            done = threading.Event()
            with _exec_lock:
                _exec_jobs.append((req["code"], holder, done))
            if not done.wait(timeout=30.0):
                return {"ok": False, "error": "exec timed out waiting for the main loop"}
            return holder["resp"]
        return {"ok": False, "error": f"unknown op {op!r}"}


socketserver.ThreadingTCPServer.allow_reuse_address = True  # survive TIME_WAIT
server = socketserver.ThreadingTCPServer((os.environ.get("WRC_BRIDGE_BIND", "127.0.0.1"), args.port), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True, name="bridge-tcp").start()
print(f"[bridge] serving wrc_demo bridge on :{args.port}", flush=True)

import cv2  # noqa: E402  (ships with the isaacsim python)


def _refresh_frames():
    t = time.monotonic()
    for cam_name, (sensor, K) in _annotators.items():
        rgb_data, _ = sensor.get_data("rgb")
        if rgb_data is None:
            continue
        rgba = np.asarray(rgb_data.numpy() if hasattr(rgb_data, "numpy") else rgb_data)
        if rgba.size == 0:
            continue
        bgr = cv2.cvtColor(rgba[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        depth_data, _ = sensor.get_data("distance_to_image_plane")
        depth_b64 = None
        if depth_data is not None:
            d = np.asarray(depth_data.numpy() if hasattr(depth_data, "numpy") else depth_data)
            if d.size:
                # sensor returns (H, W, 1); the wire format is flat (H, W)
                d = np.ascontiguousarray(np.squeeze(d), dtype=np.float32) * MPU
                d[~np.isfinite(d)] = 0.0  # RTX far-clip returns inf
                depth_b64 = base64.b64encode(zlib.compress(d.tobytes(), 3)).decode()
        h, w = bgr.shape[:2]
        _frames[cam_name] = {
            "ok": True, "width": w, "height": h, "K": K,
            "rgb_jpeg_b64": base64.b64encode(jpg.tobytes()).decode(),
            "depth_z_b64": depth_b64, "t": t,
        }


# ── main loop: physics + rendering stay on the main thread (Kit rule) ────
# RTX warmup before the first served frame
for _ in range(60):
    app.update()
_refresh_frames()
print(f"[bridge] cameras ready: {list(_frames)}", flush=True)

# ── one-shot scene diagnostics (validate units/transforms/contacts) ──────
xf_cache = UsdGeom.XformCache()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
robot_prim = stage.GetPrimAtPath(args.prim)
if robot_prim:
    b = bbox_cache.ComputeWorldBound(robot_prim).ComputeAlignedRange()
    print(f"[bridge] robot bound (stage units): min={b.GetMin()} max={b.GetMax()}", flush=True)
for cam_name in CAM_DEFS:
    T = xf_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World_Cams/{cam_name}"))
    fwd_w = Gf.Vec3d(-T[2][0], -T[2][1], -T[2][2])  # camera -Z in world
    print(f"[bridge] {cam_name} world pos: {T.ExtractTranslation()} "
          f"forward: {fwd_w}", flush=True)
try:
    from isaacsim.core.experimental.prims import RigidPrim

    _props_rp = RigidPrim("/World_Props/(pink|green|blue)_cube")
    pos_t, _ = _props_rp.get_world_poses()
    print(f"[bridge] prop poses (PHYSICS truth): "
          f"{np.round(pos_t.numpy().reshape(-1, 3), 3).tolist()} "
          f"(fell through table if z << {BASE_Z:.2f})", flush=True)
except Exception as e:
    print(f"[bridge] RigidPrim probe failed: {e}", flush=True)
for prop_name, _, _, _ in PROPS:
    prim = stage.GetPrimAtPath(f"/World_Props/{prop_name}")
    if prim:
        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        print(f"[bridge] prop {prop_name} world bound: min={rng.GetMin()} "
              f"max={rng.GetMax()}  (base plane at z={BASE_Z:.3f})", flush=True)
for cam_name, (sensor, _) in _annotators.items():
    d_data, _info = sensor.get_data("distance_to_image_plane")
    if d_data is not None:
        d = np.squeeze(np.asarray(d_data.numpy() if hasattr(d_data, "numpy") else d_data))
        finite = d[np.isfinite(d)]
        if finite.size:
            center = float(d[d.shape[0] // 2, d.shape[1] // 2])
            print(f"[bridge] {cam_name} RAW depth: min={float(finite.min()):.4f} "
                  f"max={float(finite.max()):.4f} center={center:.4f}", flush=True)

import omni.timeline  # noqa: E402

_tl = omni.timeline.get_timeline_interface()
_was_playing = True


def _resume_scene():
    """After an editor Stop->Play: handles are invalid and physics
    re-parsed. Rebuild the articulation view, restore the ready pose and
    put every prop back on its spot (gain-tuner gotcha: recreate the
    Articulation after every Stop)."""
    global art
    art = Articulation(args.prim)
    with _state_lock:
        _targets["q"] = list(HOME_Q)
        _targets["grip_frac"] = 1.0
        _targets["stopped"] = False
    try:
        from isaacsim.core.experimental.prims import RigidPrim

        for _n, _pos in _PROP_SPAWNS.items():
            _p = stage.GetPrimAtPath(f"/World_Props/{_n}")
            if _p:
                _rp = RigidPrim(f"/World_Props/{_n}", reset_xform_op_properties=True)
                _rp.set_world_poses(
                    np.array([[_pos[0], _pos[1], _pos[2] + BASE_Z]]),
                    np.array([[1.0, 0.0, 0.0, 0.0]]),
                )
    except Exception as _e:
        print(f"[bridge] prop reset skipped: {_e}", flush=True)
    print("[bridge] editor Play detected: articulation + scene restored", flush=True)


step = 0
try:
    while app.is_running():
        playing = _tl.is_playing()
        if not playing:
            _was_playing = False
            app.update()
            _run_exec_jobs()
            continue
        if not _was_playing:
            _was_playing = True
            for _ in range(5):
                app.update()  # let physics finish re-attaching
            try:
                _resume_scene()
            except Exception as _e:
                print(f"[bridge] resume failed: {_e}", flush=True)
        with _state_lock:
            q6 = _targets["q"]
            gf = _targets["grip_frac"]
            stopped = _targets["stopped"]
        if not stopped:
            try:
                tgt = art.get_dof_position_targets().numpy()[0].astype(np.float32).copy()
                if q6 is not None:
                    for k, i in enumerate(ARM_IDX):
                        tgt[i] = q6[k]
                if gf is not None:
                    for i in GRIP_IDX:
                        tgt[i] = lower[i] + gf * (upper[i] - lower[i])
                art.set_dof_position_targets(tgt.reshape(1, -1))
            except Exception:
                pass  # stale view during a Stop/Play transition
        app.update()
        _run_exec_jobs()
        step += 1
        if step % args.cam_every == 0:
            _refresh_frames()
finally:
    server.shutdown()
    app_utils.stop()
    app.close()
    sys.exit(0)
