#!/usr/bin/env python3
"""Isaac Sim side of the cascade bridge -- Isaac Sim 6.0 (develop, Newton).

Run with Isaac Sim's Python:

    export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-$(uname -m)/release
    $ISAACSIM_PATH/python.sh scripts/isaac_bridge.py \
        --usd assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda

Opens the gain-tuned reBot RS asset, adds a tabletop + colored props (incl.
a PINK cube) + two RTX cameras, and serves the newline-JSON protocol from
cascade.sim.bridge_client so the FULL agentic demo (reflex tier, livestream
dashboard, Hermes MCP) runs against the sim:

    cascade --cameras isaac,isaac_side --arm isaac --interactive
    # or MCP:  CASCADE_CAMERAS=isaac,isaac_side CASCADE_ARM=isaac cascade-mcp

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

# Default to the gain-tuned RS asset that ships in this repo (drives, robot
# schema, self-collision off, solver caps already baked by
# assets/usd/RS-rebot-dev-arm/scripts/prep_asset.py). Overridable with --usd
# or $CASCADE_USD so a machine-specific `-plus` variant still slots in.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_USD = os.environ.get(
    "CASCADE_USD",
    os.path.join(_REPO_ROOT, "assets", "usd", "RS-rebot-dev-arm", "RS-rebot-dev-arm.usda"),
)
DEFAULT_PRIM = "/tn__00armrs_asmv3_hJ6D/Geometry/base_link"

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--usd", default=DEFAULT_USD, help="reBot RS scene USD (gain-tuned asset)")
p.add_argument("--prim", default=DEFAULT_PRIM, help="articulation root prim path")
p.add_argument("--engine", default="newton", choices=["newton", "physx"])
# Newton is the default engine. Both historical blockers were re-tested on
# 2026-07-31 against the current build and neither survived:
#  - "offscreen render products return garbage under the newton kit
#    experience": FALSE now. All three cameras return real RGB-D (cam0
#    19k distinct colours, depth 85% valid; side 47%; wrist 93%), verified
#    by eye on a captured frame.
#  - "props free-fall through the table": that was OUR bug, not Newton's,
#    and it is fixed below in two places -- the arm/furniture collision
#    groups (PhysX deny-list semantics vs Newton's "different group == no
#    contact") and the set_velocities signature, which differs between the
#    engines and was being swallowed by a bare except so the velocity was
#    never actually zeroed.
# Pass --engine physx to fall back to the old path.
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
HOME_Q = [0.0, -1.2, -1.2, 0.0, -0.75, 0.0]  # gripper elbow-up, high
# THIS asset's j2/j3 limits are [-3.14, 0] (mirror convention). The old
# [0, +1.2, +1.2, ...] was for the spark `-plus` asset (j2/j3 in [0,+pi]) and
# is ILLEGAL here: PhysX clamps +1.2 to 0, the arm collapses, and the gripper
# hangs BELOW the table (link7 at z=-0.023) so every motion trips the harness
# "link would hit the table". These legal negative values put the gripper
# high and elbow-up, matching the URDF joint limits.

# SimulationApp must exist before ANY isaacsim/omni import.
#
# Engine <-> experience coupling:
# - newton REQUIRES the newton kit experience (enabling the ext post-boot
#   does not create the tensor backend -- gain-tuner gotcha).
# - physx runs under the DEFAULT experience.
# The 2026-07-18 note that offscreen render products return garbage under the
# newton experience NO LONGER HOLDS on this build: cam0/side/wrist all return
# real RGB + depth through the same code path (measured 2026-07-31; cam0 19k
# distinct colours, 85% valid depth, and the frame renders the arm, bin and
# both cubes correctly). The asset's colliders are engine-agreement validated
# (gain tuner 8/8).
from isaacsim import SimulationApp  # noqa: E402

_release = os.environ.get(
    "ISAACSIM_PATH",
    os.path.expanduser(f"~/Projects/isaac/IsaacSim/_build/linux-{os.uname().machine}/release"),
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
    # This repo's base asset applies PhysicsArticulationRootAPI +
    # NewtonArticulationRootAPI but NOT PhysxArticulationAPI, so a bare
    # `physxArticulation:enabledSelfCollisions` attribute is orphaned --
    # PhysX never reads it and falls back to its default (self-collision
    # ENABLED). The home pose interpenetrates adjacent elbow colliders
    # (~6900 self-contacts), so with PhysX self-collision on the whole
    # articulation NaNs on the first physics tick (q -> nan, links fly to
    # 1e12). Apply the PhysX articulation schema FIRST, then the flag, so
    # PhysX actually honours it. (The -plus asset variant instead ships 19
    # physics:filteredPairs; self-collision OFF is the gain-tuner-validated
    # 8/8 configuration and is what the shared base asset needs here.)
    from pxr import PhysxSchema  # noqa: E402

    _px_art = PhysxSchema.PhysxArticulationAPI.Apply(_root)
    _px_art.CreateEnabledSelfCollisionsAttr(False)
    print("[bridge] PhysxArticulationAPI applied; self-collision disabled (PhysX)",
          flush=True)

for prim in stage.Traverse():
    attr = prim.GetAttribute("newton:solver:nconmax")
    if attr and attr.IsValid() and attr.HasValue():
        old = attr.Get()
        if old and old > 4600:
            attr.Set(4600)  # just under the geometry estimate (~4745)
            print(f"[bridge] clamped {prim.GetPath()} newton:solver:nconmax {old} -> 4600",
                  flush=True)


def _raise_newton_contact_cap(min_contacts: int = 4600) -> None:
    """Give MJWarp room for this scene's real contact count.

    Authoring `newton:solver:nconmax` on the prim is NOT enough: the MJWarp
    solver instantiates with its own default (200 on this build) and then
    DISCARDS every contact beyond it, printing

        Number of Newton contacts (1015) exceeded MJWarp limit (200).

    once per step. Dropped contacts is exactly the observed symptom -- props
    resting on the table for a while and then sinking through it, fingers
    closing on an object without holding it.

    But the cap must also stay BELOW the allocated contact buffer. Newton
    sizes `contacts.rigid_contact_max` from the geometry (~6915 here, which
    matches the self-contact count in the asset's own evidence package), and
    asking for more than that fails EVERY step with

        MuJoCo naconmax (8192) exceeds contacts.rigid_contact_max (6915)

    which drives the articulation to NaN. 4600 is the value the asset itself
    persists in `newton:solver:nconmax`, comfortably under the allocation and
    ~3x the measured peak during a grasp (1695).

    `njmax` (constraint rows) is raised alongside it: the stock 1200 is the
    other half of the pair the asset's evidence package calls out as
    "overflow instantly with this asset" (analysis_2026-07-07 finding 6,
    which pairs 8192 nconmax with 32768 njmax).
    """
    try:
        import isaacsim.physics.newton.impl.extension as _ne
    except Exception as _e:                       # physx run: nothing to do
        return
    _ns = getattr(_ne, "_newton_stage", None)
    if _ns is None:
        return
    _cfg = getattr(_ns, "cfg", None)
    _solver = getattr(_cfg, "solver_cfg", None) if _cfg is not None else None
    if _solver is None:
        print("[bridge] newton solver_cfg unavailable; contact cap left at default",
              flush=True)
        return
    for _name in ("nconmax", "ncon_max", "rigid_contact_max"):
        if hasattr(_solver, _name):
            _old = getattr(_solver, _name)
            if _old is None or _old < min_contacts:
                setattr(_solver, _name, min_contacts)
                print(f"[bridge] newton solver {_name}: {_old} -> {min_contacts}",
                      flush=True)
    # constraint rows: the other half of the documented pair
    if hasattr(_solver, "njmax"):
        _oldj = getattr(_solver, "njmax")
        if _oldj is None or _oldj < 32768:
            _solver.njmax = 32768
            print(f"[bridge] newton solver njmax: {_oldj} -> 32768", flush=True)

    # ── anti-tunnelling, the Newton way ──────────────────────────────────
    # Measured: during a grasp the cube reached 7.5 m/s and ended at
    # z=-0.188 -- THROUGH a 3 cm table slab -- while the arm was almost still
    # (max |dq| 0.075 rad/s) and with zero contact penetration. That is a
    # MISSED contact: at num_substeps=1 and 1/60 s, a body only needs
    # ~1.8 m/s to clear the whole slab between two collision checks.
    #
    # The PhysX knobs for this (maxLinearVelocity, enableCCD,
    # enableSpeculativeCCD) are authored on the props but Newton IGNORES
    # them: its model exposes only `particle_max_velocity`, nothing for rigid
    # bodies. The lever Newton does respect is substepping -- each substep is
    # a fresh collision check, so N substeps raise the tunnelling threshold
    # by N.
    #
    # 4 substeps puts the threshold at ~7.2 m/s, just above the measured
    # ejection speed; combined with a wider contact margin (which lets the
    # solver see an approaching body before it overlaps) this closes the gap
    # without a big step-cost increase.
    if _cfg is not None and getattr(_cfg, "num_substeps", 1) < 4:
        _old_ss = _cfg.num_substeps
        _cfg.num_substeps = 4
        print(f"[bridge] newton num_substeps: {_old_ss} -> 4 "
              f"(tunnelling threshold ~1.8 -> ~7.2 m/s)", flush=True)
    if (_cfg is not None and hasattr(_cfg, "contact_margin")
            and _cfg.contact_margin < 0.02):
        _old_cm = _cfg.contact_margin
        _cfg.contact_margin = 0.02
        print(f"[bridge] newton contact_margin: {_old_cm} -> 0.02",
              flush=True)


if args.engine == "newton":
    _raise_newton_contact_cap()


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
def _fix_gravity() -> None:
    """Repair a degenerate gravity on the physics scene(s).

    This converter-exported asset ships physics:gravityDirection = (0,0,0)
    and physics:gravityMagnitude = -inf. Isaac turns that into a non-finite
    gravity vector, so every dynamic prop gets infinite downward
    acceleration and tunnels straight through the table on the first tick
    (props end up at z ~ -2000..-5000). Author sane Earth gravity along -Z
    (Z-up stage) on BOTH the generic UsdPhysics and PhysX-scene attrs.
    Idempotent; scans for ANY UsdPhysics.Scene so it works whether the scene
    resolves from a payload or lives at /PhysicsScene. Must run BEFORE
    setup_simulation so the engine initialises with finite gravity.
    """
    fixed = False
    for _p in stage.Traverse():
        if _p.GetTypeName() == "PhysicsScene" or _p.IsA(UsdPhysics.Scene):
            _gd = _p.GetAttribute("physics:gravityDirection")
            _gm = _p.GetAttribute("physics:gravityMagnitude")
            _cd = _gd.Get() if (_gd and _gd.IsValid()) else None
            _cm = _gm.Get() if (_gm and _gm.IsValid()) else None
            _bad = (_cd is None or tuple(_cd) == (0.0, 0.0, 0.0)
                    or _cm is None or not math.isfinite(float(_cm)))
            if not _bad:
                continue
            _down = Gf.Vec3f(0.0, 0.0, -1.0)
            for _tok in ("physics:gravityDirection", "physxScene:gravityDirection"):
                _a = _p.GetAttribute(_tok)
                if _a and _a.IsValid():
                    _a.Set(_down)
                else:
                    _p.CreateAttribute(_tok, Sdf.ValueTypeNames.Vector3f).Set(_down)
            for _tok in ("physics:gravityMagnitude", "physxScene:gravityMagnitude"):
                _a = _p.GetAttribute(_tok)
                if _a and _a.IsValid():
                    _a.Set(9.81)
                else:
                    _p.CreateAttribute(_tok, Sdf.ValueTypeNames.Float).Set(9.81)
            print(f"[bridge] FIXED degenerate gravity on {_p.GetPath()} "
                  f"(was dir={_cd} mag={_cm}) -> (0,0,-1)*9.81", flush=True)
            fixed = True
    if not fixed:
        print("[bridge] gravity already finite on all physics scenes", flush=True)


_fix_gravity()

# This asset ships with the arm hoisted in the air (contact-free gravity
# validation). Find where the base actually sits and author the tabletop
# world at THAT height -- cascade only ever sees base-relative geometry,
# so the world offset is invisible to it.
_bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
_robot_range = _bb.ComputeWorldBound(stage.GetPrimAtPath(args.prim)).ComputeAlignedRange()
BASE_Z = float(_robot_range.GetMin()[2]) * MPU  # meters
print(f"[bridge] robot base plane at world z={BASE_Z:.3f} m; "
      f"authoring the tabletop there", flush=True)

# GPU physics: device="cpu" (inherited from the gain-tuner precision
# scripts) runs MuJoCo-Warp on the CPU and stutters badly with the full
# booth scene while RTX renders on the GPU. But GPU PhysX (cuda:0) NaNs the
# whole scene at boot on some builds/GPUs (Blackwell RTX PRO 6000 here:
# arm + props explode to ~1e12 on the first tick). $CASCADE_PHYSICS_DEVICE
# overrides; default cuda:0 with a cpu fallback if GPU pipelines are
# unavailable.
_phys_dev = os.environ.get("CASCADE_PHYSICS_DEVICE", "cuda:0")
try:
    SimulationManager.setup_simulation(dt=args.dt, device=_phys_dev)
except Exception:
    SimulationManager.setup_simulation(dt=args.dt, device="cpu")
    print("[bridge] WARNING: GPU physics unavailable, using CPU", flush=True)
else:
    print(f"[bridge] physics device: {_phys_dev}", flush=True)

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
from pxr import PhysxSchema, UsdShade  # noqa: E402

_pmat = UsdShade.Material.Define(stage, "/World_Props/physics_material")
_pmat_api = UsdPhysics.MaterialAPI.Apply(_pmat.GetPrim())
_pmat_api.CreateStaticFrictionAttr(1.1)
_pmat_api.CreateDynamicFrictionAttr(1.0)
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
    # boundingCube for STATICS TOO (not just the dynamic-body requirement):
    # it is EXACT for these axis-aligned boxes on both engines, and under
    # Newton a raw-trimesh table paired against dense YCB meshes overflows
    # the triangle-pair buffer (6.7M > 1M) -> contacts get dropped at random
    # (boot NaN, floating props, fingers passing through objects).
    mcol = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mcol.CreateApproximationAttr("boundingCube")
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        mapi = UsdPhysics.MassAPI.Apply(mesh.GetPrim())
        mapi.CreateMassAttr(mass)
        # ── anti-tunnelling ──────────────────────────────────────────────
        # Measured failure: during a grasp the cube reached 7.5 m/s and ended
        # at z=-0.188 -- THROUGH a 3 cm table slab -- while the arm was almost
        # still (max |dq| 0.075 rad/s). That is a missed contact, not a
        # mis-resolved one: the scene runs at num_substeps=1, so at 1/60 s a
        # body only needs ~1.8 m/s to skip the whole slab in one step, and
        # `physxRigidBody:maxLinearVelocity` ships as inf with CCD off
        # (enableCCD=False, enableSpeculativeCCD=False).
        #
        # Cap the speed below the tunnelling threshold and turn CCD on. The
        # cap is deliberately well under 1.8 m/s: nothing in this demo should
        # ever legitimately move a prop that fast, so clamping it turns an
        # unrecoverable escape into a visible, verifiable failure.
        _rb = PhysxSchema.PhysxRigidBodyAPI.Apply(mesh.GetPrim())
        _rb.CreateMaxLinearVelocityAttr(1.5)
        _rb.CreateMaxAngularVelocityAttr(20.0)
        _rb.CreateEnableCCDAttr(True)
        _rb.CreateEnableSpeculativeCCDAttr(True)
    else:
        # Static furniture: collected so it can be collision-filtered against
        # the (also-static) arm base_link -- see the collision-group block
        # before the articulation is created.
        _STATIC_PROPS.append(path)
    return mesh


#: paths of every STATIC prop (table + bin walls), filled by _cube(dynamic=False)
_STATIC_PROPS: list[str] = []


# Open-top bin centered on the demo DROP ZONE (grasp.drop_zone in
# configs/demo.yaml): destination-less pick_and_place drops INTO it, and
# "put it in the box" resolves it as a named destination. Walls only --
# the table is the floor; static colliders so visitors can't topple it.
_BIN = (0.18, -0.17)
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
# Props sit in the arm's TOP-DOWN IK envelope (x~0.18) AND clear of the
# robot base_link footprint (base spans y in [-0.10, 0.10]) -- a dynamic
# prop born inside the base collider gets ejected on the first tick (the
# arm<->prop contact is NOT filtered, only arm<->furniture). So keep props
# at |y| >= 0.13 where they are both reachable top-down and outside the base.
PROPS = [
    # Pink that reads as PINK (not orange) to the HSV color tagger: needs a
    # high blue channel so hue lands in the magenta band, not the red/orange
    # band. (0.97,0.38,0.56) tagged as "orange" under the sim lighting.
    # pink_cube is the graspable target: x~0.17, y=+0.15, well inside the
    # top-down envelope with margin for perception error, clear of both the
    # arm-base footprint (|y|>=0.13) and the bin (which sits at y=-0.17).
    # green_cube sits farther out purely as a second perception target.
    ("pink_cube", (0.17, 0.15, 0.04), (0.95, 0.30, 0.70), 0.05, 0.08),
    ("green_cube", (0.30, 0.16, 0.04), (0.10, 0.75, 0.20), 0.05, 0.08),
]
for name, pos, rgb, width, height in PROPS:
    _cube(f"/World_Props/{name}", pos, (width, width, height), rgb, dynamic=True)

# YCB props from the Isaac asset library: textured REAL objects the
# open-vocab detector actually recognizes (flat-shaded cubes register as
# nothing). Skipped gracefully when the asset CDN is unreachable.
try:
    from isaacsim.storage.native import get_assets_root_path

    _assets = get_assets_root_path()
    if _assets:
        # Spawn AT rest height: dropping groceries onto the table reads
        # as "flying cereal" -- born settled = realistic from frame one.
        # NOTE: the tomato_soup_can YCB is excluded -- its mesh ships a baked
        # cm->m scale (extent ±3.38 vs true ±0.034 m) that PhysX boundingCube
        # reads unscaled, producing a ~100x collider that detonates the
        # contact solver every boot. cracker_box is excluded too: at ~20 cm
        # it exceeds the 90 mm jaw (ungraspable) and its size dominates the
        # frame, hiding the graspable cubes. The banana (~4 cm across) is
        # kept -- a real textured object the open-vocab detector recognises
        # AND the parallel jaw can actually pick up.
        # NOTE: banana temporarily disabled -- being debugged separately for a
        # spawn instability. Cubes are the reliable graspable/perception props.
        YCB: list = [
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
            # colliders the groceries fall straight through the table. Their
            # shipped meshes are non-watertight/non-manifold, so SDF fails,
            # convexDecomposition makes degenerate hulls, and even a convex
            # hull of the raw (mis-scaled/rotated) mesh can explode the
            # contact solver (soup can gains 260 m/s on the first tick).
            # For a robust object-agnostic booth demo the collider MUST be
            # stable, so we wrap each YCB in a boundingCube (its own AABB) --
            # an exact-enough parallel-jaw grasp target that never leaks
            # contacts. The textured visual mesh is untouched, so the
            # open-vocab detector still sees a real banana / box / can.
            n_col = 0
            for desc in Usd.PrimRange(prim):
                if desc.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(desc)
                    _bind_pmat(desc)
                    mcol = UsdPhysics.MeshCollisionAPI.Apply(desc)
                    mcol.CreateApproximationAttr("boundingCube")
                    n_col += 1
            print(f"[bridge] YCB {name}: {n_col} boundingCube colliders", flush=True)
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

    # Print the cascade extrinsics (OpenCV camera -> BASE frame) for the
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
    # Eye-in-hand D435i on the wrist: 69.4 deg HFOV (D435i RGB) -> focal
    # 15.13 mm at the default 20.955 aperture. Spawn pose is a placeholder;
    # the main loop re-poses it every camera tick from the gripper_end
    # link's PHYSICS pose (see _update_wrist_cam) and serves a per-frame
    # cam->base extrinsics matrix with the image.
    "wrist": _camera("/World_Cams/wrist", (0.50, 0.0, 0.50), (0.30, 0.0, 0.0),
                     (0.0, 0.0, 1.0), focal_mm=15.13),
}

# Mount (ee frame): the EE convention is x = approach, y = jaw-opening,
# z = x cross y (z points UP at home). Bracket 6 cm above the wrist, view
# axis = approach tilted 30 deg toward -z_ee: during a top-down descent the
# camera looks at the object between the fingers, and even at HOME (approach
# horizontal) the down-tilt catches the table instead of the black void
# beyond it (validated by live offset sweep: most wrist poses at home see
# NOTHING within the 20 m clip -- black rgb + invalid depth is geometry,
# not a sensor bug).
_WRIST_TILT = np.deg2rad(30.0)
_wrist_fwd = np.array([np.cos(_WRIST_TILT), 0.0, -np.sin(_WRIST_TILT)])
_wrist_right = np.array([0.0, -1.0, 0.0])
_WRIST_MOUNT = np.eye(4)
_WRIST_MOUNT[:3, :3] = np.column_stack(
    [_wrist_right, np.cross(_wrist_right, _wrist_fwd), -_wrist_fwd])
_WRIST_MOUNT[:3, 3] = [-0.02, 0.0, 0.06]
_wrist_rp = None  # RigidPrim on the gripper_end link (physics-truth pose)
_wrist_xp = None  # XformPrim on the camera: the ONLY pose channel the RTX
# render product follows live (raw USD xformOp edits during play are
# ignored -- validated by moving cam0 both ways)
_wrist_T: list | None = None  # latest cam->base (OpenCV) 4x4, row lists


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _init_wrist_cam():
    """(Re)bind the wrist camera to the gripper_end link. Call after every
    play (the physics view dies on Stop, gain-tuner gotcha)."""
    global _wrist_rp, _wrist_xp
    from isaacsim.core.experimental.prims import RigidPrim, XformPrim

    lp = getattr(art, "link_paths", None) or []
    paths = list(lp[0]) if (len(lp) and isinstance(lp[0], (list, tuple))) else list(lp)
    link = next((p for p in paths if str(p).endswith("gripper_end")), None)
    if link is None:
        print("[bridge] wrist cam: gripper_end link not found", flush=True)
        return
    _wrist_rp = RigidPrim(str(link))
    _wrist_xp = XformPrim("/World_Cams/wrist")
    print(f"[bridge] wrist cam bound to {link}", flush=True)


def _update_wrist_cam():
    """Follow the gripper: author the camera prim at T_world_ee @ MOUNT and
    refresh the cam->base OpenCV extrinsics served with wrist frames."""
    global _wrist_T
    if _wrist_rp is None or _wrist_xp is None:
        return
    try:
        pos, quat = _wrist_rp.get_world_poses()
    except Exception:
        return  # stale view during a Stop/Play transition
    pos = (pos.numpy() if hasattr(pos, "numpy") else pos).reshape(3)
    quat = (quat.numpy() if hasattr(quat, "numpy") else quat).reshape(4)
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        return
    T_we = np.eye(4)
    T_we[:3, :3] = _quat_wxyz_to_mat(quat)
    T_we[:3, 3] = pos
    T_wc = T_we @ _WRIST_MOUNT
    try:
        _wrist_xp.set_world_poses(
            T_wc[:3, 3].reshape(1, 3),
            _mat_to_quat_wxyz(T_wc[:3, :3]).reshape(1, 4),
        )
    except Exception:
        return
    # OpenCV frame: x=+X(right), y=-Y(image y down), z=-Z(forward)
    R_cv = T_wc[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    t_cv = T_wc[:3, 3] / U - np.array([0.0, 0.0, BASE_Z])
    T = np.eye(4)
    T[:3, :3] = R_cv
    T[:3, 3] = t_cv
    _wrist_T = [[round(float(v), 6) for v in row] for row in T]
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
# tooling), alongside the bridge's own exec op. Path is machine-specific;
# skip cleanly when the extension folder is absent (set $CASCADE_COMPANION_EXTS
# to enable on a machine that has it).
try:
    import omni.kit.app as _kit_app

    _companion = os.environ.get("CASCADE_COMPANION_EXTS", "")
    if _companion and os.path.isdir(_companion):
        _mgr = _kit_app.get_app().get_extension_manager()
        _mgr.add_path(_companion)
        _mgr.set_extension_enabled_immediate("isaacsim.code_editor.python_server", True)
        print("[bridge] isaacsim.code_editor.python_server enabled", flush=True)
    else:
        print("[bridge] companion python_server not configured (CASCADE_COMPANION_EXTS unset)",
              flush=True)
except Exception as _e:
    print(f"[bridge] python_server not enabled: {_e}", flush=True)

# ── gripper friction pads ─────────────────────────────────────────────────
# The arm USD authors no physics material on the finger colliders, so they
# get the PhysX default (0.5 friction, "average" combine): smooth SDF props
# (the YCB banana) slip out of the jaws on lift. Rubber-pad material with
# combine=max makes the pair use OUR friction against any prop.
from pxr import PhysxSchema  # noqa: E402

_gmat = UsdShade.Material.Define(stage, "/World_Props/gripper_material")
_gmat_api = UsdPhysics.MaterialAPI.Apply(_gmat.GetPrim())
_gmat_api.CreateStaticFrictionAttr(1.5)
_gmat_api.CreateDynamicFrictionAttr(1.3)
_gmat_api.CreateRestitutionAttr(0.0)
PhysxSchema.PhysxMaterialAPI.Apply(_gmat.GetPrim()).CreateFrictionCombineModeAttr("max")
for _prim in stage.Traverse():
    _p = str(_prim.GetPath())
    if _prim.HasAPI(UsdPhysics.CollisionAPI) and (
        "gripper_left" in _p or "gripper_right" in _p
    ):
        UsdShade.MaterialBindingAPI.Apply(_prim)
        UsdShade.MaterialBindingAPI(_prim).Bind(
            _gmat, UsdShade.Tokens.strongerThanDescendants, "physics"
        )
        print(f"[bridge] gripper pad material -> {_p}", flush=True)

# ── arm visual materials ─────────────────────────────────────────────────
# The reBot palette now lives in the asset itself (payloads/materials.usda +
# per-piece bindings in instances.usda: pla*_green -> green, cnc/link ->
# aluminium, motor_* -> dark, gripper/black -> black). Nothing to paint at
# runtime; kept as a no-op marker so the fix is discoverable here.
print("[bridge] arm visual materials come from the asset (materials.usda)", flush=True)

# ── static-furniture collision filtering (THE boot-NaN fix) ──────────────
# The arm's base_link collider rests at z=0 and the table top is authored at
# z=0 too, so the two STATIC bodies deeply interpenetrate at spawn. PhysX
# generates huge static-vs-static contact forces there and the whole scene
# NaNs within ~60 steps (arm q -> nan, every prop flung to ~1e12). Two
# static bodies never need mutual contacts, so put the arm in one collision
# group and all static furniture (table + bin walls) in another, and filter
# the pair. (Isolated repro + fix verified in
# scripts/diag_physics.py --add-table {--filter-arm-table}.)
#
# ENGINE SEMANTICS DIFFER -- this is why the groups are PhysX-only:
#   PhysX  : `filteredGroups` is an explicit deny-list. A collider in NO group
#            still collides with everything, so dynamic props fall onto the
#            table normally.
#   Newton : every shape ends up assigned to a group, and shapes in DIFFERENT
#            groups do not generate contact pairs at all. Measured on this rig:
#            405 registered contact pairs, ZERO of them cross-group; the 5
#            furniture shapes were isolated from the 227 others, so both cubes
#            free-fell through the table to z = -227849 m at 6244 m/s with x/y
#            untouched -- pure free fall, no contact, straight from boot.
# Newton does not need this workaround: it resolves the static arm/table
# overlap without the boot NaN. So author the groups ONLY under PhysX.
if args.engine == "physx":
    _arm_grp = UsdPhysics.CollisionGroup.Define(stage, "/World_Props/arm_group")
    _furn_grp = UsdPhysics.CollisionGroup.Define(stage, "/World_Props/furniture_group")
    _arm_grp.CreateFilteredGroupsRel().AddTarget(_furn_grp.GetPath())
    _arm_grp.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(args.prim)
    _furn_inc = _furn_grp.GetCollidersCollectionAPI().GetIncludesRel()
    for _static_path in _STATIC_PROPS:
        _furn_inc.AddTarget(_static_path)
    print(f"[bridge] arm<->furniture collision filtered "
          f"({len(_STATIC_PROPS)} static bodies)", flush=True)
else:
    print("[bridge] collision groups skipped (newton: different group == no "
          "contact, which would isolate the furniture)", flush=True)

# ── gravity sanity (re-assert after payloads + before the articulation) ──
# Idempotent: if the physics parser reset gravity from a late-composed
# payload, put it back before we create the articulation and play.
_fix_gravity()

# ── articulation (create AFTER play, gain-tuner gotcha) ──────────────────
from isaacsim.core.experimental.prims import Articulation  # noqa: E402

app_utils.play(commit=True)
for _ in range(10):
    app.update()
# Gravity can be reset by the physics parser on play; re-assert once more.
_fix_gravity()
engine = str(SimulationManager.get_active_physics_engine()).lower()
print(f"[bridge] physics engine: {engine}", flush=True)

art = Articulation(args.prim)
assert art.num_dofs > 0, "0 DOFs: articulation created before play?"
names = list(art.dof_names)
print(f"[bridge] {art.num_dofs} DOFs: {names}", flush=True)

_init_wrist_cam()
ARM_IDX = [names.index(f"joint{i}") for i in range(1, 7)]
GRIP_IDX = [names.index(n) for n in ("joint_left", "joint_right") if n in names]
lower, upper = [x.numpy()[0].astype(float) for x in art.get_dof_limits()]
print(f"[bridge] arm idx {ARM_IDX} grip idx {GRIP_IDX} "
      f"grip range {[(round(lower[i], 4), round(upper[i], 4)) for i in GRIP_IDX]}",
      flush=True)


# World-spawn poses for every dynamic prop: re-applied when the user
# presses Stop/Play in the editor (physics re-parse scatters them).
_PROP_SPAWNS = {
    "pink_cube": (0.17, 0.15, 0.04), "green_cube": (0.30, 0.16, 0.04),
}

_state_lock = threading.Lock()
# Spawn STRAIGHT UP (presentation pose): q=0 lies flat OVER the table and
# sits exactly ON the j2/j3 lower limits. Straight vertical = j2 at +90 deg
# (local convention), j3 kept 1 deg inside its 0 lower limit. The TCP is
# outside the demo workspace AABB here (x~0) -- the harness's workspace
# escape rule lets the first commanded motion come home.
# CASCADE_BRIDGE_NO_TARGETS=1: asset-inspection mode -- apply NO runtime targets
# so the asset's own authored joint state/drive targets are what you see
# (used to validate the initial-pose PR; also note HOME_Q is in the LOCAL
# joint convention and would fight a mirror-convention asset).
_NO_TARGETS = os.environ.get("CASCADE_BRIDGE_NO_TARGETS", "0") == "1"
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
        if op == "reset_props":
            # Re-settle every prop back onto its spawn (main thread, between
            # sim steps) so a repeated demo starts fresh after a pick moved a
            # cube into the bin. Reuses the exec-job queue for main-thread
            # execution.
            #
            # KNOWN LIMITATION (Newton): a teleport of a RESTING body does not
            # reliably stick. The write reaches both solver state buffers
            # (verified by reading them straight back) and one step later the
            # body can be at its old pose again. A timeline Stop -> Play does
            # make the solver re-parse the stage, but it throws during
            # re-attach on this build and leaves the scene unusable, so it is
            # NOT done here.
            #
            # Consequence for benchmarks: after an episode that put a cube in
            # the bin, reset_props may leave it there. A sweep MUST verify the
            # post-reset pose and skip/restart rather than trust it -- see
            # benchmark/rig/ablation.py, which checks the start pose and
            # aborts. Without that check a sweep silently measures the
            # previous episode's end state, `truth=True` becomes a tautology,
            # and it reports a perfect score with the cube having "moved"
            # 1.5 cm.
            holder: dict = {}
            done = threading.Event()
            with _exec_lock:
                _exec_jobs.append(("_settle_props()", holder, done))
            done.wait(timeout=60)
            return holder.get("resp", {"ok": False, "error": "reset timed out"})
        if op == "place_prop":
            # Put ONE prop at an arbitrary pose. Benchmarks need this: a sweep
            # over initial states is only a sweep if the states differ.
            #
            # This exists because a bare `RigidPrim.set_world_poses` is a
            # silent NO-OP for a resting body under Newton (MuJoCo is
            # reduced-coordinate -- see _newton_teleport). A harness that used
            # RigidPrim directly ran its six "different" start states at the
            # SAME position and reported them as six independent episodes.
            _name = str(req.get("name", "pink_cube"))
            _pos = [float(v) for v in req["pos"][:3]]
            holder: dict = {}
            done = threading.Event()
            _code = (
                f"_ok = _newton_teleport({_name!r}, {tuple(_pos)!r})\n"
                if args.engine == "newton" else
                "from isaacsim.core.experimental.prims import RigidPrim as _R\n"
                f"_rp = _R('/World_Props/{_name}', reset_xform_op_properties=True)\n"
                f"_rp.set_world_poses(np.array([[{_pos[0]}, {_pos[1]}, "
                f"{_pos[2]} + BASE_Z]]), np.array([[1.0, 0.0, 0.0, 0.0]]))\n"
                "_zero_prop_velocity(_rp)\n"
                "_ok = True\n"
            ) + (
                "for _ in range(30):\n"
                "    app.update()\n"
                "print('placed' if _ok else 'place FAILED')\n"
            )
            with _exec_lock:
                _exec_jobs.append((_code, holder, done))
            done.wait(timeout=45)
            return holder.get("resp", {"ok": False, "error": "place timed out"})
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
server = socketserver.ThreadingTCPServer((os.environ.get("CASCADE_BRIDGE_BIND", "127.0.0.1"), args.port), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True, name="bridge-tcp").start()
print(f"[bridge] serving cascade bridge on :{args.port}", flush=True)

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
        entry = {
            "ok": True, "width": w, "height": h, "K": K,
            "rgb_jpeg_b64": base64.b64encode(jpg.tobytes()).decode(),
            "depth_z_b64": depth_b64, "t": t,
        }
        if cam_name == "wrist" and _wrist_T is not None:
            # eye-in-hand: extrinsics move with the arm; serve the matrix
            # that was current when this frame rendered
            entry["T_base_cam"] = _wrist_T
        _frames[cam_name] = entry


# ── main loop: physics + rendering stay on the main thread (Kit rule) ────
# RTX warmup before the first served frame
for _ in range(60):
    app.update()


def _zero_prop_velocity(_rp) -> bool:
    """Zero a rigid body's linear + angular velocity on EITHER engine.

    The two backends disagree on the signature, and the mismatch used to be
    swallowed by a bare `except`, so under Newton the velocity was NEVER
    actually cleared: a prop that picked up speed kept it, tunnelled through
    the table on the next step, and left latent NaNs that detonated the solver
    at the next contact.
      PhysX  : set_velocities(v)                 with v shaped (N, 6)
      Newton : set_velocities(linear, angular)   each shaped (N, 3)
    Returns True only if the velocity was really cleared, so a silent no-op
    cannot masquerade as success.
    """
    try:                                        # newton: two (N,3) args
        _rp.set_velocities(np.zeros((1, 3), dtype=np.float32),
                           np.zeros((1, 3), dtype=np.float32))
        return True
    except (TypeError, ValueError):
        pass
    try:                                        # physx: one (N,6) arg
        _rp.set_velocities(np.zeros((1, 6), dtype=np.float32))
        return True
    except Exception:
        return False


def _newton_teleport(name, pos) -> bool:
    """Teleport a prop under Newton by writing `state.joint_q`.

    Newton's MuJoCo solver is REDUCED-COORDINATE. From its own source
    (newton/_src/solvers/mujoco/solver_mujoco.py, reset_state docstring):

        "Because MuJoCo is a reduced-coordinate solver, state.body_q /
         state.body_qd are DERIVED from the joint coordinates by forward
         kinematics on the next step; the corresponding BODY_Q / BODY_QD
         flags are not actionable here and are ignored."

    and `step()` calls `_update_mjc_data(..., state_in)` every step, pushing
    the joint coordinates into `mjw_data.qpos`. Every other write target is
    downstream of that and gets overwritten within one step. Measured:

        RigidPrim.set_world_poses  -> reverted after 1 step
        state.body_q (one buffer)  -> reverted after 1 step
        state.body_q (both)        -> body ejected at 72 m/s
        mjw_data.qpos              -> reverted after 1 step
        state.joint_q              -> STICKS, |vel| 0.000, stable at 40 steps

    A free body occupies 7 coordinates (3 pos + 4 quat) in joint_q but 6 dofs
    in joint_qd -- the layouts differ, so the dof slice must be derived from
    the joint ordering rather than reused from the coordinate index. Zeroing
    the wrong dofs leaves the body's velocity intact and it flies off (that
    failure looked like [12.5, -22.7] after 40 steps).

    `model.joint_q_start` is unreliable on this build (its entries repeat), so
    the slice is located by matching the prop's current position instead.
    """
    import isaacsim.physics.newton.impl.extension as _ne

    _ns = getattr(_ne, "_newton_stage", None)
    if _ns is None or getattr(_ns, "state_0", None) is None:
        return False

    def _arr(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    from isaacsim.core.experimental.prims import RigidPrim as _XRP

    _cur = _arr(_XRP(f"/World_Props/{name}").get_world_poses()[0]).reshape(-1)[:3]
    if not np.all(np.isfinite(_cur)):
        return False

    _jq = _arr(_ns.state_0.joint_q)
    _qd = _arr(_ns.state_0.joint_qd)

    _idx = None
    for _i in range(len(_jq) - 6):
        if np.allclose(_jq[_i:_i + 3], _cur, atol=3e-3):
            _idx = _i
            break
    if _idx is None:
        return False

    _n_after = (len(_jq) - _idx) // 7
    _dof = len(_qd) - 6 * _n_after
    _target = [pos[0], pos[1], pos[2] + BASE_Z]

    for _nm in ("state_0", "state_1"):
        _st = getattr(_ns, _nm, None)
        if _st is None or getattr(_st, "joint_q", None) is None:
            continue
        _q = _arr(_st.joint_q).copy()
        _q[_idx:_idx + 3] = _target
        _q[_idx + 3:_idx + 7] = [1.0, 0.0, 0.0, 0.0]
        _st.joint_q.assign(_q)
        if getattr(_st, "joint_qd", None) is not None:
            _v = _arr(_st.joint_qd).copy()
            _v[_dof:_dof + 6] = 0.0
            _st.joint_qd.assign(_v)
    return True


def _settle_props() -> None:
    """Deterministically settle every dynamic prop onto the table.

    Props authored with their bottom face exactly on the table plane are
    born in contact, and PhysX occasionally resolves that first-tick contact
    badly and flings a random prop to z ~ -2000 (the "contacts dropped at
    random" failure -- non-deterministic across boots). Iterate: re-place
    each prop a few mm above its spot with ZERO velocity, settle, and repeat
    for any that escaped, then hard-clamp stragglers. Zeroing velocity each
    pass is what breaks the runaway -- a prop that picked up 200 m/s keeps it
    across a plain teleport otherwise.
    """
    from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402

    def _zero_vel(_rp) -> bool:
        return _zero_prop_velocity(_rp)

    def _place(_n, _pos, dz):
        _rp = RigidPrim(f"/World_Props/{_n}", reset_xform_op_properties=True)
        _rp.set_world_poses(
            np.array([[_pos[0], _pos[1], _pos[2] + BASE_Z + dz]]),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
        )
        _zero_vel(_rp)
        # Under Newton the RigidPrim write above is a no-op for a RESTING body
        # (MuJoCo is reduced-coordinate; body_q is derived from joint_q by FK
        # and is overwritten within one step). Write the authoritative
        # coordinate too -- see _newton_teleport for the measurements.
        if args.engine == "newton":
            try:
                _newton_teleport(_n, (_pos[0], _pos[1], _pos[2] + dz))
            except Exception as _e:
                print(f"[bridge] newton teleport failed for {_n}: {_e}",
                      flush=True)
        return _rp

    def _escaped(_rp, _pos):
        """Is this prop somewhere other than the spawn we asked for?

        This gates the settle retry loop AND, through `reset_props`, whether a
        benchmark episode starts from the state it thinks it does.

        It used to test only |x|>1, |y|>1, or z off by >0.1 -- i.e. it caught
        props that had been launched into orbit, but not a prop sitting calmly
        30 cm away. A cube resting INSIDE THE BIN after a successful pick
        passes all three tests, so `_settle_props` reported "settled cleanly"
        without having moved it, and every later sweep episode started with
        the cube already at the goal. That turns `truth=True` into a tautology
        and produced a 6/6 result where the cube had "moved" 1.5 cm.

        Check the actual distance from the requested spawn instead.
        """
        _p = _rp.get_world_poses()[0].numpy().reshape(-1)[:3]
        _want = (_pos[0], _pos[1], _pos[2] + BASE_Z)
        _d = float(np.linalg.norm(np.asarray(_p) - np.asarray(_want)))
        return _d > 0.02

    names = [n for n in _PROP_SPAWNS if stage.GetPrimAtPath(f"/World_Props/{n}")]
    for _attempt in range(4):
        for _n in names:
            _place(_n, _PROP_SPAWNS[_n], 0.03)
        for _ in range(120):
            app.update()
        # re-zero velocity mid-settle to kill any contact runaway early
        for _n in names:
            _zero_vel(RigidPrim(f"/World_Props/{_n}"))
        for _ in range(60):
            app.update()
        stragglers = [n for n in names
                      if _escaped(RigidPrim(f"/World_Props/{n}"), _PROP_SPAWNS[n])]
        if not stragglers:
            print(f"[bridge] props settled cleanly (attempt {_attempt + 1})",
                  flush=True)
            return
        print(f"[bridge] settle attempt {_attempt + 1}: retrying {stragglers}",
              flush=True)
    # final hard clamp: park stragglers exactly on their spot, zero velocity
    for _n in names:
        _rp = RigidPrim(f"/World_Props/{_n}")
        if _escaped(_rp, _PROP_SPAWNS[_n]):
            _place(_n, _PROP_SPAWNS[_n], 0.0)
            print(f"[bridge] settle: {_n} hard-clamped onto table", flush=True)
    for _ in range(30):
        app.update()


# The MJWarp solver only exists once physics has been created and stepped, so
# the early call above is a no-op on most boots. Raise the cap again here,
# before the props are settled -- settling is the first thing that depends on
# contacts actually being resolved rather than discarded.
if args.engine == "newton":
    _raise_newton_contact_cap()

_settle_props()
print("[bridge] props settled onto the table", flush=True)

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
for prop_name, *_ in PROPS:
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
    _init_wrist_cam()
    with _state_lock:
        _targets["q"] = list(HOME_Q)
        _targets["grip_frac"] = 1.0
        _targets["stopped"] = False
    # Props: the USD re-parse already rebirths them at their authored spawn
    # poses. The old code skipped this under Newton because "RigidPrim
    # teleports leave latent NaNs that detonate the sim on the next contact"
    # -- that was the same velocity bug fixed in _settle_props: the teleport
    # left the body's velocity untouched (the (N,6) call Newton rejects was
    # swallowed by a bare except), so a prop carrying speed tunnelled and took
    # the solver with it. Zero the velocity through the engine-aware helper
    # and the teleport is safe on both engines.
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
                _zero_prop_velocity(_rp)
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
            _update_wrist_cam()
            _refresh_frames()
finally:
    server.shutdown()
    app_utils.stop()
    app.close()
    sys.exit(0)
