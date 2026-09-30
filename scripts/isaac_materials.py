"""USD physics-material binding helpers; safe to import outside Isaac Kit."""

GRIPPER_BODIES = ("gripper_end", "gripper_left", "gripper_right")

# Newton SDF collision for the gripper (NewtonSDFCollisionAPI). The asset asks
# for physics:approximation=convexDecomposition; PhysX cooks its own hulls, but
# Newton runs CoACD at threshold 0.05 on these three meshes and gets 6452
# hulls (measured offline with Newton's own call: gripper_end 3180, left
# 1421, right 1851), which in contact overflow MJWarp's 32768 slots and let
# props fall through. An SDF keeps the real triangle mesh as the contact
# surface instead of an approximation. 1 mm voxels over a narrow band of
# +-10 mm around the surface; no margin, so the surface is not inflated.
GRIPPER_SDF = {
    "newton:sdfTargetVoxelSize": 0.001,
    "newton:sdfNarrowBandInner": -0.01,
    "newton:sdfNarrowBandOuter": 0.01,
    "newton:contactMargin": 0.0,
}

# Newton/MuJoCo contact for the gripper, from MuJoCo Menagerie's grasp-tuned
# pads (robotiq_2f85, franka hand: solref 0.004/1, solimp 0.95/0.99/0.001,
# priority 1). Without them every kitchen contact uses solref 0.02/0.5, so the
# 5000 N/m finger drive sinks the pads ~2 cm into a 50 g cube and ejects it.
# Measured in plain MuJoCo on the vendor reBot MJCF (NEWTON/mj_grip_check.py):
# 0.02/0.5 -> cube thrown 306 mm; these pads -> 0.01 mm, 5.6 mm penetration.
# priority 1 makes the pad's solref/solimp/friction win against any prop.
GRIPPER_CONTACT = {
    "mjc:solref": (0.004, 1.0),
    "mjc:solimp": (0.95, 0.99, 0.001, 0.5, 2.0),
    "mjc:priority": 1,
}

# The vendor's reBot model runs with cone="elliptic" impratio="10"; Newton reads
# both from the PhysicsScene (mjc:option:*). With a pyramidal cone the held
# cube creeps ~0.2 mm/s in MuJoCo; elliptic holds it (0.01 mm/s).
# impratio 10 is not enough for the kitchen orange under Newton: over the
# 30 cm carry it slides 5-15 mm through the pads (gripper frame, sampled every
# 6 physics steps) and was dropped short of the box in 7 of 9 rounds. At 100
# (friction constraints 100x stiffer than normal ones: harder to break) the
# slip is 0.3-2.7 mm and 6/6 rounds land in the box, at 3 and 5 substeps.
SCENE_MJC_OPTIONS = {"mjc:option:cone": "elliptic", "mjc:option:impratio": 100.0}

# Contact stiffness/damping of every collider without an authored material:
# counter, open box, props and arm links. Isaac's NewtonConfig default
# (ke=1e4, kd=100) becomes MuJoCo solref (0.02, 0.5), an UNDER-damped contact.
# The kitchen releases the green cube 7 cm above the counter; with that
# contact it bounces onto its side in 5 of 7 rounds. Newton's own ShapeConfig
# default (ke=2500, kd=100) becomes (0.02, 1.0), critically damped: 8/8
# upright, 0.3-0.6 cm from the square's centre. Critical damping alone is not
# enough: ke=1e4, kd=200 (solref 0.01/1.0) still tipped it in 4/4 rounds.
# The gripper pads are unaffected: they carry their own raw mjc:solref and
# win every pad contact by priority.
NEWTON_DEFAULT_CONTACT = {"contact_ke": 2500.0, "contact_kd": 100.0}


def _editable_colliders(stage, body):
    """Return collision prims under ``body``, de-instancing only their branches.

    The shipped gripper collision meshes are instance proxies, which ordinary
    stage traversal omits and which cannot be edited directly. Only the
    collision-instance branches become editable; mesh references, coordinates
    and transforms stay unchanged.
    """
    from pxr import Usd, UsdPhysics

    # Snapshot paths before changing instancing invalidates proxy prims. A
    # nested rigid body (the fingers sit under gripper_end) owns its own
    # colliders, so prune there.
    paths = []
    walk = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
    for p in walk:
        if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
            walk.PruneChildren()
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI):
            paths.append(str(p.GetPath()))
    deinstanced, colliders = [], []
    for path in paths:
        collider = stage.GetPrimAtPath(path)
        while collider.IsInstanceProxy():
            instance = collider.GetParent()
            while instance.IsInstanceProxy():
                instance = instance.GetParent()
            if not instance.IsInstance():
                raise RuntimeError(f"Cannot locate instance for {path}")
            deinstanced.append(str(instance.GetPath()))
            instance.SetInstanceable(False)
            collider = stage.GetPrimAtPath(path)
        colliders.append(collider)
    return colliders, deinstanced


def _author(prim, name, value):
    from pxr import Sdf, Vt

    if isinstance(value, bool):
        kind, value = Sdf.ValueTypeNames.Bool, value
    elif isinstance(value, int):
        kind = Sdf.ValueTypeNames.Int
    elif isinstance(value, float):
        kind = Sdf.ValueTypeNames.Float if name.startswith("newton:") else Sdf.ValueTypeNames.Double
    elif isinstance(value, str):
        kind = Sdf.ValueTypeNames.Token
    else:
        kind, value = Sdf.ValueTypeNames.DoubleArray, Vt.DoubleArray(list(value))
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, kind)
    attr.Set(value)


def apply_newton_gripper_sdf(stage, robot_prim_path, settings=None, contact=None):
    """Make every gripper collider a Newton SDF mesh collider with pad contact.

    Applies NewtonSDFCollisionAPI with ``settings`` (default GRIPPER_SDF), the
    MuJoCo contact parameters in ``contact`` (default GRIPPER_CONTACT) and sets
    physics:approximation to "none": Newton ignores the approximation on an
    SDF shape, and "none" keeps its parser from warning about it. Mesh points,
    faces and transforms are untouched. PhysX does not read the Newton or
    mjc schemas, so call this only for the Newton engine.
    """
    from pxr import Sdf, Usd, UsdPhysics

    settings = dict(GRIPPER_SDF if settings is None else settings)
    contact = dict(GRIPPER_CONTACT if contact is None else contact)
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root:
        raise ValueError(f"Robot prim not found: {robot_prim_path}")
    done = []
    for body in Usd.PrimRange(root):
        if body.GetName() not in GRIPPER_BODIES or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        colliders, deinstanced = _editable_colliders(stage, body)
        if not colliders:
            raise RuntimeError(f"No gripper colliders found under {body.GetPath()}")
        for collider in colliders:
            if not collider.HasAPI("NewtonSDFCollisionAPI"):
                collider.ApplyAPI("NewtonSDFCollisionAPI")
            if collider.HasAPI("NewtonMeshCollisionAPI"):
                # SDF and mesh collision are exclusive representations.
                collider.RemoveAPI("NewtonMeshCollisionAPI")
            for name, value in settings.items():
                _author(collider, name, value)
            if contact and not collider.HasAPI("MjcCollisionAPI"):
                collider.ApplyAPI("MjcCollisionAPI")
            for name, value in contact.items():
                _author(collider, name, value)
            approx = collider.GetAttribute("physics:approximation")
            if not approx:
                approx = collider.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
            approx.Set("none")
        done.append({"body": str(body.GetPath()),
                     "colliders": [str(c.GetPath()) for c in colliders],
                     "deinstanced_collision_branches": deinstanced})
    if {d["body"].split("/")[-1] for d in done} != set(GRIPPER_BODIES):
        raise RuntimeError(f"Expected gripper bodies {GRIPPER_BODIES}, found "
                           f"{sorted(d['body'].split('/')[-1] for d in done)}")
    return done


def apply_newton_gripper_hulls(stage, robot_prim_path, hulls_usda, contact=None, material=None):
    """Collide the gripper against the vendor's convex hulls under Newton.

    For each gripper body, disables collision on the shipped (high-poly,
    convexDecomposition) meshes and adds, under the body, a reference to the
    matching body's hulls in ``hulls_usda`` (built by
    scripts/build_newton_gripper_hulls.py, already in the body frame). Each
    hull is a convexHull collider carrying the pad contact ``contact``
    (default GRIPPER_CONTACT) and, when given, the pad physics ``material``
    bound directly (Newton reads its dynamic friction as the shape mu, so the
    pads grip with the same friction as under PhysX). The visual meshes and
    every PhysX-facing attribute are left as authored; call this only for the
    Newton engine.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    contact = dict(GRIPPER_CONTACT if contact is None else contact)
    hulls_usda = str(hulls_usda)
    source = Usd.Stage.Open(hulls_usda)
    if source is None:
        raise FileNotFoundError(hulls_usda)
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root:
        raise ValueError(f"Robot prim not found: {robot_prim_path}")
    done = []
    for body in Usd.PrimRange(root):
        if body.GetName() not in GRIPPER_BODIES or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        src = source.GetPrimAtPath(f"/Hulls/{body.GetName()}")
        if not src or not src.GetChildren():
            raise RuntimeError(f"{hulls_usda} has no hulls for {body.GetName()}")
        shipped, deinstanced = _editable_colliders(stage, body)
        for collider in shipped:
            UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr(False)
        holder = stage.DefinePrim(body.GetPath().AppendChild("newton_hulls"), "Xform")
        UsdGeom.Imageable(holder).CreatePurposeAttr(UsdGeom.Tokens.guide)  # never rendered
        holder.GetReferences().ClearReferences()
        holder.GetReferences().AddReference(hulls_usda, Sdf.Path(f"/Hulls/{body.GetName()}"))
        hulls = [p for p in holder.GetChildren() if p.IsA(UsdGeom.Mesh)]
        for hull in hulls:
            UsdPhysics.CollisionAPI.Apply(hull)
            UsdPhysics.MeshCollisionAPI.Apply(hull).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            if contact and not hull.HasAPI("MjcCollisionAPI"):
                hull.ApplyAPI("MjcCollisionAPI")
            for name, value in contact.items():
                _author(hull, name, value)
            if material is not None:
                UsdShade.MaterialBindingAPI.Apply(hull).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants, "physics")
        done.append({"body": str(body.GetPath()), "hulls": len(hulls),
                     "disabled": [str(c.GetPath()) for c in shipped],
                     "deinstanced_collision_branches": deinstanced})
    if {d["body"].split("/")[-1] for d in done} != set(GRIPPER_BODIES):
        raise RuntimeError(f"Expected gripper bodies {GRIPPER_BODIES}, found "
                           f"{sorted(d['body'].split('/')[-1] for d in done)}")
    return done


# Seeed vendor reBot DevArm actuators (rebot_devarm.xml, classes rs06 /
# rs00): position servo gains in SI (N*m/rad, N*m*s/rad). Newton's MuJoCo
# solver clamps the servo torque explicitly at the joint effort limit (36 / 14
# N*m), so the asset's PhysX drive gains (e.g. joint2 1500/deg = 85944 N*m/rad
# against a 36 N*m limit) turn every joint into a saturated relay under Newton:
# measured in the kitchen, joint2 limit-cycled 0.1 rad while holding the grasp
# pose, overshot its upper limit by 0.37 rad on the shutdown park and went NaN
# when those gains were restored on a live scene. With these gains the same
# park/hold loop holds within 0.024 rad (PhysX: 0.07 rad).
MENAGERIE_ARM_GAINS = {
    "joint1": (900.0, 60.0), "joint2": (900.0, 60.0), "joint3": (900.0, 60.0),
    "joint4": (120.0, 10.0), "joint5": (120.0, 10.0), "joint6": (120.0, 10.0),
}


def apply_newton_arm_gains(stage, robot_prim_path, gains=None):
    """Author the vendor's servo gains on the arm's angular drives (Newton only).

    USD angular drive stiffness/damping are per degree, so the SI gains are
    divided by 180/pi. Effort limits (maxForce) are left exactly as authored.
    Returns {joint name: (stiffness, damping)} as authored in USD units.
    """
    import math

    from pxr import Usd, UsdPhysics

    gains = dict(MENAGERIE_ARM_GAINS if gains is None else gains)
    top = "/" + str(robot_prim_path).strip("/").split("/")[0]
    root = stage.GetPrimAtPath(top)
    if not root:
        raise ValueError(f"Robot prim not found: {top}")
    per_deg = math.pi / 180.0
    done = {}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name not in gains or not prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            continue
        kp, kv = gains[name]
        drive = UsdPhysics.DriveAPI(prim, "angular")
        drive.GetStiffnessAttr().Set(kp * per_deg)
        drive.GetDampingAttr().Set(kv * per_deg)
        done[name] = (round(kp * per_deg, 4), round(kv * per_deg, 4))
    if set(done) != set(gains):
        raise RuntimeError(f"Expected drives {sorted(gains)}, found {sorted(done)}")
    return done


def apply_newton_scene_options(stage, options=None):
    """Author MuJoCo solver options (default SCENE_MJC_OPTIONS) on every PhysicsScene."""
    from pxr import UsdPhysics

    options = dict(SCENE_MJC_OPTIONS if options is None else options)
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise RuntimeError("No PhysicsScene to carry the MuJoCo solver options")
    for scene in scenes:
        if not scene.HasAPI("MjcSceneAPI"):
            scene.ApplyAPI("MjcSceneAPI")
        for name, value in options.items():
            _author(scene, name, value)
    return [str(s.GetPath()) for s in scenes]


def bind_gripper_physics_material(stage, robot_prim_path, material):
    """Bind a physics material directly on the two gripper collision meshes.

    The shipped gripper collision meshes are instance proxies, which ordinary
    stage traversal omits and which cannot be edited directly. Make only their
    collision-instance branches editable; mesh references, coordinates and
    transforms remain unchanged. A direct binding is necessary: the native USD
    physics parser does not include inherited bindings in shape.materials even
    when UsdShade.ComputeBoundMaterial resolves them.
    """
    from pxr import Usd, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root:
        raise ValueError(f"Robot prim not found: {robot_prim_path}")
    bound = []
    for body in Usd.PrimRange(root):
        if body.GetName() not in ("gripper_left", "gripper_right"):
            continue
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # Snapshot paths before changing instancing invalidates proxy prims.
        colliders = [str(p.GetPath()) for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        deinstanced = []
        for collider_path in colliders:
            collider = stage.GetPrimAtPath(collider_path)
            while collider.IsInstanceProxy():
                instance = collider.GetParent()
                while instance.IsInstanceProxy():
                    instance = instance.GetParent()
                if not instance.IsInstance():
                    raise RuntimeError(f"Cannot locate instance for {collider_path}")
                deinstanced.append(str(instance.GetPath()))
                instance.SetInstanceable(False)
                collider = stage.GetPrimAtPath(collider_path)
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                material, UsdShade.Tokens.strongerThanDescendants, "physics"
            )
            resolved, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            if not resolved or resolved.GetPath() != material.GetPath():
                raise RuntimeError(f"Gripper material did not resolve on {collider.GetPath()}")
        if not colliders:
            raise RuntimeError(f"No gripper colliders found under {body.GetPath()}")
        bound.append({"body": str(body.GetPath()), "colliders": colliders,
                      "deinstanced_collision_branches": deinstanced})
    if {b["body"].split("/")[-1] for b in bound} != {"gripper_left", "gripper_right"}:
        raise RuntimeError("Expected both gripper rigid bodies")
    return bound
