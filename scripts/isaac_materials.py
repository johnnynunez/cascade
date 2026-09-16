"""USD physics-material binding helpers; safe to import outside Isaac Kit."""


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
