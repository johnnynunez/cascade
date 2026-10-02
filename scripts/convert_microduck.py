#!/usr/bin/env python3
"""Offline, CPU-only MicroDuck MJCF -> USD admission (no Kit or rollout).

Optional conversion dependencies are imported only when used. USD validation
recompiles the independent MJCF source and reads native USD APIs; metadata is
not accepted as a substitute for physics. A confirmed asset is NOT a confirmed
engine, actuator or locomotion profile.
"""
from __future__ import annotations

from functools import lru_cache
import argparse
import ast
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import stat
from types import SimpleNamespace
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
SOURCE_PIN = "8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec"
MODEL_PATH = Path("microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml")
CONSTANTS_PATH = Path("microduck_rl/src/mjlab_microduck/robot/microduck_constants.py")


@lru_cache(maxsize=1)
def _deps():
    import mujoco
    import newton_usd_schemas  # noqa: F401 -- register BEFORE importing USD
    import mujoco_usd_converter  # also registers the Mjc schemas
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    return SimpleNamespace(mj=mujoco, converter=mujoco_usd_converter, np=np,
                           Gf=Gf, Sdf=Sdf, Usd=Usd, Geom=UsdGeom,
                           Physics=UsdPhysics, Shade=UsdShade)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _near(actual, expected, label, *, atol=2e-7, rtol=2e-5):
    np = _deps().np
    _require(actual is not None, f"{label}: missing value")
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _require(a.shape == b.shape and np.all(np.isfinite(a))
             and np.allclose(a, b, atol=atol, rtol=rtol),
             f"{label}: USD {a} != MJCF {b}")


def _quat(q):
    return _deps().np.array([q.GetReal(), *q.GetImaginary()], dtype=float)


def _rotation(q):
    d = _deps()
    out = d.np.empty(9)
    d.mj.mju_quat2Mat(out, d.np.asarray(q, dtype=float))
    return out.reshape(3, 3)


def _matrix(prim, cache):
    # Gf matrices are row-vector; MuJoCo matrices are column-vector.
    return _deps().np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T


def _by_name(prims):
    result = {}
    for p in prims:
        name = p.GetDisplayName() or p.GetName()
        _require(name not in result, f"ambiguous USD name: {name}")
        result[name] = p
    return result


def validate_model(source_xml, usd_path):
    """Read-only independent MJCF compilation vs reopened native USD physics."""
    d = _deps()
    source_spec = d.mj.MjSpec.from_file(str(source_xml))
    model = source_spec.compile()
    data = d.mj.MjData(model)
    d.mj.mj_forward(model, data)
    stage = d.Usd.Stage.Open(str(usd_path))
    _require(stage is not None and bool(stage.GetDefaultPrim()), "USD default prim missing")
    _require(d.Geom.GetStageUpAxis(stage) == "Z", "USD must be Z-up")
    _near(d.Geom.GetStageMetersPerUnit(stage), 1.0, "metersPerUnit")
    _near(d.Physics.GetStageKilogramsPerUnit(stage), 1.0, "kilogramsPerUnit")
    prims = list(stage.Traverse())
    bodies = _by_name(p for p in prims if p.HasAPI(d.Physics.RigidBodyAPI))
    _require(set(bodies) == {model.body(i).name for i in range(1, model.nbody)}, "rigid body set differs")
    cache = d.Geom.XformCache()
    for i in range(1, model.nbody):
        body = bodies[model.body(i).name]
        rigid = d.Physics.RigidBodyAPI(body)
        _require(rigid.GetRigidBodyEnabledAttr().Get() is True
                 and rigid.GetKinematicEnabledAttr().Get() is False, f"body not dynamic: {body.GetPath()}")
        mass = d.Physics.MassAPI(body)
        _near(mass.GetMassAttr().Get(), model.body_mass[i], f"mass {body.GetName()}")
        _near(mass.GetCenterOfMassAttr().Get(), model.body_ipos[i], f"center of mass {body.GetName()}")
        r = _rotation(_quat(mass.GetPrincipalAxesAttr().Get()))
        tensor = r @ d.np.diag(mass.GetDiagonalInertiaAttr().Get()) @ r.T
        mr = _rotation(model.body_iquat[i])
        expected_tensor = mr @ d.np.diag(model.body_inertia[i]) @ mr.T
        _near(tensor, expected_tensor, f"inertia {body.GetName()}", atol=1e-10)
        full = body.GetAttribute("newton:inertia")
        if full and full.HasAuthoredValueOpinion():
            values = full.Get()
            _require(values is not None and len(values) == 6, "invalid Newton full inertia")
            xx, yy, zz, xy, xz, yz = values
            _near([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], expected_tensor,
                  f"Newton full inertia {body.GetName()}", atol=1e-10)
        transform = _matrix(body, cache)
        _near(transform[:3, 3], data.xpos[i], f"body position {body.GetName()}")
        _near(transform[:3, :3], data.xmat[i].reshape(3, 3), f"body frame {body.GetName()}")
    free = [j for j in range(model.njnt) if model.jnt_type[j] == d.mj.mjtJoint.mjJNT_FREE]
    _require(len(free) == 1 and model.body_parentid[model.jnt_bodyid[free[0]]] == 0,
             "exactly one free root is required")
    root = bodies[model.body(int(model.jnt_bodyid[free[0]])).name]
    _require(root.HasAPI(d.Physics.ArticulationRootAPI), "free root articulation API missing")
    _require(root.GetAttribute("newton:jointsAddMobility").Get() is True, "Newton free mobility disabled")
    _require(root.GetAttribute("newton:selfCollisionEnabled").Get() is True, "blanket self-collision disabled")
    joints = _by_name(p for p in prims if p.IsA(d.Physics.Joint))
    hinge_ids = [j for j in range(model.njnt) if model.jnt_type[j] == d.mj.mjtJoint.mjJNT_HINGE]
    _require(len(hinge_ids) == model.njnt - 1, "only hinges and free root are supported")
    _require(set(joints) == {model.joint(j).name for j in hinge_ids}, "joint set differs (fixed/world joint or missing hinge)")
    for j in hinge_ids:
        prim = joints[model.joint(j).name]
        _require(prim.IsA(d.Physics.RevoluteJoint), f"not revolute: {prim.GetName()}")
        joint = d.Physics.RevoluteJoint(prim)
        _require(joint.GetJointEnabledAttr().Get() is True, f"disabled joint: {prim.GetName()}")
        _require(not joint.GetExcludeFromArticulationAttr().Get(), "joint removed from articulation")
        _require(not any(name.startswith("PhysicsDriveAPI") for name in prim.GetAppliedSchemas()), "unexpected USD drive; preserve source actuators")
        bid = int(model.jnt_bodyid[j])
        parent = bodies[model.body(int(model.body_parentid[bid])).name]
        child = bodies[model.body(bid).name]
        _require(joint.GetBody0Rel().GetTargets() == [parent.GetPath()]
                 and joint.GetBody1Rel().GetTargets() == [child.GetPath()], f"joint body relationship: {prim.GetName()}")
        frames = []
        for end, b in [(0, parent), (1, child)]:
            transform = _matrix(b, cache)
            anchor = prim.GetAttribute(f"physics:localPos{end}").Get()
            _near(transform[:3, :3] @ d.np.array(anchor) + transform[:3, 3],
                  data.xanchor[j], f"joint anchor{end} {prim.GetName()}")
            r = _rotation(_quat(prim.GetAttribute(f"physics:localRot{end}").Get()))
            axis = d.np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
            _near(transform[:3, :3] @ r @ axis, data.xaxis[j], f"joint axis{end} {prim.GetName()}")
            frames.append(transform[:3, :3] @ r)
        _near(frames[0], frames[1], f"joint full frame alignment {prim.GetName()}")
        _require(bool(model.jnt_limited[j]), "unlimited hinge not supported by this admission profile")
        _near([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()],
              d.np.rad2deg(model.jnt_range[j]), f"joint limits degrees {prim.GetName()}")
        dof = model.jnt_dofadr[j]
        for attr, value in {
            "newton:armature": model.dof_armature[dof],
            "newton:damping": model.dof_damping[dof] * d.np.pi / 180,
            "newton:friction": model.dof_frictionloss[dof],
            "mjc:damping": model.dof_damping[dof],
            "mjc:solreflimit": model.jnt_solref[j],
            "mjc:solimplimit": model.jnt_solimp[j],
            "mjc:solreffriction": model.dof_solref[dof],
            "mjc:solimpfriction": model.dof_solimp[dof],
        }.items():
            _near(prim.GetAttribute(attr).Get(), value, f"{prim.GetName()} {attr}")
    colliders = [p for p in prims if p.GetPath().HasPrefix(stage.GetDefaultPrim().GetPath())
                 and p.HasAPI(d.Physics.CollisionAPI)]
    _require(len(colliders) == int(d.np.count_nonzero(model.geom_contype | model.geom_conaffinity)), "collider count differs")
    _validate_geoms(model, data, stage, bodies, cache)
    _validate_actuators(model, stage, joints, source_spec)
    _validate_masks(model, stage)
    _validate_excludes(model, stage, bodies)
    sites = _by_name(p for p in prims if p.HasAPI("MjcSiteAPI"))
    _require(set(sites) == {model.site(i).name for i in range(model.nsite)}, "site frame set differs")
    for i in range(model.nsite):
        prim = sites[model.site(i).name]
        _require(prim.GetParent() == bodies[model.body(int(model.site_bodyid[i])).name], "site parent differs")
        transform = _matrix(prim, cache)
        _near(transform[:3, 3], data.site_xpos[i], f"site position {prim.GetName()}")
        # Composed float32 USD quaternions lose a few ulps near zero; this is
        # a representation bound, not a physical motion acceptance tolerance.
        _near(transform[:3, :3], data.site_xmat[i].reshape(3, 3), f"site frame {prim.GetName()}",
              atol=8 * d.np.finfo(d.np.float32).eps)
    return {"status": "confirmed", "rigid_bodies": len(bodies),
            "hinge_joints": len(joints), "colliders": len(colliders), "free_root": str(root.GetPath()),
            "site_frames": len(sites), "actuators": model.nu, "body_exclusions": model.nexclude,
            "mass_kg": float(model.body_mass.sum()), "geometries": model.ngeom,
            "nq": model.nq, "nv": model.nv}


def _geoms(model, stage):
    d = _deps()
    available = _by_name(p for p in stage.Traverse() if p.IsA(d.Geom.Gprim)
                         and not p.HasAPI("MjcSiteAPI"))
    names = [model.geom(i).name for i in range(model.ngeom)]
    _require(all(names) and len(set(names)) == len(names), "geoms need unique names before conversion")
    _require(set(names) <= set(available), "source geometry missing from USD")
    return {i: available[name] for i, name in enumerate(names)}


def _validate_geoms(model, data, stage, bodies, cache):
    d = _deps()
    for i, prim in _geoms(model, stage).items():
        label = str(prim.GetPath())
        _require(prim.GetParent() == bodies[model.body(int(model.geom_bodyid[i])).name], f"geom parent {label}")
        transform = _matrix(prim, cache)
        kind = int(model.geom_type[i])
        if kind != d.mj.mjtGeom.mjGEOM_MESH:
            _near(transform[:3, 3], data.geom_xpos[i], f"geom position {label}")
            scale = d.np.linalg.norm(transform[:3, :3], axis=0)
            _require(d.np.isfinite(scale).all() and (scale > 0).all(), f"invalid geom scale {label}")
            rotation = transform[:3, :3] / scale
            _near(rotation, data.geom_xmat[i].reshape(3, 3), f"geom orientation {label}")
            if kind in (d.mj.mjtGeom.mjGEOM_SPHERE, d.mj.mjtGeom.mjGEOM_CAPSULE, d.mj.mjtGeom.mjGEOM_CYLINDER):
                # Inspect the composed transform, not one local scale op.
                # Uniform equivalent encodings are valid; ellipsoids/sheared
                # or anisotropically scaled curved primitives are not admitted.
                _near(scale / scale[0], [1, 1, 1], f"unsupported nonuniform geom scale {label}")
        if kind == d.mj.mjtGeom.mjGEOM_MESH:
            _validate_mesh(model, data, i, prim, transform)
        elif kind == d.mj.mjtGeom.mjGEOM_BOX:
            _require(prim.IsA(d.Geom.Cube), f"geom type {label}")
            _near(d.np.linalg.norm(transform[:3, :3], axis=0) * prim.GetAttribute("size").Get(),
                  model.geom_size[i] * 2, f"box dimensions {label}")
        elif kind == d.mj.mjtGeom.mjGEOM_SPHERE:
            _require(prim.IsA(d.Geom.Sphere), f"geom type {label}")
            _near(d.np.asarray(prim.GetAttribute("radius").Get(), dtype=float) * scale[0],
                  model.geom_size[i, 0], f"effective radius {label}")
        elif kind in (d.mj.mjtGeom.mjGEOM_CAPSULE, d.mj.mjtGeom.mjGEOM_CYLINDER):
            expected = d.Geom.Capsule if kind == d.mj.mjtGeom.mjGEOM_CAPSULE else d.Geom.Cylinder
            _require(prim.IsA(expected), f"geom type {label}")
            # The converter's primitives use Z in the MJCF geometry frame.
            # Non-Z encodings need a different frame comparison; fail closed.
            _require(prim.GetAttribute("axis").Get() == "Z", f"unsupported geom axis {label}")
            _near(d.np.asarray(prim.GetAttribute("radius").Get(), dtype=float) * scale[0],
                  model.geom_size[i, 0], f"effective radius {label}")
            _near(d.np.asarray(prim.GetAttribute("height").Get(), dtype=float) * scale[2],
                  2 * model.geom_size[i, 1], f"effective height {label}")
        else:
            raise ValueError(f"unsupported geometry type {kind}")
        _validate_visual(model, i, prim)
        collision = bool(model.geom_contype[i] or model.geom_conaffinity[i])
        _require(prim.HasAPI(d.Physics.CollisionAPI) == collision, f"collision API {label}")
        if not collision:
            continue
        _require(d.Physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True, f"disabled collider {label}")
        for attr, value in {
            "mjc:condim": model.geom_condim[i], "mjc:priority": model.geom_priority[i],
            "mjc:solref": model.geom_solref[i], "mjc:solimp": model.geom_solimp[i],
            "mjc:solmix": model.geom_solmix[i], "newton:contactMargin": model.geom_margin[i],
            "newton:contactGap": model.geom_gap[i],
        }.items():
            _near(prim.GetAttribute(attr).Get(), value, f"{label} {attr}")
        material, _ = d.Shade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        _require(bool(material) and material.GetPrim().HasAPI(d.Physics.MaterialAPI), f"physics material binding {label}")
        for attr, value in {
            "physics:dynamicFriction": model.geom_friction[i, 0],
            "physics:staticFriction": model.geom_friction[i, 0],
            "newton:torsionalFriction": model.geom_friction[i, 1],
            "newton:rollingFriction": model.geom_friction[i, 2],
        }.items():
            _near(material.GetPrim().GetAttribute(attr).Get(), value, f"{label} {attr}")


def _validate_mesh(model, data, i, prim, transform):
    # MuJoCo recenters and rotates mesh vertices into principal-inertia space.
    # Compare actual world geometry, not the differing storage-frame origins.
    from scipy.spatial import cKDTree
    d = _deps()
    _require(prim.IsA(d.Geom.Mesh), f"mesh type {prim.GetPath()}")
    mesh = d.Geom.Mesh(prim)
    p = d.np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = d.np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = d.np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    mid = model.geom_dataid[i]
    vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
    _require(len(counts) == len(faces) and (counts == 3).all(), f"mesh face count/topology {prim.GetPath()}")
    _require(len(indices) == 3 * len(faces) and len(p) and (indices >= 0).all()
             and (indices < len(p)).all(), "mesh indices invalid")
    actual = p @ transform[:3, :3].T + transform[:3, 3]
    expected = vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
    for a, b in [(actual, expected), (actual[indices.reshape(-1, 3)].mean(axis=1), expected[faces].mean(axis=1))]:
        _require(d.np.isfinite(a).all() and len(a) > 0, "mesh points invalid")
        error = max(float(cKDTree(a).query(b)[0].max()), float(cKDTree(b).query(a)[0].max()))
        _require(error <= 2e-7, f"mesh geometry/topology {prim.GetPath()}: error {error} m")
    if prim.HasAPI(d.Physics.CollisionAPI):
        _require(d.Physics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull",
                 f"mesh collision approximation {prim.GetPath()}")


def _validate_visual(model, i, prim):
    d = _deps()
    geom = d.Geom.Gprim(prim)
    _near(geom.GetDisplayColorPrimvar().ComputeFlattened(), [model.geom_rgba[i, :3]], f"visual color {prim.GetName()}")
    _near(geom.GetDisplayOpacityPrimvar().ComputeFlattened(), [model.geom_rgba[i, 3]], f"visual opacity {prim.GetName()}")
    mid = model.geom_matid[i]
    if mid < 0:
        return
    material, _ = d.Shade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    _require(bool(material), f"visual material binding {prim.GetName()}")
    shader, _, _ = material.ComputeSurfaceSource()
    _require(bool(shader), f"visual surface shader {prim.GetName()}")
    color = model.mat_rgba[mid, :3].astype(float)
    linear = d.np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
    for name, expected in [("diffuseColor", linear), ("opacity", model.mat_rgba[mid, 3]),
                           ("roughness", 1 - model.mat_shininess[mid])]:
        attrs = shader.GetInput(name).GetValueProducingAttributes()
        _require(len(attrs) == 1, f"material input {name} missing/ambiguous")
        _near(attrs[0].Get(), expected, f"material {name} {prim.GetName()}")


def _validate_excludes(model, stage, bodies):
    d = _deps()
    expected = set()
    for signature in model.exclude_signature:
        a, b = int(signature) >> 16, int(signature) & 65535
        expected.add(frozenset([str(bodies[model.body(a).name].GetPath()), str(bodies[model.body(b).name].GetPath())]))
    actual = set()
    for p in stage.Traverse():
        if p.HasAPI(d.Physics.FilteredPairsAPI):
            for target in d.Physics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():
                actual.add(frozenset([str(p.GetPath()), str(target)]))
    _require(actual == expected, "body collision excludes differ")


def _validate_actuators(model, stage, joints, source_spec):
    d = _deps()
    actuators = _by_name(p for p in stage.Traverse() if p.GetTypeName() == "MjcActuator")
    _require(set(actuators) == {model.actuator(i).name for i in range(model.nu)}, "actuator set differs")
    for i in range(model.nu):
        prim = actuators[model.actuator(i).name]
        source = source_spec.actuator(model.actuator(i).name)
        limited_tokens = {d.mj.mjtLimited.mjLIMITED_FALSE: "false", d.mj.mjtLimited.mjLIMITED_TRUE: "true",
                          d.mj.mjtLimited.mjLIMITED_AUTO: "auto"}
        for attr, value in [("mjc:forceLimited", source.forcelimited), ("mjc:ctrlLimited", source.ctrllimited),
                            ("mjc:actLimited", source.actlimited)]:
            _require(prim.GetAttribute(attr).Get() == limited_tokens[value], f"actuator limit flag {attr}")
        _require(model.actuator_trntype[i] == d.mj.mjtTrn.mjTRN_JOINT, "non-joint actuator unsupported")
        joint = joints[model.joint(int(model.actuator_trnid[i, 0])).name]
        _require(prim.GetRelationship("mjc:target").GetTargets() == [joint.GetPath()], "actuator target changed")
        for attr, index, tokens in [
            ("mjc:gainType", model.actuator_gaintype[i], ["fixed", "affine", "muscle", "user"]),
            ("mjc:biasType", model.actuator_biastype[i], ["none", "affine", "muscle", "user"]),
            ("mjc:dynType", model.actuator_dyntype[i], ["none", "integrator", "filter", "filterexact", "muscle", "user"]),
        ]:
            _require(0 <= index < len(tokens) and prim.GetAttribute(attr).Get() == tokens[index], f"actuator type {attr}")
        for attr, value in {
            "mjc:gainPrm": model.actuator_gainprm[i], "mjc:biasPrm": model.actuator_biasprm[i],
            "mjc:dynPrm": model.actuator_dynprm[i], "mjc:gear": model.actuator_gear[i],
            "mjc:ctrlRange:min": model.actuator_ctrlrange[i, 0], "mjc:ctrlRange:max": model.actuator_ctrlrange[i, 1],
            "mjc:forceRange:min": model.actuator_forcerange[i, 0], "mjc:forceRange:max": model.actuator_forcerange[i, 1],
        }.items():
            _near(prim.GetAttribute(attr).Get(), value, f"actuator {prim.GetName()} {attr}")


def _mask_signatures(model):
    return sorted({(1, 1)} | {(int(a), int(b)) for a, b in zip(model.geom_contype, model.geom_conaffinity) if a or b})


def _mask_allows(a, b):
    return bool((a[0] & b[1]) or (b[0] & a[1]))


def _group_path(root, signature):
    return root.AppendPath(f"CollisionGroups/mask_{signature[0]}_{signature[1]}")


def _author_masks(model, stage):
    """USD collision groups implement the MJCF bitwise OR rule, not an AND."""
    d = _deps()
    root, geoms = stage.GetDefaultPrim().GetPath(), _geoms(model, stage)
    signatures = _mask_signatures(model)
    for signature in signatures:
        group = d.Physics.CollisionGroup.Define(stage, _group_path(root, signature))
        collection = group.GetCollidersCollectionAPI()
        collection.CreateExpansionRuleAttr("explicitOnly")
        paths = [p.GetPath() for i, p in geoms.items()
                 if (model.geom_contype[i], model.geom_conaffinity[i]) == signature]
        collection.CreateIncludesRel().SetTargets(paths)
        group.CreateFilteredGroupsRel().SetTargets([_group_path(root, other) for other in signatures
                                                    if not _mask_allows(signature, other)])
    for i, prim in geoms.items():
        for name, value in [("contype", model.geom_contype[i]), ("conaffinity", model.geom_conaffinity[i])]:
            # Not a schema attribute: provenance plus the native groups above.
            prim.CreateAttribute(f"mjc:{name}", d.Sdf.ValueTypeNames.Int, custom=True).Set(int(value))
    stage.GetRootLayer().Save()


def _validate_masks(model, stage):
    d = _deps()
    root, geoms = stage.GetDefaultPrim().GetPath(), _geoms(model, stage)
    signatures = _mask_signatures(model)
    paths = {_group_path(root, signature) for signature in signatures}
    groups = [p for p in stage.Traverse() if p.IsA(d.Physics.CollisionGroup)]
    _require({p.GetPath() for p in groups} == paths, "collision mask group set differs")
    # Relationship text alone misses merges and inactive groups. Require the
    # native effective table, including diagonal pairs, with no fallback.
    table = d.Physics.CollisionGroup.ComputeCollisionGroupTable(stage)
    _require(set(table.GetGroups()) == paths, "effective collision mask group set differs")
    for a in signatures:
        for b in signatures:
            _require(table.IsCollisionEnabled(_group_path(root, a), _group_path(root, b))
                     == _mask_allows(a, b), f"effective collision mask filtering {a}, {b}")
    for signature in signatures:
        group = d.Physics.CollisionGroup(stage.GetPrimAtPath(_group_path(root, signature)))
        _require(not group.GetMergeGroupNameAttr().Get(), "collision group merge unsupported")
        _require(bool(group), f"missing collision mask group {signature}")
        collection = group.GetCollidersCollectionAPI()
        expected = {p.GetPath() for i, p in geoms.items()
                    if (model.geom_contype[i], model.geom_conaffinity[i]) == signature}
        actual = set(collection.GetIncludesRel().GetTargets())
        _require({p for p in actual if p.HasPrefix(root)} == expected, f"mask group membership {signature}")
        _require(collection.GetExpansionRuleAttr().Get() == "explicitOnly"
                 and not collection.GetExcludesRel().GetTargets(), "collision collection must be explicit")
        _require(not group.GetInvertFilteredGroupsAttr().Get(), "inverted collision groups")
        _require(set(group.GetFilteredGroupsRel().GetTargets()) == {_group_path(root, other) for other in signatures
                                                                  if not _mask_allows(signature, other)},
                 f"collision mask filtering {signature}")
    for i, prim in geoms.items():
        _near(prim.GetAttribute("mjc:contype").Get(), model.geom_contype[i], "contype")
        _near(prim.GetAttribute("mjc:conaffinity").Get(), model.geom_conaffinity[i], "conaffinity")


def bind_ground(stage, collision_paths, *, root_path):
    """Enroll explicitly supplied scene ground in the MJCF 1/1 group.

    The robot USD contains NO floor. An integrating scene MUST call this (or
    author equivalent validated membership); an unenrolled floor would collide
    with self-only geoms. This authors USD, not proof that an engine consumes it.
    The caller owns saving its scene layer. Only core USD is needed here;
    importing offline converters/schema wheels into live Kit can conflict with
    Kit's own USD build and is not necessary for collision-group membership.
    """
    from pxr import Sdf, UsdPhysics

    root = Sdf.Path(root_path)
    group = UsdPhysics.CollisionGroup(stage.GetPrimAtPath(_group_path(root, (1, 1))))
    _require(bool(group), "ground mask group missing")
    _require(bool(collision_paths), "explicit ground collider paths required")
    prims = [stage.GetPrimAtPath(path) for path in collision_paths]
    for p in prims:
        _require(bool(p) and p.HasAPI(UsdPhysics.CollisionAPI) and not p.GetPath().HasPrefix(root),
                 "ground must be an external CollisionAPI prim")
    targets = group.GetCollidersCollectionAPI().GetIncludesRel()
    for p in prims:
        targets.AddTarget(p.GetPath())


def _safe(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        _require(not part.is_symlink(), f"symlink refused: {part}")
    return path


def _relative(text):
    _require(isinstance(text, str) and text and not any(c in text for c in "\\\\:%?#")
             and all(ord(c) >= 32 for c in text) and not text.startswith("/")
             and not any(x in ("", ".", "..") for x in text.split("/")), "unsafe resource path")
    return text


def _hash(path):
    path = _safe(path)
    _require(stat.S_ISREG(path.stat().st_mode), f"not a regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _files(root):
    root = _safe(root)
    result = []
    for path in sorted(root.rglob("*")):
        _safe(path)
        if not path.is_dir():
            result.append({"path": path.relative_to(root).as_posix(), "sha256": _hash(path), "size": path.stat().st_size})
    return result


def _json_bytes(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_new(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)


def _contact_rules(constants_path):
    """Read the pinned declarative CollisionCfg with AST, never import mjlab."""
    tree = ast.parse(Path(constants_path).read_text())
    values = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    full = values["FULL_COLLISION"]
    _require(isinstance(full, ast.Call) and isinstance(full.func, ast.Name) and full.func.id == "CollisionCfg",
             "unsupported FULL_COLLISION source")
    rules = {keyword.arg: ast.literal_eval(keyword.value) for keyword in full.keywords}
    _require(set(rules) == {"geom_names_expr", "condim", "priority", "friction"}, "unsupported CollisionCfg fields")
    rules["servo_mesh"] = ast.literal_eval(values["SERVO_MESH_NAME"])
    rules["servo_suffix"] = ast.literal_eval(values["SERVO_GEOM_SUFFIX"])
    rules["source_sha256"] = _hash(constants_path)
    return json.loads(json.dumps(rules))


def _prepare_source(source_xml, *, collision_profile="source", contact_rules=None):
    text = Path(source_xml).read_bytes()
    _require(b"<!DOCTYPE" not in text and b"<!ENTITY" not in text, "XML entities unsupported")
    root = ET.fromstring(text)
    _require(root.tag == "mujoco" and root.find("include") is None, "standalone MJCF required; include unsupported")
    _require(root.find("extension") is None, "plugin extensions unsupported")
    _require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", root.get("model", "")) is not None, "unsafe model name")
    for node in root.iter():
        _require(node.tag != "include", "MJCF include unsupported")
        for key in ("file", "meshdir", "texturedir", "assetdir"):
            if key in node.attrib:
                _relative(node.attrib[key].rstrip("/"))
    spec = _deps().mj.MjSpec.from_file(str(source_xml))
    geoms = root.findall(".//worldbody//geom")
    _require(len(geoms) == len(spec.geoms), "expanded/attached geometry unsupported")
    elements = {node.get("name"): node for node in root.findall(".//worldbody//body")}
    counts, servo_counts = {}, {}
    _require(collision_profile in ("source", "velstand"), "unsupported collision profile")
    _require(collision_profile == "source" or contact_rules is not None, "verified contact rules required")
    identities = []
    for i, geom in enumerate(spec.geoms):
        body = geom.parent.name
        _require(body in elements, "world/unnamed-body geometry unsupported")
        index = counts.get(body, 0)
        direct = elements[body].findall("geom")
        _require(index < len(direct), "nested geometry frames unsupported")
        element = direct[index]
        counts[body] = index + 1
        _require(element.get("name", "") == geom.name, "source geom identity/order disagreement")
        name = geom.name
        if collision_profile == "velstand":
            if geom.meshname == contact_rules["servo_mesh"] and (geom.contype or geom.conaffinity):
                serial = servo_counts.get(body, 0)
                servo_counts[body] = serial + 1
                name = f"{body}_{serial}{contact_rules['servo_suffix']}"
            # First-match-wins, as in upstream's CollisionCfg. Match BEFORE
            # generating anonymous USD identity names, which are not task names.
            if any(re.fullmatch(pattern, name) for pattern in contact_rules["geom_names_expr"]):
                for field in ("condim", "priority", "friction"):
                    for pattern, value in contact_rules[field].items():
                        if re.fullmatch(pattern, name):
                            if field == "friction":
                                friction = list(geom.friction)
                                friction[:len(value)] = value
                                element.set(field, " ".join(str(x) for x in friction))
                            else:
                                element.set(field, str(value))
                            break
        name = name or f"cascade_geom_{i:04d}"
        element.set("name", name)
        identities.append({"id": i, "source_name": geom.name, "usd_name": name,
                           "body": geom.parent.name, "mesh": geom.meshname})
    _require(len({row["usd_name"] for row in identities}) == len(identities), "geometry identity collision")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), identities


def _snapshot(model):
    # Capture source parameters (including sensors/sites not realized as engine
    # devices). Geometry bytes stay in the hashed source tree, not this JSON.
    prefixes = ("body_", "jnt_", "dof_", "geom_", "actuator_", "site_", "sensor_", "exclude_")
    arrays = {key: getattr(model, key).tolist() for key in dir(model)
              if key.startswith(prefixes) and isinstance(getattr(model, key), _deps().np.ndarray)}
    arrays["qpos0"] = model.qpos0.tolist()
    arrays["gravity"] = model.opt.gravity.tolist()
    arrays["timestep"] = float(model.opt.timestep)
    return arrays


def _versions():
    d = _deps()
    packages = ["mujoco-usd-converter", "mujoco", "usd-exchange", "newton-usd-schemas", "numpy", "scipy"]
    package_dir = Path(d.converter.__file__).parent
    return {"packages": {name: importlib.metadata.version(name) for name in packages},
            "usd": list(d.Usd.GetVersion()),
            # A version banner alone does not identify patched installed code.
            "converter_sources": [{"path": p.relative_to(package_dir).as_posix(), "sha256": _hash(p)}
                                  for p in sorted(package_dir.rglob("*")) if p.suffix in (".py", ".json", ".usda")],
            "adapter_sha256": _hash(__file__)}


def _receipt(destination, result):
    result["outputs"] = [r for r in _files(destination) if r["path"] not in ("receipt.json", "receipt.sha256")]
    content = _json_bytes(result)
    digest = hashlib.sha256(content).hexdigest()
    _write_new(destination / "receipt.json", content)
    _write_new(destination / "receipt.sha256", (digest + "\n").encode())
    return {**result, "receipt_sha256": digest}


def convert_model(source_xml, destination, *, collision_profile="source", _contact_rules=None,
                  _provenance=None, _extra_files=None):
    """Low-level local fixture conversion; only convert() admits official pins.

    Source and destination must be operator-owned, not concurrently modified.
    Whole source directory is copied byte-for-byte without following symlinks.
    Never overwrites a prior output, even if that conversion failed.
    """
    source_xml, destination = _safe(source_xml), _safe(destination)
    _require(destination != REPO and REPO not in destination.parents, "USD must remain outside checkout")
    _require(not destination.exists(), "refusing to overwrite existing destination")
    _require(source_xml.parent not in (destination, *destination.parents)
             and destination not in source_xml.parents, "source/output overlap")
    source_files = _files(source_xml.parent)
    prepared, identities = _prepare_source(source_xml, collision_profile=collision_profile, contact_rules=_contact_rules)
    source_snapshot = _snapshot(_deps().mj.MjModel.from_xml_path(str(source_xml)))
    versions = _versions()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(source_xml.parent, destination / "source")
        _require(_files(destination / "source") == source_files, "source copy hash mismatch")
        staged = destination / "source" / "cascade_effective.xml"
        _write_new(staged, prepared)
        effective_snapshot = _snapshot(_deps().mj.MjModel.from_xml_path(str(staged)))
        delta = {key: {"source": source_snapshot[key], "effective": value} for key, value in effective_snapshot.items()
                 if value != source_snapshot[key]}
        allowed = {"geom_condim", "geom_priority", "geom_friction"} if collision_profile == "velstand" else set()
        _require(set(delta) <= allowed, "unexpected source parameter delta")
        reference_delta = {}
        if _contact_rules is not None:
            reference, _ = _prepare_source(source_xml, collision_profile="velstand", contact_rules=_contact_rules)
            reference_path = destination / "source/cascade_velstand_reference.xml"
            _write_new(reference_path, reference)
            snapshot = _snapshot(_deps().mj.MjModel.from_xml_path(str(reference_path)))
            reference_delta = {key: {"source": source_snapshot[key], "effective": value} for key, value in snapshot.items()
                               if value != source_snapshot[key]}
            _require(set(reference_delta) <= {"geom_condim", "geom_priority", "geom_friction"}, "unexpected current contact cfg delta")
        for name, content in (_extra_files or {}).items():
            target = destination / "provenance" / _relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_new(target, content)
        _write_new(destination / "source-contract.json", _json_bytes(source_snapshot))
        _write_new(destination / "effective-contract.json", _json_bytes(effective_snapshot))
        output = _deps().converter.Converter(layer_structure=True, scene=False).convert(str(staged), str(destination / "usd"))
        usd = Path(output.path)
        _author_masks(_deps().mj.MjModel.from_xml_path(str(staged)), _deps().Usd.Stage.Open(str(usd)))
        validation = validate_model(staged, usd)
        return _receipt(destination, {
            "schema_version": 1, "usd_path": str(usd.relative_to(destination)),
            "collision_profile": collision_profile, "contact_rules": _contact_rules,
            "source_vs_current_contact_cfg_delta": reference_delta,
            "historical_checkpoint_training_match": "unverified",
            "source_xml": source_xml.name, "effective_xml": "source/cascade_effective.xml",
            "source_files": source_files, "geometry_identity": identities, "physical_delta": delta,
            "validation": validation, "versions": versions,
            "provenance": _provenance or {"status": "local_fixture_not_official_admission"},
            "live_engine_status": "unverified", "actuator_fidelity_status": "unverified",
            "locomotion_status": "unverified",
            "gaps": ["No simulation or engine-consumption validation",
                     "No USD-to-MuJoCo decoder is used: round-trip is MJCF compile vs reopened USD APIs",
                     "Physics scene/floor/cameras/lights/keyframes are not imported",
                     "Sensors are preserved as source metadata; no engine sensor devices are created",
                     "XML actuators remain source actuators, not BAM; runtime overrides must be explicit",
                     "Runtime ground must join CollisionGroups/mask_1_1 before contact validation"],
        })
    except Exception as exc:
        _write_new(destination / "failure.json", _json_bytes({"status": "refuted", "error": str(exc)}))
        raise


def check_model(source_xml, destination, *, contact_rules=None):
    """Read-only bundle integrity AND independent source-vs-USD revalidation."""
    source_xml, destination = _safe(source_xml), _safe(destination)
    receipt_path = destination / "receipt.json"
    _require(_hash(receipt_path) == (destination / "receipt.sha256").read_text().strip(), "receipt SHA mismatch")
    result = json.loads(receipt_path.read_text())
    _require(result.get("schema_version") == 1, "unsupported receipt schema")
    _require(_files(source_xml.parent) == result["source_files"], "independent source hash mismatch")
    actual = [r for r in _files(destination) if r["path"] not in ("receipt.json", "receipt.sha256")]
    _require(actual == result["outputs"], "output hash/file set mismatch")
    _require(result.get("collision_profile") in ("source", "velstand"), "invalid collision profile")
    _require(result.get("contact_rules") == contact_rules, "contact rules differ from independent source")
    expected, identities = _prepare_source(source_xml, collision_profile=result["collision_profile"], contact_rules=contact_rules)
    effective = destination / _relative(result["effective_xml"])
    _require(effective.read_bytes() == expected and result["geometry_identity"] == identities, "effective source mismatch")
    source_contract = _snapshot(_deps().mj.MjModel.from_xml_path(str(source_xml)))
    effective_contract = _snapshot(_deps().mj.MjModel.from_xml_path(str(effective)))
    _require(json.loads((destination / "source-contract.json").read_text()) == source_contract, "source contract metadata differs")
    _require(json.loads((destination / "effective-contract.json").read_text()) == effective_contract, "effective contract metadata differs")
    delta = {key: {"source": source_contract[key], "effective": value} for key, value in effective_contract.items()
             if value != source_contract[key]}
    _require(delta == result["physical_delta"], "physical delta metadata differs")
    reference_delta = {}
    if contact_rules is not None:
        reference, _ = _prepare_source(source_xml, collision_profile="velstand", contact_rules=contact_rules)
        reference_path = destination / "source/cascade_velstand_reference.xml"
        _require(reference_path.read_bytes() == reference, "current contact reference metadata differs")
        snapshot = _snapshot(_deps().mj.MjModel.from_xml_path(str(reference_path)))
        reference_delta = {key: {"source": source_contract[key], "effective": value} for key, value in snapshot.items()
                           if value != source_contract[key]}
    _require(reference_delta == result["source_vs_current_contact_cfg_delta"], "current contact delta metadata differs")
    for key in ("locomotion_status", "live_engine_status", "actuator_fidelity_status", "historical_checkpoint_training_match"):
        _require(result.get(key) == "unverified", f"{key} must remain unverified")
    validation = validate_model(effective, destination / _relative(result["usd_path"]))
    _require(validation == result["validation"], "recorded validation differs from actual USD")
    return {**result, "receipt_sha256": _hash(receipt_path)}


def _admission(admitted_source):
    spec = importlib.util.spec_from_file_location("cascade_microduck_assets", REPO / "scripts/microduck_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.load_manifest()
    _require(manifest["sources"]["microduck_rl"]["revision"] == SOURCE_PIN, "source pin mismatch")
    result = module.check(manifest, admitted_source)
    _require(result["ok"], f"source admission failed: {result['errors'][:3]}")
    admitted_source = _safe(admitted_source)
    prefix = MODEL_PATH.parent.as_posix() + "/"
    expected = sorted([{**r, "path": r["path"][len(prefix):]} for r in manifest["files"] if r["path"].startswith(prefix)], key=lambda r: r["path"])
    expected = [{k: r[k] for k in ("path", "sha256", "size")} for r in expected]
    _require(_files(admitted_source / MODEL_PATH.parent) == expected, "source admission forbids extra/unpinned model files")
    return manifest, result


def convert(admitted_source, destination, *, variant="allcollisions", collision_profile="source", accept_model_license=False):
    _require(variant == "allcollisions", "unsupported variant: only allcollisions, without rollers/backlash")
    _require(accept_model_license is True, "explicit --accept-model-license is required")
    manifest, admission = _admission(admitted_source)
    root = Path(admitted_source)
    extras = {name: (root / name).read_bytes() for name in
              ("microduck_rl/LICENSE", "microduck_rl/README.md", CONSTANTS_PATH.as_posix())}
    extras["manifest.json"] = (REPO / "assets/microduck/manifest.json").read_bytes()
    extras["NOTICE.md"] = (REPO / "assets/microduck/NOTICE.md").read_bytes()
    return convert_model(root / MODEL_PATH, destination, collision_profile=collision_profile,
                         _contact_rules=_contact_rules(root / CONSTANTS_PATH), _extra_files=extras,
                         _provenance={"sources": manifest["sources"], "licenses": manifest["licenses"],
                                      "admission": admission, "manifest_sha256": _hash(REPO / "assets/microduck/manifest.json")})


def check(admitted_source, destination):
    manifest, _ = _admission(admitted_source)
    result = check_model(Path(admitted_source) / MODEL_PATH, destination,
                         contact_rules=_contact_rules(Path(admitted_source) / CONSTANTS_PATH))
    _require(result["provenance"]["sources"] == manifest["sources"], "receipt source provenance mismatch")
    _require(result["provenance"]["manifest_sha256"] == _hash(REPO / "assets/microduck/manifest.json"), "admission manifest mismatch")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--convert", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--admitted-source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--variant", default="allcollisions")
    parser.add_argument("--collision-profile", choices=["source", "velstand"], default="source",
                        help="velstand applies ONLY the pinned current CollisionCfg, not BAM/HOME/randomization")
    parser.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check:
            _require(not args.accept_model_license and args.variant == "allcollisions" and args.collision_profile == "source",
                     "check reads the receipt profile; conversion-only flags refused")
            result = check(args.admitted_source, args.destination)
        else:
            result = convert(args.admitted_source, args.destination, variant=args.variant,
                             collision_profile=args.collision_profile, accept_model_license=args.accept_model_license)
        print(json.dumps({"ok": True, "usd_path": str(args.destination / result["usd_path"]),
                          "receipt_sha256": result["receipt_sha256"], "validation": result["validation"],
                          "live_engine_status": result["live_engine_status"]}, sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
