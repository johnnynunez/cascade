"""CPU-only USD conversion; real converter/artifacts, never Kit or simulation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/convert_microduck.py"


def module():
    assert SCRIPT.is_file(), "offline MicroDuck conversion entry point missing"
    spec = importlib.util.spec_from_file_location("microduck_conversion_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def physics():
    pytest.importorskip("mujoco_usd_converter")
    pytest.importorskip("newton_usd_schemas")
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input" / "robot.xml"
    path.parent.mkdir()
    path.write_text('''<mujoco model="fixture">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0.1 -0.2 0.7" quat="0.9238795325 0 0 0.3826834324">
      <freejoint name="root"/>
      <site name="imu" pos="0.02 0.03 0.01" quat="0.9238795325 0 0 0.3826834324" size="0.004"/>
      <inertial pos="0.01 0.02 -0.03" mass="2" fullinertia="0.03 0.04 0.05 0.002 0.001 -0.003"/>
      <geom name="base_collision" type="sphere" size="0.05" friction="0.7 0.03 0.002"/>
      <body name="link" pos="0.03 0.01 0.15" quat="0.9659258263 0.2588190451 0 0">
        <joint name="hinge" type="hinge" axis="1 1 0" pos="0.01 0.02 0.03" range="-0.7 1.2" damping="0.053" armature="0.0018" frictionloss="0.0048"/>
        <inertial pos="0.02 0.01 0" mass="0.4" diaginertia="0.001 0.002 0.0025" quat="0.9238795325 0 0.3826834324 0"/>
        <geom name="tip_collision" type="capsule" size="0.02 0.04" pos="0.02 0 0.08" friction="0.9 0.01 0.003"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="motor" joint="hinge" kp="0.55" kv="0.1" ctrllimited="true" ctrlrange="-2 2" forcelimited="true" forcerange="-0.96 0.96"/></actuator>
</mujoco>''')
    return path


def test_real_conversion_reopens_free_mass_inertia_joint_frames(source, tmp_path, physics):
    m = module()
    result = m.convert_model(source, tmp_path / "converted")
    assert result["validation"]["status"] == "confirmed"
    assert result["validation"]["rigid_bodies"] == 2
    assert result["validation"]["hinge_joints"] == 1
    assert result["validation"]["colliders"] == 2
    assert result["live_engine_status"] == "unverified"
    asset = tmp_path / "converted" / result["usd_path"]
    assert asset.is_file()
    assert m.validate_model(source, asset)["status"] == "confirmed"
    Usd, _, Physics = physics
    stage = Usd.Stage.Open(str(asset))
    bodies = [p for p in stage.Traverse() if p.HasAPI(Physics.RigidBodyAPI)]
    assert all(not Physics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() for p in bodies)
    assert not any(p.IsA(Physics.FixedJoint) for p in stage.Traverse())


def test_ground_binding_requires_only_core_usd_not_offline_converter(monkeypatch):
    Usd = pytest.importorskip("pxr.Usd")
    Geom = pytest.importorskip("pxr.UsdGeom")
    Physics = pytest.importorskip("pxr.UsdPhysics")
    m = module()
    stage = Usd.Stage.CreateInMemory()
    group = Physics.CollisionGroup.Define(stage, "/Robot/CollisionGroups/mask_1_1")
    floor = Geom.Plane.Define(stage, "/World/floor").GetPrim()
    Physics.CollisionAPI.Apply(floor)

    def offline_dependencies_forbidden():
        raise AssertionError("live floor binding must not load the offline converter")

    monkeypatch.setattr(m, "_deps", offline_dependencies_forbidden)
    m.bind_ground(stage, [floor.GetPath()], root_path="/Robot")
    assert group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets() == [floor.GetPath()]
    with pytest.raises(ValueError, match="external CollisionAPI"):
        m.bind_ground(stage, ["/World/missing"], root_path="/Robot")


def test_masks_native_groups_and_explicit_external_floor_binding(source, tmp_path, physics):
    import xml.etree.ElementTree as ET
    tree = ET.parse(source)
    root = tree.getroot()
    ET.SubElement(root.find("worldbody/body"), "geom", name="self_only", type="sphere",
                  size="0.01", pos="0.1 0 0", contype="2", conaffinity="2")
    tree.write(source)
    m = module()
    result = m.convert_model(source, tmp_path / "converted")
    Usd, Geom, Physics = physics
    stage = Usd.Stage.Open(str(tmp_path / "converted" / result["usd_path"]))
    g1 = Physics.CollisionGroup(stage.GetPrimAtPath("/fixture/CollisionGroups/mask_1_1"))
    g2 = Physics.CollisionGroup(stage.GetPrimAtPath("/fixture/CollisionGroups/mask_2_2"))
    assert g1 and g2, "contype/conaffinity must have native USD filtering, not metadata alone"
    assert g1.GetPath() in g2.GetFilteredGroupsRel().GetTargets()
    ground = Geom.Plane.Define(stage, "/World/floor").GetPrim()
    Physics.CollisionAPI.Apply(ground)
    m.bind_ground(stage, [ground.GetPath()], root_path="/fixture")
    assert ground.GetPath() in g1.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
    assert m.validate_model(source, stage.GetRootLayer().realPath)["status"] == "confirmed"


def _rehash_bundle(m, out):
    """Keep integrity valid so negatives exercise semantic admission."""
    import hashlib
    import json
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["outputs"] = [r for r in m._files(out)
                          if r["path"] not in ("receipt.json", "receipt.sha256")]
    raw = m._json_bytes(receipt)
    (out / "receipt.json").write_bytes(raw)
    (out / "receipt.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    return receipt


@pytest.mark.parametrize("mutation", ["merge", "merge_unique", "extra_inside", "extra_outside", "invert", "disable"])
def test_native_mask_semantics_refuted_after_rehash(source, tmp_path, physics, mutation):
    import xml.etree.ElementTree as ET
    tree = ET.parse(source)
    ET.SubElement(tree.getroot().find("worldbody/body"), "geom", name="self_only",
                  type="sphere", size="0.01", contype="2", conaffinity="2")
    tree.write(source)
    m = module()
    out = tmp_path / "converted"
    receipt = m.convert_model(source, out)
    Usd, _, Physics = physics
    asset = out / receipt["usd_path"]
    stage = Usd.Stage.Open(str(asset))
    paths = [f"/fixture/CollisionGroups/mask_{n}_{n}" for n in (1, 2)]
    groups = [Physics.CollisionGroup(stage.GetPrimAtPath(p)) for p in paths]
    table = Physics.CollisionGroup.ComputeCollisionGroupTable(stage)
    assert all(table.IsCollisionEnabled(p, p) for p in paths)
    assert not table.IsCollisionEnabled(*paths)
    if mutation == "merge":
        for group in groups:
            group.CreateMergeGroupNameAttr("merged_masks")
        table = Physics.CollisionGroup.ComputeCollisionGroupTable(stage)
        assert not any(table.IsCollisionEnabled(p, p) for p in paths)
    elif mutation == "merge_unique":
        for index, group in enumerate(groups):
            group.CreateMergeGroupNameAttr(f"unique_{index}")
        table = Physics.CollisionGroup.ComputeCollisionGroupTable(stage)
        assert all(table.IsCollisionEnabled(p, p) for p in paths)
    elif mutation.startswith("extra"):
        path = "/fixture/extra" if mutation == "extra_inside" else "/World/extra"
        extra = Physics.CollisionGroup.Define(stage, path)
        extra.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(
            groups[0].GetCollidersCollectionAPI().GetIncludesRel().GetTargets())
        extra.CreateFilteredGroupsRel().SetTargets([extra.GetPath()])
    elif mutation == "invert":
        groups[0].CreateInvertFilteredGroupsAttr(True)
    else:
        groups[0].GetPrim().SetActive(False)
    stage.GetRootLayer().Save()
    del table, groups, stage
    _rehash_bundle(m, out)
    with pytest.raises(ValueError, match="collision|mask"):
        m.validate_model(source, asset)
    with pytest.raises(ValueError, match="collision|mask"):
        m.check_model(source, out)


def test_native_mask_table_preserves_asymmetric_mjcf_or(source, tmp_path, physics):
    import xml.etree.ElementTree as ET
    tree = ET.parse(source)
    for name, contype, affinity in [("sender", "2", "0"), ("receiver", "0", "2")]:
        ET.SubElement(tree.getroot().find("worldbody/body"), "geom", name=name,
                      type="sphere", size="0.01", contype=contype, conaffinity=affinity)
    tree.write(source)
    m = module()
    out = tmp_path / "converted"
    receipt = m.convert_model(source, out)
    Usd, _, Physics = physics
    stage = Usd.Stage.Open(str(out / receipt["usd_path"]))
    table = Physics.CollisionGroup.ComputeCollisionGroupTable(stage)
    signatures = [(1, 1), (2, 0), (0, 2)]
    for a in signatures:
        for b in signatures:
            assert table.IsCollisionEnabled(f"/fixture/CollisionGroups/mask_{a[0]}_{a[1]}",
                                            f"/fixture/CollisionGroups/mask_{b[0]}_{b[1]}") == bool(
                                                (a[0] & b[1]) or (b[0] & a[1]))
    assert m.check_model(source, out)["validation"]["status"] == "confirmed"


def test_official_merged_masks_refuted_even_with_updated_hashes(admitted_source, tmp_path, physics):
    m = module()
    out = tmp_path / "official"
    receipt = m.convert(admitted_source, out, collision_profile="velstand", accept_model_license=True)
    assert m.check(admitted_source, out)["validation"]["colliders"] == 70
    Usd, _, Physics = physics
    asset = out / receipt["usd_path"]
    stage = Usd.Stage.Open(str(asset))
    groups = [Physics.CollisionGroup(p) for p in stage.Traverse() if p.IsA(Physics.CollisionGroup)]
    for group in groups:
        group.CreateMergeGroupNameAttr("merged_masks")
    table = Physics.CollisionGroup.ComputeCollisionGroupTable(stage)
    assert not any(table.IsCollisionEnabled(g.GetPath(), g.GetPath()) for g in groups)
    stage.GetRootLayer().Save()
    del table, groups, group, stage
    _rehash_bundle(m, out)
    with pytest.raises(ValueError, match="collision|mask"):
        m.validate_model(out / receipt["effective_xml"], asset)
    with pytest.raises(ValueError, match="collision|mask"):
        m.check(admitted_source, out)


@pytest.mark.parametrize("mutation", ["mass", "inertia", "body_frame", "joint_anchor", "joint_axis",
    "joint_limit", "kinematic", "fixed_root", "collider_disabled", "damping", "armature",
    "frictionloss", "condim", "solref", "static_friction", "dynamic_friction", "torsional_friction",
    "actuator_gain", "actuator_target", "geom_radius", "frame_twist", "newton_inertia",
    "self_collision_off", "mobility_off", "extra_drive", "actuator_type", "site_frame",
    "force_limited", "ctrl_limited", "joint_limit_solver", "joint_friction_solver"])
def test_validator_refutes_real_saved_corruption(source, tmp_path, physics, mutation):
    m = module()
    result = m.convert_model(source, tmp_path / "converted")
    Usd, _, Physics = physics
    asset = tmp_path / "converted" / result["usd_path"]
    stage = Usd.Stage.Open(str(asset))
    prims = list(stage.Traverse())
    body = next(p for p in prims if p.GetName() == "base")
    joint = next(p for p in prims if p.IsA(Physics.RevoluteJoint))
    geom = next(p for p in prims if p.GetName() == "base_collision")
    material = next(p for p in prims if p.HasAPI(Physics.MaterialAPI))
    actuator = next(p for p in prims if p.GetTypeName() == "MjcActuator")
    if mutation == "mass": body.GetAttribute("physics:mass").Set(3.0)
    if mutation == "inertia": body.GetAttribute("physics:diagonalInertia").Set((0.03, 0.06, 0.09))
    if mutation == "body_frame": body.GetAttribute("xformOp:translate").Set((0, 0, 0))
    if mutation == "joint_anchor": joint.GetAttribute("physics:localPos1").Set((0.02, 0.03, 0.04))
    if mutation == "joint_axis": joint.GetAttribute("physics:axis").Set("Y")
    if mutation == "joint_limit": joint.GetAttribute("physics:upperLimit").Set(1.2)
    if mutation == "kinematic": body.GetAttribute("physics:kinematicEnabled").Set(True)
    if mutation == "fixed_root":
        weld = Physics.FixedJoint.Define(stage, "/fixture/world_weld")
        weld.CreateBody1Rel().SetTargets([body.GetPath()])
    if mutation == "collider_disabled": Physics.CollisionAPI(geom).CreateCollisionEnabledAttr(False)
    if mutation == "damping": joint.GetAttribute("newton:damping").Set(0.053)
    if mutation == "armature": joint.GetAttribute("newton:armature").Set(0.0)
    if mutation == "frictionloss": joint.GetAttribute("newton:friction").Set(0.0)
    if mutation == "condim": geom.GetAttribute("mjc:condim").Set(1)
    if mutation == "solref": geom.GetAttribute("mjc:solref").Set([0.01, 0.5])
    if mutation == "static_friction": material.GetAttribute("physics:staticFriction").Set(0.0)
    if mutation == "dynamic_friction": material.GetAttribute("physics:dynamicFriction").Set(0.0)
    if mutation == "torsional_friction": material.GetAttribute("newton:torsionalFriction").Set(0.0)
    if mutation == "actuator_gain": actuator.GetAttribute("mjc:gainPrm").Set([0.0] * 10)
    if mutation == "actuator_target": actuator.GetRelationship("mjc:target").SetTargets([])
    if mutation == "geom_radius": geom.GetAttribute("radius").Set(0.2)
    if mutation == "frame_twist":
        from pxr import Gf
        old = joint.GetAttribute("physics:localRot0").Get()
        axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[joint.GetAttribute("physics:axis").Get()]
        twist = Gf.Quatf(Gf.Rotation(Gf.Vec3d(*axis), 45).GetQuat())
        joint.GetAttribute("physics:localRot0").Set(old * twist)
    if mutation == "newton_inertia": body.GetAttribute("newton:inertia").Set([0.03, 0.04, 0.08, 0, 0, 0])
    if mutation == "self_collision_off": body.GetAttribute("newton:selfCollisionEnabled").Set(False)
    if mutation == "mobility_off": body.GetAttribute("newton:jointsAddMobility").Set(False)
    if mutation == "extra_drive": Physics.DriveAPI.Apply(joint, "angular").CreateStiffnessAttr(100)
    if mutation == "actuator_type": actuator.GetAttribute("mjc:gainType").Set("affine")
    if mutation == "force_limited": actuator.GetAttribute("mjc:forceLimited").Set("false")
    if mutation == "ctrl_limited": actuator.GetAttribute("mjc:ctrlLimited").Set("false")
    if mutation == "joint_limit_solver": joint.GetAttribute("mjc:solreflimit").Set([0.2, 0.1])
    if mutation == "joint_friction_solver": joint.GetAttribute("mjc:solreffriction").Set([0.2, 0.1])
    if mutation == "site_frame":
        site = next(p for p in prims if p.GetName() == "imu")
        site.GetAttribute("xformOp:translate").Set((0, 0, 0))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        m.validate_model(source, asset)


@pytest.fixture
def geometry_source(source):
    """Include real cylinder, box and asymmetric scaled mesh exports."""
    import struct
    import xml.etree.ElementTree as ET
    tree = ET.parse(source)
    root = tree.getroot()
    body = root.find("worldbody/body")
    ET.SubElement(body, "geom", name="cylinder", type="cylinder", size="0.03 0.07", pos="0.2 0 0")
    ET.SubElement(body, "geom", name="box", type="box", size="0.02 0.03 0.04", pos="0.3 0 0")
    vertices = [(0.02, 0, 0), (0, 0.04, 0), (0, 0, 0.06), (0, 0, 0)]
    triangles = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    (source.parent / "tetra.stl").write_bytes(b"fixture".ljust(80, b"\0") + struct.pack("<I", 4) + b"".join(
        struct.pack("<12fH", 0, 0, 0, *vertices[a], *vertices[b], *vertices[c], 0)
        for a, b, c in triangles))
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="tetra", file="tetra.stl", scale="1 0.8 1.2")
    ET.SubElement(body, "geom", name="mesh", type="mesh", mesh="tetra", pos="0.4 0 0")
    tree.write(source)
    return source


@pytest.mark.parametrize("name,mutation", [
    ("base_collision", "uniform"), ("tip_collision", "uniform"), ("cylinder", "uniform"),
    ("base_collision", "nonuniform"), ("tip_collision", "nonuniform"), ("cylinder", "nonuniform"),
    ("base_collision", "reflection"), ("tip_collision", "reflection"), ("cylinder", "reflection"),
    ("tip_collision", "axis_x"), ("tip_collision", "axis_y"),
    ("cylinder", "axis_x"), ("cylinder", "axis_y"),
    ("mesh", "uniform"), ("mesh", "nonuniform"), ("mesh", "reflection"),
    ("box", "uniform"), ("base", "uniform"), ("root", "uniform"),
    ("base", "cancel_children"),
])
def test_effective_geometry_refuted_after_rehash(geometry_source, tmp_path, physics, name, mutation):
    m = module()
    out = tmp_path / "converted"
    receipt = m.convert_model(geometry_source, out)
    Usd, Geom, _ = physics
    from pxr import Gf
    asset = out / receipt["usd_path"]
    stage = Usd.Stage.Open(str(asset))
    prim = stage.GetDefaultPrim() if name == "root" else next(p for p in stage.Traverse() if p.GetName() == name)
    if mutation.startswith("axis_"):
        prim.GetAttribute("axis").Set(mutation[-1].upper())
    else:
        scale = {"uniform": (10, 10, 10), "nonuniform": (2, 1, 1),
                 "reflection": (-1, 1, 1), "cancel_children": (2, 2, 2)}[mutation]
        Geom.Xformable(prim).AddScaleOp(opSuffix="mutation").Set(Gf.Vec3f(*scale))
        if mutation == "cancel_children":
            # A collider's world scale can hide a scaled rigid body. Body frame
            # admission must still reject it, including meshes/child bodies.
            for child in prim.GetChildren():
                if child.IsA(Geom.Xformable):
                    Geom.Xformable(child).AddScaleOp(opSuffix="cancelParentScale").Set(Gf.Vec3f(0.5))
            mesh = next(p for p in stage.Traverse() if p.GetName() == "mesh")
            assert m._deps().np.allclose(m._deps().np.linalg.norm(m._matrix(mesh, Geom.XformCache())[:3, :3], axis=0), [1, .8, 1.2])
    stage.GetRootLayer().Save()
    del prim, stage
    _rehash_bundle(m, out)
    with pytest.raises(ValueError, match="geom|radius|height|scale|axis|body|mesh|box"):
        m.validate_model(geometry_source, asset)
    with pytest.raises(ValueError, match="geom|radius|height|scale|axis|body|mesh|box"):
        m.check_model(geometry_source, out)


def test_collision_mesh_hull_limit_is_authored_not_left_to_schema_fallback(geometry_source, tmp_path, physics):
    # Newton resolves only authored limits; unauthored meshes get its 64-vertex
    # hull cap, which made the mirrored MicroDuck soles asymmetric.
    m = module()
    result = m.convert_model(geometry_source, tmp_path / "converted")
    assert result["validation"]["collision_meshes"] == 1
    assert result["validation"]["unlimited_collision_hulls"] == 1
    Usd, _, _ = physics
    stage = Usd.Stage.Open(str(tmp_path / "converted" / result["usd_path"]))
    mesh = next(p for p in stage.Traverse() if p.GetName() == "mesh")
    limit = mesh.GetAttribute("newton:maxHullVertices")
    assert limit.HasAuthoredValue() and limit.Get() == -1
    assert m.check_model(geometry_source, tmp_path / "converted")["validation"]["status"] == "confirmed"


def test_explicit_source_hull_limit_is_preserved(geometry_source, tmp_path, physics):
    import xml.etree.ElementTree as ET
    tree = ET.parse(geometry_source)
    tree.getroot().find("asset/mesh").set("maxhullvert", "16")
    tree.write(geometry_source)
    m = module()
    result = m.convert_model(geometry_source, tmp_path / "converted")
    assert result["validation"]["collision_meshes"] == 1
    assert result["validation"]["unlimited_collision_hulls"] == 0
    Usd, _, _ = physics
    stage = Usd.Stage.Open(str(tmp_path / "converted" / result["usd_path"]))
    mesh = next(p for p in stage.Traverse() if p.GetName() == "mesh")
    assert mesh.GetAttribute("newton:maxHullVertices").Get() == 16


@pytest.mark.parametrize("mutation", ["engine_cap", "cleared", "competing_mjc", "competing_physx"])
def test_hull_limit_loss_refuted_after_rehash(geometry_source, tmp_path, physics, mutation):
    from pxr import Sdf
    m = module()
    out = tmp_path / "converted"
    receipt = m.convert_model(geometry_source, out)
    Usd, _, _ = physics
    asset = out / receipt["usd_path"]
    stage = Usd.Stage.Open(str(asset))
    mesh = next(p for p in stage.Traverse() if p.GetName() == "mesh")
    if mutation == "engine_cap": mesh.GetAttribute("newton:maxHullVertices").Set(64)
    if mutation == "cleared": mesh.GetAttribute("newton:maxHullVertices").Clear()
    if mutation == "competing_mjc": mesh.CreateAttribute("mjc:maxhullvert", Sdf.ValueTypeNames.Int).Set(64)
    if mutation == "competing_physx":
        mesh.CreateAttribute("physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(64)
    stage.GetRootLayer().Save()
    _rehash_bundle(m, out)
    with pytest.raises(ValueError, match="hull vertex limit"):
        m.validate_model(geometry_source, asset)
    with pytest.raises(ValueError):
        m.check_model(geometry_source, out)


@pytest.mark.parametrize("name", ["base_collision", "tip_collision", "cylinder"])
def test_equivalent_uniform_primitive_encoding_is_admitted(geometry_source, tmp_path, physics, name):
    m = module()
    out = tmp_path / "converted"
    receipt = m.convert_model(geometry_source, out)
    Usd, Geom, _ = physics
    from pxr import Gf
    stage = Usd.Stage.Open(str(out / receipt["usd_path"]))
    prim = next(p for p in stage.Traverse() if p.GetName() == name)
    Geom.Xformable(prim).AddScaleOp(opSuffix="equivalent").Set(Gf.Vec3f(2))
    for attr_name in ("radius", "height"):
        attr = prim.GetAttribute(attr_name)
        if attr:
            attr.Set(attr.Get() / 2)
    stage.GetRootLayer().Save()
    del prim, stage
    _rehash_bundle(m, out)
    assert m.check_model(geometry_source, out)["validation"]["status"] == "confirmed"


def test_mesh_topology_material_and_exclusion_survive_then_losses_fail(source, tmp_path, physics):
    import struct
    import xml.etree.ElementTree as ET
    vertices = [(0.02, 0, 0), (0, 0.04, 0), (0, 0, 0.06), (0, 0, 0)]
    triangles = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    mesh = source.parent / "tetra.stl"
    mesh.write_bytes(b"fixture".ljust(80, b"\0") + struct.pack("<I", 4) + b"".join(
        struct.pack("<12fH", 0, 0, 0, *vertices[a], *vertices[b], *vertices[c], 0)
        for a, b, c in triangles))
    tree = ET.parse(source)
    root = tree.getroot()
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="tetra", file="tetra.stl", scale="1 0.8 1.2")
    ET.SubElement(asset, "material", name="paint", rgba="0.2 0.6 0.8 0.7", shininess="0.3")
    body = root.find("worldbody/body")
    ET.SubElement(body, "geom", name="mesh_contact", type="mesh", mesh="tetra", material="paint",
                  pos="-0.03 0.02 0.04", quat="0.9238795325 0 0.3826834324 0")
    ET.SubElement(body, "geom", name="box_contact", type="box", size="0.02 0.01 0.03", pos="0.1 0 0")
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="base", body2="link")
    tree.write(source)
    m = module()
    result = m.convert_model(source, tmp_path / "converted")
    assert result["validation"]["status"] == "confirmed"
    Usd, Geom, Physics = physics
    path = tmp_path / "converted" / result["usd_path"]
    stage = Usd.Stage.Open(str(path))
    mesh_prim = next(p for p in stage.Traverse() if p.GetName() == "mesh_contact")
    points = list(Geom.Mesh(mesh_prim).GetPointsAttr().Get())
    points[0] = (0, 0, 0.1)
    Geom.Mesh(mesh_prim).GetPointsAttr().Set(points)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="mesh"):
        m.validate_model(source, path)


@pytest.mark.parametrize("mutation", ["group_membership", "mask", "exclusion", "visual_color"])
def test_contact_and_color_metadata_alone_cannot_certify(source, tmp_path, physics, mutation):
    import xml.etree.ElementTree as ET
    tree = ET.parse(source)
    contact = ET.SubElement(tree.getroot(), "contact")
    ET.SubElement(contact, "exclude", body1="base", body2="link")
    tree.write(source)
    m = module()
    result = m.convert_model(source, tmp_path / "converted")
    Usd, Geom, Physics = physics
    path = tmp_path / "converted" / result["usd_path"]
    stage = Usd.Stage.Open(str(path))
    geom = next(p for p in stage.Traverse() if p.GetName() == "base_collision")
    if mutation == "group_membership":
        group = Physics.CollisionGroup(stage.GetPrimAtPath("/fixture/CollisionGroups/mask_1_1"))
        group.GetCollidersCollectionAPI().GetIncludesRel().SetTargets([])
    if mutation == "mask": geom.GetAttribute("mjc:contype").Set(2)
    if mutation == "exclusion":
        body = next(p for p in stage.Traverse() if p.GetName() == "base")
        Physics.FilteredPairsAPI(body).GetFilteredPairsRel().SetTargets([])
    if mutation == "visual_color": Geom.Gprim(geom).GetDisplayColorPrimvar().Set([(0, 0, 0)])
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError): m.validate_model(source, path)


def test_reproducible_receipt_rechecks_independent_source_and_no_clobber(source, tmp_path, physics):
    import hashlib
    import json
    m = module()
    first = m.convert_model(source, tmp_path / "a")
    second = m.convert_model(source, tmp_path / "b")
    receipt = tmp_path / "a/receipt.json"
    assert receipt.is_file(), "durable conversion receipt missing"
    assert first["outputs"] == second["outputs"], "location-independent deterministic output hashes"
    assert receipt.read_bytes() == (tmp_path / "b/receipt.json").read_bytes()
    assert first["receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert m.check_model(source, tmp_path / "a")["validation"]["status"] == "confirmed"
    with pytest.raises(ValueError, match="overwrite"):
        m.convert_model(source, tmp_path / "a")
    stored = json.loads(receipt.read_text())
    assert stored["live_engine_status"] == "unverified"
    usd = tmp_path / "a" / stored["usd_path"]
    usd.write_text(usd.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="hash|SHA"):
        m.check_model(source, tmp_path / "a")
    source.write_text(source.read_text().replace('mass="2"', 'mass="3"'))
    with pytest.raises(ValueError, match="source|hash|SHA"):
        m.check_model(source, tmp_path / "b")


def test_unnamed_geometry_identity_is_stable_without_changing_physics(source, tmp_path, physics):
    m = module()
    source.write_text(source.read_text().replace('name="base_collision"', '').replace('name="tip_collision"', ''))
    receipt = m.convert_model(source, tmp_path / "converted")
    assert receipt["validation"]["colliders"] == 2
    assert receipt["physical_delta"] == {}
    assert len(receipt["geometry_identity"]) == 2
    assert m.check_model(source, tmp_path / "converted")["validation"]["status"] == "confirmed"


def test_rejects_symlink_and_unsafe_resources_before_writes(source, tmp_path, physics):
    m = module()
    (source.parent / "link").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        m.convert_model(source, tmp_path / "a")
    assert not (tmp_path / "a").exists()
    (source.parent / "link").unlink()
    source.write_text(source.read_text().replace('<compiler angle="radian"/>', '<include file="../escaped.xml"/>'))
    with pytest.raises(ValueError, match="include|path|resource"):
        m.convert_model(source, tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_official_admission_is_offline_explicit_and_rejects_variants(tmp_path):
    m = module()
    for variant in ("rollers", "backlash", "groundcontact"):
        with pytest.raises(ValueError, match="variant"):
            m.convert(tmp_path / "missing", tmp_path / variant, variant=variant, accept_model_license=True)
    with pytest.raises(ValueError, match="license"):
        m.convert(tmp_path / "missing", tmp_path / "a")
    with pytest.raises(ValueError, match="admission"):
        m.convert(tmp_path / "missing", tmp_path / "a", accept_model_license=True)
    assert not (tmp_path / "a").exists()


@pytest.fixture
def admitted_source():
    import os
    location = os.environ.get("CASCADE_MICRODUCK_ADMITTED")
    if not location:
        pytest.skip("set CASCADE_MICRODUCK_ADMITTED to the verified external source tree")
    return Path(location)


def test_official_cli_roundtrip_real_pinned_model(admitted_source, tmp_path, physics):
    import json
    import os
    import subprocess
    import sys
    output = tmp_path / "official"
    cmd = [sys.executable, str(SCRIPT), "--admitted-source", str(admitted_source),
           "--destination", str(output)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(cmd + ["--convert", "--accept-model-license", "--collision-profile", "source"],
                               capture_output=True, text=True, env=env, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip(), "conversion CLI must produce a real receipt"
    result = json.loads(completed.stdout)
    assert result["ok"]
    assert result["validation"]["colliders"] == 70
    assert result["validation"]["hinge_joints"] == 14
    assert result["validation"]["rigid_bodies"] == 15
    assert result["validation"]["collision_meshes"] > 0
    assert result["validation"]["unlimited_collision_hulls"] == result["validation"]["collision_meshes"]
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    checked = subprocess.run(cmd + ["--check"], capture_output=True, text=True, env=env, timeout=180)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["receipt_sha256"] == result["receipt_sha256"]
    assert before == {p: p.stat().st_mtime_ns for p in before}
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["provenance"]["sources"]["microduck_rl"]["revision"] == module().SOURCE_PIN
    assert receipt["physical_delta"] == {}
    assert receipt["source_vs_current_contact_cfg_delta"]
    assert receipt["historical_checkpoint_training_match"] == "unverified"
    assert (output / "provenance/microduck_rl/LICENSE").is_file()
    assert (output / "provenance/microduck_rl/README.md").is_file()


def test_current_contact_cfg_applied_explicitly_not_actuator_tuning(admitted_source, tmp_path, physics):
    m = module()
    output = tmp_path / "velstand"
    result = m.convert(admitted_source, output, collision_profile="velstand", accept_model_license=True)
    assert result["collision_profile"] == "velstand"
    assert set(result["physical_delta"]) <= {"geom_condim", "geom_priority", "geom_friction"}
    assert result["physical_delta"] == result["source_vs_current_contact_cfg_delta"]
    import mujoco
    compiled = mujoco.MjModel.from_xml_path(str(output / result["effective_xml"]))
    for i in range(compiled.ngeom):
        name = compiled.geom(i).name
        if name.endswith("_servo_collision"):
            assert compiled.geom_condim[i] == 1
        if name in ("left_foot_collision", "right_foot_collision"):
            assert compiled.geom_condim[i] == 3
            assert compiled.geom_priority[i] == 1
            assert compiled.geom_friction[i, 0] == 1
    assert result["locomotion_status"] == "unverified"
    assert m.check(admitted_source, output)["receipt_sha256"] == result["receipt_sha256"]


@pytest.mark.parametrize("field", ["physical_delta", "locomotion_status", "source-contract.json"])
def test_receipt_cannot_self_certify_metadata_even_with_updated_hashes(source, tmp_path, physics, field):
    import hashlib
    import json
    m = module()
    out = tmp_path / "converted"
    m.convert_model(source, out)
    receipt = json.loads((out / "receipt.json").read_text())
    if field == "source-contract.json":
        contract = json.loads((out / field).read_text())
        contract["body_mass"][1] = 999
        (out / field).write_text(json.dumps(contract))
        row = next(row for row in receipt["outputs"] if row["path"] == field)
        row.update(sha256=hashlib.sha256((out / field).read_bytes()).hexdigest(), size=(out / field).stat().st_size)
    elif field == "physical_delta":
        receipt[field] = {"body_mass": {"source": [1], "effective": [2]}}
    else:
        receipt[field] = "confirmed"
    content = json.dumps(receipt).encode()
    (out / "receipt.json").write_bytes(content)
    (out / "receipt.sha256").write_text(hashlib.sha256(content).hexdigest() + "\n")
    with pytest.raises(ValueError, match="metadata|contract|delta|unverified"):
        m.check_model(source, out)


def test_import_and_help_do_not_load_optional_physics_modules():
    import os
    import subprocess
    import sys
    code = ("import runpy,sys; runpy.run_path(sys.argv[1]); "
            "assert not any(x in sys.modules for x in ['numpy','mujoco','pxr','newton','warp','isaacsim'])")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-S", "-c", code, str(SCRIPT)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    help_result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, env=env)
    assert help_result.returncode == 0 and "--admitted-source" in help_result.stdout
