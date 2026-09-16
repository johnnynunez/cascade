"""Pre-play kitchen authoring for scripts/isaac_bridge.py; no Isaac imports.

The caller owns play, settling, reset and physical readback. This module never
moves a live body. Lengths and returned body-center spawns are in metres.
"""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


@dataclass(frozen=True)
class PropSpec:
    name: str
    position: tuple[float, float, float]
    color: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    visual_dimensions: tuple[float, float, float]
    visual_offset: tuple[float, float, float]
    mass: float
    path: str

    def bridge_tuple(self):
        """Legacy PROPS record. Full non-square dimensions remain on this spec."""
        return (self.name, self.position, self.color,
                max(self.dimensions[:2]), self.dimensions[2])


def read_config(path):
    config_path = Path(path).resolve()
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if config.get("version") != 1:
        raise ValueError("Unsupported kitchen scene configuration")
    cube_positions = config.get("cube_positions", {})
    if not isinstance(cube_positions, dict) or set(cube_positions) - {"pink_cube", "green_cube"}:
        raise ValueError("Kitchen cube positions must name existing cube bodies")
    for position in cube_positions.values():
        if (not isinstance(position, list) or len(position) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in position)):
            raise ValueError("Kitchen cube positions must be three finite metres")
    cube_dimensions = config.get("cube_dimensions", {})
    if not isinstance(cube_dimensions, dict) or set(cube_dimensions) - {"pink_cube", "green_cube"}:
        raise ValueError("Kitchen cube dimensions must name existing cube bodies")
    for dimensions in cube_dimensions.values():
        if (not isinstance(dimensions, list) or len(dimensions) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v <= 0 for v in dimensions)
                or dimensions[0] != dimensions[1]):
            raise ValueError("Kitchen cube dimensions must have a square footprint and positive metres")
    surfaces = config.get("cube_surfaces", {})
    if (not isinstance(surfaces, dict) or set(surfaces) - {"green_cube", "pink_cube"}
            or any(style not in {"stone", "puzzle"} for style in surfaces.values())
            or surfaces.get("green_cube", "stone") != "stone"):
        raise ValueError("Cube surfaces must preserve green stone and optional pink puzzle styling")
    config["_directory"] = str(config_path.parent)
    config["_identity"] = {"scene_config": str(config_path),
                           "scene_config_sha256": hashlib.sha256(raw).hexdigest()}
    return config


def _asset(config, name):
    path = (Path(config["_directory"]) / name).resolve()
    kitchen = Path(__file__).resolve().parent
    if not path.is_relative_to(kitchen) or not path.is_file():
        raise ValueError(f"Kitchen asset must be a local kitchen file: {path}")
    return path


def _matrix(prim, xyz):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*xyz)))


def convex_points(kind, dimensions, segments=24, rings=12):
    """Small, baked-vertex convex proxies; no scale ops or external meshes."""
    a, b, c = [v / 2 for v in dimensions]
    if kind == "cylinder":
        pts = [(a * math.cos(2 * math.pi * j / segments),
                b * math.sin(2 * math.pi * j / segments), z)
               for z in (-c, c) for j in range(segments)]
        faces = [list(reversed(range(segments))), list(range(segments, 2 * segments))]
        faces += [[j, (j + 1) % segments, (j + 1) % segments + segments,
                   j + segments] for j in range(segments)]
    elif kind == "ellipsoid":
        pts = [(0, 0, -c)]
        for i in range(1, rings):
            phi = -math.pi / 2 + math.pi * i / rings
            for j in range(segments):
                theta = 2 * math.pi * j / segments
                pts.append((a * math.cos(phi) * math.cos(theta),
                            b * math.cos(phi) * math.sin(theta), c * math.sin(phi)))
        pts.append((0, 0, c))
        faces = [[0, 1 + (j + 1) % segments, 1 + j] for j in range(segments)]
        for i in range(rings - 2):
            lo, hi = 1 + i * segments, 1 + (i + 1) * segments
            faces += [[lo + j, lo + (j + 1) % segments,
                       hi + (j + 1) % segments, hi + j] for j in range(segments)]
        lo, top = 1 + (rings - 2) * segments, len(pts) - 1
        faces += [[lo + j, lo + (j + 1) % segments, top] for j in range(segments)]
    else:
        raise ValueError(f"Unknown proxy {kind}")
    return pts, faces


def mesh_geometry(mesh, points, faces):
    pts = [Gf.Vec3f(*p) for p in points]
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(pts))


def configure_cubes(config, defaults):
    """Bind spawn and physical/visual dimensions to the hashed scene config."""
    positions, dimensions = config.get("cube_positions", {}), config.get("cube_dimensions", {})
    return [(name, tuple(positions.get(name, pos)), color,
             dimensions.get(name, (width, width, height))[0],
             dimensions.get(name, (width, width, height))[2])
            for name, pos, color, width, height in defaults]


def _surface_material(stage, path, color, *, roughness=.58, opacity=1.0, vertex_color=False):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if vertex_color:
        reader = UsdShade.Shader.Define(stage, path + "/GrainColor")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*color))
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _subdivide_surface(mesh, divisions):
    """Add visual sampling vertices without changing any planar face or bounds."""
    original = [Gf.Vec3f(*p) for p in mesh.GetPointsAttr().Get()]
    counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
    points, faces, cursor = [], [], 0
    for count in counts:
        face = [original[i] for i in indices[cursor:cursor + count]]
        cursor += count
        start = len(points)
        if count != 4:
            points.extend(face)
            faces.append(list(range(start, start + count)))
            continue
        for row in range(divisions + 1):
            v = row / divisions
            for column in range(divisions + 1):
                u = column / divisions
                points.append((1-u)*(1-v)*face[0] + u*(1-v)*face[1]
                              + u*v*face[2] + (1-u)*v*face[3])
        for row in range(divisions):
            for column in range(divisions):
                a = start + row*(divisions+1) + column
                faces.append([a, a+1, a+divisions+2, a+divisions+1])
    mesh_geometry(mesh, points, faces)


def _grain_colors(mesh, color, *, wood=False):
    """Deterministic subtle pigment/wood grain authored as native USD primvars."""
    colors = []
    for p in mesh.GetPointsAttr().Get():
        x, y, z = map(float, p)
        grain = math.sin(2300*y + 1.8*math.sin(180*x) + .7*math.sin(260*z))
        pores = math.sin(917*x + 1319*y + 1877*z) * math.sin(1747*x - 971*y + 2719*z)
        variation = 1 + (.085 if wood else .04)*grain + .035*pores
        colors.append(Gf.Vec3f(*[max(0, min(1, c*variation)) for c in color]))
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(colors)


def decorate_cube(stage, mesh, color, *, wood=False, style="stone"):
    """Original rounded targets or box wood; boundingCube remains exact."""
    if not wood:
        return decorate_cube_surface(stage, mesh, color, style=style)
    _subdivide_surface(mesh, 16)
    _grain_colors(mesh, color, wood=wood)
    name = "Birch" if wood else mesh.GetPrim().GetName() + "_paint"
    material = _surface_material(stage, "/World_Props/Looks/" + name, color,
                                 roughness=.64 if wood else .56, vertex_color=True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _prepare_box(stage, config, cube):
    box = config["open_box"]
    x, y = box["center_xy_m"]
    width, depth, height = box["outer_dimensions_m"]
    wall, floor = box["wall_thickness_m"], box["base_thickness_m"]
    if not (0 < floor < height and 0 < 2*wall < min(width, depth)):
        raise ValueError("Open box must have a positive floor and open cavity")
    expected = (width-2*wall, depth-2*wall, height-floor)
    if any(abs(a-b) > 1e-8 for a, b in zip(expected, box["interior_dimensions_m"])):
        raise ValueError("Open box metadata differs from its authored cavity")
    root = stage.DefinePrim("/World_Props/open_box", "Xform")
    root.SetCustomDataByKey("description", "Open birch box with four walls and a separate floor; cavity faces +Z")
    root.SetCustomDataByKey("support_top_z_m", float(floor))
    parts = [
        ("floor", (x,y,floor/2), (width,depth,floor)),
        ("left", (x-(width-wall)/2,y,(height+floor)/2), (wall,depth,height-floor)),
        ("right", (x+(width-wall)/2,y,(height+floor)/2), (wall,depth,height-floor)),
        ("front", (x,y-(depth-wall)/2,(height+floor)/2), (width-2*wall,wall,height-floor)),
        ("back", (x,y+(depth-wall)/2,(height+floor)/2), (width-2*wall,wall,height-floor)),
    ]
    color = tuple(box["color"])
    for name, position, dimensions in parts:
        mesh = cube("/World_Props/open_box/" + name, position, dimensions, color, dynamic=False)
        decorate_cube(stage, mesh, color, wood=True)


def _prepare_target_pad(stage, config):
    pad = config["target_pad"]
    x, y = pad["center_xy_m"]
    size, border, z = pad["outer_size_m"], pad["border_width_m"], pad["surface_z_m"]
    if not (0 < border < size/2 and 0 < z <= .002 and 0 < pad["fill_opacity"] < 1):
        raise ValueError("Target pad must be a thin square with a translucent interior")
    root = stage.DefinePrim("/World_Props/green_square", "Xform")
    root.SetCustomDataByKey("description", "Green square painted on the countertop; visual marker only")
    half, inner = size/2, size/2-border
    fill = UsdGeom.Mesh.Define(stage, str(root.GetPath()) + "/fill")
    mesh_geometry(fill, [(x-inner,y-inner,z),(x+inner,y-inner,z),
                         (x+inner,y+inner,z),(x-inner,y+inner,z)], [[0,1,2,3]])
    fill.CreateDisplayColorAttr([Gf.Vec3f(*pad["color"])])
    fill.CreateDisplayOpacityAttr([float(pad["fill_opacity"])])
    fill.CreateDoubleSidedAttr(True)
    material = _surface_material(stage, "/World_Props/Looks/GreenSquareFill", pad["color"],
                                 roughness=.72, opacity=float(pad["fill_opacity"]))
    UsdShade.MaterialBindingAPI.Apply(fill.GetPrim()).Bind(material)
    frame = UsdGeom.Mesh.Define(stage, str(root.GetPath()) + "/border")
    points = [(x+a,y+b,z) for a,b in [(-half,-half),(half,-half),(half,half),(-half,half),
                                    (-inner,-inner),(inner,-inner),(inner,inner),(-inner,inner)]]
    mesh_geometry(frame, points, [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])
    frame.CreateDisplayColorAttr([Gf.Vec3f(*pad["color"])])
    frame.CreateDoubleSidedAttr(True)
    material = _surface_material(stage, "/World_Props/Looks/GreenSquareBorder", pad["color"], roughness=.62)
    UsdShade.MaterialBindingAPI.Apply(frame.GetPrim()).Bind(material)


def _scale_and_texture_visual(stage, visual, scale):
    """Bake scale into referenced vertices; this Hydra build drops scale ops."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Kitchen visual scale must be positive and at most one")
    for prim in Usd.PrimRange(visual):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = [Gf.Vec3f(*p)*scale for p in mesh.GetPointsAttr().Get()]
        mesh.GetPointsAttr().Set(points)
        mesh.GetExtentAttr().Set(UsdGeom.PointBased(mesh).ComputeExtent(points))
        _subdivide_surface(mesh, 4)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        surface, _, _ = material.ComputeSurfaceSource()
        if not surface:
            continue
        color = surface.GetInput("diffuseColor").Get() or Gf.Vec3f(.5)
        _grain_colors(mesh, color)
        reader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/GrainColor")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*color))
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        surface.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")


def prepare_scene(stage, config, *, cube, bind_pmat, base_z, is_playing):
    """Author background, counter, open box, marked square and dynamic props.

    Call after defining _cube/_bind_pmat/_STATIC_PROPS, instead of the demo
    bin/table block, and before app_utils.play. _cube preserves the bridge's
    material, mass, CCD, velocity caps and static-furniture registration.
    Append returned records to PROPS *after* the baseline cube creation loop.
    """
    if is_playing:
        raise RuntimeError("Kitchen mutations are permitted only before play")
    if abs(base_z) > 1e-5 or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-9:
        raise ValueError("Kitchen requires the shipped metre-scale robot with base at zero")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("Kitchen requires a Z-up stage")
    if stage.GetPrimAtPath("/Kitchen") or stage.GetPrimAtPath("/World_Props/table"):
        raise ValueError("Kitchen is already installed, or demo furniture was authored first")
    if not stage.GetPrimAtPath(config["robot_prim"]):
        raise ValueError("Expected shipped reBot articulation root is missing")

    background = stage.DefinePrim("/Kitchen", "Xform")
    background.GetReferences().AddReference(str(_asset(config, config["background"])), "/Kitchen")
    # Fail closed if future vendor/plugin composition reintroduces physics.
    for prim in Usd.PrimRange(background):
        schemas = prim.GetMetadata("apiSchemas")
        names = schemas.GetAppliedItems() if schemas else []
        if any(str(s).lower().startswith(("physics", "physx", "newton")) for s in names):
            raise ValueError(f"Background physics schema remains: {prim.GetPath()}")
        if prim.GetTypeName().startswith(("Physics", "Physx", "OmniGraph")):
            raise ValueError(f"Background simulation prim remains: {prim.GetPath()}")

    counter = config["counter"]
    lo, hi = counter["min"], counter["max"]
    slab = cube("/World_Props/table", tuple((a + b) / 2 for a, b in zip(lo, hi)),
                tuple(b - a for a, b in zip(lo, hi)), (0.55, 0.45, 0.35), dynamic=False)
    UsdGeom.Imageable(slab.GetPrim()).CreateVisibilityAttr("invisible")
    if "open_box" in config:
        _prepare_box(stage, config, cube)
    else:
        shelf = config["shelf"]
        cube("/World_Props/shelf", tuple(shelf["position"]), tuple(shelf["dimensions"]),
             tuple(shelf["color"]), dynamic=False)
    if "target_pad" in config:
        _prepare_target_pad(stage, config)

    specs = []
    for item in config["props"]:
        name = item["name"]
        if name not in {"tomato_can", "lemon", "orange"}:
            raise ValueError(f"Unexpected prop name {name}")
        path = f"/World_Props/{name}"
        if stage.GetPrimAtPath(path):
            raise ValueError(f"Prop already exists: {path}")
        # Obtain the exact bridge dynamic-body setup, then replace its box
        # geometry by one compound body's visual child and convex collider.
        root_mesh = cube(path, tuple(item["position"]), tuple(item["dimensions"]),
                         tuple(item["color"]), dynamic=True, mass=item["mass"])
        body = root_mesh.GetPrim()
        body.RemoveAPI(UsdPhysics.CollisionAPI)
        body.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        body.SetTypeName("Xform")
        for attr in ("points", "faceVertexCounts", "faceVertexIndices", "extent",
                     "subdivisionScheme", "primvars:displayColor", "physics:approximation"):
            body.RemoveProperty(attr)
        UsdPhysics.RigidBodyAPI(body).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI(body).CreateCenterOfMassAttr(Gf.Vec3f(0))
        visual = stage.DefinePrim(path + "/Visual", "Xform")
        visual.GetReferences().AddReference(str(_asset(config, item["asset"])), "/Prop")
        if "visual_scale" in item:
            _scale_and_texture_visual(stage, visual, float(item["visual_scale"]))
        _matrix(visual, item["visual_offset"])
        collision = UsdGeom.Mesh.Define(stage, path + "/Collision")
        mesh_geometry(collision, *convex_points(item["proxy"], item["dimensions"]))
        collision.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
        bind_pmat(collision.GetPrim())
        specs.append(PropSpec(name, tuple(item["position"]), tuple(item["color"]),
                              tuple(item["dimensions"]), tuple(item["visual_dimensions"]),
                              tuple(item["visual_offset"]), float(item["mass"]), path))
    return specs


# Original rounded painted-stone and unbranded puzzle surfaces.
# The boundingCube collider retains the existing exact AABB.
def _coordinates(half, radius, divisions, bevel_divisions, puzzle):
    inner = half - radius
    values = [-inner + 2 * inner * i / divisions for i in range(divisions + 1)]
    values += [sign * (inner + radius * i / bevel_divisions)
               for sign in (-1, 1) for i in range(1, bevel_divisions + 1)]
    if puzzle:
        # Include both sides of each visual seam, rather than aliasing the
        # 3x3 pattern into whichever coarse surface vertices happen to exist.
        for seam in (-half / 3, half / 3):
            # The outer samples keep each facelet's small polished edge from
            # smearing into the broad satin-painted tile at camera distance.
            values += [seam + d for d in (-.00065, -.00035, -.00020, 0,
                                         .00020, .00035, .00065)]
    return sorted(set(values))


def rounded_box_geometry(half_extents, *, radius=.0009, divisions=16,
                         bevel_divisions=3, puzzle=False):
    """Watertight rounded box, with original extrema and broad flat faces.

    Project a subdivided box onto the Minkowski sum of an inset box and a
    radius sphere. Shared edges are welded; analytical normals are smooth
    around the rounding and exactly axial across each flat gripping face.
    """
    half = tuple(map(float, half_extents))
    if (len(half) != 3 or not all(math.isfinite(x) and x > 0 for x in half)
            or not math.isfinite(radius) or not 0 < radius < min(half) / 4
            or not isinstance(divisions, int) or divisions < 4
            or not isinstance(bevel_divisions, int) or bevel_divisions < 2):
        raise ValueError("Invalid cube dimensions, radius or surface resolution")
    coords = [_coordinates(h, radius, divisions, bevel_divisions, puzzle) for h in half]
    inner = [h - radius for h in half]
    points, normals, faces, axes, welded = [], [], [], [], {}
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1, 1):
            grid = []
            for y in coords[v]:
                row = []
                for x in coords[u]:
                    p = [0., 0., 0.]
                    p[axis], p[u], p[v] = sign * half[axis], x, y
                    q = [min(inner[k], max(-inner[k], p[k])) for k in range(3)]
                    d = [p[k] - q[k] for k in range(3)]
                    norm = math.sqrt(sum(z * z for z in d))
                    normal = tuple(z / norm for z in d)
                    rounded = tuple(q[k] + radius * normal[k] for k in range(3))
                    key = tuple(round(z, 12) for z in rounded)
                    if key not in welded:
                        welded[key] = len(points)
                        points.append(Gf.Vec3f(*rounded))
                        normals.append(Gf.Vec3f(*normal))
                    row.append(welded[key])
                grid.append(row)
            for j in range(len(coords[v]) - 1):
                for i in range(len(coords[u]) - 1):
                    f = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]]
                    faces.append(f if sign > 0 else list(reversed(f)))
                    axes.append(axis)
    return points, faces, normals, axes


def _stone_noise(x, y, z, *, scale, seed):
    """Small deterministic mineral variation without repeating sine bands."""
    coords = (x / scale, y / scale, z / scale)
    origin = tuple(math.floor(value) for value in coords)
    fraction = tuple(value - index for value, index in zip(coords, origin))
    blend = tuple(value * value * (3 - 2 * value) for value in fraction)
    result = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix, iy, iz = origin[0] + dx, origin[1] + dy, origin[2] + dz
                hashed = (ix * 1619 ^ iy * 31337 ^ iz * 6971 ^ seed * 1013) & 0xffffffff
                hashed = ((hashed ^ (hashed >> 13)) * 1274126177) & 0xffffffff
                value = ((hashed ^ (hashed >> 16)) & 0xffffffff) / 4294967295.0
                weight = ((blend[0] if dx else 1 - blend[0])
                          * (blend[1] if dy else 1 - blend[1])
                          * (blend[2] if dz else 1 - blend[2]))
                result += weight * value
    return 2 * result - 1


def _pigment(point, color, *, half, axis, puzzle, strength=1.0):
    x, y, z = point
    # Two bounded spatial bands: small aggregate + fine pigment variation.
    grain = math.sin(371*x + 593*y + 719*z) * math.sin(827*x - 419*y + 557*z)
    fine = math.sin(1831*x - 1291*y + 2203*z) * math.sin(1277*x + 1949*y - 1487*z)
    variation = 1 + .021 * grain + .012 * fine
    roughness = .60 + .025 * grain + .012 * fine
    if strength != 1.0:
        # Two smooth mineral scales keep green readable while replacing the
        # old broad wave pattern with restrained painted-stone aggregate.
        aggregate = _stone_noise(x, y, z, scale=.006, seed=11)
        mineral = _stone_noise(x, y, z, scale=.0025, seed=29)
        fleck = max(0., (mineral - .25) / .75)
        variation = .94 + .16 * aggregate + .08 * mineral - .14 * fleck
        roughness = .62 + .065 * aggregate + .045 * mineral
    seam = 0.
    if puzzle:
        u, v = (axis + 1) % 3, (axis + 2) % 3
        distance = min(abs(point[a] - boundary) for a in (u, v)
                       for boundary in (-half[a] / 3, half[a] / 3))
        seam = max(0., min(1., (.00035 - distance) / .00015))
        tile = [min(2, max(0, int((point[a] / half[a] + 1) * 1.5))) for a in (u, v)]
        facelet = (tile[0] + 2 * tile[1] + axis) % 3
        # Original unbranded pink facelets: rose, raspberry and pale pink,
        # with restrained aggregate that survives the live camera reduction.
        aggregate = (math.sin(163*x + 227*y + 191*z + .8)
                     * math.sin(271*x - 179*y + 139*z + 1.3))
        variation = 1 + .040 * grain + .018 * fine + .060 * aggregate
        tint = ((.88, .91, .95), (.975, 1.06, 1.035), (1.02, .97, 1.00))[facelet]
        edge = max(0., 1 - abs(distance - .00038) / .00027)
        paint = [c * t * variation * (1 + .055 * edge) for c, t in zip(color, tint)]
        # Muted plum joints and a smoother narrow facelet edge add separation
        # without separate bodies, painted logos or external texture assets.
        joint = (.16, .025, .095)
        rgb = [max(0., min(1., p * (1 - seam) + j * seam)) for p, j in zip(paint, joint)]
        roughness = (.57, .61, .59)[facelet] + .030 * grain + .015 * fine + .035 * aggregate
        roughness = roughness * (1 - seam) + .78 * seam - .085 * edge * (1 - seam)
        return Gf.Vec3f(*rgb), roughness
    rgb = [max(0., min(1., c * variation * (1 - .72 * seam))) for c in color]
    return Gf.Vec3f(*rgb), roughness


def decorate_cube_surface(stage, mesh, color, *, style="stone", radius=None):
    """Apply to the existing dynamic green_cube/pink_cube, before play only.

    style='stone' is the default for both; 'puzzle' is allowed only for pink.
    This helper intentionally does not handle the open box's wood surfaces.
    Caller owns the pre-play guard and subsequent live acceptance proof.
    """
    prim = mesh.GetPrim()
    name = prim.GetName()
    if name not in {"green_cube", "pink_cube"}:
        raise ValueError("Only canonical green_cube and pink_cube are supported")
    if style not in {"stone", "puzzle"} or (style == "puzzle" and name != "pink_cube"):
        raise ValueError("Puzzle styling is supported only on pink_cube")
    if prim.GetAttribute("physics:approximation").Get() != "boundingCube":
        raise ValueError("Existing boundingCube collision is required")
    old = mesh.GetPointsAttr().Get()
    if not old:
        raise ValueError("Existing cube points are required")
    lo = [min(float(p[k]) for p in old) for k in range(3)]
    hi = [max(float(p[k]) for p in old) for k in range(3)]
    if any(abs(lo[k] + hi[k]) > 1e-8 for k in range(3)):
        raise ValueError("Existing cube must be centred at its local origin")
    half = tuple((hi[k] - lo[k]) / 2 for k in range(3))
    if abs(half[0] - half[1]) > 1e-8:
        raise ValueError("Existing cube requires its square footprint")
    puzzle = style == "puzzle"
    if radius is None:
        radius = .0015 if name == "green_cube" else (.00125 if puzzle else .0009)
    strength = 2.5 if name == "green_cube" else 1.0
    points, faces, normals, axes = rounded_box_geometry(half, radius=radius, puzzle=puzzle)
    indices = [i for face in faces for i in face]
    colors, roughness = [], []
    for face, axis in zip(faces, axes):
        for index in face:
            rgb, rough = _pigment(points[index], color, half=half, axis=axis, puzzle=puzzle, strength=strength)
            colors.append(rgb)
            roughness.append(rough)
    # No scale operations and no external texture dependencies.
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    display = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.faceVarying)
    display.SetInterpolation(UsdGeom.Tokens.faceVarying)
    display.Set(colors)
    rough = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("surfaceRoughness", Sdf.ValueTypeNames.FloatArray,
                                                UsdGeom.Tokens.faceVarying)
    rough.Set(roughness)
    path = "/World_Props/Looks/" + name + "_rounded_" + style
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    for field, varname, shaderid, valuetype, fallback in (
        ("diffuseColor", "displayColor", "UsdPrimvarReader_float3", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
        ("roughness", "surfaceRoughness", "UsdPrimvarReader_float", Sdf.ValueTypeNames.Float, .60),
    ):
        reader = UsdShade.Shader.Define(stage, path + "/" + field)
        reader.CreateIdAttr(shaderid)
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(varname)
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3 if field == "diffuseColor" else valuetype).Set(fallback)
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3 if field == "diffuseColor" else valuetype)
        surface.CreateInput(field, valuetype).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return {"style": style, "radius_m": radius, "dimensions_m": [2*h for h in half],
            "vertices": len(points), "quads": len(faces), "flat_face_spans_m": [2*(h-radius) for h in half]}
