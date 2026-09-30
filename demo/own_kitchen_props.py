"""Original procedural citrus and ceramic bowl, in metres with Z up.

All meshes, pigment and glaze are authored here; there are no image, asset,
network or simulation dependencies. The USD imports are delayed so the shape
and provenance checks also run without an Isaac installation. These functions
author visuals only. The scene owns the separate dynamic collision proxies,
body masses, contact materials, placement and reset behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


ORANGE_DIMENSIONS = (0.05163983628153801, 0.05232211947441101, 0.05276046320796013)
PROVENANCE = "Original procedural CASCADE geometry and materials; no external assets."


@dataclass(frozen=True)
class Surface:
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    normals: tuple[tuple[float, float, float], ...]
    colors: tuple[tuple[float, float, float], ...] = ()
    roughness: tuple[float, ...] = ()


def _dimensions(values):
    values = tuple(float(value) for value in values)
    if len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Dimensions must contain three positive finite metres")
    return values


def _hash(x, y, z, seed):
    value = (x * 1619 ^ y * 31337 ^ z * 6971 ^ seed * 1013) & 0xffffffff
    value = ((value ^ (value >> 13)) * 1274126177) & 0xffffffff
    return ((value ^ (value >> 16)) & 0xffffffff) / 4294967295.0


def _noise(point, scale, seed):
    coordinate = tuple(value / scale for value in point)
    cell = tuple(math.floor(value) for value in coordinate)
    blend = [value - origin for value, origin in zip(coordinate, cell)]
    blend = [value * value * (3 - 2 * value) for value in blend]
    result = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = ((blend[0] if dx else 1 - blend[0])
                          * (blend[1] if dy else 1 - blend[1])
                          * (blend[2] if dz else 1 - blend[2]))
                result += weight * _hash(cell[0] + dx, cell[1] + dy, cell[2] + dz, seed)
    return result


def _normals(points, faces):
    """Area weighted outward normals, including the welded sphere poles."""
    sums = [[0.0, 0.0, 0.0] for _ in points]
    for face in faces:
        a = points[face[0]]
        for index in range(1, len(face) - 1):
            b, c = points[face[index]], points[face[index + 1]]
            u, v = [b[k] - a[k] for k in range(3)], [c[k] - a[k] for k in range(3)]
            normal = (u[1] * v[2] - u[2] * v[1],
                      u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0])
            for vertex in (face[0], face[index], face[index + 1]):
                for axis in range(3):
                    sums[vertex][axis] += normal[axis]
    output = []
    for normal in sums:
        length = math.sqrt(sum(value * value for value in normal))
        if length <= 1e-15:
            raise ValueError("Degenerate visual surface")
        output.append(tuple(value / length for value in normal))
    return tuple(output)


def _surface(points, faces, colors=(), roughness=()):
    points, faces = tuple(points), tuple(tuple(face) for face in faces)
    return Surface(points, faces, _normals(points, faces), tuple(colors), tuple(roughness))


def _sphere_topology(segments, rings):
    faces = [(0, 1 + (j + 1) % segments, 1 + j) for j in range(segments)]
    for ring in range(rings - 2):
        low, high = 1 + ring * segments, 1 + (ring + 1) * segments
        faces.extend((low + j, low + (j + 1) % segments,
                      high + (j + 1) % segments, high + j) for j in range(segments))
    low, top = 1 + (rings - 2) * segments, 1 + (rings - 1) * segments
    faces.extend((low + j, low + (j + 1) % segments, top) for j in range(segments))
    return faces


def orange_surface(dimensions=ORANGE_DIMENSIONS, *, seed=29, segments=256, rings=128):
    """Closed, slightly oblate fruit within the unchanged grasp proxy bounds.

    The bottom touches -height/2. A 2.7% compression and matching downward
    shift leave space for the recessed calyx and tiny stem inside height/2.
    Shallow, irregular pores displace the actual mesh, so skin has detail in
    both ordinary preview-surface renderers and close-up silhouettes.
    """
    a, b, half_height = (value / 2 for value in _dimensions(dimensions))
    if (not isinstance(segments, int) or segments < 16 or segments % 4
            or not isinstance(rings, int) or rings < 8 or rings % 2):
        raise ValueError("Citrus resolution needs segments divisible by four and even rings")
    if not isinstance(seed, int):
        raise ValueError("Citrus seed must be an integer")
    c, center_z = .973 * half_height, -.027 * half_height
    size = min(a, b, half_height) / .026
    directions = [(0.0, 0.0, -1.0)]
    for ring in range(1, rings):
        phi = -math.pi / 2 + math.pi * ring / rings
        directions.extend((math.cos(phi) * math.cos(2 * math.pi * j / segments),
                           math.cos(phi) * math.sin(2 * math.pi * j / segments),
                           math.sin(phi)) for j in range(segments))
    directions.append((0.0, 0.0, 1.0))
    points, colors, roughness = [], [], []
    for nx, ny, nz in directions:
        sample = (a * nx, b * ny, c * nz)
        fine = _noise(sample, .00082 * size, seed)
        medium = _noise(sample, .0026 * size, seed + 19)
        broad = _noise(sample, .0065 * size, seed + 7)
        # Bounded, inward dimples preserve the proxy silhouette. The poles
        # are welded and never receive a seam-dependent displacement.
        pore = max(0.0, (fine - .45) / .45) ** 1.4
        depression = (.012 * pore + .001 * broad) * (1 - nz * nz)
        recess = .00025 * size * max(0.0, (nz - .91) / .09) ** 2
        points.append((a * nx * (1 - depression), b * ny * (1 - depression),
                       center_z + c * nz * (1 - depression) - recess))
        # Broad ripening patches and smaller peel pigment are deliberately
        # independent of the displacement: pores should not resemble painted
        # polka dots. Contrast survives the kitchen's soft ceiling lighting.
        pigment = (.97 + .065 * (2 * broad - 1) + .07 * (2 * medium - 1)
                   + .10 * (2 * fine - 1))
        colors.append((min(.98, .96 * pigment), .278 * pigment + .025 * broad - .014 * pore,
                       .014 + .012 * broad))
        roughness.append(.40 + .06 * fine + .025 * medium)
    return _surface(points, _sphere_topology(segments, rings), colors, roughness)


def _lathe(profile, segments=96):
    """Closed surface of revolution; profile runs underside, outer rim, inside."""
    points, rows, faces = [], [], []
    for radius, z in profile:
        if radius == 0:
            rows.append([len(points)])
            points.append((0.0, 0.0, z))
        else:
            row = []
            for j in range(segments):
                theta = 2 * math.pi * j / segments
                row.append(len(points))
                points.append((radius * math.cos(theta), radius * math.sin(theta), z))
            rows.append(row)
    for low, high in zip(rows, rows[1:]):
        for j in range(segments):
            following = (j + 1) % segments
            if len(low) == 1:
                faces.append((low[0], high[following], high[j]))
            elif len(high) == 1:
                faces.append((low[j], low[following], high[0]))
            else:
                faces.append((low[j], low[following], high[following], high[j]))
    return _surface(points, faces)


# The underside has a flat annular foot exactly on z=0. The bowl's wall and
# rounded lip have real thickness; no double-sided sheet hides an open mesh.
_BOWL_PROFILE = (
    (0.0, .0030), (.045, .0030), (.049, .0020), (.050, .0000),
    (.055, .0000), (.057, .0030), (.070, .0040), (.085, .0055),
    (.100, .0080), (.120, .0140), (.138, .0200), (.147, .0240),
    (.149, .0250), (.150, .0260), (.1497, .0270), (.1487, .0275),
    (.1477, .0272), (.1460, .0260), (.137, .0220), (.120, .0170),
    (.100, .0110), (.085, .0080), (.070, .0065), (.050, .0055),
    (.030, .0050), (0.0, .0050),
)
# Keep the annular foot on the table while giving the original platter its
# reference depth. Fruit support is recomputed against this inner profile.
_BOWL_PROFILE = tuple((radius, height * 2.055) for radius, height in _BOWL_PROFILE)


def bowl_surface(*, diameter=.30, segments=128):
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Bowl diameter must be positive finite metres")
    if not isinstance(segments, int) or segments < 16 or segments % 4:
        raise ValueError("Bowl segments must be divisible by four and at least sixteen")
    scale = diameter / .30
    return _lathe([(radius * scale, z * scale) for radius, z in _BOWL_PROFILE], segments)


def _material(stage, path, color, *, roughness=.5, metallic=0.0, skin=False):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if skin:
        for name, type_name, varname, output_type in (
            ("diffuseColor", "float3", "displayColor", Sdf.ValueTypeNames.Float3),
            ("roughness", "float", "peelRoughness", Sdf.ValueTypeNames.Float),
        ):
            reader = UsdShade.Shader.Define(stage, path + "/" + varname)
            reader.CreateIdAttr("UsdPrimvarReader_" + type_name)
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(varname)
            fallback = Gf.Vec3f(*color) if name == "diffuseColor" else roughness
            reader.CreateInput("fallback", output_type).Set(fallback)
            reader.CreateOutput("result", output_type)
            shader.GetInput(name).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mesh(stage, path, surface, material):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [Gf.Vec3f(*point) for point in surface.points]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in surface.faces])
    mesh.CreateFaceVertexIndicesAttr([vertex for face in surface.faces for vertex in face])
    mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in surface.normals])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(points))
    if surface.colors:
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
            [Gf.Vec3f(*color) for color in surface.colors])
    if surface.roughness:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "peelRoughness", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        ).Set(surface.roughness)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _root(stage, path, description):
    from pxr import Sdf, UsdGeom

    path = Sdf.Path(path)
    if not path.IsAbsoluteRootOrPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError("Visual root must be an absolute prim path")
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().SetCustomDataByKey("provenance", PROVENANCE)
    root.GetPrim().SetCustomDataByKey("description", description)
    root.GetPrim().SetCustomDataByKey("physicsStatus", "Visual only; no physics schemas or contacts")
    return root


def author_orange(stage, root_path, dimensions=ORANGE_DIMENSIONS, *, seed=29,
                  segments=256, rings=128):
    """Author an original orange centered on its existing dynamic body origin.

    Do not apply the previous imported asset's scale or offset to this root.
    Geometry, stem and normals are baked in metres; no scale ops are used.
    Returns the visual Xform, with no collision or rigid-body API attached.
    """
    dimensions = _dimensions(dimensions)
    surface = orange_surface(dimensions, seed=seed, segments=segments, rings=rings)
    root = _root(stage, root_path, "Original dimpled orange with a recessed calyx and small stem")
    path = str(root.GetPath())
    peel = _material(stage, path + "/Looks/Peel", (.95, .32, .018), skin=True)
    _mesh(stage, path + "/Peel", surface, peel)
    scale = min(dimensions) / .052
    # The pole recess puts the five-lobed calyx in the peel instead of on a
    # hovering disc. Its tiny stem remains below the unchanged proxy top.
    tip_z = .946 * dimensions[2] / 2 - .00025 * scale
    cap_points = [(0.0, 0.0, tip_z + .00005 * scale)]
    count = 40
    for j in range(count):
        theta = 2 * math.pi * j / count
        radius = (.00091 + .00030 * math.cos(5 * theta)) * scale
        cap_points.append((radius * math.cos(theta), radius * math.sin(theta),
                           tip_z + .00003 * scale - .00003 * math.cos(5 * theta) * scale))
    # Close the calyx beneath its visible top; all authored meshes remain
    # watertight even though the base lies safely within the orange.
    cap_points.append((0.0, 0.0, tip_z - .00012 * scale))
    cap_faces = [(0, 1 + j, 1 + (j + 1) % count) for j in range(count)]
    cap_faces += [(count + 1, 1 + (j + 1) % count, 1 + j) for j in range(count)]
    calyx = _material(stage, path + "/Looks/Calyx", (.19, .135, .043), roughness=.78)
    _mesh(stage, path + "/Calyx", _surface(cap_points, cap_faces), calyx)
    stem = _lathe([(0.0, tip_z), (.00036 * scale, tip_z),
                   (.00032 * scale, tip_z + .00038 * scale),
                   (.00024 * scale, tip_z + .00048 * scale),
                   (0.0, tip_z + .00049 * scale)], segments=20)
    bark = _material(stage, path + "/Looks/Stem", (.22, .12, .035), roughness=.86)
    _mesh(stage, path + "/Stem", stem, bark)
    return root


def orange_half_surface(dimensions=ORANGE_DIMENSIONS, *, seed=29, segments=96, rings=48):
    """Watertight lower hemisphere with a planar cut through the equator."""
    full = orange_surface(dimensions, seed=seed, segments=segments, rings=rings)
    count = 1 + (rings // 2) * segments
    points = full.points[:count]
    faces = [face for face in full.faces if max(face) < count]
    faces.append(tuple(range(count - segments, count)))
    return _surface(points, faces, full.colors[:count], full.roughness[:count])


def _pulp_segment(a, b, cut_z, index, seed, *, divisions=28, arc_divisions=20):
    """A thin closed juice segment; pith is visible between neighboring wedges."""
    angle0 = index * 2 * math.pi / 10 + .016 * math.sin(index * .6 * math.pi + seed) + .010
    angle1 = ((index + 1) * 2 * math.pi / 10
              + .016 * math.sin((index + 1) * .6 * math.pi + seed) - .010)
    points, colors, faces, roughness = [], [], [], []
    for upper in (False, True):
        for row in range(divisions + 1):
            radius = .065 + .860 * row / divisions
            for column in range(arc_divisions + 1):
                theta = angle0 + (angle1 - angle0) * column / arc_divisions
                # Slightly meandering membranes, with no repeating straight
                # spokes. The same boundary is used by both closed layers.
                theta += .014 * math.sin(row * .72 + index * 1.9 + seed)
                grain = _hash(row, column, index, seed)
                height = cut_z + (.000055 + .000060 * (.15 + .85 * grain * grain) if upper else 0)
                points.append((a * radius * math.cos(theta), b * radius * math.sin(theta), height))
                # Small pigment variation reads as juice vesicles. The
                # perimeter lightens into the surrounding pith membrane.
                edge = .025 if row in (0, divisions) or column in (0, arc_divisions) else 0.0
                glint = max(0.0, (grain - .65) / .35)
                colors.append((.91 + .075 * grain, .28 + .15 * grain + .16 * glint + edge,
                               .009 + .018 * grain + .10 * glint + edge))
                roughness.append(.34 + .16 * grain)
    width = arc_divisions + 1
    layer = (divisions + 1) * width
    for row in range(divisions):
        for column in range(arc_divisions):
            low = row * width + column
            face = (low, low + width, low + width + 1, low + 1)
            faces.append(tuple(reversed(face)))
            faces.append(tuple(vertex + layer for vertex in face))
    boundary = list(range(width))
    boundary += [row * width + arc_divisions for row in range(1, divisions + 1)]
    boundary += [divisions * width + column for column in range(arc_divisions - 1, -1, -1)]
    boundary += [row * width for row in range(divisions - 1, 0, -1)]
    # The boundary above runs clockwise in the XY plane for a radial grid.
    for low, high in zip(boundary, boundary[1:] + boundary[:1]):
        faces.append((low, low + layer, high + layer, high))
    return _surface(points, faces, colors, roughness)


def author_orange_half(stage, root_path, dimensions=ORANGE_DIMENSIONS, *, seed=29,
                       segments=96, rings=48):
    """Original cut citrus, with a pale rind and ten individually modeled segments."""
    dimensions = _dimensions(dimensions)
    surface = orange_half_surface(dimensions, seed=seed, segments=segments, rings=rings)
    root = _root(stage, root_path, "Original orange half with rind, pith and ten pulp segments")
    path = str(root.GetPath())
    peel = _material(stage, path + "/Looks/Peel", (.95, .32, .018), skin=True)
    _mesh(stage, path + "/Peel", surface, peel)
    a, b, height = dimensions[0] / 2, dimensions[1] / 2, dimensions[2]
    cut_z = -.027 * height / 2 + .000035
    # A closed thin pith disk covers the hemisphere's cap. The small offset
    # avoids coincident faces; all flesh remains inside the outer peel rim.
    disk = _lathe([(0, cut_z - .000025), (.992, cut_z - .000025),
                   (.992, cut_z), (0, cut_z)], segments=segments)
    disk = _surface([(x * a, y * b, z) for x, y, z in disk.points], disk.faces)
    pith = _material(stage, path + "/Looks/Pith", (.98, .91, .72), roughness=.62)
    _mesh(stage, path + "/Pith", disk, pith)
    pulp = _material(stage, path + "/Looks/Pulp", (.95, .29, .025), roughness=.42, skin=True)
    for index in range(10):
        _mesh(stage, path + f"/Segment_{index + 1:02d}",
              _pulp_segment(a, b, cut_z + .000010, index, seed), pulp)
    return root


def _rotate(point, angles):
    x, y, z = point
    ax, ay, az = (math.radians(value) for value in angles)
    y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
    x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
    x, y = x * math.cos(az) - y * math.sin(az), x * math.sin(az) + y * math.cos(az)
    return x, y, z


def _inner_height(radius):
    # The inside is the latter, monotonic part of the profile, in reverse.
    profile = tuple(reversed(_BOWL_PROFILE[17:]))
    for (r0, z0), (r1, z1) in zip(profile, profile[1:]):
        if r0 <= radius <= r1:
            return z0 + (z1 - z0) * (radius - r0) / (r1 - r0)
    raise ValueError("Decorative fruit lies outside the bowl interior")


def bowl_fruit_layout(*, diameter=.30):
    """Four whole oranges and two cut halves resting on the platter interior.

    Each placement is computed from its rotated mesh and the bowl profile.
    Nothing is dynamically simulated; the arrangement cannot settle or jitter.
    """
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Bowl diameter must be positive finite metres")
    scale = diameter / .30
    rows = []
    positions = ((-.043, .044), (.038, .047), (-.008, -.026),
                 (.075, -.027), (-.079, -.024), (.030, -.086))
    for index in range(6):
        seed = 81 + index * 11
        factor = 1.32 * (1.0, .973, 1.013, .985, 1.005, .99)[index]
        dimensions = tuple(value * factor for value in ORANGE_DIMENSIONS)
        angles = ((12, -19, 17), (-24, 10, 62), (17, 21, 130),
                  (8, -28, 193), (-7, -12, 12), (-10, 15, -10))[index]
        half = index >= 4
        make_surface = orange_half_surface if half else orange_surface
        points = tuple(_rotate(point, angles) for point in make_surface(
            dimensions, seed=seed, segments=64, rings=32).points)
        x, y = positions[index]
        z = max(_inner_height(math.hypot(px + x, py + y)) - pz
                for px, py, pz in points) + .00006
        rows.append({"position": tuple(value * scale for value in (x, y, z)),
                     "rotation": angles, "dimensions": tuple(value * scale for value in dimensions),
                     "seed": seed, "half": half})
    return tuple(rows)


def author_bowl_of_oranges(stage, root_path, *, position=(0.0, 0.0, 0.0), diameter=.30):
    """Author a stationary pale mustard platter and six original citrus pieces.

    Local z=0 is the foot's contact plane. The caller must place this visual
    assembly outside the arm's work area; it intentionally has no physics.
    """
    from pxr import Gf, Usd, UsdGeom

    position = tuple(float(value) for value in position)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("Bowl position must contain three finite metres")
    bowl = bowl_surface(diameter=diameter)
    layout = bowl_fruit_layout(diameter=diameter)
    root = _root(stage, root_path, "Original pale mustard platter with four whole oranges and two cut halves")
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    path = str(root.GetPath())
    ceramic = _material(stage, path + "/Looks/GlazedCeramic", (.90, .62, .22), roughness=.23)
    _mesh(stage, path + "/Bowl", bowl, ceramic)
    for index, item in enumerate(layout):
        author = author_orange_half if item["half"] else author_orange
        fruit = author(stage, path + f"/Orange_{index + 1:02d}", item["dimensions"],
                       seed=item["seed"], segments=256, rings=128)
        # Bake rotations for exact local extents as well as native Hydra
        # geometry. Rotating a hemisphere's box would otherwise report a
        # fictitious corner below the tabletop, outside the actual mesh.
        for prim in Usd.PrimRange(fruit.GetPrim()):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = [Gf.Vec3f(*_rotate(point, item["rotation"]))
                          for point in mesh.GetPointsAttr().Get()]
                mesh.GetPointsAttr().Set(points)
                mesh.GetNormalsAttr().Set([Gf.Vec3f(*_rotate(normal, item["rotation"]))
                                           for normal in mesh.GetNormalsAttr().Get()])
                mesh.GetExtentAttr().Set(UsdGeom.PointBased(mesh).ComputeExtent(points))
        fruit.AddTranslateOp().Set(Gf.Vec3d(*item["position"]))
    return root
