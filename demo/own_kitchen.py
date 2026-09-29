"""Original, texture-free CASCADE kitchen, authored before simulation starts.

The visual design and generated geometry/materials are dedicated to CC0-1.0.
All coordinates are metres in the robot base frame. Furniture is visual only:
the caller owns the independently calibrated countertop collision geometry.
"""
from __future__ import annotations

import math
import random

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


KITCHEN_ID = "paai-own-kitchen-v1"
DEFAULT_COUNTER = {
    "min": [-0.15987323186177282, -0.379085082641188, -0.03],
    "max": [0.9867574274915567, 0.3795904990656434, 0.0],
}


def _material(stage, path, color, *, roughness=.55, metallic=0, vertex=False):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if vertex:
        reader = UsdShade.Shader.Define(stage, path + "/Color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*color))
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mesh(stage, path, points, faces, material, colors=None, normals=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [Gf.Vec3f(*p) for p in points]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(points))
    if colors is not None:
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set([Gf.Vec3f(*c) for c in colors])
    if normals is not None:
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _pigment(p, color, wood=False):
    x, y, z = p
    coarse = math.sin(19*x + 13*y + 17*z) * math.sin(29*x - 23*y + 11*z)
    fine = math.sin(127*x + 193*y + 211*z) * math.sin(251*x - 137*y + 173*z)
    if wood:
        across = x + y
        grain = math.sin(97*across + 1.9*math.sin(9*z) + .8*math.sin(29*across))
        grain += .35*math.sin(271*across + 2.3*math.sin(11*z + .6))
        variation = 1 + .042*grain + .022*coarse + .009*fine
    else:
        variation = 1 + .025*coarse + .012*fine
    return tuple(max(0, min(1, c*variation)) for c in color)


def _box(stage, path, center, size, material, *, color=None, wood=False, samples=1):
    """Baked vertices avoid renderer-dependent scale transforms."""
    center, half = tuple(center), tuple(v/2 for v in size)
    points, faces, colors, normals = [], [], [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        nu = max(1, min(samples, math.ceil(size[u]/.018)))
        nv = max(1, min(samples, math.ceil(size[v]/.018)))
        for sign in (-1, 1):
            start = len(points)
            normal = [0., 0., 0.]
            normal[axis] = sign
            for j in range(nv + 1):
                for i in range(nu + 1):
                    p = list(center)
                    p[axis] += sign*half[axis]
                    p[u] += half[u]*(2*i/nu - 1)
                    p[v] += half[v]*(2*j/nv - 1)
                    points.append(p)
                    normals.append(normal)
                    if color is not None:
                        colors.append(_pigment(p, color, wood=wood))
            for j in range(nv):
                for i in range(nu):
                    a = start + j*(nu + 1) + i
                    face = [a, a + 1, a + nu + 2, a + nu + 1]
                    faces.append(face if sign > 0 else list(reversed(face)))
    return _mesh(stage, path, points, faces, material, colors if color is not None else None, normals)


def _lathe(stage, path, center, profile, material, *, color=None, segments=48):
    points, faces, colors = [], [], []
    for radius, z in profile:
        for j in range(segments):
            theta = 2*math.pi*j/segments
            p = (center[0] + radius*math.cos(theta), center[1] + radius*math.sin(theta), center[2] + z)
            points.append(p)
            if color is not None:
                colors.append(_pigment(p, color))
    for i in range(len(profile) - 1):
        for j in range(segments):
            a, b = i*segments + j, i*segments + (j + 1) % segments
            faces.append([a, b, b + segments, a + segments])
    return _mesh(stage, path, points, faces, material, colors if color is not None else None)


def _tube(stage, path, points, radius, material, segments=10):
    vertices, faces = [], []
    for i, point in enumerate(points):
        tangent = Gf.Vec3d(*(points[min(i + 1, len(points) - 1)][k] - points[max(i - 1, 0)][k]
                               for k in range(3))).GetNormalized()
        guide = Gf.Vec3d(0, 0, 1) if abs(tangent[2]) < .9 else Gf.Vec3d(0, 1, 0)
        u = Gf.Cross(tangent, guide).GetNormalized()
        v = Gf.Cross(tangent, u).GetNormalized()
        for j in range(segments):
            angle = j*2*math.pi/segments
            vertices.append(Gf.Vec3d(*point) + radius*(math.cos(angle)*u + math.sin(angle)*v))
    for i in range(len(points) - 1):
        for j in range(segments):
            a, b = i*segments + j, i*segments + (j + 1) % segments
            faces.append([a, b, b + segments, a + segments])
    faces.extend([list(reversed(range(segments))),
                  list(range((len(points) - 1)*segments, len(points)*segments))])
    return _mesh(stage, path, vertices, faces, material)


def _sphere(stage, path, center, radii, material, segments=20, rings=10):
    points, faces = [], []
    for i in range(rings + 1):
        phi = -math.pi/2 + math.pi*i/rings
        for j in range(segments):
            theta = 2*math.pi*j/segments
            points.append((center[0] + radii[0]*math.cos(phi)*math.cos(theta),
                           center[1] + radii[1]*math.cos(phi)*math.sin(theta),
                           center[2] + radii[2]*math.sin(phi)))
    for i in range(rings):
        for j in range(segments):
            a, b = i*segments + j, i*segments + (j + 1) % segments
            faces.append([a, b, b + segments, a + segments])
    return _mesh(stage, path, points, faces, material)


def _front_box(stage, path, axis, front, u, z, width, height, depth, material, **kwargs):
    center = (front, u, z) if axis == "x" else (u, front, z)
    size = (depth, width, height) if axis == "x" else (width, depth, height)
    return _box(stage, path, center, size, material, **kwargs)


def _door(stage, path, axis, front, u, z, width, height, material, color, knob,
          *, sign=-1, handle=True, handle_side=1):
    """Original inset joinery: broad frame, recessed panel, small bronze knob."""
    _front_box(stage, path + "/Panel", axis, front, u, z, width, height, .018,
               material, color=color, wood=True, samples=16)
    rail = min(.038, width*.10)
    for label, du, dz, w, h in (
        ("Left", -(width - rail)/2, 0, rail, height),
        ("Right", (width - rail)/2, 0, rail, height),
        ("Top", 0, (height - rail)/2, width - 2*rail, rail),
        ("Bottom", 0, -(height - rail)/2, width - 2*rail, rail),
    ):
        _front_box(stage, path + "/" + label, axis, front + sign*.012, u + du, z + dz,
                   w, h, .018, material, color=color, wood=True, samples=12)
    if handle:
        hu, hz = u + handle_side*(width/2 - .037), z + height/2 - .043
        pos = (front + sign*.034, hu, hz) if axis == "x" else (hu, front + sign*.034, hz)
        _sphere(stage, path + "/Knob", pos, (.010, .010, .010), knob)


def _stone(stage, path, lo, hi, material, grain, *, seed):
    """Poured pale terrazzo with independently generated angular mineral chips."""
    _box(stage, path + "/Slab", [(a + b)/2 for a, b in zip(lo, hi)],
         [b - a for a, b in zip(lo, hi)], material)
    rng = random.Random(seed)
    points, faces, colors = [], [], []
    palette = ((.16, .155, .16), (.26, .255, .245), (.39, .385, .37),
               (.53, .525, .505), (.71, .708, .675), (.84, .835, .81))
    # Also cover exposed vertical slab edges, preserving the exact outer bounds.
    surfaces = [(2, hi[2] + .000012, 0, 1, lo[0], hi[0], lo[1], hi[1], 1),
                (0, lo[0] - .000012, 1, 2, lo[1], hi[1], lo[2], hi[2], -1),
                (0, hi[0] + .000012, 1, 2, lo[1], hi[1], lo[2], hi[2], 1),
                (1, lo[1] - .000012, 2, 0, lo[2], hi[2], lo[0], hi[0], -1),
                (1, hi[1] + .000012, 2, 0, lo[2], hi[2], lo[0], hi[0], 1)]
    for axis, fixed, u, v, ulo, uhi, vlo, vhi, sign in surfaces:
        count = round((uhi - ulo)*(vhi - vlo)*22000)
        for _ in range(count):
            cu, cv = rng.uniform(ulo, uhi), rng.uniform(vlo, vhi)
            radius = .0014 + .0060*rng.random()**1.8
            stretch = rng.uniform(.45, 1.5)
            angle = rng.uniform(0, math.tau)
            n = rng.randrange(3, 7)
            start = len(points)
            color = rng.choice(palette)
            for j in range(n):
                theta = angle + j*math.tau/n
                r = radius*rng.uniform(.62, 1.05)
                p = [0., 0., 0.]
                p[axis] = fixed
                p[u] = min(uhi, max(ulo, cu + r*math.cos(theta)))
                p[v] = min(vhi, max(vlo, cv + r*stretch*math.sin(theta)))
                points.append(p)
                colors.append(tuple(c*rng.uniform(.95, 1.05) for c in color))
            face = list(range(start, start + n))
            faces.append(face if sign > 0 else list(reversed(face)))
    _mesh(stage, path + "/MineralChips", points, faces, grain, colors)


def _floor(stage, path, material, grout):
    z = -.8588
    _box(stage, path + "/Base", (.20, -.17, z - .035), (5.3, 5.0, .07), grout)
    points, faces, colors = [], [], []
    rng = random.Random(481)
    step = .48
    for ix in range(-5, 7):
        for iy in range(-6, 6):
            left, front = ix*step, iy*step
            tint = rng.uniform(.94, 1.04)
            start = len(points)
            for j in range(9):
                for i in range(9):
                    p = (left + .001 + (step - .002)*i/8,
                         front + .001 + (step - .002)*j/8, z + .0001)
                    points.append(p)
                    cloud = (.97 + .09*math.sin(23*p[0])*math.sin(19*p[1])
                             + .032*math.sin(79*p[0] + 53*p[1]))
                    colors.append(tuple(tint*cloud*c for c in _pigment(p, (.48, .319, .153))))
            for j in range(8):
                for i in range(8):
                    a = start + j*9 + i
                    faces.append([a, a + 1, a + 10, a + 9])
    _mesh(stage, path + "/Tiles", points, faces, material, colors)


def _backsplash(stage, path, axis, fixed, umin, umax, zmin, zmax, material, *, seed):
    """Small diagonal ceramic tiles with original mottled mineral pigment."""
    # Clip each rotated square to the backsplash boundary.
    def clip(poly, dimension, bound, positive):
        result = []
        for a, b in zip(poly, poly[1:] + poly[:1]):
            ain = a[dimension] >= bound if positive else a[dimension] <= bound
            bin = b[dimension] >= bound if positive else b[dimension] <= bound
            if ain:
                result.append(a)
            if ain != bin:
                t = (bound - a[dimension])/(b[dimension] - a[dimension])
                result.append(tuple(a[k] + t*(b[k] - a[k]) for k in range(2)))
        return result
    rng = random.Random(seed)
    points, faces, colors = [], [], []
    pitch = .105
    for i in range(-30, 45):
        for j in range(-30, 45):
            cu, cz = (i - j)*pitch, (i + j)*pitch
            if not (umin - pitch < cu < umax + pitch and zmin - pitch < cz < zmax + pitch):
                continue
            edge = pitch - .0007
            poly = [(cu - edge, cz), (cu, cz - edge), (cu + edge, cz), (cu, cz + edge)]
            for dim, bound, positive in ((0, umin, True), (0, umax, False),
                                         (1, zmin, True), (1, zmax, False)):
                poly = clip(poly, dim, bound, positive)
                if not poly:
                    break
            if len(poly) < 3:
                continue
            base = (.60, .365, .135) if (i + j) % 2 else (.80, .795, .735)
            tint = rng.uniform(.84, 1.07)
            center = (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))
            # Subdivided triangle fans give mottled pigment on planar tiles.
            for a, b in zip(poly, poly[1:] + poly[:1]):
                start = len(points)
                for row in range(5):
                    for col in range(row + 1):
                        u, z = (center[k]*(1 - row/4) + a[k]*(row - col)/4 + b[k]*col/4
                                for k in range(2))
                        p = (fixed, u, z) if axis == "x" else (u, fixed, z)
                        points.append(p)
                        cloud = .84 + .22*math.sin(u*73 + z*119)**2 + .12*math.sin(u*167 - z*91)**2
                        colors.append(tuple(min(1, c*tint*cloud) for c in base))
                for row in range(4):
                    for col in range(row + 1):
                        lower = start + row*(row + 1)//2 + col
                        upper = start + (row + 1)*(row + 2)//2 + col
                        triangles = [[lower, upper, upper + 1]]
                        if col < row:
                            triangles.append([lower, upper + 1, lower + 1])
                        # These two orientations face the open room.
                        faces.extend(list(reversed(f)) if axis == "x" else f for f in triangles)
    mesh = _mesh(stage, path, points, faces, material, colors)
    mesh.CreateDoubleSidedAttr(True)


def _appliances(stage, path, mats):
    black, steel, bronze = mats["Black"], mats["Steel"], mats["Bronze"]
    # Four-door dark refrigerator faces +X, with a slim satin-metal handle pair.
    _box(stage, path + "/Refrigerator/Body", (-1.79, .43, .167), (.98, 1.045, 2.052), black)
    for side, y in enumerate((.166, .694)):
        for level, (z, h) in enumerate(((.695, .98), (-.334, 1.05))):
            prefix = path + f"/Refrigerator/Door{side}{level}"
            _box(stage, prefix, (-1.284, y, z), (.038, .516, h), mats["Fridge"])
            hy = y + (.222 if side == 0 else -.222)
            _box(stage, prefix + "Handle", (-1.248, hy, z), (.038, .017, h*.83), mats["Handle"])
    _box(stage, path + "/Refrigerator/DispenserRecess", (-1.261, .150, .77), (.008, .19, .25), black)
    _box(stage, path + "/Refrigerator/DispenserShelf", (-1.234, .150, .651), (.065, .18, .012), steel)
    _box(stage, path + "/Refrigerator/DispenserButton", (-1.252, .15, .917), (.010, .075, .027), steel)
    for i, y in enumerate((.045, .255)):
        _box(stage, path + f"/Refrigerator/DispenserJamb{i}", (-1.238, y, .785),
             (.041, .014, .29), mats["Fridge"])
    _box(stage, path + "/Refrigerator/DispenserRoof", (-1.230, .15, .929),
         (.061, .224, .017), mats["Fridge"])
    _lathe(stage, path + "/Refrigerator/WaterSpout", (-1.213, .15, .839),
           [(0, 0), (.012, 0), (.012, .079), (0, .079)], steel, segments=20)
    _lathe(stage, path + "/Refrigerator/Cup", (-1.205, .15, .658),
           [(0, 0), (.032, 0), (.035, .012), (.035, .10), (.031, .10),
            (.029, .012), (0, .012)], steel, segments=32)
    _box(stage, path + "/Refrigerator/DispenserPlaque", (-1.227, .15, .969),
         (.009, .097, .043), black)
    for i in range(2):
        _box(stage, path + f"/Refrigerator/PlaqueMark{i}", (-1.221, .15, .974 - i*.011),
             (.003, .057 - i*.02, .002), steel)
    # Stacked oven and microwave in the left tall cabinet.
    for label, z, height in (("Oven", -.1216, .6012), ("Microwave", .3841, .3711)):
        prefix = path + "/" + label
        _box(stage, prefix + "/Case", (-1.54, 1.288, z), (.29, .610, height), steel)
        _box(stage, prefix + "/Glass", (-1.386, 1.288, z - .02), (.020, .554, height*.70), black)
        _box(stage, prefix + "/Control", (-1.381, 1.288, z + height*.38), (.025, .552, height*.15), black)
        _box(stage, prefix + "/Handle", (-1.344, 1.288, z + height*.21), (.032, .445, .025), steel)
        for index, yy in enumerate((1.053, 1.516)):
            _sphere(stage, prefix + f"/Dial{index}", (-1.355, yy, z + height*.38), (.013, .018, .018), steel)
        for index in range(4):
            _box(stage, prefix + f"/Indicator{index}", (-1.363, 1.23 + index*.026, z + height*.38),
                 (.005, .011, .006), mats["Display"])
    # Range with four original burner/grate assemblies.
    _box(stage, path + "/Stove/Top", (.76, 1.936, -.022), (.838, .553, .018), black)
    for index, (x, y) in enumerate(((.51, 1.82), (.98, 1.82), (.51, 2.07), (.98, 2.07))):
        _lathe(stage, path + f"/Stove/Burner{index}", (x, y, -.01),
               [(0, 0), (.057, 0), (.057, .01), (0, .01)], steel, segments=24)
        for k in range(2):
            dims = (.21, .011, .014) if k == 0 else (.011, .19, .014)
            _box(stage, path + f"/Stove/Grate{index}_{k}", (x, y, .008), dims, black)
    for index in range(4):
        _sphere(stage, path + f"/Stove/Dial{index}", (.61 + index*.10, 1.676, -.005), (.016, .015, .012), black)
    _box(stage, path + "/Hood/Case", (.737, 2.05, .742), (.895, .38, .51), black)
    _box(stage, path + "/Hood/Glass", (.737, 1.851, .685), (.816, .016, .287), mats["Fridge"])
    _box(stage, path + "/Hood/ControlBar", (.737, 1.839, .939), (.895, .027, .117), black)
    _box(stage, path + "/Hood/LowerTrim", (.737, 1.847, .490), (.895, .041, .028), black)
    for i in range(15):
        _box(stage, path + f"/Hood/Vent{i}", (.737, 1.837, .559 + i*.017), (.769, .008, .005), black)
    digit_segments = ((1, 2), (0, 1, 6, 4, 3), (0, 1, 6, 2, 3), tuple(range(7)))
    segment_layout = ((0, .015, .013, .0025), (.007, .008, .0025, .012),
                      (.007, -.008, .0025, .012), (0, -.015, .013, .0025),
                      (-.007, -.008, .0025, .012), (-.007, .008, .0025, .012),
                      (0, 0, .013, .0025))
    for i, segments in enumerate(digit_segments):
        x = .966 + i*.023 + (.010 if i > 1 else 0)
        for segment in segments:
            dx, dz, w, h = segment_layout[segment]
            _box(stage, path + f"/Hood/Display{i}_{segment}", (x + dx, 1.819, .950 + dz),
                 (w, .005, h), mats["Display"])
    for i, z in enumerate((.943, .957)):
        _box(stage, path + f"/Hood/ClockColon{i}", (1.007, 1.819, z), (.0025, .005, .0025), mats["Display"])
    # Worktop appliances and cookware all stay on the distant counter.
    _box(stage, path + "/Coffee/Body", (-.83, 1.98, .146), (.275, .25, .348), black)
    _box(stage, path + "/Coffee/Face", (-.83, 1.846, .164), (.247, .018, .263), steel)
    _box(stage, path + "/Coffee/Inset", (-.83, 1.833, .073), (.194, .020, .153), black)
    _box(stage, path + "/Coffee/DripTray", (-.83, 1.793, -.007), (.271, .16, .028), steel)
    for i, x in enumerate((-.91, -.75)):
        _sphere(stage, path + f"/Coffee/Dial{i}", (x, 1.827, .243), (.022, .010, .022), black)
    _tube(stage, path + "/Coffee/SteamWand", [(-.736, 1.82, .162), (-.721, 1.81, .087), (-.727, 1.77, .036)], .007, steel)
    _box(stage, path + "/Toaster/Body", (-.53, 1.97, .080), (.184, .27, .216), steel)
    _box(stage, path + "/Toaster/Foot", (-.53, 1.97, -.018), (.17, .252, .019), black)
    for i, y in enumerate((1.918, 2.020)):
        _box(stage, path + f"/Toaster/Slot{i}", (-.53, y, .189), (.132, .023, .002), black)
    _box(stage, path + "/Toaster/Lever", (-.53, 1.826, .096), (.061, .024, .021), black)
    _lathe(stage, path + "/Pot/Body", (.512, 1.857, -.012),
           [(0, 0), (.106, 0), (.115, .015), (.121, .115), (.117, .127), (.105, .127), (.103, .017), (0, .017)], steel)
    _lathe(stage, path + "/Pot/Lid", (.512, 1.857, .110),
           [(0, 0), (.124, 0), (.124, .003), (.09, .018), (.030, .029), (0, .030)], steel)
    _sphere(stage, path + "/Pot/Knob", (.512, 1.857, .158), (.021, .018, .016), black)
    for side in (-1, 1):
        _tube(stage, path + f"/Pot/Handle{'Left' if side < 0 else 'Right'}",
              [(.512 + side*.111, 1.81, .092), (.512 + side*.164, 1.81, .092),
               (.512 + side*.164, 1.90, .092), (.512 + side*.111, 1.90, .092)], .008, black)
    # Recessed-looking stainless basin and simple bent-tube faucet.
    _box(stage, path + "/Sink/Rim", (2.49, 1.105, -.024), (.462, .68, .008), steel)
    _box(stage, path + "/Sink/Basin", (2.49, 1.105, -.019), (.398, .598, .006), mats["Sink"])
    _sphere(stage, path + "/Sink/Drain", (2.49, 1.09, -.013), (.024, .024, .001), black)
    _tube(stage, path + "/Sink/Faucet", [(2.60, 1.43, -.019), (2.60, 1.43, .17),
           (2.60, 1.42, .225), (2.60, 1.39, .255), (2.60, 1.34, .26),
           (2.60, 1.29, .239), (2.60, 1.27, .199)], .011, steel)


def _distant_details(stage, path, mats):
    # Original unlabelled bottles, jars and hanging utensils add familiar scale.
    for i, (x, y, h) in enumerate(((-.36, 2.025, .26), (-.25, 2.015, .265),
                                    (-.14, 2.025, .27), (1.83, 2.08, .18), (1.92, 2.08, .19))):
        material = mats["BottleAmber"] if i % 2 else mats["BottleOlive"]
        _lathe(stage, path + f"/Bottle{i}", (x, y, -.025),
               [(0, 0), (.026, 0), (.029, .02), (.027, h*.67), (.013, h*.80), (.011, h), (0, h)], material)
        _lathe(stage, path + f"/BottleLabel{i}", (x, y, -.025),
               [(.0291, h*.17), (.0291, h*.43)], mats["Ivory"])
        _lathe(stage, path + f"/BottleCap{i}", (x, y, -.025),
               [(.012, h*.93), (.012, h + .006), (0, h + .006)], mats["Black"])
    _tube(stage, path + "/Utensils/Rail", [(-.57, 2.178, .53), (.15, 2.178, .53)], .007, mats["Black"])
    for i, x in enumerate((-.50, -.32, -.14, .04)):
        _tube(stage, path + f"/Utensils/Stem{i}", [(x, 2.163, .53), (x, 2.155, .29)], .007, mats["Black"])
        if i % 2:
            _box(stage, path + f"/Utensils/Head{i}", (x, 2.151, .245), (.071, .013, .098), mats["Black"])
            for k in range(3):
                _box(stage, path + f"/Utensils/Slit{i}_{k}", (x - .020 + k*.020, 2.143, .244),
                     (.006, .003, .064), mats["Steel"])
        else:
            _sphere(stage, path + f"/Utensils/Head{i}", (x, 2.151, .235), (.039, .012, .05), mats["Black"])
    for i, x in enumerate((1.40, 1.53)):
        _box(stage, path + f"/Canister{i}/Body", (x, 2.015, .093), (.104, .105, .228), mats["Ivory"])
        _box(stage, path + f"/Canister{i}/Lid", (x, 2.015, .215), (.109, .11, .025), mats["Steel"])


def author_kitchen(stage, path="/Kitchen", *, counter=None):
    """Create the original visual room and platter; never author physics.

    ``counter`` has the same min/max convention as kitchen_config.json.
    Geometry is deterministic, contains no asset references and uses no maps.
    """
    counter = DEFAULT_COUNTER if counter is None else counter
    lo, hi = tuple(counter["min"]), tuple(counter["max"])
    if len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in lo + hi):
        raise ValueError("Counter bounds must be finite metre coordinates")
    if not all(a < b for a, b in zip(lo, hi)) or abs(hi[2]) > 1e-9:
        raise ValueError("The calibrated kitchen countertop must have its top at z=0")
    if stage.GetPrimAtPath(path):
        raise ValueError("Original kitchen root already exists")
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    root.SetCustomDataByKey("design", KITCHEN_ID)
    root.SetCustomDataByKey("assetLicense", "CC0-1.0")
    root.SetCustomDataByKey("construction", "Original procedural meshes, pigments and analytic lighting")
    palette = {
        "Ivory": ((.73, .709, .645), .58, 0, True),
        "Sage": ((.30, .335, .17), .59, 0, True),
        "Wall": ((.67, .663, .628), .80, 0, False),
        "Stone": ((.77, .776, .753), .47, 0, False),
        "Mineral": ((.42, .41, .40), .48, 0, True),
        "Floor": ((.49, .335, .167), .67, 0, True),
        "Grout": ((.48, .405, .29), .84, 0, False),
        "Ceramic": ((.8, .78, .7), .58, 0, True),
        "Black": ((.015, .018, .017), .36, .15, False),
        "Fridge": ((.025, .030, .023), .26, .32, False),
        "Steel": ((.49, .51, .515), .26, .86, False),
        "Bronze": ((.21, .115, .063), .31, .60, False),
        "Handle": ((.38, .35, .23), .26, .80, False),
        "Sink": ((.20, .23, .235), .24, .83, False),
        "Display": ((.055, .30, .37), .25, .1, False),
        "BottleAmber": ((.23, .13, .025), .21, .15, False),
        "BottleOlive": ((.12, .145, .04), .21, .1, False),
    }
    mats = {name: _material(stage, path + "/Looks/" + name, color, roughness=rough,
                            metallic=metal, vertex=vertex)
            for name, (color, rough, metal, vertex) in palette.items()}
    _floor(stage, path + "/Room/Floor", mats["Floor"], mats["Grout"])
    _box(stage, path + "/Room/BackWall", (.38, 2.305, .49), (5.17, .10, 2.70), mats["Wall"])
    _box(stage, path + "/Room/LeftWall", (-2.045, -.127, .49), (.10, 3.494, 2.70), mats["Wall"])
    _box(stage, path + "/Room/RightWall", (2.96, .254, .49), (.10, 4.26, 2.70), mats["Wall"])
    _box(stage, path + "/Room/Ceiling", (.38, .254, 1.84), (5.17, 4.26, .055), mats["Ivory"])
    _backsplash(stage, path + "/Room/BackTiles", "y", 2.252, -1.12, 2.78, -.85, 1.63, mats["Ceramic"], seed=42)
    _backsplash(stage, path + "/Room/CornerTiles", "x", -1.112, 1.61, 2.25, -.85, 1.64, mats["Ceramic"], seed=43)
    _backsplash(stage, path + "/Room/LeftTiles", "x", -1.994, -1.874, 1.62, -.858, 1.838, mats["Ceramic"], seed=44)
    ivory, sage = palette["Ivory"][0], palette["Sage"][0]
    # Island: creamy near/end panels, sage framed doors on the far side.
    cx, cy = (lo[0] + hi[0])/2, (lo[1] + hi[1])/2
    width, depth = hi[0] - lo[0], hi[1] - lo[1]
    _box(stage, path + "/Island/Plinth", (cx, cy, -.813), (width - .105, depth - .12, .085), mats["Black"])
    _box(stage, path + "/Island/Body", (cx, cy, -.43), (width - .055, depth - .057, .775), mats["Ivory"], color=ivory, wood=True, samples=12)
    for i in range(3):
        u = lo[0] + .032 + (width - .064)*(i + .5)/3
        _door(stage, path + f"/Island/SageDoor{i}", "y", hi[1] - .044, u, -.407,
              (width - .072)/3 - .004, .685, mats["Sage"], sage, mats["Bronze"], sign=1,
              handle_side=1 if i < 2 else -1)
        _door(stage, path + f"/Island/IvoryPanel{i}", "y", lo[1] + .030, u, -.41,
              (width - .072)/3 - .004, .73, mats["Ivory"], ivory, mats["Bronze"], handle=False)
    _stone(stage, path + "/Island/Countertop", lo, hi, mats["Stone"], mats["Mineral"], seed=271828)
    # Two base runs form an L; their top remains outside the arm workspace.
    _box(stage, path + "/Base/BackBody", (.834, 1.952, -.452), (3.868, .588, .796), mats["Sage"], color=sage, wood=True, samples=12)
    _box(stage, path + "/Base/RightBody", (2.477, .63, -.452), (.58, 2.01, .796), mats["Sage"], color=sage, wood=True, samples=12)
    _stone(stage, path + "/Base/BackCounter", (-1.1, 1.636, -.063), (2.768, 2.235, -.025), mats["Stone"], mats["Mineral"], seed=161803)
    _stone(stage, path + "/Base/RightCounter", (2.167, -.374, -.063), (2.768, 1.636, -.025), mats["Stone"], mats["Mineral"], seed=314159)
    for i, (u, w) in enumerate(((-.839, .472), (-.37, .462), (.092, .46), (1.484, .469), (1.955, .468))):
        _door(stage, path + f"/Base/BackDoor{i}", "y", 1.665, u, -.4445, w - .004, .68,
              mats["Sage"], sage, mats["Bronze"], handle_side=(-1 if i % 2 else 1))
    for i, (z, h) in enumerate(((-.671, .223), (-.442, .217), (-.247, .165))):
        _door(stage, path + f"/Base/Drawer{i}", "y", 1.665, .787, z, .908, h,
              mats["Sage"], sage, mats["Bronze"], handle_side=0)
    for i, (u, w) in enumerate(((1.43, .493), (.923, .516), (-.169, .408))):
        _door(stage, path + f"/Base/RightDoor{i}", "x", 2.197, u, -.4445, w - .004, .68,
              mats["Sage"], sage, mats["Bronze"], sign=-1)
    _box(stage, path + "/Base/Dishwasher", (2.187, .350, -.447), (.031, .596, .683), mats["Steel"])
    _box(stage, path + "/Base/DishwasherHandle", (2.153, .350, -.161), (.022, .45, .023), mats["Black"])
    # Cream tall pantry and oven housing on the left wall.
    _box(stage, path + "/Tall/Pantry", (-1.692, -.411, .337), (.602, .623, 2.397), mats["Ivory"], color=ivory, wood=True, samples=12)
    for i, (z, h) in enumerate(((-.583, .540), (-.040, .542), (.504, .541), (1.1235, .695))):
        _door(stage, path + f"/Tall/PantryDoor{i}", "x", -1.393, -.411, z, .5875, h - .004,
              mats["Ivory"], ivory, mats["Bronze"], sign=1)
    _box(stage, path + "/Tall/OvenHousing", (-1.692, 1.287, .337), (.602, .658, 2.397), mats["Ivory"], color=ivory, wood=True, samples=12)
    _door(stage, path + "/Tall/UpperOvenDoor", "x", -1.393, 1.287, 1.035, .616, .87,
          mats["Ivory"], ivory, mats["Bronze"], sign=1, handle=False)
    _door(stage, path + "/Tall/LowerOvenDoor", "x", -1.393, 1.287, -.645, .616, .42,
          mats["Ivory"], ivory, mats["Bronze"], sign=1, handle=False)
    # Upper units and simple opaque glazing keep camera rendering predictable.
    _box(stage, path + "/Upper/Body", (.8314, 2.053, 1.0834), (3.8514, .3781, .8744), mats["Ivory"], color=ivory, wood=True, samples=12)
    for i, (u, w) in enumerate(((-.868, .451), (-.409, .463), (.056, .463))):
        _door(stage, path + f"/Upper/Door{i}", "y", 1.859, u, 1.083, w - .003, .874,
              mats["Ivory"], ivory, mats["Bronze"], handle_side=(-1 if i % 2 else 1))
    for i, u in enumerate((1.3975, 1.7857, 2.174, 2.5623)):
        _door(stage, path + f"/Upper/GlazedFrame{i}", "y", 1.859, u, 1.19, .385, .66,
              mats["Ivory"], ivory, mats["Bronze"], handle=False)
        _box(stage, path + f"/Upper/Glass{i}", (u, 1.843, 1.185), (.299, .003, .574), mats["Sink"])
        for j, zz in enumerate((1.03, 1.30)):
            _box(stage, path + f"/Upper/GlassShelf{i}_{j}", (u, 1.840, zz), (.29, .005, .008), mats["Ivory"])
    _appliances(stage, path + "/Appliances", mats)
    _distant_details(stage, path + "/BackCounterDetails", mats)
    # Original analytic daylight/ceiling illumination; no environment map.
    dome = UsdLux.DomeLight.Define(stage, path + "/Lighting/Ambient")
    dome.CreateIntensityAttr(650)
    dome.CreateColorAttr(Gf.Vec3f(1, .985, .955))
    ceiling = UsdLux.RectLight.Define(stage, path + "/Lighting/CeilingSoftbox")
    ceiling.CreateWidthAttr(3.6)
    ceiling.CreateHeightAttr(4.3)
    ceiling.CreateNormalizeAttr(True)
    ceiling.CreateIntensityAttr(22500)
    ceiling.CreateColorAttr(Gf.Vec3f(1, .978, .94))
    UsdGeom.Xformable(ceiling).AddTranslateOp().Set(Gf.Vec3d(.72, -.04, 1.705))
    for i, (x, y, w, h) in enumerate(((.83, 1.80, 3.70, .028), (.83, -1.70, 3.70, .028),
                                       (-1.0, .17, .028, 3.5), (2.6, .17, .028, 3.5))):
        light = UsdLux.RectLight.Define(stage, path + f"/Lighting/Cove{i}")
        light.CreateWidthAttr(w)
        light.CreateHeightAttr(h)
        light.CreateIntensityAttr(1050)
        light.CreateColorAttr(Gf.Vec3f(1, .96, .88))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, y, 1.55))
    from own_kitchen_props import author_bowl_of_oranges
    author_bowl_of_oranges(stage, path + "/DecorativeBowl", position=(.7514, .0106, 0), diameter=.3023)
    return root
