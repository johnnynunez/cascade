# SPDX-License-Identifier: Apache-2.0
"""Generate the agentic MicroDuck demo scenes (metre, Z-up USDA courses for Isaac Lab + Newton and OVRTX).

Every solid is a convex-hull mesh collider with a displayColor (the Newton Warp head camera shades displayColor; the
OVRTX render binds a preview material built from the same color). Floor decals carry no collider. The root prim is
``/Course`` so Isaac Lab spawns it per environment under ``{ENV_REGEX_NS}/Course``.

Run with any Python that has ``pxr`` (usd-core), e.g. the Isaac Lab venv:
    python scripts/microduck_agentic/build_scenes.py --output demo/microduck_agentic
"""
import argparse
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

GREY = (0.55, 0.56, 0.58)
WALL = (0.80, 0.78, 0.74)
CASE = (0.12, 0.12, 0.13)
RED = (0.85, 0.12, 0.10)
BLUE = (0.10, 0.30, 0.85)
YELLOW = (0.95, 0.75, 0.10)
GREEN = (0.10, 0.75, 0.25)
WOOD = (0.55, 0.38, 0.22)
STAGE_TOP = (0.20, 0.20, 0.24)


def _stage(path):
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, "Z")
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Course")
    st.SetDefaultPrim(root.GetPrim())
    return st


def _mesh(st, name, points, counts, indices, color, collide=True, purpose=None):
    m = UsdGeom.Mesh.Define(st, f"/Course/{name}")
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(indices)
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    m.GetPrim().SetCustomDataByKey("cascade:color", Gf.Vec3f(*color))
    return m


def box(st, name, center, size, color, yaw_deg=0.0, collide=True):
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    pts = []
    for z in (-sz, sz):
        for x, y in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)):
            pts.append((cx + c * x - s * y, cy + s * x + c * y, cz + z))
    idx = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    return _mesh(st, name, pts, [4] * 6, idx, color, collide)


def cylinder(st, name, center, radius, height, color, segments=20, collide=True, top_radius=None):
    cx, cy, z0 = center
    pts = []
    for z, radius in ((z0, radius), (z0 + height, radius if top_radius is None else top_radius)):
        for k in range(segments):
            a = 2 * math.pi * k / segments
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z))
    counts, idx = [segments, segments], list(range(segments - 1, -1, -1)) + list(range(segments, 2 * segments))
    for k in range(segments):
        a, b = k, (k + 1) % segments
        counts.append(4)
        idx += [a, b, segments + b, segments + a]
    return _mesh(st, name, pts, counts, idx, color, collide)


TAPE = [(0.08, 0.08, 0.09), (0.93, 0.93, 0.90), (0.95, 0.80, 0.10), (0.15, 0.35, 0.80), (0.80, 0.25, 0.20),
        (0.25, 0.55, 0.30), (0.45, 0.30, 0.20)]


def dress_floor(st, x0, y0, x1, y1, n, seed, prefix="DressFloor", z0=0.0):
    """Set dressing (no colliders): gaffer-tape marks, crosses and cable runs on the floor, like a real backstage or
    street. It gives camera-based localization the texture any real floor has; physics is unchanged."""
    import random

    rng = random.Random(seed)
    for k in range(n):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        col = rng.choice(TAPE)
        yaw = rng.uniform(0, 180)
        kind = rng.random()
        z = z0 + 0.0005 + 0.0001 * (k % 7)
        if kind < 0.45:  # tape strip
            box(st, f"{prefix}{k:03d}", (x, y, z), (rng.uniform(0.12, 0.45), rng.uniform(0.025, 0.05), 0.001), col,
                yaw_deg=yaw, collide=False)
        elif kind < 0.75:  # spike cross
            L = rng.uniform(0.08, 0.16)
            box(st, f"{prefix}{k:03d}a", (x, y, z), (L, 0.025, 0.001), col, yaw_deg=yaw, collide=False)
            box(st, f"{prefix}{k:03d}b", (x, y, z), (L, 0.025, 0.001), col, yaw_deg=yaw + 90, collide=False)
        else:  # scuff / mat patch
            box(st, f"{prefix}{k:03d}", (x, y, z), (rng.uniform(0.1, 0.35), rng.uniform(0.1, 0.3), 0.001),
                tuple(min(1.0, c * rng.uniform(0.6, 1.0)) for c in col), yaw_deg=yaw, collide=False)


def walls(st, prefix, x0, y0, x1, y1, height=0.25, t=0.05, color=WALL, gaps=(), posters=True):
    """Rectangle of walls; gaps: list of (side, lo, hi) with side in {'x0','x1','y0','y1'}."""
    def run(side, a0, a1, fixed, along_x):
        cuts = sorted((lo, hi) for s, lo, hi in gaps if s == side)
        segs, cur = [], a0
        for lo, hi in cuts:
            segs.append((cur, lo))
            cur = hi
        segs.append((cur, a1))
        for k, (u0, u1) in enumerate(segs):
            if u1 - u0 < 1e-3:
                continue
            if along_x:
                box(st, f"{prefix}_{side}_{k}", ((u0 + u1) / 2, fixed, height / 2), (u1 - u0, t, height), color)
            else:
                box(st, f"{prefix}_{side}_{k}", (fixed, (u0 + u1) / 2, height / 2), (t, u1 - u0, height), color)
            if posters:  # flyers / signs on both faces (no collider)
                npost = max(1, int((u1 - u0) / 0.45))
                for j in range(npost):
                    u = u0 + (j + 0.5) * (u1 - u0) / npost
                    col = TAPE[(len(prefix) + ord(side[1]) + 3 * k + 5 * j) % len(TAPE)]
                    w, hgt = 0.10 + 0.04 * ((j + k) % 3), 0.07 + 0.03 * (j % 2)
                    zc = 0.06 + 0.10 * ((j * 7 + k) % 3) / 2
                    for sgn in (-1, 1):
                        if along_x:
                            box(st, f"DressPoster_{prefix}_{side}_{k}_{j}_{'ab'[sgn > 0]}",
                                (u, fixed + sgn * (t / 2 + 0.002), zc), (w, 0.002, hgt), col, collide=False)
                        else:
                            box(st, f"DressPoster_{prefix}_{side}_{k}_{j}_{'ab'[sgn > 0]}",
                                (fixed + sgn * (t / 2 + 0.002), u, zc), (0.002, w, hgt), col, collide=False)
    run("y0", x0, x1, y0, True)
    run("y1", x0, x1, y1, True)
    run("x0", y0, y1, x0, False)
    run("x1", y0, y1, x1, False)


def decal(st, name, center, size, color):
    return box(st, name, (center[0], center[1], 0.0015), (size[0], size[1], 0.003), color, collide=False)


def backstage(path):
    st = _stage(path)
    walls(st, "Wall", -0.5, -1.0, 3.1, 1.0)
    box(st, "FlightCase", (0.95, 0.05, 0.15), (0.45, 0.35, 0.30), CASE)
    cylinder(st, "CableReel", (1.65, -0.45, 0.0), 0.16, 0.16, YELLOW)
    box(st, "SpeakerBox", (1.95, 0.45, 0.13), (0.30, 0.30, 0.26), WOOD, yaw_deg=20)
    box(st, "LowCase", (2.2, -0.2, 0.06), (0.25, 0.20, 0.12), CASE)
    decal(st, "GoalPad", (2.6, 0.2), (0.30, 0.30), GREEN)
    cylinder(st, "GoalMarker", (2.85, 0.2, 0.0), 0.05, 0.40, GREEN)
    st.GetRootLayer().customLayerData = {"cascade:scenario": "backstage_obstacles", "goal_xy": "2.6 0.2"}
    dress_floor(st, -1.6, -2.2, 4.2, 2.2, 110, seed=11)
    st.Save()


def stage_edge(path):
    st = _stage(path)
    top = 0.30
    box(st, "Stage", (0.9, 0.0, top / 2), (2.4, 1.6, top), STAGE_TOP)
    box(st, "StageLip", (2.09, 0.0, top + 0.004), (0.02, 1.6, 0.008), YELLOW, collide=False)
    box(st, "Monitor", (0.3, 0.55, top + 0.08), (0.30, 0.18, 0.16), CASE)
    cylinder(st, "MicStand", (1.3, -0.45, top), 0.04, 0.45, GREY)
    for k, y in enumerate((-0.6, -0.2, 0.2, 0.6)):
        box(st, f"AudienceSeat{k}", (2.9, y, 0.2), (0.35, 0.3, 0.4), RED)
    st.GetRootLayer().customLayerData = {"cascade:scenario": "stage_edge", "stage_top_z": str(top)}
    dress_floor(st, -0.25, -0.75, 2.05, 0.75, 28, seed=21, prefix="DressStage", z0=top)
    dress_floor(st, -1.5, -2.2, 4.0, 2.2, 80, seed=22)
    st.Save()


def multi_room(path):
    st = _stage(path)
    walls(st, "Outer", -0.5, -1.25, 4.5, 1.25)
    # inner wall at x = 2.0 with a 0.5 m doorway between y = 0.25 and 0.75
    box(st, "Inner_0", (2.0, -0.5, 0.125), (0.05, 1.5, 0.25), WALL)
    box(st, "Inner_1", (2.0, 1.0, 0.125), (0.05, 0.5, 0.25), WALL)
    box(st, "RedBox", (3.8, -0.75, 0.1), (0.2, 0.2, 0.2), RED)
    box(st, "BlueBox", (1.45, 0.85, 0.1), (0.2, 0.2, 0.2), BLUE)
    cylinder(st, "YellowReel", (3.0, 0.7, 0.0), 0.15, 0.15, YELLOW)
    box(st, "Crate", (0.9, -0.1, 0.11), (0.3, 0.3, 0.22), WOOD)
    box(st, "CaseB", (3.4, 0.1, 0.13), (0.4, 0.3, 0.26), CASE, yaw_deg=-15)
    decal(st, "RoomA", (0.75, 0.0), (0.4, 0.4), (0.82, 0.86, 0.92))
    st.GetRootLayer().customLayerData = {"cascade:scenario": "multi_room_fetch",
                                         "objects": "RedBox 3.8 -0.75; BlueBox 1.45 0.85; YellowReel 3.0 0.7"}
    dress_floor(st, -1.5, -2.3, 5.5, 2.3, 140, seed=31)
    st.Save()


ORANGE = (0.98, 0.42, 0.05)
CONE_ORANGE = (0.85, 0.18, 0.02)  # traffic-cone orange (the brighter ORANGE renders yellow under the plaza light)
PAVING = (0.62, 0.60, 0.57)
WHITE = (0.92, 0.92, 0.90)


SKATE_CONES = {"ConeA": (1.6, 0.5), "ConeB": (3.4, -0.5), "ConeC": (5.2, 0.4)}
SKATE_STOP = (6.6, 0.0)


def skate_plaza(path):
    """Skate plaza: three cones to pass in order; the straight lines A->B and B->C run through a bench and a planter;
    a yellow stop mark at the end."""
    st = _stage(path)
    decal(st, "Paving", (3.3, 0.0), (8.6, 3.4), PAVING)
    for name, (x, y) in SKATE_CONES.items():  # traffic cones: orange taper with a white band on a black base plate
        box(st, name + "Base", (x, y, 0.008), (0.20, 0.20, 0.016), (0.08, 0.08, 0.09))
        cylinder(st, name, (x, y, 0.016), 0.075, 0.105, CONE_ORANGE, top_radius=0.045)
        cylinder(st, name + "Band", (x, y, 0.121), 0.046, 0.04, WHITE, top_radius=0.034, collide=False)
        cylinder(st, name + "Tip", (x, y, 0.161), 0.035, 0.07, CONE_ORANGE, top_radius=0.012)
    box(st, "Bench", (2.5, 0.0, 0.11), (0.70, 0.25, 0.22), WOOD)
    cylinder(st, "Planter", (4.3, -0.05, 0.0), 0.20, 0.28, GREEN)
    decal(st, "StopMark", SKATE_STOP, (0.30, 0.30), YELLOW)
    st.GetRootLayer().customLayerData = {"cascade:scenario": "skate_plaza", "objects": "; ".join(
        f"{k} {x} {y}" for k, (x, y) in SKATE_CONES.items()) + f"; StopMark {SKATE_STOP[0]} {SKATE_STOP[1]}"}
    dress_floor(st, -1.0, -2.6, 7.8, 2.6, 120, seed=41, z0=0.003)
    st.Save()


# Skate oval (stadium): straights x in [-OVAL_HALF, OVAL_HALF] at y = +-r, semicircles around (+-OVAL_HALF, 0).
OVAL_HALF = 1.5  # 3.0 m straights and 0.3 m lanes from r 0.6 m: the oval race policy's training geometry
OVAL_LANES = (0.6, 0.9, 1.2)  # lane centre radii = lanes 1-3 of the trained 5-lane oval (centre lines r 0.6-1.8)
OVAL_LANE_WIDTH = 0.3


def skate_oval(path):
    st = _stage(path)
    rmax = OVAL_LANES[-1] + OVAL_LANE_WIDTH / 2
    rmin = OVAL_LANES[0] - OVAL_LANE_WIDTH / 2
    # rink surface + lane lines (decals, no collider)
    decal(st, "Rink", (0.0, 0.0), (2 * OVAL_HALF + 2 * rmax + 0.4, 2 * rmax + 0.4), (0.70, 0.55, 0.38))
    for k, r in enumerate([OVAL_LANES[0] - OVAL_LANE_WIDTH / 2] + [x + OVAL_LANE_WIDTH / 2 for x in OVAL_LANES]):
        for side, y in (("S", -r), ("N", r)):
            box(st, f"LaneLine{k}{side}", (0.0, y, 0.0035), (2 * OVAL_HALF, 0.02, 0.003), WHITE, collide=False)
        for end, cx in (("E", OVAL_HALF), ("W", -OVAL_HALF)):
            for j in range(14):
                a = -math.pi / 2 + math.pi * (j + 0.5) / 14
                a = a if end == "E" else math.pi - a  # west end mirrors the east end
                box(st, f"LaneLine{k}{end}{j}", (cx + r * math.cos(a), r * math.sin(a), 0.0035),
                    (math.pi * r / 14, 0.02, 0.003), WHITE, collide=False, yaw_deg=math.degrees(a) + 90)
    box(st, "FinishLine", (0.0, -(rmin + rmax) / 2, 0.004), (0.04, rmax - rmin, 0.004), (0.1, 0.1, 0.1), collide=False)
    # infield: painted (decal, no collider; the race policy was trained on a flat rink) + outer boards (box segments)
    box(st, "Infield", (0.0, 0.0, 0.0025), (2 * OVAL_HALF, 2 * rmin - 0.04, 0.003), GREEN, collide=False)
    for end, cx in (("E", OVAL_HALF), ("W", -OVAL_HALF)):
        cylinder(st, f"InfieldEnd{end}", (cx, 0.0, 0.001), rmin - 0.02, 0.003, GREEN, segments=24, collide=False)
    bh, bt = 0.15, 0.06
    rb = rmax + 0.30
    for side, y in (("S", -rb - bt / 2), ("N", rb + bt / 2)):
        box(st, f"Board{side}", (0.0, y, bh / 2), (2 * OVAL_HALF, bt, bh), BLUE)
    for end, cx in (("E", OVAL_HALF), ("W", -OVAL_HALF)):
        n = 12
        for j in range(n):
            a = -math.pi / 2 + math.pi * (j + 0.5) / n
            a = a if end == "E" else math.pi - a
            seg = 2 * (rb + bt / 2) * math.tan(math.pi / (2 * n)) + 0.01
            box(st, f"Board{end}{j:02d}", (cx + (rb + bt / 2) * math.cos(a), (rb + bt / 2) * math.sin(a), bh / 2),
                (bt, seg, bh), BLUE, yaw_deg=math.degrees(a))
    st.GetRootLayer().customLayerData = {"cascade:scenario": "skate_race", "oval_half": str(OVAL_HALF),
                                         "lanes": " ".join(map(str, OVAL_LANES)), "finish_x": "0.0"}
    dress_floor(st, -3.6, -2.6, 3.6, 2.6, 90, seed=51, z0=0.006)
    st.Save()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for fn, name in ((backstage, "backstage_obstacles"), (stage_edge, "stage_edge"), (multi_room, "multi_room_fetch"),
                     (skate_plaza, "skate_plaza"), (skate_oval, "skate_race")):
        path = a.output / f"{name}.usda"
        if path.exists():
            path.unlink()
        fn(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
