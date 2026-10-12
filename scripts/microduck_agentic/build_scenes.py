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


def cylinder(st, name, center, radius, height, color, segments=20, collide=True):
    cx, cy, z0 = center
    pts = []
    for z in (z0, z0 + height):
        for k in range(segments):
            a = 2 * math.pi * k / segments
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z))
    counts, idx = [segments, segments], list(range(segments - 1, -1, -1)) + list(range(segments, 2 * segments))
    for k in range(segments):
        a, b = k, (k + 1) % segments
        counts.append(4)
        idx += [a, b, segments + b, segments + a]
    return _mesh(st, name, pts, counts, idx, color, collide)


def walls(st, prefix, x0, y0, x1, y1, height=0.25, t=0.05, color=WALL, gaps=()):
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
    st.Save()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for fn, name in ((backstage, "backstage_obstacles"), (stage_edge, "stage_edge"), (multi_room, "multi_room_fetch")):
        path = a.output / f"{name}.usda"
        if path.exists():
            path.unlink()
        fn(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
