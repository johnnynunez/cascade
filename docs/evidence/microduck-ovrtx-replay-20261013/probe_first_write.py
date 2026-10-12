# Probe used for docs/MICRODUCK_OVRTX_REPLAY.md: flat body prims WITHOUT the replay sentinel, see README.md here.
"""Debug: rest pose rotated by yaw psi, flat layout (world writes). Assembled at every psi => transform path is right."""
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom, UsdPhysics

from cascade.sim.microduck_ovrtx_replay import build_render_scene, look_at
from cascade.sim.ovrtx_renderer import CameraSpec, OvrtxRenderer, SceneSnapshot

out = Path(sys.argv[1])
order = sys.argv[2]
out.mkdir(exist_ok=True)
duck = Path.home() / "Projects/microducks/microduck_gtceu/assets/MicroDuck.usd"
st = Usd.Stage.Open(str(duck))
cache = UsdGeom.XformCache()
rest = {p.GetName(): np.array(cache.GetLocalToWorldTransform(p), dtype=float).T for p in st.Traverse()
        if p.HasAPI(UsdPhysics.RigidBodyAPI)}
layout = build_render_scene(out / "s.usdc", course_usd=None, microduck_usd=duck, robots=["d"])
cam = CameraSpec("c", 320, 240, 300.0, 300.0, np.eye(4), near_m=0.02, far_m=20)
T = look_at(np.array([0.45, -0.45, 0.35]), np.array([0.0, 0.0, 0.12]))
tiles = []
paths = list(layout["bodies"]["d"].values())
if order == "last":
    paths = paths[1:] + paths[:1]
with OvrtxRenderer(out / "s.usdc", [cam], dynamic_paths=paths, device=0,
                   sdk_versions=(("ovrtx", "0.5.1.385782"), ("ovstage", "0.2.1.385922"))) as r:
    for j, psi in enumerate((0, 45, 90, 180)):
        Rz = np.eye(4)
        c, s = math.cos(math.radians(psi)), math.sin(math.radians(psi))
        Rz[:2, :2] = [[c, -s], [s, c]]
        Rz[2, 3] = 0.12
        Rz[0, 3] = 0.08 * j
        tf = {layout["bodies"]["d"][n]: Rz @ rest[n] for n in layout["bodies"]["d"]}
        f = r.render(SceneSnapshot("dbg", "yaw", j + 1, 0.01 * (j + 1), time.monotonic(), tf, {"c": T}))["c"]
        im = Image.fromarray(np.ascontiguousarray(f.rgb[..., ::-1]))
        ImageDraw.Draw(im).text((5, 5), f"yaw {psi}", fill=(255, 0, 0))
        tiles.append(im)
W = Image.new("RGB", (320 * 4, 240))
for i, im in enumerate(tiles):
    W.paste(im, (320 * i, 0))
W.save(out / f"yaw_{order}.png")
print("done")
