#!/usr/bin/env python3
"""Same-colour twins vs the belief store, measured against MuJoCo physics truth.

Backlog B31 (ARCHITECTURE "Known limitations": same-colour identical objects
closer than 8 cm blurred into one belief). Renders RGB-D of two identical red
cubes (3.5 cm, the demo prop colour `sim/demo_scene.PROP_RGBA["red"]`) from a
declared MJCF camera at a sweep of centre-to-centre separations, feeds the
frames through the REAL always-on fusion path (`WorldWatcher._tick` ->
`BeliefStore.update_frame`), and scores the belief store against the cubes'
`data.xpos`. Two independent switches:

  detector  union      `MockDetector()` default: ONE detection per colour
                       (every red pixel), i.e. detector-side merging
            instances  `MockDetector(instances=True)`: one detection per
                       connected red blob in the image
  store     legacy     `BeliefStore(instance_association=False)`: one
                       `update()` per detection (the pre-2026-10-08 store)
            instance   `BeliefStore()`: per-frame instance association

A pair counts as RESOLVED when the store holds exactly two beliefs and, under
the optimal belief<->cube matching, each belief's xy error exceeds the error
the SAME camera makes on that cube ALONE (`floor_m`, the watcher fuses the raw
OBB centre of the visible surface) by at most `TOL_M`. Subtracting the
single-cube floor separates association from lifting bias.

No robot MJCF, no GPU work beyond the offscreen GL context (MUJOCO_GL=egl on
Linux), no network. Usage:

    MUJOCO_GL=egl PYTHONPATH=src python scripts/measure_same_colour_sweep.py \\
        --json benchmark/results/same_colour_separation_sweep.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HALF = 0.0175  # 3.5 cm cube, the demo prop
MID = (0.25, 0.0)
#: centre-to-centre separations (m); 0.035 = faces touching
SEPARATIONS = (0.035, 0.0375, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065,
               0.070, 0.075, 0.080, 0.085, 0.090, 0.100, 0.120)
#: name -> (eye, target, up); "top" = straight down (mock.yaml-like),
#: "oblique" = the perception-truth probe's camera (test_perception_truth_mujoco)
CAMERAS = {
    "top": ((0.25, 0.0, 0.60), (0.25, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "oblique": ((0.70, -0.30, 0.50), (0.25, 0.0, HALF), (0.0, 0.0, 1.0)),
}
AXES = {"x": (1.0, 0.0), "y": (0.0, 1.0)}
TICKS = 3
TOL_M = 0.005
FAR = (5.0, 5.0, HALF)  # out of every view: "the other cube is not there"

SCENE = """<mujoco model="same_colour_twins">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="table" type="plane" size="2 2 0.05" rgba="0.72 0.72 0.72 1"/>
    <body name="cube_a" pos="0.25 -0.05 {h}">
      <freejoint/>
      <geom type="box" size="{h} {h} {h}" rgba="{rgba}"/>
    </body>
    <body name="cube_b" pos="0.25 0.05 {h}">
      <freejoint/>
      <geom type="box" size="{h} {h} {h}" rgba="{rgba}"/>
    </body>
    {cameras}
  </worldbody>
</mujoco>
"""


def write_scene(directory: Path) -> Path:
    from cascade.sim.demo_scene import PROP_RGBA
    from cascade.sim.mujoco_rgbd import camera_xml

    cams = "\n    ".join(camera_xml(name, eye, target, 58.0, up=up)
                         for name, (eye, target, up) in CAMERAS.items())
    path = Path(directory) / "same_colour_twins.xml"
    path.write_text(SCENE.format(h=HALF, rgba=PROP_RGBA["red"], cameras=cams))
    return path


def twin_positions(sep: float, axis: str):
    dx, dy = AXES[axis]
    a = (MID[0] - dx * sep / 2, MID[1] - dy * sep / 2, HALF)
    b = (MID[0] + dx * sep / 2, MID[1] + dy * sep / 2, HALF)
    return a, b


def fuse(sim, *, instances: bool, instance_association: bool, ticks: int = TICKS):
    """Render the current scene once and run `ticks` watcher ticks on it
    through the real WorldWatcher fusion path; returns (beliefs, number of
    detections the detector returned on the last tick)."""
    from cascade.memory.beliefs import BeliefStore
    from cascade.perception.detector import MockDetector
    from cascade.perception.grounding import Extrinsics
    from cascade.perception.world import WatchedCamera, WorldWatcher

    store = BeliefStore(instance_association=instance_association)
    rgb, depth = sim.render()
    holder = {"frame": None}
    stream = SimpleNamespace(name="sweep", latest=lambda: holder["frame"],
                             set_overlay=lambda **kw: None)
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f),
                        Extrinsics(T=sim.extrinsic()))
    watcher = WorldWatcher([cam], MockDetector(label="red cube", instances=instances), store)
    from cascade.types import Frame

    t = 100.0
    for tick in range(ticks):
        t += 0.35
        holder["frame"] = Frame(rgb=rgb.copy(), depth_m=depth.copy(), K=sim.K, t=t,
                                frame_id=tick + 1, depth_source="sensor")
        watcher._tick(cam)
    return store.all(now=t), len(watcher.last_dets.get("sweep", []))


def xy_errors(beliefs, truths) -> list[float | None]:
    """Per cube: xy error of its belief under the optimal one-to-one matching
    (None when fewer beliefs than cubes leave a cube without one)."""
    pos = [np.asarray(b.position, float)[:2] for b in beliefs]
    tru = [np.asarray(t, float)[:2] for t in truths]
    if not pos:
        return [None] * len(tru)
    best = None
    k = min(len(pos), len(tru))
    for cubes in itertools.permutations(range(len(tru)), k):
        for bel in itertools.permutations(range(len(pos)), k):
            errs = [float(np.linalg.norm(pos[b] - tru[c])) for b, c in zip(bel, cubes)]
            total = sum(errs)
            if best is None or total < best[0]:
                out = [None] * len(tru)
                for e, c in zip(errs, cubes):
                    out[c] = e
                best = (total, out)
    return best[1]


def run(separations=SEPARATIONS, cameras=tuple(CAMERAS), axes=tuple(AXES),
        width: int = 640, height: int = 480) -> dict:
    from cascade.sim.mujoco_rgbd import MujocoRGBD

    rows = []
    with tempfile.TemporaryDirectory(prefix="b31_sweep_") as tmp:
        scene = write_scene(Path(tmp))
        for cam_name in cameras:
            sim = MujocoRGBD(scene, camera=cam_name, width=width, height=height)
            try:
                for axis in axes:
                    for sep in separations:
                        a, b = twin_positions(sep, axis)
                        floors = []
                        for here, gone in (("cube_a", "cube_b"), ("cube_b", "cube_a")):
                            sim.place_free_body(here, a if here == "cube_a" else b)
                            sim.place_free_body(gone, FAR)
                            alone, _ = fuse(sim, instances=True, instance_association=True)
                            err = xy_errors(alone, [sim.body_pos(here)])[0]
                            floors.append(None if len(alone) != 1 else err)
                        sim.place_free_body("cube_a", a)
                        sim.place_free_body("cube_b", b)
                        truths = [sim.body_pos("cube_a"), sim.body_pos("cube_b")]
                        for detector, store in itertools.product(("union", "instances"),
                                                                  ("legacy", "instance")):
                            beliefs, n_dets = fuse(sim, instances=detector == "instances",
                                                   instance_association=store == "instance")
                            errs = xy_errors(beliefs, truths)
                            resolved = (
                                len(beliefs) == 2 and None not in floors
                                and all(e is not None and e - f <= TOL_M
                                        for e, f in zip(errs, floors))
                            )
                            rows.append({
                                "camera": cam_name, "axis": axis, "separation_m": sep,
                                "detector": detector, "store": store,
                                "detections": n_dets,
                                "beliefs": len(beliefs),
                                "xy_err_m": [None if e is None else round(e, 4) for e in errs],
                                "floor_m": [None if f is None else round(f, 4) for f in floors],
                                "resolved": bool(resolved),
                            })
            finally:
                sim.close()
    return {"rows": rows, "summary": summarize(rows)}


def summarize(rows) -> dict:
    """Per (camera, axis, detector, store): the separations resolved, and the
    smallest separation from which EVERY larger one in the sweep is resolved."""
    out: dict = {}
    keys = sorted({(r["camera"], r["axis"], r["detector"], r["store"]) for r in rows})
    for key in keys:
        sub = sorted((r for r in rows if (r["camera"], r["axis"], r["detector"], r["store"]) == key),
                     key=lambda r: r["separation_m"])
        ok = [r["separation_m"] for r in sub if r["resolved"]]
        from_m = None
        for r in reversed(sub):
            if not r["resolved"]:
                break
            from_m = r["separation_m"]
        out["/".join(key)] = {"resolved_m": ok, "resolved_from_m": from_m,
                              "detections": [r["detections"] for r in sub],
                              "beliefs": [r["beliefs"] for r in sub]}
    return out


def _print(report: dict) -> None:
    seps = sorted({r["separation_m"] for r in report["rows"]})
    print("separation (cm):              " + "".join(f"{s * 100:6.2f}" for s in seps))
    for key, s in report["summary"].items():
        marks = []
        for sep, d, n in zip(seps, s["detections"], s["beliefs"]):
            marks.append("    ok" if sep in s["resolved_m"] else f"  {d}d{n}b")
        print(f"{key:30}" + "".join(marks))
    print("\n(ok = two beliefs, each within the single-cube floor + "
          f"{TOL_M * 100:.1f} cm of physics truth; NdMb = N detections, M beliefs,\n"
          " not resolved: 1d = the DETECTOR merged the pair, 2d1b = the STORE did)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, help="write the report here")
    args = ap.parse_args()
    report = run()
    report["meta"] = {
        "cube_m": 2 * HALF, "mid_xy_m": list(MID), "ticks": TICKS, "tol_m": TOL_M,
        "cameras": {k: {"eye": list(v[0]), "target": list(v[1])} for k, v in CAMERAS.items()},
        "truth": "MuJoCo data.xpos of each free cube body",
        "path": "WorldWatcher._tick -> BeliefStore.update_frame (MockDetector colour masks)",
    }
    _print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        # meta readable, one line per summary key and per measured row
        meta = json.dumps(report["meta"], indent=1).replace("\n", "\n ")
        summary = ",\n  ".join(f"{json.dumps(k)}: {json.dumps(v)}"
                               for k, v in report["summary"].items())
        body = ",\n  ".join(json.dumps(r) for r in report["rows"])
        args.json.write_text(f'{{\n "meta": {meta},\n "summary": {{\n  {summary}\n }},\n'
                             f' "rows": [\n  {body}\n ]\n}}\n')
        json.loads(args.json.read_text())  # never leave a broken report behind
        print(f"[+] wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
