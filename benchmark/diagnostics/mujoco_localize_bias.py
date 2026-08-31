"""Localization bias in MuJoCo, measured against physics ground truth.

`grounding._recentre_by_size` corrects a 1.6-1.9 cm bias in the grasp target.
It was designed and measured entirely on the ISAAC rig -- every script that
produced those numbers hardcodes a /home/johnny path and a bridge on :8611 --
so this re-measures it on a second engine, on a laptop, against `data.xpos`.

    python benchmark/diagnostics/mujoco_localize_bias.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cascade.perception.grounding import (  # noqa: E402
    _recentre_by_size, oriented_bbox,
)
from cascade.sim.mujoco_rgbd import (  # noqa: E402
    DEPTH_NOISE_M, MujocoRGBD, write_probe_scene,
)
from cascade.types import transform_points  # noqa: E402

ROBOT_MJCF = REPO / "assets" / "mjcf" / "so101" / "so101.xml"
POSITIONS = [(0.25, 0.00), (0.22, 0.07), (0.29, -0.05),
             (0.20, -0.08), (0.31, 0.06)]


def segment_prop(rgb_bgr: np.ndarray) -> np.ndarray:
    """Mask the green prop by colour.

    A colour threshold stands in for the detector on purpose: this bench
    measures the GEOMETRY pipeline (backprojection -> OBB -> de-bias), not
    YOLOE's mask quality. A perfect mask is also the harder test for the
    correction -- it removes the shadow-inflation term and leaves only the
    density bias the fix actually claims to remove.
    """
    b, g, r = (rgb_bgr[:, :, i].astype(np.int16) for i in range(3))
    return (g > 90) & (g - r > 40) & (g - b > 40)


def measure() -> list[dict]:
    scene = write_probe_scene(ROBOT_MJCF)
    try:
        sim = MujocoRGBD(scene, camera="scene", width=640, height=480)
    except Exception:
        scene.unlink(missing_ok=True)
        raise
    out = []
    try:
        T = sim.extrinsic()
        for (x, y) in POSITIONS:
            sim.place_free_body("probe_cube", [x, y, 0.025])
            truth = sim.body_pos("probe_cube")
            rgb, depth = sim.render()
            mask = segment_prop(rgb)
            if int(mask.sum()) < 50:
                continue
            ys, xs = np.nonzero(mask)
            zs = depth[ys, xs]
            keep = zs > 0
            ys, xs, zs = ys[keep], xs[keep], zs[keep]
            K = sim.K
            pts_cam = np.stack(
                [(xs - K[0, 2]) / K[0, 0] * zs,
                 (ys - K[1, 2]) / K[1, 1] * zs, zs], axis=-1
            )
            pts_world = transform_points(T, pts_cam)
            centre, extents, _ = oriented_bbox(pts_world)
            fixed = _recentre_by_size(centre, pts_world, extents, T[:3, 3])
            out.append({
                "xy": (x, y),
                "raw": float(np.linalg.norm(centre - truth)),
                "fixed": float(np.linalg.norm(fixed - truth)),
                "px": int(mask.sum()),
            })
    finally:
        sim.close()
        scene.unlink(missing_ok=True)
    return out


def main() -> int:
    if not ROBOT_MJCF.exists():
        print("needs the fetched SO-101 MJCF: "
              "python scripts/fetch_robot_assets.py so101")
        return 2
    rows = measure()
    if not rows:
        print("prop never visible; check the camera pose in the scene")
        return 1
    print(f"{'placed (x,y)':>14} {'px':>6} {'raw OBB':>9} {'de-biased':>11} {'delta':>9}")
    print("-" * 54)
    for r in rows:
        print(f"  ({r['xy'][0]:.2f},{r['xy'][1]:.2f}) {r['px']:8d} "
              f"{r['raw']*100:8.2f}cm {r['fixed']*100:9.2f}cm "
              f"{(r['raw']-r['fixed'])*100:+8.2f}cm")
    raw = np.array([r["raw"] for r in rows])
    fix = np.array([r["fixed"] for r in rows])
    print("-" * 54)
    print(f"{'mean':>14} {'':6} {raw.mean()*100:8.2f}cm {fix.mean()*100:9.2f}cm")
    print(f"{'max':>14} {'':6} {raw.max()*100:8.2f}cm {fix.max()*100:9.2f}cm")
    print(f"\ndepth-accuracy floor on this GL path: {DEPTH_NOISE_M*100:.2f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
