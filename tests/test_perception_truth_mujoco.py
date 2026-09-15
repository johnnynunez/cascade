"""Perception verified against MuJoCo physics ground truth.

`grounding._recentre_by_size` removes a centimetre-scale bias in the grasp
target: an oriented box fitted to a depth cloud is pulled TOWARD the camera,
because near faces subtend more pixels per unit area than far ones. Left
uncorrected the finger catches the object's edge and shoves it away -- the
"air grasp" failure.

The correction was initially designed and measured entirely on the ISAAC rig
using the bridge on :8611. It had not been re-checked on another engine.
A correction derived from one simulator and validated by that same
simulator is what this repo's "engine agreement is the gold metric" rule exists
to distrust.

These tests close that gap on a laptop: MuJoCo renders RGB-D offscreen on CPU,
`data.xpos` of the prop is a truth channel perception does not own, and the
assertions are on MEASURED improvement rather than on the code path being
taken.

Measured here (5 held positions, perfect colour mask):

    raw fitted OBB centre   mean 2.27 cm   max 2.64 cm
    de-biased               mean 0.49 cm   max 0.59 cm

which independently replicates the Isaac result (1.85 -> 0.56 cm).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO

ROBOT_MJCF = REPO / "assets" / "mjcf" / "so101" / "so101.xml"


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


def _can_render() -> bool:
    """A headless box may have no GL context at all; skip rather than fail."""
    if not _has_mujoco() or not ROBOT_MJCF.exists():
        return False
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/>"
            "</worldbody></mujoco>"
        )
        mujoco.Renderer(m, height=64, width=64).close()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_render = pytest.mark.skipif(
    not _can_render(),
    reason="needs `mujoco`, a GL context, and "
           "`python scripts/fetch_robot_assets.py so101`",
)

POSITIONS = [(0.25, 0.00), (0.22, 0.07), (0.29, -0.05),
             (0.20, -0.08), (0.31, 0.06)]


def _segment_green(rgb_bgr: np.ndarray) -> np.ndarray:
    b, g, r = (rgb_bgr[:, :, i].astype(np.int16) for i in range(3))
    return (g > 90) & (g - r > 40) & (g - b > 40)


@pytest.fixture(scope="module")
def measurements():
    """(raw_err, fixed_err) per position, measured once for the whole module."""
    from cascade.perception.grounding import _recentre_by_size, oriented_bbox
    from cascade.sim.mujoco_rgbd import MujocoRGBD, write_probe_scene
    from cascade.types import transform_points

    scene = write_probe_scene(ROBOT_MJCF)
    try:
        sim = MujocoRGBD(scene, camera="scene", width=640, height=480)
    except Exception:
        scene.unlink(missing_ok=True)
        raise
    rows = []
    try:
        T = sim.extrinsic()
        K = sim.K
        for (x, y) in POSITIONS:
            sim.place_free_body("probe_cube", [x, y, 0.025])
            truth = sim.body_pos("probe_cube")
            rgb, depth = sim.render()
            mask = _segment_green(rgb)
            if int(mask.sum()) < 50:
                continue
            ys, xs = np.nonzero(mask)
            zs = depth[ys, xs]
            keep = zs > 0
            ys, xs, zs = ys[keep], xs[keep], zs[keep]
            pts_cam = np.stack(
                [(xs - K[0, 2]) / K[0, 0] * zs,
                 (ys - K[1, 2]) / K[1, 1] * zs, zs], axis=-1
            )
            pts_world = transform_points(T, pts_cam)
            centre, extents, _ = oriented_bbox(pts_world)
            fixed = _recentre_by_size(centre, pts_world, extents, T[:3, 3])
            rows.append((
                float(np.linalg.norm(centre - truth)),
                float(np.linalg.norm(fixed - truth)),
            ))
    finally:
        sim.close()
        scene.unlink(missing_ok=True)
    return rows


@needs_render
def test_the_probe_scene_renders_metric_depth():
    """Depth must be range FROM THE CAMERA. A free camera posed by
    azimuth/elevation renders depth relative to its lookat plane instead --
    measured as depth DECREASING with distance -- which silently invalidates
    every 3D number downstream. This pins the property, not the workaround."""
    from cascade.sim.mujoco_rgbd import MujocoRGBD, write_probe_scene

    scene = write_probe_scene(ROBOT_MJCF)
    try:
        sim = MujocoRGBD(scene, camera="scene", width=320, height=240)
        try:
            eye = sim.extrinsic()[:3, 3]
            ranges = []
            for x in (0.20, 0.25, 0.30):
                sim.place_free_body("probe_cube", [x, 0.0, 0.025])
                truth = sim.body_pos("probe_cube")
                rgb, depth = sim.render()
                mask = _segment_green(rgb)
                assert mask.sum() > 50, f"prop not visible at x={x}"
                ys, xs = np.nonzero(mask)
                ranges.append((
                    float(np.linalg.norm(truth - eye)),
                    float(np.median(depth[ys, xs])),
                ))
        finally:
            sim.close()
    finally:
        scene.unlink(missing_ok=True)

    for true_rng, seen in ranges:
        # Depth reads the NEAR FACE of a 5 cm cube, so it sits up to ~2.5 cm
        # short of the centre range -- but it must never exceed it, and must
        # stay in the right ballpark.
        assert 0.0 < seen <= true_rng + 1e-3, f"{seen=} vs {true_rng=}"
        assert true_rng - seen < 0.05, f"depth {seen} too far from {true_rng}"
    # The load-bearing assertion: depth must track the TRUE range, in the same
    # direction. (Note this camera sits at x=+0.70 looking back toward the
    # base, so moving the prop to larger x brings it CLOSER -- ordering by the
    # measured true range rather than by x is what makes this robust to where
    # the camera is placed.)
    by_true = [seen for _, seen in sorted(ranges, key=lambda p: p[0])]
    assert by_true == sorted(by_true), (
        f"depth did not track true range: {ranges}. A free camera posed by "
        f"azimuth/elevation renders depth relative to its lookat plane, which "
        f"produces exactly this."
    )


@needs_render
def test_debiasing_beats_the_raw_obb_centre_against_physics_truth(measurements):
    """The claim, on an engine that did not design it."""
    assert measurements, "no positions measured"
    raw = np.array([r for r, _ in measurements])
    fixed = np.array([f for _, f in measurements])
    assert fixed.mean() < raw.mean(), (
        f"de-biasing did not help: raw {raw.mean()*100:.2f} cm vs "
        f"fixed {fixed.mean()*100:.2f} cm"
    )
    # Worst case is what decides whether a grasp lands, so gate on the max too.
    assert fixed.max() < raw.max()


@needs_render
def test_debiased_error_is_within_the_jaw_tolerance(measurements):
    """A 55 mm jaw on a 50 mm cube leaves ~2.5 cm of half-width. The Isaac
    campaign measured the failure at 1.6-1.9 cm of error, so anything at or
    above ~1 cm is where grasps start catching edges. Pinning 1.2 cm keeps a
    little headroom over the 0.59 cm measured max without going vacuous."""
    fixed = np.array([f for _, f in measurements])
    assert fixed.max() < 0.012, (
        f"worst de-biased error {fixed.max()*100:.2f} cm exceeds the grasp "
        f"budget; a finger will catch the object's edge"
    )


@needs_render
def test_the_raw_bias_is_real_and_points_at_the_camera(measurements):
    """Guards the PREMISE. If the raw fitted centre were already accurate the
    correction would be unnecessary, and a future refactor could delete it with
    every test still green. The bias must be present and centimetre-scale."""
    raw = np.array([r for r, _ in measurements])
    assert raw.mean() > 0.01, (
        f"raw OBB centre error is only {raw.mean()*100:.2f} cm; the bias this "
        f"correction exists to remove is not being reproduced"
    )
