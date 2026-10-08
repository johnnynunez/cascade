"""B32c: the neighbour-colour overlap box ignores a cloud's lowest centimetre.

Measured live (Isaac 6.2 PhysX + YOLOE, bare scene, both demo cameras; 86
decisions logged where BeliefStore._identity_ok compared the side camera's view
of the bin with the bin belief the top camera named): with the B32b box (the
2nd-98th percentile of the whole cloud) the two views of the bin scored IoU
0.708-0.926, median 0.744, so the shipped 0.75 threshold split the bin in about
half of the first decisions (live A/B: 3 beliefs in 2/3 runs). The loss is
one axis only: YOLOE's masks include table pixels in front of the bin (a
median 35 % of the side camera's bin cloud sits at table height, 12 % of the
top camera's), and the top camera's box ran 3.5 cm past the near wall. Those
points sit at the object's own lowest height. Dropping every point within
1 cm of the cloud's own low point (2nd percentile of z) before the box gives
0.870-0.945 (median 0.921) for the same 86 decisions, while a 5 x 5 x 8 cm prop
in the bin corner or next to it scores <= 0.077 (<= 0.118 before). The
threshold stays 0.75.

The real clouds of the hardest decisions are pinned in
tests/fixtures/b32b_live_bin_clouds/ (receipt.json: run, decision, hashes).
"""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from conftest import REPO

from cascade.memory.beliefs import (
    NEIGHBOUR_COLOUR_IOU, BeliefStore, FrameObservation, _box_iou, _cloud_box,
)
from cascade.perception.grounding import oriented_bbox
from test_colour_identity_beliefs import BIN, _cube, _obs, _render

FIX = REPO / "tests" / "fixtures" / "b32b_live_bin_clouds"
RECEIPT = json.loads((FIX / "receipt.json").read_text())
CLOUDS = np.load(FIX / "clouds.npz")
SAME = [p["pair"] for p in RECEIPT["pairs"] if p["kind"] == "same"]
DIFFERENT = [p["pair"] for p in RECEIPT["pairs"] if p["kind"] == "different"]


def _whole_cloud_box(p, q=2.0):
    """The B32b box (whole cloud, 2nd-98th percentile): the premise."""
    p = np.asarray(p, dtype=float).reshape(-1, 3)
    return np.percentile(p, q, axis=0), np.percentile(p, 100 - q, axis=0)


def _cloud_obs(points, colour, source, label="bin"):
    pts = np.asarray(points, dtype=float)
    c, e, _ = oriented_bbox(pts)
    return FrameObservation(label, c, 0.8, extent=e, top_z=float(pts[:, 2].max()),
                            color=colour, points=pts, source=source)


def _two_views(belief_cloud, obs_cloud, colours=("orange", "yellow")):
    store = BeliefStore()
    (first,) = store.update_frame([_cloud_obs(belief_cloud, colours[0], "isaac")], t=1.0)
    (second,) = store.update_frame([_cloud_obs(obs_cloud, colours[1], "isaac_side")], t=1.5)
    return store, first, second


def test_the_fixture_is_the_one_the_receipt_describes():
    assert hashlib.sha256((FIX / "clouds.npz").read_bytes()).hexdigest() == RECEIPT["clouds_npz_sha256"]
    assert len(SAME) >= 3 and len(DIFFERENT) >= 2
    for p in RECEIPT["pairs"]:
        a, b = CLOUDS[f"p{p['pair']}_obs"], CLOUDS[f"p{p['pair']}_belief"]
        assert a.shape == (p["points"][0], 3) and b.shape == (p["points"][1], 3)


@pytest.mark.parametrize("pair", SAME[:3])
def test_premise_the_whole_cloud_box_splits_the_live_bin(pair):
    """What happened live: the bin's two views under the B32b box."""
    a, b = CLOUDS[f"p{pair}_obs"], CLOUDS[f"p{pair}_belief"]
    v = _box_iou(_whole_cloud_box(a), _whole_cloud_box(b))
    assert v < NEIGHBOUR_COLOUR_IOU
    assert v == pytest.approx(RECEIPT["pairs"][pair]["iou_shipped_box"], abs=1e-3)


@pytest.mark.parametrize("pair", SAME)
def test_live_views_of_the_bin_fuse_into_one_belief(pair):
    """RED on main: the four hardest live decisions each made a second bin."""
    store, first, second = _two_views(CLOUDS[f"p{pair}_belief"], CLOUDS[f"p{pair}_obs"])
    assert second is first, RECEIPT["pairs"][pair]
    assert len(store.all(now=2.0)) == 1
    assert first.source_colors == {"isaac": "orange", "isaac_side": "yellow"}
    v = _box_iou(_cloud_box(CLOUDS[f"p{pair}_obs"]), _cloud_box(CLOUDS[f"p{pair}_belief"]))
    assert v == pytest.approx(RECEIPT["pairs"][pair]["iou_floor_box"], abs=1e-3)
    assert v >= NEIGHBOUR_COLOUR_IOU + 0.10, "the live margin above the threshold"


@pytest.mark.parametrize("pair", DIFFERENT)
def test_live_prop_in_or_next_to_the_bin_stays_apart(pair):
    """A yellow 5 cm prop in the bin corner / next to it vs the orange bin."""
    store, first, second = _two_views(CLOUDS[f"p{pair}_belief"], CLOUDS[f"p{pair}_obs"])
    assert second is not first
    assert len(store.all(now=2.0)) == 2
    v = _box_iou(_cloud_box(CLOUDS[f"p{pair}_obs"]), _cloud_box(CLOUDS[f"p{pair}_belief"]))
    assert v < 0.15


def test_the_bin_on_a_shelf_fuses_too():
    """The low band is relative to the cloud, not to a table height: the same
    live pair lifted 30 cm (a bin on a shelf) scores the same."""
    pair = SAME[0]
    lift = np.array([0.0, 0.0, 0.30], np.float32)
    store, first, second = _two_views(CLOUDS[f"p{pair}_belief"] + lift, CLOUDS[f"p{pair}_obs"] + lift)
    assert second is first


# ── ray-cast: the same effect from first principles ───────────────────────


def _bin_views(dilate_px=0, skirt_cam=None, skirt_m=0.035):
    """Both cameras' bin observations; masks dilated by `dilate_px`, and the
    `skirt_cam` mask extended by the table strip `skirt_m` in front of the
    bin's -y wall (the live top camera's box ran to y = -0.28)."""
    out = []
    for cam, colour in (("isaac", "orange"), ("isaac_side", "yellow")):
        frame, masks = _render(cam, {"bin": BIN})
        m = masks["bin"]
        if dilate_px:
            k = np.ones((2 * dilate_px + 1,) * 2, np.uint8)
            m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        if cam == skirt_cam:
            from test_colour_identity_beliefs import CAMS, FX, H, K, W  # noqa: F401
            T = CAMS[cam]
            us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
            rays = np.stack([(us - W / 2) / FX, (vs - H / 2) / FX, np.ones_like(us)], -1)
            pts = T[:3, 3] + (rays * frame.depth_m[..., None]) @ T[:3, :3].T
            on_table = np.abs(pts[..., 2]) < 0.002
            m = m | (on_table & (pts[..., 1] < -0.245) & (pts[..., 1] > -0.245 - skirt_m)
                     & (pts[..., 0] > 0.105) & (pts[..., 0] < 0.255))
        out.append(_obs(cam, frame, m, colour, "bin"))
    return out


def test_a_table_skirt_in_one_cameras_mask_does_not_split_the_bin():
    """The live failure from first principles: RED on main."""
    top, side = _bin_views(skirt_cam="isaac")
    assert _box_iou(_whole_cloud_box(top.points), _whole_cloud_box(side.points)) < NEIGHBOUR_COLOUR_IOU
    store = BeliefStore()
    (first,) = store.update_frame([top], t=1.0)
    assert store.update_frame([side], t=1.5)[0] is first


def test_heavy_mask_bleed_does_not_split_the_bin():
    """2 px at 320 x 180 (~8 px at the live 1280 x 720): B32b's box fell to
    0.27 and kept the bin as two; the floor-trimmed box keeps ~0.9."""
    top, side = _bin_views(dilate_px=2)
    assert _box_iou(_whole_cloud_box(top.points), _whole_cloud_box(side.points)) < 0.5
    store = BeliefStore()
    (first,) = store.update_frame([top], t=1.0)
    assert store.update_frame([side], t=1.5)[0] is first


@pytest.mark.parametrize("dilate_px", [0, 1])
@pytest.mark.parametrize("where", [(0.18, -0.17), (0.145, -0.205)])
def test_a_short_cube_inside_the_bin_stays_apart_under_bleed(where, dilate_px):
    """The ray-cast worst case of B32b's notes: a 3.5 cm cube wholly inside
    the bin, seen by a camera that never named the bin, masks bleeding."""
    props = {"bin": BIN, "cube": _cube(*where)}
    top, tm = _render("isaac", {"bin": BIN})
    store = BeliefStore()
    (bin_,) = store.update_frame([_obs("isaac", top, tm["bin"], "orange", "bin")], t=1.0)
    side, sm = _render("isaac_side", props)
    m = sm["cube"]
    if dilate_px:
        k = np.ones((2 * dilate_px + 1,) * 2, np.uint8)
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    if m.sum() < 10:
        pytest.skip("the cube is hidden from the side camera at this position")
    (cube,) = store.update_frame([_obs("isaac_side", side, m, "yellow", "cube")], t=1.5)
    assert cube is not bin_
    assert len(store.all(now=2.0)) == 2


# ── the box itself ────────────────────────────────────────────────────────


def test_the_box_drops_the_clouds_lowest_centimetre():
    """Walls 0-6 cm high around a 15 cm square, plus a table skirt at z = 0
    reaching 3.5 cm past one wall: the box is the walls' footprint."""
    rng = np.random.default_rng(0)
    walls = np.column_stack([rng.uniform(0.105, 0.255, 600), rng.uniform(-0.245, -0.095, 600),
                             rng.uniform(0.0, 0.06, 600)])
    skirt = np.column_stack([rng.uniform(0.105, 0.255, 250), rng.uniform(-0.28, -0.245, 250),
                             np.zeros(250)])
    lo, hi = _cloud_box(np.vstack([walls, skirt]))
    assert lo[1] > -0.25 and hi[1] < -0.09
    assert lo[2] >= 0.01 - 1e-9 and hi[2] <= 0.06


def test_a_flat_cloud_keeps_its_whole_box():
    """Nothing above the low band (a mat, a sheet): the whole cloud is the
    box, as in B32b -- the band only applies when there is shape above it."""
    rng = np.random.default_rng(1)
    flat = np.column_stack([rng.uniform(0.2, 0.3, 300), rng.uniform(0.0, 0.1, 300),
                            rng.uniform(0.0, 0.004, 300)])
    lo, hi = _cloud_box(flat)
    np.testing.assert_allclose(lo[:2], np.percentile(flat, 2, axis=0)[:2])
    np.testing.assert_allclose(hi[:2], np.percentile(flat, 98, axis=0)[:2])
    store, first, second = _two_views(flat, flat + np.array([0.001, 0.001, 0.0]))
    assert second is first
