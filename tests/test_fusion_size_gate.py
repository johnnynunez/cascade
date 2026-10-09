"""B40: a size-consistency check in the belief store's fusion gate.

The defect (B32b live evidence, `docs/evidence/b32b-colour-identity-live-20261008/`):
the side camera `isaac_side` names the open bin AND a yellow prop inside it
"yellow". On frames where it sees the bin but not the prop, its view of the
bin (~23 cm OBB, ~19 cm robust diameter) was fused into the PROP's belief
(~9 cm OBB, ~7 cm diameter), because the names match and the prop's centre was
the nearer one -- 3 frames per run, identical with the one-name rule. For those
ticks the prop belief carried the bin's cloud (what grasp-from-memory plans
on), the bin's extent and an alias "building block".

Mechanism (memory/beliefs.py): a view's size is the robust horizontal
DIAMETER of its real-mask cloud (`_view_diameter`: the largest 2-98 % span
over 8 directions in the table plane, on the cloud without its lowest
centimetre, the B32c floor band); a belief remembers the LARGEST diameter of
any view fused into it (`ObjectBelief.diameter_m`, a running max, because a
partial or occluded view only ever looks smaller). The gate refuses a fusion
when the view is more than SIZE_GATE_RATIO (2.0) times that AND more than
SIZE_GATE_EXCESS_M (5 cm) larger. It only refuses: without a real-mask cloud
on either side it abstains (the old behaviour). `memory.size_gate: false`
restores the old store byte for byte (golden test below).

Measured (CPU, `docs/evidence/b40-fusion-size-gate-20261009/`): live views of
the bin 0.183-0.235 m diameter (205 views, both cameras), live views of the
5 x 5 x 8 cm prop 0.066-0.074 (15); same object across views <= x1.29 live;
container view / prop's largest view >= x2.48 and +10.9 cm live. 2.0 and 5 cm
are estimates between those, checked against ray-cast occlusion and bleed
(below). Geometry: the bare Isaac scene ray-cast of test_colour_identity_beliefs
(calibrated poses, bridge optics, 320 x 180), the live clouds of
tests/fixtures/b32b_live_bin_clouds.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import cascade.memory.beliefs as bm
import fusion_size_gate_fixture as golden
from conftest import REPO
from cascade.memory.beliefs import BeliefStore, FrameObservation
from cascade.perception.grounding import Extrinsics, oriented_bbox
from test_colour_identity_beliefs import (
    BIN, BIN_SIDE_HSV, BIN_TOP_HSV, CAMS, FX, H, W, YELLOW_HSV, _cloud, _cube, _det, _grid_box,
    _obs, _paint, _PerFrameDetector, _render, _runtime, _two_camera_watcher,
)

LIVE = REPO / "tests" / "fixtures" / "b32b_live_bin_clouds"
CLOUDS = np.load(LIVE / "clouds.npz")
RECEIPT = json.loads((LIVE / "receipt.json").read_text())
SAME = [p["pair"] for p in RECEIPT["pairs"] if p["kind"] == "same"]
#: the live in_bin_centre run had the 5 x 5 x 8 cm prop here (PhysX truth)
LIVE_PROP_XY = (0.1833, -0.1584)


def _prop(x: float, y: float):
    """The bridge's bare-scene prop: 5 x 5 cm, 8 cm tall (scripts/isaac_bridge.py)."""
    return [((x, y, 0.04), (0.05, 0.05, 0.08))]


def _diameter_ref(points) -> float:
    """The size the gate measures, written out independently: drop the points
    within 1 cm of the cloud's own 2nd-percentile z (if >= 10 remain), project
    x-y on 8 directions 22.5 deg apart, take the largest 2nd-98th percentile
    span."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    above = p[p[:, 2] > np.percentile(p[:, 2], 2.0) + 0.01]
    if above.shape[0] >= 10:
        p = above
    ang = np.deg2rad(np.arange(0.0, 180.0, 22.5))
    proj = p[:, :2] @ np.stack([np.cos(ang), np.sin(ang)])
    lo, hi = np.percentile(proj, [2.0, 98.0], axis=0)
    return float(np.max(hi - lo))


def _cloud_obs(points, colour, source, label):
    pts = np.asarray(points, dtype=float)
    c, e, _ = oriented_bbox(pts)
    return FrameObservation(label, c, 0.8, extent=e, top_z=float(pts[:, 2].max()),
                            color=colour, points=pts, source=source)


def _skirted(cam, frame, mask, skirt_m=0.035):
    """The bin mask plus the table strip in front of its -y wall: the live top
    camera's bin box ran to y = -0.28 (B32c), which is also what put the live
    bin belief's centre FARTHER from the side view than the prop's."""
    T = CAMS[cam]
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(us - W / 2) / FX, (vs - H / 2) / FX, np.ones_like(us)], -1)
    pts = T[:3, 3] + (rays * frame.depth_m[..., None]) @ T[:3, :3].T
    strip = np.logical_and.reduce([np.abs(pts[..., 2]) < 0.002, pts[..., 1] < -0.245,
                                   pts[..., 1] > -0.245 - skirt_m, pts[..., 0] > 0.105,
                                   pts[..., 0] < 0.255])
    return np.logical_or(mask, strip)


def _in_bin_scene():
    """Top: bin (with the live table skirt) + prop; side: bin + prop. Exact
    ray-cast masks; the prop where the live run had it."""
    props = {"bin": BIN, "prop": _prop(*LIVE_PROP_XY)}
    top, tm = _render("isaac", props)
    side, sm = _render("isaac_side", props)
    tm["bin"] = _skirted("isaac", top, tm["bin"])
    return (top, tm), (side, sm)


def _grid_obs(centre, sx, sy, colour="yellow", source="isaac", label="box", sz=0.06):
    pts = _grid_box((centre[0], centre[1], sz / 2), (sx, sy, sz))
    return FrameObservation(label, np.array([centre[0], centre[1], sz / 2]), 0.8,
                            extent=np.sort(np.array([sx, sy, sz]))[::-1], top_z=sz,
                            color=colour, points=pts, source=source)


# ── measured premises (pass on main by design) ────────────────────────────


def test_premise_live_views_separate_by_diameter():
    """The live clouds: every view of the bin (both cameras, 10 clouds) vs the
    two live side views of the 5 x 5 x 8 cm prop (in the bin corner / next to
    it). Same object: within x1.25. Container vs prop: >= x2.4, >= 10 cm."""
    bins, props = [], []
    for p in RECEIPT["pairs"]:
        bins.append(_diameter_ref(CLOUDS[f"p{p['pair']}_belief"]))
        (bins if p["kind"] == "same" else props).append(_diameter_ref(CLOUDS[f"p{p['pair']}_obs"]))
    assert len(bins) == 10 and len(props) == 2
    assert 0.18 < min(bins) and max(bins) < 0.24
    assert 0.06 < min(props) and max(props) < 0.08
    assert max(bins) / min(bins) < 1.25
    assert min(bins) / max(props) > 2.4
    assert min(bins) - max(props) > 0.10


def test_premise_the_obb_extent_is_not_a_size():
    """Why the gate does not use `extent`: the OBB is a min/max box, and the
    live side view of the prop in the bin corner has an OBB of 32 cm (bleed far
    behind it) for a 6.8 cm robust diameter -- as large as the bin's OBB."""
    p4 = CLOUDS["p4_obs"].astype(float)
    _, e, _ = oriented_bbox(p4)
    assert float(np.max(e)) > 0.30
    assert _diameter_ref(p4) < 0.08


def test_premise_partial_views_look_smaller_and_bleed_onto_a_container_does_not():
    """Ray-cast, both cameras: hiding half of a view (left/right/top/bottom
    in the image) makes it at most 3 % LARGER (a percentile span of a subset
    can grow a little) and the bin's halves stay within x1.5 of its full
    view. The limit of the rule, pinned: with 1 px of mask bleed at
    320 x 180 (~4 px at the live 1280 x 720) the side camera's view of a prop
    INSIDE the bin takes in the bin's far wall and looks twice its size (live
    YOLOE views of that prop: 0.066-0.074 m, measured)."""
    import cv2

    props = {"bin": BIN, "prop": _prop(*LIVE_PROP_XY)}
    for cam in CAMS:
        frame, masks = _render(cam, props)
        full = _diameter_ref(_cloud(cam, frame, masks["bin"]))
        ys, xs = np.nonzero(masks["bin"])
        for half in (xs < np.median(xs), xs >= np.median(xs), ys < np.median(ys), ys >= np.median(ys)):
            m = np.zeros_like(masks["bin"])
            m[ys[half], xs[half]] = True
            d = _diameter_ref(_cloud(cam, frame, m))
            assert d < 1.05 * full and full / d < 1.5, (cam, full, d)
    side, sm = _render("isaac_side", props)
    clean = _diameter_ref(_cloud("isaac_side", side, sm["prop"]))
    bled = cv2.dilate(sm["prop"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert clean < 0.07 and _diameter_ref(_cloud("isaac_side", side, bled)) > 2.0 * clean


# ── the defect: a container view must not land in the prop's belief ───────


def _prop_belief_from_top(store):
    (top, tm), _ = _in_bin_scene()
    (prop,) = store.update_frame([_obs("isaac", top, tm["prop"], "yellow", "cube")], t=1.0)
    return prop


def test_live_side_view_of_the_bin_goes_to_the_bin_not_the_prop():
    """Live clouds: the bin as the top camera saw it in the in_bin_centre run
    (fixture pair 1, "orange"), the prop (ray-cast, where that run had it),
    then the side camera's live view of the bin, "yellow". The prop is nearer
    (2.4 vs 3.3 cm), so on main the bin view went INTO the prop."""
    store = BeliefStore()
    (bin_,) = store.update_frame([_cloud_obs(CLOUDS["p1_belief"], "orange", "isaac", "gift box")], t=0.5)
    prop = _prop_belief_from_top(store)
    pts_before, pos_before = prop.points.copy(), prop.position.copy()
    side = _cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "building block")
    assert np.linalg.norm(side.position - prop.position) < np.linalg.norm(side.position - bin_.position)
    (got,) = store.update_frame([side], t=1.5)
    assert got is bin_, (got.label, got.color)
    assert bin_.source_colors == {"isaac": "orange", "isaac_side": "yellow"}
    assert prop.observations == 1 and "building block" not in prop.aliases
    np.testing.assert_array_equal(prop.points, pts_before)
    np.testing.assert_array_equal(prop.position, pos_before)
    assert prop.diameter_m < 0.09 < 0.18 < bin_.diameter_m


def test_one_name_rule_arm_gives_the_bin_view_a_belief_of_its_own():
    """`per_camera_colour: false`: the orange bin cannot take a "yellow" view
    and the prop must not either, so the view is a (yellow) bin of its own --
    what the one-name rule did for the bare scene."""
    store = BeliefStore(per_camera_colour=False)
    store.update_frame([_cloud_obs(CLOUDS["p1_belief"], "orange", "isaac", "gift box")], t=0.5)
    prop = _prop_belief_from_top(store)
    (got,) = store.update_frame([_cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")], t=1.5)
    assert got is not prop and got.color == "yellow"
    assert len(store.all(now=2.0)) == 3 and prop.observations == 1


def test_the_first_sight_of_the_bin_does_not_land_in_the_prop():
    """No bin belief yet (only the prop seen): the side camera's live bin view
    is born as its own belief instead of becoming the prop."""
    store = BeliefStore()
    prop = _prop_belief_from_top(store)
    (got,) = store.update_frame([_cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")], t=1.5)
    assert got is not prop
    assert prop.observations == 1 and prop.label == "cube"


@pytest.mark.parametrize("pair", SAME)
def test_the_bins_live_views_still_fuse(pair):
    """The bin's two live views (the B32b fixture's hardest decisions) are one
    belief with the gate on: the same object across views is within x1.3."""
    store = BeliefStore()
    (first,) = store.update_frame([_cloud_obs(CLOUDS[f"p{pair}_belief"], "orange", "isaac", "bin")], t=1.0)
    (second,) = store.update_frame([_cloud_obs(CLOUDS[f"p{pair}_obs"], "yellow", "isaac_side", "bin")], t=1.5)
    assert second is first and len(store.all(now=2.0)) == 1


def test_the_per_detection_path_applies_the_gate_too():
    """`instance_association: false` (the B31 A/B baseline) fuses through
    `update()`: same gate."""
    store = BeliefStore(instance_association=False)
    store.update_frame([_cloud_obs(CLOUDS["p1_belief"], "orange", "isaac", "gift box")], t=0.5)
    prop = _prop_belief_from_top(store)
    (got,) = store.update_frame([_cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")], t=1.5)
    assert got is not prop and prop.observations == 1


def test_a_single_observation_with_a_cloud_is_gated_in_update():
    store = BeliefStore()
    prop = _prop_belief_from_top(store)
    side = _cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")
    got = store.update("box", side.position, 0.7, extent=side.extent, top_z=side.top_z, t=1.5,
                       color="yellow", points=side.points, source="isaac_side")
    assert got is not prop and prop.observations == 1


# ── through the real fusion paths ─────────────────────────────────────────


def _painted_in_bin_frames():
    (top, tm), (side, sm) = _in_bin_scene()
    _paint(top, tm["bin"], BIN_TOP_HSV)
    _paint(top, tm["prop"], YELLOW_HSV)
    _paint(side, sm["bin"], BIN_SIDE_HSV)
    _paint(side, sm["prop"], YELLOW_HSV)
    return (top, tm), (side, sm)


def test_watcher_keeps_the_side_cameras_bin_out_of_the_prop():
    """The REAL WorldWatcher path, colours named from the pixels. The side
    camera sees the prop on one frame of three (live: 2 of 5). RED on main:
    the prop belief takes the bin view (alias "building block", the bin's
    cloud) on the frames where the side camera sees only the bin."""
    store = BeliefStore()
    watcher, cams, det, current = _two_camera_watcher(store)
    (top, tm), (side, sm) = _painted_in_bin_frames()
    side_dets = [[_det("building block", sm["bin"])],
                 [_det("building block", sm["bin"]), _det("passbook", sm["prop"], 0.4)],
                 [_det("building block", sm["bin"])]]
    n = 0
    for rnd in range(3):
        for cam, frame, dets in ((cams[0], top, [_det("storage box", tm["bin"]), _det("cube", tm["prop"])]),
                                 (cams[1], side, side_dets[rnd])):
            n += 1
            frame.frame_id, frame.t = n, 100.0 + n
            det.by_frame[n] = dets
            current[cam.stream.name] = frame
            watcher._tick(cam)
    beliefs = store.all(now=100.0 + n)
    assert len(beliefs) == 2, [(b.label, b.color, b.aliases) for b in beliefs]
    prop = next(b for b in beliefs if "cube" in b.aliases | {b.label})
    bin_ = next(b for b in beliefs if b is not prop)
    assert prop.label == "cube" and "building block" not in prop.aliases, (prop.label, prop.aliases)
    assert prop.observations == 4  # 3 top views + the side camera's one view of it
    assert bin_.observations == 6 and bin_.source_colors == {"isaac": "orange", "isaac_side": "yellow"}
    assert prop.diameter_m < 0.09
    assert np.linalg.norm(prop.position[:2] - np.array(LIVE_PROP_XY)) < 0.02


def test_get_observation_and_reobserve_keep_the_bin_out_of_the_prop():
    """`_update_beliefs_from_frame` (get_observation) and `_reobserve` (every
    other fusing camera): the same gate. RED on main."""
    from cascade.skills.runtime import SkillRuntime

    store = BeliefStore()
    rt = _runtime(store)
    (top, tm), (side, sm) = _painted_in_bin_frames()
    top.t, side.t = 50.0, 50.2
    SkillRuntime._update_beliefs_from_frame(rt, top, [_det("storage box", tm["bin"]), _det("cube", tm["prop"])])
    side_cam = SimpleNamespace(stream=SimpleNamespace(name="isaac_side", get_frame=lambda: side),
                               depth=SimpleNamespace(ensure_depth=lambda f: f),
                               extrinsics=Extrinsics(T=CAMS["isaac_side"]), fuse=True)
    det = _PerFrameDetector()
    top.frame_id, side.frame_id = 1, 2
    det.by_frame = {1: [_det("storage box", tm["bin"]), _det("cube", tm["prop"])],
                    2: [_det("building block", sm["bin"])]}
    rt.held_object, rt._held_det_label, rt._default_classes = None, None, None
    rt.detector = det
    rt.observe = lambda: top
    rt._show_detections = lambda dets: None
    rt.watcher = SimpleNamespace(_cams=[SimpleNamespace(stream=SimpleNamespace(name="isaac")), side_cam])
    SkillRuntime._reobserve(rt, frames=1)
    beliefs = store.all(now=51.0)
    assert len(beliefs) == 2, [(b.label, b.color, b.aliases) for b in beliefs]
    prop = next(b for b in beliefs if "cube" in b.aliases | {b.label})
    bin_ = next(b for b in beliefs if b is not prop)
    assert prop.label == "cube" and "building block" not in prop.aliases, (prop.label, prop.aliases)
    assert prop.observations == 2
    assert bin_.source_colors == {"isaac": "orange", "isaac_side": "yellow"}


# ── what must still fuse ──────────────────────────────────────────────────


@pytest.mark.parametrize("part", ["half", "quarter"])
def test_an_occluded_smaller_view_still_fuses(part):
    """A view with half / three quarters of the object hidden (the arm, a
    neighbour) is smaller, never larger: it joins the full view's belief --
    for the bin and for the prop, both cameras."""
    props = {"bin": BIN, "prop": _prop(*LIVE_PROP_XY)}
    for cam in CAMS:
        frame, masks = _render(cam, props)
        for name in ("bin", "prop"):
            m = masks[name]
            ys, xs = np.nonzero(m)
            keep = xs < np.median(xs)
            if part == "quarter":
                keep = np.logical_and(keep, ys < np.median(ys))
            occluded = np.zeros_like(m)
            occluded[ys[keep], xs[keep]] = True
            if occluded.sum() < 10:
                continue
            store = BeliefStore()
            (full,) = store.update_frame([_obs(cam, frame, m, "yellow", name)], t=1.0)
            size = full.diameter_m
            (again,) = store.update_frame([_obs(cam, frame, occluded, "yellow", name)], t=1.5)
            assert again is full, (cam, name, part)
            assert full.diameter_m == size, "a smaller view never shrinks the belief's size"


@pytest.mark.parametrize("part", ["half", "quarter"])
def test_a_belief_born_from_part_of_a_view_takes_the_full_view(part):
    """The other order: the first sight was occluded, then the whole bin --
    one belief, and its size grows to the whole view. The quarter is the
    closest same-object case to the threshold in the ray-cast: the top
    camera's corner quarter of the bin is x1.94 smaller than its full view
    (half views: <= x1.48, both cameras, bin and prop)."""
    frame, masks = _render("isaac", {"bin": BIN, "prop": _prop(*LIVE_PROP_XY)})
    m = masks["bin"]
    ys, xs = np.nonzero(m)
    keep = xs < np.median(xs)
    if part == "quarter":
        keep = np.logical_and(keep, ys < np.median(ys))
    first = np.zeros_like(m)
    first[ys[keep], xs[keep]] = True
    store = BeliefStore()
    (b,) = store.update_frame([_obs("isaac", frame, first, "orange", "bin")], t=1.0)
    small = b.diameter_m
    whole = _obs("isaac", frame, m, "orange", "bin")
    full = bm._view_diameter(BeliefStore._remembered_cloud(whole.points))
    assert full / small > (1.9 if part == "quarter" else 1.1)
    assert store.update_frame([whole], t=1.5)[0] is b
    assert b.diameter_m == full


def test_identical_twins_are_unchanged_by_the_gate():
    """B31: identical red cubes 5 cm apart, both cameras: two beliefs at their
    own positions, exactly as with the gate off."""
    props = {"t1": _cube(0.24, 0.04), "t2": _cube(0.24, 0.09)}
    views = {}
    for cam in CAMS:
        frame, masks = _render(cam, props)
        views[cam] = [_obs(cam, frame, masks[n], "red", n) for n in ("t1", "t2")]
    dumps = []
    for gate in (True, False):
        store = BeliefStore(size_gate=gate)
        for i in range(3):
            store.update_frame(views["isaac"], t=1.0 + i)
            store.update_frame(views["isaac_side"], t=1.5 + i)
        beliefs = store.all(now=4.0)
        assert len(beliefs) == 2
        dumps.append([(b.label, b.position.tolist(), b.observations) for b in beliefs])
    assert dumps[0] == dumps[1]


# ── the rule ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scale, fused", [(1.9, True), (2.1, False)])
def test_the_ratio_threshold(scale, fused):
    """Square grid boxes on one centre: the view is `scale` times the belief's
    diameter (and > 5 cm larger in both cases): x1.9 fuses, x2.1 is refused."""
    store = BeliefStore()
    (b,) = store.update_frame([_grid_obs((0.30, 0.10), 0.06, 0.06)], t=1.0)
    obs = _grid_obs((0.30, 0.10), 0.06 * scale, 0.06 * scale)
    assert bm._view_diameter(obs.points) - b.diameter_m > bm.SIZE_GATE_EXCESS_M
    assert bm._view_diameter(obs.points) / b.diameter_m == pytest.approx(scale, rel=1e-6)
    assert (store.update_frame([obs], t=1.5)[0] is b) is fused


@pytest.mark.parametrize("side_m, fused", [(0.05, True), (0.08, False)])
def test_the_excess_threshold(side_m, fused):
    """A 2 cm object: a view 2.5x its diameter but only ~3.5 cm larger is
    mask/neighbour noise and fuses; 4x and ~7 cm larger is refused."""
    store = BeliefStore()
    (b,) = store.update_frame([_grid_obs((0.30, 0.10), 0.02, 0.02)], t=1.0)
    obs = _grid_obs((0.30, 0.10), side_m, side_m)
    assert bm._view_diameter(obs.points) > bm.SIZE_GATE_RATIO * b.diameter_m
    assert (store.update_frame([obs], t=1.5)[0] is b) is fused


def test_the_belief_remembers_its_largest_view():
    """Running max: a full view, then a corner sliver (fuses: smaller), then
    the full view again (fuses: the belief has been that big). Against the
    LAST view's size the full view would be x2.8 the sliver and refused."""
    store = BeliefStore()
    (b,) = store.update_frame([_grid_obs((0.30, 0.10), 0.15, 0.15)], t=1.0)
    full = b.diameter_m
    assert store.update_frame([_grid_obs((0.35, 0.15), 0.05, 0.05)], t=1.5)[0] is b
    assert b.diameter_m == full
    assert bm._view_diameter(b.points) < full / bm.SIZE_GATE_RATIO
    assert store.update_frame([_grid_obs((0.30, 0.10), 0.15, 0.15)], t=2.0)[0] is b


def test_a_larger_accepted_view_raises_the_belief_size():
    """Born from a 10 cm view, then a 15 cm view (x1.5, fuses): the belief is
    now 15 cm, so a 28 cm view (x2.8 of the first, x1.9 of the second) fuses."""
    store = BeliefStore()
    (b,) = store.update_frame([_grid_obs((0.30, 0.10), 0.10, 0.10)], t=1.0)
    assert store.update_frame([_grid_obs((0.30, 0.10), 0.15, 0.15)], t=1.5)[0] is b
    assert store.update_frame([_grid_obs((0.30, 0.10), 0.28, 0.28)], t=2.0)[0] is b


@pytest.mark.parametrize("born_by", ["update_frame", "update"])
def test_a_view_too_small_to_measure_keeps_the_belief_size(born_by):
    """A view with a cloud too small to size (< 10 points) fuses without a
    veto and replaces the remembered cloud, but not the size the belief got
    when it was born (either writer): the bin view is still refused."""
    store = BeliefStore()
    (top, tm), _ = _in_bin_scene()
    o = _obs("isaac", top, tm["prop"], "yellow", "cube")
    if born_by == "update_frame":
        (prop,) = store.update_frame([o], t=1.0)
    else:
        prop = store.update(o.label, o.position, o.conf, extent=o.extent, top_z=o.top_z, t=1.0,
                            color=o.color, points=o.points, source=o.source)
    size = prop.diameter_m
    assert size is not None and size < 0.09
    tiny = prop.points[:5].copy()
    assert store.update("cube", prop.position.copy(), 0.8, color="yellow", points=tiny,
                        source="isaac", t=1.2) is prop
    assert prop.points.shape[0] == 5 and prop.diameter_m == size
    (got,) = store.update_frame([_cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")], t=1.5)
    assert got is not prop


def test_no_cloud_no_veto():
    """Missing evidence keeps the old behaviour: a bbox-only view (no
    real-mask cloud) or a belief without a cloud is never refused by size."""
    store = BeliefStore()
    prop = _prop_belief_from_top(store)
    side = _cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")
    side.points = None
    assert store.update_frame([side], t=1.5)[0] is prop

    store = BeliefStore()
    blind = store.update("cube", prop.position.copy(), 0.9, color="yellow", t=1.0)
    assert blind.points is None and blind.diameter_m is None
    side = _cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")
    assert store.update_frame([side], t=1.5)[0] is blind


def test_the_gate_only_refuses():
    """A same-size red view at a blue belief stays apart: size consistency
    never overrides the colour rule (or the radius)."""
    store = BeliefStore()
    (b,) = store.update_frame([_grid_obs((0.30, 0.10), 0.05, 0.05, colour="blue")], t=1.0)
    assert store.update_frame([_grid_obs((0.30, 0.10), 0.05, 0.05, colour="red")], t=1.5)[0] is not b
    assert store.update_frame([_grid_obs((0.50, 0.10), 0.05, 0.05, colour="blue")], t=2.0)[0] is not b


def test_the_diameter_is_the_spec_and_ignores_yaw():
    """`_view_diameter` is `_diameter_ref` on the live clouds; a 15 cm square
    turned 0 / 22.5 / 45 deg keeps its diameter within 4 % while its robust
    axis-aligned box grows by 20 % (an axis-aligned size would read a turned
    bin as a bigger object)."""
    for k in CLOUDS.files:
        assert bm._view_diameter(CLOUDS[k]) == pytest.approx(_diameter_ref(CLOUDS[k]), abs=1e-12)
    sq = _grid_box((0.0, 0.0, 0.03), (0.15, 0.15, 0.06), n=(21, 21, 5))
    ds, spans = [], []
    for deg in (0.0, 22.5, 45.0):
        t = np.deg2rad(deg)
        R = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
        p = sq @ R.T + np.array([0.30, 0.10, 0.0])
        ds.append(bm._view_diameter(p))
        lo, hi = bm._cloud_box(p)
        spans.append(float(np.max((hi - lo)[:2])))
    assert max(ds) / min(ds) < 1.04
    assert max(spans) / min(spans) > 1.15


def test_the_diameter_ignores_a_table_skirt():
    """A 5 cm prop whose mask bled 6 cm onto the table behind it: the floor
    band (B32c) drops the skirt, so the view keeps the prop's size and fuses."""
    rng = np.random.default_rng(3)
    body = np.column_stack([rng.uniform(0.275, 0.325, 400), rng.uniform(0.075, 0.125, 400),
                            rng.uniform(0.0, 0.08, 400)])
    skirt = np.column_stack([rng.uniform(0.325, 0.385, 120), rng.uniform(0.075, 0.125, 120),
                             rng.uniform(0.0, 0.002, 120)])
    assert bm._view_diameter(np.vstack([body, skirt])) == pytest.approx(bm._view_diameter(body), rel=0.08)
    store = BeliefStore()
    (b,) = store.update_frame([_cloud_obs(body, "yellow", "isaac", "cube")], t=1.0)
    assert store.update_frame([_cloud_obs(np.vstack([body, skirt]), "yellow", "isaac_side", "cube")],
                              t=1.5)[0] is b


def test_the_size_is_not_persisted_and_falls_back_to_the_cloud(tmp_path):
    """`diameter_m` is session state: `save()` writes nothing new (the file
    format is unchanged), and a restored belief is measured from its restored
    cloud, so the gate still refuses the bin view after a restart."""
    store = BeliefStore()
    prop = _prop_belief_from_top(store)
    path = tmp_path / "beliefs.json"
    store.save(path)
    assert "diameter" not in path.read_text()
    again = BeliefStore()
    # no age limit: the belief was seen at monotonic t=1.0 (seconds since
    # boot), so the default 6 h limit would drop it on a host up longer
    assert again.load(path, max_age_s=float("inf")) == 1
    (restored,) = again.all()
    assert restored.diameter_m is None and restored.points is not None
    got = again.update_frame([_cloud_obs(CLOUDS["p1_obs"], "yellow", "isaac_side", "box")], t=1e9)
    assert got[0] is not restored and restored.observations == prop.observations


# ── the switch ────────────────────────────────────────────────────────────


def test_golden_inputs_are_the_ones_the_receipt_describes():
    receipt = json.loads(golden.RECEIPT.read_text())
    assert hashlib.sha256(golden.INPUTS.read_bytes()).hexdigest() == receipt["inputs_npz_sha256"]
    assert hashlib.sha256(golden.GOLDEN.read_bytes()).hexdigest() == receipt["golden_json_sha256"]
    assert "4e896c3" in receipt["store_source"]


def test_switch_off_is_byte_identical_to_the_pre_b40_store():
    """`size_gate=False` replays the golden scenario (the defect, the bin's
    live views, twins, update() with and without a cloud; default store,
    one-name rule, per-detection path) exactly as the store of main 4e896c3
    did: same assignments, same beliefs, every float identical."""
    arrays = golden.load_inputs()
    doc = golden.dump(lambda **kw: BeliefStore(size_gate=False, **kw), arrays)
    assert golden.as_text(doc) == golden.GOLDEN.read_text()
    for name, kw in golden.CONFIGS.items():
        store = BeliefStore(size_gate=False, **kw)
        golden.replay(store, arrays)
        assert all(b.diameter_m is None for b in store.all()), name


def test_switch_on_changes_only_where_a_bin_view_met_the_prop():
    """With the gate on, the golden scenario differs exactly where the side
    camera's bin view used to land in the prop belief (belief 1); the twins'
    beliefs are byte-identical to the golden."""
    arrays = golden.load_inputs()
    old = json.loads(golden.GOLDEN.read_text())
    new = json.loads(golden.as_text(golden.dump(BeliefStore, arrays)))
    for name in golden.CONFIGS:
        assert old[name]["steps"][1] == [1], "premise: the defect is in the golden"
        assert new[name]["steps"][1] != [1], name
        props = [b for b in new[name]["beliefs"] if b["label"] == "cube" and b["color"] == "yellow"]
        assert len(props) == 1, (name, [(b["label"], b["color"]) for b in new[name]["beliefs"]])
        assert not {"box", "building block", "gift box", "storage box"} & set(props[0]["aliases"]), name
        twins_old = [b for b in old[name]["beliefs"] if b["color"] == "red"]
        twins_new = [b for b in new[name]["beliefs"] if b["color"] == "red"]
        assert twins_new == twins_old, name


def test_demo_config_turns_the_gate_on_and_the_switch_parses():
    from cascade.apps.demo import _belief_store

    mem = yaml.safe_load((REPO / "configs" / "demo.yaml").read_text())["memory"]
    assert mem["size_gate"] is True
    assert _belief_store(mem)._size_gate is True
    assert _belief_store({})._size_gate is True
    assert _belief_store({"size_gate": "false"})._size_gate is False
    assert _belief_store({"size_gate": "off"})._size_gate is False
    assert _belief_store({"size_gate": False})._size_gate is False
    assert _belief_store({"size_gate": "true"})._size_gate is True
