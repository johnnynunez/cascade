"""B32b: one object seen by two cameras under neighbouring colour names is ONE
belief; genuinely different objects stay apart.

Measured on Isaac 6.2 PhysX + YOLOE (bare scene, 16 rounds of per-camera median
HSV on robot-free pixels, backlog B32 report): the open bin's hue is H 22 in the
top camera `isaac` -> "orange" (16/16 frames) and H 23 in `isaac_side` ->
"yellow" (7/7). `classify_hsv` puts H <= 22 at orange and 23-33 at yellow, and
the 2026-09-10 colour rule (two different measured names are two objects)
kept the two views apart: 4 beliefs for 3 props. A hue margin cannot fix it:
the green cube's hue differs by 3 units between the cameras, and a yellow prop
next to the bin would differ from it by 3-4. Each camera's names were 100 %
stable.

Mechanism (memory/beliefs.py): a belief keeps its colour name PER SOURCE
CAMERA. An observation from camera k is held to the belief's camera-k name
when camera k has named it (the 2026-09-10 rule, per camera). When camera k
never named it, a perceptual-neighbour name (colors._NEIGHBORS: orange~yellow)
may fuse only with strong 3D overlap: IoU of the axis-aligned boxes of the two
robot-free clouds (2nd-98th percentile per axis) >= `neighbour_colour_iou`
(0.75). Intersection over UNION, not over the smaller box, so a prop INSIDE the
bin stays separate. Non-neighbour names never fuse.

Geometry is the bare Isaac scene ray-cast on CPU: the bridge's open bin
(15 x 15 x 6 cm, 1 cm walls, centred at (0.18, -0.17), scripts/isaac_bridge.py),
the two calibrated camera poses read from configs/cameras/isaac*.yaml and the
bridge optics (18 mm focal length, 20.955 mm aperture) at 320 x 180. Masks are
exact (ray hits, no detector): these are the geometric premises; the live A/B
on Isaac checks them with YOLOE masks.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from conftest import REPO

from cascade.memory.beliefs import BeliefStore, FrameObservation
from cascade.perception.colors import detection_color
from cascade.perception.grounding import Extrinsics, mask_to_points_cam, oriented_bbox
from cascade.perception.workspace import WorkspaceFilter
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.types import Detection, Frame, transform_points

W, H = 320, 180
FX = 18.0 / 20.955 * W  # scripts/isaac_bridge.py `_camera`: focal 18 mm / aperture 20.955 mm
K = np.array([[FX, 0.0, W / 2], [0.0, FX, H / 2], [0.0, 0.0, 1.0]])


def _cam_T(name: str) -> np.ndarray:
    cfg = yaml.safe_load((REPO / "configs" / "cameras" / f"{name}.yaml").read_text())
    return np.asarray(cfg["extrinsics"]["T"], dtype=float)


CAMS = {"isaac": _cam_T("isaac"), "isaac_side": _cam_T("isaac_side")}
#: scripts/isaac_bridge.py bare scene: four static walls, the table is the floor
BIN = [((0.25, -0.17, 0.03), (0.01, 0.15, 0.06)), ((0.11, -0.17, 0.03), (0.01, 0.15, 0.06)),
       ((0.18, -0.10, 0.03), (0.15, 0.01, 0.06)), ((0.18, -0.24, 0.03), (0.15, 0.01, 0.06))]
_BIN_FLOOR = ((0.115, 0.245), (-0.235, -0.105))  # the table inside the walls is part of its mask
_TABLE = ((0.30, 0.0, -0.015), (0.9, 0.9, 0.03))

#: measured median OpenCV HSV (B32 report, `b32b_hsv_diag.py`)
BIN_TOP_HSV = (22, 97, 207)    # -> orange in `isaac`
BIN_SIDE_HSV = (23, 81, 217)   # -> yellow in `isaac_side`
YELLOW_HSV = (27, 150, 210)
ORANGE_HSV = (15, 160, 210)
RED_HSV = (2, 200, 200)
BLUE_HSV = (115, 200, 200)


def _cube(x: float, y: float, s: float = 0.035):
    return [((x, y, s / 2), (s, s, s))]


def _render(cam: str, props: dict) -> tuple[Frame, dict]:
    """Ray-cast RGB-D of axis-aligned boxes on the table from camera `cam`.

    `props` maps a name to its boxes [(centre, size), ...]; returns the frame
    and one exact mask per name (the bin's includes the floor it encloses)."""
    T = CAMS[cam]
    boxes = [("table", _TABLE)] + [(n, b) for n, bs in props.items() for b in bs]
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(us - W / 2) / FX, (vs - H / 2) / FX, np.ones_like(us)], -1).reshape(-1, 3)
    d = rays @ T[:3, :3].T
    o = T[:3, 3]
    best = np.full(d.shape[0], np.inf)
    owner = np.full(d.shape[0], "", dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, (c, s) in boxes:
            lo = np.subtract(c, np.divide(s, 2))
            hi = np.add(c, np.divide(s, 2))
            t1, t2 = (lo - o) / d, (hi - o) / d
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (near <= far) & (far > 0) & (near < best)
            best[hit] = near[hit]
            owner[hit] = name
    t = np.where(np.isfinite(best), best, 0.0)
    depth = t.reshape(H, W).astype(np.float32)  # rays have z_cam = 1: t IS the z-depth
    owner = owner.reshape(H, W)
    pts = (o + t[:, None] * d).reshape(H, W, 3)
    masks = {n: owner == n for n in props}
    if "bin" in masks:
        (x0, x1), (y0, y1) = _BIN_FLOOR
        masks["bin"] = masks["bin"] | ((owner == "table") & (pts[..., 0] > x0) & (pts[..., 0] < x1)
                                       & (pts[..., 1] > y0) & (pts[..., 1] < y1))
    frame = Frame(rgb=np.full((H, W, 3), 128, np.uint8), depth_m=depth, K=K.copy(),
                  depth_source="sensor")
    return frame, masks


def _paint(frame: Frame, mask, hsv) -> None:
    bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame.rgb[mask] = bgr


def _cloud(cam: str, frame: Frame, mask) -> np.ndarray:
    return transform_points(CAMS[cam], mask_to_points_cam(frame, mask))


def _obs(cam: str, frame: Frame, mask, colour: str | None, label: str = "object",
         *, points: bool = True) -> FrameObservation:
    """One observation lifted exactly like the watcher lifts it."""
    pts = _cloud(cam, frame, mask)
    c, e, _ = oriented_bbox(pts)
    return FrameObservation(label, c, 0.8, extent=e, top_z=float(pts[:, 2].max()),
                            color=colour, points=pts if points else None, source=cam)


def _box(p, q: float = 2.0):
    return np.percentile(p, q, axis=0), np.percentile(p, 100 - q, axis=0)


def _iou(a, b) -> float:
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    return inter / (float(np.prod(a[1] - a[0])) + float(np.prod(b[1] - b[0])) - inter)


def _views(props: dict, colours: dict[str, dict[str, str | None]]):
    """{cam: [FrameObservation per prop]} for every prop each camera sees."""
    out = {}
    for cam in CAMS:
        frame, masks = _render(cam, props)
        out[cam] = [_obs(cam, frame, m, colours[cam][n], n) for n, m in masks.items()
                    if m.sum() >= 10 and n in colours[cam]]
    return out


# ── the geometric premise ─────────────────────────────────────────────────


def test_premise_the_bins_two_views_overlap_strongly_and_a_prop_inside_does_not():
    """Pins what the gate relies on, independent of the store: the two partial
    views of the bin overlap far above 0.75 IoU, a cube inside it overlaps it
    by ~0.02 IoU although it is entirely INSIDE (intersection over the smaller
    box = 1.0, which is why that measure would merge it)."""
    props = {"bin": BIN, "cube": _cube(0.145, -0.205)}
    cl = {}
    for cam in CAMS:
        frame, masks = _render(cam, props)
        assert masks["cube"].sum() >= 10, f"the cube must be visible from {cam}"
        cl[cam] = {n: _cloud(cam, frame, m) for n, m in masks.items()}
    assert _iou(_box(cl["isaac"]["bin"]), _box(cl["isaac_side"]["bin"])) > 0.85
    for cam in CAMS:
        cube, bin_ = _box(cl[cam]["cube"]), _box(cl["isaac"]["bin"])
        assert _iou(cube, bin_) < 0.05
        assert np.all(cube[0] >= bin_[0] - 0.005) and np.all(cube[1] <= bin_[1] + 0.005)


# ── the bin across two cameras: one belief (RED on main: two) ─────────────


def _bin_frames():
    props = {"bin": BIN}
    top, top_masks = _render("isaac", props)
    side, side_masks = _render("isaac_side", props)
    _paint(top, top_masks["bin"], BIN_TOP_HSV)
    _paint(side, side_masks["bin"], BIN_SIDE_HSV)
    return (top, top_masks["bin"]), (side, side_masks["bin"])


def _det(label, mask, conf=0.8):
    ys, xs = np.nonzero(mask)
    return Detection(label, conf, np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                                           np.float32), mask=mask)


def test_the_bins_measured_colours_name_it_orange_and_yellow():
    """Premise: painted with the measured medians, the pipeline's own colour
    namer reproduces the live names (one hue unit across a band boundary)."""
    (top, mt), (side, ms) = _bin_frames()
    assert detection_color(top.rgb, _det("storage box", mt)) == "orange"
    assert detection_color(side.rgb, _det("building block", ms)) == "yellow"


def test_the_bin_seen_by_two_cameras_is_one_belief():
    store = BeliefStore()
    (top, mt), (side, ms) = _bin_frames()
    for i in range(3):
        store.update_frame([_obs("isaac", top, mt, "orange", "storage box")], t=10.0 + i)
        store.update_frame([_obs("isaac_side", side, ms, "yellow", "building block")], t=10.5 + i)
    beliefs = store.all(now=13.0)
    assert len(beliefs) == 1, [(b.label, b.color) for b in beliefs]
    (b,) = beliefs
    assert b.source_colors == {"isaac": "orange", "isaac_side": "yellow"}
    assert b.observations == 6
    assert b.color == "orange", "the belief's colour is its first measured name and stays"
    assert np.linalg.norm(b.position[:2] - np.array([0.18, -0.17])) < 0.03


class _PerFrameDetector:
    def __init__(self):
        self.by_frame: dict[int, list[Detection]] = {}

    def detect(self, frame, classes=None):
        return list(self.by_frame.get(frame.frame_id, []))


def _two_camera_watcher(store):
    det = _PerFrameDetector()
    current: dict[str, Frame | None] = {"isaac": None, "isaac_side": None}
    cams = []
    for name in CAMS:
        stream = SimpleNamespace(name=name, latest=lambda n=name: current[n],
                                 set_overlay=lambda **kw: None)
        cams.append(WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f),
                                  Extrinsics(T=CAMS[name])))
    return WorldWatcher(cams, det, store), cams, det, current


def test_watcher_fuses_the_bin_across_its_two_cameras():
    """The REAL WorldWatcher path: two cameras, colour named from the pixels.
    RED on main: two beliefs (an orange bin and a yellow bin)."""
    store = BeliefStore()
    watcher, cams, det, current = _two_camera_watcher(store)
    n = 0
    for rnd in range(3):
        for cam, (frame, mask), label in zip(cams, _bin_frames(), ("storage box", "building block")):
            n += 1
            frame.frame_id, frame.t = n, 100.0 + n
            det.by_frame[n] = [_det(label, mask)]
            current[cam.stream.name] = frame
            watcher._tick(cam)
    beliefs = store.all(now=100.0 + n)
    assert len(beliefs) == 1, [(b.label, b.color, np.round(b.position, 3).tolist()) for b in beliefs]
    assert beliefs[0].source_colors == {"isaac": "orange", "isaac_side": "yellow"}


def _runtime(store):
    from cascade.skills.runtime import SkillRuntime

    rt = object.__new__(SkillRuntime)
    rt.beliefs = store
    rt._workspace = WorkspaceFilter()
    rt.extrinsics = Extrinsics(T=CAMS["isaac"])
    rt.camera = SimpleNamespace(name="isaac")
    return rt


def test_get_observation_path_names_the_camera_of_every_frame():
    """`_update_beliefs_from_frame` (get_observation / describe) is fed by the
    primary camera, `_reobserve` also by every other fusing camera; both must
    say which camera the frame came from. RED on main: two beliefs."""
    from cascade.skills.runtime import SkillRuntime

    store = BeliefStore()
    rt = _runtime(store)
    (top, mt), (side, ms) = _bin_frames()
    top.t, side.t = 50.0, 50.2
    SkillRuntime._update_beliefs_from_frame(rt, top, [_det("storage box", mt)])
    side_cam = SimpleNamespace(stream=SimpleNamespace(name="isaac_side", get_frame=lambda: side),
                               depth=SimpleNamespace(ensure_depth=lambda f: f),
                               extrinsics=Extrinsics(T=CAMS["isaac_side"]), fuse=True)
    det = _PerFrameDetector()
    top.frame_id, side.frame_id = 1, 2
    det.by_frame = {1: [_det("storage box", mt)], 2: [_det("building block", ms)]}
    rt.held_object, rt._held_det_label, rt._default_classes = None, None, None
    rt.detector = det
    rt.observe = lambda: top
    rt._show_detections = lambda dets: None
    rt.watcher = SimpleNamespace(_cams=[SimpleNamespace(stream=SimpleNamespace(name="isaac")),
                                        side_cam])
    SkillRuntime._reobserve(rt, frames=1)
    beliefs = store.all(now=51.0)
    assert len(beliefs) == 1, [(b.label, b.color) for b in beliefs]
    assert beliefs[0].source_colors == {"isaac": "orange", "isaac_side": "yellow"}


# ── what must stay apart ──────────────────────────────────────────────────


def test_a_yellow_prop_inside_the_orange_bin_stays_two():
    """Both cameras see bin and cube: per-camera names keep them apart."""
    store = BeliefStore()
    props = {"bin": BIN, "cube": _cube(0.145, -0.205)}
    views = _views(props, {"isaac": {"bin": "orange", "cube": "yellow"},
                           "isaac_side": {"bin": "yellow", "cube": "yellow"}})
    for i in range(3):
        store.update_frame(views["isaac"], t=1.0 + i)
        store.update_frame(views["isaac_side"], t=1.5 + i)
    beliefs = store.all(now=4.0)
    assert len(beliefs) == 2, [(b.label, b.color) for b in beliefs]
    by = {b.label: b for b in beliefs}
    assert by["bin"].source_colors == {"isaac": "orange", "isaac_side": "yellow"}
    assert by["cube"].source_colors == {"isaac": "yellow", "isaac_side": "yellow"}
    assert np.linalg.norm(by["cube"].position[:2] - np.array([0.145, -0.205])) < 0.02


def test_a_camera_that_never_named_the_bin_cannot_merge_a_prop_inside_it():
    """The side camera's FIRST sight of a yellow cube inside the bin (the top
    camera has named only the bin): the names are neighbours and the cube is
    entirely inside the bin's box, but IoU ~0.02 << 0.75 -> a new belief.
    Intersection over the smaller box would be 1.0 here and merge them."""
    store = BeliefStore()
    top, tm = _render("isaac", {"bin": BIN})
    store.update_frame([_obs("isaac", top, tm["bin"], "orange", "bin")], t=1.0)
    side, sm = _render("isaac_side", {"bin": BIN, "cube": _cube(0.145, -0.205)})
    (cube_belief,) = store.update_frame([_obs("isaac_side", side, sm["cube"], "yellow", "cube")], t=1.5)
    beliefs = store.all(now=2.0)
    assert len(beliefs) == 2
    assert cube_belief.label == "cube" and cube_belief.source_colors == {"isaac_side": "yellow"}


def test_an_orange_cube_next_to_a_yellow_cube_stays_two():
    """4 cm apart, inside the 8 cm gate. The side camera's first view of the
    yellow cube is a neighbour name with ~zero overlap -> its own belief; then
    both cameras see both and each belief keeps its own names."""
    store = BeliefStore()
    props = {"A": _cube(0.26, 0.02), "B": _cube(0.26, 0.06)}
    top, tm = _render("isaac", {"A": props["A"]})
    store.update_frame([_obs("isaac", top, tm["A"], "orange", "A")], t=1.0)
    side, sm = _render("isaac_side", {"B": props["B"]})
    store.update_frame([_obs("isaac_side", side, sm["B"], "yellow", "B")], t=1.5)
    assert len(store.all(now=2.0)) == 2
    views = _views(props, {"isaac": {"A": "orange", "B": "yellow"},
                           "isaac_side": {"A": "orange", "B": "yellow"}})
    for i in range(3):
        store.update_frame(views["isaac"], t=2.0 + i)
        store.update_frame(views["isaac_side"], t=2.5 + i)
    beliefs = store.all(now=5.0)
    assert len(beliefs) == 2
    by = {b.label: b for b in beliefs}
    assert by["A"].color == "orange" and by["B"].color == "yellow"
    assert set(by["A"].source_colors.values()) == {"orange"}
    assert set(by["B"].source_colors.values()) == {"yellow"}


def test_red_and_blue_cubes_3cm_apart_stay_two_even_at_full_overlap():
    """The 2026-09-10 regression stays fixed: red and blue are not neighbours.
    Adversarial half: a "blue" observation from a camera that never named the
    red cube, carrying the red belief's own cloud (IoU 1.0) -> still two."""
    store = BeliefStore()
    props = {"red": _cube(0.26, 0.0), "blue": _cube(0.26, 0.065)}
    views = _views(props, {"isaac": {"red": "red", "blue": "blue"},
                           "isaac_side": {"red": "red", "blue": "blue"}})
    for i in range(2):
        store.update_frame(views["isaac"], t=1.0 + i)
        store.update_frame(views["isaac_side"], t=1.5 + i)
    assert sorted(b.color for b in store.all(now=3.0)) == ["blue", "red"]

    store = BeliefStore()
    top, tm = _render("isaac", {"red": props["red"]})
    red = _obs("isaac", top, tm["red"], "red", "cube")
    (belief,) = store.update_frame([red], t=1.0)
    clone = FrameObservation("cube", red.position.copy(), 0.9, extent=red.extent,
                             top_z=red.top_z, color="blue", points=red.points.copy(),
                             source="isaac_side")
    (other,) = store.update_frame([clone], t=1.5)
    assert other is not belief and len(store.all(now=2.0)) == 2


def test_a_camera_is_held_to_its_own_name():
    """Defined behaviour for one camera flipping a name between frames: a
    camera that has named a belief is held to that name, so a different name
    from the SAME camera is a different object -- exactly the 2026-09-10 rule,
    applied per camera, even at full overlap (the measured names were 100 %
    stable per camera; a flip is evidence the object changed). The bin's side
    view flipping yellow -> orange after fusion is held to "yellow" too."""
    store = BeliefStore()
    (top, mt), (side, ms) = _bin_frames()
    (first,) = store.update_frame([_obs("isaac", top, mt, "orange", "bin")], t=1.0)
    (flip,) = store.update_frame([_obs("isaac", top, mt, "yellow", "bin")], t=1.5)
    assert flip is not first
    assert first.source_colors == {"isaac": "orange"} and first.color == "orange"
    assert flip.source_colors == {"isaac": "yellow"}

    store = BeliefStore()
    store.update_frame([_obs("isaac", top, mt, "orange", "bin")], t=1.0)
    (fused,) = store.update_frame([_obs("isaac_side", side, ms, "yellow", "bin")], t=1.5)
    (again,) = store.update_frame([_obs("isaac_side", side, ms, "orange", "bin")], t=2.0)
    assert again is not fused
    assert fused.source_colors == {"isaac": "orange", "isaac_side": "yellow"}


# ── the overlap gate itself ───────────────────────────────────────────────


def _grid_box(centre, size, n=(9, 9, 5)) -> np.ndarray:
    axes = [np.linspace(c - s / 2, c + s / 2, k) for c, s, k in zip(centre, size, n)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _box_obs(centre, size, colour, source, label="box"):
    pts = _grid_box(centre, size)
    return FrameObservation(label, np.asarray(centre, float), 0.8, extent=np.sort(size)[::-1],
                            top_z=float(pts[:, 2].max()), color=colour, points=pts,
                            source=source)


@pytest.mark.parametrize("shift, fused", [(0.010, True), (0.035, False)])
def test_a_neighbour_name_from_a_new_camera_needs_strong_overlap(shift, fused):
    """Two 15 x 15 x 6 cm boxes shifted along x: 1.0 cm -> IoU 0.875 (one
    belief), 3.5 cm -> IoU 0.62 (two), both inside the 8 cm gate."""
    size = np.array([0.15, 0.15, 0.06])
    a = _box_obs((0.18, -0.17, 0.03), size, "orange", "isaac")
    b = _box_obs((0.18 + shift, -0.17, 0.03), size, "yellow", "isaac_side")
    store = BeliefStore()
    (first,) = store.update_frame([a], t=1.0)
    (second,) = store.update_frame([b], t=1.5)
    assert (second is first) is fused
    assert len(store.all(now=2.0)) == (1 if fused else 2)


def test_the_overlap_threshold_is_the_store_parameter():
    size = np.array([0.15, 0.15, 0.06])
    a = _box_obs((0.18, -0.17, 0.03), size, "orange", "isaac")
    b = _box_obs((0.215, -0.17, 0.03), size, "yellow", "isaac_side")
    store = BeliefStore(neighbour_colour_iou=0.5)
    (first,) = store.update_frame([a], t=1.0)
    assert store.update_frame([b], t=1.5)[0] is first
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            BeliefStore(neighbour_colour_iou=bad)


def test_a_small_box_wholly_inside_a_neighbour_named_one_stays_apart():
    store = BeliefStore()
    store.update_frame([_box_obs((0.18, -0.17, 0.03), np.array([0.15, 0.15, 0.06]),
                                 "orange", "isaac")], t=1.0)
    store.update_frame([_box_obs((0.18, -0.17, 0.02), np.array([0.05, 0.05, 0.04]),
                                 "yellow", "isaac_side")], t=1.5)
    assert len(store.all(now=2.0)) == 2


def test_mask_bleed_behind_the_bin_does_not_split_it():
    """A mask one pixel too wide (here = 4 px at the live 1280 x 720) lifts
    table points up to ~20 cm behind the bin's far rim. A min/max box takes
    them at face value (IoU 0.34); the 2nd-98th percentile box keeps 0.90."""
    store = BeliefStore()
    for t, (cam, colour) in enumerate((("isaac", "orange"), ("isaac_side", "yellow"))):
        frame, masks = _render(cam, {"bin": BIN})
        bled = cv2.dilate(masks["bin"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        store.update_frame([_obs(cam, frame, bled, colour, "bin")], t=1.0 + t)
    assert len(store.all(now=3.0)) == 1


def test_no_cloud_no_neighbour_fusion():
    """Missing evidence stays apart: a bbox-only detection (no real-mask
    cloud) or a belief without one never takes the neighbour exception."""
    (top, mt), (side, ms) = _bin_frames()
    store = BeliefStore()
    store.update_frame([_obs("isaac", top, mt, "orange")], t=1.0)
    store.update_frame([_obs("isaac_side", side, ms, "yellow", points=False)], t=1.5)
    assert len(store.all(now=2.0)) == 2

    store = BeliefStore()
    store.update_frame([_obs("isaac", top, mt, "orange", points=False)], t=1.0)
    store.update_frame([_obs("isaac_side", side, ms, "yellow")], t=1.5)
    assert len(store.all(now=2.0)) == 2


def test_an_observation_without_a_camera_keeps_the_name_rule():
    """`update()` callers (place_at, push, localize) and frames without a
    source name compare with the belief's names exactly, as before."""
    (top, mt), (side, ms) = _bin_frames()
    store = BeliefStore()
    (b,) = store.update_frame([_obs("isaac", top, mt, "orange")], t=1.0)
    side_obs = _obs("isaac_side", side, ms, "yellow")
    other = store.update("bin", side_obs.position, 0.8, extent=side_obs.extent,
                         color="yellow", points=side_obs.points, t=1.5)
    assert other is not b
    side_obs.source = None
    assert store.update_frame([side_obs], t=2.0)[0] is not b
    assert store.update("bin", b.position, 0.8, color="orange", t=2.5) is b

    # A belief no camera has named (born from an unsourced writer) gives the
    # neighbour exception nothing to stand on: a camera's neighbour name
    # stays apart even at full overlap.
    store = BeliefStore()
    top_obs = _obs("isaac", top, mt, "orange")
    unsourced = store.update("bin", top_obs.position, 0.8, extent=top_obs.extent,
                             color="orange", points=top_obs.points, t=1.0)
    assert unsourced.source_colors == {}
    top_obs.color = "yellow"
    assert store.update_frame([top_obs], t=1.5)[0] is not unsourced


def test_same_colour_twins_seen_by_two_cameras_stay_two():
    """B31 unaffected: identical red cubes 5 cm apart, both cameras."""
    store = BeliefStore()
    props = {"t1": _cube(0.24, 0.04), "t2": _cube(0.24, 0.09)}
    views = _views(props, {cam: {"t1": "red", "t2": "red"} for cam in CAMS})
    for i in range(3):
        store.update_frame(views["isaac"], t=1.0 + i)
        store.update_frame(views["isaac_side"], t=1.5 + i)
    beliefs = store.all(now=4.0)
    assert len(beliefs) == 2
    for b, y in zip(sorted(beliefs, key=lambda b: b.position[1]), (0.04, 0.09)):
        assert abs(b.position[1] - y) < 0.015
        assert b.source_colors == {"isaac": "red", "isaac_side": "red"}


def test_the_legacy_switch_restores_one_name_per_belief():
    """`per_camera_colour=False` (memory.per_camera_colour: false) is the
    pre-B32b store for the live A/B: the bin is two beliefs again."""
    store = BeliefStore(per_camera_colour=False)
    (top, mt), (side, ms) = _bin_frames()
    store.update_frame([_obs("isaac", top, mt, "orange")], t=1.0)
    store.update_frame([_obs("isaac_side", side, ms, "yellow")], t=1.5)
    assert sorted(b.color for b in store.all(now=2.0)) == ["orange", "yellow"]


# ── what the rest of the system reads ─────────────────────────────────────


def test_find_resolves_another_cameras_exact_name_before_neighbours():
    """A fused belief whose first name is yellow (the side camera saw it
    first) is still the exact answer to "orange object" -- the top camera
    names it orange -- ahead of a red cube that is only a neighbour."""
    (top, mt), (side, ms) = _bin_frames()
    store = BeliefStore()
    (bin_,) = store.update_frame([_obs("isaac_side", side, ms, "yellow", "bin")], t=1.0)
    store.update_frame([_obs("isaac", top, mt, "orange", "bin")], t=1.5)
    assert bin_.color == "yellow" and bin_.source_colors["isaac"] == "orange"
    store.update("cube", np.array([0.30, 0.10, 0.02]), 0.9, color="red", t=5.0)
    assert store.find("orange object") is bin_
    assert store.find("yellow object") is bin_
    assert store.find("red object").label == "cube"


def test_per_camera_names_survive_a_restart(tmp_path):
    (top, mt), (side, ms) = _bin_frames()
    store = BeliefStore()
    store.update_frame([_obs("isaac", top, mt, "orange", "bin")], t=1.0)
    store.update_frame([_obs("isaac_side", side, ms, "yellow", "bin")], t=1.5)
    path = tmp_path / "beliefs.json"
    store.save(path)
    again = BeliefStore()
    # no age limit: the views were taken at monotonic t=1.0/1.5 (seconds since
    # boot), so the default 6 h limit would drop them on a host up longer
    assert again.load(path, max_age_s=float("inf")) == 1
    assert again.all()[0].source_colors == {"isaac": "orange", "isaac_side": "yellow"}
    blob = json.loads(path.read_text())
    for r in blob["beliefs"]:
        r.pop("source_colors")
    path.write_text(json.dumps(blob))
    old = BeliefStore()
    assert old.load(path, max_age_s=float("inf")) == 1 and old.all()[0].source_colors == {}


def test_demo_config_turns_it_on_and_the_switch_parses():
    from cascade.apps.demo import _belief_store

    mem = yaml.safe_load((REPO / "configs" / "demo.yaml").read_text())["memory"]
    assert mem["per_camera_colour"] is True
    assert mem["neighbour_colour_iou"] == 0.75
    assert _belief_store(mem)._per_camera_colour is True
    assert _belief_store({})._per_camera_colour is True
    assert _belief_store({"per_camera_colour": "false"})._per_camera_colour is False
    assert _belief_store({"per_camera_colour": False})._per_camera_colour is False
    assert _belief_store({"neighbour_colour_iou": 0.6})._neighbour_colour_iou == 0.6
    with pytest.raises(ValueError):
        _belief_store({"neighbour_colour_iou": 0})


def test_colour_neighbours_are_symmetric_and_never_red_blue():
    from cascade.perception.colors import COLOR_NAMES, are_neighbours

    assert are_neighbours("orange", "yellow") and are_neighbours("yellow", "orange")
    assert are_neighbours("red", "brown") and are_neighbours("brown", "red")
    assert not are_neighbours("red", "blue") and not are_neighbours("orange", "orange")
    for a in COLOR_NAMES:
        for b in COLOR_NAMES:
            assert are_neighbours(a, b) == are_neighbours(b, a)
