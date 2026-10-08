"""Same-colour identical objects inside the 8 cm fusion gate stay separate
beliefs (backlog B31, ARCHITECTURE "Known limitations").

The limitation, reproduced on the unchanged tree with an IDEAL instance
detector (one detection per cube): the WorldWatcher and get_observation fused
a frame's detections one at a time through `BeliefStore.update()`, so the
second cube of a frame matched the belief the first cube had just created
whenever it sat inside the 8 cm proximity gate -- two 3.5 cm red cubes 4-7 cm
apart became ONE belief at an EMA blend of both (e.g. y = 0.087 for cubes at
y = 0.06 and 0.10). Colour does not help (both red) and neither does the label.

The fix is instance-level association per FRAME: detections of one frame are
first grouped into instances (only heavily overlapping image boxes -- the same
object under two open-vocabulary names, or a part inside its whole -- are one
instance), then instances and beliefs are matched one-to-one by a min-cost
assignment on 3D distance inside the unchanged gates (colour, label rule,
size-scaled radius). Two instances of one frame never claim one belief; a
belief that a frame shows to be two objects splits; beliefs nobody matched are
untouched (occlusion / object permanence unchanged) and no belief is ever
created without a detection.

Geometry: the straight-down mock camera (configs/cameras/mock.yaml
extrinsics, fx = 600, 640 x 480, table at 0.60 m) and the demo's 3.5 cm props,
painted with a one-pixel mid-height skirt like `mock_camera.synthetic_tabletop`
so the lifted cloud has height. Truth is the painted cube centre.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.memory.beliefs import BeliefStore
from cascade.perception.grounding import Extrinsics
from cascade.perception.workspace import WorkspaceFilter
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.types import Detection, Frame

FX = 600.0
W, H = 640, 480
TABLE = 0.60
CUBE = 0.035
#: configs/cameras/mock.yaml: camera at base (0.28, 0, 0.60) looking down.
T_CAM2BASE = np.array([
    [0.0, -1.0, 0.0, 0.28],
    [-1.0, 0.0, 0.0, 0.00],
    [0.0, 0.0, -1.0, 0.60],
    [0.0, 0.0, 0.0, 1.00],
])
RED_BGR = (40, 40, 200)
BLUE_BGR = (200, 40, 40)


def _px(x: float, y: float, z: float) -> tuple[float, float]:
    p = np.linalg.inv(T_CAM2BASE) @ np.array([x, y, z, 1.0])
    return FX * p[0] / p[2] + W / 2, FX * p[1] / p[2] + H / 2


def _scene(cubes, *, size: float = CUBE, colors=None):
    """Top-down RGB-D of axis-aligned cubes centred at base (x, y).

    Returns (frame, masks, truths): one full-frame mask (top face + skirt) and
    one base-frame truth centre per cube, in input order.
    """
    rgb = np.full((H, W, 3), 190, np.uint8)
    depth = np.full((H, W), TABLE, np.float32)
    masks, truths = [], []
    for i, (x, y) in enumerate(cubes):
        corners = [_px(x + dx * size / 2, y + dy * size / 2, size)
                   for dx in (-1, 1) for dy in (-1, 1)]
        us = [c[0] for c in corners]
        vs = [c[1] for c in corners]
        u0, u1 = int(round(min(us))), int(round(max(us)))
        v0, v1 = int(round(min(vs))), int(round(max(vs)))
        m = np.zeros((H, W), bool)
        m[v0 - 1:v1 + 1, u0 - 1:u1 + 1] = True
        rgb[m] = (colors[i] if colors else RED_BGR)
        depth[m] = TABLE - size / 2           # skirt at mid-height
        depth[v0:v1, u0:u1] = TABLE - size    # top face
        masks.append(m)
        truths.append(np.array([x, y, size / 2]))
    K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])
    return Frame(rgb=rgb, depth_m=depth, K=K, depth_source="sensor"), masks, truths


def _bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


class _InstanceDetector:
    """An ideal instance segmenter: one detection per mask it is handed."""

    def __init__(self, label: str = "red cube"):
        self.label = label
        self.dets: list[Detection] = []

    def show(self, masks, labels=None, confs=None, boxes=None):
        self.dets = [
            Detection(
                (labels[i] if labels else self.label),
                (confs[i] if confs else 0.9),
                (boxes[i] if boxes else _bbox(m)),
                mask=m,
            )
            for i, m in enumerate(masks)
        ]

    def detect(self, frame, classes=None):
        return list(self.dets)


class _Watch:
    """The REAL WorldWatcher fusion path fed one synthetic frame per tick."""

    def __init__(self, store: BeliefStore, label: str = "red cube"):
        self.store = store
        self.detector = _InstanceDetector(label)
        self._frame = None
        stream = SimpleNamespace(name="top", latest=lambda: self._frame,
                                 set_overlay=lambda **kw: None)
        self.cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f),
                                 Extrinsics(T=T_CAM2BASE))
        self.watcher = WorldWatcher([self.cam], self.detector, store)
        self.t = 100.0
        self.n = 0

    def tick(self, frame: Frame, masks, **show) -> float:
        self.n += 1
        self.t += 0.35
        frame.frame_id = self.n
        frame.t = self.t
        self.detector.show(masks, **show)
        self._frame = frame
        self.watcher._tick(self.cam)
        return self.t


def _xy_err(belief, truth) -> float:
    return float(np.linalg.norm(np.asarray(belief.position)[:2] - truth[:2]))


def _assert_one_belief_per_cube(beliefs, truths, tol=0.01):
    assert len(beliefs) == len(truths), (
        f"{len(beliefs)} belief(s) for {len(truths)} cubes: "
        f"{[np.round(b.position, 3).tolist() for b in beliefs]}")
    for perm in itertools.permutations(range(len(truths))):
        if all(_xy_err(beliefs[i], truths[j]) < tol for i, j in enumerate(perm)):
            return
    raise AssertionError(
        f"beliefs {[np.round(b.position, 3).tolist() for b in beliefs]} do not "
        f"sit one per cube {[np.round(t, 3).tolist() for t in truths]} within {tol} m")


# ── the limitation, through the watcher (always-on) path ─────────────────


@pytest.mark.parametrize("sep", [0.040, 0.050, 0.060, 0.070])
def test_watcher_keeps_two_same_colour_cubes_inside_the_old_gate_apart(sep):
    """RED on main: one belief at an EMA blend of the two cubes."""
    store = BeliefStore()
    w = _Watch(store)
    for _ in range(3):
        frame, masks, truths = _scene([(0.24, 0.06), (0.24, 0.06 + sep)])
        w.tick(frame, masks)
    beliefs = store.all(now=w.t)
    _assert_one_belief_per_cube(beliefs, truths)
    assert all(b.color == "red" for b in beliefs)


def test_two_same_colour_cubes_outside_the_old_gate_were_already_two():
    """Control: 10 cm apart was never the problem; it must stay two."""
    store = BeliefStore()
    w = _Watch(store)
    for _ in range(3):
        frame, masks, truths = _scene([(0.24, 0.06), (0.24, 0.16)])
        w.tick(frame, masks)
    _assert_one_belief_per_cube(store.all(now=w.t), truths)


# ── the same limitation through get_observation (runtime) ────────────────


def _runtime(store: BeliefStore):
    from cascade.skills.runtime import SkillRuntime

    rt = object.__new__(SkillRuntime)
    rt.beliefs = store
    rt._workspace = WorkspaceFilter()
    rt.extrinsics = Extrinsics(T=T_CAM2BASE)
    return rt


def test_get_observation_path_counts_two_same_colour_cubes():
    """`_update_beliefs_from_frame` is the get_observation / eval_detector
    path; `count_objects("red cube")` is what the user sees. RED on main: 1."""
    from cascade.skills.runtime import SkillRuntime

    store = BeliefStore()
    rt = _runtime(store)
    det = _InstanceDetector()
    frame, masks, truths = _scene([(0.24, 0.06), (0.24, 0.11)])
    det.show(masks)
    frame.t = 50.0
    seen = SkillRuntime._update_beliefs_from_frame(rt, frame, det.detect(frame))
    assert len(seen) == 2, "both detections are reported as visible either way"
    assert SkillRuntime.skill_count_objects(rt, "red cube")["count"] == 2
    _assert_one_belief_per_cube(store.all(now=50.0), truths)


# ── tracking: one moves, one is occluded, a blurred belief splits ────────


def test_one_cube_pushed_toward_its_twin_keeps_both_identities():
    """Twins 10 cm apart (two beliefs on main too); B is pushed to 4 cm from
    A, i.e. closer to A's belief than to its own. RED on main: B's detection
    is absorbed by A (A drifts ~2 cm toward B) and B's belief stays stale."""
    store = BeliefStore()
    w = _Watch(store)
    a, b0, b1 = (0.24, 0.06), (0.24, 0.16), (0.24, 0.10)
    for _ in range(2):
        frame, masks, _ = _scene([a, b0])
        w.tick(frame, masks)
    assert len(store.all(now=w.t)) == 2
    belief_a = min(store.all(now=w.t), key=lambda b: b.position[1])
    belief_b = max(store.all(now=w.t), key=lambda b: b.position[1])
    for _ in range(8):
        frame, masks, truths = _scene([a, b1])
        w.tick(frame, masks)
    beliefs = store.all(now=w.t)
    assert len(beliefs) == 2, [np.round(x.position, 3).tolist() for x in beliefs]
    assert _xy_err(belief_a, truths[0]) < 0.01, "A must not be pulled toward its twin"
    assert _xy_err(belief_b, truths[1]) < 0.01, "B's own belief must follow B"


def test_occluded_twin_keeps_its_belief_untouched():
    """Two twins 5 cm apart, then B is hidden (an arm passing over it). B's
    belief must not move, refresh or vanish, and A's detection must not be
    handed to it. RED on main: there is only one belief to begin with.
    (Five hidden ticks = 1.75 s, past the 1.5 s visible horizon.)"""
    store = BeliefStore()
    w = _Watch(store)
    a, b = (0.24, 0.06), (0.24, 0.11)
    for _ in range(3):
        frame, masks, truths = _scene([a, b])
        w.tick(frame, masks)
    _assert_one_belief_per_cube(store.all(now=w.t), truths)
    belief_b = max(store.all(now=w.t), key=lambda x: x.position[1])
    pos_b, seen_b, obs_b = belief_b.position.copy(), belief_b.last_seen_t, belief_b.observations
    for _ in range(5):
        frame, masks, _ = _scene([a])              # B occluded
        w.tick(frame, masks)
    beliefs = store.all(now=w.t)
    assert len(beliefs) == 2
    assert any(x is belief_b for x in beliefs)
    assert np.array_equal(belief_b.position, pos_b)
    assert belief_b.last_seen_t == seen_b and belief_b.observations == obs_b
    assert belief_b.state(now=w.t) == "remembered"
    belief_a = next(x for x in beliefs if x is not belief_b)
    assert belief_a.state(now=w.t) == "visible"
    assert _xy_err(belief_a, truths[0]) < 0.01


def test_occluded_twin_reappears_into_its_own_belief():
    store = BeliefStore()
    w = _Watch(store)
    a, b = (0.24, 0.06), (0.24, 0.11)
    for _ in range(2):
        frame, masks, _ = _scene([a, b])
        w.tick(frame, masks)
    before = {id(x) for x in store.all(now=w.t)}
    for _ in range(3):
        frame, masks, _ = _scene([a])
        w.tick(frame, masks)
    frame, masks, truths = _scene([a, b])
    w.tick(frame, masks)
    beliefs = store.all(now=w.t)
    assert {id(x) for x in beliefs} == before, "no new identity for a returning twin"
    _assert_one_belief_per_cube(beliefs, truths)
    assert all(x.state(now=w.t) == "visible" for x in beliefs)


def test_a_frame_with_two_instances_splits_a_blurred_belief():
    """A belief left at the midpoint of two twins (an old merged belief, a
    detector that merged them, a restored file) must split as soon as one
    frame shows two instances inside it -- and the surviving belief is
    RE-ANCHORED on its own detection, not EMA-blended with the old midpoint
    (which was never an object). RED on main: still one belief."""
    store = BeliefStore()
    blur = store.update("red cube", np.array([0.24, 0.085, 0.026]), 0.9,
                        extent=np.array([0.07, 0.035, 0.02]), color="red", t=10.0)
    first_seen = blur.first_seen_t
    w = _Watch(store)
    frame, masks, truths = _scene([(0.24, 0.06), (0.24, 0.11)])
    t = w.tick(frame, masks)
    beliefs = store.all(now=t)
    _assert_one_belief_per_cube(beliefs, truths, tol=0.005)
    assert any(x is blur for x in beliefs), "the old belief survives as one of the two"
    assert blur.first_seen_t == first_seen and blur.observations == 2
    newborn = next(x for x in beliefs if x is not blur)
    assert newborn.observations == 1 and newborn.first_seen_t == t
    assert all(x.last_seen_t == t for x in beliefs)


def test_no_belief_is_created_without_a_detection():
    store = BeliefStore()
    w = _Watch(store)
    frame, masks, _ = _scene([(0.24, 0.06), (0.24, 0.11)])
    w.tick(frame, [])                               # detector saw nothing
    assert store.all(now=w.t) == []
    w.tick(frame, masks)
    two = store.all(now=w.t)
    frame, masks, _ = _scene([(0.24, 0.06)])
    w.tick(frame, masks)
    assert len(store.all(now=w.t)) == len(two) == 2


# ── what must NOT change (pass on main and after the fix) ────────────────


def test_two_colours_inside_the_gate_stay_two_in_one_frame():
    """The 2026-09-10 rule: different measured colours never fuse."""
    store = BeliefStore()
    w = _Watch(store)
    frame, masks, truths = _scene([(0.24, 0.06), (0.24, 0.10)],
                                  colors=[RED_BGR, BLUE_BGR])
    w.tick(frame, masks, labels=["red cube", "blue cube"])
    beliefs = store.all(now=w.t)
    _assert_one_belief_per_cube(beliefs, truths)
    assert {b.color for b in beliefs} == {"red", "blue"}


def test_one_object_under_two_names_in_one_frame_is_one_belief():
    """Open-vocabulary detectors run class-aware NMS, so ONE object can come
    back under two names in the SAME frame. Those boxes overlap almost fully;
    they are one instance and must stay one belief with an alias (the
    label-agnostic fusion that took the Isaac open-vocab phantom rate to 0 %)."""
    store = BeliefStore()
    w = _Watch(store)
    frame, masks, truths = _scene([(0.24, 0.06)])
    m = masks[0]
    w.tick(frame, [m, m], labels=["storage box", "building block"], confs=[0.7, 0.5],
           boxes=[_bbox(m), _bbox(m) + np.array([1, -1, 2, 1], np.float32)])
    beliefs = store.all(now=w.t)
    assert len(beliefs) == 1
    assert beliefs[0].label == "storage box" and "building block" in beliefs[0].aliases


def test_a_part_detected_inside_its_whole_is_one_belief():
    """'mug' + 'handle' in one frame: the part's box sits inside the whole's.
    Same instance; on main the 8 cm gate absorbed it, and it must still."""
    store = BeliefStore()
    w = _Watch(store)
    frame, masks, truths = _scene([(0.24, 0.06)])
    whole = masks[0]
    part = np.zeros_like(whole)
    ys, xs = np.nonzero(whole)
    part[ys.min():ys.min() + 12, xs.min():xs.min() + 12] = True
    w.tick(frame, [whole, part], labels=["mug", "handle"], confs=[0.8, 0.4])
    beliefs = store.all(now=w.t)
    assert len(beliefs) == 1 and beliefs[0].label == "mug"
    assert "handle" in beliefs[0].aliases


def test_a_whole_with_two_different_parts_stays_one_belief():
    store = BeliefStore()
    w = _Watch(store)
    frame, masks, truths = _scene([(0.24, 0.06)])
    whole = masks[0]
    ys, xs = np.nonzero(whole)
    lid = np.zeros_like(whole)
    lid[ys.min():ys.min() + 10, xs.min():xs.max()] = True
    handle = np.zeros_like(whole)
    handle[ys.max() - 10:ys.max(), xs.min():xs.min() + 10] = True
    w.tick(frame, [whole, lid, handle], labels=["jar", "lid", "handle"],
           confs=[0.8, 0.5, 0.4])
    beliefs = store.all(now=w.t)
    assert len(beliefs) == 1 and beliefs[0].label == "jar"


def test_a_detection_spanning_both_twins_rejoins_them_by_design():
    """Documented limit, pinned so that changing it is a decision: a third
    detection whose support covers BOTH twins (a box or mask around the pair,
    e.g. a plural "cubes") shares pixels with each, so the three are one
    instance and one belief -- the same structure as a jar detected with its
    lid and its handle, which must stay one object. Telling a group label
    from a whole with parts is a DETECTOR problem (like the mock detector's
    one blob per colour); the store keeps the conservative answer."""
    store = BeliefStore()
    w = _Watch(store)
    frame, masks, truths = _scene([(0.24, 0.06), (0.24, 0.10)])
    group = masks[0] | masks[1]
    w.tick(frame, [masks[0], masks[1], group], confs=[0.9, 0.9, 0.6])
    assert len(store.all(now=w.t)) == 1
    # ... while the same two twins WITHOUT the spanning detection are two
    w2 = _Watch(BeliefStore())
    w2.tick(frame, [masks[0], masks[1]])
    _assert_one_belief_per_cube(w2.store.all(now=w2.t), truths)


def test_a_label_strict_store_still_never_fuses_two_labels():
    store = BeliefStore(label_agnostic=False)
    w = _Watch(store)
    frame, masks, _ = _scene([(0.24, 0.06)])
    m = masks[0]
    w.tick(frame, [m, m], labels=["storage box", "building block"])
    assert len(store.all(now=w.t)) == 2


# ── the new API ──────────────────────────────────────────────────────────


def _obs(label, xy, *, color: str | None = "red", conf=0.9, bbox=None,
         extent=(0.035, 0.035, 0.02)):
    from cascade.memory.beliefs import FrameObservation

    return FrameObservation(label=label, position=np.array([xy[0], xy[1], 0.026]),
                            conf=conf, extent=np.array(extent), top_z=0.035,
                            color=color, bbox=None if bbox is None else np.asarray(bbox, float))


def test_update_frame_returns_one_belief_per_observation_in_input_order():
    store = BeliefStore()
    out = store.update_frame([_obs("red cube", (0.24, 0.06), bbox=(0, 0, 10, 10)),
                              _obs("red cube", (0.24, 0.11), bbox=(20, 0, 30, 10)),
                              _obs("cube", (0.24, 0.06), bbox=(0, 0, 10, 10))], t=5.0)
    assert len(out) == 3 and out[0] is out[2] and out[0] is not out[1]
    assert len(store.all(now=5.0)) == 2


def test_update_frame_without_boxes_is_strictly_one_to_one():
    """No image boxes -> no evidence two detections are one object -> two."""
    store = BeliefStore()
    store.update_frame([_obs("red cube", (0.24, 0.06)), _obs("red cube", (0.24, 0.065))], t=1.0)
    assert len(store.all(now=1.0)) == 2


def test_legacy_switch_reproduces_per_detection_fusion():
    """`instance_association=False` is the pre-B31 behaviour (the live A/B
    baseline, `memory.instance_association: false`)."""
    store = BeliefStore(instance_association=False)
    w = _Watch(store)
    frame, masks, _ = _scene([(0.24, 0.06), (0.24, 0.11)])
    w.tick(frame, masks)
    assert len(store.all(now=w.t)) == 1


def test_assignment_prefers_the_right_pairing_over_the_most_pairs():
    """A at 0, B at 7 cm; a detection at A and a NEW object 7 cm on the other
    side of A. Maximising the number of matches would hand A to the new object
    and A's detection to B; the right answer keeps A and births the new one."""
    store = BeliefStore()
    A, B = store.update_frame([_obs("red cube", (0.24, 0.00)),
                               _obs("red cube", (0.24, 0.07))], t=1.0)
    assert A is not B
    out = store.update_frame([_obs("red cube", (0.24, 0.005)),
                              _obs("red cube", (0.24, -0.07))], t=2.0)
    assert out[0] is A and out[1] is not A and out[1] is not B
    assert B.last_seen_t == 1.0
    assert len(store.all(now=2.0)) == 3


def test_a_lone_detection_goes_where_update_would_send_it():
    """One instance alone is matched exactly as `update()` matches one
    observation: the NEAREST belief inside its own gate, in metres -- not
    the one nearest relative to a size-scaled gate. Here a 30 cm bin's gate
    (15 cm) admits a detection 6 cm from its centre, but a cube's belief sits
    5 cm away and must win, as it does through `update()`."""
    def two_beliefs():
        store = BeliefStore()
        cube, bin_ = store.update_frame([
            _obs("cube", (0.24, 0.05)),
            _obs("bin", (0.24, -0.06), extent=(0.30, 0.20, 0.10), color=None),
        ], t=1.0)
        return store, cube, bin_

    store, cube, bin_ = two_beliefs()
    got = store.update_frame([_obs("cube", (0.24, 0.0), color=None)], t=2.0)[0]
    ref_store, ref_cube, _ = two_beliefs()
    ref = ref_store.update("cube", np.array([0.24, 0.0, 0.026]), 0.9,
                           extent=np.array([0.035, 0.035, 0.02]), t=2.0)
    assert ref is ref_cube and got is cube
    np.testing.assert_allclose(got.position, ref.position)
    assert bin_.last_seen_t == 1.0


def test_one_frame_counts_once_per_instance():
    """Two names for ONE object in one frame are one observation of it, not
    two: `observations` is what the visual interface uses to call a
    remembered single sighting `confirmed`."""
    store = BeliefStore()
    box = (100, 100, 140, 140)
    out = store.update_frame([_obs("storage box", (0.24, 0.05), bbox=box, conf=0.7),
                              _obs("building block", (0.24, 0.05), bbox=box, conf=0.5)],
                             t=1.0)
    assert out[0] is out[1] and out[0].observations == 1
    store.update_frame([_obs("storage box", (0.24, 0.05), bbox=box, conf=0.7),
                        _obs("building block", (0.24, 0.05), bbox=box, conf=0.5)], t=2.0)
    assert out[0].observations == 2
    assert out[0].label == "storage box" and out[0].aliases == {"building block"}


def test_an_instance_never_joins_two_measured_colours():
    """A colourless detection overlapping a red one and a blue one cannot
    chain them into one instance (the 2026-09-10 colour rule, applied inside
    a frame): red and blue stay two beliefs."""
    store = BeliefStore()
    out = store.update_frame([
        _obs("cube", (0.24, 0.050), color="red", bbox=(0, 0, 10, 10)),
        _obs("thing", (0.24, 0.055), color=None, bbox=(5, 0, 15, 10)),
        _obs("cube", (0.24, 0.060), color="blue", bbox=(10, 0, 20, 10)),
    ], t=1.0)
    beliefs = store.all(now=1.0)
    assert len(beliefs) == 2 and {b.color for b in beliefs} == {"red", "blue"}
    assert out[0] is out[1] and out[2] is not out[0]


def test_image_overlap_never_fuses_what_the_gate_keeps_apart():
    """Shared pixels are necessary, not sufficient: a small object whose box
    lies inside a big object's box but 20 cm away in 3D (beyond the gate) is
    its own object, exactly as `update()` would keep it."""
    store = BeliefStore()
    out = store.update_frame([
        _obs("tray", (0.24, 0.00), bbox=(0, 0, 100, 100), extent=(0.10, 0.08, 0.02)),
        _obs("red cube", (0.24, 0.20), bbox=(10, 10, 20, 20)),
    ], t=1.0)
    assert out[0] is not out[1] and len(store.all(now=1.0)) == 2


def test_two_twins_moving_together_do_not_swap():
    """Greedy nearest-first would pair A' with B (1.9 cm) and B' with A; the
    min-cost assignment keeps each with its own belief (2.1 + 2.0 cm)."""
    store = BeliefStore()
    A = store.update("red cube", np.array([0.24, 0.00, 0.026]), 0.9, color="red", t=1.0)
    B = store.update("red cube", np.array([0.24, 0.10, 0.026]), 0.9, color="red", t=1.0)
    B.position = np.array([0.24, 0.04, 0.026])       # established apart, now 4 cm
    out = store.update_frame([_obs("red cube", (0.24, 0.021)),
                              _obs("red cube", (0.24, 0.060))], t=2.0)
    assert out[0] is A and out[1] is B


def test_min_cost_assignment_matches_brute_force():
    from cascade.memory.beliefs import _min_cost_assignment

    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(1, 5))
        m = int(rng.integers(n, 7))
        cost = rng.random((n, m))
        got = _min_cost_assignment(cost)
        assert len(got) == n and len(set(got)) == n
        best = min(sum(cost[i, p[i]] for i in range(n))
                   for p in itertools.permutations(range(m), n))
        assert sum(cost[i, got[i]] for i in range(n)) == pytest.approx(best)


# ── the mock detector can now see instances (opt-in) ─────────────────────


def test_mock_detector_instances_mode_splits_same_colour_blobs():
    from cascade.perception.detector import MockDetector

    frame, masks, _ = _scene([(0.24, 0.06), (0.24, 0.11)])
    union = MockDetector(label="red cube").detect(frame)
    assert len(union) == 1, "default unchanged: one blob per colour"
    split = MockDetector(label="red cube", instances=True).detect(frame)
    assert len(split) == 2
    assert not (split[0].mask & split[1].mask).any()
    assert {tuple(_bbox(d.mask)) for d in split} == {tuple(_bbox(m)) for m in masks}


def test_instances_switches_are_wired_from_config():
    """`memory.instance_association` (default true, a YAML "false" string is
    false) and the mock detector's opt-in `instances`."""
    from conftest import REPO

    import yaml
    from cascade.apps.demo import _instance_association_enabled
    from cascade.config import Cfg

    shipped = yaml.safe_load((REPO / "configs" / "demo.yaml").read_text())["memory"]
    assert shipped["instance_association"] is True
    assert _instance_association_enabled(Cfg({})) is True
    assert _instance_association_enabled(Cfg({"instance_association": False})) is False
    assert _instance_association_enabled(Cfg({"instance_association": "false"})) is False
    assert BeliefStore()._instance_association is True


# ── rendered RGB-D against MuJoCo physics truth (the separation sweep) ───


@pytest.fixture(scope="module")
def sweep():
    """scripts/measure_same_colour_sweep.py, the instrument behind the
    numbers in docs: two red 3.5 cm cubes rendered by MuJoCo, fused through
    the real WorldWatcher path, scored against data.xpos."""
    pytest.importorskip("mujoco")
    import sys

    from conftest import REPO
    from mujoco_gl_probe import probe_offscreen_gl

    script = REPO / "scripts" / "measure_same_colour_sweep.py"
    gl = probe_offscreen_gl(script, height=64, width=64,
                            model_xml="<mujoco><worldbody><geom type='box' size='.1 .1 .1'/>"
                                      "</worldbody></mujoco>")
    if not gl.available:
        pytest.skip(f"needs an offscreen GL context (MUJOCO_GL=egl): {gl.reason}")
    sys.path.insert(0, str(REPO / "scripts"))
    import measure_same_colour_sweep as module

    return module.run(separations=(0.035, 0.045, 0.060, 0.075), cameras=("top",), axes=("y",))


def _rows(report, **match):
    return {r["separation_m"]: r for r in report["rows"]
            if all(r[k] == v for k, v in match.items())}


def test_rendered_twins_inside_the_gate_are_two_beliefs_on_physics_truth(sweep):
    """RED on main: the per-detection store returns ONE belief 1.7-3.1 cm off
    both cubes for every separation below 8 cm, although the detector hands
    it two instances. With frame association each cube has its own belief,
    as accurate as when that cube is alone on the table."""
    new = _rows(sweep, detector="instances", store="instance")
    old = _rows(sweep, detector="instances", store="legacy")
    for sep in (0.045, 0.060, 0.075):
        assert new[sep]["detections"] == 2 and old[sep]["detections"] == 2
        assert new[sep]["beliefs"] == 2 and new[sep]["resolved"], new[sep]
        assert max(e - f for e, f in zip(new[sep]["xy_err_m"], new[sep]["floor_m"])) <= 0.002
        assert old[sep]["beliefs"] == 1, "premise: the old store merges this pair"


def test_rendered_twins_the_detector_merges_stay_merged(sweep):
    """What association cannot fix, pinned: touching cubes are ONE colour blob
    (one detection) and the default mock detector returns one blob per colour
    at ANY separation -- one detection, one belief."""
    assert _rows(sweep, detector="instances", store="instance")[0.035]["detections"] == 1
    for sep, row in _rows(sweep, detector="union", store="instance").items():
        assert row["detections"] == 1 and row["beliefs"] == 1, (sep, row)
