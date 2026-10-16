"""Two-phase brain localization (spec 2026-10-16-perception-two-phase-design.md, 7.2 B1, tests 1-6 of 7.4).

The brain names and boxes the object on frame0, which takes seconds; SAM then
re-anchors that box on a FRESH frame1 and every geometry the arm uses comes
from frame1, under the unchanged 5 s analysed-frame rule. The brain's own step
is capped by `perception_loop.semantic_max_age_s` instead.

Time is scaled 10x down: the analysed-frame limit is 0.5 s, the semantic cap
2.0 s (20 s live). The fake brain boxes the coloured object it SEES in the JPEG
it is sent, and the fake SAM masks the coloured pixels inside each prompt box,
so a scene edit made while the brain "thinks" (between frame0 and frame1)
reaches every stage the way a real one would. No network, no GPU: mock camera,
mock arm, the real runtime, BrainDetector and LockedDetector.
"""

from __future__ import annotations

import base64
import functools
import json
import re
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cascade.perception.brain_detector import BrainDetector
from cascade.perception.mock_camera import MockCamera, synthetic_tabletop
from cascade.types import SkillError

SCALE = 0.1  # seconds in this test per second live
RED = (40, 40, 200)  # BGR, what synthetic_tabletop paints the cube with
HOME_BOX = (280, 200, 360, 260)  # the cube's pixels in the mock camera
MOVED_BOX = (380, 200, 460, 260)  # 100 px to the right: ~9 cm on the table
ISO = (150, 190, 230)  # BGR orange whose grey level (~197) is the table's (190): iso-luminant
PX_PER_CM = 600.0 * 0.01 / 0.55  # mock fx 600 px at the cube's top face (0.55 m): ~10.9 px per cm
TIMING_KEYS = {"semantic_s", "frame0_age_at_reanchor_s", "reanchor_s", "frame1_age_s",
               "scene_diff_global", "scene_diff_local", "moved_m"}
# SAM calls of one two-phase attempt, in order (`_ObjectSam.script` keys,
# 1-based): the naming mask on frame0 (prompt: the brain's box), frame0
# re-masked with the grown prompt (fix round 3), frame1 with that same prompt.
# A retry after a scene change or a failed frame1 re-anchor adds RETRY.
NAME, REF, FRESH = 1, 2, 3
RETRY = 3


def _object_mask(bgr: np.ndarray) -> np.ndarray:
    """Coloured pixels: the cube, never the grey table or a grey patch."""
    return (bgr.max(axis=-1).astype(int) - bgr.min(axis=-1).astype(int)) > 60


def _pixel_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class _ImageBrain:
    """OpenAI-compatible fake: boxes the coloured object in the image it is
    sent (every prompt), after `refer_s`. `on_refer(n)` runs at the start of
    the n-th request (the scene edits between frame0 and frame1); `reply`
    replaces the answer text, `error` is raised instead of answering;
    `region(query, bgr) -> mask` picks what a query boxes (default: every
    coloured pixel); `queries` records the query of every request."""

    def __init__(self):
        self.refer_s = 0.02
        self.on_refer = None
        self.reply: str | None = None
        self.error: BaseException | None = None
        self.region = lambda query, bgr: _object_mask(bgr)
        self.refers = 0
        self.queries: list[str] = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        url = content[0]["image_url"]["url"]
        text_parts = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        asked = re.search(r'Locate every instance of "(.*?)" in the image', text_parts)
        query = asked.group(1) if asked else ""
        with self._lock:
            self.refers += 1
            self.queries.append(query)
            n = self.refers
        if self.on_refer is not None:
            self.on_refer(n)
        time.sleep(self.refer_s)
        if self.error is not None:
            raise self.error
        text = self.reply
        if text is None:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
            box = _pixel_box(self.region(query, img))
            h, w = img.shape[:2]
            text = "[]" if box is None else json.dumps([{"label": "red cube", "bbox_2d": [
                round(box[0] * 1000 / w), round(box[1] * 1000 / h),
                round(box[2] * 1000 / w), round(box[3] * 1000 / h)]}])
        message = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class _ObjectSam:
    """Box-prompted SAM stand-in: each box's mask is the coloured pixels inside
    it (None when there are none). Records (frame_id, boxes) per call;
    `script[n]` (1-based call number) replaces the n-th answer: a list of
    masks, or a callable(frame, boxes) -> list (e.g. one that sleeps).
    `default(frame, boxes)`, when set, answers every unscripted call (None
    from it: the normal answer)."""

    def __init__(self):
        self.calls: list[tuple[int, np.ndarray]] = []
        self.script: dict[int, object] = {}
        self.default = None

    def masks(self, frame, boxes):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.calls.append((int(frame.frame_id), boxes.copy()))
        scripted = self.script.get(len(self.calls), self.default)
        if callable(scripted):
            scripted = scripted(frame, boxes)
        if scripted is not None:
            return scripted
        obj = _object_mask(frame.rgb)
        h, w = obj.shape
        out = []
        for x0, y0, x1, y1 in boxes:
            m = np.zeros((h, w), dtype=bool)
            xa, ya = max(int(np.floor(x0)), 0), max(int(np.floor(y0)), 0)
            xb, yb = min(int(np.ceil(x1)), w), min(int(np.ceil(y1)), h)
            m[ya:yb, xa:xb] = obj[ya:yb, xa:xb]
            out.append(m if m.any() else None)
        return out


class _Scene:
    """What a mock camera renders from its next grab on. One tuple, swapped
    atomically: the stream's pump thread renders while the brain edits."""

    def __init__(self, camera: MockCamera):
        self.state = dict(box=HOME_BOX, depth_shift=0.0, patch=None, colour=None, K=None, T=None, capture=None,
                          dest=None, neighbour=None, lift=0.0)
        camera._render = self._render  # instance attribute: MockCamera._grab calls self._render()

    def set(self, **changes) -> None:
        self.state = {**self.state, **changes}

    def toggle(self, key, a, b) -> None:
        self.set(**{key: b if self.state[key] == a else a})

    def _render(self):
        s = self.state
        frame = synthetic_tabletop(box_px=s["box"])
        if s["colour"] is not None:
            frame.rgb[np.all(frame.rgb == RED, axis=-1)] = s["colour"]
        if s["patch"] is not None:
            x0, y0, x1, y1, grey = s["patch"]
            frame.rgb[y0:y1, x0:x1] = grey
        if s["dest"] is not None:  # a flat coloured pad on the table (a place destination)
            x0, y0, x1, y1, bgr = s["dest"]
            frame.rgb[y0:y1, x0:x1] = bgr
        if s["lift"]:  # the cube alone (its pixels and skirt) rises; the table stays
            x0, y0, x1, y1 = s["box"]
            frame.depth_m[y0 - 1:y1 + 1, x0 - 1:x1 + 1] -= np.float32(s["lift"])
        if s["neighbour"] is not None:  # a second, 5 cm tall object
            x0, y0, x1, y1, bgr = s["neighbour"]
            frame.rgb[y0:y1, x0:x1] = bgr
            frame.depth_m[y0:y1, x0:x1] = np.float32(0.6 - 0.05)
        if s["depth_shift"]:
            frame.depth_m = frame.depth_m + np.float32(s["depth_shift"])
        if s["K"] is not None:
            frame.K = np.array(s["K"], dtype=np.float64)
        if s["T"] is not None:
            frame.T_base_cam = np.array(s["T"], dtype=np.float64)
        if s["capture"] is not None:
            frame.capture = dict(s["capture"])
        return frame


def _build(tmp_path, monkeypatch, *, segmenter=True):
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.skills import runtime as runtime_mod

    brain, sam = _ImageBrain(), _ObjectSam()
    detector = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", api_key_file=None, client=brain,
                             segmenter=sam if segmenter else None, refresh_s=4.0 * SCALE,
                             max_cache_age_s=30.0 * SCALE, idle_refresh_s=30.0 * SCALE)
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    # The 5 s analysed-frame limit, scaled like the brain.
    monkeypatch.setattr(runtime_mod, "_analyzed_frame_age_s",
                        functools.partial(runtime_mod._analyzed_frame_age_s, max_age_s=5.0 * SCALE))
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    # No watcher: the runtime's own calls only. The semantic cap, scaled.
    cfg._data["perception_loop"] = {"enabled": False, "semantic_max_age_s": 20.0 * SCALE}
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    scene = _Scene(runtime.camera._camera)
    runtime.camera.get_frame()  # the next delivery is rendered by the scene
    runtime.camera.get_frame()
    return SimpleNamespace(rt=runtime, arm=arm, brain=brain, sam=sam, scene=scene, detector=detector,
                           mod=runtime_mod, close=lambda: demo.shutdown_runtime(runtime, arm))


@pytest.fixture
def make(tmp_path, monkeypatch):
    from owned_server import held_dead_port

    built = []
    # The mock profile's grasp backend reaches GraspGen-X: keep it offline.
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))

        def factory(**kwargs):
            t = _build(tmp_path / f"rt{len(built)}", monkeypatch, **kwargs)
            built.append(t)
            return t

        try:
            yield factory
        finally:
            for t in built:
                t.close()


@pytest.fixture
def tp(make):
    return make()


def _commands(t) -> int:
    raw = getattr(t.arm, "commands", None)
    assert raw is not None, "the mock arm records every streamed command"
    return len(raw)


def _cube_xy(t, box) -> np.ndarray:
    """Base-frame XY of a cube painted at `box` (mock extrinsics: straight down from 0.6 m)."""
    K = synthetic_tabletop().K
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    z = 0.6 - 0.025
    p_cam = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, 1.0])
    return (np.asarray(t.rt.extrinsics.cam_to_base()) @ p_cam)[:2]


# -- 1. the slow brain no longer expires the geometry --------------------------


def test_slow_refer_localizes_on_a_fresh_frame_and_grasps(tp):
    """RED on main (probe_red.py case A): a 6.5 s refer (0.65 scaled) blew the
    5 s limit on every pick. Now the brain's frame only names the object; the
    returned frame is a newer one whose analysis took a fraction of the limit."""
    tp.brain.refer_s = 0.65
    frame, fix = tp.rt._localize("red cube")
    semantic_frame = tp.sam.calls[0][0]  # SAM masks the brain's boxes on the brain's frame
    assert frame.frame_id > semantic_frame, "the geometry must come from a frame newer than the brain's"
    assert [c[0] for c in tp.sam.calls] == [semantic_frame, semantic_frame, frame.frame_id], (
        "the naming mask and its same-prompt re-mask on the brain's frame, one re-anchor on the returned frame")
    assert tp.brain.refers == 1
    assert np.linalg.norm(np.asarray(fix.position)[:2] - _cube_xy(tp, HOME_BOX)) < 0.01

    n0 = _commands(tp)
    tp.arm.object_stop_frac = 0.5  # the mock jaws jam halfway: the cube is in the gripper
    result = tp.rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    assert _commands(tp) > n0, "the arm grasped"
    timing = result["perception_timing"]
    assert TIMING_KEYS <= set(timing), timing
    assert timing["frame1_age_s"] < 5.0 * SCALE
    assert timing["semantic_s"] >= 0.65 and timing["frame0_age_at_reanchor_s"] >= 0.65
    assert timing["moved_m"] < 0.005
    assert tp.rt.trace.rows()[-1]["result"]["perception_timing"] == timing, "the trace carries it too"


def test_without_a_segmenter_the_single_frame_path_and_its_limit_stay(make):
    """7.3: no segmenter configured -> today's single-frame path and 5 s rule."""
    t = make(segmenter=False)
    t.brain.refer_s = 0.65
    n0 = _commands(t)
    with pytest.raises(t.mod.SlowPerceptionError, match="Perception took"):
        t.rt.skill_grasp_object("red cube")
    assert _commands(t) == n0
    result = t.rt.execute("localize_object", {"label": "red cube"})
    assert "perception_timing" not in result


# -- 2. the 5 s rule still applies to the geometry -----------------------------


def test_a_slow_reanchor_is_still_refused_by_the_frame1_limit(tp):
    def slow(frame, boxes):
        time.sleep(0.6)  # 6 s live: SAM + lift on frame1 past the 5 s limit
        return None  # then the normal masks

    tp.sam.script[FRESH] = slow
    n0 = _commands(tp)
    with pytest.raises(tp.mod.SlowPerceptionError, match="Perception took"):
        tp.rt.skill_grasp_object("red cube")
    assert _commands(tp) == n0, "refused before any motion"
    assert len(tp.sam.calls) == FRESH and tp.brain.refers == 1, "no retry after a slow analysis"


# -- 3. the semantic cap -------------------------------------------------------


def test_a_refer_older_than_the_semantic_cap_is_refused(tp):
    tp.brain.refer_s = 2.5  # 25 s live > semantic_max_age_s 20 s
    n0 = _commands(tp)
    with pytest.raises(tp.mod.SlowPerceptionError, match="semantic"):
        tp.rt.skill_grasp_object("red cube")
    assert _commands(tp) == n0
    assert len(tp.sam.calls) == 1, "never re-anchored"


# -- 4. scene changed between frame0 and frame1 ---------------------------------


def _moves_once(t, **change):
    def on_refer(n):
        if n == 1:
            t.scene.set(**change)
    t.brain.on_refer = on_refer


def test_a_global_change_retries_and_finds_the_new_position(tp):
    # A hand fills the left of the view and the cube moves while the brain thinks.
    _moves_once(tp, box=MOVED_BOX, patch=(0, 0, 200, 480, 40))
    frame, fix = tp.rt._localize("red cube")
    assert tp.brain.refers == 2, "one retry of the whole two-phase"
    xy = np.asarray(fix.position)[:2]
    assert np.linalg.norm(xy - _cube_xy(tp, MOVED_BOX)) < 0.01, "the NEW position"
    assert np.linalg.norm(xy - _cube_xy(tp, HOME_BOX)) > 0.05, "never the old one"


def test_a_global_change_alone_is_a_scene_change(tp):
    # Only far from the cube: the local check and the geometry see nothing.
    _moves_once(tp, patch=(0, 0, 200, 480, 40))
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    assert tp.brain.refers == 2, "the global change cost one retry"
    timing = result["perception_timing"]
    assert timing["scene_diff_global"] <= 6.0 and timing["scene_diff_local"] <= 8.0


def test_a_cube_moved_twice_raises_scene_changed_and_never_moves(tp):
    tp.brain.on_refer = lambda n: tp.scene.toggle("box", HOME_BOX, MOVED_BOX)
    n0 = _commands(tp)
    with pytest.raises(tp.mod.SceneChangedError, match="scene changed during perception"):
        tp.rt.skill_grasp_object("red cube")
    assert _commands(tp) == n0
    assert tp.brain.refers == 2, "exactly one retry"


def test_a_small_local_change_is_caught_by_the_local_check(tp):
    # A grey finger covers the cube's left 30 % (then its right 30 %) while
    # the brain thinks: the global diff stays tiny and the visible part of the
    # cube moves less than 2 cm, so only the local check sees it.
    fingers = {1: (279, 199, 304, 262, 190), 2: (336, 199, 362, 262, 190)}
    tp.brain.on_refer = lambda n: tp.scene.set(patch=fingers.get(n))
    result = tp.rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is False and "SceneChangedError" in result["error"], result.get("error")
    timing = result["perception_timing"]
    assert timing["scene_diff_global"] <= 6.0 < 8.0 < timing["scene_diff_local"], timing
    assert tp.brain.refers == 2


def test_a_cube_moved_3cm_with_identical_pixels_is_refused(tp):
    # Same pixels everywhere; only depth says the cube is 3 cm higher.
    tp.brain.on_refer = lambda n: tp.scene.toggle("depth_shift", 0.0, -0.03)
    n0 = _commands(tp)
    result = tp.rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is False and "SceneChangedError" in result["error"], result.get("error")
    timing = result["perception_timing"]
    assert timing["scene_diff_global"] == 0.0 and timing["scene_diff_local"] == 0.0
    assert timing["moved_m"] == pytest.approx(0.03, abs=0.005)
    assert _commands(tp) == n0


def test_a_colour_change_on_frame1_is_never_grasped(tp):
    # Same grey level, another colour: frame1's mask is green, so the red cube
    # the brain named is not what frame1 shows. The colour local check sees
    # this change before SAM runs; with it out of the way the frame1 colour
    # rule alone still refuses.
    tp.detector._reanchor_local_diff = float("inf")
    _moves_once(tp, colour=(40, 121, 40))
    n0 = _commands(tp)
    with pytest.raises(SkillError) as err:
        tp.rt.skill_grasp_object("red cube")
    assert "green" in str(err.value), err.value
    assert _commands(tp) == n0
    assert len(tp.sam.calls) >= FRESH and tp.sam.calls[FRESH - 1][0] > tp.sam.calls[NAME - 1][0], "frame1 was re-anchored"


@pytest.mark.parametrize("change", ["intrinsics", "camera", "shift", "turn"])
def test_a_camera_change_between_the_frames_is_a_scene_change(tp, change):
    T = np.asarray(tp.rt.extrinsics.cam_to_base(), dtype=float)
    small = np.eye(4)
    if change == "shift":
        small[0, 3] = 0.003  # 3 mm > 2 mm
    elif change == "turn":
        a = np.radians(1.0)  # 1 deg > 0.5 deg about the optical axis
        small[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    K = synthetic_tabletop().K
    K2 = K.copy()
    K2[0, 0] = K2[1, 1] = 610.0
    key, first, second = {
        "intrinsics": ("K", K.tolist(), K2.tolist()),
        "camera": ("capture", {"source": "mock", "camera": "cam0"}, {"source": "mock", "camera": "side"}),
        "shift": ("T", T.tolist(), (T @ small).tolist()),
        "turn": ("T", T.tolist(), (T @ small).tolist()),
    }[change]
    tp.scene.set(**{key: first})
    tp.rt.camera.get_frame()
    tp.rt.camera.get_frame()
    tp.brain.on_refer = lambda n: tp.scene.toggle(key, first, second)
    n0 = _commands(tp)
    with pytest.raises(tp.mod.SceneChangedError):
        tp.rt.skill_grasp_object("red cube")
    assert _commands(tp) == n0
    assert tp.brain.refers == 2


def test_a_camera_pose_within_tolerance_is_the_same_view(tp):
    T = np.asarray(tp.rt.extrinsics.cam_to_base(), dtype=float)
    small = np.eye(4)
    a = np.radians(0.2)
    small[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    small[0, 3] = 0.001
    tp.scene.set(T=T.tolist())
    tp.rt.camera.get_frame()
    tp.rt.camera.get_frame()
    tp.brain.on_refer = lambda n: tp.scene.toggle("T", T.tolist(), (T @ small).tolist())
    tp.rt._localize("red cube")
    assert tp.brain.refers == 1


def test_pick_and_place_ends_on_scene_changed_without_motion_recovery(tp):
    tp.brain.on_refer = lambda n: tp.scene.toggle("box", HOME_BOX, MOVED_BOX)
    n0 = _commands(tp)
    result = tp.rt.execute("pick_and_place", {"object": "red cube"})
    assert result["ok"] is False and "scene changed during perception" in result["error"], result.get("error")
    assert result.get("grasp_attempts") == 1
    assert _commands(tp) == n0, "no re-home, no retry motion"
    assert tp.brain.refers == 2


def _side_cameras(t, n=1):
    """`n` more fusing cameras, each with its own mock scene (the primary sees no cube)."""
    t.scene.set(box=(0, 0, 0, 0))  # nothing on the primary's table
    t.rt.camera.get_frame()
    t.rt.camera.get_frame()
    sides, scenes = [], []
    for i in range(n):
        camera = MockCamera()
        camera.open()
        scenes.append(_Scene(camera))
        sides.append(SimpleNamespace(stream=SimpleNamespace(name=f"side{i}", get_frame=camera.get_frame),
                                     depth=t.rt.depth, extrinsics=t.rt.extrinsics, fuse=True))
    t.rt._other_cams = lambda: list(sides)
    return scenes


def _side_camera(t):
    """A second fusing camera with its own mock scene (the primary sees no cube)."""
    return _side_cameras(t, 1)[0]


def test_the_other_camera_pass_is_two_phase_too(tp):
    side = _side_camera(tp)
    seen = []
    orig = tp.detector.reanchor

    def spy(frame, det, **kw):
        seen.append((kw["since"].frame_id, frame.frame_id))
        return orig(frame, det, **kw)

    tp.detector.reanchor = spy  # instance attribute: the class still defines reanchor
    frame, fix = tp.rt._localize("red cube")
    assert len(seen) == 1 and seen[0][1] > seen[0][0]
    assert frame.frame_id == seen[0][1], "the side camera's fresh frame is the one returned"
    assert np.linalg.norm(np.asarray(fix.position)[:2] - _cube_xy(tp, HOME_BOX)) < 0.01
    assert side.state["box"] == HOME_BOX


def test_scene_changed_on_the_other_camera_is_not_swallowed(tp):
    side = _side_camera(tp)
    tp.brain.on_refer = lambda n: side.toggle("box", HOME_BOX, MOVED_BOX)
    with pytest.raises(tp.mod.SceneChangedError):
        tp.rt._localize("red cube")


# -- 5. the brain down: unchanged ------------------------------------------------


def _api_timeout() -> Exception:
    """openai's APITimeoutError (built without its transport's request object,
    whose module differs across openai versions), or a same-named stand-in on
    CI's minimal install, which has no openai."""
    try:
        from openai import APITimeoutError
    except ImportError:
        class APITimeoutError(Exception):
            pass
    err = APITimeoutError.__new__(APITimeoutError)
    Exception.__init__(err, "Request timed out.")
    return err


@pytest.mark.parametrize("failure", ["timeout", "prose"])
def test_a_brain_failure_is_the_same_miss_as_main(make, failure):
    outcomes = []
    for segmenter in (True, False):  # two-phase capable; the single-frame path (= main's)
        t = make(segmenter=segmenter)
        if failure == "timeout":
            t.brain.error = _api_timeout()
        else:
            t.brain.reply = "I can see a red cube on the table."
        n0 = _commands(t)
        with pytest.raises(SkillError) as err:
            t.rt.skill_grasp_object("red cube")
        assert _commands(t) == n0
        assert t.sam.calls == [], "no SAM call and no re-anchor without a box"
        outcomes.append((type(err.value), str(err.value)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] is SkillError and "no detections" in outcomes[0][1]


# -- 6. SAM finds nothing usable on frame1 ---------------------------------------


def _depth_hole(frame, boxes):
    """SAM masks the cube on frame1 (the right mask: same pixels as on
    frame0), but the depth camera has a hole there: only the four corner
    pixels of the mask keep their depth, fewer than min_points."""
    x0, y0, x1, y1 = (int(v) for v in boxes[0])
    obj = np.zeros(frame.rgb.shape[:2], dtype=bool)
    obj[y0:y1, x0:x1] = _object_mask(frame.rgb)[y0:y1, x0:x1]
    ys, xs = np.nonzero(obj)
    keep = np.zeros_like(obj)
    for y in (ys.min(), ys.max()):
        row = xs[ys == y]
        keep[y, row.min()] = keep[y, row.max()] = True
    frame.depth_m[obj & ~keep] = 0.0
    return [obj]


def _outside(frame, boxes):
    m = np.zeros(frame.rgb.shape[:2], dtype=bool)
    m[0:60, 0:80] = True  # table, far from the box
    return [m]


SAM_FAILURES = {
    "none": lambda frame, boxes: [None],
    "empty": lambda frame, boxes: [np.zeros(frame.rgb.shape[:2], dtype=bool)],
    "outside": _outside,
    "few_points": _depth_hole,
}
SAM_FAILURE_REASONS = {"none": "no mask", "empty": "empty mask", "outside": "off the box",
                       "few_points": "valid depth points"}


@pytest.mark.parametrize("failure", sorted(SAM_FAILURES))
def test_no_usable_mask_on_frame1_is_refused_after_one_fresh_refer(tp, failure):
    tp.sam.script[FRESH] = tp.sam.script[FRESH + RETRY] = SAM_FAILURES[failure]
    n0 = _commands(tp)
    with pytest.raises(SkillError, match="could not re-anchor 'red cube' on a fresh image") as err:
        tp.rt.skill_grasp_object("red cube")
    assert SAM_FAILURE_REASONS[failure] in str(err.value), err.value
    assert not isinstance(err.value, (tp.mod.SlowPerceptionError, tp.mod.SceneChangedError))
    assert _commands(tp) == n0, "no box fallback, no motion"
    assert tp.brain.refers == 2 and len(tp.sam.calls) == 2 * RETRY, "one retry with a fresh refer"


def test_one_failed_reanchor_is_retried_with_a_fresh_refer(tp):
    tp.sam.script[FRESH] = SAM_FAILURES["none"]
    frame, fix = tp.rt._localize("red cube")
    assert tp.brain.refers == 2 and frame.frame_id == tp.sam.calls[FRESH + RETRY - 1][0]


def test_a_reanchored_detection_without_a_mask_is_never_measured_as_its_box(tp):
    """Whatever a detector's reanchor() returns, the runtime measures frame1
    only through a mask: a box cloud includes table pixels."""
    from cascade.perception.brain_detector import ReanchorResult
    from cascade.types import Detection

    def maskless(frame, det, **_):
        return ReanchorResult(Detection(det.label, det.conf, np.array(det.bbox, np.float32)), False, "", 0.0, 0.0)

    tp.detector.reanchor = maskless  # instance attribute: the class still defines reanchor
    with pytest.raises(tp.mod.ReanchorError):
        tp.rt._localize("red cube")


def test_a_fix_that_leaves_the_workspace_on_frame1_is_never_used(tp):
    # The cube rises 1 cm (identical pixels, less than the 2 cm move limit)
    # across the workspace's top while the brain thinks.
    frame, fix = tp.rt._localize("red cube")
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, float(fix.position[2]) + 0.005])
    tp.rt._localization_workspace_bounds = lambda: (lo, hi)
    refers = tp.brain.refers
    tp.brain.on_refer = lambda n: tp.scene.set(depth_shift=-0.01) if n == refers + 1 else None
    with pytest.raises(SkillError, match="outside the active arm workspace"):
        tp.rt._localize("red cube")


def test_reanchor_refuses_without_a_segmenter_or_across_image_sizes():
    from cascade.types import Detection

    det = Detection("red cube", 1.0, np.array(HOME_BOX, np.float32))
    a, b = synthetic_tabletop(), synthetic_tabletop()
    det.mask = _object_mask(a.rgb)  # SAM's mask on the frame the brain named it on
    bare = BrainDetector(base_url="http://b/v1", model="m")
    found = bare.reanchor(b, det, since=a)
    assert (found.detection, found.scene_changed) == (None, False) and "segmenter" in found.reason
    with_sam = BrainDetector(base_url="http://b/v1", model="m", segmenter=_ObjectSam())
    assert with_sam.reanchor(b, det, since=a).detection is not None
    other = synthetic_tabletop(width=320, height=240)
    found = with_sam.reanchor(other, det, since=a)
    assert (found.detection, found.scene_changed) == (None, True)


# -- config (B5, the keys B1 reads) ----------------------------------------------


def test_the_two_phase_config_keys_ship_with_their_defaults():
    from cascade.apps import demo
    from cascade.config import Cfg, load_demo_config

    for camera in ("mock", "isaac"):
        cfg = load_demo_config(camera=camera)
        assert cfg.perception_loop.semantic_max_age_s == 20.0
        assert cfg.detector.reanchor_local_diff == 8.0
        assert cfg.detector.reanchor_mask_agreement == 0.95
        assert cfg.detector.get("reanchor_mask_iou") is None, "replaced by reanchor_mask_agreement"
    det = demo._make_brain_detector(Cfg({}), {"type": "brain", "api_key_file": None, "reanchor_local_diff": "3.5",
                                              "reanchor_mask_agreement": "0.7"}, warm_up=False)
    assert det._reanchor_local_diff == 3.5
    assert det._reanchor_mask_agreement == 0.7
    assert BrainDetector(base_url="http://b/v1", model="m")._reanchor_local_diff == 8.0
    assert BrainDetector(base_url="http://b/v1", model="m")._reanchor_mask_agreement == 0.95



# -- fix round 1 (review I1): frame1's mask describes the object where it is NOW --


def _iso_cube(t):
    """Paint the cube iso-luminant orange: its grey level (~197) is the table's
    (190), so a slide changes the GREY image by about one level per cm."""
    t.scene.set(colour=ISO)
    t.rt.camera.get_frame()
    t.rt.camera.get_frame()


def _slid(px: int) -> tuple[int, int, int, int]:
    return (HOME_BOX[0] + px, HOME_BOX[1], HOME_BOX[2] + px, HOME_BOX[3])


def _first_timing(t) -> dict:
    return t.rt.trace.rows()[-1]["context"]["perception_timings"][0]


@pytest.mark.parametrize("cm", [0.75, 1.0, 1.5, 3.0])
def test_an_iso_luminant_slide_is_never_measured_on_the_stale_box(tp, cm):
    """Review I1: the cube slides while the brain thinks and SAM keeps its
    mask inside the box it is prompted with. On 199b6af the grey checks
    passed and that clipped mask put the cube up to 1.7 cm short of where it
    is. Now the slide is a scene change: one retry, then the NEW position."""
    _iso_cube(tp)
    new = _slid(round(cm * PX_PER_CM))
    _moves_once(tp, box=new)
    result = tp.rt.execute("localize_object", {"label": "orange cube"})
    assert result["ok"], result.get("error")
    assert tp.brain.refers == 2, "the slide is a scene change: one retry on a new frame"
    assert "scene_changed" in _first_timing(tp), _first_timing(tp)
    xy = np.asarray(result["position"])[:2]
    assert np.linalg.norm(xy - _cube_xy(tp, new)) < 0.003, "the NEW position, not biased toward the old one"


@pytest.mark.parametrize("px", [0, round(0.25 * PX_PER_CM)])
def test_a_still_or_barely_moved_iso_luminant_cube_is_accepted_at_once(tp, px):
    """No retry for a static scene or a 0.25 cm slide (SAM and camera jitter
    live there): the new checks must not refuse ordinary picks."""
    _iso_cube(tp)
    new = _slid(px)
    _moves_once(tp, box=new)
    frame, fix = tp.rt._localize("orange cube")
    assert tp.brain.refers == 1, "no retry"
    assert np.linalg.norm(np.asarray(fix.position)[:2] - _cube_xy(tp, new)) < 0.003


def test_a_sam_stuck_on_the_old_mask_is_caught_by_the_colour_check(tp):
    """Only the COLOUR local check sees this: the iso-luminant cube slides 3 cm
    (grey difference in the box ~2.6, under 8) and SAM answers a prompt at the
    old place with the old mask (a tracker's memory), so mask IoU and borders
    agree with frame0 and the geometry would be the OLD position."""
    _iso_cube(tp)
    home = synthetic_tabletop(box_px=HOME_BOX)
    home.rgb[np.all(home.rgb == RED, axis=-1)] = ISO
    stale = _object_mask(home.rgb)
    home_cx = (HOME_BOX[0] + HOME_BOX[2]) / 2

    def stuck(frame, boxes):
        x0, _, x1, _ = boxes[0]
        return [stale.copy()] if abs((x0 + x1) / 2 - home_cx) <= 5 else None

    tp.sam.default = stuck
    new = _slid(round(3.0 * PX_PER_CM))
    _moves_once(tp, box=new)
    frame, fix = tp.rt._localize("orange cube")
    assert tp.brain.refers == 2, "a scene change: one retry"
    assert np.linalg.norm(np.asarray(fix.position)[:2] - _cube_xy(tp, new)) < 0.003


def test_a_mask_on_part_of_the_object_is_a_scene_change(tp):
    """Only the MASK AGREEMENT sees this: the scene is still, but SAM masks
    just the left 60 % of the cube on frame1 (a face, a label). The mask lies
    on the box (box IoU 0.6 >= 0.5) and the image did not change, but only
    ~64 % of frame0's mask lies within the tolerance of it (< 0.95, the
    frame0 -> frame1 direction); measured, the cube would sit 1.5 cm left of
    its centre."""
    def part(frame, boxes):
        x0, y0, x1, y1 = (int(round(v)) for v in boxes[0])
        m = np.zeros(frame.rgb.shape[:2], dtype=bool)
        m[y0:y1, x0:x1] = _object_mask(frame.rgb)[y0:y1, x0:x1]
        xs = np.flatnonzero(m.any(0))
        m[:, xs[0] + int(0.6 * xs.size):] = False
        return [m]

    tp.sam.script[FRESH] = part
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    assert tp.brain.refers == 2, "a scene change: one retry"
    assert "agreement" in _first_timing(tp)["scene_changed"], _first_timing(tp)
    assert _first_timing(tp)["mask_agreement"] < 0.7, _first_timing(tp)
    assert np.linalg.norm(np.asarray(result["position"])[:2] - _cube_xy(tp, HOME_BOX)) < 0.003


def _with_strip(to_x):
    """SAM's normal answer plus a 2-px strip leaking right along the table
    from the cube's right edge to column `to_x(boxes)` (exclusive)."""
    def answer(frame, boxes):
        x0, y0, x1, y1 = (int(round(v)) for v in boxes[0])
        m = np.zeros(frame.rgb.shape[:2], dtype=bool)
        m[y0:y1, x0:x1] = _object_mask(frame.rgb)[y0:y1, x0:x1]
        m[228:230, HOME_BOX[2]:to_x(boxes)] = True
        return [m]
    return answer


def test_a_mask_reaching_past_its_prompt_box_is_a_scene_change(tp):
    """Only the BORDER check sees this: on frame1 the still cube's mask leaks
    a 2-px strip along the table past the right side of SAM's prompt (the
    brain's box grown by 20 %). The image and the mask IoU (~0.99) agree with
    frame0, but the mask now reaches a border frame0's did not: it describes
    more than the object the brain named."""
    tp.sam.script[FRESH] = _with_strip(lambda boxes: int(boxes[0][2]) + 3)
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    assert tp.brain.refers == 2, "a scene change: one retry"
    assert "border" in _first_timing(tp)["scene_changed"], _first_timing(tp)
    assert _first_timing(tp)["mask_iou"] >= 0.85
    assert len(_first_timing(tp)["reanchor_prompt_box"]) == 4, "SAM's prompt box is kept for evidence"
    assert np.linalg.norm(np.asarray(result["position"])[:2] - _cube_xy(tp, HOME_BOX)) < 0.003


def test_a_mask_already_past_the_box_on_the_named_image_is_no_change(tp):
    """The border rule compares with frame0: a mask that already reached past
    the box on the image the object was named on (a cable, a handle SAM
    keeps) is the same object on frame1, not a scene change."""
    tp.sam.default = _with_strip(lambda boxes: 420)
    frame, fix = tp.rt._localize("red cube")
    assert tp.brain.refers == 1, "no retry"


def test_the_local_check_box_grows_by_ten_percent_of_its_size():
    """_REANCHOR_BOX_GROWTH (the local image check's region): half of 10 % of
    the width/height on each side, outward to whole pixels, clipped to the
    image. SAM's prompt (20 %) is pinned by the device-mask test below."""
    from cascade.perception.brain_detector import _grown_box

    assert _grown_box((100, 100, 200, 150), (480, 640)) == (95, 97, 205, 153)
    assert _grown_box((0, 470, 40, 480), (480, 640)) == (0, 469, 42, 480)


def test_reanchor_refuses_a_detection_named_without_a_mask():
    """No frame0 mask, no mask consistency check: never re-anchored (fail
    closed: one retry, then ReanchorError)."""
    from cascade.types import Detection

    a, b = synthetic_tabletop(), synthetic_tabletop()
    det = Detection("red cube", 1.0, np.array(HOME_BOX, np.float32))
    found = BrainDetector(base_url="http://b/v1", model="m", segmenter=_ObjectSam()).reanchor(b, det, since=a)
    assert (found.detection, found.scene_changed) == (None, False)
    assert "named" in found.reason, found.reason


class _DeviceMask:
    """A CUDA tensor stand-in (strict GPU mode): torch-like `.any(dim)`,
    slicing and `.cpu().numpy()`. Converting it to numpy implicitly raises,
    like `np.asarray(cuda_tensor)`; `copies` records the element count of
    every host copy."""

    def __init__(self, a, copies):
        self._a = np.asarray(a, dtype=bool)
        self.copies = copies

    @property
    def shape(self):
        return self._a.shape

    def any(self, dim):
        return _DeviceMask(self._a.any(dim), self.copies)

    def __getitem__(self, index):
        return _DeviceMask(self._a[index], self.copies)

    def cpu(self):
        self.copies.append(self._a.size)
        return SimpleNamespace(numpy=lambda: self._a.copy())

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    def __bool__(self):
        raise RuntimeError("Boolean value of a CUDA tensor is ambiguous here")


def test_reanchor_checks_device_masks_without_copying_them_to_the_host():
    """Review minor: the CUDA path of the mask box and the new mask checks.
    The brain's box is loose (10 px around the cube): the re-anchored
    detection's bbox is frame1's MASK box, the grown prompt box is kept for
    evidence, and only row/column projections and the object's crop ever
    reach the host."""
    from cascade.perception.brain_detector import _mask_box
    from cascade.types import Detection

    copies: list[int] = []
    a, b = synthetic_tabletop(), synthetic_tabletop()
    obj = _object_mask(a.rgb)
    sam = _ObjectSam()
    sam.default = lambda frame, boxes: [_DeviceMask(_ObjectSam.masks(_ObjectSam(), frame, boxes)[0], copies)]
    det = Detection("red cube", 1.0, np.array((270, 190, 370, 270), np.float32), mask=_DeviceMask(obj, copies))
    detector = BrainDetector(base_url="http://b/v1", model="m", segmenter=sam)

    assert _mask_box(det.mask).tolist() == [279.0, 199.0, 361.0, 261.0]
    found = detector.reanchor(b, det, since=a)
    assert found.detection is not None, found.reason
    assert isinstance(found.detection.mask, _DeviceMask), "masks stay opaque"
    assert found.detection.bbox.tolist() == [279.0, 199.0, 361.0, 261.0], "frame1's mask box"
    assert found.prompt_box == (260, 182, 380, 278), "the prompt box (grown 20 %), for evidence"
    assert found.mask_iou == pytest.approx(1.0)
    assert found.mask_agreement == pytest.approx(1.0)
    assert copies and max(copies) <= 110 * 88 < obj.size, copies

    # A neighbour inside the prompt (4 px right of the cube): its component is
    # dropped on the host crop; the detection's mask is the cube's pixels only.
    copies.clear()
    beside = obj.copy()
    beside[205:255, 365:376] = True
    sam.default = lambda frame, boxes: [_DeviceMask(beside, copies)]
    found = detector.reanchor(b, det, since=a)
    assert found.detection is not None, found.reason
    assert np.array_equal(np.asarray(found.detection.mask, dtype=bool), obj), "the neighbour is dropped"
    assert found.detection.bbox.tolist() == [279.0, 199.0, 361.0, 261.0]
    assert found.mask_agreement == pytest.approx(1.0)
    assert copies and max(copies) <= 110 * 88 < obj.size, copies

    tight = Detection("red cube", 1.0, np.array((279, 199, 361, 261), np.float32), mask=_DeviceMask(obj, copies))
    part = obj.copy()
    part[:, 328:] = False  # SAM masks the left 60 % of the still cube on frame1 (frame0's re-mask: all of it)
    sam.default = lambda frame, boxes: [_DeviceMask(part if frame is b else obj, copies)]
    found = detector.reanchor(b, tight, since=a)
    assert (found.detection, found.scene_changed) == (None, True), found.reason
    assert found.mask_iou == pytest.approx(part.sum() / obj.sum(), abs=0.01)


# -- fix round 1 (review I2): bounded time to a refusal ------------------------


@pytest.mark.parametrize("frames", [3, 1])
def test_an_absent_object_on_a_slow_brain_is_refused_within_the_semantic_cap(tp, frames):
    """Review I2: nothing to find, a 6 s brain, two other cameras. 199b6af
    asked the brain on every frame retry and every camera (~62 s live, 10
    requests) before the fallback refused; now no new frame or camera starts
    once the localization is older than semantic_max_age_s (20 s live).
    `frames` = localize_frames: with 1 the cap stops the camera pass."""
    tp.rt.cfg._data["perception_loop"]["localize_frames"] = frames
    for scene in _side_cameras(tp, 2):
        scene.set(box=(0, 0, 0, 0))
    tp.brain.refer_s = 0.6
    n0 = _commands(tp)
    started = time.monotonic()
    with pytest.raises(tp.mod.SlowPerceptionError, match="Perception took"):
        tp.rt.skill_grasp_object("red cube")
    elapsed = time.monotonic() - started
    prompts = len(set(tp.brain.queries))  # brain requests per frame (one per prompt)
    one_round = 0.6 * prompts
    assert tp.brain.refers == 2 * prompts, ("two frames, then refused", tp.brain.queries)
    assert elapsed <= 20.0 * SCALE + one_round + 0.4, f"refused after {elapsed:.2f}s"
    assert _commands(tp) == n0


def test_no_scene_change_retry_starts_past_the_semantic_cap(tp):
    """The cap also ends the scene-change retry: the first frame misses (the
    cube is not there yet), the second names it and sees it move, and by then
    the localization is past the cap: refused, no third naming round."""
    tp.scene.set(box=(0, 0, 0, 0))
    tp.rt.camera.get_frame()
    tp.rt.camera.get_frame()
    tp.brain.refer_s = 0.7

    def on_refer(n):
        if n == 2:
            tp.scene.set(box=HOME_BOX)  # there for the next frame
        elif n == 3:
            tp.scene.set(box=MOVED_BOX)  # and moved while the brain names it

    tp.brain.on_refer = on_refer
    n0 = _commands(tp)
    with pytest.raises(tp.mod.SlowPerceptionError, match="Perception took"):
        tp.rt.skill_grasp_object("red cube")
    assert _commands(tp) == n0
    assert tp.brain.queries.count("red cube") == 2, tp.brain.queries


def test_an_absent_object_on_a_fast_brain_is_the_same_miss_as_main(make):
    """Below the cap nothing changes: every frame retry and camera runs and
    the miss is the single-frame path's (= main's), word for word."""
    outcomes = []
    for segmenter in (True, False):
        t = make(segmenter=segmenter)
        for scene in _side_cameras(t, 2):
            scene.set(box=(0, 0, 0, 0))
        with pytest.raises(SkillError) as err:
            t.rt.skill_grasp_object("red cube")
        outcomes.append((type(err.value), str(err.value), t.brain.refers))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] is SkillError and outcomes[0][2] >= 5


def test_the_single_frame_path_ignores_the_semantic_cap(make):
    """7.1: without a segmenter nothing of the two-phase path applies, the
    cap on retries included (its own 5 s rule bounds every frame)."""
    t = make(segmenter=False)
    t.rt.cfg._data["perception_loop"]["semantic_max_age_s"] = 1e-6
    for scene in _side_cameras(t, 2):
        scene.set(box=(0, 0, 0, 0))
    with pytest.raises(SkillError, match="no detections") as err:
        t.rt.skill_grasp_object("red cube")
    assert not isinstance(err.value, t.mod.SlowPerceptionError)
    assert t.brain.refers == 5 * len(set(t.brain.queries)), "3 frames and 2 cameras, as on main"


# -- fix round 1 (review minor): every localization's timing in the trace -------


def test_pick_and_place_reports_the_objects_timing_and_traces_every_localization(tp):
    blue = (200, 60, 40)
    tp.scene.set(dest=(400, 290, 460, 350, blue))
    tp.rt.camera.get_frame()
    tp.rt.camera.get_frame()

    def region(query, bgr):
        b, r = bgr[..., 0].astype(int), bgr[..., 2].astype(int)
        return (b - r > 60) if "blue" in query else (r - b > 60)

    tp.brain.region = region
    tp.arm.object_stop_frac = 0.5
    result = tp.rt.execute("pick_and_place", {"object": "red cube", "destination": "blue square"})
    timings = tp.rt.trace.rows()[-1]["context"]["perception_timings"]
    queries = [t["query"] for t in timings]
    assert "red cube" in queries and "blue square" in queries, (queries, result.get("error"))
    assert queries.index("red cube") < queries.index("blue square")
    assert result["perception_timing"] == [t for t in timings if t["query"] == "red cube"][-1]



def test_pick_and_place_reports_the_objects_timing_when_the_destination_comes_first(tp):
    """Review minor (fix round 2): the eye-in-hand (ROS 2) pick_and_place
    localizes the DESTINATION from the overview before the object. The
    result's perception_timing is still the object's: matched by the query
    the call names (`object` / `label`), not by order."""
    blue = (200, 60, 40)
    tp.scene.set(dest=(400, 290, 460, 350, blue))
    tp.rt.camera.get_frame()
    tp.rt.camera.get_frame()

    def region(query, bgr):
        b, r = bgr[..., 0].astype(int), bgr[..., 2].astype(int)
        return (b - r > 60) if "blue" in query else (r - b > 60)

    tp.brain.region = region
    tp.arm.object_stop_frac = 0.5
    tp.rt._native_wrist_camera = lambda: True  # the overview-destination flow of the eye-in-hand rig
    result = tp.rt.execute("pick_and_place", {"object": "red cube", "destination": "blue square"})
    timings = tp.rt.trace.rows()[-1]["context"]["perception_timings"]
    queries = [t["query"] for t in timings]
    assert queries[0] == "blue square" and "red cube" in queries, (queries, result.get("error"))
    assert result["perception_timing"] == [t for t in timings if t["query"] == "red cube"][-1]


# -- fix round 2 (re-review): still scenes pass, moved objects are still caught --

GREEN = (40, 160, 40)  # BGR: a second object, coloured, never red
SIZES = {"80x60": HOME_BOX, "40x30": (300, 215, 340, 245), "24x18": (308, 221, 332, 239)}


def _red_only(query, bgr):
    """The brain names the red cube only (never the green neighbour)."""
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    return (r - b > 60) & (r - g > 60)


def _whole_objects(frame, boxes):
    """A SAM that returns whole objects: every coloured connected component
    with a pixel inside the prompt box, never clipped at the box."""
    obj = _object_mask(frame.rgb).astype(np.uint8)
    _, labels = cv2.connectedComponents(obj, connectivity=8)
    h, w = obj.shape
    out = []
    for x0, y0, x1, y1 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        xa, ya = max(int(np.floor(x0)), 0), max(int(np.floor(y0)), 0)
        xb, yb = min(int(np.ceil(x1)), w), min(int(np.ceil(y1)), h)
        ids = np.unique(labels[ya:yb, xa:xb])
        m = np.isin(labels, ids[ids != 0])
        out.append(m if m.any() else None)
    return out


SAM_KINDS = {"clip": None, "whole": _whole_objects}  # None: _ObjectSam's own answer, clipped at the prompt


def _beside(gap: int):
    """A green 30x50 px object `gap` px right of the cube's last column (360)."""
    return (361 + gap, 205, 391 + gap, 255, GREEN)


def _morphed(px: int):
    """SAM's normal (clipped) answer grown (px > 0) or shrunk (px < 0) by |px| pixels."""
    def answer(frame, boxes):
        m = _ObjectSam.masks(_ObjectSam(), frame, boxes)[0]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(px) + 1, 2 * abs(px) + 1))
        op = cv2.dilate if px > 0 else cv2.erode
        return [op(m.astype(np.uint8), k).astype(bool)]
    return answer


def _tight(box, k: int):
    """The brain's box `k` px inside the object painted at `box` (with its
    1 px skirt) on every side: measured on the true object, not on the JPEG
    the brain decodes (its chroma bleeds a pixel or two)."""
    x0, y0, x1, y1 = box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1

    def region(query, bgr):
        m = np.zeros(bgr.shape[:2], dtype=bool)
        m[y0 + k:y1 - k, x0 + k:x1 - k] = True
        return m
    return region


def _show(t, box=None, **scene):
    t.scene.set(**({"box": box} if box is not None else {}), **scene)
    t.rt.camera.get_frame()
    t.rt.camera.get_frame()


@pytest.mark.parametrize("sam", sorted(SAM_KINDS))
@pytest.mark.parametrize("gap", [2, 4, 8, 16])
def test_a_still_neighbour_inside_the_prompt_margin_is_not_a_scene_change(tp, gap, sam):
    """Re-review Important: a still neighbour inside SAM's grown prompt box
    joined frame1's mask (the clipping SAM's part of it, or the whole object),
    reached the box border and refused a STILL scene twice. Now only frame1
    components within 2 px of frame0's mask are the object: accepted at once,
    measured exactly as the single-frame path measures that frame."""
    from cascade.perception.grounding import localize_object

    tp.brain.region = _red_only
    tp.sam.default = SAM_KINDS[sam]
    _show(tp, neighbour=_beside(gap))
    frame, fix = tp.rt._localize("red cube")
    assert tp.brain.refers == 1, "a still scene: no retry"
    green = np.all(frame.rgb == GREEN, axis=-1)
    assert green.any() and not (np.asarray(fix.detection.mask, dtype=bool) & green).any(), "the neighbour is dropped"
    single = localize_object(frame, "red cube", tp.detector, tp.rt.extrinsics)
    assert np.allclose(fix.position, single.position, atol=1e-3), (fix.position, single.position)
    assert np.allclose(fix.extent, single.extent, atol=2e-3), (fix.extent, single.extent)


@pytest.mark.parametrize("sam", sorted(SAM_KINDS))
def test_a_neighbour_moving_into_contact_is_still_caught(tp, sam):
    """The component filter keeps whatever is CONNECTED to the object: a
    neighbour that slides into contact while the brain thinks becomes part
    of the target's component on frame1 and is a scene change (one retry;
    it moved away again, so the retry measures the cube alone). Filtering
    pixel by pixel would keep its first 2 px and measure the cube with them."""
    tp.brain.region = _red_only
    tp.sam.default = SAM_KINDS[sam]
    _show(tp, neighbour=_beside(16))
    tp.brain.on_refer = lambda n: tp.scene.set(neighbour=_beside(0)) if n == 1 else None
    orig, seen = tp.detector.reanchor, []

    def spy(frame, det, **kw):
        seen.append(orig(frame, det, **kw))
        if len(seen) == 1:
            _show(tp, neighbour=_beside(16))  # and away again before the retry's frame0
        return seen[-1]

    tp.detector.reanchor = spy  # instance attribute: the class still defines reanchor
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    assert seen[0].scene_changed and seen[0].detection is None, seen[0].reason
    assert tp.brain.refers == 2, "the contact is a scene change: one retry"
    assert "scene_changed" in _first_timing(tp), _first_timing(tp)
    assert np.linalg.norm(np.asarray(result["position"])[:2] - _cube_xy(tp, HOME_BOX)) < 0.003


@pytest.mark.parametrize(("size", "px"), [(size, px) for size in sorted(SIZES) for px in (-2, -1, 1, 2)]
                         + [("80x60", -3), ("80x60", 3)])
def test_sam_mask_jitter_on_a_still_object_is_accepted(tp, size, px):
    """Re-review minor: SAM's frame1 mask 1-2 px larger or smaller than
    frame0's (real SAM between two calls) read a pixel IoU under 0.85 on
    small objects. The size-aware agreement (tol max(2 px, 3 % of the
    diagonal), both ways) accepts it on 80x60, 40x30 and 24x18 objects, and
    3 px on the 80x60 one (its tolerance is 3.1 px: 3 % of its diagonal)."""
    _show(tp, box=SIZES[size])
    tp.sam.script[FRESH] = _morphed(px)
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    timing = result["perception_timing"]
    assert tp.brain.refers == 1, timing
    assert timing["mask_agreement"] >= 0.95, timing


@pytest.mark.parametrize(("size", "k"), [(size, k) for size in sorted(SIZES) for k in (2, 3)])
def test_a_tight_brain_box_on_a_still_object_is_accepted(tp, size, k):
    """Re-review minor + fix round 3: the brain's box k px inside the object
    on every side and the SAM that clips at its prompt: the naming mask
    lacks a k px ring. Round 2 compared it with frame1's (grown prompt) mask
    and refused k = 3 on 40x30 and 24x18 (agreement 0.92 / 0.93); frame0 is
    now re-masked with frame1's prompt, so both masks are the whole object:
    accepted at once, measured whole. Without the 6 px prompt floor the
    24x18 object's frame1 mask reached the 20 % prompt's 2 px border."""
    _show(tp, box=SIZES[size])
    tp.brain.region = _tight(SIZES[size], k)
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    timing = result["perception_timing"]
    assert tp.brain.refers == 1, timing
    assert timing["mask_agreement"] >= 0.95, timing
    assert np.linalg.norm(np.asarray(result["position"])[:2] - _cube_xy(tp, SIZES[size])) < 0.003


def test_a_mask_grown_onto_the_table_is_a_scene_change(tp):
    """Only the frame1 -> frame0 direction of the agreement sees this: on a
    still scene SAM's frame1 mask is the cube plus a 5 px halo of table,
    inside the prompt box (no border reached) and covering all of frame0's
    mask, but ~9 % of its pixels lie farther than the tolerance from it."""
    tp.sam.script[FRESH] = _morphed(5)
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    assert tp.brain.refers == 2, "a scene change: one retry"
    assert "agreement" in _first_timing(tp)["scene_changed"], _first_timing(tp)
    assert np.linalg.norm(np.asarray(result["position"])[:2] - _cube_xy(tp, HOME_BOX)) < 0.003


def test_mask_jitter_between_the_frames_is_not_a_move(tp):
    """Re-review (the 2 cm move check): frame0's masks 1 px eroded, frame1's
    1 px grown onto the table. The two fixes' centres read a ~2 cm move
    (frame1's cloud takes in table pixels); the medians of the depth points
    inside each mask eroded by 2 px agree."""
    tp.sam.script[NAME] = tp.sam.script[REF] = _morphed(-1)
    tp.sam.script[FRESH] = _morphed(1)
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    timing = result["perception_timing"]
    assert tp.brain.refers == 1, timing
    assert timing["moved_m"] < 0.005, timing


def test_a_3cm_rise_is_still_caught_through_mask_jitter(tp):
    """The robust move check still sees a real move: the cube alone rises
    3 cm (identical pixels) while the brain thinks, on top of the same 1 px
    mask jitter: refused, no motion."""
    for attempt in (0, RETRY):
        tp.sam.script[NAME + attempt] = tp.sam.script[REF + attempt] = _morphed(-1)
        tp.sam.script[FRESH + attempt] = _morphed(1)
    tp.brain.on_refer = lambda n: tp.scene.toggle("lift", 0.0, 0.03)
    n0 = _commands(tp)
    result = tp.rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is False and "SceneChangedError" in result["error"], result.get("error")
    assert result["perception_timing"]["moved_m"] == pytest.approx(0.03, abs=0.005)
    assert _commands(tp) == n0



def test_median_point_erodes_falls_back_for_thin_masks_and_uses_the_frames_pose():
    """median_point, the move check's position: the median inside the mask
    eroded by 2 px (the skirt and any edge pixel on the table never count);
    the whole mask when erosion leaves fewer than min_points pixels (a 3 px
    wide strip); the frame's own camera pose when it carries one."""
    from cascade.perception.grounding import Extrinsics, median_point
    from cascade.types import Detection

    frame = synthetic_tabletop()
    ext = Extrinsics(T=np.eye(4))
    cube = _object_mask(frame.rgb)
    grown = cv2.dilate(cube.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # 1 px onto the table
    top = median_point(frame, Detection("cube", 1.0, np.array(HOME_BOX, np.float32), mask=cube), ext)
    assert top[2] == pytest.approx(0.55), "the top face only: the skirt is eroded away"
    edge = median_point(frame, Detection("cube", 1.0, np.array(HOME_BOX, np.float32), mask=grown), ext)
    assert np.allclose(edge, top, atol=1e-3), "a pixel of table at the edge changes nothing"

    strip = np.zeros_like(cube)
    strip[150:160, 100:103] = True  # 3 px wide: nothing survives a 2 px erosion
    frame.depth_m[150:160, 100:103] = 0.5
    thin = median_point(frame, Detection("strip", 1.0, np.array((100, 150, 103, 160), np.float32), mask=strip), ext)
    assert np.isfinite(thin).all() and thin[2] == pytest.approx(0.5), "the un-eroded mask"

    # A 4 px wide object whose frame1 mask took in 2 px of table on each
    # side: half its pixels are table. Eroded, the median is the object's top.
    frame.depth_m[100:160, 200:204] = 0.5
    wide = np.zeros_like(cube)
    wide[100:160, 198:206] = True
    rod = median_point(frame, Detection("rod", 1.0, np.array((198, 100, 206, 160), np.float32), mask=wide), ext)
    assert rod[2] == pytest.approx(0.5), "the 2 px of table on each side are eroded away"

    moved = np.eye(4)
    moved[:3, 3] = (1.0, 2.0, 3.0)
    frame.T_base_cam = moved
    posed = median_point(frame, Detection("cube", 1.0, np.array(HOME_BOX, np.float32), mask=cube), ext)
    assert np.allclose(posed - top, (1.0, 2.0, 3.0)), "the frame's own pose, as fix_from_detection uses"



def test_a_mask_part_within_two_px_of_the_named_mask_is_the_object():
    """The component filter keeps what comes within 2 px of the old mask (the
    old mask dilated by 2 px), not only what overlaps it: a part of SAM's
    mask 1 px off the object (a handle, a neighbour pressed against it) is
    judged with the object. Here it is 10 px wide: the agreement refuses it
    as a scene change, where dropping it would have measured the object
    without it."""
    from cascade.types import Detection

    a, b = synthetic_tabletop(), synthetic_tabletop()
    obj = _object_mask(a.rgb)  # columns 279..360
    near = obj.copy()
    near[205:255, 362:372] = True  # column 361 empty: 2 px from the object
    sam = _ObjectSam()
    sam.default = lambda frame, boxes: [near.copy()]
    det = Detection("red cube", 1.0, np.array((270, 190, 370, 270), np.float32), mask=obj)
    found = BrainDetector(base_url="http://b/v1", model="m", segmenter=sam).reanchor(b, det, since=a)
    assert (found.detection, found.scene_changed) == (None, True), found.reason
    assert found.dropped_components == 0 and "agreement" in found.reason, found

    # Fix round 3: the 2 px reach is measured from frame0's same-prompt
    # re-mask (the whole object), not from a naming mask a tight brain box
    # clipped 3 px inside it (from which the part lies 5 px away).
    clipped = cv2.erode(obj.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    tight = Detection("red cube", 1.0, np.array((282, 202, 358, 258), np.float32), mask=clipped)
    sam.default = lambda frame, boxes: [(near if frame is b else obj).copy()]
    found = BrainDetector(base_url="http://b/v1", model="m", segmenter=sam).reanchor(b, tight, since=a)
    assert (found.detection, found.scene_changed) == (None, True), found.reason
    assert found.dropped_components == 0, found


# -- fix round 3: frame0 is re-masked with frame1's prompt (like with like) ------


def test_frame0_is_re_masked_with_frame1s_prompt_before_frame1_is_taken(tp):
    """Round-2 concern 1: frame0's naming mask came from the brain's box and
    frame1's from the grown prompt, so a box that cuts the object read as a
    change. Now SAM re-masks frame0 with the SAME prompt frame1 gets, BEFORE
    frame1 is captured: that call (here 0.6 s, past the 0.5 s scaled limit)
    never counts against frame1's age."""
    def slow(frame, boxes):
        time.sleep(0.6)
        return None  # then the normal mask

    tp.sam.script[REF] = slow
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    assert tp.brain.refers == 1
    (f_name, b_name), (f_ref, b_ref), (f_fresh, b_fresh) = tp.sam.calls
    assert f_ref == f_name < f_fresh, "the re-mask is on the brain's frame, the re-anchor on a newer one"
    assert np.array_equal(b_ref, b_fresh), "the same prompt on both frames"
    assert (b_ref[0, :2] < b_name[0, :2]).all() and (b_ref[0, 2:] > b_name[0, 2:]).all(), "the brain's box, grown"
    timing = result["perception_timing"]
    assert timing["reference_s"] >= 0.6 and timing["frame1_age_s"] < 5.0 * SCALE, timing


@pytest.mark.parametrize("call", ["frame0", "frame1"])
@pytest.mark.parametrize(("size", "px"), [(size, px) for size in sorted(SIZES) for px in (-2, -1, 1, 2)])
def test_sam_jitter_on_a_still_iso_luminant_object_is_accepted(tp, size, px, call):
    """Round-2 probe: the iso cube's JPEG box is ~1 px tight, so its naming
    mask lacked a ring and 2 px of frame1 jitter on 40x30 read agreement
    0.93 (refused twice on a still scene). Against frame0's same-prompt
    re-mask only the jitter is left: accepted at once, whichever frame's
    mask jitters."""
    _show(tp, box=SIZES[size], colour=ISO)
    tp.sam.script[REF if call == "frame0" else FRESH] = _morphed(px)
    result = tp.rt.execute("localize_object", {"label": "orange cube"})
    assert result["ok"], result.get("error")
    timing = result["perception_timing"]
    assert tp.brain.refers == 1, timing
    assert timing["mask_agreement"] >= 0.95, timing


def _blob(frame, boxes):
    """SAM's normal answer plus the table to its right, connected to it: the
    mask's box overlaps the brain's by IoU ~0.25."""
    m = _ObjectSam.masks(_ObjectSam(), frame, boxes)[0]
    m[200:260, HOME_BOX[2]:600] = True
    return [m]


REF_FAILURES = {**{k: SAM_FAILURES[k] for k in ("none", "empty", "outside")}, "blob": _blob}
REF_FAILURE_REASONS = {**SAM_FAILURE_REASONS, "blob": "IoU"}


@pytest.mark.parametrize("failure", sorted(REF_FAILURES))
def test_a_failed_re_mask_of_frame0_is_a_failed_re_anchor(tp, failure):
    """Fix round 3: no usable same-prompt mask on frame0 (none, empty, none of
    it on the object, or its box off the brain's) means nothing to compare
    frame1 with: a failed re-anchor (one retry with a fresh refer, then
    ReanchorError), never a scene change, and frame1 is never masked."""
    tp.sam.script[REF] = tp.sam.script[2 * REF] = REF_FAILURES[failure]  # a failed re-mask skips frame1's call
    n0 = _commands(tp)
    result = tp.rt.execute("grasp_object", {"label": "red cube"})
    error = str(result.get("error"))
    assert result["ok"] is False and "ReanchorError" in error, error
    assert "could not re-anchor 'red cube' on a fresh image" in error, error
    assert REF_FAILURE_REASONS[failure] in error and "named on" in error, error
    assert _commands(tp) == n0, "no motion"
    assert tp.brain.refers == 2 and len(tp.sam.calls) == 2 * REF, "two namings and re-masks, no frame1 call"
    timings = tp.rt.trace.rows()[-1]["context"]["perception_timings"]
    assert len(timings) == 2 and all("reanchor_failed" in t and "reference_s" in t for t in timings), timings
    assert not any("frame1_age_s" in t for t in timings), "no frame1 is taken without a re-mask to compare it with"


def test_an_error_in_the_re_mask_is_a_failed_re_anchor(tp):
    """Whatever a detector's reanchor_reference raises is a failed re-anchor
    (fail closed: one retry with a fresh refer, then ReanchorError naming
    it), never an escaped exception or a fallback."""
    def boom(frame, det):
        raise RuntimeError("SAM lost its CUDA context")

    tp.detector.reanchor_reference = boom  # instance attribute: the class still defines it
    n0 = _commands(tp)
    with pytest.raises(tp.mod.ReanchorError, match="RuntimeError: SAM lost its CUDA context"):
        tp.rt.skill_grasp_object("red cube")
    assert _commands(tp) == n0
    assert tp.brain.refers == 2


def test_a_re_mask_on_part_of_the_named_object_is_never_measured(tp):
    """Comparing like with like must not let a consistent SAM error through:
    with the grown prompt SAM masks only the cube's left 60 %, on frame0 and
    frame1 alike, so the two masks agree. The re-mask has to cover the mask
    the object was named with (>= reanchor_mask_agreement of its pixels
    within the tolerance), else a failed re-anchor: measured, the cube would
    sit 1.5 cm left of its centre."""
    def part_with_grown_prompt(frame, boxes):
        if boxes[0][0] > HOME_BOX[0] - 4:
            return None  # the naming call (the brain's box): the whole cube
        m = _ObjectSam.masks(_ObjectSam(), frame, boxes)[0]
        xs = np.flatnonzero(m.any(0))
        m[:, xs[0] + int(0.6 * xs.size):] = False
        return [m]

    tp.sam.default = part_with_grown_prompt
    n0 = _commands(tp)
    with pytest.raises(tp.mod.ReanchorError, match="covers only") as err:
        tp.rt.skill_grasp_object("red cube")
    assert "named on" in str(err.value), err.value
    assert _commands(tp) == n0
    assert tp.brain.refers == 2


def test_the_move_check_measures_frame0_on_its_re_mask(tp):
    """The 2 cm move check's frame0 position comes from the same-prompt
    re-mask as well: the brain's box cuts 20 px off the cube's left side, so
    the naming mask's median sits ~3 px (~2.7 mm) right of the re-mask's
    (both are clipped at their prompts). Measured on the re-mask (fix round
    5: clipped to the naming mask exactly as frame1's mask is), a still cube
    reads no move at all."""
    x0, y0, x1, y1 = HOME_BOX[0] - 1, HOME_BOX[1] - 1, HOME_BOX[2] + 1, HOME_BOX[3] + 1

    def cut_left(query, bgr):
        m = np.zeros(bgr.shape[:2], dtype=bool)
        m[y0:y1, x0 + 20:x1] = True
        return m

    tp.brain.region = cut_left
    result = tp.rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"], result.get("error")
    timing = result["perception_timing"]
    assert tp.brain.refers == 1, timing
    assert timing["moved_m"] < 0.001, timing


def test_two_phase_needs_a_class_defined_re_mask_too():
    """can_reanchor (and LockedDetector's) require the CLASS to define both
    reanchor and reanchor_reference: the runtime re-masks frame0 before it
    takes frame1, so a detector with reanchor alone keeps the single-frame
    path."""
    from cascade.perception.detector import can_reanchor
    from cascade.perception.world import LockedDetector

    class OnlyReanchor:
        def detect(self, frame, classes=None):
            return []

        def reanchor(self, frame, det, *, since, reference=None):
            raise AssertionError("never called")

    class Both(OnlyReanchor):
        def reanchor_reference(self, frame, det):
            raise AssertionError("never called")

    assert not can_reanchor(OnlyReanchor()) and not can_reanchor(LockedDetector(OnlyReanchor()))
    assert can_reanchor(Both()) and can_reanchor(LockedDetector(Both()))
    with pytest.raises(TypeError, match="reanchor_reference"):
        LockedDetector(OnlyReanchor()).reanchor_reference(None, None)



# -- fix round 5: the geometry is the re-anchored mask clipped to the naming mask --


def _localize_timed(t, query):
    """One localization: ((frame, fix) or the exception it raised, its
    perception_timing dicts as execute() collects them)."""
    t.rt._call_perception_timings = []
    try:
        outcome = t.rt._localize(query)
    except SkillError as exc:
        outcome = exc
    finally:
        timings, t.rt._call_perception_timings = t.rt._call_perception_timings, None
    return outcome, timings


def _main_fix(t, frame, query, sam_kind):
    """Main's single-frame fix of `frame`: a fresh fake brain with the test
    brain's box rule and a fresh SAM of `sam_kind` (the runtime's own fakes
    keep their call counts)."""
    from cascade.perception.grounding import localize_object

    brain, seg = _ImageBrain(), _ObjectSam()
    brain.region, seg.default = t.brain.region, SAM_KINDS[sam_kind]
    detector = BrainDetector(base_url="http://b/v1", model="m", client=brain, segmenter=seg)
    return localize_object(frame, query, detector, t.rt.extrinsics)


def _next_to(box, gap):
    """A green 30 px wide object `gap` table columns right of the cube at
    `box` (0: touching it, 8-connected; negative: painted over its last
    columns), over the middle two thirds of its rows."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    return (x1 + gap, y0 + h // 6, x1 + gap + 30, y1 - h // 6, GREEN)


def _loose(box, k):
    """The brain's box `k` px outside the object painted at `box` (with its
    1 px skirt) on every side."""
    x0, y0, x1, y1 = box[0] - 1 - k, box[1] - 1 - k, box[2] + 1 + k, box[3] + 1 + k

    def region(query, bgr):
        m = np.zeros(bgr.shape[:2], dtype=bool)
        m[y0:y1, x0:x1] = True
        return m
    return region


#: The acceptance bar of fix round 5 for a scene the arm acts on: an accepted
#: fix lies within 0.2 cm (XY) of main's single-frame fix of the returned frame.
MAIN_XY_M = 0.002


def _as_main(t, query, sam_kind, *, accept: bool):
    """Localize once. A refusal is allowed unless `accept`; an accepted fix
    must be main's single-frame fix of the returned frame within MAIN_XY_M
    (XY). Returns (outcome, timings, XY error vs main or None)."""
    outcome, timings = _localize_timed(t, query)
    if isinstance(outcome, SkillError):
        assert not accept, f"a still scene refused: {outcome} ({timings})"
        return outcome, timings, None
    frame, fix = outcome
    main = _main_fix(t, frame, query, sam_kind)
    err = float(np.linalg.norm(np.asarray(fix.position)[:2] - np.asarray(main.position)[:2]))
    assert err <= MAIN_XY_M, f"accepted {err * 100:.2f} cm (XY) from main's single-frame fix ({timings})"
    return outcome, timings, err


TOUCHING = {"still touch": (0, 0), "still 1 px": (1, 1), "into contact": (16, 0), "to 1 px": (16, 1),
            "3 px onto the cube": (16, -3)}


@pytest.mark.parametrize("sam", sorted(SAM_KINDS))
@pytest.mark.parametrize("case", sorted(TOUCHING))
@pytest.mark.parametrize("size", sorted(SIZES))
def test_a_touching_neighbour_never_enters_the_geometry(tp, size, case, sam):
    """Round-3 re-review Important: a neighbour touching the cube (or sliding
    into contact, or onto it) joined the grown-prompt masks on BOTH frames
    (the clipping SAM's part of it, or the whole object), which then agreed:
    accepted with the neighbour in the geometry, 0.2-0.3 cm and an extent
    0.5-1.3 cm off main's single-frame fix. The comparisons stay as they
    were; the GEOMETRY is frame1's mask clipped to the naming mask dilated by
    the agreement tolerance, so it is main's within 0.2 cm. A still scene is
    accepted; the whole-object SAM's 24x18 case is refused as before (its
    naming mask already holds the neighbour: the grown-prompt re-mask's box
    fails the IoU check)."""
    box = SIZES[size]
    first, last = TOUCHING[case]
    _show(tp, box=box, neighbour=_next_to(box, first))
    tp.brain.region = _red_only
    tp.sam.default = SAM_KINDS[sam]
    if first != last:
        _moves_once(tp, neighbour=_next_to(box, last))
    still = first == last and not (sam == "whole" and size == "24x18")
    outcome, timings, _ = _as_main(tp, "red cube", sam, accept=still)
    if still:
        assert tp.brain.refers == 1, timings
        if sam == "clip":
            frame, fix = outcome
            red = np.all(frame.rgb == RED, axis=-1)
            # The largest tolerance any re-mask here can get (its box is at
            # most the prompt box), plus the 2 px the JPEG box may be loose.
            w, h = box[2] - box[0] + 2, box[3] - box[1] + 2
            tol = max(2.0, 0.03 * float(np.hypot(1.2 * w + 12, 1.2 * h + 12)))
            far = cv2.distanceTransform((~red).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) > tol + 2
            assert not (np.asarray(fix.detection.mask, dtype=bool) & far).any(), (
                "no part of the neighbour beyond the tolerance sliver")
            assert timings[0]["mask_clipped_px"] > 0, timings


@pytest.mark.parametrize("sam", sorted(SAM_KINDS))
@pytest.mark.parametrize("loose", [0, 2])
@pytest.mark.parametrize("gap", [0, 1, 2])
@pytest.mark.parametrize("size", sorted(SIZES))
def test_a_loose_box_beside_a_neighbour_is_measured_as_main(tp, size, gap, loose, sam):
    """Round-3 re-review probe: the brain's box 0-2 px loose beside a
    neighbour 0-2 px away. Accepted only within 0.2 cm of main's
    single-frame fix (a box reaching the neighbour gives a whole-object SAM
    the neighbour on main too: main's own error, not the re-anchor's)."""
    box = SIZES[size]
    _show(tp, box=box, neighbour=_next_to(box, gap))
    tp.brain.region = _loose(box, loose)
    tp.sam.default = SAM_KINDS[sam]
    _as_main(tp, "red cube", sam, accept=False)


def test_the_move_check_measures_both_frames_on_the_clipped_masks(tp):
    """A neighbour as big as the cube, in contact: SAM keeps the brain's
    box for the naming mask (cube only) but merges the neighbour under the
    grown prompt on frame0's re-mask and on frame1 alike (the comparisons
    agree: a still scene). Measured on the unclipped re-mask, frame0's
    median sits ~3 cm right of frame1's clipped one: a false move, refused
    twice. Both medians come from masks clipped to the naming mask: no move,
    accepted at once, as main measures the cube."""
    x0, y0, x1, y1 = HOME_BOX
    _show(tp, neighbour=(x1, y0, x1 + 60, y1, GREEN))
    tp.brain.region = _red_only
    for call in (REF, FRESH):
        tp.sam.script[call] = _whole_objects
    outcome, timings, _ = _as_main(tp, "red cube", "clip", accept=True)
    assert tp.brain.refers == 1 and timings[0]["moved_m"] < 0.002, timings
    assert timings[0]["mask_agreement"] == pytest.approx(1.0), "the comparisons see the merged masks"


@pytest.mark.parametrize(("size", "k"), [("80x60", k) for k in (1, 2, 3, 5, 8)]
                         + [(size, k) for size in ("40x30", "24x18") for k in (1, 2, 3)])
@pytest.mark.parametrize("sam", sorted(SAM_KINDS))
def test_a_tight_brain_box_is_accepted_at_once_and_measured_as_main(tp, size, k, sam):
    """A brain box 1-8 px inside the object (80x60) or 1-3 px (40x30, 24x18):
    accepted at once as on c1be487 (the comparisons use the same-prompt
    re-mask, not the clip: a reference clipped to a naming mask cut more than
    twice the tolerance would refuse at the border), and the clipped geometry
    is main's single-frame fix within 0.2 cm."""
    box = SIZES[size]
    _show(tp, box=box)
    tp.brain.region = _tight(box, k)
    tp.sam.default = SAM_KINDS[sam]
    _, timings, _ = _as_main(tp, "red cube", sam, accept=True)
    assert tp.brain.refers == 1 and timings[0]["mask_agreement"] >= 0.95, timings


@pytest.mark.parametrize(("size", "px"), [("80x60", 2), ("80x60", 5), ("120x90", 3), ("120x90", 6)])
def test_an_iso_slide_is_measured_where_the_cube_is_now(tp, size, px):
    """Round-4 iso slides: an iso-luminant cube slides 2-6 px (0.18-0.55 cm)
    while the brain thinks. Accepted (at once, or after one retry), the
    clipped geometry loses at most the leading sliver past the tolerance:
    within 0.2 cm of main's fix of the returned frame and of the cube's
    new centre."""
    box = {"120x90": (260, 195, 380, 285)}.get(size) or SIZES[size]
    _show(tp, box=box, colour=ISO)
    new = (box[0] + px, box[1], box[2] + px, box[3])
    _moves_once(tp, box=new)
    outcome, timings, _ = _as_main(tp, "orange cube", "clip", accept=True)
    frame, fix = outcome
    assert np.linalg.norm(np.asarray(fix.position)[:2] - _cube_xy(tp, new)) <= MAIN_XY_M, timings


SLACK_SLIDES = {"long bar 240x12, 16 px": ((200, 230, 440, 242), 16),
                "cube 160x120, 12 px": ((240, 180, 400, 300), 12)}


@pytest.mark.parametrize("case", sorted(SLACK_SLIDES))
def test_an_object_sliding_within_the_agreement_slack_is_measured_where_it_is(tp, case):
    """The agreement lets up to 5 % of frame1's pixels lie past the tolerance:
    on a long or large object that is a slide well past the tolerance (an
    iso bar 240 px long, tolerance 7 px, slid 16 px along its length; a
    160x120 cube, tolerance 6 px, slid 12 px), under the prompt border and
    the colour check: accepted at once, as on c1be487. Those leading pixels
    are new on frame1 (far from frame0's re-mask): the geometry keeps them,
    so the object is measured where it is now (cutting everything past the
    naming mask's tolerance put it 0.3-0.4 cm back toward where it was)."""
    box, px = SLACK_SLIDES[case]
    _show(tp, box=box, colour=ISO)
    new = (box[0] + px, box[1], box[2] + px, box[3])
    _moves_once(tp, box=new)
    outcome, timings, _ = _as_main(tp, "orange cube", "clip", accept=True)
    frame, fix = outcome
    assert tp.brain.refers == 1 and 0.95 <= timings[0]["mask_agreement"] < 1.0, timings
    assert np.linalg.norm(np.asarray(fix.position)[:2] - _cube_xy(tp, new)) <= MAIN_XY_M, timings


def test_a_slow_re_mask_past_the_semantic_cap_is_refused_before_frame1(tp):
    """Round-4 re-review minor: the semantic cap is checked again after
    frame0's re-mask, before the fresh frame is taken: a 25 s re-mask (2.5 s
    scaled, cap 2.0) is SlowPerceptionError naming the semantic step, with no
    SAM call on a fresh frame, no retry and no motion."""
    def slow(frame, boxes):
        time.sleep(2.5)
        return None  # then the normal mask

    tp.sam.script[REF] = slow
    n0 = _commands(tp)
    with pytest.raises(tp.mod.SlowPerceptionError, match="semantic step") as err:
        tp.rt.skill_grasp_object("red cube")
    assert "re-mask" in str(err.value), err.value
    assert len(tp.sam.calls) == REF and tp.brain.refers == 1, "no SAM call on a fresh frame, no retry"
    assert _commands(tp) == n0


def test_reanchor_measures_the_clipped_mask_and_compares_the_unclipped_ones():
    """reanchor(): the component filter, border and agreement compare SAM's
    own masks (frame0's same-prompt re-mask and frame1's), exactly as before;
    the detection it returns (the geometry) is frame1's mask without the
    pixels farther than the agreement tolerance from the NAMING mask. A
    whole-object SAM merges a touching neighbour under the grown prompt on
    both frames: they agree (1.0), the geometry is the cube plus a sliver of
    at most the tolerance. reanchor_reference's `geometry` is frame0's
    re-mask clipped the same way (the move check's frame0 side); its
    `detection` keeps SAM's mask. Masks stay opaque: crops only."""
    from cascade.perception.brain_detector import _mask_box
    from cascade.types import Detection

    copies: list[int] = []
    a, b = synthetic_tabletop(), synthetic_tabletop()
    for frame in (a, b):
        frame.rgb[210:250, 361:391] = GREEN  # touching the cube's right side (8-connected)
    obj = _object_mask(a.rgb) & ~np.all(a.rgb == GREEN, axis=-1)
    naming = _DeviceMask(obj, copies)
    sam = _ObjectSam()
    sam.default = lambda frame, boxes: [_DeviceMask(_whole_objects(frame, boxes)[0], copies)]
    det = Detection("red cube", 1.0, np.array((279, 199, 361, 261), np.float32), mask=naming)
    detector = BrainDetector(base_url="http://b/v1", model="m", segmenter=sam)
    tol = max(2.0, 0.03 * float(np.hypot(391 - 279, 261 - 199)))  # the re-mask's box: cube + neighbour
    far = cv2.distanceTransform((~obj).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) > tol
    green = np.all(a.rgb == GREEN, axis=-1)

    ref = detector.reanchor_reference(a, det)
    assert ref.detection is not None, ref.reason
    assert (np.asarray(ref.detection.mask._a if isinstance(ref.detection.mask, _DeviceMask) else ref.detection.mask,
                       dtype=bool) & green).sum() == green.sum(), "the comparison's re-mask keeps SAM's mask"
    geometry = np.asarray(ref.geometry.mask, dtype=bool)
    assert geometry[obj].all() and not (geometry & far).any(), "frame0's geometry: clipped to the naming mask"
    assert ref.geometry.bbox.tolist() == _mask_box(geometry).tolist()

    copies.clear()
    found = detector.reanchor(b, det, since=a, reference=ref)
    assert found.detection is not None, found.reason
    assert found.mask_agreement == pytest.approx(1.0) and found.dropped_components == 0, found
    mask = np.asarray(found.detection.mask, dtype=bool)
    assert mask[obj].all(), "the whole cube stays"
    assert not (mask & far).any(), "nothing farther than the tolerance from the naming mask"
    assert found.clipped_px == int((green & far).sum()) > 0, found.clipped_px
    assert found.detection.bbox.tolist() == _mask_box(mask).tolist(), "the clipped mask's own box"
    assert copies and max(copies) <= 120 * 70 < obj.size, copies

    # Nothing to clip: the mask SAM gave stays the (opaque) detection mask.
    sam.default = lambda frame, boxes: [_DeviceMask(_ObjectSam.masks(_ObjectSam(), frame, boxes)[0] & obj, copies)]
    found = detector.reanchor(b, det, since=a)
    assert found.detection is not None and found.clipped_px == 0, found.reason
    assert isinstance(found.detection.mask, _DeviceMask), "masks stay opaque"
