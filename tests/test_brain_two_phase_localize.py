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
TIMING_KEYS = {"semantic_s", "frame0_age_at_reanchor_s", "reanchor_s", "frame1_age_s",
               "scene_diff_global", "scene_diff_local", "moved_m"}


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
    replaces the answer text, `error` is raised instead of answering."""

    def __init__(self):
        self.refer_s = 0.02
        self.on_refer = None
        self.reply: str | None = None
        self.error: BaseException | None = None
        self.refers = 0
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        url = content[0]["image_url"]["url"]
        with self._lock:
            self.refers += 1
            n = self.refers
        if self.on_refer is not None:
            self.on_refer(n)
        time.sleep(self.refer_s)
        if self.error is not None:
            raise self.error
        text = self.reply
        if text is None:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
            box = _pixel_box(_object_mask(img))
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
    masks, or a callable(frame, boxes) -> list (e.g. one that sleeps)."""

    def __init__(self):
        self.calls: list[tuple[int, np.ndarray]] = []
        self.script: dict[int, object] = {}

    def masks(self, frame, boxes):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.calls.append((int(frame.frame_id), boxes.copy()))
        scripted = self.script.get(len(self.calls))
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
        self.state = dict(box=HOME_BOX, depth_shift=0.0, patch=None, colour=None, K=None, T=None, capture=None)
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
    assert [c[0] for c in tp.sam.calls] == [semantic_frame, frame.frame_id], "one re-anchor, on the returned frame"
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

    tp.sam.script[2] = slow
    n0 = _commands(tp)
    with pytest.raises(tp.mod.SlowPerceptionError, match="Perception took"):
        tp.rt.skill_grasp_object("red cube")
    assert _commands(tp) == n0, "refused before any motion"
    assert len(tp.sam.calls) == 2 and tp.brain.refers == 1, "no retry after a slow analysis"


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
    # Same grey level (the image diffs pass), another colour: frame1's mask is
    # green, so the red cube the brain named is not what frame1 shows.
    _moves_once(tp, colour=(40, 121, 40))
    n0 = _commands(tp)
    with pytest.raises(SkillError) as err:
        tp.rt.skill_grasp_object("red cube")
    assert "green" in str(err.value), err.value
    assert _commands(tp) == n0
    assert len(tp.sam.calls) >= 2 and tp.sam.calls[1][0] > tp.sam.calls[0][0], "frame1 was re-anchored"


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


def _side_camera(t):
    """A second fusing camera with its own mock scene (the primary sees no cube)."""
    camera = MockCamera()
    camera.open()
    scene = _Scene(camera)
    t.scene.set(box=(0, 0, 0, 0))  # nothing on the primary's table
    t.rt.camera.get_frame()
    t.rt.camera.get_frame()
    side = SimpleNamespace(stream=SimpleNamespace(name="side", get_frame=camera.get_frame),
                           depth=t.rt.depth, extrinsics=t.rt.extrinsics, fuse=True)
    t.rt._other_cams = lambda: [side]
    return scene


def test_the_other_camera_pass_is_two_phase_too(tp):
    side = _side_camera(tp)
    seen = []
    orig = tp.detector.reanchor

    def spy(frame, det, *, since):
        seen.append((since.frame_id, frame.frame_id))
        return orig(frame, det, since=since)

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


def _corners(frame, boxes):
    """The four corner pixels of the cube's coloured region in the prompt box:
    the right colour and a box that IS the prompt box, but a cloud with fewer
    than min_points depth points."""
    x0, y0, x1, y1 = (int(v) for v in boxes[0])
    obj = np.zeros(frame.rgb.shape[:2], dtype=bool)
    obj[y0:y1, x0:x1] = _object_mask(frame.rgb)[y0:y1, x0:x1]
    ys, xs = np.nonzero(obj)
    m = np.zeros_like(obj)
    for y in (ys.min(), ys.max()):
        row = xs[ys == y]
        m[y, row.min()] = m[y, row.max()] = True
    return [m]


def _outside(frame, boxes):
    m = np.zeros(frame.rgb.shape[:2], dtype=bool)
    m[0:60, 0:80] = True  # table, far from the box
    return [m]


SAM_FAILURES = {
    "none": lambda frame, boxes: [None],
    "empty": lambda frame, boxes: [np.zeros(frame.rgb.shape[:2], dtype=bool)],
    "outside": _outside,
    "few_points": _corners,
}
SAM_FAILURE_REASONS = {"none": "no mask", "empty": "empty mask", "outside": "off the box",
                       "few_points": "valid depth points"}


@pytest.mark.parametrize("failure", sorted(SAM_FAILURES))
def test_no_usable_mask_on_frame1_is_refused_after_one_fresh_refer(tp, failure):
    tp.sam.script[2] = tp.sam.script[4] = SAM_FAILURES[failure]
    n0 = _commands(tp)
    with pytest.raises(SkillError, match="could not re-anchor 'red cube' on a fresh image") as err:
        tp.rt.skill_grasp_object("red cube")
    assert SAM_FAILURE_REASONS[failure] in str(err.value), err.value
    assert not isinstance(err.value, (tp.mod.SlowPerceptionError, tp.mod.SceneChangedError))
    assert _commands(tp) == n0, "no box fallback, no motion"
    assert tp.brain.refers == 2 and len(tp.sam.calls) == 4, "one retry with a fresh refer"


def test_one_failed_reanchor_is_retried_with_a_fresh_refer(tp):
    tp.sam.script[2] = SAM_FAILURES["none"]
    frame, fix = tp.rt._localize("red cube")
    assert tp.brain.refers == 2 and frame.frame_id == tp.sam.calls[3][0]


def test_a_reanchored_detection_without_a_mask_is_never_measured_as_its_box(tp):
    """Whatever a detector's reanchor() returns, the runtime measures frame1
    only through a mask: a box cloud includes table pixels."""
    from cascade.perception.brain_detector import ReanchorResult
    from cascade.types import Detection

    def maskless(frame, det, *, since):
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
    det = demo._make_brain_detector(Cfg({}), {"type": "brain", "api_key_file": None, "reanchor_local_diff": "3.5"},
                                    warm_up=False)
    assert det._reanchor_local_diff == 3.5
    assert BrainDetector(base_url="http://b/v1", model="m")._reanchor_local_diff == 8.0
