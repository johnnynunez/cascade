"""The runtime's default-vocabulary passes use the brain detector's cache (Ruling 30).

With `detect_classes` set, describe_scene, _reobserve and the localization
vocabulary round ask for the WHOLE vocabulary: one brain refer per class. Run
synchronously every time, 7 classes at ~1 s each blow the 5 s analysed-image
age limit on every call. They go through `perception.detector.detect_vocabulary`
instead, which the brain detector serves from the (camera, tuple(classes))
entry the watcher keeps warm. An agent's own single query ("find the red
cube") stays a fresh, synchronous refer.

Time is scaled 10x down so the test stays short: each fake brain request
takes 0.1 s (~1 s live), the age limit is 0.5 s (5 s live), refresh_s 0.4 s.
No network, no GPU: the brain is a fake client, the rig is the mock stack.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from types import SimpleNamespace

import pytest

from cascade.perception.brain_detector import REFER_PROMPT, BrainDetector
from cascade.perception.world import LockedDetector
from fake_brain_stream import reply as answer

SCALE = 0.1  # seconds in this test per second live
VOCAB = ["red cube", "banana", "disc", "box", "bowl", "cup", "green cube"]


class _SlowBrain:
    """OpenAI-compatible client: every request takes 1 s (scaled); "red cube"
    answers the cube's box, every other class "[]". Records (thread, query)."""

    def __init__(self):
        self.box_1000: list[int] | None = None
        self.requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"][1]["text"]
        query = next((q for q in VOCAB if prompt == REFER_PROMPT.format(q=q)), prompt)
        with self._lock:
            self.requests.append((threading.current_thread().name, query))
        time.sleep(1.0 * SCALE)
        reply = "[]"
        if query == "red cube" and self.box_1000 is not None:
            reply = json.dumps([{"label": "red cube", "bbox_2d": self.box_1000}])
        return answer(kwargs, reply)  # streamed for a background refresh

    def foreground(self) -> list[str]:
        """Queries asked on the caller's thread (not a background refresh)."""
        with self._lock:
            return [q for name, q in self.requests if not name.startswith("brain-refresh")]


@pytest.fixture
def brain_runtime(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.detector import MockDetector
    from cascade.skills import runtime as runtime_mod

    brain = _SlowBrain()
    detector = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", api_key_file=None, client=brain,
                             refresh_s=4.0 * SCALE, max_cache_age_s=30.0 * SCALE,
                             idle_refresh_s=30.0 * SCALE)
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    # The 5 s analysed-image age limit, scaled like the brain.
    monkeypatch.setattr(runtime_mod, "_analyzed_frame_age_s",
                        functools.partial(runtime_mod._analyzed_frame_age_s, max_age_s=5.0 * SCALE))
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["detect_classes"] = list(VOCAB)
    cfg._data["perception_loop"] = {"enabled": False}  # no watcher: the runtime's own calls only
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        assert type(runtime.detector) is LockedDetector and runtime.detector._detector is detector
        # The brain boxes the red cube where the mock scene painted it.
        frame = runtime.observe()
        (cube,) = MockDetector().detect(frame)
        h, w = frame.rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in cube.bbox)
        brain.box_1000 = [round(x1 * 1000 / w), round(y1 * 1000 / h), round(x2 * 1000 / w), round(y2 * 1000 / h)]
        yield runtime, brain
    finally:
        demo.shutdown_runtime(runtime, arm)


def test_describe_scene_with_a_vocabulary_is_served_from_the_cache(brain_runtime):
    runtime, brain = brain_runtime
    first = runtime.skill_get_observation()
    # Cold: ONE synchronous pass over the vocabulary (7 requests, slower than
    # the age limit), then the retry on a newer frame is a track step.
    assert brain.foreground() == VOCAB, "the first observation's brain cost is one vocabulary pass"
    assert "observation_retry" in first
    assert [o["label"] for o in first["objects_visible"]] == ["red cube"]

    second = runtime.skill_get_observation()  # no SlowPerceptionError, no retry, no brain wait
    assert "observation_retry" not in second
    assert [o["label"] for o in second["objects_visible"]] == ["red cube"]
    assert second["observation_frame"]["frame_age_s"] < 5.0 * SCALE
    assert brain.foreground() == VOCAB, "the second observation asked the brain on the caller's thread"


def test_a_single_query_localize_still_asks_the_brain_fresh(brain_runtime):
    runtime, brain = brain_runtime
    runtime.skill_get_observation()  # the vocabulary entry exists now
    for _ in range(2):  # every time: an agent's query is never served from a cache
        asked = len(brain.foreground())
        result = runtime.skill_localize_object("red cube")
        assert brain.foreground()[asked:] == ["red cube"], "the agent's query was not a fresh refer"
        assert result["detected_as"] == "red cube"


class _Recording:
    """A detector whose class defines detect_vocabulary; records every call."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def detect(self, frame, classes=None):
        self.calls.append(("detect", classes))
        return []

    def detect_vocabulary(self, frame, classes=None):
        self.calls.append(("vocabulary", classes))
        return []

    def set_classes(self, classes):
        pass


@pytest.fixture
def recording_runtime(tmp_path):
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["detect_classes"] = list(VOCAB)
    cfg._data["perception_loop"] = {"enabled": False}
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    recording = _Recording()
    runtime.detector = LockedDetector(recording)
    try:
        yield runtime, recording
    finally:
        demo.shutdown_runtime(runtime, arm)


def test_every_default_vocabulary_pass_goes_through_detect_vocabulary(recording_runtime, monkeypatch):
    from cascade.types import SkillError

    runtime, recording = recording_runtime
    # A second fusing camera (the mock rig has one): _reobserve and the
    # localization loop over the other cameras ask it for the vocabulary too.
    side = SimpleNamespace(stream=SimpleNamespace(name="side", get_frame=runtime.camera.get_frame),
                           depth=runtime.depth, extrinsics=runtime.extrinsics)
    monkeypatch.setattr(runtime, "_other_cams", lambda: [side])
    runtime.skill_get_observation()
    assert recording.calls == [("vocabulary", VOCAB)]
    runtime._reobserve(frames=1)
    assert recording.calls[1:] == [("vocabulary", VOCAB)] * 2, recording.calls  # primary, then the side camera
    monkeypatch.setattr(runtime, "_vlm_ground_fix", lambda frame, query: None)
    with pytest.raises(SkillError):
        runtime._localize("pink object")  # no noun: the localization vocabulary round, every view
    assert len(recording.calls) > 3
    assert all(c == ("vocabulary", VOCAB) for c in recording.calls[3:]), recording.calls


def test_a_noun_query_stays_a_plain_detect(recording_runtime, monkeypatch):
    from cascade.types import SkillError

    runtime, recording = recording_runtime
    monkeypatch.setattr(runtime, "_vlm_ground_fix", lambda frame, query: None)
    with pytest.raises(SkillError):
        runtime._localize("red cube")
    assert recording.calls and all(c == ("detect", ["red cube"]) or c == ("detect", ["cube"])
                                   for c in recording.calls), recording.calls


def test_an_agent_query_that_is_literally_vocabulary_is_a_fresh_refer():
    # The localization round kind is a flag, not its display name: an agent
    # asking for the word "vocabulary" gets a plain detect like any other noun.
    import numpy as np

    from cascade.perception.grounding import Extrinsics, localize_object
    from cascade.perception.mock_camera import synthetic_tabletop
    from cascade.types import SkillError

    recording = _Recording()
    frame = synthetic_tabletop()
    extrinsics = Extrinsics(mode="eye_to_hand", T=np.eye(4))
    for kwargs in ({}, {"prompts": ["vocabulary"]}):
        with pytest.raises(SkillError):
            localize_object(frame, "vocabulary", recording, extrinsics, **kwargs)
    assert recording.calls == [("detect", ["vocabulary"])] * 2, recording.calls
    with pytest.raises(SkillError):  # the configured vocabulary round still reads the cache
        localize_object(frame, "pink object", recording, extrinsics, vocab=["vocabulary"])
    assert recording.calls[2:] == [("vocabulary", ["vocabulary"])], recording.calls
