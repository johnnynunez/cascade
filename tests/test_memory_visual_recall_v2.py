"""B43 (ROADMAP #7, v2): visual recall from WATCHER detections, persisted across restarts.

B21 (#253) landed the opt-in memory embedder with two points left open:

1. crops came from LOCALIZATIONS only (`_note_localized` inside a call), never
   from what the always-on WorldWatcher sees -- an object nobody asked about
   had no remembered appearance;
2. the visual index was in-process only: a restart forgot every appearance.

Both are OPT-IN here and the shipped default is byte-identical:

* `memory.visual_recall_detections` -- the watcher hands the detections of a
  COMMITTED fusion to a `DetectionCropRecorder` (one crop per belief per frame,
  at most one per belief per `visual_recall_interval_s`, at most
  `visual_recall_max_per_tick` per tick, own ring capped at
  `visual_recall_max_detections`). Fusion is paused during motion skills, so a
  motion frame never yields a crop.
* `memory.persist_episodic` (+ `CASCADE_EPISODIC`, `CASCADE_EPISODIC_PATH`) --
  `EpisodicMemory.save_visual/load_visual`, modelled on BeliefStore.save/load:
  wall-clock times, max-age drop, a minimum apparent age, atomic writes; every
  restored hit says it is REMEMBERED.

Premise / golden tests pass on main by design (they pin the cause and the
default path); the rest are RED on main. The embedders are the shipped
`HashEmbedder` and B21's deterministic `_JointStub`; no weights are loaded and
no recall QUALITY is measured. Recall stays advisory: it never confirms an
outcome and never aims motion.
"""
from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from test_memory_visual_recall import _Clock, _crop, _JointStub, _noisy
from test_same_colour_beliefs import BLUE_BGR, RED_BGR, T_CAM2BASE, _InstanceDetector, _scene

from cascade.memory import EpisodicMemory
from cascade.memory.beliefs import BeliefStore, FrameObservation, ObjectBelief
from cascade.memory.embedder import EmbedderUnavailable, HashEmbedder
from cascade.perception.grounding import Extrinsics
from cascade.perception.world import WatchedCamera, WorldWatcher

try:
    from cascade.memory.episodic import DetectionCropRecorder
except ImportError:  # main: keep every test collectable so each one FAILS on its own
    DetectionCropRecorder = None

RED = (0, 0, 220)  # BGR
BLUE = (220, 0, 0)
GREEN = (0, 200, 0)
WALL = 1_800_000_000.0  # a fixed wall clock for the save side
B21_HIT_KEYS = {"score", "kind", "label", "age_s", "text", "verdict", "step"}
B21_LOOKS_LIKE_NOTE = ("remembered appearance matches (cosine >= the embedder's floor), "
                       "not a current observation: localize_object before acting on one")


# ── fixtures ────────────────────────────────────────────────────────────


class _Watch:
    """The REAL WorldWatcher fusion path, one synthetic RGB-D frame per tick."""

    def __init__(self, store, recorder=None):
        self.store = store
        self.detector = _InstanceDetector()
        self._frame = None
        stream = SimpleNamespace(name="top", latest=lambda: self._frame,
                                 set_overlay=lambda **kw: None)
        self.cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f),
                                 Extrinsics(T=T_CAM2BASE))
        kw = {} if recorder is None else {"visual_recall": recorder}
        self.watcher = WorldWatcher([self.cam], self.detector, store, **kw)
        self.n = 0
        self.t = 100.0

    def tick(self, cubes=((0.24, 0.06), (0.24, 0.16)), labels=("red cube", "blue cube"),
             colors=(RED_BGR, BLUE_BGR)):
        self.n += 1
        self.t += 0.35
        frame, masks, _ = _scene(list(cubes), colors=list(colors))
        frame.frame_id, frame.t = self.n, self.t
        self.detector.show(masks, labels=list(labels))
        self._frame = frame
        self.watcher._tick(self.cam)
        return frame


def _obs(label, bbox, color=None, pos=(0.3, 0.0, 0.02)):
    return FrameObservation(label, np.asarray(pos, float), 0.9, color=color,
                            bbox=None if bbox is None else np.asarray(bbox, float))


def _image(*blocks, h=120, w=160):
    """A grey image with coloured blocks: (colour, x0, y0, size)."""
    img = np.full((h, w, 3), 128, np.uint8)
    for color, x0, y0, size in blocks:
        img[y0:y0 + size, x0:x0 + size] = color
    return img


def _recorder(mem, **kw):
    kw.setdefault("interval_s", 30.0)
    kw.setdefault("max_per_tick", 2)
    return DetectionCropRecorder(mem, **kw)


# ── premises (pass on main by design) ───────────────────────────────────


def test_premise_update_frame_returns_the_belief_each_observation_fused_into():
    """Dedupe-per-belief rests on this: one frame's detections of ONE object
    (an open-vocabulary second name over the same pixels) come back as the
    SAME belief, a second object as another."""
    store = BeliefStore()
    frame, masks, _ = _scene([(0.24, 0.06), (0.24, 0.16)], colors=[RED_BGR, BLUE_BGR])
    box = lambda m: np.array([np.nonzero(m)[1].min(), np.nonzero(m)[0].min(),  # noqa: E731
                              np.nonzero(m)[1].max() + 1, np.nonzero(m)[0].max() + 1], float)
    obs = [FrameObservation("red cube", np.array([0.24, 0.06, 0.02]), 0.9, color="red",
                            bbox=box(masks[0]), mask=masks[0]),
           FrameObservation("cube", np.array([0.24, 0.06, 0.02]), 0.6, color="red",
                            bbox=box(masks[0]), mask=masks[0]),
           FrameObservation("blue cube", np.array([0.24, 0.16, 0.02]), 0.9, color="blue",
                            bbox=box(masks[1]), mask=masks[1])]
    fused = store.update_frame(obs, t=1.0)
    assert fused[0] is fused[1] and fused[2] is not fused[0]
    assert len(store.all(now=1.0)) == 2


def test_premise_without_the_flag_a_watcher_tick_indexes_nothing():
    """Golden: the watcher of main (and of the shipped default) never touches
    the visual index, embedder or not -- B21 indexed localizations only."""
    store = BeliefStore()
    w = _Watch(store)
    mem = EpisodicMemory(embedder=HashEmbedder())
    for _ in range(3):
        w.tick()
    assert len(store.all(now=w.t)) == 2
    assert mem.visual_stats()["entries"] == 0
    assert getattr(w.watcher, "_visual_recall", None) is None


def test_premise_the_visual_index_is_per_process():
    """The cause of 'lost on restart': a new EpisodicMemory starts empty."""
    first = EpisodicMemory(embedder=HashEmbedder())
    first.add("action", "localize_object(label=red cube) -> ok", crops={"red cube": _crop(RED)})
    assert first.visual_stats()["entries"] == 1
    second = EpisodicMemory(embedder=HashEmbedder())
    assert second.visual_stats()["entries"] == 0
    assert second.recall_visual(_crop(RED)) == []


def test_golden_session_hits_keep_exactly_the_b21_keys():
    clock = _Clock()
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock)
    mem.add("action", "pick -> ok", rgb=_image((RED, 20, 20, 30)), crops={"red cube": _crop(RED)},
            data={"verdict": "confirmed"})
    clock.t += 3.0
    for hit in mem.recall_visual(_crop(RED), k=5):
        assert set(hit) == B21_HIT_KEYS, hit


def test_golden_shipped_config_leaves_both_features_off(demo_cfg):
    mem = demo_cfg.memory
    assert str(mem.get("visual_recall_detections", False)).lower() in ("false", "0")
    assert str(mem.get("persist_episodic", False)).lower() in ("false", "0")


@needs_pin
def test_golden_an_embedder_alone_adds_no_recorder_no_store_and_no_teardown_stage(demo_cfg, tmp_path):
    """B21's configuration (embedder on, B43 flags off) keeps B21's runtime:
    no watcher recorder, no episodic file, the same teardown stages."""
    from cascade.apps.demo import build_runtime, shutdown_runtime

    demo_cfg._data["memory"]["embedder"] = {"backend": "hash"}
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        assert isinstance(runtime.memory.embedder, HashEmbedder)
        assert getattr(runtime.watcher, "_visual_recall", None) is None
        assert getattr(runtime, "episodic_path", None) is None
    finally:
        receipt = shutdown_runtime(runtime, arm)
    assert [s["stage"] for s in receipt["stages"]] == [
        "park", "beliefs", "watcher", "stream_server", "viewer", "cameras", "arms"]
    assert not list(tmp_path.rglob("episodic*.json"))


# ── watcher-detection crops: the memory side ────────────────────────────


def test_detection_crops_have_their_own_kind_and_stay_out_of_the_text_and_frame_rings():
    clock = _Clock()
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock)
    assert mem.add_detection_crop("red cube", _crop(RED), text="watcher saw red cube") is True
    assert mem.add_detection_crop("blue cube", _crop(BLUE), text="watcher saw blue cube") is True
    clock.t += 4.0
    hits = mem.recall_visual(_noisy(_crop(RED)), k=2, kinds=("detection",))
    assert [(h["kind"], h["label"]) for h in hits] == [("detection", "red cube"),
                                                        ("detection", "blue cube")]
    assert hits[0]["text"] == "watcher saw red cube" and hits[0]["age_s"] == pytest.approx(4.0)
    assert hits[0]["step"] is None and hits[0]["score"] > 0.9
    assert mem.digest() == "(memory empty)" and mem.memory_frames(4) == []
    assert mem.visual_stats()["entries"] == 2
    assert mem.visual_breakdown() == {"session": 0, "detections": 2, "restored": 0}


def test_an_embedder_fault_on_a_detection_crop_is_counted_never_raised():
    class _Broken(HashEmbedder):
        def embed_image(self, bgr):
            raise RuntimeError("device lost")

    mem = EpisodicMemory(embedder=_Broken())
    assert mem.add_detection_crop("red cube", _crop(RED), text="x") is False
    assert mem.visual_stats()["embed_errors"] == 1 and mem.visual_stats()["entries"] == 0


def test_the_detection_ring_is_bounded_and_never_evicts_session_entries():
    clock = _Clock()
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock, frame_horizon_s=60.0,
                         max_visual=8, max_detections=4)
    mem.add("action", "pick -> ok", rgb=_image((RED, 20, 20, 30)), crops={"red cube": _crop(RED)})
    for i in range(20):
        clock.t += 0.5
        mem.add_detection_crop(f"thing {i}", _crop(BLUE), text=f"watcher saw thing {i}")
    assert [e.label for e in mem._detections] == ["thing 16", "thing 17", "thing 18", "thing 19"]
    assert len(mem._det_index) == 4
    assert len(mem._index) == 2  # the motion frame and the localized crop survived
    assert mem.visual_stats()["entries"] == 6
    clock.t += 61.0  # the task-scale horizon applies to watcher crops too
    mem.prune()
    assert len(mem._detections) == 0 == len(mem._det_index)


def test_detections_without_an_embedder_are_an_explicit_refusal():
    with pytest.raises(EmbedderUnavailable):
        EpisodicMemory().add_detection_crop("red cube", _crop(RED), text="x")


# ── watcher-detection crops: the recorder ───────────────────────────────


def test_recorder_dedupes_per_belief_rate_limits_and_caps_each_tick():
    clock = _Clock()
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock)
    rec = _recorder(mem, interval_s=30.0, max_per_tick=2)
    rgb = _image((RED, 10, 10, 30), (BLUE, 60, 10, 30), (GREEN, 110, 10, 30))
    b1, b2, b3 = (ObjectBelief(label=n, position=np.zeros(3)) for n in ("red cube", "blue cube", "box"))
    obs = [_obs("red cube", (10, 10, 40, 40), "red"),
           _obs("cube", (12, 12, 38, 38), "red"),          # same belief, same frame
           _obs("blue cube", (60, 10, 90, 40), "blue"),
           _obs("box", (110, 10, 140, 40), "green")]
    fused = [b1, b1, b2, b3]
    assert rec.offer(rgb, obs, fused, source="top") == 2   # per-tick cap: b1, b2
    assert [e.label for e in mem._detections] == ["red cube", "blue cube"]
    assert mem._detections[0].event.text == "watcher saw red cube at (0.30, 0.00, 0.02) m [top]"
    clock.t += 10.0
    assert rec.offer(rgb, obs, fused, source="top") == 1   # b1/b2 inside their interval
    assert [e.label for e in mem._detections][-1] == "box"
    clock.t += 19.0                                         # b1/b2 29 s, b3 19 s
    assert rec.offer(rgb, obs, fused) == 0
    clock.t += 1.5                                          # b1/b2 due again, b3 not yet
    assert rec.offer(rgb, obs, fused) == 2
    assert [e.label for e in mem._detections] == ["red cube", "blue cube", "box",
                                                  "red cube", "blue cube"]
    assert mem._detections[-1].event.text == "watcher saw blue cube at (0.30, 0.00, 0.02) m"


def test_recorder_skips_unfused_and_unusable_detections():
    mem = EpisodicMemory(embedder=HashEmbedder())
    rec = _recorder(mem, max_per_tick=5)
    rgb = _image((RED, 10, 10, 30))
    b = [ObjectBelief(label=f"o{i}", position=np.zeros(3)) for i in range(5)]
    obs = [_obs("no belief", (10, 10, 40, 40)),
           _obs("no box", None),
           _obs("sliver", (10, 10, 11, 40)),              # 1 px wide
           _obs("off image", (500, 500, 600, 600)),
           _obs("ok", (10, 10, 40, 40), "red"),
           _obs("edge", (-10, -10, 30, 30))]              # clipped to the image
    assert rec.offer(rgb, obs, [None, *b]) == 2
    assert [e.label for e in mem._detections] == ["ok", "edge"]
    assert mem._detections[0].event.text.startswith("watcher saw red ok at")
    assert mem.visual_stats()["embed_errors"] == 0  # no empty crop ever reached the embedder


def test_a_belief_that_died_does_not_shadow_a_new_one_with_the_same_id():
    clock = _Clock()
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock)
    rec = _recorder(mem)
    rgb = _image((RED, 10, 10, 30))
    obs = [_obs("red cube", (10, 10, 40, 40))]
    b = ObjectBelief(label="red cube", position=np.zeros(3))
    assert rec.offer(rgb, obs, [b]) == 1
    stale = rec._last[id(b)]
    del b
    newcomer = ObjectBelief(label="red cube", position=np.zeros(3))
    rec._last[id(newcomer)] = stale  # simulate CPython reusing the dead belief's id
    assert rec.offer(rgb, obs, [newcomer]) == 1
    assert rec.offer(rgb, obs, [newcomer]) == 0


@pytest.mark.parametrize("bad", [{"interval_s": -1.0}, {"interval_s": math.nan},
                                 {"max_per_tick": 0}, {"max_per_tick": 1.5}])
def test_recorder_limits_fail_closed(bad):
    with pytest.raises(ValueError):
        _recorder(EpisodicMemory(embedder=HashEmbedder()), **bad)


def test_recorder_needs_an_embedder():
    with pytest.raises(ValueError, match="embedder"):
        _recorder(EpisodicMemory())


# ── watcher-detection crops: the REAL WorldWatcher tick ─────────────────


def test_watcher_indexes_crops_of_committed_detections_only():
    store = BeliefStore()
    mem = EpisodicMemory(embedder=HashEmbedder())
    w = _Watch(store, _recorder(mem))
    w.tick()
    assert sorted(e.label for e in mem._detections) == ["blue cube", "red cube"]
    assert all(e.kind == "detection" and e.event.text.startswith("watcher saw ")
               and e.event.text.endswith("[top]") for e in mem._detections)
    # the crop is THAT detection's pixels: a red crop recalls the red cube first
    hits = mem.recall_visual(_crop(RED_BGR), k=2, kinds=("detection",))
    assert [h["label"] for h in hits] == ["red cube", "blue cube"], hits
    hits = mem.recall_visual(_crop(BLUE_BGR), k=1, kinds=("detection",))
    assert hits[0]["label"] == "blue cube", hits
    # fusion paused (a motion skill is running): no belief update, no crop
    three = dict(cubes=((0.24, 0.06), (0.24, 0.16), (0.24, -0.04)),
                 labels=("red cube", "blue cube", "green cube"),
                 colors=(RED_BGR, BLUE_BGR, (40, 200, 40)))
    n = len(mem._detections)
    seen = sorted(b.observations for b in store.all(now=w.t))
    with w.watcher.paused():
        w.tick(**three)
    assert len(mem._detections) == n
    assert sorted(b.observations for b in store.all(now=w.t)) == seen
    # resumed: only the NEW object is due, the other two are inside their interval
    w.tick(**three)
    assert len(store.all(now=w.t)) == 3
    assert [e.label for e in mem._detections][n:] == ["green cube"]


def test_a_pause_that_begins_after_the_commit_embeds_nothing():
    store = BeliefStore()
    mem = EpisodicMemory(embedder=HashEmbedder())
    w = _Watch(store, _recorder(mem))
    commit = store.update_frame

    def commit_then_motion_starts(observations, t=None):
        out = commit(observations, t=t)
        w.watcher._pause_count += 1  # a motion skill took the pause right after the commit
        return out

    store.update_frame = commit_then_motion_starts
    w.tick()
    assert len(store.all(now=w.t)) == 2  # fused
    assert len(mem._detections) == 0     # but no embedding work on the motion's time


def test_a_recorder_fault_never_breaks_the_watcher_tick(capsys):
    class _Exploding:
        def offer(self, *a, **k):
            raise RuntimeError("embedder device lost")

    store = BeliefStore()
    w = _Watch(store, _Exploding())
    w.tick()
    assert len(store.all(now=w.t)) == 2 and w.watcher.last_update_t is not None
    w.tick()
    assert w.watcher.visual_recall_errors == 2
    err = capsys.readouterr().err
    assert err.count("visual recall: RuntimeError: embedder device lost") == 1  # logged once


# ── persistence: save / load ────────────────────────────────────────────


def _filled_memory(embedder=None):
    """Session entries at t=1000 (frame + localized crop, confirmed), a watcher
    crop at 1039.5, an explicit caller embedding; saved at clock 1040."""
    clock = _Clock(1000.0)
    mem = EpisodicMemory(embedder=embedder or HashEmbedder(), clock=clock)
    mem.add("action", "pick_and_place(object=red cube) -> ok", rgb=_image((RED, 20, 20, 30)),
            data={"verdict": "confirmed"}, crops={"red cube": _crop(RED)})
    clock.t = 1039.5
    mem.add_detection_crop("blue cube", _crop(BLUE), text="watcher saw blue cube")
    mem.add("note", "caller vector", embedding=np.ones(mem.embedder.dim))
    clock.t = 1040.0
    return mem


def test_visual_index_round_trips_with_wall_clock_ages(tmp_path):
    path = tmp_path / "episodic.json"
    src = _filled_memory()
    original = src.recall_visual(_crop(RED), k=1, kinds=("object",))[0]["score"]
    assert src.save_visual(path, now_wall=WALL) == 3  # the caller's own vector is not persisted
    blob = json.loads(path.read_text())
    assert blob["embedder"] == "hash-v1" and blob["dim"] == HashEmbedder().dim
    assert sorted(e["kind"] for e in blob["entries"]) == ["detection", "frame", "object"]

    # a NEW process: another monotonic origin, 100 s of wall clock later
    dst = EpisodicMemory(embedder=HashEmbedder(), clock=_Clock(50.0))
    assert dst.load_visual(path, now_wall=WALL + 100.0) == 3
    assert dst.visual_breakdown() == {"session": 0, "detections": 0, "restored": 3}
    obj = dst.recall_visual(_crop(RED), k=1, kinds=("object",))[0]
    assert obj["label"] == "red cube" and obj["verdict"] == "confirmed"
    assert obj["text"] == "pick_and_place(object=red cube) -> ok"
    assert obj["age_s"] == pytest.approx(140.0) and obj["step"] is None
    assert obj["restored"] is True and obj["state"] == "remembered"
    assert obj["score"] == pytest.approx(original, abs=0.03)
    det = dst.recall_visual(_crop(BLUE), k=1, kinds=("detection",))[0]
    assert det["label"] == "blue cube" and det["age_s"] == pytest.approx(100.5)
    frame = dst.recall_visual(_image((RED, 20, 20, 30)), k=1, kinds=("frame",))[0]
    assert frame["label"] is None and frame["restored"] is True
    # saved again (the next shutdown), the restored entries keep their wall-clock times
    again = tmp_path / "again.json"
    assert dst.save_visual(again, now_wall=WALL + 100.0) == 3
    walls = sorted(e["wall"] for e in json.loads(again.read_text())["entries"])
    assert walls == pytest.approx([WALL - 40.0, WALL - 40.0, WALL - 0.5], rel=0, abs=0.01)


def test_nothing_restored_reads_as_current_and_max_age_drops_before_the_floor(tmp_path):
    path = tmp_path / "episodic.json"
    _filled_memory().save_visual(path, now_wall=WALL)
    # saved half a second after the watcher crop: restored at the 2 s floor
    fresh = EpisodicMemory(embedder=HashEmbedder(), clock=_Clock(7.0))
    assert fresh.load_visual(path, now_wall=WALL) == 3
    det = fresh.recall_visual(_crop(BLUE), k=1, kinds=("detection",))[0]
    assert det["age_s"] == pytest.approx(EpisodicMemory.LOADED_MIN_AGE_S) == pytest.approx(2.0)
    # a wall clock that moved BACKWARDS reads as "just saved", never as the future
    back = EpisodicMemory(embedder=HashEmbedder(), clock=_Clock(7.0))
    assert back.load_visual(path, now_wall=WALL - 500.0) == 3
    assert min(h["age_s"] for h in back.recall_visual(_crop(BLUE), k=5)) == pytest.approx(2.0)
    # the age limit is tested BEFORE the floor: 0.5 s old with a 1 s limit is kept
    # (and then reads 2 s old); the 40 s old entries are past a 39 s limit
    aged = EpisodicMemory(embedder=HashEmbedder(), clock=_Clock(7.0))
    assert aged.load_visual(path, max_age_s=39.0, now_wall=WALL) == 1
    assert [h["label"] for h in aged.recall_visual(_crop(BLUE), k=5)] == ["blue cube"]
    assert aged.load_visual(path, max_age_s=1.0, now_wall=WALL) == 1
    assert aged.visual_breakdown()["restored"] == 1  # a reload replaces, never stacks
    assert aged.load_visual(path, max_age_s=1.0, now_wall=WALL + 1.5) == 0
    # default limit: 6 h, like the belief store
    assert EpisodicMemory.DEFAULT_MAX_AGE_S == 6 * 3600.0
    stale = EpisodicMemory(embedder=HashEmbedder())
    assert stale.load_visual(path, now_wall=WALL + 6 * 3600.0 + 1.0) == 0


def test_restored_entries_keep_their_own_horizon_and_cap(tmp_path):
    path = tmp_path / "episodic.json"
    _filled_memory().save_visual(path, now_wall=WALL)
    clock = _Clock(0.0)
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock, frame_horizon_s=60.0)
    assert mem.load_visual(path, max_age_s=3600.0, now_wall=WALL + 100.0) == 3
    clock.t += 30.0
    mem.add("note", "tick")  # any write prunes: 130-170 s old, past the 60 s task horizon
    assert mem.visual_breakdown()["restored"] == 3
    clock.t += 3600.0
    mem.prune()
    assert mem.visual_breakdown()["restored"] == 0 and len(mem._restored_index) == 0
    capped = EpisodicMemory(embedder=HashEmbedder(), max_visual=2)
    assert capped.load_visual(path, now_wall=WALL) == 2  # the newest two, oldest first
    assert [e.kind for e in capped._restored] == ["object", "detection"]
    assert all(e.restored for e in capped._restored)


def test_restored_and_session_hits_are_ranked_together(tmp_path):
    path = tmp_path / "episodic.json"
    _filled_memory().save_visual(path, now_wall=WALL)
    clock = _Clock(0.0)
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock)
    mem.load_visual(path, now_wall=WALL + 10.0)
    mem.add("action", "localize_object(label=green cube) -> ok", crops={"green cube": _crop(GREEN)})
    mem.add_detection_crop("red cube", _crop(RED), text="watcher saw red red cube")
    hits = mem.recall_visual(_crop(GREEN), k=10)
    assert hits[0]["label"] == "green cube" and "restored" not in hits[0]
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert {(h["kind"], h.get("restored", False)) for h in hits} == {
        ("object", False), ("detection", False), ("frame", True), ("object", True),
        ("detection", True)}
    reds = mem.recall_visual(_crop(RED), k=2, kinds=("object",))
    assert reds[0]["label"] == "red cube" and reds[0]["restored"] is True


def test_load_refuses_another_embedder_and_survives_bad_files(tmp_path):
    path = tmp_path / "episodic.json"
    _filled_memory().save_visual(path, now_wall=WALL)
    with pytest.raises(ValueError, match="hash-v1"):
        EpisodicMemory(embedder=_JointStub()).load_visual(path)
    blob = json.loads(path.read_text())
    blob["dim"] = 7
    (tmp_path / "dim.json").write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="dim"):
        EpisodicMemory(embedder=HashEmbedder()).load_visual(tmp_path / "dim.json")
    with pytest.raises(EmbedderUnavailable):
        EpisodicMemory().load_visual(path)
    with pytest.raises(EmbedderUnavailable):
        EpisodicMemory().save_visual(path)
    mem = EpisodicMemory(embedder=HashEmbedder())
    assert mem.load_visual(tmp_path / "missing.json") == 0
    (tmp_path / "corrupt.json").write_text("{not json")
    assert mem.load_visual(tmp_path / "corrupt.json") == 0
    # bad records (a vector of the wrong length, an unreadable time) are
    # skipped, the rest load
    blob = json.loads(path.read_text())
    blob["entries"][0]["vec"] = base64.b64encode(np.zeros(3, np.float16).tobytes()).decode()
    blob["entries"][1]["wall"] = "yesterday"
    (tmp_path / "bad_records.json").write_text(json.dumps(blob))
    assert mem.load_visual(tmp_path / "bad_records.json", now_wall=WALL) == 1


def test_save_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "episodic.json"
    mem = _filled_memory()
    assert mem.save_visual(path, now_wall=WALL) == 3
    before = path.read_text()
    assert [p.name for p in tmp_path.iterdir()] == ["episodic.json"]  # no temp leftovers

    def _crash(*a, **k):
        raise OSError("disk full")

    mem.add_detection_crop("green cube", _crop(GREEN), text="watcher saw green cube")
    with monkeypatch.context() as m:
        m.setattr(os, "replace", _crash)
        with pytest.raises(OSError):
            mem.save_visual(path, now_wall=WALL + 5.0)
    assert path.read_text() == before  # the old store was never half-overwritten
    assert [p.name for p in tmp_path.iterdir()] == ["episodic.json"]  # temp file removed
    nested = tmp_path / "a" / "b" / "episodic.json"
    assert mem.save_visual(nested, now_wall=WALL) == 4 and nested.exists()


def test_recall_memory_says_a_restored_match_is_remembered(tmp_path):
    from cascade.skills.runtime import SkillRuntime

    path = tmp_path / "episodic.json"
    src = EpisodicMemory(embedder=_JointStub(), clock=_Clock(10.0))
    src.add("action", "localize_object(label=red cube) -> ok", crops={"red cube": _crop(RED)})
    src.save_visual(path, now_wall=WALL)
    mem = EpisodicMemory(embedder=_JointStub())
    assert mem.load_visual(path, now_wall=WALL + 60.0) == 1
    out = SkillRuntime.skill_recall_memory(SimpleNamespace(memory=mem, beliefs=BeliefStore()), "red")
    hit = out["looks_like"][0]
    assert hit["label"] == "red cube" and hit["restored"] is True and hit["state"] == "remembered"
    assert hit["age_s"] >= 60.0
    assert out["looks_like_note"].startswith(B21_LOOKS_LIKE_NOTE)
    assert "restored from an earlier session" in out["looks_like_note"]
    # a session-only answer keeps B21's note verbatim
    live = EpisodicMemory(embedder=_JointStub())
    live.add("action", "localize_object(label=red cube) -> ok", crops={"red cube": _crop(RED)})
    out = SkillRuntime.skill_recall_memory(SimpleNamespace(memory=live, beliefs=BeliefStore()), "red")
    assert out["looks_like"] and out["looks_like_note"] == B21_LOOKS_LIKE_NOTE


# ── configuration and wiring ────────────────────────────────────────────


def test_conftest_isolates_the_episodic_store(tmp_path):
    assert os.environ["CASCADE_EPISODIC_PATH"] == str(tmp_path / "episodic.json")
    assert "CASCADE_EPISODIC" not in os.environ


def test_configs_document_every_new_flag_with_its_shipped_value():
    import yaml

    from cascade.config import CONFIG_DIR

    memory = yaml.safe_load((CONFIG_DIR / "demo.yaml").read_text())["memory"]
    assert {k: memory.get(k, "MISSING") for k in (
        "visual_recall_detections", "visual_recall_interval_s", "visual_recall_max_per_tick",
        "visual_recall_max_detections", "persist_episodic", "episodic_path",
        "episodic_max_age_s")} == {
        "visual_recall_detections": False, "visual_recall_interval_s": 30.0,
        "visual_recall_max_per_tick": 2, "visual_recall_max_detections": 128,
        "persist_episodic": False, "episodic_path": None, "episodic_max_age_s": 21600}


def test_switch_and_path_resolution(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.config import Cfg

    on, off = Cfg({"persist_episodic": True}), Cfg({})
    monkeypatch.delenv("CASCADE_EPISODIC", raising=False)
    assert demo._episodic_persist_enabled(off) is False
    assert demo._episodic_persist_enabled(on) is True
    assert demo._episodic_persist_enabled(Cfg({"persist_episodic": "false"})) is False
    monkeypatch.setenv("CASCADE_EPISODIC", "1")
    assert demo._episodic_persist_enabled(off) is True
    for value in ("0", "off", "false", "no"):
        monkeypatch.setenv("CASCADE_EPISODIC", value)
        assert demo._episodic_persist_enabled(on) is False, value
    assert demo._episodic_path(off) == tmp_path / "episodic.json"  # conftest's env
    monkeypatch.delenv("CASCADE_EPISODIC_PATH")
    assert demo._episodic_path(Cfg({"episodic_path": str(tmp_path / "x.json")})) == tmp_path / "x.json"
    assert demo._episodic_path(off) == demo.PACKAGE_ROOT / "runs" / "episodic.json"


def test_invalid_visual_recall_limits_fail_the_build_before_any_hardware(demo_cfg, tmp_path,
                                                                        monkeypatch):
    from cascade.apps import demo

    built = []

    def _no_arm(*a, **k):
        built.append(a)
        raise AssertionError("an arm was built before the visual recall limits were checked")

    monkeypatch.setattr(demo, "_build_arm", _no_arm)
    demo_cfg._data["memory"]["embedder"] = {"backend": "hash"}
    demo_cfg._data["memory"]["visual_recall_detections"] = True
    for key, bad in (("visual_recall_max_per_tick", 0), ("visual_recall_max_per_tick", 1.5),
                     ("visual_recall_max_per_tick", True), ("visual_recall_interval_s", -5.0),
                     ("visual_recall_interval_s", float("nan")), ("visual_recall_interval_s", "soon"),
                     ("visual_recall_interval_s", None),
                     ("visual_recall_max_detections", 0), ("episodic_max_age_s", -1.0)):
        cfg_mem = dict(demo_cfg._data["memory"])
        demo_cfg._data["memory"][key] = bad
        if key == "episodic_max_age_s":
            demo_cfg._data["memory"]["persist_episodic"] = True
        with pytest.raises(ValueError, match=key):
            demo.build_runtime(demo_cfg, tmp_path / "run")
        demo_cfg._data["memory"] = cfg_mem
    assert built == []


@needs_pin
def test_flags_without_an_embedder_warn_and_do_nothing(demo_cfg, tmp_path, capsys):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    demo_cfg._data["memory"].update(visual_recall_detections=True, persist_episodic=True)
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        assert getattr(runtime.watcher, "_visual_recall", None) is None
        assert getattr(runtime, "episodic_path", None) is None
    finally:
        shutdown_runtime(runtime, arm)
    err = capsys.readouterr().err
    assert "memory.visual_recall_detections needs memory.embedder" in err
    assert "memory.persist_episodic needs memory.embedder" in err
    assert not (tmp_path / "episodic.json").exists()


def _wait(pred, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not pred():
        time.sleep(0.05)
    return pred()


@needs_pin
def test_runtime_watcher_indexes_what_it_sees_without_any_call(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    demo_cfg._data["memory"].update(embedder={"backend": "hash"}, visual_recall_detections=True,
                                    visual_recall_interval_s=3600.0, visual_recall_max_per_tick=3,
                                    visual_recall_max_detections=64)
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        rec = runtime.watcher._visual_recall
        assert isinstance(rec, DetectionCropRecorder) and rec.memory is runtime.memory
        assert (rec.interval_s, rec.max_per_tick) == (3600.0, 3)
        assert runtime.memory.max_detections == 64
        assert _wait(lambda: runtime.beliefs.find("red cube") is not None)
        assert _wait(lambda: "red cube" in {e.label for e in runtime.memory._detections})
        time.sleep(1.0)  # a few more watcher ticks: one crop per belief per interval
        labels = [e.label for e in runtime.memory._detections]
        assert len(labels) == len(set(labels)) <= len(runtime.beliefs.all())
        assert all(e.kind != "detection" for e in runtime.memory._visual)  # own ring
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_runtime_persists_and_restores_the_visual_index_across_a_restart(demo_cfg, tmp_path, capsys):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    path = Path(os.environ["CASCADE_EPISODIC_PATH"])
    demo_cfg._data["memory"].update(embedder={"backend": "hash"}, persist_episodic=True,
                                    visual_recall_detections=True)
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run1")
    try:
        assert runtime.episodic_path == path
        assert _wait(lambda: len(runtime.memory._detections) > 0)
        assert runtime.execute("localize_object", {"label": "red cube"})["ok"]
        kinds = {e.kind for e in runtime.memory._visual} | {e.kind for e in runtime.memory._detections}
    finally:
        receipt = shutdown_runtime(runtime, arm)
    rows = {s["stage"]: s for s in receipt["stages"]}
    assert list(rows)[:3] == ["park", "beliefs", "episodic"] and rows["episodic"]["ok"] is True
    saved = json.loads(path.read_text())
    assert {e["kind"] for e in saved["entries"]} == kinds >= {"object", "detection"}
    assert f"appearance(s) -> {path}" in capsys.readouterr().out

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run2")
    try:
        restored = list(runtime.memory._restored)
        assert len(restored) == len(saved["entries"]) and all(e.restored for e in restored)
        assert f"remembered appearance(s) from {path}" in capsys.readouterr().out
        hits = runtime.memory.recall_visual(_crop(RED_BGR), k=50)
        old = [h for h in hits if h.get("restored")]
        assert {h["kind"] for h in old} >= {"object", "detection"}
        assert all(h["age_s"] >= EpisodicMemory.LOADED_MIN_AGE_S and h["state"] == "remembered"
                   for h in old)
    finally:
        shutdown_runtime(runtime, arm)

    # the configured max age reaches the load: 0 s keeps nothing from before
    demo_cfg._data["memory"]["episodic_max_age_s"] = 0.0
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run3")
    try:
        assert runtime.memory.visual_breakdown()["restored"] == 0
    finally:
        shutdown_runtime(runtime, arm)


def test_fleet_refuses_a_global_episodic_path_for_two_manipulation_robots(monkeypatch):
    from cascade.apps import fleet as app

    monkeypatch.setattr(app, "describe_robot", lambda cfg: None)
    monkeypatch.setattr(app, "RobotRuntime", lambda *a, **k: None)
    monkeypatch.setattr(app, "FleetRuntime", lambda passive: None)
    for key in ("CASCADE_BELIEFS_PATH", "CASCADE_GRASP_MEMORY_PATH", "CASCADE_ENVELOPE_PATH",
                "CASCADE_EPISODIC_PATH"):
        monkeypatch.delenv(key, raising=False)
    cfgs = [SimpleNamespace(robot_id=r, domains=SimpleNamespace(
        as_dict=lambda: {"arm": {"kind": "manipulation"}})) for r in ("a", "b")]
    app._validate_fleet(cfgs)  # nothing global: accepted
    monkeypatch.setenv("CASCADE_EPISODIC_PATH", "/tmp/shared-episodic.json")
    with pytest.raises(ValueError, match="global memory path"):
        app._validate_fleet(cfgs)


def test_fleet_refuses_two_robots_writing_one_episodic_file(tmp_path):
    from test_fleet_runtime import member

    from cascade.robotics.fleet import FleetRuntime

    first, _ = member("first", tmp_path)
    second, _ = member("second", tmp_path)
    directory = tmp_path / "stores"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    first.episodic_path, second.episodic_path = directory / "e.json", alias / "e.json"
    with pytest.raises(ValueError, match="shared.*file"):
        FleetRuntime({"first": first, "second": second})


def test_composed_manipulation_domains_get_their_own_episodic_store(tmp_path, monkeypatch):
    from cascade.apps import demo
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import load_robot_config

    seen = []

    class _Stop(Exception):
        pass

    def _capture(cfg, run_dir, **kw):
        seen.append(dict(cfg.memory.as_dict()))
        raise _Stop()

    monkeypatch.setattr(demo, "build_runtime", _capture)
    with pytest.raises(_Stop):
        build_robot_runtime(load_robot_config("fixed_so101_mock"), tmp_path)
    stores = tmp_path / "domains" / "manipulation" / "stores"
    assert seen[0]["episodic_path"] == str(stores / "episodic.json")
    assert seen[0]["beliefs_path"] == str(stores / "beliefs.json")
