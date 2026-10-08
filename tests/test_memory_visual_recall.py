"""ROADMAP #7: embedder-backed episodic recall, action<->object consolidation
over tier-2 experience, and embedding retrieval for skill-library notes.

Everything here is OPT-IN. The golden tests at the bottom pin the shipped
default path (no embedder, no consolidation) to today's behaviour; they hold on
main by construction and exist so a later change cannot quietly alter it.

The embedders used are deterministic and dependency-free: the shipped
`HashEmbedder` (colour histogram + layout for images, stemmed hashed words for
text, NO shared image-text space) and a test-local `_JointStub` whose image and
text vectors live in one three-colour space, standing in for SigLIP/CLIP. None
of this measures semantic recall quality with real weights; memory and recall
stay advisory -- they never confirm an outcome or veto motion.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import needs_pin

from cascade.memory import EpisodicMemory

try:
    from cascade.memory.embedder import EmbedderUnavailable, HashEmbedder
except ImportError:  # pre-change tree: keep every test collectable so each FAILS on its own
    EmbedderUnavailable = HashEmbedder = None

RED = (0, 0, 220)      # BGR
BLUE = (220, 0, 0)
GREEN = (0, 200, 0)
GREY = (128, 128, 128)


def _scene(color=RED, x0=20, y0=45, size=30, h=120, w=160):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = GREY
    img[y0:y0 + size, x0:x0 + size] = color
    return img


def _crop(color=RED, size=40, border=6):
    img = np.empty((size, size, 3), np.uint8)
    img[:] = GREY
    img[border:size - border, border:size - border] = color
    return img


def _noisy(img, sigma=6.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(float) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _JointStub:
    """A deterministic JOINT image-text space over three colours (+ grey):
    an image maps to the fraction of red/green/blue-dominant pixels, a text to
    the colour words it names. Stands in for SigLIP/CLIP in CPU tests."""

    name = "joint-stub"
    joint_space = True
    dim = text_dim = 4
    text_image_floor = 0.5
    text_floor = 0.5

    def embed_image(self, bgr):
        px = np.asarray(bgr, float).reshape(-1, 3)
        b, g, r = px[:, 0], px[:, 1], px[:, 2]
        spread = px.max(1) - px.min(1)
        chroma = spread > 60
        v = np.array([
            np.mean(chroma & (r >= g) & (r >= b)),
            np.mean(chroma & (g > r) & (g >= b)),
            np.mean(chroma & (b > r) & (b > g)),
            np.mean(~chroma),
        ])
        return (v / np.linalg.norm(v)).astype(np.float32)

    def embed_text(self, text):
        words = text.lower().split()
        v = np.array([("red" in words), ("green" in words), ("blue" in words), 0.0], float)
        if not v.any():
            v[3] = 1.0
        return (v / np.linalg.norm(v)).astype(np.float32)


# ── EpisodicMemory with an embedder ─────────────────────────────────────


def test_frames_and_object_crops_are_embedded_and_recalled_by_image():
    clock = _Clock()
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock)
    assert mem.embedder is not None and mem._index is not None
    mem.add("action", "pick_and_place(object=red cube) -> ok", rgb=_scene(RED),
            data={"verdict": "confirmed"}, crops={"red cube": _crop(RED)})
    clock.t += 5.0
    mem.add("action", "pick_and_place(object=blue cube) -> ok", rgb=_scene(BLUE, x0=100),
            data={"verdict": "refuted"}, crops={"blue cube": _crop(BLUE)})
    clock.t += 5.0

    hits = mem.recall_visual(_noisy(_crop(RED)), k=2, kinds=("object",))
    assert [h["label"] for h in hits] == ["red cube", "blue cube"]
    top = hits[0]
    assert top["kind"] == "object" and top["score"] > 0.9
    assert top["text"].startswith("pick_and_place(object=red cube)")
    assert top["verdict"] == "confirmed" and top["age_s"] == pytest.approx(10.0)

    frames = mem.recall_visual(_scene(BLUE, x0=100), k=1, kinds=("frame",))
    assert len(frames) == 1 and frames[0]["kind"] == "frame"
    assert frames[0]["verdict"] == "refuted" and frames[0]["label"] is None
    assert mem.visual_stats() == {"embedder": "hash-v1", "joint_space": False,
                                  "entries": 4, "embed_errors": 0}


def test_text_queries_need_a_joint_image_text_space():
    mem = EpisodicMemory(embedder=HashEmbedder())
    mem.add("action", "picked", rgb=_scene(RED), crops={"red cube": _crop(RED)})
    with pytest.raises(ValueError, match="joint image-text"):
        mem.recall_visual("the red thing")


def test_a_joint_embedder_recalls_the_thing_that_looked_like_x_above_its_floor():
    clock = _Clock()
    mem = EpisodicMemory(embedder=_JointStub(), clock=clock)
    mem.add("action", "localize_object(label=red cube) -> ok", crops={"red cube": _crop(RED)})
    mem.add("action", "localize_object(label=blue cube) -> ok", crops={"blue cube": _crop(BLUE)})
    hits = mem.recall_visual("something red")
    # the blue crop scores 0 and the floor (the embedder's own) drops it
    assert [(h["kind"], h["label"]) for h in hits] == [("object", "red cube")]
    assert hits[0]["score"] >= _JointStub.text_image_floor
    assert mem.recall_visual("something green") == []
    # an explicit floor overrides the embedder's
    assert len(mem.recall_visual("something red", min_sim=-1.0)) == 2


def test_visual_index_is_pruned_with_the_task_scale_horizon_and_bounded():
    clock = _Clock()
    mem = EpisodicMemory(embedder=HashEmbedder(), clock=clock, frame_horizon_s=60.0,
                         max_visual=5)
    for i in range(3):
        mem.add("action", f"step {i}", rgb=_scene(RED, x0=10 + 10 * i))
        clock.t += 1.0
    assert len(mem._index) == 3
    clock.t += 120.0  # past the task-scale horizon
    mem.add("action", "fresh step", rgb=_scene(GREEN))
    assert len(mem._index) == 1
    assert [h["text"] for h in mem.recall_visual(_scene(RED), k=5)] == ["fresh step"]
    for i in range(10):  # the entry cap holds inside the horizon too
        mem.add("action", f"burst {i}", rgb=_scene(BLUE), crops={"blue cube": _crop(BLUE)})
    assert len(mem._index) == 5 == mem.visual_stats()["entries"]


def test_explicit_embeddings_keep_legacy_recall_and_no_longer_grow_without_bound():
    clock = _Clock()
    mem = EpisodicMemory(horizon_s=15.0, embed_dim=8, clock=clock)
    e1, e2 = np.ones(8), -np.ones(8)
    ev1 = mem.add("observation", "red cube seen", embedding=e1)
    mem.add("observation", "blue ball seen", embedding=e2)
    assert mem.recall_similar(e1, k=1)[0] is ev1
    clock.t += 20.0  # past the 15 s text window: legacy recall forgets, as before
    mem.add("note", "later")  # (any write prunes the rings)
    assert mem.recall_similar(e1, k=1) == []
    for i in range(2000):  # a long session of explicitly embedded events
        clock.t += 1.0
        mem.add("observation", f"event {i}", embedding=np.roll(e1, i % 8))
    assert len(mem._index) <= mem.max_visual


def test_an_embedder_fault_never_breaks_recording():
    class _Broken(HashEmbedder):
        def embed_image(self, bgr):
            raise RuntimeError("device lost")

    mem = EpisodicMemory(embedder=_Broken())
    ev = mem.add("action", "pick -> ok", rgb=_scene(RED), crops={"red cube": _crop(RED)})
    assert ev.thumb_jpeg is not None and mem.memory_frames(1)
    stats = mem.visual_stats()
    assert stats["entries"] == 0 and stats["embed_errors"] == 2


def test_embedder_dimension_must_agree_with_an_explicit_embed_dim():
    with pytest.raises(ValueError, match="embed_dim"):
        EpisodicMemory(embed_dim=7, embedder=HashEmbedder())


def test_without_an_embedder_visual_recall_is_an_explicit_refusal():
    mem = EpisodicMemory()
    mem.add("action", "pick -> ok", rgb=_scene(RED))
    assert mem.embedder is None and mem._index is None
    with pytest.raises(EmbedderUnavailable, match="memory.embedder"):
        mem.recall_visual(_crop(RED))


# ── the runtime records what it looked at (mock stack) ──────────────────


def _wait_for(runtime, label="red cube", timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and runtime.beliefs.find(label) is None:
        time.sleep(0.05)
    assert runtime.beliefs.find(label) is not None, "watcher never saw the mock cube"


def _build(cfg, tmp_path):
    from cascade.apps.demo import build_runtime

    return build_runtime(cfg, tmp_path / "run")


@needs_pin
def test_runtime_with_an_embedder_indexes_frames_and_localized_object_crops(demo_cfg, tmp_path):
    from cascade.apps.demo import shutdown_runtime

    demo_cfg._data["memory"]["embedder"] = {"backend": "hash"}
    runtime, arm = _build(demo_cfg, tmp_path)
    try:
        assert isinstance(runtime.memory.embedder, HashEmbedder)
        _wait_for(runtime)
        res = runtime.execute("localize_object", {"label": "red cube"})
        assert res["ok"], res
        objects = [e for e in runtime.memory._visual if e.kind == "object"]
        assert [e.label for e in objects] == [res["detected_as"]]
        assert runtime._call_crops is None  # the per-call scratchpad was released
        res = runtime.execute("move_home", {})
        assert res["ok"], res
        assert any(e.kind == "frame" for e in runtime.memory._visual)
        # the mock camera re-renders one identical image, so frames tie: any
        # order among them is right, but the move_home frame must be found
        hits = runtime.memory.recall_visual(runtime.last_frame.rgb, k=5, kinds=("frame",))
        assert any(h["text"].startswith("move_home") for h in hits), hits
        # the hash space has no text side shared with images: said explicitly
        out = runtime.execute("recall_memory", {"query": "red cube"})
        assert out["looks_like"] == [] and "joint image-text" in out["looks_like_note"]
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_recall_memory_answers_looks_like_with_a_joint_embedder(demo_cfg, tmp_path):
    from cascade.apps.demo import shutdown_runtime

    runtime, arm = _build(demo_cfg, tmp_path)
    try:
        runtime.memory = EpisodicMemory(embedder=_JointStub())
        _wait_for(runtime)
        assert runtime.execute("localize_object", {"label": "red cube"})["ok"]
        out = runtime.execute("recall_memory", {"query": "red"})
        assert out["ok"] and out["looks_like"], out
        hit = out["looks_like"][0]
        assert hit["kind"] == "object" and hit["score"] >= _JointStub.text_image_floor
        assert "not a current observation" in out["looks_like_note"]
    finally:
        shutdown_runtime(runtime, arm)


def test_a_requested_but_unavailable_backend_fails_the_build_before_any_hardware(
        demo_cfg, tmp_path, monkeypatch):
    import sys

    from cascade.apps import demo

    monkeypatch.setitem(sys.modules, "transformers", None)
    demo_cfg._data["memory"]["embedder"] = {"backend": "siglip"}
    built = []

    def _no_arm(*a, **k):
        built.append(a)
        raise AssertionError("an arm was built before the embedder was checked")

    monkeypatch.setattr(demo, "_build_arm", _no_arm)
    with pytest.raises(EmbedderUnavailable, match="memory-embed"):
        demo.build_runtime(demo_cfg, tmp_path / "run")
    assert built == []


# ── action <-> object consolidation over tier-2 experience ──────────────


def _aom(tmp_path):
    from cascade.memory.consolidation import ActionObjectMemory

    return ActionObjectMemory(tmp_path / "action_objects.json")


def test_outcomes_consolidate_per_action_and_object_across_wordings(tmp_path):
    aom = _aom(tmp_path)
    pick = lambda obj: [("pick_and_place", {"object": obj, "target": "bin"})]  # noqa: E731
    aom.record_plan(pick("red cube"), True, instruction="put the red cube in the bin")
    aom.record_plan(pick("the Red Cube"), True, instruction="Bin the RED cube, please")
    aom.record_plan(pick("red cubes"), False, instruction="tidy the red cubes", failed_at=0)
    aom.record_plan(pick("blue cube"), True, instruction="put the blue cube in the bin")
    # perception calls are not actions on an object
    aom.record_plan([("localize_object", {"label": "red cube"})], True, instruction="where is it")
    rows = {(r["action"], r["object"]): r for r in aom.stats()}
    assert set(rows) == {("pick_and_place", "red cube"), ("pick_and_place", "blue cube")}
    red = rows[("pick_and_place", "red cube")]
    assert (red["wins"], red["losses"]) == (2, 1)
    assert red["aliases"] == ["red cubes"]
    assert red["instructions"] == 3
    # persisted and reloaded identically; a corrupt file is an empty memory
    again = _aom(tmp_path)
    assert again.stats() == aom.stats()
    (tmp_path / "action_objects.json").write_text("{not json")
    assert _aom(tmp_path).stats() == []


def test_a_failed_plan_credits_executed_calls_only(tmp_path):
    aom = _aom(tmp_path)
    plan = [("grasp_object", {"label": "red cube"}),
            ("place_on_object", {"label": "blue bowl"}),
            ("grasp_object", {"label": "green cube"})]
    aom.record_plan(plan, False, instruction="stack them", failed_at=1)
    rows = {(r["action"], r["object"]): (r["wins"], r["losses"]) for r in aom.stats()}
    assert rows == {("grasp_object", "red cube"): (1, 0),
                    ("place_on_object", "blue bowl"): (0, 1)}  # green cube never ran


def test_symbolic_recipe_targets_are_skipped_not_stored_as_objects(tmp_path):
    aom = _aom(tmp_path)
    target = {"$target": {"query": "localize_object", "label": "blue bowl", "offset_m": [0, 0]}}
    aom.record_plan([("grasp_object", {"label": "red cube"}), ("place_at", target)], True,
                    instruction="move the red cube next to the bowl")
    assert [(r["action"], r["object"]) for r in aom.stats()] == [("grasp_object", "red cube")]


def test_digest_names_only_objects_in_the_task_and_says_it_is_advisory(tmp_path):
    aom = _aom(tmp_path)
    aom.record_plan([("pick_and_place", {"object": "red cube"})], True, instruction="a")
    aom.record_plan([("pick_and_place", {"object": "red cube"})], False, instruction="b",
                    failed_at=0)
    aom.record_plan([("push_object", {"label": "blue cube"})], True, instruction="c")
    digest = aom.agent_digest("could you move the red cubes over there")
    assert "advisory" in digest
    assert "pick_and_place x red cube: 1 succeeded, 1 failed" in digest
    assert "blue cube" not in digest
    assert aom.agent_digest("wave at the visitors") == ""


def _fake_runtime(results):
    class _Trace:
        def finish(self, text):
            self.text = text

    class _Memory:
        def add(self, *a, **k):
            pass

    class _Runtime:
        robot_mode = None
        current_tier = None
        last_path = None
        trace = _Trace()
        memory = _Memory()

        def __init__(self):
            self.results = list(results)
            self.calls = []

        def execute(self, name, args):
            self.calls.append(name)
            return self.results.pop(0)

        def unverified_actions(self):
            return []

    return _Runtime()


def test_the_fast_tier_credits_each_executed_plan_once_and_subgoals_never_twice(tmp_path):
    from cascade.agent.llm import MockLLM
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.agent.reflex import ExperienceMemory, FastPlanner

    exp = ExperienceMemory(tmp_path / "experience.json")
    aom = _aom(tmp_path)
    planner = FastPlanner(exp, action_objects=aom)
    assert planner.action_objects is aom
    # (wordings chosen so tier-2 recall cannot match the whole command to one
    # clause's habit: "pick up the red cube and then pick up the blue cube"
    # recalls "pick up the red cube" at cosine 0.901 on main -- see REPORT)
    task = "put the red cube in the bin and then put the blue cube in the bowl"
    plan_calls = [("pick_and_place", {"object": "red cube"}),
                  ("pick_and_place", {"object": "blue cube"})]
    exp.record("put the red cube in the bin", plan_calls[:1], True, 1.0)
    exp.record("put the blue cube in the bowl", plan_calls[1:], True, 1.0)
    runtime = _fake_runtime([{"ok": True}, {"ok": True}])
    orch = AgentOrchestrator(MockLLM([]), runtime, fast_planner=planner, decompose=False,
                             verify_milestones=False)
    report, note = orch._try_fast_path(task, time.monotonic())
    assert report is not None and report.success and note is None
    assert report.path == "curriculum" and runtime.calls == ["pick_and_place"] * 2
    rows = {(r["action"], r["object"]): (r["wins"], r["losses"]) for r in aom.stats()}
    assert rows == {("pick_and_place", "red cube"): (1, 0),
                    ("pick_and_place", "blue cube"): (1, 0)}

    # a failure at the second call credits the first as executed, the second as lost
    runtime.results = [{"ok": True}, {"ok": False, "error": "SkillError: slipped"}]
    report, note = orch._try_fast_path(task, time.monotonic())
    assert report is None and "FAILED" in note
    rows = {(r["action"], r["object"]): (r["wins"], r["losses"]) for r in aom.stats()}
    assert rows == {("pick_and_place", "red cube"): (2, 0),
                    ("pick_and_place", "blue cube"): (1, 1)}


@needs_pin
def test_the_llm_tier_sees_the_consolidated_digest_only_when_enabled(demo_cfg, tmp_path):
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.agent.reflex import ExperienceMemory, FastPlanner
    from cascade.apps.demo import shutdown_runtime

    aom = _aom(tmp_path)
    aom.record_plan([("pick_and_place", {"object": "red cube"})], True, instruction="x")
    runtime, arm = _build(demo_cfg, tmp_path)
    try:
        intros = []
        for planner in (FastPlanner(ExperienceMemory(tmp_path / "e1.json")),
                        FastPlanner(ExperienceMemory(tmp_path / "e2.json"), action_objects=aom)):
            llm = MockLLM([LLMResponse(tool_calls=[ToolCall("task_done", {"success": False,
                                                                          "summary": "only looked"})])])
            AgentOrchestrator(llm, runtime, fast_planner=planner, decompose=False,
                              verify_milestones=False, attach_images=False,
                              plausibility=None).run_task("describe the red cube carefully")
            intros.append(llm.requests[0]["messages"][0]["content"])
        assert "Action-object outcomes" not in intros[0]
        assert "Action-object outcomes" in intros[1]
        assert "pick_and_place x red cube: 1 succeeded, 0 failed" in intros[1]
    finally:
        shutdown_runtime(runtime, arm)


# ── skill-library notes retrieved by embedding ──────────────────────────


def _library(tmp_path, *, promote=True):
    from cascade.skills.library import SkillLibrary

    lib = SkillLibrary(tmp_path / "skills_library")
    tasks = ("pick up the red cube", "put the red cube in the bowl") if promote else ("t1",)
    for i, task in enumerate(tasks):
        lib.add("grasp_object empty_grasp", "grasp_object -> empty_grasp: jaws closed on air",
                "grasp cube empty object red", "lower the grasp by 1 cm",
                source_task=task, source_run=f"run{i}")
        lib.add("place_at release_drift", "place_at -> release_drift: object rolled",
                "bowl place release roll", "release lower", source_task=task + " now",
                source_run=f"run{i}b")
    return lib


def test_embedding_retrieval_finds_inflected_paraphrases_guard_words_miss(tmp_path):
    lib = _library(tmp_path)
    task = "grasping cubes keeps failing"
    assert lib.relevant_entries(task, promoted_only=True) == []
    hits = lib.relevant_entries(task, promoted_only=True, embedder=HashEmbedder())
    assert [e.name for e in hits] == ["grasp-object-empty-grasp"]
    assert hits[0].similarity is not None and hits[0].similarity >= HashEmbedder.text_floor


def test_embedding_retrieval_keeps_the_promotion_gate_and_a_floor(tmp_path):
    candidates = _library(tmp_path / "c", promote=False)
    assert candidates.relevant_entries("grasping cubes", promoted_only=True,
                                       embedder=HashEmbedder()) == []
    lib = _library(tmp_path / "p")
    assert lib.relevant_entries("wave at the audience", promoted_only=True,
                                embedder=HashEmbedder()) == []
    # a guard-word match is never lost by switching the embedder on
    plain = [e.name for e in lib.relevant_entries("release the bowl", promoted_only=True)]
    embedded = [e.name for e in lib.relevant_entries("release the bowl", promoted_only=True,
                                                     embedder=HashEmbedder())]
    assert plain and set(plain) <= set(embedded)


def test_aspire_retrieve_tags_embedding_hits(tmp_path):
    from cascade.agent.aspire import retrieve

    lib = _library(tmp_path)
    text = retrieve(lib, "grasping cubes keeps failing", embedder=HashEmbedder())
    assert "grasp_object empty_grasp" in text
    assert "retrieved by embedding similarity" in text and "hash-v1" in text
    assert retrieve(lib, "grasping cubes keeps failing") == ""


@needs_pin
def test_the_orchestrator_hands_the_memory_embedder_to_library_retrieval(demo_cfg, tmp_path,
                                                                         monkeypatch):
    from cascade.agent import orchestrator as orch_mod
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.apps.demo import shutdown_runtime

    seen = []
    monkeypatch.setattr(orch_mod, "retrieve_skills",
                        lambda lib, task, **kw: seen.append(kw) or "")
    runtime, arm = _build(demo_cfg, tmp_path)
    try:
        for memory in (runtime.memory, EpisodicMemory(embedder=HashEmbedder())):
            runtime.memory = memory
            llm = MockLLM([LLMResponse(tool_calls=[ToolCall("task_done", {"success": False,
                                                                          "summary": "n/a"})])])
            orch_mod.AgentOrchestrator(llm, runtime, skill_library=object(), decompose=False,
                                       verify_milestones=False, attach_images=False,
                                       plausibility=None).run_task("inspect the table carefully")
    finally:
        shutdown_runtime(runtime, arm)
    assert seen[0] == {}  # the default call is exactly the pre-embedder call
    assert isinstance(seen[1]["embedder"], HashEmbedder)
