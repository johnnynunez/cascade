"""Vesta memory harness (arXiv:2606.20905 §2.4) -- the planner sees its own
history as captioned frames, not text alone.

Three layers, each pinned:

1. `EpisodicMemory.memory_frames()` -- the sampler: first frame always kept,
   the rest uniform, newest last; frames live past the 15 s text horizon;
   reset per episode.
2. `SkillRuntime.execute()` -- what gets recorded: the initial state before
   the first motion, then the AFTER frame of every motion skill with its
   independent verdict; observation-only skills record no frame.
3. The two planner tiers -- the orchestrator injects the harness on every
   LLM turn (images sent once per request, never accumulated); the MCP
   `task_memory` tool serves the same frames to a chat host as image
   content items.

The recording tests run the REAL wiring (mock camera/detector/arm, scripted
LLM) -- a frame is only recorded if the runtime actually passes one.
"""
from __future__ import annotations

import base64
import hashlib
import json

import numpy as np
import pytest
from conftest import needs_pin

from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.memory.episodic import EpisodicMemory


def _img(v: int, shape=(48, 64, 3)) -> np.ndarray:
    a = np.zeros(shape, np.uint8)
    a[:] = v
    return a


def _mem(**kw) -> tuple[EpisodicMemory, list[float]]:
    t = [0.0]
    return EpisodicMemory(horizon_s=15.0, clock=lambda: t[0], **kw), t


# ── 1. sampler ─────────────────────────────────────────────────────────────


def test_first_frame_always_kept_rest_uniform_newest_last():
    """Vesta: 'The first frame is always retained to preserve the initial
    state'; the remaining K-1 slots are spread over the rest and the newest
    frame closes the list (it is what the current view is compared to)."""
    m, t = _mem()
    m.add("observation", "initial", rgb=_img(0))
    for i in range(1, 10):
        t[0] += 1.0
        m.add("action", f"step{i}", rgb=_img(i * 20))
    steps = [f["step"] for f in m.memory_frames(4)]
    assert steps[0] == 1, "initial state dropped"
    assert steps[-1] == 10, "newest frame not last"
    assert steps == sorted(steps) and len(steps) == 4
    assert [f["step"] for f in m.memory_frames(2)] == [1, 10]
    assert [f["step"] for f in m.memory_frames(1)] == [10]
    assert len(m.memory_frames(50)) == 10  # never pads


def test_frames_outlive_the_text_horizon_and_reset_per_episode():
    """A 15 s text window would forget the initial state before one pick
    finished (~20 s). Frames have their own task-scale horizon; a new
    episode clears them and leaves the text ring alone."""
    m, t = _mem(frame_horizon_s=600.0)
    m.add("observation", "initial", rgb=_img(0))
    t[0] += 120.0
    m.add("action", "later", rgb=_img(9))
    assert "initial" not in m.digest(), "text ring should have pruned the 120 s old line"
    assert [f["text"] for f in m.memory_frames(4)] == ["initial", "later"]
    m.reset_frames()
    assert m.memory_frames(4) == []
    assert "later" in m.digest(), "episode reset must not touch the text ring"


def test_frame_horizon_prunes_and_caption_carries_verdict():
    m, t = _mem(frame_horizon_s=30.0)
    m.add("action", "old", rgb=_img(1))
    t[0] += 31.0
    m.add("action", "pick_and_place(object=red) -> ok", data={"verdict": "refuted"}, rgb=_img(2))
    fr = m.memory_frames(4)
    assert [f["text"] for f in fr] == ["pick_and_place(object=red) -> ok"]
    cap = EpisodicMemory.frame_caption(fr[0])
    assert "[REFUTED]" in cap and "memory frame" in cap and fr[0]["age_s"] == 0.0


def test_pre_encoded_thumbnail_is_stored_verbatim():
    m, _ = _mem()
    jpeg = b"\xff\xd8fake\xff\xd9"
    m.add("action", "x", thumb_jpeg=jpeg)
    assert m.memory_frames(1)[0]["jpeg"] == jpeg
    assert m.last_frame_jpeg() == jpeg


# ── 2. what the runtime records ─────────────────────────────────────────────


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    yield runtime, arm
    runtime.camera.close()
    arm.disconnect()


def _script(*calls: ToolCall) -> list[LLMResponse]:
    return [LLMResponse(text="", tool_calls=[c]) for c in calls]


@needs_pin
def test_runtime_records_one_frame_per_motion_with_verdict(runtime_and_arm):
    """`get_observation` already stores its frame (Vesta's o_i: what the
    agent SAW); `list_objects` reads beliefs and stores none. Every motion
    then appends its AFTER frame carrying the independent verdict, and the
    'initial state' pin is NOT added when an observation frame already
    anchors the episode."""
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    runtime.execute("get_observation", {})
    runtime.execute("list_objects", {})
    fr = runtime.memory.memory_frames(8)
    assert len(fr) == 1 and fr[0]["kind"] == "observation", fr

    r = runtime.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert r["ok"], r
    fr = runtime.memory.memory_frames(8)
    assert len(fr) == 2 and fr[1]["text"].startswith("grasp_object("), fr
    assert fr[1]["verdict"] == (r.get("postcondition") or {}).get("status", ""), fr[1]
    assert not any(f["text"].startswith("initial state") for f in fr)

    r2 = runtime.execute("place_at", {"x": 0.20, "y": -0.15})
    assert r2["ok"], r2
    fr = runtime.memory.memory_frames(8)
    assert len(fr) == 3 and fr[2]["text"].startswith("place_at("), fr


@needs_pin
def test_first_motion_without_prior_observation_pins_the_initial_state(runtime_and_arm):
    """A chat host may call pick_and_place as the very first tool. The
    harness must still open with the scene BEFORE anything moved, or frame
    1 would be the aftermath of step one."""
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    assert runtime.memory.memory_frames(8) == []
    r = runtime.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert r["ok"], r
    fr = runtime.memory.memory_frames(8)
    assert [f["text"][:13] for f in fr] == ["initial state", "grasp_object("], fr
    # only once per episode
    runtime.execute("place_at", {"x": 0.20, "y": -0.15})
    assert sum(f["text"].startswith("initial state") for f in runtime.memory.memory_frames(8)) == 1


@needs_pin
def test_orchestrator_injects_captioned_frames_once_per_request(runtime_and_arm):
    """Every planner turn carries the harness as ONE trailing user message:
    K past frames + the current view, captioned; images never accumulate in
    the conversation (the pruned history has none left)."""
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    llm = MockLLM(
        _script(
            ToolCall("get_observation", {}),
            ToolCall("grasp_object", {"label": "red cube", "material": "rigid"}),
            ToolCall("place_at", {"x": 0.20, "y": -0.15}),
            ToolCall("task_done", {"success": True, "summary": "moved"}),
        )
    )
    agent = AgentOrchestrator(llm, runtime, advisor=None, decompose=False, max_steps=10,
                              memory_frames_k=4)
    report = agent.run_task("move the red cube to the front-left")
    assert report.success
    # request 4 (task_done turn) follows one observation and two motions:
    # saw + grasp + place + current view
    req = llm.requests[3]["messages"]
    harness = [m for m in req if m.get("images")]
    assert len(harness) == 1, "images must enter the request in exactly one message"
    h = harness[0]
    assert h["role"] == "user"
    assert len(h["images"]) == 4, [ln for ln in h["content"].splitlines()]
    text = h["content"]
    assert "memory frame 1" in text and "saw [" in text
    assert "grasp_object(" in text and "place_at(" in text
    assert text.rstrip().endswith("current view (now)") or "current view (now)" in text
    assert "Observation" in text and "Progress" in text and "Reasoning" in text
    # before anything was observed there is no frame at all -- no image
    # message is fabricated
    assert not [m for m in llm.requests[0]["messages"] if m.get("images")]
    # after get_observation: exactly the frame just seen (memory frame 1 ==
    # current view content-wise, still two slots: history and now)
    req2 = [m for m in llm.requests[1]["messages"] if m.get("images")]
    assert len(req2) == 1 and len(req2[0]["images"]) == 2
    # frames are per-episode: a new task starts clean (only the current view,
    # which the runtime still holds from the previous task)
    llm2 = MockLLM(_script(ToolCall("task_done", {"success": True, "summary": "nothing"})))
    agent2 = AgentOrchestrator(llm2, runtime, advisor=None, decompose=False, max_steps=3,
                               memory_frames_k=4)
    agent2.run_task("do nothing")
    imgs = [m for m in llm2.requests[0]["messages"] if m.get("images")]
    assert len(imgs) == 1 and len(imgs[0]["images"]) == 1, "previous episode's frames leaked"


@needs_pin
def test_k_zero_is_text_only_history(runtime_and_arm):
    """memory_frames_k=0 reproduces the pre-harness behaviour exactly: only
    the newest image in context, no harness message."""
    runtime, arm = runtime_and_arm
    llm = MockLLM(
        _script(
            ToolCall("get_observation", {}),
            ToolCall("task_done", {"success": True, "summary": "looked"}),
        )
    )
    agent = AgentOrchestrator(llm, runtime, advisor=None, decompose=False, max_steps=5,
                              memory_frames_k=0)
    agent.run_task("look around")
    for req in llm.requests:
        for m in req["messages"]:
            assert "memory frame" not in str(m.get("content", ""))


def test_system_prompt_asks_for_four_phases():
    from cascade.agent.prompts import SYSTEM_PROMPT

    for phase in ("Observation", "Progress", "Reasoning", "Action"):
        assert phase in SYSTEM_PROMPT
    assert "REFUTED" in SYSTEM_PROMPT


# ── 3. the chat-host tool ────────────────────────────────────────────────────


@needs_pin
def test_task_memory_tool_serves_frames_as_image_content(runtime_and_arm):
    from cascade.apps.mcp_server import McpSkillServer, _EXTRA_TOOLS

    assert any(t["name"] == "task_memory" for t in _EXTRA_TOOLS)
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    server = McpSkillServer()
    server._runtime = runtime  # bypass the lazy build: the real runtime is already up

    out = server._task_memory(runtime, {"new_task": True})
    kinds = [c["type"] for c in out["content"]]
    assert kinds.count("image") == 1, "before any action: only the current view"
    summary = json.loads(out["content"][-1]["text"])
    assert summary["frames"] == 0 and summary["ok"]

    r = runtime.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert r["ok"], r
    out = server._task_memory(runtime, {})
    content = out["content"]
    images = [c for c in content if c["type"] == "image"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert len(images) == 3, texts  # initial + grasp + current
    assert texts[0].startswith("memory frame 1") and "initial state" in texts[0]
    assert "grasp_object(" in texts[1]
    assert texts[2] == "current view (now)"
    for im in images:
        raw = base64.b64decode(im["data"])
        assert raw[:2] == b"\xff\xd8", "not a JPEG"
    summary = json.loads(texts[-1])
    assert summary["frames"] == 2
    assert summary["steps_recorded"][1]["verdict"] in ("confirmed", "refuted", "unverified", "none")
    # initial state and current view differ once something moved (mock camera
    # paints the cube where the belief says it is)
    assert hashlib.md5(base64.b64decode(images[0]["data"])).hexdigest() != \
        hashlib.md5(base64.b64decode(images[-1]["data"])).hexdigest()

    # new_task forgets the history
    out = server._task_memory(runtime, {"new_task": True, "k": 2})
    assert [c["type"] for c in out["content"]].count("image") == 1
