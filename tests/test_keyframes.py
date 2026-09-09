"""Keyframes are the outcome judge's only evidence (eval/progress_judge.py),
so the runtime must record a BEFORE that exists and an AFTER that was
captured once the motion ENDED.

Measured failure this pins: two chat-driven pick_and_place runs logged
byte-identical before/after JPEGs (md5 8cb509e7...) while physics confirmed
a 19.8 cm move -- `last_frame` was whatever the skill's last observe() saw,
which predates the place motion; and the FIRST skill of every run logged
`keyframe_before: null` because nothing had observed yet.
"""

from __future__ import annotations

import hashlib
import json
import threading

import numpy as np
import pytest

from cascade.apps.demo import build_runtime


class _TickingCamera:
    """Wraps the runtime's camera so every grab paints a different frame:
    a camera that can WITNESS motion. Counts grabs."""

    def __init__(self, inner):
        self._inner = inner
        self.grabs = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_frame(self, *a, **k):
        frame = self._inner.get_frame(*a, **k)
        with self._lock:
            self.grabs += 1
            n = self.grabs
        rgb = np.ascontiguousarray(frame.rgb).copy()
        rgb[:8, :, :] = (n * 37) % 256   # distinct stripe per grab
        return type(frame)(**{**frame.__dict__, "rgb": rgb})


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    runtime.camera = _TickingCamera(runtime.camera)
    yield runtime, arm
    runtime.camera._inner.close()
    arm.disconnect()


def _md5(p) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _rows(runtime):
    return [json.loads(l) for l in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]


def test_first_skill_of_a_run_has_a_before_keyframe(runtime_and_arm):
    runtime, _ = runtime_and_arm
    assert runtime.last_frame is None                     # nothing observed yet
    runtime.execute("move_home", {})
    row = _rows(runtime)[0]
    assert row["keyframe_before"], "first skill logged keyframe_before: null"
    assert (runtime.trace.run_dir / row["keyframe_before"]).exists()


def test_motion_skill_after_keyframe_is_a_fresh_frame(runtime_and_arm):
    runtime, _ = runtime_and_arm
    runtime.observe()
    grabs_before = runtime.camera.grabs
    runtime.execute("move_home", {})
    row = _rows(runtime)[0]
    kb, ka = runtime.trace.run_dir / row["keyframe_before"], runtime.trace.run_dir / row["keyframe_after"]
    assert kb.exists() and ka.exists()
    # the AFTER frame was grabbed AFTER the motion (a new grab happened) ...
    assert runtime.camera.grabs > grabs_before
    # ... and it is not the BEFORE frame re-saved
    assert _md5(kb) != _md5(ka), "before/after keyframes are byte-identical"


def test_non_motion_skill_does_not_grab_an_extra_frame(runtime_and_arm):
    """Only motion skills pay for a fresh AFTER frame; a query skill's AFTER
    is the frame it just observed (no duplicate grab per tool call)."""
    runtime, _ = runtime_and_arm
    runtime.execute("get_observation", {})
    grabs = runtime.camera.grabs
    runtime.execute("robot_status", {}) if hasattr(runtime, "skill_robot_status") else runtime.execute("describe_scene", {})
    # describe_scene / robot_status observe at most once themselves; the
    # execute() wrapper must not add a second grab for a non-motion skill
    assert runtime.camera.grabs - grabs <= 1


def test_trace_rows_carry_the_dispatch_tier(runtime_and_arm):
    """per_tier() in the judge keys on it; 'unknown' for every row was the
    measured state before the dispatcher set runtime.current_tier."""
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.agent.orchestrator import AgentOrchestrator

    runtime, _ = runtime_and_arm
    llm = MockLLM([LLMResponse(text="", tool_calls=[ToolCall("move_home", {})]),
                   LLMResponse(text="", tool_calls=[ToolCall("task_done", {"success": True, "summary": "ok"})])])
    agent = AgentOrchestrator(llm, runtime, advisor=None, decompose=False, max_steps=4)
    agent.run_task("go home please, via the llm tier")   # no reflex plan for this phrasing
    tiers = {r["skill"]: r.get("tier") for r in _rows(runtime)}
    assert tiers["move_home"] in ("llm", "reflex", "experience"), tiers
    assert runtime.current_tier is None                    # always reset after the call
