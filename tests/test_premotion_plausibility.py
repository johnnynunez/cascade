"""ROADMAP follow-up #6: pre-motion plausibility check (Human-CLAW), advisory only.

Before the tier-3 planner's motion call is dispatched, the configured VLM is
asked a skill-specific question ("is this call, with these args, plausible
given the current view, beliefs, reachability and held state?"). The answer
is attached to the result and the trace row as ``plausibility`` and, when
implausible, surfaced to the planner as a caution on its next turn.

What these tests pin, in order of importance:

* it is NEVER a veto -- an implausible verdict still executes the call,
  unchanged, through the normal ``SkillRuntime.execute()`` path (the safety
  harness remains the sole authority that refuses motion);
* it is rate-limited per task with the milestone tracker's own budget
  mechanism, so it cannot blow the booth clock;
* a verifier fault, a missing frame, a missing vision model or an exhausted
  budget all become ``skipped`` with a reason -- never an exception, never a
  stall;
* ``plausibility=None`` (``agent.premotion_check: false``) reproduces the
  pre-change orchestrator path exactly: same runtime interactions, same
  messages, no extra keys.
"""

from __future__ import annotations

import json

import pytest

from conftest import needs_pin

from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.milestones import (
    IMPLAUSIBLE,
    PLAUSIBLE,
    SKIPPED,
    UNSURE,
    MilestoneTracker,
    PlausibilityChecker,
    VisualBudget,
    make_plausibility_verifier,
)
from cascade.agent.orchestrator import AgentOrchestrator


def _script(*calls: ToolCall) -> list[LLMResponse]:
    return [LLMResponse(text="", tool_calls=[c]) for c in calls]


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


def _checker(runtime, critic_llm, **kw) -> PlausibilityChecker:
    return PlausibilityChecker(
        make_plausibility_verifier(critic_llm),
        beliefs=runtime.beliefs,
        held_getter=lambda: runtime.held_object,
        **kw,
    )


def _trace_rows(runtime) -> list[dict]:
    path = runtime.trace.run_dir / "trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# ── the headline: implausible is a caution, never a refusal ─────────────────


@needs_pin
def test_implausible_verdict_is_a_caution_and_the_motion_still_executes(runtime_and_arm):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5  # an object is in the jaws: the grasp succeeds

    planner = MockLLM(
        _script(
            ToolCall("get_observation", {}),
            ToolCall("grasp_object", {"label": "red cube"}),
            ToolCall("task_done", {"success": True, "summary": "picked"}),
        )
    )
    critic_llm = MockLLM(
        [LLMResponse(text="NO. The gripper already holds something and the cube is out of reach.")]
    )
    agent = AgentOrchestrator(
        planner, runtime, advisor=None, decompose=False, max_steps=10,
        plausibility=_checker(runtime, critic_llm, max_checks=3),
    )
    commands_before = len(arm.commands)
    report = agent.run_task("pick up the red cube")

    grasp = next(e for e in report.tool_log if e["tool"] == "grasp_object")
    # The motion ran, unchanged, through the normal path: the arm was driven
    # and the skill reports the grasp -- the critic refused nothing.
    assert grasp["result"]["ok"], grasp["result"]
    assert grasp["result"]["held"] == "red cube"
    assert grasp["args"] == {"label": "red cube"}
    assert len(arm.commands) > commands_before
    assert grasp["result"]["plausibility"] == {
        "verdict": IMPLAUSIBLE,
        "reasons": ["NO. The gripper already holds something and the cube is out of reach."],
        "source": "vlm:MockLLM",
    }
    # Exactly one critic turn, carrying the current frame, the proposed call
    # and the belief digest (the critic judges the call, not a blank page).
    assert len(critic_llm.requests) == 1
    msg = critic_llm.requests[0]["messages"][0]
    assert msg.get("images"), "the critic must see the current frame"
    assert "grasp_object" in msg["content"] and "red cube" in msg["content"]
    assert "gripper" in msg["content"].lower()  # held-state question for a grasp
    # The planner sees the caution on its NEXT turn, worded as advice.
    nxt = planner.requests[2]["messages"]
    cautions = [m for m in nxt if m["role"] == "user" and "Caution" in str(m.get("content"))]
    assert len(cautions) == 1, [m.get("content") for m in nxt]
    text = cautions[0]["content"]
    assert "advisory" in text and "executed" in text and "grasp_object" in text
    assert "refus" not in text.lower(), "the caution must not read as a refusal"
    # ... and the tool result it reads back carries the verdict too.
    tool_msgs = [m for m in nxt if m["role"] == "tool" and m["name"] == "grasp_object"]
    assert json.loads(tool_msgs[0]["content"])["plausibility"]["verdict"] == IMPLAUSIBLE
    # The trace row carries the verdict next to the untouched args.
    row = next(r for r in _trace_rows(runtime) if r["skill"] == "grasp_object")
    assert row["result"]["plausibility"]["verdict"] == IMPLAUSIBLE
    assert row["args"] == {"label": "red cube"}
    # Non-motion calls are never interrogated and carry no key.
    obs = next(e for e in report.tool_log if e["tool"] == "get_observation")
    assert "plausibility" not in obs["result"]
    assert "plausibility" in runtime.memory.digest()


@needs_pin
def test_plausible_verdict_is_attached_without_a_caution(runtime_and_arm):
    runtime, arm = runtime_and_arm
    planner = MockLLM(
        _script(
            ToolCall("move_home", {}),
            ToolCall("task_done", {"success": True, "summary": "home"}),
        )
    )
    critic_llm = MockLLM([LLMResponse(text="YES, the path home is clear.")])
    agent = AgentOrchestrator(
        planner, runtime, advisor=None, decompose=False, max_steps=5,
        plausibility=_checker(runtime, critic_llm, max_checks=3),
    )
    report = agent.run_task("go home please")
    home = next(e for e in report.tool_log if e["tool"] == "move_home")
    assert home["result"]["ok"]
    assert home["result"]["plausibility"]["verdict"] == PLAUSIBLE
    assert home["result"]["plausibility"]["source"] == "vlm:MockLLM"
    # A first motion with no prior observation still gets a frame to judge.
    assert critic_llm.requests[0]["messages"][0].get("images")
    assert not [
        m for m in planner.requests[1]["messages"]
        if m["role"] == "user" and "Caution" in str(m.get("content"))
    ]


# ── rate limit: the milestone budget, per task ──────────────────────────────


@needs_pin
def test_rate_limiter_bounds_critic_calls_per_task_and_resets_per_task(runtime_and_arm):
    runtime, arm = runtime_and_arm
    critic_llm = MockLLM([LLMResponse(text="NO. Not now.")] * 10)
    checker = _checker(runtime, critic_llm, max_checks=2)

    def run(task: str):
        planner = MockLLM(
            _script(
                *([ToolCall("move_home", {})] * 4),
                ToolCall("task_done", {"success": True, "summary": "x"}),
            )
        )
        agent = AgentOrchestrator(
            planner, runtime, advisor=None, decompose=False, max_steps=10, plausibility=checker
        )
        return agent.run_task(task)

    report = run("go home four times")
    motions = [e["result"] for e in report.tool_log if e["tool"] == "move_home"]
    assert [m["plausibility"]["verdict"] for m in motions] == [
        IMPLAUSIBLE, IMPLAUSIBLE, SKIPPED, SKIPPED,
    ]
    assert len(critic_llm.requests) == 2, "the budget bounds VLM turns per task"
    assert all(m["ok"] for m in motions), "skipped or not, every motion ran"
    skipped = motions[2]["plausibility"]
    assert skipped["source"] == "none" and "budget" in skipped["reasons"][0]
    assert "2" in skipped["reasons"][0]

    run("again")
    assert len(critic_llm.requests) == 4, "the budget resets with the task"


def test_plausibility_budget_is_the_milestone_tracker_rate_limiter():
    """One mechanism, optionally one pool: a shared ``VisualBudget`` makes the
    pre-motion critic and the milestone visual tier draw the same VLM turns."""
    budget = VisualBudget(2)
    tracker_calls, critic_calls = [], []

    def milestone_verifier(text, jpeg):
        tracker_calls.append(text)
        return None, "cannot tell"

    def critic_verifier(text, jpeg):
        critic_calls.append(text)
        return False, "NO"

    tracker = MilestoneTracker(beliefs=None, vlm_verify=milestone_verifier, budget=budget)
    checker = PlausibilityChecker(critic_verifier, budget=budget)
    tracker.reset(["something unverifiable happened"])

    assert checker.check("move_home", {}, b"jpeg")["verdict"] == IMPLAUSIBLE
    tracker.update(b"jpeg")                       # takes the second and last turn
    assert len(tracker_calls) == 1 and len(critic_calls) == 1
    assert checker.check("move_home", {}, b"jpeg")["verdict"] == SKIPPED
    tracker.update(b"jpeg")
    assert len(tracker_calls) == 1, "the tracker abstains once the shared pool is gone"
    assert tracker.max_visual_checks == 2          # the tracker's old knob still reads

    budget.reset()
    assert checker.check("move_home", {}, b"jpeg")["verdict"] == IMPLAUSIBLE


# ── never blocks: faults and missing inputs become `skipped` ────────────────


@needs_pin
def test_verifier_exception_becomes_skipped_and_never_blocks(runtime_and_arm):
    runtime, arm = runtime_and_arm

    def exploding_verifier(text, jpeg):
        raise RuntimeError("VLM endpoint down")

    class ExplodingLLM(MockLLM):
        def chat(self, system, messages, tools=None, max_tokens=1024):
            super().chat(system, messages, tools, max_tokens)
            raise TimeoutError("read timed out")

    cases = [
        (PlausibilityChecker(exploding_verifier, beliefs=runtime.beliefs), "RuntimeError"),
        (_checker(runtime, ExplodingLLM([]), max_checks=3), "TimeoutError"),
    ]
    for checker, expected in cases:
        planner = MockLLM(
            _script(
                ToolCall("move_home", {}),
                ToolCall("task_done", {"success": True, "summary": "x"}),
            )
        )
        agent = AgentOrchestrator(
            planner, runtime, advisor=None, decompose=False, max_steps=5, plausibility=checker
        )
        report = agent.run_task("go home")
        home = next(e for e in report.tool_log if e["tool"] == "move_home")
        assert home["result"]["ok"], home["result"]
        adv = home["result"]["plausibility"]
        assert adv["verdict"] == SKIPPED and adv["source"] == "none"
        assert expected in adv["reasons"][0], adv
        assert not [
            m for m in planner.requests[1]["messages"]
            if m["role"] == "user" and "Caution" in str(m.get("content"))
        ]
    assert runtime.current_tier is None


def test_skipped_when_no_vision_model_frame_or_budget():
    no_model = PlausibilityChecker(None)
    adv = no_model.check("grasp_object", {"label": "cube"}, b"jpeg")
    assert adv == {
        "verdict": SKIPPED,
        "reasons": ["no vision-capable model configured"],
        "source": "none",
    }
    mock_brain = PlausibilityChecker(None, skip_reason="mock brain is not a vision model")
    assert mock_brain.check("move_home", {}, b"jpeg")["reasons"] == [
        "mock brain is not a vision model"
    ]

    calls = []

    def verifier(text, jpeg):
        calls.append(text)
        return True, "YES"

    checker = PlausibilityChecker(verifier, max_checks=1)
    assert checker.check("move_home", {}, None)["verdict"] == SKIPPED
    assert "frame" in checker.check("move_home", {}, b"")["reasons"][0]
    assert calls == [], "no frame, no VLM turn, no budget spent"
    assert checker.check("move_home", {}, b"jpeg")["verdict"] == PLAUSIBLE
    assert checker.check("move_home", {}, b"jpeg")["verdict"] == SKIPPED
    assert len(calls) == 1


def test_verifier_parses_yes_no_unsure_and_names_its_source():
    answers = {
        "YES -- the cube is free.": PLAUSIBLE,
        "no. the gripper is full": IMPLAUSIBLE,
        "Unsure, the object is occluded.": UNSURE,
        "": UNSURE,
        "Maybe?": UNSURE,
    }
    for text, verdict in answers.items():
        llm = MockLLM([LLMResponse(text=text)])
        checker = PlausibilityChecker(make_plausibility_verifier(llm))
        adv = checker.check("grasp_object", {"label": "cube"}, b"jpeg")
        assert adv["verdict"] == verdict, (text, adv)
        assert adv["source"] == "vlm:MockLLM"
        assert adv["reasons"] == [text[:200] or "no answer"]
        req = llm.requests[0]
        assert req["messages"][0]["images"] == [b"jpeg"]
        assert "grasp_object" in req["messages"][0]["content"]
        assert "critic" in req["system"].lower()

    class Named(MockLLM):
        model = "Qwen/Qwen3.8-27B"

    checker = PlausibilityChecker(make_plausibility_verifier(Named([LLMResponse(text="YES")])))
    assert checker.check("move_home", {}, b"jpeg")["source"] == "vlm:Qwen/Qwen3.8-27B"


# ── the question is about THIS call ─────────────────────────────────────────


def test_questions_are_skill_specific_and_carry_the_world_model():
    class _Belief:
        def __init__(self, label, pos, state, color=None):
            self.label, self.position, self.color = label, pos, color
            self._state = state
            self.last_seen_t, self.conf = 0.0, 0.9

        def state(self, now=None):
            return self._state

    class _Beliefs:
        def summary(self, now=None):
            return [
                {"label": "red cube", "color": "red", "position": [0.29, 0.10, 0.02],
                 "state": "visible", "age_s": 0.2, "conf": 0.9},
                {"label": "blue cube", "color": "blue", "position": [0.20, -0.20, 0.02],
                 "state": "remembered", "age_s": 12.5, "conf": 0.7},
            ]

    checker = PlausibilityChecker(
        None, beliefs=_Beliefs(), held_getter=lambda: "blue cube",
        workspace={"min": [0.10, -0.30, -0.01], "max": [0.50, 0.30, 0.55]},
    )
    q_grasp = checker.question("grasp_object", {"label": "red cube"})
    q_place = checker.question("place_at", {"x": 0.2, "y": -0.15})
    q_push = checker.question("push_object", {"label": "red cube", "direction": "left"})
    assert "red cube" in q_grasp and "empty" in q_grasp.lower()
    assert "0.2" in q_place and "holding" in q_place.lower()
    assert "left" in q_push
    assert q_grasp != q_place != q_push
    # a motion skill without a bespoke question still gets a real one
    assert "plausible" in checker.question("wave", {"cycles": 2}).lower()
    # missing args degrade the wording, never the check
    assert "?" in checker.question("grasp_object", {})

    digest = checker.belief_digest()
    assert "red cube" in digest and "visible" in digest
    assert "blue cube" in digest and "remembered" in digest
    assert "holding: blue cube" in digest
    assert "0.10" in digest and "0.50" in digest  # reachability bounds
    assert "nothing" in PlausibilityChecker(None, beliefs=_Beliefs()).belief_digest()

    prompt = checker.prompt("grasp_object", {"label": "red cube"})
    assert "grasp_object(label=red cube)" in prompt or "grasp_object(label='red cube')" in prompt
    assert q_grasp in prompt and digest in prompt
    assert "YES" in prompt and "NO" in prompt and "UNSURE" in prompt


# ── the opt-out is today's path, byte for byte ──────────────────────────────


class _StubMemory:
    def __init__(self):
        self.added = []

    def reset_frames(self):
        pass

    def digest(self, max_lines=None):
        return "(nothing yet)"

    def memory_frames(self, k):
        return []

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))


class _StubTrace:
    def __init__(self):
        self.summaries = []

    def finish(self, summary):
        self.summaries.append(summary)


class _StubRuntime:
    """Deterministic stand-in for ``SkillRuntime`` that records every
    attribute the orchestrator writes and every call it dispatches."""

    def __init__(self):
        object.__setattr__(self, "sets", [])
        self.memory = _StubMemory()
        self.trace = _StubTrace()
        self.calls = []
        self.last_frame = object()
        self.current_tier = None
        self.held_object = None
        self.sets.clear()

    def __setattr__(self, key, value):
        self.sets.append((key, value))
        object.__setattr__(self, key, value)

    def execute(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True, "echo": name}

    def frame_jpeg(self):
        return b"jpeg"


_SCRIPT = (
    ToolCall("move_home", {}, id="c1"),
    ToolCall("place_at", {"x": 0.2, "y": -0.1}, id="c2"),
    ToolCall("task_done", {"success": True, "summary": "done"}, id="c3"),
)


def _run_stub(plausibility):
    rt = _StubRuntime()
    llm = MockLLM(_script(*_SCRIPT))
    agent = AgentOrchestrator(
        llm, rt, advisor=None, decompose=False, attach_images=False,
        verify_milestones=False, max_steps=5, plausibility=plausibility,
    )
    return rt, llm, agent.run_task("stub task")


def test_opt_out_is_byte_identical_to_the_pre_change_path():
    """``plausibility=None`` must leave no trace: the golden below is the
    pre-change orchestrator's exact behaviour on this stub (same runtime
    writes, same dispatches, same planner messages, no extra keys)."""
    rt, llm, report = _run_stub(None)

    assert rt.calls == [(name, args) for name, args in ((c.name, c.arguments) for c in _SCRIPT)]
    assert rt.sets == [("current_tier", "llm"), ("current_tier", None)] * 3 + [("last_path", "llm")]
    assert rt.memory.added == []
    assert report.success is True and report.steps == 3 and report.path == "llm"
    assert report.tool_log == [
        {"step": 1, "tier": "llm", "tool": "move_home", "args": {},
         "result": {"ok": True, "echo": "move_home"}},
        {"step": 2, "tier": "llm", "tool": "place_at", "args": {"x": 0.2, "y": -0.1},
         "result": {"ok": True, "echo": "place_at"}},
        {"step": 3, "tier": "llm", "tool": "task_done", "args": {"success": True, "summary": "done"},
         "result": {"ok": True, "task_complete": True}},
    ]
    # Planner-visible conversation on the last turn: intro, then exactly
    # (assistant, tool) per executed call -- no cautions, no annotations.
    last = llm.requests[2]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "tool", "assistant", "tool"]
    assert last[0]["content"].startswith("Task: stub task\n")
    assert last[2] == {"role": "tool", "tool_call_id": "c1", "name": "move_home",
                       "content": json.dumps({"ok": True, "echo": "move_home"})}
    assert last[4] == {"role": "tool", "tool_call_id": "c2", "name": "place_at",
                       "content": json.dumps({"ok": True, "echo": "place_at"})}
    assert not hasattr(rt, "pending_plausibility")


def test_critic_on_the_stub_adds_only_the_advisory_and_still_dispatches_unchanged():
    """Same stub, critic on: the ONLY differences from the golden are the
    hand-off attribute, the ``plausibility`` key and one caution message."""
    critic_llm = MockLLM([LLMResponse(text="NO. Nothing is held.")] * 5)
    checker = PlausibilityChecker(make_plausibility_verifier(critic_llm), max_checks=1)
    rt, llm, report = _run_stub(checker)

    # Dispatch is identical: same calls, same args, same order, nothing skipped.
    assert rt.calls == [(name, args) for name, args in ((c.name, c.arguments) for c in _SCRIPT)]
    writes = [k for k, _ in rt.sets]
    assert [k for k in writes if k != "pending_plausibility"] == (
        ["current_tier", "current_tier"] * 3 + ["last_path"]
    )
    # The hand-off is set before and cleared after each interrogated motion.
    assert [v for k, v in rt.sets if k == "pending_plausibility" and v is None] == [None, None]
    assert not hasattr(rt, "pending_plausibility") or rt.pending_plausibility is None
    results = {e["tool"]: e["result"] for e in report.tool_log}
    assert results["move_home"]["plausibility"]["verdict"] == IMPLAUSIBLE
    assert results["place_at"]["plausibility"]["verdict"] == SKIPPED   # budget of 1
    assert "plausibility" not in results["task_done"]
    assert len(critic_llm.requests) == 1
    last = llm.requests[2]["messages"]
    roles = [m["role"] for m in last]
    assert roles == ["user", "assistant", "tool", "user", "assistant", "tool"], roles
    assert "Caution" in last[3]["content"] and "move_home" in last[3]["content"]
    assert rt.memory.added and "implausible" in str(rt.memory.added[0])


# ── composition root: config -> checker ─────────────────────────────────────


def test_bare_runtime_without_the_hand_off_attribute_still_dispatches(tmp_path):
    """Unit fixtures build ``SkillRuntime`` via ``__new__`` (no ``__init__``):
    the trace hand-off must be optional there -- absent it adds nothing, set
    it rides on the result and the trace row and is consumed by the call."""
    import threading
    from types import SimpleNamespace
    from typing import Any

    from cascade.agent.task_effects import TaskEffects
    from cascade.agent.trace import TraceLogger
    from cascade.skills.runtime import SkillRuntime

    rt: Any = SkillRuntime.__new__(SkillRuntime)
    rt._task_effects = TaskEffects()
    rt._arm = object()
    rt._arm_override = threading.local()
    rt.arm_rig = None
    rt.held_object = None
    rt.last_frame = rt.watcher = rt._motion_t0 = rt.effects = None
    rt.current_tier = "test"
    rt.memory = SimpleNamespace(memory_frames=lambda _: [], add=lambda *a, **kw: None)
    rt.envelope = SimpleNamespace(record=lambda *a, **kw: None)
    rt.observe = lambda: None
    rt._show_status = lambda _: None
    rt.trace = TraceLogger(tmp_path / "bare")
    rt.skill_noop = lambda: {"ok": True}

    assert not hasattr(rt, "pending_plausibility")
    assert rt.execute("noop", {}) == {"ok": True}
    rt.pending_plausibility = {"verdict": UNSURE, "reasons": ["x"], "source": "vlm:t"}
    out = rt.execute("noop", {})
    assert out["plausibility"]["verdict"] == UNSURE
    assert rt.pending_plausibility is None, "consumed by the call it was meant for"
    assert "plausibility" not in rt.execute("noop", {})
    rows = _trace_rows(rt)
    assert "plausibility" not in rows[0]["result"]
    assert rows[1]["result"]["plausibility"]["verdict"] == UNSURE
    assert "plausibility" not in rows[2]["result"]


def test_demo_config_default_and_opt_out(demo_cfg, monkeypatch):
    from cascade.apps.demo import _premotion_critic

    monkeypatch.delenv("CASCADE_PREMOTION_CHECK", raising=False)
    assert demo_cfg.agent.premotion_check is True
    assert demo_cfg.agent.premotion_max_checks == 3

    class _RT:
        beliefs = None
        held_object = None

    vision = MockLLM([])
    text_only = MockLLM([])
    text_only.supports_vision = False

    critic = _premotion_critic(demo_cfg, vision, _RT(), is_mock=False)
    assert isinstance(critic, PlausibilityChecker)
    assert critic.budget.max_checks == 3
    assert critic.check("move_home", {}, b"jpeg")["source"] == "vlm:MockLLM"

    # the mock brain is a script, not a judge: construct, but record skipped
    script = MockLLM([LLMResponse(text="", tool_calls=[ToolCall("task_done", {})])])
    mock = _premotion_critic(demo_cfg, script, _RT(), is_mock=True)
    adv = mock.check("move_home", {}, b"jpeg")
    assert adv["verdict"] == SKIPPED and "mock" in adv["reasons"][0]
    assert script.requests == [], "asking the mock brain would consume its script"

    textual = _premotion_critic(demo_cfg, text_only, _RT(), is_mock=False)
    assert textual.check("move_home", {}, b"jpeg")["reasons"] == [
        "no vision-capable model configured"
    ]

    demo_cfg._data["agent"] = {"premotion_check": False}
    assert _premotion_critic(demo_cfg, vision, _RT(), is_mock=False) is None
    demo_cfg._data["agent"] = {"premotion_check": "false"}   # YAML/env string form
    assert _premotion_critic(demo_cfg, vision, _RT(), is_mock=False) is None
    demo_cfg._data["agent"] = {"premotion_check": True, "premotion_max_checks": 1}
    assert _premotion_critic(demo_cfg, vision, _RT(), is_mock=False).budget.max_checks == 1
    del demo_cfg._data["agent"]                               # absent block -> default on
    assert _premotion_critic(demo_cfg, vision, _RT(), is_mock=False) is not None
    # the launcher kill switch wins over the config, both ways
    monkeypatch.setenv("CASCADE_PREMOTION_CHECK", "0")
    assert _premotion_critic(demo_cfg, vision, _RT(), is_mock=False) is None
    demo_cfg._data["agent"] = {"premotion_check": False}
    monkeypatch.setenv("CASCADE_PREMOTION_CHECK", "1")
    assert _premotion_critic(demo_cfg, vision, _RT(), is_mock=False) is not None
