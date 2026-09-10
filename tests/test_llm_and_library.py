
from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.skills.library import SkillLibrary
from cascade.skills.runtime import TOOL_SPECS


def test_tool_specs_are_valid_schemas():
    names = set()
    for spec in TOOL_SPECS:
        assert spec["name"] not in names
        names.add(spec["name"])
        assert spec["description"]
        params = spec["parameters"]
        assert params["type"] == "object"
        for req in params.get("required", []):
            assert req in params["properties"]
    # Every spec maps to a runtime method.
    from cascade.skills.runtime import SkillRuntime

    for spec in TOOL_SPECS:
        assert hasattr(SkillRuntime, f"skill_{spec['name']}")


def test_every_skill_method_has_a_spec_and_motion_skills_exist():
    """The reverse of the check above -- CLAUDE.md warned for months that
    'forgetting the spec entry fails no test, the skill just never becomes
    visible to the LLM/MCP'. Now it fails a test. Also: every name in
    _MOTION_SKILLS must be a real skill, or a typo there silently stops
    pausing belief fusion for the skill it meant."""
    import re

    from cascade.skills.runtime import _MOTION_SKILLS, SkillRuntime

    methods = {m[len("skill_"):] for m in dir(SkillRuntime) if m.startswith("skill_")}
    specs = {t["name"] for t in TOOL_SPECS}
    assert methods == specs, (
        f"skills without a TOOL_SPECS entry: {sorted(methods - specs)}; "
        f"specs without a method: {sorted(specs - methods)}"
    )
    assert _MOTION_SKILLS <= specs, sorted(_MOTION_SKILLS - specs)
    # README headline counts are derived from these numbers; keep them honest
    readme = (pathlib_repo() / "README.md").read_text()
    n_skills = len(specs) - 1  # task_done is loop-internal
    assert f"## The {n_skills} skills" in readme, f"README still says a different skill count than {n_skills}"
    m = re.search(r"(\d+) tools total", readme)
    from cascade.apps.mcp_server import _EXCLUDED_TOOLS, _EXTRA_TOOLS

    n_tools = len(specs - _EXCLUDED_TOOLS) + len(_EXTRA_TOOLS)
    assert m and int(m.group(1)) == n_tools, (m.group(0) if m else None, n_tools)


def pathlib_repo():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def test_mock_llm_scripting():
    llm = MockLLM([LLMResponse(text="hi"), LLMResponse(tool_calls=[ToolCall("a", {})])])
    r1 = llm.chat("sys", [{"role": "user", "content": "x"}])
    assert r1.text == "hi"
    r2 = llm.chat("sys", [])
    assert r2.tool_calls[0].name == "a"
    r3 = llm.chat("sys", [])
    assert "exhausted" in r3.text
    assert len(llm.requests) == 3


class _StubRuntime:
    """Minimal runtime for orchestrator-only behavior tests."""

    def __init__(self):
        from cascade.memory import EpisodicMemory

        self.memory = EpisodicMemory()
        self.last_frame = None
        self.calls = []

        class _T:
            def finish(self, s):
                self.summary = s

        self.trace = _T()

    def execute(self, name, args):
        self.calls.append((name, args))
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True}

    def frame_jpeg(self):
        return None


def test_orchestrator_nudges_on_text_only_reply():
    rt = _StubRuntime()
    llm = MockLLM(
        [
            LLMResponse(text="thinking out loud, no tool"),
            LLMResponse(tool_calls=[ToolCall("task_done", {"success": True, "summary": "ok"})]),
        ]
    )
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task("t")
    assert report.success
    # Second request must contain the nudge.
    msgs = llm.requests[1]["messages"]
    assert any("tool call" in str(m.get("content", "")) for m in msgs)


def test_orchestrator_step_budget():
    rt = _StubRuntime()
    llm = MockLLM([LLMResponse(tool_calls=[ToolCall("get_observation", {})]) for _ in range(20)])
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=3).run_task("t")
    assert not report.success
    assert report.steps == 3


def test_skill_library_roundtrip(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(
        title="Bottle grasp yaw",
        failure="jaws slip off cylindrical bottles",
        when="grasping a bottle or cylinder",
        strategy="Use the OBB major-axis yaw and two-stage close at 50%/70%.",
        origin="wrc bring-up",
    )
    entries = lib.entries()
    assert len(entries) == 1
    hits = lib.relevant("please grasp the bottle on the left")
    assert len(hits) == 1 and "two-stage" in hits[0]
    assert lib.relevant("wave hello") == []
