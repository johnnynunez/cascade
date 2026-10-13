"""An MCP host can reuse the selected vision profile without an agent brain."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cascade.apps.mcp_server import McpSkillServer
from cascade.config import load_demo_config, load_profile


def test_default_mcp_configuration_does_not_enable_external_grounding(monkeypatch):
    monkeypatch.delenv("CASCADE_GROUNDING_LLM_PROFILE", raising=False)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    before = cfg.as_dict()
    McpSkillServer._configure_grounder(cfg)
    assert cfg.as_dict() == before


def test_explicit_vision_profile_is_used_by_real_mcp_build_without_chat_brain(monkeypatch):
    from cascade.apps import demo

    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.delenv("CASCADE_ROBOT", raising=False)
    monkeypatch.setenv("CASCADE_ARM", "fr3")
    monkeypatch.setenv("CASCADE_CAMERA", "d435i")
    monkeypatch.setenv("CASCADE_GROUNDING_LLM_PROFILE", "local_cosmos")
    monkeypatch.setattr(demo, "make_llm", lambda *_: pytest.fail("MCP constructed a chat brain"))
    built = []
    rt = SimpleNamespace(stream_server=None)

    def build(cfg, *args, **kwargs):
        built.append(cfg)
        return rt, object()

    monkeypatch.setattr(demo, "build_runtime", build)
    server = McpSkillServer()
    server._stop_pending = False
    monkeypatch.setattr(server, "_attach_program_tier", Mock())
    monkeypatch.setattr(server, "_announce_surface", Mock())
    assert server._ensure_runtime() is rt
    cfg = built[0]
    source = load_profile("llm", "local_cosmos")
    assert cfg.llm.type == "mock"
    assert cfg.grounder.base_url == source.base_url
    assert cfg.grounder.model == source.model
    assert cfg.grounder.enable_thinking is False
    assert cfg.grounder.timeout_s == source.request_timeout_s


def test_nonvision_profile_is_refused_before_runtime_build(monkeypatch):
    from cascade.apps import demo

    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.delenv("CASCADE_ROBOT", raising=False)
    monkeypatch.setenv("CASCADE_GROUNDING_LLM_PROFILE", "mock")
    build = Mock(side_effect=AssertionError("unexpected runtime build"))
    monkeypatch.setattr(demo, "build_runtime", build)
    with pytest.raises(ValueError, match="requires vision"):
        McpSkillServer()._ensure_runtime()
    build.assert_not_called()
