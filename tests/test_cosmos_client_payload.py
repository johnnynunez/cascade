"""Exercise the real client-to-SDK boundary, not the config attribute."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from cascade.agent.llm import make_llm
from cascade.config import Cfg


@pytest.mark.parametrize("thinking", [False, True])
def test_cosmos_thinking_setting_reaches_the_http_sdk(monkeypatch, thinking):
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))])

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: sdk
    monkeypatch.setitem(sys.modules, "openai", module)
    client = make_llm(Cfg({"type": "cosmos3", "model": "cosmos3-edge", "enable_thinking": thinking}))
    client.chat("Read-only test.", [{"role": "user", "content": "Report state."}])
    assert sent.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is thinking
