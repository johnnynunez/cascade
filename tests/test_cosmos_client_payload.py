"""Exercise the real client-to-SDK boundary, not the config attribute."""
from types import SimpleNamespace

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
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: sdk)
    client = make_llm(Cfg({"type": "cosmos3", "model": "cosmos3-edge", "enable_thinking": thinking}))
    client.chat("Read-only test.", [{"role": "user", "content": "Report state."}])
    assert sent.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is thinking
