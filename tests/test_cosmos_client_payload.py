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


def test_cosmos_transport_uses_bounded_timeout_without_implicit_retries(monkeypatch):
    options = {}
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: options.update(kwargs))
    make_llm(Cfg({"type": "cosmos3", "request_timeout_s": 12.0, "max_retries": 0}))
    assert options["timeout"] == 12.0
    assert options["max_retries"] == 0


def test_cosmos_does_not_print_images_or_prompt(monkeypatch, capsys):
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="ready", tool_calls=[]))]))))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: sdk)
    client = make_llm(Cfg({"type": "cosmos3"}))
    response = client.chat("Private robot context", [{"role": "user", "content": "state", "images": [b'raw-camera-bytes']}])
    assert response.text == "ready"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("timeout", [0, -1, float('nan'), float('inf')])
def test_cosmos_rejects_invalid_timeout(monkeypatch, timeout):
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: pytest.fail("Invalid timeout reached SDK"))
    with pytest.raises(ValueError, match="positive finite"):
        make_llm(Cfg({"type": "cosmos3", "request_timeout_s": timeout}))


@pytest.mark.parametrize('arguments', [
    {'label':'</label>','camera':'d435i'},
    {'label':'ball','camera':'analyze_scene'},
    {'label':'ball','camera':'d435i','u':.5},
    {'camera':'d435i'},
])
def test_cosmos_rejects_malformed_tool_arguments_before_dispatch(monkeypatch, arguments):
    import json
    from cascade.config import load_demo_config
    from cascade.skills.runtime import tool_specs_for_config
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        call = SimpleNamespace(id='call_1',function=SimpleNamespace(
            name='locate_pixel', arguments=json.dumps(arguments)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='',tool_calls=[call]))])

    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    cfg = load_demo_config(arm='fr3',camera='d435i',llm='local_cosmos')
    tools = tool_specs_for_config(cfg)
    response = make_llm(cfg.llm).chat('Inspect only.', [{'role':'user','content':'locate ball'}], tools=tools)
    assert response.tool_calls == []
    assert 'Invalid model tool call' in response.text
    wire = next(t['function'] for t in sent['tools'] if t['function']['name']=='locate_pixel')
    assert wire['strict'] is True
    assert wire['parameters']['properties']['camera']['enum'] == ['d435i']
    assert wire['parameters']['additionalProperties'] is False
    assert all('strict' not in t for t in tools), 'client mutated shared schemas'
