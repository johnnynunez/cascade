"""Hermes/Nous Portal as the default brain, and `--llm auto`.

The risk here is not a crash. It is that `--llm auto` picks a paid endpoint when
someone expected the offline mock, or picks mock when they expected their
configured brain -- both quiet, and one of them costs money.
"""

from __future__ import annotations

import pytest

from cascade.agent import llm as llm_mod
from cascade.agent.llm import AUTO_LLM_PROFILES, LLM_ENV, resolve_llm_profile
from cascade.config import CONFIG_DIR, load_demo_config, load_profile

ALL_KEYS = [key for _, key in AUTO_LLM_PROFILES]


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """A developer's exported ANTHROPIC_API_KEY must not change what this suite
    certifies -- otherwise `auto` is tested as whatever the machine happens to
    have."""
    monkeypatch.delenv(LLM_ENV, raising=False)
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_auto_falls_back_to_mock_with_no_credentials():
    """A fresh clone must still run the documented offline wiring check."""
    assert resolve_llm_profile("auto") == "mock"


def test_auto_picks_hermes_first():
    """Hermes is the project default host; one Portal subscription covers both
    it and this brain profile."""
    assert AUTO_LLM_PROFILES[0] == ("hermes", "NOUS_API_KEY")


@pytest.mark.parametrize("profile,key", AUTO_LLM_PROFILES)
def test_auto_selects_the_profile_whose_key_is_present(monkeypatch, profile, key):
    monkeypatch.setenv(key, "sk-test")
    assert resolve_llm_profile("auto") == profile


def test_auto_respects_the_documented_priority(monkeypatch):
    """With several keys set the order must be deterministic, not dict order."""
    for k in ALL_KEYS:
        monkeypatch.setenv(k, "sk-test")
    assert resolve_llm_profile("auto") == AUTO_LLM_PROFILES[0][0]


def test_an_explicit_profile_is_never_second_guessed(monkeypatch):
    """`--llm mock` on a fully credentialled rig must stay offline: this is how
    tests and dry-runs avoid spending tokens."""
    monkeypatch.setenv("NOUS_API_KEY", "sk-test")
    assert resolve_llm_profile("mock") == "mock"
    assert resolve_llm_profile("local_qwen") == "local_qwen"


def test_env_var_overrides_the_flag_default(monkeypatch):
    monkeypatch.setenv(LLM_ENV, "anthropic")
    assert resolve_llm_profile("auto") == "anthropic"
    assert resolve_llm_profile(None) == "anthropic"


def test_blank_credentials_do_not_count(monkeypatch):
    """An exported-but-empty key is a common shell accident, and treating it as
    present turns a mock run into a 401."""
    monkeypatch.setenv("NOUS_API_KEY", "   ")
    assert resolve_llm_profile("auto") == "mock"


# ── the profile itself ───────────────────────────────────────────────────


def test_every_auto_profile_exists_on_disk():
    """`auto` naming a profile that was renamed would fail only on the machine
    that has that key set."""
    for profile, _ in AUTO_LLM_PROFILES:
        assert (CONFIG_DIR / "llm" / f"{profile}.yaml").exists(), profile


def test_hermes_profile_points_at_portal_with_its_own_key_variable():
    cfg = load_profile("llm", "hermes")
    assert cfg.get("type") == "openai_compat"
    assert "inference-api.nousresearch.com" in cfg.get("base_url")
    # OpenAI-compatible does not mean OpenAI-keyed: without api_key_env the
    # client looks for OPENAI_API_KEY and sends an empty credential to Portal.
    assert cfg.get("api_key_env") == "NOUS_API_KEY"
    assert cfg.get("api_key") is None


def test_hermes_profile_loads_as_the_demo_brain():
    cfg = load_demo_config(camera="mock", arm="so101_mock", llm="hermes")
    assert cfg.llm.get("api_key_env") == "NOUS_API_KEY"
    assert cfg.llm.get("model")


@pytest.fixture
def fake_openai(monkeypatch):
    """A stub `openai` module.

    Deliberately not `importorskip("openai")`: the real package lives in the
    optional `llm` extra, which CI does not install, so a skip here would mean
    the api_key_env contract is verified on NO machine. The stub records what
    the client was constructed with, which is the whole assertion.
    """
    import sys
    import types

    calls: list[dict] = []

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None, **kw):
            calls.append({"base_url": base_url, "api_key": api_key, **kw})
            self.base_url = base_url
            self.api_key = api_key

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


def test_a_named_key_variable_is_required_not_silently_empty(monkeypatch, fake_openai):
    """The failure has to be loud at construction. A client built with no
    credential fails later, mid-episode, as an opaque 401 from a tool call."""
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NOUS_API_KEY"):
        llm_mod.make_llm(load_profile("llm", "hermes"))
    assert not fake_openai, "must fail before building a keyless client"


def test_a_named_key_variable_is_used_when_present(monkeypatch, fake_openai):
    monkeypatch.setenv("NOUS_API_KEY", "sk-portal-test")
    client = llm_mod.make_llm(load_profile("llm", "hermes"))
    assert client.supports_vision is False, "Hermes 4.5 is text-only"
    assert fake_openai[0]["api_key"] == "sk-portal-test"
    assert "nousresearch.com" in fake_openai[0]["base_url"]


def test_openai_api_key_is_not_borrowed_for_another_provider(monkeypatch, fake_openai):
    """The bug api_key_env exists to prevent: sending an OpenAI credential (or
    the "EMPTY" local-server placeholder) to Portal because the profile named a
    base_url but the client only knew one key variable."""
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-someone-elses")
    with pytest.raises(RuntimeError, match="NOUS_API_KEY"):
        llm_mod.make_llm(load_profile("llm", "hermes"))


def test_a_local_server_profile_still_gets_the_placeholder_key(monkeypatch, fake_openai):
    """llama.cpp/vLLM need a dummy key and name no api_key_env; that path must
    be untouched by the change."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm_mod.make_llm(load_profile("llm", "local_qwen"))
    assert fake_openai[0]["api_key"] == "EMPTY"


def test_openai_profile_still_uses_the_openai_variable():
    """Adding api_key_env must not have moved the default provider's key."""
    cfg = load_profile("llm", "openai")
    assert cfg.get("api_key_env") is None
