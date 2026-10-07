"""Hermes/Nous Portal as the default API-key brain, Codex as the default
subscription brain, and `--llm auto`.

The risk here is not a crash. It is that `--llm auto` picks a paid endpoint when
someone expected the offline mock, or picks mock when they expected their
configured brain -- both quiet, and one of them costs money.
"""

from __future__ import annotations

import json
import stat

import pytest

from cascade.agent import llm as llm_mod
from cascade.agent.llm import AUTO_LLM_PROFILES, LLM_ENV, resolve_llm_profile
from cascade.config import CONFIG_DIR, load_demo_config, load_profile

ALL_KEYS = [key for _, key in AUTO_LLM_PROFILES]


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch, tmp_path):
    """A developer's exported ANTHROPIC_API_KEY -- or the Codex login in his
    ~/.codex/auth.json -- must not change what this suite certifies; otherwise
    `auto` is tested as whatever the machine happens to have."""
    monkeypatch.delenv(LLM_ENV, raising=False)
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Codex auth detection reads $CODEX_HOME/auth.json: point it at nothing.
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex-home"))
    monkeypatch.delenv("CASCADE_CODEX_BIN", raising=False)


def _codex_login(tmp_path, monkeypatch, with_binary: bool = True):
    """Simulate a logged-in Codex CLI without touching the real ~/.codex."""
    home = tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    (home / "auth.json").write_text(json.dumps(
        {"auth_mode": "chatgpt", "OPENAI_API_KEY": None,
         "tokens": {"access_token": "test-token", "refresh_token": "r", "account_id": "a"}}))
    monkeypatch.setenv("CODEX_HOME", str(home))
    if with_binary:
        fake = tmp_path / "codex"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("CASCADE_CODEX_BIN", str(fake))
    else:
        # neither the override nor anything on PATH
        monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
        monkeypatch.delenv("CASCADE_CODEX_BIN", raising=False)


def test_auto_falls_back_to_mock_with_no_credentials():
    """A fresh clone must still run the documented offline wiring check."""
    assert resolve_llm_profile("auto") == "mock"


def test_auto_picks_hermes_first():
    """Hermes is the project default host; one Portal subscription covers both
    it and this brain profile. First among the API-key profiles: the Codex
    login check below runs before this table."""
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


# ── Codex first ──────────────────────────────────────────────────────────


def test_auto_picks_codex_astra_first_when_codex_is_logged_in(tmp_path, monkeypatch):
    """User decision: GPT-6-Astra through the Codex subscription is now the
    brain. A logged-in Codex CLI outranks every exported API key."""
    _codex_login(tmp_path, monkeypatch)
    assert llm_mod.codex_auth_present() is True
    assert resolve_llm_profile("auto") == llm_mod.AUTO_LLM_CODEX_PROFILE == "codex_astra"
    for k in ALL_KEYS:
        monkeypatch.setenv(k, "sk-test")
    assert resolve_llm_profile("auto") == "codex_astra"


def test_codex_login_without_the_cli_does_not_capture_auto(tmp_path, monkeypatch):
    """A leftover ~/.codex/auth.json on a box whose Codex CLI is gone must not
    send `auto` into a RuntimeError; it degrades to the API-key order."""
    _codex_login(tmp_path, monkeypatch, with_binary=False)
    assert llm_mod.codex_auth_present() is True
    assert resolve_llm_profile("auto") == "mock"
    monkeypatch.setenv("NOUS_API_KEY", "sk-test")
    assert resolve_llm_profile("auto") == "hermes"


def test_a_codex_login_without_a_token_is_not_a_login(tmp_path, monkeypatch):
    _codex_login(tmp_path, monkeypatch)
    (tmp_path / "codex-home" / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": None, "tokens": {}}))
    assert llm_mod.codex_auth_present() is False
    assert resolve_llm_profile("auto") == "mock"


def test_an_explicit_profile_still_pins_over_codex(tmp_path, monkeypatch):
    _codex_login(tmp_path, monkeypatch)
    assert resolve_llm_profile("hermes") == "hermes"
    assert resolve_llm_profile("mock") == "mock"
    monkeypatch.setenv(LLM_ENV, "local_qwen")
    assert resolve_llm_profile("auto") == "local_qwen"


def test_codex_astra_profile_exists_and_names_no_credential():
    path = CONFIG_DIR / "llm" / f"{llm_mod.AUTO_LLM_CODEX_PROFILE}.yaml"
    assert path.exists()
    cfg = load_profile("llm", llm_mod.AUTO_LLM_CODEX_PROFILE)
    assert cfg.get("type") == "codex_exec"
    assert cfg.get("model") == "gpt-6-astra"
    assert cfg.get("api_key") is None and cfg.get("api_key_env") is None
    text = path.read_text()
    assert "OPENAI_API_KEY" in text, "the header must say why no API key is involved"


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
