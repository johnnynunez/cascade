"""A cosmos3 LLM profile reads its bearer key from a FILE (`api_key_file`), as the brain detector does (#416 review).

configs/llm/local_cosmos_sglang.yaml points at scripts/serve_cosmos_sglang.sh, which serves only with a key since #416:
`--llm local_cosmos_sglang` and `CASCADE_GROUNDING_LLM_PROFILE=local_cosmos_sglang` sent `api_key: EMPTY` and got 401.
The profile now names the key FILE (`${CASCADE_MODEL_API_KEY_FILE:-~/.cascade/nemoclaw/llama.key}`), never a key; the
cosmos3 client and the VLM grounder read it like `BrainDetector._read_key` (an unreadable or empty file sends no key,
with a warning naming the path). Profiles without `api_key_file` build exactly what they built before.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cascade.agent.llm import make_llm
from cascade.config import Cfg, load_profile

KEY = "cosmos3-profile-key-91d0"
DEFAULT_KEY = Path(".cascade") / "nemoclaw" / "llama.key"


@pytest.fixture
def sdk(monkeypatch):
    """The OpenAI SDK as the cosmos3 client builds it: records the constructor's options."""
    options = {}
    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: options.update(kwargs) or SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openai", module)
    return options


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CASCADE_MODEL_API_KEY", "CASCADE_MODEL_API_KEY_FILE", "CASCADE_BRAIN_KEY_FILE"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _key(path: Path, text: str = KEY + "\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_the_sglang_profile_names_the_brain_key_file_and_no_key(home):
    cfg = load_profile("llm", "local_cosmos_sglang").as_dict()
    assert cfg["api_key_file"] == str(home / DEFAULT_KEY)
    assert "api_key" not in cfg, "the profile holds no key, not even EMPTY"
    text = (Path(__file__).resolve().parents[1] / "configs/llm/local_cosmos_sglang.yaml").read_text()
    assert "private copy" not in text and "set `api_key`" not in text


def test_the_sglang_profile_sends_the_default_brain_key(home, sdk):
    _key(home / DEFAULT_KEY)
    make_llm(load_profile("llm", "local_cosmos_sglang"))
    assert sdk["api_key"] == KEY


def test_cascade_model_api_key_file_overrides_the_default(home, sdk, monkeypatch):
    _key(home / DEFAULT_KEY, "default-key\n")
    monkeypatch.setenv("CASCADE_MODEL_API_KEY_FILE", str(_key(home / "other.key")))
    make_llm(load_profile("llm", "local_cosmos_sglang"))
    assert sdk["api_key"] == KEY


@pytest.mark.parametrize("problem", ["missing", "empty"])
def test_an_unusable_key_file_sends_no_key_and_names_the_file(home, sdk, caplog, problem):
    path = home / "brain.key"
    if problem == "empty":
        _key(path, " \n")
    with caplog.at_level(logging.WARNING):
        make_llm(Cfg({"type": "cosmos3", "api_key_file": str(path)}))
    assert sdk["api_key"] == "EMPTY"
    said = {"missing": "is unreadable (FileNotFoundError); sending no key", "empty": "is empty; sending no key"}
    assert f"{path} {said[problem]}" in caplog.text, caplog.text


def test_api_key_file_wins_over_a_literal_key(home, sdk):
    make_llm(Cfg({"type": "cosmos3", "api_key": "literal", "api_key_file": str(_key(home / "brain.key"))}))
    assert sdk["api_key"] == KEY


@pytest.mark.parametrize("profile", ["local_cosmos", "local_qwen", "local_nemotron_bolt"])
def test_the_other_local_profiles_build_what_they_built_before(home, sdk, profile):
    """The brain key file present changes nothing for a profile without api_key_file."""
    _key(home / DEFAULT_KEY)
    cfg = load_profile("llm", profile)
    assert cfg.get("api_key_file") is None
    make_llm(cfg)
    assert sdk["api_key"] == "EMPTY"


# ── the grounding profile (CASCADE_GROUNDING_LLM_PROFILE) ──────────────────


def _ground(monkeypatch, profile: str) -> dict:
    """Configure the grounder as the MCP server does, then let the runtime build it (a recording VLMGrounder)."""
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.config import load_demo_config
    from cascade.memory import EpisodicMemory
    from cascade.skills.runtime import SkillRuntime
    from cascade.types import Detection, Frame

    built = []

    class Grounder:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def ground(self, frame, query):
            return Detection(label=query, conf=.5, bbox=np.array([10, 10, 30, 30]))

    monkeypatch.setattr("cascade.perception.vlm_ground.VLMGrounder", Grounder)
    monkeypatch.setenv("CASCADE_GROUNDING_LLM_PROFILE", profile)
    cfg = load_demo_config(arm="fr3", camera="d435i", llm="mock")
    McpSkillServer._configure_grounder(cfg)
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg, rt._grounder, rt.watcher, rt.memory = cfg, None, None, EpisodicMemory()
    frame = Frame(np.zeros((40, 40, 3), np.uint8), np.full((40, 40), .5), np.eye(3), T_base_cam=np.eye(4))
    assert rt._vlm_ground_fix(frame, "ball") is not None
    (kwargs,) = built
    return kwargs


def test_the_sglang_grounding_profile_sends_the_brain_key(home, monkeypatch):
    _key(home / DEFAULT_KEY)
    kwargs = _ground(monkeypatch, "local_cosmos_sglang")
    assert kwargs["api_key"] == KEY
    assert kwargs["base_url"] == "http://127.0.0.1:8083/v1"


def test_the_vllm_grounding_profile_builds_the_grounder_as_before(home, monkeypatch):
    _key(home / DEFAULT_KEY)
    kwargs = _ground(monkeypatch, "local_cosmos")
    assert kwargs == {"base_url": "http://127.0.0.1:8082/v1", "model": "cosmos3-edge", "timeout_s": 45.0,
                      "enable_thinking": False}
