"""Profile string expansion: `${NAME:-default}` from the environment and a leading `~`.

The brain detector's endpoint lives in configs/demo.yaml as
`${CASCADE_PERCEPTION_URL:-http://127.0.0.1:8080/v1}` and its key file as
`${CASCADE_PERCEPTION_KEY_FILE:-~/.cascade/nemoclaw/llama.key}`, so the loader
must resolve both forms; `${repo}` / `${assets}` keep working as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.config import (
    ASSET_DIR,
    PACKAGE_ROOT,
    _resolve_paths,
    load_demo_config,
    load_profile,
)

VAR = "CASCADE_TEST_EXPANSION_VAR"


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def resolve(value):
    return _resolve_paths(value, PACKAGE_ROOT / "configs")


def test_env_default_takes_the_variable_when_it_is_set(monkeypatch):
    monkeypatch.setenv(VAR, "http://brain:9000/v1")
    assert resolve(f"${{{VAR}:-http://x/v1}}") == "http://brain:9000/v1"


def test_env_default_takes_the_default_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    assert resolve(f"${{{VAR}:-http://x/v1}}") == "http://x/v1"


def test_an_empty_variable_counts_as_unset(monkeypatch):
    # The shell's `${VAR:-default}`: an exported-but-empty variable is unset.
    monkeypatch.setenv(VAR, "")
    assert resolve(f"${{{VAR}:-http://x/v1}}") == "http://x/v1"


def test_env_default_inside_a_longer_string(monkeypatch):
    monkeypatch.setenv(VAR, "gx10")
    assert resolve(f"http://${{{VAR}:-localhost}}:8081/v1") == "http://gx10:8081/v1"


def test_an_empty_default_is_allowed(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    assert resolve(f"a${{{VAR}:-}}b") == "ab"


def test_a_variable_value_is_not_expanded_again(monkeypatch):
    monkeypatch.setenv(VAR, "${OTHER:-x}")
    assert resolve(f"${{{VAR}:-d}}") == "${OTHER:-x}"


def test_a_bare_placeholder_without_default_stays_literal(monkeypatch):
    monkeypatch.setenv(VAR, "set")
    assert resolve(f"${{{VAR}}}") == f"${{{VAR}}}"
    assert resolve("${unknown}/x") == "${unknown}/x"


def test_leading_tilde_slash_expands_to_home(home):
    assert resolve("~/.cascade/k") == str(home / ".cascade" / "k")


def test_a_lone_tilde_expands_to_home(home):
    assert resolve("~") == str(home)


@pytest.mark.parametrize("value", ["~root/x", "~user/x", "a~/b", "x/~/y", "~~/z", " ~/k"])
def test_tilde_elsewhere_stays_literal(home, value):
    assert resolve(value) == value


def test_env_default_with_a_tilde_default_expands_home(monkeypatch, home):
    monkeypatch.delenv(VAR, raising=False)
    assert resolve(f"${{{VAR}:-~/.cascade/nemoclaw/llama.key}}") == str(
        home / ".cascade" / "nemoclaw" / "llama.key")


def test_a_tilde_in_the_variable_value_expands_home(monkeypatch, home):
    monkeypatch.setenv(VAR, "~/keys/brain.key")
    assert resolve(f"${{{VAR}:-/etc/none}}") == str(home / "keys" / "brain.key")


def test_strings_without_placeholders_are_unchanged():
    for value in ("plain", "http://127.0.0.1:8081/v1", "$HOME/x", "${", "a}b", "$x:-y", ""):
        assert resolve(value) == value


def test_repo_and_assets_placeholders_still_expand():
    assert resolve("${repo}/models/x.pt") == f"{PACKAGE_ROOT}/models/x.pt"
    assert resolve("${assets}/urdf/a.urdf") == f"{ASSET_DIR}/urdf/a.urdf"


def test_repo_placeholder_inside_a_default(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    assert resolve(f"${{{VAR}:-${{repo}}/models/sam.pt}}") == f"{PACKAGE_ROOT}/models/sam.pt"


def test_nested_containers_expand_and_other_types_pass_through(monkeypatch, home):
    monkeypatch.setenv(VAR, "v")
    data = {"a": [f"${{{VAR}:-d}}", {"b": "~/k"}], "n": 20, "f": 4.0, "t": True, "z": None}
    assert resolve(data) == {"a": ["v", {"b": str(home / "k")}], "n": 20, "f": 4.0, "t": True, "z": None}


def test_profiles_expand_env_defaults(monkeypatch, tmp_path):
    (tmp_path / "cameras").mkdir()
    (tmp_path / "cameras" / "probe.yaml").write_text(
        f"type: mock\nendpoint: ${{{VAR}:-http://default/v1}}\n")
    monkeypatch.setenv(VAR, "http://from-env/v1")
    assert load_profile("cameras", "probe", tmp_path).endpoint == "http://from-env/v1"


@pytest.fixture
def perception_env(monkeypatch):
    for name in ("CASCADE_PERCEPTION_URL", "CASCADE_PERCEPTION_MODEL", "CASCADE_PERCEPTION_KEY_FILE",
                 "CASCADE_BOOTH", "CASCADE_ROBOT", "CASCADE_BASE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_demo_yaml_brain_detector_defaults(perception_env, home):
    det = load_demo_config().detector
    assert det.type == "brain"
    assert det.base_url == "http://127.0.0.1:8080/v1"
    assert det.model == "nemotron-3.5-bolt"
    assert det.api_key_file == str(home / ".cascade" / "nemoclaw" / "llama.key")


def test_demo_yaml_brain_detector_follows_the_perception_variables(perception_env, tmp_path):
    perception_env.setenv("CASCADE_PERCEPTION_URL", "http://10.0.0.7:8000/v1")
    perception_env.setenv("CASCADE_PERCEPTION_MODEL", "qwen-other")
    perception_env.setenv("CASCADE_PERCEPTION_KEY_FILE", str(tmp_path / "brain.key"))
    det = load_demo_config().detector
    assert (det.base_url, det.model, det.api_key_file) == (
        "http://10.0.0.7:8000/v1", "qwen-other", str(tmp_path / "brain.key"))


def test_no_shipped_config_still_names_yoloe():
    """Ruling 3: every profile left YOLOE in the same change that made the
    brain the default (comments included, so nobody re-reads a stale story)."""
    offenders = [f"{p.relative_to(PACKAGE_ROOT)}:{i}"
                 for p in sorted((PACKAGE_ROOT / "configs").rglob("*.yaml"))
                 for i, line in enumerate(Path(p).read_text().splitlines(), 1)
                 if "yoloe" in line.lower()]
    assert not offenders, offenders
