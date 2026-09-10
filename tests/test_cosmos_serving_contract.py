"""Cosmos sidecar contracts. No CUDA, checkpoint download or project installs."""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "serve_cosmos_vllm.sh"
REVISION = "a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"


def run_script(tmp_path, *args, overrides=None):
    # Even the old script must not install anything during the RED run.
    commands = tmp_path / "commands"
    commands.mkdir(exist_ok=True)
    uv = commands / "uv"
    uv.write_text('#!/bin/sh\nprintf "UNEXPECTED_INSTALL\\n" >&2\nexit 97\n')
    uv.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{commands}:{os.environ['PATH']}",
        VENV=str(tmp_path / "isolated cosmos"),
        MODEL_DIR=str(tmp_path / "models"),
        COSMOS_PYTHON=sys.executable,
    )
    for key in (
        "EXPORT_DIR",
        "HF_REPO",
        "MODEL_REVISION",
        "SERVED_NAME",
        "PORT",
        "CTX",
        "GPU_FRAC",
    ):
        env.pop(key, None)
    env.update(overrides or {})
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_dry_run_pins_isolated_native_openai_contract_without_writes(tmp_path):
    result = run_script(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["venv"] == str(tmp_path / "isolated cosmos")
    assert plan["model"] == {
        "repository": "nvidia/Cosmos3-Edge",
        "revision": REVISION,
        "served_name": "cosmos3-edge",
    }
    assert {"vllm==0.29.0", "torch==2.13.0", "transformers==5.17.0"} <= set(
        plan["requirements"]
    )
    argv = plan["serve_command"]
    assert argv[0] == str(tmp_path / "isolated cosmos" / "bin" / "vllm")
    assert argv[1] == "serve"
    assert argv[argv.index("--served-model-name") + 1] == "cosmos3-edge"
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--enable-auto-tool-choice" in argv
    assert "--enforce-eager" in argv
    assert not (tmp_path / "isolated cosmos").exists()
    assert not (tmp_path / "models").exists()
    assert "git+" not in result.stdout


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "cosmos_serving", REPO / "scripts/cosmos_serving.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", [[], ["--setup-only"], ["--check"]])
def test_mac_runtime_refuses_cuda_before_installing(
    tmp_path, monkeypatch, capsys, mode
):
    helper = load_helper()
    monkeypatch.setenv("VENV", str(tmp_path / "cosmos"))
    monkeypatch.setattr(helper.platform, "system", lambda: "Darwin")
    assert helper.main(mode) == 1
    assert "Linux" in capsys.readouterr().err
    assert not (tmp_path / "cosmos").exists()


def test_help_and_invalid_flags_never_touch_environment(tmp_path):
    help_result = run_script(tmp_path, "--help")
    assert help_result.returncode == 0
    assert "--setup-only" in help_result.stdout
    assert "--check" in help_result.stdout
    bad = run_script(tmp_path, "--does-not-exist")
    assert bad.returncode == 2
    assert "UNEXPECTED_INSTALL" not in bad.stderr
    assert not (tmp_path / "isolated cosmos").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("VENV", str(REPO / ".venv")),
        ("VENV", str(REPO / ".isaacsim")),
        ("PORT", "0"),
        ("CTX", "0"),
        ("GPU_FRAC", "nan"),
        ("MODEL_REVISION", "main"),
        ("HF_REPO", "other/model"),
    ],
)
def test_invalid_or_unsafe_plan_refused_before_install(tmp_path, key, value):
    result = run_script(tmp_path, "--dry-run", overrides={key: value})
    assert result.returncode != 0
    assert "UNEXPECTED_INSTALL" not in result.stderr
    assert not (tmp_path / "models").exists()


def test_environment_reconciles_pins_without_recreating_existing_venv(
    tmp_path, monkeypatch
):
    helper = load_helper()
    venv = tmp_path / "cosmos"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").touch()
    (venv / "pyvenv.cfg").write_text("version = 3.12.14\n")
    sentinel = venv / "keep-me"
    sentinel.write_text("existing data")
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    plan = helper.serving_plan({"VENV": str(venv)})
    helper.ensure_environment(plan)
    helper.ensure_environment(plan)
    assert all(c[1:3] == ["pip", "install"] for c in calls)
    assert len(calls) == 2
    assert all(c[3:5] == ["--python", str(venv / "bin/python")] for c in calls)
    assert all(set(plan["requirements"]) <= set(c) for c in calls)
    assert sentinel.read_text() == "existing data"


def test_environment_refuses_non_venv_directory(tmp_path):
    helper = load_helper()
    venv = tmp_path / "valuable"
    venv.mkdir()
    (venv / "data").write_text("preserve")
    with pytest.raises(RuntimeError, match="not a virtual environment"):
        helper.ensure_environment(helper.serving_plan({"VENV": str(venv)}))
    assert (venv / "data").read_text() == "preserve"


def test_check_never_calls_setup_on_linux(tmp_path, monkeypatch, capsys):
    helper = load_helper()
    monkeypatch.setattr(helper.platform, "system", lambda: "Linux")
    monkeypatch.setenv("VENV", str(tmp_path / "missing"))
    monkeypatch.setattr(
        helper,
        "ensure_environment",
        lambda *_: pytest.fail("read-only check attempted install"),
    )
    assert helper.main(["--check"]) == 1
    assert "--setup-only" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


def write_tiny_export(path):
    """Format fixture, NOT Cosmos weights or an inference result."""
    import struct

    path.mkdir(exist_ok=True)
    docs = {
        "config.json": {
            "model_type": "cosmos3_edge",
            "architectures": ["Cosmos3EdgeForConditionalGeneration"],
        },
        "processor_config.json": {
            "processor_class": "Cosmos3EdgeProcessor",
            "image_processor": {},
            "video_processor": {},
        },
        "tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"},
        "tokenizer.json": {"model": {"type": "BPE"}},
    }
    for name, value in docs.items():
        (path / name).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text(
        "<tool_call><function=example><parameter=value>"
    )
    header = json.dumps(
        {"test.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    (path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 8
    )


@pytest.mark.parametrize("tamper", ["revision", "bytes"])
def test_upstream_probe_refuses_stale_or_modified_source_cache(tmp_path, tamper):
    import hashlib

    spec = importlib.util.spec_from_file_location(
        "cosmos_probe_contract", REPO / "scripts/cosmos_upstream_probe.py"
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert hasattr(probe, "validate_sources"), "offline probe must verify its source revision"
    manifest = {}
    # Synthetic cache bytes only test provenance rejection; never executed.
    for name, url in probe.source_urls().items():
        content = b"# inert source-cache fixture\n"
        (tmp_path / name).write_bytes(content)
        manifest[name] = {"url": url, "bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()}
    path = tmp_path / "probe-sources.json"
    path.write_text(json.dumps(manifest))
    probe.validate_sources(tmp_path)
    name = next(iter(probe.SOURCES))
    if tamper == "revision":
        manifest[name]["url"] = manifest[name]["url"].replace(probe.VLLM_SHA, "0" * 40)
        path.write_text(json.dumps(manifest))
    else:
        (tmp_path / name).write_bytes(b"# changed\n")
    with pytest.raises(ValueError, match="source cache"):
        probe.validate_sources(tmp_path)


def test_atomic_export_promotes_once_only_after_full_validation(tmp_path):
    helper = load_helper()
    target = tmp_path / "export"
    target.mkdir()
    (target / "config.json").write_text('{"incomplete":true}')
    seen = []

    def builder(stage):
        assert (target / "config.json").read_text() == '{"incomplete":true}'
        assert stage.parent == target.parent
        assert stage != target
        seen.append(stage)
        write_tiny_export(stage)

    assert helper.atomic_export(target, builder) == "exported"
    assert helper.validate_export(target)["revision"] == REVISION
    assert (
        helper.atomic_export(target, lambda _: pytest.fail("valid export rebuilt"))
        == "reused"
    )
    assert len(seen) == 1


def test_failed_export_preserves_old_directory_and_never_leaves_valid_marker(tmp_path):
    helper = load_helper()
    target = tmp_path / "export"
    target.mkdir()
    (target / "config.json").write_text("old")

    def interrupted(stage):
        write_tiny_export(stage)
        (stage / "model.safetensors").write_bytes(b"truncated")

    with pytest.raises((ValueError, RuntimeError)):
        helper.atomic_export(target, interrupted)
    assert (target / "config.json").read_text() == "old"
    assert not (target / "cosmos-export.json").exists()
    assert not list(tmp_path.glob(".export.staging-*"))


@pytest.mark.parametrize(
    "damage",
    [
        "config_only",
        "missing_processor",
        "truncated_weights",
        "missing_shard",
        "wrong_revision",
    ],
)
def test_cached_export_is_not_complete_just_because_config_exists(tmp_path, damage):
    helper = load_helper()
    target = tmp_path / "export"
    helper.atomic_export(target, write_tiny_export)
    if damage == "config_only":
        for p in target.iterdir():
            if p.name != "config.json":
                p.unlink()
    elif damage == "missing_processor":
        (target / "processor_config.json").unlink()
    elif damage == "truncated_weights":
        p = target / "model.safetensors"
        p.write_bytes(p.read_bytes()[:-1])
    elif damage == "missing_shard":
        (target / "model.safetensors").unlink()
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"test.weight": "missing.safetensors"}})
        )
    else:
        p = target / "cosmos-export.json"
        metadata = json.loads(p.read_text())
        metadata["revision"] = "main"
        p.write_text(json.dumps(metadata))
    with pytest.raises((ValueError, RuntimeError, OSError)):
        helper.validate_export(target)


def test_upstream_fixed_rope_is_success_without_rewriting_unknown_source(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    helper = load_helper()
    source = tmp_path / "modeling.py"
    source.write_text("# future implementation without the old guard\n")
    module = SimpleNamespace(__file__=str(source))
    monkeypatch.setattr(helper, "rope_smoke", lambda _: {"cases": 6})
    assert (
        helper.ensure_rope_compat(module, version="future", allow_patch=True)["status"]
        == "upstream-ok"
    )
    assert source.read_text() == "# future implementation without the old guard\n"


def test_rope_patch_is_exact_version_scoped_and_read_only_check_refuses_patch(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    helper = load_helper()
    source = tmp_path / "modeling.py"
    source.write_text("# changed upstream\n")
    module = SimpleNamespace(__file__=str(source))

    def broken(_):
        raise IndexError("too many indices")

    monkeypatch.setattr(helper, "rope_smoke", broken)
    with pytest.raises(RuntimeError, match="--setup-only"):
        helper.ensure_rope_compat(module, version="5.17.0", allow_patch=False)
    with pytest.raises(RuntimeError, match="Unrecognized"):
        helper.ensure_rope_compat(module, version="future", allow_patch=True)
    assert source.read_text() == "# changed upstream\n"


def test_real_cuda_availability_not_import_or_nvidia_smi_is_required(monkeypatch):
    from types import SimpleNamespace

    helper = load_helper()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda=None),
        ),
    )
    with pytest.raises(RuntimeError, match=r"torch.cuda.is_available.*False"):
        helper.check_cuda()


def test_export_builder_pins_both_loads_and_rejects_missing_weights(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    helper = load_helper()
    calls = []
    info = {"missing_keys": [], "mismatched_keys": []}

    def save_model(path, **kwargs):
        assert kwargs["save_original_format"] is False
        write_tiny_export(path)
        config_file = path / "config.json"
        config = json.loads(config_file.read_text())
        config["allow_patterns_overrides"] = ["*/*.safetensors"]
        config_file.write_text(json.dumps(config))

    model = SimpleNamespace(save_pretrained=save_model)
    processor = SimpleNamespace(
        save_pretrained=lambda path: calls.append(("save_processor", str(path)))
    )

    def load_model(repo, **kwargs):
        calls.append(("model", repo, kwargs))
        return model, info

    def load_processor(repo, **kwargs):
        calls.append(("processor", repo, kwargs))
        return processor

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16-fixture"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Cosmos3EdgeForConditionalGeneration=SimpleNamespace(
                from_pretrained=load_model
            ),
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        ),
    )
    monkeypatch.setattr(
        helper,
        "processor_smoke",
        lambda path: calls.append(("processor_smoke", str(path))),
    )
    helper.build_export(tmp_path)
    assert [c[0] for c in calls] == [
        "model",
        "processor",
        "save_processor",
        "processor_smoke",
    ]
    for c in calls[:2]:
        assert c[1] == "nvidia/Cosmos3-Edge"
        assert c[2]["revision"] == REVISION
    assert calls[0][2]["output_loading_info"] is True
    assert "allow_patterns_overrides" not in json.loads(
        (tmp_path / "config.json").read_text()
    )
    info["missing_keys"] = ["model.language_model.layers.0.weight"]
    with pytest.raises(RuntimeError, match="missing"):
        helper.build_export(tmp_path)


def test_prepared_runtime_runs_cuda_export_and_exec_in_order(tmp_path, monkeypatch):
    helper = load_helper()
    monkeypatch.setenv("VENV", str(tmp_path / "cosmos"))
    monkeypatch.setenv("EXPORT_DIR", str(tmp_path / "export"))
    monkeypatch.setattr(helper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(helper.sys, "prefix", str(tmp_path / "cosmos"))
    events = []
    monkeypatch.setattr(helper, "check_stack", lambda: events.append("stack"))
    monkeypatch.setattr(helper, "check_cuda", lambda: events.append("cuda"))
    monkeypatch.setattr(
        helper, "ensure_rope_compat", lambda **kwargs: events.append("rope")
    )
    monkeypatch.setattr(
        helper, "atomic_export", lambda path, builder: events.append("export")
    )
    monkeypatch.setattr(
        helper, "processor_smoke", lambda path: events.append("processor")
    )

    class ExecBoundary(Exception):
        pass

    def exec_server(executable, arguments, environment):
        events.append("serve")
        assert arguments[0] == str(tmp_path / "cosmos/bin/vllm")
        assert "--enable-auto-tool-choice" in arguments
        assert environment["PATH"].split(os.pathsep)[0] == str(tmp_path / "cosmos/bin")
        raise ExecBoundary

    monkeypatch.setattr(helper.os, "execve", exec_server)
    with pytest.raises(ExecBoundary):
        helper.main(["--_prepared"])
    assert events == ["stack", "cuda", "rope", "export", "processor", "serve"]


@pytest.mark.parametrize("raw", ["null", "[]", '"broken"'])
def test_corrupt_completion_manifest_rebuilds_safely(tmp_path, raw):
    helper = load_helper()
    target = tmp_path / "export"
    helper.atomic_export(target, write_tiny_export)
    (target / "cosmos-export.json").write_text(raw)
    assert helper.atomic_export(target, write_tiny_export) == "exported"
    helper.validate_export(target)


@pytest.mark.parametrize(
    "destination",
    [str(REPO), str(REPO.parent), str(REPO / ".venv"), str(REPO / ".isaacsim/models")],
)
def test_export_destination_cannot_replace_checkout_or_environments(
    tmp_path, destination
):
    result = run_script(tmp_path, "--dry-run", overrides={"EXPORT_DIR": destination})
    assert result.returncode != 0
    assert "EXPORT_DIR" in result.stderr


def test_export_detects_same_size_weight_corruption(tmp_path):
    helper = load_helper()
    target = tmp_path / "export"
    helper.atomic_export(target, write_tiny_export)
    weights = target / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksums"):
        helper.validate_export(target)


def test_venv_symlink_cannot_bypass_project_environment_guard(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(REPO / ".venv", target_is_directory=True)
    result = run_script(tmp_path, "--dry-run", overrides={"VENV": str(alias)})
    assert result.returncode != 0
    assert "VENV must be isolated" in result.stderr
