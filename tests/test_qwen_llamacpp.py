"""Exercise owned/default and explicit GGUF/server boundaries without large downloads."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / "scripts/serve_qwen_llamacpp.sh", repo / "scripts/serve_qwen_llamacpp.sh")
    directory = repo / "models/qwen3.8-27b"
    directory.mkdir(parents=True)
    for name in ("Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj-Qwen3.8-27B-BF16.gguf"):
        (directory / name).write_bytes(b"GGUF" + b"\x00" * 20)
    server = repo / "explicit-server/bin/llama-server"
    server.parent.mkdir(parents=True)
    server.write_text(f"#!{sys.executable}\n" + """
import json,sys
if sys.argv[1:] == ['--help']:
    print('--mmproj --alias --chat-template-kwargs --reasoning')
else:
    print(json.dumps({'server_args': sys.argv[1:]}))
""")
    server.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path / "private-home"), "PY": sys.executable}
    for key in ("LLAMA_DIR", "LLAMA_SERVER", "MODEL_ROOT", "PORT", "CTX", "CASCADE_QWEN_MODEL", "CASCADE_QWEN_MMPROJ",
                "CASCADE_INSTALL_PROFILE", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    # The owned build/receipt depend on the GPU's CUDA architecture; pin it so the
    # contract tests do not depend on nvidia-smi being present on the test host.
    env["CASCADE_LLAMA_CUDA_ARCH"] = "121"
    env["CASCADE_QWEN_MMPROJ"] = str(directory / "mmproj-Qwen3.8-27B-BF16.gguf")
    env["LLAMA_SERVER"] = str(server)
    return repo, directory, server, env


def invoke(prepared, *args):
    repo, _, _, env = prepared
    return subprocess.run(["bash", str(repo / "scripts/serve_qwen_llamacpp.sh"), *args],
                          env=env, capture_output=True, text=True, timeout=30)


def test_server_uses_exact_model_projector_alias_and_no_reasoning(prepared):
    result = invoke(prepared)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = next(json.loads(line) for line in result.stdout.splitlines() if '"server_args"' in line)
    args = payload["server_args"]
    directory = prepared[1]
    expected = {"--model": str(directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
                "--mmproj": str(directory / "mmproj-Qwen3.8-27B-BF16.gguf"),
                "--alias": "Qwen/Qwen3.8-27B", "--host": "127.0.0.1", "--port": "8080",
                "--ctx-size": "32768", "--parallel": "1", "--reasoning": "off"}
    for flag, value in expected.items():
        assert args[args.index(flag) + 1] == value
    assert json.loads(args[args.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert "--jinja" in args


@pytest.mark.parametrize("failure", [None, "missing_projector", "missing_model", "invalid_model"])
def test_check_is_read_only_and_requires_valid_configured_gguf_files(prepared, failure):
    repo, directory, _, _ = prepared
    if failure == "missing_projector":
        (directory / "mmproj-Qwen3.8-27B-BF16.gguf").unlink()
    elif failure == "missing_model":
        (directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf").unlink()
    elif failure == "invalid_model":
        (directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf").write_bytes(b"invalid download")
    def snapshot():
        return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in repo.rglob("*") if p.is_file()}
    before = snapshot()
    result = invoke(prepared, "--check")
    assert (result.returncode == 0) is (failure is None), result.stdout + result.stderr
    if failure:
        assert "Missing or invalid explicit local GGUF" in result.stderr
    assert before == snapshot()
    assert "server_args" not in result.stdout


def test_setup_only_checks_local_files_without_starting_server(prepared):
    result = invoke(prepared, "--setup-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "server_args" not in result.stdout


def test_default_bundle_is_checksum_verified_and_setup_receipts_are_reused(prepared):
    repo, directory, _, env = prepared
    env.pop("CASCADE_QWEN_MMPROJ")
    relative = "deploy/brev/profiles/qwen3.8-27b-q4.json"
    manifest = json.loads((REPO / relative).read_text())
    for item in [*manifest["quantization"]["files"], manifest["source_model"]["license"]]:
        name = item.get("filename", "LICENSE")
        content = b"GGUF" + name.encode()
        (directory / name).write_bytes(content)
        item["size_bytes"] = len(content)
        item["sha256"] = hashlib.sha256(content).hexdigest()
    (repo / relative).parent.mkdir(parents=True)
    (repo / relative).write_text(json.dumps(manifest))
    shutil.copy2(REPO / "deploy/brev/fetch_models.py", repo / "deploy/brev/fetch_models.py")
    result = invoke(prepared, "--setup-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(directory.glob("*.verified.json"))) == 3
    def snapshot():
        return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in repo.rglob("*") if p.is_file()}
    before = snapshot()
    result = invoke(prepared, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot() == before
    model = directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    model.write_bytes(b"x" * model.stat().st_size)
    result = invoke(prepared, "--check")
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stdout + result.stderr


def test_explicit_override_failure_is_not_silently_replaced_by_a_download(prepared):
    _, directory, server, _ = prepared
    (directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf").unlink()
    server.unlink()
    result = invoke(prepared, "--setup-only")
    assert result.returncode != 0
    assert "CASCADE_QWEN_MODEL" in result.stderr
    assert "unset both model overrides" in result.stderr
    assert "Building llama.cpp" not in result.stderr


def test_explicit_local_model_and_projector_work_without_managed_model_tree(prepared, tmp_path):
    _, directory, _, env = prepared
    local = tmp_path / "existing models" / "Q4.gguf"
    local.parent.mkdir()
    local.write_bytes(b"GGUF" + b"\x00" * 20)
    projector = tmp_path / "existing models" / "projector.gguf"
    projector.write_bytes(b"GGUF" + b"\x00" * 20)
    shutil.rmtree(directory)
    env["CASCADE_QWEN_MODEL"] = str(local)
    env["CASCADE_QWEN_MMPROJ"] = str(projector)
    result = invoke(prepared)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)["server_args"]
    assert args[args.index("--model") + 1] == str(local)
    assert args[args.index("--mmproj") + 1] == str(projector)
    assert not directory.exists()


def test_explicit_model_requires_projector_for_camera_inspection(prepared):
    env = prepared[3]
    env["CASCADE_INSTALL_PROFILE"] = "spark"
    env["CASCADE_QWEN_MODEL"] = str(prepared[1] / "Qwen3.8-27B-UD-Q4_K_XL.gguf")
    env.pop("CASCADE_QWEN_MMPROJ")
    result = invoke(prepared, "--check")
    assert result.returncode != 0
    assert "Missing or invalid explicit local GGUF" in result.stderr
    assert "CASCADE_QWEN_MMPROJ" in result.stderr


def test_saved_local_paths_survive_desktop_launch_and_explicit_env_wins(prepared, tmp_path):
    repo, directory, server, env = prepared
    local = tmp_path / "local Q4.gguf"
    local.write_bytes(b"GGUF" + b"\x00" * 20)
    (directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf").unlink()
    record = repo / "runs/.install/install.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"repo": str(repo), "model_environment": {
        "CASCADE_QWEN_MODEL": str(local),
        "CASCADE_QWEN_MMPROJ": env.pop("CASCADE_QWEN_MMPROJ"),
        "LLAMA_SERVER": str(server),
    }}))
    result = invoke(prepared)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)["server_args"]
    assert args[args.index("--model") + 1] == str(local)
    assert "--mmproj" in args
    env["CASCADE_QWEN_MODEL"] = str(tmp_path / "explicit missing model.gguf")
    result = invoke(prepared, "--check")
    assert result.returncode != 0
    assert env["CASCADE_QWEN_MODEL"] in result.stderr


def test_explicit_missing_server_does_not_fall_back_to_a_host_binary(prepared):
    _, _, server, env = prepared
    host = Path(env["HOME"]) / "llama.cpp/build/bin/llama-server"
    host.parent.mkdir(parents=True)
    shutil.copy2(server, host)
    env["LLAMA_SERVER"] = str(server.parent / "missing-server")
    result = invoke(prepared, "--check")
    assert result.returncode != 0
    assert "Missing llama-server" in result.stderr


def owned_build_fixture(prepared):
    """Owned `.llama.cpp` layout with stubbed git/nvcc/ninja/cmake; returns (server, env)."""
    repo, _, explicit_server, env = prepared
    server_source = explicit_server.read_text()
    server = repo / ".llama.cpp/build/bin/llama-server"
    server.parent.mkdir(parents=True)
    env.pop("LLAMA_SERVER")
    bins = repo / "build-fixture-bin"
    bins.mkdir()
    commands = {
        "git": """import sys
a=sys.argv[1:]
if a[-2:] == ['rev-parse', 'HEAD']: print('4695f001fece1660d8bb1b3748f50726ddcc100b')
elif 'status' not in a: raise SystemExit('unexpected source mutation: ' + str(a))
""",
        "nvcc": "",
        "ninja": "",
        "cmake": f"""import pathlib,sys
if '--build' in sys.argv:
    path=pathlib.Path({str(server)!r}); path.write_text({server_source!r}); path.chmod(0o755)
else:
    pathlib.Path({str(bins / 'cmake-configure-args')!r}).write_text(' '.join(sys.argv[1:]))
""",
    }
    for name, source in commands.items():
        path = bins / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)
    env["PATH"] = str(bins) + os.pathsep + os.environ.get("PATH", os.defpath)
    return server, server_source, bins, env


def test_setup_resumes_interrupted_build_of_unchanged_pinned_checkout(prepared):
    repo = prepared[0]
    server, server_source, _, env = owned_build_fixture(prepared)
    result = invoke(prepared, "--setup-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert server.is_file()
    assert (repo / ".llama.cpp/build/.cascade-runtime.json").is_file()
    assert "server_args" not in result.stdout
    result = invoke(prepared, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    server.write_text(server_source + "\n# changed after preparation\n")
    result = invoke(prepared, "--check")
    assert result.returncode != 0
    assert "differs from its pinned build receipt" in result.stderr


def test_owned_build_targets_this_gpu_architecture_and_the_receipt_names_it(prepared):
    """GB10 (Spark) is sm_121 and the x86 rig's Blackwell is sm_120: never hardcode one."""
    repo = prepared[0]
    server, _, bins, env = owned_build_fixture(prepared)
    env["CASCADE_LLAMA_CUDA_ARCH"] = "120"
    result = invoke(prepared, "--setup-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "-DCMAKE_CUDA_ARCHITECTURES=120" in (bins / "cmake-configure-args").read_text()
    receipt = json.loads((repo / ".llama.cpp/build/.cascade-runtime.json").read_text())
    assert receipt["cuda_architecture"] == "120"
    assert invoke(prepared, "--check").returncode == 0
    # The same build checked on the other host's architecture is refused explicitly,
    # without being mistaken for a tampered binary.
    env["CASCADE_LLAMA_CUDA_ARCH"] = "121"
    result = invoke(prepared, "--check")
    assert result.returncode != 0
    assert "built for CUDA architecture '120' but this GPU needs '121'" in result.stderr
    env["CASCADE_LLAMA_CUDA_ARCH"] = "sm_120"
    result = invoke(prepared, "--check")
    assert result.returncode != 0 and "CASCADE_LLAMA_CUDA_ARCH must be" in result.stderr


def test_unknown_gpu_architecture_fails_closed_instead_of_assuming_the_spark(prepared):
    _, _, bins, env = owned_build_fixture(prepared)
    env.pop("CASCADE_LLAMA_CUDA_ARCH")
    # nvidia-smi answers nothing (no driver / no GPU): the build must not guess.
    stub = bins / "nvidia-smi"
    stub.write_text(f"#!{sys.executable}\nraise SystemExit(9)\n")
    stub.chmod(0o755)
    result = invoke(prepared, "--setup-only")
    assert result.returncode != 0
    assert "Cannot read the CUDA compute capability" in result.stderr
    assert not (bins / "cmake-configure-args").exists()
