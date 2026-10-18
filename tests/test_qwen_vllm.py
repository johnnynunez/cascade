"""scripts/serve_qwen_vllm.sh: the measured Spark brain, keyed and loopback-only.

Exercised hermetically: a fake `vllm` prints the argv and the key it would
serve with, and a fake `hf` prints a snapshot path and records the pinned
revision it was asked for. Nothing is downloaded and no GPU is touched.

What the script must keep (each line was found or measured on the Spark):
- Qwen3.8-27B NVFP4 (Inferact, BF16 lm_head/embeddings) plus the DFlash2
  drafter, both at pinned revisions: 18.8-29.8 tok/s single stream against
  10.6 for llama.cpp Q4_K_XL.
- `--safetensors-load-strategy eager`: mmap reads on GB10 run at ~5 MB/s,
  27 min of loading against 50-90 s.
- A bare served model id (OpenClaw splits provider/model on '/'), tool calls
  through vLLM's qwen3_coder parser (llama.cpp's grammar conversion
  rejected an unanchored tool `pattern` and the chat died), thinking off.
- Loopback by default. A bearer key read from a 0600 file into VLLM_API_KEY,
  never argv. No network bind without a key.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "serve_qwen_vllm.sh"
TARGET = ("Inferact/Qwen3.8-27B-NVFP4", "6128240ebaf4eaa7bad2b3d1c72c37d677c5f462")
DRAFT = ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4")


@pytest.fixture
def venv(tmp_path):
    v = tmp_path / "venv"
    (v / "bin").mkdir(parents=True)
    (v / "bin" / "vllm").write_text(f"#!{sys.executable}\n" + """
import json, os, shutil, sys
print(json.dumps({"vllm_args": sys.argv[1:], "key": os.environ.get("VLLM_API_KEY"),
                  "offline": os.environ.get("HF_HUB_OFFLINE"), "nvcc": shutil.which("nvcc")}))
""")
    (v / "bin" / "hf").write_text(f"#!{sys.executable}\n" + f"""
import json, os, sys
a = sys.argv[1:]
offline = os.environ.get("HF_HUB_OFFLINE") == "1"
with open({str(tmp_path / "hf.log")!r}, "a") as f:
    f.write(json.dumps(a + (["<offline>"] if offline else [])) + "\\n")
if offline and os.environ.get("FAKE_HF_CACHED") != "1":
    sys.exit("not in the local cache")
print("/hub/" + a[1].replace("/", "--") + "/" + a[a.index("--revision") + 1])
""")
    for b in ("vllm", "hf"):
        (v / "bin" / b).chmod(0o755)
    cuda = tmp_path / "cuda"           # a toolkit with nvcc, outside PATH
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin" / "nvcc").write_text("#!/bin/sh\nexit 0\n")
    (cuda / "bin" / "nvcc").chmod(0o755)
    return v


def run(tmp_path, venv, *args, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "API_KEY", "HOST", "PORT", "SPEC", "KV_DTYPE"))}
    e.update({"VENV": str(venv), "HOME": str(tmp_path / "home"), "CUDA_HOME": str(tmp_path / "cuda"),
              "PATH": "/usr/bin:/bin", **env})
    return subprocess.run(["bash", str(SCRIPT), *args], env=e, capture_output=True, text=True, timeout=60)


def served(r) -> dict:
    assert r.returncode == 0, r.stdout + r.stderr
    return next(json.loads(line) for line in r.stdout.splitlines() if '"vllm_args"' in line)


def flag(args, name):
    return args[args.index(name) + 1]


def test_default_is_the_measured_spark_brain_on_loopback(tmp_path, venv):
    out = served(run(tmp_path, venv))
    a = out["vllm_args"]
    assert a[0] == "serve" and a[1] == f"/hub/{TARGET[0].replace('/', '--')}/{TARGET[1]}"
    assert flag(a, "--host") == "127.0.0.1"
    names = a[a.index("--served-model-name") + 1:a.index("--host")]
    assert names == ["qwen3.8-27b", "Qwen/Qwen3.8-27B"], "bare id first (OpenClaw), then the repo-wide id"
    assert flag(a, "--safetensors-load-strategy") == "eager"
    assert flag(a, "--tool-call-parser") == "qwen3_coder" and "--enable-auto-tool-choice" in a
    assert json.loads(flag(a, "--default-chat-template-kwargs")) == {"enable_thinking": False}
    spec = json.loads(flag(a, "--speculative-config"))
    assert spec["method"] == "dflash"
    assert spec["model"] == f"/hub/{DRAFT[0].replace('/', '--')}/{DRAFT[1]}"
    assert out["key"] is None and out["offline"] == "1"
    asked = [json.loads(line) for line in (tmp_path / "hf.log").read_text().splitlines()]
    assert {(q[1], q[q.index("--revision") + 1]) for q in asked} == {TARGET, DRAFT}


def test_a_cached_model_starts_without_the_network(tmp_path, venv):
    """Booths have no reliable internet (BOOTH_RUNBOOK.md): the pinned
    snapshots are resolved from the local cache first, and the hub is only
    asked on a cache miss."""
    served(run(tmp_path, venv, FAKE_HF_CACHED="1"))
    asked = [json.loads(line) for line in (tmp_path / "hf.log").read_text().splitlines()]
    assert len(asked) == 2 and all(q[-1] == "<offline>" for q in asked)


def test_the_key_comes_from_a_private_file_never_argv(tmp_path, venv):
    key = tmp_path / "brain.key"
    key.write_text("s3cret-brain-key\n")
    key.chmod(0o600)
    out = served(run(tmp_path, venv, API_KEY_FILE=str(key), PORT="8081"))
    assert out["key"] == "s3cret-brain-key"
    assert not any("s3cret" in x for x in out["vllm_args"])
    assert "--api-key" not in out["vllm_args"]
    assert flag(out["vllm_args"], "--port") == "8081"


@pytest.mark.parametrize("problem", ["loose", "empty", "missing"])
def test_a_bad_key_file_is_refused_before_serving(tmp_path, venv, problem):
    key = tmp_path / "brain.key"
    if problem != "missing":
        key.write_text("" if problem == "empty" else "k\n")
        key.chmod(0o644 if problem == "loose" else 0o600)
    r = run(tmp_path, venv, API_KEY_FILE=str(key))
    assert r.returncode != 0
    assert "vllm_args" not in r.stdout


def test_no_network_bind_without_a_key(tmp_path, venv):
    r = run(tmp_path, venv, HOST="0.0.0.0")
    assert r.returncode != 0 and "API_KEY_FILE" in r.stderr
    assert "vllm_args" not in r.stdout
    key = tmp_path / "brain.key"
    key.write_text("k\n")
    key.chmod(0o600)
    assert flag(served(run(tmp_path, venv, HOST="0.0.0.0", API_KEY_FILE=str(key)))["vllm_args"],
                "--host") == "0.0.0.0"


def test_speculation_modes(tmp_path, venv):
    mtp = json.loads(flag(served(run(tmp_path, venv, SPEC="mtp"))["vllm_args"], "--speculative-config"))
    assert mtp["method"] == "mtp"
    assert "--speculative-config" not in served(run(tmp_path, venv, SPEC="none"))["vllm_args"]
    r = run(tmp_path, venv, SPEC="medusa")
    assert r.returncode == 2 and "vllm_args" not in r.stdout


def test_setup_only_fetches_the_pinned_models_without_serving(tmp_path, venv):
    r = run(tmp_path, venv, "--setup-only")
    assert r.returncode == 0, r.stderr
    assert "vllm_args" not in r.stdout
    online = [line for line in (tmp_path / "hf.log").read_text().splitlines() if "<offline>" not in line]
    assert len(online) == 2, "a cache miss fetches each pinned model once"


def test_nvcc_reaches_vllm_even_from_a_minimal_environment(tmp_path, venv):
    """vLLM registers its FlashInfer FP4 GEMM op only when FlashInfer is
    usable, and it deems FlashInfer unusable without nvcc on PATH (no
    flashinfer-cubin). Started from a minimal environment (a systemd unit
    with PATH=/usr/bin:/bin), the NVFP4 engine then died after 3 minutes of
    loading: "'vllm' object has no attribute 'flashinfer_mm_fp4'" (Spark,
    2026-10-11). The script puts $CUDA_HOME/bin on PATH."""
    out = served(run(tmp_path, venv))
    assert out["nvcc"] == str(tmp_path / "cuda" / "bin" / "nvcc")


def test_no_nvcc_anywhere_fails_fast_not_after_loading(tmp_path, venv):
    r = run(tmp_path, venv, CUDA_HOME=str(tmp_path / "nowhere"))
    assert r.returncode != 0 and "nvcc" in r.stderr
    assert "vllm_args" not in r.stdout


def test_every_local_profile_asks_for_a_name_the_server_answers_to(tmp_path, venv):
    """vLLM rejects a request whose `model` it does not serve (llama.cpp
    ignores it, which is how configs/llm/local_qwen.yaml stayed on
    Qwen3.6 behind a 3.8 server). This replaces its KEEP IN SYNC comment."""
    import yaml

    a = served(run(tmp_path, venv))["vllm_args"]
    names = set(a[a.index("--served-model-name") + 1:a.index("--host")])
    profile = yaml.safe_load((REPO / "configs" / "llm" / "local_qwen.yaml").read_text())
    assert profile["model"] in names


def test_a_missing_venv_is_built_with_uv_on_the_pinned_vllm(tmp_path, venv):
    """vLLM 0.32.0 ships CUDA 13 arm64 wheels: a uv venv per version (shared
    with the Bolt brain, hence the b12x extra), uv picking the matching torch,
    then the precompiled FlashInfer kernels instead of a JIT next to the model."""
    import shutil

    fresh = tmp_path / "fresh-venv"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "uv.log"
    (bindir / "uv").write_text(f"#!{sys.executable}\n" + f"""
import json, os, shutil, sys
a = sys.argv[1:]
with open({str(log)!r}, "a") as f:
    f.write(json.dumps(a) + "\\n")
if a[:1] == ["venv"]:
    os.makedirs(os.path.join(a[1], "bin"), exist_ok=True)
    for name in ("vllm", "hf"):
        shutil.copy2(os.path.join({str(venv)!r}, "bin", name), os.path.join(a[1], "bin", name))
""")
    (bindir / "uv").chmod(0o755)
    r = run(tmp_path, fresh, "--setup-only", PATH=f"{bindir}:/usr/bin:/bin")
    assert r.returncode == 0, r.stdout + r.stderr
    uv = [json.loads(line) for line in log.read_text().splitlines()]
    assert uv[0][:2] == ["venv", str(fresh)]
    assert "--torch-backend=cu130" in uv[1] and "vllm[b12x]==0.32.0" in uv[1], "torch cu130, never auto (cu132 faulted)"
    assert any(json.loads(line).get("vllm_args") == ["download-kernels"]
               for line in r.stdout.splitlines() if '"vllm_args"' in line)
    assert '"serve"' not in r.stdout, "setup never serves"
    shutil.rmtree(fresh)


def test_the_default_venv_is_per_vllm_version():
    text = SCRIPT.read_text()
    assert 'VLLM_VERSION="0.32.0"' in text and 'TORCH_BACKEND="cu130"' in text
    assert 'VENV="${VENV:-$HOME/.venvs/vllm-$VLLM_VERSION-$TORCH_BACKEND}"' in text


def test_the_kv_cache_is_nvfp4_on_flashinfer_by_default(tmp_path, venv):
    """vLLM 0.32.0 writes and reads an NVFP4 KV cache on SM12x (FlashInfer: fa2
    prefill, XQA decode); the FLASHINFER attention backend is the one that has it."""
    a = served(run(tmp_path, venv))["vllm_args"]
    assert flag(a, "--kv-cache-dtype") == "nvfp4"
    assert flag(a, "--attention-backend") == "FLASHINFER"


@pytest.mark.parametrize("kv, expected", [("fp8", ["--kv-cache-dtype", "fp8"]), ("auto", [])])
def test_kv_dtype_knob(tmp_path, venv, kv, expected):
    a = served(run(tmp_path, venv, KV_DTYPE=kv))["vllm_args"]
    got = [x for i, x in enumerate(a) if x == "--kv-cache-dtype" or (i and a[i - 1] == "--kv-cache-dtype")]
    assert got == expected
    assert "--attention-backend" not in a, "only NVFP4 pins the backend"


def test_an_unknown_kv_dtype_is_refused_before_serving(tmp_path, venv):
    r = run(tmp_path, venv, KV_DTYPE="int4")
    assert r.returncode == 2 and "KV_DTYPE must be nvfp4|fp8|auto" in r.stderr
    assert '"vllm_args"' not in r.stdout

