"""scripts/serve_nemotron_bolt_vllm.sh: the default brain, keyed and loopback-only.

Exercised hermetically: a fake `docker` records every call (argv, and whether
VLLM_API_KEY reached its environment) and a fake `curl` answers /health.
Nothing is pulled, no container starts and no GPU is touched.

What the script must keep (each line was found or measured on the Spark,
docs/evidence/brain-nemotron-bolt-2026-10-18/):
- The model card's container: the vendor vLLM nightly pinned by DIGEST, plus
  b12x by wheel hash in a locally built image; the LiLiCorr DFlash drafter
  with 7 speculative tokens; the card's ultra_v3 reasoning parser by sha256.
- A bare served id `nemotron-3.5-bolt` (OpenClaw splits provider/model on
  '/'), 64k context, one image and no video per prompt, thinking off, tool
  calls through vLLM's qwen3_coder parser, GPU_UTIL 0.45.
- Prefix caching that hits: attention blocks a multiple of 64 and a 64-token
  match unit (the drafter's sliding-window group otherwise forces 17,984-token
  hits, so 13-18k agent prompts never hit).
- Loopback by default; the key read from a 0600 file into VLLM_API_KEY and
  passed to docker by NAME, never argv; no network bind without a key; GPU by
  CDI; weights offline from the host's HF cache, mounted read-only.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "serve_nemotron_bolt_vllm.sh"
TARGET = ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a")
DRAFT = ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr", "abd478065678ee83a93821c4b3f512da24e4cecb")
BASE_IMAGE = "vllm/vllm-openai@sha256:a7deff2c216def98c193352aca50fb03a2ee29231355b3033dc3789009f6d23e"
IMAGE = "cascade/nemotron-bolt-vllm:77154ca6-b12x1.5.1"
PARSER = "ultra_v3_reasoning_parser.py"
REAL_HUB = Path.home() / ".cache" / "huggingface" / "hub"


def snapshot(hub: Path, repo_rev) -> Path:
    repo, rev = repo_rev
    return hub / f"models--{repo.replace('/', '--')}" / "snapshots" / rev


@pytest.fixture
def host(tmp_path):
    """A fake docker + curl on PATH and an HF cache with both checkpoints."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "docker.jsonl"
    (bindir / "docker").write_text(f"#!{sys.executable}\n" + f"""
import json, os, sys
a = sys.argv[1:]
with open({str(log)!r}, "a") as f:
    f.write(json.dumps({{"argv": a, "key": os.environ.get("VLLM_API_KEY")}}) + "\\n")
if a[:2] == ["image", "inspect"]:
    sys.exit(0 if os.environ.get("FAKE_NO_IMAGE") != "1" else 1)
if a[:1] == ["inspect"]:
    with open({str(log)!r}) as f:
        started = any(json.loads(line)["argv"][:2] == ["run", "-d"] for line in f)
    print("true" if started or os.environ.get("FAKE_RUNNING") == "1" else "false")
    sys.exit(0)
if a[:1] == ["run"]:
    print("c0ffee")
if a[:1] == ["wait"] and os.environ.get("FAKE_WAIT_BLOCK") == "1":
    import time
    time.sleep(30)
sys.exit(0)
""")
    (bindir / "curl").write_text("#!/bin/sh\nprintf 200\n")
    for name in ("docker", "curl"):
        (bindir / name).chmod(0o755)
    hub = tmp_path / "hub"
    for pin in (TARGET, DRAFT):
        snapshot(hub, pin).mkdir(parents=True)
        (snapshot(hub, pin) / "config.json").write_text("{}")
    real_parser = snapshot(REAL_HUB, TARGET) / PARSER
    if real_parser.is_file():
        shutil.copyfile(real_parser, snapshot(hub, TARGET) / PARSER)
    return {"bin": bindir, "log": log, "hub": hub, "tmp": tmp_path, "real_parser": real_parser.is_file()}


def run(host, *args, **env):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("VLLM_", "API_KEY", "HOST", "PORT", "SPEC", "NSPEC", "GPU_UTIL", "CTX", "MAX_SEQS",
                              "SERVED_NAME", "PREFIX_MATCH", "MEM_FLOOR", "CACHE_DIR", "HF_", "CONTAINER", "EXTRA"))}
    e.update({"HOME": str(host["tmp"] / "home"), "HF_HUB_CACHE": str(host["hub"]),
              "PATH": f"{host['bin']}:/usr/bin:/bin", **env})
    return subprocess.run(["bash", str(SCRIPT), *args], env=e, capture_output=True, text=True, timeout=60)


def calls(host) -> list[dict]:
    if not host["log"].exists():
        return []
    return [json.loads(line) for line in host["log"].read_text().splitlines()]


def docker_run(host) -> dict:
    [call] = [c for c in calls(host) if c["argv"][:1] == ["run"]]
    return call


def dry(host, **env) -> list[str]:
    r = run(host, "--dry-run", **env)
    assert r.returncode == 0, r.stdout + r.stderr
    return shlex.split(r.stdout)


def flag(args, name):
    return args[args.index(name) + 1]


def test_dry_run_is_the_measured_default_brain_on_loopback(host):
    args = dry(host)
    assert args[:3] == ["docker", "run", "-d"]
    assert flag(args, "--device") == "nvidia.com/gpu=all"
    assert flag(args, "-p") == "127.0.0.1:8080:8000"
    assert flag(args, "--entrypoint") == "vllm" and args[args.index("--entrypoint") + 2] == IMAGE
    assert flag(args, "serve") == f"/hf/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/{TARGET[1]}"
    assert flag(args, "--served-model-name") == "nemotron-3.5-bolt"
    assert flag(args, "--max-model-len") == "65536"
    assert flag(args, "--gpu-memory-utilization") == "0.45"
    assert flag(args, "--max-num-seqs") == "4"
    assert json.loads(flag(args, "--limit-mm-per-prompt")) == {"video": 0, "image": 1}
    assert json.loads(flag(args, "--default-chat-template-kwargs")) == {"enable_thinking": False}
    assert flag(args, "--tool-call-parser") == "qwen3_coder" and "--enable-auto-tool-choice" in args
    assert flag(args, "--reasoning-parser") == "ultra_v3"
    assert flag(args, "--reasoning-parser-plugin").endswith(f"/snapshots/{TARGET[1]}/{PARSER}")
    assert flag(args, "--safetensors-load-strategy") == "eager"
    assert (flag(args, "--linear-backend"), flag(args, "--moe-backend")) == ("b12x", "b12x")
    spec = json.loads(flag(args, "--speculative-config"))
    assert (spec["method"], spec["num_speculative_tokens"]) == ("dflash", 7)
    assert spec["model"] == f"/hf/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr/snapshots/{DRAFT[1]}"
    for env in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert env in args
    assert f"{host['hub']}:/hf:ro" in args
    assert flag(args, "--user") == f"{os.getuid()}:{os.getgid()}"
    assert "VLLM_API_KEY" not in args
    assert calls(host) == [], "a dry run runs nothing"


def test_prefix_caching_hits_below_the_hybrid_lcm_block(host):
    """The drafter's sliding-window group (block 64) makes vLLM refuse
    sub-block hits unless every group's block is a multiple of the match
    unit; 64/64 lets an agent call reuse its earlier prompt."""
    args = dry(host)
    assert "--enable-prefix-caching" in args
    assert (flag(args, "--block-size"), flag(args, "--prefix-match-unit")) == ("64", "64")
    off = dry(host, PREFIX_MATCH="0")
    assert "--enable-prefix-caching" in off and "--block-size" not in off and "--prefix-match-unit" not in off


def test_knobs_reach_the_server(host):
    args = dry(host, PORT="18491", GPU_UTIL="0.5", CTX="32768", MAX_SEQS="2", SERVED_NAME="bolt",
               SPEC="none", CONTAINER="bolt-x")
    assert flag(args, "-p") == "127.0.0.1:18491:8000" and flag(args, "--name") == "bolt-x"
    assert (flag(args, "--gpu-memory-utilization"), flag(args, "--max-model-len"), flag(args, "--max-num-seqs"),
            flag(args, "--served-model-name")) == ("0.5", "32768", "2", "bolt")
    assert "--speculative-config" not in args


@pytest.mark.parametrize("env, message", [
    ({"SERVED_NAME": "nvidia/bolt"}, "bare id"),
    ({"SPEC": "eagle"}, "SPEC must be"),
    ({"PORT": "80;rm"}, "PORT must be"),
    ({"GPU_UTIL": "1.5"}, "GPU_UTIL must be"),
    ({"CTX": "64k"}, "CTX must be"),
    ({"CONTAINER": "a b"}, "CONTAINER must be"),
])
def test_bad_knobs_are_refused(host, env, message):
    r = run(host, "--dry-run", **env)
    assert r.returncode == 2 and message in r.stderr
    assert calls(host) == []


def test_unknown_arguments_are_refused(host):
    r = run(host, "--serve")
    assert r.returncode == 2 and "unknown argument" in r.stderr


def test_no_network_bind_without_a_key(host):
    r = run(host, "--dry-run", HOST="0.0.0.0")
    assert r.returncode == 1 and "unauthenticated brain" in r.stderr


@pytest.mark.parametrize("problem", ["missing", "open", "empty"])
def test_a_bad_key_file_is_refused_before_serving(host, problem):
    key = host["tmp"] / "brain.key"
    if problem != "missing":
        key.write_text("" if problem == "empty" else "k-123\n")
        key.chmod(0o644 if problem == "open" else 0o600)
    r = run(host, "--dry-run", API_KEY_FILE=str(key))
    assert r.returncode == 1 and "API_KEY_FILE" in r.stderr
    assert calls(host) == []


def test_the_key_reaches_docker_by_name_never_argv(host):
    if not host["real_parser"]:
        pytest.skip("the pinned ultra_v3 parser is not in this host's HF cache")
    key = host["tmp"] / "brain.key"
    key.write_text("bolt-secret-91f\n")
    key.chmod(0o600)
    r = run(host, API_KEY_FILE=str(key), HOST="0.0.0.0")
    assert r.returncode == 1 and "the vLLM container stopped" in r.stderr   # fake docker: `wait` returns at once
    started = docker_run(host)
    assert started["key"] == "bolt-secret-91f"
    assert started["argv"][started["argv"].index("-e") + 1] == "VLLM_API_KEY"
    assert flag(started["argv"], "-p") == "0.0.0.0:8080:8000"
    for c in calls(host):
        assert "bolt-secret-91f" not in json.dumps(c["argv"])
    assert "bolt-secret-91f" not in r.stdout + r.stderr
    assert "ready: nemotron-3.5-bolt on http://0.0.0.0:8080/v1" in r.stderr


def test_a_signal_stops_the_container_it_started(host):
    """The unit stops (SIGTERM from systemd / process_owner): the container
    goes with it, inside process_owner's 10 s grace."""
    if not host["real_parser"]:
        pytest.skip("the pinned ultra_v3 parser is not in this host's HF cache")
    e = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "API_KEY", "HOST", "PORT", "HF_"))}
    e.update({"HOME": str(host["tmp"] / "home"), "HF_HUB_CACHE": str(host["hub"]),
              "PATH": f"{host['bin']}:/usr/bin:/bin", "FAKE_WAIT_BLOCK": "1"})
    p = subprocess.Popen(["bash", str(SCRIPT)], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not [c for c in calls(host) if c["argv"][:1] == ["wait"]]:
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=20)
    names = [c["argv"][:3] for c in calls(host)]
    assert p.returncode == 143
    assert ["stop", "--time", "8"] in names and names.index(["stop", "--time", "8"]) > names.index(
        ["run", "-d", "--rm"])


def test_a_tampered_parser_is_refused(host):
    (snapshot(host["hub"], TARGET) / PARSER).write_text("print('not the card parser')\n")
    r = run(host)
    assert r.returncode == 1 and "sha256" in r.stderr and "--setup-only" in r.stderr
    assert not [c for c in calls(host) if c["argv"][:1] == ["run"]]


def test_missing_weights_or_image_are_refused_before_starting(host):
    shutil.rmtree(snapshot(host["hub"], DRAFT))
    r = run(host)
    assert r.returncode == 1 and "weights missing" in r.stderr
    r = run(host, FAKE_NO_IMAGE="1")
    assert r.returncode == 1 and "is missing; run" in r.stderr
    assert not [c for c in calls(host) if c["argv"][:1] == ["run"]]


def test_without_the_drafter_its_weights_are_not_required(host):
    if not host["real_parser"]:
        pytest.skip("the pinned ultra_v3 parser is not in this host's HF cache")
    shutil.rmtree(snapshot(host["hub"], DRAFT))
    run(host, SPEC="none")
    assert "--speculative-config" not in docker_run(host)["argv"]


def test_an_already_running_container_is_not_replaced(host):
    if not host["real_parser"]:
        pytest.skip("the pinned ultra_v3 parser is not in this host's HF cache")
    r = run(host, FAKE_RUNNING="1")
    assert r.returncode == 1 and "already running" in r.stderr
    assert not [c for c in calls(host) if c["argv"][:1] in (["run"], ["rm"], ["stop"])]


def test_low_memory_during_startup_kills_the_brain(host):
    """MemAvailable below MEM_FLOOR_GB while loading: kill, exit 3."""
    if not host["real_parser"]:
        pytest.skip("the pinned ultra_v3 parser is not in this host's HF cache")
    (host["bin"] / "curl").write_text("#!/bin/sh\nprintf 000\n")
    r = run(host, MEM_FLOOR_GB="100000")
    assert r.returncode == 3 and "killed the brain" in r.stderr
    assert ["kill"] == [c["argv"][0] for c in calls(host) if c["argv"][0] == "kill"]


def test_setup_only_builds_the_pinned_image_and_fetches_pinned_weights(host):
    if not host["real_parser"]:
        pytest.skip("the pinned ultra_v3 parser is not in this host's HF cache")
    r = run(host, "--setup-only", FAKE_NO_IMAGE="1")
    assert r.returncode == 0, r.stderr
    argvs = [c["argv"] for c in calls(host)]
    assert ["pull", BASE_IMAGE] in argvs
    [build] = [a for a in argvs if a[0] == "build"]
    assert build[build.index("-t") + 1] == IMAGE
    downloads = [a for a in argvs if a[:2] == ["run", "--rm"] and "download" in a]
    assert [(a[a.index("download") + 1], flag(a, "--revision")) for a in downloads] == [TARGET, DRAFT]
    assert not [a for a in argvs if a[:2] == ["run", "-d"]], "setup never serves"


def test_every_bolt_client_asks_for_the_served_name_and_thinking_off():
    """configs/llm/local_nemotron_bolt.yaml (cascade's non-streaming agent
    client) names the served id and sends enable_thinking false itself."""
    import yaml

    profile = yaml.safe_load((REPO / "configs" / "llm" / "local_nemotron_bolt.yaml").read_text())
    assert profile["model"] == "nemotron-3.5-bolt"
    assert profile["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    text = SCRIPT.read_text()
    assert 'SERVED_NAME="${SERVED_NAME:-nemotron-3.5-bolt}"' in text


def _captured_chat(profile: str) -> dict:
    """The request fields cascade's own agent client sends for a profile."""
    pytest.importorskip("openai")
    from types import SimpleNamespace

    from cascade.agent import llm as llm_mod
    from cascade.config import load_profile

    client = llm_mod.make_llm(load_profile("llm", profile))
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content="OK", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client._client.chat.completions.create = create
    client.chat("system", [{"role": "user", "content": "hi"}])
    return seen


def test_cascades_agent_client_turns_bolts_thinking_off():
    sent = _captured_chat("local_nemotron_bolt")
    assert sent["model"] == "nemotron-3.5-bolt"
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_qwens_agent_client_request_is_unchanged():
    assert "extra_body" not in _captured_chat("local_qwen")


def test_a_profile_extra_body_must_be_a_mapping():
    pytest.importorskip("openai")
    from cascade.agent.llm import OpenAICompatClient

    with pytest.raises(ValueError, match="extra_body"):
        OpenAICompatClient(model="m", base_url="http://127.0.0.1:1/v1", extra_body=["enable_thinking"])
