"""scripts/serve_nemotron_bolt_vllm.sh: the default brain on vLLM 0.32.0 from a uv venv, keyed and loopback-only.

Exercised hermetically: a fake venv whose `vllm` prints its argv and the key it would serve with (or plays a slow
server for the startup guards), whose `hf` resolves pinned snapshots in a fake HF cache, and a fake `uv` that records
how the venv would be built. Nothing is downloaded and no GPU is touched.

What the script must keep (each line was found or measured on the Spark,
docs/evidence/brain-nemotron-bolt-2026-10-18/):
- vLLM 0.32.0 with its b12x extra from a uv venv (`--torch-backend=auto`), `vllm download-kernels`; both
  checkpoints by revision; the card's ultra_v3 reasoning parser by sha256.
- A bare served id `nemotron-3.5-bolt`, 64k context, one image and no video per prompt, thinking off, qwen3_coder
  tool calls, GPU_UTIL 0.45.
- Prefix caching that hits: 64-token match unit and block, Mamba `align` mode; no drafter by default (its
  sliding-window group limits hits to whole ~4.5k-token blocks); SPEC=dflash keeps the LiLiCorr drafter.
- Loopback by default; the key read from a 0600 file into VLLM_API_KEY, never argv; no network bind without a key.
- A C++ compiler that accepts the GB10's -march for torch inductor (Ubuntu's g++ 11 does not).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "serve_nemotron_bolt_vllm.sh"
TARGET = ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a")
DRAFT = ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr", "abd478065678ee83a93821c4b3f512da24e4cecb")
PARSER = "ultra_v3_reasoning_parser.py"
REAL_PARSER = (Path.home() / ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots"
               / TARGET[1] / PARSER)


@pytest.fixture
def host(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    log = tmp_path / "calls.jsonl"
    (venv / "bin" / "vllm").write_text(f"#!{sys.executable}\n" + f"""
import json, os, sys, time
a = sys.argv[1:]
with open({str(log)!r}, "a") as f:
    f.write(json.dumps({{"vllm": a, "key": os.environ.get("VLLM_API_KEY"), "cxx": os.environ.get("CXX"),
                        "offline": os.environ.get("HF_HUB_OFFLINE"), "xdg": os.environ.get("XDG_CACHE_HOME"),
                        "hf_home": os.environ.get("HF_HOME")}}) + "\\n")
if a[:1] == ["serve"] and os.environ.get("FAKE_SERVE_SECONDS"):
    import signal

    def stopped(*_):
        with open({str(log)!r}, "a") as f:
            f.write(json.dumps({{"stopped_by": "TERM"}}) + "\\n")
        sys.exit(0)
    signal.signal(signal.SIGTERM, stopped)
    time.sleep(float(os.environ["FAKE_SERVE_SECONDS"]))
""")
    hub = tmp_path / "hub"
    (venv / "bin" / "hf").write_text(f"#!{sys.executable}\n" + f"""
import os, sys
a = sys.argv[1:]
repo, rev = a[1], a[a.index("--revision") + 1]
path = os.path.join({str(hub)!r}, repo.replace("/", "--"), rev)
if os.environ.get("HF_HUB_OFFLINE") == "1" and not os.path.isdir(path):
    sys.exit("not cached")
os.makedirs(path, exist_ok=True)
with open({str(log)!r}, "a") as f:
    f.write('{{"hf": "%s@%s", "offline": "%s"}}\\n' % (repo, rev, os.environ.get("HF_HUB_OFFLINE")))
print(path)
""")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "uv").write_text(f"#!{sys.executable}\n" + f"""
import json, os, sys
a = sys.argv[1:]
with open({str(log)!r}, "a") as f:
    f.write(json.dumps({{"uv": a}}) + "\\n")
""")
    (bindir / "curl").write_text("#!/bin/sh\nprintf \"${FAKE_HEALTH:-200}\"\n")
    (bindir / "nvcc").write_text("#!/bin/sh\nexit 0\n")
    for path in (venv / "bin" / "vllm", venv / "bin" / "hf", bindir / "uv", bindir / "curl", bindir / "nvcc"):
        path.chmod(0o755)
    target = hub / TARGET[0].replace("/", "--") / TARGET[1]
    target.mkdir(parents=True)
    (hub / DRAFT[0].replace("/", "--") / DRAFT[1]).mkdir(parents=True)
    # A stand-in parser, pinned in a copy of the script (CI has no HF cache); the real pin is checked
    # against the real parser in test_the_parser_pin_is_the_cards_parser.
    parser = b"# ultra_v3 stand-in\n"
    (target / PARSER).write_bytes(parser)
    script = tmp_path / "scripts" / SCRIPT.name
    script.parent.mkdir()
    text = SCRIPT.read_text()
    pin = re.search(r'^PARSER_SHA256="([0-9a-f]{64})"$', text, re.M)
    assert pin, "the parser must stay pinned by sha256"
    script.write_text(text.replace(pin.group(1), hashlib.sha256(parser).hexdigest()))
    return {"tmp": tmp_path, "venv": venv, "log": log, "hub": hub, "bin": bindir, "target": target, "script": script}


def run(host, *args, **env):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("VLLM_", "API_KEY", "HOST", "PORT", "SPEC", "NSPEC", "GPU_UTIL", "CTX", "MAX_SEQS",
                              "SERVED_NAME", "PREFIX_MATCH", "MEM_FLOOR", "HF_", "EXTRA", "CXX", "VENV", "XDG_",
                              "CACHE_DIR"))}
    e.update({"HOME": str(host["tmp"] / "home"), "VENV": str(host["venv"]),
              "PATH": f"{host['bin']}:/usr/bin:/bin", **env})
    return subprocess.run(["bash", str(host["script"]), *args], env=e, capture_output=True, text=True, timeout=60)


def calls(host) -> list[dict]:
    return [json.loads(x) for x in host["log"].read_text().splitlines()] if host["log"].exists() else []


def served(host) -> dict:
    [call] = [c for c in calls(host) if c.get("vllm", [""])[:1] == ["serve"]]
    return call


def flag(args, name):
    return args[args.index(name) + 1]


def dry(host, **env) -> list[str]:
    r = run(host, "--dry-run", **env)
    assert r.returncode == 0, r.stdout + r.stderr
    return shlex.split(r.stdout)


def test_dry_run_is_the_measured_default_brain_on_loopback(host):
    args = dry(host)
    assert args[:2] == [str(host["venv"] / "bin" / "vllm"), "serve"]
    assert flag(args, "--served-model-name") == "nemotron-3.5-bolt"
    assert (flag(args, "--host"), flag(args, "--port")) == ("127.0.0.1", "8080")
    assert (flag(args, "--max-model-len"), flag(args, "--gpu-memory-utilization"), flag(args, "--max-num-seqs")) == (
        "65536", "0.45", "4")
    assert json.loads(flag(args, "--limit-mm-per-prompt")) == {"video": 0, "image": 1}
    assert json.loads(flag(args, "--default-chat-template-kwargs")) == {"enable_thinking": False}
    assert flag(args, "--tool-call-parser") == "qwen3_coder" and "--enable-auto-tool-choice" in args
    assert flag(args, "--reasoning-parser") == "ultra_v3" and flag(args, "--reasoning-parser-plugin").endswith(PARSER)
    assert flag(args, "--safetensors-load-strategy") == "eager"
    assert (flag(args, "--linear-backend"), flag(args, "--moe-backend")) == ("b12x", "b12x")
    assert "--speculative-config" not in args, "the drafter costs prefix-cache hits; it is opt-in"
    assert calls(host) == [], "a dry run runs nothing"


def test_prefix_caching_hits_at_64_tokens_in_mamba_align_mode(host):
    args = dry(host)
    assert "--enable-prefix-caching" in args and flag(args, "--mamba-cache-mode") == "align"
    assert (flag(args, "--block-size"), flag(args, "--prefix-match-unit")) == ("64", "64")
    off = dry(host, PREFIX_MATCH="0")
    assert "--block-size" not in off and "--prefix-match-unit" not in off


def test_spec_dflash_keeps_the_lilicorr_drafter(host):
    spec = json.loads(flag(dry(host, SPEC="dflash"), "--speculative-config"))
    assert (spec["method"], spec["num_speculative_tokens"]) == ("dflash", 7)
    assert DRAFT[0] in spec["model"] and DRAFT[1] in spec["model"]


def test_knobs_reach_the_server(host):
    args = dry(host, PORT="18491", GPU_UTIL="0.5", CTX="32768", MAX_SEQS="2", SERVED_NAME="bolt")
    assert (flag(args, "--port"), flag(args, "--gpu-memory-utilization"), flag(args, "--max-model-len"),
            flag(args, "--max-num-seqs"), flag(args, "--served-model-name")) == ("18491", "0.5", "32768", "2", "bolt")


def test_unknown_arguments_are_refused(host):
    r = run(host, "--serve")
    assert r.returncode == 2 and "unknown argument" in r.stderr


def test_the_parser_pin_is_the_cards_parser():
    if not REAL_PARSER.is_file():
        pytest.skip("the card's ultra_v3 parser is not in this host's HF cache")
    pin = re.search(r'^PARSER_SHA256="([0-9a-f]{64})"$', SCRIPT.read_text(), re.M).group(1)
    assert hashlib.sha256(REAL_PARSER.read_bytes()).hexdigest() == pin


def test_without_the_drafter_its_weights_are_not_required(host):
    import shutil

    shutil.rmtree(host["hub"] / DRAFT[0].replace("/", "--"))
    r = run(host)
    assert r.returncode == 0, r.stderr
    assert "--speculative-config" not in served(host)["vllm"]
    r = run(host, SPEC="dflash")
    assert r.returncode == 1 and "weights missing" in r.stderr and DRAFT[0] in r.stderr


@pytest.mark.parametrize("env, message", [
    ({"SERVED_NAME": "nvidia/bolt"}, "bare id"), ({"SPEC": "eagle"}, "SPEC must be"), ({"PORT": "80;rm"}, "PORT must be"),
    ({"GPU_UTIL": "1.5"}, "GPU_UTIL must be"), ({"CTX": "64k"}, "CTX must be"), ({"PREFIX_MATCH": "-1"}, "PREFIX_MATCH"),
])
def test_bad_knobs_are_refused(host, env, message):
    r = run(host, "--dry-run", **env)
    assert r.returncode == 2 and message in r.stderr


def test_no_network_bind_without_a_key(host):
    r = run(host, "--dry-run", HOST="0.0.0.0")
    assert r.returncode == 1 and "unauthenticated brain" in r.stderr


@pytest.mark.parametrize("problem", ["missing", "open", "empty"])
def test_a_bad_key_file_is_refused_before_serving(host, problem):
    key = host["tmp"] / "brain.key"
    if problem != "missing":
        key.write_text("" if problem == "empty" else "k-123\n")
        key.chmod(0o644 if problem == "open" else 0o600)
    r = run(host, API_KEY_FILE=str(key))
    assert r.returncode == 1 and "API_KEY_FILE" in r.stderr
    assert calls(host) == []


def test_the_key_reaches_vllm_by_environment_never_argv(host):
    key = host["tmp"] / "brain.key"
    key.write_text("bolt-secret-91f\n")
    key.chmod(0o600)
    r = run(host, API_KEY_FILE=str(key), HOST="0.0.0.0")
    assert r.returncode == 0, r.stderr
    call = served(host)
    assert call["key"] == "bolt-secret-91f" and call["offline"] == "1"
    assert flag(call["vllm"], "--host") == "0.0.0.0"
    assert "bolt-secret-91f" not in json.dumps(call["vllm"]) + r.stdout + r.stderr
    assert "ready: nemotron-3.5-bolt on http://0.0.0.0:8080/v1" in r.stderr


def test_serving_reads_the_pinned_snapshots_offline(host):
    run(host)
    call = served(host)
    assert call["vllm"][1] == str(host["target"])
    assert [c["hf"] for c in calls(host) if "hf" in c] == [f"{TARGET[0]}@{TARGET[1]}"]
    assert {c["offline"] for c in calls(host) if "hf" in c} == {"1"}


def test_a_tampered_parser_is_refused(host):
    (host["target"] / PARSER).write_text("print('not the card parser')\n")
    r = run(host)
    assert r.returncode == 1 and "sha256" in r.stderr and "--setup-only" in r.stderr
    assert not [c for c in calls(host) if "vllm" in c]


def test_missing_weights_are_refused_and_never_downloaded_when_serving(host):
    import shutil

    shutil.rmtree(host["hub"])
    r = run(host)
    assert r.returncode == 1 and "weights missing" in r.stderr
    assert not [c for c in calls(host) if "vllm" in c]


def test_without_a_venv_serving_says_to_run_setup(host):
    import shutil

    shutil.rmtree(host["venv"])
    r = run(host)
    assert r.returncode == 1 and "--setup-only" in r.stderr and not calls(host)


def test_setup_builds_the_pinned_uv_venv_and_fetches_both_checkpoints(host):
    import shutil

    vllm, hf = (host["venv"] / "bin" / "vllm").read_text(), (host["venv"] / "bin" / "hf").read_text()
    shutil.rmtree(host["venv"])
    # The fake uv creates the venv's executables the way `uv venv` + `uv pip install` would.
    (host["bin"] / "uv").write_text(f"#!{sys.executable}\n" + f"""
import json, os, sys
a = sys.argv[1:]
with open({str(host["log"])!r}, "a") as f:
    f.write(json.dumps({{"uv": a}}) + "\\n")
if a[:1] == ["venv"]:
    os.makedirs(os.path.join(a[1], "bin"), exist_ok=True)
    for name, body in (("vllm", {vllm!r}), ("hf", {hf!r})):
        p = os.path.join(a[1], "bin", name)
        open(p, "w").write(body)
        os.chmod(p, 0o755)
""")
    r = run(host, "--setup-only")
    assert r.returncode == 0, r.stderr
    uv = [c["uv"] for c in calls(host) if "uv" in c]
    assert uv[0][:3] == ["venv", str(host["venv"]), "--python"]
    install = uv[1]
    assert "--torch-backend=auto" in install and "vllm[b12x]==0.32.0" in install
    assert ["download-kernels"] in [c["vllm"] for c in calls(host) if "vllm" in c]
    assert [c["hf"] for c in calls(host) if "hf" in c] == [f"{TARGET[0]}@{TARGET[1]}", f"{DRAFT[0]}@{DRAFT[1]}"]
    assert not [c for c in calls(host) if c.get("vllm", [""])[:1] == ["serve"]], "setup never serves"


def test_inductor_gets_a_compiler_that_accepts_the_gb10_march(host):
    run(host, CXX="my-c++")
    assert served(host)["cxx"] == "my-c++", "an operator's CXX wins"


def test_kernel_caches_live_in_a_directory_the_brain_owns(host):
    run(host, CACHE_DIR=str(host["tmp"] / "kcache"))
    call = served(host)
    assert call["xdg"] == str(host["tmp"] / "kcache") and (host["tmp"] / "kcache").is_dir()
    assert call["hf_home"] == str(host["tmp"] / "home" / ".cache" / "huggingface"), "the weights cache does not move"


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state not in ("Z", "X")


def test_low_memory_during_startup_kills_the_brain(host):
    if not Path("/proc/self/stat").exists():
        pytest.skip("needs /proc to see the brain process")
    started = time.monotonic()
    r = run(host, MEM_FLOOR_GB="100000", FAKE_HEALTH="000", FAKE_SERVE_SECONDS="30")
    assert r.returncode == 3 and "killed the brain" in r.stderr
    # A vLLM left running would hold the output pipes open for its 30 s.
    assert time.monotonic() - started < 15, "the guard must kill vLLM, not just exit"
    pid = int(re.search(r"vLLM 0\.32\.0 pid (\d+)", r.stderr).group(1))
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid), "the guard must kill vLLM, not just exit"


def test_a_signal_stops_vllm_with_the_unit(host):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "API_KEY", "HOST", "PORT", "HF_", "VENV"))}
    e.update({"HOME": str(host["tmp"] / "home"), "VENV": str(host["venv"]), "PATH": f"{host['bin']}:/usr/bin:/bin",
              "FAKE_SERVE_SECONDS": "60"})
    p = subprocess.Popen(["bash", str(host["script"])], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not [c for c in calls(host) if c.get("vllm", [""])[:1] == ["serve"]]:
        time.sleep(0.05)
    time.sleep(0.5)
    started = time.monotonic()
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=20)
    assert p.returncode == 143 and time.monotonic() - started < 5
    assert {"stopped_by": "TERM"} in calls(host), "the stop reaches vLLM itself"


def test_vllm_is_pinned_in_step_with_the_qwen_brain():
    pins = {name: next(line.split('"')[1] for line in (REPO / "scripts" / name).read_text().splitlines()
                       if line.startswith("VLLM_VERSION="))
            for name in ("serve_nemotron_bolt_vllm.sh", "serve_qwen_vllm.sh")}
    assert set(pins.values()) == {"0.32.0"}


def test_every_bolt_client_asks_for_the_served_name_and_thinking_off():
    """configs/llm/local_nemotron_bolt.yaml (cascade's non-streaming agent
    client) names the served id and sends enable_thinking false itself."""
    import yaml

    profile = yaml.safe_load((REPO / "configs" / "llm" / "local_nemotron_bolt.yaml").read_text())
    assert profile["model"] == "nemotron-3.5-bolt"
    assert profile["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert 'SERVED_NAME="${SERVED_NAME:-nemotron-3.5-bolt}"' in SCRIPT.read_text()


def _stub_openai(monkeypatch, sent: dict) -> None:
    """A stub `openai` module (CI installs no extras, AGENTS.md): the client's
    request fields land in `sent`."""
    from types import ModuleType, SimpleNamespace

    def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content="OK", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", module)


def _captured_chat(monkeypatch, profile: str) -> dict:
    """The request fields cascade's own agent client sends for a profile."""
    from cascade.agent import llm as llm_mod
    from cascade.config import load_profile

    sent: dict = {}
    _stub_openai(monkeypatch, sent)
    llm_mod.make_llm(load_profile("llm", profile)).chat("system", [{"role": "user", "content": "hi"}])
    return sent


def test_cascades_agent_client_turns_bolts_thinking_off(monkeypatch):
    sent = _captured_chat(monkeypatch, "local_nemotron_bolt")
    assert sent["model"] == "nemotron-3.5-bolt"
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_qwens_agent_client_request_is_unchanged(monkeypatch):
    sent = _captured_chat(monkeypatch, "local_qwen")
    assert sent["model"] == "Qwen/Qwen3.8-27B" and "extra_body" not in sent


def test_a_profile_extra_body_must_be_a_mapping(monkeypatch):
    from cascade.agent.llm import OpenAICompatClient

    _stub_openai(monkeypatch, {})
    with pytest.raises(ValueError, match="extra_body"):
        OpenAICompatClient(model="m", base_url="http://127.0.0.1:1/v1", extra_body=["enable_thinking"])
