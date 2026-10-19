"""scripts/serve_nemotron_bolt_llamacpp.sh: the Bolt GGUF brain on llama.cpp v0.7.0, hermetically.

A fake llama-server prints its argv; the pinned files are small stand-ins described by a manifest of the real
shape, so fetch_models.py verifies them exactly as it verifies the 24.8 GB originals. Nothing is downloaded or built.
The warm-start helper is a stub that records its argv (tests/test_brain_warm_start.py tests the real one), so no test
here ever reaches a port.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/serve_nemotron_bolt_llamacpp.sh"
MANIFEST = "deploy/brev/profiles/nemotron-3.5-bolt-24gb.json"
HELP = ("--mmproj --alias --chat-template-kwargs --chat-template-file --reasoning --spec-draft-model --api-key-file "
        "--spec-type none,draft-simple,draft-dflash --load-mode MODE --slot-save-path PATH")
TEMPLATE = "scripts/chat_templates/nemotron-3.5-bolt-openclaw.jinja"
WARM = "scripts/brain_warm_start.py"


@pytest.fixture
def prepared(tmp_path):
    repo = tmp_path / "repo"
    for rel in (SCRIPT, "deploy/brev/fetch_models.py", TEMPLATE):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    real = json.loads((REPO / MANIFEST).read_text())
    directory = repo / "models/nemotron-3.5-bolt"
    directory.mkdir(parents=True)
    for item in real["quantization"]["files"]:     # same names, URLs and roles; small bodies
        body = b"GGUF" + item["filename"].encode()
        (directory / item["filename"]).write_bytes(body)
        item.update(size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    licence = b"OpenMDW License Agreement, version 1.1\n"
    (directory / "LICENSE").write_bytes(licence)
    real["source_model"]["license"].update(size_bytes=len(licence), sha256=hashlib.sha256(licence).hexdigest())
    (repo / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (repo / MANIFEST).write_text(json.dumps(real))
    server = repo / "explicit-server/bin/llama-server"
    server.parent.mkdir(parents=True)
    server.write_text(f"#!{sys.executable}\n" + f"""
import json, os, sys
if sys.argv[1:] == ['--help']:
    print({HELP!r})
else:
    print(json.dumps({{'server_args': sys.argv[1:], 'pid': os.getpid(),
                      'slots_debug': os.environ.get('LLAMA_SERVER_SLOTS_DEBUG')}}))
""")
    server.chmod(0o755)
    (repo / WARM).write_text("import json, os, sys\n"
                             "with open(os.path.join(os.environ['HOME'], 'warm.jsonl'), 'a') as f:\n"
                             "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    (tmp_path / "home").mkdir()
    env = {**os.environ, "HOME": str(tmp_path / "home"), "PY": sys.executable, "LLAMA_SERVER": str(server),
           "CASCADE_LLAMA_CUDA_ARCH": "121"}
    for key in ("LLAMA_DIR", "MODEL_ROOT", "PORT", "CTX", "PARALLEL", "SPEC", "NSPEC", "ALIAS", "API_KEY_FILE",
                "CASCADE_MODEL_MIRROR_URL", "CHAT_TEMPLATE_FILE", "WARM_START", "WARM_PREFIX_FILE", "XDG_STATE_HOME",
                "LLAMA_SERVER_SLOTS_DEBUG"):
        env.pop(key, None)
    return {"repo": repo, "dir": directory, "server": server, "env": env, "home": tmp_path / "home"}


def invoke(p, *args, **env):
    return subprocess.run(["bash", str(p["repo"] / SCRIPT), *args], env={**p["env"], **env},
                          capture_output=True, text=True, timeout=60)


def served(result) -> list[str]:
    assert result.returncode == 0, result.stdout + result.stderr
    return next(json.loads(line) for line in result.stdout.splitlines() if '"server_args"' in line)["server_args"]


def flag(args, name):
    return args[args.index(name) + 1]


def test_the_cards_recipe_on_loopback_with_two_slots_and_thinking_off(prepared):
    args = served(invoke(prepared))
    d = prepared["dir"]
    assert flag(args, "--model") == str(d / "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")
    assert flag(args, "--mmproj") == str(d / "mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")
    assert flag(args, "--spec-type") == "draft-dflash"
    assert flag(args, "--spec-draft-model") == str(d / "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf")
    assert (flag(args, "--spec-draft-n-max"), flag(args, "--spec-draft-ngl")) == ("7", "999")
    assert (flag(args, "--alias"), flag(args, "--host"), flag(args, "--port")) == ("nemotron-3.5-bolt", "127.0.0.1", "8080")
    assert (flag(args, "--parallel"), flag(args, "--ctx-size")) == ("2", str(2 * 65536))
    assert (flag(args, "--cache-type-k"), flag(args, "--cache-type-v")) == ("q8_0", "q8_0")
    assert flag(args, "--reasoning") == "off"
    assert json.loads(flag(args, "--chat-template-kwargs")) == {"enable_thinking": False}
    assert "--jinja" in args and flag(args, "--flash-attn") == "on"
    assert flag(args, "--load-mode") == "none"   # read once into GPU memory (no mmap): no page-cache copy
    assert "--api-key-file" not in args


def test_the_openclaw_cache_boundary_template_is_served_by_default(prepared):
    # The per-session part of OpenClaw's system text goes after the tool block, so a new chat reuses
    # the system text + tools prefix (scripts/chat_templates/, tests/test_bolt_chat_template.py).
    args = served(invoke(prepared))
    assert flag(args, "--chat-template-file") == str(prepared["repo"] / TEMPLATE)
    assert "--jinja" in args


def test_an_empty_chat_template_file_serves_the_ggufs_own_template(prepared):
    args = served(invoke(prepared, CHAT_TEMPLATE_FILE=""))
    assert "--chat-template-file" not in args and "--jinja" in args


def test_a_chosen_chat_template_file_reaches_the_server(prepared, tmp_path):
    other = tmp_path / "other.jinja"
    other.write_text("{{ messages }}")
    assert flag(served(invoke(prepared, CHAT_TEMPLATE_FILE=str(other))), "--chat-template-file") == str(other)


def test_a_missing_chat_template_file_is_refused(prepared):
    (prepared["repo"] / TEMPLATE).unlink()
    for args in ((), ("--check",)):
        result = invoke(prepared, *args)
        assert result.returncode == 1 and "CHAT_TEMPLATE_FILE" in result.stderr, result.stderr
        assert '"server_args"' not in result.stdout


def fake_help(prepared, help_text):
    prepared["server"].write_text(f"#!{sys.executable}\n" + f"""
import json, sys
if sys.argv[1:] == ['--help']:
    print({help_text!r})
else:
    print(json.dumps({{'server_args': sys.argv[1:]}}))
""")


def test_a_server_without_chat_template_file_is_refused(prepared):
    fake_help(prepared, HELP.replace(" --chat-template-file", ""))
    result = invoke(prepared)
    assert result.returncode == 1 and "llama-server lacks --chat-template-file" in result.stderr
    assert served(invoke(prepared, CHAT_TEMPLATE_FILE=""))  # the GGUF's own template needs no such flag


def test_a_server_without_slot_save_path_is_refused_unless_the_warm_start_is_off(prepared):
    fake_help(prepared, HELP.replace(" --slot-save-path PATH", ""))
    result = invoke(prepared)
    assert result.returncode == 1 and "llama-server lacks --slot-save-path" in result.stderr
    assert served(invoke(prepared, WARM_START="0"))


def helper_runs(p) -> list[list[str]]:
    """argv of every warm-start helper the script started (the stub appends one line each)."""
    log = p["home"] / "warm.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_the_warm_start_helper_watches_the_server_the_script_execs(prepared):
    # Started just before `exec llama-server`, so the PID it watches IS the server: it waits for the server's own
    # readiness, prefills the recorded OpenClaw prefix into both slots and ends with the server.
    result = invoke(prepared)
    args = served(result)
    started = next(json.loads(line) for line in result.stdout.splitlines() if '"server_args"' in line)
    pid = started["pid"]
    assert started["slots_debug"] == "1"     # GET /slots shows the last prompts: what the helper records
    (warm,) = helper_runs(prepared)
    assert (flag(warm, "--port"), flag(warm, "--slots"), flag(warm, "--server-pid")) == ("8080", "2", str(pid))
    state = prepared["home"] / ".local/state/cascade"
    assert flag(warm, "--prefix-file") == str(state / "bolt-openclaw-prefix-8080.json")
    assert flag(warm, "--chat-template-file") == str(prepared["repo"] / TEMPLATE)
    assert flag(warm, "--model-file") == flag(args, "--model")
    assert "--api-key-file" not in warm
    # The helper reads the exact tokens of a changed prefix from a slot file it saves into a private directory.
    assert flag(warm, "--slot-save-dir") == flag(args, "--slot-save-path") == str(state / "bolt-slots-8080")
    assert stat.S_IMODE((state / "bolt-slots-8080").stat().st_mode) == 0o700
    # Otherwise the server starts exactly as without the warm start: its readiness is its own.
    prepared["home"].joinpath("warm.jsonl").unlink()
    off = invoke(prepared, WARM_START="0")
    at = args.index("--slot-save-path")
    assert served(off) == args[:at] + args[at + 2:] and helper_runs(prepared) == []
    assert next(json.loads(line) for line in off.stdout.splitlines() if '"server_args"' in line)["slots_debug"] is None


def test_the_warm_start_follows_the_servers_knobs(prepared, tmp_path):
    key = tmp_path / "brain.key"
    key.write_text("warm-knob-key-91\n")
    key.chmod(0o600)
    record = tmp_path / "records" / "prefix.json"
    result = invoke(prepared, PORT="18492", PARALLEL="3", API_KEY_FILE=str(key), CHAT_TEMPLATE_FILE="",
                    WARM_PREFIX_FILE=str(record))
    served(result)
    (warm,) = helper_runs(prepared)
    assert (flag(warm, "--port"), flag(warm, "--slots"), flag(warm, "--prefix-file")) == ("18492", "3", str(record))
    assert flag(warm, "--api-key-file") == str(key) and flag(warm, "--chat-template-file") == ""
    assert flag(warm, "--slot-save-dir") == flag(served(result), "--slot-save-path") == str(record.parent / "bolt-slots-18492")
    assert "warm-knob-key-91" not in json.dumps(warm) + result.stdout + result.stderr


def test_the_default_record_lives_in_the_xdg_state_home(prepared, tmp_path):
    served(invoke(prepared, XDG_STATE_HOME=str(tmp_path / "xdg")))
    (warm,) = helper_runs(prepared)
    assert flag(warm, "--prefix-file") == str(tmp_path / "xdg/cascade/bolt-openclaw-prefix-8080.json")


def test_check_never_starts_the_helper(prepared):
    assert invoke(prepared, "--check").returncode == 0
    assert helper_runs(prepared) == []


def test_a_missing_warm_start_helper_is_refused_unless_the_warm_start_is_off(prepared):
    (prepared["repo"] / WARM).unlink()
    for args in ((), ("--check",)):
        result = invoke(prepared, *args)
        assert result.returncode == 1 and "WARM_START" in result.stderr, result.stderr
        assert '"server_args"' not in result.stdout
    assert served(invoke(prepared, WARM_START="0"))


def test_spec_none_serves_without_the_drafter(prepared):
    args = served(invoke(prepared, SPEC="none"))
    assert not [a for a in args if a.startswith("--spec")]
    assert "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf" not in " ".join(args)
    assert flag(args, "--model").endswith("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf") and "--mmproj" in args
    assert flag(args, "--load-mode") == "none" and flag(args, "--parallel") == "2"


def test_knobs_reach_the_server(prepared):
    args = served(invoke(prepared, PORT="18492", CTX="32768", PARALLEL="1", NSPEC="6", ALIAS="bolt"))
    assert (flag(args, "--port"), flag(args, "--ctx-size"), flag(args, "--parallel"), flag(args, "--spec-draft-n-max"),
            flag(args, "--alias")) == ("18492", "32768", "1", "6", "bolt")


@pytest.mark.parametrize("env, message", [({"ALIAS": "nvidia/bolt"}, "bare id"), ({"PARALLEL": "0"}, "PARALLEL"),
                                          ({"PORT": "70000"}, "PORT"), ({"CTX": "64k"}, "CTX"),
                                          ({"SPEC": "eagle"}, "SPEC must be dflash|none"),
                                          ({"WARM_START": "yes"}, "WARM_START must be 0|1")])
def test_bad_knobs_are_refused(prepared, env, message):
    result = invoke(prepared, **env)
    assert result.returncode == 2 and message in result.stderr


def test_the_key_file_path_reaches_the_server_never_the_key(prepared, tmp_path):
    key = tmp_path / "brain.key"
    key.write_text("bolt-gguf-key-4e1\n")
    key.chmod(0o600)
    result = invoke(prepared, API_KEY_FILE=str(key))
    assert flag(served(result), "--api-key-file") == str(key)
    assert "bolt-gguf-key-4e1" not in result.stdout + result.stderr


@pytest.mark.parametrize("problem", ["loose", "empty", "missing"])
def test_a_bad_key_file_is_refused(prepared, tmp_path, problem):
    key = tmp_path / "brain.key"
    if problem != "missing":
        key.write_text("" if problem == "empty" else "k\n")
        key.chmod(0o644 if problem == "loose" else 0o600)
    result = invoke(prepared, API_KEY_FILE=str(key))
    assert result.returncode == 1 and "API_KEY_FILE" in result.stderr
    assert '"server_args"' not in result.stdout


def test_check_verifies_the_pinned_files_and_the_server_without_starting_it(prepared):
    result = invoke(prepared, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"server_args"' not in result.stdout and "no runtime proof performed" in result.stdout


def test_a_tampered_model_is_refused(prepared):
    (prepared["dir"] / "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf").write_bytes(b"GGUF tampered")
    result = invoke(prepared)
    assert result.returncode != 0 and '"server_args"' not in result.stdout


def test_a_server_without_dflash_is_refused(prepared):
    prepared["server"].write_text(f"#!{sys.executable}\nprint('--mmproj --alias --chat-template-kwargs --reasoning')\n")
    result = invoke(prepared)
    assert result.returncode == 1 and "llama-server lacks" in result.stderr


def test_both_llamacpp_brains_pin_llamacpp_v070():
    pins = {name: re.search(r"^LLAMA_REF=([0-9a-f]{40})", (REPO / "scripts" / name).read_text(), re.M).group(1)
            for name in ("serve_nemotron_bolt_llamacpp.sh", "serve_qwen_llamacpp.sh")}
    assert set(pins.values()) == {"620d2225cc436ab0b001e3200e590e3b286ef9ab"}


def test_the_bolt_manifest_pins_the_official_gguf_files():
    sys.path.insert(0, str(REPO / "deploy" / "brev"))
    import fetch_models

    artifacts = fetch_models.load_manifest(REPO / MANIFEST, 32 * 1024**3)
    assert {a.filename: a.sha256 for a in artifacts} == {
        "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf": "91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75",
        "mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf": "0ee891eadc7bf0cae6289b249c74eb1ef420534351f6ef8b64d49f0ca6a926b0",
        "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf": "080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194",
        "LICENSE": "8a0c5e232551abcb102d1bd39a34866ef4520361b613bd6405d55b36562e4d88"}
    assert fetch_models.manifest_directory(REPO / MANIFEST) == Path("models/nemotron-3.5-bolt")
    assert sum(a.size_bytes for a in artifacts) == 24750264442


@pytest.mark.parametrize("change, message", [
    (lambda m: m["quantization"]["files"][2].update(repo_id="evil/Bolt-GGUF"), "reviewed quantization"),
    (lambda m: m["source_model"].update(repo_id="nvidia/Other"), "reviewed official model"),
    (lambda m: m["source_model"]["license"].update(repo_id="evil/x"), "reviewed quantization"),
    (lambda m: m["quantization"]["files"][2].update(revision="main"), "complete quantization revision"),
    (lambda m: m["quantization"].update(repo_id="evil/Bolt-GGUF"), "reviewed quantization"),
    (lambda m: m["quantization"].pop("repo_id"), "reviewed quantization"),
    (lambda m: m["quantization"].update(revision="main"), "complete quantization revision"),
])
def test_the_bolt_manifest_refuses_unreviewed_sources(tmp_path, change, message):
    sys.path.insert(0, str(REPO / "deploy" / "brev"))
    import fetch_models

    data = json.loads((REPO / MANIFEST).read_text())
    change(data)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(data))
    with pytest.raises(fetch_models.DownloadError, match=message):
        fetch_models.load_manifest(path, 32 * 1024**3)
