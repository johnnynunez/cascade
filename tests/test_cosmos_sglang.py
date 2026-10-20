"""scripts/serve_cosmos_sglang.sh: Cosmos3-Edge on SGLang, keyed from a file, scrubbed, loopback, no remote code.

#409 made it a route of the OpenShell Cosmos agent (`launch.sh --brain cosmos-sglang`); the #415 review found it served
without a key, passed the caller's whole environment and ran with --trust-remote-code.

What the script must keep (SGLang 0.5.22, the version it pins, read from its wheel):
- A key, read from a 0600 file holding one key. SGLang takes it only as `--api-key KEY` (no key file, no variable), so
  the script's launcher reads the file inside the server process: /proc/<pid>/cmdline, readable by every local user,
  carries the path, never the key.
- No inherited variable that adds a listener, moves a route or swaps the HTTP server (HOSTILE_SGLANG_ENV).
- 127.0.0.1 only.
- No --trust-remote-code: Cosmos3-Edge is native in SGLang (srt/models/cosmos3_edge.py) and in the transformers it
  pins (5.19.0, models/cosmos3_edge), and nvidia/Cosmos3-Edge ships no code (no .py, no auto_map at a9d944e2). The
  vLLM route serves the same export with trust_remote_code=False (docs/evidence/cosmos-vllm-032-nvfp4-2026-10-19).

Fakes only: the venv's python is a stand-in that runs the script's launcher for real against a fake `sglang` package,
whose `launch_server` prints its argv, its environment and /proc/self/cmdline. The transformers preflight, the modeling
patch and the export (`python -` heredocs) are skipped. Nothing is installed, exported or served.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from inherited_server_env import BENIGN_ENV, BENIGN_SGLANG_ENV, HOSTILE_SGLANG_ENV, dropped, leaked

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "serve_cosmos_sglang.sh"
SGLANG_VERSION = "0.5.22"
#: SGLang 0.5.22's own pin, and scripts/cosmos_serving.py's (the vLLM route): Cosmos3-Edge is native in it.
TRANSFORMERS_VERSION = "5.19.0"
KEY = "sglang-brain-key-7"

#: The venv's python: `-` (a heredoc: the transformers preflight, the modeling patch, the export) is skipped;
#: `-c` and `-m` run for real with the fake `sglang` package (fake-site, beside bin/) first on sys.path.
FAKE_PYTHON = """
import os, runpy, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fake-site"))
args = sys.argv[1:]
if args[:1] == ["-"]:
    sys.stdin.read()
    raise SystemExit(0)
if args[:1] == ["-c"]:
    sys.argv = ["-c", *args[2:]]
    exec(compile(args[1], "<string>", "exec"), {"__name__": "__main__"})
elif args[:1] == ["-m"]:
    sys.argv = [args[1], *args[2:]]
    runpy.run_module(args[1], run_name="__main__", alter_sys=True)
else:
    raise SystemExit(f"fake python: unexpected {args}")
"""
LAUNCH_SERVER = """
import json, os, sys
if __name__ == "__main__":
    with open("/proc/self/cmdline", "rb") as f:
        cmdline = f.read().decode().split("\\0")
    print(json.dumps({"sglang_args": sys.argv[1:], "env": dict(os.environ), "cmdline": cmdline}))
"""
#: Installs the fake `sglang` (`launch_server` above, metadata of `version`) into a fake venv's fake-site, and the
#: metadata of another distribution (`transformers`) the script checks.
INSTALL_FAKE_SGLANG = """
def install_fake_dist(venv, name, version):
    import os
    dist = os.path.join(venv, "fake-site", name + "-" + version + ".dist-info")
    os.makedirs(dist)
    with open(os.path.join(dist, "METADATA"), "w") as f:
        f.write("Metadata-Version: 2.1\\nName: " + name + "\\nVersion: " + version + "\\n")


def install_fake_sglang(venv, version):
    import os
    site = os.path.join(venv, "fake-site")
    os.makedirs(os.path.join(site, "sglang"), exist_ok=True)
    open(os.path.join(site, "sglang", "__init__.py"), "w").close()
    with open(os.path.join(site, "sglang", "launch_server.py"), "w") as f:
        f.write(LAUNCH_SERVER)
    install_fake_dist(venv, "sglang", version)
"""
_namespace = {"LAUNCH_SERVER": LAUNCH_SERVER}
exec(INSTALL_FAKE_SGLANG, _namespace)
install_fake_sglang = _namespace["install_fake_sglang"]
install_fake_dist = _namespace["install_fake_dist"]
#: `uv` as the script calls it: `venv PATH` makes the fake venv, `pip install` of `sglang[==X]` installs the fake
#: package (as X; unpinned: 9.9.9), `transformers==X` its metadata (a git URL: 5.20.0.dev0, git main). Every call
#: is logged.
FAKE_UV = f"LAUNCH_SERVER = {LAUNCH_SERVER!r}\n" + INSTALL_FAKE_SGLANG + """
import json, os, re, shutil, sys
a = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as log:
    log.write(json.dumps(a) + "\\n")
if a[0] == "venv":
    os.makedirs(os.path.join(a[1], "bin"))
    shutil.copy(os.environ["FAKE_UV_PYTHON"], os.path.join(a[1], "bin", "python"))
elif a[:2] == ["pip", "install"]:
    venv = os.path.dirname(os.path.dirname(a[a.index("--python") + 1]))
    for spec in a[2:]:
        m = re.fullmatch(r"sglang(?:\\[[a-z,]+\\])?(?:==(.+))?", spec)
        if m:
            install_fake_sglang(venv, m.group(1) or "9.9.9")
        m = re.fullmatch(r"transformers==(.+)", spec)
        if m:
            install_fake_dist(venv, "transformers", m.group(1))
        if spec.startswith("git+https://github.com/huggingface/transformers"):
            install_fake_dist(venv, "transformers", "5.20.0.dev0")
"""


def _python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{FAKE_PYTHON}")
    path.chmod(0o755)
    return path


@pytest.fixture
def venv(tmp_path) -> Path:
    v = tmp_path / "venv"
    _python(v / "bin" / "python")
    install_fake_sglang(str(v), SGLANG_VERSION)
    install_fake_dist(str(v), "transformers", TRANSFORMERS_VERSION)
    return v


def _key(tmp_path, text: str = KEY + "\n", mode: int = 0o600) -> dict:
    key = tmp_path / "brain.key"
    key.write_text(text)
    key.chmod(mode)
    return {"API_KEY_FILE": str(key)}


def run(tmp_path, venv, *, bash=("bash",), **env) -> subprocess.CompletedProcess:
    export = tmp_path / "export"
    export.mkdir(exist_ok=True)
    (export / "config.json").write_text("{}")          # an existing export: no re-export
    (tmp_path / "home").mkdir(exist_ok=True)
    base = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), "VENV": str(venv), "EXPORT_DIR": str(export)}
    return subprocess.run([*bash, str(SCRIPT)], env={**base, **env}, capture_output=True, text=True, timeout=60)


def served(r: subprocess.CompletedProcess) -> dict:
    assert r.returncode == 0, r.stdout + r.stderr
    return next(json.loads(line) for line in r.stdout.splitlines() if '"sglang_args"' in line)


def refused(r: subprocess.CompletedProcess) -> str:
    assert r.returncode != 0 and '"sglang_args"' not in r.stdout, r.stdout + r.stderr
    return r.stderr


def flag(args: list[str], name: str) -> str:
    assert args.count(name) == 1, (name, args)
    return args[args.index(name) + 1]


def test_cosmos_serves_on_loopback_with_its_key_off_argv(tmp_path, venv):
    key = _key(tmp_path)
    out = served(run(tmp_path, venv, **key))
    a = out["sglang_args"]
    assert flag(a, "--host") == "127.0.0.1"
    assert flag(a, "--api-key") == KEY                       # the server's own argv (sys.argv), read from the file
    assert not any(KEY in part for part in out["cmdline"]), out["cmdline"]
    assert key["API_KEY_FILE"] in out["cmdline"]             # what every local user sees: the path
    assert "--trust-remote-code" not in a
    assert flag(a, "--model-path") == str(tmp_path / "export")
    assert (flag(a, "--served-model-name"), flag(a, "--port")) == ("cosmos3-edge", "8083")


@pytest.mark.parametrize("problem", ["unset", "missing", "group-readable", "world-readable", "empty", "two keys"])
def test_a_brain_without_one_private_key_is_refused(tmp_path, venv, problem):
    """The Cosmos agent route reaches this brain: it never serves without a key, and the key file is as the other
    brains' (present, no group/other bits, non-empty), holding exactly one key (SGLang takes one)."""
    env = {
        "unset": lambda: {},
        "missing": lambda: {"API_KEY_FILE": str(tmp_path / "no-such.key")},
        "group-readable": lambda: _key(tmp_path, mode=0o640),
        "world-readable": lambda: _key(tmp_path, mode=0o604),
        "empty": lambda: _key(tmp_path, text=" \n"),
        "two keys": lambda: _key(tmp_path, text=f"{KEY}\nanother-key\n"),
    }[problem]()
    why = refused(run(tmp_path, venv, **env))
    assert "API_KEY_FILE" in why
    assert ("is required" in why) == (problem == "unset"), why


def test_inherited_sglang_route_switches_never_reach_the_server(tmp_path, venv):
    """SGLang 0.5.22 takes these from the environment: a gRPC listener outside the key, another HTTP server, routes
    moved into /v1/**, a keyless dumper server on 0.0.0.0, header overrides of request fields, a web search tool."""
    out = served(run(tmp_path, venv, **_key(tmp_path), **HOSTILE_SGLANG_ENV))
    assert leaked(out["env"], HOSTILE_SGLANG_ENV) == {}


def test_the_rest_of_the_callers_environment_reaches_sglang(tmp_path, venv):
    benign = {**BENIGN_ENV, **BENIGN_SGLANG_ENV}
    out = served(run(tmp_path, venv, **_key(tmp_path), **HOSTILE_SGLANG_ENV, **benign))
    assert dropped(out["env"], benign) == {}


def test_a_bash_without_compgen_refuses_instead_of_serving_unscrubbed(tmp_path, venv):
    why = refused(run(tmp_path, venv, bash=("bash", "-c", 'enable -n compgen && . "$0" "$@"'),
                      **_key(tmp_path), **HOSTILE_SGLANG_ENV))
    assert "compgen" in why


def _scrub_patterns() -> list[str]:
    (line,) = [x for x in SCRIPT.read_text().splitlines() if x.strip().startswith('case "$name" in ')]
    return line.split(" in ", 1)[1].split(")", 1)[0].strip().split("|")


def test_the_scrub_is_the_audited_list_and_every_entry_is_exercised():
    patterns = _scrub_patterns()
    assert patterns == ["SGLANG_GRPC_PORT", "SGLANG_RUST_SERVER", "SGLANG_OLLAMA_*", "AIP_*", "DUMPER_*",
                        "SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES", "EXA_API_KEY", "SGLANG_EXTERNAL_*",
                        "SGLANG_LOGGING_CONFIG_PATH", "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"]
    assert all(any(fnmatch.fnmatchcase(name, p) for name in HOSTILE_SGLANG_ENV) for p in patterns)
    assert not any(fnmatch.fnmatchcase(name, p) for name in {**BENIGN_ENV, **BENIGN_SGLANG_ENV} for p in patterns)
    assert f'SGLANG_VERSION={SGLANG_VERSION}' in SCRIPT.read_text().splitlines()


def test_a_venv_with_another_sglang_is_refused(tmp_path):
    """The scrub and the route audit hold for the pinned SGLang only."""
    v = tmp_path / "venv"
    _python(v / "bin" / "python")
    install_fake_sglang(str(v), "0.5.21")
    why = refused(run(tmp_path, v, **_key(tmp_path)))
    assert "0.5.21" in why and SGLANG_VERSION in why


def test_a_missing_venv_is_built_on_the_pinned_sglang(tmp_path):
    bin_dir = tmp_path / "uv-bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(f"#!{sys.executable}\n{FAKE_UV}")
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "uv.jsonl"
    out = served(run(tmp_path, tmp_path / "fresh-venv", **_key(tmp_path), PATH=f"{bin_dir}:/usr/bin:/bin",
                     FAKE_UV_LOG=str(log), FAKE_UV_PYTHON=str(_python(tmp_path / "template" / "python"))))
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    installs = [c for c in calls if c[:2] == ["pip", "install"]]
    assert any(f"sglang=={SGLANG_VERSION}" in c for c in installs), calls
    assert any(f"transformers=={TRANSFORMERS_VERSION}" in c for c in installs), calls
    assert not any(spec.startswith("git+") for c in installs for spec in c), "no unpinned transformers over the pin"
    assert flag(out["sglang_args"], "--api-key") == KEY


def test_a_venv_with_another_transformers_is_refused(tmp_path):
    """Cosmos3-Edge loads without remote code on transformers 5.19.0 (SGLang 0.5.22's pin); a venv an earlier
    version of this script built holds transformers git main."""
    v = tmp_path / "venv"
    _python(v / "bin" / "python")
    install_fake_sglang(str(v), SGLANG_VERSION)
    install_fake_dist(str(v), "transformers", "5.20.0.dev0")
    why = refused(run(tmp_path, v, **_key(tmp_path)))
    assert "transformers 5.20.0.dev0" in why and TRANSFORMERS_VERSION in why


def test_sglang_logs_at_warning_so_its_start_line_never_prints_the_key(tmp_path, venv):
    """SGLang 0.5.22 logs its resolved arguments, --api-key included, at INFO (entrypoints/engine.py)."""
    out = served(run(tmp_path, venv, **_key(tmp_path)))
    assert flag(out["sglang_args"], "--log-level") == "warning"
    assert "--log-level-http" not in out["sglang_args"], "the HTTP log level follows --log-level"


def _cosmos_serving():
    spec = importlib.util.spec_from_file_location("cosmos_serving_pins", REPO / "scripts" / "cosmos_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_model_and_transformers_are_pinned_as_on_the_vllm_route():
    """The same checkpoint revision and transformers as scripts/cosmos_serving.py; every download names it."""
    pins = _cosmos_serving()
    lines = SCRIPT.read_text().splitlines()
    assert f"MODEL_REVISION={pins.MODEL_REVISION}" in lines
    assert f"TRANSFORMERS_VERSION={pins.TRANSFORMERS_VERSION}" in lines and pins.TRANSFORMERS_VERSION == TRANSFORMERS_VERSION
    downloads = [x for x in lines if "from_pretrained(" in x and not x.lstrip().startswith("#")]
    assert len(downloads) == 3, downloads    # the config preflight, the model, the processor
    assert all("revision=" in x for x in downloads), downloads
    assert "git+https://github.com/huggingface/transformers" not in SCRIPT.read_text()
    assert "has no `all` extra" not in SCRIPT.read_text()
