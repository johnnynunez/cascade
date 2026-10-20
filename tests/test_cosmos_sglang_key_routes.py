"""The keyed Cosmos SGLang brain through every shipped start path (#416 review, Important 1).

scripts/serve_cosmos_sglang.sh serves only with API_KEY_FILE since #416: the one local brain that cannot run keyless.
Every start path that reaches it sends that key, read from a FILE (never argv, never a config file of this repo), and
needs no extra variable when the brain key file every local brain uses (~/.cascade/nemoclaw/llama.key) exists:

- launch.sh, OpenShell route: the brain checks (demo_proof --check-brain / --probe-native) and the agent's brain
  provider (`openshell_rig.py up --brain-key-file`) use the same file; no provider key is minted for it.
- launch.sh, host route (macOS, the Spark kitchen): `openclaw onboard` gets the key as CUSTOM_API_KEY in its own
  environment (OpenClaw 2026.9.3 reads a custom provider's key only from --custom-api-key or CUSTOM_API_KEY), and the
  provider's context window is then set without the key on any argv.
- scripts/openclaw_demo.sh: its /models probe sends the key as a header read from a file descriptor, and onboarding
  gets CUSTOM_API_KEY the same way.
- demo_proof --check-brain cosmos-sglang falls back to the brain key file.

The other brains (vLLM Cosmos, Bolt) keep their start paths: no key sent, a minted provider key, the same commands.

Fakes only: the brain is an HTTP server on an ephemeral loopback port that answers /v1/* only with `Authorization:
Bearer <key>` (401 otherwise, as SGLang 0.5.22's key middleware does), and every test runs with a private HOME, so the
host's real key file is never read. The end-to-end test starts the real serve_cosmos_sglang.sh on a fake venv whose
`sglang.launch_server` serves that keyed API on an ephemeral port. No model, GPU, docker or OpenShell is started.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from test_launch_delivery import launcher_boundary as _launcher_boundary
from test_launch_openshell_route import _rig
from test_launch_openshell_route import linux_launcher as _linux_launcher

# The launcher fixtures, resolved against this module's keyed model_http_boundary.
launcher_boundary = _launcher_boundary
linux_launcher = _linux_launcher

REPO = Path(__file__).resolve().parents[1]
KEY = "cosmos-sglang-test-key-5c1e"
DEFAULT_KEY = Path(".cascade") / "nemoclaw" / "llama.key"   # under HOME: the brain key file of every local brain
KEY_VARS = ("CASCADE_MODEL_API_KEY", "CASCADE_MODEL_API_KEY_FILE", "CASCADE_BRAIN_KEY_FILE", "CUSTOM_API_KEY")


def _keyed_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _authorized(self) -> bool:
            state["auth"].append((self.command, self.path, self.headers.get("Authorization")))
            if not self.path.startswith("/v1") or not state["require_key"]:
                return True
            if self.headers.get("Authorization") == f"Bearer {state['key']}":
                return True
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": "Unauthorized"}')
            return False

        def do_GET(self):
            if not self._authorized():
                return
            state["requests"].append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": state["model"]}]}).encode())

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self._authorized():
                return
            state["requests"].append(data)
            payload = {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "cascade__world_state", "arguments": "{}"}}]}}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
    return Handler


@pytest.fixture
def model_http_boundary():
    """A Cosmos SGLang brain that answers /v1/* only with its key (the launcher fixtures use this one)."""
    state = {"model": "cosmos3-edge", "requests": [], "auth": [], "key": KEY, "require_key": True}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _keyed_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_key(path: Path, text: str = KEY + "\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)
    return path


def _keyed_env(h, keys: str) -> tuple[dict, Path | None]:
    """The launch's env for one way of naming the key, with a private HOME; returns (env, the key file)."""
    tmp = h["tmp"] if "tmp" in h else Path(h["env"]["CASCADE_LAUNCH_STATE"]).parent
    home = tmp / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in h["env"].items() if k not in KEY_VARS}
    env.update(HOME=str(home), CASCADE_COSMOS_SGLANG_BASE_URL=h["http"]["base_url"])
    key_file = {
        "default key file": lambda: _write_key(home / DEFAULT_KEY),
        "model key file": lambda: _write_key(tmp / "model.key"),
        "brain key file": lambda: _write_key(tmp / "brain.key"),
        "both": lambda: _write_key(tmp / "brain.key"),
        "no key file": lambda: None,
    }[keys]()
    if keys in ("model key file", "both"):
        env["CASCADE_MODEL_API_KEY_FILE"] = str(key_file)
    if keys in ("brain key file", "both"):
        env["CASCADE_BRAIN_KEY_FILE"] = str(key_file)
    return env, key_file


def _brain(command: list[str], brain: str = "cosmos-sglang") -> list[str]:
    cmd = list(command)
    cmd[cmd.index("--brain") + 1] = brain
    return cmd


def _launch(h, env, *extra, brain="cosmos-sglang") -> subprocess.CompletedProcess:
    return subprocess.run([*_brain(h["command"], brain), *extra], env=env, capture_output=True, text=True,
                          timeout=120, check=False)


def _unauthorized(h) -> list:
    return [a for a in h["http"]["auth"] if a[1].startswith("/v1") and a[2] != f"Bearer {KEY}"]


def _argv_logs(h) -> str:
    """Every argv the fakes recorded (rig, python, sandbox and host OpenClaw) and the robot server's env file."""
    names = ("RIG_LOG", "PY_LOG", "SANDBOX_LOG", "HOST_LOG", "OC_ENV_LOG")
    paths = [Path(h["env"][n]) for n in names if n in h["env"]]
    if "RIG_HOME" in h["env"]:
        paths.append(Path(h["env"]["RIG_HOME"]) / "mcp-env.seen.json")
    return "\n".join(p.read_text() for p in paths if p.exists())


# ── launch.sh, OpenShell route (Linux) ─────────────────────────────────────


@pytest.mark.parametrize("keys", ["default key file", "model key file", "brain key file", "both"])
def test_the_openshell_route_sends_one_key_file_to_the_checks_and_the_provider(linux_launcher, keys):
    """No variable at all when the brain key file exists; CASCADE_MODEL_API_KEY_FILE or CASCADE_BRAIN_KEY_FILE alone
    is enough otherwise. The provider gets that same file (a minted key would 401 at `up`)."""
    h = linux_launcher
    env, key_file = _keyed_env(h, keys)
    p = _launch(h, env)
    assert p.returncode == 0, p.stdout + p.stderr
    (up,) = [c[2:] for c in _rig(h) if c[2] == "up"]
    assert Path(up[up.index("--brain-key-file") + 1]) == key_file, "the brain's own key file, never a minted one"
    assert not (h["state"] / "openshell-brain.key").exists()
    assert h["http"]["requests"] and _unauthorized(h) == [], "every brain check sent the key"
    assert KEY not in _argv_logs(h)


def test_the_openshell_route_without_a_key_file_refuses_and_names_the_key_file(linux_launcher):
    h = linux_launcher
    env, _ = _keyed_env(h, "no key file")
    p = _launch(h, env)
    assert p.returncode != 0
    assert f"--brain cosmos-sglang: no key file at {Path(env['HOME']) / DEFAULT_KEY}" in p.stderr, p.stderr
    assert "HTTP 401" in p.stderr and "serve_cosmos_sglang.sh" in p.stderr, p.stderr
    assert "API_KEY_FILE" in p.stderr and str(DEFAULT_KEY) in p.stderr, p.stderr
    assert "up" not in [c[2] for c in _rig(h)], "nothing came up"


def test_a_literal_model_key_still_wins_over_the_key_file(linux_launcher):
    """CASCADE_MODEL_API_KEY is what the brain checks send first: the provider gets that key (in a minted 0600
    file, as before), not a key file the checks did not use."""
    h = linux_launcher
    env, _ = _keyed_env(h, "model key file")
    _write_key(Path(env["CASCADE_MODEL_API_KEY_FILE"]), "a-stale-key\n")
    env["CASCADE_MODEL_API_KEY"] = KEY
    p = _launch(h, env)
    assert p.returncode == 0, p.stdout + p.stderr
    (up,) = [c[2:] for c in _rig(h) if c[2] == "up"]
    provider_key = Path(up[up.index("--brain-key-file") + 1])
    assert provider_key == h["state"] / "openshell-brain.key" and provider_key.read_text().strip() == KEY
    assert _unauthorized(h) == []


def test_check_reports_the_keyed_brain_ok_with_the_default_key_file(linux_launcher):
    h = linux_launcher
    env, _ = _keyed_env(h, "default key file")
    p = _launch(h, env, "--check")
    assert "  [ok]      brain: " in p.stdout and "cosmos3-edge" in p.stdout, p.stdout + p.stderr
    assert _unauthorized(h) == []
    env, _ = _keyed_env(h, "no key file")
    (Path(env["HOME"]) / DEFAULT_KEY).unlink()
    p = _launch(h, env, "--check")
    assert "  [MISSING] brain: " in p.stdout and "HTTP 401" in p.stdout, p.stdout


@pytest.mark.parametrize("brain, model", [("cosmos", "cosmos3-edge"), ("nemotron-bolt", "nemotron-3.5-bolt")])
def test_the_other_brains_keep_their_openshell_start_path(linux_launcher, brain, model):
    """Even with the brain key file present: vLLM Cosmos is checked with no key, and neither brain's provider reuses
    the brain checks' key file (each gets the minted one, as before)."""
    h = linux_launcher
    h["http"].update(require_key=False, model=model)
    env, _ = _keyed_env(h, "default key file")
    env["CASCADE_NEMOTRON_BOLT_BASE_URL"] = h["http"]["base_url"]
    p = _launch(h, env, brain=brain)
    assert p.returncode == 0, p.stdout + p.stderr
    (up,) = [c[2:] for c in _rig(h) if c[2] == "up"]
    assert Path(up[up.index("--brain-key-file") + 1]) == h["state"] / "openshell-brain.key"
    sent = {a[2] for a in h["http"]["auth"]}
    assert sent == ({None} if brain == "cosmos" else {f"Bearer {KEY}"}), sent


# ── launch.sh, host route (macOS; the Spark kitchen) ───────────────────────

#: OpenClaw 2026.9.3 as the host route drives it, for the calls that touch the custom provider (the rest goes to the
#: launcher_boundary fake). Real words (dist/onboard-custom-config): the provider id is custom-<host>-<port>, and the
#: key from --custom-api-key or CUSTOM_API_KEY is stored as models.providers.<id>.apiKey.
FAKE_OPENCLAW = r'''
import json, os, re, sys
from pathlib import Path
from urllib.parse import urlsplit
args = sys.argv[1:]
with open(os.environ["OC_ENV_LOG"], "a") as f:
    f.write(json.dumps({"argv": args, "CUSTOM_API_KEY": os.environ.get("CUSTOM_API_KEY")}) + "\n")
rest = args[2:] if args[:1] == ["--profile"] else args
cfg = Path(os.environ["HOST_CFG"])
provider_path = rest[:2] == ["config", "set"] and len(rest) > 3 and rest[2].startswith("models.providers.")
if rest[:1] == ["onboard"] or provider_path:
    with open(os.environ["HOST_LOG"], "a") as f:
        f.write(json.dumps(args) + "\n")
if rest[:1] == ["onboard"]:
    base = rest[rest.index("--custom-base-url") + 1]
    u = urlsplit(base)
    pid = "custom-" + re.sub(r"[^a-z0-9]+", "-", u.hostname, flags=re.I).lower() + (f"-{u.port}" if u.port else "")
    provider = {"baseUrl": base, "api": "openai-completions",
                "models": [{"id": rest[rest.index("--custom-model-id") + 1], "contextWindow": 128000}]}
    if os.environ.get("CUSTOM_API_KEY"):
        provider["apiKey"] = os.environ["CUSTOM_API_KEY"]
    cfg.write_text(json.dumps({"models": {"providers": {pid: provider}}}))
elif provider_path:
    _, _, pid, field = rest[2].split(".")
    data = json.loads(cfg.read_text())
    data["models"]["providers"][pid][field] = json.loads(rest[3])
    cfg.write_text(json.dumps(data))
else:
    os.execv(os.environ["OC_BASE"], [os.environ["OC_BASE"], *args])
'''


def _host_openclaw(h, env: dict) -> None:
    bindir = Path(h["env"]["PATH"].split(os.pathsep)[0])
    base = bindir / "openclaw-base"
    if not base.exists():
        (bindir / "openclaw").rename(base)
        (bindir / "openclaw").write_text(f"#!{sys.executable}\n{FAKE_OPENCLAW}")
        (bindir / "openclaw").chmod(0o755)
    tmp = Path(h["env"]["CASCADE_LAUNCH_STATE"]).parent
    env.update(OC_BASE=str(base), OC_ENV_LOG=str(tmp / "oc-env.jsonl"))
    h["env"]["OC_ENV_LOG"] = env["OC_ENV_LOG"]


def _oc_calls(h) -> list[dict]:
    log = Path(h["env"]["OC_ENV_LOG"])
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("keys", ["default key file", "model key file"])
def test_the_host_route_onboards_the_provider_with_the_key_in_its_environment(launcher_boundary, keys):
    h = launcher_boundary
    env, _ = _keyed_env(h, keys)
    _host_openclaw(h, env)
    p = _launch(h, env)
    assert p.returncode == 0, p.stdout + p.stderr
    onboard = [c for c in _oc_calls(h) if "onboard" in c["argv"]]
    assert len(onboard) == 1 and onboard[0]["CUSTOM_API_KEY"] == KEY, onboard
    assert [c["argv"] for c in _oc_calls(h) if c["CUSTOM_API_KEY"] == KEY] == [onboard[0]["argv"]], \
        "only onboarding gets the key"
    (provider,) = json.loads(Path(h["env"]["HOST_CFG"]).read_text())["models"]["providers"].values()
    assert provider["apiKey"] == KEY, "OpenClaw keeps the key it was given"
    assert provider["models"][0]["contextWindow"] == 32768, "the brain's context, set without the key"
    assert _unauthorized(h) == []
    assert KEY not in _argv_logs(h).replace(f'"CUSTOM_API_KEY": "{KEY}"', ""), "the key is on no argv"


def test_the_host_route_without_a_key_file_refuses_before_onboarding(launcher_boundary):
    h = launcher_boundary
    env, _ = _keyed_env(h, "no key file")
    _host_openclaw(h, env)
    p = _launch(h, env)
    assert p.returncode != 0 and "HTTP 401" in p.stderr and "serve_cosmos_sglang.sh" in p.stderr, p.stderr
    assert not [c for c in _oc_calls(h) if "onboard" in c["argv"]]


@pytest.mark.parametrize("keys", ["default key file", "model key file"])
def test_vllm_cosmos_keeps_its_host_start_path(launcher_boundary, keys):
    """The brain key file present, or an operator's model key file, changes nothing for vLLM Cosmos: its checks send
    what they sent before, onboarding gets no key, and the whole providers map is set as before."""
    h = launcher_boundary
    h["http"]["require_key"] = False
    env, _ = _keyed_env(h, keys)
    _host_openclaw(h, env)
    p = _launch(h, env, brain="cosmos")
    assert p.returncode == 0, p.stdout + p.stderr
    sent = {a[2] for a in h["http"]["auth"]}
    assert sent == ({None} if keys == "default key file" else {f"Bearer {KEY}"}), sent
    assert all(c["CUSTOM_API_KEY"] is None for c in _oc_calls(h))
    sets = [c["argv"] for c in _oc_calls(h) if c["argv"][2:5] == ["config", "set", "models.providers"]]
    assert len(sets) == 1, [c["argv"] for c in _oc_calls(h)]


# ── scripts/openclaw_demo.sh ───────────────────────────────────────────────

#: curl as openclaw_demo.sh calls it, against a keyed brain on :8083 that is never dialled: the headers it would
#: send (`-H @file` read here, like curl does) decide 200 or 401; `--fail` maps 401 to exit 22, no server to exit 7.
FAKE_CURL = r'''
import json, os, sys
args = sys.argv[1:]
headers = []
for i, a in enumerate(args):
    if a == "-H":
        value = args[i + 1]
        headers += open(value[1:]).read().splitlines() if value.startswith("@") else [value]
with open(os.environ["CURL_LOG"], "a") as f:
    f.write(json.dumps({"argv": args, "headers": headers}) + "\n")
url = args[-1]
if os.environ.get("DEMO_BRAIN") == "none" or not url.endswith("/v1/models"):
    print("000" if "-w" in args else "", end="")
    sys.exit(7)
ok = os.environ.get("DEMO_BRAIN") == "open" or f"Authorization: Bearer {os.environ['DEMO_KEY']}" in headers
code = 200 if ok else 401
if "-w" in args:
    print(code, end="")
    sys.exit(0)
if not ok:
    sys.exit(22 if ("-sf" in args or "-f" in args) else 0)
print(json.dumps({"data": [{"id": "cosmos3-edge"}]}))
'''
FAKE_DEMO_OPENCLAW = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["OC_ENV_LOG"], "a") as f:
    f.write(json.dumps({"argv": args, "CUSTOM_API_KEY": os.environ.get("CUSTOM_API_KEY")}) + "\n")
if args[:1] == ["onboard"]:
    cfg = Path(os.environ["HOME"]) / ".openclaw" / "openclaw.json"
    cfg.parent.mkdir(exist_ok=True)
    mid = args[args.index("--custom-model-id") + 1]
    cfg.write_text(json.dumps({"models": {"providers": {"custom": {"models": [{"id": mid}]}}}}))
'''
#: openclaw_demo.sh's "$PY": the editable-install check passes; everything else runs for real.
FAKE_DEMO_PYTHON = r'''
import os, sys
args = sys.argv[1:]
if args == ["-c", "import cascade"]:
    sys.exit(0)
os.execv(os.environ["REAL_PY"], [os.environ["REAL_PY"], *args])
'''


@pytest.fixture
def demo_host(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, text in (("curl", FAKE_CURL), ("openclaw", FAKE_DEMO_OPENCLAW), ("python", FAKE_DEMO_PYTHON)):
        (bindir / name).write_text(f"#!{sys.executable}\n{text}")
    (bindir / "ss").write_text("#!/bin/sh\necho 'LISTEN 0 4096 127.0.0.1:18789 0.0.0.0:*'\n")
    for path in bindir.iterdir():
        path.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CASCADE_", "CUDA_")) and k not in KEY_VARS
           and k not in {"DISPLAY", "PYTHONPATH", "PY"}}
    env.update(PATH=str(bindir) + os.pathsep + os.environ["PATH"], HOME=str(home), PY=str(bindir / "python"),
               REAL_PY=sys.executable, CURL_LOG=str(tmp_path / "curl.jsonl"), OC_ENV_LOG=str(tmp_path / "oc.jsonl"),
               DEMO_KEY=KEY, DEMO_BRAIN="keyed")
    return {"env": env, "home": home, "tmp": tmp_path}


def _demo(d, brain="cosmos-sglang", **extra):
    p = subprocess.run(["bash", str(REPO / "scripts" / "openclaw_demo.sh"), "--brain", brain],
                       env={**d["env"], **extra}, cwd=d["tmp"], capture_output=True, text=True, timeout=120)
    logs = {}
    for name in ("CURL_LOG", "OC_ENV_LOG"):
        log = Path(d["env"][name])
        logs[name] = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return p, logs["CURL_LOG"], logs["OC_ENV_LOG"]


@pytest.mark.parametrize("named", ["default key file", "CASCADE_MODEL_API_KEY_FILE"])
def test_openclaw_demo_probes_and_onboards_the_keyed_brain_with_its_key_file(demo_host, named):
    d = demo_host
    extra = {}
    if named == "default key file":
        _write_key(d["home"] / DEFAULT_KEY)
    else:
        extra["CASCADE_MODEL_API_KEY_FILE"] = str(_write_key(d["tmp"] / "model.key"))
    p, curls, ocs = _demo(d, **extra)
    assert p.returncode == 0, p.stdout + p.stderr
    probes = [c for c in curls if c["argv"][-1] == "http://127.0.0.1:8083/v1/models"]
    assert probes and all(f"Authorization: Bearer {KEY}" in c["headers"] for c in probes), probes
    onboard = [c for c in ocs if c["argv"][:1] == ["onboard"]]
    assert len(onboard) == 1 and onboard[0]["CUSTOM_API_KEY"] == KEY
    assert all(c["CUSTOM_API_KEY"] is None for c in ocs if c is not onboard[0])
    assert KEY not in json.dumps([c["argv"] for c in [*curls, *ocs]]), "the key is on no argv"


@pytest.mark.parametrize("problem, says", [
    ("no key file", "API_KEY_FILE"),
    ("wrong key", "refused the key"),
    ("no server", "start scripts/serve_cosmos_sglang.sh"),
])
def test_openclaw_demo_names_what_is_missing_for_the_keyed_brain(demo_host, problem, says):
    d = demo_host
    if problem != "no key file":
        _write_key(d["home"] / DEFAULT_KEY, "another-key\n" if problem == "wrong key" else KEY + "\n")
    p, _, ocs = _demo(d, DEMO_BRAIN="none" if problem == "no server" else "keyed")
    assert p.returncode != 0
    assert says in p.stdout + p.stderr, p.stdout + p.stderr
    assert "serve_cosmos-sglang" not in p.stdout + p.stderr
    assert not [c for c in ocs if c["argv"][:1] == ["onboard"]]


def test_openclaw_demo_keeps_the_vllm_cosmos_probe_and_onboarding(demo_host):
    d = demo_host
    _write_key(d["home"] / DEFAULT_KEY)
    p, curls, ocs = _demo(d, brain="cosmos", DEMO_BRAIN="open")
    assert p.returncode == 0, p.stdout + p.stderr
    assert curls[0]["argv"] == ["-sf", "-m", "5", "http://127.0.0.1:8082/v1/models"]
    assert all(c["CUSTOM_API_KEY"] is None for c in ocs)


# ── demo_proof --check-brain ───────────────────────────────────────────────


def _proof():
    sys.path.insert(0, str(REPO / "scripts"))
    import demo_proof
    return demo_proof


@pytest.mark.parametrize("named", ["default key file", "CASCADE_BRAIN_KEY_FILE", "CASCADE_MODEL_API_KEY_FILE"])
def test_check_brain_cosmos_sglang_sends_the_brain_key_file(model_http_boundary, monkeypatch, tmp_path, named):
    demo_proof = _proof()
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CASCADE_COSMOS_SGLANG_BASE_URL", model_http_boundary["base_url"])
    if named == "default key file":
        _write_key(tmp_path / DEFAULT_KEY)
    else:
        monkeypatch.setenv(named, str(_write_key(tmp_path / "other.key")))
    info = demo_proof.check_brain("cosmos-sglang")
    assert (info["brain"], info["model"]) == ("cosmos-sglang", "cosmos3-edge")
    assert [a[2] for a in model_http_boundary["auth"]] == [f"Bearer {KEY}"]


def test_check_brain_cosmos_sglang_without_a_key_file_says_which_file(model_http_boundary, monkeypatch, tmp_path):
    demo_proof = _proof()
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CASCADE_COSMOS_SGLANG_BASE_URL", model_http_boundary["base_url"])
    with pytest.raises(demo_proof.ProofError) as refused:
        demo_proof.check_brain("cosmos-sglang")
    assert "HTTP 401" in str(refused.value) and "API_KEY_FILE" in str(refused.value)
    assert str(DEFAULT_KEY) in str(refused.value)


def test_check_brain_vllm_cosmos_still_sends_no_key(model_http_boundary, monkeypatch, tmp_path):
    demo_proof = _proof()
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_key(tmp_path / DEFAULT_KEY)
    model_http_boundary["require_key"] = False
    monkeypatch.setenv("CASCADE_COSMOS_BASE_URL", model_http_boundary["base_url"])
    assert demo_proof.check_brain("cosmos")["brain"] == "cosmos"
    assert [a[2] for a in model_http_boundary["auth"]] == [None]


# ── end to end: the real serve script, then the launcher, with only the brain key file ──

#: The venv's python: `-` heredocs (transformers preflight, modeling patch, export) are skipped; `-c` runs the serve
#: script's launcher for real, with the fake packages (fake-site, beside bin/) first on sys.path.
FAKE_VENV_PYTHON = """
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fake-site"))
args = sys.argv[1:]
if args[:1] == ["-"]:
    sys.stdin.read()
    raise SystemExit(0)
if args[:1] != ["-c"]:
    raise SystemExit(f"fake python: unexpected {args}")
sys.argv = ["-c", *args[2:]]
exec(compile(args[1], "<string>", "exec"), {"__name__": "__main__"})
"""
#: `sglang.launch_server` that serves what the brain checks and the agent call, on --host/--port, keyed with the
#: --api-key it was given (SGLang 0.5.22: HTTP routes under /v1 need `Authorization: Bearer <key>`).
SERVING_LAUNCH_SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
if __name__ == "__main__":
    a = sys.argv[1:]
    flag = lambda name: a[a.index(name) + 1]
    key, served = flag("--api-key"), flag("--served-model-name")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code, payload):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def _keyed(self):
            if self.headers.get("Authorization") == "Bearer " + key:
                return True
            self._reply(401, {"error": "Unauthorized"})
            return False

        def do_GET(self):
            if self._keyed():
                self._reply(200, {"data": [{"id": served}]})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self._keyed():
                self._reply(200, {"id": "x", "object": "chat.completion", "created": 0, "model": served,
                                  "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                                      "role": "assistant", "content": "", "tool_calls": [{
                                          "id": "call-1", "type": "function", "function": {
                                              "name": "cascade__world_state", "arguments": "{}"}}]}}]})

    ThreadingHTTPServer((flag("--host"), int(flag("--port"))), H).serve_forever()
"""


def _fake_venv(venv: Path) -> Path:
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(f"#!{sys.executable}\n{FAKE_VENV_PYTHON}")
    (venv / "bin" / "python").chmod(0o755)
    site = venv / "fake-site"
    (site / "sglang").mkdir(parents=True)
    (site / "sglang" / "__init__.py").write_text("")
    (site / "sglang" / "launch_server.py").write_text(SERVING_LAUNCH_SERVER)
    for name, version in (("sglang", "0.5.22"), ("transformers", "5.19.0")):
        dist = site / f"{name}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return venv


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served_cosmos_sglang(tmp_path):
    """scripts/serve_cosmos_sglang.sh as the docs start it (API_KEY_FILE = the brain key file), on a fake venv."""
    home = tmp_path / "home"
    key_file = _write_key(home / DEFAULT_KEY)
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text("{}")
    port = _free_port()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "VENV": str(_fake_venv(tmp_path / "venv")),
           "EXPORT_DIR": str(export), "PORT": str(port), "API_KEY_FILE": "~/" + str(DEFAULT_KEY)}
    server = subprocess.Popen(["bash", str(REPO / "scripts" / "serve_cosmos_sglang.sh")], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    try:
        for _ in range(200):   # bounded: 20 s
            if server.poll() is not None:
                pytest.fail(f"serve_cosmos_sglang.sh exited: {server.stderr.read().decode()}")
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        else:
            pytest.fail("the fake SGLang server never listened")
        yield {"home": home, "key_file": key_file, "base_url": f"http://127.0.0.1:{port}/v1", "pid": server.pid}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        server.stderr.close()


def test_the_served_keyed_brain_takes_the_launch_with_no_key_variable(served_cosmos_sglang, linux_launcher):
    """The owner's commands, unchanged: `API_KEY_FILE=~/.cascade/nemoclaw/llama.key serve_cosmos_sglang.sh`, then
    `launch.sh --brain cosmos-sglang`. The brain checks, the native tool probe and the agent's brain provider all
    hold the key the server was started with; the server's command line never shows it."""
    s, h = served_cosmos_sglang, linux_launcher
    with open(f"/proc/{s['pid']}/cmdline", "rb") as f:
        cmdline = f.read().decode()
    assert KEY not in cmdline and "--log-level\0warning" in cmdline
    env = {k: v for k, v in h["env"].items() if k not in KEY_VARS}
    env.update(HOME=str(s["home"]), CASCADE_COSMOS_SGLANG_BASE_URL=s["base_url"])
    p = _launch(h, env)
    assert p.returncode == 0, p.stdout + p.stderr
    (up,) = [c[2:] for c in _rig(h) if c[2] == "up"]
    assert Path(up[up.index("--brain-key-file") + 1]) == s["key_file"]
    assert up[up.index("--brain-port") + 1] == s["base_url"].split(":")[-1].split("/")[0]
    assert KEY not in _argv_logs(h)


def test_the_served_keyed_brain_answers_the_cosmos3_profile_and_the_brain_check(served_cosmos_sglang, monkeypatch):
    """`--llm local_cosmos_sglang` (and the grounding profile) read the same key file through api_key_file."""
    from cascade.agent.llm import make_llm
    from cascade.config import load_profile

    s = served_cosmos_sglang
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(s["home"]))
    cfg = load_profile("llm", "local_cosmos_sglang")
    cfg._data["base_url"] = s["base_url"]   # the profile's :8083, moved to the test's port
    reply = make_llm(cfg).chat("Read-only test.", [{"role": "user", "content": "What is in the world model?"}])
    assert [c.name for c in reply.tool_calls] == ["cascade__world_state"]
    monkeypatch.setenv("CASCADE_COSMOS_SGLANG_BASE_URL", s["base_url"])
    assert _proof().check_brain("cosmos-sglang")["model"] == "cosmos3-edge"
