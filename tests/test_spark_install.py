"""Installer subprocess boundary tests, NOT evidence of Isaac/CUDA execution.

The real shell entry point runs with a private HOME; external installers/services
are replaced only in the tests that explicitly use boundary doubles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from conftest import start_sleeping_process


def boundary_env(tmp_path):
    """Fake package installation / network only; never installs GPU code."""
    env = host_env(tmp_path)
    bins = tmp_path / "bin"
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "scripts").mkdir()
    for name in (
        "install.sh",
        "install_isaac.sh",
        "install_support.py",
        "kitchen_assets.py",
        "fetch_robot_assets.py",
        "desktop.py",
    ):
        if (ROOT / "scripts" / name).exists():
            shutil.copy2(ROOT / "scripts" / name, source / "scripts" / name)
    (source / "pyproject.toml").write_text(
        '[project]\nname="boundary-double"\nversion="0.0.0"\n'
    )
    (source / ".gitignore").write_text(
        ".venv/\n.isaacsim/\n.llama.cpp/\nmodels/qwen3.8-27b/\nruns/\nmodels/mobileclip_blt.ts\n"
    )
    subprocess.run(
        ["/usr/bin/git", "init", "-b", "main", str(source)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["/usr/bin/git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(source),
            "-c",
            "user.name=Boundary",
            "-c",
            "user.email=boundary@example.invalid",
            "commit",
            "-m",
            "Fixture",
        ],
        check=True,
        capture_output=True,
    )
    log = tmp_path / "commands.jsonl"
    wrapper = f'#!{sys.executable}\nimport os,sys,json,pathlib,subprocess\na=sys.argv[1:]\nwith open(os.environ["BOUNDARY_LOG"],"a") as f: f.write(json.dumps([pathlib.Path(sys.argv[0]).name,*a])+"\\n")\n'
    executable(
        bins / "git",
        wrapper
        + 'a=[os.environ["BOUNDARY_SOURCE"] if x=="https://github.com/johnnynunez/cascade.git" else x for x in a]\nsys.exit(subprocess.call(["/usr/bin/git",*a]))\n',
    )
    python_double = (
        wrapper
        + """
if a and a[0] == "--version": print("Python 3.12.14")
elif len(a)>1 and a[0]=="-c" and "version_info" in a[1]: print("3.12")
elif a and a[0].endswith("/desktop.py") and os.environ.get("BOUNDARY_REAL_DESKTOP") == "1":
    sys.exit(subprocess.call([sys.executable, *a]))
sys.exit(int(os.environ.get("BOUNDARY_PY_FAIL", "0")))
"""
    )
    double = executable(tmp_path / "python-double", python_double)
    executable(
        bins / "uv",
        wrapper
        + """
if a[0]=="venv":
    dest=pathlib.Path(a[-1]); (dest/"bin").mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(os.environ["BOUNDARY_PYTHON"],dest/"bin/python")
sys.exit(int(os.environ.get("BOUNDARY_UV_FAIL","0")))
""",
    )
    env.update(
        {
            "BOUNDARY_SOURCE": str(source),
            "BOUNDARY_LOG": str(log),
            "BOUNDARY_PYTHON": str(double),
        }
    )
    installer = (
        wrapper
        + """
prefix=pathlib.Path(a[a.index("--prefix")+1]); (prefix/"bin").mkdir(parents=True,exist_ok=True)
p=prefix/"bin/openclaw"; p.write_text("#!/bin/bash\\nprintf 'OpenClaw 2026.9.3\\\\n'\\n"); p.chmod(0o755)
"""
    )
    import shlex

    installer = (
        f'#!/bin/bash\nmain() {{ {shlex.quote(sys.executable)} -c {shlex.quote(installer)} "$@"; }}\n'
        '[[ "${OPENCLAW_INSTALL_CLI_SH_NO_RUN:-0}" == 1 ]] || main "$@"\n'
    )
    executable(
        bins / "curl",
        wrapper
        + f'content={installer!r}\nif any("raw.githubusercontent.com/johnnynunez/cascade/" in x for x in a): content=pathlib.Path(os.environ["BOUNDARY_SOURCE"],"scripts/install.sh").read_text()\nif "-o" in a: pathlib.Path(a[a.index("-o")+1]).write_text(content)\nelse: print(content)\n',
    )
    executable(
        source / "scripts/serve_qwen_llamacpp.sh",
        wrapper
        + """
if a != ["--setup-only"]: sys.exit(77)
v=pathlib.Path(os.environ["MODEL_ROOT"]) / ".llama.cpp"; (v/"build/bin").mkdir(parents=True,exist_ok=True); (v/"build/bin/llama-server").touch()
""",
    )
    executable(source / "scripts/launch.sh", wrapper + "sys.exit(79)\n")
    subprocess.run(["/usr/bin/git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(source),
            "-c",
            "user.name=Boundary",
            "-c",
            "user.email=boundary@example.invalid",
            "commit",
            "-m",
            "Service boundary doubles",
        ],
        check=True,
        capture_output=True,
    )
    return env, log


def commands(log):
    return (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )


@pytest.mark.parametrize("entry", ["install.sh", "bootstrap.sh"])
def test_ci_clones_selected_ref_into_private_python_without_services(tmp_path, entry):
    env, log = boundary_env(tmp_path)
    target = tmp_path / "home" / "cascade checkout"
    result = run_stdin(
        tmp_path,
        "--profile",
        "ci",
        "--dir",
        str(target),
        "--ref",
        "main",
        entry=entry,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    calls = commands(log)
    assert any(
        c[:2] == ["uv", "venv"] and "3.12" in c and c[-1] == str(target / ".venv")
        for c in calls
    )
    assert any(
        c[:3] == ["uv", "pip", "install"]
        and "--python" in c
        and str(target / ".venv/bin/python") in c
        and any("[dev,kinematics]" in x for x in c)
        for c in calls
    )
    assert not any("openclaw" in c[0] or "serve_qwen" in " ".join(c) for c in calls)
    assert "READY" not in result.stdout
    assert (target / ".git").is_dir()


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def host_env(
    tmp_path,
    *,
    system="Linux",
    arch="aarch64",
    glibc="glibc 2.39",
    gpu="NVIDIA GB10, 580.95.05",
):
    bins = tmp_path / "bin"
    executable(
        bins / "uname",
        f'#!/bin/bash\ncase "$1" in -s) printf "%s\\n" "{system}";; -m) printf "%s\\n" "{arch}";; esac\n',
    )
    executable(bins / "getconf", f'#!/bin/bash\nprintf "%s\\n" "{glibc}"\n')
    executable(bins / "nvidia-smi", f'#!/bin/bash\nprintf "%s\\n" "{gpu}"\n')
    for name in ("curl", "git", "uv", "sudo", "apt-get", "openclaw"):
        executable(
            bins / name,
            '#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$HOME/forbidden"\nexit 91\n',
        )
    return {"PATH": str(bins) + ":/usr/bin:/bin"}


@pytest.mark.parametrize(
    "arguments,host,reason",
    [
        ([], {}, "--accept-eula"),
        (["--accept-eula"], {"system": "Darwin", "arch": "arm64"}, "Linux aarch64"),
        (["--accept-eula"], {"arch": "x86_64"}, "Linux aarch64"),
        (["--accept-eula"], {"glibc": "glibc 2.31"}, "glibc"),
        (["--accept-eula"], {"gpu": ""}, "NVIDIA"),
    ],
)
def test_rejects_unsafe_spark_install_before_mutation(
    tmp_path, arguments, host, reason
):
    result = run_stdin(tmp_path, *arguments, env=host_env(tmp_path, **host))
    assert result.returncode != 0
    assert reason in result.stderr
    assert not list((tmp_path / "home").iterdir())


ROOT = Path(__file__).resolve().parents[1]


def run_stdin(tmp_path, *args, entry="install.sh", env=None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    base = {"HOME": str(home), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    base.update(env or {})
    return subprocess.run(
        ["/bin/bash", "-s", "--", *args],
        input=(ROOT / "scripts" / entry).read_text(),
        cwd=tmp_path,
        env=base,
        text=True,
        capture_output=True,
        timeout=90,
    )


def test_rerun_reuses_venv_and_dirty_checkout_without_fetch(tmp_path):
    env, log = boundary_env(tmp_path)
    target = tmp_path / "home/cascade"
    first = run_stdin(tmp_path, "--profile", "ci", env=env)
    assert first.returncode == 0, first.stderr
    (target / "pyproject.toml").write_text("# user modification\n")
    (target / "user.txt").write_text("untracked user data\n")
    start = len(commands(log))
    again = run_stdin(tmp_path, "--profile", "ci", env=env)
    assert again.returncode == 0, again.stderr
    assert (target / "pyproject.toml").read_text() == "# user modification\n"
    calls = commands(log)[start:]
    assert not any(
        "fetch" in c
        or "checkout" in c
        or "pull" in c
        or "reset" in c
        or "clone" in c
        or "venv" in c
        for c in calls
    )
    assert any("record" in c and "existing checkout" in c for c in calls)
    assert (target / "user.txt").read_text() == "untracked user data\n"


def test_requested_different_ref_refuses_dirty_sources(tmp_path):
    env, log = boundary_env(tmp_path)
    assert run_stdin(tmp_path, "--profile", "ci", env=env).returncode == 0
    target = tmp_path / "home/cascade"
    (target / "pyproject.toml").write_text("# user edit\n")
    start = len(commands(log))
    result = run_stdin(tmp_path, "--profile", "ci", "--ref", "other-release", env=env)
    assert result.returncode != 0
    assert "dirty" in result.stderr
    assert (target / "pyproject.toml").read_text() == "# user edit\n"
    assert not any("fetch" in c or c[0] == "uv" for c in commands(log)[start:])


def test_explicit_ref_change_does_not_treat_managed_cli_as_dirty_source(tmp_path):
    env, _ = boundary_env(tmp_path)
    upstream = Path(env["BOUNDARY_SOURCE"])
    subprocess.run(
        ["/usr/bin/git", "-C", str(upstream), "checkout", "-b", "other"],
        check=True,
        capture_output=True,
    )
    (upstream / "release.txt").write_text("other release\n")
    subprocess.run(
        ["/usr/bin/git", "-C", str(upstream), "add", "release.txt"], check=True
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(upstream),
            "-c",
            "user.name=Boundary",
            "-c",
            "user.email=boundary@example.invalid",
            "commit",
            "-m",
            "Other fixture ref",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(upstream), "checkout", "main"],
        check=True,
        capture_output=True,
    )
    assert run_stdin(tmp_path, "--profile", "ci", env=env).returncode == 0
    repo = tmp_path / "home/cascade"
    managed = repo / ".openclaw-cli/private.txt"
    managed.parent.mkdir()
    managed.write_text("retained managed data\n")
    result = run_stdin(tmp_path, "--profile", "ci", "--ref", "other", env=env)
    assert result.returncode == 0, result.stderr
    assert (repo / "release.txt").read_text() == "other release\n"
    assert managed.read_text() == "retained managed data\n"


def test_install_failure_propagates_and_never_claims_prepared(tmp_path):
    env, log = boundary_env(tmp_path)
    env["BOUNDARY_UV_FAIL"] = "42"
    result = run_stdin(tmp_path, "--profile", "ci", env=env)
    assert result.returncode == 42
    assert "retry" in result.stderr
    assert "installed" not in result.stdout.lower()
    assert "READY" not in result.stdout


def test_isaac_installer_pins_python_wheel_and_does_not_touch_cascade_env(tmp_path):
    assert (ROOT / "scripts/install_isaac.sh").exists(), (
        "dedicated Isaac installer is missing"
    )
    env, log = boundary_env(tmp_path)
    target = tmp_path / "isaac target"
    target.mkdir()
    result = run_stdin(
        tmp_path,
        "--dir",
        str(target),
        "--accept-eula",
        entry="install_isaac.sh",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    calls = commands(log)
    assert any(
        c[:2] == ["uv", "venv"] and "3.12" in c and c[-1] == str(target / ".isaacsim")
        for c in calls
    )
    wheel_calls = [c for c in calls if "isaacsim[all,extscache]==6.1.0.0" in c]
    assert len(wheel_calls) == 1
    assert "tinyobjloader==2.0.0rc13" in wheel_calls[0]
    assert str(target / ".isaacsim/bin/python") in wheel_calls[0]
    assert "https://pypi.nvidia.com" in wheel_calls[0]
    assert (
        "--index-strategy" in wheel_calls[0] and "unsafe-best-match" in wheel_calls[0]
    )
    assert any("libgomp.so.1" in " ".join(c) for c in calls)
    assert any("torch==2.11.0+cu130" in c for c in calls), "Isaac6.1 pins torch2.11"
    assert not (target / ".venv").exists()
    assert "READY" not in result.stdout


def test_prepare_only_installs_complete_spark_with_pinned_rootless_host(tmp_path):
    env, log = boundary_env(tmp_path)
    result = run_stdin(tmp_path, "--accept-eula", "--prepare-only", env=env)
    assert result.returncode == 0, result.stderr
    calls = commands(log)
    robot_prepare = next(i for i, call in enumerate(calls)
                         if "robot-assets" in call and any(value.endswith("install_support.py") for value in call))
    first_runtime_install = next(i for i, call in enumerate(calls)
                                 if "torch==2.14.0+cu130" in call)
    assert robot_prepare < first_runtime_install, "resolve robot LFS prerequisites before large runtime downloads"
    for spec in (
        "torch==2.14.0+cu130",
        "torchvision==0.29.0+cu130",
        "isaacsim[all,extscache]==6.1.0.0",
    ):
        assert any(spec in c for c in calls), spec
    assert any("torch.cuda.is_available" in " ".join(c) for c in calls)
    assert any("https://openclaw.ai/install-cli.sh" in c for c in calls)
    assert any(
        "--version" in c
        and "2026.9.3" in c
        and str(tmp_path / "home/cascade/.openclaw-cli") in c
        for c in calls
    )
    assert any(
        "--install-method" in c and "npm" in c and "--no-onboard" in c for c in calls
    )
    assert any(
        "serve_qwen_llamacpp.sh" in c[0] and c[1:] == ["--setup-only"] for c in calls
    )
    assert any(
        "assets" in c and any(x.endswith("install_support.py") for x in c)
        for c in calls
    )
    assert not any(
        "launch.sh" in " ".join(c) or "piper" in c or "h1" in c for c in calls
    )
    assert "PREPARED" in result.stdout
    assert "READY" not in result.stdout
    assert not list((tmp_path / "home/cascade/runs/.launch").glob("*.pid"))


def support_module():
    import importlib.util

    path = ROOT / "scripts/install_support.py"
    assert path.exists(), "installer stdlib helper is missing"
    spec = importlib.util.spec_from_file_location("install_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("failure", ["truncated", "http503"])
def test_model_asset_fetch_retries_truncation_then_reuses_complete_zip(
    tmp_path, monkeypatch, failure
):
    # urllib on macOS also sees system proxies; this test is loopback-only.
    monkeypatch.setenv("no_proxy", "*")
    import io
    import zipfile
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    support = support_module()
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        z.writestr("archive/data.pkl", "boundary double; NOT model weights")
    payload = data.getvalue()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if failure == "http503" and len(seen) == 1:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload[:10] if len(seen) == 1 else payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        dest = tmp_path / "models/encoder.ts"
        url = f"http://127.0.0.1:{server.server_port}/encoder.ts"
        support.ensure_model_asset(url, dest, len(payload))
        support.ensure_model_asset(url, dest, len(payload))
        assert dest.read_bytes() == payload
        assert len(seen) == 2
        assert not dest.with_suffix(".ts.part").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_spark_assets_manifest_contains_only_shipped_detector_and_encoder():
    support = support_module()
    assert set(support.MODEL_ASSETS) == {
        "yoloe-11s-seg.pt",
        "yoloe-11s-seg-pf.pt",
        "mobileclip_blt.ts",
    }
    assert support.MODEL_ASSETS["mobileclip_blt.ts"] == 599764649
    assert support.scene_problems(ROOT) == []


def fixture_model_health(repo):
    """Only the sleeping process in this fixture can satisfy readiness."""
    from cascade.apps.process_owner import process_identity

    try:
        pid = int((repo / "service.pid").read_text())
    except (OSError, ValueError):
        return False
    current = process_identity(pid)
    return current is not None and str(repo / "scripts") in current["command"]


@pytest.mark.parametrize("payload, expected", [
    (b'{"data":[{"id":"Qwen/Qwen3.8-27B"}]}', True),
    (b'{"data":[{"id":"another-model"}]}', "wrong model"),
    (b'{"data":[]}', "wrong model"),
    (b'[]', "wrong model"),
    (b'not json', False),
    (OSError("offline"), False),
])
def test_model_health_validates_response_without_using_a_proxy(monkeypatch, payload, expected):
    import io
    from types import SimpleNamespace

    support = support_module()
    calls = []

    def open_response(url, timeout):
        calls.append((url, timeout))
        if isinstance(payload, OSError):
            raise payload
        return io.BytesIO(payload)

    def opener(proxy_handler):
        assert proxy_handler.proxies == {}
        return SimpleNamespace(open=open_response)

    monkeypatch.setattr(support.urllib.request, "build_opener", opener)
    if isinstance(expected, str):
        with pytest.raises(RuntimeError, match=expected):
            support.model_health()
    else:
        assert support.model_health() is expected
    assert calls == [(f"http://127.0.0.1:{support.QWEN_PORT}/v1/models", 2)]


def launch_fixture(tmp_path, monkeypatch):
    import socket

    monkeypatch.delenv("CASCADE_LAUNCH_STATE", raising=False)
    monkeypatch.delenv("CASCADE_OPENCLAW_PROFILE", raising=False)
    monkeypatch.delenv("ISAACSIM_PATH", raising=False)
    monkeypatch.delenv("ISAACSIM_PYTHON_EXE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    support = support_module()
    assert hasattr(support, "launch"), "installer launch supervision is missing"
    monkeypatch.setattr(support, "kitchen_problems", lambda repo: [])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setattr(support, "QWEN_PORT", port)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                 "models/qwen3.8-27b/mmproj-BF16.gguf", ".llama.cpp/build/bin/llama-server"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"process ownership boundary fixture")
    consent = repo / "runs/.install/install.json"
    consent.parent.mkdir(parents=True)
    consent.write_text(json.dumps({"repo": str(repo.resolve()), "eula_accepted": True,
                                   "eula_url": support.EULA_URL}))  # boundary operator consent
    # These tests exercise process ownership, not an HTTP/model server.
    executable(
        repo / "scripts/serve_qwen_llamacpp.sh",
        f"#!{sys.executable}\n"
        + """
import os,sys,json,pathlib,time
pathlib.Path("service-env.json").write_text(json.dumps({k:os.environ.get(k) for k in ("MODEL_ROOT","PORT","CTX","PY")}))
if os.environ.get("BOUNDARY_DIE"): sys.exit(47)
pathlib.Path("service.pid").write_text(str(os.getpid()))
time.sleep(60)
""",
    )
    monkeypatch.setattr(support, "model_health", lambda: fixture_model_health(repo))
    launcher = """
import json,os,pathlib,sys
pathlib.Path("launch-record.json").write_text(json.dumps({"args":sys.argv[1:], "env":{k:os.environ.get(k) for k in ("ISAACSIM_PYTHON_EXE","CASCADE_OPENCLAW_PROFILE","PY")}}))
sys.exit(int(os.environ.get("BOUNDARY_LAUNCH_FAIL","0")))
"""
    import shlex

    executable(
        repo / "scripts/launch.sh",
        f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -c {shlex.quote(launcher)} "$@"\n',
    )
    # Other workers can be running the full physics suite concurrently. This
    # bounds startup without treating scheduling latency as a service failure.
    monkeypatch.setenv("CASCADE_MODEL_WAIT_S", "30")
    real_popen = subprocess.Popen
    setattr(support, "_fixture_qwen", [])

    def spawn(command, *args, **kwargs):
        process = real_popen(command, *args, **kwargs)
        if command[0] == str(repo / "scripts/serve_qwen_llamacpp.sh"):
            support._fixture_qwen.append(process)
        return process

    monkeypatch.setattr(support.subprocess, "Popen", spawn)
    return support, repo


@pytest.mark.parametrize("final_exec", [False, True])
@pytest.mark.parametrize("custom_state", [False, True])
def test_launch_hands_exact_env_to_launcher_and_records_owned_qwen(
    tmp_path, monkeypatch, custom_state, final_exec
):
    from cascade.apps.process_owner import live_records, load_owner, stop_owned

    support, repo = launch_fixture(tmp_path, monkeypatch)
    final = repo / "scripts/final_service.py"
    if final_exec:
        service = repo / "scripts/serve_qwen_llamacpp.sh"
        final.write_text(service.read_text())
        executable(service, f"#!{sys.executable}\nimport os,sys,time\n"
                   f"time.sleep(0.3)\nos.execv(sys.executable, [sys.executable, {str(final)!r}])\n")
    root = tmp_path / "custom state" if custom_state else repo / "runs/.launch"
    if custom_state:
        monkeypatch.setenv("CASCADE_LAUNCH_STATE", str(root))
    state = root / "profile-cascade-demo"
    try:
        assert support.launch(repo, "spark", "qwen", no_open=True) == 0
        record = json.loads((repo / "launch-record.json").read_text())
        assert record["args"] == ["--sim", "isaac", "--brain", "qwen", "--no-open"]
        assert record["env"] == {
            "ISAACSIM_PYTHON_EXE": str(repo / ".isaacsim/bin/python"),
            "CASCADE_OPENCLAW_PROFILE": "cascade-demo",
            "PY": str(repo / ".venv/bin/python"),
        }
        assert (state / "qwen.pid").exists(), "installer did not use the launcher's profile state"
        assert (state / "qwen.log").exists()
        assert not (root / "qwen.pid").exists()
        owner = load_owner(state, repo, "cascade-demo")
        assert owner is not None
        receipt = json.loads((state / "qwen.pid").read_text())
        assert receipt["pid"] == int((repo / "service.pid").read_text())
        assert receipt["role"] == "qwen"
        assert receipt["profile"] == "cascade-demo"
        assert receipt["repo"] == str(repo.resolve())
        if final_exec:
            assert str(final) in receipt["command"]
        assert live_records(state, owner, role="qwen") == [receipt]
        service_env = json.loads((repo / "service-env.json").read_text())
        assert service_env == {
            "MODEL_ROOT": str(repo),
            "PORT": str(support.QWEN_PORT),
            "CTX": "32768",
            "PY": str(repo / ".venv/bin/python"),
        }
        # Exercise the exact shutdown consumer, not only the record producer.
        assert stop_owned(state, owner) == [receipt["pid"]]
        assert live_records(state, owner, role="qwen") == []
    finally:
        _stop_fixture_qwen(support)


def _stop_fixture_qwen(support):
    """Reap the real child; a second killpg on an exiting Mac group can EPERM."""
    for process in support._fixture_qwen:
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            support.stop_group(process)


def test_launch_finds_bootstrapped_uv_without_a_shell_profile(tmp_path, monkeypatch):
    support, repo = launch_fixture(tmp_path, monkeypatch)
    home = tmp_path / "shell-free-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", os.defpath)
    executable(home / ".local/bin/uv", "#!/bin/sh\nexit 0\n")
    service = repo / "scripts/serve_qwen_llamacpp.sh"
    shebang, _, body = service.read_text().partition("\n")
    executable(service, shebang + "\nimport subprocess\n"
               "subprocess.run(['uv', '--version'], check=True)\n" + body)
    try:
        assert support.launch(repo, "spark", "qwen", no_open=True) == 0
    finally:
        _stop_fixture_qwen(support)


def test_launch_reuses_healthy_qwen_instead_of_leaking_another_process(
    tmp_path, monkeypatch
):
    support, repo = launch_fixture(tmp_path, monkeypatch)
    state = repo / "runs/.launch/profile-cascade-demo/qwen.pid"
    try:
        assert support.launch(repo, "spark", "qwen", no_open=True) == 0
        receipt = json.loads(state.read_text())
        assert support.launch(repo, "spark", "qwen", no_open=True) == 0
        assert json.loads(state.read_text()) == receipt
    finally:
        _stop_fixture_qwen(support)


def test_unhealthy_owned_qwen_refuses_duplicate_without_touching_it(tmp_path, monkeypatch):
    from cascade.apps.process_owner import load_owner, register_process

    support, repo = launch_fixture(tmp_path, monkeypatch)
    state = repo / "runs/.launch/profile-cascade-demo"
    owner = load_owner(state, repo, "cascade-demo", create=True)
    assert owner is not None
    other = start_sleeping_process(start_new_session=True)
    try:
        receipt = register_process(state, owner, other.pid, "qwen")
        with pytest.raises(RuntimeError, match="recorded Qwen process.*unhealthy"):
            support.launch(repo, "spark", "qwen")
        assert support._fixture_qwen == []
        assert other.poll() is None
        assert json.loads((state / "qwen.pid").read_text()) == receipt
        assert not (repo / "launch-record.json").exists()
    finally:
        _stop_fixture_qwen(support)
        other.terminate()
        other.wait(timeout=5)


def test_borrowed_qwen_is_rejected_and_survives_other_owner_shutdown(tmp_path, monkeypatch):
    from cascade.apps.process_owner import load_owner, stop_owned

    support, repo = launch_fixture(tmp_path, monkeypatch)
    try:
        assert support.launch(repo, "spark", "qwen") == 0
        original = repo / "runs/.launch/profile-cascade-demo/qwen.pid"
        before = original.read_bytes()
        root = tmp_path / "second-owner"
        monkeypatch.setenv("CASCADE_LAUNCH_STATE", str(root))
        monkeypatch.setenv("BOUNDARY_LAUNCH_FAIL", "51")
        with pytest.raises(RuntimeError, match="outside this installation's ownership"):
            support.launch(repo, "spark", "qwen")
        state = root / "profile-cascade-demo"
        owner = load_owner(state, repo, "cascade-demo")
        assert owner is not None
        assert not (state / "qwen.pid").exists()
        assert original.read_bytes() == before
        assert stop_owned(state, owner) == []
        assert support.model_health() is True
        assert len(support._fixture_qwen) == 1
        assert support._fixture_qwen[0].poll() is None
    finally:
        _stop_fixture_qwen(support)


@pytest.mark.parametrize("change", ["selected_model", "model_contents", "server_contents"])
def test_healthy_owned_qwen_is_not_reused_after_model_or_server_change(tmp_path, monkeypatch, change):
    support, repo = launch_fixture(tmp_path, monkeypatch)
    try:
        assert support.launch(repo, "spark", "qwen") == 0
        original = repo / "runs/.launch/profile-cascade-demo/qwen.pid"
        before = original.read_bytes()
        if change == "selected_model":
            replacement = repo / "different-model.gguf"
            replacement.write_bytes(b"different selected weights")
            monkeypatch.setenv("CASCADE_QWEN_MODEL", str(replacement))
        else:
            relative = ("models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
                        if change == "model_contents" else ".llama.cpp/build/bin/llama-server")
            (repo / relative).write_bytes(b"different file after server startup")
        with pytest.raises(RuntimeError, match="different or changed model/server files"):
            support.launch(repo, "spark", "qwen")
        assert original.read_bytes() == before
        assert len(support._fixture_qwen) == 1
        assert support._fixture_qwen[0].poll() is None
    finally:
        _stop_fixture_qwen(support)


def test_failed_launch_preserves_a_replacement_qwen_receipt(tmp_path, monkeypatch):
    import shlex
    from cascade.apps.process_owner import load_owner, register_process

    support, repo = launch_fixture(tmp_path, monkeypatch)
    state = repo / "runs/.launch/profile-cascade-demo"
    owner = load_owner(state, repo, "cascade-demo", create=True)
    assert owner is not None
    other = start_sleeping_process(start_new_session=True)
    try:
        replacement = register_process(state, owner, other.pid, "qwen")
        marker = state / "qwen.pid"
        marker.unlink()  # the second invocation registers later, during proof
        body = (
            "import os,pathlib,sys; "
            f"pathlib.Path({str(marker)!r}).write_text(os.environ['BOUNDARY_REPLACEMENT']); "
            "sys.exit(51)"
        )
        executable(repo / "scripts/launch.sh",
                   f"#!/bin/bash\nexec {shlex.quote(sys.executable)} -c {shlex.quote(body)}\n")
        monkeypatch.setenv("BOUNDARY_REPLACEMENT", json.dumps(replacement))
        assert support.launch(repo, "spark", "qwen") == 51
        assert marker.is_file(), "failed invocation erased another process's receipt"
        assert json.loads(marker.read_text()) == replacement
        assert other.poll() is None
        assert all(p.poll() is not None for p in support._fixture_qwen)
    finally:
        _stop_fixture_qwen(support)
        other.terminate()
        other.wait(timeout=5)


def test_qwen_early_exit_is_detected_and_cleans_pid(tmp_path, monkeypatch):
    support, repo = launch_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("BOUNDARY_DIE", "1")
    with pytest.raises(RuntimeError, match="exited.*47"):
        support.launch(repo, "spark", "qwen", no_open=True)
    assert not (repo / "runs/.launch/profile-cascade-demo/qwen.pid").exists()
    assert not (repo / "launch-record.json").exists()


def test_launcher_failure_stops_owned_qwen(tmp_path, monkeypatch):
    from cascade.apps.process_owner import live_records, load_owner

    support, repo = launch_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("BOUNDARY_LAUNCH_FAIL", "51")
    assert support.launch(repo, "spark", "qwen", no_open=True) == 51
    state = repo / "runs/.launch/profile-cascade-demo"
    owner = load_owner(state, repo, "cascade-demo")
    assert owner is not None
    assert live_records(state, owner, role="qwen") == []
    assert all(p.poll() is not None for p in support._fixture_qwen)
    assert support.model_health() is False


@pytest.mark.parametrize("role", ["qwen", "launcher"])
def test_cleanup_kills_descendants_even_when_the_server_leader_exits(
    tmp_path, monkeypatch, role
):
    import time
    import shlex

    support, repo = launch_fixture(tmp_path, monkeypatch)
    server = repo / "scripts/serve_qwen_llamacpp.sh"
    child = 'import signal,time,os,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path("child.pid").write_text(str(os.getpid())); time.sleep(60)'
    spawn = (
        "import subprocess,time,sys,pathlib\n"
        + f'subprocess.Popen([sys.executable,"-c",{child!r}])\n'
        + 'while not pathlib.Path("child.pid").exists(): time.sleep(0.01)\n'
    )
    if role == "qwen":
        first, rest = server.read_text().split("\n", 1)
        server.write_text(first + "\n" + spawn + rest)
    else:
        # The launcher can fail AFTER starting a background sidecar. Killing
        # only a still-running launcher misses its already-orphaned workers.
        body = spawn + "sys.exit(51)\n"
        executable(repo / "scripts/launch.sh",
                   f"#!/bin/bash\nexec {shlex.quote(sys.executable)} -c {shlex.quote(body)}\n")
    monkeypatch.setenv("BOUNDARY_LAUNCH_FAIL", "51")
    pid = None
    try:
        assert support.launch(repo, "spark", "qwen") == 51
        pid = int((repo / "child.pid").read_text())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="], text=True, capture_output=True
            ).stdout.strip()
            if not state or "Z" in state:
                break
            time.sleep(0.05)
        assert not state or "Z" in state, (
            "descendant survived cleanup of its service leader"
        )
    finally:
        if pid is None and (repo / "child.pid").exists():
            pid = int((repo / "child.pid").read_text())
        if pid is not None:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


def test_sigterm_cleans_new_qwen_group_instead_of_orphaning_it(tmp_path, monkeypatch):
    import time

    support, repo = launch_fixture(tmp_path, monkeypatch)
    executable(repo / "scripts/launch.sh", "#!/bin/bash\nexec sleep 60\n")
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
        "sys.path.insert(0,sys.argv[4]); import install_support as s; "
        "from test_spark_install import fixture_model_health; "
        "s.model_health=lambda: fixture_model_health(Path(sys.argv[3])); "
        "s.kitchen_problems=lambda repo: []; "
        's.QWEN_PORT=int(sys.argv[2]); raise SystemExit(s.main(["launch","--repo",sys.argv[3]]))'
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(ROOT / "scripts"),
            str(support.QWEN_PORT),
            str(repo),
            str(ROOT / "tests"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    marker = repo / "runs/.launch/profile-cascade-demo/qwen.pid"
    pid = None
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.05)
        assert marker.exists()
        pid = json.loads(marker.read_text())["pid"]
        proc.terminate()
        assert proc.wait(timeout=15) != 0
        assert json.loads(marker.read_text())["pid"] == pid  # inert audit receipt
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if pid is not None:
            try:
                os.killpg(pid, 9)
            except ProcessLookupError:
                pass


def test_qwen_wait_budget_must_be_finite_before_spawning(tmp_path, monkeypatch):
    support, repo = launch_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("CASCADE_MODEL_WAIT_S", "nan")
    marker = repo / "runs/.launch/profile-cascade-demo/qwen.pid"
    try:
        with pytest.raises(ValueError, match="finite"):
            support.launch(repo, "spark", "qwen")
        assert not marker.exists()
    finally:
        _stop_fixture_qwen(support)


def test_invalid_existing_model_asset_is_not_overwritten(tmp_path):
    support = support_module()
    dest = tmp_path / "custom.pt"
    dest.write_text("user data, not an archive")
    with pytest.raises(RuntimeError, match="move it aside"):
        support.ensure_model_asset("https://example.invalid/no-network", dest, 42)
    assert dest.read_text() == "user data, not an archive"


def test_installer_delegates_launch_without_falling_back_to_hosted_brain(tmp_path):
    env, log = boundary_env(tmp_path)
    result = run_stdin(tmp_path, "--accept-eula", "--no-open", env=env)
    assert result.returncode == 0, result.stderr
    calls = commands(log)
    assert any(
        "launch" in c
        and "--brain" in c
        and "qwen" in c
        and "--no-open" in c
        and any(x.endswith("install_support.py") for x in c)
        for c in calls
    )
    assert "READY" not in result.stdout  # only the real proof launcher may emit it


def test_install_record_preserves_source_identity_and_reusable_environment(tmp_path, monkeypatch):
    support = support_module()
    assert hasattr(support, "record_install"), "installation identity is not recorded"
    env, _ = boundary_env(tmp_path)
    repo = Path(env["BOUNDARY_SOURCE"])
    (repo / "pyproject.toml").write_text("# dirty user source\n")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ISAACSIM_PATH", raising=False)
    monkeypatch.delenv("ISAACSIM_PYTHON_EXE", raising=False)
    support.record_install(repo, "spark", "qwen", "main", accept_eula=True)
    record = json.loads((repo / "runs/.install/install.json").read_text())
    actual = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert record["source_commit"] == actual
    assert record["requested_ref"] == "main"
    assert record["source_dirty"] is True
    assert record["gpu_validated"] is False
    assert (repo / "pyproject.toml").read_text() == "# dirty user source\n"
    shell = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1"; printf "%s\\n" "$ISAACSIM_PYTHON_EXE" "$CASCADE_OPENCLAW_PROFILE" "$OMNI_KIT_ACCEPT_EULA" "$PY" "$PATH"',
            "--",
            str(repo / "runs/.install/env.sh"),
        ],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    values = shell.stdout.splitlines()
    assert values[:4] == [
        str(repo / ".isaacsim/bin/python"),
        "cascade-demo",
        "YES",
        str(repo / ".venv/bin/python"),
    ]
    assert values[4].startswith(str(repo / ".openclaw-cli/bin") + ":")


def test_local_installer_defaults_to_its_checkout_not_home_clone(tmp_path):
    env, log = boundary_env(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    repo = Path(env["BOUNDARY_SOURCE"])
    result = subprocess.run(
        ["/bin/bash", str(repo / "scripts/install.sh"), "--profile", "ci"],
        cwd=tmp_path,
        env=dict(env, HOME=str(home), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (repo / ".venv/bin/python").exists()
    assert not (home / "cascade").exists()
    assert not any("clone" in c for c in commands(log))


def test_local_model_selection_survives_install_receipt_and_shell_launch(tmp_path, monkeypatch):
    support = support_module()
    env, _ = boundary_env(tmp_path)
    repo = Path(env["BOUNDARY_SOURCE"])
    selected = {
        "CASCADE_QWEN_MODEL": str(tmp_path / "existing models" / "Q4.gguf"),
        "CASCADE_QWEN_MMPROJ": str(tmp_path / "existing models" / "projector.gguf"),
        "LLAMA_SERVER": str(tmp_path / "existing server" / "llama-server"),
    }
    for key in support.MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in selected.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MODEL_UNRELATED_SECRET", "must-not-be-persisted")
    support.record_install(repo, "spark", "qwen", "main", accept_eula=True)
    for key in selected:
        monkeypatch.delenv(key)
    # A later repair launched from the desktop has no terminal model variables.
    support.record_install(repo, "spark", "qwen", "main")
    record = json.loads((repo / "runs/.install/install.json").read_text())
    assert record["model_environment"] == selected
    assert "must-not-be-persisted" not in json.dumps(record)
    result = subprocess.run(
        ["/bin/bash", "-c", 'source "$1"; exec "$2" -c "$3"', "--",
         str(repo / "runs/.install/env.sh"), sys.executable,
         "import json,os; print(json.dumps({k:os.environ[k] for k in " + repr(tuple(selected)) + "}))"],
        env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == selected
    replacement = str(tmp_path / "replacement Q4.gguf")
    monkeypatch.setenv("CASCADE_QWEN_MODEL", replacement)
    assert support.model_environment(repo)["CASCADE_QWEN_MODEL"] == replacement


def test_spark_default_does_not_discover_a_host_isaac_install(tmp_path, monkeypatch):
    support = support_module()
    home = tmp_path / "private-home"
    executable(home / "isaacsim/python.sh", "#!/bin/bash\necho forbidden-host-runtime\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ISAACSIM_PATH", raising=False)
    monkeypatch.delenv("ISAACSIM_PYTHON_EXE", raising=False)
    repo = tmp_path / "new-project"
    assert support.isaac_environment(repo) == {"ISAACSIM_PYTHON_EXE": str(repo / ".isaacsim/bin/python")}
    result = subprocess.run(["bash", str(ROOT / "scripts/install_isaac.sh"), "--dir", str(repo), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 3
    assert "forbidden-host-runtime" not in result.stdout
    assert str(repo / ".isaacsim/bin/python") in result.stderr
    assert not repo.exists()


def test_package_inventory_records_actual_versions_without_claiming_a_missing_runtime(tmp_path):
    support = support_module()
    inventory = support.package_inventory(Path(sys.executable))
    assert inventory["available"] is True
    assert inventory["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}.")
    assert inventory["packages"]["pytest"] == pytest.__version__
    missing = support.package_inventory(tmp_path / "missing-runtime")
    assert missing["available"] is False
    assert "packages" not in missing


@pytest.mark.parametrize("entry", ["install.sh", "bootstrap.sh"])
@pytest.mark.parametrize(
    "args, reason",
    [
        (["--dir"], "missing"),
        (["--profile", "unknown"], "profile"),
        (["--brain", "cosmos"], "brain"),
        (["--ref", "bad..ref", "--dry-run"], "ref"),
        (["--profile", "laptop", "--brain", "qwen", "--dry-run"], "qwen"),
    ],
)
def test_exact_argument_validation_is_non_mutating(tmp_path, entry, args, reason):
    result = run_stdin(tmp_path, *args, entry=entry, env=host_env(tmp_path))
    assert result.returncode != 0
    assert reason in result.stderr
    assert not list((tmp_path / "home").iterdir())


def test_check_reports_remaining_gaps_even_on_unsupported_host(tmp_path):
    result = run_stdin(
        tmp_path, "--check", env=host_env(tmp_path, system="Darwin", arch="arm64")
    )
    assert result.returncode != 0
    assert "Linux aarch64" in result.stderr
    assert "MISSING:" in result.stdout
    assert not list((tmp_path / "home").iterdir())


def test_spark_preflight_uses_the_resolvable_cuda_version_pair(tmp_path, monkeypatch):
    # Package-boundary contract, not an imported CUDA runtime. This pair was
    # resolved for CPython3.12/Linux-aarch64; the app can use the latest pair
    # independently of Qwen and Isaac's more restrictive torch pins.
    helper = support_module()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").touch()
    seen = []

    def run(command, **kwargs):
        if "-c" in command:
            seen.append(json.loads(command[-1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(helper.subprocess, "run", run)
    monkeypatch.setattr(helper, "scene_problems", lambda repo: [])
    monkeypatch.setattr(helper, "kitchen_problems", lambda repo: [])
    helper.installation_problems(tmp_path, "spark", "keep")
    assert len(seen) == 1
    assert seen[0]["torch"] == "2.14.0+cu130"
    assert seen[0]["torchvision"] == "0.29.0+cu130"


def test_check_rejects_empty_python_environment_without_mutation(tmp_path):
    env, _ = boundary_env(tmp_path)
    repo = Path(env["BOUNDARY_SOURCE"])
    (repo / ".venv/bin").mkdir(parents=True)
    # A genuine interpreter, not a package-metadata double; no venv packages.
    (repo / ".venv/bin/python").symlink_to(Path(sys.executable).resolve())
    before = {
        str(p): (p.stat().st_mtime_ns, p.read_bytes())
        for p in repo.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    result = run_stdin(
        tmp_path, "--profile", "ci", "--check", "--dir", str(repo), env=env
    )
    assert result.returncode != 0
    assert "MISSING" in result.stdout + result.stderr
    after = {
        str(p): (p.stat().st_mtime_ns, p.read_bytes())
        for p in repo.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    assert before == after


@pytest.mark.parametrize("entry", ["install.sh", "bootstrap.sh"])
def test_check_reports_missing_spark_components_without_creating_anything(
    tmp_path, entry
):
    result = run_stdin(tmp_path, "--check", entry=entry, env=host_env(tmp_path))
    assert result.returncode != 0
    for missing in ("MISSING", ".venv", ".isaacsim", "llama-server", "OpenClaw", "source"):
        assert missing in result.stdout + result.stderr
    assert not list((tmp_path / "home").iterdir())


@pytest.mark.parametrize("entry", ["install.sh", "bootstrap.sh"])
def test_stdin_dry_run_defaults_to_complete_spark_without_mutation(tmp_path, entry):
    result = run_stdin(tmp_path, "--dry-run", entry=entry)
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for expected in (
        "profile=spark",
        "6.1.0.0",
        "3.12",
        "Qwen Q4 + vision projector",
        "OpenClaw",
        "2026.9.3",
        ".isaacsim",
        "models/qwen3.8-27b",
        ".venv",
    ):
        assert expected in output
    assert not list((tmp_path / "home").iterdir())
    assert "READY" not in output
