"""scripts/openshell_rig.py: OpenClaw in a standalone NVIDIA OpenShell 0.1.x sandbox.

The NemoClaw route (scripts/nemoclaw_mcp.py) pins OpenShell 0.0.116 and its
managed `inference.local`. OpenShell 0.1.x replaced that with provider
profiles: the sandbox only ever holds an opaque placeholder, and the
supervisor swaps in the real credential at the one endpoint the profile
authorizes, for the binaries it names. These tests pin the contract the
live spike on the Spark established (2026-10-11):

- a pinned, checksummed OpenShell, isolated from NemoClaw's 0.0.116 state;
- the brain and the robot reached only as host.openshell.internal endpoints
  (the supervisor maps that to the host's loopback on native Linux), with
  their secrets in env, never in argv;
- OpenShell itself refusing `reset_stop` at L7, whatever the agent tries;
- the two Landlock findings baked into the image (template readable under
  /usr, SQLite temp files in /tmp).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest
import tomllib

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "openshell_rig.py"
DEPLOY = REPO / "deploy" / "openshell"


def _load():
    spec = importlib.util.spec_from_file_location("openshell_rig", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


# ── pinned, verified OpenShell ────────────────────────────────────────────


@pytest.mark.parametrize("machine", ["aarch64", "x86_64"])
def test_every_artifact_is_pinned_to_a_sha256(machine):
    rig = _load()
    for kind in ("cli", "gateway"):
        name, digest = rig.artifact(machine, kind)
        assert machine in name and name.endswith(".tar.gz")
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_an_unknown_machine_is_refused():
    rig = _load()
    with pytest.raises(SystemExit, match="riscv64"):
        rig.artifact("riscv64", "cli")


def test_a_download_whose_digest_differs_is_refused(tmp_path):
    rig = _load()
    blob = tmp_path / "x.tar.gz"
    blob.write_bytes(b"not the release")
    good = hashlib.sha256(b"not the release").hexdigest()
    rig.verify_sha256(blob, good)  # matches: no error
    with pytest.raises(SystemExit, match="sha256"):
        rig.verify_sha256(blob, "0" * 64)


# ── isolation from NemoClaw's OpenShell 0.0.116 ───────────────────────────


def test_the_openshell_environment_never_points_at_the_shared_config(tmp_path):
    """NemoClaw's 0.0.116 CLI keeps its gateway registry in
    ~/.config/openshell; two CLI generations sharing it would overwrite each
    other's active gateway, so every path of this route lives under its own
    home."""
    rig = _load()
    home = tmp_path / "os-home"
    env = rig.openshell_env(home, base={"PATH": "/usr/bin", "HOME": "/home/x"})
    for key in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "OPENSHELL_LOCAL_TLS_DIR"):
        assert Path(env[key]).is_relative_to(home), key
    assert env["PATH"].split(":")[0] == str(home / rig.OPENSHELL_VERSION / "bin")
    assert env["PATH"].endswith("/usr/bin")


def _server_protocol_versions() -> tuple[str, ...]:
    """cascade's MCP server revisions, read without importing the runtime."""
    import ast

    src = (REPO / "src" / "cascade" / "apps" / "mcp_server.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROTOCOL_VERSIONS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("PROTOCOL_VERSIONS not found")


def test_brain_profile_binds_the_key_to_the_host_brain_only():
    rig = _load()
    p = rig.brain_profile(port=8123)
    assert p["category"] == "inference" and p["inference_capable"] is True
    (cred,) = p["credentials"]
    assert cred["env_vars"] == [rig.BRAIN_KEY_ENV] and cred["auth_style"] == "bearer"
    (ep,) = p["endpoints"]
    assert (ep["host"], ep["port"], ep["protocol"], ep["enforcement"]) == (
        "host.openshell.internal", 8123, "rest", "enforce")
    assert "/usr/local/bin/node" in p["binaries"], "OpenClaw runs on the image's node"


def test_robot_profile_admits_mcp_and_refuses_reset_stop_at_l7():
    rig = _load()
    p = rig.robot_profile(port=18999)
    (cred,) = p["credentials"]
    assert cred["env_vars"] == [rig.MCP_TOKEN_ENV] and cred["auth_style"] == "bearer"
    (ep,) = p["endpoints"]
    assert (ep["host"], ep["port"], ep["path"], ep["protocol"], ep["enforcement"]) == (
        "host.openshell.internal", 18999, "/mcp", "mcp", "enforce")
    assert ep["mcp"]["allow_all_known_mcp_methods"] is True
    assert {"method": "tools/call", "tool": "reset_stop"} in ep["deny_rules"]
    assert "/usr/local/bin/node" in p["binaries"]


def test_robot_profile_accepts_every_revision_the_handshake_uses():
    """OpenShell selects a request's MCP revision from its header and admits
    only the listed ones: `initialize` carries no header (OpenShell reads it
    as 2025-03-26), cascade answers one of its PROTOCOL_VERSIONS, and the
    OpenClaw SDK offers 2025-11-25 first. Missing any one 403s the session."""
    rig = _load()
    versions = set(rig.robot_profile(port=1)["endpoints"][0]["mcp"]["versions"])
    assert "2025-03-26" in versions
    assert "2025-11-25" in versions
    spoken = set(_server_protocol_versions()) & set(rig.OPENSHELL_MCP_REVISIONS)
    assert spoken and spoken <= versions
    assert versions <= set(rig.OPENSHELL_MCP_REVISIONS)


def test_extra_denied_tools_become_deny_rules():
    rig = _load()
    rules = rig.robot_profile(port=1, deny=("reset_stop", "throw_object"))["endpoints"][0]["deny_rules"]
    assert {"method": "tools/call", "tool": "throw_object"} in rules
    assert len(rules) == 2


# ── observe-only: OpenShell, not the prompt, keeps the arm still ──────────


def test_motion_tools_track_the_runtime_and_the_program_tier():
    """The observe-only deny list is read from the source of truth, so a new
    motion skill is locked the day it lands."""
    from cascade.skills.runtime import _MOTION_SKILLS

    rig = _load()
    assert rig.motion_tools() == frozenset(_MOTION_SKILLS) | {"run_program"}


def test_observe_only_denies_every_motion_tool_and_reset_stop():
    rig = _load()
    deny = rig.deny_set(observe_only=True)
    assert rig.motion_tools() <= set(deny)
    assert "reset_stop" in deny
    assert set(rig.deny_set(observe_only=False)) == {"reset_stop"}


@pytest.mark.parametrize("stop", ["emergency_stop", "halt_motion"])
def test_a_stop_tool_can_never_be_denied(stop):
    """Whatever the lock, the agent can always stop the arm."""
    rig = _load()
    assert stop not in rig.deny_set(observe_only=True)
    with pytest.raises(SystemExit, match=stop):
        rig.deny_set(observe_only=False, extra=(stop,))


def test_motion_lock_and_unlock_update_the_live_profile(tmp_path):
    """Profile updates reach a running sandbox in seconds (live, 0.1.3), so
    the lock toggles without recreating the agent or its chat token."""
    rig = _load()
    run = FakeRun(outs={"provider profile export": "resource_version: 3\n"})
    rig.set_motion_lock(tmp_path, rig.Settings(mcp_port=18999), locked=True, run=run)
    upd = [a for a in run.argvs() if a[1:4] == ["provider", "profile", "update"]]
    prof = json.loads(Path(upd[-1][upd[-1].index("--file") + 1]).read_text())
    denied = {r["tool"] for r in prof["endpoints"][0]["deny_rules"]}
    assert rig.motion_tools() <= denied and prof["endpoints"][0]["port"] == 18999
    assert rig.motion_locked(tmp_path) is True
    rig.set_motion_lock(tmp_path, rig.Settings(mcp_port=18999), locked=False, run=run)
    upd = [a for a in run.argvs() if a[1:4] == ["provider", "profile", "update"]]
    prof = json.loads(Path(upd[-1][upd[-1].index("--file") + 1]).read_text())
    assert {r["tool"] for r in prof["endpoints"][0]["deny_rules"]} == {"reset_stop"}
    assert rig.motion_locked(tmp_path) is False
    assert not [a for a in run.argvs() if a[1:3] == ["sandbox", "delete"]]


def test_gateway_config_is_schema_v2_docker_on_loopback():
    rig = _load()
    cfg = tomllib.loads(rig.gateway_toml())
    assert cfg["openshell"]["version"] == 2
    gw = cfg["openshell"]["gateway"]
    assert gw["compute_driver"] == "docker"
    assert gw.get("bind_address", "127.0.0.1").startswith("127.0.0.1")
    assert cfg["openshell"]["drivers"]["docker"]["sandbox_label"] == rig.SANDBOX_LABEL


# ── the sandbox image (deploy/openshell) ──────────────────────────────────


def _dockerfile() -> str:
    return (DEPLOY / "Dockerfile").read_text()


def test_image_pins_node_by_digest_and_openclaw_to_the_repo_pin():
    """OpenClaw 2026.9.3 needs node >= 24.16 (npm refused node 22 in the
    spike); the base is pinned by digest and OpenClaw to the version every
    other cascade route installs."""
    src = _dockerfile()
    assert re.search(r"^FROM node:24[\w.-]*@sha256:[0-9a-f]{64}\s*$", src, re.MULTILINE)
    pin = re.search(r'!= "(\d{4}\.\d+\.\d+)"', (REPO / "scripts" / "install_support.py").read_text()).group(1)
    assert re.search(rf"^ARG OPENCLAW_VERSION={re.escape(pin)}\s*$", src, re.MULTILINE)
    assert "openclaw@${OPENCLAW_VERSION}" in src


def test_image_survives_the_default_landlock_policy():
    """OpenShell's default filesystem policy lets the agent read /usr and
    write /tmp: a template under /opt was unreadable at start ("Permission
    denied", the sandbox went to phase Error) and SQLite's temp files failed
    until TMPDIR pointed at /tmp. Both are pinned here."""
    src = _dockerfile()
    assert "/opt/" not in src
    assert re.search(r"COPY .*openclaw\.json /usr/local/share/cascade/", src)
    assert re.search(r"COPY .*render-config\.mjs /usr/local/share/cascade/", src)
    assert re.search(r"TMPDIR=/tmp", src) and re.search(r"SQLITE_TMPDIR=/tmp", src)
    assert re.search(r"^USER node\s*$", src, re.MULTILINE), "the agent never runs as root"
    assert re.search(r"HOME=/sandbox", src)


def test_config_template_holds_placeholders_never_secrets():
    raw = (DEPLOY / "openclaw.json").read_text()
    cfg = json.loads(raw)
    prov = cfg["models"]["providers"]["cascade"]
    rig = _load()
    assert prov["apiKey"] == "${" + rig.BRAIN_KEY_ENV + "}"
    assert cfg["mcp"]["servers"]["cascade"]["headers"]["Authorization"] == "Bearer ${" + rig.MCP_TOKEN_ENV + "}"
    assert cfg["gateway"]["bind"] == "loopback"
    assert cfg["gateway"]["auth"]["token"] == "", "minted inside the sandbox at first start"
    assert not re.search(r"(sk-|[0-9a-f]{32,})", raw)


# ── render-config.mjs, run with the host's node ───────────────────────────


def _render(tmp_path, **env):
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node not on PATH")
    e = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""),
         "CASCADE_OPENCLAW_TEMPLATE": str(DEPLOY / "openclaw.json"), **env}
    return subprocess.run([node, str(DEPLOY / "render-config.mjs")], env=e,
                          capture_output=True, text=True, timeout=30, check=False)


def test_render_fills_ports_and_model_and_mints_a_private_token(tmp_path):
    r = _render(tmp_path, CASCADE_BRAIN_PORT="8123", CASCADE_MCP_PORT="18999",
                CASCADE_CHAT_PORT="18799", CASCADE_MODEL="qwen3.8-27b")
    assert r.returncode == 0, r.stderr
    out = tmp_path / ".openclaw" / "openclaw.json"
    assert out.stat().st_mode & 0o077 == 0, "the gateway token is a secret"
    cfg = json.loads(out.read_text())
    assert cfg["models"]["providers"]["cascade"]["baseUrl"] == "http://host.openshell.internal:8123/v1"
    assert cfg["mcp"]["servers"]["cascade"]["url"] == "http://host.openshell.internal:18999/mcp"
    assert cfg["agents"]["defaults"]["model"]["primary"] == "cascade/qwen3.8-27b"
    assert cfg["models"]["providers"]["cascade"]["models"][0]["id"] == "qwen3.8-27b"
    assert "http://127.0.0.1:18799" in cfg["gateway"]["controlUi"]["allowedOrigins"]
    assert re.fullmatch(r"[0-9a-f]{48}", cfg["gateway"]["auth"]["token"])
    assert cfg["models"]["providers"]["cascade"]["apiKey"] == "${CASCADE_BRAIN_API_KEY}"
    assert r.stdout.count(cfg["gateway"]["auth"]["token"]) == 0, "never echoed"


def test_render_keeps_an_existing_config(tmp_path):
    """A restart must not rotate the chat token out from under an open tab."""
    assert _render(tmp_path).returncode == 0
    out = tmp_path / ".openclaw" / "openclaw.json"
    first = out.read_text()
    assert _render(tmp_path, CASCADE_BRAIN_PORT="9000").returncode == 0
    assert out.read_text() == first


@pytest.mark.parametrize("env", [{"CASCADE_BRAIN_PORT": "80;rm"}, {"CASCADE_MCP_PORT": "70000"},
                                 {"CASCADE_MODEL": "Qwen/Qwen3.8-27B"}])
def test_render_refuses_values_that_would_break_the_config(tmp_path, env):
    """Ports must be ports, and the model id must not look like a path
    (OpenClaw splits provider/model on '/': docs/NEMOCLAW.md)."""
    r = _render(tmp_path, **env)
    assert r.returncode != 0
    assert not (tmp_path / ".openclaw" / "openclaw.json").exists()


# ── orchestration (a fake runner records every child) ─────────────────────


class FakeRun:
    """Stands in for subprocess.run: records (argv, env) and answers rc by
    the first matching argv prefix in `rcs` (default 0)."""

    def __init__(self, rcs=None, outs=None):
        self.calls = []
        self.rcs = rcs or {}
        self.outs = outs or {}

    def __call__(self, argv, env=None, **kw):
        import subprocess

        self.calls.append((list(argv), dict(env or {})))
        line = " ".join(argv[1:])
        rc = next((v for k, v in self.rcs.items() if line.startswith(k)), 0)
        out = next((v for k, v in self.outs.items() if line.startswith(k)), "")
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")

    def argvs(self):
        return [a for a, _ in self.calls]


def _secret(tmp_path, name="brain.key", value="s3cret-brain-key-value", mode=0o600):
    p = tmp_path / name
    p.write_text(value + "\n")
    p.chmod(mode)
    return p


def test_a_secret_file_must_be_private_and_non_empty(tmp_path):
    rig = _load()
    assert rig.read_secret(_secret(tmp_path)) == "s3cret-brain-key-value"
    with pytest.raises(SystemExit, match="0600"):
        rig.read_secret(_secret(tmp_path, "loose.key", mode=0o640))
    with pytest.raises(SystemExit, match="empty"):
        rig.read_secret(_secret(tmp_path, "empty.key", value=""))
    with pytest.raises(SystemExit, match="missing"):
        rig.read_secret(tmp_path / "nope.key")


def test_a_minted_token_is_private_and_kept(tmp_path):
    rig = _load()
    p = tmp_path / "sub" / "mcp.token"
    first = rig.ensure_token(p)
    assert len(first) >= 32 and p.stat().st_mode & 0o077 == 0
    assert rig.ensure_token(p) == first


def test_provider_secret_travels_in_the_child_env_never_argv(tmp_path):
    rig = _load()
    run = FakeRun(rcs={"provider get": 1})
    rig.sync_provider(tmp_path, "cascade-brain", rig.BRAIN_KEY_ENV, _secret(tmp_path), run=run)
    (argv, env), = [c for c in run.calls if c[0][1:3] == ["provider", "create"]]
    assert argv[1:] == ["provider", "create", "--name", "cascade-brain", "--type", "cascade-brain", "--from-existing"]
    assert env[rig.BRAIN_KEY_ENV] == "s3cret-brain-key-value"
    assert Path(env["XDG_CONFIG_HOME"]).is_relative_to(tmp_path)
    assert all("s3cret" not in " ".join(a) for a in run.argvs())


def test_an_existing_provider_is_updated_so_a_rotated_key_lands(tmp_path):
    rig = _load()
    run = FakeRun()  # provider get -> 0: it exists
    rig.sync_provider(tmp_path, "cascade-robot", rig.MCP_TOKEN_ENV, _secret(tmp_path), run=run)
    assert [a[1:] for a in run.argvs() if a[1:3] == ["provider", "update"]] == [
        ["provider", "update", "cascade-robot", "--from-existing"]]
    assert not [a for a in run.argvs() if a[1:3] == ["provider", "create"]]


def test_profiles_are_imported_once_then_updated(tmp_path):
    rig = _load()
    run = FakeRun(rcs={"provider profile export": 1})
    rig.sync_profile(tmp_path, rig.brain_profile(), run=run)
    imp = [a for a in run.argvs() if a[1:5] == ["provider", "profile", "import", "--file"]]
    assert imp and "resource_version" not in json.loads(Path(imp[0][5]).read_text())
    # update needs the stored profile's resource_version (live, 0.1.3:
    # "custom provider profile update requires a non-zero resource_version")
    run = FakeRun(outs={"provider profile export": "id: cascade-brain\nresource_version: 7\ncategory: inference\n"})
    rig.sync_profile(tmp_path, rig.brain_profile(), run=run)
    upd = [a for a in run.argvs() if a[1:4] == ["provider", "profile", "update"]]
    assert upd and json.loads(Path(upd[0][upd[0].index("--file") + 1]).read_text())["resource_version"] == 7
    assert any(a[1:4] == ["provider", "profile", "update"] and a[-1] == "cascade-brain" for a in run.argvs())
    assert all(a[1:4] != ["provider", "profile", "import"] for a in run.argvs())
    # `profile lint` reports an existing id as an error (live, 0.1.3): an
    # update must not lint first, or every second `up` fails.
    assert all(a[1:4] != ["provider", "profile", "lint"] for a in run.argvs())


def test_sandbox_gets_both_providers_and_only_non_secret_env():
    rig = _load()
    cfg = rig.Settings(brain_port=8123, mcp_port=18999, chat_port=18799, model="qwen3.8-27b")
    argv = rig.sandbox_create_argv(cfg, image="cascade/openclaw-sandbox:t")
    assert argv[:2] == ["sandbox", "create"]
    assert argv[argv.index("--from") + 1] == "cascade/openclaw-sandbox:t"
    provs = [argv[i + 1] for i, a in enumerate(argv) if a == "--provider"]
    assert provs == ["cascade-brain", "cascade-robot"]
    envs = {argv[i + 1] for i, a in enumerate(argv) if a == "--env"}
    assert envs == {"CASCADE_BRAIN_PORT=8123", "CASCADE_MCP_PORT=18999",
                    "CASCADE_CHAT_PORT=18799", "CASCADE_MODEL=qwen3.8-27b"}
    assert argv[argv.index("--restart-policy") + 1] == "on-failure"
    assert "--expose" not in argv, "the chat reaches the host only via a loopback forward"
    assert argv[-2:] == ["--", "cascade-openclaw"] and "--detach" in argv


def test_chat_forward_binds_the_host_loopback_only():
    rig = _load()
    argv = rig.forward_argv(rig.Settings(chat_port=18799))
    assert argv[:2] == ["forward", "service"]
    assert argv[argv.index("--target-port") + 1] == "18789"
    assert argv[argv.index("--local") + 1] == "127.0.0.1:18799"


def test_the_robot_mcp_server_is_served_on_loopback_with_a_token_file(tmp_path):
    rig = _load()
    argv = rig.serve_mcp_argv(rig.Settings(mcp_port=18999), token_file=tmp_path / "t", python="/py")
    assert argv[:3] == ["/py", "-m", "cascade.apps.mcp_server"]
    assert argv[argv.index("--http") + 1] == "127.0.0.1:18999"
    assert argv[argv.index("--token-file") + 1] == str(tmp_path / "t")


def test_the_mcp_server_runs_like_the_other_routes_run_it():
    """Same contract as nemoclaw_mcp.py serve and launch.sh: cwd models/
    (YOLOE resolves its text encoder relative to cwd, so from the repo root
    open-vocab detection fails offline), offline hub flags, and the token
    file as the only token source."""
    rig = _load()
    env = rig.serve_mcp_env({"CASCADE_MCP_TOKEN": "leak", "YOLO_OFFLINE": "False", "PATH": "/bin"})
    assert "CASCADE_MCP_TOKEN" not in env
    assert env["YOLO_OFFLINE"] == "False", "an explicit choice is kept"
    assert env["ULTRALYTICS_OFFLINE"] == "True"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO / "src")
    assert rig.serve_mcp_cwd() == REPO / "models"


def test_the_image_tag_follows_the_deploy_files(tmp_path):
    """`up` rebuilds exactly when deploy/openshell changed: the tag is the
    OpenClaw pin plus a hash of the build context."""
    import shutil

    rig = _load()
    ctx = tmp_path / "ctx"
    shutil.copytree(DEPLOY, ctx)
    tag = rig.image_tag(ctx)
    assert re.fullmatch(r"cascade/openclaw-sandbox:2026\.9\.3-[0-9a-f]{12}", tag)
    assert rig.image_tag(ctx) == tag
    (ctx / "openclaw.json").write_text((ctx / "openclaw.json").read_text().replace("65536", "32768"))
    assert rig.image_tag(ctx) != tag


def test_release_extraction_refuses_path_traversal(tmp_path):
    import io
    import tarfile

    rig = _load()
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tf:
        data = b"#!/bin/sh\n"
        ti = tarfile.TarInfo("../evil")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    with pytest.raises(SystemExit):
        rig.extract_binary(bad, "evil", tmp_path / "bin")
    assert not (tmp_path / "evil").exists()

    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as tf:
        ti = tarfile.TarInfo("openshell")
        ti.size = len(data)
        ti.mode = 0o755
        tf.addfile(ti, io.BytesIO(data))
    out = rig.extract_binary(good, "openshell", tmp_path / "bin")
    assert out == tmp_path / "bin" / "openshell" and os.access(out, os.X_OK)


# ── the demo UI: OpenClaw's own web UI + the cascade camera extension ─────


def test_the_mcp_server_is_headless_by_default():
    """Under a systemd user unit DISPLAY=:0 is set, and the server's default
    view=1 popped an OpenCV "cascade :: live" window next to the demo. The
    demo's screen is OpenClaw's web UI with the camera extension; the
    cameras reach it through the live-view server, not a desktop window."""
    rig = _load()
    assert rig.serve_mcp_env({"PATH": "/bin"})["CASCADE_VIEW"] == "0"
    assert rig.serve_mcp_env({"PATH": "/bin", "CASCADE_VIEW": "1"})["CASCADE_VIEW"] == "1"


def test_open_puts_the_tokenized_chat_in_the_extension_browser_never_on_stdout(tmp_path, capsys):
    rig = _load()
    run = FakeRun(outs={"sandbox exec": "deadbeefcafe0123"})
    opened = []
    rig.open_chat(tmp_path, rig.Settings(chat_port=18799, sandbox="cascade-agent"), run=run,
                  opener=lambda repo, url: opened.append((repo, url)))
    assert opened == [(rig.REPO, "http://127.0.0.1:18799/#token=deadbeefcafe0123")]
    exec_argv = next(a for a in run.argvs() if a[1:3] == ["sandbox", "exec"])
    assert exec_argv[exec_argv.index("--name") + 1] == "cascade-agent"
    out = capsys.readouterr()
    assert "deadbeefcafe0123" not in out.out + out.err
    assert "18799" in out.out


def test_open_refuses_without_a_token(tmp_path):
    rig = _load()
    with pytest.raises(SystemExit, match="Ready"):
        rig.open_chat(tmp_path, rig.Settings(), run=FakeRun(), opener=lambda repo, url: None)


# ── timed unlock: motion permission always expires ────────────────────────


def _relock_calls(run):
    return [a for a in run.argvs() if a[:2] == ["systemd-run", "--user"]
            and any(x.startswith("--unit=cascade-openshell-relock") for x in a)]


def test_an_unlock_expires_by_default(tmp_path, monkeypatch):
    """A forgotten `motion unlock` must not leave the agent able to move the
    arm indefinitely: every unlock schedules its own re-lock (a transient
    systemd timer that runs `motion lock`)."""
    rig = _load()
    monkeypatch.setattr(rig.time, "time", lambda: 1000.0)
    run = FakeRun(outs={"provider profile export": "resource_version: 3\n"})
    rig.unlock_motion(tmp_path, rig.Settings(), run=run)
    relock = _relock_calls(run)
    assert len(relock) == 1
    assert f"--on-active={rig.DEFAULT_UNLOCK_S}" in relock[0]
    assert relock[0][-2:] == ["motion", "lock"]
    assert rig.motion_locked(tmp_path) is False
    assert rig._read_state(tmp_path)["motion_unlocked_until"] == 1000.0 + rig.DEFAULT_UNLOCK_S


def test_the_relock_is_scheduled_before_the_arm_is_unlocked(tmp_path):
    """If the re-lock cannot be scheduled, the unlock never happens."""
    rig = _load()
    run = FakeRun(rcs={"--user --unit=cascade-openshell-relock": 1},
                  outs={"provider profile export": "resource_version: 3\n"})
    rig.set_motion_lock(tmp_path, rig.Settings(), locked=True, run=run)
    n_updates = len([a for a in run.argvs() if a[1:4] == ["provider", "profile", "update"]])
    with pytest.raises(SystemExit, match="re-lock"):
        rig.unlock_motion(tmp_path, rig.Settings(), run=run)
    assert rig.motion_locked(tmp_path) is True
    assert len([a for a in run.argvs() if a[1:4] == ["provider", "profile", "update"]]) == n_updates


def test_locking_cancels_a_pending_relock(tmp_path):
    rig = _load()
    run = FakeRun(outs={"provider profile export": "resource_version: 3\n"})
    rig.lock_motion(tmp_path, rig.Settings(), run=run)
    assert ["systemctl", "--user", "stop", "cascade-openshell-relock.timer"] in run.argvs()
    assert rig.motion_locked(tmp_path) is True
    assert "motion_unlocked_until" not in rig._read_state(tmp_path)


@pytest.mark.parametrize("text, seconds", [("90s", 90), ("15m", 900), ("1h", 3600), ("2h", 7200), ("45", 45),
                                           ("8h", 28800), ("12h", 43200), ("43200", 43200)])
def test_unlock_durations(text, seconds):
    assert _load().parse_duration(text) == seconds


def test_one_unlock_covers_a_presentation_day_but_still_expires():
    """A presenter demos for hours: a grant that re-locks mid-talk makes the
    agent refuse every motion in front of the audience. One unlock may last a
    working day (12 h); it still expires on its own the same day, so a rig
    left running overnight is observe-only again."""
    rig = _load()
    assert rig.MAX_UNLOCK_S == 12 * 3600
    assert rig.parse_duration("12h") == rig.MAX_UNLOCK_S
    with pytest.raises(SystemExit):
        rig.parse_duration(str(rig.MAX_UNLOCK_S + 1))


@pytest.mark.parametrize("text", ["0", "0m", "13h", "43201", "24h", "-5m", "forever", "", "1.5x"])
def test_an_unlock_is_never_unbounded_or_longer_than_the_cap(text):
    with pytest.raises(SystemExit):
        _load().parse_duration(text)


def test_status_reports_the_time_left(tmp_path, monkeypatch, capsys):
    rig = _load()
    rig._write_state(tmp_path, motion_locked=False, motion_unlocked_until=1600.0)
    monkeypatch.setattr(rig.time, "time", lambda: 1000.0)
    assert rig.motion_status(tmp_path) == "allowed for 10 min more (then re-locked automatically)"
    rig._write_state(tmp_path, motion_locked=True)
    assert rig.motion_status(tmp_path) == "LOCKED (observe-only)"


@pytest.mark.parametrize("saved_lock", [None, True, False])
def test_up_starts_observe_only_unless_motion_is_asked_for(tmp_path, saved_lock):
    """Safe by default: a fresh or restarted agent can look, not move. Motion
    needs an explicit --allow-motion, which is a timed unlock like any other."""
    rig = _load()
    if saved_lock is not None:
        rig._write_state(tmp_path, motion_locked=saved_lock)
    assert rig.startup_locked(tmp_path, observe_only=False, allow_motion=False) is True
    assert rig.startup_locked(tmp_path, observe_only=True, allow_motion=False) is True
    assert rig.startup_locked(tmp_path, observe_only=False, allow_motion=True) is False


def test_extra_denies_survive_a_lock_toggle(tmp_path):
    """`up --deny-tool X` is an operator decision: a later `motion lock` or
    `motion unlock` must keep denying X, not silently drop it."""
    rig = _load()
    run = FakeRun(outs={"provider profile export": "resource_version: 3\n"})
    rig.set_motion_lock(tmp_path, rig.Settings(), locked=True, extra=("push_object",), run=run)
    rig.unlock_motion(tmp_path, rig.Settings(), run=run)
    upd = [a for a in run.argvs() if a[1:4] == ["provider", "profile", "update"]]
    prof = json.loads(Path(upd[-1][upd[-1].index("--file") + 1]).read_text())
    assert {r["tool"] for r in prof["endpoints"][0]["deny_rules"]} == {"reset_stop", "push_object"}
    rig.lock_motion(tmp_path, rig.Settings(), run=run)
    upd = [a for a in run.argvs() if a[1:4] == ["provider", "profile", "update"]]
    prof = json.loads(Path(upd[-1][upd[-1].index("--file") + 1]).read_text())
    assert "push_object" in {r["tool"] for r in prof["endpoints"][0]["deny_rules"]}


def test_the_relock_unit_does_not_stop_itself_midway(tmp_path, monkeypatch):
    """Found live: the re-lock service runs `motion lock`, which cancels the
    re-lock unit; stopping its own .service killed it before it finished
    (the lock was already applied, its bookkeeping was not). Inside the
    re-lock unit only the timer is cancelled."""
    rig = _load()
    monkeypatch.setenv(rig.RELOCK_ENV, "1")
    run = FakeRun(outs={"provider profile export": "resource_version: 3\n"})
    rig._write_state(tmp_path, motion_locked=False, motion_unlocked_until=5.0)
    rig.lock_motion(tmp_path, rig.Settings(), run=run)
    assert ["systemctl", "--user", "stop", "cascade-openshell-relock.service"] not in run.argvs()
    assert ["systemctl", "--user", "stop", "cascade-openshell-relock.timer"] in run.argvs()
    assert rig.motion_locked(tmp_path) and "motion_unlocked_until" not in rig._read_state(tmp_path)


def test_the_relock_unit_identifies_itself(tmp_path):
    rig = _load()
    run = FakeRun(outs={"provider profile export": "resource_version: 3\n"})
    rig.unlock_motion(tmp_path, rig.Settings(), run=run)
    relock = _relock_calls(run)[0]
    assert f"{rig.RELOCK_ENV}=1" in relock


# ── instances: a second, isolated route next to the default one ───────────
#
# The default instance is the live rig: its names, ports, units, provider
# ids, home and state file must stay exactly what they were before
# instances existed, or the next `up` would recreate (or miss) the running
# sandbox. A named instance must share none of them.

#: Today's default route, written out literally (never derived from the module).
TODAY = {
    "gateway_name": "cascade", "gateway_port": 17670,
    "gateway_unit": "cascade-openshell-gateway", "chat_unit": "cascade-openshell-chat",
    "relock_unit": "cascade-openshell-relock", "sandbox_label": "cascade-openshell",
    "brain_provider": "cascade-brain", "robot_provider": "cascade-robot",
    "sandbox": "cascade-agent", "mcp_port": 18741, "chat_port": 18799,
}
TODAY_TOML = ('[openshell]\nversion = 2\n\n[openshell.gateway]\nname = "cascade"\ncompute_driver = "docker"\n'
              'bind_address = "127.0.0.1:17670"\n\n[openshell.drivers.docker]\nsandbox_label = "cascade-openshell"\n')
TODAY_SANDBOX_ARGV = [
    "sandbox", "create", "--name", "cascade-agent", "--from", "IMG",
    "--provider", "cascade-brain", "--provider", "cascade-robot",
    "--env", "CASCADE_BRAIN_PORT=8081", "--env", "CASCADE_MCP_PORT=18741",
    "--env", "CASCADE_CHAT_PORT=18799", "--env", "CASCADE_MODEL=qwen3.8-27b",
    "--restart-policy", "on-failure", "--detach", "--", "cascade-openclaw"]
TODAY_SETTINGS = {"brain_port": 8081, "mcp_port": 18741, "chat_port": 18799,
                  "model": "qwen3.8-27b", "sandbox": "cascade-agent",
                  # added with the launcher route; a deployed.json written before it
                  # carries no such key and is NOT stale for it (see the test below)
                  "context_window": 65536}
#: sha256 of json.dumps(profile, sort_keys=True): the profiles the live
#: gateway holds. Any change here is an update pushed to the running sandbox.
TODAY_PROFILE_SHA = {
    "cascade-brain": "7e81c98a4f6dfa80dbd0be5b5253d5c1e6c60b2db961e2146e813d634e3a8ba4",
    "cascade-robot": "cb34708159f1048412ba6da017c1a953771f47d6ffd3582143ebc089881b2ef7",
}


def _profile_sha(profile: dict) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


def test_golden_todays_default_constants_and_builders(monkeypatch, tmp_path):
    rig = _load()
    assert (rig.GATEWAY_NAME, rig.GATEWAY_PORT, rig.GATEWAY_UNIT, rig.CHAT_UNIT, rig.RELOCK_UNIT,
            rig.SANDBOX_LABEL, rig.DEFAULT_SANDBOX, rig.DEFAULT_MCP_PORT, rig.DEFAULT_CHAT_PORT) == (
        "cascade", 17670, "cascade-openshell-gateway", "cascade-openshell-chat", "cascade-openshell-relock",
        "cascade-openshell", "cascade-agent", 18741, 18799)
    assert rig.DEFAULT_BRAIN_PORT == 8081
    assert rig.gateway_toml() == TODAY_TOML
    for p in (rig.brain_profile(), rig.robot_profile()):
        assert _profile_sha(p) == TODAY_PROFILE_SHA[p["id"]]
    assert rig.sandbox_create_argv(rig.Settings(), "IMG") == TODAY_SANDBOX_ARGV
    assert rig.forward_argv(rig.Settings()) == [
        "forward", "service", "cascade-agent", "--target-port", "18789", "--local", "127.0.0.1:18799"]
    assert asdict(rig.Settings()) == TODAY_SETTINGS
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CASCADE_OPENSHELL_HOME", raising=False)
    assert rig.home_dir() == tmp_path / ".cascade" / "openshell"
    monkeypatch.setenv("CASCADE_OPENSHELL_HOME", str(tmp_path / "elsewhere"))
    assert rig.home_dir() == tmp_path / "elsewhere"


class Host:
    """The host as main() sees it: every child process any command starts is
    recorded and answered here (nothing is executed), and the loopback
    probes of the brain and the MCP server answer 200 without a socket."""

    def __init__(self, rig, tmp_path):
        self.rig, self.tmp, self.user = rig, tmp_path, tmp_path / "user"
        self.argvs: list[list[str]] = []
        self.gateways_up: set[str] = set()   # openshell binaries whose gateway is added and active
        self.registered: dict[str, str] = {}   # openshell binary -> its registered gateway's name

    def gateway_name(self, binary: str) -> str:
        """The gateway a home's CLI has active (its instance's, by default)."""
        home = Path(binary).parents[2]
        return self.registered.get(binary, "cascade" + home.name.removeprefix("openshell"))

    def run(self, argv, env=None, **kw):
        argv = [str(a) for a in argv]
        self.argvs.append(argv)
        rc, out = self.answer(argv)
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")

    def answer(self, argv):
        if argv[0] == "systemctl" and "is-active" in argv:
            return 3, ""   # inactive: `gateway start` launches its unit
        if argv[0].endswith("/openshell"):
            args = argv[1:]
            if args == ["status"]:
                # As 0.1.3 answers: exit 0 even with no gateway registered
                # (found live on a fresh instance home); only the text tells.
                if argv[0] in self.registered and argv[0] not in self.gateways_up:
                    return 1, DOWN_STATUS   # registered, its gateway not running
                if argv[0] not in self.gateways_up:
                    return 0, "Gateway Status\n\n  Status: No gateway configured.\n"
                return 0, (f"Server Status\n\n  Gateway: {self.gateway_name(argv[0])}\n"
                           "  Server: https://127.0.0.1\n  Status: Connected\n")
            if args[:2] == ["gateway", "list"]:
                rows = "".join(f"* {n}  https://127.0.0.1  local  user  mtls\n"
                               for b, n in self.registered.items() if b == argv[0])
                return 0, "  NAME  ENDPOINT  TYPE  SOURCE  AUTH\n" + rows if rows else NO_GATEWAYS
            if args[:2] == ["gateway", "select"]:
                self.gateways_up.add(argv[0])
            if args[:2] == ["gateway", "add"]:
                self.registered[argv[0]] = args[args.index("--name") + 1]
                self.gateways_up.add(argv[0])
            if args[:2] == ["sandbox", "get"]:
                return 0, "Phase: Ready\n"
            if args[:2] == ["sandbox", "exec"]:
                return 0, "chat-token"
            if args[:3] == ["provider", "profile", "export"]:
                return 0, "resource_version: 3\n"
        return 0, ""

    def install(self, home: Path, name: str = "") -> Path:
        """What `[--instance name] install` + a first `serve-mcp` leave
        behind, minus the download: the home claimed by its instance,
        placeholder binaries (never executed) and the MCP token."""
        if name:
            self.rig.claim_home(self.rig.instance(name), home)
        bindir = home / self.rig.OPENSHELL_VERSION / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        for binary in ("openshell", "openshell-gateway"):
            (bindir / binary).write_text("")
        tok = home / "mcp.token"
        tok.write_text("mcp-token\n")
        tok.chmod(0o600)
        return home

    def drive(self, *pre: str) -> None:
        """Every command that touches a unit, a provider or the sandbox."""
        for cmd in (["gateway", "start"], ["up"], ["status"], ["chat-url"],
                    ["motion", "unlock", "--for", "90s"], ["motion", "status"], ["motion", "lock"],
                    ["down", "--gateway"]):
            assert self.rig.main([*pre, *cmd]) == 0, cmd


@pytest.fixture
def host(monkeypatch, tmp_path):
    rig = _load()
    h = Host(rig, tmp_path)
    monkeypatch.setattr(rig, "subprocess", types.SimpleNamespace(
        run=h.run, CompletedProcess=subprocess.CompletedProcess))
    monkeypatch.setattr(rig, "_http_status", lambda *a, **k: 200)
    monkeypatch.setattr(rig, "time", types.SimpleNamespace(time=lambda: 1000.0, sleep=lambda s: None))
    monkeypatch.setenv("HOME", str(h.user))   # Path.home(): never the real ~/.cascade
    for k in ("CASCADE_OPENSHELL_HOME", "CASCADE_OPENSHELL_INSTANCE", "CASCADE_OPENSHELL_RELOCK",
              "CASCADE_BRAIN_KEY_FILE"):
        monkeypatch.delenv(k, raising=False)
    key = h.user / ".cascade" / "nemoclaw" / "llama.key"
    key.parent.mkdir(parents=True)
    key.write_text("brain-key\n")
    key.chmod(0o600)
    return h


def _openshell_targets(args: list[str]) -> set[str]:
    t = set()
    if args[:2] in (["provider", "get"], ["provider", "update"]):
        t.add(f"provider:{args[2]}")
    if args[:2] == ["provider", "create"]:
        t |= {f"provider:{args[args.index('--name') + 1]}", f"provider:{args[args.index('--type') + 1]}"}
    if args[:2] == ["provider", "profile"]:
        if "--file" in args:
            t.add(f"profile:{json.loads(Path(args[args.index('--file') + 1]).read_text())['id']}")
        if args[2] in ("export", "update"):
            t.add(f"profile:{args[-1]}")
    if args[:2] in (["sandbox", "get"], ["sandbox", "delete"]):
        t.add(f"sandbox:{args[2]}")
    if args[:2] in (["sandbox", "create"], ["sandbox", "exec"]):
        t.add(f"sandbox:{args[args.index('--name') + 1]}")
        t |= {f"provider:{args[i + 1]}" for i, a in enumerate(args) if a == "--provider"}
    if args[:2] == ["gateway", "add"]:
        t.add(f"gateway:{args[args.index('--name') + 1]}@{args[2]}")
    if args[:2] == ["forward", "service"]:
        t.add(f"forward:{args[2]}@{args[args.index('--local') + 1]}")
    return t


def _targets(argvs: list[list[str]]) -> set[str]:
    """What every recorded child acts on: units, gateway, providers,
    profiles, sandbox, chat forward, the binaries and the re-lock's command."""
    t = set()
    for a in argvs:
        if a[0] == "systemctl":
            t |= {f"unit:{u}" for u in a[3:] if not u.startswith("-")}
        elif a[0] == "systemd-run":
            t |= {f"unit:{x.split('=', 1)[1]}" for x in a if x.startswith("--unit=")}
            t |= {f"relock-home:{x.split('=', 1)[1]}" for x in a if x.startswith("CASCADE_OPENSHELL_HOME=")}
            for i, x in enumerate(a):
                if "=" in x:
                    continue
                if x.endswith(("/openshell", "/openshell-gateway")):
                    t.add(f"bin:{x}")
                    t |= _openshell_targets(a[i + 1:]) if x.endswith("/openshell") else set()
                    break
                if x.endswith("openshell_rig.py"):
                    t.add(f"relock:{' '.join(a[i + 1:])}")
                    break
        elif a[0].endswith("/openshell"):
            t.add(f"bin:{a[0]}")
            t |= _openshell_targets(a[1:])
        elif a[0] != "docker":
            raise AssertionError(f"unexpected child {a}")
    return t


def _expected_targets(home: Path, i: dict, relock: str) -> set[str]:
    b = home / "0.1.3" / "bin"
    return {
        f"unit:{i['gateway_unit']}", f"gateway:{i['gateway_name']}@https://127.0.0.1:{i['gateway_port']}",
        f"unit:{i['chat_unit']}", f"forward:{i['sandbox']}@127.0.0.1:{i['chat_port']}",
        f"unit:{i['relock_unit']}", f"unit:{i['relock_unit']}.timer", f"unit:{i['relock_unit']}.service",
        f"relock:{relock}", f"relock-home:{home}",
        f"profile:{i['brain_provider']}", f"profile:{i['robot_provider']}",
        f"provider:{i['brain_provider']}", f"provider:{i['robot_provider']}",
        f"sandbox:{i['sandbox']}", f"bin:{b / 'openshell'}", f"bin:{b / 'openshell-gateway'}",
    }


def test_golden_the_default_route_targets_exactly_todays_units_providers_and_sandbox(host):
    """gateway start, up, status, chat-url, unlock, lock, down --gateway of
    the default instance act on today's names and nothing else, keep
    today's home and state file, and save today's settings (a settings
    diff would recreate the live sandbox on the next `up`)."""
    home = host.install(host.user / ".cascade" / "openshell")
    host.drive()
    assert _targets(host.argvs) == _expected_targets(home, TODAY, relock="motion lock")
    state = json.loads((home / "deployed.json").read_text())
    assert state["settings"] == TODAY_SETTINGS
    assert sorted(p.name for p in (host.user / ".cascade").iterdir()) == ["nemoclaw", "openshell"]
    upd = [a for a in host.argvs if a[1:4] == ["provider", "profile", "update"]]
    brain = json.loads(Path(next(a for a in upd if a[-1] == "cascade-brain")[5]).read_text())
    assert _profile_sha({k: v for k, v in brain.items() if k != "resource_version"}) == TODAY_PROFILE_SHA["cascade-brain"]
    # byte for byte: every systemctl call, in order, and the re-lock unit
    relock = ["systemctl", "--user", "stop", "cascade-openshell-relock.timer"], \
        ["systemctl", "--user", "stop", "cascade-openshell-relock.service"], \
        ["systemctl", "--user", "reset-failed", "cascade-openshell-relock.timer", "cascade-openshell-relock.service"]
    assert [a for a in host.argvs if a[0] == "systemctl"] == [
        ["systemctl", "--user", "is-active", "--quiet", "cascade-openshell-gateway"],     # gateway start
        ["systemctl", "--user", "stop", "cascade-openshell-gateway"],
        ["systemctl", "--user", "reset-failed", "cascade-openshell-gateway"],
        *relock,                                                                        # up
        ["systemctl", "--user", "stop", "cascade-openshell-chat"],
        ["systemctl", "--user", "reset-failed", "cascade-openshell-chat"],
        ["systemctl", "--user", "is-active", "--quiet", "cascade-openshell-chat"],        # status
        *relock,                                                                        # unlock
        *relock,                                                                        # lock
        ["systemctl", "--user", "stop", "cascade-openshell-chat"],                       # down --gateway
        ["systemctl", "--user", "stop", "cascade-openshell-gateway"]]
    assert [a for a in host.argvs if a[0] == "systemd-run" and "--on-active=90" in a] == [[
        "systemd-run", "--user", "--unit=cascade-openshell-relock", "--collect", "--on-active=90",
        "-E", f"CASCADE_OPENSHELL_HOME={home}", "-E", "CASCADE_OPENSHELL_RELOCK=1",
        sys.executable, str(SCRIPT.resolve()), "motion", "lock"]]


#: The documented scheme for an example named instance `sim`: slot
#: 1 + sha256("sim") mod 99 = 40, so ports 17670+40, 17770+40, 17870+40.
SIM = {
    "gateway_name": "cascade-sim", "gateway_port": 17710,
    "gateway_unit": "cascade-openshell-sim-gateway", "chat_unit": "cascade-openshell-sim-chat",
    "relock_unit": "cascade-openshell-sim-relock", "sandbox_label": "cascade-openshell-sim",
    "brain_provider": "cascade-sim-brain", "robot_provider": "cascade-sim-robot",
    "sandbox": "cascade-sim-agent", "mcp_port": 17810, "chat_port": 17910,
}


def test_golden_the_default_instance_is_todays_route(monkeypatch, tmp_path):
    """No --instance, no CASCADE_OPENSHELL_INSTANCE: the live rig's route,
    with today's names, ports, units, ids, home and settings."""
    rig = _load()
    d = rig.instance("")
    assert asdict(d) == {"name": "", **TODAY, "mcp_unit": "cascade-openshell-mcp", "home_name": "openshell"}
    assert rig.DEFAULT_INSTANCE == d
    assert rig.resolve_instance(None, env={}) == d
    assert rig.resolve_instance(None, env={"CASCADE_OPENSHELL_INSTANCE": ""}) == d
    assert d.settings() == rig.Settings() and asdict(d.settings()) == TODAY_SETTINGS
    assert rig.gateway_toml(d) == TODAY_TOML
    for p in (rig.brain_profile(inst=d), rig.robot_profile(inst=d)):
        assert _profile_sha(p) == TODAY_PROFILE_SHA[p["id"]]
    assert rig.sandbox_create_argv(d.settings(), "IMG", inst=d) == TODAY_SANDBOX_ARGV
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CASCADE_OPENSHELL_HOME", raising=False)
    assert d.home() == rig.home_dir() == tmp_path / ".cascade" / "openshell"
    assert rig._state_file(d.home()) == tmp_path / ".cascade" / "openshell" / "deployed.json"


def test_the_sim_instance_follows_the_documented_scheme(monkeypatch, tmp_path):
    rig = _load()
    s = rig.instance("sim")
    assert asdict(s) == {"name": "sim", **SIM, "mcp_unit": "cascade-openshell-sim-mcp", "home_name": "openshell-sim"}
    assert rig.port_slot("sim") == 40
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CASCADE_OPENSHELL_HOME", raising=False)
    assert s.home() == rig.home_dir(s) == tmp_path / ".cascade" / "openshell-sim"
    monkeypatch.setenv("CASCADE_OPENSHELL_HOME", str(tmp_path / "x"))
    assert s.home() == tmp_path / "x", "an explicit CASCADE_OPENSHELL_HOME still wins"


def test_a_named_instance_shares_no_name_port_unit_id_or_home_with_the_default(monkeypatch, tmp_path):
    """Every field of the instance (so a field added later is covered too)
    differs from the default's, and so do the home, the gateway config, the
    provider profile ids and the sandbox's providers. The brain is shared on
    purpose: one host LLM serves every instance."""
    rig = _load()
    d, s = rig.instance(""), rig.instance("sim")
    dv = {k: v for k, v in asdict(d).items() if k != "name"}
    sv = {k: v for k, v in asdict(s).items() if k != "name"}
    assert dv.keys() == sv.keys()
    assert set(dv.values()).isdisjoint(sv.values()), set(dv.values()) & set(sv.values())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CASCADE_OPENSHELL_HOME", raising=False)
    assert d.home() != s.home() and not s.home().is_relative_to(d.home())
    dt, st = (tomllib.loads(rig.gateway_toml(i))["openshell"] for i in (d, s))
    assert dt["gateway"]["name"] != st["gateway"]["name"]
    assert dt["gateway"]["bind_address"] != st["gateway"]["bind_address"] == f"127.0.0.1:{s.gateway_port}"
    assert st["drivers"]["docker"]["sandbox_label"] == s.sandbox_label != d.sandbox_label
    assert {rig.brain_profile(inst=s)["id"], rig.robot_profile(inst=s)["id"]} == {"cascade-sim-brain", "cascade-sim-robot"}
    argv = rig.sandbox_create_argv(s.settings(), "IMG", inst=s)
    assert [argv[i + 1] for i, a in enumerate(argv) if a in ("--name", "--provider")] == [
        "cascade-sim-agent", "cascade-sim-brain", "cascade-sim-robot"]
    assert {"CASCADE_MCP_PORT=17810", "CASCADE_CHAT_PORT=17910", "CASCADE_BRAIN_PORT=8081"} <= set(argv)


def test_named_instances_never_collide_with_each_other_or_a_known_listener():
    """Per field, two names never map to the same unit, id, label or sandbox
    (roles are one word, so `a` + `-chat` never meets `a-chat`), and the
    port bands of all 99 slots are pairwise distinct and clear of every
    port the rig, NemoClaw and OpenClaw already use."""
    rig = _load()
    names = ["", "a", "a-chat", "a-relock", "a-gateway", "a-mcp", "a-brain", "a-robot", "a-agent",
             "chat", "brain", "openshell", "sim", "sim-2", "x" * 16]
    rows = [asdict(rig.instance(n)) for n in names]
    for field in rows[0]:
        if field.endswith("_port"):
            continue
        values = [r[field] for r in rows]
        assert len(set(values)) == len(values), field
    known = {17670, 18741, 18799, 18740, 18789, 18790, 18791, 18080, 18877, 18878,
             8080, 8081, 8090, 8092, 5556, 5557}
    ports = [p for k in range(1, 100) for p in rig.named_ports(k)]
    assert len(set(ports)) == len(ports) == 3 * 99
    assert not set(ports) & known
    assert {rig.port_slot(n) for n in names[1:]} <= set(range(1, 100))


@pytest.mark.parametrize("bad", ["Sim", "1sim", "-sim", "sim_1", "sim.1", "sim 1", "x" * 17, "sim\n",
                                 "../sim", "sim/x", "default"])
def test_invalid_instance_names_are_refused(bad):
    rig = _load()
    with pytest.raises(SystemExit, match="instance"):
        rig.resolve_instance(bad, env={})
    with pytest.raises(SystemExit, match="instance"):
        rig.resolve_instance(None, env={"CASCADE_OPENSHELL_INSTANCE": bad})


def test_the_instance_comes_from_the_cli_or_the_environment_and_the_cli_wins():
    rig = _load()
    env = {"CASCADE_OPENSHELL_INSTANCE": "sim"}
    assert rig.resolve_instance(None, env=env).name == "sim"
    assert rig.resolve_instance("dev", env=env).name == "dev"
    assert rig.resolve_instance("", env=env) == rig.DEFAULT_INSTANCE, "--instance '' selects the default"
    assert rig.resolve_instance("x" * 16, env={}).name == "x" * 16


SIM_SETTINGS = {**TODAY_SETTINGS, "mcp_port": 17810, "chat_port": 17910, "sandbox": "cascade-sim-agent"}


def test_golden_default_status_reads_as_before(host, capsys):
    home = host.install(host.user / ".cascade" / "openshell")
    host.gateways_up.add(str(home / "0.1.3" / "bin" / "openshell"))
    assert host.rig.main(["status"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "             openshell: 0.1.3 installed",
        "               gateway: healthy on 127.0.0.1:17670",
        "               sandbox: cascade-agent: Ready",
        "provider cascade-brain: present",
        "provider cascade-robot: present",
        "          chat forward: 127.0.0.1:18799 (down)",
        "                motion: allowed",
        "                 brain: 127.0.0.1:8081 -> 200",
        "             robot MCP: 127.0.0.1:18741 -> 200"]


def test_a_named_route_only_ever_targets_its_own_units_providers_and_sandbox(host, capsys):
    """gateway start, up, status, chat-url, unlock, lock, down --gateway of
    `--instance sim` act on sim's units, gateway, providers, profiles,
    sandbox, binaries and home, and on nothing of the default route."""
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    host.drive("--instance", "sim")
    got = _targets(host.argvs)
    assert got == _expected_targets(home, SIM, relock="--instance sim motion lock")
    assert got.isdisjoint(_expected_targets(host.user / ".cascade" / "openshell", TODAY, relock="motion lock"))
    assert json.loads((home / "deployed.json").read_text())["settings"] == SIM_SETTINGS
    assert sorted(p.name for p in (host.user / ".cascade").iterdir()) == ["nemoclaw", "openshell-sim"]
    out = capsys.readouterr().out
    assert "scripts/openshell_rig.py --instance sim chat-url" in out, "hints name the instance"
    status = [line.strip() for line in out.splitlines()]
    assert f"instance: sim ({home})" in status, "status says which instance it reports"
    assert {"gateway: healthy on 127.0.0.1:17710", "sandbox: cascade-sim-agent: Ready",
            "provider cascade-sim-brain: present", "provider cascade-sim-robot: present",
            "chat forward: 127.0.0.1:17910 (down)", "robot MCP: 127.0.0.1:17810 -> 200"} <= set(status)


@pytest.mark.parametrize("env, pre, post", [("sim", [], []), ("dev", ["--instance", "sim"], []),
                                            ("dev", [], ["--instance", "sim"])])
def test_the_instance_comes_from_the_cli_or_the_environment_and_the_cli_wins_end_to_end(
        host, monkeypatch, env, pre, post):
    monkeypatch.setenv("CASCADE_OPENSHELL_INSTANCE", env)
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    host.gateways_up.add(str(home / "0.1.3" / "bin" / "openshell"))
    for cmd in (["status"], ["motion", "unlock", "--for", "90s"], ["motion", "lock"], ["down"]):
        assert host.rig.main([*pre, *cmd, *post]) == 0
    assert _targets(host.argvs) <= _expected_targets(home, SIM, relock="--instance sim motion lock")
    assert "unit:cascade-openshell-sim-relock.timer" in _targets(host.argvs)


def test_locking_or_unlocking_one_instance_never_touches_anothers_relock(host):
    """Each instance's re-lock is its own timer: `motion lock` of sim must
    not cancel the default route's pending re-lock (that would leave the
    live arm unlocked with no expiry), and vice versa."""
    default_home = host.install(host.user / ".cascade" / "openshell")
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    main = host.rig.main
    assert main(["motion", "unlock", "--for", "90s"]) == 0
    n = len(host.argvs)
    assert main(["--instance", "sim", "motion", "unlock", "--for", "90s"]) == 0
    assert main(["--instance", "sim", "motion", "lock"]) == 0
    sim = _targets(host.argvs[n:])
    assert {t for t in sim if t.startswith("unit:")} == {
        "unit:cascade-openshell-sim-relock", "unit:cascade-openshell-sim-relock.timer",
        "unit:cascade-openshell-sim-relock.service"}
    assert {t for t in sim if t.startswith(("profile:", "provider:"))} == {"profile:cascade-sim-robot"}
    n = len(host.argvs)
    assert main(["motion", "lock"]) == 0
    assert {t for t in _targets(host.argvs[n:]) if t.startswith(("unit:", "profile:"))} == {
        "unit:cascade-openshell-relock.timer", "unit:cascade-openshell-relock.service", "profile:cascade-robot"}
    assert host.rig.motion_locked(default_home)


def test_the_relock_timer_locks_the_instance_that_scheduled_it(host, monkeypatch):
    """The timer re-enters this CLI with `--instance sim`, so it locks sim
    even if the unit's environment names another instance."""
    rig = host.rig
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    assert rig.main(["--instance", "sim", "motion", "unlock", "--for", "90s"]) == 0
    (unit,) = [a for a in host.argvs if a[0] == "systemd-run"]
    assert unit[:5] == ["systemd-run", "--user", "--unit=cascade-openshell-sim-relock", "--collect", "--on-active=90"]
    assert f"CASCADE_OPENSHELL_HOME={home}" in unit
    tail = unit[unit.index(str(SCRIPT.resolve())) + 1:]
    assert tail == ["--instance", "sim", "motion", "lock"]
    seen = []
    monkeypatch.setattr(rig, "cmd_motion", lambda inst, h, args: seen.append((inst.name, h, args.action)) or 0)
    monkeypatch.setenv("CASCADE_OPENSHELL_INSTANCE", "dev")
    monkeypatch.setenv("CASCADE_OPENSHELL_HOME", str(home))
    assert rig.main(tail) == 0
    assert seen == [("sim", home, "lock")]


def test_explicit_flags_still_win_over_the_instance_defaults(host):
    rig = host.rig
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    host.gateways_up.add(str(home / "0.1.3" / "bin" / "openshell"))
    assert rig.main(["--instance", "sim", "up", "--mcp-port", "17999", "--chat-port", "17998",
                     "--sandbox", "my-agent"]) == 0
    saved = json.loads((home / "deployed.json").read_text())["settings"]
    assert saved == {**SIM_SETTINGS, "mcp_port": 17999, "chat_port": 17998, "sandbox": "my-agent"}
    assert "forward:my-agent@127.0.0.1:17998" in _targets(host.argvs)
    # ...and are remembered by the instance, never by the default route
    assert rig._load_settings(home, types.SimpleNamespace(), rig.instance("sim")) == rig.Settings(**saved)
    assert rig._load_settings(tmp := host.tmp / "empty", types.SimpleNamespace(), rig.instance("")) == rig.Settings()
    assert not tmp.exists()


# ── nothing outside the instance table names a default ────────────────────

#: The default instance's table: only `instance()` / `named_ports()` (and the
#: Settings field defaults, which equal the default instance) may read it.
_TABLE = {"GATEWAY_NAME", "GATEWAY_PORT", "GATEWAY_UNIT", "CHAT_UNIT", "RELOCK_UNIT", "MCP_UNIT",
          "SANDBOX_LABEL", "BRAIN_PROVIDER", "ROBOT_PROVIDER", "HOME_NAME",
          "DEFAULT_SANDBOX", "DEFAULT_MCP_PORT", "DEFAULT_CHAT_PORT"}
_TABLE_READERS = {"instance", "named_ports", "Settings"}
_DEFAULT_STRINGS = ("cascade-brain", "cascade-robot", "cascade-agent", "cascade-openshell-gateway",
                    "cascade-openshell-chat", "cascade-openshell-relock", "cascade-openshell-mcp")


def _top_level_owners(tree):
    """(node, name of the top-level def/class holding it, or None)."""
    for top in tree.body:
        owner = top.name if isinstance(top, (ast.FunctionDef, ast.ClassDef)) else None
        for node in ast.walk(top):
            yield node, owner, top


def test_nothing_outside_the_instance_table_hardcodes_a_default_name_or_port():
    tree = ast.parse(SCRIPT.read_text())
    docstrings = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    bad = []
    for node, owner, top in _top_level_owners(tree):
        table_def = isinstance(top, ast.Assign) and all(getattr(t, "id", "") in _TABLE for t in top.targets)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in _TABLE \
                and owner not in _TABLE_READERS:
            bad.append((node.lineno, node.id))
        if isinstance(node, ast.Constant) and id(node) not in docstrings and not table_def:
            v = node.value
            if (isinstance(v, str) and any(s in v for s in _DEFAULT_STRINGS)) \
                    or (isinstance(v, int) and not isinstance(v, bool) and v in (17670, 18741, 18799)):
                bad.append((node.lineno, v))
    assert not bad, f"default names outside the instance table: {bad}"


def test_every_call_inside_the_module_passes_its_instance():
    """A forgotten `inst=` would silently act on the default (live) route."""
    tree = ast.parse(SCRIPT.read_text())
    takes = {}
    for f in tree.body:
        if isinstance(f, ast.FunctionDef):
            pos = [a.arg for a in f.args.args]
            if "inst" in pos or "inst" in [a.arg for a in f.args.kwonlyargs]:
                takes[f.name] = pos.index("inst") if "inst" in pos else None
    assert {"gateway_toml", "brain_profile", "robot_profile", "sandbox_create_argv", "home_dir", "set_motion_lock",
            "_cancel_relock", "unlock_motion", "lock_motion", "_load_settings", "cmd_up", "cmd_gateway"} <= set(takes)
    missing = []
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in takes:
            i = takes[c.func.id]
            if not any(k.arg == "inst" for k in c.keywords) and not (i is not None and len(c.args) > i):
                missing.append((c.lineno, c.func.id))
    assert not missing, f"calls that fall back to the default instance: {missing}"


# ── one home, one instance ────────────────────────────────────────────────


def test_a_named_instance_never_adopts_the_live_routes_home(host, monkeypatch):
    """CASCADE_OPENSHELL_HOME overrides the home of any instance, so a stray
    export could point `--instance sim` at the live route's state (its
    token, its lock, its gateway). A named instance only claims an empty
    home or its own; nothing runs and nothing is written otherwise."""
    rig = host.rig
    live = host.install(host.user / ".cascade" / "openshell")
    rig._write_state(live, motion_locked=True)
    before = {p: p.read_bytes() for p in live.rglob("*") if p.is_file()}
    monkeypatch.setenv("CASCADE_OPENSHELL_HOME", str(live))
    for cmd in (["status"], ["motion", "unlock"], ["motion", "lock"], ["up"], ["down", "--gateway"],
                ["gateway", "start"], ["install"]):
        with pytest.raises(SystemExit, match="default instance"):
            rig.main(["--instance", "sim", *cmd])
    assert host.argvs == []
    assert {p: p.read_bytes() for p in live.rglob("*") if p.is_file()} == before


def test_the_default_route_never_adopts_a_named_instances_home(host, monkeypatch):
    rig = host.rig
    sim = host.tmp / "sim-home"
    monkeypatch.setenv("CASCADE_OPENSHELL_HOME", str(sim))
    assert rig.main(["--instance", "sim", "motion", "status"]) == 0, "an empty home is claimed"
    assert rig.main(["--instance", "sim", "motion", "status"]) == 0, "and stays its instance's"
    for pre in ([], ["--instance", "dev"]):
        with pytest.raises(SystemExit, match="instance 'sim'"):
            rig.main([*pre, "motion", "lock"])
    assert host.argvs == []


def test_the_instance_command_prints_the_scheme_and_touches_nothing(host, capsys):
    rig = host.rig
    assert rig.main(["--instance", "sim", "instance"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown == {"name": "sim", **SIM, "mcp_unit": "cascade-openshell-sim-mcp", "home_name": "openshell-sim",
                     "home": str(host.user / ".cascade" / "openshell-sim")}
    assert rig.main(["instance"]) == 0
    assert json.loads(capsys.readouterr().out)["gateway_port"] == 17670
    assert not (host.user / ".cascade" / "openshell-sim").exists() and host.argvs == []


# ── the launcher's route: exec, mcp-start, teardown ───────────────────────
# scripts/launch.sh drives an instance end to end on Linux: it starts the
# robot MCP server as the instance's unit, brings the sandbox up, runs the
# OpenClaw CLI INSIDE the sandbox for its proof, and tears it all down.


def _rig_exec(host, monkeypatch):
    seen = []
    monkeypatch.setattr(host.rig, "_replace_process", lambda argv, env: seen.append((argv, env)) or 0)
    return seen


def test_exec_runs_the_command_inside_the_instances_sandbox(host, monkeypatch):
    """The launcher's proof drives OpenClaw where the agent lives: the
    process is REPLACED by `openshell sandbox exec` (stdout and the exit
    code are the remote command's), never run on the host."""
    seen = _rig_exec(host, monkeypatch)
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    assert host.rig.main(["--instance", "sim", "exec", "--", "openclaw", "agent", "--json"]) == 0
    ((argv, env),) = seen
    assert argv == [str(home / "0.1.3" / "bin" / "openshell"), "sandbox", "exec", "--name", "cascade-sim-agent",
                    "--no-tty", "--", "openclaw", "agent", "--json"]
    assert env["XDG_CONFIG_HOME"] == str(home / "config"), "this instance's gateway registry"
    assert host.argvs == []


def test_exec_needs_a_command(host, monkeypatch):
    _rig_exec(host, monkeypatch)
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    with pytest.raises(SystemExit, match="command"):
        host.rig.main(["--instance", "sim", "exec"])


def test_only_exec_serve_mcp_and_mcp_start_take_a_trailing_command(host):
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    with pytest.raises(SystemExit, match="status"):
        host.rig.main(["--instance", "sim", "status", "--", "rm", "-rf", "/"])
    assert host.argvs == []


def _env_file(tmp_path, env):
    f = tmp_path / "mcp-env.json"
    f.write_text(json.dumps(env))
    f.chmod(0o600)
    return f


def test_mcp_start_runs_serve_mcp_as_the_instances_unit(host, tmp_path, capsys):
    """One long-lived robot server per instance, on the instance's MCP port,
    with the launcher's runtime environment (from a private file, never the
    unit's argv) and its launch-owner flags passed through."""
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    f = _env_file(tmp_path, {"CASCADE_ARM": "so101_mujoco", "CASCADE_CAMERAS": "mujoco_scene"})
    assert host.rig.main(["--instance", "sim", "mcp-start", "--env-file", str(f), "--replace",
                          "--", "--launch-owner", "o1", "--launch-state-dir", "/state"]) == 0
    (unit,) = [a for a in host.argvs if a[0] == "systemd-run"]
    assert "--unit=cascade-openshell-sim-mcp" in unit and "MemoryMax=infinity" in unit
    tail = unit[unit.index(str(SCRIPT.resolve())) + 1:]
    saved = home / "mcp-env.json"
    assert tail == ["--instance", "sim", "serve-mcp", "--mcp-port", "17810", "--mcp-token-file",
                    str(home / "mcp.token"), "--env-file", str(saved),
                    "--", "--launch-owner", "o1", "--launch-state-dir", "/state"]
    assert json.loads(saved.read_text()) == {"CASCADE_ARM": "so101_mujoco", "CASCADE_CAMERAS": "mujoco_scene"}
    assert saved.stat().st_mode & 0o777 == 0o600
    assert not any("so101_mujoco" in a for a in unit), "the environment never rides the unit's argv"
    assert "127.0.0.1:17810" in capsys.readouterr().out


def test_mcp_start_never_replaces_a_running_server_unless_asked(host, monkeypatch):
    """The default instance's MCP unit is the live robot's server: starting
    over it would cut torque mid-motion. Without --replace a running unit
    is refused and nothing is touched."""
    host.install(host.user / ".cascade" / "openshell")
    answer = host.answer
    monkeypatch.setattr(host, "answer", lambda argv: (0, "") if argv[:4] == [
        "systemctl", "--user", "is-active", "--quiet"] and argv[-1] == "cascade-openshell-mcp" else answer(argv))
    with pytest.raises(SystemExit, match="cascade-openshell-mcp is already running"):
        host.rig.main(["mcp-start"])
    assert [a for a in host.argvs if a[0] != "systemctl" or "is-active" not in a] == []


def test_mcp_start_waits_for_the_server_to_answer(host, monkeypatch):
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    codes = iter(["down (refused)", "down (refused)", 200])
    monkeypatch.setattr(host.rig, "_check_mcp", lambda cfg, token: next(codes))
    answer = host.answer
    monkeypatch.setattr(host, "answer", lambda argv: (0, "") if "is-active" in argv and argv[-1].endswith("-mcp")
                        and any(a[0] == "systemd-run" for a in host.argvs) else answer(argv))
    assert host.rig.main(["--instance", "sim", "mcp-start", "--replace"]) == 0
    with pytest.raises(StopIteration):
        next(codes)


def test_mcp_start_fails_with_the_log_when_the_server_dies(host, monkeypatch):
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    (home / "mcp-server.log").write_text("ImportError: no module named mujoco\n")
    monkeypatch.setattr(host.rig, "_check_mcp", lambda cfg, token: "down (refused)")
    with pytest.raises(SystemExit, match="(?s)exited.*no module named mujoco"):
        host.rig.main(["--instance", "sim", "mcp-start", "--replace"])


def test_mcp_start_gives_up_after_its_wait(host, monkeypatch):
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    monkeypatch.setattr(host.rig, "_check_mcp", lambda cfg, token: 401)
    answer = host.answer
    monkeypatch.setattr(host, "answer", lambda argv: (0, "") if "is-active" in argv and argv[-1].endswith("-mcp")
                        and any(a[0] == "systemd-run" for a in host.argvs) else answer(argv))
    with pytest.raises(SystemExit, match="did not answer within 3 s.*401"):
        host.rig.main(["--instance", "sim", "mcp-start", "--replace", "--wait", "3"])


@pytest.mark.parametrize("bad", ['["not", "a", "dict"]', '{"CASCADE_ARM": 3}', "not json"])
def test_mcp_start_refuses_a_malformed_environment_file(host, tmp_path, bad):
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    f = tmp_path / "env.json"
    f.write_text(bad)
    with pytest.raises(SystemExit, match="env-file"):
        host.rig.main(["--instance", "sim", "mcp-start", "--replace", "--env-file", str(f)])
    assert host.argvs == []


def test_serve_mcp_applies_the_environment_file_and_the_owner_flags(host, monkeypatch, tmp_path):
    seen = _rig_exec(host, monkeypatch)
    monkeypatch.setattr(host.rig.os, "chdir", lambda p: None)
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    f = _env_file(tmp_path, {"CASCADE_ARM": "so101_mujoco", "CASCADE_MCP_TOKEN": "leak"})
    monkeypatch.setenv("CASCADE_ARM", "rebot_rs")
    assert host.rig.main(["--instance", "sim", "serve-mcp", "--env-file", str(f),
                          "--", "--launch-owner", "o1", "--launch-state-dir", "/state"]) == 0
    ((argv, env),) = seen
    assert argv[1:] == ["-m", "cascade.apps.mcp_server", "--http", "127.0.0.1:17810", "--token-file",
                        str(home / "mcp.token"), "--launch-owner", "o1", "--launch-state-dir", "/state"]
    assert env["CASCADE_ARM"] == "so101_mujoco", "the launcher's runtime wins over the unit's environment"
    assert "CASCADE_MCP_TOKEN" not in env, "the token file stays the only source"


# ── the robot server reaches the instance's brain for perception ──────────
# configs/demo.yaml identifies objects with the brain (`detector: type: brain`)
# at ${CASCADE_PERCEPTION_URL:-...}. serve-mcp points the robot server at the
# brain its instance's sandbox uses; the key travels as a FILE path only.

PERCEPTION = ("CASCADE_PERCEPTION_URL", "CASCADE_PERCEPTION_MODEL", "CASCADE_PERCEPTION_KEY_FILE")
#: what serve-mcp adds to an environment that sets none of it
SERVE_SCRUB = (*PERCEPTION, "PYTHONPATH", "YOLO_OFFLINE", "ULTRALYTICS_OFFLINE", "CASCADE_VIEW", "CASCADE_MCP_TOKEN")


def _serve(host, monkeypatch, *args):
    """`serve-mcp` through main(): the process it would become and its environment."""
    seen = _rig_exec(host, monkeypatch)
    monkeypatch.setattr(host.rig.os, "chdir", lambda p: None)
    assert host.rig.main(list(args)) == 0
    ((argv, env),) = seen
    return argv, env


def test_golden_the_default_robot_server_is_todays_plus_the_brain_endpoint(host, monkeypatch):
    """The live rig's server (default instance, no flags): today's argv on
    :18741 and today's environment, plus the brain its sandbox uses (:8081,
    qwen3.8-27b, NemoClaw's key file). The key's contents appear nowhere."""
    for name in SERVE_SCRUB:
        monkeypatch.delenv(name, raising=False)
    home = host.install(host.user / ".cascade" / "openshell")
    before = dict(os.environ)
    argv, env = _serve(host, monkeypatch, "serve-mcp")
    assert argv == [sys.executable, "-m", "cascade.apps.mcp_server", "--http", "127.0.0.1:18741",
                    "--token-file", str(home / "mcp.token")]
    key = host.user / ".cascade" / "nemoclaw" / "llama.key"
    assert env == {**before, "PYTHONPATH": str(REPO / "src"), "YOLO_OFFLINE": "True", "ULTRALYTICS_OFFLINE": "True",
                   "CASCADE_VIEW": "0", "CASCADE_PERCEPTION_URL": "http://127.0.0.1:8081/v1",
                   "CASCADE_PERCEPTION_MODEL": "qwen3.8-27b", "CASCADE_PERCEPTION_KEY_FILE": str(key)}
    secret = key.read_text().strip()
    assert secret == "brain-key"
    assert not [k for k, v in env.items() if secret in v] and not [a for a in argv if secret in a]


def test_a_named_instance_points_perception_at_its_own_brain(host, monkeypatch):
    """The sim instance's server asks the brain that instance's sandbox was
    brought up with (`up --brain-port`, saved), not the default route's."""
    for name in PERCEPTION:
        monkeypatch.delenv(name, raising=False)
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    host.gateways_up.add(str(home / "0.1.3" / "bin" / "openshell"))
    assert host.rig.main(["--instance", "sim", "up", "--brain-port", "8093"]) == 0
    argv, env = _serve(host, monkeypatch, "--instance", "sim", "serve-mcp")
    assert argv[argv.index("--http") + 1] == "127.0.0.1:17810"
    assert env["CASCADE_PERCEPTION_URL"] == "http://127.0.0.1:8093/v1"
    assert env["CASCADE_PERCEPTION_MODEL"] == "qwen3.8-27b"
    assert env["CASCADE_PERCEPTION_KEY_FILE"] == str(host.user / ".cascade" / "nemoclaw" / "llama.key")


def test_a_brain_port_flag_reaches_perception(host, monkeypatch):
    for name in PERCEPTION:
        monkeypatch.delenv(name, raising=False)
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    _, env = _serve(host, monkeypatch, "--instance", "sim", "serve-mcp", "--brain-port", "8095")
    assert env["CASCADE_PERCEPTION_URL"] == "http://127.0.0.1:8095/v1"


def test_the_brain_key_file_follows_the_instances_key_setting(host, monkeypatch, tmp_path):
    """$CASCADE_BRAIN_KEY_FILE (as the sandbox's provider reads it) and
    --brain-key-file pick the key FILE perception names; neither puts the
    key into the environment."""
    for name in PERCEPTION:
        monkeypatch.delenv(name, raising=False)
    host.install(host.user / ".cascade" / "openshell")
    other = tmp_path / "other.key"
    other.write_text("other-brain-key\n")
    monkeypatch.setenv("CASCADE_BRAIN_KEY_FILE", str(other))
    _, env = _serve(host, monkeypatch, "serve-mcp")
    assert env["CASCADE_PERCEPTION_KEY_FILE"] == str(other)
    flagged = tmp_path / "flagged.key"
    _, env = _serve(host, monkeypatch, "serve-mcp", "--brain-key-file", str(flagged))
    assert env["CASCADE_PERCEPTION_KEY_FILE"] == str(flagged)
    assert not [k for k, v in env.items() if "other-brain-key" in v]


def test_an_operators_or_launchers_perception_endpoint_wins(host, monkeypatch, tmp_path):
    """Explicit values win over the instance's brain: the unit's own
    environment, and the launcher's --env-file over that. An empty value
    means unset, as in configs/demo.yaml."""
    host.install(host.user / ".cascade" / "openshell")
    monkeypatch.setenv("CASCADE_PERCEPTION_URL", "http://10.0.0.7:8000/v1")
    monkeypatch.setenv("CASCADE_PERCEPTION_MODEL", "")
    monkeypatch.setenv("CASCADE_PERCEPTION_KEY_FILE", "/operator/brain.key")
    f = _env_file(tmp_path, {"CASCADE_PERCEPTION_KEY_FILE": "/launcher/brain.key"})
    _, env = _serve(host, monkeypatch, "serve-mcp", "--env-file", str(f))
    assert {name: env[name] for name in PERCEPTION} == {
        "CASCADE_PERCEPTION_URL": "http://10.0.0.7:8000/v1", "CASCADE_PERCEPTION_MODEL": "qwen3.8-27b",
        "CASCADE_PERCEPTION_KEY_FILE": "/launcher/brain.key"}


def test_teardown_stops_only_the_instances_own_pieces(host):
    """`launch.sh --down` of a sim run: chat forward, sandbox, robot server,
    pending re-lock and (with --gateway) the gateway -- all sim's."""
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    host.rig._write_state(home, motion_unlocked_until=2000.0)
    assert host.rig.main(["--instance", "sim", "teardown", "--gateway"]) == 0
    stopped = [a[3:] for a in host.argvs if a[:3] == ["systemctl", "--user", "stop"]]
    assert stopped == [["cascade-openshell-sim-chat"], ["cascade-openshell-sim-mcp"],
                       ["cascade-openshell-sim-relock.timer"], ["cascade-openshell-sim-relock.service"],
                       ["cascade-openshell-sim-gateway"]]
    assert [a[1:4] for a in host.argvs if a[0].endswith("/openshell")][-1] == ["sandbox", "delete", "cascade-sim-agent"]
    assert "motion_unlocked_until" not in host.rig._read_state(home)
    assert _targets(host.argvs).isdisjoint(_expected_targets(host.user / ".cascade" / "openshell", TODAY,
                                                             relock="motion lock"))


# ── the brain's context window reaches the sandbox's OpenClaw ─────────────

def test_render_sets_the_brains_context_window(tmp_path):
    """OpenClaw sizes its prompt from the model's contextWindow: a 32k brain
    (Cosmos, the Spark's Qwen) configured as 64k overflows mid-session."""
    assert _render(tmp_path, CASCADE_CONTEXT_WINDOW="32768").returncode == 0
    cfg = json.loads((tmp_path / ".openclaw" / "openclaw.json").read_text())
    assert cfg["models"]["providers"]["cascade"]["models"][0]["contextWindow"] == 32768


def test_render_keeps_the_templates_context_window_by_default(tmp_path):
    assert _render(tmp_path).returncode == 0
    cfg = json.loads((tmp_path / ".openclaw" / "openclaw.json").read_text())
    template = json.loads((DEPLOY / "openclaw.json").read_text())
    assert cfg["models"]["providers"]["cascade"]["models"][0]["contextWindow"] == \
        template["models"]["providers"]["cascade"]["models"][0]["contextWindow"] == 65536


@pytest.mark.parametrize("bad", ["32k", "1024", "4194304", ""])
def test_render_refuses_an_implausible_context_window(tmp_path, bad):
    r = _render(tmp_path, CASCADE_CONTEXT_WINDOW=bad)
    assert r.returncode != 0 and "CASCADE_CONTEXT_WINDOW" in r.stderr
    assert not (tmp_path / ".openclaw" / "openclaw.json").exists()


def test_a_non_default_context_window_rides_the_sandbox_env():
    rig = _load()
    argv = rig.sandbox_create_argv(rig.Settings(context_window=32768), image="t")
    assert "CASCADE_CONTEXT_WINDOW=32768" in argv
    assert not any(a.startswith("CASCADE_CONTEXT_WINDOW") for a in rig.sandbox_create_argv(rig.Settings(), image="t")), \
        "the default sandbox argv is unchanged"


def test_settings_saved_before_the_context_window_existed_are_not_stale(host):
    """The live rig's deployed.json predates `context_window`: reading it
    must not count as a settings change, which would recreate its sandbox
    (and rotate the chat token) on the next `up`."""
    rig = host.rig
    home = host.install(host.user / ".cascade" / "openshell")
    host.gateways_up.add(str(home / "0.1.3" / "bin" / "openshell"))
    old = {k: v for k, v in TODAY_SETTINGS.items() if k != "context_window"}
    rig._write_state(home, image=rig.image_tag(), settings=old)
    assert rig.main(["up"]) == 0
    assert not [a for a in host.argvs if a[0].endswith("/openshell") and a[1:3] in (["sandbox", "delete"],
                                                                                     ["sandbox", "create"])]
    assert rig.main(["up", "--context-window", "32768"]) == 0
    assert [a[1:3] for a in host.argvs if a[0].endswith("/openshell") and a[1] == "sandbox"
            and a[2] in ("delete", "create")] == [["sandbox", "delete"], ["sandbox", "create"]], \
        "a real change recreates it"


# ── `gateway start` on a fresh home (found live: 0.1.3's `status` exits 0) ──

def test_a_fresh_home_registers_its_gateway_before_calling_it_healthy(host, capsys):
    """0.1.3's `openshell status` exits 0 with "No gateway configured", so
    an exit code cannot say the gateway is up. A fresh instance must
    register ITS gateway (add sets it active) and see it Connected."""
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    assert host.rig.main(["--instance", "sim", "gateway", "start"]) == 0
    adds = [a[1:] for a in host.argvs if a[0].endswith("/openshell") and a[1:3] == ["gateway", "add"]]
    assert adds == [["gateway", "add", "https://127.0.0.1:17710", "--name", "cascade-sim", "--local"]]
    assert host.gateway_name(str(home / "0.1.3" / "bin" / "openshell")) == "cascade-sim"
    assert "gateway `cascade-sim` healthy on 127.0.0.1:17710" in capsys.readouterr().out


def test_another_active_gateway_is_not_this_instances(host):
    """A home whose CLI has another gateway active (NemoClaw's `openshell`,
    say) is not healthy for this instance: its own gateway is selected."""
    home = host.install(host.user / ".cascade" / "openshell-sim", "sim")
    binary = str(home / "0.1.3" / "bin" / "openshell")
    host.registered[binary] = "openshell"
    host.gateways_up.add(binary)
    answer = host.answer

    def two_registered(argv):
        if argv[0] == binary and argv[1:3] == ["gateway", "list"]:
            return 0, "  NAME  ENDPOINT\n* openshell  https://127.0.0.1:8080\n  cascade-sim  https://127.0.0.1:17710\n"
        if argv[0] == binary and argv[1:3] == ["gateway", "select"]:
            host.registered[binary] = argv[3]
        return answer(argv)
    host.answer = two_registered
    assert host.rig.main(["--instance", "sim", "gateway", "start"]) == 0
    assert [a[1:] for a in host.argvs if a[0] == binary and a[1:3] == ["gateway", "select"]] == [
        ["gateway", "select", "cascade-sim"]]
    assert not [a for a in host.argvs if a[0] == binary and a[1:3] == ["gateway", "add"]]


def test_up_refuses_a_home_with_no_gateway_registered(host):
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    with pytest.raises(SystemExit, match="gateway is down"):
        host.rig.main(["--instance", "sim", "up"])
    assert not [a for a in host.argvs if a[0].endswith("/openshell") and a[1] in ("provider", "sandbox")]


def test_status_reports_an_unregistered_gateway_as_down(host, capsys):
    host.install(host.user / ".cascade" / "openshell-sim", "sim")
    assert host.rig.main(["--instance", "sim", "status"]) == 0
    assert "gateway: down" in [line.strip() for line in capsys.readouterr().out.splitlines()]


# ── the sandboxed agent: the robot's tools, nothing else ──────────────────

def test_the_sandboxed_agent_sees_only_the_robots_tools():
    """Measured live (MuJoCo, Qwen3.8-27B, 2026-10-12): with OpenClaw's own
    tools exposed, "What is the session status?" called OpenClaw's
    `session_status` instead of cascade's `world_state`, and a two-call turn
    sent 73k prompt tokens; limited to the robot's tools it called
    `world_state` and sent 22.5k. The host route's dedicated profile has
    always set this allowlist (scripts/launch.sh)."""
    cfg = json.loads((DEPLOY / "openclaw.json").read_text())
    (server,) = cfg["mcp"]["servers"]
    assert cfg["tools"]["allow"] == [f"{server}__*"]
    assert "alsoAllow" not in cfg["tools"], "OpenClaw refuses allow and alsoAllow in one scope"
    assert cfg["agents"]["defaults"]["skills"] == [], "no general-purpose skills in the robot's prompt"


def test_the_rendered_config_keeps_the_allowlist(tmp_path):
    assert _render(tmp_path).returncode == 0
    cfg = json.loads((tmp_path / ".openclaw" / "openclaw.json").read_text())
    assert cfg["tools"]["allow"] == ["cascade__*"] and cfg["agents"]["defaults"]["skills"] == []


# 0.1.3's own words (captured 2026-10-12 from the real CLI on this host).
NO_GATEWAYS = "No gateways found.\n\nRegister a gateway with: openshell gateway add <endpoint>\n"
DOWN_STATUS = ("Error:   \u00d7 client error (Connect)\n  \u251c\u2500\u25b6 tcp connect error\n"
               "  \u2570\u2500\u25b6 Connection refused (os error 111)\n")
CONNECTED = "Server Status\n\n  Gateway: cascade-sim\n  Server: https://127.0.0.1:17710\n  Status: Connected\n"


def _done(argv, rc, out):
    return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")


def test_gateway_list_parses_only_its_table():
    """The empty answer is prose, not a table: "Register" is no gateway."""
    rig = _load()
    home = Path("/nonexistent")
    assert rig.registered_gateways(home, run=lambda argv, **kw: _done(argv, 0, NO_GATEWAYS)) == set()
    table = "  NAME         ENDPOINT                 TYPE   SOURCE  AUTH\n* cascade-sim  https://127.0.0.1:17710  local  user    mtls\n"
    assert rig.registered_gateways(home, run=lambda argv, **kw: _done(argv, 0, table)) == {"cascade-sim"}


@pytest.mark.parametrize("rc, out, healthy", [
    (0, CONNECTED, True),
    (1, DOWN_STATUS, False),                                          # registered, not running
    (0, "Gateway Status\n\n  Status: No gateway configured.\n", False),
    (0, CONNECTED.replace("cascade-sim", "openshell"), False),         # another gateway is active
    (0, CONNECTED.replace("Status: Connected", "Status: Connecting"), False),
])
def test_gateway_health_is_the_instance_named_and_connected(rc, out, healthy):
    rig = _load()
    inst = rig.instance("sim")
    assert rig.gateway_healthy(Path("/nonexistent"), inst, run=lambda argv, **kw: _done(argv, rc, out)) is healthy
