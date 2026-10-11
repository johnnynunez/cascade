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

import hashlib
import importlib.util
import json
import os
import re
import sys
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
