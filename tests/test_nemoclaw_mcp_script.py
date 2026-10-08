"""scripts/nemoclaw_mcp.py: the operator side of the NemoClaw/OpenShell profile.

The certificate material has to satisfy NemoClaw's documented admission
rules (a CA with basicConstraints CA:TRUE for NEMOCLAW_CORPORATE_CA_BUNDLE, a
server certificate whose IP SAN matches the private URL host), secrets must
be 0600 and never reach argv, and the registration must name the exact
trusted-private host. A server started by `serve` must answer MCP over TLS
verified against that CA.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import os
import shutil
import ssl
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "nemoclaw_mcp.py"
needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")


def _load():
    spec = importlib.util.spec_from_file_location("nemoclaw_mcp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path, *args, env=None, path_prefix=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("CASCADE_", "NEMOCLAW_"))}
    e["CASCADE_NEMOCLAW_DIR"] = str(tmp_path / "state")
    if path_prefix:
        e["PATH"] = f"{path_prefix}{os.pathsep}{e['PATH']}"
    e.update(env or {})
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=e,
                          timeout=120)


def _x509_text(path: Path) -> str:
    return subprocess.run(["openssl", "x509", "-noout", "-text", "-in", str(path)],
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("host", ["127.0.0.1", "8.8.8.8", "169.254.1.1", "::1", "fd00::1", "localhost"])
def test_only_routable_private_ipv4_hosts_are_accepted(host):
    mod = _load()
    with pytest.raises(SystemExit):
        mod.check_private(host)


@pytest.mark.parametrize("host", ["172.17.0.1", "10.0.0.5", "192.168.1.142", "100.99.118.73"])
def test_rfc1918_and_cgnat_hosts_pass(host):
    assert _load().check_private(host) == host


@needs_openssl
def test_certs_meet_nemoclaw_admission_rules(tmp_path):
    out = _run(tmp_path, "certs", "--host", "172.17.0.1", "--port", "18741")
    assert out.returncode == 0, out.stderr
    rec = json.loads(out.stdout)
    assert rec["url"] == "https://172.17.0.1:18741/mcp"
    d = tmp_path / "state"
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    for secret in ("ca.key", "server.key", "token"):
        assert stat.S_IMODE((d / secret).stat().st_mode) == 0o600, secret
    # NEMOCLAW_CORPORATE_CA_BUNDLE: not group/world writable, CA:TRUE only
    ca_mode = stat.S_IMODE((d / "ca.pem").stat().st_mode)
    assert not ca_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ca_text = _x509_text(d / "ca.pem")
    assert "CA:TRUE" in ca_text and "Certificate Sign" in ca_text
    assert (d / "ca.pem").read_text().count("BEGIN CERTIFICATE") == 1
    srv = _x509_text(d / "server.pem")
    assert "IP Address:172.17.0.1" in srv and "CA:FALSE" in srv and "TLS Web Server Authentication" in srv
    verify = subprocess.run(["openssl", "verify", "-CAfile", str(d / "ca.pem"), str(d / "server.pem")],
                            capture_output=True, text=True)
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert len((d / "token").read_text().strip()) >= 24
    assert _run(tmp_path, "ca-path").stdout.strip() == str(d / "ca.pem")


@needs_openssl
def test_reissuing_keeps_ca_and_token_unless_forced(tmp_path):
    assert _run(tmp_path, "certs", "--host", "10.1.2.3").returncode == 0
    d = tmp_path / "state"
    ca, token = (d / "ca.pem").read_text(), (d / "token").read_text()
    assert _run(tmp_path, "certs", "--host", "10.1.2.4").returncode == 0
    assert (d / "ca.pem").read_text() == ca and (d / "token").read_text() == token
    assert "IP Address:10.1.2.4" in _x509_text(d / "server.pem")
    assert _run(tmp_path, "certs", "--host", "10.1.2.4", "--force").returncode == 0
    assert (d / "ca.pem").read_text() != ca and (d / "token").read_text() != token


@needs_openssl
def test_register_names_the_exact_private_host_and_keeps_the_token_out_of_argv(tmp_path):
    assert _run(tmp_path, "certs", "--host", "172.17.0.1", "--port", "18742").returncode == 0
    token = (tmp_path / "state" / "token").read_text().strip()
    fake = tmp_path / "bin"
    fake.mkdir()
    record = tmp_path / "nemoclaw.json"
    (fake / "nemoclaw").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'token': os.environ.get('CASCADE_MCP_TOKEN')}}, open({str(record)!r}, 'w'))\n")
    (fake / "nemoclaw").chmod(0o755)
    out = _run(tmp_path, "register", "--sandbox", "robot-box", "--deny-tool", "reset_*", path_prefix=str(fake))
    assert out.returncode == 0, out.stderr
    got = json.loads(record.read_text())
    assert got["argv"] == ["robot-box", "mcp", "add", "cascade", "--url", "https://172.17.0.1:18742/mcp",
                           "--env", "CASCADE_MCP_TOKEN", "--trusted-private-host", "172.17.0.1",
                           "--deny-tool", "reset_*"]
    assert got["token"] == token
    assert token not in out.stderr and token not in out.stdout


def test_register_before_certs_fails_cleanly(tmp_path):
    out = _run(tmp_path, "register", "--sandbox", "x")
    assert out.returncode != 0 and "certs" in out.stderr


def test_configure_agent_sets_the_stdio_profile_parity_settings(tmp_path):
    """Measured on 8 Oct: without these the sandboxed agent got the image as
    base64 text (tool search), the model was text-only, and the 60 s default
    per-call timeout would cancel a 74 s pick (a cancelled motion latches the
    e-stop)."""
    fake = tmp_path / "bin"
    fake.mkdir()
    record = tmp_path / "calls.jsonl"
    (fake / "nemoclaw").write_text(
        "#!/usr/bin/env python3\nimport json, sys\n"
        f"open({str(record)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n")
    (fake / "nemoclaw").chmod(0o755)
    out = _run(tmp_path, "configure-agent", "--sandbox", "robot-box", path_prefix=str(fake))
    assert out.returncode == 0, out.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert all(c[:5] == ["robot-box", "exec", "--no-tty", "--", "openclaw"] for c in calls)
    settings = {c[7]: (c[8], "--strict-json" in c) for c in calls}
    assert settings == {
        "tools.toolSearch.enabled": ("false", False),
        "models.providers.inference.models[0].input": ('["text","image"]', True),
        "mcp.servers.cascade.requestTimeoutMs": ("300000", True),
        "mcp.servers.cascade.connectionTimeoutMs": ("120000", True),
    }
    # the per-call budget must cover a measured pick (74 s) with the launch.sh margin
    assert int(settings["mcp.servers.cascade.requestTimeoutMs"][0]) >= 300_000


@needs_openssl
def test_serve_answers_mcp_over_tls_verified_by_the_private_ca(tmp_path):
    """Loopback is not a NemoClaw-admissible host, so this binds the server
    on a private address only when the host has one; the TLS chain and the
    bearer are what is under test."""
    host = None
    for candidate in subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split():
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if ip.version == 4 and ip.is_private and not ip.is_loopback:
            host = candidate
            break
    if host is None:
        pytest.skip("no private IPv4 address on this host")
    port_file = tmp_path / "port"
    assert _run(tmp_path, "certs", "--host", host, "--port", "0").returncode == 0
    d = tmp_path / "state"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CASCADE_", "NEMOCLAW_"))}
    env.update(CASCADE_NEMOCLAW_DIR=str(d), CASCADE_CAMERA="mock", CASCADE_ARM="mock", CASCADE_STREAM="0",
               CASCADE_RUN_DIR=str(tmp_path / "run"), CASCADE_PREWARM="0", PYTHONPATH=str(REPO / "src"))
    mod = _load()
    argv = mod.server_argv(host, 0, d) + ["--http-port-file", str(port_file)]
    proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 60
        while not (port_file.exists() and port_file.read_text().strip()):
            assert proc.poll() is None, proc.stderr.read()
            assert time.monotonic() < deadline
            time.sleep(0.05)
        port = int(port_file.read_text())
        import http.client

        ctx = ssl.create_default_context(cafile=str(d / "ca.pem"))
        conn = http.client.HTTPSConnection(host, port, timeout=30, context=ctx)
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
        token = (d / "token").read_text().strip()
        conn.request("POST", "/mcp", body=body, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"})
        resp = conn.getresponse()
        msg = json.loads(resp.read())
        assert resp.status == 200 and resp.getheader("Mcp-Session-Id")
        assert msg["result"]["serverInfo"]["name"]
        assert token not in " ".join(argv)
    finally:
        proc.terminate()
        proc.wait(timeout=20)
