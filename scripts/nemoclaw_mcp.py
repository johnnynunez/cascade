#!/usr/bin/env python3
"""Serve CASCADE's robot tools to an agent sandboxed by NVIDIA NemoClaw/OpenShell.

Opt-in profile. Plain OpenClaw over stdio (scripts/launch.sh) stays the
default; this path exists so the agent can run inside an OpenShell sandbox
(Landlock + seccomp + netns, deny-by-default egress) while the robot side --
harness, perception, Isaac/MuJoCo bridge, GPU -- stays on the host.

NemoClaw admits only authenticated Streamable HTTP MCP endpoints. It rejects
loopback, host.openshell.internal and plain HTTP; a host-local server must be
HTTPS on a stable private address whose certificate chains to a CA the
sandbox trusts (NEMOCLAW_CORPORATE_CA_BUNDLE at onboarding). So:

    scripts/nemoclaw_mcp.py certs              # private CA + server cert + bearer token
    export NEMOCLAW_CORPORATE_CA_BUNDLE=$(scripts/nemoclaw_mcp.py ca-path)
    nemoclaw onboard                           # or `nemoclaw <sandbox> rebuild`
    scripts/nemoclaw_mcp.py serve              # the MCP server, HTTPS, bearer-gated
    scripts/nemoclaw_mcp.py register --sandbox my-assistant
    scripts/nemoclaw_mcp.py configure-agent --sandbox my-assistant
    nemoclaw my-assistant mcp status cascade --json

State lives in $CASCADE_NEMOCLAW_DIR (default ~/.cascade/nemoclaw), mode 0700;
keys and the token are 0600. The token never appears in argv or logs: `serve`
passes a file path, `register` hands it to `nemoclaw` through that one child's
environment, where NemoClaw moves it into the OpenShell provider store.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER_NAME = "cascade"
TOKEN_ENV = "CASCADE_MCP_TOKEN"
DEFAULT_PORT = 18740


def state_dir() -> Path:
    d = Path(os.environ.get("CASCADE_NEMOCLAW_DIR", Path.home() / ".cascade" / "nemoclaw"))
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def default_host() -> str:
    """The Docker bridge gateway: a stable RFC1918 address the OpenShell
    gateway container routes to, never published to the LAN."""
    override = os.environ.get("CASCADE_NEMOCLAW_HOST")
    if override:
        return override
    try:
        out = subprocess.run(["docker", "network", "inspect", "bridge", "--format",
                              "{{range .IPAM.Config}}{{.Gateway}} {{end}}"],
                             capture_output=True, text=True, timeout=20, check=True).stdout.split()
        for candidate in out:
            if ipaddress.ip_address(candidate).version == 4:
                return candidate
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    raise SystemExit("cannot find the Docker bridge gateway; pass --host <private IPv4> "
                     "or set CASCADE_NEMOCLAW_HOST")


def check_private(host: str) -> str:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise SystemExit(f"--host must be a private IPv4 literal (got {host!r}); NemoClaw pins the "
                         "certificate's IP SAN") from None
    cgnat = ipaddress.ip_network("100.64.0.0/10")
    if ip.version != 4 or ip.is_loopback or not (ip.is_private or ip in cgnat) or ip.is_link_local:
        raise SystemExit(f"{host} is not a routable private IPv4 address; NemoClaw rejects loopback, "
                         "link-local and public listeners for a trusted-private MCP server")
    return host


def _write_secret(path: Path, data: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(data)
    path.chmod(0o600)


def _openssl(*args: str) -> None:
    if shutil.which("openssl") is None:
        raise SystemExit("openssl CLI not found")
    subprocess.run(["openssl", *args], check=True, capture_output=True)


def cmd_certs(args) -> int:
    d = state_dir()
    host = check_private(args.host or default_host())
    ca_key, ca, key, cert, token = (d / n for n in ("ca.key", "ca.pem", "server.key", "server.pem", "token"))
    if not ca.exists() or args.force:
        _openssl("req", "-x509", "-newkey", "rsa:3072", "-nodes", "-days", "825", "-sha256",
                 "-subj", "/CN=CASCADE robot MCP private CA",
                 "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
                 "-addext", "keyUsage=critical,keyCertSign,cRLSign",
                 "-keyout", str(ca_key), "-out", str(ca))
        ca_key.chmod(0o600)
        ca.chmod(0o644)
    csr = d / "server.csr"
    ext = d / "server.ext"
    ext.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                   f"extendedKeyUsage=serverAuth\nsubjectAltName=IP:{host}\n")
    _openssl("req", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={host}", "-keyout", str(key), "-out", str(csr))
    key.chmod(0o600)
    _openssl("x509", "-req", "-in", str(csr), "-CA", str(ca), "-CAkey", str(ca_key), "-CAcreateserial",
             "-days", "397", "-sha256", "-extfile", str(ext), "-out", str(cert))
    csr.unlink(missing_ok=True)
    if not token.exists() or args.force:
        _write_secret(token, secrets.token_urlsafe(32) + "\n")
    (d / "endpoint.json").write_text(json.dumps({"host": host, "port": args.port}) + "\n")
    print(json.dumps({"state_dir": str(d), "ca": str(ca), "host": host, "port": args.port,
                      "url": f"https://{host}:{args.port}/mcp"}))
    return 0


def _endpoint(args) -> tuple[str, int]:
    saved = state_dir() / "endpoint.json"
    host, port = args.host, args.port
    if saved.exists():
        rec = json.loads(saved.read_text())
        host = host or rec["host"]
        port = port or rec["port"]
    if not host:
        raise SystemExit("no endpoint yet: run `nemoclaw_mcp.py certs` first")
    return check_private(host), int(port or DEFAULT_PORT)


def server_argv(host: str, port: int, d: Path) -> list[str]:
    return [sys.executable, "-m", "cascade.apps.mcp_server", "--http", f"{host}:{port}",
            "--tls-cert", str(d / "server.pem"), "--tls-key", str(d / "server.key"),
            "--token-file", str(d / "token")]


def cmd_serve(args) -> int:
    d = state_dir()
    host, port = _endpoint(args)
    for need in ("server.pem", "server.key", "token"):
        if not (d / need).exists():
            raise SystemExit(f"{d / need} missing: run `nemoclaw_mcp.py certs` first")
    env = dict(os.environ)
    env.pop(TOKEN_ENV, None)  # the file is the only source
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    env.setdefault("YOLO_OFFLINE", "True")
    env.setdefault("ULTRALYTICS_OFFLINE", "True")
    (REPO / "models").mkdir(exist_ok=True)
    os.chdir(REPO / "models")  # YOLOE resolves its text encoder relative to cwd (launch.sh)
    argv = server_argv(host, port, d)
    os.execvpe(argv[0], argv, env)
    return 0  # unreachable


def register_argv(sandbox: str, host: str, port: int, deny: list[str]) -> list[str]:
    argv = ["nemoclaw", sandbox, "mcp", "add", SERVER_NAME, "--url", f"https://{host}:{port}/mcp",
            "--env", TOKEN_ENV, "--trusted-private-host", host]
    for tool in deny:
        argv += ["--deny-tool", tool]
    return argv


# OpenClaw settings inside the sandbox that the stdio profile (launch.sh) already
# has and NemoClaw's generated config lacks; each was measured on 8 Oct 2026:
# - toolSearch on: MCP tools sit behind tool_search/tool_call, whose result is
#   JSON text capped at 16,000 chars, so get_observation's image arrived as
#   base64 text and the turn died with "Context overflow".
# - model input ["text"]: the llama-cpp provider registers the VL model as
#   text-only, so no image reaches it even with direct tools.
# - no requestTimeoutMs: OpenClaw's 60 s per-call default would cancel a
#   74 s pick, and a cancelled motion latches the e-stop (launch.sh: 300 s).
AGENT_SETTINGS = (
    ("tools.toolSearch.enabled", "false", False),
    ("models.providers.inference.models[0].input", '["text","image"]', True),
    (f"mcp.servers.{SERVER_NAME}.requestTimeoutMs", "300000", True),
    (f"mcp.servers.{SERVER_NAME}.connectionTimeoutMs", "120000", True),
)


def configure_argv(sandbox: str) -> list[list[str]]:
    out = []
    for path, value, strict in AGENT_SETTINGS:
        cmd = ["nemoclaw", sandbox, "exec", "--no-tty", "--", "openclaw", "config", "set", path, value]
        if strict:
            cmd.append("--strict-json")
        out.append(cmd)
    return out


def cmd_configure(args) -> int:
    rc = 0
    for argv in configure_argv(args.sandbox):
        print("+ " + " ".join(argv), file=sys.stderr)
        rc = rc or subprocess.run(argv).returncode
    return rc


def cmd_register(args) -> int:
    d = state_dir()
    host, port = _endpoint(args)
    token = (d / "token").read_text().strip()
    env = dict(os.environ)
    env[TOKEN_ENV] = token
    argv = register_argv(args.sandbox, host, port, args.deny_tool or [])
    print("+ " + " ".join(argv), file=sys.stderr)
    return subprocess.run(argv, env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("certs", help="create the private CA, server cert (IP SAN) and bearer token")
    c.add_argument("--host", help="private IPv4 to serve on (default: Docker bridge gateway)")
    c.add_argument("--port", type=int, default=DEFAULT_PORT)
    c.add_argument("--force", action="store_true", help="also rotate the CA and the token")
    s = sub.add_parser("serve", help="run the MCP server over HTTPS with the bearer token")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    r = sub.add_parser("register", help="`nemoclaw <sandbox> mcp add cascade ...` with the token")
    r.add_argument("--sandbox", required=True)
    r.add_argument("--host")
    r.add_argument("--port", type=int)
    r.add_argument("--deny-tool", action="append", help="tool name/glob blocked at the OpenShell MCP proxy")
    g = sub.add_parser("configure-agent", help="give the sandbox's OpenClaw the stdio profile's tool/image/timeout settings")
    g.add_argument("--sandbox", required=True)
    sub.add_parser("ca-path", help="print the CA bundle path for NEMOCLAW_CORPORATE_CA_BUNDLE")
    args = ap.parse_args(argv)
    if args.cmd == "certs":
        return cmd_certs(args)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "register":
        return cmd_register(args)
    if args.cmd == "configure-agent":
        return cmd_configure(args)
    print(state_dir() / "ca.pem")
    return 0


if __name__ == "__main__":
    sys.exit(main())
