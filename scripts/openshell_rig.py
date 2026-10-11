#!/usr/bin/env python3
"""OpenClaw in a standalone NVIDIA OpenShell 0.1.x sandbox, driving the real rig.

Opt-in profile, next to the NemoClaw one (scripts/nemoclaw_mcp.py). NemoClaw
pins OpenShell 0.0.116 and its managed `inference.local` route; OpenShell
0.1.x replaced that route with provider profiles, and this profile uses them
directly:

    agent (OpenClaw, in the sandbox)
      | sends an opaque placeholder: $CASCADE_BRAIN_API_KEY / $CASCADE_MCP_TOKEN
      v
    OpenShell supervisor (host network)
      | admits only the profile's endpoint for the profile's binaries,
      | swaps the placeholder for the real secret there, nowhere else,
      | and refuses MCP `tools/call reset_stop` at L7 before it leaves
      v
    host.openshell.internal:<brain>   (vLLM / llama-server on 127.0.0.1)
    host.openshell.internal:<mcp>/mcp (cascade's MCP server on 127.0.0.1)

On native Linux the supervisor maps host.openshell.internal to the host's
loopback, so neither service listens on a Docker bridge or the LAN (the
NemoClaw route needs 172.17.0.1 + TLS from a private CA). The real secrets
stay in 0600 files on the host and reach `openshell provider create|update`
through that one child's environment, never argv.

    scripts/openshell_rig.py install         # pinned + sha256-checked OpenShell, certs
    scripts/openshell_rig.py gateway start   # its own gateway, 127.0.0.1:17670
    scripts/openshell_rig.py serve-mcp       # cascade's MCP server on loopback (systemd it)
    scripts/openshell_rig.py up              # image, profiles, providers, sandbox, chat
    scripts/openshell_rig.py chat-url        # http://127.0.0.1:<chat>/#token=... (secret!)
    scripts/openshell_rig.py motion lock     # observe-only: OpenShell denies every motion tool
    scripts/openshell_rig.py status | down

Everything this route owns lives under $CASCADE_OPENSHELL_HOME (default
~/.cascade/openshell, 0700), including its own XDG config/state: NemoClaw's
0.0.116 CLI keeps its gateway registry in ~/.config/openshell, and two CLI
generations sharing one registry would overwrite each other's active gateway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "openshell"

OPENSHELL_VERSION = "0.1.3"
RELEASE_URL = f"https://github.com/NVIDIA/OpenShell/releases/download/v{OPENSHELL_VERSION}"
# From the release's openshell-checksums-sha256.txt and
# openshell-gateway-checksums-sha256.txt (v0.1.3). A digest mismatch refuses
# the install; bump these together with OPENSHELL_VERSION.
ARTIFACTS = {
    ("aarch64", "cli"): ("openshell-aarch64-unknown-linux-musl.tar.gz",
                         "07687dacefc52bc73b01943f5b58f7a5082c1ceaa228461b9a2a78023b8057c0"),
    ("aarch64", "gateway"): ("openshell-gateway-aarch64-unknown-linux-gnu.tar.gz",
                             "801214322227f64f2d3faa6989daaa5e6cd9ef1ab768b5860496deeb1158c27b"),
    ("x86_64", "cli"): ("openshell-x86_64-unknown-linux-musl.tar.gz",
                        "ed08ea7874c47f22d6e5dcc4b27d9ef1ef5d2060f83035b35ff725d5160830a3"),
    ("x86_64", "gateway"): ("openshell-gateway-x86_64-unknown-linux-gnu.tar.gz",
                            "da8dce943fe01cce90243b2bf2587c71a243e4d641a4b616528829b681ced198"),
}
BINARY = {"cli": "openshell", "gateway": "openshell-gateway"}

GATEWAY_NAME = "cascade"
GATEWAY_PORT = 17670
GATEWAY_UNIT = "cascade-openshell-gateway"
CHAT_UNIT = "cascade-openshell-chat"
SANDBOX_LABEL = "cascade-openshell"

HOST_ALIAS = "host.openshell.internal"   # the supervisor maps it to the host's loopback
BRAIN_KEY_ENV = "CASCADE_BRAIN_API_KEY"
MCP_TOKEN_ENV = "CASCADE_MCP_TOKEN"
DEFAULT_BRAIN_PORT = 8081   # the authenticated brain of docs/NEMOCLAW.md
DEFAULT_MCP_PORT = 18741    # loopback, plain HTTP + bearer (NemoClaw's TLS listener is 18740)
DEFAULT_CHAT_PORT = 18799   # NemoClaw's dashboard forward holds 18789
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_SANDBOX = "cascade-agent"   # NemoClaw's sandbox is `cascade`
SANDBOX_CHAT_PORT = 18789   # OpenClaw's gateway, on the sandbox's own loopback
DEFAULT_DENY_TOOLS = ("reset_stop",)  # clearing an e-stop is staff-only (BOOTH_RUNBOOK.md)
# Binaries allowed to use the credentials: OpenClaw runs on the image's node.
AGENT_BINARIES = ("/usr/local/bin/node",)
# MCP revisions OpenShell 0.1.x can inspect (policies/schema.mdx).
OPENSHELL_MCP_REVISIONS = ("2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28")
# initialize carries no MCP-Protocol-Version header (OpenShell reads it as
# 2025-03-26); cascade answers 2025-06-18; the OpenClaw SDK offers 2025-11-25.
ROBOT_MCP_REVISIONS = ("2025-03-26", "2025-06-18", "2025-11-25")


# ── pure pieces (tested) ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    brain_port: int = DEFAULT_BRAIN_PORT
    mcp_port: int = DEFAULT_MCP_PORT
    chat_port: int = DEFAULT_CHAT_PORT
    model: str = DEFAULT_MODEL
    sandbox: str = DEFAULT_SANDBOX


def home_dir() -> Path:
    return Path(os.environ.get("CASCADE_OPENSHELL_HOME", Path.home() / ".cascade" / "openshell"))


def artifact(machine: str, kind: str) -> tuple[str, str]:
    """(release asset name, pinned sha256) for this machine's CLI or gateway."""
    try:
        return ARTIFACTS[(machine, kind)]
    except KeyError:
        raise SystemExit(f"no pinned OpenShell {OPENSHELL_VERSION} {kind} build for {machine!r}; "
                         f"known: {sorted({m for m, _ in ARTIFACTS})}") from None


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"{Path(path).name}: sha256 {digest} != pinned {expected}; refusing to install")


def extract_binary(tarball: Path, member: str, dest: Path) -> Path:
    """Extract one regular file from a release tarball into dest (0755).
    An archive with any absolute, `..` or link member is refused whole."""
    with tarfile.open(tarball, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or ".." in p.parts or m.issym() or m.islnk():
                raise SystemExit(f"{Path(tarball).name}: unsafe member {m.name!r}; refusing to extract")
        wanted = [m for m in members if m.isfile() and PurePosixPath(m.name).parts[-1:] == (member,)
                  and len(PurePosixPath(m.name).parts) <= 2]
        if len(wanted) != 1:
            raise SystemExit(f"{Path(tarball).name}: expected one {member!r}, found {len(wanted)}")
        src = tf.extractfile(wanted[0])
        assert src is not None
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / member
        tmp = dest / f".{member}.part"
        tmp.write_bytes(src.read())
        tmp.chmod(0o755)
        tmp.replace(out)
        return out


def openshell_env(home: Path, base: dict | None = None) -> dict:
    """The environment every OpenShell CLI/gateway call of this route runs in:
    its own XDG config/state (never ~/.config/openshell, NemoClaw's) and the
    pinned binaries first on PATH."""
    env = dict(os.environ if base is None else base)
    env["XDG_CONFIG_HOME"] = str(home / "config")
    env["XDG_STATE_HOME"] = str(home / "state")
    env["OPENSHELL_LOCAL_TLS_DIR"] = str(home / "state" / "openshell" / "tls")
    env["PATH"] = os.pathsep.join(p for p in (str(home / OPENSHELL_VERSION / "bin"), env.get("PATH", "")) if p)
    return env


def gateway_toml() -> str:
    """Schema v2 gateway config: Docker driver, a 127.0.0.1:17670 listener,
    sandboxes labelled apart from NemoClaw's."""
    return (
        "[openshell]\n"
        "version = 2\n\n"
        "[openshell.gateway]\n"
        f'name = "{GATEWAY_NAME}"\n'
        'compute_driver = "docker"\n'
        f'bind_address = "127.0.0.1:{GATEWAY_PORT}"\n\n'
        "[openshell.drivers.docker]\n"
        f'sandbox_label = "{SANDBOX_LABEL}"\n'
    )


def brain_profile(port: int = DEFAULT_BRAIN_PORT) -> dict:
    """Provider profile for the host brain: its bearer key is substituted only
    on requests to host.openshell.internal:<port> from the agent's node."""
    return {
        "id": "cascade-brain",
        "display_name": "cascade brain (host OpenAI-compatible server)",
        "description": "Qwen3.8 on the host (vLLM or llama-server), bearer-protected, loopback-only",
        "category": "inference",
        "inference_capable": True,
        "credentials": [{
            "name": "api_key", "description": "bearer key of the host brain",
            "env_vars": [BRAIN_KEY_ENV], "required": True,
            "auth_style": "bearer", "header_name": "authorization",
        }],
        "discovery": {"credentials": ["api_key"]},
        "endpoints": [{
            "host": HOST_ALIAS, "port": int(port), "protocol": "rest",
            "access": "read-write", "enforcement": "enforce",
        }],
        "binaries": list(AGENT_BINARIES),
    }


def robot_profile(port: int = DEFAULT_MCP_PORT, deny: tuple[str, ...] = DEFAULT_DENY_TOOLS) -> dict:
    """Provider profile for cascade's MCP server: the bearer token is
    substituted only on MCP requests to host.openshell.internal:<port>/mcp,
    and OpenShell itself refuses `tools/call` of every denied tool (L7,
    403 policy_denied) before the request leaves the sandbox."""
    return {
        "id": "cascade-robot",
        "display_name": "cascade robot tools (MCP)",
        "description": "cascade's MCP server on the host loopback (arm + cameras), bearer-protected",
        "category": "other",
        "credentials": [{
            "name": "token", "description": "bearer token of cascade's MCP server",
            "env_vars": [MCP_TOKEN_ENV], "required": True,
            "auth_style": "bearer", "header_name": "authorization",
        }],
        "discovery": {"credentials": ["token"]},
        "endpoints": [{
            "host": HOST_ALIAS, "port": int(port), "path": "/mcp",
            "protocol": "mcp", "enforcement": "enforce",
            "mcp": {"versions": list(ROBOT_MCP_REVISIONS), "allow_all_known_mcp_methods": True},
            "deny_rules": [{"method": "tools/call", "tool": t} for t in dict.fromkeys(deny)],
        }],
        "binaries": list(AGENT_BINARIES),
    }


# Stops are never denied, whatever the lock: the agent can always halt the arm.
NEVER_DENY = frozenset({"emergency_stop", "halt_motion"})


def _assigned_set(path: Path, name: str) -> frozenset[str]:
    """A module-level `NAME = {...}` / `frozenset({...})` string set, read
    from source so this script needs neither the runtime nor its deps."""
    import ast

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
            value = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
            return frozenset(ast.literal_eval(value))
    raise SystemExit(f"{name} not found in {path}")


def motion_tools() -> frozenset[str]:
    """Every MCP tool that moves the arm: the runtime's motion skills plus the
    program tier's run_program (the same sets mcp_server serializes on)."""
    src = REPO / "src" / "cascade"
    return (_assigned_set(src / "skills" / "runtime.py", "_MOTION_SKILLS")
            | _assigned_set(src / "apps" / "mcp_server.py", "_PROGRAM_MOTION_TOOLS"))


def deny_set(observe_only: bool, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The tools OpenShell refuses at L7: always reset_stop, plus every motion
    tool when observe-only. A stop tool in the list is refused outright."""
    deny = set(DEFAULT_DENY_TOOLS) | set(extra) | (set(motion_tools()) if observe_only else set())
    if bad := sorted(deny & NEVER_DENY):
        raise SystemExit(f"refusing to deny {', '.join(bad)}: the agent must always be able to stop the arm")
    return tuple(sorted(deny))


def read_secret(path: Path) -> str:
    """A secret from a private file. Refuses missing, empty, or
    group/other-accessible files: the key must not be readable by anyone the
    sandbox boundary is meant to keep out."""
    path = Path(path).expanduser()
    try:
        st = path.stat()
    except FileNotFoundError:
        raise SystemExit(f"secret file missing: {path}") from None
    if st.st_mode & 0o077:
        raise SystemExit(f"{path} is mode {stat.S_IMODE(st.st_mode):04o}; chmod 0600 it first")
    value = path.read_text().strip()
    if not value:
        raise SystemExit(f"secret file {path} is empty")
    return value


def ensure_token(path: Path) -> str:
    """The MCP bearer token: minted once into a 0600 file, then kept."""
    path = Path(path)
    if path.exists():
        return read_secret(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_urlsafe(32) + "\n")
    return read_secret(path)


def image_tag(ctx: Path = DEPLOY) -> str:
    """cascade/openclaw-sandbox:<OpenClaw pin>-<hash of the build context>,
    so `up` rebuilds exactly when deploy/openshell changes."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(ctx).rglob("*") if p.is_file()):
        h.update(f.relative_to(ctx).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    pin = "unknown"
    for line in (Path(ctx) / "Dockerfile").read_text().splitlines():
        if line.startswith("ARG OPENCLAW_VERSION="):
            pin = line.split("=", 1)[1].strip()
    return f"cascade/openclaw-sandbox:{pin}-{h.hexdigest()[:12]}"


def sandbox_create_argv(cfg: Settings, image: str) -> list[str]:
    """`openshell sandbox create` for the agent. Only non-secret values go
    through --env; both credentials ride the providers."""
    env = {"CASCADE_BRAIN_PORT": cfg.brain_port, "CASCADE_MCP_PORT": cfg.mcp_port,
           "CASCADE_CHAT_PORT": cfg.chat_port, "CASCADE_MODEL": cfg.model}
    argv = ["sandbox", "create", "--name", cfg.sandbox, "--from", image,
            "--provider", "cascade-brain", "--provider", "cascade-robot"]
    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]
    return argv + ["--restart-policy", "on-failure", "--detach", "--", "cascade-openclaw"]


def forward_argv(cfg: Settings) -> list[str]:
    """The chat reaches the host through the gateway's gRPC forward, bound to
    127.0.0.1 only (never `--expose`, never the LAN)."""
    return ["forward", "service", cfg.sandbox, "--target-port", str(SANDBOX_CHAT_PORT),
            "--local", f"127.0.0.1:{cfg.chat_port}"]


def serve_mcp_argv(cfg: Settings, token_file: Path, python: str = sys.executable) -> list[str]:
    """cascade's MCP server on the host loopback, plain HTTP + bearer: the
    sandbox reaches it as host.openshell.internal through the supervisor."""
    return [python, "-m", "cascade.apps.mcp_server", "--http", f"127.0.0.1:{cfg.mcp_port}",
            "--token-file", str(token_file)]


# ── OpenShell calls ───────────────────────────────────────────────────────


def _default_run(argv, env=None, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("timeout", 300)
    # callers read returncode themselves (`_openshell(check=...)`)
    return subprocess.run(argv, env=env, check=kw.pop("check", False), **kw)


def _openshell(home: Path, *args: str, run=_default_run, extra_env: dict | None = None,
               check: bool = True, **kw) -> subprocess.CompletedProcess:
    env = openshell_env(home)
    env.update(extra_env or {})
    argv = [str(home / OPENSHELL_VERSION / "bin" / "openshell"), *args]
    r = run(argv, env=env, **kw)
    if check and r.returncode != 0:
        detail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-3:]
        raise SystemExit(f"openshell {' '.join(args[:3])} failed (rc {r.returncode}): {' | '.join(detail)}")
    return r


def sync_profile(home: Path, profile: dict, run=_default_run) -> None:
    """Import the profile on first use, update it afterwards. Live 0.1.3
    rules: import refuses an existing id, `lint` reports one as an error (so
    only a new profile is linted; `update` validates on the gateway), and
    update needs the stored profile's resource_version, read from `export`.
    JSON is valid YAML, so no YAML dependency is needed."""
    d = home / "profiles"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    f = d / f"{profile['id']}.yaml"
    cur = _openshell(home, "provider", "profile", "export", profile["id"], run=run, check=False)
    if cur.returncode == 0:
        version = next((int(line.split(":", 1)[1]) for line in (cur.stdout or "").splitlines()
                        if line.startswith("resource_version:")), 0)
        f.write_text(json.dumps({**profile, "resource_version": version}, indent=2) + "\n")
        _openshell(home, "provider", "profile", "update", "--file", str(f), profile["id"], run=run)
    else:
        f.write_text(json.dumps(profile, indent=2) + "\n")
        _openshell(home, "provider", "profile", "lint", "--file", str(f), run=run)
        _openshell(home, "provider", "profile", "import", "--file", str(f), run=run)


def sync_provider(home: Path, name: str, env_var: str, secret_file: Path, run=_default_run) -> None:
    """Create the provider, or update it so a rotated key lands. The secret
    reaches the CLI only as `env_var` in that one child's environment."""
    secret = {env_var: read_secret(secret_file)}
    if _openshell(home, "provider", "get", name, run=run, check=False).returncode == 0:
        _openshell(home, "provider", "update", name, "--from-existing", run=run, extra_env=secret)
    else:
        _openshell(home, "provider", "create", "--name", name, "--type", name, "--from-existing",
                   run=run, extra_env=secret)


def sandbox_phase(home: Path, name: str, run=_default_run) -> str | None:
    r = _openshell(home, "sandbox", "get", name, run=run, check=False)
    if r.returncode != 0:
        return None
    for line in (r.stdout or "").splitlines():
        if line.strip().startswith("Phase:"):
            return line.split(":", 1)[1].strip()
    return "Unknown"


# ── host-side helpers ─────────────────────────────────────────────────────


def machine() -> str:
    m = platform.machine().lower()
    return {"arm64": "aarch64", "amd64": "x86_64"}.get(m, m)


def _systemd_run(unit: str, argv: list[str], env: dict, log: Path) -> None:
    """A transient user unit (survives this shell; MemoryMax=infinity because
    agent shells on the Spark run in a memory-capped scope)."""
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, check=False)
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True, check=False)
    cmd = ["systemd-run", "--user", f"--unit={unit}", "--collect", "-p", "UMask=0077",
           "-p", "MemoryMax=infinity", "-p", f"StandardOutput=append:{log}",
           "-p", f"StandardError=append:{log}"]
    for k in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "OPENSHELL_LOCAL_TLS_DIR", "PATH", "HOME"):
        if k in env:
            cmd += ["-E", f"{k}={env[k]}"]
    subprocess.run(cmd + argv, check=True, capture_output=True, text=True)


def _unit_active(unit: str) -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], check=False).returncode == 0


def _http_status(url: str, headers: dict, body: bytes | None = None, timeout: float = 10) -> int | str:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError) as e:
        return f"down ({getattr(e, 'reason', e)})"


def _check_brain(cfg: Settings, key: str) -> int | str:
    return _http_status(f"http://127.0.0.1:{cfg.brain_port}/v1/models", {"Authorization": f"Bearer {key}"})


def _check_mcp(cfg: Settings, token: str) -> int | str:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "openshell_rig", "version": "1"}}}).encode()
    return _http_status(f"http://127.0.0.1:{cfg.mcp_port}/mcp",
                        {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream"}, body, timeout=30)


def _state_file(home: Path) -> Path:
    return home / "deployed.json"


def _read_state(home: Path) -> dict:
    f = _state_file(home)
    return json.loads(f.read_text()) if f.exists() else {}


def _write_state(home: Path, **fields) -> None:
    state = {**_read_state(home), **fields}
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = _state_file(home).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(_state_file(home))


def motion_locked(home: Path) -> bool:
    return bool(_read_state(home).get("motion_locked", False))


def set_motion_lock(home: Path, cfg: Settings, locked: bool, extra: tuple[str, ...] = (),
                    run=_default_run) -> tuple[str, ...]:
    """Lock (observe-only) or unlock the arm at the OpenShell layer by
    updating the robot profile's deny rules. Live 0.1.3 propagates a profile
    update to the running sandbox within seconds, so neither the agent nor
    its chat token is recreated."""
    deny = deny_set(observe_only=locked, extra=extra)
    sync_profile(home, robot_profile(cfg.mcp_port, deny), run=run)
    _write_state(home, motion_locked=locked, deny=list(deny))
    return deny


def _load_settings(home: Path, args) -> Settings:
    saved = _read_state(home).get("settings", {})
    base = Settings(**saved)
    pick = {f: getattr(args, f, None) for f in asdict(base)}
    return Settings(**{k: (v if v is not None else getattr(base, k)) for k, v in pick.items()})


# ── commands ──────────────────────────────────────────────────────────────


def cmd_install(home: Path, args) -> int:
    if sys.platform != "linux":
        raise SystemExit("this profile needs native Linux (Docker driver, host.openshell.internal -> loopback)")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home.chmod(0o700)
    bindir, dl = home / OPENSHELL_VERSION / "bin", home / OPENSHELL_VERSION / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    for kind in ("cli", "gateway"):
        name, digest = artifact(machine(), kind)
        blob = dl / name
        if not blob.exists() or hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
            print(f"downloading {name}")
            urllib.request.urlretrieve(f"{RELEASE_URL}/{name}", blob)
        verify_sha256(blob, digest)
        extract_binary(blob, BINARY[kind], bindir)
    env = openshell_env(home)
    conf = home / "config" / "openshell" / "gateway.toml"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(gateway_toml())
    tls = Path(env["OPENSHELL_LOCAL_TLS_DIR"])
    if not (tls / "ca.crt").exists():  # mTLS between this CLI and this gateway only
        tls.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run([str(bindir / "openshell-gateway"), "generate-certs", "--output-dir", str(tls),
                        "--server-san", HOST_ALIAS], env=env, check=True)
    print(f"OpenShell {OPENSHELL_VERSION} in {bindir} (sha256 verified); gateway config {conf}")
    print("next: scripts/openshell_rig.py gateway start")
    return 0


def cmd_gateway(home: Path, args) -> int:
    env = openshell_env(home)
    if args.action == "stop":
        subprocess.run(["systemctl", "--user", "stop", GATEWAY_UNIT], capture_output=True, check=False)
        print(f"{GATEWAY_UNIT} stopped")
        return 0
    gw = home / OPENSHELL_VERSION / "bin" / "openshell-gateway"
    if not gw.exists():
        raise SystemExit("OpenShell is not installed: scripts/openshell_rig.py install")
    if args.action == "start" and not _unit_active(GATEWAY_UNIT):
        _systemd_run(GATEWAY_UNIT, [str(gw)], env, home / "gateway.log")
    for _ in range(60):
        if _openshell(home, "status", check=False).returncode == 0:
            break
        if _openshell(home, "gateway", "add", f"https://127.0.0.1:{GATEWAY_PORT}", "--name", GATEWAY_NAME,
                      "--local", check=False).returncode == 0:
            continue
        time.sleep(1)
    else:
        raise SystemExit(f"gateway not healthy; see {home / 'gateway.log'}")
    print(f"gateway `{GATEWAY_NAME}` healthy on 127.0.0.1:{GATEWAY_PORT} (unit {GATEWAY_UNIT})")
    return 0


def cmd_serve_mcp(home: Path, args) -> int:
    cfg = _load_settings(home, args)
    token_file = Path(args.mcp_token_file or home / "mcp.token")
    ensure_token(token_file)
    argv = serve_mcp_argv(cfg, token_file)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    os.chdir(REPO)
    os.execvpe(argv[0], argv, env)
    return 0  # pragma: no cover


def cmd_up(home: Path, args) -> int:
    cfg = _load_settings(home, args)
    brain_key_file = Path(args.brain_key_file).expanduser()
    token_file = Path(args.mcp_token_file or home / "mcp.token")
    if not (home / OPENSHELL_VERSION / "bin" / "openshell").exists():
        raise SystemExit("OpenShell is not installed: scripts/openshell_rig.py install")
    if _openshell(home, "status", check=False).returncode != 0:
        raise SystemExit("cascade's OpenShell gateway is down: scripts/openshell_rig.py gateway start")
    # Preflight on the host loopback: what the supervisor will reach.
    brain = _check_brain(cfg, read_secret(brain_key_file))
    if brain != 200:
        raise SystemExit(f"brain on 127.0.0.1:{cfg.brain_port} answered {brain} with {brain_key_file}; "
                         "start it (scripts/serve_qwen_vllm.sh or serve_qwen_llamacpp.sh) with that key")
    mcp = _check_mcp(cfg, read_secret(token_file) if token_file.exists() else "")
    if mcp != 200:
        raise SystemExit(f"cascade MCP on 127.0.0.1:{cfg.mcp_port} answered {mcp}; start "
                         "`scripts/openshell_rig.py serve-mcp` (under systemd-run, see docs/OPENSHELL.md)")
    tag = image_tag()
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False).returncode != 0:
        print(f"building {tag}")
        subprocess.run(["docker", "build", "-t", tag, str(DEPLOY)], check=True)
    sync_profile(home, brain_profile(cfg.brain_port))
    locked = args.observe_only or (motion_locked(home) and not args.allow_motion)
    deny = set_motion_lock(home, cfg, locked, extra=tuple(args.deny_tool or ()))
    sync_provider(home, "cascade-brain", BRAIN_KEY_ENV, brain_key_file)
    sync_provider(home, "cascade-robot", MCP_TOKEN_ENV, token_file)
    state = _read_state(home)
    phase = sandbox_phase(home, cfg.sandbox)
    stale = phase is not None and (args.recreate or state.get("image") != tag
                                   or state.get("settings") != asdict(cfg) or phase == "Error")
    if stale:
        print(f"recreating sandbox {cfg.sandbox} (phase {phase})")
        _openshell(home, "sandbox", "delete", cfg.sandbox)
        phase = None
    if phase is None:
        _openshell(home, *sandbox_create_argv(cfg, tag))
    for _ in range(90):
        phase = sandbox_phase(home, cfg.sandbox)
        if phase in ("Ready", "Error"):
            break
        time.sleep(2)
    if phase != "Ready":
        raise SystemExit(f"sandbox {cfg.sandbox} is {phase}; see `{home}/os logs {cfg.sandbox}`")
    _write_state(home, image=tag, settings=asdict(cfg))
    env = openshell_env(home)
    _systemd_run(CHAT_UNIT, [str(home / OPENSHELL_VERSION / "bin" / "openshell"), *forward_argv(cfg)],
                 env, home / "chat-forward.log")
    print(f"sandbox {cfg.sandbox} Ready ({tag}); chat on http://127.0.0.1:{cfg.chat_port}/")
    print(f"motion: {'LOCKED by OpenShell (observe-only)' if locked else 'allowed'}; "
          f"denied at L7: {', '.join(deny) if len(deny) < 4 else f'{len(deny)} tools'}")
    print("open it with the token: scripts/openshell_rig.py chat-url  (prints a secret)")
    return 0


def cmd_motion(home: Path, args) -> int:
    cfg = _load_settings(home, args)
    if args.action == "status":
        print("LOCKED (observe-only)" if motion_locked(home) else "allowed")
        return 0
    deny = set_motion_lock(home, cfg, locked=args.action == "lock")
    print(f"motion {'LOCKED' if args.action == 'lock' else 'allowed'}: OpenShell denies {', '.join(deny)}")
    print("takes effect in the running sandbox within ~10 s")
    return 0


def cmd_chat_url(home: Path, args) -> int:
    cfg = _load_settings(home, args)
    js = ("process.stdout.write(JSON.parse(require('fs').readFileSync("
          "process.env.HOME + '/.openclaw/openclaw.json')).gateway.auth.token)")
    r = _openshell(home, "sandbox", "exec", "--name", cfg.sandbox, "--", "node", "-e", js)
    token = (r.stdout or "").strip()
    if not token:
        raise SystemExit("no chat token yet; is the sandbox Ready?")
    print(f"http://127.0.0.1:{cfg.chat_port}/#token={token}")
    return 0


def cmd_status(home: Path, args) -> int:
    cfg = _load_settings(home, args)
    installed = (home / OPENSHELL_VERSION / "bin" / "openshell").exists()
    rows = [("openshell", f"{OPENSHELL_VERSION} installed" if installed else "not installed")]
    if installed:
        gw = _openshell(home, "status", check=False).returncode == 0
        rows.append(("gateway", f"healthy on 127.0.0.1:{GATEWAY_PORT}" if gw else "down"))
        if gw:
            rows.append(("sandbox", f"{cfg.sandbox}: {sandbox_phase(home, cfg.sandbox) or 'absent'}"))
            for p in ("cascade-brain", "cascade-robot"):
                ok = _openshell(home, "provider", "get", p, check=False).returncode == 0
                rows.append((f"provider {p}", "present" if ok else "absent"))
    rows.append(("chat forward", f"127.0.0.1:{cfg.chat_port} ({'up' if _unit_active(CHAT_UNIT) else 'down'})"))
    rows.append(("motion", "LOCKED by OpenShell (observe-only)" if motion_locked(home) else "allowed"))
    key_file = Path(args.brain_key_file).expanduser()
    brain = _check_brain(cfg, read_secret(key_file)) if key_file.exists() else "no key file"
    rows.append(("brain", f"127.0.0.1:{cfg.brain_port} -> {brain}"))
    token_file = Path(args.mcp_token_file or home / "mcp.token")
    mcp = _check_mcp(cfg, read_secret(token_file)) if token_file.exists() else "no token yet"
    rows.append(("robot MCP", f"127.0.0.1:{cfg.mcp_port} -> {mcp}"))
    for k, v in rows:
        print(f"{k:>22}: {v}")
    return 0


def cmd_down(home: Path, args) -> int:
    cfg = _load_settings(home, args)
    subprocess.run(["systemctl", "--user", "stop", CHAT_UNIT], capture_output=True, check=False)
    if sandbox_phase(home, cfg.sandbox) is not None:
        _openshell(home, "sandbox", "delete", cfg.sandbox)
    print(f"chat forward stopped, sandbox {cfg.sandbox} deleted")
    if args.gateway:
        return cmd_gateway(home, argparse.Namespace(action="stop"))
    return 0


def main(argv: list[str] | None = None) -> int:
    doc = __doc__ or ""
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def settings_flags(p):
        p.add_argument("--sandbox", default=None, help=f"sandbox name (default {DEFAULT_SANDBOX})")
        p.add_argument("--brain-port", type=int, default=None, help=f"host brain port (default {DEFAULT_BRAIN_PORT})")
        p.add_argument("--mcp-port", type=int, default=None, help=f"host MCP port (default {DEFAULT_MCP_PORT})")
        p.add_argument("--chat-port", type=int, default=None, help=f"host chat port (default {DEFAULT_CHAT_PORT})")
        p.add_argument("--model", default=None, help=f"served model id (default {DEFAULT_MODEL})")
        p.add_argument("--brain-key-file", default=os.environ.get(
            "CASCADE_BRAIN_KEY_FILE", str(Path.home() / ".cascade" / "nemoclaw" / "llama.key")),
            help="0600 file with the brain's bearer key")
        p.add_argument("--mcp-token-file", default=None, help="0600 MCP token file (default <home>/mcp.token)")

    sub.add_parser("install", help="download + verify OpenShell, write gateway config and certs")
    g = sub.add_parser("gateway", help="cascade's own OpenShell gateway (systemd user unit)")
    g.add_argument("action", choices=["start", "stop", "check"])
    for name, help_ in (("serve-mcp", "run cascade's MCP server on loopback (foreground)"),
                        ("up", "image, profiles, providers, sandbox, chat forward"),
                        ("chat-url", "print the chat URL with its token (a secret)"),
                        ("status", "read-only health of every piece"),
                        ("down", "stop the chat forward and delete the sandbox")):
        p = sub.add_parser(name, help=help_)
        settings_flags(p)
        if name == "up":
            p.add_argument("--recreate", action="store_true", help="delete and recreate the sandbox")
            p.add_argument("--deny-tool", action="append", help="extra MCP tool OpenShell refuses (repeatable)")
            lock = p.add_mutually_exclusive_group()
            lock.add_argument("--observe-only", action="store_true",
                              help="OpenShell denies every motion tool (lift with `motion unlock`)")
            lock.add_argument("--allow-motion", action="store_true", help="clear a saved observe-only lock")
        if name == "down":
            p.add_argument("--gateway", action="store_true", help="also stop the gateway")
    m = sub.add_parser("motion", help="lock/unlock the arm at the OpenShell layer, live")
    m.add_argument("action", choices=["lock", "unlock", "status"])
    settings_flags(m)
    args = ap.parse_args(argv)
    home = home_dir()
    return {"install": cmd_install, "gateway": cmd_gateway, "serve-mcp": cmd_serve_mcp, "up": cmd_up,
            "chat-url": cmd_chat_url, "status": cmd_status, "down": cmd_down, "motion": cmd_motion}[args.cmd](home, args)


if __name__ == "__main__":
    sys.exit(main())
