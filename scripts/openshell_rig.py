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
    scripts/openshell_rig.py open            # OpenClaw's web UI + the cascade camera extension
    scripts/openshell_rig.py chat-url        # http://127.0.0.1:<chat>/#token=... (secret!)
    scripts/openshell_rig.py motion lock     # observe-only: OpenShell denies every motion tool
    scripts/openshell_rig.py up --agent-profile franka   # the Franka FR3 operator agent (docs/FRANKA_OPENCLAW.md)
    scripts/openshell_rig.py status | down
    scripts/openshell_rig.py brain-profile --read-write   # rollback of the /v1-only brain profile (docs/OPENSHELL.md)

Everything this route owns lives under $CASCADE_OPENSHELL_HOME (default
~/.cascade/openshell, 0700), including its own XDG config/state: NemoClaw's
0.0.116 CLI keeps its gateway registry in ~/.config/openshell, and two CLI
generations sharing one registry would overwrite each other's active gateway.

The names, ports, units, provider ids and home above are the default
instance's. `--instance NAME` (or $CASCADE_OPENSHELL_INSTANCE) runs another,
isolated route next to it with its own of each (docs/OPENSHELL.md, Instances):

    scripts/openshell_rig.py --instance sim instance   # print its scheme, touch nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
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

# The default instance: the route exactly as it ran before instances existed
# (the live rig's). A named instance derives each of these from its name in
# `instance()`; everything else reaches them only through an `Instance`.
GATEWAY_NAME = "cascade"
GATEWAY_PORT = 17670
GATEWAY_UNIT = "cascade-openshell-gateway"
CHAT_UNIT = "cascade-openshell-chat"
#: The transient systemd user timer that runs `motion lock` at expiry.
RELOCK_UNIT = "cascade-openshell-relock"
#: The unit docs/OPENSHELL.md runs `serve-mcp` under (this script never starts it).
MCP_UNIT = "cascade-openshell-mcp"
SANDBOX_LABEL = "cascade-openshell"
BRAIN_PROVIDER = "cascade-brain"   # provider name = its profile id = its --type
ROBOT_PROVIDER = "cascade-robot"
HOME_NAME = "openshell"            # ~/.cascade/<HOME_NAME>

HOST_ALIAS = "host.openshell.internal"   # the supervisor maps it to the host's loopback
BRAIN_KEY_ENV = "CASCADE_BRAIN_API_KEY"
BRAIN_KEY_FILE_ENV = "CASCADE_BRAIN_KEY_FILE"   # the key FILE, for --brain-key-file's default
MCP_TOKEN_ENV = "CASCADE_MCP_TOKEN"
DEFAULT_BRAIN_PORT = 8081   # the authenticated brain (docs/OPENSHELL.md)
DEFAULT_MCP_PORT = 18741    # loopback, plain HTTP + bearer (NemoClaw's TLS listener is 18740)
DEFAULT_CHAT_PORT = 18799   # NemoClaw's dashboard forward holds 18789
DEFAULT_MODEL = "nemotron-3.5-bolt"   # scripts/serve_nemotron_bolt_llamacpp.sh's served id
#: The local brains whose served id also identifies objects for the robot
#: server (configs/demo.yaml `detector: type: brain`): the default brain and
#: Qwen3.8-27B (`--model qwen3.8-27b`, scripts/serve_qwen_vllm.sh).
LOCAL_BRAIN_MODELS = (DEFAULT_MODEL, "qwen3.8-27b")
DEFAULT_CONTEXT_WINDOW = 65536   # deploy/openshell/openclaw.json's contextWindow
DEFAULT_SANDBOX = "cascade-agent"   # NemoClaw's sandbox is `cascade`
#: The agent's robot profiles (deploy/openshell/render-config.mjs): `generic`
#: is the config every route rendered before profiles existed; `franka` adds
#: the FR3 operator command plugin, its instructions and tool catalog; `cosmos`
#: is the Cosmos3-Edge brain's (short prompt, no JSON envelopes, 10 tools).
AGENT_PROFILES = ("generic", "franka", "cosmos")
#: Runs a command after sourcing the ROS 2 + franka_ros2 setup (native-controller arms).
ROS_ENV = REPO / "scripts" / "ros_env.sh"
_CPU_LIST = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")
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
    #: the brain's context window, as the sandbox's OpenClaw must size prompts
    #: (deploy/openshell/openclaw.json carries this default)
    context_window: int = DEFAULT_CONTEXT_WINDOW
    #: the agent's robot profile (AGENT_PROFILES); a change recreates the sandbox
    agent_profile: str = "generic"


# ── instances: several isolated routes on one host ────────────────────────

INSTANCE_ENV = "CASCADE_OPENSHELL_INSTANCE"
#: OpenShell's gateway refuses a sandbox name longer than this (0.1.3:
#: MAX_ROUTABLE_NAME_LEN, a DNS label shared with workspace and service names)
#: or one that is not a-z, 0-9 and single '-' between them
#: (crates/openshell-server/src/grpc/validation.rs, validate_sandbox_name).
SANDBOX_NAME_MAX = 19
_SANDBOX_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: An instance name: a letter, then a-z, 0-9 and single '-' between them, so
#: that every name derived from it is one OpenShell takes (the sandbox above;
#: profile ids are '-'-separated runs of a-z, 0-9, so no '--' and no '-' last).
_INSTANCE_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


def named_sandbox(name: str) -> str:
    """A named instance's default sandbox."""
    return f"cascade-{name}-agent"


#: The longest instance name whose sandbox still fits SANDBOX_NAME_MAX (5).
INSTANCE_NAME_MAX = SANDBOX_NAME_MAX - len(named_sandbox(""))
#: Exit status of a name OpenShell would refuse (an instance or a sandbox name),
#: so launch.sh can tell it from any other failure of this script.
NAME_REFUSED = 3


class NameRefused(SystemExit):
    """An instance or sandbox name OpenShell would refuse; the message says why."""


#: A named instance's ports come from one of this many slots (sha256 of its name).
PORT_SLOTS = 99


@dataclass(frozen=True)
class Instance:
    """Everything that tells one route apart from another on a shared host.
    Provider ids are global per gateway and units per user, so a second
    route reusing any of these would update or stop the first one's."""
    name: str               # "" = the default instance
    gateway_name: str
    gateway_port: int
    gateway_unit: str
    chat_unit: str
    relock_unit: str
    mcp_unit: str
    sandbox_label: str
    brain_provider: str
    robot_provider: str
    sandbox: str            # default for --sandbox
    mcp_port: int           # default for --mcp-port
    chat_port: int          # default for --chat-port
    home_name: str          # ~/.cascade/<home_name>

    def home(self) -> Path:
        """Its state directory; CASCADE_OPENSHELL_HOME still overrides it."""
        return Path(os.environ.get("CASCADE_OPENSHELL_HOME", Path.home() / ".cascade" / self.home_name))

    def settings(self) -> Settings:
        """The settings `up` uses when neither a flag nor a saved value says otherwise."""
        return Settings(mcp_port=self.mcp_port, chat_port=self.chat_port, sandbox=self.sandbox)

    def cli(self) -> str:
        """How to address this instance in a hint to the operator."""
        return "scripts/openshell_rig.py" + (f" --instance {self.name}" if self.name else "")


def port_slot(name: str) -> int:
    """1..PORT_SLOTS, the same on every run and host (sha256, not hash())."""
    return 1 + int(hashlib.sha256(name.encode()).hexdigest(), 16) % PORT_SLOTS


def named_ports(slot: int) -> tuple[int, int, int]:
    """(gateway, MCP, chat) of a named instance: one band of 100 per role
    right above the default gateway (17671.., 17771.., 17871..), clear of
    the default route (17670, 18741, 18799), NemoClaw (18740, 18789) and
    OpenClaw (18790, 18791)."""
    return GATEWAY_PORT + slot, GATEWAY_PORT + 100 + slot, GATEWAY_PORT + 200 + slot


def instance(name: str = "") -> Instance:
    """The instance table. "" is the default instance: today's names,
    ports, units, ids and home. A named instance derives every one of them
    from its name (see docs/OPENSHELL.md, Instances)."""
    if not name:
        return Instance(name="", gateway_name=GATEWAY_NAME, gateway_port=GATEWAY_PORT,
                        gateway_unit=GATEWAY_UNIT, chat_unit=CHAT_UNIT, relock_unit=RELOCK_UNIT,
                        mcp_unit=MCP_UNIT, sandbox_label=SANDBOX_LABEL, brain_provider=BRAIN_PROVIDER,
                        robot_provider=ROBOT_PROVIDER, sandbox=DEFAULT_SANDBOX, mcp_port=DEFAULT_MCP_PORT,
                        chat_port=DEFAULT_CHAT_PORT, home_name=HOME_NAME)
    if name == "default" or len(name) > INSTANCE_NAME_MAX or not _INSTANCE_NAME.fullmatch(name):
        raise NameRefused(f"invalid instance name {name!r}: use 1-{INSTANCE_NAME_MAX} characters of a-z, 0-9 and "
                         "single '-' between them, starting with a letter (its sandbox "
                         f"{named_sandbox('<name>')} must fit OpenShell's {SANDBOX_NAME_MAX}-character "
                         "sandbox names); 'default' is reserved: omit --instance and CASCADE_OPENSHELL_INSTANCE "
                         "for the default instance")
    gateway_port, mcp_port, chat_port = named_ports(port_slot(name))
    return Instance(name=name, gateway_name=f"cascade-{name}", gateway_port=gateway_port,
                    gateway_unit=f"cascade-openshell-{name}-gateway", chat_unit=f"cascade-openshell-{name}-chat",
                    relock_unit=f"cascade-openshell-{name}-relock", mcp_unit=f"cascade-openshell-{name}-mcp",
                    sandbox_label=f"cascade-openshell-{name}", brain_provider=f"cascade-{name}-brain",
                    robot_provider=f"cascade-{name}-robot", sandbox=named_sandbox(name),
                    mcp_port=mcp_port, chat_port=chat_port, home_name=f"openshell-{name}")


def check_sandbox_name(name: str) -> None:
    """Refuse a sandbox name OpenShell would refuse: an explicit `up --sandbox`
    before the instance's home is claimed (main), a saved one before `up`
    touches anything (cmd_up)."""
    if len(name) > SANDBOX_NAME_MAX or not _SANDBOX_NAME.fullmatch(name):
        raise NameRefused(f"sandbox name {name!r}: OpenShell takes 1-{SANDBOX_NAME_MAX} characters of a-z, 0-9 "
                         "and single '-' between them; pass another --sandbox")


DEFAULT_INSTANCE = instance("")


def resolve_instance(cli: str | None, env: Mapping[str, str] | None = None) -> Instance:
    """--instance wins over $CASCADE_OPENSHELL_INSTANCE; empty or absent
    selects the default instance (so `--instance ''` overrides the env)."""
    environ = os.environ if env is None else env
    return instance(cli if cli is not None else environ.get(INSTANCE_ENV, ""))


def home_dir(inst: Instance = DEFAULT_INSTANCE) -> Path:
    return inst.home()


def claim_home(inst: Instance, home: Path) -> None:
    """One home, one instance. CASCADE_OPENSHELL_HOME overrides the home of
    any instance, so a stray export could point a named instance at the
    live route's state (its token, its lock, its gateway). A named instance
    marks its home with an `instance` file and only claims an empty home or
    its own; the default route (whose home carries no mark, as before)
    refuses a marked one."""
    marker = home / "instance"
    owner = marker.read_text().strip() if marker.is_file() else ""
    if owner == inst.name:
        return
    if owner or (inst.name and home.is_dir() and any(home.iterdir())):
        theirs = f"instance {owner!r}" if owner else "the default instance"
        ours = f"instance {inst.name!r}" if inst.name else "the default instance"
        raise SystemExit(f"{home} belongs to {theirs}, not to {ours}: unset CASCADE_OPENSHELL_HOME "
                         "or point it at this instance's own directory")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker.write_text(inst.name + "\n")


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


def gateway_toml(inst: Instance = DEFAULT_INSTANCE) -> str:
    """Schema v2 gateway config: Docker driver, the instance's loopback
    listener, sandboxes labelled apart from NemoClaw's and other instances'."""
    return (
        "[openshell]\n"
        "version = 2\n\n"
        "[openshell.gateway]\n"
        f'name = "{inst.gateway_name}"\n'
        'compute_driver = "docker"\n'
        f'bind_address = "127.0.0.1:{inst.gateway_port}"\n\n'
        "[openshell.drivers.docker]\n"
        f'sandbox_label = "{inst.sandbox_label}"\n'
    )


#: The sandbox reaches every brain only on its OpenAI API: GET and POST under
#: /v1/ (OpenClaw's chat completions and model list: render-config.mjs sets its
#: baseUrl to .../v1). Any other path is denied by OpenShell (REST rules, 0.1.3):
#: the Bolt warm start's GET /slots with every slot's prompt text and POST
#: /slots/N?action=save|restore|erase (scripts/serve_nemotron_bolt_llamacpp.sh),
#: llama-server's /props and /tokenize, vLLM's /metrics. The rules name no model:
#: a brain served under a custom ALIAS, Qwen or Cosmos gets the same profile
#: (#403 review, Minor 2). Every brain this profile describes is on this host's
#: loopback (HOST_ALIAS); launch_openshell.sh refuses any other.
OPENAI_API_METHODS = ("GET", "POST")


def brain_profile(port: int = DEFAULT_BRAIN_PORT, *, inst: Instance = DEFAULT_INSTANCE) -> dict:
    """Provider profile for the host brain: its bearer key is substituted only
    on requests to host.openshell.internal:<port> from the agent's node, and
    only on its /v1/* (OPENAI_API_METHODS), whatever model it serves.
    The brain itself is shared: every instance reaches the same host LLM."""
    access = {"rules": [{"allow": {"method": m, "path": "/v1/**"}} for m in OPENAI_API_METHODS]}
    return {
        "id": inst.brain_provider,
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
            **access, "enforcement": "enforce",
        }],
        "binaries": list(AGENT_BINARIES),
    }


def read_write_brain_profile(port: int = DEFAULT_BRAIN_PORT, *, inst: Instance = DEFAULT_INSTANCE) -> dict:
    """The brain profile main shipped before the /v1 path rules (#403's review): GET, HEAD, OPTIONS, POST, PUT and
    PATCH on every path of the brain (sha256 7e81c98a... for the default instance on 8081). Only the rollback
    `brain-profile --read-write` pushes it; the next `up` or `brain-profile` puts /v1-only back."""
    profile = brain_profile(port, inst=inst)
    endpoint = profile["endpoints"][0]
    del endpoint["rules"]
    endpoint["access"] = "read-write"
    return profile


def robot_profile(port: int | None = None, deny: tuple[str, ...] = DEFAULT_DENY_TOOLS, *,
                  inst: Instance = DEFAULT_INSTANCE) -> dict:
    """Provider profile for cascade's MCP server: the bearer token is
    substituted only on MCP requests to host.openshell.internal:<port>/mcp
    (default: the instance's MCP port), and OpenShell itself refuses
    `tools/call` of every denied tool (L7, 403 policy_denied) before the
    request leaves the sandbox."""
    port = inst.mcp_port if port is None else port
    return {
        "id": inst.robot_provider,
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


def sandbox_create_argv(cfg: Settings, image: str, *, inst: Instance = DEFAULT_INSTANCE) -> list[str]:
    """`openshell sandbox create` for the agent. Only non-secret values go
    through --env; both credentials ride the instance's providers."""
    env = {"CASCADE_BRAIN_PORT": cfg.brain_port, "CASCADE_MCP_PORT": cfg.mcp_port,
           "CASCADE_CHAT_PORT": cfg.chat_port, "CASCADE_MODEL": cfg.model}
    if cfg.context_window != DEFAULT_CONTEXT_WINDOW:   # the template's value needs no variable
        env["CASCADE_CONTEXT_WINDOW"] = cfg.context_window
    if cfg.agent_profile != "generic":   # the generic profile is the template's own
        env["CASCADE_AGENT_PROFILE"] = cfg.agent_profile
    argv = ["sandbox", "create", "--name", cfg.sandbox, "--from", image,
            "--provider", inst.brain_provider, "--provider", inst.robot_provider]
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


def ros_env_argv(argv: list[str]) -> list[str]:
    """`argv` inside the ROS 2 environment of a native-controller arm (the
    Franka FR3): scripts/ros_env.sh sources CASCADE_ROS_SETUP and
    CASCADE_FRANKA_SETUP and defaults ROS_DOMAIN_ID to 31, then execs it. A
    user unit inherits none of the launcher's shell, so the unit sources the
    same setup itself (the launcher's --env-file names the files)."""
    return ["/bin/bash", str(ROS_ENV), *argv]


def cpu_list(text: str) -> str:
    """A CPU list for systemd's CPUAffinity= (`0-7`, `0,2,4-6`), else refuse."""
    if not _CPU_LIST.fullmatch(str(text)):
        raise SystemExit(f"--cpus {text!r}: use a CPU list such as 0-7 or 0,2,4-6")
    return str(text)


def default_brain_key_file() -> Path:
    """$CASCADE_BRAIN_KEY_FILE, else NemoClaw's ~/.cascade/nemoclaw/llama.key:
    the brain's bearer key file (0600) when --brain-key-file is not given."""
    return Path(os.environ.get(BRAIN_KEY_FILE_ENV, str(Path.home() / ".cascade" / "nemoclaw" / "llama.key")))


def serve_mcp_env(base: dict | None = None, cfg: Settings = Settings(),
                  brain_key_file: Path | None = None) -> dict:
    """The server's environment, as nemoclaw_mcp.py serve and launch.sh build
    it: this checkout's src first, offline hub flags (explicit values kept),
    and no CASCADE_MCP_TOKEN, so the token file is the only source.

    The server identifies objects with the brain (configs/demo.yaml
    `detector: type: brain`). When this instance's saved brain is a local
    brain (`LOCAL_BRAIN_MODELS`: Nemotron 3.5 Bolt, the default, or Qwen3.8),
    it asks that brain: `cfg.brain_port`, its served id and the brain key
    FILE, each unless the launcher or the operator named it (an empty value
    counts as unset, as in demo.yaml). Any other saved brain (Cosmos, ...)
    sets none of the three, so demo.yaml's defaults apply and that brain
    never gets the grounding prompt; an operator's own values still reach
    the server. The key itself never enters the environment."""
    env = dict(os.environ if base is None else base)
    env.pop(MCP_TOKEN_ENV, None)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    env.setdefault("YOLO_OFFLINE", "True")
    env.setdefault("ULTRALYTICS_OFFLINE", "True")
    # Headless: a systemd user unit has DISPLAY=:0, and the default view=1
    # pops an OpenCV window next to the demo. The demo's screen is OpenClaw's
    # web UI with the camera extension (`open`), fed by the live-view server.
    env.setdefault("CASCADE_VIEW", "0")
    if cfg.model not in LOCAL_BRAIN_MODELS:
        return env
    brain = {"CASCADE_PERCEPTION_URL": f"http://127.0.0.1:{cfg.brain_port}/v1",
             "CASCADE_PERCEPTION_MODEL": cfg.model,
             "CASCADE_PERCEPTION_KEY_FILE": str(brain_key_file or default_brain_key_file())}
    for name, value in brain.items():
        if not env.get(name):
            env[name] = value
    return env


def serve_mcp_cwd() -> Path:
    """models/, the cwd every launch route gives the server (launch.sh runs
    the runtime there too). No longer needed for SAM: configs/demo.yaml loads
    the weights by absolute path (`${repo}/models/sam2.1_l.pt`)."""
    return REPO / "models"


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


def gateway_healthy(home: Path, inst: Instance, run=_default_run) -> bool:
    """This instance's gateway is the CLI's active one and Connected. 0.1.3's
    `openshell status` exits 0 even with no gateway registered ("No gateway
    configured", found live on a fresh instance home), so only its text tells."""
    r = _openshell(home, "status", run=run, check=False)
    lines = {line.strip() for line in (r.stdout or "").splitlines()}
    return f"Gateway: {inst.gateway_name}" in lines and "Status: Connected" in lines


def registered_gateways(home: Path, run=_default_run) -> set[str]:
    """Names in `openshell gateway list` (its table: NAME first, `*` = active)."""
    out = (_openshell(home, "gateway", "list", run=run, check=False).stdout or "").splitlines()
    if not out or "NAME" not in out[0]:
        return set()
    return {cols[0] for cols in (line.replace("*", " ").split() for line in out[1:]) if cols}


def _systemd_run(unit: str, argv: list[str], env: dict, log: Path, properties: tuple[str, ...] = ()) -> None:
    """A transient user unit (survives this shell; MemoryMax=infinity because
    agent shells on the Spark run in a memory-capped scope). `properties`
    adds unit properties (e.g. CPUAffinity=0-7)."""
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, check=False)
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True, check=False)
    cmd = ["systemd-run", "--user", f"--unit={unit}", "--collect", "-p", "UMask=0077",
           "-p", "MemoryMax=infinity", "-p", f"StandardOutput=append:{log}",
           "-p", f"StandardError=append:{log}"]
    for prop in properties:
        cmd += ["-p", prop]
    for k in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "OPENSHELL_LOCAL_TLS_DIR", "PATH", "HOME"):
        if k in env:
            cmd += ["-E", f"{k}={env[k]}"]
    subprocess.run(cmd + argv, check=True, capture_output=True, text=True)


def _unit_active(unit: str) -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], check=False).returncode == 0


#: Where `brain-profile --read-write` looks for the Bolt warm start (the tests point it at a fake tree).
PROC = Path("/proc")


def _argv_port(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--port="):
            return arg.split("=", 1)[1]
    return None


def _c_atoi(text: str) -> int:
    """C atoi(), as llama.cpp reads LLAMA_SERVER_SLOTS_DEBUG: leading digits, else 0."""
    m = re.match(r"\s*([+-]?\d+)", text)
    return int(m.group(1)) if m else 0


def warm_start_on(port: int) -> list[str]:
    """Why the brain on `port` runs with the Bolt warm start's wider API: one line per process (empty: it does not).

    The warm start (scripts/serve_nemotron_bolt_llamacpp.sh) runs llama-server with --slot-save-path and
    LLAMA_SERVER_SLOTS_DEBUG=1 beside scripts/brain_warm_start.py, every one of them with `--port <port>`. A
    process on that port whose environment cannot be read counts too: then nobody can tell."""
    found = []
    for d in PROC.iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = [a.decode(errors="replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:     # exited meanwhile
            continue
        if not argv or _argv_port(argv) != str(port):
            continue
        who = f"pid {d.name} ({Path(argv[0]).name})"
        if any(Path(a).name == "brain_warm_start.py" for a in argv):
            found.append(f"{who}: the warm-start helper")
            continue
        if any(a == "--slot-save-path" or a.startswith("--slot-save-path=") for a in argv):
            found.append(f"{who}: --slot-save-path")
            continue
        try:
            raw = (d / "environ").read_bytes()
        except PermissionError:
            found.append(f"{who}: its environment cannot be read, so LLAMA_SERVER_SLOTS_DEBUG cannot be ruled out")
            continue
        except OSError:
            continue
        env = dict(item.split("=", 1) for item in raw.decode(errors="replace").split("\0") if "=" in item)
        if _c_atoi(env.get("LLAMA_SERVER_SLOTS_DEBUG", "")):
            found.append(f"{who}: LLAMA_SERVER_SLOTS_DEBUG={env['LLAMA_SERVER_SLOTS_DEBUG']}")
        elif env.get("LLAMA_ARG_SLOT_SAVE_PATH"):
            found.append(f"{who}: LLAMA_ARG_SLOT_SAVE_PATH")
    return found


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


def _drop_state(home: Path, *keys: str) -> None:
    state = {k: v for k, v in _read_state(home).items() if k not in keys}
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = _state_file(home).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(_state_file(home))


def motion_locked(home: Path) -> bool:
    return bool(_read_state(home).get("motion_locked", False))


def set_motion_lock(home: Path, cfg: Settings, locked: bool, extra: tuple[str, ...] | None = None,
                    run=_default_run, *, inst: Instance = DEFAULT_INSTANCE) -> tuple[str, ...]:
    """Lock (observe-only) or unlock the arm at the OpenShell layer by
    updating the robot profile's deny rules. Live 0.1.3 propagates a profile
    update to the running sandbox within seconds, so neither the agent nor
    its chat token is recreated.

    `extra` (from `up --deny-tool`) is remembered: a later lock or unlock
    without it keeps denying those tools. An operator's deny is never dropped
    as a side effect of toggling motion."""
    if extra is None:
        extra = tuple(_read_state(home).get("deny_extra", ()))
    deny = deny_set(observe_only=locked, extra=extra)
    sync_profile(home, robot_profile(cfg.mcp_port, deny, inst=inst), run=run)
    _write_state(home, motion_locked=locked, deny=list(deny), deny_extra=list(extra))
    return deny


# ── timed unlock: motion permission always expires ────────────────────────

#: A `motion unlock` without --for lasts this long, then re-locks itself.
DEFAULT_UNLOCK_S = 20 * 60
#: No unlock may last longer: a presentation day (a presenter demos for hours,
#: and a re-lock mid-talk makes the agent refuse motion on stage), never a
#: standing permission -- a rig left running overnight is observe-only again.
MAX_UNLOCK_S = 12 * 3600
#: Set in the re-lock unit's environment, so `motion lock` there does not
#: stop its own service halfway through.
RELOCK_ENV = "CASCADE_OPENSHELL_RELOCK"


def parse_duration(text: str) -> int:
    """'90s' / '15m' / '1h' / '45' (seconds) -> seconds, in (0, MAX_UNLOCK_S].
    Zero, negative, unbounded ('forever') and over-the-cap values are refused:
    there is deliberately no way to unlock the arm without an expiry."""
    t = str(text).strip().lower()
    unit = {"s": 1, "m": 60, "h": 3600}.get(t[-1:], None)
    number = t[:-1] if unit else t
    try:
        value = int(number) * (unit or 1)
    except ValueError:
        raise SystemExit(f"unlock duration {text!r}: use e.g. 90s, 15m or 1h") from None
    if not 0 < value <= MAX_UNLOCK_S:
        raise SystemExit(f"unlock duration {text!r} must be > 0 and <= {MAX_UNLOCK_S // 3600} h")
    return value


def _cancel_relock(run=_default_run, *, inst: Instance = DEFAULT_INSTANCE) -> None:
    """Drop this instance's pending re-lock (never another instance's).
    Inside the re-lock unit itself (found live: stopping its own .service
    killed `motion lock` halfway), only the timer is touched; systemd
    collects the finished service."""
    unit = inst.relock_unit
    run(["systemctl", "--user", "stop", f"{unit}.timer"], capture_output=True, check=False)
    if os.environ.get(RELOCK_ENV) == "1":
        return
    run(["systemctl", "--user", "stop", f"{unit}.service"], capture_output=True, check=False)
    run(["systemctl", "--user", "reset-failed", f"{unit}.timer", f"{unit}.service"],
        capture_output=True, check=False)


def unlock_motion(home: Path, cfg: Settings, seconds: int = DEFAULT_UNLOCK_S, run=_default_run, *,
                  inst: Instance = DEFAULT_INSTANCE) -> tuple[str, ...]:
    """Allow motion for `seconds`, then re-lock automatically.

    The re-lock (a transient systemd timer running `motion lock`) is
    scheduled BEFORE the profile is unlocked: if it cannot be scheduled, the
    arm stays locked. A timer outlives this shell and a crashed agent; it
    does not outlive a reboot, after which `up` starts observe-only anyway.
    The timer re-enters this script with the same instance and home, so it
    re-locks exactly the instance that unlocked."""
    seconds = int(seconds)
    if not 0 < seconds <= MAX_UNLOCK_S:
        raise SystemExit(f"unlock duration must be > 0 and <= {MAX_UNLOCK_S} s")
    _cancel_relock(run, inst=inst)
    relock = [sys.executable, str(Path(__file__).resolve()), *(["--instance", inst.name] if inst.name else []),
              "motion", "lock"]
    r = run(["systemd-run", "--user", f"--unit={inst.relock_unit}", "--collect", f"--on-active={seconds}",
             "-E", f"CASCADE_OPENSHELL_HOME={home}", "-E", f"{RELOCK_ENV}=1", *relock],
            capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"could not schedule the automatic re-lock ({(r.stderr or '').strip()[:200]}); "
                         "motion stays LOCKED")
    deny = set_motion_lock(home, cfg, locked=False, run=run, inst=inst)
    _write_state(home, motion_unlocked_until=time.time() + seconds)
    return deny


def lock_motion(home: Path, cfg: Settings, run=_default_run, *, inst: Instance = DEFAULT_INSTANCE) -> tuple[str, ...]:
    """Observe-only now (first), then drop this instance's pending re-lock timer."""
    deny = set_motion_lock(home, cfg, locked=True, run=run, inst=inst)
    _cancel_relock(run, inst=inst)
    _drop_state(home, "motion_unlocked_until")
    return deny


def motion_status(home: Path) -> str:
    state = _read_state(home)
    if state.get("motion_locked", False):
        return "LOCKED (observe-only)"
    until = state.get("motion_unlocked_until")
    if until is None:
        return "allowed WITHOUT an expiry (an older unlock): run `motion lock`, or `motion unlock --for 20m`"
    left = until - time.time()
    if left <= 0:
        return "allowed, but the automatic re-lock is overdue: run `motion lock`"
    return f"allowed for {-(-int(left) // 60)} min more (then re-locked automatically)"


def startup_locked(home: Path, *, observe_only: bool, allow_motion: bool) -> bool:
    """`up` starts observe-only unless motion is asked for explicitly (and
    then it is a timed unlock). A saved unlock from an earlier session does
    not carry over: a fresh agent can look, not move."""
    return not allow_motion or observe_only


def _load_settings(home: Path, args, inst: Instance) -> Settings:
    """A flag wins over the saved value, which wins over the instance's default."""
    saved = _read_state(home).get("settings", {})
    base = Settings(**{**asdict(inst.settings()), **saved})
    pick = {f: getattr(args, f, None) for f in asdict(base)}
    return Settings(**{k: (v if v is not None else getattr(base, k)) for k, v in pick.items()})


# ── commands ──────────────────────────────────────────────────────────────


def cmd_install(inst: Instance, home: Path, args) -> int:
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
    conf.write_text(gateway_toml(inst))
    tls = Path(env["OPENSHELL_LOCAL_TLS_DIR"])
    if not (tls / "ca.crt").exists():  # mTLS between this CLI and this gateway only
        tls.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run([str(bindir / "openshell-gateway"), "generate-certs", "--output-dir", str(tls),
                        "--server-san", HOST_ALIAS], env=env, check=True)
    print(f"OpenShell {OPENSHELL_VERSION} in {bindir} (sha256 verified); gateway config {conf}")
    print(f"next: {inst.cli()} gateway start")
    return 0


def cmd_gateway(inst: Instance, home: Path, args) -> int:
    env = openshell_env(home)
    if args.action == "stop":
        subprocess.run(["systemctl", "--user", "stop", inst.gateway_unit], capture_output=True, check=False)
        print(f"{inst.gateway_unit} stopped")
        return 0
    gw = home / OPENSHELL_VERSION / "bin" / "openshell-gateway"
    if not gw.exists():
        raise SystemExit(f"OpenShell is not installed: {inst.cli()} install")
    if args.action == "start" and not _unit_active(inst.gateway_unit):
        _systemd_run(inst.gateway_unit, [str(gw)], env, home / "gateway.log")
    for _ in range(60):
        if gateway_healthy(home, inst):
            break
        # A fresh home has no gateway registered (`add` also selects it); a
        # home whose CLI has another one active needs ours selected.
        if inst.gateway_name not in registered_gateways(home):
            if _openshell(home, "gateway", "add", f"https://127.0.0.1:{inst.gateway_port}", "--name", inst.gateway_name,
                          "--local", check=False).returncode == 0:
                continue
        else:
            _openshell(home, "gateway", "select", inst.gateway_name, check=False)
        time.sleep(1)
    else:
        raise SystemExit(f"gateway not healthy; see {home / 'gateway.log'}")
    print(f"gateway `{inst.gateway_name}` healthy on 127.0.0.1:{inst.gateway_port} (unit {inst.gateway_unit})")
    return 0


def _replace_process(argv: list[str], env: dict) -> int:
    """exec(2): this process becomes `argv` (its output and exit code are the
    caller's), the way a wrapper must hand over a long-running command."""
    os.execve(argv[0], argv, env)
    return 0  # pragma: no cover


def read_env_file(path: Path) -> dict[str, str]:
    """A launcher's runtime environment for the robot server: one JSON
    object of strings, from a private file (never a unit's argv)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--env-file {path}: not a readable JSON object ({exc})") from None
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise SystemExit(f"--env-file {path}: must be one JSON object of string values")
    return data


def cmd_serve_mcp(inst: Instance, home: Path, args) -> int:
    cfg = _load_settings(home, args, inst)
    token_file = Path(args.mcp_token_file or home / "mcp.token")
    ensure_token(token_file)
    argv = serve_mcp_argv(cfg, token_file) + list(args.extra)
    if getattr(args, "ros_env", False):
        argv = ros_env_argv(argv)
    base = dict(os.environ)
    if args.env_file:
        base.update(read_env_file(Path(args.env_file)))
    serve_mcp_cwd().mkdir(exist_ok=True)
    os.chdir(serve_mcp_cwd())
    return _replace_process(argv, serve_mcp_env(base, cfg, Path(args.brain_key_file).expanduser()))


def _log_tail(log: Path, lines: int = 8) -> str:
    try:
        return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def cmd_mcp_start(inst: Instance, home: Path, args) -> int:
    """cascade's robot MCP server as this instance's transient user unit
    (`serve-mcp` under systemd-run), then wait until it answers. The
    launcher's runtime environment comes from --env-file; the arguments
    after `--` (its launch-owner flags) reach the server unchanged."""
    cfg = _load_settings(home, args, inst)
    runtime_env = read_env_file(Path(args.env_file)) if args.env_file else None
    unit = inst.mcp_unit
    if not args.replace and _unit_active(unit):
        raise SystemExit(f"{unit} is already running (another route's robot server?); stop it first, "
                         f"or pass --replace if this launcher owns it")
    token_file = Path(args.mcp_token_file or home / "mcp.token")
    token = ensure_token(token_file)
    saved = home / "mcp-env.json"
    saved.unlink(missing_ok=True)
    serve = [sys.executable, str(Path(__file__).resolve()), *(["--instance", inst.name] if inst.name else []),
             "serve-mcp", "--mcp-port", str(cfg.mcp_port), "--mcp-token-file", str(token_file)]
    if args.ros_env:
        serve.append("--ros-env")
    properties = (f"CPUAffinity={cpu_list(args.cpus)}",) if args.cpus else ()
    if runtime_env is not None:
        fd = os.open(saved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(runtime_env, f)
        serve += ["--env-file", str(saved)]
    log = home / "mcp-server.log"
    _systemd_run(unit, [*serve, "--", *args.extra] if args.extra else serve, openshell_env(home), log, properties)
    code: int | str = "not probed"
    for _ in range(max(1, int(args.wait))):
        code = _check_mcp(cfg, token)
        if code == 200:
            print(f"robot MCP server on 127.0.0.1:{cfg.mcp_port} (unit {unit}, log {log})")
            return 0
        if not _unit_active(unit):
            raise SystemExit(f"{unit} exited before answering; last lines of {log}:\n{_log_tail(log)}")
        time.sleep(1)
    raise SystemExit(f"{unit} did not answer within {int(args.wait)} s (last answer: {code}); see {log}")


def exec_argv(home: Path, cfg: Settings, command: list[str]) -> list[str]:
    """`openshell sandbox exec` of `command` in this instance's sandbox."""
    return [str(home / OPENSHELL_VERSION / "bin" / "openshell"), "sandbox", "exec", "--name", cfg.sandbox,
            "--no-tty", "--", *command]


def cmd_exec(inst: Instance, home: Path, args) -> int:
    """Run one command INSIDE the sandbox (the launcher's OpenClaw CLI for
    its proof): this process is replaced, so the caller sees the remote
    command's output and exit code."""
    if not args.extra:
        raise SystemExit(f"exec needs a command after --, e.g. {inst.cli()} exec -- openclaw health --json")
    cfg = _load_settings(home, args, inst)
    if not (home / OPENSHELL_VERSION / "bin" / "openshell").exists():
        raise SystemExit(f"OpenShell is not installed: {inst.cli()} install")
    return _replace_process(exec_argv(home, cfg, list(args.extra)), openshell_env(home))


def cmd_teardown(inst: Instance, home: Path, args) -> int:
    """Everything a launcher started for this instance: the chat forward,
    the sandbox, the robot MCP server, a pending re-lock, and with
    --gateway the gateway. Nothing of another instance."""
    cfg = _load_settings(home, args, inst)
    subprocess.run(["systemctl", "--user", "stop", inst.chat_unit], capture_output=True, check=False)
    if (home / OPENSHELL_VERSION / "bin" / "openshell").exists() and sandbox_phase(home, cfg.sandbox) is not None:
        _openshell(home, "sandbox", "delete", cfg.sandbox)
    subprocess.run(["systemctl", "--user", "stop", inst.mcp_unit], capture_output=True, check=False)
    _cancel_relock(inst=inst)
    _drop_state(home, "motion_unlocked_until")
    print(f"{inst.cli()}: chat forward, sandbox {cfg.sandbox}, robot server and re-lock stopped")
    if args.gateway:
        return cmd_gateway(inst, home, argparse.Namespace(action="stop"))
    return 0


def _require_gateway(home: Path, inst: Instance) -> None:
    """Refuse before any profile or sandbox call when this instance's gateway is not the active, Connected one."""
    if not gateway_healthy(home, inst):
        raise SystemExit(f"cascade's OpenShell gateway is down ({inst.gateway_name} on 127.0.0.1:{inst.gateway_port}): "
                         f"{inst.cli()} gateway start")


def cmd_up(inst: Instance, home: Path, args) -> int:
    cfg = _load_settings(home, args, inst)
    check_sandbox_name(cfg.sandbox)
    brain_key_file = Path(args.brain_key_file).expanduser()
    token_file = Path(args.mcp_token_file or home / "mcp.token")
    if not (home / OPENSHELL_VERSION / "bin" / "openshell").exists():
        raise SystemExit(f"OpenShell is not installed: {inst.cli()} install")
    _require_gateway(home, inst)
    # Preflight on the host loopback: what the supervisor will reach.
    brain = _check_brain(cfg, read_secret(brain_key_file))
    if brain != 200:
        raise SystemExit(f"brain on 127.0.0.1:{cfg.brain_port} answered {brain} with {brain_key_file}; "
                         "start it with that key file: scripts/serve_nemotron_bolt_llamacpp.sh, serve_qwen_vllm.sh "
                         "for --model qwen3.8-27b or serve_cosmos_sglang.sh for --model cosmos3-edge (which serves "
                         f"only with a key), each with API_KEY_FILE={brain_key_file}")
    mcp = _check_mcp(cfg, read_secret(token_file) if token_file.exists() else "")
    if mcp != 200:
        raise SystemExit(f"cascade MCP on 127.0.0.1:{cfg.mcp_port} answered {mcp}; start "
                         f"`{inst.cli()} serve-mcp` (under systemd-run, see docs/OPENSHELL.md)")
    tag = image_tag()
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False).returncode != 0:
        print(f"building {tag}")
        subprocess.run(["docker", "build", "-t", tag, str(DEPLOY)], check=True)
    sync_profile(home, brain_profile(cfg.brain_port, inst=inst))
    locked = startup_locked(home, observe_only=args.observe_only, allow_motion=args.allow_motion)
    if locked:
        deny = set_motion_lock(home, cfg, True, extra=tuple(args.deny_tool or ()), inst=inst)
        _cancel_relock(inst=inst)
        _drop_state(home, "motion_unlocked_until")
    else:
        _write_state(home, deny_extra=list(args.deny_tool or ()))
        deny = unlock_motion(home, cfg, parse_duration(args.for_) if args.for_ else DEFAULT_UNLOCK_S, inst=inst)
    sync_provider(home, inst.brain_provider, BRAIN_KEY_ENV, brain_key_file)
    sync_provider(home, inst.robot_provider, MCP_TOKEN_ENV, token_file)
    state = _read_state(home)
    phase = sandbox_phase(home, cfg.sandbox)
    # A state file older than a settings field holds that field's default.
    deployed = {**asdict(Settings()), **state.get("settings", {})} if "settings" in state else None
    stale = phase is not None and (args.recreate or state.get("image") != tag
                                   or deployed != asdict(cfg) or phase == "Error")
    if stale:
        print(f"recreating sandbox {cfg.sandbox} (phase {phase})")
        _openshell(home, "sandbox", "delete", cfg.sandbox)
        phase = None
    if phase is None:
        _openshell(home, *sandbox_create_argv(cfg, tag, inst=inst))
    for _ in range(90):
        phase = sandbox_phase(home, cfg.sandbox)
        if phase in ("Ready", "Error"):
            break
        time.sleep(2)
    if phase != "Ready":
        raise SystemExit(f"sandbox {cfg.sandbox} is {phase}; see `{home}/os logs {cfg.sandbox}`")
    _write_state(home, image=tag, settings=asdict(cfg))
    env = openshell_env(home)
    _systemd_run(inst.chat_unit, [str(home / OPENSHELL_VERSION / "bin" / "openshell"), *forward_argv(cfg)],
                 env, home / "chat-forward.log")
    print(f"sandbox {cfg.sandbox} Ready ({tag}); chat on http://127.0.0.1:{cfg.chat_port}/")
    print(f"motion: {'LOCKED by OpenShell (observe-only)' if locked else motion_status(home)}; "
          f"denied at L7: {', '.join(deny) if len(deny) < 4 else f'{len(deny)} tools'}")
    print(f"open it with the token: {inst.cli()} chat-url  (prints a secret)")
    return 0


def cmd_brain_profile(inst: Instance, home: Path, args) -> int:
    """Push the brain profile alone, live (the sandbox keeps running): the /v1-only one `up` pushes, or with
    --read-write the rollback to main's old read-write one. The rollback is refused while the Bolt warm start is on:
    read-write would give the sandbox GET /slots (every slot's prompt text) and slot save/restore/erase. Both check
    the gateway before pushing anything, as `up` does (the warm-start refusal reads only /proc, before any call)."""
    cfg = _load_settings(home, args, inst)
    if not (home / OPENSHELL_VERSION / "bin" / "openshell").exists():
        raise SystemExit(f"OpenShell is not installed: {inst.cli()} install")
    if not args.read_write:
        _require_gateway(home, inst)
        sync_profile(home, brain_profile(cfg.brain_port, inst=inst))
        print(f"{inst.brain_provider}: the sandbox reaches the brain on :{cfg.brain_port} only on GET/POST /v1/**")
        return 0
    if why := warm_start_on(cfg.brain_port):
        raise SystemExit(
            f"refusing read-write: the brain on :{cfg.brain_port} runs the Bolt warm start ({'; '.join(why)}). "
            "Read-write would give the sandbox GET /slots with every slot's prompt text and slot save/restore/erase. "
            "Restart the brain with WARM_START=0 first, then rerun.")
    _require_gateway(home, inst)
    sync_profile(home, read_write_brain_profile(cfg.brain_port, inst=inst))
    print(f"{inst.brain_provider}: read-write on every path of the brain on :{cfg.brain_port} (the old profile), "
          f"until the next `up` or `{inst.cli()} brain-profile`")
    return 0


def cmd_motion(inst: Instance, home: Path, args) -> int:
    cfg = _load_settings(home, args, inst)
    if args.action == "status":
        print(motion_status(home))
        return 0
    if args.action == "lock":
        deny = lock_motion(home, cfg, inst=inst)
    else:
        deny = unlock_motion(home, cfg, parse_duration(args.for_) if args.for_ else DEFAULT_UNLOCK_S, inst=inst)
    print(f"motion {motion_status(home)}: OpenShell denies {', '.join(deny)}")
    print("takes effect in the running sandbox within ~10 s; emergency_stop and halt_motion always work")
    return 0


def chat_url(home: Path, cfg: Settings, run=_default_run) -> str:
    """The tokenized OpenClaw chat URL. The token is minted inside the
    sandbox and read back through `sandbox exec`; it is a secret."""
    js = ("process.stdout.write(JSON.parse(require('fs').readFileSync("
          "process.env.HOME + '/.openclaw/openclaw.json')).gateway.auth.token)")
    r = _openshell(home, "sandbox", "exec", "--name", cfg.sandbox, "--", "node", "-e", js, run=run)
    token = (r.stdout or "").strip()
    if not token:
        raise SystemExit("no chat token yet; is the sandbox Ready?")
    return f"http://127.0.0.1:{cfg.chat_port}/#token={token}"


def open_chat(home: Path, cfg: Settings, run=_default_run, opener=None) -> None:
    """The demo's screen: OpenClaw's own web UI in the Chromium profile that
    loads the cascade camera extension (scripts/spark_browser.py, the same
    profile and extension every route uses). The extension finds the rig's
    cameras through the live-view server on this loopback page
    (extensions/chrome/discovery.mjs). The tokenized URL goes from the
    sandbox to the browser; it is never printed."""
    if opener is None:
        sys.path.insert(0, str(REPO / "scripts"))
        from spark_browser import open_browser as opener
    opener(REPO, url=chat_url(home, cfg, run=run))
    print(f"opened OpenClaw on 127.0.0.1:{cfg.chat_port} with the cascade camera extension "
          "(click the extension, then Connect cameras)")


def cmd_open(inst: Instance, home: Path, args) -> int:
    open_chat(home, _load_settings(home, args, inst))
    return 0


def cmd_chat_url(inst: Instance, home: Path, args) -> int:
    print(chat_url(home, _load_settings(home, args, inst)))
    return 0


def cmd_status(inst: Instance, home: Path, args) -> int:
    cfg = _load_settings(home, args, inst)
    installed = (home / OPENSHELL_VERSION / "bin" / "openshell").exists()
    rows = [("instance", f"{inst.name} ({home})")] if inst.name else []
    rows.append(("openshell", f"{OPENSHELL_VERSION} installed" if installed else "not installed"))
    if installed:
        gw = gateway_healthy(home, inst)
        rows.append(("gateway", f"healthy on 127.0.0.1:{inst.gateway_port}" if gw else "down"))
        if gw:
            rows.append(("sandbox", f"{cfg.sandbox}: {sandbox_phase(home, cfg.sandbox) or 'absent'}"))
            for p in (inst.brain_provider, inst.robot_provider):
                ok = _openshell(home, "provider", "get", p, check=False).returncode == 0
                rows.append((f"provider {p}", "present" if ok else "absent"))
    rows.append(("chat forward", f"127.0.0.1:{cfg.chat_port} ({'up' if _unit_active(inst.chat_unit) else 'down'})"))
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


def cmd_down(inst: Instance, home: Path, args) -> int:
    cfg = _load_settings(home, args, inst)
    subprocess.run(["systemctl", "--user", "stop", inst.chat_unit], capture_output=True, check=False)
    if sandbox_phase(home, cfg.sandbox) is not None:
        _openshell(home, "sandbox", "delete", cfg.sandbox)
    print(f"chat forward stopped, sandbox {cfg.sandbox} deleted")
    if args.gateway:
        return cmd_gateway(inst, home, argparse.Namespace(action="stop"))
    return 0


#: Commands that take a command or server arguments after `--`.
TRAILING_COMMAND = ("exec", "serve-mcp", "mcp-start")


def main(argv: list[str] | None = None) -> int:
    doc = __doc__ or ""
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    instance_help = (f"route instance (default ${INSTANCE_ENV}, else the default instance): its own home, "
                     "gateway, units, providers, sandbox and ports; see docs/OPENSHELL.md, Instances")
    ap.add_argument("--instance", default=None, metavar="NAME", help=instance_help)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        p = sub.add_parser(name, help=help_)
        # also accepted after the command; absent there, the global value stands
        p.add_argument("--instance", default=argparse.SUPPRESS, metavar="NAME", help=instance_help)
        return p

    d = DEFAULT_INSTANCE

    def settings_flags(p):
        p.add_argument("--sandbox", default=None, help=f"sandbox name (default: the instance's; {d.sandbox})")
        p.add_argument("--brain-port", type=int, default=None, help=f"host brain port (default {DEFAULT_BRAIN_PORT})")
        p.add_argument("--mcp-port", type=int, default=None, help=f"host MCP port (default: the instance's; {d.mcp_port})")
        p.add_argument("--chat-port", type=int, default=None,
                       help=f"host chat port (default: the instance's; {d.chat_port})")
        p.add_argument("--model", default=None, help=f"served model id (default {DEFAULT_MODEL})")
        p.add_argument("--context-window", type=int, default=None,
                       help=f"the brain's context window in tokens (default {DEFAULT_CONTEXT_WINDOW})")
        p.add_argument("--brain-key-file", default=str(default_brain_key_file()),
            help="0600 file with the brain's bearer key")
        p.add_argument("--mcp-token-file", default=None, help="0600 MCP token file (default <home>/mcp.token)")
        p.add_argument("--agent-profile", choices=AGENT_PROFILES, default=None,
                       help="the agent's profile (default generic; franka = the FR3 operator agent; "
                            "cosmos = the Cosmos3-Edge brain's compact agent)")

    add("install", "download + verify OpenShell, write gateway config and certs")
    g = add("gateway", "cascade's own OpenShell gateway (systemd user unit)")
    g.add_argument("action", choices=["start", "stop", "check"])
    for name, help_ in (("serve-mcp", "run cascade's MCP server on loopback (foreground)"),
                        ("up", "image, profiles, providers, sandbox, chat forward"),
                        ("open", "open OpenClaw's web UI with the cascade camera extension"),
                        ("chat-url", "print the chat URL with its token (a secret)"),
                        ("status", "read-only health of every piece"),
                        ("down", "stop the chat forward and delete the sandbox")):
        p = add(name, help_)
        settings_flags(p)
        if name == "up":
            p.add_argument("--recreate", action="store_true", help="delete and recreate the sandbox")
            p.add_argument("--deny-tool", action="append", help="extra MCP tool OpenShell refuses (repeatable)")
            lock = p.add_mutually_exclusive_group()
            lock.add_argument("--observe-only", action="store_true",
                              help="OpenShell denies every motion tool (lift with `motion unlock`)")
            lock.add_argument("--allow-motion", action="store_true",
                              help="start with motion allowed, for --for (default 20m, max 12h)")
            p.add_argument("--for", dest="for_", help="with --allow-motion: unlock duration (90s, 15m, 1h)")
        if name == "down":
            p.add_argument("--gateway", action="store_true", help="also stop the gateway")
        if name == "serve-mcp":
            p.add_argument("--env-file", help="JSON object of the server's runtime environment (0600)")
            p.add_argument("--ros-env", action="store_true",
                           help="run the server through scripts/ros_env.sh (ROS 2 + franka_ros2 setup)")
    ms = add("mcp-start", "serve-mcp as this instance's user unit, then wait until it answers; "
                          "`-- ARGS` reach the server (e.g. its launch-owner flags)")
    settings_flags(ms)
    ms.add_argument("--env-file", help="JSON object of the server's runtime environment (copied 0600 into the home)")
    ms.add_argument("--replace", action="store_true", help="restart the unit if it runs (only if you own it)")
    ms.add_argument("--wait", type=int, default=180, help="seconds to wait for the server to answer (default 180)")
    ms.add_argument("--ros-env", action="store_true",
                    help="the server sources the ROS 2 + franka_ros2 setup first (scripts/ros_env.sh)")
    ms.add_argument("--cpus", default=None, help="CPU affinity of the server's unit (e.g. 0-7)")
    ex = add("exec", "run `-- COMMAND` inside this instance's sandbox (the launcher's OpenClaw CLI)")
    settings_flags(ex)
    bp = add("brain-profile", "push the brain profile alone, live: /v1-only (what `up` pushes), or the --read-write "
                              "rollback")
    settings_flags(bp)
    bp.add_argument("--read-write", action="store_true",
                    help="rollback: main's old read-write brain profile until the next `up`; refused while the Bolt "
                         "warm start is on (restart the brain with WARM_START=0 first)")
    td = add("teardown", "stop the chat forward, delete the sandbox, stop the robot server and the re-lock")
    settings_flags(td)
    td.add_argument("--gateway", action="store_true", help="also stop the gateway")
    m = add("motion", "lock/unlock the arm at the OpenShell layer, live")
    m.add_argument("action", choices=["lock", "unlock", "status"])
    m.add_argument("--for", dest="for_", help="unlock duration (90s, 15m, 8h; default 20m, max 12h); "
                                             "every unlock re-locks itself")
    settings_flags(m)
    add("instance", "print this instance's names, ports, units and home (JSON); touches nothing")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        argv, extra = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    args = ap.parse_args(argv)
    if extra and args.cmd not in TRAILING_COMMAND:
        raise SystemExit(f"`{args.cmd}` takes no command after --")
    args.extra = extra
    inst = resolve_instance(args.instance)
    home = home_dir(inst)
    if args.cmd == "instance":
        print(json.dumps({**asdict(inst), "home": str(home)}, indent=2))
        return 0
    if args.cmd == "up" and args.sandbox is not None:
        check_sandbox_name(args.sandbox)    # a new instance's home is not claimed for a refused name
    claim_home(inst, home)
    return {"install": cmd_install, "gateway": cmd_gateway, "serve-mcp": cmd_serve_mcp, "up": cmd_up,
            "chat-url": cmd_chat_url, "open": cmd_open, "status": cmd_status, "down": cmd_down,
            "motion": cmd_motion, "mcp-start": cmd_mcp_start, "exec": cmd_exec,
            "teardown": cmd_teardown, "brain-profile": cmd_brain_profile}[args.cmd](inst, home, args)


def _cli() -> int:
    """main() from the command line: a refused name exits NAME_REFUSED with its reason."""
    try:
        return main()
    except NameRefused as refused:
        print(refused, file=sys.stderr)
        return NAME_REFUSED


if __name__ == "__main__":
    sys.exit(_cli())
