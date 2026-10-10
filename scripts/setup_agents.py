#!/usr/bin/env python3
"""Register the cascade MCP server with every supported agent platform.

One robot tool-server, many brains. Supported hosts:

    hermes    ~/.hermes/config.yaml           (mcp_servers.<name>, YAML)
    codex     $CODEX_HOME/config.toml         ([mcp_servers.<name>], TOML; or a
              `codex -p NAME` layer NAME.config.toml via --codex-profile NAME)
    claude    <repo>/.mcp.json (project)      (Claude Code; auto-detected)
              + `claude mcp add` one-liner for user scope / Desktop JSON
    openclaw  `openclaw mcp set` one-liner    (native mcp.servers) or a
              Claude-Desktop-style JSON block (also valid for mcporter)

By default this PRINTS what each host needs; `--write` applies the ones
that can be safely edited in place (hermes YAML, codex TOML, project
.mcp.json), always preserving unrelated entries.

    python scripts/setup_agents.py                          # print all
    python scripts/setup_agents.py --host codex --write
    python scripts/setup_agents.py --host codex --codex-profile robot --write   # codex -p robot
    python scripts/setup_agents.py --camera d455f --arm rebot_rs --write

Runtime switches: every variable `cascade.apps.mcp_env.FORWARDED` names that
is set in this shell is copied verbatim into every entry's env -- the same
list scripts/launch.sh registers (B63): memory paths, endpoints
(CASCADE_BRIDGE_PORT, CASCADE_GRASPGENX_PORT/_HOST, CASCADE_OCCUPANCY_PORT,
CASCADE_HUG_PORT/_HOST, CASCADE_STREAM_PORT, ...), devices, the grasp
executor and its policy port (CASCADE_GRASP_EXECUTOR, CASCADE_VLA_PORT),
CASCADE_BOOTH, ... A stdio host may start the server without this shell's
environment, so a switch that is not copied silently does not apply. Nothing
is invented (an unset variable is not written; an empty one is copied empty),
the copied names are printed, and rig selectors (CASCADE_ROBOT, CASCADE_BASE,
CASCADE_ARMS), per-process values and secrets (CASCADE_MCP_TOKEN) are never
copied (`NOT_FORWARDED` says why for each). A value the runtime would refuse
stops the registration naming the variable, before any file is written: a
*_PORT must be ASCII digits in 1..65535, a *_HOST a hostname or IPv4 address
(cascade.config.env_port / env_host; an interpreter that cannot import it is
refused too -- use the repo's Python). `--env KEY=VALUE` (and
--detect-classes / --hide-tools) beat the inherited value and are written as
given.

    CASCADE_BRIDGE_PORT=45311 CASCADE_GRASP_EXECUTOR=vla CASCADE_VLA_PORT=8000 \\
        python scripts/setup_agents.py --host hermes --write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from setup_hermes import upsert as hermes_upsert  # noqa: E402
from setup_hermes import yaml_block as hermes_yaml_block  # noqa: E402

# The shared .demo uv venv sits next to the repo checkout on every rig
# (…/Projects/demo/.demo), so derive it from the repo location instead of
# hardcoding one machine's home.
DEFAULT_PY = str(REPO.parent / ".demo" / "bin" / "python")
SERVER = "cascade"
if str(REPO / "src") not in sys.path:  # this checkout's registry (stdlib only; no PyYAML needed)
    sys.path.insert(0, str(REPO / "src"))
from cascade.apps.mcp_env import FORWARDED, forwarded_env  # noqa: E402


def inherited_env(skip=()) -> dict[str, str]:
    """Every variable cascade.apps.mcp_env forwards that is set in this environment, verbatim.

    The same list scripts/launch.sh registers (B63): memory paths, endpoints,
    devices, the grasp executor and its policy port, booth/stream/view
    switches, ... -- see `FORWARDED`. Only variables that are present are
    returned, never a default; an empty one stays empty. Each *_PORT / *_HOST
    is checked with the runtime's own rule first -- `cascade.config.env_port`
    / `env_host` -- so a value the server would refuse at startup (or one that
    would break the YAML/TOML blocks) raises ValueError naming the variable
    instead of being written into a host's config; so does an interpreter that
    cannot import cascade.config (no PyYAML) when there is one to check, as
    nothing is written unchecked. Names in `skip` (given explicitly with
    --env) are neither copied nor checked.
    """
    env = forwarded_env(skip=skip)
    names = [name for name in env if name.endswith(("_PORT", "_HOST"))]
    if not names:
        return env
    try:
        from cascade.config import env_host, env_port
    except ImportError as e:  # e.g. a bare python3 without PyYAML: never write unchecked
        raise ValueError(
            f"cannot check {' '.join(names)}: cascade.config does not import with "
            f"{sys.executable} ({e}); run this script with the repo's Python, or pass "
            "the values with --env KEY=VALUE") from e

    for name in names:
        (env_port if name.endswith("_PORT") else env_host)(name)
    return env


def server_env(
    camera: str,
    arm: str,
    display: str,
    detect_classes: str | None = None,
    offline: bool = True,
    hide_tools: str | None = None,
    extra: list[str] | None = None,
) -> dict[str, str]:
    env = {
        "PYTHONPATH": str(REPO / "src"),
        "CASCADE_CAMERAS": camera,
        "CASCADE_ARM": arm,
        "DISPLAY": display,
    }
    # Stdio hosts may discard their inherited environment, so every runtime
    # switch set in this shell is copied: the same list launch.sh registers
    # (cascade.apps.mcp_env, see inherited_env). Device selection stays
    # literal: empty visibility disables CUDA, and UUIDs / ordinal lists must
    # not be rewritten into a different device space. An explicit --env value
    # wins below and leaves the inherited one unread; so do --detect-classes
    # and --hide-tools.
    env.update(inherited_env(skip={kv.partition("=")[0].strip() for kv in extra or []}))
    if offline:
        # without these, ultralytics phones GitHub on class re-embeds and
        # stalls the perception watcher for seconds -- never at a venue
        env["YOLO_OFFLINE"] = "True"
        env["ULTRALYTICS_OFFLINE"] = "True"
    if detect_classes:
        env["CASCADE_DETECT_CLASSES"] = detect_classes
    if hide_tools:
        env["CASCADE_HIDE_TOOLS"] = hide_tools
    for kv in extra or []:
        k, _, v = kv.partition("=")
        env[k.strip()] = v
    return env


# ── Codex CLI (TOML) ─────────────────────────────────────────────────────

CODEX_START = f"[mcp_servers.{SERVER}]"
CODEX_ENV_START = f"[mcp_servers.{SERVER}.env]"
# `codex -p NAME` layers $CODEX_HOME/NAME.config.toml; keep the name a plain
# token so it cannot escape CODEX_HOME or collide with the `.config.toml` suffix.
_CODEX_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def codex_home_dir() -> Path:
    """`$CODEX_HOME`, else `~/.codex` -- the same resolution the CLI uses."""
    return Path(os.environ.get("CODEX_HOME", "").strip() or Path.home() / ".codex")


def codex_config_path(profile: str | None = None, codex_home: Path | None = None) -> Path:
    """Where the `[mcp_servers.cascade]` block goes.

    Without a profile: the base `config.toml`, loaded by every Codex session.
    With `--codex-profile NAME`: `NAME.config.toml`, which Codex layers only
    under `codex -p NAME` -- a robot tool server should not be loaded into
    every coding session."""
    home = Path(codex_home) if codex_home is not None else codex_home_dir()
    if profile is None:
        return home / "config.toml"
    if not _CODEX_PROFILE_RE.match(profile):
        raise ValueError(f"--codex-profile must match [A-Za-z0-9_-]+, got {profile!r}")
    return home / f"{profile}.config.toml"


def codex_toml_block(python: str, env: dict[str, str]) -> str:
    args = ", ".join(f'"{a}"' for a in ["-m", "cascade.apps.mcp_server"])
    lines = [CODEX_START, f'command = "{python}"', f"args = [{args}]", CODEX_ENV_START]
    lines += [f'{k} = "{v}"' for k, v in env.items()]
    return "\n".join(lines) + "\n"


def codex_upsert(existing: str | None, block: str) -> str:
    """Replace our `[mcp_servers.cascade]` tables, preserve everything else."""
    if not existing or not existing.strip():
        return block
    lines = existing.split("\n")
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped in (CODEX_START, CODEX_ENV_START):
            skipping = True
            continue
        if skipping and stripped.startswith("[") and stripped not in (CODEX_START, CODEX_ENV_START):
            skipping = False
        if not skipping:
            out.append(line)
    base = "\n".join(out).rstrip()
    return (base + "\n\n" if base else "") + block


# ── Claude Code / Desktop (JSON) ─────────────────────────────────────────


def claude_json_entry(python: str, env: dict[str, str]) -> dict:
    return {
        "type": "stdio",
        "command": python,
        "args": ["-m", "cascade.apps.mcp_server"],
        "env": env,
    }


def claude_mcp_json(existing: str | None, python: str, env: dict[str, str]) -> str:
    data = json.loads(existing) if existing and existing.strip() else {}
    data.setdefault("mcpServers", {})[SERVER] = claude_json_entry(python, env)
    return json.dumps(data, indent=2) + "\n"


def claude_add_command(python: str, env: dict[str, str]) -> str:
    import shlex

    envs = " ".join(f"--env {shlex.quote(f'{k}={v}')}" for k, v in env.items())
    return (
        f"claude mcp add --scope user {envs} {SERVER} -- {python} -m cascade.apps.mcp_server"
    )


# ── OpenClaw (native mcp + mcporter) ─────────────────────────────────────


def openclaw_command(python: str, env: dict[str, str]) -> str:
    import shlex

    # OpenClaw blocks PYTHONPATH for stdio servers ("startup safety") — the
    # package must be editable-installed in the venv instead. `--cwd` keeps
    # YOLOE's CWD-relative text-encoder resolution working.
    envs = " ".join(
        f"--env {shlex.quote(f'{k}={v}')}" for k, v in env.items() if k != "PYTHONPATH"
    )
    return (
        f"openclaw mcp add {SERVER} --command {python} "
        f"--arg -m --arg cascade.apps.mcp_server --cwd {REPO} "
        f"--connect-timeout 120 {envs}"
    )


def openclaw_json_block(python: str, env: dict[str, str]) -> str:
    """Claude-Desktop-shaped block: accepted by mcporter (~/.mcporter/
    mcporter.json) and most MCP-compatible launchers."""
    return json.dumps(
        {"mcpServers": {SERVER: {"command": python,
                                 "args": ["-m", "cascade.apps.mcp_server"],
                                 "env": env}}},
        indent=2,
    )


# ── driver ───────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", choices=["all", "hermes", "codex", "claude", "openclaw"],
                   default="all")
    p.add_argument("--camera", default="d455f", help="camera profile, or comma-separated list (first = manipulation camera)")
    p.add_argument("--arm", default="mock")
    p.add_argument("--display", default=":1")
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--detect-classes", default=None,
                   help="comma-separated CASCADE_DETECT_CLASSES vocabulary; beliefs "
                        "are keyed by these labels, so they must name what users "
                        "will ask for (previously env-only: regenerating wiped it)")
    p.add_argument("--no-offline", dest="offline", action="store_false",
                   help="omit YOLO_OFFLINE/ULTRALYTICS_OFFLINE (emitted by "
                        "default: online ultralytics stalls the watcher)")
    p.add_argument("--hide-tools", default=None,
                   help="comma-separated CASCADE_HIDE_TOOLS (e.g. reset_stop for "
                        "attendee-facing booth sessions; emergency_stop is "
                        "never hideable)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra env var for the server entry (repeatable)")
    p.add_argument("--codex-profile", default=None, metavar="NAME",
                   help="Codex host only: write the block to $CODEX_HOME/NAME.config.toml, "
                        "which Codex layers only under `codex -p NAME`, instead of the "
                        "base config.toml -- so the robot tool server is not loaded "
                        "into every coding session")
    p.add_argument("--write", action="store_true",
                   help="apply file edits (hermes yaml, codex toml, project .mcp.json)")
    args = p.parse_args()

    if args.codex_profile is not None:
        try:
            codex_config_path(args.codex_profile)
        except ValueError as e:
            p.error(str(e))

    for kv in args.env:
        if "=" not in kv or not kv.split("=", 1)[0].strip():
            p.error(f"--env expects KEY=VALUE, got {kv!r}")

    try:
        env = server_env(args.camera, args.arm, args.display,
                         detect_classes=args.detect_classes, offline=args.offline,
                         hide_tools=args.hide_tools, extra=args.env)
    except ValueError as e:  # an inherited endpoint variable the server would refuse
        p.error(str(e))
    explicit = {kv.partition("=")[0].strip() for kv in args.env}
    explicit |= {"CASCADE_DETECT_CLASSES"} if args.detect_classes else set()
    explicit |= {"CASCADE_HIDE_TOOLS"} if args.hide_tools else set()
    inherited = [k for k in env if k in FORWARDED and k not in explicit]
    if inherited:
        print("# forwarded from this shell into every server entry: "
              + " ".join(f"{k}={env[k]}" for k in inherited))
    hosts = [args.host] if args.host != "all" else ["hermes", "codex", "claude", "openclaw"]

    for host in hosts:
        print(f"\n=== {host} " + "=" * (60 - len(host)))
        if host == "hermes":
            # the hermes block renders PYTHONPATH/CASCADE_CAMERAS/CASCADE_ARM itself
            # and never carried DISPLAY -- but keep DISPLAY when the user
            # forced it via --env (value differs from the --display default)
            extras = {k: v for k, v in env.items()
                      if k not in ("PYTHONPATH", "CASCADE_CAMERAS", "CASCADE_ARM")
                      and not (k == "DISPLAY" and v == args.display)}
            block = hermes_yaml_block(args.camera, args.arm, args.python,
                                      extra_env=extras)
            path = Path.home() / ".hermes" / "config.yaml"
            if args.write:
                merged = hermes_upsert(path.read_text() if path.exists() else None, block)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(merged)
                print(f"[written] {path} (restart the Hermes gateway)")
            else:
                print(f"# {path}\n{block}")
            print("# or interactively: ./scripts/hermes_demo.sh")

        elif host == "codex":
            block = codex_toml_block(args.python, env)
            path = codex_config_path(args.codex_profile)
            hint = f" (loaded only by `codex -p {args.codex_profile}`)" if args.codex_profile else ""
            if args.write:
                merged = codex_upsert(path.read_text() if path.exists() else None, block)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(merged)
                print(f"[written] {path}{hint}")
            else:
                print(f"# append to {path}{hint}:\n{block}")

        elif host == "claude":
            path = REPO / ".mcp.json"
            content = claude_mcp_json(path.read_text() if path.exists() else None,
                                      args.python, env)
            if args.write:
                path.write_text(content)
                print(f"[written] {path} (Claude Code picks it up in this repo)")
            else:
                print(f"# project-scoped {path}:\n{content}")
            print("# user-scoped (works from any directory):")
            print(claude_add_command(args.python, env))

        elif host == "openclaw":
            print("# native (OpenClaw >= 2026, docs.openclaw.ai/cli/mcp):")
            print(openclaw_command(args.python, env))
            print("\n# or mcporter / Claude-Desktop-style JSON:")
            print(openclaw_json_block(args.python, env))

    if not args.write:
        print("\n(nothing was modified; re-run with --write to apply file edits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
