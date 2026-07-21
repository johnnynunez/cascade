#!/usr/bin/env python3
"""Register the wrc-demo MCP server with every supported agent platform.

One robot tool-server, many brains. Supported hosts:

    hermes    ~/.hermes/config.yaml           (mcp_servers.<name>, YAML)
    codex     ~/.codex/config.toml            ([mcp_servers.<name>], TOML)
    claude    <repo>/.mcp.json (project)      (Claude Code; auto-detected)
              + `claude mcp add` one-liner for user scope / Desktop JSON
    openclaw  `openclaw mcp set` one-liner    (native mcp.servers) or a
              Claude-Desktop-style JSON block (also valid for mcporter)

By default this PRINTS what each host needs; `--write` applies the ones
that can be safely edited in place (hermes YAML, codex TOML, project
.mcp.json), always preserving unrelated entries.

    python scripts/setup_agents.py                          # print all
    python scripts/setup_agents.py --host codex --write
    python scripts/setup_agents.py --camera l515 --arm rebot_rs --write
"""

from __future__ import annotations

import argparse
import json
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
SERVER = "wrc-demo"


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
        "WRC_CAMERAS": camera,
        "WRC_ARM": arm,
        "DISPLAY": display,
    }
    if offline:
        # without these, ultralytics phones GitHub on class re-embeds and
        # stalls the perception watcher for seconds -- never at a venue
        env["YOLO_OFFLINE"] = "True"
        env["ULTRALYTICS_OFFLINE"] = "True"
    if detect_classes:
        env["WRC_DETECT_CLASSES"] = detect_classes
    if hide_tools:
        env["WRC_HIDE_TOOLS"] = hide_tools
    for kv in extra or []:
        k, _, v = kv.partition("=")
        env[k.strip()] = v
    return env


# ── Codex CLI (TOML) ─────────────────────────────────────────────────────

CODEX_START = f"[mcp_servers.{SERVER}]"
CODEX_ENV_START = f"[mcp_servers.{SERVER}.env]"


def codex_toml_block(python: str, env: dict[str, str]) -> str:
    args = ", ".join(f'"{a}"' for a in ["-m", "wrc_demo.apps.mcp_server"])
    lines = [CODEX_START, f'command = "{python}"', f"args = [{args}]", CODEX_ENV_START]
    lines += [f'{k} = "{v}"' for k, v in env.items()]
    return "\n".join(lines) + "\n"


def codex_upsert(existing: str | None, block: str) -> str:
    """Replace our `[mcp_servers.wrc-demo]` tables, preserve everything else."""
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
        "args": ["-m", "wrc_demo.apps.mcp_server"],
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
        f"claude mcp add --scope user {envs} {SERVER} -- {python} -m wrc_demo.apps.mcp_server"
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
        f"--arg -m --arg wrc_demo.apps.mcp_server --cwd {REPO} "
        f"--connect-timeout 120 {envs}"
    )


def openclaw_json_block(python: str, env: dict[str, str]) -> str:
    """Claude-Desktop-shaped block: accepted by mcporter (~/.mcporter/
    mcporter.json) and most MCP-compatible launchers."""
    return json.dumps(
        {"mcpServers": {SERVER: {"command": python,
                                 "args": ["-m", "wrc_demo.apps.mcp_server"],
                                 "env": env}}},
        indent=2,
    )


# ── driver ───────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", choices=["all", "hermes", "codex", "claude", "openclaw"],
                   default="all")
    p.add_argument("--camera", default="l515", help="camera profile, or comma-separated list (first = manipulation camera)")
    p.add_argument("--arm", default="mock")
    p.add_argument("--display", default=":1")
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--detect-classes", default=None,
                   help="comma-separated WRC_DETECT_CLASSES vocabulary; beliefs "
                        "are keyed by these labels, so they must name what users "
                        "will ask for (previously env-only: regenerating wiped it)")
    p.add_argument("--no-offline", dest="offline", action="store_false",
                   help="omit YOLO_OFFLINE/ULTRALYTICS_OFFLINE (emitted by "
                        "default: online ultralytics stalls the watcher)")
    p.add_argument("--hide-tools", default=None,
                   help="comma-separated WRC_HIDE_TOOLS (e.g. reset_stop for "
                        "attendee-facing booth sessions; emergency_stop is "
                        "never hideable)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra env var for the server entry (repeatable)")
    p.add_argument("--write", action="store_true",
                   help="apply file edits (hermes yaml, codex toml, project .mcp.json)")
    args = p.parse_args()

    for kv in args.env:
        if "=" not in kv or not kv.split("=", 1)[0].strip():
            p.error(f"--env expects KEY=VALUE, got {kv!r}")

    env = server_env(args.camera, args.arm, args.display,
                     detect_classes=args.detect_classes, offline=args.offline,
                     hide_tools=args.hide_tools, extra=args.env)
    hosts = [args.host] if args.host != "all" else ["hermes", "codex", "claude", "openclaw"]

    for host in hosts:
        print(f"\n=== {host} " + "=" * (60 - len(host)))
        if host == "hermes":
            # the hermes block renders PYTHONPATH/WRC_CAMERAS/WRC_ARM itself
            # and never carried DISPLAY -- but keep DISPLAY when the user
            # forced it via --env (value differs from the --display default)
            extras = {k: v for k, v in env.items()
                      if k not in ("PYTHONPATH", "WRC_CAMERAS", "WRC_ARM")
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
            path = Path.home() / ".codex" / "config.toml"
            if args.write:
                merged = codex_upsert(path.read_text() if path.exists() else None, block)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(merged)
                print(f"[written] {path}")
            else:
                print(f"# append to {path}:\n{block}")

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
