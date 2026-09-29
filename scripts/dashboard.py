#!/usr/bin/env python3
"""Open this checkout's native OpenClaw Control UI without starting services."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cascade.apps.process_owner import live_records, load_owner, profile_state_dir


def spark_install(repo: Path) -> bool:
    try:
        return json.loads((repo / "runs/.install/install.json").read_text()).get("profile") == "spark"
    except (OSError, ValueError, AttributeError):
        return False


def native_links(command: list[str], env: dict[str, str], repo: Path) -> dict:
    result = subprocess.run([*command, "dashboard", "--json"], env=env, cwd=repo,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    details = json.loads(result.stdout) if result.returncode == 0 else {}
    if not isinstance(details, dict) or not details.get("url") or details.get("ok") is False:
        raise RuntimeError("OpenClaw could not produce a dashboard URL. Check this checkout's gateway logs.")
    return details


def dashboard(repo: Path, *, no_open: bool = False) -> int:
    profile = "cascade-demo"
    state = profile_state_dir(os.environ.get("CASCADE_LAUNCH_STATE", repo / "runs/.launch"), profile)
    owner = load_owner(state, repo, profile)
    if owner is None or not live_records(state, owner, role="gateway_child"):
        raise RuntimeError("This checkout's cascade-demo stack is not running. Start PAAI (Spark) first.")
    config = state / "openclaw/openclaw.json"
    if not config.is_file():
        raise RuntimeError("This checkout's OpenClaw configuration is missing. Start PAAI (Spark) first.")
    cli = repo / ".openclaw-cli/bin/openclaw"
    if not os.access(cli, os.X_OK):
        raise RuntimeError("The installed OpenClaw CLI is missing. Rerun the Spark installer.")
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENCLAW_")}
    env.update(OPENCLAW_PROFILE=profile, CASCADE_OPENCLAW_PROFILE=profile,
               OPENCLAW_STATE_DIR=str(config.parent), OPENCLAW_CONFIG_PATH=str(config),
               NODE_COMPILE_CACHE=str(config.parent / "cache/node-compile"))
    command = [str(cli), "--profile", profile]
    # Check the selected gateway, not merely an occupied port. Keep diagnostic
    # output private: a failed native probe can contain configuration details.
    try:
        health = subprocess.run([*command, "health", "--json", "--timeout", "5000"],
                                env=env, cwd=repo, capture_output=True, text=True, timeout=10)
        healthy = health.returncode == 0 and json.loads(health.stdout).get("ok") is True
    except (ValueError, AttributeError, subprocess.TimeoutExpired):
        healthy = False
    if not healthy:
        raise RuntimeError("This checkout's OpenClaw gateway is not running or healthy. Start PAAI (Spark) and check its logs.")
    # The native CLI reads the current token from this profile and handles the
    # authenticated browser URL. Never store that URL in a launcher receipt.
    if no_open:
        print(native_links(command, env, repo)["url"])
        return 0
    if spark_install(repo) and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # The native CLI would open the default browser, which does not load
        # the camera extension. Hand its one-time sign-in link to the PAAI
        # browser profile instead; the link expires unused after ten minutes.
        url = native_links(command, env, repo).get("browserUrl")
        target = urlsplit(url) if isinstance(url, str) else None
        if (target is None or target.scheme != "http" or target.hostname != "127.0.0.1"
                or target.username or target.password or "bootstrapToken=" not in target.fragment):
            raise RuntimeError("OpenClaw could not produce a sign-in link. Check this checkout's gateway logs.")
        sys.path.insert(0, str(repo / "scripts"))
        from spark_browser import open_browser
        open_browser(repo, url=url)
        print("[dashboard] OpenClaw opened in the PAAI browser. Press Ctrl+Shift+Y "
              "or click the camera extension to show the cameras beside the chat.", flush=True)
        return 0
    return subprocess.run([*command, "dashboard"],
                          env=env, cwd=repo, stdin=subprocess.DEVNULL).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="Print the authenticated URL without opening a browser")
    args = parser.parse_args()
    return dashboard(REPO, no_open=args.no_open)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"[dashboard] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError, subprocess.SubprocessError):
        print("[dashboard] ERROR: Cannot read or reach this checkout's OpenClaw profile. Check the installation and start PAAI (Spark).", file=sys.stderr)
        raise SystemExit(1)
