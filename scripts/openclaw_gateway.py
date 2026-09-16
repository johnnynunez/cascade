#!/usr/bin/env python3
"""A checkout-owned foreground OpenClaw gateway; no service-manager writes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from cascade.apps.process_owner import live_records, load_owner, register_process, stop_owned
from demo_proof import wait_gateway


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.5):
            return True
    except OSError:
        return False


def wait_closed(port, timeout=5):
    # Process identity can disappear a moment before the kernel closes its
    # listening socket. Bound that transition before a restart binds the port.
    deadline = time.monotonic() + timeout
    while port_open(port):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"OpenClaw port :{port} remains occupied after stopping the owned gateway")
        time.sleep(.05)


def prepare(state, owner, port):
    """Refuse borrowed endpoints before configuration, then retire our old child."""
    children = live_records(state, owner, role="gateway_child")
    if port_open(port) and not children:
        raise RuntimeError(f"OpenClaw port :{port} is occupied by an unowned gateway; leave it running and use a free port")
    stop_owned(state, owner, roles={"gateway_child", "mcp"})
    wait_closed(port)


def start(repo, state, owner, port, *, wait_seconds=60):
    if live_records(state, owner, role="gateway_child") or port_open(port):
        raise RuntimeError(f"OpenClaw port :{port} or gateway owner is already active; restart through the launcher")
    cli = repo / ".openclaw-cli/bin/openclaw"
    if not os.access(cli, os.X_OK):
        raise RuntimeError("Missing project OpenClaw CLI; rerun the Spark installer")
    env = os.environ.copy()
    env.update(OPENCLAW_STATE_DIR=str(state / "openclaw"),
               OPENCLAW_CONFIG_PATH=str(state / "openclaw/openclaw.json"),
               OPENCLAW_PROFILE=owner["profile"],
               NODE_COMPILE_CACHE=str(state / "openclaw/cache/node-compile"))
    command = [str(cli), "--profile", owner["profile"], "gateway", "run", "--bind", "loopback", "--port", str(port)]
    child = None
    success = False
    handlers = {}

    def interrupted(number, frame):
        raise InterruptedError(f"OpenClaw gateway startup interrupted by signal {number}")

    try:
        for number in (signal.SIGTERM, signal.SIGHUP):
            handlers[number] = signal.signal(number, interrupted)
        with (state / "openclaw-gateway.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_gateway(wait_seconds, environment=env,
                     command_prefix=[str(cli), "--profile", owner["profile"]])
        if child.poll() is not None:
            raise RuntimeError(f"OpenClaw foreground gateway exited with status {child.returncode}")
        record = register_process(state, owner, child.pid, "gateway_child")
        success = True
        return record
    finally:
        if not success and child is not None:
            # This Popen was created here in its own group. It is not a
            # discovered endpoint or an old numeric PID file.
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
        for number, previous in handlers.items():
            signal.signal(number, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "stop"))
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("port must be in 1..65535")
    repo, state = args.repo.resolve(), args.state_dir.resolve()
    owner = load_owner(state, repo, args.profile)
    if owner is None:
        raise RuntimeError("Missing checkout gateway owner receipt")
    if args.action == "prepare":
        prepare(state, owner, args.port)
    elif args.action == "start":
        start(repo, state, owner, args.port)
    else:
        stop_owned(state, owner, roles={"gateway_child", "mcp"})
        wait_closed(args.port)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[gateway] {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
