"""Serve one explicit robot runtime to a separately deployed conversation service.

``cascade-robot-service`` builds the composed robot runtime exactly like the
in-process conversation service does (``build_robot_runtime``) and serves it
over ``cascade.robot-runtime/1`` (see ``cascade/robotics/endpoint.py``): on
loopback in clear by default, or with ``--tls-cert``/``--tls-key`` (B71, opt-in)
over TLS on any explicit unicast IP; a non-loopback ``--host`` without TLS is
refused before anything is built.
``cascade-conversation --robot-endpoint ...`` then supervises it from its own
process. The robot side owns its runtime, harnesses, verifiers and trace; the
supervisor owns only a renewable lease whose expiry latches the robot stop.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import signal
import sys
import threading
import time
import uuid
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--robot", required=True, help="Robot profile (configs/robots/<name>.yaml)")
    result.add_argument("--config-dir", type=Path, help="Explicit installed robot/LLM profile directory")
    result.add_argument("--port", type=int, required=True, help="Port to bind (0 = ephemeral)")
    result.add_argument("--host", default="127.0.0.1",
                        help="Explicit IP to bind: loopback, or any unicast IP with --tls-cert/--tls-key")
    result.add_argument("--tls-cert", type=Path,
                        help="Opt-in TLS (B71): PEM server certificate (chain, leaf first); required off loopback")
    result.add_argument("--tls-key", type=Path, help="PEM private key of --tls-cert")
    runs = result.add_mutually_exclusive_group(required=True)
    runs.add_argument("--run-dir", type=Path, help="New task-owned run directory")
    runs.add_argument("--run-root", type=Path, help="Create a new private child directory on each start")
    result.add_argument("--token-env", required=True,
                        help="Environment variable holding the shared bearer token (>= 32 characters)")
    result.add_argument("--lease-ttl-s", type=float, default=3.0,
                        help="Supervisor lease; its expiry latches the robot stop (0.2 < ttl <= 30, default 3)")
    return result


def serve(args, token):
    from ..config import load_robot_config
    from ..lifecycle import teardown_receipt, teardown_step
    from ..robotics.endpoint import PROTOCOL, RobotRuntimeEndpoint, catalog_digest
    from .robot_runtime import build_robot_runtime
    run_dir = args.run_dir
    if args.run_root is not None:
        run_dir = args.run_root / ("robot-service-" + uuid.uuid4().hex)
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    stop, signals = threading.Event(), []

    def requested_stop(signum, _frame):
        signals.append(signal.Signals(signum).name)
        stop.set()

    previous = {sig: signal.signal(sig, requested_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    runtime = endpoint = error = None
    try:
        cfg = load_robot_config(args.robot, config_dir=args.config_dir)
        runtime, _ = build_robot_runtime(cfg, run_dir)
        endpoint = RobotRuntimeEndpoint(runtime, robot_id=cfg.robot_id, token=token,
                                        lease_ttl_s=args.lease_ttl_s)
        tls_cert, tls_key = getattr(args, "tls_cert", None), getattr(args, "tls_key", None)
        origin = endpoint.start(host=args.host, port=args.port,
                                **({} if tls_cert is None else {"tls_cert": tls_cert, "tls_key": tls_key}))
        description = endpoint.describe()
        ready = {"schema": 1, "protocol": PROTOCOL, "state": "listening_at_publication", "pid": os.getpid(),
                 "published_monotonic_s": time.monotonic(), "origin": origin, "robot_id": cfg.robot_id,
                 "tools": list(runtime.tool_descriptors), "catalog_sha256": catalog_digest(description),
                 "lease_ttl_s": endpoint.lease_ttl_s, "runtime_stopped": runtime.stopped,
                 "physical_admission": False,
                 "robot_config_sha256": hashlib.sha256(json.dumps(
                     cfg.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        if endpoint.tls_certificate_sha256 is not None:
            ready["tls_certificate_sha256"] = endpoint.tls_certificate_sha256
        with (run_dir / "ready.json").open("x") as stream:
            json.dump(ready, stream, indent=2)
        print(f"Robot runtime endpoint {origin} ({PROTOCOL})", flush=True)
        print("Bearer token from the configured environment variable; no physical result is claimed.", flush=True)
        if endpoint.tls_certificate_sha256 is not None:
            print(f"TLS certificate sha256 {endpoint.tls_certificate_sha256}", flush=True)
        while not stop.wait(.5):
            pass
    except Exception as exc:  # noqa: BLE001 -- record the type; never a credential or path
        error = {"type": type(exc).__name__}
        print(f"Robot runtime service failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        stages = []
        if endpoint is not None:
            stages.append(teardown_step("endpoint", endpoint.close))
        if runtime is not None:
            stages.append(teardown_step("runtime", runtime.close))
        closure = teardown_receipt(stages)
        closure.update(signals=signals, service_error=error,
                       lease_expiry_stops=list(endpoint.lease_expiry_stops) if endpoint is not None else [])
        if error is not None:
            closure["ok"] = False
        try:
            with (run_dir / "closure.json").open("x") as stream:
                json.dump(closure, stream, indent=2)
        except Exception as exc:  # noqa: BLE001
            closure["ok"] = False
            print("Robot service closure persistence failed: " + type(exc).__name__, file=sys.stderr)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if closure["ok"] else 1


def main():
    from ..robotics.endpoint import endpoint_token, server_tls_context
    cli = parser()
    args = cli.parse_args()
    if not math.isfinite(args.lease_ttl_s) or not .2 < args.lease_ttl_s <= 30:
        cli.error("--lease-ttl-s must be finite and in (0.2, 30]")
    if not 0 <= args.port <= 65535:
        cli.error("--port must be in 0..65535")
    # Fail closed before any runtime is built: off loopback the token never travels in clear.
    if (args.tls_cert is None) != (args.tls_key is None):
        cli.error("--tls-cert and --tls-key go together")
    try:
        address = ipaddress.ip_address(args.host)
    except ValueError:
        address = None               # refused by the endpoint as before
    if address is not None and args.tls_cert is None and not address.is_loopback:
        cli.error(f"a non-loopback --host ({args.host}) requires TLS (--tls-cert/--tls-key): "
                  "the bearer token must not cross a network in clear")
    if address is not None and (address.is_unspecified or address.is_multicast):
        cli.error("--host must be one explicit unicast IP")
    if args.tls_cert is not None:
        try:
            server_tls_context(args.tls_cert, args.tls_key)
        except (OSError, ValueError) as exc:
            cli.error(f"TLS certificate/key unusable: {type(exc).__name__}")
    try:
        token = endpoint_token(args.token_env)
    except ValueError as exc:
        cli.error(str(exc))
    raise SystemExit(serve(args, token))


if __name__ == "__main__":
    main()
