"""Run an explicit robot profile behind the optional local speech gateway."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, help="Versioned JSON service configuration")
    result.add_argument("--config-dir", type=Path, help="Explicit installed robot/LLM profile directory")
    result.add_argument("--robot")
    result.add_argument("--robot-lifecycle", choices=["bounded_hand"],
                        help="Start one bounded hand episode only after explicit prepared-browser activation")
    result.add_argument("--provider-url", help="Operator-owned HF GA Realtime ws(s) endpoint")
    result.add_argument("--token-env", help="Environment variable containing provider bearer token")
    result.add_argument("--provider-release-contract", choices=["hf_pool"],
                        help="Require observed HF pool release before session closure succeeds")
    result.add_argument("--allow-tool", action="append")
    result.add_argument("--allow-motion", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable explicitly listed curated semantic motions")
    result.add_argument("--barge-in", choices=["stop_robot", "speech_only"])
    result.add_argument("--port", type=int)
    result.add_argument("--run-dir", type=Path, help="New task-owned run directory")
    result.add_argument("--run-root", type=Path, help="Create a new private child directory on each service start")
    result.add_argument("--start-stopped", action=argparse.BooleanOptionalAction, default=None,
                        help="Require an explicit operator reset before this process can dispatch tools")
    result.add_argument("--intent-timeout-s", type=float,
                        help="Input-origin budget, at most 60 seconds (default 10)")
    result.add_argument("--execution-timeout-s", type=float,
                        help="Action budget, at most 300 seconds (default 30)")
    result.add_argument("--robot-endpoint",
                        help="Split deployment: http://<loopback>:<port> of a cascade-robot-service, or "
                             "https://<host>:<port> with --robot-tls-ca/--robot-tls-fingerprint "
                             "(--robot then names its expected robot_id); default builds the robot in-process")
    result.add_argument("--robot-token-env", help="Environment variable holding the robot endpoint bearer token")
    result.add_argument("--robot-tls-ca", type=Path,
                        help="Opt-in TLS (B71): the only CA trusted for the https robot endpoint (hostname checked)")
    result.add_argument("--robot-tls-fingerprint",
                        help="Opt-in TLS (B71): pinned SHA-256 of the robot endpoint's certificate (hex)")
    return result


async def serve(args):
    from ..config import load_robot_config
    from ..conversation.domain import ConversationDomain
    from ..conversation.gateway import ConversationGateway
    from ..conversation.provider import RealtimeConfig, RealtimeWebSocket
    from ..conversation.lifecycle import close_stage
    from ..lifecycle import teardown_receipt
    from .robot_runtime import build_robot_runtime
    config = RealtimeConfig(args.provider_url, args.token_env,
                            release_contract=getattr(args, "provider_release_contract", None))
    run_dir = args.run_dir
    if getattr(args, "run_root", None) is not None:
        run_dir = args.run_root / ("conversation-" + uuid.uuid4().hex)
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    runtime = gateway = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    signals, error = [], None
    def requested_stop(sig):
        signals.append(sig.name)
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, requested_stop, sig)
    try:
        remote = getattr(args, "robot_endpoint", None) is not None
        activation = None
        if remote:
            # Split deployment (B51): the robot runtime is owned by its own
            # cascade-robot-service process; no local profile or driver here.
            from ..robotics.endpoint import RemoteRobotRuntime, endpoint_token
            runtime = RemoteRobotRuntime(args.robot_endpoint, endpoint_token(args.robot_token_env),
                                         tls_ca=getattr(args, "robot_tls_ca", None),
                                         tls_fingerprint=getattr(args, "robot_tls_fingerprint", None))
            runtime.connect()
            if runtime.robot_id != args.robot:
                raise ValueError("remote robot identity differs from the configured robot")
            robot_id, robot_config_sha256 = runtime.robot_id, None
        else:
            cfg = load_robot_config(args.robot, config_dir=getattr(args, "config_dir", None))
            if getattr(args, "robot_lifecycle", None) == "bounded_hand":
                from ..conversation.activation import build_bounded_hand_service
                runtime, activation = build_bounded_hand_service(cfg, run_dir)
            else:
                runtime, _ = build_robot_runtime(cfg, run_dir)
            robot_id = cfg.robot_id
            robot_config_sha256 = hashlib.sha256(json.dumps(
                cfg.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        domain = ConversationDomain(runtime, robot_id=robot_id, allow_tools=args.allow_tool,
                                    allow_motion=args.allow_motion, barge_in=args.barge_in,
                                    intent_timeout_s=args.intent_timeout_s,
                                    execution_timeout_s=args.execution_timeout_s)
        gateway = ConversationGateway(domain, lambda: RealtimeWebSocket(config),
                                      **({"activation": activation} if activation is not None else {}),
                                      **({"probes": runtime.ready} if remote else {}))
        if getattr(args, "start_stopped", False):
            stopped = await domain.stop()
            if stopped.get("ok") is not True:
                raise RuntimeError("initial stop was not acknowledged")
        origin = await gateway.start(port=args.port)
        ready = {"schema": 1, "state": "listening_at_publication", "pid": os.getpid(),
                 "published_monotonic_s": time.monotonic(), "origin": origin,
                 "robot_id": robot_id, "tools": list(domain.tools),
                 "runtime_stopped": runtime.stopped, "provider_connected": False,
                 "physical_admission": False,
                 "intent_timeout_s": args.intent_timeout_s,
                 "execution_timeout_s": args.execution_timeout_s,
                 "robot_lifecycle": getattr(args, "robot_lifecycle", None),
                 "service_config_sha256": getattr(args, "service_config_sha256", None),
                 "robot_config_sha256": robot_config_sha256}
        if remote:
            ready["robot_endpoint"] = runtime.endpoint_receipt()
        with (run_dir / "ready.json").open("x") as stream:
            json.dump(ready, stream, indent=2)
        # Fragment is not sent in HTTP requests. Browser exchanges it for a
        # single-use media ticket; neither credential is written to run files.
        print(f"Open {origin}/#{gateway.token}", flush=True)
        print("No speech inference or physical result is claimed before provider connection and evidence.", flush=True)
        await stop.wait()
    except (Exception, asyncio.CancelledError) as exc:
        error = {"type": type(exc).__name__}
    finally:
        stages = []
        if gateway is not None:
            stages.append(await close_stage("conversation", gateway.close))
        if runtime is not None:
            stages.append(await close_stage("runtime", lambda: asyncio.to_thread(runtime.close)))
        closure = teardown_receipt(stages)
        closure.update(signals=signals, service_error=error)
        # Preserve the original receipt keys for existing supervisors.
        for stage in stages:
            closure[stage["stage"]] = stage.get("result", stage)
        if error is not None:
            closure["ok"] = False
        try:
            with (run_dir / "closure.json").open("x") as stream:
                json.dump(closure, stream, indent=2)
        except Exception as exc:
            closure["ok"] = False
            print("Conversation closure persistence failed: " + type(exc).__name__, file=sys.stderr)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
    return 0 if closure["ok"] else 1


def main():
    from ..conversation.service import configuration
    cli = parser()
    args = cli.parse_args()
    try:
        values = configuration(args)
    except (ValueError, OSError) as exc:
        cli.error(str(exc))
    raise SystemExit(asyncio.run(serve(argparse.Namespace(**values))))


if __name__ == "__main__":
    main()
