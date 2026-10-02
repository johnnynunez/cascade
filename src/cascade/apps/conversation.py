"""Run an explicit robot profile behind the optional local speech gateway."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--robot", default="conversation_mock")
    result.add_argument("--provider-url", required=True, help="Operator-owned HF GA Realtime ws(s) endpoint")
    result.add_argument("--token-env", help="Environment variable containing provider bearer token")
    result.add_argument("--allow-tool", action="append", default=[])
    result.add_argument("--allow-motion", action="store_true", help="Enable explicitly listed curated semantic motions")
    result.add_argument("--barge-in", choices=["stop_robot", "speech_only"], default="stop_robot")
    result.add_argument("--port", type=int, default=8780)
    result.add_argument("--run-dir", type=Path, required=True, help="New task-owned run directory")
    return result


async def serve(args):
    from ..config import load_robot_config
    from ..conversation.domain import ConversationDomain
    from ..conversation.gateway import ConversationGateway
    from ..conversation.provider import RealtimeConfig, RealtimeWebSocket
    from .robot_runtime import build_robot_runtime
    config = RealtimeConfig(args.provider_url, args.token_env)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    cfg = load_robot_config(args.robot)
    runtime, _ = build_robot_runtime(cfg, args.run_dir)
    gateway = None
    try:
        domain = ConversationDomain(runtime, robot_id=cfg.robot_id, allow_tools=args.allow_tool,
                                    allow_motion=args.allow_motion, barge_in=args.barge_in)
        gateway = ConversationGateway(domain, lambda: RealtimeWebSocket(config))
        origin = await gateway.start(port=args.port)
        # Fragment is not sent in HTTP requests. Browser exchanges it for a
        # single-use media ticket; neither credential is written to run files.
        print(f"Open {origin}/#{gateway.token}", flush=True)
        print("No speech inference or physical result is claimed before provider connection and evidence.", flush=True)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
    finally:
        conversation = await gateway.close() if gateway else {"ok": True}
        closure = await asyncio.to_thread(runtime.close)
        (args.run_dir / "closure.json").write_text(json.dumps({"conversation": conversation, "runtime": closure}, indent=2))


def main():
    args = parser().parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
