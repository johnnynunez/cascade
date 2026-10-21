#!/usr/bin/env python3
"""One operator request through CASCADE's agent on a composed humanoid robot, recorded for review.

operator prompt -> AgentOrchestrator (the configured brain, e.g. Qwen3.8 on vLLM) -> namespaced tool
calls -> domain skills (GR00T loco-manipulation behind the humanoid gate, or the G1 wrist-driver
fastening skill) -> independent verifiers -> the brain's reply. Writes ``trace.json`` (brain
identity, every brain turn and tool call with wall-clock times, compact results, the final report)
next to the runtime's own trace and domain records.

    # loco-manipulation (bridge + GR00T server already running; cascade's venv)
    python scripts/humanoid_agent_episode.py --robot g1_arena_locomanip --llm local_qwen \
        --prompt "Put the brown box into the blue bin" --out runs/humanoid/carry-01

    # fastening (Newton env; pins the model identity into a private config copy first)
    <newton-env>/bin/python scripts/humanoid_agent_episode.py --robot g1_wrist_driver_fastening \
        --llm local_qwen --pin-fastening --robot-asset <g1 mjcf> --device cuda:0 \
        --prompt "Tighten the nut one full turn" --out runs/humanoid/fasten-01

``--harness <dir|harness.json>`` serves a Harness²-adopted harness (advice text only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def pinned_fastening_config(out: Path, robot: str, robot_asset: Path, device: str, warmup_steps: int = 40) -> Path:
    """Private copy of the configs with the fastening profile pinned (device + model identity);
    the shipped profile stays unprepared (null), as scripts/validate_humanoid_fastening.py does."""
    from cascade.apps.humanoid_fastening_runtime import RECIPE, prepare_humanoid_model
    from cascade.config import CONFIG_DIR, _load_yaml

    shipped = _load_yaml(CONFIG_DIR / "robots" / f"{robot}.yaml")
    name, profile = next(iter(shipped["domains"].items()))
    if profile.get("recipe") != RECIPE:
        raise SystemExit(f"{robot} is not the G1 wrist-driver fastening profile")
    profile = {**profile, "assets": str((REPO / "assets/factory/nut_bolt").resolve()),
               "robot_asset": str(robot_asset.resolve()), "device": device, "model_identity_sha256": None}
    model = prepare_humanoid_model(profile, out / "prepare-sdf-cache")
    profile["model_identity_sha256"] = model.binding.model_sha256
    del model
    # Load the stepping kernels into this process on a throwaway copy of the scene (never the one
    # the runtime owns): the runtime's 10 s readiness window otherwise also pays every first-launch
    # GPU module load (measured ~15 s on a shared GPU), and readiness fails before the first quiet
    # interval.
    from cascade.sim.newton_humanoid_screw import HumanoidThreadingScene
    warm = HumanoidThreadingScene(profile["assets"], profile["robot_asset"], out / "prepare-sdf-cache", device=device)
    for _ in range(warmup_steps):
        warm.step()
        warm.observe()
    del warm
    config = out / "config"
    (config / "robots").mkdir(parents=True, exist_ok=True)
    shutil.copytree(CONFIG_DIR / "llm", config / "llm", dirs_exist_ok=True)
    lines = ["version: 1", f"robot_id: {shipped['robot_id']}", "domains:", f"  {name}:"]
    lines += [f"    {key}: {json.dumps(value)}" for key, value in profile.items()]
    (config / "robots" / f"{robot}.yaml").write_text("\n".join(lines) + "\n")
    return config


def brain_identity(llm_cfg) -> dict:
    out = {"profile_model": llm_cfg.get("model"), "base_url": llm_cfg.get("base_url"), "type": llm_cfg.get("type")}
    base = str(llm_cfg.get("base_url") or "").rstrip("/")
    for key, url in (("served_models", base + "/models"), ("server_version", base.rsplit("/v1", 1)[0] + "/version")):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            out[key] = [m["id"] for m in data.get("data", [])] if key == "served_models" else data.get("version")
        except Exception as exc:  # noqa: BLE001 -- identity is evidence, never a precondition
            out[key] = f"unavailable: {exc}"
    return out


class TimedLLM:
    def __init__(self, inner, clock):
        self.inner, self.clock = inner, clock
        self.turns: list[dict] = []
        self.supports_vision = getattr(inner, "supports_vision", False)

    def chat(self, system, messages, tools=None, max_tokens=1024):
        t0 = self.clock()
        resp = self.inner.chat(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
        self.turns.append({"t_start": round(t0, 3), "t_end": round(self.clock(), 3), "planner": tools is not None,
                           "text": resp.text or "",
                           "calls": [{"name": c.name, "arguments": dict(c.arguments or {})} for c in resp.tool_calls]})
        return resp

    def close(self):
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _compact(result, limit=30000):
    text = json.dumps(result, default=str)
    return result if len(text) <= limit else {"truncated": text[:limit]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    # The repository's default brain (Nemotron 3.5 Bolt), as scripts/launch.sh resolves
    # `--brain auto` and scripts/booth_rehearsal.py defaults; any other brain is named explicitly.
    ap.add_argument("--llm", default="local_nemotron_bolt",
                    help="configs/llm profile of the brain (default: local_nemotron_bolt, the launch default; "
                         "it talks to the LIVE local brain server on 127.0.0.1:8080, as local_qwen does)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--harness", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--operator-arm", action="store_true",
                    help="the operator clears the startup stop latch (reset_stop) before the request; recorded")
    ap.add_argument("--pin-fastening", action="store_true")
    ap.add_argument("--robot-asset", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)

    from cascade.agent.llm import make_llm
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import load_robot_config
    from cascade.harnessing import AgentHarness

    t_origin = time.time()
    clock = lambda: time.time() - t_origin  # noqa: E731
    config_dir = args.config_dir
    if args.pin_fastening:
        if args.robot_asset is None:
            raise SystemExit("--pin-fastening needs --robot-asset")
        config_dir = pinned_fastening_config(args.out, args.robot, args.robot_asset, args.device)
    cfg = load_robot_config(args.robot, llm=args.llm, config_dir=config_dir)
    harness = AgentHarness.read(args.harness) if args.harness else None
    t_build = clock()
    runtime, _ = build_robot_runtime(cfg, args.out / "run")
    built = clock()
    tools: list[dict] = []
    original = runtime.execute

    def timed_execute(name, tool_args=None, **kwargs):
        t0 = clock()
        result = original(name, tool_args, **kwargs)
        tools.append({"t_start": round(t0, 3), "t_end": round(clock(), 3), "tool": name, "args": tool_args or {},
                      "result": _compact(result)})
        return result

    operator = []
    if args.operator_arm:
        operator.append({"t": round(clock(), 3), "action": "reset_stop",
                         "result": runtime.execute("reset_stop", {}, operator=True)})
    runtime.execute = timed_execute
    llm = TimedLLM(make_llm(cfg.llm), clock)
    trace = {"robot": args.robot, "prompt": args.prompt, "brain": brain_identity(cfg.llm), "llm_profile": args.llm,
             "harness": None if harness is None else {"sha": harness.sha(), "path": str(args.harness),
                                                      "rules": list(harness.rules),
                                                      "tool_guidance": dict(harness.tool_guidance)},
             "runtime_build_s": round(built - t_build, 2), "t_origin_unix": t_origin, "operator_actions": operator}
    try:
        orch = AgentOrchestrator(llm, runtime, decompose=args.decompose, verify_milestones=False,
                                 max_steps=args.max_steps, harness=harness)
        t_prompt = clock()
        report = orch.run_task(args.prompt)
        trace.update(t_prompt=round(t_prompt, 3), t_end=round(clock(), 3), turns=llm.turns, tools=tools,
                     report={"success": report.success, "summary": report.summary, "steps": report.steps,
                             "unverified": report.unverified, "duration_s": report.duration_s})
        done = next((t for t in reversed(tools) if t["tool"] == "task_done"), None)
        trace["claim"] = None if done is None else {"success": done["args"].get("success"),
                                                    "summary": done["args"].get("summary")}
    finally:
        try:
            trace["closure"] = _compact(runtime.close(), 1500)
        finally:
            llm.close()
            (args.out / "trace.json").write_text(json.dumps(trace, indent=1, default=str) + "\n")
    print(json.dumps({k: trace.get(k) for k in ("robot", "prompt", "claim", "report")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
