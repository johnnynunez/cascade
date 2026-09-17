#!/usr/bin/env python3
"""Read-only brain preflight and evidence-based launcher verification.

Kept outside the robot package: bootstrapping must work before its extras
are installed. No model/server output is accepted as physics evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class ProofError(RuntimeError):
    """A required component or independent proof is missing."""


class EndpointUnavailable(ProofError):
    """Connection failed, unlike an answering endpoint with the wrong model."""


def oc_command(*args: str) -> list[str]:
    profile = os.environ.get("CASCADE_OPENCLAW_PROFILE", "")
    return ["openclaw", *(["--profile", profile] if profile else []), *args]


def wait_gateway(wait_s: float = 45, *, retry_delay: float = 1, environment=None, command_prefix=None) -> dict:
    import time

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            command = [*command_prefix, "health", "--json", "--timeout", "5000"] if command_prefix else oc_command("health", "--json", "--timeout", "5000")
            p = subprocess.run(command, env=environment,
                               capture_output=True, text=True, timeout=max(0.1, min(8, remaining)))
            if p.returncode == 0 and json.loads(p.stdout).get("ok") is True:
                return {"ok": True, "profile": os.environ.get("CASCADE_OPENCLAW_PROFILE", "")}
        except (ValueError, subprocess.TimeoutExpired):
            pass
        time.sleep(max(0, min(retry_delay, deadline - time.monotonic())))
    raise ProofError("gateway did not become healthy for the selected profile; port ownership/authentication may conflict")


def request_json(url: str, payload: dict | None = None, timeout: float = 5) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ProofError("model base URL must be HTTP(S)")
    # Respect remote corporate proxies, but never send local model traffic
    # through the system proxy (macOS VPNs otherwise intercept loopback).
    opener = build_opener(ProxyHandler({})) if parsed.hostname in ("localhost", "127.0.0.1", "::1") else build_opener()
    headers = {"Accept": "application/json"}
    key = os.environ.get("CASCADE_MODEL_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    try:
        with opener.open(Request(url, data=data, headers=headers), timeout=timeout) as response:
            result = json.loads(response.read(4 * 1024 * 1024))
    except HTTPError as exc:
        raise ProofError(f"model endpoint returned HTTP {exc.code}; refusing silent fallback") from exc
    except (URLError, OSError) as exc:
        raise EndpointUnavailable(f"model endpoint {parsed.hostname}:{parsed.port if parsed.port is not None else 80} unavailable ({type(exc).__name__})") from exc
    except ValueError as exc:
        raise ProofError("model endpoint returned invalid JSON; refusing silent fallback") from exc
    if not isinstance(result, dict):
        raise ProofError("model endpoint did not return a JSON object")
    return result


def local_brain(base: str, expected: str | None) -> dict:
    data = request_json(base.rstrip("/") + "/models")
    ids = [row.get("id") or row.get("model") for row in data.get("data", data.get("models", [])) if isinstance(row, dict)]
    if not ids or (expected and expected not in ids):
        raise ProofError(f"model {expected or '<any>'} not served at {base}; available={ids}")
    return {"base_url": base, "model": expected or ids[0]}


def probe_native_tools(base: str, model: str) -> dict:
    """A real inference request, never a GET /models masquerading as proof."""
    info = local_brain(base, model)
    payload = {
        "model": model, "stream": False, "temperature": 0, "max_tokens": 512,
        "messages": [{"role": "system", "content": "Use the available tools to check readiness or explain capabilities."},
                     {"role": "user", "content": "Are you ready?"}],
        "tools": [{"type": "function", "function": {
            "name": "cascade_readiness", "description": "Check whether the demo is ready. Set ready=true to request the readiness check. This diagnostic has no side effects.",
            "parameters": {"type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]},
        }}, {"type": "function", "function": {
            "name": "cascade_capabilities", "description": "List the available demo capabilities when someone asks what the demo can do.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }}],
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = request_json(base.rstrip("/") + "/chat/completions", payload, timeout=120)
    choices = response.get("choices") or []
    calls = (choices[0].get("message") or {}).get("tool_calls") if choices else None
    if not isinstance(calls, list) or len(calls) != 1:
        raise ProofError("model did not emit one native tool_call; XML in prose is not OpenClaw-compatible")
    function = calls[0].get("function") or {}
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, ValueError) as exc:
        raise ProofError("native tool arguments are not JSON") from exc
    if calls[0].get("type") != "function" or function.get("name") != "cascade_readiness" or not isinstance(arguments, dict) or len(arguments) != 1 or arguments.get("ready") is not True:
        raise ProofError("native tool call did not match the readiness request")
    return {**info, "native_tools": True}


def check_brain(brain: str) -> dict:
    endpoints = {
        "cosmos": (os.environ.get("CASCADE_COSMOS_BASE_URL", "http://127.0.0.1:8082/v1"), "cosmos3-edge"),
        "cosmos-sglang": (os.environ.get("CASCADE_COSMOS_SGLANG_BASE_URL", "http://127.0.0.1:8083/v1"), "cosmos3-edge"),
        "qwen": (os.environ.get("CASCADE_QWEN_BASE_URL", "http://127.0.0.1:8080/v1"),
                 "Qwen/Qwen3.8-27B" if os.environ.get("CASCADE_INSTALL_PROFILE") == "spark" else None),
    }
    if brain in endpoints:
        context = 32768 if os.environ.get("CASCADE_INSTALL_PROFILE") == "spark" else (65536 if brain == "qwen" else 32768)
        return {**local_brain(*endpoints[brain]), "brain": brain, "context_window": context}
    if brain not in ("auto", "keep"):
        raise ProofError(f"unknown brain: {brain}")
    if brain == "auto":
        for name in ("cosmos", "qwen"):
            try:
                return check_brain(name)
            except EndpointUnavailable:
                pass
    p = subprocess.run(oc_command("models", "status", "--json"), capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise ProofError("OpenClaw model configuration is unavailable")
    try:
        d = json.loads(p.stdout)
    except ValueError as exc:
        raise ProofError("OpenClaw models status returned invalid JSON") from exc
    model = d.get("resolvedDefault") or d.get("defaultModel")
    if not model:
        raise ProofError("OpenClaw has no model; configure one explicitly")
    return {"brain": "keep", "model": model, "note": "configured; inference is tested at launch, not by read-only check"}


def validate_pick_trace(path, started: float, target: str, *, placement_audit=None):
    """Read ONLY the trace explicitly bound to this session's live process."""
    from pathlib import Path

    path = Path(path)
    if path.name != "trace.jsonl" or path.is_symlink():
        raise ProofError("an explicitly bound current trace.jsonl is required (not a runs directory)")
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ProofError(f"unreadable current trace: {path}") from exc
    if placement_audit is not None:
        current = [row for row in rows if row.get("t", 0) >= started and row.get("skill") == "pick_and_place"]
        if len(current) != 1:
            raise ProofError("Spark acceptance requires exactly one current pick_and_place action")
    candidates = []
    for row in rows:
        if row.get("t", 0) < started or row.get("skill") != "pick_and_place":
            continue
        result = row.get("result") or {}
        pc = result.get("postcondition") or {}
        if (row.get("args") or {}).get("object", "").strip().casefold() != target.strip().casefold():
            continue
        confirmed = (result.get("verified") is True and pc.get("status") == "confirmed"
                     and pc.get("channel") == "physics")
        if placement_audit is not None:
            # Bounded destinations need the passive scene-bound geometry audit.
            # A refuted or failed native action can never be rescued by it.
            confirmed = (placement_audit.get("pass") is True
                         and placement_audit.get("object_name") == "_".join(target.casefold().split())
                         and placement_audit.get("destination_name") == (row.get("args") or {}).get("destination")
                         and pc.get("status") in ("confirmed", "unverified")
                         and pc.get("channel") == "physics")
        if result.get("ok") is True and confirmed:
            candidates.append(path)
    if len(candidates) != 1:
        raise ProofError(f"expected one current physics-confirmed pick for {target!r}; found {len(candidates)}")
    return candidates[0]


def validate_reset_trace(path, started: float, sim: str, target: str) -> list[str]:
    """Reset must target the exact process/world used for the pick proof."""
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ProofError(f"reset trace unavailable: {path}") from exc
    resets = [r for r in rows if r.get("t", 0) >= started and r.get("skill") == "reset_scene"]
    if len(resets) != 1:
        raise ProofError("expected one current reset in the pick world")
    result = resets[0].get("result") or {}
    props = result.get("props_reset")
    if result.get("ok") is not True or result.get("world") != sim or not isinstance(props, list) or not props:
        raise ProofError("reset did not restore simulation props")
    expected_prop = "_".join(target.casefold().split())
    if not expected_prop or expected_prop not in props:
        raise ProofError(f"reset did not restore manipulated prop {expected_prop!r}; got {props!r}")
    return props


def validate_envelope(data: dict, required_tool: str | tuple[str, ...] | None = None) -> dict:
    if data.get("status") != "ok":
        raise ProofError("OpenClaw turn did not complete successfully")
    result = data.get("result") or {}
    meta = result.get("meta") or {}
    summary = meta.get("toolSummary")
    if meta.get("aborted"):
        raise ProofError("OpenClaw turn aborted")
    if summary is None and required_tool is None:
        return result  # actual OpenClaw schema for a zero-tool text turn
    if not isinstance(summary, dict) or type(summary.get("calls")) is not int or type(summary.get("failures")) is not int:
        raise ProofError("OpenClaw turn has no valid toolSummary")
    if meta.get("aborted") or summary["failures"] != 0:
        raise ProofError("OpenClaw turn aborted or had failed tools")
    required = (required_tool,) if isinstance(required_tool, str) else (required_tool or ())
    prefix = os.environ.get("CASCADE_MCP_NAME", "cascade") + "__"
    tools = summary.get("tools", [])
    for tool in required:
        if not isinstance(tools, list) or prefix + tool not in tools:
            raise ProofError(f"OpenClaw did not execute {prefix + tool}")
    if summary["calls"] < len(required):
        raise ProofError("OpenClaw toolSummary has fewer calls than required tools")
    return result


def proof_timeout(timeout: int) -> int:
    try:
        scale = float(os.environ.get("CASCADE_PROOF_TIMEOUT_SCALE", "1"))
    except ValueError as exc:
        raise ProofError("CASCADE_PROOF_TIMEOUT_SCALE must be finite and between 1 and 4") from exc
    if not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ProofError("CASCADE_PROOF_TIMEOUT_SCALE must be finite and between 1 and 4")
    return math.ceil(timeout * scale)


def agent_turn(session: str, model: str, message: str, output, timeout: int, required_tool=None) -> dict:
    timeout = proof_timeout(timeout)
    command = oc_command("agent", "--session-id", session, "--model", model,
                         "--message", message, "--json", "--timeout", str(timeout))
    p = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 30)
    output.write_text(p.stdout)
    output.with_suffix(".stderr.txt").write_text(p.stderr)
    if p.returncode:
        raise ProofError(f"OpenClaw failed; see {output.with_suffix('.stderr.txt')}")
    try:
        result = validate_envelope(json.loads(p.stdout), required_tool)
    except ValueError as exc:
        raise ProofError(f"OpenClaw returned invalid JSON; see {output}") from exc
    meta = result.get("meta", {}).get("agentMeta", {})
    if meta.get("sessionId") != session:
        raise ProofError(f"session changed: requested {session}, effective {meta.get('sessionId')}")
    actual = f"{meta.get('provider')}/{meta.get('model')}"
    if actual != model:
        raise ProofError(f"model changed: requested {model}, effective {actual}")
    return result


def _write_receipt(report: dict, state_dir) -> None:
    temporary = state_dir / (report["session_id"] + ".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(state_dir / "proof.json")


def _bound_world(state_dir, owner, baseline, started, expected=None) -> dict:
    from pathlib import Path
    from cascade.apps.process_owner import live_records

    fresh = [r for r in live_records(state_dir, owner, role="mcp")
             if r.get("instance_id") not in baseline and r.get("registered_at", 0) >= started]
    if len(fresh) != 1 or (expected and fresh[0] != expected):
        raise ProofError("expected exactly one NEW live MCP of this launch owner; cannot bind the session to a world")
    record = fresh[0]
    run_dir = Path(record.get("run_dir", "")).resolve()
    if run_dir.parent != (Path(owner["repo"]) / "runs").resolve() or not run_dir.name.startswith(f"mcp_{record['pid']}_"):
        raise ProofError("owner record has an invalid runtime trace directory")
    return record


def make_spark_witness(repo, evidence, *, object_name, destination_name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cascade_spark_placement_proof", repo / "demo/kitchen/physics/spark_proof.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SparkKitchenWitness(
        evidence / "physics", scene_config=repo / "demo/scene/kitchen_config.json",
        object_name=object_name, destination_name=destination_name,
        port=int(os.environ.get("CASCADE_BRIDGE_PORT", "8611")))


def _run_spark_cases(repo, state_dir, evidence, report, owner, baseline):
    """Two native OpenClaw orders, each followed by an observed same-world reset."""
    from pathlib import Path
    import time

    model, session = report["model"], report["session_id"]
    prefix = os.environ.get("CASCADE_MCP_NAME", "cascade") + "__"

    def turn(message, tool, output, timeout=150):
        # Expected tools validate the completed result; they never alter the
        # visitor's message or constrain native model selection.
        result = agent_turn(session, model, message, output, timeout, tool)
        summary = result["meta"]["toolSummary"]
        read_tools = {prefix + name for name in
                      ("describe_scene", "localize_object", "camera_snapshot", "world_state")}
        if not set(summary["tools"]).issubset(read_tools | {prefix + tool}):
            raise ProofError("Spark acceptance used an unexpected action or non-attendee tool")
        # Current owner-bound trace validators count motion/reset calls. A
        # native inspection before that action is allowed when needed.
        return result

    turn("What can you see on the table?", "describe_scene", evidence / "01-scene.json")
    process = _bound_world(state_dir, owner, baseline, report["started_at"])
    trace = Path(process["run_dir"]) / "trace.jsonl"
    report.update(process=process, trace=str(trace), cases=[])
    for number, (target, destination) in enumerate(
            (("green cube", "green square"), ("orange", "open box")), 1):
        case_dir = evidence / f"case-{number}"
        case_dir.mkdir()
        witness = make_spark_witness(repo, case_dir,
                                    object_name="_".join(target.split()), destination_name=destination)
        with witness:
            witness.mark("pick_begin")
            started = time.time()
            message = ("Could you put the green cube in the green square?" if number == 1
                       else "Please put the orange in the open box.")
            turn(message, "pick_and_place", case_dir / "02-pick.json", 300)
            _bound_world(state_dir, owner, baseline, report["started_at"], process)
            witness.settle(wall_timeout=90)
            witness.mark("pick_end")
            witness.capture_frames("placed")
            reset_started = time.time()
            witness.mark("reset_begin")
            turn("Let's start over.", "reset_scene", case_dir / "03-reset.json", 240)
            _bound_world(state_dir, owner, baseline, report["started_at"], process)
            props = validate_reset_trace(trace, reset_started, "isaac", target)
            turn("What is the session status?", "world_state", case_dir / "04-world-reset.json")
            turn("What can you see on the table?", "describe_scene", case_dir / "05-inspect-reset.json")
            _bound_world(state_dir, owner, baseline, report["started_at"], process)
            witness.settle(wall_timeout=90)
            witness.mark("reset_end")
            witness.capture_frames("reset")
        physical = witness.audit()
        validate_pick_trace(trace, started, target, placement_audit=physical)
        report["cases"].append({"object": target, "destination": destination,
                                "physics": physical, "props_reset": props,
                                "evidence_dir": str(case_dir)})
        _write_receipt(report, state_dir)
    report.update(verified=True, props_reset=report["cases"][-1]["props_reset"])


def run_proof(repo, state_dir, sim: str, robot_turn: bool = True) -> dict:
    """Run the REAL host, keeping pick and reset in one persistent session."""
    from pathlib import Path
    import time
    import uuid

    repo, state_dir = Path(repo), Path(state_dir)
    session = "cascade-proof-" + uuid.uuid4().hex
    evidence = state_dir / session
    evidence.mkdir(parents=True, mode=0o700)
    report = {"verified": False, "model": None, "sim": sim, "session_id": session,
              "started_at": time.time(), "evidence_dir": str(evidence),
              "profile": os.environ.get("CASCADE_OPENCLAW_PROFILE", "")}
    _write_receipt(report, state_dir)  # invalidate the previous success BEFORE any failing probe
    owner, baseline = None, set()
    if robot_turn and sim != "none":
        from cascade.apps.process_owner import load_owner, records

        owner = load_owner(state_dir, repo, report["profile"])
        if owner is None:
            raise ProofError("no launch owner: register the MCP with --launch-owner before the proof")
        baseline = {r.get("instance_id") for r in records(state_dir)}
    model = check_brain("keep")["model"]
    report["model"] = model
    print(f"[proof] brain={model} session={session}", file=sys.stderr, flush=True)
    if robot_turn and sim == "isaac" and os.environ.get("CASCADE_INSTALL_PROFILE") == "spark":
        _run_spark_cases(repo, state_dir, evidence, report, owner, baseline)
        (evidence / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
        _write_receipt(report, state_dir)
        return report
    brain = agent_turn(session, model, "Reply with exactly OK and nothing else. Do not call tools.", evidence / "01-brain.json", 120)
    text = " ".join(p.get("text", "") for p in brain.get("payloads", [])).strip()
    if text != "OK":
        raise ProofError(f"brain did not answer exactly OK; see {evidence}")
    if robot_turn and sim != "none":
        target = "pink cube" if sim == "isaac" else "red cube"
        agent_turn(session, model,
                   "What is the session status?", evidence / "02-world.json", 150, "world_state")
        process = _bound_world(state_dir, owner, baseline, report["started_at"])
        trace = Path(process["run_dir"]) / "trace.jsonl"
        report["process"] = process
        started = time.time()
        print(f"[proof] picking {target} (physics confirmation required)", file=sys.stderr, flush=True)
        agent_turn(session, model,
                   f"Please put the {target} in the drop zone.",
                   evidence / "03-pick.json", 300, "pick_and_place")
        _bound_world(state_dir, owner, baseline, report["started_at"], process)
        validate_pick_trace(trace, started, target)
        reset_started = time.time()
        print("[proof] resetting the SAME world", file=sys.stderr, flush=True)
        agent_turn(session, model,
                   "Let's start over, then show the session status.",
                   evidence / "04-reset.json", 150, ("reset_scene", "world_state"))
        _bound_world(state_dir, owner, baseline, report["started_at"], process)
        props = validate_reset_trace(trace, reset_started, sim, target)
        report.update(verified=True, trace=str(trace), props_reset=props)
    else:
        report["note"] = "robot proof explicitly skipped; this is not a verified demo"
    (evidence / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    # Single atomic pointer for the launcher/operator. Detailed evidence remains
    # in the immutable per-attempt directory when a later attempt fails.
    _write_receipt(report, state_dir)
    return report


def main() -> int:
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-brain", choices=["auto", "keep", "cosmos", "cosmos-sglang", "qwen"])
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--probe-native", metavar="BASE_URL")
    mode.add_argument("--wait-gateway", type=float, metavar="SECONDS")
    p.add_argument("--model", default="cosmos3-edge")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--state-dir", type=Path)
    p.add_argument("--sim", choices=["isaac", "mujoco", "none"], default="mujoco")
    p.add_argument("--no-robot-turn", action="store_true")
    args = p.parse_args()
    try:
        if args.check_brain:
            result = check_brain(args.check_brain)
        elif args.probe_native:
            result = probe_native_tools(args.probe_native, args.model)
        elif args.wait_gateway is not None:
            result = wait_gateway(args.wait_gateway)
        else:
            from cascade.apps.process_owner import profile_state_dir

            state_dir = args.state_dir or profile_state_dir(
                os.environ.get("CASCADE_LAUNCH_STATE", args.repo / "runs/.launch"),
                os.environ.get("CASCADE_OPENCLAW_PROFILE", ""),
            )
            result = run_proof(args.repo, state_dir, args.sim, not args.no_robot_turn)
        print(json.dumps(result))
    except (ProofError, OSError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
