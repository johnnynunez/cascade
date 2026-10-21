#!/usr/bin/env python3
"""Replay a recorded humanoid segment's proposed actions through today's humanoid safety gate, offline.

Run it on a recorded Arena run before the first real Arena run on new gate code: it reports, per
commanded channel, the first step whose recorded action the gate (``HumanoidSafetyGate.check``)
would refuse now, so a limit that would stop the real policy shows up here and not mid-run.

    python scripts/replay_gate_jsonl.py runs/<run>/domains/carry/segments/001/gate.jsonl
    python scripts/replay_gate_jsonl.py <gate.jsonl> --json                 # machine-readable report
    python scripts/replay_gate_jsonl.py <gate.jsonl> --recorded-limits      # the limits the run used

Inputs: ``gate.jsonl`` (one line per control step: the gate's rows and the proposed ``action``) and
the domain's ``session.json`` two directories up (the bridge's joint model; ``--session`` names
another). Limits: the robot profile's (``--robot``/``--domain``, default ``g1_arena_locomanip`` /
``carry``: what the next run will use), or the run's own with ``--recorded-limits``.

``gate.jsonl`` records the proposed action, not the measured state, so each action is paired with a
neutral measured state (upright at the workspace centre, joints at rest inside their range, still):
only the commanded channels decide. Channels: ``command_joint_rad`` (arm/waist targets),
``command_wbc_joint_rad`` (leg targets the whole-body controller owns), ``command_nav``,
``command_base_height_m``, ``command_torso_rpy``, and ``state_finite`` (a malformed or non-finite
action). Finger targets are clamped, never refused; the report counts the clamps.

Edge case to look for: leg targets are limited like arm targets, range + ``command_joint_tolerance_rad``
(0.02 rad), while a MEASURED joint may sit up to ``joint_position_tolerance_rad`` (0.05 rad) past its
range. A whole-body-controller leg target 0.02-0.05 rad past its range is refused although the leg
could physically stand there. That is deliberate (fail closed: 0.02 stays); this replay shows whether
the policy's actions hit it.

Exit status: 0 = no channel refused, 1 = at least one refused step, 2 = unusable input.
CPU only; no simulator, bridge, model or network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cascade.safety.humanoid_gate import HumanoidGateLimits, HumanoidSafetyGate, JointModel  # noqa: E402

#: the gate rows that judge the proposed action (``state_finite``: malformed or non-finite)
CHANNELS = ("state_finite", "command_joint_rad", "command_wbc_joint_rad", "command_nav", "command_base_height_m",
            "command_torso_rpy")


def neutral_state(step: int, limits: HumanoidGateLimits, joints: JointModel) -> dict:
    """A measured state every state row passes: upright, inside the workspace, joints at rest in range."""
    (x0, y0), (x1, y1) = limits.workspace_xy_m
    return {"step": step, "floor_z": 0.0,
            "pelvis_pos": [(x0 + x1) / 2, (y0 + y1) / 2, limits.min_pelvis_height_m + 0.3],
            "pelvis_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_pos": [min(max(0.0, lo), hi) for lo, hi in zip(joints.lower, joints.upper)],
            "joint_vel": [0.0] * len(joints.names)}


def replay(lines: list[dict], limits: HumanoidGateLimits, joints: JointModel) -> dict:
    """Check every recorded action; the first refusal per channel, refusal counts and finger clamps."""
    gate = HumanoidSafetyGate(limits, joints)
    channels = {name: {"first": None, "refused_steps": 0} for name in CHANNELS}
    clamps = {"steps": 0, "targets": 0}
    actions = skipped = 0
    for number, line in enumerate(lines, 1):
        if "action" not in line:
            skipped += 1  # the pre-motion admission line carries no action
            continue
        actions += 1
        decision = gate.check(number, neutral_state(number, limits, joints), line["action"])
        gate.reset()  # every step is judged on its own: the first refusal per channel, not per run
        if decision.clamps:
            clamps["steps"] += 1
            clamps["targets"] += len(decision.clamps)
        for row in decision.rows:
            if row.ok or row.name not in channels:
                continue
            entry = channels[row.name]
            entry["refused_steps"] += 1
            if entry["first"] is None:
                entry["first"] = {"step": line.get("step"), "phase": line.get("phase"), "line": number,
                                  "value": row.value, "limit": row.limit, "detail": row.detail,
                                  "recorded_verdict": line.get("verdict")}
    return {"actions": actions, "skipped": skipped, "channels": channels, "finger_clamps": clamps}


def _read_jsonl(path: Path) -> list[dict]:
    lines = []
    for number, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: not a JSON object")
        lines.append(value)
    return lines


def _limits(args, session: dict) -> tuple[HumanoidGateLimits, str]:
    if args.recorded_limits:
        return HumanoidGateLimits.from_profile(session["limits"]), f"recorded session {args.session}"
    from cascade.config import load_robot_config

    safety = load_robot_config(args.robot).as_dict()["domains"][args.domain]["safety"]
    return HumanoidGateLimits.from_profile(safety), f"robot profile {args.robot} (domain {args.domain})"


def _text(report: dict) -> str:
    out = [f"{report['actions']} recorded actions replayed ({report['skipped']} lines without an action skipped); "
           f"limits: {report['limits']['source']}"]
    for name, entry in report["channels"].items():
        first = entry["first"]
        if first is None:
            out.append(f"  {name:22s} never refused")
        else:
            value = "malformed" if first["value"] is None else f"{first['value']:.4f}"
            out.append(f"  {name:22s} first refused at step {first['step']} ({first['phase']}, line {first['line']}): "
                       f"{value} vs limit {first['limit']}{' (' + first['detail'] + ')' if first['detail'] else ''}; "
                       f"{entry['refused_steps']} refused steps")
    clamps = report["finger_clamps"]
    out.append(f"  finger targets clamped (not a refusal): {clamps['targets']} in {clamps['steps']} steps")
    refused = [name for name, entry in report["channels"].items() if entry["first"] is not None]
    out.append(f"refused channels: {', '.join(refused)}" if refused else "no channel refused")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gate_jsonl", type=Path, help="a recorded segments/NNN/gate.jsonl")
    ap.add_argument("--session", type=Path, default=None,
                    help="the domain's session.json (default: two directories above gate.jsonl)")
    ap.add_argument("--robot", default="g1_arena_locomanip", help="robot profile whose limits to apply")
    ap.add_argument("--domain", default="carry", help="the humanoid_locomanip domain in that profile")
    ap.add_argument("--recorded-limits", action="store_true", help="apply the limits recorded in session.json")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    args.session = args.session or args.gate_jsonl.parent.parent.parent / "session.json"
    try:
        lines = _read_jsonl(args.gate_jsonl)
        session = json.loads(args.session.read_text())
        joints = JointModel.from_dict(session["hello"]["joints"])
        limits, source = _limits(args, session)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"replay_gate_jsonl: unusable input: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report = replay(lines, limits, joints)
    report["limits"] = {"source": source, **limits.as_dict()}
    report["gate_jsonl"], report["session"] = str(args.gate_jsonl), str(args.session)
    print(json.dumps(report, indent=1) if args.json else _text(report))
    return 1 if any(entry["first"] is not None for entry in report["channels"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
