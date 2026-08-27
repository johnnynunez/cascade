#!/usr/bin/env python3
"""Benchmark cascade ITSELF -- ablation over the layers you actually built.

WHY THIS FILE REPLACES THE LIBERO WORK
--------------------------------------
The LIBERO experiments measured a scripted pick-place primitive that this
harness author wrote inside LIBERO. That is a fine way to test *the Pigey
hypothesis*, but it is NOT a test of cascade: none of your code ran. The only
honest way to benchmark what you built is to run YOUR runtime, YOUR skills,
YOUR safety harness and YOUR verification on YOUR robot, and ablate YOUR
layers one at a time.

APPLES TO APPLES
----------------
Everything is held fixed across conditions except ONE layer:

  * same arm (Isaac B601-RS), same scene, same objects
  * same N initial states, replayed identically for every condition
    (placed via the bridge `exec` op, verified by physics before starting)
  * same task string
  * same success criterion, judged by PhysX -- never by the skill's own report

The independent variable is which cascade layer is enabled. That makes the
deltas attributable to a layer instead of to luck or to a different benchmark.

CONDITIONS (each adds one layer to the one above)

  A. skill-only        SkillRuntime.execute("pick_and_place") once. No agent,
                       no postcondition, no retry. The floor: what the frozen
                       skill does open-loop.
  B. +verification     Postconditions ON (agent/effects.py). A refuted effect
                       is reported, still no retry. Measures DETECTION.
  C. +retry            Postconditions drive up to 3 attempts. Measures
                       RECOVERY -- the orchestration gap, on your stack.
  D. +reflex agent     Full AgentOrchestrator, tier-1 regex reflex path
                       (no LLM). Adds task decomposition and re-planning.
  E. +LLM agent        Full orchestrator with the local Qwen3-VL. The complete
                       system as it runs at the booth.

SUCCESS is judged ONLY by TruthPoseReader (PhysX RigidPrim): the cube must end
inside the bin footprint. The skill's own `ok` flag is recorded separately so
we can quantify how often the system LIES about its own success -- which is
the metric this repo exists for.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import TruthPoseReader

#: Bin footprint in base frame (x_min, x_max, y_min, y_max) -- from the USD
#: scene, not guessed. A cube whose centre lands inside counts as placed.
BIN_XY = (0.11, 0.25, -0.24, -0.10)

#: Initial cube positions. Chosen to span the arm's top-down IK band
#: (x ~ 0.155-0.185 on this arm), NOT random: a position outside the envelope
#: is unreachable for every condition and would just add noise.
INIT_STATES = [
    (0.170, 0.150), (0.175, 0.130), (0.165, 0.170),
    (0.180, 0.140), (0.160, 0.160), (0.172, 0.120),
    (0.168, 0.180), (0.178, 0.155), (0.163, 0.135), (0.174, 0.165),
]

TASK = "pick up the pink cube and put it in the box"
CUBE, BIN = "pink_cube", "bin"


def place_cube(client: BridgeClient, x: float, y: float, z: float = 0.045):
    """Put the cube at an exact spot, on the sim's main thread.

    Zero the velocity in the SAME call as the teleport. A bare
    `set_world_poses` leaves whatever velocity the body had, and under Newton
    that momentum survives the teleport: the cube keeps moving from its new
    pose, tunnels, and the solver goes NaN -- which then poisons every later
    episode in the sweep (they all report an identical 0.0 cm because they are
    running against a dead scene, not because the skill behaved identically).

    Under Newton a plain teleport of a RESTING body does not stick either: the
    write reaches both of the solver's state buffers (verified by reading them
    back) and one step later the body is at its old pose again. Only a
    timeline Stop -> Play makes the solver honour the authored spawn, which is
    what the bridge's `reset_props` op now does.

    So: reset through the bridge first (returns every prop to its spawn), then
    nudge to the requested offset. If we skipped the reset, a sweep would
    silently reuse the previous episode's end state -- the cube stays in the
    bin, `truth=True` becomes a tautology, and the sweep reports a perfect
    score with the cube having "moved" 1.5 cm. That is not a hypothetical: it
    is exactly what a 6/6 run here turned out to be.
    """
    # generous timeout: reset can take a while on the sim's main thread
    client.request({"op": "reset_props"}, timeout_s=90.0)
    time.sleep(1.0)

    # Then place at the requested state through the bridge op. Do NOT poke
    # RigidPrim from here: under Newton that is a silent no-op for a resting
    # body, and this sweep previously ran all six "different" start states at
    # the spawn position -- one state measured six times, reported as six
    # independent episodes. Measured proof of that bug:
    #     asked (0.175,0.130) -> actual (0.170,0.150)   2.06 cm off
    #     asked (0.172,0.120) -> actual (0.170,0.150)   3.01 cm off
    client.request({"op": "place_prop", "name": "pink_cube",
                    "pos": [x, y, z]}, timeout_s=60.0)
    time.sleep(0.5)


def in_bin(pose) -> bool:
    if pose is None:
        return False
    x, y = float(pose[0]), float(pose[1])
    return BIN_XY[0] <= x <= BIN_XY[1] and BIN_XY[2] <= y <= BIN_XY[3]


def run_condition(rt, truth, client, condition: str, n_states: int) -> dict:
    """Run every initial state once under one condition."""
    from cascade.agent.orchestrator import AgentOrchestrator

    results = []
    for i, (x, y) in enumerate(INIT_STATES[:n_states]):
        place_cube(client, x, y)
        start = truth.pose(CUBE)

        # --- refuse to measure an episode that did not start where we asked -
        # Under Newton a teleport of a resting body does not reliably stick,
        # so the cube can still be wherever the LAST episode left it -- often
        # already in the bin. Then `truth=True` is a tautology and the sweep
        # reports a perfect score with the cube having "moved" 1.5 cm. That is
        # not a hypothetical; it is what a 6/6 run here turned out to be.
        if start is None:
            raise SystemExit(f"ABORT at episode {i}: no readable cube pose.")
        _off = float(np.linalg.norm(np.asarray(start[:2]) - np.asarray([x, y])))
        # 5 mm, NOT 3 cm. The INIT_STATES are only 2-3 cm apart, so a 3 cm
        # tolerance is wider than the sweep's own resolution: it passed happily
        # while every episode actually ran at the spawn position. A guard that
        # cannot distinguish state i from state j is not guarding anything.
        if _off > 0.005:
            raise SystemExit(
                f"ABORT at episode {i}: asked for ({x:.3f}, {y:.3f}) but the "
                f"cube is at ({start[0]:.3f}, {start[1]:.3f}), {_off*100:.1f} cm "
                f"away. The reset did not take, so this episode would be "
                f"measuring the previous one's end state. Restart the bridge."
            )

        # --- refuse to measure a dead scene --------------------------------
        # Once the articulation is NaN the whole scene is unrecoverable, and
        # every remaining episode returns an identical 0.0 cm. That looks like
        # a clean run of honest failures; it is actually one broken sim
        # reported six times. Abort loudly instead of emitting fake data.
        _q = np.asarray(client.request({"op": "state"})["q"], dtype=float)
        if np.isnan(_q).any():
            raise SystemExit(
                f"ABORT at episode {i}: the articulation is NaN, so every "
                f"later episode would be measuring a dead scene. Restart the "
                f"bridge and re-run."
            )

        # --- configure the layer under test -------------------------------
        # Postconditions live on the runtime; toggling `effects` is exactly
        # the difference between "the skill says ok" and "physics says ok".
        saved_effects = rt.effects
        if condition == "skill_only":
            rt.effects = None

        t0 = time.monotonic()
        self_reported_ok = None
        try:
            if condition in ("skill_only", "verify_only"):
                r = rt.execute("pick_and_place",
                               {"object": "pink cube", "destination": "box"})
                self_reported_ok = bool(r.get("ok"))
            elif condition == "verify_retry":
                for attempt in range(3):
                    r = rt.execute("pick_and_place",
                                   {"object": "pink cube", "destination": "box"})
                    self_reported_ok = bool(r.get("ok"))
                    pc = (r.get("postcondition") or {})
                    # ONLY an independently-refuted effect triggers a retry.
                    if pc.get("status") != "refuted":
                        break
            elif condition in ("reflex_agent", "llm_agent"):
                # NOTE the argument order: AgentOrchestrator(llm, runtime).
                # run_task returns a TaskReport dataclass, not a dict.
                agent = AgentOrchestrator(rt.llm, rt)
                report = agent.run_task(TASK)
                self_reported_ok = bool(getattr(report, "ok", True))
                path = getattr(report, "path", "?")
                print(f"        via: {path}", flush=True)
            else:
                raise ValueError(condition)
        except Exception as e:                        # a crash is a failure
            self_reported_ok = False
            r = {"error": f"{type(e).__name__}: {e}"}
        finally:
            rt.effects = saved_effects
        dt = time.monotonic() - t0

        # --- the ONLY verdict that counts ---------------------------------
        rt.execute("move_home", {})
        time.sleep(0.8)
        end = truth.pose(CUBE)
        truth_ok = in_bin(end)
        moved = (float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
                 if (start is not None and end is not None) else 0.0)

        results.append({
            "state": i, "init": [x, y],
            "truth_success": truth_ok,
            "self_reported_ok": self_reported_ok,
            "final_pose": [round(float(v), 4) for v in (end or [])],
            "moved_m": round(moved, 4),
            "seconds": round(dt, 1),
        })
        flag = "OK " if truth_ok else "   "
        lie = " <-- CLAIMED SUCCESS, PHYSICS SAYS NO" if (
            self_reported_ok and not truth_ok) else ""
        print(f"    [{i}] {flag} truth={truth_ok} self={self_reported_ok} "
              f"moved={moved*100:5.1f}cm {dt:5.1f}s{lie}", flush=True)

    n = len(results)
    succ = sum(r["truth_success"] for r in results)
    claimed = sum(bool(r["self_reported_ok"]) for r in results)
    lies = sum(1 for r in results
               if r["self_reported_ok"] and not r["truth_success"])
    return {
        "condition": condition, "n": n,
        "truth_successes": succ,
        "self_reported_successes": claimed,
        "false_claims": lies,
        "mean_seconds": round(sum(r["seconds"] for r in results) / max(1, n), 1),
        "episodes": results,
    }


MEASURE_LOCK = Path("/tmp/wrc_measuring.lock")


@contextlib.contextmanager
def claim_rig():
    """Hold the measurement lock so the watchdog cannot respawn night_runner.

    `scripts/night_runner.sh` drives the arm. When the cascade-watchdog cron
    revives it mid-run, joint readings become garbage that looks like a
    plausible asset bug rather than an obvious failure -- it has already
    produced one retracted "the gripper tops out at 54 mm instead of 71.5 mm"
    result (real error with the rig quiet: 0.002 mm on Newton, 0.088 mm on
    PhysX, full travel reached).

    Pausing the cron by hand is not enough; it gets re-enabled and forgotten.
    The watchdog stands down while this file exists (and ignores it after 6 h
    so a crashed benchmark cannot disable it forever).
    """
    pre_existing = MEASURE_LOCK.exists()
    if not pre_existing:
        MEASURE_LOCK.touch()
    try:
        yield
    finally:
        if not pre_existing:
            MEASURE_LOCK.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="skill_only,verify_only,verify_retry")
    ap.add_argument("--states", type=int, default=10)
    ap.add_argument("--llm", default="mock")
    ap.add_argument("--json", default="/home/johnny/bench/results/wrc_ablation.json")
    a = ap.parse_args()

    with claim_rig():
        return _run(a)


def _run(a) -> int:
    cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm=a.llm)
    cfg._data.setdefault("stream", {})["mode"] = "off"
    rt, arm = build_runtime(cfg, Path("/tmp/wrc_ablation"), view=False, serve=False)

    client = BridgeClient(port=8611)
    client.connect()
    truth = TruthPoseReader(client)
    rt.attach_verifier(object_pose=truth.pose)

    out = {}
    try:
        for cond in a.conditions.split(","):
            print(f"\n=== {cond} ({a.states} states) ===", flush=True)
            out[cond] = run_condition(rt, truth, client, cond, a.states)
            d = out[cond]
            print(f"  -> truth {d['truth_successes']}/{d['n']}   "
                  f"claimed {d['self_reported_successes']}/{d['n']}   "
                  f"FALSE CLAIMS {d['false_claims']}", flush=True)
    finally:
        shutdown_runtime(rt, arm)

    print("\n" + "=" * 66)
    print(f"{'condition':<16}{'truth':>10}{'claimed':>10}{'false':>8}{'mean s':>9}")
    print("-" * 66)
    for c, d in out.items():
        print(f"{c:<16}{d['truth_successes']:>4}/{d['n']:<5}"
              f"{d['self_reported_successes']:>5}/{d['n']:<4}"
              f"{d['false_claims']:>8}{d['mean_seconds']:>9}")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=2))
    print(f"\n[+] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
