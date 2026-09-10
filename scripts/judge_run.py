#!/usr/bin/env python3
"""Judge a recorded cascade run with a Robo-Dopamine-style progress model.

    scripts/judge_run.py runs/<run_dir> [--judge grm|vlm|fake] [--model ...]
                          [--base-url ...] [--mode incremental|forward|backward]
                          [--ref-end goal.jpg] [--skills pick_and_place,...]

Writes <run_dir>/judge.json and prints: per-step hop next to the physics
verdict, the fused progress curve, the judge-vs-physics confusion matrix,
and hop-per-second per dispatch tier. Reads configs/demo.yaml `eval.judge`
for defaults; flags override. Offline -- never touches the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--judge", choices=["grm", "vlm", "fake"], help="override eval.judge.backend")
    ap.add_argument("--model", help="model name (GRM served name, or the VLM)")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (vLLM serving GRM, local Cosmos3, ...)")
    ap.add_argument("--mode", default=None, choices=["incremental", "forward", "backward"])
    ap.add_argument("--ref-end", help="reference END (goal) image; blank when omitted, as upstream")
    ap.add_argument("--skills", help="comma-separated skills to judge (default: all traced calls)")
    ap.add_argument("--fake-score", type=float, default=None, help="with --judge fake: the constant hop")
    ap.add_argument("--strict", action="store_true", help="exit 3 when the judge missed a physics-confirmed step (fn > 0)")
    args = ap.parse_args()

    from cascade.config import load_demo_config
    from cascade.eval.progress_judge import JudgeError, judge_run, make_judge

    cfg = load_demo_config()
    ev = cfg.get("eval")
    jd = ev.get("judge") if ev is not None else None
    ecfg = jd.as_dict() if jd is not None and hasattr(jd, "as_dict") else dict(jd or {})
    if args.judge:
        ecfg["backend"] = args.judge
    if args.model:
        ecfg["model"] = args.model
    if args.base_url:
        ecfg["base_url"] = args.base_url
    if args.fake_score is not None:
        ecfg["score"] = args.fake_score
    mode = args.mode or str(ecfg.get("mode", "incremental"))
    # OpenClaw gateway path: fill OPENCLAW_GATEWAY_TOKEN from the local
    # config when the judge is pointed at the gateway and the env is empty.
    import os

    if str(ecfg.get("api_key", "")) == "$OPENCLAW_GATEWAY_TOKEN" and not os.environ.get("OPENCLAW_GATEWAY_TOKEN"):
        oc = Path.home() / ".openclaw" / "openclaw.json"
        if oc.exists():
            try:
                tok = (json.loads(oc.read_text()).get("gateway") or {}).get("auth", {}).get("token")
                if tok:
                    os.environ["OPENCLAW_GATEWAY_TOKEN"] = str(tok)
            except Exception:  # noqa: BLE001
                pass
    try:
        judge = make_judge(ecfg)
    except JudgeError as e:
        print(f"error: {e}\n(set eval.judge in configs/demo.yaml or pass --judge/--model/--base-url)", file=sys.stderr)
        return 2
    ref_end = Path(args.ref_end).read_bytes() if args.ref_end else None
    skills = set(args.skills.split(",")) if args.skills else None
    v = judge_run(args.run_dir, judge, mode=mode, ref_end=ref_end, skills=skills)

    print(f"run: {v.run_dir}\njudge: {v.judge}   mode: {v.mode}")
    print(f"{'step':>4} {'skill':<22} {'hop':>7} {'physics':<10} {'claim':<6} {'tier':<10} {'s':>6}")
    for s in v.steps:
        hop = f"{s.hop:+.2f}" if s.hop is not None else "  --  "
        phys = s.physics or ("-" if s.channel is None else s.channel)
        agree = s.agrees_with_physics()
        mark = "" if agree is None else (" ok" if agree else " DISAGREE")
        print(f"{s.step:>4} {s.skill:<22} {hop:>7} {phys:<10} {str(s.ok):<6} {str(s.tier or '-'):<10} {s.duration_s or 0:>6.1f}{mark}"
              + (f"   [{s.error}]" if s.error else ""))
    if v.progress:
        print("progress: " + " -> ".join(f"{p:.2f}" for p in v.progress) + f"   (final {v.final_progress:.2f})")
    c = v.confusion()
    agreement = "n/a" if c["agreement"] is None else f"{c['agreement']:.0%}"
    print(f"judge vs physics: n={c['n_with_physics']} agreement={agreement} "
          f"tp={c['tp']} tn={c['tn']} fp={c['fp']} fn={c['fn']}   (scored {c['n_scored']} of {len(v.steps)} steps)")
    for tier, d in v.per_tier().items():
        hps = f"{d['hop_per_s']:+.3f}/s" if d["hop_per_s"] is not None else "n/a"
        print(f"tier {tier:<10} n={d['n']} hop_mean={d['hop_mean']:+.2f} {hps}")
    out = v.write()
    print(f"wrote {out} and appended to summary.txt: {v.summary_line()}")
    # exit status is the metric: 3 = the judge missed physics-confirmed
    # progress (fn > 0), so a launcher or CI can gate on it
    return 3 if v.confusion()["fn"] > 0 and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
