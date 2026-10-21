"""Recommend the rig's no-marker rest tolerance from its logged rest readings.

Every reBot rest check (the startup check `scripts/check_rebot_gripper.py`, and every connect of
`rebot_rs` / `rebot_rs_mb`) reads every arm joint twice with parameter reads and appends the reading
to `~/.cascade/logs/rest_readings.jsonl` (`CASCADE_REST_LOG` overrides it; the one older file is
`<name>.1`). With no hold marker on file, a connect refuses an arm whose joints are not all within
`rest_check_tol_rad` of park_q (the arm profile; default and ceiling 0.12 rad), because motors a
killed process left ENABLED may still hold it up. The tighter that tolerance, the smaller the drop
such an arm takes when the connect's fault clear cuts its torque -- but a folded, unpowered arm must
still pass, or every cold start needs the operator.

This reads the log and recommends the tightest tolerance the measured folded, unpowered rest spread
allows:

- learns only from rows with NO hold marker (a held arm reads wherever it is held) whose two reads
  agree (cascade.control.arm_hold.READ_AGREE_RAD) and lie within the 0.12 ceiling;
- counts rows beyond the ceiling apart (a raised arm, or a rest the ceiling already refuses);
- recommends max(0.05, the largest |q - park_q| seen + a margin), rounded up to 0.01 rad, at most
  0.12. 0.05 rad is the robot server's own torque-release tolerance: an arm it released is within
  it, so a tighter value only adds refusals;
- recommends nothing until it has `--min-samples` readings.

    python scripts/rest_tolerance_report.py            # a table and the advice
    python scripts/rest_tolerance_report.py --json     # the same, machine-readable

Read only: it changes nothing. Set the value it recommends as `rest_check_tol_rad:` in the rig's arm
profiles (configs/arms/rebot_rs.yaml and rebot_rs_mb.yaml), then restart the robot server.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cascade.control import arm_hold  # noqa: E402

#: Readings needed before a recommendation (cold starts, startup checks).
MIN_SAMPLES = 20


def _rows(path: Path) -> list[dict]:
    rows = []
    for p in (path.with_name(path.name + ".1"), path):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _numbers(vals) -> list[float] | None:
    if not isinstance(vals, list) or not vals:
        return None
    out = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out


def analyse(rows, *, margin: float, min_samples: int) -> dict:
    """The report over logged rest rows (see the module docstring)."""
    samples, ignored, beyond = [], 0, 0
    for row in rows:
        q, again, park = _numbers(row.get("q")), _numbers(row.get("again")), _numbers(row.get("park"))
        if (row.get("marker") is not False or q is None or again is None or park is None
                or not len(q) == len(again) == len(park)
                or max(abs(a - b) for a, b in zip(q, again)) > arm_hold.READ_AGREE_RAD):
            ignored += 1
            continue
        dev = [max(abs(a - p), abs(b - p)) for a, b, p in zip(q, again, park)]
        if max(dev) > arm_hold.REST_TOL_RAD:
            beyond += 1
            continue
        samples.append(dev)
    out = {"samples": len(samples), "ignored": ignored, "beyond_tolerance": beyond, "margin_rad": margin,
           "min_samples": min_samples, "ceiling_rad": arm_hold.REST_TOL_RAD,
           "floor_rad": arm_hold.MARKER_CLEAR_TOL_RAD, "joints": [], "max_dev_rad": None, "worst_joint": None,
           "recommended_rad": None}
    if samples:
        n = len(samples[0])
        for j in range(n):
            col = sorted(s[j] for s in samples if len(s) == n)
            p95 = col[min(len(col) - 1, int(math.ceil(0.95 * len(col))) - 1)]
            out["joints"].append({"joint": j + 1, "n": len(col), "max_rad": round(col[-1], 6),
                                  "p95_rad": round(p95, 6)})
        worst = max(out["joints"], key=lambda r: r["max_rad"])
        out["max_dev_rad"], out["worst_joint"] = worst["max_rad"], worst["joint"]
    if len(samples) < min_samples:
        out["advice"] = (f"not enough readings ({len(samples)} of {min_samples}): keep the current value until "
                         "more cold starts are logged")
        return out
    want = math.ceil(round((out["max_dev_rad"] + margin) * 100.0, 6)) / 100.0
    rec = min(max(want, arm_hold.MARKER_CLEAR_TOL_RAD), arm_hold.REST_TOL_RAD)
    out["recommended_rad"] = round(rec, 2)
    if want > arm_hold.REST_TOL_RAD:
        out["advice"] = (f"the folded rest spreads up to {out['max_dev_rad']:.3f} rad (joint {out['worst_joint']}): "
                         f"keep {arm_hold.REST_TOL_RAD:g}, the ceiling; check that joint's rest")
    else:
        out["advice"] = (f"set rest_check_tol_rad: {out['recommended_rad']:g} in the rig's arm profiles "
                         "(configs/arms/rebot_rs.yaml, rebot_rs_mb.yaml)")
    return out


def _current() -> str:
    try:
        from cascade.config import load_demo_config

        tol = arm_hold.rest_tolerance(load_demo_config(camera="mock", arm="rebot_rs", llm="mock").arm)
        return f"{tol:g} rad (configs/arms/rebot_rs.yaml)"
    except Exception as exc:  # noqa: BLE001 -- a report: say it and go on
        return f"unknown ({exc})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None, help="the rest-readings log (default: CASCADE_REST_LOG or "
                                                "~/.cascade/logs/rest_readings.jsonl)")
    ap.add_argument("--margin", type=float, default=arm_hold.READ_AGREE_RAD,
                    help="rad added to the largest measured deviation (default: the read-agreement band)")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    path = Path(args.log).expanduser() if args.log else arm_hold.rest_log_path()
    report = {"log": str(path), **analyse(_rows(path), margin=args.margin, min_samples=args.min_samples)}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"rest readings: {path}")
    print(f"  learned from {report['samples']} folded, unpowered readings (no hold marker, the two reads agree); "
          f"ignored {report['ignored']}; beyond the {arm_hold.REST_TOL_RAD:g} rad ceiling {report['beyond_tolerance']}")
    for r in report["joints"]:
        print(f"  joint {r['joint']}: max |q - park_q| {r['max_rad']:.4f} rad, p95 {r['p95_rad']:.4f} ({r['n']} readings)")
    print(f"current: {_current()}")
    if report["recommended_rad"] is not None:
        print(f"recommended: rest_check_tol_rad: {report['recommended_rad']:g}")
    print(report["advice"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
