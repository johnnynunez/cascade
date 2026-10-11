"""Summarise contact-monitor reports and suggest thresholds (offline only).

The motor-feedback contact monitor (`safety.contact_monitor`,
docs/CONTACT_STOP.md) files one report per streamed motion: per joint the
worst lag-compensated tracking error, the worst raw error (measured minus
the instantaneous command), the longest lag the measured joint trailed its
command by, and the lowest stall advance ratio. In `mode: measure` it never
stops a motion. Run a few normal moves on the rig in measure mode, then
point this at what they left behind:

* run traces -- each `trace.jsonl` row carries `context.contact_monitor`;
* report logs -- one report per line, e.g. from
  `scripts/home_rebot_rs.py --contact-log rig.jsonl`.

It prints the per-joint worst values, every motion that WOULD have tripped,
and a suggested `contact_monitor` block:

* `lag_s`: --lag-margin x the longest lag seen (0.10..0.40 s, 0.05 steps);
* `track_tol_rad` per joint: --margin x the worst lag-compensated error,
  never below --settle-tol + 0.02 rad (the monitor must not stop a motion
  whose final error the settle check accepts).

The errors were measured at the profile's lag_s at the time; when the
suggested lag is shorter, re-run the measure moves with it before trusting
the tolerances. Reads files only: no arm, no bus, no services.

    python scripts/contact_calibrate.py rig.jsonl runs/*/trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_reports(paths) -> list[dict]:
    """Monitor reports from trace.jsonl rows or plain report lines."""
    out: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            nested = (row.get("context") or {}).get("contact_monitor")
            if isinstance(nested, list):
                out.extend(r for r in nested if isinstance(r, dict))
            elif "max_track_err_rad" in row:
                out.append(row)
    return out


def _per_joint(reports, key, pick):
    vectors = [r.get(key) or [] for r in reports]
    n = max((len(v) for v in vectors), default=0)
    out = []
    for j in range(n):
        vals = [float(v[j]) for v in vectors if j < len(v) and v[j] is not None]
        out.append(pick(vals) if vals else None)
    return out


def _ceil(x: float, step: float) -> float:
    return round(math.ceil(round(x / step, 6)) * step, 3)


def suggest(reports, *, settle_tol=0.05, margin=3.0, lag_margin=1.5) -> dict:
    """The calibration summary and the suggested thresholds."""
    measured = [r for r in reports if r.get("samples") and sum(r["samples"]) > 0]
    worst_err = _per_joint(measured, "max_track_err_rad", max)
    worst_lag = _per_joint(measured, "max_effective_lag_s", max)
    per_cycle = [r["reads"]["per_cycle"] for r in measured
                 if isinstance(r.get("reads"), dict) and r["reads"].get("per_cycle")]
    lags = sorted({float(r["lag_s"]) for r in measured if r.get("lag_s") is not None})
    s = {
        "motions": len(measured),
        "measured_at_lag_s": lags,
        "max_track_err_rad": worst_err,
        "max_raw_err_rad": _per_joint(measured, "max_raw_err_rad", max),
        "max_effective_lag_s": worst_lag,
        "min_advance_ratio": _per_joint(measured, "min_advance_ratio", min),
        "stall_ratio": next((r.get("stall_ratio") for r in measured if r.get("stall_ratio")), None),
        "reads_per_cycle": round(sum(per_cycle) / len(per_cycle), 2) if per_cycle else None,
        "max_feedback_gap_s": max((float(r.get("max_feedback_gap_s") or 0.0) for r in measured),
                                  default=None),
        "read_failures": sum(int(r.get("read_failures") or 0) for r in measured),
        "would_trip": [f"{r.get('motion')}: {r['would_trip']} ({r.get('would_trips', 1)} sample(s))"
                       for r in measured if r.get("would_trip")],
        "tripped": [r["tripped"] for r in measured if r.get("tripped")],
        "suggested": None,
    }
    if measured:
        lag_seen = max((v for v in worst_lag if v is not None), default=0.0)
        lag = min(0.40, max(0.10, _ceil(lag_margin * lag_seen, 0.05)))
        floor = settle_tol + 0.02
        tol = [round(max(_ceil(margin * (e or 0.0), 0.005), floor), 3) for e in worst_err]
        s["suggested"] = {"lag_s": lag, "track_tol_rad": tol}
    return s


def _vec(v, fmt="{:.3f}"):
    return "[" + " ".join("-" if x is None else fmt.format(x) for x in v) + "]"


def render(s: dict) -> str:
    if not s["motions"]:
        return ("0 monitored motions: nothing to calibrate from (is safety.contact_monitor "
                "enabled for this profile, and did the moves stream?)")
    lines = [
        f"{s['motions']} monitored motion(s), measured at lag_s {s['measured_at_lag_s']}",
        f"worst lag-compensated tracking error (rad): {_vec(s['max_track_err_rad'])}",
        f"worst raw error vs the instantaneous command (rad): {_vec(s['max_raw_err_rad'])}",
        f"longest lag behind the command (s): {_vec(s['max_effective_lag_s'])}",
        (f"lowest stall advance ratio: {_vec(s['min_advance_ratio'])} "
         f"(stall_ratio {s['stall_ratio']})"),
        (f"reads per cycle {s['reads_per_cycle']}; longest feedback gap "
         f"{s['max_feedback_gap_s']} s; read failures {s['read_failures']}"),
    ]
    ratio = s["stall_ratio"] or 0.0
    close = [j + 1 for j, x in enumerate(s["min_advance_ratio"]) if x is not None and x < 2 * ratio]
    if close:
        lines.append(f"WARNING: joint(s) {close} came within 2x of stall_ratio on these moves; "
                     "were they all free motions? If so raise stall_min_travel_rad.")
    for key in ("tripped", "would_trip"):
        for msg in s[key]:
            lines.append(f"{key.upper()}: {msg}")
    if s["would_trip"]:
        lines.append("Every WOULD_TRIP above must be explained (a real contact?) before enforcing.")
    sug = s["suggested"]
    lines += ["suggested (configs/arms/<profile>.yaml, overrides.safety.contact_monitor):",
              f"  lag_s: {sug['lag_s']}",
              f"  track_tol_rad: {sug['track_tol_rad']}"]
    if s["measured_at_lag_s"] and sug["lag_s"] < min(s["measured_at_lag_s"]):
        lines.append("  (shorter than the lag these errors were measured at: re-run the measure "
                     "moves with it before trusting the tolerances)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="trace.jsonl files and/or report logs")
    ap.add_argument("--settle-tol", type=float, default=0.05, help="the arm's settle_tol (rad)")
    ap.add_argument("--margin", type=float, default=3.0, help="tolerance = margin x worst error")
    ap.add_argument("--lag-margin", type=float, default=1.5, help="lag_s = this x longest lag")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)
    s = suggest(load_reports(args.paths), settle_tol=args.settle_tol, margin=args.margin,
                lag_margin=args.lag_margin)
    print(json.dumps(s, indent=2) if args.json else render(s))
    return 0 if s["motions"] else 1


if __name__ == "__main__":
    sys.exit(main())
