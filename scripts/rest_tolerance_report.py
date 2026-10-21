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
allows. It only ever TIGHTENS:

- it learns only from rows the check itself judged at rest: no hold marker (a held arm reads
  wherever it is held), the verdict 'rest' within the tolerance applied at that time, both reads
  numbers that agree (cascade.control.arm_hold.READ_AGREE_RAD). A row the check refused is never
  rest spread: those rows are listed instead (a raised arm, or a rest the tolerance refused);
- it counts STARTS, not rows: an `up` logs its startup check and that check's own connect within
  seconds, and the robot server's first connect later; rows of one bus less than START_GAP_S apart
  are one start, counted once at its largest deviation;
- it recommends max(0.05, the largest |q - park_q| seen + a margin), rounded up to 0.01 rad, and
  never above the current value (`--current`, else the TIGHTER of the two profiles it advises,
  rebot_rs and rebot_rs_mb: one arm, one bus): it never advises loosening either. 0.05
  rad is the robot server's own torque-release tolerance: an arm it released is within it, so a
  tighter value only adds refusals;
- it recommends nothing until it has `--min-samples` starts (never fewer than MIN_SAMPLES_FLOOR).

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

#: Starts needed before a recommendation (cold starts, startup checks).
MIN_SAMPLES = 20
#: `--min-samples` may raise MIN_SAMPLES, never go below this: a handful of starts says nothing
#: about a rig's rest spread.
MIN_SAMPLES_FLOOR = 10
#: Rows of one bus less than this far apart (seconds) are one start (see the module docstring).
START_GAP_S = 600.0
#: The refused rows shown (the latest).
SHOW_REFUSED = 10
#: The arm profiles whose rest_check_tol_rad this report advises: the rig's two reBot transports (one
#: arm, one bus). The current value is the tightest of them, so the advice never loosens one.
PROFILES = ("rebot_rs", "rebot_rs_mb")


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


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return None
    return float(v)


def _tolerance(v) -> float | None:
    """`v` as a tolerance a profile may hold, (0, ceiling] rad, else None."""
    x = _number(v)
    return x if x is not None and 0.0 < x <= arm_hold.REST_TOL_RAD else None


def _deviation(row) -> list[float] | None:
    """Per joint, the larger |q - park| of the two reads, when both reads are numbers that agree."""
    q, again, park = _numbers(row.get("q")), _numbers(row.get("again")), _numbers(row.get("park"))
    if q is None or again is None or park is None or not len(q) == len(again) == len(park):
        return None
    if max(abs(a - b) for a, b in zip(q, again)) > arm_hold.READ_AGREE_RAD:
        return None
    return [max(abs(a - p), abs(b - p)) for a, b, p in zip(q, again, park)]


def _kind(row) -> tuple[str, list[float] | None]:
    """'learn' (with its deviation), 'refused' (no marker, the check refused it) or 'ignored'."""
    if row.get("marker") is not False:
        return "ignored", None  # a held arm reads wherever it is held
    if row.get("verdict") != "rest":
        return "refused", None
    dev, tol = _deviation(row), _tolerance(row.get("tol_rad"))
    if dev is None or tol is None or max(dev) > tol:
        return "ignored", None  # a 'rest' the row itself does not bear out
    return "learn", dev


def _starts(rows) -> list[list[dict]]:
    """Rows grouped into starts: per bus, consecutive rows less than START_GAP_S apart. A row with
    no readable time is a start of its own."""
    starts, last = [], {}
    for row in rows:
        at, key = _number(row.get("at")), str(row.get("key"))
        prev = last.get(key)
        if at is not None and prev is not None and prev[0] is not None and 0.0 <= at - prev[0] < START_GAP_S:
            prev[1].append(row)
            last[key] = (at, prev[1])
            continue
        group = [row]
        starts.append(group)
        last[key] = (at, group)
    return starts


def _current_from(rows) -> float | None:
    """The tightest tolerance a logged check applied (the running value when no profile says it): never
    the latest row alone, which may come from the looser of the two profiles."""
    tols = [tol for tol in (_tolerance(row.get("tol_rad")) for row in rows) if tol is not None]
    return min(tols) if tols else None


def _refused_row(row) -> dict:
    dev = _numbers(row.get("dev")) if isinstance(row.get("dev"), list) else None
    return {"at": row.get("at_text"), "source": row.get("source"), "key": row.get("key"),
            "tol_rad": row.get("tol_rad"), "max_dev_rad": max(dev) if dev else row.get("max_dev"),
            "verdict": str(row.get("verdict"))[:160]}


def check_options(*, margin: float, min_samples: int, current: float | None) -> None:
    """Raise ValueError for options that would weaken the advice: a negative or non-finite margin
    (a recommendation below the measured spread), fewer than MIN_SAMPLES_FLOOR starts, or a current
    value no profile may hold."""
    if _number(margin) is None or margin < 0.0:
        raise ValueError(f"--margin {margin!r}: a margin is a finite number of rad >= 0")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < MIN_SAMPLES_FLOOR:
        raise ValueError(f"--min-samples {min_samples!r}: at least {MIN_SAMPLES_FLOOR} starts")
    if current is not None and _tolerance(current) is None:
        raise ValueError(f"--current {current!r}: a rest_check_tol_rad is in (0, {arm_hold.REST_TOL_RAD:g}] rad")


def analyse(rows, *, margin: float, min_samples: int, current: float | None = None) -> dict:
    """The report over logged rest rows (see the module docstring). `current`: the rig's
    rest_check_tol_rad now (None: the tightest tolerance a logged check applied, else the ceiling).
    Raises ValueError for options that would weaken the advice (`check_options`)."""
    check_options(margin=margin, min_samples=min_samples, current=current)
    rows = list(rows)
    if current is None:
        current = _current_from(rows) or arm_hold.REST_TOL_RAD
    samples, readings, refused, ignored = [], 0, [], 0
    for start in _starts(rows):
        devs = []
        for row in start:
            kind, dev = _kind(row)
            if kind == "learn":
                devs.append(dev)
            elif kind == "refused":
                refused.append(row)
            else:
                ignored += 1
        n = len(devs[0]) if devs else 0
        devs = [d for d in devs if len(d) == n]
        if devs:
            readings += len(devs)
            samples.append([max(d[j] for d in devs) for j in range(n)])
    out = {"samples": len(samples), "readings": readings, "refused": len(refused), "ignored": ignored,
           "refused_rows": [_refused_row(r) for r in refused[-SHOW_REFUSED:]], "margin_rad": margin,
           "min_samples": min_samples, "start_gap_s": START_GAP_S, "current_rad": current,
           "ceiling_rad": arm_hold.REST_TOL_RAD, "floor_rad": arm_hold.MARKER_CLEAR_TOL_RAD, "joints": [],
           "max_dev_rad": None, "worst_joint": None, "recommended_rad": None}
    if samples:
        n = len(samples[0])
        for j in range(n):
            col = sorted(s[j] for s in samples if len(s) == n)
            p95 = col[min(len(col) - 1, int(math.ceil(0.95 * len(col))) - 1)]
            out["joints"].append({"joint": j + 1, "n": len(col), "max_rad": round(col[-1], 6),
                                  "p95_rad": round(p95, 6)})
        worst = max(out["joints"], key=lambda r: r["max_rad"])
        out["max_dev_rad"], out["worst_joint"] = worst["max_rad"], worst["joint"]
    refused_note = (f"; {len(refused)} no-marker check(s) were refused (listed: a raised arm, or a rest this "
                    "tolerance refused; check the arm, never loosen the value for them)" if refused else "")
    if len(samples) < min_samples:
        out["advice"] = (f"not enough starts ({len(samples)} of {min_samples}): keep the current value until more "
                         f"cold starts are logged{refused_note}")
        return out
    want = math.ceil(round((out["max_dev_rad"] + margin) * 100.0, 6)) / 100.0
    rec = round(min(max(want, arm_hold.MARKER_CLEAR_TOL_RAD), arm_hold.REST_TOL_RAD), 2)
    if rec > current:
        out["advice"] = (f"keep rest_check_tol_rad: {current:g}: the folded rest spreads up to "
                         f"{out['max_dev_rad']:.3f} rad (joint {out['worst_joint']}), and this report never "
                         f"recommends above the current value{refused_note}")
    elif rec == current:
        out["recommended_rad"] = rec
        out["advice"] = f"keep rest_check_tol_rad: {current:g}: it fits the measured folded rest{refused_note}"
    else:
        out["recommended_rad"] = rec
        out["advice"] = (f"set rest_check_tol_rad: {rec:g} in the rig's arm profiles (configs/arms/rebot_rs.yaml, "
                         f"rebot_rs_mb.yaml){refused_note}")
    return out


def _profile_tolerance(rows=()) -> tuple[float | None, str]:
    """The current value: the tightest rest_check_tol_rad of PROFILES (#431 re-review Minor 4: the advice
    names both files, so a value read from one alone could loosen the other). A profile that cannot be
    read: the tightest of those that can and of every tolerance a logged check applied (`rows`)."""
    found, unknown = {}, []
    for name in PROFILES:
        try:
            from cascade.config import load_demo_config

            found[f"configs/arms/{name}.yaml"] = arm_hold.rest_tolerance(
                load_demo_config(camera="mock", arm=name, llm="mock").arm)
        except Exception as exc:  # noqa: BLE001 -- a report: say it and go on
            unknown.append(f"{name}.yaml unreadable ({exc})")
    values = dict(found)
    logged = _current_from(rows) if unknown else None
    if logged is not None:
        values["the tightest logged check"] = logged
    if not values:
        return None, f"unknown ({'; '.join(unknown)}): the tightest logged check's"
    where = min(values, key=values.get)
    parts = [f"{k} {v:g}" for k, v in values.items()] + unknown
    return values[where], f"{values[where]:g} rad ({where}; the tightest of: {', '.join(parts)})"


def _option(convert, name: str):
    def parse(text: str):
        try:
            value = convert(text)
            check_options(**{"margin": 0.0, "min_samples": MIN_SAMPLES, "current": None, name: value})
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None
        return value

    return parse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None, help="the rest-readings log (default: CASCADE_REST_LOG or "
                                                "~/.cascade/logs/rest_readings.jsonl)")
    ap.add_argument("--margin", type=_option(float, "margin"), default=arm_hold.READ_AGREE_RAD,
                    help="rad added to the largest measured deviation, >= 0 (default: the read-agreement band)")
    ap.add_argument("--min-samples", type=_option(int, "min_samples"), default=MIN_SAMPLES,
                    help=f"starts needed before a recommendation (default {MIN_SAMPLES}, at least "
                         f"{MIN_SAMPLES_FLOOR})")
    ap.add_argument("--current", type=_option(float, "current"), default=None,
                    help="the rig's rest_check_tol_rad now (default: the tighter of the rebot_rs and rebot_rs_mb "
                         "profiles'); never recommended above")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    path = Path(args.log).expanduser() if args.log else arm_hold.rest_log_path()
    rows = _rows(path)
    current, where = (args.current, f"{args.current:g} rad (--current)") if args.current is not None \
        else _profile_tolerance(rows)
    report = {"log": str(path), **analyse(rows, margin=args.margin, min_samples=args.min_samples, current=current)}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"rest readings: {path}")
    print(f"  learned from {report['samples']} starts ({report['readings']} folded, unpowered readings judged "
          f"at rest: no hold marker, the two reads agree); refused {report['refused']}; ignored {report['ignored']}")
    for r in report["joints"]:
        print(f"  joint {r['joint']}: max |q - park_q| {r['max_rad']:.4f} rad, p95 {r['p95_rad']:.4f} ({r['n']} starts)")
    for r in report["refused_rows"]:
        print(f"  refused {r['at']} {r['source']} (tolerance {r['tol_rad']}): {r['verdict']}")
    print(f"current: {where if current is not None else report['current_rad']}")
    if report["recommended_rad"] is not None:
        print(f"recommended: rest_check_tol_rad: {report['recommended_rad']:g}")
    print(report["advice"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
