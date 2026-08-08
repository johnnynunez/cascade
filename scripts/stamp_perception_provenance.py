"""Stamp perception provenance onto LIBERO result files written before the
`--perception` flag existed.

Every `gap_*.json` in benchmark/results was produced by run_wrc.py when
`seed_beliefs()` was unconditional, i.e. object poses came from
`sim.data.body_xpos` rather than from a camera. Without a marker those files
can be quoted later as if they were comparable to ASPIRE / Pigey /
Harness-VLA, which perceive their own scenes.

The no-op and OpenVLA files are NOT touched: those run a policy directly
against the env and never consult the belief store.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "benchmark" / "results"

NOTE = (
    "oracle: object poses read from sim.data.body_xpos; measures orchestration "
    "given perfect perception; NOT comparable to ASPIRE/Pigey/Harness-VLA/VIA, "
    "and ASPIRE forbids this API (arXiv:2607.00272)"
)

# Files produced by run_wrc.py's orchestration path.
ORACLE_FILES = ["gap_libero_10.json", "gap_libero_goal.json",
                "gap_libero_object.json", "gap_spatial.json",
                "ablation_v2.json", "wrc_ablation.json"]


def main() -> int:
    for name in ORACLE_FILES:
        p = RESULTS / name
        if not p.exists():
            print(f"  skip (absent): {name}")
            continue
        d = json.loads(p.read_text())
        if d.get("perception"):
            print(f"  already stamped: {name}")
            continue
        # Preserve the original shape; only prepend provenance.
        stamped = {"perception": "oracle", "perception_note": NOTE}
        stamped.update(d)
        p.write_text(json.dumps(stamped, indent=2))
        print(f"  stamped: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
