#!/usr/bin/env python3
"""Advisory, time-bounded judge pass over the launcher's proof turn.

    scripts/judge_proof.py --proof runs/.launch/<profile>/proof.json --judge fake|vlm|grm
                           [--timeout-s 180]

Runs scripts/judge_run.py over the proof turn's pick_and_place rows, writes
the judge-vs-physics confusion matrix into <evidence_dir>/run-summary.json
and prints ONE line for the launcher banner. Never writes proof.json; always
exits 0 (a judge failure reads "unavailable"). See cascade.eval.proof_judge.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cascade.eval.proof_judge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(repo=REPO))
