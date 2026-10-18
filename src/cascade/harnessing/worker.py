# SPDX-License-Identifier: Apache-2.0
"""One rollout in its own process: ``python -m cascade.harnessing.worker <job.json>``.

`SubprocessExecutor` writes the job (executor settings, harness, task with its hidden check, label,
output directory); the rollout's result lands in ``<out_dir>/rollout.json`` as with an in-process
`CascadeExecutor`.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .executor import CascadeExecutor, TaskSpec
from .surface import AgentHarness


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m cascade.harnessing.worker <job.json>", file=sys.stderr)
        return 2
    job = json.loads(Path(argv[0]).read_text())
    executor = CascadeExecutor(**job["executor"])
    rollout = executor.run(AgentHarness.from_dict(job["harness"]), TaskSpec.from_dict(job["spec"]), job["label"],
                           Path(job["out_dir"]))
    return 0 if rollout.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
