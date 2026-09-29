#!/usr/bin/env python3
"""Verify the original kitchen sources already included in the checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
from scene_identity import MANIFEST, kitchen_manifest, kitchen_problems, source_manifest


def prepare_kitchen(repo: Path) -> None:
    """Fail closed on missing/changed sources. Kitchen installation is offline."""
    problems = kitchen_problems(repo.resolve())
    if problems:
        raise RuntimeError("Kitchen source verification failed: " + "; ".join(problems)
                           + ". Restore the matching source checkout before launching.")
    print(f"[cascade-install] Original kitchen: {len(kitchen_manifest(repo))} source files verified; no downloads.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="Verify sources without writing files")
    parser.add_argument("--write-manifest", action="store_true",
                        help="Maintainer operation after reviewing intentional source changes")
    args = parser.parse_args(argv)
    if args.check and args.write_manifest:
        parser.error("--check and --write-manifest are mutually exclusive")
    try:
        if args.write_manifest:
            (args.repo / MANIFEST).write_text(json.dumps(source_manifest(args.repo), indent=2) + "\n")
        prepare_kitchen(args.repo)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
