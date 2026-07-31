"""Shared paths and environment setup for the benchmark suite.

Every path here is overridable by an environment variable so the suite runs on
a machine that is not the one it was written on. Defaults match the layout in
docs/BENCHMARKS.md.

    WRC_BENCH_LIBERO   checkout of Lifelong-Robot-Learning/LIBERO
    WRC_BENCH_MODELS   directory holding openvla-7b-libero-* checkpoints
    WRC_BENCH_RESULTS  where result JSON is written (default: ./results)
    WRC_BENCH_VENV     python interpreter that has LIBERO installed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: benchmark/ itself
ROOT = Path(__file__).resolve().parent

#: repo root (benchmark/ lives directly under it)
REPO = ROOT.parent

#: wrc_demo source tree, so `import wrc_demo` works without installation
SRC = REPO / "src"


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


#: LIBERO checkout. Clone with:
#:   git clone https://github.com/Lifelong-Robot-Learning/LIBERO
LIBERO_DIR = _env_path("WRC_BENCH_LIBERO", Path.home() / "bench" / "LIBERO")

#: OpenVLA checkpoints, one per suite (openvla-7b-libero-spatial, -object, ...)
MODELS_DIR = _env_path("WRC_BENCH_MODELS", Path.home() / "models")

#: result JSON
RESULTS_DIR = _env_path("WRC_BENCH_RESULTS", ROOT / "results")

#: interpreter with LIBERO + robosuite installed (numpy<2, so it is NOT the
#: wrc_demo venv -- see docs/BENCHMARKS.md)
LIBERO_PYTHON = os.environ.get(
    "WRC_BENCH_VENV", str(Path.home() / ".venvs" / "libero" / "bin" / "python"))


def add_paths() -> None:
    """Put LIBERO, wrc_demo and this package on sys.path."""
    for p in (str(LIBERO_DIR), str(SRC), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def require_libero() -> None:
    """Fail loudly and usefully if LIBERO is not where we expect."""
    if not (LIBERO_DIR / "libero").is_dir():
        raise SystemExit(
            f"LIBERO not found at {LIBERO_DIR}.\n"
            "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO\n"
            "  export WRC_BENCH_LIBERO=/path/to/LIBERO")


def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
