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


def _sync_libero_config() -> None:
    """Point LIBERO's asset config at the checkout we are actually running.

    MEASURED BUG this prevents: `libero.libero` resolves bddl_files,
    init_files and assets from `~/.libero/config.yaml`, which is written once
    at install time and names ONE checkout. Setting `WRC_BENCH_LIBERO` put
    LIBERO-PRO on sys.path but left that config pointing at plain LIBERO, so
    a LIBERO-Pro run loaded Pro's task list and then tried to read Pro's
    init_files from the standard checkout:

        FileNotFoundError: .../bench/LIBERO/libero/libero/init_files/
                           libero_spatial_lan/....pruned_init

    It also makes the two benchmarks mutually exclusive: whichever one the
    global config names is the only one that can run, and running both in
    parallel silently mixes assets.

    LIBERO honours `LIBERO_CONFIG_PATH`, so give each checkout its own config
    directory under the checkout itself. Nothing global is modified, and two
    suites can run side by side.
    """
    cfg_dir = LIBERO_DIR / ".libero_config"
    root = LIBERO_DIR / "libero" / "libero"
    if not root.is_dir():
        return
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / "config.yaml"
    want = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(LIBERO_DIR / "datasets"),
        "assets": str(root / "assets"),
    }
    body = "".join(f"{k}: {v}\n" for k, v in sorted(want.items()))
    if not cfg.exists() or cfg.read_text() != body:
        cfg.write_text(body)
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)


def add_paths() -> None:
    """Put LIBERO, wrc_demo and this package on sys.path."""
    _sync_libero_config()
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
