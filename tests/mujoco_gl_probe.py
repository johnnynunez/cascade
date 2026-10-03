"""Contain optional native GL capability checks outside the pytest process."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlProbe:
    available: bool
    reason: str
    returncode: int | None = None


_RENDERER_PROBE = """
import sys
# A failed native renderer can abort rather than raise a Python exception.
# Disable core dumps in this disposable child only, where supported.
try:
    import resource
except ImportError:
    pass
else:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import mujoco
model = (mujoco.MjModel.from_xml_string(sys.argv[4]) if sys.argv[4]
         else mujoco.MjModel.from_xml_path(sys.argv[1]))
renderer = mujoco.Renderer(model, height=int(sys.argv[2]), width=int(sys.argv[3]))
renderer.close()
"""


def probe_offscreen_gl(
    mjcf: Path, *, timeout_s: float = 15.0,
    model_xml: str | None = None, height: int = 16, width: int = 16,
) -> GlProbe:
    """Use this interpreter and its selected backend; never switch GL providers.

    Only a clean child exit admits rendering tests. ``subprocess.run`` kills
    and reaps this child on timeout. Native aborts stay in the child, and its
    potentially noisy diagnostics cannot fill a pipe or pytest's memory.
    This checks optional rendering availability, not physics correctness.
    """
    if not mjcf.is_file():
        return GlProbe(False, "SO-101 MJCF assets are unavailable")
    try:
        result = subprocess.run(
            [sys.executable, "-c", _RENDERER_PROBE, str(mjcf.resolve()),
             str(height), str(width), model_xml or ""],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GlProbe(False, f"renderer probe exceeded {timeout_s:g}s")
    except OSError as exc:
        return GlProbe(False, f"renderer probe could not start ({type(exc).__name__})")
    if result.returncode != 0:
        return GlProbe(False, f"renderer probe exited {result.returncode}", result.returncode)
    return GlProbe(True, "renderer created and closed", 0)
