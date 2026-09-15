"""Compute-device resolution: one place that decides where models run.

CASCADE is deployed on whatever box is available -- an NVIDIA DGX Spark or DGX
Station, a Jetson (Orin/Thor), an RTX workstation, a generic CUDA server, an
AMD/ROCm box, an Apple Silicon laptop, or a CPU-only booth machine. The
perception stack must not care, so no module hardcodes an accelerator: configs
say `device: auto` and every consumer routes through `resolve_device()`.

Two rules make this safe rather than merely convenient:

  1. `auto` PICKS the best accelerator actually present, in order
     CUDA/ROCm -> Apple MPS -> CPU. Nothing about the choice is compiled in;
     it is a runtime probe of the host the process happens to be on.

  2. An EXPLICIT device that is not present DEGRADES to the best available one
     with a warning, instead of raising. Config files travel between machines
     (a rig config opened on a laptop, a booth checkout cloned from the rig),
     and `device: cuda:0` on a machine with no CUDA should cost a warning line
     and a slow run, not a crashed demo. This is the same booth rule the grasp
     and occupancy backends follow: degrade the capability, never the session.

Nothing here imports torch at module load. Probing is lazy and cached, so the
mock stack (and CI, which has no torch at all) pays nothing.
"""

from __future__ import annotations

import logging
import os
import platform
from functools import lru_cache

logger = logging.getLogger(__name__)

#: Config/env value that means "probe the host and pick the best accelerator".
AUTO = "auto"

#: Env override, applied before any config value. Set CASCADE_DEVICE=cpu to
#: force every model onto the CPU without editing a single YAML file -- the
#: fastest way to rule out a GPU/driver problem on an unfamiliar machine.
DEVICE_ENV = "CASCADE_DEVICE"


@lru_cache(maxsize=1)
def _torch():
    """The torch module, or None. Cached: the import is seconds, not ms."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - a broken CUDA install raises, not just ImportError
        return None
    return torch


@lru_cache(maxsize=1)
def best_device() -> str:
    """Best accelerator actually present on this host, as a torch device string.

    Order is CUDA/ROCm, then Apple MPS, then CPU. ROCm reports itself through
    the same `torch.cuda` API surface, so a ROCm box correctly answers
    "cuda:0" -- that is what PyTorch wants to be told there, not "rocm".
    """
    torch = _torch()
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
    except Exception:  # noqa: BLE001 - driver/library mismatch surfaces here
        logger.debug("CUDA probe failed; falling through", exc_info=True)
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        logger.debug("MPS probe failed; falling through", exc_info=True)
    return "cpu"


def _family(device: str) -> str:
    return str(device).split(":", 1)[0].strip().lower()


def _is_available(device: str) -> bool:
    """Can this host actually run on `device`?"""
    fam = _family(device)
    if fam == "cpu":
        return True
    torch = _torch()
    if torch is None:
        return False
    try:
        if fam == "cuda":
            if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
                return False
            # An explicit index must exist: "cuda:1" on a single-GPU box is a
            # config written for a bigger machine.
            _, _, idx = str(device).partition(":")
            return not idx.strip() or 0 <= int(idx) < torch.cuda.device_count()
        if fam == "mps":
            mps = getattr(torch.backends, "mps", None)
            return mps is not None and mps.is_available()
    except Exception:  # noqa: BLE001
        return False
    # Anything else (xpu, hpu, a future backend) is taken at face value: we
    # cannot verify it, and refusing it would be worse than trying it.
    return True


def resolve_device(spec: str | None = AUTO, *, what: str = "model") -> str:
    """Config/env device spec -> a device string this host can actually use.

    `what` only names the caller in the warning ("detector", "segmenter"), so
    a degraded run says which model moved rather than just that something did.
    """
    env = os.environ.get(DEVICE_ENV, "").strip()
    requested = env or (spec if spec is not None else AUTO)
    requested = str(requested).strip() or AUTO

    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        chosen = best_device() if requested.lower() == AUTO else requested
        torch = _torch()
        version = getattr(torch, "version", None)
        if (_family(chosen) != "cuda" or not _is_available(chosen)
                or torch is None or not getattr(version, "cuda", None)
                or getattr(version, "hip", None)):
            raise RuntimeError(f"{what} requires NVIDIA CUDA; requested {requested!r}, available {chosen!r}; CPU fallback is forbidden")
        return chosen

    if requested.lower() == AUTO:
        chosen = best_device()
        logger.info("%s device: auto -> %s", what, chosen)
        return chosen

    if _is_available(requested):
        return requested

    fallback = best_device()
    logger.warning(
        "%s device %r is not available on this host (%s); using %r instead. "
        "Set %s or the profile's `device:` to silence this.",
        what, requested, host_summary(), fallback, DEVICE_ENV,
    )
    return fallback


def host_summary() -> str:
    """One-line host description for logs and diagnostics."""
    bits = [f"{platform.system()}/{platform.machine()}", f"python {platform.python_version()}"]
    torch = _torch()
    if torch is None:
        bits.append("torch absent")
        return ", ".join(bits)
    bits.append(f"torch {torch.__version__}")
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
            hip = getattr(torch.version, "hip", None)
            api = f"ROCm {hip}" if hip else f"CUDA {torch.version.cuda}"
            bits.append(f"{torch.cuda.device_count()}x {'/'.join(sorted(names))} ({api})")
        elif _family(best_device()) == "mps":
            bits.append("Apple MPS")
        else:
            bits.append("no accelerator")
    except Exception:  # noqa: BLE001
        bits.append("accelerator probe failed")
    return ", ".join(bits)
