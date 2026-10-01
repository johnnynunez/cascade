"""Optional, non-authoritative Python zones for the Isaac bridge.

This module imports only the standard library. The factory must be called
*after* SimulationApp exists; the default path never imports Carbonite.
Profiler failures invalidate diagnostics without changing the enclosed operation.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
import time


class _NoZone:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


_NO_ZONE = _NoZone()


class _Zone:
    def __init__(self, owner, name):
        self.owner, self.name, self.begun = owner, name, False

    def __enter__(self):
        try:
            self.owner.backend.begin_with_location(
                1, self.name, "isaac_bridge", self.owner.source_path, 0)
            self.begun = True
        except Exception as exc:
            self.owner.fail("begin", exc)
        return self

    def __exit__(self, *_exc):
        if self.begun:
            try:
                self.owner.backend.end(1)
            except Exception as exc:
                self.owner.fail("end", exc)
        return False


class PythonSpans:
    """No tensor/SDK reads, global tracing hooks, or per-loop event storage."""

    def __init__(self, *, enabled=False, backend=None, source_path="",
                 clock_ns=time.monotonic_ns, emit=print):
        self.enabled = enabled
        self.backend = backend
        self.source_path = source_path
        self.clock_ns, self.emit = clock_ns, emit
        self.error_count = 0
        self.first_error = None
        self.anchor = None
        self.source_sha256 = None
        self.helper_sha256 = None
        self._anchor_attempted = False

    def fail(self, stage, exc):
        self.error_count += 1
        if self.first_error is None:
            self.first_error = {"stage": stage, "type": type(exc).__name__}
        # An exception can leave the native stack's state unknown. Do not add
        # more zones; already-entered scopes still attempt their matching end.

    def zone(self, name):
        if not self.enabled or self.error_count:
            return _NO_ZONE
        return _Zone(self, name)

    def anchor_once(self):
        """Emit one same-thread clock bracket; its zone must exist in the trace.

        Both native zone boundaries lie within [before, after].
        A trace without this unique zone cannot use this mapping.
        """
        if not self.enabled or self.error_count or self._anchor_attempted:
            return
        self._anchor_attempted = True
        try:
            before = self.clock_ns()
            name = "bridge.clock_anchor"
            with self.zone(name):
                pass
            after = self.clock_ns()
            if after < before:
                raise ValueError("monotonic clock regressed")
            self.anchor = {"zone": name, "before_monotonic_ns": before,
                           "after_monotonic_ns": after,
                           "native_thread_id": threading.get_native_id(),
                           "pid": os.getpid(), "source_path": self.source_path,
                           "source_sha256": self.source_sha256,
                           "helper_sha256": self.helper_sha256}
            self.emit("[bridge-python-spans] " + json.dumps(
                {"event": "clock_anchor", **self.anchor,
                 "diagnostic_valid": not bool(self.error_count)}), flush=True)
        except Exception as exc:
            self.fail("anchor", exc)

    def report(self):
        if not self.enabled:
            return
        try:
            self.emit("[bridge-python-spans] " + json.dumps(
                {"event": "shutdown", "enabled": True,
                 "diagnostic_valid": not bool(self.error_count),
                 "error_count": self.error_count, "first_error": self.first_error,
                 "anchor": self.anchor}), flush=True)
        except Exception as exc:
            self.fail("report", exc)


def from_environment(source_path):
    enabled = os.environ.get("CASCADE_ISAAC_PYTHON_SPANS", "0") == "1"
    spans = PythonSpans(enabled=enabled, source_path=source_path)
    if enabled:
        try:
            backend = importlib.import_module("carb.profiler")
            if (not callable(backend.begin_with_location) or not callable(backend.end)
                    or not backend.is_profiler_active()):
                raise RuntimeError("active Carbonite profiler is required")
            spans.backend = backend
            spans.source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
            spans.helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            with spans.zone("bridge.profiler_preflight"):
                pass
        except Exception as exc:
            spans.fail("initialization", exc)
    return spans
