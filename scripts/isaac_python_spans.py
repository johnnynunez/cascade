"""Optional, non-authoritative Python zones and bounded CPU timing totals.

This module imports only the standard library. The factory must be called
*after* SimulationApp exists; the default path never imports Carbonite.
Profiler failures invalidate diagnostics without changing the enclosed operation.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
import time


_CLOCK_SAMPLE_INTERVAL_NS = 1_000_000_000
_MAX_CLOCK_SAMPLES = 2048
_TIMING_REPORT_INTERVAL_NS = 10_000_000_000
_MAX_TIMING_ZONES = 32


class _NoZone:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


_NO_ZONE = _NoZone()


class _Zone:
    def __init__(self, owner, name):
        self.owner, self.name, self.begun = owner, name, False
        self.started = None

    def __enter__(self):
        if self.owner.timings and not self.name.startswith('bridge.clock_') and self.name != 'bridge.profiler_preflight':
            try:
                self.started = self.owner.timing_clock()
            except Exception as exc:
                self.owner.fail('timing_begin', exc)
        if not self.owner.enabled:
            return self
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
        if self.started is not None:
            try:
                self.owner.add_timing(self.name, self.started, bool(_exc[0]))
            except Exception as exc:
                self.owner.fail('timing_end', exc)
        return False


class _GCAccounting:
    """Bounded passive collector accounting for the timing summaries.

    One callback aggregates completed collections per generation (count, total
    and maximum interval, whether the longest ran on the registering thread),
    pairing start/stop per thread and generation without storing events. It
    observes collector settings and never enables, disables, retunes, forces or
    freezes the collector; closing removes only its own callback. Intervals
    include scheduling and other callbacks and are not isolated CPU time.
    """

    def __init__(self, gc_module, clock_ns, thread_id):
        self.gc, self.clock_ns, self.thread_id = gc_module, clock_ns, thread_id
        self.pending, self.generations = {}, {}
        self.events = self.unmatched = self.errors = 0
        self.initial = self._settings()
        self.callback = self._record  # retain this exact bound method for removal
        gc_module.callbacks.append(self.callback)

    def _settings(self):
        return {'enabled': bool(self.gc.isenabled()), 'thresholds': list(self.gc.get_threshold())}

    def _record(self, phase, info):
        try:
            self.events += 1
            key = (threading.get_ident(), int(info['generation']))
            now = self.clock_ns()
            if phase == 'start':
                self.pending[key] = now
                return
            started = self.pending.pop(key, None)
            if started is None:
                self.unmatched += 1
                return
            duration = now - started
            row = self.generations.setdefault(key[1], {'count': 0, 'total_ns': 0, 'max_ns': 0,
                                                       'last_ns': 0, 'max_on_registering_thread': None})
            row['count'] += 1
            row['total_ns'] += duration
            row['last_ns'] = duration
            if duration >= row['max_ns']:
                row['max_ns'] = duration
                row['max_on_registering_thread'] = key[0] == self.thread_id
        except Exception:
            self.errors += 1  # never raise inside a collector callback

    def summary(self):
        frozen = getattr(self.gc, 'get_freeze_count', None)
        return {'events': self.events, 'unmatched': self.unmatched, 'errors': self.errors,
                'pending': len(self.pending), 'settings_initial': self.initial, 'settings_now': self._settings(),
                'frozen_objects': None if frozen is None else int(frozen()),
                'generations': {str(k): dict(v) for k, v in sorted(self.generations.items())},
                'scope': 'passive callback intervals; include scheduling and other callbacks; '
                         'not isolated CPU time; collector settings observed only'}

    def close(self):
        for index in range(len(self.gc.callbacks) - 1, -1, -1):
            if self.gc.callbacks[index] is self.callback:
                del self.gc.callbacks[index]


class PythonSpans:
    """No tensor/SDK reads, global tracing hooks, or per-loop event storage."""

    def __init__(self, *, enabled=False, backend=None, source_path="",
                 clock_ns=time.monotonic_ns, emit=print, timings=False,
                 cpu_clock_ns=time.thread_time_ns, gc_module=None):
        self.enabled = enabled
        self.timings, self.cpu_clock_ns = timings, cpu_clock_ns
        self.gc = (_GCAccounting(gc_module, clock_ns, threading.get_ident())
                   if timings and gc_module is not None else None)
        self.timing_totals = {}
        self.timing_start = self.timing_thread_id = self.last_timing_report = None
        self.backend = backend
        self.source_path = source_path
        self.clock_ns, self.emit = clock_ns, emit
        self.error_count = 0
        self.first_error = None
        self.anchor = None
        self.source_sha256 = None
        self.helper_sha256 = None
        self._anchor_attempted = False
        self.clock_sample_count = 0
        self.last_clock_sample = None
        self._last_clock_ns = None

    def fail(self, stage, exc):
        self.error_count += 1
        if self.first_error is None:
            self.first_error = {"stage": stage, "type": type(exc).__name__}
            try:
                self.emit("[bridge-python-spans] " + json.dumps(
                    {"event": "diagnostic_error", "diagnostic_valid": False,
                     "error_count": self.error_count, "first_error": self.first_error,
                     "pid": os.getpid(), "native_thread_id": threading.get_native_id(),
                     "source_path": self.source_path, "source_sha256": self.source_sha256,
                     "helper_sha256": self.helper_sha256}), flush=True)
            except Exception:
                # A broken output sink must not recurse or replace an SDK error.
                self.error_count += 1
        # An exception can leave the native stack's state unknown. Do not add
        # more zones; already-entered scopes still attempt their matching end.

    def zone(self, name):
        if not (self.enabled or self.timings) or self.error_count:
            return _NO_ZONE
        return _Zone(self, name)

    def timing_clock(self):
        thread_id = threading.get_native_id()
        wall, cpu = self.clock_ns(), self.cpu_clock_ns()
        if type(wall) is not int or type(cpu) is not int or min(wall, cpu) < 0:
            raise ValueError('invalid timing clock')
        if self.timing_thread_id is None:
            self.timing_thread_id, self.timing_start = thread_id, wall
        if self.timing_thread_id != thread_id:
            raise ValueError('timing zone changed native thread')
        return wall, cpu

    def add_timing(self, name, started, failed):
        if self.error_count:
            return
        finished = self.timing_clock()
        wall, cpu = (end - start for start, end in zip(started, finished))
        if min(wall, cpu) < 0:
            raise ValueError('timing clock regressed')
        if name not in self.timing_totals:
            if len(self.timing_totals) >= _MAX_TIMING_ZONES or len(name) > 128:
                raise ValueError('timing zone budget exceeded')
            self.timing_totals[name] = {'count': 0, 'exceptions': 0}
        record = self.timing_totals[name]
        record['count'] += 1
        record['exceptions'] += int(failed)
        for kind, value in (('wall', wall), ('thread_cpu', cpu)):
            total, low, high = f'{kind}_total_ns', f'{kind}_min_ns', f'{kind}_max_ns'
            record[total] = record.get(total, 0) + value
            record[low] = min(record.get(low, value), value)
            record[high] = max(record.get(high, value), value)

    def report_timings(self, *, final=False):
        if not self.timings or (self.error_count and not final):
            return
        try:
            now, _ = self.timing_clock()
            previous = self.timing_start if self.last_timing_report is None else self.last_timing_report
            if now < previous:
                raise ValueError('timing report clock regressed')
            if not final and now - previous < _TIMING_REPORT_INTERVAL_NS:
                return
            self.last_timing_report = now
            self.emit('[bridge-python-spans] ' + json.dumps({
                'event': 'timing_summary', 'final': final,
                'diagnostic_valid': not bool(self.error_count), 'error_count': self.error_count,
                'pid': os.getpid(), 'native_thread_id': self.timing_thread_id,
                'source_path': self.source_path, 'source_sha256': self.source_sha256,
                'helper_sha256': self.helper_sha256, 'started_monotonic_ns': self.timing_start,
                'snapshot_monotonic_ns': now, 'zones': self.timing_totals,
                'gc': None if self.gc is None else self.gc.summary(),
                'scope': 'completed inclusive zones; nested totals overlap; thread CPU excludes other threads and GPU',
            }), flush=True)
        except Exception as exc:
            self.fail('timing_report', exc)

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
            self._last_clock_ns = after
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

    def sample_clock_if_due(self):
        """Bracket a bounded, uniquely named CPU ordering marker at most 1 Hz.

        No catch-up, sleep, SDK read, or growing event history. These markers
        qualify CPU event order; they do not calibrate Tracy or GPU durations.
        """
        self.report_timings()
        if not self.enabled or self.error_count:
            return
        try:
            before = self.clock_ns()
            if (type(before) is not int or self._last_clock_ns is None
                    or before < self._last_clock_ns):
                raise ValueError("missing anchor or regressing monotonic clock")
            self._last_clock_ns = before
            if (self.last_clock_sample is not None
                    and before - self.last_clock_sample["after_monotonic_ns"]
                    < _CLOCK_SAMPLE_INTERVAL_NS):
                return
            if self.clock_sample_count >= _MAX_CLOCK_SAMPLES:
                raise RuntimeError("clock sample limit reached")
            if threading.get_native_id() != self.anchor["native_thread_id"]:
                raise ValueError("clock sample changed native thread")
            seq = self.clock_sample_count + 1
            name = f"bridge.clock_sample.{seq:06d}"
            with self.zone(name):
                pass
            after = self.clock_ns()
            if type(after) is not int or after < before:
                raise ValueError("monotonic clock regressed")
            self._last_clock_ns = after
            self.clock_sample_count = seq
            self.last_clock_sample = {
                "zone": name, "seq": seq, "before_monotonic_ns": before,
                "after_monotonic_ns": after,
                "native_thread_id": threading.get_native_id(), "pid": os.getpid(),
                "source_path": self.source_path, "source_sha256": self.source_sha256,
                "helper_sha256": self.helper_sha256}
            self.emit("[bridge-python-spans] " + json.dumps(
                {"event": "clock_sample", **self.last_clock_sample,
                 "diagnostic_valid": not bool(self.error_count),
                 "error_count": self.error_count}), flush=True)
        except Exception as exc:
            self.fail("clock_sample", exc)

    def report(self):
        self.report_timings(final=True)
        if self.gc is not None:
            try:
                self.gc.close()  # after the final summary; foreign callbacks are untouched
            except Exception as exc:
                self.fail('gc_close', exc)
        if not self.enabled:
            return
        try:
            self.emit("[bridge-python-spans] " + json.dumps(
                {"event": "shutdown", "enabled": True,
                 "diagnostic_valid": not bool(self.error_count),
                 "error_count": self.error_count, "first_error": self.first_error,
                 "anchor": self.anchor, "clock_sample_count": self.clock_sample_count,
                 "last_clock_sample": self.last_clock_sample}), flush=True)
        except Exception as exc:
            self.fail("report", exc)


def from_environment(source_path):
    enabled = os.environ.get("CASCADE_ISAAC_PYTHON_SPANS", "0") == "1"
    timings = os.environ.get("CASCADE_ISAAC_PYTHON_TIMINGS", "0") == "1"
    spans = PythonSpans(enabled=enabled, timings=timings, source_path=source_path,
                        gc_module=gc if timings else None)
    if enabled or timings:
        try:
            spans.source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
            spans.helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if enabled:
                backend = importlib.import_module("carb.profiler")
                if (not callable(backend.begin_with_location) or not callable(backend.end)
                        or not backend.is_profiler_active()):
                    raise RuntimeError("active Carbonite profiler is required")
                spans.backend = backend
                with spans.zone("bridge.profiler_preflight"):
                    pass
        except Exception as exc:
            spans.fail("initialization", exc)
    return spans
