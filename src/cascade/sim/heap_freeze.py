"""Opt-in startup-heap freeze for long-lived native owner processes.

A CPython full (generation-2) collection traverses every tracked container in
the oldest generation. Inside an Isaac Kit process the SDK, extension and scene
startup heap dominates that set, so each automatic generation-2 collection has
paused the owner thread for hundreds of milliseconds inside readiness and
control windows (docs/MICRODUCK.md, docs/PROJECT_STATUS_20261003.md).
Reducing per-attempt garbage changes how often those collections start; only
reducing the *retained* tracked population shortens them, and the startup heap
is the dominant retained population.

`StartupHeapFreeze.apply()` runs one explicit full collection and then moves
the surviving startup objects to the collector's permanent generation
(`gc.freeze`), so later automatic collections traverse only objects allocated
after admission. `release()` returns them at close and counts the cyclic
garbage that stayed frozen meanwhile. The collector stays enabled with
unchanged thresholds and callbacks; nothing here runs inside an attempt. The
receipts bound the collector's traversal set; they are not a control-deadline
guarantee, a speedup claim or physical admission.

Collector state is tracked separately from receipt construction and is
published *before* each collector call: `freezing` is set immediately before
`gc.freeze()` and `releasing` immediately before `gc.unfreeze()`, so a signal
landing right after either call still leaves a state from which `release()`
restores the collector (`gc.unfreeze()` is idempotent). A failure after
`unfreeze()` leaves the collector restored and the missing receipt explicit. CPython 3.12 keeps interpreter-immortal
objects in the same permanent generation and moves more of them there during
any full collection, so a bare interpreter already reports a few hundred frozen
objects and `unfreeze()` is interpreter-wide; frozen objects freed by reference
counting leave the permanent generation by themselves. Receipts therefore record
the foreign frozen count before `apply()`, the policy's own delta and the change
since `apply()`, and never claim a zero count after release.
"""
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

POLICY = 'freeze-startup-heap-v1'
SCOPE = ('Objects alive at apply() moved to the permanent generation; later automatic '
         'collections traverse only objects allocated afterwards. Collector enabled, '
         'thresholds and callbacks unchanged. Cyclic garbage alive at apply() stays '
         'retained until release(). Not a control-deadline, speedup or physical claim.')


def implementation_sha256():
    """Digest of this module's source, so receipts bind the exact policy implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _snapshot(collector):
    blocks = getattr(sys, 'getallocatedblocks', None)
    return dict(count=list(collector.get_count()), stats=list(collector.get_stats()),
                frozen=collector.get_freeze_count(),
                allocated_blocks=None if blocks is None else blocks())


def _settings(collector):
    return dict(enabled=bool(collector.isenabled()), thresholds=list(collector.get_threshold()),
                callbacks=len(collector.callbacks))


def _peak_rss_kib():
    """Process peak RSS in KiB (ru_maxrss is bytes on macOS); None where unavailable."""
    try:
        import resource
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw // 1024 if sys.platform == 'darwin' else raw
    except Exception:
        return None


def _process():
    return dict(pid=os.getpid(), interpreter=sys.version.split()[0], executable=sys.executable,
                peak_rss_kib=_peak_rss_kib())


class StartupHeapFreeze:
    """One `collect()` then `freeze()` before admission; `release()` after close.

    Call `apply()` after every SDK, scene and identity startup step and before
    any readiness, attempt or control loop starts; call `release()` once that
    loop has closed. `state` is `idle`, `freezing`, `frozen`, `releasing` or
    `released`; `frozen` is true whenever the collector may still hold this
    policy's freeze. `receipt` and `released` hold the complete receipts,
    `apply_error` / `release_error` the exception type that interrupted a call
    after its collector operation may already have happened. A disabled
    collector is refused: this policy bounds the automatic collector's traversal
    set and nothing else.
    """

    _MAYBE_FROZEN = ('freezing', 'frozen', 'releasing')

    def __init__(self, *, clock=time.monotonic_ns, collector=gc):
        self.clock, self.collector = clock, collector
        self.state = 'idle'
        self.receipt = self.released = None
        self.apply_error = self.release_error = None

    @property
    def frozen(self):
        """True while the collector may hold this policy's freeze; release() is then required."""
        return self.state in self._MAYBE_FROZEN

    def apply(self):
        if self.state != 'idle':
            raise RuntimeError('startup heap freeze already applied for this owner')
        settings = _settings(self.collector)
        if not settings['enabled']:
            raise RuntimeError('startup heap freeze requires the automatic collector enabled')
        try:
            before = _snapshot(self.collector)
            started = self.clock()
            unreachable = self.collector.collect()
            collected = self.clock()
            frozen_after_collect = self.collector.get_freeze_count()
            self.state = 'freezing'  # published first: an interruption after freeze() still owns the heap
            self.collector.freeze()
            self.state = 'frozen'
            frozen = self.clock()
            after = _snapshot(self.collector)
            if _settings(self.collector) != settings:
                raise RuntimeError('collector settings changed during startup heap freeze')
            self.receipt = dict(policy=POLICY, implementation_sha256=implementation_sha256(),
                thread_id=threading.get_ident(), process=_process(),
                applied_monotonic_ns=started, settings=settings,
                collect=dict(duration_ns=collected-started, unreachable=unreachable,
                             frozen_after_collect=frozen_after_collect),
                freeze=dict(duration_ns=frozen-collected, frozen=after['frozen']-frozen_after_collect,
                            foreign_frozen_before=before['frozen']),
                before=before, after=after, scope=SCOPE)
        except BaseException as exc:
            self.apply_error = type(exc).__name__
            raise
        return self.receipt

    def release(self, *, collect=True, measure_frozen=True):
        """Unfreeze at close; optionally first time one full collection while still frozen.

        That measured collection runs after the owner loop has closed and outside
        any control window: it is the native cost of a post-freeze full collection
        over everything retained by the episode, which an episode too short to
        trigger an automatic generation-2 collection would otherwise never measure.
        """
        if self.state == 'idle':
            raise RuntimeError('startup heap freeze was never applied')
        if self.state == 'released':
            raise RuntimeError('startup heap freeze already released')
        try:
            started = frozen_collect = None
            try:
                before = _snapshot(self.collector)
                if measure_frozen and self.state == 'frozen':
                    measure_started = self.clock()
                    unreachable_frozen = self.collector.collect()
                    frozen_collect = dict(duration_ns=self.clock()-measure_started, unreachable=unreachable_frozen,
                                          tracked_after=len(self.collector.get_objects()))
                started = self.clock()
            finally:
                # Restore the collector even if the pre-release snapshot or measurement
                # failed. 'releasing' is published first so an interruption right after
                # unfreeze() can only lead to a harmless repeated unfreeze on retry.
                self.state = 'releasing'
                self.collector.unfreeze()
                self.state = 'released'
            unfrozen = self.clock()
            unreachable = self.collector.collect() if collect else None
            ended = self.clock()
            applied = self.receipt
            self.released = dict(policy=POLICY, implementation_sha256=implementation_sha256(),
                thread_id=threading.get_ident(), process=_process(),
                released_monotonic_ns=started, unfrozen=before['frozen'], frozen_collect=frozen_collect,
                apply_receipt_complete=applied is not None, apply_error=self.apply_error,
                policy_frozen=None if applied is None else applied['freeze']['frozen'],
                frozen_changed_since_apply=None if applied is None else before['frozen']-applied['after']['frozen'],
                unfreeze=dict(duration_ns=unfrozen-started),
                collect=None if not collect else dict(duration_ns=ended-unfrozen, unreachable=unreachable),
                before=before, after=_snapshot(self.collector), settings=_settings(self.collector))
        except BaseException as exc:
            self.release_error = type(exc).__name__
            raise
        return self.released
