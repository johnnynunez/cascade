#!/usr/bin/env python3
"""CPU mechanism check for cascade.sim.heap_freeze: full-collection cost before and after
a startup-heap freeze on a synthetic heap. Writes one JSON receipt. This measures the
CPython collector on this host only; it is not a native Kit, physics or control result."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cascade.sim.heap_freeze import StartupHeapFreeze  # noqa: E402


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns()-started)/1e6)
    return dict(repeats=repeats, median_ms=round(statistics.median(samples), 3),
                max_ms=round(max(samples), 3), min_ms=round(min(samples), 3))


def records(count):
    """Dict/list/tuple rows resembling retained physics and policy records.

    Full collections untrack scalar-only tuples and dicts, so roughly two tracked
    containers per row survive; the receipt reports the measured tracked counts.
    """
    return [dict(i=i, q=[float(i)]*4, meta={'k': (i, 'x')}) for i in range(count)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--startup-records', type=int, default=1_100_000,
                        help='synthetic startup heap rows (about 4 tracked containers each)')
    parser.add_argument('--retained-records', type=int, nargs='*', default=[100_000, 300_000],
                        help='post-freeze retained rows measured cumulatively')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--out', type=Path, required=True, help='new JSON receipt path')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'refusing to overwrite {args.out}')
    gc.collect()
    repo = Path(__file__).resolve().parents[1]
    receipt = dict(schema='cascade.heap-freeze.cpu-benchmark.v1',
        scope='Synthetic CPython heap on this host; collector mechanism only. Not a native Kit, '
              'physics, control-deadline or physical result; no speedup claim for any episode.',
        limitations=['Synthetic rows, not Kit or SDK objects; tracked counts after the first full collection are measured, not assumed.',
                     'Does not measure cyclic-garbage retention growth while frozen, RSS headroom or any native owner loop.'],
        source_sha256={name: hashlib.sha256((repo/name).read_bytes()).hexdigest()
                       for name in ('scripts/bench_heap_freeze.py', 'src/cascade/sim/heap_freeze.py')},
        host=dict(platform=platform.platform(), interpreter=sys.version.split()[0],
                  executable=sys.executable, thresholds=list(gc.get_threshold())),
        startup_records=args.startup_records, repeats=args.repeats)
    startup = records(args.startup_records)
    receipt['startup_heap'] = dict(tracked_objects=len(gc.get_objects()),
                                   allocated_blocks=sys.getallocatedblocks(),
                                   full_collect_live=timed(gc.collect, args.repeats))
    policy = StartupHeapFreeze()
    applied = policy.apply()
    receipt['apply'] = dict(collect_ms=round(applied['collect']['duration_ns']/1e6, 3),
                            freeze_ms=round(applied['freeze']['duration_ns']/1e6, 3),
                            frozen=applied['freeze']['frozen'],
                            foreign_frozen_before=applied['freeze']['foreign_frozen_before'])
    receipt['after_freeze'] = [dict(retained_records=0, full_collect=timed(gc.collect, args.repeats))]
    retained = []
    for count in args.retained_records:
        retained.extend(records(count))  # cumulative: rows retained for the episode's lifetime
        receipt['after_freeze'].append(dict(retained_records=len(retained),
                                            tracked_objects=len(gc.get_objects()),
                                            full_collect=timed(gc.collect, args.repeats)))
    released = policy.release()
    receipt['release'] = dict(unfreeze_ms=round(released['unfreeze']['duration_ns']/1e6, 3),
                              collect_ms=round(released['collect']['duration_ns']/1e6, 3),
                              unfrozen=released['unfrozen'], policy_frozen=released['policy_frozen'],
                              frozen_changed_since_apply=released['frozen_changed_since_apply'])
    receipt['after_release'] = dict(full_collect_live=timed(gc.collect, min(3, args.repeats)))
    del startup, retained
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: receipt[k] for k in ('startup_heap', 'apply', 'after_freeze', 'release')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
