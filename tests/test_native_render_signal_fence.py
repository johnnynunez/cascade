"""Held-out CPU regression of real native capture orchestration, not physical acceptance.

Only the SDK objects are doubles. KitNewtonBackend.capture() and
capture_bound_rgb() are imported unchanged from the frozen CASCADE source.
Real POSIX signals are delivered to our own subprocess and deliberately caught
by its SDK callback, matching the callback-swallow failure mechanism.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys

import pytest


@pytest.mark.parametrize('signum', [0, int(signal.SIGINT), int(signal.SIGTERM)])
@pytest.mark.parametrize('captures_before', [0, 1], ids=['cold-16-updates', 'warm-3-updates'])
def test_native_capture_fences_swallowed_callback_signal(tmp_path, signum, captures_before):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, __file__, '--child', str(signum), str(captures_before)],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, (result.stdout, result.stderr)
    receipt = json.loads(result.stdout)
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    assert receipt['handlers_restored']
    assert Path(receipt['source']).resolve() == Path(__file__).resolve().parents[1] / 'src/cascade/sim/microduck_newton.py'
    if signum:
        assert receipt['callback_swallowed'] and receipt['recorded_signum'] == signum, receipt
        assert receipt['updates_after_signal'] == 0, receipt
        assert receipt['readbacks_after_signal'] == 0, receipt
        assert receipt['signal_escaped_capture'] == signum, receipt
        assert not receipt['capture_returned'], receipt
    else:
        assert receipt['capture_returned'] and not receipt['signal_escaped_capture'], receipt
        assert receipt['app_updates'] == (16 if captures_before == 0 else 3), receipt
        assert receipt['readbacks'] == 1, receipt


@pytest.mark.parametrize('signum', [int(signal.SIGINT), int(signal.SIGTERM)])
@pytest.mark.parametrize('captures_before', [0, 1], ids=['cold-last-update', 'warm-last-update'])
def test_last_render_callback_stops_before_readback(tmp_path, signum, captures_before):
    """Held-out boundary: there are no remaining updates to reveal a missing fence."""
    updates = 16 if captures_before == 0 else 3
    result = subprocess.run([sys.executable, __file__, '--child', str(signum),
                             str(captures_before), str(updates)],
                            env=dict(os.environ), capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, (result.stdout, result.stderr)
    receipt = json.loads(result.stdout)
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    assert receipt['handlers_restored'] and receipt['callback_swallowed'], receipt
    assert receipt['updates_after_signal'] == 0, receipt
    assert receipt['readbacks_after_signal'] == 0, receipt
    assert receipt['signal_escaped_capture'] == signum and not receipt['capture_returned'], receipt


def child(signum, captures_before, trigger_update=1):
    from types import SimpleNamespace as NS
    import numpy as np
    from cascade.apps.signal_stop import SignalRequest, StopSignals
    import cascade.sim.microduck_newton as native
    before_handlers = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM]}
    receipt = dict(source=native.__file__, app_updates=0, updates_after_signal=0,
                   readbacks=0, readbacks_after_signal=0, callback_swallowed=False,
                   signal_escaped_capture=None, capture_returned=False,
                   signum=signum, captures_before=captures_before,
                   evidence_scope='CPU software regression; SDK doubles; no physical acceptance')
    backend = native.KitNewtonBackend(NS(), {}, None)
    backend._captures = captures_before
    backend._guard = lambda: None  # No SDK installed/loaded; not the unit under test.
    backend.ns = NS(simulation_step_count=2, sim_time=.01, update_fabric=lambda: None)
    def update():
        if receipt['callback_swallowed']:
            receipt['updates_after_signal'] += 1
        receipt['app_updates'] += 1
        if signum and receipt['app_updates'] == trigger_update:
            try:
                os.kill(os.getpid(), signum)
            except SignalRequest:
                receipt['callback_swallowed'] = True
    def readback(_name):
        receipt['readbacks'] += 1
        if receipt['callback_swallowed']:
            receipt['readbacks_after_signal'] += 1
        return np.zeros((480, 640, 3), dtype=np.uint8), {}
    backend.app = NS(update=update)
    backend.readback = NS(get_data=readback, get_render_times=lambda: {
        'rpFabricTime': {'fabricFrameTimeNumerator': 10000000, 'fabricFrameTimeDenominator': 1000000000},
        'IsaacReadSimulationTime': {'simulationTime': .01}})
    with StopSignals(protect_registration=True) as signals:
        backend.signals = signals
        try:
            backend.capture()
            receipt['capture_returned'] = True
        except SignalRequest as exc:
            receipt['signal_escaped_capture'] = exc.signum
        receipt['recorded_signum'] = signals.signum
    receipt['handlers_restored'] = all(signal.getsignal(s) == h for s, h in before_handlers.items())
    print(json.dumps(receipt))


if __name__ == '__main__':
    assert sys.argv[1] == '--child'
    child(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
