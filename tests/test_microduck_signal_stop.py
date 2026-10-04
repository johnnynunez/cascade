"""Real OS signals through the native CLI, CPU backend only; NOT physics."""
from pathlib import Path
from cascade.sim.microduck_policy_admission import target_contract

import json
import os
import signal
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('site', ['startup', 'inference', 'commit', 'upload', 'bam', 'capture', 'cleanup', 'sdk_constructor', 'sdk_after_boot',
    'receipt_before', 'receipt_after', 'sdk_return', 'sdk_raise',
    'sdk_boot_swallowed', 'capture_swallowed', 'inference_swallowed',
    pytest.param('blocked_inference', marks=pytest.mark.skipif(
        not sys.platform.startswith('linux'), reason='kernel pipe wait proof uses Linux procfs'))])
def test_native_signal_unwinds_before_more_work(tmp_path, signum, site):
    env = {k: v for k, v in os.environ.items() if not k.startswith('CASCADE_')}
    stores = tmp_path / 'stores'
    stores.mkdir()
    home, temporary = tmp_path / 'home', tmp_path / 'tmp'
    home.mkdir()
    temporary.mkdir()
    env.update(HOME=str(home), TMPDIR=str(temporary), XDG_CACHE_HOME=str(home/'cache'),
               CASCADE_BELIEFS_PATH=str(stores/'beliefs.json'),
               CASCADE_GRASP_MEMORY_PATH=str(stores/'grasp.json'),
               CASCADE_ENVELOPE_PATH=str(stores/'envelope.json'),
               PYTHONPATH=f'{REPO}/src:{REPO}/scripts:{REPO}/tests',
               CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, str(Path(__file__).resolve()), '--child',
               site, str(int(signum)), str(tmp_path)]
    if site == 'blocked_inference':
        result = _interrupt_blocked_child(command, env, signum)
    else:
        result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=6)
    details = json.loads((tmp_path/'signal-result.json').read_text())
    assert details['sent'], (result.stdout, result.stderr, details)
    if site.endswith('_swallowed'):
        assert details['callback_swallowed_signal'] is True, details
    assert result.returncode == 128 + signum, (result.stdout, result.stderr, details)
    assert details['solves_after_signal'] == 0, details
    assert details['uploads_after_signal'] == 0, details
    assert details['bam_preparations_after_signal'] == 0, details
    assert details['server_starts_after_signal'] == 0, details
    assert details['handlers_restored'] and not details['owned_threads'], details
    assert not details.get('foreign_handler_called', False), details
    # A signal arriving inside SDK.close cannot change its already-consumed
    # argument. If it returns, process/result/receipt must still become 128+signal.
    requested_exit = 0 if site in ('sdk_return', 'sdk_raise') else 128 + signum
    assert details['closed'] == 1 and details['shutdown_code'] == requested_exit, details
    receipt = json.loads((tmp_path/'run/receipt.json').read_text())
    assert receipt['completed'] is False and receipt['signal'] == signum, receipt
    assert receipt['physical_acceptance'] is False, receipt
    assert bool(receipt['teardown_errors']) is (site == 'sdk_raise'), receipt
    assert receipt == details['returned_result'], (receipt, details)
    if site in ('inference', 'inference_swallowed', 'blocked_inference', 'commit', 'upload'):
        rows = [json.loads(x) for x in (tmp_path/'run/policy.jsonl').read_text().splitlines()]
        interrupted = rows[-1]
        assert interrupted['status'] == 'interrupted', interrupted
        if site == 'commit':
            assert interrupted['committed'] is None, interrupted
            assert interrupted['commit_outcome'] == 'unknown_due_to_interruption', interrupted
        else:
            assert interrupted['committed'] is (site == 'upload'), interrupted
        assert 'first_step_after_commit' not in interrupted, interrupted
        if site in ('inference', 'inference_swallowed', 'blocked_inference'):
            assert interrupted['commands'][0] > 0
            assert details['raw_history_changed_after_signal'] is False


def _interrupt_blocked_child(command, env, signum):
    import select
    import time
    notify_read, notify_write = os.pipe()
    env['MICRODUCK_SIGNAL_NOTIFY_FD'] = str(notify_write)
    child = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, pass_fds=(notify_write,))
    os.close(notify_write)
    try:
        assert select.select([notify_read], [], [], 3)[0], 'child did not reach native inference read'
        assert os.read(notify_read, 1) == b'I'
        deadline = time.monotonic() + 2
        while 'pipe_read' not in Path(f'/proc/{child.pid}/wchan').read_text():
            assert time.monotonic() < deadline, 'main thread never entered kernel read'
            os.sched_yield()
        child.send_signal(signum)
        try:
            stdout, stderr = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate(timeout=2)
            pytest.fail(f'native CLI left inference blocked after signal: {stdout}\n{stderr}')
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
    finally:
        os.close(notify_read)
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=2)


def _child(site, signum, directory):
    import threading
    from types import SimpleNamespace
    import numpy as np
    import isaac_microduck_bridge as cli
    if os.environ.get('MICRODUCK_SIGNAL_CLI_SNAPSHOT'):
        # RED replay of the frozen pre-fix CLI; never edit a shared source.
        import importlib.util
        spec = importlib.util.spec_from_file_location('native_signal_baseline',
            os.environ['MICRODUCK_SIGNAL_CLI_SNAPSHOT'])
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cli.REPO = REPO
    from cascade.sim.mobile_bridge import MobileBridgeServer
    from test_microduck_bridge_cli import software_limits, software_model_identity
    from cascade.sim import mobile_identity
    # This child tests signals with a synthetic backend, never native admission.
    mobile_identity.build_model_identity = software_model_identity
    from test_microduck_stepper import SoftwareBackend, SoftwarePolicy, SoftwareActuator, render_times

    directory = Path(directory)
    state = {'sent': False, 'solves_after_signal': 0, 'uploads_after_signal': 0,
             'bam_preparations_after_signal': 0, 'server_starts_after_signal': 0}
    original_setter = signal.signal
    original_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    before_threads = set(threading.enumerate())
    owners = {}
    lock = threading.Lock()

    def send():
        if state['sent']:
            return
        state['sent'] = True
        if 'policy' in owners:
            state['raw_before'] = owners['policy'].previous_action.tolist()
        # A real non-reentrant critical section, unwound before cleanup/stop.
        with lock:
            if site == 'blocked_inference':
                rfd, wfd = os.pipe()
                try:
                    os.write(int(os.environ['MICRODUCK_SIGNAL_NOTIFY_FD']), b'I')
                    os.read(rfd, 1)  # parent verifies kernel pipe_read before signalling
                finally:
                    os.close(rfd)
                    os.close(wfd)
            else:
                os.kill(os.getpid(), signum)

    def swallowed_callback():
        # Kit/async callbacks can catch an exception delivered by a real OS
        # signal. The recorded signum must still fence the next operation.
        from cascade.apps.signal_stop import SignalRequest
        try:
            send()
        except SignalRequest:
            state['callback_swallowed_signal'] = True

    class Actuator(SoftwareActuator):
        def set_targets(self, q):
            if site == 'upload' and len(self.targets) == 1:
                send()
            if state['sent']:
                state['uploads_after_signal'] += 1
            return super().set_targets(q)

        def before_step(self, dt):
            if site == 'bam' and self.backend.step_count >= 6:
                send()
            if state['sent']:
                state['bam_preparations_after_signal'] += 1
            return super().before_step(dt)

    class Backend(SoftwareBackend):
        def __init__(self, *_args):
            super().__init__()
            self.bam = Actuator(self)
            self.receipt = {'software_fixture': True}
            self.captures = 0
            owners['backend'] = self

        def open(self):
            if site in ('sdk_constructor', 'sdk_after_boot'):
                # Installed SimulationApp unconditionally replaces SIGINT after
                # starting Kit, then its handler exits0 and unloads native code.
                def sdk_handler(_sig, _frame):
                    state['foreign_handler_called'] = True
                    raise SystemExit(0)
                signal.signal(signal.SIGINT, sdk_handler)
            if site in ('startup', 'sdk_constructor'):
                send()
            if site == 'sdk_boot_swallowed':
                swallowed_callback()

        def step(self):
            if state['sent']:
                state['solves_after_signal'] += 1
            return super().step()

        def capture(self):
            self.captures += 1
            if site == 'capture' and self.captures == 2:
                send()
            if site == 'capture_swallowed' and self.captures == 2:
                swallowed_callback()
            return dict(rgb=np.zeros((24, 32, 3), np.uint8), step=self.step_count,
                        sim_time_s=self.sim_time, captured_at=0., render_times=render_times(self.sim_time))

        def close(self):
            if site == 'cleanup':
                send()
            with lock:
                super().close()

        def shutdown(self, exit_code):
            assert (directory/'run/receipt.json').is_file()
            state['shutdown_code'] = exit_code
            if site in ('sdk_return', 'sdk_raise'):
                send()
            if site == 'sdk_raise':
                raise RuntimeError('software SDK close failure')
            # This software adapter does not call an SDK: None means unknown.

    class Server(MobileBridgeServer):
        def __init__(self, controller, **kwargs):
            owners['controller'] = controller
            super().__init__(controller, **kwargs)

        def start(self):
            if state['sent']:
                state['server_starts_after_signal'] += 1
            super().start()
            h = self.controller.hello()
            ack = self.controller.command_velocity(dict(robot_id=h['robot_id'], source=h['source'],
                epoch=h['epoch'], generation=h['generation'], owner='signal-fixture',
                command_id='signal-active-motion', vx=.1, vy=0., wz=0., duration_s=.1))
            assert ack['ok']

    def policy_factory(*_args, **kwargs):
        p = SoftwarePolicy(owners['backend'])
        def infer(obs):
            if site in ('inference', 'blocked_inference', 'sdk_after_boot') and obs[0, 48] != 0:
                with owners['controller']._lock:
                    send()
            if site == 'inference_swallowed' and obs[0, 48] != 0:
                swallowed_callback()
            return np.full(14, obs[0, 48], np.float32)
        p.infer = infer
        commit = p.commit
        def commit_then_signal(action):
            commit(action)
            if site == 'commit' and np.any(action):
                send()  # history copied, but the owner did not receive a return
        p.commit = commit_then_signal
        owners['policy'] = p
        return p

    args = SimpleNamespace(out=directory/'run', device='cuda:0', robot_id='microduck',
        source='signal-software-only', max_wall_s=3., max_steps=9, port=0,
        camera_every=4, max_jpeg_bytes=100000, policy=directory/'fixture.onnx',
        policy_sha256='b'*64, target_profile='direct-v1', python_extra_path=[], check_only=False)
    admission = dict(target_contract=target_contract('b'*64, 'direct-v1'), asset_sha256='a'*64, asset_receipt_sha256='c'*64,
                     bam_params={}, limits=software_limits(), experience_text='software fixture\n')
    original_write = cli.write_json
    def checked_write(path, value):
        if Path(path).name == 'receipt.json' and site == 'receipt_before':
            send()
        original_write(path, value)
        if Path(path).name == 'receipt.json' and site == 'receipt_after':
            send()
    cli.write_json = checked_write
    real_run = cli.run
    def run(a, b, **kw):
        result = real_run(a, b, backend_factory=Backend, policy_factory=policy_factory,
                          server_factory=Server, **kw)
        state['returned_result'] = result
        return result
    cli.parse_args = lambda _argv: args
    cli.admit = lambda _args: admission
    cli.run = run
    try:
        code = cli.main([])
    finally:
        state['handlers_restored'] = (signal.signal is original_setter and
            all(signal.getsignal(s) == h for s, h in original_handlers.items()))
        state['closed'] = owners['backend'].closed
        state['owned_threads'] = [t.name for t in set(threading.enumerate()) - before_threads if t.is_alive()]
        if 'raw_before' in state:
            state['raw_history_changed_after_signal'] = state['raw_before'] != owners['policy'].previous_action.tolist()
        (directory/'signal-result.json').write_text(json.dumps(state, indent=2))
    return code


if __name__ == '__main__':
    assert sys.argv[1] == '--child'
    raise SystemExit(_child(sys.argv[2], int(sys.argv[3]), sys.argv[4]))
