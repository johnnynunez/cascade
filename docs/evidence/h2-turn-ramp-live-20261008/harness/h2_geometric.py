"""Hermes-owned harness: H2 settle measurement + geometric skills + MCP-driven command, one owner.

Diagnostic, not physical admission. Reuses h2_episode.py (owner launch on GPU 0, private
config pinned to the owner identity). Phases:
  A  standing: no command, BaseTruthReader sampled for --standing-s (how still is the H2 at rest?)
  B  walk_velocity 0.3 m/s x 3 s through RobotRuntime -> SafeBase (verifier), then the reader
     sampled for --settle-s: speed/yaw-rate decay after the zero twist (what the verifier's
     0.4 s settle window sees, and when the robot would pass the candidate stop limits)
  C  walk_distance +0.5 / turn +0.8 rad / walk_distance -0.5 (candidate distance_control /
     turn_control in configs/bases/h2_velocity_candidate.yaml), verifier verdicts
  D  the same walk_velocity through `python -m cascade.apps.mcp_server` (CASCADE_ROBOT=h2,
     JSON-RPC over stdio) -- the chat-host path, not a Python call.
Usage: h2_geometric.py RUN_NAME [--standing-s 8] [--settle-s 10]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2_episode as H  # noqa: E402

REPO = H.REPO


class _SkipToMcp(Exception):
    pass


def sample_reader(base, seconds, label):
    from cascade.sim.base_truth import BaseTruthReader
    reader = BaseTruthReader(base)
    rows = []
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            s = reader()
            if s is not None:
                d = s.as_dict()
                rows.append({k: d.get(k) for k in ('sim_time_s', 'position_world', 'orientation_wxyz',
                                                   'linear_velocity_world', 'angular_velocity_body',
                                                   'controller_status', 'fallen', 'latched')})
            time.sleep(0.02)
    finally:
        reader.close()
    # dedupe by sim time (the reader may return the same completed state twice)
    out, seen = [], set()
    for r in rows:
        if r['sim_time_s'] not in seen:
            seen.add(r['sim_time_s']); out.append(r)
    return out


def stats(rows, t0=None):
    """speed / yaw-rate profile; t relative to t0 (sim seconds)."""
    if not rows:
        return {}
    t0 = rows[0]['sim_time_s'] if t0 is None else t0
    sp = [math.hypot(r['linear_velocity_world'][0], r['linear_velocity_world'][1]) for r in rows]
    om = [abs(r['angular_velocity_body'][2]) for r in rows]
    omn = [math.sqrt(sum(v*v for v in r['angular_velocity_body'])) for r in rows]
    def pct(v, q):
        v = sorted(v); return v[min(len(v)-1, int(q*len(v)))]
    pos = [r['position_world'] for r in rows]
    drift = sum(math.dist(a[:2], b[:2]) for a, b in zip(pos, pos[1:]))
    yaw = [2*math.atan2(r['orientation_wxyz'][3], r['orientation_wxyz'][0]) for r in rows]
    # first sim time after which every later sample is below the candidate stop limits
    below_from = None
    for i in range(len(rows)):
        if all(sp[j] < .08 and omn[j] < .20 for j in range(i, len(rows))):
            below_from = rows[i]['sim_time_s'] - t0; break
    # per 0.5 s bins: max speed, max |omega|
    bins = {}
    for r, s, o in zip(rows, sp, omn):
        b = round(math.floor((r['sim_time_s']-t0)/.5)*.5, 1)
        m = bins.setdefault(b, [0., 0.]); m[0] = max(m[0], s); m[1] = max(m[1], o)
    return {'samples': len(rows), 'sim_span_s': round(rows[-1]['sim_time_s']-rows[0]['sim_time_s'], 3),
            'speed_max': round(max(sp), 3), 'speed_p95': round(pct(sp, .95), 3), 'speed_median': round(pct(sp, .5), 3),
            'yaw_rate_abs_max': round(max(om), 3), 'yaw_rate_abs_p95': round(pct(om, .95), 3),
            'omega_norm_max': round(max(omn), 3), 'omega_norm_p95': round(pct(omn, .95), 3), 'omega_norm_median': round(pct(omn, .5), 3),
            'xy_path_m': round(drift, 4), 'yaw_change_rad': round(yaw[-1]-yaw[0], 4),
            'controller_status': sorted(set(r['controller_status'] for r in rows)),
            'all_later_samples_below_stop_limits_from_s': None if below_from is None else round(below_from, 2),
            'bins_0p5s_max_speed_omega': {str(k): [round(v[0], 3), round(v[1], 3)] for k, v in sorted(bins.items())}}


class McpClient:
    def __init__(self, run_dir, configs_src):
        env = {k: v for k, v in os.environ.items() if not k.startswith('CASCADE_')}
        env.update(CASCADE_ROBOT='h2', CASCADE_PREWARM='0', CASCADE_STREAM='0', CASCADE_VIEW='0',
                   CASCADE_RUN_DIR=str(run_dir / 'mcp'), CASCADE_BELIEFS_PATH=str(run_dir / 'mcp-beliefs.json'),
                   CASCADE_GRASP_MEMORY_PATH=str(run_dir / 'mcp-grasp.json'), CASCADE_ENVELOPE_PATH=str(run_dir / 'mcp-envelope.json'),
                   PYTHONPATH=str(REPO / 'src'), CUDA_VISIBLE_DEVICES='-1')
        env.pop('PYTHONPATH_IGNORED', None)
        self.proc = subprocess.Popen([str(Path('/home/johnny/Projects/demo/cascade/.venv/bin/python')), '-m', 'cascade.apps.mcp_server'],
                                     cwd=str(REPO), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(run_dir / 'mcp-stderr.log', 'w'), text=True, bufsize=1)
        self.q, self.n = queue.Queue(), 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.q.put(line)
        self.q.put(None)

    def call(self, method, params=None, timeout=240):
        self.n += 1
        self.proc.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': self.n, 'method': method, 'params': params or {}}) + '\n')
        self.proc.stdin.flush()
        while True:
            line = self.q.get(timeout=timeout)
            if line is None:
                raise RuntimeError('mcp server closed')
            msg = json.loads(line)
            if msg.get('id') == self.n:
                return msg

    def tool(self, name, arguments=None, timeout=240):
        msg = self.call('tools/call', {'name': name, 'arguments': arguments or {}}, timeout)
        if 'error' in msg:
            return {'ok': False, 'rpc_error': msg['error']}
        text = msg['result']['content'][-1]['text']
        try:
            return json.loads(text)
        except ValueError:
            return {'raw': text}

    def close(self):
        try:
            self.proc.stdin.close(); self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name')
    ap.add_argument('--port', type=int, default=18401)
    ap.add_argument('--max-wall-s', type=float, default=1100.)
    ap.add_argument('--standing-s', type=float, default=8.)
    ap.add_argument('--settle-s', type=float, default=10.)
    ap.add_argument('--camera-every', type=int, default=10)
    ap.add_argument('--only-mcp', action='store_true', help='skip A-C; chat-host path only: reset_stop, walk_velocity, turn')
    args = ap.parse_args()
    run = HERE / 'runs' / args.name
    run.mkdir(parents=True, exist_ok=False)
    report = {'name': args.name, 'physical_admission': False, 'repo': str(REPO), 'phases': {}}
    record = H.launch_owner(run, port=args.port, max_wall_s=args.max_wall_s, max_steps=400000, camera_every=args.camera_every)
    runtime = None
    private = []
    try:
        marker = H.wait_listening(record)
        report['hello'] = marker['hello']
        configs, report['model_identity_sha256'] = H.write_robot_config(run, record, marker)
        cfg, base = H.load_cfg(configs, marker['hello']['epoch'])
        H.save(run / 'resolved-config.json', cfg.as_dict())
        report['passive_samples'] = len(H.passive_ready(base))
        if args.only_mcp:
            raise _SkipToMcp()
        # A: standing
        rows = sample_reader(base, args.standing_s, 'standing')
        (run / 'standing.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        report['phases']['A_standing'] = stats(rows)
        print(json.dumps({'phase': 'A_standing', **{k: report['phases']['A_standing'][k] for k in ('samples', 'sim_span_s', 'speed_p95', 'omega_norm_p95', 'omega_norm_max', 'xy_path_m')}}), flush=True)
        from cascade.apps.robot_runtime import build_robot_runtime
        runtime, _ = build_robot_runtime(cfg, run / 'runtime')

        def step(tool, call, tag):
            t0 = time.monotonic()
            result = runtime.execute(tool, call)
            row = {'tool': tool, 'args': call, 'wall_s': round(time.monotonic()-t0, 2),
                   'confirmed': result.get('ok') is True and result.get('execution_ok') is True
                   and (result.get('postcondition') or {}).get('status') == 'confirmed', **H.compact(result)}
            for k in ('requested_distance_m', 'measured_distance_m', 'requested_angle_rad', 'measured_angle_rad',
                      'measured_translation_path_m'):
                if k in result:
                    row[k] = result[k]
            samples = (result.get('measured') or {}).get('samples') or []
            (run / f'verifier-samples-{tag}.jsonl').write_text('\n'.join(json.dumps({k: s.get(k) for k in (
                'sim_time_s', 'position_world', 'orientation_wxyz', 'linear_velocity_world', 'angular_velocity_body',
                'controller_status')}) for s in samples) + '\n')
            report['phases'][tag] = row
            print(json.dumps({'phase': tag, **{k: row.get(k) for k in ('args', 'confirmed', 'post_status', 'post_reason', 'error', 'measured_distance_m', 'measured_angle_rad', 'measured_translation_path_m')}}, default=str), flush=True)
            H.save(run / 'episode.json', report)
            after = H.state_row(runtime)
            if after.get('fallen') or after.get('controller_status') in ('fault', 'disabled'):
                raise RuntimeError(f'robot not healthy after {tag}: {after}')
            return result

        # B: walk + settle profile
        step('locomotion.walk_velocity', {'vx': 0.3, 'vy': 0.0, 'wz': 0.0, 'duration_s': 3.0}, 'B_walk_velocity')
        rows = sample_reader(base, args.settle_s, 'settle')
        (run / 'settle.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        report['phases']['B_settle_after_walk'] = stats(rows)
        print(json.dumps({'phase': 'B_settle', **{k: report['phases']['B_settle_after_walk'][k] for k in ('samples', 'sim_span_s', 'speed_p95', 'omega_norm_p95', 'omega_norm_max', 'all_later_samples_below_stop_limits_from_s')}}), flush=True)
        # C: geometric skills
        step('locomotion.walk_distance', {'distance_m': 0.5}, 'C1_walk_distance_fwd')
        time.sleep(1.5)
        step('locomotion.turn', {'angle_rad': 0.8}, 'C2_turn')
        time.sleep(1.5)
        step('locomotion.walk_distance', {'distance_m': -0.5}, 'C3_walk_distance_back')
        report['after_direct'] = H.state_row(runtime)
        report['runtime_close'] = runtime.close(); runtime = None
        time.sleep(2.0)
    except _SkipToMcp:
        pass
    except BaseException:
        report['error'] = traceback.format_exc()
        print(report['error'][-2500:], flush=True)
    try:
        if 'error' in report:
            raise RuntimeError('direct phases failed; MCP phase skipped')
        # D: MCP-driven (the chat-host path). The composed robot profile must be visible to the
        # server's own config dir: copy the two private files into the worktree configs (untracked,
        # removed in finally).
        for sub, name in (('robots', 'h2.yaml'), ('bases', 'h2_episode.yaml')):
            dst = REPO / 'configs' / sub / name
            shutil.copy(configs / sub / name, dst); private.append(dst)
        mcp = McpClient(run, configs)
        try:
            listed = mcp.call('tools/list')
            names = sorted(t['name'] for t in listed.get('result', {}).get('tools', []))
            report['phases']['D_mcp_tools'] = names
            print(json.dumps({'phase': 'D_mcp_tools', 'n': len(names), 'locomotion': [n for n in names if n.startswith('locomotion')]}), flush=True)
            # the previous runtime's close (or an operator stop) leaves the stop latched: the chat
            # host must clear it explicitly before any travel; this is the staff reset channel.
            # A read first: the server builds the mobile runtime lazily on the first tool call and
            # refuses a reset while it is still starting.
            report['phases']['D_mcp_first_state'] = mcp.tool('locomotion.get_base_state', {})
            report['phases']['D_mcp_reset_stop'] = mcp.tool('reset_stop', {})
            print(json.dumps({'phase': 'D_mcp_reset_stop', 'ok': report['phases']['D_mcp_reset_stop'].get('ok')}), flush=True)
            for tag, tool, call in (('D_mcp_walk_velocity', 'locomotion.walk_velocity', {'vx': 0.3, 'vy': 0.0, 'wz': 0.0, 'duration_s': 3.0}),
                                    ('D_mcp_turn', 'locomotion.turn', {'angle_rad': 0.6}),
                                    ('D_mcp_walk_velocity_back', 'locomotion.walk_velocity', {'vx': -0.3, 'vy': 0.0, 'wz': 0.0, 'duration_s': 2.0})):
                t0 = time.monotonic()
                res = mcp.tool(tool, call)
                row = {'tool': tool, 'via': 'mcp stdio (python -m cascade.apps.mcp_server, CASCADE_ROBOT=h2)',
                       'wall_s': round(time.monotonic()-t0, 2), 'confirmed': res.get('ok') is True and res.get('execution_ok') is True
                       and (res.get('postcondition') or {}).get('status') == 'confirmed', **H.compact(res)}
                for k in ('requested_angle_rad', 'measured_angle_rad', 'measured_translation_path_m'):
                    if k in res:
                        row[k] = res[k]
                report['phases'][tag] = row
                print(json.dumps({'phase': tag, **{k: row.get(k) for k in ('args', 'confirmed', 'post_status', 'post_reason', 'error', 'measured_angle_rad')}}, default=str), flush=True)
                H.save(run / 'episode.json', report)
                time.sleep(2.0)
            report['phases']['D_mcp_state'] = mcp.tool('locomotion.get_base_state', {})
        finally:
            mcp.close()
    except BaseException:
        report['error'] = traceback.format_exc()
        print(report['error'][-2500:], flush=True)
    finally:
        for p in private:
            try:
                p.unlink()
            except OSError:
                pass
        if runtime is not None:
            try:
                report['runtime_close'] = runtime.close()
            except BaseException as exc:
                report['runtime_close_error'] = repr(exc)
        report['owner_terminate'] = H.terminate(record)
        try:
            receipt = json.loads((Path(record['out']) / 'receipt.json').read_text())
            report['owner_receipt'] = {k: receipt.get(k) for k in ('completed', 'end_reason', 'error', 'steps', 'frame_count',
                                                                   'wall_duration_s', 'exit_code', 'last_fall_evidence')}
        except Exception as exc:  # noqa: BLE001
            report['owner_receipt_error'] = repr(exc)
        H.save(run / 'episode.json', report)
        print(json.dumps({k: report.get(k) for k in ('owner_receipt', 'error')}, default=str)[:1200])


if __name__ == '__main__':
    main()
