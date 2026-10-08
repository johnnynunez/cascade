"""Hermes-owned harness: B29 live recipe (docs/HUMANOID_H2.md, candidate revision 3 turn goal ramp).

Diagnostic, not physical admission. One H2 owner (PhysX, Isaac Sim 6.2 build, GPU 0) started from the PR #247
tree WITH `--velocity-scaling`; hello.capabilities must list `velocity_scaling` or the run aborts before motion.
Blocks on the same owner (ABA, so drift over time is visible):
  R1  ramp ON  (h2_velocity_candidate turn_control.goal_ramp, inherited): turn +0.8, -0.8, +1.0, -1.0
  N1  ramp OFF (same private base + `turn_control: {goal_ramp: null}`, revision-2 behaviour): same four turns
  R2  ramp ON again: +0.8, -0.8
  M   ramp ON through `python -m cascade.apps.mcp_server` (CASCADE_ROBOT=h2, stdio): +0.8, -0.8
Verifier limits are the profile's, unchanged. Per turn: verdict, reason, measured angle/path, turn_rate_updates,
commanded_rate_at_stop_rad_s, wall time, sim time used (from the verifier samples), health after.

Usage: CASCADE_H2_REPO=<PR #247 worktree> h2_ramp.py RUN_NAME [--port 18421] [--dry-config]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2_episode as H  # noqa: E402
import h2_geometric as G  # noqa: E402

REPO = H.REPO
TURNS_FULL = (0.8, -0.8, 1.0, -1.0)
TURNS_SHORT = (0.8, -0.8)
PAUSE_S = 2.5


def write_noramp(configs: Path, name='configs_noramp') -> Path:
    """Copy of the ramp config dir whose base turns the inherited goal ramp off (explicit null)."""
    import yaml
    dst = configs.parent / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(configs, dst)
    path = dst / 'bases' / 'h2_episode.yaml'
    base = yaml.safe_load(path.read_text())
    base['turn_control'] = {'goal_ramp': None}
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return dst


def diff(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else k)
    elif a != b:
        out.append((path, a, b))
    return out


def resolved_base(configs, epoch):
    cfg, base = H.load_cfg(configs, epoch)
    return cfg, base


def check_arms(configs_ramp, configs_noramp, epoch):
    _, a = resolved_base(configs_ramp, epoch)
    _, b = resolved_base(configs_noramp, epoch)
    d = diff(a, b)
    ramp = (a.get('turn_control') or {}).get('goal_ramp')
    ok = bool(ramp) and [p for p, _, _ in d] == ['turn_control.goal_ramp']
    return {'ok': ok, 'diff': [(p, x, y) for p, x, y in d], 'ramp_values': ramp}


def turn_row(runtime_or_mcp, angle, tag, run, via):
    t0 = time.monotonic()
    if via == 'runtime':
        res = runtime_or_mcp.execute('locomotion.turn', {'angle_rad': angle})
    else:
        res = runtime_or_mcp.tool('locomotion.turn', {'angle_rad': angle})
    wall = time.monotonic() - t0
    (run / f'turn-{tag}.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    post = res.get('postcondition') or {}
    samples = (res.get('measured') or {}).get('samples') or []
    sim = [s.get('sim_time_s') for s in samples if s.get('sim_time_s') is not None]
    row = {'tag': tag, 'via': via, 'angle_rad': angle, 'wall_s': round(wall, 2),
           'confirmed': res.get('ok') is True and res.get('execution_ok') is True and post.get('status') == 'confirmed',
           'ok': res.get('ok'), 'execution_ok': res.get('execution_ok'), 'post_status': post.get('status'),
           'post_reason': post.get('reason') or post.get('reasons'), 'error': res.get('error'),
           'measured_angle_rad': res.get('measured_angle_rad'),
           'measured_translation_path_m': res.get('measured_translation_path_m'),
           'commanded_rate_at_stop_rad_s': res.get('commanded_rate_at_stop_rad_s'),
           'turn_rate_updates': res.get('turn_rate_updates'),
           'verifier_samples': len(samples), 'verifier_sim_span_s': round(max(sim) - min(sim), 3) if sim else None}
    print(json.dumps({k: row[k] for k in ('tag', 'confirmed', 'post_status', 'post_reason', 'error', 'measured_angle_rad',
                                          'commanded_rate_at_stop_rad_s', 'wall_s')}, default=str), flush=True)
    return row, res


def block(cfg, run, label, turns, report, reset_on_latch=False):
    from cascade.apps.robot_runtime import build_robot_runtime
    runtime, _ = build_robot_runtime(cfg, run / f'runtime-{label}')
    rows = []
    try:
        before = H.state_row(runtime)
        reset = runtime.execute('reset_stop', {})
        report.setdefault('resets', {})[label] = H.compact(reset)
        if reset.get('ok') is not True:
            raise RuntimeError(f'reset_stop refused before block {label}: {H.compact(reset)}')
        t_wall = time.monotonic()
        latched = False
        for i, angle in enumerate(turns):
            if latched and reset_on_latch:
                # the previous turn ended in a fail-closed latched stop (e.g. the 3 s command deadline):
                # the staff reset channel clears it so this trial is independent; recorded per turn.
                r = runtime.execute('reset_stop', {})
                report.setdefault('per_turn_resets', []).append({'before': f'{label}-{i+1}', 'ok': r.get('ok')})
                time.sleep(PAUSE_S)
            row, res = turn_row(runtime, angle, f'{label}-{i+1}-{angle:+.1f}', run, 'runtime')
            latched = bool((res.get('stop_ack') or {}).get('latched')) or 'latched' in str(res.get('error') or '')
            row['stop_latched'] = latched
            after = H.state_row(runtime)
            row['after'] = after
            rows.append(row)
            report['blocks'][label] = rows
            H.save(run / 'ramp.json', report)
            if after.get('fallen') or after.get('controller_status') in ('fault', 'disabled'):
                raise RuntimeError(f'robot not healthy after {row["tag"]}: {after}')
            time.sleep(PAUSE_S)
        end = H.state_row(runtime)
        if before.get('sim_time_s') is not None and end.get('sim_time_s') is not None:
            report.setdefault('rt_factor', {})[label] = round(
                (end['sim_time_s'] - before['sim_time_s']) / (time.monotonic() - t_wall), 3)
    finally:
        report.setdefault('runtime_close', {})[label] = runtime.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name')
    ap.add_argument('--port', type=int, default=18421)
    ap.add_argument('--max-wall-s', type=float, default=1500.)
    ap.add_argument('--dry-config', action='store_true',
                    help='no owner: write both arms with dummy pins and check they differ only in goal_ramp')
    ap.add_argument('--plan', default='',
                    help='blocks "R:+0.8,-0.8|N:+0.8,-0.8|..." (R ramp on, N ramp off); empty = run-1 design')
    ap.add_argument('--mcp-turns', default='+0.8,-0.8', help='ramp-on turns through the MCP server ("" = skip)')
    ap.add_argument('--reset-on-latch', action='store_true',
                    help='reset_stop before a turn whose predecessor ended latched (independent trials)')
    args = ap.parse_args()
    run = HERE / 'runs' / args.name
    run.mkdir(parents=True, exist_ok=False)
    report = {'name': args.name, 'physical_admission': False, 'repo': str(REPO), 'blocks': {},
              'loadavg_start': os.getloadavg()}
    if args.dry_config:
        z = '0' * 64
        marker = {'port': args.port, 'hello': {'model_identity_sha256': z, 'device': 'cuda:0', 'asset_sha256': z,
                                               'policy_sha256': z, 'epoch': 'dry-epoch'}}
        kit = run / 'kit'
        kit.mkdir()
        (kit / 'model-identity.json').write_text(json.dumps({'model_identity_sha256': z, 'support_contract': {}}))
        configs, _ = H.write_robot_config(run, {'out': str(kit)}, marker)
        noramp = write_noramp(configs)
        report['arms'] = check_arms(configs, noramp, 'dry-epoch')
        print(json.dumps(report['arms'], default=str))
        H.save(run / 'ramp.json', report)
        return 0 if report['arms']['ok'] else 1
    record = H.launch_owner(run, port=args.port, max_wall_s=args.max_wall_s, max_steps=600000, camera_every=10,
                            extra_args=('--velocity-scaling',))
    private = []
    try:
        marker = H.wait_listening(record)
        report['hello'] = marker['hello']
        caps = marker['hello'].get('capabilities') or []
        report['velocity_scaling_advertised'] = 'velocity_scaling' in caps
        print(json.dumps({'phase': 'hello', 'capabilities': caps}), flush=True)
        if not report['velocity_scaling_advertised']:
            raise RuntimeError('owner does not advertise velocity_scaling; every ramped turn would be refused')
        configs, report['model_identity_sha256'] = H.write_robot_config(run, record, marker)
        noramp = write_noramp(configs)
        epoch = marker['hello']['epoch']
        report['arms'] = check_arms(configs, noramp, epoch)
        print(json.dumps({'phase': 'arms', **report['arms']}, default=str), flush=True)
        if not report['arms']['ok']:
            raise RuntimeError('ramp/no-ramp arms differ in more than goal_ramp')
        cfg_r, base_r = H.load_cfg(configs, epoch)
        cfg_n, _ = H.load_cfg(noramp, epoch)
        H.save(run / 'resolved-config-ramp.json', cfg_r.as_dict())
        H.save(run / 'resolved-config-noramp.json', cfg_n.as_dict())
        report['passive_samples'] = len(H.passive_ready(base_r))
        report['plan'] = args.plan or 'R1:TURNS_FULL|N1:TURNS_FULL|R2:TURNS_SHORT'
        report['reset_on_latch'] = args.reset_on_latch
        if args.plan:
            counts = {'R': 0, 'N': 0}
            for spec in args.plan.split('|'):
                arm, turns = spec.split(':')
                counts[arm] += 1
                block(cfg_r if arm == 'R' else cfg_n, run, f'{arm}{counts[arm]}',
                      tuple(float(t) for t in turns.split(',')), report, reset_on_latch=args.reset_on_latch)
        else:
            block(cfg_r, run, 'R1', TURNS_FULL, report)
            block(cfg_n, run, 'N1', TURNS_FULL, report)
            block(cfg_r, run, 'R2', TURNS_SHORT, report)
        mcp_turns = tuple(float(t) for t in args.mcp_turns.split(',')) if args.mcp_turns else ()
        if not mcp_turns:
            return 0
        # M: the chat-host path, ramp ON. The server reads only REPO/configs: copy the private pair in
        # (untracked) and remove it in finally.
        for sub, name in (('robots', 'h2.yaml'), ('bases', 'h2_episode.yaml')):
            dst = REPO / 'configs' / sub / name
            if dst.exists():
                raise RuntimeError(f'refusing to overwrite {dst}')
            shutil.copy(configs / sub / name, dst)
            private.append(dst)
        mcp = G.McpClient(run, configs)
        try:
            report['mcp_first_state'] = mcp.tool('locomotion.get_base_state', {})
            report['mcp_reset_stop'] = mcp.tool('reset_stop', {})
            rows = []
            latched = False
            for i, angle in enumerate(mcp_turns):
                if latched and args.reset_on_latch:
                    r = mcp.tool('reset_stop', {})
                    report.setdefault('per_turn_resets', []).append({'before': f'M-{i+1}', 'ok': r.get('ok')})
                    time.sleep(PAUSE_S)
                row, res = turn_row(mcp, angle, f'M-{i+1}-{angle:+.1f}', run, 'mcp')
                latched = bool((res.get('stop_ack') or {}).get('latched')) or 'latched' in str(res.get('error') or '')
                row['stop_latched'] = latched
                rows.append(row)
                report['blocks']['M'] = rows
                H.save(run / 'ramp.json', report)
                time.sleep(PAUSE_S)
            report['mcp_last_state'] = mcp.tool('locomotion.get_base_state', {})
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
        report['owner_terminate'] = H.terminate(record)
        try:
            receipt = json.loads((Path(record['out']) / 'receipt.json').read_text())
            report['owner_receipt'] = {k: receipt.get(k) for k in ('completed', 'end_reason', 'error', 'steps',
                                                                   'frame_count', 'wall_duration_s', 'exit_code',
                                                                   'last_fall_evidence')}
        except Exception as exc:  # noqa: BLE001
            report['owner_receipt_error'] = repr(exc)
        report['loadavg_end'] = os.getloadavg()
        H.save(run / 'ramp.json', report)
        summary = {b: [(r['tag'], r['confirmed'], r['post_status'], r.get('commanded_rate_at_stop_rad_s'))
                       for r in rows] for b, rows in report['blocks'].items()}
        print(json.dumps({'summary': summary, 'rt_factor': report.get('rt_factor'), 'error': bool(report.get('error')),
                          'owner_receipt': report.get('owner_receipt')}, default=str)[:3000], flush=True)
    return 1 if report.get('error') else 0


if __name__ == '__main__':
    sys.exit(main())
