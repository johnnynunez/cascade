"""Build docs/evidence/h2-turn-ramp-live-20261008/ from the raw B29 live runs (tables computed from the rows)."""
import collections
import hashlib
import json
import math
import shutil
from pathlib import Path

LAB = Path('/home/johnny/Projects/demo/cascade-lab')
H2 = LAB / 'HERMES_AUDIT_20261007/h2-owner'
RUNS = ['b29-ramp-live1', 'b29-ramp-live2']
WT = LAB / 'HERMES_BACKLOG_20261007/h2-turn-ramp/cascade'
EV = WT / 'docs/evidence/h2-turn-ramp-live-20261008'
EV.mkdir(parents=True, exist_ok=True)


def yaw(s):
    w, x, y, z = s['orientation_wxyz']
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def arm_of(block):
    return 'MCP, ramp on' if block == 'M' else ('ramp on' if block.startswith('R') else 'ramp off')


rows, profiles, meta = [], {}, {}
for run in RUNS:
    rd = H2 / 'runs' / run
    rep = json.loads((rd / 'ramp.json').read_text())
    launch = json.loads((rd / 'launch.json').read_text())
    meta[run] = {'hello_capabilities': rep['hello'].get('capabilities'), 'model_identity_sha256': rep.get('model_identity_sha256'),
                 'velocity_scaling_advertised': rep.get('velocity_scaling_advertised'), 'arms_check': rep.get('arms'),
                 'plan': rep.get('plan'), 'reset_on_latch': rep.get('reset_on_latch', False), 'rt_factor': rep.get('rt_factor'),
                 'owner_argv_flags': [a for a in launch['argv'] if a.startswith('--')], 'gpu': launch.get('gpu'),
                 'owner_receipt': rep.get('owner_receipt'), 'loadavg_start': rep.get('loadavg_start'),
                 'per_turn_resets': rep.get('per_turn_resets', [])}
    latched_by_tag = {r['tag']: r.get('stop_latched') for b in rep['blocks'].values() for r in b}
    for f in sorted(rd.glob('turn-*.json')):
        tag = f.stem[5:]
        d = json.loads(f.read_text())
        post = d.get('postcondition') or {}
        m = post.get('metrics') or {}
        ack = d.get('ack') or {}
        start = ack.get('start_sim_time_s')
        req = d.get('requested_angle_rad')
        S = [s for s in (d.get('measured') or {}).get('samples') or [] if s.get('sim_time_s') is not None]
        tol = (post.get('limits') or {}).get('rotation_tolerance_rad') or 0.05
        t_goal, acc, prev = None, 0.0, None
        for s in S:
            yv = yaw(s)
            if prev is not None:
                acc += (yv - prev + math.pi) % (2 * math.pi) - math.pi
            prev = yv
            if t_goal is None and req is not None and abs(req - acc) <= tol:
                t_goal = s['sim_time_s']
        sign = 1 if (req or 0) > 0 else -1
        wins = collections.OrderedDict()
        for s in S:
            if start is None or s['sim_time_s'] < start:
                continue
            k = round(math.floor((s['sim_time_s'] - start) / 0.25) * 0.25, 2)
            if k > 3.25:
                continue
            wins.setdefault(k, []).append(sign * s['angular_velocity_body'][2])
        prof = {f'{k:.2f}': round(sum(v) / len(v), 3) for k, v in wins.items()}
        upd = [{'t_s': round(u['sim_time_s'] - start, 3), 'remaining_rad': round(abs(u['remaining_rad']), 4),
                'rate_rad_s': round(u['rate_rad_s'], 3)} for u in d.get('turn_rate_updates') or []] if start is not None else []
        # stall: lowest 0.25 s mean between 0.75 and 2.0 s that ends >= 0.25 s before the goal (so the ramp's
        # intended final deceleration is not counted); outcome: goal reached within the command or not
        goal_rel = None if t_goal is None or start is None else t_goal - start
        mid = {float(k): v for k, v in prof.items()
               if 0.75 <= float(k) <= 2.0 and (goal_rel is None or float(k) <= goal_rel - 0.5)}
        stall_t = min(mid, key=mid.get) if mid and min(mid.values()) < 0.2 else None
        later = [v for k, v in prof.items() if stall_t is not None and float(k) > stall_t]
        cmd_at_stall = None
        if stall_t is not None and upd:
            before = [u['rate_rad_s'] for u in upd if u['t_s'] <= stall_t + 0.25]
            cmd_at_stall = before[-1] if before else abs((d.get('command') or {}).get('wz') or 0.5)
        elif stall_t is not None:
            cmd_at_stall = abs((d.get('command') or {}).get('wz') or 0.5)
        err = d.get('error') or ''
        row = {'run': run[-1], 'tag': tag, 'block': tag.split('-')[0], 'arm': arm_of(tag.split('-')[0]),
               'angle_rad': req, 'status': post.get('status'), 'reason': post.get('reason'), 'error': err or None,
               'independent': 'latched' not in err,
               'measured_angle_rad': None if d.get('measured_angle_rad') is None else round(d['measured_angle_rad'], 4),
               'translation_path_m': None if d.get('measured_translation_path_m') is None else round(d['measured_translation_path_m'], 4),
               'goal_reached_after_s': None if t_goal is None or start is None else round(t_goal - start, 3),
               'commanded_rate_at_stop_rad_s': d.get('commanded_rate_at_stop_rad_s'), 'ramp_updates': len(upd),
               'settle_drift_rad': None if 'settle_drift_rad' not in m else round(m['settle_drift_rad'], 4),
               'settle_max_angular_speed_rad_s': None if 'settle_max_angular_speed_rad_s' not in m else round(m['settle_max_angular_speed_rad_s'], 3),
               'verifier_path_length_m': None if 'path_length_m' not in m else round(m['path_length_m'], 4),
               'stall_window_s': stall_t, 'stall_yaw_rate_rad_s': None if stall_t is None else mid[stall_t],
               'commanded_rate_during_stall_rad_s': cmd_at_stall,
               'goal_reached_in_command': goal_rel is not None and goal_rel <= 3.0 and 'deadline' not in err,
               'stop_latched': latched_by_tag.get(tag)}
        rows.append(row)
        profiles[f'{run[-1]}:{tag}'] = {'yaw_rate_toward_goal_per_0p25s': prof, 'rate_updates': upd}

(EV / 'turns.json').write_text(json.dumps(rows, indent=1) + '\n')
(EV / 'yaw_rate_profiles.json').write_text(json.dumps(profiles, indent=1) + '\n')
(EV / 'run_meta.json').write_text(json.dumps(meta, indent=1, default=str) + '\n')

hd = EV / 'harness'
hd.mkdir(exist_ok=True)
for name in ('h2_ramp.py', 'analyse_ramp.py', 'h2_episode.py', 'h2_geometric.py'):
    shutil.copy(H2 / name, hd / name)
shutil.copy(Path(__file__), hd / 'make_live_evidence.py')

raw = []
for run in RUNS:
    rd = H2 / 'runs' / run
    files = sorted(rd.glob('turn-*.json')) + [rd / n for n in ('ramp.json', 'launch.json', 'owner.log', 'analysis.jsonl',
                                                                'resolved-config-ramp.json', 'resolved-config-noramp.json')]
    files += sorted((rd / 'kit').glob('*.json')) + sorted((rd / 'kit').glob('*.jsonl'))
    files.append(H2 / f'runs_ramp{run[-1]}.log')
    for f in files:
        if f.exists() and f.is_file():
            raw.append({'path': str(f.relative_to(LAB.parent)), 'bytes': f.stat().st_size,
                        'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
(EV / 'manifest.json').write_text(json.dumps({'raw': raw, 'note': 'raw rows stay in cascade-lab/; this record is derived from them'}, indent=1) + '\n')

# summaries for the REPORT
ind = [r for r in rows if r['independent']]
cells = collections.defaultdict(list)
for r in ind:
    cells[(r['arm'], abs(r['angle_rad']))].append(r)
summary = {}
for (arm, ang), v in sorted(cells.items()):
    summary[f'{arm} {ang:.1f}'] = {'n': len(v), 'confirmed': sum(r['status'] == 'confirmed' for r in v),
                                    'refuted': sum(r['status'] == 'refuted' for r in v),
                                    'timeout': sum(bool(r['error']) and 'deadline' in r['error'] for r in v),
                                    'settle_drift_rad': [r['settle_drift_rad'] for r in v]}
stalls = [r for r in ind if r['stall_yaw_rate_rad_s'] is not None]
stall_tab = collections.defaultdict(lambda: [0, 0])
for r in stalls:
    key = 'commanded >= 0.45' if (r['commanded_rate_during_stall_rad_s'] or 0) >= 0.45 else 'commanded < 0.45 (ramp already lowering)'
    stall_tab[key][0] += 1
    stall_tab[key][1] += bool(r['goal_reached_in_command'])
(EV / 'summary.json').write_text(json.dumps({'cells': summary, 'excluded': [f"{r['run']}:{r['tag']}" for r in rows if not r['independent']],
                                             'stall_then_goal_in_command': {k: {'stalled_turns': a, 'goal_reached_in_command': b} for k, (a, b) in stall_tab.items()},
                                             'stall_count': len(stalls), 'independent_turns': len(ind)}, indent=1) + '\n')
print(json.dumps({'cells': summary, 'stall_then_goal': dict(stall_tab), 'stalls': len(stalls), 'independent': len(ind),
                  'excluded': [f"{r['run']}:{r['tag']}" for r in rows if not r['independent']], 'raw_files': len(raw)}))
for r in stalls:
    print(r['run'], r['tag'], r['status'], 'stall@', r['stall_window_s'], r['stall_yaw_rate_rad_s'], 'cmd', r['commanded_rate_during_stall_rad_s'], 'goal_in_cmd', r['goal_reached_in_command'], 'goal@', r['goal_reached_after_s'])
print('timeouts:', [(r['run'], r['tag'], r['stall_window_s'], r['commanded_rate_during_stall_rad_s']) for r in ind if r['error'] and 'deadline' in r['error']])
