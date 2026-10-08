"""Analyse a b29 ramp live run: per turn, command time used, ramp profile, settle metrics vs the unchanged limits."""
import glob
import json
import math
import os
import sys

run = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(run, 'turn-*.json'))):
    tag = os.path.basename(f)[5:-5]
    d = json.load(open(f))
    post = d.get('postcondition') or {}
    m = post.get('metrics') or {}
    lim = post.get('limits') or {}
    ack = d.get('ack') or {}
    upd = d.get('turn_rate_updates') or []
    start, end = ack.get('start_sim_time_s'), ack.get('end_sim_time_s')
    samples = (d.get('measured') or {}).get('samples') or []
    req = d.get('requested_angle_rad') or 0.0
    tol = (post.get('limits') or {}).get('rotation_tolerance_rad') or 0.05

    def yaw(s):
        w, x, y, z = s['orientation_wxyz']
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    t_stop = None
    if samples:
        y0, prev, acc = yaw(samples[0]), None, 0.0
        for s in samples:
            yv = yaw(s)
            if prev is not None:
                dy = (yv - prev + math.pi) % (2 * math.pi) - math.pi
                acc += dy
            prev = yv
            if t_stop is None and abs(req - acc) <= tol and s.get('sim_time_s') is not None:
                t_stop = s['sim_time_s']
    pre_cut = [abs(s['angular_velocity_body'][2]) for s in samples
               if t_stop is not None and s.get('sim_time_s') is not None and t_stop - 0.3 <= s['sim_time_s'] <= t_stop]
    row = {
        'tag': tag, 'status': post.get('status'), 'reason': (post.get('reason') or '')[:60], 'error': d.get('error'),
        'requested': d.get('requested_angle_rad'), 'measured': d.get('measured_angle_rad'),
        'cmd_wz': (d.get('command') or {}).get('wz'), 'cmd_budget_s': (d.get('command') or {}).get('duration_s'),
        'cmd_window_sim_s': round(end - start, 3) if start is not None and end is not None else None,
        'goal_reached_after_sim_s': round(t_stop - start, 3) if t_stop is not None and start is not None else None,
        'ramp_updates': len(upd), 'first_update_sim_s': round(upd[0]['sim_time_s'] - start, 3) if upd and start else None,
        'rate_at_stop': d.get('commanded_rate_at_stop_rad_s'),
        'remaining_at_last_update': round(upd[-1]['remaining_rad'], 4) if upd else None,
        'yaw_rate_last_0p3s_mean': round(sum(pre_cut) / len(pre_cut), 3) if pre_cut else None,
    }
    for k in ('settle_drift_rad', 'settle_max_angular_speed_rad_s', 'settle_sim_duration_s', 'settle_veto_observations'):
        if k in m:
            row[k] = round(m[k], 4) if isinstance(m[k], float) else m[k]
    rows.append(row)
for r in rows:
    print(json.dumps(r, default=str))
