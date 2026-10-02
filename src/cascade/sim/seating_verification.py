"""Read-only seat check for the pinned Factory M20 fixture experiment.

This confirms geometric shoulder contact, illustrative motor effort and retained
motor-off rest. It deliberately makes no calibrated preload claim.
"""
from __future__ import annotations

import math

import numpy as np


def shoulder_force_n(row):
    """Sum only solved contacts on the head's horizontal support annulus."""
    center = np.asarray(row['fixture_position_m'], dtype=float)
    force = 0.
    for record in row['shoulder_contact_records']:
        point = np.asarray(record['position_m'], dtype=float)-center
        normal = np.asarray(record['normal'], dtype=float)
        effort = float(record['normal_force_n'])
        if (point.shape != (3,) or normal.shape != (3,) or not np.isfinite(point).all()
                or not np.isfinite(normal).all() or not math.isfinite(effort)):
            raise ValueError('invalid solved shoulder contact')
        radius = np.linalg.norm(point[:2])
        if (abs(point[2]-.023) <= .0003 and .011 <= radius <= .018
                and abs(normal[2]) >= .95 and abs(np.linalg.norm(normal)-1.) < .001
                and effort > 0):
            force += effort
    return force


def loaded_shoulder(row):
    return (shoulder_force_n(row) >= .1 and abs(row['nut_bottom_gap_m']) < .0003
            and .045 <= row['motor_torque_nm'] <= .050001
            and row['nut_angular_speed_rad_s'] < .08
            and row['tool_contacts'] > 0 and row['tool_fixture_contacts'] == 0)


def verify_factory_seating(rows, threading, motor_off_at):
    """Require half a second under torque, then two seconds of zero-motor rest.

    The producer also latches tool/fixture interference at every physics substep;
    last-frame contact snapshots alone cannot establish the absence of collision.
    """
    result = {'status':'unverified', 'seating_verified':False, 'preload_verified':False, 'checks':{}}
    try:
        if len(rows) < 3: raise ValueError('insufficient seating observations')
        first = rows[0]
        identity = tuple(first[k] for k in ('epoch','fastener_id','fixture_id'))
        previous = None
        for row in rows:
            if tuple(row[k] for k in ('epoch','fastener_id','fixture_id')) != identity:
                raise ValueError('seat body identity or epoch changed')
            if previous is not None and (row['step'] <= previous['step']
                    or not 0 < row['time_s']-previous['time_s'] <= .02):
                raise ValueError('seat sampling clock changed or has gaps')
            if not np.allclose(row['fixture_quaternion_xyzw'], [0,0,0,1], atol=1e-6, rtol=0):
                raise ValueError('Factory seat observer requires aligned fixed fixture')
            if not np.allclose(row['fixture_position_m'], first['fixture_position_m'], atol=1e-6, rtol=0):
                raise ValueError('Factory seat fixture moved')
            for key in ('time_s','nut_bottom_gap_m','motor_torque_nm','nut_angular_speed_rad_s'):
                if not math.isfinite(row[key]): raise ValueError(f'non-finite {key}')
            shoulder_force_n(row)
            previous = row
        loaded = [row for row in rows if motor_off_at is not None
                  and motor_off_at-.5-1e-8 <= row['time_s'] <= motor_off_at+1e-8]
        rest = [row for row in rows if motor_off_at is not None and row['time_s'] > motor_off_at+1e-8]
        stable_rest = [row for row in rest if row['time_s'] >= motor_off_at+1.]
        checks = {
            'threading': bool(threading.get('threading_verified')),
            'loaded_shoulder_contact_half_second': (bool(loaded)
                and loaded[-1]['time_s']-loaded[0]['time_s'] >= .5-1e-8
                and all(loaded_shoulder(row) for row in loaded)),
            'no_tool_fixture_contact': all(row['tool_fixture_contacts'] == 0
                and not row['tool_fixture_interference_seen'] for row in rows),
            'exact_zero_motor_after_stop': bool(rest) and all(row['motor_torque_nm'] == 0. for row in rest),
            'two_second_rest': bool(rest) and rest[-1]['time_s']-motor_off_at >= 2.-1e-8,
            'retained_shoulder_seat': bool(stable_rest) and all(shoulder_force_n(row) >= .1
                and abs(row['nut_bottom_gap_m']) < .0003 and row['nut_angular_speed_rad_s'] < .02
                for row in stable_rest),
            'sampling_speed_bound': all(row['maximum_nut_angular_speed_rad_s'] <= 10. for row in rows),
        }
        result.update(checks=checks, seating_verified=all(checks.values()),
                      status='confirmed' if all(checks.values()) else 'refuted',
                      loaded_seat_samples=len(loaded), retained_rest_samples=len(stable_rest))
    except (ValueError, TypeError, KeyError) as exc:
        result['reason'] = str(exc)
    return result
