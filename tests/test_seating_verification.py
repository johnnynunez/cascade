"""Counterexamples to inferring a seat from stalled rotation and motor torque."""
import unittest

from cascade.sim.seating_verification import verify_factory_seating


def observations():
    rows = []
    for step in range(181):
        rows.append({'epoch':'one-run', 'fastener_id':'nut', 'fixture_id':'bolt',
            'step':step, 'time_s':step/60, 'fixture_position_m':[.24,0,0],
            'fixture_quaternion_xyzw':[0,0,0,1], 'nut_bottom_gap_m':.00001,
            'motor_torque_nm':.05 if step <= 60 else 0., 'nut_angular_speed_rad_s':0.,
            'maximum_nut_angular_speed_rad_s':3., 'tool_contacts':4, 'tool_fixture_contacts':0,
            'tool_fixture_interference_seen':False,
            'shoulder_contact_records':[{'position_m':[.252,0,.023],
                'normal':[0,0,1], 'normal_force_n':12.}]})
    return rows


class SeatingVerificationTests(unittest.TestCase):
    def verify(self, rows):
        return verify_factory_seating(rows, {'threading_verified':True}, motor_off_at=1.)

    def test_seat_needs_shoulder_torque_then_zero_motor_rest(self):
        self.assertTrue(self.verify(observations())['seating_verified'])

    def test_thread_friction_stall_is_not_shoulder_contact(self):
        rows = observations()
        for row in rows: row['shoulder_contact_records'] = []
        self.assertFalse(self.verify(rows)['seating_verified'])

    def test_thread_contact_above_shoulder_does_not_count(self):
        rows = observations()
        for row in rows: row['shoulder_contact_records'][0]['position_m'][2] = .025
        self.assertFalse(self.verify(rows)['seating_verified'])

    def test_radial_thread_normal_does_not_count(self):
        rows = observations()
        for row in rows: row['shoulder_contact_records'][0]['normal'] = [1,0,0]
        self.assertFalse(self.verify(rows)['seating_verified'])

    def test_transient_tool_bottoming_remains_refuted(self):
        rows = observations()
        rows[25]['tool_fixture_interference_seen'] = True
        self.assertFalse(self.verify(rows)['seating_verified'])

    def test_brake_during_rest_is_not_zero_motor(self):
        rows = observations()
        for row in rows[61:]: row['motor_torque_nm'] = -.001
        self.assertFalse(self.verify(rows)['seating_verified'])

    def test_nut_backing_off_during_rest_is_not_retained_seat(self):
        rows = observations()
        rows[-1]['shoulder_contact_records'] = []
        self.assertFalse(self.verify(rows)['seating_verified'])

    def test_epoch_reset_cannot_splice_two_runs(self):
        rows = observations()
        rows[-1]['epoch'] = 'second-run'
        self.assertEqual(self.verify(rows)['status'], 'unverified')

    def test_moving_fixture_cannot_fake_retention(self):
        rows = observations()
        rows[-1]['fixture_position_m'] = [.24,0,.01]
        self.assertFalse(self.verify(rows)['seating_verified'])

    def test_missing_observation_cannot_hide_motion(self):
        rows = observations()
        del rows[80]
        self.assertEqual(self.verify(rows)['status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
