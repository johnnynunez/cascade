"""Observed-turn admission regressions; synthetic feedback proves no gait."""
from dataclasses import replace
import math

import pytest

from cascade.control.mock_base import MockMobileBase
from cascade.safety.base_harness import SafeBase
from test_mobile_safety import limits


@pytest.mark.parametrize('angle', [.2, -.2])
@pytest.mark.parametrize('when', ['delivery', 'after_expiry'])
def test_turn_cannot_complete_from_delivery_drift_or_late_rotation(angle, when):
    class RecordedTurn(MockMobileBase):
        ack = None

        def command_velocity(self, command, *, generation):
            self.ack = super().command_velocity(command, generation=generation)
            return self.ack

        def get_state(self):
            self.advance(.03)
            state = super().get_state()
            rotated = self.ack is not None and (
                when == 'delivery' or state.sim_time_s > self.ack['end_sim_time_s'])
            yaw = angle if rotated else 0.
            return replace(state, orientation_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2)),
                           angular_velocity_body=(0., 0., 0.))

    raw = RecordedTurn(wall_lease_s=2., auto_step=False)
    safe = SafeBase(raw, limits(max_duration_s=.1))
    safe.connect()
    try:
        result = safe.turn(angle)
        assert not result['execution_ok']
        assert 'before simulation deadline' in result['error']
        assert result['requested_angle_rad'] == angle
        assert result['measured_angle_rad'] == 0.
        # Retain the late observed pose even though it supplies no progress.
        assert result['measured']['after']['sim_time_s'] > raw.ack['end_sim_time_s']
        assert abs(SafeBase._yaw(raw.get_state())-angle) < 1e-12
        assert raw.get_state().latched
    finally:
        safe.disconnect()
