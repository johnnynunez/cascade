"""Actual CPU contact dynamics, including the occupied-destination refusal."""
import numpy as np
import pytest
from test_memory_task_mujoco import needs_gl, needs_pin
from test_memory_task_mujoco import (
    two_prop_runtime as two_prop_runtime,  # noqa: PLC0414
)

from cascade.types import SafetyViolation


@needs_pin
@needs_gl
def test_withdrawal_home_preserves_first_cube_and_occupied_target_refuses_before_carry(two_prop_runtime):
    cfg, runtime, arm = two_prop_runtime
    first = runtime.execute('pick_and_place', {'object': 'red cube'})
    assert first['ok'] and first['postcondition']['status'] == 'confirmed', first
    assert first['postcondition']['channel'] == 'physics', first
    assert first['return_home']['ok'] and first['post_place_retreat']['ok'], first
    placed = np.asarray(arm.world.body_pos('red_cube'))
    assert np.linalg.norm(placed[:2]-cfg.grasp.get('drop_zone')) < .06
    second = runtime.execute('pick_and_place', {'object': 'blue cube'})
    assert second['ok'] is False and second['stage'] == 'retreat_plan', second
    assert 'planned carry intersects another physical object' in second['error'], second
    assert second['holding'] == 'blue cube' and second['home_skipped'] is True
    np.testing.assert_allclose(arm.world.body_pos('red_cube'), placed, atol=1e-6, rtol=0)


@needs_pin
@needs_gl
def test_explicit_reset_replans_new_generation_but_retains_debt_until_physical_reset_and_observation(
        two_prop_runtime, monkeypatch):
    cfg, runtime, arm = two_prop_runtime
    cfg.grasp._data['home_after_place'] = False
    spawn = {name: arm.world.body_pos(name) for name in arm.world.free_body_names()}
    result = runtime.execute('pick_and_place', {'object': 'red cube'})
    assert result['ok'] and not result['return_home']['attempted'], result
    pending = runtime._mujoco_withdrawals[id(runtime.arm)]
    old_generation = pending.generation
    runtime.arm.harness.halt('cancel the old release context')
    runtime.arm.harness.clear_halt()  # Explicit operator release of the soft stop.
    with pytest.raises(SafetyViolation):
        runtime.skill_move_home()
    other_arm = object()
    runtime._mujoco_withdrawals[id(other_arm)] = other_arm
    frames = runtime._reset_camera_frames
    def no_frame(*args, **kwargs):
        raise TimeoutError('synthetic camera outage after real physical reset')
    monkeypatch.setattr(runtime, '_reset_camera_frames', no_frame)
    failed = runtime.execute('reset_scene', {})
    assert not failed['ok'] and failed['withdrawal_reset_verification']['at_model_spawn'], failed
    assert runtime._mujoco_withdrawals[id(runtime.arm)] is pending
    assert pending.generation == old_generation
    def cancelled_frame(*args, **kwargs):
        observed = frames(*args, **kwargs)
        runtime.arm.harness.halt('stop delivered after reset observation')
        return observed
    monkeypatch.setattr(runtime, '_reset_camera_frames', cancelled_frame)
    cancelled = runtime.execute('reset_scene', {})
    assert not cancelled['ok'] and cancelled['observation_refreshed'], cancelled
    assert runtime._mujoco_withdrawals[id(runtime.arm)] is pending
    assert runtime.arm.harness._pending_model_withdrawal is pending
    runtime.arm.harness.clear_halt()
    monkeypatch.setattr(runtime, '_reset_camera_frames', frames)
    completed = runtime.execute('reset_scene', {})
    assert completed['ok'] and completed['observation_refreshed'], completed
    assert completed['withdrawal_recovery']['generation'] == old_generation+2
    assert id(runtime.arm) not in runtime._mujoco_withdrawals
    assert runtime.arm.harness._pending_model_withdrawal is None
    assert runtime._mujoco_withdrawals[id(other_arm)] is other_arm
    for name, position in spawn.items():
        np.testing.assert_allclose(arm.world.body_pos(name), position, atol=1e-12, rtol=0)
