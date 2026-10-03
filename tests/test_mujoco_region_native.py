"""Small CPU journal qualification, not robot/task admission.

The synthetic model has two passive gravity-compensated hinges far from one
free cube and its plane. There are no actuator commands or renderer. The sole
step writer is the existing _MjcEngine method; the observer must add no step.
"""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from cascade.control.mujoco_arm import _MjcEngine
from cascade.sim.mujoco_placement import PlacementHistory
from test_mujoco_region import REGION


@pytest.fixture
def native_history():
    mj = pytest.importorskip('mujoco')
    model = mj.MjModel.from_xml_string('''<mujoco>
      <option timestep=".005" integrator="implicitfast"/>
      <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
        <body name="base"><body name="link" pos=".5 0 .2" gravcomp="1">
          <joint name="arm" type="hinge"/><geom type="sphere" size=".02"/>
          <body name="tool" pos=".06 0 0" gravcomp="1">
            <joint name="grip" type="hinge"/><geom type="sphere" size=".01"/>
          </body></body></body>
        <body name="cube" pos=".1975 -.1225 .0175"><freejoint/>
          <geom name="cube_geom" type="box" size=".0175 .0175 .0175" mass=".05"/>
        </body>
      </worldbody></mujoco>''')
    data = mj.MjData(model)
    world = SimpleNamespace(model=model, data=data, mj=mj, lock=threading.RLock(), path='synthetic-native-fixture')
    arm = SimpleNamespace(_cfg={'mj_delivery_area': REGION, 'mj_release_articulation_root': 'base',
        'mj_release_support_plane': {'geoms': ['floor'], 'optional_geoms': [],
             'position_m': [0., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.]}},
        _joint_names=['arm'], _grip_joint='grip', _grip_open=.6, _grip_closed=-.15,
        _qadr=[0], _dadr=[0], _grip_qadr=1)
    history = world.placement_history = PlacementHistory(world, arm)
    engine = SimpleNamespace(_world=world, _mj=mj, model=model, data=data, _viewer=None)
    return engine, history


def test_native_last_solve_phase_contact_sign_and_no_unrequested_step(native_history):
    engine, history = native_history
    for _ in range(12):
        _MjcEngine.step(engine, 12)
    assert history.error is None, history.error
    assert len(history.rows) == 12 and history.step == 144
    assert engine.data.time == pytest.approx(144*.005, abs=1e-12)
    row = history.rows[-1]
    assert row['last_force_interval_s'] == [row['time_s'], row['final_state_time_s']]
    assert row['final_state_time_s']-row['time_s'] == pytest.approx(.005)
    cube = row['objects']['cube']
    assert cube['support_contacts'] == 4 and cube['arm_active_contacts'] == 0
    assert cube['support_up_n'] == pytest.approx(.05*9.81, abs=1e-4)
    assert row['solved_contacts'] and row['native_contact_count'] == 4
    # A stable supported body does not fabricate a measured gripper opening.
    report, _ = history.read('cube')
    assert report['status'] == 'unverified' and 'opening' in report['evidence']
    before = (engine.data.time, engine.data.qpos.copy(), engine.data.qvel.copy())
    history.read('cube')
    assert engine.data.time == before[0]
    np.testing.assert_array_equal(engine.data.qpos, before[1])
    np.testing.assert_array_equal(engine.data.qvel, before[2])


def test_native_phase_boundary_retains_distinct_input_and_integrated_position(native_history):
    engine, history = native_history
    before = engine.data.qpos.copy()
    _MjcEngine.step(engine, 1)
    assert history.error is None, history.error
    row = history.rows[-1]
    np.testing.assert_array_equal(row['solve_qpos'], before)
    np.testing.assert_array_equal(row['final_qpos'], engine.data.qpos)
    assert row['solve_qpos'] != row['final_qpos']
    assert row['time_s'] == 0. and row['final_state_time_s'] == .005
    # Starting a new episode cannot keep support/release samples from the old.
    old = history.epoch
    history.reset()
    assert history.epoch != old and not history.rows
    with pytest.raises(ValueError, match='current native state'):
        history.read('cube')
