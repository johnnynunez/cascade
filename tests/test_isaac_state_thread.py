"""SDK articulation reads belong to the simulator thread, never socket workers."""
import socketserver
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import load_isaac_bridge_definitions


@pytest.mark.parametrize("fail", [False, True])
def test_state_dispatch_runs_sdk_read_on_the_main_loop(fail):
    main = threading.current_thread()
    reads = []
    def positions():
        assert threading.current_thread() is main
        reads.append(True)
        if fail:
            raise RuntimeError("stale articulation")
        return SimpleNamespace(numpy=lambda: np.array([[.1, .2]]))
    env = {"threading":threading,"socketserver":socketserver,"np":np,"_exec_lock":threading.Lock(),
        "_exec_jobs":[],"ARM_IDX":[0],"_grip_frac_now":lambda q:float(q[1]),
        "engine":"physx", "_motion_clock_epoch":"test-epoch",
        "args":SimpleNamespace(prim="/robot", dt=1/120),
        "SimulationManager":SimpleNamespace(get_simulation_time=lambda: 1.,
            get_num_physics_steps=lambda: 120),
        "art":SimpleNamespace(get_dof_positions=positions,
            get_dof_velocities=lambda:SimpleNamespace(numpy=lambda:np.zeros((1,2))))}
    load_isaac_bridge_definitions({"Handler", "_run_exec_jobs", "_motion_clock_snapshot"}, env)
    handler=object.__new__(env["Handler"]);result={}
    worker=threading.Thread(target=lambda:result.update(handler._dispatch({"op":"state"})))
    worker.start()
    end=time.monotonic()+1
    while not env["_exec_jobs"] and time.monotonic()<end:time.sleep(.005)
    assert env["_exec_jobs"] and not reads
    env["_run_exec_jobs"]();worker.join(timeout=1)
    assert not worker.is_alive() and reads
    assert result["ok"] is (not fail)
    if fail:
        assert "stale articulation" in result["error"]
    else:
        assert result["q"] == [.1] and result["gripper_pos"] == .2
        assert result["physics_clock"]["physics_step"] == 120
        assert result["physics_clock"]["sim_time"] == 1.


def test_newton_motion_clock_uses_authoritative_stage(monkeypatch):
    import sys
    current = SimpleNamespace(initialized=True, sim_time=3.5, simulation_step_count=420)
    monkeypatch.setitem(sys.modules, 'isaacsim.physics.newton',
                        SimpleNamespace(acquire_stage=lambda: current))
    env = {'engine':'newton', '_motion_clock_epoch':'epoch',
           'args':SimpleNamespace(prim='/robot', dt=1/120)}
    load_isaac_bridge_definitions({'_motion_clock_snapshot'}, env)
    value = env['_motion_clock_snapshot']()
    assert value == dict(version=1, engine='newton', clock='newton_stage',
                         epoch='epoch', robot_id='/robot', sim_time=3.5,
                         physics_step=420, physics_dt_s=1/120)
    current.initialized = False
    with pytest.raises(RuntimeError, match='clock unavailable'):
        env['_motion_clock_snapshot']()
