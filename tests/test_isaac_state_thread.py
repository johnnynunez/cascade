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
        "art":SimpleNamespace(get_dof_positions=positions,
            get_dof_velocities=lambda:SimpleNamespace(numpy=lambda:np.zeros((1,2))))}
    load_isaac_bridge_definitions({"Handler", "_run_exec_jobs"}, env)
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
