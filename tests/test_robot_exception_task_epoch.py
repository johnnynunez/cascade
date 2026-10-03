"""Late exception accounting cannot attach an old motion to a new task."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from cascade.robotics.runtime import RobotRuntime
from test_robot_runtime import Domain


class _ReleasedAdmission:
    """Pause only the failing worker after its real admission lock is released."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = runtime._gate
        self.worker = None
        self.saw_active = False
        self.released = threading.Event()
        self.resume = threading.Event()

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, *exc):
        worker = threading.get_ident() == self.worker
        if worker and self.runtime._active:
            self.saw_active = True
        pause = worker and self.saw_active and not self.runtime._active
        if pause:
            self.saw_active = False
        self.lock.release()
        if pause:
            self.released.set()
            assert self.resume.wait(3), "test host did not finish the task handoff"


@pytest.mark.parametrize("next_task", [False, True])
def test_exception_debt_stays_with_the_admitted_task(next_task):
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    gate = _ReleasedAdmission(runtime)
    runtime._gate = gate

    def fail(name, args):
        domain.calls.append((name, args))
        raise RuntimeError("admitted motion failed")

    domain.execute = fail

    def dispatch():
        gate.worker = threading.get_ident()
        return runtime.execute("locomotion.move", {"distance": .1})

    original_task = runtime._task_id
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(dispatch)
            try:
                assert gate.released.wait(3), "motion did not release admission"
                assert runtime.unverified_actions() == ["locomotion.move"]
                if next_task:
                    runtime.begin_task()
                    assert runtime._task_id != original_task
                    assert runtime.unverified_actions() == []
            finally:
                gate.resume.set()
            result = pending.result(3)
        assert not result["ok"] and "admitted motion failed" in result["error"]
        assert len(domain.calls) == 1
        assert runtime.unverified_actions() == ([] if next_task else ["locomotion.move"])
        report = runtime.execute("task_done", {"success": True, "summary": "done"})
        assert report["success"] is next_task
    finally:
        gate.resume.set()
        assert runtime.close()["ok"]


def test_invalid_motion_arguments_still_block_the_current_task():
    domain = Domain()
    runtime = RobotRuntime({domain.domain_id: domain})
    try:
        result = runtime.execute("locomotion.move", {})
        assert not result["ok"] and domain.calls == []
        assert runtime.unverified_actions() == ["locomotion.move"]
        assert not runtime.execute("task_done", {"success": True, "summary": "done"})["success"]
    finally:
        assert runtime.close()["ok"]
