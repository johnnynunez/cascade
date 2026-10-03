"""A diagnostic formatter must not make a failed owner look cleanly closed."""
from concurrent.futures import Future

import pytest

from cascade.control.fastening import FasteningFault
from cascade.sim.factory_owner import _Request
from test_factory_owner import owner_fixture


class UnformattableError(RuntimeError):
    def __str__(self):
        raise RuntimeError("exception formatter failed")


class UnformattableSignal(BaseException):
    def __str__(self):
        raise KeyboardInterrupt("exception formatter interrupted")


class UnformattableText(str):
    def __format__(self, _spec):
        raise KeyboardInterrupt("string subclass formatter interrupted")


class ErrorWithUnformattableText(Exception):
    def __str__(self):
        return UnformattableText("original backend fault")


@pytest.mark.parametrize("error_type", [RuntimeError, UnformattableError, UnformattableSignal,
                                       ErrorWithUnformattableText])
@pytest.mark.parametrize("phase", ["enter", "advance", "final_zero"])
def test_owner_fault_survives_exception_formatting_and_finishes_cleanup(phase, error_type):
    now, backend, owner = owner_fixture()
    error = error_type("original backend fault")
    pending = Future()
    ticket = _Request("turn", {}, owner.controller.generation, now[0] + 1., pending)

    def fail(*_args):
        # This request arrives while the backend call is in progress. The
        # owner's finally must reject it even when zero upload also fails.
        owner._requests.put_nowait(ticket)
        raise error

    if phase == "enter":
        backend.enter_owner = fail
    elif phase == "advance":
        backend.prepare_hook = fail
    else:
        # A completed cycle can leave pending callers when shutdown starts.
        # Use the actual owner exit condition; no native or clock replacement.
        backend.solve_hook = owner._exit.set
        backend.upload_zero = fail
    owner._start = now[0]
    escaped = None
    try:
        owner._run()
    except BaseException as exc:
        escaped = exc

    closure = owner.close()
    assert closure["ok"] is False, closure
    assert escaped is None, "diagnostic formatting escaped the owner's containment"
    assert closure["owner_thread_closed"] is True
    assert closure["physical_stop_verified"] is False
    assert error_type.__name__ in closure["error"]
    assert owner._exit.is_set()
    assert owner.controller.guard.current_permit is None
    assert closure["zero_spindle"]["uploaded"] is (phase != "final_zero")
    if phase != "final_zero":
        assert backend.effort == 0.
    with pytest.raises(FasteningFault, match=error_type.__name__):
        owner.journal.read()
    assert pending.done(), "pending admission was stranded by diagnostic formatting"
    with pytest.raises(FasteningFault, match="closed before admission"):
        pending.result()


def test_stop_is_latched_before_formatting_and_first_fault_survives_failed_zero():
    now, backend, owner = owner_fixture()
    formatted = []

    class PrimaryError(Exception):
        def __str__(self):
            formatted.append(owner.controller.guard.current_permit is None)
            raise RuntimeError("primary formatter failed")

    def enter():
        raise PrimaryError()

    def zero(_effort):
        raise UnformattableSignal()

    backend.enter_owner, backend.upload_zero = enter, zero
    owner._start = now[0]
    owner._run()
    result = owner.close()
    assert formatted == [True]
    assert result["ok"] is False
    assert result["error"] == "PrimaryError: exception message unavailable"
    assert result["zero_spindle"]["error"] == "UnformattableSignal: exception message unavailable"
    assert result["zero_spindle"]["uploaded"] is False
    with pytest.raises(FasteningFault, match="PrimaryError"):
        owner.journal.read()
