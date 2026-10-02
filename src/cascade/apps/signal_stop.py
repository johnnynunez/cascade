"""Scoped signal-to-unwind adapter for the main-thread entrypoints.

Handlers ONLY publish Python scalar state and raise a BaseException. In
particular Event.set(), Queue.put(), logging and robot/transport methods are
not signal-safe: they can re-enter a lock held by the interrupted thread.
The caller catches SignalRequest after its critical sections have unwound,
invalidates work, and only then logs/cleans up. No helper thread or fd is owned.
"""
from contextlib import contextmanager
import signal


class SignalRequest(BaseException):
    def __init__(self, signum):
        self.signum = signum


class StopSignals:
    """First INT requests a stop; TERM requests exit; repeated INT forces exit.

    ``defer`` is for passive construction/cleanup only, never execution.
    Call ``checkpoint`` immediately after construction, before dispatch. This
    lets resource owners finish publishing their handles before unwinding.
    A second INT still interrupts a stuck constructor/teardown, as before.
    """

    def __init__(self, *, staff_reset=False, protect_registration=False):
        self.signum = None
        self._pending = None
        self._deferred = 0
        self._interrupt_seen = False
        self._signals = [signal.SIGINT, signal.SIGTERM]
        if staff_reset and hasattr(signal, "SIGUSR1"):
            self._signals.append(signal.SIGUSR1)
        self._previous = {}
        self._protect_registration = protect_registration
        self._setter = signal.signal
        self.registration_attempts = 0

    def _handle(self, signum, _frame):
        # Do not call ANY application code here, including Event.set().
        if signum == signal.SIGINT:
            if self._interrupt_seen:
                raise KeyboardInterrupt
            self._interrupt_seen = True
        elif signum == signal.SIGTERM and self.signum == signal.SIGTERM:
            raise KeyboardInterrupt
        if signum in (signal.SIGINT, signal.SIGTERM):
            self.signum = signum
        elif self._pending in (signal.SIGINT, signal.SIGTERM):
            # A staff reset cannot erase a deferred shutdown before its stop
            # has even run. Staff may explicitly reset again afterwards.
            return
        self._pending = signum
        if not self._deferred:
            self._pending = None
            raise SignalRequest(signum)

    def __enter__(self):
        self._setter = signal.signal
        for signum in self._signals:
            self._previous[signum] = self._setter(signum, self._handle)
        if self._protect_registration:
            # SimulationApp registers its own immediate sys.exit(0) SIGINT
            # callback DURING construction. Reinstalling ours afterwards is
            # too late for an interrupt inside that constructor. Scope this
            # Python registration guard to the owned native CLI process only;
            # the SDK installation and default demo/MCP behavior stay untouched.
            signal.signal = self._register
        return self

    def _register(self, signum, handler):
        if signum not in self._signals:
            return self._setter(signum, handler)
        import threading
        if threading.current_thread() is not threading.main_thread():
            raise ValueError('signal only works in main thread of the main interpreter')
        if handler not in (signal.SIG_DFL, signal.SIG_IGN) and not callable(handler):
            raise TypeError('signal handler must be SIG_IGN, SIG_DFL, or a callable object')
        self.registration_attempts += 1
        previous = signal.getsignal(signum)
        # Reinstall the Python dispatcher too: native Kit startup may have
        # changed the OS disposition behind signal.getsignal's Python cache.
        self._setter(signum, self._handle)
        return previous

    def __exit__(self, *_exc):
        if self._protect_registration:
            signal.signal = self._setter
        for signum, previous in self._previous.items():
            self._setter(signum, previous)

    @contextmanager
    def defer(self):
        self._deferred += 1
        try:
            yield
        finally:
            self._deferred -= 1

    def checkpoint(self):
        if self._pending is not None:
            signum, self._pending = self._pending, None
            raise SignalRequest(signum)
