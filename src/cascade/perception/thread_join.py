"""Stop worker threads completely before the interpreter finalizes.

Why this module exists (measured on the kitchen launcher). The launcher's
runtime check aborted with ``terminate called without an active exception``
(also seen: ``free(): invalid size``) in 3 of 20 kitchen runs, and in 3 of 3
with a cold CUDA kernel cache. ``WorldWatcher.stop()`` joined its daemon
thread with ``timeout=5`` and returned while the first YOLOE inference (more
than 5 s while CUDA JIT-compiles its kernels) was still running. The main
thread went on to interpreter exit. When the daemon thread came back from its
GIL-released torch op, CPython 3.12 ``pthread_exit``-ed it from inside
pybind11's ``gil_scoped_release`` destructor, and the forced unwind through
that noexcept destructor called ``std::terminate``. gdb, same run: main thread
in ``exit() -> ~TorchLibraryInit``; watcher thread in ``take_gil ->
PyThread_exit_thread -> _Unwind_ForcedUnwind -> std::terminate``.

Native code cannot be cancelled from Python, so the only safe stop is to WAIT
until the thread has ended. A join with a timeout is never enough: any bound
can be exceeded by a cold GPU, a slow bridge or a loaded box, and the
failure then shows up far away, as an abort at exit. ``join_thread`` waits
with no bound by default and says where the thread is every
``warn_every_s``, so a slow or hung tick is visible instead of silent.

``stop_before_exit`` is the safety net for entry points that never call
``stop()``: an ``atexit`` hook runs while the interpreter is still intact,
before daemon threads are killed, and joins the thread there.
"""

from __future__ import annotations

import atexit
import sys
import threading
import time
import weakref

#: Upper bound for the atexit safety net only (see stop_before_exit). The
#: slowest tick measured is the first cold YOLOE inference (> 5 s with the
#: CUDA kernel cache disabled); a minute leaves a wide margin.
EXIT_JOIN_S = 60.0


def thread_location(thread: threading.Thread) -> str:
    """`file:line in function` of the thread's innermost Python frame."""
    frame = sys._current_frames().get(thread.ident) if thread.ident is not None else None
    if frame is None:
        return "no Python frame (native code or exiting)"
    return f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"


def join_thread(
    thread: threading.Thread | None,
    *,
    what: str,
    timeout_s: float | None = None,
    warn_every_s: float = 10.0,
) -> bool:
    """Wait until `thread` has ended. True once it has.

    `timeout_s=None` (the default) never gives up. With a bound, returns
    False when it expires and the thread is still alive: the caller must
    keep its handle, because the thread will run native code at exit if
    nothing joins it later. Joining the calling thread itself is impossible
    and returns False as well.
    """
    if thread is None or not thread.is_alive():
        return True
    if thread is threading.current_thread():
        return False
    t0 = time.monotonic()
    next_warn = t0 + max(float(warn_every_s), 0.1)
    while True:
        now = time.monotonic()
        wait = next_warn - now
        if timeout_s is not None:
            wait = min(wait, t0 + float(timeout_s) - now)
        thread.join(timeout=max(wait, 0.0))
        if not thread.is_alive():
            return True
        now = time.monotonic()
        if timeout_s is not None and now - t0 >= float(timeout_s):
            print(f"[{what}] thread {thread.name!r} still running after "
                  f"{now - t0:.1f}s: {thread_location(thread)}", file=sys.stderr)
            return False
        if now >= next_warn:
            print(f"[{what}] waiting for thread {thread.name!r} to finish its "
                  f"in-flight work ({now - t0:.0f}s): {thread_location(thread)}",
                  file=sys.stderr)
            next_warn = now + max(float(warn_every_s), 0.1)


def stop_before_exit(owner, method: str = "stop", *, timeout_s: float = EXIT_JOIN_S):
    """Call `owner.<method>(timeout_s=...)` at interpreter exit unless cancelled.

    atexit callbacks run after non-daemon threads are joined but BEFORE the
    interpreter starts finalizing, so a daemon thread can still finish its
    tick there. The wait is bounded (`EXIT_JOIN_S`) only so that a thread
    stuck for good cannot hang process exit forever; it is logged if it
    expires. Holds only a weak reference, so it never keeps `owner` alive.
    Returns the registered hook; pass it to `cancel_stop_before_exit` once
    the owner has stopped on its own.
    """
    ref = weakref.ref(owner)

    def _stop_at_exit() -> None:
        target = ref()
        if target is None:
            return
        try:
            getattr(target, method)(timeout_s=timeout_s)
        except Exception as e:  # noqa: BLE001 -- exit must go on
            print(f"[exit] {type(target).__name__}.{method}() failed: {e}",
                  file=sys.stderr)

    atexit.register(_stop_at_exit)
    return _stop_at_exit


def cancel_stop_before_exit(hook) -> None:
    if hook is not None:
        atexit.unregister(hook)
