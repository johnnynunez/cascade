"""The teardown never leaves a powered arm without its hold marker because a later wait hung (#435 review).

`shutdown_runtime` decided which arms stay powered and holding (a park that did not bring a
torque-cutting arm to rest) and wrote their markers only in its `arms` stage, AFTER it had waited for
every thread owner: the watcher, the dashboard, the viewer, the cameras. Those waits had no bound (by
design: a watcher inside a cold CUDA inference must finish before exit), so one that hung, or a SIGKILL
during it, left the arm raised and powered with no marker for the next process to read. Now:

- the markers are written right after the park decision, before the first owner wait;
- every owner wait is bounded (`demo.TEARDOWN_JOIN_S`, logged when it expires);
- the MCP server's wait for an in-flight runtime build (`_init_lock`) is bounded too
  (`mcp_server.SHUTDOWN_BUILD_WAIT_S`), so the receipt is always written.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest
from test_shutdown_unparked_hold import fast, feetech, runtime_for  # noqa: F401

from cascade.apps import demo, mcp_server
from cascade.apps.demo import shutdown_runtime
from cascade.control import arm_hold


class _Rig:
    """A camera rig double: iterable (owned_threads), closable with a bound."""

    def __init__(self, close):
        self.close = close

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 1


def _owners(runtime, record):
    """Every owner shutdown_runtime waits for, each recording (stage, timeout_s, marker seen then)."""
    key = demo._hold_key(runtime.arm.raw)

    def owner(stage):
        def stop(timeout_s=None):
            record.append((stage, timeout_s, arm_hold.read(key) if key else None))
        return stop

    runtime.drift_monitors = [SimpleNamespace(name="d0", stop=owner("drift_monitors"))]
    runtime.watcher = SimpleNamespace(stop=owner("watcher"))
    runtime.stream_server = SimpleNamespace(stop=lambda: record.append(("stream_server", None,
                                                                         arm_hold.read(key) if key else None)))
    runtime.viewer = SimpleNamespace(stop=owner("viewer"))
    runtime.rig = _Rig(owner("cameras"))
    return key


def test_the_hold_marker_is_written_before_the_teardown_waits_for_any_owner(fast):  # noqa: F811
    """An e-stopped so101 (every MCP SIGTERM latches first): its park is skipped and it stays powered.
    Main: the marker appeared only in the arms stage, after every owner wait; a wait that hung (or a
    SIGKILL during it) left the raised arm held with no marker."""
    events, record = [], []
    arm = feetech(events, fresh=True)
    arm.stop()
    rt = runtime_for(arm)
    key = _owners(rt, record)
    assert key and arm_hold.read(key) is None
    receipt = shutdown_runtime(rt, arm.raw)
    assert [stage for stage, _t, _m in record] == ["drift_monitors", "watcher", "stream_server", "viewer", "cameras"]
    for stage, _timeout, marker in record:
        assert marker is not None, f"no hold marker yet when the teardown waited for the {stage}"
        assert marker["reason"] == "the e-stop is latched, so the shutdown park never moved it", marker
    assert receipt["torque_held"] == {"0": "the e-stop is latched, so the shutdown park never moved it"}
    assert receipt["hold_markers"]["0"]["key"] == key
    assert [e for e in events if e[0] == "torque_off"] == [], "still held: no torque cut"


def test_a_hold_decision_that_fails_after_the_park_never_cuts_torque(fast, monkeypatch, capsys):  # noqa: F811
    """The decision right after the park raises (a broken marker store, a runtime that cannot say its
    arms): the arms stage decides again, and when it raises there too, nothing is disconnected, as on
    main. The e-stopped so101 keeps its torque."""
    events = []
    arm = feetech(events, fresh=True)
    arm.stop()
    calls = []

    def broken(*_a, **_k):
        calls.append(1)
        raise RuntimeError("cannot tell which arms are held")

    monkeypatch.setattr(demo, "_arms_to_hold", broken)
    receipt = shutdown_runtime(runtime_for(arm), arm.raw)
    assert len(calls) == 2, "decided after the park, then again before the disconnect"
    arms = next(s for s in receipt["stages"] if s["stage"] == "arms")
    assert arms["ok"] is False and "cannot tell which arms are held" in str(arms), arms
    assert [e for e in events if e[0] == "torque_off"] == [] and arm.raw._connected is True
    assert "retried before the disconnect" in capsys.readouterr().err


def test_every_owner_wait_of_the_teardown_is_bounded(fast):  # noqa: F811
    events, record = [], []
    arm = feetech(events, fresh=True)
    rt = runtime_for(arm)
    _owners(rt, record)
    shutdown_runtime(rt, arm.raw)
    bounded = {stage: timeout for stage, timeout, _m in record if stage != "stream_server"}
    assert bounded == dict.fromkeys(("drift_monitors", "watcher", "viewer", "cameras"), demo.TEARDOWN_JOIN_S)
    assert 0 < demo.TEARDOWN_JOIN_S < 120


def test_an_owner_past_its_bound_is_named_and_the_arms_still_disconnect(fast, monkeypatch, capsys):  # noqa: F811
    """A watcher whose tick never ends (a hung native call): the teardown waits TEARDOWN_JOIN_S, names it
    in the receipt, and the arms stage runs (a parked so101's torque off)."""
    monkeypatch.setattr(demo, "TEARDOWN_JOIN_S", 0.2)
    events = []
    arm = feetech(events, fresh=True)
    rt = runtime_for(arm)
    gate = threading.Event()
    tick = threading.Thread(target=gate.wait, daemon=True, name="world-watcher")
    tick.start()
    from cascade.perception.thread_join import join_thread

    rt.watcher = SimpleNamespace(_thread=tick, stop=lambda timeout_s=None: join_thread(tick, what="watcher",
                                                                                      timeout_s=timeout_s))
    try:
        t0 = time.monotonic()
        receipt = shutdown_runtime(rt, arm.raw)
        took = time.monotonic() - t0
    finally:
        gate.set()
        tick.join(5)
    assert took < 5, took
    assert receipt["pending_threads"] == [{"owner": "watcher", "thread": "world-watcher"}], receipt
    arms = next(s for s in receipt["stages"] if s["stage"] == "arms")
    assert arms["ok"] is True, arms
    assert len([e for e in events if e[0] == "torque_off"]) == 6, "parked, then its torque cut as always"
    assert "still running after" in capsys.readouterr().err


# ── the MCP server's wait for an in-flight build ─────────────────────────────────────────────────


def _shutdown_in_a_thread(server):
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("receipt", server.shutdown()), daemon=True)
    worker.start()
    return worker, out


def test_the_mcp_teardown_waits_for_a_runtime_build_only_so_long(monkeypatch, tmp_path):
    """A prewarm build that never ends (a camera open stuck in native code) holds `_init_lock`. Main: the
    SIGTERM teardown waited for it forever: no receipt, alive until SIGKILL. Now: at most
    SHUTDOWN_BUILD_WAIT_S, then the receipt says the runtime was not torn down."""
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(mcp_server, "SHUTDOWN_BUILD_WAIT_S", 0.3)
    server = mcp_server.McpSkillServer()
    building = threading.Event()
    release = threading.Event()

    def build():
        with server._init_lock:
            building.set()
            release.wait(30)

    builder = threading.Thread(target=build, daemon=True)
    builder.start()
    assert building.wait(5)
    try:
        worker, out = _shutdown_in_a_thread(server)
        worker.join(5)
        assert not worker.is_alive(), "the teardown still waits for the build"
    finally:
        release.set()
        builder.join(5)
        worker.join(5)
    receipt = out["receipt"]
    assert receipt["ok"] is False and receipt["runtime_built"] is False, receipt
    stage = next(s for s in receipt["stages"] if s["stage"] == "runtime")
    assert stage["errors"][0]["type"] == "RuntimeBuildStillRunning", stage
    assert json.loads((tmp_path / "run" / "teardown.json").read_text())["ok"] is False


def test_a_build_that_ends_within_the_bound_is_still_torn_down(monkeypatch, tmp_path):
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    server = mcp_server.McpSkillServer()
    building = threading.Event()
    torn = []

    def build():
        with server._init_lock:
            building.set()
            time.sleep(0.3)
            server._runtime = SimpleNamespace()

    monkeypatch.setattr(server, "_teardown_exclusive", lambda shutdown: torn.append(shutdown) or
                        {"ok": True, "complete": True, "stages": []})
    builder = threading.Thread(target=build, daemon=True)
    builder.start()
    assert building.wait(5)
    receipt = server.shutdown()
    builder.join(5)
    assert torn == [shutdown_runtime] and receipt["runtime_built"] is True, receipt
    assert next(s for s in receipt["stages"] if s["stage"] == "runtime")["ok"] is True


@pytest.mark.parametrize("bound", ["SHUTDOWN_BUILD_WAIT_S"])
def test_the_build_wait_bound_is_finite(bound):
    assert 0 < getattr(mcp_server, bound) <= 90  # systemd's default stop timeout
