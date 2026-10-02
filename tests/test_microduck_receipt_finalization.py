"""Receipt publication is a signal-checked decision, not just byte equality.

CPU unit seams here set the notification scalar deterministically. Separate
owned-process replays exercise the same production path with OS signals.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
import isaac_microduck_bridge as cli  # noqa: E402
from cascade.sim.microduck_newton import KitNewtonBackend  # noqa: E402


@pytest.mark.parametrize("signum", [2, 15])
@pytest.mark.parametrize("boundary", ["read", "encode", "equal"])
def test_stable_receipt_requires_current_signal_after_all_work(
    tmp_path, monkeypatch, signum, boundary
):
    signals = SimpleNamespace(signum=None, registration_attempts=0)
    result = dict(completed=True, teardown_errors=[], receipt_revision=0)
    cli._resolve_outcome(result, signals)
    path = tmp_path / "receipt.json"
    cli.write_json(path, result)
    original = path.read_bytes()
    fired = []
    read, dumps = Path.read_bytes, json.dumps

    def deliver():
        if not fired:
            fired.append(boundary)
            signals.signum = signum

    class ComparedBytes(bytes):
        def __eq__(self, other):
            equal = super().__eq__(other)
            deliver()  # model delivery after equality computed, before return
            return equal

    def at_read(p):
        value = read(p)
        if p == path:
            if boundary == "read":
                deliver()
            elif boundary == "equal":
                return ComparedBytes(value)
        return value

    def at_encode(value, *args, **kwargs):
        encoded = dumps(value, *args, **kwargs)
        if value is result and boundary == "encode":
            deliver()
        return encoded

    monkeypatch.setattr(Path, "read_bytes", at_read)
    monkeypatch.setattr(json, "dumps", at_encode)
    cli._refresh_receipt(tmp_path, result, signals)
    final = json.loads(read(path))
    assert fired == [boundary]
    assert final["completed"] is False and final["exit_code"] == 128 + signum
    assert final["signal"] == signum and result == final
    assert read(tmp_path / "receipt-history-000.json") == original
    assert final["receipt_revision"] == 1
    assert not (tmp_path / ".receipt-update.json").exists()


@pytest.mark.parametrize("acquired", [False, True])
def test_sdk_shutdown_explicitly_reports_whether_close_was_called(acquired):
    backend = KitNewtonBackend(SimpleNamespace(), {}, Path("unused.kit"))
    backend._closed = True
    calls = []
    if acquired:
        backend.app = SimpleNamespace(close=lambda **kw: calls.append(kw))
    result = backend.shutdown(exit_code=17)
    assert result is acquired  # no handle is NOT a returned SDK.close
    assert calls == ([{"exit_code": 17}] if acquired else [])
    assert backend.app is None


@pytest.mark.parametrize("signum", [2, 15])
def test_constructor_cancellation_prevents_active_native_initialization(
    tmp_path, monkeypatch, signum
):
    """Real run/open lifecycle, unavoidable SDK acquisition double only."""
    import os
    import types
    from cascade.apps.signal_stop import StopSignals
    from test_microduck_bridge_cli import software_limits

    calls = []

    class App:
        def __init__(self, *args, **kwargs):
            os.kill(os.getpid(), signum)

        def close(self, **kwargs):
            calls.append(("sdk_close", kwargs))

    def active_initialize(self):
        calls.append(("active_initialize", None))
        raise AssertionError(
            "active native initialization reached after constructor cancellation"
        )

    module = types.ModuleType("isaacsim")
    module.SimulationApp = App
    monkeypatch.setitem(sys.modules, "isaacsim", module)
    monkeypatch.setattr(KitNewtonBackend, "_initialize", active_initialize)
    args = SimpleNamespace(
        out=tmp_path / "run",
        device="cuda:0",
        robot_id="microduck",
        source="software-only",
        max_wall_s=3.0,
        max_steps=5,
        port=0,
        camera_every=4,
        max_jpeg_bytes=100000,
        policy=tmp_path / "fixture.onnx",
        policy_sha256="b" * 64,
        python_extra_path=[],
    )
    admission = dict(
        asset_sha256="a" * 64,
        asset_receipt_sha256="c" * 64,
        bam_params={},
        limits=software_limits(),
        experience_text="software fixture\n",
    )
    with StopSignals(protect_registration=True) as signals:
        result = cli.run(
            args, admission, backend_factory=KitNewtonBackend, signals=signals
        )
    assert not [c for c in calls if c[0] == "active_initialize"], calls
    assert calls == [("sdk_close", {"exit_code": 128 + signum})]
    persisted = json.loads((args.out / "receipt.json").read_text())
    assert result == persisted
    assert result["exit_code"] == 128 + signum and result["completed"] is False


def lifecycle(
    tmp_path, monkeypatch, *, signum=None, fail_read=False, fail_teardown_write=False
):
    """Real run/KitNewtonBackend lifecycle; only SDK acquisition is a double.

    Startup ends at a CPU endpoint before any native initialization. A single
    injected I/O failure, optionally with an OS signal delivered at that exact
    boundary, must neither skip the mandatory SDK shutdown nor lose precedence.
    """
    import os
    import types
    from cascade.apps.signal_stop import StopSignals
    from test_microduck_bridge_cli import software_limits

    calls, fired = [], []
    out = tmp_path / "run"

    class App:
        def __init__(self, *args, **kwargs):
            calls.append("acquired")

        def close(self, **kwargs):
            calls.append(("close", kwargs["exit_code"]))

    module = types.ModuleType("isaacsim")
    module.SimulationApp = App
    monkeypatch.setitem(sys.modules, "isaacsim", module)
    monkeypatch.setattr(
        KitNewtonBackend,
        "_initialize",
        lambda self: (_ for _ in ()).throw(RuntimeError("CPU startup endpoint")),
    )

    def inject(kind):
        if fired:
            return
        fired.append(kind)
        if signum is not None:
            os.kill(os.getpid(), signum)
        raise OSError("injected one-shot " + kind + " I/O failure")

    real_read, real_write = Path.read_bytes, cli.write_json

    def read(path):
        if fail_read and path == out / "receipt.json":
            inject("receipt read")
        return real_read(path)

    def write(path, value):
        if fail_teardown_write and Path(path).name == "teardown.json":
            inject("teardown write")
        return real_write(path, value)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(cli, "write_json", write)
    args = SimpleNamespace(
        out=out,
        device="cuda:0",
        robot_id="microduck",
        source="software-only",
        max_wall_s=3.0,
        max_steps=5,
        port=0,
        camera_every=4,
        max_jpeg_bytes=100000,
        policy=tmp_path / "fixture.onnx",
        policy_sha256="b" * 64,
        python_extra_path=[],
    )
    admission = dict(
        asset_sha256="a" * 64,
        asset_receipt_sha256="c" * 64,
        bam_params={},
        limits=software_limits(),
        experience_text="software fixture\n",
    )
    with StopSignals(protect_registration=True) as signals:
        result = cli.run(
            args, admission, backend_factory=KitNewtonBackend, signals=signals
        )
    assert fired, "injection point never reached"
    return result, calls, json.loads(real_read(out / "receipt.json")), out


@pytest.mark.parametrize("signum", [None, 2, 15])
def test_receipt_read_failure_keeps_signal_precedence_and_mandatory_sdk_shutdown(
    tmp_path, monkeypatch, signum
):
    result, calls, receipt, _ = lifecycle(
        tmp_path, monkeypatch, signum=signum, fail_read=True
    )
    expected = 1 if signum is None else 128 + signum
    assert result["exit_code"] == expected and result["completed"] is False
    assert calls == ["acquired", ("close", expected)], calls
    assert (
        result["persistence_errors"]
        and "receipt read" in result["persistence_errors"][0]
    )
    assert result["sdk_shutdown"] == "returned"
    assert receipt == result  # the one-shot failure was reconciled after shutdown


@pytest.mark.parametrize("signum", [None, 2, 15])
def test_teardown_write_failure_is_not_reported_as_sdk_close_failure(
    tmp_path, monkeypatch, signum
):
    result, calls, receipt, out = lifecycle(
        tmp_path, monkeypatch, signum=signum, fail_teardown_write=True
    )
    expected = 1 if signum is None else 128 + signum
    # The signal is delivered after SDK.close already consumed the pre-signal
    # exit code (1); it cannot be rewritten, but the final outcome must be the signal's.
    assert calls == ["acquired", ("close", 1)], calls
    assert result["exit_code"] == expected and result["completed"] is False
    assert (
        result["sdk_shutdown"] == "returned"
        and result["receipt_phase"] == "sdk_shutdown_returned"
    )
    assert (
        result["persistence_errors"]
        and "teardown write" in result["persistence_errors"][0]
    )
    assert not (
        out / "teardown.json"
    ).exists()  # never a false sdk_close_returned=false
    assert receipt == result
