"""CPU tests for the optional private provider's admission and owned lifecycle."""
import importlib.util
import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path

import pytest

from test_conversation_provider_source import source_bundle  # noqa: F401


@pytest.fixture
def host(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/conversation_provider.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("conversation_provider_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(host, source_bundle, monkeypatch):  # noqa: F811 (imported pytest fixture)
    from conversation_provider_source import apply_patch
    config, source, config_dir = source_bundle
    state = host.private_state(source.parent / "private", create=True)
    source.rename(state / "source")
    monkeypatch.setattr(host, "RECIPE", config_dir / "recipe.json")
    monkeypatch.setattr(host, "recipe", lambda: config)
    record = apply_patch(state / "source", config, config_dir)
    host.write_json(state / "source-patch.json", record)
    ref = state / "hf/hub/models--hexgrad--Kokoro-82M/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("a" * 40)
    host.write_json(state / "prepared.json", {
        "recipe_sha256": host.digest(host.RECIPE),
        "requirements_sha256": host.digest(host.REQUIREMENTS),
        "source_patch_sha256": host.digest(state / "source-patch.json"),
        "files_sha256": host.inventory(state),
    })
    return state


@pytest.mark.parametrize("change", ["source", "ref", "new_source", "missing"])
def test_verify_refuses_source_mutation_and_retargeted_default_revision(host, prepared, change):
    assert host.verify(prepared)["files_sha256"]
    source = prepared / "source/src/speech_to_speech/__init__.py"
    if change == "source":
        source.write_text("# changed")
    elif change == "ref":
        (prepared / "hf/hub/models--hexgrad--Kokoro-82M/refs/main").write_text("b" * 40)
    elif change == "missing":
        source.unlink()
    else:
        source.with_name("injected.py").write_text("# new module")
    with pytest.raises((ValueError, FileNotFoundError)):
        host.verify(prepared)


def test_environment_is_private_offline_cpu_and_does_not_change_parent(host, tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "HF_TOKEN", "HF_ENDPOINT", "PYTHONPATH", "PYTHONHOME"):
        monkeypatch.setenv(key, "foreign-value")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "foreign-gpu")
    before = dict(os.environ)
    env = host.environment(tmp_path, offline=True)
    assert os.environ == before
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HOME"] == before["HOME"]
    assert all(key not in env for key in ("OPENAI_API_KEY", "HF_TOKEN", "HF_ENDPOINT", "PYTHONPATH", "PYTHONHOME"))
    assert env["HF_HOME"] == str(tmp_path / "hf")


def test_owner_and_exclusive_operation_lock(host, tmp_path):
    state = host.private_state(tmp_path / "state", create=True)
    with (host.state_lock(state), pytest.raises(ValueError, match="already in use"),
          host.state_lock(state)):
        pytest.fail("second owner admitted")
    with host.state_lock(state):
        pass
    (state / "owner.json").write_text("{}")
    with pytest.raises(ValueError, match="ownership"):
        host.private_state(state)


def test_occupied_endpoint_is_preserved_without_starting_child(host, prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(host, "supervise", lambda *a, **k: pytest.fail("must not launch"))
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", 0))
        foreign.listen()
        run = tmp_path / "run"
        with pytest.raises(OSError):
            host.serve(prepared, run, port=foreign.getsockname()[1], timeout_s=1)
        assert foreign.fileno() >= 0
        assert not run.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux waitid/subreaper supervisor")
@pytest.mark.parametrize("request_stop", [False, True])
def test_real_child_is_reaped_on_deadline_or_explicit_stop(host, tmp_path, request_stop):
    event = threading.Event()
    if request_stop:
        event.set()
    code = host.supervise([sys.executable, "-c", "import time; time.sleep(30)"],
                          env=os.environ.copy(), run_dir=tmp_path, timeout_s=.15,
                          stop_event=event)
    closure = json.loads((tmp_path / "closure.json").read_text())
    assert code == -signal.SIGTERM
    assert closure["leader_reaped"] is True
    assert closure["owned_group_closed"] is True
    assert closure["remaining_group_members"] == []
    assert closure["reason"] == ("requested_stop" if request_stop else "deadline")
    assert closure["wall_s"] < 5
    pid = json.loads((tmp_path / "owner.json").read_text())["child_pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux waitid/subreaper supervisor")
@pytest.mark.parametrize("natural_exit", [True, False])
def test_orphaned_grandchild_cannot_outlive_provider_scope(host, tmp_path, natural_exit):
    child_pid = tmp_path / "grandchild.pid"
    script = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); "
        + ("sys.exit(0)" if natural_exit else "time.sleep(30)")
    )
    import ctypes
    libc = ctypes.CDLL(None)
    before = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(before), 0, 0, 0) == 0
    code = host.supervise([sys.executable, "-c", script], env=os.environ.copy(),
                          run_dir=tmp_path, timeout_s=1)
    after = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(after), 0, 0, 0) == 0
    assert after.value == before.value
    assert code == (0 if natural_exit else -signal.SIGTERM)
    receipt = json.loads((tmp_path / "closure.json").read_text())
    assert receipt["owned_group_closed"] and receipt["leader_reaped"]
    assert receipt["remaining_group_members"] == []
    assert int(child_pid.read_text()) in receipt["group_cleanup"]["descendants_reaped"]
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid.read_text()), 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux waitid/subreaper supervisor")
def test_group_cleanup_escalates_for_term_ignoring_descendant(host, tmp_path, monkeypatch):
    ready = tmp_path / "ready.pid"
    grandchild = (
        "import signal,time,os,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )
    parent = (
        f"import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-c',{grandchild!r}]);"
        f"p=pathlib.Path({str(ready)!r})\nwhile not p.exists():time.sleep(.005)\n"
    )
    original = host.close_group
    monkeypatch.setattr(host, "close_group", lambda p: original(p, grace_s=.05))
    assert host.supervise([sys.executable, "-c", parent], env=os.environ.copy(),
                          run_dir=tmp_path, timeout_s=5) != 0
    receipt = json.loads((tmp_path / "closure.json").read_text())
    assert receipt["owned_group_closed"]
    assert receipt["group_cleanup"]["signals"] == ["SIGTERM", "SIGKILL"]
    assert int(ready.read_text()) in receipt["group_cleanup"]["descendants_reaped"]


def test_bad_timeout_has_no_signal_handler_side_effect(host, tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="timeout"):
        host.supervise([], env={}, run_dir=tmp_path, timeout_s=float("nan"))
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS watchdog")
def test_memory_watchdog_records_failure_and_closes_own_group(host, tmp_path):
    code = host.supervise([sys.executable, "-c", "import time;x=bytearray(32*1024**2);time.sleep(30)"],
                          env=os.environ.copy(), run_dir=tmp_path, timeout_s=5, max_rss_bytes=1024**2)
    receipt = json.loads((tmp_path / "closure.json").read_text())
    assert code != 0 and receipt["reason"] == "memory_budget"
    assert receipt["sampled_peak_group_rss_bytes"] > receipt["rss_budget_bytes"]
    assert receipt["owned_group_closed"] and receipt["leader_reaped"]




def test_historical_state_recipe_is_not_silently_migrated(host, prepared):
    path = prepared / "prepared.json"
    value = json.loads(path.read_text())
    value["recipe_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="new private directory"):
        host.verify(prepared)
    assert path.read_bytes() == before


def test_source_record_mutation_is_rejected(host, prepared):
    path = prepared / "source-patch.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source record changed"):
        host.verify(prepared)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux waitid/subreaper supervisor")
def test_deadline_remains_failure_when_term_handler_exits_zero(host, tmp_path):
    ready = tmp_path / "ready"
    script = ("import signal,sys,time,pathlib;"
              "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0));"
              f"pathlib.Path({str(ready)!r}).write_text('ready');time.sleep(30)")
    code = host.supervise([sys.executable, "-c", script], env=os.environ.copy(),
                          run_dir=tmp_path, timeout_s=1.)
    closure = json.loads((tmp_path / "closure.json").read_text())
    assert ready.read_text() == "ready"
    assert closure["reason"] == "deadline" and closure["exit_code"] == 0
    assert closure["owned_group_closed"] and not closure["remaining_group_members"]
    assert code != 0
