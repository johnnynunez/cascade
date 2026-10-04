"""Same-host startup deadline contracts; no provider/model/CUDA imports."""
import copy
import socket
from types import SimpleNamespace as NS

import pytest

from test_conversation_provider_recipe import host  # noqa: F401

GPU = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture
def selected(host, monkeypatch):  # noqa: F811
    host.select_profile("cuda-llm-fp32")
    monkeypatch.setattr(host.acceleration, "monotonic", lambda: 100.0)
    return host


@pytest.mark.parametrize("deadline", [None, True, "200", float("nan"), float("inf"), 100, 251])
def test_inherited_deadline_is_finite_future_and_at_most150(selected, deadline):
    with pytest.raises(ValueError):
        selected.selected_startup_deadline(deadline)


def test_created_deadline_and_each_hop_keep_one_original_date(selected, tmp_path, monkeypatch):
    deadline = selected.selected_startup_deadline(None, create=True)
    assert deadline == 250.0
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: 180.0)
    assert selected.selected_startup_deadline(deadline, create=True) == deadline
    argv = selected.provider_argv(tmp_path, 18878, startup_deadline=deadline)
    key = "--llm_startup_deadline_monotonic_s"
    assert argv[argv.index(key)+1] == "250.0"
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: 250.0)
    with pytest.raises(ValueError):
        selected.provider_argv(tmp_path, 18878, startup_deadline=deadline)


def test_cpu_default_omits_and_refuses_startup_override(host, tmp_path):  # noqa: F811
    original = host.provider_argv(tmp_path, 18878)
    assert host.selected_startup_deadline(None, create=True) is None
    assert not any("startup" in value for value in original)
    with pytest.raises(ValueError, match="only"):
        host.provider_argv(tmp_path, 18878, startup_deadline=250)


def test_expired_startup_refuses_before_state_read_or_child(selected, monkeypatch, tmp_path):
    monkeypatch.setattr(selected, "verify", lambda *a: pytest.fail("late state read"))
    monkeypatch.setattr(selected, "supervise", lambda *a, **k: pytest.fail("late child"))
    with pytest.raises(ValueError, match="expired"):
        selected.serve(tmp_path, tmp_path/"run", port=18878, timeout_s=300,
                       cuda_device_uuid=GPU, startup_deadline=100)
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("inherited", [None, 240.0])
def test_serve_spends_state_verification_time_without_renewing_deadline(
        selected, monkeypatch, tmp_path, inherited):
    now = [100.0]
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: now[0])
    def verify(state):
        now[0] = 180.0
        return {}
    monkeypatch.setattr(selected, "verify", verify)
    calls = []
    monkeypatch.setattr(selected, "supervise", lambda command, **kw: calls.append((command, kw)) or 0)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    assert selected.serve(tmp_path, tmp_path / "run", port=port, timeout_s=360,
                          cuda_device_uuid=GPU, startup_deadline=inherited) == 0
    assert len(calls) == 1
    command, arguments = calls[0]
    flag = "--startup-deadline-monotonic-s"
    assert float(command[command.index(flag) + 1]) == (250.0 if inherited is None else inherited)
    assert arguments["timeout_s"] == 360  # Overall ownership is not the startup deadline.


def test_state_verification_cannot_authorize_child_after_original_startup_deadline(
        selected, monkeypatch, tmp_path):
    now = [100.0]
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: now[0])
    monkeypatch.setattr(selected, "verify", lambda state: now.__setitem__(0, 250.0))
    monkeypatch.setattr(selected, "supervise", lambda *a, **kw: pytest.fail("expired child start"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with pytest.raises(ValueError, match="expired"):
        selected.serve(tmp_path, tmp_path / "run", port=port, timeout_s=360,
                       cuda_device_uuid=GPU)


def warmup(index):
    return dict(index=index, startup_deadline_monotonic_s=250.0,
                start_monotonic_s=110.0+index*10, end_monotonic_s=115.0+index*10,
                work_deadline_monotonic_s=245.0, cleanup_deadline_monotonic_s=119.0+index*10,
                completed=True, stream_exhausted=True, producer_started=True,
                producer_alive=False, producer_start_uncertain=False,
                error_type=None, producer_error_type=None, cleanup_error_type=None,
                decoded_chunks=2, decoded_characters=4)


def actual_llm():
    return NS(startup_deadline_monotonic_s=250.0, streamer=NS(timeout=10.),
              warmup_diagnostics=[warmup(0), warmup(1)])


def test_actual_warmups_attest_original_deadline_without_changing_rows(selected, monkeypatch):
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: 200.0)
    llm = actual_llm()
    before = copy.deepcopy(llm.warmup_diagnostics)
    receipt = selected.acceleration.attest_startup(llm, 250.0)
    assert receipt["deadline_monotonic_s"] == 250.0
    assert receipt["conversation_streamer_wait_s"] == 10.0
    assert llm.warmup_diagnostics == before
    receipt["warmup_rows"][0]["decoded_chunks"] = 1000
    assert llm.warmup_diagnostics == before


@pytest.mark.parametrize("key,value", [
    ("startup_deadline_monotonic_s", 251.0), ("completed", 1), ("producer_alive", True),
    ("stream_exhausted", False), ("producer_start_uncertain", True),
    ("error_type", "Empty"), ("producer_error_type", "ValueError"),
    ("cleanup_error_type", "TimeoutError"), ("observation_error_type", "OSError"),
    ("work_deadline_monotonic_s", 250.0), ("cleanup_deadline_monotonic_s", 251.0),
    ("start_monotonic_s", 100.0), ("end_monotonic_s", 251.0),
    ("end_monotonic_s", float("nan")), ("decoded_chunks", True),
])
def test_late_renewed_uncertain_or_failed_warmup_never_attests(selected, monkeypatch, key, value):
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: 200.0)
    llm = actual_llm()
    llm.warmup_diagnostics[1][key] = value
    with pytest.raises(ValueError):
        selected.acceleration.attest_startup(llm, 250.0)


def test_deadline_is_rechecked_after_reading_actual_warmup_rows(selected, monkeypatch):
    times = iter((249.0, 249.0, 249.0, 250.0))
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: next(times))
    llm = actual_llm()
    with pytest.raises(ValueError, match="expired"):
        selected.acceleration.attest_startup(llm, 250.0)


@pytest.mark.parametrize("timeout", [None, True, 9., 11., "10"])
def test_actual_conversation_timeout_cannot_be_reported_as_an_assumed_ten(selected, timeout):
    llm = actual_llm()
    llm.streamer.timeout = timeout
    with pytest.raises(ValueError, match="actual conversational"):
        selected.acceleration.attest_startup(llm, 250.0)


@pytest.mark.parametrize("start,end,cleanup", [(99., 110., 114.), (245., 246., 249.), (195., 201., 205.)])
def test_warmup_cannot_predate_owner_start_enter_cleanup_reserve_or_report_future_end(
        selected, monkeypatch, start, end, cleanup):
    monkeypatch.setattr(selected.acceleration, "monotonic", lambda: 200.0)
    llm = actual_llm()
    llm.warmup_diagnostics[1].update(start_monotonic_s=start, end_monotonic_s=end,
                                   cleanup_deadline_monotonic_s=cleanup)
    with pytest.raises(ValueError):
        selected.acceleration.attest_startup(llm, 250.0)
