"""Explicit gateway stop is restricted to the same verified launch owner."""
import json
from types import SimpleNamespace

import pytest

from cascade.apps import process_owner as owners


@pytest.mark.parametrize("owned,status_pid,expected", [(True, 12345, "stop"),
                                                       (True, 54321, "refuse"),
                                                       (False, 12345, "ignore")])
def test_force_shutdown_requires_owner_and_service_pid(monkeypatch, tmp_path, owned, status_pid, expected):
    record = {"role": "gateway", "pid": 12345}
    owner = {"profile": "isolated-test"}
    live = [owned]
    commands = []
    monkeypatch.setattr(owners, "records", lambda _: [record])
    monkeypatch.setattr(owners, "is_live", lambda r, o: live[0])

    def run(command, **kwargs):
        commands.append(command)
        if command[-3:] == ["gateway", "status", "--json"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"service": {"runtime": {"pid": status_pid}}}))
        assert command == ["openclaw", "--profile", "isolated-test", "gateway", "stop", "--force"]
        assert live[0] and status_pid == record["pid"]
        live[0] = False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(owners.subprocess, "run", run)
    # The bounded stop owns a dedicated Popen group; this owner/PID test must
    # isolate that service boundary as well as the read-only status command.
    monkeypatch.setattr(owners, "_run_gateway_stop", lambda command: run(command))
    if expected == "refuse":
        with pytest.raises(ValueError, match="no longer belongs"):
            owners.stop_owned(tmp_path, owner)
        assert len(commands) == 1
    else:
        stopped = owners.stop_owned(tmp_path, owner)
        assert stopped == ([12345] if expected == "stop" else [])
        assert len(commands) == (2 if expected == "stop" else 0)
