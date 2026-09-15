"""Check narrow rule ownership and receipt retirement with fake kernel replies."""

from copy import deepcopy
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

import network_service as service


@pytest.fixture
def nft_reply():
    # Independent expected wire shape, using the documented libnftables JSON schema.
    return json.loads('''{"nftables":[
      {"metainfo":{"version":"1.0.9","json_schema_version":1}},
      {"table":{"family":"inet","name":"paai_camera","handle":12}},
      {"chain":{"family":"inet","table":"paai_camera","name":"input","handle":1,
        "type":"filter","hook":"input","prio":-200,"policy":"accept"}},
      {"rule":{"family":"inet","table":"paai_camera","chain":"input","handle":2,"expr":[
        {"match":{"op":"!=","left":{"meta":{"key":"iifname"}},"right":"lo"}},
        {"match":{"op":"==","left":{"payload":{"protocol":"tcp","field":"dport"}},
          "right":{"set":[8554,8555,8556]}}},{"drop":null}]}},
      {"rule":{"family":"inet","table":"paai_camera","chain":"input","handle":3,"expr":[
        {"match":{"op":"!=","left":{"meta":{"key":"iifname"}},"right":"lo"}},
        {"match":{"op":"==","left":{"payload":{"protocol":"udp","field":"dport"}},
          "right":{"set":[8554,8555,8556]}}},{"drop":null}]}}
    ]}''')


def test_reviewed_dual_stack_input_drop_rules_are_accepted(nft_reply):
    assert service.validate_rules(nft_reply) is True


@pytest.mark.parametrize("corruption", ["wrong-hook", "dormant", "ipv4-only", "earlier-accept",
                                        "wrong-port", "missing-udp", "loopback-reversed", "drop-reordered"])
def test_plausible_firewall_changes_cannot_issue_admission(nft_reply, corruption):
    rows = nft_reply["nftables"]
    if corruption == "wrong-hook":
        rows[2]["chain"]["hook"] = "forward"
    elif corruption == "dormant":
        rows[1]["table"]["flags"] = ["dormant"]
    elif corruption == "ipv4-only":
        for row in rows[1:]:
            next(iter(row.values()))["family"] = "ip"
    elif corruption == "earlier-accept":
        rule = deepcopy(rows[3])
        rule["rule"]["expr"] = [{"accept": None}]
        rows.insert(3, rule)
    elif corruption == "wrong-port":
        rows[3]["rule"]["expr"][1]["match"]["right"]["set"][-1] = 8557
    elif corruption == "missing-udp":
        rows.pop()
    elif corruption == "loopback-reversed":
        rows[3]["rule"]["expr"][0]["match"]["op"] = "=="
    else:
        rows[3]["rule"]["expr"].reverse()
    with pytest.raises(ValueError, match="differs"):
        service.validate_rules(nft_reply)


@pytest.mark.parametrize("existing", [False, True])
def test_only_missing_owned_table_is_created_and_other_tables_are_preserved(monkeypatch, nft_reply, existing):
    calls = []
    tables = [{"table": {"family": "inet", "name": "tailscale"}},
              {"table": {"family": "ip", "name": "paai_camera"}}]
    if existing:
        tables.append({"table": {"family": "inet", "name": "paai_camera"}})

    def nft(*args, input_text=None):
        calls.append((args, input_text))
        if args == ("-j", "list", "tables"):
            return {"nftables": tables}
        if args == ("-j", "list", "table", "inet", "paai_camera"):
            return nft_reply
        if args == ("-f", "-"):
            assert not existing
            assert "flush" not in input_text and "delete" not in input_text
            assert input_text.count("table ") == 1 and "table inet paai_camera" in input_text
            assert "tailscale" not in input_text
            return {}
        pytest.fail(f"Unexpected firewall operation: {args}")

    monkeypatch.setattr(service, "nft", nft)
    service.ensure_rules()
    assert len(calls) == (2 if existing else 3)


def test_foreign_existing_owned_name_is_refused_without_flush_or_repair(monkeypatch, nft_reply):
    nft_reply["nftables"][2]["chain"]["hook"] = "forward"
    calls = []

    def nft(*args, **kwargs):
        calls.append(args)
        assert args[:2] == ("-j", "list"), "A mismatch must never be repaired implicitly"
        if args[-1] == "tables":
            return {"nftables": [{"table": {"family": "inet", "name": "paai_camera"}}]}
        return nft_reply

    monkeypatch.setattr(service, "nft", nft)
    with pytest.raises(ValueError, match="differs"):
        service.ensure_rules()
    assert len(calls) == 2


def test_nft_subprocess_has_finite_timeout_and_does_not_disclose_output(monkeypatch):
    def fail(args, **kwargs):
        assert args == ["/usr/sbin/nft", "-j", "list", "tables"]
        assert kwargs["timeout"] == 5 and kwargs["capture_output"] is True
        return SimpleNamespace(returncode=1, stdout="private-host-output", stderr="private-host-error")

    monkeypatch.setattr(service.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as caught:
        service.nft("-j", "list", "tables")
    assert "private-host" not in str(caught.value)


@pytest.fixture
def host_service(tmp_path, monkeypatch, nft_reply):
    directory = tmp_path / "runtime"
    directory.mkdir()
    admission = directory / "admission.json"
    admission.write_text('{"status":"old"}')
    monkeypatch.setattr(service.guard, "DIRECTORY", directory)
    monkeypatch.setattr(service.guard, "ADMISSION", admission)
    monkeypatch.setattr(service.guard, "host_identity", lambda: {"boot_id": "fixture-boot", "netns_inode": 42})
    monkeypatch.setattr(service.os, "geteuid", lambda: 0)
    monkeypatch.setattr(service.socket, "gethostname", lambda: "paai-demo-l40s")
    lstat = Path.lstat
    monkeypatch.setattr(Path, "lstat", lambda path, *args, **kwargs:
        SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0)
        if path == directory else lstat(path, *args, **kwargs))
    monkeypatch.setattr(service.signal, "signal", lambda *args: None)
    state = SimpleNamespace(admission=admission, now=100.0, beats=[], failure=None)
    monkeypatch.setattr(service.time, "monotonic", lambda: state.now)

    class Event:
        def wait(self, timeout):
            assert timeout == 5
            state.beats.append(json.loads(admission.read_text()))
            state.now += 5
            return len(state.beats) == 2

    def ensure():
        assert not admission.exists(), "Retire a previous process receipt before checking the kernel"

    def nft(*args):
        if state.failure is not None:
            raise state.failure
        return nft_reply

    monkeypatch.setattr(service.threading, "Event", Event)
    monkeypatch.setattr(service, "ensure_rules", ensure)
    monkeypatch.setattr(service, "nft", nft)
    monkeypatch.setattr(service, "notify_ready", lambda: json.loads(admission.read_text()))
    return state


def test_supervision_publishes_only_current_checks_and_removes_receipt_on_stop(host_service):
    service.main()
    assert [row["checked_monotonic"] for row in host_service.beats] == [100.0, 105.0]
    assert all(row["boot_id"] == "fixture-boot" and row["netns_inode"] == 42 for row in host_service.beats)
    assert not host_service.admission.exists()
    assert not list(host_service.admission.parent.glob(".admission-*.json"))


@pytest.mark.parametrize("failure", [RuntimeError("rules missing"), subprocess.TimeoutExpired("nft", 5)])
def test_kernel_check_failure_retires_last_good_receipt(host_service, failure):
    host_service.failure = failure
    with pytest.raises(type(failure)):
        service.main()
    assert len(host_service.beats) == 1
    assert not host_service.admission.exists()
