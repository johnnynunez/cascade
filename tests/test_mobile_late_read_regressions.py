"""In-flight sampler returns cannot lose veto authority after their deadline.

Scripted CPU states and a module-local logical clock are software fixtures,
not physical evidence. No sleep is used to cross the observation deadline.
"""
from copy import deepcopy
import json
import threading
import time
from types import SimpleNamespace

import pytest

from cascade.agent import base_effects
from cascade.control.isaac_base import state_from_wire
from cascade.skills import mobile_runtime
from test_mobile_effects import fixture_support_contract, state
from test_mobile_runtime import verifier_limits


class LogicalClock:
    def __init__(self, now=100.0):
        self.now = now
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.now

    def advance_to(self, value):
        with self.lock:
            assert value >= self.now
            self.now = value


class InflightReader:
    """Six timely supported states, then one bounded in-flight return."""
    last_error = None

    def __init__(self, observer, clock, kind):
        self.observer, self.clock, self.kind = observer, clock, kind
        self.calls = []
        self.closed = False

    def __call__(self):
        op = self.observer.checker._active
        number = len(self.calls) + 1
        assert number <= 7, "sampler read again after the late terminal return"
        if number == 1:
            started = self.clock.monotonic()
            ended = started + .001
        else:
            # Wait only for the real finish() lifecycle transition. Scheduling
            # this handshake does not choose freshness, RTT or the verdict.
            limit = time.monotonic() + 2.0
            while op.finished is None:
                assert time.monotonic() < limit, "finish did not publish its boundary"
                threading.Event().wait(.001)
            deadline = min(op.deadline, op.finished + self.observer.limits["settle_timeout_s"])
            started = self.clock.monotonic()
            ended = (op.finished + (number - 1) * .15 if number < 6 else
                     deadline - .02 if number == 6 else deadline + .005)
        self.clock.advance_to(ended)
        fields = {"controller_status": "fault"} if number == 7 and self.kind == "fault" else {}
        value = state(number, sim_time_s=(number - 1) * .03,
                      received_monotonic_s=started,
                      producer_age_s=.21 if number == 7 and self.kind == "stale" else 0.,
                      **fields)
        # Exercise the same conservative RTT addition as BaseTruthReader.
        result = state_from_wire({"ok": True, "state": value.as_dict()},
                                 received_monotonic_s=ended, round_trip_s=ended - started)
        self.calls.append({"started": started, "ended": ended, "state": result.as_dict(),
                           "worker_ident": threading.get_ident()})
        return result

    def close(self):
        self.closed = True


@pytest.mark.parametrize("skill", ["stop_navigation", "emergency_stop"])
@pytest.mark.parametrize("kind", ["fresh", "fault", "stale"])
def test_sampler_late_return_preserves_veto_without_positive_credit(
        tmp_path, monkeypatch, skill, kind):
    """The former sampler discarded all late returns, including real vetoes."""
    limits = verifier_limits()
    clock = LogicalClock()
    real_monotonic = time.monotonic
    # Replace these modules' bindings, never time.monotonic on the shared
    # standard-library module. Thread/event timeouts retain real bounded waits.
    monkeypatch.setattr(base_effects, "time", clock)
    monkeypatch.setattr(mobile_runtime, "time", clock)
    assert time.monotonic is real_monotonic

    sample = state(1)
    identity = {key: getattr(sample, key) for key in
                ("robot_id", "source", "epoch", "model_identity_sha256", "generation")}
    profile = {**identity, "engine": "physx", "device": "cuda:0",
               "asset_sha256": "a" * 64, "policy_sha256": "b" * 64,
               "bridge_host": "localhost", "bridge_port": 1,
               "timeout_s": limits["read_timeout_s"], "verifier": limits,
               "support_contract": fixture_support_contract()}
    observer = mobile_runtime._StopObserver(profile, lambda _job: True, lambda *_args: None)
    observer.reader.reader.close()  # unopened passive transport; never connect
    reader = InflightReader(observer, clock, kind)
    observer.reader.reader = reader
    ack = {**identity, "ok": True, "latched": True, "physical_stop_verified": False}
    job = {"ack_monotonic_s": clock.monotonic(), "ack": ack, "skill": skill,
           "base": "synthetic", "receipt_id": "late-return-regression"}
    original_ack = deepcopy(ack)
    try:
        verdict = observer._observe(job)
        evidence = verdict["evidence"]
        (tmp_path / "receipt.json").write_text(json.dumps({
            "verdict": verdict, "reads": reader.calls, "quarantined": observer._quarantined,
            "physical_acceptance": False, "limits_changed": observer.limits != verifier_limits(),
        }, indent=2, allow_nan=False) + "\n")
        assert observer.limits == verifier_limits()
        assert time.monotonic is real_monotonic and ack == original_ack
        assert len(reader.calls) == evidence["attempts"] == 7
        assert all(call["worker_ident"] != threading.get_ident() for call in reader.calls)
        assert len({call["worker_ident"] for call in reader.calls}) == 1
        assert not observer.checker._worker.is_alive()
        assert evidence["late_reads"] == 1
        assert [row["state"]["step"] for row in evidence["samples"]] == [1, 2, 3, 4, 5, 6]
        prior = SimpleNamespace(stop_boundary=evidence["stop_boundary"],
                                finished=evidence["finished_monotonic_s"],
                                samples=evidence["samples"],
                                observations=evidence["observations"][:-1])
        assert observer.checker._settle(prior)[0] == "confirmed"
        late = evidence["observations"][-1]
        assert late["state"]["step"] == 7 and late["late"]
        assert not late["confirmation_eligible"]
        deadline = evidence["finished_monotonic_s"] + limits["settle_timeout_s"]
        assert reader.calls[-1]["started"] < deadline < reader.calls[-1]["ended"]
        assert reader.calls[-1]["ended"] - reader.calls[-1]["started"] < limits["read_timeout_s"]
        if kind == "stale":
            assert verdict["status"] == "unverified"
            assert verdict["reason"] == "stale independent state"
            assert evidence["channel_failed"] and observer._quarantined
            assert not late["valid"]
            calls = len(reader.calls)
            assert "quarantined" in observer._observe(job)["reason"]
            assert len(reader.calls) == calls  # no new worker/read repairs this channel
        else:
            assert late["valid"] and not evidence["channel_failed"]
            assert not observer._quarantined  # a measured fault is a valid channel
            assert verdict["metrics"]["settle_sim_duration_s"] >= limits["settle_window_s"]
            if kind == "fresh":
                assert verdict["status"] == "confirmed", verdict["reason"]
            else:
                assert verdict["status"] == "refuted"
                assert verdict["reason"] == "controller fault/disabled observed"
    finally:
        observer.close()
    assert reader.closed
