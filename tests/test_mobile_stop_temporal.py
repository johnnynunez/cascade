"""Stop temporal contract: CPU/scripted states, NEVER physical acceptance."""
import copy
import json
import threading
import time

import pytest
from test_mobile_effects import ScriptedReader, checker_for, state
from test_mobile_frames import frame_endpoint  # noqa: F401
from test_mobile_runtime import SyntheticTicks, await_stop, camera_cfg, verifier_limits


class HeldTickProducer:
    """Hold one genuinely post-ACK capture for two reads, then resume .005s ticks."""
    def __init__(self, controller, velocity):
        self.c, self.velocity = controller, velocity
        self.ack, self.published, self.resume, self.halt = (threading.Event() for _ in range(4))
        self.job = None
        self.captured_monotonic_s = None
        self.errors = []
        self.thread = threading.Thread(target=self.run, name="temporal-synthetic-ticks")
        self.thread.start()

    def run(self):
        from mobile_support_fixture import support
        try:
            if not self.ack.wait(2) or self.halt.is_set():
                return
            if self.halt.wait(max(0., self.job["ack_monotonic_s"] + .06 - time.monotonic())):
                return
            step = self.c.state()["state"]["step"]
            first = True
            while not self.halt.is_set():
                step += 1
                self.c.control_at(step * .005)
                self.c.publish({"step": step, "sim_time": step * .005, "position": [0., 0., .3],
                                "orientation_wxyz": [1., 0., 0., 0.], "linear_velocity": [self.velocity, 0., 0.],
                                "angular_velocity": [0., 0., 0.], "q": [0.] * 14, "dq": [0.] * 14,
                                "joint_names": [f"fixture-{i}" for i in range(14)],
                                "contacts": [], "fallen": False, "balance_active": True,
                                "support": support(step, step * .005)})
                if first:
                    first = False
                    self.captured_monotonic_s = time.monotonic()
                    self.published.set()
                    if not self.resume.wait(2.):
                        self.errors.append("two-read hold was not released")
                        return
                self.halt.wait(.005)
        except Exception as exc:  # noqa: BLE001 - fail teardown for ANY producer failure
            self.errors.append(repr(exc))

    def close(self):
        self.halt.set()
        self.ack.set()
        self.resume.set()
        self.thread.join(1)
        assert not self.thread.is_alive()
        assert not self.errors, self.errors


@pytest.mark.parametrize("jitter_read", [1, 2], ids=["before-baseline", "after-baseline"])
@pytest.mark.parametrize("velocity,expected", [(0., "confirmed"), (.05, "refuted")])
def test_real_subtimeout_jitter_preserves_stop_discrimination(tmp_path, frame_endpoint, monkeypatch, jitter_read, velocity, expected):  # noqa: F811
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, server, profile, _, _, _ = frame_endpoint
    assert server.address[0] == "127.0.0.1"
    # This fixture tests healthy but temporally ambiguous reads. Keep their
    # real TCP latency/age inside an explicit budget on slower CI schedulers;
    # physical sample windows and stop discrimination thresholds are unchanged.
    test_limits = {**verifier_limits(), "read_timeout_s": .5,
                   "settle_timeout_s": .8, "max_wall_duration_s": 3., "max_state_age_s": 1.}
    profile.update(timeout_s=.5, verifier=test_limits)
    profile.pop("cameras")
    ticks = HeldTickProducer(c, velocity)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    original = observer.reader.reader
    request = original._client.request
    observe = observer._observe
    reads = []
    calls = [0]
    def outgoing(payload, **kwargs):
        if payload["op"] == "state":
            calls[0] += 1
            if calls[0] == jitter_read:
                # Actual RTT must straddle the ACK even if publication was
                # scheduled late. No fixture timestamp or wire field is forged.
                time.sleep(ticks.captured_monotonic_s - ticks.job["ack_monotonic_s"] + .04)
        return request(payload, **kwargs)
    class Tap:
        def __call__(self):
            started = time.monotonic()
            value = original()
            reads.append({"start": started, "end": time.monotonic(),
                          "state": None if value is None else value.as_dict(), "last_error": original.last_error})
            if len(reads) == 2:
                ticks.resume.set()
            return value
        @property
        def last_error(self):
            return original.last_error
        def close(self):
            original.close()
    def scheduled(job):
        ticks.job = job
        ticks.ack.set()
        assert ticks.published.wait(2.)
        return observe(job)
    monkeypatch.setattr(original._client, "request", outgoing)
    monkeypatch.setattr(observer.reader, "reader", Tap())
    monkeypatch.setattr(observer, "_observe", scheduled)
    try:
        assert rt.reset_stop()["ok"]
        ack = rt.execute("emergency_stop", {})
        frozen = copy.deepcopy(ack)
        proof = await_stop(rt, ack["receipt_id"])
        assert proof["status"] == expected, proof
        assert not observer._quarantined and observer.limits == test_limits
        assert ack == frozen and not ack["physical_stop_verified"]
        assert all(r["last_error"] is None and r["state"] is not None and r["end"] - r["start"] < test_limits["read_timeout_s"] for r in reads)
        assert reads[0]["state"]["step"] == reads[1]["state"]["step"]
        excluded = reads[jitter_read - 1]["state"]
        assert excluded["received_monotonic_s"] - excluded["producer_age_s"] < ack["ack_monotonic_s"]
        evidence = proof["postcondition"]["evidence"]
        assert evidence["attempts"] == len(reads) == calls[0]  # every read charged once
        assert evidence["temporal_pending"] and evidence["rejected"] == []
        assert evidence["samples"][0]["phase"] == "before"
        assert sum(s["phase"] == "before" for s in evidence["samples"]) == 1
        assert evidence["wall_deadline_monotonic_s"] == ack["ack_monotonic_s"] + test_limits["max_wall_duration_s"]
        for s in evidence["samples"]:
            assert s["state"]["received_monotonic_s"] - s["state"]["producer_age_s"] >= ack["ack_monotonic_s"]
        done = rt.execute("task_done", {"success": True, "summary": "socket fixture"})
        assert done["success"] is (velocity == 0.)
        (tmp_path / "jitter.json").write_text(json.dumps({"ack": ack, "proof": proof, "reads": reads,
                                                       "evidence_class": "software / synthetic ticks / real TCP"}, indent=2))
    finally:
        rt.close()
        ticks.close()


def test_receipt_callback_failure_is_a_channel_failure_not_silent_sampler_death():
    def broken():
        raise RuntimeError("fixture receipt checker failed")
    checker = checker_for(ScriptedReader())
    try:
        token = checker.begin("stop", {}, stop_boundary=boundary(), is_current=broken)
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified"
        assert verdict["evidence"]["channel_failed"] is True
        assert "fixture receipt checker failed" in verdict["reason"]
    finally:
        checker.close()


@pytest.mark.parametrize("supersede", [False, True])
def test_real_socket_timeout_quarantines_even_superseded_reader(tmp_path, frame_endpoint, monkeypatch, supersede):  # noqa: F811
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, server, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    entered = threading.Event()
    calls, dispatch = [], server.dispatch
    def delay(req):
        if req["op"] == "state":
            calls.append(time.monotonic())
            entered.set()
            time.sleep(.060)  # actual socket recv timeout, not an injected exception
        return dispatch(req)
    try:
        assert rt.reset_stop()["ok"]
        monkeypatch.setattr(server, "dispatch", delay)
        first = rt.stop()
        frozen = copy.deepcopy(first)
        assert entered.wait(1)
        second = rt.stop() if supersede else first
        proof = await_stop(rt, second["receipt_id"])
        assert proof["status"] == "unverified" and not proof["physical_stop_verified"]
        assert observer._quarantined and len(calls) == 1
        assert first == frozen and first["physical_stop_verified"] is False
        assert not rt.execute("task_done", {"success": True, "summary": "timeout"})["success"]
        rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        original = next(r["result"] for r in rows if r["skill"] == "stop_verification"
                        and r["result"]["receipt_id"] == first["receipt_id"])
        assert "timed out" in original["postcondition"]["reason"]
        assert original["postcondition"]["evidence"]["channel_failed"] is True
        if supersede:
            assert original["superseded"] and "quarantined" in proof["reason"]
        retry = rt.stop()
        assert "quarantined" in await_stop(rt, retry["receipt_id"])["reason"]
        assert len(calls) == 1  # quarantined channel creates no replacement reader
        (tmp_path / "socket-negative.json").write_text(json.dumps({
            "evidence_class": "software / real loopback / synthetic producer, NOT physics",
            "endpoint": list(server.address), "ack": first, "original": original,
            "proof": proof, "calls": calls}, indent=2))
    finally:
        rt.close()
        ticks.close()
        server.close()


@pytest.mark.parametrize("bad", ["missing", "malformed", "robot_id", "source", "epoch", "generation", "stale", "frozen", "frozen_aged"])
def test_real_invalid_stop_channel_is_unverified_and_stays_quarantined(tmp_path, frame_endpoint, monkeypatch, bad):  # noqa: F811
    from cascade.apps.mobile_runtime import build_mobile_runtime
    c, server, profile, _, _, _ = frame_endpoint
    profile.update(timeout_s=.03, verifier=verifier_limits())
    profile.pop("cameras")
    ticks = None if bad in ("frozen", "frozen_aged") else SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    dispatch = server.dispatch
    def corrupt(req):
        value = dispatch(req)
        if req["op"] == "state":
            if bad == "missing":
                value["state"] = None
            elif bad == "malformed":
                value["state"]["orientation_wxyz"] = [0., 0., 0., 0.]
            elif bad in ("robot_id", "source", "epoch"):
                value["state"][bad] = "wrong-identity"
            elif bad == "generation":
                value["state"]["generation"] += 1
            elif bad == "stale":
                value["state"]["producer_age_s"] = .3
            elif bad == "frozen_aged":
                # Model setup consuming the freshness budget before the first
                # read, without sleeping or advancing the frozen producer.
                value["state"]["producer_age_s"] += .3
        return value
    try:
        assert rt.reset_stop()["ok"]
        monkeypatch.setattr(server, "dispatch", corrupt)
        ack = rt.stop()
        proof = await_stop(rt, ack["receipt_id"])
        assert proof["status"] == "unverified", proof
        assert proof["postcondition"]["evidence"]["channel_failed"] and observer._quarantined
        if bad in ("frozen", "frozen_aged"):
            assert "stale" in proof["reason"]
            # A frozen snapshot may already be stale at the first read. Healthy
            # pre-ACK pending reads are not required for rejection/quarantine.
            if bad == "frozen_aged":
                assert proof["postcondition"]["evidence"]["temporal_pending"] == []
            assert proof["postcondition"]["evidence"]["samples"] == []
        assert not rt.execute("task_done", {"success": True, "summary": "invalid"})["success"]
        monkeypatch.setattr(server, "dispatch", dispatch)  # a later healthy packet cannot clear quarantine
        retry = rt.stop()
        assert "quarantined" in await_stop(rt, retry["receipt_id"])["reason"]
        (tmp_path / "invalid-channel.json").write_text(json.dumps(proof, indent=2))
    finally:
        rt.close()
        if ticks:
            ticks.close()



INVALID = ["missing", "malformed", "raise", "nan", "quaternion", "stale", "future", "receipt_regression",
           "robot_id", "source", "epoch", "generation", "step", "sim_time_s", "gap", "plausibility",
           "duplicate_conflict", "duplicate_fault", "fault", "disabled"]


@pytest.mark.parametrize("prebaseline", [True, False])
@pytest.mark.parametrize("bad", INVALID)
def test_temporal_exclusion_never_hides_invalid_channel(bad, prebaseline):
    stamps = []
    def make(n):
        value = state(n, producer_age_s=.15 if n == 2 or prebaseline else 0.)
        if n == 2:
            if bad == "missing":
                return None
            if bad == "malformed":
                return {"not": "BaseState"}
            if bad == "raise":
                raise RuntimeError("fixture independent channel failed")
            edits = {
                "nan": {"position_world": (float("nan"), 0., .3)},
                "quaternion": {"orientation_wxyz": (0., 0., 0., 0.)},
                "stale": {"producer_age_s": .3},
                "future": {"received_monotonic_s": time.monotonic() + 1},
                "receipt_regression": {"received_monotonic_s": stamps[-1] - .001},
                "robot_id": {"robot_id": "wrong"}, "source": {"source": "wrong"},
                "epoch": {"epoch": "wrong"}, "generation": {"generation": 1},
                "step": {"step": 0}, "sim_time_s": {"sim_time_s": 0.},
                "gap": {"sim_time_s": 3.}, "plausibility": {"linear_velocity_world": (1e6, 0., 0.)},
                "duplicate_conflict": {"step": 1, "sim_time_s": .02, "position_world": (.001, 0., .3)},
                "duplicate_fault": {"step": 1, "sim_time_s": .02, "controller_status": "fault"},
                "fault": {"controller_status": "fault"}, "disabled": {"controller_status": "disabled"},
            }[bad]
            for key, v in edits.items():
                object.__setattr__(value, key, v)  # deliberately hostile reader, validated by production
        stamps.append(value.received_monotonic_s)
        return value
    reader = ScriptedReader(make)
    checker = checker_for(reader)
    try:
        token = checker.begin("stop_navigation", {}, stop_boundary=boundary())
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified", verdict
        assert verdict["evidence"]["channel_failed"] is True
        assert verdict["evidence"]["rejected"][0]["attempt"] == 2
        assert reader.calls == 2  # real faults are terminal; no later healthy read clears them
        assert len(verdict["evidence"]["samples"]) == (0 if prebaseline else 1)
    finally:
        checker.close()


@pytest.mark.parametrize("prebaseline", [True, False])
def test_duplicates_cannot_supply_or_rejuvenate_baseline(prebaseline):
    reader = ScriptedReader(lambda n: state(1, producer_age_s=.15 if prebaseline and n == 1 else 0.))
    checker = checker_for(reader, max_samples=8)
    try:
        token = checker.begin("stop", {}, stop_boundary=boundary())
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified" and verdict["reason"] == "sample_limit"
        evidence = verdict["evidence"]
        assert evidence["attempts"] == reader.calls == 8
        assert len(evidence["samples"]) == (0 if prebaseline else 1)
        assert evidence["duplicates"] == 7 and not evidence["channel_failed"]
    finally:
        checker.close()


def test_many_healthy_pending_reads_are_not_one_reader_timeout():
    started = time.monotonic()
    # Keep the first 25 captures before the ACK independently of OS sleep
    # granularity; a constant .15s age could cross the fence partway through.
    reader = ScriptedReader(lambda n: state(n, sim_time_s=n * .005,
        producer_age_s=time.monotonic() - started + .01 if n <= 25 else 0.))
    checker = checker_for(reader, max_state_age_s=3.)
    try:
        token = checker.begin("stop", {}, stop_boundary=boundary(started))
        assert time.monotonic() - started > checker._limits["read_timeout_s"]
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "confirmed", verdict
        assert len(verdict["evidence"]["temporal_pending"]) == 25
        assert verdict["evidence"]["samples"][0]["state"]["step"] == 26
        assert not verdict["evidence"]["channel_failed"]
    finally:
        checker.close()


@pytest.mark.parametrize("baseline", [False, True])
def test_ack_wall_deadline_is_not_restarted_by_begin_or_finish(baseline):
    fence = boundary(time.monotonic() - .08)
    reader = ScriptedReader(lambda n: state(n, producer_age_s=0. if baseline else .15))
    checker = checker_for(reader, max_wall_duration_s=.1, settle_timeout_s=.1)
    try:
        started = time.monotonic()
        token = checker.begin("stop", {}, stop_boundary=fence)
        verdict = checker.finish(token, {"ok": True})
        assert time.monotonic() - started < .06  # remaining wall budget, not a new .1s window
        assert verdict["status"] == "unverified" and "wall_deadline" in verdict["reason"]
        evidence = verdict["evidence"]
        assert evidence["wall_deadline_monotonic_s"] == fence["ack_monotonic_s"] + .1
        assert evidence["attempts"] == reader.calls
        assert not evidence["channel_failed"]
        assert all(s["observed_monotonic_s"] <= evidence["wall_deadline_monotonic_s"] for s in evidence["samples"])
    finally:
        checker.close()


def test_baseline_and_settling_share_one_attempt_quota():
    reader = ScriptedReader(lambda n: state(n, producer_age_s=.15 if n < 5 else 0.))
    checker = checker_for(reader, max_samples=8)
    try:
        token = checker.begin("stop", {}, stop_boundary=boundary())
        verdict = checker.finish(token, {"ok": True})
        evidence = verdict["evidence"]
        assert verdict["status"] == "unverified" and verdict["reason"] == "sample_limit"
        assert evidence["attempts"] == reader.calls == 8
        assert len(evidence["temporal_pending"]) == 4
        assert len(evidence["samples"]) == 4 and evidence["samples"][0]["phase"] == "before"
    finally:
        checker.close()


@pytest.mark.parametrize("change", [
    {"ack_monotonic_s": True}, {"ack_monotonic_s": float("nan")}, {"ack_monotonic_s": 1e30},
    {"ack_monotonic_s": -1.}, {"generation": True}, {"generation": -1},
    {"epoch": " padded "}, {"robot_id": ""}, {"extra": "no"},
])
def test_stop_boundary_rejects_malformed_contract_without_reading(change):
    reader = ScriptedReader()
    checker = checker_for(reader)
    try:
        with pytest.raises(ValueError):
            checker.begin("stop", {}, stop_boundary={**boundary(), **change})
        assert reader.calls == 0 and checker._worker is None
    finally:
        checker.close()


def test_stop_boundary_is_restricted_and_defensively_copied():
    reader = ScriptedReader()
    checker = checker_for(reader)
    fence = boundary()
    try:
        with pytest.raises(ValueError):
            checker.begin("walk_velocity", {}, stop_boundary=fence)
        with pytest.raises(ValueError):
            checker.begin("stop", {}, is_current=lambda: True)
        with pytest.raises(ValueError):
            checker.begin("stop", {}, stop_boundary=fence, is_current=False)
        assert reader.calls == 0
        token = checker.begin("stop", {}, stop_boundary=fence)
        fence["generation"] = 999
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "confirmed"
        assert verdict["evidence"]["stop_boundary"]["generation"] == 0
    finally:
        checker.close()


def test_close_cannot_erase_original_stop_channel_failure():
    checker = checker_for(ScriptedReader(lambda n: None))
    token = checker.begin("stop", {}, stop_boundary=boundary())
    checker.close()
    verdict = checker.finish(token, {"ok": True})
    assert verdict["status"] == "unverified" and verdict["reason"] == "missing_state"
    assert verdict["evidence"]["channel_failed"] is True


@pytest.fixture(autouse=True)
def no_temporal_thread_leaks():
    before = set(threading.enumerate())
    yield
    deadline = time.monotonic() + 1.
    while True:
        leaked = [t.name for t in set(threading.enumerate()) - before if t.is_alive()]
        if not leaked or time.monotonic() >= deadline:
            break
        time.sleep(.005)
    assert not leaked, leaked


def boundary(ack_time=None):
    return {"ack_monotonic_s": time.monotonic() if ack_time is None else ack_time,
            "robot_id": "synthetic-microduck", "source": "scripted-software-fixture",
            "epoch": "fixture-epoch", "generation": 0, "model_identity_sha256": "e" * 64}


@pytest.mark.parametrize("velocity,expected", [(0., "confirmed"), (.05, "refuted")])
def test_pending_before_and_after_baseline_preserves_useful_stop_verdict(velocity, expected):
    # Independent state fixtures, no actor targets or checker verdict injection.
    reader = ScriptedReader(lambda n: state(n, producer_age_s=.15 if n in (1, 2, 4) else 0.,
                                            linear_velocity_world=(velocity, 0., 0.)))
    checker = checker_for(reader)
    fence = boundary()
    try:
        token = checker.begin("emergency_stop", {}, stop_boundary=fence)
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == expected, verdict
        evidence = verdict["evidence"]
        assert evidence["channel_failed"] is False
        assert evidence["rejected"] == []
        assert [p["attempt"] for p in evidence["temporal_pending"]] == [1, 2, 4]
        assert evidence["samples"][0]["phase"] == "before"
        assert evidence["samples"][0]["state"]["step"] == 3
        assert sum(e["phase"] == "before" for e in evidence["samples"]) == 1
        assert all(e["state"]["received_monotonic_s"] - e["state"]["producer_age_s"]
                   >= fence["ack_monotonic_s"] for e in evidence["samples"])
        assert evidence["attempts"] == reader.calls
    finally:
        checker.close()


@pytest.mark.parametrize("controller", ["fault", "disabled"])
def test_supersession_only_excuses_healthy_state_not_controller_fault(controller):
    current = [True]
    def make(n):
        current[0] = False  # receipt changes during the in-flight read
        return state(n, generation=1, controller_status=controller)
    checker = checker_for(ScriptedReader(make))
    try:
        token = checker.begin("emergency_stop", {}, stop_boundary=boundary(), is_current=lambda: current[0])
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified"
        assert verdict["evidence"]["channel_failed"] is True
        assert "controller" in verdict["reason"]
    finally:
        checker.close()


def test_pending_samples_do_not_relax_accepted_measurement_gap():
    reader = ScriptedReader(lambda n: state(n, sim_time_s=n * .06,
                                            producer_age_s=.15 if n in (2, 3) else 0.))
    checker = checker_for(reader)
    try:
        token = checker.begin("stop", {}, stop_boundary=boundary())
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified"
        assert "gap" in verdict["reason"]
        assert verdict["evidence"]["channel_failed"] is True
    finally:
        checker.close()
