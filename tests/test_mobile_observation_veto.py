"""Canonical-main temporal checks: actual checker + real TCP, no physics.

These use CURRENT unchanged repository budgets and explicit valid synthetic
support. Prior-suite interface/timing incompatibilities are not product bugs.
"""

import copy
import json
import time

import pytest
from cascade.agent.base_effects import BasePostconditionChecker, _Window
from test_mobile_effects import state, limits, fixture_support_contract
from test_mobile_frames import frame_endpoint as frame_endpoint
from test_mobile_runtime import SyntheticTicks, verifier_limits, camera_cfg, await_stop


def save(directory, name, data):
    with (directory / (name + ".json")).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def yield_until(deadline):
    """Hold a test reply to a fixed deadline without a positive-duration timer.

    The requested hold is at most read_timeout / 3 + sample_interval here.
    Zero-duration yields let the producer run without accumulating timer
    overshoot. Actual transport time and state age remain subject to all
    production checks; this is not a real-time scheduling guarantee.
    """
    while time.monotonic() < deadline:
        time.sleep(0)


@pytest.mark.parametrize(
    "case",
    [
        "quiet_pending",
        "moving_pending",
        "active_pending",
        "duplicate_active_short",
        "duplicate_active_full",
    ],
)
def test_excluded_states_veto_but_never_supply_positive_rest(case, tmp_path):
    lim = limits()
    checker = BasePostconditionChecker(
        lambda: pytest.fail("no worker expected"),
        limits=lim,
        support_contract=fixture_support_contract(),
    )
    start = 100.0
    boundary = dict(
        ack_monotonic_s=start,
        robot_id="synthetic-microduck",
        source="scripted-software-fixture",
        epoch="fixture-epoch",
        generation=0,
        model_identity_sha256="e" * 64,
    )
    op = _Window(
        "canonical",
        "emergency_stop",
        {},
        start,
        start + lim["max_wall_duration_s"],
        finished=start,
        stop_boundary=boundary,
    )
    rows = []
    last = None
    step = 0

    def add(sim, pending=False, **fields):
        nonlocal last, step
        if sim != last:
            step += 1
        last = sim
        op.attempts += 1
        ended = start + op.attempts * lim["sample_interval_s"]
        began = ended - lim["sample_interval_s"] / 2
        age = ended - start + lim["sample_interval_s"] if pending else 0.0
        value = state(
            step,
            sim_time_s=sim,
            received_monotonic_s=ended,
            producer_age_s=age,
            **fields,
        )
        checker._accept(op, value, began, ended)
        rows.append({"state": value.as_dict(), "started": began, "ended": ended})

    try:
        for sim in (0.0, 0.04, 0.08, 0.12):
            add(sim)
        if case.endswith("pending"):
            for sim in (0.14, 0.16):
                add(
                    sim,
                    pending=True,
                    linear_velocity_world=(
                        0.05 if case == "moving_pending" else 0.0,
                        0.0,
                        0.0,
                    ),
                    controller_status="active" if case == "active_pending" else "ready",
                )
        else:
            add(0.12, controller_status="active")
            for sim in (0.14,) if case.endswith("short") else (0.14, 0.18, 0.22):
                add(sim)
        verdict = checker._receipt(op, {"ok": True})
        checker._evaluate(op, verdict)
        save(
            tmp_path,
            case,
            {
                "verdict": verdict,
                "input": rows,
                "limits": lim,
                "physical_acceptance": False,
            },
        )
        assert checker._limits == limits()
        if case in ("quiet_pending", "duplicate_active_full"):
            assert verdict["status"] == "confirmed", verdict
        else:
            assert verdict["status"] != "confirmed", verdict
    finally:
        checker.close()


@pytest.mark.parametrize("skill", ["stop_navigation", "emergency_stop"])
@pytest.mark.parametrize("kind", ["healthy_late", "fault_timely", "fault_late"])
def test_real_tcp_inflight_fault_must_block_task_done(
    tmp_path, frame_endpoint, monkeypatch, skill, kind
):
    from cascade.apps.mobile_runtime import build_mobile_runtime

    c, server, profile, *_ = frame_endpoint
    lim = verifier_limits()
    profile.update(timeout_s=lim["read_timeout_s"], verifier=lim)
    profile.pop("cameras")
    ticks = SyntheticTicks(c)
    rt, _ = build_mobile_runtime(camera_cfg(profile), tmp_path)
    observer = rt.stop_observers["microduck_isaac"]
    dispatch = server.dispatch
    captured = []
    reads = []
    original = observer.reader.reader

    class Tap:
        def __call__(self):
            begin = time.monotonic()
            value = original()
            reads.append(
                {
                    "start": begin,
                    "end": time.monotonic(),
                    "state": None if value is None else value.as_dict(),
                    "error": original.last_error,
                }
            )
            return value

        @property
        def last_error(self):
            return original.last_error

        def close(self):
            original.close()

    def scheduled(req):
        op = observer.checker._active
        if (
            req["op"] == "state"
            and op is not None
            and op.finished is not None
            and not captured
        ):
            deadline = min(op.deadline, op.finished + lim["settle_timeout_s"])
            remaining = deadline - time.monotonic()
            if 0.0 < remaining < lim["read_timeout_s"] / 3:
                if kind.startswith("fault"):
                    c.fault("canonical controlled fault BEFORE settle deadline")
                dispatch_started = time.monotonic()
                value = dispatch(req)
                dispatch_returned = time.monotonic()
                # MobileBridgeController.state() returns a detached snapshot.
                # Retain that exact reply: a second deepcopy here can collect
                # unrelated suite garbage while a fresh packet is being held.
                captured.append(
                    {
                        "capture": dispatch_returned,
                        "dispatch_started": dispatch_started,
                        "deadline": deadline,
                        "wire": value,
                    }
                )
                captured[-1]["record_ready"] = time.monotonic()
                if kind.endswith("late"):
                    captured[-1]["wait_target"] = deadline + lim["sample_interval_s"]
                    captured[-1]["wait_started"] = time.monotonic()
                    yield_until(captured[-1]["wait_target"])
                captured[-1]["return"] = time.monotonic()
                return value
        return dispatch(req)

    try:
        assert rt.reset_stop()["ok"]
        monkeypatch.setattr(observer.reader, "reader", Tap())
        monkeypatch.setattr(server, "dispatch", scheduled)
        ack = rt.execute(skill, {})
        ack_before = copy.deepcopy(ack)
        proof = await_stop(rt, ack["receipt_id"])
        done = rt.execute(
            "task_done",
            {"success": True, "summary": "canonical late-read CPU TCP probe"},
        )
        current = c.state()["state"]
        record = {
            "ack": ack,
            "proof": proof,
            "task_done": done,
            "captured": captured,
            "reads": reads,
            "actual_controller_state_at_verdict": current,
            "quarantined": observer._quarantined,
            "limits": lim,
            "evidence": "real TCP and synthetic states, NOT Kit/physics",
            "physical_acceptance": False,
        }
        save(tmp_path, kind + "_" + skill, record)
        assert ack == ack_before and ack["physical_stop_verified"] is False
        assert (
            len(captured) == 1 and captured[0]["capture"] < captured[0]["deadline"]
        ), record
        assert all(r["error"] is None for r in reads), record
        assert max(r["end"] - r["start"] for r in reads) < lim["read_timeout_s"], record
        # The retained snapshot must still match the packet actually decoded
        # by the reader. Only the reader's local receipt/age fields may differ.
        packet = captured[0]["wire"]["state"]
        packet_keys = packet.keys() - {"received_monotonic_s", "producer_age_s"}
        packet_json = json.dumps({key: packet[key] for key in packet_keys}, sort_keys=True)
        assert any(
            r["state"] is not None and packet_json == json.dumps(
                {key: r["state"][key] for key in packet_keys}, sort_keys=True
            )
            for r in reads
        ), "retained snapshot differs from the packet actually received"
        assert observer.limits == verifier_limits()
        if kind.endswith("late"):
            assert captured[0]["return"] > captured[0]["deadline"], record
            assert proof["postcondition"]["evidence"]["late_reads"] == 1, record
        if kind == "healthy_late":
            # A large record is truncated by pytest's dict repr, hiding the
            # reason we need when this real-TCP timing probe fails on CI.
            assert proof["status"] == "confirmed" and done["success"], json.dumps(
                {
                    "status": proof["status"],
                    "reason": proof.get("reason"),
                    "task_done_success": done["success"],
                    "postcondition": {
                        key: proof["postcondition"].get(key)
                        for key in ("status", "reason", "metrics")
                    },
                    "evidence": {
                        key: proof["postcondition"]["evidence"].get(key)
                        for key in ("rejected", "attempts", "late_reads", "channel_failed")
                    },
                    "eligible_samples": len(proof["postcondition"]["evidence"]["samples"]),
                    "max_read_duration_s": max(r["end"] - r["start"] for r in reads),
                    "max_received_state_age_s": max(
                        r["state"]["producer_age_s"] + r["end"] - r["state"]["received_monotonic_s"]
                        for r in reads if r["state"] is not None
                    ),
                    "wire_producer_age_s": packet["producer_age_s"],
                    "capture_timing": {key: value for key, value in captured[0].items() if key != "wire"},
                    "late_return_after_deadline_s": captured[0]["return"] - captured[0]["deadline"],
                    "record_path": str(tmp_path / (kind + "_" + skill + ".json")),
                },
                sort_keys=True,
            )
        else:
            assert (
                current["controller_status"]
                == captured[0]["wire"]["state"]["controller_status"]
                == "fault"
            )
            assert proof["status"] != "confirmed" and done["success"] is False, record
    finally:
        rt.close()
        ticks.close()
