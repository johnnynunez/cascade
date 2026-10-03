"""Frozen-clock diagnostic controls; synthetic rows, no native SDK or solves."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace as NS

import pytest

from cascade.apps.factory_runtime import _ready
from cascade.control.fastening import (
    FasteningController, FasteningFault, FasteningWriteGuard, SolveJournal, check_solve,
)
from cascade.skills.fastening_runtime import FasteningDomain
from cascade.sim.factory_owner import FactorySolveOwner
from test_fastening_runtime import ScriptedOwner, armed, binding, limits, row


def diagnostic(error):
    result = getattr(error, "observation_age", None)
    assert result is not None, "freshness fault lost exact capture/check/age/stage provenance"
    assert json.loads(str(error).split("; observation_age=", 1)[1]) == result
    return result


@pytest.mark.parametrize("captured,checked,reason", [(10., 10.25, "stale"), (10.1, 10., "future")])
def test_freshness_fault_retains_frozen_clock_and_bound_without_restamping(captured, checked, reason):
    sample = row(step=7, captured=captured, generation=2)
    original = asdict(sample)
    with pytest.raises(FasteningFault) as caught:
        check_solve(sample, binding(), limits(), checked)
    evidence = diagnostic(caught.value)
    assert evidence == {"stage": "check_solve", "reason": reason,
        "captured_monotonic_s": captured, "checked_monotonic_s": checked,
        "age_s": checked-captured, "max_observation_age_s": .2,
        "step": 7, "simulation_time_s": .07, "generation": 2,
        "epoch": "epoch", "binding_sha256": binding().sha256,
        "model_identity_sha256": binding().model_sha256}
    assert asdict(sample) == original
    evidence["age_s"] = 0.
    assert diagnostic(caught.value)["age_s"] == checked-captured
    assert len(str(caught.value)) < 2048


@pytest.mark.parametrize("age", [0., .1, .2])
def test_exact_original_freshness_boundary_still_accepts(age):
    sample = row(captured=0.)
    original = asdict(sample)
    assert check_solve(sample, binding(), limits(), age) is None
    assert asdict(sample) == original


def test_controller_fault_keeps_stage_and_journal_failure_without_publishing():
    journal = SolveJournal(binding())
    controller = FasteningController(binding(), limits(), journal, synthetic=True,
                                    close_owner=lambda: None, clock=lambda: 10.25)
    sample = row(step=1, captured=10.)
    with pytest.raises(FasteningFault) as caught:
        controller.accept_solve(sample)
    assert diagnostic(caught.value)["stage"] == "controller_accept_solve"
    assert controller.guard.current_permit is None
    assert controller.generation == 1
    with pytest.raises(FasteningFault) as propagated:
        journal.read()
    assert str(propagated.value) == str(caught.value)
    assert not journal._rows


def test_readiness_fault_identifies_passive_reader_without_renewing_capture():
    sample = row(step=1, captured=10.)
    owner = NS(backend=NS(binding=binding(), limits=limits()),
               journal=NS(read=lambda *a, **kw: (sample,)))
    with pytest.raises(FasteningFault) as caught:
        _ready(owner, clock=lambda: 10.25)
    evidence = diagnostic(caught.value)
    assert evidence["stage"] == "startup_readiness"
    assert evidence["age_s"] == .25
    assert sample.captured_monotonic_s == 10.


def test_owner_delayed_decode_preserves_raw_row_then_faults_and_zeroes():
    now = [10.]
    uploads = []
    class Backend:
        synthetic = True
        step = 0
        binding = binding()
        limits = limits()
        def enter_owner(self):
            pass
        def upload_zero(self, effort):
            uploads.append((self.step, effort))
        def advance(self, upload):
            stamp = upload()
            self.step += 1
            captured = now[0]
            sample = row(step=self.step, captured=captured, generation=stamp.generation)
            # Synthetic delay AFTER capture, like readback/encoding/scheduling.
            # No specific cause is assigned to the retained native failure.
            now[0] += .25
            return sample, {"solve": asdict(sample), "raw_retained": True}
    backend = Backend()
    owner = FactorySolveOwner(backend, clock=lambda: now[0])
    owner._start = now[0]
    owner._run()  # One synthetic cycle, immediate controlled fault; no thread/SDK.
    assert backend.step == 1
    assert uploads == [(0, 0.), (1, 0.)]
    assert owner._exit.is_set()
    records = owner.records()
    assert len(records) == 1 and records[0]["solve"]["captured_monotonic_s"] == 10.
    assert owner._row is None
    assert owner._zero_receipt["uploaded"] is True
    assert owner._zero_receipt["physical_stop_verified"] is False
    with pytest.raises(FasteningFault) as caught:
        owner.journal.read()
    retained = json.loads(str(caught.value).split("; observation_age=", 1)[1])
    assert retained["stage"] == "controller_accept_solve"
    assert retained["checked_monotonic_s"] == 10.25 and retained["age_s"] == .25
    assert owner._error.endswith(str(caught.value))
    assert owner.close()["ok"] is False


@pytest.mark.parametrize("checked", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_local_check_clock_remains_a_fault_with_json_safe_provenance(checked):
    with pytest.raises(FasteningFault) as caught:
        check_solve(row(), binding(), limits(), checked)
    evidence = diagnostic(caught.value)
    assert evidence["reason"] == "nonfinite_check_clock"
    assert evidence["checked_monotonic_s"] is None and evidence["age_s"] is None
    assert evidence["captured_monotonic_s"] == 10.
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("operation", ["reset_stop", "turn_admission", "control_upload"])
def test_stale_row_cannot_acquire_or_use_authority_and_identifies_consumer(operation):
    writes = []
    if operation == "reset_stop":
        guard = FasteningWriteGuard(binding(), limits(), clock=lambda: 10.25)
        invoke = lambda: guard.reset_stop(row())
    elif operation == "turn_admission":
        now = [10.]
        guard = FasteningWriteGuard(binding(), limits(), clock=lambda: now[0])
        guard.reset_stop(row())
        now[0] = 10.25
        invoke = lambda: guard.admit(row(generation=1), expected_generation=1)
    else:
        now, guard, current, permit = armed()
        now[0] = 10.25
        invoke = lambda: guard.apply(current, (0.,),
            (current.geometry_min_m, current.geometry_max_m), .01,
            lambda *args: writes.append(args), permit=permit)
    with pytest.raises(FasteningFault) as caught:
        invoke()
    assert diagnostic(caught.value)["stage"] == operation
    assert diagnostic(caught.value)["age_s"] == .25
    assert not writes and guard.current_permit is None


@pytest.mark.parametrize("during_rest", [False, True])
def test_domain_retains_delayed_observer_stage_without_granting_task_credit(during_rest):
    owner = ScriptedOwner()
    def delayed_reader(*args, **kwargs):
        samples = owner.read(*args, **kwargs)
        if owner.stopped == during_rest:
            owner.now += .25
        return samples
    domain = FasteningDomain(owner, delayed_reader, controller_id="fixture",
                             clock=lambda: owner.now)
    result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
    assert result["ok"] is False and result["verified"] is False
    assert result["physical_stop_verified"] is False and owner.stopped
    evidence = json.loads(result["error"].split("; observation_age=", 1)[1])
    assert evidence["stage"] == ("rest_observation" if during_rest else "turn_observation")
    assert evidence["age_s"] == .25


def test_identity_veto_still_precedes_age_and_error_payload_is_bounded():
    with pytest.raises(FasteningFault, match="identity") as caught:
        check_solve(row(binding_sha256="f"*64), binding(), limits(), 10.25)
    assert not hasattr(caught.value, "observation_age")
    with pytest.raises(FasteningFault) as caught:
        check_solve(replace(row(), epoch="\U0001f600"*512), binding(), limits(), 10.25)
    assert len(str(caught.value).encode()) < 8192
    assert diagnostic(caught.value)["epoch"] == "\U0001f600"*512
