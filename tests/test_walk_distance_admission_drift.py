"""B74: opt-in bound on the lateral/heading change a ``walk_distance`` baseline absorbs.

B67 (PR #289) pinned the cause: ``SafeBase`` integrates lateral/heading drift from
the first completed post-ACK sample (the distance baseline), so a constant offset
first seen AT that baseline never enters the drift veto and the walk completes.
Posture is already checked on every sample.

Opt-in ``distance_control.max_admission_lateral_m`` / ``max_admission_heading_rad``
(both or neither; an explicit null pair is OFF) bound the change from the last
pre-ACK sample (``start``) through every delivery sample to that baseline, with
the walk's own midpoint-heading increments, and latch like the existing veto.
Forward delivery motion stays out of ``measured_distance`` and is not bounded.

The independent checker has the same structure (premise below): it integrates
from its own baseline, which may be the first admitted sample up to
``max_sample_gap_s`` after the ACK's admission clock. The same two keys in
``verifier`` refute a confirmation whose change from the last independent sample
at or before the admission clock to that baseline exceeds them; a runtime that
sets the pair on only one layer has no independent verifier.

Absent (or null) = byte-identical behaviour: golden digests recorded on main
38f6d08. Kinematic fixtures prove no gait; every SafeBase result stays
``unverified``. Every fixture uses explicit steps (``auto_step=False``) and keys
its faults to the fixture's own simulated step, never to a read count.
"""
import hashlib
import json
import math
from dataclasses import replace

import pytest

from cascade.agent.base_effects import BasePostconditionChecker, _Window
from cascade.control.mock_base import MockMobileBase
from test_mobile_effects import checker_for, fixture_motion_receipt, limits as verifier_limits, state
from test_walk_distance_control import _SteppedDistanceMock, configured, distance_limits

ADMISSION = dict(max_admission_lateral_m=.005, max_admission_heading_rad=.04)
OFF = dict(max_admission_lateral_m=None, max_admission_heading_rad=None)
VETO = "distance control admission change exceeded before the distance baseline"
KEYS = ("measured_admission_lateral_m", "measured_admission_heading_rad")

_VOLATILE = {"epoch", "received_monotonic_s", "producer_age_s"}


def _canonical(value):
    """Wall clocks and the random epoch removed; floats to 12 significant digits
    so a last-ulp libm difference between CI platforms cannot move a digest."""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value


def _digest(value):
    text = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _yaw_q(yaw):
    return (math.cos(yaw/2), 0., 0., math.sin(yaw/2))


# --- SafeBase fixtures: ten explicit .002 steps per read (post-ACK sample k = step admission+10k) ---

class _Offset(_SteppedDistanceMock):
    """A constant lateral/heading offset from post-ACK sample ``onset`` on (1 = the baseline)."""

    def __init__(self, *, onset, lateral=0., yaw=0., **kwargs):
        super().__init__(**kwargs)
        self.onset, self.lateral, self.yaw, self.admitted_step = onset, lateral, yaw, None

    def command_velocity(self, command, *, generation):
        ack = super().command_velocity(command, generation=generation)
        if ack.get("accepted") is True:
            self.admitted_step = self._step
        return ack

    def get_state(self):
        state = super().get_state()
        if self.admitted_step is None or state.step < self.admitted_step + 10 * self.onset:
            return state
        x, y, z = state.position_world
        return replace(state, position_world=(x, y + self.lateral, z),
                       orientation_wxyz=_yaw_q(self.yaw) if self.yaw else state.orientation_wxyz)


class _DelayedAdmission(_SteppedDistanceMock):
    """The backend admits the command ``delay_steps`` after dispatch (a delivery window).

    Post-ACK samples 1 and 2 sit at or before the ACK's admission clock (delivery,
    never progress); sample 3 is the distance baseline. ``excursion`` adds a
    lateral offset to post-ACK sample 1 only: it is gone again by the baseline.
    """

    def __init__(self, *, delay_steps=25, excursion=0., **kwargs):
        super().__init__(**kwargs)
        self.delay_steps, self.excursion, self.dispatch_step = delay_steps, excursion, None

    def command_velocity(self, command, *, generation):
        ack = super().command_velocity(command, generation=generation)
        if ack.get("accepted") is True:
            self.dispatch_step = self._step
            shift = self.delay_steps * self._dt
            ack = {**ack, "start_sim_time_s": ack["start_sim_time_s"] + shift,
                   "end_sim_time_s": ack["end_sim_time_s"] + shift}
        return ack

    def get_state(self):
        state = super().get_state()
        if self.dispatch_step is not None and state.step == self.dispatch_step + 10:
            x, y, z = state.position_world
            return replace(state, position_world=(x, y + self.excursion, z))
        return state


class _ForwardDelivery(_SteppedDistanceMock):
    """A .02 m forward jump at dispatch (delivery), then optionally the mock's own walk."""

    def __init__(self, *, walks, **kwargs):
        super().__init__(**kwargs)
        self.walks, self.dispatched = walks, False

    def command_velocity(self, command, *, generation):
        self.dispatched = True
        return super().command_velocity(command, generation=generation)

    def get_state(self):
        state = super().get_state()
        x, y, z = state.position_world
        x = (x if self.walks else 0.) + (.02 if self.dispatched else 0.)
        return replace(state, position_world=(x, y, z),
                       linear_velocity_world=state.linear_velocity_world if self.walks else (0., 0., 0.))


class _LowStart(_SteppedDistanceMock):
    def get_state(self):
        state = super().get_state()
        return replace(state, position_world=(*state.position_world[:2], .04))


def _mock(cls=_SteppedDistanceMock, **kwargs):
    return cls(wall_lease_s=2., dt_s=.002, auto_step=False, **kwargs)


def _walk(raw, distance=.02, **control):
    safe = configured(raw, **control)
    safe.connect()
    try:
        result = safe.walk_distance(distance)
        return result, raw.get_state().latched, safe.latched
    finally:
        safe.disconnect()


SCENARIOS = {
    "forward": lambda: (_mock(), .02),
    "reverse": lambda: (_mock(), -.02),
    "lateral_at_baseline": lambda: (_mock(_Offset, onset=1, lateral=.02), .02),
    "heading_at_baseline": lambda: (_mock(_Offset, onset=1, yaw=.2), .02),
    "lateral_after_baseline": lambda: (_mock(_Offset, onset=3, lateral=.02), .02),
    "delivery_excursion": lambda: (_mock(_DelayedAdmission, excursion=.02), .02),
    "forward_delivery_inert": lambda: (_mock(_ForwardDelivery, walks=False), .02),
}

# Recorded on an export of main 38f6d08 (flag absent) with ``_digest`` above.
GOLDEN = {
    "forward": "6d5dd927d2a5e18777927bd60536c9ccf65630eca52a0810e6e610601bf75ed3",
    "reverse": "d37055e4e2eee081b75940c44b53df2d5b1775be7c7817e0edca5d09b5766e27",
    "lateral_at_baseline": "68a743d621be6cccecf0aafbc156f5bcdd465be9f7b337dca2d6cdede43ee1fd",
    "heading_at_baseline": "c941abe029fc0574227d701f0ea5394d7224593c9c84444767e35c000a78c310",
    "lateral_after_baseline": "3c05e460af136739a0dbb5a0a96ce2d6ce5c44a1ded48d44ed1fcd6e3e5f9c4b",
    "delivery_excursion": "2c89771a0b304a9f682fa34a26e68dd212984e797bee82686be2bcc16cd18832",
    "forward_delivery_inert": "0f0bf97e43c22c9a6bf0f8e8c01fbc1c1e23594ff503f5553bfedd3d444dd810",
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_golden_flag_absent_is_byte_identical_to_main(scenario):
    raw, distance = SCENARIOS[scenario]()
    result, raw_latched, safe_latched = _walk(raw, distance)
    assert not any(key in result for key in KEYS)
    assert _digest([result, raw_latched, safe_latched]) == GOLDEN[scenario], _canonical(result)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_explicit_null_pair_is_the_flag_off_contract(scenario):
    # An ``extends:`` child can A/B the bound away with two nulls.
    raw, distance = SCENARIOS[scenario]()
    result, raw_latched, safe_latched = _walk(raw, distance, **OFF)
    assert _digest([result, raw_latched, safe_latched]) == GOLDEN[scenario]


def test_flag_off_keeps_the_exact_seven_key_contract():
    expected = distance_limits()
    for control in ({}, OFF):
        safe = configured(MockMobileBase(wall_lease_s=2.), **control)
        assert dict(safe.distance_control) == expected


@pytest.mark.parametrize("updates,stored", [
    (ADMISSION, ADMISSION),
    (dict(max_admission_lateral_m=.01, max_admission_heading_rad=.08),   # equal to the drift bounds
     dict(max_admission_lateral_m=.01, max_admission_heading_rad=.08)),
    (dict(max_admission_lateral_m=1, max_admission_heading_rad=.04,
          max_lateral_drift_m=2), dict(max_admission_lateral_m=1., max_admission_heading_rad=.04)),
])
def test_admission_bounds_are_stored_as_validated_reals(updates, stored):
    safe = configured(MockMobileBase(wall_lease_s=2.), **updates)
    for key, value in stored.items():
        assert safe.distance_control[key] == value and type(safe.distance_control[key]) is float


@pytest.mark.parametrize("updates", [
    dict(max_admission_lateral_m=.005),                                    # one of the pair
    dict(max_admission_heading_rad=.04),
    dict(max_admission_lateral_m=None),                                    # a lone null
    dict(max_admission_lateral_m=.005, max_admission_heading_rad=None),    # half switched off
    dict(max_admission_lateral_m=None, max_admission_heading_rad=.04),
    dict(max_admission_lateral_m=0., max_admission_heading_rad=.04),       # not positive
    dict(max_admission_lateral_m=.005, max_admission_heading_rad=-.01),
    dict(max_admission_lateral_m=.0101, max_admission_heading_rad=.04),    # looser than the walk's bound
    dict(max_admission_lateral_m=.005, max_admission_heading_rad=.0801),
    dict(max_admission_lateral_m=float("nan"), max_admission_heading_rad=.04),
    dict(max_admission_lateral_m=.005, max_admission_heading_rad=float("inf")),
    dict(max_admission_lateral_m=True, max_admission_heading_rad=.04),
    dict(max_admission_lateral_m=.005, max_admission_heading_rad="0.04"),
    dict(max_admission_lateral_drift_m=.005, max_admission_heading_rad=.04),  # unknown key
    dict(max_admission_drift_m=.005),                                      # unknown key alone
    dict(max_admission_lateral_m=.005, max_admission_heading_rad=.04,
         max_admission_forward_m=.01),                                     # no forward bound exists
])
def test_admission_bounds_are_validated_like_the_existing_bounds(updates):
    with pytest.raises(ValueError):
        configured(MockMobileBase(wall_lease_s=2.), **updates)


@pytest.mark.parametrize("sign", [1., -1.])
@pytest.mark.parametrize("fault", ["lateral", "heading"])
def test_flag_on_vetoes_a_change_first_seen_at_the_baseline_and_latches(fault, sign):
    offset = dict(lateral=sign*.02) if fault == "lateral" else dict(yaw=sign*.2)
    result, raw_latched, safe_latched = _walk(_mock(_Offset, onset=1, **offset), **ADMISSION)
    assert not result["execution_ok"] and result["error"] == VETO, result
    samples = result["measured"]["samples"]
    assert all(b["step"] - a["step"] == 10 for a, b in zip(samples, samples[1:]))
    assert len(samples) == 3  # first, start, then the faulted would-be baseline
    assert result["distance_baseline"] is None and result["measured_distance_m"] is None
    if fault == "lateral":
        assert result["measured_admission_lateral_m"] == sign*.02
        assert result["measured_admission_heading_rad"] == 0.
    else:
        # The .002 m walked from start to the baseline, seen at the .1 rad midpoint heading.
        assert result["measured_admission_lateral_m"] == pytest.approx(-math.sin(sign*.1) * .002, rel=0, abs=1e-12)
        assert result["measured_admission_heading_rad"] == pytest.approx(sign*.2, rel=0, abs=1e-12)
    assert result["stop_ack"]["latched"] and raw_latched and safe_latched
    assert result["ok"] is False and result["outcome"] == "unverified"


def test_flag_on_bounds_every_delivery_sample_not_only_the_baseline():
    # Flag off: the excursion sits in the delivery window and is absorbed; the walk completes.
    result, raw_latched, _ = _walk(_mock(_DelayedAdmission, excursion=.02))
    samples = result["measured"]["samples"]
    assert result["execution_ok"] and not raw_latched, result
    assert samples[2]["position_world"][1] == .02 and samples[2]["sim_time_s"] <= result["ack"]["start_sim_time_s"]
    assert samples[3]["sim_time_s"] <= result["ack"]["start_sim_time_s"] < samples[4]["sim_time_s"]
    assert result["distance_baseline"] == samples[4] and result["measured_lateral_m"] == 0.
    # Flag on: vetoed on the first delivery sample, although the offset is gone by the baseline.
    result, raw_latched, safe_latched = _walk(_mock(_DelayedAdmission, excursion=.02), **ADMISSION)
    assert not result["execution_ok"] and result["error"] == VETO, result
    assert len(result["measured"]["samples"]) == 3 and result["distance_baseline"] is None
    assert result["measured_admission_lateral_m"] == .02
    assert raw_latched and safe_latched


def test_flag_on_delivery_samples_and_forward_delivery_earn_no_progress():
    result, raw_latched, _ = _walk(_mock(_DelayedAdmission), **ADMISSION)
    assert result["execution_ok"] and not raw_latched, result
    samples, baseline = result["measured"]["samples"], result["distance_baseline"]
    start = samples[1]
    assert baseline == samples[4]
    delivered = baseline["position_world"][0] - start["position_world"][0]
    assert delivered == pytest.approx(.006, rel=0, abs=1e-9)  # three sample intervals of .1 m/s, uncredited
    assert abs(result["measured_distance_m"] - .02) <= .002
    assert samples[-1]["position_world"][0] - start["position_world"][0] == pytest.approx(
        delivered + result["measured_distance_m"], rel=0, abs=1e-12)
    assert result["measured_admission_lateral_m"] == 0. and result["measured_admission_heading_rad"] == 0.


@pytest.mark.parametrize("walks", [False, True])
def test_flag_on_forward_delivery_jump_is_neither_credited_nor_vetoed(walks):
    result, raw_latched, _ = _walk(_mock(_ForwardDelivery, walks=walks), **ADMISSION)
    samples, start = result["measured"]["samples"], result["measured"]["samples"][1]
    assert result["distance_baseline"]["position_world"][0] - start["position_world"][0] >= .02
    assert result["measured_admission_lateral_m"] == 0. and result["measured_admission_heading_rad"] == 0.
    if walks:
        assert result["execution_ok"] and not raw_latched, result
        assert abs(result["measured_distance_m"] - .02) <= .002
        assert samples[-1]["position_world"][0] - start["position_world"][0] > .04 - .002
    else:
        assert not result["execution_ok"] and "did not reach" in result["error"], result
        assert result["measured_distance_m"] == 0. and raw_latched


def test_flag_on_below_bound_completes_and_reports_the_absorbed_segment():
    result, raw_latched, _ = _walk(_mock(_Offset, onset=1, lateral=.004, yaw=.03), **ADMISSION)
    assert result["execution_ok"] and not raw_latched, result
    # Midpoint-heading frame (.015 rad) of the .002 m walked plus the .004 m offset.
    assert result["measured_admission_lateral_m"] == pytest.approx(
        -math.sin(.015) * .002 + math.cos(.015) * .004, rel=0, abs=1e-12)
    assert result["measured_admission_heading_rad"] == pytest.approx(.03, rel=0, abs=1e-12)
    assert result["distance_baseline"] == result["measured"]["samples"][2]
    assert abs(result["measured_lateral_m"]) < .001 and result["measured_heading_rad"] == 0.
    assert list(result)[-2:] == list(KEYS)


def test_flag_on_after_the_baseline_the_existing_veto_still_decides():
    result, _, safe_latched = _walk(_mock(_Offset, onset=3, lateral=.02), **ADMISSION)
    assert not result["execution_ok"] and "lateral/heading drift exceeded" in result["error"], result
    assert result["error"] != VETO and result["distance_baseline"] == result["measured"]["samples"][2]
    assert result["measured_admission_lateral_m"] == 0. and safe_latched


def test_flag_on_no_admitted_feedback_is_not_zero_change():
    result, raw_latched, _ = _walk(_mock(_LowStart), **ADMISSION)
    assert not result["execution_ok"] and "posture bound" in result["error"], result
    assert result["measured_admission_lateral_m"] is None and result["measured_admission_heading_rad"] is None
    assert "ack" not in result and raw_latched


class _StopOnSample(_Offset):
    """Runs ``on_sample`` (a priority stop) when post-ACK sample ``stop_at`` is read."""

    def __init__(self, *, stop_at, **kwargs):
        super().__init__(**kwargs)
        self.stop_at, self.on_sample = stop_at, None

    def get_state(self):
        state = super().get_state()
        if self.admitted_step is not None and state.step == self.admitted_step + 10 * self.stop_at:
            self.on_sample()
        return state


def test_flag_on_a_cancelled_walk_still_reports_the_segment():
    raw = _mock(_StopOnSample, stop_at=2, onset=1, lateral=.003)
    safe = configured(raw, **ADMISSION)
    raw.on_sample = lambda: safe.stop(latch=True)
    safe.connect()
    try:
        result = safe.walk_distance(.02)
    finally:
        safe.disconnect()
    assert not result["execution_ok"] and result["error"] == "cancelled by stop", result
    assert "stop_ack" not in result and result["ack"]["accepted"] is True
    assert result["distance_baseline"] == result["measured"]["samples"][2]
    assert result["measured_admission_lateral_m"] == .003 and result["measured_admission_heading_rad"] == 0.


@pytest.mark.parametrize("skill", ["turn", "walk_velocity"])
def test_flag_changes_nothing_outside_walk_distance(skill):
    def run(**control):
        raw = _mock()
        safe = configured(raw, **control)
        safe.connect()
        try:
            return (safe.turn(.1) if skill == "turn" else safe.walk_velocity(.05, 0., 0., .1))
        finally:
            safe.disconnect()
    on, off = run(**ADMISSION), run()
    assert not any(key in on for key in KEYS)
    assert on["execution_ok"] and _canonical(on) == _canonical(off)


def test_shipped_profiles_leave_the_bound_unset():
    # No threshold is invented: every shipped base profile keeps the flag absent on both layers.
    from cascade.config import CONFIG_DIR, load_profile

    names = sorted(p.stem for p in (CONFIG_DIR / "bases").glob("*.yaml"))
    assert "microduck_distance_candidate" in names
    for name in names:
        profile = load_profile("bases", name).as_dict()
        for block in ("distance_control", "verifier"):
            assert not set(profile.get(block) or {}) & set(ADMISSION), (name, block)


# --- independent checker: same structure, same opt-in pair in ``verifier`` ---

CHECK = dict(max_admission_lateral_m=.002, max_admission_heading_rad=.02)  # fixture drift bounds: .003 / .04


def _verdict(*, start=.03, lateral=0., yaw=0., onset=2, transform=None, **limit_updates):
    """The real acceptance/evaluation path on explicit completed clocks (sim .02 per sample).

    ``start`` is the ACK's admission clock. With .03 the checker's baseline is the
    first admitted sample (n=2, .04); n=1 (.02) is the last sample before it. The
    offset applies from sample ``onset`` on.
    """
    checker = checker_for(lambda: None, **limit_updates)
    op = _Window("fixture", "walk_distance", {"distance_m": .01}, started=100., deadline=103.)
    result = fixture_motion_receipt()
    result["ack"].update(start_sim_time_s=start, end_sim_time_s=start + 3.)
    for n in range(1, 14):
        x = min(n - 1, 5) * .002
        stamp = 100. + n * .002 if n < 9 else 100.2 + (n - 8) * .002
        if n == 9:
            op.finished = 100.2
        shifted = n >= onset
        value = state(n, generation=0 if n == 1 else 1 if n <= 7 else 2,
                      position_world=(x, lateral if shifted else 0., .3),
                      orientation_wxyz=_yaw_q(yaw if shifted else 0.), received_monotonic_s=stamp + .001)
        if transform is not None:
            value = transform(n, value)
        op.attempts += 1
        checker._accept(op, value, stamp, stamp + .001)
    verdict = checker._receipt(op, result)
    checker._evaluate(op, verdict)
    checker.close()
    return verdict


def _first_admitted(n, value):
    return replace(value, generation=1) if n == 1 else value


CHECKER_SCENARIOS = {
    "exact_admission_clock": dict(start=.02),
    "first_admitted_baseline": dict(),
    "lateral_at_baseline": dict(lateral=.006),
    "heading_at_baseline": dict(yaw=.06),
    "lateral_after_baseline": dict(lateral=.006, onset=3),
    "no_sample_before_admission": dict(start=.01, transform=_first_admitted),
}

CHECKER_GOLDEN = {
    "exact_admission_clock": "ed67caf83012a6cdeb2c40e8a23a22f82456e7a5c93c86272f0e6ef1c33b6977",
    "first_admitted_baseline": "987a8a9088da07a863deb497d951b6f11e01969b6244733fafc90faa18fb70b0",
    "lateral_at_baseline": "791bffbea9c06415deb585c283a8137e7839f55fdc54cecbfc60906231e001fd",
    "heading_at_baseline": "3e5243b4be26e4b3338e070a9071e4d3382e6ac070db67c8ad14d297e21f1842",
    "lateral_after_baseline": "a08a36eb15927bb6f2626cc90fb0bdda0eb7890c7cfa3694c9a044b53e0bd297",
    "no_sample_before_admission": "d1c0a97cf15bce6c0cbd6ccec8e44eb4c379890cae970c49af87b39f0ce25206",
}


@pytest.mark.parametrize("fault", ["lateral", "heading"])
def test_premise_checker_absorbs_a_change_first_seen_at_its_baseline(fault):
    """Passes on main by design: the evidence that the checker needs the bound too.

    A .006 m / .06 rad offset (fixture drift bounds .003 / .04) that first appears
    on the first admitted sample after the admission clock is part of the
    checker's baseline and confirms; one sample later it is refuted.
    """
    offset = dict(lateral=.006) if fault == "lateral" else dict(yaw=.06)
    absorbed = _verdict(**offset)
    assert absorbed["status"] == "confirmed", absorbed["reason"]
    interval = absorbed["evidence"]["effect_interval"]
    assert interval["baseline_step"] == 2 and interval["baseline_sim_time_s"] - interval["admitted_start_sim_time_s"] \
        <= verifier_limits()["max_sample_gap_s"]
    refuted = _verdict(onset=3, **offset)
    assert refuted["status"] == "refuted"
    assert ("vy drift" if fault == "lateral" else "wz drift") in refuted["reason"]


@pytest.mark.parametrize("scenario", list(CHECKER_SCENARIOS))
@pytest.mark.parametrize("limits", ["absent", "null"])
def test_checker_golden_flag_absent_is_byte_identical_to_main(scenario, limits):
    verdict = _verdict(**CHECKER_SCENARIOS[scenario], **({} if limits == "absent" else OFF))
    assert "admission_drift" not in verdict["evidence"] and set(verdict["limits"]) == set(verifier_limits())
    assert _digest(verdict) == CHECKER_GOLDEN[scenario], (verdict["status"], verdict["reason"])


@pytest.mark.parametrize("sign", [1., -1.])
@pytest.mark.parametrize("fault", ["lateral", "heading"])
def test_checker_flag_on_refutes_the_change_its_baseline_absorbs(fault, sign):
    offset = dict(lateral=sign*.006) if fault == "lateral" else dict(yaw=sign*.06)
    verdict = _verdict(**offset, **CHECK)
    assert verdict["status"] == "refuted"
    assert verdict["reason"] == "admission drift before the distance baseline exceeds bound"
    drift = verdict["evidence"]["admission_drift"]
    assert (drift["reference_step"], drift["baseline_step"]) == (1, 2)
    assert drift["reference_sim_time_s"] <= verdict["evidence"]["effect_interval"]["admitted_start_sim_time_s"]
    if fault == "lateral":
        assert drift["lateral_m"] == pytest.approx(sign*.006, rel=0, abs=1e-12) and drift["heading_rad"] == 0.
    else:
        assert drift["heading_rad"] == pytest.approx(sign*.06, rel=0, abs=1e-12)
    assert verdict["limits"]["max_admission_lateral_m"] == .002
    # The post-baseline outcome is unchanged: only the absorbed segment refutes.
    assert verdict["metrics"]["distance_outcome"]["yaw_change_rad"] == pytest.approx(0., rel=0, abs=1e-12)


def test_checker_segment_starts_at_the_admission_clock_not_at_its_first_sample():
    # Two samples precede admission (.05): a shift between them is standing, outside the
    # command and outside what the baseline (n=3, .06) absorbs; the reference is n=2.
    def standing(n, value):
        return replace(value, generation=0) if n <= 2 else value
    verdict = _verdict(start=.05, lateral=.006, transform=standing, **CHECK)
    assert verdict["status"] == "confirmed", verdict["reason"]
    drift = verdict["evidence"]["admission_drift"]
    assert (drift["reference_step"], drift["baseline_step"]) == (2, 3)
    assert drift["lateral_m"] == 0. and drift["heading_rad"] == 0.


def test_checker_flag_on_below_bound_and_exact_clock_confirm():
    below = _verdict(lateral=.0015, yaw=.015, **CHECK)
    assert below["status"] == "confirmed", below["reason"]
    drift = below["evidence"]["admission_drift"]
    assert drift["lateral_m"] == pytest.approx(-math.sin(.0075) * .002 + math.cos(.0075) * .0015, rel=0, abs=1e-12)
    assert drift["heading_rad"] == pytest.approx(.015, rel=0, abs=1e-12)
    exact = _verdict(start=.02, **CHECK)
    assert exact["status"] == "confirmed", exact["reason"]
    assert exact["evidence"]["admission_drift"] == {
        "reference_step": 1, "reference_sim_time_s": .02, "baseline_step": 1, "baseline_sim_time_s": .02,
        "lateral_m": 0., "heading_rad": 0.}
    # At the exact clock nothing is absorbed: a later offset is the existing drift veto's.
    later = _verdict(start=.02, lateral=.006, **CHECK)
    assert later["status"] == "refuted" and "vy drift" in later["reason"]


def test_checker_flag_on_without_an_admission_reference_stays_unverified():
    off = _verdict(start=.01, transform=_first_admitted)
    assert off["status"] == "confirmed", off["reason"]
    on = _verdict(start=.01, transform=_first_admitted, **CHECK)
    assert on["status"] == "unverified"
    assert on["reason"] == "no independent sample at or before the admission clock for the admission drift bound"
    assert on["evidence"]["admission_drift"] is None


def test_checker_flag_on_only_turns_a_confirmation_into_a_refutation():
    # Missing rest evidence stays unverified with its own reason; the bound adds no verdict there.
    verdict = _verdict(lateral=.006, min_settle_samples=10, **CHECK)
    assert verdict["status"] == "unverified"
    assert verdict["reason"] == "insufficient advancing states captured after outcome"
    assert verdict["evidence"]["admission_drift"]["lateral_m"] == pytest.approx(.006, rel=0, abs=1e-12)
    missing = _verdict(start=.01, transform=_first_admitted, min_settle_samples=10, **CHECK)
    assert missing["reason"] == "insufficient advancing states captured after outcome"


@pytest.mark.parametrize("updates", [
    dict(max_admission_lateral_m=.002),
    dict(max_admission_heading_rad=.02),
    dict(max_admission_lateral_m=None),
    dict(max_admission_lateral_m=.002, max_admission_heading_rad=None),
    dict(max_admission_lateral_m=0., max_admission_heading_rad=.02),
    dict(max_admission_lateral_m=.002, max_admission_heading_rad=-.02),
    dict(max_admission_lateral_m=.0031, max_admission_heading_rad=.02),
    dict(max_admission_lateral_m=.002, max_admission_heading_rad=.041),
    dict(max_admission_lateral_m=float("nan"), max_admission_heading_rad=.02),
    dict(max_admission_lateral_m=.002, max_admission_heading_rad=True),
    dict(max_admission_drift_m=.002),
])
def test_checker_admission_limits_are_validated(updates):
    with pytest.raises(ValueError):
        BasePostconditionChecker(lambda: None, limits=verifier_limits(**updates))


def test_checker_admission_limits_at_the_drift_bounds_are_stored():
    checker = BasePostconditionChecker(lambda: None, limits=verifier_limits(
        max_admission_lateral_m=.003, max_admission_heading_rad=.04))
    try:
        verdict = checker.finish(checker.begin("walk_distance", {"distance_m": .01}), {})
    finally:
        checker.close()
    assert verdict["limits"]["max_admission_lateral_m"] == .003
    assert verdict["limits"]["max_admission_heading_rad"] == .04


@pytest.mark.parametrize("sides", [("distance_control",), ("verifier",), ("distance_control", "verifier"), ()])
def test_runtime_builds_the_independent_verifier_only_with_the_flag_on_both_layers(tmp_path, sides):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.config import load_demo_config
    from test_mobile_runtime import verifier_limits as runtime_verifier_limits

    cfg = load_demo_config(base="microduck_mock")
    profile = cfg._data["bases"][0]
    profile["distance_control"] = distance_limits()
    profile["verifier"] = runtime_verifier_limits()
    bounds = dict(max_admission_lateral_m=.002, max_admission_heading_rad=.02)
    for side in sides:
        profile[side].update(bounds)
    rt, rig = build_mobile_runtime(cfg, tmp_path)
    try:
        checker = rt.checkers.get("microduck_mock")
        if len(sides) == 1:
            assert checker is None
            assert "admission drift bound must be set in both distance_control and verifier" in \
                rt.verifier_errors["microduck_mock"]
        else:
            assert isinstance(checker, BasePostconditionChecker) and "microduck_mock" not in rt.verifier_errors
            assert ("max_admission_lateral_m" in checker._limits) == bool(sides)
            assert rig.get(None).distance_control.get("max_admission_lateral_m") == (.002 if sides else None)
    finally:
        rt.close()
