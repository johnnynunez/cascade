"""Temporal provenance, stale-packet ownership, and clock failure contracts."""
from fractions import Fraction
from pathlib import Path
import runpy

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/isaac_frame_history.py"))
FrameHistory = MODULE["FrameHistory"]
ClockDiscontinuity = MODULE["ClockDiscontinuity"]
reference_key = MODULE["reference_key"]


def sample(clock=1_700_000, step=204, sim=1.7, t=10., epoch="boot-a"):
    return dict(reference=(clock, 1_000_000), simulation_time=sim, physics_step=step,
                started_monotonic=t, finished_monotonic=t + .1, epoch=epoch,
                payload={"proprioception": {"q": [1., 2.], "gripper_joints": {"position_m": [.01, .02]},
                                            "t": t, "producer_epoch": epoch},
                         "wrist_T": [[1., 0., 0., .3]],
                         "contacts": {"paths": ["/held"], "error": None}})


def resolve(history, clock=1_700_000_000, sim=1.7, epoch="boot-a"):
    return history.resolve(reference=(clock, 1_000_000_000), simulation_time=sim, epoch=epoch)


def test_delayed_camera_uses_its_historical_physics_and_owns_all_state():
    history = FrameHistory()
    first = sample()
    history.record(**first)
    newer = sample(1_716_666, 206, 206 / 120, 11.)
    newer["payload"]["proprioception"]["q"] = [8., 9.]
    newer["payload"]["contacts"]["paths"] = []
    newer["payload"]["wrist_T"][0][3] = .9
    history.record(**newer)
    first["payload"]["proprioception"]["q"][0] = 999
    frame = resolve(history)
    assert frame["physics_step"] == 204 and frame["started_monotonic"] == 10.
    assert frame["payload"]["proprioception"]["q"] == [1., 2.]
    assert frame["payload"]["contacts"]["paths"] == ["/held"]
    assert frame["payload"]["wrist_T"][0][3] == .3
    frame["payload"]["proprioception"]["gripper_joints"]["position_m"][0] = 99
    assert resolve(history)["payload"]["proprioception"]["gripper_joints"]["position_m"] == [.01, .02]


def test_documented_sdk_precision_does_not_reduce_to_a_coarse_fraction_cell():
    history = FrameHistory()
    history.record(**sample(1_650_000, 198, 1.65))  # Fraction reduces to 33/20.
    assert resolve(history, 1_650_000_666, 1.65) is not None
    assert resolve(history, 1_666_666_666, 1.65) is None  # Not a 50 ms bucket.
    assert resolve(history, 1_650_001_000, 1.65) is None
    history.record(**sample(1_716_666, 206, 206 / 120, 11.))
    assert resolve(history, 1_716_666_666, 1.716666666) is not None


def test_simulation_time_cannot_substitute_for_missing_render_product_provenance():
    history = FrameHistory()
    history.record(**sample())
    assert resolve(history, 1_716_666_666, 1.7) is None
    assert resolve(history, sim=1.716666666) is None
    assert resolve(history, epoch="other-boot") is None


def test_identical_repeat_retains_first_capture_anchor_and_payload_timestamp():
    history = FrameHistory()
    assert history.record(**sample()) is True
    assert history.record(**sample(t=15.)) is False
    frame = resolve(history)
    assert frame["started_monotonic"] == 10.
    assert frame["finished_monotonic"] == 10.1
    assert frame["payload"]["proprioception"]["t"] == 10.


@pytest.mark.parametrize("change", ["q", "jaw", "camera", "contact", "contact_error", "other_t", "step"])
def test_conflicting_repeat_is_ambiguous_even_if_original_state_returns(change):
    history = FrameHistory()
    first = sample()
    history.record(**first)
    contradictory = sample(t=11.)
    p = contradictory["payload"]
    if change == "q": p["proprioception"]["q"][0] += 1
    if change == "jaw": p["proprioception"]["gripper_joints"]["position_m"][0] += .01
    if change == "camera": p["wrist_T"][0][3] += .2
    if change == "contact": p["contacts"]["paths"] = []
    if change == "contact_error": p["contacts"]["error"] = "unknown"
    if change == "other_t": p["contacts"]["t"] = 11.
    if change == "step": contradictory["physics_step"] += 1
    assert history.record(**contradictory) is False
    assert resolve(history) is None
    if change != "step":
        assert history.record(**sample(t=12.)) is False
        assert resolve(history) is None


@pytest.mark.parametrize("field,value", [("reference", (1_000_000, 1_000_000)),
    ("reference", (1_700_000_000, 1_000_000_000)), ("simulation_time", .5),
    ("physics_step", 1), ("started_monotonic", 1.), ("finished_monotonic", 10.05), ("epoch", "boot-b")])
def test_clock_discontinuity_clears_old_authority(field, value):
    history = FrameHistory()
    history.record(**sample())
    changed = sample()
    changed[field] = value
    with pytest.raises(ClockDiscontinuity):
        history.record(**changed)
    assert resolve(history) is None
    assert history.record(**sample(1_000_000, 120, 1., 20., "boot-b"))
    assert resolve(history, 1_000_000_000, 1., "boot-b") is not None


def test_bounded_history_never_replaces_an_evicted_pose_with_the_current_one():
    history = FrameHistory(maxlen=2)
    for i in range(3):
        history.record(**sample(1_000_000 + i * 50_000, 120 + i * 6, 1 + i * .05, 10 + i))
    assert resolve(history, 1_000_000_000, 1.) is None
    assert resolve(history, 1_050_000_000, 1.05) is not None
    history.clear()
    assert resolve(history, 1_100_000_000, 1.1) is None


def test_undocumented_clock_precision_allows_only_exact_references():
    history = FrameHistory()
    s = sample(); s["reference"] = (17, 10)
    history.record(**s)
    assert resolve(history) is not None
    assert resolve(history, 1_700_000_001) is None


@pytest.mark.parametrize("bad", [None, (), (1,), (1, 0), (-1, 2), (True, 2), (1.0, 2), (1, False)])
def test_malformed_reference_has_no_pose_fallback(bad):
    history = FrameHistory(); history.record(**sample())
    with pytest.raises(ValueError): reference_key(bad)
    assert history.resolve(reference=bad, simulation_time=1.7, epoch="boot-a") is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., True, "1.7", 1e308])
def test_unusable_simulation_metadata_cannot_authorize_a_frame(bad):
    history = FrameHistory(); history.record(**sample())
    assert resolve(history, sim=bad) is None


def test_reference_identity_stays_exact_independently_of_history_quantization():
    assert reference_key((17, 10)) == Fraction(17, 10)
    assert reference_key((1_700_000_001, 1_000_000_000)) != Fraction(17, 10)


@pytest.mark.parametrize('value', [8.45, 17.15])
def test_sdk_double_encodings_match_without_adjacent_timestamp_search(value):
    history = FrameHistory()
    sdk = int(value * 1_000_000)
    render = int(value * 1_000_000_000)
    assert render // 1000 == sdk - 1  # Actual ARM failure: same double, two encodings.
    history.record(**sample(sdk, round(value * 120), value))
    frame = resolve(history, render, float(Fraction(render, 1_000_000_000)))
    assert frame is not None and frame['physics_step'] == round(value * 120)
    assert frame['started_monotonic'] == 10.
    assert frame['payload']['proprioception']['q'] == [1., 2.]
    assert resolve(history, render - 1, float(Fraction(render - 1, 1_000_000_000))) is None
    # This adjacent value already meets the ordinary microsecond rule.
    assert resolve(history, render + 1, value) is not None
    assert resolve(history, render, value) is None  # Wrong annotator for alias.
    assert resolve(history, render, value + .0000001) is None
    assert resolve(history, render, float(Fraction(render, 1_000_000_000)), 'other') is None


def test_alias_requires_both_raw_sdk_and_render_double_encodings():
    value = 17.15
    render = int(value * 1_000_000_000)
    history = FrameHistory()
    history.record(**sample(int(value * 1_000_000) + 1, 2058, value))
    assert resolve(history, render, render / 1_000_000_000) is None
    history = FrameHistory()
    s = sample(int(value * 1_000_000), 2058, value)
    s['reference'] = (343, 20)  # Same rational identity, undocumented raw precision.
    history.record(**s)
    assert resolve(history, render, render / 1_000_000_000) is None
    history = FrameHistory(); history.record(**sample(17_150_000, 2058, value))
    assert history.resolve(reference=(render * 2, 2_000_000_000),
                           simulation_time=render / 1_000_000_000, epoch='boot-a') is None


@pytest.mark.parametrize('poison', [False, True])
def test_multiple_encoding_candidates_remain_ambiguous_even_if_one_is_poisoned(poison):
    import math
    value = 17.15
    earlier = math.nextafter(value, 0)
    assert int(earlier * 1_000_000_000) == int(value * 1_000_000_000)
    assert int(earlier * 1_000_000) != int(value * 1_000_000)
    history = FrameHistory()
    old = sample(int(earlier * 1_000_000), 2056, earlier)
    history.record(**old)
    if poison:
        old['payload']['proprioception']['q'][0] += 1
        history.record(**old)
    history.record(**sample(int(value * 1_000_000), 2058, value, 11.))
    render = int(value * 1_000_000_000)
    assert resolve(history, render, render / 1_000_000_000) is None


def test_encoding_alias_cannot_bypass_poison_eviction_or_clear():
    history = FrameHistory(maxlen=1)
    s = sample(17_150_000, 2058, 17.15)
    history.record(**s)
    s['payload']['proprioception']['gripper_joints']['position_m'][0] += .01
    history.record(**s)
    assert resolve(history, 17_149_999_999, 17.149999999) is None
    history.record(**sample(17_200_000, 2064, 17.2, 11.))
    assert resolve(history, 17_149_999_999, 17.149999999) is None
    history.clear()
    assert resolve(history, 17_149_999_999, 17.149999999) is None
