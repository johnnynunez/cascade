"""Derived envelope features (ROADMAP follow-up #4, Harness-VLA).

`memory/envelope.py` learned its operating ranges over the RAW skill
arguments (`place_at.x`, `grasp_object` has none), which is a proxy for the
constraint that actually bites on the reBot B601-RS: where the TCP ends up
when the jaws close, how tall and wide the thing under them is, and how far
the tool landed from the object the detector pointed at. These tests pin the
contract for features the RUNTIME measures at call time:

* they are recorded from the runtime's own measurements (FK of the closed
  jaw pose, the localized object's cloud), never from the agent's arguments;
* a feature that could not be measured on a call is counted as MISSING for
  that call -- never defaulted, never interpolated;
* `envelope_digest()` / `export_markdown()` show them, marked as measured;
* the three-tier confidence and `contradictions` behaviour of
  tests/test_envelope_confidence.py is unchanged.
"""

from __future__ import annotations

import json
import time

import pytest

from conftest import needs_pin

from cascade.memory.envelope import OperatingEnvelope


# ── unit level: the envelope's own contract ───────────────────────────────


def test_grasp_skills_declare_the_four_derived_features():
    from cascade.memory.envelope import DERIVED_FEATURES

    expected = {"tcp_z_at_grasp_m", "object_height_m", "object_width_m",
                "object_tcp_lateral_offset_m"}
    for skill in ("grasp_object", "pick_and_place", "grasp_at_pixel"):
        assert set(DERIVED_FEATURES[skill]) == expected, skill
    # a skill that never grasps declares nothing: no phantom "missing" rows
    assert "move_home" not in DERIVED_FEATURES and "wave" not in DERIVED_FEATURES


def test_measured_features_are_learned_like_any_other_span():
    env = OperatingEnvelope(min_support=2)
    env.record("grasp_object", {"label": "red cube"}, ok=True,
               measured={"tcp_z_at_grasp_m": 0.046, "object_height_m": 0.05,
                         "object_width_m": 0.055, "object_tcp_lateral_offset_m": 0.002})
    env.record("grasp_object", {"label": "red cube"}, ok=True,
               measured={"tcp_z_at_grasp_m": 0.052, "object_height_m": 0.05,
                         "object_width_m": 0.055, "object_tcp_lateral_offset_m": 0.004})
    st = env.stats()["grasp_object"]
    span = st["spans"]["tcp_z_at_grasp_m"]
    assert span["n"] == 2 and span["lo"] == pytest.approx(0.046) and span["hi"] == pytest.approx(0.052)
    assert "tcp_z_at_grasp_m" in st["derived"], "a measured feature must be marked as derived"
    # a raw arg is NOT marked derived
    env.record("place_at", {"x": 0.2, "y": -0.1}, ok=True)
    assert env.stats()["place_at"]["derived"] == []


def test_a_feature_not_measured_on_a_call_is_recorded_as_missing_never_defaulted():
    env = OperatingEnvelope(min_support=2)
    # The grasp failed at IK: the object was localized (height/width known)
    # but the jaws never closed, so TCP z and the lateral offset do not exist.
    env.record("grasp_object", {"label": "red cube"}, ok=False, error="no IK solution",
               measured={"object_height_m": 0.05, "object_width_m": 0.055})
    st = env.stats()["grasp_object"]
    assert st["missing"]["tcp_z_at_grasp_m"] == 1
    assert st["missing"]["object_tcp_lateral_offset_m"] == 1
    assert "object_height_m" not in st["missing"]
    # nothing was invented for the absent features
    assert "tcp_z_at_grasp_m" not in st["spans"]
    assert "object_tcp_lateral_offset_m" not in st["spans"]


def test_none_and_non_finite_measurements_count_as_missing():
    env = OperatingEnvelope()
    env.record("grasp_object", {"label": "cube"}, ok=True,
               measured={"tcp_z_at_grasp_m": None, "object_height_m": float("nan"),
                         "object_width_m": 0.05, "object_tcp_lateral_offset_m": 0.0})
    st = env.stats()["grasp_object"]
    assert st["missing"] == {"tcp_z_at_grasp_m": 1, "object_height_m": 1}
    assert set(st["spans"]) == {"object_width_m", "object_tcp_lateral_offset_m"}


def test_a_record_without_a_measurement_channel_is_not_counted_as_missing():
    """`measured=None` (an old trace replayed by `ingest_trace`, a caller that
    predates this feature) means UNKNOWN, which is not the same as missing."""
    env = OperatingEnvelope()
    env.record("grasp_object", {"label": "cube"}, ok=True)
    assert env.stats()["grasp_object"]["missing"] == {}


def test_measured_features_feed_the_contradiction_signal():
    env = OperatingEnvelope(min_support=2)
    for z in (0.045, 0.050, 0.055):
        env.record("grasp_object", {"label": "cube"}, ok=True, measured={"tcp_z_at_grasp_m": z})
    env.record("grasp_object", {"label": "cube"}, ok=False, error="air grasp: closed on nothing",
               measured={"tcp_z_at_grasp_m": 0.050})
    assert env.stats()["grasp_object"]["spans"]["tcp_z_at_grasp_m"]["contradictions"] == 1


def test_check_consults_measured_features_and_stays_advisory():
    env = OperatingEnvelope(min_support=2)
    for z in (0.045, 0.050, 0.055):
        env.record("grasp_object", {"label": "cube"}, ok=True, measured={"tcp_z_at_grasp_m": z})
    v = env.check("grasp_object", {"label": "cube"}, measured={"tcp_z_at_grasp_m": 0.12})
    assert v.ok is False and v.outliers[0]["feature"] == "tcp_z_at_grasp_m"
    # inside the proven range -> fine; unknown feature -> silent
    assert env.check("grasp_object", {"label": "cube"}, measured={"tcp_z_at_grasp_m": 0.05}).ok
    assert env.check("grasp_object", {"label": "cube"}, measured={"object_height_m": 0.9}).ok


def test_digest_and_markdown_show_derived_features_and_missing_counts():
    env = OperatingEnvelope(min_support=2)
    for z, h in ((0.045, 0.05), (0.050, 0.05), (0.055, 0.05)):
        env.record("grasp_object", {"label": "cube"}, ok=True,
                   measured={"tcp_z_at_grasp_m": z, "object_height_m": h,
                             "object_width_m": 0.055, "object_tcp_lateral_offset_m": 0.003})
    env.record("grasp_object", {"label": "cube"}, ok=False, error="no IK solution",
               measured={"object_height_m": 0.05, "object_width_m": 0.055})
    digest = env.envelope_digest()
    assert "tcp_z_at_grasp_m" in digest and "measured" in digest
    assert "missing" in digest and "tcp_z_at_grasp_m" in digest.split("missing")[1]
    md = env.export_markdown()
    assert "`tcp_z_at_grasp_m`" in md and "measured at call time" in md
    assert "missing in 1 call" in md
    # the four derived features are not crowded out by the per-skill cap on
    # raw-arg features in the digest
    for _ in range(3):
        env.record("grasp_object", {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "label": "cube"}, ok=True,
                   measured={"tcp_z_at_grasp_m": 0.05, "object_height_m": 0.05,
                             "object_width_m": 0.055, "object_tcp_lateral_offset_m": 0.003})
    assert "object_tcp_lateral_offset_m" in env.envelope_digest()


def test_derived_and_missing_round_trip_through_json(tmp_path):
    path = tmp_path / "env.json"
    env = OperatingEnvelope(path=path)
    env.record("grasp_object", {"label": "cube"}, ok=True,
               measured={"tcp_z_at_grasp_m": 0.05, "object_height_m": None})
    reloaded = OperatingEnvelope(path=path)
    st = reloaded.stats()["grasp_object"]
    assert st["derived"] == ["tcp_z_at_grasp_m"]
    # everything the skill declares and the call did not measure is missing
    assert st["missing"] == {"object_height_m": 1, "object_width_m": 1,
                             "object_tcp_lateral_offset_m": 1}
    # a pre-feature file (no derived/missing keys) still loads
    blob = json.loads(path.read_text())
    for s in blob["skills"].values():
        s.pop("derived", None)
        s.pop("missing", None)
    path.write_text(json.dumps(blob))
    old = OperatingEnvelope(path=path)
    assert old.stats()["grasp_object"]["derived"] == [] and old.stats()["grasp_object"]["missing"] == {}


def test_ingest_trace_replays_measurements_recorded_in_the_trace(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": {"ok": True},
         "duration_ms": 900.0, "context": {"measured": {"tcp_z_at_grasp_m": 0.047,
                                                           "object_height_m": 0.05}}},
        # an older record without the channel: unknown, not missing
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": {"ok": True},
         "duration_ms": 900.0, "context": {"arm": "default"}},
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    env = OperatingEnvelope()
    assert env.ingest_trace(trace) == 2
    st = env.stats()["grasp_object"]
    assert st["spans"]["tcp_z_at_grasp_m"]["n"] == 1
    assert st["missing"] == {"object_width_m": 1, "object_tcp_lateral_offset_m": 1}


# ── runtime level: the measurements come from the runtime itself ─────────


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


def _wait_for_cube(runtime) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
        time.sleep(0.05)
    assert runtime.beliefs.find("red cube") is not None, "watcher never saw the mock cube"


@needs_pin
def test_a_mock_grasp_records_the_four_derived_features_from_runtime_measurements(runtime_and_arm):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _wait_for_cube(runtime)
    with runtime.watcher.paused():
        result = runtime.execute("grasp_object", {"label": "red cube"})
    assert result["ok"], result
    st = runtime.envelope.stats()["grasp_object"]
    spans = st["spans"]
    for feat in ("tcp_z_at_grasp_m", "object_height_m", "object_width_m",
                 "object_tcp_lateral_offset_m"):
        assert feat in spans, f"{feat} was not recorded; spans={sorted(spans)}"
        assert feat in st["derived"]
    assert st["missing"] == {}
    # Plausible against the synthetic scene: a 5 cm tall, ~5.5 x 7.3 cm box
    # at base (0.29, 0) grasped top-down with the mock arm landing on it.
    assert 0.03 < spans["tcp_z_at_grasp_m"]["lo"] < 0.08
    assert spans["object_height_m"]["lo"] == pytest.approx(0.05, abs=0.01)
    assert 0.04 < spans["object_width_m"]["lo"] < 0.08
    assert spans["object_tcp_lateral_offset_m"]["hi"] < 0.01
    # the measurements also ride in the trace, so `ingest_trace` can replay them
    rec = [json.loads(l) for l in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    grasp = next(r for r in rec if r["skill"] == "grasp_object")
    assert grasp["context"]["measured"]["object_height_m"] == pytest.approx(0.05, abs=0.01)


@needs_pin
def test_a_grasp_that_never_closes_records_tcp_features_as_missing(runtime_and_arm):
    """`grasp_object("unicorn")` fails at localization: nothing was measured,
    and all four features are counted missing for that call -- not defaulted
    to zero, not to the last grasp's values."""
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _wait_for_cube(runtime)
    with runtime.watcher.paused():
        result = runtime.execute("grasp_object", {"label": "unicorn"})
    assert not result["ok"]
    st = runtime.envelope.stats()["grasp_object"]
    assert st["spans"] == {}
    assert st["missing"] == {"tcp_z_at_grasp_m": 1, "object_height_m": 1,
                             "object_width_m": 1, "object_tcp_lateral_offset_m": 1}


@needs_pin
def test_pick_and_place_credits_its_inner_grasp_measurements_to_the_top_level_call(runtime_and_arm):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _wait_for_cube(runtime)
    with runtime.watcher.paused():
        result = runtime.execute("pick_and_place", {"object": "red cube", "destination": "drop zone"})
    assert result["ok"], result
    stats = runtime.envelope.stats()
    assert "tcp_z_at_grasp_m" in stats["pick_and_place"]["spans"]
    # the nested skill_grasp_object call did not go through execute(), so it
    # must not have produced a separate envelope row of its own
    assert "grasp_object" not in stats


def test_measurements_never_come_from_the_agents_arguments():
    """An agent cannot smuggle a 'measured' value through the call args: a
    raw arg named like a derived feature is just a raw arg."""
    env = OperatingEnvelope()
    env.record("grasp_object", {"label": "cube", "tcp_z_at_grasp_m": 0.5}, ok=True, measured={})
    st = env.stats()["grasp_object"]
    assert "tcp_z_at_grasp_m" not in st["derived"]
    assert st["missing"]["tcp_z_at_grasp_m"] == 1
