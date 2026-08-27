"""Graduated confidence + contradiction tracking in memory/envelope.py.

Ported from checking the actual RLinf/RPent repo (the code behind the
Harness-VLA paper this module already cites), not just its abstract: RPent's
memory entries carry a three-tier confidence (single-shot -> probable ->
verified) by evidence breadth, and a `contradicted_by` field for when a
"proven" entry is later falsified. This module previously had neither -- a
span was either trusted (>= MIN_SUPPORT) or unknown, and nothing recorded a
failure landing inside a range this same skill had proven safe.
"""

from __future__ import annotations

from wrc_demo.memory.envelope import OperatingEnvelope, _Span


def test_span_confidence_is_single_shot_below_min_support():
    span = _Span(lo=0.1, hi=0.1, n=1)
    assert span.confidence(min_support=4) == "single-shot"


def test_span_confidence_is_probable_at_min_support():
    span = _Span(lo=0.1, hi=0.2, n=4)
    assert span.confidence(min_support=4) == "probable"


def test_span_confidence_is_verified_at_double_min_support():
    span = _Span(lo=0.1, hi=0.2, n=8)
    assert span.confidence(min_support=4) == "verified"


def test_digest_labels_the_confidence_tier():
    env = OperatingEnvelope()
    for x in (0.16, 0.17, 0.18, 0.165, 0.175, 0.172, 0.168, 0.171):  # n=8 -> verified
        env.record("grasp_object", {"x": x}, ok=True)
    digest = env.agent_digest()
    assert "verified" in digest


def test_a_failure_inside_a_proven_range_is_a_contradiction():
    env = OperatingEnvelope()
    for x in (0.16, 0.17, 0.18, 0.165):
        env.record("grasp_object", {"x": x}, ok=True)
    # 0.17 is INSIDE the proven [0.16, 0.18] range -- this call still failed.
    env.record("grasp_object", {"x": 0.17}, ok=False, error="air-grasp: closed on nothing")
    stats = env.stats()
    assert stats["grasp_object"]["spans"]["x"]["contradictions"] == 1


def test_a_failure_outside_the_range_is_not_a_contradiction():
    env = OperatingEnvelope()
    for x in (0.16, 0.17, 0.18, 0.165):
        env.record("grasp_object", {"x": x}, ok=True)
    env.record("grasp_object", {"x": 0.9}, ok=False, error="no IK solution")
    stats = env.stats()
    assert stats["grasp_object"]["spans"]["x"]["contradictions"] == 0


def test_contradiction_surfaces_as_a_non_blocking_note_not_a_veto():
    """Booth rule: envelopes are advisory. A contradicted range must still
    return ok=True -- the caller gets a caution, never a block."""
    env = OperatingEnvelope()
    for x in (0.16, 0.17, 0.18, 0.165):
        env.record("grasp_object", {"x": x}, ok=True)
    env.record("grasp_object", {"x": 0.17}, ok=False, error="air-grasp: closed on nothing")

    verdict = env.check("grasp_object", {"x": 0.17})
    assert verdict.ok is True
    assert verdict.notes and "treat cautiously" in verdict.notes[0].lower()
    assert verdict.as_dict()["notes"] == verdict.notes


def test_contradiction_is_absent_from_notes_when_the_span_never_failed():
    env = OperatingEnvelope()
    for x in (0.16, 0.17, 0.18, 0.165):
        env.record("grasp_object", {"x": x}, ok=True)
    verdict = env.check("grasp_object", {"x": 0.17})
    assert verdict.notes == []


def test_span_json_round_trips_contradictions(tmp_path):
    path = tmp_path / "env.json"
    env = OperatingEnvelope(path=path)
    for x in (0.16, 0.17, 0.18, 0.165):
        env.record("grasp_object", {"x": x}, ok=True)
    env.record("grasp_object", {"x": 0.17}, ok=False, error="air-grasp: closed on nothing")

    reloaded = OperatingEnvelope(path=path)
    assert reloaded.stats()["grasp_object"]["spans"]["x"]["contradictions"] == 1
