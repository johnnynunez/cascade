"""An operator halt is not a loss of the primitive it stopped.

`OperatingEnvelope.record` counts a failure whose signature is `safety:halt`
in `_SkillStats.halts` only: no win/loss, no contradiction on a proven range,
no missing measurement, no last error, no duration. An e-stop still counts
as a loss, and so does a failed skill whose park or view-home return the
operator halted afterwards: it is filed by its own error (`skill_error`).
"""
import json

import pytest

from cascade.memory.envelope import DERIVED_FEATURES, OperatingEnvelope, _SkillStats

SKILL = "pick_and_place"
PICK_MISSED = "grasp failed after 1 attempts (0.4s): grasp missed: closed on air"
#: what the operator's halt leaves (of the skill, its route, a view-home return skipped)
HALTS = [
    "MotionHalted: halted: operator: stop",
    "MotionHalted: halt received during route planning or execution",
    "return to saved home skipped: motion halted: operator: stop",
]


def _measured(i):
    return {"tcp_z_at_grasp_m": 0.08 + 0.002 * i, "object_height_m": 0.05, "object_width_m": 0.04 + 0.001 * i,
            "object_tcp_lateral_offset_m": 0.004}


def _eight_wins(env):
    for i in range(8):
        env.record(SKILL, {"x": 0.30 + 0.01 * i, "y": -0.05 + 0.01 * i}, ok=True, duration_ms=1000.0,
                   measured=_measured(i))


def _view(env):
    """Everything a failure could change, as the agent and the dashboard see it."""
    st = env.stats()[SKILL]
    probe = env.check(SKILL, {"x": 0.33, "y": -0.02}, measured=_measured(3))
    return {
        "wins": st["wins"], "losses": st["losses"], "failures": st["failures"],
        "last_error": st["last_error"], "duration_ms_mean": st["duration_ms_mean"], "missing": st["missing"],
        "contradictions": {k: v["contradictions"] for k, v in st["spans"].items()},
        "notes": probe.notes, "failure_digest": env.failure_digest(), "agent_digest": env.agent_digest(),
    }


def _one_failure(env, error, **kw):
    # in every proven range; measured: nothing (the call failed before the close)
    env.record(SKILL, {"x": 0.33, "y": -0.02}, ok=False, error=error, duration_ms=60000.0, measured={}, **kw)


@pytest.mark.parametrize("error", HALTS)
def test_an_operator_halt_is_not_a_loss(error):
    env, reference = OperatingEnvelope(), OperatingEnvelope()
    _eight_wins(env)
    _eight_wins(reference)

    _one_failure(env, error)

    assert _view(env) == _view(reference)
    assert env.stats()[SKILL]["halts"] == 1
    assert DERIVED_FEATURES[SKILL] and not env.stats()[SKILL]["missing"]


@pytest.mark.parametrize("error, signature, filed", [
    # an e-stop can be the operator reacting to the primitive's motion: still a loss
    ("SafetyViolation: e-stop latched", None, "safety:estop"),
    # a skill that missed, then the operator halted its final park: the miss counts
    (f"final park halted: operator: stop; {PICK_MISSED}", "other:grasp_failed_after", "other:grasp_failed_after"),
])
def test_an_estop_or_a_failure_before_a_halt_is_still_a_loss(error, signature, filed):
    env = OperatingEnvelope()
    _eight_wins(env)

    _one_failure(env, error, signature=signature)

    st = env.stats()[SKILL]
    assert (st["wins"], st["losses"], st["halts"]) == (8, 1, 0)
    assert st["failures"] == {filed: 1} and st["last_error"] == error[:200]
    assert st["spans"]["x"]["contradictions"] == 1
    assert sorted(st["missing"]) == sorted(DERIVED_FEATURES[SKILL])
    assert f"{SKILL}: 8W/1L; {filed} x1" in env.failure_digest()


def test_halts_persist_across_instances(tmp_path):
    path = tmp_path / "envelope.json"
    env = OperatingEnvelope(path=path)
    _eight_wins(env)
    _one_failure(env, HALTS[0])
    _one_failure(env, HALTS[1])

    saved = json.loads(path.read_text())["skills"][SKILL]
    assert saved["halts"] == 2 and saved["losses"] == 0
    assert OperatingEnvelope(path=path).stats()[SKILL] == env.stats()[SKILL]
    # a file written before the counter existed reads as no halts
    assert _SkillStats.from_json({"wins": 3}).halts == 0


def test_ingest_trace_counts_a_halt_row_as_a_halt_not_a_loss(tmp_path):
    rows = [{"skill": SKILL, "args": {"x": 0.3 + 0.01 * i}, "result": {"ok": True}, "duration_ms": 900}
            for i in range(4)]
    rows.append({"skill": SKILL, "args": {"x": 0.31}, "result": {"ok": False, "error": HALTS[0]},
                 "duration_ms": 50000})
    rows.append({"skill": SKILL, "args": {"x": 0.32}, "duration_ms": 900, "result": {
        "ok": False, "error": f"final park halted: operator: stop; {PICK_MISSED}", "skill_error": PICK_MISSED}})
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in rows))
    env = OperatingEnvelope()

    assert env.ingest_trace(trace) == 6

    st = env.stats()[SKILL]
    assert (st["wins"], st["losses"], st["halts"]) == (4, 1, 1)
    assert st["failures"] == {"other:grasp_failed_after": 1}
    assert st["duration_ms_mean"] == 900.0


#: the line export_markdown (MEMORY.md) adds for a skill's operator halts
HALTS_LINE = "- operator halts: {} (not counted as losses)"


def test_the_memory_export_lists_halts_apart_from_the_record():
    env, reference = OperatingEnvelope(), OperatingEnvelope()
    _eight_wins(env)
    _eight_wins(reference)
    _one_failure(env, HALTS[0])
    _one_failure(env, HALTS[1])

    lines, ref = env.export_markdown().splitlines(), reference.export_markdown().splitlines()

    assert lines[2] == ref[2] == f"## {SKILL} — 8W/0L (100% success, ~1.0s)"
    assert HALTS_LINE.format(2) in lines
    assert [line for line in lines if line != HALTS_LINE.format(2)] == ref   # nothing else changes
    assert not any(line.startswith("- operator halts") for line in ref)


def test_a_skill_with_only_halts_gets_no_win_loss_header():
    """No win or loss yet: the export does not print a 0W/0L, 0% success
    header for it, only its halts."""
    env = OperatingEnvelope()
    _one_failure(env, HALTS[0])

    lines = env.export_markdown().splitlines()

    assert f"## {SKILL}" in lines and not any("0W/0L" in line or "success" in line for line in lines)
    assert lines[lines.index(f"## {SKILL}") + 1] == HALTS_LINE.format(1)


def test_a_skill_with_neither_a_call_nor_a_halt_keeps_its_header(tmp_path):
    """Only a halt-only skill drops the win/loss header: a stored skill with
    no calls and no halts exports exactly as before."""
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps({"skills": {SKILL: {"wins": 0, "losses": 0}}}))

    assert f"## {SKILL} — 0W/0L (0% success, ~0.0s)" in OperatingEnvelope(path=path).export_markdown().splitlines()

