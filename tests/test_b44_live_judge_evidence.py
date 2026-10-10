"""B44-live evidence pin: the local Qwen judge over 72 recorded live Isaac picks.

docs/evidence/b44-live-qwen-judge-20261009/results.json holds one record per
physics-graded pick (hop from pass 1 at 1536 tokens, pass 2 at 8192 for the
steps pass 1 left unscored); manifest.json pins each record to its live source
(trace + keyframe sha256). These tests recompute every number the docs quote,
with the CODE's decision rule (StepVerdict.agrees_with_physics), so neither a
stale JSON nor a drifted doc passes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cascade.eval.progress_judge import StepVerdict

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "docs/evidence/b44-live-qwen-judge-20261009"


def _results() -> dict:
    return json.loads((EV / "results.json").read_text())


def _hop(rec: dict, passes: int):
    if passes == 1 or rec["hop_1536"] is not None:
        return rec["hop_1536"]
    return rec.get("hop_8192")


def _matrix(records: list[dict], passes: int, success_above: float = 0.0) -> dict:
    c = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "scored": 0}
    for rec in records:
        hop = _hop(rec, passes)
        if hop is None:
            continue
        c["scored"] += 1
        if success_above == 0.0:
            agree = StepVerdict(step=0, skill="pick_and_place", task="", hop=hop, physics=rec["physics"],
                                channel="physics", ok=None, tier=None, duration_s=None).agrees_with_physics()
            if agree is None:
                continue
            pos = hop > 0
            assert agree == (pos == (rec["physics"] == "confirmed"))
        else:
            if rec["physics"] not in ("confirmed", "refuted"):
                continue
            pos = hop > success_above
        conf = rec["physics"] == "confirmed"
        c["tp" if pos and conf else "fn" if conf else "fp" if pos else "tn"] += 1
    c["n"] = c["tp"] + c["tn"] + c["fp"] + c["fn"]
    return c


def test_every_record_is_pinned_to_a_live_source():
    res, man = _results(), json.loads((EV / "manifest.json").read_text())
    runs = [r["run"] for r in res["steps"]]
    assert len(runs) == len(set(runs)) == 72
    assert [m["run"] for m in man] == runs
    for m in man:
        for key in ("trace_sha256", "keyframe_before_sha256", "keyframe_after_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", m[key]), (m["run"], key)
        assert m["keyframe_before"] != m["keyframe_after"]
        assert m["keyframe_before_sha256"] != m["keyframe_after_sha256"], m["run"]
    assert {r["physics"] for r in res["steps"]} == {"confirmed", "refuted", "unverified"}
    assert sum(r["physics"] == "confirmed" for r in res["steps"]) == 40
    assert sum(r["physics"] == "refuted" for r in res["steps"]) == 27
    # pass 2 re-judged exactly the steps pass 1 left unscored
    assert {r["run"] for r in res["steps"] if r["hop_1536"] is None} == {r["run"] for r in res["steps"] if "hop_8192" in r}
    assert res["config_pass1"]["max_tokens"] == 1536 and res["config_pass2"]["max_tokens"] == 8192
    assert res["config_pass1"]["base_url"] == "http://127.0.0.1:8080/v1"


def test_the_documented_matrices_are_the_recomputed_ones():
    steps = _results()["steps"]
    assert _matrix(steps, 1) == {"tp": 40, "tn": 1, "fp": 15, "fn": 0, "scored": 60, "n": 56}
    assert _matrix(steps, 2) == {"tp": 40, "tn": 7, "fp": 19, "fn": 0, "scored": 71, "n": 66}
    # the post-hoc reading (success = +100 %), in-sample, not shipped
    assert _matrix(steps, 2, success_above=0.9) == {"tp": 40, "tn": 26, "fp": 0, "fn": 0, "scored": 71, "n": 66}


def test_confirmed_picks_all_scored_full_progress_and_no_confirmed_pick_went_unscored():
    steps = _results()["steps"]
    assert {r["hop_1536"] for r in steps if r["physics"] == "confirmed"} == {1.0}
    unscored = [r for r in steps if r["hop_1536"] is None]
    assert len(unscored) == 12 and not any(r["physics"] == "confirmed" for r in unscored)
    assert all(r["error_1536"].startswith("no <score>") for r in unscored)


@pytest.mark.parametrize("doc", ["docs/ROADMAP.md", "docs/ARCHITECTURE.md",
                                 "docs/evidence/b44-live-qwen-judge-20261009/README.md"])
def test_docs_quote_the_measured_numbers(doc):
    text = " ".join((REPO / doc).read_text().split())
    assert "tp=40 tn=1 fp=15 fn=0" in text or "| 40 | 1 | 15 | 0 |" in text, doc
    assert "tp=40 tn=7 fp=19 fn=0" in text or "| 40 | 7 | 19 | 0 |" in text, doc
