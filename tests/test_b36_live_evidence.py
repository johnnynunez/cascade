"""B36 live evidence (8-10 Oct 2026): pins docs/evidence/b36-pick-reliability-20261010/.

Recomputes every number the docs and the Isaac profile quote from runs.json (one row per live run, built
from each run's trace and grasp-evidence receipt) and checks that README / ROADMAP / ARCHITECTURE /
configs/arms/isaac.yaml / the runtime docstrings quote them, so neither a stale JSON nor a drifted doc
passes. The README's tables must be exactly what render_tables.py prints for runs.json.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EV = REPO / "docs/evidence/b36-pick-reliability-20261010"
SHA = re.compile(r"[0-9a-f]{64}")
OUTCOMES = {"confirmed", "S1", "S2", "S3", "S4"}


def _runs() -> list[dict]:
    return json.loads((EV / "runs.json").read_text())


def _rate(rows: list[dict]) -> str:
    return f"{sum(r['outcome'] == 'confirmed' for r in rows)}/{len(rows)}"


def _physx_ggx(arm_is_branch: bool) -> list[dict]:
    """The hold A/B on PhysX with learned grasps: A/B2, A/B6, W8 (main vs a hold-configured arm)."""
    out = []
    for r in _runs():
        if r["engine"] != "physx" or r["planner"] != "graspgenx" or r["series"] not in ("A/B2", "A/B6", "W8"):
            continue
        if r["arm"].startswith("main") != (not arm_is_branch):
            continue
        out.append(r)
    return out


def _doc(rel: str) -> str:
    return re.sub(r"\s+", " ", (REPO / rel).read_text())


def test_every_run_row_is_complete_and_classified():
    rows = _runs()
    assert len(rows) == 76
    assert len({r["tag"] for r in rows}) == len(rows)
    for r in rows:
        assert r["outcome"] in OUTCOMES, r["tag"]
        assert r["engine"] in ("physx", "newton"), r["tag"]
        assert r["planner"] in ("graspgenx", "obb"), r["tag"]
        if r["outcome"] == "confirmed":
            assert r["physics"] == "confirmed" and r["target_err_m"] < 0.05, r["tag"]


def test_manifest_hashes_every_input_run():
    manifest = json.loads((EV / "manifest.json").read_text())["inputs_sha256"]
    traces = {k.split("/")[-2][len("run_"):] for k in manifest if k.endswith("/trace.jsonl")}
    assert traces == {r["tag"] for r in _runs()}
    assert all(SHA.fullmatch(v) for v in manifest.values())


def test_readme_tables_are_rendered_from_runs_json():
    out = subprocess.run([sys.executable, str(EV / "render_tables.py"), str(EV / "runs.json")],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out and out in (EV / "README.md").read_text()


def test_the_hold_ab_on_physx_is_quoted_as_measured():
    main, branch = _rate(_physx_ggx(False)), _rate(_physx_ggx(True))
    assert (main, branch) == ("9/16", "15/16")
    applied = [r for r in _runs() if r["engine"] == "physx" and r["planner"] == "graspgenx" and r["hold"] == "applied"]
    assert _rate(applied) == "16/16"
    for rel in ("docs/ROADMAP.md", "configs/arms/isaac.yaml", "src/cascade/skills/runtime.py"):
        text = _doc(rel)
        assert main in text and branch in text, rel
    assert "main 9/16 vs branch 15/16" in _doc(str(EV.relative_to(REPO) / "README.md"))


def test_newton_keeps_the_plain_close_as_measured():
    rows = [r for r in _runs() if r["engine"] == "newton"]
    with_hold = [r for r in rows if r["arm"].startswith("recovery+hold")]
    without = [r for r in rows if r not in with_hold]
    assert all(r["hold"] != "applied" for r in without)
    assert (_rate(with_hold), _rate(without)) == ("1/4", "14/14")
    for rel in ("docs/ROADMAP.md", "configs/arms/isaac.yaml", "src/cascade/skills/runtime.py"):
        text = _doc(rel)
        assert "1/4" in text and "14/14" in text, rel


def test_s1_never_recurs_where_the_recovery_ran():
    rows = _runs()
    recovery = [r for r in rows if not r["arm"].startswith("main")]
    main = [r for r in rows if r["arm"].startswith("main")]
    assert sum(r["outcome"] == "S1" for r in recovery) == 0 and len(recovery) == 39
    assert sum(r["outcome"] == "S1" for r in main) == 7 and len(main) == 37
    assert "0 of 39 live recovery-arm runs vs 7 of 37 on main" in _doc("docs/ROADMAP.md")
    # the two live recoveries carried the cube on and failed only at the place
    rec = [r for r in rows if r["recovered_after_failed_attempt"]]
    assert sorted(r["tag"] for r in rec) == ["ab6pf4", "b361"]
    assert {r["outcome"] for r in rec} == {"S2"}


def test_wrist_roll_mechanism_numbers():
    rows = [r for r in _runs() if r["lift_q6_err"] is not None]
    squeeze = [r for r in rows if r["engine"] == "physx" and r["hold"] != "applied" and "fragile" not in r["arm"]]
    held = [r for r in rows if r["engine"] == "physx" and r["hold"] == "applied"]
    newton = [r for r in rows if r["engine"] == "newton"]
    assert len(squeeze) == 35 and round(statistics.median(r["lift_q6_err"] for r in squeeze), 3) == 0.045
    over = [r for r in squeeze if r["lift_q6_err"] >= 0.045]
    assert len(over) == 9
    # every lift over settle_tol failed to settle (S1, or S2 after the recovery); none under it did
    assert all(r["outcome"] in ("S1", "S2") for r in over)
    assert not [r for r in squeeze if r["lift_q6_err"] < 0.045 and r["outcome"] == "S1"]
    assert len(held) == 20 and round(statistics.median(r["lift_q6_err"] for r in held), 3) == 0.011
    assert round(max(r["lift_q6_err"] for r in held), 3) == 0.015
    assert len(newton) == 18 and round(max(r["lift_q6_err"] for r in newton), 3) == 0.0
    doc = _doc("src/cascade/skills/runtime.py")
    assert "(9 of 35 lifts" in doc and "0.011 rad (max 0.015, 20 lifts)" in doc and "(0.000 rad, 18 lifts)" in doc


def test_the_air_grasp_check_after_a_hold_still_verifies_every_recorded_hold():
    """The post-lift check judges the hold opening when a hold applied: every recorded live hold grasp
    must still verify (width after lift above hold opening + 0.03)."""
    rows = [r for r in _runs() if r["hold"] == "applied" and r["width_after_lift"] is not None]
    assert len(rows) == 22
    margins = [r["width_after_lift"] - (r["hold_open_frac"] + 0.03) for r in rows]
    assert min(margins) > 0.02


def test_the_one_missed_stall_is_the_replayed_ab6pf4():
    rows = [r for r in _runs() if r["engine"] == "physx" and r["planner"] == "graspgenx"
            and not r["arm"].startswith("main") and str(r["hold"]).startswith("not applied")]
    assert [r["tag"] for r in rows if r["series"] in ("A/B6", "W8")] == ["ab6pf4"]
    assert "1 of 16 live runs" in _doc("docs/ARCHITECTURE.md")
