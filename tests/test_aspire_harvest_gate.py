"""ASPIRE harvests only runs with the sim retry ledger (#388 re-review, Minor 1).

Only the ledger (the kitchen sim at gate 1) marks a failed call that an operator stop
ended when the call's error does not name the stop (`context.sim_retry.ended_in_stop`).
Without it (gate 0, every hardware profile) such a call looks like a real failure: a
rebot_rs pick that missed and was then e-stopped during its final park files as the
miss, and a confirmed retry after it taught the library a note (measured `learned=1`
on #388's head). Nothing is learned across an operator stop, so `aspire.harvest` and
`scripts/learn_from_runs.py` read only runs whose `run.json` shows the ledger on;
every other run is skipped with a one-line reason and counted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_retry_loops_end_on_halt import _one_marked_miss, _stop_in_rehome, rig  # noqa: F401  (fixture)

from cascade.agent.aspire import diagnose, harvest, harvest_skip_reason, skipped_runs
from cascade.agent.effects import CONFIRMED, POSTCONDITIONS
from cascade.skills.library import SkillLibrary

REPO = Path(__file__).resolve().parents[1]
TASKS = (("run-a", "pick the red cube"), ("run-b", "put the red cube in the bowl"))
LEDGER_OFF = "run.json shows the sim retry ledger off (gate 0 or a hardware profile)"


def _confirmed(skill="pick_and_place"):
    return {"ok": True, "postcondition": {
        "skill": skill, "kind": POSTCONDITIONS[skill], "status": CONFIRMED, "channel": "physics",
        "evidence": "Synthetic fixture: object moved to the destination", "measured": {"moved_m": 0.2}}}


def _teach_runs(runtime, root: Path) -> Path:
    """The runtime's own trace and run record, then a measured-confirmed retry of the last
    failed pick_and_place (same goal, arm and possession), as two runs of two tasks."""
    rows = runtime.trace.rows()
    record = (runtime.trace.run_dir / "run.json").read_text()
    failed = [r for r in rows if r.get("skill") == "pick_and_place" and not (r.get("result") or {}).get("ok", True)]
    context = failed[-1].get("context") or {}
    repair = {"skill": "pick_and_place", "args": dict(failed[-1]["args"]), "result": _confirmed(),
              "context": {"arm": context.get("arm"), "held_object": context.get("held_object")}}
    runs = root / "runs"
    for name, task in TASKS:
        run = runs / name
        run.mkdir(parents=True)
        (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows + [repair]) + "\n")
        (run / "summary.txt").write_text(f"task: {task}\nsuccess: true")
        (run / "run.json").write_text(record)
    return runs


@pytest.mark.parametrize("gate", [0, 1], ids=["gate 0", "gate 1"])
@pytest.mark.parametrize("kind", ["estop_harness", "estop_safearm"])
def test_a_rebot_rs_miss_estopped_at_the_final_park_is_never_learned(rig, monkeypatch, tmp_path, gate, kind):  # noqa: F811
    """The review's row: rebot_rs, one missed grasp, the operator's e-stop during the final
    park. The error names only the miss and no ledger marks the row (hardware), so the
    run's own diagnosis is still teachable; harvest skips the run on its run.json and
    learns nothing. At gate 1 the pick returns after its miss without a park (no stop at
    all): the hardware run is skipped all the same."""
    if gate:
        monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    r = rig("rebot_rs")
    _one_marked_miss(r, monkeypatch)
    _stop_in_rehome(r, monkeypatch, kind=kind)  # the 1st home move: the final park
    out = r.runtime.execute("pick_and_place", {"object": "red cube"})
    assert out["ok"] is False and (r.at_stop is not None) is (not gate), out
    assert bool(r.runtime.arm.harness.estopped) is (not gate) and r.runtime.sim_retry is None
    runs = _teach_runs(r.runtime, tmp_path)
    assert json.loads((runs / "run-a" / "run.json").read_text())["sim_retry"] is False
    assert diagnose(runs / "run-a").teachable  # the per-run rules alone would learn it

    library = SkillLibrary(tmp_path / "lib")
    learned = harvest(runs, library)

    assert learned["learned"] == 0 and learned["diagnosed"] == 0 and learned["entries"] == [], learned
    assert learned["skipped"] == 2 and learned["skipped_runs"] == [f"{n}: {LEDGER_OFF}" for n, _ in TASKS]
    assert library.records() == [] and list((tmp_path / "lib").glob("*.md")) == []


@pytest.mark.parametrize("gate", [0, 1], ids=["gate 0", "gate 1"])
def test_only_a_gate1_kitchen_run_is_harvested(rig, monkeypatch, tmp_path, gate):  # noqa: F811
    """The kitchen sim profile, one missed grasp and no stop: at gate 1 the runtime has the
    ledger, its run.json says so and the confirmed retry is learned; at gate 0 the same
    miss is skipped."""
    if gate:
        monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    r = rig("isaac_kitchen")
    _one_marked_miss(r, monkeypatch)
    out = r.runtime.execute("pick_and_place", {"object": "red cube"})
    assert out["ok"] is False and r.at_stop is None, out
    assert (r.runtime.sim_retry is not None) is bool(gate)
    runs = _teach_runs(r.runtime, tmp_path)
    assert diagnose(runs / "run-a").teachable

    learned = harvest(runs, SkillLibrary(tmp_path / "lib"))

    if gate:
        assert learned["skipped"] == 0 and learned["skipped_runs"] == [], learned
        assert learned["diagnosed"] == 2 and learned["learned"] == 1 and len(learned["promoted"]) == 1, learned
    else:
        assert learned["skipped"] == 2 and learned["learned"] == 0 and learned["diagnosed"] == 0, learned


def _run(root: Path, name: str, records: dict[str, str | bytes] | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text("{}\n")
    for file, data in (records or {}).items():
        (run / file).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return run


ON = json.dumps({"schema": 1, "sim_retry": True, "purpose": "live"})
OFF = json.dumps({"schema": 1, "sim_retry": False, "purpose": "live"})
#: (run.json and kept records, the one-line reason; None: harvested)
RECORDS = {
    "ledger on": ({"run.json": ON}, None),
    "ledger on, kept earlier record on": ({"run.json": ON, "run.2026-10-18T10:00:00+00:00.json": ON}, None),
    "ledger off": ({"run.json": OFF}, LEDGER_OFF),
    "no ledger key": ({"run.json": json.dumps({"schema": 1})}, LEDGER_OFF),
    "ledger as a string": ({"run.json": json.dumps({"sim_retry": "true"})}, LEDGER_OFF),
    "ledger as 1": ({"run.json": json.dumps({"sim_retry": 1})}, LEDGER_OFF),
    "not an object": ({"run.json": json.dumps([["sim_retry", True]])}, LEDGER_OFF),
    "no run.json": ({}, "no run.json"),
    "only a kept record": ({"run.2026-10-18T10:00:00+00:00.json": ON}, "no run.json"),
    "not JSON": ({"run.json": "{sim_retry: true"}, "run.json is not JSON"),
    "not UTF-8": ({"run.json": b"\xff\xfe{}"}, "run.json is not JSON"),
    "too large": ({"run.json": ON[:-1] + ", \"pad\": \"" + "x" * (1 << 20) + "\"}"}, "run.json too large for a run record"),
    "a rebuild without the ledger": ({"run.json": ON, "run.2026-10-18T10:00:00+00:00.json": OFF},
                                     "run.2026-10-18T10:00:00+00:00.json shows the sim retry ledger off "
                                     "(gate 0 or a hardware profile)"),
}


@pytest.mark.parametrize("case", sorted(RECORDS))
def test_the_run_record_decides(tmp_path, case):
    """Only a run.json with a literal `"sim_retry": true`, and no kept earlier record of the
    same directory (a rebuild appends to the same trace) without it, is harvested."""
    records, reason = RECORDS[case]
    assert harvest_skip_reason(_run(tmp_path, "run-a", records)) == reason


def test_a_directory_as_run_json_is_skipped(tmp_path):
    run = _run(tmp_path, "run-a")
    (run / "run.json").mkdir()
    assert harvest_skip_reason(run).startswith("run.json unreadable (")


def _teachable_run(root: Path, name: str, task: str, record: str | None) -> Path:
    context = {"arm": "left", "held_object": None}
    rows = [{"skill": "grasp_object", "args": {"label": "cube", "material": "rigid"}, "context": context,
             "result": {"ok": False, "error": "link/joint 7 would hit the table"}},
            {"skill": "grasp_object", "args": {"label": "cube", "material": "soft"}, "context": context,
             "result": _confirmed("grasp_object")}]
    run = _run(root, name, {} if record is None else {"run.json": record})
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "summary.txt").write_text(f"task: {task}\nsuccess: true")
    return run


def _mixed_runs(root: Path) -> Path:
    """Four teachable runs: two gate-1 kitchen runs (two tasks), a gate-0 run, a run with no record."""
    runs = root / "runs"
    _teachable_run(runs, "run-1", "pick the cube", ON)
    _teachable_run(runs, "run-2", "stack the cube on the box", ON)
    _teachable_run(runs, "run-3", "pick the cube again", OFF)
    _teachable_run(runs, "run-4", "pick the cube once more", None)
    return runs


SKIPPED = [f"run-3: {LEDGER_OFF}", "run-4: no run.json"]


def test_harvest_counts_and_names_every_skipped_run(tmp_path):
    runs = _mixed_runs(tmp_path)
    library = SkillLibrary(tmp_path / "lib")

    out = harvest(runs, library)

    assert out["runs"] == 4 and out["diagnosed"] == 2 and out["learned"] == 1, out
    assert out["skipped"] == 2 and out["skipped_runs"] == SKIPPED, out
    (entry,) = library.records()
    assert entry.source_runs == ["run-1", "run-2"] and entry.promoted
    assert skipped_runs(runs) == SKIPPED
    assert skipped_runs(runs, limit=1) == SKIPPED[1:]  # the same selection as harvest: the last `limit` by name
    assert harvest(runs, SkillLibrary(tmp_path / "lib2"), limit=1)["skipped_runs"] == SKIPPED[1:]
    assert harvest(tmp_path / "none", library)["skipped_runs"] == [] and skipped_runs(tmp_path / "none") == []


def _learn(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "learn_from_runs.py"), *args],
                          cwd=cwd, capture_output=True, text=True, timeout=120, check=True)


@pytest.mark.parametrize("dry_run", [False, True], ids=["harvest", "dry run"])
def test_learn_from_runs_reports_the_skipped_runs(tmp_path, dry_run):
    """The CLI: --json carries the count and every line; the text output the count and the
    lines; a dry run lists the same skips and writes no note."""
    runs = _mixed_runs(tmp_path)
    common = ["--runs", str(runs), "--library", str(tmp_path / "lib"),
              "--envelope", str(tmp_path / "envelope.json")] + (["--dry-run"] if dry_run else [])

    skills = json.loads(_learn(common + ["--json"], tmp_path).stdout)["skills"]
    text = _learn(common, tmp_path).stdout

    assert skills["skipped"] == 2 and skills["skipped_runs"] == SKIPPED, skills
    assert skills["learned"] == (0 if dry_run else 1), skills
    assert "runs skipped  : 2 (notes come only from runs with the sim retry ledger on)" in text, text
    assert all(f"  {line}\n" in text for line in SKIPPED), text
    assert bool(list((tmp_path / "lib").glob("*.md"))) is not dry_run
