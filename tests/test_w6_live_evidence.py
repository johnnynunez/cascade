"""Wave-6 live validations (10 Oct 2026): pins docs/evidence/w6-live-20261010/.

Three items that landed CPU-tested owed one live Isaac run each: B46 (read-only MCP lane during a pick), B63
(launch.sh's registration carries CASCADE_GRASP_EXECUTOR / CASCADE_VLA_PORT to the server) and B44 (the
launcher's judge pass over a live proof turn). These tests recompute every number the docs quote from the
evidence JSON, with the CODE's own constants and decision functions where one exists (READONLY_LANE_TOOLS,
mcp_env.FORWARDED, load_demo_config's executor selection, StepVerdict / RunVerdict.confusion), and check that
README / ROADMAP / ARCHITECTURE quote them, so neither a stale JSON nor a drifted doc passes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "docs/evidence/w6-live-20261010"
READS = ("world_state", "robot_knowledge", "verify_last_action", "camera_snapshot")
SHA = re.compile(r"[0-9a-f]{64}")


def _load(name: str) -> dict:
    return json.loads((EV / name).read_text())


def _b46_runs(lane: str | None = None, *, with_stop: bool = True) -> list[dict]:
    runs = _load("b46_lane_ab.json")["runs"]
    return [r for r in runs if (lane is None or r["lane"] == lane) and (with_stop or not r["stop"])]


def _confirmed(rec: dict) -> bool:
    pc = rec["motion_postcondition"]
    return pc["status"] == "confirmed" and pc["channel"] == "physics"


def _span(values) -> str:
    return f"{min(values):.3f}–{max(values):.3f}"


def _text(doc: str) -> str:
    return " ".join((REPO / doc).read_text().split())


# ── premises (true on main by design: what the live runs exercised) ─────────


def test_premise_the_probe_reads_are_exactly_the_lane_tools():
    from cascade.apps.mcp_server import READONLY_LANE_TOOLS

    assert READONLY_LANE_TOOLS == frozenset(READS)


def test_premise_the_registry_forwards_the_executor_pair():
    from cascade.apps.mcp_env import FORWARDED

    assert {"CASCADE_GRASP_EXECUTOR", "CASCADE_VLA_PORT", "CASCADE_BRIDGE_PORT"} <= set(FORWARDED)


def test_premise_launch_sh_has_the_judge_block_and_one_registration_heredoc():
    text = (REPO / "scripts/launch.sh").read_text()
    assert text.count("    # >>> judge pass\n") == 1 and text.count("    # <<< judge pass\n") == 1
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", text, re.DOTALL)
    assert len([b for b in blocks if '"requestTimeoutMs"' in b]) == 1


# ── B46: read-only lane during a live pick ───────────────────────────────────


def test_b46_every_run_is_pinned_to_its_live_source_and_interleaved():
    data, manifest = _load("b46_lane_ab.json"), _load("manifest.json")["files"]
    runs = data["runs"]
    assert [r["tag"] for r in runs] == data["protocol"]["order"] == [
        "on1", "off1", "on2", "off2", "on3", "off3", "on4", "off4", "stop1"]
    assert [r["lane"] for r in runs] == ["1", "0"] * 4 + ["1"]
    for r in runs:
        for key in ("probe_json", "trace"):
            assert manifest[r["sources"][key]] == r["sources"][key + "_sha256"], (r["tag"], key)
            assert SHA.fullmatch(r["sources"][key + "_sha256"])
        assert r["motion"] == "pick_and_place" and r["trace_pick_rows"] == 1
        assert r["trace_pick_postcondition"] == [r["motion_postcondition"]["status"]]
        assert r["bridge_gpu"]["backend"] == "physx" and r["bridge_gpu"]["cpu_fallback_allowed"] is False


def test_b46_lane_on_reads_answer_during_the_pick_and_are_marked():
    for r in _b46_runs("1"):
        assert [x["tool"] for x in r["reads"]] == list(READS), r["tag"]
        for x in r["reads"]:
            assert x["answered_before_motion"] is True and x["served_during_motion"] == r["motion"], (r["tag"], x)
            assert x["isError"] is False
        assert r["after_motion_world_state_marked"] is False
    on = [x for r in _b46_runs("1", with_stop=False) for x in r["reads"]]
    assert len(on) == 16
    assert _span([x["latency_s"] for x in on if x["tool"] != "camera_snapshot"]) == "0.001–0.003"
    assert _span([x["latency_s"] for x in on if x["tool"] == "camera_snapshot"]) == "0.024–0.038"


def test_b46_lane_off_reads_wait_for_the_whole_pick_unmarked():
    off_runs = _b46_runs("0")
    off = [x for r in off_runs for x in r["reads"]]
    assert len(off) == 16
    assert not any(x["answered_before_motion"] for x in off)
    assert all(x["served_during_motion"] is None and x["isError"] is False for x in off)
    for r in off_runs:  # queued behind the motion: latency = motion time minus the ~1 s read delay
        assert all(abs((r["motion_s"] - x["latency_s"]) - 1.0) < 0.1 for x in r["reads"]), r["tag"]
    assert _span([x["latency_s"] for x in off if x["tool"] != "camera_snapshot"]) == "65.418–164.703"
    assert _span([x["latency_s"] for x in off if x["tool"] == "camera_snapshot"]) == "65.468–164.778"


def test_b46_pick_outcomes_with_the_lane_are_no_worse_and_failures_are_b36_signatures():
    on, off = _b46_runs("1", with_stop=False), _b46_runs("0")
    assert [r["tag"] for r in on if _confirmed(r)] == ["on1", "on4"]
    assert [r["tag"] for r in off if _confirmed(r)] == ["off3"]
    assert sum(map(_confirmed, on)) >= sum(map(_confirmed, off))
    for r in on + off:
        if _confirmed(r):
            assert r["motion_result"]["ok"] is True and r["motion_result"]["verified"] is True
            continue
        assert r["motion_result"]["ok"] is False and r["motion_postcondition"]["status"] == "refuted"
        fails = r["grasp_attempt_failures"]
        lift = fails[0] == "did not settle at grasp lift pose" and set(fails[1:]) == {
            "already holding 'pink cube'; place it first"}
        placed_off = fails == [None] and "postcondition failed: pink cube ended" in r["motion_result"]["error"]
        assert lift or placed_off, r["tag"]
    assert sorted(r["tag"] for r in on + off if r["grasp_attempt_failures"][:1] == [
        "did not settle at grasp lift pose"]) == ["off1", "on2", "on3"]


def test_b46_the_stop_still_preempts_a_lane_on_pick():
    [stop] = [r for r in _b46_runs("1") if r["stop"]]
    assert stop["tag"] == "stop1" and stop["stop"] == {"after_s": 3.0, "latency_s": 0.032,
                                                       "answered_before_motion": True, "stopped": True}
    assert stop["motion_s"] == 3.27 and stop["motion_result"]["ok"] is False
    assert "e-stop latched" in stop["motion_result"]["error"]
    assert stop["grasp_attempt_failures"] == ["simulation motion stopped"]


# ── B63: the registered env reaches a live server ────────────────────────────


def _b63() -> list[dict]:
    return _load("b63_vla_env.json")["runs"]


def _selected_executor(entry_env: dict) -> dict:
    """What the runtime's config loader selects with ONLY the recorded entry env (CUDA / DISPLAY dropped:
    device selection is not under test; paths are never opened by the loader)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CASCADE_", "CUDA_"))}
    env.update({k: v for k, v in entry_env.items() if k.startswith("CASCADE_")})
    env["PYTHONPATH"] = str(REPO / "src")
    code = ("import json\nfrom cascade.config import load_demo_config\n"
            "g = load_demo_config(cameras=['mock'], arm='mock', llm='mock').as_dict()['grasp']\n"
            "print(json.dumps({'executor': g.get('executor'), 'vla_port': g['vla']['port']}))\n")
    child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout)


def test_b63_runs_are_pinned_and_used_one_registration():
    runs, manifest = _b63(), _load("manifest.json")["files"]
    assert [r["tag"] for r in runs] == ["vla1", "vla2", "an1", "an2", "an3", "an4", "an5"]
    assert len({r["heredoc_sha256"] for r in runs}) == 1 and len({r["launch_sh_sha256"] for r in runs}) == 1
    for r in runs:
        assert SHA.fullmatch(r["trace_sha256"]) and r["trace_sha256"] in manifest.values(), r["tag"]
        assert r["entry"]["args"][:4] == ["-m", "cascade.apps.mcp_server", "--launch-owner", "<owner>"]
        assert r["entry"]["cwd"].endswith("/models") and r["server_exit"] == 0
        ports = {k: v for k, v in r["entry"]["env"].items() if k.endswith("_PORT")}
        assert ports and all(47000 <= int(v) <= 47099 for v in ports.values()), ports
        # the server got ONLY the entry env (+ PATH, HOME, the editable-install stand-in)
        assert set(r["server_env_keys"]) == set(r["entry"]["env"]) | {"PATH", "HOME", "PYTHONPATH"}


def test_b63_every_runtime_variable_in_an_entry_is_forwarded_or_written_by_the_registration():
    from cascade.apps.mcp_env import FORWARDED, NOT_FORWARDED

    written = {k for k, (cat, _) in NOT_FORWARDED.items() if cat == "registration"}
    for r in _b63():
        names = {k for k in r["entry"]["env"] if k.startswith(("CASCADE_", "CUDA_"))}
        assert names <= set(FORWARDED) | written, (r["tag"], names - set(FORWARDED) - written)


def test_b63_the_entry_env_selects_the_executor_that_ran_live():
    for r in _b63():
        env = r["entry"]["env"]
        if r["arm"] == "vla":
            assert (env["CASCADE_GRASP_EXECUTOR"], env["CASCADE_VLA_PORT"]) == ("vla", "47010")
            assert _selected_executor(env) == {"executor": "vla", "vla_port": 47010}
            assert "[cascade] grasp executor: vla -> 127.0.0.1:47010 (policy=scripted-stub, chunks=2)" in r["server_log"]
            assert any("vla_policy=yes" in ln for ln in r["server_log"])
            assert r["world_state_backends"]["grasp_executor"].startswith("vla (127.0.0.1:47010")
            assert r["pick"]["error"].startswith("grasp failed after 8 attempts")
            assert r["pick"]["error"].endswith("air grasp: gripper closed fully, object not held")
            assert r["pick"]["postcondition"]["evidence"] == "pink cube is still within 0.0 cm of where it started"
        else:
            assert "CASCADE_GRASP_EXECUTOR" not in env and "CASCADE_VLA_PORT" not in env
            assert _selected_executor(env)["executor"] == "analytic"
            assert not any("grasp executor" in ln or "vla_policy" in ln for ln in r["server_log"])
            assert "grasp_executor" not in r["world_state_backends"]
            assert r["world_state_backends"]["grasp_planner"] == "graspgenx (learned 6-DoF)"


def test_b63_the_stub_saw_live_observations_for_every_episode():
    [vla2] = [r for r in _b63() if r["tag"] == "vla2"]
    req = vla2["vla_requests"]
    assert [q["call"] for q in req] == list(range(1, 18))
    assert {q["prompt"] for q in req} == {"pick up the pink cube"}
    assert {(tuple(q["image_shape"]), q["image_dtype"]) for q in req} == {((224, 224, 3), "uint8")}
    home = [0.0, 1.2, 1.2, 0.0, 0.0, 0.0]
    for q in req:  # MEASURED joints (near home_q) + the jaw fraction, never invented
        assert len(q["state"]) == 7 and max(abs(a - b) for a, b in zip(q["state"][:6], home)) < 0.02
        assert 0.0 <= q["state"][6] <= 1.0
    # the script: chunk 1 holds open, every later request gets the close chunk
    assert [q["reply_gripper"] for q in req] == [1.0] + [0.0] * 16
    # 8 grasp attempts: the first episode ran 3 chunks, the 7 later ones 2 each (chunks_after_close = 1)
    attempts = int(re.search(r"after (\d+) attempts", vla2["pick"]["error"]).group(1))
    assert attempts == 8 and 3 + 2 * (attempts - 1) == len(req) == 17


# ── B44: the launcher judge pass over a live proof turn ──────────────────────


def test_b44_the_receipt_came_from_the_confirmed_launcher_turn_and_was_never_written():
    rec, runs = _load("b44_judge_turn.json"), {r["tag"]: r for r in _b63()}
    assert rec["receipt"]["verified"] is True and rec["receipt"]["sim"] == "isaac"
    assert rec["receipt"]["props_reset"] == ["pink_cube", "green_cube"]
    an5 = runs["an5"]
    assert an5["proof"]["verified"] is True and an5["pick"]["postcondition"]["status"] == "confirmed"
    assert an5["pick"]["postcondition"]["channel"] == "physics"
    assert [runs[t]["proof"]["verified"] for t in ("an1", "an2", "an3", "an4")] == [False] * 4
    before, after = rec["receipt_sha256_before"], rec["receipt_sha256_after"]
    assert len(before) == 2 and len(set(before)) == 1 and before == after
    assert rec["run_summary"]["proof"]["sha256"] == before[0]
    assert rec["run_summary"]["proof"]["verified"] is True


def test_b44_the_confusion_matrix_is_the_codes_own_over_the_judged_step():
    from cascade.eval.progress_judge import RunVerdict, StepVerdict

    rec = _load("b44_judge_turn.json")
    steps = [StepVerdict(step=s["step"], skill=s["skill"], task=s["task"], hop=s["hop"], physics=s["physics"],
                         channel=s["channel"], ok=s["ok"], tier=s["tier"], duration_s=s["duration_s"])
             for s in rec["judge_verdict"]["steps"]]
    assert [(s.skill, s.hop, s.physics) for s in steps] == [("pick_and_place", 1.0, "confirmed")]
    confusion = RunVerdict(run_dir="", judge=rec["judge_verdict"]["judge"], steps=steps).confusion()
    judge = rec["run_summary"]["judge"]
    assert judge["status"] == "ok" and judge["backend"] == "vlm" and judge["judge"] == "vlm:Qwen/Qwen3.8-27B"
    assert {k: confusion[k] for k in ("tp", "tn", "fp", "fn", "n_scored")} == {
        k: judge["confusion"][k] for k in ("tp", "tn", "fp", "fn", "n_scored")} == {
        "tp": 1, "tn": 0, "fp": 0, "fn": 0, "n_scored": 1}
    assert rec["judge_verdict"]["steps"][0]["raw_tail"].rstrip().endswith("<score>+100%</score>")
    [banner] = rec["banner"]
    assert judge["summary_line"] in banner and "(physics verdict unchanged;" in banner
    assert round(judge["elapsed_s"], 1) == 7.7


# ── the docs quote the measured numbers ──────────────────────────────────────


def _numbers() -> dict[str, str]:
    on = [x for r in _b46_runs("1", with_stop=False) for x in r["reads"]]
    [stop] = [r for r in _b46_runs("1") if r["stop"]]
    vla2 = [r for r in _b63() if r["tag"] == "vla2"][0]
    c = _load("b44_judge_turn.json")["run_summary"]["judge"]["confusion"]
    return {
        "lane": _span([x["latency_s"] for x in on if x["tool"] != "camera_snapshot"]) + " s",
        "snapshot": _span([x["latency_s"] for x in on if x["tool"] == "camera_snapshot"]) + " s",
        "on": f"{sum(map(_confirmed, _b46_runs('1', with_stop=False)))}/4",
        "off": f"{sum(map(_confirmed, _b46_runs('0')))}/4",
        "stop": f"{stop['stop']['latency_s']:.3f} s",
        "requests": f"{len(vla2['vla_requests'])} policy requests over 8 grasp episodes",
        "matrix": f"tp={c['tp']} tn={c['tn']} fp={c['fp']} fn={c['fn']}",
    }


@pytest.mark.parametrize("doc,keys", [
    ("docs/evidence/w6-live-20261010/README.md", ("lane", "snapshot", "stop", "matrix")),
    ("docs/ROADMAP.md", ("lane", "snapshot", "on", "off", "stop", "requests", "matrix")),
    ("docs/ARCHITECTURE.md", ("lane", "snapshot", "on", "off", "stop", "matrix")),
])
def test_docs_quote_the_measured_numbers(doc, keys):
    text, numbers = _text(doc), _numbers()
    for key in keys:
        assert numbers[key] in text, (doc, key, numbers[key])


def test_readme_quotes_the_tables_and_the_stub_count():
    text = _text("docs/evidence/w6-live-20261010/README.md")
    on = [x for r in _b46_runs("1", with_stop=False) for x in r["reads"]]
    assert f"| reads answered before the pick returned | {len(on)} / 16 | 0 / 16 |" in text
    assert "| pick confirmed on the physics channel | 2 / 4 (on1, on4) | 1 / 4 (off3) |" in text
    assert "| `world_state` / `robot_knowledge` / `verify_last_action` latency | 0.001–0.003 s | 65.418–164.703 s |" in text
    assert "| `camera_snapshot` latency | 0.024–0.038 s | 65.468–164.778 s |" in text
    assert "**17** policy requests for the 8 grasp episodes" in text
    sha = _load("b44_judge_turn.json")["receipt_sha256_before"][0]
    assert sha in text
    n = _numbers()
    assert text.count(n["matrix"]) == 2  # the banner and the run-summary confusion line
    assert text.count(f"({n['on']} vs {n['off']})") == 1 and text.count(f"{n['on']} vs {n['off']} confirmed") == 1
