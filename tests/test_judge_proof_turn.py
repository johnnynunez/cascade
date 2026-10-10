"""B44 (ROADMAP near-term #6): the outcome judge as a METRIC of the launcher's
proof turn.

`scripts/launch.sh` proves READY with physics (`scripts/demo_proof.py`: one
pick confirmed on the physics channel, the same world reset). The judge pass
after it is opt-in (`--judge fake|vlm|grm` / `CASCADE_JUDGE`; default off),
runs `scripts/judge_run.py` over the proof turn's `pick_and_place` rows under
a hard wall-clock bound (`CASCADE_JUDGE_TIMEOUT_S`), and writes the
judge-vs-physics confusion matrix into `<evidence_dir>/run-summary.json` plus
ONE banner line. It is advisory: `proof.json` stays byte-identical, READY and
the exit status never change, and every judge failure reads `unavailable`.

Premises (pass on main by design): `judge_run.py` already computes `fn` for a
physics-confirmed pick the pictures missed, and on its own it has no
wall-clock bound against a judge endpoint that never answers -- which is why
the launcher pass needs one. Everything runs on CPU with the fake judge, a
recorded mock-stack trace, or a stub OpenAI-compatible server bound inside
this item's port block (45700-45799); no shared port is ever dialled.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from test_demo_proof import model_http_boundary as _model_http_boundary
from test_launch_delivery import launcher_boundary as _launcher_boundary

model_http_boundary = _model_http_boundary   # fixtures only (no test_* re-collection)
launcher_boundary = _launcher_boundary

REPO = Path(__file__).resolve().parents[1]
JUDGE_PROOF = REPO / "scripts/judge_proof.py"
JUDGE_RUN = REPO / "scripts/judge_run.py"
PORT_BLOCK = range(45700, 45800)
PRIVATE_PORTS = {"CASCADE_GRASPGENX_PORT": "45701", "CASCADE_OCCUPANCY_PORT": "45702",
                 "CASCADE_BRIDGE_PORT": "45703", "CASCADE_HUG_PORT": "45704"}


# --- fixtures: a launcher state dir exactly as demo_proof.run_proof leaves it ---
def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((24, 32, 3), value, np.uint8))
    assert ok
    return bytes(buf)


def _write_receipt(report: dict, state: Path) -> None:
    """demo_proof's own receipt writer (state/proof.json, atomic) -- the real
    format the pass reads, never a reimplementation."""
    runpy.run_path(str(REPO / "scripts/demo_proof.py"))["_write_receipt"](report, state)


def _proof_state(tmp_path: Path, rows: list[dict] | None = None, *, verified: bool = True,
                 trace: bool = True) -> dict:
    """STATE/proof.json + STATE/cascade-proof-<id>/ (evidence) + the bound MCP
    run dir (trace.jsonl + keyframes) of one launcher proof turn."""
    state = tmp_path / "state"
    session = "cascade-proof-" + "b44" * 8
    evidence = state / session
    evidence.mkdir(parents=True)
    run = tmp_path / "runs" / "mcp_4242_b44"
    (run / "keyframes").mkdir(parents=True)
    if rows is None:
        rows = [{"skill": "world_state"}, {"skill": "pick_and_place", "physics": "confirmed"},
                {"skill": "reset_scene"}, {"skill": "world_state"}]
    with (run / "trace.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            kb = ka = None
            if r.get("frames", True):
                kb, ka = f"keyframes/{i:04d}_before.jpg", f"keyframes/{i:04d}_after.jpg"
                (run / kb).write_bytes(_jpeg(20 + i))
                (run / ka).write_bytes(_jpeg(140 + i))
            res = {"ok": True, "tier": "llm"}
            if "physics" in r:
                res["postcondition"] = {"status": r["physics"], "channel": "physics"}
            f.write(json.dumps({"step": i, "t": 100.0 + i, "skill": r["skill"],
                                "args": {"object": "pink cube", "destination": "drop zone"},
                                "duration_ms": 2000, "result": res,
                                "keyframe_before": kb, "keyframe_after": ka}) + "\n")
    report = {"verified": verified, "model": "local/Qwen/Qwen3.8-27B", "sim": "isaac",
              "session_id": session, "started_at": 99.0, "evidence_dir": str(evidence), "profile": ""}
    if verified:
        report.update(trace=str(run / "trace.jsonl") if trace else str(tmp_path / "gone" / "trace.jsonl"),
                      props_reset=["pink_cube"])
    else:
        report["note"] = "robot proof explicitly skipped; this is not a verified demo"
    (evidence / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_receipt(report, state)
    return {"state": state, "proof": state / "proof.json", "evidence": evidence, "run": run,
            "report": report}


def _digests(*paths: Path) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _judge_config(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "judge-config.json"
    path.write_text(json.dumps(cfg))
    return path


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("CASCADE_JUDGE", "CASCADE_JUDGE_TIMEOUT_S", "CASCADE_JUDGE_CONFIG")}
    env.update(PRIVATE_PORTS)
    env.update({k: str(v) for k, v in extra.items()})
    for name in PRIVATE_PORTS:
        assert int(env[name]) in PORT_BLOCK
    return env


def _run_pass(h: dict, *args: str, env: dict | None = None, timeout: float = 240):
    """scripts/judge_proof.py exactly as launch.sh calls it."""
    return subprocess.run([sys.executable, str(JUDGE_PROOF), "--proof", str(h["proof"]), *args],
                          env=env or _env(), capture_output=True, text=True, timeout=timeout)


def _summary(h: dict) -> dict:
    return json.loads((h["evidence"] / "run-summary.json").read_text())


# --- stub OpenAI-compatible judge endpoints, bound inside this item's block ---
def _serve(behaviour: str, answer: str = "<score>+40%</score>"):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"posts": [], "stop": threading.Event(), "client_gone": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["posts"].append(body)
            if behaviour == "hang":
                # never answer; notice when the client process dies (its
                # socket is closed by the kernel)
                while not state["stop"].is_set():
                    ready, _, _ = select.select([self.connection], [], [], 0.05)
                    if ready:
                        try:
                            data = self.connection.recv(1, socket.MSG_PEEK)
                        except OSError:
                            data = b""
                        if not data:
                            state["client_gone"].set()
                            return
                return
            payload = json.dumps({"id": "b44", "object": "chat.completion", "created": 0,
                                  "model": body.get("model"),
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": answer}}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                            "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = None
    for port in range(45710, 45800):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        pytest.skip("no free port in this item's block 45710-45799")
    assert server.server_port in PORT_BLOCK
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"

    def close():
        state["stop"].set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    state["close"] = close
    return state


@pytest.fixture
def judge_endpoint(request):
    pytest.importorskip("openai")
    servers = []

    def make(behaviour: str, answer: str = "<score>+40%</score>"):
        s = _serve(behaviour, answer)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s["close"]()


def _local_vlm_config(tmp_path: Path, base_url: str, timeout_s: float = 30) -> Path:
    # the shape deploy/runtime/runtime.py writes for the local Qwen judge
    return _judge_config(tmp_path, {
        "backend": "vlm", "base_url": base_url, "api_key": "local-endpoint-no-auth",
        "model": "Qwen/Qwen3.8-27B", "fresh_session": False, "mode": "incremental",
        "temperature": 0.1, "top_p": 0.9, "max_tokens": 64, "timeout_s": timeout_s,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})


# --- premises: what main already has, and why the pass needs a bound -----------
def test_premise_judge_run_already_counts_fn_for_a_missed_confirmed_pick(tmp_path):
    """The metric exists (judge_run + RunVerdict.confusion): a judge that does
    not see the physics-confirmed pick as progress is `fn`. B44 is plumbing:
    getting that number into the launcher's run summary, bounded and advisory."""
    h = _proof_state(tmp_path)
    cfg = _judge_config(tmp_path, {"backend": "fake", "score": -0.3})
    p = subprocess.run([sys.executable, str(JUDGE_RUN), str(h["run"]), "--config", str(cfg)],
                       env=_env(), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    verdict = json.loads((h["run"] / "judge.json").read_text())
    assert verdict["confusion"]["fn"] == 1 and verdict["confusion"]["tp"] == 0
    assert verdict["confusion"]["n_with_physics"] == 1


def test_premise_judge_run_alone_has_no_wall_clock_bound(tmp_path, judge_endpoint):
    """A judge endpoint that accepts and never answers holds judge_run.py for
    the configured per-call timeout x the client's retries x every traced
    call. Nothing in judge_run cuts that -- the bound must live in the pass."""
    hung = judge_endpoint("hang")
    h = _proof_state(tmp_path)
    cfg = _local_vlm_config(tmp_path, hung["base_url"], timeout_s=600)
    child = subprocess.Popen([sys.executable, str(JUDGE_RUN), str(h["run"]), "--config", str(cfg)],
                             env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=8)
    finally:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


# --- the pass: config validation ----------------------------------------------
def test_backend_and_timeout_resolution_fail_closed():
    from cascade.eval.proof_judge import (DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S, OFF, PassConfigError,
                                          resolve_backend, resolve_timeout)

    for value in (None, "", "  ", "off", "OFF"):
        assert resolve_backend(value) == OFF
    for value in ("fake", "vlm", "grm", " VLM "):
        assert resolve_backend(value) == value.strip().lower()
    for value in ("gpt", "1", "true", "fake,vlm"):
        with pytest.raises(PassConfigError):
            resolve_backend(value)
    assert resolve_timeout(None) == resolve_timeout("") == DEFAULT_TIMEOUT_S == 180.0
    assert resolve_timeout("3") == 3.0 and resolve_timeout(str(MAX_TIMEOUT_S)) == MAX_TIMEOUT_S == 1800.0
    for value in ("0", "-1", "nan", "inf", "1800.5", "soon"):
        with pytest.raises(PassConfigError):
            resolve_timeout(value)


def test_invalid_timeout_reads_unavailable_and_never_starts_the_judge(tmp_path):
    h = _proof_state(tmp_path)
    before = _digests(h["proof"], h["evidence"] / "proof.json")
    p = _run_pass(h, "--judge", "fake", env=_env(CASCADE_JUDGE_TIMEOUT_S="nan"))
    assert p.returncode == 0, p.stderr
    assert p.stdout.count("\n") == 1 and p.stdout.startswith("unavailable (")
    assert "CASCADE_JUDGE_TIMEOUT_S" in p.stdout
    assert not (h["run"] / "judge.json").exists(), "the judge ran despite an invalid bound"
    assert _summary(h)["judge"]["status"] == "unavailable"
    assert _digests(h["proof"], h["evidence"] / "proof.json") == before


# --- the pass: confusion matrix into the run summary + one banner line -----------
def test_pass_writes_the_confusion_matrix_into_the_run_summary_and_one_banner_line(tmp_path):
    h = _proof_state(tmp_path)
    before = _digests(h["proof"], h["evidence"] / "proof.json")
    cfg = _judge_config(tmp_path, {"backend": "fake", "score": 0.4})
    p = _run_pass(h, "--judge", "fake", env=_env(CASCADE_JUDGE_CONFIG=cfg))
    assert p.returncode == 0, p.stderr
    line = p.stdout
    assert line.count("\n") == 1, f"banner must be ONE line: {line!r}"
    assert line.startswith("advisory judge=fake ") and "tp=1" in line and "fn=0" in line
    assert "physics verdict unchanged" in line and str(h["evidence"] / "run-summary.json") in line

    s = _summary(h)
    assert s["schema"] == "cascade.run_summary/1"
    j = s["judge"]
    assert j["status"] == "ok" and j["advisory"] is True and j["backend"] == "fake"
    assert j["confusion"] == {"tp": 1, "tn": 0, "fp": 0, "fn": 0, "n_scored": 1,
                              "n_with_physics": 1, "agreement": 1.0}
    assert j["skills"] == ["pick_and_place"] and j["timeout_s"] == 180.0
    assert j["summary_line"].startswith("judge=fake mode=incremental scored=1/1 ")
    assert j["mode"] == "incremental" and j["final_progress"] == 0.4 and j["unscored_physics"] == 0
    assert j["reason"] is None and Path(j["log"]).is_file()
    # the proof turn only: world_state / reset_scene rows were not judged
    verdict = json.loads(Path(j["verdict"]).read_text())
    assert [st["skill"] for st in verdict["steps"]] == ["pick_and_place"]
    # the physics verdict is copied, never re-derived, and the receipt is untouched
    assert s["proof"]["verified"] is True and s["proof"]["session_id"] == h["report"]["session_id"]
    assert s["proof"]["sha256"] == before[str(h["proof"])]
    assert _digests(h["proof"], h["evidence"] / "proof.json") == before
    # judge_run's own artifacts in the MCP run dir, as before
    assert "judge=fake " in (h["run"] / "summary.txt").read_text()


def test_fn_reaches_the_summary_and_the_banner_without_touching_the_verdict(tmp_path):
    h = _proof_state(tmp_path)
    before = _digests(h["proof"], h["evidence"] / "proof.json")
    cfg = _judge_config(tmp_path, {"backend": "fake", "score": -0.3})
    p = _run_pass(h, "--judge", "fake", env=_env(CASCADE_JUDGE_CONFIG=cfg))
    assert p.returncode == 0, p.stderr
    assert "fn=1" in p.stdout and "the pictures missed physics-confirmed progress" in p.stdout
    j = _summary(h)["judge"]
    assert j["status"] == "ok" and j["confusion"]["fn"] == 1 and j["confusion"]["tp"] == 0
    assert _summary(h)["proof"]["verified"] is True
    assert _digests(h["proof"], h["evidence"] / "proof.json") == before


def test_a_physics_refuted_pick_counts_as_fp_or_tn(tmp_path):
    h = _proof_state(tmp_path, [{"skill": "pick_and_place", "physics": "refuted"},
                                {"skill": "pick_and_place", "physics": "confirmed"}])
    cfg = _judge_config(tmp_path, {"backend": "fake", "score": 0.4})
    assert _run_pass(h, "--judge", "fake", env=_env(CASCADE_JUDGE_CONFIG=cfg)).returncode == 0
    c = _summary(h)["judge"]["confusion"]
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1, 1, 0, 0) and c["agreement"] == 0.5


def test_a_recorded_mock_stack_trace_is_judged_but_never_counted_as_physics(tmp_path, monkeypatch):
    """The runtime's real recording (build_runtime on the mock stack, one
    pick_and_place): keyframes are scored, but the mock's postcondition is not
    the physics channel, so the matrix stays empty -- an unknown is not an
    agreement."""
    from cascade.apps.demo import build_runtime
    from cascade.config import load_demo_config

    for name, port in PRIVATE_PORTS.items():
        monkeypatch.setenv(name, port)
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")   # after the port env (B34)
    assert int(cfg.get("grasp").get("graspgenx").get("port")) in PORT_BLOCK
    h = _proof_state(tmp_path, [])
    runtime, arm = build_runtime(cfg, h["run"] / "rec")
    try:
        runtime.execute("pick_and_place", {"object": "red cube"})
    finally:
        runtime.camera.close()
        arm.disconnect()
    recorded = h["run"] / "rec" / "trace.jsonl"
    row = json.loads(recorded.read_text().splitlines()[-1])
    assert row["skill"] == "pick_and_place" and row["keyframe_before"] and row["keyframe_after"]
    assert (row["result"].get("postcondition") or {}).get("channel") != "physics"
    report = {**h["report"], "trace": str(recorded)}
    _write_receipt(report, h["state"])
    p = _run_pass(h, "--judge", "fake", env=_env(CASCADE_JUDGE_CONFIG=_judge_config(
        tmp_path, {"backend": "fake", "score": 0.2})))
    assert p.returncode == 0, p.stderr
    j = _summary(h)["judge"]
    assert j["status"] == "ok" and j["confusion"]["n_scored"] == 1
    assert j["confusion"]["n_with_physics"] == 0 and j["confusion"]["agreement"] is None
    assert "agreement=n/a" in p.stdout


# --- the pass: every failure is `unavailable`, never a verdict change -------------
def test_judge_config_error_reads_unavailable(tmp_path):
    h = _proof_state(tmp_path)
    before = _digests(h["proof"], h["evidence"] / "proof.json")
    cfg = _judge_config(tmp_path, {"backend": "vlm"})   # no model: judge_run refuses (exit 2)
    p = _run_pass(h, "--judge", "vlm", env=_env(CASCADE_JUDGE_CONFIG=cfg))
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith("unavailable (") and "the physics verdict stands" in p.stdout
    j = _summary(h)["judge"]
    assert j["status"] == "unavailable" and "exited 2" in j["reason"] and "confusion" not in j
    assert _digests(h["proof"], h["evidence"] / "proof.json") == before


def test_unparseable_judge_answers_read_unavailable_not_zero_progress(tmp_path, judge_endpoint):
    vlm = judge_endpoint("answer", "I cannot tell from these images.")
    h = _proof_state(tmp_path)
    p = _run_pass(h, "--judge", "vlm", env=_env(CASCADE_JUDGE_CONFIG=_local_vlm_config(tmp_path, vlm["base_url"])))
    assert p.returncode == 0, p.stderr
    assert len(vlm["posts"]) == 1, "the real OpenAI-compatible path was not exercised"
    j = _summary(h)["judge"]
    assert j["status"] == "unavailable" and "no <score>" in j["reason"]
    assert j["confusion"]["n_scored"] == 0
    assert p.stdout.startswith("unavailable (")


def test_a_local_vlm_judge_scores_the_proof_pick_over_the_real_wire(tmp_path, judge_endpoint):
    """The parent's live command (local Qwen, OpenAI-compatible) uses this
    path: eight images, the model id and extra_body reach the endpoint."""
    vlm = judge_endpoint("answer", "<score>+40%</score>")
    h = _proof_state(tmp_path)
    p = _run_pass(h, "--judge", "vlm", env=_env(CASCADE_JUDGE_CONFIG=_local_vlm_config(tmp_path, vlm["base_url"])))
    assert p.returncode == 0, p.stderr
    j = _summary(h)["judge"]
    assert j["status"] == "ok" and j["judge"] == "vlm:Qwen/Qwen3.8-27B"
    assert j["confusion"]["tp"] == 1 and j["confusion"]["fn"] == 0
    body = vlm["posts"][0]
    assert body["model"] == "Qwen/Qwen3.8-27B"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert sum(part["type"] == "image_url" for part in body["messages"][0]["content"]) == 8


def test_a_hung_judge_is_cut_at_the_bound_and_its_process_killed(tmp_path, judge_endpoint):
    hung = judge_endpoint("hang")
    h = _proof_state(tmp_path)
    before = _digests(h["proof"], h["evidence"] / "proof.json")
    cfg = _local_vlm_config(tmp_path, hung["base_url"], timeout_s=600)   # per call: 600 s
    p = _run_pass(h, "--judge", "vlm", env=_env(CASCADE_JUDGE_CONFIG=cfg, CASCADE_JUDGE_TIMEOUT_S=12),
                  timeout=120)
    assert p.returncode == 0, p.stderr
    j = _summary(h)["judge"]
    assert j["status"] == "unavailable" and j["reason"].startswith("timed out after 12 s")
    assert j["timeout_s"] == 12.0 and 12 <= j["elapsed_s"] < 300 and "verdict" not in j
    assert p.stdout.startswith("unavailable (timed out after 12 s")
    # the judge process is gone (killed with its group, then reaped) ...
    with pytest.raises(ProcessLookupError):
        os.kill(j["judge_pid"], 0)
    # ... and, when it got as far as the endpoint, its connection died with it
    if hung["posts"]:
        assert hung["client_gone"].wait(10), "the killed judge still holds its connection"
    assert _digests(h["proof"], h["evidence"] / "proof.json") == before


def test_an_unverified_proof_is_skipped_and_the_judge_never_runs(tmp_path):
    h = _proof_state(tmp_path, verified=False)
    before = _digests(h["proof"], h["evidence"] / "proof.json")
    p = _run_pass(h, "--judge", "fake")
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith("skipped (") and p.stdout.count("\n") == 1
    s = _summary(h)
    assert s["judge"]["status"] == "skipped" and s["proof"]["verified"] is False
    assert not (h["run"] / "judge.json").exists() and not (h["evidence"] / "judge.json").exists()
    assert _digests(h["proof"], h["evidence"] / "proof.json") == before


@pytest.mark.parametrize("trace", ["missing", "not-a-trace"])
def test_a_receipt_without_its_trace_reads_unavailable(tmp_path, trace):
    h = _proof_state(tmp_path, trace=trace == "not-a-trace")
    if trace == "not-a-trace":   # an existing file that is not the runtime's trace.jsonl
        _write_receipt({**h["report"], "trace": str(h["run"] / "keyframes" / "0001_before.jpg")}, h["state"])
    p = _run_pass(h, "--judge", "fake")
    assert p.returncode == 0, p.stderr
    j = _summary(h)["judge"]
    assert j["status"] == "unavailable" and j["reason"].startswith("the proof receipt names no readable trace.jsonl")
    assert _summary(h)["proof"]["verified"] is True


def test_an_existing_run_summary_keeps_its_other_keys(tmp_path):
    h = _proof_state(tmp_path)
    (h["evidence"] / "run-summary.json").write_text(json.dumps({"operator_note": "booth A"}))
    assert _run_pass(h, "--judge", "fake").returncode == 0
    s = _summary(h)
    assert s["operator_note"] == "booth A" and s["judge"]["status"] == "ok"


def test_a_corrupt_run_summary_is_replaced_not_fatal(tmp_path):
    h = _proof_state(tmp_path)
    (h["evidence"] / "run-summary.json").write_text("[1, 2]")
    p = _run_pass(h, "--judge", "fake")
    assert p.returncode == 0 and p.stdout.startswith("advisory judge=fake ")
    assert _summary(h)["judge"]["status"] == "ok"


def test_a_physics_graded_pick_without_keyframes_is_named_unscored(tmp_path):
    h = _proof_state(tmp_path, [{"skill": "pick_and_place", "physics": "confirmed", "frames": False},
                                {"skill": "pick_and_place", "physics": "confirmed"}])
    p = _run_pass(h, "--judge", "fake")
    assert p.returncode == 0, p.stderr
    j = _summary(h)["judge"]
    assert j["status"] == "ok" and j["unscored_physics"] == 1 and j["confusion"]["n_scored"] == 1
    assert "1 physics-graded step(s) unscored" in p.stdout


def test_a_failed_rerun_leaves_no_stale_verdict(tmp_path):
    h = _proof_state(tmp_path)
    assert _run_pass(h, "--judge", "fake").returncode == 0
    assert (h["evidence"] / "judge.json").is_file()
    cfg = _judge_config(tmp_path, {"backend": "vlm"})   # judge_run refuses before judging
    assert _run_pass(h, "--judge", "vlm", env=_env(CASCADE_JUDGE_CONFIG=cfg)).returncode == 0
    assert not (h["evidence"] / "judge.json").exists(), "a failed re-run left the previous verdict"
    assert "verdict" not in _summary(h)["judge"]


def test_off_runs_nothing_and_writes_nothing(tmp_path):
    h = _proof_state(tmp_path)
    p = _run_pass(h, "--judge", "off")
    assert p.returncode == 0 and p.stdout.startswith("off (")
    assert not (h["evidence"] / "run-summary.json").exists() and not (h["run"] / "judge.json").exists()


@pytest.mark.parametrize("content, reason", [("{not json", "proof receipt unreadable: JSONDecodeError"),
                                             ("[]", "proof receipt is not a JSON object")])
def test_an_unreadable_receipt_reads_unavailable(tmp_path, content, reason):
    h = _proof_state(tmp_path)
    h["proof"].write_text(content)
    p = _run_pass(h, "--judge", "fake")
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith(f"unavailable ({reason})")
    assert not (h["run"] / "judge.json").exists()


def test_a_judge_that_cannot_start_or_writes_nothing_reads_unavailable(tmp_path):
    from cascade.eval.proof_judge import run_judge_pass

    h = _proof_state(tmp_path)
    block = run_judge_pass(h["proof"], "fake", repo=REPO, python=str(tmp_path / "no-such-python"), env=_env())
    assert block["status"] == "unavailable" and block["reason"].startswith("cannot start judge_run.py")
    silent = tmp_path / "silent-python"
    silent.write_text("#!/bin/sh\nexit 0\n")
    silent.chmod(0o755)
    block = run_judge_pass(h["proof"], "fake", repo=REPO, python=str(silent), env=_env())
    assert block["status"] == "unavailable" and block["reason"] == "judge_run.py wrote no verdict"


@pytest.mark.parametrize("where", ["outside", "missing", "empty"])
def test_the_pass_writes_only_into_the_evidence_dir_beside_the_receipt(tmp_path, where):
    h = _proof_state(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    report = {**h["report"], "evidence_dir": {"outside": str(outside), "missing": str(h["state"] / "gone"),
                                              "empty": ""}[where]}
    _write_receipt(report, h["state"])
    cwd = h["evidence"]   # a cwd inside the state dir: "" must never resolve to it
    p = subprocess.run([sys.executable, str(JUDGE_PROOF), "--proof", str(h["proof"]), "--judge", "fake"],
                       env=_env(), cwd=cwd, capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith("unavailable (the proof receipt names no evidence directory beside it)")
    assert not list(tmp_path.rglob("run-summary.json")) and not (h["run"] / "judge.json").exists()


def test_the_banner_stays_one_line_whatever_the_paths_hold(tmp_path):
    h = _proof_state(tmp_path / "odd\nname")
    p = _run_pass(h, "--judge", "fake")
    assert p.returncode == 0, p.stderr
    assert p.stdout.count("\n") == 1 and p.stdout.startswith("advisory judge=fake ")


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory modes")
def test_an_internal_error_still_prints_one_unavailable_line_and_exits_zero(tmp_path):
    h = _proof_state(tmp_path)
    h["evidence"].chmod(0o500)   # the pass cannot write its log or the summary
    try:
        p = _run_pass(h, "--judge", "fake")
    finally:
        h["evidence"].chmod(0o700)
    assert p.returncode == 0, p.stderr
    assert p.stdout.count("\n") == 1 and p.stdout.startswith("unavailable (judge pass error: PermissionError")


def _gone(pid: int, wait_s: float = 10.0) -> bool:
    """Dead (or a zombie waiting for its reaper) within wait_s."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        try:
            if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except (OSError, IndexError):
            return True
        time.sleep(0.05)
    return False


def _spawning_judge(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A stand-in judge interpreter that starts a grandchild and blocks: the
    bound must take the whole process group down, not just the child."""
    script, child_pid, grandchild_pid = tmp_path / "judge-python", tmp_path / "child.pid", tmp_path / "grandchild.pid"
    script.write_text(f"#!/bin/sh\necho $$ > '{child_pid}'\nsleep 300 &\necho $! > '{grandchild_pid}'\nwait\n")
    script.chmod(0o755)
    return script, child_pid, grandchild_pid


def _wait_file(path: Path, wait_s: float = 30.0) -> int:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError(f"{path} never written")


def _kill_quietly(*pidfiles: Path) -> None:
    for f in pidfiles:
        if f.exists() and f.read_text().strip():
            try:
                os.kill(int(f.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_the_bound_kills_the_judges_whole_process_group(tmp_path):
    from cascade.eval.proof_judge import run_judge_pass

    h = _proof_state(tmp_path)
    script, child_pid, grandchild_pid = _spawning_judge(tmp_path)
    try:
        block = run_judge_pass(h["proof"], "fake", repo=REPO, timeout_s=3, python=str(script), env=_env())
        assert block["status"] == "unavailable" and block["reason"].startswith("timed out after 3 s")
        assert block["judge_pid"] == _wait_file(child_pid)
        assert _gone(_wait_file(child_pid)) and _gone(_wait_file(grandchild_pid)), \
            "the judge's grandchild survived the bound"
    finally:
        _kill_quietly(child_pid, grandchild_pid)


def test_ctrl_c_during_the_pass_kills_the_judge_group_and_propagates(tmp_path, monkeypatch):
    from cascade.eval import proof_judge

    h = _proof_state(tmp_path)
    script, child_pid, grandchild_pid = _spawning_judge(tmp_path)

    class InterruptedPopen(subprocess.Popen):
        def wait(self, timeout=None):
            if timeout is not None:          # the bounded wait: the operator hits Ctrl-C
                _wait_file(grandchild_pid)
                raise KeyboardInterrupt
            return super().wait()

    monkeypatch.setattr(proof_judge.subprocess, "Popen", InterruptedPopen)
    try:
        with pytest.raises(KeyboardInterrupt):
            proof_judge.run_judge_pass(h["proof"], "fake", repo=REPO, timeout_s=60, python=str(script), env=_env())
        assert _gone(_wait_file(child_pid)) and _gone(_wait_file(grandchild_pid)), \
            "Ctrl-C left an orphan judge"
    finally:
        _kill_quietly(child_pid, grandchild_pid)


# --- the launcher: opt-in, one banner line, status and exit unchanged ------------
def _launch(h: dict, *extra: str, drop_no_judge: bool = True, env: dict | None = None):
    command = [c for c in h["command"] if not (drop_no_judge and c == "--no-judge")] + list(extra)
    e = {**h["env"], **PRIVATE_PORTS}
    for k in ("CASCADE_JUDGE", "CASCADE_JUDGE_TIMEOUT_S", "CASCADE_JUDGE_CONFIG"):
        e.pop(k, None)
    e.update(env or {})
    return subprocess.run(command, env=e, capture_output=True, text=True, timeout=120)


def _launcher_with_judge_scripts(h: dict) -> dict:
    for name in ("judge_proof.py", "judge_run.py"):
        if (REPO / "scripts" / name).exists():     # main has no judge_proof.py (RED export)
            shutil.copy2(REPO / "scripts" / name, h["repo"] / "scripts" / name)
    return h


def _state(h: dict) -> Path:
    return Path(h["env"]["CASCADE_LAUNCH_STATE"]) / "profile-isolated-test"


def test_launcher_default_runs_no_judge(launcher_boundary):
    """Golden: with neither --judge nor CASCADE_JUDGE the launcher starts no
    judge process and prints no judge line (the same as --no-judge)."""
    h = _launcher_with_judge_scripts(launcher_boundary)
    launch = _launch(h)
    assert launch.returncode == 0, launch.stdout + launch.stderr
    assert "judge:" not in launch.stdout and "judging" not in launch.stdout
    assert not list(_state(h).rglob("run-summary.json"))
    assert "STARTED (UNVERIFIED" in launch.stdout


@pytest.mark.parametrize("how", ["env", "flag"])
def test_launcher_opt_in_adds_one_judge_line_and_keeps_status_and_exit(launcher_boundary, how):
    h = _launcher_with_judge_scripts(launcher_boundary)
    launch = _launch(h, *(("--judge", "fake") if how == "flag" else ()),
                     env={"CASCADE_JUDGE": "fake"} if how == "env" else None)
    assert launch.returncode == 0, launch.stdout + launch.stderr
    banner = [line for line in launch.stdout.splitlines() if line.startswith("         judge:")]
    assert len(banner) == 1 and "skipped (" in banner[0], launch.stdout
    assert "STARTED (UNVERIFIED" in launch.stdout and "[launch] READY" not in launch.stdout
    summaries = list(_state(h).rglob("run-summary.json"))
    assert len(summaries) == 1
    s = json.loads(summaries[0].read_text())
    receipt = json.loads((_state(h) / "proof.json").read_text())
    assert s["judge"]["status"] == "skipped" and s["proof"]["verified"] is False
    assert s["proof"]["session_id"] == receipt["session_id"]
    assert summaries[0].parent == Path(receipt["evidence_dir"])


def test_no_judge_wins_over_the_environment(launcher_boundary):
    h = _launcher_with_judge_scripts(launcher_boundary)
    launch = _launch(h, drop_no_judge=False, env={"CASCADE_JUDGE": "fake"})
    assert launch.returncode == 0, launch.stdout + launch.stderr
    assert "judge:" not in launch.stdout
    assert not list(_state(h).rglob("run-summary.json"))


@pytest.mark.parametrize("how", ["env", "flag"])
def test_launcher_refuses_an_unknown_judge_before_starting_anything(launcher_boundary, how):
    h = _launcher_with_judge_scripts(launcher_boundary)
    launch = _launch(h, *(("--judge", "gpt") if how == "flag" else ()),
                     env={"CASCADE_JUDGE": "gpt"} if how == "env" else None)
    assert launch.returncode == 2, launch.stdout + launch.stderr
    assert "off|fake|vlm|grm" in launch.stderr
    host_log = Path(h["env"]["HOST_LOG"])
    assert not host_log.exists() or not host_log.read_text().strip(), "OpenClaw was touched"


def _judge_block() -> str:
    text = (REPO / "scripts/launch.sh").read_text()
    blocks = re.findall(r"^\s*# >>> judge pass[^\n]*\n(.*?)^\s*# <<< judge pass", text, flags=re.M | re.S)
    assert len(blocks) == 1, "launch.sh must carry exactly one marked judge-pass block"
    return blocks[0]


def _run_block(state: Path, judge: str, py: str, timeout: float = 240) -> subprocess.CompletedProcess:
    script = ("set -euo pipefail\nlog() { printf '[launch] %s\\n' \"$*\"; }\n" + _judge_block()
              + '\nprintf "NOTE=%s\\n" "$JUDGE_NOTE"\n')
    env = _env(JUDGE=judge, PY=py, REPO=str(REPO), STATE_DIR=str(state))
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=timeout)


def test_launcher_block_judges_a_verified_proof_turn(tmp_path):
    h = _proof_state(tmp_path)
    before = _digests(h["proof"], h["evidence"] / "proof.json")
    p = _run_block(h["state"], "fake", sys.executable)
    assert p.returncode == 0, p.stdout + p.stderr
    note = [line for line in p.stdout.splitlines() if line.startswith("NOTE=")]
    assert len(note) == 1 and note[0].startswith("NOTE=advisory judge=fake ") and "tp=1" in note[0]
    assert _summary(h)["judge"]["status"] == "ok"
    assert _digests(h["proof"], h["evidence"] / "proof.json") == before


def test_launcher_block_survives_a_crashing_judge_pass(tmp_path):
    h = _proof_state(tmp_path)
    crash = tmp_path / "crash-python"
    crash.write_text("#!/bin/sh\necho 'Traceback: boom' >&2\nexit 1\n")
    crash.chmod(0o755)
    p = _run_block(h["state"], "fake", str(crash))
    assert p.returncode == 0, p.stdout + p.stderr
    note = [line for line in p.stdout.splitlines() if line.startswith("NOTE=")]
    assert len(note) == 1 and note[0].startswith("NOTE=unavailable (the judge pass did not report")


def test_launcher_block_is_a_no_op_when_off(tmp_path):
    h = _proof_state(tmp_path)
    p = _run_block(h["state"], "off", sys.executable)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "NOTE=\n" in p.stdout and "judging" not in p.stdout
    assert not (h["evidence"] / "run-summary.json").exists()
