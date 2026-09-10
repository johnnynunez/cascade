"""Robo-Dopamine-style progress judge (eval/progress_judge.py).

Pins: the upstream prompt/parse contract (8 interleaved images, last <score>
wins, clip to [-1,1], garbled = error not 0), the three fusion modes'
arithmetic (examples/inference.py), judging a recorded run from its
trace.jsonl + keyframes, and the judge-vs-physics calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cascade.eval.progress_judge import (
    GRM_PROMPT,
    N_IMAGES,
    FakeJudge,
    JudgeError,
    build_images,
    fuse_progress,
    interleave,
    judge_run,
    make_judge,
    parse_score,
)


def _jpeg(v: int) -> bytes:
    img = np.full((24, 32, 3), v, dtype=np.uint8)
    return bytes(cv2.imencode(".jpg", img)[1])


# --- prompt + parser -------------------------------------------------------
def test_prompt_has_exactly_eight_image_slots_and_the_task_twice():
    assert GRM_PROMPT.count("<image>") == N_IMAGES
    assert GRM_PROMPT.count("{task}") == 2                      # <Task> block + criterion 1


def test_interleave_alternates_text_and_images_in_prompt_order():
    imgs = [_jpeg(i * 10) for i in range(N_IMAGES)]
    content = interleave("pick the cube", imgs)
    kinds = [c["type"] for c in content]
    assert kinds.count("image_url") == N_IMAGES
    assert kinds[0] == "text" and "pick the cube" in content[0]["text"]
    # images appear in the given order (base64 of each)
    import base64
    urls = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
    for u, b in zip(urls, imgs):
        assert u.endswith(base64.b64encode(b).decode())
    with pytest.raises(JudgeError):
        interleave("x", imgs[:7])


@pytest.mark.parametrize("raw,score", [
    ("<score>+35%</score>", 0.35),
    ("<score>-20%</score>", -0.20),
    ("<score>0%</score>", 0.0),
    ("thinking...\n<score>+100%</score>", 1.0),
    ("<score>+150%</score>", 1.0),                                # clipped like upstream
    ("<score>+10%</score> no wait <score>+40%</score>", 0.40),    # last block wins (upstream split)
    ("<SCORE> 12 </SCORE>", 0.12),
])
def test_parse_score(raw, score):
    assert parse_score(raw) == pytest.approx(score)


def test_parse_score_refuses_garbage_instead_of_scoring_zero():
    # upstream silently scores 0.0 on parse failure; here it is an error so a
    # broken judge cannot masquerade as "no progress"
    with pytest.raises(JudgeError):
        parse_score("I think the robot did well.")
    with pytest.raises(JudgeError):
        parse_score("")


def test_build_images_single_view_repeats_front_and_blanks_missing_goal():
    b, a = _jpeg(10), _jpeg(200)
    imgs = build_images(b, a)
    assert len(imgs) == N_IMAGES
    assert imgs[0] == b                       # ref start falls back to BEFORE
    assert imgs[1] not in (a, b)              # blank goal, not a real frame
    assert imgs[2] == imgs[3] == imgs[4] == b  # before front + 2 wrist slots
    assert imgs[5] == imgs[6] == imgs[7] == a
    goal = _jpeg(77)
    assert build_images(b, a, ref_end=goal)[1] == goal


# --- fusion arithmetic (upstream examples/inference.py) --------------------
def test_incremental_fusion_matches_upstream_formula():
    prog, hops = fuse_progress([0.5, 0.5, -0.5], "incremental")
    # p0 = 0.5; p1 = 0.5 + 0.5*0.5 = 0.75; p2 = 0.75 + 0.75*(-0.5) = 0.375
    assert prog == pytest.approx([0.5, 0.75, 0.375])
    assert hops == pytest.approx([0.5, 0.5, -0.5])


def test_forward_and_backward_fusion():
    prog, hops = fuse_progress([0.2, 0.7], "forward")
    assert prog == pytest.approx([0.2, 0.7]) and hops == pytest.approx([0.2, 0.5])
    prog, hops = fuse_progress([-0.8, -0.3], "backward")      # goal->current is negative
    assert prog == pytest.approx([0.2, 0.7]) and hops == pytest.approx([0.2, 0.5])
    with pytest.raises(JudgeError):
        fuse_progress([0.1], "sideways")


# --- judging a run ----------------------------------------------------------
def _write_run(tmp_path: Path, rows: list[dict]) -> Path:
    run = tmp_path / "run"
    (run / "keyframes").mkdir(parents=True)
    with (run / "trace.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            kb = ka = None
            if r.get("frames", True):
                kb, ka = f"keyframes/{i:04d}_before.jpg", f"keyframes/{i:04d}_after.jpg"
                (run / kb).write_bytes(_jpeg(20 + i))
                (run / ka).write_bytes(_jpeg(120 + i))
            res = {"ok": r.get("ok", True), "tier": r.get("tier", "reflex")}
            if "physics" in r:
                res["postcondition"] = {"status": r["physics"], "channel": r.get("channel", "physics")}
            f.write(json.dumps({"step": i, "skill": r.get("skill", "pick_and_place"), "args": r.get("args", {"object": "red object"}),
                                "duration_ms": r.get("ms", 1000), "result": res,
                                "keyframe_before": kb, "keyframe_after": ka}) + "\n")
    return run


def test_judge_run_scores_each_pair_and_renders_the_task_text(tmp_path):
    run = _write_run(tmp_path, [{"physics": "confirmed"}, {"skill": "grasp_object", "args": {"label": "cup"}, "physics": "refuted"}])
    j = FakeJudge(script=["<score>+60%</score>", "<score>-30%</score>"])
    v = judge_run(run, j)
    assert [s.hop for s in v.steps] == pytest.approx([0.6, -0.3])
    assert j.calls[0]["task"] == "pick and place the red object"
    assert j.calls[1]["task"] == "grasp object the cup"
    assert j.calls[0]["ref_start"] is True and j.calls[0]["ref_end"] is False
    assert v.progress == pytest.approx([0.6, 0.6 + 0.6 * -0.3])
    assert v.judge == "fake"


def test_confusion_matrix_against_the_physics_channel(tmp_path):
    run = _write_run(tmp_path, [
        {"physics": "confirmed"},                       # judge + -> tp
        {"physics": "refuted"},                         # judge - -> tn
        {"physics": "refuted"},                         # judge + -> fp
        {"physics": "confirmed"},                       # judge - -> fn
        {"physics": "confirmed", "channel": "belief"},  # not physics: excluded from calibration
        {},                                             # no postcondition: excluded
    ])
    j = FakeJudge(script=["<score>+50%</score>", "<score>-50%</score>", "<score>+50%</score>",
                          "<score>-50%</score>", "<score>+50%</score>", "<score>+50%</score>"])
    c = judge_run(run, j).confusion()
    assert (c["tp"], c["tn"], c["fp"], c["fn"]) == (1, 1, 1, 1)
    assert c["n_with_physics"] == 4 and c["n_scored"] == 6
    assert c["agreement"] == pytest.approx(0.5)


def test_missing_keyframes_are_counted_not_skipped(tmp_path):
    run = _write_run(tmp_path, [{"frames": False, "physics": "confirmed"}, {"physics": "confirmed"}])
    v = judge_run(run, FakeJudge(0.4))
    assert len(v.steps) == 2
    assert v.steps[0].hop is None and v.steps[0].error == "missing keyframe pair"
    assert v.steps[1].hop == pytest.approx(0.4)
    assert v.confusion()["n_scored"] == 1
    assert v.steps[0].agrees_with_physics() is None          # unknown is not agreement


def test_per_tier_hop_per_second(tmp_path):
    run = _write_run(tmp_path, [{"tier": "reflex", "ms": 2000}, {"tier": "llm", "ms": 8000}, {"tier": "reflex", "ms": 2000}])
    v = judge_run(run, FakeJudge(script=["<score>+40%</score>", "<score>+40%</score>", "<score>+20%</score>"]))
    t = v.per_tier()
    assert t["reflex"]["n"] == 2 and t["reflex"]["hop_per_s"] == pytest.approx(0.6 / 4.0)
    assert t["llm"]["hop_per_s"] == pytest.approx(0.4 / 8.0)


def test_judge_error_on_one_step_does_not_lose_the_run(tmp_path):
    run = _write_run(tmp_path, [{}, {}])
    v = judge_run(run, FakeJudge(script=["garbage", "<score>+30%</score>"]))
    assert v.steps[0].hop is None and "no <score>" in (v.steps[0].error or "")
    assert v.steps[1].hop == pytest.approx(0.3)


def test_transport_failure_on_one_step_is_recorded_not_fatal(tmp_path):
    """A 503 / timeout from the model endpoint (measured against the OpenClaw
    gateway) must land in that step's error, and the next pair still gets
    judged."""
    run = _write_run(tmp_path, [{}, {}])

    class Flaky(FakeJudge):
        def score(self, task, before, after, **refs):
            if not self.calls:
                self.calls.append({})
                raise ConnectionError("Error code: 503")
            return 0.25

    v = judge_run(run, Flaky())
    assert v.steps[0].hop is None and "503" in (v.steps[0].error or "")
    assert v.steps[1].hop == pytest.approx(0.25)
    assert v.progress == pytest.approx([0.25])


def test_skill_filter(tmp_path):
    run = _write_run(tmp_path, [{"skill": "get_observation"}, {"skill": "pick_and_place"}])
    v = judge_run(run, FakeJudge(0.5), skills={"pick_and_place"})
    assert [s.skill for s in v.steps] == ["pick_and_place"]


def test_judge_run_requires_a_trace(tmp_path):
    with pytest.raises(JudgeError):
        judge_run(tmp_path, FakeJudge())


# --- factory ----------------------------------------------------------------
def test_make_judge_has_no_silent_default():
    with pytest.raises(JudgeError):
        make_judge({})
    with pytest.raises(JudgeError):
        make_judge({"backend": "grm"})          # model required
    assert isinstance(make_judge({"backend": "fake", "score": 0.1}), FakeJudge)


def test_openai_judge_describes_which_weights_answered(monkeypatch):
    pytest.importorskip("openai")
    from cascade.eval.progress_judge import OpenAICompatJudge

    grm = OpenAICompatJudge(model="grm", base_url="http://localhost:8000/v1", kind="grm")
    vlm = OpenAICompatJudge(model="gpt-x", base_url=None, api_key="k", kind="vlm")
    assert grm.describe() == "grm:grm" and vlm.describe() == "vlm:gpt-x"
    assert grm.temperature == 0.1 and grm.top_p == 0.9          # upstream sampling


def test_openai_judge_sends_the_interleaved_prompt_and_parses(monkeypatch):
    pytest.importorskip("openai")
    from cascade.eval.progress_judge import OpenAICompatJudge

    j = OpenAICompatJudge(model="grm", base_url="http://localhost:1/v1", kind="grm")
    seen = {}

    class _Resp:
        class _C:
            class _M:
                content = "<score>+45%</score>"
            message = _M()
        choices = [_C()]

    def fake_create(**kw):
        seen.update(kw)
        return _Resp()

    monkeypatch.setattr(j._client.chat.completions, "create", fake_create)
    hop = j.score("pick and place the red object", _jpeg(1), _jpeg(2))
    assert hop == pytest.approx(0.45)
    assert seen["model"] == "grm" and seen["temperature"] == 0.1
    content = seen["messages"][0]["content"]
    assert sum(c["type"] == "image_url" for c in content) == N_IMAGES
    assert "pick and place the red object" in content[0]["text"]


def test_local_endpoints_bypass_the_system_proxy():
    """Measured: the macOS VPN proxy (127.0.0.1:1082) inherited from the
    environment turned every multi-image POST to the loopback gateway into an
    empty 503. Local base_urls must build a client that ignores proxy env."""
    pytest.importorskip("openai")
    from cascade.eval.progress_judge import OpenAICompatJudge, _is_local_url

    assert _is_local_url("http://127.0.0.1:18789/v1")
    assert _is_local_url("http://localhost:8000/v1")
    assert _is_local_url("http://192.168.1.20:8000/v1")
    assert _is_local_url("http://spark.local:8000/v1")
    assert not _is_local_url("https://api.openai.com/v1")
    local = OpenAICompatJudge(model="grm", base_url="http://127.0.0.1:18789/v1", kind="grm")
    remote = OpenAICompatJudge(model="gpt", base_url="https://api.example.com/v1", api_key="k", kind="vlm")
    assert local.proxy_bypassed and not remote.proxy_bypassed
    assert getattr(local._client._client, "trust_env", None) is False


def test_local_endpoint_ignores_proxy_env(monkeypatch):
    pytest.importorskip("openai")
    from cascade.eval.progress_judge import OpenAICompatJudge

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1082")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1082")
    local = OpenAICompatJudge(model="grm", base_url="http://127.0.0.1:18789/v1", kind="grm")
    remote = OpenAICompatJudge(model="gpt", base_url="https://api.example.com/v1", api_key="k", kind="vlm")
    # under trust_env the SDK's httpx client mounts one proxy transport per
    # env key; the local client must have none, the remote one keeps them
    assert getattr(local._client._client, "_mounts", {}) == {}
    assert len(getattr(remote._client._client, "_mounts", {})) >= 1


# --- the judge as a METRIC in the run artifact (ROADMAP #6) -------------------
def test_write_appends_one_metric_line_to_summary_and_is_idempotent(tmp_path):
    run = _write_run(tmp_path, [{"physics": "confirmed"}, {"physics": "confirmed"}])
    (run / "summary.txt").write_text("task: pick and place the red object\nresult: ok\n")
    v = judge_run(run, FakeJudge(script=["<score>+60%</score>", "<score>-20%</score>"]))
    out = v.write()
    assert out == run / "judge.json" and out.exists()
    lines = (run / "summary.txt").read_text().splitlines()
    assert lines[:2] == ["task: pick and place the red object", "result: ok"]  # the run's own lines survive
    assert lines[-1].startswith("judge=fake ")
    assert " tp=1 " in lines[-1] and " fn=1 " in lines[-1] and "scored=2/2" in lines[-1]
    # re-judging replaces THIS judge's line rather than stacking a second one
    v2 = judge_run(run, FakeJudge(script=["<score>+60%</score>", "<score>+20%</score>"]))
    v2.write()
    lines = (run / "summary.txt").read_text().splitlines()
    assert sum(l.startswith("judge=fake ") for l in lines) == 1
    assert " fn=0 " in lines[-1]


def test_judge_run_script_strict_exit_code_is_the_metric(tmp_path):
    """`scripts/judge_run.py --strict` exits 3 when the judge missed a
    physics-confirmed step (fn > 0), 0 otherwise -- so the launcher and CI
    can gate on the pictures agreeing with the physics."""
    import subprocess
    import sys
    from pathlib import Path as _P

    repo = _P(__file__).resolve().parents[1]
    run = _write_run(tmp_path, [{"physics": "confirmed"}])
    # fake score < 0 on a confirmed step: fn=1
    r = subprocess.run([sys.executable, str(repo / "scripts/judge_run.py"), str(run),
                        "--judge", "fake", "--fake-score", "-0.5", "--strict"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "fn=1" in r.stdout
    r = subprocess.run([sys.executable, str(repo / "scripts/judge_run.py"), str(run),
                        "--judge", "fake", "--fake-score", "0.5", "--strict"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (run / "summary.txt").exists() and "judge=fake" in (run / "summary.txt").read_text()
