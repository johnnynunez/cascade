"""B66: the outcome judge's score must fit its token budget.

B44-live (docs/evidence/b44-live-qwen-judge-20261009/) asked the local
Qwen3.8-27B for ONE line `<score>+NN%</score>` with the GRM prompt and the
1536-token budget `deploy/runtime/runtime.py` ships. 12 of 72 steps came back
with no `<score>`: 11 cut mid-reasoning at exactly 1536 tokens
(finish_reason=length) and 1 stopped with the score written WITHOUT the tags
("Score: -50%."). 11 of the 12 were physics-refuted and none confirmed, so the
abstentions silently remove `tn` cases from the judge-vs-physics matrix.

The opt-in fix (`eval.judge.score_followup_tokens`, default 0 = the old single
call): when the first answer has no parseable score, ONE bounded follow-up
call re-sends the same turn plus that answer and asks only for the GRM
prompt's own output-format line. The step record says how the score was
obtained (`score_via` first / follow-up / none) and what the follow-up cost;
a follow-up without a score stays unscored with its reason -- never a made-up
score -- and the whole thing runs inside judge_run.py, i.e. inside the B44
launcher pass bound (`CASCADE_JUDGE_TIMEOUT_S`, process-group kill).

Everything runs on CPU against an in-process OpenAI-compatible stub bound to a
port the OS assigns (read back from the server, B64), replaying the recorded
B44-live answers. Premises and goldens pass on main by design.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading

import cv2
import numpy as np
import pytest

from test_judge_proof_turn import _judge_config, _proof_state

REPO = Path(__file__).resolve().parents[1]
JUDGE_RUN = REPO / "scripts/judge_run.py"
JUDGE_PROOF = REPO / "scripts/judge_proof.py"
EVIDENCE = REPO / "docs/evidence/b44-live-qwen-judge-20261009/results.json"
PORT_BLOCK = range(46700, 46800)
PRIVATE_PORTS = {"CASCADE_GRASPGENX_PORT": "46701", "CASCADE_OCCUPANCY_PORT": "46702",
                 "CASCADE_BRIDGE_PORT": "46703", "CASCADE_HUG_PORT": "46704"}
MODEL = "Qwen/Qwen3.8-27B"
#: the request keys a plain OpenAI chat-completions call carries here
#: (`chat_template_kwargs` is the judge's extra_body, merged by the SDK)
STANDARD_KEYS = {"model", "messages", "temperature", "top_p", "max_tokens", "chat_template_kwargs"}


# --- recorded B44-live behaviour ------------------------------------------------
def _b44_steps() -> list[dict]:
    return json.loads(EVIDENCE.read_text())["steps"]


def _b44_unscored() -> list[dict]:
    return [s for s in _b44_steps() if s["hop_1536"] is None]


def _b44(run: str) -> dict:
    return next(s for s in _b44_steps() if s["run"] == run)


def _first_answer(step: dict) -> dict:
    """The recorded 1536-token answer of a B44-live step, as the stub replays it."""
    return {"content": step["raw_tail_1536"], "finish_reason": step["finish_1536"],
            "completion_tokens": step["tokens_1536"]}


def _score_line(hop: float) -> str:
    pct = round(hop * 100)
    return f"<score>{'+' if pct > 0 else ''}{pct}%</score>"


# --- stub OpenAI-compatible endpoint on an OS-assigned port ----------------------
class _Stub:
    """Replays a script of answers, one per POST, in order. Each entry is
    `{"content", "finish_reason", "completion_tokens"}`, `{"status": 4xx}` (an
    error reply) or `{"hang": True}` (accept, never answer; notice when the
    client process dies). `strict=True` answers 400 to any request carrying a
    key a plain chat-completions endpoint does not accept (`grammar`,
    `response_format`, ...), the way OpenAI rejects llama.cpp's extensions."""

    def __init__(self, script: list[dict], *, strict: bool = False):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.script = list(script)
        self.posts: list[dict] = []
        self.headers: list[dict] = []
        self.rejected: list[dict] = []
        self.stop = threading.Event()
        self.client_gone = threading.Event()
        lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                unknown = sorted(set(body) - STANDARD_KEYS)
                if strict and unknown:
                    stub.rejected.append(body)
                    self._reply(400, {"error": {"message": f"Unrecognized request argument supplied: "
                                                           f"{', '.join(unknown)}", "type": "invalid_request_error"}})
                    return
                with lock:
                    stub.posts.append(body)
                    stub.headers.append({k.lower(): v for k, v in self.headers.items()})
                    step = stub.script.pop(0) if stub.script else {"status": 500, "message": "stub script exhausted"}
                if step.get("hang"):
                    while not stub.stop.is_set():
                        ready, _, _ = select.select([self.connection], [], [], 0.05)
                        if ready:
                            try:
                                data = self.connection.recv(1, socket.MSG_PEEK)
                            except OSError:
                                data = b""
                            if not data:
                                stub.client_gone.set()
                                return
                    return
                if "status" in step:
                    self._reply(step["status"], {"error": {"message": step.get("message", "rejected"),
                                                           "type": "invalid_request_error"}})
                    return
                self._reply(200, {"id": "b66", "object": "chat.completion", "created": 0, "model": body.get("model"),
                                  "choices": [{"index": 0, "finish_reason": step["finish_reason"],
                                               "message": {"role": "assistant", "content": step["content"]}}],
                                  "usage": {"prompt_tokens": 2400, "completion_tokens": step["completion_tokens"],
                                            "total_tokens": 2400 + step["completion_tokens"]}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def stub():
    pytest.importorskip("openai")
    made: list[_Stub] = []

    def make(script, **kw):
        s = _Stub(script, **kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


# --- helpers ---------------------------------------------------------------------
def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("CASCADE_JUDGE", "CASCADE_JUDGE_TIMEOUT_S", "CASCADE_JUDGE_CONFIG")}
    env.update(PRIVATE_PORTS)
    env.update({k: str(v) for k, v in extra.items()})
    for name in PRIVATE_PORTS:
        assert int(env[name]) in PORT_BLOCK
    return env


def _qwen_config(base_url: str, **extra) -> dict:
    """The judge JSON deploy/runtime/runtime.py writes for the local Qwen
    (1536 tokens), pointed at the stub."""
    return {"backend": "vlm", "base_url": base_url, "api_key": "local-endpoint-no-auth", "model": MODEL,
            "fresh_session": False, "mode": "incremental", "temperature": 0.1, "top_p": 0.9,
            "max_tokens": 1536, "timeout_s": 30,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, **extra}


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((24, 32, 3), value, np.uint8))
    assert ok
    return bytes(buf)


def _recorded_run(tmp_path: Path, physics: list[str | None]) -> Path:
    """A run dir as the runtime records it: one pick_and_place row per entry,
    each with a BEFORE/AFTER keyframe pair and its physics postcondition."""
    run = tmp_path / "run"
    (run / "keyframes").mkdir(parents=True)
    rows = []
    for i, status in enumerate(physics):
        kb, ka = f"keyframes/{i:04d}_before.jpg", f"keyframes/{i:04d}_after.jpg"
        (run / kb).write_bytes(_jpeg(20 + i))
        (run / ka).write_bytes(_jpeg(140 + i))
        res = {"ok": True, "tier": "llm"}
        if status is not None:
            res["postcondition"] = {"status": status, "channel": "physics"}
        rows.append({"step": i, "skill": "pick_and_place",
                     "args": {"object": "pink cube", "destination": "drop zone"},
                     "duration_ms": 2000, "result": res, "keyframe_before": kb, "keyframe_after": ka})
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def _judge_cli(run: Path, cfg: dict, tmp_path: Path) -> tuple[subprocess.CompletedProcess, dict | None]:
    """scripts/judge_run.py exactly as the launcher pass and the B44-live
    script call it (complete JSON config)."""
    path = tmp_path / "judge-config.json"
    path.write_text(json.dumps(cfg))
    p = subprocess.run([sys.executable, str(JUDGE_RUN), str(run), "--config", str(path),
                        "--skills", "pick_and_place"], env=_env(), capture_output=True, text=True, timeout=180)
    verdict = json.loads((run / "judge.json").read_text()) if (run / "judge.json").exists() else None
    return p, verdict


def _in_process(run: Path, cfg: dict):
    from cascade.eval.progress_judge import judge_run, make_judge

    return judge_run(run, make_judge(cfg), skills={"pick_and_place"})


def _normalized_body(body: dict) -> dict:
    """A request with every image replaced by a placeholder (JPEG bytes differ
    across OpenCV builds; the text, order and parameters must not)."""
    body = json.loads(json.dumps(body))
    for msg in body["messages"]:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image_url":
                    part["image_url"]["url"] = "<image>"
    return body


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _run_pass(proof: Path, cfg_path: Path, **env):
    return subprocess.run([sys.executable, str(JUDGE_PROOF), "--proof", str(proof), "--judge", "vlm"],
                          env=_env(CASCADE_JUDGE_CONFIG=cfg_path, **env), capture_output=True, text=True,
                          timeout=240)


# --- premises: what B44-live measured, and what main does with it -----------------
def test_premise_b44_live_abstentions_are_budget_cuts_and_one_untagged_score():
    """Pins the cause in the committed evidence: of the 12 unscored steps at
    1536 tokens, 11 were cut at exactly the budget (finish_reason=length) and
    1 stopped with its score written without the tags. None was a confirmed
    pick: the abstentions are almost all physics failures."""
    unscored = _b44_unscored()
    assert len(_b44_steps()) == 72 and len(unscored) == 12
    cut = [s for s in unscored if s["finish_1536"] == "length"]
    assert len(cut) == 11 and {s["tokens_1536"] for s in cut} == {1536}
    (stopped,) = [s for s in unscored if s["finish_1536"] == "stop"]
    assert stopped["run"] == "b36_run_ab6pm3" and stopped["raw_tail_1536"].rstrip().endswith("Score: -50%.")
    physics = [s["physics"] for s in unscored]
    assert physics.count("refuted") == 11 and physics.count("confirmed") == 0
    # given 8192 tokens, 11 of the 12 wrote a tagged score
    assert sum(s["hop_8192"] is not None for s in unscored) == 11


def test_premise_default_judge_abstains_on_a_cut_answer_after_one_call(tmp_path, stub):
    """Main's behaviour, kept by default: the recorded cut answer is ONE call
    and an unscored step (hop None, `no <score>`), never a 0."""
    rec = _b44("b36_run_ab2main4")
    s = stub([_first_answer(rec)])
    v = _in_process(_recorded_run(tmp_path, ["refuted"]), _qwen_config(s.base_url))
    assert len(s.posts) == 1
    (step,) = v.steps
    assert step.hop is None and step.error.startswith("no <score> in judge output")
    assert step.response_metadata["finish_reason"] == "length"
    assert v.confusion()["n_scored"] == 0 and v.confusion()["tn"] == 0


#: sha256 of the normalized first request and of the normalized step record of
#: main 86373d7 for `_qwen_config` + one refuted pick (recorded with main's
#: export): the default path must stay byte-identical.
GOLDEN_REQUEST_SHA = "d24a948f94434cc4e727293731f8bb459379003d3a6aa050c80d21436f8a4ec6"
GOLDEN_RECORD_SHA = "71a7fd9571a99bec059c58eefe13c26c5900f7ecfd9d44a0b64d3d198548f3ec"
GOLDEN_SUMMARY = ("judge=vlm:Qwen/Qwen3.8-27B mode=incremental scored=0/1 agreement=n/a tp=0 tn=0 fp=0 fn=0 "
                  "final_progress=n/a wrist=n/a")


def _golden_record(verdict: dict) -> dict:
    (step,) = verdict["steps"]
    rm = dict(step["response_metadata"])
    rm["usage"] = {k: rm["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {"top": sorted(verdict), "step": {**step, "response_metadata": rm}}


@pytest.mark.parametrize("switch", [{}, {"score_followup_tokens": 0}], ids=["absent", "zero"])
def test_golden_default_request_and_record_are_byte_identical(tmp_path, stub, switch):
    """Switch absent or 0: one request per step, the same body and the same
    judge.json record as main (sha256 of both, images normalized), and no
    trace of the new mode in the record or the summary line."""
    rec = _b44("b36_run_ab2main4")
    s = stub([_first_answer(rec)], strict=True)
    p, verdict = _judge_cli(_recorded_run(tmp_path, ["refuted"]), _qwen_config(s.base_url, **switch), tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(s.posts) == 1 and not s.rejected
    assert _sha(_normalized_body(s.posts[0])) == GOLDEN_REQUEST_SHA
    assert _sha(_golden_record(verdict)) == GOLDEN_RECORD_SHA
    assert "score_via" not in verdict and "score_via" not in verdict["steps"][0]["response_metadata"]
    assert (tmp_path / "run" / "summary.txt").read_text() == GOLDEN_SUMMARY + "\n"


# --- the follow-up --------------------------------------------------------------------
def test_a_cut_answer_gets_one_bounded_follow_up_and_the_record_says_so(tmp_path, stub):
    """The recorded b36_run_ab2main4 answer (cut at 1536 tokens, no score)
    followed by the score B44 pass 2 got for it: judged through the real
    judge_run.py, the step scores -100 % via the follow-up, the record keeps
    the first answer and names the follow-up's budget, answer and tokens."""
    from cascade.eval import progress_judge as pj

    rec = _b44("b36_run_ab2main4")
    assert rec["finish_1536"] == "length" and rec["hop_8192"] == -1.0
    s = stub([_first_answer(rec), {"content": "<score>-100%</score>", "finish_reason": "stop",
                                   "completion_tokens": 8}], strict=True)
    p, verdict = _judge_cli(_recorded_run(tmp_path, ["refuted"]),
                            _qwen_config(s.base_url, score_followup_tokens=64), tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(s.posts) == 2 and not s.rejected, "the follow-up is a plain chat-completions call"
    first, follow = s.posts
    # the same turn (8 images, GRM prompt), the first answer, then only the score request
    assert follow["messages"][0] == first["messages"][0]
    assert sum(p_["type"] == "image_url" for p_ in follow["messages"][0]["content"]) == 8
    assert follow["messages"][1] == {"role": "assistant", "content": rec["raw_tail_1536"]}
    assert follow["messages"][2] == {"role": "user", "content": pj.SCORE_FOLLOWUP_PROMPT}
    assert len(follow["messages"]) == 3
    # the prompt asks for the GRM prompt's own output line, verbatim
    fmt = pj.GRM_PROMPT[pj.GRM_PROMPT.index("Return ONLY one line"):]
    assert pj.SCORE_FOLLOWUP_PROMPT.endswith(fmt)
    assert follow["max_tokens"] == 64 and first["max_tokens"] == 1536
    for key in ("model", "temperature", "top_p", "chat_template_kwargs"):
        assert follow[key] == first[key], key
    assert set(follow) == set(first) <= STANDARD_KEYS

    (step,) = verdict["steps"]
    assert step["hop"] == -1.0 and step["error"] is None
    assert step["raw"] == rec["raw_tail_1536"], "the record keeps the first answer"
    rm = step["response_metadata"]
    assert rm["finish_reason"] == "length" and rm["usage"]["completion_tokens"] == 1536
    assert rm["score_via"] == "follow-up" and rm["extra_completion_tokens"] == 8
    fu = rm["followup"]
    assert fu["max_tokens"] == 64 and fu["raw"] == "<score>-100%</score>" and fu["finish_reason"] == "stop"
    assert fu["usage"]["completion_tokens"] == 8 and fu["error"] is None and fu["elapsed_s"] >= 0
    assert rm.get("abstention") is None
    # the refuted pick is a true negative again instead of a silent abstention
    assert verdict["confusion"]["tn"] == 1 and verdict["confusion"]["n_scored"] == 1
    assert verdict["score_via"] == {"follow-up": 1}
    summary = (tmp_path / "run" / "summary.txt").read_text()
    assert summary.endswith(" score_via=follow-up:1\n") and "scored=1/1" in summary


def test_an_answer_with_a_score_makes_no_follow_up(tmp_path, stub):
    rec = _b44("b36_run_ab2hold1")   # confirmed, scored +100 % in 239 tokens
    assert rec["hop_1536"] == 1.0 and rec["raw_tail_1536"].endswith("<score>+100%</score>")
    s = stub([_first_answer(rec)])
    v = _in_process(_recorded_run(tmp_path, ["confirmed"]), _qwen_config(s.base_url, score_followup_tokens=64))
    assert len(s.posts) == 1
    (step,) = v.steps
    assert step.hop == 1.0 and step.response_metadata["score_via"] == "first"
    assert step.response_metadata["followup"] is None and step.response_metadata["extra_completion_tokens"] == 0
    assert v.to_dict()["score_via"] == {"first": 1} and v.summary_line().endswith(" score_via=first:1")


def test_the_untagged_stop_answer_is_followed_up_too(tmp_path, stub):
    """b36_run_ab6pm3 stopped at 1241 tokens with "Score: -50%." and no tags:
    a format failure, not a cut. The follow-up asks for the tagged line; the
    plain-text number is never parsed as a score by cascade itself."""
    rec = _b44("b36_run_ab6pm3")
    s = stub([_first_answer(rec), {"content": "<score>-50%</score>", "finish_reason": "stop",
                                   "completion_tokens": 7}])
    v = _in_process(_recorded_run(tmp_path, ["refuted"]), _qwen_config(s.base_url, score_followup_tokens=32))
    assert len(s.posts) == 2 and s.posts[1]["max_tokens"] == 32
    (step,) = v.steps
    assert step.hop == -0.5 and step.response_metadata["score_via"] == "follow-up"
    assert step.response_metadata["finish_reason"] == "stop"


def test_a_follow_up_without_a_score_stays_unscored_with_its_reason(tmp_path, stub):
    """Never invent a score: b36_run_ab4pm2's recorded 8192-token answer ends
    "Score: -100%." with no tags. Replayed as the follow-up, the step stays
    unscored, and the record says why (both calls' finish reasons and
    tokens), with the follow-up's own answer kept."""
    rec = _b44("b36_run_ab4pm2")
    assert rec["hop_8192"] is None and rec["raw_tail_8192"].rstrip().endswith("Score: -100%.")
    s = stub([_first_answer(rec), {"content": rec["raw_tail_8192"], "finish_reason": "stop",
                                   "completion_tokens": 61}])
    v = _in_process(_recorded_run(tmp_path, ["refuted"]), _qwen_config(s.base_url, score_followup_tokens=64))
    assert len(s.posts) == 2
    (step,) = v.steps
    assert step.hop is None, "a follow-up without <score> must not become a score"
    assert step.error.startswith("no <score> in judge output") and "; follow-up: no <score>" in step.error
    rm = step.response_metadata
    assert rm["score_via"] == "none" and rm["followup"]["raw"] == rec["raw_tail_8192"]
    assert rm["extra_completion_tokens"] == 61
    assert rm["abstention"] == ("no <score> in the first answer (finish_reason=length, 1536 completion "
                                "tokens) nor in the follow-up (finish_reason=stop, 61 of 64 tokens)")
    assert v.confusion()["n_scored"] == 0
    assert v.to_dict()["score_via"] == {"none": 1} and v.summary_line().endswith(" score_via=none:1")


def test_a_failed_follow_up_call_stays_unscored_and_names_the_failure(tmp_path, stub):
    """The follow-up request itself is refused (HTTP 400): the step stays
    unscored, the first answer's metadata is kept, and the reason names the
    follow-up's failure (not retried: one extra call at most)."""
    rec = _b44("b36_run_main3")
    s = stub([_first_answer(rec), {"status": 400, "message": "context window exceeded"}])
    v = _in_process(_recorded_run(tmp_path, ["refuted"]), _qwen_config(s.base_url, score_followup_tokens=64))
    assert len(s.posts) == 2
    (step,) = v.steps
    assert step.hop is None and "follow-up failed: BadRequestError" in step.error
    rm = step.response_metadata
    assert rm["finish_reason"] == "length" and rm["usage"]["completion_tokens"] == 1536
    assert rm["score_via"] == "none" and rm["followup"]["error"].startswith("BadRequestError")
    assert rm["followup"]["raw"] is None and rm["extra_completion_tokens"] is None
    assert rm["followup"]["elapsed_s"] >= 0 and rm["followup"]["max_tokens"] == 64
    assert rm["abstention"].startswith("no <score> in the first answer (finish_reason=length, 1536 completion "
                                       "tokens); the follow-up failed: BadRequestError")
    assert step.raw == rec["raw_tail_1536"]


def test_an_endpoint_that_rejects_grammar_still_serves_the_follow_up(tmp_path, stub):
    """The stub refuses llama.cpp's `grammar` / `response_format` the way an
    OpenAI endpoint does (HTTP 400). The follow-up mode needs neither, so it
    works there unchanged -- nothing to degrade."""
    import openai

    s = stub([], strict=True)
    client = openai.OpenAI(base_url=s.base_url, api_key="x", max_retries=0)
    for extra in ({"grammar": 'root ::= "<score>" [0-9]+ "%</score>"'},
                  {"response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {}}}}):
        with pytest.raises(openai.BadRequestError):
            client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                           max_tokens=8, extra_body=extra)
    assert len(s.rejected) == 2 and not s.posts
    rec = _b44("nemoclaw_run_isaac")
    s.script = [_first_answer(rec), {"content": "<score>-100%</score>", "finish_reason": "stop",
                                     "completion_tokens": 8}]
    v = _in_process(_recorded_run(tmp_path, ["refuted"]), _qwen_config(s.base_url, score_followup_tokens=64))
    assert len(s.posts) == 2 and len(s.rejected) == 2
    assert v.steps[0].hop == -1.0 and v.steps[0].response_metadata["score_via"] == "follow-up"


def test_replaying_all_twelve_b44_abstentions(tmp_path, stub):
    """Every B44-live step left unscored at 1536 tokens, replayed: its
    recorded first answer, then -- as the follow-up -- the score pass 2 got
    for it (or pass 2's untagged answer when it got none). 11 score via the
    follow-up, b36_run_ab4pm2 stays unscored, 24 calls in all; a step that
    scored at once (b36_run_ab2hold1) makes one call. (A replay of
    recorded answers, not a measurement of what Qwen does with the follow-up
    prompt: that is the parent's live run.)"""
    recs = [_b44("b36_run_ab2hold1")] + _b44_unscored()   # plus one that scored at once
    script = [_first_answer(recs[0])]
    for r in recs[1:]:
        script.append(_first_answer(r))
        script.append({"content": _score_line(r["hop_8192"]), "finish_reason": "stop", "completion_tokens": 8}
                      if r["hop_8192"] is not None else
                      {"content": r["raw_tail_8192"], "finish_reason": "stop", "completion_tokens": 61})
    s = stub(script)
    v = _in_process(_recorded_run(tmp_path, [r["physics"] if r["physics"] != "unverified" else None
                                             for r in recs]),
                    _qwen_config(s.base_url, score_followup_tokens=64))
    assert len(s.posts) == 1 + 24 and not s.script
    assert [st.hop for st in v.steps] == [1.0] + [r["hop_8192"] for r in recs[1:]]
    vias = [st.response_metadata["score_via"] for st in v.steps]
    assert vias[0] == "first" and vias.count("follow-up") == 11 and vias.count("none") == 1
    assert v.steps[[r["run"] for r in recs].index("b36_run_ab4pm2")].response_metadata["score_via"] == "none"
    assert v.to_dict()["score_via"] == {"first": 1, "follow-up": 11, "none": 1}
    assert v.summary_line().endswith(" score_via=first:1,follow-up:11,none:1")
    assert v.confusion()["n_scored"] == 12


@pytest.mark.parametrize("value", [-1, 513, "64", True, 1.5, [64]])
def test_an_invalid_follow_up_budget_fails_closed(tmp_path, value):
    """A budget nobody can interpret is refused (judge_run.py exit 2, which
    the launcher pass reads as `unavailable`), never silently ignored."""
    pytest.importorskip("openai")
    from cascade.eval.progress_judge import JudgeError, make_judge

    cfg = _qwen_config("http://127.0.0.1:9/v1", score_followup_tokens=value)
    with pytest.raises(JudgeError, match="score_followup_tokens"):
        make_judge(cfg)
    p, verdict = _judge_cli(_recorded_run(tmp_path, ["refuted"]), cfg, tmp_path)
    assert p.returncode == 2 and "score_followup_tokens" in p.stderr and verdict is None


def test_the_follow_up_budget_bounds():
    pytest.importorskip("openai")
    from cascade.eval import progress_judge as pj

    assert pj.MAX_SCORE_FOLLOWUP_TOKENS == 512
    for value, expected in ((None, 0), (0, 0), (1, 1), (512, 512)):
        cfg = _qwen_config("http://127.0.0.1:9/v1", score_followup_tokens=value)
        assert pj.make_judge(cfg).score_followup_tokens == expected
    assert pj.make_judge(_qwen_config("http://127.0.0.1:9/v1")).score_followup_tokens == 0


# --- through the launcher pass (scripts/judge_proof.py, B44) ---------------------------
def test_the_launcher_pass_recovers_the_refuted_pick_as_tn(tmp_path, stub):
    """The B44 bias, end to end: a refuted proof pick whose first answer is
    cut gets its score from the follow-up, so the run summary counts it as
    `tn` instead of losing it, and the banner names how it was scored."""
    rec = _b44("b36_run_ab4pm3")
    s = stub([_first_answer(rec), {"content": "<score>-100%</score>", "finish_reason": "stop",
                                   "completion_tokens": 8}])
    h = _proof_state(tmp_path, [{"skill": "pick_and_place", "physics": "refuted"}])
    cfg = _judge_config(tmp_path, _qwen_config(s.base_url, score_followup_tokens=64))
    p = _run_pass(h["proof"], cfg)
    assert p.returncode == 0, p.stderr
    j = json.loads((h["evidence"] / "run-summary.json").read_text())["judge"]
    assert j["status"] == "ok" and j["confusion"]["tn"] == 1 and j["unscored_physics"] == 0
    assert j["score_via"] == {"follow-up": 1}
    assert "score_via=follow-up:1" in p.stdout and p.stdout.count("\n") == 1
    assert len(s.posts) == 2


def test_the_follow_up_runs_inside_the_launcher_bound(tmp_path, stub):
    """A follow-up that never answers is cut by the existing B44 bound: the
    pass reads `unavailable (timed out ...)`, the judge's process group is
    killed and its connection to the endpoint dies with it. The bound (25 s)
    leaves judge_run.py's start-up and the replayed first answer a wide margin
    on a slow runner, so it is the follow-up that the bound cuts."""
    rec = _b44("b36_run_e1rigid1")
    s = stub([_first_answer(rec), {"hang": True}])
    h = _proof_state(tmp_path, [{"skill": "pick_and_place", "physics": "refuted"}])
    cfg = _judge_config(tmp_path, _qwen_config(s.base_url, score_followup_tokens=64, timeout_s=600))
    p = _run_pass(h["proof"], cfg, CASCADE_JUDGE_TIMEOUT_S=25)
    assert p.returncode == 0, p.stderr
    j = json.loads((h["evidence"] / "run-summary.json").read_text())["judge"]
    assert j["status"] == "unavailable" and j["reason"].startswith("timed out after 25 s")
    assert 25 <= j["elapsed_s"] < 300
    assert len(s.posts) == 2, "the bound must have cut the follow-up, not the first call"
    with pytest.raises(ProcessLookupError):
        os.kill(j["judge_pid"], 0)
    assert s.client_gone.wait(10), "the killed judge still holds its follow-up connection"


def test_a_gateway_follow_up_keeps_the_model_pin_with_its_own_session(tmp_path, stub):
    """Through an agent gateway (`fresh_session`, `backend_model`), the
    follow-up carries its whole history in the request, so it gets a session
    of its own (no hidden agent history doubles it) and the same model pin."""
    rec = _b44("b36_run_ab4pm1")
    s = stub([_first_answer(rec), {"content": "<score>-100%</score>", "finish_reason": "stop",
                                   "completion_tokens": 8}])
    cfg = _qwen_config(s.base_url, score_followup_tokens=64, fresh_session=True, backend_model="local/qwen")
    v = _in_process(_recorded_run(tmp_path, ["refuted"]), cfg)
    assert v.steps[0].hop == -1.0 and len(s.headers) == 2
    keys = [h["x-openclaw-session-key"] for h in s.headers]
    assert all(k.startswith("cascade-judge-") for k in keys) and keys[0] != keys[1]
    assert [h["x-openclaw-model"] for h in s.headers] == ["local/qwen", "local/qwen"]


def test_the_default_launcher_pass_writes_nothing_about_the_follow_up(tmp_path, stub):
    """Mode off (the shipped default): the run summary and the banner carry no
    `score_via`, and the endpoint saw one call, without a session header."""
    rec = _b44("b36_run_ab2hold1")
    s = stub([_first_answer(rec)])
    h = _proof_state(tmp_path)
    p = _run_pass(h["proof"], _judge_config(tmp_path, _qwen_config(s.base_url)))
    assert p.returncode == 0, p.stderr
    j = json.loads((h["evidence"] / "run-summary.json").read_text())["judge"]
    assert j["status"] == "ok" and j["confusion"]["tp"] == 1
    assert "score_via" not in j and "score_via" not in p.stdout
    assert len(s.posts) == 1 and "x-openclaw-session-key" not in s.headers[0]
