"""Outcome judging for cascade runs with a Robo-Dopamine-style progress
reward model (GRM): https://robo-dopamine.github.io/

What it is. Robo-Dopamine's General Reward Model is a VLM prompted with the
task text, optional REFERENCE START/END images and BEFORE/AFTER image sets,
answering ONE line `<score>+NN%</score>`: the relative progress ("hop") the
AFTER set made toward the goal, in [-100 %, +100 %]. Three prompting modes
-- incremental (previous step -> current), forward (start -> current),
backward (goal -> current) -- are fused into a trajectory-level progress
curve. The same model is what RoboChallenge uses as its "PRM-as-a-Judge" for
scoring VLA policies. That is exactly how cascade uses it: an EXTERNAL judge
of what the agentic policy did, never a reward that trains anything (the
Dopamine-RL half needs a gradient-trainable policy; cascade's Cosmos3 +
skills tiers have none).

What it judges here. The runtime already saves a BEFORE and an AFTER
keyframe per skill call (`skills/runtime.py`, `trace.jsonl` rows carry
`keyframe_before` / `keyframe_after`). The judge scores each pair with the
GRM prompt VERBATIM (examples/inference.py in the upstream repo; single-view
= the front image repeated for the two wrist slots, blank goal when no
reference end image exists -- both are documented upstream usages of
GRM-2.0), then folds the per-skill hops into a run progress curve with the
upstream fusion arithmetic.

Why it is worth having next to the physics channel. Cascade's postcondition
verifier returns `confirmed` / `refuted` from an independent ground-truth
channel (MuJoCo / Isaac free-body poses, `channel: physics`). A judge that
looks only at pixels can be CALIBRATED against that: `calibrate()` builds
the judge-vs-physics confusion matrix per run, so the operator learns how
much to trust the judge on THIS rig (top-down rendered MuJoCo camera is a
new distribution for a model trained on multi-view real + LIBERO/RoboCasa
footage) before its number is quoted anywhere. And per-tier hops (reflex /
experience / LLM) come from the `tier` the runtime already records, giving
a progress-per-second metric for the agentic dispatch itself.

Backends (config `eval.judge`), one prompt, one parser, one interface:

  grm    the released GRM checkpoint (tanhuajie2001/Robo-Dopamine-GRM-2.0-*)
         served by vLLM's OpenAI-compatible server on a CUDA box; cascade
         talks to `base_url`. Upstream sampling: temperature 0.1, top_p 0.9.
  vlm    the same prompt against ANY OpenAI-compatible vision model (a
         local Cosmos3 server, GPT, the OpenClaw brain) -- what upstream's
         eval_api.py does to compare GRM against API models, and the only
         thing executable on a GPU-less laptop.
  fake   deterministic stand-in for tests (fixed or scripted scores).

All offline: `scripts/judge_run.py runs/<run>` reads the artifacts a demo
left behind. Nothing here runs in the control loop.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------
# The upstream prompt, verbatim (FlagOpen/Robo-Dopamine examples/inference.py
# SYSTEM_PROMPT). `<image>` markers are where the 8 images interleave, in
# this order: ref start, ref end, before {front, left wrist, right wrist},
# after {front, left wrist, right wrist}.
# --------------------------------------------------------------------------
GRM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”
   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.
2) Direction:
   - AFTER better than BEFORE → positive score.
   - AFTER worse than BEFORE → negative score.
   - Essentially the same → 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""

N_IMAGES = 8  # the prompt has exactly eight <image> slots
_SCORE_RE = re.compile(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</score>", re.IGNORECASE)


class JudgeError(RuntimeError):
    pass


def _is_local_url(url: str) -> bool:
    """Loopback / link-local / RFC1918 / .local hosts: proxy-exempt."""
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def parse_score(raw: str) -> float:
    """`<score>+35%</score>` -> 0.35. Upstream takes the LAST <score> block
    and clips to [-1, 1]; a missing/garbled score is a JudgeError (upstream
    silently scores 0.0 -- here the caller decides, because a judge that
    quietly says "no progress" on a parse failure biases every metric)."""
    ms = _SCORE_RE.findall(raw or "")
    if not ms:
        raise JudgeError(f"no <score> in judge output: {raw[:120]!r}")
    return float(np.clip(float(ms[-1]) / 100.0, -1.0, 1.0))


def blank_image_jpeg(size: tuple[int, int] = (64, 64)) -> bytes:
    """Upstream ships examples/blank_goal.png for the 'no goal image' case;
    a neutral grey JPEG plays the same role."""
    import cv2

    img = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise JudgeError("could not encode blank image")
    return bytes(buf)


def build_images(before: bytes, after: bytes, *, ref_start: bytes | None = None,
                 ref_end: bytes | None = None,
                 before_wrists: tuple[bytes, bytes] | None = None,
                 after_wrists: tuple[bytes, bytes] | None = None) -> list[bytes]:
    """The 8-image list in prompt order. Single-view rigs repeat the front
    image into the wrist slots (upstream's documented single-view usage);
    missing references fall back to `before` (start) and a blank (end)."""
    blank = blank_image_jpeg()
    bw = before_wrists or (before, before)
    aw = after_wrists or (after, after)
    return [ref_start if ref_start is not None else before,
            ref_end if ref_end is not None else blank,
            before, bw[0], bw[1], after, aw[0], aw[1]]


def interleave(task: str, images: list[bytes]) -> list[dict]:
    """OpenAI-style multimodal content: text segments between the images,
    exactly like upstream's `SYSTEM_PROMPT.split("<image>")` interleave."""
    if len(images) != N_IMAGES:
        raise JudgeError(f"GRM prompt needs {N_IMAGES} images, got {len(images)}")
    parts = GRM_PROMPT.format(task=task).split("<image>")
    assert len(parts) == N_IMAGES + 1
    content: list[dict] = []
    for i, text in enumerate(parts):
        if text:
            content.append({"type": "text", "text": text})
        if i < N_IMAGES:
            b64 = base64.b64encode(images[i]).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------
class ProgressJudge:
    """score() -> hop in [-1, 1]: how much closer AFTER is to done than BEFORE."""

    name = "judge"

    def score(self, task: str, before: bytes, after: bytes, **refs) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class OpenAICompatJudge(ProgressJudge):
    """GRM served by vLLM (`--served-model-name` any) or any OpenAI-compatible
    vision model. One class, two `kind`s, because the wire is identical --
    the difference is WHICH weights answer, and that is what `describe()`
    reports so a run judged by GPT is never mistaken for one judged by GRM."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 kind: str = "vlm", temperature: float = 0.1, top_p: float = 0.9,
                 max_tokens: int = 64, timeout_s: float = 120.0,
                 extra_headers: dict | None = None, fresh_session: bool = False):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - extra not installed
            raise JudgeError("the judge needs the `llm` extra (openai client)") from e
        import os

        if api_key is None and base_url is not None and "OPENAI_API_KEY" not in os.environ:
            api_key = "not-needed"  # local vLLM / llama.cpp servers ignore it
        # A LOCAL endpoint must never be routed through the machine's HTTP
        # proxy. Measured on this Mac: the macOS system proxy (VPN, 127.0.0.1:
        # 1082) is inherited by httpx via the environment and answered every
        # multi-image POST to the loopback gateway with an empty 503 -- while
        # curl, which ignores system proxies, got 200 from the same URL.
        http_client = None
        if base_url and _is_local_url(base_url):
            # the SDK's own httpx class (openai>=1 `DefaultHttpxClient`; 3.x
            # vendors a fork as `httpx2`, so never import httpx directly)
            from openai import DefaultHttpxClient

            http_client = DefaultHttpxClient(trust_env=False, timeout=timeout_s)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, http_client=http_client)
        self.proxy_bypassed = http_client is not None
        self.model = model
        self.kind = kind
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.name = f"{kind}:{model}"
        self.last_raw: str | None = None
        # Gateways that front an AGENT (OpenClaw's /v1/chat/completions) pick
        # the backend weights from a header and keep per-session history;
        # `extra_headers` carries the model pin (x-openclaw-model) and
        # `fresh_session` gives every score() its own session key so one
        # verdict never sees another pair. `describe()` then names the
        # backend model, not the gateway alias.
        self.extra_headers = dict(extra_headers or {})
        self.fresh_session = fresh_session
        backend = self.extra_headers.get("x-openclaw-model")
        if backend:
            self.name = f"{kind}:{backend} via {model}"

    def score(self, task: str, before: bytes, after: bytes, **refs) -> float:
        content = interleave(task, build_images(before, after, **refs))
        headers = dict(self.extra_headers)
        if self.fresh_session:
            import uuid

            headers["x-openclaw-session-key"] = f"cascade-judge-{uuid.uuid4().hex[:12]}"
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
            extra_headers=headers or None,
        )
        self.last_raw = (resp.choices[0].message.content or "") if resp.choices else ""
        return parse_score(self.last_raw)


class FakeJudge(ProgressJudge):
    """Tests: fixed score, or a script of raw model outputs consumed in order
    (so the parser and the fusion arithmetic run for real)."""

    name = "fake"

    def __init__(self, score: float = 0.5, script: Iterable[str] | None = None):
        self._score = score
        self._script = list(script) if script is not None else None
        self.calls: list[dict] = []

    def score(self, task: str, before: bytes, after: bytes, **refs) -> float:
        self.calls.append({"task": task, "n_before": len(before), "n_after": len(after), **{k: v is not None for k, v in refs.items()}})
        if self._script is not None:
            if not self._script:
                raise JudgeError("FakeJudge script exhausted")
            return parse_score(self._script.pop(0))
        return float(self._score)


def make_judge(cfg: dict | None) -> ProgressJudge:
    """`eval.judge` config -> judge. No silent default: an unknown or missing
    backend raises, because a judge you did not choose is a number you cannot
    interpret."""
    cfg = dict(cfg or {})
    kind = str(cfg.get("backend", "")).lower()
    if kind == "fake":
        return FakeJudge(float(cfg.get("score", 0.5)))
    if kind in ("grm", "vlm"):
        model = cfg.get("model")
        if not model:
            raise JudgeError(f"eval.judge.backend={kind} needs `model`")
        headers = dict(cfg.get("extra_headers") or {})
        if cfg.get("backend_model"):
            headers["x-openclaw-model"] = str(cfg["backend_model"])
        api_key = cfg.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("$"):
            import os

            api_key = os.environ.get(api_key[1:])
        return OpenAICompatJudge(
            model=str(model), base_url=cfg.get("base_url"), api_key=api_key, kind=kind,
            temperature=float(cfg.get("temperature", 0.1)), top_p=float(cfg.get("top_p", 0.9)),
            max_tokens=int(cfg.get("max_tokens", 64)), timeout_s=float(cfg.get("timeout_s", 120.0)),
            extra_headers=headers, fresh_session=bool(cfg.get("fresh_session", False)),
        )
    raise JudgeError(f"eval.judge.backend must be grm|vlm|fake, got {kind!r}")


# --------------------------------------------------------------------------
# Trajectory fusion (upstream arithmetic, examples/inference.py)
# --------------------------------------------------------------------------
def fuse_progress(raw_scores: list[float], mode: str = "incremental") -> tuple[list[float], list[float]]:
    """Per-step raw GRM scores -> (progress[], hop[]) using the upstream rule
    for each mode. incremental: progress_k = p + (1-p)*s (s>=0) or p + p*s
    (s<0), hop = s. forward: progress = s, hop = delta. backward: progress =
    1 + s, hop = delta."""
    prog, hops = [], []
    prev = 0.0
    for i, s in enumerate(raw_scores):
        s = float(np.clip(s, -1.0, 1.0))
        if mode == "incremental":
            cur = s if i == 0 else (prev + (1.0 - prev) * s if s >= 0 else prev + prev * s)
            hop = s
        elif mode == "forward":
            cur, hop = s, s - prev
        elif mode == "backward":
            cur = 1.0 + s
            hop = cur - prev
        else:
            raise JudgeError(f"unknown fusion mode {mode!r}")
        prog.append(cur)
        hops.append(hop)
        prev = cur
    return prog, hops


# --------------------------------------------------------------------------
# Judging a recorded run
# --------------------------------------------------------------------------
@dataclass
class StepVerdict:
    step: int
    skill: str
    task: str
    hop: float | None                 # GRM score for this BEFORE->AFTER pair
    physics: str | None               # postcondition status if channel == physics
    channel: str | None
    ok: bool | None                   # the skill's own claim
    tier: str | None
    duration_s: float | None
    error: str | None = None

    def agrees_with_physics(self) -> bool | None:
        """Judge says progress (hop > 0) iff physics confirmed. None when
        either side is missing -- an unknown is not an agreement."""
        if self.hop is None or self.physics not in ("confirmed", "refuted"):
            return None
        return (self.hop > 0) == (self.physics == "confirmed")


@dataclass
class RunVerdict:
    run_dir: str
    judge: str
    steps: list[StepVerdict]
    progress: list[float] = field(default_factory=list)
    hops: list[float] = field(default_factory=list)
    mode: str = "incremental"

    @property
    def final_progress(self) -> float | None:
        return self.progress[-1] if self.progress else None

    def confusion(self) -> dict:
        """Judge (hop>0) vs physics (confirmed) over steps that have both."""
        c = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n_scored": 0, "n_with_physics": 0}
        for s in self.steps:
            if s.hop is not None:
                c["n_scored"] += 1
            a = s.agrees_with_physics()
            if a is None:
                continue
            c["n_with_physics"] += 1
            judge_pos = s.hop > 0
            if a:
                c["tp" if judge_pos else "tn"] += 1
            else:
                c["fp" if judge_pos else "fn"] += 1
        n = c["n_with_physics"]
        c["agreement"] = (c["tp"] + c["tn"]) / n if n else None
        return c

    def per_tier(self) -> dict:
        """Hop per second per dispatch tier -- the agentic-policy metric."""
        out: dict[str, dict] = {}
        for s in self.steps:
            if s.hop is None:
                continue
            t = s.tier or "unknown"
            d = out.setdefault(t, {"n": 0, "hop_sum": 0.0, "seconds": 0.0})
            d["n"] += 1
            d["hop_sum"] += s.hop
            d["seconds"] += float(s.duration_s or 0.0)
        for d in out.values():
            d["hop_mean"] = d["hop_sum"] / d["n"]
            d["hop_per_s"] = d["hop_sum"] / d["seconds"] if d["seconds"] > 0 else None
        return out

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir, "judge": self.judge, "mode": self.mode,
            "final_progress": self.final_progress, "progress": self.progress, "hops": self.hops,
            "confusion": self.confusion(), "per_tier": self.per_tier(),
            "steps": [s.__dict__ for s in self.steps],
        }


def _task_text(row: dict) -> str:
    """The instruction the judge is given. A skill call has no free-text
    task, so it is rendered from the skill name + args: `pick_and_place
    {"object": "red object"}` -> 'pick and place the red object'."""
    skill = str(row.get("skill", ""))
    args = row.get("args") or {}
    obj = args.get("object") or args.get("label") or args.get("target")
    text = skill.replace("_", " ")
    if obj:
        text += f" the {obj}"
    dest = args.get("destination") or args.get("place")
    if dest:
        text += f" at {dest}"
    return text


def judge_run(run_dir: str | Path, judge: ProgressJudge, mode: str = "incremental",
              ref_end: bytes | None = None, skills: set[str] | None = None) -> RunVerdict:
    """Score every traced skill call that has both keyframes. Steps without
    a pair are kept (hop=None) so the count is honest."""
    run_dir = Path(run_dir)
    trace = run_dir / "trace.jsonl"
    if not trace.exists():
        raise JudgeError(f"no trace.jsonl in {run_dir}")
    steps: list[StepVerdict] = []
    raw_scores: list[float] = []
    ref_start: bytes | None = None
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if skills is not None and row.get("skill") not in skills:
            continue
        res = row.get("result") or {}
        pc = res.get("postcondition") or {}
        sv = StepVerdict(
            step=int(row.get("step", len(steps))), skill=str(row.get("skill")), task=_task_text(row),
            hop=None, physics=pc.get("status") if pc.get("channel") == "physics" else None,
            channel=pc.get("channel"), ok=res.get("ok"), tier=res.get("tier") or row.get("tier"),
            duration_s=(row.get("duration_ms") or 0) / 1000.0 or None,
        )
        kb, ka = row.get("keyframe_before"), row.get("keyframe_after")
        if kb and ka and (run_dir / kb).exists() and (run_dir / ka).exists():
            before = (run_dir / kb).read_bytes()
            after = (run_dir / ka).read_bytes()
            if ref_start is None:
                ref_start = before  # first BEFORE of the run anchors "just starting"
            try:
                sv.hop = judge.score(sv.task, before, after, ref_start=ref_start, ref_end=ref_end)
                raw_scores.append(sv.hop)
            except JudgeError as e:
                sv.error = str(e)
            except Exception as e:  # noqa: BLE001 -- transport/server failure on ONE pair
                # A 503 or timeout from the model endpoint is a fact about this
                # step, not about the run: record it and keep judging.
                sv.error = f"{type(e).__name__}: {str(e)[:160]}"
        else:
            sv.error = "missing keyframe pair"
        steps.append(sv)
    progress, hops = fuse_progress(raw_scores, mode)
    return RunVerdict(str(run_dir), judge.describe(), steps, progress, hops, mode)
