"""Learned operating envelopes for the fixed primitive library.

Harness VLA (arXiv:2607.08448) makes one claim we can implement directly:
*don't grow the skill library -- learn the OPERATING RANGE of the primitives
you already have*, from execution traces, global success rules and failure
models.  The repo already does exactly this for grasps
(``memory.grasp_memory.GraspOutcomeMemory``); this module generalises it to
**every** skill.

Why it matters on this rig: the B601-RS has a brutal top-down IK envelope
(x ~ 0.16-0.18, TCP z in [0.06, 0.12]).  Today that knowledge is frozen in
hand-tuned YAML constants and in a human's head.  Here it becomes *data*: a
per-skill, per-feature record of where calls succeeded and where they failed,
learned from the same traces ASPIRE feeds to its debugger.

2026-08-27: confidence is now graduated, not a single MIN_SUPPORT cliff, after
checking the actual ``RLinf/RPent`` repo (the code behind the Harness-VLA
paper, not just its abstract) -- its memory layer tags entries
``single-shot -> probable -> verified`` by evidence breadth, and separately
tracks when a "proven" entry gets falsified by a later contrary observation
(``contradicted_by``). This module had no analogue of either: a span was
either trusted or unknown, and nothing recorded a failure landing INSIDE a
range this same skill had "proven" safe. See ``_Span.confidence()`` and the
``contradictions`` counter below.

Three products, all cheap to compute:

1. ``check(skill, args, **extra)`` -- a PRE-FLIGHT verdict.  A motion costs
   2-20 s; a lookup costs microseconds.  When a requested pose sits outside
   the region where this primitive has ever worked, say so *before* moving.
   Advisory by default ("booth rule": never let learned priors hard-block a
   live demo -- the harness is the only authority that refuses motion).
2. ``failure_digest()`` -- the Harness-VLA "failure model": normalised
   failure signatures ranked by frequency, so the agent sees *how* this
   primitive usually breaks rather than one opaque error string.
3. ``agent_digest()`` -- both of the above, formatted for the LLM context.

Scene-independent by construction: only dimensionless or base-frame-local
scalars are stored, never a specific object or scene layout.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Minimum successful samples before an envelope is allowed to warn.  Below
#: this we know nothing and must stay silent (cold start must be a no-op).
MIN_SUPPORT = 4

#: Fraction of the observed success span added as slack on each side, so the
#: envelope generalises slightly beyond the exact samples seen.
SLACK_FRAC = 0.15

#: Numeric args that carry no spatial meaning and would only add noise.
_IGNORED_KEYS = frozenset(
    {"cycles", "max_objects", "success", "cover", "timeout_ms", "num_grasps", "topk"}
)

#: Failure taxonomy.  Ordered: first match wins.  Keep the patterns tied to
#: what the harness / skills actually emit (see safety/harness.py, skills/
#: runtime.py) -- a signature nobody produces is dead weight.
_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"would hit the table|below the table|link/joint \d+", "geometry:link_below_table"),
    (r"outside .*workspace|workspace aabb|out of workspace", "geometry:outside_workspace"),
    (r"keep-?out|keepout", "geometry:keep_out"),
    (r"no ik|ik failed|unreachable|no solution|could not solve", "kinematics:ik_unreachable"),
    (r"jaw|too wide|max_width|width .*exceed", "geometry:object_too_wide"),
    (r"air.?grasp|missed the object|closed on nothing|nothing in the", "contact:air_grasp"),
    (r"slip|dropped|lost the object", "contact:slip"),
    (r"never seen|not seen|no object|could not find|not found|unknown object", "perception:not_found"),
    (r"stale|watchdog|too old", "safety:stale_perception"),
    (r"e-?stop|emergency|stopped by operator", "safety:estop"),
    (r"velocity|too fast|jerk", "safety:velocity_cap"),
    (r"timeout|timed out", "runtime:timeout"),
    (r"bad arguments|unknown skill", "agent:bad_call"),
)


def normalize_failure(error: str) -> str:
    """Map a raw error string onto a stable failure signature.

    Unknown errors degrade to their first few words rather than being
    dropped -- an un-taxonomised failure is still a clusterable one.
    """
    text = (error or "").strip().lower()
    if not text:
        return "unknown"
    for pattern, sig in _FAILURE_PATTERNS:
        if re.search(pattern, text):
            return sig
    stripped = re.sub(r"^(skillerror|safetyviolation|unexpected \w+)\s*:\s*", "", text)
    words = re.findall(r"[a-z_]+", stripped)[:3]
    return "other:" + "_".join(words) if words else "unknown"


def _numeric_features(args: dict, extra: dict | None = None) -> dict[str, float]:
    """Flatten a skill call into scalar features worth learning over."""
    feats: dict[str, float] = {}
    for key, value in (args or {}).items():
        if key in _IGNORED_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            feats[key] = float(value)
    for key, value in (extra or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            feats[key] = float(value)
    return feats


@dataclass
class _Span:
    """Running min/max/mean over the successes of one (skill, feature)."""

    lo: float
    hi: float
    n: int = 1
    mean: float = 0.0
    #: RPent-style regression signal: a later call whose feature value fell
    #: INSIDE this "proven" range still failed. The range stays (advisory,
    #: never rewritten from a single loss), but confidence downgrades.
    contradictions: int = 0

    def add(self, v: float) -> None:
        self.lo = min(self.lo, v)
        self.hi = max(self.hi, v)
        self.n += 1
        self.mean += (v - self.mean) / self.n

    def slack(self) -> tuple[float, float]:
        pad = max((self.hi - self.lo) * SLACK_FRAC, 1e-4)
        return self.lo - pad, self.hi + pad

    def confidence(self, min_support: int) -> str:
        """RPent's evidence-breadth tiers, ported to this module's only
        breadth signal (sample count -- there is no per-task grouping here,
        spans are deliberately scene-independent, see module docstring)."""
        if self.n < min_support:
            return "single-shot"
        if self.n < 2 * min_support:
            return "probable"
        return "verified"

    def to_json(self) -> dict:
        return {
            "lo": self.lo, "hi": self.hi, "n": self.n, "mean": self.mean,
            "contradictions": self.contradictions,
        }

    @classmethod
    def from_json(cls, d: dict) -> "_Span":
        return cls(lo=float(d["lo"]), hi=float(d["hi"]),
                   n=int(d.get("n", 1)), mean=float(d.get("mean", d["lo"])),
                   contradictions=int(d.get("contradictions", 0)))


@dataclass
class _SkillStats:
    wins: int = 0
    losses: int = 0
    spans: dict[str, _Span] = field(default_factory=dict)
    #: feature values seen ONLY on failures, per failure signature
    failures: dict[str, int] = field(default_factory=dict)
    last_error: str = ""
    duration_ms_mean: float = 0.0

    @property
    def attempts(self) -> int:
        return self.wins + self.losses

    @property
    def success_rate(self) -> float:
        return self.wins / self.attempts if self.attempts else 0.0

    def to_json(self) -> dict:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "spans": {k: v.to_json() for k, v in self.spans.items()},
            "failures": dict(self.failures),
            "last_error": self.last_error[:200],
            "duration_ms_mean": round(self.duration_ms_mean, 1),
        }

    @classmethod
    def from_json(cls, d: dict) -> "_SkillStats":
        return cls(
            wins=int(d.get("wins", 0)),
            losses=int(d.get("losses", 0)),
            spans={k: _Span.from_json(v) for k, v in (d.get("spans") or {}).items()},
            failures=dict(d.get("failures") or {}),
            last_error=str(d.get("last_error", "")),
            duration_ms_mean=float(d.get("duration_ms_mean", 0.0)),
        )


@dataclass
class Verdict:
    """Result of a pre-flight envelope check."""

    ok: bool
    reason: str = ""
    #: features that fell outside the learned success span
    outliers: list[dict] = field(default_factory=list)
    #: non-blocking cautions: e.g. the requested value sits inside a
    #: "proven" range that has since been contradicted by a real failure.
    #: Never flips ok to False -- booth rule: advisory only.
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "reason": self.reason,
            "outliers": self.outliers, "notes": self.notes,
        }


class OperatingEnvelope:
    """Per-skill success envelopes + failure models learned from outcomes.

    Thread-safe: the skill thread records while MCP/dashboard threads read.
    """

    def __init__(self, path: str | Path | None = None, min_support: int = MIN_SUPPORT):
        self.path = Path(path).expanduser() if path else None
        self.min_support = int(min_support)
        self._skills: dict[str, _SkillStats] = {}
        self._lock = threading.RLock()
        self._load()

    # ── recording ────────────────────────────────────────────────────────

    def record(
        self,
        skill: str,
        args: dict,
        ok: bool,
        error: str = "",
        duration_ms: float = 0.0,
        **extra: Any,
    ) -> None:
        """Fold one primitive outcome into the model."""
        feats = _numeric_features(args, extra)
        with self._lock:
            st = self._skills.setdefault(skill, _SkillStats())
            if ok:
                st.wins += 1
                for key, val in feats.items():
                    span = st.spans.get(key)
                    if span is None:
                        st.spans[key] = _Span(lo=val, hi=val, n=1, mean=val)
                    else:
                        span.add(val)
            else:
                st.losses += 1
                sig = normalize_failure(error)
                st.failures[sig] = st.failures.get(sig, 0) + 1
                st.last_error = error or ""
                # RPent contradiction signal: this failing call's feature
                # value landed INSIDE a range this skill had "proven" safe.
                # The range itself is untouched (one loss shouldn't erase
                # many wins), but its confidence must reflect that it just
                # failed here.
                for key, val in feats.items():
                    span = st.spans.get(key)
                    if span is not None and span.lo <= val <= span.hi:
                        span.contradictions += 1
            if duration_ms > 0:
                n = max(st.attempts, 1)
                st.duration_ms_mean += (duration_ms - st.duration_ms_mean) / n
        self._save()

    # ── pre-flight ───────────────────────────────────────────────────────

    def check(self, skill: str, args: dict, **extra: Any) -> Verdict:
        """Would this call land where this primitive has ever worked?

        Advisory only.  Returns ``ok=True`` whenever we lack the evidence to
        say otherwise, so a cold start never interferes.
        """
        with self._lock:
            st = self._skills.get(skill)
            if st is None or st.wins < self.min_support:
                return Verdict(ok=True)
            spans = {k: (v, v.slack()) for k, v in st.spans.items() if v.n >= self.min_support}
        if not spans:
            return Verdict(ok=True)

        feats = _numeric_features(args, extra)
        outliers = []
        notes = []
        for key, val in feats.items():
            entry = spans.get(key)
            if entry is None:
                continue
            span, (lo, hi) = entry
            if val < lo or val > hi:
                outliers.append(
                    {
                        "feature": key,
                        "value": round(val, 4),
                        "known_good": [round(span.lo, 4), round(span.hi, 4)],
                        "n": span.n,
                    }
                )
            elif span.contradictions:
                notes.append(
                    f"{key}={round(val, 4)} is inside the proven range "
                    f"[{round(span.lo, 4)}, {round(span.hi, 4)}] but that range "
                    f"has {span.contradictions} recorded failure(s) since -- "
                    "treat cautiously"
                )
        if not outliers:
            return Verdict(ok=True, notes=notes)
        parts = [
            f"{o['feature']}={o['value']} outside proven range "
            f"[{o['known_good'][0]}, {o['known_good'][1]}] (n={o['n']})"
            for o in outliers
        ]
        return Verdict(
            ok=False,
            reason=f"{skill}: " + "; ".join(parts),
            outliers=outliers,
            notes=notes,
        )

    # ── reporting ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {k: v.to_json() for k, v in self._skills.items()}

    def failure_digest(self, top: int = 5) -> str:
        """Harness-VLA style failure model, most frequent first."""
        with self._lock:
            rows = []
            for skill, st in self._skills.items():
                if not st.failures:
                    continue
                worst = sorted(st.failures.items(), key=lambda kv: -kv[1])[:2]
                rows.append(
                    (
                        st.losses,
                        f"{skill}: {st.wins}W/{st.losses}L; "
                        + ", ".join(f"{sig} x{n}" for sig, n in worst),
                    )
                )
        if not rows:
            return ""
        rows.sort(key=lambda r: -r[0])
        return "\n".join(f"- {line}" for _, line in rows[:top])

    def _span_label(self, v: "_Span") -> str:
        label = f"{v.lo:.3f}, {v.hi:.3f}] ({v.confidence(self.min_support)}"
        if v.contradictions:
            label += f", {v.contradictions} contradiction{'s' if v.contradictions != 1 else ''}"
        return label + ")"

    def envelope_digest(self, top: int = 4) -> str:
        """Where each primitive is known to work (base-frame scalars)."""
        with self._lock:
            lines = []
            for skill, st in sorted(self._skills.items()):
                if st.wins < self.min_support:
                    continue
                good = [
                    f"{k} in [{self._span_label(v)}"
                    for k, v in sorted(st.spans.items())
                    if v.n >= self.min_support
                ][:top]
                if good:
                    lines.append(f"- {skill} ({st.wins}W): " + "; ".join(good))
        return "\n".join(lines)

    def agent_digest(self) -> str:
        """Compact block for the LLM context.  Empty on a cold start."""
        env = self.envelope_digest()
        fail = self.failure_digest()
        if not env and not fail:
            return ""
        out = ["Learned primitive envelopes (from past runs on THIS robot):"]
        if env:
            out += ["Proven operating ranges:", env]
        if fail:
            out += ["How these primitives usually fail:", fail]
        out.append(
            "Treat these as priors, not laws: prefer poses inside a proven "
            "range, and if you must go outside one, expect the listed failure."
        )
        return "\n".join(out)

    # ── ingestion from ASPIRE traces ─────────────────────────────────────

    def ingest_trace(self, trace_path: str | Path) -> int:
        """Replay a ``trace.jsonl`` into the model.  Returns records folded.

        This is what lets the OUTER loop (Hermes, offline) learn from runs
        that happened while nothing was watching.
        """
        path = Path(trace_path)
        if not path.exists():
            return 0
        n = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            skill = rec.get("skill")
            if not skill or skill == "task_done":
                continue
            result = rec.get("result") or {}
            self.record(
                skill,
                rec.get("args") or {},
                ok=bool(result.get("ok")),
                error=str(result.get("error", "")),
                duration_ms=float(rec.get("duration_ms") or 0.0),
            )
            n += 1
        return n

    def ingest_runs(self, runs_dir: str | Path, limit: int = 200) -> dict:
        """Fold every ``*/trace.jsonl`` under ``runs_dir``."""
        root = Path(runs_dir).expanduser()
        traces = sorted(root.glob("*/trace.jsonl"))[-limit:]
        total = sum(self.ingest_trace(p) for p in traces)
        return {"traces": len(traces), "records": total}

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        with self._lock:
            self._skills = {
                k: _SkillStats.from_json(v)
                for k, v in (raw.get("skills") or {}).items()
            }

    def _save(self) -> None:
        if not self.path:
            return
        with self._lock:
            payload = {
                "version": 1,
                "updated": time.time(),
                "skills": {k: v.to_json() for k, v in self._skills.items()},
            }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.path)
        except OSError:
            pass

    def export_markdown(self) -> str:
        """RPent-style MEMORY.md section."""
        lines = ["# Primitive operating envelopes", ""]
        with self._lock:
            for skill, st in sorted(self._skills.items()):
                lines.append(
                    f"## {skill} — {st.wins}W/{st.losses}L "
                    f"({st.success_rate*100:.0f}% success, ~{st.duration_ms_mean/1000:.1f}s)"
                )
                for key, span in sorted(st.spans.items()):
                    conf = span.confidence(self.min_support)
                    contra = (
                        f", {span.contradictions} contradiction"
                        f"{'s' if span.contradictions != 1 else ''}"
                        if span.contradictions else ""
                    )
                    lines.append(
                        f"- `{key}` succeeded in [{span.lo:.4f}, {span.hi:.4f}] "
                        f"(mean {span.mean:.4f}, n={span.n}, {conf}{contra})"
                    )
                for sig, n in sorted(st.failures.items(), key=lambda kv: -kv[1]):
                    lines.append(f"- FAILURE `{sig}` x{n}")
                lines.append("")
        return "\n".join(lines)


def merge_digests(*digests: Iterable[str]) -> str:
    """Join non-empty context blocks with blank lines."""
    parts = [d for d in digests if d and str(d).strip()]
    return "\n\n".join(str(p).strip() for p in parts)
