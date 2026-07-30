"""ASPIRE's self-improvement loop: diagnose a run, distil a reusable skill.

ASPIRE (NVIDIA GEAR) is a continual-learning system that "inspects rollout
traces, diagnoses failures, repairs programs, validates corrected behaviors,
and saves reusable skills for future tasks", and its ablations credit the
growing skill library with the largest transfer gain -- skills learned in sim
carry into real-robot runs *as in-context guidance*, reaching first success
with fewer tokens.

wrc_demo already ships two thirds of this:

* the rich multimodal trace (``agent/trace.py`` -- trace.jsonl + keyframes),
* the library schema and retrieval (``skills/library.py``).

What was missing is the arrow between them.  ``SkillLibrary``'s own docstring
admits it: *"the orchestrator does not call this yet -- the load-into-context
loop is a ROADMAP item"*.  Nothing ever wrote an entry and nothing ever read
one back, so every run started as ignorant as the first.

This module is that arrow, in two directions:

``diagnose(run_dir)``
    Read a finished run's trace, localize the failure (the ASPIRE "selectively
    inspect salient primitive logs" step), and emit a structured ``Diagnosis``
    -- what broke, which primitive, under what signature, and what the run did
    afterwards that worked.

``distil(diagnosis)``
    Turn a *repaired* failure into a library entry: failure signature,
    when-to-apply guard, and the strategy that actually fixed it.  Only
    validated repairs are written -- a failure with no subsequent success
    teaches nothing except "this breaks", which the envelope model already
    records.

``retrieve(task)``
    Pull the guard-matching entries into the agent's context for the next run.

The evolutionary-search half of ASPIRE (parallel program variants) is out of
scope for a live booth demo, but the loop below is the part that compounds.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..memory.envelope import normalize_failure

#: A repair is only credited when the same primitive later succeeded within
#: this many steps -- otherwise the "fix" is just an unrelated later action.
REPAIR_WINDOW = 6


@dataclass
class Diagnosis:
    """What went wrong in one run, and what (if anything) fixed it."""

    run: str
    task: str = ""
    succeeded: bool = False
    failed_skill: str = ""
    signature: str = ""
    error: str = ""
    failed_args: dict = field(default_factory=dict)
    repair_skill: str = ""
    repair_args: dict = field(default_factory=dict)
    repaired: bool = False
    n_calls: int = 0
    n_failures: int = 0
    keyframe: str = ""

    @property
    def teachable(self) -> bool:
        """Only validated repairs become skills (ASPIRE Sec 2.2)."""
        return self.repaired and bool(self.failed_skill) and bool(self.signature)

    def as_dict(self) -> dict:
        return {
            "run": self.run,
            "task": self.task,
            "succeeded": self.succeeded,
            "failed_skill": self.failed_skill,
            "signature": self.signature,
            "error": self.error[:300],
            "repair_skill": self.repair_skill,
            "repaired": self.repaired,
            "n_calls": self.n_calls,
            "n_failures": self.n_failures,
        }

    def summary(self) -> str:
        if not self.failed_skill:
            return f"{self.run}: {self.n_calls} calls, no failures"
        base = f"{self.run}: {self.failed_skill} failed ({self.signature})"
        return base + (f" -> repaired by {self.repair_skill}" if self.repaired else " -> never repaired")


def _read_trace(run_dir: Path) -> list[dict]:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _read_summary(run_dir: Path) -> tuple[str, bool]:
    path = run_dir / "summary.txt"
    if not path.exists():
        return "", False
    text = path.read_text()
    task = ""
    m = re.search(r"^task:\s*(.+)$", text, re.M)
    if m:
        task = m.group(1).strip()
    ok = bool(re.search(r"^success:\s*true$", text, re.M | re.I))
    return task, ok


def diagnose(run_dir: str | Path) -> Diagnosis | None:
    """Localize the salient failure in one run directory."""
    run_dir = Path(run_dir)
    records = _read_trace(run_dir)
    if not records:
        return None
    task, succeeded = _read_summary(run_dir)
    diag = Diagnosis(run=run_dir.name, task=task, succeeded=succeeded, n_calls=len(records))

    failures = [
        (i, r) for i, r in enumerate(records)
        if not (r.get("result") or {}).get("ok", True)
    ]
    diag.n_failures = len(failures)
    if not failures:
        return diag

    # The salient failure is the LAST one that was subsequently repaired;
    # falling back to the first failure when nothing was ever repaired.
    chosen = None
    for idx, rec in failures:
        skill = rec.get("skill", "")
        for later in records[idx + 1: idx + 1 + REPAIR_WINDOW]:
            if later.get("skill") == skill and (later.get("result") or {}).get("ok"):
                chosen = (idx, rec, later)
                break
    if chosen is None:
        idx, rec = failures[0]
        diag.failed_skill = rec.get("skill", "")
        diag.failed_args = rec.get("args") or {}
        diag.error = str((rec.get("result") or {}).get("error", ""))
        diag.signature = normalize_failure(diag.error)
        diag.keyframe = rec.get("keyframe_after") or ""
        return diag

    idx, rec, fix = chosen
    diag.failed_skill = rec.get("skill", "")
    diag.failed_args = rec.get("args") or {}
    diag.error = str((rec.get("result") or {}).get("error", ""))
    diag.signature = normalize_failure(diag.error)
    diag.keyframe = rec.get("keyframe_after") or ""
    diag.repair_skill = fix.get("skill", "")
    diag.repair_args = fix.get("args") or {}
    diag.repaired = True
    return diag


def _arg_delta(before: dict, after: dict) -> str:
    """Human-readable description of what changed between the two calls."""
    parts = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if b == a:
            continue
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            parts.append(f"`{key}` {b} -> {a} ({a - b:+.4g})")
        else:
            parts.append(f"`{key}` {b!r} -> {a!r}")
    return "; ".join(parts) if parts else "same arguments, retried after re-observing"


#: Signature -> generalisable strategy text.  These encode what this rig has
#: actually taught us (see the repo's CLAUDE.md and the B601-RS IK envelope).
_STRATEGY_HINTS: dict[str, str] = {
    "geometry:link_below_table": (
        "A wrist/elbow link dipped under the table plane. Raise the grasp "
        "target (grasp nearer the object's TOP face, i.e. a smaller "
        "`depth_fraction`) rather than translating in XY -- on the B601-RS the "
        "safe top-down TCP window is roughly z in [0.06, 0.12] at x ~ 0.16-0.18."
    ),
    "kinematics:ik_unreachable": (
        "Top-down IK has a narrow envelope on this arm (x ~ 0.16-0.18; strict "
        "top-down poses fail above z ~ 0.15). Re-home first so IK seeds from "
        "the elbow-up branch, then request a pose inside the envelope, or push "
        "the object closer before grasping."
    ),
    "contact:air_grasp": (
        "The jaw closed on nothing: the grasp was too shallow or mis-centred. "
        "Deepen the grasp slightly and re-localize the object before retrying "
        "-- a stale belief centre is the usual cause."
    ),
    "perception:not_found": (
        "The object was never grounded. Re-observe from a second camera or "
        "localize by an explicit label/spatial hint before any motion; do not "
        "retry the motion against a belief that was never confirmed."
    ),
    "geometry:object_too_wide": (
        "The object exceeds the jaw span at the planned grasp. Re-plan a grasp "
        "across the object's SHORT axis, or push it instead of lifting it."
    ),
    "safety:stale_perception": (
        "Perception went stale mid-motion. Re-observe immediately before the "
        "motion skill so the freshness check at begin_motion() passes."
    ),
    "geometry:outside_workspace": (
        "The requested pose left the workspace AABB. Clamp the target into the "
        "configured workspace and prefer the known-reachable drop zone."
    ),
}


def distil(diag: Diagnosis, library) -> Path | None:
    """Write a validated repair into the ``SkillLibrary``.  None if not teachable."""
    if not diag.teachable:
        return None
    delta = _arg_delta(diag.failed_args, diag.repair_args)
    hint = _STRATEGY_HINTS.get(diag.signature, "")
    obj = diag.failed_args.get("label") or diag.failed_args.get("query") or ""
    guard_terms = sorted({w for w in re.findall(r"[a-z]{3,}", f"{diag.failed_skill} {obj} {diag.task}".lower())})

    strategy = [
        f"When `{diag.failed_skill}` fails with **{diag.signature}**, the repair "
        f"that worked on this rig was: {delta}.",
    ]
    if hint:
        strategy.append("")
        strategy.append(f"Why: {hint}")
    strategy += [
        "",
        f"Observed error: `{diag.error[:200]}`",
        f"Repaired by: `{diag.repair_skill}({json.dumps(diag.repair_args, default=str)[:200]})`",
    ]

    title = f"{diag.failed_skill} {diag.signature.replace(':', ' ')}"
    return library.add(
        title=title,
        failure=f"{diag.failed_skill} -> {diag.signature}: {diag.error[:160]}",
        when=" ".join(guard_terms[:14]),
        strategy="\n".join(strategy),
        origin=f"{diag.run} (task: {diag.task or 'n/a'})",
    )


def harvest(runs_dir: str | Path, library, limit: int = 100) -> dict:
    """Diagnose every run under ``runs_dir`` and distil the repaired ones.

    This is the batch entry point for the OUTER loop -- a Hermes cron job can
    call it after a demo session so the next session starts smarter.
    """
    root = Path(runs_dir).expanduser()
    if not root.exists():
        return {"runs": 0, "diagnosed": 0, "learned": 0, "entries": []}
    run_dirs = sorted([p for p in root.iterdir() if (p / "trace.jsonl").exists()])[-limit:]
    diagnoses, learned = [], []
    seen_signatures: set[str] = set()
    for run in run_dirs:
        diag = diagnose(run)
        if diag is None:
            continue
        diagnoses.append(diag)
        if not diag.teachable:
            continue
        # One entry per (skill, signature): the library is guidance, not a log.
        key = f"{diag.failed_skill}|{diag.signature}"
        if key in seen_signatures:
            continue
        seen_signatures.add(key)
        path = distil(diag, library)
        if path is not None:
            learned.append(path.name)
    return {
        "runs": len(run_dirs),
        "diagnosed": len(diagnoses),
        "learned": len(learned),
        "entries": learned,
        "failure_histogram": dict(
            Counter(d.signature for d in diagnoses if d.signature).most_common()
        ),
        "summaries": [d.summary() for d in diagnoses[-10:]],
    }


def retrieve(library, task: str, max_entries: int = 2, max_chars: int = 1400) -> str:
    """Guard-matched library entries, trimmed for the agent context."""
    try:
        entries = library.relevant(task, max_entries=max_entries)
    except Exception:
        return ""
    if not entries:
        return ""
    out = ["Learned skills from earlier runs (ASPIRE library) -- apply if they match:"]
    budget = max_chars
    for text in entries:
        block = text.strip()
        if len(block) > budget:
            block = block[:budget].rsplit("\n", 1)[0] + "\n..."
        out.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "\n\n".join(out)


def report(runs_dir: str | Path, limit: int = 50) -> str:
    """Human-readable diagnosis board (for the nightly runner / Hermes)."""
    root = Path(runs_dir).expanduser()
    if not root.exists():
        return "no runs directory"
    lines = [f"# Run diagnostics ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    run_dirs = sorted([p for p in root.iterdir() if (p / "trace.jsonl").exists()])[-limit:]
    sigs: Counter = Counter()
    for run in run_dirs:
        diag = diagnose(run)
        if diag is None:
            continue
        lines.append(f"- {diag.summary()}")
        if diag.signature:
            sigs[diag.signature] += 1
    if sigs:
        lines += ["", "## Failure histogram", ""]
        lines += [f"- {sig}: {n}" for sig, n in sigs.most_common()]
    return "\n".join(lines)
