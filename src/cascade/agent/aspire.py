"""Scoped retry evidence for the ASPIRE trace-to-library path.

``diagnose(run_dir)``
    Match a failed call to a later measured-confirmed retry within a bounded
    trace window. Require the same goal, resolved arm and pre-call possession;
    explicit reset/task-end rows stop matching. Unknown context is rejected.

``distil(diagnosis, library)`` / ``harvest(runs_dir, library)``
    Recheck eligibility and persist the error, context, argument delta and
    verifier receipt as guidance. A matching retry is an observed association,
    not proof of a causal repair or a transferable control policy. Runs with
    the same (skill, signature) fold into ONE entry whose front matter counts
    ``occurrences`` (distinct runs) and ``source_tasks`` (distinct tasks).

``retrieve(library, task)``
    Load keyword-matched PROMOTED notes into the built-in orchestrator at task
    start. Upstream ASPIRE's cross-task rule (``skills/library.py``): an entry
    is promoted only once it recurs in >= ``PROMOTION_MIN_TASKS`` (2) distinct
    tasks; one confirmed retry in one task stays a stored candidate and is not
    retrieved. ``SkillLibrary(root, min_tasks=1)`` is the explicit relaxation.

Harvesting runs offline through ``scripts/learn_from_runs.py``. This module
does not execute retries, change a controller or independently re-evaluate
historical measurements. See ``docs/DREAM_RSI_ADAPTATION.md`` for the contract.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..memory.envelope import normalize_failure
from .effects import CONFIRMED, POSTCONDITIONS

#: A repair is only credited when the same primitive later succeeded within
#: this many steps -- otherwise the "fix" is just an unrelated later action.
REPAIR_WINDOW = 6

# Preserve recorded task identifiers; changing the goal is not repairing it.
_GOAL_KEYS = ("object", "label", "query", "destination", "arm", "spatial_hint",
              "camera", "x", "y", "z", "direction", "distance_m")


def _same_goal(before: dict, after: dict) -> bool:
    return (
        isinstance(before, dict) and isinstance(after, dict)
        and {k: v for k, v in before.items() if k in _GOAL_KEYS}
        == {k: v for k, v in after.items() if k in _GOAL_KEYS}
    )


def _confirmed_postcondition(skill: str, pc: object) -> bool:
    if not isinstance(pc, dict) or not isinstance(skill, str) or skill not in POSTCONDITIONS:
        return False
    return (
        pc.get("status") == CONFIRMED and pc.get("skill") == skill
        and pc.get("kind") == POSTCONDITIONS[skill]
        and pc.get("channel") in ("physics", "belief", "gripper", "visual_diff")
        and isinstance(pc.get("evidence"), str) and bool(pc["evidence"].strip())
        and isinstance(pc.get("measured"), dict) and bool(pc["measured"])
    )


def _same_context(skill: str, before: object, after: object) -> bool:
    """Require explicit routing and pre-call possession, never infer legacy state."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    arm = before.get("arm")
    if not isinstance(arm, str) or not arm.strip() or arm != after.get("arm"):
        return False
    if "held_object" not in before or "held_object" not in after:
        return False
    subject = before["held_object"]
    if subject != after["held_object"]:
        return False
    if skill in {"place_at", "place_on_object", "throw", "handover"}:
        return isinstance(subject, str) and bool(subject.strip())
    return subject is None or isinstance(subject, str)


@dataclass
class Diagnosis:
    """Recorded failure and subsequent confirmed retry, not causal attribution."""

    run: str
    task: str = ""
    succeeded: bool = False
    failed_skill: str = ""
    signature: str = ""
    error: str = ""
    failed_args: dict = field(default_factory=dict)
    failed_context: dict = field(default_factory=dict)
    repair_skill: str = ""
    repair_args: dict = field(default_factory=dict)
    repair_context: dict = field(default_factory=dict)
    repair_postcondition: dict = field(default_factory=dict)
    repaired: bool = False
    n_calls: int = 0
    n_failures: int = 0
    keyframe: str = ""

    @property
    def teachable(self) -> bool:
        """Only scoped, measured-confirmed retry associations become notes."""
        return (self.repaired and bool(self.failed_skill) and bool(self.signature)
                and self.repair_skill == self.failed_skill
                and _same_goal(self.failed_args, self.repair_args)
                and _same_context(self.failed_skill, self.failed_context, self.repair_context)
                and _confirmed_postcondition(self.repair_skill, self.repair_postcondition))

    def as_dict(self) -> dict:
        return {
            "run": self.run,
            "task": self.task,
            "succeeded": self.succeeded,
            "failed_skill": self.failed_skill,
            "signature": self.signature,
            "error": self.error[:300],
            "repair_skill": self.repair_skill,
            "failed_context": copy.deepcopy(self.failed_context),
            "repair_context": copy.deepcopy(self.repair_context),
            "repair_postcondition": copy.deepcopy(self.repair_postcondition),
            "repaired": self.repaired,
            "n_calls": self.n_calls,
            "n_failures": self.n_failures,
        }

    def summary(self) -> str:
        if not self.failed_skill:
            return f"{self.run}: {self.n_calls} calls, no failures"
        base = f"{self.run}: {self.failed_skill} failed ({self.signature})"
        return base + (f" -> later matching {self.repair_skill} confirmed"
                       if self.teachable else " -> no matching confirmed retry")


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

    # Select the LAST failure with a matching confirmed retry, or the first
    # failure when no eligible retry was recorded.
    chosen = None
    for idx, rec in failures:
        skill = rec.get("skill", "")
        for later in records[idx + 1: idx + 1 + REPAIR_WINDOW]:
            result = later.get("result") or {}
            if (later.get("skill") in {"reset_scene", "task_done"}
                    or result.get("task_complete") is True):
                break
            postcondition = result.get("postcondition")
            if (later.get("skill") == skill and result.get("ok") is True
                    and result.get("verified", True) is True
                    and _same_goal(rec.get("args") or {}, later.get("args") or {})
                    and _same_context(skill, rec.get("context"), later.get("context"))
                    and _confirmed_postcondition(skill, postcondition)):
                chosen = (rec, later)
                break
    rec, fix = chosen if chosen is not None else (failures[0][1], None)
    diag.failed_skill = rec.get("skill", "")
    diag.failed_args = rec.get("args") or {}
    diag.failed_context = copy.deepcopy(rec.get("context") or {})
    diag.error = str((rec.get("result") or {}).get("error", ""))
    diag.signature = normalize_failure(diag.error)
    diag.keyframe = rec.get("keyframe_after") or ""
    if fix is None:
        return diag
    diag.repair_skill = fix.get("skill", "")
    diag.repair_args = fix.get("args") or {}
    diag.repair_context = copy.deepcopy(fix.get("context") or {})
    diag.repair_postcondition = copy.deepcopy(fix["result"]["postcondition"])
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
    return "; ".join(parts) if parts else "same arguments (no parameter change recorded)"


def distil(diag: Diagnosis, library) -> Path | None:
    """Write a scoped retry association into the library; None if not teachable."""
    if not diag.teachable:
        return None
    delta = _arg_delta(diag.failed_args, diag.repair_args)
    pc = diag.repair_postcondition
    obj = diag.failed_args.get("label") or diag.failed_args.get("query") or ""
    guard_terms = sorted({w for w in re.findall(r"[a-z]{3,}", f"{diag.failed_skill} {obj} {diag.task}".lower())})

    strategy = [
        f"Recorded retry of `{diag.failed_skill}` after **{diag.signature}**: {delta}.",
        "",
        "This is an observed association, not a proven causal repair or a policy for "
        "other robots. Check the current task and arm profile; retain all safety checks.",
        "",
        f"Recorded pre-call context: `{json.dumps(diag.repair_context, sort_keys=True)}`",
        f"Recorded confirmation: `{json.dumps(pc, sort_keys=True)}`",
        f"Observed error: `{diag.error[:200]}`",
        f"Confirmed retry: `{diag.repair_skill}({json.dumps(diag.repair_args, default=str)[:200]})`",
    ]

    title = f"{diag.failed_skill} {diag.signature.replace(':', ' ')}"
    return library.add(
        title=title,
        failure=f"{diag.failed_skill} -> {diag.signature}: {diag.error[:160]}",
        when=" ".join(guard_terms[:14]),
        strategy="\n".join(strategy),
        origin=f"{diag.run} (task: {diag.task or 'n/a'})",
        # Cross-task evidence: the library counts distinct runs and tasks per
        # (skill, signature); an unknown task adds an occurrence, not a task.
        source_task=diag.task,
        source_run=diag.run,
    )


def _occurrence_index(library) -> dict[str, int]:
    try:
        return {e.name: e.occurrences for e in library.records()}
    except AttributeError:
        return {}


def harvest(runs_dir: str | Path, library, limit: int = 100) -> dict:
    """Diagnose selected runs and distil eligible retry associations offline.

    Every teachable run folds into the entry for its (skill, signature); the
    library counts it once per run directory, so harvesting the same runs
    again changes nothing. ``learned``/``entries`` report entries that gained
    evidence this harvest, ``promoted``/``candidates`` their status after it.
    """
    root = Path(runs_dir).expanduser()
    if not root.exists():
        return {"runs": 0, "diagnosed": 0, "learned": 0, "entries": [],
                "promoted": [], "candidates": []}
    run_dirs = sorted([p for p in root.iterdir() if (p / "trace.jsonl").exists()])[-limit:]
    before = _occurrence_index(library)
    diagnoses = []
    for run in run_dirs:
        diag = diagnose(run)
        if diag is None:
            continue
        diagnoses.append(diag)
        if diag.teachable:
            distil(diag, library)
    after = _occurrence_index(library)
    # One entry per (skill, signature): the library is guidance, not a log.
    learned = sorted(name for name, n in after.items() if n > before.get(name, 0))
    try:
        status = {e.name: e.promoted for e in library.records()}
        summary = library.summary()
    except AttributeError:
        status, summary = {}, {}
    return {
        "runs": len(run_dirs),
        "diagnosed": len(diagnoses),
        "learned": len(learned),
        "entries": learned,
        "promoted": [n for n in learned if status.get(n)],
        "candidates": [n for n in learned if not status.get(n)],
        "library": summary,
        "failure_histogram": dict(
            Counter(d.signature for d in diagnoses if d.signature).most_common()
        ),
        "summaries": [d.summary() for d in diagnoses[-10:]],
    }


def retrieve(library, task: str, max_entries: int = 2, max_chars: int = 1400) -> str:
    """Guard-matched PROMOTED library entries, trimmed for the agent context.

    Candidates (one task, any number of runs; legacy notes without counters)
    stay on disk and out of the prompt. A library without the promotion API
    yields nothing rather than bypassing the gate.
    """
    try:
        entries = library.relevant_entries(task, max_entries=max_entries, promoted_only=True)
    except Exception:
        return ""
    if not entries:
        return ""
    out = ["Learned skills from earlier runs (ASPIRE library) -- apply if they match:"]
    budget = max_chars
    for entry in entries:
        if entry.n_tasks >= 2:
            tag = (f"[promoted: recurred in {entry.n_tasks} distinct tasks, "
                   f"{entry.occurrences} confirmed runs]")
        else:  # only reachable through an explicit min_tasks=1 library
            tag = (f"[admitted by explicit min_tasks=1: {entry.n_tasks} known task(s), "
                   f"{entry.occurrences} run(s); not cross-task validated]")
        block = tag + "\n" + entry.text.strip()
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
