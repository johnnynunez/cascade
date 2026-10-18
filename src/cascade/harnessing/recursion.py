# SPDX-License-Identifier: Apache-2.0
"""Task-level recursion: Propose -> Probe -> Compose, then blind selection, gate and reflection.

Harness²'s two modes, with its budget identity (``R = 4k`` rollouts, ``P = 3k`` improver
sessions; selection and reflection are counted outside the ledger):

- **parallel**: ``k`` base rollouts; ``k`` INV and ``k`` DOM candidates proposed from them and
  rolled out once each; ``k`` independent compositions from the whole pool, rolled out once each.
- **sequential**: ``k`` base rollouts; then ``k`` rounds of (INV + DOM proposal from the current
  harness and all evidence so far, probe both, compose, roll out the composition).

An invalid proposal gets one retry with its problems listed, then inherits the harness it started
from (``fell_back``). Selection is blind: base and final executions, shuffled under random ids.

CASCADE adds an adoption **gate**. The selector never sees a measured outcome; the gate does: with
``gate="verified"`` (default) a selected final is adopted only if CASCADE's own verifier confirmed
that execution, otherwise the base harness is kept. ``gate="none"`` is Harness²'s behaviour.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import re
import shutil
import time
from typing import Callable

from . import prompts as P
from .executor import Executor, Rollout, TaskSpec
from .improver import Improver, ImproverError
from .library import Library
from .surface import LOCKED_TOOLS, AgentHarness

STAGE_ATTEMPTS = 2
_HARNESS = re.compile(r"<harness>\s*(.*?)\s*</harness>", re.S)
_NOTES = re.compile(r"<notes>(.*?)</notes>", re.S)
_CHOICE = re.compile(r"```choice\s*\n?\s*([A-Za-z0-9_-]+)\s*\n?\s*```", re.S)
_WHY = re.compile(r"<why>(.*?)</why>", re.S)
_EDITS = re.compile(r"<edits>(.*?)</edits>", re.S)
_EXPERIENCE = re.compile(r"<experience>(.*?)</experience>", re.S)


class BudgetError(AssertionError):
    """The ledger did not land on the budget identity."""


@dataclass
class Ledger:
    mode: str
    k: int
    rollouts: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)

    def assert_identity(self) -> None:
        if len(self.rollouts) != 4 * self.k or len(self.sessions) != 3 * self.k:
            raise BudgetError(f"{self.mode} k={self.k}: {len(self.rollouts)} rollouts (expected {4 * self.k}), "
                              f"{len(self.sessions)} sessions (expected {3 * self.k})")


@dataclass
class Stage:
    label: str
    dim: str
    harness: AgentHarness
    kept_base: bool = False
    fell_back: bool = False
    attempts: int = 0
    notes: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "harness"}
        d["harness_sha"] = self.harness.sha()
        return d


def parse_harness_answer(text: str) -> tuple[AgentHarness | None, bool, str]:
    """``(harness, keep_base, notes)``; raises ValueError when neither is present or the JSON is bad."""
    notes_m = _NOTES.findall(text)
    notes = notes_m[-1].strip() if notes_m else ""
    blocks = _HARNESS.findall(text)
    if blocks:
        body = blocks[-1].strip()
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body).strip()
        if body == "KEEP_BASE":  # an improver may wrap its keep in the harness tags
            return None, True, notes
        return AgentHarness.from_json(body), False, notes
    if "KEEP_BASE" in text:
        return None, True, notes
    raise ValueError("answer has neither a <harness> block nor KEEP_BASE")


def parse_experience(text: str) -> tuple[str, dict | None]:
    edits = _EDITS.findall(text)
    edits_text = edits[-1].strip() if edits else ""
    exp = _EXPERIENCE.findall(text)
    if not exp:
        return edits_text, None
    fields = {}
    for line in exp[-1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    if not all(fields.get(k) for k in ("cue", "lesson", "evidence")):
        return edits_text, None
    return edits_text, fields


def _task_md(spec: TaskSpec, tool_specs: list[dict]) -> str:
    lines = [f"# Task `{spec.id}` (family `{spec.domain}`)", "", spec.task, "",
             f"The agent has at most {spec.max_steps} planner steps.", "",
             "## Tools this rig offers (name: first sentence of its description)", ""]
    for s in tool_specs:
        if s.get("name") in LOCKED_TOOLS:
            continue
        first = re.split(r"(?<=[.!?])\s", (s.get("description") or "").strip(), maxsplit=1)[0]
        lines.append(f"- `{s.get('name')}`: {first[:200]}")
    lines += ["", "Locked tools (never mention or guide): " + ", ".join(sorted(LOCKED_TOOLS)), ""]
    return "\n".join(lines)


@dataclass
class StepResult:
    task_id: str
    mode: str
    k: int
    base_sha: str
    rollouts: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    finals: list = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    adopted_sha: str = ""
    adopted_from: str = ""
    experience: dict | None = None
    edits: str = ""
    identical_finals: bool = False
    ledger: dict = field(default_factory=dict)
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class Harnessing:
    """One Propose–Probe–Compose step per task, with a persistent per-family library."""

    def __init__(self, executor: Executor, improver: Improver, *, work_dir: Path, tool_specs: list[dict],
                 library: Library | None = None, workers: int = 2, gate: str = "verified", rollout_workers: int = 1,
                 rng: random.Random | None = None, log: Callable[[str], None] | None = None):
        if gate not in ("verified", "none"):
            raise ValueError("gate must be 'verified' or 'none'")
        self.executor = executor
        self.improver = improver
        self.work_dir = Path(work_dir)
        self.tool_specs = tool_specs
        self.tool_names = {s.get("name") for s in tool_specs}
        self.library = library
        self.workers = max(1, int(workers))
        #: concurrent rollouts; only for an executor that declares itself thread-safe
        #: (`SubprocessExecutor`: one process per rollout)
        self.rollout_workers = max(1, int(rollout_workers)) if getattr(executor, "thread_safe", False) else 1
        self.gate = gate
        self.rng = rng or random.Random(0)
        self.log = log or (lambda message: print(message, flush=True))

    # -- plumbing ----------------------------------------------------------------------------
    def _roll(self, ledger: Ledger, harness: AgentHarness, spec: TaskSpec, label: str, root: Path,
              out: dict) -> Rollout:
        self.log(f"[harness2] rollout {label} (harness {harness.sha()[:12]})")
        rollout = self.executor.run(harness, spec, label, root / "rollouts" / label)
        ledger.rollouts.append(label)
        out[label] = rollout
        self.log(f"[harness2]   {label}: ended by {rollout.ended_by}, {rollout.steps} steps, "
                 f"{rollout.duration_s:.0f} s")
        return rollout

    def _roll_many(self, ledger: Ledger, jobs: list[tuple[AgentHarness, str]], spec: TaskSpec, root: Path,
                   out: dict) -> list[Rollout]:
        """Roll every (harness, label) out; concurrently up to `rollout_workers`. Results keep job order."""
        if self.rollout_workers == 1 or len(jobs) == 1:
            return [self._roll(ledger, h, spec, label, root, out) for h, label in jobs]
        with ThreadPoolExecutor(max_workers=min(self.rollout_workers, len(jobs))) as pool:
            return list(pool.map(lambda job: self._roll(ledger, job[0], spec, job[1], root, out), jobs))

    def _context(self, root: Path, label: str, spec: TaskSpec, harness: AgentHarness,
                 evidence: dict[str, Rollout], candidates: dict[str, Stage] | None = None,
                 base: AgentHarness | None = None) -> Path:
        ctx = root / "context" / label
        if ctx.exists():
            shutil.rmtree(ctx)
        ctx.mkdir(parents=True)
        (ctx / "TASK.md").write_text(_task_md(spec, self.tool_specs))
        harness.write(ctx / "harness")
        if base is not None:
            base.write(ctx / "base_harness")
        for name, rollout in evidence.items():
            dst = ctx / "rollouts" / name
            dst.mkdir(parents=True)
            shutil.copy2(rollout.trajectory, dst / "TRAJECTORY.md")
        for name, stage in (candidates or {}).items():
            dst = ctx / "candidates" / name
            stage.harness.write(dst)
            (dst / "NOTES.md").write_text(
                f"# Candidate {name} ({stage.dim})\n\n"
                f"{'KEEP_BASE' if stage.kept_base else ''}{' (fell back to its starting harness)' if stage.fell_back else ''}\n\n"
                f"{stage.notes or '(no notes)'}\n")
        if self.library is not None:
            self.library.context_into(ctx / "library")
        return ctx

    def _session(self, ledger: Ledger, label: str, dim: str, prompt: str, ctx: Path,
                 fallback: AgentHarness) -> Stage:
        stage = Stage(label, dim, fallback)
        problems_note = ""
        for attempt in range(1, STAGE_ATTEMPTS + 1):
            stage.attempts = attempt
            self.log(f"[harness2] session {label} ({dim}) attempt {attempt}")
            try:
                text = self.improver.run(prompt + problems_note, ctx, label=label)
            except ImproverError as exc:
                stage.error = str(exc)
                continue
            (ctx / f"answer-{attempt}.md").write_text(text)
            try:
                harness, keep, notes = parse_harness_answer(text)
            except ValueError as exc:
                stage.error = str(exc)
                problems_note = f"\n\n# YOUR PREVIOUS ANSWER WAS REJECTED\n\n{exc}. Answer again."
                continue
            stage.notes = notes
            if keep:
                stage.kept_base = True
                stage.error = ""
                break
            assert harness is not None
            problems = harness.problems(self.tool_names)
            if problems:
                stage.error = "; ".join(problems[:8])
                problems_note = ("\n\n# YOUR PREVIOUS ANSWER WAS REJECTED\n\nThe harness failed validation: "
                                 + "; ".join(problems[:8]) + ". Fix every problem and answer again.")
                continue
            stage.harness = harness
            stage.error = ""
            break
        else:
            stage.fell_back = True
            stage.harness = fallback
        ledger.sessions.append(label)
        self.log(f"[harness2]   {label}: {'KEEP_BASE' if stage.kept_base else 'fell back' if stage.fell_back else 'harness ' + stage.harness.sha()[:12]}"
                 + (f" ({stage.error[:120]})" if stage.error else ""))
        return stage

    def _sessions(self, ledger: Ledger, jobs: list[tuple]) -> list[Stage]:
        if self.workers == 1 or len(jobs) == 1:
            return [self._session(ledger, *job) for job in jobs]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(jobs))) as pool:
            return list(pool.map(lambda job: self._session(ledger, *job), jobs))

    # -- the step ------------------------------------------------------------------------------
    def improve(self, spec: TaskSpec, base: AgentHarness, *, k: int = 1, mode: str = "parallel") -> StepResult:
        if mode not in ("parallel", "sequential"):
            raise ValueError("mode must be parallel or sequential")
        problems = base.problems(self.tool_names)
        if problems:
            raise ValueError("base harness refused: " + "; ".join(problems))
        t0 = time.monotonic()
        root = self.work_dir / spec.id
        root.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(mode, k)
        rollouts: dict[str, Rollout] = {}
        result = StepResult(spec.id, mode, k, base.sha())
        base_labels = [f"base{i}" for i in range(1, k + 1)]
        self._roll_many(ledger, [(base, label) for label in base_labels], spec, root, rollouts)
        stages: list[Stage] = []
        finals: list[str] = []
        harness_of: dict[str, AgentHarness] = {lbl: base for lbl in base_labels}

        if mode == "parallel":
            evidence = {lbl: rollouts[lbl] for lbl in base_labels}
            jobs = []
            for i in range(1, k + 1):
                for dim in ("INV", "DOM"):
                    label = f"{dim.lower()}{i}"
                    ctx = self._context(root, label, spec, base, evidence)
                    jobs.append((label, dim, P.propose(spec.task, spec.domain, dim, k=k, index=i), ctx, base))
            proposals = self._sessions(ledger, jobs)
            stages += proposals
            for stage in proposals:
                harness_of[stage.label] = stage.harness
            self._roll_many(ledger, [(s.harness, s.label) for s in proposals], spec, root, rollouts)
            pool_evidence = {lbl: rollouts[lbl] for lbl in base_labels + [s.label for s in proposals]}
            cands = {s.label: s for s in proposals}
            jobs = []
            for i in range(1, k + 1):
                label = f"agg{i}"
                ctx = self._context(root, label, spec, base, pool_evidence, cands)
                jobs.append((label, "AGG", P.compose(spec.task, spec.domain, k=k, index=i,
                                                     n_candidates=len(cands)), ctx, base))
            aggs = self._sessions(ledger, jobs)
            stages += aggs
            for stage in aggs:
                harness_of[stage.label] = stage.harness
                finals.append(stage.label)
            self._roll_many(ledger, [(s.harness, s.label) for s in aggs], spec, root, rollouts)
            result.identical_finals = len({harness_of[f].sha() for f in finals}) == 1
        else:
            current = base
            evidence = {lbl: rollouts[lbl] for lbl in base_labels}
            for t in range(1, k + 1):
                jobs = []
                for dim in ("INV", "DOM"):
                    label = f"{dim.lower()}{t}"
                    ctx = self._context(root, label, spec, current, evidence)
                    jobs.append((label, dim, P.propose(spec.task, spec.domain, dim, k=k, index=t), ctx, current))
                proposals = self._sessions(ledger, jobs)
                stages += proposals
                for stage in proposals:
                    harness_of[stage.label] = stage.harness
                rolled = self._roll_many(ledger, [(s.harness, s.label) for s in proposals], spec, root, rollouts)
                for stage, rollout in zip(proposals, rolled, strict=True):
                    evidence[stage.label] = rollout
                label = f"c{t}"
                ctx = self._context(root, label, spec, current, evidence, {s.label: s for s in proposals})
                comp = self._session(ledger, label, "COMPOSE", P.compose_round(spec.task, spec.domain, t=t, k=k),
                                     ctx, current)
                stages.append(comp)
                harness_of[label] = comp.harness
                evidence[label] = self._roll(ledger, comp.harness, spec, label, root, rollouts)
                finals.append(label)
                current = comp.harness

        ledger.assert_identity()
        result.rollouts = {lbl: r.as_dict() for lbl, r in rollouts.items()}
        result.stages = [s.as_dict() for s in stages]
        result.finals = finals
        result.ledger = {"rollouts": ledger.rollouts, "sessions": ledger.sessions,
                         "expected": {"rollouts": 4 * k, "sessions": 3 * k}}

        chosen = self._select(spec, root, base_labels + finals, rollouts)
        result.selection = chosen
        adopted_label = chosen.get("chosen_label") or base_labels[0]
        adopted = harness_of.get(adopted_label, base)
        gate = {"policy": self.gate, "selected": adopted_label,
                "selected_measured": rollouts[adopted_label].measured_success if adopted_label in rollouts else None}
        if self.gate == "verified" and adopted_label not in base_labels and \
                rollouts[adopted_label].measured_success is not True:
            gate["decision"] = "kept base: the selected execution was not verified by CASCADE"
            adopted, adopted_label = base, base_labels[0]
        else:
            gate["decision"] = "adopted the selected execution's harness"
        result.gate = gate
        result.adopted_sha = adopted.sha()
        result.adopted_from = adopted_label
        adopted.write(root / "adopted")

        edits, experience = self._reflect(spec, root, base, stages, rollouts, adopted_label)
        result.edits, result.experience = edits, experience
        if self.library is not None:
            lesson = (f"lesson: {experience['lesson']}" if experience else "NO_ENTRY")
            exp_md = (f"# Experience: task `{spec.id}`\n\n## Edit rows\n\n{edits or '(none)'}\n\n## Lesson\n\n"
                      + ("\n".join(f"{k}: {v}" for k, v in experience.items()) if experience else "NO_ENTRY")
                      + "\n")
            self.library.append(spec.id, adopted, exp_md,
                                {"task": spec.task, "adopted_from": adopted_label, "gate": gate,
                                 "measured": {lbl: r.measured_success for lbl, r in rollouts.items()}},
                                lesson)
        result.duration_s = round(time.monotonic() - t0, 1)
        (root / "result.json").write_text(json.dumps(result.as_dict(), indent=2, default=str) + "\n")
        (root / "REPORT.md").write_text(render_report(spec, result))
        return result

    def _select(self, spec: TaskSpec, root: Path, labels: list[str], rollouts: dict[str, Rollout]) -> dict:
        ids = [f"x{self.rng.randrange(16**4):04x}" for _ in labels]
        while len(set(ids)) != len(ids):
            ids = [f"x{self.rng.randrange(16**4):04x}" for _ in labels]
        order = list(range(len(labels)))
        self.rng.shuffle(order)
        id_map = {ids[i]: labels[i] for i in range(len(labels))}
        ctx = root / "context" / "select"
        if ctx.exists():
            shutil.rmtree(ctx)
        (ctx / "candidates").mkdir(parents=True)
        (ctx / "TASK.md").write_text(_task_md(spec, self.tool_specs))
        for i in order:
            dst = ctx / "candidates" / ids[i]
            dst.mkdir()
            text = Path(rollouts[labels[i]].trajectory).read_text()
            text = re.sub(r"^# Execution .*$", f"# Execution {ids[i]}", text, count=1, flags=re.M)
            text = re.sub(r"^Harness: .*$", "Harness: (hidden)", text, count=1, flags=re.M)
            (dst / "TRAJECTORY.md").write_text(text)
        out = {"id_map": id_map, "n": len(labels), "fallback": False}
        try:
            text = self.improver.run(P.select(spec.task, spec.domain, len(labels)), ctx, label="select")
            (ctx / "answer.md").write_text(text)
            m = _CHOICE.findall(text)
            choice = m[-1] if m else ""
            if choice not in id_map:
                raise ValueError(f"choice {choice!r} is not a candidate id")
            why = _WHY.findall(text)
            out.update(chosen_id=choice, chosen_label=id_map[choice], why=why[-1].strip() if why else "")
        except (ImproverError, ValueError) as exc:
            out.update(fallback=True, error=str(exc), chosen_label=labels[0])
        self.log(f"[harness2] selection: {out.get('chosen_label')}{' (fallback)' if out['fallback'] else ''}")
        return out

    def _reflect(self, spec: TaskSpec, root: Path, base: AgentHarness, stages: list[Stage],
                 rollouts: dict[str, Rollout], adopted_label: str) -> tuple[str, dict | None]:
        evidence = dict(rollouts)
        ctx = self._context(root, "reflect", spec, base, evidence, {s.label: s for s in stages}, base=base)
        (ctx / "CHOSEN.md").write_text(f"The execution finally chosen: {adopted_label}\n")
        try:
            text = self.improver.run(P.experience(spec.task, spec.domain), ctx, label="reflect")
        except ImproverError as exc:
            self.log(f"[harness2] reflection failed: {exc}")
            return "", None
        (ctx / "answer.md").write_text(text)
        return parse_experience(text)


def render_report(spec: TaskSpec, result: StepResult) -> str:
    lines = [f"# Harness² step: `{spec.id}`", "", f"Task: {spec.task}", f"Mode: {result.mode}, k={result.k}; "
             f"{result.duration_s:.0f} s", "", "| execution | harness | measured (hidden check) | CASCADE verifier | "
             "agent claim | steps | ended by |", "|---|---|---|---|---|---|---|"]
    for label, r in result.rollouts.items():
        lines.append(f"| {label} | `{r['harness_sha'][:12]}` | {r['measured_success']} | {r.get('verifier_success')} | "
                     f"{r['claimed_success']} | {r['steps']} | {r['ended_by']} |")
    lines += ["", "## Stages", ""]
    for s in result.stages:
        state = "KEEP_BASE" if s["kept_base"] else ("fell back" if s["fell_back"] else f"`{s['harness_sha'][:12]}`")
        lines.append(f"- {s['label']} ({s['dim']}): {state}" + (f"; error: {s['error'][:160]}" if s["error"] else ""))
    sel = result.selection
    lines += ["", "## Selection and gate", "",
              f"- blind selection: {sel.get('chosen_label')} ({'fallback' if sel.get('fallback') else sel.get('why', '')[:300]})",
              f"- gate ({result.gate.get('policy')}): {result.gate.get('decision')}",
              f"- adopted: `{result.adopted_sha[:12]}` from {result.adopted_from}", "",
              "## Edits and lesson", "", result.edits or "(none)", "",
              ("\n".join(f"{k}: {v}" for k, v in result.experience.items()) if result.experience else "NO_ENTRY"), ""]
    return "\n".join(lines)
