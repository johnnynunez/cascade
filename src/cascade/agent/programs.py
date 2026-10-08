"""The programs tier (Waddle "programs"; ROADMAP follow-up #8).

Design note: ``docs/PROGRAMS_TIER.md``. Waddle's hierarchy is primitives ->
skills -> programs ("full task-specific policies composed from skills, written
fresh per instruction"). Here a PROGRAM is a declarative, bounded list of
REGISTERED tool calls -- never code, never a second control path:

* every step names a ``TOOL_SPECS`` tool and is checked against its schema;
  tools that are not task steps (``task_done``, ``reset_scene``,
  ``halt_motion``, pixel tools, ``recall_step``, the live-view controls) are
  refused, and the MCP stop channel is not a tool, so a program can neither
  stop nor reset stop permission;
* object labels are the only parameters (``{"$param": name}``), and positions
  exist only as Task-Specific Memory queries (``memory/recipes.py``):
  ``{"$target": localize_object(label) + offset_m}``, re-grounded through
  live perception BEFORE the first step -- one query that does not resolve
  aborts the program with zero motion. A literal coordinate is refused;
* the orchestrator executes a program as TOP-LEVEL ``SkillRuntime.execute()``
  calls, so each step keeps its own trace row, keyframes, watcher pause,
  three-state postcondition, task-effects obligation and memory frame. The
  SafetyHarness, reached through each skill's ``SafeArm``, stays the sole
  authority that refuses motion: this module has no motion API.

A program never claims success from its own authorship. Each step's verdict
is read from the task-effects LEDGER rows the step added (the independent
checker's verdict), never from the step's self-reported result. The program
stops at the first step that failed, was refused by the harness, was refuted,
or left a registered effect UNVERIFIED (later steps are never built on an
effect nobody confirmed), with a ``next_action`` for whoever plans next. A
``stuck`` step ends the task (RPent semantics, as in every tier).

Scoping decision (ROADMAP #8's question): programs are BOTH authored by the
agent for one instruction and DISTILLED from verified LLM-tier runs, under one
admission rule -- see ``memory/programs.py``: stored only from a CONFIRMED
execution, offered for reuse only once promoted across distinct tasks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..memory import recipes
from ..memory.programs import ProgramLibrary, ProgramRecord
from ..skills.runtime import TOOL_SPECS, _MOTION_SKILLS
from .effects import CONFIRMED, POSTCONDITIONS, REFUTED, UNVERIFIED

PROGRAM_VERSION = 1
#: A program is a task-level script, not an unbounded plan.
MAX_PROGRAM_STEPS = 12
MAX_PROGRAM_PARAMS = 6
#: A bound parameter is an object label, nothing longer.
MAX_LABEL_CHARS = 80
#: Offsets are compared at this resolution when two programs are folded into
#: one record: perception noise between two verified runs folds, a real change
#: of placement does not.
OFFSET_QUANTUM_M = 0.01
#: A z offset beyond this is not "relative to" a tabletop anchor.
MAX_Z_OFFSET_M = 0.30

#: Registered tools that are not task steps, with the reason a program may not
#: use them. Everything else in TOOL_SPECS composes.
NON_COMPOSABLE: dict[str, str] = {
    "task_done": "a program never declares its own success; independent verification of each step decides",
    "reset_scene": "it undoes a task (props back to spawn, memory cleared); it is not a step of one",
    "halt_motion": "it acts on a motion already in flight; program steps run one at a time",
    "grasp_at_pixel": "a pixel is a scene-bound coordinate; name the object (grasp_object)",
    "probe_point": "a pixel is a scene-bound coordinate",
    "recall_step": "it reads past keyframes back for a planner; a program has no reader between steps",
    "open_live_view": "a dashboard control, not a task step",
    "close_live_view": "a dashboard control, not a task step",
    "live_view_status": "a dashboard control, not a task step",
}

# Step verdicts beyond the three postcondition states (CONFIRMED / REFUTED /
# UNVERIFIED from agent/effects.py).
FAILED = "failed"        # ok=false: the skill could not do it
REFUSED = "refused"      # ok=false and the error names the safety harness
STUCK = "stuck"          # the skill needs a human (`ask`); ends the task
UNCHECKED = "unchecked"  # a motion with no registered postcondition (wave, ...)
NO_EFFECT = "no_effect"  # a read-only step

#: Step verdicts that stop a program.
STOPPING = frozenset({FAILED, REFUSED, REFUTED, UNVERIFIED, STUCK})

# Run statuses.
COMPLETED = "completed"
STOPPED = "stopped"
ABORTED = "aborted"      # never started: binding or grounding failed, zero motion

# Authoring reply kinds.
NONE = "none"
FRESH = "fresh"
REUSE = "reuse"
INVALID = "invalid"

_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_REFUSAL_MARKERS = ("SafetyViolation", "e-stop", "unsafe", "harness")
#: Arguments that carry an object label (lifted to parameters by distillation).
_LABEL_KEYS = ("label", "object", "destination", "query")
_LABEL_ROLES = {
    ("grasp_object", "label"): "object",
    ("pick_and_place", "object"): "object",
    ("pick_and_place", "destination"): "destination",
    ("place_on_object", "label"): "destination",
    ("push_object", "label"): "object",
    ("handover", "label"): "object",
    ("throw", "label"): "object",
    ("point_at", "label"): "target",
    ("turn_screw", "label"): "fastener",
    ("search_for_object", "label"): "object",
    ("$target", "label"): "anchor",
}


class ProgramError(ValueError):
    """A program (or its bindings) is not well formed; nothing was run."""


# ── validation ───────────────────────────────────────────────────────────


def _slug(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")[:48]


def _is_param(value: Any) -> bool:
    return isinstance(value, Mapping) and "$param" in value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_value(tool: str, key: str, schema: Mapping, value: Any, used: set[str]) -> Any:
    where = f"{tool}.{key}"
    if isinstance(value, Mapping):
        if set(value) != {"$param"}:
            raise ProgramError(f"{where}: an object value must be exactly {{'$param': <name>}}")
        if schema.get("type") != "string" or "enum" in schema:
            raise ProgramError(f"{where}: a parameter may only fill a free-form string argument "
                               "(an object label), never a number, flag or fixed choice")
        name = value["$param"]
        if not isinstance(name, str) or not _IDENT.match(name):
            raise ProgramError(f"{where}: invalid parameter name {name!r}")
        used.add(name)
        return {"$param": name}
    typ = schema.get("type")
    if typ == "string":
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ProgramError(f"{where}: expected a non-empty string, got {value!r}")
        if value.strip().startswith("$"):
            raise ProgramError(f"{where}: '$' values are reserved for parameters and targets")
    elif typ == "number":
        if not _finite_number(value):
            raise ProgramError(f"{where}: expected a finite number, got {value!r}")
    elif typ == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProgramError(f"{where}: expected an integer, got {value!r}")
    elif typ == "boolean":
        if not isinstance(value, bool):
            raise ProgramError(f"{where}: expected true/false, got {value!r}")
    else:
        raise ProgramError(f"{where}: unsupported argument type {typ!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ProgramError(f"{where}: {value!r} is not one of {list(schema['enum'])}")
    if typ in ("number", "integer"):
        if "minimum" in schema and value < schema["minimum"]:
            raise ProgramError(f"{where}: {value!r} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ProgramError(f"{where}: {value!r} is above the maximum {schema['maximum']}")
    return value


def _check_target(tool: str, target: Any, coord_keys: tuple, used: set[str]) -> dict:
    where = f"{tool}.$target"
    if not isinstance(target, Mapping):
        raise ProgramError(f"{where}: expected {{'label', 'offset_m'}}")
    unknown = set(target) - {"query", "label", "offset_m", "args"}
    if unknown:
        raise ProgramError(f"{where}: unknown key(s) {sorted(unknown)}")
    query = target.get("query", "localize_object")
    if query != "localize_object":
        raise ProgramError(f"{where}: only localize_object queries can ground a position, got {query!r}")
    label = target.get("label")
    if label is None:
        raise ProgramError(f"{where}: a perception query needs the label of the object it is relative to")
    if _is_param(label):
        label = _check_value(tool, "$target.label", {"type": "string"}, label, used)
    elif not isinstance(label, str) or not label.strip() or label.strip().startswith("$"):
        raise ProgramError(f"{where}: label must be an object label or a parameter, got {label!r}")
    offset = target.get("offset_m")
    if (not isinstance(offset, (list, tuple)) or not 2 <= len(offset) <= 3
            or not all(_finite_number(v) for v in offset)):
        raise ProgramError(f"{where}: offset_m must be 2 or 3 finite numbers (metres), got {offset!r}")
    if math.hypot(float(offset[0]), float(offset[1])) > recipes.ANCHOR_RADIUS_M:
        raise ProgramError(
            f"{where}: offset {list(offset)} m is beyond the {recipes.ANCHOR_RADIUS_M:.2f} m anchor radius: "
            "a position that far from its object is not relative to it")
    if len(offset) == 3 and abs(float(offset[2])) > MAX_Z_OFFSET_M:
        raise ProgramError(f"{where}: z offset {offset[2]} m exceeds {MAX_Z_OFFSET_M} m")
    keys = target.get("args") or list(coord_keys[: len(offset)])
    if (not isinstance(keys, (list, tuple)) or len(keys) != len(offset) or len(set(keys)) != len(keys)
            or any(k not in coord_keys for k in keys)):
        raise ProgramError(f"{where}: args must name {len(offset)} of {list(coord_keys)} in offset order, got {keys!r}")
    return {"query": "localize_object", "label": label,
            "offset_m": [float(v) for v in offset], "args": [str(k) for k in keys]}


def _check_args(tool: str, spec: Mapping, args: Any, used: set[str]) -> dict:
    if not isinstance(args, Mapping):
        raise ProgramError(f"{tool}: args must be an object")
    schema = spec.get("parameters") or {}
    props = schema.get("properties") or {}
    coord_keys = tuple(recipes.COORDINATE_ARGS.get(tool, ()))
    literal = [k for k in coord_keys if k in args]
    if literal:
        raise ProgramError(
            f"{tool}: raw coordinate {literal} refused -- a program refers to positions only as "
            "{'$target': {'label': <object>, 'offset_m': [dx, dy]}}, re-measured before the first step")
    out: dict = {}
    covered: list[str] = []
    for key, value in args.items():
        if key == "$target":
            if not coord_keys:
                raise ProgramError(f"{tool}: $target is only accepted by skills that take base-frame "
                                   f"coordinates ({sorted(recipes.COORDINATE_ARGS)})")
            out[key] = _check_target(tool, value, coord_keys, used)
            covered = out[key]["args"]
            continue
        if key not in props:
            raise ProgramError(f"{tool}: unknown argument {key!r}")
        out[key] = _check_value(tool, key, props[key], value, used)
    missing = [r for r in schema.get("required") or [] if r not in out and r not in covered]
    if missing:
        raise ProgramError(f"{tool}: missing required argument(s) {missing}")
    return out


def validate_program(spec: Any, tool_specs: Iterable[Mapping] | None = None) -> dict:
    """The normalized program, or ``ProgramError`` naming the first problem.

    Checks structure, the registered-tool catalog and each tool's schema, the
    no-raw-coordinate rule, parameter declarations and bounds. It decides
    only whether a program is WELL FORMED -- never whether a motion is safe;
    that stays with the harness at execute() time.
    """
    if not isinstance(spec, Mapping):
        raise ProgramError("a program must be a JSON object")
    unknown = set(spec) - {"version", "name", "description", "params", "steps"}
    if unknown:
        raise ProgramError(f"unknown program key(s) {sorted(unknown)}: a program is a flat list of "
                           "tool calls, with no loops, branches or code")
    version = spec.get("version", PROGRAM_VERSION)
    if version != PROGRAM_VERSION or isinstance(version, bool):
        raise ProgramError(f"unsupported program version {version!r}")
    name = _slug(spec.get("name"))
    if not name:
        raise ProgramError("a program needs a name")
    description = spec.get("description", "")
    if not isinstance(description, str) or len(description) > 300:
        raise ProgramError("description must be a string of at most 300 characters")
    params = spec.get("params", {}) or {}
    if not isinstance(params, Mapping):
        raise ProgramError("params must map a parameter name to what it is")
    if len(params) > MAX_PROGRAM_PARAMS:
        raise ProgramError(f"at most {MAX_PROGRAM_PARAMS} parameters, got {len(params)}")
    for key, about in params.items():
        if not isinstance(key, str) or not _IDENT.match(key):
            raise ProgramError(f"invalid parameter name {key!r}")
        if not isinstance(about, str) or len(about) > 160:
            raise ProgramError(f"parameter {key}: describe it in at most 160 characters")
    steps = spec.get("steps")
    if not isinstance(steps, (list, tuple)) or not 1 <= len(steps) <= MAX_PROGRAM_STEPS:
        raise ProgramError(f"a program needs 1..{MAX_PROGRAM_STEPS} steps, got "
                           f"{len(steps) if isinstance(steps, (list, tuple)) else steps!r}")
    catalog = {s["name"]: s for s in (TOOL_SPECS if tool_specs is None else tool_specs)}
    used: set[str] = set()
    norm_steps = []
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or "tool" not in step or set(step) - {"tool", "args"}:
            raise ProgramError(f"step {i}: expected {{'tool': <name>, 'args': {{...}}}}")
        tool = step["tool"]
        if not isinstance(tool, str):
            raise ProgramError(f"step {i}: tool must be a name")
        if tool in NON_COMPOSABLE:
            raise ProgramError(f"step {i}: {tool} cannot be a program step: {NON_COMPOSABLE[tool]}")
        if tool not in catalog:
            raise ProgramError(f"step {i}: {tool!r} is not a registered tool (a program calls only "
                               "TOOL_SPECS tools; the stop channel is not one)")
        norm_steps.append({"tool": tool, "args": _check_args(tool, catalog[tool], step.get("args") or {}, used)})
    undeclared = sorted(used - set(params))
    if undeclared:
        raise ProgramError(f"undeclared parameter(s) {undeclared}")
    unused = sorted(set(params) - used)
    if unused:
        raise ProgramError(f"declared but unused parameter(s) {unused}")
    return {"version": PROGRAM_VERSION, "name": name, "description": description,
            "params": dict(params), "steps": norm_steps}


def program_signature(spec: Mapping) -> str:
    """sha256 of the program's structure.

    Parameter names become their position of first use, offsets are quantized
    to ``OFFSET_QUANTUM_M`` and the name/description are left out, so an
    authored program and the same structure distilled from a verified run
    fold into ONE library record.
    """
    slots: dict[str, int] = {}

    def canon(value):
        if _is_param(value):
            return {"$slot": slots.setdefault(value["$param"], len(slots))}
        if isinstance(value, Mapping):
            return {k: (offset(value[k]) if k == "offset_m" else canon(value[k])) for k in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [canon(v) for v in value]
        if _finite_number(value):
            return round(float(value), 4)
        return value

    def offset(values):
        return [round(round(float(v) / OFFSET_QUANTUM_M) * OFFSET_QUANTUM_M, 4) for v in values]

    skeleton = [[s["tool"], canon(s["args"])] for s in spec["steps"]]
    blob = json.dumps({"v": PROGRAM_VERSION, "steps": skeleton}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _render(value) -> str:
    if _is_param(value):
        return f"<{value['$param']}>"
    return repr(value) if isinstance(value, str) else str(value)


def describe_steps(steps: Iterable[tuple[str, dict]]) -> str:
    parts = []
    for tool, args in steps:
        rendered = []
        for key, value in args.items():
            if key == "$target" and isinstance(value, Mapping):
                off = ", ".join(f"{float(v):+.3f}" for v in value.get("offset_m") or [])
                rendered.append(f"{'/'.join(value.get('args') or [])}=localize_object("
                                f"{_render(value.get('label'))})+({off})")
            else:
                rendered.append(f"{key}={_render(value)}")
        parts.append(f"{tool}({', '.join(rendered)})")
    return "; ".join(parts)


@dataclass(frozen=True, eq=False)
class Program:
    """A validated program: normalized spec + structural signature."""

    spec: dict
    signature: str

    @classmethod
    def from_spec(cls, spec: Any, *, tool_specs: Iterable[Mapping] | None = None) -> "Program":
        norm = validate_program(spec, tool_specs)
        return cls(norm, program_signature(norm))

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def description(self) -> str:
        return self.spec.get("description", "")

    @property
    def params(self) -> dict:
        return dict(self.spec.get("params") or {})

    @property
    def steps(self) -> list[tuple[str, dict]]:
        return [(s["tool"], copy.deepcopy(s["args"])) for s in self.spec["steps"]]

    def describe(self) -> str:
        return f"{self.name}({', '.join(self.params)}): {describe_steps(self.steps)}"

    def bind(self, bindings: Any) -> list[tuple[str, dict]]:
        """Concrete-but-symbolic calls: every ``$param`` replaced by its label.

        Bindings must be exactly the declared parameters, each a non-empty
        object label; ``$target`` queries stay queries (grounded later).
        """
        if bindings is None:
            bindings = {}
        if not isinstance(bindings, Mapping):
            raise ProgramError("bind must map each parameter to an object label")
        params = self.params
        missing = sorted(set(params) - set(bindings))
        if missing:
            raise ProgramError(f"unbound parameter(s) {missing}")
        extra = sorted(set(bindings) - set(params))
        if extra:
            raise ProgramError(f"binding(s) for undeclared parameter(s) {extra}")
        values = {}
        for key, value in bindings.items():
            if (not isinstance(value, str) or not value.strip() or len(value) > MAX_LABEL_CHARS
                    or value.strip().startswith("$")):
                raise ProgramError(f"binding {key}={value!r}: a parameter is bound to an object label "
                                   "(a non-empty string), never a number or a marker")
            values[key] = value.strip()

        def sub(value):
            if _is_param(value):
                return values[value["$param"]]
            if isinstance(value, Mapping):
                return {k: sub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [sub(v) for v in value]
            return value

        return [(s["tool"], sub(s["args"])) for s in self.spec["steps"]]


# ── execution ────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """One executed step and the verdict the ledger gave it."""

    index: int
    tool: str
    args: dict
    verdict: str
    evidence: str
    result: dict = field(repr=False)

    def as_dict(self) -> dict:
        result = self.result if isinstance(self.result, Mapping) else {}
        return {"index": self.index, "tool": self.tool, "args": self.args, "verdict": self.verdict,
                "evidence": self.evidence, "ok": result.get("ok"), "outcome": result.get("outcome"),
                "error": result.get("error")}


def _short(args: Mapping) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


@dataclass
class ProgramRun:
    """What happened when a program ran, step by step, with its verdicts."""

    program: Program
    bindings: dict
    status: str = ABORTED
    steps: list[StepRecord] = field(default_factory=list)
    stopped_at: int | None = None
    reason: str = ""
    next_action: str | None = None
    grounded: list[dict] = field(default_factory=list)
    held: str | None = None

    @property
    def verdict(self) -> str:
        """CONFIRMED only for a completed run with at least one registered
        effect, every registered effect confirmed and no motion whose only
        evidence is its own report. Anything else is REFUTED (a refuted step)
        or UNVERIFIED -- a program's authorship is never evidence."""
        if any(s.verdict == REFUTED for s in self.steps):
            return REFUTED
        if (self.status == COMPLETED and any(s.verdict == CONFIRMED for s in self.steps)
                and all(s.verdict in (CONFIRMED, NO_EFFECT) for s in self.steps)):
            return CONFIRMED
        return UNVERIFIED

    @property
    def verified(self) -> bool:
        return self.verdict == CONFIRMED

    def summary(self) -> str:
        confirmed = sum(1 for s in self.steps if s.verdict == CONFIRMED)
        n = len(self.grounded)
        line = (f"done via program {self.program.name!r} ({len(self.steps)} step(s), "
                f"{confirmed} effect(s) confirmed")
        if n:
            line += f", {n} perception quer{'y' if n == 1 else 'ies'} re-grounded before motion"
        return line + "): " + "; ".join(f"{s.tool}({_short(s.args)})" for s in self.steps)

    def note(self) -> str:
        """The hand-off to the tier-3 planner when the program did not finish."""
        action = self.next_action or (f"Program {self.program.name!r} ended {self.status}: "
                                      f"{self.reason or self.verdict}.")
        return f"(A program was tried before this turn. {action})"

    def receipt(self) -> dict:
        return {"program": self.program.spec, "signature": self.program.signature,
                "bindings": self.bindings, "status": self.status, "verdict": self.verdict,
                "stopped_at": self.stopped_at, "reason": self.reason, "next_action": self.next_action,
                "grounded": self.grounded, "held": self.held,
                "steps": [s.as_dict() for s in self.steps]}


def _ledger(runtime) -> tuple[str | None, list] | None:
    """(task id, effect rows) from the runtime's task-effects ledger, or None."""
    fn = getattr(runtime, "task_effects", None)
    if fn is None:
        return None
    try:
        snap = fn()
    except Exception:  # noqa: BLE001 -- no readable ledger = no evidence
        return None
    rows = snap.get("effects") if isinstance(snap, Mapping) else None
    if not isinstance(rows, list):
        return None
    return snap.get("task_id"), list(rows)


def _step_rows(before, after) -> list | None:
    if before is None or after is None or before[0] != after[0] or len(after[1]) < len(before[1]):
        return None
    return after[1][len(before[1]):]


def _is_refusal(error: str) -> bool:
    return any(marker in error for marker in _REFUSAL_MARKERS)


def classify_step(tool: str, result: Any, rows: list | None) -> tuple[str, str]:
    """(verdict, evidence) for one executed step.

    Registered effects are judged ONLY by the ledger rows this step added
    (the checker's verdict). A result that says ``verified: true`` or carries
    its own ``postcondition`` is a claim, not evidence; with no ledger the
    effect stays UNVERIFIED.
    """
    if not isinstance(result, Mapping):
        return FAILED, "the step returned no result mapping"
    if result.get("outcome") == "stuck":
        return STUCK, str(result.get("ask") or result.get("error") or "stuck")
    own = [r for r in (rows or []) if isinstance(r, Mapping) and r.get("skill") == tool]
    statuses = [str(r.get("status")) for r in own]
    reasons = "; ".join(str(r.get("reason")) for r in own if r.get("reason"))
    if result.get("ok") is not True:
        if REFUTED in statuses:
            return REFUTED, reasons or str(result.get("error") or "refuted")
        error = str(result.get("error") or "the step reported failure")
        return (REFUSED if _is_refusal(error) else FAILED), error
    if tool in POSTCONDITIONS:
        if rows is None:
            return UNVERIFIED, "no task-effects ledger on this runtime: missing evidence stays unverified"
        if not own:
            return UNVERIFIED, "the task-effects ledger recorded no effect for this step"
        if REFUTED in statuses:
            return REFUTED, reasons
        if all(s == CONFIRMED for s in statuses):
            return CONFIRMED, reasons or "confirmed"
        return UNVERIFIED, reasons or "independent effect verdict unavailable"
    if tool in _MOTION_SKILLS:
        return UNCHECKED, f"{tool} has no registered postcondition: its own report is the only evidence"
    return NO_EFFECT, "read-only step"


_STOP_PHRASES = {
    REFUSED: "the safety harness refused it -- {e}",
    FAILED: "it failed -- {e}",
    REFUTED: "an independent channel REFUTED its effect -- {e}",
    UNVERIFIED: ("it reported ok but its effect is UNVERIFIED -- {e}; the program does not build later "
                 "steps on an effect nobody confirmed"),
}


def _stopped_action(run: ProgramRun) -> str:
    step = run.steps[-1]
    n = len(run.program.spec["steps"])
    done = ", ".join(f"{s.index}:{s.tool}" for s in run.steps[:-1]) or "none"
    phrase = _STOP_PHRASES[step.verdict].format(e=step.evidence)
    return (f"Program {run.program.name!r} stopped at step {step.index}/{n} {step.tool}({_short(step.args)}): "
            f"{phrase}. Earlier steps executed: {done}; {n - step.index} later step(s) were not run; "
            f"the gripper holds {run.held or 'nothing'}. Do not re-run this program or repeat the step "
            "unchanged: observe the scene and plan what is left from the current state.")


def _not_started(program: Program, reason: str) -> str:
    return (f"Program {program.name!r} was not started: {reason}. No motion was attempted; observe and "
            "plan from what is actually on the table.")


def run_program(program: Program, bindings: Any, runtime, *, tier: str = "program",
                tool_log: list | None = None) -> ProgramRun:
    """Bind, ground, then execute every step through ``runtime.execute()``.

    Grounding (perception only) happens for ALL ``$target`` queries before the
    first step; a query that does not resolve aborts with zero motion. Each
    step is an ordinary top-level call tagged ``tier``; the run stops at the
    first step whose verdict is in ``STOPPING``.
    """
    run = ProgramRun(program, dict(bindings or {}))
    log = tool_log if tool_log is not None else []
    tag = program.signature[:12]
    try:
        calls = program.bind(bindings)
    except ProgramError as e:
        run.reason = f"binding refused: {e}"
        run.next_action = _not_started(program, run.reason)
        return run

    def dispatch(name: str, args: dict):
        runtime.current_tier = tier
        try:
            return runtime.execute(name, args)
        finally:
            runtime.current_tier = None

    def localize(label: str):
        result = dispatch("localize_object", {"label": label})
        log.append({"step": len(log) + 1, "tier": tier, "tool": "localize_object", "args": {"label": label},
                    "result": result, "grounding": True, "program": tag})
        ok = isinstance(result, Mapping) and bool(result.get("ok"))
        run.grounded.append({"label": label, "ok": ok,
                             "position": result.get("position") if isinstance(result, Mapping) else None})
        return result

    try:
        calls, _ = recipes.ground(calls, localize)
    except recipes.GroundingError as e:
        run.reason = f"grounding failed before any step: {e}"
        run.held = getattr(runtime, "held_object", None)
        run.next_action = _not_started(program, run.reason)
        return run

    offset = len(log)
    run.status = COMPLETED
    for i, (name, args) in enumerate(calls, start=1):
        before = _ledger(runtime)
        result = dispatch(name, args)
        verdict, evidence = classify_step(name, result, _step_rows(before, _ledger(runtime)))
        run.steps.append(StepRecord(i, name, dict(args), verdict, evidence,
                                    result if isinstance(result, dict) else {"ok": False}))
        log.append({"step": offset + i, "tier": tier, "tool": name, "args": args, "result": result,
                    "program": tag})
        if verdict in STOPPING:
            run.status = STUCK if verdict == STUCK else STOPPED
            run.stopped_at = i
            run.reason = f"step {i} {name}: {verdict} -- {evidence}"
            break
    run.held = getattr(runtime, "held_object", None)
    if run.status == STOPPED:
        run.next_action = _stopped_action(run)
    elif run.status == STUCK:
        run.next_action = f"STUCK at {run.steps[-1].tool}: relay the ask to the human verbatim and stop."
    return run


def write_receipt(run: ProgramRun, run_dir) -> Path | None:
    """``<run>/programs/<n>_<name>.json``: spec, signature, bindings, grounded
    queries and per-step verdicts. Reporting never fails a task."""
    if run_dir is None:
        return None
    try:
        out = Path(run_dir) / "programs"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{len(list(out.glob('*.json'))):02d}_{run.program.name}.json"
        path.write_text(json.dumps(run.receipt(), indent=1, default=str) + "\n")
        return path
    except Exception:  # noqa: BLE001
        return None


# ── distillation (a verified LLM-tier run -> a program candidate) ─────────


def distill_program(tool_log: Iterable[dict], scene: Iterable[dict], *, task: str = "") -> dict:
    """A verified run's motion steps as a validated program spec.

    Coordinates become ``localize_object(label)+offset`` queries exactly as
    for a tier-2 recipe (``recipes.symbolize_run``; an unanchorable coordinate
    refuses the whole program), then every object label is lifted to a
    parameter named by its role. Raises ``ProgramError`` / ``RecipeError``
    when the run cannot be a program: nothing moved, no registered effect to
    verify, or a motion whose only evidence is its own report.
    """
    steps = recipes.symbolize_run(list(tool_log), list(scene))
    if not steps:
        raise ProgramError("the run moved nothing: an answer, not a program")
    if not any(name in POSTCONDITIONS for name, _ in steps):
        raise ProgramError("the run has no registered effect an independent channel could confirm")
    unchecked = [name for name, _ in steps if name in _MOTION_SKILLS and name not in POSTCONDITIONS]
    if unchecked:
        raise ProgramError(f"motion step(s) {unchecked} have no registered postcondition, so the run "
                           "cannot be verified end to end")
    params: dict[str, str] = {}
    by_value: dict[str, str] = {}

    def lift(tool: str, key: str, value):
        if not isinstance(value, str) or not value.strip():
            return value
        if value not in by_value:
            role = _LABEL_ROLES.get((tool, key), "label")
            name, n = role, 2
            while name in params:
                name, n = f"{role}_{n}", n + 1
            params[name] = f"{role} (was {value!r} in the verified run)"[:160]
            by_value[value] = name
        return {"$param": by_value[value]}

    out = []
    for tool, args in steps:
        new = {}
        for key, value in args.items():
            if key == "$target" and isinstance(value, Mapping):
                target = dict(value)
                target["label"] = lift("$target", "label", target.get("label"))
                new[key] = target
            elif key in _LABEL_KEYS:
                new[key] = lift(tool, key, value)
            else:
                new[key] = value
        out.append({"tool": tool, "args": new})
    name = _slug("-".join(dict.fromkeys(tool for tool, _ in steps)))[:40] or "program"
    return validate_program({"version": PROGRAM_VERSION, "name": name,
                             "description": f"distilled from a verified run of {task!r}"[:300],
                             "params": params, "steps": out})


# ── authoring ────────────────────────────────────────────────────────────


@dataclass
class Proposal:
    """The brain's answer to the authoring turn, parsed and validated."""

    kind: str
    program: Program | None = None
    bindings: dict = field(default_factory=dict)
    reason: str = ""
    record: ProgramRecord | None = None


def _first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


_EXAMPLE = {
    "program": {
        "name": "place-in-front-of",
        "description": "put an object 6 cm in front of another object",
        "params": {"object": "what to move", "anchor": "what to place it in front of"},
        "steps": [
            {"tool": "grasp_object", "args": {"label": {"$param": "object"}}},
            {"tool": "place_at", "args": {"$target": {"label": {"$param": "anchor"}, "offset_m": [-0.06, 0.0]}}},
            {"tool": "move_home", "args": {}},
        ],
    },
    "bind": {"object": "red cube", "anchor": "blue bowl"},
}


class ProgramTier:
    """Author / validate / run / account -- the orchestrator's program tier.

    Owns no runtime and no LLM: the orchestrator makes the one authoring call
    and hands the text here; execution goes through the runtime it passes in.
    """

    def __init__(self, library: ProgramLibrary, *, tool_specs: Iterable[Mapping] | None = None,
                 max_offered: int = 3):
        self.library = library
        self.tool_specs = list(TOOL_SPECS if tool_specs is None else tool_specs)
        self.max_offered = int(max_offered)

    def offered(self, task: str) -> list[ProgramRecord]:
        """Promoted programs worth showing the brain for this task (never candidates)."""
        try:
            return self.library.retrievable(task, max_entries=self.max_offered)
        except Exception:  # noqa: BLE001 -- a broken store offers nothing
            return []

    def catalog(self) -> list[str]:
        lines = []
        for spec in self.tool_specs:
            name = spec["name"]
            if name in NON_COMPOSABLE:
                continue
            schema = spec.get("parameters") or {}
            required = set(schema.get("required") or [])
            args = []
            for key, prop in (schema.get("properties") or {}).items():
                if key == "arm":
                    continue
                kind = "|".join(map(str, prop["enum"])) if "enum" in prop else str(prop.get("type"))
                args.append(f"{key}{'*' if key in required else ''}: {kind}")
            if name in recipes.COORDINATE_ARGS:
                args = ["$target*: {label, offset_m}"]
            lines.append(f"- {name}({', '.join(args)}){' [moves]' if name in _MOTION_SKILLS else ''}")
        return lines

    def prompt(self, task: str, offered: Iterable[ProgramRecord] = (), world: Iterable[str] = ()) -> str:
        world = list(world)
        lines = [
            "Write a PROGRAM for this task, or answer NONE.",
            "",
            f"Task: {task}",
            "",
            "A program is a fixed list of the robot's registered tools that runs in order WITHOUT further "
            "turns. Every step is still checked by the safety harness and verified independently; the "
            "first step that fails, is refused or cannot be confirmed stops the program and hands the "
            "task back to you.",
            "Rules:",
            "- Use only the tools below (* = required argument; [moves] = moves the arm).",
            "- Refer to objects by label, never by coordinates. place_at takes {\"$target\": {\"label\": L, "
            "\"offset_m\": [dx, dy]}}: the position of L, measured again before the first step, plus an "
            "offset in metres (+x away from the robot, +y to its left; at most 0.40 m). Numeric x/y/z "
            "are refused.",
            "- Put the labels this task names in \"params\" (name -> what it is) and use "
            "{\"$param\": name} in the steps; \"bind\" gives this task's labels.",
            f"- At most {MAX_PROGRAM_STEPS} steps and {MAX_PROGRAM_PARAMS} params; no loops or branches; "
            "never task_done -- independent verification decides the outcome, not the program.",
            "- Answer NONE if the task is a question, needs to look before deciding, or is not a fixed "
            "sequence of steps.",
            "",
            "Tools:",
            *self.catalog(),
            "",
            "World model (labels only):",
            *([f"- {w}" for w in world] or ["- nothing tracked yet"]),
        ]
        offered = list(offered)
        if offered:
            lines += ["", f"Reusable programs (each verified in >= {self.library.min_tasks} distinct tasks); "
                          "reuse one with {\"use\": <name>, \"bind\": {...}}:"]
            for rec in offered:
                try:
                    shown = Program.from_spec(rec.program, tool_specs=self.tool_specs).describe()
                except ProgramError:
                    continue
                lines.append(f"- {shown} [promoted: {rec.n_tasks} distinct tasks, {rec.occurrences} "
                             f"verified runs, {rec.losses} failed]")
        lines += [
            "",
            "Reply with ONE JSON object -- {\"program\": {\"name\", \"description\", \"params\", \"steps\": "
            "[{\"tool\", \"args\"}]}, \"bind\": {...}} or {\"use\": <name>, \"bind\": {...}} -- or NONE.",
            "Example: " + json.dumps(_EXAMPLE),
        ]
        return "\n".join(lines)

    def parse(self, text: str, offered: Iterable[ProgramRecord] = ()) -> Proposal:
        """Validate the brain's reply. Nothing here touches the robot."""
        raw = str(text or "").strip()
        obj = _first_json_object(raw)
        if obj is None:
            if not raw or re.match(r"^\W*none\b", raw, re.IGNORECASE):
                return Proposal(NONE, reason="the brain declined to write a program")
            return Proposal(INVALID, reason="the answer held no JSON program object")
        bindings = obj.get("bind") or {}
        try:
            if "use" in obj:
                key = obj.get("use")
                record = next((r for r in offered if key in (r.name, r.signature)), None)
                if record is None:
                    known = self.library.get(key) if isinstance(key, str) else None
                    if known is not None and known.retrievable(self.library.min_tasks):
                        record = known
                    elif known is not None:
                        return Proposal(INVALID, reason=(
                            f"{key!r} is a {known.status(self.library.min_tasks)} program, not promoted: only "
                            f"programs verified in >= {self.library.min_tasks} distinct tasks (and more "
                            "often than they failed) are reused"))
                    else:
                        return Proposal(INVALID, reason=f"{key!r} is an unknown program")
                # Re-validated against the CURRENT catalog: a tool may have changed since admission.
                program = Program.from_spec(record.program, tool_specs=self.tool_specs)
                program.bind(bindings)
                return Proposal(REUSE, program, dict(bindings), record=record)
            if "program" in obj:
                spec = obj["program"]
            elif "steps" in obj:
                spec = {k: v for k, v in obj.items() if k != "bind"}
            else:
                return Proposal(INVALID, reason="expected {'program': {...}, 'bind': {...}} or "
                                                "{'use': <name>, 'bind': {...}}")
            program = Program.from_spec(spec, tool_specs=self.tool_specs)
            program.bind(bindings)
            return Proposal(FRESH, program, dict(bindings))
        except ProgramError as e:
            return Proposal(INVALID, reason=str(e))

    def run(self, proposal: Proposal, runtime, *, tool_log: list | None = None) -> ProgramRun:
        return run_program(proposal.program, proposal.bindings, runtime, tool_log=tool_log)

    def account(self, run: ProgramRun, proposal: Proposal, *, task: str, run_id: str, success: bool) -> str:
        """Library bookkeeping for one execution; never raises.

        Admitted only when the task report succeeded AND the run's own
        verdict is CONFIRMED. An admitted program that executed without
        verifying counts a loss; an aborted run (zero motion, e.g. its anchor
        is not on this table) counts nothing.
        """
        try:
            if success and run.verified:
                origin = "reused" if proposal.kind == REUSE else "authored"
                self.library.admit(run.program.spec, run.program.signature, task=task, run=run_id,
                                   origin=origin, verdict=run.verdict, summary=run.summary()[:200])
                return "admitted"
            if run.steps and self.library.get(run.program.signature) is not None:
                self.library.record_failure(run.program.signature, run=run_id,
                                            reason=run.reason or run.verdict)
                return "loss"
        except Exception as e:  # noqa: BLE001 -- bookkeeping never fails a task
            return f"library error: {type(e).__name__}: {e}"
        return "not admitted"

    def distil(self, tool_log: Iterable[dict], scene: Iterable[dict], *, task: str, run_id: str,
               verdict: str, summary: str = "") -> tuple[str, ProgramRecord | None]:
        """Fold a VERIFIED LLM-tier run into the library as a program."""
        try:
            spec = distill_program(tool_log, scene, task=task)
            record = self.library.admit(spec, program_signature(spec), task=task, run=run_id,
                                        origin="distilled", verdict=verdict, summary=summary)
            return "admitted", record
        except (ProgramError, recipes.RecipeError, ValueError) as e:
            return f"not kept as a program: {e}", None
