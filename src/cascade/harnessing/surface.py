# SPDX-License-Identifier: Apache-2.0
"""The agent harness a run of the built-in orchestrator reads, as data.

Harness² (Xu et al., arXiv:2610.14224) edits the text around a fixed model: prompt, guardrails,
memory, tool guidance, skills, scripts, subagents and plugins. CASCADE exposes the three of those
that are pure text and cannot move the robot by themselves:

- ``rules``: short always-on lines appended to the system prompt;
- ``tool_guidance``: one note per tool, appended to that tool's description;
- ``skills``: triggered advice, injected into a task's intro only when a trigger word occurs in
  the task.

Everything else is LOCKED: the safety harness, every ``safety.*`` / ``grasp.*`` value, profiles,
code, and the tools in :data:`LOCKED_TOOLS`. A harness can only add advice text; the safety
harness still refuses any motion it refuses today. :func:`AgentHarness.problems` validates a
harness before any rollout uses it (caps, known tools, locked tools, and a lint that refuses text
that would weaken a guard, claim an unverified success or override the rules). The lint is the
second line: the first is that a harness has no channel to configuration.

The canonical form is one JSON document (``harness.json``); :meth:`AgentHarness.sha` hashes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 1

#: tools a harness may neither guide nor name: raw joint/jog motion, history rewrites, the
#: scene reset the agent never runs on its own, the stop tool, the throw, and clearing a safety
#: stop latch (composed robots' ``reset_stop``: only an operator clears a stop)
LOCKED_TOOLS = frozenset({"rotate_joint", "move_relative", "undo_move", "reset_scene",
                          "halt_motion", "throw", "reset_stop"})

RULES_MAX = 12
RULE_MAX_CHARS = 240
ALWAYS_ON_MAX_CHARS = 1500
GUIDANCE_MAX_TOOLS = 10
GUIDANCE_MAX_CHARS = 300
SKILLS_MAX = 8
SKILL_BODY_MAX_CHARS = 800
TRIGGER_MAX_WORDS = 8

_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
_WORD = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_IDENT = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
_SENTENCE = re.compile(r"[^.!?;\n]+")

# Lint vocabulary. A sentence that mentions a guard must not also carry a weakening cue; a
# sentence that talks about reporting success must not mention failure; rule-hierarchy and
# concealment wording is refused outright.
_GUARD = re.compile(r"\b(safety|harness|collision|clearance|margin|limits?|interlock|watchdog|"
                    r"e-?stop|guard|gate|check(s|ing)?|verif\w*|refus\w*|reject\w*|warning)\b")
_WEAKEN = re.compile(r"\b(disabl\w*|bypass\w*|ignor\w*|overrid\w*|skip\w*|turn(ing)? off|switch\w* off|"
                     r"deactivat\w*|disarm\w*|unlock\w*|kill\w*|cancel\w*|lower\w*|reduc\w*|lessen\w*|"
                     r"relax\w*|loosen\w*|remov\w*|suppress\w*|mut(e|ing)|silenc\w*|regardless|"
                     r"despite|noise|manual mode|forc\w*|through them|no heed|pay no|accept no|"
                     r"brings? (it|the \w+) down|work around|circumvent\w*|defeat\w*)\b")
_CLAIM = re.compile(r"\b(say|tell|report|claim|inform|answer|reply|let .{0,30}know|summar\w*)\b")
_SUCCESS = re.compile(r"\b(succe\w*|placed|in place|delivered|done|worked|went well|complete\w*|"
                      r"in the (square|box|bowl|zone))\b")
_FAILURE = re.compile(r"\b(fail\w*|refus\w*|unverified|regardless|even if|even when|anyway|"
                      r"error\w*|rejected)\b")
_CONCEAL = re.compile(r"\b(hide|conceal|never mention|do not mention|don't mention|omit|keep .{0,20}"
                      r"secret|withhold)\b")
_HIERARCHY = re.compile(r"\b(rules?|instructions?|prompt|system message)\b.{0,60}\b(outrank\w*|"
                        r"overrid\w*|supersed\w*|no longer apply|take precedence|win|ignore)\b|"
                        r"\b(outrank\w*|supersed\w*|take precedence over)\b")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text.lower()) if s.strip()]


def lint_text(text: str) -> list[str]:
    """Problems in one piece of advice text; [] when it may be served."""
    problems = []
    if any(ord(c) < 32 and c != "\n" for c in text):
        problems.append("control character")
    for sentence in _sentences(text):
        if _GUARD.search(sentence) and _WEAKEN.search(sentence):
            problems.append(f"weakens a guard: {sentence[:80]!r}")
        if _CLAIM.search(sentence) and _SUCCESS.search(sentence) and _FAILURE.search(sentence):
            problems.append(f"claims success against a failure: {sentence[:80]!r}")
        if _CONCEAL.search(sentence):
            problems.append(f"concealment: {sentence[:80]!r}")
        if _HIERARCHY.search(sentence):
            problems.append(f"rule hierarchy: {sentence[:80]!r}")
    return problems


@dataclass(frozen=True)
class Skill:
    name: str
    trigger: tuple[str, ...]
    body: str

    def matches(self, task: str) -> bool:
        words = set(re.findall(r"[a-z0-9-]+", task.lower()))
        return any(t in words for t in self.trigger)

    def as_dict(self) -> dict:
        return {"name": self.name, "trigger": list(self.trigger), "body": self.body}


@dataclass(frozen=True)
class AgentHarness:
    rules: tuple[str, ...] = ()
    tool_guidance: tuple[tuple[str, str], ...] = ()
    skills: tuple[Skill, ...] = field(default=())

    # -- canonical form ------------------------------------------------------------------
    def as_dict(self) -> dict:
        return {"schema": SCHEMA, "rules": list(self.rules),
                "tool_guidance": {tool: text for tool, text in self.tool_guidance},
                "skills": [s.as_dict() for s in self.skills]}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def sha(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @property
    def empty(self) -> bool:
        return not (self.rules or self.tool_guidance or self.skills)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentHarness":
        """Build from untrusted JSON; raises ValueError on a wrong shape (content is checked by
        :meth:`problems`)."""
        if not isinstance(data, dict):
            raise ValueError("harness must be a JSON object")
        if data.get("schema", SCHEMA) != SCHEMA:
            raise ValueError(f"unsupported harness schema {data.get('schema')!r}")
        unknown = set(data) - {"schema", "rules", "tool_guidance", "skills"}
        if unknown:
            raise ValueError(f"unknown harness components {sorted(unknown)}: the editable surface is "
                             "rules, tool_guidance and skills; everything else is locked")
        rules = data.get("rules") or []
        guidance = data.get("tool_guidance") or {}
        skills = data.get("skills") or []
        if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
            raise ValueError("rules must be a list of strings")
        if not isinstance(guidance, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                     for k, v in guidance.items()):
            raise ValueError("tool_guidance must map tool names to strings")
        if not isinstance(skills, list):
            raise ValueError("skills must be a list")
        parsed = []
        for s in skills:
            if not isinstance(s, dict) or set(s) - {"name", "trigger", "body"}:
                raise ValueError("each skill is {name, trigger, body}")
            trigger = s.get("trigger") or []
            if isinstance(trigger, str):
                trigger = [w for w in re.split(r"[,\s]+", trigger) if w]
            if not isinstance(trigger, list) or not all(isinstance(w, str) for w in trigger):
                raise ValueError("skill trigger must be a list of words")
            parsed.append(Skill(str(s.get("name", "")).strip(), tuple(w.strip().lower() for w in trigger),
                                str(s.get("body", "")).strip()))
        return cls(tuple(r.strip() for r in rules if r.strip()),
                   tuple(sorted((k.strip(), v.strip()) for k, v in guidance.items() if v.strip())),
                   tuple(sorted(parsed, key=lambda s: s.name)))

    @classmethod
    def from_json(cls, text: str) -> "AgentHarness":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"harness is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    # -- disk ------------------------------------------------------------------------------
    def write(self, directory: Path) -> Path:
        """Write ``harness.json`` plus a human-readable ``HARNESS.md``; returns the JSON path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "harness.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)
        (directory / "HARNESS.md").write_text(self.render_markdown())
        return path

    @classmethod
    def read(cls, path: Path) -> "AgentHarness":
        path = Path(path)
        if path.is_dir():
            path = path / "harness.json"
        return cls.from_json(path.read_text())

    # -- validation ------------------------------------------------------------------------
    def problems(self, tool_names) -> list[str]:
        """Every reason this harness must not run, against the tools the rig actually offers."""
        tools = set(tool_names)
        out: list[str] = []
        if len(self.rules) > RULES_MAX:
            out.append(f"{len(self.rules)} rules > {RULES_MAX}")
        for rule in self.rules:
            if len(rule) > RULE_MAX_CHARS:
                out.append(f"rule over {RULE_MAX_CHARS} characters: {rule[:60]!r}")
            if "\n" in rule:
                out.append(f"rule spans lines: {rule[:60]!r}")
        if sum(len(r) + 3 for r in self.rules) > ALWAYS_ON_MAX_CHARS:
            out.append(f"always-on rules exceed {ALWAYS_ON_MAX_CHARS} characters")
        if len(self.tool_guidance) > GUIDANCE_MAX_TOOLS:
            out.append(f"guidance for {len(self.tool_guidance)} tools > {GUIDANCE_MAX_TOOLS}")
        for tool, text in self.tool_guidance:
            if tool in LOCKED_TOOLS or tool.rsplit(".", 1)[-1] in LOCKED_TOOLS:
                out.append(f"guidance for locked tool {tool}")
            elif tool not in tools:
                out.append(f"guidance for unknown tool {tool}")
            if len(text) > GUIDANCE_MAX_CHARS:
                out.append(f"guidance for {tool} over {GUIDANCE_MAX_CHARS} characters")
        if len(self.skills) > SKILLS_MAX:
            out.append(f"{len(self.skills)} skills > {SKILLS_MAX}")
        names = [s.name for s in self.skills]
        if len(set(names)) != len(names):
            out.append("duplicate skill names")
        for s in self.skills:
            if not _SKILL_NAME.match(s.name):
                out.append(f"bad skill name {s.name!r}")
            if not s.trigger or len(s.trigger) > TRIGGER_MAX_WORDS or not all(_WORD.match(w) for w in s.trigger):
                out.append(f"skill {s.name}: trigger must be 1-{TRIGGER_MAX_WORDS} lowercase words")
            if not s.body or len(s.body) > SKILL_BODY_MAX_CHARS:
                out.append(f"skill {s.name}: body must be 1-{SKILL_BODY_MAX_CHARS} characters")
        texts = list(self.rules) + [t for _, t in self.tool_guidance] + [s.body for s in self.skills]
        # composed robots namespace their tools (`carry.carry_object`); advice may use the local name
        local_names = {name.rsplit(".", 1)[-1] for name in tools}
        for text in texts:
            for ident in set(_IDENT.findall(text)):
                if ident in LOCKED_TOOLS:
                    out.append(f"names locked tool {ident}")
                elif ident.endswith(("_object", "_scene", "_gripper", "_at", "_pixel", "_point",
                                     "_home", "_rest", "_done", "_step", "_memory", "_grasp",
                                     "_place", "_observation", "_objects", "_view", "_color",
                                     "_arms", "_joint", "_relative", "_move", "_screw", "_near_object")) \
                        and ident not in tools and ident not in local_names:
                    out.append(f"names a tool this rig does not offer: {ident}")
            out.extend(lint_text(text))
        return out

    # -- rendering into the orchestrator -----------------------------------------------------
    def system_suffix(self) -> str:
        if not self.rules:
            return ""
        return ("\n\nLearned working rules (advice distilled from earlier runs; tool results and the "
                "safety harness always take precedence):\n" + "\n".join(f"- {r}" for r in self.rules) + "\n")

    def apply_tool_specs(self, specs: list[dict]) -> list[dict]:
        if not self.tool_guidance:
            return specs
        import copy

        guidance = dict(self.tool_guidance)
        out = copy.deepcopy(specs)
        for spec in out:
            note = guidance.get(spec.get("name"))
            if note:
                spec["description"] = f"{spec.get('description', '').rstrip()} Learned guidance: {note}"
        return out

    def intro_block(self, task: str) -> str:
        hits = [s for s in self.skills if s.matches(task)]
        if not hits:
            return ""
        return ("Learned skills for this kind of task (advice from earlier runs, not orders; tool "
                "results take precedence):\n" + "\n".join(f"[{s.name}] {s.body}" for s in hits))

    def render_markdown(self) -> str:
        lines = [f"# Agent harness `{self.sha()[:12]}`", ""]
        lines += ["## Rules", ""] + ([f"- {r}" for r in self.rules] or ["(none)"]) + [""]
        lines += ["## Tool guidance", ""] + ([f"- `{t}`: {g}" for t, g in self.tool_guidance] or ["(none)"]) + [""]
        lines += ["## Skills", ""]
        for s in self.skills:
            lines += [f"### {s.name}", f"trigger: {', '.join(s.trigger)}", "", s.body, ""]
        if not self.skills:
            lines += ["(none)", ""]
        return "\n".join(lines)

    def diff(self, other: "AgentHarness") -> list[str]:
        """Component-level changes from ``self`` to ``other``, one line each."""
        out = []
        for r in other.rules:
            if r not in self.rules:
                out.append(f"rules + {r}")
        for r in self.rules:
            if r not in other.rules:
                out.append(f"rules - {r}")
        a, b = dict(self.tool_guidance), dict(other.tool_guidance)
        for tool in sorted(set(a) | set(b)):
            if a.get(tool) != b.get(tool):
                out.append(f"tool_guidance[{tool}] {'changed' if tool in a and tool in b else ('+' if tool in b else '-')}")
        sa, sb = {s.name: s for s in self.skills}, {s.name: s for s in other.skills}
        for name in sorted(set(sa) | set(sb)):
            if sa.get(name) != sb.get(name):
                out.append(f"skills[{name}] {'changed' if name in sa and name in sb else ('+' if name in sb else '-')}")
        return out
