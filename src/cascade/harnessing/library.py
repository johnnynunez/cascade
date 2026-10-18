# SPDX-License-Identifier: Apache-2.0
"""Per-family experience library: adopted harnesses and evidence-backed lessons, in task order.

Layout (Harness²'s ``library.py`` shape, scoped for CASCADE):

    <root>/<scope>/<domain>/INDEX.md
    <root>/<scope>/<domain>/entries/<NNN>-<task>/harness.json, HARNESS.md, EXPERIENCE.md, meta.json

``scope`` binds the rig and the brain (arm, camera, executor LLM profile and model): a harness
adopted for one brain is evidence about that brain only. Improver sessions read only EARLIER
entries of the same scope and domain, and only ``harness.json`` / ``HARNESS.md`` /
``EXPERIENCE.md``: ``meta.json`` holds the measured outcomes and never enters an improver's
context (answer-free).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

from .surface import AgentHarness

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str) -> str:
    return _SAFE.sub("-", str(text)).strip("-")[:80] or "x"


def scope_key(*, arm: str, camera: str, llm_profile: str, model: str = "") -> str:
    return slug("__".join(p for p in (arm, camera, llm_profile, model) if p))


@dataclass
class Library:
    root: Path
    scope: str
    domain: str

    @property
    def dir(self) -> Path:
        return Path(self.root) / slug(self.scope) / slug(self.domain)

    @property
    def index(self) -> Path:
        return self.dir / "INDEX.md"

    def entries(self) -> list[Path]:
        base = self.dir / "entries"
        return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []

    def latest_adopted(self) -> AgentHarness | None:
        for entry in reversed(self.entries()):
            path = entry / "harness.json"
            if path.is_file():
                return AgentHarness.read(path)
        return None

    def base_or(self, seed: AgentHarness) -> AgentHarness:
        """The harness the family's next task starts from: the latest adopted one, else `seed`."""
        return self.latest_adopted() or seed

    def context_into(self, dst: Path) -> bool:
        """Copy the readable part of every earlier entry into ``dst``; False when empty."""
        entries = self.entries()
        if not entries:
            return False
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        if self.index.is_file():
            shutil.copy2(self.index, dst / "INDEX.md")
        for entry in entries:
            target = dst / entry.name
            target.mkdir(exist_ok=True)
            for name in ("harness.json", "HARNESS.md", "EXPERIENCE.md"):
                if (entry / name).is_file():
                    shutil.copy2(entry / name, target / name)
        return True

    def append(self, task_id: str, harness: AgentHarness, experience_md: str, meta: dict,
               lesson_line: str) -> Path:
        n = len(self.entries()) + 1
        entry = self.dir / "entries" / f"{n:03d}-{slug(task_id)}"
        entry.mkdir(parents=True, exist_ok=False)
        harness.write(entry)
        (entry / "EXPERIENCE.md").write_text(experience_md)
        (entry / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        header = f"# Library `{self.scope}` / `{self.domain}`\n\n" if not self.index.is_file() else ""
        with self.index.open("a") as stream:
            stream.write(header + f"- {entry.name}: harness `{harness.sha()[:12]}`; {lesson_line}\n")
        return entry
