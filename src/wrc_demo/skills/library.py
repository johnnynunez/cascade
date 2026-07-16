"""Learned-skill library (ASPIRE Sec 2.2): validated repairs persisted as
markdown so future runs (and future agents) start smarter.

Entry schema mirrors the paper: failure signature, when-to-apply guard,
strategy, optional parameters (e.g. per-object grasp yaw/z-offset), origin
task. Files live in <repo>/skills_library/<slug>.md and are loaded verbatim
into the agent's context when their guard keywords match the task.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

ENTRY_TEMPLATE = """# {title}

- **Failure signature:** {failure}
- **When to apply:** {when}
- **Origin task:** {origin}
- **Recorded:** {date}

## Strategy

{strategy}
"""


class SkillLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, title: str, failure: str, when: str, strategy: str, origin: str = "") -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "entry"
        path = self.root / f"{slug}.md"
        path.write_text(
            ENTRY_TEMPLATE.format(
                title=title,
                failure=failure,
                when=when,
                origin=origin or "(manual)",
                date=time.strftime("%Y-%m-%d"),
                strategy=strategy,
            )
        )
        return path

    def entries(self) -> list[tuple[str, str]]:
        """-> [(name, markdown), ...]"""
        return [(p.stem, p.read_text()) for p in sorted(self.root.glob("*.md"))]

    def relevant(self, task: str, max_entries: int = 3) -> list[str]:
        """Naive keyword guard match against the 'When to apply' line."""
        task_words = set(re.findall(r"[a-z]+", task.lower()))
        scored = []
        for name, text in self.entries():
            m = re.search(r"\*\*When to apply:\*\*(.+)", text)
            guard_words = set(re.findall(r"[a-z]+", (m.group(1) if m else "").lower()))
            overlap = len(task_words & guard_words)
            if overlap:
                scored.append((overlap, text))
        scored.sort(key=lambda t: -t[0])
        return [text for _, text in scored[:max_entries]]
