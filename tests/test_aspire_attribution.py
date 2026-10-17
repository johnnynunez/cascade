"""ASPIRE attribution: what cascade took from NVlabs/ASPIRE and what is its own.

Upstream (not vendored): v0.1 ``aspire/sim/cap/skills/library.py``
``get_promoted_skills(min_occurrences=2)`` counts occurrences in any task, and
v1.0 promotes an entry on a blind verifier's verdict
(``.claude/skills/iterative-debugging/verifier-prompt.md``). Cascade's ">= 2
distinct tasks" rule is therefore its own. v1.0's API
(``src/cap_harness/api.py``) has no ``preview_grasp``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Matched against each file with line breaks and comment markers folded to one
# space, so a claim wrapped over two lines is still found.
CLAIMS = re.compile(r"ASPIRE'?s? (cross-task |promotion )?rule|follows upstream ASPIRE|upstream ASPIRE promotes|"
                    r"Cross-task gate \(upstream ASPIRE\)|as upstream ASPIRE does|"
                    r"upstream's recurrence threshold|recurs beyond a single task", re.IGNORECASE)
SKIP = ("docs/superpowers/", "tests/test_aspire_attribution.py")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".sh", ".mjs", ".js", ".txt", ".toml"}
_WRAP = re.compile(r"[ \t]*\n[ \t]*(?:#:?|//|\*)?[ \t]*")


def _text_files(roots: tuple[str, ...]):
    for root in roots:
        base = REPO / root
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts:
                yield path


def _claims(text: str) -> list[str]:
    return [m.group(0) for m in CLAIMS.finditer(_WRAP.sub(" ", text))]


def test_claim_scanner_finds_wrapped_claims():
    assert _claims("#: Upstream ASPIRE promotes a skill when it recurs beyond a single\n#: task.") == [
        "Upstream ASPIRE promotes", "recurs beyond a single task"]
    assert _claims("(>= 2 distinct tasks,\nASPIRE's rule)") == ["ASPIRE's rule"]
    assert _claims("cascade's own rule: >= 2 distinct tasks") == []


def test_no_file_claims_cascades_promotion_rule_is_upstreams():
    hits = {str(p.relative_to(REPO)): found
            for p in _text_files(("src", "scripts", "configs", "docs", "tests", "README.md"))
            if not str(p.relative_to(REPO)).startswith(SKIP)
            and (found := _claims(p.read_text(errors="replace")))}
    assert hits == {}


def test_architecture_credits_aspire_with_its_own_api():
    line = next(l for l in (REPO / "docs/ARCHITECTURE.md").read_text().splitlines()
                if l.startswith("- **ASPIRE**"))
    for name in ("get_observation", "localize_object", "generate_grasps", "select_grasp", "move_to_pose"):
        assert name in line
    assert "`preview_grasp` is cascade's own" in line
