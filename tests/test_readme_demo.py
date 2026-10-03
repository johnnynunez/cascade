"""Keep the PAAI roadmap explicit and concise."""
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]


def _roadmap():
    readme = (REPO / "README.md").read_text()
    headings = re.findall(r"^## (.+)$", readme, re.MULTILINE)
    assert headings.count("Roadmap") == 1
    index = headings.index("Roadmap")
    assert headings[index - 1:index + 2] == [
        "PAAI, Physical Agentic AI", "Roadmap", "Brev deployment"
    ]
    return readme.split("## Roadmap\n", 1)[1].split("\n## ", 1)[0].strip()


def test_readme_roadmap_separates_acceptance_and_future_backend_work():
    roadmap = " ".join(_roadmap().split())
    acceptance, future = roadmap.split("**Newton parity.**", 1)
    assert "docs/PROJECT_STATUS_20261003.md" in acceptance
    assert all(word in acceptance for word in (
        "Spark", "nvblox", "acceptance", "five-object", "restart"
    ))
    newton, future = future.split("**Cosmos 3 Edge.**", 1)
    assert re.search(r"Spark.*\bdefaults?\b.*\bPhysX\b", newton)
    assert all(word in newton for word in ("Newton", "explicit", "separate", "validation"))
    cosmos, ovrtx = future.split("**ovrtx**", 1)
    assert all(word in cosmos for word in ("native tool calling", "OpenClaw", "Qwen"))
    assert re.search(r"\bwhen\b.*OpenClaw", cosmos)
    assert re.search(r"\bUntil\b.*Qwen", cosmos)
    assert re.search(r"\bseparate(?:ly)?\b", ovrtx)
    assert re.search(r"\bnot included\b.*\bruntime baseline\b", ovrtx)
    assert "nondeterminism" not in roadmap.lower()


def test_readme_roadmap_stays_compact_and_uses_plain_copy():
    roadmap = _roadmap()
    assert len(roadmap.split()) <= 110
    assert roadmap.isascii()
    assert not re.search(
        r"\b(seamless|robust|game-changing|moreover|furthermore|in conclusion|"
        r"in summary|to summarize)\b", roadmap, re.IGNORECASE
    )
