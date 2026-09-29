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


def test_readme_roadmap_describes_pending_parity_and_tool_calling_gates():
    roadmap = " ".join(_roadmap().split())
    assert roadmap.startswith("We are working on two things next.")
    newton, cosmos = roadmap.split("**Cosmos 3 Edge.**", 1)
    assert "**Newton parity.**" in newton
    assert "current event path uses CUDA PhysX" in newton
    assert "repeatable setup today" in newton
    assert "We are closing the cross-engine reproducibility gap" in newton
    assert "contact, grasp and placement behavior" in newton
    assert "then we can move the full demo to Newton" in newton
    assert "We will add it when native tool calling through OpenClaw" in cosmos
    assert "consistent enough for normal attendee requests" in cosmos
    assert "Until then, Qwen remains the working event path" in cosmos
    assert "nondeterminism" not in roadmap.lower()


def test_readme_roadmap_stays_compact_and_uses_plain_copy():
    roadmap = _roadmap()
    assert len(roadmap.split()) <= 110
    assert roadmap.isascii()
    assert not re.search(
        r"\b(seamless|robust|game-changing|moreover|furthermore|in conclusion|"
        r"in summary|to summarize)\b", roadmap, re.IGNORECASE
    )
