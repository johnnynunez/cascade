"""Keep the HD Spark demo readable and the PAAI roadmap explicit."""
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import re
import struct


REPO = Path(__file__).resolve().parents[1]
DEMO = "docs/assets/paai-dgx-spark-natural-language-demo.gif"


class _DemoHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def test_readme_first_block_is_centered_spark_demo_with_install_link():
    title, body = (REPO / "README.md").read_text().split("\n", 1)
    assert title.startswith("# ")
    first_block = re.match(r"\s*(<p\b.*?</p>)", body, re.DOTALL)
    assert first_block, "The demo must appear immediately below the H1"
    demo = _DemoHTML()
    demo.feed(first_block.group(1))
    assert demo.tags[0] == ("p", {"align": "center"})
    images = [attrs for tag, attrs in demo.tags if tag == "img"]
    assert len(images) == 1
    assert images[0]["src"] == DEMO
    assert all(word in images[0]["alt"] for word in ("OpenClaw", "tomato can", "green square"))
    assert [attrs["href"] for tag, attrs in demo.tags if tag == "a"] == [
        "docs/DGX_SPARK_SETUP.md"
    ]
    assert " ".join("".join(demo.text).split()) == (
        "PAAI demo running on NVIDIA DGX Spark · 6× speed"
    )
    assert (REPO / "docs/DGX_SPARK_SETUP.md").is_file()


def test_readme_demo_uses_readable_width_without_fixed_height():
    demo = _DemoHTML()
    demo.feed((REPO / "README.md").read_text())
    image = next(attrs for tag, attrs in demo.tags if attrs.get("src") == DEMO)
    # GitHub caps images at the content width. An automatic height preserves
    # the aspect ratio when its column is narrower than the 720p source.
    assert image.get("width") == "1280"
    assert "height" not in image


def test_readme_gif_preserves_fully_decoded_supplied_animation():
    data = (REPO / DEMO).read_bytes()
    # This exact asset was fully decoded: 322 frames, each 50 ms (16.1 s).
    # Pinning the bytes also guards its validated source and 6× playback.
    assert sha256(data).hexdigest() == (
        "7ee20fec8228508c0565179d4ab6134c1f3e617f9be469058fa0919ebe25fada"
    )
    assert len(data) == 29_581_644
    assert data[:6] == b"GIF89a"
    assert struct.unpack("<HH", data[6:10]) == (1280, 720)
    assert b"NETSCAPE2.0\x03\x01\x00\x00\x00" in data  # Infinite loop.
    assert data[-1:] == b";"  # Complete GIF trailer, not an LFS pointer.


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
