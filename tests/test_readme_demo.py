"""Keep the verified DGX Spark animation visible at the README entry point."""
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


def test_readme_gif_preserves_fully_decoded_supplied_animation():
    data = (REPO / DEMO).read_bytes()
    # This exact asset was fully decoded: 322 frames, each 50 ms (16.1 s).
    # Pinning the bytes also guards its validated source and 6× playback.
    assert sha256(data).hexdigest() == (
        "574215fb339fb5611c7833299348b16671bd6edbc798e7a7cbb107f922eaa5d2"
    )
    assert len(data) == 5_689_730
    assert data[:6] == b"GIF89a"
    assert struct.unpack("<HH", data[6:10]) == (480, 270)
    assert b"NETSCAPE2.0\x03\x01\x00\x00\x00" in data  # Infinite loop.
    assert data[-1:] == b";"  # Complete GIF trailer, not an LFS pointer.
