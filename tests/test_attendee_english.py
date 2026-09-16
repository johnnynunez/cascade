"""Public attendee examples use ordinary English, without tool instructions."""
from html.parser import HTMLParser
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "What can you see on the table?",
    "Could you put the green cube in the green square?",
    "Please put the orange in the open box.",
    "Let's start over.",
}


class Quotes(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.examples = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "blockquote":
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "blockquote" and self.current is not None:
            self.examples.append(" ".join("".join(self.current).split()))
            self.current = None


def test_staff_examples_cover_natural_scene_actions_without_internal_names():
    page = (REPO / "deploy/brev/staff.html").read_text()
    quotes = Quotes()
    quotes.feed(page)
    assert '<html lang="en">' in page
    assert EXAMPLES <= set(quotes.examples)
    assert not any("red cube" in quote or "red zone" in quote for quote in quotes.examples)


def test_staff_recovery_does_not_require_a_tool_name():
    quotes = Quotes()
    quotes.feed((REPO / "deploy/brev/staff.html").read_text())
    assert not re.search(r"cascade__|reset_scene|pick_and_place|exactly once", "\n".join(quotes.examples))


def test_spark_attendee_examples_use_the_same_english_requests():
    guide = (REPO / "docs/DGX_SPARK_SETUP.md").read_text()
    examples = {line.removeprefix("> ") for line in guide.splitlines() if line.startswith("> ")}
    assert EXAMPLES <= examples
    assert not re.search(r"cascade__|reset_scene|pick_and_place|exactly once", "\n".join(examples))


def test_attendee_guidance_is_short_english_and_uses_only_attendee_tools():
    instructions = (REPO / "demo/kitchen/visitor-instructions.md").read_text()
    assert "Answer attendees in English." in instructions
    assert len(instructions.split()) <= 400
    names = set(re.findall(r"cascade__([a-z_]+)", instructions))
    assert names <= {"describe_scene", "localize_object", "pick_and_place", "reset_scene", "camera_snapshot", "world_state"}
    assert "clarification" in instructions and "ambiguous" in instructions
    assert "verified" in instructions and "failed" in instructions


def test_public_quickstart_does_not_advertise_spanish_attendee_instructions():
    quickstart = (REPO / "docs/QUICKSTART.md").read_text()
    assert "accepts English and Spanish instructions" not in quickstart


def test_public_camera_page_offers_the_supported_english_orange_request():
    page = (REPO / "deploy/brev/visitor.html").read_text()
    assert '<html lang="en">' in page
    assert "What can you see on the table?" in page
    assert "Please put the orange in the open box." in page
