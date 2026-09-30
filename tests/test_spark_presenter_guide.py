"""Contracts for the offline Spark presenter guide, not live-demo acceptance."""
from html.parser import HTMLParser
from pathlib import Path
import re

import pytest


GUIDE = Path(__file__).resolve().parents[1] / "docs/spark-presenter/index.html"


class GuideDocument(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.text = []
        self.code = {}
        self._code_id = None
        self._excluded = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if tag in {"script", "style"}:
            self._excluded += 1
        if tag == "code" and attrs.get("id"):
            self._code_id = attrs["id"]
            self.code[self._code_id] = ""

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._excluded -= 1
        if tag == "code":
            self._code_id = None

    def handle_data(self, data):
        if not self._excluded:
            self.text.append(data)
        if self._code_id:
            self.code[self._code_id] += data


def test_presenter_guide_is_offline_accessible_and_keeps_proof_boundaries():
    assert GUIDE.is_file(), "Missing in-repository presenter HTML"
    source = GUIDE.read_text(encoding="utf-8")
    doc = GuideDocument()
    doc.feed(source)
    text = " ".join(" ".join(doc.text).split())
    assert ("html", {"lang": "en"}) in doc.elements
    assert any(tag == "meta" and attrs.get("name") == "viewport" for tag, attrs in doc.elements)
    assert any(tag == "main" for tag, _ in doc.elements)
    assert sum(tag == "h1" for tag, _ in doc.elements) == 1
    assert any(attrs.get("aria-live") == "polite" for _, attrs in doc.elements)
    ids = [attrs["id"] for _, attrs in doc.elements if "id" in attrs]
    assert len(ids) == len(set(ids)), "Duplicate HTML ids"
    for tag, attrs in doc.elements:
        if tag == "a" and attrs.get("href", "").startswith("#"):
            assert attrs["href"][1:] in ids, attrs["href"]
        if tag == "button":
            assert attrs.get("type") == "button"
        if "data-copy" in attrs:
            assert attrs["data-copy"] in doc.code
        assert tag not in {"iframe", "form", "video", "audio"}
        assert not (tag == "script" and "src" in attrs), "No remote scripts"
        assert not (tag == "link" and attrs.get("rel") == "stylesheet"), "CSS must be embedded"
    assert doc.code["dashboard-command"] == 'cd "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" && ./run.sh dashboard'
    assert doc.code["inspect-prompt"] == "What can you see on the table?"
    assert doc.code["cube-prompt"] == "Could you put the green cube in the green square?"
    assert doc.code["orange-prompt"] == "Please put the orange in the open box."
    assert doc.code["reset-prompt"] == "Reset the scene."
    for phrase in (
        "A new chat does not reset the kitchen",
        "UNVERIFIED",
        "not a complete fast-restart recipe",
        "57ecd62dfec3fb361c6db8ee3c096293aec77ca8",
        "fac5332416f688fd5d035e0b0a8088d52ef4d469",
        "PhysX", "OBB", "Qwen", "18790", "8091", "8092",
    ):
        assert phrase in text, phrase
    assert "@media print" in source
    assert ":focus-visible" in source
    assert "prefers-reduced-motion" in source
    # This document must not control the rig, poll a status API, or expose tokens.
    script = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", source, flags=re.S))
    assert not re.search(r"\b(?:fetch|WebSocket|XMLHttpRequest|EventSource|sendBeacon)\b", script)
    assert not re.search(r"(?:#|[?&])(?:token|bootstrapToken)=", source)


def test_installation_commands_match_the_canonical_spark_guide():
    """Keep the full install path in sync; parse commands, never install the demo."""
    import subprocess

    doc = GuideDocument()
    doc.feed(GUIDE.read_text(encoding="utf-8"))
    setup = GUIDE.parents[1] / "DGX_SPARK_SETUP.md"
    commands = re.findall(r"```bash\n(.*?)\n```", setup.read_text(encoding="utf-8"), re.S)
    required = {
        "prerequisites-command": "/scripts/spark_prerequisites.py",
        "packages-command": "sudo apt-get update",
        "install-command": "/scripts/bootstrap.sh",
        "install-check-command": "bash scripts/install.sh",
        "first-launch-command": "scripts/spark_verify.py",
        "desktop-command": "gio launch",
        "stop-check-command": "remaining=$(./run.sh down --dry-run)",
        "public-enable-command": "scripts/spark_public.py\" enable",
        "public-check-command": "scripts/spark_public.py\" check",
        "public-disable-command": "scripts/spark_public.py\" disable",
    }
    for code_id, marker in required.items():
        assert code_id in doc.code, f"Missing installation step: {code_id}"
        canonical = [command for command in commands if marker in command]
        assert len(canonical) == 1, marker
        assert doc.code[code_id].strip() == canonical[0].strip(), code_id
    copy_targets = {attrs["data-copy"] for _, attrs in doc.elements if "data-copy" in attrs}
    assert set(required) <= copy_targets
    text = " ".join(" ".join(doc.text).split())
    for phrase in (
        "PREREQUISITES_OK", "PREPARED", "CHECKED", "READY", "STOPPED",
        "150 GiB", "aarch64", "explicit acceptance", "Python 3.12",
        "Isaac Sim", "OpenClaw", "Qwen", "llama.cpp", "Git LFS", "Chromium",
        "existing installation", "PUBLIC ENABLED", "PUBLIC READY", "PUBLIC STOPPED",
    ):
        assert phrase in text, phrase
    # All shell snippets must parse, including heredocs and command templates.
    for code_id, command in doc.code.items():
        if code_id.endswith("-command"):
            result = subprocess.run(["bash", "-n"], input=command, text=True,
                                    capture_output=True, timeout=10)
            assert result.returncode == 0, f"{code_id}: {result.stderr}"


def test_fresh_install_destination_and_checkout_specific_launch_are_explicit():
    doc = GuideDocument()
    source = GUIDE.read_text(encoding="utf-8")
    doc.feed(source)
    text = " ".join(" ".join(doc.text).split()).lower()
    setup = " ".join((GUIDE.parents[1] / "DGX_SPARK_SETUP.md").read_text().split()).lower()
    for content in (text, setup):
        assert "must not exist" in content, "A fresh destination cannot be an existing empty folder"
        assert "the installer creates it" in content
    for pattern in (r'<section id="start".*?</section>', r'<details id="crash-recovery".*?</details>'):
        region = re.search(pattern, source, re.S)
        assert region and 'href="#desktop-command"' in region[0], "Select the intended checkout, not an ambiguous app-grid name"


@pytest.mark.parametrize("destination, terminal_name", [("spark", "the Spark"), ("laptop", "your laptop")])
def test_copy_button_label_recovers_after_repeated_copy(destination, terminal_name):
    """Execute the page's actual script with only browser boundaries doubled."""
    import json
    import shutil
    import subprocess
    from unittest import SkipTest

    node = shutil.which("node")
    if not node:
        raise SkipTest("Node is required to exercise the HTML copy control")
    source = GUIDE.read_text(encoding="utf-8")
    script = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", source, flags=re.S))
    harness = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const timers = new Map(); let timerId = 0;
const copied = [];
const feedback = {textContent: ''};
const button = {
  textContent: 'Copy command', disabled: false, hidden: true,
  dataset: {copy: 'command', kind: 'command', destination: DESTINATION},
  addEventListener(type, handler) { this[type] = handler; },
};
const source = {textContent: 'exact command\nincluding newline'};
const printButton = {addEventListener() {}};
const context = {
  document: {
    getElementById(id) { return {'copy-feedback': feedback, command: source, 'print-guide': printButton}[id] || null; },
    querySelectorAll(selector) { return selector === '[data-copy]' || selector === '[data-enhance]' ? [button] : []; },
    addEventListener() {},
  },
  navigator: {clipboard: {async writeText(value) { copied.push(value); }}},
  window: {addEventListener() {}, print() {}},
  location: {hash: ''},
  setTimeout(fn, delay) { const id = ++timerId; timers.set(id, {fn, delay}); return id; },
  clearTimeout(id) { timers.delete(id); },
};
vm.runInNewContext(SCRIPT, context);
(async () => {
  await button.click();
  await button.click();
  assert.deepEqual(copied, [source.textContent, source.textContent]);
  assert.equal(feedback.textContent, EXPECTED_FEEDBACK);
  for (const timer of timers.values()) if (timer.delay === 2200) timer.fn();
  assert.equal(button.textContent, 'Copy command', 'Repeated copy leaves a stale Copied label');
  assert.equal(button.disabled, false);
  assert.equal(button.hidden, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(
        [node, "-e", "const SCRIPT = " + json.dumps(script) + ";\n"
         + "const DESTINATION = " + json.dumps(destination) + ";\n"
         + "const EXPECTED_FEEDBACK = "
         + json.dumps(f"Command copied. Paste it into a terminal on {terminal_name}.")
         + ";\n" + harness],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
