"""Every brain serve script downloads its weights into <repo>/models/<name>/ by default
(MODEL_ROOT = the checkout). Those directories hold tens of GB plus `.part` files and the
fetch lock, so each one must be gitignored, or the live checkout shows them as untracked."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = re.compile(r'^MODEL_DIR="\$MODEL_ROOT/(models/[^"/$]+)"', re.MULTILINE)


def _model_dirs() -> list[tuple[str, str]]:
    found = []
    for script in sorted((REPO / "scripts").glob("serve_*.sh")):
        for rel in MODEL_DIR.findall(script.read_text()):
            found.append((script.name, rel))
    return found


def test_serve_scripts_name_their_model_dirs():
    names = {script for script, _ in _model_dirs()}
    assert {"serve_nemotron_bolt_llamacpp.sh", "serve_qwen_llamacpp.sh"} <= names


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs a git checkout")
@pytest.mark.parametrize(("script", "rel"), _model_dirs())
def test_model_dir_is_gitignored(script, rel):
    for leaf in ("weights.gguf", "weights.gguf.part", ".fetch-models.lock"):
        path = f"{rel}/{leaf}"
        res = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "--no-index", path], check=False)
        assert res.returncode == 0, f"{script} downloads into {rel}/ but {path} is not gitignored"
