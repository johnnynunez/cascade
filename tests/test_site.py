"""The GitHub Pages site (site/) and its deploy workflow stay consistent with the repo."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _checker():
    spec = importlib.util.spec_from_file_location("check_site", REPO / "scripts" / "check_site.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_site_links_anchors_images_and_budget():
    # Every local file, #anchor and repo link the page names exists; nothing loads from a CDN.
    assert _checker().check() == []


def test_checker_finds_broken_anchors_missing_repo_paths_and_a_cdn(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        '<a href="#nowhere">x</a>'
        '<a href="https://github.com/johnnynunez/cascade/blob/main/docs/NOT_A_DOC.md">y</a>'
        '<a href="https://github.com/johnnynunez/cascade/blob/main/README.md#no-such-heading">z</a>'
        '<script src="https://cdn.example.com/lib.js"></script>'
        '<img src="missing.png">', encoding="utf-8")
    problems = _checker().check(site=site, repo=REPO)
    joined = "\n".join(problems)
    assert "#nowhere" in joined
    assert "NOT_A_DOC.md" in joined
    assert "#no-such-heading" in joined
    assert "cdn.example.com" in joined
    assert "missing.png" in joined and "no alt" in joined


def test_pages_workflow_deploys_site_from_main_only():
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "pages.yml").read_text())
    on = wf.get("on", wf.get(True))
    assert on["push"]["branches"] == ["main"]
    assert "site/**" in on["push"]["paths"]
    assert "workflow_dispatch" in on
    assert wf["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert wf["concurrency"]["group"] == "pages"
    job = wf["jobs"]["deploy"]
    assert job["environment"]["name"] == "github-pages"
    uses = [s.get("uses", "") for s in job["steps"]]
    assert any(u.startswith("actions/upload-pages-artifact@") for u in uses)
    assert any(u.startswith("actions/deploy-pages@") for u in uses)
    upload = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-pages-artifact@"))
    assert upload["with"]["path"] == "site"
