"""The license URL shown to the operator is the one recorded with their consent."""
from __future__ import annotations

import json
from pathlib import Path
import re

from test_spark_install import support_module

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/DGX_SPARK_SETUP.md"


def guide_license_url():
    match = re.search(r"\[NVIDIA Isaac Sim / Omniverse\s+licenses\]\((https://[^)]+)\)", GUIDE.read_text())
    assert match, "the setup guide must link the license page before the install command"
    return match[1]


def test_guide_installer_and_consent_receipt_name_the_same_license_page():
    support = support_module()
    url = guide_license_url()
    assert support.EULA_URL == url
    assert url in (ROOT / "scripts/install.sh").read_text()
    for legacy in support.LEGACY_EULA_URLS:
        assert legacy not in (ROOT / "scripts/install.sh").read_text()
        assert legacy not in GUIDE.read_text()


def _receipt(repo: Path, url: str, accepted: bool = True):
    state = repo / "runs/.install"
    state.mkdir(parents=True, exist_ok=True)
    (state / "install.json").write_text(json.dumps({
        "repo": str(repo.resolve()), "eula_accepted": accepted, "eula_url": url}))


def test_existing_consent_under_the_retired_link_still_launches(tmp_path):
    support = support_module()
    for url in (support.EULA_URL, *support.LEGACY_EULA_URLS):
        _receipt(tmp_path, url)
        assert support.eula_accepted(tmp_path)
        assert support.recorded_eula_url(tmp_path) == url


def test_unknown_or_refused_license_url_grants_nothing(tmp_path):
    support = support_module()
    _receipt(tmp_path, "https://example.invalid/eula")
    assert not support.eula_accepted(tmp_path)
    _receipt(tmp_path, support.EULA_URL, accepted=False)
    assert not support.eula_accepted(tmp_path)
    other = tmp_path / "moved"
    other.mkdir()
    (other / "runs/.install").mkdir(parents=True)
    (other / "runs/.install/install.json").write_text(json.dumps({
        "repo": str(tmp_path.resolve()), "eula_accepted": True, "eula_url": support.EULA_URL}))
    assert not support.eula_accepted(other)


def test_repair_keeps_the_url_the_operator_agreed_to_and_new_consent_uses_the_current_page(tmp_path):
    from test_spark_install import boundary_env
    support = support_module()
    env, _ = boundary_env(tmp_path)
    repo = Path(env["BOUNDARY_SOURCE"])
    legacy = support.LEGACY_EULA_URLS[0]
    _receipt(repo, legacy)
    support.record_install(repo, "spark", "qwen", "main")          # desktop repair, no new consent
    assert json.loads((repo / "runs/.install/install.json").read_text())["eula_url"] == legacy
    support.record_install(repo, "spark", "qwen", "main", accept_eula=True)
    assert json.loads((repo / "runs/.install/install.json").read_text())["eula_url"] == support.EULA_URL
