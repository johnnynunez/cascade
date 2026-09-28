"""Always-on services are private, bounded and owned by one Spark checkout."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/spark_public.py"
spec = importlib.util.spec_from_file_location("spark_public", SCRIPT)
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


def test_user_units_restore_install_home_and_supervise_only_their_stack(tmp_path):
    settings = {"runtime_home": str(tmp_path / "private home")}
    units = public.units(tmp_path, settings)
    assert set(units) == {"paai-spark-demo", "paai-spark-visitor", "paai-spark-ngrok"}
    for contents in units.values():
        assert "Restart=always" in contents and "KillMode=mixed" in contents
        assert "WantedBy=default.target" in contents
        assert 'Environment="HOME=' + settings["runtime_home"] + '"' in contents
        assert "sudo" not in contents and "8090" not in contents
        assert "gateway.token" not in contents
    assert "8093" in units["paai-spark-visitor"]
    assert "--auth-file" in units["paai-spark-visitor"]


def test_existing_unrelated_service_is_never_adopted(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "unit_directory", lambda: tmp_path)
    path = tmp_path / "paai-spark-demo.service"
    path.write_text("[Service]\nExecStart=/other/application\n")
    monkeypatch.setattr(public, "run", lambda *a, **kw: f"LoadState=loaded\nActiveState=active\nFragmentPath={path}\nDropInPaths=\n")
    with pytest.raises(ValueError, match="another installation"):
        public.owned_units(tmp_path / "checkout")
    assert path.read_text() == "[Service]\nExecStart=/other/application\n"


def test_service_overrides_require_owner_review(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "run", lambda *a, **kw: "LoadState=loaded\nDropInPaths=/another/unit.conf\n")
    with pytest.raises(ValueError, match="overrides"):
        public.owned_units(tmp_path)


def test_secrets_require_private_owner_files_and_never_enter_failed_command_output(tmp_path, monkeypatch):
    path = tmp_path / "token"
    path.write_text("private-token-fixture")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        public.private_file(path)
    path.chmod(0o600)
    assert public.private_file(path) == path
    monkeypatch.setattr(public.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="private-token-fixture", stderr="private-token-fixture"))
    with pytest.raises(RuntimeError) as failure:
        public.run(["ngrok", "config", "check"])
    assert "private-token-fixture" not in str(failure.value)


def test_disable_checks_ownership_then_stops_only_new_units(tmp_path, monkeypatch):
    owned = {name: {"installed": True, "active": True} for name in public.SERVICES}
    calls = []
    monkeypatch.setattr(public, "owned_units", lambda repo: owned)
    monkeypatch.setattr(public, "run", lambda command, **kw: calls.append(command))
    public.disable(tmp_path)
    assert calls == [["systemctl", "--user", "disable", "--now", "paai-spark-ngrok.service", "paai-spark-visitor.service", "paai-spark-demo.service"]]


def test_systemd_argument_escaping_never_interprets_percent_or_dollars():
    assert public.unit_arg('/tmp/a $name 100%') == '"/tmp/a $$name 100%%"'
    with pytest.raises(ValueError):
        public.unit_arg("/tmp/name\nExecStart=/other")
