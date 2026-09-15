"""Bind container MCP traces to the configured persistent runs directory."""
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("container_demo_proof", ROOT / "scripts/demo_proof.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


@pytest.fixture
def owned_process(tmp_path, monkeypatch):
    from cascade.apps import process_owner

    repo = tmp_path / "source"
    repo.mkdir()
    persistent = tmp_path / "data/runs"
    persistent.mkdir(parents=True)
    (repo / "runs").symlink_to(persistent, target_is_directory=True)
    run = persistent / "mcp_42_current"
    run.mkdir()
    owner = {"repo": str(repo)}
    record = {"pid": 42, "instance_id": "current", "registered_at": 20,
              "run_dir": str(repo / "runs" / run.name)}
    monkeypatch.setattr(process_owner, "live_records", lambda *args, **kwargs: [record])
    return owner, record, repo


def test_bound_world_accepts_owned_persistent_runs_symlink(owned_process):
    owner, record, repo = owned_process
    assert proof._bound_world(repo / "state", owner, set(), 10, expected=record) == record


def test_bound_world_rejects_trace_outside_configured_runs(owned_process, tmp_path):
    owner, record, repo = owned_process
    foreign = tmp_path / "foreign/mcp_42_current"
    foreign.mkdir(parents=True)
    record["run_dir"] = str(foreign)
    with pytest.raises(proof.ProofError, match="invalid runtime trace directory"):
        proof._bound_world(repo / "state", owner, set(), 10, expected=record)
