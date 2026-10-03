"""Ordinary MCP/composition wiring, with kinematic evidence kept unverified."""
import shutil

import pytest
import yaml

from cascade.config import load_profile, load_robot_config
from test_mobile_mcp import MobileClient, REPO


@pytest.fixture
def distance_config(tmp_path, monkeypatch):
    import cascade.config as config

    directory = tmp_path / "configs"
    shutil.copytree(REPO / "configs", directory)
    candidate = load_profile("bases", "microduck_distance_candidate").as_dict()
    # Explicit test-only backend. The shipped physical candidate stays unpinned.
    profile = {"extends": "microduck_mock", "capabilities": candidate["capabilities"],
               "safety": candidate["safety"], "distance_control": candidate["distance_control"]}
    (directory / "bases/distance_fixture.yaml").write_text(yaml.safe_dump(profile))
    (directory / "robots/distance_fixture.yaml").write_text(yaml.safe_dump({
        "version": 1, "robot_id": "microduck-mock", "domains": {
            "locomotion": {"kind": "locomotion", "bases": ["distance_fixture"]}}}))
    monkeypatch.setattr(config, "CONFIG_DIR", directory)
    return directory


def test_stdio_distance_uses_opt_in_profile_and_never_promotes_mock_motion(distance_config, tmp_path):
    client = MobileClient(tmp_path / "mcp", config_dir=distance_config,
                          CASCADE_BASE="distance_fixture")
    try:
        tools = {s["name"] for s in client.request("tools/list")["result"]["tools"]}
        assert "walk_distance" in tools and "turn" in tools
        assert "walk_velocity" not in tools
        result = client.call("walk_distance", {"distance_m": .03})
        assert result["execution_ok"] is True, result
        assert result["ok"] is False
        assert result["postcondition"]["status"] == "unverified"
        assert result["requested_distance_m"] == .03
        assert abs(result["measured_distance_m"] - .03) <= .005
        assert client.call("verify_last_action")["recent"][-1]["status"] == "unverified"
        refused = client.call("walk_distance", {"distance_m": .100001})
        assert not refused["execution_ok"] and not refused["ok"]
    finally:
        client.close()
    assert client.proc.returncode == 0


def test_composed_distance_keeps_writer_claim_and_task_debt(distance_config, tmp_path):
    from cascade.apps.robot_runtime import build_robot_runtime, robot_tool_descriptors

    cfg = load_robot_config("distance_fixture")
    descriptor = robot_tool_descriptors(cfg)["locomotion.walk_distance"]
    assert descriptor.effect == "motion" and descriptor.writes
    runtime, _ = build_robot_runtime(cfg, tmp_path / "composed")
    try:
        result = runtime.execute("locomotion.walk_distance", {"distance_m": .03})
        assert result["execution_ok"] is True and result["ok"] is False, result
        assert result["postcondition"]["status"] == "unverified"
        done = runtime.execute("task_done", {"success": True, "summary": "requested success"})
        assert done["success"] is False
        assert "locomotion.walk_distance" in done["unverified"]
    finally:
        assert runtime.close()["ok"]


@pytest.mark.parametrize("name", ["microduck_distance_candidate", "microduck_distance_native_slow"])
def test_native_distance_requires_new_independent_identity_before_transport(name, tmp_path, monkeypatch):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.config import load_demo_config
    from cascade.sim.bridge_client import BridgeClient

    profile = load_profile("bases", name).as_dict()
    assert profile["distance_control"]["max_distance_m"] == .1
    assert profile["model_identity_sha256"] is None
    assert profile["support_contract"] is None
    assert profile["admission"] == ("pending_geometric_locomotion_admission"
        if name == "microduck_distance_candidate" else "pending_slow_native_geometric_validation")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unpinned composition opened a transport")
    monkeypatch.setattr(BridgeClient, "__init__", forbidden)
    cfg = load_demo_config(base=name, llm="mock")
    with pytest.raises(ValueError, match="explicit .* required"):
        build_mobile_runtime(cfg, tmp_path / "refused")
