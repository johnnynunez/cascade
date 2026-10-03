"""Actual per-robot orchestrators with labelled mock inference and kinematics."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading

import pytest
import yaml

from cascade.apps import fleet as app
from cascade.config import Cfg
from cascade.skills.mobile_runtime import MobileSkillRuntime


def config(tmp_path, count=2, **options):
    entries = [{"profile": "microduck_conversation_mock", "mock_id": f"duck{i:02}"} for i in range(count)]
    path = tmp_path / "fleet.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "robots": entries, **options}))
    return path


def prepare(tmp_path, count=2, **options):
    plan = app.load_fleet(config(tmp_path, count, **options), task="inspect, then move briefly")
    fleet = app.build_fleet(plan, tmp_path / "run")
    return app.FleetAgents(fleet, plan["tasks"], Cfg({"type": "mock"}),
        deadline_s=plan["deadline_s"], max_workers=plan["max_workers"], max_steps=plan["max_steps"])


@pytest.mark.parametrize("count", [2, 12])
def test_orchestrators_reach_motion_barrier_with_private_llm_histories(tmp_path, monkeypatch, count):
    barrier = threading.Barrier(count)
    original = MobileSkillRuntime.execute
    entered = []

    def execute(runtime, name, args=None):
        if name == "walk_velocity":
            entered.append(runtime.base_rig.primary.raw.metadata["robot_id"])
            barrier.wait(5)
        return original(runtime, name, args)

    monkeypatch.setattr(MobileSkillRuntime, "execute", execute)
    agents = prepare(tmp_path, count)
    try:
        result = agents.run()
        assert result["agent"] == "AgentOrchestrator" and result["llm"] == "scripted_mock"
        assert set(entered) == set(agents.tasks) and len(entered) == count
        assert not result["physical_fleet_admission"] and not result["pending_agents"]
        for robot_id, row in result["robots"].items():
            assert row["state"] == "completed"
            assert row["report"]["path"] == "llm" and not row["report"]["success"]
            motions = [t for t in row["report"]["tool_log"] if t["tool"] == "locomotion.walk_velocity"]
            assert len(motions) == 1 and motions[0]["result"]["execution_ok"]
            assert motions[0]["result"]["postcondition"]["status"] == "unverified"
            client = agents.clients[robot_id]
            assert len(client.requests) == 3
            assert robot_id in client.requests[0]["messages"][0]["content"]
            assert all(other not in client.requests[0]["messages"][0]["content"]
                       for other in agents.tasks if other != robot_id)
        assert len({id(client) for client in agents.clients.values()}) == count
    finally:
        assert agents.close()["complete"]


def test_individual_cancel_rejects_delayed_llm_motion_without_cancelling_peer(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = app._mock_client

    def client(runtime):
        result = original(runtime)
        if runtime.cfg.robot_id == "duck00":
            chat = result.chat
            def delayed(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return chat(*args, **kwargs)
            result.chat = delayed
        return result

    monkeypatch.setattr(app, "_mock_client", client)
    agents = prepare(tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agents.run)
            try:
                assert entered.wait(3)
                assert agents.stop("duck00")["physical_stop_verified"] is False
            finally:
                release.set()
            report = future.result(5)
        assert report["robots"]["duck00"]["state"] == "cancelled"
        assert report["robots"]["duck01"]["state"] == "completed"
        assert not (tmp_path / "run/robots/000/trace.jsonl").exists()
        assert (tmp_path / "run/robots/001/trace.jsonl").is_file()
    finally:
        assert agents.close()["complete"]


def test_deadline_fences_a_running_llm_and_does_not_claim_http_cancelled(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = app._mock_client

    def client(runtime):
        result = original(runtime)
        chat = result.chat
        def delayed(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return chat(*args, **kwargs)
        result.chat = delayed
        return result

    monkeypatch.setattr(app, "_mock_client", client)
    agents = prepare(tmp_path, 2, deadline_s=1., max_workers=1)
    try:
        result = agents.run()
        assert entered.is_set() and set(result["pending_agents"]) == {"duck00", "duck01"}
        assert all(runtime.stopped for runtime in agents.fleet.robots.values())
        assert set(agents.clients) == {"duck00"}  # queued robot never starts inference
        assert not (tmp_path / "run/robots/000/trace.jsonl").exists()
    finally:
        release.set()
        assert agents.close()["complete"]
    assert all(row["state"] == "cancelled" for row in agents.snapshot()["robots"].values())
    assert not (tmp_path / "run/robots/000/trace.jsonl").exists()


@pytest.mark.parametrize("entries", [
    [{"profile": "microduck_conversation_mock"}] * 2,
    [{"profile": "microduck_conversation_native", "mock_id": "renamed"}],
])
def test_invalid_composition_never_opens_the_first_runtime(tmp_path, monkeypatch, entries):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "robots": entries}))
    monkeypatch.setattr(app, "build_robot_runtime", lambda *_: pytest.fail("opened runtime before whole-fleet validation"))
    with pytest.raises(ValueError):
        app.main(["--fleet", str(path), "--task", "move", "--run-dir", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_mock_diagnostic_cli_records_twelve_distinct_agents(tmp_path):
    directory = tmp_path / "cli"
    assert app.main(["--fleet", "microduck_mock12", "--task", "inspect and move", "--run-dir", str(directory)]) == 0
    report = json.loads((directory / "report.json").read_text())
    catalog = json.loads((directory / "catalog.json").read_text())
    assert len(report["robots"]) == len(catalog) == 12
    assert report["shutdown"]["complete"] and not report["physical_fleet_admission"]
    assert report["software_complete"] and not report["task_success"]
    assert report["llm"] == "scripted_mock"
    assert all(not row["report"]["success"] for row in report["robots"].values())


@pytest.mark.skipif(os.name != "posix", reason="owned subprocess SIGTERM test")
def test_cli_real_sigterm_preserves_report_and_closes_owned_mock_robots(tmp_path):
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "signal"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="-1")
    process = subprocess.Popen([sys.executable, "-m", "cascade.apps.fleet", "--fleet", "microduck_mock12",
        "--task", "inspect and move", "--run-dir", str(directory)], cwd=root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0], "owned CLI did not announce readiness"
        assert process.stdout.readline().startswith("Fleet ready:")
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == 128 + signal.SIGTERM, stderr
        report = json.loads((directory / "report.json").read_text())
        assert report["signal"] == signal.SIGTERM and report["shutdown"]["complete"]
        assert not report["physical_fleet_admission"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_native_profiles_keep_exact_pins_and_reject_duplicate_endpoints_before_io(tmp_path, monkeypatch):
    import socket
    from mobile_support_fixture import support_contract
    directory = tmp_path / "profiles"
    for folder in ("robots", "bases", "llm"):
        (directory / folder).mkdir(parents=True)
    (directory / "demo.yaml").write_text("memory: {}\n")
    (directory / "llm/mock.yaml").write_text("type: mock\n")
    for key in list(os.environ):
        if key.startswith("CASCADE_MICRODUCK_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("preflight dialed a device"))
    base = yaml.safe_load((app.CONFIG_DIR / "bases/microduck_isaac.yaml").read_text())
    base.update(device="cuda:0", asset_sha256="a" * 64, policy_sha256="b" * 64,
                model_identity_sha256="e" * 64, support_contract=support_contract())
    entries = []
    for index, name in enumerate(("left", "right")):
        profile = {**base, "robot_id": f"duck:{name}", "bridge_port": 41000 + index}
        (directory / f"bases/{name}.yaml").write_text(yaml.safe_dump(profile))
        (directory / f"robots/{name}.yaml").write_text(yaml.safe_dump({"version": 1,
            "robot_id": f"duck:{name}", "domains": {"locomotion": {"kind": "locomotion", "bases": [name]}}}))
        entries.append({"profile": name, "config_dir": "profiles"})
    path = tmp_path / "native.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "robots": entries}))
    plan = app.load_fleet(path, task="inspect only")
    for index, cfg in enumerate(plan["configs"]):
        profile = cfg.domains.as_dict()["locomotion"]["resolved"]["bases"][0]
        assert profile["robot_id"] == cfg.robot_id
        assert profile["bridge_port"] == 41000 + index
        assert all(profile[key] == base[key] for key in
                   ("asset_sha256", "policy_sha256", "model_identity_sha256", "support_contract"))
    plan["configs"][1]._data["domains"]["locomotion"]["resolved"]["bases"][0]["bridge_port"] = 41000
    monkeypatch.setattr(app, "build_robot_runtime", lambda *_a, **_k: pytest.fail("opened first robot before ownership validation"))
    with pytest.raises(ValueError, match="incompatible writers"):
        app.build_fleet(plan, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_partial_build_failure_retains_cleanup_errors_and_closes_other_owners(tmp_path, monkeypatch):
    plan = app.load_fleet(config(tmp_path, 3), task="inspect")
    original, built, closed = app.build_robot_runtime, [], []

    def build(cfg, directory):
        if len(built) == 2:
            raise RuntimeError("third robot construction failed")
        runtime, owner = original(cfg, directory)
        close = runtime.close
        def record_close():
            closed.append(cfg.robot_id)
            return close()
        runtime.close = record_close
        if not built:
            runtime.queue_shutdown = lambda: (_ for _ in ()).throw(RuntimeError("queue failure"))
        built.append(runtime)
        return runtime, owner

    monkeypatch.setattr(app, "build_robot_runtime", build)
    with pytest.raises(RuntimeError, match="third robot") as caught:
        app.build_fleet(plan, tmp_path / "run")
    assert set(closed) == {"duck00", "duck01"}
    assert caught.value.fleet_teardown["ok"] is False
    assert any(row.get("error") == "queue failure" for row in caught.value.fleet_teardown["stages"])


def test_client_close_occurs_after_chat_and_is_reported_on_failure(tmp_path, monkeypatch):
    original = app._mock_client
    closes = []

    def client(runtime):
        result = original(runtime)
        def close():
            closes.append((runtime.cfg.robot_id, len(result.requests)))
            raise RuntimeError("close failed")
        result.close = close
        return result

    monkeypatch.setattr(app, "_mock_client", client)
    agents = prepare(tmp_path, 1)
    try:
        report = agents.run()
        row = report["robots"]["duck00"]
        assert closes == [("duck00", 3)] and row["state"] == "failed"
        assert "close failed" in json.dumps(row["client_close_error"])
        assert row["report"]["tool_log"]  # cleanup cannot erase the task's evidence
        assert report["unverified"]["duck00"]
    finally:
        closure = agents.close()
        assert not closure["complete"] and not closure["llm_clients"]["duck00"]["ok"]


def test_non_mock_task_failure_is_nonzero_even_when_episode_and_cleanup_finish(tmp_path, monkeypatch):
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    llm_path = tmp_path / "llm.yaml"
    llm_path.write_text("type: openai_compat\nmodel: injected-test-client\n")
    monkeypatch.setattr(app, "make_llm", lambda _cfg: MockLLM([
        LLMResponse(text="inspect"),
        LLMResponse(tool_calls=[ToolCall("list_resources", {})]),
        LLMResponse(tool_calls=[ToolCall("task_done", {"success": False, "summary": "cannot complete task"})]),
    ]))
    directory = tmp_path / "failed-task"
    assert app.main(["--fleet", str(config(tmp_path, 1)), "--llm", str(llm_path),
        "--task", "move", "--run-dir", str(directory)]) == 1
    report = json.loads((directory / "report.json").read_text())
    assert report["software_complete"] and report["shutdown"]["complete"]
    assert not report["task_success"] and not report["physical_fleet_admission"]


def test_worker_baseexception_stops_its_robot_and_reports_without_waiting_for_deadline(tmp_path, monkeypatch):
    original = app._mock_client
    def client(runtime):
        result = original(runtime)
        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt("inference interrupted")
        result.chat = interrupted
        return result
    monkeypatch.setattr(app, "_mock_client", client)
    agents = prepare(tmp_path, 1)
    try:
        report = agents.run()
        assert report["robots"]["duck00"]["state"] == "failed" and not report["pending_agents"]
        assert "KeyboardInterrupt" in report["robots"]["duck00"]["error"]
        assert agents.fleet.robots["duck00"].stopped
    finally:
        assert agents.close()["complete"]
