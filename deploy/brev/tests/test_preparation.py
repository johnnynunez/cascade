"""Check interruption and resumption without Docker, model downloads or a GPU."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deploy
import prepare_runtime as worker


@pytest.fixture
def local_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "HERE", tmp_path)
    monkeypatch.setattr(deploy, "STOP", tmp_path / "STOP")
    return tmp_path


def active_install(monkeypatch, tmp_path, identity="same"):
    monkeypatch.setattr(deploy, "staged_bundle", lambda: (tmp_path, ["source.py"]))
    monkeypatch.setattr(deploy, "bundle_identity", lambda *args: "same")
    monkeypatch.setattr(deploy, "remote_installation", lambda profile: {})
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {
        "active": True, "plan_identity": identity})
    for name in ("openclaw_archive", "sync", "preflight", "launch_preparation"):
        monkeypatch.setattr(deploy, name, lambda *args, **kwargs: pytest.fail("Repeated preparation"))


def test_existing_job_resumes_without_transfers_or_second_worker(tmp_path, monkeypatch):
    active_install(monkeypatch, tmp_path)
    monkeypatch.setattr(deploy, "wait_preparation", lambda *args: {"status": "PREPARED"})
    monkeypatch.setattr(deploy, "start", lambda *args: {"started": True})
    assert deploy.install({}, tmp_path / "profile.json", {}) == {"started": True}


def test_different_active_bundle_is_preserved(tmp_path, monkeypatch):
    active_install(monkeypatch, tmp_path, "other")
    monkeypatch.setattr(deploy, "wait_preparation", lambda *args: pytest.fail("Wrong job observed"))
    monkeypatch.setattr(deploy, "start", lambda *args: pytest.fail("Wrong source started"))
    with pytest.raises(RuntimeError, match="different bundle"):
        deploy.install({}, tmp_path / "profile.json", {})


def test_observation_deadline_leaves_owned_job_running(local_campaign, monkeypatch):
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {
        "active": True, "plan_identity": "same", "preparation": {"stage": "image"}})
    monkeypatch.setattr(deploy, "cancel_preparation", lambda: pytest.fail("Finite worker cancelled"))
    result = deploy.wait_preparation({"storage": {"root": "/fixture"}}, "same", timeout=0)
    assert result["status"] == "PREPARING"
    assert result["stage"] == "image"


def test_failed_worker_cannot_reuse_an_old_success_receipt(local_campaign, monkeypatch):
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {
        "active": False, "plan_identity": "same", "preparation": {"status": "FAILED"}})
    monkeypatch.setattr(deploy, "remote_installation", lambda profile: {"bundle_identity": "same"})
    with pytest.raises(RuntimeError, match="did not finish"):
        deploy.wait_preparation({"storage": {"root": "/fixture"}}, "same")
    assert json.loads((local_campaign / "proof/PREPARATION.json").read_text())["preparation"]["status"] == "FAILED"


def test_pass_requires_the_matching_installed_bundle(local_campaign, monkeypatch):
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {
        "active": False, "plan_identity": "same", "preparation": {"status": "PASS"}})
    monkeypatch.setattr(deploy, "remote_installation", lambda profile: {"bundle_identity": "other"})
    with pytest.raises(RuntimeError, match="did not finish"):
        deploy.wait_preparation({"storage": {"root": "/fixture"}}, "same")


def test_stop_cancels_only_preparation_before_another_probe(local_campaign, monkeypatch):
    (local_campaign / "STOP").touch()
    cancelled = []
    monkeypatch.setattr(deploy, "cancel_preparation", lambda: cancelled.append(worker.UNIT))
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: pytest.fail("Probe after STOP"))
    with pytest.raises(InterruptedError):
        deploy.wait_preparation({"storage": {"root": "/fixture"}}, "same")
    assert cancelled == [worker.UNIT]


def test_stop_during_remote_probe_still_cancels_preparation(local_campaign, monkeypatch):
    cancelled = []

    def interrupted(profile):
        raise InterruptedError("Matrix stop appeared during the command")

    monkeypatch.setattr(deploy, "remote_preparation", interrupted)
    monkeypatch.setattr(deploy, "cancel_preparation", lambda: cancelled.append(worker.UNIT))
    with pytest.raises(InterruptedError):
        deploy.wait_preparation({"storage": {"root": "/fixture"}}, "same")
    assert cancelled == [worker.UNIT]


def test_cancellation_timeout_is_recorded_without_exposing_output(local_campaign, monkeypatch):
    (local_campaign / "access").mkdir()
    (local_campaign / "access/PASSED.json").write_text(json.dumps({
        "status": "PASS", "target": "paai-demo-l40s", "alias": "paai-demo-l40s-tailnet"}))
    (local_campaign / "proof").mkdir()
    (local_campaign / "proof/PREPARATION_OWNER.json").write_text(json.dumps({
        "target": "paai-demo-l40s", "root": "/fixture"}))

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("private fixture command", 40, output="private fixture output")

    monkeypatch.setattr(deploy.subprocess, "run", timed_out)
    result = deploy.cancel_preparation()
    assert result["stopped"] is False
    assert result["error_type"] == "TimeoutExpired"
    assert "private fixture" not in json.dumps(result)


def test_initial_stop_cancels_prior_preparation_without_new_access(local_campaign, monkeypatch):
    (local_campaign / "STOP").touch()
    cancelled = []
    monkeypatch.setattr(sys, "argv", ["deploy.py", "install"])
    monkeypatch.setattr(deploy, "require_access", lambda: pytest.fail("New access after STOP"))
    monkeypatch.setattr(deploy, "cancel_preparation", lambda: cancelled.append(worker.UNIT))
    assert deploy.main() == 0
    assert cancelled == [worker.UNIT]


def test_start_refuses_active_worker_before_any_preflight_write(monkeypatch):
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {"active": True})
    monkeypatch.setattr(deploy, "remote_installation", lambda profile: pytest.fail("Concurrent start"))
    monkeypatch.setattr(deploy, "preflight", lambda *args, **kwargs: pytest.fail("Active helper replaced"))
    with pytest.raises(RuntimeError, match="Preparation is active"):
        deploy.start({}, Path("profile.json"))


def test_start_refuses_incomplete_source_receipt(monkeypatch):
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {"active": False})
    monkeypatch.setattr(deploy, "remote_installation", lambda profile: {"status": "UPDATING"})
    monkeypatch.setattr(deploy, "preflight", lambda *args, **kwargs: pytest.fail("Incomplete source started"))
    with pytest.raises(RuntimeError, match="not prepared"):
        deploy.start({}, Path("profile.json"))


def test_preflight_does_not_replace_active_worker_inputs(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {"active": True})
    monkeypatch.setattr(deploy, "remote", lambda *args, **kwargs: pytest.fail("Remote mutation"))
    monkeypatch.setattr(deploy, "sync", lambda *args, **kwargs: pytest.fail("Active helper replaced"))
    with pytest.raises(RuntimeError, match="control files must remain unchanged"):
        deploy.preflight(profile)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    root = tmp_path / "deployment-root"
    (root / "deployment").mkdir(parents=True)
    plan = {"bundle_identity": "a" * 64, "image_identity": "b" * 64,
            "source_files": 3, "source_revision": "c" * 40, "storage_mount": str(tmp_path)}
    monkeypatch.setattr(worker, "validate_plan", lambda *args: None)
    return root, plan


def test_worker_orders_models_then_image_then_install_receipt(preparation, monkeypatch):
    root, plan = preparation
    commands = []

    def run(command, path, timeout):
        assert not (root / "deployment/installed.json").exists()
        commands.append(command)
        path.write_text("private fixture log")

    images = iter([None, "sha256:" + "d" * 64])
    monkeypatch.setattr(worker, "run_logged", run)
    monkeypatch.setattr(worker, "image_id", lambda: next(images))
    result = worker.prepare(root, plan)
    assert commands[0][1].endswith("fetch_models.py")
    assert commands[1][-1] == "build"
    assert result["status"] == "PASS"
    receipt = json.loads((root / "deployment/installed.json").read_text())
    assert receipt["bundle_identity"] == plan["bundle_identity"]
    assert receipt["runtime_started"] is False
    assert (root / "deployment/installed.json").stat().st_mode & 0o777 == 0o600


def test_verified_image_is_reused_for_source_only_update(preparation, monkeypatch):
    root, plan = preparation
    image = "sha256:" + "d" * 64
    worker.atomic(root / "deployment/image-build.json", {
        "image_identity": plan["image_identity"], "image_id": image})
    commands = []
    monkeypatch.setattr(worker, "image_id", lambda: image)
    monkeypatch.setattr(worker, "run_logged", lambda command, path, timeout: commands.append(command))
    result = worker.prepare(root, plan)
    assert len(commands) == 1
    assert commands[0][1].endswith("fetch_models.py")
    assert result["image_reused"] is True


def test_missing_or_replaced_image_invalidates_its_receipt(preparation, monkeypatch):
    root, plan = preparation
    worker.atomic(root / "deployment/image-build.json", {
        "image_identity": plan["image_identity"], "image_id": "sha256:" + "d" * 64})
    commands = []
    monkeypatch.setattr(worker, "image_id", lambda: "sha256:" + "e" * 64)
    monkeypatch.setattr(worker, "run_logged", lambda command, path, timeout: commands.append(command))
    result = worker.prepare(root, plan)
    assert len(commands) == 2
    assert result["image_reused"] is False


def test_model_failure_never_builds_or_certifies_installation(preparation, monkeypatch):
    root, plan = preparation
    monkeypatch.setattr(worker, "image_id", lambda: pytest.fail("Image operation after model failure"))

    def fail(*args):
        raise RuntimeError("Fixture model failure")

    monkeypatch.setattr(worker, "run_logged", fail)
    with pytest.raises(RuntimeError, match="Fixture model"):
        worker.prepare(root, plan)
    assert not (root / "deployment/installed.json").exists()
    record = json.loads((root / "deployment/preparation.json").read_text())
    assert record["status"] == "FAILED"
    assert record["stage"] == "models"


def test_worker_lock_preserves_an_existing_job(preparation, monkeypatch):
    root, plan = preparation
    monkeypatch.setattr(worker, "run_logged", lambda *args: pytest.fail("Concurrent worker"))
    with (root / "deployment/.prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another preparation"):
            worker.prepare(root, plan)
    assert not (root / "deployment/preparation.json").exists()


def test_command_timeout_reaps_child_and_preserves_its_log(tmp_path):
    log = tmp_path / "finite-command.log"
    code = "import os,time; print(os.getpid(),flush=True); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run_logged([sys.executable, "-c", code], log, timeout=0.2)
    pid = int(log.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert log.stat().st_mode & 0o777 == 0o600


def test_changed_installation_preserves_original_without_transfers(local_campaign, monkeypatch):
    previous = {"status": "PREPARED", "bundle_identity": "previous"}
    monkeypatch.setattr(deploy, "staged_bundle", lambda: (local_campaign, ["source.py"]))
    monkeypatch.setattr(deploy, "bundle_identity", lambda *args: "replacement")
    monkeypatch.setattr(deploy, "remote_installation", lambda profile: previous)
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {"active": False})
    for name in ("sync", "compose", "preflight", "openclaw_archive", "remote"):
        monkeypatch.setattr(deploy, name, lambda *args, **kwargs: pytest.fail("Changed installation was mutated"))
    with pytest.raises(ValueError, match="installed bundle differs"):
        deploy.install({}, local_campaign / "profile.json", {})
    assert previous == {"status": "PREPARED", "bundle_identity": "previous"}
    assert not list(local_campaign.iterdir())
