"""CLI wiring succeeds without treating a static mock image as moved physics.

The minimal-install CI job separately requires all nine optional stacks absent.
These checks need only base + dev + kinematics and exercise the ordinary CLI,
including the failed fast path followed by the default mock LLM's completion.
"""
from dataclasses import asdict
import json
import time

import pytest

from cascade.apps import demo
from cascade.control.mock_arm import MockArm


@pytest.fixture
def cli_episode(monkeypatch, tmp_path):
    # Keep ordinary persistent paths and optional service discovery out of this
    # local mock check. FK/IK, safety limits and independent checking are intact.
    monkeypatch.setattr(demo, "PACKAGE_ROOT", tmp_path / "package")
    for name in ("CASCADE_ROBOT", "CASCADE_BASE", "CASCADE_LLM", "CASCADE_ARMS",
                 "CASCADE_ARM", "CASCADE_CAMERA", "CASCADE_CAMERAS"):
        monkeypatch.delenv(name, raising=False)
    load_config = demo.load_demo_config

    def local_config(**kwargs):
        cfg = load_config(**kwargs)
        cfg._data["grasp"]["backend"] = "obb"
        cfg._data["occupancy"]["enabled"] = False
        for arm in cfg._data["arms"]:
            arm["resolved"]["grasp"]["backend"] = "obb"
            arm["resolved"]["occupancy"]["enabled"] = False
        return cfg

    monkeypatch.setattr(demo, "load_demo_config", local_config)
    reports, closed = [], []
    build_runtime = demo.build_runtime
    original_print = demo._print_report
    original_close = demo.shutdown_runtime

    def build_mock(*args, **kwargs):
        runtime, arm = build_runtime(*args, **kwargs)
        try:
            assert isinstance(arm, MockArm) and arm.n_joints == 5
            # Existing mock fixture hook: emulate jaws closing on an object so
            # the whole grasp/place/home command path runs. The camera stays
            # static; relocation must remain unverified or refuted by the
            # independent checker, whichever belief snapshot it observes.
            arm.object_stop_frac = 0.5
            deadline = time.monotonic() + 5.0
            while runtime.beliefs.find("red object") is None:
                assert time.monotonic() < deadline, "mock detector did not publish an object"
                time.sleep(0.01)
        except BaseException:
            original_close(runtime, arm)
            raise
        return runtime, arm

    def observe_report(report):
        reports.append(report)
        original_print(report)

    def observe_close(runtime, arm):
        receipt = original_close(runtime, arm)
        closed.append({"receipt": receipt, "held_object": runtime.held_object,
                       "effects": runtime.task_effects(),
                       "joint_commands": len(arm.commands)})
        return receipt

    monkeypatch.setattr(demo, "_print_report", observe_report)
    monkeypatch.setattr(demo, "build_runtime", build_mock)
    monkeypatch.setattr(demo, "shutdown_runtime", observe_close)

    def run(task):
        directory = tmp_path / "run"
        code = demo.main(["--arm", "so101_mock", "--camera", "mock_small",
                          "--llm", "mock", "--task", task, "--no-view",
                          "--no-serve", "--run-dir", str(directory)])
        assert len(reports) == len(closed) == 1
        report, closure = reports[0], closed[0]
        records = [json.loads(line) for line in
                   (directory / "trace.jsonl").read_text().splitlines()]
        # The result is retained even if a later assertion fails. No result,
        # shutdown failure, signal exit or unrelated exit1 qualifies as a smoke.
        (directory / "smoke-receipt.json").write_text(json.dumps({
            "exit_code": code, "report": asdict(report), "closure": closure,
        }, indent=2) + "\n")
        assert closure["receipt"]["ok"] is True
        assert closure["receipt"]["complete"] is True
        assert closure["held_object"] is None
        assert closure["joint_commands"] > 0  # also covers the ordinary shutdown park
        assert report.task == task
        assert [r["step"] for r in records] == list(range(len(records)))
        assert (directory / "summary.txt").is_file()
        assert list((directory / "keyframes").glob("*.jpg"))
        return code, report, records, closure["effects"]

    return run


def test_minimal_mock_pick_runs_but_cannot_claim_physical_success(cli_episode):
    code, report, records, ledger = cli_episode("pick and place the red object")
    assert code == 1
    assert report.success is False and report.unverified
    assert [row["skill"] for row in records] == [
        "pick_and_place", "get_observation", "task_done"]
    assert [row["tier"] for row in records] == ["reflex", "llm", "llm"]
    pick, observed, done = [row["result"] for row in records]
    assert pick["picked"] == "red object"
    assert len(pick["placed_at"]) == 3
    assert pick["grip_verified"] is True
    assert pick["grasp_attempts"] == pick["place_attempts"] == 1
    assert pick["return_home"] == {"attempted": True, "ok": True, "at": "home"}
    assert "error" not in pick
    postcondition = pick["postcondition"]
    # Background fusion can recover the static image before verification
    # (refuted), or the skill's own belief write is still present (unverified).
    # Neither channel establishes an independently observed physical move.
    assert postcondition["status"] in {"refuted", "unverified"}
    assert postcondition["kind"] == "relocated"
    assert postcondition["channel"] == "belief" and postcondition["measured"]
    assert pick["verified"] is False
    assert observed["ok"] is True and observed["objects_visible"]
    assert done["task_complete"] is True and done["success"] is False
    assert done["task_effects"] == ledger
    assert len(ledger["effects"]) == 1
    effect = ledger["effects"][0]
    assert (effect["actor"], effect["skill"], effect["status"]) == (
        "so101_mock", "pick_and_place", postcondition["status"])
    assert effect["reason"] == postcondition["evidence"]
    assert effect["task_id"] == ledger["task_id"]
    assert effect["action_id"] == 0


def test_minimal_mock_read_only_task_still_succeeds(cli_episode):
    code, report, records, ledger = cli_episode("look at the table and report what objects you see")
    assert code == 0
    assert report.success is True and report.unverified == []
    assert [row["skill"] for row in records] == ["get_observation", "task_done"]
    observation, done = [row["result"] for row in records]
    assert observation["ok"] is True and observation["objects_visible"]
    assert done["task_complete"] is True and done["success"] is True
    assert done["task_effects"] == ledger
    assert ledger["effects"] == []
