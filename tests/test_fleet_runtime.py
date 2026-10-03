"""Independent robot coordination, with synthetic domains and no physics claims."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import pytest

from cascade.agent.trace import TraceLogger
from cascade.memory.episodic import EpisodicMemory
from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.fleet import FleetRuntime
from cascade.robotics.runtime import RobotRuntime


class FixtureDomain:
    domain_id = "locomotion"

    def __init__(self, robot_id):
        self.resources = (ResourceDescriptor("locomotion/body", "base", robot_id,
            controller_id=f"mock:{robot_id}", writer_id="locomotion/body", synthetic=True),)
        self.tool_descriptors = (ToolDescriptor("locomotion.move", "Synthetic routing fixture",
            {"type": "object", "properties": {"distance": {"type": "number"}}, "required": ["distance"]},
            "locomotion", "move", effect="motion", writes=("locomotion/body",)),)
        self.entered, self.release, self.stopped = threading.Event(), threading.Event(), threading.Event()
        self.barrier = None
        self.hold = False
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        self.entered.set()
        if self.barrier is not None:
            self.barrier.wait(5)
        if self.hold:
            assert self.release.wait(5), "test must release its synthetic domain"
        return {"ok": True, "execution_ok": True, "postcondition": {"status": "confirmed"}}

    def stop(self):
        self.stopped.set()
        self.release.set()
        return {"ok": True}

    def reset_stop(self):
        return {"ok": True}

    def begin_task(self):
        pass

    def close(self):
        return {"ok": True, "complete": True}


def member(name, tmp_path):
    domain = FixtureDomain(name)
    runtime = RobotRuntime({domain.domain_id: domain}, memory=EpisodicMemory(),
                           trace=TraceLogger(tmp_path / name))
    return runtime, domain


def test_twelve_simultaneous_actions_keep_catalog_trace_and_debt_per_robot(tmp_path):
    pairs = {f"duck{i:02}": member(f"duck{i:02}", tmp_path) for i in range(12)}
    fleet = FleetRuntime({name: rt for name, (rt, _) in pairs.items()})
    barrier = threading.Barrier(13)
    for _, domain in pairs.values():
        domain.barrier, domain.hold = barrier, True
    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(fleet.execute, name, "locomotion.move", {"distance": i / 100})
                       for i, name in enumerate(pairs)]
            try:
                barrier.wait(5)  # cannot pass unless all twelve actually entered
                assert all(domain.entered.is_set() for _, domain in pairs.values())
                assert not fleet.execute("duck00", "locomotion.move", {"distance": 1})["ok"]
            finally:
                for _, domain in pairs.values():
                    domain.release.set()
            assert all(f.result(5)["ok"] for f in futures)
        catalog = fleet.catalog()
        for i, (name, (rt, domain)) in enumerate(pairs.items()):
            assert domain.calls == [("move", {"distance": i / 100})]
            assert catalog[name]["resources"][0]["robot_id"] == name
            assert {t["name"] for t in catalog[name]["tools"]} == set(rt.tool_descriptors)
            assert fleet.unverified_actions()[name] == ["locomotion.move"]  # synthetic is never physical proof
            assert (tmp_path / name / "trace.jsonl").is_file()
        pairs["duck00"][0].begin_task()
        assert fleet.unverified_actions()["duck00"] == []
        assert all(fleet.unverified_actions()[n] for n in pairs if n != "duck00")
    finally:
        assert fleet.close()["complete"]


def test_individual_stop_invalidates_late_result_without_stopping_another_robot(tmp_path):
    first, a = member("first", tmp_path)
    second, b = member("second", tmp_path)
    fleet = FleetRuntime({"first": first, "second": second})
    a.hold = b.hold = True
    old = first.cancellation_token
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(fleet.execute, "first", "locomotion.move", {"distance": .03})
            fb = pool.submit(fleet.execute, "second", "locomotion.move", {"distance": .02})
            try:
                assert a.entered.wait(3) and b.entered.wait(3)
                assert fleet.stop("first")["ok"]
                assert a.stopped.is_set() and not b.stopped.is_set()
                assert not fb.done()
                assert not fa.result(3)["ok"]
                assert fleet.reset_stop("first")["ok"]
                assert not fleet.execute("first", "locomotion.move", {"distance": .04}, expected_generation=old)["ok"]
                assert len(a.calls) == 1
                assert "locomotion.move" in first.unverified_actions()
            finally:
                a.release.set()
                b.release.set()
            assert fb.result(3)["ok"]
    finally:
        assert fleet.close()["complete"]


def test_global_stop_has_one_deadline_and_does_not_wait_for_blocked_rpc(tmp_path):
    pairs = {name: member(name, tmp_path) for name in ("stuck", "other", "third")}
    fleet = FleetRuntime({name: rt for name, (rt, _) in pairs.items()})
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    result = []

    def blocked_stop():
        entered.set()
        assert release.wait(5)
        return {"ok": True}

    pairs["stuck"][1].stop = blocked_stop
    worker = threading.Thread(target=lambda: (result.append(fleet.stop()), returned.set()))
    try:
        worker.start()
        assert entered.wait(3)
        assert pairs["other"][1].stopped.wait(3) and pairs["third"][1].stopped.wait(3)
        assert returned.wait(2), "fleet stop waited for the stuck RPC"
        assert not result[0]["ok"] and not result[0]["physical_stop_verified"]
        assert result[0]["robots"]["stuck"]["domains"]["locomotion"]["pending"]
        assert all(rt.stopped for rt, _ in pairs.values())
        assert all(not domain.calls for _, domain in pairs.values())
    finally:
        release.set()
        worker.join(3)
        assert fleet.close()["complete"]


@pytest.mark.parametrize("transition", ["stop_reset", "new_episode"])
def test_dispatch_paused_after_fleet_selection_cannot_cross_episode_boundary(tmp_path, transition):
    runtime, domain = member("robot", tmp_path)
    fleet = FleetRuntime({"robot": runtime})
    entered, release = threading.Event(), threading.Event()
    execute = runtime.execute

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return execute(*args, **kwargs)

    runtime.execute = delayed
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fleet.execute, "robot", "locomotion.move", {"distance": .03})
            try:
                assert entered.wait(3)
                if transition == "stop_reset":
                    assert fleet.stop()["ok"] and fleet.reset_stop("robot")["ok"]
                else:
                    runtime.begin_task()
            finally:
                release.set()
            assert not future.result(3)["ok"] and not domain.calls
            runtime.execute = execute
            assert fleet.execute("robot", "locomotion.move", {"distance": .02})["ok"]
    finally:
        assert fleet.close()["complete"]


@pytest.mark.parametrize("shared", ["robot_id", "runtime", "episode", "controller", "memory", "trace", "trace_dir", "identity"])
def test_duplicate_ownership_is_rejected_before_any_dispatch(tmp_path, shared):
    first, a = member("first", tmp_path)
    second, b = member("second", tmp_path)
    pairs = [("first", first), ("second", second)]
    if shared == "robot_id":
        pairs[1] = ("first", second)
    elif shared == "runtime":
        pairs[1] = ("second", first)
    elif shared == "episode":
        second._task_id = first.episode_id
    elif shared in {"controller", "identity"}:
        changes = {"controller_id": a.resources[0].controller_id} if shared == "controller" else {"robot_id": "first"}
        from cascade.robotics.resources import ResourceCatalog
        second.resources = ResourceCatalog([replace(b.resources[0], **changes)])
    elif shared == "trace_dir":
        second.trace = TraceLogger(tmp_path / "first")
    else:
        setattr(second, shared, getattr(first, shared))
    with pytest.raises(ValueError):
        FleetRuntime(pairs)
    assert not a.calls and not b.calls and not a.stopped.is_set() and not b.stopped.is_set()


def test_unknown_robot_and_unsupported_tool_never_fall_back_to_primary(tmp_path):
    runtime, domain = member("robot", tmp_path)
    fleet = FleetRuntime({"robot": runtime})
    with pytest.raises(ValueError, match="unknown fleet robot"):
        fleet.execute("other", "locomotion.move", {"distance": .03})
    assert not fleet.execute("robot", "manipulation.grasp", {})["ok"]
    assert not domain.calls
    assert fleet.close()["complete"]


def test_old_stop_receipt_cannot_credit_a_newer_stop_or_reset(tmp_path):
    runtime, _ = member("robot", tmp_path)
    old = runtime.request_stop()
    assert runtime.wait_for_stop(old, deadline_monotonic_s=time.monotonic() + .5)["ok"]
    assert runtime.reset_stop()["ok"]
    assert not runtime.wait_for_stop(old, deadline_monotonic_s=time.monotonic())["ok"]
    assert runtime.stop()["ok"]
    assert not runtime.wait_for_stop(old, deadline_monotonic_s=time.monotonic())["ok"]
    assert runtime.close()["complete"]


def test_enqueue_failure_does_not_prevent_other_robots_being_fenced(tmp_path):
    first, a = member("first", tmp_path)
    second, b = member("second", tmp_path)
    fleet = FleetRuntime({"first": first, "second": second})

    def fail_enqueue():
        raise RuntimeError("worker unavailable")

    first.request_stop = fail_enqueue
    result = fleet.stop()
    assert not result["ok"] and result["robots"]["first"]["error"] == "worker unavailable"
    assert second.stopped and b.stopped.is_set()
    assert not fleet.execute("first", "locomotion.move", {"distance": .01})["ok"]
    assert not fleet.execute("first", "task_done", {"success": True, "summary": "claim"})["ok"]
    assert "fleet stop unresolved" in fleet.unverified_actions()["first"]
    assert not a.calls
    assert fleet.close()["complete"]


def test_delayed_reset_cannot_clear_a_newer_robot_stop(tmp_path):
    runtime, domain = member("robot", tmp_path)
    fleet = FleetRuntime({"robot": runtime})
    entered, release = threading.Event(), threading.Event()
    original_reset, resets = runtime.reset_stop, []
    domain.reset_stop = lambda: resets.append(True) or {"ok": True}

    def delayed(**kwargs):
        entered.set()
        assert release.wait(5)
        return original_reset(**kwargs)

    runtime.reset_stop = delayed
    try:
        assert fleet.stop()["ok"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fleet.reset_stop, "robot")
            try:
                assert entered.wait(3)
                assert fleet.stop()["ok"]
            finally:
                release.set()
            assert not future.result(3)["ok"]
        assert runtime.stopped and not resets
    finally:
        assert fleet.close()["complete"]


def test_close_fences_every_robot_before_waiting_for_first_teardown(tmp_path):
    first, _ = member("first", tmp_path)
    second, b = member("second", tmp_path)
    fleet = FleetRuntime({"first": first, "second": second})
    entered, release = threading.Event(), threading.Event()
    original_close = first.close
    graceful = threading.Event()
    b.request_shutdown = lambda: graceful.set() or {"ok": True}

    def delayed():
        entered.set()
        assert release.wait(5)
        return original_close()

    first.close = delayed
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fleet.close)
        try:
            assert entered.wait(3)
            assert second.stopped and graceful.wait(3)
            assert not b.stopped.is_set()  # idle closure retains graceful policy
            assert not second.execute("locomotion.move", {"distance": .03})["ok"]
            assert not b.calls
        finally:
            release.set()
        assert future.result(3)["complete"]


@pytest.mark.parametrize("store", ["beliefs", "grasp_memory", "envelope", "cross_kind"])
def test_distinct_store_objects_cannot_write_the_same_resolved_file(tmp_path, store):
    first, _ = member("first", tmp_path)
    second, _ = member("second", tmp_path)
    directory = tmp_path / "stores"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    if store == "beliefs":
        first.beliefs_path, second.beliefs_path = directory / "state.json", alias / "state.json"
    elif store == "cross_kind":
        first.beliefs_path = directory / "state.json"
        second.envelope = SimpleNamespace(path=alias / "state.json")
    else:
        if store == "grasp_memory":
            from cascade.memory.grasp_memory import GraspOutcomeMemory
            first.grasp_memory = GraspOutcomeMemory(directory / "state.json")
            second.grasp_memory = GraspOutcomeMemory(alias / "state.json")
        else:
            setattr(first, store, SimpleNamespace(path=directory / "state.json"))
            setattr(second, store, SimpleNamespace(path=alias / "state.json"))
    with pytest.raises(ValueError, match="shared.*file"):
        FleetRuntime({"first": first, "second": second})
    assert not (directory / "state.json").exists()


def test_stop_receipt_requires_an_actual_request_even_without_actuating_domains():
    runtime = RobotRuntime({})
    for _ in range(2):
        receipt = runtime.wait_for_stop(runtime.cancellation_token, deadline_monotonic_s=time.monotonic())
        assert not receipt["ok"] and receipt["latched"] == runtime.stopped
        assert runtime.stop()["ok"]
        runtime.begin_task()
    assert not runtime.wait_for_stop(runtime.cancellation_token, deadline_monotonic_s=time.monotonic())["ok"]
    assert runtime.close()["complete"]


def test_published_microduck_profile_keeps_its_exact_identity(tmp_path):
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import load_robot_config
    cfg = load_robot_config("microduck_conversation_mock")
    runtime, _ = build_robot_runtime(cfg, tmp_path / "microduck")
    fleet = FleetRuntime({cfg.robot_id: runtime})
    try:
        assert set(fleet.catalog()) == {"microduck-mock"}
        result = fleet.execute("microduck-mock", "locomotion.list_bases")
        assert result["ok"] and result["robot_id"] == "microduck-mock"
        assert "manipulation.open_gripper" not in runtime.tool_descriptors
    finally:
        assert fleet.close()["complete"]


@pytest.mark.parametrize("invalid", [True, ""])
def test_robot_identity_uses_the_existing_contract(tmp_path, invalid):
    runtime, _ = member("robot", tmp_path)
    with pytest.raises(ValueError, match="fleet robot ID"):
        FleetRuntime([(invalid, runtime)])


def test_slash_separated_ids_cannot_alias_two_writers_to_one_controller(tmp_path):
    from cascade.robotics.resources import ResourceCatalog
    first, a = member("a", tmp_path)
    second, b = member("a/b", tmp_path)
    first.resources = ResourceCatalog([replace(a.resources[0], controller_id="endpoint", writer_id="b/c")])
    second.resources = ResourceCatalog([replace(b.resources[0], controller_id="endpoint", writer_id="c")])
    with pytest.raises(ValueError, match="incompatible writers"):
        FleetRuntime({"a": first, "a/b": second})


def test_full_length_identity_and_one_writer_for_multiple_resources_are_preserved(tmp_path):
    from cascade.robotics.resources import ResourceCatalog
    name = "r" * 128
    runtime, domain = member(name, tmp_path)
    first = domain.resources[0]
    runtime.resources = ResourceCatalog([first, replace(first, resource_id="body/second")])
    fleet = FleetRuntime({name: runtime})
    assert set(fleet.catalog()) == {name}
    assert fleet.catalog()[name]["resources"] == runtime.resources.describe()
    assert len({r.writer_id for r in fleet.resources}) == 1
    assert {r.controller_id for r in fleet.resources} == {first.controller_id}
    assert fleet.close()["complete"]
