"""Graph routing never upgrades an ACK, synthetic motion or late success."""
import copy
import threading

import pytest

from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.graph import SkillGraph, run_skill_graph
from cascade.robotics.runtime import RobotRuntime


class Domain:
    domain_id = "body"
    motion_skills = {"move"}

    def __init__(self, *, synthetic=False):
        self.resources = (ResourceDescriptor("body/base", "base", "fixture", synthetic=synthetic,
                                             controller_id="fixture/control", writer_id="body"),)
        self.tool_descriptors = (
            ToolDescriptor("body.read", "fixture read", {"type": "object", "properties": {}},
                           "body", "read"),
            ToolDescriptor("body.move", "fixture move", {"type": "object", "properties": {
                "distance": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["distance"]}, "body", "move", effect="motion", writes=("body/base",)),
        )
        self.calls, self.stops = [], 0
        self.reply = {"ok": True, "postcondition": {"status": "confirmed"}}
        self.distance = .1
        self.block = None

    def execute(self, name, args):
        self.calls.append((name, args))
        if name == "read":
            return {"ok": True, "distance": self.distance}
        if self.block:
            assert self.block.wait(2), "stop did not reach blocked fixture"
        return self.reply

    def stop(self):
        self.stops += 1
        if self.block:
            self.block.set()
        return {"ok": True}

    def close(self):
        return {"ok": True}

    def reset_stop(self):
        return {"ok": True}


def graph_data():
    return {"version": 1, "entry": "observe", "max_steps": 3, "timeout_s": 1,
            "nodes": {
                "observe": {"tool": "body.read", "arguments": {},
                            "edges": {"success": "move", "error": "$failure"}},
                "move": {"tool": "body.move", "arguments": {
                    "distance": {"$result": {"node": "observe", "path": ["distance"]}}},
                    "edges": {"confirmed": "$success", "refuted": "$failure",
                              "unverified": "$failure", "error": "$failure"}}}}


@pytest.fixture
def pair():
    domain = Domain()
    runtime = RobotRuntime({"body": domain})
    yield domain, runtime
    runtime.close()


def test_graph_binds_data_through_real_runtime_and_preserves_receipt(pair):
    domain, runtime = pair
    spec = graph_data()
    graph = SkillGraph(spec)
    spec["nodes"]["move"]["tool"] = "reset_stop"
    result = run_skill_graph(graph, runtime)
    assert result["ok"] and not result["physical_admission"]
    assert domain.calls == [("read", {}), ("move", {"distance": .1})]
    assert result["steps"][-1]["result"] == domain.reply
    assert len(result["catalog_sha256"]) == len(result["graph_sha256"]) == 64


def test_stop_and_reset_between_graph_check_and_runtime_admission(pair, monkeypatch):
    domain, runtime = pair
    original_execute = runtime.execute

    def race(name, args, *, expected_generation, deadline_monotonic_s):
        assert runtime.stop()["ok"]
        assert runtime.reset_stop()["ok"]
        return original_execute(name, args, expected_generation=expected_generation,
                                deadline_monotonic_s=deadline_monotonic_s)

    monkeypatch.setattr(runtime, "execute", race)
    result = run_skill_graph(SkillGraph(graph_data()), runtime)
    assert not result["ok"]
    assert domain.calls == []
    assert "stale execution generation" in result["steps"][0]["result"]["error"]


@pytest.mark.parametrize("reply", [
    {"ok": True},
    {"ok": True, "postcondition": {"status": "unverified"}},
    {"ok": False, "execution_ok": True, "postcondition": {"status": "refuted"}},
    {"ok": True, "execution_ok": False, "postcondition": {"status": "confirmed"}},
    {"ok": True, "delivery_uncertain": True, "postcondition": {"status": "confirmed"}},
])
def test_success_edge_cannot_hide_unverified_or_refuted_motion(pair, reply):
    domain, runtime = pair
    domain.reply = reply
    spec = graph_data()
    spec["nodes"]["move"]["edges"] = dict.fromkeys(spec["nodes"]["move"]["edges"], "$success")
    result = run_skill_graph(SkillGraph(spec), runtime)
    assert result["terminal"] == "$success"
    assert not result["ok"] and result["unresolved_motion"]


def test_synthetic_motion_never_proves_physical_success():
    domain = Domain(synthetic=True)
    runtime = RobotRuntime({"body": domain})
    try:
        result = run_skill_graph(SkillGraph(graph_data()), runtime)
        assert not result["ok"] and result["unresolved_motion"][0]["synthetic"]
    finally:
        runtime.close()


@pytest.mark.parametrize("distance", [True, "0.1", 2, float("nan")])
def test_dynamic_argument_rejected_before_motion(pair, distance):
    domain, runtime = pair
    domain.distance = distance
    result = run_skill_graph(SkillGraph(graph_data()), runtime)
    assert not result["ok"]
    assert all(name != "move" for name, _ in domain.calls)


@pytest.mark.parametrize("change", [
    lambda s: s["nodes"]["move"].update(tool="reset_stop"),
    lambda s: s["nodes"]["move"]["edges"].update(confirmed="observe"),
    lambda s: s["nodes"]["move"]["edges"].pop("unverified"),
    lambda s: s.update(max_steps=True),
    lambda s: s.update(timeout_s=float("inf")),
    lambda s: s["nodes"]["move"]["arguments"].update(distance={"$result": {"node": "move", "path": []}}),
    lambda s: s["nodes"].update(unreachable=copy.deepcopy(s["nodes"]["observe"])),
])
def test_invalid_graph_does_no_runtime_io(pair, change):
    domain, runtime = pair
    spec = graph_data()
    change(spec)
    with pytest.raises(ValueError):
        run_skill_graph(SkillGraph(spec), runtime)
    assert domain.calls == [] and domain.stops == 0


def test_binding_source_must_exist_on_every_branch(pair):
    domain, runtime = pair
    spec = graph_data()
    spec["nodes"]["start"] = {"tool": "body.read", "arguments": {},
                               "edges": {"success": "observe", "error": "move"}}
    spec["entry"] = "start"
    with pytest.raises(ValueError, match="every path"):
        run_skill_graph(SkillGraph(spec), runtime)
    assert domain.calls == []


def test_deadline_reaches_priority_stop_and_rejects_late_success(pair):
    domain, runtime = pair
    domain.block = threading.Event()
    spec = graph_data()
    spec["timeout_s"] = .03
    result = run_skill_graph(SkillGraph(spec), runtime)
    assert result["interrupted"] == "deadline" and not result["ok"]
    assert domain.stops == 1 and runtime.stopped


def test_precancelled_graph_never_dispatches(pair):
    domain, runtime = pair
    cancel = threading.Event()
    cancel.set()
    result = run_skill_graph(SkillGraph(graph_data()), runtime, cancel_event=cancel)
    assert result["interrupted"] == "cancelled" and not result["ok"]
    assert not domain.calls and runtime.stopped


def test_step_limit_is_not_success(pair):
    domain, runtime = pair
    spec = graph_data()
    spec["max_steps"] = 1
    result = run_skill_graph(SkillGraph(spec), runtime)
    assert not result["ok"] and result["interrupted"] == "step_limit"
    assert domain.calls == [("read", {})] and runtime.stopped


def test_stop_during_watchdog_teardown_invalidates_terminal(pair, monkeypatch):
    from types import SimpleNamespace
    import cascade.robotics.graph as module
    domain, runtime = pair
    real_thread = threading.Thread

    class StopAtJoin(real_thread):
        def join(self, *args, **kwargs):
            super().join(*args, **kwargs)
            runtime.stop()

    monkeypatch.setattr(module, "threading", SimpleNamespace(
        Thread=StopAtJoin, Event=threading.Event, Lock=threading.Lock))
    result = run_skill_graph(SkillGraph(graph_data()), runtime)
    assert result["terminal"] == "$success"
    assert not result["ok"] and result["interrupted"] == "runtime_invalidated"
