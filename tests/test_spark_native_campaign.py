"""I/O and sequencing checks for the external harness; no live service calls."""

import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "native_campaign", Path(__file__).resolve().parents[1] / "benchmark/diagnostics/spark_native_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
ID = "a" * 32
IDLE = {"enabled": True, "ready": True, "busy": False, "agent": "cascade-demo"}


@pytest.mark.parametrize("failure", [None, "post_disconnect", "wrong_id", "uncertain", "tool_failed"])
def test_turn_posts_once_and_preserves_confirmed_or_uncertain_evidence(tmp_path, failure):
    calls = []

    def transport(method, path, payload=None):
        calls.append((method, path, payload))
        if path == "/api/chat" and method == "GET":
            return IDLE
        if method == "POST":
            if failure == "post_disconnect":
                raise ConnectionError("test response lost after submission")
            return {"id": ID, "status": "running"}
        return {**IDLE, "order": {"id": "b" * 32 if failure == "wrong_id" else ID,
                "status": "uncertain" if failure == "uncertain" else "done",
                "tools": ["pick_and_place"], "tool_failures": int(failure == "tool_failed")}}

    api = campaign.VisitorClient(lambda: None, transport=transport, poll_interval=0)
    output = tmp_path / "turn.json"
    if failure in ("post_disconnect", "wrong_id", "uncertain"):
        with pytest.raises(campaign.UncertainOrder):
            api.turn("Please put the orange in the open box.", "pick_and_place", output)
        assert json.loads(output.read_text())["status"] == "uncertain"
    elif failure == "tool_failed":
        with pytest.raises(RuntimeError) as error:
            api.turn("Please put the orange in the open box.", "pick_and_place", output)
        assert type(error.value) is RuntimeError
        assert json.loads(output.read_text())["status"] == "done"
    else:
        assert api.turn("Please put the orange in the open box.", "pick_and_place", output)["status"] == "done"
    assert sum(method == "POST" for method, _, _ in calls) == 1


def test_busy_visitor_never_receives_a_campaign_post(tmp_path):
    calls = []

    def transport(method, path, payload=None):
        calls.append(method)
        return {**IDLE, "busy": True}

    api = campaign.VisitorClient(lambda: None, transport=transport)
    with pytest.raises(RuntimeError, match="idle"):
        api.turn("Reset the scene.", "reset_scene", tmp_path / "none.json")
    assert calls == ["GET"]


@pytest.mark.parametrize("uncertain", [False, True])
def test_only_a_confirmed_finished_pick_failure_can_be_followed_by_reset(tmp_path, uncertain):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("")
    calls = []

    class API:
        def turn(self, message, tool, path):
            calls.append(tool)
            if tool == "pick_and_place":
                raise (campaign.UncertainOrder("unknown state") if uncertain else RuntimeError("completed failure"))
            return {"status": "done"}

    class Witness:
        def __init__(self, out, *args):
            self.out, self.records, self.errors = out, [{"physics": {"engine": "physx"}}], []

        def __enter__(self):
            self.out.mkdir()
            return self

        def __exit__(self, *args):
            pass

        def mark(self, *args):
            pass

        def settle(self, **kwargs):
            pass

        def capture_frames(self, *args):
            pass

        def audit(self):
            return {"pass": False}

    def validate_reset_trace(path, started, sim, target):
        assert isinstance(path, Path)
        path.read_text()
        assert sim == "isaac" and target == "orange"
        return ["orange"]

    def validate_pick_trace(path, started, target, *, placement_audit=None):
        assert isinstance(path, Path)
        path.read_text()
        assert target == "orange" and placement_audit == {"pass": False}

    validators = SimpleNamespace(validate_reset_trace=validate_reset_trace,
                                 validate_pick_trace=validate_pick_trace)
    result = campaign.run_case(API(), Witness, validators, {"trace": str(trace)},
                               tmp_path / "case", "orange", "open box")
    assert result["pass"] is False
    assert result["uncertain_order"] is uncertain
    assert result["reset_confirmed"] is not uncertain
    assert calls == (["pick_and_place"] if uncertain else
                     ["pick_and_place", "reset_scene", "world_state", "describe_scene"])


def test_changed_world_binding_during_poll_aborts_without_another_post(tmp_path):
    guarded = 0
    calls = []

    def guard():
        nonlocal guarded
        guarded += 1
        if guarded > 1:
            raise RuntimeError("proof process changed")

    def transport(method, path, payload=None):
        calls.append(method)
        return IDLE if method == "GET" else {"id": ID, "status": "running"}

    api = campaign.VisitorClient(guard, transport=transport)
    with pytest.raises(campaign.UncertainOrder, match="changed"):
        api.turn("Please put the lemon in the open box.", "pick_and_place", tmp_path / "turn.json")
    assert calls == ["GET", "POST"]


@pytest.mark.parametrize("object_name,destination", campaign.CASES)
def test_completed_case_uses_original_trace_validators(tmp_path, object_name, destination):
    """Exercise the actual product contracts using synthetic native trace I/O."""
    repo = Path(__file__).resolve().parents[1]
    validators = campaign.load_module("original_demo_proof", repo / "scripts/demo_proof.py")
    trace = tmp_path / "trace.jsonl"
    trace.write_text("")
    calls, phases, captures = [], [], []

    class API:
        def turn(self, message, required_tool, path):
            assert isinstance(path, Path)
            calls.append(required_tool)
            if required_tool == "pick_and_place":
                args = {"object": object_name.replace("_", " "), "destination": destination}
                result = {"ok": True, "verified": False,
                          "postcondition": {"status": "unverified", "channel": "physics"}}
            elif required_tool == "reset_scene":
                args, result = {}, {"ok": True, "world": "isaac", "props_reset": [object_name]}
            else:
                args, result = {}, {"ok": True}
            with trace.open("a") as stream:
                stream.write(json.dumps({"t": time.time(), "skill": required_tool,
                                         "args": args, "result": result}) + "\n")
            return {"status": "done"}

    class Witness:
        def __init__(self, out, received_object, received_destination):
            assert isinstance(out, Path)
            assert (received_object, received_destination) == (object_name, destination)
            self.out, self.records, self.errors = out, [{"physics": {"engine": "physx"}}], []
            self.stopped = False

        def __enter__(self):
            self.out.mkdir()
            return self

        def __exit__(self, typ, value, traceback):
            self.stopped = True

        def mark(self, phase):
            phases.append(phase)

        def settle(self, simulation_seconds=.65, wall_timeout=20):
            assert simulation_seconds == .65 and wall_timeout == 90

        def capture_frames(self, phase):
            assert phase in ("placed", "reset")
            captures.append(phase)

        def audit(self, *, target_xy=None, object_half_height=None):
            assert self.stopped
            assert target_xy is None and object_half_height is None
            return {"pass": True, "object_name": object_name, "destination_name": destination}

    # Binding matches JSON proof.json: trace is deliberately a string here.
    result = campaign.run_case(API(), Witness, validators, {"trace": str(trace)},
                               tmp_path / "case", object_name, destination)
    assert result["errors"] == []
    assert result["pass"] is True
    assert result["reset_confirmed"] is True and result["pick_trace_confirmed"] is True
    assert result["props_reset"] == [object_name]
    assert calls == ["pick_and_place", "reset_scene", "world_state", "describe_scene"]
    assert phases == ["pick_begin", "pick_end", "reset_begin", "reset_end"]
    assert captures == ["placed", "reset"]
    assert len(json.loads((tmp_path / "case/trace-evidence.json").read_text())) == 4
