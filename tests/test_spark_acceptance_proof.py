"""OpenClaw/process boundary doubles, never simulator acceptance evidence."""
import json
import re
import time

import pytest

from test_demo_proof import demo_proof, host_boundary as _host_boundary


host_boundary = _host_boundary


@pytest.fixture
def spark_boundary(host_boundary, monkeypatch):
    h = host_boundary
    h.update(turns=[], audits=[], witnesses=[], audit_fail=None, duplicate_call=False,
             reset_failure=False, wrong_object=False)
    monkeypatch.setenv("CASCADE_INSTALL_PROFILE", "spark")

    def turn(session, model, message, output, timeout, required_tool=None):
        arguments = json.loads(re.search(r"with arguments (\{.*?\})\.", message).group(1))
        h["turns"].append((required_tool, arguments, session))
        output.write_text(json.dumps({"synthetic_test_only": True, "arguments": arguments}))
        if "trace" not in h:
            h["spawn"]()
        trace = h["trace"]
        trace.parent.mkdir(parents=True, exist_ok=True)
        if required_tool == "pick_and_place":
            if h["wrong_object"]:
                arguments["object"] = "pink cube"
            row = {"t": time.time(), "skill": required_tool, "args": arguments,
                   "result": {"ok": True, "verified": False,
                              "postcondition": {"status": "unverified", "channel": "physics"}}}
            with trace.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        elif required_tool == "reset_scene":
            row = {"t": time.time(), "skill": "reset_scene", "result": {
                "ok": not h["reset_failure"], "world": "isaac",
                "props_reset": ["pink_cube", "green_cube", "tomato_can", "lemon", "orange"]}}
            with trace.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return {"meta": {"toolSummary": {"calls": 2 if h["duplicate_call"] else 1,
                "tools": ["cascade__" + required_tool], "failures": 0}}}

    class Witness:
        def __init__(self, repo, evidence, *, object_name, destination_name):
            self.object_name, self.destination_name = object_name, destination_name
            self.marks, self.frames = [], []
            h["witnesses"].append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def mark(self, phase):
            self.marks.append(phase)

        def settle(self, **kwargs):
            assert kwargs == {"wall_timeout": 90}

        def capture_frames(self, phase):
            self.frames.append(phase)

        def audit(self):
            h["audits"].append(self.object_name)
            return {"pass": h["audit_fail"] != self.object_name,
                    "object_name": self.object_name, "destination_name": self.destination_name,
                    "synthetic_test_only": True}

    monkeypatch.setattr(demo_proof, "agent_turn", turn)
    monkeypatch.setattr(demo_proof, "make_spark_witness", Witness)
    return h


def test_two_native_orders_with_observed_resets_share_one_real_process_binding(spark_boundary):
    h = spark_boundary
    report = demo_proof.run_proof(h["repo"], h["state"], "isaac", True)
    assert report["verified"] is True
    assert [(case["object"], case["destination"]) for case in report["cases"]] == [
        ("green cube", "green square"), ("orange", "open box")]
    tools = [tool for tool, _, _ in h["turns"]]
    assert tools == ["world_state", *(["get_observation", "pick_and_place", "reset_scene",
                                      "world_state", "get_observation"] * 2)]
    assert len({session for _, _, session in h["turns"]}) == 1
    assert h["audits"] == ["green_cube", "orange"]
    assert len(h["children"]) == 1
    for witness in h["witnesses"]:
        assert witness.marks == ["pick_begin", "pick_end", "reset_begin", "reset_end"]
        assert witness.frames == ["placed", "reset"]


@pytest.mark.parametrize("fault", ["green_audit", "orange_audit", "reset_failure", "duplicate_call", "wrong_object"])
def test_no_ready_receipt_for_failed_or_substituted_cases(spark_boundary, fault):
    h = spark_boundary
    if fault.endswith("audit"):
        h["audit_fail"] = "green_cube" if fault == "green_audit" else "orange"
    else:
        h[fault] = True
    with pytest.raises(demo_proof.ProofError):
        demo_proof.run_proof(h["repo"], h["state"], "isaac", True)
    assert json.loads((h["state"] / "proof.json").read_text())["verified"] is False
    if fault != "orange_audit":
        assert not any(arguments.get("object") == "orange" for _, arguments, _ in h["turns"])
