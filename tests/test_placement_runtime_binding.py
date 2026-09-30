"""Exercise the runtime hook with mock motion and independent fake observations."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from cascade.apps.demo import _truth_pose_fn
from cascade.control.arm_rig import ArmRig
from cascade.sim import placement
from test_placement_reporting import runtime  # Shared mock runtime lifecycle fixture.
from test_truth_isaac_initial import setup_camera_truth


def attach_reader(runtime, *, status="confirmed"):
    calls = []

    class Reader:
        def __call__(self, label):
            return None

        def placement(self, label, destination, *, evidence_dir=None):
            calls.append((label, destination, evidence_dir))
            return {"status": status, "evidence": "synthetic independent placement observation",
                    "measured": {"object": label, "destination": destination,
                                 "scope": "postplacement only", "final_samples": 7}}

    runtime.attach_verifier(object_pose=Reader())
    runtime.cfg._data["grasp"]["drop_zone_name"] = "green square"
    return calls


def mock_motion(runtime, *, destination="open box", ok=True):
    motion = {"ok": ok, "picked": "orange", "destination": destination,
              "destination_kind": "configured_point", "placed_at": [.30, -.14, .104],
              "return_home": {"attempted": True, "ok": False, "error": "synthetic home fault"}}
    if not ok:
        motion["error"] = "synthetic motion failure"
    runtime.skill_pick_and_place = lambda **kwargs: deepcopy(motion)
    return motion


@pytest.mark.parametrize("requested,canonical", [
    (None, "green square"), ("", "green square"), ("drop zone", "green square"),
    (" Cuadrado Verde ", "green square"), ("el cuadrado verde", "green square"),
    ("delivery area", "green square"), ("open_box", "open box"), ("The Beige Box", "open box"),
])
def test_execute_binds_requested_alias_to_configured_area(runtime, requested, canonical):
    calls = attach_reader(runtime)
    mock_motion(runtime, destination=canonical)
    args = {"object": "orange"}
    if requested is not None:
        args["destination"] = requested
    result = runtime.execute("pick_and_place", args)
    assert calls == [("orange", canonical, runtime.trace.run_dir / "placement")]
    assert result["ok"] and result["verified"]
    assert result["postcondition"]["measured"]["destination"] == canonical


@pytest.mark.parametrize("destination", ["another box", "red plate", "cuadrado rojo"])
def test_unknown_request_cannot_borrow_the_motion_results_known_area(runtime, destination):
    calls = attach_reader(runtime)
    mock_motion(runtime)
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": destination})
    assert calls == []
    assert result["verified"] is False
    assert result["postcondition"]["status"] == "unverified"


def test_spanish_square_request_is_not_replaced_by_reported_box(runtime):
    calls = attach_reader(runtime)
    mock_motion(runtime, destination="open box")
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": "cuadrado verde"})
    assert calls == [("orange", "green square", runtime.trace.run_dir / "placement")]
    assert result["verified"] is False
    assert "destinations differ" in result["postcondition"]["evidence"]


@pytest.mark.parametrize("kind", [None, "named_object"])
def test_missing_or_wrong_result_kind_cannot_bypass_independent_audit(runtime, kind):
    calls = attach_reader(runtime, status="refuted")
    motion = mock_motion(runtime)
    if kind is None:
        motion.pop("destination_kind")
    else:
        motion["destination_kind"] = kind
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": "open box"})
    assert calls == [("orange", "open box", runtime.trace.run_dir / "placement")]
    assert result["ok"] is False and result["self_reported_ok"] is True
    assert result["postcondition"]["status"] == "refuted"
    assert result["postcondition"]["evidence"] == "synthetic independent placement observation"


@pytest.mark.parametrize("reported", [None, "green square"])
def test_missing_or_wrong_reported_destination_cannot_be_confirmed(runtime, reported):
    calls = attach_reader(runtime)
    motion = mock_motion(runtime)
    motion.pop("destination_kind")  # Neither result field may disable the audit.
    if reported is None:
        motion.pop("destination")
    else:
        motion["destination"] = reported
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": "open box"})
    assert calls == [("orange", "open box", runtime.trace.run_dir / "placement")]
    assert result["verified"] is False and result["postcondition"]["status"] == "unverified"
    assert "destinations differ" in result["verification_note"]


@pytest.mark.parametrize("destination,kind,status", [
    ("open box", "configured_point", "unverified"),
    ("work surface", "named_object", "confirmed"),
])
def test_reader_without_placement_auditor_preserves_existing_verification(runtime, destination, kind, status):
    poses = iter(([.1, .2, .03], [.30, -.14, .104]))
    runtime.attach_verifier(object_pose=lambda label: next(poses) if label == "orange" else None)
    motion = mock_motion(runtime, destination=destination)
    motion["destination_kind"] = kind
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": destination})
    assert result["postcondition"]["status"] == status
    assert result["verified"] is (status == "confirmed")
    assert result["postcondition"]["channel"] == "physics"


def test_non_kitchen_configured_drop_zone_does_not_activate_kitchen_audit(runtime):
    calls = attach_reader(runtime)
    runtime.cfg._data["grasp"]["drop_zone_name"] = "packing station"
    assert runtime._verify_configured_placement("orange", "drop zone") is None
    assert calls == []


@pytest.mark.parametrize("selected", [None, "first", "second"])
def test_multi_arm_execution_cannot_confirm_with_primary_arms_placement_reader(runtime, monkeypatch, selected):
    calls = attach_reader(runtime)
    original = mock_motion(runtime)
    first, second = runtime.arm, object()
    rig = ArmRig([first, second], ["first", "second"])
    motion_arms = []

    def motion(**kwargs):
        motion_arms.append(runtime.arm)
        return deepcopy(original)

    # No method on the second arm is used: this regression exercises dispatch,
    # the independent-reader boundary and tracing, with a mock motion body.
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "arm_rig", rig)
        patch.setattr(runtime, "skill_pick_and_place", motion)
        args = {"object": "orange", "destination": "open box"}
        if selected is not None:
            args["arm"] = selected
        result = runtime.execute("pick_and_place", args)
    assert motion_arms == [second if selected == "second" else first]
    assert calls == [] and result["verified"] is False
    assert result["postcondition"]["status"] == "unverified"
    rows = [json.loads(line) for line in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["context"]["arm"] == (selected or "first")


@pytest.mark.parametrize("status", ["confirmed", "refuted", "unverified"])
def test_final_trace_contains_independent_verdict_and_original_motion_details(runtime, status):
    calls = attach_reader(runtime, status=status)
    original = mock_motion(runtime)
    args = {"object": "orange", "destination": "open box"}
    result = runtime.execute("pick_and_place", args)
    rows = [json.loads(line) for line in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    row = rows[-1]
    assert len(calls) == 1 and row["skill"] == "pick_and_place" and row["args"] == args
    assert row["result"] == json.loads(json.dumps(result, allow_nan=False))
    assert row["result"]["postcondition"]["status"] == status
    assert row["result"]["postcondition"]["channel"] == "physics"
    assert row["result"]["postcondition"]["measured"]["final_samples"] == 7
    assert row["result"]["return_home"] == original["return_home"]
    if status == "refuted":
        assert row["result"]["ok"] is False and row["result"]["self_reported_ok"] is True
        assert row["result"].get("verified") is not True
    else:
        assert row["result"]["ok"] is True
        assert row["result"]["verified"] is (status == "confirmed")


def test_failed_motion_keeps_failure_and_skips_placement_success_hook(runtime):
    calls = attach_reader(runtime)
    original = mock_motion(runtime, ok=False)
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": "open box"})
    assert calls == []
    assert result["ok"] is False and result["error"] == original["error"]
    assert result["return_home"] == original["return_home"]
    assert "Do not start another pick" in result["next_action"]
    rows = [json.loads(line) for line in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["result"] == result


def test_lazy_placement_uses_configured_identity_without_materializing_arm(monkeypatch, tmp_path):
    cfg, arm, client, _, rig, factory_calls = setup_camera_truth()
    observed = []

    def read(endpoint, robot_id, label, destination, *, evidence_dir=None):
        observed.append((endpoint, robot_id, label, destination, evidence_dir))
        return {"status": "unverified", "evidence": "synthetic observation only", "measured": {}}

    monkeypatch.setattr(placement, "read_placement", read)
    truth = _truth_pose_fn(SimpleNamespace(raw=arm), camera_rig=rig, arm_cfg=cfg)
    result = truth.placement("orange", "open box", evidence_dir=tmp_path)
    assert result["status"] == "unverified"
    assert observed == [(client._addr, "/test_robot", "orange", "open box", tmp_path)]
    assert not arm.connected and factory_calls == [] and client.calls == []


@pytest.mark.parametrize("fault", ["wrong_endpoint", "wrong_robot", "missing_robot", "wrong_backend"])
def test_lazy_placement_rejects_unbound_profile_or_camera(monkeypatch, fault):
    cfg, arm, client, frame, rig, factory_calls = setup_camera_truth()
    observed = []
    monkeypatch.setattr(placement, "read_placement", lambda *args, **kwargs: observed.append(args))
    if fault == "wrong_endpoint": cfg._data["bridge_port"] = 9999
    elif fault == "wrong_robot": cfg._data["bridge_robot_id"] = "/different_robot"
    elif fault == "missing_robot": cfg._data.pop("bridge_robot_id")
    elif fault == "wrong_backend": cfg._data["type"] = "mock"
    truth = _truth_pose_fn(SimpleNamespace(raw=arm), camera_rig=rig, arm_cfg=cfg)
    assert truth is None or truth.placement("orange", "open box") is None
    assert observed == [] and factory_calls == [] and not arm.connected and client.calls == []


def test_cached_truth_reader_still_checks_configured_endpoint_before_placement(monkeypatch):
    cfg, arm, client, _, rig, factory_calls = setup_camera_truth()
    truth = _truth_pose_fn(SimpleNamespace(raw=arm), camera_rig=rig, arm_cfg=cfg)
    assert truth.bound
    cfg._data["bridge_port"] = 9999
    observed = []
    monkeypatch.setattr(placement, "read_placement", lambda *args, **kwargs: observed.append(args))
    assert truth.placement("orange", "open box") is None
    assert observed == [] and factory_calls == [] and not arm.connected and client.calls == []
