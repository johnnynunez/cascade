"""Read-only transport and independent request binding for placement verdicts."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from cascade.agent.effects import PostconditionChecker
from cascade.sim import placement
from test_spark_placement_proof import trajectory


def setup_reader(monkeypatch):
    rows, expected = trajectory()
    samples = iter(row["physics"] for row in rows[16:26])
    clients = []
    now = [100.]

    class Client:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.requests = []
            self.closed = False
            clients.append(self)

        def connect(self):
            pass

        def request(self, request, timeout_s):
            assert request["op"] == "exec"
            assert 0 < timeout_s <= placement.REQUEST_TIMEOUT_S
            self.requests.append(request)
            return next(samples)

        def close(self):
            self.closed = True

    monkeypatch.setattr(placement, "BridgeClient", Client)
    monkeypatch.setattr(placement.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(placement.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    real = placement._auditor()
    gpu = SimpleNamespace(
        load_expected_scene_geometry=lambda path: expected,
        snapshot_code=lambda **kwargs: "read-only snapshot",
        original=SimpleNamespace(parse_snapshot_reply=lambda reply: reply))
    monkeypatch.setattr(placement, "_auditor", lambda: SimpleNamespace(gpu=gpu, audit_placement=real.audit_placement))
    return clients, Client, now


def read(**kwargs):
    return placement.read_placement(("localhost", 8611), "/SYNTHETIC_TEST_ROBOT", "orange", "open box", **kwargs)


def test_fresh_private_observation_saves_a_replayable_receipt(monkeypatch, tmp_path):
    clients, _, _ = setup_reader(monkeypatch)
    report = read(evidence_dir=tmp_path)
    assert report["status"] == "confirmed", report
    assert len(clients) == 1 and clients[0].closed
    assert len(clients[0].requests) >= 6
    raw = next(tmp_path.glob("*.json")).read_bytes()
    assert report["measured"]["evidence_sha256"] == hashlib.sha256(raw).hexdigest()
    evidence = json.loads(raw)
    assert evidence["requested_object"] == "orange"
    assert len(evidence["records"]) == report["measured"]["observed_samples"]
    assert evidence["verdict"]["status"] == "confirmed"


def test_timeout_closes_only_the_private_client_and_stays_unverified(monkeypatch):
    clients, client, _ = setup_reader(monkeypatch)
    def broken(*args, **kwargs):
        raise TimeoutError("test timeout")
    monkeypatch.setattr(client, "request", broken)
    assert read()["status"] == "unverified"
    assert clients[0].closed


@pytest.mark.parametrize("fault", ["frozen", "rewound", "nonfinite", "late"])
def test_invalid_or_stalled_clock_never_confirms(monkeypatch, fault):
    clients, client, now = setup_reader(monkeypatch)
    original = client.request
    def request(self, *args, **kwargs):
        sample = original(self, *args, **kwargs)
        if fault == "frozen":
            sample["sim_time"], sample["physics_step"] = 1., 1
        elif fault == "rewound":
            sample["sim_time"] = 5. - len(self.requests) * .1
        elif fault == "nonfinite":
            sample["sim_time"] = float("nan")
        else:
            now[0] += placement.WALL_BUDGET_S + 1
        return sample
    monkeypatch.setattr(client, "request", request)
    report = read()
    assert report["status"] == "unverified"
    assert clients[0].closed


def test_missing_checkout_resources_do_not_claim_success(monkeypatch):
    def missing():
        raise FileNotFoundError("optional kitchen auditor")
    monkeypatch.setattr(placement, "_auditor", missing)
    report = read()
    assert report["status"] == "unverified"
    assert report["measured"]["observed_samples"] == 0


def test_ambiguous_requested_object_is_not_replaced_with_a_result_label(monkeypatch):
    clients, _, _ = setup_reader(monkeypatch)
    report = placement.read_placement(("localhost", 8611), "/SYNTHETIC_TEST_ROBOT", "cube", "open box")
    assert report["status"] == "unverified"
    assert not clients


@pytest.mark.parametrize("status", ["confirmed", "refuted", "unverified"])
def test_configured_check_uses_requested_subject_and_destination(status):
    calls = []
    def check(label, destination):
        calls.append((label, destination))
        return {"status": status, "evidence": "independent bounded placement window",
                "measured": {"destination": "open box"}}
    checker = PostconditionChecker(placement_check=check, object_pose=lambda name: [0., 0., 0.])
    result = checker.verify("pick_and_place", {"object": "orange", "destination": "open box"},
        {"ok": True, "picked": "green cube", "destination": "open box", "destination_kind": "configured_point"},
        {"label": "orange", "pose": [0., 0., 0.], "channel": "physics"})
    assert calls == [("orange", "open box")]
    assert result.status == status  # no false failure from a repeated/no-displacement placement
    assert result.channel == "physics"


def test_reported_destination_cannot_replace_requested_destination():
    checker = PostconditionChecker(placement_check=lambda *args: {
        "status": "confirmed", "evidence": "at requested square", "measured": {"destination": "green square"}})
    pc = checker.verify("pick_and_place", {"object": "orange", "destination": "green square"},
        {"ok": True, "destination": "open box", "destination_kind": "configured_point"})
    assert pc.status == "unverified"
    assert "destinations differ" in pc.evidence


def test_motion_result_cannot_supply_its_own_physical_verdict():
    checker = PostconditionChecker(object_pose=lambda name: [.3, -.14, .03])
    pc = checker.verify("pick_and_place", {"object": "orange", "destination": "open box"},
        {"ok": True, "destination": "open box", "destination_kind": "configured_point",
         "placed_at": [.3, -.14, .03], "placement_verdict": {"status": "confirmed"}})
    assert pc.status == "unverified"
