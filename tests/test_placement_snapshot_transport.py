"""Placement snapshots must cross the Isaac bridge's exec stdout intact on both engines.

CPU only; never live acceptance. A recorded PhysX placement receipt (demo video v2,
main c9f8465, verdict confirmed) and a Newton-shaped copy of it at the measured live
Newton size are replayed through the REAL generated snapshot code. Each reply gets the
same stdout tail cut as ``scripts/isaac_bridge.py`` ``_run_exec_jobs``, then goes through
the real ``placement.read_placement`` reader and verdict. The verdict must equal the
verdict computed directly on the same physics: the transport may not change a decision.
"""
from __future__ import annotations

import base64
import contextlib
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import random
import re
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.sim import placement
from cascade.sim.bridge_client import BridgeError
from test_kitchen_placement_verdict import settled


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/isaac_placement_physx_c9f8465.json.gz"
#: Plain snapshot of the live Newton kitchen placement check (demo video v3).
NEWTON_LIVE_PLAIN_BYTES = 7648
#: The observer's single-line guard below the bridge's exec stdout tail.
LINE_GUARD_BYTES = 7400
#: Newton GPU attestation exactly as the live Newton bridge reported it (v3 `ping`).
NEWTON_ATTESTATION = {
    "required": True, "backend": "newton", "device": "cuda:0", "tensor_device": "cuda:0",
    "tensor_device_ordinal": 0, "cuda_context_present": True, "gpu_dynamics": None, "broadphase": [],
    "cpu_fallback_allowed": False, "newton": {
        "device": "cuda:0", "cuda": True, "ordinal": 0, "cuda_context_present": True,
        "pci_bus_id": "0000000F:01:00", "uuid": "GPU-351eae57-e78f-55ed-a08a-686a5713086f",
        "array_devices": ["cuda:0"], "solver": "SolverMuJoCo", "mujoco_cpu": False, "cuda_graph": True},
    "fallback_log_count": 0}
#: `_newton_contact_snapshot` in scripts/isaac_bridge.py adds this key on Newton.
NEWTON_FORCE_SOURCE = "MJWarp efc.force via SolverMuJoCo.update_contacts"
#: The generated line that serializes the GPU proof sample; the wire tail follows it.
PAYLOAD_LINE = '_obs_payload = _obs_json.dumps(_gpu_observed, separators=(",",":"), allow_nan=False)'
READER_ONLY = ("observed_samples", "observation_wall_s", "evidence_path", "evidence_sha256")


def recorded_receipt():
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))["receipt"]


def plain_bytes(sample):
    return len(json.dumps(sample, separators=(",", ":"), allow_nan=False).encode())


def newton_clock(step, dt=1 / 120):
    """The Newton stage accumulates dt once per step (full-precision sim_time)."""
    clock = 0.
    for _ in range(step):
        clock += dt
    return clock


def as_newton_sized(samples):
    """Rewrite recorded PhysX samples the way the live Newton bridge reports them.

    Engine, clock, attestation and contact channel follow the Newton bridge. MJWarp
    bodies never sleep, so the two props PhysX reported exactly at rest carry
    full-precision residual motion, as each prop of the live Newton receipt did
    (385-394 B per prop), and the contact net force on the cube (cube-table
    contacts included) has residual horizontal terms. The verdict's inputs for
    the target cube (pose, velocities, jaws, contact step and jaw forces) are
    untouched.
    """
    rng = np.random.default_rng(7648)
    samples = deepcopy(samples)
    for sample in samples:
        sample.update(engine="newton", physics_clock="newton_stage", sim_time=newton_clock(sample["physics_step"]),
                      gpu_attestation=deepcopy(NEWTON_ATTESTATION))
        sample["contacts"].update(channel="newton_mjwarp_contact_force", force_source=NEWTON_FORCE_SOURCE)
        sample["contacts"]["net_force_n"][:2] = rng.normal(scale=1e-4, size=2).tolist()
        for name in ("lemon", "orange"):
            prop = sample["props"][name]
            assert prop["orientation_wxyz"] == [1., 0., 0., 0.]  # PhysX: exactly at rest
            tilt = np.array([1., *rng.normal(scale=1e-7, size=3)])
            prop["orientation_wxyz"] = (tilt / np.linalg.norm(tilt)).tolist()
            prop["linear_velocity_m_s"] = rng.normal(scale=1e-5, size=3).tolist()
            prop["angular_velocity_rad_s"] = rng.normal(scale=1e-4, size=3).tolist()
    return samples


def engine_samples(engine):
    samples = [row["physics"] for row in recorded_receipt()["records"]]
    return as_newton_sized(samples) if engine == "newton" else samples


def bridge_stdout_chars():
    """`_run_exec_jobs` returns only the LAST N characters of a successful job's stdout."""
    found = re.findall(r'\{\*\*resp, "stdout": buf\.getvalue\(\)\[-(\d+):\]\}',
                       (ROOT / "scripts/isaac_bridge.py").read_text())
    assert len(found) == 1, "bridge exec stdout contract changed; review the observer guards"
    return int(found[0])


def bridge_exec(code, sample):
    """Run the generated wire tail like the bridge's exec queue: captured, tail-cut stdout."""
    assert code.count(PAYLOAD_LINE) == 1, "snapshot payload boundary changed"
    tail = PAYLOAD_LINE + code.split(PAYLOAD_LINE, 1)[1]
    namespace = {"_gpu_observed": deepcopy(sample), "_obs_json": json, "_obs_np": np,
                 "_obs_base64": base64}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(tail, namespace)
    except Exception:
        return {"ok": False, "error": traceback.format_exc()[-2000:], "stdout": stdout.getvalue()[-2000:]}
    return {"ok": True, "stdout": stdout.getvalue()[-bridge_stdout_chars():]}


def verifier_code(expected, object_name="green_cube"):
    """The snapshot code `read_placement` sends (same arguments as src/cascade/sim/placement.py)."""
    return placement._auditor().gpu.snapshot_code(object_name=object_name, include_scene_geometry=True,
        include_open_box="open_box" in expected, expected_props=tuple(expected["prop_dimensions_m"]))


def replay(monkeypatch, samples, expected):
    """Serve each exec request from the next sample through the real generated code.

    The reviewed scene geometry comes from the receipt, so a later kitchen edit
    cannot turn this transport test into a scene-identity test.
    """
    feed = iter(samples)
    now = [100.]
    clients = []

    class Client:
        def __init__(self, **kwargs):
            self.requests = []
            clients.append(self)

        def connect(self):
            pass

        def close(self):
            pass

        def request(self, request, timeout_s):
            assert set(request) == {"op", "code"} and request["op"] == "exec"
            self.requests.append(request)
            reply = bridge_exec(request["code"], next(feed))
            if not reply["ok"]:  # BridgeClient.request raises on an `ok: false` reply
                raise BridgeError(reply["error"])
            return reply

    real = placement._auditor()
    gpu = SimpleNamespace(load_expected_scene_geometry=lambda path: deepcopy(expected),
                          snapshot_code=real.gpu.snapshot_code, original=real.gpu.original)
    monkeypatch.setattr(placement, "_auditor", lambda: SimpleNamespace(gpu=gpu, audit_placement=real.audit_placement))
    monkeypatch.setattr(placement, "BridgeClient", Client)
    monkeypatch.setattr(placement.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(placement.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    return clients


def test_fixtures_reproduce_the_live_plain_snapshot_sizes():
    physx, newton = engine_samples("physx"), engine_samples("newton")
    assert len(physx) == len(newton) == 32
    # PhysX fit under the single-line guard (v2 confirmed); Newton did not (v3 unverified,
    # 7,648 B measured live). Every Newton-shaped sample is over the guard; the largest
    # is at least the live size.
    assert max(map(plain_bytes, physx)) < LINE_GUARD_BYTES
    assert min(map(plain_bytes, newton)) > LINE_GUARD_BYTES
    assert max(map(plain_bytes, newton)) >= NEWTON_LIVE_PLAIN_BYTES
    assert all(len(s["props"]) == 5 for s in physx + newton)
    assert bridge_stdout_chars() > LINE_GUARD_BYTES


@pytest.mark.parametrize("engine", ["physx", "newton"])
def test_every_recorded_sample_crosses_the_bridge_bit_exact(engine):
    receipt = recorded_receipt()
    code = verifier_code(receipt["expected_scene_geometry"])
    for sample in engine_samples(engine):
        reply = bridge_exec(code, sample)
        assert reply["ok"], reply["error"]
        (line,) = reply["stdout"].splitlines()
        assert len(line.encode()) < LINE_GUARD_BYTES
        assert placement._auditor().gpu.original.parse_snapshot_reply(reply) == sample


@pytest.mark.parametrize("engine", ["physx", "newton"])
@pytest.mark.parametrize("case,status", [
    ("as_recorded", "confirmed"), ("outside_square", "refuted"), ("gpu_fallback", "unverified")])
def test_read_placement_verdict_equals_the_direct_verdict(monkeypatch, tmp_path, engine, case, status):
    receipt = recorded_receipt()
    expected = receipt["expected_scene_geometry"]
    samples = engine_samples(engine)
    for sample in samples:
        if case == "outside_square":
            sample["props"]["green_cube"]["position_m"][0] += .06
        elif case == "gpu_fallback":
            sample["gpu_attestation"]["fallback_log_count"] = 1
    clients = replay(monkeypatch, samples, expected)
    report = placement.read_placement(("127.0.0.1", 9), receipt["robot_id"], receipt["requested_object"],
                                      receipt["destination"], evidence_dir=tmp_path)
    assert report["status"] == status, report["evidence"]

    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    used = [row["physics"] for row in saved["records"]]
    assert len(used) >= 6 and len(clients) == 1 and len(clients[0].requests) == len(used)
    assert used == samples[:len(used)]  # lossless: the verdict saw the recorded physics
    direct = placement._auditor().audit_placement(
        [{"physics": s} for s in samples[:len(used)]], object_name="green_cube",
        destination_name=receipt["destination"], expected_scene_geometry=expected, robot_id=receipt["robot_id"])
    direct["measured"]["destination"] = receipt["destination"]
    for key in READER_ONLY:
        report["measured"].pop(key, None)
    assert report == direct


def test_recorded_verdict_is_reproduced_through_the_bridge(monkeypatch, tmp_path):
    receipt = recorded_receipt()
    replay(monkeypatch, engine_samples("physx"), receipt["expected_scene_geometry"])
    report = placement.read_placement(("127.0.0.1", 9), receipt["robot_id"], receipt["requested_object"],
                                      receipt["destination"], evidence_dir=tmp_path)
    recorded = deepcopy(receipt["verdict"])
    for verdict in (report, recorded):
        for key in READER_ONLY:
            verdict["measured"].pop(key, None)
    assert report == recorded


@pytest.mark.parametrize("engine", ["physx", "newton"])
@pytest.mark.parametrize("object_name,destination", [
    ("orange", "open box"), ("orange", "green square"), ("pink_cube", "green square"),
    ("green_cube", "open box"), ("lemon", "open box"), ("tomato_can", "green square")])
def test_synthetic_tails_keep_their_verdict_for_every_target(engine, object_name, destination):
    rows, expected = settled(object_name, destination, engine)
    code = verifier_code(expected, object_name)
    decoded = []
    for row in rows:
        reply = bridge_exec(code, row["physics"])
        assert reply["ok"], reply["error"]
        decoded.append({"physics": placement._auditor().gpu.original.parse_snapshot_reply(reply)})
    assert decoded == [{"physics": row["physics"]} for row in rows]
    verdict = lambda records: placement._auditor().audit_placement(records, object_name=object_name,
        destination_name=destination, expected_scene_geometry=expected, robot_id="/SYNTHETIC_TEST_ROBOT")
    assert verdict(decoded) == verdict(rows)


def test_compressed_receipt_still_refuses_a_line_over_the_guard():
    receipt = recorded_receipt()
    sample = deepcopy(engine_samples("newton")[0])
    noise = random.Random(7400)
    sample["frame"]["jpeg_sha256"] = "".join(noise.choice("0123456789abcdef") for _ in range(12000))
    reply = bridge_exec(verifier_code(receipt["expected_scene_geometry"]), sample)
    assert not reply["ok"]
    assert "RuntimeError: Observer compressed snapshot exceeds safe bridge stdout size" in reply["error"]


def test_compressed_receipt_refuses_an_oversized_decoded_snapshot():
    receipt = recorded_receipt()
    sample = deepcopy(engine_samples("newton")[0])
    sample["frame"]["jpeg_sha256"] = "0" * 70000  # compresses far below the line guard
    reply = bridge_exec(verifier_code(receipt["expected_scene_geometry"]), sample)
    assert not reply["ok"]
    assert "RuntimeError: Observer decoded snapshot exceeds 65536 bytes" in reply["error"]


def test_snapshot_without_scene_geometry_keeps_the_plain_receipt():
    code = placement._auditor().gpu.snapshot_code(object_name="pink_cube")
    assert "print('KITCHEN_OBSERVER ' + _obs_payload)" in code
    assert "KITCHEN_OBSERVER_ZLIB" not in code
