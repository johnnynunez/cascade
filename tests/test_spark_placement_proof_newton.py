"""The kitchen placement proof accepts a Newton (MJWarp) run under the same physical contract.

Synthetic adversarial fixtures only, like tests/test_spark_placement_proof.py:
the PhysX trajectory is rewritten as a Newton sample (engine, attestation,
contact channel, convex support method). Every physical requirement is
unchanged; the only thing that differs is where each engine's evidence is
read from. Mixing engines, a MuJoCo CPU backend, or a channel that does not
belong to the sample's engine must all fail.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spark_placement_for_newton", ROOT / "tests/test_spark_placement_proof.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
proof, gpu = base.proof, base.gpu


def as_newton(rows):
    rows = deepcopy(rows)
    for row in rows:
        s = row["physics"]
        s["engine"] = "newton"
        s["physics_clock"] = "newton_stage"
        s["gpu_attestation"].update(backend="newton", gpu_dynamics=None, broadphase=[], newton={
            "device": "cuda:0", "cuda": True, "ordinal": 0, "cuda_context_present": True,
            "array_devices": ["cuda:0"], "solver": "SolverMuJoCo", "mujoco_cpu": False, "cuda_graph": True})
        s["contacts"]["channel"] = "newton_mjwarp_contact_force"
        support = s.get("scene_geometry", {}).get("convex_collider", {}).get("physx_support")
        if support is not None:
            support.update(method="newton_model_shape_source", engine="newton")
    return rows


@pytest.mark.parametrize("object_name,destination", proof.CASES)
def test_both_cases_pass_on_newton_evidence(object_name, destination):
    rows, expected = base.trajectory(object_name, destination)
    result = base.audit(as_newton(rows), expected, object_name, destination)
    assert result["pass"], result
    assert result["checks"]["single_supported_engine"]
    assert result["checks"]["gpu_backend_no_fallback"]
    assert result["checks"]["two_jaw_contacts_during_real_lift"]
    assert result["checks"]["whole_footprint_enters_destination_with_bilateral_support"]
    assert result["checks"]["all_props_physically_reset_and_settled"]


@pytest.mark.parametrize("object_name,destination", proof.CASES)
def test_physx_evidence_still_passes_unchanged(object_name, destination):
    rows, expected = base.trajectory(object_name, destination)
    result = base.audit(rows, expected, object_name, destination)
    assert result["pass"], result
    assert result["checks"]["single_supported_engine"]


@pytest.mark.parametrize("fault", ["mujoco_cpu", "arrays_elsewhere", "physx_channel_on_newton",
                                   "newton_channel_on_physx", "mixed_engines", "physx_support_on_newton",
                                   "no_newton_block", "unknown_engine"])
def test_engine_evidence_must_belong_to_the_sample_engine(fault):
    object_name, destination = "orange", "open box"
    rows, expected = base.trajectory(object_name, destination)
    if fault == "newton_channel_on_physx":
        for row in rows:
            row["physics"]["contacts"]["channel"] = "newton_mjwarp_contact_force"
    else:
        rows = as_newton(rows)
    for i, row in enumerate(rows):
        s = row["physics"]
        if fault == "mujoco_cpu":
            s["gpu_attestation"]["newton"]["mujoco_cpu"] = True
        elif fault == "arrays_elsewhere":
            s["gpu_attestation"]["newton"]["array_devices"] = ["cpu"]
        elif fault == "physx_channel_on_newton":
            s["contacts"]["channel"] = "physx_gpu_contact_tensor"
        elif fault == "mixed_engines" and i % 2:
            s["engine"] = "physx"
        elif fault == "physx_support_on_newton":
            s["scene_geometry"]["convex_collider"]["physx_support"].update(
                method="physx_collision_representation", engine="physx")
        elif fault == "no_newton_block":
            s["gpu_attestation"].pop("newton")
        elif fault == "unknown_engine":
            s["engine"] = s["gpu_attestation"]["backend"] = "bullet"
    result = base.audit(rows, expected, object_name, destination)
    assert not result["pass"], fault
