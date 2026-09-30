"""Newton GPU attestation: evidence comes from the Newton stage, not the PhysX-only view context.

Newton's SimulationView reports `cuda_context == 0` even on a GPU run, so the
PhysX-style check alone rejected every Newton bridge under
CASCADE_REQUIRE_CUDA=1. The attestation now reads the Warp device holding the
live model/state/contact arrays and the MuJoCo solver backend.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from isaac_runtime import physics_device_identity  # noqa: E402


class _Dev:
    def __init__(self, alias):
        self.alias = alias
        self.is_cuda = alias.startswith("cuda:")
        self.ordinal = int(alias.split(":")[1]) if self.is_cuda else -1
        self.context = 0xABC if self.is_cuda else None
        self.pci_bus_id = "00000000:21:00" if self.is_cuda else None
        self.uuid = "GPU-test" if self.is_cuda else None


@pytest.fixture
def fake_warp(monkeypatch):
    mod = types.SimpleNamespace(get_device=lambda d: _Dev(str(d)))
    monkeypatch.setitem(sys.modules, "warp", mod)
    return mod


def _arr(dev):
    return types.SimpleNamespace(device=dev)


def _stage(model_dev="cuda:0", array_dev="cuda:0", cpu=False, solver=True):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(device=model_dev),
        state_0=types.SimpleNamespace(body_q=_arr(array_dev), joint_q=_arr(array_dev)),
        contacts=types.SimpleNamespace(rigid_contact_count=_arr(array_dev)),
        solver=types.SimpleNamespace(use_mujoco_cpu=cpu) if solver else None,
        graph=object())


class _Manager:
    def __init__(self, engine="newton", device="cuda:0"):
        self.engine, self.device = engine, device

    def get_active_physics_engine(self):
        return self.engine

    def get_device(self):
        return self.device

    def get_physics_scenes(self):
        return [object()]

    def get_physics_simulation_view(self):
        # What Newton's view really reports: device + ordinal, no context handle.
        return types.SimpleNamespace(is_valid=True, device=self.device,
                                     device_ordinal=int(self.device.split(":")[1]) if ":" in self.device else -1,
                                     cuda_context=0)


def test_newton_on_cuda_is_attested(fake_warp):
    out = physics_device_identity(_Manager(), require_cuda=True, newton_stage=_stage())
    att = out["gpu_attestation"]
    assert out["physics_gpu"] is True
    assert att["backend"] == "newton" and att["cuda_context_present"] is True
    assert att["newton"]["array_devices"] == ["cuda:0"] and att["newton"]["mujoco_cpu"] is False


@pytest.mark.parametrize("stage,why", [
    (_stage(cpu=True), "MuJoCo CPU backend"),
    (_stage(model_dev="cpu", array_dev="cpu"), "model on the CPU"),
    (_stage(array_dev="cuda:1"), "arrays on another GPU than the tensor view"),
    (_stage(solver=False), "no solver"),
    (None, "no Newton stage to inspect"),
])
def test_newton_without_gpu_evidence_is_refused(fake_warp, monkeypatch, stage, why):
    if stage is None:  # the extension import must not silently certify anything
        monkeypatch.setitem(sys.modules, "isaacsim.physics.newton.impl.extension",
                            types.SimpleNamespace(_newton_stage=None))
    with pytest.raises(RuntimeError, match="GPU physics required"):
        physics_device_identity(_Manager(), require_cuda=True, newton_stage=stage)
    out = physics_device_identity(_Manager(), require_cuda=False, newton_stage=stage)
    assert out["physics_gpu"] is False, why


def test_physx_path_does_not_consult_newton(fake_warp):
    class _Scene:
        def get_enabled_gpu_dynamics(self):
            return True

        def get_broadphase_type(self):
            return "GPU"

    m = _Manager(engine="physx")
    m.get_physics_scenes = lambda: [_Scene()]
    view = m.get_physics_simulation_view()
    view.cuda_context = 0x1
    m.get_physics_simulation_view = lambda: view
    out = physics_device_identity(m, require_cuda=True, newton_stage=_stage(cpu=True))
    assert out["physics_gpu"] is True and "newton" not in out["gpu_attestation"]
