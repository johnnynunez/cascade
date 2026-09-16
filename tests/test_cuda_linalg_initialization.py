"""Cold concurrent geometry callers must not race PyTorch's lazy CUDA loader."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys
import threading

import pytest

from cascade.perception import cuda_math


@pytest.fixture
def isolated_context(monkeypatch):
    cuda_math.context.cache_clear()
    monkeypatch.setattr(cuda_math, "_linalg_ready_devices", set())
    monkeypatch.setattr(cuda_math, "_linalg_init_lock", threading.Lock())
    from cascade import device
    monkeypatch.setattr(device, "resolve_device", lambda *args, **kwargs: "cuda:0")
    yield
    cuda_math.context.cache_clear()


def fake_torch(monkeypatch, eigensolver, synchronize=lambda *_: None):
    device = SimpleNamespace(type="cuda")
    allocation = SimpleNamespace(device=device)
    calls = []

    def eye(size, *, dtype, device):
        calls.append((size, dtype, device.type))
        return allocation

    torch = SimpleNamespace(device=lambda *_: device, float64="float64",
        version=SimpleNamespace(cuda="test-cuda", hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=synchronize),
        eye=eye, linalg=SimpleNamespace(eigh=eigensolver),
        as_tensor=lambda *args, **kwargs: allocation)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_two_cold_tensor_callers_wait_for_one_completed_cuda_initialization(monkeypatch, isolated_context):
    entered, finish = threading.Event(), threading.Event()
    initialized, synchronized = [], []

    def eigh(probe):
        assert probe.device.type == "cuda"
        initialized.append(True)
        if len(initialized) != 1:
            raise RuntimeError("lazy wrapper should be called at most once")
        entered.set()
        assert finish.wait(2)

    calls = fake_torch(monkeypatch, eigh, lambda dev: synchronized.append(dev.type))
    barrier = threading.Barrier(3)

    def caller():
        barrier.wait(2)
        result = cuda_math.tensor([1, 2, 3])
        assert synchronized == ["cuda"], "readiness escaped before the GPU operation completed"
        return result.device.type

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(caller) for _ in range(2)]
        barrier.wait(2)
        try:
            assert entered.wait(2)
            assert not any(f.done() for f in futures)
        finally:
            finish.set()
        assert [f.result(timeout=2) for f in futures] == ["cuda", "cuda"]
    assert initialized == [True]
    assert calls == [(3, "float64", "cuda")]
    assert cuda_math.tensor([4]).device.type == "cuda"
    assert initialized == [True]


def test_failed_cuda_initialization_is_not_cached_or_converted_to_cpu(monkeypatch, isolated_context):
    attempts = []

    def eigh(probe):
        assert probe.device.type == "cuda"
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("CUDA solver initialization failed")

    fake_torch(monkeypatch, eigh)
    with pytest.raises(RuntimeError, match="CUDA solver initialization failed"):
        cuda_math.tensor([1])
    assert not cuda_math._linalg_ready_devices
    assert cuda_math.context.cache_info().currsize == 0
    assert cuda_math.tensor([1]).device.type == "cuda"
    assert attempts == [True, True]
