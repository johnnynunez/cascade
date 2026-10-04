"""Admission and CLI tests; no Torch import, model construction or CUDA work."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from test_conversation_provider_recipe import host  # noqa: F401

GPU = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER = "GPU-11111111-2222-3333-4444-555555555555"


@pytest.fixture
def selected(host):  # noqa: F811
    host.select_profile("cuda-llm-fp32")
    return host


class Tensor:
    def __init__(self, device, dtype="torch.float32"):
        self.device, self.dtype = device, dtype

    def is_floating_point(self):
        return self.dtype in ("torch.float32", "torch.float16", "torch.float64")

    def numel(self):
        return 7


def model(device):
    params = [Tensor(device)]
    buffers = [Tensor(device), Tensor(device, "torch.int64")]
    return NS(parameters=lambda: params, buffers=lambda: buffers, params=params, stored=buffers)


class CUuuid:
    """Real pybind API shape: object, bare __str__, bytes as 16 unsigned ints."""
    def __init__(self, bare=GPU[4:]):
        self.bare = bare

    @property
    def bytes(self):
        return list(bytes.fromhex(self.bare.replace("-", "")))

    def __str__(self):
        return self.bare


def torch_contract():
    properties = NS(uuid=CUuuid(), name="selected GPU")
    value = NS(__version__="2.11.0+cu130", version=NS(cuda="13.0"), precision="high")
    value.cuda = NS(is_available=lambda: True, device_count=lambda: 1,
                    current_device=lambda: 0, get_device_properties=lambda i: properties,
                    get_device_capability=lambda i: (12, 0), get_arch_list=lambda: ["sm_120"],
                    mem_get_info=lambda i: (20 * 1024**3, 96 * 1024**3))
    value.backends = NS(cuda=NS(matmul=NS(allow_tf32=True)), cudnn=NS(allow_tf32=True))
    value.set_float32_matmul_precision = lambda mode: setattr(value, "precision", mode)
    value.get_float32_matmul_precision = lambda: value.precision
    return value


def test_import_and_help_are_passive(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/conversation_provider.py"
    code = """
import runpy, sys
def hook(event, args):
    if event == 'import' and args[0].split('.')[0] in {'torch', 'speech_to_speech'}:
        raise AssertionError('heavy import during help')
    if event in {'socket.connect', 'socket.bind', 'subprocess.Popen'}:
        raise AssertionError('side effect during help')
sys.addaudithook(hook)
sys.argv = [sys.argv[1], '--help']
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "cuda-llm-fp32" in result.stdout


def test_cuda_recipe_changes_only_device_environment_and_pinned_torch(host, tmp_path, monkeypatch):  # noqa: F811
    cfg = host.recipe()
    cpu = host.provider_argv(tmp_path, 18878)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", OTHER)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/foreign/cuda")
    before = dict(os.environ)
    assert host.environment(tmp_path, offline=True)["CUDA_VISIBLE_DEVICES"] == ""
    host.select_profile("cuda-llm-fp32")
    gpu = host.provider_argv(tmp_path, 18878)
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(cpu, gpu)) if a != b]
    assert differences == [(cpu.index("--llm_device") + 1, "cpu", "cuda")]
    new = host.recipe()
    assert {k: v for k, v in new.items() if k not in {"recipe", "acceleration", "torch_wheels"}} == {
        k: v for k, v in cfg.items() if k != "recipe"}
    assert host.environment(tmp_path, offline=False)["CUDA_VISIBLE_DEVICES"] == ""
    env = host.environment(tmp_path, offline=True, cuda_device_uuid=GPU)
    assert env["CUDA_VISIBLE_DEVICES"] == GPU
    assert "LD_LIBRARY_PATH" not in env
    assert os.environ == before


@pytest.mark.parametrize("value", [None, "0", "cuda:0", GPU + "," + OTHER, "MIG-" + GPU, True])
def test_gpu_profile_requires_one_full_uuid(selected, value):
    with pytest.raises(ValueError):
        selected.selected_gpu(value)


def test_uuid_canonicalization_is_idempotent(selected):
    helper = selected.acceleration
    value = helper.uuid_value(GPU.upper().replace("GPU-", "GPU-"))
    assert value == GPU and helper.uuid_value(value) == GPU


def test_cpu_refuses_gpu_selection(host):  # noqa: F811
    assert host.selected_gpu(None) is None
    with pytest.raises(ValueError, match="CPU profile"):
        host.selected_gpu(GPU)


@pytest.mark.parametrize("failure", ["visibility", "wheel", "cuda", "count", "logical", "uuid", "capability", "arch", "memory"])
def test_cuda_admission_refuses_wrong_device_before_models(selected, monkeypatch, failure):
    torch = torch_contract()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    if failure == "visibility":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", OTHER)
    elif failure == "wheel":
        torch.__version__ = "2.11.0+cpu"
    elif failure == "cuda":
        torch.version.cuda = None
    elif failure == "count":
        torch.cuda.device_count = lambda: 2
    elif failure == "logical":
        torch.cuda.current_device = lambda: 1
    elif failure == "uuid":
        torch.cuda.get_device_properties(0).uuid = CUuuid(OTHER[4:])
    elif failure == "capability":
        torch.cuda.get_device_capability = lambda i: (9, 0)
    elif failure == "arch":
        torch.cuda.get_arch_list = lambda: ["sm_90"]
    elif failure == "memory":
        torch.cuda.mem_get_info = lambda i: (16 * 1024**3 - 1, 96 * 1024**3)
    with pytest.raises(ValueError):
        selected.acceleration.admit_cuda(torch, selected.recipe(), GPU)
    assert torch.precision == "high"


@pytest.mark.parametrize("failure", [None, "llm_device", "llm_dtype", "buffer_dtype", "buffer_device", "stt", "tts", "tf32", "uuid"])
def test_actual_postwarmup_models_and_precision_are_attested(selected, monkeypatch, failure):
    torch = torch_contract()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    before = selected.acceleration.admit_cuda(torch, selected.recipe(), GPU)
    assert before["uuid"] == GPU and before["logical_device"] == "cuda:0"
    llm, stt, tts = NS(model=model("cuda:0")), NS(model=model("cpu")), NS(pipeline=NS(model=model("cpu")))
    if failure == "llm_device":
        llm.model.params[0].device = "cpu"
    elif failure == "llm_dtype":
        llm.model.params[0].dtype = "torch.float16"
    elif failure == "buffer_dtype":
        llm.model.stored[0].dtype = "torch.float16"
    elif failure == "buffer_device":
        llm.model.stored[1].device = "cpu"
    elif failure == "stt":
        stt.model.params[0].device = "cuda:0"
    elif failure == "tts":
        tts.pipeline.model.stored[0].device = "cuda:0"
    elif failure == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif failure == "uuid":
        torch.cuda.get_device_properties(0).uuid = CUuuid(OTHER[4:])
    args = (torch, selected.recipe(), GPU)
    kwargs = dict(llm=llm, stt=stt, tts=tts)
    if failure is not None:
        with pytest.raises(ValueError):
            selected.acceleration.attest_pipeline(*args, **kwargs)
    else:
        actual = selected.acceleration.attest_pipeline(*args, **kwargs)
        assert actual["llm"]["buffer_dtypes"] == ["torch.float32", "torch.int64"]
        assert actual["stt"]["parameter_devices"] == actual["tts"]["parameter_devices"] == ["cpu"]


def test_serving_propagates_profile_and_uuid_to_owned_child(selected, tmp_path, monkeypatch):
    monkeypatch.setattr(selected, "verify", lambda state: {})
    observed = {}
    def supervisor(command, **kwargs):
        observed.update(command=command, **kwargs)
        return 0
    monkeypatch.setattr(selected, "supervise", supervisor)
    import socket
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    assert selected.serve(tmp_path, tmp_path / "run", port=port, timeout_s=30,
                          trace_generation=True, cuda_device_uuid=GPU) == 0
    command = observed["command"]
    assert command[command.index("--profile") + 1] == "cuda-llm-fp32"
    assert command[command.index("--cuda-device-uuid") + 1] == GPU
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == GPU
    assert observed["timeout_s"] == 30 and "--trace-generation" in command


def test_gpu_selection_refused_before_state_verification_or_output(selected, tmp_path, monkeypatch):
    monkeypatch.setattr(selected, "verify", lambda state: pytest.fail("must refuse before reading state"))
    with pytest.raises(ValueError):
        selected.serve(tmp_path, tmp_path / "run", port=18878, timeout_s=30)
    assert not (tmp_path / "run").exists()


def test_cuda_config_cannot_change_precision_or_cpu_roles(selected):
    cfg = selected.recipe()
    for key, value in (("tts_device", "cuda"), ("allow_tf32", True), ("torch_version", "2.13.0+cu130")):
        changed = copy.deepcopy(cfg)
        changed["acceleration"][key] = value
        with pytest.raises(ValueError):
            selected.acceleration.configuration(changed)
    assert json.loads(selected.RECIPE.read_text()) == cfg


def test_actual_torch_uuid_boundary_converts_only_the_property(selected):
    value = CUuuid()
    assert type(value) is not str and len(value.bytes) == 16
    with pytest.raises(ValueError):
        selected.acceleration.uuid_value(value)  # Original strict-string failure.
    assert selected.acceleration.torch_uuid(value) == GPU
    with pytest.raises(ValueError):
        selected.selected_gpu(value)
    with pytest.raises(ValueError):
        selected.selected_gpu(str(value))  # CLI still requires GPU- prefix.


@pytest.mark.parametrize("bare", [GPU, "0", "a" * 32, GPU[4:] + " ", "{" + GPU[4:] + "}"])
def test_torch_uuid_boundary_rejects_unexpected_format(selected, bare):
    with pytest.raises(ValueError, match="CUuuid"):
        selected.acceleration.torch_uuid(CUuuid(bare))


@pytest.mark.parametrize("corrupt", [False, True])
def test_installation_hashes_both_wheels_before_invoking_package_manager(selected, tmp_path, monkeypatch, corrupt):
    calls = []
    def download(url, path, expected):
        calls.append((url, path.name, expected))
        if corrupt and len(calls) == 2:
            raise ValueError("download hash mismatch")
    monkeypatch.setattr(selected, "download", download)
    monkeypatch.setattr(selected.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    env = selected.environment(tmp_path, offline=False)
    if corrupt:
        with pytest.raises(ValueError, match="hash mismatch"):
            selected.install_torch("uv", tmp_path / "python", tmp_path, env)
        assert len(calls) == 2  # No partial environment install.
    else:
        selected.install_torch("uv", tmp_path / "python", tmp_path, env)
        assert calls[:2] == [(pin["url"], name, pin["sha256"])
                            for name, pin in selected.recipe()["torch_wheels"].items()]
        cmd, kwargs = calls[2]
        assert cmd[:5] == ["uv", "pip", "install", "--python", str(tmp_path / "python")]
        assert cmd[5:7] == ["--constraint", str(selected.REQUIREMENTS)]
        assert cmd[7:] == [str(tmp_path / name) for name in selected.recipe()["torch_wheels"]]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" and kwargs["check"] is True
