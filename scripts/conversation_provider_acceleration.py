"""Passive admission helpers for the explicit LLM-only CUDA recipe.

Importing this module never imports Torch or initializes an accelerator.
"""
from __future__ import annotations

import os
import re
import math
from time import monotonic


PROFILE = "cuda-llm-fp32"
TORCH_VERSION = "2.11.0+cu130"
CUDA_VERSION = "13.0"
STARTUP_MAX_S = 150.0


def startup_deadline(value, *, create=False, clock=None):
    """Validate one same-host monotonic deadline; never renew an inherited one."""
    clock = monotonic if clock is None else clock
    now = clock()
    if value is None and create:
        value = now + STARTUP_MAX_S
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("startup requires a finite monotonic deadline")
    if not 0 < value - now <= STARTUP_MAX_S:
        raise ValueError("startup deadline expired or exceeds the original 150 seconds")
    return float(value)


def attest_startup(llm, deadline):
    """Admit the actual two completed warmups against the inherited deadline."""
    deadline = startup_deadline(deadline)
    if getattr(llm, "startup_deadline_monotonic_s", None) != deadline:
        raise ValueError("handler startup deadline differs from its owner")
    timeout = getattr(getattr(llm, "streamer", None), "timeout", None)
    if type(timeout) not in (int, float) or timeout != 10.0:
        raise ValueError("actual conversational streamer timeout must remain 10 seconds")
    rows = getattr(llm, "warmup_diagnostics", None)
    if type(rows) is not list or len(rows) != 2:
        raise ValueError("two actual completed warmup records are required")
    observed = []
    previous_end = None
    for index, row in enumerate(rows):
        if type(row) is not dict or type(row.get("index")) is not int or row["index"] != index:
            raise ValueError("warmup record sequence changed")
        if (row.get("startup_deadline_monotonic_s") != deadline
                or row.get("completed") is not True or row.get("stream_exhausted") is not True
                or row.get("producer_started") is not True or row.get("producer_alive") is not False
                or row.get("producer_start_uncertain") is not False
                or any(row.get(key) is not None for key in (
                    "error_type", "producer_error_type", "cleanup_error_type", "observation_error_type"))):
            raise ValueError("warmup did not complete under the original startup deadline")
        times = {}
        for key in ("start_monotonic_s", "end_monotonic_s", "work_deadline_monotonic_s",
                    "cleanup_deadline_monotonic_s"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("warmup needs actual finite monotonic phase times")
            times[key] = float(value)
        start, end = times["start_monotonic_s"], times["end_monotonic_s"]
        if (not deadline - STARTUP_MAX_S <= start < deadline - 5.0
                or not start <= end <= min(deadline, monotonic())
                or (previous_end is not None and start < previous_end)
                or times["work_deadline_monotonic_s"] != deadline - 5.0
                or not end <= times["cleanup_deadline_monotonic_s"] <= deadline):
            raise ValueError("warmup phase timing exceeds the original startup allowance")
        counts = {}
        for key in ("decoded_chunks", "decoded_characters"):
            value = row.get(key)
            if type(value) is not int or value < 0:
                raise ValueError("invalid actual warmup decoded count")
            counts[key] = value
        observed.append({"index": index, **times, **counts, "completed": True,
                         "stream_exhausted": True, "producer_alive": False})
        previous_end = end
    startup_deadline(deadline)
    return {"clock": "time.monotonic", "deadline_monotonic_s": deadline,
            "warmup_rows": observed, "conversation_streamer_wait_s": float(timeout),
            "scope": "Startup-only allowance; not a speech or physical task deadline."}


def uuid_value(value):
    if type(value) is not str or re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value
    ) is None:
        raise ValueError("one complete CUDA GPU UUID is required")
    return "GPU-" + value[4:].lower()


def torch_uuid(value):
    """Torch 2.11 exposes CUuuid, whose __str__ is the bare UUID (no GPU-).

    Keep this conversion at the trusted Torch property boundary. CLI and
    environment values still require full GPU-prefixed strings via uuid_value.
    """
    bare = str(value)
    if re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", bare
    ) is None:
        raise ValueError("unexpected Torch CUuuid representation")
    return uuid_value("GPU-" + bare)


def configuration(cfg):
    value = cfg.get("acceleration")
    if value is None:
        return None
    expected = {
        "profile": PROFILE, "llm_device": "cuda", "stt_device": "cpu", "tts_device": "cpu",
        "torch_version": TORCH_VERSION, "torchaudio_version": TORCH_VERSION,
        "cuda_version": CUDA_VERSION, "compute_capability": [12, 0],
        "required_arch": "sm_120", "minimum_free_bytes": 16 * 1024**3,
        "float32_matmul_precision": "highest", "allow_tf32": False,
        "llm_load_strategy": "direct-single-cuda",
        "accelerate_version": "1.15.0", "psutil_version": "7.2.2",
    }
    if value != expected or cfg.get("llm_torch_dtype") != "float32":
        raise ValueError("unreviewed CUDA acceleration recipe")
    return value


def admit_cuda(torch, cfg, expected_uuid):
    """Run only in the new serving child, before any model constructor."""
    selected = configuration(cfg)
    if selected is None:
        raise ValueError("explicit CUDA recipe required")
    expected_uuid = uuid_value(expected_uuid)
    if uuid_value(os.environ.get("CUDA_VISIBLE_DEVICES")) != expected_uuid:
        raise ValueError("CUDA visibility differs from the selected UUID")
    if str(torch.__version__) != TORCH_VERSION or torch.version.cuda != CUDA_VERSION:
        raise ValueError("CUDA Torch build differs from the recipe")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible CUDA GPU is required")
    if torch.cuda.current_device() != 0:
        raise ValueError("selected GPU must be logical CUDA device zero")
    properties = torch.cuda.get_device_properties(0)
    if torch_uuid(properties.uuid) != expected_uuid:
        raise ValueError("actual Torch GPU UUID differs from admission")
    capability = tuple(torch.cuda.get_device_capability(0))
    architectures = list(torch.cuda.get_arch_list())
    if capability != (12, 0) or "sm_120" not in architectures:
        raise ValueError("this explicit recipe requires a Blackwell sm_120 binary and GPU")
    free, total = torch.cuda.mem_get_info(0)
    if free < selected["minimum_free_bytes"]:
        raise ValueError("insufficient free GPU memory before model construction")
    # Keep FP32 arithmetic explicit. No autocast, quantization, compilation,
    # attention replacement or sampling change is introduced here.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify_precision(torch)
    return {"uuid": expected_uuid, "logical_device": "cuda:0", "name": properties.name,
            "compute_capability": list(capability), "compiled_architectures": architectures,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "free_bytes_before_models": free, "total_bytes": total,
            "minimum_free_bytes": selected["minimum_free_bytes"],
            "float32_matmul_precision": "highest", "allow_tf32": False}


def verify_precision(torch):
    if (torch.get_float32_matmul_precision() != "highest"
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("declared FP32 arithmetic was changed")


def model_observation(model, *, device, float32=False):
    """Inspect actual parameters/buffers without changing or copying tensors."""
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    if not parameters:
        raise ValueError("cannot attest an empty model")
    tensors = parameters + buffers
    if any(str(value.device) != device for value in tensors):
        raise ValueError("model parameter or buffer is on an unadmitted device")
    if float32 and (any(str(value.dtype) != "torch.float32" for value in parameters)
                    or any(value.is_floating_point() and str(value.dtype) != "torch.float32" for value in buffers)):
        raise ValueError("LLM floating parameters and buffers must remain FP32")
    return {"parameter_dtypes": sorted({str(value.dtype) for value in parameters}),
            "buffer_dtypes": sorted({str(value.dtype) for value in buffers}),
            "parameter_devices": sorted({str(value.device) for value in parameters}),
            "buffer_devices": sorted({str(value.device) for value in buffers}),
            "parameter_count": sum(value.numel() for value in parameters),
            "buffer_count": sum(value.numel() for value in buffers)}


def attest_direct_loader(llm):
    """Report the bound handler's opt-in and actual pipeline placement after warmup."""
    if getattr(llm, "direct_cuda_load", None) is not True or str(llm.pipe.device) != "cuda:0":
        raise ValueError("direct CUDA handler or pipeline placement was not retained")
    mapping = getattr(llm.model, "hf_device_map", None)
    if mapping is not None:
        if type(mapping) is not dict or set(mapping) != {""} or str(mapping[""]) != "cuda:0":
            raise ValueError("direct CUDA model gained offload or a different device map")
        mapping = {"": "cuda:0"}
    return {"direct_cuda_load": True, "pipeline_device": "cuda:0", "hf_device_map": mapping,
            "scope": "Bound source requests one device; no automatic dispatch or CPU/disk offload."}


def attest_pipeline(torch, cfg, expected_uuid, *, llm, stt, tts):
    """After upstream construction/warmup, refuse any placement drift."""
    if configuration(cfg) is None:
        raise ValueError("explicit CUDA recipe required")
    if (torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0
            or torch_uuid(torch.cuda.get_device_properties(0).uuid) != uuid_value(expected_uuid)):
        raise ValueError("CUDA identity changed during pipeline construction")
    verify_precision(torch)
    return {"llm": model_observation(llm.model, device="cuda:0", float32=True),
            "stt": model_observation(stt.model, device="cpu"),
            "tts": model_observation(tts.pipeline.model, device="cpu")}
