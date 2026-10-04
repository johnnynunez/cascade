"""Passive admission helpers for the explicit LLM-only CUDA recipe.

Importing this module never imports Torch or initializes an accelerator.
"""
from __future__ import annotations

import os
import re


PROFILE = "cuda-llm-fp32"
TORCH_VERSION = "2.11.0+cu130"
CUDA_VERSION = "13.0"


def uuid_value(value):
    if type(value) is not str or re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value
    ) is None:
        raise ValueError("one complete CUDA GPU UUID is required")
    return "GPU-" + value[4:].lower()


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
    if uuid_value(properties.uuid) != expected_uuid:
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


def attest_pipeline(torch, cfg, expected_uuid, *, llm, stt, tts):
    """After upstream construction/warmup, refuse any placement drift."""
    if configuration(cfg) is None:
        raise ValueError("explicit CUDA recipe required")
    if (torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0
            or uuid_value(torch.cuda.get_device_properties(0).uuid) != uuid_value(expected_uuid)):
        raise ValueError("CUDA identity changed during pipeline construction")
    verify_precision(torch)
    return {"llm": model_observation(llm.model, device="cuda:0", float32=True),
            "stt": model_observation(stt.model, device="cpu"),
            "tts": model_observation(tts.pipeline.model, device="cpu")}
