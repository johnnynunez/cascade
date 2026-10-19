#!/usr/bin/env python3
"""Pinned Cosmos3-Edge HF reasoner on ordinary vLLM, NOT vLLM-Omni.

This process serves a model; it never starts cascade's in-process agent loop.
Heavy imports are deliberately lazy so --help/--dry-run work without CUDA.
Omni's Cosmos3 diffusion pipeline is a different deployment and does not
register this reasoner-only export for native chat/tool calling.

Stack: vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, in one venv per
vLLM version and torch build (default <checkout>/.cosmos-vllm-0.32.0-cu130).
An upgrade builds a fresh venv beside the old one instead of mutating it, so
the previous stack stays runnable; a VENV holding another vLLM is refused.

KV cache: NVFP4 by default on the FLASHINFER attention backend (vLLM 0.32.0
on SM12x); KV_DTYPE=fp8 or KV_DTYPE=auto (BF16) restore the other caches.
Measured on an RTX PRO 6000 (docs/evidence/cosmos-vllm-032-nvfp4-2026-10-19)
at GPU_FRAC 0.20: NVFP4 holds 3.6x the KV tokens of BF16 (440,672 vs
123,936), recalls a passkey at 16k tokens and keeps OpenClaw-shaped tool
calls at 8/8. The cost is accuracy on long prompts: the built-in agent's
12k-token, 42-tool requests fell from 9/14 to 4/14 correct calls (FP8: 5/14),
decode is ~14% slower (113 vs 132 tok/s, eager) and a 16k prefill ~12%
slower (0.37 vs 0.33 s). KV_DTYPE=auto is the measured-best cache.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid

REPO = Path(__file__).resolve().parents[1]
HF_REPO = "nvidia/Cosmos3-Edge"
MODEL_REVISION = "a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
VLLM_VERSION = "0.32.0"
# vLLM 0.32.0 is built against this torch ABI, not the independent torch latest.
TORCH_VERSION = "2.13.0"
# torch cu130 (FlashInfer's prebuilt kernels' CUDA), as serve_qwen_vllm.sh,
# not uv's `--torch-backend=auto` pick.
TORCH_BACKEND = "cu130"
TRANSFORMERS_VERSION = "5.19.0"
REQUIREMENTS = (
    f"vllm=={VLLM_VERSION}",
    f"torch=={TORCH_VERSION}",
    f"transformers=={TRANSFORMERS_VERSION}",
    "ninja",
)
DEFAULT_VENV = REPO / f".cosmos-vllm-{VLLM_VERSION}-{TORCH_BACKEND}"
# Exports written by 5.17.0 (the previous pin) hold the same 670 tensors
# (equal per-tensor sha256), tokenizer, processor and chat template as 5.19.0
# exports; only shard boundaries and version strings differ. Reuse them
# rather than re-exporting 5 GB; new exports record the running version.
EXPORT_TRANSFORMERS = ("5.17.0", TRANSFORMERS_VERSION)
TOOL_PARSER = "qwen3_coder"
REASONING_PARSER = "qwen3"
# vLLM 0.32.0 builds no grammar for tool_choice=auto with non-strict tools,
# by design; `parameter` pins every tool's argument schema as if it were
# strict. CASCADE's own client already marks its tools strict; OpenClaw's
# catalog does not, so the server sets the floor.
TOOL_STRICT_LEVEL = "parameter"
KV_CACHE_ARGS = {
    "nvfp4": ["--kv-cache-dtype", "nvfp4", "--attention-backend", "FLASHINFER"],
    "fp8": ["--kv-cache-dtype", "fp8"],
    "auto": [],
}
# vLLM 0.32.0 adds HTTP routes from these, or swaps its HTTP server: the dev and
# RL routes (/collective_rpc, /sleep, /update_weights; outside /v1, where a key
# does not reach), /v1/load_lora_adapter (inside the OpenShell sandbox's /v1/**),
# /cohere/v2/chat, a Rust frontend binary. VLLM_ENABLE_RESPONSES_API_STORE keeps
# every stored /v1/responses request and answer in memory until the server exits
# (vllm/envs.py: never freed), so a caller inside /v1/** grows it without bound.
# The server never takes them from the caller. KEEP IN SYNC with
# serve_qwen_vllm.sh (tests pin both).
VLLM_ROUTE_ENV = (
    "VLLM_SERVER_DEV_MODE",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
    "VLLM_ENABLE_COHERE_API",
    "VLLM_USE_RUST_FRONTEND",
    "VLLM_RUST_FRONTEND_PATH",
    "VLLM_ENABLE_RESPONSES_API_STORE",
)


class UsageError(ValueError):
    """A knob outside its allowed values: exit 2 before anything runs."""


def serving_plan(env=None):
    env = os.environ if env is None else env
    kv_dtype = env.get("KV_DTYPE", "nvfp4")
    if kv_dtype not in KV_CACHE_ARGS:
        raise UsageError(f"KV_DTYPE must be nvfp4|fp8|auto (got {kv_dtype!r})")
    engine = env.get("COSMOS_ENGINE", "vllm")
    if engine != "vllm":
        raise ValueError(
            f"COSMOS_ENGINE={engine!r} cannot serve this HF reasoner-only export. "
            "This script uses ordinary vLLM, not vLLM-Omni. The inspected Omni "
            "0.29.0rc1 supports the full Cosmos3 diffusion model, but does not "
            "register the HF reasoner as an Omni chat pipeline. Keep its "
            "incompatible dependencies in a separate environment."
        )
    venv = Path(env.get("VENV", str(DEFAULT_VENV))).expanduser().absolute()
    for protected in (REPO, REPO / ".venv", REPO / ".isaacsim"):
        if venv.resolve() == protected.resolve() or (
            protected != REPO and protected.resolve() in venv.resolve().parents
        ):
            raise ValueError(
                f"VENV must be isolated from {protected}; use {DEFAULT_VENV}"
            )
    for key, expected in (("HF_REPO", HF_REPO), ("MODEL_REVISION", MODEL_REVISION)):
        if env.get(key, expected) != expected:
            raise ValueError(f"{key} must be pinned to {expected}")
    if not 1 <= int(env.get("PORT", "8082")) <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    if not 1 <= int(env.get("CTX", "32768")) <= 131072:
        raise ValueError("CTX must be between 1 and 131072")
    fraction = float(env.get("GPU_FRAC", "0.20"))
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("GPU_FRAC must be finite and in (0, 1]")
    model_dir = Path(env.get("MODEL_DIR", str(REPO / "models"))).expanduser().absolute()
    export_dir = (
        Path(env.get("EXPORT_DIR", str(model_dir / "Cosmos3-Edge-hf")))
        .expanduser()
        .absolute()
    )
    destination = export_dir.resolve()
    if destination == REPO.resolve() or destination in REPO.resolve().parents:
        raise ValueError("EXPORT_DIR cannot replace the checkout or one of its parents")
    for protected in (REPO / ".venv", REPO / ".isaacsim", venv):
        protected = protected.resolve()
        if (
            destination == protected
            or protected in destination.parents
            or destination in protected.parents
        ):
            raise ValueError(
                f"EXPORT_DIR must be separate from environment {protected}"
            )
    served_name = env.get("SERVED_NAME", "cosmos3-edge")
    command = [
        str(venv / "bin" / "vllm"),
        "serve",
        str(export_dir),
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        env.get("PORT", "8082"),
        "--max-model-len",
        env.get("CTX", "32768"),
        "--gpu-memory-utilization",
        env.get("GPU_FRAC", "0.20"),
        "--model-impl",
        "transformers",
        "--enforce-eager",
        "--max-num-seqs",
        "2",
        # Keep vision prefill bounded beside Isaac; images remain enabled.
        "--mm-processor-kwargs",
        json.dumps({"size": {"shortest_edge": 65536, "longest_edge": 1048576}}),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        # vLLM 0.32.0 registers this alias to Qwen3EngineToolParser, whose
        # schema-constrained auto-tool grammar is checked below.
        TOOL_PARSER,
        "--tool-strict-level",
        TOOL_STRICT_LEVEL,
        "--reasoning-parser",
        REASONING_PARSER,
        "--chat-template",
        str(venv / "cosmos-chat-template.jinja"),
        *KV_CACHE_ARGS[kv_dtype],
    ]
    return {
        "engine": "vllm",
        "model_component": "hf-reasoner-only",
        "vllm_omni": False,
        "venv": str(venv),
        "export_dir": str(export_dir),
        "model": {
            "repository": HF_REPO,
            "revision": MODEL_REVISION,
            "served_name": served_name,
        },
        "requirements": list(REQUIREMENTS),
        "torch_backend": TORCH_BACKEND,
        "kv_cache_dtype": kv_dtype,
        "serve_command": command,
    }


def check_host():
    if platform.system() != "Linux":
        raise RuntimeError(
            f"Cosmos serving requires Linux + NVIDIA CUDA; detected {platform.system()}/{platform.machine()}. "
            "MPS/CPU is not this Cosmos endpoint. Use --dry-run to inspect the contract."
        )


def _installed(venv):
    """{distribution name: version} of a venv, from metadata (nothing imported)."""
    import importlib.metadata

    sites = [str(path) for path in Path(venv).glob("lib*/python*/site-packages")]
    return {
        (dist.metadata["Name"] or "").lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions(path=sites)
    }


def ensure_environment(plan):
    venv = Path(plan["venv"])
    if venv.exists():
        if not (venv / "pyvenv.cfg").is_file() or not (venv / "bin/python").is_file():
            raise RuntimeError(
                f"{venv} is not a virtual environment; refusing to overwrite it"
            )
        # uv pip install reconciles requested packages, not the requirements of
        # every existing package. It would replace Omni's Transformers <5.15
        # and leave a broken Omni installation behind. Read metadata without
        # importing that environment's packages or executing its code.
        installed = _installed(venv)
        if "vllm-omni" in installed:
            raise RuntimeError(
                f"{venv} contains vLLM-Omni; keep a separate reasoner VENV "
                "rather than replacing its incompatible Transformers pin"
            )
        # One venv per vLLM version: an in-place upgrade would keep the older
        # torch build (2.13.0 satisfies the pin whatever its CUDA) and destroy
        # the stack a rollback needs.
        if installed.get("vllm", VLLM_VERSION) != VLLM_VERSION:
            raise RuntimeError(
                f"{venv} holds vLLM {installed['vllm']}; this script builds one venv "
                f"per vLLM version (default {DEFAULT_VENV}) and never upgrades "
                "another version's environment in place. Choose another VENV."
            )
    else:
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True, stdout=sys.stderr)
    # uv reconciles the exact pins on every run; merely finding bin/vllm is
    # not enough (an old environment can contain an incompatible torch/HF).
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin/python"),
            f"--torch-backend={TORCH_BACKEND}",
            *plan["requirements"],
        ],
        check=True,
        stdout=sys.stderr,  # stdout carries only the JSON report
    )
    # Prebuilt FlashInfer kernels (flashinfer-cubin, flashinfer-jit-cache)
    # instead of a JIT next to the loaded model, as serve_qwen_vllm.sh.
    installed = _installed(venv)
    if not {"flashinfer-cubin", "flashinfer-jit-cache"} <= installed.keys():
        result = subprocess.run([str(venv / "bin/vllm"), "download-kernels"], check=False, stdout=sys.stderr)
        if result.returncode != 0:
            print(
                "[cosmos] precompiled FlashInfer kernels not installed; the first "
                "start JIT-compiles them",
                file=sys.stderr,
            )


def _weight_keys(path):
    """Validate safetensors byte layout without importing CUDA or mapping tensors."""
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        size = path.stat().st_size
        if not 0 < length <= min(100_000_000, size - 8):
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    if not entries:
        raise ValueError(f"No tensors in {path}")
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    end = 0
    for tensor in sorted(entries.values(), key=lambda t: t["data_offsets"]):
        start, stop = tensor["data_offsets"]
        shape = tensor["shape"]
        width = sizes.get(tensor["dtype"])
        if width is None or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError(f"Unsupported tensor dtype/shape in {path}")
        if start != end or stop - start != math.prod(shape) * width:
            raise ValueError(f"Invalid tensor byte offsets in {path}")
        end = stop
    if 8 + length + end != size:
        raise ValueError(f"Truncated or oversized tensor data in {path}")
    return set(entries)


def _export_inventory(path):
    def read_json(name):
        value = json.loads((path / name).read_text())
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Invalid {name} in {path}")
        return value

    config = read_json("config.json")
    if config.get("model_type") != "cosmos3_edge" or config.get("architectures") != [
        "Cosmos3EdgeForConditionalGeneration"
    ]:
        raise ValueError("Export is not a Cosmos3-Edge reasoner")
    processor = read_json("processor_config.json")
    if (
        processor.get("processor_class") != "Cosmos3EdgeProcessor"
        or not {"image_processor", "video_processor"} <= processor.keys()
    ):
        raise ValueError("Incomplete Cosmos3-Edge processor")
    read_json("tokenizer_config.json")
    read_json("tokenizer.json")
    template = (path / "chat_template.jinja").read_text()
    if not all(
        marker in template for marker in ("<tool_call>", "<function=", "<parameter=")
    ):
        raise ValueError("Chat template lacks the Cosmos/Qwen3 XML tool dialect")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        mapping = read_json(index.name).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Empty safetensors shard index")
        for filename in set(mapping.values()):
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(
                    "Export shards must be local root-level safetensors files"
                )
            expected = {name for name, shard in mapping.items() if shard == filename}
            if _weight_keys(path / filename) != expected:
                raise ValueError(f"Shard/index tensor mismatch: {filename}")
    else:
        _weight_keys(path / "model.safetensors")
    inventory = {}
    for file in sorted(path.iterdir()):
        if file.name == "cosmos-export.json":
            continue
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"Export must be self-contained regular files: {file}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        inventory[file.name] = {
            "size": file.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return inventory


def validate_export(path):
    path = Path(path)
    manifest = json.loads((path / "cosmos-export.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Invalid Cosmos export completion manifest")
    if (
        manifest.get("format") != 1
        or manifest.get("repository") != HF_REPO
        or manifest.get("revision") != MODEL_REVISION
        or manifest.get("transformers") not in EXPORT_TRANSFORMERS
    ):
        raise ValueError("Export provenance differs from the pinned Cosmos contract")
    if _export_inventory(path) != manifest.get("files"):
        raise ValueError(
            "Export file inventory/checksums do not match its completion manifest"
        )
    return manifest


def atomic_export(target, builder):
    """Serialize writers; publish only a fully validated same-filesystem export.

    builder writes model + processor into staging (never the live directory).
    Old unvalidated directories are preserved, not recursively deleted.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / f".{target.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            validate_export(target)
            return "reused"
        except (ValueError, OSError):
            pass
        stage = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        previous = None
        try:
            builder(stage)
            manifest = {
                "format": 1,
                "repository": HF_REPO,
                "revision": MODEL_REVISION,
                "transformers": TRANSFORMERS_VERSION,
                "files": _export_inventory(stage),
            }
            marker = stage / "cosmos-export.json"
            with marker.open("w") as stream:
                json.dump(manifest, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            validate_export(stage)
            if target.exists() or target.is_symlink():
                previous = target.with_name(
                    f".{target.name}.previous-{uuid.uuid4().hex}"
                )
                os.replace(target, previous)
            try:
                os.replace(stage, target)
            except OSError:
                if previous is not None:
                    os.replace(previous, target)
                raise
            if previous is not None:
                print(
                    f"[cosmos] Preserved previous export: {previous}", file=sys.stderr
                )
            return "exported"
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def check_cuda():
    import torch

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError(
            f"torch.cuda.is_available()={torch.cuda.is_available()}; "
            "Cosmos requires the NVIDIA CUDA torch build and a working GPU/driver (no MPS/CPU fallback)"
        )
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"CUDA allocation/kernel smoke failed: {exc}") from exc
    return {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}


# ── compatibility smokes (CPU, no weights; each run in a fresh process) ─────


class CompatMismatch(RuntimeError):
    """A smoke saw the known pre-patch behaviour, not an unexpected crash."""


def rope_smoke(path=None):
    """Exercise the actual installed method, with no model/weights allocated."""
    import torch
    from types import MethodType, SimpleNamespace
    from transformers.models.cosmos3_edge import modeling_cosmos3_edge as module

    owner = module.Cosmos3EdgeModel
    obj = SimpleNamespace(config=module.Cosmos3EdgeConfig())
    obj.get_vision_position_ids = MethodType(owner.get_vision_position_ids, obj)
    text = torch.tensor([[100, 101, 102]])
    mixed = torch.tensor([[100, 19, 19, 19, 19, 101]])
    scenarios = [
        (text, torch.zeros_like(text), None),
        (mixed, torch.tensor([[0, 1, 1, 1, 1, 0]]), torch.tensor([[1, 4, 4]])),
    ]
    cases = 0
    for ids, types, image_grid in scenarios:
        baseline = owner.get_rope_index(obj, ids, types, image_grid_thw=image_grid)
        if image_grid is None:
            expected = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
            if not torch.equal(baseline[0], expected) or torch.count_nonzero(
                baseline[1]
            ):
                raise RuntimeError("No-video text RoPE positions/deltas are incorrect")
        # vLLM 0.32.0 passes None without videos; (0, 3) is the documented
        # empty grid. A 1-D torch.empty(0) is outside that contract (it fails
        # on transformers 5.19.0 too) and no caller produces it.
        for video_grid in (None, torch.empty((0, 3), dtype=torch.long)):
            positions, deltas = owner.get_rope_index(
                obj, ids, types, image_grid_thw=image_grid, video_grid_thw=video_grid
            )
            if (
                positions.shape != (3, 1, ids.shape[1])
                or not torch.equal(positions, baseline[0])
                or not torch.equal(deltas, baseline[1])
            ):
                raise RuntimeError(
                    "Empty-video RoPE differs from the no-video baseline"
                )
            cases += 1
    return {
        "cases": cases,
        "inputs": "text and image+text; video None and empty (0,3)",
    }


def multimodal_image_smoke(path):
    """Exercise vLLM's actual processor selection, counts and HF pixel layout on CPU."""
    import torch
    from transformers import AutoProcessor
    from vllm.config import ModelConfig
    from vllm.model_executor.models.transformers import multimodal as module
    from vllm.multimodal.processing import InputProcessingContext

    reference = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    config = ModelConfig(model=str(path), model_impl="transformers", max_model_len=32768)
    info = module.MultiModalProcessingInfo(InputProcessingContext(config, reference.tokenizer))
    processor = info.get_hf_processor()
    sizes = [(32, 32), (193, 317), (320, 640)]
    for height, width in sizes:
        counts = processor._get_num_multimodal_tokens(image_sizes=[(height, width)])
        image = torch.arange(3 * height * width).reshape(3, height, width).remainder(256).to(torch.uint8)
        actual = processor.image_processor(image, return_tensors="pt")
        expected = reference.image_processor(image, return_tensors="pt")
        if (not torch.equal(actual["pixel_values"], expected["pixel_values"])
                or not torch.equal(actual["image_grid_thw"], expected["image_grid_thw"])
                or counts["num_image_patches"] != [actual["pixel_values"].shape[0]]
                or counts["num_image_tokens"] != [int(actual["image_grid_thw"].prod()) // processor.image_processor.merge_size**2]):
            raise CompatMismatch("vLLM image counts or patch layout differ from official Cosmos HF pixels")
    return {"cases": len(sizes), "max_image_tokens": info.get_max_image_tokens(),
            "processor": type(processor).__module__ + "." + type(processor).__name__}


def multimodal_video_smoke(path):
    """Compare allocation-free counts to real CPU video patchification (no weights)."""
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    # No do_resize=False case: every Qwen2-VL-family counter (this one
    # included) ignores it and counts the resized grid. CASCADE always serves
    # with resizing on (--mm-processor-kwargs size bounds).
    cases = [((4, 63, 97), {}),
             ((3, 128, 256), {"size": {"shortest_edge": 1024, "longest_edge": 12288}, "merge_size": 1})]
    for shape, options in cases:
        frames, height, width = shape
        video = torch.zeros((frames, 3, height, width), dtype=torch.uint8)
        counts = processor._get_num_multimodal_tokens(video_sizes=[shape], videos_kwargs=options)
        actual = processor.video_processor(video, return_tensors="pt", do_sample_frames=False, **options)
        patches = int(actual["video_grid_thw"].prod())
        expected = patches // options.get("merge_size", processor.video_processor.merge_size)**2
        if patches != actual["pixel_values_videos"].shape[0] or counts["num_video_tokens"] != [expected]:
            raise RuntimeError("Cosmos video token counts differ from actual video patches")
    return {"cases": len(cases), "video_http_verified": False}


def model_config_smoke(path):
    """Catch registry/config substitution before GPU weight allocation."""
    import torch
    from transformers import AutoModel
    from transformers.models.cosmos3_edge.configuration_cosmos3_edge import Cosmos3EdgeConfig
    from vllm.config import ModelConfig

    config = ModelConfig(model=str(path), model_impl="transformers", max_model_len=32768)
    reference = Cosmos3EdgeConfig.from_pretrained(str(path), local_files_only=True)
    with torch.device("meta"):
        model = AutoModel.from_config(config.hf_config)
    layers = len(model.get_decoder().layers)
    if (type(config.hf_config) is not Cosmos3EdgeConfig
            or layers != reference.text_config.num_hidden_layers
            or "MoE" in config._architecture):
        raise CompatMismatch("vLLM selected a different Cosmos architecture than the HF export")
    return {"decoder_layers": layers, "architecture": config._architecture, "weights_allocated": False}


def native_grammar_smoke(path):
    """The server's own parser stack must build a grammar that admits complete
    tool calls, wrapped or bare, and rejects a call missing a required argument."""
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    tool = {"type": "function", "function": {"name": "probe", "parameters": {
        "type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]}}}
    parser_cls = ParserManager.get_parser(
        tool_parser_name=TOOL_PARSER, reasoning_parser_name=REASONING_PARSER,
        enable_auto_tools=True, tool_strict_level=TOOL_STRICT_LEVEL,
    )
    request = parser_cls(tokenizer, tools=[tool]).adjust_request(
        ChatCompletionRequest(model="probe", messages=[], tools=[tool], tool_choice="auto"))
    tag = getattr(request.structured_outputs, "structural_tag", None)
    if tag is None:
        raise CompatMismatch("auto tools get no schema grammar at this strict level")
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer)).compile_structural_tag(tag)
    cases = [("The table is empty.", True)]
    for prefix, suffix in [("", ""), ("<tool_call>\n", "\n</tool_call>")]:
        call = prefix + "<function=probe>\n{}</function>" + suffix
        cases += [(call.format("<parameter=ready>true</parameter>\n"), True),
                  (call.format(""), False),
                  (call.format("<parameter=ready>maybe</parameter>\n"), False)]
    for text, expected in cases:
        if xgr.GrammarMatcher(compiled).accept_string(text) != expected:
            raise CompatMismatch(
                f"Cosmos tool grammar {'rejects' if expected else 'admits'} {text!r}")
    return {"cases": len(cases), "tool_strict_level": TOOL_STRICT_LEVEL,
            "decoding": "schema-constrained auto tools, wrapped and bare", "generation_verified": False}


SMOKES = {
    "rope": rope_smoke,
    "video": multimodal_video_smoke,
    "model_config": model_config_smoke,
    "image": multimodal_image_smoke,
    "tool_grammar": native_grammar_smoke,
}
# Failures a known patch fixes. Anything else is never patched.
KNOWN_FAILURES = {
    "model_config": (AttributeError, CompatMismatch),
    "image": (AttributeError, CompatMismatch),
    "tool_grammar": (CompatMismatch,),
}
SMOKE_MARKER = "COSMOS_SMOKES="


class _Overlay(importlib.abc.MetaPathFinder):
    """Load named modules from candidate files: a smoke proves exact bytes."""

    def __init__(self, files):
        self.files = files

    def find_spec(self, name, path=None, target=None):
        candidate = self.files.get(name)
        if candidate is None:
            return None
        original = importlib.machinery.PathFinder.find_spec(name, path)
        if original is None:
            return None
        return importlib.util.spec_from_file_location(
            name, candidate, submodule_search_locations=original.submodule_search_locations
        )


def _smoke_child(spec):
    if spec.get("overlay"):
        sys.meta_path.insert(0, _Overlay(spec["overlay"]))
    results = {}
    for name in spec["smokes"]:
        try:
            results[name] = {"ok": True, "result": SMOKES[name](Path(spec["export"]))}
        except Exception as exc:  # noqa: BLE001 - reported to the parent, which decides
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:600],
                             "known": isinstance(exc, KNOWN_FAILURES.get(name, ()))}
    print(SMOKE_MARKER + json.dumps(results, default=str), flush=True)
    return 0


def run_smokes(export_path, overlay=None, *, python=None, names=tuple(SMOKES)):
    """Run the smokes in a fresh interpreter: no cached registry, processor or
    AutoConfig state from an earlier failure leaks into a later verdict."""
    spec = {"export": str(export_path), "smokes": list(names),
            "overlay": {name: str(file) for name, file in (overlay or {}).items()}}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [python or sys.executable, "-B", str(Path(__file__).resolve()), "--_smokes", json.dumps(spec)],
        env=env, capture_output=True, text=True, timeout=900, check=False,
    )
    line = next((x for x in reversed(result.stdout.splitlines()) if x.startswith(SMOKE_MARKER)), None)
    if result.returncode != 0 or line is None:
        raise RuntimeError(f"Compatibility smoke process failed (exit {result.returncode}): "
                           + (result.stderr or result.stdout)[-2000:])
    return json.loads(line[len(SMOKE_MARKER):])


# ── wheel patches: exact versions + file sha256, proven, atomic inode swap ──
#
# Each patch is the minimal backport of an open upstream vLLM PR and is
# dropped once a vLLM release contains it. A smoke that passes without the
# patch reports "upstream-ok" and nothing is written.

_PR_60823 = "https://github.com/vllm-project/vllm/pull/60823"
_PR_62368 = "https://github.com/vllm-project/vllm/pull/62368"

_QWEN_IMPORT = "from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter\n"
_QWEN_IMPORTS = """from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

if TYPE_CHECKING:
    from xgrammar import StructuralTag
"""
_QWEN_MODEL = '    structural_tag_model = "qwen_3_coder"\n'
_QWEN_METHOD = _QWEN_MODEL + """
    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> StructuralTag | None:
        tag = super().get_structural_tag(
            request, reasoning=reasoning, strict_level=strict_level
        )
        if tag is None:
            return None
        from vllm.tool_parsers.structural_tag_registry import (
            allow_unwrapped_qwen_xml_tool_calls,
        )

        # The parser also extracts calls written without <tool_call>.
        return allow_unwrapped_qwen_xml_tool_calls(tag)
"""
_REGISTRY_ANCHOR = "\ndef _dump_tool_for_xgrammar(\n"
_REGISTRY_BLOCK = r'''
_QWEN_XML_TOOL_CALL_BEGIN = "<tool_call>\n"
_QWEN_XML_TOOL_CALL_END = "\n</tool_call>"
_QWEN_XML_FUNCTION_BEGIN = "<function="


def allow_unwrapped_qwen_xml_tool_calls(tag: StructuralTag) -> StructuralTag:
    """Also constrain Qwen XML calls written without ``<tool_call>``, in place.

    The Qwen3 parser extracts ``<function=NAME>...</function>`` as a tool call
    with or without the ``<tool_call>`` wrapper, but the ``qwen_3_coder`` tag
    only triggers on the wrapped form. An unwrapped call was free text to the
    grammar, so its arguments were never checked against the tool schema.
    """

    def visit(node: Any) -> None:
        if isinstance(node, TriggeredTagsFormat):
            unwrapped = [
                TagFormat(
                    begin=tool_tag.begin.removeprefix(_QWEN_XML_TOOL_CALL_BEGIN),
                    content=tool_tag.content,
                    end=tool_tag.end.removesuffix(_QWEN_XML_TOOL_CALL_END),
                )
                for tool_tag in node.tags
                if isinstance(tool_tag.begin, str)
                and tool_tag.begin.startswith(
                    _QWEN_XML_TOOL_CALL_BEGIN + _QWEN_XML_FUNCTION_BEGIN
                )
                and isinstance(tool_tag.end, str)
                and tool_tag.end.endswith(_QWEN_XML_TOOL_CALL_END)
            ]
            if unwrapped and _QWEN_XML_FUNCTION_BEGIN not in node.triggers:
                node.tags.extend(unwrapped)
                node.triggers.append(_QWEN_XML_FUNCTION_BEGIN)
        elif isinstance(node, BaseModel):
            for name in type(node).model_fields:
                visit(getattr(node, name))
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(tag.format)
    return tag

'''

COMPAT_PATCHES = (
    {
        # vLLM's vendored Nemotron-H-backed Cosmos3EdgeConfig is registered
        # for every model_impl; HF modeling then fails ('Cosmos3EdgeTextConfig'
        # has no 'hidden_act'). Deleting the registration disables only vLLM's
        # native Cosmos3-Edge backend, which cannot load this export anyway.
        "name": "vllm#60823 config registry",
        "upstream": _PR_60823,
        "smoke": "model_config",
        "files": (
            {
                "module": "vllm.transformers_utils.config",
                "path": "vllm/transformers_utils/config.py",
                "sha256": "4bba67c5f85e1fc0b956e6bb9f09f27dbcb4a8683ab0e4c3eb2a4de0150228cc",
                "patched_sha256": "ee4cd61ae4da636fce6902013fba9a262fa108497f6a80534651c3dab980b063",
                "edits": (('    cosmos3_edge="Cosmos3EdgeConfig",\n', ""),),
            },
        ),
    },
    {
        # get_processor prefers vLLM's vendored Cosmos3EdgeProcessor
        # (row-major patches, no image patch counting) over the HF one the
        # Cosmos vision encoder was trained with (block-major patches).
        "name": "vllm#60823 processor registry",
        "upstream": _PR_60823,
        "smoke": "image",
        "files": (
            {
                "module": "vllm.transformers_utils.processors",
                "path": "vllm/transformers_utils/processors/__init__.py",
                "sha256": "851d8a75cac4a08e43e54e31236c289e417164571ca43f43be485c9b8fefdc9c",
                "patched_sha256": "3e094e1db0af5efb82d9eadf46f291777445349d0780dbd3e6ed7a87668cca4d",
                "edits": (
                    ('    "Cosmos3EdgeProcessor",\n', ""),
                    ('    "Cosmos3EdgeProcessor": "vllm.transformers_utils.processors.cosmos3_edge",\n', ""),
                ),
            },
        ),
    },
    {
        # Cosmos emits <function=NAME> calls without <tool_call>. The Qwen3
        # parser extracts them, but the qwen_3_coder grammar only triggers on
        # the wrapped form, so their arguments were never schema-checked.
        "name": "vllm#62368 unwrapped tool-call grammar",
        "upstream": _PR_62368,
        "smoke": "tool_grammar",
        "files": (
            {
                "module": "vllm.tool_parsers.qwen3_engine_tool_parser",
                "path": "vllm/tool_parsers/qwen3_engine_tool_parser.py",
                "sha256": "3cf83a2a9408d72c79082825464b2c4dea1147ff390289dfb8936c5501114be9",
                "patched_sha256": "f7386e563a3dbc64e7a2359f82bc5fbc2ff76993cbc6746a9588461bd55ffe02",
                "edits": ((_QWEN_IMPORT, _QWEN_IMPORTS), (_QWEN_MODEL, _QWEN_METHOD)),
            },
            {
                "module": "vllm.tool_parsers.structural_tag_registry",
                "path": "vllm/tool_parsers/structural_tag_registry.py",
                "sha256": "89cba76c81d5976a7b3fa7403817bf369b8bb244cc5c424d04ba6da6ef36e47f",
                "patched_sha256": "e745add19b74f5a19234a9f47083d2f8a22bc7a7db55cc1fc25b783be74d1136",
                "edits": ((_REGISTRY_ANCHOR, _REGISTRY_BLOCK + _REGISTRY_ANCHOR),),
            },
        ),
    },
)


def _locate_installed(relative):
    """Installed file of a distribution (vllm/...), found without importing it."""
    import importlib.metadata

    return Path(str(importlib.metadata.distribution(relative.split("/", 1)[0]).locate_file(relative)))


def _stack_matches():
    import importlib.metadata

    return (importlib.metadata.version("vllm") == VLLM_VERSION
            and importlib.metadata.version("transformers") == TRANSFORMERS_VERSION)


def apply_edits(source, edits):
    for before, after in edits:
        if source.count(before) != 1:
            raise RuntimeError("Unrecognized compatibility patch anchor; not patched")
        source = source.replace(before, after, 1)
    return source


def _write_replacing_inode(path, data):
    """uv hardlinks wheel files into its cache and other environments: never
    write those shared bytes in place, replace this path's inode."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".cosmos-compat-", delete=False) as out:
            temporary = Path(out.name)
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def ensure_compat(export_path, *, allow_patch=False, runner=None, locate=None,
                  versions_ok=None, patches=COMPAT_PATCHES):
    """Smoke the installed stack; patch only identified wheel bytes, after the
    exact candidate bytes pass every smoke in a fresh process, then re-prove."""
    runner = runner or (lambda overlay: run_smokes(export_path, overlay))
    locate = locate or _locate_installed
    first = runner({})
    report = {"smokes": first, "patches": {}}
    failed = {name for name, result in first.items() if not result["ok"]}
    pending = []
    for patch in patches:
        files = [(spec, locate(spec["path"])) for spec in patch["files"]]
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for _, path in files]
        entry = {"upstream": patch["upstream"], "files": {spec["path"]: digest for (spec, _), digest in zip(files, digests)}}
        report["patches"][patch["name"]] = entry
        if patch["smoke"] not in failed:
            earlier = all(d == spec["patched_sha256"] for (spec, _), d in zip(files, digests))
            entry["status"] = "patched-earlier" if earlier else "upstream-ok"
            continue
        result = first[patch["smoke"]]
        if not allow_patch:
            raise RuntimeError(f"{patch['smoke']} smoke failed ({result['error']}); run --setup-only")
        if not result.get("known"):
            raise RuntimeError(f"Unexpected {patch['smoke']} smoke failure, not patched: {result['error']}")
        if versions_ok is None:
            versions_ok = _stack_matches()
        if not versions_ok or any(d != spec["sha256"] for (spec, _), d in zip(files, digests)):
            raise RuntimeError(f"Unrecognized {patch['name']} code/version; not patched: {result['error']}")
        pending.append((patch, files))
    unpatched = sorted(failed - {patch["smoke"] for patch, _ in pending})
    if unpatched:
        errors = "; ".join(f"{name}: {first[name]['error']}" for name in unpatched)
        hint = "run --setup-only" if not allow_patch else "no known patch"
        raise RuntimeError(f"Cosmos compatibility smoke failed ({hint}): {errors}")
    if not pending:
        return report
    changes = []  # (module, path, original bytes, candidate bytes)
    for patch, files in pending:
        for spec, path in files:
            original = path.read_bytes()
            candidate = apply_edits(original.decode(), spec["edits"])
            compile(candidate, str(path), "exec")
            changes.append((spec["module"], path, original, candidate.encode()))
    with tempfile.TemporaryDirectory(prefix="cosmos-compat-") as stage:
        overlay = {}
        for module, _, _, candidate in changes:
            overlay[module] = Path(stage) / (module + ".py")
            overlay[module].write_bytes(candidate)
        proof = runner(overlay)
    if not all(result["ok"] for result in proof.values()):
        failures = {name: result["error"] for name, result in proof.items() if not result["ok"]}
        raise RuntimeError(f"Candidate compatibility patches failed their smokes; nothing written: {failures}")
    written = []
    try:
        for _, path, _, candidate in changes:
            _write_replacing_inode(path, candidate)
            written.append(path)
        after = runner({})
        if not all(result["ok"] for result in after.values()):
            raise RuntimeError(f"Installed compatibility patches failed their smokes: {after}")
    except BaseException:
        for _, path, original, _ in changes:
            if path in written:
                _write_replacing_inode(path, original)
        raise
    report["smokes"] = after
    for patch, files in pending:
        entry = report["patches"][patch["name"]]
        entry["status"] = "patched"
        entry["files"] = {spec["path"]: hashlib.sha256(path.read_bytes()).hexdigest() for spec, path in files}
    return report


def serving_chat_template(source):
    """Keep NVIDIA's images/history/XML calls; encode *tool schemas* as JSON.

    The pinned Edge weights confuse XML schema <parameter> tags with emitted
    <parameter=NAME> calls. JSON schemas avoid that ambiguity. The native vLLM
    Qwen parser still handles model-generated arguments, unchanged.
    """
    if hashlib.sha256(source.encode()).hexdigest() != "7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8":
        raise RuntimeError("Unrecognized Cosmos chat template; not patched")
    start = source.index('    {{- "<tools>" }}')
    last = '</tools>" }}'
    end = source.index(last, start) + len(last)
    return source[:start] + '    {{- "<tools>\\n" ~ (tools | tojson) ~ "\\n</tools>" }}' + source[end:]


def ensure_chat_template(export_path, venv, *, allow_write=False):
    """Publish the derived serving template outside the checksummed model export."""
    source = (Path(export_path) / "chat_template.jinja").read_text()
    candidate = serving_chat_template(source)
    path = Path(venv) / "cosmos-chat-template.jinja"
    if path.is_file() and path.read_text() == candidate:
        return {"status": "reused", "path": str(path)}
    if not allow_write:
        raise RuntimeError("Missing/mismatched Cosmos serving template; run --setup-only")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".cosmos-template-", delete=False) as out:
            temporary = Path(out.name)
            out.write(candidate)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"status": "created", "path": str(path)}


def check_stack():
    import importlib.metadata

    versions = {}
    for name, expected in (
        ("vllm", VLLM_VERSION),
        ("torch", f"{TORCH_VERSION}+{TORCH_BACKEND}"),
        ("transformers", TRANSFORMERS_VERSION),
    ):
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(
                f"{name}=={expected} required, found {actual}; run --setup-only"
            )
        versions[name] = actual
    from transformers import AutoConfig

    if AutoConfig.for_model("cosmos3_edge").model_type != "cosmos3_edge":
        raise RuntimeError("Installed transformers lacks cosmos3_edge")
    return versions


def processor_smoke(path):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "cascade_readiness",
                "description": "Read-only protocol probe",
                "parameters": {
                    "type": "object",
                    "properties": {"ready": {"type": "boolean"}},
                    "required": ["ready"],
                },
            },
        }
    ]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": "Call cascade_readiness with ready=true."}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not all(
        marker in prompt
        for marker in (
            "cascade_readiness",
            "<tool_call>",
            "<function=",
            "<parameter=",
            "<think></think>",
        )
    ):
        raise RuntimeError(
            "Saved processor does not render the native-tool prompt dialect"
        )
    for token in ("<think>", "</think>", "<tool_call>", "</tool_call>"):
        ids = processor.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or processor.tokenizer.decode(ids) != token:
            raise RuntimeError(f"Saved tokenizer lacks a parser terminal: {token}")
    return {
        "processor": type(processor).__name__,
        "tool_prompt": "rendered, not generated",
    }


def build_export(stage):
    import torch
    from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

    model, info = Cosmos3EdgeForConditionalGeneration.from_pretrained(
        HF_REPO,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    if (
        info.get("missing_keys")
        or info.get("mismatched_keys")
        or info.get("error_msgs")
    ):
        raise RuntimeError(
            f"Refusing incomplete reasoner weights (missing/mismatched): {info}"
        )
    # v5 defaults to save_original_format=True, which reverses checkpoint key
    # mappings! vLLM needs the canonical HF reasoner keys, not diffusers keys.
    model.save_pretrained(stage, max_shard_size="2GB", save_original_format=False)
    config_path = stage / "config.json"
    config = json.loads(config_path.read_text())
    config.pop(
        "allow_patterns_overrides", None
    )  # source uses nested shards; export is flat
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    del model
    AutoProcessor.from_pretrained(HF_REPO, revision=MODEL_REVISION).save_pretrained(
        stage
    )
    processor_smoke(stage)


def serve_environment(plan, environ=None):
    environ = os.environ if environ is None else environ
    # Never from the caller: the variables that add vLLM routes or swap its HTTP
    # server (VLLM_ROUTE_ENV). VLLM_API_KEY stays: exporting it keys this sidecar.
    environ = {k: v for k, v in environ.items() if k not in VLLM_ROUTE_ENV}
    path = str(Path(plan["venv"]) / "bin") + os.pathsep + environ.get("PATH", "")
    # FlashInfer (the NVFP4 KV backend) JIT-compiles kernels its prebuilt
    # cache lacks with nvcc: on the first NVFP4 start, the fp4 prefill and the
    # XQA decode kernel for this head layout. Keep the toolkit reachable from a
    # minimal environment (a systemd unit) and cap parallel nvcc jobs beside
    # Isaac, as serve_qwen_vllm.sh does.
    if not shutil.which("nvcc", path=path):
        cuda_bin = Path(environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
        if (cuda_bin / "nvcc").is_file():
            path += os.pathsep + str(cuda_bin)
    return dict(environ, PATH=path, VIRTUAL_ENV=plan["venv"], MAX_JOBS=environ.get("MAX_JOBS", "4"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment: VENV (default checkout/.cosmos-vllm-0.32.0-cu130), MODEL_DIR, "
            "EXPORT_DIR, PORT=8082, CTX=32768, GPU_FRAC=0.20, SERVED_NAME=cosmos3-edge, "
            "KV_DTYPE=nvfp4|fp8|auto (nvfp4; auto is BF16). "
            "COSMOS_ENGINE=vllm is the only supported engine for this HF export; "
            "an explicit Omni request fails rather than falling back. "
            "HF_REPO and MODEL_REVISION are locked; this is not a hosted-OpenAI fallback."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a JSON plan; no installs, files or CUDA access",
    )
    mode.add_argument(
        "--setup-only",
        action="store_true",
        help="Prepare the isolated environment and atomic model export; do not serve",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Read-only stack, CUDA and export preflight; never proof of native tool generation",
    )
    parser.add_argument("--_prepared", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_smokes", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._smokes:
        return _smoke_child(json.loads(args._smokes))
    try:
        plan = serving_plan()
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        check_host()
        python = Path(plan["venv"]) / "bin/python"
        if not args._prepared:
            if args.check:
                if not python.is_file():
                    raise RuntimeError(
                        f"Missing {python}; run --setup-only on the Spark first"
                    )
            else:
                ensure_environment(plan)
            child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            # Do not inherit app-specific imports into the sidecar.
            child_env.pop("PYTHONPATH", None)
            child_env.pop("PYTHONHOME", None)
            child_args = list(sys.argv[1:] if argv is None else argv)
            os.execve(
                str(python),
                [
                    str(python),
                    "-B",
                    str(Path(__file__).resolve()),
                    *child_args,
                    "--_prepared",
                ],
                child_env,
            )
        if Path(sys.prefix).resolve() != Path(plan["venv"]).resolve():
            raise RuntimeError(
                "Prepared process is not running in the isolated Cosmos VENV"
            )
        stack = check_stack()
        cuda = check_cuda()
        export_path = Path(plan["export_dir"])
        if args.check:
            validate_export(export_path)
            export_status = "validated"
        else:
            export_status = atomic_export(export_path, build_export)
        compat = ensure_compat(export_path, allow_patch=not args.check)
        processor = processor_smoke(export_path)
        chat_template = ensure_chat_template(export_path, plan["venv"], allow_write=not args.check)
        report = {
            "state": "prepared-unverified",
            "engine": plan["engine"],
            "model_component": plan["model_component"],
            "vllm_omni": plan["vllm_omni"],
            "native_generation_verified": False,
            "stack": stack,
            "cuda": cuda,
            "kv_cache_dtype": plan["kv_cache_dtype"],
            "compat": compat,
            "chat_template": chat_template,
            "export": export_status,
            "export_dir": str(export_path),
            "processor": processor,
        }
        if args.check or args.setup_only:
            print(json.dumps(report, indent=2))
            return 0
        print("[cosmos] " + json.dumps(report), file=sys.stderr, flush=True)
        print(f"[cosmos] KV cache: {plan['kv_cache_dtype']} "
              f"{' '.join(KV_CACHE_ARGS[plan['kv_cache_dtype']]) or '(BF16, vLLM default)'}",
              file=sys.stderr, flush=True)
        os.execve(plan["serve_command"][0], plan["serve_command"], serve_environment(plan))
    except UsageError as exc:
        print(f"[cosmos] {exc}", file=sys.stderr)
        return 2
    except (
        ValueError,
        RuntimeError,
        OSError,
        ImportError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"[cosmos] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
