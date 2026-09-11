#!/usr/bin/env python3
"""Pinned Cosmos3-Edge HF reasoner on ordinary vLLM, NOT vLLM-Omni.

This process serves a model; it never starts cascade's in-process agent loop.
Heavy imports are deliberately lazy so --help/--dry-run work without CUDA.
Omni's Cosmos3 diffusion pipeline is a different deployment and does not
register this reasoner-only export for native chat/tool calling.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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
# Latest stable vLLM requires this torch ABI, not the independent torch latest.
REQUIREMENTS = ("vllm==0.29.0", "torch==2.13.0", "transformers==5.17.0", "ninja")


def serving_plan(env=None):
    env = os.environ if env is None else env
    engine = env.get("COSMOS_ENGINE", "vllm")
    if engine != "vllm":
        raise ValueError(
            f"COSMOS_ENGINE={engine!r} cannot serve this HF reasoner-only export. "
            "This script uses ordinary vLLM, not vLLM-Omni. The inspected Omni "
            "0.29.0rc1 supports the full Cosmos3 diffusion model, but does not "
            "register the HF reasoner as an Omni chat pipeline. Keep its "
            "incompatible dependencies in a separate environment."
        )
    venv = Path(env.get("VENV", str(REPO / ".cosmos"))).expanduser().absolute()
    for protected in (REPO, REPO / ".venv", REPO / ".isaacsim"):
        if venv.resolve() == protected.resolve() or (
            protected != REPO and protected.resolve() in venv.resolve().parents
        ):
            raise ValueError(
                f"VENV must be isolated from {protected}; use {REPO / '.cosmos'}"
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
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--chat-template",
        str(venv / "cosmos-chat-template.jinja"),
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
        "serve_command": command,
    }


def check_host():
    if platform.system() != "Linux":
        raise RuntimeError(
            f"Cosmos serving requires Linux + NVIDIA CUDA; detected {platform.system()}/{platform.machine()}. "
            "MPS/CPU is not this Cosmos endpoint. Use --dry-run to inspect the contract."
        )


def ensure_environment(plan):
    venv = Path(plan["venv"])
    if venv.exists():
        if not (venv / "pyvenv.cfg").is_file() or not (venv / "bin/python").is_file():
            raise RuntimeError(
                f"{venv} is not a virtual environment; refusing to overwrite it"
            )
        # uv pip install reconciles requested packages, not the requirements of
        # every existing package. It would replace Omni's Transformers <5.15
        # with 5.17.0 and leave a broken Omni installation behind. Read metadata
        # without importing that environment's packages or executing its code.
        import importlib.metadata

        sites = [str(path) for path in venv.glob("lib*/python*/site-packages")]
        for distribution in importlib.metadata.distributions(path=sites):
            name = (distribution.metadata["Name"] or "").lower().replace("_", "-")
            if name == "vllm-omni":
                raise RuntimeError(
                    f"{venv} contains vLLM-Omni; keep a separate reasoner VENV "
                    "rather than replacing its incompatible Transformers pin"
                )
    else:
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
    # uv reconciles the exact pins on every run; merely finding bin/vllm is
    # not enough (an old environment can contain an incompatible torch/HF).
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin/python"),
            *plan["requirements"],
        ],
        check=True,
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
        or manifest.get("transformers") != "5.17.0"
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
                "transformers": "5.17.0",
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


def rope_smoke(module):
    """Exercise the actual installed method, with no model/weights allocated."""
    import torch
    from types import MethodType, SimpleNamespace

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
        for video_grid in (
            None,
            torch.empty(0, dtype=torch.long),
            torch.empty((0, 3), dtype=torch.long),
        ):
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
        "inputs": "text and image+text; video None, empty 1-D and empty (0,3)",
    }


def ensure_rope_compat(module=None, *, version=None, allow_patch=False):
    # Behavior first: a later upstream fix is SUCCESS even if its source no
    # longer resembles the day-one workaround. Never patch unknown failures.
    if module is None:
        import transformers
        from transformers.models.cosmos3_edge import modeling_cosmos3_edge as module

        version = transformers.__version__
    try:
        return {"status": "upstream-ok", **rope_smoke(module)}
    except IndexError as exc:
        if not allow_patch:
            raise RuntimeError(
                "No-video RoPE smoke failed; run --setup-only to apply the tested 5.17.0 guard"
            ) from exc
        failure = exc
    path = Path(module.__file__)
    source = path.read_text()
    bad = "if video_grid_thw is not None:\n            video_grid_thw = torch.repeat_interleave("
    good = "if video_grid_thw is not None and video_grid_thw.numel() > 0:\n            video_grid_thw = torch.repeat_interleave("
    if version != "5.17.0" or source.count(bad) != 1:
        raise RuntimeError(
            f"Unrecognized failing RoPE code/version {version} at {path}; not patched"
        ) from failure
    candidate = source.replace(bad, good, 1)
    import ast

    tree = ast.parse(candidate)
    owner_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Cosmos3EdgeModel"
    )
    method = next(
        n
        for n in owner_node.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_rope_index"
    )
    method_tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(method_tree), str(path), "exec"), namespace)
    previous = module.Cosmos3EdgeModel.get_rope_index
    module.Cosmos3EdgeModel.get_rope_index = namespace["get_rope_index"]
    temporary = None
    try:
        result = rope_smoke(module)  # Candidate must pass before any file write.
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=".cosmos-rope-", delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(candidate)
            out.flush()
            os.fsync(out.fileno())
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    except Exception:
        module.Cosmos3EdgeModel.get_rope_index = previous
        raise
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"status": "patched-5.17.0", "path": str(path), **result}


def _install_compat_method(module, owner_name, method_name, method_source, *,
                           expected_sha256, version_ok, smoke, allow_patch,
                           replacement=None):
    """Patch only identified wheel bytes, after exercising the candidate in memory.

    Replace the inode: uv wheels can be hardlinked to other environments/cache.
    Never mutate those shared bytes with write_text on the installed file.
    """
    try:
        return {"status": "upstream-ok", **smoke()}
    except AttributeError as failure:
        if not allow_patch:
            raise RuntimeError("Multimodal compatibility smoke failed; run --setup-only") from failure
        path = Path(module.__file__)
        source = path.read_text()
        if not version_ok or hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
            raise RuntimeError(f"Unrecognized failing multimodal code/version at {path}; not patched") from failure

    import textwrap

    if replacement is None:
        anchor = f"class {owner_name}("
        start = source.index("\n", source.index(anchor)) + 1
        # The target classes have a one-line declaration; insert one method only.
        candidate = source[:start] + method_source + "\n" + source[start:]
    else:
        import ast

        before, after = replacement
        if source.count(before) != 1:
            raise RuntimeError("Unrecognized multimodal replacement anchor")
        candidate = source.replace(before, after, 1)
        tree = ast.parse(candidate)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name)
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
        method_source = ast.get_source_segment(candidate, method)
    compile(candidate, str(path), "exec")
    namespace = dict(module.__dict__)
    exec(compile("from __future__ import annotations\n" + textwrap.dedent(method_source),
                 str(path), "exec"), namespace)
    owner = getattr(module, owner_name)
    previous = owner.__dict__.get(method_name)
    setattr(owner, method_name, namespace[method_name])
    temporary = None
    try:
        result = smoke()
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".cosmos-mm-", delete=False) as out:
            temporary = Path(out.name)
            out.write(candidate)
            out.flush()
            os.fsync(out.fileno())
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    except Exception:
        if previous is None:
            delattr(owner, method_name)
        else:
            setattr(owner, method_name, previous)
        raise
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"status": "patched", "path": str(path), **result}


def multimodal_image_smoke(path, module):
    """Exercise vLLM's actual processor selection, counts and HF pixel layout on CPU."""
    import torch
    from transformers import AutoProcessor
    from vllm.config import ModelConfig
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
            raise RuntimeError("vLLM image counts or patch layout differ from official Cosmos HF pixels")
    return {"cases": len(sizes), "max_image_tokens": info.get_max_image_tokens(),
            "processor": type(processor).__module__ + "." + type(processor).__name__}


def ensure_transformers_processor_compat(path, *, allow_patch=False):
    import transformers
    import vllm
    from vllm.model_executor.models.transformers import multimodal as module

    # The vLLM-native processor is row-major and lacks image patch counting.
    # HF's Cosmos vision encoder instead consumes block-major patches. Explicit
    # type selection bypasses that registry ONLY inside the Transformers backend;
    # the native vLLM backend and all other models keep their existing processors.
    method = '''    def get_hf_processor(self, **kwargs):
        if self.get_hf_config().model_type == "cosmos3_edge":
            return self.ctx.get_hf_processor(transformers.Cosmos3EdgeProcessor, **kwargs)
        return self.ctx.get_hf_processor(**kwargs)
'''
    return _install_compat_method(
        module, "MultiModalProcessingInfo", "get_hf_processor", method,
        expected_sha256="3c7ea7bc25e6148ac8c4f65a9928b2a3304ef0a2a9e61884308fcc959ce88047",
        version_ok=transformers.__version__ == "5.17.0" and vllm.__version__ == "0.29.0",
        smoke=lambda: multimodal_image_smoke(path, module), allow_patch=allow_patch,
    )


def multimodal_video_smoke(path):
    """Compare allocation-free counts to real CPU video patchification (no weights)."""
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    cases = [((4, 63, 97), {}), ((1, 32, 32), {"do_resize": False}),
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


def ensure_video_processor_compat(path, *, allow_patch=False):
    import transformers
    from transformers.models.cosmos3_edge import video_processing_cosmos3_edge as module

    # Count the frames already selected by the loader, using the same resize and
    # temporal padding as this Cosmos processor (not Qwen's temporal/cap policy).
    method = '''    def get_number_of_video_patches(self, num_frames, height, width, videos_kwargs=None):
        options = videos_kwargs or {}
        patch = options.get("patch_size", self.patch_size)
        merge = options.get("merge_size", self.merge_size)
        temporal = options.get("temporal_patch_size", self.temporal_patch_size)
        size = options.get("size", self.size)
        if options.get("do_resize", self.do_resize):
            height, width = smart_resize(
                num_frames=num_frames, height=height, width=width,
                temporal_factor=temporal, factor=patch * merge,
                min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"],
            )
        return ((num_frames + temporal - 1) // temporal) * (height // patch) * (width // patch)
'''
    return _install_compat_method(
        module, "Cosmos3EdgeVideoProcessor", "get_number_of_video_patches", method,
        expected_sha256="5286800003bb76d043b1b02a3f175fb17c198d29a7c8a676b51091121d3f0014",
        version_ok=transformers.__version__ == "5.17.0",
        smoke=lambda: multimodal_video_smoke(path), allow_patch=allow_patch,
    )


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
        raise RuntimeError("vLLM selected a different Cosmos architecture than the HF export")
    return {"decoder_layers": layers, "architecture": config._architecture, "weights_allocated": False}


def ensure_model_config_compat(path, *, allow_patch=False):
    import transformers
    import vllm
    from vllm.config import model as module

    before = "        self.hf_config = hf_config\n"
    # Do NOT reconstruct from native to_dict(): Nemotron defaults leak into the
    # dense HF config and trigger MoE detection. Reload the original config and
    # preserve the same explicit overrides; leave the native backend unchanged.
    after = '''        if self.model_impl == "transformers" and hf_config.model_type == "cosmos3_edge":
            from transformers.models.cosmos3_edge.configuration_cosmos3_edge import Cosmos3EdgeConfig
            hf_config = Cosmos3EdgeConfig.from_pretrained(
                self.hf_config_path or self.model, revision=self.revision,
                code_revision=self.code_revision, token=self.hf_token,
            )
            if hf_overrides_kw:
                hf_config.update(hf_overrides_kw)
            if hf_overrides_fn:
                hf_config = hf_overrides_fn(hf_config)
        self.hf_config = hf_config
'''
    return _install_compat_method(
        module, "ModelConfig", "__post_init__", "",
        expected_sha256="de2309f4f710f04588ac0835e1cfd215e1310bcb136bf53142685c926256a27c",
        version_ok=transformers.__version__ == "5.17.0" and vllm.__version__ == "0.29.0",
        smoke=lambda: model_config_smoke(path), allow_patch=allow_patch,
        replacement=(before, after),
    )


def native_grammar_smoke(path, module):
    """Compile the native grammar; reject incomplete generated calls on CPU."""
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    request = ChatCompletionRequest(
        model=os.environ.get("SERVED_NAME", "cosmos3-edge"), messages=[], tool_choice="auto",
        tools=[{"type": "function", "function": {"name": "probe", "parameters": {
            "type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]}}}],
    )
    parser = module.Qwen3EngineToolParser(tokenizer, tools=request.tools)
    tag = parser.get_structural_tag(request)
    tag.model_dump()  # Missing auto grammar is the exact pre-patch failure.
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
    compiled = compiler.compile_structural_tag(tag)
    for prefix, suffix in [("", ""), ("<tool_call>\n", "\n</tool_call>")]:
        good = prefix + "<function=probe>\n<parameter=ready>true</parameter>\n</function>" + suffix
        bad = prefix + "<function=probe>\n</function>" + suffix
        if not xgr.GrammarMatcher(compiled).accept_string(good) or xgr.GrammarMatcher(compiled).accept_string(bad):
            raise RuntimeError("Native Cosmos grammar admits incomplete or rejects valid calls")
    return {"cases": 4, "decoding": "native schema-constrained auto tools", "generation_verified": False}


def ensure_native_grammar_compat(path, *, allow_patch=False):
    import transformers
    import vllm
    from vllm.tool_parsers import qwen3_engine_tool_parser as module

    # Keep the actual Qwen parser and all argument extraction unchanged. Apply
    # schema constraints DURING decoding, including Cosmos' unwrapped functions
    # (already understood by this parser, but missed by its default grammar).
    method = r'''    def get_structural_tag(self, request, *, reasoning=False):
        import os
        from copy import deepcopy
        from xgrammar import StructuralTag
        if request.model != os.environ.get("SERVED_NAME", "cosmos3-edge") or request.tool_choice != "auto":
            return Qwen3ParserToolAdapter.get_structural_tag(self, request, reasoning=reasoning)
        bounded = request.model_copy(deep=True)
        for tool in bounded.tools or []:
            function = getattr(tool, "function", tool)
            if getattr(function, "strict", False) is None:
                function.strict = True
        tag = Qwen3ParserToolAdapter.get_structural_tag(self, bounded, reasoning=reasoning)
        if tag is None:
            return None
        data = tag.model_dump()
        def widen(node):
            if isinstance(node, dict):
                if node.get("type") == "triggered_tags":
                    added = []
                    for item in node.get("tags", []):
                        if item.get("begin", "").startswith("<tool_call>\n<function="):
                            bare = deepcopy(item)
                            bare["begin"] = bare["begin"].removeprefix("<tool_call>\n")
                            bare["end"] = bare["end"].removesuffix("\n</tool_call>")
                            added.append(bare)
                    if added:
                        node["tags"].extend(added)
                        node["triggers"] = ["<tool_call>", "<function="]
                for value in list(node.values()):
                    widen(value)
            elif isinstance(node, list):
                for value in node:
                    widen(value)
        widen(data)
        return StructuralTag.model_validate(data)
'''
    return _install_compat_method(
        module, "Qwen3EngineToolParser", "get_structural_tag", method,
        expected_sha256="3cf83a2a9408d72c79082825464b2c4dea1147ff390289dfb8936c5501114be9",
        version_ok=transformers.__version__ == "5.17.0" and vllm.__version__ == "0.29.0",
        smoke=lambda: native_grammar_smoke(path, module), allow_patch=allow_patch,
    )


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
        ("vllm", "0.29.0"),
        ("torch", "2.13.0"),
        ("transformers", "5.17.0"),
    ):
        actual = importlib.metadata.version(name)
        if actual.split("+", 1)[0] != expected:
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Environment: VENV (default checkout/.cosmos), MODEL_DIR, EXPORT_DIR, "
            "PORT=8082, CTX=32768, GPU_FRAC=0.20, SERVED_NAME=cosmos3-edge. "
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
    args = parser.parse_args(argv)
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
        rope = ensure_rope_compat(allow_patch=not args.check)
        export_path = Path(plan["export_dir"])
        if args.check:
            validate_export(export_path)
            export_status = "validated"
        else:
            export_status = atomic_export(export_path, build_export)
        model_compat = ensure_model_config_compat(export_path, allow_patch=not args.check)
        image_compat = ensure_transformers_processor_compat(export_path, allow_patch=not args.check)
        video_compat = ensure_video_processor_compat(export_path, allow_patch=not args.check)
        native_grammar = ensure_native_grammar_compat(export_path, allow_patch=not args.check)
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
            "rope": rope,
            "model_compat": model_compat,
            "image_compat": image_compat,
            "video_compat": video_compat,
            "native_grammar": native_grammar,
            "chat_template": chat_template,
            "export": export_status,
            "export_dir": str(export_path),
            "processor": processor,
        }
        if args.check or args.setup_only:
            print(json.dumps(report, indent=2))
            return 0
        print("[cosmos] " + json.dumps(report), file=sys.stderr, flush=True)
        serve_env = dict(
            os.environ,
            PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
            VIRTUAL_ENV=plan["venv"],
        )
        os.execve(plan["serve_command"][0], plan["serve_command"], serve_env)
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
