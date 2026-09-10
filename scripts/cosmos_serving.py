#!/usr/bin/env python3
"""Pinned, isolated Cosmos3-Edge reasoner sidecar for an OpenAI/MCP chat host.

This process serves a model; it never starts cascade's in-process agent loop.
Heavy imports are deliberately lazy so --help/--dry-run work without CUDA.
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
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
    ]
    return {
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
        processor = processor_smoke(export_path)
        report = {
            "state": "prepared-unverified",
            "native_generation_verified": False,
            "stack": stack,
            "cuda": cuda,
            "rope": rope,
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
