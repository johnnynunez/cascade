#!/usr/bin/env python3
"""Prepare, verify and supervise one private, CPU-only HF speech provider.

No robot or audio device is opened here. Preparation explicitly downloads
dependencies/models; serving requires their recorded bytes and runs offline.
The upstream server has no authentication, so this recipe binds loopback only.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

from conversation_provider_source import apply_patch, verify_source, verify_installed
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/conversation/hf_kokoro_cpu.json"
REQUIREMENTS = ROOT / "configs/conversation/hf_kokoro_cpu.requirements.txt"
OWNER = {"owner": "cascade-conversation-provider", "version": 1}


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def recipe():
    return json.loads(RECIPE.read_text())


def private_state(path, *, create=False):
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise ValueError("state directory must not be a symlink")
    if create and not path.exists():
        path.mkdir(mode=0o700, parents=False)
        write_json(path / "owner.json", OWNER)
    if not path.is_dir() or (path.stat().st_mode & 0o077):
        raise ValueError("use a task-owned state directory with mode 0700")
    if path.stat().st_uid != os.getuid():
        raise ValueError("state directory must belong to the current user")
    if json.loads((path / "owner.json").read_text()) != OWNER:
        raise ValueError("state ownership marker mismatch")
    return path.resolve()


@contextmanager
def state_lock(state):
    """No simultaneous installation/server can mutate this private environment."""
    import fcntl
    with (state / "operation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("private provider state is already in use") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def environment(state, *, offline):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(("HF_", "HUGGINGFACE_", "HUGGING_FACE_", "OPENAI_", "ANTHROPIC_")):
            del env[key]
    # Do not inherit another environment's imports, models, credentials or GPU.
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_PRELOAD"):
        env.pop(key, None)
    env.update({
        "HF_HOME": str(state / "hf"), "HF_HUB_CACHE": str(state / "hf/hub"),
        "TORCH_HOME": str(state / "torch"), "NLTK_DATA": str(state / "nltk"),
        "XDG_CACHE_HOME": str(state / "cache"), "UV_CACHE_DIR": str(state / "cache/uv"),
        "NUMBA_CACHE_DIR": str(state / "numba"), "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_OFFLINE": str(int(offline)), "TRANSFORMERS_OFFLINE": str(int(offline)),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(state / "bytecode"),
    })
    return env


def download(url, path, expected):
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"existing download hash mismatch: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as target:
        shutil.copyfileobj(response, target)
    if digest(temporary) != expected:
        raise ValueError(f"download hash mismatch: {path.name}")
    temporary.rename(path)


def extract_tar(path, destination):
    destination.mkdir()
    with tarfile.open(path) as archive:
        archive.extractall(destination, filter="data")
    children = list(destination.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise ValueError("source archive must have exactly one root")
    return children[0]


def inventory(state):
    """Content bytes, excluding generated caches; relative names are portable."""
    result = {}
    for directory in ("source", "models", "nltk", "torch/hub/snakers4_silero-vad_master"):
        for path in sorted((state / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                result[str(path.relative_to(state))] = digest(path)
    for path in sorted((state / "hf/hub").glob("models--*/refs/main")):
        result[str(path.relative_to(state))] = digest(path)
    return result


def versions():
    return {d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in importlib.metadata.distributions()}


def prepare_assets(state):
    """Invoked only inside the new provider venv, after explicit prepare."""
    from huggingface_hub import snapshot_download

    source_record = verify_source(state, recipe(), RECIPE.parent)
    installed = verify_installed(state, source_record)
    # Check the selected import graph before declaring preparation complete.
    # Some shared upstream text utilities import Pillow even for text-only LLMs.
    from speech_to_speech.LLM.language_model import LanguageModelHandler  # noqa: F401
    from speech_to_speech.STT.whisper_stt_handler import WhisperSTTHandler  # noqa: F401
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler  # noqa: F401
    if sys.version_info[:2] != (3, 12):
        raise ValueError("provider recipe requires Python 3.12")
    cfg = recipe()
    models = []
    (state / "models").mkdir(exist_ok=True)
    for spec in cfg["models"]:
        snapshot = Path(snapshot_download(spec["repo"], revision=spec["revision"],
                                         allow_patterns=spec["files"], max_workers=2))
        target = state / "models" / spec["role"]
        shutil.copytree(snapshot, target)
        # Kokoro's HF adapter ignores model_name. Its default repository is
        # resolved against this exact private snapshot while HF is offline.
        cache_repo = snapshot.parent.parent
        (cache_repo / "refs").mkdir(exist_ok=True)
        (cache_repo / "refs/main").write_text(spec["revision"])
        models.append({**spec, "local": str(target.relative_to(state))})
    # Admit the exact Kokoro cache bytes too, since KPipeline reads that path.
    tts = cfg["models"][-1]
    snapshot = state / "hf/hub/models--hexgrad--Kokoro-82M/snapshots" / tts["revision"]
    cache_files = {str(p.relative_to(state)): digest(p) for p in snapshot.rglob("*") if p.is_file()}
    files = inventory(state)
    files.update(cache_files)
    write_json(state / "prepared.json", {
        "version": 2, "recipe_sha256": digest(RECIPE),
        "source_patch_sha256": digest(state / "source-patch.json"), "installed_provider": installed,
        "requirements_sha256": digest(REQUIREMENTS), "models": models,
        "files_sha256": files, "versions": versions(),
        "python_version": platform.python_version(),
        "source_patch": json.loads((state / "source-patch.json").read_text()),
    })


def verify(state):
    prepared = json.loads((state / "prepared.json").read_text())
    if (prepared["recipe_sha256"] != digest(RECIPE)
            or prepared["requirements_sha256"] != digest(REQUIREMENTS)):
        raise ValueError("recipe/requirements changed; prepare a new private directory")
    if digest(state / "source-patch.json") != prepared["source_patch_sha256"]:
        raise ValueError("effective source record changed")
    verify_source(state, recipe(), RECIPE.parent)
    if not prepared["files_sha256"]:
        raise ValueError("empty prepared inventory")
    for name, expected in prepared["files_sha256"].items():
        path = state / name
        if not path.resolve().is_relative_to(state) or digest(path) != expected:
            raise ValueError(f"prepared file changed: {name}")
    actual = inventory(state)
    expected_inventory = {k: v for k, v in prepared["files_sha256"].items()
                          if not k.startswith("hf/hub/models--hexgrad--Kokoro-82M/snapshots/")}
    if actual != expected_inventory:
        raise ValueError("prepared inventory gained or lost files")
    return prepared


def prepare(state):
    cfg = recipe()
    if platform.system().lower() != cfg["platform"] or platform.machine() != cfg["architecture"]:
        raise ValueError("this validated dependency recipe is Linux x86_64; other platforms need a separate lock")
    if sys.version_info < (3, 12):
        raise ValueError("explicit preparation requires a Python 3.12+ launcher for safe archive extraction")
    if (state / "prepared.json").exists():
        raise ValueError("already prepared; verify or choose a new state directory")
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("install uv before explicit preparation")
    if not shutil.which("git"):
        raise ValueError("install git before explicit source-patch preparation")
    env = environment(state, offline=False)
    downloads = state / "downloads"
    downloads.mkdir(exist_ok=True)
    for name, pin in cfg["archives"].items():
        download(pin["url"], downloads / name, pin["sha256"])
    python = state / "venv/bin/python"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", cfg["python"], str(state / "venv")], env=env, check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python), "--index-url",
                    "https://download.pytorch.org/whl/cpu", "torch==2.11.0+cpu", "torchaudio==2.11.0+cpu"], env=env, check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python), "--no-deps", "-r", str(REQUIREMENTS),
                    str(downloads / "en_core_web_sm-3.8.0-py3-none-any.whl")], env=env, check=True)
    source = extract_tar(downloads / "speech-to-speech.tar.gz", state / "source-unpack")
    source.rename(state / "source")
    (state / "source-unpack").rmdir()
    write_json(state / "source-patch.json", apply_patch(state / "source", cfg, RECIPE.parent))
    hub = state / "torch/hub"
    hub.mkdir(parents=True, exist_ok=True)
    silero = extract_tar(downloads / "silero.tar.gz", state / "silero-unpack")
    silero.rename(hub / "snakers4_silero-vad_master")
    (state / "silero-unpack").rmdir()
    (hub / "trusted_list").write_text("snakers4_silero-vad\n")
    for archive, directory in (("punkt_tab.zip", "tokenizers"),
                               ("averaged_perceptron_tagger_eng.zip", "taggers")):
        with zipfile.ZipFile(downloads / archive) as bundle:
            target = state / "nltk" / directory
            for name in bundle.namelist():
                if not (target / name).resolve().is_relative_to(target.resolve()):
                    raise ValueError("unsafe NLTK archive path")
            bundle.extractall(target)
    subprocess.run([uv, "pip", "install", "--python", str(python), "--no-deps",
                    "--no-build-isolation", str(state / "source")], env=env, check=True)
    subprocess.run([str(python), str(Path(__file__).resolve()), "_assets", "--state-dir", str(state)], env=env, check=True)


def provider_argv(state, port):
    cfg = recipe()
    return ["speech-to-speech", "serve", "--host", "127.0.0.1", "--port", str(port),
            "--stt", "whisper", "--stt_model_name", str(state / "models/stt"), "--stt_device", "cpu",
            "--stt_gen_max_new_tokens", "64", "--llm_backend", "transformers",
            "--model_name", str(state / "models/llm"), "--llm_device", "cpu",
            "--llm_torch_dtype", cfg["llm_torch_dtype"],
            "--llm_gen_max_new_tokens", "128", "--llm_gen_temperature", "0.7",
            "--llm_gen_do_sample", "True", "--tts", "kokoro", "--kokoro_device", "cpu",
            "--kokoro_voice", cfg["voice"], "--kokoro_lang_code", cfg["language"],
            "--enable_live_transcription", "False", "--smart_turn", "False",
            "--num_pipelines", "1", "--chat_size", "6", "--log_transcripts", "False"]


def child(state, port, run_dir, *, trace_generation=False):
    prepared = verify(state)
    if sys.version_info[:2] != (3, 12):
        raise ValueError("provider recipe requires Python 3.12")
    if versions() != prepared["versions"]:
        raise ValueError("provider dependency versions changed after preparation")
    source_record = verify_source(state, recipe(), RECIPE.parent)
    installed = verify_installed(state, source_record)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(recipe()["seed"])
    sys.argv = provider_argv(state, port)
    write_json(run_dir / "admission.json", {"argv": sys.argv, "versions": versions(),
               "python_version": platform.python_version(),
               "prepared_sha256": digest(state / "prepared.json"), "script_sha256": digest(__file__),
               "installed_provider": installed, "source_patch_sha256": digest(state / "source-patch.json"),
               "cpu_only": True, "audio_format": "PCM16 mono 24000 Hz", "provider_authentication": False})
    from speech_to_speech import s2s_pipeline
    from speech_to_speech.cli import main
    from speech_to_speech.LLM.language_model import LanguageModelHandler
    original_builder = s2s_pipeline.build_pipeline
    observation = []

    def observed_builder(*args, **kwargs):
        manager = original_builder(*args, **kwargs)
        llms = [h for h in manager.handlers if isinstance(h, LanguageModelHandler)]
        if len(llms) != 1:
            raise ValueError("expected one actual text-model handler for attestation")
        parameters = tuple(llms[0].model.parameters())
        write_json(run_dir / "model-runtime.json", {
            "observation": "read-only inspection after upstream pipeline construction and warmup",
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "parameter_devices": sorted({str(p.device) for p in parameters}),
            "parameter_count": sum(p.numel() for p in parameters),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "configured_dtype": recipe()["llm_torch_dtype"],
        })
        if trace_generation:
            from speech_to_speech.LLM.generation_trace import GenerationTrace
            observer = GenerationTrace(run_dir / "generation.jsonl")
            previous = getattr(llms[0], "generation_observer", None)
            llms[0].generation_observer = observer
            observation.append((llms[0], previous, observer))
        return manager

    # Read-only observation hook: original factory builds/warms the unchanged
    # pipeline; no model state, generation, action or audio buffer is changed.
    s2s_pipeline.build_pipeline = observed_builder
    primary = None
    try:
        main()
    except BaseException as exc:
        primary = exc
        raise
    finally:
        s2s_pipeline.build_pipeline = original_builder
        for handler, previous, observer in observation:
            handler.generation_observer = previous
            try:
                status = observer.close()
                write_json(run_dir / "generation-status.json", status)
                # Interrupted responses may be expected on cancel. Lost
                # diagnostics or in-flight spans do not establish completeness.
                if not status["accounting_complete"]:
                    raise RuntimeError("provider generation observation incomplete")
            except BaseException as exc:
                if primary is None:
                    raise
                if hasattr(primary, "add_note"):
                    primary.add_note("generation observation cleanup: " + type(exc).__name__)


def group_members(pgid):
    """Linux-only process scope inspection, including exited, unreaped members."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            parts = raw[raw.rfind(")") + 2:].split()
            if int(parts[2]) == pgid:
                members.append({"pid": int(entry.name), "state": parts[0], "start_ticks": int(parts[19])})
        except (OSError, ValueError):
            continue
    return members


@contextmanager
def child_subreaper():
    """Own launcher process adopts/reaps its provider's orphaned descendants."""
    if sys.platform != "linux" or not hasattr(os, "WNOWAIT"):
        raise ValueError("provider supervision currently requires Linux waitid/procfs")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "cannot establish child subreaper")
    try:
        yield
    finally:
        if libc.prctl(36, previous.value, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "cannot restore child subreaper")


def close_group(process, *, grace_s=10):
    # The leader has not been reaped. Its PID therefore cannot be reused for
    # another group between this signal and cleanup, even if it already exited.
    observed = {}
    reaped = []
    signals = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for member in group_members(process.pid):
            observed[member["pid"]] = member
        try:
            os.killpg(process.pid, sig)
            signals.append(sig.name)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + (grace_s if sig == signal.SIGTERM else 5)
        while time.monotonic() < deadline:
            members = group_members(process.pid)
            for member in members:
                observed[member["pid"]] = member
                if member["pid"] != process.pid:
                    try:
                        pid, _ = os.waitpid(member["pid"], os.WNOHANG)
                        if pid:
                            reaped.append(pid)
                    except ChildProcessError:
                        pass  # still a child of the living provider
            members = group_members(process.pid)
            if not any(m["pid"] != process.pid or m["state"] != "Z" for m in members):
                process.wait()
                return {"remaining": [], "observed": list(observed.values()),
                        "descendants_reaped": reaped, "signals": signals}
            time.sleep(.02)
    remaining = group_members(process.pid)
    # Do not advertise closure if an uninterruptible process survives SIGKILL.
    if os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
        process.wait()
    return {"remaining": remaining, "observed": list(observed.values()),
            "descendants_reaped": reaped, "signals": signals}


def group_rss_bytes(pgid):
    total = 0
    for member in group_members(pgid):
        try:
            for line in Path(f"/proc/{member['pid']}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass
    return total


def supervise(command, *, env, run_dir, timeout_s, stop_event=None, max_rss_bytes=12 * 1024**3):
    """Signal only the child we created; never discover/kill an existing server."""
    import threading
    if not 0 < timeout_s <= 900:
        raise ValueError("provider timeout must be finite and in (0,900]")
    if type(max_rss_bytes) is not int or max_rss_bytes <= 0:
        raise ValueError("RSS budget must be a positive byte count")
    stop = stop_event or threading.Event()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    start = time.monotonic()
    process = None
    reason = "natural_exit"
    peak_rss = 0
    try:
        with child_subreaper(), (run_dir / "provider.log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                write_json(run_dir / "owner.json", {"child_pid": process.pid, "supervisor_pid": os.getpid(),
                           "timeout_s": timeout_s, "command": command})
                while not os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                    peak_rss = max(peak_rss, group_rss_bytes(process.pid))
                    if peak_rss > max_rss_bytes:
                        reason = "memory_budget"
                        break
                    if stop.wait(.1) or time.monotonic() - start >= timeout_s:
                        reason = "requested_stop" if stop.is_set() else "deadline"
                        break
            except BaseException:
                reason = "supervisor_error"
                raise
            finally:
                group_closure = close_group(process)
                remaining = group_closure["remaining"]
                write_json(run_dir / "closure.json", {"reason": reason, "exit_code": process.returncode,
                           "leader_reaped": process.returncode is not None,
                           "owned_group_closed": not remaining, "remaining_group_members": remaining,
                           "group_cleanup": group_closure,
                           "sampled_peak_group_rss_bytes": peak_rss, "rss_budget_bytes": max_rss_bytes,
                           "memory_budget_scope": "100ms RSS watchdog, not a kernel hard limit; shared pages may be double counted",
                           "scope": "owned POSIX process group; provider must not daemonize into another session",
                           "wall_s": time.monotonic() - start})
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    admitted_closure = (process is not None and not remaining
                        and reason in {"natural_exit", "requested_stop"}
                        and "SIGKILL" not in group_closure["signals"])
    if admitted_closure:
        return process.returncode
    # A cooperative zero exit cannot turn a watchdog timeout or forced
    # descendant cleanup into successful supervision.
    return process.returncode if process is not None and process.returncode not in (None, 0) else 1


def serve(state, run_dir, *, port, timeout_s, trace_generation=False):
    verify(state)
    # Occupied endpoints are an error, never an invitation to stop their owner.
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    run_dir = run_dir.resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    command = [str(state / "venv/bin/python"), str(Path(__file__).resolve()),
               "_child", "--state-dir", str(state), "--run-dir", str(run_dir), "--port", str(port)]
    if trace_generation:
        command.append("--trace-generation")
    env = environment(state, offline=True)
    env["PYTHONPYCACHEPREFIX"] = str(run_dir / "bytecode")
    return supervise(command, env=env, run_dir=run_dir, timeout_s=timeout_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "serve", "_assets", "_child"])
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--port", type=int, default=18878)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--trace-generation", action="store_true",
                        help="private bounded decoded-output diagnostics, attached after model warmup")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 0 < args.timeout_s <= 900:
        parser.error("port must be [1024,65535], timeout finite and in (0,900]")
    state = private_state(args.state_dir, create=args.action == "prepare")
    # Internal subprocesses inherit the parent's lifetime lock; they do not
    # reacquire it. Public prepare/verify/serve are mutually exclusive.
    if args.action == "_assets":
        prepare_assets(state)
    elif args.action == "_child":
        child(state, args.port, args.run_dir, trace_generation=args.trace_generation)
    else:
        with state_lock(state):
            if args.action == "prepare":
                prepare(state)
            elif args.action == "verify":
                value = verify(state)
                print(json.dumps({"ok": True, "files": len(value["files_sha256"]),
                                  "prepared_sha256": digest(state / "prepared.json")}))
            else:
                if args.run_dir is None:
                    parser.error("serve requires a new --run-dir")
                code = serve(state, args.run_dir, port=args.port, timeout_s=args.timeout_s,
                             trace_generation=args.trace_generation)
                raise SystemExit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
