"""Fully offloaded Qwen worker supervised by the Brev runtime."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from urllib.request import build_opener, ProxyHandler, Request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG_TELEMETRY = re.compile(r"CUDA|CPU|GPU KV|buffer size|offload|SPLIT #|graph splits|print_timing|model loaded|server is listening|loading model|failed|error|warm|slot.*new slot|clip.*model|using.*device", re.I)
LLAMA_RUNTIME_SOURCE = re.compile(
    r"^(?:ggml_[a-z0-9_]+|load_tensors|llama_[a-z0-9_]+|clip_[a-z0-9_]+|"
    r"sched_reserve|reserve_compute_meta|system_info|common_init_from_params|"
    r"done_getting_tensors|warmup|set_warmup|slot\s+(?:print_timing|load_model)|"
    r"srv\s+(?:load_model|llama_server)):", re.I)


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def run(argv, timeout=20, check=True):
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=check)


def resolve_file(value, *, require_files):
    expanded = Path(value).expanduser()
    if any(c in str(expanded) for c in "*?["):
        found = sorted(expanded.parent.glob(expanded.name))
        if len(found) != 1 and require_files:
            raise ValueError(f"Expected one installed file for {value}; set the explicit CASCADE_GPU_MODEL/MMPROJ path")
        expanded = found[0] if len(found) == 1 else expanded
    path = expanded.resolve()
    if require_files and not path.is_file():
        raise ValueError(f"Required installed runtime/model file is unavailable: {path}")
    return str(path)


def make_plan(*, require_files=True):
    defaults = json.loads((HERE / "brain_qwen.json").read_text())
    fields = {}
    for field, env in (("server", "CASCADE_GPU_SERVER"), ("model_path", "CASCADE_GPU_MODEL"),
                       ("mmproj_path", "CASCADE_GPU_MMPROJ")):
        fields[field] = resolve_file(os.environ.get(env, defaults[field]), require_files=require_files)
    if require_files and not os.access(fields["server"], os.X_OK):
        raise ValueError("Installed llama-server is not executable")
    model_id = os.environ.get("CASCADE_GPU_MODEL_ID", defaults["model_id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", model_id):
        raise ValueError("Invalid served model alias")
    port = int(os.environ.get("CASCADE_GPU_PORT", "8041"))
    if not 1024 <= port <= 65535:
        raise ValueError("GPU brain needs an unprivileged loopback port")
    argv = [fields["server"], "--model", fields["model_path"], "--mmproj", fields["mmproj_path"],
            "--host", "127.0.0.1", "--port", str(port), "--alias", model_id,
            "--device", "CUDA0", "--n-gpu-layers", "all", "--override-tensor", ".*=CUDA0",
            "--fit", "off", "--kv-offload", "--op-offload", "--mmproj-offload",
            "--flash-attn", "on", "--jinja", "--reasoning", "off",
            "--ctx-size", str(defaults["context_window"]), "--parallel", "1",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--cache-ram", "0",
            "--threads", "8", "--threads-batch", "12", "--metrics", "--log-verbosity", "5"]
    digest = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
    return {"profile": "qwen", "backend": "llama.cpp CUDA", "model_id": model_id,
            **fields, "argv": argv, "host": "127.0.0.1", "port": port,
            "base_url": f"http://127.0.0.1:{port}/v1", "context_window": defaults["context_window"],
            "unit": "paai-demo", "cpu_fallback_allowed": False,
            "plan_sha256": digest, "plan_file": str(HERE / "brain-plan.json"),
            "state_file": str(HERE / "brain-state.json")}


def process_identity(pid):
    if type(pid) is not int or pid <= 1:
        return None
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z" or path.stat().st_uid != os.getuid():
            return None
        return {"pid": pid, "birth": fields[19], "ppid": int(fields[1]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "argv": (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
                "cwd": str((path / "cwd").resolve())}
    except (OSError, ValueError):
        return None


def same_process(a, b):
    return a is not None and b is not None and all(a.get(k) == b.get(k) for k in
        ("pid", "birth", "boot_id", "argv", "cwd"))


def owns_listener(pid, port):
    inodes = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            columns = line.split()
            if columns[3] == "0A" and int(columns[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(columns[9])
    try:
        for path in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                link = os.readlink(path)
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in inodes:
                return True
    except OSError:
        pass
    return False


def request(url, payload=None, timeout=5, text=False, status_only=False):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(url, data=data, headers={} if data is None else {"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
        if status_only:
            return response.status
        result = response.read(8 * 1024 * 1024).decode()
    return result if text else json.loads(result)


def endpoint_health(plan):
    url = plan["base_url"].removesuffix("/v1") + "/health"
    return request(url).get("status") == "ok"


def keep_runtime_line(line):
    # Match telemetry emitters so verbose request logs cannot retain prompts or images.
    message = re.sub(r"^\d+(?:\.\d+)+\s+[A-Z]\s+", "", line).lstrip()
    if LLAMA_RUNTIME_SOURCE.match(message):
        return bool(LOG_TELEMETRY.search(message))
    if re.match(r"^## SPLIT #\d+:\s+\w+", message):
        return True
    return False


def decode_log_line(raw):
    """Keep malformed native log bytes visible without exposing private lines.

    A native runtime may interleave/split UTF-8 output. Strict TextIO decoding
    must not take down an otherwise healthy CUDA model. Only the malformed
    bytes and their character offsets are recorded for excluded request lines;
    ordinary UTF-8 text and all CUDA enforcement remain unchanged.
    """
    line = raw.decode("utf-8", errors="surrogateescape")
    groups, byte_count = [], 0
    for match in re.finditer(r"[\udc80-\udcff]+", line):
        bad = match.group()
        byte_count += len(bad)
        if len(groups) < 16:
            groups.append({"character_offset": match.start(), "byte_count": len(bad),
                           "byte_escapes": "".join(f"\\x{ord(c)-0xdc00:02x}" for c in bad[:64])})
    if not byte_count:
        return line, None
    escaped = re.sub(r"[\udc80-\udcff]+",
                     lambda match: "".join(f"\\x{ord(c)-0xdc00:02x}" for c in match.group()), line)
    return escaped, {"invalid_byte_count": byte_count, "input_bytes": len(raw), "ranges": groups,
                     "byte_evidence_truncated": byte_count > sum(min(g["byte_count"], 64) for g in groups),
                     "private_line_content_recorded": False}


def reap_child(child, *, terminate=False):
    """Bounded ownership cleanup even if log filtering/recording raises."""
    if terminate and child.poll() is None:
        child.terminate()
    try:
        return child.wait(timeout=5 if terminate else 30)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait(timeout=5)
    finally:
        if child.stdout is not None:
            child.stdout.close()


def runtime_violation(line, profile):
    """One live telemetry line: never let requests masquerade as GPU evidence."""
    if not keep_runtime_line(line):
        return None
    if re.search(r"failed to initialize CUDA|no CUDA-capable device|CUDA error|out of memory", line, re.I):
        return "CUDA initialization/allocation failed"
    for device, kind, size in re.findall(r"(CUDA\w*|CPU\w*)\s+(model|KV) buffer size\s*=\s*([\d.]+) MiB", line):
        if (device.startswith("CPU") or device == "CUDA_Host") and float(size) > 0:
            return f"CPU/host {kind} allocation is forbidden"
    splits = re.findall(r"## SPLIT #\d+:\s+(\w+)", line)
    if any(not name.startswith("CUDA") or name == "CUDA_Host" for name in splits):
        return "Model compute split is not executing on CUDA"
    return None


def audit_log(log):
    log = "\n".join(line for line in log.splitlines() if keep_runtime_line(line))
    for line in log.splitlines():
        if reason := runtime_violation(line, "qwen"):
            raise ValueError(reason)
    counts = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    if not counts or any(int(a) != int(b) or int(b) <= 0 for a, b in counts):
        raise ValueError("All layers must be observed on CUDA")
    buffers = re.findall(r"(CUDA\w*|CPU\w*)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+) MiB", log)
    if any(kind in ("model", "KV") and (device.startswith("CPU") or device == "CUDA_Host") and float(size) > 0
           for device, kind, size in buffers):
        raise ValueError("CPU model/KV allocation is forbidden")
    if not any(device.startswith("CUDA") and device != "CUDA_Host" and kind == "model" and float(size) > 0
               for device, kind, size in buffers):
        raise ValueError("CUDA model buffer telemetry is missing")
    if not any(device.startswith("CUDA") and device != "CUDA_Host" and kind == "KV" and float(size) > 0
               for device, kind, size in buffers):
        raise ValueError("CUDA KV buffer telemetry is missing")
    splits = re.findall(r"## SPLIT #\d+:\s+(\w+)", log)
    if not splits or any(not name.startswith("CUDA") or name == "CUDA_Host" for name in splits):
        raise ValueError("Every observed model compute split must execute on CUDA")
    return {"all_layers_cuda": True, "layers": int(counts[-1][1]), "kv_cuda": True,
            "cuda_schedule": True, "cpu_fallback": False,
            "buffers_mib": [{"device": dev, "kind": kind, "mib": float(size)} for dev, kind, size in buffers],
            "scheduler_backends": sorted(set(splits)),
            "host_staging_explanation": "CUDA_Host buffers transfer inputs/outputs; they are not CPU model/KV or compute splits."}


def telemetry(pid):
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"], timeout=10).stdout.strip()
    rows = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits"], timeout=10).stdout.splitlines()
    pids = {str(pid)}
    own = [line for line in rows if line.split(",", 1)[0].strip() in pids]
    if not own:
        raise ValueError("The exact model PID is absent from NVIDIA CUDA telemetry")
    return {"gpu_csv": gpu, "process_csv": own, "at": now()}


def serve(plan):
    """Unit-only child supervisor, checking every emitted scheduling decision."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logfile = Path(plan["state_file"]).parent / "logs" / f"brain-{stamp}-{os.getpid()}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
    child_env.update(CUDA_VISIBLE_DEVICES="0", GGML_SCHED_DEBUG="1")
    child = subprocess.Popen(plan["argv"], cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=64 * 1024)
    def terminate(*_):
        if child.poll() is None:
            child.terminate()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    identity = None
    for _ in range(100):
        current = process_identity(child.pid)
        if current and current["argv"] == plan["argv"]:
            identity = current
            break
        time.sleep(.01)
    if identity is None:
        return reap_child(child, terminate=True) or 1
    state = {"plan_sha256": plan["plan_sha256"], "identity": identity, "started_at": now(),
             "log_file": str(logfile), "unit": plan["unit"], "native_tool_generation_verified": False,
             "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(plan["state_file"], state)
    # Preserve exact runtime telemetry lines. Exclude detailed request/prompt
    # dumps at verbosity 5, which are unnecessary for allocation proof.
    violation = None
    supervisor_error = None
    decode_recoveries = 0
    try:
        with logfile.open("w", buffering=1, encoding="utf-8") as stream:
            stream.write(json.dumps({"started_at": state["started_at"], "argv": plan["argv"],
                                     "log_scope": "exact selected runtime allocation/scheduler/timing lines; request bodies excluded"}) + "\n")
            for raw in child.stdout:
                line, decode_event = decode_log_line(raw)
                if decode_event:
                    decode_recoveries += 1
                    stream.write(json.dumps({"at": now(), "log_decode_recovery": decode_event}) + "\n")
                if keep_runtime_line(line):
                    stream.write(line)
                if reason := runtime_violation(line, plan["profile"]):
                    violation = reason + ": " + line.strip()
                    state.update(gpu_violation=violation, violation_at=now())
                    write(plan["state_file"], state)
                    terminate()
                    break
    except BaseException as error:
        supervisor_error = type(error).__name__
        raise
    finally:
        rc = reap_child(child, terminate=supervisor_error is not None or violation is not None)
        final = json.loads(Path(plan["state_file"]).read_text())
        final.update(ended_at=now(), return_code=rc, gpu_violation=violation,
                     log_decode_recoveries=decode_recoveries, supervisor_error=supervisor_error)
        write(plan["state_file"], final)
    return 2 if violation else rc
