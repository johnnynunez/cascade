#!/usr/bin/env python3
"""Extra-hop latency of the split conversation deployment: in-process vs plaintext vs TLS (B71).

`cascade-conversation` with `robot_endpoint` reaches its robot runtime through
`cascade.robot-runtime/1` instead of a function call. This measures what that hop
costs per call, on this host, with real `cascade-robot-service` child processes:

* in_process -- the same robot profile built in this process (`build_robot_runtime`), called directly;
* plaintext  -- a child service on loopback in clear (the default split deployment);
* tls        -- a child service on loopback with a throwaway private CA and server certificate
                (scripts/robot_endpoint_certs.py), the client pinned to that CA.

Each call is the client's real path (`RemoteRobotRuntime`): a new TCP connection per
request (HTTP/1.0), so a TLS call includes a full handshake (no session resumption).
Two operations: `state` (the generation read the conversation does per status/session,
GET /v1/state) and `execute` (one read-only tool through the robot runtime). Calls are
interleaved across the three modes so drift and load hit all of them alike.

    python scripts/bench_robot_endpoint.py --calls 500 --out latency.json

Reports n, min, p50, p95 (nearest rank), max and mean in milliseconds per mode and
operation, every sample, and the extra hop = remote minus in-process. It judges nothing: numbers
depend on the host and its load (recorded alongside). Loopback only -- a second host
is not measured here. Keys are throwaway and deleted at exit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
TOKEN_ENV = "BENCH_ROBOT_ENDPOINT_TOKEN"
MODES = ("in_process", "plaintext", "tls")


def stats(samples_s, ok=None):
    ms = sorted(sample * 1000.0 for sample in samples_s)
    rank = lambda q: ms[max(0, math.ceil(q * len(ms)) - 1)]   # nearest rank
    result = {"n": len(ms), "min_ms": ms[0], "p50_ms": rank(.5), "p95_ms": rank(.95), "max_ms": ms[-1],
              "mean_ms": sum(ms) / len(ms), "samples_ms": [round(sample * 1000.0, 6) for sample in samples_s]}
    if ok is not None:
        result["ok"] = ok
    return result


def succeeded(result):
    """A measured execute counts only if the robot runtime ran it (a motion result may be ok=false/unverified)."""
    return isinstance(result, dict) and result.get("execution_ok", result.get("ok")) is True


def _service(work, mode, token, robot, pki):
    run = work / f"robot-{mode}"
    argv = [sys.executable, "-m", "cascade.apps.robot_service", "--robot", robot, "--port", "0",
            "--run-dir", str(run), "--token-env", TOKEN_ENV, "--lease-ttl-s", "30"]
    if mode == "tls":
        argv += ["--tls-cert", pki["cert"], "--tls-key", pki["key"]]
    env = {key: value for key, value in os.environ.items() if not key.startswith("CASCADE_")}
    env.update({TOKEN_ENV: token, "PYTHONPATH": str(REPO / "src"), "CUDA_VISIBLE_DEVICES": "-1"})
    process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(64):
        line = process.stdout.readline()
        if not line:
            break
        if line.startswith("Robot runtime endpoint "):
            origin = line.split()[3]
            ready = json.loads((run / "ready.json").read_text())
            if ready["origin"] != origin or ready["pid"] != process.pid:
                raise RuntimeError(f"{mode}: ready.json does not belong to this child")
            return process, origin, ready
    process.kill()
    raise RuntimeError(f"{mode} robot service did not start: {process.stderr.read()}")


def _negotiated(origin, ca):
    """TLS version and cipher the client actually negotiates (one extra probe connection)."""
    from urllib.parse import urlsplit
    parts = urlsplit(origin)
    context = ssl.create_default_context(cafile=ca)
    with socket.create_connection((parts.hostname, parts.port), timeout=10) as raw:
        with context.wrap_socket(raw, server_hostname=parts.hostname) as tls:
            return {"version": tls.version(), "cipher": tls.cipher()[0]}


def measure(args):
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import load_robot_config
    from cascade.robotics.endpoint import PROTOCOL, RemoteRobotRuntime
    from robot_endpoint_certs import issue
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="bench-robot-endpoint-"))
    work.mkdir(parents=True, exist_ok=True)
    pki_dir = work / "pki"
    token = secrets.token_urlsafe(32)
    processes, clients, runtime = [], {}, None
    load_start = os.getloadavg() if hasattr(os, "getloadavg") else None
    try:
        pki = issue(pki_dir, ["IP:127.0.0.1"])
        services = {}
        for mode in ("plaintext", "tls"):
            process, origin, ready = _service(work, mode, token, args.robot, pki)
            processes.append(process)
            services[mode] = {"origin": origin, "ready": ready}
        if services["tls"]["ready"].get("tls_certificate_sha256") != pki["certificate_sha256"]:
            raise RuntimeError("TLS service does not serve the generated certificate")
        clients["plaintext"] = RemoteRobotRuntime(services["plaintext"]["origin"], token).connect()
        clients["tls"] = RemoteRobotRuntime(services["tls"]["origin"], token, tls_ca=pki["ca"]).connect()
        (work / "in-process").mkdir(mode=0o700)
        runtime, _ = build_robot_runtime(load_robot_config(args.robot), work / "in-process")
        if args.tool not in runtime.tool_descriptors or runtime.tool_descriptors[args.tool].effect != "read":
            raise SystemExit(f"--tool must name a read-only tool of {args.robot}")
        generation = {"in_process": runtime.cancellation_token,
                      **{mode: client.cancellation_token for mode, client in clients.items()}}
        state = {"in_process": lambda: (runtime.cancellation_token, runtime.stopped),
                 **{mode: (lambda c=client: c.cancellation_token) for mode, client in clients.items()}}
        targets = {"in_process": runtime, **clients}
        samples = {mode: {"state": [], "execute": []} for mode in MODES}
        ok = dict.fromkeys(MODES, 0)
        for index in range(args.warmup + args.calls):
            order = MODES[index % 3:] + MODES[:index % 3]
            for mode in order:
                started = time.perf_counter()
                state[mode]()
                elapsed_state = time.perf_counter() - started
                started = time.perf_counter()
                result = targets[mode].execute(args.tool, {}, expected_generation=generation[mode],
                                               deadline_monotonic_s=time.monotonic() + 30)
                elapsed_execute = time.perf_counter() - started
                if index >= args.warmup:
                    samples[mode]["state"].append(elapsed_state)
                    samples[mode]["execute"].append(elapsed_execute)
                    ok[mode] += succeeded(result)
        results = {mode: {"state": stats(samples[mode]["state"]), "execute": stats(samples[mode]["execute"], ok[mode])}
                   for mode in MODES}
        hop = {mode: {operation: {key: results[mode][operation][key] - results["in_process"][operation][key]
                                  for key in ("p50_ms", "p95_ms")}
                      for operation in ("state", "execute")}
               for mode in ("plaintext", "tls")}
        negotiated = _negotiated(services["tls"]["origin"], pki["ca"])
        return {"schema": 1, "protocol": PROTOCOL, "robot": args.robot, "tool": args.tool, "calls": args.calls,
                "warmup": args.warmup, "connection": "one new TCP connection per call (HTTP/1.0); TLS = full "
                                                     "handshake per call, no session resumption; loopback",
                "tls": {"server_key": "rsa:2048", "ca_key": "rsa:3072", "signature": "sha256",
                        "certificate_sha256": pki["certificate_sha256"], **negotiated},
                "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                         "openssl": ssl.OPENSSL_VERSION, "cpus": os.cpu_count(), "loadavg_start": load_start,
                         "loadavg_end": os.getloadavg() if hasattr(os, "getloadavg") else None},
                "results": results, "extra_hop_ms": hop}
    finally:
        for client in clients.values():
            client.close()
        for process in processes:
            process.send_signal(signal.SIGTERM)
        for process in processes:
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if runtime is not None:
            runtime.close()
        shutil.rmtree(pki_dir, ignore_errors=True)          # throwaway keys never outlive the run
        if not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", type=int, default=500, help="measured calls per mode and operation")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured calls first")
    parser.add_argument("--robot", default="microduck_conversation_mock", help="robot profile (mock by default)")
    parser.add_argument("--tool", default="locomotion.get_base_state", help="read-only tool to execute")
    parser.add_argument("--out", type=Path, required=True, help="JSON report")
    parser.add_argument("--work-dir", type=Path, help="keep run directories here (keys are still deleted)")
    args = parser.parse_args(argv)
    if args.calls < 1 or args.warmup < 0:
        parser.error("--calls must be >= 1 and --warmup >= 0")
    report = measure(args)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for mode in MODES:
        for operation in ("state", "execute"):
            row = report["results"][mode][operation]
            print(f"{mode:10s} {operation:7s} p50 {row['p50_ms']:.3f} ms  p95 {row['p95_ms']:.3f} ms  (n={row['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
