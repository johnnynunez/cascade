#!/usr/bin/env python3
"""Read-only platform, disk and GPU admission for the selected Brev profile."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import socket
import subprocess

GIB = 1024 ** 3
DOCKER_SOCKET = "unix:///run/paai-demo-docker.sock"
COMPOSE_PROJECT = "paai-demo"
PROC_ROOT = Path("/proc")
CONTAINER_FORMAT = (
    '{"id":{{json .Id}},"running":{{json .State.Running}},"pid":{{json .State.Pid}},'
    '"project":{{json (index .Config.Labels "com.docker.compose.project")}},'
    '"service":{{json (index .Config.Labels "com.docker.compose.service")}},'
    '"working_dir":{{json (index .Config.Labels "com.docker.compose.project.working_dir")}},'
    '"config_files":{{json (index .Config.Labels "com.docker.compose.project.config_files")}},'
    '"mounts":{{json .Mounts}}}'
)


def positive_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def output(command, timeout=15):
    result = subprocess.run(command, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed with exit {result.returncode}; output withheld")
    return result.stdout.strip()


def validate_storage(profile, *, installed=False):
    storage = profile["storage"]
    mount = Path(storage["mount"]).resolve(strict=True)
    root = Path(storage["root"]).resolve()
    if root == mount or not root.is_relative_to(mount):
        raise ValueError("Deployment root must be a dedicated child of the data volume")
    if mount.stat().st_dev == Path("/").stat().st_dev or not os.path.ismount(mount):
        raise ValueError("The data volume is not mounted separately from the root disk")
    existing = root
    while not existing.exists():
        existing = existing.parent
    if not existing.is_dir() or existing.stat().st_dev != mount.stat().st_dev:
        raise ValueError("Deployment root must reside on the configured data-volume filesystem")
    fs = json.loads(output(["findmnt", "--json", "--target", str(mount),
                            "--output", "TARGET,SOURCE,FSTYPE"]))["filesystems"][0]
    if Path(fs["target"]).resolve() != mount or fs["fstype"] != "ext4":
        raise ValueError("The expected ext4 data volume is not mounted at the configured path")
    usage, system = shutil.disk_usage(mount), shutil.disk_usage("/")
    minimum = 20 if installed else positive_number(storage["min_available_gib"], "Data disk reserve")
    if usage.free < minimum * GIB:
        raise ValueError(f"Data volume needs at least {minimum} GiB free")
    if system.free < positive_number(storage["min_root_available_gib"], "Root disk reserve") * GIB:
        raise ValueError("Root disk has insufficient operating-system headroom")
    return {"mount": str(mount), "root": str(root), "filesystem": fs["fstype"],
            "data_available_gib": round(usage.free / GIB, 2),
            "root_available_gib": round(system.free / GIB, 2)}


def validate_gpu(profile, raw, *, running=False):
    rows = [line.split(",") for line in raw.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError("Exactly one NVIDIA GPU is required")
    name, total, used, free, capability, driver = (part.strip() for part in rows[0])
    selected = profile["accelerator"]
    if name != selected["name"] or capability != selected["compute_capability"]:
        raise ValueError("The GPU name or compute capability differs from the selected profile")
    total, used, free = float(total), float(used), float(free)
    if not all(math.isfinite(value) and value >= 0 for value in (total, used, free)):
        raise ValueError("GPU memory telemetry is invalid")
    if used > total or free > total or used + free > total + 2:
        raise ValueError("GPU memory telemetry is inconsistent")
    if total < positive_number(selected["min_vram_mib"], "GPU VRAM requirement"):
        raise ValueError("GPU VRAM is below the profile requirement")
    reserve = positive_number(selected["minimum_headroom_mib"] if running
                              else selected["minimum_free_before_start_mib"], "GPU free VRAM reserve")
    if free < reserve:
        raise ValueError(f"GPU free VRAM is below the required {reserve} MiB reserve")
    return {"name": name, "memory_total_mib": total, "memory_used_mib": used,
            "memory_free_mib": free, "compute_capability": capability, "driver": driver}


def process_cgroups(pid):
    groups = {}
    for line in (PROC_ROOT / str(pid) / "cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].startswith("/"):
            raise ValueError("Process cgroup telemetry is invalid")
        path = PurePosixPath(fields[2])
        if ".." in path.parts:
            raise ValueError("Process cgroup path is not rooted in the host namespace")
        key = (fields[0], fields[1])
        if key in groups:
            raise ValueError("Process cgroup telemetry is ambiguous")
        groups[key] = path
    if not groups:
        raise ValueError("Process cgroup telemetry is empty")
    return groups


def process_identity(pid):
    path = PROC_ROOT / str(pid)
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    if len(fields) < 20 or not fields[1].isdigit() or not fields[19].isdigit():
        raise ValueError("Process identity telemetry is invalid")
    return {"parent": int(fields[1]), "start_ticks": int(fields[19]),
            "executable": os.readlink(path / "exe"),
            "argv": (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
            "cgroups": process_cgroups(pid)}


def validate_public_video(profile, pids):
    settings = profile.get("public_visitor", {})
    if (not isinstance(settings, dict) or settings.get("enabled") is not True
            or settings.get("video") is not True):
        raise ValueError("A GPU process is outside the exact owned container")
    if len(pids) > 3:
        raise ValueError("Public video permits at most three camera encoders")
    if type(settings.get("local_auth", False)) is not bool:
        raise ValueError("Public visitor local_auth must be a boolean")
    root = Path(profile["storage"]["root"]).resolve()
    service_group = PurePosixPath("/system.slice/paai-visitor.service")
    expected_groups = {("0", ""): service_group}
    expected_parent = ["/usr/bin/python3", str(root / "tools/visitor/visitor.py")]
    expected_parent.append("--auth-file=" + str(root / "data/private/public-visitor/auth.json"))
    expected_parent.append("--video")
    command = ["systemctl", "show", "paai-visitor.service", "--no-pager",
               "--property=ActiveState,MainPID,User,ExecStart,ControlGroup,FragmentPath,DropInPaths"]

    def service():
        return dict(line.split("=", 1) for line in output(command).splitlines() if "=" in line)

    unit = service()
    main_pid = unit.get("MainPID", "")
    match = re.search(r"argv\[\]=([^;]+);", unit.get("ExecStart", ""))
    if (unit.get("ActiveState") != "active" or unit.get("User") != "ubuntu"
            or not main_pid.isdigit() or int(main_pid) <= 0
            or unit.get("ControlGroup") != str(service_group) or unit.get("DropInPaths")
            or unit.get("FragmentPath") != "/etc/systemd/system/paai-visitor.service"
            or match is None or shlex.split(match[1]) != expected_parent):
        raise ValueError("Public video does not belong to the expected active visitor service")
    main_pid = int(main_pid)
    parent = process_identity(main_pid)
    if (parent["argv"] != expected_parent or parent["cgroups"] != expected_groups
            or parent["executable"] != str(Path("/usr/bin/python3").resolve())):
        raise ValueError("The visitor main process does not match its service")
    spec = importlib.util.spec_from_file_location("visitor_encoder_profile", Path(__file__).with_name("visitor_video.py"))
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    expected = {camera: encoder.encoder_command("/usr/bin/ffmpeg", "http://127.0.0.1:8091", camera)
                for camera in encoder.CAMERAS}
    identities, cameras = {}, set()
    for pid in sorted(pids):
        identity = process_identity(pid)
        camera = next((name for name, argv in expected.items() if identity["argv"] == argv), None)
        if (identity["parent"] != main_pid or identity["executable"] != "/usr/bin/ffmpeg"
                or identity["cgroups"] != expected_groups or camera is None or camera in cameras):
            raise ValueError(f"GPU compute PID {pid} is not an owned public camera encoder")
        identities[pid] = identity
        cameras.add(camera)
    if service() != unit or process_identity(main_pid) != parent:
        raise ValueError("The public visitor changed during GPU ownership admission")
    for pid, identity in identities.items():
        if process_identity(pid) != identity:
            raise ValueError("A public video process changed during GPU ownership admission")
    return {"service": "paai-visitor.service", "main_pid": main_pid,
            "encoder_pids": sorted(pids), "cameras": sorted(cameras)}


def validate_gpu_ownership(profile, processes, docker_root):
    root = Path(profile["storage"]["root"]).resolve()
    if Path(docker_root).resolve() != root / "docker":
        raise ValueError("Running admission requires the dedicated Docker data root")
    config_files = [str(root / "deployment/compose.yaml")]
    streaming = profile.get("streaming", {}).get("enabled", False)
    if type(streaming) is not bool:
        raise ValueError("The streaming profile needs an explicit boolean enabled value")
    if streaming:
        config_files.append(str(root / "deployment/compose.streaming.yaml"))
    docker = ["docker", "--host", DOCKER_SOCKET]
    ids = output([*docker, "ps", "--quiet", "--no-trunc", "--filter", "status=running",
                  "--filter", f"label=com.docker.compose.project={COMPOSE_PROJECT}",
                  "--filter", "label=com.docker.compose.service=demo"]).splitlines()
    if len(ids) != 1 or not re.fullmatch(r"[0-9a-f]{64}", ids[0]):
        raise ValueError("Running admission requires one exact owned demo container")
    container_id = ids[0]
    inspect_command = [*docker, "inspect", "--type", "container", "--format",
                       CONTAINER_FORMAT, container_id]
    metadata = json.loads(output(inspect_command))
    if (metadata.get("id") != container_id or metadata.get("running") is not True
            or type(metadata.get("pid")) is not int or metadata["pid"] <= 0
            or metadata.get("project") != COMPOSE_PROJECT or metadata.get("service") != "demo"
            or metadata.get("working_dir") != str(root / "deployment")
            or metadata.get("config_files") != ",".join(config_files)):
        raise ValueError("Container metadata does not identify the owned deployment")
    mounts = {row.get("Destination"): row for row in metadata.get("mounts", [])}
    for destination, source in (("/workspace", root / "source"), ("/data", root / "data"),
                                ("/data/models", root / "models")):
        row = mounts.get(destination, {})
        if row.get("Type") != "bind" or Path(row.get("Source", "")).resolve() != source:
            raise ValueError("Running container mounts do not match the owned deployment")
    init_pid = metadata["pid"]
    init_groups = process_cgroups(init_pid)
    owned_groups = {}
    names = {container_id, f"docker-{container_id}.scope"}
    for key, path in init_groups.items():
        for index, part in enumerate(path.parts):
            if part in names:
                owned_groups[key] = PurePosixPath(*path.parts[:index + 1])
                break
    if not owned_groups:
        raise ValueError("The container init process has no verifiable Docker cgroup")
    pids = set()
    for row in csv.reader(processes.splitlines()):
        if len(row) != 3 or not row[0].strip().isdigit() or int(row[0]) <= 0:
            raise ValueError("GPU compute process telemetry has an invalid PID")
        pids.add(int(row[0]))
    outside = set()
    for pid in sorted(pids):
        groups = process_cgroups(pid)
        if any(key not in groups or not groups[key].is_relative_to(owned)
               for key, owned in owned_groups.items()):
            outside.add(pid)
    public_video = validate_public_video(profile, outside) if outside else None
    final = json.loads(output(inspect_command))
    if (final.get("id"), final.get("pid"), final.get("running")) != (container_id, init_pid, True):
        raise ValueError("Container changed during GPU ownership admission")
    return {"verified": True, "docker_socket": DOCKER_SOCKET, "container_id": container_id,
            "init_pid": init_pid, "gpu_compute_pids": sorted(pids),
            "public_video": public_video,
            "container_cgroups": sorted({str(path) for path in owned_groups.values()})}


def check(profile, *, installed=False, running=False):
    if platform.machine() != profile["architecture"]:
        raise ValueError("Target architecture does not match the selected profile")
    if output(["id", "-un"]) != "ubuntu":
        raise ValueError("The selected Brev profile requires the ubuntu user")
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError("Admission must run as the unprivileged ubuntu account")
    output(["sudo", "-n", "true"])
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    ram_gib = int(memory["MemTotal"].split()[0]) / 1024 ** 2
    system = profile.get("system", {})
    min_cpu = positive_number(system.get("min_vcpu", 8), "CPU requirement")
    min_ram = positive_number(system.get("min_ram_gib", 55), "RAM requirement")
    if (os.cpu_count() or 0) < min_cpu or ram_gib < min_ram:
        raise ValueError(f"Target needs at least {min_cpu:g} vCPUs and {min_ram:g} GiB RAM")
    disk = validate_storage(profile, installed=installed)
    gpu = validate_gpu(profile, output(["nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version",
        "--format=csv,noheader,nounits"]), running=running)
    processes = output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                        "--format=csv,noheader,nounits"])
    if processes and not running:
        raise ValueError("An existing GPU compute process must be identified before a new workload starts")
    docker_command = ["docker", "--host", DOCKER_SOCKET] if running else ["docker"]
    docker = output([*docker_command, "version", "--format", "{{.Server.Version}}"])
    docker_root = output([*docker_command, "info", "--format", "{{.DockerRootDir}}"])
    ownership = validate_gpu_ownership(profile, processes, docker_root) if running else None
    return {"status": "PASS", "checked_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(), "architecture": platform.machine(),
            "uid": uid, "gid": gid,
            "cpu_count": os.cpu_count(), "ram_gib": round(ram_gib, 2), "disk": disk,
            "gpu": gpu, "gpu_processes": processes.splitlines(),
            "gpu_ownership": ownership,
            "docker_version": docker, "docker_root": docker_root,
            "dedicated_docker_required": not Path(docker_root).resolve().is_relative_to(Path(disk["mount"])),
            "root_or_driver_changes": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--running", action="store_true")
    args = parser.parse_args()
    try:
        result = check(json.loads(args.profile.read_text()), installed=args.installed, running=args.running)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        result = {"status": "FAIL", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "reason": str(error) if not isinstance(error, subprocess.TimeoutExpired)
                  else "Finite preflight command timed out"}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
