#!/usr/bin/env python3
"""Deploy the reviewed kitchen bundle from controller through the Tailscale wrapper."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time

import access_gate
import prepare_runtime
import prepare_bundle
import public as visitor_deployment
from streaming import deployment as camera_deployment

HERE = Path(__file__).resolve().parent
STOP = HERE.parents[1] / "STOP"
CONTROL = "/opt/dlami/nvme/paai-demo/control"
DOCKER_SOCKET = "unix:///run/paai-demo-docker.sock"
COMPOSE_PROJECT = "paai-demo"
OPENCLAW_SHA = "73c595199cd2a4cc46e503e09101318f09c21ca878596f662a540d32cd7243bb"
DEPLOYMENT_FILES = ("compose.yaml", "compose.streaming.yaml", "container/Dockerfile", "container/requirements.txt",
                    "container/entrypoint.py", "container/materialize_ui.mjs",
                    "container/clip-source.tar.gz", "container/.dockerignore")
CONTROL_FILES = ("preflight.py", "visitor_video.py", "host_setup.py", "fetch_models.py", "prepare_runtime.py",
                 "profiles/qwen3.8-27b-q8.json", "streaming/install_network.py",
                 "streaming/network_guard.py", "streaming/network_service.py")
PREPARATION_UNIT_GUARD = """import json,pathlib,re,shlex,subprocess,sys
root,helper,unit,operation=sys.argv[1:]
root=str(pathlib.Path(root).resolve())
command=['systemctl','show',unit,'--no-pager','--property=LoadState,ActiveState,ExecStart,User,FragmentPath,DropInPaths']
r=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=15)
if r.returncode: raise RuntimeError('Cannot establish preparation ownership')
p=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
active=p.get('ActiveState') in ('activating','active','reloading','deactivating')
if p.get('LoadState')=='not-found' and not active:
    print(json.dumps({'active':False,'unit':unit,'root':root}))
    raise SystemExit(0)
match=re.search(r'argv\\[\\]=([^;]+);',p.get('ExecStart',''))
argv=shlex.split(match[1]) if match else []
if (p.get('LoadState')!='loaded' or p.get('User')!='ubuntu' or p.get('DropInPaths')
        or p.get('FragmentPath')!='/run/systemd/transient/'+unit
        or len(argv)!=5 or argv[0] not in ('python3','/usr/bin/python3')
        or argv[1:]!=[helper,'run','--root',root]):
    raise ValueError('The preparation unit belongs to another deployment or is masked')
if operation=='stop' and active:
    r=subprocess.run(['sudo','-n','systemctl','stop',unit],stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=35)
    if r.returncode: raise RuntimeError('Owned preparation did not stop')
print(json.dumps({'active':active,'unit':unit,'root':root}))
"""
STOP_CONTAINER_FORMAT = ('{"id":{{json .Id}},"labels":{{json .Config.Labels}},'
                         '"mounts":{{json .Mounts}}}')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def boundary():
    if STOP.exists():
        raise InterruptedError("Repository STOP exists")


def execute(argv, *, timeout=60, log=None):
    boundary()
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(log, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        os.fchmod(descriptor, 0o600)
        stream_context = os.fdopen(descriptor, "w+")
    else:
        stream_context = tempfile.TemporaryFile(mode="w+")
    with stream_context as stream:
        child = subprocess.Popen(list(map(str, argv)), stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while child.poll() is None:
                boundary()
                if time.monotonic() > deadline:
                    raise TimeoutError("Finite deployment operation timed out")
                time.sleep(.25)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            raise
        stream.seek(0)
        output = stream.read(4 * 1024 * 1024)
    if child.returncode:
        raise RuntimeError(f"{Path(str(argv[0])).name} failed with exit {child.returncode}; see the operation log")
    return output


def remote(argv, *, timeout=60, log=None):
    command = shlex.join(["timeout", "--signal=TERM", str(max(1, timeout - 5)), *map(str, argv)])
    return execute([access_gate.WRAPPER, "-F", access_gate.PRIVATE_CONFIG,
                    access_gate.ALIAS, command], timeout=timeout, log=log)


def sync(source, destination, files, *, timeout=900, log=None):
    boundary()
    source = Path(source).resolve()
    selected = sorted(set(files))
    for value in selected:
        source_member(source, value)
    with tempfile.NamedTemporaryFile(mode="w", prefix="paai-transfer-", delete=False) as stream:
        stream.write("\n".join(selected) + "\n")
        file_list = Path(stream.name)
    try:
        transport = shlex.join([str(access_gate.WRAPPER), "-F", str(access_gate.PRIVATE_CONFIG)])
        return execute(["rsync", "-ar", "--partial", "--timeout=60", "--protect-args",
                        "--files-from", file_list, "-e", transport,
                        str(source) + "/", f"{access_gate.ALIAS}:{destination}/"], timeout=timeout, log=log)
    finally:
        file_list.unlink(missing_ok=True)


def update_gate(name, status, evidence):
    path = HERE / "STATE.json"
    if path.exists():
        state = json.loads(path.read_text())
    else:
        state = {"campaign": "Brev kitchen deployment", "started_at": stamp(),
                 "status": "WAITING_FOR_ACCESS", "blockers": [],
                 "target": {"instance": access_gate.TARGET},
                 "gates": {key: {"status": "pending", "evidence": []} for key in (
                     "controller_access", "preflight", "qwen_gpu_vision_tools",
                     "isaac_kitchen_cameras", "authenticated_visitor_openclaw",
                     "physics_verified_pick", "chrome_media_streaming", "publication_cleanup",
                     "one_command_deploy", "representative_assets")}}
    state["gates"][name] = {"status": status, "evidence": [evidence]}
    state["updated_at"] = stamp()
    if name == "controller_access":
        state["status"] = "IN_PROGRESS" if status == "PASS" else "WAITING_FOR_ACCESS"
        state["blockers"] = [] if status == "PASS" else ["Controller-to-Brev Tailscale access has not passed"]
    state["first_unfinished_gate"] = next((key for key, value in state["gates"].items()
                                           if value["status"] != "PASS"), None)
    atomic(path, state)
    if not (HERE / "READY.json").exists():
        atomic(HERE / "READY.json", {"status": "NOT_READY", "checked_at": stamp(),
               "first_unfinished_gate": state["first_unfinished_gate"],
               "reason": "Live deployment, visitor, physics and media gates remain unproven."})


def require_access(profile=None):
    boundary()
    if profile is not None:
        access_gate.select_profile(profile)
    record = access_gate.probe()
    update_gate("controller_access", record["status"], record.get("evidence", "access/current.json"))
    if record["status"] != "PASS":
        raise ConnectionError(record.get("reason", "Access gate has not passed"))
    return record


def compose(profile, *args, timeout=120, log=None):
    deployment = profile["storage"]["root"] + "/deployment"
    files = ["-f", deployment + "/compose.yaml"]
    if camera_deployment.enabled(profile):
        files += ["-f", deployment + "/compose.streaming.yaml"]
    return remote(["docker", "--host", DOCKER_SOCKET, "compose", "--project-name", COMPOSE_PROJECT,
                   "--env-file", deployment + "/site.env",
                   *files, *args], timeout=timeout, log=log)


def prepare_camera_video(profile, ip):
    files = camera_deployment.configuration(profile, ip)
    if not files:
        return
    output = remote(["python3", CONTROL + "/streaming/install_network.py"], timeout=90,
                    log=HERE / "logs/camera-network-setup.log")
    admission = json.loads(output)
    if admission.get("status") != "PASS_HOST_SETUP" or admission.get("active") is not True:
        raise ValueError("The host camera firewall monitor has not started")
    atomic(HERE / "proof/CAMERA_NETWORK_INSTALL.json", admission)
    with tempfile.TemporaryDirectory(prefix="paai-camera-config-") as directory:
        source = Path(directory)
        for name, value in files.items():
            (source / name).write_text(value)
        sync(source, profile["storage"]["root"] + "/deployment", files)
    image = camera_deployment.RELAY_IMAGE
    inspect = ["docker", "--host", DOCKER_SOCKET, "image", "inspect", image, "--format", "{{.Id}}"]
    reused = True
    try:
        image_id = remote(inspect).strip()
    except RuntimeError:
        reused = False
        remote(["docker", "--host", DOCKER_SOCKET, "pull", image], timeout=300,
               log=HERE / "logs/camera-relay-pull.log")
        image_id = remote(inspect).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("The prepared camera relay has no image identity")
    atomic(HERE / "proof/CAMERA_RELAY_IMAGE.json", {"checked_at": stamp(), "image": image,
           "image_id": image_id, "reused": reused, "live_media_verified": False})


def preflight(profile_path, *, installed=False, running=False):
    profile = json.loads(profile_path.read_text())
    if remote_preparation(profile)["active"]:
        raise RuntimeError("Preparation is active; its control files must remain unchanged")
    remote(["install", "-d", "-m", "0700", CONTROL])
    sync(HERE, CONTROL, CONTROL_FILES)
    # A selected external profile is staged as a single controlled filename.
    with tempfile.TemporaryDirectory(prefix="paai-profile-") as directory:
        local = Path(directory)
        (local / "profile.json").write_bytes(profile_path.read_bytes())
        sync(local, CONTROL, ["profile.json"])
    command = ["python3", CONTROL + "/preflight.py", "--profile", CONTROL + "/profile.json"]
    if installed:
        command.append("--installed")
    if running:
        command.append("--running")
    log = HERE / "logs/preflight-latest.log"
    try:
        output = remote(command, log=log)
    except (RuntimeError, TimeoutError):
        try:
            record = json.loads(log.read_text())
            if not isinstance(record, dict) or record.get("status") != "FAIL":
                raise ValueError("No complete failed preflight record")
        except (OSError, ValueError):
            record = {"status": "FAIL", "checked_at": stamp(),
                      "reason": "Remote preflight failed or timed out; inspect the private operation log"}
        atomic(HERE / "proof/PREFLIGHT.json", record)
        update_gate("preflight", "FAIL", "proof/PREFLIGHT.json")
        raise
    record = json.loads(output)
    if not isinstance(record, dict) or record.get("status") != "PASS":
        raise ValueError("Remote preflight did not return a passing admission record")
    atomic(HERE / "proof/PREFLIGHT.json", record)
    update_gate("preflight", record["status"], "proof/PREFLIGHT.json")
    return record


def source_member(source, value):
    if not isinstance(value, str):
        raise ValueError("Transfer members must be relative path strings")
    relative = Path(value)
    if (relative.is_absolute() or ".." in relative.parts or not relative.parts
            or "\n" in value or "\r" in value):
        raise ValueError("Transfer members must be relative paths without traversal")
    target = source / relative
    if not target.resolve().is_relative_to(source) or not target.is_file():
        raise ValueError("Transfer member is missing or leaves the selected source")
    return target


def staged_bundle():
    record = json.loads((HERE / "publication/STAGED.json").read_text())
    if record.get("status") not in ("PREPARED", "PASS", "STAGED", "COMPLETE"):
        raise ValueError("The reviewed source bundle is not complete")
    root = Path(record["staged_root"]).resolve()
    files = record["deploy_files"]
    if (not files or not (root / "src/cascade").is_dir() or len(set(files)) != len(files)
            or record.get("deploy_file_count") != len(files)):
        raise ValueError("The reviewed deploy manifest is incomplete")
    receipts = record.get("deploy_receipts", {})
    if not isinstance(receipts, dict) or set(receipts) != set(files):
        raise ValueError("Every deploy member needs a certified source receipt")
    total = 0
    for name in files:
        entry = source_member(root, name)
        metadata = entry.stat()
        receipt = receipts[name]
        if (not isinstance(receipt, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", "")))
                or not receipt.get("provenance")):
            raise ValueError(f"Source checksum receipt is invalid: {name}")
        current = (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
        certified = tuple(receipt.get(key) for key in ("size_bytes", "mtime_ns", "ctime_ns"))
        if current != certified:
            raise ValueError(f"Source changed after certification: {name}")
        approved = record.get("source_sha256", {}).get(name)
        if approved is not None and receipt["sha256"] != approved:
            raise ValueError(f"Source receipt differs from the approved source checksum: {name}")
        total += metadata.st_size
    if total != record.get("deploy_bytes"):
        raise ValueError("Source byte count differs from the reviewed deploy manifest")
    return root, files


def bundle_identity(source, files, profile):
    record = json.loads((HERE / "publication/STAGED.json").read_text())
    if (record.get("upstream_revision") != profile["source"]["revision"]
            or record.get("upstream_url") != profile["source"]["origin"]):
        raise ValueError("Selected source profile differs from the reviewed upstream revision")
    entries = [[name, record["deploy_receipts"][name]["sha256"]] for name in sorted(files)]
    inputs = {name: hashlib.sha256(source_member(HERE, name).read_bytes()).hexdigest()
              for name in (*DEPLOYMENT_FILES, *CONTROL_FILES, "deploy.py", "access_gate.py", "prepare_bundle.py",
                           "streaming/deployment.py", "streaming/camera_stream_profile.py")}
    payload = json.dumps({"files": entries, "profile": profile, "deployment_inputs": inputs},
                         sort_keys=True).encode()
    # Full asset hashes are reused only after receipt metadata matches.
    return hashlib.sha256(payload).hexdigest()


def openclaw_archive():
    if not (HERE / "publication/LOCAL_INPUTS.json").is_file():
        raise ValueError("Run prepare with the reviewed OpenClaw archive before installation")
    return prepare_bundle.admitted_openclaw(HERE)


def remote_installation(profile):
    path = profile["storage"]["root"] + "/deployment/installed.json"
    code = "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.is_file() else '{}')"
    return json.loads(remote(["python3", "-c", code, path]))


def remote_preparation(profile):
    code = """import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1])/'deployment'
def read(name):
    p=root/name
    if not p.exists(): return {}
    if p.is_symlink() or not p.is_file() or p.stat().st_size>65536:
        raise ValueError('Invalid preparation record')
    return json.loads(p.read_text())
r=subprocess.run(['systemctl','show',sys.argv[2],'--property=ActiveState','--value'],
                 stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=15)
if r.returncode: raise RuntimeError('Cannot inspect supervised preparation')
state=r.stdout.strip()
print(json.dumps({'unit_state':state,'active':state in ('activating','active','reloading','deactivating'),
                  'plan_identity':read('install-plan.json').get('bundle_identity'),
                  'preparation':read('preparation.json')}))
"""
    return json.loads(remote(["python3", "-c", code, profile["storage"]["root"], prepare_runtime.UNIT]))


def cancel_preparation():
    # This is the sole cleanup command allowed after the controller stop marker.
    if not (HERE / "access/PASSED.json").is_file():
        return {"attempted": False, "reason": "No prior access proof"}
    try:
        proof = json.loads((HERE / "access/PASSED.json").read_text())
        if proof.get("status") != "PASS":
            raise ValueError("No passing access receipt")
        target, alias, private_config = access_gate.target_config(proof.get("target"))
        if proof.get("alias") != alias:
            raise ValueError("The access receipt has no matching private alias")
        owner = json.loads((HERE / "proof/PREPARATION_OWNER.json").read_text())
        if (owner.get("target") != target or not isinstance(owner.get("root"), str)
                or not Path(owner["root"]).is_absolute()):
            raise ValueError("Preparation has no matching deployment owner")
    except (OSError, ValueError, AttributeError):
        return {"attempted": False, "reason": "Prior access proof has no valid target binding"}
    command = shlex.join(["timeout", "55", "python3", "-c", PREPARATION_UNIT_GUARD,
                          owner["root"], CONTROL + "/prepare_runtime.py", prepare_runtime.UNIT, "stop"])
    record = {"checked_at": stamp(), "unit": prepare_runtime.UNIT, "target": target,
              "attempted": True, "stopped": False}
    try:
        result = subprocess.run([str(access_gate.WRAPPER), "-F", str(private_config),
                                 alias, command], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        record.update(stopped=result.returncode == 0, exit_code=result.returncode)
    except (OSError, subprocess.TimeoutExpired) as error:
        record["error_type"] = type(error).__name__
    atomic(HERE / "proof/PREPARATION_CANCEL.json", record)
    return record


def wait_preparation(profile, identity, timeout=13200):
    try:
        if not STOP.exists():
            remember_preparation_owner(profile)
        return _wait_preparation(profile, identity, timeout)
    except InterruptedError as error:
        error.preparation_cleanup = cancel_preparation()
        raise


def remember_preparation_owner(profile):
    atomic(HERE / "proof/PREPARATION_OWNER.json", {
        "target": access_gate.TARGET, "root": profile["storage"]["root"],
    })


def _wait_preparation(profile, identity, timeout):
    deadline = time.monotonic() + timeout
    while True:
        boundary()
        progress = remote_preparation(profile)
        progress["checked_at"] = stamp()
        atomic(HERE / "proof/PREPARATION.json", progress)
        if progress.get("plan_identity") != identity:
            raise RuntimeError("Preparation belongs to a different source bundle; its work was preserved")
        record = progress.get("preparation", {})
        if not progress["active"]:
            installed = remote_installation(profile)
            if record.get("status") == "PASS" and installed.get("bundle_identity") == identity:
                atomic(HERE / "proof/INSTALLATION.json", installed)
                return installed
            raise RuntimeError("Supervised preparation did not finish successfully; inspect its retained private logs")
        if time.monotonic() >= deadline:
            return {"status": "PREPARING", "unit": prepare_runtime.UNIT,
                    "bundle_identity": identity, "stage": record.get("stage"),
                    "note": "The finite supervised job continues; repeat install to resume observing it."}
        until = min(deadline, time.monotonic() + 30)
        while time.monotonic() < until and not STOP.exists():
            time.sleep(min(1, until - time.monotonic()))


def launch_preparation(profile, identity, source_count):
    root = profile["storage"]["root"]
    image_inputs = {name: hashlib.sha256(source_member(HERE, name).read_bytes()).hexdigest()
                    for name in DEPLOYMENT_FILES}
    image_inputs["openclaw_archive"] = OPENCLAW_SHA
    image_inputs["cuda_architectures"] = profile["model"]["cuda_architectures"]
    image_identity = hashlib.sha256(json.dumps(image_inputs, sort_keys=True).encode()).hexdigest()
    plan = {"bundle_identity": identity, "image_identity": image_identity,
            "storage_mount": profile["storage"]["mount"], "source_files": source_count,
            "source_revision": profile["source"]["revision"],
            "model_manifest": profile["model"].get("manifest", "profiles/qwen3.8-27b-q8.json")}
    with tempfile.TemporaryDirectory(prefix="paai-prepare-") as directory:
        local = Path(directory)
        atomic(local / "install-plan.json", plan)
        sync(local, root + "/deployment", ["install-plan.json"])
    remember_preparation_owner(profile)
    remote(["sudo", "-n", "systemd-run", "--unit", prepare_runtime.UNIT, "--collect",
            "--uid", "ubuntu", "--property=Type=exec", "--property=RuntimeMaxSec=13200",
            "--property=TimeoutStopSec=30", "--property=KillMode=control-group",
            "--property=UMask=0077", f"--property=RequiresMountsFor={root}",
            "--description=Physical Agentic AI finite runtime preparation",
            "python3", CONTROL + "/prepare_runtime.py", "run", "--root", root],
           timeout=45, log=HERE / "logs/preparation-launch.log")
    return wait_preparation(profile, identity)


def runtime_image_matches(profile, receipt):
    recorded = receipt.get("image_id", "")
    if not isinstance(recorded, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", recorded):
        return False
    try:
        current = remote(["docker", "--host", DOCKER_SOCKET, "image", "inspect",
                          prepare_runtime.IMAGE, "--format", "{{.Id}}"])
    except (RuntimeError, TimeoutError):
        return False
    return current.strip() == recorded


def require_installed_bundle(profile, receipt):
    source, files = staged_bundle()
    if receipt.get("bundle_identity") != bundle_identity(source, files, profile):
        raise RuntimeError("The selected source or profile differs from installation; resume install")


def running_rows(profile):
    output = compose(profile, "ps", "--format", "json", log=HERE / "logs/status.log")
    rows = []
    for line in output.splitlines():
        if line.strip():
            value = json.loads(line)
            rows.extend(value if isinstance(value, list) else [value])
    return [{key: row.get(key) for key in ("Name", "Service", "State", "Health", "ExitCode")}
            for row in rows]


def install(profile, profile_path, access):
    source, files = staged_bundle()
    identity = bundle_identity(source, files, profile)
    previous = remote_installation(profile)
    preparation = remote_preparation(profile)
    if preparation["active"]:
        if preparation.get("plan_identity") != identity:
            raise RuntimeError("An existing preparation job owns a different bundle; no files were changed")
        outcome = wait_preparation(profile, identity)
        return outcome if outcome.get("status") == "PREPARING" else start(profile, profile_path, True)
    if (previous.get("status") == "PREPARED" and previous.get("bundle_identity") == identity
            and runtime_image_matches(profile, previous)):
        return start(profile, profile_path, True)
    if previous:
        raise ValueError("The installed bundle differs; preserve it and provision a separate target for this revision")
    archive = openclaw_archive()
    existing_rows = running_rows(profile) if previous else []
    running = any(row["Service"] == "demo" and row["State"] == "running" for row in existing_rows)
    admitted = preflight(profile_path, installed=bool(previous), running=running)
    root = profile["storage"]["root"]
    output = remote(["python3", CONTROL + "/host_setup.py", "--profile", CONTROL + "/profile.json"],
                    timeout=150, log=HERE / "logs/docker-setup.log")
    atomic(HERE / "proof/DOCKER_STORAGE.json", json.loads(output))
    remote(["install", "-d", "-m", "0700", root + "/source", root + "/deployment/container",
            root + "/models", root + "/data", root + "/data/logs/isaac", root + "/data/cache/kit"])
    sync(source, root + "/source", files, timeout=1800, log=HERE / "logs/source-transfer.log")
    sync(HERE, root + "/deployment", DEPLOYMENT_FILES, log=HERE / "logs/deployment-transfer.log")
    sync(archive.parent, root + "/deployment/container", [archive.name], timeout=600,
         log=HERE / "logs/openclaw-transfer.log")
    ip = next(value for value in access["peer"]["TailscaleIPs"] if ":" not in value)
    prepare_camera_video(profile, ip)
    with tempfile.TemporaryDirectory(prefix="paai-site-") as directory:
        local = Path(directory)
        (local / "site.env").write_text(f"PAAI_ROOT={root}\nPAAI_TAILNET_IP={ip}\n"
                                        f"PAAI_INSTANCE={profile['network']['tailscale_hostname']}\n"
                                        f"PAAI_UID={admitted['uid']}\nPAAI_GID={admitted['gid']}\n"
                                        f"PAAI_EXPECTED_GPU={profile['accelerator']['name']}\n"
                                        f"PAAI_MIN_VRAM_MIB={profile['accelerator']['min_vram_mib']}\n"
                                        f"PAAI_CUDA_ARCHITECTURES={profile['model']['cuda_architectures']}\n"
                                        f"PAAI_MODEL_FILENAME={profile['model']['model_filename']}\n")
        sync(local, root + "/deployment", ["site.env"])
    outcome = launch_preparation(profile, identity, len(files))
    return outcome if outcome.get("status") == "PREPARING" else start(profile, profile_path, True)


def public_visitor(profile, operation):
    settings = profile.get("public_visitor", {})
    if not isinstance(settings, dict) or type(settings.get("enabled", False)) is not bool:
        raise ValueError("public_visitor.enabled must be a boolean")
    if not settings.get("enabled", False):
        return {"enabled": False}
    root = profile["storage"]["root"]
    helper = root + "/tools/visitor/public.py"
    command = ["python3", helper, operation, "--root", root]
    if operation == "install":
        token_file = settings.get("token_file")
        if not isinstance(token_file, str) or not Path(token_file).is_absolute():
            raise ValueError("public_visitor.token_file must be an absolute path on Brev")
        for option in ("local_auth", "video"):
            if type(settings.get(option, False)) is not bool:
                raise ValueError(f"public_visitor.{option} must be a boolean")
        control = root + "/control/visitor"
        remote(["install", "-d", "-m", "0700", control])
        sync(HERE, control, ("public.py", *visitor_deployment.VISITOR_FILES), timeout=180)
        command[1] = control + "/public.py"
        command += ["--token-file", token_file]
        if settings.get("local_auth", False):
            command.append("--local-auth")
        if settings.get("video", False):
            command.append("--video")
    else:
        try:
            remote(["test", "-f", helper])
        except RuntimeError:
            if operation in ("status", "stop"):
                return {"enabled": True, "installed": False, "healthy": False, "url": None}
            raise RuntimeError("The public visitor is not installed; run install first") from None
    result = json.loads(remote(command, timeout=360 if operation == "install" else 120,
                               log=HERE / f"logs/public-{operation}.log"))
    return {"enabled": True, **result}


def validate_start(profile, profile_path):
    if remote_preparation(profile)["active"]:
        raise RuntimeError("Preparation is active; resume install before starting the demo")
    installed = remote_installation(profile)
    if installed.get("status") != "PREPARED" or not runtime_image_matches(profile, installed):
        raise RuntimeError("The source and runtime image are not prepared; resume install")
    require_installed_bundle(profile, installed)
    rows = running_rows(profile)
    running = any(row["Service"] == "demo" and row["State"] == "running" for row in rows)
    preflight(profile_path, installed=True, running=running)
    return rows, running


def start(profile, profile_path, install_public=False):
    rows, running = validate_start(profile, profile_path)
    required_services = {"demo", "camera-relay"} if camera_deployment.enabled(profile) else {"demo"}
    if (running and {row["Service"] for row in rows if row["State"] == "running"} == required_services
            and all(row.get("Health") == "healthy" for row in rows if row["Service"] == "demo")):
        public_visitor(profile, "install" if install_public else "start")
        return status(profile)
    try:
        compose(profile, "up", "-d", "--no-build", "--no-recreate", "--wait", "--wait-timeout", "1200",
                timeout=1300, log=HERE / "logs/start.log")
    except (RuntimeError, TimeoutError) as failure:
        cleanup = {"checked_at": stamp(), "logs_collected": False, "owned_project_stopped": False,
                   "errors": []}
        for args, timeout, filename, field in (
                (("logs", "--no-color", "--tail", "100"), 30, "startup-failure.log", "logs_collected"),
                (("stop", "--timeout", "90"), 150, "failed-start-stop.log", "owned_project_stopped")):
            try:
                if args[0] == "stop":
                    stop_container_ids(stop_ownership(profile)["containers"], HERE / "logs" / filename)
                else:
                    compose(profile, *args, timeout=timeout, log=HERE / "logs" / filename)
                cleanup[field] = True
            except Exception as error:
                cleanup["errors"].append({"operation": args[0], "error_type": type(error).__name__})
        failure.deployment_cleanup = cleanup
        raise
    public_visitor(profile, "install" if install_public else "start")
    return status(profile)


def preparation_unit(profile, operation="inspect"):
    return json.loads(remote(["python3", "-c", PREPARATION_UNIT_GUARD, profile["storage"]["root"],
                              CONTROL + "/prepare_runtime.py", prepare_runtime.UNIT, operation], timeout=60))


def stop_ownership(profile):
    root = Path(profile["storage"]["root"])
    if not root.is_absolute() or str(root) != str(root.resolve()):
        raise ValueError("Stop requires the canonical deployment root")
    preparation = preparation_unit(profile)
    docker = ["docker", "--host", DOCKER_SOCKET]
    if remote([*docker, "info", "--format", "{{.DockerRootDir}}" ]).strip() != str(root / "docker"):
        raise ValueError("Stop refused a Docker daemon owned by another deployment")
    enabled = camera_deployment.enabled(profile)
    configs = [str(root / "deployment/compose.yaml")]
    if enabled:
        configs.append(str(root / "deployment/compose.streaming.yaml"))
    mounts = {"demo": {"/workspace": root / "source", "/data": root / "data", "/data/models": root / "models"}}
    if enabled:
        mounts["camera-relay"] = {"/mediamtx.yml": root / "deployment/mediamtx.yml"}
    ids = remote([*docker, "ps", "--all", "--quiet", "--no-trunc", "--filter",
                  f"label=com.docker.compose.project={COMPOSE_PROJECT}"]).splitlines()
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in ids):
        raise ValueError("Stop could not establish container identities")
    for identity in ids:
        row = json.loads(remote([*docker, "inspect", "--type", "container", "--format", STOP_CONTAINER_FORMAT, identity]))
        labels = row.get("labels") or {}
        service = labels.get("com.docker.compose.service")
        if (row.get("id") != identity or labels.get("com.docker.compose.project") != COMPOSE_PROJECT
                or service not in mounts or labels.get("com.docker.compose.project.working_dir") != str(root / "deployment")
                or labels.get("com.docker.compose.project.config_files") != ",".join(configs)):
            raise ValueError("Stop refused a Compose container owned by another deployment")
        actual = {item.get("Destination"): item for item in row.get("mounts", [])}
        for destination, source in mounts[service].items():
            item = actual.get(destination, {})
            if item.get("Type") != "bind" or item.get("Source") != str(source):
                raise ValueError("Stop refused mismatched deployment mounts")
    return {"containers": ids, "preparation": preparation}


def stop(profile):
    ownership = stop_ownership(profile)
    visitor = public_visitor(profile, "stop")
    if ownership["preparation"]["active"]:
        preparation_unit(profile, "stop")
    stop_container_ids(ownership["containers"], HERE / "logs/stop.log")
    return {"status": "STOPPED", "public_visitor": visitor,
            "stopped_containers": ownership["containers"]}


def stop_container_ids(identities, log):
    if identities:
        remote(["docker", "--host", DOCKER_SOCKET, "stop", "--time", "90", *identities], timeout=150, log=log)


def restart(profile, profile_path):
    validate_start(profile, profile_path)
    stop(profile)
    return start(profile, profile_path)


def status(profile):
    rows = running_rows(profile)
    access = json.loads((HERE / "access/PASSED.json").read_text())
    ip = next(value for value in access["peer"]["TailscaleIPs"] if ":" not in value)
    origin = f"http://{ip}:8092"
    visitor = public_visitor(profile, "status")
    result = {"checked_at": stamp(), "containers": rows, "public_visitor": visitor,
              "preparation": remote_preparation(profile),
              "urls": {"learner": origin + "/guide/", "openclaw": origin + "/openclaw/",
                       "cameras": origin + "/cameras/"},
              "ready": False, "note": "Container health alone does not certify live visitor and physics gates."}
    if visitor.get("url"):
        result["urls"]["public_visitor"] = visitor["url"]
    atomic(HERE / "proof/CONTAINERS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", default="install",
                        choices=("prepare", "install", "access", "preflight", "start", "status", "restart", "stop"))
    parser.add_argument("--profile", type=Path, default=HERE / "profiles/brev-rtx6000.json")
    parser.add_argument("--source", type=Path, help="Complete reviewed distribution source directory")
    parser.add_argument("--manifest", type=Path, help="Portable distribution manifest")
    parser.add_argument("--openclaw", type=Path, help="Verified openclaw.tar.zst build input")
    parser.add_argument("--clip", type=Path, help="Verified CLIP source archive")
    args = parser.parse_args()
    if not args.profile.is_absolute() and not args.profile.exists():
        args.profile = HERE / args.profile
    operation_lock = None
    try:
        boundary()
        if args.operation in ("prepare", "install", "preflight", "start", "restart", "stop"):
            descriptor = os.open(HERE / ".operation.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            operation_lock = os.fdopen(descriptor, "w")
            fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profile = json.loads(args.profile.read_text())
        if profile.get("name") != "brev-rtx6000":
            raise ValueError("Only the Brev RTX PRO 6000 runtime is implemented; Spark remains a plan")
        preparation_args = (args.source, args.manifest, args.openclaw, args.clip)
        if args.operation == "prepare":
            if not all(preparation_args):
                raise ValueError("prepare requires --source, --manifest, --openclaw and --clip")
            result = prepare_bundle.prepare(HERE, profile, *preparation_args, boundary=boundary)
            print(json.dumps(result, indent=2))
            return 0
        if any(preparation_args):
            raise ValueError("Distribution input options belong to the prepare operation")
        access = require_access(profile)
        if args.operation == "access":
            result = {"status": "PASS", "evidence": access["evidence"]}
        elif args.operation == "preflight":
            result = preflight(args.profile)
        elif args.operation == "install":
            result = install(profile, args.profile, access)
        elif args.operation == "start":
            result = start(profile, args.profile)
        elif args.operation == "status":
            result = status(profile)
        elif args.operation == "restart":
            result = restart(profile, args.profile)
        else:
            result = stop(profile)
        print(json.dumps(result, indent=2))
        return 0
    except ConnectionError as error:
        print(json.dumps({"status": "WAITING_FOR_ACCESS", "reason": str(error)}))
        return 75
    except Exception as error:
        result = {"status": "STOPPED" if isinstance(error, InterruptedError) else "FAILED",
                  "checked_at": stamp(), "operation": args.operation, "reason": str(error)}
        if hasattr(error, "deployment_cleanup"):
            result["cleanup"] = error.deployment_cleanup
        if isinstance(error, InterruptedError):
            if args.operation == "prepare":
                result["preparation_cleanup"] = {"attempted": False, "reason": "Offline input preparation started no remote work"}
            else:
                result["preparation_cleanup"] = (error.preparation_cleanup if hasattr(error, "preparation_cleanup")
                                                 else cancel_preparation())
        atomic(HERE / "proof/LAST_OPERATION.json", result)
        print(json.dumps(result, indent=2))
        return 0 if isinstance(error, InterruptedError) else 1
    finally:
        if operation_lock is not None:
            operation_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
