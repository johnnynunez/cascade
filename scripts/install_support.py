#!/usr/bin/env python3
"""Stdlib installation checks and process ownership; not GPU certification.

Package/model installs remain in the shell entry points. This helper never
imports torch/Isaac in the installer process or changes third-party sources.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

from fetch_robot_assets import fetch
from kitchen_assets import kitchen_problems, prepare_kitchen

# Release asset byte counts from github.com/ultralytics/assets v8.3.0.
# The two YOLOE weights are already tracked; only the TS encoder is not.
MODEL_ASSETS = {
    "yoloe-11s-seg.pt": 27803986,
    "yoloe-11s-seg-pf.pt": 27948751,
    "mobileclip_blt.ts": 599764649,
}
MODEL_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
ROBOT_ASSET_DIRS = ("assets/usd/RS-rebot-dev-arm", "assets/urdf/00-arm-rs_asm-v3")
ROBOT_LFS_INCLUDE = ",".join(f"{directory}/**" for directory in ROBOT_ASSET_DIRS)


def model_asset_valid(path: Path, size: int) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def model_lfs_pointer(path: Path, size: int) -> bool:
    """Recognize a canonical placeholder for this exact expected payload size."""
    try:
        if not path.is_file() or path.stat().st_size > 1024:
            return False
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return False
    return (
        len(lines) == 3
        and lines[0] == "version https://git-lfs.github.com/spec/v1"
        and lines[1].startswith("oid sha256:")
        and len(lines[1][11:]) == 64
        and all(char in "0123456789abcdef" for char in lines[1][11:])
        and lines[2] == f"size {size}"
    )


def ensure_model_asset(url: str, dest: Path, size: int) -> None:
    if dest.exists():
        if model_asset_valid(dest, size):
            return
        if not model_lfs_pointer(dest, size):
            raise RuntimeError(
                f"invalid model asset {dest}; preserve/move it aside explicitly, then retry"
            )
        # A clone without LFS leaves tracked placeholders. The existing pinned
        # release download can hydrate these; arbitrary corrupt files stay put.
    staged = dest.with_suffix(dest.suffix + ".download")
    try:
        # Reuse the existing byte-count checked, retrying downloader. Validate
        # the archive and the release's expected size BEFORE publishing it.
        for attempt in range(3):
            try:
                fetch(url, staged, force=True)
                break
            except SystemExit as exc:
                cause = exc.__cause__
                if (
                    not isinstance(cause, urllib.error.HTTPError)
                    or (cause.code != 429 and cause.code < 500)
                    or attempt == 2
                ):
                    raise
                time.sleep(attempt + 1)
        if not model_asset_valid(staged, size):
            raise RuntimeError(f"incomplete/invalid release asset: {url}")
        os.replace(staged, dest)
    finally:
        staged.unlink(missing_ok=True)


def scene_problems(repo: Path) -> list[str]:
    """Check only the shipped reBot asset, without loading USD/Kit."""
    paths = [
        repo / "assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda",
        repo / "assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf",
    ]
    tracked = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "-z",
                "--",
                *ROBOT_ASSET_DIRS,
            ],
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .split("\0")
    )
    paths.extend(repo / name for name in tracked if name)
    problems = []
    for path in dict.fromkeys(paths):
        if not path.is_file() or not path.stat().st_size:
            problems.append(f"missing scene asset: {path}")
        else:
            with path.open("rb") as stream:
                if stream.read(80).startswith(
                    b"version https://git-lfs.github.com/spec/"
                ):
                    problems.append(f"unresolved Git LFS pointer: {path}")
    return problems


def prepare_robot_assets(repo: Path) -> None:
    """Hydrate only unresolved shipped robot payloads, without changing Git config."""
    problems = scene_problems(repo)
    if any(problem.startswith("unresolved Git LFS pointer:") for problem in problems):
        if shutil.which("git-lfs") is None:
            raise RuntimeError(
                "The shipped robot assets need Git LFS. Install git-lfs, verify "
                "`git lfs version`, then rerun the Spark installer; it will fetch "
                "only the required robot USD/URDF subtrees."
            )
        result = subprocess.run(
            ["git", "-C", str(repo), "lfs", "pull", f"--include={ROBOT_LFS_INCLUDE}", "--exclude="],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "Robot Git LFS download failed; rerun the Spark installer after "
                f"repairing access: {result.stderr.strip() or result.stdout.strip()}"
            )
        problems = scene_problems(repo)
    if problems:
        raise RuntimeError("\n".join(problems))


def prepare_assets(repo: Path) -> None:
    prepare_robot_assets(repo)
    for name, size in MODEL_ASSETS.items():
        ensure_model_asset(f"{MODEL_RELEASE}/{name}", repo / "models" / name, size)
    prepare_kitchen(repo)
    print(
        "[cascade-install] reBot, perception and Build a Claw kitchen assets validated."
    )


QWEN_PORT = 8080
EULA_URL = "https://docs.omniverse.nvidia.com/eula"
MODEL_ENV_KEYS = ("CASCADE_QWEN_MODEL", "CASCADE_QWEN_MMPROJ", "LLAMA_SERVER", "LLAMA_DIR")


def model_environment(repo: Path) -> dict[str, str]:
    """Retain only local model/server paths, with explicit environment overrides."""
    try:
        record = json.loads((repo / "runs/.install/install.json").read_text())
        saved = record.get("model_environment", {}) if record.get("repo") == str(repo.resolve()) else {}
        if not isinstance(saved, dict):
            saved = {}
    except (OSError, ValueError, AttributeError):
        saved = {}
    result = {}
    for key in MODEL_ENV_KEYS:
        value = os.environ.get(key, saved.get(key))
        if isinstance(value, str):
            result[key] = str(Path(value).expanduser().resolve()) if value else ""
    return result


def eula_accepted(repo: Path) -> bool:
    """Only a checkout-bound explicit-consent receipt grants future launches."""
    try:
        record = json.loads((repo / "runs/.install/install.json").read_text())
        return (record.get("repo") == str(repo.resolve())
                and record.get("eula_accepted") is True
                and record.get("eula_url") == EULA_URL)
    except (OSError, ValueError, AttributeError):
        return False


def isaac_environment(repo: Path) -> dict[str, str]:
    """Match launcher/installer selection, without importing or modifying Kit."""
    selected = os.environ.get("ISAACSIM_PYTHON_EXE")
    source = os.environ.get("ISAACSIM_PATH")
    if not selected and source:
        selected = str(Path(source).expanduser() / "python.sh")
    if not selected:
        selected = str(repo / ".isaacsim/bin/python")
    python = Path(selected).expanduser().absolute()
    env = {"ISAACSIM_PYTHON_EXE": str(python)}
    if python.name == "python.sh":
        env["ISAACSIM_PATH"] = str(python.parent)
    return env


def model_health() -> bool:
    # Loopback must not inherit a workstation/system proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            f"http://127.0.0.1:{QWEN_PORT}/v1/models", timeout=2
        ) as response:
            data = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    models = (
        {row.get("id") for row in data.get("data", []) if isinstance(row, dict)}
        if isinstance(data, dict)
        else set()
    )
    if "Qwen/Qwen3.8-27B" not in models:
        raise RuntimeError(
            f"wrong model on :{QWEN_PORT}: {sorted(models, key=str)}; expected Qwen/Qwen3.8-27B"
        )
    return True


def qwen_binding(repo: Path, env: dict[str, str]) -> dict:
    """Bind a supervised model process to its selected files and launch script."""
    directory = repo / "models/qwen3.8-27b"
    server_directory = Path(env.get("LLAMA_DIR") or repo / ".llama.cpp")
    paths = {
        "model": Path(env.get("CASCADE_QWEN_MODEL") or directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
        "projector": Path(env.get("CASCADE_QWEN_MMPROJ") or directory / "mmproj-BF16.gguf"),
        "server": Path(env.get("LLAMA_SERVER") or server_directory / "build/bin/llama-server"),
    }
    identities = {}
    for role, path in paths.items():
        try:
            info = path.stat()
        except OSError as exc:
            raise RuntimeError(f"Qwen {role} is missing at {path}; rerun the Spark installer") from exc
        identities[role] = {"path": str(path.resolve()), "size_bytes": info.st_size,
                            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
                            "device": info.st_dev, "inode": info.st_ino}
    return {"files": identities, "port": QWEN_PORT,
            "launcher_sha256": hashlib.sha256((repo / "scripts/serve_qwen_llamacpp.sh").read_bytes()).hexdigest()}


def _group_is_stopped(pgid: int) -> bool:
    """An exited leader alone cannot prove that its private workers stopped."""
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True,
            timeout=5, env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        return False
    if result.returncode or result.stderr or not result.stdout.endswith("\n"):
        return False
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3 or not all(value.isdecimal() for value in fields[:2]):
            return False
        if int(fields[1]) == pgid and not re.fullmatch(r"Z[+<>AELNPSVWXsl]*", fields[2]):
            return False
    return True


def _signal_owned_group(pgid: int, number: int) -> None:
    try:
        os.killpg(pgid, number)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        # Some kernels deny signals to zombie-only groups. Preserve a real
        # denial unless a complete process snapshot proves no live members.
        if error.errno != errno.EPERM or not _group_is_stopped(pgid):
            raise


def stop_group(process: subprocess.Popen) -> None:
    # This group was created by THIS invocation (never signal a stale PID file).
    _signal_owned_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # A leader may exit on TERM while a worker ignores it. Reap the entire
    # private group, not just Popen's direct child (vLLM has worker processes).
    _signal_owned_group(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def launch(repo: Path, profile: str, brain: str, *, no_open: bool = False, headless: bool = False) -> int:
    from cascade.apps.process_owner import (
        live_records, load_owner, profile_state_dir, register_process,
    )

    wait_seconds = float(os.environ.get("CASCADE_MODEL_WAIT_S", "1800"))
    if not math.isfinite(wait_seconds) or wait_seconds <= 0:
        raise ValueError("CASCADE_MODEL_WAIT_S must be positive and finite")
    env = os.environ.copy()
    env["PY"] = str(repo / ".venv/bin/python")
    if profile == "spark":
        if brain != "qwen":
            raise RuntimeError("Spark delivery requires the local qwen brain; use laptop explicitly for other modes")
        if not eula_accepted(repo):
            raise RuntimeError("Spark launch requires explicit consent: run install.sh --accept-eula first")
        env.pop("ISAACSIM_PATH", None)
        env.update(isaac_environment(repo))
        env.update(model_environment(repo))
        env["CASCADE_INSTALL_PROFILE"] = "spark"
        env["CASCADE_OPENCLAW_PROFILE"] = "cascade-demo"
        env["CASCADE_QWEN_BASE_URL"] = f"http://127.0.0.1:{QWEN_PORT}/v1"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"  # validated persistent consent above
        # install.sh bootstraps uv here without editing shell profiles. Desktop
        # and remote sessions may not inherit that directory in PATH.
        env["PATH"] = os.pathsep.join((
            str(repo / ".openclaw-cli/bin"), str(Path.home() / ".local/bin"),
            env.get("PATH", os.defpath),
        ))
    root = Path(env.get("CASCADE_LAUNCH_STATE", repo / "runs/.launch")).expanduser()
    if not root.is_absolute():
        root = repo / root
    root = root.resolve()
    # Both processes receive the same absolute root, even when invoked from
    # different working directories. Only the shared helper scopes the profile.
    env["CASCADE_LAUNCH_STATE"] = str(root)
    oc_profile = env.get("CASCADE_OPENCLAW_PROFILE", "")
    state = profile_state_dir(root, oc_profile)
    owner = load_owner(state, repo, oc_profile, create=True)
    assert owner is not None
    # Qwen starts before launch.sh. Invalidate here too, so a failed model
    # startup cannot leave the previous attendee's READY receipt current.
    attempt = "startup-attempt-" + uuid.uuid4().hex
    evidence = state / attempt
    evidence.mkdir(mode=0o700)
    proof = state / "proof.json"
    if proof.is_file():
        (evidence / "previous-proof.json").write_bytes(proof.read_bytes())
    temporary = evidence / "initial-proof.json"
    temporary.write_text(json.dumps({
        "verified": False, "sim": "isaac" if profile == "spark" else "mujoco",
        "profile": oc_profile, "started_at": time.time(), "attempt": attempt,
        "note": "supervised startup in progress; no current proof",
    }) + "\n")
    temporary.replace(proof)
    qwen = None
    launcher = None
    success = False
    try:
        if profile == "spark":
            problems = kitchen_problems(repo)
            if problems:
                raise RuntimeError("\n".join(problems))
        healthy_model = brain == "qwen" and model_health()
        if healthy_model and profile == "spark":
            owned = live_records(state, owner, role="qwen")
            if len(owned) != 1:
                raise RuntimeError(
                    f"Qwen port :{QWEN_PORT} is served outside this installation's ownership; "
                    "the existing process was preserved. Stop its own deployment before starting this one."
                )
            if owned[0].get("model_binding") != qwen_binding(repo, env):
                raise RuntimeError(
                    f"Owned Qwen on :{QWEN_PORT} uses different or changed model/server files; "
                    "the existing process was preserved. Run this installation's stop command, then start again."
                )
        if brain == "qwen" and not healthy_model:
            try:
                with socket.create_connection(("127.0.0.1", QWEN_PORT), timeout=0.5):
                    raise RuntimeError(
                        f"port :{QWEN_PORT} is occupied but Qwen is not healthy; refusing to start a duplicate"
                    )
            except (ConnectionRefusedError, TimeoutError, OSError):
                pass
            if live_records(state, owner, role="qwen"):
                raise RuntimeError(
                    f"recorded Qwen process is still starting/unhealthy; inspect {state / 'qwen.log'}"
                )
            service_env = dict(
                env,
                MODEL_ROOT=str(repo),
                PORT=str(QWEN_PORT),
                CTX="32768",
            )
            binding = qwen_binding(repo, env)
            with (state / "qwen.log").open("ab") as log:
                qwen = subprocess.Popen(
                    [str(repo / "scripts/serve_qwen_llamacpp.sh")],
                    cwd=repo,
                    env=service_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + wait_seconds
            while True:
                if qwen.poll() is not None:
                    raise RuntimeError(
                        f"Qwen exited with status {qwen.returncode}; see {state / 'qwen.log'}"
                    )
                if model_health():
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Qwen health timeout; see {state / 'qwen.log'}"
                    )
                time.sleep(0.25)
            # Register only this invocation's child and the files it was given.
            if binding != qwen_binding(repo, env):
                raise RuntimeError("Qwen model/server files changed during startup; retry after restoring this installation")
            receipt = register_process(state, owner, qwen.pid, "qwen")
            receipt["model_binding"] = binding
            temporary = state / f"qwen.{qwen.pid}.tmp"
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(state / "qwen.pid")
        command = [
            "bash",
            str(repo / "scripts/launch.sh"),
            "--sim",
            "isaac" if profile == "spark" else "mujoco",
            "--brain",
            brain,
        ]
        if no_open:
            command.append("--no-open")
        if headless:
            command.append("--headless")
        launcher = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True)
        while launcher.poll() is None:
            if qwen is not None and qwen.poll() is not None:
                raise RuntimeError(
                    f"Qwen exited with status {qwen.returncode} during launcher proof; see {state / 'qwen.log'}"
                )
            time.sleep(0.1)
        success = launcher.returncode == 0
        return launcher.returncode
    finally:
        if launcher is not None and (not success or launcher.poll() is None):
            # A failed leader may already have exited while its sidecars are
            # still running in this invocation's private process group.
            stop_group(launcher)
        if not success and qwen is not None:
            stop_group(qwen)
            # Like stop_owned(), retain the inert receipt for diagnosis.
            # Liveness is checked by kernel identity, not file existence.
            # Unlinking here could erase a newer invocation's receipt.


def package_inventory(python: Path) -> dict:
    """Read actual package metadata without importing the application or Kit."""
    if not os.access(python, os.X_OK):
        return {"available": False, "error": "interpreter is missing"}
    code = """import importlib.metadata as m, json, platform, sys
print(json.dumps({"available": True, "python": platform.python_version(),
                  "executable": sys.executable, "architecture": platform.machine(),
                  "packages": {d.metadata["Name"]: d.version for d in m.distributions() if d.metadata["Name"]}}))
"""
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX")}
    try:
        result = subprocess.run([str(python), "-B", "-c", code], capture_output=True,
                                text=True, check=True, timeout=30, env=env)
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": type(exc).__name__}


def record_install(repo: Path, profile: str, brain: str, ref: str, *, accept_eula: bool = False) -> None:
    if profile == "spark" and not (accept_eula or eula_accepted(repo)):
        raise RuntimeError("cannot record Spark installation without explicit --accept-eula consent")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            capture_output=True,
            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
        ).stdout.strip()

    dirty = git(
        "status",
        "--porcelain",
        "--",
        ".",
        ":!.venv",
        ":!.isaacsim",
        ":!.cosmos",
        ":!.llama.cpp",
        ":!models/qwen3.8-27b",
        ":!.openclaw-cli",
        ":!runs",
        ":!models/Cosmos3-Edge-hf",
    )
    isaac = isaac_environment(repo) if profile == "spark" else {}
    model = model_environment(repo) if profile == "spark" else {}
    app_inventory = package_inventory(repo / ".venv/bin/python")
    isaac_inventory = package_inventory(Path(isaac["ISAACSIM_PYTHON_EXE"])) if isaac else {}
    def read_json(path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None
    openclaw_package = next((value for path in (
        repo / ".openclaw-cli/tools/node/lib/node_modules/openclaw/package.json",
        repo / ".openclaw-cli/lib/node_modules/openclaw/package.json",
    ) if isinstance(value := read_json(path), dict)), {})
    record = {
        "repo": str(repo.resolve()),
        "eula_accepted": profile == "spark" and (accept_eula or eula_accepted(repo)),
        "eula_url": EULA_URL if profile == "spark" else None,
        "isaac_environment": isaac,
        "model_environment": model,
        "source_commit": git("rev-parse", "HEAD"),
        "requested_ref": ref,
        "source_dirty": bool(dirty),
        "profile": profile,
        "brain": brain,
        "isaac_package": isaac_inventory.get("packages", {}).get("isaacsim"),
        "openclaw": openclaw_package.get("version"),
        "python": app_inventory.get("python"),
        "dependencies": {"app": app_inventory, "isaac": isaac_inventory},
        "model_receipts": [dict(path=str(path), receipt=read_json(path))
                           for path in sorted((repo / "models/qwen3.8-27b").glob("*.verified.json"))],
        "llama_runtime": read_json(repo / ".llama.cpp/build/.cascade-runtime.json"),
        "gpu_validated": False,
    }
    state = repo / "runs/.install"
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / f"install.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(state / "install.json")
    exports = {"PY": str(repo / ".venv/bin/python"), "CASCADE_INSTALL_PROFILE": profile}
    if profile == "spark":
        exports.update(isaac)
        exports.update(model)
        exports.update(
            CASCADE_OPENCLAW_PROFILE="cascade-demo",
            OMNI_KIT_ACCEPT_EULA="YES",
        )
    text = "# Generated after explicit installation consent. No credentials stored.\n"
    text += "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in exports.items()
    )
    bins = os.pathsep.join((str(repo / ".openclaw-cli/bin"), str(repo / ".venv/bin"), str(Path.home() / ".local/bin")))
    text += f'\nexport PATH={shlex.quote(bins)}:"$PATH"\n'
    temporary.write_text(text)
    temporary.replace(state / "env.sh")
    print(
        f"[cascade-install] Source commit={record['source_commit']} dirty={record['source_dirty']} requested_ref={ref}"
    )
    print(
        f"[cascade-install] Reusable environment: source {shlex.quote(str(state / 'env.sh'))}"
    )


def installation_problems(repo: Path, profile: str, brain: str) -> list[str]:
    """Distribution metadata only; imports can create caches or accept licenses."""
    expected: dict[str, str | None] = {
        "cascade": None,
        "numpy": None,
        "opencv-python-headless": None,
        "pyyaml": None,
        "pin": None,
    }
    if profile != "ci":
        expected.update(
            {
                name: None
                for name in ("pyzmq", "msgpack-numpy", "warp-lang", "scipy", "openai")
            }
        )
    if profile == "laptop":
        expected["mujoco"] = None
    if profile == "spark":
        expected.update(
            ultralytics=None,
            clip=None,
            torch="2.14.0+cu130",
            torchvision="0.29.0+cu130",
        )
    code = """import importlib.metadata as m, json, sys
errors=[]
if sys.version_info[:2] != (3,12): errors.append("Python 3.12 required")
for name, expected in json.loads(sys.argv[1]).items():
    try:
        actual=m.version(name)
        if expected and expected!=actual: errors.append(f"{name}: expected {expected}, found {actual}")
    except m.PackageNotFoundError: errors.append(f"distribution {name}")
print("\\n".join(errors))
sys.exit(bool(errors))
"""
    problems = []
    python = repo / ".venv/bin/python"
    if not python.is_file():
        problems.append(str(python))
    else:
        result = subprocess.run(
            [str(python), "-I", "-B", "-c", code, json.dumps(expected)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            problems.append(
                f"{python}: {result.stdout.strip() or result.stderr.strip()}"
            )
    if profile != "ci":
        packages = [
            repo / ".openclaw-cli/tools/node/lib/node_modules/openclaw/package.json",
            repo / ".openclaw-cli/lib/node_modules/openclaw/package.json",
        ]
        package = next((p for p in packages if p.is_file()), packages[0])
        try:
            if json.loads(package.read_text()).get("version") != "2026.9.3":
                problems.append("OpenClaw package version !=2026.9.3")
        except (OSError, ValueError):
            problems.append(f"OpenClaw 2026.9.3 package metadata: {package}")
    if profile == "spark":
        problems.extend(scene_problems(repo))
        problems.extend(kitchen_problems(repo))
        for name, size in MODEL_ASSETS.items():
            if not model_asset_valid(repo / "models" / name, size):
                problems.append(f"detector/encoder asset {repo / 'models' / name}")
        result = subprocess.run(
            [
                "bash",
                str(repo / "scripts/install_isaac.sh"),
                "--dir",
                str(repo),
                "--check",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            problems.append(result.stdout.strip() + " " + result.stderr.strip())
        if brain == "qwen":
            result = subprocess.run(
                [str(repo / "scripts/serve_qwen_llamacpp.sh"), "--check"],
                env=dict(
                    os.environ,
                    MODEL_ROOT=str(repo),
                    CASCADE_INSTALL_PROFILE=profile,
                    PY=sys.executable,
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                capture_output=True,
                text=True,
            )
            if result.returncode:
                problems.append(
                    "Qwen preflight: "
                    + result.stdout.strip()
                    + " "
                    + result.stderr.strip()
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["robot-assets", "assets", "record", "launch", "check"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--profile", choices=["spark", "laptop", "ci"], default="spark")
    parser.add_argument("--brain", choices=["qwen", "keep"], default="qwen")
    parser.add_argument("--ref", default="existing checkout")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--headless", action="store_true", help="launch Isaac without an editor window")
    parser.add_argument("--accept-eula", action="store_true", help="record explicit consent, never inferred from environment")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    if args.action == "launch":
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
    try:
        if args.action == "robot-assets":
            prepare_robot_assets(repo)
        elif args.action == "assets":
            prepare_assets(repo)
        elif args.action == "record":
            record_install(repo, args.profile, args.brain, args.ref, accept_eula=args.accept_eula)
        elif args.action == "check":
            problems = installation_problems(repo, args.profile, args.brain)
            for problem in problems:
                print(f"[cascade-install] MISSING: {problem}")
            return 3 if problems else 0
        else:
            return launch(repo, args.profile, args.brain, no_open=args.no_open, headless=args.headless)
    except KeyboardInterrupt:
        print(
            "[cascade-install] interrupted; newly started services stopped",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[cascade-install] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
