#!/usr/bin/env python3
"""Stdlib installation checks and process ownership; not GPU certification.

Package/model installs remain in the shell entry points. This helper never
imports torch/Isaac in the installer process or changes third-party sources.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

from fetch_robot_assets import fetch

# Release asset byte counts from github.com/ultralytics/assets v8.3.0.
# The two YOLOE weights are already tracked; only the TS encoder is not.
MODEL_ASSETS = {
    "yoloe-11s-seg.pt": 27803986,
    "yoloe-11s-seg-pf.pt": 27948751,
    "mobileclip_blt.ts": 599764649,
}
MODEL_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def model_asset_valid(path: Path, size: int) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def ensure_model_asset(url: str, dest: Path, size: int) -> None:
    if dest.exists():
        if not model_asset_valid(dest, size):
            raise RuntimeError(
                f"invalid model asset {dest}; preserve/move it aside explicitly, then retry"
            )
        return
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
                "assets/usd/RS-rebot-dev-arm",
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


def prepare_assets(repo: Path) -> None:
    problems = scene_problems(repo)
    if problems:
        raise RuntimeError("\n".join(problems))
    for name, size in MODEL_ASSETS.items():
        ensure_model_asset(f"{MODEL_RELEASE}/{name}", repo / "models" / name, size)
    print(
        "[cascade-install] Shipped reBot assets and YOLOE/TS encoder validated; no unrelated robot assets fetched."
    )


COSMOS_PORT = 8082


def model_health() -> bool:
    # Loopback must not inherit a workstation/system proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            f"http://127.0.0.1:{COSMOS_PORT}/v1/models", timeout=2
        ) as response:
            data = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    models = (
        {row.get("id") for row in data.get("data", []) if isinstance(row, dict)}
        if isinstance(data, dict)
        else set()
    )
    if "cosmos3-edge" not in models:
        raise RuntimeError(
            f"wrong model on :{COSMOS_PORT}: {sorted(models, key=str)}; expected cosmos3-edge"
        )
    return True


def stop_group(process: subprocess.Popen) -> None:
    # This group was created by THIS invocation (never signal a stale PID file).
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # A leader may exit on TERM while a worker ignores it. Reap the entire
    # private group, not just Popen's direct child (vLLM has worker processes).
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def launch(repo: Path, profile: str, brain: str, *, no_open: bool = False) -> int:
    from cascade.apps.process_owner import (
        live_records, load_owner, profile_state_dir, register_process,
    )

    wait_seconds = float(os.environ.get("CASCADE_COSMOS_WAIT_S", "1800"))
    if not math.isfinite(wait_seconds) or wait_seconds <= 0:
        raise ValueError("CASCADE_COSMOS_WAIT_S must be positive and finite")
    env = os.environ.copy()
    env["PY"] = str(repo / ".venv/bin/python")
    if profile == "spark":
        env["ISAACSIM_PYTHON_EXE"] = str(repo / ".isaacsim/bin/python")
        env["CASCADE_OPENCLAW_PROFILE"] = "cascade-demo"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"  # caller already checked explicit consent
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
    cosmos = None
    launcher = None
    success = False
    try:
        if brain == "cosmos" and not model_health():
            try:
                with socket.create_connection(("127.0.0.1", COSMOS_PORT), timeout=0.5):
                    raise RuntimeError(
                        f"port :{COSMOS_PORT} is occupied but Cosmos is not healthy; refusing to start a duplicate"
                    )
            except (ConnectionRefusedError, TimeoutError, OSError):
                pass
            if live_records(state, owner, role="cosmos"):
                raise RuntimeError(
                    f"recorded Cosmos process is still starting/unhealthy; inspect {state / 'cosmos.log'}"
                )
            service_env = dict(
                env,
                VENV=str(repo / ".cosmos"),
                MODEL_DIR=str(repo / "models"),
                PORT=str(COSMOS_PORT),
                GPU_FRAC="0.20",
                CTX="32768",
                COSMOS_PYTHON=env["PY"],
            )
            with (state / "cosmos.log").open("ab") as log:
                cosmos = subprocess.Popen(
                    [str(repo / "scripts/serve_cosmos_vllm.sh")],
                    cwd=repo,
                    env=service_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + wait_seconds
            while True:
                if cosmos.poll() is not None:
                    raise RuntimeError(
                        f"Cosmos exited with status {cosmos.returncode}; see {state / 'cosmos.log'}"
                    )
                if model_health():
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Cosmos health timeout; see {state / 'cosmos.log'}"
                    )
                time.sleep(0.25)
            # Register only the child created here, after its final exec and
            # semantic health. Never adopt a borrowed healthy endpoint.
            register_process(state, owner, cosmos.pid, "cosmos")
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
        launcher = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True)
        while launcher.poll() is None:
            if cosmos is not None and cosmos.poll() is not None:
                raise RuntimeError(
                    f"Cosmos exited with status {cosmos.returncode} during launcher proof; see {state / 'cosmos.log'}"
                )
            time.sleep(0.1)
        success = launcher.returncode == 0
        return launcher.returncode
    finally:
        if launcher is not None and (not success or launcher.poll() is None):
            # A failed leader may already have exited while its sidecars are
            # still running in this invocation's private process group.
            stop_group(launcher)
        if not success and cosmos is not None:
            stop_group(cosmos)
            # Like stop_owned(), retain the inert receipt for diagnosis.
            # Liveness is checked by kernel identity, not file existence.
            # Unlinking here could erase a newer invocation's receipt.


def record_install(repo: Path, profile: str, brain: str, ref: str) -> None:
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
        ":!.openclaw-cli",
        ":!runs",
        ":!models/Cosmos3-Edge-hf",
    )
    record = {
        "source_commit": git("rev-parse", "HEAD"),
        "requested_ref": ref,
        "source_dirty": bool(dirty),
        "profile": profile,
        "brain": brain,
        "isaac_package": "6.1.0.0" if profile == "spark" else None,
        "openclaw": "2026.9.3" if profile != "ci" else None,
        "python": "3.12",
        "gpu_validated": False,
    }
    state = repo / "runs/.install"
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / f"install.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(state / "install.json")
    exports = {"PY": str(repo / ".venv/bin/python")}
    if profile == "spark":
        exports.update(
            ISAACSIM_PYTHON_EXE=str(repo / ".isaacsim/bin/python"),
            CASCADE_OPENCLAW_PROFILE="cascade-demo",
            OMNI_KIT_ACCEPT_EULA="YES",
        )
    text = "# Generated after explicit installation consent. No credentials stored.\n"
    text += "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in exports.items()
    )
    text += f'\nexport PATH={shlex.quote(str(repo / ".openclaw-cli/bin"))}:"$PATH"\n'
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
        package = (
            repo / ".openclaw-cli/tools/node/lib/node_modules/openclaw/package.json"
        )
        try:
            if json.loads(package.read_text()).get("version") != "2026.9.3":
                problems.append("OpenClaw package version !=2026.9.3")
        except (OSError, ValueError):
            problems.append(f"OpenClaw 2026.9.3 package metadata: {package}")
    if profile == "spark":
        problems.extend(scene_problems(repo))
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
        if brain == "cosmos":
            result = subprocess.run(
                [str(repo / "scripts/serve_cosmos_vllm.sh"), "--check"],
                env=dict(
                    os.environ,
                    VENV=str(repo / ".cosmos"),
                    MODEL_DIR=str(repo / "models"),
                    COSMOS_PYTHON=sys.executable,
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                capture_output=True,
                text=True,
            )
            if result.returncode:
                problems.append(
                    "Cosmos preflight: "
                    + result.stdout.strip()
                    + " "
                    + result.stderr.strip()
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["assets", "record", "launch", "check"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--profile", choices=["spark", "laptop", "ci"], default="spark")
    parser.add_argument("--brain", choices=["cosmos", "keep"], default="cosmos")
    parser.add_argument("--ref", default="existing checkout")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    if args.action == "launch":
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
    try:
        if args.action == "assets":
            prepare_assets(repo)
        elif args.action == "record":
            record_install(repo, args.profile, args.brain, args.ref)
        elif args.action == "check":
            problems = installation_problems(repo, args.profile, args.brain)
            for problem in problems:
                print(f"[cascade-install] MISSING: {problem}")
            return 3 if problems else 0
        else:
            return launch(repo, args.profile, args.brain, no_open=args.no_open)
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
