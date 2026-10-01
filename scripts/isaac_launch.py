#!/usr/bin/env python3
"""Start Isaac in an isolated environment with owned shutdown.

A managed wheel's Python replaces this adapter in the launcher's group.
A source release's python.sh spawns (does not exec) Kit Python. Owning only
that shell PID leaks Kit on shutdown. This stable adapter owns its private
child group; it never signals a reused or externally started simulator.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid


class StartupExited(RuntimeError):
    """The original launcher exited before publishing its interpreter marker."""


def kernel_identity(pid: int) -> dict:
    """Read birth independently of cmdline, which can be empty during exec."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Isaac interpreter readiness requires Linux /proc process identities")
    proc = Path("/proc") / str(pid)
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError("Isaac launcher belongs to a different user")
    return {"pid": pid, "birth": Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + fields[19],
            "state": fields[0]}


def publish_ready() -> None:
    """Called by the final bridge interpreter, before it imports Isaac."""
    value = os.environ.get("CASCADE_ISAAC_STARTUP_READY")
    if not value:
        return
    path = Path(value)
    launcher = kernel_identity(int(os.environ["CASCADE_ISAAC_STARTUP_PID"]))
    if launcher["birth"] != os.environ["CASCADE_ISAAC_STARTUP_BIRTH"]:
        raise RuntimeError("Isaac launcher identity changed before bridge startup")
    bridge = kernel_identity(os.getpid())
    record = {"launcher_pid": launcher["pid"], "launcher_birth": launcher["birth"],
              "bridge_pid": bridge["pid"], "bridge_birth": bridge["birth"]}
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        temporary.chmod(0o600)
        json.dump(record, stream)
    temporary.replace(path)


def wait_ready(pid: int, path: Path, *, timeout: float = 30) -> dict:
    """Wait for one interpreter; never respawn it or adopt a reused PID."""
    try:
        expected = kernel_identity(pid)
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise StartupExited(f"Isaac launcher {pid} exited before interpreter readiness") from exc
    deadline = time.monotonic() + timeout
    while True:
        try:
            current = kernel_identity(pid)
        except (FileNotFoundError, ProcessLookupError) as exc:
            raise StartupExited(f"Isaac launcher {pid} exited before interpreter readiness") from exc
        if current["birth"] != expected["birth"]:
            raise RuntimeError("Isaac launcher PID was reused during interpreter startup")
        if current["state"] in ("Z", "X"):
            raise StartupExited(f"Isaac launcher {pid} exited before interpreter readiness")
        if path.exists():
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_size > 4096):
                raise RuntimeError("Isaac interpreter marker is not a private owned file")
            record = json.loads(path.read_text())
            if (not isinstance(record, dict) or record.get("launcher_pid") != pid or record.get("launcher_birth") != expected["birth"]
                    or type(record.get("bridge_pid")) is not int or record["bridge_pid"] <= 1):
                raise RuntimeError("Isaac interpreter marker does not match the launched process")
            bridge = kernel_identity(record["bridge_pid"])
            if bridge["birth"] != record.get("bridge_birth") or bridge["state"] in ("Z", "X"):
                raise RuntimeError("Isaac bridge identity changed before registration")
            return record
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Isaac interpreter readiness timed out after {timeout:g}s for PID {pid}")
        time.sleep(.02)


def clean_environment(original: Mapping[str, str], *, source: str | None) -> dict[str, str]:
    # Keep desktop/device selection, not the caller's venv, Kit settings,
    # profiling injection, site packages, authentication tokens or PythonEXE.
    keep = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "DISPLAY", "WAYLAND_DISPLAY",
            "XAUTHORITY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "VK_ICD_FILENAMES", "__GLX_VENDOR_LIBRARY_NAME",
            "OMNI_KIT_ACCEPT_EULA", "CASCADE_PHYSICS_DEVICE", "CASCADE_REQUIRE_CUDA", "CASCADE_BRIDGE_BIND",
            "CASCADE_BRIDGE_NO_TARGETS", "CASCADE_COMPANION_EXTS",
            "CASCADE_PROOF_CAMERA", "CASCADE_ISAAC_PIXEL_MASK", "CASCADE_ISAAC_CONTACT_MASK", "XDG_CACHE_HOME",
            "CASCADE_ISAAC_WIDTH", "CASCADE_ISAAC_HEIGHT", "CASCADE_ISAAC_CAM_EVERY", "CASCADE_ISAAC_DT",
            "XDG_CONFIG_HOME", "CUDA_CACHE_PATH", "WARP_CACHE_PATH",
            "__GL_SHADER_DISK_CACHE_PATH")
    env = {key: original[key] for key in keep if key in original}
    env.update(PATH="/usr/local/bin:/usr/bin:/bin", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    if source:
        env["ISAACSIM_PATH"] = source
    if sys.platform.startswith("linux") and os.uname().machine == "aarch64":
        library = Path("/lib/aarch64-linux-gnu/libgomp.so.1")
        if library.is_file():
            env["LD_PRELOAD"] = str(library)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--wait-ready", type=int, metavar="PID")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (args.ready_file is not None or args.wait_ready is not None) and not sys.platform.startswith("linux"):
        parser.error("Isaac interpreter readiness requires Linux /proc process identities")
    if args.wait_ready is not None:
        if args.python or args.args or args.ready_file is None:
            parser.error("--wait-ready requires only --ready-file")
        try:
            record = wait_ready(args.wait_ready, args.ready_file)
        except StartupExited as exc:
            print(f"[isaac-launch] {exc}", file=sys.stderr, flush=True)
            return 4
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[isaac-launch] {exc}", file=sys.stderr, flush=True)
            return 3
        print(f"[isaac-launch] bridge interpreter ready (launcher={record['launcher_pid']}, bridge={record['bridge_pid']})", flush=True)
        return 0
    if not args.python:
        parser.error("--python is required to start Isaac")
    python = Path(args.python).expanduser().absolute()
    source = str(python.parent) if python.name == "python.sh" else None
    command = args.args[1:] if args.args[:1] == ["--"] else args.args
    if not command:
        parser.error("an Isaac script/metadata command is required after --")
    env = clean_environment(os.environ, source=source)
    if args.ready_file is not None:
        if args.ready_file.exists():
            parser.error("the interpreter marker path must be new")
        identity = kernel_identity(os.getpid())
        env.update(CASCADE_ISAAC_STARTUP_READY=str(args.ready_file.absolute()),
                   CASCADE_ISAAC_STARTUP_PID=str(identity["pid"]), CASCADE_ISAAC_STARTUP_BIRTH=identity["birth"])
    if source is None:
        # Keep Kit in the owned launcher group even if the launcher exits first.
        os.execve(str(python), [str(python), *command], env)
    child = None
    pending = []

    def forward(signum, _frame):
        if child is None:
            pending.append(signum)
        else:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    child = subprocess.Popen([str(python), *command], env=env,
                             start_new_session=True)
    for signum in pending:
        forward(signum, None)
    code = child.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
