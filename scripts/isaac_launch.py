#!/usr/bin/env python3
"""Start Isaac in an isolated environment with owned shutdown.

A managed wheel's Python replaces this adapter in the launcher's group.
A source release's python.sh spawns (does not exec) Kit Python. Owning only
that shell PID leaks Kit on shutdown. This stable adapter owns its private
child group and waits for its real shutdown. On Linux it also adopts and
reaps orphaned descendants, including escaped sessions. It never signals a
reused or externally started simulator.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import ctypes
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
    # The explicit bridge-zone flag enables only our bounded diagnostic helper.
    keep = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "DISPLAY", "WAYLAND_DISPLAY",
            "XAUTHORITY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "VK_ICD_FILENAMES", "__GLX_VENDOR_LIBRARY_NAME",
            "OMNI_KIT_ACCEPT_EULA", "CASCADE_PHYSICS_DEVICE", "CASCADE_REQUIRE_CUDA", "CASCADE_BRIDGE_BIND",
            "CASCADE_BRIDGE_NO_TARGETS", "CASCADE_COMPANION_EXTS",
            "CASCADE_PROOF_CAMERA", "CASCADE_ISAAC_PIXEL_MASK", "CASCADE_ISAAC_CONTACT_MASK", "XDG_CACHE_HOME",
            "CASCADE_ISAAC_WIDTH", "CASCADE_ISAAC_HEIGHT", "CASCADE_ISAAC_CAM_EVERY", "CASCADE_ISAAC_DT",
            "CASCADE_ISAAC_PYTHON_SPANS", "CASCADE_ISAAC_PYTHON_TIMINGS", "CASCADE_ISAAC_TARGET_RECEIPTS",
            "CASCADE_CAMERA_RENDERER", "CASCADE_OVRTX_PYTHON", "CASCADE_OVRTX_OUTPUT", "CASCADE_OVRTX_DEVICE",
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


def source_subreaper() -> bool:
    """This dedicated CLI adopts source-shell orphans before it starts Kit."""
    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return True


def source_group_live(pgid: int) -> bool:
    """Portable fallback, while the unreaped shell still reserves this PGID.

    Linux uses adoption instead, including descendants which leave the group.
    Other platforms can attest only this private group's live members.
    """
    result = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="],
                            capture_output=True, text=True, timeout=5,
                            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    if result.returncode or result.stderr or not result.stdout.endswith("\n"):
        raise RuntimeError("cannot attest source process-group shutdown")
    live = anchor = False
    for line in result.stdout.splitlines():
        pid, group, state = line.split()
        if int(pid) <= 0 or int(group) < 0 or not state[0].isalpha():
            raise RuntimeError(f"invalid source process-group snapshot: {line!r}")
        if int(group) == pgid and state[0] not in ("Z", "X"):
            live = True
        anchor |= int(pid) == pgid and int(group) == pgid
    if not anchor:
        raise RuntimeError("source process-group anchor is missing")
    return live


def wait_source(child, pending: list[int], *, adopted: bool) -> int:
    """Retain the adapter until actual exit; never escalate or respawn.

    Do not poll/reap the original shell early: its reserved PID pins the
    private group against reuse even after the shell has become a zombie.
    Linux orphans become our children and are reaped here, not left to PID 1.
    """
    failed = 0
    last_signal = None
    signalled = set()
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        while pending:
            last_signal = pending.pop(0)
            try:
                os.killpg(child.pid, last_signal)
            except ProcessLookupError:
                pass
            signalled.clear()
        if adopted:
            # Observe shell exit BEFORE enumerating adopted children. Its
            # exit reparents descendants before waitid reports completion.
            exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            for pid in map(int, children_path.read_text().split()):
                if pid == child.pid:
                    continue
                got, status = os.waitpid(pid, os.WNOHANG)
                if got:
                    code = os.waitstatus_to_exitcode(status)
                    failed = failed or (code if code >= 0 else 128 - code)
                    signalled.discard(pid)
                elif last_signal is not None and pid not in signalled and os.getpgid(pid) != child.pid:
                    # An escaped orphan is now our unreaped direct child, so
                    # its PID cannot be reused between this check and signal.
                    os.kill(pid, last_signal)
                    signalled.add(pid)
            if exited is None or set(children_path.read_text().split()) != {str(child.pid)}:
                time.sleep(.02)
                continue
        elif source_group_live(child.pid):
            time.sleep(.02)
            continue
        code = child.wait()
        return (code if code >= 0 else 128 - code) or failed or (128 + last_signal if last_signal else 0)


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
    pending = []

    def forward(signum, _frame):
        pending.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    # An inherited SIG_IGN would auto-reap the shell and release our PGID pin.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    adopted = source_subreaper()
    child = subprocess.Popen([str(python), *command], env=env,
                             start_new_session=True)
    return wait_source(child, pending, adopted=adopted)


if __name__ == "__main__":
    raise SystemExit(main())
