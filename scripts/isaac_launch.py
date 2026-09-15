#!/usr/bin/env python3
"""Start Isaac in an isolated environment and forward stop to its process group.

A source release's python.sh spawns (does not exec) Kit Python. Owning only
that shell PID leaks Kit on shutdown. This stable adapter owns its private
child group; it never signals a reused or externally started simulator.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import signal
import subprocess
import sys


def clean_environment(original: Mapping[str, str], *, source: str | None) -> dict[str, str]:
    # Keep desktop/device selection, not the caller's venv, Kit settings,
    # profiling injection, site packages, authentication tokens or PythonEXE.
    keep = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "DISPLAY", "WAYLAND_DISPLAY",
            "XAUTHORITY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "VK_ICD_FILENAMES", "__GLX_VENDOR_LIBRARY_NAME",
            "OMNI_KIT_ACCEPT_EULA", "CASCADE_PHYSICS_DEVICE", "CASCADE_REQUIRE_CUDA", "CASCADE_BRIDGE_BIND",
            "CASCADE_BRIDGE_NO_TARGETS", "CASCADE_COMPANION_EXTS",
            "CASCADE_PROOF_CAMERA", "CASCADE_ISAAC_PIXEL_MASK", "XDG_CACHE_HOME",
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
    parser.add_argument("--python", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    python = Path(args.python).expanduser().absolute()
    source = str(python.parent) if python.name == "python.sh" else None
    command = args.args[1:] if args.args[:1] == ["--"] else args.args
    if not command:
        parser.error("an Isaac script/metadata command is required after --")
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
    child = subprocess.Popen([str(python), *command], env=clean_environment(os.environ, source=source),
                             start_new_session=True)
    for signum in pending:
        forward(signum, None)
    code = child.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
