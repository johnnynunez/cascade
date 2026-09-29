#!/usr/bin/env python3
"""Check DGX Spark host prerequisites without installing or starting the demo."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


MIN_FREE_BYTES = 150 * 1024**3
COMMANDS = ("bash", "git", "curl", "python3", "c++", "gio", "gnome-terminal")


def output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(command)}")
    return result.stdout.strip()


def check() -> list[str]:
    problems: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    require(platform.system() == "Linux" and platform.machine() == "aarch64",
            "Use NVIDIA DGX Spark with Linux aarch64.")
    for command in COMMANDS:
        require(shutil.which(command) is not None, f"Missing command: {command}.")

    for label, command, expected in (
        ("glibc", ["getconf", "GNU_LIBC_VERSION"], r"glibc (\d+)\.(\d+)"),
        ("Git LFS", ["git", "lfs", "version"], r"git-lfs/"),
        ("NVIDIA driver", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], r".+,\s*\d+\."),
    ):
        try:
            value = output(command)
            match = re.search(expected, value)
            require(match is not None, f"Cannot verify {label}.")
            if label == "glibc" and match:
                require(tuple(map(int, match.groups())) >= (2, 35), "glibc 2.35 or newer is required.")
            print(f"{label}: {value}")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            problems.append(f"{label}: {exc}")

    nvcc = shutil.which("nvcc")
    if nvcc is None and os.access("/usr/local/cuda/bin/nvcc", os.X_OK):
        nvcc = "/usr/local/cuda/bin/nvcc"
    if nvcc is None:
        problems.append("CUDA toolkit: nvcc is missing; ask the DGX OS administrator.")
    else:
        try:
            value = output([nvcc, "--version"])
            version = re.search(r"release (\d+)\.(\d+)", value)
            require(version is not None and int(version[1]) >= 13,
                    "CUDA toolkit 13 or newer is required.")
            print(f"CUDA toolkit: {version[0] if version else 'unrecognized version'}")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            problems.append(f"CUDA toolkit: {exc}")

    try:
        ctypes.CDLL("libgomp.so.1")
    except OSError:
        problems.append("Missing libgomp.so.1; install libgomp1.")

    # A snap's --version can initialize user state and reject an isolated HOME.
    # The launcher's browser preflight checks execution in its own profile.
    browser = next((candidate for candidate in
                    (shutil.which("chromium"), shutil.which("chromium-browser"), "/snap/bin/chromium")
                    if candidate and os.access(candidate, os.X_OK)), None)
    require(browser is not None, "Chromium is missing; install chromium-browser.")
    if browser:
        print(f"Chromium executable: {browser}")

    home = Path.home()
    require(home.is_dir() and os.access(home, os.W_OK | os.X_OK), "HOME must be a writable user directory.")
    try:
        free = shutil.disk_usage(home).free
        print(f"Free disk in HOME: {free / 1024**3:.1f} GiB")
        require(free >= MIN_FREE_BYTES, "Keep at least 150 GiB free in HOME for the first installation.")
    except OSError as exc:
        problems.append(f"Cannot check HOME disk space: {exc}")
    return problems


def main() -> int:
    problems = check()
    if problems:
        for problem in problems:
            print(f"MISSING: {problem}", file=sys.stderr)
        return 2
    print("PREREQUISITES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
