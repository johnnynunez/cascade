#!/usr/bin/env python3
"""Count the tracked files that name each NVIDIA technology (stdlib only).

Run from the repository root:  python3 docs/evidence/nvidia-stack-20261018/tools/scan_nvidia.py
Writes nothing; prints a Markdown table (code = src/ scripts/ configs/ deploy/ extensions/ demo/,
tests = tests/ benchmark/, docs = docs/ README.md AGENTS.md without docs/evidence/).
A file counts once per technology however often it names it. Name matches are a lead,
not proof of use: read the hits before you classify a technology.
"""
from __future__ import annotations

import re
import subprocess
import sys

TERMS = {
    "Isaac Sim / Kit": r"isaac ?sim|isaacsim\.|omni\.(kit|usd|timeline|replicator|physx)",
    "Omniverse Replicator": r"omni\.replicator",
    "PhysX": r"physx",
    "Newton": r"\bnewton\b",
    "Isaac Lab": r"isaac ?lab|isaaclab",
    "Isaac Lab-Arena": r"isaaclab[-_ ]arena|lab-arena|arena_?experiment",
    "Warp": r"warp[-_]lang|import warp|omni\.warp|\bwp\.",
    "MuJoCo Warp": r"mujoco[-_ ]?warp|mjwarp",
    "OVRTX / ovstage": r"ovrtx|ovstage",
    "nvblox": r"nvblox",
    "cuVSLAM": r"cuvslam",
    "cuMotion": r"cumotion",
    "GraspGen-X": r"graspgen",
    "Cosmos": r"cosmos",
    "Nemotron": r"nemotron",
    "OpenShell": r"openshell",
    "NemoClaw": r"nemoclaw",
    "ASPIRE": r"\baspire\b",
    "DGX Spark / GB10": r"dgx ?spark|\bgb10\b",
    "Jetson": r"jetson",
    "Brev": r"\bbrev\b",
    "NGC (nvcr.io)": r"nvcr\.io",
    "NVENC": r"nvenc",
    "NVFP4": r"nvfp4",
    "CUDA": r"\bcuda\b",
    "TensorRT": r"tensorrt",
    "Isaac ROS": r"isaac[-_ ]ros",
    "GR00T": r"gr00t",
    "cuRobo": r"curobo",
}
BINARY = (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".npz", ".npy", ".gz", ".pt", ".onnx", ".bin",
          ".glb", ".stl", ".obj", ".dae", ".usd", ".usdc", ".usdz", ".ply", ".safetensors", ".pdf")
GROUPS = {
    "code": ("src/", "scripts/", "configs/", "deploy/", "extensions/", "demo/", "run.sh", "install_deps.sh",
             "pyproject.toml"),
    "tests": ("tests/", "benchmark/"),
    "docs": ("docs/", "README.md", "AGENTS.md"),
}


def group_of(path: str) -> str | None:
    if path.startswith("docs/evidence/"):
        return None
    for name, prefixes in GROUPS.items():
        if path.startswith(prefixes):
            return name
    return None


def main() -> int:
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    texts: dict[str, tuple[str, str]] = {}
    for path in files:
        group = group_of(path)
        if path.lower().endswith(BINARY) or group is None:
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                texts[path] = (group, handle.read())
        except OSError:
            continue
    head = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    print(f"source {head}, {len(texts)} text files scanned\n")
    print("| technology | code | tests | docs |")
    print("|---|---|---|---|")
    for name, pattern in TERMS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        counts = {"code": 0, "tests": 0, "docs": 0}
        for group, text in texts.values():
            if rx.search(text):
                counts[group] += 1
        print(f"| {name} | {counts['code']} | {counts['tests']} | {counts['docs']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
