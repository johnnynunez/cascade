#!/usr/bin/env python3
"""Extract NVIDIA's checksummed CUDA 13.2.2 redistributions into a private prefix."""

import argparse
import concurrent.futures
import hashlib
import json
import platform
import subprocess
import urllib.request
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
PACKAGES = (
    "cuda_nvcc",
    "cuda_cudart",
    "cuda_cccl",
    "cuda_nvrtc",
    "cuda_cupti",
    "libcurand",
    "libcublas",
    "libcusparse",
    "libcusolver",
    "libcufft",
    "libnvjitlink",
    "libnpp",
    "cuda_nvtx",
    "cuda_profiler_api",
    "cuda_crt",
    "libnvvm",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    arches = {"x86_64": "linux-x86_64", "aarch64": "linux-sbsa"}
    arch = arches[platform.machine()]
    root = args.prefix.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + "redistrib_13.2.2.json", timeout=60) as response:
        manifest = json.load(response)
    (root / "redistrib.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cache = root / "archives"
    cache.mkdir(exist_ok=True)

    def fetch(name):
        entry = manifest[name][arch]
        archive = cache / Path(entry["relative_path"]).name
        if not archive.exists():
            pending = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(BASE + entry["relative_path"], pending)
            pending.replace(archive)
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch: {archive}; remove it before retrying"
            )
        return name, archive

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name, archive in pool.map(fetch, PACKAGES):
            subprocess.run(
                ["tar", "-xJf", str(archive), "--strip-components=1", "-C", str(root)],
                check=True,
            )
            print(f"Installed {name}", flush=True)
    if not (root / "lib64").exists():
        (root / "lib64").symlink_to("lib", target_is_directory=True)
    subprocess.run([str(root / "bin/nvcc"), "--version"], check=True)


if __name__ == "__main__":
    main()
