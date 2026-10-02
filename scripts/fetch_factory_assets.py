"""Fetch the two licensed Factory thread meshes at reviewed SHA-256 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        target = directory / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']:
            continue
        url = (f"https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/"
               f"{manifest['commit']}/{record['upstream_path']}")
        data = urllib.request.urlopen(url, timeout=60).read()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Factory asset download failed integrity check: {name}')
        temporary = target.with_suffix(target.suffix + '.download')
        temporary.write_bytes(data)
        temporary.replace(target)
        print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'assets/factory/nut_bolt')
    fetch(parser.parse_args().directory)
