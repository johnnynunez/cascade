#!/usr/bin/env python3
"""Assemble the Firefox build from the shared Chrome sources. Standard library only.

Writes a loadable directory (build/) and a deterministic .xpi. The file
allowlist comes from ../chrome/package.py; only manifest.json and README.md
are Firefox specific.
"""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
import argparse, hashlib, importlib.util, shutil
ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / 'chrome'
spec = importlib.util.spec_from_file_location('chrome_package', SHARED / 'package.py')
assert spec and spec.loader
chrome = importlib.util.module_from_spec(spec); spec.loader.exec_module(chrome)
OWN = ('manifest.json', 'README.md')
def sources():
    files = {name: SHARED / name for name in chrome.FILES if name not in OWN}
    files.update({name: ROOT / name for name in OWN})
    return dict(sorted(files.items()))
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT, help='output directory (default: this directory)')
    out = parser.parse_args(argv).out.resolve()
    build, destination = out / 'build', out / 'Physical-Agentic-AI-OpenClaw-Demo.xpi'
    shutil.rmtree(build, ignore_errors=True); build.mkdir(parents=True)
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in sources().items():
            data = path.read_bytes()
            (build / name).write_bytes(data)
            item = ZipInfo(name, date_time=(2026,1,1,0,0,0))
            item.compress_type = ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data, compress_type=ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (out / (destination.name + '.sha256')).write_text(f'{digest}  {destination.name}\n')
    print(f'{digest}  {destination.name} ({destination.stat().st_size} bytes); unpacked: {build}')
if __name__ == '__main__': main()
