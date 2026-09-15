#!/usr/bin/env python3
"""Deterministic, explicit allowlist package. Python standard library only."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
import hashlib
ROOT = Path(__file__).resolve().parent
FILES = ['manifest.json','worker.js','inject.js','popup.html','popup.mjs','panel.html','panel.mjs','zoom.mjs','core.mjs','discovery.mjs','history.mjs','ui.css','README.md']
def main():
    destination = ROOT / 'Physical-Agentic-AI-OpenClaw-Demo.zip'
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(FILES):
            item = ZipInfo(name, date_time=(2026,1,1,0,0,0))
            item.compress_type = ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            archive.writestr(item, (ROOT / name).read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT / 'Physical-Agentic-AI-OpenClaw-Demo.zip.sha256').write_text(f'{digest}  {destination.name}\n')
    print(f'{digest}  {destination.name} ({destination.stat().st_size} bytes)')
if __name__ == '__main__': main()
