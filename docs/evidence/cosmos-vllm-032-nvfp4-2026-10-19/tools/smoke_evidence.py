#!/usr/bin/env python3
"""Compatibility smoke verdicts on a PRISTINE vLLM 0.32.0 venv: as installed, each patch alone, all patches.

    smoke_evidence.py <scripts/cosmos_serving.py> <venv site-packages> <export> <venv python> <out.json>

Candidates are overlaid from a temporary directory (the script's own proof mechanism); nothing is installed.
"""
import json
import runpy
import sys
import tempfile
from pathlib import Path

script, site, export, python, out = sys.argv[1:6]
h = runpy.run_path(script)
def overlay_for(patches, d):
    files = {}
    for patch in patches:
        for spec in patch["files"]:
            f = Path(d) / (spec["module"] + ".py")
            f.write_text(h["apply_edits"]((Path(site) / spec["path"]).read_text(), spec["edits"]))
            files[spec["module"]] = f
    return files
result = {"stack": "vllm 0.32.0, torch 2.13.0+cu130, transformers 5.19.0 (pristine wheels)"}
with tempfile.TemporaryDirectory() as d:
    result["installed"] = h["run_smokes"](export, {}, python=python)
for patch in h["COMPAT_PATCHES"]:
    with tempfile.TemporaryDirectory() as d:
        result["only " + patch["name"]] = {k: v["ok"] for k, v in h["run_smokes"](export, overlay_for([patch], d), python=python).items()}
with tempfile.TemporaryDirectory() as d:
    result["all patches"] = h["run_smokes"](export, overlay_for(h["COMPAT_PATCHES"], d), python=python)
Path(out).write_text(json.dumps(result, indent=1))
print(json.dumps({k: (v if isinstance(v, str) else {n: (r if isinstance(r, bool) else r["ok"]) for n, r in v.items()}) for k, v in result.items()}, indent=1))
