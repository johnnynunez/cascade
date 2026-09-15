"""Exercise portable assembly and admission without native runtimes or downloads."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_bundle as builder
import prepare_bundle as bundle


def expected(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "provenance": "Licensed fixture"}


@pytest.fixture
def distribution(tmp_path, monkeypatch):
    checkout, assets, output = (tmp_path / name for name in ("checkout", "assets", "distribution"))
    for name in bundle.RUNTIME_FILES:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Source fixture\n")
    (checkout / "demo/scene/kitchen_config.json").write_text(json.dumps({
        "background": "assets/background.usda", "props": [{"asset": "assets/orange70.usda"}]}))
    data = {"demo/scene/assets/background.usda": b"licensed background fixture",
            "demo/scene/assets/orange70.usda": b"licensed orange fixture",
            "demo/vendor-kitchen/LICENSE.txt": b"license fixture"}
    for name, value in data.items():
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    pin = tmp_path / "asset-manifest.json"
    pin.write_text(json.dumps({"files": {n: expected(v) for n, v in data.items()},
                               "licenses": {"fixtures": "Retain fixture notices"}}))
    monkeypatch.setattr(bundle, "ASSET_MANIFEST", pin)
    profile = {"source": {"origin": "https://example.invalid/cascade.git", "revision": "1" * 40}}
    return checkout, assets, output, profile


def assemble(distribution, boundary=lambda: None):
    return builder.assemble(*distribution, boundary=boundary)


def test_assembled_runtime_is_admitted_by_existing_prepare(distribution, tmp_path, monkeypatch):
    result = assemble(distribution)
    _, _, output, profile = distribution
    archive, clip = tmp_path / "openclaw.tar.zst", tmp_path / "clip.tar.gz"
    archive.write_bytes(b"native fixture")
    clip.write_bytes(b"clip fixture")
    monkeypatch.setattr(bundle, "BUILD_INPUTS", {
        "openclaw": expected(archive.read_bytes()), "clip": expected(clip.read_bytes())})
    manifest = output / "PORTABLE_BUNDLE.json"
    record = json.loads(manifest.read_text())
    record["build_inputs"] = bundle.BUILD_INPUTS
    manifest.write_text(json.dumps(record))
    admitted = bundle.prepare(tmp_path / "controller", profile, output / "source", manifest,
                              archive, clip, boundary=lambda: None)
    assert admitted["source_files"] == result["files"]
    assert result["downloads"] == admitted["downloads"] == 0
    assert (output / "source/demo/vendor-kitchen/LICENSE.txt").read_bytes() == b"license fixture"
    assert not (output / "source/kitchen").exists()


def test_repeat_assembly_reuses_certified_asset_bytes(distribution, monkeypatch):
    first = assemble(distribution)
    _, assets, output, _ = distribution
    old_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(assets) or path.is_relative_to(output / "source"):
            if path.name != "kitchen_config.json":
                pytest.fail("Unchanged certified asset or output was read again")
        return old_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = assemble(distribution)
    assert result["copied"] == 0
    assert result["reused"] == first["files"]


def test_missing_licensed_asset_prevents_manifest(distribution):
    checkout, assets, output, _ = distribution
    (assets / "demo/vendor-kitchen/LICENSE.txt").rename(assets / "held-license.txt")
    with pytest.raises(ValueError, match="missing"):
        assemble(distribution)
    assert not (output / "PORTABLE_BUNDLE.json").exists()


def test_source_change_preserves_existing_distribution(distribution):
    assemble(distribution)
    checkout, _, output, _ = distribution
    before = (output / "source/LICENSE").read_bytes()
    (checkout / "LICENSE").write_bytes(b"replacement source notice")
    with pytest.raises(ValueError, match="unexpected size|checksum differs"):
        assemble(distribution)
    assert (output / "source/LICENSE").read_bytes() == before


def test_output_symlink_cannot_write_outside_bundle(distribution, tmp_path):
    _, _, output, _ = distribution
    outside = tmp_path / "outside"
    outside.mkdir()
    output.mkdir()
    (output / "source").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves its output"):
        assemble(distribution)
    assert not list(outside.iterdir())


def test_output_inside_checkout_is_refused(distribution):
    checkout, assets, _, profile = distribution
    with pytest.raises(ValueError, match="separate"):
        builder.assemble(checkout, assets, checkout / "output", profile, boundary=lambda: None)


def test_scene_reference_must_belong_to_admitted_files(distribution):
    checkout, _, output, _ = distribution
    (checkout / "demo/scene/kitchen_config.json").write_text(json.dumps({
        "background": "../../private/background.usda", "props": []}))
    with pytest.raises(ValueError, match="outside the admitted bundle"):
        assemble(distribution)
    assert not (output / "PORTABLE_BUNDLE.json").exists()


def test_stop_before_manifest_preserves_inputs_without_admission(distribution, monkeypatch):
    _, _, output, _ = distribution
    stopped = False
    original = bundle.runtime_inventory

    def inventory(*args):
        nonlocal stopped
        original(*args)
        stopped = True

    def boundary():
        if stopped:
            raise InterruptedError("STOP fixture")

    monkeypatch.setattr(bundle, "runtime_inventory", inventory)
    with pytest.raises(InterruptedError):
        assemble(distribution, boundary)
    assert not (output / "PORTABLE_BUNDLE.json").exists()
    assert (output / "source/LICENSE").exists()


def test_source_selection_omits_operator_material_and_tests(distribution):
    checkout = distribution[0]
    for name in ("OWNER_UPDATE.md", "kitchen/proof/private.py", "deploy/brev/tests/test_private.py"):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private fixture")
    assert not set(builder.source_files(checkout)) & {
        "OWNER_UPDATE.md", "kitchen/proof/private.py", "deploy/brev/tests/test_private.py"}


def test_assembled_extension_loads_manifest_worker_and_contains_page_modules(distribution):
    checkout, _, output, _ = distribution
    extension_source = builder.HERE.parents[1] / "extensions/chrome"
    extension_copy = checkout / "extensions/chrome"
    extension_copy.mkdir(parents=True)
    for path in extension_source.iterdir():
        if path.is_file() and path.suffix in {".js", ".mjs", ".html", ".json", ".css"}:
            shutil.copyfile(path, extension_copy / path.name)
    assemble(distribution)
    extension = output / "source/extensions/chrome"
    manifest = json.loads((extension / "manifest.json").read_text())
    pending = [manifest["background"]["service_worker"], manifest["action"]["default_popup"],
               manifest["side_panel"]["default_path"].split("?", 1)[0]]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        path = extension / name
        content = path.read_text()
        if path.suffix == ".html":
            references = re.findall(r'<script\b[^>]*\bsrc=["\x27]([^"\x27]+)', content)
        else:
            references = re.findall(r'\bfrom\s*["\x27](\./[^"\x27]+)["\x27]', content)
        pending.extend((Path(name).parent / ref).as_posix() for ref in references)
    javascript = """
import {pathToFileURL} from 'node:url';
const registered = [];
const event = name => ({addListener: listener => registered.push(name)});
globalThis.chrome = {
  history: {onVisited: event('history')},
  runtime: {onInstalled: event('install'), onMessage: event('message')},
  tabs: {onRemoved: event('remove'), onUpdated: event('update')}
};
await import(pathToFileURL(process.argv[2]).href);
if (!registered.includes('message') || !registered.includes('history')) process.exit(1);
"""
    result = subprocess.run(["node", "--experimental-default-type=module", "--input-type=module", "-",
                             str(extension / manifest["background"]["service_worker"])],
                            input=javascript, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_generated_mcp_and_isaac_entrypoints_are_selected(tmp_path):
    checkout = builder.HERE.parents[1]
    names = set(builder.source_files(checkout))
    tree = ast.parse((checkout / "deploy/runtime/runtime.py").read_text())
    setup = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup")
    config = next(node.value for node in ast.walk(setup) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "config" for target in node.targets))
    env = {"PAAI_OPENCLAW_UI_ROOT": str(tmp_path / "ui"), "PAAI_APP_PYTHON": sys.executable,
           "PAAI_PERCEPTION_DIR": str(tmp_path / "models")}
    context = {"env": env, "token": "fixture", "origins": [], "workspace": tmp_path / "workspace",
               "provider": {}, "owner": {"owner": "fixture"}, "launch": tmp_path / "launch",
               "profile": tmp_path / "profile",
               "mcp_env": {}, "MODEL_ID": "Qwen/Qwen3.8-27B"}
    generated = eval(compile(ast.Expression(config), "runtime.py", "eval"), context)
    mcp = generated["mcp"]["servers"]["cascade"]
    assert mcp["command"] == sys.executable
    assert mcp["args"][:2] == ["-m", "cascade.apps.mcp_server"]
    assert "src/" + mcp["args"][1].replace(".", "/") + ".py" in names
    scripts = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
               and isinstance(node.value, str) and node.value.startswith("scripts/")}
    assert scripts and scripts <= names
    pending = ["deploy/runtime/runtime.py", "scripts/isaac_bridge.py", "src/cascade/apps/mcp_server.py"]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(ast.parse((checkout / name).read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and not node.level else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                for prefix in ("scripts", "demo", "demo/kitchen", "deploy/runtime", "deploy/brev/streaming"):
                    dependency = f"{prefix}/{module.replace('.', '/')}.py"
                    if (checkout / dependency).is_file():
                        assert dependency in names
                        pending.append(dependency)
