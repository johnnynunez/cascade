"""Offline admission, real git patching, and installed origin; no speech imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def source_bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    source = tmp_path / 'source'
    name = 'src/speech_to_speech/__init__.py'
    path = source / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b'OLD = True\n')
    (path.parent / 'other.py').write_bytes(b'# untouched\n')
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    patch = b'diff --git a/src/speech_to_speech/__init__.py b/src/speech_to_speech/__init__.py\n--- a/src/speech_to_speech/__init__.py\n+++ b/src/speech_to_speech/__init__.py\n@@ -1 +1 @@\n-OLD = True\n+NEW = True\n'
    (config_dir / 'source.patch').write_bytes(patch)
    spec = {'schema': 'cascade.speech-provider-source.v1', 'upstream_revision': 'a'*40,
            'candidate_revision': 'b'*40, 'archive_sha256': 'c'*64, 'patch': 'source.patch',
            'patch_sha256': sha(patch), 'files': {name: {'before_sha256': sha(path.read_bytes()),
                                                       'after_sha256': sha(b'NEW = True\n')}}}
    raw = json.dumps(spec).encode()
    (config_dir / 'manifest.json').write_bytes(raw)
    config = {'source_manifest': 'manifest.json', 'source_manifest_sha256': sha(raw),
              'source_revision': 'a'*40, 'archives': {'speech-to-speech.tar.gz': {'sha256': 'c'*64}}}
    (config_dir / 'recipe.json').write_text(json.dumps(config))
    return config, source, config_dir


def test_real_patch_exact_outputs_and_refuses_repeat(source_bundle):
    from conversation_provider_source import apply_patch, inventory
    config, source, root = source_bundle
    before = inventory(source)
    result = apply_patch(source, config, root)
    assert result['runtime_files']['__init__.py'] == sha(b'NEW = True\n')
    assert inventory(source)['src/speech_to_speech/other.py'] == before['src/speech_to_speech/other.py']
    with pytest.raises(ValueError, match='input mismatch'):
        apply_patch(source, config, root)


@pytest.mark.parametrize('change', ['source', 'patch', 'manifest', 'symlink'])
def test_patch_refuses_changed_source_or_bundle_without_applying(source_bundle, change):
    from conversation_provider_source import apply_patch
    config, source, root = source_bundle
    path = source / 'src/speech_to_speech/__init__.py'
    if change == 'source':
        path.write_text('foreign\n')
    elif change == 'symlink':
        other = path.with_name('other.py'); path.unlink(); path.symlink_to(other)
    else:
        p = root / ('source.patch' if change == 'patch' else 'manifest.json')
        p.write_bytes(p.read_bytes() + b' ')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        apply_patch(source, config, root)
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['bytes', 'extra', 'missing', 'origin', 'environment'])
def test_installed_package_admits_exact_normal_install_and_rejects_drift(source_bundle, monkeypatch, change):
    import conversation_provider_source as verifier
    config, source, root = source_bundle
    record = verifier.apply_patch(source, config, root)
    state = source.parent
    package = state / 'venv/lib/python3.12/site-packages/speech_to_speech'
    shutil.copytree(source / 'src/speech_to_speech', package)
    monkeypatch.setattr(sys, 'prefix', str(state / 'venv'))
    monkeypatch.setattr(verifier.sysconfig, 'get_path', lambda _: str(package.parent))
    spec = importlib.util.spec_from_file_location('speech_to_speech', package / '__init__.py',
                                                 submodule_search_locations=[str(package)])
    monkeypatch.setattr(verifier.importlib.util, 'find_spec', lambda _: spec)
    assert verifier.verify_installed(state, record)['files'] == 2
    if change == 'bytes': (package / '__init__.py').write_text('changed')
    elif change == 'extra': (package / 'extra.py').write_text('unexpected')
    elif change == 'missing': (package / 'other.py').unlink()
    elif change == 'origin': spec.origin = str(source / 'src/speech_to_speech/__init__.py')
    else: monkeypatch.setattr(sys, 'prefix', str(state / 'other-venv'))
    with pytest.raises(ValueError):
        verifier.verify_installed(state, record)
