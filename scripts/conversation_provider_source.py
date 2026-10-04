"""Admit the reviewed source patch and its normal (non-editable) installation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inventory(directory, *, runtime=False):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError('provider source/package must not contain symlinks')
        if path.is_file() and not (runtime and path.suffix == '.md'):
            result[str(path.relative_to(directory))] = digest(path)
    return result


def bundle(recipe, config_dir):
    root = Path(config_dir).resolve()
    path = (root / recipe['source_manifest']).resolve()
    raw = path.read_bytes()
    if not path.is_relative_to(root) or hashlib.sha256(raw).hexdigest() != recipe['source_manifest_sha256']:
        raise ValueError('provider source manifest changed')
    spec = json.loads(raw)
    if (spec['schema'] != 'cascade.speech-provider-source.v1'
            or spec['upstream_revision'] != recipe['source_revision']
            or spec['archive_sha256'] != recipe['archives']['speech-to-speech.tar.gz']['sha256']):
        raise ValueError('provider source/archive identity mismatch')
    patch = (path.parent / spec['patch']).resolve()
    patch_bytes = patch.read_bytes()
    if not patch.is_relative_to(path.parent) or hashlib.sha256(patch_bytes).hexdigest() != spec['patch_sha256']:
        raise ValueError('provider source patch changed')
    for name in spec['files']:
        member = Path(name)
        if member.is_absolute() or '..' in member.parts or not name.startswith('src/speech_to_speech/'):
            raise ValueError('unsafe provider patch member')
    return spec, patch_bytes


def apply_patch(source, recipe, config_dir):
    """Only called on the fresh extraction of the hash-checked upstream archive."""
    source = Path(source).resolve()
    spec, patch_bytes = bundle(recipe, config_dir)
    before = inventory(source)
    expected = dict(before)
    for name, hashes in spec['files'].items():
        if before.get(name) != hashes['before_sha256']:
            raise ValueError('provider patch input mismatch: ' + name)
        expected[name] = hashes['after_sha256']
    # git applies only this pinned patch; no shell, hooks, fetch or repository mutation.
    command = ['git', 'apply', '--unidiff-zero', '--whitespace=error', '-']
    subprocess.run(command[:2] + ['--check'] + command[2:], cwd=source, check=True,
                   capture_output=True, input=patch_bytes)
    subprocess.run(command, cwd=source, check=True, capture_output=True, input=patch_bytes)
    if inventory(source) != expected:
        raise ValueError('provider patch changed unexpected files')
    return {'source_manifest_sha256': recipe['source_manifest_sha256'],
            'patch_sha256': spec['patch_sha256'], 'candidate_revision': spec['candidate_revision'],
            'files': spec['files'],
            'runtime_files': inventory(source / 'src/speech_to_speech', runtime=True)}


def verify_source(state, recipe, config_dir):
    spec, _ = bundle(recipe, config_dir)
    record = json.loads((Path(state) / 'source-patch.json').read_bytes())
    if (record['source_manifest_sha256'] != recipe['source_manifest_sha256']
            or record['patch_sha256'] != spec['patch_sha256']
            or record['candidate_revision'] != spec['candidate_revision']
            or record['files'] != spec['files']):
        raise ValueError('effective provider source identity changed')
    source = Path(state) / 'source'
    for name, hashes in spec['files'].items():
        if digest(source / name) != hashes['after_sha256']:
            raise ValueError('effective provider patch output changed: ' + name)
    current = inventory(source / 'src/speech_to_speech', runtime=True)
    if not current or current != record['runtime_files']:
        raise ValueError('effective provider runtime inventory changed')
    return record


def verify_installed(state, record):
    """Check bytes and import origin before importing the provider or its models."""
    state = Path(state).resolve()
    if Path(sys.prefix).resolve() != state / 'venv':
        raise ValueError('provider must use its private installed environment')
    purelib = Path(sysconfig.get_path('purelib')).resolve()
    expected = purelib / 'speech_to_speech'
    spec = importlib.util.find_spec('speech_to_speech')
    if (not purelib.is_relative_to(state / 'venv') or spec is None
            or spec.origin is None or Path(spec.origin).resolve() != expected / '__init__.py'
            or list(spec.submodule_search_locations or ()) != [str(expected)]):
        raise ValueError('provider import origin is not the private normal installation')
    if inventory(expected, runtime=True) != record['runtime_files']:
        raise ValueError('installed provider runtime bytes or membership changed')
    return {'package': str(expected), 'files': len(record['runtime_files']),
            'candidate_revision': record['candidate_revision'],
            'source_manifest_sha256': record['source_manifest_sha256']}
