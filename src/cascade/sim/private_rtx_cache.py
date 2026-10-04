"""Opt-in writable RTX caches for one exact SDK; no Kit import during admission.

Cache bytes are deployment inputs, not physical model evidence. A seed contains
manifest.json plus data/{shadercache,nv_shadercache}; the caller independently
pins its manifest after the warming process has closed successfully.
"""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import stat

from .mobile_identity import canonical_bytes, digest_token
from .microduck_sdk import admit_release


SDK_SOURCES = {
    'exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py':
        'c2a2173be57abac20ef97f81b80d58319c94c9a4017c98709f8d13489ad0b746',
    'extscache/omni.gpu_foundation-0.0.0+1066600b.lx64.r.cp312/omni/gpu_foundation_factory/impl/foundation_extension.py':
        'd03d1f584ab8153ed796b040b15b7ef8caf37f6454e540434db5c674273720e4',
}
IMPORTED_SOURCES = dict(zip((
    'isaacsim.simulation_app.simulation_app',
    'omni.gpu_foundation_factory.impl.foundation_extension'), SDK_SOURCES))
LAYOUT = {
    '/rtx/shaderDb/shaderCachePath': 'shadercache',
    '/rtx/shaderDb/driverShaderCachePath': 'nv_shadercache',
}


def _path(path):
    path = Path(path).expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('private RTX cache paths must not contain symlinks')
    if path != path.resolve():
        raise ValueError('private RTX cache paths must be canonical')
    return path


@contextmanager
def _regular_file(path):
    # Refuse special files and leaf symlinks before reading. Detect replacement
    # or writes across a read, including same-size changes during seed copying.
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('RTX cache accepts regular files only')
        yield stream
        after = os.fstat(stream.fileno())
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    final = Path(path).lstat()
    if any(getattr(before, k) != getattr(after, k) or getattr(after, k) != getattr(final, k)
           for k in fields):
        raise ValueError('RTX cache file changed during read')


def _read(path):
    with _regular_file(path) as stream:
        return stream.read()


def _file(path, destination=None):
    from contextlib import nullcontext
    hasher, size = hashlib.sha256(), 0
    with _regular_file(path) as source, (Path(destination).open('xb') if destination else nullcontext()) as target:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
            if target is not None:
                target.write(chunk)
    return {'size': size, 'sha256': hasher.hexdigest()}


def _sha(path):
    return _file(path)['sha256']


def policy(release, sdk_recipe):
    """Exact local SDK bytes, with no live dependency imports or writes."""
    admit_release(release, sdk_recipe)
    actual = {name: _sha(Path(release) / name) for name in SDK_SOURCES}
    if actual != SDK_SOURCES:
        raise ValueError('private RTX cache SDK source mismatch')
    return {'schema': 'cascade.private-rtx-cache.v1', 'sdk_recipe': sdk_recipe,
            'sdk_source_sha256': actual, 'relative_settings': dict(LAYOUT)}


def inventory(root):
    """Complete membership including empty directories; reject links/devices."""
    root = _path(root)
    if not root.is_dir():
        raise ValueError('RTX cache data directory missing')
    def members():
        fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        paths = [root, *sorted(root.rglob('*'))]
        result = {}
        for path in paths:
            metadata = path.lstat()
            result[path.relative_to(root).as_posix()] = tuple(getattr(metadata, k) for k in fields)
        return result
    before = members()
    dirs, files = [], []
    for name, metadata in before.items():
        if name == '.':
            continue
        path, mode = root / name, metadata[2]
        if PurePosixPath(name).parts[0] not in LAYOUT.values():
            raise ValueError('unexpected RTX cache member')
        if stat.S_ISDIR(mode):
            dirs.append(name)
        elif stat.S_ISREG(mode):
            files.append({'path': name, **_file(path)})
        else:
            raise ValueError('RTX cache member must be regular; symlinks forbidden')
    if not set(LAYOUT.values()).issubset(dirs):
        raise ValueError('both RTX cache directories are required')
    if members() != before:
        raise ValueError('RTX cache tree changed during inventory')
    return {'directories': dirs, 'files': files}


def _seed(root, digest, expected_policy):
    root = _path(root)
    if not root.is_dir() or {p.name for p in root.iterdir()} != {'manifest.json', 'data'}:
        raise ValueError('RTX cache seed membership differs from manifest/data contract')
    raw = _read(_path(root / 'manifest.json'))
    if hashlib.sha256(raw).hexdigest() != digest_token(digest):
        raise ValueError('RTX cache seed manifest SHA mismatch')
    from .microduck_newton import strict_json
    value = strict_json(raw)
    if (not isinstance(value, dict) or set(value) != {'schema', 'policy', 'inventory'}
            or value['schema'] != 'cascade.private-rtx-cache-seed.v1'
            or value['policy'] != expected_policy):
        raise ValueError('RTX cache seed policy/source mismatch')
    actual = inventory(root / 'data')
    if canonical_bytes(value['inventory']) != canonical_bytes(actual):
        raise ValueError('RTX cache seed inventory differs; missing, extra or changed member')
    return actual


def admit(args):
    enabled = getattr(args, 'private_rtx_cache', False)
    seed = getattr(args, 'rtx_cache_seed', None)
    digest = getattr(args, 'rtx_cache_seed_sha256', None)
    if type(enabled) is not bool or (seed is None) != (digest is None):
        raise ValueError('private RTX cache requires paired seed path/SHA')
    if not enabled:
        if seed is not None:
            raise ValueError('RTX cache seed requires --private-rtx-cache')
        return None
    contract = policy(args.release, args.sdk_recipe)
    root = _path(Path(args.out) / 'rtx-cache')
    if root.exists():
        raise ValueError('private RTX cache output already exists')
    if root.is_relative_to(Path(args.release).resolve()):
        raise ValueError('private RTX cache must not modify SDK inputs')
    selected = {'root': str(root), 'policy': contract, 'seed': None}
    if seed is not None:
        seed = _path(seed)
        if root.is_relative_to(seed) or seed.is_relative_to(root):
            raise ValueError('RTX cache seed and output overlap')
        selected['seed'] = {'path': str(seed), 'manifest_sha256': digest_token(digest),
                            'inventory': _seed(seed, digest, contract)}
    return selected


def prepare(args, admitted):
    """Recheck, then materialize an exclusive writable copy before Kit imports."""
    if admit(args) != admitted:
        raise ValueError('private RTX cache admission changed before bootstrap')
    root = _path(admitted['root'])
    root.mkdir(mode=0o700, exist_ok=False)
    seed = admitted['seed']
    expected = seed['inventory'] if seed else {'directories': sorted(LAYOUT.values()), 'files': []}
    for name in sorted(expected['directories'], key=lambda s: (len(PurePosixPath(s).parts), s)):
        (root / name).mkdir(mode=0o700, exist_ok=False)
    if seed:
        for row in expected['files']:
            copied = _file(_path(Path(seed['path']) / 'data' / row['path']), root / row['path'])
            if copied != {k: row[k] for k in ('size', 'sha256')}:
                raise ValueError('RTX cache seed changed during copy')
        if _seed(seed['path'], seed['manifest_sha256'], admitted['policy']) != expected:
            raise ValueError('RTX cache seed changed after copy')
    if inventory(root) != expected:
        raise ValueError('private RTX cache copied inventory mismatch')
    record = {**admitted, 'input_inventory': expected,
              'requested_settings': {k: str(root / v) for k, v in LAYOUT.items()}}
    with (Path(args.out) / 'rtx-cache.json').open('xb') as stream:
        stream.write(canonical_bytes(record))
    return record


def extra_args(admitted):
    return [f'--{key}={Path(admitted["root"]) / relative}' for key, relative in LAYOUT.items()]


def verify_effective(admitted, settings, imported_files):
    """Verify effective settings and actual loaded implementations, not defaults."""
    actual_sources = {relative: _sha(imported_files[module])
                      for module, relative in IMPORTED_SOURCES.items()}
    if actual_sources != admitted['policy']['sdk_source_sha256']:
        raise ValueError('private RTX cache loaded SDK source mismatch')
    measured = {}
    for key, relative in LAYOUT.items():
        actual = settings.get(key)
        expected = Path(admitted['root']) / relative
        if not isinstance(actual, str) or _path(actual) != expected or not expected.is_dir():
            raise ValueError('private RTX cache effective setting mismatch: ' + key)
        measured[key] = actual
    record = {'effective_settings': measured, 'loaded_sdk_source_sha256': actual_sources}
    with (Path(admitted['root']).parent / 'rtx-cache-effective.json').open('xb') as stream:
        stream.write(canonical_bytes(record))
    return record
