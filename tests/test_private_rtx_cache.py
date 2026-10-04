"""Offline cache admission/copy tests; fake Kit proves ordering, not RTX startup."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from cascade.sim import microduck_sdk as sdk
from cascade.sim import private_rtx_cache as cache
from cascade.sim.microduck_newton import KitNewtonBackend
from test_mobile_identity import recipe_inputs as recipe_inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def args(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    release = tmp_path / 'release'
    release.mkdir()
    (release / 'VERSION').write_text('synthetic exact SDK')
    monkeypatch.setattr(sdk, 'INTERNAL_VERSION_SHA256', digest(release / 'VERSION'))
    pins = {}
    for name, relative in sdk.INTERNAL_SOURCE_PATHS.items():
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic ' + name)
        pins[name] = digest(path)
    monkeypatch.setattr(sdk, 'INTERNAL_SOURCE_SHA256', pins)
    pins = {}
    for relative in cache.SDK_SOURCES:
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic cache routing ' + relative)
        pins[relative] = digest(path)
    monkeypatch.setattr(cache, 'SDK_SOURCES', pins)
    return NS(release=release, sdk_recipe=sdk.INTERNAL_RECIPE, out=tmp_path / 'episode',
              private_rtx_cache=True, rtx_cache_seed=None, rtx_cache_seed_sha256=None,
              device='cuda:0', reuse_solved_read=False)


def seed(args):
    root = args.out.parent / 'seed'
    data = root / 'data'
    for relative in ('shadercache/empty', 'nv_shadercache'):
        (data / relative).mkdir(parents=True)
    (data / 'shadercache/pso.bin').write_bytes(b'compiled fixture only')
    (data / 'nv_shadercache/driver.bin').write_bytes(b'opaque fixture')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'cascade.private-rtx-cache-seed.v1',
        'policy': cache.policy(args.release, args.sdk_recipe), 'inventory': cache.inventory(data)}))
    args.rtx_cache_seed, args.rtx_cache_seed_sha256 = root, digest(manifest)
    return root


def test_default_does_not_select_cache_or_change_app_settings(args):
    args.private_rtx_cache = False
    args.sdk_recipe = None
    assert cache.admit(args) is None
    backend = KitNewtonBackend(args, {}, 'unused.kit')
    assert backend._app_config() == {
        'headless': True, 'disable_viewport_updates': True, 'multi_gpu': False,
        'width': 320, 'height': 240, 'renderer': 'RayTracedLighting', 'physics_gpu': 0}
    assert not args.out.exists()
    args.rtx_cache_seed, args.rtx_cache_seed_sha256 = Path('unread'), 'f'*64
    with pytest.raises(ValueError, match='requires --private'):
        cache.admit(args)


@pytest.mark.parametrize('change', ['neighbor', 'source', 'version'])
def test_exact_sdk_required_before_any_output(args, change):
    if change == 'neighbor':
        args.sdk_recipe = 'isaac62_48b2d952'
    elif change == 'source':
        (args.release / next(iter(cache.SDK_SOURCES))).write_text('changed implementation')
    else:
        (args.release / 'VERSION').write_text('other release')
    with pytest.raises(ValueError):
        cache.admit(args)
    assert not args.out.exists()


@pytest.mark.parametrize('change', ['parent_link', 'sdk_output', 'existing', 'seed_link', 'overlap'])
def test_invalid_paths_refused_without_writes(args, change):
    original = args.out
    if change == 'parent_link':
        link = args.out.parent / 'link'
        link.symlink_to(args.out.parent, target_is_directory=True)
        args.out = link / 'episode'
    elif change == 'sdk_output':
        args.out = args.release / 'episode'
    elif change == 'existing':
        (args.out / 'rtx-cache').mkdir(parents=True)
    elif change == 'seed_link':
        root = seed(args)
        link = args.out.parent / 'link'
        link.symlink_to(root, target_is_directory=True)
        args.rtx_cache_seed = link
    else:
        args.rtx_cache_seed, args.rtx_cache_seed_sha256 = args.out.parent, 'f'*64
    with pytest.raises(ValueError):
        cache.admit(args)
    assert not (original / 'rtx-cache.json').exists()


@pytest.mark.parametrize('change', ['missing', 'extra', 'changed', 'empty_directory', 'symlink',
                                   'traversal', 'duplicates', 'numeric_type', 'source', 'digest'])
def test_seed_manifest_requires_exact_complete_membership(args, change):
    root = seed(args)
    manifest = root / 'manifest.json'
    value = json.loads(manifest.read_text())
    data = root / 'data'
    if change == 'missing':
        (data / 'shadercache/pso.bin').unlink()
    elif change == 'extra':
        (data / 'shadercache/unbound.bin').write_bytes(b'extra')
    elif change == 'changed':
        (data / 'shadercache/pso.bin').write_bytes(b'changed')
    elif change == 'empty_directory':
        (data / 'shadercache/unbound-empty').mkdir()
    elif change == 'symlink':
        (data / 'shadercache/link').symlink_to(args.release, target_is_directory=True)
    elif change == 'traversal':
        value['inventory']['files'][0]['path'] = '../outside'
    elif change == 'duplicates':
        value['inventory']['files'].append(value['inventory']['files'][0])
    elif change == 'numeric_type':
        value['inventory']['files'][0]['size'] = float(value['inventory']['files'][0]['size'])
    elif change == 'source':
        value['policy']['sdk_source_sha256'][next(iter(cache.SDK_SOURCES))] = 'f'*64
    else:
        args.rtx_cache_seed_sha256 = 'f'*64
    if change in ('traversal', 'duplicates', 'numeric_type', 'source'):
        manifest.write_text(json.dumps(value))
        args.rtx_cache_seed_sha256 = digest(manifest)
    with pytest.raises(ValueError):
        cache.admit(args)
    assert not args.out.exists()


def test_seed_is_rechecked_before_creation_and_after_copy(args, monkeypatch):
    root = seed(args)
    admission = cache.admit(args)
    args.out.mkdir()
    original = cache._file
    reads = 0
    def changed(path, destination=None):
        nonlocal reads
        result = original(path, destination)
        if Path(path) == root / 'data/shadercache/pso.bin':
            reads += 1
            if reads == 2:  # admission recheck, then actual materialization
                (root / 'data/shadercache/new.bin').write_bytes(b'late writer')
        return result
    monkeypatch.setattr(cache, '_file', changed)
    with pytest.raises(ValueError, match='inventory differs'):
        cache.prepare(args, admission)
    assert not (args.out / 'rtx-cache.json').exists()
    assert (args.out / 'rtx-cache').exists()  # preserve partial copy, never overwrite/retry it
    with pytest.raises(ValueError, match='already exists'):
        cache.prepare(args, admission)


def test_successful_copy_is_independent_and_receipt_bound(args):
    root = seed(args)
    admission = cache.admit(args)
    assert not args.out.exists()  # admission never writes
    args.out.mkdir()
    record = cache.prepare(args, admission)
    copied = Path(record['root'])
    assert cache.inventory(copied) == cache.inventory(root / 'data')
    assert json.loads((args.out / 'rtx-cache.json').read_text()) == record
    (copied / 'shadercache/pso.bin').write_bytes(b'runtime cache update')
    assert (root / 'data/shadercache/pso.bin').read_bytes() == b'compiled fixture only'
    assert (copied / 'shadercache/empty').is_dir()


def test_source_change_after_admission_is_refused_before_cache_creation(args):
    admission = cache.admit(args)
    args.out.mkdir()
    (args.release / next(iter(cache.SDK_SOURCES))).write_text('changed before bootstrap')
    with pytest.raises(ValueError, match='source mismatch'):
        cache.prepare(args, admission)
    assert not (args.out / 'rtx-cache').exists()


def test_mutation_of_an_already_hashed_member_invalidates_inventory(args, monkeypatch):
    root = seed(args)
    original = cache._file
    def changed(path, destination=None):
        result = original(path, destination)
        if Path(path) == root / 'data/shadercache/pso.bin':
            (root / 'data/nv_shadercache/driver.bin').write_bytes(b'late driver update')
        return result
    monkeypatch.setattr(cache, '_file', changed)
    with pytest.raises(ValueError, match='tree changed'):
        cache.admit(args)
    assert not args.out.exists()


@pytest.mark.parametrize('failure', [None, 'effective_path', 'imported_source',
    'app_origin', 'constructor_origin', 'foundation_origin', 'foundation_code', 'foundation_missing'])
def test_settings_passed_before_constructor_and_checked_before_native_init(args, monkeypatch, failure):
    # Match SDK48's expose_api: a class loaded from the implementation is exported
    # through an alias module that is not a package. No native imports or launch.
    import importlib.util
    source = args.release / cache.SIMULATION_APP_SOURCE
    source.write_text('class SimulationApp:\n'
        '    def __init__(self, config, experience):\n'
        '        constructor(config, experience)\n'
        '    def close(self, **kwargs):\n'
        '        closed()\n')
    monkeypatch.setitem(cache.SDK_SOURCES, cache.SIMULATION_APP_SOURCE, digest(source))
    foundation_file = args.release / cache.GPU_FOUNDATION_SOURCE
    foundation_file.write_text('class ShaderCacheConfig:\n'
        '    def setup_shadercache_locations(self, *args): pass\n')
    monkeypatch.setitem(cache.SDK_SOURCES, cache.GPU_FOUNDATION_SOURCE, digest(foundation_file))
    admission = cache.admit(args)
    args.out.mkdir()
    backend = KitNewtonBackend(args, {'private_rtx_cache': admission}, 'fixture.kit')
    events = []
    settings = {key: str(Path(admission['root']) / relative) for key, relative in cache.LAYOUT.items()}
    def constructor(config, experience):
        events.append('constructor')
        assert (args.out / 'rtx-cache.json').is_file()
        assert config['extra_args'] == [f'--{k}={v}' for k, v in settings.items()]
        assert config['renderer'] == 'RayTracedLighting'
        if failure == 'effective_path':
            settings[next(iter(settings))] = str(args.release)
        if failure == 'imported_source':
            source.write_text('loaded source mutated')
    loaded_source = source
    if failure == 'app_origin':
        loaded_source = args.out / 'same-bytes-wrong-origin.py'
        loaded_source.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location('simulation_app.simulation_app', loaded_source)
    implementation = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, implementation)
    spec.loader.exec_module(implementation)
    implementation.constructor = constructor
    implementation.closed = lambda: events.append('closed')
    if failure == 'constructor_origin':
        # A plausible __file__ is insufficient when executable code is elsewhere.
        def replaced_constructor(self, config, experience):
            constructor(config, experience)
        implementation.SimulationApp.__init__ = replaced_constructor
    public = ModuleType('isaacsim.simulation_app')
    public.SimulationApp = implementation.SimulationApp
    monkeypatch.setitem(sys.modules, 'isaacsim', NS(SimulationApp=public.SimulationApp))
    monkeypatch.setitem(sys.modules, public.__name__, public)
    monkeypatch.delitem(sys.modules, 'isaacsim.simulation_app.simulation_app', raising=False)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('isaacsim.simulation_app.simulation_app')
    carb_settings = NS(get_settings=lambda: settings)
    monkeypatch.setitem(sys.modules, 'carb', NS(settings=carb_settings))
    monkeypatch.setitem(sys.modules, 'carb.settings', carb_settings)
    if failure == 'foundation_origin':
        wrong = args.out / 'same-bytes-wrong-foundation.py'
        wrong.write_bytes(foundation_file.read_bytes())
        foundation_file = wrong
    spec = importlib.util.spec_from_file_location(cache.GPU_FOUNDATION_MODULE, foundation_file)
    foundation = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, foundation)
    spec.loader.exec_module(foundation)
    if failure == 'foundation_code':
        foundation.ShaderCacheConfig.setup_shadercache_locations = lambda *args: None
    if failure == 'foundation_missing':
        monkeypatch.delitem(sys.modules, cache.GPU_FOUNDATION_MODULE)
    monkeypatch.setattr(backend, '_initialize', lambda: events.append('native_init'))
    argv = sys.argv
    try:
        if failure:
            with pytest.raises(ValueError, match='mismatch|unavailable'):
                backend.open()
            assert events == ['constructor']
            assert not (args.out / 'rtx-cache-effective.json').exists()
        else:
            backend.open()
            assert events == ['constructor', 'native_init']
            effective = json.loads((args.out / 'rtx-cache-effective.json').read_text())
            assert effective['effective_settings'] == settings
    finally:
        backend.close()
        assert backend.shutdown(1 if failure else 0)
    assert events[-1] == 'closed' and sys.argv is argv


def test_recipe_binds_policy_without_absolute_paths_or_mutable_cache_bytes(args, recipe_inputs):
    from cascade.sim.mobile_identity import build_model_identity
    admission, native, paths = recipe_inputs
    before = build_model_identity(admission, native, **paths)
    selected = cache.admit(args)
    admission['private_rtx_cache'] = selected
    with pytest.raises(ValueError, match='differs from admission'):
        build_model_identity(admission, native, **paths)
    native['configuration']['private_rtx_cache'] = selected['policy']
    first = build_model_identity(admission, native, **paths)
    assert first != before and str(args.out) not in json.dumps(first)
    selected['root'] = '/different/episode/rtx-cache'
    selected['seed'] = {'manifest_sha256': 'f'*64, 'path': '/other/seed'}
    native['private_rtx_cache_effective'] = {'absolute_paths': '/different/episode'}
    assert build_model_identity(admission, native, **paths) == first
    selected['policy']['schema'] = 'changed-cache-policy'
    assert build_model_identity(admission, native, **paths) != first
