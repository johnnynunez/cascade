"""SDK admission only: synthetic module files, no Kit/Warp imports or solves."""
import hashlib
import sys
from types import SimpleNamespace as NS

import pytest

from cascade.control import newton_bam
from cascade.sim import microduck_sdk as sdk


@pytest.fixture
def imported_sdk(tmp_path, monkeypatch):
    pins = {}
    paths = {}
    for name in sdk.INTERNAL_SOURCE_SHA256:
        path = tmp_path / (name + '.py')
        path.write_text('# synthetic SDK admission fixture: ' + name)
        paths[name] = path
        pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setitem(sys.modules, name, NS(__file__=str(path)))
    monkeypatch.setattr(sdk, 'INTERNAL_SOURCE_SHA256', pins)
    monkeypatch.setitem(sys.modules, 'newton', NS(__version__='1.6.1rc1'))
    monkeypatch.setitem(sys.modules, 'warp', NS())
    return paths


def test_rc_requires_explicit_recipe_even_with_exact_sources(imported_sdk):
    with pytest.raises(RuntimeError, match='stable Newton'):
        newton_bam._runtime()
    wp, newton = newton_bam._runtime(sdk_recipe=sdk.INTERNAL_RECIPE)
    assert newton.__version__ == '1.6.1rc1' and wp is sys.modules['warp']


@pytest.mark.parametrize('version', ['1.6.0rc1', '1.6.1rc2', '1.7.0rc1', '1.6.1', 'unknown'])
def test_internal_recipe_never_accepts_a_version_neighbor(imported_sdk, version):
    sys.modules['newton'].__version__ = version
    with pytest.raises(RuntimeError, match='exact Newton'):
        newton_bam._runtime(sdk_recipe=sdk.INTERNAL_RECIPE)


def test_unknown_recipe_is_not_an_rc_escape_hatch(imported_sdk):
    with pytest.raises(ValueError, match='unknown MicroDuck SDK recipe'):
        newton_bam._runtime(sdk_recipe='auto')


@pytest.mark.parametrize('name', tuple(sdk.INTERNAL_SOURCE_SHA256))
def test_every_consumed_source_is_rechecked_before_rc_admission(imported_sdk, name):
    newton_bam._runtime(sdk_recipe=sdk.INTERNAL_RECIPE)
    imported_sdk[name].write_text('# changed after initial admission')
    with pytest.raises(RuntimeError, match='SDK source mismatch'):
        newton_bam._runtime(sdk_recipe=sdk.INTERNAL_RECIPE)


@pytest.mark.parametrize('version', ['1.6.0', '1.6.1', '1.7.0+local'])
def test_stable_default_keeps_existing_admission_without_kit(monkeypatch, version):
    monkeypatch.setitem(sys.modules, 'newton', NS(__version__=version))
    monkeypatch.setitem(sys.modules, 'warp', NS())
    assert newton_bam._runtime()[1].__version__ == version


def test_support_recipe_reports_actual_files_and_refuses_mutation(imported_sdk):
    from cascade.sim.microduck_contact_support import extraction_provenance
    result = extraction_provenance(sdk_recipe=sdk.INTERNAL_RECIPE)
    assert result['source_admitted'] is True
    assert result['source_sha256'] == sdk.INTERNAL_SOURCE_SHA256
    assert result['sdk_recipe'] == sdk.INTERNAL_RECIPE
    next(iter(imported_sdk.values())).write_text('# different')
    with pytest.raises(RuntimeError, match='SDK source mismatch'):
        extraction_provenance(sdk_recipe=sdk.INTERNAL_RECIPE)


def test_release_admission_is_offline_and_checks_build_and_all_files(tmp_path, monkeypatch):
    pins = {}
    for name, relative in sdk.INTERNAL_SOURCE_PATHS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic release: ' + name)
        pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    version = tmp_path / 'VERSION'
    version.write_text('synthetic pinned release')
    monkeypatch.setattr(sdk, 'INTERNAL_SOURCE_SHA256', pins)
    monkeypatch.setattr(sdk, 'INTERNAL_VERSION_SHA256', hashlib.sha256(version.read_bytes()).hexdigest())
    receipt = sdk.admit_release(tmp_path, sdk.INTERNAL_RECIPE)
    assert receipt['source_sha256'] == pins
    assert receipt['physical_acceptance'] is False
    version.write_text('other build')
    with pytest.raises(ValueError, match='release VERSION'):
        sdk.admit_release(tmp_path, sdk.INTERNAL_RECIPE)


def test_internal_outputs_are_explicit_and_mutations_refused():
    cfg = NS(solver_outputs=NS(contact_forces=False, link_incoming_joint_force=True))
    sdk.configure_outputs(cfg, sdk.INTERNAL_RECIPE)
    sdk.check_outputs(cfg, sdk.INTERNAL_RECIPE)
    assert cfg.solver_outputs.contact_forces is True
    assert cfg.solver_outputs.link_incoming_joint_force is False
    cfg.solver_outputs.contact_forces = False
    with pytest.raises(RuntimeError, match='solver outputs'):
        sdk.check_outputs(cfg, sdk.INTERNAL_RECIPE)
    sdk.configure_outputs(NS(), None)  # legacy SDK has no solver_outputs member
