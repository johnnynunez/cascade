"""Explicit SDK source admission, separate from policy and physical acceptance.

The default remains the existing stable-Newton path. The internal RC is accepted
only by name, exact version and consumed source hashes; no version-range fallback.
Imports here are portable; SDK imports occur only in verify_runtime_recipe().
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import MappingProxyType

INTERNAL_RECIPE = 'isaac62_48b2d951'
INTERNAL_NEWTON_VERSION = '1.6.1rc1'
INTERNAL_VERSION_SHA256 = '59a6d16d8f3f247b73fa68a837867dafc0f568e491ae593f0ab1c2cb2fa62786'
INTERNAL_SOURCE_SHA256 = MappingProxyType({
    'mujoco_warp._src.support': '1a08ef15f149d1cf3c9ec5df6c688669fe86aa237ee2512a7463caf1b74e3d54',
    'newton._src.solvers.mujoco.kernels': 'bd770d39d20c208e980477e0186799f531364b03f16d76968ee69bacdc490a3d',
    'newton._src.solvers.mujoco.solver_mujoco': 'a2f2466ad9d35e32f6e87f12bbaa0e08d18295f4e95b1c059c805d2ff4091178',
    'isaacsim.physics.newton.impl.newton_stage': 'a9d4e95305b23e6cea8e4e8e5fb63dc8b6c5c5917ad5c3ba48e2ba29df84d958',
    'isaacsim.physics.newton.impl.newton_config': 'b1212639ebf22448a1da36b272b36da02eb340dcc1ad2312da075a66dadf1de0',
    'isaacsim.physics.newton.impl.solver_config': 'e6c17f958cbd2534ac71743e62f56b79c4e8d86147ee676a6e80ee0d7ea9fe03',
})
# CPU contract tests against the exact pre-release bundled with the admitted Isaac
# Sim release: same Newton/MJWarp module bytes, no Kit. This is a named test
# recipe, never an SDK admission; the native launchers do not accept it.
CPU_CONTRACT_RECIPE = 'newton161rc1_cpu'
CPU_CONTRACT_SOURCES = ('mujoco_warp._src.support', 'newton._src.solvers.mujoco.kernels',
                        'newton._src.solvers.mujoco.solver_mujoco')
INTERNAL_SOURCE_PATHS = MappingProxyType({
    name: ('exts/isaacsim.physics.newton/' if name.startswith('isaacsim.')
           else 'exts/isaacsim.pip.newton/pip_prebundle/') + name.replace('.', '/') + '.py'
    for name in INTERNAL_SOURCE_SHA256
})


def require_recipe(recipe):
    if recipe != INTERNAL_RECIPE:
        raise ValueError(f'unknown MicroDuck SDK recipe: {recipe!r}')


def admit_release(release, recipe):
    """Offline preflight: files only, before Kit, model construction or writes."""
    require_recipe(recipe)
    release = Path(release)
    if hashlib.sha256((release / 'VERSION').read_bytes()).hexdigest() != INTERNAL_VERSION_SHA256:
        raise ValueError('MicroDuck SDK release VERSION differs from the explicit recipe')
    actual = {name: hashlib.sha256((release / path).read_bytes()).hexdigest()
              for name, path in INTERNAL_SOURCE_PATHS.items()}
    if actual != INTERNAL_SOURCE_SHA256:
        raise ValueError('MicroDuck SDK source mismatch in release')
    return dict(recipe=recipe, newton_version=INTERNAL_NEWTON_VERSION,
                release_version_sha256=INTERNAL_VERSION_SHA256, source_sha256=actual,
                physical_acceptance=False)


def verify_runtime_recipe(recipe, *, newton_version):
    """Recheck the files actually imported, not just the declared release tree."""
    require_recipe(recipe)
    if newton_version != INTERNAL_NEWTON_VERSION:
        raise RuntimeError(f'MicroDuck SDK recipe requires exact Newton {INTERNAL_NEWTON_VERSION}')
    actual = {}
    for name, expected in INTERNAL_SOURCE_SHA256.items():
        path = Path(importlib.import_module(name).__file__)
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[name] != expected:
            raise RuntimeError(f'MicroDuck SDK source mismatch: {name}')
    return actual


def verify_cpu_contract_recipe(recipe, *, newton_version):
    """Exact pre-release and the portable solver sources of the SDK recipe, outside Kit."""
    if recipe != CPU_CONTRACT_RECIPE:
        raise ValueError(f'unknown MicroDuck CPU contract recipe: {recipe!r}')
    if newton_version != INTERNAL_NEWTON_VERSION:
        raise RuntimeError(f'MicroDuck CPU contract recipe requires exact Newton {INTERNAL_NEWTON_VERSION}')
    actual = {}
    for name in CPU_CONTRACT_SOURCES:
        path = Path(importlib.import_module(name).__file__)
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[name] != INTERNAL_SOURCE_SHA256[name]:
            raise RuntimeError(f'MicroDuck CPU contract source mismatch: {name}')
    return actual


def configure_outputs(cfg, recipe):
    if recipe is not None:
        require_recipe(recipe)
        cfg.solver_outputs.contact_forces = True
        cfg.solver_outputs.link_incoming_joint_force = False


def check_outputs(cfg, recipe):
    if recipe is not None:
        require_recipe(recipe)
        if (cfg.solver_outputs.contact_forces is not True
                or cfg.solver_outputs.link_incoming_joint_force is not False):
            raise RuntimeError('MicroDuck SDK solver outputs changed')
