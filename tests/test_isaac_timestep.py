"""Exercise the real pre-Kit parser and environment boundary without a GPU."""
import ast
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def parse(monkeypatch, env_value=None, argv=()):
    monkeypatch.setattr(sys, 'argv', ['isaac_bridge.py', *argv])
    if env_value is None:
        monkeypatch.delenv('CASCADE_ISAAC_DT', raising=False)
    else:
        monkeypatch.setenv('CASCADE_ISAAC_DT', env_value)
    tree=ast.parse((ROOT/'scripts/isaac_bridge.py').read_text())
    stop=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.If)
              and 'args.dt' in ast.unparse(n.test))
    namespace={'__file__':str(ROOT/'scripts/isaac_bridge.py'),'__name__':'parser_test'}
    exec(compile(ast.Module(body=tree.body[:stop+1],type_ignores=[]),'bridge_parser','exec'),namespace)
    return namespace['args']


def test_calibrated_timestep_crosses_adapter_and_parser(monkeypatch):
    spec=importlib.util.spec_from_file_location('isaac_adapter',ROOT/'scripts/isaac_launch.py')
    adapter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    env=adapter.clean_environment({'CASCADE_ISAAC_DT':'0.008333333333333333'},source=None)
    assert parse(monkeypatch,env['CASCADE_ISAAC_DT']).dt == 1/120
    assert parse(monkeypatch).dt == 1/60
    assert parse(monkeypatch,env['CASCADE_ISAAC_DT'],['--dt','0.004']).dt == .004


@pytest.mark.parametrize('value',['0','-0.1','nan','inf','2'])
def test_invalid_timestep_fails_before_kit_import(monkeypatch,value):
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch,value)
    assert exc.value.code == 2
