"""A persistent STOP must block every automatic container restart before setup."""

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('entrypoint_stop', HERE / 'container/entrypoint.py')
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


@pytest.mark.parametrize('kind', ['file', 'dangling_symlink'])
def test_retained_stop_prevents_directory_setup_ui_and_native_runtime(tmp_path, monkeypatch, kind):
    data = tmp_path / 'data'
    marker = data / 'acceptance/STOP'
    marker.parent.mkdir(parents=True)
    if kind == 'file':
        marker.touch()
    else:
        marker.symlink_to('missing-retained-stop-target')
    monkeypatch.setenv('PAAI_DATA_DIR', str(data))
    monkeypatch.setattr(entrypoint.subprocess, 'run', lambda *a, **kw: pytest.fail('Runtime setup after STOP'))
    monkeypatch.setattr(entrypoint.os, 'execv', lambda *a, **kw: pytest.fail('Runtime launch after STOP'))
    old_umask = os.umask(0o077)
    try:
        with pytest.raises(InterruptedError, match='retained STOP'):
            entrypoint.main()
    finally:
        os.umask(old_umask)
    assert list(data.iterdir()) == [marker.parent]
    assert os.path.lexists(marker)


def test_real_container_entry_script_reports_stop_without_workspace(tmp_path):
    marker = tmp_path / 'acceptance/STOP'
    marker.parent.mkdir()
    marker.touch()
    result = subprocess.run(['/usr/bin/python3', str(HERE / 'container/entrypoint.py')],
                            env={'PAAI_DATA_DIR': str(tmp_path)}, capture_output=True,
                            text=True, timeout=5)
    assert result.returncode == 0 and '"status": "STOPPED"' in result.stdout
    assert not result.stderr and marker.exists()
