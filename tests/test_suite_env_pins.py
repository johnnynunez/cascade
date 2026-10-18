"""The suite certifies the shipped default with the observed-finger gate OFF,
in every shell (#370 review, Minor 1).

A Spark shell that sourced the installer's env.sh exports
CASCADE_OBSERVED_FINGER_GATE=1. The autouse `_no_ambient_observed_finger_gate`
fixture scrubs it per test, but test modules' import-time code and module /
session-scoped fixtures run before any function fixture and saw it. conftest's
`pytest_configure` now scrubs it for the whole session (like the GraspGen-X
port pin) and `pytest_unconfigure` gives the ambient value back.

Each case runs a child pytest that loads tests/conftest.py as a plugin over a
probe module, with the gate exported or absent; a last unconfigure hook
records what the shell gets back.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from conftest import REPO

GATE = "CASCADE_OBSERVED_FINGER_GATE"

PROBE = '''
import os

import pytest

GATE = "CASCADE_OBSERVED_FINGER_GATE"
AT_IMPORT = os.environ.get(GATE)


@pytest.fixture(scope="session")
def session_seen():
    return os.environ.get(GATE)


@pytest.fixture(scope="module")
def module_seen():
    return os.environ.get(GATE)


def test_no_scope_sees_the_ambient_gate(session_seen, module_seen):
    seen = {"import": AT_IMPORT, "session": session_seen, "module": module_seen, "test": os.environ.get(GATE)}
    assert seen == dict.fromkeys(seen), seen


def test_a_gate_on_test_still_turns_it_on(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    assert os.environ[GATE] == "1"
'''

AFTER = '''
import os
import pathlib

import pytest


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    pathlib.Path(os.environ["GATE_PROBE_OUT"]).write_text(repr(os.environ.get("CASCADE_OBSERVED_FINGER_GATE")))
'''


@pytest.mark.parametrize("ambient", ["1", None], ids=["exported", "absent"])
def test_the_session_never_sees_an_ambient_observed_finger_gate_and_gives_it_back(tmp_path, ambient):
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / "test_gate_probe.py").write_text(PROBE)
    (tmp_path / "plugin").mkdir()
    (tmp_path / "plugin" / "gate_after.py").write_text(AFTER)
    out = tmp_path / "after.txt"
    env = {k: v for k, v in os.environ.items() if k != GATE}
    if ambient is not None:
        env[GATE] = ambient
    env["GATE_PROBE_OUT"] = str(out)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(tmp_path / "plugin"), str(REPO / "tests"), str(REPO / "src"), os.environ.get("PYTHONPATH"))
        if p)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-p", "conftest",
         "-p", "gate_after", f"--basetemp={tmp_path / 'child'}", str(tmp_path / "probe")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0 and "2 passed" in result.stdout, result.stdout + result.stderr
    assert out.read_text() == repr(ambient)
