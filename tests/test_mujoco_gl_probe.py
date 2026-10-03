"""Real child processes with a tiny synthetic MuJoCo module; no GL required."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from mujoco_gl_probe import probe_offscreen_gl


def _module(tmp_path, monkeypatch, constructor, *, close="pass"):
    # The disposable child imports this test double, not an installed SDK.
    module = tmp_path / "mujoco.py"
    module.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "class MjModel:\n"
        "    @staticmethod\n"
        "    def from_xml_path(path):\n"
        "        assert Path(path).is_file()\n"
        "        return path\n"
        "class Renderer:\n"
        "    def __init__(self, model, *, height, width):\n"
        "        assert height == width == 16\n"
        f"        {constructor}\n"
        "    def close(self):\n"
        f"        {close}\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    mjcf = tmp_path / "scene.xml"
    mjcf.write_text("synthetic child-protocol fixture; not a physics model")
    return mjcf


def test_success_requires_renderer_close_in_same_interpreter_and_environment(tmp_path, monkeypatch):
    marker = tmp_path / "closed"
    monkeypatch.setenv("MUJOCO_GL", "synthetic-selected-backend")
    mjcf = _module(
        tmp_path, monkeypatch,
        f"assert sys.executable == {sys.executable!r} and os.environ['MUJOCO_GL'] == 'synthetic-selected-backend'",
        close=f"Path({str(marker)!r}).write_text('closed')",
    )
    result = probe_offscreen_gl(mjcf)
    assert result.available and result.returncode == 0
    assert marker.read_text() == "closed"
    assert os.environ["MUJOCO_GL"] == "synthetic-selected-backend"


@pytest.mark.parametrize("failure", ["raise RuntimeError('no GL')", "os._exit(23)"])
def test_failed_constructor_is_unavailable_without_exiting_pytest(tmp_path, monkeypatch, failure):
    result = probe_offscreen_gl(_module(tmp_path, monkeypatch, failure))
    assert not result.available and result.returncode != 0
    assert "exited" in result.reason


def test_failed_close_is_not_available(tmp_path, monkeypatch):
    result = probe_offscreen_gl(_module(tmp_path, monkeypatch, "pass", close="raise RuntimeError('close')"))
    assert not result.available and result.returncode == 1


def test_inline_xml_probe_keeps_its_model_and_render_dimensions(tmp_path, monkeypatch):
    mjcf = tmp_path / "robot.xml"
    mjcf.touch()  # This test's existing asset gate is independent of its GL model.
    (tmp_path / "mujoco.py").write_text(
        "class MjModel:\n"
        "    @staticmethod\n"
        "    def from_xml_string(xml):\n"
        "        assert xml == '<mujoco/>'\n"
        "        return 'inline'\n"
        "class Renderer:\n"
        "    def __init__(self, model, *, height, width):\n"
        "        assert model == 'inline' and height == width == 64\n"
        "    def close(self): pass\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert probe_offscreen_gl(mjcf, model_xml="<mujoco/>", height=64, width=64).available


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGABRT return-code contract")
def test_native_abort_is_contained_in_child(tmp_path, monkeypatch):
    result = probe_offscreen_gl(_module(tmp_path, monkeypatch, "os.abort()"))
    assert not result.available and result.returncode == -signal.SIGABRT


def test_timeout_kills_and_reaps_probe_child(tmp_path, monkeypatch):
    mjcf = _module(tmp_path, monkeypatch, "time.sleep(60)")
    children = []
    original = subprocess.Popen

    def owned_child(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", owned_child)
    result = probe_offscreen_gl(mjcf, timeout_s=0.2)
    assert not result.available and result.returncode is None
    assert "exceeded" in result.reason
    assert len(children) == 1 and children[0].returncode is not None


def test_missing_assets_does_not_spawn(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("missing asset must not launch a probe")

    monkeypatch.setattr(subprocess, "run", unexpected)
    assert not probe_offscreen_gl(tmp_path / "missing.xml").available


def test_spawn_failure_is_unavailable(tmp_path, monkeypatch):
    mjcf = tmp_path / "scene.xml"
    mjcf.touch()

    def unavailable(*args, **kwargs):
        raise OSError("interpreter unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    result = probe_offscreen_gl(mjcf)
    assert not result.available and result.returncode is None
    assert "could not start" in result.reason
