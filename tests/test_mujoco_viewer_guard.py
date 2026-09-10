"""The MuJoCo window must never take the MCP server down.

Measured 2026-09-10: with the display asleep (lid shut during the launcher's
proof turn) CGGetActiveDisplayList returns 0 displays, GLFW has no monitor,
and `mujoco.viewer.launch_passive` segfaults in `_glfwGetVideoModeCocoa` --
inside a tool call, killing the whole server ("MCP error -32000: Connection
closed" for the visitor). The engine now asks the OS first, skips with a
reason, and retries on later motion so the window appears once the screen
is awake.
"""
from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from cascade.control import mujoco_arm as ma  # noqa: E402

_XML = "<mujoco><worldbody><body name='b' pos='0 0 .1'><joint name='j' type='hinge'/><geom type='box' size='.05 .05 .05'/></body></worldbody></mujoco>"


@pytest.fixture
def engine(tmp_path):
    """An _MjcEngine on a throwaway scene; ALWAYS released (the shared-world
    registry must be empty for the next test -- test_mujoco_world asserts it)."""
    p = tmp_path / "s.xml"
    p.write_text(_XML)
    eng = ma._MjcEngine(str(p))
    yield eng
    eng.close()


def test_viewer_is_skipped_not_launched_when_no_display(engine, monkeypatch, capsys):
    eng = engine
    launched = []
    monkeypatch.setattr(ma, "_display_unavailable_reason", lambda: "no ACTIVE display (test)")
    import mujoco.viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda m, d: launched.append(1))
    eng.realize(view=True)
    assert launched == [] and eng._viewer is None
    assert "viewer skipped: no ACTIVE display (test)" in capsys.readouterr().err
    # still wanted: the display may wake up later
    assert eng._view_wanted is True


def test_viewer_retries_on_motion_once_display_is_back(engine, monkeypatch):
    eng = engine
    state: dict[str, str | None] = {"reason": "asleep"}
    monkeypatch.setattr(ma, "_display_unavailable_reason", lambda: state["reason"])

    class _Viewer:
        def sync(self):
            pass

        def close(self):
            pass

    import mujoco.viewer

    opened = []
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda m, d: opened.append(1) or _Viewer())
    eng.realize(view=True)
    eng.step(1)
    assert opened == [] and eng._viewer is None  # display still asleep: no attempt
    state["reason"] = None
    eng._view_retry_at = 0.0  # the 5 s back-off has elapsed
    eng.step(1)
    assert opened == [1] and eng._viewer is not None  # opened on the first motion after wake
    eng.step(1)
    assert opened == [1]  # not re-opened while a viewer is live


def test_viewer_real_launch_failure_stops_retrying(engine, monkeypatch):
    eng = engine
    monkeypatch.setattr(ma, "_display_unavailable_reason", lambda: None)
    import mujoco.viewer

    calls = []

    def _boom(m, d):
        calls.append(1)
        raise RuntimeError("launch_passive requires mjpython")

    monkeypatch.setattr(mujoco.viewer, "launch_passive", _boom)
    eng.realize(view=True)
    assert eng._viewer is None and eng._view_wanted is False
    eng._view_retry_at = 0.0
    eng.step(1)
    assert calls == [1]  # a real failure is final; no retry storm on every step


def test_display_probe_answers_on_this_platform():
    """Whatever the machine state, the probe returns None or a reason string
    -- never raises (it runs inside the arm's connect and on every step)."""
    r = ma._display_unavailable_reason()
    assert r is None or (isinstance(r, str) and r)
    import platform

    if platform.system() == "Darwin":
        # cross-check against CoreGraphics directly
        import ctypes
        import ctypes.util

        cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
        n = ctypes.c_uint32(0)
        arr = (ctypes.c_uint32 * 16)()
        cg.CGGetActiveDisplayList(16, arr, ctypes.byref(n))
        assert (r is None) == (n.value > 0)
