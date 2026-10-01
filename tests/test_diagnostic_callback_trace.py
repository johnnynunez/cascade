"""Diagnostic recording must not evaluate or replace motion safety callbacks."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_observed_finger_runtime import fake_gate, runtime

ROOT = Path(__file__).resolve().parents[1]


def diagnostic(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark" / "diagnostics" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kitchen = diagnostic("kitchen_acceptance")
camera = diagnostic("nvblox_camera_recovery")


@pytest.fixture
def scene_callbacks(monkeypatch):
    """Capture the actual closures produced by SkillRuntime._scene_motion."""
    checks = fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    captured = []

    class Captured(RuntimeError):
        pass

    def stop_before_motion(*args, **kwargs):
        captured.append(kwargs)
        raise Captured("callbacks captured before the first motion")

    rt.arm.move_joints = stop_before_motion
    with pytest.raises(Captured):
        rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    assert len(captured) == 1
    assert not [c for c in calls if c[0] == "joints"]
    # Runtime legitimately checks feedback before its initial open command.
    assert [check[0] for check in checks] == ["feedback"]
    result = captured[0]
    assert "_scene_motion.<locals>.preflight" in result["preflight"].__qualname__
    assert "_scene_motion.<locals>.before_stream" in result["before_stream"].__qualname__
    return result, checks, len(checks)


def traced(tmp_path, mode, original):
    raw = SimpleNamespace(reset_props=lambda: None, send_joint_target=lambda *a, **kw: None,
                          set_gripper=lambda *a, **kw: None)
    arm = SimpleNamespace(raw=raw, move_joints=original, set_gripper=lambda *a, **kw: None,
        harness=SimpleNamespace(_grasp_exempt=None, allow_grasp_descent=lambda *a, **kw: None,
                                clear_grasp_exemption=lambda: None))
    rt = SimpleNamespace(arm=arm, held_object=None, memory=SimpleNamespace(add=lambda *a, **kw: None))
    paths = []
    if mode in ("kitchen", "both"):
        path = tmp_path / "skills.jsonl"
        kitchen.install_command_trace(rt, path)
        paths.append(path)
    if mode in ("camera", "both"):
        path = tmp_path / "actuators.jsonl"
        camera.ActuatorTrace(rt, path)
        paths.append(path)
    return rt.arm.move_joints, paths


@pytest.mark.parametrize("mode", ["kitchen", "camera", "both"])
def test_real_scene_callbacks_are_opaque_and_original_arguments_survive(tmp_path, scene_callbacks, mode):
    kwargs, checks, prior_checks = scene_callbacks
    target = np.array([.2, .0, .1])
    calls = []

    def original(*args, **received):
        calls.append((args, received))
        assert args[0] is target
        assert set(received) == set(kwargs)
        assert all(received[k] is value for k, value in kwargs.items())
        # Keep the result JSON-compatible, as the normal move API is boolean.
        return True

    move, paths = traced(tmp_path, mode, original)
    assert move(target, **kwargs) is True
    assert len(calls) == 1 and len(checks) == prior_checks
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["begin", "end"]
        saved = rows[0]["kwargs"]
        assert "_halt_generation" not in saved
        for name in ("preflight", "before_stream", "feedback_guard"):
            assert saved[name] == {
                "diagnostic_kind": "opaque_callback", "module": kwargs[name].__module__,
                "qualname": kwargs[name].__qualname__}
        assert "0x" not in path.read_text()


@pytest.mark.parametrize("mode", ["kitchen", "camera", "both"])
def test_callback_typeerror_is_propagated_once_without_retry(tmp_path, mode):
    counts = {"callback": 0, "original": 0}
    failure = TypeError("callback failed after original received it")

    def callback():
        counts["callback"] += 1
        raise failure

    def original(*args, **kwargs):
        counts["original"] += 1
        assert kwargs["before_stream"] is callback
        kwargs["before_stream"]()

    move, paths = traced(tmp_path, mode, original)
    with pytest.raises(TypeError) as caught:
        move(np.zeros(3), duration_s=1., before_stream=callback)
    assert caught.value is failure
    assert counts == {"callback": 1, "original": 1}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["begin", "exception"]


@pytest.mark.parametrize("mode", ["kitchen", "camera", "both"])
def test_callable_object_repr_and_body_are_never_evaluated(tmp_path, mode):
    class Callback:
        def __repr__(self):
            raise AssertionError("closure/object repr must not be logged")
        def __call__(self, *args):
            raise AssertionError("recording must not invoke callbacks")

    callback = Callback()
    calls = []
    def original(*args, **kwargs):
        assert kwargs["feedback_guard"] is callback
        calls.append(1)
        return True
    move, paths = traced(tmp_path, mode, original)
    assert move(np.zeros(3), feedback_guard=callback)
    assert calls == [1]
    for path in paths:
        saved = json.loads(path.read_text().splitlines()[0])["kwargs"]["feedback_guard"]
        assert saved == {"diagnostic_kind": "opaque_callback", "type_module": Callback.__module__,
                         "type_qualname": Callback.__qualname__}
